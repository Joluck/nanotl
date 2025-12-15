import torch
import tilelang
import tilelang.language as T
from einops import rearrange
from tilelang.profiler import do_bench
@tilelang.jit
def linear_model(x, w, b):
    return x@w + b

def linear_model_recurrent(q, k, v, w):
    B, T, H, D = q.shape
    output_dtype = q.dtype  # 保存输出精度
    
    # 累积计算使用 float32 提高精度
    q = q.float()
    k = k.float()
    v = v.float()
    
    o = torch.empty(B, H, T, D, device="cuda", dtype=torch.float32)
    state = torch.zeros(B, H, D, D, device="cuda", dtype=torch.float32)
    k = k.permute(0,2,3,1)
    v = v.transpose(1, 2)
    q = q.transpose(1, 2)
    for t in range(T):
        local_state = k[:,:,:,t:t+1]@v[:,:,t:t+1,:]
        state += local_state
        o[:,:,t:t+1,:] = q[:,:,t:t+1,:]@state
    
    # 转回目标精度
    return o.transpose(1, 2).to(output_dtype)



def outer(q, k, v, num_chunks):
    outer_state = k@v  # [b,h,n,d,d]
    outer_s = torch.zeros_like(outer_state)
    for n in range(num_chunks-1): #cumsum for outer_state
        outer_s[:,:,n+1] = outer_s[:,:,n] + outer_state[:,:,n]
    outer_o = q@outer_s  # [b,h,n,c,d]
    return outer_o

@tilelang.jit(out_idx=[3])
def outer_kernel(B, S, H, DK, DV, dtype='float16', accum_dtype='float32'):

    accum_dtype = 'float'

    chunk_size = 64
    BK = BV = 64  # Set to 128 can be faster, but has some numerical differences with FLA
    assert S % chunk_size == 0 and DK % BK == 0 and DV % BV == 0
    NK = tilelang.cdiv(DK, BK)
    NV = tilelang.cdiv(DV, BV)
    NT = tilelang.cdiv(S, chunk_size)

    @T.prim_func
    def kernel(
        q: T.Tensor([B, S, H, DK], dtype),
        k: T.Tensor([B, S, H, DK], dtype),
        v: T.Tensor([B, S, H, DV], dtype),
        O: T.Tensor([B, S, H, DV], accum_dtype),
    ):
        with T.Kernel(NV, NK, B * H) as (i_v, i_k, i_bh):

            i_b = i_bh // H
            i_h = i_bh % H
            q_shared = T.alloc_shared([chunk_size, BK], dtype)
            k_shared = T.alloc_shared([chunk_size, BK], dtype)
            v_shared = T.alloc_shared([chunk_size, BV], dtype)
            h = T.alloc_fragment([BK, BV], accum_dtype)
            h_shared = T.alloc_shared([BK, BV], dtype)
            o = T.alloc_fragment([chunk_size, BV], accum_dtype)
            o_shared = T.alloc_shared([chunk_size, BV], accum_dtype)
            for i in T.Pipelined(0, NT):
                T.copy(q[i_b, i*chunk_size:(i+1)*chunk_size, i_h, :], q_shared)
                T.copy(k[i_b, i*chunk_size:(i+1)*chunk_size, i_h, :], k_shared)
                T.copy(v[i_b, i*chunk_size:(i+1)*chunk_size, i_h, :], v_shared)
                T.copy(h, h_shared)
                T.gemm(k_shared, v_shared, h, transpose_A=True)
                T.gemm(q_shared, h_shared, o)
                T.copy(o, o_shared)
                T.copy(o_shared, O[i_b, i*chunk_size:(i+1)*chunk_size, i_h, :])
    return kernel


def inner(q, k, v, num_chunks, chunk_size, outer_o):
    inter_state = torch.zeros(B, H, num_chunks, D, D, device="cuda", dtype=torch.float32)
        
    # 计算 chunk 内的累积
    inter_o = torch.zeros_like(outer_o)
    for t in range(chunk_size):
        local_state = k[:,:,:,:,t:t+1]@v[:,:,:,t:t+1,:]
        inter_state += local_state
        inter_o[:,:,:,t:t+1,:] = q[:,:,:,t:t+1,:]@inter_state
    return inter_o


def linear_model_chunk(q, k, v, w, chunk_size=64):
    B, S, H, D = q.shape
    output_dtype = q.dtype  # 保存输出精度
    num_chunks = S//chunk_size

    q, k, v, w = map(lambda x: rearrange(x, 'b (n c) h d -> b h n c d', c=chunk_size).float(), (q, k, v, w))
    k = k.transpose(-1, -2)  

    outer_o = outer(q, k, v, num_chunks)
    
    inter_o = inner(q, k, v, num_chunks, chunk_size, outer_o)
    
    # 合并并转回目标精度
    o = rearrange(inter_o + outer_o, 'b h n c d -> b (n c) h d').to(output_dtype)
    return o


if __name__ == "__main__":
    B, S, H, D = 2, 4096, 64, 64
    dtype = torch.float16
    q = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
    k = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
    v = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
    w = torch.randn(B, S, D, D, device="cuda", dtype=dtype)

    print("=" * 60)
    print(f"测试配置: B={B}, S={S}, H={H}, D={D}")
    print("=" * 60)

    # 正确性测试
    print("\n正在验证正确性...")
    o1 = linear_model_recurrent(q.clone(), k.clone(), v.clone(), w.clone())
    o2 = linear_model_chunk(q.clone(), k.clone(), v.clone(), w.clone(), chunk_size=64)
    torch.testing.assert_close(o1, o2, rtol=1e-2, atol=1e-2)
    print("✓ 正确性验证通过！")

    # 性能测试
    print("\n正在测速...")
    recurrent_ms = do_bench(lambda: linear_model_recurrent(q.clone(), k.clone(), v.clone(), w.clone()))
    chunk_ms = do_bench(lambda: linear_model_chunk(q.clone(), k.clone(), v.clone(), w.clone(), chunk_size=128))

    print(f"\n{'方法':<20} {'耗时 (ms)':<15} {'加速比':<10}")
    print("-" * 60)
    print(f"{'linear_model_recurrent':<20} {recurrent_ms:>10.3f} ms   {1.0:>6.2f}x")
    print(f"{'linear_model_chunk':<20} {chunk_ms:>10.3f} ms   {recurrent_ms/chunk_ms:>6.2f}x")
    print("=" * 60)
    
    if chunk_ms < recurrent_ms:
        print(f"\n🎉 chunk 版本快 {recurrent_ms/chunk_ms:.2f}x!")
    else:
        print(f"\n⚠️  recurrent 版本快 {chunk_ms/recurrent_ms:.2f}x")