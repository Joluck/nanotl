import tilelang
import tilelang.language as T
import torch
from tilelang.profiler import do_bench
from einops import rearrange

def outer_naive(q, k, v, num_chunks, chunk_size=64):
    q, k, v = map(lambda x: rearrange(x, 'b (n c) h d -> b h n c d', c=chunk_size).float(), (q, k, v))
    k = k.transpose(-1, -2)  
    outer_state = k@v  # [b,h,n,d,d]
    outer_s = torch.zeros_like(outer_state)
    for n in range(num_chunks-1): #cumsum for outer_state
        outer_s[:,:,n+1] = outer_s[:,:,n] + outer_state[:,:,n]
    outer_o = q@outer_s  # [b,h,n,c,d]
    return rearrange(outer_o, 'b h n c d -> b (n c) h d')
@tilelang.jit(out_idx=[4])
def outer_kernel(B, S, H, DK, DV, dtype='float16', accum_dtype='float32'):

    accum_dtype = 'float32'

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
        final_state: T.Tensor([B, H, DK, DV], accum_dtype),
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
            # o_shared = T.alloc_shared([chunk_size, BV], accum_dtype)

            T.clear(h)
            T.clear(o)

            for i in T.Pipelined(0, NT):
                T.copy(q[i_b, i*chunk_size:(i+1)*chunk_size, i_h, i_k * BK:(i_k + 1) * BK], q_shared)
                T.copy(k[i_b, i*chunk_size:(i+1)*chunk_size, i_h, i_k * BK:(i_k + 1) * BK], k_shared)
                T.copy(v[i_b, i*chunk_size:(i+1)*chunk_size, i_h, i_v * BV:(i_v + 1) * BV], v_shared)
                
                T.copy(h, h_shared)
                T.gemm(k_shared, v_shared, h, transpose_A=True)
                T.gemm(q_shared, h_shared, o, clear_accum=True)
                # T.copy(o, o_shared)
                # T.atomic_add(
                #     O[i_b, i * chunk_size:(i + 1) * chunk_size, i_h, i_v * BV:(i_v + 1) * BV],
                #     o_shared)
                T.copy(o, O[i_b, i*chunk_size:(i+1)*chunk_size, i_h, i_v * BV:(i_v + 1) * BV])
            T.copy(h, final_state[i_b, i_h, i_k * BK:(i_k + 1) * BK, i_v * BV:(i_v + 1) * BV])
    return kernel



def outer_cumsum(q, k, v, num_chunks, chunk_size=64):
    q, k, v = map(lambda x: rearrange(x, 'b (n c) h d -> b h n c d', c=chunk_size).float(), (q, k, v))
    k = k.transpose(-1, -2)  
    kv = k@v  # [b,h,n,d,d]
    kv = kv.cumsum(2)
    h = kv[:, :, -1, :, :]
    kv = torch.cat([torch.zeros_like(kv[:, :, :1]), kv[:, :, :-1]], dim=2)
    outer_o = q@kv  # [b,h,n,c,d]
    return rearrange(outer_o, 'b h n c d -> b (n c) h d'), h

if __name__ == "__main__":
    B, S, H, D = 8, 512, 64, 64
    dtype = torch.float16
    q = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
    k = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
    v = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
    w = torch.randn(B, S, D, D, device="cuda", dtype=dtype)
    o = torch.zeros((B, S, H, D), device='cuda', dtype=torch.float32)
    print("=" * 60)
    print(f"测试配置: B={B}, S={S}, H={H}, D={D}")
    print("=" * 60)
    chunk_size = 64
    # 正确性测试
    print("\n正在验证正确性...")
    o1 = outer_naive(q.clone(), k.clone(), v.clone(), S//chunk_size, chunk_size)
    o2, h = outer_cumsum(q.clone(), k.clone(), v.clone(),  S//chunk_size, chunk_size)

    outer_tl = outer_kernel(B, S, H, D, D, dtype='float16', accum_dtype='float32')
    state = outer_tl(q.clone(), k.clone(), v.clone(), o)
    # torch.testing.assert_close(h, state, rtol=1e-2, atol=1e-2)
    # print("✓ state正确性验证通过！")

    
    torch.testing.assert_close(o1, o, rtol=1e-2, atol=1e-2)
    print("✓ 正确性验证通过！")

    tilelang_ms = do_bench(lambda: outer_tl(q.clone(), k.clone(), v.clone(), o))
    native_ms = do_bench(lambda: outer_cumsum(q.clone(), k.clone(), v.clone(),  S//chunk_size, chunk_size))

    print(f"TileLang kernel avg latency: {tilelang_ms:.3f} ms")
    print(f"PyTorch native avg latency: {native_ms:.3f} ms")
    if tilelang_ms > 0:
        print(f"Speedup (native / TileLang): {native_ms / tilelang_ms:.2f}x")