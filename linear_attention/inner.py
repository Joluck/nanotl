import torch
from einops import rearrange
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


def inner(q, k, v, num_chunks, chunk_size, outer_o=None):
    _,_,_,_,DK = q.shape
    B, H, N, C,DV = v.shape
    inter_state = torch.zeros(B, H, num_chunks, DK, DV, device="cuda", dtype=torch.float32)

    inter_o = torch.zeros(B, H, N, C, DV, dtype=torch.float, device=q.device)#torch.zeros_like(outer_o)
    for t in range(chunk_size):
        local_state = k[:,:,:,:,t:t+1]@v[:,:,:,t:t+1,:]
        inter_state += local_state
        inter_o[:,:,:,t:t+1,:] = q[:,:,:,t:t+1,:]@inter_state
    return inter_o

def inner_attn(q, k, v, num_chunks, chunk_size, outer_o):
        
    qk = (q@k).masked_fill_(
        torch.triu(torch.ones(chunk_size, chunk_size, dtype=bool, device=q.device), diagonal=1),0)
    inter_o = qk@v

    return inter_o

@tilelang.jit
def inner_forward(B, S, H, DK, DV, dtype='float16', accum_dtype='float32'):
    chunk_size = 64
    NT = tilelang.cdiv(S, chunk_size)

    @T.prim_func
    def kernel(
        q: T.Tensor([B, S, H, DK], dtype),
        k: T.Tensor([B, S, H, DK], dtype),
        v: T.Tensor([B, S, H, DV], dtype),
        O: T.Tensor([B, S, H, DV], accum_dtype),
    ):
        with T.Kernel(B * H) as (i_bh):

            i_b = i_bh // H
            i_h = i_bh % H
            q_shared = T.alloc_shared([chunk_size, DK], dtype)
            k_shared = T.alloc_shared([chunk_size, DK], dtype)
            v_shared = T.alloc_shared([chunk_size, DV], dtype)
            h = T.alloc_fragment([DK, DV], accum_dtype)
            h_shared = T.alloc_shared([DK, DV], dtype)
            o = T.alloc_fragment([chunk_size, DV], accum_dtype)
            for i in T.Pipelined(0, NT):
                T.copy(q[i_b, i*chunk_size:(i+1)*chunk_size, i_h, :], q_shared)
                T.copy(k[i_b, i*chunk_size:(i+1)*chunk_size, i_h, :], k_shared)
                T.copy(v[i_b, i*chunk_size:(i+1)*chunk_size, i_h, :], v_shared)
                T.copy(h, h_shared)
                T.gemm(k_shared, v_shared, h, transpose_A=True)
                T.gemm(q_shared, h_shared, o)
            T.clear(h)
            T.clear(o)

            

            T.copy(o, O[i_b, i_n*chunk_size:(i_n+1)*chunk_size, i_h, :])
        T.Kernel
    return kernel

if __name__ == "__main__":
    B, S, H, D = 2, 2048, 64, 64
    DK, DV = 32, 16
    dtype = torch.float16
    q = torch.randn(B, S, H, DK, device="cuda", dtype=dtype)
    k = torch.randn(B, S, H, DK, device="cuda", dtype=dtype)
    v = torch.randn(B, S, H, DV, device="cuda", dtype=dtype)
    # w = torch.randn(B, S, D, DV, device="cuda", dtype=dtype)
    state = torch.zeros(B, H, D, D, device="cuda", dtype=torch.float32)
    chunk_size=64

    B, S, H, D = q.shape
    output_dtype = q.dtype  # 保存输出精度
    num_chunks = S//chunk_size

    q, k, v = map(lambda x: rearrange(x, 'b (n c) h d -> b h n c d', c=chunk_size).float(), (q, k, v))
    k = k.transpose(-1, -2)  

    recurrent_intra_o = inner(q, k, v, num_chunks, chunk_size, outer_o=None)

    attn_intra_o = inner_attn(q, k, v, num_chunks, chunk_size, outer_o=None)
    print(recurrent_intra_o.shape, attn_intra_o.shape)

    torch.testing.assert_close(recurrent_intra_o, attn_intra_o, rtol=1e-2, atol=1e-2)

    native = do_bench(lambda: inner(q, k, v, num_chunks, chunk_size, outer_o=None))
    attn = do_bench(lambda: inner_attn(q, k, v, num_chunks, chunk_size, outer_o=None))
    print(native, attn)
    print(f"Speedup (native / TileLang): {attn / native:.2f}x")
