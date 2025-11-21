import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


@tilelang.jit(out_idx=[1])
def tl_cumsum(B, S, N, block_B, dtype='float16', accum_dtype='float32'):
    @T.prim_func
    def forward(
        x: T.Tensor([B, S, N], dtype),
        y: T.Tensor([B, S, N], dtype),
    ):
        with T.Kernel(B) as (bx,):
            # x_shared = T.alloc_shared((N), dtype)
            x_shared = T.alloc_fragment((N), dtype)
            local_sum = T.alloc_fragment((N), accum_dtype)
            T.clear(local_sum)
            for i in T.Pipelined(0, S):
                T.copy(x[bx, i, :], x_shared)
                for j in T.Parallel(N):  
                    local_sum[j] += x_shared[j].astype(accum_dtype) 
                # T.atomic_add(local_sum, x_shared)
                T.copy(local_sum, y[bx, i, :])
    return forward

# def naive_cumsum(x):
#     for i in range(x.shape[0]-1):
#         x[i+1, :] += x[i, :]
#     return x

if __name__ == "__main__":
    B, S, N = 8, 1024, 1024
    block_B = 128
    x = torch.randn(B, S, N, device="cuda", dtype=torch.float16)
    y = x.float().cumsum(dim=1).to(torch.float16)

    tl_cumsum_kernel = tl_cumsum(B, S, N, block_B, dtype='float16', accum_dtype='float32')
    y2 = tl_cumsum_kernel(x)

    torch.testing.assert_close(y, y2, rtol=1e-2, atol=1e-2)
    tilelang_ms = do_bench(lambda: tl_cumsum_kernel(x))
    native_ms = do_bench(lambda:x.cumsum(dim=1))

    print(f"TileLang kernel avg latency: {tilelang_ms:.3f} ms")
    print(f"PyTorch native avg latency: {native_ms:.3f} ms")
    if tilelang_ms > 0:
        print(f"Speedup (native / TileLang): {native_ms / tilelang_ms:.2f}x")
