import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


# @tilelang.jit(out_idx=[1])
# def sum_kernel(S, D, R, dtype='float16', accum_dtype='float32'):
    
#     a_shape = [S,D]
#     c_shape = [S,R]
#     BR = D // R
#     @T.prim_func
#     def kernel(
#         A: T.Tensor(a_shape, dtype),
#         C: T.Tensor(c_shape, dtype),
#     ):
#         with T.Kernel(S) as (by,):  

#             A_shared = T.alloc_fragment((R), dtype=dtype)
#             sum_A = T.alloc_fragment((1, R), dtype=accum_dtype)

#             T.clear(sum_A)

#             for i in T.Pipelined(0, BR):
#                 T.copy(A[by, i*R:(i+1)*R], A_shared)
#                 for j in T.Parallel(R):  
#                     sum_A[0, j] += A_shared[j].astype(accum_dtype)  

#             T.copy(sum_A, C[by, :])
#     return kernel

@tilelang.jit(out_idx=[1])
def sum_kernel(S, D, R, block_S, dtype='float16', accum_dtype='float32'):
    
    a_shape = [S,D]
    c_shape = [S,R]
    BR = D // R
    @T.prim_func
    def kernel(
        A: T.Tensor(a_shape, dtype),
        C: T.Tensor(c_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(S, block_S)) as (by,):  

            A_frag = T.alloc_fragment((block_S, R), dtype=dtype)
            sum_A = T.alloc_fragment((block_S, R), accum_dtype)

            T.clear(sum_A)

            for i in T.Pipelined(0, BR):
                T.copy(A[by*block_S:(by+1)*block_S, i*R:(i+1)*R], A_frag)
                for s, j in T.Parallel(block_S, R):  
                    sum_A[s, j] += A_frag[s, j].astype(accum_dtype)
            T.copy(sum_A, C[by*block_S:(by+1)*block_S, :])
    return kernel
if __name__ == "__main__":
    S, D = 1024*4, 1024*4
    block_S = 16
    R = 64
    dtype = torch.float16
    x = torch.randn(S, D, device="cuda", dtype=dtype)
    y = torch.sum(x.to(torch.float32).reshape(*x.shape[:-1], x.size(-1) // R, R), dim=-2).to(dtype)

    sum_tl = sum_kernel(S, D, R, block_S, dtype='float16', accum_dtype='float32')
    y2 = sum_tl(x)

    torch.testing.assert_close(y, y2, rtol=1e-2, atol=1e-2)
    tilelang_ms = do_bench(lambda:sum_tl(x))
    native_ms = do_bench(lambda:torch.sum(x.reshape(*x.shape[:-1], x.size(-1) // R, R), dim=-2))

    print(f"TileLang kernel avg latency: {tilelang_ms:.3f} ms")
    print(f"PyTorch native avg latency: {native_ms:.3f} ms")
    if tilelang_ms > 0:
        print(f"Speedup (native / TileLang): {native_ms / tilelang_ms:.2f}x")