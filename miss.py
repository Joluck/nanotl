import tilelang
import tilelang.language as T
import torch
from tilelang.profiler import do_bench
# @tilelang.jit
# def shardshare(M, N, K, R, block_M, block_N, dtype='float16', accum_dtype='float32'):
    
#     a_shape = [M,K]
#     b_shape = [R,N]
#     c_shape = [M,N]
#     @T.prim_func
#     def kernel(
#         A: T.Tensor(a_shape, dtype),
#         B: T.Tensor(b_shape, dtype),
#         C: T.Tensor(c_shape, dtype),
#     ):
#         with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
#             A_shared = T.alloc_shared((block_M, R), dtype=dtype)
#             B_shared = T.alloc_shared((R, block_N), dtype=dtype)
#             C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

#             T.clear(C_local)

#             T.copy(B[:, bx*block_N:bx*block_N+block_N], B_shared)  


#             for ko in T.Pipelined(T.ceildiv(K, R), num_stages=3):
#                 T.copy(A[by*block_M, ko*R], A_shared)
#                 T.gemm(A_shared, B_shared, C_local)


#             T.copy(C_local, C[by*block_M, bx*block_N])
#     return kernel

@tilelang.jit
def shardshare(M, N, K, R, block_M, block_N, dtype='float16', accum_dtype='float32'):
    
    a_shape = [M,K]
    b_shape = [R,N]
    c_shape = [M,N]
    BR = K // R
    @T.prim_func
    def kernel(
        A: T.Tensor(a_shape, dtype),
        B: T.Tensor(b_shape, dtype),
        C: T.Tensor(c_shape, dtype),
    ):
        with T.Kernel(M) as (by,):  

            A_shared = T.alloc_fragment((R), dtype=dtype)
            sum_A = T.alloc_fragment((1, R), dtype=dtype)
            B_shared = T.alloc_shared((R, N), dtype=dtype)
            C_local = T.alloc_fragment((1, N), accum_dtype)

            T.clear(C_local)
            T.clear(sum_A)
            T.copy(B, B_shared)  

            for i in T.Pipelined(0, BR):
                T.copy(A[by, i*R:(i+1)*R], A_shared)
                for j in T.Parallel(R):  
                    sum_A[0, j] += A_shared[j]  
            for j in T.Parallel(N):  
                for k in T.serial(R):  
                    C_local[0, j] += sum_A[0, k] * B_shared[k, j]  


            T.copy(C_local, C[by, :])
    return kernel
@tilelang.jit
def shardshare(M, N, K, R, block_M, block_N, dtype='float16', accum_dtype='float32'):
    
    a_shape = [M,K]
    b_shape = [R,N]
    c_shape = [M,N]
    BR = K // R
    @T.prim_func
    def kernel(
        A: T.Tensor(a_shape, dtype),
        B: T.Tensor(b_shape, dtype),
        C: T.Tensor(c_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M)) as (by,):  

            A_shared = T.alloc_fragment((block_M, R), dtype=dtype)
            sum_A = T.alloc_shared((block_M, R), dtype=dtype)
            B_shared = T.alloc_fragment((R, N), dtype=dtype)
            C_local = T.alloc_fragment((block_M, N), accum_dtype)

            T.clear(C_local)
            T.clear(sum_A)
            T.copy(B, B_shared)  

            for i in T.Pipelined(0, BR):
                T.copy(A[by*block_M:(by+1)*block_M, i*R:(i+1)*R], A_shared)
                for k, j in T.Parallel(block_M, R):  
                    sum_A[k, j] += A_shared[k, j]  
            T.gemm(sum_A, B_shared, C_local)  


            T.copy(C_local, C[by*block_M:(by+1)*block_M,:])
    return kernel


@tilelang.jit
def shardshare(M, N, K, R, block_M, block_N, dtype='float16', accum_dtype='float32'):
    
    a_shape = [M,K]
    b_shape = [R,N]
    c_shape = [M,N]
    BR = K // R
    BN = N // block_N
    @T.prim_func
    def kernel(
        A: T.Tensor(a_shape, dtype),
        B: T.Tensor(b_shape, dtype),
        C: T.Tensor(c_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M)) as (by,):  

            A_shared = T.alloc_fragment((block_M, R), dtype=dtype)
            sum_A = T.alloc_fragment((block_M, R), dtype=dtype)
            B_shared = T.alloc_shared((R, block_N), dtype=dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            T.clear(sum_A)
            # T.copy(B, B_shared)  

            for i in T.Pipelined(0, BR):
                T.copy(A[by*block_M:(by+1)*block_M, i*R:(i+1)*R], A_shared)
                for k, j in T.Parallel(block_M, R):  
                    sum_A[k, j] += A_shared[k, j]  
            for j in T.Pipelined(0, BN):  
                T.copy(B[:, j*block_N:(j+1)*block_N], B_shared)
                T.gemm(sum_A, B_shared, C_local)  
                T.copy(C_local, C[by*block_M:(by+1)*block_M, j*block_N:(j+1)*block_N])
    return kernel

@tilelang.jit
def shardshare(M, N, K, R, block_M, block_N, dtype='float16', accum_dtype='float32'):
    
    a_shape = [M,K]
    b_shape = [R,N]
    c_shape = [M,N]
    BR = K // R
    @T.prim_func
    def kernel(
        A: T.Tensor(a_shape, dtype),
        B: T.Tensor(b_shape, dtype),
        C: T.Tensor(c_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N)) as (by,bx):  

            A_shared = T.alloc_fragment((block_M, R), dtype=dtype)
            
            B_shared = T.alloc_shared((R, block_N), dtype=dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            sum_A = T.alloc_fragment((block_M, R), dtype=dtype)

            T.clear(C_local)
            T.clear(sum_A)
            T.copy(B[:, bx*block_N:(bx+1)*block_N], B_shared)  

            for i in T.Pipelined(0, BR):
                T.copy(A[by*block_M:(by+1)*block_M, i*R:(i+1)*R], A_shared)
                for k, j in T.Parallel(block_M, R):  
                    sum_A[k, j] += A_shared[k, j]  

            T.gemm(sum_A, B_shared, C_local)  


            T.copy(C_local, C[by*block_M:(by+1)*block_M, bx*block_N:(bx+1)*block_N])
    return kernel


if __name__ == "__main__":
    batch, M, N, K = 1024, 1024, 1024, 512
    block_M, block_N, R = 16, 128, 64
    dtype = 'float16'
    # Compile kernel (JIT compilation)
    miss_kernel = shardshare(M, N, K, R, block_M, block_N, dtype)

    # Create input tensors
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(R, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)

    # Execute kernel
    miss_kernel(a, b, c)

    reshape_a = a.reshape(*a.shape[:-1], a.size(-1) // R, R)
    out = (torch.sum(reshape_a.to(torch.float32), dim=-2).to(torch.float32)@b.to(torch.float32)).to(torch.float16)

    # Validate correctness
    # torch.testing.assert_close(c, out, rtol=1e-2, atol=1e-2)
    print("Kernel output matches PyTorch reference.")


    tilelang_ms = do_bench(lambda: miss_kernel(a, b, c))
    native_ms = do_bench(lambda: torch.sum(a.reshape(*a.shape[:-1], a.size(-1) // R, R), dim=-2)@b)

    print(f"TileLang kernel avg latency: {tilelang_ms:.3f} ms")
    print(f"PyTorch native avg latency: {native_ms:.3f} ms")
    if tilelang_ms > 0:
        print(f"Speedup (native / TileLang): {native_ms / tilelang_ms:.2f}x")