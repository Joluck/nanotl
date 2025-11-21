import tilelang  
import tilelang.language as T  
  
@tilelang.jit  
def fused_reduce_matmul(M, K, R, N, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):  
      
    @T.prim_func  
    def kernel(  
        A: T.Tensor((M, K), dtype),  
        B: T.Tensor((R, N), dtype),  
        C: T.Tensor((M, N), dtype),  
    ):  
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):  
            # Allocate shared memory  
            A_shared = T.alloc_shared((block_M, block_K), dtype)  
            B_shared = T.alloc_shared((R, block_N), dtype)  
            A_reduced = T.alloc_shared((block_M, R), accum_dtype)  
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)  
              
            # Load B once (it's small and constant)  
            T.copy(B[0:R, bx*block_N:bx*block_N+block_N], B_shared)  
              
            # Clear accumulator  
            T.clear(C_local)  
            T.clear(A_reduced)  
              
            # Loop over K dimension, reducing as we go  
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):  
                # Load A tile  
                T.copy(A[by*block_M:by*block_M+block_M, ko*block_K:ko*block_K+block_K], A_shared)  
                  
                # Reduce A_shared along K dimension into A_reduced  
                # Each element A_reduced[i, r] accumulates block_K/R elements from A_shared  
                for i, k in T.Parallel(block_M, block_K):  
                    r_idx = (ko * block_K + k) % R  
                    A_reduced[i, r_idx] = A_reduced[i, r_idx] + A_shared[i, k]  
              
            # Now perform GEMM: A_reduced @ B_shared -> C_local  
            T.gemm(A_reduced, B_shared, C_local)  
              
            # Write result  
            T.copy(C_local, C[by*block_M:by*block_M+block_M, bx*block_N:bx*block_N+block_N])  
      
    return kernel  
  
# Usage  
M, K, R, N = 1024, 2048, 64, 1024  
block_M, block_N, block_K = 64, 64, 64
  
kernel = fused_reduce_matmul(M, K, R, N, block_M, block_N, block_K)  
  
a = torch.randn(M, K, device="cuda", dtype=torch.float16)  
b = torch.randn(R, N, device="cuda", dtype=torch.float16)  
c = torch.empty(M, N, device="cuda", dtype=torch.float16)  
  
kernel(a, b, c)