## MiSS: Revisiting the Trade-off in LoRA with an Efficient Shard-Sharing Structure
## [Paper](https://arxiv.org/abs/2409.15371)        [Github](https://github.com/Joluck/MiSS.git)
## torch
```
M, K, N, R = 1024, 1024, 1024, 64
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(R, N, device="cuda", dtype=torch.float16)
out = torch.sum(a.reshape(*a.shape[:-1], a.size(-1) // R, R), dim=-2)@b
```
## tilelang 不对矩阵M进行分块，不考虑sm大小情况时
```
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
            T.copy(B[by*R:(by+1)*R, :], B_shared)  

            for i in T.Pipelined(0, BR):
                T.copy(A[by, i*R:(i+1)*R], A_shared)
                for j in T.Parallel(R):  
                    sum_A[0, j] += A_shared[j].astype(accum_dtype)  
            for j in T.Parallel(N):  
                for k in T.serial(R):  
                    C_local[0, j] += sum_A[0, k] * B_shared[k, j]  


            T.copy(C_local, C[by, :])
    return kernel
```

## 为了使用gemm，高效进行矩阵乘所以我们对M进行分块满足T.gemm要求
```
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
```

## share memory显然放不下B_shared所以我们需要对N进行分块
## 当K增大时显著变慢
```
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

            # A_shared = T.alloc_shared((block_M, K), dtype=dtype)
            A_frag = T.alloc_fragment((block_M, R), dtype=dtype) # 临时寄存器
            B_shared = T.alloc_shared((R, block_N), dtype=dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            sum_A = T.alloc_fragment((block_M, R), dtype=dtype)

            T.clear(C_local)
            T.clear(sum_A)
            T.copy(B[:, bx*block_N:(bx+1)*block_N], B_shared)  
            for i in T.Pipelined(0, BR):
                T.copy(A[by*block_M:(by+1)*block_M, i*R:(i+1)*R], A_frag)
                for k, j in T.Parallel(block_M, R):  
                    sum_A[k, j] += A_frag[k, j].astype(accum_dtype)  

            T.gemm(sum_A, B_shared, C_local)  


            T.copy(C_local, C[by*block_M:(by+1)*block_M, bx*block_N:(bx+1)*block_N])
    return kernel
```