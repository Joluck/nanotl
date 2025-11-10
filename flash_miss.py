import tilelang
import tilelang.language as T
import torch
from torch._C import R

@tilelang.jit
def baddmm(batch, M, N, K, R, block_M, block_N, block_K, dtype='float16', accum_dtype='float32'):
    
    a_shape = [batch,M,K]
    b_shape = [R,N]
    c_shape = [batch,M,N]
    @T.prim_func
    def kernel(
        A: T.Tensor(a_shape, dtype),
        B: T.Tensor(b_shape, dtype),              # 或 (B, K, N)
        C: T.Tensor(c_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), batch, threads=128) as (bx, by, bb):
            A_shared = T.alloc_shared((block_M, block_K), dtype=dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype=dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            T.copy(B[:, bx*block_N], B_shared)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[bb, by*block_M, ko*block_K], A_shared)
                # 若 B 有批维：T.copy(B[bb, ko*block_K, bx*block_N], B_shared)
                # T.copy(B[bb, ko*block_K, bx*block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            # for i, j in T.Parallel(block_M, block_N):
                # C_local[i, j] += A[bb, by*block_M + i, bx*block_N + j]

            T.copy(C_local, C[bb, by*block_M, bx*block_N])
    return kernel



if __name__ == "__main__":
    batch, M, N, K = 8, 1024*4, 1024*2, 1024*2
    block_M, block_N, block_K = 128, 128, 128
    r = 8

    # Compile kernel (JIT compilation)
    matmul_relu_kernel = baddmm(batch, M, N, K,r, block_M, block_N, block_K)

    # Create input tensors
    a = torch.randn(batch, M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(r, N, device="cuda", dtype=torch.float16)
    c = torch.empty(batch, M, N, device="cuda", dtype=torch.float16)

    # Execute kernel
    matmul_relu_kernel(a, b, c)
    print(c)
    # Reference computation
    ref_c = (a @ b) + a

    out =  torch.sum(a.reshape(*a.shape[:-1], a.size(-1) // r, r), dim=-2)
    print(c.view(-1))
    print(ref_c.view(-1))

    # Validate correctness
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Kernel output matches PyTorch reference.")

    # Profile performance: compare TileLang kernel vs PyTorch native (cuBLAS)
    def benchmark(fn, iters=100, warmup=10):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters  # ms per iteration

    tilelang_ms = benchmark(lambda: matmul_relu_kernel(a, b, c))
    native_ms = benchmark(lambda: (a @ b) + a)

    print(f"TileLang kernel avg latency: {tilelang_ms:.3f} ms")
    print(f"PyTorch native avg latency: {native_ms:.3f} ms")
    if tilelang_ms > 0:
        print(f"Speedup (native / TileLang): {native_ms / tilelang_ms:.2f}x")