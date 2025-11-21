import tilelang
import tilelang.language as T
import torch
from tilelang.profiler import do_bench



def block_softmax(x):
    x_max = x.max(dim=-1, keepdim=True).values
    x_stable = x-x_max
    x_exp = torch.exp(x_stable)
    l = torch.sum(x_exp, dim=-1, keepdim=True)
    return l, x_max

def online_softmax(x, chunk=32):
    b, t = x.shape
    block = t // chunk
    local_max = torch.zeros(B, 1, device=x.device, dtype=x.dtype)
    local_l = torch.zeros(B, 1, device=x.device, dtype=x.dtype)
    # new_max = torch.zeros(B, 1, device=x.device, dtype=x.dtype)
    old_max = torch.zeros(B, 1, device=x.device, dtype=x.dtype)
    l = torch.zeros(B, 1, device=x.device, dtype=x.dtype)
    xx = x.reshape(x.size(0), block, chunk)
    for t in range(block):
        
        local_l, local_max = block_softmax(xx[:, t, :])
        new_max = torch.max(local_max, old_max)
        l = l*(torch.exp(old_max-new_max))+local_l*(torch.exp((local_max-new_max)))
        old_max  = new_max

    x_stable = torch.exp(x-new_max)
    return x_stable/l





def torch_softmax(x):
    return x.softmax(dim=-1)


def naive_softmax(x):
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret

# @tilelang.jit(out_idx=[1])
# def tilelang_softmax(M, N, block_M, dtype='float32', accum_dtype='float32'):
#     @T.prim_func
#     def forward(
#         x: T.Tensor([M, N], dtype),
#         y: T.Tensor([M, N], dtype),
#     ):
#         with T.Kernel(M, threads=128) as (by):
#             x_shared = T.alloc_shared([N], dtype)
#             y_shared = T.alloc_shared([N], dtype)  

#             max_x = T.alloc_fragment([1], dtype)

#             T.copy(x[by, 0:N], x_shared)

#             T.reduce_max(x_shared, max_x, dim=0, clear=True)

#             # for i,j in T.Parallel(block_M, N):  
#             #     exp_x[i, j] = T.exp(T.cast(x_shared[i, j] - max_x[i], accum_dtype))
            
#             # T.reduce_sum(exp_x, sum_x, dim=-1, clear=True)
#             # for i,j in T.Parallel(block_M, N):
#             #     y_shared[i, j] = x_shared[i, j] / sum_x[i]  

#             # T.copy(y_shared, y[by*block_M:(by+1)*block_M, :])  

#     return forward

@tilelang.jit(out_idx=[1])
def tilelang_softmax(M, N, block_M, dtype='float16', accum_dtype='float32'):
    @T.prim_func
    def forward(
        x: T.Tensor([M, N], dtype),
        y: T.Tensor([M, N], dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=128) as by:
            x_shared = T.alloc_shared((block_M, N), dtype)
            y_shared = T.alloc_shared((block_M, N), dtype)
            
            # 每行的 max 和 sum，用 shared memory
            max_x = T.alloc_fragment((block_M,), accum_dtype)
            sum_x = T.alloc_shared((block_M,), accum_dtype)

            # copy global -> shared
            T.copy(x[by*block_M:(by+1)*block_M, :], x_shared)
            T.fill(max_x, -T.infinity(dtype))  

            # per-row reduce max
            #T.reduce_max(x_shared, max_x, dim=-1, clear=True)
            for i, j in T.Parallel(block_M, N):  
                max_x[i] = T.max(max_x[i], T.cast(x_shared[i, j], accum_dtype))  
            # # 计算 exp(x - max)
            for i, j in T.Parallel(block_M, N):
                y_shared[i, j] = T.exp(T.cast(x_shared[i, j] - max_x[i], accum_dtype))

            # # per-row reduce sum
            # T.reduce_sum(y_shared, sum_x, dim=-1, clear=True)

            # # normalize
            # for i, j in T.Parallel(block_M, N):
            #     y_shared[i, j] = y_shared[i, j] / sum_x[i]

            # # copy shared -> global
            # T.copy(y_shared, y[by*block_M:(by+1)*block_M, :])

    return forward

import tilelang  
import tilelang.language as T  
  
@tilelang.jit(out_idx=[-1])  
def softmax(M, N, block_M=128, block_N=128, dtype="float16", accum_dtype="float"):  
      
    @T.prim_func  
    def main(  
        A: T.Tensor((M, N), dtype),  
        Output: T.Tensor((M, N), dtype),  
    ):  
        with T.Kernel(T.ceildiv(M, block_M), threads=128) as (bx,):  
            # 分配共享内存和寄存器  
            A_shared = T.alloc_shared((block_M, N), dtype)  
            Output_shared = T.alloc_shared((block_M, N), dtype)  
              
            # 分配用于存储中间结果的寄存器  
            max_val = T.alloc_fragment((block_M,), accum_dtype)  
            sum_val = T.alloc_fragment((block_M,), accum_dtype)  
              
            # 加载数据到共享内存  
            T.copy(A[bx * block_M:(bx + 1) * block_M, :], A_shared)  
              
            # 第一步: 找到每行的最大值  
            T.fill(max_val, -T.infinity(accum_dtype))  
            for i in T.Parallel(block_M):  
                for j in T.serial(N):  
                    max_val[i] = T.max(max_val[i], T.cast(A_shared[i, j], accum_dtype))  
              
            # 第二步: 计算 exp(x - max) 并求和  
            T.fill(sum_val, 0)  
            for i, j in T.Parallel(block_M, N):  
                exp_val = T.exp(T.cast(A_shared[i, j], accum_dtype) - max_val[i])  
                Output_shared[i, j] = T.cast(exp_val, dtype)  
                sum_val[i] += exp_val  
              
            # 第三步: 归一化  
            for i, j in T.Parallel(block_M, N):  
                Output_shared[i, j] = T.cast(  
                    T.cast(Output_shared[i, j], accum_dtype) / sum_val[i],   
                    dtype  
                )  
              
            # 写回全局内存  
            T.copy(Output_shared, Output[bx * block_M:(bx + 1) * block_M, :])  
      
    return main
if __name__=='__main__':
    B, D = 1024, 128
    block_s = 8
    x = torch.randn(B,D, device="cuda", dtype=torch.float16)
    y = torch.empty(B,D, device="cuda", dtype=torch.float16)
    o1 = naive_softmax(x)
    kernel = tilelang_softmax(B, D, block_s)
    o2 = kernel(x)
    print(o2)

    torch.testing.assert_close(o1, o2, rtol=1e-2, atol=1e-2)
    print("Kernel output matches PyTorch reference.")
    do_bench(lambda: online_softmax(x))
    t1 = do_bench(lambda: torch_softmax(x))
    t2 = do_bench(lambda: naive_softmax(x))
    print(f"torch latency: {t1:.3f} ms")
    print(f"TileLang latency: {t2:.3f} ms")
    print(f"Speedup: {t1/t2:.3f}x")