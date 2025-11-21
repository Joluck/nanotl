import tilelang as tl  
import tilelang.language as T  
import torch  
  
M, K, R = 4096, 2048, 64  
  
@T.prim_func  
def kernel(  
    A: T.Tensor((M, K), "float16"),  
    B: T.Tensor((M, R), "float16"),  
):  
    with T.Kernel(T.ceildiv(M, 128), threads=128) as (bx,):  
        A_frag = T.alloc_fragment((128, K), "float16")  
        B_frag = T.alloc_fragment((128, R), "float16")  
          
        T.copy(A[bx*128:(bx+1)*128, :], A_frag)  
        A_reshaped = T.reshape(A_frag, [128, K // R, R])  
        T.reduce_sum(A_reshaped, B_frag, dim=1)  
        T.copy(B_frag, B[bx*128:(bx+1)*128, :])  
  
# 编译  
compiled = tl.compile(  
    kernel,   
    out_idx=-1,  
    pass_configs={  
        tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,  
        tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,  
    }  
)  
  
# 测试  
a = torch.randn(M, K, device="cuda", dtype=torch.float16)  
b = torch.empty(M, R, device="cuda", dtype=torch.float16)  
  
compiled(a, b)  
  
# 验证  
ref = torch.sum(a.reshape(M, K // R, R), dim=1)  
torch.testing.assert_close(b, ref, rtol=1e-2, atol=1e-2)