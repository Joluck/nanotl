import tilelang
import tilelang.language as T
import torch

# q [b,s,h,d]
def navie_attn(q, k, v):

    qk = q.transpose(1, 2)@k.permute(0,2,3,1)
    sm = qk.softmax(dim=-1)
    o = sm@v.transpose(1, 2)
    return o.transpose(1, 2)

@tilelang.jit
def flash_attn_tl(B, S, H, D, block):
    return None

if __name__ == "__main__":

    B, S, H, D = 2, 1024, 8, 64
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)

    o = navie_attn(q, k, v)
    print(o)

    