import torch
import tilelang
import tilelang.language as T
from einops import rearrange
@tilelang.jit
def linear_model(x, w, b):
    return x@w + b

def linear_model_naive(q, k, v, w):
    B, T, H, D = q.shape
    output_dtype = q.dtype  # 保存输出精度
    
    # 累积计算使用 float32 提高精度
    q = q.float()
    k = k.float()
    v = v.float()
    
    o = torch.empty(B, H, T, D, device="cuda", dtype=torch.float32)
    state = torch.zeros(B, H, D, D, device="cuda", dtype=torch.float32)
    k = k.permute(0,2,3,1)
    v = v.transpose(1, 2)
    q = q.transpose(1, 2)
    for t in range(T):
        local_state = k[:,:,:,t:t+1]@v[:,:,t:t+1,:]
        state += local_state
        o[:,:,t:t+1,:] = q[:,:,t:t+1,:]@state
    
    # 转回目标精度
    return o.transpose(1, 2).to(output_dtype)


def linear_model_chunk(q, k, v, w, chunk_size=32):
    B, T, H, D = q.shape
    output_dtype = q.dtype  # 保存输出精度
    num_chunks = T//chunk_size
    
    # 累积计算使用 float32 提高精度
    q, k, v, w = map(lambda x: rearrange(x, 'b (n c) h d -> b h n c d', c=chunk_size).float(), (q, k, v, w))
    
    k = k.transpose(-1, -2)
    inter_state = torch.zeros(B, H, num_chunks, D, D, device="cuda", dtype=torch.float32)
    
    # 计算跨 chunk 的累积状态
    outer_state = k@v  # [b,h,n,d,d]
    outer_s = torch.zeros_like(outer_state)
    for n in range(num_chunks-1):
        outer_s[:,:,n+1] = outer_s[:,:,n] + outer_state[:,:,n]
    outer_o = q@outer_s
    
    # 计算 chunk 内的累积
    inter_o = torch.zeros_like(outer_o)
    for t in range(chunk_size):
        local_state = k[:,:,:,:,t:t+1]@v[:,:,:,t:t+1,:]
        inter_state += local_state
        inter_o[:,:,:,t:t+1,:] = q[:,:,:,t:t+1,:]@inter_state
    
    # 合并并转回目标精度
    o = rearrange(inter_o + outer_o, 'b h n c d -> b (n c) h d').to(output_dtype)
    return o

def benchmark(fn, iters=100, warmup=10):
    """性能测试函数"""
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

if __name__ == "__main__":
    B, T, H, D = 8, 4096, 8, 64
    dtype = torch.bfloat16
    q = torch.randn(B, T, H, D, device="cuda", dtype=dtype)
    k = torch.randn(B, T, H, D, device="cuda", dtype=dtype)
    v = torch.randn(B, T, H, D, device="cuda", dtype=dtype)
    w = torch.randn(B, T, D, D, device="cuda", dtype=dtype)

    print("=" * 60)
    print(f"测试配置: B={B}, T={T}, H={H}, D={D}")
    print("=" * 60)

    # 正确性测试
    print("\n正在验证正确性...")
    o1 = linear_model_naive(q.clone(), k.clone(), v.clone(), w.clone())
    o2 = linear_model_chunk(q.clone(), k.clone(), v.clone(), w.clone(), chunk_size=64)
    torch.testing.assert_close(o1, o2, rtol=1e-2, atol=1e-2)
    print("✓ 正确性验证通过！")

    # 性能测试
    print("\n正在测速...")
    naive_ms = benchmark(lambda: linear_model_naive(q.clone(), k.clone(), v.clone(), w.clone()))
    chunk_ms = benchmark(lambda: linear_model_chunk(q.clone(), k.clone(), v.clone(), w.clone(), chunk_size=128))

    print(f"\n{'方法':<20} {'耗时 (ms)':<15} {'加速比':<10}")
    print("-" * 60)
    print(f"{'linear_model_naive':<20} {naive_ms:>10.3f} ms   {1.0:>6.2f}x")
    print(f"{'linear_model_chunk':<20} {chunk_ms:>10.3f} ms   {naive_ms/chunk_ms:>6.2f}x")
    print("=" * 60)
    
    if chunk_ms < naive_ms:
        print(f"\n🎉 chunk 版本快 {naive_ms/chunk_ms:.2f}x!")
    else:
        print(f"\n⚠️  naive 版本快 {chunk_ms/naive_ms:.2f}x")