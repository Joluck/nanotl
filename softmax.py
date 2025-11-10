
import torch



def navie_softmax(x):
    x_max = x.max(dim=-1, keepdim=True).values
    x_stable = x-x_max
    x_exp = torch.exp(x_stable)
    l = torch.sum(x_exp, dim=-1, keepdim=True)
    return x_exp/l

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
if __name__=='__main__':
    B, D = 1024, 1024*8
    block_s = 128
    x = torch.randn(B,D, device="cuda", dtype=torch.float16)
    y = torch.empty(B, device="cuda", dtype=torch.float16)
    o1 = navie_softmax(x)
    o2 = torch_softmax(x)
    o3 = online_softmax(x)
    print(o3)
    # print(o1)
    print(o2)
    torch.testing.assert_close(o3, o2, rtol=1e-2, atol=1e-2)
    print("Kernel output matches PyTorch reference.")

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

    tilelang_ms = benchmark(lambda: online_softmax(x))
    native_ms = benchmark(lambda: torch_softmax(x))

    print(f"TileLang kernel avg latency: {tilelang_ms:.3f} ms")
    print(f"PyTorch native avg latency: {native_ms:.3f} ms")
    if tilelang_ms > 0:
        print(f"Speedup (native / TileLang): {native_ms / tilelang_ms:.2f}x")
