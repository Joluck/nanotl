import numpy as np
import torch


class Linear():
    def __init__(self, in_features, out_features, bias=True):
        self.weight = np.random.randn(in_features, out_features)
        self.bias = np.zeros(out_features) if bias else None

        self.grad_weight = None
        self.grad_bias = None
        self.x = None
    def forward(self, x):
        self.x = x
        out = x @ self.weight
        if self.bias is not None:
            out += self.bias
        return out
    
    def backward(self, grad_output):
        self.grad_x = grad_output @ self.weight.T
        self.grad_weight = self.x.T @ grad_output

        if self.bias is not None:
            self.grad_bias = grad_output.sum(axis=0)
        return self.grad_x
    def step(self, lr):
        self.weight -= lr * self.grad_weight
        if self.bias is not None:
            self.bias -= lr * self.grad_bias


        

class MLP():
    def __init__(self, in_features, hidden_features, out_features, bias=True):
        self.fc1 = Linear(in_features, hidden_features, bias)
        self.fc2 = Linear(hidden_features, out_features, bias)
        self.hidden = None

    def forward(self, x):
        x = self.fc1.forward(x)
        # self.hidden = np.maximum(0, x)  # ReLU activation
        self.hidden = 1/(1+ np.exp(-x))                     # sigmiod: 1 / (1 + np.exp(-x))
        x = self.fc2.forward(self.hidden)
        return x

    def backward(self, grad_output):
        grad_hidden = self.fc2.backward(grad_output)
        # grad_hidden[self.hidden <= 0] = 0  # ReLU backward
        grad_hidden = self.hidden * (1 - self.hidden) * grad_hidden  # sigmoid backward
        grad_input = self.fc1.backward(grad_hidden)
        return grad_input

    def step(self, lr):
        self.fc1.step(lr)
        self.fc2.step(lr)
# 与PyTorch对比测试
def test_linear():
    np.random.seed(42)
    
    # 创建测试数据
    batch_size = 4
    in_features = 3
    out_features = 2
    
    # 随机输入
    x_np = np.random.randn(batch_size, in_features)
    
    # 我们的实现
    linear_our = Linear(in_features, out_features, bias=True)
    linear_our.weight = np.random.randn(in_features, out_features)
    linear_our.bias = np.random.randn(out_features)
    
    # 前向传播
    output_our = linear_our.forward(x_np)
    
    # 随机梯度
    grad_output = np.random.randn(batch_size, out_features)
    
    # 反向传播
    grad_input_our = linear_our.backward(grad_output)
    
    # PyTorch实现
    import torch
    import torch.nn as nn
    
    # 设置相同的权重
    x_torch = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    linear_torch = nn.Linear(in_features, out_features, bias=True)
    
    # 手动设置相同的权重和偏置
    with torch.no_grad():
        linear_torch.weight.data = torch.tensor(linear_our.weight.T, dtype=torch.float32)  # PyTorch权重是转置的
        linear_torch.bias.data = torch.tensor(linear_our.bias.flatten(), dtype=torch.float32)
    
    # 前向传播
    output_torch = linear_torch(x_torch)
    
    # 反向传播
    output_torch.backward(torch.tensor(grad_output, dtype=torch.float32))
    
    # 对比结果
    print("前向传播对比:")
    print("我们的实现:", output_our[:2])
    print("PyTorch实现:", output_torch.detach().numpy()[:2])
    print("是否一致:", np.allclose(output_our, output_torch.detach().numpy(), rtol=1e-6))
    
    print("\n输入梯度对比:")
    print("我们的实现 grad_x:", grad_input_our)
    print("PyTorch grad_x:", x_torch.grad.numpy())
    print("是否一致:", np.allclose(grad_input_our, x_torch.grad.numpy(), rtol=1e-6))
    
    print("\n权重梯度对比:")
    print("我们的实现 grad_weight shape:", linear_our.grad_weight.shape)
    print("PyTorch grad_weight shape:", linear_torch.weight.grad.numpy().T.shape)  # 注意转置
    
    print("\n偏置梯度对比:")
    print("我们的实现 grad_bias:", linear_our.grad_bias)
    print("PyTorch grad_bias:", linear_torch.bias.grad.numpy())
    print("是否一致:", np.allclose(linear_our.grad_bias.flatten(), linear_torch.bias.grad.numpy(), rtol=1e-6))

if __name__ == "__main__":
    test_linear()