# flops计算 = 如果改了yolo里面的用原来的算 + 手动计算的MF
import torch

# 创建一个 tensor
x = torch.rand(5, 3)
print(x)

# 检查 CUDA 是否可用
if torch.cuda.is_available():
    device = torch.device('cuda')  # 如果有 GPU 可用
else:
    device = torch.device('cpu')    # 使用 CPU

print(f"Using device: {device}")