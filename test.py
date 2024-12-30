import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from ultralytics.nn.AddModules import *
#
# # 假设输入是batch_size=8的RGB和红外图像
# r1 = torch.randn(8, 3, 640, 640)  # RGB图像
# i1 = torch.randn(8, 3, 640, 640)  # 红外图像
#
# # 初始化模型
# model = FusionNetwork(in_channels_rgb=3, in_channels_ir=3, out_channels=64)
#
# # 打印模型结构
# print(model)
#
# # 前向传播
# output = model(r1, i1)
# print(output.shape)  # 输出（8, 64, 640, 640）


# # 示例输入
# rgb_input = torch.randn(1, 3, 32, 32)  # RGB输入 (Batch, Channels, Height, Width)
# ir_input = torch.randn(1, 1, 32, 32)   # IR输入 (Batch, Channels, Height, Width)
#
# add = My_add()
# out = add(rgb_input,ir_input)
#
# print(out.shape)

# # 创建ACDF模块
# acdf_block = ACDFBlock(rgb_channels=3, ir_channels=1, reduction=16, fused_channels=64)
#
# # 前向计算
# fused_output = acdf_block(rgb_input, ir_input)
# print(fused_output.shape)  # 输出形状: torch.Size([1, 192, 32, 32])

# 示例输入
# rgb_input = torch.randn(1, 3, 32, 32)  # RGB输入 (Batch, Channels, Height, Width)
# ir_input = torch.randn(1, 3, 32, 32)   # IR输入 (Batch, Channels, Height, Width)
#
# add = MF5(3)
# out = add(rgb_input,ir_input)
#
# print(out.shape)
start = time.time()
rgb_input = torch.randn(2, 3, 32, 32) # RGB输入 (Batch, Channels, Height, Width)
ir_input = torch.randn(2, 3, 32, 32) # IR输入 (Batch, Channels, Height, Width)

add = MF_25(3)
out = add(rgb_input,ir_input)
end = time.time()

print(out.shape, end - start)
# 0.008503198623657227
# 0.008976936340332031
# 0.007519721984863281