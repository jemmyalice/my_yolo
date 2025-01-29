import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from ultralytics.nn.AddModules import *
import torch
import torch.nn as nn


class CustomModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        """
        初始化自定义模块

        参数:
        in_channels (int): 输入特征图的通道数
        reduction (int): 通道缩减比例，默认为16
        """
        super(CustomModule, self).__init__()
        # 自适应平均池化层，将输入特征图池化为1x1的大小
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 第一个1x1卷积层，将输入特征图的通道数降为1
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2)
        # Sigmoid激活函数层，将输出值映射到(0, 1)区间
        self.sigmoid = nn.Sigmoid()
        # Softmax激活函数层，对指定维度进行归一化，输出概率分布
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 对输入特征图进行自适应平均池化，得到形状为 (batch_size, in_channels, 1, 1) 的特征图
        avg_pool_result = self.avgpool(x)
        avg_pool_result = torch.flatten(avg_pool_result, 1)
        avg_pool_result = avg_pool_result.unsqueeze(1)
        spatial_weight = torch.matmul(avg_pool_result, x.flatten(2))
        # 在第1维上增加一个维度，形状变为 (batch_size, 1, 1, height * width)
        spatial_weight = spatial_weight.unsqueeze(1)
        # 对空间注意力权重应用Sigmoid激活函数，将其值映射到(0, 1)区间
        spatial_weight = self.sigmoid(spatial_weight)

        # 右侧分支
        # 对输入特征图应用第一个1x1卷积层，得到形状为 (batch_size, 1, height, width) 的特征图
        conv1_result = self.conv1x1_1(x)
        # 将卷积结果展平为二维张量，形状为 (batch_size, height * width)
        conv1_result = torch.flatten(conv1_result, 1)
        # 在第0维上增加一个维度，形状变为 (batch_size, 1, height * width)
        conv1_result = conv1_result.unsqueeze(0)
        # 将卷积结果与输入特征图展平后的转置矩阵进行矩阵乘法，得到注意力权重，形状为 (batch_size, 1, 1, height * width)
        attention_weight = torch.matmul(conv1_result, x.flatten(2).transpose(1, 2))
        # 对注意力权重应用Softmax激活函数，得到概率分布
        attention_weight = self.softmax(attention_weight)
        # 将注意力权重与输入特征图展平后的转置矩阵进行矩阵乘法，再转置得到形状为 (batch_size, in_channels, 1, 1) 的特征图
        attention_result = torch.matmul(attention_weight, x.flatten(2).transpose(1, 2)).transpose(1, 2)
        # 对注意力结果应用第二个1x1卷积层，保持通道数不变
        attention_result = self.conv1x1_2(attention_result)

        # 中间部分
        # 对注意力结果应用第三个1x1卷积层，将通道数缩减为 in_channels // reduction
        mid_result = self.conv1x1_3(attention_result)
        # 调整特征图的维度顺序，将通道维度移到最后，方便进行层归一化
        mid_result = mid_result.permute(2, 3, 0, 1).contiguous()
        # 对特征图应用层归一化
        mid_result = self.ln(mid_result)
        # 恢复特征图的维度顺序
        mid_result = mid_result.permute(2, 3, 0, 1).contiguous()
        # 对特征图应用ReLU激活函数，增加非线性
        mid_result = self.relu(mid_result)
        # 对特征图应用第四个1x1卷积层，将通道数恢复为 in_channels
        mid_result = self.conv1x1_4(mid_result)

        # 最终融合
        # 将中间结果与空间注意力权重相乘
        final_result = mid_result * spatial_weight
        # 将融合结果与输入特征图相加，实现残差连接
        final_result = final_result + x
        return final_result


# 示例使用
# 生成一个随机输入特征图，形状为 (batch_size=32, in_channels=64, height=28, width=28)
input_tensor = torch.randn(32, 64, 28, 28)
# 初始化自定义模块，输入通道数为64
module = CustomModule(64)
# 调用模块的前向传播方法，得到输出特征图
output = module(input_tensor)
# 打印输出特征图的形状
print(output.shape)
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