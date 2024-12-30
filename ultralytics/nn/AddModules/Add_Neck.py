import torch  # 导入 PyTorch 库
from torch import nn  # 从 PyTorch 中导入神经网络模块
import torch.nn.functional as F
import torch.nn.Conv2d as conv

class Upsample(nn.Module):
    """Applies convolution followed by upsampling."""

    # ---1.渐进架构部分（融合前的准备）--- #
    def __init__(self, c1, c2, scale_factor=2):
        super().__init__()
        # self.cv1 = Conv(c1, c2, 1)
        # self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')  # or model='bilinear' non-deterministic
        if scale_factor==2:
            self.cv1 = nn.ConvTranspose2d(c1, c2, 2, 2, 0, bias=True)  # 如果下采样率为2，就用Stride为2的2×2卷积来实现2次下采样
        elif scale_factor==4:
            self.cv1 = nn.ConvTranspose2d(c1, c2, 4, 4, 0, bias=True)  # 如果下采样率为4，就用Stride为4的4×4卷积来实现4次下采样

    def forward(self, x):
        # return self.upsample(self.cv1(x))
        return self.cv1(x)


# ---2.自适应空间融合（ASFF）--- #
class ASFF2(nn.Module):
    """ASFF2 module for YOLO AFPN head https://arxiv.org/abs/2306.15988"""

    def __init__(self, c1, c2, level=0):
        super().__init__()
        c1_l, c1_h = c1[0], c1[1]
        self.level = level
        self.dim = c1_l, c1_h
        self.inter_dim = self.dim[self.level]
        compress_c = 8

        # 如果是第0层
        if level==0:
            # self.stride_level_1调整level-1出来的特征图，通道调整为和level-0出来的特征图一样大小
            self.stride_level_1 = Upsample(c1_h, self.inter_dim)
        # 如果是第1层
        if level==1:
            # self.stride_level_0通道调整为和level-1出来的特征图一样大小
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 2, 2, 0)  # stride=2 下采样为2倍

        # 两个卷积为了学习权重
        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        # 用于调整拼接后的两个权重的通道
        self.weights_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, x):
        x_level_0, x_level_1 = x[0], x[1]

        # 如果在第0层
        # level-0出来的特征图保持不变
        # 调整level-1的特征图，使得其channel、width、height与level-0一致
        if self.level==0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
        # 如果在第1层，同上
        elif self.level==1:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = x_level_1

        # 将N*C*H*W的level-0特征图卷积得到权重，权重level_0_weight_v:N*256*H*W
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)

        # 将各个权重矩阵按照通道拼接
        # levels_weight_v：N*3C*H*W
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v), 1)

        # 将拼接后的矩阵调整，每个通道对应着不同的level_0_resized，level_1_resized的权重
        levels_weight = self.weights_levels(levels_weight_v)

        # 在通道维度，对权重做归一化，也就是对于二通道tmp：tmp[0][0]+tmp[1][0]=1
        levels_weight = F.softmax(levels_weight, dim=1)

        # 将levels_weight各个通道分别乘level_0_resized level_1_resized
        # 点乘用到了广播机制
        fused_out_reduced = level_0_resized * levels_weight[:, 0:1] + level_1_resized * levels_weight[:, 1:2]
        return self.conv(fused_out_reduced)


# ASFF3的运算流程同上
class ASFF3(nn.Module):
    """ASFF3 module for YOLO AFPN head https://arxiv.org/abs/2306.15988"""

    def __init__(self, c1, c2, level=0):
        super().__init__()
        c1_l, c1_m, c1_h = c1[0], c1[1], c1[2]
        self.level = level
        self.dim = c1_l, c1_m, c1_h
        self.inter_dim = self.dim[self.level]
        compress_c = 8

        if level==0:
            self.stride_level_1 = Upsample(c1_m, self.inter_dim)
            self.stride_level_2 = Upsample(c1_h, self.inter_dim, scale_factor=4)

        if level==1:
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 2, 2, 0)  # downsample 2x
            self.stride_level_2 = Upsample(c1_h, self.inter_dim)

        if level==2:
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 4, 4, 0)  # downsample 4x
            self.stride_level_1 = Conv(c1_m, self.inter_dim, 2, 2, 0)  # downsample 2x

        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weights_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, x):
        x_level_0, x_level_1, x_level_2 = x[0], x[1], x[2]

        if self.level==0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = self.stride_level_2(x_level_2)

        elif self.level==1:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)

        elif self.level==2:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        w = self.weights_levels(levels_weight_v)
        w = F.softmax(w, dim=1)

        fused_out_reduced = level_0_resized * w[:, :1] + level_1_resized * w[:, 1:2] + level_2_resized * w[:, 2:]
        return self.conv(fused_out_reduced)