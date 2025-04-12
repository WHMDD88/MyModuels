import torch
import torch.nn as nn
from torchsummary import summary

# 通道注意力模块，用于计算通道维度上的注意力权重
class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16):
        super(ChannelAttention, self).__init__()
        # 全局平均池化层，将输入特征图在高度和宽度维度上池化为1x1大小
        # 接收一个整数或者一个元组作为参数，意味着输出特征图的高度和宽度都会被池化为该整数所指定的值
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 第一个卷积层，用于通道降维，将通道数从 channels 降到 channels // ratio
        self.fc1 = nn.Conv2d(channels, channels // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        # 第二个卷积层，用于通道升维，将通道数从 channels // ratio 恢复到 channels
        self.fc2 = nn.Conv2d(channels // ratio, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)#(B, N, 1, 1)
        max_out = self.max_pool(x)#(B, N, 1, 1)
        # 通过第一个卷积层进行通道降维
        avg_out = self.fc1(avg_out)# (B, N//ratio, 1, 1)
        avg_out = self.relu(avg_out)
        # 最大池化结果同样通过第一个卷积层进行通道降维
        max_out = self.fc1(max_out)
        max_out = self.relu(max_out)
        # 通过第二个卷积层进行通道升维
        avg_out = self.fc2(avg_out)#(B, N, 1, 1)
        max_out = self.fc2(max_out)
        # 将平均池化和最大池化得到的结果相加
        out = avg_out + max_out#(B, N, 1, 1)
        # 经过Sigmoid激活函数，将输出值映射到(0, 1)区间，作为通道注意力权重
        out = self.sigmoid(out)
        return out

# 空间注意力模块，用于计算空间维度上的注意力权重
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 确保卷积核大小为3或7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入形状 (B, N, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)#(B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)#(B, 1, H, W)
        # 将平均池化和最大池化的结果在通道维度上拼接
        x = torch.cat([avg_out, max_out], dim=1)#(B, 2, H, W)
        # 通过卷积层将2通道特征图卷积为1通道特征图
        x = self.conv1(x)# (B, 1, H, W)
        x = self.sigmoid(x)
        return x  #输出形状 (B, 1, H, W)

# CBAM模块，结合了通道注意力和空间注意力
class CBAM(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        # 实例化通道注意力模块
        self.channel_attention = ChannelAttention(channels, ratio)
        # 实例化空间注意力模块
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # 输入维度: (B, N, H, W)>(4,16,32,32)
        out = x * self.channel_attention(x)#广播机制 (B, N, H, W)*(B, N, 1, 1)
        out = out * self.spatial_attention(out)#广播机制 (B, N, H, W)*(B, 1, H, W)
        return out  # 输出形状 (B, N, H, W)>(4,16,32,32)

if __name__ == '__main__':
    batch_size = 4
    channels = 16
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, channels, height, width)
    cbam = CBAM(channels)
    output = cbam(input_tensor)#(B,N,H,W)>(4,16,32,32)
    summary(cbam, input_size=(channels, height, width))
    print()