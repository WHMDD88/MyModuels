import torch
import torch.nn as nn
import math
from torchsummary import summary


class ECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECALayer, self).__init__()
        # 根据通道数自适应计算一维卷积核的大小
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 定义一维卷积层
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x 形状: (B, N, H, W)
        y = self.avg_pool(x)#y 形状: (B, N, 1, 1)
        # squeeze 函数的作用是去除张量中维度大小为 1 的维度,-1 代表最后一个维度,使得 y 的形状变为 (B, N, 1)
        # transpose 函数用于交换张量的两个维度,-1 表示最后一个维度，-2 表示倒数第二个维度
        y = y.squeeze(-1).transpose(-1, -2)# y 形状: (B, 1, N)
        y = self.conv(y) # 一维卷积，y 形状: (B, 1, N)
        y = self.sigmoid(y)# y 形状: (B, 1, N)
        y = y.transpose(-1, -2).unsqueeze(-1) # 调整形状以与输入特征图匹配，y 形状: (B, N, 1, 1)
        out=x * y.expand_as(x) # 通道加权，输出形状: (B, N, H, W)
        return out

if __name__ == '__main__':
    batch_size = 4
    channels = 16
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, channels, height, width)
    ECA = ECALayer(channels)
    output = ECA(input_tensor)#(B,N,H,W)>(4,16,32,32)
    summary(ECA, input_size=(channels, height, width))