import torch
import torch.nn as nn
from torchsummary import summary

class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x 形状: (B, N, H, W)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # 经过全局平均池化并调整形状后，y 形状: (B, N)
        y = self.fc(y).view(b, c, 1, 1) # 经过全连接层和调整形状后，y 形状: (B, N, 1, 1)
        # 将输入特征图与通道注意力权重逐元素相乘，以增强重要通道的特征响
        out=x * y.expand_as(x) # 输出形状: (B, N, H, W)
        return out

if __name__ == '__main__':
    batch_size = 4
    channels = 16
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, channels, height, width)
    SE_Net = SELayer(channels)
    output = SE_Net(input_tensor)#(B,N,H,W)>(4,16,32,32)
    summary(SE_Net, input_size=(channels, height, width))