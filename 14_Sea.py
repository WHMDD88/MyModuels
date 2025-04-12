import torch
import torch.nn as nn
from torchsummary import summary


class SeaModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SeaModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x 形状: (B, N, H, W)
        B, N, H, W = x.size()

        # 挤压（Squeeze）：全局平均池化
        y = self.avg_pool(x) # y 形状: (B, N, 1, 1)
        y = y.view(B, N)# y 形状: (B, N)

        # 激励（Excitation）：全连接网络
        y = self.fc(y) # y 形状: (B, N)
        y = y.view(B, N, 1, 1) # y 形状: (B, N, 1, 1)

        # 缩放（Scale）：逐元素相乘
        out = x * y.expand_as(x)# out 形状: (B, N, H, W)
        return out

# 示例使用
if __name__ == "__main__":
    batch_size = 4
    channels = 16
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, channels, height, width)
    sea_module = SeaModule(in_channels=channels)
    summary(sea_module,input_size=(channels,height,width))
    output = sea_module(input_tensor)
    print("输入形状:", input_tensor.shape)
    print("输出形状:", output.shape)