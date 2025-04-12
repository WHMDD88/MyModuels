import torch
import torch.nn as nn
from torchsummary import summary

class GAM(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(GAM, self).__init__()

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, N, H, W) -> (B, N, 1, 1)
            nn.Flatten(),  # 将 (B, N, 1, 1) 展平为 (B, N)
            nn.Linear(in_channels, in_channels // rate),  # (B, N) -> (B, N // rate)
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // rate, in_channels),  # (B, N // rate) -> (B, N)
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // rate, kernel_size=1),  # (B, N, H, W) -> (B, N // rate, H, W)
            nn.BatchNorm2d(in_channels // rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // rate, in_channels // rate, kernel_size=3, padding=1),  # (B, N // rate, H, W) -> (B, N // rate, H, W)
            nn.BatchNorm2d(in_channels // rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // rate, 1, kernel_size=1),  # (B, N // rate, H, W) -> (B, 1, H, W)
            nn.Sigmoid()
        )

    def forward(self, x):
        b, n, h, w = x.size()  # x: (B, N, H, W)
        # 通道注意力
        channel_weights = self.channel_attention(x).view(b, n, 1, 1)  # (B, N) -> (B, N, 1, 1)
        channel_out = x * channel_weights  # (B, N, H, W)
        # 空间注意力
        spatial_weights = self.spatial_attention(x)  # (B, N, H, W) -> (B, 1, H, W)
        spatial_out = x * spatial_weights  # (B, N, H, W)
        # 特征融合
        out = channel_out + spatial_out  # (B, N, H, W)
        return out

if __name__ == "__main__":
    batch_size = 4
    channels = 16
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, channels, height, width)
    gam = GAM(in_channels=channels)
    summary(gam, input_size=(channels, height, width))
    output = gam(input_tensor)
    print("输入形状:", input_tensor.shape)
    print("输出形状:", output.shape)