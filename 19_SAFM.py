import torch
import torch.nn as nn
from torchsummary import summary

class SAFM(nn.Module):
    def __init__(self, in_channels):
        super(SAFM, self).__init__()
        # 用于下采样和上采样的卷积层
        self.downsample2 = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1)
        )
        self.downsample4 = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1)
        )
        self.downsample8 = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1)
        )
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        # 用于生成注意力图的卷积层
        self.conv_attn = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        # x: (B, N, H, W)
        b, n, h, w = x.size()
        # 通道分割
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)  # x1, x2, x3, x4: (B, N//4, H, W)
        # 下采样
        x2_down = self.downsample2(x2)  # x2_down: (B, N//4, H//2, W//2)
        x3_down = self.downsample4(x3)  # x3_down: (B, N//4, H//4, W//4)
        x4_down = self.downsample8(x4)  # x4_down: (B, N//4, H//8, W//8)
        # 上采样
        x2_up = self.upsample2(x2_down)  # x2_up: (B, N//4, H, W)
        x3_up = self.upsample4(x3_down)  # x3_up: (B, N//4, H, W)
        x4_up = self.upsample8(x4_down)  # x4_up: (B, N//4, H, W)
        # 拼接多尺度特征
        multi_scale_features = torch.cat([x1, x2_up, x3_up, x4_up], dim=1)  # multi_scale_features: (B, N, H, W)
        # 生成注意力图
        attn_map = self.conv_attn(multi_scale_features)  # attn_map: (B, N, H, W)
        attn_map = self.gelu(attn_map)  # attn_map: (B, N, H, W)
        # 特征调制
        modulated_features = x * attn_map  # modulated_features: (B, N, H, W)
        return modulated_features

if __name__ == '__main__':
    batch_size = 4
    channels = 16
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, channels, height, width)
    safm = SAFM(channels)
    summary(safm, input_size=(channels, height, width))
    output = safm(input_tensor)
    print("输入形状:", input_tensor.shape)
    print("输出形状:", output.shape)