import torch
import torch.nn as nn
from torchsummary import summary

class CoTAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CoTAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        # 通道注意力分支
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

        # 空间注意力分支
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入特征图形状: (B, N, H, W)
        B, N, H, W = x.size()
        # --------------------- 通道注意力计算 ---------------------
        channel_avg = self.channel_avg_pool(x)# (B, N, H, W) -> (B, N, 1, 1)
        channel_vec = channel_avg.view(B, N)# 展平为向量: (B, N, 1, 1) -> (B, N)
        channel_weight = self.channel_fc(channel_vec).view(B, N, 1, 1)
        # 通道权重形状: (B, N, 1, 1)
        # --------------------- 空间注意力计算 ---------------------
        spatial_avg = x.mean(dim=1, keepdim=True)# 通道维度平均池化: (B, N, H, W) -> (B, 1, H, W)
        spatial_max, _ = x.max(dim=1, keepdim=True)# 通道维度最大池化: (B, N, H, W) -> (B, 1, H, W)
        spatial_concat = torch.cat([spatial_avg, spatial_max], dim=1)# 拼接平均池化和最大池化结果: (B, 2, H, W)
        spatial_weight = self.spatial_conv(spatial_concat)#卷积生成空间权重: (B, 2, H, W) -> (B, 1, H, W)
        spatial_weight = self.spatial_sigmoid(spatial_weight)
        # 空间权重形状: (B, 1, H, W)
        # --------------------- 注意力融合 ---------------------
        # 联合注意力权重: (B, N, 1, 1) * (B, 1, H, W) = (B, N, H, W)
        cot_weight = channel_weight * spatial_weight
        # 特征增强: 原始特征 * 联合权重
        output = x * cot_weight # 输出形状: (B, N, H, W)
        return output

if __name__ == "__main__":
    batch_size = 4
    channels = 16
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, channels, height, width)
    cot_attn = CoTAttention(in_channels=channels)
    output = cot_attn(input_tensor)
    summary(cot_attn, input_size=(channels, height, width))