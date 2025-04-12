import torch
import torch.nn as nn
from torchsummary import summary


class TripleAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(TripleAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        # 通道注意力
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

        # 尺度注意力
        self.scale_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.scale_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.scale_fc = nn.Sequential(
            nn.Linear(in_channels * 3, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, 3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x 形状: (B, N, H, W)
        B, N, H, W = x.size()

        # 通道注意力
        channel_avg = self.channel_avg_pool(x)# channel_avg 形状: (B, N, 1, 1)
        channel_vec = channel_avg.view(B, N)# channel_vec 形状: (B, N)
        channel_weight = self.channel_fc(channel_vec)#(B, N)
        channel_weight = channel_weight.view(B, N, 1, 1)#channel_weight 形状: (B, N, 1, 1)

        # 空间注意力
        spatial_avg = x.mean(dim=1, keepdim=True)# spatial_avg 形状: (B, 1, H, W)
        spatial_max, _ = x.max(dim=1, keepdim=True)# spatial_max 形状: (B, 1, H, W)
        spatial_concat = torch.cat([spatial_avg, spatial_max], dim=1)# 拼接平均池化和最大池化结果，spatial_concat 形状: (B, 2, H, W)
        spatial_weight = self.spatial_conv(spatial_concat)# 卷积生成空间权重，spatial_weight 形状: (B, 1, H, W)
        spatial_weight = self.spatial_sigmoid(spatial_weight)

        # 尺度注意力
        scale_out1 = self.scale_conv1(x) # 不同尺度卷积，scale_out1 形状: (B, N, H, W)
        scale_out2 = self.scale_conv2(x)# scale_out2 形状: (B, N, H, W)
        scale_avg1 = self.channel_avg_pool(scale_out1) # 全局平均池化，scale_avg1 形状: (B, N, 1, 1)
        scale_avg2 = self.channel_avg_pool(scale_out2)# scale_avg2 形状: (B, N, 1, 1)
        x_avg = self.channel_avg_pool(x)# x_avg 形状: (B, N, 1, 1)
        scale_vec = torch.cat([scale_avg1.view(B, N), scale_avg2.view(B, N), x_avg.view(B, N)], dim=1)# 展平并拼接，scale_vec 形状: (B, N * 3)
        scale_weight = self.scale_fc(scale_vec)# 全连接层生成尺度权重，scale_weight 形状: (B, 3)
        scale_weight = scale_weight.view(B, 3, 1, 1)# 调整形状，scale_weight 形状: (B, 3, 1, 1)
        # 加权融合不同尺度特征，scale_out 形状: (B, N, H, W)
        scale_out = scale_weight[:, 0:1] * x + scale_weight[:, 1:2] * scale_out1 + scale_weight[:, 2:3] * scale_out2

        # 注意力融合
        triple_weight = channel_weight * spatial_weight# 联合注意力权重，triple_weight 形状: (B, N, H, W)
        output = scale_out * triple_weight# 特征增强，output 形状: (B, N, H, W)
        return output

if __name__ == "__main__":
    batch_size = 4
    channels = 16
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, channels, height, width)
    triple_attn_layer = TripleAttention(in_channels=channels)
    output = triple_attn_layer(input_tensor)
    summary(triple_attn_layer, input_size=(channels, height, width))