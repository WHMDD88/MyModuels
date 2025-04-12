import torch
import torch.nn as nn
from torchsummary import summary

class EfficientAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(EfficientAttention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_channels = in_channels // num_heads

        # 定义线性层用于生成Q, K, V
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        # x: (B, N, H, W)
        b, n, h, w = x.size()
        # 将特征图展平为序列
        x = x.flatten(2).transpose(1, 2)  # x: (B, H*W, N)
        # 生成Q, K, V
        q = self.query(x).view(b, -1, self.num_heads, self.head_channels).transpose(1,
                                                                                    2)  # q: (B, num_heads, H*W, head_channels)
        k = self.key(x).view(b, -1, self.num_heads, self.head_channels).transpose(1,
                                                                                  2)  # k: (B, num_heads, H*W, head_channels)
        v = self.value(x).view(b, -1, self.num_heads, self.head_channels).transpose(1,
                                                                                    2)  # v: (B, num_heads, H*W, head_channels)
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (
                    self.head_channels ** 0.5)  # attn_scores: (B, num_heads, H*W, H*W)
        # 应用softmax函数得到注意力权重
        attn_weights = torch.softmax(attn_scores, dim=-1)  # attn_weights: (B, num_heads, H*W, H*W)
        # 计算加权和
        out = torch.matmul(attn_weights, v)  # out: (B, num_heads, H*W, head_channels)
        # 合并多头
        out = out.transpose(1, 2).contiguous().view(b, -1, self.in_channels)  # out: (B, H*W, N)
        out = self.out_proj(out)  # out: (B, H*W, N)
        # 将序列重新转换为特征图
        out = out.transpose(1, 2).view(b, n, h, w)  # out: (B, N, H, W)
        return out

if __name__ == '__main__':
    batch_size = 4
    channels = 16
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, channels, height, width)
    efficient_attn = EfficientAttention(channels)
    summary(efficient_attn, input_size=(channels, height, width))
    output = efficient_attn(input_tensor)
    print("输入形状:", input_tensor.shape)
    print("输出形状:", output.shape)
