import torch
import torch.nn as nn
from typing import Tuple
from torchsummary import summary

class DilateForme(nn.Module):
    def __init__(self, in_channels: int, num_heads: int = 8, dilation_rate: int = 2):
        super(DilateForme, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_channels = in_channels // num_heads
        self.dilation_rate = dilation_rate

        # 空洞卷积层
        self.dilated_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=dilation_rate,
                                      dilation=dilation_rate)

        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, in_channels, 1, 1))

        # 多头自注意力机制
        self.qkv_proj = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.attn_drop = nn.Dropout(0.1)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1),
            nn.Dropout(0.1)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 形状: (B, N, H, W)
        B, N, H, W = x.size()

        # 空洞卷积特征提取
        dilated_out = self.dilated_conv(x)  # dilated_out 形状: (B, N, H, W)

        # 添加位置编码
        pos_encoded = dilated_out + self.pos_encoding  # pos_encoded 形状: (B, N, H, W)

        # 多头自注意力计算
        qkv = self.qkv_proj(pos_encoded)  # qkv 形状: (B, 3*N, H, W)
        q, k, v = qkv.chunk(3, dim=1)  # q, k, v 形状: (B, N, H, W)

        # 调整形状以进行多头注意力计算
        # q, k, v 形状: (B, num_heads, head_channels, H, W)
        q = q.reshape(B, self.num_heads, self.head_channels, H, W)
        k = k.reshape(B, self.num_heads, self.head_channels, H, W)
        v = v.reshape(B, self.num_heads, self.head_channels, H, W)

        # 计算注意力分数
        # q 形状: (B * num_heads, head_channels, H * W)
        q = q.reshape(B * self.num_heads, self.head_channels, H * W).transpose(1,
                                                                               2)  # (B * num_heads, H * W, head_channels)
        # k 形状: (B * num_heads, head_channels, H * W)
        k = k.reshape(B * self.num_heads, self.head_channels, H * W)

        # 使用 torch.bmm 函数对查询（Query，即 q）和键（Key，即 k）进行批量矩阵乘法
        attn_scores = torch.bmm(q, k) / (self.head_channels ** 0.5)  # attn_scores 形状: (B * num_heads, H * W, H * W)
        # 应用 softmax 函数得到注意力权重
        # attn_probs 形状: (B * num_heads, H * W, H * W)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        # 应用注意力权重到值上
        # v 形状: (B * num_heads, head_channels, H * W)
        v = v.reshape(B * self.num_heads, self.head_channels, H * W)
        # attn_out 形状: (B * num_heads, H * W, head_channels)
        attn_out = torch.bmm(attn_probs, v.transpose(1, 2))
        # attn_out 形状: (B, num_heads, head_channels, H, W)
        attn_out = attn_out.transpose(1, 2).reshape(B, self.num_heads, self.head_channels, H, W)
        # attn_out 形状: (B, N, H, W)
        attn_out = attn_out.reshape(B, N, H, W)

        # 通过输出投影层
        # attn_output 形状: (B, N, H, W)
        attn_output = self.out_proj(attn_out)

        # 残差连接和层归一化
        x1 = x + attn_output  # x1 形状: (B, N, H, W)
        # 调整形状以进行层归一化
        # contiguous 函数用于确保张量在内存中是连续存储的
        x1 = x1.permute(0, 2, 3, 1).contiguous()  # x1 形状: (B, H, W, N)
        x1 = self.norm1(x1)  # x1 形状: (B, H, W, N)
        x1 = x1.permute(0, 3, 1, 2).contiguous()  # x1 形状: (B, N, H, W)

        # 前馈网络
        # ff_out 形状: (B, N, H, W)
        ff_out = self.feed_forward(x1)

        # 残差连接和层归一化
        output = x1 + ff_out# output 形状: (B, N, H, W)
        # 调整形状以进行层归一化
        output = output.permute(0, 2, 3, 1).contiguous()# output 形状: (B, H, W, N)
        output = self.norm2(output)# output 形状: (B, H, W, N)
        output = output.permute(0, 3, 1, 2).contiguous()# output 形状: (B, N, H, W)
        return output

if __name__ == "__main__":
    batch_size = 4
    channels = 16
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, channels, height, width)
    dilateforme_layer = DilateForme(in_channels=channels)
    output = dilateforme_layer(input_tensor)
    summary(dilateforme_layer, input_size=(channels, height, width))
