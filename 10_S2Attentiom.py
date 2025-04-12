import torch
import torch.nn as nn
from  torchsummary import summary

class S2Attention(nn.Module):
    def __init__(self, in_channels, num_heads=8, shift_size=1):
        super(S2Attention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.head_channels = in_channels // num_heads

        # 用于生成查询、键和值的线性变换
        self.qkv_proj = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # x 形状: (B, N, H, W)
        B, N, H, W = x.size()

        qkv = self.qkv_proj(x)  # qkv 形状: (B, 3*N, H, W)
        q, k, v = qkv.chunk(3, dim=1)  # q, k, v 形状: (B, N, H, W)

        # 调整形状以进行多头注意力计算
        # q, k, v 形状: (B, num_heads, head_channels, H, W)
        q = q.view(B, self.num_heads, self.head_channels, H, W)
        k = k.view(B, self.num_heads, self.head_channels, H, W)
        v = v.view(B, self.num_heads, self.head_channels, H, W)

        # 移位操作
        if self.shift_size > 0:
            # 对查询、键和值进行移位
            # q_shift 形状: (B, num_heads, head_channels, H, W)
            q_shift = torch.roll(q, shifts=(self.shift_size, self.shift_size), dims=(-2, -1))
            # k_shift 形状: (B, num_heads, head_channels, H, W)
            k_shift = torch.roll(k, shifts=(self.shift_size, self.shift_size), dims=(-2, -1))
            # v_shift 形状: (B, num_heads, head_channels, H, W)
            v_shift = torch.roll(v, shifts=(self.shift_size, self.shift_size), dims=(-2, -1))
        else:
            q_shift = q
            k_shift = k
            v_shift = v

        # 计算注意力分数
        # 调整形状以进行矩阵乘法
        # q_shift 形状: (B * num_heads, head_channels, H * W)
        q_shift = q_shift.view(B * self.num_heads, self.head_channels, H * W).transpose(1,2)
        # k_shift 形状: (B * num_heads, head_channels, H * W)
        k_shift = k_shift.view(B * self.num_heads, self.head_channels,H * W)
        # attn_scores 形状: (B * num_heads, H * W, H * W)
        attn_scores = torch.bmm(q_shift, k_shift) / (self.head_channels ** 0.5)
        # 应用 softmax 函数得到注意力权重
        attn_probs = torch.softmax(attn_scores, dim=-1)  # attn_probs 形状: (B * num_heads, H * W, H * W)

        # 应用注意力权重到值上
        # v_shift 形状: (B * num_heads, head_channels, H * W)
        v_shift = v_shift.view(B * self.num_heads, self.head_channels,H * W)
        out = torch.bmm(attn_probs, v_shift.transpose(1, 2))  # out 形状: (B * num_heads, H * W, head_channels)
        # out 形状: (B, num_heads, head_channels, H, W)
        out = out.transpose(1, 2).view(B, self.num_heads, self.head_channels, H,W)
        # 合并多头注意力结果
        # out 形状: (B, N, H, W)
        out = out.reshape(B, N, H, W)
        output = self.out_proj(out)  # output 形状: (B, N, H, W)
        return output


# 示例使用
if __name__ == "__main__":
    batch_size = 4
    channels = 16
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, channels, height, width)
    s2_attn_layer = S2Attention(in_channels=channels)
    output = s2_attn_layer(input_tensor)
    summary(s2_attn_layer, input_size=(channels, height, width))

