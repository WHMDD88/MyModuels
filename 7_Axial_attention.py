import torch
import torch.nn as nn
from torchsummary import summary


class AxialAttention(nn.Module):
    def __init__(self, in_channels, num_heads):
        super(AxialAttention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_channels = in_channels // num_heads

        # 用于生成查询、键和值的线性变换
        self.qkv_proj = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # x 形状: (B, N, H, W)
        B, N, H, W = x.size()

        # 生成查询、键和值
        qkv = self.qkv_proj(x)# qkv 形状: (B, 3*N, H, W)
        # 分割查询、键和值
        q, k, v = qkv.chunk(3, dim=1)# q, k, v 形状: (B, N, H, W)

        # 调整形状以进行多头注意力计算
        # q, k, v 形状: (B, num_heads, head_channels, H, W)
        q = q.view(B, self.num_heads, self.head_channels, H, W)
        k = k.view(B, self.num_heads, self.head_channels, H, W)
        v = v.view(B, self.num_heads, self.head_channels, H, W)

        # 行轴向注意力计算
        # q_row, k_row, v_row 形状: (B, num_heads, head_channels, H, W)
        q_row = q.permute(0, 1, 3, 4, 2).contiguous().view(B * self.num_heads * H, W, self.head_channels)
        k_row = k.permute(0, 1, 3, 2, 4).contiguous().view(B * self.num_heads * H, self.head_channels, W)
        v_row = v.permute(0, 1, 3, 4, 2).contiguous().view(B * self.num_heads * H, W, self.head_channels)

        attn_scores_row = torch.bmm(q_row, k_row) / (self.head_channels ** 0.5) # attn_scores_row 形状: (B * num_heads * H, W, W)
        attn_probs_row = torch.softmax(attn_scores_row, dim=-1)# attn_probs_row 形状: (B * num_heads * H, W, W)
        out_row = torch.bmm(attn_probs_row, v_row)# out_row 形状: (B * num_heads * H, W, head_channels)
        out_row = out_row.view(B, self.num_heads, H, W, self.head_channels).permute(0, 1, 4, 2, 3).contiguous()# out_row 形状: (B, num_heads, head_channels, H, W)

        # 列轴向注意力计算
        # q_col, k_col, v_col 形状: (B, num_heads, head_channels, H, W)
        q_col = q.permute(0, 1, 4, 3, 2).contiguous().view(B * self.num_heads * W, H, self.head_channels)
        k_col = k.permute(0, 1, 4, 2, 3).contiguous().view(B * self.num_heads * W, self.head_channels, H)
        v_col = v.permute(0, 1, 4, 3, 2).contiguous().view(B * self.num_heads * W, H, self.head_channels)
        attn_scores_col = torch.bmm(q_col, k_col) / (self.head_channels ** 0.5)# attn_scores_col 形状: (B * num_heads * W, H, H)
        attn_probs_col = torch.softmax(attn_scores_col, dim=-1)# attn_probs_col 形状: (B * num_heads * W, H, H)
        out_col = torch.bmm(attn_probs_col, v_col)# out_col 形状: (B * num_heads * W, H, head_channels)
        out_col = out_col.view(B, self.num_heads, W, H, self.head_channels).permute(0, 1, 4, 3, 2).contiguous()# out_col 形状: (B, num_heads, head_channels, H, W)

        # 融合行和列轴向的注意力输出
        out = out_row + out_col# out 形状: (B, num_heads, head_channels, H, W)
        out = out.view(B, N, H, W)# out 形状: (B, N, H, W)

        # 通过输出投影层
        output = self.out_proj(out)# output 形状: (B, N, H, W)
        return output

if __name__ == "__main__":
    batch_size = 4
    channels = 16
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, channels, height, width)
    axial_attention_layer = AxialAttention(in_channels=channels, num_heads=8)
    summary(axial_attention_layer,input_size=(channels,height,width))