import torch
import torch.nn as nn
from torchsummary import summary


class ACmix(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ACmix, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # 卷积部分
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        # 自注意力部分
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, 1, bias=bias)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.proj_drop = nn.Dropout(0.1)

        # 融合权重
        self.fusion_weight = nn.Parameter(torch.ones(2))

    def forward(self, x):
        # x 形状: (B, N, H, W)
        B, N, H, W = x.size()
        conv_out = self.conv(x) # conv_out 形状: (B, N, H, W)
        qkv = self.qkv(x) # qkv 形状: (B, 3*N, H, W)

        # 分割出 query, key, value
        q, k, v = qkv.chunk(3, dim=1)  # q, k, v 形状: (B, N, H, W)

        # 计算注意力分数
        # attn 形状: (B, N, H*W, H*W)
        attn = (q.flatten(2) @ k.flatten(2).transpose(-2, -1)) / (N ** 0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 应用注意力
        # out 形状: (B, N, H*W)
        out = (attn @ v.flatten(2)).reshape(B, N, H, W)
        # attn_out 形状: (B, N, H, W)
        attn_out = self.proj(out)
        attn_out = self.proj_drop(attn_out)

        # 特征融合
        # 计算融合权重
        w1, w2 = self.fusion_weight.softmax(dim=0)
        # output 形状: (B, N, H, W)
        output = w1 * conv_out + w2 * attn_out
        return output

if __name__ == '__main__':
    # 定义输入参数
    batch_size = 4
    in_channels = 64
    height = 32
    width = 32
    out_channels = 64
    """很难直接确定最佳的 kernel_size,
    通常需要通过实验和调优来找到最适合当前任务和数据集的卷积核大小。
    可以尝试不同的 kernel_size,
    并比较模型在验证集上的性能，选择性能最好的 kernel_size。"""
    kernel_size = 3
    padding = 1
    input_tensor = torch.randn(batch_size, in_channels, height, width)
    acmix_layer = ACmix(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
    # 进行前向传播
    output = acmix_layer(input_tensor)
    summary(acmix_layer, input_size=(in_channels,height, width))
