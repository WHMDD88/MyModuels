import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class EMSA(nn.Module):
    def __init__(self, in_channels, num_heads=8, window_size=7):
        super(EMSA, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_channels = in_channels // num_heads
        self.window_size = window_size

        # 生成Q、K、V的线性层
        self.qkv_proj = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.attn_drop = nn.Dropout(0.1)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # x 形状: (B, N, H, W)
        B, N, H, W = x.size()

        # 填充特征图，使其高度和宽度能被窗口大小整除
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        _, _, H_pad, W_pad = x.size()

        # 生成Q、K、V
        qkv = self.qkv_proj(x) # qkv 形状: (B, 3*N, H_pad, W_pad)
        q, k, v = qkv.chunk(3, dim=1) # q, k, v 形状: (B, N, H_pad, W_pad)

        # 划分局部窗口
        q_windows = self.window_partition(q, self.window_size)  # (B*num_windows, N, window_size, window_size)
        k_windows = self.window_partition(k, self.window_size)  # (B*num_windows, N, window_size, window_size)
        v_windows = self.window_partition(v, self.window_size)  # (B*num_windows, N, window_size, window_size)

        # 调整形状以进行多头注意力计算
        # q_windows 形状: (B*num_windows, num_heads, head_channels, window_size*window_size)
        q_windows = q_windows.view(-1, self.num_heads, self.head_channels, self.window_size * self.window_size)
        # k_windows 形状: (B*num_windows, num_heads, head_channels, window_size*window_size)
        k_windows = k_windows.view(-1, self.num_heads, self.head_channels, self.window_size * self.window_size)
        # v_windows 形状: (B*num_windows, num_heads, head_channels, window_size*window_size)
        v_windows = v_windows.view(-1, self.num_heads, self.head_channels, self.window_size * self.window_size)

        # 计算注意力分数
        # attn_scores 形状: (B*num_windows, num_heads, window_size*window_size, window_size*window_size)
        attn_scores = torch.matmul(q_windows.transpose(-2, -1), k_windows) / (self.head_channels ** 0.5)
        # 应用softmax函数得到注意力权重
        # attn_probs 形状: (B*num_windows, num_heads, window_size*window_size, window_size*window_size)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        # 应用注意力权重到值上
        # attn_out_windows 形状: (B*num_windows, num_heads, head_channels, window_size*window_size)
        attn_out_windows = torch.matmul(v_windows, attn_probs.transpose(-2, -1))
        # attn_out_windows 形状: (B*num_windows, N, window_size, window_size)
        attn_out_windows = attn_out_windows.view(-1, N, self.window_size, self.window_size)

        # 窗口拼接
        # attn_out 形状: (B, N, H_pad, W_pad)
        attn_out = self.window_reverse(attn_out_windows, self.window_size, H_pad, W_pad)

        # 去除填充
        attn_out = attn_out[:, :, :H, :W]

        # 通过输出投影层
        output = self.out_proj(attn_out)# output 形状: (B, N, H, W)
        return output

    def window_partition(self, x, window_size):
        """
        将输入特征图划分为多个不重叠的局部窗口
        :param x: 输入特征图，形状为 (B, N, H, W)
        :param window_size: 窗口大小
        :return: 划分后的窗口，形状为 (B*num_windows, N, window_size, window_size)
        """
        B, N, H, W = x.shape
        x = x.view(B, N, H // window_size, window_size, W // window_size, window_size)
        windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, N, window_size, window_size)
        return windows

    def window_reverse(self, windows, window_size, H, W):
        """
        将划分后的窗口拼接成原始的特征图
        :param windows: 划分后的窗口，形状为 (B*num_windows, N, window_size, window_size)
        :param window_size: 窗口大小
        :param H: 原始特征图的高度
        :param W: 原始特征图的宽度
        :return: 拼接后的特征图，形状为 (B, N, H, W)
        """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, -1, window_size, window_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, -1, H, W)
        return x


if __name__ == "__main__":
    batch_size = 4
    channels = 16
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, channels, height, width)
    emsa_module = EMSA(in_channels=channels)
    summary(emsa_module,input_size=(channels,height,width))
    output = emsa_module(input_tensor)
    print("输入形状:", input_tensor.shape)
    print("输出形状:", output.shape)