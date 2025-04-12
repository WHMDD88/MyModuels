import torch
import torch.nn as nn
import torch.nn.functional as F


class HiLoAttention(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.window_size = window_size

    def forward(self, x):
        B, N, C = x.shape  # x: (B, N, C)
        H = W = int(N ** 0.5)  # Assume square feature map

        # Global attention (Hi)
        # Downsample
        x_hi = F.avg_pool2d(x.permute(0, 2, 1).view(B, C, H, W), kernel_size=self.window_size,
                            stride=self.window_size).flatten(2).permute(0, 2, 1)  # (B, N_hi, C)
        N_hi = x_hi.shape[1]

        # Compute qkv for global attention
        qkv_hi = self.qkv(x_hi).reshape(B, N_hi, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv_hi: (3, B, num_heads, N_hi, C // num_heads)
        q_hi, k_hi, v_hi = qkv_hi[0], qkv_hi[1], qkv_hi[2]  # (B, num_heads, N_hi, C // num_heads)

        attn_hi = (q_hi @ k_hi.transpose(-2, -1)) * self.scale  # (B, num_heads, N_hi, N_hi)
        attn_hi = attn_hi.softmax(dim=-1)
        attn_hi = self.attn_drop(attn_hi)

        out_hi = (attn_hi @ v_hi).transpose(1, 2).reshape(B, N_hi, C)  # (B, N_hi, C)

        # Upsample
        out_hi = F.interpolate(out_hi.permute(0, 2, 1).view(B, C, int(N_hi ** 0.5), int(N_hi ** 0.5)),
                               size=(H, W), mode='bilinear', align_corners=False).flatten(2).permute(0, 2, 1)
        # (B, N, C)

        # Local attention (Lo)
        x = x.view(B, H, W, C)
        windows = x.unfold(1, self.window_size, self.window_size).unfold(2, self.window_size,
                                                                         self.window_size).reshape(B, -1,
                                                                                                   self.window_size *
                                                                                                   self.window_size,
                                                                                                   C)
        # (B, num_windows, window_size^2, C)
        num_windows = windows.shape[1]

        # Compute qkv for local attention
        qkv_lo = self.qkv(windows).reshape(B, num_windows, self.window_size * self.window_size, 3, self.num_heads,
                                           C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        # qkv_lo: (3, B, num_windows, num_heads, window_size^2, C // num_heads)
        q_lo, k_lo, v_lo = qkv_lo[0], qkv_lo[1], qkv_lo[2]  # (B, num_windows, num_heads, window_size^2, C // num_heads)

        attn_lo = (q_lo @ k_lo.transpose(-2,
                                         -1)) * self.scale  # (B, num_windows, num_heads, window_size^2, window_size^2)
        attn_lo = attn_lo.softmax(dim=-1)
        attn_lo = self.attn_drop(attn_lo)

        out_lo = (attn_lo @ v_lo).transpose(2, 3).reshape(B, num_windows, self.window_size * self.window_size, C)
        # (B, num_windows, window_size^2, C)

        # Merge windows
        out_lo = out_lo.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size,
                             C).permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C).flatten(1, 2)  # (B, N, C)

        # Fusion
        out = out_hi + out_lo  # (B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

if __name__ == "__main__":
    B = 2  # Batch size
    H = W = 32  # Height and width of the feature map
    C = 64  # Number of channels
    x = torch.randn(B, H * W, C)
    hilo_attn = HiLoAttention(dim=C)
    output = hilo_attn(x)
    print(output.shape)  # 输出: torch.Size([B, H * W, C])
