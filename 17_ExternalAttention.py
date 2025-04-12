import torch
import torch.nn as nn
from torchsummary import summary

class ExternalAttention(nn.Module):
    def __init__(self, in_channels, S=64):
        super().__init__()
        self.mk = nn.Linear(in_channels, S, bias=False)
        self.mv = nn.Linear(S, in_channels, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: 输入特征图，形状为 (B, N, H, W)
        B, N, H, W = x.shape
        # 将特征图展平为 (B, N, H*W)
        x = x.view(B, N, -1)  # (B, N, H*W)
        # 线性变换得到注意力权重
        attn = self.mk(x.transpose(1, 2))  # (B, H*W, S)
        attn = self.softmax(attn)  # (B, H*W, S)
        # 特征聚合
        out = self.mv(attn)  # (B, H*W, N)
        out = out.transpose(1, 2)  # (B, N, H*W)
        # 恢复特征图形状
        out = out.view(B, N, H, W)  # (B, N, H, W)
        return out
if __name__ == '__main__':
    batch_size = 4
    channels = 16
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, channels, height, width)
    ea = ExternalAttention(in_channels=channels)
    summary(ea, input_size=(channels, height, width))
    output = ea(input_tensor)
    print("输入形状:", input_tensor.shape)
    print("输出形状:", output.shape)