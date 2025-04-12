import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inchannels, outchannels, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inchannels // reduction)
        self.conv1 = nn.Conv2d(inchannels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, outchannels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, outchannels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x shape: (B, N, H, W)
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)#  (B, N, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2) # (B, N, 1, W)> (B, N, W, 1)
        y = torch.cat([x_h, x_w], dim=2)#(B, N, H+W, 1)
        y = self.conv1(y)# y shape after conv1: (B, M, H+W, 1), where M = max(8, N // reduction)
        y = self.bn1(y)# (B, M, H+W, 1)
        y = self.act(y) # (B, M, H+W, 1)

        x_h, x_w = torch.split(y, [h, w], dim=2)#x_h 的形状是 (B, N, h, 1)，它是从 y 的第 2 个维度的前 h 个元素中提取出来的。
                                                                 #x_w 的形状是 (B, N, w, 1)，它是从 y 的第 2 个维度的后 w 个元素中提取出来的。
        x_w = x_w.permute(0, 1, 3, 2) # x_w shape: (B, M, 1, W)
        a_h = self.conv_h(x_h).sigmoid() # a_h shape: (B, N, H, 1)
        a_w = self.conv_w(x_w).sigmoid()# a_w shape: (B, N, 1, W)
        out = identity * a_w * a_h# out shape: (B, N, H, W)
        return out


if __name__ == "__main__":
    in_channels = 64
    out_channels = 64
    height, width = 32, 32
    model = CoordAtt(in_channels, out_channels)
    # input_tensor shape: (B, N, H, W) = (4, 64, 32, 32)
    input_tensor = torch.randn(4, in_channels, height, width)
    # 前向传播
    output = model(input_tensor)
    print("Output shape:", output.shape)
    summary(model, input_size=(in_channels, height, width))