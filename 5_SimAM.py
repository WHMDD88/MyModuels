import torch
import torch.nn as nn
from torchsummary import summary

class SimAM(nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        # x 形状: (B, N, H, W)
        b, c, h, w = x.size()
        n = w * h - 1
        x_mean = x.mean(dim=[2, 3], keepdim=True)# 计算每个通道的均值，x_mean 形状: (B, N, 1, 1)
        x_minus_mu_square = (x - x_mean).pow(2)# 计算 (x - 均值)^2，x_minus_mu_square 形状: (B, N, H, W)
        sum_x_minus_mu_square = x_minus_mu_square.sum(dim=[2, 3], keepdim=True) # 计算分母部分，sum_x_minus_mu_square 形状: (B, N, 1, 1)
        y = x_minus_mu_square / (4 * (sum_x_minus_mu_square / n + self.e_lambda)) + 0.5# 计算 y 的中间结果，y 形状: (B, N, H, W)
        # 经过 Sigmoid 激活得到注意力权重，y 形状: (B, N, H, W)
        y = self.activaton(y)
        # 特征加权，输出形状: (B, N, H, W)
        out=x * y
        return out

if __name__ == '__main__':
    batch_size = 4
    channels = 16
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, channels, height, width)
    simam = SimAM(channels)
    output = simam(input_tensor)#(B,N,H,W)>(4,16,32,32)
    summary(simam, input_size=(channels, height, width))