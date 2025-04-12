import torch
import torch.nn as nn
from torchsummary import summary

class PAM_Module(nn.Module):
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: 输入特征图，形状为 (B, N, H, W)
        B, N, H, W = x.size()
        # 生成查询特征图，形状为 (B, N//8, H, W)
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        # 生成键特征图，形状为 (B, N//8, H, W)
        proj_key = self.key_conv(x).view(B, -1, H * W)
        # 计算相似度矩阵，形状为 (B, H*W, H*W)
        energy = torch.bmm(proj_query, proj_key)
        # 计算注意力权重矩阵，形状为 (B, H*W, H*W)
        attention = self.softmax(energy)
        # 生成值特征图，形状为 (B, N, H, W)
        proj_value = self.value_conv(x).view(B, -1, H * W)
        # 特征聚合，形状为 (B, N, H*W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # 恢复特征图形状，形状为 (B, N, H, W)
        out = out.view(B, N, H, W)
        # 特征融合
        out = self.gamma * out + x
        return out

if __name__ == '__main__':
    batch_size = 4
    channels = 16
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, channels, height, width)
    pam = PAM_Module(in_dim=channels)
    summary(pam, input_size=(channels, height, width))
    output = pam(input_tensor)
    print("输入形状:", input_tensor.shape)
    print("输出形状:", output.shape)