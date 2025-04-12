import torch
import torch.nn as nn
from torchsummary import summary

class PSA(nn.Module):
    def __init__(self, in_channels, num_groups=4):
        super(PSA, self).__init__()
        self.num_groups = num_groups
        self.group_channels = in_channels // num_groups

        # 用于计算注意力权重的卷积层
        self.conv_attention = nn.Conv2d(in_channels, num_groups, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, n, h, w = x.size()  # x: (B, N, H, W)

        # 特征分组
        grouped_x = x.view(b, self.num_groups, self.group_channels, h, w)  # (B, num_groups, group_channels, H, W)

        # 计算注意力权重
        attention_weights = self.conv_attention(x)  # (B, num_groups, H, W)
        attention_weights = self.softmax(attention_weights.view(b, self.num_groups, -1)).view(b, self.num_groups, h, w)  # (B, num_groups, H, W)

        # 注意力加权
        attention_weights = attention_weights.unsqueeze(2)  # (B, num_groups, 1, H, W)
        weighted_x = grouped_x * attention_weights  # (B, num_groups, group_channels, H, W)

        # 特征融合
        output = weighted_x.view(b, n, h, w)  # (B, N, H, W)
        return output

# 测试代码
if __name__ == "__main__":
    batch_size = 4
    channels = 16
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, channels, height, width)
    psa = PSA(in_channels=channels)
    summary(psa, input_size=(channels, height, width))
    output = psa(input_tensor)
    print("输入形状:", input_tensor.shape)
    print("输出形状:", output.shape)