import torch
import torch.nn as nn
from torchsummary import summary


class ViP(nn.Module):
    def __init__(self, in_channels, num_groups=4):
        super(ViP, self).__init__()
        self.num_groups = num_groups
        self.group_channels = in_channels // num_groups
        self.linear = nn.Linear(self.group_channels, self.group_channels)

    def forward(self, x):
        B, N, H, W = x.size()  # x: (B, N, H, W)
        # 特征划分
        grouped_x = x.view(B, self.num_groups, self.group_channels, H, W)  # (B, num_groups, group_channels, H, W)
        # 排列操作及线性变换
        outputs = []
        for i in range(self.num_groups):
            group = grouped_x[:, i]  # (B, group_channels, H, W)

            # 高度维度排列
            permuted_h = group.permute(0, 1, 3, 2).contiguous().view(B, self.group_channels, -1).transpose(1, 2)
            # (B, H * W, group_channels)
            output_h = self.linear(permuted_h).transpose(1, 2).view(B, self.group_channels, W, H).permute(0, 1, 3, 2)
            # (B, group_channels, H, W)

            # 宽度维度排列
            permuted_w = group.view(B, self.group_channels, -1).transpose(1, 2)
            # (B, H * W, group_channels)
            output_w = self.linear(permuted_w).transpose(1, 2).view(B, self.group_channels, H, W)
            # (B, group_channels, H, W)

            # 通道维度排列
            permuted_c = group.transpose(1, 2).contiguous().view(B, -1, self.group_channels)
            # (B, H * W, group_channels)
            output_c = self.linear(permuted_c).transpose(1, 2).view(B, self.group_channels, H, W)
            # (B, group_channels, H, W)

            # 融合排列结果
            output = output_h + output_w + output_c  # (B, group_channels, H, W)
            outputs.append(output)

        # 特征融合
        out = torch.cat(outputs, dim=1)  # (B, N, H, W)
        return out

if __name__ == "__main__":
    batch_size = 4
    channels = 16
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, channels, height, width)
    vip = ViP(in_channels=channels)
    summary(vip, input_size=(channels, height, width))
    output = vip(input_tensor)
    print("输入形状:", input_tensor.shape)
    print("输出形状:", output.shape)