import torch
import torch.nn as nn
from torchsummary import summary


class EMA(nn.Module):
    def __init__(self, in_channels, decay=0.999):
        super(EMA, self).__init__()
        self.in_channels = in_channels
        self.decay = decay  # 衰减因子，越接近1则历史平均值影响越大
        self.register_buffer('ema_state', torch.zeros(1, in_channels, 1, 1))  # 初始化EMA状态（全局平均，可扩展为空间相关）

    def forward(self, x):
        # 输入特征图形状: (B, N, H, W)
        B, N, H, W = x.size()

        # 计算当前输入的全局平均值（若需空间独立EMA，可跳过全局平均，直接处理每个空间位置）
        # 当前输入的全局平均: (B, N, 1, 1)
        x_mean = x.mean(dim=(2, 3), keepdim=True)  # 对H和W维度求平均

        # 指数移动平均更新（首次调用时ema_state为全零）
        # 公式: ema_state = decay * ema_state + (1 - decay) * x_mean
        # 注意：这里假设ema_state是全局的（即不区分空间位置），若需要每个空间位置独立EMA，需调整ema_state形状为(1, N, H, W)
        if not self.training:
            # 推理时直接使用当前EMA状态，不更新
            ema_output = self.ema_state.repeat(B, 1, H, W)  # 扩展为输入形状
        else:
            # 训练时更新EMA状态
            self.ema_state = self.decay * self.ema_state + (1 - self.decay) * x_mean.mean(dim=0,keepdim=True)  # 对批次维度求平均，保持(B=1, N, 1, 1)
            ema_output = self.ema_state.repeat(B, 1, H, W)  # 扩展为输入形状
        # 输出形状: (B, N, H, W)
        return ema_output


if __name__ == "__main__":
    batch_size = 4
    channels = 16
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, channels, height, width)
    ema_module = EMA(in_channels=channels)
    # 训练模式：更新EMA状态
    ema_module.train()
    output_train = ema_module(input_tensor)
    print("训练模式输出形状:", output_train.shape)