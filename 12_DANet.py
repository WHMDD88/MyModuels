import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class PAM_Module(nn.Module):
    """ Position attention module"""
    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B, N, H, W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        B, N, H, W = x.size()
        # proj_query 形状: (B, N//8, H, W)
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        # proj_key 形状: (B, N//8, H, W)
        proj_key = self.key_conv(x).view(B, -1, H * W)
        # energy 形状: (B, H*W, H*W)
        energy = torch.bmm(proj_query, proj_key)
        # attention 形状: (B, H*W, H*W)
        attention = self.softmax(energy)
        # proj_value 形状: (B, N, H, W)
        proj_value = self.value_conv(x).view(B, -1, H * W)

        # out 形状: (B, N, H, W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, N, H, W)

        out = self.gamma * out + x
        return out

class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B, N, H, W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        B, N, H, W = x.size()
        proj_query = x.view(B, N, -1)# proj_query 形状: (B, N, H*W)
        proj_key = x.view(B, N, -1).permute(0, 2, 1)# proj_key 形状: (B, N, H*W)
        # energy 形状: (B, N, N)
        energy = torch.bmm(proj_query, proj_key)
        # energy_new 形状: (B, N, N)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        # attention 形状: (B, N, N)
        attention = self.softmax(energy_new)
        # proj_value 形状: (B, N, H*W)
        proj_value = x.view(B, N, -1)

        # out 形状: (B, N, H, W)
        out = torch.bmm(attention, proj_value)
        out = out.view(B, N, H, W)

        out = self.gamma * out + x
        return out

class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        # x 形状: (B, N, H, W)
        B, N, H, W = x.size()
        # feat1 形状: (B, N//4, H, W)
        feat1 = self.conv5a(x)
        # sa_feat 形状: (B, N//4, H, W)
        sa_feat = self.sa(feat1)
        # sa_conv 形状: (B, N//4, H, W)
        sa_conv = self.conv51(sa_feat)
        # sa_output 形状: (B, out_channels, H, W)
        sa_output = self.conv6(sa_conv)

        # feat2 形状: (B, N//4, H, W)
        feat2 = self.conv5c(x)
        # sc_feat 形状: (B, N//4, H, W)
        sc_feat = self.sc(feat2)
        # sc_conv 形状: (B, N//4, H, W)
        sc_conv = self.conv52(sc_feat)
        # sc_output 形状: (B, out_channels, H, W)
        sc_output = self.conv7(sc_conv)

        # feat_sum 形状: (B, N//4, H, W)
        feat_sum = sa_conv + sc_conv
        # sasc_output 形状: (B, out_channels, H, W)
        sasc_output = self.conv8(feat_sum)

        #sa_output（Self-Attention Output，自注意力输出）
        #sc_output（Spatial Context Output，空间上下文输出）
        #sasc_output（Self-Attention-Spatial Context Output，自注意力 - 空间上下文融合输出）
        outputs = [sasc_output, sa_output, sc_output]
        return tuple(outputs)


if __name__ == "__main__":
    # 输入特征图，假设批次大小为4，通道数为2048，高度和宽度为32
    batch=4
    in_channels=2048
    out_channels=19
    height=32
    width=32
    input_tensor = torch.randn(batch,in_channels,height,width)
    danet_head = DANetHead(in_channels=in_channels, out_channels=out_channels)
    summary(danet_head,input_size=(in_channels,height,width))
    outputs = danet_head(input_tensor)
    for output in outputs:
        print("输出形状:", output.shape)

