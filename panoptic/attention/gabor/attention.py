import torch
import torch.nn as nn

from panoptic.attention.attention import DualAttentionHead
from panoptic.attention.gabor.gabor import GaborPool, IGConv


class GAMReal_Module(nn.Module):
    """
    Gabor attention module
    """
    def __init__(self, no_g):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.align = nn.Softmax(dim=-1)
        self.no_g = no_g

    def forward(self, x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X G X H X W)
            returns :
                out : attention value + input feature
        """
        m_batchsize, C, height, width = x.size()
        C = C // self.no_g
        proj_query = x.view(m_batchsize, self.no_g, -1)
        proj_key = x.view(m_batchsize, self.no_g, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_query, proj_key)
        energy = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.align(energy)
        proj_value = x.view(m_batchsize, self.no_g, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C * self.no_g, height, width)

        out = self.gamma * out + x
        return out


class AttentionLayerGabor(nn.Module):
    """
    Helper Function for real-valued gabor attention modules
    """
    def __init__(self, in_ch, no_g, gp=None):
        super().__init__()

        self.attn = nn.Sequential(
            IGConv(in_ch * 2, in_ch * no_g, kernel_size=3, padding=1, no_g=no_g),
            nn.PReLU(),
            GAMReal_Module(no_g),
            GaborPool(no_g, gp),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.PReLU()
        )

    def forward(self, x):
        return self.attn(x)


class GaborTriAttentionHead(DualAttentionHead):
    def __init__(self, channels):
        super().__init__(channels)
        self.gam = AttentionLayerGabor(channels, 4)

    def forward(self, fused, semantic):
        return self.conv(
            (
                self.pam(fused) +
                self.cam(fused) +
                self.gam(fused)
            ) *
            self.conv_semantic(semantic)
        )
