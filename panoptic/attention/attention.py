import torch
import torch.nn.functional as F
import torch.nn as nn

from panoptic.attention.grid import Assemble, CombineScales


class EncoderBlock(nn.Module):
    """
    Encoder block for Semantic Attention Module
    """
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class DecoderBlock(nn.Module):
    """
    Decoder Block for Semantic Attention Module
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.decode(x)


class SemanticModule(nn.Module):
    """
    Semantic attention module
    """
    def __init__(self, in_channels):
        super().__init__()
        self.chanel_in = in_channels

        self.enc1 = EncoderBlock(in_channels, in_channels * 2)
        self.enc2 = EncoderBlock(in_channels * 2, in_channels * 4)
        self.dec2 = DecoderBlock(in_channels * 4, in_channels * 2)
        self.dec1 = DecoderBlock(in_channels * 2, in_channels)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)

        dec2 = self.dec2(enc2)
        dec1 = self.dec1(F.interpolate(dec2, enc1.size()[2:], mode='bilinear'))

        dec1 = F.interpolate(dec1, x.size()[2:], mode='bilinear')

        return enc2.view(-1), dec1


class PAM_Module(nn.Module):
    """ Position attention module"""
    def __init__(self, in_channels):
        super().__init__()

        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.align = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.align(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.align = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_query, proj_key)
        energy = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.align(energy)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class AttentionLayer(nn.Module):
    """
    Helper Function for PAM and CAM attention

    Parameters:
    ----------
    input:
        in_channels : input channels
        use_pam : Boolean value whether to use PAM_Module or CAM_Module
    output:
        returns the attention map
    """
    def __init__(self, in_channels, AttentionModule=PAM_Module):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            AttentionModule(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.attn(x)


class DualAttentionHead(nn.Module):
    """
    """
    def __init__(self, channels):
        super().__init__()
        self.pam = AttentionLayer(channels, PAM_Module)
        self.cam = AttentionLayer(channels, CAM_Module)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_semantic = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)

    def forward(self, fused, semantic):
        return self.conv(
            (
                self.pam(fused) +
                self.cam(fused)
            ) *
            self.conv_semantic(semantic)
        )


class PositionAttentionHead(nn.Module):
    """
    """
    def __init__(self, channels):
        super().__init__()
        self.pam = AttentionLayer(channels, PAM_Module)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_semantic = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)

    def forward(self, fused, semantic):
        return self.conv(
            (
                self.pam(fused)
            ) *
            self.conv_semantic(semantic)
        )


class ChannelAttentionHead(nn.Module):
    """
    """
    def __init__(self, channels):
        super().__init__()
        self.cam = AttentionLayer(channels, CAM_Module)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_semantic = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)

    def forward(self, fused, semantic):
        return self.conv(
            (
                self.cam(fused)
            ) *
            self.conv_semantic(semantic)
        )


class StandardAttention(nn.Module):
    """
    """
    def __init__(self, channels, disassembles=0, scale_factor=2, semantic_module1=None, attention_head=None, **kwargs):
        super().__init__()
        self.semantic_attention1 = SemanticAttentionBody(channels, disassembles, scale_factor, semantic_module1, attention_head)
        self.reassemble = Assemble(-disassembles, scale_factor)

    def forward(self, x, fused):
        attention, _, _, _ = self.semantic_attention1(x, fused)
        attention = self.reassemble(attention)
        return attention, None


class GuidedAttention(nn.Module):
    """
    """
    def __init__(self, channels, disassembles=0, scale_factor=2, semantic_module1=None, semantic_module2=None, attention_head=None):
        super().__init__()
        self.semantic_attention1 = SemanticAttentionBody(channels, disassembles, scale_factor, semantic_module1, attention_head)
        self.semantic_attention2 = SemanticAttentionBody(channels, disassembles, scale_factor, semantic_module2, attention_head)
        self.reassemble = Assemble(-disassembles, scale_factor)

    def forward(self, x, fused):
        attention, semv1, semo1, comb1 = self.semantic_attention1(x, fused)
        refined_attention, semv2, semo2, comb2 = self.semantic_attention2(x, fused, att=attention)
        refined_attention = self.reassemble(refined_attention)
        return refined_attention, {
            'in_semantic_vectors': semv1,
            'out_semantic_vectors': semv2,
            'in_attention_encodings': torch.cat([comb1, comb2], dim=1),
            'out_attention_encodings': torch.cat([semo1, semo2], dim=1),
        }


class SemanticAttentionBody(nn.Module):
    """
    """
    def __init__(self, channels, disassembles=0, scale_factor=2, semantic_module=None, attention_head=None):
        super().__init__()
        self.combine = CombineScales(disassembles, scale_factor)
        if semantic_module is None:
            semantic_module = SemanticModule(channels * 2)
        self.get_semantic = semantic_module
        if attention_head is None:
            attention_head = DualAttentionHead
        self.attention_head = attention_head(channels)

    def forward(self, x, fused, att=None):
        combined = self.combine(x, fused, att=att)
        semantic_vector, semantic_output = self.get_semantic(combined)
        attention = self.attention_head(combined, semantic_output)
        return attention, semantic_vector, semantic_output, combined


class MultiConv(nn.Module):
    """
    Helper function for combining multiple convolutional layers to refine input.

    Parameters:
    ----------
    inputs:
        in_channels : input channels
        out_channels : output channels
        attn : Boolean value whether to use Softmax or PReLU
    outputs:
        returns the refined convolution tensor
    """
    def __init__(self, in_channels, out_channels, attn=True):
        super().__init__()

        self.fuse_attn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.Softmax2d() if attn else nn.PReLU()
        )

    def forward(self, x):
        return self.fuse_attn(x)
