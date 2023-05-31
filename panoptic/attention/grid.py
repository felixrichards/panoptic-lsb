import torch
import torch.nn as nn
import torch.nn.functional as F


class CombineScales(nn.Module):
    """
    """
    def __init__(self, disassembles=0, factor=2):
        super().__init__()
        self.disassemble = Assemble(disassembles, factor)

    def forward(self, x, other, att=None):
        other = F.interpolate(other, size=x.size()[-2:], mode='bilinear')
        x, other = self.disassemble(x), self.disassemble(other)
        if att is not None:
            other = other * att
        return torch.cat((
            x,
            other
        ), dim=1)


class Assemble(nn.Module):
    def __init__(self, disassembles=0, factor=2):
        super().__init__()
        self.d = abs(disassembles)
        if disassembles > 0:
            self.assemble = Disassemble(factor=factor)
        elif disassembles < 0:
            self.assemble = Reassemble(factor=factor)

    def forward(self, x):
        for d in range(self.d):
            x = self.assemble(x)
        return x


class Disassemble(nn.Module):
    def __init__(self, factor=2):
        super().__init__()
        self.f = factor

    def forward(self, x):
        x, xs = compress(x)
        x = disassemble(x, self.f)
        x = recover(x, xs)
        return x


class Reassemble(nn.Module):
    def __init__(self, factor=2):
        super().__init__()
        self.f = factor

    def forward(self, x):
        x, xs = compress(x)
        if xs is not None:
            xs = [xs[0] // 4, *xs[1:]]
        x = reassemble(x, self.f)
        x = recover(x, xs)
        return x


def disassemble(x, f=2):
    _, c, w, h = x.size()
    x = x.unfold(2, w // f, w // f)
    x = x.unfold(3, h // f, h // f)
    x = x.permute(0, 2, 3, 1, 4, 5)
    x = x.reshape(-1, c, w // f, h // f)
    return x


def reassemble(x, f=2):
    b, c, w, h = x.size()
    x = x.view(b // f ** 2, f ** 2, c, w, h)
    x = x.permute(0, 2, 3, 4, 1)
    x = x.reshape(b // f ** 2, c * w * h, f ** 2)
    x = F.fold(x, (w * f, h * f), (w, h), (1, 1), stride=(w, h))
    return x


def compress(x):
    xs = None
    if x.dim() == 5:
        xs = x.size()
        x = x.view(
            xs[0],
            xs[1] * xs[2],
            *xs[3:]
        )

    return x, xs


def recover(x, xs):
    if xs is not None:
        x = x.view(
            -1,
            *xs[1:3],
            x.size(-2),
            x.size(-1)
        )

    return x