import sys
import torch
import torch.nn as nn


def get_scale(key):
    return getattr(sys.modules[__name__], f'Scale{key}')


class ScaleBase(nn.Module):
    init_values = {
        'g': {
            'a1': 1.1243270635604858,
            'b1': 0.1694217026233673,
            'a2': 1.3301806449890137,
            'b2': 0.20295464992523193,
        },
        'r': {
            'a1': 0.9545445442199707,
            'b1': 0.4929998517036438,
            'a2': 1.2766683101654053,
            'b2': 0.4387783408164978,
        }
    }

    def __init__(self, n_channels, method='arcsinh'):
        super().__init__()
        self.n_channels = n_channels
        self.band_order = ['g', 'r']

    def scale(self, x):
        return scale(x, self.a1, self.b1, self.a2, self.b2)


class ScaleMultiple(ScaleBase):
    """Applies multiple arcsinh scaling operations with learned parameters

    Args:
        n_channels (int): Number of different scaling channels to learn.
        method (str, optional): What scaling operation to use - not implemented.
    """
    def __init__(self, n_channels, n_scaling=4, method='arcsinh'):
        super().__init__(n_channels, method)
        self.n_scaling = n_scaling
        self.a1 = nn.Parameter(data=torch.Tensor(1, self.n_scaling, n_channels, 1, 1))
        self.b1 = nn.Parameter(data=torch.Tensor(1, self.n_scaling, n_channels, 1, 1))
        self.a2 = nn.Parameter(data=torch.Tensor(1, self.n_scaling, n_channels, 1, 1))
        self.b2 = nn.Parameter(data=torch.Tensor(1, self.n_scaling, n_channels, 1, 1))
        self.init_params()

    def init_params(self):
        for j in range(self.n_scaling):
            for i in range(self.n_channels):
                self.a1.data[0, j, i, 0, 0] = torch.normal(torch.tensor(self.init_values[self.band_order[i]]['a1']), torch.tensor(.5)) / (j + 1)
                self.b1.data[0, j, i, 0, 0] = torch.normal(torch.tensor(self.init_values[self.band_order[i]]['b1']), torch.tensor(.5)) / (j + 1)
                self.a2.data[0, j, i, 0, 0] = torch.normal(torch.tensor(self.init_values[self.band_order[i]]['a2']), torch.tensor(.5)) / (j + 1)
                self.b2.data[0, j, i, 0, 0] = torch.normal(torch.tensor(self.init_values[self.band_order[i]]['b2']), torch.tensor(.5)) / (j + 1)

    def forward(self, x):
        xs = x.size()
        x = x.unsqueeze(1)
        x = self.scale(x)
        x = x.view(xs[0], -1, xs[2], xs[3])
        return x


class ScaleParallel(ScaleBase):
    """Stacks one learned arcsinh scaling operation with input

    Args:
        n_channels (int): Number of different scaling channels to learn.
        method (str, optional): What scaling operation to use - not implemented.
    """
    def __init__(self, n_channels, method='arcsinh', **kwargs):
        super().__init__(n_channels, method)
        self.n_scaling = 2
        self.a1 = nn.Parameter(data=torch.Tensor(1, n_channels, 1, 1))
        self.b1 = nn.Parameter(data=torch.Tensor(1, n_channels, 1, 1))
        self.a2 = nn.Parameter(data=torch.Tensor(1, n_channels, 1, 1))
        self.b2 = nn.Parameter(data=torch.Tensor(1, n_channels, 1, 1))
        self.init_params()

    def init_params(self):
        for i in range(self.n_channels):
            self.a1.data[0, i, 0, 0] = torch.normal(torch.tensor(self.init_values[self.band_order[i]]['a1']), torch.tensor(.5))
            self.b1.data[0, i, 0, 0] = torch.normal(torch.tensor(self.init_values[self.band_order[i]]['b1']), torch.tensor(.5))
            self.a2.data[0, i, 0, 0] = torch.normal(torch.tensor(self.init_values[self.band_order[i]]['a2']), torch.tensor(.5))
            self.b2.data[0, i, 0, 0] = torch.normal(torch.tensor(self.init_values[self.band_order[i]]['b2']), torch.tensor(.5))

    def forward(self, x):
        return torch.cat([x, self.scale(x)], 1)


def scale(x, a1, b1, a2, b2):
    scaled = torch.arcsinh(a1 * x + b1)
    scaled = torch.sigmoid(a2 * scaled + b2)
    return scaled
