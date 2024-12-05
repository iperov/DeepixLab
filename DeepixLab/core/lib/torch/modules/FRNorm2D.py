import torch
import torch.nn as nn


class FRNorm2D(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self._in_ch = in_ch
        self._weight = nn.parameter.Parameter( torch.Tensor(1, in_ch, 1, 1), requires_grad=True)
        self._bias = nn.parameter.Parameter( torch.Tensor(1, in_ch, 1, 1), requires_grad=True)
        self._eps = nn.parameter.Parameter(torch.Tensor(1), requires_grad=True)
        nn.init.ones_(self._weight)
        nn.init.zeros_(self._bias)
        nn.init.constant_(self._eps, 1e-6)

    def forward(self, x):
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)
        x = x * torch.rsqrt(nu2 + self._eps.abs())
        return self._weight * x + self._bias