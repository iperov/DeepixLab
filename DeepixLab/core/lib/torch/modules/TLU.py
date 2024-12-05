import torch
import torch.nn as nn


class TLU(nn.Module):
    def __init__(self, in_ch):
        super(TLU, self).__init__()
        self._in_ch = in_ch
        self._tau = nn.parameter.Parameter(torch.Tensor(1, in_ch, 1, 1), requires_grad=True)
        nn.init.zeros_(self._tau)

    def forward(self, x):
        return torch.max(x, self._tau)