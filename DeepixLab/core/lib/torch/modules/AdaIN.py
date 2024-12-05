from typing import Tuple

import torch
import torch.nn as nn


class AdaIN(nn.Module):
    def __init__(self, in_ch, mlp_dim):
        super().__init__()

        self._gamma_fc = nn.Linear(mlp_dim, in_ch)
        self._beta_fc = nn.Linear(mlp_dim, in_ch)

    def forward(self, inputs : Tuple[torch.Tensor, torch.Tensor]):
        x, mlp = inputs

        x_mean = x.mean((-2,-1), keepdim=True)
        x_std = x.std((-2,-1), keepdim=True) + 1e-5

        x = (x - x_mean) / x_std
        x = x * self._gamma_fc(mlp)[:,:,None,None] + self._beta_fc(mlp)[:,:,None,None]

        return x