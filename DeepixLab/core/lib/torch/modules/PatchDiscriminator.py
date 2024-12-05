import torch
import torch.nn as nn
import torch.nn.functional as F

from ..init import xavier_uniform
from ..utils import RFA


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch : int, out_ch : int, patch_size : int, base_dim=32, max_downs=5):
        super().__init__()

        layers_cfg = RFA.create(layers=max_downs, k_list=(3,5,7), s_list=(1,2)).nearest(patch_size)

        dim_mults=[ min(2**i, 8) for i in range(layers_cfg.layers_count+1) ]
        dims = [ base_dim * mult for mult in dim_mults ]

        self._in_beta = nn.parameter.Parameter( torch.zeros(in_ch,), requires_grad=True)
        self._in_gamma = nn.parameter.Parameter( torch.ones(in_ch,), requires_grad=True)
        self._in = nn.Conv2d(in_ch, base_dim, 1, 1, 0, bias=False)

        down_c_list = self._down_c_list = nn.ModuleList()
        up_c_list = self._up_c_list = nn.ModuleList()
        up_s_list = self._up_s_list = nn.ModuleList()
        logits_list = self._logits_list = nn.ModuleList()

        for level, (rfa_layer, up_ch, down_ch) in enumerate(list(zip(layers_cfg, dims[:-1], dims[1:]))):
            ks     = rfa_layer.kernel_size
            stride = rfa_layer.stride

            down_c_list.append( nn.Conv2d(up_ch, down_ch, ks, stride, ks//2) )
            up_c_list.insert(0, nn.ConvTranspose2d(down_ch, up_ch, ks, stride, padding=ks//2, output_padding=stride-1) )
            up_s_list.insert(0, nn.ConvTranspose2d(down_ch, up_ch, ks, stride, padding=ks//2, output_padding=stride-1) )

            logits_list.insert(0, nn.Conv2d(up_ch, out_ch, 1, 1) )

            if level == layers_cfg.layers_count-1:
                self._mid_logit = nn.Conv2d(down_ch, out_ch, 1, 1)

        xavier_uniform(self)


    def forward(self, inp : torch.Tensor):
        x = inp

        x = x + self._in_beta[None,:,None,None]
        x = x * self._in_gamma[None,:,None,None]
        x = self._in(x)

        shortcuts = []
        for down_c in self._down_c_list:
            x = F.leaky_relu(down_c(x), 0.2)
            shortcuts.insert(0, x)

        logits = [ self._mid_logit(x) ]

        for shortcut_x, up_c, up_s, logit in zip(shortcuts, self._up_c_list, self._up_s_list, self._logits_list):
            x = F.leaky_relu(up_c(x) + up_s(shortcut_x), 0.2 )
            logits.append( logit(x) )

        return logits

    @staticmethod
    def get_max_patch_size(max_downs=5):
        return RFA.create(layers=max_downs, k_list=(3,5,7), s_list=(1,2)).max_rfs
