import torch.nn as nn
import torch.nn.functional as F

from .. import init as torch_init
from ..modules import BlurPool


class XSeg(nn.Module):
    def __init__(self,  in_ch = 1,
                        out_ch = 1,
                        base_dim = 32,
                        depth = 6,
                        generalization_level = 0):
        super().__init__()
        self._generalization_level = generalization_level

        self._in = nn.Conv2d(in_ch, base_dim, 1, 1, 0)

        down_list = self._down_list = nn.ModuleList()
        up_s_list = self._up_s_list = nn.ModuleList()
        up_r_list = self._up_r_list = nn.ModuleList()
        up_c_list = self._up_c_list = nn.ModuleList()

        dim_mult = [ min(2**i, 8) for i in range(depth+1) ]
        dims = [ base_dim * mult for mult in dim_mult ]


        for level, (up_ch, down_ch) in enumerate(list(zip(dims[:-1], dims[1:]))):

            down_list.append( nn.Sequential(nn.Conv2d (up_ch, down_ch, 3, 1, 1), nn.ReLU(),
                                            nn.Conv2d (down_ch, down_ch, 3, 1, 1), nn.ReLU(),
                                            BlurPool (down_ch, kernel_size=max(2, 4-level)),
                                        ))

            up_s_list.insert(0, nn.Sequential(  nn.Conv2d(down_ch, down_ch, 3, 1, 1),
                                                nn.LeakyReLU(0.2),
                                                nn.Conv2d(down_ch, down_ch, 3, 1, 1)))
            up_r_list.insert(0, nn.Conv2d(down_ch, down_ch, 3, 1, 1) )
            up_c_list.insert(0, nn.Conv2d(down_ch, up_ch*4, 3, 1, 1) )


        self._out = nn.Conv2d(base_dim, out_ch, 1, 1, 0)

        torch_init.cai(self)

    def set_generalization_level(self, level):
        self._generalization_level = level

    def reset_shortcut(self, n):
        torch_init.cai(self._up_s_list[n])

    def reset_encoder(self):
        torch_init.cai(self._in)
        torch_init.cai(self._down_list)
        torch_init.cai(self._up_s_list)

    def reset_decoder(self):
        torch_init.cai(self._up_r_list)
        torch_init.cai(self._up_c_list)
        torch_init.cai(self._out)

    def forward(self, inp):
        x = self._in(inp)

        shortcuts = []
        for down in self._down_list:
            x = down(x)
            shortcuts.insert(0, x)

        for i, (shortcut_x, up_s, up_r, up_c) in enumerate(zip(shortcuts, self._up_s_list, self._up_r_list, self._up_c_list)):
            level = len(shortcuts)-i-1

            x = x + up_r(x)
            if level >= self._generalization_level:
                x = x + up_s(shortcut_x)
            x = F.leaky_relu(x, 0.2)

            x = F.leaky_relu(up_c(x), 0.1)
            x = F.pixel_shuffle(x, 2)

        return self._out(x)