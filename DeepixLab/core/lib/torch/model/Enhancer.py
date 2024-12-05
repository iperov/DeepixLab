import torch
import torch.nn as nn
import torch.nn.functional as F

from core.lib.torch import init as torch_init


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self._c0 = nn.Conv2d(in_ch, in_ch, 3, 1, 1)
        self._c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)

    def forward(self, inp):
        x = inp
        x = F.leaky_relu(self._c0(x), 0.1)
        x = F.leaky_relu(self._c1(x), 0.1)
        return x

class SimpleAtten(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self._c0 = nn.Conv2d(ch, ch, 3, 1, 1)
        self._c1 = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, inp):
        a = F.leaky_relu(self._c0(inp), 0.1)
        a = F.leaky_relu(self._c1(a), 0.1)

        _, _, H, W = a.size()
        d = (a - a.mean(dim=[2,3], keepdim=True)).pow(2)
        a = d / (4 * (d.sum(dim=[2,3], keepdim=True) / (W * H - 1) + 1e-4)) + 0.5

        return inp*torch.sigmoid(a)

class ResidualBlock(nn.Module):
    def __init__(self, ch, mid_ch = None, atten=False):
        """emb should match mid_ch"""
        super().__init__()
        if mid_ch is None:
            mid_ch = ch

        self._c0 = nn.Conv2d(ch, mid_ch, 3, 1, 1)
        self._c1 = nn.Conv2d(mid_ch, ch, 3, 1, 1)
        self._atten = SimpleAtten(ch) if atten else None

    def forward(self, inp, emb=None):
        x = inp
        x = self._c0(x)
        if emb is not None:
            x = x + emb
        x = F.leaky_relu(x, 0.2)
        x = self._c1(x)
        if self._atten is not None:
            x = self._atten(x)
        x = F.leaky_relu(x + inp, 0.2)
        return x

class Enhancer(nn.Module):
    def __init__(self, in_ch = 3,
                        out_ch = 3,
                        base_dim = 32,
                        depth = 4,):
        super().__init__()

        self._in = nn.Conv2d(in_ch, base_dim, 1, 1, 0)

        down_c1_list = self._down_c1_list = nn.ModuleList()
        down_c2_list = self._down_c2_list = nn.ModuleList()
        down_p_list = self._down_p_list = nn.ModuleList()

        up_c1_list = self._up_c1_list = nn.ModuleList()
        up_s_list = self._up_s_list = nn.ModuleList()
        up_r_list = self._up_r_list = nn.ModuleList()
        up_c2_list = self._up_c2_list = nn.ModuleList()

        dim_mult = [ min(2**i, 8) for i in range(depth+1) ]
        dims = [ base_dim * mult for mult in dim_mult ]
        for up_ch, down_ch in list(zip(dims[:-1], dims[1:])):
            down_c1_list.append( ConvBlock(up_ch, up_ch) )
            down_c2_list.append( ConvBlock(up_ch, down_ch) )
            down_p_list.append( nn.MaxPool2d(2) )

            up_c1_list.insert(0, ConvBlock(down_ch, up_ch*4) )
            up_s_list.insert(0, nn.Sequential(  nn.Conv2d(down_ch, up_ch*4, 3, 1, 1),
                                                nn.LeakyReLU(0.2),
                                                nn.PixelShuffle(2),
                                                nn.Conv2d(up_ch, up_ch, 3, 1, 1)))
            up_r_list.insert(0, ResidualBlock(up_ch, atten=True) )
            up_c2_list.insert(0, ConvBlock(up_ch, up_ch) )

        self._out = nn.Conv2d(base_dim, out_ch, 1, 1, 0)

        torch_init.cai(self)


    def forward(self, inp):
        x = self._in(inp)

        shortcuts = []
        for down_c1, down_c2, down_p in zip(self._down_c1_list, self._down_c2_list, self._down_p_list):
            x = down_c1(x)
            x = down_c2(x)
            x = down_p(x)
            shortcuts.insert(0, x)

        x = x * (x.square().mean(dim=[1,2,3], keepdim=True) + 1e-06).rsqrt()

        for shortcut_x, up_c1, up_s, up_r, up_c2 in zip(shortcuts, self._up_c1_list, self._up_s_list, self._up_r_list, self._up_c2_list):
            x = F.pixel_shuffle(up_c1(x), 2)

            x = up_r(x, emb=up_s(shortcut_x))
            x = up_c2(x)

        return self._out(x)