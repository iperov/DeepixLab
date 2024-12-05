import numpy as np
import torch
import torch.nn as nn


def cai(module : nn.Module):
    """
    init module and all submoduleswith Convolution Aware Initialization

    supported modules: nn.Conv2d
    """
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, ) ):
            init_cai_conv2d(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def init_cai_conv2d( weight, eps_std=0.05):
    out_ch, in_ch, KH, KW = weight.shape
    dtype = weight.dtype
    device = weight.device

    fan_in = in_ch * (KH * KW)
    kernel_shape = (KH, KW)
    kernel_fft_shape = torch.fft.rfft2(torch.zeros(kernel_shape)).shape

    basis_size = np.prod(kernel_fft_shape)
    if basis_size == 1:
        x = torch.normal( 0.0, eps_std, (out_ch, in_ch, basis_size), dtype=dtype, device=device)
    else:
        nbb = in_ch // basis_size + 1
        x = torch.normal(0.0, 1.0, (out_ch, nbb, basis_size, basis_size), dtype=dtype, device=device)
        x = x * (1-torch.eye(basis_size, dtype=dtype, device=device)) + torch.permute(x, (0,1,3,2) )

        u, _, _ = torch.linalg.svd(x)

        x = u.permute(0,1,3,2).reshape(out_ch, -1, basis_size)[:,:in_ch,:]

    x = torch.reshape(x, ( (out_ch,in_ch,) + kernel_fft_shape ) )
    x = torch.fft.irfft2( x, kernel_shape, out=weight.data )
    x += torch.normal(0, eps_std, (out_ch,in_ch,)+kernel_shape, dtype=dtype, device=device)
    x *= torch.sqrt( (2/fan_in) / torch.var(x) )
    return x