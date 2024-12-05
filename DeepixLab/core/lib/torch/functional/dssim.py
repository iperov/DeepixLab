import torch

from .ssim import ssim


def dssim(img1 : torch.Tensor, img2 : torch.Tensor,
          max_val : float = 1.0, kernel_size : int = 11, sigma : float = 1.5, k1 : float = 0.01, k2 : float = 0.03,
          use_padding=False,):

    value = ssim(img1, img2, max_val=max_val, kernel_size=kernel_size, sigma=sigma, k1=k1, k2=k2, use_padding=use_padding)
    return (1-value) / 2.0
