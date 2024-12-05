import torch
import torch.nn.functional as F

from .gaussian import get_gaussian_kernel


def ssim(img1_t : torch.Tensor, img2_t : torch.Tensor,
         max_val : float = 1.0, kernel_size : int = 11, sigma : float = 1.5, k1 : float = 0.01, k2 : float = 0.03,
         use_padding=False,
         ):

    if img1_t.shape != img2_t.shape:
        raise ValueError('Image shapes must be equal')

    _,C,_,_ = img1_t.shape

    kernel_size = max(1, kernel_size)
    kernel_padding = kernel_size // 2 if use_padding else 0

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    kernel_t = torch.tensor(get_gaussian_kernel(C, kernel_size, sigma), device=img1_t.device)

    mean0 = F.conv2d(img1_t, kernel_t, stride=1, padding=kernel_padding, groups=C)
    mean1 = F.conv2d(img2_t, kernel_t, stride=1, padding=kernel_padding, groups=C)

    num0 = mean0 * mean1 * 2.0
    den0 = mean0.square() + mean1.square()
    luminance = (num0 + c1) / (den0 + c1)

    num1 = F.conv2d( img1_t*img2_t,                 kernel_t, padding=kernel_padding, groups=C) * 2.0
    den1 = F.conv2d( img1_t*img1_t + img2_t*img2_t, kernel_t, padding=kernel_padding, groups=C)

    cs = (num1 - num0 + c2) / (den1 - den0 + c2)

    return torch.mean(luminance * cs, dim=[-2,-1])