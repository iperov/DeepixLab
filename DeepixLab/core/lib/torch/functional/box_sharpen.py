import functools

import numpy as np
import torch
import torch.nn.functional as F


@functools.cache
def get_box_sharpen_kernel(ch, kernel_size : int, power : float, dtype=np.float32) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1

    k = np.zeros( (kernel_size, kernel_size), dtype=dtype)
    k[ kernel_size//2, kernel_size//2] = 1.0
    b = np.ones( (kernel_size, kernel_size), dtype=dtype) / (kernel_size**2)
    k = k + (k - b) * power

    return np.tile (k[None,None,...], (ch,1,1,1))


def box_sharpen(img_t : torch.Tensor, kernel_size : int, power : float):
    _,C,_,_ = img_t.shape

    kernel_t = torch.tensor(get_box_sharpen_kernel(C, kernel_size, power), device=img_t.device)

    return F.conv2d(img_t, kernel_t, stride=1, padding=kernel_size // 2, groups=C)