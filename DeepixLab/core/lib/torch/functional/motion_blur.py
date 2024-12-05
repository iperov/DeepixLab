import functools

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@functools.cache
def get_motion_blur_kernel(ch, kernel_size : int, angle : float, dtype=np.float32) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1

    k = np.zeros((kernel_size, kernel_size), dtype=dtype)
    k[ (kernel_size-1)// 2 , :] = np.ones(kernel_size, dtype=dtype)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D( (kernel_size / 2 -0.5 , kernel_size / 2 -0.5 ) , angle, 1.0), (kernel_size, kernel_size) )
    k = k * ( 1.0 / np.sum(k) )

    return np.tile (k[None,None,...], (ch,1,1,1)).astype(dtype, copy=False)


def motion_blur(img_t : torch.Tensor, kernel_size : int, angle : float):
    _,C,_,_ = img_t.shape
    kernel_t = torch.tensor(get_motion_blur_kernel(C, kernel_size, angle), device=img_t.device)

    out_t = F.conv2d(img_t, kernel_t, stride=1, padding=kernel_size // 2, groups=C)
    return out_t