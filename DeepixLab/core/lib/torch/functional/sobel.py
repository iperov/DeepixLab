import functools

import numpy as np
import torch
import torch.nn.functional as F

from .gaussian import gaussian_blur


def rot_matrix(m, diff):

    H, W = m.shape

    if H != W or (W % 2) == 0:
        raise

    out = m.copy()

    ring_count = W//2
    for ring in range(0, W//2):
        # unwrap ring

        T = m[ring,ring:W-ring]; T_len = len(T)
        R = m[ring+1:W-ring :,W-ring-1]; R_len = len(R)
        B = m[W-ring-1,ring:W-ring-1][::-1]; B_len = len(B)
        L = m[ring+1:W-ring-1:,ring][::-1]; L_len = len(L)

        x = np.concatenate([T,R,B,L], 0)
        x = np.roll(x, diff*(ring_count-ring) )

        # wrap ring

        out[ring,ring:W-ring] = x[:T_len]
        out[ring+1:W-ring :,W-ring-1] = x[T_len:T_len+R_len]

        out[W-ring-1,ring:W-ring-1][::-1] = x[T_len+R_len:T_len+R_len+B_len]
        out[ring+1:W-ring-1:,ring][::-1] = x[T_len+R_len+B_len:]
    return out

@functools.cache
def get_sobel_kernel(ch, kernel_size=5, diagonals=True, dtype=np.float32) -> np.ndarray:
    """returns (ksize,ksize) np sobel kernel"""

    range = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)


    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[kernel_size // 2, kernel_size // 2] = 1  # avoid division by zero


    kx = x / sobel_2D_denominator
    ky = y / sobel_2D_denominator

    k = np.concatenate([kx[None,...],ky[None,...]]
                            + ([ rot_matrix(kx, 1)[None,...], rot_matrix(ky, 1)[None,...]  ] if diagonals else [])
                        , 0)[:,None,:,:]

    k = np.tile(k, (ch,1,1,1))

    return k.astype(dtype, copy=False)


def sobel_edges_2d(x : torch.Tensor, kernel_size=5, flat_ch=False, blur=False, norm=False):
    """
    some hand-crafted func for sobel edges.
    """
    _, C, H, W = x.shape

    kernel_np = get_sobel_kernel(C, kernel_size=kernel_size, diagonals=True, )
    kernel_t = torch.tensor(kernel_np, dtype=x.dtype, device=x.device)

    x_sobel = F.conv2d(F.pad(x, (kernel_size//2,)*4, mode='reflect'), kernel_t, stride=1, padding=0, groups=C)

    if flat_ch:
        x_sobel = x_sobel.pow(2).mean(1, keepdim=True).sqrt()

    if blur:
        x_sobel = gaussian_blur(x_sobel, sigma=max(H,W)/256.0)
    if norm:
        x_sobel = x_sobel / x_sobel.max().detach()

    return x_sobel