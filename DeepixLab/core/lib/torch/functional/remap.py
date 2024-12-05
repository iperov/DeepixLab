import numpy as np
import torch
import torch.nn.functional as F


def remap(img_t : torch.Tensor, grid_t : torch.Tensor, mode : str = 'bilinear', padding_mode : str = 'zeros'):
    """
        img_t   N,C,H,W

        coord_t N,H,W,C of x,y coords

        mat     6x[ number ]

    """
    _,C,H,W = img_t.shape
    device = img_t.device

    grid_t /= torch.tensor( ( 0.5*(W-1), 0.5*(H-1) ), dtype=torch.float32).to(device, non_blocking=True)
    grid_t -= 1

    return F.grid_sample(img_t, grid_t, mode=mode, padding_mode=padding_mode, align_corners=True)

def warp_affine(img_t : torch.Tensor, mat : np.ndarray, OW : int, OH : int, mode : str = 'bilinear', padding_mode : str = 'zeros'):
    """
        img_t   N,C,H,W

        mat     np.ndarray (2,3) float32

    """
    _,C,H,W = img_t.shape
    device = img_t.device


    mat = torch.tensor( np.concatenate( (mat, np.float32([[0,0,1]])),0), dtype=torch.float32).to(device, non_blocking=True).reshape(3,3)
    mat = torch.linalg.inv(mat)[:2]

    grid_t = torch.stack(torch.meshgrid(torch.arange(OW, device=device, dtype=torch.float32),
                                        torch.arange(OH, device=device, dtype=torch.float32), indexing='xy')
                        + (torch.ones( (OH,OW), dtype=torch.float32, device=device),)
                        , -1)

    grid_t = (grid_t @ mat.T )

    return remap(img_t, grid_t[None,...], mode=mode, padding_mode=padding_mode)