import torch
import torch.nn.functional as F


def PatchSWD(x, y, patch_size=8, num_proj=64):
    """
    Patch Sliced Wasserstein Distance.

    ref: Generating natural images with direct Patch Distributions Matching https://arxiv.org/abs/2203.11862
    """
    _, C, _, _ = x.shape

    # Sample random normalized projections
    rand = torch.randn(num_proj, C*patch_size**2).to(x.device)
    rand = rand / torch.norm(rand, dim=1, keepdim=True)
    rand = rand.reshape(num_proj, C, patch_size, patch_size)

    # Project patches
    projx = F.conv2d(x, rand).transpose(1,0).reshape(num_proj, -1)
    projy = F.conv2d(y, rand).transpose(1,0).reshape(num_proj, -1)

    # Sort and compute L1 dist
    projx, _ = torch.sort(projx, dim=1)
    projy, _ = torch.sort(projy, dim=1)

    return torch.abs(projx - projy).mean()
