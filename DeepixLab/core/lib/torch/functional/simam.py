import torch


def simam(x : torch.Tensor):
    """
    SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks
    http://proceedings.mlr.press/v139/yang21o/yang21o.pdf
    """
    _, _, H, W = x.size()
    n = W * H - 1
    d = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
    v = d / (4 * (d.sum(dim=[2,3], keepdim=True) / n + 1e-4)) + 0.5
    return x * torch.sigmoid(v)