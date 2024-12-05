import torch

class Optimizer(torch.optim.Optimizer):
    def __init__(self, params, **defaults):
        super().__init__(params, defaults)

    def step(self, iteration : int = None, grad_mult : float = 1.0, lr : float = 1e-3, lr_dropout : float = 1.0):
        raise NotImplementedError()
