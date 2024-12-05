import torch

from .Optimizer import Optimizer


class AdaBelief(Optimizer):
    """
    https://juntang-zhuang.github.io/adabelief/
    """

    def __init__(self, params, betas=(0.9, 0.999), eps=None,):
        if eps is None:
            eps = torch.finfo(torch.float32).eps
        self._betas = betas
        self._eps = eps

        super().__init__(params)


    def step(self, iteration : int = None, grad_mult : float = 1.0, lr=1e-3, lr_dropout : float = 1.0, release_grad=False):
        beta1, beta2 = self._betas
        eps = self._eps

        for group in self.param_groups:
            for p in group['params']:
                if (grad := p.grad) is not None:
                    grad = grad.data

                    state = self.state[p]
                    if len(state) == 0:
                        m_t = state['m_t'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                        v_t = state['v_t'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    else:
                        m_t = state['m_t']
                        v_t = state['v_t']

                    if grad_mult != 1.0:
                        grad = grad * grad_mult

                    m_t.mul_(beta1).add_(  grad          , alpha=1 - beta1)
                    v_t.mul_(beta2).add_( (grad - m_t)**2, alpha=1 - beta2)

                    v_diff = (-lr * m_t).div_( v_t.sqrt().add_(eps) )

                    if lr_dropout != 1.0:
                        lrd = torch.full(p.size(), lr_dropout, dtype=torch.float16, device=v_diff.device)
                        torch.bernoulli(lrd, out=lrd)
                        v_diff.mul_(lrd)

                    p.data.add_(v_diff)

                    if release_grad:
                        p.grad = None
