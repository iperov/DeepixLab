import torch.nn as nn


def xavier_uniform(module : nn.Module):
    """
    init module and all submodules with Xavier Uniform

    supported modules: nn.Conv2d, nn.Conv1d, nn.Linear
    """
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear) ):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
