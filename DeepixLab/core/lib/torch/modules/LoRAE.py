import torch
import torch.linalg
import torch.nn as nn


class LoRAE(nn.Module):
    """
    Learning Low-Rank Latent Spaces with Simple Deterministic Autoencoder: Theoretical and Empirical Insights
    https://github.com/tirthajit/LoRAE_WACV24
    """
    def __init__(self, dim : int):
        super().__init__()
        self.mlp = nn.Linear(dim, dim, bias=False)

        nn.init.zeros_(self.mlp.weight)

    def forward(self, x : torch.Tensor):
        return self.mlp(x)

    def loss(self, strength : float = 0.00005):
        return torch.linalg.norm(self.mlp.weight, ord='nuc')*strength
