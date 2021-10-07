import torch
import torch.nn
import torch.nn.functional as F
from typing import Optional


class TiedEmbedding(torch.nn.Module):
    def __init__(self, weights: torch.Tensor, batch_dim: Optional[int] = None):
        super().__init__()

        # Hack: won't save it as a parameter
        self.w = [weights]
        self.bias = torch.nn.Parameter(torch.zeros(self.w[0].shape[0]))
        self.batch_dim = batch_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return F.linear(t, self.w[0], self.bias, self.batch_dim)
