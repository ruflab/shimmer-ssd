import numpy as np
import torch
from torch import nn


def symlog(x, alpha=1):
    return torch.sign(x) * torch.log(1 + alpha * torch.abs(x)) / np.log(1 + alpha)


def symexp(x, alpha=1):
    return torch.sign(x) * (torch.exp(alpha * torch.abs(x)) - 1) / alpha


class SymLog(nn.Module):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return symlog(x, self.alpha)

    def inverse(self, x):
        return symexp(x, self.alpha)
