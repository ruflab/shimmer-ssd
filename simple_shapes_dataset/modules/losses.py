from typing import Literal

import torch
from torch.nn.functional import mse_loss


def margin_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    margin: float,
    reduction: Literal["mean", "sum", "none"] = "mean",
    **kwargs,
) -> torch.Tensor:
    out = torch.max(
        torch.zeros_like(x),
        margin - mse_loss(x, y, reduction="none", **kwargs),
    )
    if reduction == "mean":
        return out.mean()
    if reduction == "sum":
        return out.sum()
    return out


class RBF(torch.nn.Module):
    """
    from https://github.com/yiftachbeer/mmd_loss_pytorch
    """

    def __init__(
        self,
        n_kernels: int = 5,
        mul_factor: float = 2.0,
        bandwidth: torch.Tensor | None = None,
    ):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (
            torch.arange(n_kernels) - n_kernels // 2
        )
        self.bandwidth = bandwidth

    def get_bandwidth(self, l2_distances: torch.Tensor) -> torch.Tensor:
        if self.bandwidth is None:
            n_samples = l2_distances.size(0)
            return l2_distances.sum() / (n_samples**2 - n_samples)

        return self.bandwidth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l2_distances = torch.cdist(x, x) ** 2
        return torch.exp(
            -l2_distances[None, ...]
            / (
                self.get_bandwidth(l2_distances)
                * self.bandwidth_multipliers.to(x.device)
            )[:, None, None]
        ).sum(dim=0)


class MMDLoss(torch.nn.Module):
    def __init__(self, kernel: torch.nn.Module | None = None):
        super().__init__()
        self.kernel = kernel or RBF()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        K = self.kernel(torch.vstack([x, y]))

        X_size = x.size(0)
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


def mmd_loss(
    x: torch.Tensor,
    target: torch.Tensor,
    bandwidth: list[int] | None = None,
    squared_mmd: bool = False,
) -> torch.Tensor:
    bandwidth = bandwidth or [2, 5, 10, 20, 40, 80]

    X = torch.cat((x, target), 0)
    XX = X @ X.t()
    X2 = torch.sum(X * X, dim=1, keepdim=True)

    K = XX - 0.5 * X2 - 0.5 * X2.t()

    s1 = torch.ones((x.size(0), 1)) / x.size(0)
    s2 = -torch.ones((target.size(0), 1)) / target.size(0)
    s = torch.cat((s1, s2), 0)
    S = s @ s.t()

    loss = torch.tensor(0.0).to(x.device)

    S = S.to(x.device)

    for i in range(len(bandwidth)):
        k = S * torch.exp(K / bandwidth[i])
        loss += k.sum()

    if squared_mmd:
        return loss
    return torch.sqrt(loss)
