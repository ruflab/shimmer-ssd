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
