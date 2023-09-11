from typing import Literal

import torch
from torch.nn.functional import mse_loss


def margin_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    margin: float,
    reduction: Literal["mean", "sum", "none"] = "mean",
    **kwargs
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
