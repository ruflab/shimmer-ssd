"""
From https://github.com/fartashf/vsepp
"""

from typing import Literal

import torch
from shimmer import ContrastiveLoss as CLIPContrastiveLoss
from shimmer import LossOutput
from torch import nn
from torch.nn.functional import normalize


def cosine_sim(im: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between all the image and sentence pairs"""
    return im.mm(s.t())


def order_sim(im: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Order embeddings similarity measure $max(0, s-im)$"""
    YmX = s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1)) - im.unsqueeze(
        0
    ).expand(s.size(0), im.size(0), s.size(1))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(
        self,
        margin: float,
        measure: Literal["cosine", "order"],
        max_violation: bool,
    ):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == "order":
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, im: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        im = normalize(im)
        s = normalize(s)
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0), device=im.device) > 0.5
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class VSEPPContrastiveLoss(nn.Module):
    def __init__(
        self,
        margin: float,
        measure: Literal["cosine", "order"],
        max_violation: bool,
        logit_scale: torch.Tensor,
    ):
        """
        Args:
            margin: rank loss margin
            measure: similarity measure used (cosine|order)
            max_violation: use max instead of sum in the rank loss
        """
        super().__init__()
        self.vsepp_contrastive_loss = ContrastiveLoss(margin, measure, max_violation)
        self.clip_contrastive_loss = CLIPContrastiveLoss(logit_scale)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> LossOutput:
        return LossOutput(
            self.vsepp_contrastive_loss(x, y),
            {"clip": self.clip_contrastive_loss(x, y).loss},
        )
