import torch
from simple_shapes_dataset import Attribute
from torch import nn

from shimmer.modules import DomainModule


class ProtoLanguage(DomainModule):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential()

    def encode(self, x: Attribute) -> torch.Tensor:
        raise NotImplementedError
