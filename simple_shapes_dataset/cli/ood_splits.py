from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from random import randint, sample
from typing import Any

import numpy as np

from simple_shapes_dataset.cli.utils import Dataset


class BoundaryKind(StrEnum):
    shape = "shape"
    color = "color"
    size = "size"
    rotation = "rotation"
    position = "position"


class BoundaryBase(ABC):
    @abstractmethod
    def choice(self) -> Any: ...

    @abstractmethod
    def filter(self, x: Any) -> bool: ...

    @abstractmethod
    def description(self) -> list[str]: ...


class ChoiceBoundary(BoundaryBase):
    def __init__(self, choices: Sequence):
        self.choices = choices

        self.boundary = self.choice()

    def choice(self):
        return sample(self.choices, 1)[0]

    def filter(self, x) -> bool:
        return x == self.boundary

    def description(self) -> list[str]:
        return [str(self.boundary), str(self.boundary)]


class BinsBoundary(BoundaryBase):
    def __init__(self, bins: Sequence):
        self.bins = bins

        self.boundary = self.choice()

    def choice(self):
        k = randint(1, len(self.bins) - 1)
        return k

    def filter(self, x) -> bool:
        return self.bins[self.boundary - 1] <= x < self.bins[self.boundary]

    def description(self) -> list[str]:
        return [
            str(self.bins[self.boundary - 1]),
            str(self.bins[self.boundary]),
        ]


class MultiBinsBoundary(BoundaryBase):
    def __init__(self, bins: Sequence):
        self.bins = [BinsBoundary(b) for b in bins]

        self.choice()

    def choice(self) -> list[Any]:
        return [b.choice() for b in self.bins]

    def filter(self, x) -> bool:
        return all(b.filter(k) for k, b in zip(x, self.bins, strict=True))

    def description(self) -> list[str]:
        return [
            "/".join(b.description()[0] for b in self.bins),
            "/".join(b.description()[1] for b in self.bins),
        ]


@dataclass
class BoundaryInfo:
    kind: BoundaryKind
    boundary: BoundaryBase


def attr_boundaries(imsize: int, min_size: int, max_size: int) -> list[BoundaryInfo]:
    size_range = (max_size - min_size) / 3
    margin = max_size // 2
    x_range = (imsize - 2 * margin) / 3

    boundaries: list[BoundaryInfo] = [
        BoundaryInfo(
            BoundaryKind.shape,
            ChoiceBoundary([0, 1, 2]),
        ),
        BoundaryInfo(
            BoundaryKind.color,
            BinsBoundary([0, 85, 170, 255]),
        ),
        BoundaryInfo(
            BoundaryKind.size,
            BinsBoundary(
                [
                    min_size,
                    min_size + size_range,
                    min_size + 2 * size_range,
                    max_size,
                ]
            ),
        ),
        BoundaryInfo(
            BoundaryKind.rotation,
            BinsBoundary(
                [
                    0,
                    2 * np.pi / 3,
                    4 * np.pi / 3,
                    2 * np.pi,
                ]
            ),
        ),
        BoundaryInfo(
            BoundaryKind.position,
            MultiBinsBoundary(
                [
                    [
                        margin,
                        margin + x_range,
                        margin + 2 * x_range,
                        imsize - margin,
                    ],
                    [
                        margin,
                        margin + x_range,
                        margin + 2 * x_range,
                        imsize - margin,
                    ],
                ]
            ),
        ),
    ]
    return boundaries


def ood_split(imsize: int, min_size: int, max_size: int) -> list[BoundaryInfo]:
    boundaries = attr_boundaries(imsize, min_size, max_size)
    return sample(boundaries, 2)


def filter_dataset(
    dataset: Dataset, boundary_infos: list[BoundaryInfo]
) -> tuple[list[int], list[int]]:
    in_dist_idx: list[int] = []
    ood_idx: list[int] = []
    for k in range(dataset.classes.shape[0]):
        for boundary_info in boundary_infos:
            match boundary_info.kind:
                case BoundaryKind.shape:
                    value = dataset.classes[k]
                case BoundaryKind.color:
                    value = dataset.colors_hls[k][0]
                case BoundaryKind.size:
                    value = dataset.sizes[k]
                case BoundaryKind.rotation:
                    value = dataset.rotations[k]
                case BoundaryKind.position:
                    value = (dataset.locations[k][0], dataset.locations[k][1])
            if boundary_info.boundary.filter(value):
                ood_idx.append(k)
                break
        else:
            in_dist_idx.append(k)
    return in_dist_idx, ood_idx
