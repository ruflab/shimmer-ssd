from enum import StrEnum
from random import sample
from typing import NamedTuple

import numpy as np

from simple_shapes_dataset.cli.utils import Dataset


class BoundaryKind(StrEnum):
    shape = "shape"
    color = "color"
    size = "size"
    rotation = "rotation"
    x = "x"
    y = "y"


class Boundary(NamedTuple):
    kind: BoundaryKind
    low_bound: float
    high_bound: float
    min_val: float
    max_val: float


def attr_boundaries(
    imsize: int, min_size: int, max_size: int
) -> list[Boundary]:
    boundaries: list[Boundary] = []
    shape_boundary = np.random.randint(0, 2)
    color_boundary = np.random.randint(0, 255)
    size_range = (max_size - min_size) / 3
    size_boundary = np.random.randint(min_size, max_size)
    rotation_boundary = np.random.rand() * 2 * np.pi
    margin = max_size // 2
    x_range = (imsize - 2 * margin) / 3
    x_boundary = np.random.randint(margin, imsize - margin)
    y_boundary = np.random.randint(margin, imsize - margin)
    for k in range(3):
        boundaries.append(
            Boundary(
                BoundaryKind.shape,
                (shape_boundary + k) % 3,
                (shape_boundary + k + 1) % 3,
                0,
                2,
            )
        )

        boundaries.append(
            Boundary(
                BoundaryKind.color,
                (color_boundary + k * 85) % 256,
                (color_boundary + (k + 1) * 85) % 256,
                0,
                255,
            )
        )

        boundaries.append(
            Boundary(
                BoundaryKind.size,
                int(min_size + (size_boundary + k * size_range) % max_size),
                int(
                    min_size
                    + (size_boundary + (k + 1) * size_range) % max_size
                ),
                min_size,
                max_size,
            )
        )

        boundaries.append(
            Boundary(
                BoundaryKind.rotation,
                (rotation_boundary + 2 * k * np.pi / 3) % (2 * np.pi),
                (rotation_boundary + 2 * (k + 1) * np.pi / 3) % (2 * np.pi),
                0,
                2 * np.pi,
            )
        )

        boundaries.append(
            Boundary(
                BoundaryKind.x,
                int(margin + (x_boundary + k * x_range) % imsize),
                int(margin + (x_boundary + (k + 1) * x_range) % imsize),
                margin,
                imsize - margin,
            )
        )

        boundaries.append(
            Boundary(
                BoundaryKind.y,
                int(margin + (y_boundary + k * x_range) % imsize),
                int(margin + (y_boundary + (k + 1) * x_range) % imsize),
                margin,
                imsize - margin,
            )
        )

    return boundaries


def filter_value(
    label: int,
    low_bound: float,
    high_bound: float,
    min_val: float,
    max_val: float,
) -> bool:
    if low_bound <= high_bound:
        return low_bound <= label < high_bound
    return low_bound <= label <= max_val or min_val <= label < high_bound


def ood_split(imsize: int, min_size: int, max_size: int) -> list[Boundary]:
    boundaries = attr_boundaries(imsize, min_size, max_size)
    return sample(boundaries, 2)


def filter_dataset(
    dataset: Dataset, boundaries: list[Boundary]
) -> tuple[list[int], list[int]]:
    in_dist_idx: list[int] = []
    ood_idx: list[int] = []
    for k in range(dataset.classes.shape[0]):
        for boundary in boundaries:
            match boundary.kind:
                case BoundaryKind.shape:
                    value = dataset.classes[k]
                case BoundaryKind.color:
                    value = dataset.colors_hls[k][0]
                case BoundaryKind.size:
                    value = dataset.sizes[k]
                case BoundaryKind.rotation:
                    value = dataset.rotations[k]
                case BoundaryKind.x:
                    value = dataset.locations[k][0]
                case BoundaryKind.y:
                    value = dataset.locations[k][1]
            if filter_value(
                value,
                boundary.low_bound,
                boundary.high_bound,
                boundary.min_val,
                boundary.max_val,
            ):
                ood_idx.append(k)
                break
        else:
            in_dist_idx.append(k)
    return in_dist_idx, ood_idx
