from typing import TypedDict

import numpy as np


class Boundaries(TypedDict):
    shape: tuple[int, int, int]
    color: tuple[int, int, int]
    scale: tuple[int, int, int]
    rotation: tuple[float, float, float]
    x: tuple[int, int, int]
    y: tuple[int, int, int]


def attr_boundaries(imsize: int, min_scale: int, max_scale: int):
    shape_boundary = np.random.randint(0, 2)
    shape_boundaries = (
        shape_boundary,
        (shape_boundary + 1) % 3,
        (shape_boundary + 2) % 3,
    )

    color_boundary = np.random.randint(0, 255)
    color_boundaries = (
        color_boundary,
        (color_boundary + 85) % 256,
        (color_boundary + 170) % 256,
    )

    scale_range = (max_scale - min_scale) / 3
    scale_boundary = np.random.randint(min_scale, max_scale)
    scale_boundaries = (
        scale_boundary,
        int(min_scale + (scale_boundary + scale_range) % max_scale),
        int(min_scale + (scale_boundary + 2 * scale_range) % max_scale),
    )

    rotation_boundary = np.random.rand() * 2 * np.pi
    rotation_boundaries = (
        rotation_boundary,
        (rotation_boundary + 2 * np.pi / 3) % (2 * np.pi),
        (rotation_boundary + 4 * np.pi / 3) % (2 * np.pi),
    )

    margin = max_scale // 2
    x_range = (imsize - 2 * margin) / 3
    x_boundary = np.random.randint(margin, imsize - margin)
    x_boundaries = (
        x_boundary,
        int(margin + (x_boundary + x_range) % imsize),
        int(margin + (x_boundary + 2 * x_range) % imsize),
    )

    y_boundary = np.random.randint(margin, imsize - margin)
    y_boundaries = (
        y_boundary,
        int(margin + (y_boundary + x_range) % imsize),
        int(margin + (y_boundary + 2 * x_range) % imsize),
    )

    return Boundaries(
        shape=shape_boundaries,
        color=color_boundaries,
        scale=scale_boundaries,
        rotation=rotation_boundaries,
        x=x_boundaries,
        y=y_boundaries,
    )


def ood_split(imsize: int, min_scale: int, max_scale: int):
    boundaries = attr_boundaries(imsize, min_scale, max_scale)
    items = list(boundaries.values())
    selected = []
    for _ in range(2):
        choice = np.random.randint(0, len(items) - 1)
        selected.append(items.pop(choice))
