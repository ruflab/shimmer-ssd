from collections.abc import Sequence

import torch
import torch.nn.functional as F

from simple_shapes_dataset.dataset.domain import Attribute


class NormalizeAttributes:
    def __init__(self, image_size: int = 32):
        self.image_size = image_size

    def __call__(self, attr: Attribute) -> Attribute:
        return Attribute(
            category=attr.category,
            x=(attr.x / self.image_size) * 2 - 1,
            y=(attr.y / self.image_size) * 2 - 1,
            size=(attr.size / self.image_size) * 2 - 1,
            rotation=attr.rotation,
            color_r=(attr.color_r) * 2 - 1,
            color_g=(attr.color_g) * 2 - 1,
            color_b=(attr.color_b) * 2 - 1,
        )


def to_unit_range(x: torch.Tensor) -> torch.Tensor:
    return (x + 1) / 2


class UnnormalizeAttributes:
    def __init__(self, image_size: int = 32):
        self.image_size = image_size

    def __call__(self, attr: Attribute) -> Attribute:
        return Attribute(
            category=attr.category,
            x=to_unit_range(attr.x) * self.image_size,
            y=to_unit_range(attr.y) * self.image_size,
            size=to_unit_range(attr.size) * self.image_size,
            rotation=to_unit_range(attr.rotation),
            color_r=to_unit_range(attr.color_r),
            color_g=to_unit_range(attr.color_g),
            color_b=to_unit_range(attr.color_b),
        )


def attribute_to_tensor(attr: Attribute) -> list[torch.Tensor]:
    return [
        F.one_hot(attr.category, num_classes=3),
        torch.cat(
            [
                attr.x.unsqueeze(0),
                attr.y.unsqueeze(0),
                attr.size.unsqueeze(0),
                attr.rotation.cos().unsqueeze(0),
                attr.rotation.sin().unsqueeze(0),
                attr.color_r.unsqueeze(0),
                attr.color_g.unsqueeze(0),
                attr.color_b.unsqueeze(0),
            ]
        ),
    ]


def tensor_to_attribute(tensor: Sequence[torch.Tensor]) -> Attribute:
    category = tensor[0]
    attributes = tensor[1]

    return Attribute(
        category=category.argmax(dim=1),
        x=attributes[:, 0],
        y=attributes[:, 1],
        size=attributes[:, 2],
        rotation=torch.atan2(attributes[:, 6], attributes[:, 7]),
        color_r=attributes[:, 5],
        color_g=attributes[:, 6],
        color_b=attributes[:, 7],
    )
