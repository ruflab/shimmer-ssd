from collections.abc import Sequence

import torch
import torch.nn.functional as F

from simple_shapes_dataset.dataset.domain import Attribute, Text
from simple_shapes_dataset.text import composer
from simple_shapes_dataset.text.utils import (
    choices_from_structure_categories,
    structure_category_from_choice,
)


class NormalizeAttributes:
    def __init__(self, min_size: int = 7, max_size: int = 14, image_size: int = 32):
        self.min_size = min_size
        self.max_size = max_size
        self.scale_size = self.max_size - self.min_size

        self.image_size = image_size
        self.min_position = self.max_size // 2
        self.max_position = self.image_size - self.min_position
        self.scale_position = self.max_position - self.min_position

    def __call__(self, attr: Attribute) -> Attribute:
        return Attribute(
            category=attr.category,
            x=((attr.x - self.min_position) / self.scale_position) * 2 - 1,
            y=((attr.y - self.min_position) / self.scale_position) * 2 - 1,
            size=((attr.size - self.min_size) / self.scale_size) * 2 - 1,
            rotation=attr.rotation,
            color_r=(attr.color_r) * 2 - 1,
            color_g=(attr.color_g) * 2 - 1,
            color_b=(attr.color_b) * 2 - 1,
            unpaired=attr.unpaired,
        )


def to_unit_range(x: torch.Tensor) -> torch.Tensor:
    return (x + 1) / 2


class UnnormalizeAttributes:
    def __init__(self, min_size: int = 7, max_size: int = 14, image_size: int = 32):
        self.min_size = min_size
        self.max_size = max_size
        self.scale_size = self.max_size - self.min_size

        self.image_size = image_size
        self.min_position = self.max_size // 2
        self.max_position = self.image_size - self.min_position
        self.scale_position = self.max_position - self.min_position

    def __call__(self, attr: Attribute) -> Attribute:
        return Attribute(
            category=attr.category,
            x=to_unit_range(attr.x) * self.scale_position + self.min_position,
            y=to_unit_range(attr.y) * self.scale_position + self.min_position,
            size=to_unit_range(attr.size) * self.scale_size + self.min_size,
            rotation=attr.rotation,
            color_r=to_unit_range(attr.color_r) * 255,
            color_g=to_unit_range(attr.color_g) * 255,
            color_b=to_unit_range(attr.color_b) * 255,
            unpaired=attr.unpaired,
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
        attr.unpaired,
    ]


def nullify_attribute_rotation(
    attr: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    new_attr = attr[1].clone()
    new_attr[3] = 0.0
    new_attr[4] = 1.0
    return [attr[0], new_attr, attr[2]]


def tensor_to_attribute(tensor: Sequence[torch.Tensor]) -> Attribute:
    category = tensor[0]
    attributes = tensor[1]
    unpaired = tensor[2]

    rotation = torch.atan2(attributes[:, 4], attributes[:, 3])
    constrained_rotation = torch.where(rotation < 0, rotation + 2 * torch.pi, rotation)

    return Attribute(
        category=category.argmax(dim=1),
        x=attributes[:, 0],
        y=attributes[:, 1],
        size=attributes[:, 2],
        rotation=constrained_rotation,
        color_r=attributes[:, 5],
        color_g=attributes[:, 6],
        color_b=attributes[:, 7],
        unpaired=unpaired,
    )


def color_blind_visual_domain(image: torch.Tensor) -> torch.Tensor:
    return image.mean(dim=0, keepdim=True).expand(3, -1, -1)


def text_to_bert(text: Text) -> torch.Tensor:
    return text.bert


class TextAndAttrs:
    def __init__(self, min_size: int = 7, max_size: int = 14, image_size: int = 32):
        self.normalize = NormalizeAttributes(min_size, max_size, image_size)

    def __call__(self, x: Text) -> dict[str, torch.Tensor]:
        text: dict[str, torch.Tensor] = {"bert": x.bert}
        attr = self.normalize(x.attr)
        attr_list = attribute_to_tensor(attr)
        text["cls"] = attr_list[0]
        text["attr"] = attr_list[1]
        text["unpaired"] = attr_list[2]
        grammar_categories = structure_category_from_choice(composer, x.choice)
        text.update(
            {
                name: torch.Tensor([category])
                for name, category in grammar_categories.items()
            }
        )
        return text


def attr_to_str(
    attr: Attribute, grammar_predictions: dict[str, list[int]]
) -> list[str]:
    captions: list[str] = []
    choices = choices_from_structure_categories(composer, grammar_predictions)
    for k in range(attr.category.size(0)):
        caption, _ = composer(
            {
                "shape": attr.category[k].detach().cpu().item(),
                "rotation": attr.rotation[k].detach().cpu().item(),
                "color": (
                    attr.color_r[k].detach().cpu().item(),
                    attr.color_g[k].detach().cpu().item(),
                    attr.color_b[k].detach().cpu().item(),
                ),
                "size": attr.size[k].detach().cpu().item(),
                "location": (
                    attr.x[k].detach().cpu().item(),
                    attr.y[k].detach().cpu().item(),
                ),
            },
            choices[k],
        )
        captions.append(caption)
    return captions
