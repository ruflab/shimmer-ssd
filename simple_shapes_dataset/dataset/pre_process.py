import torch
import torch.nn.functional as F

from simple_shapes_dataset.dataset.domain import Attribute


def attribute_to_tensor(attr: Attribute) -> torch.Tensor:
    return torch.cat(
        [
            F.one_hot(attr.category, num_classes=3),
            attr.x.unsqueeze(0),
            attr.y.unsqueeze(0),
            attr.size.unsqueeze(0),
            attr.rotation.cos().unsqueeze(0),
            attr.rotation.sin().unsqueeze(0),
            attr.color_r.unsqueeze(0),
            attr.color_g.unsqueeze(0),
            attr.color_b.unsqueeze(0),
        ]
    )
