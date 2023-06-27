import simple_shapes_dataset.cli as cli
import simple_shapes_dataset.text as text
from simple_shapes_dataset.modules.dataset import SimpleShapesDataset
from simple_shapes_dataset.modules.domain import (
    AVAILABLE_DOMAINS,
    Attribute,
    RawText,
    SimpleShapesAttributes,
    SimpleShapesImages,
    SimpleShapesRawText,
    SimpleShapesText,
    Text,
)
from simple_shapes_dataset.modules.domain_alignment import get_aligned_datasets

from .version import __version__

__all__ = [
    "cli",
    "text",
    "__version__",
    "SimpleShapesImages",
    "SimpleShapesAttributes",
    "SimpleShapesText",
    "SimpleShapesRawText",
    "SimpleShapesDataset",
    "Attribute",
    "RawText",
    "Text",
    "AVAILABLE_DOMAINS",
    "get_aligned_datasets",
]
