from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

import numpy as np
import torch
from PIL import Image
from shimmer import DataDomain, DomainDesc

from simple_shapes_dataset.types import DomainType


class SimpleShapesImages(DataDomain):
    """
    Domain for the images of the ShimmerDataset.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        max_size: int | None = None,
        transform: Callable[[Image.Image], Any] | None = None,
        additional_args: dict[str, Any] | None = None,
    ) -> None:
        assert split in ("train", "val", "test"), "Invalid split"

        self.dataset_path = Path(dataset_path)
        self.split = split
        self.image_path = (self.dataset_path / self.split).resolve()
        self.transform = transform
        self.additional_args = additional_args
        dataset_size = len(list(self.image_path.iterdir()))
        self.dataset_size = min(
            dataset_size,
            max_size or dataset_size,
        )

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index: int):
        """
        Params:
            index: The index of the image to retrieve.
        Returns:
            A PIL image at the given index.
        """
        path = self.image_path / f"{index}.png"
        with Image.open(path) as image:
            image = image.convert("RGB")

            if self.transform is not None:
                return self.transform(image)
            return image


class PretrainedVisualAdditionalArgs(TypedDict):
    presaved_path: str


class SimpleShapesPretrainedVisual(DataDomain):
    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        max_size: int | None = None,
        transform: Callable[[torch.Tensor], Any] | None = None,
        additional_args: PretrainedVisualAdditionalArgs | None = None,
    ) -> None:
        assert split in ("train", "val", "test"), "Invalid split"

        self.dataset_path = Path(dataset_path)
        self.split = split
        self.transform = transform
        self.additional_args = additional_args

        assert self.additional_args is not None

        self.presaved_path = (
            self.dataset_path
            / f"saved_latents/{split}/{self.additional_args['presaved_path']}"
        )
        self.latents = torch.from_numpy(np.load(self.presaved_path.resolve()))
        dataset_size = self.latents.size(0)
        self.dataset_size = min(dataset_size, max_size or dataset_size)

        assert (self.dataset_path / f"{split}_unpaired.npy").exists()
        unpaired = np.load(self.dataset_path / f"{split}_unpaired.npy")
        self.unpaired = torch.from_numpy(unpaired[:, 1]).float()

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index: int):
        x = torch.cat([self.latents[index], self.unpaired[index].unsqueeze(0)], dim=0)

        if self.transform is not None:
            return self.transform(x)
        return x


class Attribute(NamedTuple):
    """
    NamedTuple for the attributes of the ShimmerDataset.
    NamedTuples are used as they are correcly handled by pytorch's collate function.
    """

    category: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    size: torch.Tensor
    rotation: torch.Tensor
    color_r: torch.Tensor
    color_g: torch.Tensor
    color_b: torch.Tensor
    unpaired: torch.Tensor


class AttributesAdditionalArgs(TypedDict):
    n_unpaired: int


class SimpleShapesAttributes(DataDomain):
    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        max_size: int | None = None,
        transform: Callable[[Attribute], Any] | None = None,
        additional_args: AttributesAdditionalArgs | None = None,
    ) -> None:
        assert split in ("train", "val", "test"), "Invalid split"

        self.dataset_path = Path(dataset_path).resolve()
        self.split = split
        self.labels: torch.Tensor = torch.from_numpy(
            np.load(self.dataset_path / f"{split}_labels.npy")
        )
        self.transform = transform

        default_args = AttributesAdditionalArgs(n_unpaired=1)
        self.additional_args = additional_args or default_args
        dataset_size = self.labels.size(0)
        self.dataset_size = min(dataset_size, max_size or dataset_size)

        assert (self.dataset_path / f"{split}_unpaired.npy").exists()
        assert self.additional_args["n_unpaired"] >= 1, "n_unpaired should be >= 1"
        self.unpaired = torch.from_numpy(
            np.load(self.dataset_path / f"{split}_unpaired.npy")[
                :, 2 : 2 + self.additional_args["n_unpaired"]
            ]
        ).float()

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index: int):
        """
        Returns:
            An Attribute named tuple at the given index.
        """
        label = self.labels[index]
        item = Attribute(
            category=label[0].long(),
            x=label[1],
            y=label[2],
            size=label[3],
            rotation=label[4],
            color_r=label[5] / 255,
            color_g=label[6] / 255,
            color_b=label[7] / 255,
            unpaired=self.unpaired[index],
        )

        if self.transform is not None:
            return self.transform(item)
        return item


class Choice(NamedTuple):
    structure: int
    groups: list[int]
    writers: dict[str, dict[str, int]]
    variants: dict[str, int]


class RawText(NamedTuple):
    caption: str
    choice: Choice


class Text(NamedTuple):
    caption: str
    bert: torch.Tensor
    choice: Choice
    attr: Attribute


class SimpleShapesRawText(DataDomain):
    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        max_size: int | None = None,
        transform: Callable[[RawText], Any] | None = None,
        additional_args: dict[str, Any] | None = None,
    ) -> None:
        assert split in ("train", "val", "test"), "Invalid split"

        self.dataset_path = Path(dataset_path).resolve()
        self.split = split

        self.captions = np.load(self.dataset_path / f"{split}_captions.npy")
        self.choices = np.load(
            self.dataset_path / f"{split}_caption_choices.npy",
            allow_pickle=True,
        )
        self.transform = transform
        self.additional_args = additional_args or {}
        dataset_size = len(self.captions)
        self.dataset_size = min(dataset_size, max_size or dataset_size)

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index: int):
        item = RawText(
            caption=self.captions[index], choice=Choice(**self.choices[index])
        )

        if self.transform is not None:
            return self.transform(item)
        return item


class SimpleShapesText(DataDomain):
    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        max_size: int | None = None,
        transform: Callable[[Text], Any] | None = None,
        additional_args: dict[str, Any] | None = None,
    ) -> None:
        """
        Possible additional args:
            latent_filename: The name of the model used to get the latent file.
                It will load files of the form {split}_{latent_filename}.npy.
        """
        assert split in ("train", "val", "test"), "Invalid split"

        self.dataset_path = Path(dataset_path).resolve()
        self.split = split

        self.additional_args = additional_args or {}
        self.latent_filename = self.additional_args.get("latent_filename", "latent")

        self.raw_text = SimpleShapesRawText(self.dataset_path, self.split)
        self.attributes = SimpleShapesAttributes(self.dataset_path, self.split)

        self.bert_mean = torch.from_numpy(
            np.load(self.dataset_path / f"{self.latent_filename}_mean.npy")
        )
        self.bert_std = torch.from_numpy(
            np.load(self.dataset_path / f"{self.latent_filename}_std.npy")
        )

        bert_data = torch.from_numpy(
            np.load(self.dataset_path / f"{self.split}_{self.latent_filename}.npy")
        )
        assert bert_data.ndim == 2
        self.bert_data = (bert_data - self.bert_mean) / self.bert_std
        self.transform = transform
        dataset_size = self.bert_data.size(0)
        self.dataset_size = min(dataset_size, max_size or dataset_size)

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index: int):
        item = Text(
            caption=self.raw_text[index].caption,
            bert=self.bert_data[index],
            choice=self.raw_text[index].choice,
            attr=self.attributes[index],
        )

        if self.transform is not None:
            return self.transform(item)
        return item


DEFAULT_DOMAINS: dict[str, type[DataDomain]] = {
    "v": SimpleShapesImages,
    "v_latents": SimpleShapesPretrainedVisual,
    "attr": SimpleShapesAttributes,
    "raw_text": SimpleShapesRawText,
    "t": SimpleShapesText,
}


def get_default_domains(
    domains: Iterable[DomainDesc | str],
) -> dict[DomainDesc, type[DataDomain]]:
    domain_classes = {}
    for domain in domains:
        if isinstance(domain, str):
            domain = DomainType[domain].value
        domain_classes[domain] = DEFAULT_DOMAINS[domain.kind]
    return domain_classes
