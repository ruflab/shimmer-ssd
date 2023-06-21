from pathlib import Path
from typing import Any, Callable, NamedTuple

import numpy as np
import torch
from PIL import Image


class SimpleShapesImages:
    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        transform: Callable[[Any], Any] | None = None,
    ) -> None:
        assert split in ("train", "val", "test"), "Invalid split"

        self.dataset_path = Path(dataset_path)
        self.split = split
        self.image_path = self.dataset_path / self.split
        self.transform = transform

    def __len__(self) -> int:
        return len(list(self.image_path.iterdir()))

    def __getitem__(self, index: int) -> Image.Image:
        path = self.image_path / f"{index}.png"
        with open(path, "rb") as f:
            image = Image.open(f)
            image = image.convert("RGB")

        if self.transform is not None:
            return self.transform(image)
        return image


class Attribute(NamedTuple):
    category: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    size: torch.Tensor
    rotation: torch.Tensor
    color_r: torch.Tensor
    color_g: torch.Tensor
    color_b: torch.Tensor
    color_h: torch.Tensor
    color_s: torch.Tensor
    color_v: torch.Tensor


class SimpleShapesAttributes:
    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        transform: Callable[[Any], Any] | None = None,
    ) -> None:
        assert split in ("train", "val", "test"), "Invalid split"

        self.dataset_path = Path(dataset_path)
        self.split = split
        self.labels: torch.Tensor = torch.from_numpy(
            np.load(self.dataset_path / f"{split}_labels.npy")
        )
        self.transform = transform

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, index: int) -> Attribute:
        label = self.labels[index]
        item = Attribute(
            category=label[0],
            x=label[1],
            y=label[2],
            size=label[3],
            rotation=label[4],
            color_r=label[5] / 255,
            color_g=label[6] / 255,
            color_b=label[7] / 255,
            color_h=label[8] / 255,
            color_s=label[9] / 255,
            color_v=label[10] / 255,
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


class SimpleShapesRawText:
    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        transform: Callable[[Any], Any] | None = None,
    ) -> None:
        assert split in ("train", "val", "test"), "Invalid split"

        self.dataset_path = Path(dataset_path)
        self.split = split

        self.captions = np.load(self.dataset_path / f"{split}_captions.npy")
        self.choices = np.load(
            self.dataset_path / f"{split}_caption_choices.npy",
            allow_pickle=True,
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, index: int) -> RawText:
        item = RawText(
            caption=self.captions[index], choice=Choice(**self.choices[index])
        )

        if self.transform is not None:
            return self.transform(item)
        return item


class SimpleShapesText:
    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        transform: Callable[[Any], Any] | None = None,
    ) -> None:
        assert split in ("train", "val", "test"), "Invalid split"

        self.dataset_path = Path(dataset_path)
        self.split = split

        self.raw_text = SimpleShapesRawText(self.dataset_path, self.split)

        self.bert_mean = torch.from_numpy(
            np.load(self.dataset_path / "latent_mean.npy")
        )
        self.bert_std = torch.from_numpy(
            np.load(self.dataset_path / "latent_std.npy")
        )

        bert_data = torch.from_numpy(
            np.load(self.dataset_path / f"{self.split}_latent.npy")[0]
        )
        self.bert_data = (bert_data - self.bert_mean) / self.bert_std
        self.transform = transform

    def __len__(self) -> int:
        return self.bert_data.size(0)

    def __getitem__(self, index: int) -> Text:
        item = Text(
            caption=self.raw_text[index].caption,
            bert=self.bert_data[index],
            choice=self.raw_text[index].choice,
        )

        if self.transform is not None:
            return self.transform(item)
        return item


AVAILABLE_MODALITIES = {
    "v": SimpleShapesImages,
    "attr": SimpleShapesAttributes,
    "raw_text": SimpleShapesRawText,
    "t": SimpleShapesText,
}
