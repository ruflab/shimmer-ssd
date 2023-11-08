from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple, TypedDict, overload

import numpy as np
import torch
from PIL import Image

# TODO: Consider handling CPU usage
# with a workaround in:
# https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662


class SimpleShapesDomain(Sequence):
    """
    Base class for a domain of the SimpleShapesDataset.
    All domains extend this base class and implement the
    __getitem__ and __len__ methods.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        transform: Callable[[Any], Any] | None = None,
        additional_args: dict[str, Any] | None = None,
    ) -> None:
        """
        Params:
            dataset_path (str | pathlib.Path): Path to the dataset.
            split (str): The split of the dataset to use. One of "train", "val", "test".
            transform (Any -> Any): Optional transform to apply to the data.
            additional_args (dict[str, Any]): Optional additional arguments to pass
                to the domain.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    @overload
    def __getitem__(self, index: int) -> Any:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Any]:
        ...

    def __getitem__(self, index):
        raise NotImplementedError


class SimpleShapesImages(SimpleShapesDomain):
    """
    Domain for the images of the SimpleShapesDataset.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        transform: Callable[[Any], Any] | None = None,
        additional_args: dict[str, Any] | None = None,
    ) -> None:
        assert split in ("train", "val", "test"), "Invalid split"

        self.dataset_path = Path(dataset_path)
        self.split = split
        self.image_path = (self.dataset_path / self.split).resolve()
        self.transform = transform
        self.additional_args = additional_args
        self.dataset_size = len(list(self.image_path.iterdir()))

    def __len__(self) -> int:
        return self.dataset_size

    @overload
    def __getitem__(self, index: int) -> Image.Image:
        ...

    @overload
    def __getitem__(self, index: slice) -> list[Image.Image]:
        ...

    def __getitem__(self, index):
        """
        Params:
            index: The index of the image to retrieve.
        Returns:
            A PIL image at the given index.
        """
        if isinstance(index, slice):
            determined_slice_indices = index.indices(len(self))
            return [self[i] for i in range(*determined_slice_indices)]

        path = self.image_path / f"{index}.png"
        with Image.open(path) as image:
            image = image.convert("RGB")

            if self.transform is not None:
                return self.transform(image)
            return image


class PretrainedVisualAdditionalArgs(TypedDict):
    presaved_path: str


class SimpleShapesPretrainedVisual(SimpleShapesDomain):
    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        transform: Callable[[Any], Any] | None = None,
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
        self.dataset_size = self.latents.size(0)

        assert (self.dataset_path / f"{split}_unpaired.npy").exists()
        unpaired = np.load(self.dataset_path / f"{split}_unpaired.npy")
        self.unpaired = torch.from_numpy(unpaired[:, 1]).float()

    def __len__(self) -> int:
        return self.dataset_size

    @overload
    def __getitem__(self, index: int) -> torch.Tensor:
        ...

    @overload
    def __getitem__(self, index: slice) -> list[torch.Tensor]:
        ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            determined_slice_indices = index.indices(len(self))
            return [self[i] for i in range(*determined_slice_indices)]

        x = torch.cat(
            [self.latents[index], self.unpaired[index].unsqueeze(0)], dim=0
        )

        if self.transform is not None:
            return self.transform(x)
        return x


class Attribute(NamedTuple):
    """
    NamedTuple for the attributes of the SimpleShapesDataset.
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


class SimpleShapesAttributes(SimpleShapesDomain):
    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        transform: Callable[[Any], Any] | None = None,
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
        self.dataset_size = self.labels.size(0)

        assert (self.dataset_path / f"{split}_unpaired.npy").exists()
        assert (
            self.additional_args["n_unpaired"] >= 1
        ), "n_unpaired should be >= 1"
        self.unpaired = torch.from_numpy(
            np.load(self.dataset_path / f"{split}_unpaired.npy")[
                :, 2 : 2 + self.additional_args["n_unpaired"]
            ]
        ).float()

    def __len__(self) -> int:
        return self.dataset_size

    @overload
    def __getitem__(self, index: int) -> Attribute:
        ...

    @overload
    def __getitem__(self, index: slice) -> list[Attribute]:
        ...

    def __getitem__(self, index):
        """
        Returns:
            An Attribute named tuple at the given index.
        """
        if isinstance(index, slice):
            determined_slice_indices = index.indices(len(self))
            return [self[i] for i in range(*determined_slice_indices)]

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


class SimpleShapesRawText(SimpleShapesDomain):
    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        transform: Callable[[Any], Any] | None = None,
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
        self.dataset_size = len(self.captions)

    def __len__(self) -> int:
        return self.dataset_size

    @overload
    def __getitem__(self, index: int) -> RawText:
        ...

    @overload
    def __getitem__(self, index: slice) -> list[RawText]:
        ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            determined_slice_indices = index.indices(len(self))
            return [self[i] for i in range(*determined_slice_indices)]

        item = RawText(
            caption=self.captions[index], choice=Choice(**self.choices[index])
        )

        if self.transform is not None:
            return self.transform(item)
        return item


class SimpleShapesText(SimpleShapesDomain):
    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        transform: Callable[[Any], Any] | None = None,
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
        self.latent_filename = self.additional_args.get(
            "latent_filename", "latent"
        )

        self.raw_text = SimpleShapesRawText(self.dataset_path, self.split)
        self.attributes = SimpleShapesAttributes(self.dataset_path, self.split)

        self.bert_mean = torch.from_numpy(
            np.load(self.dataset_path / f"{self.latent_filename}_mean.npy")
        )
        self.bert_std = torch.from_numpy(
            np.load(self.dataset_path / f"{self.latent_filename}_std.npy")
        )

        bert_data = torch.from_numpy(
            np.load(
                self.dataset_path / f"{self.split}_{self.latent_filename}.npy"
            )
        )[0]
        assert bert_data.ndim == 2
        self.bert_data = (bert_data - self.bert_mean) / self.bert_std
        self.transform = transform
        self.dataset_size = self.bert_data.size(0)

    def __len__(self) -> int:
        return self.dataset_size

    @overload
    def __getitem__(self, index: int) -> Text:
        ...

    @overload
    def __getitem__(self, index: slice) -> list[Text]:
        ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            determined_slice_indices = index.indices(len(self))
            return [self[i] for i in range(*determined_slice_indices)]

        item = Text(
            caption=self.raw_text[index].caption,
            bert=self.bert_data[index],
            choice=self.raw_text[index].choice,
            attr=self.attributes[index],
        )

        if self.transform is not None:
            return self.transform(item)
        return item


AVAILABLE_DOMAINS: dict[str, type[SimpleShapesDomain]] = {
    "v": SimpleShapesImages,
    "v_latents": SimpleShapesPretrainedVisual,
    "attr": SimpleShapesAttributes,
    "raw_text": SimpleShapesRawText,
    "t": SimpleShapesText,
}
