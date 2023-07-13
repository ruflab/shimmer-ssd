import io
import logging
from collections.abc import Sequence
from typing import Any

import lightning.pytorch as pl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from matplotlib import gridspec
from PIL import Image
from torchvision.utils import make_grid

from simple_shapes_dataset.cli.utils import generate_image
from simple_shapes_dataset.dataset.pre_process import (
    UnnormalizeAttributes,
    tensor_to_attribute,
)

matplotlib.use("Agg")


class LogSamplesCallback(pl.Callback):
    def __init__(
        self,
        reference_samples: Any,
        log_key: str,
        every_n_epochs: int | None = 1,
    ) -> None:
        super().__init__()
        self.reference_samples = reference_samples
        self.every_n_epochs = every_n_epochs
        self.log_key = log_key

    def to(self, samples: Any, device: torch.device) -> Any:
        if isinstance(samples, torch.Tensor):
            return samples.to(device)
        raise NotImplementedError

    def on_callback(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.logger is None:
            return

        samples = self.to(self.reference_samples, pl_module.device)
        if trainer.current_epoch == 0:
            self.log_samples(trainer.logger, samples, "reference")

        with torch.no_grad():
            pl_module.eval()
            generated_samples = pl_module(samples)
            pl_module.train()

        for logger in trainer.loggers:
            self.log_samples(logger, generated_samples, "prediction")

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if (
            self.every_n_epochs is None
            or trainer.current_epoch % self.every_n_epochs != 0
        ):
            return

        return self.on_callback(trainer, pl_module)

    def on_fit_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        return self.on_callback(trainer, pl_module)

    def log_samples(self, logger: Logger, samples: Any, mode: str) -> None:
        raise NotImplementedError


def get_pil_image(figure: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    figure.savefig(buf)
    buf.seek(0)
    return Image.open(buf)


def figure_grid(
    categories: np.ndarray,
    locations: np.ndarray,
    sizes: np.ndarray,
    rotations: np.ndarray,
    colors: np.ndarray,
    image_size: int,
    ncols: int = 8,
    dpi: float = 100,
) -> Image.Image:
    reminder = 1 if categories.shape[0] % ncols else 0
    nrows = categories.shape[0] // ncols + reminder

    width = ncols * (image_size + 1) + 1
    height = nrows * (image_size + 1) + 1

    figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    gs = gridspec.GridSpec(
        nrows,
        ncols,
        wspace=1 / 10,
        hspace=1 / 10,
        left=0,
        right=1,
        bottom=0,
        top=1,
    )
    for i in range(nrows):
        for j in range(ncols):
            k = i * ncols + j
            if k >= categories.shape[0]:
                break
            ax = plt.subplot(gs[i, j])
            generate_image(
                ax,
                categories[k],
                locations[k],
                sizes[k],
                rotations[k],
                colors[k],
                image_size,
            )
            ax.set_facecolor("black")
    image = get_pil_image(figure)  # type: ignore
    plt.close(figure)
    return image


def log_attribute(
    logger: WandbLogger,
    samples: Sequence[torch.Tensor],
    image_size: int,
    key: str,
    ncols: int = 8,
    dpi: float = 100,
) -> None:
    unnormalizer = UnnormalizeAttributes(image_size=image_size)
    attributes = unnormalizer(tensor_to_attribute(samples))

    categories = attributes.category.cpu().numpy()
    locations = torch.stack([attributes.x, attributes.y], dim=1).cpu().numpy()
    colors = (
        (
            torch.stack(
                [
                    attributes.color_r,
                    attributes.color_g,
                    attributes.color_b,
                ],
                dim=1,
            )
        )
        .cpu()
        .numpy()
    )
    sizes = attributes.size.cpu().numpy()
    rotations = attributes.rotation.cpu().numpy()

    image = figure_grid(
        categories, locations, sizes, rotations, colors, image_size, ncols, dpi
    )
    logger.log_image(key=key, images=[image])


class LogAttributesCallback(LogSamplesCallback):
    def __init__(
        self,
        reference_samples: Sequence[torch.Tensor],
        log_key: str,
        image_size: int,
        every_n_epochs: int | None = 1,
        ncols: int = 8,
        dpi: float = 100,
    ) -> None:
        super().__init__(reference_samples, log_key, every_n_epochs)
        self.image_size = image_size
        self.ncols = ncols
        self.dpi = dpi

    def to(
        self, samples: Sequence[torch.Tensor], device: torch.device
    ) -> list[torch.Tensor]:
        return [x.to(device) for x in samples]

    def log_samples(
        self, logger: Logger, samples: Sequence[torch.Tensor], mode: str
    ) -> None:
        if not isinstance(logger, WandbLogger):
            logging.warning("Only logging to wandb is supported")
            return

        log_attribute(
            logger,
            samples,
            image_size=self.image_size,
            key=f"{self.log_key}_{mode}",
            ncols=self.ncols,
            dpi=self.dpi,
        )


class LogVisualCallback(LogSamplesCallback):
    def __init__(
        self,
        reference_samples: torch.Tensor,
        log_key: str,
        every_n_epochs: int | None = 1,
        ncols: int = 8,
    ) -> None:
        super().__init__(reference_samples, log_key, every_n_epochs)
        self.ncols = ncols

    def log_samples(
        self, logger: Logger, samples: torch.Tensor, mode: str
    ) -> None:
        if not isinstance(logger, WandbLogger):
            logging.warning("Only logging to wandb is supported")
            return

        images = make_grid(samples, nrow=self.ncols, pad_value=1)
        logger.log_image(key=f"{self.log_key}_{mode}", images=[images])
