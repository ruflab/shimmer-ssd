import io
import logging
from collections.abc import Mapping, Sequence
from typing import Any, cast

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
from simple_shapes_dataset.modules.domains.visual import (
    VisualLatentDomainModule,
)
from simple_shapes_dataset.modules.global_workspace import (
    DeterministicGlobalWorkspaceLightningModule,
    VariationalGlobalWorkspaceLightningModule,
)

matplotlib.use("Agg")
GlobalWorkspaceLightningModuleType = (
    DeterministicGlobalWorkspaceLightningModule
    | VariationalGlobalWorkspaceLightningModule
)


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
        self,
        current_epoch: int,
        loggers: Sequence[Logger],
        pl_module: pl.LightningModule,
    ) -> None:
        if not (len(loggers)) is None:
            return

        samples = self.to(self.reference_samples, pl_module.device)
        if current_epoch == 0:
            for logger in loggers:
                self.log_samples(logger, samples, "reference")

        with torch.no_grad():
            pl_module.eval()
            generated_samples = pl_module(samples)
            pl_module.train()

        for logger in loggers:
            self.log_samples(logger, generated_samples, "prediction")

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if (
            self.every_n_epochs is None
            or trainer.current_epoch % self.every_n_epochs != 0
        ):
            return

        return self.on_callback(
            trainer.current_epoch, trainer.loggers, pl_module
        )

    def on_fit_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        return self.on_callback(
            trainer.current_epoch, trainer.loggers, pl_module
        )

    def log_samples(self, logger: Logger, samples: Any, mode: str) -> None:
        raise NotImplementedError


def get_pil_image(figure: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    figure.savefig(buf)
    buf.seek(0)
    return Image.open(buf)


def get_attribute_figure_grid(
    categories: np.ndarray,
    locations: np.ndarray,
    sizes: np.ndarray,
    rotations: np.ndarray,
    colors: np.ndarray,
    image_size: int,
    ncols: int = 8,
    padding: float = 2,
) -> Image.Image:
    reminder = 1 if categories.shape[0] % ncols else 0
    nrows = categories.shape[0] // ncols + reminder

    width = ncols * (image_size + padding) + padding
    height = nrows * (image_size + padding) + padding
    dpi = 1

    figure = plt.figure(
        figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white"
    )
    gs = gridspec.GridSpec(
        nrows,
        ncols,
        wspace=padding / image_size,
        hspace=padding / image_size,
        left=padding / width,
        right=1 - padding / width,
        bottom=padding / height,
        top=1 - padding / height,
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


def make_attribute_grid(
    samples: Sequence[torch.Tensor],
    image_size: int,
    ncols: int = 8,
    padding: float = 2,
) -> Image.Image:
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

    return get_attribute_figure_grid(
        categories,
        locations,
        sizes,
        rotations,
        colors,
        image_size,
        ncols,
        padding,
    )


class LogAttributesCallback(LogSamplesCallback):
    def __init__(
        self,
        reference_samples: Sequence[torch.Tensor],
        log_key: str,
        image_size: int,
        every_n_epochs: int | None = 1,
        ncols: int = 8,
    ) -> None:
        super().__init__(reference_samples, log_key, every_n_epochs)
        self.image_size = image_size
        self.ncols = ncols

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

        image = make_attribute_grid(
            samples,
            image_size=self.image_size,
            ncols=self.ncols,
        )
        logger.log_image(key=f"{self.log_key}_{mode}", images=[image])


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


class LogGWImagesCallback(pl.Callback):
    def __init__(
        self,
        reference_samples: Mapping[frozenset[str], Mapping[str, Any]],
        log_key: str,
        every_n_epochs: int | None = 1,
        image_size: int = 32,
        ncols: int = 8,
    ) -> None:
        super().__init__()
        self.reference_samples = reference_samples
        self.every_n_epochs = every_n_epochs
        self.log_key = log_key
        self.image_size = image_size
        self.ncols = ncols

    def to(
        self,
        samples: Mapping[
            frozenset[str], Mapping[str, torch.Tensor | Sequence[torch.Tensor]]
        ],
        device: torch.device,
    ) -> dict[
        frozenset[str], dict[str, torch.Tensor | Sequence[torch.Tensor]]
    ]:
        out: dict[
            frozenset[str], dict[str, torch.Tensor | Sequence[torch.Tensor]]
        ] = {}
        for domain_names, domains in samples.items():
            latents: dict[str, torch.Tensor | Sequence[torch.Tensor]] = {}
            for domain_name, domain in domains.items():
                if isinstance(domain, torch.Tensor):
                    latents[domain_name] = domain.to(device)
                else:
                    latents[domain_name] = [x.to(device) for x in domain]
            out[domain_names] = latents
        return out

    def on_callback(
        self,
        current_epoch: int,
        loggers: Sequence[Logger],
        pl_module: GlobalWorkspaceLightningModuleType,
    ) -> None:
        if not (len(loggers)):
            return

        samples = self.to(self.reference_samples, pl_module.device)
        if current_epoch == 0:
            for domain_names, domains in samples.items():
                for domain_name, domain in domains.items():
                    for logger in loggers:
                        self.log_samples(
                            logger,
                            pl_module,
                            domain,
                            domain_name,
                            f"ref_{'-'.join(domain_names)}_{domain_name}",
                        )

        latents = pl_module.encode_domains(samples)

        with torch.no_grad():
            pl_module.eval()
            prediction_demi_cycles = pl_module.demi_cycle(latents)
            prediction_cycles = pl_module.cycle(latents)
            prediction_translations = pl_module.translation(latents)
            pl_module.train()

        for logger in loggers:
            for domain, prediction in prediction_demi_cycles.items():
                self.log_samples(
                    logger,
                    pl_module,
                    pl_module.decode_domain(prediction, domain),
                    domain,
                    f"pred_dcy_{domain}",
                )
            for (domain_s, domain_t), prediction in prediction_cycles.items():
                self.log_samples(
                    logger,
                    pl_module,
                    pl_module.decode_domain(prediction, domain_s),
                    domain_s,
                    f"pred_cy_{domain_s}_in_{domain_t}",
                )
            for (
                domain_s,
                domain_t,
            ), prediction in prediction_translations.items():
                self.log_samples(
                    logger,
                    pl_module,
                    pl_module.decode_domain(prediction, domain_t),
                    domain_t,
                    f"pred_trans_{domain_s}_to_{domain_t}",
                )

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: GlobalWorkspaceLightningModuleType,
    ) -> None:
        if (
            self.every_n_epochs is None
            or trainer.current_epoch % self.every_n_epochs != 0
        ):
            return

        return self.on_callback(
            trainer.current_epoch, trainer.loggers, pl_module
        )

    def on_fit_end(
        self,
        trainer: pl.Trainer,
        pl_module: GlobalWorkspaceLightningModuleType,
    ) -> None:
        return self.on_callback(
            trainer.current_epoch, trainer.loggers, pl_module
        )

    def log_samples(
        self,
        logger: Logger,
        pl_module: GlobalWorkspaceLightningModuleType,
        samples: Any,
        domain: str,
        mode: str,
    ) -> None:
        if not isinstance(logger, WandbLogger):
            logging.warning("Only logging to wandb is supported")
            return

        match domain:
            case "v":
                self.log_visual_samples(logger, samples, mode)
            case "v_latents":
                assert "v_latents" in pl_module.domain_modules

                module = cast(
                    VisualLatentDomainModule,
                    pl_module.domain_modules["v_latents"],
                )
                self.log_visual_samples(
                    logger, module.decode_images(samples), mode
                )
            case "attr":
                self.log_attribute_samples(logger, samples, mode)

    def log_visual_samples(
        self,
        logger: WandbLogger,
        samples: Any,
        mode: str,
    ) -> None:
        images = make_grid(samples, nrow=self.ncols, pad_value=1)
        logger.log_image(key=f"{self.log_key}/{mode}", images=[images])

    def log_attribute_samples(
        self,
        logger: WandbLogger,
        samples: Any,
        mode: str,
    ) -> None:
        image = make_attribute_grid(
            samples,
            image_size=self.image_size,
            ncols=self.ncols,
        )
        logger.log_image(key=f"{self.log_key}/{mode}", images=[image])
