import io
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Generic, Literal, TypeVar, cast

import lightning.pytorch as pl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.loggers import Logger, TensorBoardLogger
from lightning.pytorch.loggers.wandb import WandbLogger
from matplotlib import gridspec
from matplotlib.figure import Figure
from PIL import Image
from shimmer.modules.global_workspace import GlobalWorkspaceBase, GWPredictionsBase
from simple_shapes_dataset import (
    UnnormalizeAttributes,
    tensor_to_attribute,
)
from simple_shapes_dataset.cli import generate_image
from tokenizers.implementations import ByteLevelBPETokenizer
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid

from shimmer_ssd import LOGGER
from shimmer_ssd.modules.domains.text import GRUTextDomainModule, Text2Attr
from shimmer_ssd.modules.domains.visual import VisualLatentDomainModule

matplotlib.use("Agg")

_T = TypeVar("_T")


def log_image(
    logger: Logger, key: str, image: torch.Tensor | Image.Image, step: int | None = None
):
    if isinstance(logger, WandbLogger):
        logger.log_image(key, [image], step)
    elif isinstance(logger, TensorBoardLogger):
        torch_image = to_tensor(image) if isinstance(image, Image.Image) else image
        logger.experiment.add_image(key, torch_image, step)
    else:
        LOGGER.warning(
            "[Sample Logger] Only logging to tensorboard or wandb is supported"
        )
        return


def log_text(
    logger: Logger,
    key: str,
    columns: list[str],
    data: list[list[str]],
    step: int | None = None,
):
    if isinstance(logger, WandbLogger):
        logger.log_text(key, columns, data, step=step)
    elif isinstance(logger, TensorBoardLogger):
        text = ", ".join(columns) + "\n"
        text += "\n".join([", ".join(d) for d in data])
        logger.experiment.add_image(key, text, step)
    else:
        LOGGER.warning(
            "[Sample Logger] Only logging to tensorboard or wandb is supported"
        )
        return


class LogSamplesCallback(Generic[_T], ABC, pl.Callback):
    def __init__(
        self,
        reference_samples: _T,
        log_key: str,
        mode: Literal["train", "val", "test"],
        every_n_epochs: int | None = 1,
    ) -> None:
        super().__init__()
        self.reference_samples = reference_samples
        self.every_n_epochs = every_n_epochs
        self.log_key = log_key
        self.mode = mode

    def to(self, samples: _T, device: torch.device) -> _T:
        raise NotImplementedError

    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        if stage != "fit":
            return
        device = trainer.strategy.root_device
        self.reference_samples = self.to(self.reference_samples, device)
        for logger in trainer.loggers:
            self.log_samples(logger, self.reference_samples, "reference")

    def on_callback(
        self,
        loggers: Sequence[Logger],
        pl_module: pl.LightningModule,
    ) -> None:
        if not len(loggers):
            LOGGER.debug("[LOGGER] No logger found.")
            return

        samples = self.to(self.reference_samples, pl_module.device)

        with torch.no_grad():
            pl_module.eval()
            generated_samples = pl_module(samples)
            pl_module.train()

        for logger in loggers:
            self.log_samples(logger, generated_samples, "prediction")

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.mode != "train":
            return

        if (
            self.every_n_epochs is None
            or trainer.current_epoch % self.every_n_epochs != 0
        ):
            LOGGER.debug("[LOGGER] on_train_epoch_end")
            return

        LOGGER.debug("[LOGGER] on_train_epoch_end called")
        return self.on_callback(trainer.loggers, pl_module)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.mode == "test":
            return

        return self.on_callback(trainer.loggers, pl_module)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.mode != "val":
            return

        if (
            self.every_n_epochs is None
            or trainer.current_epoch % self.every_n_epochs != 0
        ):
            return

        return self.on_callback(trainer.loggers, pl_module)

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.mode != "test":
            return

        return self.on_callback(trainer.loggers, pl_module)

    @abstractmethod
    def log_samples(self, logger: Logger, samples: _T, mode: str) -> None: ...


def get_pil_image(figure: Figure) -> Image.Image:
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

    figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
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
    image = get_pil_image(figure)
    plt.close(figure)
    return image


def attribute_image_grid(
    samples: Sequence[torch.Tensor],
    image_size: int,
    ncols: int,
) -> Image.Image:
    unnormalizer = UnnormalizeAttributes(image_size=image_size)
    attributes = unnormalizer(tensor_to_attribute(samples))

    categories = attributes.category.detach().cpu().numpy()
    locations = torch.stack([attributes.x, attributes.y], dim=1).detach().cpu().numpy()
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
    sizes = attributes.size.detach().cpu().numpy()
    rotations = attributes.rotation.detach().cpu().numpy()

    return get_attribute_figure_grid(
        categories,
        locations,
        sizes,
        rotations,
        colors,
        image_size,
        ncols,
        padding=2,
    )


class LogAttributesCallback(LogSamplesCallback[Sequence[torch.Tensor]]):
    def __init__(
        self,
        reference_samples: Sequence[torch.Tensor],
        log_key: str,
        mode: Literal["train", "val", "test"],
        image_size: int,
        every_n_epochs: int | None = 1,
        ncols: int = 8,
    ) -> None:
        super().__init__(reference_samples, log_key, mode, every_n_epochs)
        self.image_size = image_size
        self.ncols = ncols

    def to(
        self, samples: Sequence[torch.Tensor], device: torch.device
    ) -> list[torch.Tensor]:
        return [x.to(device) for x in samples]

    def log_samples(
        self, logger: Logger, samples: Sequence[torch.Tensor], mode: str
    ) -> None:
        image = attribute_image_grid(
            samples, image_size=self.image_size, ncols=self.ncols
        )
        log_image(logger, f"{self.log_key}_{mode}", image)


class LogTextCallback(LogSamplesCallback[Mapping[str, torch.Tensor]]):
    def __init__(
        self,
        reference_samples: Mapping[str, torch.Tensor],
        log_key: str,
        mode: Literal["train", "val", "test"],
        image_size: int,
        vocab: str,
        merges: str,
        every_n_epochs: int | None = 1,
        ncols: int = 8,
    ) -> None:
        super().__init__(reference_samples, log_key, mode, every_n_epochs)
        self.image_size = image_size
        self.ncols = ncols
        self.tokenizer = ByteLevelBPETokenizer(vocab, merges)

    def to(
        self, samples: Mapping[str, torch.Tensor], device: torch.device
    ) -> dict[str, torch.Tensor]:
        return {x: samples[x].to(device) for x in samples}

    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        if stage != "fit":
            return
        assert isinstance(pl_module, GRUTextDomainModule)
        device = trainer.strategy.root_device
        self.reference_samples = self.to(self.reference_samples, device)
        for logger in trainer.loggers:
            self.log_samples(logger, self.reference_samples, "reference")

    def on_callback(
        self,
        loggers: Sequence[Logger],
        pl_module: pl.LightningModule,
    ) -> None:
        assert isinstance(pl_module, GRUTextDomainModule)

        samples = self.to(self.reference_samples, pl_module.device)

        if not len(loggers):
            LOGGER.debug("[LOGGER] No logger found.")
            return

        with torch.no_grad():
            pl_module.eval()
            generated_samples = pl_module(samples)
            pl_module.train()

        for logger in loggers:
            self.log_samples(logger, generated_samples, "prediction")

    def log_samples(
        self, logger: Logger, samples: Mapping[str, torch.Tensor], mode: str
    ) -> None:
        if not isinstance(logger, WandbLogger):
            LOGGER.warning("Only logging to wandb is supported")
            return

        assert self.tokenizer is not None
        text = self.tokenizer.decode_batch(
            samples["tokens"].detach().cpu().tolist(), skip_special_tokens=True
        )
        text = [[t.replace("<pad>", "")] for t in text]
        log_text(logger, f"{self.log_key}_{mode}_str", ["text"], text)


class LogVisualCallback(LogSamplesCallback[torch.Tensor]):
    def __init__(
        self,
        reference_samples: torch.Tensor,
        log_key: str,
        mode: Literal["train", "val", "test"],
        every_n_epochs: int | None = 1,
        ncols: int = 8,
    ) -> None:
        super().__init__(reference_samples, log_key, mode, every_n_epochs)
        self.ncols = ncols

    def to(self, samples: torch.Tensor, device: torch.device) -> torch.Tensor:
        return samples.to(device)

    def log_samples(self, logger: Logger, samples: torch.Tensor, mode: str) -> None:
        images = make_grid(samples, nrow=self.ncols, pad_value=1)
        log_image(logger, f"{self.log_key}_{mode}", images)


class LogText2AttrCallback(
    LogSamplesCallback[
        Mapping[str, Mapping[str, torch.Tensor] | Sequence[torch.Tensor]]
    ]
):
    def __init__(
        self,
        reference_samples: Mapping[
            str, Mapping[str, torch.Tensor] | Sequence[torch.Tensor]
        ],
        log_key: str,
        mode: Literal["train", "val", "test"],
        every_n_epochs: int | None = 1,
        image_size: int = 32,
        ncols: int = 8,
        vocab: str | None = None,
        merges: str | None = None,
    ) -> None:
        super().__init__(reference_samples, log_key, mode, every_n_epochs)
        self.image_size = image_size
        self.ncols = ncols
        self.tokenizer = ByteLevelBPETokenizer(vocab, merges)
        self.reference_samples = reference_samples

    def to(
        self,
        samples: Mapping[str, Mapping[str, torch.Tensor] | Sequence[torch.Tensor]],
        device: torch.device,
    ) -> dict[str, dict[str, torch.Tensor] | list[torch.Tensor]]:
        latents: dict[str, dict[str, torch.Tensor] | list[torch.Tensor]] = {}
        for domain_name, domain in samples.items():
            if isinstance(domain, dict):
                latents[domain_name] = {k: x.to(device) for k, x in domain.items()}
            elif isinstance(domain, list):
                latents[domain_name] = [x.to(device) for x in domain]
        return latents

    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        if stage != "fit":
            return
        assert isinstance(pl_module, Text2Attr)
        device = trainer.strategy.root_device
        self.reference_samples = self.to(self.reference_samples, device)
        for logger in trainer.loggers:
            self.log_samples(logger, self.reference_samples, "reference")

    def on_callback(
        self,
        loggers: Sequence[Logger],
        pl_module: pl.LightningModule,
    ) -> None:
        assert isinstance(pl_module, Text2Attr)

        samples = self.to(self.reference_samples, pl_module.device)

        if not len(loggers):
            LOGGER.debug("[LOGGER] No logger found.")
            return

        with torch.no_grad():
            pl_module.eval()
            generated_samples = pl_module(samples["t"])
            pl_module.train()

        for logger in loggers:
            self.log_samples(logger, generated_samples, "prediction")

    def log_samples(
        self,
        logger: Logger,
        samples: Mapping[str, Mapping[str, torch.Tensor] | Sequence[torch.Tensor]],
        mode: str,
    ) -> None:
        for domain_name, domain in samples.items():
            if domain_name == "t":
                assert self.tokenizer is not None
                assert isinstance(domain, dict)
                text = self.tokenizer.decode_batch(
                    domain["tokens"].detach().cpu().tolist(), skip_special_tokens=True
                )
                text = [[t.replace("<pad>", "")] for t in text]
                log_text(logger, f"{self.log_key}_{mode}_str", ["text"], text)
            elif domain_name == "attr":
                assert isinstance(domain, list)
                image = attribute_image_grid(
                    domain,
                    image_size=self.image_size,
                    ncols=self.ncols,
                )
                log_image(logger, f"{self.log_key}_{mode}", image)


def batch_to_device(
    samples: Mapping[
        frozenset[str],
        Mapping[str, Any],
    ],
    device: torch.device,
) -> dict[frozenset[str], dict[str, Any]]:
    out: dict[frozenset[str], dict[str, Any]] = {}
    for domain_names, domains in samples.items():
        latents: dict[str, Any] = {}
        for domain_name, domain in domains.items():
            if isinstance(domain, torch.Tensor):
                latents[domain_name] = domain.to(device)
            elif (
                isinstance(domain, Mapping)
                and len(domain)
                and isinstance(next(iter(domain.values())), torch.Tensor)
            ):
                latents[domain_name] = {k: x.to(device) for k, x in domain.items()}
            elif (
                isinstance(domain, Sequence)
                and len(domain)
                and isinstance(domain[0], torch.Tensor)
            ):
                latents[domain_name] = [x.to(device) for x in domain]
            else:
                latents[domain_name] = domain
        out[domain_names] = latents
    return out


class LogGWImagesCallback(pl.Callback):
    def __init__(
        self,
        reference_samples: Mapping[frozenset[str], Mapping[str, Any]],
        log_key: str,
        mode: Literal["train", "val", "test"],
        every_n_epochs: int | None = 1,
        image_size: int = 32,
        ncols: int = 8,
        filter: Sequence[str] | None = None,
        vocab: str | None = None,
        merges: str | None = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.reference_samples = reference_samples
        self.every_n_epochs = every_n_epochs
        self.log_key = log_key
        self.image_size = image_size
        self.ncols = ncols
        self.filter = filter
        self.tokenizer = None
        if vocab is not None and merges is not None:
            self.tokenizer = ByteLevelBPETokenizer(vocab, merges)

    def to(
        self,
        samples: Mapping[
            frozenset[str],
            Mapping[
                str, torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor]
            ],
        ],
        device: torch.device,
    ) -> dict[
        frozenset[str],
        dict[str, torch.Tensor | list[torch.Tensor] | dict[Any, torch.Tensor]],
    ]:
        return batch_to_device(samples, device)

    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        if stage != "fit":
            return
        assert isinstance(pl_module, GlobalWorkspaceBase)
        device = trainer.strategy.root_device
        self.reference_samples = self.to(self.reference_samples, device)

        for domain_names, domains in self.reference_samples.items():
            for domain_name, domain_tensor in domains.items():
                for logger in trainer.loggers:
                    self.log_samples(
                        logger,
                        pl_module,
                        domain_tensor,
                        domain_name,
                        f"ref_{'-'.join(domain_names)}_{domain_name}",
                    )

    def on_callback(
        self,
        loggers: Sequence[Logger],
        pl_module: GlobalWorkspaceBase,
    ) -> None:
        if not (len(loggers)):
            return

        with torch.no_grad():
            latent_groups = pl_module.encode_domains(self.reference_samples)
            predictions = cast(GWPredictionsBase, pl_module(latent_groups))

            for logger in loggers:
                for domains, preds in predictions["broadcasts"].items():
                    domain_from = ",".join(domains)
                    for domain, pred in preds.items():
                        log_name = f"pred_trans_{domain_from}_to_{domain}"
                        if self.filter is not None and log_name not in self.filter:
                            continue
                        self.log_samples(
                            logger,
                            pl_module,
                            pl_module.decode_domain(pred, domain),
                            domain,
                            log_name,
                        )
                for domains, preds in predictions["cycles"].items():
                    domain_from = ",".join(domains)
                    for domain, pred in preds.items():
                        log_name = f"pred_cycle_{domain_from}_to_{domain}"
                        if self.filter is not None and log_name not in self.filter:
                            continue
                        self.log_samples(
                            logger,
                            pl_module,
                            pl_module.decode_domain(pred, domain),
                            domain,
                            log_name,
                        )

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self.mode != "train":
            return

        if not isinstance(pl_module, GlobalWorkspaceBase):
            return

        if (
            self.every_n_epochs is None
            or trainer.current_epoch % self.every_n_epochs != 0
        ):
            return

        return self.on_callback(trainer.loggers, pl_module)

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self.mode != "val":
            return

        if not isinstance(pl_module, GlobalWorkspaceBase):
            return

        if (
            self.every_n_epochs is None
            or trainer.current_epoch % self.every_n_epochs != 0
        ):
            return

        return self.on_callback(trainer.loggers, pl_module)

    def on_test_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self.mode != "test":
            return

        if not isinstance(pl_module, GlobalWorkspaceBase):
            return

        return self.on_callback(trainer.loggers, pl_module)

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self.mode == "test":
            return

        if not isinstance(pl_module, GlobalWorkspaceBase):
            return

        return self.on_callback(trainer.loggers, pl_module)

    def log_samples(
        self,
        logger: Logger,
        pl_module: GlobalWorkspaceBase,
        samples: Any,
        domain: str,
        mode: str,
    ) -> None:
        match domain:
            case "v":
                self.log_visual_samples(logger, samples, mode)
            case "v_latents":
                assert "v_latents" in pl_module.domain_mods

                module = cast(
                    VisualLatentDomainModule,
                    pl_module.domain_mods["v_latents"],
                )
                self.log_visual_samples(logger, module.decode_images(samples), mode)
            case "attr":
                self.log_attribute_samples(logger, samples, mode)
            case "t":
                self.log_text_samples(logger, samples, mode)
                if "attr" in samples:
                    self.log_attribute_samples(logger, samples["attr"], mode + "_attr")

    def log_visual_samples(
        self,
        logger: Logger,
        samples: Any,
        mode: str,
    ) -> None:
        images = make_grid(samples, nrow=self.ncols, pad_value=1)
        log_image(logger, f"{self.log_key}/{mode}", images)

    def log_attribute_samples(
        self,
        logger: Logger,
        samples: Any,
        mode: str,
    ) -> None:
        image = attribute_image_grid(
            samples,
            image_size=self.image_size,
            ncols=self.ncols,
        )
        log_image(logger, f"{self.log_key}/{mode}", image)

    def log_text_samples(
        self,
        logger: Logger,
        samples: Any,
        mode: str,
    ) -> None:
        assert self.tokenizer is not None
        text = self.tokenizer.decode_batch(
            samples["tokens"].detach().cpu().tolist(), skip_special_tokens=True
        )
        text = [[t.replace("<pad>", "")] for t in text]
        log_text(logger, f"{self.log_key}/{mode}", ["text"], text)
