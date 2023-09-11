from collections.abc import Mapping
from typing import Any

import torch
from shimmer.modules.domain import DomainModule
from shimmer.modules.vae import VAE, gaussian_nll, kl_divergence_loss
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import OneCycleLR

from simple_shapes_dataset import LOGGER
from simple_shapes_dataset.modules.losses import margin_loss
from simple_shapes_dataset.modules.vae import RAEDecoder, RAEEncoder


class VisualDomainModule(DomainModule):
    def __init__(
        self,
        num_channels: int,
        latent_dim: int,
        ae_dim: int,
        beta: float = 1,
        optim_lr: float = 1e-3,
        optim_weight_decay: float = 0,
        scheduler_args: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        vae_encoder = RAEEncoder(
            num_channels,
            ae_dim,
            latent_dim,
            use_batchnorm=True,
        )
        vae_decoder = RAEDecoder(
            num_channels,
            latent_dim,
            ae_dim,
        )
        self.vae = VAE(vae_encoder, vae_decoder, beta)
        self.latent_dim = latent_dim
        self.optim_lr = optim_lr
        self.optim_weight_decay = optim_weight_decay
        self.scheduler_args: dict[str, Any] = {
            "max_lr": optim_lr,
            "total_steps": 1,
        }
        self.scheduler_args.update(scheduler_args or {})

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.encode((x,))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.vae.decode(z)[0]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def generic_step(
        self,
        x: torch.Tensor,
        mode: str = "train",
    ) -> torch.Tensor:
        (mean, logvar), reconstruction = self.vae((x,))

        reconstruction_loss = gaussian_nll(
            reconstruction[0], torch.tensor(0), x
        ).sum()

        kl_loss = kl_divergence_loss(mean, logvar)
        total_loss = reconstruction_loss + self.vae.beta * kl_loss

        self.log(f"{mode}/reconstruction_loss", reconstruction_loss)
        self.log(f"{mode}/kl_loss", kl_loss)
        self.log(f"{mode}/loss", total_loss)
        return total_loss

    def validation_step(
        self,
        batch: Mapping[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x = batch["v"]
        return self.generic_step(x, "val")

    def training_step(
        self,
        batch: Mapping[frozenset[str], Mapping[str, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        x = batch[frozenset(["v"])]["v"]
        return self.generic_step(x, "train")

    def configure_optimizers(
        self,
    ) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optim_lr,
            weight_decay=self.optim_weight_decay,
        )
        lr_scheduler = OneCycleLR(optimizer, **self.scheduler_args)

        # FIXME: This might trigger a warning if mixed-precision, as per
        # https://github.com/Lightning-AI/lightning/issues/5558

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }


class VisualLatentDomainModule(DomainModule):
    def __init__(self, visual_module: VisualDomainModule):
        super().__init__()
        self.visual_module = visual_module
        self.latent_dim = self.visual_module.latent_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :-1]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        extra = torch.zeros_like(z[:, -1]).unsqueeze(1)
        return torch.cat([z, extra], dim=1)

    def decode_images(self, z: torch.Tensor) -> torch.Tensor:
        LOGGER.debug(
            f"VisualLatentDomainModule.decode_images: z.shape = {z.size()}"
        )
        return self.visual_module.decode(z[:, :-1])


class VisualLatentDomainWithUnpairedModule(DomainModule):
    def __init__(self, visual_module: VisualDomainModule):
        super().__init__()
        self.visual_module = visual_module
        self.latent_dim = self.visual_module.latent_dim + 1

    def on_before_gw_encode_cont(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        out[:, -1] = 0
        return out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {
            "loss": mse_loss(pred, target, reduction="sum"),
            "unpaired": mse_loss(pred[:, -1], target[:, -1]),
            "other": mse_loss(pred[:, 0], target[:, 0]),
        }

    def compute_tr_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        loss = margin_loss(
            pred,
            target,
            margin=1.0 / 11.0,
            reduction="none",
        )
        return {
            "loss": loss[:, :-1].sum(),
            "unpaired": loss[:, -1].sum(),
            "other": loss[:, 0].sum(),
        }

    def decode_images(self, z: torch.Tensor) -> torch.Tensor:
        LOGGER.debug(
            f"VisualLatentDomainModule.decode_images: z.shape = {z.size()}"
        )
        return self.visual_module.decode(z[:, :-1])
