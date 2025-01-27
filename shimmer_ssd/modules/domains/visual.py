from collections.abc import Mapping
from typing import Any

import torch
from shimmer import LossOutput
from shimmer.modules.domain import DomainModule
from shimmer.modules.vae import VAE, gaussian_nll, kl_divergence_loss
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import OneCycleLR

from shimmer_ssd import LOGGER
from shimmer_ssd.modules.vae import RAEDecoder, RAEEncoder


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
        """
        Visual domain module. This defines shimmer's `DomainModule` for the vision
        side with a VAE.

        Args:
            num_channels (`int`): number of input channels (for RGB image, use 3)
            latent_dim (`int`): latent dimension of the vision domain
            ae_dim (`int`): internal auto-encoder dimension of the VAE
            beta (`float`): beta value if beta-VAE. (Defaults to 1.0)
            optim_lr (`float`): training learning rate
            optim_weight_decay (`float`): training weight decay
            scheduler_args (`Mapping[str, Any] | None`): Args for the scheduler.
        """

        super().__init__(latent_dim)
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
        self.optim_lr = optim_lr
        self.optim_weight_decay = optim_weight_decay
        self.scheduler_args: dict[str, Any] = {
            "max_lr": optim_lr,
            "total_steps": 1,
        }
        self.scheduler_args.update(scheduler_args or {})

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, raw_target: Any
    ) -> LossOutput:
        return LossOutput(mse_loss(pred, target, reduction="mean"))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.encode((x,))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.vae.decode(z)[0]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.decode(self.encode(x))

    def generic_step(
        self,
        x: torch.Tensor,
        mode: str = "train",
    ) -> torch.Tensor:
        (mean, logvar), reconstruction = self.vae((x,))

        reconstruction_loss = gaussian_nll(reconstruction[0], torch.tensor(0), x).sum()

        kl_loss = kl_divergence_loss(mean, logvar)
        total_loss = reconstruction_loss + self.vae.beta * kl_loss

        self.log(f"{mode}/reconstruction_loss", reconstruction_loss)
        self.log(f"{mode}/kl_loss", kl_loss)
        self.log(f"{mode}/loss", total_loss)
        return total_loss

    def validation_step(  # type: ignore
        self,
        batch: Mapping[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x = batch["v"]
        return self.generic_step(x, "val")

    def training_step(  # type: ignore
        self,
        batch: Mapping[frozenset[str], Mapping[str, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        x = batch[frozenset(["v"])]["v"]
        return self.generic_step(x, "train")

    def configure_optimizers(  # type: ignore
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
        super().__init__(visual_module.latent_dim)
        self.visual_module = visual_module

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, raw_target: Any
    ) -> LossOutput:
        return LossOutput(mse_loss(pred, target, reduction="mean"))

    def decode_images(self, z: torch.Tensor) -> torch.Tensor:
        LOGGER.debug(f"VisualLatentDomainModule.decode_images: z.shape = {z.size()}")
        return self.visual_module.decode(z)


class VisualLatentDomainWithUnpairedModule(DomainModule):
    def __init__(self, visual_module: VisualDomainModule, coef_unpaired: float = 0.5):
        super().__init__(visual_module.latent_dim + 1)

        if coef_unpaired < 0 or coef_unpaired > 1:
            raise ValueError("coef_unpaired should be in [0, 1]")

        self.visual_module = visual_module
        self.paired_dim = self.visual_module.latent_dim
        self.coef_unpaired = coef_unpaired

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, raw_target: Any
    ) -> LossOutput:
        paired_loss = mse_loss(pred[:, : self.paired_dim], target[:, : self.paired_dim])
        unpaired_loss = mse_loss(
            pred[:, self.paired_dim :], target[:, self.paired_dim :]
        )
        total_loss = (
            self.coef_unpaired * unpaired_loss + (1 - self.coef_unpaired) * paired_loss
        )
        return LossOutput(
            loss=total_loss,
            metrics={
                "unpaired": unpaired_loss,
                "paired": paired_loss,
            },
        )

    def decode_images(self, z: torch.Tensor) -> torch.Tensor:
        LOGGER.debug(f"VisualLatentDomainModule.decode_images: z.shape = {z.size()}")
        return self.visual_module.decode(z[:, :-1])
