from collections.abc import Mapping, Sequence
from typing import Any

import torch
from shimmer.modules.domain import DomainModule
from shimmer.modules.vae import VAE, gaussian_nll, kl_divergence_loss
from torch.optim.lr_scheduler import OneCycleLR

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

        self.optim_lr = optim_lr
        self.optim_weight_decay = optim_weight_decay
        self.scheduler_args = scheduler_args or {}

    def encode(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.vae.encode(x)

    def decode(self, z: torch.Tensor) -> Sequence[torch.Tensor]:
        return self.vae.decode(z)

    def forward(self, x: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        return self.decode(self.encode(x))

    def generic_step(
        self,
        x: Sequence[torch.Tensor],
        mode: str = "train",
    ) -> torch.Tensor:
        x_images = x[0]

        (mean, logvar), reconstruction = self.vae(x)
        reconstruction_images = reconstruction[1]

        reconstruction_loss = gaussian_nll(
            reconstruction_images, torch.tensor(0), x_images
        ).sum()

        kl_loss = kl_divergence_loss(mean, logvar)
        total_loss = reconstruction_loss + self.vae.beta * kl_loss

        self.log(f"{mode}/reconstruction_loss", reconstruction_loss)
        self.log(f"{mode}/kl_loss", kl_loss)
        self.log(f"{mode}/loss", total_loss)
        return total_loss

    def validation_step(
        self,
        batch: Mapping[str, Sequence[torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        x = batch["v"]
        return self.generic_step(x, "val")

    def training_step(
        self,
        batch: Mapping[frozenset[str], Mapping[str, Sequence[torch.Tensor]]],
        batch_idx: int,
    ) -> torch.Tensor:
        x = batch[frozenset(["v"])]["v"]
        return self.generic_step(x, "train")

    def configure_optimizers(
        self,
    ) -> dict[str, torch.optim.Adam | torch.optim.lr_scheduler.LRScheduler]:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.optim_lr,
            weight_decay=self.optim_weight_decay,
        )
        lr_scheduler = OneCycleLR(optimizer, **self.scheduler_args)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
