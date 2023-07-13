from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from shimmer.modules.domain import DomainModule
from shimmer.modules.vae import VAE, gaussian_nll, kl_divergence_loss
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR


class VAEEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.encoder = nn.Sequential(
            nn.Linear(11, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

        self.q_mean = nn.Linear(self.out_dim, self.out_dim)
        self.q_logvar = nn.Linear(self.out_dim, self.out_dim)

    def forward(
        self, x: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = torch.cat(x, dim=-1)
        out = self.encoder(out)
        return self.q_mean(out), self.q_logvar(out)


class VAEDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.decoder = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

        self.decoder_categories = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 3),
        )

        self.decoder_attributes = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 8),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> list[torch.Tensor]:
        out = self.decoder(z)
        return [self.decoder_categories(out), self.decoder_attributes(out)]


class AttributeDomainModule(DomainModule):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        beta: float = 1,
        coef_categories: float = 1,
        coef_attributes: float = 1,
        optim_lr: float = 1e-3,
        optim_weight_decay: float = 0,
        scheduler_args: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.in_dim = 11
        self.hidden_dim = hidden_dim
        self.coef_categories = coef_categories
        self.coef_attributes = coef_attributes

        vae_encoder = VAEEncoder(self.hidden_dim, self.latent_dim)
        vae_decoder = VAEDecoder(self.latent_dim, self.hidden_dim)
        self.vae = VAE(vae_encoder, vae_decoder, beta)

        self.optim_lr = optim_lr
        self.optim_weight_decay = optim_weight_decay
        self.scheduler_args = scheduler_args or {}

    def encode(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.vae.encode(x)

    def decode(self, z: torch.Tensor) -> Sequence[torch.Tensor]:
        out = self.vae.decode(z)
        if not isinstance(out, Sequence):
            raise ValueError("The output of vae.decode should be a sequence.")
        return out

    def forward(self, x: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        return self.decode(self.encode(x))

    def generic_step(
        self,
        x: Sequence[torch.Tensor],
        mode: str = "train",
    ) -> torch.Tensor:
        x_categories, x_attributes = x[0], x[1]

        (mean, logvar), reconstruction = self.vae(x)
        reconstruction_categories = reconstruction[0]
        reconstruction_attributes = reconstruction[1]

        reconstruction_loss_categories = F.cross_entropy(
            reconstruction_categories,
            x_categories.argmax(dim=1),
            reduction="sum",
        )
        reconstruction_loss_attributes = gaussian_nll(
            reconstruction_attributes, torch.tensor(0), x_attributes
        ).sum()

        reconstruction_loss = (
            self.coef_categories * reconstruction_loss_categories
            + self.coef_attributes * reconstruction_loss_attributes
        )
        kl_loss = kl_divergence_loss(mean, logvar)
        total_loss = reconstruction_loss + self.vae.beta * kl_loss

        self.log(
            f"{mode}/reconstruction_loss_categories",
            reconstruction_loss_categories,
        )
        self.log(
            f"{mode}/reconstruction_loss_attributes",
            reconstruction_loss_attributes,
        )
        self.log(f"{mode}/reconstruction_loss", reconstruction_loss)
        self.log(f"{mode}/kl_loss", kl_loss)
        self.log(f"{mode}/loss", total_loss)
        return total_loss

    def validation_step(
        self,
        batch: Mapping[str, Sequence[torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        x = batch["attr"]
        return self.generic_step(x, "val")

    def training_step(
        self,
        batch: Mapping[frozenset[str], Mapping[str, Sequence[torch.Tensor]]],
        batch_idx: int,
    ) -> torch.Tensor:
        x = batch[frozenset(["attr"])]["attr"]
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

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        self._should_skip_lr_scheduler_step = False
        scaler = getattr(
            self.trainer.strategy.precision_plugin, "scaler", None
        )
        if scaler:
            scale_before_step = scaler.get_scale()
            optimizer.step(closure=optimizer_closure)
            scale_after_step = scaler.get_scale()
            self._should_skip_lr_scheduler_step = (
                scale_before_step > scale_after_step
            )
        else:
            optimizer.step(closure=optimizer_closure)

    def lr_scheduler_step(self, scheduler, metric):
        if self._should_skip_lr_scheduler_step:
            return
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)
