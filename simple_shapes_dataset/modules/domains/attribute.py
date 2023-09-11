from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from shimmer.modules.domain import DomainModule
from shimmer.modules.global_workspace import SchedulerArgs
from shimmer.modules.vae import (
    VAE,
    VAEDecoder,
    VAEEncoder,
    gaussian_nll,
    kl_divergence_loss,
)
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR

from simple_shapes_dataset.modules.losses import margin_loss


class Encoder(VAEEncoder):
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
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

        self.q_mean = nn.Linear(self.out_dim, self.out_dim)
        self.q_logvar = nn.Linear(self.out_dim, self.out_dim)

    def forward(
        self, x: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = torch.cat(list(x), dim=-1)
        out = self.encoder(out)
        return self.q_mean(out), self.q_logvar(out)


class Decoder(VAEDecoder):
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
            nn.Linear(self.hidden_dim, 3),
        )

        self.decoder_attributes = nn.Sequential(
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
        scheduler_args: SchedulerArgs | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.in_dim = 11
        self.hidden_dim = hidden_dim
        self.coef_categories = coef_categories
        self.coef_attributes = coef_attributes

        vae_encoder = Encoder(self.hidden_dim, self.latent_dim)
        vae_decoder = Decoder(self.latent_dim, self.hidden_dim)
        self.vae = VAE(vae_encoder, vae_decoder, beta)

        self.optim_lr = optim_lr
        self.optim_weight_decay = optim_weight_decay

        self.scheduler_args = SchedulerArgs(
            max_lr=optim_lr,
            total_steps=1,
        )
        self.scheduler_args.update(scheduler_args or {})

    def encode(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.vae.encode(x[:-1])

    def decode(self, z: torch.Tensor) -> list[torch.Tensor]:
        out = list(self.vae.decode(z))
        if not isinstance(out, Sequence):
            raise ValueError("The output of vae.decode should be a sequence.")
        out.append(torch.zeros_like(z[:, -1]))
        return out

    def forward(self, x: Sequence[torch.Tensor]) -> list[torch.Tensor]:
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
        self, batch: Mapping[str, Sequence[torch.Tensor]], _
    ) -> torch.Tensor:
        x = batch["attr"]
        return self.generic_step(x, "val")

    def training_step(
        self,
        batch: Mapping[frozenset[str], Mapping[str, Sequence[torch.Tensor]]],
        _,
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


class AttributeWithUnpairedDomainModule(DomainModule):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        beta: float = 1,
        coef_categories: float = 1,
        coef_attributes: float = 1,
        optim_lr: float = 1e-3,
        optim_weight_decay: float = 0,
        scheduler_args: SchedulerArgs | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.latent_dim = latent_dim + 1
        self.in_dim = 11
        self.hidden_dim = hidden_dim
        self.coef_categories = coef_categories
        self.coef_attributes = coef_attributes

        # -1 for the unpaired attribute that is artificially added to the latent space.
        vae_encoder = Encoder(self.hidden_dim, self.latent_dim - 1)
        vae_decoder = Decoder(self.latent_dim - 1, self.hidden_dim)
        self.vae = VAE(vae_encoder, vae_decoder, beta)

        self.optim_lr = optim_lr
        self.optim_weight_decay = optim_weight_decay

        self.scheduler_args = SchedulerArgs(
            max_lr=optim_lr,
            total_steps=1,
        )
        self.scheduler_args.update(scheduler_args or {})

    def on_before_gw_encode_cont(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        out[:, -1] = 0
        return out

    def encode(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        z = self.vae.encode(x[:-1])
        return torch.cat([z, x[-1]], dim=-1)

    def decode(self, z: torch.Tensor) -> list[torch.Tensor]:
        out = list(self.vae.decode(z[:, :-1]))
        if not isinstance(out, Sequence):
            raise ValueError("The output of vae.decode should be a sequence.")
        out.append(z[:, -1])
        return out

    def forward(self, x: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        return self.decode(self.encode(x))

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {
            "loss": F.mse_loss(pred, target, reduction="sum"),
            "unpaired": F.mse_loss(pred[:, -1], target[:, -1]),
            "other": F.mse_loss(pred[:, 0], target[:, 0]),
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
            "loss": loss.sum(),
            "unpaired": loss[:, -1].sum(),
            "other": loss[:, 0].sum(),
        }
