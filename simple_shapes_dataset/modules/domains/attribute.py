from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from shimmer import LossOutput
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

    def forward(self, x: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
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

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        out = self.decoder(x)
        return [self.decoder_categories(out), self.decoder_attributes(out)]


class AttributeDomainModule(DomainModule):
    in_dim = 11

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
        super().__init__(latent_dim)
        self.save_hyperparameters()

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

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> LossOutput:
        return LossOutput(F.mse_loss(pred, target, reduction="mean"))

    def encode(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.vae.encode(x[:-1])

    def decode(self, z: torch.Tensor) -> list[torch.Tensor]:
        out = list(self.vae.decode(z))
        if not isinstance(out, Sequence):
            raise ValueError("The output of vae.decode should be a sequence.")
        out.append(torch.zeros_like(z[:, -1]))
        return out

    def forward(self, x: Sequence[torch.Tensor]) -> list[torch.Tensor]:  # type: ignore
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

    def validation_step(  # type: ignore
        self, batch: Mapping[str, Sequence[torch.Tensor]], _
    ) -> torch.Tensor:
        x = batch["attr"]
        return self.generic_step(x, "val")

    def training_step(  # type: ignore
        self,
        batch: Mapping[frozenset[str], Mapping[str, Sequence[torch.Tensor]]],
        _,
    ) -> torch.Tensor:
        x = batch[frozenset(["attr"])]["attr"]
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

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }


class AttributeWithUnpairedDomainModule(DomainModule):
    in_dim = 11

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        beta: float = 1,
        coef_categories: float = 1,
        coef_attributes: float = 1,
        n_unpaired: int = 1,
        optim_lr: float = 1e-3,
        optim_weight_decay: float = 0,
        scheduler_args: SchedulerArgs | None = None,
    ):
        super().__init__(latent_dim + n_unpaired)

        self.save_hyperparameters()
        self.paired_dim = latent_dim
        self.n_unpaired = n_unpaired
        self.hidden_dim = hidden_dim
        self.coef_categories = coef_categories
        self.coef_attributes = coef_attributes

        vae_encoder = Encoder(self.hidden_dim, self.latent_dim - self.n_unpaired)
        vae_decoder = Decoder(self.latent_dim - self.n_unpaired, self.hidden_dim)
        self.vae = VAE(vae_encoder, vae_decoder, beta)

        self.optim_lr = optim_lr
        self.optim_weight_decay = optim_weight_decay

        self.scheduler_args = SchedulerArgs(
            max_lr=optim_lr,
            total_steps=1,
        )
        self.scheduler_args.update(scheduler_args or {})

    def encode(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        z = self.vae.encode(x[:-1])
        return torch.cat([z, x[-1]], dim=-1)

    def decode(self, z: torch.Tensor) -> list[torch.Tensor]:
        paired = z[:, : self.paired_dim]
        unpaired = z[:, self.paired_dim :]
        out = list(self.vae.decode(paired))
        if not isinstance(out, Sequence):
            raise ValueError("The output of vae.decode should be a sequence.")
        out.append(unpaired)
        return out

    def forward(self, x: Sequence[torch.Tensor]) -> list[torch.Tensor]:  # type: ignore
        return self.decode(self.encode(x))

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> LossOutput:
        return LossOutput(
            loss=F.mse_loss(pred, target, reduction="mean"),
            metrics={
                "unpaired": F.mse_loss(
                    pred[:, self.paired_dim :], target[:, self.paired_dim :]
                ),
                "paired": F.mse_loss(
                    pred[:, : self.paired_dim], target[:, : self.paired_dim]
                ),
            },
        )


class AttributeLegacyDomainModule(DomainModule):
    latent_dim = 11

    def __init__(self):
        super().__init__(self.latent_dim)
        self.save_hyperparameters()

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> LossOutput:
        pred_cat, pred_attr, _ = self.decode(pred)
        target_cat, target_attr, _ = self.decode(target)

        loss_attr = F.mse_loss(pred_attr, target_attr, reduction="mean")
        loss_cat = F.nll_loss(pred_cat, torch.argmax(target_cat, 1))
        loss = loss_attr + loss_cat

        return LossOutput(loss, metrics={"loss_attr": loss_attr, "loss_cat": loss_cat})

    def encode(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat(list(x)[:-1], dim=-1)

    def decode(self, z: torch.Tensor) -> list[torch.Tensor]:
        categories = z[:, :3]
        attr = z[:, 3:11]
        unpaired = torch.zeros_like(z[:, 0])
        return [categories, attr, unpaired]

    def forward(self, x: Sequence[torch.Tensor]) -> list[torch.Tensor]:  # type: ignore
        return self.decode(self.encode(x))
