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

from simple_shapes_dataset.text import composer
from simple_shapes_dataset.text.utils import inspect_all_choices


class Encoder(VAEEncoder):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
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
        out_dim: int,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.decoder = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> list[torch.Tensor]:
        return [self.decoder(z)]


class TextDomainModule(DomainModule):
    in_dim = 768

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        beta: float = 1,
        optim_lr: float = 1e-3,
        optim_weight_decay: float = 0,
        scheduler_args: SchedulerArgs | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        vae_encoder = Encoder(self.in_dim, self.hidden_dim, self.latent_dim)
        vae_decoder = Decoder(self.latent_dim, self.hidden_dim, self.in_dim)
        self.vae = VAE(vae_encoder, vae_decoder, beta)

        self.attribute_cls = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.attribute_cls_cat = nn.Linear(self.hidden_dim, 3)
        self.attribute_cls_attr = nn.Sequential(
            nn.Linear(self.hidden_dim, 8), nn.Tanh()
        )

        self.composer_grammar_options = inspect_all_choices(composer)

        self.grammar_cls = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.grammar_heads = nn.ModuleDict(
            {
                name: nn.Linear(self.hidden_dim, n_outputs)
                for name, n_outputs in self.composer_grammar_options.items()
            }
        )

        self.optim_lr = optim_lr
        self.optim_weight_decay = optim_weight_decay

        self.scheduler_args = SchedulerArgs(
            max_lr=optim_lr,
            total_steps=1,
        )
        self.scheduler_args.update(scheduler_args or {})

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {"loss": F.mse_loss(pred, target, reduction="mean")}

    def encode(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.vae.encode((x["bert"],))

    def decode(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        text: dict[str, torch.Tensor] = {"bert": self.vae.decode(z)[0]}
        attr_pred_cat, attr_pred_attr = self.predict_attr(z)
        text["cls"] = attr_pred_cat
        text["attr"] = attr_pred_attr
        text["unpaired"] = torch.zeros_like(z[:, -1])
        text.update(self.predict_grammar(z))
        return text

    def predict_attr(
        self, mean: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attr_pred = self.attribute_cls(mean.detach())
        attr_pred_cat = self.attribute_cls_cat(attr_pred)
        attr_pred_attr = self.attribute_cls_attr(attr_pred)
        return attr_pred_cat, attr_pred_attr

    def predict_grammar(self, mean: torch.Tensor) -> dict[str, torch.Tensor]:
        grammar_pred = self.grammar_cls(mean.detach())
        return {
            name: head(grammar_pred)
            for name, head in self.grammar_heads.items()
        }

    def grammar_losses(
        self, mean: torch.Tensor, targets
    ) -> dict[str, torch.Tensor]:
        grammar_pred = self.predict_grammar(mean)
        return {
            f"{name}_ce": F.cross_entropy(
                pred, targets[name][:, 0].long(), reduction="sum"
            )
            for name, pred in grammar_pred.items()
        }

    def forward(
        self, x: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return self.decode(self.encode(x))

    def generic_step(
        self,
        x: Mapping[str, torch.Tensor],
        mode: str = "train",
    ) -> torch.Tensor:
        (mean, logvar), reconstruction = self.vae((x["bert"],))

        reconstruction_loss = gaussian_nll(
            reconstruction[0], torch.tensor(0), x["bert"]
        ).sum()

        kl_loss = kl_divergence_loss(mean, logvar)

        attr_pred_cat, attr_pred_attr = self.predict_attr(mean)

        loss_attr_cat = F.cross_entropy(
            attr_pred_cat, x["cls"].argmax(dim=1), reduction="sum"
        )
        loss_attr = F.mse_loss(attr_pred_attr, x["attr"], reduction="sum")
        grammar_targets = {
            name: x[name] for name in self.composer_grammar_options
        }
        grammar_losses = self.grammar_losses(mean, grammar_targets)

        total_loss = (
            reconstruction_loss
            + self.vae.beta * kl_loss
            + loss_attr_cat
            + loss_attr
        )

        for grammar_loss_name, grammar_loss in grammar_losses.items():
            total_loss += grammar_loss
            self.log(f"{mode}/{grammar_loss_name}", grammar_loss)

        self.log(f"{mode}/reconstruction_loss", reconstruction_loss)
        self.log(f"{mode}/kl_loss", kl_loss)
        self.log(f"{mode}/attr_category", loss_attr_cat)
        self.log(f"{mode}/attr_attr", loss_attr)
        self.log(f"{mode}/loss", total_loss)
        return total_loss

    def validation_step(
        self, batch: Mapping[str, Mapping[str, torch.Tensor]], _
    ) -> torch.Tensor:
        x = batch["t"]
        return self.generic_step(x, "val")

    def training_step(
        self,
        batch: Mapping[
            frozenset[str], Mapping[str, Mapping[str, torch.Tensor]]
        ],
        _,
    ) -> torch.Tensor:
        x = batch[frozenset(["t"])]["t"]
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