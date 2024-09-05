from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

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
from simple_shapes_dataset.text import composer
from simple_shapes_dataset.text.utils import inspect_all_choices
from collections.abc import Mapping



def symlog(x, alpha=1):
    return (
        torch.sign(x) * torch.log(1 + alpha * torch.abs(x)) / np.log(1 + alpha)
    )


def symexp(x, alpha=1):
    return torch.sign(x) * (torch.exp(alpha * torch.abs(x)) - 1) / alpha


class SymLog(nn.Module):
    def __init__(self, alpha=1):
        super(SymLog, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return symlog(x, self.alpha)

    def inverse(self, x):
        return symexp(x, self.alpha)


class Bimgw_vae_text(nn.Module):
    def __init__(
        self,
        z_size: int,
        hidden_size: int,
        n_classes: int,
    ):
        super(Bimgw_vae_text, self).__init__()

        self.n_classes = n_classes
        self.bert_size = 768
        self.z_size = z_size
        self.hidden_size = hidden_size

        self.transformer = None
        self.tokenizer = None

        self.encoder = nn.Sequential(
            nn.Linear(self.bert_size, self.bert_size),
            nn.ReLU(),
            nn.Linear(self.bert_size, self.bert_size // 2),
            nn.ReLU(),
            nn.Linear(self.bert_size // 2, self.z_size * 2),
            SymLog(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.z_size, self.bert_size // 2),
            nn.ReLU(),
            nn.Linear(self.bert_size // 2, self.bert_size),
            nn.ReLU(),
            nn.Linear(self.bert_size, self.bert_size),
        )


    def encode(self, text_item: Dict[str, Any]) -> Dict[str, Any]:
        z, _ = self.encode_stats(text_item["bert"])
        return {"z": z[:,:self.z_size]}


    def encode_stats(self, text_latent):
        z = self.encoder(text_latent)
        return z[:, : self.z_size], z[:, self.z_size :]

# Define the TextDomainModule class
class BimgTextDomainModule(DomainModule):  # Inherit from DomainModule
    in_dim = 768

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        beta: float = 1,
        optim_lr: float = 1e-3,
        optim_weight_decay: float = 0,
        scheduler_args: Optional[Dict[str, Any]] = None,
        checkpoint_path: str = '../checkpoints/vae_t.ckpt',
        z_size: int = 12,
        n_classes: int = 3
    ):
        super(BimgTextDomainModule, self).__init__(latent_dim)

        self.hidden_dim = hidden_dim

        # Instantiate the Bimgw_vae_text class
        vae = Bimgw_vae_text(z_size=z_size, hidden_size=hidden_dim, n_classes=n_classes)

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path)

        # Filter the state dictionary to only include keys from the VAE's encoder and decoder
        vae_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if ('encoder' in k or 'decoder' in k) and not 'attribute' in k}

        # Load the filtered state dictionary into the VAE model
        vae.load_state_dict(vae_state_dict, strict=True)

        # Use the loaded VAE's encoder and decoder
        self.vae = vae

        self.attribute_cls = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
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
            nn.ReLU(),
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


    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> LossOutput:
        return LossOutput(F.mse_loss(pred, target, reduction="mean"))

    # Update the encode function
    def encode(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.vae.encode({"bert": x["bert"]})["z"]

    # Update the decode function
    def decode(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        text: dict[str, torch.Tensor] = {"bert": self.vae.decoder(z)}
        text["unpaired"] = torch.zeros_like(z[:, -1])
        return text


    def predict_attr(self, mean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attr_pred = self.attribute_cls(mean)
        attr_pred_cat = self.attribute_cls_cat(attr_pred)
        attr_pred_attr = self.attribute_cls_attr(attr_pred)
        return attr_pred_cat, attr_pred_attr

    def predict_grammar(self, mean: torch.Tensor) -> dict[str, torch.Tensor]:
        grammar_pred = self.grammar_cls(mean)
        return {name: head(grammar_pred) for name, head in self.grammar_heads.items()}

    def grammar_losses(self, mean: torch.Tensor, targets) -> dict[str, torch.Tensor]:
        grammar_pred = self.predict_grammar(mean)
        return {
            f"{name}_ce": F.cross_entropy(
                pred, targets[name][:, 0].long(), reduction="sum"
            )
            for name, pred in grammar_pred.items()
        }

    def forward(self, x: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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
        grammar_targets = {name: x[name] for name in self.composer_grammar_options}
        grammar_losses = self.grammar_losses(mean, grammar_targets)

        total_loss = (
            reconstruction_loss + self.vae.beta * kl_loss + 1000000*(loss_attr_cat + loss_attr)
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

    def validation_step(  # type: ignore
        self, batch: Mapping[str, Mapping[str, torch.Tensor]], _
    ) -> torch.Tensor:
        x = batch["t"]
        return self.generic_step(x, "val")

    def training_step(  # type: ignore
        self,
        batch: Mapping[frozenset[str], Mapping[str, Mapping[str, torch.Tensor]]],
        _,
    ) -> torch.Tensor:
        x = batch[frozenset(["t"])]["t"]
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
