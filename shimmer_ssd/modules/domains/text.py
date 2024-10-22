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
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from shimmer_ssd.text import composer
from shimmer_ssd.text.utils import inspect_all_choices


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

    def forward(self, x: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
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

    def forward(self, z: torch.Tensor) -> list[torch.Tensor]:  # type: ignore
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
        super().__init__(latent_dim)
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim

        vae_encoder = Encoder(self.in_dim, self.hidden_dim, self.latent_dim)
        vae_decoder = Decoder(self.latent_dim, self.hidden_dim, self.in_dim)
        self.vae = VAE(vae_encoder, vae_decoder, beta)

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

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, raw_target: Any
    ) -> LossOutput:
        return LossOutput(F.mse_loss(pred, target, reduction="mean"))

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
            reconstruction_loss + self.vae.beta * kl_loss + loss_attr_cat + loss_attr
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


class GRUEncoder(nn.Module):
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

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        out = torch.cat(list(x), dim=-1)
        out = self.encoder(out)
        return out


class GRUTextDomainModule(DomainModule):
    in_dim = 768

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        vocab_size: int,
        seq_length: int,
        optim_lr: float = 1e-3,
        optim_weight_decay: float = 0,
        scheduler_args: SchedulerArgs | None = None,
        padding_token: int = 0,
        coef_token_loss: float = 0.5,
    ):
        super().__init__(latent_dim)
        self.is_frozen = False
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.seq_length = seq_length

        self._padding_token = padding_token

        self.projector = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.Tanh(),
        )

        self.embeddings = nn.Embedding(vocab_size, self.latent_dim)

        self.num_layers = 2
        self.decoder = nn.GRU(
            self.latent_dim,
            self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.text_head = nn.Linear(self.hidden_dim, self.vocab_size)

        self.optim_lr = optim_lr
        self.optim_weight_decay = optim_weight_decay
        self.logit_scale = nn.Parameter(torch.tensor([1 / 0.07]).log())

        if coef_token_loss > 1 or coef_token_loss < 0:
            raise ValueError("coef_token_loss should be between 0 and 1")
        self.coef_token_loss = coef_token_loss

        self.scheduler_args = SchedulerArgs(
            max_lr=optim_lr,
            total_steps=1,
        )
        self.scheduler_args.update(scheduler_args or {})

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, raw_target: Any
    ) -> LossOutput:
        text_token_loss, acc = self.text_token_loss(pred, raw_target)

        return LossOutput(
            2 * text_token_loss,
            {"loss_tokens": text_token_loss, "pred_t_acc": acc},
        )
        # latent_loss = F.mse_loss(pred, target, reduction="mean")
        # total_loss = (
        #     self.coef_token_loss * unimodal_loss
        #     + (1 - self.coef_token_loss) * latent_loss
        # )
        # return LossOutput(
        #     total_loss,
        #     {"loss_tokens": unimodal_loss, "latent": latent_loss, "pred_t_acc": acc},
        # )

    def compute_domain_loss(self, domain: Any) -> LossOutput:
        z = self.encode(domain)
        loss, acc = self.text_token_loss(z, domain)
        return LossOutput(loss, {"acc": acc})

    def encode(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.projector(x["bert"])

    def decode(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        context = z.unsqueeze(1)
        pad_tokens = self.embeddings(
            torch.zeros(
                z.size(0), self.seq_length - 1, dtype=torch.long, device=z.device
            )
        )
        seq = torch.cat([context, pad_tokens], dim=1)
        for k in range(0, self.seq_length - 1):
            out = self.decode_one(seq)
            seq[:, k + 1] = self.embeddings(out["tokens"][:, k])
        return self.decode_one(seq)

    def decode_one(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        out, _ = self.decoder(z)
        tokens_dist = self.text_head(out)
        tokens = torch.argmax(tokens_dist, -1)
        return {"token_dist": tokens_dist, "tokens": tokens}

    def forward(self, x: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.decode(self.encode(x))

    def text_token_loss(
        self, z: torch.Tensor, target: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = z.unsqueeze(1)
        real_tokens = self.embeddings(target["tokens"][:, :-1])
        seq = torch.cat([context, real_tokens], dim=1)

        out = self.decode_one(seq)
        loss = F.cross_entropy(out["token_dist"].transpose(1, 2), target["tokens"])
        padding_mask = target["tokens"] != self._padding_token

        padded_out = out["tokens"][padding_mask]
        padded_target = target["tokens"][padding_mask]

        acc = (padded_out == padded_target).sum() / padded_out.size(0)
        return loss, acc

    def generic_step(
        self, x: Mapping[str, torch.Tensor], mode: str = "train"
    ) -> torch.Tensor:
        z = self.encode(x)
        loss, acc = self.text_token_loss(z, x)
        self.log(f"{mode}/loss", loss)
        self.log(f"{mode}/acc", acc)
        return loss

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
        optimizer = AdamW(
            self.parameters(),
            lr=self.optim_lr,
            weight_decay=self.optim_weight_decay,
        )

        return {"optimizer": optimizer}


class Text2Attr(DomainModule):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        text_model: GRUTextDomainModule,
        optim_lr: float = 1e-3,
        optim_weight_decay: float = 0,
        scheduler_args: SchedulerArgs | None = None,
    ) -> None:
        super().__init__(latent_dim)
        self.save_hyperparameters(ignore=["text_model"])

        self.text_model = text_model

        self.pred_attr = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8),
            nn.Tanh(),
        )
        self.pred_cat = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=1),
        )

        self.optim_lr = optim_lr
        self.optim_weight_decay = optim_weight_decay

        self.scheduler_args = SchedulerArgs(
            max_lr=optim_lr,
            total_steps=1,
        )
        self.scheduler_args.update(scheduler_args or {})

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, raw_target: Any
    ) -> LossOutput:
        pred_attr = self.pred_attr(pred)
        pred_cat = self.pred_cat(pred)
        target_attr = self.pred_attr(target)
        target_cat = self.pred_cat(target)
        loss_attr = F.mse_loss(pred_attr, target_attr)
        loss_cat = F.cross_entropy(pred_cat, target_cat.argmax(dim=1))
        loss = loss_attr + loss_cat
        return LossOutput(loss, {"attr": loss_attr, "cat": loss_cat})

    def encode(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.text_model.encode(x)

    def decode(self, z: torch.Tensor) -> dict[str, Any]:
        out: dict[str, Any] = {}
        out.update(self.text_model.decode(z))
        pred_attr = self.pred_attr(z)
        pred_cat = self.pred_cat(z)
        out.update({"attr": [pred_cat, pred_attr, pred_attr]})
        return out

    def forward(self, x: Mapping[str, Any]) -> dict[str, list[torch.Tensor]]:
        return self.decode(self.encode(x))

    def generic_step(
        self,
        x: Mapping[str, Any],
        mode: str = "train",
    ) -> torch.Tensor:
        text: Mapping[str, torch.Tensor] = x["t"]
        attr: Sequence[torch.Tensor] = x["attr"]
        text_l = self.text_model.encode(text)
        pred_attr = self.pred_attr(text_l)
        pred_cat = self.pred_cat(text_l)
        cats = torch.argmax(attr[0], dim=1)
        loss_cat = F.cross_entropy(pred_cat, cats)
        loss_attr = F.mse_loss(pred_attr, attr[1])
        total_loss = loss_cat + loss_attr
        pred_cats = pred_cat.argmax(dim=1)
        acc = (cats == pred_cats).sum() / cats.size(0)

        self.log(f"{mode}/loss_cat", loss_cat)
        self.log(f"{mode}/loss_attr", loss_attr)
        self.log(f"{mode}/acc_cat", acc)
        self.log(f"{mode}/loss", total_loss)

        return total_loss

    def validation_step(  # type: ignore
        self, batch: Mapping[str, Mapping[str, torch.Tensor]], _
    ) -> torch.Tensor:
        return self.generic_step(batch, "val")

    def training_step(  # type: ignore
        self,
        batch: Mapping[frozenset[str], Mapping[str, Mapping[str, torch.Tensor]]],
        _,
    ) -> torch.Tensor:
        data = batch[frozenset(["t", "attr"])]
        return self.generic_step(data, "train")

    def configure_optimizers(  # type: ignore
        self,
    ) -> dict[str, Any]:
        optimizer = AdamW(
            self.parameters(),
            lr=self.optim_lr,
            weight_decay=self.optim_weight_decay,
        )

        return {"optimizer": optimizer}
