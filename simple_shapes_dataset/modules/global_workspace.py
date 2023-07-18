from collections.abc import Mapping
from typing import Any

import torch
from info_nce import info_nce
from lightning.pytorch import LightningModule
from shimmer.modules.domain import DomainModule
from shimmer.modules.global_workspace import GlobalWorkspace
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import OneCycleLR


class GlobalWorkspaceLightningModule(LightningModule):
    def __init__(
        self,
        global_workspace: GlobalWorkspace,
        domain_modules: Mapping[str, DomainModule],
        demi_cycle_loss_coefficient: float,
        cycle_loss_coefficient: float,
        translation_loss_coefficient: float,
        contrastive_loss_coefficient: float,
        optim_lr: float,
        optim_weight_decay: float,
        scheduler_args: dict[str, Any],
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["global_workspace", "domain_modules"]
        )

        self.global_workspace = global_workspace

        for module in domain_modules.values():
            module.eval().freeze()

        self.domain_modules = domain_modules

        self.loss_coefficients = {
            "demi_cycles": demi_cycle_loss_coefficient,
            "cycles": cycle_loss_coefficient,
            "translations": translation_loss_coefficient,
            "contrastives": contrastive_loss_coefficient,
        }

        self.optim_lr = optim_lr
        self.optim_weight_decay = optim_weight_decay
        self.scheduler_args = scheduler_args

    def demi_cycle_loss(
        self, latent_domains: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        for domain_name in latent_domains.keys():
            z = self.global_workspace.translate(latent_domains, to=domain_name)
            losses[f"demi_cycle_{domain_name}"] = mse_loss(
                z, latent_domains[domain_name]
            )
        losses["demi_cycles"] = torch.stack(
            list(losses.values()), dim=0
        ).mean()
        return losses

    def cycle_loss(
        self, latent_domains: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        for domain_name_target in self.domain_modules.keys():
            z = self.global_workspace.cycle(
                latent_domains, through=domain_name_target
            )
            for domain_name_source in latent_domains.keys():
                loss_name = (
                    f"cycle_{domain_name_source}_through_{domain_name_target}"
                )
                losses[loss_name] = mse_loss(
                    z[domain_name_source], latent_domains[domain_name_source]
                )
        losses["cycles"] = torch.stack(list(losses.values()), dim=0).mean()
        return losses

    def translation_loss(
        self, latent_domains: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        for domain_name_source in latent_domains.keys():
            z = self.global_workspace.encode(
                {domain_name_source: latent_domains[domain_name_source]}
            )
            for domain_name_target in self.domain_modules.keys():
                prediction = self.global_workspace.decode(
                    z, domains={domain_name_target}
                )[domain_name_target]
                loss_name = (
                    f"translation_{domain_name_source}_to_{domain_name_target}"
                )
                losses[loss_name] = mse_loss(
                    prediction, latent_domains[domain_name_target]
                )
        losses["translations"] = torch.stack(
            list(losses.values()), dim=0
        ).mean()
        return losses

    def contrastive_loss(
        self, latent_domains: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}

        for domain_name_source in latent_domains.keys():
            z_source = self.global_workspace.encode(
                {domain_name_source: latent_domains[domain_name_source]}
            )
            for domain_name_target in self.domain_modules.keys():
                z_target = self.global_workspace.encode(
                    {domain_name_target: latent_domains[domain_name_target]}
                )
                loss_name = f"contrastive_{domain_name_source}_and_{domain_name_target}"
                losses[loss_name] = info_nce(z_source, z_target)
        losses["contrastives"] = torch.stack(
            list(losses.values()), dim=0
        ).mean()
        return losses

    def encode_domains(
        self,
        domains: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        return {
            name: self.domain_modules[name].encode(domain)
            for name, domain in domains.items()
        }

    def decode_domains(
        self,
        domains: Mapping[str, torch.Tensor],
    ) -> dict[str, Any]:
        return {
            name: self.domain_modules[name].decode(domain)
            for name, domain in domains.items()
        }

    def _get_batch_size(
        self, domain_latents: Mapping[str, torch.Tensor]
    ) -> int:
        for data in domain_latents.values():
            return data.size(0)
        return 0

    def generic_step(
        self,
        batch: Mapping[frozenset[str], Mapping[str, Any]],
        mode: str,
    ) -> torch.Tensor:
        losses: dict[str, torch.Tensor] = {}
        batch_size = 0
        for domains, data in batch.items():
            domain_latents = self.encode_domains(data)
            batch_size = self._get_batch_size(domain_latents)

            match len(domains):
                case 1:
                    losses.update(self.demi_cycle_loss(domain_latents))
                    losses.update(self.cycle_loss(domain_latents))
                case 2:
                    losses.update(self.translation_loss(domain_latents))
                    losses.update(self.contrastive_loss(domain_latents))
        losses["loss"] = torch.stack(
            [
                self.loss_coefficients[name] * losses[name]
                for name in self.loss_coefficients.keys()
            ],
            dim=0,
        ).mean()

        for name, loss in losses.items():
            self.log(f"{mode}/{name}", loss, batch_size=batch_size)

        return losses["loss"]

    def validation_step(
        self,
        data: Mapping[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        batch = {frozenset(data.keys()): data}
        for domain in data.keys():
            batch[frozenset([domain])] = {domain: data[domain]}
        return self.generic_step(batch, mode="val")

    def training_step(
        self,
        batch: Mapping[frozenset[str], Mapping[str, Any]],
        batch_idx: int,
    ) -> torch.Tensor:
        return self.generic_step(batch, mode="train")

    def configure_optimizers(self) -> dict[str, Any]:
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
