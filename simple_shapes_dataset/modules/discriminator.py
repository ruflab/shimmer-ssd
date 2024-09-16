from collections.abc import Mapping

import torch
import torch.nn.functional as F
from shimmer import (
    BroadcastLossCoefs,
    ContrastiveLoss,
    ContrastiveLossType,
    DomainModule,
    GlobalWorkspaceBase,
    GWModule,
    LatentsDomainGroupsT,
    LossOutput,
    ModelModeT,
    RandomSelection,
    SchedulerArgs,
    SelectionBase,
)
from shimmer.modules.global_workspace import freeze_domain_modules
from shimmer.modules.losses import GWLosses
from torch import nn
from torch.optim.lr_scheduler import LRScheduler


class CoefWithDiscriminator(BroadcastLossCoefs, total=False):
    generator: float
    discriminator: float


class GWLossesWithDiscriminator(GWLosses):
    def __init__(
        self,
        domain_name: str,
        hidden_dim: int,
        gw_mod: GWModule,
        selection_mod: SelectionBase,
        domain_mods: dict[str, DomainModule],
        loss_coefs: CoefWithDiscriminator,
        contrastive_fn: ContrastiveLossType,
        generator_loss_every: int = 1,
    ):
        super().__init__(gw_mod, selection_mod, domain_mods, loss_coefs, contrastive_fn)

        self._domain_name = domain_name
        latent_dim = self.domain_mods[domain_name].latent_dim
        self.generator_loss_every = generator_loss_every
        self.num_step = 1

        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def discriminator_loss(
        self, latent_domains: LatentsDomainGroupsT
    ) -> Mapping[str, torch.Tensor]:
        """
        Computes the discriminator loss.
        - "real" examples are taken from all domain latent representations
        - "fake" examples are taken from all predictions of the domain latent from
            the available domains

        Args:
            latent_domains (`LatentsDomainGroupsT`):

        Returns:
            `Mapping[str, torch.Tensor]`:
        """
        for param in self.discriminator.parameters():
            param.requires_grad_(True)

        real_vecs: list[torch.Tensor] = []
        fake_vecs: list[torch.Tensor] = []

        for domains, latents in latent_domains.items():
            if self._domain_name in domains:
                real_vecs.append(latents[self._domain_name])

            x_recons = self.gw_mod.decode(
                self.gw_mod.encode_and_fuse(latents, self.selection_mod),
                domains={self._domain_name},
            )[self._domain_name]
            fake_vecs.append(x_recons)

        real_pred = self.discriminator(torch.cat(real_vecs, dim=0).detach())
        real_target = torch.ones_like(real_pred)
        fake_pred = self.discriminator(torch.cat(fake_vecs, dim=0).detach())
        fake_target = torch.zeros_like(fake_pred)

        loss_real = F.binary_cross_entropy_with_logits(real_pred, real_target)
        loss_fake = F.binary_cross_entropy_with_logits(fake_pred, fake_target)
        loss = loss_real + loss_fake
        real_pred_bin = torch.sigmoid(real_pred) >= 0.5
        fake_pred_bin = torch.sigmoid(fake_pred) >= 0.5
        acc_real = (real_pred_bin == real_target.to(torch.bool)).sum() / real_pred.size(
            0
        )
        acc_fake = (fake_pred_bin == fake_target.to(torch.bool)).sum() / fake_pred.size(
            0
        )

        return {
            "discriminator_real": loss_real,
            "discriminator_fake": loss_fake,
            "discriminator_real_acc": acc_real,
            "discriminator_fake_acc": acc_fake,
            "discriminator": loss,
        }

    def generator_loss(
        self, latent_domains: LatentsDomainGroupsT
    ) -> Mapping[str, torch.Tensor]:
        """
        Computes the "generator" loss. Generation is successful if the discriminator
        cannot differenciate the generated sample from actual samples.

        Args:
            latent_domains (`LatentsDomainGroupsT`):

        Returns:
            `Mapping[str, torch.Tensor]`:
        """
        for param in self.discriminator.parameters():
            param.requires_grad_(False)

        fake_vecs: list[torch.Tensor] = []

        for latents in latent_domains.values():
            x_recons = self.gw_mod.decode(
                self.gw_mod.encode_and_fuse(latents, self.selection_mod),
                domains={self._domain_name},
            )[self._domain_name]
            fake_vecs.append(x_recons)

        fake_pred = self.discriminator(torch.cat(fake_vecs, dim=0))
        fake_target = torch.ones_like(fake_pred)

        loss_fake = F.binary_cross_entropy_with_logits(fake_pred, fake_target)
        fake_pred_bin = torch.sigmoid(fake_pred) >= 0.5
        acc_fake = (fake_pred_bin == fake_target.to(torch.bool)).sum() / fake_pred.size(
            0
        )
        return {
            "generator": loss_fake,
            "generator_acc": acc_fake,
        }

    def step(
        self, domain_latents: LatentsDomainGroupsT, mode: ModelModeT
    ) -> LossOutput:
        """
        Performs a step of loss computation.

        Args:
            domain_latents: Latent representations for all domains.
            mode: The mode in which the model is currently operating.

        Returns:
            A LossOutput object containing the loss and metrics for this step.
        """
        metrics: dict[str, torch.Tensor] = {}

        metrics.update(self.contrastive_loss(domain_latents))
        metrics.update(self.broadcast_loss(domain_latents))

        metrics["broadcast_loss"] = torch.stack(
            [
                metrics[name]
                for name, coef in self.loss_coefs.items()
                if isinstance(coef, float)
                and coef > 0
                and name != "contrastives"
                and name in metrics
            ],
            dim=0,
        ).mean()

        if self.num_step % (self.generator_loss_every + 1) == 0:
            generator_loss = self.generator_loss(domain_latents)
            metrics.update(generator_loss)
        else:
            discriminatol_loss = self.discriminator_loss(domain_latents)
            metrics.update(discriminatol_loss)

        loss = torch.stack(
            [
                metrics[name] * coef
                for name, coef in self.loss_coefs.items()
                if isinstance(coef, float) and coef > 0 and name in metrics
            ],
            dim=0,
        ).mean()

        self.num_step += 1

        return LossOutput(loss, metrics)


class GlobalWorkspaceWithDiscriminator(
    GlobalWorkspaceBase[GWModule, RandomSelection, GWLossesWithDiscriminator]
):
    """The 2-domain fusion (with broadcast loss) flavor of GlobalWorkspaceBase.

    This is used to simplify a Global Workspace instanciation and only overrides the
    `__init__` method.
    """

    def __init__(
        self,
        domain_mods: Mapping[str, DomainModule],
        gw_encoders: Mapping[str, nn.Module],
        gw_decoders: Mapping[str, nn.Module],
        workspace_dim: int,
        loss_coefs: CoefWithDiscriminator,
        discriminated_domain_name: str,
        hidden_dim: int,
        generator_loss_every: int = 1,
        selection_temperature: float = 0.2,
        optim_lr: float = 1e-3,
        optim_weight_decay: float = 0.0,
        scheduler_args: SchedulerArgs | None = None,
        learn_logit_scale: bool = False,
        contrastive_loss: ContrastiveLossType | None = None,
        scheduler: LRScheduler | None = None,
    ) -> None:
        """
        Initializes a Global Workspace

        Args:
            domain_mods (`Mapping[str, DomainModule]`): mapping of the domains
                connected to the GW. Keys are domain names, values are the
                `DomainModule`.
            gw_encoders (`Mapping[str, torch.nn.Module]`): mapping for each domain
                name to a `torch.nn.Module` class which role is to encode a
                unimodal latent representations into a GW representation (pre fusion).
            gw_decoders (`Mapping[str, torch.nn.Module]`): mapping for each domain
                name to a `torch.nn.Module` class which role is to decode a
                GW representation into a unimodal latent representations.
            workspace_dim (`int`): dimension of the GW.
            loss_coefs (`BroadcastLossCoefs`): loss coefs for the losses.
            selection_temperature (`float`): temperature value for the RandomSelection
                module.
            optim_lr (`float`): learning rate
            optim_weight_decay (`float`): weight decay
            scheduler_args (`SchedulerArgs | None`): optimization scheduler's arguments
            learn_logit_scale (`bool`): whether to learn the contrastive learning
                contrastive loss when using the default contrastive loss.
            contrastive_loss (`ContrastiveLossType | None`): a contrastive loss
                function used for alignment. `learn_logit_scale` will not affect custom
                contrastive losses.
            scheduler: The scheduler to use for traning. If None is explicitely given,
                no scheduler will be used. Defaults to use OneCycleScheduler
        """
        domain_mods = freeze_domain_modules(domain_mods)
        gw_mod = GWModule(domain_mods, workspace_dim, gw_encoders, gw_decoders)

        if contrastive_loss is None:
            contrastive_loss = ContrastiveLoss(
                torch.tensor([1 / 0.07]).log(), "mean", learn_logit_scale
            )

        selection_mod = RandomSelection(selection_temperature)
        loss_mod = GWLossesWithDiscriminator(
            discriminated_domain_name,
            hidden_dim,
            gw_mod,
            selection_mod,
            domain_mods,
            loss_coefs,
            contrastive_loss,
            generator_loss_every,
        )

        super().__init__(
            gw_mod,
            selection_mod,
            loss_mod,
            optim_lr,
            optim_weight_decay,
            scheduler_args,
            scheduler,
        )
