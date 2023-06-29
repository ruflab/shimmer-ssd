from collections.abc import Mapping

import torch
from shimmer.modules.domain import DomainModule
from shimmer.modules.global_workspace import Encoder
from torch import nn

from simple_shapes_dataset.config.domains import VAEType
from simple_shapes_dataset.modules.vae import VAE


class VAEEncoder(Encoder):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
    ):
        super().__init__(in_dim, hidden_dim, out_dim, n_layers)

        self.q_mean = nn.Linear(self.out_dim, self.out_dim)
        self.q_logvar = nn.Linear(self.out_dim, self.out_dim)

    def forward(self, input):
        out = torch.relu(super().forward(input))
        return self.q_mean(out), self.q_logvar(out)


class AttributeDomainModule(DomainModule):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        n_layers: int,
        beta: float = 1,
        vae_type: VAEType = VAEType.beta,
        optim_lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.in_dim = 11
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        vae_encoder = VAEEncoder(
            self.in_dim, self.hidden_dim, self.latent_dim, self.n_layers
        )
        vae_decoder = Encoder(
            self.latent_dim, self.hidden_dim, self.in_dim, self.n_layers
        )
        self.vae = VAE(vae_encoder, vae_decoder, beta, vae_type)

        self.optim_lr = optim_lr

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z)

    def generic_step(
        self,
        x: torch.Tensor,
        mode: str = "train",
    ) -> torch.Tensor:
        (
            reconstruction_loss,
            kl_loss,
            total_loss,
        ) = self.vae.get_losses(x)

        self.log(f"{mode}/reconstruction_loss", reconstruction_loss)
        self.log(f"{mode}/kl_loss", kl_loss)
        self.log(f"{mode}/loss", total_loss)
        return total_loss

    def validation_step(
        self,
        batch: Mapping[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x = batch["attr"]
        return self.generic_step(x, "val")

    def training_step(
        self,
        batch: Mapping[frozenset[str], Mapping[str, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        x = batch[frozenset(["attr"])]["attr"]
        return self.generic_step(x, "train")

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=1e-3)
