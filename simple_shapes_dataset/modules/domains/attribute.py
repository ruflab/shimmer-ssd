import lightning.pytorch as pl
import torch
from shimmer.modules.domain import DomainModule
from shimmer.modules.global_workspace import Encoder

from simple_shapes_dataset.config.types import VAEType
from simple_shapes_dataset.modules.vae import VAE


class AttributeDomainModule(DomainModule):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        n_layers: int,
        beta: float = 1,
        vae_type: VAEType = VAEType.beta,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_dim = 11
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        vae_encoder = Encoder(
            self.in_dim, self.hidden_dim, self.latent_dim, self.n_layers
        )
        vae_decoder = Encoder(
            self.latent_dim, self.hidden_dim, self.in_dim, self.n_layers
        )
        self.vae = VAE(vae_encoder, vae_decoder, beta, vae_type)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z)


class AttributeModule(pl.LightningModule):
    def __init__(
        self,
        attribute_domain: AttributeDomainModule,
        optim_lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.attribute_domain = attribute_domain

        self.optim_lr = optim_lr

    def generic_step(
        self, batch: torch.Tensor, mode: str = "train"
    ) -> torch.Tensor:
        (
            reconstruction_loss,
            kl_loss,
            total_loss,
        ) = self.attribute_domain.vae.get_losses(batch)

        self.log(f"{mode}/reconstruction_loss", reconstruction_loss)
        self.log(f"{mode}/kl_loss", kl_loss)
        self.log(f"{mode}/loss", total_loss)
        return total_loss

    def validation_step(self, batch: torch.Tensor) -> torch.Tensor:
        return self.generic_step(batch, "val")

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        return self.generic_step(batch, "train")

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=1e-3)
