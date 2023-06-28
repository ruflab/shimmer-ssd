import torch
from shimmer.modules.domain import DomainModule
from shimmer.modules.global_workspace import Encoder

from simple_shapes_dataset.modules.vae import VAE, VAEType


class AttibuteModule(DomainModule):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        n_layers: int,
        beta: float = 1,
        vae_type: VAEType = VAEType.beta,
    ):
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
