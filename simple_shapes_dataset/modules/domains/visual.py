import torch
from shimmer.modules.domain import DomainModule
from shimmer.modules.vae import VAE

from simple_shapes_dataset.modules.vae import RAEDecoder, RAEEncoder


class VisualDomainModule(DomainModule):
    def __init__(
        self,
        num_channels: int,
        ae_dim: int,
        latent_dim: int,
        beta: float = 1,
    ):
        vae_encoder = RAEEncoder(
            num_channels,
            ae_dim,
            latent_dim,
            use_batchnorm=True,
        )
        vae_decoder = RAEDecoder(
            num_channels,
            latent_dim,
            ae_dim,
        )
        self.vae = VAE(vae_encoder, vae_decoder, beta)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z)
