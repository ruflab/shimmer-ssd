import math
from enum import StrEnum

import torch
from shimmer.modules.domain.domain import DomainModule
from torch import nn


class VAEType(StrEnum):
    sigma = "sigma"
    beta = "beta"


def reparameterize(mean, logvar):
    std = logvar.mul(0.5).exp()
    eps = torch.randn_like(std)
    return eps.mul(std).add(mean)


def kl_divergence_loss(
    mean: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return kl


def gaussian_nll(mu, log_sigma, x):
    return (
        0.5 * torch.pow((x - mu) / log_sigma.exp(), 2)
        + log_sigma
        + 0.5 * math.log(2 * math.pi)
    )


class VAE(DomainModule):
    def __init__(
        self,
        channel_num: int,
        ae_size: int,
        z_size: int,
        beta: float = 1,
        vae_type: VAEType = VAEType.beta,
    ):
        super().__init__()

        assert channel_num in [1, 3]
        assert beta >= 0

        self.channel_num = channel_num
        self.ae_size = ae_size
        self.z_size = z_size
        self.beta = beta
        self.vae_type = vae_type

        if self.vae_type == VAEType.sigma:
            self.log_sigma = nn.Parameter(
                torch.tensor(0.0), requires_grad=True
            )
        else:
            self.register_buffer("log_sigma", torch.tensor(0.0))

        self.encoder = Encoder(
            channel_num, ae_size=ae_size, use_batchnorm=True
        )
        self.q_mean = nn.Linear(self.encoder.out_size, self.z_size)
        self.q_logvar = nn.Linear(self.encoder.out_size, self.z_size)

        self.decoder = Decoder(
            channel_num,
            ae_size=ae_size,
            z_size=self.z_size,
            use_batchnorm=True,
        )

    def encode_stats(self, x: torch.Tensor):
        out = self.encoder(x)
        out = out.view(out.size(0), -1)

        mean_z = self.q_mean(out)
        var_z = self.q_logvar(out)
        return mean_z, var_z

    def encode(self, x: torch.Tensor):
        mean_z, _ = self.encode_stats(x)
        return mean_z

    def decode(self, z):
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        mean, logvar = self.encode_stats(x)
        z = reparameterize(mean, logvar)

        x_reconstructed = self.decoder(z)

        return (mean, logvar), x_reconstructed

    def reconstruction_loss(
        self, x_reconstructed: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        return gaussian_nll(x_reconstructed, self.log_sigma, x).sum()


class Encoder(nn.Module):
    def __init__(
        self,
        num_channels: int,
        ae_size: int = 1028,
        kernel_size: int = 4,
        padding: int = 1,
        use_batchnorm: bool = False,
    ):
        super().__init__()

        self.sizes = [
            ae_size // (2**i) for i in reversed(range(4))
        ]  # 1 2 4 8 # 32 64 128 256

        self.kernel_size = kernel_size
        self.padding = padding
        self.sizes[-1] = ae_size
        self.use_batchnorm = use_batchnorm

        self.out_size = self.sizes[3] * 2 * 2

        self.layers = nn.Sequential(
            nn.Conv2d(
                num_channels,
                self.sizes[0],
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.sizes[0])
            if self.use_batchnorm
            else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(
                self.sizes[0],
                self.sizes[1],
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.sizes[1])
            if self.use_batchnorm
            else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(
                self.sizes[1],
                self.sizes[2],
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.sizes[2])
            if self.use_batchnorm
            else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(
                self.sizes[2],
                self.sizes[3],
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.sizes[3])
            if self.use_batchnorm
            else nn.Identity(),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x).view(x.size(0), -1)


class Decoder(nn.Module):
    def __init__(
        self,
        num_channels: int,
        z_size: int,
        ae_size: int = 1028,
        kernel_size: int = 4,
        padding: int = 1,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.sizes = [ae_size // (2**i) for i in reversed(range(3))]
        self.sizes[-1] = ae_size

        self.kernel_size = kernel_size
        self.padding = padding
        self.use_batchnorm = use_batchnorm

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                z_size,
                self.sizes[2],
                kernel_size=8,
                stride=1,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.sizes[2])
            if self.use_batchnorm
            else nn.Identity(),
            nn.ReLU(),
            nn.ConvTranspose2d(
                self.sizes[2],
                self.sizes[1],
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.sizes[1])
            if self.use_batchnorm
            else nn.Identity(),
            nn.ReLU(),
            nn.ConvTranspose2d(
                self.sizes[1],
                self.sizes[0],
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.sizes[0])
            if self.use_batchnorm
            else nn.Identity(),
            nn.ReLU(),
        )

        self.out_layer = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(
                self.sizes[0],
                self.num_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding,
            ),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.out_layer(self.layers(z[:, :, None, None]))
