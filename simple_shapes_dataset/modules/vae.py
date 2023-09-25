from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from PIL.Image import Image
from shimmer.modules.vae import VAE, VAEDecoder, VAEEncoder
from torch import nn


class RAEEncoder(VAEEncoder):
    def __init__(
        self,
        num_channels: int,
        ae_dim: int = 1028,
        z_dim: int = 16,
        kernel_dim: int = 4,
        padding: int = 1,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        self.dims = [
            ae_dim // (2**i) for i in reversed(range(4))
        ]  # 1 2 4 8 # 32 64 128 256

        self.kernel_dim = kernel_dim
        self.padding = padding
        self.dims[-1] = ae_dim
        self.use_batchnorm = use_batchnorm

        self.out_dim = self.dims[3] * 2 * 2
        self.z_dim = z_dim

        self.layers = nn.Sequential(
            nn.Conv2d(
                num_channels,
                self.dims[0],
                kernel_size=self.kernel_dim,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[0])
            if self.use_batchnorm
            else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(
                self.dims[0],
                self.dims[1],
                kernel_size=self.kernel_dim,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[1])
            if self.use_batchnorm
            else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(
                self.dims[1],
                self.dims[2],
                kernel_size=self.kernel_dim,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[2])
            if self.use_batchnorm
            else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(
                self.dims[2],
                self.dims[3],
                kernel_size=self.kernel_dim,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[3])
            if self.use_batchnorm
            else nn.Identity(),
            nn.ReLU(),
        )

        self.q_mean = nn.Linear(self.out_dim, self.z_dim)
        self.q_logvar = nn.Linear(self.out_dim, self.z_dim)

    def forward(
        self, x: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.layers(x[0]).view(x[0].size(0), -1)
        out = out.view(out.size(0), -1)

        return self.q_mean(out), self.q_logvar(out)


class RAEDecoder(VAEDecoder):
    def __init__(
        self,
        num_channels: int,
        z_dim: int,
        ae_dim: int = 1028,
        kernel_dim: int = 4,
        padding: int = 1,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.dims = [ae_dim // (2**i) for i in reversed(range(3))]
        self.dims[-1] = ae_dim

        self.kernel_dim = kernel_dim
        self.padding = padding
        self.use_batchnorm = use_batchnorm

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                z_dim,
                self.dims[2],
                kernel_size=8,
                stride=1,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[2])
            if self.use_batchnorm
            else nn.Identity(),
            nn.ReLU(),
            nn.ConvTranspose2d(
                self.dims[2],
                self.dims[1],
                kernel_size=self.kernel_dim,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[1])
            if self.use_batchnorm
            else nn.Identity(),
            nn.ReLU(),
            nn.ConvTranspose2d(
                self.dims[1],
                self.dims[0],
                kernel_size=self.kernel_dim,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[0])
            if self.use_batchnorm
            else nn.Identity(),
            nn.ReLU(),
        )

        self.out_layer = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(
                self.dims[0],
                self.num_channels,
                kernel_size=self.kernel_dim,
                stride=1,
                padding=self.padding,
            ),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> list[torch.Tensor]:
        return [self.out_layer(self.layers(z[:, :, None, None]))]


def dim_exploration_figure(
    vae: VAE,
    z_size: int,
    device: torch.device,
    ax_from_tensors: Callable[[Any, int, int], Image],
    num_samples: int = 5,
    range_start: int = -6,
    range_end: int = 6,
    image_size: int = 32,
    plot_dims: Sequence[int] | None = None,
    fig_dim: int = 5,
) -> Figure:
    possible_dims = plot_dims or np.arange(z_size)

    fig_size = (len(possible_dims) - 1) * fig_dim

    fig = plt.figure(
        constrained_layout=True, figsize=(fig_size, fig_size), dpi=1
    )

    gs = GridSpec(len(possible_dims), len(possible_dims), figure=fig)
    done_dims: list[set[int]] = []

    for i, dim_i in enumerate(possible_dims):
        for j, dim_j in enumerate(possible_dims):
            if dim_i == dim_j or {dim_i, dim_j} in done_dims:
                continue

            done_dims.append({dim_i, dim_j})

            ax = fig.add_subplot(gs[j, i])

            z = (
                torch.zeros(z_size)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(num_samples, num_samples, -1)
                .to(device)
            )

            for p in range(num_samples):
                step = range_start + (range_end - range_start) * float(
                    p
                ) / float(num_samples - 1)
                z[p, :, dim_i] = step
            for q in range(num_samples):
                step = range_start + (range_end - range_start) * float(
                    q
                ) / float(num_samples - 1)
                z[:, q, dim_j] = step

            decoded_x = vae.decoder(z.reshape(-1, z_size))

            img_grid = ax_from_tensors(decoded_x, image_size, num_samples)

            ax.imshow(img_grid)
            ax.set_xlabel(f"dim {dim_j}")
            ax.set_ylabel(f"dim {dim_i}")
            ax.set_xticks(
                image_size * np.arange(num_samples) + image_size // 2
            )
            ax.set_xticklabels(
                list(
                    map(
                        lambda x: f"{x:.1f}",
                        np.linspace(range_start, range_end, num_samples),
                    )
                )
            )
            ax.set_yticks(
                image_size * np.arange(num_samples) + image_size // 2
            )
            ax.set_yticklabels(
                list(
                    map(
                        lambda x: f"{x:.1f}",
                        np.linspace(range_start, range_end, num_samples),
                    )
                )
            )

    return fig
