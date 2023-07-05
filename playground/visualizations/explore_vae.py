import os
from collections.abc import Sequence
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from PIL import Image
from shimmer.config import load_config

from simple_shapes_dataset import PROJECT_DIR
from simple_shapes_dataset.config.root import Config
from simple_shapes_dataset.dataset.pre_process import (
    UnnormalizeAttributes,
    tensor_to_attribute,
)
from simple_shapes_dataset.logging import figure_grid
from simple_shapes_dataset.modules.domains.attribute import (
    AttributeDomainModule,
)


def get_figure(
    image_size: int,
    samples: Sequence[torch.Tensor],
    ncols: int = 8,
    dpi: float = 100,
) -> Image.Image:
    unnormalizer = UnnormalizeAttributes(image_size)
    attributes = unnormalizer(tensor_to_attribute(samples))

    categories = attributes.category.cpu().numpy()
    locations = torch.stack([attributes.x, attributes.y], dim=1).cpu().numpy()
    colors = (
        (
            torch.stack(
                [
                    attributes.color_r,
                    attributes.color_g,
                    attributes.color_b,
                ],
                dim=1,
            )
        )
        .cpu()
        .numpy()
    )
    sizes = attributes.size.cpu().numpy()
    rotations = attributes.rotation.cpu().numpy()
    return figure_grid(
        categories, locations, sizes, rotations, colors, image_size, ncols, dpi
    )


def main() -> None:
    debug_mode = bool(int(os.getenv("DEBUG", "0")))
    config = load_config(
        PROJECT_DIR / "config",
        load_dirs=["viz-vae"],
        structure=Config,
        debug_mode=debug_mode,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    domain_module = cast(
        AttributeDomainModule,
        AttributeDomainModule.load_from_checkpoint(
            config.visualization.explore_vae.checkpoint
        ),
    )
    domain_module.eval().freeze()

    n = 15
    start = -3
    end = 3
    imsize = 32
    z_size = domain_module.latent_dim
    fig_size = (z_size - 1) * 5

    fig = plt.figure(constrained_layout=True, figsize=(fig_size, fig_size))
    gs = GridSpec(z_size - 1, z_size - 1, figure=fig)

    for dim_i in range(domain_module.latent_dim - 1):
        for dim_j in range(dim_i + 1, domain_module.latent_dim):
            ax = fig.add_subplot(gs[dim_j - 1, dim_i])

            z = (
                torch.randn(domain_module.latent_dim)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(n, n, -1)
                .to(device)
            )

            for i in range(n):
                step = start + (end - start) * float(i) / float(n - 1)
                z[i, :, dim_i] = step
            for j in range(n):
                step = start + (end - start) * float(j) / float(n - 1)
                z[:, j, dim_j] = step

            decoded_x = domain_module.vae.decoder(
                z.reshape(-1, domain_module.latent_dim)
            )

            img_grid = get_figure(imsize, decoded_x, n)

            ax.imshow(img_grid)
            ax.set_xlabel(f"dim {dim_j}")
            ax.set_ylabel(f"dim {dim_i}")
            ax.set_xticks(imsize * np.arange(n) + imsize // 2)
            ax.set_xticklabels(
                list(map(lambda x: f"{x:.1f}", np.linspace(start, end, n)))
            )
            ax.set_yticks(imsize * np.arange(n) + imsize // 2)
            ax.set_yticklabels(
                list(map(lambda x: f"{x:.1f}", np.linspace(start, end, n)))
            )

    plt.show()


if __name__ == "__main__":
    main()
