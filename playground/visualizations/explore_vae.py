import os
from collections.abc import Sequence
from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from PIL import Image
from shimmer.config import load_structured_config
from shimmer.modules.vae import VAE

from simple_shapes_dataset import PROJECT_DIR
from simple_shapes_dataset.config.root import Config
from simple_shapes_dataset.dataset.pre_process import (
    UnnormalizeAttributes,
    tensor_to_attribute,
)
from simple_shapes_dataset.logging import figure_grid, get_pil_image
from simple_shapes_dataset.modules.domains.attribute import (
    AttributeDomainModule,
)

matplotlib.use("Agg")


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


def plot_vae(
    vae: VAE,
    z_size: int,
    device: torch.device,
    num_samples: int = 5,
    range_start: int = -6,
    range_end: int = 6,
    image_size: int = 32,
    plot_dims: Sequence[int] | None = None,
    fig_dim: int = 5,
    dpi: int = 100,
) -> plt.Figure:
    possible_dims = plot_dims or np.arange(z_size)

    fig_size = (len(possible_dims) - 1) * fig_dim

    fig = cast(
        plt.Figure,
        plt.figure(
            constrained_layout=True, figsize=(fig_size, fig_size), dpi=dpi
        ),
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
                torch.randn(z_size)
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

            img_grid = get_figure(image_size, decoded_x, num_samples)

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


def main() -> None:
    debug_mode = bool(int(os.getenv("DEBUG", "0")))
    config = load_structured_config(
        PROJECT_DIR / "config",
        Config,
        load_dirs=["viz-vae"],
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

    num_samples = config.visualization.explore_vae.num_samples
    range_start = config.visualization.explore_vae.range_start
    range_end = config.visualization.explore_vae.range_end
    fig = plot_vae(
        domain_module.vae,
        domain_module.latent_dim,
        device,
        num_samples,
        range_start,
        range_end,
    )
    caption = f"VAE_Exploration_from_{range_start}_to_{range_end}"
    image = get_pil_image(fig)
    plt.savefig(PROJECT_DIR / f"data/{caption}.pdf")
    # plt.show()
    # if config.visualization.explore_vae.wandb_name is not None:
    #     wandb.init(
    #         project=config.wandb.project,
    #         entity=config.wandb.entity,
    #         id=config.visualization.explore_vae.wandb_name,
    #         resume=True,
    #     )
    #     wandb.log({"vae_exploration": [wandb.Image(image, caption=caption)]})
    #
    # else:
    #     plt.show()


if __name__ == "__main__":
    main()
