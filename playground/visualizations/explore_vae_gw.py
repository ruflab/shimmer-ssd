import os
from collections.abc import Sequence
from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from matplotlib.gridspec import GridSpec
from PIL.Image import Image
from shimmer.config import load_structured_config
from shimmer.modules.lightning.global_workspace import (
    VariationalGlobalWorkspaceLightningModule,
)
from torchvision.utils import make_grid

import wandb
from simple_shapes_dataset import PROJECT_DIR
from simple_shapes_dataset.config.root import Config
from simple_shapes_dataset.logging import attribute_image_grid, get_pil_image
from simple_shapes_dataset.modules.domains.pretrained import (
    load_pretrained_domains,
)
from simple_shapes_dataset.modules.domains.visual import (
    VisualLatentDomainModule,
)

matplotlib.use("Agg")


def image_grid_from_v_tensor(
    samples: torch.Tensor,
    _: int,
    ncols: int,
) -> Image:
    image = make_grid(samples, nrow=ncols, pad_value=1).detach()
    return F.to_pil_image(image)


def dim_exploration_figure(
    module: VariationalGlobalWorkspaceLightningModule,
    z_size: int,
    device: torch.device,
    domain: str,
    num_samples: int = 5,
    range_start: int = -6,
    range_end: int = 6,
    image_size: int = 32,
    plot_dims: Sequence[int] | None = None,
    fig_dim: int = 5,
) -> plt.Figure:
    possible_dims = plot_dims or np.arange(z_size)

    fig_size = (len(possible_dims) - 1) * fig_dim

    fig = cast(
        plt.Figure,
        plt.figure(
            constrained_layout=True, figsize=(fig_size, fig_size), dpi=1
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

            decoded_z = module.global_workspace.decode(z.reshape(-1, z_size))[
                domain
            ]
            decoded_x = module.decode_domain(decoded_z, domain)

            match domain:
                case "v":
                    img_grid = image_grid_from_v_tensor(
                        decoded_x, image_size, num_samples
                    )
                case "v_latents":
                    decoded_img = cast(
                        VisualLatentDomainModule,
                        module.domain_modules[domain],
                    ).decode_images(decoded_x)
                    img_grid = image_grid_from_v_tensor(
                        decoded_img,
                        image_size,
                        num_samples,
                    )
                case "attr":
                    img_grid = attribute_image_grid(
                        decoded_x, image_size, num_samples
                    )
                case _:
                    raise NotImplementedError

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
        load_dirs=["viz_vae_gw"],
        debug_mode=debug_mode,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    domain_description = load_pretrained_domains(
        config.global_workspace.domains
    )

    domain_module = cast(
        VariationalGlobalWorkspaceLightningModule,
        VariationalGlobalWorkspaceLightningModule.load_from_checkpoint(
            config.visualization.explore_gw.checkpoint,
            domain_description=domain_description,
        ),
    )
    domain_module.eval().freeze()

    num_samples = config.visualization.explore_gw.num_samples
    range_start = config.visualization.explore_gw.range_start
    range_end = config.visualization.explore_gw.range_end
    fig = dim_exploration_figure(
        domain_module,
        domain_module.global_workspace.latent_dim,
        device,
        config.visualization.explore_gw.domain,
        num_samples,
        range_start,
        range_end,
    )
    caption = f"VAE_Exploration_from_{range_start}_to_{range_end}"
    image = get_pil_image(fig)

    if config.visualization.explore_gw.wandb_name is not None:
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            id=config.visualization.explore_gw.wandb_name,
            resume=True,
        )
        wandb.log({"vae_exploration": [wandb.Image(image, caption=caption)]})

    else:
        plt.show()
        plt.savefig(PROJECT_DIR / f"data/{caption}.pdf")


if __name__ == "__main__":
    main()
