from collections.abc import Sequence
from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from PIL.Image import Image
from shimmer.modules.global_workspace import (
    GlobalWorkspaceBase,
    VariationalGlobalWorkspace,
)
from torchvision.utils import make_grid

import wandb
from simple_shapes_dataset import DEBUG_MODE, PROJECT_DIR
from simple_shapes_dataset.config import load_config
from simple_shapes_dataset.logging import attribute_image_grid, get_pil_image
from simple_shapes_dataset.modules.domains.pretrained import load_pretrained_domains
from simple_shapes_dataset.modules.domains.visual import VisualLatentDomainModule

matplotlib.use("Agg")


def image_grid_from_v_tensor(
    samples: torch.Tensor,
    _: int,
    ncols: int,
) -> Image:
    image = make_grid(samples, nrow=ncols, pad_value=1).detach()
    return F.to_pil_image(image)


def dim_exploration_figure(
    module: GlobalWorkspaceBase,
    z_size: int,
    device: torch.device,
    domain: str,
    num_samples: int = 5,
    range_start: int = -6,
    range_end: int = 6,
    image_size: int = 32,
    plot_dims: Sequence[int] | None = None,
    fig_dim: int = 5,
) -> Figure:
    possible_dims = plot_dims or np.arange(z_size)

    fig_size = (len(possible_dims) - 1) * fig_dim

    fig = cast(
        Figure,
        plt.figure(constrained_layout=True, figsize=(fig_size, fig_size), dpi=1),
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
                step = range_start + (range_end - range_start) * float(p) / float(
                    num_samples - 1
                )
                z[p, :, dim_i] = step
            for q in range(num_samples):
                step = range_start + (range_end - range_start) * float(q) / float(
                    num_samples - 1
                )
                z[:, q, dim_j] = step

            decoded_z = module.decode(z.reshape(-1, z_size))[domain]
            decoded_x = module.decode_domain(decoded_z, domain)

            match domain:
                case "v":
                    img_grid = image_grid_from_v_tensor(
                        decoded_x, image_size, num_samples
                    )
                case "v_latents":
                    decoded_img = cast(
                        VisualLatentDomainModule,
                        module.domain_mods[domain],
                    ).decode_images(decoded_x)
                    img_grid = image_grid_from_v_tensor(
                        decoded_img,
                        image_size,
                        num_samples,
                    )
                case "attr":
                    img_grid = attribute_image_grid(decoded_x, image_size, num_samples)
                case _:
                    raise NotImplementedError

            ax.imshow(img_grid)
            ax.set_xlabel(f"dim {dim_j}")
            ax.set_ylabel(f"dim {dim_i}")
            ax.set_xticks(image_size * np.arange(num_samples) + image_size // 2)
            ax.set_xticklabels(
                list(
                    map(
                        lambda x: f"{x:.1f}",
                        np.linspace(range_start, range_end, num_samples),
                    )
                )
            )
            ax.set_yticks(image_size * np.arange(num_samples) + image_size // 2)
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
    config = load_config(
        PROJECT_DIR / "config",
        load_files=["viz_vae_gw"],
        debug_mode=DEBUG_MODE,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    domain_description, interfaces = load_pretrained_domains(
        config.default_root_dir,
        config.global_workspace.domains,
        config.global_workspace.latent_dim,
        config.global_workspace.encoders.hidden_dim,
        config.global_workspace.encoders.n_layers,
        config.global_workspace.decoders.hidden_dim,
        config.global_workspace.decoders.n_layers,
        is_variational=True,
    )

    domain_module = cast(
        VariationalGlobalWorkspace,
        VariationalGlobalWorkspace.load_from_checkpoint(
            config.default_root_dir / config.visualization.explore_gw.checkpoint,
            domain_mods=domain_description,
            gw_interfaces=interfaces,
        ),
    )
    domain_module.eval().freeze()

    num_samples = config.visualization.explore_gw.num_samples
    range_start = config.visualization.explore_gw.range_start
    range_end = config.visualization.explore_gw.range_end
    fig = dim_exploration_figure(
        domain_module,
        domain_module.gw_mod.latent_dim,
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
