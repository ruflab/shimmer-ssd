from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import torch
from shimmer.config import load_structured_config

import wandb
from simple_shapes_dataset import DEBUG_MODE, PROJECT_DIR
from simple_shapes_dataset.config.root import Config
from simple_shapes_dataset.logging import attribute_image_grid, get_pil_image
from simple_shapes_dataset.modules.domains.attribute import AttributeDomainModule
from simple_shapes_dataset.modules.vae import dim_exploration_figure

matplotlib.use("Agg")


def main() -> None:
    config = load_structured_config(
        PROJECT_DIR / "config",
        Config,
        load_dirs=["viz_vae_attr"],
        debug_mode=DEBUG_MODE,
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
    fig = dim_exploration_figure(
        domain_module.vae,
        domain_module.latent_dim,
        device,
        attribute_image_grid,
        num_samples,
        range_start,
        range_end,
    )
    caption = f"VAE_Exploration_from_{range_start}_to_{range_end}"
    image = get_pil_image(fig)

    if config.visualization.explore_vae.wandb_name is not None:
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            id=config.visualization.explore_vae.wandb_name,
            resume=True,
        )
        wandb.log({"vae_exploration": [wandb.Image(image, caption=caption)]})

    else:
        plt.savefig(PROJECT_DIR / f"data/{caption}.pdf")
        plt.show()


if __name__ == "__main__":
    main()
