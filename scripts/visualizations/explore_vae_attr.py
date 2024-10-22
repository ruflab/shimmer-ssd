import matplotlib
import matplotlib.pyplot as plt
import torch
import wandb

from shimmer_ssd import DEBUG_MODE, PROJECT_DIR
from shimmer_ssd.ckpt_migrations import migrate_model
from shimmer_ssd.config import load_config
from shimmer_ssd.logging import attribute_image_grid, get_pil_image
from shimmer_ssd.modules.domains.attribute import AttributeDomainModule
from shimmer_ssd.modules.vae import dim_exploration_figure

matplotlib.use("Agg")


def main() -> None:
    config = load_config(
        PROJECT_DIR / "config",
        load_files=["viz_vae_attr.yaml"],
        debug_mode=DEBUG_MODE,
    )

    if config.visualization is None:
        raise ValueError("Visualization config should be defined for this script.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = config.default_root_dir / config.visualization.explore_vae.checkpoint
    migrate_model(ckpt_path, PROJECT_DIR / "migrations" / "attr_mod")
    domain_module = AttributeDomainModule.load_from_checkpoint(ckpt_path)
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
    main()
    main()
    main()
