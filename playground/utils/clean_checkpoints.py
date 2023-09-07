import shutil

from shimmer import load_structured_config

import wandb
from simple_shapes_dataset import DEBUG_MODE, PROJECT_DIR
from simple_shapes_dataset.config.root import Config


def main():
    config = load_structured_config(
        PROJECT_DIR / "config",
        Config,
        load_dirs=["train_gw"],
        debug_mode=DEBUG_MODE,
    )

    api = wandb.Api()

    runs = api.runs(f"{config.wandb.entity}/{config.wandb.project}")

    keep_ids = set([])
    for run in runs:
        keep_ids.add(f"{config.wandb.project}-{run.id}")

    checkpoint_dir = config.default_root_dir
    for dir in checkpoint_dir.iterdir():
        if not dir.is_dir():
            continue
        if dir.name in keep_ids:
            continue
        print("Removing", dir)
        shutil.rmtree(dir)


if __name__ == "__main__":
    main()
