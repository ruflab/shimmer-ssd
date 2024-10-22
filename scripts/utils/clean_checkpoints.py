import shutil

import wandb

from shimmer_ssd import DEBUG_MODE, PROJECT_DIR
from shimmer_ssd.config import load_config


def main():
    config = load_config(
        PROJECT_DIR / "config",
        load_files=["train_gw.yaml"],
        debug_mode=DEBUG_MODE,
    )

    api = wandb.Api()  # type: ignore

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
