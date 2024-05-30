from collections.abc import Callable
from typing import Any, cast

import numpy as np
import torch
from tqdm import tqdm

from simple_shapes_dataset import DEBUG_MODE, PROJECT_DIR
from simple_shapes_dataset.config import load_config
from simple_shapes_dataset.dataset import SimpleShapesDataModule
from simple_shapes_dataset.dataset.pre_process import color_blind_visual_domain
from simple_shapes_dataset.modules.domains.pretrained import load_pretrained_module
from simple_shapes_dataset.modules.domains.visual import VisualDomainModule
from simple_shapes_dataset.types import DomainType


def main():
    config = load_config(
        PROJECT_DIR / "config",
        load_files=["save_v_latents.yaml"],
        debug_mode=DEBUG_MODE,
    )

    additional_transforms: dict[str, list[Callable[[Any], Any]]] = {}
    if config.domain_modules.visual.color_blind:
        additional_transforms["v"] = [color_blind_visual_domain]

    data_module = SimpleShapesDataModule(
        config.dataset.path,
        {frozenset(["v"]): 1.0},
        batch_size=config.training.batch_size,
        max_size=config.dataset.max_size,
        num_workers=config.training.num_workers,
        seed=config.seed,
        additional_transforms=additional_transforms,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert (
        config.domain_checkpoint is not None
    ), "Please add domain_checkpoint entry in the configuration"
    assert config.domain_checkpoint.domain_type == DomainType.v

    visual_domain = cast(
        VisualDomainModule,
        load_pretrained_module(config.default_root_dir, config.domain_checkpoint),
    )
    visual_domain.to(device)
    visual_domain.freeze()

    data_module.prepare_data()
    data_module.setup()

    dataloaders = {
        "train": data_module.train_dataloader(shuffle=False, drop_last=False),
        "val": data_module.val_dataloader(),
        "test": data_module.test_dataloader(),
    }

    for split, dataloader in dataloaders.items():
        latents: list[np.ndarray] = []

        print(f"Saving {split}.")
        for batch, _, _ in tqdm(iter(dataloader), total=len(dataloader)):
            if split == "train":
                images = batch[frozenset(["v"])]["v"].to(device)
            else:
                images = batch["v"].to(device)
            latent = visual_domain.encode(images)
            latents.append(latent.detach().cpu().numpy())

        latent_vectors = np.concatenate(latents, axis=0)

        path = (
            config.dataset.path
            / f"saved_latents/{split}/{config.presaved_latents_path['v']}"
        )
        print(f"Saving in {path}.")
        np.save(path, latent_vectors)


if __name__ == "__main__":
    main()
