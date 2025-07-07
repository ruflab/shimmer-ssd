from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from simple_shapes_dataset import (
    SimpleShapesDataModule,
    color_blind_visual_domain,
    get_default_domains,
)
from tqdm import tqdm

from shimmer_ssd import DEBUG_MODE, PROJECT_DIR
from shimmer_ssd.config import DomainModuleVariant, load_config
from shimmer_ssd.modules.domains.pretrained import load_pretrained_module
from shimmer_ssd.modules.domains.visual import VisualDomainModule


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
        get_default_domains(["v"]),
        {frozenset(["v"]): 1.0},
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        seed=config.seed,
        additional_transforms=additional_transforms,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain_checkpoint = None
    for domain in config.domains:
        if domain.domain_type == DomainModuleVariant.v:
            domain_checkpoint = domain

    assert (
        domain_checkpoint is not None
    ), "Please add domain_checkpoint entry in the configuration"
    assert domain_checkpoint.domain_type == DomainModuleVariant.v

    visual_domain = cast(
        VisualDomainModule,
        load_pretrained_module(domain_checkpoint),
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
                images = batch[frozenset(["v"])]["v"].to(device)
            latent = visual_domain.encode(images)
            latents.append(latent.detach().cpu().numpy())

        latent_vectors = np.concatenate(latents, axis=0)

        presaved_path = config.domain_data_args["v_latents"]["presaved_path"]
        Path(
            f"{config.dataset.path}/saved_latents/{split}/"
        ).mkdir(parents=True, exist_ok=True)
        path = config.dataset.path / f"saved_latents/{split}/{presaved_path}"
        print(f"Saving in {path}.")
        np.save(path, latent_vectors)


if __name__ == "__main__":
    main()
