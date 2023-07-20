import os
from typing import cast

import numpy as np
import torch
from shimmer import load_structured_config
from tqdm import tqdm

from simple_shapes_dataset import PROJECT_DIR
from simple_shapes_dataset.config.global_workspace import DomainType
from simple_shapes_dataset.config.root import Config
from simple_shapes_dataset.dataset import SimpleShapesDataModule
from simple_shapes_dataset.modules.domains.pretrained import (
    load_pretrained_domain,
)
from simple_shapes_dataset.modules.domains.visual import VisualDomainModule


def main():
    debug_mode = bool(int(os.getenv("DEBUG", "0")))
    config = load_structured_config(
        PROJECT_DIR / "config",
        Config,
        load_dirs=["save_v_latents"],
        debug_mode=debug_mode,
    )

    data_module = SimpleShapesDataModule(
        config.dataset.path,
        {frozenset(["v"]): 1.0},
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        seed=config.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert (
        config.domain_checkpoint is not None
    ), "Please add domain_checkpoint entry in the configuration"
    assert config.domain_checkpoint.domain_type == DomainType.v

    visual_domain = cast(
        VisualDomainModule,
        load_pretrained_domain(config.domain_checkpoint).module,
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
        for batch in tqdm(iter(dataloader), total=len(dataloader)):
            if split == "train":
                images = batch[frozenset(["v"])]["v"].to(device)
            else:
                images = batch[0]["v"].to(device)
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
