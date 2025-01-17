from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import click
import numpy as np
import torch
from simple_shapes_dataset import (
    SimpleShapesDataModule,
    color_blind_visual_domain,
    get_default_domains,
)
from tqdm import tqdm

from shimmer_ssd import DEBUG_MODE, LOGGER, PROJECT_DIR
from shimmer_ssd.config import DomainModuleVariant, LoadedDomainConfig, load_config
from shimmer_ssd.modules.domains.pretrained import load_pretrained_module
from shimmer_ssd.modules.domains.visual import VisualDomainModule


def save_v_latents(
    checkpoin_path: Path,
    dataset_path: Path | None = None,
    latent_name: str | None = None,
    config_path: Path | None = None,
    debug_mode: bool | None = None,
    log_config: bool = False,
    extra_config_files: list[str] | None = None,
    argv: list[str] | None = None,
):
    if config_path is None:
        config_path = PROJECT_DIR / "config"
    if debug_mode is None:
        debug_mode = DEBUG_MODE
    if argv is None:
        argv = []

    LOGGER.debug(f"Debug mode: {debug_mode}")

    config = load_config(
        config_path,
        load_files=extra_config_files,
        debug_mode=debug_mode,
        log_config=log_config,
        argv=argv,
    )

    if dataset_path is None:
        dataset_path = config.dataset.path

    additional_transforms: dict[str, list[Callable[[Any], Any]]] = {}
    if config.domain_modules.visual.color_blind:
        additional_transforms["v"] = [color_blind_visual_domain]

    data_module = SimpleShapesDataModule(
        dataset_path,
        get_default_domains(["v"]),
        {frozenset(["v"]): 1.0},
        batch_size=config.training.batch_size,
        max_train_size=config.dataset.max_train_size,
        num_workers=config.training.num_workers,
        seed=config.seed,
        additional_transforms=additional_transforms,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    visual_domain = cast(
        VisualDomainModule,
        load_pretrained_module(
            LoadedDomainConfig(
                domain_type=DomainModuleVariant.v, checkpoint_path=checkpoin_path
            )
        ),
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

        latent_name = latent_name or (checkpoin_path.stem + ".npy")
        path = dataset_path / f"saved_latents/{split}/{latent_name}"
        print(f"Saving in {path}.")
        np.save(path, latent_vectors)


@click.command(
    "v",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    },
    help="Extract and save the visual VAE latent representations.",
)
@click.argument(
    "model_checkpoint",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),  # type: ignore
)
@click.option(
    "--dataset_path",
    "-p",
    default=None,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),  # type: ignore
)
@click.option(
    "--latent_name",
    "-n",
    default=None,
    type=str,
    help="Name of the latent file to use in `presaved_path`.",
)
@click.option(
    "--config_path",
    "-c",
    default=None,
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),  # type: ignore
)
@click.option("--debug", "-d", is_flag=True, default=None)
@click.option("--log_config", is_flag=True, default=False)
@click.option(
    "--extra_config_files",
    "-e",
    multiple=True,
    type=str,
    help=(
        "Additional files to `local.yaml` to load in the config path. "
        "By default `save_v_latents.yaml`"
    ),
)
@click.pass_context
def save_v_latents_command(
    ctx: click.Context,
    model_checkpoint: Path,
    dataset_path: Path | None,
    latent_name: str | None,
    config_path: Path | None,
    debug: bool | None,
    log_config: bool,
    extra_config_files: list[str],
):
    return save_v_latents(
        model_checkpoint,
        dataset_path,
        latent_name,
        config_path,
        debug,
        log_config,
        extra_config_files if len(extra_config_files) else None,
        ctx.args,
    )
