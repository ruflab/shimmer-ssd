from typing import cast

from lightning.pytorch import seed_everything
from shimmer import load_structured_config

from simple_shapes_dataset import DEBUG_MODE, PROJECT_DIR
from simple_shapes_dataset.config.root import Config
from simple_shapes_dataset.dataset import SimpleShapesDataModule
from simple_shapes_dataset.modules.domains.text import TextDomainModule


def main():
    config = load_structured_config(
        PROJECT_DIR / "config",
        Config,
        load_dirs=["exp_test_t"],
        debug_mode=DEBUG_MODE,
    )

    seed_everything(config.seed, workers=True)

    data_module = SimpleShapesDataModule(
        config.dataset.path,
        {frozenset(["t"]): 1.0, frozenset(["v"]): 1.0},
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        domain_args={
            "t": {"latent_filename": config.domain_modules.text.latent_filename}
        },
    )

    val_samples = data_module.get_samples("val", 32)
    test_samples = data_module.get_samples("test", 32)
    for domains in val_samples.keys():
        for domain in domains:
            val_samples[frozenset([domain])] = {domain: val_samples[domains][domain]}
            test_samples[frozenset([domain])] = {domain: test_samples[domains][domain]}
        break

    module = cast(
        TextDomainModule,
        TextDomainModule.load_from_checkpoint(config.exploration.gw_checkpoint),
    )
    module.freeze()
    print(val_samples[frozenset({"t"})]["t"]["caption"][8].item())


if __name__ == "__main__":
    main()
