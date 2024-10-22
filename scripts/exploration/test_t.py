from lightning.pytorch import seed_everything
from simple_shapes_dataset import SimpleShapesDataModule, get_default_domains

from shimmer_ssd import DEBUG_MODE, PROJECT_DIR
from shimmer_ssd.ckpt_migrations import migrate_model
from shimmer_ssd.config import load_config
from shimmer_ssd.modules.domains.text import TextDomainModule


def main():
    config = load_config(
        PROJECT_DIR / "config",
        load_files=["exp_test_t.yaml"],
        debug_mode=DEBUG_MODE,
    )

    if config.exploration is None:
        raise ValueError("Exploration config should be set for this script")

    seed_everything(config.seed, workers=True)

    data_module = SimpleShapesDataModule(
        config.dataset.path,
        get_default_domains(["t", "v"]),
        {frozenset(["t"]): 1.0, frozenset(["v"]): 1.0},
        batch_size=config.training.batch_size,
        max_train_size=config.dataset.max_train_size,
        num_workers=config.training.num_workers,
        domain_args={
            "t": {"latent_filename": config.domain_modules.text.latent_filename}
        },
    )

    val_samples = data_module.get_samples("val", 32)
    test_samples = data_module.get_samples("test", 32)
    for domains in val_samples:
        for domain in domains:
            val_samples[frozenset([domain])] = {domain: val_samples[domains][domain]}
            test_samples[frozenset([domain])] = {domain: test_samples[domains][domain]}
        break

    ckpt_path = config.default_root_dir / config.exploration.gw_checkpoint
    migrate_model(ckpt_path, PROJECT_DIR / "migrations" / "text_mod")
    module = TextDomainModule.load_from_checkpoint(ckpt_path)
    module.freeze()
    print(val_samples[frozenset({"t"})]["t"]["caption"][8].item())


if __name__ == "__main__":
    main()

    main()
