from auto_sbatch import ExperimentHandler, GridSearch, SBatch
from omegaconf import OmegaConf
from shimmer import load_config

from simple_shapes_dataset import DEBUG_MODE, LOGGER, PROJECT_DIR


def main():
    LOGGER.debug(f"DEBUG_MODE: {DEBUG_MODE}")

    config = load_config(
        PROJECT_DIR / "config",
        debug_mode=DEBUG_MODE,
    )

    OmegaConf.resolve(config)

    handler = ExperimentHandler(
        config.slurm.script,
        str(PROJECT_DIR.absolute()),
        config.slurm.run_workdir,
        config.slurm.python_env,
        pre_modules=config.slurm.pre_modules,
        run_modules=config.slurm.run_modules,
        setup_experiment=False,
        exclude_in_rsync=["tests", "sample_dataset", ".vscode", ".github"],
    )

    grid_search = None
    extra_config = config.__shimmer__.cli

    # Add all grid search parameters as parameters to auto_sbatch
    if config.slurm.grid_search is not None:
        extra_config = OmegaConf.unsafe_merge(
            OmegaConf.from_dotlist(
                [
                    f"{arg}="
                    + str(OmegaConf.select(config, arg, throw_on_missing=True))
                    for arg in config.slurm.grid_search
                ]
            ),
            extra_config,
        )

        grid_search = GridSearch(
            config.slurm.grid_search, config.slurm.grid_search_exclude
        )

    sbatch = SBatch(
        config.slurm.args,
        extra_config,
        grid_search=grid_search,
        experiment_handler=handler,
    )
    sbatch.run(config.slurm.command, schedule_all_tasks=True)


if __name__ == "__main__":
    main()
