from auto_sbatch import ExperimentHandler, GridSearch, SBatch

from simple_shapes_dataset import DEBUG_MODE, LOGGER, PROJECT_DIR
from simple_shapes_dataset.config import load_config


def main():
    LOGGER.debug(f"DEBUG_MODE: {DEBUG_MODE}")

    config = load_config(
        PROJECT_DIR / "config",
        debug_mode=DEBUG_MODE,
    )

    if config.slurm is None:
        raise ValueError("slurm config should be defined for this script")

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
    grid_search_config = config.slurm.grid_search

    # Add all grid search parameters as parameters to auto_sbatch
    if grid_search_config is not None:
        grid_search = GridSearch(grid_search_config, config.slurm.grid_search_exclude)
        extra_config = {
            key: val
            for key, val in config.__shimmer__.cli.items()
            if key not in grid_search_config
        }

    sbatch = SBatch(
        config.slurm.args,
        extra_config,
        grid_search=grid_search,
        experiment_handler=handler,
    )
    sbatch.run(config.slurm.command, schedule_all_tasks=True)


if __name__ == "__main__":
    main()
