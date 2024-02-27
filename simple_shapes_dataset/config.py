from pathlib import Path

from cfg_tools import load_config_files
from shimmer import __version__

from simple_shapes_dataset.types import Config


def load_config(
    path: str | Path,
    load_files: list[str] | None = None,
    use_cli: bool = True,
    debug_mode: bool = False,
    argv: list[str] | None = None,
) -> Config:
    config_dict, cli_config = load_config_files(
        path, load_files, use_cli, debug_mode, argv
    )

    config_dict.update(
        {
            "__shimmer__": {
                "version": __version__,
                "debug": debug_mode,
                "cli": cli_config,
            }
        }
    )

    return Config.model_validate(config_dict)
