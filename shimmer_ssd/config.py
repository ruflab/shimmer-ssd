from pathlib import Path

from cfg_tools import load_config_files
from shimmer import __version__

from shimmer_ssd.types import Config


def load_config(
    path: str | Path,
    load_files: list[str] | None = None,
    use_cli: bool = True,
    debug_mode: bool = False,
    argv: list[str] | None = None,
) -> Config:
    conf_files = ["default.yaml"]
    if load_files is not None:
        conf_files.extend(load_files)
    conf_files.append("local.yaml")

    if debug_mode:
        conf_files.append("debug.yaml")

    config_dict, cli_config = load_config_files(path, conf_files, use_cli, argv)

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
