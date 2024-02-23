import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, model_validator
from shimmer import __version__

from simple_shapes_dataset.types import Config


def parse_args(argv: list[str] | None = None) -> dict[str, Any]:
    """
    Parse argument list into a dictionary.
    Nested dict can be provided using dot notation:
        "a.b=1" will create {"a": {"b": 1}}.
    All keys must have a value and be separated by a "="
    """
    if argv is None:
        argv = sys.argv[1:]

    config: dict[str, Any] = {}
    for arg in argv:
        idx = arg.find("=")
        if idx == -1:
            raise ValueError(f'{arg} has no value. Use "{arg}=value" instead.')
        keys = arg[0:idx].split(".")
        value = arg[idx + 1 :]
        c = config
        for key in keys[:-1]:
            if key not in c.keys():
                c[key] = {}
            c = c[key]
        c[keys[-1]] = value
    return config


def merge_dicts(a: dict[str, Any], b: dict[str, Any]):
    """
    Deep merge two dicts. a will be updated with values in b.
    Example:
        a = {"a": {"b": 1, "c": 3}}
        b = {"a": {"b": 2}}
        merge_dicts(a, b)
        assert a == {"a": {"b": 2, "c": 3}}
    """
    for k in b.keys():
        if k in a:
            if isinstance(a[k], dict) and isinstance(b[k], dict):
                merge_dicts(a[k], b[k])
            else:
                a[k] = b[k]
        else:
            a[k] = b[k]


def load_config(
    path: str | Path,
    load_files: list[str] | None = None,
    use_cli: bool = True,
    debug_mode: bool = False,
    argv: list[str] | None = None,
) -> Config:
    config_path = Path(path)
    if not config_path.is_dir():
        raise FileNotFoundError(f"Config path {config_path} does not exist.")

    load_files = ["default.yaml"]
    if load_files is not None:
        load_files.extend(load_files)
    load_files.append("local.yaml")

    if debug_mode:
        load_files.append("debug.yaml")

    config_dict: dict[str, Any] = {}
    for file in load_files:
        path_file = config_path / file
        if not path_file.is_file():
            raise FileNotFoundError(f"Config file {path_file} does not exist.")
        with open(path_file, "r") as f:
            merge_dicts(config_dict, yaml.safe_load(f))

    cli_config = {}
    if use_cli:
        cli_config = parse_args(argv)
        merge_dicts(config_dict, cli_config)

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
