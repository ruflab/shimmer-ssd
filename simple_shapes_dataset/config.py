import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from shimmer import __version__

from simple_shapes_dataset.types import Config


@dataclass
class ParsingContext:
    in_interpolation: bool = False
    is_escaped: bool = False
    interpolation_key: str = ""


def from_dotlist(
    dotlist: Sequence[str], data: Mapping[str, Any], full_key: str | None = None
) -> str:
    if full_key is None:
        full_key = ".".join(dotlist)
    if len(dotlist) == 0 or dotlist[0] not in data:
        raise ValueError(f"{full_key} cannot be interpolated because does not exist.")
    elif len(dotlist) == 1:
        return str(data[dotlist[0]])
    else:
        return from_dotlist(dotlist[1:], data[dotlist[0]], full_key)


def interpolate(
    query: str, data: Mapping[str, Any], context: ParsingContext | None = None
) -> str:
    if not len(query):
        return ""

    if context is None:
        context = ParsingContext()

    letter, rest = query[0], query[1:]
    match letter:
        case "\\" if not context.is_escaped:
            return interpolate(
                rest,
                data,
                ParsingContext(
                    context.in_interpolation,
                    is_escaped=True,
                ),
            )
        case "{" if not context.is_escaped:
            return interpolate(
                rest,
                data,
                ParsingContext(True, False, ""),
            )
        case "}" if context.in_interpolation and not context.is_escaped:
            return from_dotlist(
                context.interpolation_key.split("."), data
            ) + interpolate(rest, data, ParsingContext(False, False, ""))
        case x if context.in_interpolation:
            return interpolate(
                rest,
                data,
                ParsingContext(
                    True,
                    False,
                    context.interpolation_key + x,
                ),
            )
        case "{" | "}" | "\\" if context.is_escaped:
            return letter + interpolate(rest, data, ParsingContext(False, False, ""))
        case x if context.is_escaped:
            return (
                "\\"
                + letter
                + interpolate(rest, data, ParsingContext(False, False, ""))
            )
        case x:
            return letter + interpolate(rest, data, ParsingContext(False, False, ""))


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
