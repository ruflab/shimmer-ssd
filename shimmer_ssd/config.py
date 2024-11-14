import warnings
from pathlib import Path
from typing import Any

from cfg_tools import load_config_files, merge_dicts
from pydantic import ValidationError
from pydantic_core import InitErrorDetails
from rich import print as rprint
from ruamel.yaml import YAML
from shimmer import __version__

from shimmer_ssd.types import Config


def use_deprecated_vals(config: Config) -> Config:
    # use deprecated values
    if config.global_workspace.domain_args is not None:
        config.domain_data_args = config.global_workspace.domain_args
        warnings.warn(
            "Deprecated `config.global_workspace.domain_args`, "
            "use `config.domain_data_args` instead",
            DeprecationWarning,
            stacklevel=2,
        )
    if config.global_workspace.domains is not None:
        config.domains = config.global_workspace.domains
        warnings.warn(
            "Deprecated `config.global_workspace.domains`, "
            "use `config.domains` instead",
            DeprecationWarning,
            stacklevel=2,
        )
    if config.global_workspace.domain_proportions is not None:
        config.domain_proportions = config.global_workspace.domain_proportions
        warnings.warn(
            "Deprecated `config.global_workspace.domain_proportions`, "
            "use `config.domain_proportions` instead",
            DeprecationWarning,
            stacklevel=2,
        )
    return config


def load_config(
    path: str | Path,
    load_files: list[str] | None = None,
    use_cli: bool = True,
    debug_mode: bool = False,
    argv: list[str] | None = None,
) -> Config:
    path = Path(path)
    conf_files = []
    if load_files is not None:
        conf_files.extend(load_files)
    if (path / "local.yaml").exists():
        conf_files.append("local.yaml")

    if debug_mode:
        conf_files.append("debug.yaml")
        if not (path / "debug.yaml").exists():
            (path / "debug.yaml").touch()

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

    def make_missing_dict(loc: list[str | int], val: Any) -> Any:
        if not len(loc):
            return val
        if isinstance(loc[0], str):
            return {loc[0]: make_missing_dict(loc[1:], val)}
        elif loc[0] == 0:
            return [make_missing_dict(loc[1:], val)]

    yaml = YAML()
    for _ in range(2):
        try:
            return use_deprecated_vals(Config.model_validate(config_dict))
        except ValidationError as e:
            printed_header = False
            other_errors: list[InitErrorDetails] = []
            ask_should_save = False
            set_config_dynamically = False
            new_vals: list[Any] = []
            for error in e.errors():
                if error["type"] == "missing":
                    if not printed_header:
                        set_config = input(
                            "Your config is missing some values. Do you want to "
                            "set them dynamically? [Y/n]"
                        )
                        set_config_dynamically = set_config.lower() in ["y", "yes", ""]
                        if not set_config_dynamically:
                            raise e

                        printed_header = True

                    error_name = ".".join(map(str, error["loc"]))
                    rprint(f"[blue]{error_name}[/blue]: ", end="")
                    new_val = input()
                    new_conf = make_missing_dict(list(error["loc"]), new_val)
                    new_vals.append(new_conf)
                    merge_dicts(config_dict, new_conf)
                    ask_should_save = True
                else:
                    init_error = InitErrorDetails(
                        type=error["type"],
                        loc=error["loc"],
                        input=error["input"],
                    )
                    if "ctx" in error:
                        init_error["ctx"] = error["ctx"]
                    other_errors.append(init_error)

            if ask_should_save:
                rprint(
                    "Do you want to save the values in "
                    "`[green]config/local.yaml[/green]`? [Y/n]",
                    end="",
                )
                do_save = input()
                if do_save.lower() in ["yes", "y", ""]:
                    local_file = {}
                    if (path / "local.yaml").exists():
                        with open(path / "local.yaml") as f:
                            local_file = yaml.load(f)
                    for new_val in new_vals:
                        merge_dicts(local_file, new_val)
                    with open(path / "local.yaml", "w") as f:
                        yaml.dump(local_file, f)

            if len(other_errors):
                raise ValidationError.from_exception_data(e.title, other_errors) from e
    raise ValueError("Could not parse config")
