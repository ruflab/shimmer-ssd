from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING


@dataclass
class Slurm:
    script: str = MISSING
    run_workdir: str = MISSING
    python_env: str = MISSING
    command: str = MISSING

    pre_modules: list[str] = field(default_factory=list)
    run_modules: list[str] = field(default_factory=list)
    args: dict[str, Any] = field(default_factory=dict)

    grid_search: list[str] | None = None
    grid_search_exclude: list[dict[str, Any]] | None = None
