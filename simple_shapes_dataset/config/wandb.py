from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class WanDB:
    enabled: bool = False
    save_dir: str = MISSING
    project: str = MISSING
    entity: str = MISSING
    reinit: bool = MISSING
