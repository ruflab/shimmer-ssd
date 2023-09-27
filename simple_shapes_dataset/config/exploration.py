from dataclasses import dataclass
from pathlib import Path

from omegaconf import MISSING


@dataclass
class Exploration:
    gw_checkpoint: Path = MISSING
