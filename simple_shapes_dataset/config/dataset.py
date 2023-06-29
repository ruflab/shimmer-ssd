from dataclasses import dataclass
from pathlib import Path

from omegaconf import MISSING


@dataclass
class Dataset:
    path: Path = MISSING
