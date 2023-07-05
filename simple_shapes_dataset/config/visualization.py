from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import MISSING


@dataclass
class ExploreVAE:
    checkpoint: Path = MISSING


@dataclass
class Visualization:
    explore_vae: ExploreVAE = field(default_factory=ExploreVAE)
