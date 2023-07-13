from dataclasses import dataclass, field
from typing import Any


@dataclass
class Optim:
    lr: float = 1e-4
    max_lr: float = 5e-3
    weight_decay: float = 0.0


@dataclass
class Training:
    batch_size: int = 1024
    num_workers: int = 0

    fast_dev_run: bool = False
    max_steps: int = 100_000
    enable_progress_bar: bool = True

    precision: Any = 32
    float32_matmul_precision: str = "highest"

    optim: Optim = field(default_factory=Optim)
