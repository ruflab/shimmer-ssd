from dataclasses import dataclass, field


@dataclass
class Optim:
    lr: float = 1e-3
    weight_decay: float = 0.0


@dataclass
class Training:
    batch_size: int = 512
    num_workers: int = 0

    fast_dev_run: bool = False
    max_epochs: int = 100
    enable_progress_bar: bool = True

    optim: Optim = field(default_factory=Optim)
