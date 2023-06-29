from dataclasses import dataclass


@dataclass
class Training:
    batch_size: int = 512
    num_workers: int = 0
