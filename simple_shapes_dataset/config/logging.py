from dataclasses import dataclass


@dataclass
class Logging:
    log_train_medias_every_n_epochs: int | None = None
    log_val_medias_every_n_epochs: int | None = None
