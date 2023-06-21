from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch.utils.data as torchdata

from simple_shapes_dataset.cli.utils import get_deterministic_name
from simple_shapes_dataset.modules.modality import AVAILABLE_MODALITIES


class SimpleShapesDataset(torchdata.Dataset):
    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        selected_modalities: list[str],
        modality_proportions: dict[frozenset[str], float],
        seed: int,
        transforms: dict[str, Callable[[Any], Any]] | None = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.modality_proportions = modality_proportions

        self.selected_modalities = selected_modalities
        self.modalities = {}

        for modality in self.selected_modalities:
            transform = None
            if transforms is not None and modality in transforms:
                transform = transforms[modality]
            self.modalities[modality] = AVAILABLE_MODALITIES[modality](
                dataset_path, split, transform
            )

        modality_split_name = get_deterministic_name(
            modality_proportions, seed
        )

        modality_split_path = (
            self.dataset_path
            / f"{split}_{modality_split_name}_modality_split.npy"
        )
        if not modality_split_path.exists():
            modality_alignment = [
                f'--modality_alignment {",".join(sorted(list(modality)))} {prop}'
                for modality, prop in modality_proportions.items()
            ]
            raise ValueError(
                "Modality split not found. "
                "To create it, use `shapesd split "
                f'--dataset_path "{str(self.dataset_path.resolve())}" '
                f"--seed {seed} {' '.join(modality_alignment)}`"
            )
        self.modality_split = np.load(
            modality_split_path, allow_pickle=True
        ).item()

    def __len__(self) -> int:
        for modality in self.modalities.values():
            return len(modality)
        return 0

    def __getitem__(self, index) -> dict[str, Any]:
        return {
            modality_name: modality[index]
            for modality_name, modality in self.modalities.items()
        }
