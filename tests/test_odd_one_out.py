from utils import PROJECT_DIR

from simple_shapes_dataset.dataset.domain import get_default_domains
from simple_shapes_dataset.dataset.downstream.odd_one_out.dataset import (
    OddOneOutDataset,
)


def test_dataset():
    selected_domains = ["v", "attr"]
    dataset = OddOneOutDataset(
        PROJECT_DIR / "sample_dataset",
        split="train",
        domain_classes=get_default_domains(selected_domains),
    )

    assert len(dataset) == 4

    item = dataset[0]
    for domain in ["v", "attr", "target"]:
        assert domain in item
