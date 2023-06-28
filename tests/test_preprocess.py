import torch
import torch.utils.data
from utils import PROJECT_DIR

from simple_shapes_dataset.dataset.dataset import SimpleShapesDataset
from simple_shapes_dataset.dataset.pre_process import attribute_to_tensor


def test_attr_preprocess():
    transform = {
        "attr": attribute_to_tensor,
    }
    dataset = SimpleShapesDataset(
        PROJECT_DIR / "sample_dataset",
        split="train",
        selected_domains=["attr"],
        transforms=transform,
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    item = next(iter(dataloader))
    assert isinstance(item["attr"], torch.Tensor)
    assert item["attr"].shape == (2, 11)
