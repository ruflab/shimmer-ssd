# Simple Shapes dataset

## Installation
First clone and cd to the downloaded directory.

Using poetry:

```
poetry install
```

With pip:
```
pip install .
```

## Create dataset
```
shapesd create --output_path "/path/to/dataset"
```
Configuration values:
- `--output_path, -o OUTPUT_PATH` where to save the dataset
- `--seed, -s SEED` random seed (defaults to 0)
- `--img_size` size of the images (defaults to 32)
- `--ntrain` number of train examples (defaults to 500,000)
- `--nval` number of validation examples (defaults to 1000)
- `--ntest` number of test examples (defaults to 1000)
- `--bert_path, -b` which pretrained BERT model to use (defaults to "bert-base-uncased")
- `--domain_alignment, --da, -a` domain alignment to generate (see next section).

For more configurations, see
```
shapesd create --help
```

## Add a domain alignment split
```
shapesd alignment add --dataset_path "/path/to/dataset" --seed 0 --da t,v 0.01 --da t 1. --da v 1.
```
will create an alignment split where 0.01% of the example between domains "t" and "v" will
be aligned, and that contains all data for "t" and "v".

You can list all available alignment with:
```
shapesd alignment list --dataset_path "/path/to/dataset"
```
You can also filter for a particular split with e.g. `--split train` or seed 
(e.g. `--seed 0`)

If you want to restrict the size of the training set to the N first samples only, you
can use `--max_train_size N` or `--ms N`. The same option is also available for `shapesd create`.

## Create an out of distribution split
```
shapesd ood --dataset_path "/path/to/dataset" --seed 0
```
will create an out-of-distribution/in-distribution split for all sets. The output will
be in `/path/to/dataset/ood_splits`. It will also generate a `boundaries_{seed}.csv`
with information about OOD boundaries.

You can obtain different splits with different boundaries by changing the seed.

For more configurations, see
```
shapesd ood --help
```


## Use the dataset
Load the dataset:
```python
import torchvision

from shimmer import ShimmerDataset

from simple_shapes_dataset.dataset import (
    NormalizeAttributes, attribute_to_tensor, get_default_domains
)


dataset = ShimmerDataset(
    "/path/to/dataset",
    split="train",
    domain_classes=get_default_domains(["v", "attr"]),  # Will only load the visual and attr domains

    # transform to apply to the domains domain
    transforms={
        "v": torchvision.transforms.ToTensor(),
        "attr": torchvision.Compose([
            NormalizeAttributes(image_size=32),
            attribute_to_tensor,
        ])
    }
)

item = dataset[0]
assert isinstance(item, dict)

visual_domain = item["v"]
attr_domain = item["attr"]
```

If you need to use the alignment splits, use:
```python
from simple_shapes_dataset.dataset import get_aligned_datasets, get_default_domains

datasets = get_aligned_datasets(
    "/path/to/dataset",
    split="train",
    domain_classes=get_default_domains(["v", "t"]),
    # Node that this will load the file created using `shapesd alignement`
    # if the requested configuration does not exist, it will fail.
    domain_proportions={
        frozenset(["v", "t"]): 0.5,  # proportion of data where visual and text are aligned
        # you also need to provide the proportion for individual domains.
        frozenset(["v"]): 1.0,
        frozenset(["t"]): 1.0,
    },
    seed=0,
    transforms={
        "v": torchvision.transforms.ToTensor(),
    }
)

assert isinstance(datasets, dict)

v_dataset = datasets[frozenset(["v"])]  # all items
t_dataset = datasets[frozenset(["t"])]  # all items
vt_dataset = datasets[frozenset(["v", "t"])]  # 50% of items
```

`v_dataset`, `t_dataset`, and `vt_dataset` are `torch.utils.data.Subset` of
the `ShimmerDataset`.

## Old style dataset
If `train_latent.npy` is not available in your dataset, you may need to specify to path
to the latent vectors (probably something like `train_bert-base-uncased.npy`).


```python
ShimmerDataset(
    "/path/to/dataset",
    split="train",
    domain_classes=get_default_domains(["t"]),  # Will only load the text domain
    domain_args={
        "t": {
            "latent_filename": "bert-base-uncased"
        }
    }
)
```
The `domain_args` argument is also available in `get_aligned_datasets`.
