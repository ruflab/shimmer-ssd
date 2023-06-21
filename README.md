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

For more configurations, use
```
shapesd create --help
```

## Add a modality split
```
shapesd split --dataset_path "/path/to/dataset" --seed 0 --modality_alignement t,v 0.01
```
will create a modality split where 0.01% of the example between modalities "t" and "v" will
be aligned.
