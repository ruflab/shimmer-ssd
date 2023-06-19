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

## Add a domain split
```
shapesd split --dataset_path "/path/to/dataset" --seed 0 --domain_alignement t,v 0.01
```
will create a domain split where 0.01% of the example between domains "t" and "v" will
be aligned.
