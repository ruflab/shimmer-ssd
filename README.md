# shimmer-ssd

GW implementation of the [simple shapes
dataset](https://github.com/ruflab/simple-shapes-dataset).

## Installation
> [!NOTE]
> This repository depends on both [shimmer](https://github.com/ruflab/shimmer)
> and [simple-shapes-dataset](https://github.com/ruflab/simple-shapes-dataset) so make
> sure that you have access to those repos.

First clone and cd to the downloaded directory.

We use [poetry](https://python-poetry.org/) to manage the dependency. Please follow
[these instructions](https://github.com/ruflab/shimmer/blob/main/CONTRIBUTING.md) first 
to setup your environment.

To install the project and dependencies:

```bash
poetry install --sync
```

## Config
The config files are located in the `config` folder. 

You can see all possible config values in the `shimmer_ssd/config.py` file.
The root is the `Config` file, and all values from the yaml files are mapped to the
corresponding values using [pydantic](https://docs.pydantic.dev/latest/).

We also use some custom code to allow some interpolations, described here:
[https://github.com/bdvllrs/cfg-tools](https://github.com/bdvllrs/cfg-tools).

To load the config, we first try to load script related config (the files given to
`load_files` in `load_config`), the add `local.yaml`, then `debug.yaml` if you start
scripts in debug mode. `local.yaml` is not tracked in github and is used for
user-related config options, like paths to dataset and checkpoints.

When running each script, you can adapt the config with the cli by addind as an option.
For example, if you want to run with a differend seed: `python train_gw.py seed=5`.
For nested values, use `.` as a separator: `python train_gw.py "training.max_steps=20"`.

Some particularly useful ones:
```bash
python train_gw.py t="my-wandb-run-title" d="And an associated description"
```

> [!NOTE]
> There is no "-" or "--" in front of the arguments. They exactly correspond to the
> config values.
> Also, you have to put "=" between the key and the value. This won't work:
> `python train_gw.py seed 5`

## Training scripts
Scripts are located in the `scripts` folder.

* `train_v.py`: train the image domain module
* `train_attr.py`: train the attribute domain module
* `train_t.py`: train the text domain module
* `train_gw.py`: train a Global Workspace.
