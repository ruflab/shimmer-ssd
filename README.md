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
For example, if you want to run with a differend seed: `ssd train gw seed=5`.
For nested values, use `.` as a separator: `ssd train gw "training.max_steps=20"`.

Some particularly useful ones:
```bash
ssd train gw t="my-wandb-run-title" d="And an associated description"
```

> [!NOTE]
> There is no "-" or "--" in front of the arguments. They exactly correspond to the
> config values.
> Also, you have to put "=" between the key and the value. This won't work:
> `ssd train gw seed 5`

You can change the location of the config folder (particularly useful if you installed
this repository using pip) with `--config_path`.

## Training scripts

* `ssd train v`: train the image domain module
* `ssd train attr`: train the attribute domain module
* `ssd train t`: train the text domain module
* `ssd train gw`: train a Global Workspace.

All this scripts accept some options:
* `--config_path`, `-c`, path to the folder containing the config files.
* `--debug`, `-d`, whether to start on debug mode.
* `--log_config`, will log the exact config object used for the run.
* `--extra_config_files`, `-e`, list of additional config files to load in addition to
`local.yaml` relative to the `--config_path` 
(use several `-e CONFIG_FILE -e CONFIG_FILE` to add several files).

You can also edit any config files from the config folder as argument without the "-"
or "--" as explained in the previous section.

## Extract visual latent representations
You can extract the visual latent representations of a given checkpoint with:
```
ssd extract v CHECKPOINT_PATH
```
Available options:
* `--dataset_path`, `-p`, path to the simple-shapes-dataset (defaults to the config
value `dataset.pah`).
* `--latent_name`, `-n`, name of the latent file to create (default: CHECKPOINT_PATH
file with extension ".npy").
* `--config_path`, `-c`, path to the folder containing the config files.
* `--debug`, `-d`, whether to start on debug mode.
* `--log_config`, will log the exact config object used for the run.
* `--extra_config_files`, `-e`, list of additional config files to load in addition to
`local.yaml` relative to the `--config_path` 
(use several `-e CONFIG_FILE -e CONFIG_FILE` to add several files).

## Migrate old checkpoint
```
ssd migrate CHECKPOINT_PATH
```

## Pretrained checkpoints
Pretrained model weights can be downloaded here:
[https://zenodo.org/records/14289631](https://zenodo.org/records/14289631).

You can download them using:
```
ssd download checkpoints
```
