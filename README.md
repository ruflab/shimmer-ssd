# shimmer-ssd

GW implementation of the [simple shapes
dataset](https://github.com/ruflab/simple-shapes-dataset).

## Installation
> [!NOTE]
> This repository depends on both [shimmer](https://github.com/ruflab/shimmer)
> and [simple-shapes-dataset](https://github.com/ruflab/simple-shapes-dataset) so make
> sure that you have access to those repos.

First clone and cd to the downloaded directory.

We use [poetry](https://python-poetry.org/) (version >= 2.0) to manage the dependency. Please follow
[these instructions](https://github.com/ruflab/shimmer/blob/main/CONTRIBUTING.md) first 
to setup your environment.

To install the project and dependencies:

```bash
poetry sync [--with dev]
```

## Tutorials
See
[https://github.com/ruflab/shimmer-tutorials](https://github.com/ruflab/shimmer-tutorials)
for tutorials on `shimmer` and this repository.

## Config
All details can be read in the [config docs](./docs/config_parameters.md).

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
Optional arguments:
* `--migration_path` where the path with migrations is located. Defaults to the
migrations provided by this repo.
* `--type`, `-t`, type of migration. One of "gw", "attr_mod", "text_mod", "visual_mod".
Defaults to "gw".

## Pretrained checkpoints
Pretrained model weights can be downloaded here:
[https://zenodo.org/records/14747474](https://zenodo.org/records/14747474).

You can download them using:
```
ssd download checkpoints
```
Optional argument:
* `--path`, `-p`, location to the checkpoints folder. Defaults to `./checkpoints`.

## Tokenizer data
You can download the tokenizer data with:
```
ssd download tokenizer
```
Optional argument:
* `--path`, `-p`, location to the tokenizer folder. Defaults to `./tokenizer`.

It can also be access from this repository:
[https://github.com/ruflab/shimmer-ssd/tree/main/tokenizer](https://github.com/ruflab/shimmer-ssd/tree/main/tokenizer).
