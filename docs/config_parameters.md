# Available config parameters in the project
Note: We use a "dot" separated syntax to represent nested values in the yaml config.
`dataset.path` refers to:
```yaml
dataset:
    path: 
```

## Create the config folder
You first need to generate your local `config` folder with:
```
ssd config create
```
This will locally create a `config` folder for the project.
Additional arguments:
* `--path`, `-p`, where to save the config files (defaults to `./config`)
* `--force`, `-f`, whether to override files if the destination of `--path` already
exists.

The folder contains several files:
* `main.yaml` (or historically `local.yaml`) where you will put most of your
configuration
* `debug.yaml` some config overides used when starting scripts in "debug mode" (see 
later section).
* other config files containing overides related to specific scripts. For example, 
`train_v.yaml` is also loaded (before `main.yaml`)

### File priority
The files are loaded in the following order (lower priority, to higher priority):
* script specific config files (lowest priority)
* `main.yaml`
* `local.yaml`
* `debug.yaml` (if in debug mode)
* command line arguments (highest priority)


## All config options
```yaml
# Seed used for the alignment splits generated with `shapesd alignment add` and 
# for training.
seed: 0  # (type: int)

# Seed to load the out of distribution data.
ood_seed: null  # (type: int | None)

# Path where wandb logs and checkpoints will be stored.
default_root_dir: "./checkpoints"  # (type: Path)

dataset:
  # Path to the simple-shapes-dataset. Can be downloaded with `shapesd download`
  path: "./simple_shapes_dataset"  # (type: Path)

training:
  batch_size: 2056  # (type: int)
  num_workers: 16  # (type: int)

  #  Number of devices (gpus) to use.
  devices: 1  # (type: int)

  # See https://lightning.ai/docs/pytorch/stable/common/trainer.html#accelerator
  accelerator: "gpu"  # (type: str)

  # See https://lightning.ai/docs/pytorch/stable/common/trainer.html#fast-dev-run
  fast_dev_run: false  # (type: bool)

  # Number of training steps
  max_steps: 200_000  # (type: int)

  enable_progress_bar: true  # (type: bool)

  # See https://lightning.ai/docs/pytorch/stable/common/trainer.html#precision
  # you may want to set to "16-mixed" if your gpu allows mixed precision
  precision: 32  # (type: Any)

  # See https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch-set-float32-matmul-precision
  # you may want to decrease to "medium"
  float32_matmul_precision: "highest"  # (type: str)

  optim:
    # learning rate for the optimizer
    lr: 5e-3  # (type: float)
    weight_decay: 1e-5  # (type: float)
    # max learning rate for the OneCycleScheduler
    max_lr: 5e-3  # (type: float)
    # starting learning rate for the OneCycleScheduler
    start_lr: 5e-4  # (type: float)
    # The ending learning rate for the OneCycleScheduler
    end_lr: 5e-4  # (type: float)
    # See pct_start here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR
    pct_start: 0.2  # (type: float)

wandb:
  # whether to use wandb logging
  enabled: false  # (type: bool)

  # Where to save the logs
  save_dir: ""  # (type: str)

  #The "entity/project" values of your wandb project
  project: ""  # (type: str)
  entity: ""  # (type: str)

  # See https://docs.wandb.ai/ref/python/init/
  reinit: false  # (type: bool)

logging:
  # List of medias (images/text) that will be logged.
  # The list is defined in individual train config (e.g. train_v.yaml, train_gw.yaml) to
  # only affect individual scripts.
  filter_images: null  # (type: Sequence[str] | None)
  # Will log the images or text every x epochs
  log_train_medias_every_n_epochs: 10  # (type: int | None)
  log_val_medias_every_n_epochs: 10  # (type: int | None)

# Add a title to your wandb run
# alias `t`
title: null  # (type: str | None)

# Add a description to your wandb run
# alias `d`
desc: null # (type: str | None)

# Proportion of each domain in the dataset relative to the size of the dataset
domain_proportions: []  # (type: Sequence[DomainProportion])
# For example:
# domain_proportions:
#     - domains: ["attr"]  # possible values: "v", "attr", "t"
#       proportion: 1.0
#     - domains: ["v"]
#       proportion: 1.0
#     - domains: ["v", "attr"]
#       proportion: 0.5

# Configuration specific to the domain modules
domain_modules:
  visual:
    # Num channels of image to configure the VAE
    num_channels: 3  # (type: int)

    # AE hidden dim
    ae_dim: 256  # (type: int)

    # Latent dim of the VAE
    latent_dim: 8  # (type: int)

    # Beta for betaVAE
    beta: 0.1  # (type: float)

    # Whether the model is color blind.
    # It adds a transform on the dataset that averages all color channels.
    # [NOTE] This only works when using the "v" domain and not "v_latents" where
    # visual latent representations are already extracted. In that case, use a different
    # presaved-path in `domain_args`.
    color_blind: false  # (type: bool)

  attribute:
    # Latent dim of the VAE
    latent_dim: 10  # (type: int)

    # Hidden dim of the AE (encoders and decoders)
    hidden_dim: 64  # (type: int)

    # For betaVAE
    beta: 0.05  # (type: float)

    # Coefficients to use for the loss on the object categories
    coef_categories: 1  # (type: float)
    # Coefficients to use for the loss on the attributes
    coef_attributes: 1  # (type: float)

  text:

    # Which file to load for text
    # The file generated with `shapesd create` is "latent"
    # If you use an older version (like if you downloaded directly the dataset) it
    # should be "bert-base-uncased" path to the vocab file
    latent_filename: "latent"  # (type: str)

    # Path to the vocab.json path
    vocab_path: "./tokenizer/vocab.json"  # (type: str)

    # Path to the merge path
    merges_path: "./tokenizer/merges.txt"  # (type: str)

    # Max sequence length of text sequence
    seq_length: 64  # (type: int)

    vocab_size: 822  # (type: int)

    # VAE configuration
    latent_dim: 64  # (type: int)

    hidden_dim: 256  # (type: int)

    beta: 0.1  # (type: float)

    # Whether to remove rotation information from the attribute.
    nullify_rotation: False  # (type: bool)

# Domain params for the active domains of the run.
domains: []  # (type: Sequence[LoadedDomainConfig])
# For example:
# domains:
#       # Path To the pretrained module.
#     - checkpoint_path: ./path/to/attr_checkpoint.ckpt # (type: Path)
#       # Domain to select. For example:
#       domain_type: attr  # (type: DomainModuleVariant)
#       args: {}  # (type: Mapping[str, Any])
#     - checkpoint_path: ./path/to/v_checkpoint.ckpt
#       domain_type: v
```

Here are the available values for `domain_type`. It takes a value of the
enum `DomainModuleVariant` which can be:
 * `attr`: attribute module using a VAE to encode the attribute vector.
 * `attr_legacy`: this is the module used in Devillers et al. paper. 
 There is no VAE and the attributes are used directly as the unimodal latent representations
 * `attr_unpaired`: Same as "attr" but adds an unpaired attributes 
 (information not available in the other domains).
 * `v`: visual VAE 
 * `v_latents`: same as "v", but uses pre-saved latent VAE representation for faster 
 training. This skips the image loading and encoding and only loads latent
 representation. The downside is that you cannot access the default image, but you can 
 reconstruct it with "decode_images".
 `v_latents_unpaired`: same as "v_latents" but adds an unpaired value (radom information not available
 in the other domains).
 * `t`: text domain.
 * `t_attr`


```yaml
# Data related args used by the dataloader
domain_data_args: {}  # (type: Mapping[str, Mapping[str, Any]])

global_workspace:
  # Latent dim of the GW
  latent_dim: 12  # (type: int)

  # Whether to use the fusion GW
  use_fusion_model: false  # (type: bool)

  # Softmax temp for the Softmax distruted selection used by the fusion model
  selection_temperature: 0.2  # (type: float)

  # Whether to learn the logit scale of the contrastive loss like in the clip paper
  learn_logit_scale: false  # (type: bool)

  # Whether to use the VSEPP (https://github.com/fartashf/vsepp) contrastive loss
  # with associated params
  vsepp_contrastive_loss: false  # (type: bool)
  vsepp_margin: 0.2  # (type: float)
  vsepp_measure: "cosine"  # (type: Literal["cosine", "order"])
  vsepp_max_violation: true  # (type: bool)

  # Whether to use linear encoders and decoders for the GW
  linear_domains: false  # (type: bool)
  # Whether to use bias when using linear encoders and decoders
  linear_domains_use_bias: false  # (type: bool)

  # Coefs of each loss. The total loss is computed using the given values and coefs
  # you can select any available loss generated by the loss functions
  loss_coefficients:   # (type: Mapping[str, float])
    cycles: 1.0
    demi_cycles: 1.0
    translations: 1.0
    contrastives: 0.01
    fused: 1.0

  encoders:
    hidden_dim: 32  # (type: int | Mapping[DomainModuleVariant, int])
    # Can either be an int to define the hidden dimension for all domains, or a mapping
    # with a value for each domain. For example:
    # hidden_dim:
    #   attr: 32
    #   v: 12
    # 
    # The model will have an extra linear before and after the n_layers
    # Hence the total will be `2 + n_layers`
    n_layers: 3  # (type: int | Mapping[DomainModuleVariant, int])
    # Can either be an int to define the hidden dimension for all domains, or a mapping
    # with a value for each domain. Example:
    # n_layers:
    #   attr: 3
    #   v: 2

  decoders:
    hidden_dim: 32  # (type: int | Mapping[DomainModuleVariant, int])
    # Can either be an int to define the hidden dimension for all domains, or a mapping
    # with a value for each domain. Example:
    # hidden_dim:
    #   attr: 32
    #   v: 12

    # The model will have an extra linear before and after the n_layers
    # Hence the total will be `2 + n_layers`
    n_layers: 3  # (type: int | Mapping[DomainModuleVariant, int])
    # Can either be an int to define the hidden dimension for all domains, or a mapping
    # with a value for each domain. Example:
    # n_layers:
    #   attr: 3
    #   v: 2

slurm: null  # (type: Slurm | None)
```

When using the slurm script, you can define the slurm value. The different
parameters are defined as in
[https://github.com/ruflab/auto-sbatch](https://github.com/ruflab/auto-sbatch).
In details:
```yaml
slurm:
  # which script to run e.g. scripts/train_gw.py
  script: # (type: str)
  # The work_directory folder will be copied to this folder
  run_workdir: # (type: str)
  # path to python environment
  python_env: # (type: str)
  # Command to run e.g. "python {script_name} {all_params}"
  command: # (type: str) 
  # modules to load BEFORE script are batched
  pre_modules: []  # (type: Sequence[str])
  # modules to load in the batch system
  run_modules: []  # (type: Sequence[str])
  # slurm args. e.g. "-J", "-N", ...
  args: {}  # (type: Mapping[str, Any])
  # grid search params
  grid_search: null  # (type: Mapping[str, Sequence[Any]] | None)
  # grid search arguments to exclude
  grid_search_exclude: null  # (type: Sequence[Mapping[str, Any]] | None)
```

## Config formatting
We also use some custom code to allow some interpolations, described here:
[https://github.com/ruflab/cfg-tools](https://github.com/ruflab/cfg-tools).

### Value interpolation
You can use `#{other.key}` to put the value of `other.key` in its place.

## Command line arguments
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

## Debug mode
You can pass the argument `--debug` in training commands. This means that the
`debug.yaml` file will be loaded to define the config as the fast loaded file.
See "File priority" to see the exact order of loaded files.
