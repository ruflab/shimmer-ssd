import warnings
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from cfg_tools import ParsedModel, load_config_files, merge_dicts
from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_core import InitErrorDetails
from rich import print as rprint
from ruamel.yaml import YAML
from shimmer import __version__
from shimmer.version import __version__ as shimmer_version
from simple_shapes_dataset import DomainDesc

from shimmer_ssd import PROJECT_DIR


class DomainType(Enum):
    """
    Different domain types available. Each model type is
    described as with a `DomainDesc` which represents a base type and a kind.

    For example "v_latents" has a base of "v" as it a special representation of
    the visual domain.
    """

    v = DomainDesc("v", "v")  # Uses images and visual VAE to encode images
    # Uses pre-saved latent representations extracted from the visual VAE
    v_latents = DomainDesc("v", "v_latents")
    attr = DomainDesc("attr", "attr")
    raw_text = DomainDesc("t", "raw_text")  # raw text representations (str)
    t = DomainDesc("t", "t")  # loads BERT representations of the raw text


class DomainModelVariantType(Enum):
    """
    This is used to select a particular DomainModule.
    Each type can have different "flavors" of domain modules.

    For example "attr" will load the default attribute module that uses a VAE,
    whereas "attr_legacy" will load a domain module that directly passes attributes
    to the GW.
    """

    v = (DomainType.v, "default")
    attr = (DomainType.attr, "default")  # Uses a VAE to encode the attribute vector
    # No VAE, attributes are the unimodal latent representations
    attr_legacy = (DomainType.attr, "legacy")
    # Uses unpaired attributes. The other ones remove unpaired values
    attr_unpaired = (DomainType.attr, "unpaired")
    v_latents = (DomainType.v_latents, "default")
    v_latents_unpaired = (DomainType.v_latents, "unpaired")
    t = (DomainType.t, "default")
    t_attr = (DomainType.t, "t2attr")

    def __init__(self, kind: DomainType, model_variant: str) -> None:
        self.kind = kind
        self.model_variant = model_variant


class Logging(BaseModel):
    # List of medias (images/text) that will be logged.
    # The list is defined in individual train config (e.g. train_v.yaml, train_gw.yaml)
    filter_images: Sequence[str] | None = None
    log_train_medias_every_n_epochs: int | None = 10
    log_val_medias_every_n_epochs: int | None = 10


class Slurm(BaseModel):
    """
    Slurm config for https://github.com/bdvllrs/auto-sbatch
    """

    script: str
    run_workdir: str
    python_env: str
    command: str

    pre_modules: Sequence[str] = []
    run_modules: Sequence[str] = []
    args: Mapping[str, Any] = {}

    grid_search: Mapping[str, Sequence[Any]] | None = None
    grid_search_exclude: Sequence[Mapping[str, Any]] | None = None


class Optim(BaseModel):
    """
    Optimizer config
    """

    # learning rate (max learning rate when using with the default OneCycle
    # learning rate scheduler)
    # see https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR)
    # for information about the other config values
    lr: float = 5e-3
    # TODO: remove as redundant with lr
    max_lr: float = 5e-3
    start_lr: float = 5e-4
    end_lr: float = 5e-4
    pct_start: float = 0.2
    weight_decay: float = 1e-5


class Training(BaseModel):
    """
    Training related config.
    As these config depend on what you are training,
    they are defined in the script related yaml files (e.g. `train_v.yaml`,
    `train_gw.yaml`).
    """

    batch_size: int = 2056
    num_workers: int = 16
    devices: int = 1  # num of devices (gpus) to use
    accelerator: str = "gpu"

    # see https://lightning.ai/docs/pytorch/stable/common/trainer.html#fast-dev-run
    fast_dev_run: bool = False
    # number of training steps
    max_steps: int = 200_000
    enable_progress_bar: bool = True

    # see https://lightning.ai/docs/pytorch/stable/common/trainer.html#precision
    # you may want to set to "16-mixed" if your gpu allows mixed precision
    precision: Any = 32
    # see https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch-set-float32-matmul-precision
    # you may want to decrease to "medium"
    float32_matmul_precision: str = "highest"

    # Optimizer config
    optim: Optim = Optim()


class ExploreVAE(BaseModel):
    # the VAE checkpoint to use
    checkpoint: str
    num_samples: int = 9
    range_start: int = -3
    range_end: int = 3
    wandb_name: str | None = None


class ExploreGW(BaseModel):
    checkpoint: str  # The GW checkpoint to use
    domain: str
    num_samples: int = 9
    range_start: int = -3
    range_end: int = 3
    wandb_name: str | None = None


class Visualization(BaseModel):
    # when exploring a vae
    explore_vae: ExploreVAE
    # when exploring the GW
    explore_gw: ExploreGW


class WanDB(BaseModel):
    enabled: bool = False
    # where to save the logs
    save_dir: str = ""
    # the "entity/project" values of your wandb project
    project: str = ""
    entity: str = ""
    # see https://docs.wandb.ai/ref/python/init/
    reinit: bool = False


class Dataset(BaseModel):
    """
    Simple shapes dataset infos
    """

    # Path to the dataset obtainable on https://github.com/ruflab/simple-shapes-dataset
    path: Path
    # Max number of unpaired examples used during training.
    # Prefer changing `domain_proportions`. The proportion is relative to this value.
    max_train_size: int | None = 500_000


class VisualModule(BaseModel):
    """
    Config for the visual domain module
    """

    num_channels: int = 3  # num channels of image to configure the VAE
    ae_dim: int = 256  # AE hidden dim
    latent_dim: int = 8  # latent dim of the VAE
    beta: float = 0.1  # beta for betaVAE

    # Whether the model is color blind.
    # It adds a transform on the dataset that averages all color channels.
    # NOTE: this only works when using the "v" domain and not "v_latents" where
    # visual latent representations are already extracted. In that case, use a different
    # presaved-path in `domain_args`.
    color_blind: bool = False


class AttributeModule(BaseModel):
    """
    Config for the attribute domain module
    """

    latent_dim: int = 10  # latent dim of the VAE
    hidden_dim: int = 64  # hidden dim of the AE (encoders and decoders)
    beta: float = 0.05  # for betaVAE
    coef_categories: float = 1
    coef_attributes: float = 1

    # Whether to remove rotation information from the attribute.
    nullify_rotation: bool = False


class TextModule(BaseModel):
    """
    Config for the text domain module
    """

    # which file to load for text
    # The file generated with `shapesd create` is "latent"
    # If you use an older version (like if you downloaded directly the dataset) it
    # should be "bert-base-uncased"
    latent_filename: str = "latent"
    # path to the vocab file
    vocab_path: str = str((PROJECT_DIR / "tokenizer/vocab.json").resolve())
    # path to the merge path
    merges_path: str = str((PROJECT_DIR / "tokenizer/merges.txt").resolve())
    # max sequence length of text sequence
    seq_length: int = 64
    vocab_size: int = 822

    # VAE configuration
    latent_dim: int = 64
    hidden_dim: int = 256
    beta: float = 0.1


class DomainModules(BaseModel):
    visual: VisualModule = VisualModule()
    attribute: AttributeModule = AttributeModule()
    text: TextModule = Field(default=TextModule)


class EncodersConfig(BaseModel):
    """
    Encoder architecture config
    """

    hidden_dim: int | Mapping[DomainModelVariantType, int] = 32

    # The model will have an extra linear before and after the n_layers
    # Hence the total will be `2 + n_layers`
    n_layers: int | Mapping[DomainModelVariantType, int] = 3


class LoadedDomainConfig(BaseModel):
    """
    Domain params for the active domains of the run
    """

    # path to the pretrained module
    checkpoint_path: Path
    # domain to select
    domain_type: DomainModelVariantType
    # domain module specific arguments
    args: Mapping[str, Any] = {}

    @field_validator("domain_type", mode="before")
    @classmethod
    def validate_domain_type(cls, v: str) -> DomainModelVariantType:
        """
        Use names instead of values to select enums
        """
        assert v in DomainModelVariantType.__members__, (
            f"Domain type `{v}` is not a member "
            f"of {DomainModelVariantType.__members__.keys()}"
        )
        return DomainModelVariantType[v]


class DomainProportion(BaseModel):
    # proportion for some domains
    # should be in [0, 1]
    proportion: float
    # list of domains the proportion is associated to
    # e.g. if domains: ["v", "t"], then it gives the prop of paired v, t data
    domains: Sequence[str]


class GlobalWorkspace(BaseModel):
    # latent dim of the GW
    latent_dim: int = 12
    # whether to use the fusion GW
    use_fusion_model: bool = False
    # softmax temp for the Softmax distruted selection used by the fusion model
    selection_temperature: float = 0.2
    # whether to learn the logit scale of the contrastive loss like in the clip paper
    learn_logit_scale: bool = False
    # whether to use the VSEPP (https://github.com/fartashf/vsepp) contrastive loss
    # with associated params
    vsepp_contrastive_loss: bool = False
    vsepp_margin: float = 0.2
    vsepp_measure: Literal["cosine", "order"] = "cosine"
    vsepp_max_violation: bool = True
    # whether to use linear encoders and decoders for the GW
    linear_domains: bool = False
    # whether to use bias when using linear encoders and decoders
    linear_domains_use_bias: bool = False
    # encoder architecture config
    encoders: EncodersConfig = EncodersConfig()
    # decoder architecture config
    decoders: EncodersConfig = EncodersConfig()
    # coefs of each loss. The total loss is computed using the given values and coefs
    # you can select any available loss generated by the loss functions
    loss_coefficients: Mapping[str, float] = {
        "cycles": 1.0,
        "demi_cycles": 1.0,
        "translations": 1.0,
        "contrastives": 0.01,
        "fused": 1.0,
    }
    # checkpoint of the GW for downstream, visualization tasks, or migrations
    checkpoint: Path | None = None
    # deprecated, use Config.domain_data_args instead
    domain_args: Mapping[str, Mapping[str, Any]] | None = Field(
        default=None, deprecated="Use `config.domain_data_args` instead."
    )
    # deprecated, use Config.domains instead
    domains: Sequence[LoadedDomainConfig] | None = Field(
        default=None, deprecated="Use `config.domains` instead."
    )
    # deprecated, use Config.domain_proportions instead
    domain_proportions: Sequence[DomainProportion] | None = Field(
        default=None, deprecated="Use `config.domain_proportions` instead."
    )


class ShimmerConfigInfo(BaseModel):
    """
    Some shimmer related config
    """

    # version of shimmer used
    version: str = shimmer_version
    # whether started in debug mode
    debug: bool = False
    # params that were passed through CLI
    cli: Any = {}


class Config(ParsedModel):
    seed: int = 0  # training seed
    ood_seed: int | None = None  # Out of distribution seed
    default_root_dir: Path  # Path where to save and load logs and checkpoints
    # Dataset information
    dataset: Dataset = Field(default=Dataset)
    # Training config
    training: Training = Training()
    # Wandb configuration
    wandb: WanDB = WanDB()
    # Logging configuration
    logging: Logging = Logging()
    # Add a title to your wandb run
    title: str | None = Field(None, alias="t")
    # Add a description to your run
    desc: str | None = Field(None, alias="d")
    # proportion of each domain in the dataset relative to `dataset.max_train_size`
    domain_proportions: Sequence[DomainProportion] = []
    # Config of the different domain modules
    domain_modules: DomainModules = DomainModules()
    # Domain params for the active domains of the run
    domains: Sequence[LoadedDomainConfig] = []
    # data related args used by the dataloader
    domain_data_args: Mapping[str, Mapping[str, Any]] = {}
    # Global workspace configuration
    global_workspace: GlobalWorkspace = GlobalWorkspace()
    # Config used during visualization
    visualization: Visualization | None = None
    # Slurm config when startig on a cluster
    slurm: Slurm | None = None
    __shimmer__: ShimmerConfigInfo = ShimmerConfigInfo()


def use_deprecated_vals(config: Config) -> Config:
    # use deprecated values
    if config.global_workspace.domain_args is not None:
        config.domain_data_args = config.global_workspace.domain_args
        warnings.warn(
            "Deprecated `config.global_workspace.domain_args`, "
            "use `config.domain_data_args` instead",
            DeprecationWarning,
            stacklevel=2,
        )
    if config.global_workspace.domains is not None:
        config.domains = config.global_workspace.domains
        warnings.warn(
            "Deprecated `config.global_workspace.domains`, "
            "use `config.domains` instead",
            DeprecationWarning,
            stacklevel=2,
        )
    if config.global_workspace.domain_proportions is not None:
        config.domain_proportions = config.global_workspace.domain_proportions
        warnings.warn(
            "Deprecated `config.global_workspace.domain_proportions`, "
            "use `config.domain_proportions` instead",
            DeprecationWarning,
            stacklevel=2,
        )
    return config


def load_config(
    path: str | Path,
    load_files: list[str] | None = None,
    use_cli: bool = True,
    debug_mode: bool = False,
    argv: list[str] | None = None,
) -> Config:
    path = Path(path)
    conf_files = []
    if load_files is not None:
        conf_files.extend(load_files)
    if (path / "local.yaml").exists():
        conf_files.append("local.yaml")

    if debug_mode and (path / "debug.yaml").exists():
        conf_files.append("debug.yaml")

    config_dict, cli_config = load_config_files(path, conf_files, use_cli, argv)

    config_dict.update(
        {
            "__shimmer__": {
                "version": __version__,
                "debug": debug_mode,
                "cli": cli_config,
            }
        }
    )

    def make_missing_dict(loc: list[str | int], val: Any) -> Any:
        if not len(loc):
            return val
        if isinstance(loc[0], str):
            return {loc[0]: make_missing_dict(loc[1:], val)}
        elif loc[0] == 0:
            return [make_missing_dict(loc[1:], val)]

    yaml = YAML()
    for _ in range(2):
        try:
            return use_deprecated_vals(Config.model_validate(config_dict))
        except ValidationError as e:
            printed_header = False
            other_errors: list[InitErrorDetails] = []
            ask_should_save = False
            set_config_dynamically = False
            new_vals: list[Any] = []
            for error in e.errors():
                if error["type"] == "missing":
                    if not printed_header:
                        set_config = input(
                            "Your config is missing some values. Do you want to "
                            "set them dynamically? [Y/n]"
                        )
                        set_config_dynamically = set_config.lower() in ["y", "yes", ""]
                        if not set_config_dynamically:
                            raise e

                        printed_header = True

                    error_name = ".".join(map(str, error["loc"]))
                    rprint(f"[blue]{error_name}[/blue]: ", end="")
                    new_val = input()
                    new_conf = make_missing_dict(list(error["loc"]), new_val)
                    new_vals.append(new_conf)
                    merge_dicts(config_dict, new_conf)
                    ask_should_save = True
                else:
                    init_error = InitErrorDetails(
                        type=error["type"],
                        loc=error["loc"],
                        input=error["input"],
                    )
                    if "ctx" in error:
                        init_error["ctx"] = error["ctx"]
                    other_errors.append(init_error)

            if ask_should_save:
                rprint(
                    "Do you want to save the values in "
                    "`[green]config/local.yaml[/green]`? [Y/n]",
                    end="",
                )
                do_save = input()
                if do_save.lower() in ["yes", "y", ""]:
                    local_file = {}
                    if (path / "local.yaml").exists():
                        with open(path / "local.yaml") as f:
                            local_file = yaml.load(f)
                    for new_val in new_vals:
                        merge_dicts(local_file, new_val)
                    with open(path / "local.yaml", "w") as f:
                        yaml.dump(local_file, f)

            if len(other_errors):
                raise ValidationError.from_exception_data(e.title, other_errors) from e
    raise ValueError("Could not parse config")
