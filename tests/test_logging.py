from lightning.pytorch.loggers.wandb import WandbLogger
from shimmer import DomainModule, GWDecoder, GWEncoder
from shimmer.modules.global_workspace import GlobalWorkspace
from utils import PROJECT_DIR

from simple_shapes_dataset.dataset.data_module import SimpleShapesDataModule
from simple_shapes_dataset.logging import LogGWImagesCallback, attribute_image_grid
from simple_shapes_dataset.modules.domains.attribute import AttributeDomainModule
from simple_shapes_dataset.modules.domains.visual import VisualDomainModule


def test_attribute_figure_grid():
    data_module = SimpleShapesDataModule(
        PROJECT_DIR / "sample_dataset",
        domain_proportions={
            frozenset(["attr"]): 1.0,
        },
        batch_size=32,
        num_workers=0,
        seed=0,
    )
    image_size = 32
    padding = 2
    val_samples = data_module.get_samples("train", 4)[frozenset(["attr"])]["attr"]
    images = attribute_image_grid(val_samples, image_size=image_size, ncols=2)
    assert images.height == 2 * image_size + 3 * padding
    assert images.width == 2 * image_size + 3 * padding


def test_gw_logger():
    data_module = SimpleShapesDataModule(
        PROJECT_DIR / "sample_dataset",
        domain_proportions={
            frozenset(["v", "attr"]): 0.5,
            frozenset(["v"]): 1.0,
            frozenset(["attr"]): 1.0,
        },
        batch_size=32,
        num_workers=0,
        seed=0,
    )

    workspace_dim = 4

    domains: dict[str, DomainModule]
    domains = {
        "v": VisualDomainModule(3, 4, 16),
        "attr": AttributeDomainModule(4, 16),
    }

    gw_encoders = {
        "v": GWEncoder(
            in_dim=domains["v"].latent_dim,
            hidden_dim=16,
            out_dim=workspace_dim,
            n_layers=2,
        ),
        "attr": GWEncoder(
            in_dim=domains["attr"].latent_dim,
            hidden_dim=16,
            out_dim=workspace_dim,
            n_layers=2,
        ),
    }

    gw_decoders = {
        "v": GWDecoder(
            in_dim=workspace_dim,
            hidden_dim=16,
            out_dim=domains["v"].latent_dim,
            n_layers=2,
        ),
        "attr": GWDecoder(
            in_dim=workspace_dim,
            hidden_dim=16,
            out_dim=domains["attr"].latent_dim,
            n_layers=2,
        ),
    }

    module = GlobalWorkspace(
        domains, gw_encoders, gw_decoders, workspace_dim, loss_coefs={}
    )
    wandb_logger = WandbLogger(mode="disabled")

    val_samples = data_module.get_samples("val", 2)
    callback = LogGWImagesCallback(
        val_samples, log_key="test", mode="test", every_n_epochs=1
    )

    callback.on_callback(current_epoch=0, loggers=[wandb_logger], pl_module=module)
