from lightning.pytorch.loggers.wandb import WandbLogger
from shimmer.modules.domain import DomainDescription
from shimmer.modules.global_workspace import GlobalWorkspace
from utils import PROJECT_DIR

from simple_shapes_dataset.dataset.data_module import SimpleShapesDataModule
from simple_shapes_dataset.logging import (
    LogGWImagesCallback,
    make_attribute_grid,
)
from simple_shapes_dataset.modules.domains.attribute import (
    AttributeDomainModule,
)
from simple_shapes_dataset.modules.domains.visual import VisualDomainModule
from simple_shapes_dataset.modules.global_workspace import (
    _GlobalWorkspaceLightningModule,
)


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
    val_samples = data_module.get_samples("train", 4)[frozenset(["attr"])][
        "attr"
    ]
    images = make_attribute_grid(
        val_samples, image_size=image_size, ncols=2, padding=padding
    )
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

    domains = {
        "v": DomainDescription(
            module=VisualDomainModule(3, 4, 16), latent_dim=4
        ),
        "attr": DomainDescription(
            module=AttributeDomainModule(4, 16), latent_dim=4
        ),
    }
    global_workspace = GlobalWorkspace(
        {"v", "attr"},
        4,
        {"v": 4, "attr": 4},
        {"v": 16, "attr": 16},
        {"v": 2, "attr": 2},
        {"v": 16, "attr": 16},
        {"v": 2, "attr": 2},
    )

    module = _GlobalWorkspaceLightningModule(
        global_workspace,
        {name: domain.module for name, domain in domains.items()},
        1,
        1,
        1,
        1,
        1e-3,
        1e-6,
    )
    wandb_logger = WandbLogger(mode="disabled")

    val_samples = data_module.get_samples("val", 2)
    callback = LogGWImagesCallback(
        val_samples, log_key="test", every_n_epochs=1
    )

    callback.on_callback(
        current_epoch=0, loggers=[wandb_logger], pl_module=module
    )
