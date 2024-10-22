import lightning.pytorch as pl
from simple_shapes_dataset import SimpleShapesDataModule, get_default_domains
from utils import PROJECT_DIR

from shimmer_ssd.modules.domains.attribute import AttributeDomainModule
from shimmer_ssd.modules.domains.visual import VisualDomainModule


def test_attr_train():
    data_module = SimpleShapesDataModule(
        PROJECT_DIR / "sample_dataset",
        get_default_domains(["attr"]),
        domain_proportions={
            frozenset(["attr"]): 1.0,
        },
        batch_size=2,
        num_workers=0,
        seed=0,
    )

    attr_domain_module = AttributeDomainModule(4, 16)

    trainer = pl.Trainer(
        fast_dev_run=True,
        enable_progress_bar=False,
        accelerator="cpu",
    )

    trainer.fit(attr_domain_module, data_module)


def test_v_train():
    data_module = SimpleShapesDataModule(
        PROJECT_DIR / "sample_dataset",
        get_default_domains(["v"]),
        domain_proportions={
            frozenset(["v"]): 1.0,
        },
        batch_size=2,
        num_workers=0,
        seed=0,
    )

    attr_domain_module = VisualDomainModule(3, 4, 16)

    trainer = pl.Trainer(
        fast_dev_run=True,
        enable_progress_bar=False,
        accelerator="cpu",
    )

    trainer.fit(attr_domain_module, data_module)
