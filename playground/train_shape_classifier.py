import random
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from lightning.pytorch import Callback, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from shimmer import (
    BroadcastLossCoefs,
    ContrastiveLossType,
    GlobalWorkspaceBase,
    RawDomainGroupT,
    SaveMigrations,
    migrate_model,
)
from shimmer.modules.global_workspace import (
    GlobalWorkspace,
    GlobalWorkspaceBase,
    SchedulerArgs,
)
from shimmer.modules.selection import RandomSelection
from shimmer.types import (
    LatentsDomainGroupsDT,
    LatentsDomainGroupsT,
    RawDomainGroupsT,
    RawDomainGroupT,
)
from shimmer.utils import group_device, groups_batch_size
from torch import Tensor
from torch.optim.lr_scheduler import OneCycleLR

from simple_shapes_dataset.config import load_config

# from simple_shapes_dataset.config import load_config
from simple_shapes_dataset.dataset import SimpleShapesDataModule
from simple_shapes_dataset.logging import LogGWImagesCallback
from simple_shapes_dataset.modules.contrastive_loss import VSEPPContrastiveLoss
from simple_shapes_dataset.modules.domains import load_pretrained_domains


class ShapeClassifier(nn.Sequential):
    def __init__(self, input_dim, output_dim):
        layers = [
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, output_dim),
        ]
        super().__init__(*layers)


class ShapeBase(LightningModule):
    def __init__(
        self,
        module,
        shape_classifier,
        corruption_scale: float,
        domain_names: frozenset[str] | None = None,
        optim_lr: float = 1e-3,
        optim_weight_decay: float = 0.0,
        scheduler_args: SchedulerArgs | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["module", "shape_classifier"])
        self.gw = module
        self.shape_classifier = shape_classifier
        self.corruption_scale = corruption_scale
        self.domain_names = frozenset(["v_latents", "attr"])
        self.criterion = nn.CrossEntropyLoss()
        self.selection_module = RandomSelection(0.2)
        self.optim_lr = optim_lr
        self.optim_weight_decay = optim_weight_decay
        self.scheduler_args = SchedulerArgs(max_lr=optim_lr, total_steps=1)
        if scheduler_args is not None:
            self.scheduler_args.update(scheduler_args)

    def forward(self, x):
        shape_logits = self.shape_classifier(x)
        return shape_logits

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """
        Configure models optimizers.

        Here we use `AdamW` for the optimizer and `OneCycleLR` for the learning-rate
        scheduler.
        """

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optim_lr,
            weight_decay=self.optim_weight_decay,
        )

        lr_scheduler = OneCycleLR(optimizer, **self.scheduler_args)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

    def get_shape_info(self, batch: RawDomainGroupT):
        """
        Get the shape info from the batch.

        Args:
            batch: The batch of data.
        """
        if "attr" in batch:
            attr_data = batch["attr"]
            shape_info = attr_data[0]  # First tensor contains shape information
            shape_info = shape_info.float()
            return shape_info

    def apply_batch_corruption(
        self,
        batch: LatentsDomainGroupsT,
        amount_corruption: float | None = None,
        corruption_vector: torch.Tensor | None = None,
    ) -> LatentsDomainGroupsDT:
        """
        Apply corruption to the batch.

        Args:
            batch: A batch of latent domains.
            corruption_vector: A vector to be added to the corrupted domain.
            corrupted_domain: The domain to be corrupted.

        Returns:
            A batch where one of the latent domains is corrupted.
        """
        matched_data_dict: LatentsDomainGroupsDT = {}
        for domain_names, domains in batch.items():
            # Randomly select a domain to be corrupted for this instance
            corrupted_domain = random.choice(list(self.domain_names))
            for domain_name, domain in domains.items():
                if domain_names != self.domain_names or domain_name != corrupted_domain:
                    matched_data_dict.setdefault(domain_names, {})[domain_name] = domain
                    continue

                # If corruption vector is not fixed outside the loop
                if corruption_vector is None:
                    corruption_vector = torch.randn_like(domain)
                # Normalize the corruption vector
                corruption_vector = (
                    corruption_vector - corruption_vector.mean()
                ) / corruption_vector.std()
                # Scale the corruption vector based on the amount of corruption
                scaled_corruption_vector = (corruption_vector * 5) * amount_corruption

                # Apply element-wise addition to one of the domains
                matched_data_dict.setdefault(domain_names, {})[domain_name] = (
                    domain + scaled_corruption_vector
                )

        return matched_data_dict

    def apply_row_corruption(
        self,
        batch: LatentsDomainGroupsT,
        amount_corruption: float | None = None,
    ) -> LatentsDomainGroupsDT:
        """
        Apply corruption to the batch.

        Args:
            batch: A batch of latent domains.
            corruption_vector: A vector to be added to the corrupted domain.
            corrupted_domain: The domain to be corrupted.

        Returns:
            A batch where one of the latent domains is corrupted.
        """
        matched_data_dict: LatentsDomainGroupsDT = {}
        # Make a copy of the batch
        for domain_names, domains in batch.items():
            for domain_name, domain in domains.items():
                matched_data_dict.setdefault(domain_names, {})[domain_name] = domain
                continue
        device = group_device(domains)
        batch_size = groups_batch_size(batch)
        n_domains = len(self.domain_names)
        domain_size = 12
        selected_domains = torch.randint(0, n_domains, (batch_size,), device=device)
        masked_domains = torch.nn.functional.one_hot(selected_domains, n_domains).to(
            device, torch.bool
        )
        # Inverse
        corruption_vector = torch.load(
            "/home/lscipio/SimpleShapes/simple-shapes-dataset/playground/shapeclassifier/fixed_corruption_vector.pt"
        )
        corruption_vector = torch.tensor(
            [
                0.2146,
                -0.1659,
                -1.3089,
                0.6493,
                0.5529,
                -0.8427,
                -0.8761,
                -0.1472,
                -0.3247,
                0.3215,
                0.0212,
                -0.7116,
            ]
        ).to(device)
        # Expand
        corruption_vector = corruption_vector.expand(batch_size, domain_size)
        # corruption_vector = torch.randn((batch_size, domain_size), device=device)
        # corruption_vector = torch.randn((batch_size, domain_size), device=device)
        # Normalize the corruption vector
        corruption_vector = (
            corruption_vector - corruption_vector.mean(dim=1, keepdim=True)
        ) / corruption_vector.std(dim=1, keepdim=True)
        # Scale the corruption vector based on the amount of corruption
        scaled_corruption_vector = (corruption_vector * 5) * amount_corruption

        for k, (domain_names, domains) in enumerate(matched_data_dict.items()):
            if domain_names == self.domain_names:
                for domain_name, domain in domains.items():
                    if domain_name == "v_latents":
                        domain[masked_domains[:, 0]] += scaled_corruption_vector[
                            masked_domains[:, 0]
                        ]
                    if domain_name == "attr":
                        domain[~masked_domains[:, 0]] += scaled_corruption_vector[
                            ~masked_domains[:, 0]
                        ]

        return matched_data_dict

    def compute_accuracy(
        self, x: torch.Tensor, shape_info: torch.Tensor
    ) -> torch.Tensor:
        true_classes = torch.argmax(shape_info, dim=1)
        predictions = torch.argmax(x, dim=1)
        accuracy = (predictions == true_classes).float().mean().item() * 100
        accuracy_tensor = torch.tensor(accuracy)
        return accuracy_tensor

    def generic_step(self, batch: RawDomainGroupsT, mode: str) -> Tensor:
        latent_domains = self.gw.encode_domains(batch)
        # 0.0 means no scaling
        if self.corruption_scale != 0.0:
            # Apply corruption to the whole batch
            latent_domains = self.apply_row_corruption(
                latent_domains, self.corruption_scale
            )
        gw_representation = self.gw.encode_and_fuse(
            latent_domains, self.selection_module
        )
        losses = []
        accuracies = []

        for domain_names, domains in gw_representation.items():
            shape_logits = self.forward(domains)
            if self.get_shape_info(batch[domain_names]) is None:
                continue
            else:
                shape_info = self.get_shape_info(batch[domain_names])
            loss = self.criterion(shape_logits, shape_info)
            losses.append(self.criterion(shape_logits, shape_info))
            accuracies.append(self.compute_accuracy(shape_logits, shape_info))
            domain_names_str = ",".join(domain_names)
            self.log(
                f"{mode}/{domain_names_str}_loss",
                losses[-1],
                batch_size=domains.size(0),
            )
            self.log(
                f"{mode}/{domain_names_str}_accuracy",
                accuracies[-1],
                batch_size=domains.size(0),
            )
        loss = torch.stack(losses).mean()
        accuracy = torch.stack(accuracies).mean()
        self.log(f"{mode}/loss", loss, on_step=True, on_epoch=True)
        self.log(f"{mode}/accuracy", accuracy, on_step=True, on_epoch=True)

        return loss

    def training_step(
        self, batch: RawDomainGroupsT, batch_idx: int
    ) -> Tensor | Mapping[str, Any] | None:  # type: ignore
        return self.generic_step(batch, "train")

    def validation_step(  # type: ignore
        self, data: RawDomainGroupT, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Validation step used by lightning"""

        batch = {frozenset(data.keys()): data}
        for domain in data:
            batch[frozenset([domain])] = {domain: data[domain]}
        if dataloader_idx == 0:
            return self.generic_step(batch, mode="val")
        return self.generic_step(batch, mode="val/ood")

    def test_step(  # type: ignore
        self, data: RawDomainGroupT, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Test step used by lightning"""

        batch = {frozenset(data.keys()): data}
        for domain in data:
            batch[frozenset([domain])] = {domain: data[domain]}
        if dataloader_idx == 0:
            return self.generic_step(batch, mode="test")
        return self.generic_step(batch, mode="test/ood")


def main():
    path = "/home/lscipio/SimpleShapes/simple-shapes-dataset/config"
    config = load_config(path=path, load_files=["train_classifier.yaml"])

    seed_everything(config.seed, workers=True)

    domain_proportion = {
        frozenset(item.domains): item.proportion
        for item in config.global_workspace.domain_proportions
    }

    data_module = SimpleShapesDataModule(
        config.dataset.path,
        domain_proportion,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        seed=config.seed,
        ood_seed=config.ood_seed,
        domain_args=config.global_workspace.domain_args,
    )

    domain_modules, gw_encoders, gw_decoders = load_pretrained_domains(
        config.default_root_dir,
        config.global_workspace.domains,
        config.global_workspace.latent_dim,
        config.global_workspace.encoders.hidden_dim,
        config.global_workspace.encoders.n_layers,
        config.global_workspace.decoders.hidden_dim,
        config.global_workspace.decoders.n_layers,
        is_linear=config.global_workspace.linear_domains,
        bias=config.global_workspace.linear_domains_use_bias,
    )
    domain_modules = {
        domain: module.eval().requires_grad_(False)
        for domain, module in domain_modules.items()
    }
    gw_encoders = {
        domain: module.eval().requires_grad_(False)
        for domain, module in gw_encoders.items()
    }
    gw_decoders = {
        domain: module.eval().requires_grad_(False)
        for domain, module in gw_decoders.items()
    }

    contrastive_fn: ContrastiveLossType | None = None
    if config.global_workspace.vsepp_contrastive_loss:
        contrastive_fn = VSEPPContrastiveLoss(
            config.global_workspace.vsepp_margin,
            config.global_workspace.vsepp_measure,
            config.global_workspace.vsepp_max_violation,
            torch.tensor([1 / 0.07]).log(),
        )
    loss_coefs_fusion: BroadcastLossCoefs = {
        "contrastives": config.global_workspace.loss_coefficients.contrastives,
        "broadcast": config.global_workspace.loss_coefficients.broadcast,
    }

    # Initiliaze global workspace
    gw_module: GlobalWorkspaceBase
    loss_coefs_fusion: BroadcastLossCoefs = {
        "contrastives": config.global_workspace.loss_coefficients.contrastives,
        "broadcast": config.global_workspace.loss_coefficients.broadcast,
    }
    gw_module = GlobalWorkspace(
        domain_modules,
        gw_encoders,
        gw_decoders,
        config.global_workspace.latent_dim,
        loss_coefs_fusion,
        config.training.optim.lr,
        config.training.optim.weight_decay,
        scheduler_args=SchedulerArgs(
            max_lr=config.training.optim.max_lr,
            total_steps=config.training.max_steps,
        ),
        learn_logit_scale=config.global_workspace.learn_logit_scale,
        contrastive_loss=contrastive_fn,
    )

    train_samples = data_module.get_samples("train", 32)
    val_samples = data_module.get_samples("val", 32)
    test_samples = data_module.get_samples("test", 32)

    for domains in val_samples:
        for domain in domains:
            val_samples[frozenset([domain])] = {domain: val_samples[domains][domain]}
            test_samples[frozenset([domain])] = {domain: test_samples[domains][domain]}
        break

    callbacks: list[Callback] = [
        LearningRateMonitor(logging_interval="step"),
        LogGWImagesCallback(
            val_samples,
            log_key="images/val",
            mode="val",
            every_n_epochs=config.logging.log_val_medias_every_n_epochs,
        ),
        LogGWImagesCallback(
            val_samples,
            log_key="images/test",
            mode="test",
            every_n_epochs=None,
        ),
        LogGWImagesCallback(
            train_samples,
            log_key="images/train",
            mode="train",
            every_n_epochs=config.logging.log_train_medias_every_n_epochs,
        ),
    ]

    if config.training.enable_progress_bar:
        callbacks.append(RichProgressBar())

    wandb_logger = None
    if config.wandb.enabled:
        wandb_logger = WandbLogger(
            save_dir=config.wandb.save_dir,
            project=config.wandb.project,
            entity=config.wandb.entity,
            tags=[
                "lr_0.001",
                "shape_classifier",
                "no_corruption",
                "2000_steps",
                "corruption_instance",
            ],
            name="train_shape_classifier",
        )
        wandb_logger.experiment.config.update(config.model_dump())
        checkpoint_dir = (
            config.default_root_dir / f"{wandb_logger.name}-{wandb_logger.version}"
        )
        callbacks.extend(
            [
                SaveMigrations(),
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename="{epoch}",
                    mode="min",
                    monitor="val/loss",
                    save_top_k=1,
                ),
            ]
        )

    torch.set_float32_matmul_precision(config.training.float32_matmul_precision)

    ### Load pretrained gw model
    path_to_checkpoint = "/home/lscipio/SimpleShapes/simple-shapes-dataset/checkpoints/epoch=411-step=100000-0.ckpt"
    migrate_model(path_to_checkpoint)
    checkpoint = torch.load(path_to_checkpoint, map_location=torch.device("cuda:0"))
    gw_module.load_state_dict(checkpoint["state_dict"])
    gw_module.eval()
    gw_module = gw_module.requires_grad_(False)
    # Freeze the parameters of the Global Workspace
    for param in gw_module.parameters():
        param.requires_grad = False

    ### Initialize shape classifier
    domain_dim = 12
    output_dim = 3
    shape_classifier = ShapeClassifier(domain_dim, output_dim)
    shape_classifier.train()
    scheduler_args = SchedulerArgs(
        max_lr=config.training.optim.max_lr,
        total_steps=config.training.max_steps,
    )

    ### Train without corruption
    corruption_scale = 0.0

    ### Initialize Lighting Module
    model = ShapeBase(
        gw_module,
        shape_classifier,
        corruption_scale,
        scheduler_args=scheduler_args,
    )

    trainer = Trainer(
        logger=wandb_logger,
        fast_dev_run=config.training.fast_dev_run,
        max_steps=config.training.max_steps,
        enable_progress_bar=config.training.enable_progress_bar,
        default_root_dir=config.default_root_dir,
        callbacks=callbacks,
        precision=config.training.precision,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
    )

    trainer.fit(model, data_module)
    trainer.validate(model, data_module, "best")
    trainer.test(model, data_module, "best")

    # path = "/home/lscipio/SimpleShapes/simple-shapes-dataset/checkpoints/shapeclassifier/shape_checkpoint.pth"
    torch.save(shape_classifier.state_dict(), path)


if __name__ == "__main__":
    main()
