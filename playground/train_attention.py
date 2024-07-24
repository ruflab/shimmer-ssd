import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers.wandb import WandbLogger
from shimmer import (
    BroadcastLossCoefs,
    ContrastiveLossType,
    GlobalWorkspaceBase,
    RawDomainGroupT,
    SaveMigrations,
    migrate_model,
)
from shimmer.modules.attention_module import AttentionBase
from shimmer.modules.global_workspace import (
    GlobalWorkspace,
    SchedulerArgs,
)
from shimmer.modules.selection import RandomSelection
from train_classifier import ShapeClassifier

from simple_shapes_dataset.config import load_config

# from simple_shapes_dataset.config import load_config
from simple_shapes_dataset.dataset import SimpleShapesDataModule
from simple_shapes_dataset.logging import LogGWImagesCallback
from simple_shapes_dataset.modules.contrastive_loss import VSEPPContrastiveLoss
from simple_shapes_dataset.modules.domains import load_pretrained_domains


class AttentionCriterion(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        # Load the classification head
        checkpoint_path = "/home/lscipio/SimpleShapes/simple-shapes-dataset/checkpoints/shapeclassifier/shape_checkpoint.pth"
        state_dict = torch.load(checkpoint_path)
        shapes_classifier = ShapeClassifier(input_dim=latent_dim, output_dim=3)
        shapes_classifier.load_state_dict(state_dict)
        shapes_classifier.eval()
        self.shapes_classifier = shapes_classifier.requires_grad_(False)
        self.criterion = nn.CrossEntropyLoss()

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

    def compute_accuracy(
        self, x: torch.Tensor, shape_info: torch.Tensor
    ) -> torch.Tensor:
        true_classes = torch.argmax(shape_info, dim=1)
        predictions = torch.argmax(x, dim=1)
        accuracy = (predictions == true_classes).float().mean().item() * 100
        accuracy_tensor = torch.tensor(accuracy)
        return accuracy_tensor

    def forward(self, x: torch.Tensor, y: RawDomainGroupT):
        selected_dims_shape = x.size()
        if self.get_shape_info(y) is None:
            shape_info = torch.zeros(selected_dims_shape[0], 3).to(x.device)
        else:
            shape_info = self.get_shape_info(y)

        loss = self.criterion(self.shapes_classifier(x), shape_info)
        accuracy = self.compute_accuracy(self.shapes_classifier(x), shape_info)
        return loss, accuracy


def main():
    path = "/home/lscipio/SimpleShapes/simple-shapes-dataset/config"
    config = load_config(path=path, load_files=["local.yaml", "train_qa.yaml"])
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

    path_to_checkpoint = "/home/lscipio/SimpleShapes/simple-shapes-dataset/checkpoints/epoch=411-step=100000-0.ckpt"
    migrate_model(path_to_checkpoint)
    checkpoint = torch.load(path_to_checkpoint, map_location=torch.device("cuda:0"))
    gw_module.load_state_dict(checkpoint["state_dict"])
    gw_module.eval()
    gw_module = gw_module.requires_grad_(False)

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
    # config.wandb.enabled = False
    if config.wandb.enabled:
        wandb_logger = WandbLogger(
            save_dir=config.wandb.save_dir,
            project=config.wandb.project,
            entity=config.wandb.entity,
            tags=[
                "corruption_scaling",
                "fixed_corruption_vector",
                "corrupt_one_sides",
            ],
            name="f_2s_random_attention",
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

    # Initialize attention mechanism
    domain_dim = 12
    head_size = 5
    domain_names = ["v_latents", "attr"]
    shape_classifier = AttentionCriterion(domain_dim)
    shape_classifier.eval()
    shape_classifier.requires_grad_(False)
    scheduler_args = SchedulerArgs(
        max_lr=config.training.optim.max_lr,
        total_steps=config.training.max_steps,
    )
    # The attention mechanism will be trained with this attention mechanism
    corruption_scaling = np.linspace(0, 1, 100).tolist()

    fixed_corruption_vector = torch.load(
        "/home/lscipio/SimpleShapes/simple-shapes-dataset/playground/shapeclassifier/fixed_corruption_vector.pt"
    )

    # attention_mechanism = DynamicQueryAttention(
    #     head_size, domain_dim, domain_names, n_steps=3
    # )
    temperature = 0.1
    attention_mechanism = RandomSelection(temperature)
    attention_mechanism.eval()
    attention_module = AttentionBase(
        gw_module,
        attention_mechanism,
        domain_names,
        shape_classifier,
        domain_dim,
        fixed_corruption_vector=fixed_corruption_vector,
        corruption_scaling=corruption_scaling,
        corrupt_sides=True,
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

    trainer.fit(attention_module, data_module)
    trainer.validate(attention_module, data_module, "best")
    trainer.test(attention_module, data_module)


if __name__ == "__main__":
    main()
