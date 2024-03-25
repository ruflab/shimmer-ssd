from collections.abc import Callable
from typing import Any


import torch
import torch.nn as nn
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
)

from migrate_ckpt.migrate import get_folder_migrations


from lightning.pytorch.loggers.wandb import WandbLogger
from shimmer.modules.global_workspace import (
    GlobalWorkspaceFusion,
    SchedulerArgs,
)
from torch import set_float32_matmul_precision

from simple_shapes_dataset import DEBUG_MODE, PROJECT_DIR
from simple_shapes_dataset.ckpt_migrations import (
    SaveMigrations,
)
from simple_shapes_dataset.config import load_config
from simple_shapes_dataset.dataset import SimpleShapesDataModule
from simple_shapes_dataset.modules.domains import load_pretrained_domains

device = torch.device("cuda:0")

def to_device(batch, device):
    """Recursively move the batch to the specified device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [to_device(v, device) for v in batch]
    else:
        return batch

config = load_config(
    "../config",
    load_files=["train_gw.yaml"],
    debug_mode=DEBUG_MODE,
)

domain_proportion = {
    frozenset(item.domains): item.proportion
    for item in config.global_workspace.domain_proportions
}

#don't know what this does
additional_transforms: dict[str, list[Callable[[Any], Any]]] = {}

data_module = SimpleShapesDataModule(
    config.dataset.path,
    domain_proportion,
    batch_size=config.training.batch_size,
    num_workers=config.training.num_workers,
    seed=config.seed,
    ood_seed=config.ood_seed,
    domain_args=config.global_workspace.domain_args,
    additional_transforms=additional_transforms,
)

print(config)

domain_modules, gw_encoders, gw_decoders = load_pretrained_domains(
        config.default_root_dir,
        config.global_workspace.domains,
        config.global_workspace.latent_dim,
        config.global_workspace.encoders.hidden_dim,
        config.global_workspace.encoders.n_layers,
        config.global_workspace.decoders.hidden_dim,
        config.global_workspace.decoders.n_layers,
        has_uncertainty=config.global_workspace.has_uncertainty,
        is_linear=config.global_workspace.linear_domains,
        bias=config.global_workspace.linear_domains_use_bias,
    )

if config.global_workspace.use_fusion_model:
        module = GlobalWorkspaceFusion(
            domain_modules,
            gw_encoders,
            gw_decoders,
            config.global_workspace.latent_dim,
            config.training.optim.lr,
            config.training.optim.weight_decay,
            scheduler_args=SchedulerArgs(
                max_lr=config.training.optim.max_lr,
                total_steps=config.training.max_steps,
            ),
            learn_logit_scale=config.global_workspace.learn_logit_scale,
        )
else:
    raise "You need a globalworkspacefusion to train an attention module ! check your configs for the use_fusion_module argument."

########################################################################   loading the pretrained fusion translators

        # Path to your checkpoint file

#TODO : make this happen in config (I couldn't figure it out yet)
#checkpoint_path = config.fusion_checkpoint_path
checkpoint_path = "/home/rbertin/cleaned/git_synced/simple-shapes-dataset/simple_shapes_fusion-2hfqxccq/epoch=403.ckpt"
        # Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda:0'))#device ?

        # Assuming module is your DeterministicGlobalWorkspace model
        # Load the state dict from the checkpoint into your model
module.load_state_dict(checkpoint['state_dict'])
module = module.to(torch.device("cuda:0"))#I don't remember how we're supposed to deal with the device in shimmer


########################################################################   classification head



class ClassificationHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Increasing the number of layers for more complexity
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        return self.fc5(x)

########################################################################   training loop

# Define the path to the checkpoint

#TODO : make this happen in config (I couldn't figure it out yet)
#checkpoint_path = config.classification.model_path
checkpoint_path = "attention_model.pth"
state_dict = torch.load(checkpoint_path, map_location=device)

# Initialize the ClassificationHead model
classification_head = ClassificationHead(input_dim=config.global_workspace.latent_dim, output_dim=3).to(device)
classification_head.load_state_dict(state_dict)
classification_head.eval()



callbacks = list[Callback]

wandb_logger = None
if config.wandb.enabled:
    gw_type = "gw_uncertainty" if config.global_workspace.has_uncertainty else "gw"
    run_name = f"{gw_type}_z={config.global_workspace.latent_dim}"
    wandb_logger = WandbLogger(
        save_dir=config.wandb.save_dir,
        project=config.wandb.project,
        entity=config.wandb.entity,
        tags=["train_gw"],
        name=run_name,
    )
    wandb_logger.experiment.config.update(config.model_dump())

    checkpoint_dir = (
        config.default_root_dir / f"{wandb_logger.name}-{wandb_logger.version}"
    )
    callbacks.extend(
        [
            SaveMigrations(
                get_folder_migrations(PROJECT_DIR / "migrations" / "gw")
            ),
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="{epoch}",
                monitor="val/loss",
                mode="min",
                save_top_k=1,
            ),
        ]
    )

set_float32_matmul_precision(config.training.float32_matmul_precision)

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

trainer.fit(module, data_module)
trainer.validate(module, data_module, "best")
trainer.test(module, data_module, "best")
