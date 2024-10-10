import logging
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch import set_float32_matmul_precision
from lightning.pytorch import Callback, seed_everything
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    RichProgressBar,
)
import random
from tqdm import tqdm
from shimmer import (
    BroadcastLossCoefs,
    ContrastiveLossType,
    GlobalWorkspaceBase,
    LossCoefs,
    SaveMigrations,
)
from shimmer.modules.global_workspace import (
    GlobalWorkspace,
    SchedulerArgs,
)
from shimmer.modules.selection import DynamicQueryAttention
from simple_shapes_dataset import DEBUG_MODE, PROJECT_DIR
from simple_shapes_dataset.config import load_config
from simple_shapes_dataset.dataset import SimpleShapesDataModule
from simple_shapes_dataset.dataset.domain import get_default_domains
from simple_shapes_dataset.dataset.pre_process import (
    color_blind_visual_domain,
    nullify_attribute_rotation,
)
from simple_shapes_dataset.logging import LogGWImagesCallback
from simple_shapes_dataset.modules.contrastive_loss import VSEPPContrastiveLoss
from simple_shapes_dataset.modules.domains import load_pretrained_domains
import copy

def main():
    config = load_config(
        PROJECT_DIR / "config",
        load_files=["train_gw.yaml"],
        debug_mode=DEBUG_MODE,
    )

    # Seed everything
    seed_everything(config.seed, workers=True)

    # Prepare domain proportions and classes
    domain_proportion = {
        frozenset(item.domains): item.proportion
        for item in config.global_workspace.domain_proportions
    }
    domain_classes = get_default_domains(
        {domain.domain_type.kind.value for domain in config.global_workspace.domains}
    )

    # Prepare data transforms
    additional_transforms: dict[str, list[Callable[[Any], Any]]] = {}
    if config.domain_modules.attribute.nullify_rotation:
        logging.info("Nullifying rotation in the attr domain.")
        additional_transforms["attr"] = [nullify_attribute_rotation]
    if config.domain_modules.visual.color_blind:
        logging.info("v domain will be color blind.")
        additional_transforms["v"] = [color_blind_visual_domain]

    # Load data module
    data_module = SimpleShapesDataModule(
        config.dataset.path,
        domain_classes,
        domain_proportion,
        batch_size=config.training.batch_size,
        max_train_size=config.dataset.max_train_size,
        num_workers=config.training.num_workers,
        seed=config.seed,
        ood_seed=config.ood_seed,
        domain_args=config.global_workspace.domain_args,
        additional_transforms=additional_transforms,
    )

    # Load pre-trained domain modules
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

    contrastive_fn: ContrastiveLossType | None = None
    checkpoint_path = "/home/rbertin/attention/simple-shapes-dataset/simple_shapes_fusion-z4typ6dd/epoch=101.ckpt"
    
    loss_coefs_fusion: BroadcastLossCoefs = {
        "contrastives": config.global_workspace.loss_coefficients.contrastives,
        "fused": config.global_workspace.loss_coefficients.fused,
        "translations": config.global_workspace.loss_coefficients.translations,
        "demi_cycles": config.global_workspace.loss_coefficients.demi_cycles,
        "cycles": config.global_workspace.loss_coefficients.cycles,
    }

    # Load the global workspace from checkpoint
    module = GlobalWorkspace.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        domain_mods=domain_modules,
        gw_encoders=gw_encoders,
        gw_decoders=gw_decoders,
        workspace_dim=config.global_workspace.latent_dim,
        loss_coefs=loss_coefs_fusion,
        contrastive_loss=contrastive_fn,
        optim_lr=config.training.optim.lr,
        optim_weight_decay=config.training.optim.weight_decay,
        scheduler_args=SchedulerArgs(
            max_lr=config.training.optim.max_lr,
            total_steps=config.training.max_steps,
        )
    ).cuda()

    module.selection_mod = DynamicQueryAttention(5,12,["attr","v_latents"],3).cuda()

    # Define the key for encoded representations
    input_key = frozenset({'v_latents', 'attr'})

    # Set up dataloaders
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Initialize corruption vector v and random scales
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Transfer batch to device
    def transfer_batch_to_device(batch, device):
        def move_to_device(batch):
            if isinstance(batch, torch.Tensor):
                return batch.to(device)
            elif isinstance(batch, dict):
                return {k: move_to_device(v) for k, v in batch.items()}
            elif isinstance(batch, (list, tuple)):
                return [move_to_device(v) for v in batch]
            else:
                return batch

        return move_to_device(batch)

    # Define MLP Classifier
    class MLPClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(MLPClassifier, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )

        def forward(self, x):
            return self.model(x)

    # Prepare classifier training
    batch = next(iter(train_loader))
    batch = transfer_batch_to_device(batch, device)
    domain_encoded = module.encode_domains(batch[0])
    encoded = module.encode_and_fuse(domain_encoded, module.selection_mod)

    # Extract features and labels
    features = encoded[input_key]
    labels = batch[0][input_key]['attr'][0]
    labels_indices = torch.argmax(labels, dim=1)

    # Initialize classifier
    input_size = features.shape[1]
    hidden_size = 128
    output_size = labels.shape[1]
    model = MLPClassifier(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(torch.load("attention/shapes_clf/classifier_checkpoint.pth"))
    model.eval()  # Set the classifier to evaluation mode
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(module.selection_mod.parameters(), lr=0.001)

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        # === Training Phase ===
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Training", leave=False)

        for batch in train_bar:
            corr_scale_attr = torch.rand(1).item()
            corr_scale_v_latents = torch.rand(1).item()

            batch = transfer_batch_to_device(batch, device)

            # Corrupt the domain encodings
            domain_encoded = module.encode_domains(batch[0])

            v = torch.randn(12).to(device)
            v = (v - v.mean()) / v.std()

            # Perform a coin flip (randomly choose either 0 or 1)
            coin_flip = random.choice([0, 1])

            if coin_flip == 0:
                # Corrupt v_latents
                domain_encoded[input_key]['v_latents'] += corr_scale_v_latents * v
            else:
                # Corrupt attr
                domain_encoded[input_key]['attr'] += corr_scale_attr * v

            encoded = module.encode_and_fuse(domain_encoded, module.selection_mod)

            # Extract features and labels
            features = encoded[input_key]
            labels = batch[0][input_key]['attr'][0]
            labels_indices = torch.argmax(labels, dim=1)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels_indices)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update running loss and accuracy
            running_loss += loss.item() * labels_indices.size(0)
            running_corrects += (outputs.argmax(1) == labels_indices).sum().item()
            total_samples += labels_indices.size(0)

            # Update progress bar
            running_avg_loss = running_loss / total_samples
            running_avg_acc = 100.0 * running_corrects / total_samples
            train_bar.set_postfix(
                loss=f"{running_avg_loss:.4f}",
                accuracy=f"{running_avg_acc:.2f}%"
            )

        # === Validation Phase ===
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():  # Disable gradient computation
            for val_batch in val_loader:
                # Validation data transfer and batch reconstruction
                val_batch = transfer_batch_to_device(val_batch, device)
                data = val_batch[0]  # Reconstructing batch as per the original structure
                batch = {frozenset(data.keys()): data}
                for domain in data:
                    batch[frozenset([domain])] = {domain: data[domain]}

                # Encode and fuse the domains
                domain_encoded = module.encode_domains(batch)
                
                v = torch.randn(12).to(device)
                v = (v - v.mean()) / v.std()
                # Perform a coin flip (randomly choose either 0 or 1)
                coin_flip = random.choice([0, 1])

                if coin_flip == 0:
                    # Corrupt v_latents
                    domain_encoded[input_key]['v_latents'] += corr_scale_v_latents * v
                else:
                    # Corrupt attr
                    domain_encoded[input_key]['attr'] += corr_scale_attr * v

                encoded = module.encode_and_fuse(domain_encoded, module.selection_mod)
                
                # Extract features and labels
                features = encoded[input_key]
                labels = batch[input_key]['attr'][0]
                labels_indices = torch.argmax(labels, dim=1)
                
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, labels_indices)
                
                # Update validation loss and accuracy
                val_loss += loss.item() * labels_indices.size(0)
                val_corrects += (outputs.argmax(1) == labels_indices).sum().item()
                val_total += labels_indices.size(0)
        
        # Calculate epoch-level validation metrics
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100.0 * val_corrects / val_total
        
        # Logging training and validation results for the epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {running_avg_loss:.4f}, Train Accuracy: {running_avg_acc:.2f}%, "
            f"Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_acc:.2f}%")

        torch.save({
            'selection_mod_state_dict': module.selection_mod.state_dict(), 
        }, "attention/moving_vector/coin_flip_att_model_step_3.pth")


if __name__ == "__main__":
    main()
