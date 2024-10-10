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
from torch.optim.lr_scheduler import OneCycleLR  # Import OneCycleLR
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
import numpy as np

# Importing the sample drawing utility function
from scipy.stats import rv_continuous

def sample_corruption_scales(corr_scales_attr, corr_scales_v_latents, allowed_mask):
    indices = np.column_stack(np.where(allowed_mask))  # Get the coordinates of the allowed regions
    chosen_index = indices[np.random.choice(indices.shape[0])]  # Randomly pick one of the allowed regions
    return corr_scales_attr[chosen_index[0]], corr_scales_v_latents[chosen_index[1]]

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

    module.selection_mod = DynamicQueryAttention(10, 12, ["attr", "v_latents"], 3, 0.001).cuda()

    # Define the key for encoded representations
    input_key = frozenset({'v_latents', 'attr'})

    # Set up dataloaders
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

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
    batch = transfer_batch_to_device(batch, torch.device("cuda:0"))
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
    model = MLPClassifier(input_size, hidden_size, output_size).to(torch.device("cuda:0"))
    model.load_state_dict(torch.load("attention/shapes_clf/classifier_checkpoint.pth"))
    model.eval()  # Set the classifier to evaluation mode
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(module.selection_mod.parameters(), lr=0.0003)
    scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=10)

    # === Set up the corruption scale ranges with more values at the higher end ===

    # Original linear space
    corr_scales_v_latents = np.linspace(0.01, 20.0, 20)  # Start from a small positive value to avoid log(0)
    corr_scales_attr = np.linspace(0.01, 1.5, 20)

    # Apply logarithmic transformation to have more values at the higher end
    corr_scales_v_latents = np.log(corr_scales_v_latents)  # Log to skew toward higher values
    corr_scales_v_latents = np.interp(corr_scales_v_latents, (corr_scales_v_latents.min(), corr_scales_v_latents.max()), (0, 10.0))  # Normalize back to original range

    corr_scales_attr = np.log(corr_scales_attr)  # Log to skew toward higher values
    corr_scales_attr = np.interp(corr_scales_attr, (corr_scales_attr.min(), corr_scales_attr.max()), (0, 2.5))  # Normalize back to original range

    # Define the allowed regions for the corruption scales based on your conditions
    allowed_mask = np.zeros((len(corr_scales_attr), len(corr_scales_v_latents)), dtype=bool)

    easiness = 100.

    for i, corr_scale_attr_val in enumerate(corr_scales_attr):
        for j, corr_scale_v_latents_val in enumerate(corr_scales_v_latents):
            if easiness * corr_scale_attr_val <= corr_scale_v_latents_val:
                allowed_mask[i, j] = True
                print(f"allowed: corr_scale_attr = {corr_scale_attr_val:.2f}, corr_scale_v_latents = {corr_scale_v_latents_val:.2f}")
            if easiness * corr_scale_v_latents_val <= corr_scale_attr_val:
                allowed_mask[i, j] = True
                print(f"allowed: corr_scale_attr = {corr_scale_attr_val:.2f}, corr_scale_v_latents = {corr_scale_v_latents_val:.2f}")


    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        # === Training Phase ===
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Training", leave=False)

        for batch in train_bar:
            # Initialize the total loss for all steps
            total_loss_across_steps = 0.0

            # Sample the corruption scales using the distribution
            corr_scale_attr, corr_scale_v_latents = sample_corruption_scales(
                corr_scales_attr, corr_scales_v_latents, allowed_mask
            )

            # Transfer batch to device
            batch = transfer_batch_to_device(batch, torch.device("cuda:0"))

            # Corrupt the domain encodings
            domain_encoded = module.encode_domains(batch[0])

            # Random vector v
            v = torch.randn(12).to(torch.device("cuda:0"))
            v = (v - v.mean()) / v.std()

            # Apply corruption to `v_latents` and `attr`
            domain_encoded[input_key]['v_latents'] += corr_scale_v_latents * v
            domain_encoded[input_key]['attr'] += corr_scale_attr * v

            # Iterate through step limits from 1 to the max_steps (n_steps)
            for step_limit in range(1, module.selection_mod.n_steps + 1):
                # Set the current step limit in the modified selection module
                module.selection_mod.set_step_limit(step_limit)
                # Encode and fuse the corrupted domain encodings
                encoded = module.encode_and_fuse(domain_encoded, module.selection_mod)

                # Extract features and labels
                features = encoded[input_key]
                labels = batch[0][input_key]['attr'][0]
                labels_indices = torch.argmax(labels, dim=1)

                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, labels_indices)

                # Accumulate loss for this step
                total_loss_across_steps += loss

            # Average the loss across all steps
            avg_loss_across_steps = total_loss_across_steps / module.selection_mod.n_steps

            # Backward pass and optimization
            optimizer.zero_grad()
            avg_loss_across_steps.backward()  # Use the averaged loss for backpropagation
            optimizer.step()
            scheduler.step()

            # Update running loss and accuracy
            running_loss += avg_loss_across_steps.item() * labels_indices.size(0)
            running_corrects += (outputs.argmax(1) == labels_indices).sum().item()
            total_samples += labels_indices.size(0)

            # Update progress bar with average loss and accuracy
            running_avg_loss = running_loss / total_samples
            running_avg_acc = 100.0 * running_corrects / total_samples
            lr = optimizer.param_groups[0]['lr']
            train_bar.set_postfix(
                loss=f"{running_avg_loss:.4f}",
                accuracy=f"{running_avg_acc:.2f}%",
                lr=f"{lr}%"
            )

        # === Validation Phase ===
        module.selection_mod.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():  # Disable gradient computation
            for val_batch in val_loader:
                # Initialize total validation loss across steps
                total_val_loss_across_steps = 0.0

                # Validation data transfer and batch reconstruction
                val_batch = transfer_batch_to_device(val_batch, torch.device("cuda:0"))
                data = val_batch[0]  # Reconstructing batch as per the original structure
                batch = {frozenset(data.keys()): data}
                for domain in data:
                    batch[frozenset([domain])] = {domain: data[domain]}

                # Encode the domains
                domain_encoded = module.encode_domains(batch)

                # Sample the corruption scales using the distribution
                corr_scale_attr, corr_scale_v_latents = sample_corruption_scales(
                    corr_scales_attr, corr_scales_v_latents, allowed_mask
                )

                # Random vector `v` for validation
                v = torch.randn(12).to(torch.device("cuda:0"))
                v = (v - v.mean()) / v.std()

                # Apply corruption to `v_latents` and `attr`
                domain_encoded[input_key]['v_latents'] += corr_scale_v_latents * v
                domain_encoded[input_key]['attr'] += corr_scale_attr * v

                # Iterate through step limits from 1 to max_steps (n_steps)
                for step_limit in range(1, module.selection_mod.n_steps + 1):
                    # Set the current step limit in the modified selection module
                    module.selection_mod.set_step_limit(step_limit)

                    # Encode and fuse the corrupted domain encodings
                    encoded = module.encode_and_fuse(domain_encoded, module.selection_mod)

                    # Extract features and labels
                    features = encoded[input_key]
                    labels = batch[input_key]['attr'][0]
                    labels_indices = torch.argmax(labels, dim=1)

                    # Forward pass
                    outputs = model(features)
                    loss = criterion(outputs, labels_indices)

                    # Accumulate validation loss across steps
                    total_val_loss_across_steps += loss.item()

                # Average the loss across all steps
                avg_val_loss_across_steps = total_val_loss_across_steps / module.selection_mod.n_steps

                # Update validation loss and accuracy
                val_loss += avg_val_loss_across_steps * labels_indices.size(0)
                val_corrects += (outputs.argmax(1) == labels_indices).sum().item()
                val_total += labels_indices.size(0)

        # Calculate epoch-level validation metrics
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100.0 * val_corrects / val_total

        # Logging training and validation results for the epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {running_avg_loss:.4f}, Train Accuracy: {running_avg_acc:.2f}%, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_acc:.2f}%")

        # Save the model's state
        torch.save({
            'selection_mod_state_dict': module.selection_mod.state_dict(),
        }, "attention/both_sides/both_step_3_hsiz10.pth")


if __name__ == "__main__":
    main()
