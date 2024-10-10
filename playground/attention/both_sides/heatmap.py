import logging
from collections.abc import Callable
from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch import set_float32_matmul_precision
from lightning.pytorch import Callback, seed_everything
from shimmer.modules.selection import DynamicQueryAttention
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
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

    #seed_everything(config.seed, workers=True)

    domain_proportion = {
        frozenset(item.domains): item.proportion
        for item in config.global_workspace.domain_proportions
    }
    domain_classes = get_default_domains(
        {domain.domain_type.kind.value for domain in config.global_workspace.domains}
    )

    additional_transforms: dict[str, list[Callable[[Any], Any]]] = {}
    if config.domain_modules.attribute.nullify_rotation:
        logging.info("Nullifying rotation in the attr domain.")
        additional_transforms["attr"] = [nullify_attribute_rotation]
    if config.domain_modules.visual.color_blind:
        logging.info("v domain will be color blind.")
        additional_transforms["v"] = [color_blind_visual_domain]

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
    # Load the model from a checkpoint using the LightningModule
    checkpoint_path = "/home/rbertin/attention/simple-shapes-dataset/simple_shapes_fusion-z4typ6dd/epoch=101.ckpt"

    # Replace the existing global workspace logic to load from the checkpoint
    gw_type = "gw_fusion"  # Assuming it's still fusion-type, adjust as necessary
    loss_coefs_fusion: BroadcastLossCoefs = {
        "contrastives": config.global_workspace.loss_coefficients.contrastives,
        "fused": config.global_workspace.loss_coefficients.fused,
        "translations": config.global_workspace.loss_coefficients.translations,
        "demi_cycles": config.global_workspace.loss_coefficients.demi_cycles,
        "cycles": config.global_workspace.loss_coefficients.cycles,
    }

    # Load the global workspace from the checkpoint
    module = GlobalWorkspace.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        domain_mods=domain_modules,  # Reuse the previously defined domain modules
        gw_encoders=gw_encoders,  # Reuse the defined GW encoders
        gw_decoders=gw_decoders,  # Reuse the defined GW decoders
        workspace_dim=config.global_workspace.latent_dim,
        loss_coefs=loss_coefs_fusion,
        contrastive_loss=contrastive_fn,  # Use the pre-defined contrastive loss
        optim_lr=config.training.optim.lr,
        optim_weight_decay=config.training.optim.weight_decay,
        scheduler_args=SchedulerArgs(
            max_lr=config.training.optim.max_lr,
            total_steps=config.training.max_steps,
        )
    ).cuda()
    
    checkpoint = torch.load("attention/both_sides/both_step_3_hsiz10.pth")
    module.selection_mod = DynamicQueryAttention(10,12,["attr","v_latents"],3,.01).cuda()
    module.selection_mod.load_state_dict(checkpoint['selection_mod_state_dict'])

    # Train a classifier (refer back to the original code for training)
    # We assume you have already trained the classifier and saved it as "classifier_checkpoint.pth"

    # Define the key for encoded representations
    input_key = frozenset({'v_latents', 'attr'})
    
    # === Model and Optimizer Setup ===
    device = torch.device("cuda:0")

    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Get a single batch to determine input and output sizes
    try:
        initial_batch = next(iter(train_loader))
    except StopIteration:
        raise ValueError("Training loader is empty!")
    
    initial_batch = transfer_batch_to_device(initial_batch, device)
    
    # Encode the domains
    domain_encoded = module.encode_domains(initial_batch[0])
    
    # Fuse the encoded representations
    encoded = module.encode_and_fuse(domain_encoded, module.selection_mod)
    
    # Extract features and labels from the initial batch
    features = encoded[input_key]  # Shape: (batch_size, feature_size)
    labels = initial_batch[0][input_key]['attr'][0]  # Shape: (batch_size, num_classes)
    
    print("Initial Labels Shape:", labels.shape)
    
    # Convert one-hot encoded labels to class indices
    labels_indices = torch.argmax(labels, dim=1)  # Shape: (batch_size,)
    
    # Define model parameters based on the initial batch
    input_size = features.shape[1]  # Size of the encoded feature vector
    hidden_size = 128  # Hidden layer size (can be adjusted)
    output_size = labels.shape[1]  # Number of classes
    
    # Load the trained classifier from the checkpoint
    classifier = MLPClassifier(input_size, hidden_size, output_size).to(device)
    classifier.load_state_dict(torch.load("attention/shapes_clf/classifier_checkpoint.pth"))
    classifier.eval()  # Set the classifier to evaluation mode
    import copy
    from itertools import islice

    # Define the number of vectors and batches
    num_vectors = 5
    num_batches = 2

    # Generate five different random vectors v, normalized
    v_list = [torch.randn(12).to(device) for _ in range(num_vectors)]
    v_list = [(v - v.mean()) / v.std() for v in v_list]

    # Initialize a matrix to store the sum of accuracies
    accuracy_sum_matrix = np.zeros((20, 20))

    # Initialize a matrix to count the number of accuracies added per (i,j)
    accuracy_count_matrix = np.zeros((20, 20))

    # Define the scales
    corr_scales_v_latents = np.linspace(-0, 20, 20)
    corr_scales_attr = np.linspace(-0, 1.5, 20)

    # Matrix loop for validation data transformation and classification
    with torch.no_grad():  # Disable gradient computation
        # Initialize matrices to accumulate the results across batches
        accuracy_sum_matrix_per_step = {step: np.zeros((20, 20)) for step in range(0, module.selection_mod.n_steps + 1)}
        accuracy_count_matrix_per_step = {step: np.zeros((20, 20)) for step in range(0, module.selection_mod.n_steps + 1)}

        # Iterate over only five batches using islice
        for val_batch in islice(val_loader, num_batches):
            # === Validation Data Transformation ===
            data = transfer_batch_to_device(val_batch, device)[0]

            # Reconstruct the batch as per your instructions
            batch = {frozenset(data.keys()): data}
            for domain in data:
                batch[frozenset([domain])] = {domain: data[domain]}

            # Encode the domains
            domain_encoded_base = module.encode_domains(batch)

            # Iterate through step limits from 0 to max steps (n_steps)
            for step_limit in range(0, module.selection_mod.n_steps + 1):
                
                # Set the step limit for the current iteration
                module.selection_mod.set_step_limit(step_limit)
                # Loop over corr_scale_attr and corr_scale_v_latents with random sampling of v
                for i, corr_scale_attr in enumerate(corr_scales_attr):
                    for j, corr_scale_v_latents in enumerate(corr_scales_v_latents):
                        total_accuracy = 0  # Accumulate accuracy for averaging

                        for _ in range(10):  # Perform 10 random samples of v
                            # Random vector v
                            v = torch.randn(12).to(torch.device("cuda:0"))
                            v = (v - v.mean()) / v.std()

                            # Deep copy to avoid accumulation
                            domain_encoded = copy.deepcopy(domain_encoded_base)

                            # Apply perturbations to the domain encodings at the encoding level
                            domain_encoded[input_key]['v_latents'] += corr_scale_v_latents * v
                            domain_encoded[input_key]['attr'] += corr_scale_attr * v

                            # Fuse the corrupted domain encodings
                            encoded = module.encode_and_fuse(domain_encoded, module.selection_mod)

                            # Use the trained classifier to classify based on the fused representation
                            features = encoded[frozenset({'v_latents', 'attr'})]  # Get fused vector
                            labels = batch[frozenset({'v_latents', 'attr'})]['attr'][0]  # Extract labels
                            labels_indices = torch.argmax(labels, dim=1)  # Convert one-hot to class indices

                            # Make a prediction using the classifier
                            outputs = classifier(features)
                            _, predicted = torch.max(outputs, 1)
                            correct = (predicted == labels_indices).sum().item()

                            # Compute the accuracy for this iteration
                            accuracy = correct / labels_indices.size(0)
                            total_accuracy += accuracy  # Accumulate accuracy over 10 random vectors

                        # Average the accuracy over the 10 random vectors
                        avg_accuracy = total_accuracy / 10

                        # Update the sum and count matrices with the averaged accuracy for the current step
                        accuracy_sum_matrix_per_step[step_limit][i, j] += avg_accuracy
                        accuracy_count_matrix_per_step[step_limit][i, j] += 1


    n_steps = module.selection_mod.n_steps + 1  # Total number of step limits including step 0
    n_cols = 2  # Define the number of columns for the subplot grid
    n_rows = (n_steps + n_cols - 1) // n_cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 6))  # Adjust height based on number of rows
    axes = axes.ravel()  # Flatten the axes array for easy indexing

    for step_limit in range(0, n_steps):
        # Compute the mean accuracy for this step across all batches
        accuracy_matrix = accuracy_sum_matrix_per_step[step_limit] / accuracy_count_matrix_per_step[step_limit]

        # Define the extent of the image to correspond to the actual values of corr_scales
        extent = [corr_scales_v_latents[0], corr_scales_v_latents[-1], corr_scales_attr[0], corr_scales_attr[-1]]

        # Plot the accuracy matrix as a heatmap for this step limit in a subplot
        ax = axes[step_limit]  # Select the correct subplot
        im = ax.imshow(accuracy_matrix, cmap="viridis", origin="lower", aspect='auto', extent=extent, vmin=0.4, vmax=0.99)  # Set vmin and vmax
        fig.colorbar(im, ax=ax, label="Accuracy")
        
        # Update the ticks to match the actual values of corr_scales_v_latents and corr_scales_attr
        ax.set_xlabel("corr_scale_v_latents")
        ax.set_ylabel("corr_scale_attr")
        ax.set_title(f"Average Accuracy Heatmap (Step {step_limit})")

        # Set the x-ticks and y-ticks to reflect the actual scales
        ax.set_xticks(np.linspace(corr_scales_v_latents[0], corr_scales_v_latents[-1], len(corr_scales_v_latents)))
        ax.set_xticklabels(np.round(corr_scales_v_latents, 2))
        ax.set_yticks(np.linspace(corr_scales_attr[0], corr_scales_attr[-1], len(corr_scales_attr)))
        ax.set_yticklabels(np.round(corr_scales_attr, 2))

    # Hide any unused subplots
    for idx in range(n_steps, len(axes)):
        fig.delaxes(axes[idx])

    # Adjust the layout and save the final figure with all subplots
    plt.tight_layout()
    plt.savefig("average_accuracy_heatmaps_steps_0_to_max.png")

# Define the MLPClassifier class
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


def transfer_batch_to_device(batch, device):
    """
    Move all tensors in the batch to the specified device.
    """
    def move_to_device(batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, dict):
            return {k: move_to_device(v) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [move_to_device(v) for v in batch]
        else:
            return batch  # Handle other types if necessary

    return move_to_device(batch)


if __name__ == "__main__":
    main()
