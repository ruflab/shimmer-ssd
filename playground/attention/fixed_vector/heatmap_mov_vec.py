import logging
from collections.abc import Callable
from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch import set_float32_matmul_precision
from lightning.pytorch import Callback, seed_everything
from shimmer.modules.selection import DynamicQueryAttention, FixedDomainSelection, FixedSharedSelection
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
    checkpoint = torch.load("attention/fixed_vector/coin_flip_att_model.pth")
    #module.selection_mod = DynamicQueryAttention(5,12,["attr","v_latents"],0).cuda()
    #module.selection_mod.load_state_dict(checkpoint['selection_mod_state_dict'])
    module.selection_mod = FixedDomainSelection(n=1)

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

    # Initialize a matrix to store the sum of accuracies
    accuracy_sum_matrix = np.zeros((30, 30))

    # Initialize a matrix to count the number of accuracies added per (i,j)
    accuracy_count_matrix = np.zeros((30, 30))
    # Define the scales for attr and v_latents
    corr_scales_attr = np.linspace(0, 5., 30)  # Corruption scale for attr goes from 0 to 1
    corr_scales_v_latents = np.linspace(0, 5.0, 30)  # Corruption scale for v_latents goes from 0 to 5

    # Matrix loop for validation data transformation and classification
    with torch.no_grad():  # Disable gradient computation
        # Iterate over only five batches using islice
        for val_batch in islice(val_loader, 5):
            # === Validation Data Transformation ===
            data = transfer_batch_to_device(val_batch, device)[0]

            # Reconstruct the batch as per your instructions
            batch = {frozenset(data.keys()): data}
            for domain in data:
                batch[frozenset([domain])] = {domain: data[domain]}

            # Encode the domains
            domain_encoded_base = module.encode_domains(batch)

            # Loop over corr_scale_attr and corr_scale_v_latents with their respective scales
            for i, corr_scale_attr in enumerate(corr_scales_attr):
                for j, corr_scale_v_latents in enumerate(corr_scales_v_latents):
                    total_accuracy = 0  # Accumulate accuracy for averaging

                    for _ in range(10):  # Run for 10 different random vectors
                        # Random vector v
                        v = torch.randn(12).to(torch.device("cuda:0"))
                        v = (v - v.mean()) / v.std()

                        # Deep copy to avoid accumulation
                        domain_encoded = copy.deepcopy(domain_encoded_base)

                        # Apply the loaded corruption vector to the domain encodings
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

                        # Compute accuracy for this iteration
                        accuracy = correct / labels_indices.size(0)
                        total_accuracy += accuracy  # Accumulate accuracy over 10 random vectors

                    # Average the accuracy over 10 random vectors
                    avg_accuracy = total_accuracy / 10

                    # Update the sum and count matrices with the averaged accuracy
                    accuracy_sum_matrix[i, j] += avg_accuracy
                    accuracy_count_matrix[i, j] += 1

    # Compute the mean accuracy
    accuracy_matrix = accuracy_sum_matrix / accuracy_count_matrix

    # Plot the accuracy matrix as a heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(accuracy_matrix, cmap="viridis", origin="lower", aspect='auto')
    plt.colorbar(label="Accuracy")
    plt.xlabel("corr_scale_v_latents")
    plt.ylabel("corr_scale_attr")
    plt.title("Accuracy Heatmap (Varying corr_scale_v_latents and corr_scale_attr)")
    plt.tick_params(axis='both', which='major', labelsize=7)  # Adjust 'labelsize' to the desired size
    plt.xticks(np.arange(30), np.round(corr_scales_v_latents, 2))
    plt.yticks(np.arange(30), np.round(corr_scales_attr, 2))
    plt.savefig("accuracy_heatmap.png")
    plt.show()


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
