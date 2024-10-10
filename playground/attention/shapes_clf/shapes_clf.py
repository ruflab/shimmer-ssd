import logging
from collections.abc import Callable
from typing import Any

import torch
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
    LossCoefs,
    SaveMigrations,
)
from shimmer.modules.global_workspace import (
    GlobalWorkspace,
    GlobalWorkspace2Domains,
    GlobalWorkspaceBayesian,
    SchedulerArgs,
    pretrained_global_workspace,
)
from torch import set_float32_matmul_precision

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


def main():
    config = load_config(
        PROJECT_DIR / "config",
        load_files=["train_gw.yaml"],
        debug_mode=DEBUG_MODE,
    )

    seed_everything(config.seed, workers=True)

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
            filter=config.logging.filter_images,
        ),
        LogGWImagesCallback(
            val_samples,
            log_key="images/test",
            mode="test",
            every_n_epochs=None,
            filter=config.logging.filter_images,
        ),
        LogGWImagesCallback(
            train_samples,
            log_key="images/train",
            mode="train",
            every_n_epochs=config.logging.log_train_medias_every_n_epochs,
            filter=config.logging.filter_images,
        ),
    ]

    if config.training.enable_progress_bar:
        callbacks.append(RichProgressBar())


    def transfer_batch_to_device(batch, device):
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

    wandb_logger = None

    set_float32_matmul_precision(config.training.float32_matmul_precision)

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from tqdm import tqdm

    # Define the MLP classifier
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
        
    # Initialize the dataloader (setup for 'fit' stage)
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the module (ensure 'module' is defined and moved to device)
    module = module.to(device)

    # Choose the key corresponding to the encoded representation you want to use
    input_key = frozenset({'v_latents', 'attr'})

    # Initialize the classifier
    # To determine input_size and output_size, we'll process a single batch first
    batch = next(iter(train_loader))
    batch = transfer_batch_to_device(batch, device)

    # Encode the domains
    domain_encoded = module.encode_domains(batch[0])
    encoded = module.encode_and_fuse(domain_encoded, module.selection_mod)

    # Extract features and labels
    features = encoded[input_key]
    labels = batch[0][input_key]['attr'][0]

    # Convert one-hot encoded labels to class indices
    labels_indices = torch.argmax(labels, dim=1)

    # Define model parameters based on the first batch
    input_size = features.shape[1]  # Size of the encoded feature vector
    hidden_size = 128  # You can adjust this value
    output_size = labels.shape[1]  # Number of classes

    # Initialize the classifier
    model = MLPClassifier(input_size, hidden_size, output_size)
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Number of epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        # Initialize the training progress bar
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Training", leave=False)

            # Initialize the dataloader (setup for 'fit' stage)
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Set the device (GPU if available, else CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize the main module (ensure 'module' is defined and moved to device)
    module = module.to(device)
    
    # Define the key for encoded representations
    input_key = frozenset({'v_latents', 'attr'})
    
    # === Model and Optimizer Setup ===
    
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
    
    # Initialize the MLP classifier
    model = MLPClassifier(input_size, hidden_size, output_size)
    model = model.to(device)
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # === Training Loop with Validation ===
    
    num_epochs = 10  # Number of training epochs
    
    for epoch in range(num_epochs):
        # === Training Phase ===
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        # Initialize the progress bar for the training phase
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        
        for batch in progress_bar:
            # Move the batch to the appropriate device
            batch = transfer_batch_to_device(batch, device)
            
            # Encode and fuse the domains
            domain_encoded = module.encode_domains(batch[0])
            encoded = module.encode_and_fuse(domain_encoded, module.selection_mod)
            
            # Extract features and labels
            features = encoded[input_key]  # Shape: (batch_size, feature_size)
            labels = batch[0][input_key]['attr'][0]  # Shape: (batch_size, num_classes)
            labels_indices = torch.argmax(labels, dim=1)  # Shape: (batch_size,)
            
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
            
            # Calculate running metrics
            running_avg_loss = running_loss / total_samples
            running_avg_acc = 100.0 * running_corrects / total_samples
            
            # Update the progress bar with current metrics
            progress_bar.set_postfix(
                loss=f"{running_avg_loss:.4f}",
                accuracy=f"{running_avg_acc:.2f}%"
            )
        
        # Calculate epoch-level training metrics
        epoch_train_loss = running_loss / total_samples
        epoch_train_acc = 100.0 * running_corrects / total_samples
        
        # === Validation Phase ===
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        
        with torch.no_grad():  # Disable gradient computation
            for val_batch in val_loader:
                # === Validation Data Transformation ===
                # Get a batch from the dataloader
                data = transfer_batch_to_device(val_batch, device)[0]
                
                # Reconstruct the batch as per the user's instructions
                batch = {frozenset(data.keys()): data}
                for domain in data:
                    batch[frozenset([domain])] = {domain: data[domain]}
                
                # Encode and fuse the domains
                domain_encoded = module.encode_domains(batch)
                encoded = module.encode_and_fuse(domain_encoded, module.selection_mod)
                
                # Extract features and labels
                features = encoded[input_key]  # Shape: (batch_size, feature_size)
                labels = batch[input_key]['attr'][0]  # Shape: (batch_size, num_classes)
                labels_indices = torch.argmax(labels, dim=1)  # Shape: (batch_size,)
                
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
        
        # === Logging ===
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_acc:.2f}%, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_acc:.2f}%")
    
    print("Training Complete!")
    torch.save(model.state_dict(), "classifier_checkpoint.pth")

if __name__ == "__main__":
    main()
