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

from playground.text_bimgw import BimgTextDomainModule
from shimmer import DomainModule, GWDecoder, GWEncoder, GWEncoderLinear


import psutil
import time
import threading
import subprocess

def get_gpu_usage():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,nounits,noheader'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode == 0:
            usage = result.stdout.strip().split('\n')
            gpus = []
            for gpu in usage:
                utilization, mem_used, mem_total = map(int, gpu.split(', '))
                gpus.append({
                    'utilization': utilization,
                    'mem_used': mem_used,
                    'mem_total': mem_total,
                })
            return gpus
        else:
            return None
    except Exception as e:
        print(f"Failed to get GPU usage: {e}")
        return None

def log_resource_usage():
    process = psutil.Process()
    while True:
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = process.memory_info().rss / 1e6
        open_files = len(process.open_files())
        
        # Virtual memory info
        virtual_memory = psutil.virtual_memory()
        total_memory = virtual_memory.total / 1e6  # Convert to MB
        available_memory = virtual_memory.available / 1e6  # Convert to MB
        used_memory = virtual_memory.used / 1e6  # Convert to MB
        
        # GPU usage
        gpu_usage = get_gpu_usage()

        # Log the usage
        log_message = (
            f"CPU Memory: {memory_info} MB; Open files: {open_files}; CPU Usage: {cpu_usage}%\n"
            f"Total Memory: {total_memory} MB; Used Memory: {used_memory} MB; Available Memory: {available_memory} MB"
        )
        if gpu_usage:
            for i, gpu in enumerate(gpu_usage):
                log_message += (
                    f"; GPU {i} - Utilization: {gpu['utilization']}%, "
                    f"Memory: {gpu['mem_used']}/{gpu['mem_total']} MB"
                )
        
        print(log_message)
        time.sleep(5)

# Start the resource monitoring thread
#resource_thread = threading.Thread(target=log_resource_usage)
#resource_thread.start()



def main():
    print("got in main")
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

    print("got here", config.global_workspace.domains)
    exit()

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


    print("got datamodule")

    print("domains : ", config.global_workspace.domains)# Filter out the text domain and handle it separately

    filtered_domains = [
        domain for domain in config.global_workspace.domains
        if domain.domain_type.kind.value.kind != 't'
    ]

    # Load other domains excluding the text domain
    domain_modules, gw_encoders, gw_decoders = load_pretrained_domains(
        config.default_root_dir,
        filtered_domains,
        config.global_workspace.latent_dim,
        config.global_workspace.encoders.hidden_dim,
        config.global_workspace.encoders.n_layers,
        config.global_workspace.decoders.hidden_dim,
        config.global_workspace.decoders.n_layers,
        is_linear=config.global_workspace.linear_domains,
        bias=config.global_workspace.linear_domains_use_bias,
    )

    # Configuration values for BimgTextDomainModule
    latent_dim = 12
    hidden_dim = 256
    n_classes = 3
    checkpoint_path = '../checkpoints/vae_t.ckpt'

    # Instantiate the BimgTextDomainModule
    text_domain_module = BimgTextDomainModule(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        beta=1,
        optim_lr=1e-3,
        optim_weight_decay=0,
        scheduler_args=None,
        checkpoint_path=checkpoint_path,
        z_size=latent_dim,
        n_classes=n_classes
    )

    # Add the BimgTextDomainModule to the domain_modules
    domain_modules['t'] = text_domain_module

    # Manually create GWEncoder and GWDecoder for text domain
    if config.global_workspace.linear_domains:
        text_gw_encoder = GWEncoderLinear(
            latent_dim,  # in_dim (module.latent_dim)
            config.global_workspace.latent_dim,  # out_dim (workspace_dim)
            bias=config.global_workspace.linear_domains_use_bias
        )
    else:
        text_gw_encoder = GWEncoder(
            latent_dim,  # in_dim (module.latent_dim)
            config.global_workspace.encoders.hidden_dim,  # hidden_dim
            config.global_workspace.latent_dim,  # out_dim (workspace_dim)
            config.global_workspace.encoders.n_layers  # n_layers
        )

    text_gw_decoder = GWDecoder(
        config.global_workspace.latent_dim,  # in_dim (workspace_dim)
        config.global_workspace.decoders.hidden_dim,  # hidden_dim
        latent_dim,  # out_dim (module.latent_dim)
        config.global_workspace.decoders.n_layers  # n_layers
    )

    # Add the GWEncoder and GWDecoder to the respective dictionaries
    gw_encoders['t'] = text_gw_encoder
    gw_decoders['t'] = text_gw_decoder



    contrastive_fn: ContrastiveLossType | None = None
    if config.global_workspace.vsepp_contrastive_loss:
        contrastive_fn = VSEPPContrastiveLoss(
            config.global_workspace.vsepp_margin,
            config.global_workspace.vsepp_measure,
            config.global_workspace.vsepp_max_violation,
            torch.tensor([1 / 0.07]).log(),
        )

    module: GlobalWorkspaceBase
    gw_type: str
    if config.global_workspace.bayesian_gw:
        gw_type = "gw_bayesian"
        loss_coefs_bayesian: BroadcastLossCoefs = {
            "contrastives": config.global_workspace.loss_coefficients.contrastives,
            "fused": config.global_workspace.loss_coefficients.fused,
            "translations": config.global_workspace.loss_coefficients.translations,
            "demi_cycles": config.global_workspace.loss_coefficients.demi_cycles,
            "cycles": config.global_workspace.loss_coefficients.cycles,
        }
        module = GlobalWorkspaceBayesian(
            domain_modules,
            gw_encoders,
            gw_decoders,
            config.global_workspace.latent_dim,
            loss_coefs_bayesian,
            config.global_workspace.selection_temperature,
            config.training.optim.lr,
            config.training.optim.weight_decay,
            scheduler_args=SchedulerArgs(
                max_lr=config.training.optim.max_lr,
                total_steps=config.training.max_steps,
            ),
            learn_logit_scale=config.global_workspace.learn_logit_scale,
            contrastive_loss=contrastive_fn,
        )
    elif config.global_workspace.use_fusion_model:
        gw_type = "gw_fusion"
        loss_coefs_fusion: BroadcastLossCoefs = {
            "contrastives": config.global_workspace.loss_coefficients.contrastives,
            "fused": config.global_workspace.loss_coefficients.fused,
            "translations": config.global_workspace.loss_coefficients.translations,
            "demi_cycles": config.global_workspace.loss_coefficients.demi_cycles,
            "cycles": config.global_workspace.loss_coefficients.cycles,
        }
        module = GlobalWorkspace(
            domain_modules,
            gw_encoders,
            gw_decoders,
            config.global_workspace.latent_dim,
            loss_coefs_fusion,
            config.global_workspace.selection_temperature,
            config.training.optim.lr,
            config.training.optim.weight_decay,
            scheduler_args=SchedulerArgs(
                max_lr=config.training.optim.max_lr,
                total_steps=config.training.max_steps,
            ),
            learn_logit_scale=config.global_workspace.learn_logit_scale,
            contrastive_loss=contrastive_fn,
        )
    else:
        gw_type = "gw"
        loss_coefs: LossCoefs = {
            "demi_cycles": config.global_workspace.loss_coefficients.demi_cycles,
            "cycles": config.global_workspace.loss_coefficients.cycles,
            "translations": config.global_workspace.loss_coefficients.translations,
            "contrastives": config.global_workspace.loss_coefficients.contrastives,
        }

        module = GlobalWorkspace2Domains(
            domain_modules,
            gw_encoders,
            gw_decoders,
            config.global_workspace.latent_dim,
            loss_coefs,
            config.training.optim.lr,
            config.training.optim.weight_decay,
            scheduler_args=SchedulerArgs(
                max_lr=config.training.optim.max_lr,
                total_steps=config.training.max_steps,
            ),
            learn_logit_scale=config.global_workspace.learn_logit_scale,
            contrastive_loss=contrastive_fn,
        )

    
    print("got before samples")

    train_samples = data_module.get_samples("train", 32)
    val_samples = data_module.get_samples("val", 32)
    test_samples = data_module.get_samples("test", 32)

    print("got samples")


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

    print("got before if")

    if config.ood_seed is not None:
        train_samples_ood = data_module.get_samples("train", 32, ood=True)
        val_samples_ood = data_module.get_samples("val", 32, ood=True)
        test_samples_ood = data_module.get_samples("test", 32, ood=True)

        for domains in val_samples_ood:
            for domain in domains:
                val_samples_ood[frozenset([domain])] = {
                    domain: val_samples_ood[domains][domain]
                }
                test_samples_ood[frozenset([domain])] = {
                    domain: test_samples_ood[domains][domain]
                }
            break

        callbacks.extend(
            [
                LogGWImagesCallback(
                    val_samples_ood,
                    log_key="images/val/ood",
                    mode="val",
                    every_n_epochs=config.logging.log_val_medias_every_n_epochs,
                ),
                LogGWImagesCallback(
                    val_samples_ood,
                    log_key="images/test/ood",
                    mode="test",
                    every_n_epochs=None,
                ),
                LogGWImagesCallback(
                    train_samples_ood,
                    log_key="images/train/ood",
                    mode="train",
                    every_n_epochs=config.logging.log_train_medias_every_n_epochs,
                ),
            ]
        )

    if config.training.enable_progress_bar:
        callbacks.append(RichProgressBar())

    wandb_logger = None
    if config.wandb.enabled:
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
                SaveMigrations(),
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename="{epoch}",
                    monitor="val/loss",
                    mode="min",
                    save_top_k=1,
                ),
            ]
        )

    print("got before trainer")
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

    print("got train")

    trainer.fit(module, data_module)
    trainer.validate(module, data_module, "best")
    trainer.test(module, data_module, "best")


if __name__ == "__main__":
    main()
