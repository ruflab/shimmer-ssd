import torch

from simple_shapes_dataset import PROJECT_DIR

state_dict_rename_to = {
    "encoder": "vae.encoder",
    "decoder": "vae.decoder",
    "q_mean": "vae.encoder.q_mean",
    "q_logvar": "vae.encoder.q_logvar",
}

state_dict_remove = [
    "validation_sampling_z",
    "validation_reconstruction_images",
    "log_sigma",
    "output_fun",
]


hyper_parameters = {
    "num_channels": lambda old: old["channel_num"],
    "latent_dim": lambda old: old["z_size"],
    "ae_dim": lambda old: old["ae_size"],
    "beta": lambda old: old["beta"],
    "optim_lr": lambda old: old["optim_lr"],
    "optim_weight_decay": lambda old: old["optim_weight_decay"],
    "scheduler_args": lambda old: {
        "max_lr": old["optim_lr"],
        "total_steps": 100_000,
    },
}


def import_v():
    model = torch.load(PROJECT_DIR / "checkpoints/pretrained/vae_v.ckpt")

    new_state_dict = {}
    for key, val in model["state_dict"].items():
        for orig_key in state_dict_remove:
            if orig_key in key:
                break
        else:
            for orig_key, new_key in state_dict_rename_to.items():
                if orig_key in key and new_key is not None:
                    new_state_dict[key.replace(orig_key, new_key)] = val

    new_callbacks = {}
    for key, val in model["callbacks"].items():
        if "ModelCheckpoint" in key:
            new_key = key.replace(", 'save_on_train_epoch_end': True", "")
            new_callbacks[new_key.replace("val_total_loss", "val/loss")] = val

    new_hyperparams = {
        key: fn(model["hyper_parameters"]) for key, fn in hyper_parameters.items()
    }

    new_model = {
        "epoch": model["epoch"],
        "global_step": model["global_step"],
        "pytorch-lightning_version": model["pytorch-lightning_version"],
        "state_dict": new_state_dict,
        "callbacks": new_callbacks,
        "hparams_name": model["hparams_name"],
        "hyper_parameters": new_hyperparams,
    }

    torch.save(new_model, PROJECT_DIR / "checkpoints/pretrained/vae_v_shimmer.ckpt")


if __name__ == "__main__":
    import_v()
