from migrate_ckpt import CkptType


def handle(ckpt: CkptType) -> CkptType:
    """
    This removes the logit_scale key in the t domain checkpoint
    """
    del ckpt["state_dict"]["gw_mod.domain_mods.t.logit_scale"]
    del ckpt["state_dict"]["loss_mod.gw_mod.domain_mods.t.logit_scale"]
    del ckpt["state_dict"]["loss_mod.domain_mods.t.logit_scale"]
    return ckpt
