from omegaconf import OmegaConf

from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import (
    TrainDiffusionUnetHybridWorkspace,
)


FLOW_MATCHING_POLICY_TARGET = (
    "diffusion_policy.policy.flow_matching_unet_hybrid_image_policy."
    "FlowMatchingUnetHybridImagePolicy"
)


class TrainFlowMatchingUnetHybridWorkspace(
        TrainDiffusionUnetHybridWorkspace):
    """U-Net training workspace with flow-matching checkpoint validation."""

    def load_payload(self, payload, *args, **kwargs):
        checkpoint_cfg = payload.get("cfg")
        if not OmegaConf.is_config(checkpoint_cfg):
            checkpoint_cfg = OmegaConf.create(checkpoint_cfg)
        checkpoint_target = OmegaConf.select(
            checkpoint_cfg,
            "policy._target_"
        )
        if checkpoint_target != FLOW_MATCHING_POLICY_TARGET:
            raise ValueError(
                "Refusing to load an incompatible checkpoint: expected "
                f"{FLOW_MATCHING_POLICY_TARGET}, got {checkpoint_target!r}"
            )
        return super().load_payload(payload, *args, **kwargs)
