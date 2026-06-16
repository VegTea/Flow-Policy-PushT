from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
import diffusion_policy.model.vision.crop_randomizer as dmvc
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.models.base_nets as rmbn
import robomimic.utils.obs_utils as ObsUtils


def _expand_time(time: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if time.ndim != 1 or time.shape[0] != reference.shape[0]:
        raise ValueError(
            f"time must have shape ({reference.shape[0]},), got {tuple(time.shape)}"
        )
    return time.reshape(time.shape[0], *([1] * (reference.ndim - 1)))


def linear_flow_interpolation(
        source: torch.Tensor,
        target: torch.Tensor,
        time: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return x_t and the constant target velocity for a linear flow path."""
    if source.shape != target.shape:
        raise ValueError(
            f"source and target must have the same shape, got "
            f"{tuple(source.shape)} and {tuple(target.shape)}"
        )
    expanded_time = _expand_time(time, source)
    state = (1.0 - expanded_time) * source + expanded_time * target
    velocity = target - source
    return state, velocity


def euler_integrate(
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        initial_state: torch.Tensor,
        num_steps: int,
        condition_data: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
    """Integrate dx/dt = velocity_fn(x, t) from t=0 to t=1."""
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if (condition_data is None) != (condition_mask is None):
        raise ValueError("condition_data and condition_mask must be provided together")
    if condition_data is not None:
        if condition_data.shape != initial_state.shape:
            raise ValueError("condition_data must match initial_state shape")
        if condition_mask.shape != initial_state.shape:
            raise ValueError("condition_mask must match initial_state shape")

    state = initial_state
    dt = 1.0 / num_steps
    batch_size = initial_state.shape[0]
    for step_idx in range(num_steps):
        if condition_data is not None:
            state = torch.where(condition_mask, condition_data, state)
        time = torch.full(
            (batch_size,),
            fill_value=step_idx * dt,
            device=state.device,
            dtype=state.dtype
        )
        state = state + dt * velocity_fn(state, time)

    if condition_data is not None:
        state = torch.where(condition_mask, condition_data, state)
    return state


class FlowMatchingUnetHybridImagePolicy(BaseImagePolicy):
    def __init__(
            self,
            shape_meta: dict,
            horizon,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=10,
            time_embed_scale=100.0,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256, 512, 1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            **kwargs):
        super().__init__()

        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if time_embed_scale <= 0:
            raise ValueError("time_embed_scale must be positive")

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta["obs"]
        obs_config = {
            "low_dim": [],
            "rgb": [],
            "depth": [],
            "scan": []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr["shape"]
            obs_key_shapes[key] = list(shape)

            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                obs_config["rgb"].append(key)
            elif obs_type == "low_dim":
                obs_config["low_dim"].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")

        config = get_robomimic_config(
            algo_name="bc_rnn",
            hdf5_type="image",
            task_name="square",
            dataset_type="ph"
        )

        with config.unlocked():
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for _, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality["obs_randomizer_class"] = None
            else:
                crop_height, crop_width = crop_shape
                for _, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality.obs_randomizer_kwargs.crop_height = crop_height
                        modality.obs_randomizer_kwargs.crop_width = crop_width

        ObsUtils.initialize_obs_utils_with_config(config)

        policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device="cpu"
        )
        obs_encoder = policy.nets["policy"].nets["encoder"].nets["obs"]

        if obs_encoder_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16,
                    num_channels=x.num_features
                )
            )

        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.num_inference_steps = num_inference_steps
        self.time_embed_scale = float(time_embed_scale)
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        print(
            "Flow matching params: %e"
            % sum(p.numel() for p in self.model.parameters())
        )
        print(
            "Vision params: %e"
            % sum(p.numel() for p in self.obs_encoder.parameters())
        )

    def conditional_sample(
            self,
            condition_data,
            condition_mask,
            local_cond=None,
            global_cond=None,
            generator=None):
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator
        )

        def velocity_fn(state, time):
            return self.model(
                state,
                time * self.time_embed_scale,
                local_cond=local_cond,
                global_cond=global_cond
            )

        return euler_integrate(
            velocity_fn=velocity_fn,
            initial_state=trajectory,
            num_steps=self.num_inference_steps,
            condition_data=condition_data,
            condition_mask=condition_mask
        )

    def predict_action(
            self,
            obs_dict: Dict[str, torch.Tensor]
            ) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        horizon = self.horizon
        action_dim = self.action_dim
        obs_feature_dim = self.obs_feature_dim
        n_obs_steps = self.n_obs_steps

        device = self.device
        dtype = self.dtype
        local_cond = None
        global_cond = None

        if self.obs_as_global_cond:
            this_nobs = dict_apply(
                nobs,
                lambda x: x[:, :n_obs_steps, ...].reshape(
                    -1, *x.shape[2:]
                )
            )
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(batch_size, -1)
            cond_data = torch.zeros(
                size=(batch_size, horizon, action_dim),
                device=device,
                dtype=dtype
            )
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            this_nobs = dict_apply(
                nobs,
                lambda x: x[:, :n_obs_steps, ...].reshape(
                    -1, *x.shape[2:]
                )
            )
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(
                batch_size, n_obs_steps, -1
            )
            cond_data = torch.zeros(
                size=(batch_size, horizon, action_dim + obs_feature_dim),
                device=device,
                dtype=dtype
            )
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :n_obs_steps, action_dim:] = nobs_features
            cond_mask[:, :n_obs_steps, action_dim:] = True

        nsample = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            local_cond=local_cond,
            global_cond=global_cond
        )

        normalized_action_pred = nsample[..., :action_dim]
        action_pred = self.normalizer["action"].unnormalize(
            normalized_action_pred
        )

        start = n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        return {
            "action": action,
            "action_pred": action_pred
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        target_trajectory = self.normalizer["action"].normalize(
            batch["action"]
        )
        batch_size = target_trajectory.shape[0]
        horizon = target_trajectory.shape[1]

        local_cond = None
        global_cond = None
        condition_data = target_trajectory
        if self.obs_as_global_cond:
            this_nobs = dict_apply(
                nobs,
                lambda x: x[:, :self.n_obs_steps, ...].reshape(
                    -1, *x.shape[2:]
                )
            )
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            this_nobs = dict_apply(
                nobs,
                lambda x: x.reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            condition_data = torch.cat(
                [target_trajectory, nobs_features],
                dim=-1
            )
            target_trajectory = condition_data.detach()

        condition_mask = self.mask_generator(target_trajectory.shape)
        source_trajectory = torch.randn_like(target_trajectory)
        time = torch.rand(
            (batch_size,),
            device=target_trajectory.device,
            dtype=target_trajectory.dtype
        )
        interpolated, target_velocity = linear_flow_interpolation(
            source=source_trajectory,
            target=target_trajectory,
            time=time
        )

        interpolated = torch.where(
            condition_mask,
            condition_data,
            interpolated
        )
        pred_velocity = self.model(
            interpolated,
            time * self.time_embed_scale,
            local_cond=local_cond,
            global_cond=global_cond
        )

        loss_mask = ~condition_mask
        loss = F.mse_loss(
            pred_velocity,
            target_velocity,
            reduction="none"
        )
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        return loss.mean()
