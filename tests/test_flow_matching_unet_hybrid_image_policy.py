import unittest
import tempfile

from omegaconf import OmegaConf
import torch
import torch.nn as nn

from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.policy.flow_matching_unet_hybrid_image_policy import (
    FlowMatchingUnetHybridImagePolicy,
    euler_integrate,
    linear_flow_interpolation,
)
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.workspace.train_flow_matching_unet_hybrid_workspace import (
    FLOW_MATCHING_POLICY_TARGET,
    TrainFlowMatchingUnetHybridWorkspace,
)


class TinyObsEncoder(nn.Module):
    def forward(self, obs):
        image_feature = obs["image"].mean(dim=(1, 2, 3), keepdim=False)
        return torch.cat([image_feature[:, None], obs["agent_pos"]], dim=-1)


class TinyVelocityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))

    def forward(
            self,
            sample,
            timestep,
            local_cond=None,
            global_cond=None):
        time = timestep.reshape(-1, 1, 1).to(sample.dtype)
        output = self.scale * sample + 0.001 * time
        if global_cond is not None:
            output = output + 0.0 * global_cond.mean(
                dim=-1, keepdim=True
            ).unsqueeze(-1)
        return output


def make_identity_normalizer():
    normalizer = LinearNormalizer()
    normalizer["image"] = SingleFieldLinearNormalizer.create_identity()
    normalizer["agent_pos"] = SingleFieldLinearNormalizer.create_identity()
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    return normalizer


def make_tiny_policy(num_inference_steps):
    policy = FlowMatchingUnetHybridImagePolicy.__new__(
        FlowMatchingUnetHybridImagePolicy
    )
    BaseImagePolicy.__init__(policy)
    policy.obs_encoder = TinyObsEncoder()
    policy.model = TinyVelocityModel()
    policy.mask_generator = LowdimMaskGenerator(
        action_dim=2,
        obs_dim=0,
        max_n_obs_steps=2,
        fix_obs_steps=True,
        action_visible=False
    )
    policy.normalizer = make_identity_normalizer()
    policy.horizon = 16
    policy.obs_feature_dim = 3
    policy.action_dim = 2
    policy.n_action_steps = 8
    policy.n_obs_steps = 2
    policy.num_inference_steps = num_inference_steps
    policy.time_embed_scale = 100.0
    policy.obs_as_global_cond = True
    policy.kwargs = {}
    return policy


class FlowMatchingMathTest(unittest.TestCase):
    def test_linear_interpolation_endpoints_and_velocity(self):
        source = torch.tensor([
            [[0.0, 1.0], [2.0, 3.0]],
            [[4.0, 5.0], [6.0, 7.0]],
        ])
        target = source + 2.0

        state, velocity = linear_flow_interpolation(
            source,
            target,
            torch.tensor([0.0, 1.0])
        )

        torch.testing.assert_close(state[0], source[0])
        torch.testing.assert_close(state[1], target[1])
        torch.testing.assert_close(velocity, torch.full_like(source, 2.0))

    def test_linear_interpolation_broadcasts_per_batch_time(self):
        source = torch.zeros((2, 3, 4))
        target = torch.ones_like(source)
        state, _ = linear_flow_interpolation(
            source,
            target,
            torch.tensor([0.25, 0.75])
        )
        torch.testing.assert_close(state[0], torch.full((3, 4), 0.25))
        torch.testing.assert_close(state[1], torch.full((3, 4), 0.75))

    def test_euler_integrates_constant_velocity(self):
        initial = torch.zeros((3, 4, 2))

        def constant_velocity(state, time):
            return torch.full_like(state, 2.5)

        result = euler_integrate(
            constant_velocity,
            initial,
            num_steps=10
        )
        torch.testing.assert_close(result, torch.full_like(initial, 2.5))

    def test_euler_preserves_conditioned_values(self):
        initial = torch.zeros((1, 3, 2))
        condition_data = torch.full_like(initial, 7.0)
        condition_mask = torch.zeros_like(initial, dtype=torch.bool)
        condition_mask[:, :1] = True

        result = euler_integrate(
            lambda state, time: torch.ones_like(state),
            initial,
            num_steps=4,
            condition_data=condition_data,
            condition_mask=condition_mask
        )

        torch.testing.assert_close(result[:, :1], condition_data[:, :1])
        torch.testing.assert_close(result[:, 1:], torch.ones((1, 2, 2)))


class FlowMatchingPolicyTest(unittest.TestCase):
    def setUp(self):
        self.batch = {
            "obs": {
                "image": torch.rand((2, 16, 3, 4, 4)),
                "agent_pos": torch.rand((2, 16, 2)),
            },
            "action": torch.rand((2, 16, 2)),
        }

    def test_compute_loss_is_finite_and_backward_works(self):
        policy = make_tiny_policy(num_inference_steps=10)
        loss = policy.compute_loss(self.batch)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

        loss.backward()
        self.assertIsNotNone(policy.model.scale.grad)
        self.assertTrue(torch.isfinite(policy.model.scale.grad))

    def test_predict_action_shapes_for_configurable_step_counts(self):
        obs = {
            "image": self.batch["obs"]["image"][:, :2],
            "agent_pos": self.batch["obs"]["agent_pos"][:, :2],
        }
        for num_steps in (10, 20):
            policy = make_tiny_policy(num_inference_steps=num_steps)
            result = policy.predict_action(obs)
            self.assertEqual(result["action"].shape, (2, 8, 2))
            self.assertEqual(result["action_pred"].shape, (2, 16, 2))


class FlowMatchingWorkspaceTest(unittest.TestCase):
    def make_workspace(self, output_dir):
        workspace = TrainFlowMatchingUnetHybridWorkspace.__new__(
            TrainFlowMatchingUnetHybridWorkspace
        )
        BaseWorkspace.__init__(
            workspace,
            cfg=OmegaConf.create({
                "policy": {
                    "_target_": FLOW_MATCHING_POLICY_TARGET
                }
            }),
            output_dir=output_dir
        )
        workspace.model = nn.Linear(2, 2)
        workspace.optimizer = torch.optim.AdamW(
            workspace.model.parameters(),
            lr=1e-4
        )
        workspace.global_step = 123
        workspace.epoch = 45
        return workspace

    def test_rejects_diffusion_checkpoint(self):
        workspace = TrainFlowMatchingUnetHybridWorkspace.__new__(
            TrainFlowMatchingUnetHybridWorkspace
        )
        payload = {
            "cfg": {
                "policy": {
                    "_target_": (
                        "diffusion_policy.policy."
                        "diffusion_unet_hybrid_image_policy."
                        "DiffusionUnetHybridImagePolicy"
                    )
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "incompatible checkpoint"):
            workspace.load_payload(payload)

    def test_flow_checkpoint_restores_training_state(self):
        with tempfile.TemporaryDirectory() as output_dir:
            source = self.make_workspace(output_dir)
            with torch.no_grad():
                source.model.weight.fill_(3.0)
            checkpoint_path = source.save_checkpoint(use_thread=False)

            restored = self.make_workspace(output_dir)
            restored.global_step = 0
            restored.epoch = 0
            restored.load_checkpoint(checkpoint_path)

            self.assertEqual(restored.global_step, 123)
            self.assertEqual(restored.epoch, 45)
            torch.testing.assert_close(
                restored.model.weight,
                torch.full_like(restored.model.weight, 3.0)
            )


if __name__ == "__main__":
    unittest.main()
