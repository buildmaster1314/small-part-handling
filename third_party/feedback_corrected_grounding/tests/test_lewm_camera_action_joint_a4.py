from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from fcpg.lewm_action_prefix_joint_a4 import joint_batch_from_memmaps
from fcpg.lewm_camera_action_joint_a4 import (
    EXPANDED_DYNAMICS_CONFIG,
    EXPANDED_RESIDUAL_SCALE,
    ExpandedActionPrefixDynamics,
    camera_action_statistics,
    transform_world_actions_to_camera,
)
from fcpg.lewm_action_prefix import ActionPrefixDynamics
from scripts.train_lewm_action_prefix_joint_a4 import _predict_dynamics_for_model
from scripts.train_lewm_camera_action_joint_a4 import _cuda0, _parser


def test_identity_camera_rotation_preserves_xyz_and_gripper() -> None:
    action = torch.randn(2, 8, 20)
    action[..., 3::4] = torch.linspace(-1.0, 1.0, 5)
    rotation = torch.eye(3).expand(2, -1, -1).clone()

    result = transform_world_actions_to_camera(action, rotation)

    torch.testing.assert_close(result, action)


def test_camera_rotation_maps_world_xyz_before_dynamics() -> None:
    action = torch.zeros(1, 1, 20)
    action[..., 0::4] = 1.0
    action[..., 3::4] = -0.75
    rotation = torch.tensor(
        [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]
    )

    result = transform_world_actions_to_camera(action, rotation)

    torch.testing.assert_close(result[..., 0::4], torch.zeros(1, 1, 5))
    torch.testing.assert_close(result[..., 1::4], torch.ones(1, 1, 5))
    torch.testing.assert_close(result[..., 2::4], torch.zeros(1, 1, 5))
    torch.testing.assert_close(result[..., 3::4], action[..., 3::4])


def test_camera_action_statistics_use_only_train_camera_frame() -> None:
    actions = np.zeros((2, 8, 8, 20), dtype=np.float32)
    actions[0, ..., 0::4] = 2.0
    actions[1, ..., 0::4] = 100.0
    index = {
        "split": np.asarray(["train", "development"]),
        "future_action": actions,
    }
    extrinsic = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    arrays = {"camera_extrinsic_cam_from_world": extrinsic}

    result = camera_action_statistics(index, arrays)

    np.testing.assert_allclose(result["action_mean"][0::4], 2.0)
    np.testing.assert_allclose(result["action_mean"][1::4], 0.0)
    assert np.all(result["action_std"] > 0.0)


def test_joint_batch_carries_exact_camera_matrices() -> None:
    index = {
        "task_index": np.asarray([0], dtype=np.int64),
        "future_action": np.zeros((1, 8, 8, 20), dtype=np.float32),
    }
    arrays = {
        "current_rgb": np.zeros((1, 3, 8, 8), dtype=np.uint8),
        "current_proprio": np.zeros((1, 4), dtype=np.float32),
        "current_q_nominal": np.zeros((1, 3, 2), dtype=np.float32),
        "current_q_corrected": np.zeros((1, 3, 2), dtype=np.float32),
        "current_keypoints_xyz": np.zeros((1, 3, 3), dtype=np.float32),
        "current_valid": np.ones((1, 3), dtype=np.bool_),
        "current_q_true": np.zeros((1, 3, 2), dtype=np.float32),
        "future_rgb": np.zeros((1, 8, 8, 3, 8, 8), dtype=np.uint8),
        "future_proprio": np.zeros((1, 8, 8, 4), dtype=np.float32),
        "future_q_nominal": np.zeros((1, 8, 8, 3, 2), dtype=np.float32),
        "future_q_corrected": np.zeros((1, 8, 8, 3, 2), dtype=np.float32),
        "future_keypoints_xyz": np.zeros((1, 8, 8, 3, 3), dtype=np.float32),
        "future_valid": np.ones((1, 8, 8, 3), dtype=np.bool_),
        "future_q_true": np.zeros((1, 8, 8, 3, 2), dtype=np.float32),
        "camera_intrinsic": np.tile(np.eye(3, dtype=np.float32), (1, 1, 1)),
        "camera_extrinsic_cam_from_world": np.tile(
            np.eye(4, dtype=np.float32), (1, 1, 1)
        ),
    }

    batch = joint_batch_from_memmaps(
        index,
        arrays,
        bundle_indices=np.asarray([0]),
        branch_indices=np.asarray([3]),
    )

    assert batch["camera_intrinsic"].shape == (1, 3, 3)
    assert batch["camera_extrinsic_cam_from_world"].shape == (1, 4, 4)


class _CaptureDynamics(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.observed: torch.Tensor | None = None

    def forward(
        self,
        current_state: torch.Tensor,
        future_action: torch.Tensor,
        task_index: torch.Tensor,
    ) -> torch.Tensor:
        self.observed = future_action.detach().clone()
        return current_state[:, None].expand(-1, future_action.shape[1], -1, -1)


class _TransformingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dynamics = _CaptureDynamics()

    def transform_future_action(
        self, future_action: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return future_action + batch["offset"]


def test_counterfactual_evaluator_uses_model_action_transform() -> None:
    model = _TransformingModel()
    current = torch.zeros(2, 4, 3)
    action = torch.zeros(2, 8, 20)
    batch = {
        "task_index": torch.tensor([0, 1]),
        "offset": torch.ones(2, 8, 20),
    }

    _predict_dynamics_for_model(
        model,
        current_state=current,
        future_action=action,
        batch=batch,
    )

    assert model.dynamics.observed is not None
    torch.testing.assert_close(model.dynamics.observed, torch.ones_like(action))


def test_camera_action_training_rejects_cuda1() -> None:
    with pytest.raises(ValueError, match="CUDA0"):
        _cuda0("cuda:1")


def test_camera_action_training_has_fixed_sigreg_contract() -> None:
    args = _parser(["--output-dir", "/tmp/camera-action"])
    assert not hasattr(args, "sigreg_weight")


def test_expanded_dynamics_is_exact_base_at_initialization() -> None:
    config = {
        "action_dim": 4,
        "state_dim": 12,
        "num_slots": 4,
        "num_tasks": 7,
        "max_horizon": 3,
        "prefix_depth": 1,
        "prefix_heads": 3,
        "prefix_mlp_dim": 24,
        "predictor_depth": 1,
        "predictor_hidden_dim": 24,
        "fusion_dim": 24,
        "state_mlp_dim": 24,
        "dropout": 0.0,
    }
    base = ActionPrefixDynamics(**config).eval()
    expansion = ActionPrefixDynamics(**config).eval()
    module = ExpandedActionPrefixDynamics(base, expansion).eval()
    state = torch.randn(2, 4, 12)
    action = torch.randn(2, 3, 4)
    task = torch.tensor([0, 6])

    expected = base(state, action, task)
    observed = module(state, action, task)

    torch.testing.assert_close(observed, expected)


def test_expanded_capacity_is_materially_larger_than_parent_dynamics() -> None:
    assert EXPANDED_DYNAMICS_CONFIG["predictor_hidden_dim"] == 4096
    assert EXPANDED_DYNAMICS_CONFIG["predictor_depth"] == 10
    assert EXPANDED_DYNAMICS_CONFIG["prefix_depth"] == 6
    assert EXPANDED_RESIDUAL_SCALE == 0.1


def test_expanded_residual_is_bounded_during_initial_finetuning() -> None:
    config = {
        "action_dim": 4,
        "state_dim": 12,
        "num_slots": 4,
        "num_tasks": 7,
        "max_horizon": 3,
        "prefix_depth": 1,
        "prefix_heads": 3,
        "prefix_mlp_dim": 24,
        "predictor_depth": 1,
        "predictor_hidden_dim": 24,
        "fusion_dim": 24,
        "state_mlp_dim": 24,
        "dropout": 0.0,
    }
    base = ActionPrefixDynamics(**config).eval()
    expansion = ActionPrefixDynamics(**config).eval()
    module = ExpandedActionPrefixDynamics(base, expansion).eval()
    state = torch.randn(2, 4, 12)
    action = torch.randn(2, 3, 4)
    task = torch.tensor([0, 6])
    with torch.no_grad():
        for layer in expansion.residual_predictor.layers:
            layer.modulation[-1].bias[2 * config["state_dim"] :].fill_(0.5)

    expected_base = base(state, action, task)
    expected_expansion = expansion(state, action, task)
    observed = module(state, action, task)

    torch.testing.assert_close(
        observed - expected_base,
        EXPANDED_RESIDUAL_SCALE
        * (expected_expansion - state[:, None]),
    )
