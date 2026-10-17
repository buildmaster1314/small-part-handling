from __future__ import annotations

import numpy as np
import torch

from fcpg.custom_tasks.towel_wipe import SawyerTowelWipeEnvV0
from fcpg.lewm_goal_relative_sidecar import (
    ACTION_DIM,
    GoalRelativeActionSidecar,
    GoalRelativePixelSidecar,
    load_sidecar_checkpoint,
    project_world_goal,
    sidecar_payload,
    towel_goal_world,
)


def _inputs(batch_size: int = 2) -> tuple[torch.Tensor, ...]:
    base = torch.rand(batch_size, 8, 3, 2) * 223.0
    current = torch.rand(batch_size, 3, 2) * 223.0
    goal = torch.rand(batch_size, 2) * 223.0
    task = torch.zeros(batch_size, dtype=torch.long)
    valid = torch.ones(batch_size, 3, dtype=torch.bool)
    return base, current, goal, task, valid


def test_zero_initialized_sidecar_is_exactly_frozen_base() -> None:
    model = GoalRelativePixelSidecar().eval()
    base, current, goal, task, valid = _inputs()

    observed = model(
        base,
        current,
        goal,
        task,
        image_size=(224, 224),
        valid=valid,
    )

    torch.testing.assert_close(observed, base, rtol=0.0, atol=0.0)


def test_sidecar_residual_is_bounded_in_pixel_space() -> None:
    cap = 7.5
    model = GoalRelativePixelSidecar(max_residual_px=cap).eval()
    with torch.no_grad():
        model.head[-1].bias.fill_(100.0)
    base, current, goal, task, valid = _inputs()

    observed = model(
        base,
        current,
        goal,
        task,
        image_size=(224, 224),
        valid=valid,
    )

    assert torch.all(torch.abs(observed - base) <= cap + 1.0e-4)


def test_action_aware_sidecar_is_zero_initialized_and_uses_prefix_shape() -> None:
    model = GoalRelativeActionSidecar().eval()
    base, current, goal, task, valid = _inputs()
    action = torch.randn(base.shape[0], 8, ACTION_DIM)

    observed = model(
        base,
        current,
        goal,
        task,
        future_action=action,
        image_size=(224, 224),
        valid=valid,
    )

    torch.testing.assert_close(observed, base, rtol=0.0, atol=0.0)


def test_action_aware_checkpoint_protocol_round_trip() -> None:
    model = GoalRelativeActionSidecar(hidden_dim=32).eval()
    payload = sidecar_payload(
        model,
        step=2,
        seed=4,
        parent_checkpoint="parent.pt",
        feature_cache="features.npz",
        metrics={},
    )
    assert payload["protocol"] == "goal_relative_action_sidecar_v1"
    restored = load_sidecar_checkpoint(payload).eval()
    base, current, goal, task, valid = _inputs(batch_size=1)
    action = torch.randn(1, 8, ACTION_DIM)
    expected = model(
        base,
        current,
        goal,
        task,
        future_action=action,
        image_size=(224, 224),
        valid=valid,
    )
    observed = restored(
        base,
        current,
        goal,
        task,
        future_action=action,
        image_size=(224, 224),
        valid=valid,
    )
    torch.testing.assert_close(observed, expected)


def test_project_world_goal_uses_camera_from_world_convention() -> None:
    intrinsic = np.asarray(
        [[100.0, 0.0, 20.0], [0.0, 100.0, 30.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    extrinsic = np.eye(4, dtype=np.float32)

    pixel, depth, valid = project_world_goal(
        np.asarray([0.2, -0.1, 2.0], dtype=np.float32), intrinsic, extrinsic
    )

    np.testing.assert_allclose(pixel, np.asarray([30.0, 25.0], dtype=np.float32))
    assert depth == 2.0
    assert valid


def test_towel_goal_mapping_matches_controller_owned_phase_targets() -> None:
    instance = 3
    left, right = SawyerTowelWipeEnvV0.task_instance_endpoints(instance)

    descend = towel_goal_world(instance, "descend_to_grip")
    np.testing.assert_allclose(descend[:2], left[:2])
    assert np.isclose(
        descend[2], SawyerTowelWipeEnvV0.INITIAL_GRASP_LIFT_HEIGHT + 0.03
    )
    np.testing.assert_allclose(towel_goal_world(instance, "wipe_right"), right)
    np.testing.assert_allclose(towel_goal_world(instance, "wipe_left"), left)
    np.testing.assert_allclose(
        towel_goal_world(instance, "lift_after_wipe"),
        np.asarray(
            [0.0, left[1], SawyerTowelWipeEnvV0.FINISH_HEIGHT],
            dtype=np.float32,
        ),
    )


def test_sidecar_checkpoint_round_trip_preserves_predictions() -> None:
    model = GoalRelativePixelSidecar(hidden_dim=32).eval()
    payload = sidecar_payload(
        model,
        step=17,
        seed=23,
        parent_checkpoint="parent.pt",
        feature_cache="features.npz",
        metrics={"development": {"epe": 1.0}},
    )
    restored = load_sidecar_checkpoint(payload).eval()
    base, current, goal, task, valid = _inputs(batch_size=1)

    expected = model(
        base,
        current,
        goal,
        task,
        image_size=(224, 224),
        valid=valid,
    )
    observed = restored(
        base,
        current,
        goal,
        task,
        image_size=(224, 224),
        valid=valid,
    )

    torch.testing.assert_close(observed, expected)
