from __future__ import annotations
import pytest
import torch

from fcpg.lewm_goal_relative_world_sidecar import GoalRelativeWorldSidecar


def _inputs(batch: int = 3):
    return (
        torch.randn(batch, 8, 3, 3),
        torch.randn(batch, 3, 3),
        torch.randn(batch, 3),
        torch.randn(batch, 3),
        torch.tensor([0, 1, 0][:batch], dtype=torch.long),
    )


def test_zero_initialized_sidecar_preserves_frozen_forecast() -> None:
    base, current, goal, velocity, task = _inputs()
    model = GoalRelativeWorldSidecar()
    output = model(base, current, goal, velocity, task)
    torch.testing.assert_close(output, base)


def test_sidecar_only_changes_eef_and_respects_bound() -> None:
    base, current, goal, velocity, task = _inputs()
    model = GoalRelativeWorldSidecar(max_residual_m=0.02)
    with torch.no_grad():
        model.head[-1].bias.fill_(100.0)
    output = model(base, current, goal, velocity, task)
    torch.testing.assert_close(output[:, :, :2], base[:, :, :2])
    residual = output[:, :, 2] - base[:, :, 2]
    assert bool((residual.abs() <= 0.02 + 1.0e-6).all())


def test_sidecar_rejects_invalid_task_index() -> None:
    base, current, goal, velocity, task = _inputs()
    task[0] = 2
    with pytest.raises(ValueError, match="only 0 and 1"):
        GoalRelativeWorldSidecar()(base, current, goal, velocity, task)
