from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from fcpg.policy_token_sensitivity import (
    audit_policy_probe_arrays,
    probe_policy_token_sensitivity,
)


class _LinearAnchorPolicy(nn.Module):
    obs_horizon = 2
    num_keypoints = 1
    model_dim = 2

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "weight",
            torch.tensor(
                [
                    [1.0, 0.0, 2.0, 0.0],
                    [0.0, 1.0, 0.0, -1.0],
                ]
            ),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> SimpleNamespace:
        anchor = batch["grounded_token_override"].reshape(batch["obs"].shape[0], -1)
        action = anchor @ self.weight.T
        return SimpleNamespace(action=action[:, None, :])


def _batch(size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.zeros(size, 2, 3),
        "q_nominal": torch.zeros(size, 2, 1, 2),
        "keypoints_xyz": torch.zeros(size, 2, 1, 3),
        "sampled_visual": torch.zeros(size, 2, 1, 2),
        "global_visual": torch.zeros(size, 2, 2),
        "image_size": torch.tensor([[32, 32]]).expand(size, -1).clone(),
    }


def test_policy_probe_matches_exact_linear_jacobian_and_spectrum() -> None:
    policy = _LinearAnchorPolicy().requires_grad_(False).eval()
    target = torch.zeros(2, 2, 1, 2)
    predicted = target.clone()
    predicted[0, 0, 0, 0] = 1.0
    predicted[1, 1, 0, 1] = 2.0

    report = probe_policy_token_sensitivity(
        policy,
        _batch(),
        predicted,
        target,
        action_steps=1,
        jacobian_samples=2,
    )

    assert report["token_mse"]["per_sample"] == pytest.approx([0.25, 1.0])
    assert report["exact_action_mse"]["per_sample"] == pytest.approx([0.5, 2.0])
    jacobian = report["jacobian"]
    assert jacobian["linearized_action_mse_per_sample"] == pytest.approx([0.5, 2.0])
    assert jacobian["linearization_relative_error_per_sample"] == pytest.approx([0.0, 0.0])
    assert jacobian["effective_rank_per_sample"] == [2, 2]
    expected = sorted([np.sqrt(5.0), np.sqrt(2.0)], reverse=True)
    assert jacobian["singular_values_per_sample"][0] == pytest.approx(expected)
    assert report["privileged_inputs_entered_student"] is False


def test_policy_probe_rejects_privileged_carrier_and_trainable_policy() -> None:
    frozen = _LinearAnchorPolicy().requires_grad_(False).eval()
    target = torch.zeros(1, 2, 1, 2)
    privileged = _batch(1)
    privileged["q_true"] = torch.zeros(1, 2, 1, 2)
    with pytest.raises(ValueError, match="privileged/offline"):
        probe_policy_token_sensitivity(
            frozen, privileged, target, target, action_steps=1
        )

    trainable = _LinearAnchorPolicy()
    trainable.weight = nn.Parameter(trainable.weight.clone())
    trainable.eval()
    with pytest.raises(RuntimeError, match="frozen policy"):
        probe_policy_token_sensitivity(
            trainable, _batch(1), target, target, action_steps=1
        )


def test_artifact_audit_requires_full_policy_carrier_and_separate_target() -> None:
    current_fresh_keys = {
        "history_token",
        "history_action",
        "future_action",
        "history_xyz",
    }
    current_label_keys = {"target_token"}
    audit = audit_policy_probe_arrays(current_fresh_keys, current_label_keys)
    assert audit.ready is False
    assert "policy_predicted_anchor" in audit.missing_deployment
    assert "policy_target_anchor" in audit.missing_labels
    assert audit.visual_carrier_error is not None

    ready = audit_policy_probe_arrays(
        {
            "policy_obs",
            "policy_q_nominal",
            "policy_keypoints_xyz",
            "policy_predicted_anchor",
            "policy_sampled_visual",
            "policy_global_visual",
            "policy_image_size",
            "policy_fcpg_q_corrected",
        },
        {"policy_target_anchor"},
    )
    assert ready.ready is True
    assert ready.as_dict()["forbidden_deployment"] == []
