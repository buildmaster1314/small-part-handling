"""Offline control-sensitivity diagnostics for predicted policy anchor tokens.

The probe deliberately keeps the policy carrier (deployment inputs) separate
from the clean anchor target (offline label).  It measures whether a small
token-space error is also small in the action space of the frozen policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import nn

from fcpg.lewm_known_drop_policy import attach_grounded_override
from fcpg.lewm_known_drop_policy_evaluator import deployment_policy_batch


PRIVILEGED_POLICY_KEYS: Final = frozenset(
    {
        "q_true",
        "canonical_target",
        "measurement_valid",
        "sensor_valid",
        "drop_active",
        "drop_schedule",
        "target_token",
        "policy_target_anchor",
    }
)

REQUIRED_DEPLOYMENT_ARRAYS: Final = frozenset(
    {
        "policy_obs",
        "policy_q_nominal",
        "policy_keypoints_xyz",
        "policy_predicted_anchor",
    }
)
REQUIRED_LABEL_ARRAYS: Final = frozenset({"policy_target_anchor"})
COMPACT_VISUAL_ARRAYS: Final = frozenset(
    {
        "policy_sampled_visual",
        "policy_global_visual",
        "policy_image_size",
        "policy_fcpg_q_corrected",
    }
)
BACKBONE_VISUAL_ARRAYS: Final = frozenset(
    {
        "policy_vision_backbone_features",
        "policy_vision_prefix",
        "policy_image_size",
        "policy_fcpg_q_corrected",
    }
)
RGB_VISUAL_ARRAYS: Final = frozenset({"policy_rgb"})


@dataclass(frozen=True)
class ArtifactAudit:
    ready: bool
    missing_deployment: tuple[str, ...]
    missing_labels: tuple[str, ...]
    visual_carrier_error: str | None
    forbidden_deployment: tuple[str, ...]
    observed_deployment: tuple[str, ...]
    observed_labels: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "missing_deployment": list(self.missing_deployment),
            "missing_labels": list(self.missing_labels),
            "visual_carrier_error": self.visual_carrier_error,
            "forbidden_deployment": list(self.forbidden_deployment),
            "observed_deployment": list(self.observed_deployment),
            "observed_labels": list(self.observed_labels),
        }


def audit_policy_probe_arrays(
    deployment_keys: set[str] | frozenset[str],
    label_keys: set[str] | frozenset[str],
) -> ArtifactAudit:
    """Audit whether two caches can support an exact frozen-policy probe."""

    deployment = frozenset(str(value) for value in deployment_keys)
    labels = frozenset(str(value) for value in label_keys)
    missing_deployment = tuple(sorted(REQUIRED_DEPLOYMENT_ARRAYS - deployment))
    missing_labels = tuple(sorted(REQUIRED_LABEL_ARRAYS - labels))
    forbidden = tuple(sorted(PRIVILEGED_POLICY_KEYS.intersection(deployment)))
    has_rgb = RGB_VISUAL_ARRAYS.issubset(deployment)
    has_compact = COMPACT_VISUAL_ARRAYS.issubset(deployment)
    has_backbone = BACKBONE_VISUAL_ARRAYS.issubset(deployment)
    if has_rgb or has_compact or has_backbone:
        visual_error = None
    else:
        rgb_missing = sorted(RGB_VISUAL_ARRAYS - deployment)
        compact_missing = sorted(COMPACT_VISUAL_ARRAYS - deployment)
        backbone_missing = sorted(BACKBONE_VISUAL_ARRAYS - deployment)
        visual_error = (
            "requires policy_rgb, a complete compact carrier, or cached backbone "
            "features; "
            f"rgb missing={rgb_missing}, compact missing={compact_missing}, "
            f"backbone missing={backbone_missing}"
        )
    ready = not (missing_deployment or missing_labels or forbidden or visual_error)
    return ArtifactAudit(
        ready=ready,
        missing_deployment=missing_deployment,
        missing_labels=missing_labels,
        visual_carrier_error=visual_error,
        forbidden_deployment=forbidden,
        observed_deployment=tuple(sorted(deployment)),
        observed_labels=tuple(sorted(labels)),
    )


def audit_policy_probe_npz(
    deployment_path: str | Path,
    label_path: str | Path,
) -> ArtifactAudit:
    """Read only NPZ schemas; array payloads are not loaded by this audit."""

    with np.load(Path(deployment_path), allow_pickle=False) as deployment:
        deployment_keys = set(deployment.files)
    with np.load(Path(label_path), allow_pickle=False) as labels:
        label_keys = set(labels.files)
    return audit_policy_probe_arrays(deployment_keys, label_keys)


def _validate_frozen_policy(policy: nn.Module) -> None:
    if policy.training:
        raise RuntimeError("policy-sensitivity probe requires policy.eval()")
    trainable = [name for name, parameter in policy.named_parameters() if parameter.requires_grad]
    if trainable:
        raise RuntimeError(
            "policy-sensitivity probe requires a frozen policy; trainable parameters: "
            + ", ".join(trainable[:5])
        )


def _slice_batch(batch: Mapping[str, torch.Tensor], index: int) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    batch_size = int(batch["obs"].shape[0])
    for name, value in batch.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"policy batch field {name!r} is not a tensor")
        if value.ndim > 0 and int(value.shape[0]) == batch_size:
            result[name] = value[index : index + 1]
        else:
            result[name] = value
    return result


def _policy_action(
    policy: nn.Module,
    batch: Mapping[str, torch.Tensor],
    anchor: torch.Tensor,
    *,
    action_steps: int,
) -> torch.Tensor:
    selected = attach_grounded_override(batch, anchor)
    output = policy(selected)
    action = output.action[:, :action_steps]
    if action.ndim != 3 or action.shape[-1] <= 0:
        raise ValueError(f"policy action must have shape [B,A,D], got {tuple(action.shape)}")
    if int(action.shape[1]) != action_steps:
        raise ValueError(
            f"policy returned {action.shape[1]} action steps, expected {action_steps}"
        )
    if not bool(torch.isfinite(action).all()):
        raise ValueError("policy returned non-finite action")
    return action


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def probe_policy_token_sensitivity(
    policy: nn.Module,
    deployment_batch: Mapping[str, torch.Tensor],
    predicted_anchor: torch.Tensor,
    target_anchor: torch.Tensor,
    *,
    action_steps: int = 5,
    jacobian_samples: int | None = None,
    singular_tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    """Compare token error with exact and first-order frozen-policy action error.

    ``target_anchor`` is an offline teacher label.  It is supplied separately
    and is used only for the teacher action and Jacobian evaluation; it is
    never inserted into the predicted/student forward pass.
    """

    if action_steps <= 0:
        raise ValueError("action_steps must be positive")
    if singular_tolerance <= 0.0:
        raise ValueError("singular_tolerance must be positive")
    _validate_frozen_policy(policy)
    forbidden = PRIVILEGED_POLICY_KEYS.intersection(deployment_batch)
    if forbidden:
        raise ValueError(
            "deployment policy batch contains privileged/offline fields: "
            + ", ".join(sorted(forbidden))
        )
    batch = deployment_policy_batch(deployment_batch)
    if "obs" not in batch:
        raise KeyError("deployment policy carrier is missing obs")
    batch_size = int(batch["obs"].shape[0])
    expected = (
        batch_size,
        int(getattr(policy, "obs_horizon")),
        int(getattr(policy, "num_keypoints")),
        int(getattr(policy, "model_dim")),
    )
    if tuple(predicted_anchor.shape) != expected or tuple(target_anchor.shape) != expected:
        raise ValueError(
            "predicted and target anchors must both have shape "
            f"{expected}; got {tuple(predicted_anchor.shape)} and {tuple(target_anchor.shape)}"
        )
    if predicted_anchor.device != target_anchor.device:
        raise ValueError("predicted and target anchors must share a device")
    if predicted_anchor.dtype != target_anchor.dtype:
        raise TypeError("predicted and target anchors must share a dtype")
    if not bool(torch.isfinite(predicted_anchor).all()) or not bool(
        torch.isfinite(target_anchor).all()
    ):
        raise ValueError("predicted or target anchors contain non-finite values")

    with torch.no_grad():
        target_action = _policy_action(
            policy, batch, target_anchor, action_steps=action_steps
        )
        predicted_action = _policy_action(
            policy, batch, predicted_anchor, action_steps=action_steps
        )
    token_mse = (predicted_anchor.float() - target_anchor.float()).square().flatten(1).mean(1)
    exact_action_mse = (
        (predicted_action.float() - target_action.float()).square().flatten(1).mean(1)
    )

    count = batch_size if jacobian_samples is None else int(jacobian_samples)
    if not 0 <= count <= batch_size:
        raise ValueError(f"jacobian_samples must lie in [0,{batch_size}]")
    linearized_mse: list[float] = []
    linearization_relative_error: list[float] = []
    singular_values: list[list[float]] = []
    effective_rank: list[int] = []
    for index in range(count):
        sample_batch = _slice_batch(batch, index)
        teacher = target_anchor[index : index + 1].detach().clone().requires_grad_(True)

        def action_from_flat(flat: torch.Tensor) -> torch.Tensor:
            anchor = flat.reshape_as(teacher)
            return _policy_action(
                policy, sample_batch, anchor, action_steps=action_steps
            ).reshape(-1)

        flat_teacher = teacher.reshape(-1)
        jacobian = torch.autograd.functional.jacobian(
            action_from_flat,
            flat_teacher,
            create_graph=False,
            # Some anchor dimensions may be locally action-null.  Their exact
            # Jacobian rows/columns are zero and are part of the diagnostic,
            # not a reason to reject the sample.
            strict=False,
            vectorize=True,
        ).float()
        error = (predicted_anchor[index] - target_anchor[index]).detach().reshape(-1).float()
        linear_delta = jacobian @ error
        exact_delta = (
            predicted_action[index].detach().reshape(-1).float()
            - target_action[index].detach().reshape(-1).float()
        )
        linear_mse = float(linear_delta.square().mean().cpu())
        denominator = float(exact_delta.square().mean().cpu())
        approximation_error = float((linear_delta - exact_delta).square().mean().cpu())
        linearized_mse.append(linear_mse)
        linearization_relative_error.append(
            approximation_error / max(denominator, torch.finfo(torch.float32).eps)
        )
        spectrum = torch.linalg.svdvals(jacobian).detach().cpu().numpy()
        singular_values.append([float(value) for value in spectrum])
        threshold = singular_tolerance * max(float(spectrum[0]), 1.0)
        effective_rank.append(int(np.sum(spectrum > threshold)))

    token_np = token_mse.detach().cpu().numpy().astype(np.float64)
    action_np = exact_action_mse.detach().cpu().numpy().astype(np.float64)
    linear_np = np.asarray(linearized_mse, dtype=np.float64)
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "frozen_policy_anchor_sensitivity_v1",
        "sample_count": batch_size,
        "jacobian_sample_count": count,
        "action_steps": action_steps,
        "token_mse": {
            "mean": float(np.mean(token_np)),
            "per_sample": token_np.tolist(),
        },
        "exact_action_mse": {
            "mean": float(np.mean(action_np)),
            "per_sample": action_np.tolist(),
        },
        "token_vs_exact_action_pearson": _pearson(token_np, action_np),
        "jacobian": {
            "linearized_action_mse_mean": (
                float(np.mean(linear_np)) if linear_np.size else None
            ),
            "linearized_action_mse_per_sample": linearized_mse,
            "linearized_vs_exact_action_pearson": (
                _pearson(linear_np, action_np[:count]) if count else None
            ),
            "linearization_relative_error_per_sample": linearization_relative_error,
            "singular_values_per_sample": singular_values,
            "effective_rank_per_sample": effective_rank,
        },
        "privileged_inputs_entered_student": False,
    }
    return report


__all__ = [
    "ArtifactAudit",
    "audit_policy_probe_arrays",
    "audit_policy_probe_npz",
    "probe_policy_token_sensitivity",
]
