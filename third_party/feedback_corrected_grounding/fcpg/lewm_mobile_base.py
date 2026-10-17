"""Planar mobile-base motion contracts for action-conditioned LeWM pilots.

The MetaWorld Sawyer is fixed by default.  These helpers translate its root
body and mocap target together, while keeping commanded motion, true motion,
and deployable odometry as separate quantities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


RAW_ARM_ACTION_DIM = 4
RAW_BASE_ACTION_DIM = 2
RAW_STEPS_PER_MODEL_STEP = 5
ARM_PREFIX_DIM = RAW_ARM_ACTION_DIM * RAW_STEPS_PER_MODEL_STEP
BASE_PREFIX_DIM = RAW_BASE_ACTION_DIM * RAW_STEPS_PER_MODEL_STEP
MOBILE_PREFIX_DIM = (RAW_ARM_ACTION_DIM + RAW_BASE_ACTION_DIM) * RAW_STEPS_PER_MODEL_STEP


def _finite_xy(value: np.ndarray | tuple[float, float], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,):
        raise ValueError(f"{label} must have shape [2]")
    if not bool(np.isfinite(result).all()):
        raise ValueError(f"{label} must be finite")
    return result


@dataclass(frozen=True)
class PlanarBaseObservation:
    """One causal base transition and its deployable odometry observation."""

    command_xy: np.ndarray
    true_delta_xy: np.ndarray
    odometry_delta_xy: np.ndarray
    true_position_xy: np.ndarray
    odometry_position_xy: np.ndarray


class PlanarBaseTracker:
    """Accumulate commanded, true, and odometry base motion without hidden RNG."""

    def __init__(
        self,
        *,
        initial_true_xy: np.ndarray | tuple[float, float] = (0.0, 0.0),
        initial_odometry_xy: np.ndarray | tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self._true_xy = _finite_xy(initial_true_xy, "initial_true_xy").copy()
        self._odometry_xy = _finite_xy(
            initial_odometry_xy, "initial_odometry_xy"
        ).copy()

    @property
    def true_position_xy(self) -> np.ndarray:
        return self._true_xy.copy()

    @property
    def odometry_position_xy(self) -> np.ndarray:
        return self._odometry_xy.copy()

    def step(
        self,
        command_xy: np.ndarray | tuple[float, float],
        *,
        slip_xy: np.ndarray | tuple[float, float] = (0.0, 0.0),
        odometry_error_xy: np.ndarray | tuple[float, float] = (0.0, 0.0),
    ) -> PlanarBaseObservation:
        """Advance one raw step using externally supplied deterministic errors."""

        command = _finite_xy(command_xy, "command_xy")
        slip = _finite_xy(slip_xy, "slip_xy")
        odometry_error = _finite_xy(odometry_error_xy, "odometry_error_xy")
        true_delta = command + slip
        odometry_delta = command + odometry_error
        self._true_xy += true_delta
        self._odometry_xy += odometry_delta
        return PlanarBaseObservation(
            command_xy=command.copy(),
            true_delta_xy=true_delta.copy(),
            odometry_delta_xy=odometry_delta.copy(),
            true_position_xy=self._true_xy.copy(),
            odometry_position_xy=self._odometry_xy.copy(),
        )


def apply_planar_base_translation(
    env: Any,
    delta_xy: np.ndarray | tuple[float, float],
    *,
    base_body_name: str = "base",
    mocap_id: int = 0,
) -> None:
    """Translate the Sawyer root and its mocap target by the same world delta."""

    delta = _finite_xy(delta_xy, "delta_xy")
    model = getattr(env, "model", None)
    data = getattr(env, "data", None)
    if model is None or data is None:
        raise TypeError("environment must expose MuJoCo model and data")
    try:
        body_id = int(model.body(base_body_name).id)
    except Exception as error:
        raise ValueError(f"MuJoCo body {base_body_name!r} is unavailable") from error
    mocap_position = np.asarray(data.mocap_pos)
    if not 0 <= int(mocap_id) < int(mocap_position.shape[0]):
        raise ValueError("mocap_id is outside the environment mocap array")
    model.body_pos[body_id, :2] += delta
    data.mocap_pos[int(mocap_id), :2] += delta
    try:
        import mujoco
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise RuntimeError("MuJoCo is required to move the robot base") from error
    mujoco.mj_forward(model, data)


def replace_base_displacement_with_odometry(
    true_keypoints_xyz: np.ndarray,
    *,
    true_base_displacement_xy: np.ndarray | tuple[float, float],
    odometry_base_displacement_xy: np.ndarray | tuple[float, float],
) -> np.ndarray:
    """Build deployable world-frame FK by replacing true base motion with odometry.

    The simulator's keypoints contain exact arm articulation and exact base
    translation.  Subtracting the latter and adding odometry is equivalent to
    computing arm FK in the base frame and transforming it with odometry.
    """

    keypoints = np.asarray(true_keypoints_xyz, dtype=np.float64)
    if keypoints.ndim != 2 or keypoints.shape[1] != 3 or keypoints.shape[0] <= 0:
        raise ValueError("true_keypoints_xyz must have shape [K,3]")
    if not bool(np.isfinite(keypoints).all()):
        raise ValueError("true_keypoints_xyz must be finite")
    true_base = _finite_xy(
        true_base_displacement_xy, "true_base_displacement_xy"
    )
    odometry_base = _finite_xy(
        odometry_base_displacement_xy, "odometry_base_displacement_xy"
    )
    estimated = keypoints.copy()
    estimated[:, :2] += odometry_base - true_base
    return estimated


def pack_mobile_action_prefix(
    arm_prefix: np.ndarray,
    base_prefix: np.ndarray,
) -> np.ndarray:
    """Interleave five raw arm/base actions into each 30D model-step token."""

    arm = np.asarray(arm_prefix)
    base = np.asarray(base_prefix)
    if arm.ndim < 2 or arm.shape[-1] != ARM_PREFIX_DIM:
        raise ValueError(f"arm_prefix must end in [{ARM_PREFIX_DIM}]")
    if base.shape != arm.shape[:-1] + (BASE_PREFIX_DIM,):
        raise ValueError(
            "base_prefix must share leading dimensions with arm_prefix and "
            f"end in [{BASE_PREFIX_DIM}]"
        )
    if not bool(np.isfinite(arm).all()) or not bool(np.isfinite(base).all()):
        raise ValueError("mobile action prefixes must be finite")
    arm_steps = arm.reshape(*arm.shape[:-1], RAW_STEPS_PER_MODEL_STEP, RAW_ARM_ACTION_DIM)
    base_steps = base.reshape(
        *base.shape[:-1], RAW_STEPS_PER_MODEL_STEP, RAW_BASE_ACTION_DIM
    )
    return np.ascontiguousarray(
        np.concatenate([arm_steps, base_steps], axis=-1).reshape(
            *arm.shape[:-1], MOBILE_PREFIX_DIM
        )
    )


def split_mobile_action_prefix(prefix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Invert :func:`pack_mobile_action_prefix` for diagnostics and tests."""

    value = np.asarray(prefix)
    if value.ndim < 2 or value.shape[-1] != MOBILE_PREFIX_DIM:
        raise ValueError(f"prefix must end in [{MOBILE_PREFIX_DIM}]")
    if not bool(np.isfinite(value).all()):
        raise ValueError("mobile action prefix must be finite")
    steps = value.reshape(
        *value.shape[:-1],
        RAW_STEPS_PER_MODEL_STEP,
        RAW_ARM_ACTION_DIM + RAW_BASE_ACTION_DIM,
    )
    arm = steps[..., :RAW_ARM_ACTION_DIM].reshape(*value.shape[:-1], ARM_PREFIX_DIM)
    base = steps[..., RAW_ARM_ACTION_DIM:].reshape(*value.shape[:-1], BASE_PREFIX_DIM)
    return np.ascontiguousarray(arm), np.ascontiguousarray(base)


def pack_mobile_action_prefix_tensor(
    arm_prefix: torch.Tensor,
    base_prefix: torch.Tensor,
) -> torch.Tensor:
    """Torch equivalent of :func:`pack_mobile_action_prefix`."""

    if not torch.is_tensor(arm_prefix) or not torch.is_tensor(base_prefix):
        raise TypeError("mobile action prefixes must be tensors")
    if arm_prefix.ndim < 2 or arm_prefix.shape[-1] != ARM_PREFIX_DIM:
        raise ValueError(f"arm_prefix must end in [{ARM_PREFIX_DIM}]")
    if base_prefix.shape != arm_prefix.shape[:-1] + (BASE_PREFIX_DIM,):
        raise ValueError(
            "base_prefix must share leading dimensions with arm_prefix and "
            f"end in [{BASE_PREFIX_DIM}]"
        )
    if arm_prefix.device != base_prefix.device:
        raise ValueError("mobile action prefixes must share a device")
    if arm_prefix.dtype != base_prefix.dtype:
        raise TypeError("mobile action prefixes must share a dtype")
    if not arm_prefix.is_floating_point() or not base_prefix.is_floating_point():
        raise TypeError("mobile action prefixes must be floating point")
    if not bool(torch.isfinite(arm_prefix).all()) or not bool(
        torch.isfinite(base_prefix).all()
    ):
        raise ValueError("mobile action prefixes must be finite")
    arm_steps = arm_prefix.reshape(
        *arm_prefix.shape[:-1], RAW_STEPS_PER_MODEL_STEP, RAW_ARM_ACTION_DIM
    )
    base_steps = base_prefix.reshape(
        *base_prefix.shape[:-1], RAW_STEPS_PER_MODEL_STEP, RAW_BASE_ACTION_DIM
    )
    return torch.cat([arm_steps, base_steps], dim=-1).reshape(
        *arm_prefix.shape[:-1], MOBILE_PREFIX_DIM
    )


def expand_action_prefix_dynamics_for_mobile_base(
    base_dynamics: torch.nn.Module,
    *,
    base_action_std: np.ndarray | tuple[float, ...],
) -> torch.nn.Module:
    """Add base-action columns while exactly preserving the frozen arm model.

    New base columns in the first action projection are zero initialized.  The
    expanded model therefore reproduces the original model before fine-tuning,
    including for non-zero base commands, while gradients can subsequently
    learn their effect.
    """

    from fcpg.lewm_action_prefix import ActionPrefixDynamics

    if not isinstance(base_dynamics, ActionPrefixDynamics):
        raise TypeError("base_dynamics must be ActionPrefixDynamics")
    if int(base_dynamics.action_dim) != ARM_PREFIX_DIM:
        raise ValueError(
            f"base dynamics action_dim must be {ARM_PREFIX_DIM}, "
            f"got {base_dynamics.action_dim}"
        )
    base_std = np.asarray(base_action_std, dtype=np.float32)
    if base_std.shape != (BASE_PREFIX_DIM,):
        raise ValueError(f"base_action_std must have shape [{BASE_PREFIX_DIM}]")
    if not bool(np.isfinite(base_std).all()) or bool(np.any(base_std <= 0.0)):
        raise ValueError("base_action_std must be finite and positive")

    prefix_layer = base_dynamics.prefix_encoder.layers[0]
    predictor_layer = base_dynamics.residual_predictor.layers[0]
    expanded = ActionPrefixDynamics(
        action_dim=MOBILE_PREFIX_DIM,
        state_dim=int(base_dynamics.state_dim),
        num_slots=int(base_dynamics.num_slots),
        num_tasks=base_dynamics.num_tasks,
        max_horizon=int(base_dynamics.max_horizon),
        prefix_depth=int(base_dynamics.prefix_depth),
        prefix_heads=int(base_dynamics.prefix_heads),
        predictor_depth=int(base_dynamics.predictor_depth),
        dropout=float(base_dynamics.dropout),
        prefix_mlp_dim=int(prefix_layer.linear1.out_features),
        predictor_hidden_dim=int(predictor_layer.mlp[0].out_features),
        fusion_dim=int(base_dynamics.fusion_dim),
        state_mlp_dim=int(base_dynamics.state_mlp_dim),
    ).to(
        device=base_dynamics.action_mean.device,
        dtype=base_dynamics.action_projection[0].weight.dtype,
    )

    source = base_dynamics.state_dict()
    destination = expanded.state_dict()
    excluded = {
        "action_projection.0.weight",
        "action_mean",
        "action_std",
    }
    for name, source_value in source.items():
        if name in excluded:
            continue
        destination_value = destination.get(name)
        if destination_value is None or destination_value.shape != source_value.shape:
            raise ValueError(f"cannot preserve base dynamics parameter {name!r}")
        destination_value.copy_(source_value)

    arm_columns: list[int] = []
    base_columns: list[int] = []
    for raw_step in range(RAW_STEPS_PER_MODEL_STEP):
        first = raw_step * (RAW_ARM_ACTION_DIM + RAW_BASE_ACTION_DIM)
        arm_columns.extend(range(first, first + RAW_ARM_ACTION_DIM))
        base_columns.extend(
            range(
                first + RAW_ARM_ACTION_DIM,
                first + RAW_ARM_ACTION_DIM + RAW_BASE_ACTION_DIM,
            )
        )
    with torch.no_grad():
        expanded.action_projection[0].weight.zero_()
        expanded.action_projection[0].weight[:, arm_columns].copy_(
            base_dynamics.action_projection[0].weight
        )
        mobile_mean = torch.zeros(
            MOBILE_PREFIX_DIM,
            device=expanded.action_mean.device,
            dtype=expanded.action_mean.dtype,
        )
        mobile_std = torch.ones_like(mobile_mean)
        mobile_mean[arm_columns] = base_dynamics.action_mean
        mobile_std[arm_columns] = base_dynamics.action_std
        mobile_std[base_columns] = torch.as_tensor(
            base_std,
            device=mobile_std.device,
            dtype=mobile_std.dtype,
        )
        expanded.set_action_normalization(mobile_mean, mobile_std)
    expanded.train(base_dynamics.training)
    return expanded


__all__ = [
    "ARM_PREFIX_DIM",
    "BASE_PREFIX_DIM",
    "MOBILE_PREFIX_DIM",
    "PlanarBaseObservation",
    "PlanarBaseTracker",
    "apply_planar_base_translation",
    "expand_action_prefix_dynamics_for_mobile_base",
    "pack_mobile_action_prefix",
    "pack_mobile_action_prefix_tensor",
    "replace_base_displacement_with_odometry",
    "split_mobile_action_prefix",
]
