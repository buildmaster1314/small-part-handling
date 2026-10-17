"""Button-only physical diagnostics for the frozen A1 action branches.

The A1 store intentionally contains latent targets but no simulator state. This
module replays already-exposed Button train/development rows and measures the
physical branch spread at the exact same restored snapshots. It does not train
a model, change the A1 gate, or read the independent test split for scoring.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from fcpg.lewm_action_branch_collection import (
    ACTION_DIM,
    CANDIDATE_COUNT,
    FRAME_SKIP,
    restore_branch_snapshot,
)


BUTTON_TASK = "button-press-v3"
BUTTON_TASK_INDEX = 1
PHASES = ("pre_contact", "contact", "post_success")


def classify_button_phase(
    *,
    near_object: bool,
    success: bool,
    button_joint: float = 0.0,
    contact_joint_threshold: float = -0.005,
) -> str:
    """Map the source snapshot to an active-control phase."""

    if bool(success):
        return "post_success"
    if bool(near_object) or float(button_joint) <= float(contact_joint_threshold):
        return "contact"
    return "pre_contact"


@dataclass(frozen=True, slots=True)
class ButtonPhysicalState:
    tcp: np.ndarray
    qpos: np.ndarray
    button_position: np.ndarray
    button_joint: float
    object_to_target: float
    near_object: bool
    success: bool
    touching: bool

    def __post_init__(self) -> None:
        tcp = np.asarray(self.tcp, dtype=np.float32)
        qpos = np.asarray(self.qpos, dtype=np.float32)
        button = np.asarray(self.button_position, dtype=np.float32)
        if tcp.shape != (3,) or button.shape != (3,) or qpos.ndim != 1:
            raise ValueError("Button physical state has invalid vector shapes")
        if qpos.size <= 9:
            raise ValueError("Button environment qpos must contain joint index 9")
        scalars = (self.button_joint, self.object_to_target)
        if not np.isfinite(tcp).all() or not np.isfinite(qpos).all():
            raise ValueError("Button physical state must be finite")
        if not np.isfinite(button).all() or not np.isfinite(scalars).all():
            raise ValueError("Button physical state must be finite")
        object.__setattr__(self, "tcp", np.array(tcp, copy=True))
        object.__setattr__(self, "qpos", np.array(qpos, copy=True))
        object.__setattr__(self, "button_position", np.array(button, copy=True))


def read_button_physical_state(
    env: Any,
    observation: np.ndarray,
) -> ButtonPhysicalState:
    """Read deployment-independent simulator diagnostics at one state."""

    obs = np.asarray(observation, dtype=np.float64)
    _, info = env.evaluate_state(obs, np.zeros(ACTION_DIM, dtype=np.float32))
    qpos = np.asarray(env.data.qpos, dtype=np.float32).copy()
    touching_value = getattr(env, "touching_main_object", False)
    touching = bool(touching_value() if callable(touching_value) else touching_value)
    return ButtonPhysicalState(
        tcp=np.asarray(env.tcp_center, dtype=np.float32),
        qpos=qpos,
        button_position=np.asarray(env._get_pos_objects(), dtype=np.float32),
        button_joint=float(qpos[9]),
        object_to_target=float(info["obj_to_target"]),
        near_object=bool(info["near_object"]),
        success=bool(info["success"]),
        touching=touching,
    )


def replay_button_branches(
    runtime: Any,
    snapshot: Any,
    branch_actions: np.ndarray,
) -> dict[str, np.ndarray]:
    """Execute the exact saved A1 branches and retain physical endpoints."""

    actions = np.asarray(branch_actions, dtype=np.float32)
    if actions.ndim != 3 or actions.shape[0] != CANDIDATE_COUNT:
        raise ValueError("branch_actions must have shape [8,raw_steps,4]")
    if actions.shape[2] != ACTION_DIM or actions.shape[1] % FRAME_SKIP:
        raise ValueError("branch action dimensions violate A1")
    horizon = actions.shape[1] // FRAME_SKIP
    low = np.asarray(runtime.env.action_space.low, dtype=np.float32)
    high = np.asarray(runtime.env.action_space.high, dtype=np.float32)
    if not np.array_equal(actions, np.clip(actions, low, high)):
        raise ValueError("saved A1 branch actions unexpectedly require clipping")

    rows: list[list[ButtonPhysicalState]] = []
    for branch in actions:
        restore_branch_snapshot(runtime.env, snapshot)
        endpoints: list[ButtonPhysicalState] = []
        for raw_index, action in enumerate(branch, start=1):
            observation, _, terminated, truncated, _ = runtime.env.step(action)
            if raw_index % FRAME_SKIP == 0:
                endpoints.append(read_button_physical_state(runtime.env, observation))
            if (terminated or truncated) and raw_index < actions.shape[1]:
                raise RuntimeError("Button branch ended before the A1 horizon")
        if len(endpoints) != horizon:
            raise RuntimeError("Button branch endpoint count is incomplete")
        rows.append(endpoints)

    return {
        "tcp": np.asarray([[state.tcp for state in row] for row in rows], dtype=np.float32),
        "qpos": np.asarray([[state.qpos for state in row] for row in rows], dtype=np.float32),
        "button_position": np.asarray(
            [[state.button_position for state in row] for row in rows],
            dtype=np.float32,
        ),
        "button_joint": np.asarray(
            [[state.button_joint for state in row] for row in rows],
            dtype=np.float32,
        ),
        "object_to_target": np.asarray(
            [[state.object_to_target for state in row] for row in rows],
            dtype=np.float32,
        ),
        "near_object": np.asarray(
            [[state.near_object for state in row] for row in rows],
            dtype=bool,
        ),
        "success": np.asarray(
            [[state.success for state in row] for row in rows],
            dtype=bool,
        ),
        "touching": np.asarray(
            [[state.touching for state in row] for row in rows],
            dtype=bool,
        ),
    }


def flattened_branch_variance(value: np.ndarray) -> np.ndarray:
    """Per-bundle branch variance averaged over horizon and feature axes."""

    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 3 or array.shape[1] != CANDIDATE_COUNT:
        raise ValueError("value must have shape [N,8,...]")
    variance = np.var(array, axis=1, ddof=0)
    return np.mean(variance, axis=tuple(range(1, variance.ndim)), dtype=np.float64)


def endpoint_pairwise_spread(value: np.ndarray) -> np.ndarray:
    """Maximum pairwise Euclidean spread at the final branch endpoint."""

    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 4 or array.shape[1] != CANDIDATE_COUNT:
        raise ValueError("endpoint vector must have shape [N,8,H,D]")
    endpoint = array[:, :, -1, :]
    delta = endpoint[:, :, None, :] - endpoint[:, None, :, :]
    return np.max(np.linalg.norm(delta, axis=-1), axis=(1, 2))


def spearman_rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman correlation with average ranks for tied values."""

    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or x.size < 2:
        raise ValueError("Spearman inputs must be aligned one-dimensional arrays")

    def ranks(value: np.ndarray) -> np.ndarray:
        order = np.argsort(value, kind="mergesort")
        sorted_value = value[order]
        result = np.empty(value.size, dtype=np.float64)
        start = 0
        while start < value.size:
            stop = start + 1
            while stop < value.size and sorted_value[stop] == sorted_value[start]:
                stop += 1
            result[order[start:stop]] = 0.5 * (start + stop - 1)
            start = stop
        return result

    x_rank = ranks(x)
    y_rank = ranks(y)
    if np.std(x_rank) == 0.0 or np.std(y_rank) == 0.0:
        return 0.0
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def summarize_button_diagnostic(
    *,
    phase: Sequence[str],
    latent_branch_variance: np.ndarray,
    tcp: np.ndarray,
    qpos: np.ndarray,
    button_position: np.ndarray,
    button_joint: np.ndarray,
    object_to_target: np.ndarray,
    success: np.ndarray,
    hazard_step: np.ndarray | None = None,
    informative_threshold: float = 0.01,
) -> dict[str, Any]:
    """Produce phase-wise physical/latent diagnostics without a new gate."""

    labels = np.asarray(tuple(phase), dtype=str)
    latent = np.asarray(latent_branch_variance, dtype=np.float64)
    count = labels.shape[0]
    if latent.shape != (count,):
        raise ValueError("latent_branch_variance must have shape [N]")
    arrays = {
        "tcp": np.asarray(tcp),
        "qpos": np.asarray(qpos),
        "button_position": np.asarray(button_position),
        "button_joint": np.asarray(button_joint),
        "object_to_target": np.asarray(object_to_target),
        "success": np.asarray(success),
    }
    if any(value.shape[0] != count for value in arrays.values()):
        raise ValueError("all diagnostic arrays must share the bundle axis")
    step = None if hazard_step is None else np.asarray(hazard_step, dtype=np.int64)
    if step is not None and step.shape != (count,):
        raise ValueError("hazard_step must have shape [N]")

    tcp_variance = flattened_branch_variance(arrays["tcp"])
    qpos_variance = flattened_branch_variance(arrays["qpos"])
    button_position_variance = flattened_branch_variance(arrays["button_position"])
    button_joint_variance = flattened_branch_variance(arrays["button_joint"])
    target_error_variance = flattened_branch_variance(arrays["object_to_target"])
    tcp_endpoint_spread = endpoint_pairwise_spread(arrays["tcp"])
    button_endpoint_spread = endpoint_pairwise_spread(arrays["button_position"])
    success_disagreement = np.ptp(arrays["success"].astype(np.int8), axis=1)
    success_disagreement = np.any(success_disagreement != 0, axis=1)

    def stats(mask: np.ndarray) -> dict[str, Any]:
        def describe(value: np.ndarray) -> dict[str, float]:
            selected = np.asarray(value, dtype=np.float64)[mask]
            return {
                "mean": float(np.mean(selected)),
                "median": float(np.median(selected)),
                "p90": float(np.quantile(selected, 0.9)),
            }

        result = {
            "bundle_count": int(np.count_nonzero(mask)),
            "latent_branch_variance": describe(latent),
            "latent_informative_fraction": float(
                np.mean(latent[mask] >= float(informative_threshold))
            ),
            "tcp_branch_variance_m2": describe(tcp_variance),
            "qpos_branch_variance": describe(qpos_variance),
            "button_position_branch_variance_m2": describe(
                button_position_variance
            ),
            "button_joint_branch_variance": describe(button_joint_variance),
            "target_error_branch_variance_m2": describe(target_error_variance),
            "tcp_endpoint_max_pairwise_m": describe(tcp_endpoint_spread),
            "button_endpoint_max_pairwise_m": describe(button_endpoint_spread),
            "success_disagreement_fraction": float(
                np.mean(success_disagreement[mask])
            ),
        }
        if step is not None:
            result["source_model_step"] = describe(step)
        return result

    per_phase: dict[str, Any] = {}
    for name in PHASES:
        mask = labels == name
        if np.any(mask):
            per_phase[name] = stats(mask)
    return {
        "classification": "disposable_exposed_split_diagnostic",
        "scientific_gate": False,
        "informative_threshold": float(informative_threshold),
        "bundle_count": int(count),
        "latent_physical_spearman": {
            "tcp_branch_variance": spearman_rank_correlation(
                latent, tcp_variance
            ),
            "qpos_branch_variance": spearman_rank_correlation(
                latent, qpos_variance
            ),
            "button_position_branch_variance": spearman_rank_correlation(
                latent, button_position_variance
            ),
        },
        "per_phase": per_phase,
    }
