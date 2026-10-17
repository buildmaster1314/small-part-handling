"""Fail-closed preflight and development metrics for Action-Branch A4.

All statistical reductions keep branch bundles nested in their source physical
episode.  Preflight uses train/development rows only, and development scoring
refuses a cache or prediction panel containing reserved-test rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any
import zlib

import numpy as np

from fcpg.lewm_a4_storage import (
    STATE_SLOT_COUNT,
    validate_a4_ledger_cache_pair,
    validate_a4_proposal_ledger,
    validate_a4_state_cache,
)
from fcpg.lewm_action_branch_a4 import (
    BRANCH_COUNT,
    DEVELOPMENT_TRAINING_SEEDS,
    HORIZON,
    PROTOCOL,
    TASKS,
    evaluate_active_control_eligibility,
    validate_a4_contract,
)


FACTUAL_METHOD = "factual"
NO_ACTION_METHOD = "no_action"
WRONG_ACTION_METHOD = "wrong_action"
HOLD_LAST_METHOD = "hold_last"
LATENT_CONSTANT_VELOCITY_METHOD = "latent_constant_velocity"

PRIMARY_HORIZONS = (6, 7, 8)
WRONG_ACTION_SHIFT_COUNT = BRANCH_COUNT - 1
READOUT_NAMES = (
    "object_pose",
    "goal_relative_progress",
    "contact_or_grasp_state",
    "branch_outcome",
)
CURRENT_STATE_READOUT_NAMES = (
    "object_pose",
    "goal_relative_progress",
    "contact_or_grasp_state",
)
FUTURE_DIAGNOSTIC_READOUT_NAMES = ("branch_outcome",)
EXTERNAL_PREFLIGHT_GATES = (
    "environment_and_expert_rollout",
    "snapshot_restore",
    "zero_severe_penetration",
    "fcpg_anchor_smoke",
)
DATA_PREFLIGHT_GATES = (
    "active_control_support",
    "physical_support",
    "latent_support",
    "task_state_sufficiency",
)
FORMAL_MODE = "formal"
SMOKE_MODE = "smoke"
SMOKE_MAX_EPISODES_PER_TASK_PER_SPLIT = 2
SMOKE_BOOTSTRAP_SAMPLES = 256
RIDGE_ALPHA = 1e-3
NUMERIC_EPSILON = 1e-12
BOOTSTRAP_SEED = 30_407


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _mode(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("mode must be a string")
    if value not in {FORMAL_MODE, SMOKE_MODE}:
        raise ValueError("mode must be 'formal' or 'smoke'")
    return value


def _finite_number(value: object, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _numeric(value: object, *, label: str, allow_bool: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if (array.dtype == np.bool_ and not allow_bool) or not (
        np.issubdtype(array.dtype, np.number) or (allow_bool and array.dtype == np.bool_)
    ):
        raise TypeError(f"{label} must be numeric")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    return array


def _vector(value: object, *, label: str, count: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (count,):
        raise ValueError(f"{label} must have shape ({count},)")
    return array


def _string_vector(value: object, *, label: str, count: int) -> np.ndarray:
    array = _vector(value, label=label, count=count)
    if array.dtype.kind not in {"U", "S"}:
        raise TypeError(f"{label} must contain fixed-width strings")
    result = array.astype(str, copy=False)
    if np.any(result == ""):
        raise ValueError(f"{label} values must be non-empty")
    return result


def _bool_array(value: object, *, label: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if array.dtype.kind != "b":
        raise TypeError(f"{label} must contain booleans")
    return array.astype(bool, copy=False)


def validate_no_reserved_test_rows(
    arrays: Mapping[str, np.ndarray], *, label: str
) -> np.ndarray:
    """Return split labels after refusing any reserved-test row."""

    values = _mapping(arrays, label=label)
    if "split" not in values:
        raise ValueError(f"{label} is missing mandatory split metadata")
    split = np.asarray(values["split"])
    if split.ndim != 1 or split.size == 0:
        raise ValueError(f"{label} split must be a non-empty vector")
    if split.dtype.kind not in {"U", "S"}:
        raise TypeError(f"{label} split must contain fixed-width strings")
    result = split.astype(str, copy=False)
    if np.any(result == "test"):
        raise PermissionError(
            f"{label} contains reserved test rows; A4 preflight/development is not authorized to inspect them"
        )
    unknown = sorted(set(result.tolist()) - {"train", "development"})
    if unknown:
        raise ValueError(f"{label} contains unsupported split labels: {unknown}")
    return result


def validate_a4_scoring_contract(
    contract: Mapping[str, Any], *, phase: str
) -> Mapping[str, Any]:
    """Bind a scorer to the checked-in A4 contract without relaxing it."""

    root = _mapping(contract, label="A4 contract")
    if validate_a4_contract(root) != TASKS:
        raise ValueError("A4 scorer task panel drifted")
    if phase == "preflight":
        task_state = _mapping(
            root.get("task_state_sufficiency_gate"), label="task-state Gate"
        )
        if task_state.get("fit_split") != "train":
            raise ValueError("A4 ridge fit must remain train-only")
        if task_state.get("evaluation_split") != "development":
            raise ValueError("A4 ridge evaluation must remain development-only")
        if task_state.get("reserved_test_access_allowed") is not False:
            raise ValueError("A4 preflight must keep reserved test sealed")
        if tuple(task_state.get("required_readouts", ())) != READOUT_NAMES:
            raise ValueError("A4 required readout panel drifted")
        if tuple(task_state.get("current_state_gate_readouts", ())) != (
            CURRENT_STATE_READOUT_NAMES
        ):
            raise ValueError("A4 current-state Gate readouts drifted")
        if tuple(task_state.get("future_diagnostic_readouts", ())) != (
            FUTURE_DIAGNOSTIC_READOUT_NAMES
        ):
            raise ValueError("A4 future diagnostic readouts drifted")
        return task_state
    if phase == "development":
        gate = _mapping(root.get("development_gate"), label="development Gate")
        if gate.get("evaluation_split") != "development":
            raise ValueError("A4 development scorer may use development only")
        return gate
    raise ValueError("phase must be 'preflight' or 'development'")


def _cache_identity(cache: Mapping[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    count = int(np.asarray(cache["proposal_id"]).shape[0])
    return (
        np.asarray(cache["proposal_id"]).astype(str),
        np.asarray(cache["task_class_index"], dtype=np.int64),
        np.asarray(cache["task_name"]).astype(str),
        np.asarray(cache["source_episode_id"]).astype(str),
        _string_vector(cache["split"], label="cache split", count=count),
    )


def _aligned_measurements(
    measurements: Mapping[str, np.ndarray],
    cache: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    values = _mapping(measurements, label="A4 preflight labels")
    for name, value in values.items():
        if not isinstance(name, str):
            raise TypeError("A4 preflight label names must be strings")
        if not isinstance(value, np.ndarray):
            raise TypeError(f"A4 preflight label {name!r} must already be an ndarray")
    cache_id, cache_task, cache_name, cache_episode, cache_split = _cache_identity(cache)
    count = cache_id.size
    required_identity = {"proposal_id", "split"}
    missing = sorted(required_identity - set(values))
    if missing:
        raise ValueError(f"A4 preflight labels are missing identity fields: {missing}")
    split = validate_no_reserved_test_rows(values, label="A4 preflight labels")
    label_id = _string_vector(
        values["proposal_id"], label="label proposal_id", count=split.size
    )
    if len(set(label_id.tolist())) != label_id.size:
        raise ValueError("A4 preflight label proposal_id values must be unique")
    positions = {value: index for index, value in enumerate(label_id.tolist())}
    if set(positions) != set(cache_id.tolist()) or label_id.size != count:
        raise ValueError("A4 preflight labels must cover every and only cache proposal_id")
    order = np.asarray([positions[value] for value in cache_id.tolist()], dtype=np.int64)
    result: dict[str, np.ndarray] = {}
    for name, value in values.items():
        if value.ndim == 0 or value.shape[0] != count:
            raise ValueError(f"A4 preflight label {name!r} must have leading axis N={count}")
        result[name] = np.asarray(value)[order]
    if not np.array_equal(result["proposal_id"].astype(str), cache_id):
        raise RuntimeError("A4 preflight label alignment failed")
    if not np.array_equal(result["split"].astype(str), cache_split):
        raise ValueError("A4 preflight label split differs from the state cache")
    optional_identity = {
        "task_class_index": cache_task,
        "task_name": cache_name,
        "source_episode_id": cache_episode,
    }
    for name, expected in optional_identity.items():
        if name in result and not np.array_equal(result[name].astype(expected.dtype), expected):
            raise ValueError(f"A4 preflight label {name} differs from the state cache")
    return result


def _alias(
    arrays: Mapping[str, np.ndarray], aliases: Sequence[str], *, role: str
) -> np.ndarray:
    present = [name for name in aliases if name in arrays]
    if not present:
        raise ValueError(f"A4 preflight labels are missing mandatory {role}")
    reference = np.asarray(arrays[present[0]])
    for name in present[1:]:
        if not np.array_equal(reference, np.asarray(arrays[name])):
            raise ValueError(f"conflicting aliases were supplied for {role}")
    return reference


def _physical_spread(labels: Mapping[str, np.ndarray], *, count: int) -> np.ndarray:
    direct = [
        name
        for name in ("future_physical_spread", "physical_spread")
        if name in labels
    ]
    if direct:
        value = _numeric(labels[direct[0]], label=direct[0])
        if value.shape != (count,):
            raise ValueError(f"{direct[0]} must have shape ({count},)")
        if np.any(value < 0.0):
            raise ValueError(f"{direct[0]} must be non-negative")
        return value.astype(np.float64, copy=False)

    states = [name for name in ("physical_state", "tcp", "object_position") if name in labels]
    if not states:
        raise ValueError(
            "A4 preflight labels are missing mandatory physical spread/state labels"
        )
    spreads: list[np.ndarray] = []
    for name in states:
        value = _numeric(labels[name], label=name)
        if value.ndim < 3 or value.shape[:2] != (count, BRANCH_COUNT):
            raise ValueError(f"{name} must have leading shape [N,{BRANCH_COUNT},...]")
        flat = value.astype(np.float64, copy=False).reshape(count, BRANCH_COUNT, -1)
        delta = flat[:, :, None] - flat[:, None, :]
        spreads.append(np.max(np.linalg.norm(delta, axis=-1), axis=(1, 2)))
    return np.max(np.stack(spreads, axis=1), axis=1)


def _branch_validity(labels: Mapping[str, np.ndarray], *, count: int) -> np.ndarray:
    if "branch_valid" in labels or "branch_is_valid" in labels:
        value = _alias(
            labels,
            ("branch_valid", "branch_is_valid"),
            role="branch validity labels",
        )
        array = np.asarray(value)
    elif "invalid_branch" in labels:
        invalid = np.asarray(labels["invalid_branch"])
        if invalid.dtype.kind != "b":
            raise TypeError("invalid_branch must contain booleans")
        array = ~invalid
    else:
        raise ValueError("A4 preflight labels are missing mandatory branch validity labels")
    if array.dtype.kind != "b":
        raise TypeError("branch validity labels must contain booleans")
    if array.ndim not in {2, 3} or array.shape[:2] != (count, BRANCH_COUNT):
        raise ValueError("branch validity must have shape [N,8] or [N,8,H]")
    if array.ndim == 3 and array.shape[2] != HORIZON:
        raise ValueError("per-horizon branch validity must have shape [N,8,8]")
    return array.astype(bool, copy=False)


def _latent_spread(cache: Mapping[str, np.ndarray]) -> np.ndarray:
    target = np.asarray(cache["target_state"], dtype=np.float64)
    selected = target[:, :, np.asarray(PRIMARY_HORIZONS) - 1]
    centered = selected - np.mean(selected, axis=1, keepdims=True, dtype=np.float64)
    return np.mean(np.square(centered), axis=(1, 2, 3, 4), dtype=np.float64)


def _episode_rows(
    task: np.ndarray, episode: np.ndarray, selected: np.ndarray
) -> dict[int, dict[str, np.ndarray]]:
    result: dict[int, dict[str, np.ndarray]] = {}
    for task_index in range(len(TASKS)):
        task_rows = selected & (task == task_index)
        identities = np.unique(episode[task_rows])
        result[task_index] = {
            str(identity): np.flatnonzero(task_rows & (episode == identity))
            for identity in identities
        }
    return result


def _coverage_counts(
    task: np.ndarray, episode: np.ndarray, split: np.ndarray
) -> dict[str, dict[str, int]]:
    return {
        TASKS[index]: {
            name: int(np.unique(episode[(task == index) & (split == name)]).size)
            for name in ("train", "development")
        }
        for index in range(len(TASKS))
    }


def _validate_mode_coverage(
    task: np.ndarray,
    episode: np.ndarray,
    split: np.ndarray,
    *,
    mode: str,
    label: str,
    formal_train_minimum: int,
) -> dict[str, dict[str, int]]:
    counts = _coverage_counts(task, episode, split)
    for task_name, row in counts.items():
        if mode == SMOKE_MODE:
            for split_name, count in row.items():
                if count < 1 or count > SMOKE_MAX_EPISODES_PER_TASK_PER_SPLIT:
                    raise ValueError(
                        f"{label} smoke requires 1..{SMOKE_MAX_EPISODES_PER_TASK_PER_SPLIT} "
                        f"{split_name} episodes for {task_name}, got {count}"
                    )
        else:
            if row["train"] < formal_train_minimum or row["train"] > 24:
                raise ValueError(
                    f"{label} formal train coverage for {task_name} must be "
                    f"{formal_train_minimum}..24 episodes, got {row['train']}"
                )
            if row["development"] < 1 or row["development"] > 12:
                raise ValueError(
                    f"{label} formal development coverage for {task_name} must be 1..12 episodes"
                )
    return counts


def _recompute_ledger_eligibility(
    ledger: Mapping[str, np.ndarray], contract: Mapping[str, Any]
) -> None:
    count = int(np.asarray(ledger["proposal_id"]).shape[0])
    input_names = (
        "proposal",
        "complete_history",
        "success",
        "ever_success_before",
        "active_stage",
        "expert_xyz_norm",
        "recent_action_rms",
        "workspace_and_joint_margin",
        "unexpected_collision",
        "remaining_raw_steps",
    )
    observed = np.asarray(ledger["eligible"], dtype=bool)
    recomputed = np.empty(count, dtype=bool)
    for index in range(count):
        inputs = {name: np.asarray(ledger[name])[index].item() for name in input_names}
        recomputed[index] = evaluate_active_control_eligibility(
            inputs, contract=contract
        ).eligible
    if not np.array_equal(observed, recomputed):
        raise ValueError("proposal ledger eligibility differs from the frozen A4 rule")


def summarize_a4_support(
    ledger: Mapping[str, np.ndarray],
    cache: Mapping[str, np.ndarray],
    measurements: Mapping[str, np.ndarray],
    *,
    contract: Mapping[str, Any],
    mode: str = FORMAL_MODE,
) -> dict[str, Any]:
    """Score active-control, physical, and latent collection support."""

    selected_mode = _mode(mode)
    validate_a4_scoring_contract(contract, phase="preflight")
    validate_a4_ledger_cache_pair(ledger, cache)
    ledger_split = validate_no_reserved_test_rows(ledger, label="A4 proposal ledger")
    cache_split = validate_no_reserved_test_rows(cache, label="A4 state cache")
    _recompute_ledger_eligibility(ledger, contract)
    labels = _aligned_measurements(measurements, cache)

    cache_id, cache_task, _cache_name, cache_episode, _ = _cache_identity(cache)
    count = cache_id.size
    validity = _branch_validity(labels, count=count)
    physical_spread = _physical_spread(labels, count=count)
    latent_spread = _latent_spread(cache)
    source_success = _alias(
        labels,
        ("source_expert_success", "source_episode_success"),
        role="source expert success labels",
    )
    source_success = _bool_array(
        source_success, label="source_expert_success", shape=(count,)
    )

    ledger_task = np.asarray(ledger["task_class_index"], dtype=np.int64)
    ledger_episode = np.asarray(ledger["source_episode_id"]).astype(str)
    ledger_counts = _validate_mode_coverage(
        ledger_task,
        ledger_episode,
        ledger_split,
        mode=selected_mode,
        label="proposal ledger",
        formal_train_minimum=24,
    )
    cache_counts = _validate_mode_coverage(
        cache_task,
        cache_episode,
        cache_split,
        mode=selected_mode,
        label="accepted state cache",
        formal_train_minimum=int(
            contract["support_gate"]["minimum_active_bundle_episode_count_per_task"]
        ),
    )

    eligible = np.asarray(ledger["eligible"], dtype=bool)
    accepted_post_success = eligible & (
        np.asarray(ledger["success"], dtype=bool)
        | np.asarray(ledger["ever_success_before"], dtype=bool)
    )
    eligibility = contract["eligibility"]
    accepted_near_zero = eligible & (
        np.asarray(ledger["expert_xyz_norm"], dtype=np.float64)
        < float(eligibility["minimum_expert_xyz_norm"])
    )
    accepted_near_zero |= eligible & (
        np.asarray(ledger["recent_action_rms"], dtype=np.float64)
        < float(eligibility["minimum_recent_action_rms"])
    )
    accepted_count = int(np.count_nonzero(eligible))
    post_fraction = (
        float(np.count_nonzero(accepted_post_success) / accepted_count)
        if accepted_count
        else 1.0
    )
    near_zero_fraction = (
        float(np.count_nonzero(accepted_near_zero) / accepted_count)
        if accepted_count
        else 1.0
    )
    invalid_fraction = float(np.mean(~validity))

    train = cache_split == "train"
    episode_rows = _episode_rows(cache_task, cache_episode, train)
    minimum_episodes = (
        1
        if selected_mode == SMOKE_MODE
        else int(contract["support_gate"]["minimum_active_bundle_episode_count_per_task"])
    )
    per_task: dict[str, Any] = {}
    for task_index, task_name in enumerate(TASKS):
        rows_by_episode = episode_rows[task_index]
        physical_fractions = np.asarray(
            [np.mean(physical_spread[rows] > 0.0) for rows in rows_by_episode.values()],
            dtype=np.float64,
        )
        latent_fractions = np.asarray(
            [np.mean(latent_spread[rows] > 0.0) for rows in rows_by_episode.values()],
            dtype=np.float64,
        )
        success_fractions = np.asarray(
            [np.mean(source_success[rows]) for rows in rows_by_episode.values()],
            dtype=np.float64,
        )
        active_count = len(rows_by_episode)
        physical_count = int(np.count_nonzero(physical_fractions > 0.0))
        latent_count = int(np.count_nonzero(latent_fractions > 0.0))
        task_rows = train & (cache_task == task_index)
        per_task[task_name] = {
            "ledger_episode_count": ledger_counts[task_name]["train"],
            "active_bundle_episode_count": active_count,
            "bundle_count": int(np.count_nonzero(task_rows)),
            "physical_support_episode_count": physical_count,
            "latent_support_episode_count": latent_count,
            "physical_episode_macro_bundle_fraction": float(np.mean(physical_fractions)),
            "latent_episode_macro_bundle_fraction": float(np.mean(latent_fractions)),
            "median_physical_spread": float(np.median(physical_spread[task_rows])),
            "median_latent_branch_variance": float(np.median(latent_spread[task_rows])),
            "source_expert_success_episode_macro": float(np.mean(success_fractions)),
            "active_control_passed": active_count >= minimum_episodes,
            "physical_support_passed": physical_count >= minimum_episodes,
            "latent_support_passed": latent_count >= minimum_episodes,
        }

    gate = contract["support_gate"]
    active_passed = all(row["active_control_passed"] for row in per_task.values())
    physical_passed = all(row["physical_support_passed"] for row in per_task.values())
    physical_passed &= invalid_fraction <= float(gate["maximum_invalid_branch_fraction"])
    latent_passed = all(row["latent_support_passed"] for row in per_task.values())
    active_passed &= post_fraction <= float(gate["maximum_accepted_post_success_fraction"])
    active_passed &= near_zero_fraction <= float(gate["maximum_accepted_near_zero_action_fraction"])
    passed = active_passed and physical_passed and latent_passed
    return {
        "protocol": PROTOCOL,
        "stage": f"a4_{selected_mode}_collection_support",
        "mode": selected_mode,
        "decision": "Support-Go" if passed else "Support-No-Go",
        "passed": bool(passed),
        "input_contains_test_rows": False,
        "aggregation": "task_then_source_physical_episode",
        "future_or_outcome_row_filtering_used": False,
        "minimum_support_episode_count_per_task": minimum_episodes,
        "accepted_proposal_count": accepted_count,
        "accepted_post_success_fraction": post_fraction,
        "accepted_near_zero_action_fraction": near_zero_fraction,
        "invalid_branch_fraction": invalid_fraction,
        "active_control_support_passed": bool(active_passed),
        "physical_support_passed": bool(physical_passed),
        "latent_support_passed": bool(latent_passed),
        "coverage": {"ledger": ledger_counts, "cache": cache_counts},
        "per_task": per_task,
    }


def _flatten_feature(value: np.ndarray, *, label: str, count: int) -> np.ndarray:
    array = _numeric(value, label=label, allow_bool=True)
    if array.ndim < 2 or array.shape[0] != count or np.prod(array.shape[1:]) == 0:
        raise ValueError(f"{label} must have non-empty shape [N,...]")
    return array.astype(np.float64, copy=False).reshape(count, -1)


def _task_onehot(task: np.ndarray) -> np.ndarray:
    result = np.zeros((task.size, len(TASKS)), dtype=np.float64)
    result[np.arange(task.size), task] = 1.0
    return result


def _episode_equal_weights(task: np.ndarray, episode: np.ndarray) -> np.ndarray:
    weights = np.zeros(task.size, dtype=np.float64)
    for task_index in range(len(TASKS)):
        selected = task == task_index
        identities = np.unique(episode[selected])
        if identities.size == 0:
            raise ValueError(f"{TASKS[task_index]} has no rows for ridge fitting")
        for identity in identities:
            rows = selected & (episode == identity)
            weights[rows] = 1.0 / (
                len(TASKS) * identities.size * np.count_nonzero(rows)
            )
    if not np.isclose(np.sum(weights), 1.0, rtol=1e-10, atol=1e-12):
        raise RuntimeError("episode-equal ridge weights do not sum to one")
    return weights


def closed_form_ridge_predict(
    train_feature: np.ndarray,
    train_target: np.ndarray,
    evaluation_feature: np.ndarray,
    *,
    sample_weight: np.ndarray,
    alpha: float = RIDGE_ALPHA,
) -> np.ndarray:
    """Fit a weighted ridge readout in closed form and predict evaluation rows."""

    x = _numeric(train_feature, label="ridge train feature").astype(np.float64, copy=False)
    y = _numeric(train_target, label="ridge train target").astype(np.float64, copy=False)
    evaluation = _numeric(
        evaluation_feature, label="ridge evaluation feature"
    ).astype(np.float64, copy=False)
    if x.ndim != 2 or y.ndim != 2 or evaluation.ndim != 2:
        raise ValueError("ridge features and targets must be matrices")
    if x.shape[0] == 0 or x.shape[0] != y.shape[0] or evaluation.shape[1] != x.shape[1]:
        raise ValueError("ridge train/evaluation matrix axes are inconsistent")
    weights = _numeric(sample_weight, label="ridge sample_weight").astype(
        np.float64, copy=False
    )
    if weights.shape != (x.shape[0],) or np.any(weights <= 0.0):
        raise ValueError("ridge sample_weight must be strictly positive with shape [N]")
    weights = weights / np.sum(weights)
    penalty = _finite_number(alpha, label="ridge alpha", minimum=NUMERIC_EPSILON)

    x_mean = np.sum(weights[:, None] * x, axis=0)
    x_variance = np.sum(weights[:, None] * np.square(x - x_mean), axis=0)
    keep = x_variance > NUMERIC_EPSILON
    y_mean = np.sum(weights[:, None] * y, axis=0)
    if not np.any(keep):
        return np.broadcast_to(y_mean, (evaluation.shape[0], y.shape[1])).copy()
    scale = np.sqrt(x_variance[keep])
    train = (x[:, keep] - x_mean[keep]) / scale
    test = (evaluation[:, keep] - x_mean[keep]) / scale
    root_weight = np.sqrt(weights)[:, None]
    weighted_x = train * root_weight
    weighted_y = (y - y_mean) * root_weight
    if weighted_x.shape[1] <= weighted_x.shape[0]:
        gram = weighted_x.T @ weighted_x
        gram.flat[:: gram.shape[0] + 1] += penalty
        coefficient = np.linalg.solve(gram, weighted_x.T @ weighted_y)
    else:
        gram = weighted_x @ weighted_x.T
        gram.flat[:: gram.shape[0] + 1] += penalty
        coefficient = weighted_x.T @ np.linalg.solve(gram, weighted_y)
    prediction = test @ coefficient + y_mean
    if not np.isfinite(prediction).all():
        raise ValueError("closed-form ridge produced non-finite predictions")
    return prediction


@dataclass(frozen=True)
class _EpisodeReduction:
    task_values: Mapping[str, float]
    episode_values: Mapping[str, np.ndarray]


def _mean_readout_reductions(
    reductions: Mapping[str, _EpisodeReduction],
    names: Sequence[str],
) -> _EpisodeReduction:
    """Average standardized readout losses without mixing future diagnostics."""

    selected = tuple(str(name) for name in names)
    if not selected or any(name not in reductions for name in selected):
        raise ValueError("readout reduction names must be non-empty and available")
    task_values: dict[str, float] = {}
    episode_values: dict[str, np.ndarray] = {}
    for task_name in TASKS:
        values = [
            np.asarray(reductions[name].episode_values[task_name], dtype=np.float64)
            for name in selected
        ]
        if any(value.shape != values[0].shape for value in values[1:]):
            raise ValueError("readout episode reductions do not align")
        reduced = np.mean(np.stack(values, axis=0), axis=0, dtype=np.float64)
        episode_values[task_name] = reduced
        task_values[task_name] = float(np.mean(reduced))
    return _EpisodeReduction(task_values=task_values, episode_values=episode_values)


def _reduce_episode_losses(
    losses: np.ndarray,
    *,
    task: np.ndarray,
    episode: np.ndarray,
) -> _EpisodeReduction:
    values = _numeric(losses, label="episode losses").astype(np.float64, copy=False)
    if values.shape != (task.size,) or np.any(values < 0.0):
        raise ValueError("episode losses must be a non-negative vector with shape [N]")
    task_values: dict[str, float] = {}
    episode_values: dict[str, np.ndarray] = {}
    for task_index, task_name in enumerate(TASKS):
        selected = task == task_index
        identities = np.unique(episode[selected])
        if identities.size == 0:
            raise ValueError(f"{task_name} has no evaluation physical episodes")
        reduced = np.asarray(
            [np.mean(values[selected & (episode == identity)]) for identity in identities],
            dtype=np.float64,
        )
        episode_values[task_name] = reduced
        task_values[task_name] = float(np.mean(reduced))
    return _EpisodeReduction(task_values=task_values, episode_values=episode_values)


def _relative_improvement(factual: float, baseline: float, *, label: str) -> float:
    if factual < 0.0 or baseline < 0.0:
        raise ValueError(f"{label} losses must be non-negative")
    if baseline <= NUMERIC_EPSILON:
        if factual <= NUMERIC_EPSILON:
            return 0.0
        raise ValueError(f"{label} baseline MSE is zero while comparison MSE is positive")
    value = 1.0 - factual / baseline
    if not math.isfinite(value):
        raise ValueError(f"{label} relative improvement is non-finite")
    return float(value)


def _episode_comparison(
    comparison: _EpisodeReduction,
    baseline: _EpisodeReduction,
    *,
    comparison_name: str,
    baseline_name: str,
) -> dict[str, Any]:
    per_task = {
        task_name: {
            "comparison_mse": comparison.task_values[task_name],
            "baseline_mse": baseline.task_values[task_name],
            "relative_improvement": _relative_improvement(
                comparison.task_values[task_name],
                baseline.task_values[task_name],
                label=f"{task_name}/{comparison_name}/{baseline_name}",
            ),
        }
        for task_name in TASKS
    }
    effect = float(
        np.mean([row["relative_improvement"] for row in per_task.values()])
    )
    return {
        "comparison_method": comparison_name,
        "baseline_method": baseline_name,
        "orientation": "positive_means_comparison_has_lower_mse",
        "relative_improvement": effect,
        "positive_task_count": sum(
            row["relative_improvement"] > 0.0 for row in per_task.values()
        ),
        "nonnegative_task_count": sum(
            row["relative_improvement"] >= 0.0 for row in per_task.values()
        ),
        "per_task": per_task,
    }


def _bootstrap_episode_difference(
    comparison: _EpisodeReduction,
    baseline: _EpisodeReduction,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    count = _positive_integer(samples, label="bootstrap samples")
    rng = np.random.default_rng(seed)
    task_draws: list[np.ndarray] = []
    for task_name in TASKS:
        comparison_values = comparison.episode_values[task_name]
        baseline_values = baseline.episode_values[task_name]
        if comparison_values.shape != baseline_values.shape:
            raise ValueError("paired bootstrap episode axes differ")
        draws = rng.integers(
            0,
            comparison_values.size,
            size=(count, comparison_values.size),
        )
        task_draws.append(
            np.mean(
                baseline_values[draws] - comparison_values[draws],
                axis=1,
                dtype=np.float64,
            )
        )
    distribution = np.mean(np.stack(task_draws, axis=1), axis=1, dtype=np.float64)
    estimate = float(
        np.mean(
            [
                baseline.task_values[name] - comparison.task_values[name]
                for name in TASKS
            ]
        )
    )
    lower, upper = np.quantile(distribution, (0.025, 0.975))
    return {
        "metric": "paired_task_macro_mse_difference",
        "orientation": "positive_means_comparison_has_lower_mse",
        "estimate": estimate,
        "confidence": 0.95,
        "lower": float(lower),
        "upper": float(upper),
        "samples": count,
        "unit": "source_physical_episode_within_task",
        "seed": int(seed),
    }


def _readout_targets(
    labels: Mapping[str, np.ndarray], *, count: int
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in READOUT_NAMES:
        if name not in labels:
            raise ValueError(f"A4 preflight labels are missing mandatory readout {name!r}")
        result[name] = _flatten_feature(labels[name], label=name, count=count)
    return result


def _standardize_readout_targets(
    targets: Mapping[str, np.ndarray],
    train: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, slice], dict[str, int]]:
    train_parts: list[np.ndarray] = []
    evaluation_parts: list[np.ndarray] = []
    slices: dict[str, slice] = {}
    floored: dict[str, int] = {}
    offset = 0
    for name in READOUT_NAMES:
        value = targets[name]
        train_value = value[train]
        mean = np.sum(weights[:, None] * train_value, axis=0)
        variance = np.sum(weights[:, None] * np.square(train_value - mean), axis=0)
        scale = np.sqrt(np.maximum(variance, 0.0))
        floor = scale <= NUMERIC_EPSILON
        scale = np.where(floor, 1.0, scale)
        standardized = (value - mean) / scale
        width = value.shape[1]
        slices[name] = slice(offset, offset + width)
        offset += width
        floored[name] = int(np.count_nonzero(floor))
        train_parts.append(standardized[train])
        evaluation_parts.append(standardized[~train])
    return (
        np.concatenate(train_parts, axis=1),
        np.concatenate(evaluation_parts, axis=1),
        slices,
        floored,
    )


def _ridge_candidate_report(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    slices: Mapping[str, slice],
    task: np.ndarray,
    episode: np.ndarray,
) -> tuple[dict[str, Any], _EpisodeReduction, dict[str, _EpisodeReduction]]:
    per_readout_reduction: dict[str, _EpisodeReduction] = {}
    per_readout: dict[str, Any] = {}
    row_losses: list[np.ndarray] = []
    for name in READOUT_NAMES:
        selected = slices[name]
        loss = np.mean(
            np.square(prediction[:, selected] - target[:, selected]),
            axis=1,
            dtype=np.float64,
        )
        reduction = _reduce_episode_losses(loss, task=task, episode=episode)
        per_readout_reduction[name] = reduction
        row_losses.append(loss)
        per_readout[name] = {
            "task_macro_mse": float(np.mean(list(reduction.task_values.values()))),
            "per_task": {
                task_name: {
                    "mse": reduction.task_values[task_name],
                    "physical_episode_count": int(
                        reduction.episode_values[task_name].size
                    ),
                }
                for task_name in TASKS
            },
        }
    aggregate_loss = np.mean(np.stack(row_losses, axis=1), axis=1, dtype=np.float64)
    aggregate = _reduce_episode_losses(aggregate_loss, task=task, episode=episode)
    return (
        {
            "task_macro_mse": float(np.mean(list(aggregate.task_values.values()))),
            "per_task": {
                task_name: {
                    "mse": aggregate.task_values[task_name],
                    "physical_episode_count": int(
                        aggregate.episode_values[task_name].size
                    ),
                }
                for task_name in TASKS
            },
            "per_readout": per_readout,
        },
        aggregate,
        per_readout_reduction,
    )


def summarize_a4_task_state_sufficiency(
    cache: Mapping[str, np.ndarray],
    measurements: Mapping[str, np.ndarray],
    *,
    contract: Mapping[str, Any],
    mode: str = FORMAL_MODE,
    ridge_alpha: float = RIDGE_ALPHA,
) -> dict[str, Any]:
    """Compare anchor-only and summary-plus-anchor closed-form readouts."""

    selected_mode = _mode(mode)
    gate = validate_a4_scoring_contract(contract, phase="preflight")
    validate_a4_state_cache(cache)
    split = validate_no_reserved_test_rows(cache, label="A4 state cache")
    labels = _aligned_measurements(measurements, cache)
    cache_id, task, _task_name, episode, _ = _cache_identity(cache)
    count = cache_id.size
    _validate_mode_coverage(
        task,
        episode,
        split,
        mode=selected_mode,
        label="task-state readout cache",
        formal_train_minimum=int(
            contract["support_gate"]["minimum_active_bundle_episode_count_per_task"]
        ),
    )

    proprio = _alias(labels, ("proprio", "proprioception"), role="proprio labels")
    validity = _alias(
        labels,
        ("anchor_validity", "anchor_valid"),
        role="anchor validity labels",
    )
    shared = np.concatenate(
        [
            _task_onehot(task),
            _flatten_feature(proprio, label="proprio", count=count),
            _flatten_feature(validity, label="anchor_validity", count=count),
            _flatten_feature(cache["history_action"], label="history_action", count=count),
            _flatten_feature(cache["future_action"], label="future_action", count=count),
        ],
        axis=1,
    )
    history = np.asarray(cache["history_state"], dtype=np.float64)
    anchor = history[:, :, 1:, :].reshape(count, -1)
    summary = history[:, :, :1, :].reshape(count, -1)
    candidate_features = {
        "shared_only": shared,
        "anchor_only": np.concatenate([shared, anchor], axis=1),
        "summary_task_context_plus_anchors": np.concatenate(
            [shared, anchor, summary], axis=1
        ),
    }
    if "global_visual" in labels:
        candidate_features["global_visual_plus_anchors_reference"] = np.concatenate(
            [
                shared,
                anchor,
                _flatten_feature(labels["global_visual"], label="global_visual", count=count),
            ],
            axis=1,
        )

    targets = _readout_targets(labels, count=count)
    train = split == "train"
    development = split == "development"
    if not np.any(train) or not np.any(development):
        raise ValueError("A4 task-state ridge requires train and development rows")
    train_weights = _episode_equal_weights(task[train], episode[train])
    train_target, development_target, slices, floored = _standardize_readout_targets(
        targets, train, train_weights
    )
    candidate_reports: dict[str, Any] = {}
    reductions: dict[str, _EpisodeReduction] = {}
    readout_reductions: dict[str, dict[str, _EpisodeReduction]] = {}
    for name, feature in candidate_features.items():
        prediction = closed_form_ridge_predict(
            feature[train],
            train_target,
            feature[development],
            sample_weight=train_weights,
            alpha=ridge_alpha,
        )
        report, reduction, by_readout = _ridge_candidate_report(
            prediction,
            development_target,
            slices=slices,
            task=task[development],
            episode=episode[development],
        )
        candidate_reports[name] = report
        reductions[name] = reduction
        readout_reductions[name] = by_readout

    current_state_reductions = {
        name: _mean_readout_reductions(by_readout, CURRENT_STATE_READOUT_NAMES)
        for name, by_readout in readout_reductions.items()
    }

    comparison = _episode_comparison(
        reductions["summary_task_context_plus_anchors"],
        reductions["anchor_only"],
        comparison_name="summary_task_context_plus_anchors",
        baseline_name="anchor_only",
    )
    samples = (
        SMOKE_BOOTSTRAP_SAMPLES
        if selected_mode == SMOKE_MODE
        else int(gate["bootstrap_samples"])
    )
    comparison["paired_ci"] = _bootstrap_episode_difference(
        reductions["summary_task_context_plus_anchors"],
        reductions["anchor_only"],
        samples=samples,
        seed=BOOTSTRAP_SEED,
    )
    comparison["per_readout"] = {
        name: _episode_comparison(
            readout_reductions["summary_task_context_plus_anchors"][name],
            readout_reductions["anchor_only"][name],
            comparison_name="summary_task_context_plus_anchors",
            baseline_name="anchor_only",
        )
        for name in READOUT_NAMES
    }
    contextual_state_comparison = _episode_comparison(
        current_state_reductions["summary_task_context_plus_anchors"],
        current_state_reductions["shared_only"],
        comparison_name="summary_task_context_plus_anchors",
        baseline_name="shared_only",
    )
    contextual_state_comparison["paired_ci"] = _bootstrap_episode_difference(
        current_state_reductions["summary_task_context_plus_anchors"],
        current_state_reductions["shared_only"],
        samples=samples,
        seed=BOOTSTRAP_SEED,
    )
    contextual_state_comparison["per_readout"] = {
        name: _episode_comparison(
            readout_reductions["summary_task_context_plus_anchors"][name],
            readout_reductions["shared_only"][name],
            comparison_name="summary_task_context_plus_anchors",
            baseline_name="shared_only",
        )
        for name in CURRENT_STATE_READOUT_NAMES
    }
    contextual_state_checks = {
        "minimum_relative_improvement": contextual_state_comparison[
            "relative_improvement"
        ]
        >= float(gate["minimum_summary_relative_improvement"]),
        "minimum_nonnegative_tasks": contextual_state_comparison[
            "nonnegative_task_count"
        ]
        >= int(gate["minimum_nonnegative_tasks"]),
    }
    contextual_state_passed = all(contextual_state_checks.values())
    checks = {
        "minimum_summary_relative_improvement": comparison["relative_improvement"]
        >= float(gate["minimum_summary_relative_improvement"]),
        "minimum_nonnegative_tasks": comparison["nonnegative_task_count"]
        >= int(gate["minimum_nonnegative_tasks"]),
    }
    passed = all(checks.values())
    return {
        "protocol": PROTOCOL,
        "stage": f"a4_{selected_mode}_task_state_sufficiency",
        "mode": selected_mode,
        "decision": "Task-State-Go" if passed else "Task-State-No-Go",
        "passed": bool(passed),
        "input_contains_test_rows": False,
        "fit_split": "train",
        "evaluation_split": "development",
        "readout": "closed_form_ridge",
        "neural_optimizer_steps": 0,
        "ridge_alpha": float(ridge_alpha),
        "aggregation": "task_then_source_physical_episode",
        "shared_deployable_context": list(gate["shared_deployable_context"]),
        "required_readouts": list(READOUT_NAMES),
        "current_state_gate_readouts": list(CURRENT_STATE_READOUT_NAMES),
        "future_diagnostic_readouts": list(FUTURE_DIAGNOSTIC_READOUT_NAMES),
        "target_standardization": {
            "source": "episode_equal_train_rows",
            "floored_dimensions": floored,
        },
        "candidates": candidate_reports,
        "comparison": comparison,
        "contextualized_state_gate": {
            "purpose": (
                "Tests whether the complete contextualized grounded state adds "
                "current-state information beyond task/proprio/action context. Future "
                "branch outcomes remain diagnostic because the shared baseline sees "
                "candidate future actions but no learned dynamics. The nested "
                "summary-vs-anchor comparison is also diagnostic because post-attention "
                "anchor tokens already contain global context."
            ),
            "gate_readouts": list(CURRENT_STATE_READOUT_NAMES),
            "diagnostic_readouts": list(FUTURE_DIAGNOSTIC_READOUT_NAMES),
            "comparison": contextual_state_comparison,
            "minimum_relative_improvement": float(
                gate["minimum_summary_relative_improvement"]
            ),
            "minimum_nonnegative_tasks": int(gate["minimum_nonnegative_tasks"]),
            "checks": contextual_state_checks,
            "passed": bool(contextual_state_passed),
        },
        "gate": {
            "minimum_summary_relative_improvement": float(
                gate["minimum_summary_relative_improvement"]
            ),
            "minimum_nonnegative_tasks": int(gate["minimum_nonnegative_tasks"]),
            "checks": checks,
            "failed_checks": [name for name, value in checks.items() if not value],
            "passed": bool(passed),
        },
    }


def _validate_external_gates(
    values: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    gates = _mapping(values, label="external preflight gates")
    if set(gates) != set(EXTERNAL_PREFLIGHT_GATES):
        raise ValueError(
            "external preflight gates must contain exactly environment, snapshot, "
            "penetration, and FCPG checks"
        )
    reports: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for gate_name in EXTERNAL_PREFLIGHT_GATES:
        per_task = _mapping(gates[gate_name], label=gate_name)
        if set(per_task) != set(TASKS):
            raise ValueError(f"{gate_name} must report every and only the seven A4 tasks")
        normalized: dict[str, bool] = {}
        for task_name in TASKS:
            value = per_task[task_name]
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{gate_name}.{task_name} must be boolean")
            normalized[task_name] = bool(value)
        checks[gate_name] = all(normalized.values())
        reports[gate_name] = {
            "passed": checks[gate_name],
            "per_task": normalized,
        }
    return reports, checks


def summarize_a4_preflight(
    ledger: Mapping[str, np.ndarray],
    cache: Mapping[str, np.ndarray],
    measurements: Mapping[str, np.ndarray],
    external_gates: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    mode: str = FORMAL_MODE,
) -> dict[str, Any]:
    """Combine the complete practical A4 preflight without authorizing test."""

    selected_mode = _mode(mode)
    validate_a4_scoring_contract(contract, phase="preflight")
    external_reports, external_checks = _validate_external_gates(external_gates)
    support = summarize_a4_support(
        ledger, cache, measurements, contract=contract, mode=selected_mode
    )
    task_state = summarize_a4_task_state_sufficiency(
        cache, measurements, contract=contract, mode=selected_mode
    )
    checks = {
        **external_checks,
        "active_control_support": support["active_control_support_passed"],
        "physical_support": support["physical_support_passed"],
        "latent_support": support["latent_support_passed"],
        "task_state_sufficiency": task_state["passed"],
    }
    required = tuple(contract["preflight"]["required_gates_before_training"])
    if required != (*EXTERNAL_PREFLIGHT_GATES, *DATA_PREFLIGHT_GATES):
        raise ValueError("A4 required preflight gate order drifted")
    passed = all(checks[name] for name in required)
    if selected_mode == SMOKE_MODE:
        decision = "Smoke-Go" if passed else "Smoke-No-Go"
    else:
        decision = "Preflight-Go" if passed else "Preflight-No-Go"
    return {
        "protocol": PROTOCOL,
        "stage": f"a4_{selected_mode}_preflight",
        "mode": selected_mode,
        "decision": decision,
        "passed": bool(passed),
        "training_authorized": bool(passed and selected_mode == FORMAL_MODE),
        "reserved_test_access_authorized": False,
        "input_contains_test_rows": False,
        "model_training_started": False,
        "required_gates": list(required),
        "checks": checks,
        "failed_checks": [name for name in required if not checks[name]],
        "external_gates": external_reports,
        "support": support,
        "task_state_sufficiency": task_state,
    }


score_a4_preflight = summarize_a4_preflight


@dataclass(frozen=True)
class _TaskMoments:
    mean: np.ndarray
    scale: np.ndarray
    episode_count: int
    floored_dimensions: int


@dataclass(frozen=True)
class _SeedEpisodeReduction:
    task_values: Mapping[str, float]
    task_seed_values: Mapping[str, Mapping[int, float]]
    episode_seed_values: Mapping[str, np.ndarray]


def _fit_task_moments(
    target: np.ndarray,
    *,
    task: np.ndarray,
    episode: np.ndarray,
    split: np.ndarray,
    horizon_indices: np.ndarray,
) -> dict[str, _TaskMoments]:
    moments: dict[str, _TaskMoments] = {}
    for task_index, task_name in enumerate(TASKS):
        selected = (task == task_index) & (split == "train")
        identities = np.unique(episode[selected])
        if identities.size == 0:
            raise ValueError(f"{task_name} has no train physical episodes")
        episode_means: list[np.ndarray] = []
        episode_seconds: list[np.ndarray] = []
        for identity in identities:
            values = target[selected & (episode == identity)][:, :, horizon_indices]
            values = values.astype(np.float64, copy=False)
            episode_means.append(np.mean(values, axis=(0, 1, 2), dtype=np.float64))
            episode_seconds.append(
                np.mean(np.square(values), axis=(0, 1, 2), dtype=np.float64)
            )
        mean = np.mean(np.stack(episode_means), axis=0, dtype=np.float64)
        second = np.mean(np.stack(episode_seconds), axis=0, dtype=np.float64)
        scale = np.sqrt(np.maximum(second - np.square(mean), 0.0))
        floor = scale <= NUMERIC_EPSILON
        moments[task_name] = _TaskMoments(
            mean=mean,
            scale=np.where(floor, 1.0, scale),
            episode_count=int(identities.size),
            floored_dimensions=int(np.count_nonzero(floor)),
        )
    return moments


def _prediction_alias(
    predictions: Mapping[str, np.ndarray],
    aliases: Sequence[str],
    *,
    role: str,
) -> np.ndarray:
    values = _mapping(predictions, label="A4 development predictions")
    present = [name for name in aliases if name in values]
    if not present:
        raise ValueError(f"A4 development predictions are missing {role!r}")
    reference = np.asarray(values[present[0]])
    for name in present[1:]:
        if not np.array_equal(reference, np.asarray(values[name])):
            raise ValueError(f"conflicting aliases were supplied for {role!r}")
    return _numeric(reference, label=f"prediction {role!r}")


def validate_wrong_action_prediction_replay(
    factual: np.ndarray,
    wrong_action: np.ndarray,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-7,
) -> None:
    """Require all seven wrong-action panels to replay factual donor slots."""

    factual_array = _numeric(factual, label="factual prediction")
    wrong_array = _numeric(wrong_action, label="wrong-action prediction")
    if factual_array.ndim != 6:
        raise ValueError("factual prediction must have shape [seed,N,8,H,K,D]")
    expected_shape = (
        factual_array.shape[0],
        factual_array.shape[1],
        WRONG_ACTION_SHIFT_COUNT,
        *factual_array.shape[2:],
    )
    if wrong_array.shape != expected_shape:
        raise ValueError(f"wrong-action prediction must have shape {expected_shape}")
    slots = np.arange(BRANCH_COUNT, dtype=np.int64)
    for shift in range(1, BRANCH_COUNT):
        donor = np.remainder(slots + shift, BRANCH_COUNT)
        expected = factual_array[:, :, donor]
        observed = wrong_array[:, :, shift - 1]
        if not np.allclose(observed, expected, rtol=rtol, atol=atol):
            deviation = float(np.max(np.abs(observed - expected)))
            raise ValueError(
                "wrong-action prediction is not an exact cyclic replay of factual "
                f"donor shift {shift}; max_deviation={deviation}"
            )


validate_wrong_action_replay = validate_wrong_action_prediction_replay


def _validated_development_predictions(
    predictions: Mapping[str, np.ndarray],
    *,
    seed_count: int,
    evaluation_count: int,
    target_shape: tuple[int, ...],
) -> dict[str, np.ndarray]:
    factual = _prediction_alias(
        predictions,
        (FACTUAL_METHOD, "factual_prediction"),
        role=FACTUAL_METHOD,
    )
    no_action = _prediction_alias(
        predictions,
        (NO_ACTION_METHOD, "no_action_prediction", "matched_no_action"),
        role=NO_ACTION_METHOD,
    )
    wrong_action = _prediction_alias(
        predictions,
        (
            WRONG_ACTION_METHOD,
            "wrong_action_prediction",
            "shuffled_action",
            "shuffled_action_prediction",
        ),
        role=WRONG_ACTION_METHOD,
    )
    ordinary_shape = (seed_count, evaluation_count, *target_shape)
    wrong_shape = (
        seed_count,
        evaluation_count,
        WRONG_ACTION_SHIFT_COUNT,
        *target_shape,
    )
    if factual.shape != ordinary_shape:
        raise ValueError(f"factual prediction must have shape {ordinary_shape}")
    if no_action.shape != ordinary_shape:
        raise ValueError(f"no-action prediction must have shape {ordinary_shape}")
    if wrong_action.shape != wrong_shape:
        raise ValueError(f"wrong-action prediction must have shape {wrong_shape}")
    if not np.allclose(no_action, no_action[:, :, :1], rtol=0.0, atol=1e-6):
        raise ValueError("matched no-action prediction must be branch-invariant")
    validate_wrong_action_prediction_replay(factual, wrong_action)
    return {
        FACTUAL_METHOD: factual,
        NO_ACTION_METHOD: no_action,
        WRONG_ACTION_METHOD: wrong_action,
    }


def _secondary_predictions(history: np.ndarray) -> dict[str, np.ndarray]:
    values = _numeric(history, label="history_state").astype(np.float64, copy=False)
    if values.ndim != 4 or values.shape[1] < 2:
        raise ValueError("history_state must have shape [N,H,K,D] with H>=2")
    last = values[:, -1]
    velocity = last - values[:, -2]
    factors = np.arange(1, HORIZON + 1, dtype=np.float64).reshape(1, HORIZON, 1, 1)
    hold = np.broadcast_to(last[:, None], (values.shape[0], HORIZON, *last.shape[1:]))
    cv = last[:, None] + factors * velocity[:, None]
    shape = (values.shape[0], BRANCH_COUNT, HORIZON, *last.shape[1:])
    return {
        HOLD_LAST_METHOD: np.broadcast_to(hold[:, None], shape),
        LATENT_CONSTANT_VELOCITY_METHOD: np.broadcast_to(cv[:, None], shape),
    }


def _ordinary_development_losses(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    task: np.ndarray,
    moments: Mapping[str, _TaskMoments],
    horizon_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    seed_count, row_count = prediction.shape[:2]
    centered = np.empty((seed_count, row_count), dtype=np.float64)
    raw = np.empty((seed_count, row_count), dtype=np.float64)
    for task_index, task_name in enumerate(TASKS):
        selected = task == task_index
        moment = moments[task_name]
        target_w = (
            target[selected][:, :, horizon_indices].astype(np.float64, copy=False)
            - moment.mean
        ) / moment.scale
        prediction_w = (
            prediction[:, selected][:, :, :, horizon_indices].astype(
                np.float64, copy=False
            )
            - moment.mean
        ) / moment.scale
        target_centered = target_w - np.mean(
            target_w, axis=1, keepdims=True, dtype=np.float64
        )
        prediction_centered = prediction_w - np.mean(
            prediction_w, axis=2, keepdims=True, dtype=np.float64
        )
        centered[:, selected] = np.mean(
            np.square(prediction_centered - target_centered[None]),
            axis=(2, 3, 4, 5),
            dtype=np.float64,
        )
        raw[:, selected] = np.mean(
            np.square(prediction_w - target_w[None]),
            axis=(2, 3, 4, 5),
            dtype=np.float64,
        )
    return centered, raw


def _wrong_action_centered_losses(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    task: np.ndarray,
    moments: Mapping[str, _TaskMoments],
    horizon_indices: np.ndarray,
) -> np.ndarray:
    seed_count, row_count, shift_count = prediction.shape[:3]
    centered = np.empty((seed_count, row_count, shift_count), dtype=np.float64)
    for task_index, task_name in enumerate(TASKS):
        selected = task == task_index
        moment = moments[task_name]
        target_w = (
            target[selected][:, :, horizon_indices].astype(np.float64, copy=False)
            - moment.mean
        ) / moment.scale
        prediction_w = (
            prediction[:, selected][:, :, :, :, horizon_indices].astype(
                np.float64, copy=False
            )
            - moment.mean
        ) / moment.scale
        target_centered = target_w - np.mean(
            target_w, axis=1, keepdims=True, dtype=np.float64
        )
        prediction_centered = prediction_w - np.mean(
            prediction_w, axis=3, keepdims=True, dtype=np.float64
        )
        centered[:, selected] = np.mean(
            np.square(prediction_centered - target_centered[None, :, None]),
            axis=(3, 4, 5, 6),
            dtype=np.float64,
        )
    return centered


def _reduce_seed_episode_losses(
    losses: np.ndarray,
    *,
    task: np.ndarray,
    episode: np.ndarray,
    seeds: Sequence[int],
) -> _SeedEpisodeReduction:
    values = _numeric(losses, label="development losses").astype(np.float64, copy=False)
    if values.shape != (len(seeds), task.size) or np.any(values < 0.0):
        raise ValueError("development losses must have shape [seed,bundle] and be non-negative")
    task_values: dict[str, float] = {}
    task_seed_values: dict[str, dict[int, float]] = {}
    episode_seed_values: dict[str, np.ndarray] = {}
    for task_index, task_name in enumerate(TASKS):
        selected = task == task_index
        identities = np.unique(episode[selected])
        if identities.size == 0:
            raise ValueError(f"{task_name} has no development physical episodes")
        matrix = np.empty((len(seeds), identities.size), dtype=np.float64)
        for seed_index in range(len(seeds)):
            for episode_index, identity in enumerate(identities):
                matrix[seed_index, episode_index] = np.mean(
                    values[seed_index, selected & (episode == identity)],
                    dtype=np.float64,
                )
        episode_seed_values[task_name] = matrix
        task_seed_values[task_name] = {
            int(seed): float(np.mean(matrix[position]))
            for position, seed in enumerate(seeds)
        }
        task_values[task_name] = float(np.mean(matrix))
    return _SeedEpisodeReduction(
        task_values=task_values,
        task_seed_values=task_seed_values,
        episode_seed_values=episode_seed_values,
    )


def _seed_method_report(
    reduction: _SeedEpisodeReduction, *, seeds: Sequence[int]
) -> dict[str, Any]:
    return {
        "task_macro_mse": float(np.mean(list(reduction.task_values.values()))),
        "per_task": {
            task_name: {
                "mse": reduction.task_values[task_name],
                "physical_episode_count": int(
                    reduction.episode_seed_values[task_name].shape[1]
                ),
                "per_seed": {
                    str(seed): {"mse": reduction.task_seed_values[task_name][seed]}
                    for seed in seeds
                },
            }
            for task_name in TASKS
        },
        "per_seed": {
            str(seed): {
                "task_macro_mse": float(
                    np.mean(
                        [reduction.task_seed_values[name][seed] for name in TASKS]
                    )
                )
            }
            for seed in seeds
        },
    }


def _seed_comparison_report(
    factual: _SeedEpisodeReduction,
    baseline: _SeedEpisodeReduction,
    *,
    baseline_method: str,
    seeds: Sequence[int],
    bootstrap_samples: int,
) -> dict[str, Any]:
    per_task: dict[str, Any] = {}
    for task_name in TASKS:
        per_seed = {
            str(seed): {
                "factual_mse": factual.task_seed_values[task_name][seed],
                "baseline_mse": baseline.task_seed_values[task_name][seed],
                "relative_improvement": _relative_improvement(
                    factual.task_seed_values[task_name][seed],
                    baseline.task_seed_values[task_name][seed],
                    label=f"{task_name}/{seed}/{baseline_method}",
                ),
            }
            for seed in seeds
        }
        per_task[task_name] = {
            "factual_mse": factual.task_values[task_name],
            "baseline_mse": baseline.task_values[task_name],
            "relative_improvement": _relative_improvement(
                factual.task_values[task_name],
                baseline.task_values[task_name],
                label=f"{task_name}/{baseline_method}",
            ),
            "per_seed": per_seed,
        }
    per_seed: dict[str, Any] = {}
    for seed in seeds:
        effects = {
            task_name: per_task[task_name]["per_seed"][str(seed)][
                "relative_improvement"
            ]
            for task_name in TASKS
        }
        effect = float(np.mean(list(effects.values())))
        per_seed[str(seed)] = {
            "relative_improvement": effect,
            "positive": effect > 0.0,
            "per_task": effects,
        }

    seed_offset = zlib.crc32(baseline_method.encode("utf-8"))
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    bootstrap_by_task: list[np.ndarray] = []
    for task_name in TASKS:
        factual_values = factual.episode_seed_values[task_name]
        baseline_values = baseline.episode_seed_values[task_name]
        draws = rng.integers(
            0,
            factual_values.shape[1],
            size=(bootstrap_samples, factual_values.shape[1]),
        )
        factual_draw = np.mean(factual_values[:, draws], axis=(0, 2), dtype=np.float64)
        baseline_draw = np.mean(baseline_values[:, draws], axis=(0, 2), dtype=np.float64)
        bootstrap_by_task.append(baseline_draw - factual_draw)
    distribution = np.mean(
        np.stack(bootstrap_by_task, axis=1), axis=1, dtype=np.float64
    )
    lower, upper = np.quantile(distribution, (0.025, 0.975))
    effect = float(
        np.mean([per_task[name]["relative_improvement"] for name in TASKS])
    )
    return {
        "metric": "h6_h8_within_bundle_branch_centered_whitened_mse",
        "factual_method": FACTUAL_METHOD,
        "baseline_method": baseline_method,
        "orientation": "positive_means_factual_has_lower_mse",
        "relative_improvement": effect,
        "positive_task_count": sum(
            per_task[name]["relative_improvement"] > 0.0 for name in TASKS
        ),
        "positive_seed_count": sum(row["positive"] for row in per_seed.values()),
        "per_task": per_task,
        "per_seed": per_seed,
        "paired_ci": {
            "metric": "paired_task_macro_mse_difference",
            "estimate": float(
                np.mean(
                    [
                        baseline.task_values[name] - factual.task_values[name]
                        for name in TASKS
                    ]
                )
            ),
            "confidence": 0.95,
            "lower": float(lower),
            "upper": float(upper),
            "samples": bootstrap_samples,
            "unit": "source_physical_episode_within_task",
        },
    }


def summarize_a4_development(
    cache: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    *,
    contract: Mapping[str, Any],
    training_seeds: Sequence[int] = DEVELOPMENT_TRAINING_SEEDS,
    mode: str = FORMAL_MODE,
) -> dict[str, Any]:
    """Score the A4 2k development panel without opening reserved test."""

    selected_mode = _mode(mode)
    gate = validate_a4_scoring_contract(contract, phase="development")
    validate_a4_state_cache(cache)
    split = validate_no_reserved_test_rows(cache, label="A4 state cache")
    cache_id, task, _task_name, episode, _ = _cache_identity(cache)
    _validate_mode_coverage(
        task,
        episode,
        split,
        mode=selected_mode,
        label="development cache",
        formal_train_minimum=int(
            contract["support_gate"]["minimum_active_bundle_episode_count_per_task"]
        ),
    )
    seeds = tuple(int(seed) for seed in training_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("A4 development seeds must be distinct and non-empty")
    if selected_mode == FORMAL_MODE and seeds != DEVELOPMENT_TRAINING_SEEDS:
        raise ValueError(
            f"formal A4 development seeds must remain {DEVELOPMENT_TRAINING_SEEDS}"
        )
    if selected_mode == SMOKE_MODE and not set(seeds).issubset(
        DEVELOPMENT_TRAINING_SEEDS
    ):
        raise ValueError("A4 smoke seeds must be a subset of the frozen development seeds")

    horizons = tuple(int(value) for value in gate["primary_horizons"])
    if horizons != PRIMARY_HORIZONS:
        raise ValueError("A4 development horizons must remain H6-H8")
    horizon_indices = np.asarray(horizons, dtype=np.int64) - 1
    target = np.asarray(cache["target_state"])
    history = np.asarray(cache["history_state"])
    development = split == "development"
    development_rows = np.flatnonzero(development)
    checked = _validated_development_predictions(
        predictions,
        seed_count=len(seeds),
        evaluation_count=development_rows.size,
        target_shape=tuple(target.shape[1:]),
    )
    moments = _fit_task_moments(
        target,
        task=task,
        episode=episode,
        split=split,
        horizon_indices=horizon_indices,
    )
    evaluation_target = target[development]
    evaluation_task = task[development]
    evaluation_episode = episode[development]

    centered_losses: dict[str, np.ndarray] = {}
    raw_losses: dict[str, np.ndarray] = {}
    for method in (FACTUAL_METHOD, NO_ACTION_METHOD):
        centered_losses[method], raw_losses[method] = _ordinary_development_losses(
            evaluation_target,
            checked[method],
            task=evaluation_task,
            moments=moments,
            horizon_indices=horizon_indices,
        )
    wrong_losses = _wrong_action_centered_losses(
        evaluation_target,
        checked[WRONG_ACTION_METHOD],
        task=evaluation_task,
        moments=moments,
        horizon_indices=horizon_indices,
    )
    centered_losses[WRONG_ACTION_METHOD] = np.mean(
        wrong_losses, axis=2, dtype=np.float64
    )

    secondary = _secondary_predictions(history[development])
    for method in (HOLD_LAST_METHOD, LATENT_CONSTANT_VELOCITY_METHOD):
        prediction = np.broadcast_to(
            secondary[method][None],
            (len(seeds), *secondary[method].shape),
        )
        _, raw_losses[method] = _ordinary_development_losses(
            evaluation_target,
            prediction,
            task=evaluation_task,
            moments=moments,
            horizon_indices=horizon_indices,
        )

    centered_reductions = {
        method: _reduce_seed_episode_losses(
            loss,
            task=evaluation_task,
            episode=evaluation_episode,
            seeds=seeds,
        )
        for method, loss in centered_losses.items()
    }
    raw_reductions = {
        method: _reduce_seed_episode_losses(
            loss,
            task=evaluation_task,
            episode=evaluation_episode,
            seeds=seeds,
        )
        for method, loss in raw_losses.items()
    }
    shift_reductions = [
        _reduce_seed_episode_losses(
            wrong_losses[:, :, shift],
            task=evaluation_task,
            episode=evaluation_episode,
            seeds=seeds,
        )
        for shift in range(WRONG_ACTION_SHIFT_COUNT)
    ]
    bootstrap_samples = (
        SMOKE_BOOTSTRAP_SAMPLES
        if selected_mode == SMOKE_MODE
        else int(gate["bootstrap_samples"])
    )
    no_action_comparison = _seed_comparison_report(
        centered_reductions[FACTUAL_METHOD],
        centered_reductions[NO_ACTION_METHOD],
        baseline_method=NO_ACTION_METHOD,
        seeds=seeds,
        bootstrap_samples=bootstrap_samples,
    )
    wrong_average_comparison = _seed_comparison_report(
        centered_reductions[FACTUAL_METHOD],
        centered_reductions[WRONG_ACTION_METHOD],
        baseline_method="wrong_action_mean",
        seeds=seeds,
        bootstrap_samples=bootstrap_samples,
    )
    shift_reports = [
        {
            "donor_shift": shift + 1,
            **_seed_comparison_report(
                centered_reductions[FACTUAL_METHOD],
                reduction,
                baseline_method=f"wrong_action_shift_{shift + 1}",
                seeds=seeds,
                bootstrap_samples=bootstrap_samples,
            ),
        }
        for shift, reduction in enumerate(shift_reductions)
    ]
    raw_reports = {
        method: _seed_method_report(reduction, seeds=seeds)
        for method, reduction in raw_reductions.items()
    }

    jointly_positive_tasks = [
        task_name
        for task_name in TASKS
        if no_action_comparison["per_task"][task_name]["relative_improvement"] > 0.0
        and all(
            row["per_task"][task_name]["relative_improvement"] > 0.0
            for row in shift_reports
        )
    ]
    jointly_positive_seeds = [
        seed
        for seed in seeds
        if no_action_comparison["per_seed"][str(seed)]["relative_improvement"] > 0.0
        and all(
            row["per_seed"][str(seed)]["relative_improvement"] > 0.0
            for row in shift_reports
        )
    ]
    minimum_effect = float(gate["minimum_relative_improvement"])
    ci_reports = [no_action_comparison, *shift_reports]
    checks = {
        "minimum_effect_vs_no_action": no_action_comparison["relative_improvement"]
        >= minimum_effect,
        "minimum_effect_vs_every_wrong_action_shift": min(
            row["relative_improvement"] for row in shift_reports
        )
        >= minimum_effect,
        "paired_ci_lower_exceeds_zero": all(
            row["paired_ci"]["lower"]
            > float(gate["paired_ci_lower_must_exceed"])
            for row in ci_reports
        ),
        "minimum_positive_tasks": len(jointly_positive_tasks)
        >= int(gate["minimum_positive_tasks"]),
        "minimum_positive_seeds": len(jointly_positive_seeds)
        >= (
            min(int(gate["minimum_positive_seeds"]), len(seeds))
            if selected_mode == SMOKE_MODE
            else int(gate["minimum_positive_seeds"])
        ),
        "raw_factual_not_worse_than_no_action": raw_reports[FACTUAL_METHOD][
            "task_macro_mse"
        ]
        <= raw_reports[NO_ACTION_METHOD]["task_macro_mse"],
        "raw_factual_better_than_hold_last": raw_reports[FACTUAL_METHOD][
            "task_macro_mse"
        ]
        < raw_reports[HOLD_LAST_METHOD]["task_macro_mse"],
        "raw_factual_better_than_constant_velocity": raw_reports[FACTUAL_METHOD][
            "task_macro_mse"
        ]
        < raw_reports[LATENT_CONSTANT_VELOCITY_METHOD]["task_macro_mse"],
    }
    scientific_passed = all(checks.values())
    development_go = scientific_passed and selected_mode == FORMAL_MODE
    decision = (
        ("Development-Go" if development_go else "Development-No-Go")
        if selected_mode == FORMAL_MODE
        else ("Smoke-Go" if scientific_passed else "Smoke-No-Go")
    )
    coverage = _coverage_counts(task, episode, split)
    return {
        "protocol": PROTOCOL,
        "stage": f"a4_2k_{selected_mode}_development",
        "mode": selected_mode,
        "decision": decision,
        "passed": bool(scientific_passed),
        "development_go": bool(development_go),
        "reserved_test_collection_authorized": bool(development_go),
        "input_contains_test_rows": False,
        "checkpoint_step": int(gate["checkpoint_step"]),
        "primary_horizons": list(horizons),
        "aggregation": [
            "bundle_branches_and_horizons",
            "bundle_within_training_seed_and_physical_episode",
            "training_seed_within_physical_episode",
            "physical_episode_within_task",
            "equal_weight_task_macro",
        ],
        "coverage": {
            "tasks": list(TASKS),
            "training_seeds": list(seeds),
            "development_bundle_count": int(development_rows.size),
            "per_task": coverage,
        },
        "whitening": {
            "source": "episode_equal_train_rows",
            "per_task": True,
            "fit_horizons": list(horizons),
            "tasks": {
                task_name: {
                    "train_physical_episode_count": moments[task_name].episode_count,
                    "floored_dimensions": moments[task_name].floored_dimensions,
                }
                for task_name in TASKS
            },
        },
        "primary": {
            "metric": "h6_h8_within_bundle_branch_centered_whitened_mse",
            "methods": {
                method: _seed_method_report(reduction, seeds=seeds)
                for method, reduction in centered_reductions.items()
            },
            "comparisons": {
                NO_ACTION_METHOD: no_action_comparison,
                "wrong_action_mean": wrong_average_comparison,
            },
            "wrong_action_shifts": {
                "count": WRONG_ACTION_SHIFT_COUNT,
                "minimum_relative_improvement": min(
                    row["relative_improvement"] for row in shift_reports
                ),
                "per_shift": shift_reports,
            },
        },
        "raw_h6_h8": {
            "metric": "h6_h8_raw_whitened_trajectory_mse",
            "methods": raw_reports,
        },
        "gate": {
            "minimum_relative_improvement": minimum_effect,
            "checks": checks,
            "failed_checks": [name for name, value in checks.items() if not value],
            "jointly_positive_tasks": jointly_positive_tasks,
            "jointly_positive_seeds": jointly_positive_seeds,
            "passed": bool(scientific_passed),
            "development_go": bool(development_go),
        },
    }


summarize_a4_metrics = summarize_a4_development
score_a4_development = summarize_a4_development


__all__ = [
    "BOOTSTRAP_SEED",
    "CURRENT_STATE_READOUT_NAMES",
    "DATA_PREFLIGHT_GATES",
    "EXTERNAL_PREFLIGHT_GATES",
    "FACTUAL_METHOD",
    "FORMAL_MODE",
    "FUTURE_DIAGNOSTIC_READOUT_NAMES",
    "HOLD_LAST_METHOD",
    "LATENT_CONSTANT_VELOCITY_METHOD",
    "NO_ACTION_METHOD",
    "PRIMARY_HORIZONS",
    "READOUT_NAMES",
    "RIDGE_ALPHA",
    "SMOKE_MODE",
    "WRONG_ACTION_METHOD",
    "WRONG_ACTION_SHIFT_COUNT",
    "closed_form_ridge_predict",
    "score_a4_development",
    "score_a4_preflight",
    "summarize_a4_development",
    "summarize_a4_metrics",
    "summarize_a4_preflight",
    "summarize_a4_support",
    "summarize_a4_task_state_sufficiency",
    "validate_a4_scoring_contract",
    "validate_no_reserved_test_rows",
    "validate_wrong_action_prediction_replay",
    "validate_wrong_action_replay",
]
