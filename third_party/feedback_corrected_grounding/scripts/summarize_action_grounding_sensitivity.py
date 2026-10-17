#!/usr/bin/env python3
"""Aggregate action-grounding sensitivity diagnostics across policy seeds.

The primary estimand gives every task equal weight and, within each task,
every policy seed equal weight.  Overlapping sequence windows are reduced to
one value per physical episode before either point estimates or intervals are
computed.  The hierarchical bootstrap treats policy seeds and shared physical
episodes as crossed factors within each fixed task.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

METRICS = (
    "delta_epe",
    "delta_action_xyz_h1",
    "delta_action_xyz_h8",
    "delta_gripper_h1",
    "delta_gripper_h8",
)
ABSOLUTE_METRICS = (
    "nominal_epe",
    "fcpg_epe",
    "nominal_oracle_xyz_h1",
    "fcpg_oracle_xyz_h1",
    "nominal_oracle_xyz_h8",
    "fcpg_oracle_xyz_h8",
    "nominal_oracle_gripper_h1",
    "fcpg_oracle_gripper_h1",
    "nominal_oracle_gripper_h8",
    "fcpg_oracle_gripper_h8",
)
COMPONENT_FIELDS = (
    "base_valid_anchors",
    "common_valid_anchors",
    "fcpg_oob_anchors",
    "common_support_coverage",
    "nominal_epe",
    "fcpg_epe",
    "nominal_oracle_xyz_h1",
    "fcpg_oracle_xyz_h1",
    "nominal_oracle_xyz_h8",
    "fcpg_oracle_xyz_h8",
    "nominal_oracle_gripper_h1",
    "fcpg_oracle_gripper_h1",
    "nominal_oracle_gripper_h8",
    "fcpg_oracle_gripper_h8",
)
ASSOCIATION_TARGETS = (
    "delta_action_xyz_h1",
    "delta_action_xyz_h8",
    "delta_gripper_h1",
    "delta_gripper_h8",
)
STRATA = ("all", "clean", "drift")


@dataclass(frozen=True)
class InputSpec:
    path: Path
    task: str | None = None
    policy_seed: int | None = None


@dataclass(frozen=True)
class LoadedResult:
    path: Path
    task: str
    policy_seed: int
    inputs: Mapping[str, Any]
    rows: tuple[dict[str, Any], ...]


def parse_input_spec(value: str) -> InputSpec:
    """Parse TASK=SEED=PATH, or defer task/seed discovery to the payload."""

    parts = value.split("=", 2)
    if len(parts) == 1:
        if not parts[0]:
            raise argparse.ArgumentTypeError("--input path must not be empty")
        return InputSpec(path=Path(parts[0]))
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            "--input must be PATH or TASK=POLICY_SEED=PATH"
        )
    task, seed_text, path_text = parts
    try:
        policy_seed = int(seed_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("policy seed must be an integer") from error
    return InputSpec(path=Path(path_text), task=task, policy_seed=policy_seed)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _camera_clean(row: Mapping[str, Any], *, label: str) -> bool:
    required = ("actual_camera_rotation_deg", "actual_camera_translation_norm")
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"{label} is missing camera fields: {missing}")
    rotation = float(row["actual_camera_rotation_deg"])
    translation = float(row["actual_camera_translation_norm"])
    if not np.isfinite(rotation) or not np.isfinite(translation):
        raise ValueError(f"{label} camera fields must be finite")
    inferred = rotation == 0.0 and translation == 0.0
    for field in ("camera_clean", "actual_camera_clean"):
        if field in row and row[field] is not None:
            declared = row[field]
            if not isinstance(declared, (bool, np.bool_)):
                raise ValueError(f"{label}.{field} must be boolean")
            if bool(declared) != inferred:
                raise ValueError(
                    f"{label}.{field} disagrees with exact zero rotation/translation"
                )
    return inferred


def _validate_contract(payload: Mapping[str, Any], path: Path) -> Mapping[str, Any]:
    contract = _require_mapping(payload.get("contract"), f"{path}: contract")
    required_contract = {
        "training_performed": False,
        "same_checkpoint_same_batch": True,
        "grounding_interventions": ["nominal", "fcpg", "oracle"],
        "oracle_and_q_true_usage": (
            "oracle_reference_only_absent_from_deployable_modes"
        ),
        "statistical_unit": "complete_episode_within_task",
        "overlapping_windows_treated_as_independent": False,
        "exploratory": False,
        "smoke_only": False,
        "inferential": True,
        "noninferential": False,
        "confidence_intervals_reported": True,
        "artifact_contract_mismatches": [],
    }
    for key, expected in required_contract.items():
        if contract.get(key) != expected:
            raise ValueError(
                f"{path}: incomplete/incompatible contract field {key!r}: "
                f"expected {expected!r}, got {contract.get(key)!r}"
            )
    inputs = _require_mapping(payload.get("inputs"), f"{path}: inputs")
    if inputs.get("max_batches") != 0:
        raise ValueError(
            f"{path}: only full evaluations are accepted; inputs.max_batches must be 0"
        )
    if inputs.get("h_exec") != 8:
        raise ValueError(f"{path}: inputs.h_exec must be 8")
    if inputs.get("allow_cache_override") is not False:
        raise ValueError(f"{path}: inputs.allow_cache_override must be false")
    return inputs


def _validate_delta_identity(row: Mapping[str, Any], label: str) -> None:
    identities = (
        ("delta_epe", "nominal_epe", "fcpg_epe"),
        (
            "delta_action_xyz_h1",
            "nominal_oracle_xyz_h1",
            "fcpg_oracle_xyz_h1",
        ),
        (
            "delta_action_xyz_h8",
            "nominal_oracle_xyz_h8",
            "fcpg_oracle_xyz_h8",
        ),
        (
            "delta_gripper_h1",
            "nominal_oracle_gripper_h1",
            "fcpg_oracle_gripper_h1",
        ),
        (
            "delta_gripper_h8",
            "nominal_oracle_gripper_h8",
            "fcpg_oracle_gripper_h8",
        ),
    )
    for delta, nominal, fcpg in identities:
        raw_values = (row[delta], row[nominal], row[fcpg])
        missing = [value is None for value in raw_values]
        if any(missing):
            if not all(missing):
                raise ValueError(
                    f"{label}.{delta}/{nominal}/{fcpg} must be all finite or all null"
                )
            continue
        values = tuple(float(value) for value in raw_values)
        if not all(np.isfinite(value) for value in values):
            raise ValueError(f"{label}.{delta}/{nominal}/{fcpg} must be finite or null")
        expected = values[1] - values[2]
        if not np.isclose(values[0], expected, rtol=0.0, atol=1e-9):
            raise ValueError(f"{label}.{delta} disagrees with {nominal} - {fcpg}")


def _finite_optional(value: Any) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if np.isfinite(converted) else None


def load_result(spec: InputSpec) -> LoadedResult:
    path = spec.path.resolve()
    payload = _require_mapping(
        json.loads(path.read_text(encoding="utf-8")), str(path)
    )
    inputs = _validate_contract(payload, path)
    required_inputs = ("data", "split_seed", "val_fraction", "vision_cache")
    missing_inputs = [field for field in required_inputs if field not in inputs]
    if missing_inputs:
        raise ValueError(f"{path}: inputs is missing contract fields: {missing_inputs}")
    if not isinstance(inputs["data"], str) or not inputs["data"]:
        raise ValueError(f"{path}: inputs.data must be a non-empty path")
    if not isinstance(inputs["split_seed"], int) or isinstance(inputs["split_seed"], bool):
        raise ValueError(f"{path}: inputs.split_seed must be an integer")
    val_fraction = float(inputs["val_fraction"])
    if not np.isfinite(val_fraction) or not 0.0 < val_fraction < 1.0:
        raise ValueError(f"{path}: inputs.val_fraction must be finite and in (0, 1)")
    if inputs["vision_cache"] is not None and (
        not isinstance(inputs["vision_cache"], str) or not inputs["vision_cache"]
    ):
        raise ValueError(f"{path}: inputs.vision_cache must be null or a non-empty path")
    payload_seed = inputs.get("policy_seed")
    if not isinstance(payload_seed, int) or isinstance(payload_seed, bool):
        raise ValueError(f"{path}: inputs.policy_seed must be an integer")
    if spec.policy_seed is not None and spec.policy_seed != payload_seed:
        raise ValueError(
            f"{path}: explicit policy seed {spec.policy_seed} disagrees with "
            f"payload seed {payload_seed}"
        )

    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"{path}: rows must be a non-empty list")
    rows: list[dict[str, Any]] = []
    tasks: set[str] = set()
    identities: set[tuple[int, int]] = set()
    required_fields = {
        "task",
        "episode",
        "frame",
        "anchor_index",
        "policy_seed",
        *METRICS,
        *COMPONENT_FIELDS,
    }
    for index, raw_row in enumerate(raw_rows):
        row = dict(_require_mapping(raw_row, f"{path}: rows[{index}]"))
        missing = sorted(required_fields.difference(row))
        if missing:
            raise ValueError(f"{path}: rows[{index}] is missing metric fields: {missing}")
        label = f"{path}: rows[{index}]"
        task = str(row["task"])
        if not task:
            raise ValueError(f"{label}.task must not be empty")
        row_seed = row["policy_seed"]
        if not isinstance(row_seed, int) or isinstance(row_seed, bool):
            raise ValueError(f"{label}.policy_seed must be an integer")
        if row_seed != payload_seed:
            raise ValueError(f"{label}.policy_seed disagrees with payload")
        episode = int(row["episode"])
        frame = int(row["frame"])
        identity = (episode, frame)
        if identity in identities:
            raise ValueError(f"{path}: duplicate physical episode/frame identity {identity}")
        identities.add(identity)
        for field in (
            "delta_action_xyz_h1",
            "delta_gripper_h1",
            "nominal_oracle_xyz_h1",
            "fcpg_oracle_xyz_h1",
            "nominal_oracle_gripper_h1",
            "fcpg_oracle_gripper_h1",
        ):
            if _finite_optional(row[field]) is None:
                raise ValueError(f"{label}.{field} must be finite")
        counts = {}
        for field in ("base_valid_anchors", "common_valid_anchors", "fcpg_oob_anchors"):
            value = row[field]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{label}.{field} must be a non-negative integer")
            counts[field] = value
        if counts["common_valid_anchors"] + counts["fcpg_oob_anchors"] != counts[
            "base_valid_anchors"
        ]:
            raise ValueError(f"{label} support anchor counts are inconsistent")
        coverage = _finite_optional(row["common_support_coverage"])
        if counts["base_valid_anchors"]:
            expected_coverage = (
                counts["common_valid_anchors"] / counts["base_valid_anchors"]
            )
            if coverage is None or not np.isclose(
                coverage, expected_coverage, rtol=0.0, atol=1e-12
            ):
                raise ValueError(f"{label}.common_support_coverage is inconsistent")
        elif row["common_support_coverage"] is not None:
            raise ValueError(
                f"{label}.common_support_coverage must be null when base support is zero"
            )
        _validate_delta_identity(row, label)
        if counts["common_valid_anchors"]:
            for field in ("nominal_epe", "fcpg_epe", "delta_epe"):
                if _finite_optional(row[field]) is None:
                    raise ValueError(f"{label}.{field} must be finite on common support")
        elif any(
            row[field] is not None for field in ("nominal_epe", "fcpg_epe", "delta_epe")
        ):
            raise ValueError(
                f"{label} EPE fields must be null when common support is empty"
            )
        row["camera_clean"] = _camera_clean(row, label=label)
        tasks.add(task)
        rows.append(row)

    if len(tasks) != 1:
        raise ValueError(f"{path}: each input must contain exactly one task, got {tasks}")
    task = next(iter(tasks))
    if spec.task is not None and spec.task != task:
        raise ValueError(
            f"{path}: explicit task {spec.task!r} disagrees with payload task {task!r}"
        )

    summary = _require_mapping(payload.get("summary"), f"{path}: summary")
    episode_count = len({int(row["episode"]) for row in rows})
    if summary.get("snapshots") != len(rows):
        raise ValueError(f"{path}: summary.snapshots disagrees with rows")
    if summary.get("physical_episodes") != episode_count:
        raise ValueError(f"{path}: summary.physical_episodes disagrees with rows")
    if summary.get("tasks") != [task]:
        raise ValueError(f"{path}: summary.tasks disagrees with rows")
    if summary.get("inferential") is not True:
        raise ValueError(f"{path}: summary.inferential must be true")
    support = _require_mapping(summary.get("support"), f"{path}: summary.support")
    base_anchors = sum(int(row["base_valid_anchors"]) for row in rows)
    common_anchors = sum(int(row["common_valid_anchors"]) for row in rows)
    oob_anchors = sum(int(row["fcpg_oob_anchors"]) for row in rows)
    expected_support = {
        "base_valid_anchors": base_anchors,
        "common_valid_anchors": common_anchors,
        "fcpg_oob_anchors": oob_anchors,
        "overall_common_support_coverage": (
            common_anchors / base_anchors if base_anchors else None
        ),
    }
    for field, expected in expected_support.items():
        actual = support.get(field)
        if expected is None:
            valid = actual is None
        else:
            valid = actual is not None and np.isclose(
                float(actual), float(expected), rtol=0.0, atol=1e-12
            )
        if not valid:
            raise ValueError(f"{path}: summary.support.{field} disagrees with rows")
    return LoadedResult(path, task, payload_seed, dict(inputs), tuple(rows))


def _identity_payload(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["anchor_index"]),
        float(row["actual_camera_rotation_deg"]),
        float(row["actual_camera_translation_norm"]),
        bool(row["camera_clean"]),
        row["delta_action_xyz_h8"] is not None,
        row["delta_gripper_h8"] is not None,
    )


def _normalized_contract_path(value: Any) -> str | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve())


def _shared_input_contract(result: LoadedResult) -> tuple[Any, ...]:
    return (
        _normalized_contract_path(result.inputs["data"]),
        int(result.inputs["split_seed"]),
        float(result.inputs["val_fraction"]),
        _normalized_contract_path(result.inputs["vision_cache"]),
    )


def validate_collection(results: Sequence[LoadedResult]) -> None:
    if not results:
        raise ValueError("at least one --input is required")
    cells = [(result.task, result.policy_seed) for result in results]
    if len(set(cells)) != len(cells):
        raise ValueError(f"duplicate task/policy-seed inputs: {cells}")

    by_task: dict[str, list[LoadedResult]] = defaultdict(list)
    for result in results:
        by_task[result.task].append(result)
    seeds_by_task = {
        task: {result.policy_seed for result in task_results}
        for task, task_results in by_task.items()
    }
    seed_sets = list(seeds_by_task.values())
    if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise ValueError(f"tasks do not share the same policy-seed set: {seeds_by_task}")

    for task, task_results in sorted(by_task.items()):
        reference: dict[tuple[int, int], tuple[Any, ...]] | None = None
        reference_seed = -1
        reference_contract: tuple[Any, ...] | None = None
        for result in sorted(task_results, key=lambda item: item.policy_seed):
            contract = _shared_input_contract(result)
            if reference_contract is None:
                reference_contract = contract
            elif contract != reference_contract:
                raise ValueError(
                    f"task {task!r} data/split/vision input contract differs across "
                    f"policy seeds {reference_seed} and {result.policy_seed}"
                )
            identities = {
                (int(row["episode"]), int(row["frame"])): _identity_payload(row)
                for row in result.rows
            }
            if reference is None:
                reference = identities
                reference_seed = result.policy_seed
            elif identities != reference:
                missing = sorted(set(reference).difference(identities))[:3]
                extra = sorted(set(identities).difference(reference))[:3]
                differing = [
                    identity
                    for identity in sorted(set(reference).intersection(identities))
                    if reference[identity] != identities[identity]
                ][:3]
                raise ValueError(
                    f"task {task!r} physical episode/frame identities differ between "
                    f"policy seeds {reference_seed} and {result.policy_seed}; "
                    f"missing={missing}, extra={extra}, metadata_mismatch={differing}"
                )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.size < 2 or left_values.shape != right_values.shape:
        raise ValueError("Spearman inputs must have equal size >= 2")
    left_rank = _average_ranks(left_values)
    right_rank = _average_ranks(right_values)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = float(
        np.sqrt(np.square(left_centered).sum() * np.square(right_centered).sum())
    )
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def _rows_in_stratum(
    rows: Sequence[Mapping[str, Any]], stratum: str
) -> list[Mapping[str, Any]]:
    if stratum == "all":
        return list(rows)
    clean = stratum == "clean"
    return [row for row in rows if bool(row["camera_clean"]) == clean]


def _expected_episode_cells(
    results: Sequence[LoadedResult], stratum: str
) -> dict[str, dict[int, tuple[int, ...]]]:
    expected: dict[str, dict[int, tuple[int, ...]]] = defaultdict(dict)
    for result in results:
        episodes = tuple(
            sorted(
                {
                    int(row["episode"])
                    for row in _rows_in_stratum(result.rows, stratum)
                }
            )
        )
        if not episodes:
            raise ValueError(
                f"stratum {stratum!r} is missing task={result.task} "
                f"policy_seed={result.policy_seed}"
            )
        expected[result.task][result.policy_seed] = episodes
    for task, seeds in expected.items():
        episode_sets = list(seeds.values())
        if any(episodes != episode_sets[0] for episodes in episode_sets[1:]):
            raise ValueError(
                f"stratum {stratum!r} does not share physical episodes across "
                f"policy seeds for task={task}"
            )
    return {task: dict(seeds) for task, seeds in expected.items()}


def _episode_metric_values(
    results: Sequence[LoadedResult], stratum: str, metric: str
) -> dict[str, dict[int, np.ndarray]]:
    expected = _expected_episode_cells(results, stratum)
    values: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    for result in results:
        grouped: dict[int, list[float]] = defaultdict(list)
        for row in _rows_in_stratum(result.rows, stratum):
            value = _finite_optional(row[metric])
            if value is not None:
                grouped[int(row["episode"])].append(value)
        if not grouped:
            continue
        expected_episodes = expected[result.task][result.policy_seed]
        if tuple(sorted(grouped)) != expected_episodes:
            missing = sorted(set(expected_episodes) - set(grouped))[:5]
            raise ValueError(
                f"metric {metric!r} has incomplete episode support in stratum "
                f"{stratum!r}: task={result.task} seed={result.policy_seed} "
                f"missing={missing}"
            )
        values[result.task][result.policy_seed] = np.asarray(
            [float(np.mean(grouped[episode])) for episode in expected_episodes],
            dtype=np.float64,
        )
    if not values:
        return {}
    expected_cells = {
        (task, policy_seed)
        for task, seeds in expected.items()
        for policy_seed in seeds
    }
    actual_cells = {
        (task, policy_seed)
        for task, seeds in values.items()
        for policy_seed in seeds
    }
    if actual_cells != expected_cells:
        raise ValueError(
            f"metric {metric!r} has incomplete task/policy-seed support in "
            f"stratum {stratum!r}: missing={sorted(expected_cells - actual_cells)}"
        )
    return {task: dict(seeds) for task, seeds in values.items()}


def _episode_association_values(
    results: Sequence[LoadedResult], stratum: str, target: str
) -> dict[str, dict[int, np.ndarray]]:
    expected = _expected_episode_cells(results, stratum)
    values: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    for result in results:
        grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in _rows_in_stratum(result.rows, stratum):
            if (
                _finite_optional(row["delta_epe"]) is not None
                and _finite_optional(row[target]) is not None
            ):
                grouped[int(row["episode"])].append(row)
        episode_values = []
        for episode in sorted(grouped):
            rows = grouped[episode]
            if len(rows) < 2:
                continue
            episode_values.append(
                _spearman(
                    [float(row["delta_epe"]) for row in rows],
                    [float(row[target]) for row in rows],
                )
            )
        if episode_values:
            available_episodes = tuple(
                episode for episode in sorted(grouped) if len(grouped[episode]) >= 2
            )
            if available_episodes != expected[result.task][result.policy_seed]:
                missing = sorted(
                    set(expected[result.task][result.policy_seed])
                    - set(available_episodes)
                )[:5]
                raise ValueError(
                    f"association delta_epe/{target} has incomplete episode support "
                    f"in stratum {stratum!r}: task={result.task} "
                    f"seed={result.policy_seed} missing={missing}"
                )
            values[result.task][result.policy_seed] = np.asarray(
                episode_values, dtype=np.float64
            )
    if not values:
        return {}
    expected_cells = {
        (task, policy_seed)
        for task, seeds in expected.items()
        for policy_seed in seeds
    }
    actual_cells = {
        (task, policy_seed)
        for task, seeds in values.items()
        for policy_seed in seeds
    }
    if actual_cells != expected_cells:
        raise ValueError(
            f"association delta_epe/{target} has incomplete task/policy-seed "
            f"support in stratum {stratum!r}: missing={sorted(expected_cells - actual_cells)}"
        )
    return {task: dict(seeds) for task, seeds in values.items()}


def _bootstrap_mean(
    values: np.ndarray, samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    draws = np.empty(samples, dtype=np.float64)
    batch_size = min(samples, 5_000)
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        draws[start : start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def _bootstrap_task(
    arrays: Mapping[int, np.ndarray], samples: int, rng: np.random.Generator
) -> np.ndarray:
    seeds = tuple(sorted(arrays))
    episode_counts = {np.asarray(arrays[seed]).size for seed in seeds}
    if len(episode_counts) != 1:
        raise ValueError("crossed bootstrap requires shared episode counts across seeds")
    episode_count = next(iter(episode_counts))
    stacked = np.stack(
        [np.asarray(arrays[seed], dtype=np.float64) for seed in seeds]
    )
    selected = rng.integers(0, len(seeds), size=(samples, len(seeds)))
    episode_indices = rng.integers(
        0, episode_count, size=(samples, episode_count)
    )
    result = np.empty((samples, len(seeds)), dtype=np.float64)
    for position in range(len(seeds)):
        selected_values = stacked[selected[:, position]]
        result[:, position] = np.take_along_axis(
            selected_values, episode_indices, axis=1
        ).mean(axis=1)
    return result.mean(axis=1)


def summarize_episode_values(
    values: Mapping[str, Mapping[int, np.ndarray]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not values:
        return {"available": False}
    rng = np.random.default_rng(bootstrap_seed)
    tasks = tuple(sorted(values))
    per_task: dict[str, Any] = {}
    task_distributions = []
    for task in tasks:
        arrays = values[task]
        if not arrays:
            raise ValueError(f"task {task!r} has no policy seeds")
        per_seed: dict[str, Any] = {}
        for policy_seed in sorted(arrays):
            episode_values = np.asarray(arrays[policy_seed], dtype=np.float64)
            if (
                episode_values.ndim != 1
                or episode_values.size == 0
                or not np.all(np.isfinite(episode_values))
            ):
                raise ValueError(
                    f"task={task} seed={policy_seed} episode values must be finite"
                )
            per_seed[str(policy_seed)] = {
                "episodes": int(episode_values.size),
                "estimate": float(episode_values.mean()),
                "episode_bootstrap_95_ci": list(
                    _bootstrap_mean(episode_values, bootstrap_samples, rng)
                ),
            }
        task_distribution = _bootstrap_task(arrays, bootstrap_samples, rng)
        task_distributions.append(task_distribution)
        seed_estimates = [item["estimate"] for item in per_seed.values()]
        low, high = np.quantile(task_distribution, [0.025, 0.975])
        per_task[task] = {
            "policy_seeds": sorted(arrays),
            "estimate": float(np.mean(seed_estimates)),
            "hierarchical_bootstrap_95_ci": [float(low), float(high)],
            "per_seed": per_seed,
        }
    distribution = np.stack(task_distributions).mean(axis=0)
    low, high = np.quantile(distribution, [0.025, 0.975])
    estimate = float(np.mean([value["estimate"] for value in per_task.values()]))
    return {
        "available": True,
        "estimate": estimate,
        "hierarchical_bootstrap_95_ci": [float(low), float(high)],
        "tasks": list(tasks),
        "per_task": per_task,
    }


def summarize_results(
    results: Sequence[LoadedResult],
    *,
    bootstrap_samples: int = 100_000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    validate_collection(results)
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    strata: dict[str, Any] = {}
    offset = 0
    for stratum in STRATA:
        selected_rows = [
            row
            for result in results
            for row in _rows_in_stratum(result.rows, stratum)
        ]
        episodes = {
            (result.task, result.policy_seed, int(row["episode"]))
            for result in results
            for row in _rows_in_stratum(result.rows, stratum)
        }
        metrics: dict[str, Any] = {}
        for metric in METRICS:
            metrics[metric] = summarize_episode_values(
                _episode_metric_values(results, stratum, metric),
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + offset,
            )
            offset += 1
        absolute_components: dict[str, Any] = {}
        for metric in ABSOLUTE_METRICS:
            absolute_components[metric] = summarize_episode_values(
                _episode_metric_values(results, stratum, metric),
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + offset,
            )
            offset += 1
        support_summary = summarize_episode_values(
            _episode_metric_values(
                results, stratum, "common_support_coverage"
            ),
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + offset,
        )
        offset += 1
        associations: dict[str, Any] = {}
        for target in ASSOCIATION_TARGETS:
            name = f"spearman_delta_epe_vs_{target}"
            associations[name] = summarize_episode_values(
                _episode_association_values(results, stratum, target),
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + offset,
            )
            offset += 1
        base_anchors = int(
            sum(int(row["base_valid_anchors"]) for row in selected_rows)
        )
        common_anchors = int(
            sum(int(row["common_valid_anchors"]) for row in selected_rows)
        )
        fcpg_oob_anchors = int(
            sum(int(row["fcpg_oob_anchors"]) for row in selected_rows)
        )
        strata[stratum] = {
            "snapshots": len(selected_rows),
            "task_seed_episodes": len(episodes),
            "metrics": metrics,
            "absolute_components": absolute_components,
            "support": {
                "base_valid_anchors": base_anchors,
                "common_valid_anchors": common_anchors,
                "fcpg_oob_anchors": fcpg_oob_anchors,
                "pooled_common_support_coverage": (
                    common_anchors / base_anchors if base_anchors else None
                ),
                "episode_clustered_common_support_coverage": support_summary,
            },
            "associations": associations,
        }
    return {
        "schema_version": 1,
        "contract": {
            "input_runs": "full only (max_batches=0)",
            "window_reduction": "mean within physical episode before inference",
            "primary_estimand": (
                "fixed equal-task macro of equal-policy-seed means; each seed "
                "contributes its mean across physical episodes"
            ),
            "hierarchical_bootstrap": (
                "for each fixed task independently resample policy-seed indices and "
                "shared physical-episode identities; reuse each episode draw across "
                "selected seeds, then average tasks equally"
            ),
            "clean_definition": (
                "actual_camera_rotation_deg == 0 and "
                "actual_camera_translation_norm == 0"
            ),
        },
        "inputs": [
            {
                "path": str(result.path),
                "task": result.task,
                "policy_seed": result.policy_seed,
                "snapshots": len(result.rows),
                "physical_episodes": len(
                    {int(row["episode"]) for row in result.rows}
                ),
            }
            for result in sorted(results, key=lambda item: (item.task, item.policy_seed))
        ],
        "bootstrap": {"samples": bootstrap_samples, "seed": bootstrap_seed},
        "positive_delta_definition": (
            "positive means FCPG improves EPE or moves the frozen policy output "
            "closer to the same checkpoint's oracle-grounded output"
        ),
        "strata": strata,
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Action-Grounding Sensitivity Multi-Seed Summary",
        "",
        "Positive deltas favor FCPG. Point estimates weight tasks equally and policy seeds equally within task.",
    ]
    for stratum in STRATA:
        cell = summary["strata"][stratum]
        lines.extend(
            [
                "",
                f"## {stratum.title()}",
                "",
                "| metric | estimate | hierarchical 95% CI |",
                "| --- | ---: | --- |",
            ]
        )
        for metric in METRICS:
            value = cell["metrics"][metric]
            if not value["available"]:
                lines.append(f"| {metric} | unavailable | unavailable |")
                continue
            low, high = value["hierarchical_bootstrap_95_ci"]
            lines.append(
                f"| {metric} | {value['estimate']:+.6f} | [{low:+.6f}, {high:+.6f}] |"
            )
        lines.extend(
            [
                "",
                "### Absolute Components",
                "",
                "| component | estimate | hierarchical 95% CI |",
                "| --- | ---: | --- |",
            ]
        )
        for metric in ABSOLUTE_METRICS:
            value = cell["absolute_components"][metric]
            if not value["available"]:
                lines.append(f"| {metric} | unavailable | unavailable |")
                continue
            low, high = value["hierarchical_bootstrap_95_ci"]
            lines.append(
                f"| {metric} | {value['estimate']:.6f} | [{low:.6f}, {high:.6f}] |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=parse_input_spec,
        help="PATH (auto-discover payload task/seed) or TASK=POLICY_SEED=PATH",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    results = [load_result(spec) for spec in args.input]
    summary = summarize_results(
        results,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(summary), encoding="utf-8")
    print(render_markdown(summary))
    print(f"json: {args.output}")
    if args.markdown_output is not None:
        print(f"markdown: {args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
