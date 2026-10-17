#!/usr/bin/env python3
"""Audit accepted A4c physical and latent support by scheduled task phase."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


AUDIT_PROTOCOL = "fcpg_action_branch_a4c_phase_support_audit_v1"
SCHEDULE_PROTOCOLS = {
    "fcpg_action_branch_phase_schedule_a4c_v1",
    "fcpg_action_branch_phase_schedule_a4c_v2",
    "fcpg_action_branch_phase_schedule_a4c_v3",
    "fcpg_action_branch_phase_schedule_a4c_v4",
}
SCHEDULE_PROTOCOL = "fcpg_action_branch_phase_schedule_a4c_v4"
HORIZON = 8
QUANTILE_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)

_IDENTITY_KEYS = (
    "proposal_id",
    "task_name",
    "task_instance_index",
    "source_episode_id",
    "split",
)
_OPTIONAL_IDENTITY_KEYS = ("task_class_index",)
_REQUIRED_LEDGER_KEYS = (*_IDENTITY_KEYS, "proposal_raw_step", "accepted_bundle_index")
_REQUIRED_CACHE_KEYS = (*_IDENTITY_KEYS, "target_state")
_REQUIRED_LABEL_KEYS = (*_IDENTITY_KEYS, "future_physical_spread", "branch_valid")

JoinKey = tuple[str, int, int]


def load_schedule(path: str | Path) -> dict[str, Any]:
    """Load one phase schedule JSON object."""

    source = Path(path).expanduser()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("A4c phase schedule must contain one JSON object")
    return value


def load_npz(path: str | Path, *, label: str) -> dict[str, np.ndarray]:
    """Load an NPZ without permitting pickled arrays."""

    source = Path(path).expanduser()
    with np.load(source, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _nonnegative_json_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _schedule_lookup(schedule: Mapping[str, Any]) -> dict[JoinKey, dict[str, Any]]:
    if schedule.get("protocol") not in SCHEDULE_PROTOCOLS:
        raise ValueError("A4c phase schedule protocol mismatch")
    episodes = schedule.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("A4c phase schedule must contain a non-empty episodes list")

    lookup: dict[JoinKey, dict[str, Any]] = {}
    episode_identities: set[tuple[str, int]] = set()
    for episode_index, episode in enumerate(episodes):
        label = f"schedule episode {episode_index}"
        if not isinstance(episode, Mapping):
            raise TypeError(f"{label} must be an object")
        task = episode.get("task")
        if not isinstance(task, str) or not task.strip():
            raise TypeError(f"{label} task must be a non-empty string")
        instance = _nonnegative_json_integer(
            episode.get("task_instance"), f"{label} task_instance"
        )
        episode_identity = (task, instance)
        if episode_identity in episode_identities:
            raise ValueError(f"duplicate schedule episode {episode_identity}")
        episode_identities.add(episode_identity)

        steps = episode.get("proposal_steps")
        coverage = episode.get("coverage")
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"{label} proposal_steps must be a non-empty list")
        if not isinstance(coverage, Mapping):
            raise TypeError(f"{label} coverage must be an object")
        phases = coverage.get("selected_phase_names")
        if not isinstance(phases, list):
            raise TypeError(f"{label} selected_phase_names must be a list")
        if len(steps) != len(phases):
            raise ValueError(
                f"{label} proposal_steps and selected_phase_names length mismatch"
            )

        split = episode.get("split")
        if split is not None and (not isinstance(split, str) or not split.strip()):
            raise TypeError(f"{label} split must be a non-empty string")
        environment_seed = episode.get("environment_seed")
        if environment_seed is not None:
            environment_seed = _nonnegative_json_integer(
                environment_seed, f"{label} environment_seed"
            )
        task_class_index = episode.get("task_class_index")
        if task_class_index is not None:
            task_class_index = _nonnegative_json_integer(
                task_class_index, f"{label} task_class_index"
            )

        previous_step = -1
        for position, (step_value, phase_value) in enumerate(
            zip(steps, phases, strict=True)
        ):
            step = _nonnegative_json_integer(
                step_value, f"{label} proposal_steps[{position}]"
            )
            if step <= previous_step:
                raise ValueError(f"{label} proposal_steps must be sorted and unique")
            previous_step = step
            if not isinstance(phase_value, str) or not phase_value.strip():
                raise TypeError(
                    f"{label} selected_phase_names[{position}] must be non-empty"
                )
            key = (task, instance, step)
            if key in lookup:
                raise ValueError(f"duplicate scheduled proposal {key}")
            lookup[key] = {
                "phase": phase_value,
                "split": split,
                "environment_seed": environment_seed,
                "task_class_index": task_class_index,
            }
    return lookup


def _required_keys(
    arrays: Mapping[str, np.ndarray], required: Sequence[str], label: str
) -> None:
    if not isinstance(arrays, Mapping):
        raise TypeError(f"{label} must be a mapping")
    missing = sorted(set(required) - set(arrays))
    if missing:
        raise ValueError(f"{label} is missing keys: {missing}")
    for name, value in arrays.items():
        if not isinstance(name, str):
            raise TypeError(f"{label} keys must be strings")
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{label} {name} must be a numpy array")


def _aligned_leading_axes(
    arrays: Mapping[str, np.ndarray], count: int, label: str
) -> None:
    for name, value in arrays.items():
        if value.ndim == 0 or value.shape[0] != count:
            raise ValueError(f"{label} {name} must have leading axis {count}")


def _vector(
    arrays: Mapping[str, np.ndarray], name: str, count: int, label: str
) -> np.ndarray:
    value = arrays[name]
    if value.shape != (count,):
        raise ValueError(f"{label} {name} must have shape ({count},)")
    return value


def _text_vector(
    arrays: Mapping[str, np.ndarray], name: str, count: int, label: str
) -> np.ndarray:
    value = _vector(arrays, name, count, label)
    if value.dtype.kind not in {"S", "U"}:
        raise TypeError(f"{label} {name} must contain fixed-width strings")
    result = value.astype(str, copy=False)
    if any(not item.strip() for item in result):
        raise ValueError(f"{label} {name} values must be non-empty")
    return result


def _identifier_vector(
    arrays: Mapping[str, np.ndarray], name: str, count: int, label: str
) -> np.ndarray:
    value = _vector(arrays, name, count, label)
    if value.dtype.kind not in {"S", "U", "i", "u"}:
        raise TypeError(f"{label} {name} must contain strings or integers")
    result = value.astype(str)
    if any(not item.strip() for item in result):
        raise ValueError(f"{label} {name} values must be non-empty")
    return result


def _integer_vector(
    arrays: Mapping[str, np.ndarray],
    name: str,
    count: int,
    label: str,
    *,
    minimum: int,
) -> np.ndarray:
    value = _vector(arrays, name, count, label)
    if value.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{label} {name} must contain integers")
    if value.dtype.kind == "u" and bool(
        np.any(value > np.iinfo(np.int64).max)
    ):
        raise ValueError(f"{label} {name} exceeds the supported integer range")
    result = value.astype(np.int64, copy=False)
    if bool(np.any(result < minimum)):
        raise ValueError(f"{label} {name} must be at least {minimum}")
    return result


def _identity(
    arrays: Mapping[str, np.ndarray], count: int, label: str
) -> dict[str, np.ndarray]:
    result = {
        "proposal_id": _identifier_vector(arrays, "proposal_id", count, label),
        "task_name": _text_vector(arrays, "task_name", count, label),
        "task_instance_index": _integer_vector(
            arrays, "task_instance_index", count, label, minimum=0
        ),
        "source_episode_id": _identifier_vector(
            arrays, "source_episode_id", count, label
        ),
        "split": _text_vector(arrays, "split", count, label),
    }
    if len(set(result["proposal_id"].tolist())) != count:
        raise ValueError(f"{label} proposal_id values must be unique")
    if "task_class_index" in arrays:
        result["task_class_index"] = _integer_vector(
            arrays, "task_class_index", count, label, minimum=0
        )
    return result


def _ledger_schedule_join(
    schedule: Mapping[str, Any], ledger: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    label = "proposal ledger"
    _required_keys(ledger, _REQUIRED_LEDGER_KEYS, label)
    accepted_raw = ledger["accepted_bundle_index"]
    if accepted_raw.ndim != 1 or accepted_raw.shape[0] <= 0:
        raise ValueError("proposal ledger accepted_bundle_index must be a non-empty vector")
    count = int(accepted_raw.shape[0])
    _aligned_leading_axes(ledger, count, label)
    identity = _identity(ledger, count, label)
    accepted = _integer_vector(
        ledger, "accepted_bundle_index", count, label, minimum=-1
    )
    steps = _integer_vector(
        ledger, "proposal_raw_step", count, label, minimum=0
    )

    if "eligible" in ledger:
        eligible = _vector(ledger, "eligible", count, label)
        if eligible.dtype.kind != "b":
            raise TypeError("proposal ledger eligible must contain booleans")
        if not np.array_equal(eligible, accepted >= 0):
            raise ValueError(
                "proposal ledger eligible and accepted_bundle_index disagree"
            )

    environment_seed: np.ndarray | None = None
    if "environment_seed" in ledger:
        environment_seed = _integer_vector(
            ledger, "environment_seed", count, label, minimum=0
        )

    lookup = _schedule_lookup(schedule)
    phases = np.empty(count, dtype=object)
    seen: set[JoinKey] = set()
    for row_index in range(count):
        key = (
            str(identity["task_name"][row_index]),
            int(identity["task_instance_index"][row_index]),
            int(steps[row_index]),
        )
        if key in seen:
            raise ValueError(f"duplicate proposal ledger join key {key}")
        seen.add(key)
        scheduled = lookup.get(key)
        if scheduled is None:
            raise ValueError(f"ledger proposal has no scheduled phase: {key}")
        phases[row_index] = scheduled["phase"]

        if scheduled["split"] is not None:
            if str(identity["split"][row_index]) != scheduled["split"]:
                raise ValueError(f"ledger split differs from schedule for {key}")
        if scheduled["environment_seed"] is not None:
            if environment_seed is None:
                raise ValueError(
                    "proposal ledger lacks environment_seed required by schedule"
                )
            if int(environment_seed[row_index]) != scheduled["environment_seed"]:
                raise ValueError(
                    f"ledger environment_seed differs from schedule for {key}"
                )
        if scheduled["task_class_index"] is not None:
            task_class = identity.get("task_class_index")
            if task_class is None:
                raise ValueError(
                    "proposal ledger lacks task_class_index required by schedule"
                )
            if int(task_class[row_index]) != scheduled["task_class_index"]:
                raise ValueError(
                    f"ledger task_class_index differs from schedule for {key}"
                )
    return accepted, identity, phases.astype(str)


def _cache_measurements(
    cache: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
]:
    cache_label = "accepted state cache"
    labels_label = "preflight labels"
    _required_keys(cache, _REQUIRED_CACHE_KEYS, cache_label)
    _required_keys(labels, _REQUIRED_LABEL_KEYS, labels_label)

    target = cache["target_state"]
    if target.ndim < 4 or target.shape[0] <= 0:
        raise ValueError(
            "accepted state cache target_state must have non-empty shape [N,B,8,...]"
        )
    count, branch_count = int(target.shape[0]), int(target.shape[1])
    if branch_count < 2 or target.shape[2] != HORIZON:
        raise ValueError(
            "accepted state cache target_state must have shape [N,B,8,...] with B >= 2"
        )
    if target.dtype.kind != "f" or not np.isfinite(target).all():
        raise ValueError(
            "accepted state cache target_state must be finite floating point"
        )
    _aligned_leading_axes(cache, count, cache_label)
    _aligned_leading_axes(labels, count, labels_label)
    cache_identity = _identity(cache, count, cache_label)
    labels_identity = _identity(labels, count, labels_label)

    for name in _IDENTITY_KEYS:
        if not np.array_equal(cache_identity[name], labels_identity[name]):
            raise ValueError(
                f"accepted state cache and preflight labels {name} are not row aligned"
            )
    for name in _OPTIONAL_IDENTITY_KEYS:
        in_cache = name in cache_identity
        in_labels = name in labels_identity
        if in_cache != in_labels:
            raise ValueError(
                f"accepted state cache and preflight labels {name} presence differs"
            )
        if in_cache and not np.array_equal(
            cache_identity[name], labels_identity[name]
        ):
            raise ValueError(
                f"accepted state cache and preflight labels {name} are not row aligned"
            )

    physical = _vector(
        labels, "future_physical_spread", count, labels_label
    )
    if physical.dtype.kind != "f" or not np.isfinite(physical).all():
        raise ValueError(
            "preflight labels future_physical_spread must be finite floating point"
        )
    if bool(np.any(physical < 0.0)):
        raise ValueError(
            "preflight labels future_physical_spread must be non-negative"
        )
    physical = physical.astype(np.float64, copy=False)

    validity = labels["branch_valid"]
    valid_shapes = {(count, branch_count), (count, branch_count, HORIZON)}
    if validity.shape not in valid_shapes:
        raise ValueError(
            "preflight labels branch_valid must have shape [N,B] or [N,B,8]"
        )
    if validity.dtype.kind != "b":
        raise TypeError("preflight labels branch_valid must contain booleans")
    validity = validity.astype(bool, copy=False)

    branch_variance = np.var(
        target.astype(np.float64, copy=False), axis=1, dtype=np.float64
    )
    branch_variance = np.mean(
        branch_variance,
        axis=tuple(range(1, branch_variance.ndim)),
        dtype=np.float64,
    )
    if branch_variance.shape != (count,) or not np.isfinite(branch_variance).all():
        raise RuntimeError("latent branch variance reduction failed")

    h8_success: np.ndarray | None = None
    if "branch_outcome" in labels:
        outcome = labels["branch_outcome"]
        expected = (count, branch_count, HORIZON, 2)
        if outcome.shape != expected:
            raise ValueError(
                f"preflight labels branch_outcome must have shape {expected}"
            )
        if outcome.dtype.kind != "f" or not np.isfinite(outcome).all():
            raise ValueError(
                "preflight labels branch_outcome must be finite floating point"
            )
        h8_success = outcome[:, :, HORIZON - 1, 1].astype(
            np.float64, copy=False
        )
        if not bool(np.all((h8_success == 0.0) | (h8_success == 1.0))):
            raise ValueError(
                "preflight labels H8 branch success must contain only 0 or 1"
            )

    return cache_identity, physical, branch_variance, validity, h8_success


def _join(
    schedule: Mapping[str, Any],
    ledger: Mapping[str, np.ndarray],
    cache: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, set[str]], bool]:
    accepted, ledger_identity, phases = _ledger_schedule_join(schedule, ledger)
    (
        cache_identity,
        physical,
        branch_variance,
        validity,
        h8_success,
    ) = _cache_measurements(cache, labels)
    bundle_count = physical.shape[0]

    accepted_values = accepted[accepted >= 0]
    expected = np.arange(bundle_count, dtype=np.int64)
    if accepted_values.size != bundle_count or not np.array_equal(
        np.sort(accepted_values), expected
    ):
        raise ValueError(
            "accepted bundle indices must be contiguous, unique, and cover every cache row"
        )
    ledger_by_bundle = np.empty(bundle_count, dtype=np.int64)
    ledger_by_bundle[accepted_values] = np.flatnonzero(accepted >= 0)

    shared_identity = list(_IDENTITY_KEYS)
    for name in _OPTIONAL_IDENTITY_KEYS:
        in_ledger = name in ledger_identity
        in_cache = name in cache_identity
        if in_ledger != in_cache:
            raise ValueError(
                f"proposal ledger and accepted state cache {name} presence differs"
            )
        if in_ledger:
            shared_identity.append(name)
    for name in shared_identity:
        ordered = ledger_identity[name][ledger_by_bundle]
        if not np.array_equal(ordered, cache_identity[name]):
            raise ValueError(
                f"accepted bundle index does not align ledger and cache {name}"
            )

    phase_universe: dict[str, set[str]] = defaultdict(set)
    for row_index, phase in enumerate(phases):
        phase_universe[str(ledger_identity["task_name"][row_index])].add(str(phase))

    rows: list[dict[str, Any]] = []
    for bundle_index, ledger_index in enumerate(ledger_by_bundle):
        row = {
            "bundle_index": bundle_index,
            "proposal_id": str(cache_identity["proposal_id"][bundle_index]),
            "task": str(cache_identity["task_name"][bundle_index]),
            "task_instance": int(
                cache_identity["task_instance_index"][bundle_index]
            ),
            "source_episode_id": str(
                cache_identity["source_episode_id"][bundle_index]
            ),
            "split": str(cache_identity["split"][bundle_index]),
            "phase": str(phases[int(ledger_index)]),
            "future_physical_spread": float(physical[bundle_index]),
            "latent_branch_variance": float(branch_variance[bundle_index]),
            "valid_branch_fraction": float(np.mean(validity[bundle_index])),
        }
        if h8_success is not None:
            row["h8_success_fraction"] = float(
                np.mean(h8_success[bundle_index])
            )
        rows.append(row)
    return rows, phase_universe, h8_success is not None


def join_accepted_bundles(
    schedule: Mapping[str, Any],
    ledger: Mapping[str, np.ndarray],
    cache: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Join accepted ledger rows to cache, labels, and scheduled phase names."""

    rows, _phase_universe, _has_h8_success = _join(
        schedule, ledger, cache, labels
    )
    return rows


def _quantiles(values: np.ndarray) -> list[float] | None:
    if values.size == 0:
        return None
    return [
        float(value)
        for value in np.quantile(values, np.asarray(QUANTILE_LEVELS))
    ]


def _summary(
    rows: Sequence[Mapping[str, Any]], *, has_h8_success: bool
) -> dict[str, Any]:
    physical = np.asarray(
        [row["future_physical_spread"] for row in rows], dtype=np.float64
    )
    latent = np.asarray(
        [row["latent_branch_variance"] for row in rows], dtype=np.float64
    )
    result: dict[str, Any] = {
        "bundle_count": len(rows),
        "episode_count": len(
            {
                (
                    str(row["task"]),
                    int(row["task_instance"]),
                    str(row["source_episode_id"]),
                    str(row["split"]),
                )
                for row in rows
            }
        ),
        "future_physical_spread_quantiles": _quantiles(physical),
        "future_physical_spread_nonzero_fraction": (
            float(np.mean(physical > 0.0)) if physical.size else None
        ),
        "latent_branch_variance_quantiles": _quantiles(latent),
        "latent_branch_variance_nonzero_fraction": (
            float(np.mean(latent > 0.0)) if latent.size else None
        ),
        "valid_branch_fraction": (
            float(np.mean([row["valid_branch_fraction"] for row in rows]))
            if rows
            else None
        ),
    }
    if has_h8_success:
        result["h8_success_fraction"] = (
            float(np.mean([row["h8_success_fraction"] for row in rows]))
            if rows
            else None
        )
    return result


def build_report(
    schedule: Mapping[str, Any],
    ledger: Mapping[str, np.ndarray],
    cache: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Return a deterministic JSON-compatible A4c phase-support report."""

    rows, phase_universe, has_h8_success = _join(
        schedule, ledger, cache, labels
    )
    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_by_task_phase: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        task = str(row["task"])
        phase = str(row["phase"])
        rows_by_task[task].append(row)
        rows_by_task_phase[(task, phase)].append(row)

    tasks: dict[str, Any] = {}
    for task in sorted(phase_universe):
        task_summary = _summary(
            rows_by_task[task], has_h8_success=has_h8_success
        )
        task_summary["phases"] = {
            phase: _summary(
                rows_by_task_phase[(task, phase)],
                has_h8_success=has_h8_success,
            )
            for phase in sorted(phase_universe[task])
        }
        tasks[task] = task_summary

    return {
        "schema_version": 1,
        "protocol": AUDIT_PROTOCOL,
        "quantile_levels": list(QUANTILE_LEVELS),
        "overall": _summary(rows, has_h8_success=has_h8_success),
        "tasks": tasks,
    }


def run(
    schedule_path: str | Path,
    ledger_path: str | Path,
    cache_path: str | Path,
    labels_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Load all inputs, build the audit, and write deterministic JSON."""

    report = build_report(
        load_schedule(schedule_path),
        load_npz(ledger_path, label="proposal ledger"),
        load_npz(cache_path, label="accepted state cache"),
        load_npz(labels_path, label="preflight labels"),
    )
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--phase-schedule", "--schedule", dest="schedule", type=Path, required=True
    )
    parser.add_argument(
        "--proposal-ledger", "--ledger", dest="ledger", type=Path, required=True
    )
    parser.add_argument(
        "--state-cache", "--cache", dest="cache", type=Path, required=True
    )
    parser.add_argument(
        "--preflight-labels", "--labels", dest="labels", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run(
        args.schedule, args.ledger, args.cache, args.labels, args.output
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
