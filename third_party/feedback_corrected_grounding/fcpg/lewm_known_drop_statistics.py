"""Paired episode statistics for the known-drop policy-token control Gate.

The module deliberately contains no model or rollout code.  It validates a
complete long-table method panel, reduces repeated observations within each
physical episode, and then gives every task equal weight.  Bootstrap sampling
uses the physical episode as the independent unit within each task.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from numbers import Integral, Real

import numpy as np


PROTOCOL = "lewm_known_drop_paired_statistics_v1"
REQUIRED_FIELDS = (
    "method",
    "task",
    "model_seed",
    "physical_episode",
    "drop_duration",
    "success",
    "action_distance",
)


PairedKey = tuple[str, int, Hashable, float]


def _finite_real(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    result = value.strip()
    if not result:
        raise ValueError(f"{label} must be non-empty")
    return result


def _episode_identity(value: object, *, label: str) -> Hashable:
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        raise TypeError(f"{label} must be a string or integer")
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{label} must be non-empty")
        return ("str", value)
    return ("int", int(value))


def _validate_method_name(value: str, *, label: str) -> str:
    return _nonempty_string(value, label=label)


def _normalize_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence of mappings")
    if not rows:
        raise ValueError("rows must be non-empty")

    normalized: list[dict[str, object]] = []
    observed_methods: set[str] = set()
    seen: set[tuple[PairedKey, str]] = set()
    panel: dict[PairedKey, set[str]] = defaultdict(set)

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"rows[{index}] must be a mapping")
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"rows[{index}] is missing fields: {missing}")

        method = _nonempty_string(row["method"], label=f"rows[{index}].method")
        task = _nonempty_string(row["task"], label=f"rows[{index}].task")
        model_seed_value = row["model_seed"]
        if isinstance(model_seed_value, bool) or not isinstance(
            model_seed_value, Integral
        ):
            raise TypeError(f"rows[{index}].model_seed must be an integer")
        model_seed = int(model_seed_value)
        episode = _episode_identity(
            row["physical_episode"],
            label=f"rows[{index}].physical_episode",
        )
        drop_duration = _finite_real(
            row["drop_duration"], label=f"rows[{index}].drop_duration"
        )
        if drop_duration < 0.0:
            raise ValueError(f"rows[{index}].drop_duration must be non-negative")
        success = _finite_real(row["success"], label=f"rows[{index}].success")
        if not 0.0 <= success <= 1.0:
            raise ValueError(f"rows[{index}].success must lie in [0, 1]")
        action_distance = _finite_real(
            row["action_distance"], label=f"rows[{index}].action_distance"
        )
        if action_distance < 0.0:
            raise ValueError(f"rows[{index}].action_distance must be non-negative")

        paired_key: PairedKey = (task, model_seed, episode, drop_duration)
        identity = (paired_key, method)
        if identity in seen:
            raise ValueError(
                "duplicate method row for paired key: "
                f"method={method!r}, key={paired_key!r}"
            )
        seen.add(identity)
        panel[paired_key].add(method)
        observed_methods.add(method)
        normalized.append(
            {
                "method": method,
                "task": task,
                "model_seed": model_seed,
                "physical_episode": episode,
                "drop_duration": drop_duration,
                "success": success,
                "action_distance": action_distance,
                "paired_key": paired_key,
            }
        )

    expected_panel = tuple(sorted(observed_methods))
    expected_set = set(expected_panel)
    for paired_key, methods in panel.items():
        if methods != expected_set:
            missing = sorted(expected_set - methods)
            extra = sorted(methods - expected_set)
            raise ValueError(
                "incomplete method panel for paired key "
                f"{paired_key!r}: missing={missing}, extra={extra}"
            )
    return normalized, expected_panel


def _episode_effects(
    rows: Sequence[Mapping[str, object]],
    *,
    candidate: str,
    baseline: str,
) -> dict[str, dict[Hashable, dict[str, object]]]:
    indexed = {
        (row["paired_key"], row["method"]): row
        for row in rows
    }
    pair_keys = sorted(
        {row["paired_key"] for row in rows},
        key=repr,
    )
    accumulated: dict[
        str,
        dict[Hashable, dict[str, list[float] | set[int] | set[float]]],
    ] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "success": [],
                "action_distance": [],
                "model_seeds": set(),
                "drop_durations": set(),
            }
        )
    )
    for paired_key in pair_keys:
        task, model_seed, episode, drop_duration = paired_key
        candidate_row = indexed[(paired_key, candidate)]
        baseline_row = indexed[(paired_key, baseline)]
        bucket = accumulated[task][episode]
        success_values = bucket["success"]
        distance_values = bucket["action_distance"]
        model_seeds = bucket["model_seeds"]
        drop_durations = bucket["drop_durations"]
        assert isinstance(success_values, list)
        assert isinstance(distance_values, list)
        assert isinstance(model_seeds, set)
        assert isinstance(drop_durations, set)
        success_values.append(
            float(candidate_row["success"]) - float(baseline_row["success"])
        )
        distance_values.append(
            float(baseline_row["action_distance"])
            - float(candidate_row["action_distance"])
        )
        model_seeds.add(int(model_seed))
        drop_durations.add(float(drop_duration))

    result: dict[str, dict[Hashable, dict[str, object]]] = {}
    for task, episodes in accumulated.items():
        result[task] = {}
        for episode, values in episodes.items():
            success_values = np.asarray(values["success"], dtype=np.float64)
            distance_values = np.asarray(values["action_distance"], dtype=np.float64)
            if success_values.size == 0 or distance_values.size != success_values.size:
                raise RuntimeError("empty or inconsistent episode aggregation")
            result[task][episode] = {
                "success_delta": float(np.mean(success_values)),
                "action_distance_improvement": float(np.mean(distance_values)),
                "pair_count": int(success_values.size),
                "model_seed_count": len(values["model_seeds"]),
                "drop_duration_count": len(values["drop_durations"]),
            }
    return result


def _percentile_interval(
    samples: np.ndarray,
    *,
    confidence: float,
) -> tuple[float, float]:
    tail = (1.0 - confidence) * 50.0
    low, high = np.percentile(samples, [tail, 100.0 - tail])
    return float(low), float(high)


def _summarize_comparison(
    rows: Sequence[Mapping[str, object]],
    *,
    candidate: str,
    baseline: str,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> dict[str, object]:
    effects = _episode_effects(rows, candidate=candidate, baseline=baseline)
    tasks = tuple(sorted(effects))
    rng = np.random.default_rng(seed)

    task_success_samples: list[np.ndarray] = []
    task_distance_samples: list[np.ndarray] = []
    per_task: dict[str, dict[str, object]] = {}
    for task in tasks:
        episodes = effects[task]
        episode_rows = list(episodes.values())
        success = np.asarray(
            [row["success_delta"] for row in episode_rows], dtype=np.float64
        )
        distance = np.asarray(
            [row["action_distance_improvement"] for row in episode_rows],
            dtype=np.float64,
        )
        draw = rng.integers(
            0,
            len(episode_rows),
            size=(bootstrap_samples, len(episode_rows)),
        )
        success_samples = np.mean(success[draw], axis=1)
        distance_samples = np.mean(distance[draw], axis=1)
        task_success_samples.append(success_samples)
        task_distance_samples.append(distance_samples)
        success_ci = _percentile_interval(success_samples, confidence=confidence)
        distance_ci = _percentile_interval(distance_samples, confidence=confidence)
        per_task[task] = {
            "physical_episode_count": int(len(episode_rows)),
            "paired_row_count": int(
                sum(int(row["pair_count"]) for row in episode_rows)
            ),
            "success_delta": float(np.mean(success)),
            "success_ci": [success_ci[0], success_ci[1]],
            "action_distance_improvement": float(np.mean(distance)),
            "action_distance_improvement_ci": [distance_ci[0], distance_ci[1]],
        }

    macro_success_samples = np.mean(np.stack(task_success_samples, axis=1), axis=1)
    macro_distance_samples = np.mean(np.stack(task_distance_samples, axis=1), axis=1)
    success_ci = _percentile_interval(macro_success_samples, confidence=confidence)
    distance_ci = _percentile_interval(macro_distance_samples, confidence=confidence)
    task_success = [float(per_task[task]["success_delta"]) for task in tasks]
    task_distance = [
        float(per_task[task]["action_distance_improvement"]) for task in tasks
    ]
    return {
        "candidate": candidate,
        "baseline": baseline,
        "task_count": len(tasks),
        "physical_episode_count": int(
            sum(len(task_episodes) for task_episodes in effects.values())
        ),
        "paired_row_count": int(
            sum(
                int(row["pair_count"])
                for task_episodes in effects.values()
                for row in task_episodes.values()
            )
        ),
        "success": {
            "estimand": "candidate_minus_baseline",
            "task_macro_delta": float(np.mean(task_success)),
            "ci": [success_ci[0], success_ci[1]],
        },
        "action_distance": {
            "estimand": "baseline_minus_candidate",
            "task_macro_improvement": float(np.mean(task_distance)),
            "ci": [distance_ci[0], distance_ci[1]],
        },
        "per_task": per_task,
    }


def summarize_known_drop_comparison(
    rows: Sequence[Mapping[str, object]],
    *,
    candidate: str,
    baseline: str,
    matched_no_action: str | None = None,
    oracle: str | None = None,
    bootstrap_samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, object]:
    """Validate and summarize paired known-drop rollout outcomes.

    Positive values always favor the first-named method: success uses
    ``candidate - baseline`` while action distance uses
    ``baseline - candidate`` because lower distance is better.
    """

    candidate = _validate_method_name(candidate, label="candidate")
    baseline = _validate_method_name(baseline, label="baseline")
    if candidate == baseline:
        raise ValueError("candidate and baseline must differ")
    if isinstance(bootstrap_samples, bool) or not isinstance(
        bootstrap_samples, Integral
    ):
        raise TypeError("bootstrap_samples must be an integer")
    bootstrap_samples = int(bootstrap_samples)
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    confidence = _finite_real(confidence, label="confidence")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    seed = int(seed)

    normalized, observed_methods = _normalize_rows(rows)
    requested = {candidate, baseline}
    if matched_no_action is not None:
        matched_no_action = _validate_method_name(
            matched_no_action, label="matched_no_action"
        )
        requested.add(matched_no_action)
    if oracle is not None:
        oracle = _validate_method_name(oracle, label="oracle")
        requested.add(oracle)
    missing = sorted(requested - set(observed_methods))
    if missing:
        raise ValueError(f"requested methods are absent from rows: {missing}")

    comparisons: dict[str, dict[str, object]] = {}
    comparison_specs: list[tuple[str, str, str]] = [
        ("candidate_vs_baseline", candidate, baseline)
    ]
    if matched_no_action is not None:
        if matched_no_action == candidate:
            raise ValueError("matched_no_action must differ from candidate")
        comparison_specs.append(
            ("candidate_vs_matched_no_action", candidate, matched_no_action)
        )
    if oracle is not None:
        if oracle == baseline:
            raise ValueError("oracle must differ from baseline")
        comparison_specs.append(("oracle_headroom_vs_baseline", oracle, baseline))

    for ordinal, (name, first, second) in enumerate(comparison_specs):
        comparisons[name] = _summarize_comparison(
            normalized,
            candidate=first,
            baseline=second,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            seed=seed + ordinal,
        )

    return {
        "protocol": PROTOCOL,
        "method_panel": list(observed_methods),
        "pairing_key": [
            "task",
            "model_seed",
            "physical_episode",
            "drop_duration",
        ],
        "aggregation": (
            "paired rows -> physical episode mean -> task mean -> equal task macro"
        ),
        "bootstrap": {
            "method": "task_stratified_paired_physical_episode_percentile",
            "samples": bootstrap_samples,
            "confidence": confidence,
            "seed": seed,
        },
        "comparisons": comparisons,
    }


__all__ = [
    "PROTOCOL",
    "REQUIRED_FIELDS",
    "summarize_known_drop_comparison",
]
