"""Frozen statistics for the fresh P1-native complementary-observer Gate.

The evaluator supplies one already-aggregated MSE for every
``model seed x method x physical episode x horizon`` cell.  This module is
deliberately independent of model and rollout code: it validates an exact
paired panel, uses one shared physical-episode bootstrap draw matrix, and
applies the pre-registered per-seed and cross-seed decision rules.

Branches, hazards, anchors, and token dimensions must be reduced inside each
physical episode before rows reach this module.  Model seeds are replication
checks, not additional statistical units.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Any

import numpy as np


PROTOCOL = "fcpg_p1_native_complementary_fresh_gate_v1"
TASK = "pick-place-v3"
MODEL_SEEDS = (15801, 15802, 15803)
HORIZONS = (3, 4, 5, 6, 7, 8)
GAIN_SCHEDULE = (0.25, 0.50, 0.75, 0.75, 0.75, 0.75)
MINIMUM_PHYSICAL_EPISODES = 30

CANDIDATE = "complementary_factual"
MATCHED_NO_ACTION = "complementary_matched_no_action"
HOLD_LAST = "hold_last"
GEOMETRY_HOLD = "geometry_hold_exact_fk"
VISUAL_CV = "visual_cv_exact_fk"
WRONG_ACTIONS = tuple(
    f"complementary_wrong_action_cyclic_shift_{shift}" for shift in range(1, 8)
)
PRIMARY_BASELINES = (MATCHED_NO_ACTION, HOLD_LAST, GEOMETRY_HOLD)
NONLEARNING_BASELINES = (HOLD_LAST, GEOMETRY_HOLD, VISUAL_CV)
REQUIRED_METHODS = (
    CANDIDATE,
    MATCHED_NO_ACTION,
    HOLD_LAST,
    GEOMETRY_HOLD,
    VISUAL_CV,
    *WRONG_ACTIONS,
)

REQUIRED_ROW_FIELDS = (
    "protocol",
    "task",
    "model_seed",
    "method",
    "physical_episode_id",
    "horizon",
    "episode_horizon_mse",
    "statistical_unit",
    "dataset_sha256",
    "checkpoint_sha256",
)


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character lowercase SHA-256")
    if value != value.lower() or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a 64-character lowercase SHA-256")
    return value


def _finite_nonnegative(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _episode_id(value: object, *, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        raise TypeError(f"{label} must be a non-empty string or integer")
    result = str(value)
    if not result.strip():
        raise ValueError(f"{label} must be non-empty")
    return result


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_expected_provenance(
    expected_dataset_sha256: str,
    expected_checkpoint_sha256: Mapping[int, str],
) -> tuple[str, dict[int, str]]:
    dataset_sha = _sha256(expected_dataset_sha256, label="expected dataset SHA-256")
    if not isinstance(expected_checkpoint_sha256, Mapping):
        raise TypeError("expected_checkpoint_sha256 must be a mapping")
    observed_keys = {
        int(key)
        for key in expected_checkpoint_sha256
        if isinstance(key, Integral) and not isinstance(key, bool)
    }
    if observed_keys != set(MODEL_SEEDS) or len(expected_checkpoint_sha256) != len(
        MODEL_SEEDS
    ):
        raise ValueError(
            f"checkpoint provenance must contain exactly seeds {MODEL_SEEDS}"
        )
    checkpoints = {
        seed: _sha256(
            expected_checkpoint_sha256[seed],
            label=f"checkpoint SHA-256 for seed {seed}",
        )
        for seed in MODEL_SEEDS
    }
    if len(set(checkpoints.values())) != len(checkpoints):
        raise ValueError("the three model seeds must have distinct checkpoint SHA-256 values")
    return dataset_sha, checkpoints


def _normalize_panel(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_dataset_sha256: str,
    expected_checkpoint_sha256: Mapping[int, str],
) -> tuple[np.ndarray, tuple[str, ...], str, dict[int, str]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence of mappings")
    if not rows:
        raise ValueError("rows must be non-empty")
    dataset_sha, checkpoint_sha = _validate_expected_provenance(
        expected_dataset_sha256, expected_checkpoint_sha256
    )

    values: dict[tuple[int, str, str, int], float] = {}
    episodes: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"rows[{index}] must be a mapping")
        missing = [field for field in REQUIRED_ROW_FIELDS if field not in row]
        if missing:
            raise ValueError(f"rows[{index}] is missing fields: {missing}")
        if row["protocol"] != PROTOCOL:
            raise ValueError(f"rows[{index}] protocol mismatch")
        if row["task"] != TASK:
            raise ValueError(f"rows[{index}] task must be {TASK}")
        if row["statistical_unit"] != "physical_episode":
            raise ValueError(f"rows[{index}] statistical unit must be physical_episode")

        seed_value = row["model_seed"]
        if isinstance(seed_value, bool) or not isinstance(seed_value, Integral):
            raise TypeError(f"rows[{index}].model_seed must be an integer")
        seed = int(seed_value)
        if seed not in MODEL_SEEDS:
            raise ValueError(f"rows[{index}] contains unexpected model seed {seed}")
        method_value = row["method"]
        if not isinstance(method_value, str) or method_value not in REQUIRED_METHODS:
            raise ValueError(f"rows[{index}] contains unexpected method {method_value!r}")
        method = method_value
        episode = _episode_id(
            row["physical_episode_id"], label=f"rows[{index}].physical_episode_id"
        )
        horizon_value = row["horizon"]
        if isinstance(horizon_value, bool) or not isinstance(horizon_value, Integral):
            raise TypeError(f"rows[{index}].horizon must be an integer")
        horizon = int(horizon_value)
        if horizon not in HORIZONS:
            raise ValueError(f"rows[{index}] contains unexpected horizon H{horizon}")
        mse = _finite_nonnegative(
            row["episode_horizon_mse"],
            label=f"rows[{index}].episode_horizon_mse",
        )
        if _sha256(row["dataset_sha256"], label=f"rows[{index}] dataset SHA-256") != dataset_sha:
            raise ValueError(f"rows[{index}] dataset provenance mismatch")
        if (
            _sha256(
                row["checkpoint_sha256"],
                label=f"rows[{index}] checkpoint SHA-256",
            )
            != checkpoint_sha[seed]
        ):
            raise ValueError(f"rows[{index}] checkpoint provenance mismatch for seed {seed}")

        key = (seed, method, episode, horizon)
        if key in values:
            raise ValueError(f"duplicate Gate panel cell: {key!r}")
        values[key] = mse
        episodes.add(episode)

    episode_order = tuple(sorted(episodes))
    if len(episode_order) < MINIMUM_PHYSICAL_EPISODES:
        raise ValueError(
            "fresh Gate requires at least "
            f"{MINIMUM_PHYSICAL_EPISODES} physical episodes"
        )
    expected_cells = {
        (seed, method, episode, horizon)
        for seed in MODEL_SEEDS
        for method in REQUIRED_METHODS
        for episode in episode_order
        for horizon in HORIZONS
    }
    observed_cells = set(values)
    if observed_cells != expected_cells:
        missing = sorted(expected_cells - observed_cells, key=repr)
        extra = sorted(observed_cells - expected_cells, key=repr)
        raise ValueError(
            "incomplete fresh Gate panel: "
            f"missing={missing[:3]} ({len(missing)} total), "
            f"extra={extra[:3]} ({len(extra)} total)"
        )

    panel = np.empty(
        (
            len(MODEL_SEEDS),
            len(REQUIRED_METHODS),
            len(episode_order),
            len(HORIZONS),
        ),
        dtype=np.float64,
    )
    for seed_index, seed in enumerate(MODEL_SEEDS):
        for method_index, method in enumerate(REQUIRED_METHODS):
            for episode_index, episode in enumerate(episode_order):
                for horizon_index, horizon in enumerate(HORIZONS):
                    panel[seed_index, method_index, episode_index, horizon_index] = (
                        values[(seed, method, episode, horizon)]
                    )
    if not bool(np.isfinite(panel).all() and np.all(panel >= 0.0)):
        raise RuntimeError("validated Gate panel contains invalid values")
    return panel, episode_order, dataset_sha, checkpoint_sha


def _relative_effect(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> float:
    denominator = float(np.mean(reference))
    if denominator <= 0.0:
        raise ValueError("reference MSE must have a strictly positive mean")
    return float(1.0 - float(np.mean(candidate)) / denominator)


def _bootstrap_relative_effect(
    candidate: np.ndarray,
    reference: np.ndarray,
    draws: np.ndarray,
) -> np.ndarray:
    candidate_mean = candidate[draws].mean(axis=1)
    reference_mean = reference[draws].mean(axis=1)
    if bool(np.any(reference_mean <= 0.0)):
        raise ValueError("a bootstrap reference mean is not strictly positive")
    return 1.0 - candidate_mean / reference_mean


def _comparison_summary(
    candidate: np.ndarray,
    reference: np.ndarray,
    draws: np.ndarray,
) -> dict[str, object]:
    effects = _bootstrap_relative_effect(candidate, reference, draws)
    return {
        "orientation": "positive_means_complementary_factual_lower_MSE",
        "relative_improvement": _relative_effect(candidate, reference),
        "bootstrap_95_ci": [
            float(np.quantile(effects, 0.025)),
            float(np.quantile(effects, 0.975)),
        ],
    }


def _summarize_seed(
    seed: int,
    values: np.ndarray,
    draws: np.ndarray,
) -> dict[str, object]:
    method_index = {name: index for index, name in enumerate(REQUIRED_METHODS)}
    candidate_horizon = values[method_index[CANDIDATE]]
    candidate_overall = candidate_horizon.mean(axis=1)

    comparisons: dict[str, dict[str, object]] = {}
    for baseline in PRIMARY_BASELINES:
        reference = values[method_index[baseline]].mean(axis=1)
        comparisons[f"factual_vs_{baseline}"] = _comparison_summary(
            candidate_overall, reference, draws
        )
    for wrong in WRONG_ACTIONS:
        reference = values[method_index[wrong]].mean(axis=1)
        comparisons[f"factual_vs_{wrong}"] = _comparison_summary(
            candidate_overall, reference, draws
        )

    strongest_per_horizon: dict[str, dict[str, object]] = {}
    positive_horizons = 0
    for horizon_index, horizon in enumerate(HORIZONS):
        candidate = candidate_horizon[:, horizon_index]
        point_effects: dict[str, float] = {}
        bootstrap_effects: list[np.ndarray] = []
        for baseline in NONLEARNING_BASELINES:
            reference = values[method_index[baseline], :, horizon_index]
            point_effects[baseline] = _relative_effect(candidate, reference)
            bootstrap_effects.append(
                _bootstrap_relative_effect(candidate, reference, draws)
            )
        simultaneous = np.min(np.stack(bootstrap_effects, axis=1), axis=1)
        strongest_point = min(point_effects.values())
        if strongest_point > 0.0:
            positive_horizons += 1
        strongest_per_horizon[f"H{horizon}"] = {
            "per_baseline_relative_improvement": point_effects,
            "simultaneous_strongest_relative_improvement": strongest_point,
            "simultaneous_bootstrap_95_ci": [
                float(np.quantile(simultaneous, 0.025)),
                float(np.quantile(simultaneous, 0.975)),
            ],
            "simultaneous_one_sided_95_lower": float(
                np.quantile(simultaneous, 0.05)
            ),
        }

    primary_pass = {
        name: (
            float(result["relative_improvement"]) >= 0.10
            and float(result["bootstrap_95_ci"][0]) > 0.0  # type: ignore[index]
        )
        for name, result in comparisons.items()
        if name in {f"factual_vs_{baseline}" for baseline in PRIMARY_BASELINES}
    }
    wrong_pass = {
        f"factual_vs_{wrong}": (
            float(comparisons[f"factual_vs_{wrong}"]["relative_improvement"]) > 0.0
            and float(
                comparisons[f"factual_vs_{wrong}"]["bootstrap_95_ci"][0]  # type: ignore[index]
            )
            > 0.0
        )
        for wrong in WRONG_ACTIONS
    }
    h3 = strongest_per_horizon["H3"]
    h3_pass = (
        float(h3["simultaneous_strongest_relative_improvement"]) >= 0.0
        and float(h3["simultaneous_one_sided_95_lower"]) >= -0.05
    )
    horizon_pass = positive_horizons >= 5
    passed = all(primary_pass.values()) and all(wrong_pass.values()) and h3_pass and horizon_pass
    return {
        "model_seed": seed,
        "physical_episode_macro_mse": float(candidate_overall.mean()),
        "comparisons": comparisons,
        "strongest_nonlearning_per_horizon": strongest_per_horizon,
        "screen": {
            "primary_pass": primary_pass,
            "wrong_action_pass": wrong_pass,
            "positive_horizon_count": positive_horizons,
            "minimum_positive_horizons": 5,
            "horizon_pass": horizon_pass,
            "h3_point_nonnegative_and_one_sided_lower_at_least_minus_5pct": h3_pass,
            "passed": passed,
        },
    }


def summarize_complementary_fresh_gate(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_dataset_sha256: str,
    expected_checkpoint_sha256: Mapping[int, str],
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 15_891,
) -> dict[str, object]:
    """Validate and score one immutable fresh P1-native confirmation panel."""

    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, Integral):
        raise TypeError("bootstrap_samples must be an integer")
    bootstrap_samples = int(bootstrap_samples)
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, Integral):
        raise TypeError("bootstrap_seed must be an integer")
    bootstrap_seed = int(bootstrap_seed)
    if bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be non-negative")

    panel, episodes, dataset_sha, checkpoint_sha = _normalize_panel(
        rows,
        expected_dataset_sha256=expected_dataset_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.integers(
        0,
        len(episodes),
        size=(bootstrap_samples, len(episodes)),
        dtype=np.int64,
    )
    draw_digest = hashlib.sha256(np.ascontiguousarray(draws).tobytes()).hexdigest()
    per_seed = [
        _summarize_seed(seed, panel[index], draws)
        for index, seed in enumerate(MODEL_SEEDS)
    ]

    passing = [
        int(result["model_seed"])
        for result in per_seed
        if bool(result["screen"]["passed"])  # type: ignore[index]
    ]
    remaining = [result for result in per_seed if int(result["model_seed"]) not in passing]
    remaining_positive: dict[str, bool] = {}
    for result in remaining:
        seed = int(result["model_seed"])
        comparisons = result["comparisons"]
        remaining_positive[str(seed)] = all(
            float(comparisons[f"factual_vs_{baseline}"]["relative_improvement"]) > 0.0
            for baseline in (GEOMETRY_HOLD, MATCHED_NO_ACTION)
        )
    enough_seeds = len(passing) >= 2
    remaining_rule = all(remaining_positive.values())
    passed = enough_seeds and remaining_rule
    summary = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "task": TASK,
        "gain_schedule_H3_H8": list(GAIN_SCHEDULE),
        "dataset_sha256": dataset_sha,
        "checkpoint_sha256": {str(seed): checkpoint_sha[seed] for seed in MODEL_SEEDS},
        "physical_episode_count": len(episodes),
        "physical_episode_ids_sha256": _canonical_digest(list(episodes)),
        "bootstrap": {
            "unit": "physical_episode",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "shared_draw_matrix_sha256": draw_digest,
            "two_sided_95_quantiles": [0.025, 0.975],
            "h3_one_sided_95_lower_quantile": 0.05,
        },
        "per_seed": per_seed,
        "cross_seed_gate": {
            "minimum_passing_seeds": 2,
            "passing_seed_count": len(passing),
            "passing_seeds": passing,
            "remaining_seeds": [int(result["model_seed"]) for result in remaining],
            "remaining_seed_positive_vs_geometry_and_no_action": remaining_positive,
            "at_least_two_of_three_pass": enough_seeds,
            "remaining_positive_rule_passed": remaining_rule,
            "passed": passed,
            "decision": "Fresh-Go" if passed else "Fresh-No-Go",
        },
        "claim_boundary": {
            "known_drop_only": True,
            "automatic_validity_selector": False,
            "policy_rollout_authorized_if_passed": True,
            "cem_authorized": False,
            "ten_thousand_step_training_authorized": False,
        },
    }
    return summary


__all__ = [
    "CANDIDATE",
    "GAIN_SCHEDULE",
    "GEOMETRY_HOLD",
    "HOLD_LAST",
    "HORIZONS",
    "MATCHED_NO_ACTION",
    "MODEL_SEEDS",
    "NONLEARNING_BASELINES",
    "PRIMARY_BASELINES",
    "PROTOCOL",
    "REQUIRED_METHODS",
    "TASK",
    "VISUAL_CV",
    "WRONG_ACTIONS",
    "summarize_complementary_fresh_gate",
]
