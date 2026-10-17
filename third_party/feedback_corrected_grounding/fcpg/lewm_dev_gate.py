"""Development-only decision gate for the anchor-visual LeWM ablation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np

from fcpg.lewm_anchor_ablation import (
    PAIRED_INNOVATION,
    PAIRED_VISUAL,
    PARAMETER_COMPATIBLE_TRIPLET_MODES,
    TRIPLET_CORRECTED,
    TRIPLET_NOMINAL,
    validate_anchor_visual_override,
)


CellKey = tuple[str, str]
EpisodeKey = tuple[int, str, int]
PRIMARY_HORIZONS = (1, 3, 5)
TC_CELL: CellKey = (TRIPLET_CORRECTED, TRIPLET_CORRECTED)
PI_CELL: CellKey = (PAIRED_INNOVATION, PAIRED_INNOVATION)
PV_CELL: CellKey = (PAIRED_VISUAL, PAIRED_VISUAL)
TN_CELL: CellKey = (TRIPLET_NOMINAL, TRIPLET_NOMINAL)
PROTOCOL_NAME = "fcpg_lewm_anchor_visual_dev_selection_v1"
PRIMARY_AGGREGATION = "normalized_endpoint_error_equal_horizon_equal_task"
PRIMARY_ESTIMAND = (
    "mean_task(1 - mean_episode,H candidate_ratio / "
    "mean_episode,H control_ratio)"
)
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_UNIT = "episode_block_within_task"
EXPECTED_TASK_COUNT = 4
ABSOLUTE_THRESHOLD_SCOPE = "overall_and_each_task"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 5120
EXPECTED_SCHEMA_VERSION = 1
EXPECTED_FULL_COUNTS = {"clean": 56, "mild": 112, "total": 168}
EXPECTED_UNSEEN_COUNTS = {"clean": 12, "mild": 29, "total": 41}
EXPECTED_TRAINING_CONTRACT = {
    "batch_size": 192,
    "latest_checkpoint_only": True,
    "optimizer_steps": 2000,
    "parameter_count": 19_152_816,
    "seed": 5120,
    "sigreg_weight": 0.02,
    "warmup_steps": 127,
}
EXPECTED_SECONDARY_CONTRASTS = [
    "paired_visual/paired_visual_vs_triplet_corrected/triplet_corrected",
    "paired_innovation/paired_innovation_vs_paired_visual/paired_visual",
    "triplet_corrected/triplet_corrected_vs_triplet_nominal/triplet_nominal",
]
SIGN_FLIP_ALTERNATIVE = "greater"


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _sequence(value: object, *, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence")
    return value


def _finite(value: object, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _protocol_value_matches(actual: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, int):
        return (
            isinstance(actual, int)
            and not isinstance(actual, bool)
            and actual == expected
        )
    if isinstance(expected, float):
        return isinstance(actual, float) and actual == expected
    if isinstance(expected, str):
        return isinstance(actual, str) and actual == expected
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _protocol_value_matches(actual_item, expected_item)
                for actual_item, expected_item in zip(
                    actual, expected, strict=True
                )
            )
        )
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(
                _protocol_value_matches(actual[key], expected_value)
                for key, expected_value in expected.items()
            )
        )
    return type(actual) is type(expected) and actual == expected


def _expect_protocol_value(
    block: Mapping[str, Any],
    field: str,
    expected: object,
    *,
    label: str,
) -> None:
    if not _protocol_value_matches(block.get(field), expected):
        raise ValueError(f"{label} must be exactly {expected!r}")


def _expect_exact_mapping(
    block: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if not _protocol_value_matches(block, expected):
        raise ValueError(f"{label} does not match the frozen protocol")


def _validate_early_protocol_boundaries(
    protocol: Mapping[str, Any],
) -> None:
    _expect_protocol_value(
        protocol,
        "schema_version",
        EXPECTED_SCHEMA_VERSION,
        label="protocol schema_version",
    )
    _expect_protocol_value(
        protocol,
        "protocol",
        PROTOCOL_NAME,
        label="anchor-visual development protocol",
    )
    _expect_exact_mapping(
        _mapping(protocol.get("claim_boundary"), label="claim boundary"),
        {
            "development_only": True,
            "posthoc_pilot_selection_protocol": True,
            "fresh_multiseed_confirmation_required": True,
            "fresh_outer_holdout_episodes_required": True,
            "old_gate_l1_test_reuse_forbidden": True,
            "single_seed_cannot_restore_lewm_as_a_paper_contribution": True,
        },
        label="claim boundary",
    )

    bootstrap = _mapping(protocol.get("bootstrap"), label="bootstrap protocol")
    _expect_protocol_value(
        bootstrap,
        "confidence",
        BOOTSTRAP_CONFIDENCE,
        label="bootstrap confidence",
    )
    primary = _mapping(protocol.get("primary"), label="primary protocol")
    _expect_protocol_value(
        primary,
        "minimum_relative_improvement",
        0.03,
        label="minimum primary effect",
    )
    _expect_protocol_value(
        primary,
        "paired_bootstrap_ci_lower_must_exceed",
        0.0,
        label="minimum primary CI lower bound",
    )
    clean = _mapping(
        protocol.get("clean_noninferiority"), label="clean noninferiority"
    )
    _expect_protocol_value(
        clean,
        "one_sided_confidence",
        BOOTSTRAP_CONFIDENCE,
        label="clean one-sided confidence",
    )
    _expect_protocol_value(
        clean,
        "ratio_upper_margin",
        1.05,
        label="clean ratio upper margin",
    )
    direction = _mapping(protocol.get("task_direction"), label="task direction")
    for field, expected, label in (
        ("task_count", EXPECTED_TASK_COUNT, "task count"),
        ("minimum_positive_tasks", 3, "minimum positive tasks"),
        (
            "maximum_single_task_relative_degradation",
            0.05,
            "maximum task degradation",
        ),
    ):
        _expect_protocol_value(direction, field, expected, label=label)

    representation = _mapping(
        protocol.get("representation_noncollapse"),
        label="representation_noncollapse",
    )
    _expect_protocol_value(
        representation,
        "absolute_threshold_scope",
        ABSOLUTE_THRESHOLD_SCOPE,
        label="representation absolute threshold scope",
    )
    for field, expected in (
        ("absolute_effective_rank_minimum", 8.0),
        ("absolute_participation_ratio_minimum", 4.0),
        ("overall_candidate_to_control_minimum", 0.9),
        ("per_task_candidate_to_control_minimum", 0.8),
    ):
        _expect_protocol_value(
            representation,
            field,
            expected,
            label=f"representation {field}",
        )

    secondary = _mapping(
        protocol.get("secondary_family"), label="secondary family"
    )
    _expect_protocol_value(
        secondary,
        "alpha",
        0.05,
        label="secondary alpha",
    )



def _validate_protocol(
    protocol: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], int]:
    _validate_early_protocol_boundaries(protocol)

    _expect_protocol_value(
        protocol,
        "schema_version",
        EXPECTED_SCHEMA_VERSION,
        label="protocol schema_version",
    )
    if protocol.get("protocol") != PROTOCOL_NAME:
        raise ValueError("unexpected anchor-visual development protocol")

    _expect_exact_mapping(
        _mapping(protocol.get("claim_boundary"), label="claim boundary"),
        {
            "development_only": True,
            "posthoc_pilot_selection_protocol": True,
            "fresh_multiseed_confirmation_required": True,
            "fresh_outer_holdout_episodes_required": True,
            "old_gate_l1_test_reuse_forbidden": True,
            "single_seed_cannot_restore_lewm_as_a_paper_contribution": True,
        },
        label="claim boundary",
    )
    _expect_exact_mapping(
        _mapping(protocol.get("decision"), label="decision protocol"),
        {
            "go_requires": [
                "primary_superiority",
                "clean_noninferiority",
                "task_direction",
                "representation_noncollapse",
                "corrector_train_unseen_directional_agreement",
            ],
            "no_go_action": "stop_the_correction_aware_input_route",
            "pass_action": (
                "promote_only_to_a_fresh_multiseed_outer_holdout_"
                "confirmatory_candidate"
            ),
        },
        label="decision protocol",
    )
    _expect_exact_mapping(
        _mapping(protocol.get("training_contract"), label="training contract"),
        EXPECTED_TRAINING_CONTRACT,
        label="training contract",
    )

    data = _mapping(protocol.get("data"), label="data protocol")
    _expect_exact_mapping(
        _mapping(data.get("full_validation_counts"), label="full counts"),
        EXPECTED_FULL_COUNTS,
        label="full validation counts",
    )
    _expect_exact_mapping(
        _mapping(
            data.get("corrector_train_unseen_counts"),
            label="corrector-train-unseen counts",
        ),
        EXPECTED_UNSEEN_COUNTS,
        label="corrector-train-unseen counts",
    )
    for field, expected in (
        ("development_split", "validation"),
        (
            "episode_manifest",
            "config/lewm/splits/gate_l1_camera_train_val_seed3072.json",
        ),
        (
            "episode_manifest_sha256",
            "f3448ef472bd93975126c203556094ad65229a714d58236a53b11b281768e299",
        ),
        (
            "episode_set_sha256",
            "3f6202c24d6795398ff05abba46b9f0b7902bfd91ae44516dc3482eff30251c5",
        ),
        (
            "corrector_train_unseen_slice",
            "config/lewm/splits/anchor_visual_corrector_train_unseen_seed5120.json",
        ),
        (
            "corrector_train_unseen_slice_sha256",
            "5209ef4dd7ee12b59ee6f09ff5dfa2a1e950c11006ab4abfa8ec05d5f26cfef7",
        ),
        ("mixed_corrector_exposure_disclosure_required", True),
    ):
        _expect_protocol_value(data, field, expected, label=f"data {field}")

    bootstrap = _mapping(protocol.get("bootstrap"), label="bootstrap protocol")
    confidence = _finite(
        bootstrap.get("confidence"),
        label="bootstrap confidence",
        positive=True,
    )
    if confidence != BOOTSTRAP_CONFIDENCE:
        raise ValueError(
            f"bootstrap confidence must be exactly {BOOTSTRAP_CONFIDENCE!r}"
        )
    _expect_protocol_value(
        bootstrap,
        "unit",
        BOOTSTRAP_UNIT,
        label="bootstrap unit",
    )
    _expect_protocol_value(
        bootstrap,
        "samples",
        BOOTSTRAP_SAMPLES,
        label="bootstrap samples",
    )
    _expect_protocol_value(
        bootstrap,
        "seed",
        BOOTSTRAP_SEED,
        label="bootstrap seed",
    )

    primary = _mapping(protocol.get("primary"), label="primary protocol")
    _expect_protocol_value(
        primary,
        "candidate",
        "/".join(PI_CELL),
        label="primary candidate",
    )
    _expect_protocol_value(
        primary,
        "control",
        "/".join(TC_CELL),
        label="primary control",
    )
    _expect_protocol_value(
        primary,
        "condition",
        "mild",
        label="primary condition",
    )
    _expect_protocol_value(
        primary,
        "aggregation",
        PRIMARY_AGGREGATION,
        label="primary aggregation",
    )
    _expect_protocol_value(
        primary,
        "estimand",
        PRIMARY_ESTIMAND,
        label="primary estimand",
    )
    raw_horizons = _sequence(
        primary.get("horizons"),
        label="primary horizons",
    )
    horizons = tuple(
        _integer(value, label="primary horizon", minimum=1)
        for value in raw_horizons
    )
    if horizons != PRIMARY_HORIZONS:
        raise ValueError(
            f"primary horizons must be exactly {list(PRIMARY_HORIZONS)!r}"
        )
    _expect_protocol_value(
        primary,
        "minimum_relative_improvement",
        0.03,
        label="minimum primary effect",
    )
    _expect_protocol_value(
        primary,
        "paired_bootstrap_ci_lower_must_exceed",
        0.0,
        label="minimum primary CI lower bound",
    )

    clean = _mapping(
        protocol.get("clean_noninferiority"), label="clean noninferiority"
    )
    _expect_protocol_value(
        clean,
        "candidate",
        "/".join(PI_CELL),
        label="clean candidate",
    )
    _expect_protocol_value(
        clean,
        "control",
        "/".join(TC_CELL),
        label="clean control",
    )
    one_sided_confidence = _finite(
        clean.get("one_sided_confidence"),
        label="clean one-sided confidence",
        positive=True,
    )
    if one_sided_confidence != BOOTSTRAP_CONFIDENCE:
        raise ValueError(
            "clean one-sided confidence must be exactly "
            f"{BOOTSTRAP_CONFIDENCE!r}"
        )
    _expect_protocol_value(
        clean,
        "ratio_upper_margin",
        1.05,
        label="clean ratio upper margin",
    )

    direction = _mapping(protocol.get("task_direction"), label="task direction")
    task_count = _integer(
        direction.get("task_count"),
        label="task count",
        minimum=1,
    )
    if task_count != EXPECTED_TASK_COUNT:
        raise ValueError(f"task count must be exactly {EXPECTED_TASK_COUNT}")
    _expect_protocol_value(
        direction,
        "minimum_positive_tasks",
        3,
        label="minimum positive tasks",
    )
    _expect_protocol_value(
        direction,
        "maximum_single_task_relative_degradation",
        0.05,
        label="maximum task degradation",
    )

    representation = _mapping(
        protocol.get("representation_noncollapse"),
        label="representation_noncollapse",
    )
    _expect_protocol_value(
        representation,
        "absolute_threshold_scope",
        ABSOLUTE_THRESHOLD_SCOPE,
        label="representation absolute threshold scope",
    )
    for field, expected in (
        ("absolute_effective_rank_minimum", 8.0),
        ("absolute_participation_ratio_minimum", 4.0),
        ("overall_candidate_to_control_minimum", 0.9),
        ("per_task_candidate_to_control_minimum", 0.8),
    ):
        _expect_protocol_value(
            representation,
            field,
            expected,
            label=f"representation {field}",
        )

    secondary = _mapping(
        protocol.get("secondary_family"),
        label="secondary family",
    )
    alpha = _finite(secondary.get("alpha"), label="secondary alpha")
    if not 0.0 < alpha < 1.0:
        raise ValueError("secondary alpha must satisfy 0 < alpha < 1")
    _expect_protocol_value(
        secondary,
        "alpha",
        0.05,
        label="secondary alpha",
    )
    _expect_protocol_value(
        secondary,
        "contrasts",
        EXPECTED_SECONDARY_CONTRASTS,
        label="secondary contrasts",
    )
    _expect_protocol_value(
        secondary,
        "correction",
        "holm",
        label="secondary correction",
    )
    _expect_protocol_value(
        secondary,
        "test",
        "paired_episode_sign_flip",
        label="secondary test",
    )
    _expect_protocol_value(
        secondary,
        "alternative",
        SIGN_FLIP_ALTERNATIVE,
        label="secondary alternative",
    )
    _expect_protocol_value(
        secondary,
        "cannot_rescue_primary_failure",
        True,
        label="secondary rescue policy",
    )
    return bootstrap, primary, clean, task_count


def validate_anchor_visual_dev_protocol(
    protocol: Mapping[str, Any],
) -> None:
    """Validate every frozen field without requiring matrix records."""

    _validate_protocol(protocol)


def _episode_key(block: Mapping[str, Any]) -> EpisodeKey:
    source = _integer(block.get("source_index"), label="source_index")
    task = block.get("task_name")
    if not isinstance(task, str) or not task:
        raise ValueError("task_name must be a non-empty string")
    episode = _integer(block.get("episode_id"), label="episode_id")
    return (source, task, episode)


def _endpoint_vector(block: Mapping[str, Any]) -> np.ndarray:
    endpoints = _mapping(block.get("endpoints"), label="episode endpoints")
    values = []
    for horizon in PRIMARY_HORIZONS:
        endpoint = _mapping(
            endpoints.get(str(horizon)), label=f"episode endpoint H={horizon}"
        )
        predicted = _finite(
            endpoint.get("predicted_mse"),
            label=f"H={horizon} predicted_mse",
        )
        if predicted < 0.0:
            raise ValueError(f"H={horizon} predicted_mse must be non-negative")
        identity = _finite(
            endpoint.get("hold_last_identity_mse"),
            label=f"H={horizon} hold_last_identity_mse",
            positive=True,
        )
        _integer(
            endpoint.get("element_count"),
            label=f"H={horizon} element_count",
            minimum=1,
        )
        reported_ratio = _finite(
            endpoint.get("ratio"), label=f"H={horizon} ratio"
        )
        if reported_ratio < 0.0:
            raise ValueError(f"H={horizon} ratio must be non-negative")
        recomputed = predicted / identity
        if not math.isclose(reported_ratio, recomputed, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"H={horizon} ratio disagrees with its MSE values")
        values.append(recomputed)
    return np.asarray(values, dtype=np.float64)


def _parse_cells(
    records: Sequence[Mapping[str, Any]],
) -> dict[CellKey, dict[EpisodeKey, dict[str, Any]]]:
    cells: dict[CellKey, dict[EpisodeKey, dict[str, Any]]] = {}
    for record in _sequence(records, label="records"):
        value = _mapping(record, label="matrix record")
        cell = (
            validate_anchor_visual_override(
                value.get("train_mode"), field_name="train_mode"  # type: ignore[arg-type]
            ),
            validate_anchor_visual_override(
                value.get("eval_mode"), field_name="eval_mode"  # type: ignore[arg-type]
            ),
        )
        if cell in cells:
            raise ValueError(f"duplicate matrix cell: {cell}")
        episodes: dict[EpisodeKey, dict[str, Any]] = {}
        for raw_block in _sequence(value.get("episode_blocks"), label="episode_blocks"):
            block = _mapping(raw_block, label="episode block")
            key = _episode_key(block)
            if key in episodes:
                raise ValueError(f"duplicate episode block: {key}")
            condition = block.get("camera_condition")
            if condition not in {"clean", "mild"}:
                raise ValueError(f"episode {key} has invalid camera_condition")
            unseen = block.get("corrector_train_unseen")
            if not isinstance(unseen, bool):
                raise TypeError(
                    f"episode {key} corrector_train_unseen must be boolean"
                )
            sequence_count = _integer(
                block.get("num_sequences"),
                label=f"episode {key} num_sequences",
                minimum=1,
            )
            endpoints = _mapping(block.get("endpoints"), label="episode endpoints")
            endpoint_counts = tuple(
                _integer(
                    _mapping(
                        endpoints.get(str(horizon)),
                        label=f"episode {key} endpoint H={horizon}",
                    ).get("element_count"),
                    label=f"episode {key} H={horizon} element_count",
                    minimum=1,
                )
                for horizon in PRIMARY_HORIZONS
            )
            if len(set(endpoint_counts)) != 1:
                raise ValueError(
                    f"episode {key} element_count must be equal across H=1,3,5"
                )
            if any(count % sequence_count != 0 for count in endpoint_counts):
                raise ValueError(
                    f"episode {key} element_count must be divisible by num_sequences"
                )
            hold_last_identity_mses = tuple(
                _finite(
                    _mapping(
                        endpoints.get(str(horizon)),
                        label=f"episode {key} endpoint H={horizon}",
                    ).get("hold_last_identity_mse"),
                    label=(
                        f"episode {key} H={horizon} hold_last_identity_mse"
                    ),
                    positive=True,
                )
                for horizon in PRIMARY_HORIZONS
            )
            episodes[key] = {
                "condition": condition,
                "corrector_train_unseen": unseen,
                "num_sequences": sequence_count,
                "endpoint_counts": endpoint_counts,
                "hold_last_identity_mses": hold_last_identity_mses,
                "normalized_endpoints": _endpoint_vector(block),
            }
        if not episodes:
            raise ValueError(f"matrix cell {cell} has no episodes")
        cells[cell] = episodes

    expected = {
        (train_mode, eval_mode)
        for train_mode in PARAMETER_COMPATIBLE_TRIPLET_MODES
        for eval_mode in PARAMETER_COMPATIBLE_TRIPLET_MODES
    }
    if set(cells) != expected:
        raise ValueError("development gate requires the complete 4x4 matrix")
    reference = cells[TN_CELL]
    for cell, episodes in cells.items():
        if set(episodes) != set(reference):
            raise ValueError(f"episode keys differ in matrix cell {cell}")
        for key, item in episodes.items():
            expected_item = reference[key]
            for field in (
                "condition",
                "corrector_train_unseen",
                "num_sequences",
                "endpoint_counts",
                "hold_last_identity_mses",
            ):
                if item[field] != expected_item[field]:
                    raise ValueError(
                        f"episode contract field {field} differs in cell {cell}: {key}"
                    )
    return cells


def _validate_episode_contract_counts(
    cells: Mapping[CellKey, Mapping[EpisodeKey, Mapping[str, Any]]],
) -> None:
    reference = cells[TN_CELL]
    full = {"clean": 0, "mild": 0, "total": 0}
    unseen = {"clean": 0, "mild": 0, "total": 0}
    tasks: set[str] = set()
    for key, item in reference.items():
        condition = str(item["condition"])
        tasks.add(key[1])
        full[condition] += 1
        full["total"] += 1
        if bool(item["corrector_train_unseen"]):
            unseen[condition] += 1
            unseen["total"] += 1
    if full != EXPECTED_FULL_COUNTS:
        raise ValueError(
            "development episode counts differ from the frozen protocol: "
            f"expected {EXPECTED_FULL_COUNTS}, got {full}"
        )
    if unseen != EXPECTED_UNSEEN_COUNTS:
        raise ValueError(
            "corrector-train-unseen episode counts differ from the frozen "
            f"protocol: expected {EXPECTED_UNSEEN_COUNTS}, got {unseen}"
        )
    if len(tasks) != EXPECTED_TASK_COUNT:
        raise ValueError(
            f"development episodes must contain exactly {EXPECTED_TASK_COUNT} tasks"
        )


def _grouped_pair(
    cells: Mapping[CellKey, Mapping[EpisodeKey, Mapping[str, Any]]],
    *,
    candidate: CellKey,
    control: CellKey,
    condition: str,
    corrector_train_unseen: bool | None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for key, candidate_item in sorted(cells[candidate].items()):
        if candidate_item["condition"] != condition:
            continue
        if (
            corrector_train_unseen is not None
            and candidate_item["corrector_train_unseen"]
            is not corrector_train_unseen
        ):
            continue
        control_item = cells[control][key]
        grouped.setdefault(key[1], []).append(
            (
                np.asarray(candidate_item["normalized_endpoints"], dtype=np.float64),
                np.asarray(control_item["normalized_endpoints"], dtype=np.float64),
            )
        )
    if not grouped:
        raise ValueError("requested development slice contains no episodes")
    result = {}
    for task, pairs in sorted(grouped.items()):
        candidate_values = np.stack([pair[0] for pair in pairs])
        control_values = np.stack([pair[1] for pair in pairs])
        if candidate_values.shape[1] != len(PRIMARY_HORIZONS):
            raise ValueError("endpoint vector has an unexpected horizon count")
        result[task] = (candidate_values, control_values)
    return result


def _paired_effect(
    grouped: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    samples = _integer(bootstrap_samples, label="bootstrap_samples", minimum=1)
    seed = _integer(bootstrap_seed, label="bootstrap_seed")
    task_rows = {}
    candidate_task_means = []
    control_task_means = []
    task_ratios = []
    for task, (candidate, control) in sorted(grouped.items()):
        if candidate.shape != control.shape:
            raise ValueError(f"candidate/control samples differ for task {task}")
        if candidate.ndim != 2 or candidate.shape[1] != len(PRIMARY_HORIZONS):
            raise ValueError(f"task {task} has an unexpected sample shape")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            candidate_mean = float(np.mean(candidate, dtype=np.float64))
            control_mean = float(np.mean(control, dtype=np.float64))
        if not math.isfinite(candidate_mean) or not math.isfinite(control_mean):
            raise ValueError(f"task means must remain finite for task {task}")
        if control_mean <= 0.0:
            raise ValueError(f"control mean must be positive for task {task}")
        task_ratio = candidate_mean / control_mean
        if not math.isfinite(task_ratio):
            raise ValueError(f"task ratio must remain finite for task {task}")
        task_rows[task] = {
            "candidate": candidate_mean,
            "control": control_mean,
            "episode_count": int(candidate.shape[0]),
            "relative_improvement": 1.0 - task_ratio,
        }
        candidate_task_means.append(candidate_mean)
        control_task_means.append(control_mean)
        task_ratios.append(task_ratio)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        candidate_macro = float(np.mean(candidate_task_means, dtype=np.float64))
        control_macro = float(np.mean(control_task_means, dtype=np.float64))
        candidate_to_control_ratio = float(np.mean(task_ratios, dtype=np.float64))
    if not all(
        math.isfinite(value)
        for value in (candidate_macro, control_macro, candidate_to_control_ratio)
    ):
        raise ValueError("macro gate statistics must remain finite")
    if control_macro <= 0.0:
        raise ValueError("control macro must be positive")
    point = 1.0 - candidate_to_control_ratio

    rng = np.random.default_rng(seed)
    ratios = np.zeros(samples, dtype=np.float64)
    for candidate, control in grouped.values():
        indices = rng.integers(
            0, candidate.shape[0], size=(samples, candidate.shape[0])
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            candidate_task_bootstrap = np.mean(candidate[indices], axis=(1, 2))
            control_task_bootstrap = np.mean(control[indices], axis=(1, 2))
        if not np.all(np.isfinite(candidate_task_bootstrap)) or not np.all(
            np.isfinite(control_task_bootstrap)
        ):
            raise ValueError("bootstrap task means must remain finite")
        if np.any(control_task_bootstrap <= 0.0):
            raise ValueError("bootstrap produced a non-positive control task mean")
        ratios += candidate_task_bootstrap / control_task_bootstrap
    ratios /= len(grouped)
    if not np.all(np.isfinite(ratios)):
        raise ValueError("bootstrap ratios must remain finite")
    if np.any(ratios < 0.0):
        raise ValueError("bootstrap produced a non-positive control mean")
    effects = 1.0 - ratios
    return {
        "estimand": PRIMARY_ESTIMAND,
        "candidate_macro": candidate_macro,
        "control_macro": control_macro,
        "relative_improvement": point,
        "relative_improvement_95_ci": [
            float(np.quantile(effects, 0.025)),
            float(np.quantile(effects, 0.975)),
        ],
        "candidate_to_control_ratio": candidate_to_control_ratio,
        "candidate_to_control_ratio_one_sided_95_upper": float(
            np.quantile(ratios, 0.95)
        ),
        "per_task": task_rows,
        "episode_count": int(sum(row[0].shape[0] for row in grouped.values())),
        "task_count": len(grouped),
    }


def _representation_gate(
    records: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    expected_task_count: int,
    expected_tasks: set[str],
) -> dict[str, Any]:
    by_cell = {
        (str(record["train_mode"]), str(record["eval_mode"])): record
        for record in records
    }
    candidate = _mapping(by_cell[PI_CELL].get("metrics"), label="PI metrics")
    control = _mapping(by_cell[TC_CELL].get("metrics"), label="TC metrics")
    candidate_rep = _mapping(candidate.get("representation"), label="PI representation")
    control_rep = _mapping(control.get("representation"), label="TC representation")
    thresholds = _mapping(
        protocol.get("representation_noncollapse"),
        label="representation_noncollapse",
    )
    overall_ratio_min = _finite(
        thresholds.get("overall_candidate_to_control_minimum"),
        label="overall rank ratio threshold",
        positive=True,
    )
    task_ratio_min = _finite(
        thresholds.get("per_task_candidate_to_control_minimum"),
        label="per-task rank ratio threshold",
        positive=True,
    )
    absolute = {
        "effective_rank": _finite(
            thresholds.get("absolute_effective_rank_minimum"),
            label="effective rank minimum",
            positive=True,
        ),
        "participation_ratio": _finite(
            thresholds.get("absolute_participation_ratio_minimum"),
            label="participation ratio minimum",
            positive=True,
        ),
    }

    def compare(
        candidate_row: Mapping[str, Any],
        control_row: Mapping[str, Any],
        ratio_min: float,
        *,
        scope: str,
    ) -> dict[str, Any]:
        candidate_samples = _integer(
            candidate_row.get("samples"),
            label=f"PI {scope} representation samples",
            minimum=1,
        )
        control_samples = _integer(
            control_row.get("samples"),
            label=f"TC {scope} representation samples",
            minimum=1,
        )
        if candidate_samples != control_samples:
            raise ValueError(
                f"PI and TC representation samples differ for {scope}"
            )
        metrics: dict[str, Any] = {"samples": candidate_samples}
        for name in ("effective_rank", "participation_ratio"):
            candidate_value = _finite(
                candidate_row.get(name),
                label=f"PI {scope} {name}",
            )
            if candidate_value < 0.0:
                raise ValueError(
                    f"PI {scope} {name} must be non-negative"
                )
            control_value = _finite(
                control_row.get(name),
                label=f"TC {scope} {name}",
                positive=True,
            )
            ratio = candidate_value / control_value
            metrics[name] = {
                "candidate": candidate_value,
                "control": control_value,
                "ratio": ratio,
                "passed": candidate_value >= absolute[name] and ratio >= ratio_min,
            }
        metrics["passed"] = all(metrics[name]["passed"] for name in absolute)
        return metrics

    overall = compare(
        _mapping(candidate_rep.get("overall"), label="PI overall representation"),
        _mapping(control_rep.get("overall"), label="TC overall representation"),
        overall_ratio_min,
        scope="overall",
    )
    candidate_tasks = _mapping(candidate_rep.get("per_task"), label="PI per-task representation")
    control_tasks = _mapping(control_rep.get("per_task"), label="TC per-task representation")
    if not candidate_tasks or not control_tasks:
        raise ValueError("representation per_task must not be empty")
    if set(candidate_tasks) != set(control_tasks):
        raise ValueError("PI and TC representation task sets differ")
    if len(candidate_tasks) != expected_task_count:
        raise ValueError(
            "representation gate must contain exactly "
            f"{expected_task_count} tasks"
        )
    if set(candidate_tasks) != expected_tasks:
        raise ValueError("representation task set must match primary task set")
    per_task = {
        task: compare(
            _mapping(candidate_tasks[task], label=f"PI representation {task}"),
            _mapping(control_tasks[task], label=f"TC representation {task}"),
            task_ratio_min,
            scope=f"task {task}",
        )
        for task in sorted(candidate_tasks)
    }
    return {
        "overall": overall,
        "per_task": per_task,
        "passed": bool(overall["passed"] and all(row["passed"] for row in per_task.values())),
    }


def _sign_flip_pvalue(
    grouped: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    samples: int,
    seed: int,
    alternative: str = SIGN_FLIP_ALTERNATIVE,
) -> float:
    """One-sided paired randomization test for a positive task-equal effect."""

    if alternative != SIGN_FLIP_ALTERNATIVE:
        raise ValueError(
            f"sign-flip alternative must be {SIGN_FLIP_ALTERNATIVE!r}"
        )
    if samples < 1:
        raise ValueError("sign-flip samples must be positive")
    if not grouped:
        raise ValueError("sign-flip test requires at least one task")
    rng = np.random.default_rng(seed)
    observed_ratios = []
    randomized_effect = np.zeros(samples, dtype=np.float64)
    for task, (candidate, control) in sorted(grouped.items()):
        if candidate.shape != control.shape:
            raise ValueError(f"candidate/control samples differ for task {task}")
        if candidate.ndim != 2 or candidate.shape[1] != len(PRIMARY_HORIZONS):
            raise ValueError(f"task {task} has an unexpected sample shape")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            control_mean = float(np.mean(control, dtype=np.float64))
            candidate_mean = float(np.mean(candidate, dtype=np.float64))
        if not math.isfinite(control_mean) or not math.isfinite(candidate_mean):
            raise ValueError(f"sign-flip means must remain finite for task {task}")
        if control_mean <= 0.0:
            raise ValueError(f"control mean must be positive for task {task}")
        observed_ratio = candidate_mean / control_mean
        if not math.isfinite(observed_ratio):
            raise ValueError(f"sign-flip ratio must remain finite for task {task}")
        observed_ratios.append(observed_ratio)

        # One draw swaps the complete H=1/3/5 endpoint block for an episode.
        swaps = rng.integers(
            0, 2, size=(samples, candidate.shape[0]), dtype=np.int8
        ).astype(bool)
        candidate_draw = np.where(
            swaps[:, :, None], control[None, :, :], candidate[None, :, :]
        )
        control_draw = np.where(
            swaps[:, :, None], candidate[None, :, :], control[None, :, :]
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            randomized_control = np.mean(control_draw, axis=(1, 2))
            randomized_candidate = np.mean(candidate_draw, axis=(1, 2))
        if not np.all(np.isfinite(randomized_control)) or not np.all(
            np.isfinite(randomized_candidate)
        ):
            raise ValueError("randomized task means must remain finite")
        if np.any(randomized_control <= 0.0):
            raise ValueError("randomization produced a non-positive control mean")
        randomized_effect += 1.0 - (
            randomized_candidate / randomized_control
        )
        if not np.all(np.isfinite(randomized_effect)):
            raise ValueError("randomized effects must remain finite")

    observed = 1.0 - float(np.mean(observed_ratios, dtype=np.float64))
    randomized_effect /= len(grouped)
    if not math.isfinite(observed) or not np.all(np.isfinite(randomized_effect)):
        raise ValueError("sign-flip statistics must remain finite")
    return float(
        (1 + np.count_nonzero(randomized_effect >= observed)) / (samples + 1)
    )


def _holm(rows: list[dict[str, Any]], *, alpha: float) -> list[dict[str, Any]]:
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["p_value"])
    adjusted = [1.0] * len(rows)
    running = 0.0
    continue_rejecting = True
    for rank, (original_index, row) in enumerate(ordered):
        multiplier = len(rows) - rank
        running = max(running, min(1.0, multiplier * float(row["p_value"])))
        adjusted[original_index] = running
        threshold = alpha / multiplier
        row["holm_threshold"] = threshold
        row["rejected"] = bool(continue_rejecting and row["p_value"] <= threshold)
        if not row["rejected"]:
            continue_rejecting = False
    for index, value in enumerate(adjusted):
        rows[index]["holm_adjusted_p"] = value
    return rows


def summarize_anchor_visual_dev_gate(
    records: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen single-seed development gate to all 16 matrix cells."""

    bootstrap, primary_protocol, clean_protocol, expected_task_count = (
        _validate_protocol(protocol)
    )
    samples = _integer(bootstrap.get("samples"), label="bootstrap samples", minimum=1)
    seed = _integer(bootstrap.get("seed"), label="bootstrap seed")
    cells = _parse_cells(records)
    _validate_episode_contract_counts(cells)
    primary_grouped = _grouped_pair(
        cells,
        candidate=PI_CELL,
        control=TC_CELL,
        condition="mild",
        corrector_train_unseen=None,
    )
    if len(primary_grouped) != expected_task_count:
        raise ValueError(
            f"primary slice must contain exactly {expected_task_count} tasks"
        )
    primary = _paired_effect(
        primary_grouped,
        bootstrap_samples=samples,
        bootstrap_seed=seed,
    )
    minimum_effect = _finite(
        primary_protocol.get("minimum_relative_improvement"),
        label="minimum primary effect",
    )
    minimum_ci = _finite(
        primary_protocol.get("paired_bootstrap_ci_lower_must_exceed"),
        label="minimum primary CI lower bound",
    )
    primary["passed"] = bool(
        primary["relative_improvement"] >= minimum_effect
        and primary["relative_improvement_95_ci"][0] > minimum_ci
    )

    clean_grouped = _grouped_pair(
        cells,
        candidate=PI_CELL,
        control=TC_CELL,
        condition="clean",
        corrector_train_unseen=None,
    )
    if set(clean_grouped) != set(primary_grouped):
        raise ValueError("clean slice task set must match primary slice")
    clean = _paired_effect(
        clean_grouped,
        bootstrap_samples=samples,
        bootstrap_seed=seed + 1,
    )
    clean_margin = _finite(
        clean_protocol.get("ratio_upper_margin"),
        label="clean ratio upper margin",
        positive=True,
    )
    clean["passed"] = bool(
        clean["candidate_to_control_ratio_one_sided_95_upper"] < clean_margin
    )

    direction_protocol = _mapping(
        protocol.get("task_direction"), label="task direction"
    )
    task_effects = [
        row["relative_improvement"] for row in primary["per_task"].values()
    ]
    positive_tasks = sum(effect > 0.0 for effect in task_effects)
    min_positive = _integer(
        direction_protocol.get("minimum_positive_tasks"),
        label="minimum positive tasks",
        minimum=1,
    )
    max_degradation = _finite(
        direction_protocol.get("maximum_single_task_relative_degradation"),
        label="maximum task degradation",
    )
    task_direction = {
        "positive_tasks": positive_tasks,
        "task_count": len(task_effects),
        "worst_relative_improvement": min(task_effects),
        "passed": bool(
            positive_tasks >= min_positive and min(task_effects) >= -max_degradation
        ),
    }

    unseen_grouped = _grouped_pair(
        cells,
        candidate=PI_CELL,
        control=TC_CELL,
        condition="mild",
        corrector_train_unseen=True,
    )
    if len(unseen_grouped) != expected_task_count:
        raise ValueError(
            "corrector-train-unseen slice must contain exactly "
            f"{expected_task_count} tasks"
        )
    unseen = _paired_effect(
        unseen_grouped,
        bootstrap_samples=samples,
        bootstrap_seed=seed + 2,
    )
    unseen_effects = [
        row["relative_improvement"] for row in unseen["per_task"].values()
    ]
    unseen["selection_disclosure"] = (
        "Excluded from corrector gradient training but used for corrector "
        "validation and best-checkpoint selection; not an independent test."
    )
    unseen["passed"] = bool(
        unseen["relative_improvement"] > 0.0
        and sum(effect > 0.0 for effect in unseen_effects) >= min_positive
        and min(unseen_effects) >= -max_degradation
    )

    representation = _representation_gate(
        records,
        protocol,
        expected_task_count=expected_task_count,
        expected_tasks=set(primary["per_task"]),
    )
    secondary_specs = [
        ("paired_visual_vs_triplet_corrected", PV_CELL, TC_CELL),
        ("paired_innovation_vs_paired_visual", PI_CELL, PV_CELL),
        ("triplet_corrected_vs_triplet_nominal", TC_CELL, TN_CELL),
    ]
    secondary_protocol = _mapping(
        protocol.get("secondary_family"), label="secondary family"
    )
    alpha = _finite(secondary_protocol.get("alpha"), label="secondary alpha")
    alternative = secondary_protocol.get("alternative")
    secondary = []
    for index, (name, candidate, control) in enumerate(secondary_specs):
        grouped = _grouped_pair(
            cells,
            candidate=candidate,
            control=control,
            condition="mild",
            corrector_train_unseen=None,
        )
        effect = _paired_effect(
            grouped,
            bootstrap_samples=samples,
            bootstrap_seed=seed + 10 + index,
        )
        secondary.append(
            {
                "name": name,
                "relative_improvement": effect["relative_improvement"],
                "alternative": alternative,
                "p_value": _sign_flip_pvalue(
                    grouped,
                    samples=samples,
                    seed=seed + 20 + index,
                    alternative=alternative,
                ),
            }
        )
    secondary = _holm(secondary, alpha=alpha)

    gate_conditions = {
        "primary_superiority": bool(primary["passed"]),
        "clean_noninferiority": bool(clean["passed"]),
        "task_direction": bool(task_direction["passed"]),
        "representation_noncollapse": bool(representation["passed"]),
        "corrector_train_unseen_directional_agreement": bool(unseen["passed"]),
    }
    return {
        "schema_version": 1,
        "protocol": protocol["protocol"],
        "development_only": True,
        "primary": primary,
        "clean_noninferiority": clean,
        "task_direction": task_direction,
        "representation_noncollapse": representation,
        "corrector_train_unseen_sensitivity": unseen,
        "secondary_holm_family": secondary,
        "conditions": gate_conditions,
        "passed": all(gate_conditions.values()),
        "decision": (
            "fresh_multiseed_outer_holdout_confirmation_candidate"
            if all(gate_conditions.values())
            else "no_go"
        ),
    }


__all__ = [
    "PRIMARY_HORIZONS",
    "PI_CELL",
    "PV_CELL",
    "TC_CELL",
    "TN_CELL",
    "summarize_anchor_visual_dev_gate",
    "validate_anchor_visual_dev_protocol",
]
