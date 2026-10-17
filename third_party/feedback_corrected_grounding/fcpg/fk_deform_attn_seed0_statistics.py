"""Frozen seed0 development protocol and paired FK-DeformAttn statistics."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
ROW_SCHEMA_VERSION = 1
PROTOCOL = "fk-deform-attn-seed0-development-multitask-paired-v1"
DEFAULT_CONFIG = "config/fk_deform_attn/seed0_development_multitask_eval.json"
VARIANTS = ("policy_only_residual", "fk_deform_attn")
DIFFERENCE_ORDER = "fk_deform_attn-minus-policy_only_residual"

FROZEN_PROTOCOL_CONFIG: dict[str, object] = {
    "schema_version": 1,
    "protocol": PROTOCOL,
    "claim_scope": {
        "name": "seed0-development-controlled-comparison",
        "training_seeds": [0],
        "development_only": True,
        "cross_seed_generalization_supported": False,
        "ci_scope": "episode-paired-within-fixed-tasks-only",
    },
    "inputs": {
        "residual_checkpoint": (
            "outputs/fk_deform_attn/matched_policy_only_pair_seed0/"
            "policy_only_residual_best.pt"
        ),
        "deform_checkpoint": (
            "outputs/fk_deform_attn/matched_policy_only_pair_seed0/"
            "fk_deform_attn_best.pt"
        ),
        "policy_contract_checkpoint": (
            "outputs/action_chunk/lewm_stage1/"
            "shared_4task_seed0_lp0_sig0_b2048/fcpg_action_chunk_best.pt"
        ),
        "fcpg_corrector_checkpoint": (
            "outputs/transformer_residual/lewm_stage1/"
            "shared_4task_seed0_sig0/fcpg_transformer_residual_best.pt"
        ),
        "checkpoint_schema_version": 2,
        "checkpoint_state": "best",
    },
    "comparison": {
        "variants": ["policy_only_residual", "fk_deform_attn"],
        "difference_order": DIFFERENCE_ORDER,
        "structural_attribution": (
            "policy-only shared residual stencil versus independent deformable readout"
        ),
        "supervised_fcpg_scope": (
            "side-note-only system context excluded from structural statistics"
        ),
    },
    "tasks": [
        {"key": "pick", "name": "pick-place-v3"},
        {"key": "button", "name": "button-press-v3"},
        {"key": "reach", "name": "reach-v3"},
        {"key": "push", "name": "push-v3"},
    ],
    "conditions": [
        {
            "id": "clean",
            "perturbation": "clean",
            "severity": 0.0,
            "unit": "none",
        },
        {
            "id": "rotation_5deg",
            "perturbation": "rotation",
            "severity": 5.0,
            "unit": "degrees",
        },
        {
            "id": "rotation_10deg",
            "perturbation": "rotation",
            "severity": 10.0,
            "unit": "degrees",
        },
        {
            "id": "translation_2cm",
            "perturbation": "translation",
            "severity": 2.0,
            "unit": "centimeters",
        },
        {
            "id": "translation_4cm",
            "perturbation": "translation",
            "severity": 4.0,
            "unit": "centimeters",
        },
    ],
    "evaluation": {
        "camera": "corner",
        "width": 224,
        "height": 224,
        "horizon": 100,
        "action_steps": 8,
        "episodes_per_task": 20,
        "task_index_namespace": {
            "id": "fkda-seed0-dev-ti-20260718-v1",
            "base": 1726071,
            "task_stride": 100,
            "episode_stride": 1,
            "metaworld_effective_modulus": 50,
        },
        "env_seed_namespace": {
            "id": "fkda-seed0-dev-env-20260718-v1",
            "base": 1726071800,
            "task_stride": 1000,
            "episode_stride": 1,
        },
        "camera_rng_namespace": {
            "rotation_base": 1826071800,
            "translation_base": 1926071800,
            "task_stride": 1000,
            "episode_stride": 1,
            "shared_direction_across_axis_severities": True,
        },
        "freshness_guard": {
            "forbid_task_index_at_or_below": 1000000,
            "forbid_env_seed_at_or_below": 300000000,
            "reuse_physical_identity_across_camera_conditions": True,
            "claim_novel_mt1_parameter_instances": False,
        },
        "pairing": {
            "same_task_index": True,
            "same_env_seed": True,
            "same_camera_offset": True,
            "same_initial_physical_state_hash": True,
            "same_initial_view_hash": True,
            "variant_execution_order": "alternate-by-pair-index",
        },
        "amp": True,
        "device": "auto",
    },
    "statistics": {
        "bootstrap_samples": 20000,
        "bootstrap_seed": 20260718,
        "confidence_level": 0.95,
        "bootstrap_design": "episode-paired-stratified-within-fixed-task",
        "task_handling": "four-fixed-tasks-not-resampled",
        "seed_handling": "single-seed0-no-seed-resampling",
        "auc": {
            "rotation_degrees": [0.0, 5.0, 10.0],
            "translation_centimeters": [0.0, 2.0, 4.0],
            "rule": "per-episode-trapezoid-divided-by-maximum-severity",
            "shared_clean_episode": True,
        },
        "clean_noninferiority": {
            "candidate": "fk_deform_attn",
            "baseline": "policy_only_residual",
            "margin_success_probability": -0.05,
            "one_sided_confidence_level": 0.95,
            "decision_rule": "paired_task_macro_lower_bound_strictly_above_margin",
        },
    },
    "coordinate_diagnostic": {
        "name": "attention_barycenter_epe_px",
        "coordinate_source": (
            "PolicyReadoutOutput.sample_points_px weighted by attention_weights"
        ),
        "reference": "q_true projected from actual evaluation camera",
        "reference_is_model_input": False,
        "validity_mask": "in-front-of-camera-and-in-image",
        "primary_timepoint": "first-inference-latest-history-frame",
        "on_policy_rollout_metric": "descriptive-only-because-trajectories-diverge",
        "fail_closed": True,
        "unqualified_keypoint_epe_claim_supported": False,
    },
    "artifacts": {
        "output_root": "outputs/fk_deform_attn/seed0_development_multitask_eval",
        "pair_directory": "pairs",
        "rollout_csv": "paired_rollouts.csv",
        "statistics_json": "seed0_development_statistics.json",
    },
}


@dataclass(frozen=True)
class TaskSpec:
    key: str
    name: str
    ordinal: int


@dataclass(frozen=True)
class ConditionSpec:
    condition_id: str
    perturbation: str
    severity: float
    unit: str
    ordinal: int


@dataclass(frozen=True)
class EpisodeSpec:
    task: TaskSpec
    condition: ConditionSpec
    episode: int
    task_index: int
    effective_task_index: int
    env_seed: int
    camera_direction_seed: int
    physical_episode_id: str
    pair_id: str


ROW_FIELDS = (
    "row_schema_version",
    "protocol",
    "claim_scope",
    "training_seed",
    "task_key",
    "task",
    "task_ordinal",
    "condition_id",
    "perturbation",
    "severity",
    "episode",
    "physical_episode_id",
    "pair_id",
    "variant",
    "task_index",
    "effective_task_index",
    "env_seed",
    "camera_direction_seed",
    "physical_episode_sha256",
    "paired_initial_view_sha256",
    "initial_rgb_sha256",
    "initial_q_nominal_sha256",
    "initial_q_true_sha256",
    "initial_q_true_valid_sha256",
    "steps",
    "inference_calls",
    "success",
    "reward_sum",
    "action_abs_mean",
    "actual_camera_translation_cm",
    "actual_camera_translation_x_cm",
    "actual_camera_translation_y_cm",
    "actual_camera_translation_z_cm",
    "actual_camera_rotation_axis_x",
    "actual_camera_rotation_axis_y",
    "actual_camera_rotation_axis_z",
    "actual_camera_rotation_deg",
    "coordinate_diagnostic_status",
    "coordinate_diagnostic_name",
    "coordinate_diagnostic_reason",
    "coordinate_epe_claim_supported",
    "initial_attention_barycenter_epe_sum_px",
    "initial_attention_barycenter_valid_count",
    "initial_attention_barycenter_epe_px",
    "on_policy_attention_barycenter_epe_sum_px",
    "on_policy_attention_barycenter_valid_count",
    "on_policy_attention_barycenter_epe_px",
)

_INTEGER_FIELDS = {
    "row_schema_version",
    "training_seed",
    "task_ordinal",
    "episode",
    "task_index",
    "effective_task_index",
    "env_seed",
    "camera_direction_seed",
    "steps",
    "inference_calls",
    "initial_attention_barycenter_valid_count",
    "on_policy_attention_barycenter_valid_count",
}
_FLOAT_FIELDS = {
    "severity",
    "success",
    "reward_sum",
    "action_abs_mean",
    "actual_camera_translation_cm",
    "actual_camera_translation_x_cm",
    "actual_camera_translation_y_cm",
    "actual_camera_translation_z_cm",
    "actual_camera_rotation_axis_x",
    "actual_camera_rotation_axis_y",
    "actual_camera_rotation_axis_z",
    "actual_camera_rotation_deg",
    "initial_attention_barycenter_epe_sum_px",
    "initial_attention_barycenter_epe_px",
    "on_policy_attention_barycenter_epe_sum_px",
    "on_policy_attention_barycenter_epe_px",
}
_BOOLEAN_FIELDS = {"coordinate_epe_claim_supported"}


def _first_drift(expected: object, actual: object, path: str = "config") -> str | None:
    if type(expected) is not type(actual):
        return f"{path} type differs: expected {type(expected).__name__}"
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        if set(expected) != set(actual):
            return f"{path} keys differ"
        for key in expected:
            drift = _first_drift(expected[key], actual[key], f"{path}.{key}")
            if drift is not None:
                return drift
        return None
    if isinstance(expected, list):
        assert isinstance(actual, list)
        if len(expected) != len(actual):
            return f"{path} length differs"
        for index, (expected_value, actual_value) in enumerate(zip(expected, actual)):
            drift = _first_drift(
                expected_value,
                actual_value,
                f"{path}[{index}]",
            )
            if drift is not None:
                return drift
        return None
    if expected != actual:
        return f"{path} differs: expected {expected!r}, got {actual!r}"
    return None


def validate_protocol_config(config: Mapping[str, object]) -> dict[str, object]:
    payload = copy.deepcopy(dict(config))
    drift = _first_drift(FROZEN_PROTOCOL_CONFIG, payload)
    if drift is not None:
        raise ValueError(f"frozen FK-DeformAttn protocol drift: {drift}")
    schedule = expected_episode_specs(payload)
    task_floor = int(
        payload["evaluation"]["freshness_guard"][  # type: ignore[index]
            "forbid_task_index_at_or_below"
        ]
    )
    seed_floor = int(
        payload["evaluation"]["freshness_guard"][  # type: ignore[index]
            "forbid_env_seed_at_or_below"
        ]
    )
    if min(spec.task_index for spec in schedule) <= task_floor:
        raise ValueError("fresh task-index namespace overlaps prior development range")
    if min(spec.env_seed for spec in schedule) <= seed_floor:
        raise ValueError("fresh env-seed namespace overlaps prior development range")
    identities = {(spec.task.name, spec.task_index, spec.env_seed) for spec in schedule}
    expected_identity_count = len(tasks(payload)) * int(
        payload["evaluation"]["episodes_per_task"]  # type: ignore[index]
    )
    if len(identities) != expected_identity_count:
        raise ValueError("physical identities are not shared exactly across conditions")
    return payload


def load_protocol_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("protocol config must be a JSON object")
    return validate_protocol_config(payload)


def protocol_config_sha256(config: Mapping[str, object]) -> str:
    canonical = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def tasks(config: Mapping[str, object]) -> tuple[TaskSpec, ...]:
    raw = config["tasks"]
    if not isinstance(raw, list):
        raise TypeError("tasks must be a list")
    return tuple(
        TaskSpec(str(value["key"]), str(value["name"]), ordinal)
        for ordinal, value in enumerate(raw)
    )


def conditions(config: Mapping[str, object]) -> tuple[ConditionSpec, ...]:
    raw = config["conditions"]
    if not isinstance(raw, list):
        raise TypeError("conditions must be a list")
    return tuple(
        ConditionSpec(
            str(value["id"]),
            str(value["perturbation"]),
            float(value["severity"]),
            str(value["unit"]),
            ordinal,
        )
        for ordinal, value in enumerate(raw)
    )


def camera_direction_seed(
    config: Mapping[str, object],
    task: TaskSpec,
    condition: ConditionSpec,
    episode: int,
) -> int:
    if condition.perturbation == "clean":
        return 0
    namespace = config["evaluation"]["camera_rng_namespace"]  # type: ignore[index]
    base_key = f"{condition.perturbation}_base"
    return (
        int(namespace[base_key])
        + task.ordinal * int(namespace["task_stride"])
        + episode * int(namespace["episode_stride"])
    )


def expected_episode_specs(
    config: Mapping[str, object],
) -> tuple[EpisodeSpec, ...]:
    evaluation = config["evaluation"]
    task_namespace = evaluation["task_index_namespace"]  # type: ignore[index]
    seed_namespace = evaluation["env_seed_namespace"]  # type: ignore[index]
    episode_count = int(evaluation["episodes_per_task"])  # type: ignore[index]
    modulus = int(task_namespace["metaworld_effective_modulus"])
    result: list[EpisodeSpec] = []
    for task in tasks(config):
        for episode in range(episode_count):
            task_index = (
                int(task_namespace["base"])
                + task.ordinal * int(task_namespace["task_stride"])
                + episode * int(task_namespace["episode_stride"])
            )
            env_seed = (
                int(seed_namespace["base"])
                + task.ordinal * int(seed_namespace["task_stride"])
                + episode * int(seed_namespace["episode_stride"])
            )
            physical_id = (
                f"{task.key}:task-index-{task_index}:env-seed-{env_seed}:"
                f"episode-{episode:03d}"
            )
            for condition in conditions(config):
                pair_id = f"{physical_id}:{condition.condition_id}"
                result.append(
                    EpisodeSpec(
                        task=task,
                        condition=condition,
                        episode=episode,
                        task_index=task_index,
                        effective_task_index=task_index % modulus,
                        env_seed=env_seed,
                        camera_direction_seed=camera_direction_seed(
                            config,
                            task,
                            condition,
                            episode,
                        ),
                        physical_episode_id=physical_id,
                        pair_id=pair_id,
                    )
                )
    return tuple(result)


def pair_artifact_relative_path(
    config: Mapping[str, object],
    spec: EpisodeSpec,
) -> Path:
    directory = str(config["artifacts"]["pair_directory"])  # type: ignore[index]
    return (
        Path(directory)
        / spec.task.key
        / spec.condition.condition_id
        / f"episode_{spec.episode:03d}.json"
    )


def _require_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_coordinate_fields(row: Mapping[str, object]) -> None:
    supported = row["coordinate_epe_claim_supported"]
    if not isinstance(supported, bool):
        raise ValueError("coordinate_epe_claim_supported must be boolean")
    if row["coordinate_diagnostic_name"] != "attention_barycenter_epe_px":
        raise ValueError("coordinate diagnostic name changed")
    if supported:
        if row["coordinate_diagnostic_status"] != "available":
            raise ValueError("supported coordinate diagnostic is not available")
        if row["coordinate_diagnostic_reason"] != "":
            raise ValueError("available coordinate diagnostic has a failure reason")
        for prefix in ("initial_attention_barycenter", "on_policy_attention_barycenter"):
            total = _require_finite(row[f"{prefix}_epe_sum_px"], f"{prefix} sum")
            count = row[f"{prefix}_valid_count"]
            mean = _require_finite(row[f"{prefix}_epe_px"], f"{prefix} mean")
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise ValueError(f"{prefix} valid count must be a positive integer")
            if total < 0 or mean < 0 or not math.isclose(
                total / count,
                mean,
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                raise ValueError(f"{prefix} EPE fields are inconsistent")
    else:
        if row["coordinate_diagnostic_status"] != "unavailable":
            raise ValueError("unsupported coordinate diagnostic must be unavailable")
        if not str(row["coordinate_diagnostic_reason"]):
            raise ValueError("unavailable coordinate diagnostic requires a reason")
        for prefix in ("initial_attention_barycenter", "on_policy_attention_barycenter"):
            if row[f"{prefix}_epe_sum_px"] is not None:
                raise ValueError("fail-closed coordinate diagnostic exposed an EPE sum")
            if row[f"{prefix}_epe_px"] is not None:
                raise ValueError("fail-closed coordinate diagnostic exposed an EPE")
            if row[f"{prefix}_valid_count"] != 0:
                raise ValueError("fail-closed coordinate diagnostic exposed a count")


def validate_pair_rows(
    config: Mapping[str, object],
    spec: EpisodeSpec,
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    if len(rows) != len(VARIANTS):
        raise ValueError(f"pair {spec.pair_id} must contain exactly two rows")
    normalized = tuple(copy.deepcopy(dict(row)) for row in rows)
    if {str(row.get("variant")) for row in normalized} != set(VARIANTS):
        raise ValueError(f"pair {spec.pair_id} does not contain both variants")
    expected_common = {
        "row_schema_version": ROW_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "claim_scope": "seed0-development-controlled-comparison",
        "training_seed": 0,
        "task_key": spec.task.key,
        "task": spec.task.name,
        "task_ordinal": spec.task.ordinal,
        "condition_id": spec.condition.condition_id,
        "perturbation": spec.condition.perturbation,
        "severity": spec.condition.severity,
        "episode": spec.episode,
        "physical_episode_id": spec.physical_episode_id,
        "pair_id": spec.pair_id,
        "task_index": spec.task_index,
        "effective_task_index": spec.effective_task_index,
        "env_seed": spec.env_seed,
        "camera_direction_seed": spec.camera_direction_seed,
    }
    for row in normalized:
        if set(row) != set(ROW_FIELDS):
            raise ValueError(f"rollout row fields differ for {spec.pair_id}")
        for name, expected in expected_common.items():
            if row[name] != expected:
                raise ValueError(
                    f"pair {spec.pair_id} field {name} differs: "
                    f"expected {expected!r}, got {row[name]!r}"
                )
        if row["variant"] not in VARIANTS:
            raise ValueError("rollout row has an unknown structural variant")
        if not isinstance(row["steps"], int) or int(row["steps"]) < 0:
            raise ValueError("steps must be a non-negative integer")
        if not isinstance(row["inference_calls"], int) or int(
            row["inference_calls"]
        ) < 0:
            raise ValueError("inference_calls must be a non-negative integer")
        success = _require_finite(row["success"], "success")
        if success not in (0.0, 1.0):
            raise ValueError("success must be binary")
        for name in (
            "reward_sum",
            "action_abs_mean",
            "actual_camera_translation_cm",
            "actual_camera_translation_x_cm",
            "actual_camera_translation_y_cm",
            "actual_camera_translation_z_cm",
            "actual_camera_rotation_axis_x",
            "actual_camera_rotation_axis_y",
            "actual_camera_rotation_axis_z",
            "actual_camera_rotation_deg",
        ):
            _require_finite(row[name], name)
        for name in (
            "physical_episode_sha256",
            "paired_initial_view_sha256",
            "initial_rgb_sha256",
            "initial_q_nominal_sha256",
            "initial_q_true_sha256",
            "initial_q_true_valid_sha256",
        ):
            value = str(row[name])
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        _validate_coordinate_fields(row)

    first, second = normalized
    paired_fields = (
        "physical_episode_sha256",
        "paired_initial_view_sha256",
        "initial_rgb_sha256",
        "initial_q_nominal_sha256",
        "initial_q_true_sha256",
        "initial_q_true_valid_sha256",
        "actual_camera_translation_cm",
        "actual_camera_translation_x_cm",
        "actual_camera_translation_y_cm",
        "actual_camera_translation_z_cm",
        "actual_camera_rotation_axis_x",
        "actual_camera_rotation_axis_y",
        "actual_camera_rotation_axis_z",
        "actual_camera_rotation_deg",
    )
    for name in paired_fields:
        if first[name] != second[name]:
            raise ValueError(f"strict pair field {name} differs for {spec.pair_id}")

    translation = np.asarray(
        [
            first["actual_camera_translation_x_cm"],
            first["actual_camera_translation_y_cm"],
            first["actual_camera_translation_z_cm"],
        ],
        dtype=np.float64,
    )
    translation_norm = float(np.linalg.norm(translation))
    reported_norm = float(first["actual_camera_translation_cm"])
    rotation = float(first["actual_camera_rotation_deg"])
    if not math.isclose(translation_norm, reported_norm, abs_tol=1e-7):
        raise ValueError("camera translation vector and magnitude disagree")
    if spec.condition.perturbation == "clean":
        expected_translation = 0.0
        expected_rotation = 0.0
    elif spec.condition.perturbation == "rotation":
        expected_translation = 0.0
        expected_rotation = spec.condition.severity
    elif spec.condition.perturbation == "translation":
        expected_translation = spec.condition.severity
        expected_rotation = 0.0
    else:
        raise ValueError("unknown perturbation")
    if not math.isclose(translation_norm, expected_translation, abs_tol=1e-7):
        raise ValueError("camera translation does not match frozen severity")
    if not math.isclose(abs(rotation), expected_rotation, abs_tol=1e-7):
        raise ValueError("camera rotation does not match frozen severity")
    return tuple(sorted(normalized, key=lambda row: VARIANTS.index(str(row["variant"]))))


def validate_rollout_rows(
    config: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    *,
    require_complete: bool = True,
) -> dict[str, object]:
    specs = {spec.pair_id: spec for spec in expected_episode_specs(config)}
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        pair_id = str(row.get("pair_id", ""))
        if pair_id not in specs:
            raise ValueError(f"rollout contains an unregistered pair {pair_id!r}")
        grouped.setdefault(pair_id, []).append(row)
    if require_complete and set(grouped) != set(specs):
        missing = sorted(set(specs) - set(grouped))
        raise ValueError(f"rollout matrix is incomplete; missing {len(missing)} pairs")
    validated: list[dict[str, object]] = []
    for pair_id, pair_rows in grouped.items():
        validated.extend(validate_pair_rows(config, specs[pair_id], pair_rows))

    by_physical: dict[str, list[dict[str, object]]] = {}
    for row in validated:
        by_physical.setdefault(str(row["physical_episode_id"]), []).append(row)
    for physical_id, physical_rows in by_physical.items():
        hashes = {str(row["physical_episode_sha256"]) for row in physical_rows}
        if len(hashes) != 1:
            raise ValueError(
                f"physical state changed across camera conditions for {physical_id}"
            )
        if require_complete and {
            str(row["condition_id"]) for row in physical_rows
        } != {condition.condition_id for condition in conditions(config)}:
            raise ValueError(f"physical episode {physical_id} lacks a camera condition")

        by_condition = {
            str(row["condition_id"]): row
            for row in physical_rows
            if row["variant"] == VARIANTS[0]
        }
        if {"rotation_5deg", "rotation_10deg"}.issubset(by_condition):
            first = by_condition["rotation_5deg"]
            second = by_condition["rotation_10deg"]
            first_axis = np.asarray(
                [
                    first["actual_camera_rotation_axis_x"],
                    first["actual_camera_rotation_axis_y"],
                    first["actual_camera_rotation_axis_z"],
                ],
                dtype=np.float64,
            )
            second_axis = np.asarray(
                [
                    second["actual_camera_rotation_axis_x"],
                    second["actual_camera_rotation_axis_y"],
                    second["actual_camera_rotation_axis_z"],
                ],
                dtype=np.float64,
            )
            if not np.allclose(first_axis, second_axis, atol=1e-12, rtol=0.0):
                raise ValueError("rotation axis is not shared across severities")
            if math.copysign(1.0, float(first["actual_camera_rotation_deg"])) != math.copysign(
                1.0,
                float(second["actual_camera_rotation_deg"]),
            ):
                raise ValueError("rotation sign is not shared across severities")
        if {"translation_2cm", "translation_4cm"}.issubset(by_condition):
            directions = []
            for condition_id in ("translation_2cm", "translation_4cm"):
                row = by_condition[condition_id]
                vector = np.asarray(
                    [
                        row["actual_camera_translation_x_cm"],
                        row["actual_camera_translation_y_cm"],
                        row["actual_camera_translation_z_cm"],
                    ],
                    dtype=np.float64,
                )
                directions.append(vector / np.linalg.norm(vector))
            if not np.allclose(directions[0], directions[1], atol=1e-12, rtol=0.0):
                raise ValueError("translation direction is not shared across severities")

    return {
        "complete": require_complete and len(grouped) == len(specs),
        "expected_pairs": len(specs),
        "observed_pairs": len(grouped),
        "observed_rows": len(validated),
        "strict_variant_pairing": True,
        "physical_state_hash_shared_across_conditions": True,
        "camera_direction_shared_across_axis_severities": True,
    }


def _namespace_seed(base: int, label: str) -> int:
    digest = hashlib.sha256(f"{base}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _bootstrap_paired_difference(
    values_by_task: Mapping[str, np.ndarray],
    config: Mapping[str, object],
    label: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    statistics = config["statistics"]
    samples = int(statistics["bootstrap_samples"])  # type: ignore[index]
    base_seed = int(statistics["bootstrap_seed"])  # type: ignore[index]
    task_draws: dict[str, np.ndarray] = {}
    for task in tasks(config):
        values = np.asarray(values_by_task[task.key], dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"paired bootstrap values for {task.key} are invalid")
        rng = np.random.default_rng(_namespace_seed(base_seed, f"{label}:{task.key}"))
        indices = rng.integers(0, values.size, size=(samples, values.size))
        task_draws[task.key] = values[indices].mean(axis=1)
    macro = np.stack([task_draws[task.key] for task in tasks(config)], axis=0).mean(
        axis=0
    )
    return task_draws, macro


def _two_sided_interval(
    draws: np.ndarray,
    confidence_level: float,
) -> tuple[float, float]:
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)


def _paired_metric_summary(
    candidate_by_task: Mapping[str, np.ndarray],
    baseline_by_task: Mapping[str, np.ndarray],
    config: Mapping[str, object],
    label: str,
    *,
    higher_is_better: bool = True,
) -> dict[str, object]:
    differences = {
        task.key: np.asarray(candidate_by_task[task.key], dtype=np.float64)
        - np.asarray(baseline_by_task[task.key], dtype=np.float64)
        for task in tasks(config)
    }
    task_draws, macro_draws = _bootstrap_paired_difference(
        differences,
        config,
        label,
    )
    confidence = float(config["statistics"]["confidence_level"])  # type: ignore[index]
    task_rows = []
    for task in tasks(config):
        candidate = np.asarray(candidate_by_task[task.key], dtype=np.float64)
        baseline = np.asarray(baseline_by_task[task.key], dtype=np.float64)
        lower, upper = _two_sided_interval(task_draws[task.key], confidence)
        delta = candidate - baseline
        wins = delta > 0 if higher_is_better else delta < 0
        losses = delta < 0 if higher_is_better else delta > 0
        task_rows.append(
            {
                "task": task.key,
                "episodes": int(delta.size),
                "policy_only_residual": float(baseline.mean()),
                "fk_deform_attn": float(candidate.mean()),
                "paired_difference": float(delta.mean()),
                "ci_lower": lower,
                "ci_upper": upper,
                "wins": int(np.count_nonzero(wins)),
                "ties": int(np.count_nonzero(delta == 0)),
                "losses": int(np.count_nonzero(losses)),
            }
        )
    macro_lower, macro_upper = _two_sided_interval(macro_draws, confidence)
    macro_candidate = float(
        np.mean(
            [
                np.asarray(candidate_by_task[task.key], dtype=np.float64).mean()
                for task in tasks(config)
            ]
        )
    )
    macro_baseline = float(
        np.mean(
            [
                np.asarray(baseline_by_task[task.key], dtype=np.float64).mean()
                for task in tasks(config)
            ]
        )
    )
    return {
        "difference_order": DIFFERENCE_ORDER,
        "higher_is_better": higher_is_better,
        "win_definition": (
            "fk_deform_attn_greater_than_policy_only_residual"
            if higher_is_better
            else "fk_deform_attn_lower_than_policy_only_residual"
        ),
        "confidence_level": confidence,
        "ci_unit": "paired physical episode within fixed task",
        "tasks": task_rows,
        "task_macro": {
            "policy_only_residual": macro_baseline,
            "fk_deform_attn": macro_candidate,
            "paired_difference": macro_candidate - macro_baseline,
            "ci_lower": macro_lower,
            "ci_upper": macro_upper,
        },
        "_macro_bootstrap_draws": macro_draws,
    }


def _lookup_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, int, str, str], Mapping[str, object]]:
    result = {}
    for row in rows:
        key = (
            str(row["task_key"]),
            int(row["episode"]),
            str(row["condition_id"]),
            str(row["variant"]),
        )
        if key in result:
            raise ValueError(f"duplicate rollout row {key}")
        result[key] = row
    return result


def _normalized_trapezoid(values: Sequence[float], severities: Sequence[float]) -> float:
    if len(values) != len(severities) or len(values) < 2:
        raise ValueError("AUC requires aligned values and at least two severities")
    x = np.asarray(severities, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if not bool(np.all(np.diff(x) > 0)) or x[0] != 0 or x[-1] <= 0:
        raise ValueError("AUC severities must start at zero and strictly increase")
    area = float(np.sum((y[:-1] + y[1:]) * 0.5 * np.diff(x)))
    return area / float(x[-1])


def _success_by_task(
    lookup: Mapping[tuple[str, int, str, str], Mapping[str, object]],
    config: Mapping[str, object],
    condition_id: str,
    variant: str,
) -> dict[str, np.ndarray]:
    count = int(config["evaluation"]["episodes_per_task"])  # type: ignore[index]
    return {
        task.key: np.asarray(
            [
                lookup[(task.key, episode, condition_id, variant)]["success"]
                for episode in range(count)
            ],
            dtype=np.float64,
        )
        for task in tasks(config)
    }


def _auc_by_task(
    lookup: Mapping[tuple[str, int, str, str], Mapping[str, object]],
    config: Mapping[str, object],
    axis: str,
    variant: str,
) -> dict[str, np.ndarray]:
    if axis == "rotation":
        condition_ids = ("clean", "rotation_5deg", "rotation_10deg")
        severities = config["statistics"]["auc"]["rotation_degrees"]  # type: ignore[index]
    elif axis == "translation":
        condition_ids = ("clean", "translation_2cm", "translation_4cm")
        severities = config["statistics"]["auc"][  # type: ignore[index]
            "translation_centimeters"
        ]
    else:
        raise ValueError("unknown AUC axis")
    count = int(config["evaluation"]["episodes_per_task"])  # type: ignore[index]
    result = {}
    for task in tasks(config):
        result[task.key] = np.asarray(
            [
                _normalized_trapezoid(
                    [
                        float(
                            lookup[(task.key, episode, condition_id, variant)][
                                "success"
                            ]
                        )
                        for condition_id in condition_ids
                    ],
                    severities,
                )
                for episode in range(count)
            ],
            dtype=np.float64,
        )
    return result


def _strip_bootstrap_draws(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result.pop("_macro_bootstrap_draws", None)
    return result


def _coordinate_report(
    rows: Sequence[Mapping[str, object]],
    lookup: Mapping[tuple[str, int, str, str], Mapping[str, object]],
    config: Mapping[str, object],
) -> dict[str, object]:
    unsupported = [
        row for row in rows if not bool(row["coordinate_epe_claim_supported"])
    ]
    contract = copy.deepcopy(config["coordinate_diagnostic"])
    if unsupported:
        return {
            "contract": contract,
            "epe_claim_supported": False,
            "metric": None,
            "reason": "one or more rollout rows failed the coordinate semantic contract",
            "failure_reasons": sorted(
                {str(row["coordinate_diagnostic_reason"]) for row in unsupported}
            ),
            "statistics": None,
        }
    count = int(config["evaluation"]["episodes_per_task"])  # type: ignore[index]
    condition_reports = []
    for condition in conditions(config):
        values: dict[str, dict[str, np.ndarray]] = {variant: {} for variant in VARIANTS}
        for variant in VARIANTS:
            for task in tasks(config):
                values[variant][task.key] = np.asarray(
                    [
                        lookup[
                            (
                                task.key,
                                episode,
                                condition.condition_id,
                                variant,
                            )
                        ]["initial_attention_barycenter_epe_px"]
                        for episode in range(count)
                    ],
                    dtype=np.float64,
                )
        summary = _paired_metric_summary(
            values["fk_deform_attn"],
            values["policy_only_residual"],
            config,
            f"initial-epe:{condition.condition_id}",
            higher_is_better=False,
        )
        condition_reports.append(
            {
                "condition": condition.condition_id,
                **_strip_bootstrap_draws(summary),
            }
        )
    on_policy_weighted = {}
    for variant in VARIANTS:
        total = sum(
            float(row["on_policy_attention_barycenter_epe_sum_px"])
            for row in rows
            if row["variant"] == variant
        )
        valid = sum(
            int(row["on_policy_attention_barycenter_valid_count"])
            for row in rows
            if row["variant"] == variant
        )
        on_policy_weighted[variant] = total / valid
    return {
        "contract": contract,
        "epe_claim_supported": True,
        "metric": "attention_barycenter_epe_px",
        "unqualified_keypoint_epe_claim_supported": False,
        "initial_same_state_paired": condition_reports,
        "on_policy_descriptive_weighted_mean": on_policy_weighted,
        "on_policy_between_variant_inference_supported": False,
    }


def summarize_rollout_rows(
    config: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    validated_config = validate_protocol_config(config)
    audit = validate_rollout_rows(validated_config, rows, require_complete=True)
    lookup = _lookup_rows(rows)

    paired_success = []
    success_payloads: dict[str, dict[str, object]] = {}
    for condition in conditions(validated_config):
        candidate = _success_by_task(
            lookup,
            validated_config,
            condition.condition_id,
            "fk_deform_attn",
        )
        baseline = _success_by_task(
            lookup,
            validated_config,
            condition.condition_id,
            "policy_only_residual",
        )
        summary = _paired_metric_summary(
            candidate,
            baseline,
            validated_config,
            f"success:{condition.condition_id}",
        )
        success_payloads[condition.condition_id] = summary
        paired_success.append(
            {
                "condition": condition.condition_id,
                "perturbation": condition.perturbation,
                "severity": condition.severity,
                **_strip_bootstrap_draws(summary),
            }
        )

    auc_reports = []
    for axis in ("rotation", "translation"):
        candidate = _auc_by_task(
            lookup,
            validated_config,
            axis,
            "fk_deform_attn",
        )
        baseline = _auc_by_task(
            lookup,
            validated_config,
            axis,
            "policy_only_residual",
        )
        summary = _paired_metric_summary(
            candidate,
            baseline,
            validated_config,
            f"auc:{axis}",
        )
        auc_reports.append(
            {
                "axis": axis,
                "normalization": "trapezoid-area-divided-by-maximum-severity",
                **_strip_bootstrap_draws(summary),
            }
        )

    clean = success_payloads["clean"]
    clean_draws = clean["_macro_bootstrap_draws"]
    assert isinstance(clean_draws, np.ndarray)
    ni = validated_config["statistics"]["clean_noninferiority"]  # type: ignore[index]
    one_sided_confidence = float(ni["one_sided_confidence_level"])
    lower = float(np.quantile(clean_draws, 1.0 - one_sided_confidence))
    margin = float(ni["margin_success_probability"])
    clean_macro = clean["task_macro"]
    assert isinstance(clean_macro, dict)

    report = {
        "schema_version": SCHEMA_VERSION,
        "report_type": "fk_deform_attn_seed0_development_statistics",
        "complete": True,
        "protocol": PROTOCOL,
        "protocol_config_sha256": protocol_config_sha256(validated_config),
        "claim_scope": copy.deepcopy(validated_config["claim_scope"]),
        "comparison": copy.deepcopy(validated_config["comparison"]),
        "audit": audit,
        "inference_contract": {
            "training_seed_count": 1,
            "seed0_only": True,
            "cross_seed_claim_supported": False,
            "bootstrap_unit": "paired physical episode within fixed task",
            "tasks_resampled": False,
            "seeds_resampled": False,
        },
        "paired_success": paired_success,
        "robustness_auc": auc_reports,
        "clean_noninferiority": {
            "difference_order": DIFFERENCE_ORDER,
            "margin_success_probability": margin,
            "one_sided_confidence_level": one_sided_confidence,
            "paired_task_macro_difference": float(clean_macro["paired_difference"]),
            "one_sided_lower_bound": lower,
            "noninferior": lower > margin,
            "decision_rule": ni["decision_rule"],
            "development_only": True,
        },
        "coordinate_diagnostic": _coordinate_report(
            rows,
            lookup,
            validated_config,
        ),
        "supervised_fcpg_side_note": {
            "included_in_rows": False,
            "included_in_statistics": False,
            "scope": validated_config["comparison"]["supervised_fcpg_scope"],  # type: ignore[index]
        },
    }
    return report


def load_rollout_csv(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ROW_FIELDS:
            raise ValueError("rollout CSV schema differs from the frozen row schema")
        rows: list[dict[str, object]] = []
        for raw in reader:
            row: dict[str, object] = {}
            for name in ROW_FIELDS:
                value = raw[name]
                if name in _INTEGER_FIELDS:
                    row[name] = int(value)
                elif name in _FLOAT_FIELDS:
                    row[name] = None if value == "" else float(value)
                elif name in _BOOLEAN_FIELDS:
                    if value not in {"True", "False"}:
                        raise ValueError(f"CSV boolean field {name} is invalid")
                    row[name] = value == "True"
                else:
                    row[name] = value
            rows.append(row)
    return rows


def atomic_json_write(payload: Mapping[str, object], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    config = load_protocol_config(args.config)
    rows = load_rollout_csv(args.rows)
    report = summarize_rollout_rows(config, rows)
    atomic_json_write(report, args.output)
    print(f"statistics: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
