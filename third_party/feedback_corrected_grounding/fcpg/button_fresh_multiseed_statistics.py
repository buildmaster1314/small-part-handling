"""Strict identity audit and crossed-bootstrap statistics for button P1.

The physical task is fixed. Inference resamples policy seeds and the shared raw
episode keys as two crossed dimensions, while preserving method and severity
pairing inside every bootstrap draw.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


PROTOCOL = "button-p1-strict-fresh-multiseed-v1"
SCHEMA_VERSION = 1
DEFAULT_CONFIG = "config/experiments/button_p1_fresh_multiseed.json"
CAMERA_FIELDS = (
    "actual_camera_translation_cm",
    "actual_camera_translation_x_cm",
    "actual_camera_translation_y_cm",
    "actual_camera_translation_z_cm",
    "actual_camera_rotation_axis_x",
    "actual_camera_rotation_axis_y",
    "actual_camera_rotation_axis_z",
    "actual_camera_rotation_deg",
)


@dataclass(frozen=True)
class CameraCondition:
    condition_id: str
    kind: str
    severity: float
    episodes: int
    rotation_deg: float
    translation_cm: float
    aliases: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    policy_family: str
    source_mode: str
    grounding_override: str
    external_grounding: str
    model_name: str


@dataclass(frozen=True)
class EpisodeRecord:
    task: str
    policy_seed: int
    method: str
    condition_id: str
    pair_key: str
    episode_identity: str
    raw_task_index: int
    env_seed: int
    success: float
    camera_values: tuple[float, ...]


METHOD_SPECS = {
    "jitter": MethodSpec(
        "jitter", "jitter", "geoprop", "native", "none", "geoprop"
    ),
    "fcpg_nominal": MethodSpec(
        "fcpg_nominal", "fcpg", "fcpg", "nominal", "none", "fcpg_nominal"
    ),
    "fcpg_native": MethodSpec(
        "fcpg_native", "fcpg", "fcpg", "native", "none", "fcpg"
    ),
    "prototype": MethodSpec(
        "prototype", "fcpg", "fcpg", "native", "prototype", "fcpg_prototype"
    ),
    "pnp": MethodSpec("pnp", "fcpg", "fcpg", "native", "pnp", "fcpg_pnp"),
    "oracle": MethodSpec(
        "oracle", "fcpg", "fcpg", "oracle", "none", "fcpg_oracle"
    ),
}


_EXPECTED_TOP_LEVEL = {
    "schema_version",
    "protocol",
    "task",
    "inputs",
    "artifacts",
    "reuse",
    "seeds",
    "corrector",
    "policy",
    "jitter",
    "prototype",
    "pnp",
    "evaluation",
    "statistics",
}


def _nested(value: Mapping[str, object], path: str) -> object:
    current: object = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"protocol config is missing {path}")
        current = current[part]
    return current


def _require_frozen(config: Mapping[str, object], path: str, expected: object) -> None:
    actual = _nested(config, path)
    if type(actual) is not type(expected) or actual != expected:  # noqa: E721
        raise ValueError(
            f"frozen protocol value {path} must be {expected!r}, got {actual!r}"
        )


def validate_protocol_config(config: Mapping[str, object]) -> dict[str, object]:
    """Validate the complete preregistered protocol and reject silent drift."""

    if set(config) != _EXPECTED_TOP_LEVEL:
        missing = sorted(_EXPECTED_TOP_LEVEL.difference(config))
        extra = sorted(set(config).difference(_EXPECTED_TOP_LEVEL))
        raise ValueError(f"protocol config keys differ; missing={missing}, extra={extra}")

    expected_section_keys = {
        "inputs": {"data", "vision_cache", "vision_model_name", "vision_selected_layers"},
        "artifacts": {
            "output_root",
            "corrector_root",
            "grounding_cache_root",
            "policy_root",
            "prototype_bank",
            "rollout_root",
            "statistics_root",
        },
        "reuse": {"corrector_checkpoints", "grounding_caches"},
        "seeds": {"corrector", "policy", "runner"},
        "corrector": {
            "architecture",
            "obs_mode",
            "vision_dim",
            "model_dim",
            "local_grid_size",
            "local_radius_px",
            "local_layers",
            "global_layers",
            "heads",
            "dropout",
            "max_residual_px",
            "reliability_weight",
            "reliability_corruption_probability",
            "sigreg_weight",
            "epochs",
            "batch_size",
            "num_workers",
            "learning_rate",
            "weight_decay",
            "warmup_epochs",
            "validation_fraction",
            "split_unit",
            "split_seed_matches_training_seed",
        },
        "policy": {
            "architecture",
            "obs_mode",
            "obs_horizon",
            "action_horizon",
            "vision_dim",
            "model_dim",
            "encoder_layers",
            "decoder_layers",
            "heads",
            "dropout",
            "epochs",
            "batch_size",
            "num_workers",
            "learning_rate",
            "weight_decay",
            "warmup_steps",
            "latent_prediction_weight",
            "sigreg_weight",
            "patience",
            "split_seed_matches_training_seed",
        },
        "jitter": {"standard_deviation_px", "maximum_px"},
        "prototype": {
            "keypoint_source",
            "count_per_keypoint",
            "feature_level",
            "split_seed",
            "seed",
            "radius_px",
            "grid_size",
        },
        "pnp": {
            "history_frames",
            "maximum_reprojection_error_px",
            "maximum_pose_rotation_deg",
            "maximum_pose_translation_cm",
        },
        "evaluation": {
            "camera",
            "width",
            "height",
            "horizon",
            "action_steps",
            "task_index_offset",
            "env_seed_base",
            "clean_episodes_per_seed",
            "nonzero_episodes_per_seed",
            "rotation_degrees",
            "translation_centimeters",
            "zero_condition",
            "methods",
        },
        "statistics": {
            "bootstrap_samples",
            "bootstrap_seed",
            "confidence_level",
            "task_handling",
            "bootstrap_design",
            "auc_clean_episode_rule",
            "paired_comparisons",
            "clean_noninferiority",
        },
        "statistics.clean_noninferiority": {
            "margin_success_probability",
            "one_sided_confidence_level",
            "required_comparisons",
            "decision_rule",
        },
    }
    for path, expected_keys in expected_section_keys.items():
        section = _nested(config, path)
        if not isinstance(section, Mapping) or set(section) != expected_keys:
            actual_keys = sorted(section) if isinstance(section, Mapping) else []
            raise ValueError(
                f"protocol section {path} keys differ; expected={sorted(expected_keys)}, "
                f"actual={actual_keys}"
            )

    frozen = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "task": "button-press-v3",
        "inputs.data": (
            "data/metaworld_projection/"
            "scripted_button_actualcam_se3_clean25_300eps_224.npz"
        ),
        "inputs.vision_cache": "data/vision_cache/button300_224_vitb_l11_fp16",
        "inputs.vision_model_name": "vit_base_patch16_224.augreg_in1k",
        "inputs.vision_selected_layers": [11],
        "artifacts.output_root": "outputs/button_p1_fresh_multiseed",
        "artifacts.corrector_root": (
            "outputs/button_p1_fresh_multiseed/training/correctors"
        ),
        "artifacts.grounding_cache_root": (
            "data/grounding_cache/button_p1_fresh_multiseed"
        ),
        "artifacts.policy_root": "outputs/button_p1_fresh_multiseed/training/policies",
        "artifacts.prototype_bank": (
            "data/baseline_cache/button_p1_fresh_multiseed_prototypes128_seed0"
        ),
        "artifacts.rollout_root": "outputs/button_p1_fresh_multiseed/rollouts",
        "artifacts.statistics_root": "outputs/button_p1_fresh_multiseed/statistics",
        "reuse.corrector_checkpoints": {
            "0": (
                "outputs/transformer_residual/lewm_stage1/button_seed0_sig0/"
                "fcpg_transformer_residual_best.pt"
            )
        },
        "reuse.grounding_caches": {
            "0": "data/grounding_cache/lewm_stage1/button_seed0_sig0"
        },
        "seeds.corrector": [0, 1, 2],
        "seeds.policy": [0, 1, 2],
        "seeds.runner": 20260718,
        "corrector.architecture": "lewm",
        "corrector.obs_mode": "robot",
        "corrector.vision_dim": 256,
        "corrector.model_dim": 256,
        "corrector.local_grid_size": 5,
        "corrector.local_radius_px": 24.0,
        "corrector.local_layers": 4,
        "corrector.global_layers": 2,
        "corrector.heads": 8,
        "corrector.dropout": 0.1,
        "corrector.max_residual_px": 20.0,
        "corrector.reliability_weight": 0.0,
        "corrector.reliability_corruption_probability": 0.0,
        "corrector.sigreg_weight": 0.0,
        "corrector.epochs": 50,
        "corrector.batch_size": 1024,
        "corrector.num_workers": 4,
        "corrector.learning_rate": 0.0003,
        "corrector.weight_decay": 0.00001,
        "corrector.warmup_epochs": 3.0,
        "corrector.validation_fraction": 0.2,
        "corrector.split_unit": "task-episode",
        "corrector.split_seed_matches_training_seed": True,
        "policy.architecture": "legacy",
        "policy.obs_mode": "robot",
        "policy.obs_horizon": 2,
        "policy.action_horizon": 32,
        "policy.vision_dim": 256,
        "policy.model_dim": 256,
        "policy.encoder_layers": 2,
        "policy.decoder_layers": 6,
        "policy.heads": 8,
        "policy.dropout": 0.1,
        "policy.epochs": 60,
        "policy.batch_size": 1024,
        "policy.num_workers": 4,
        "policy.learning_rate": 0.0001,
        "policy.weight_decay": 0.000001,
        "policy.warmup_steps": 100,
        "policy.latent_prediction_weight": 0.0,
        "policy.sigreg_weight": 0.0,
        "policy.patience": 60,
        "policy.split_seed_matches_training_seed": True,
        "jitter.standard_deviation_px": 8.0,
        "jitter.maximum_px": 24.0,
        "prototype.keypoint_source": "primary",
        "prototype.count_per_keypoint": 128,
        "prototype.feature_level": 0,
        "prototype.split_seed": 0,
        "prototype.seed": 0,
        "prototype.radius_px": 32.0,
        "prototype.grid_size": 9,
        "pnp.history_frames": 3,
        "pnp.maximum_reprojection_error_px": 8.0,
        "pnp.maximum_pose_rotation_deg": 30.0,
        "pnp.maximum_pose_translation_cm": 20.0,
        "evaluation.camera": "corner",
        "evaluation.width": 224,
        "evaluation.height": 224,
        "evaluation.horizon": 100,
        "evaluation.action_steps": 8,
        "evaluation.task_index_offset": 2000,
        "evaluation.env_seed_base": 80000,
        "evaluation.clean_episodes_per_seed": 80,
        "evaluation.nonzero_episodes_per_seed": 20,
        "evaluation.rotation_degrees": [0.0, 5.0, 10.0],
        "evaluation.translation_centimeters": [0.0, 2.0, 4.0],
        "evaluation.zero_condition": "shared_clean",
        "evaluation.methods": list(METHOD_SPECS),
        "statistics.bootstrap_samples": 20000,
        "statistics.bootstrap_seed": 20260718,
        "statistics.confidence_level": 0.95,
        "statistics.task_handling": "fixed_not_resampled",
        "statistics.bootstrap_design": (
            "policy_seed_x_shared_episode_identity_two_way_crossed"
        ),
        "statistics.auc_clean_episode_rule": (
            "intersection_with_all_nonzero_severities"
        ),
        "statistics.clean_noninferiority.margin_success_probability": -0.03,
        "statistics.clean_noninferiority.one_sided_confidence_level": 0.95,
        "statistics.clean_noninferiority.required_comparisons": [
            "fcpg_native_minus_nominal",
            "fcpg_native_minus_jitter",
        ],
        "statistics.clean_noninferiority.decision_rule": (
            "each_one_sided_lower_bound_strictly_greater_than_margin"
        ),
    }
    for path, expected in frozen.items():
        _require_frozen(config, path, expected)

    expected_comparisons = [
        ("fcpg_native_minus_nominal", "fcpg_native", "fcpg_nominal"),
        ("fcpg_native_minus_jitter", "fcpg_native", "jitter"),
        ("fcpg_nominal_minus_jitter", "fcpg_nominal", "jitter"),
        ("prototype_minus_jitter", "prototype", "jitter"),
        ("pnp_minus_jitter", "pnp", "jitter"),
        ("oracle_minus_jitter", "oracle", "jitter"),
        ("pnp_minus_prototype", "pnp", "prototype"),
    ]
    raw_comparisons = _nested(config, "statistics.paired_comparisons")
    if not isinstance(raw_comparisons, list):
        raise ValueError("statistics.paired_comparisons must be a list")
    actual_comparisons = []
    for value in raw_comparisons:
        if not isinstance(value, Mapping) or set(value) != {
            "name",
            "candidate",
            "baseline",
        }:
            raise ValueError("each paired comparison must have exactly three fields")
        actual_comparisons.append(
            (value["name"], value["candidate"], value["baseline"])
        )
    if actual_comparisons != expected_comparisons:
        raise ValueError("statistics.paired_comparisons differs from preregistration")

    conditions = canonical_conditions(config)
    identities = [
        value["episode_identity"]
        for condition in conditions
        for value in expected_episode_contracts(config, condition)
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("evaluation schedule contains duplicate episode identities")
    return copy.deepcopy(dict(config))


def load_protocol_config(path: str | Path) -> dict[str, object]:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("protocol config must contain a JSON object")
    return validate_protocol_config(payload)


def _severity_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def canonical_conditions(config: Mapping[str, object]) -> tuple[CameraCondition, ...]:
    evaluation = _nested(config, "evaluation")
    assert isinstance(evaluation, Mapping)
    clean_id = str(evaluation["zero_condition"])
    clean_episodes = int(evaluation["clean_episodes_per_seed"])
    nonzero_episodes = int(evaluation["nonzero_episodes_per_seed"])
    rotation = tuple(float(value) for value in evaluation["rotation_degrees"])
    translation = tuple(
        float(value) for value in evaluation["translation_centimeters"]
    )
    if rotation.count(0.0) != 1 or translation.count(0.0) != 1:
        raise ValueError("each severity axis must contain exactly one zero alias")
    conditions = [
        CameraCondition(
            clean_id,
            "clean",
            0.0,
            clean_episodes,
            0.0,
            0.0,
            (("rotation_deg", 0.0), ("translation_cm", 0.0)),
        )
    ]
    conditions.extend(
        CameraCondition(
            f"rotation_{_severity_token(value)}deg",
            "rotation",
            value,
            nonzero_episodes,
            value,
            0.0,
            (("rotation_deg", value),),
        )
        for value in rotation
        if value != 0.0
    )
    conditions.extend(
        CameraCondition(
            f"translation_{_severity_token(value)}cm",
            "translation",
            value,
            nonzero_episodes,
            0.0,
            value,
            (("translation_cm", value),),
        )
        for value in translation
        if value != 0.0
    )
    condition_ids = [value.condition_id for value in conditions]
    if len(condition_ids) != len(set(condition_ids)):
        raise ValueError("canonical camera conditions must be unique")
    return tuple(conditions)


def conditions_for_axis(
    conditions: Sequence[CameraCondition], axis: str
) -> tuple[CameraCondition, ...]:
    if axis not in {"rotation_deg", "translation_cm"}:
        raise ValueError(f"unknown severity axis: {axis}")
    selected = [
        condition
        for condition in conditions
        if any(alias_axis == axis for alias_axis, _ in condition.aliases)
    ]
    return tuple(
        sorted(
            selected,
            key=lambda condition: next(
                severity
                for alias_axis, severity in condition.aliases
                if alias_axis == axis
            ),
        )
    )


def method_specs(config: Mapping[str, object]) -> tuple[MethodSpec, ...]:
    names = _nested(config, "evaluation.methods")
    assert isinstance(names, list)
    return tuple(METHOD_SPECS[str(name)] for name in names)


def resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def corrector_checkpoint_path(
    config: Mapping[str, object], repo_root: Path, seed: int
) -> Path:
    reuse = _nested(config, "reuse.corrector_checkpoints")
    assert isinstance(reuse, Mapping)
    if str(seed) in reuse:
        return resolve_repo_path(repo_root, str(reuse[str(seed)]))
    root = resolve_repo_path(repo_root, str(_nested(config, "artifacts.corrector_root")))
    return root / f"seed_{seed}" / "fcpg_transformer_residual_best.pt"


def grounding_cache_path(
    config: Mapping[str, object], repo_root: Path, seed: int
) -> Path:
    reuse = _nested(config, "reuse.grounding_caches")
    assert isinstance(reuse, Mapping)
    if str(seed) in reuse:
        return resolve_repo_path(repo_root, str(reuse[str(seed)]))
    root = resolve_repo_path(
        repo_root, str(_nested(config, "artifacts.grounding_cache_root"))
    )
    return root / f"seed_{seed}"


def policy_checkpoint_path(
    config: Mapping[str, object], repo_root: Path, family: str, seed: int
) -> Path:
    if family not in {"fcpg", "jitter"}:
        raise ValueError(f"unknown policy family: {family}")
    root = resolve_repo_path(repo_root, str(_nested(config, "artifacts.policy_root")))
    mode = "fcpg" if family == "fcpg" else "geoprop"
    return root / f"{family}_seed_{seed}" / f"{mode}_action_chunk_best.pt"


def rollout_output_dir(
    config: Mapping[str, object],
    repo_root: Path,
    seed: int,
    condition: CameraCondition,
    method: MethodSpec,
) -> Path:
    root = resolve_repo_path(repo_root, str(_nested(config, "artifacts.rollout_root")))
    return root / f"seed_{seed}" / condition.condition_id / method.name


def rollout_artifact_paths(output_dir: Path, method: MethodSpec) -> tuple[Path, Path]:
    return (
        output_dir / f"{method.model_name}_rollout.csv",
        output_dir / f"{method.model_name}_rollout_summary.json",
    )


def pair_key(task: str, raw_task_index: int, env_seed: int) -> str:
    return f"{task}|task_index={raw_task_index}|env_seed={env_seed}"


def expected_episode_contracts(
    config: Mapping[str, object], condition: CameraCondition
) -> tuple[dict[str, object], ...]:
    task = str(_nested(config, "task"))
    task_offset = int(_nested(config, "evaluation.task_index_offset"))
    env_seed_base = int(_nested(config, "evaluation.env_seed_base"))
    result = []
    identities: set[str] = set()
    for episode in range(condition.episodes):
        raw_task_index = task_offset + episode
        env_seed = env_seed_base + episode
        shared_key = pair_key(task, raw_task_index, env_seed)
        identity = f"{shared_key}|camera_condition={condition.condition_id}"
        if identity in identities:
            raise ValueError(f"duplicate scheduled episode identity: {identity}")
        identities.add(identity)
        result.append(
            {
                "episode": episode,
                "task": task,
                "raw_task_index": raw_task_index,
                "env_seed": env_seed,
                "pair_key": shared_key,
                "episode_identity": identity,
                "camera_condition": {
                    "id": condition.condition_id,
                    "kind": condition.kind,
                    "rotation_deg": condition.rotation_deg,
                    "translation_cm": condition.translation_cm,
                },
            }
        )
    return tuple(result)


def _same_path(actual: object, expected: Path) -> bool:
    if not isinstance(actual, str) or not actual:
        return False
    return Path(actual).expanduser().resolve() == expected.resolve()


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer") from error
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be an integer")
    return result


def _as_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _camera_tuple(row: Mapping[str, object]) -> tuple[float, ...]:
    missing = [field for field in CAMERA_FIELDS if field not in row]
    if missing:
        raise ValueError(f"rollout row is missing camera identity fields: {missing}")
    return tuple(_as_float(row[field], field) for field in CAMERA_FIELDS)


def _assert_camera_condition(
    values: tuple[float, ...], condition: CameraCondition
) -> None:
    translation_cm = values[0]
    rotation_deg = values[-1]
    if not math.isclose(
        translation_cm, condition.translation_cm, rel_tol=0.0, abs_tol=1e-5
    ):
        raise ValueError(
            f"actual camera translation {translation_cm} does not match "
            f"condition {condition.condition_id}"
        )
    if not math.isclose(
        abs(rotation_deg), condition.rotation_deg, rel_tol=0.0, abs_tol=1e-5
    ):
        raise ValueError(
            f"actual camera rotation {rotation_deg} does not match "
            f"condition {condition.condition_id}"
        )


def _camera_equal(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return bool(np.allclose(left, right, rtol=0.0, atol=1e-8))


def identity_contract_payload(
    config: Mapping[str, object],
    seed: int,
    condition: CameraCondition,
    method: MethodSpec,
    records: Sequence[EpisodeRecord],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "task": str(_nested(config, "task")),
        "policy_seed": seed,
        "method": method.name,
        "runner_seed": int(_nested(config, "seeds.runner")),
        "camera_condition": {
            "id": condition.condition_id,
            "kind": condition.kind,
            "rotation_deg": condition.rotation_deg,
            "translation_cm": condition.translation_cm,
        },
        "episodes": [
            {
                "pair_key": record.pair_key,
                "episode_identity": record.episode_identity,
                "raw_task_index": record.raw_task_index,
                "env_seed": record.env_seed,
                "actual_camera": dict(zip(CAMERA_FIELDS, record.camera_values, strict=True)),
            }
            for record in records
        ],
    }


def audit_rollout_artifact(
    config: Mapping[str, object],
    repo_root: Path,
    seed: int,
    condition: CameraCondition,
    method: MethodSpec,
    *,
    require_identity_contract: bool = False,
) -> tuple[EpisodeRecord, ...]:
    """Validate summary, CSV, and every raw physical identity fail-closed."""

    output_dir = rollout_output_dir(config, repo_root, seed, condition, method)
    csv_path, summary_path = rollout_artifact_paths(output_dir, method)
    if not csv_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"rollout artifact is incomplete: {output_dir}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("summary"), Mapping):
        raise ValueError(f"invalid rollout summary payload: {summary_path}")
    summary = payload["summary"]
    assert isinstance(summary, Mapping)
    evaluation = _nested(config, "evaluation")
    prototype = _nested(config, "prototype")
    pnp = _nested(config, "pnp")
    assert isinstance(evaluation, Mapping)
    assert isinstance(prototype, Mapping)
    assert isinstance(pnp, Mapping)
    checkpoint = policy_checkpoint_path(
        config, repo_root, method.policy_family, seed
    )
    exact_summary = {
        "model": method.model_name,
        "source_mode": method.source_mode,
        "grounding_override": method.grounding_override,
        "external_grounding": method.external_grounding,
        "task": str(_nested(config, "task")),
        "episodes": condition.episodes,
        "horizon": int(evaluation["horizon"]),
        "action_steps": int(evaluation["action_steps"]),
        "translation_cm_min": condition.translation_cm,
        "translation_cm_max": condition.translation_cm,
        "rotation_deg_min": condition.rotation_deg,
        "rotation_deg_max": condition.rotation_deg,
        "prototype_radius_px": float(prototype["radius_px"]),
        "prototype_grid_size": int(prototype["grid_size"]),
        "pnp_history_frames": int(pnp["history_frames"]),
        "pnp_max_reprojection_error_px": float(
            pnp["maximum_reprojection_error_px"]
        ),
        "pnp_max_pose_rotation_deg": float(pnp["maximum_pose_rotation_deg"]),
        "pnp_max_pose_translation_cm": float(
            pnp["maximum_pose_translation_cm"]
        ),
        "reliability_override_request": "native",
    }
    for key, expected in exact_summary.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"rollout summary {summary_path} has {key}={summary.get(key)!r}; "
                f"expected {expected!r}"
            )
    if not _same_path(summary.get("checkpoint"), checkpoint):
        raise ValueError(f"rollout checkpoint does not match policy seed {seed}")
    prototype_bank = resolve_repo_path(
        repo_root, str(_nested(config, "artifacts.prototype_bank"))
    )
    if method.external_grounding == "none":
        if summary.get("prototype_bank") is not None:
            raise ValueError("non-external rollout unexpectedly records a prototype bank")
    elif not _same_path(summary.get("prototype_bank"), prototype_bank):
        raise ValueError("external rollout prototype bank path is incorrect")

    raw_episodes = payload.get("episodes")
    if not isinstance(raw_episodes, list) or len(raw_episodes) != condition.episodes:
        raise ValueError(f"rollout episodes do not match {condition.condition_id}")
    expected_rows = expected_episode_contracts(config, condition)
    records: list[EpisodeRecord] = []
    seen: set[str] = set()
    for expected, row in zip(expected_rows, raw_episodes, strict=True):
        if not isinstance(row, Mapping):
            raise ValueError("rollout episode must be a JSON object")
        raw_task_index = _as_int(row.get("task_index"), "task_index")
        env_seed = _as_int(row.get("env_seed"), "env_seed")
        episode = _as_int(row.get("episode"), "episode")
        task = str(row.get("task"))
        if (
            episode != expected["episode"]
            or task != expected["task"]
            or raw_task_index != expected["raw_task_index"]
            or env_seed != expected["env_seed"]
        ):
            raise ValueError(
                f"rollout raw identity differs at episode {expected['episode']}"
            )
        identity = str(expected["episode_identity"])
        if identity in seen:
            raise ValueError(f"duplicate rollout episode identity: {identity}")
        seen.add(identity)
        camera_values = _camera_tuple(row)
        _assert_camera_condition(camera_values, condition)
        success = _as_float(row.get("success"), "success")
        if success not in {0.0, 1.0}:
            raise ValueError("rollout success must be binary")
        if row.get("model") != method.model_name:
            raise ValueError("rollout episode model label is incorrect")
        records.append(
            EpisodeRecord(
                task=task,
                policy_seed=seed,
                method=method.name,
                condition_id=condition.condition_id,
                pair_key=str(expected["pair_key"]),
                episode_identity=identity,
                raw_task_index=raw_task_index,
                env_seed=env_seed,
                success=success,
                camera_values=camera_values,
            )
        )

    reported_success = _as_float(summary.get("success_rate"), "summary success_rate")
    observed_success = float(np.mean([record.success for record in records]))
    if not math.isclose(reported_success, observed_success, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("rollout summary success_rate disagrees with episode rows")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != len(records):
        raise ValueError("rollout CSV and summary have different episode counts")
    for csv_row, record in zip(csv_rows, records, strict=True):
        if (
            str(csv_row.get("task")) != record.task
            or _as_int(csv_row.get("task_index"), "CSV task_index")
            != record.raw_task_index
            or _as_int(csv_row.get("env_seed"), "CSV env_seed") != record.env_seed
            or _as_float(csv_row.get("success"), "CSV success") != record.success
            or not _camera_equal(_camera_tuple(csv_row), record.camera_values)
        ):
            raise ValueError("rollout CSV identity disagrees with JSON summary")

    contract_path = output_dir / "identity_contract.json"
    expected_contract = identity_contract_payload(
        config, seed, condition, method, records
    )
    if require_identity_contract:
        if not contract_path.is_file():
            raise FileNotFoundError(f"missing rollout identity contract: {contract_path}")
        submitted = json.loads(contract_path.read_text(encoding="utf-8"))
        if submitted != expected_contract:
            raise ValueError(f"rollout identity contract mismatch: {contract_path}")
    return tuple(records)


def protocol_identity_manifest(
    config: Mapping[str, object], records: Iterable[EpisodeRecord]
) -> dict[str, object]:
    """Collapse method/seed replicas into one audited physical-identity manifest."""

    by_identity: dict[str, EpisodeRecord] = {}
    seen_cells: set[tuple[int, str, str]] = set()
    for record in records:
        cell = (record.policy_seed, record.method, record.episode_identity)
        if cell in seen_cells:
            raise ValueError(f"duplicate result identity in method/seed cell: {cell}")
        seen_cells.add(cell)
        reference = by_identity.get(record.episode_identity)
        if reference is None:
            by_identity[record.episode_identity] = record
        elif (
            reference.pair_key != record.pair_key
            or reference.raw_task_index != record.raw_task_index
            or reference.env_seed != record.env_seed
            or not _camera_equal(reference.camera_values, record.camera_values)
        ):
            raise ValueError(
                f"shared episode identity changed across methods/seeds: "
                f"{record.episode_identity}"
            )
    expected = {
        value["episode_identity"]
        for condition in canonical_conditions(config)
        for value in expected_episode_contracts(config, condition)
    }
    if set(by_identity) != expected:
        missing = sorted(expected.difference(by_identity))
        extra = sorted(set(by_identity).difference(expected))
        raise ValueError(
            f"identity manifest does not cover schedule; missing={missing[:3]}, "
            f"extra={extra[:3]}"
        )
    condition_lookup = {
        condition.condition_id: condition for condition in canonical_conditions(config)
    }
    episodes = []
    for identity in sorted(by_identity):
        record = by_identity[identity]
        condition = condition_lookup[record.condition_id]
        episodes.append(
            {
                "episode_identity": identity,
                "pair_key": record.pair_key,
                "raw_task_index": record.raw_task_index,
                "env_seed": record.env_seed,
                "camera_condition": {
                    "id": condition.condition_id,
                    "kind": condition.kind,
                    "rotation_deg": condition.rotation_deg,
                    "translation_cm": condition.translation_cm,
                },
                "actual_camera": dict(
                    zip(CAMERA_FIELDS, record.camera_values, strict=True)
                ),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "task": str(_nested(config, "task")),
        "runner_seed": int(_nested(config, "seeds.runner")),
        "identity_definition": (
            "task + raw_task_index + env_seed + canonical_camera_condition"
        ),
        "shared_pair_key_definition": "task + raw_task_index + env_seed",
        "episodes": episodes,
    }


class CrossedDesign:
    """Validated method x policy-seed x shared-episode result cube."""

    def __init__(
        self,
        records: Iterable[EpisodeRecord],
        *,
        seeds: Sequence[int],
        methods: Sequence[str],
        conditions: Sequence[CameraCondition],
        fixed_task: str,
    ) -> None:
        self.seeds = tuple(int(value) for value in seeds)
        self.methods = tuple(str(value) for value in methods)
        self.conditions = tuple(conditions)
        self.fixed_task = fixed_task
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("policy seeds must be unique")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("methods must be unique")
        self._values: dict[tuple[int, str, str, str], float] = {}
        self._camera: dict[tuple[str, str], tuple[float, ...]] = {}
        condition_ids = {value.condition_id for value in self.conditions}
        for record in records:
            if record.task != fixed_task:
                raise ValueError("task is fixed and must not vary in statistics")
            if record.policy_seed not in self.seeds:
                raise ValueError(f"unexpected policy seed: {record.policy_seed}")
            if record.method not in self.methods:
                raise ValueError(f"unexpected method: {record.method}")
            if record.condition_id not in condition_ids:
                raise ValueError(f"unexpected condition: {record.condition_id}")
            key = (
                record.policy_seed,
                record.method,
                record.condition_id,
                record.pair_key,
            )
            if key in self._values:
                raise ValueError(f"duplicate crossed-design result: {key}")
            self._values[key] = record.success
            camera_key = (record.condition_id, record.pair_key)
            reference = self._camera.setdefault(camera_key, record.camera_values)
            if not _camera_equal(reference, record.camera_values):
                raise ValueError("actual camera identity differs across paired cells")
        self._keys: dict[str, tuple[str, ...]] = {}
        for condition in self.conditions:
            cell_sets = []
            for seed in self.seeds:
                for method in self.methods:
                    keys = {
                        pair
                        for candidate_seed, candidate_method, candidate_condition, pair in self._values
                        if candidate_seed == seed
                        and candidate_method == method
                        and candidate_condition == condition.condition_id
                    }
                    cell_sets.append(keys)
            if not cell_sets or not cell_sets[0]:
                raise ValueError(f"condition {condition.condition_id} has no results")
            if any(keys != cell_sets[0] for keys in cell_sets[1:]):
                raise ValueError(
                    f"condition {condition.condition_id} lacks shared episode keys"
                )
            if len(cell_sets[0]) != condition.episodes:
                raise ValueError(
                    f"condition {condition.condition_id} has {len(cell_sets[0])} "
                    f"episodes; expected {condition.episodes}"
                )
            self._keys[condition.condition_id] = tuple(sorted(cell_sets[0]))

    def keys(self, condition_id: str) -> tuple[str, ...]:
        return self._keys[condition_id]

    def matrix(
        self,
        condition_id: str,
        method: str,
        keys: Sequence[str] | None = None,
    ) -> np.ndarray:
        selected = tuple(keys) if keys is not None else self.keys(condition_id)
        return np.asarray(
            [
                [self._values[(seed, method, condition_id, pair)] for pair in selected]
                for seed in self.seeds
            ],
            dtype=np.float64,
        )


def _crossed_distribution(
    values: np.ndarray, samples: int, seed: int
) -> np.ndarray:
    """Resample seed rows and shared episode columns independently."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("crossed bootstrap requires a non-empty seed x episode matrix")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    num_seeds, num_episodes = matrix.shape
    rng = np.random.default_rng(seed)
    result = np.empty(samples, dtype=np.float64)
    batch_size = min(2000, samples)
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        seed_draw = rng.integers(0, num_seeds, size=(count, num_seeds))
        episode_draw = rng.integers(0, num_episodes, size=(count, num_episodes))
        sampled = matrix[seed_draw[:, :, None], episode_draw[:, None, :]]
        result[start : start + count] = sampled.mean(axis=(1, 2))
    return result


def crossed_bootstrap_summary(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, object]:
    matrix = np.asarray(values, dtype=np.float64)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    distribution = _crossed_distribution(matrix, samples, seed)
    alpha = 1.0 - confidence
    return {
        "estimate": float(matrix.mean()),
        "ci95": [
            float(np.quantile(distribution, alpha / 2.0)),
            float(np.quantile(distribution, 1.0 - alpha / 2.0)),
        ],
        "one_sided_lower_95": float(np.quantile(distribution, alpha)),
        "per_policy_seed": {
            str(index): float(matrix[index].mean()) for index in range(matrix.shape[0])
        },
        "policy_seed_count": int(matrix.shape[0]),
        "shared_episode_count": int(matrix.shape[1]),
        "bootstrap_samples": int(samples),
        "bootstrap_seed": int(seed),
        "bootstrap_method": "policy_seed_x_shared_episode_two_way_crossed",
    }


def normalized_auc(severities: Sequence[float], values: np.ndarray) -> np.ndarray:
    x = np.asarray(severities, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("AUC severities must be a strictly increasing vector")
    if y.shape[-1] != x.size:
        raise ValueError("AUC value axis does not match severities")
    widths = np.diff(x)
    area = np.sum(widths * (y[..., :-1] + y[..., 1:]) / 2.0, axis=-1)
    return area / float(x[-1] - x[0])


def _auc_distribution(
    curves: np.ndarray,
    severities: Sequence[float],
    samples: int,
    seed: int,
) -> np.ndarray:
    values = np.asarray(curves, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("AUC curves must have shape [seed, episode, severity]")
    num_seeds, num_episodes, num_conditions = values.shape
    if num_conditions != len(severities):
        raise ValueError("AUC curve severity count is incorrect")
    rng = np.random.default_rng(seed)
    result = np.empty(samples, dtype=np.float64)
    batch_size = min(2000, samples)
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        seed_draw = rng.integers(0, num_seeds, size=(count, num_seeds))
        episode_draw = rng.integers(0, num_episodes, size=(count, num_episodes))
        means = np.empty((count, num_conditions), dtype=np.float64)
        for condition_index in range(num_conditions):
            matrix = values[:, :, condition_index]
            selected = matrix[seed_draw[:, :, None], episode_draw[:, None, :]]
            means[:, condition_index] = selected.mean(axis=(1, 2))
        result[start : start + count] = normalized_auc(severities, means)
    return result


def auc_bootstrap_summary(
    curves: np.ndarray,
    severities: Sequence[float],
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> dict[str, object]:
    values = np.asarray(curves, dtype=np.float64)
    point_curve = values.mean(axis=(0, 1))
    distribution = _auc_distribution(values, severities, samples, seed)
    alpha = 1.0 - confidence
    return {
        "estimate": float(normalized_auc(severities, point_curve)),
        "ci95": [
            float(np.quantile(distribution, alpha / 2.0)),
            float(np.quantile(distribution, 1.0 - alpha / 2.0)),
        ],
        "policy_seed_count": int(values.shape[0]),
        "shared_episode_count": int(values.shape[1]),
        "bootstrap_samples": int(samples),
        "bootstrap_seed": int(seed),
        "bootstrap_method": "policy_seed_x_shared_episode_two_way_crossed",
    }


def paired_auc_bootstrap_summary(
    candidate: np.ndarray,
    baseline: np.ndarray,
    severities: Sequence[float],
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> dict[str, object]:
    candidate_values = np.asarray(candidate, dtype=np.float64)
    baseline_values = np.asarray(baseline, dtype=np.float64)
    if candidate_values.shape != baseline_values.shape:
        raise ValueError("paired AUC curves must have identical shapes")
    difference = candidate_values - baseline_values
    summary = auc_bootstrap_summary(
        difference,
        severities,
        samples=samples,
        seed=seed,
        confidence=confidence,
    )
    summary["estimand"] = "candidate_normalized_auc_minus_baseline_normalized_auc"
    return summary


def _comparison_specs(config: Mapping[str, object]) -> tuple[dict[str, str], ...]:
    raw = _nested(config, "statistics.paired_comparisons")
    assert isinstance(raw, list)
    return tuple({key: str(value[key]) for key in value} for value in raw)


def summarize_records(
    config: Mapping[str, object],
    records: Iterable[EpisodeRecord],
    *,
    bootstrap_samples: int | None = None,
    bootstrap_seed: int | None = None,
) -> dict[str, object]:
    """Compute preregistered severity, AUC, and clean NI estimates."""

    seeds = tuple(int(value) for value in _nested(config, "seeds.policy"))
    methods = tuple(spec.name for spec in method_specs(config))
    conditions = canonical_conditions(config)
    design = CrossedDesign(
        records,
        seeds=seeds,
        methods=methods,
        conditions=conditions,
        fixed_task=str(_nested(config, "task")),
    )
    samples = int(
        bootstrap_samples
        if bootstrap_samples is not None
        else _nested(config, "statistics.bootstrap_samples")
    )
    base_seed = int(
        bootstrap_seed
        if bootstrap_seed is not None
        else _nested(config, "statistics.bootstrap_seed")
    )
    confidence = float(_nested(config, "statistics.confidence_level"))
    comparisons = _comparison_specs(config)
    seed_offset = 0

    condition_results: dict[str, dict[str, object]] = {}
    for condition in conditions:
        method_results: dict[str, object] = {}
        for method in methods:
            method_results[method] = crossed_bootstrap_summary(
                design.matrix(condition.condition_id, method),
                samples=samples,
                seed=base_seed + seed_offset,
                confidence=confidence,
            )
            seed_offset += 1
        comparison_results: dict[str, object] = {}
        for comparison in comparisons:
            candidate = design.matrix(
                condition.condition_id, comparison["candidate"]
            )
            baseline = design.matrix(condition.condition_id, comparison["baseline"])
            result = crossed_bootstrap_summary(
                candidate - baseline,
                samples=samples,
                seed=base_seed + seed_offset,
                confidence=confidence,
            )
            seed_offset += 1
            result.update(
                {
                    "candidate": comparison["candidate"],
                    "baseline": comparison["baseline"],
                    "estimand": "paired_success_probability_difference",
                }
            )
            comparison_results[comparison["name"]] = result
        condition_results[condition.condition_id] = {
            "condition_id": condition.condition_id,
            "camera_condition": {
                "kind": condition.kind,
                "rotation_deg": condition.rotation_deg,
                "translation_cm": condition.translation_cm,
            },
            "episodes_per_policy_seed": condition.episodes,
            "methods": method_results,
            "paired_comparisons": comparison_results,
        }

    severity_results = []
    for axis in ("rotation_deg", "translation_cm"):
        for condition in conditions_for_axis(conditions, axis):
            alias_severity = next(
                severity
                for alias_axis, severity in condition.aliases
                if alias_axis == axis
            )
            row = copy.deepcopy(condition_results[condition.condition_id])
            row["axis"] = axis
            row["severity"] = alias_severity
            severity_results.append(row)

    auc_results: dict[str, object] = {}
    for axis in ("rotation_deg", "translation_cm"):
        axis_conditions = conditions_for_axis(conditions, axis)
        severities = [
            next(
                severity
                for alias_axis, severity in condition.aliases
                if alias_axis == axis
            )
            for condition in axis_conditions
        ]
        nonzero_key_sets = [
            set(design.keys(condition.condition_id)) for condition in axis_conditions[1:]
        ]
        if not nonzero_key_sets or any(
            values != nonzero_key_sets[0] for values in nonzero_key_sets[1:]
        ):
            raise ValueError(f"{axis} nonzero severities do not share pair keys")
        clean_keys = set(design.keys(axis_conditions[0].condition_id))
        if not nonzero_key_sets[0].issubset(clean_keys):
            raise ValueError(f"{axis} nonzero pair keys are not a subset of clean")
        shared_keys = tuple(sorted(nonzero_key_sets[0]))
        curves: dict[str, np.ndarray] = {}
        method_auc: dict[str, object] = {}
        for method in methods:
            curve = np.stack(
                [
                    design.matrix(condition.condition_id, method, shared_keys)
                    for condition in axis_conditions
                ],
                axis=-1,
            )
            curves[method] = curve
            method_auc[method] = auc_bootstrap_summary(
                curve,
                severities,
                samples=samples,
                seed=base_seed + seed_offset,
                confidence=confidence,
            )
            seed_offset += 1
        paired_auc: dict[str, object] = {}
        for comparison in comparisons:
            result = paired_auc_bootstrap_summary(
                curves[comparison["candidate"]],
                curves[comparison["baseline"]],
                severities,
                samples=samples,
                seed=base_seed + seed_offset,
                confidence=confidence,
            )
            seed_offset += 1
            result.update(
                {
                    "candidate": comparison["candidate"],
                    "baseline": comparison["baseline"],
                }
            )
            paired_auc[comparison["name"]] = result
        auc_results[axis] = {
            "severities": severities,
            "condition_ids": [value.condition_id for value in axis_conditions],
            "shared_episode_count": len(shared_keys),
            "clean_episodes_excluded_from_auc": len(clean_keys) - len(shared_keys),
            "clean_episode_rule": str(
                _nested(config, "statistics.auc_clean_episode_rule")
            ),
            "methods": method_auc,
            "paired_comparisons": paired_auc,
        }

    ni_config = _nested(config, "statistics.clean_noninferiority")
    assert isinstance(ni_config, Mapping)
    margin = float(ni_config["margin_success_probability"])
    clean_result = condition_results[str(_nested(config, "evaluation.zero_condition"))]
    required = tuple(str(value) for value in ni_config["required_comparisons"])
    ni_comparisons: dict[str, object] = {}
    decisions = []
    for name in required:
        result = copy.deepcopy(clean_result["paired_comparisons"][name])
        demonstrated = float(result["one_sided_lower_95"]) > margin
        result.update(
            {
                "margin_success_probability": margin,
                "decision": "demonstrated" if demonstrated else "not_demonstrated",
                "decision_basis": (
                    "one-sided 95% crossed-bootstrap lower bound strictly above margin"
                ),
            }
        )
        decisions.append(demonstrated)
        ni_comparisons[name] = result
    clean_noninferiority = {
        "preregistered": True,
        "margin_success_probability": margin,
        "one_sided_confidence_level": float(
            ni_config["one_sided_confidence_level"]
        ),
        "comparisons": ni_comparisons,
        "joint_decision": "demonstrated" if all(decisions) else "not_demonstrated",
        "interpretation": (
            "Noninferiority is declared only from the preregistered one-sided "
            "confidence bounds; absence of statistical significance is not used."
        ),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "task": str(_nested(config, "task")),
        "policy_seeds": list(seeds),
        "methods": list(methods),
        "design": {
            "task": "fixed_not_resampled",
            "resampled_dimensions": ["policy_seed", "shared_episode_identity"],
            "bootstrap": "two_way_crossed_cartesian_resampling",
            "pairing": "methods_and_severities_share_each_bootstrap_draw",
            "success_units": "probability_0_to_1",
            "bootstrap_samples": samples,
            "bootstrap_seed": base_seed,
            "confidence_level": confidence,
        },
        "severity_results": severity_results,
        "auc_results": auc_results,
        "clean_noninferiority": clean_noninferiority,
    }


def render_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# Button P1 Fresh Multi-Seed Statistics",
        "",
        "Task is fixed. Confidence intervals use a policy-seed x shared-episode "
        "two-way crossed bootstrap.",
        "",
        "## Paired Success",
        "",
        "| axis | severity | comparison | delta | 95% CI |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for row in summary["severity_results"]:
        for name, result in row["paired_comparisons"].items():
            ci = result["ci95"]
            lines.append(
                f"| {row['axis']} | {float(row['severity']):g} | {name} | "
                f"{float(result['estimate']):+.4f} | "
                f"[{float(ci[0]):+.4f}, {float(ci[1]):+.4f}] |"
            )
    lines.extend(
        [
            "",
            "## Paired AUC Differences",
            "",
            "| axis | comparison | delta | 95% CI | shared episodes |",
            "| --- | --- | ---: | --- | ---: |",
        ]
    )
    for axis, axis_result in summary["auc_results"].items():
        for name, result in axis_result["paired_comparisons"].items():
            ci = result["ci95"]
            lines.append(
                f"| {axis} | {name} | {float(result['estimate']):+.4f} | "
                f"[{float(ci[0]):+.4f}, {float(ci[1]):+.4f}] | "
                f"{int(axis_result['shared_episode_count'])} |"
            )
    ni = summary["clean_noninferiority"]
    lines.extend(
        [
            "",
            "## Clean Noninferiority",
            "",
            f"Preregistered margin: {float(ni['margin_success_probability']):+.4f}.",
            "",
            "| comparison | estimate | one-sided 95% lower bound | decision |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for name, result in ni["comparisons"].items():
        lines.append(
            f"| {name} | {float(result['estimate']):+.4f} | "
            f"{float(result['one_sided_lower_95']):+.4f} | {result['decision']} |"
        )
    lines.extend(["", f"Joint decision: **{ni['joint_decision']}**.", ""])
    return "\n".join(lines)


def _atomic_json_write(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_audited_records(
    config: Mapping[str, object], repo_root: Path
) -> tuple[tuple[EpisodeRecord, ...], dict[str, object]]:
    records = []
    seeds = tuple(int(value) for value in _nested(config, "seeds.policy"))
    for seed in seeds:
        for condition in canonical_conditions(config):
            for method in method_specs(config):
                records.extend(
                    audit_rollout_artifact(
                        config,
                        repo_root,
                        seed,
                        condition,
                        method,
                        require_identity_contract=True,
                    )
                )
    manifest = protocol_identity_manifest(config, records)
    manifest_path = resolve_repo_path(
        repo_root, str(_nested(config, "artifacts.output_root"))
    ) / "episode_identity_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing protocol identity manifest: {manifest_path}")
    submitted = json.loads(manifest_path.read_text(encoding="utf-8"))
    if submitted != manifest:
        raise ValueError("protocol identity manifest does not match rollout outputs")
    return tuple(records), manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    config_path = resolve_repo_path(repo_root, args.config)
    config = load_protocol_config(config_path)
    records, _ = load_audited_records(config, repo_root)
    summary = summarize_records(config, records)
    output_root = resolve_repo_path(
        repo_root, str(_nested(config, "artifacts.statistics_root"))
    )
    json_path = output_root / "button_p1_fresh_multiseed_statistics.json"
    markdown_path = output_root / "button_p1_fresh_multiseed_statistics.md"
    _atomic_json_write(summary, json_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    print(f"statistics: {json_path}")
    print(f"report: {markdown_path}")
    return 0


__all__ = [
    "CAMERA_FIELDS",
    "CameraCondition",
    "CrossedDesign",
    "EpisodeRecord",
    "METHOD_SPECS",
    "MethodSpec",
    "PROTOCOL",
    "auc_bootstrap_summary",
    "audit_rollout_artifact",
    "canonical_conditions",
    "conditions_for_axis",
    "corrector_checkpoint_path",
    "crossed_bootstrap_summary",
    "expected_episode_contracts",
    "grounding_cache_path",
    "identity_contract_payload",
    "load_audited_records",
    "load_protocol_config",
    "method_specs",
    "normalized_auc",
    "paired_auc_bootstrap_summary",
    "policy_checkpoint_path",
    "protocol_identity_manifest",
    "render_markdown",
    "resolve_repo_path",
    "rollout_artifact_paths",
    "rollout_output_dir",
    "summarize_records",
    "validate_protocol_config",
]


if __name__ == "__main__":
    raise SystemExit(main())
