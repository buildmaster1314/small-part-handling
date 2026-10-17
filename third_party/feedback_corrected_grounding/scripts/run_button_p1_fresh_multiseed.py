#!/usr/bin/env python3
"""Run the strict fresh button-press P1 multi-seed protocol.

This is a serial, job-boundary-resumable orchestrator. It reuses only complete
artifacts whose necessary configuration metadata is exact, and refuses partial
or mismatched outputs instead of overwriting them.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.button_fresh_multiseed_statistics import (  # noqa: E402
    CameraCondition,
    EpisodeRecord,
    MethodSpec,
    PROTOCOL,
    audit_rollout_artifact,
    canonical_conditions,
    corrector_checkpoint_path,
    grounding_cache_path,
    identity_contract_payload,
    load_protocol_config,
    method_specs,
    policy_checkpoint_path,
    protocol_identity_manifest,
    resolve_repo_path,
    rollout_artifact_paths,
    rollout_output_dir,
)


DEFAULT_CONFIG = REPO_ROOT / "config" / "experiments" / "button_p1_fresh_multiseed.json"
TRAIN_CORRECTOR_SCRIPT = REPO_ROOT / "scripts" / "train_transformer_residual.py"
CACHE_CORRECTIONS_SCRIPT = REPO_ROOT / "scripts" / "cache_fcpg_corrections.py"
BUILD_PROTOTYPE_SCRIPT = REPO_ROOT / "scripts" / "build_vit_prototype_bank.py"
TRAIN_POLICY_SCRIPT = REPO_ROOT / "scripts" / "train_action_chunk_policy.py"
ROLLOUT_SCRIPT = REPO_ROOT / "scripts" / "evaluate_action_chunk_rollout.py"
STAGES = ("corrector", "cache", "prototype", "policy", "rollout", "statistics")


@dataclass(frozen=True)
class InputContract:
    sample_count: int
    num_keypoints: int
    feature_dim: int


@dataclass(frozen=True)
class StageJob:
    stage: str
    key: str
    command: tuple[str, ...]
    output_dir: Path
    required_paths: tuple[Path, ...]
    metadata: Mapping[str, object]


def _at(config: Mapping[str, object], path: str) -> object:
    current: object = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"protocol config is missing {path}")
        current = current[part]
    return current


def _string(value: object) -> str:
    return str(value)


def _number(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _same_path(actual: object, expected: Path) -> bool:
    if not isinstance(actual, str) or not actual:
        return False
    path = Path(actual).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve() == expected.resolve()


def _required_args(
    actual: Mapping[str, object], expected: Mapping[str, object], label: str
) -> None:
    for key, expected_value in expected.items():
        if key not in actual:
            raise ValueError(f"{label} metadata is missing args.{key}")
        actual_value = actual[key]
        if isinstance(expected_value, list):
            matches = isinstance(actual_value, (list, tuple)) and list(actual_value) == expected_value
        elif isinstance(expected_value, float):
            try:
                matches = float(actual_value) == expected_value
            except (TypeError, ValueError):
                matches = False
        else:
            matches = type(actual_value) is type(expected_value) and actual_value == expected_value
        if not matches:
            raise ValueError(
                f"{label} args.{key} must be {expected_value!r}, got {actual_value!r}"
            )


def validate_input_artifacts(
    config: Mapping[str, object], repo_root: Path = REPO_ROOT
) -> InputContract:
    """Check only the dataset/cache contract required by this protocol."""

    data_path = resolve_repo_path(repo_root, _string(_at(config, "inputs.data")))
    vision_cache = resolve_repo_path(
        repo_root, _string(_at(config, "inputs.vision_cache"))
    )
    if not data_path.is_file():
        raise FileNotFoundError(f"button projection dataset is missing: {data_path}")
    required_npz = {
        "q_nominal",
        "q_true",
        "episode_id",
        "frame_id",
        "task_name",
        "rgb",
        "obs",
        "action",
        "observation_action_alignment",
    }
    with np.load(data_path, allow_pickle=False) as data:
        missing = sorted(required_npz.difference(data.files))
        if missing:
            raise ValueError(f"button projection dataset is missing arrays: {missing}")
        q_nominal = np.asarray(data["q_nominal"])
        q_true = np.asarray(data["q_true"])
        episode_id = np.asarray(data["episode_id"])
        frame_id = np.asarray(data["frame_id"])
        task_names = {str(value) for value in np.asarray(data["task_name"]).reshape(-1)}
        observation_action_alignment = str(
            np.asarray(data["observation_action_alignment"]).item()
        )
    if q_nominal.ndim != 3 or q_nominal.shape[-1] != 2:
        raise ValueError("q_nominal must have shape [sample, keypoint, 2]")
    if q_true.shape != q_nominal.shape:
        raise ValueError("q_true must match q_nominal")
    sample_count, num_keypoints, _ = q_nominal.shape
    if episode_id.shape[0] != sample_count or frame_id.shape[0] != sample_count:
        raise ValueError("dataset frame identity arrays are misaligned")
    if task_names != {"button-press-v3"}:
        raise ValueError(f"button dataset task_name is unexpected: {sorted(task_names)}")
    if observation_action_alignment != "pre_action":
        raise ValueError(
            "button dataset observation_action_alignment must be 'pre_action', "
            f"got {observation_action_alignment!r}"
        )

    metadata_path = vision_cache / "metadata.json"
    feature_path = vision_cache / "backbone_features.npy"
    prefix_path = vision_cache / "prefix.npy"
    if not all(path.is_file() for path in (metadata_path, feature_path, prefix_path)):
        raise FileNotFoundError(f"ViT cache is incomplete: {vision_cache}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_metadata = {
        "sample_count": sample_count,
        "image_width": 224,
        "image_height": 224,
        "model_name": _at(config, "inputs.vision_model_name"),
        "selected_layers": _at(config, "inputs.vision_selected_layers"),
        "dtype": "float16",
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"ViT cache metadata {key} must be {expected!r}, got {metadata.get(key)!r}"
            )
    if not _same_path(metadata.get("source_data"), data_path):
        raise ValueError("ViT cache source_data does not match button dataset")
    feature_maps = np.load(feature_path, mmap_mode="r")
    prefix = np.load(prefix_path, mmap_mode="r")
    if feature_maps.ndim != 5 or feature_maps.shape[0] != sample_count:
        raise ValueError("ViT backbone cache has an invalid sample shape")
    if prefix.ndim not in {2, 3} or prefix.shape[0] != sample_count:
        raise ValueError("ViT prefix cache has an invalid sample shape")
    if feature_maps.shape[1] != len(_at(config, "inputs.vision_selected_layers")):
        raise ValueError("ViT cache layer count differs from protocol")
    return InputContract(sample_count, num_keypoints, int(feature_maps.shape[2]))


def expected_corrector_args(config: Mapping[str, object], seed: int) -> dict[str, object]:
    return {
        "corrector_arch": _at(config, "corrector.architecture"),
        "obs_mode": _at(config, "corrector.obs_mode"),
        "model_name": _at(config, "inputs.vision_model_name"),
        "vision_selected_layers": _at(config, "inputs.vision_selected_layers"),
        "vision_dim": _at(config, "corrector.vision_dim"),
        "model_dim": _at(config, "corrector.model_dim"),
        "local_grid_size": _at(config, "corrector.local_grid_size"),
        "local_radius_px": _at(config, "corrector.local_radius_px"),
        "local_layers": _at(config, "corrector.local_layers"),
        "global_layers": _at(config, "corrector.global_layers"),
        "heads": _at(config, "corrector.heads"),
        "dropout": _at(config, "corrector.dropout"),
        "max_residual_px": _at(config, "corrector.max_residual_px"),
        "reliability_weight": _at(config, "corrector.reliability_weight"),
        "reliability_corruption_prob": _at(
            config, "corrector.reliability_corruption_probability"
        ),
        "sigreg_weight": _at(config, "corrector.sigreg_weight"),
        "epochs": _at(config, "corrector.epochs"),
        "batch_size": _at(config, "corrector.batch_size"),
        "lr": _at(config, "corrector.learning_rate"),
        "weight_decay": _at(config, "corrector.weight_decay"),
        "warmup_epochs": _at(config, "corrector.warmup_epochs"),
        "val_fraction": _at(config, "corrector.validation_fraction"),
        "split_unit": _at(config, "corrector.split_unit"),
        "seed": seed,
        "split_seed": seed,
    }


def validate_corrector_checkpoint_payload(
    payload: Mapping[str, object],
    config: Mapping[str, object],
    seed: int,
    *,
    repo_root: Path = REPO_ROOT,
) -> None:
    if payload.get("schema_version") != 2:
        raise ValueError("corrector checkpoint schema_version must be exactly 2")
    if payload.get("checkpoint_type") != "fcpg_corrector":
        raise ValueError("corrector checkpoint_type is invalid")
    if payload.get("corrector_arch") != "lewm" or payload.get("has_reliability") is not False:
        raise ValueError("corrector must be residual-only LeWM without reliability")
    data_path = resolve_repo_path(repo_root, _string(_at(config, "inputs.data")))
    vision_cache = resolve_repo_path(
        repo_root, _string(_at(config, "inputs.vision_cache"))
    )
    if not _same_path(payload.get("data"), data_path):
        raise ValueError("corrector checkpoint data path is incorrect")
    if not _same_path(payload.get("vision_cache"), vision_cache):
        raise ValueError("corrector checkpoint vision cache path is incorrect")
    args = payload.get("args")
    if not isinstance(args, Mapping):
        raise ValueError("corrector checkpoint args metadata is missing")
    _required_args(args, expected_corrector_args(config, seed), "corrector checkpoint")
    if not _same_path(args.get("data"), data_path) or not _same_path(
        args.get("vision_cache"), vision_cache
    ):
        raise ValueError("corrector args input paths are incorrect")
    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping) or not model_state:
        raise ValueError("corrector checkpoint model_state is missing")


def expected_policy_args(
    config: Mapping[str, object], family: str, seed: int
) -> dict[str, object]:
    if family not in {"fcpg", "jitter"}:
        raise ValueError(f"unknown policy family: {family}")
    jitter_std = float(_at(config, "jitter.standard_deviation_px")) if family == "jitter" else 0.0
    jitter_max = float(_at(config, "jitter.maximum_px")) if family == "jitter" else 0.0
    return {
        "mode": "fcpg" if family == "fcpg" else "geoprop",
        "policy_arch": _at(config, "policy.architecture"),
        "reliability_override": "native",
        "vision_model_name": _at(config, "inputs.vision_model_name"),
        "vision_selected_layers": _at(config, "inputs.vision_selected_layers"),
        "obs_mode": _at(config, "policy.obs_mode"),
        "obs_horizon": _at(config, "policy.obs_horizon"),
        "action_horizon": _at(config, "policy.action_horizon"),
        "vision_dim": _at(config, "policy.vision_dim"),
        "model_dim": _at(config, "policy.model_dim"),
        "encoder_layers": _at(config, "policy.encoder_layers"),
        "decoder_layers": _at(config, "policy.decoder_layers"),
        "heads": _at(config, "policy.heads"),
        "dropout": _at(config, "policy.dropout"),
        "epochs": _at(config, "policy.epochs"),
        "batch_size": _at(config, "policy.batch_size"),
        "lr": _at(config, "policy.learning_rate"),
        "weight_decay": _at(config, "policy.weight_decay"),
        "warmup_steps": _at(config, "policy.warmup_steps"),
        "latent_pred_weight": _at(config, "policy.latent_prediction_weight"),
        "sigreg_weight": _at(config, "policy.sigreg_weight"),
        "patience": _at(config, "policy.patience"),
        "projection_jitter_std_px": jitter_std,
        "projection_jitter_max_px": jitter_max,
        "seed": seed,
        "split_seed": seed,
    }


def validate_policy_checkpoint_payload(
    payload: Mapping[str, object],
    config: Mapping[str, object],
    family: str,
    seed: int,
    *,
    repo_root: Path = REPO_ROOT,
) -> None:
    if payload.get("schema_version") != 2:
        raise ValueError("policy checkpoint schema_version must be exactly 2")
    if payload.get("checkpoint_type") != "fcpg_action_policy":
        raise ValueError("policy checkpoint_type is invalid")
    if payload.get("policy_arch") != "legacy":
        raise ValueError("policy checkpoint must use the legacy architecture")
    data_path = resolve_repo_path(repo_root, _string(_at(config, "inputs.data")))
    vision_cache = resolve_repo_path(
        repo_root, _string(_at(config, "inputs.vision_cache"))
    )
    if not _same_path(payload.get("data"), data_path):
        raise ValueError("policy checkpoint data path is incorrect")
    if not _same_path(payload.get("vision_cache"), vision_cache):
        raise ValueError("policy checkpoint vision cache path is incorrect")
    args = payload.get("args")
    if not isinstance(args, Mapping):
        raise ValueError("policy checkpoint args metadata is missing")
    _required_args(args, expected_policy_args(config, family, seed), "policy checkpoint")
    if not _same_path(args.get("data"), data_path) or not _same_path(
        args.get("vision_cache"), vision_cache
    ):
        raise ValueError("policy args input paths are incorrect")
    if family == "fcpg":
        expected_corrector = corrector_checkpoint_path(config, repo_root, seed)
        expected_cache = grounding_cache_path(config, repo_root, seed)
        if not _same_path(args.get("fcpg_checkpoint"), expected_corrector):
            raise ValueError("FCPG policy corrector seed/path is incorrect")
        if not _same_path(args.get("grounding_cache"), expected_cache):
            raise ValueError("FCPG policy grounding cache seed/path is incorrect")
        if not _same_path(payload.get("grounding_cache"), expected_cache):
            raise ValueError("FCPG policy top-level grounding cache is incorrect")
    else:
        if args.get("fcpg_checkpoint") not in {None, ""}:
            raise ValueError("jitter policy unexpectedly records an FCPG checkpoint")
        if args.get("grounding_cache") not in {None, ""}:
            raise ValueError("jitter policy unexpectedly records a grounding cache")
        if payload.get("grounding_cache") is not None:
            raise ValueError("jitter policy top-level grounding cache must be null")
    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping) or not model_state:
        raise ValueError("policy checkpoint model_state is missing")


def _load_checkpoint(path: Path) -> Mapping[str, object]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required to validate checkpoint metadata") from error
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint must contain a mapping: {path}")
    return payload


def validate_training_metrics(
    path: Path,
    *,
    expected_epochs: int,
    architecture_key: str,
    expected_architecture: str,
    label: str,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError(f"{label} metrics schema_version must be exactly 2")
    if payload.get(architecture_key) != expected_architecture:
        raise ValueError(f"{label} metrics {architecture_key} is incorrect")
    if payload.get("complete") is not True:
        raise ValueError(f"{label} metrics are not marked complete")
    history = payload.get("history")
    if not isinstance(history, list) or not all(
        isinstance(row, Mapping) for row in history
    ):
        raise ValueError(f"{label} metrics history is invalid")
    observed_epochs = [row.get("epoch") for row in history]
    expected_sequence = list(range(1, expected_epochs + 1))
    if observed_epochs != expected_sequence:
        raise ValueError(
            f"{label} metrics must contain epochs 1..{expected_epochs}, "
            f"got {observed_epochs[:3]}...{observed_epochs[-3:]}"
        )
    best = payload.get("best")
    if not isinstance(best, Mapping) or best.get("epoch") not in expected_sequence:
        raise ValueError(f"{label} metrics best epoch is invalid")


def _corrector_required_paths(checkpoint: Path) -> tuple[Path, ...]:
    if not checkpoint.name.endswith("_best.pt"):
        raise ValueError(f"corrector best checkpoint name is invalid: {checkpoint}")
    final = checkpoint.with_name(checkpoint.name.replace("_best.pt", ".pt"))
    return checkpoint, final, checkpoint.parent / "metrics.json"


def _policy_required_paths(checkpoint: Path, family: str) -> tuple[Path, ...]:
    mode = "fcpg" if family == "fcpg" else "geoprop"
    return (
        checkpoint,
        checkpoint.parent / f"{mode}_action_chunk.pt",
        checkpoint.parent / f"{mode}_metrics.json",
    )


def _corrector_command(
    config: Mapping[str, object], seed: int, checkpoint: Path, device: str
) -> tuple[str, ...]:
    data = resolve_repo_path(REPO_ROOT, _string(_at(config, "inputs.data")))
    cache = resolve_repo_path(REPO_ROOT, _string(_at(config, "inputs.vision_cache")))
    return (
        sys.executable,
        str(TRAIN_CORRECTOR_SCRIPT),
        "--data",
        str(data),
        "--output-dir",
        str(checkpoint.parent),
        "--corrector-arch",
        _string(_at(config, "corrector.architecture")),
        "--obs-mode",
        _string(_at(config, "corrector.obs_mode")),
        "--model-name",
        _string(_at(config, "inputs.vision_model_name")),
        "--vision-cache",
        str(cache),
        "--vision-selected-layers",
        ",".join(str(value) for value in _at(config, "inputs.vision_selected_layers")),
        "--vision-dim",
        _number(_at(config, "corrector.vision_dim")),
        "--model-dim",
        _number(_at(config, "corrector.model_dim")),
        "--local-grid-size",
        _number(_at(config, "corrector.local_grid_size")),
        "--local-radius-px",
        _number(_at(config, "corrector.local_radius_px")),
        "--local-layers",
        _number(_at(config, "corrector.local_layers")),
        "--global-layers",
        _number(_at(config, "corrector.global_layers")),
        "--heads",
        _number(_at(config, "corrector.heads")),
        "--dropout",
        _number(_at(config, "corrector.dropout")),
        "--max-residual-px",
        _number(_at(config, "corrector.max_residual_px")),
        "--reliability-weight",
        _number(_at(config, "corrector.reliability_weight")),
        "--reliability-corruption-prob",
        _number(_at(config, "corrector.reliability_corruption_probability")),
        "--sigreg-weight",
        _number(_at(config, "corrector.sigreg_weight")),
        "--epochs",
        _number(_at(config, "corrector.epochs")),
        "--batch-size",
        _number(_at(config, "corrector.batch_size")),
        "--num-workers",
        _number(_at(config, "corrector.num_workers")),
        "--lr",
        _number(_at(config, "corrector.learning_rate")),
        "--weight-decay",
        _number(_at(config, "corrector.weight_decay")),
        "--warmup-epochs",
        _number(_at(config, "corrector.warmup_epochs")),
        "--val-fraction",
        _number(_at(config, "corrector.validation_fraction")),
        "--split-unit",
        _string(_at(config, "corrector.split_unit")),
        "--seed",
        str(seed),
        "--split-seed",
        str(seed),
        "--device",
        device,
    )


def _cache_command(
    config: Mapping[str, object], seed: int, output_dir: Path, device: str
) -> tuple[str, ...]:
    return (
        sys.executable,
        str(CACHE_CORRECTIONS_SCRIPT),
        "--data",
        str(resolve_repo_path(REPO_ROOT, _string(_at(config, "inputs.data")))),
        "--vision-cache",
        str(
            resolve_repo_path(REPO_ROOT, _string(_at(config, "inputs.vision_cache")))
        ),
        "--checkpoint",
        str(corrector_checkpoint_path(config, REPO_ROOT, seed)),
        "--output-dir",
        str(output_dir),
        "--image-key",
        "rgb",
        "--batch-size",
        _number(_at(config, "corrector.batch_size")),
        "--device",
        device,
    )


def _prototype_command(
    config: Mapping[str, object], output_dir: Path, device: str
) -> tuple[str, ...]:
    return (
        sys.executable,
        str(BUILD_PROTOTYPE_SCRIPT),
        "--data",
        str(resolve_repo_path(REPO_ROOT, _string(_at(config, "inputs.data")))),
        "--keypoint-source",
        _string(_at(config, "prototype.keypoint_source")),
        "--vision-cache",
        str(
            resolve_repo_path(REPO_ROOT, _string(_at(config, "inputs.vision_cache")))
        ),
        "--output-dir",
        str(output_dir),
        "--prototypes-per-keypoint",
        _number(_at(config, "prototype.count_per_keypoint")),
        "--feature-level",
        _number(_at(config, "prototype.feature_level")),
        "--split-seed",
        _number(_at(config, "prototype.split_seed")),
        "--seed",
        _number(_at(config, "prototype.seed")),
        "--device",
        device,
    )


def _policy_command(
    config: Mapping[str, object], family: str, seed: int, checkpoint: Path, device: str
) -> tuple[str, ...]:
    data = resolve_repo_path(REPO_ROOT, _string(_at(config, "inputs.data")))
    vision_cache = resolve_repo_path(
        REPO_ROOT, _string(_at(config, "inputs.vision_cache"))
    )
    mode = "fcpg" if family == "fcpg" else "geoprop"
    command = [
        sys.executable,
        str(TRAIN_POLICY_SCRIPT),
        "--data",
        str(data),
        "--output-dir",
        str(checkpoint.parent),
        "--mode",
        mode,
        "--policy-arch",
        _string(_at(config, "policy.architecture")),
        "--vision-cache",
        str(vision_cache),
    ]
    if family == "fcpg":
        command.extend(
            (
                "--fcpg-checkpoint",
                str(corrector_checkpoint_path(config, REPO_ROOT, seed)),
                "--grounding-cache",
                str(grounding_cache_path(config, REPO_ROOT, seed)),
            )
        )
    command.extend(
        (
            "--reliability-override",
            "native",
            "--vision-model-name",
            _string(_at(config, "inputs.vision_model_name")),
            "--vision-selected-layers",
            ",".join(str(value) for value in _at(config, "inputs.vision_selected_layers")),
            "--obs-mode",
            _string(_at(config, "policy.obs_mode")),
            "--obs-horizon",
            _number(_at(config, "policy.obs_horizon")),
            "--action-horizon",
            _number(_at(config, "policy.action_horizon")),
            "--vision-dim",
            _number(_at(config, "policy.vision_dim")),
            "--model-dim",
            _number(_at(config, "policy.model_dim")),
            "--encoder-layers",
            _number(_at(config, "policy.encoder_layers")),
            "--decoder-layers",
            _number(_at(config, "policy.decoder_layers")),
            "--heads",
            _number(_at(config, "policy.heads")),
            "--dropout",
            _number(_at(config, "policy.dropout")),
            "--epochs",
            _number(_at(config, "policy.epochs")),
            "--batch-size",
            _number(_at(config, "policy.batch_size")),
            "--num-workers",
            _number(_at(config, "policy.num_workers")),
            "--lr",
            _number(_at(config, "policy.learning_rate")),
            "--weight-decay",
            _number(_at(config, "policy.weight_decay")),
            "--warmup-steps",
            _number(_at(config, "policy.warmup_steps")),
            "--latent-pred-weight",
            _number(_at(config, "policy.latent_prediction_weight")),
            "--sigreg-weight",
            _number(_at(config, "policy.sigreg_weight")),
            "--patience",
            _number(_at(config, "policy.patience")),
            "--projection-jitter-std-px",
            _number(
                _at(config, "jitter.standard_deviation_px") if family == "jitter" else 0.0
            ),
            "--projection-jitter-max-px",
            _number(_at(config, "jitter.maximum_px") if family == "jitter" else 0.0),
            "--seed",
            str(seed),
            "--split-seed",
            str(seed),
            "--device",
            device,
        )
    )
    return tuple(command)


def _rollout_command(
    config: Mapping[str, object],
    seed: int,
    condition: CameraCondition,
    method: MethodSpec,
    output_dir: Path,
    device: str,
) -> tuple[str, ...]:
    evaluation = _at(config, "evaluation")
    prototype = _at(config, "prototype")
    pnp = _at(config, "pnp")
    assert isinstance(evaluation, Mapping)
    assert isinstance(prototype, Mapping)
    assert isinstance(pnp, Mapping)
    command = [
        sys.executable,
        str(ROLLOUT_SCRIPT),
        "--checkpoint",
        str(policy_checkpoint_path(config, REPO_ROOT, method.policy_family, seed)),
        "--task",
        _string(_at(config, "task")),
        "--camera",
        _string(evaluation["camera"]),
        "--width",
        _number(evaluation["width"]),
        "--height",
        _number(evaluation["height"]),
        "--episodes",
        str(condition.episodes),
        "--task-index-offset",
        _number(evaluation["task_index_offset"]),
        "--horizon",
        _number(evaluation["horizon"]),
        "--action-steps",
        _number(evaluation["action_steps"]),
        "--translation-cm-min",
        _number(condition.translation_cm),
        "--translation-cm-max",
        _number(condition.translation_cm),
        "--rotation-deg-min",
        _number(condition.rotation_deg),
        "--rotation-deg-max",
        _number(condition.rotation_deg),
        "--clean-probability",
        "0",
        "--grounding-override",
        method.grounding_override,
        "--external-grounding",
        method.external_grounding,
        "--prototype-radius-px",
        _number(prototype["radius_px"]),
        "--prototype-grid-size",
        _number(prototype["grid_size"]),
        "--pnp-history-frames",
        _number(pnp["history_frames"]),
        "--pnp-max-reprojection-error-px",
        _number(pnp["maximum_reprojection_error_px"]),
        "--pnp-max-pose-rotation-deg",
        _number(pnp["maximum_pose_rotation_deg"]),
        "--pnp-max-pose-translation-cm",
        _number(pnp["maximum_pose_translation_cm"]),
        "--reliability-override",
        "native",
        "--seed",
        _number(_at(config, "seeds.runner")),
        "--env-seed-base",
        _number(evaluation["env_seed_base"]),
        "--device",
        device,
        "--output-dir",
        str(output_dir),
    ]
    if method.external_grounding != "none":
        command.extend(
            (
                "--prototype-bank",
                str(
                    resolve_repo_path(
                        REPO_ROOT, _string(_at(config, "artifacts.prototype_bank"))
                    )
                ),
            )
        )
    return tuple(command)


def build_stage_jobs(
    config: Mapping[str, object],
    device: str,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, tuple[StageJob, ...]]:
    seeds = tuple(int(value) for value in _at(config, "seeds.policy"))
    corrector_jobs = []
    cache_jobs = []
    policy_jobs = []
    rollout_jobs = []
    for seed in seeds:
        checkpoint = corrector_checkpoint_path(config, REPO_ROOT, seed)
        corrector_jobs.append(
            StageJob(
                "corrector",
                f"corrector_seed_{seed}",
                _corrector_command(config, seed, checkpoint, device),
                checkpoint.parent,
                _corrector_required_paths(checkpoint),
                {"seed": seed},
            )
        )
        cache_dir = grounding_cache_path(config, REPO_ROOT, seed)
        cache_jobs.append(
            StageJob(
                "cache",
                f"cache_seed_{seed}",
                _cache_command(config, seed, cache_dir, device),
                cache_dir,
                (cache_dir / "metadata.json", cache_dir / "q_corrected.npy"),
                {"seed": seed},
            )
        )
        for family in ("fcpg", "jitter"):
            policy_checkpoint = policy_checkpoint_path(config, REPO_ROOT, family, seed)
            policy_jobs.append(
                StageJob(
                    "policy",
                    f"{family}_policy_seed_{seed}",
                    _policy_command(config, family, seed, policy_checkpoint, device),
                    policy_checkpoint.parent,
                    _policy_required_paths(policy_checkpoint, family),
                    {"seed": seed, "family": family},
                )
            )
        for condition in canonical_conditions(config):
            for method in method_specs(config):
                output_dir = rollout_output_dir(
                    config, REPO_ROOT, seed, condition, method
                )
                rollout_jobs.append(
                    StageJob(
                        "rollout",
                        f"seed_{seed}_{condition.condition_id}_{method.name}",
                        _rollout_command(
                            config, seed, condition, method, output_dir, device
                        ),
                        output_dir,
                        (
                            *rollout_artifact_paths(output_dir, method),
                            output_dir / "command.json",
                        ),
                        {"seed": seed, "condition": condition, "method": method},
                    )
                )

    prototype_dir = resolve_repo_path(
        REPO_ROOT, _string(_at(config, "artifacts.prototype_bank"))
    )
    prototype_job = StageJob(
        "prototype",
        "prototype_128_seed_0",
        _prototype_command(config, prototype_dir, device),
        prototype_dir,
        (
            prototype_dir / "metadata.json",
            prototype_dir / "prototypes.npy",
            prototype_dir / "source_indices.npy",
        ),
        {},
    )
    statistics_dir = resolve_repo_path(
        REPO_ROOT, _string(_at(config, "artifacts.statistics_root"))
    )
    statistics_job = StageJob(
        "statistics",
        "crossed_bootstrap_statistics",
        (
            sys.executable,
            "-m",
            "fcpg.button_fresh_multiseed_statistics",
            "--config",
            str(config_path.resolve()),
        ),
        statistics_dir,
        (
            statistics_dir / "button_p1_fresh_multiseed_statistics.json",
            statistics_dir / "button_p1_fresh_multiseed_statistics.md",
        ),
        {},
    )
    jobs = {
        "corrector": tuple(corrector_jobs),
        "cache": tuple(cache_jobs),
        "prototype": (prototype_job,),
        "policy": tuple(policy_jobs),
        "rollout": tuple(rollout_jobs),
        "statistics": (statistics_job,),
    }
    keys = [job.key for stage in STAGES for job in jobs[stage]]
    if len(keys) != len(set(keys)):
        raise ValueError("stage plan contains duplicate job keys")
    output_dirs = [job.output_dir for stage in STAGES for job in jobs[stage]]
    if len(output_dirs) != len(set(output_dirs)):
        raise ValueError("stage plan contains duplicate output directories")
    return jobs


def validate_grounding_cache(
    config: Mapping[str, object], seed: int, contract: InputContract
) -> None:
    directory = grounding_cache_path(config, REPO_ROOT, seed)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") not in {2, 3}:
        raise ValueError("grounding cache schema_version must be 2 or 3")
    if metadata.get("has_reliability") is not False:
        raise ValueError("LeWM grounding cache must not contain reliability")
    if metadata.get("corrector_arch") != "lewm":
        raise ValueError("grounding cache corrector_arch must be lewm")
    if metadata.get("checkpoint_schema_version") != 2:
        raise ValueError("grounding cache corrector schema must be 2")
    expected_paths = {
        "source_data": resolve_repo_path(REPO_ROOT, _string(_at(config, "inputs.data"))),
        "vision_cache": resolve_repo_path(
            REPO_ROOT, _string(_at(config, "inputs.vision_cache"))
        ),
        "checkpoint": corrector_checkpoint_path(config, REPO_ROOT, seed),
    }
    for key, expected in expected_paths.items():
        if not _same_path(metadata.get(key), expected):
            raise ValueError(f"grounding cache {key} path is incorrect for seed {seed}")
    if metadata.get("sample_count") != contract.sample_count:
        raise ValueError("grounding cache sample_count is incorrect")
    if metadata.get("num_keypoints") != contract.num_keypoints:
        raise ValueError("grounding cache num_keypoints is incorrect")
    corrected = np.load(directory / "q_corrected.npy", mmap_mode="r")
    if corrected.shape != (contract.sample_count, contract.num_keypoints, 2):
        raise ValueError("grounding cache q_corrected shape is incorrect")
    if corrected.dtype != np.float32:
        raise ValueError("grounding cache q_corrected dtype must be float32")
    if (directory / "reliability.npy").exists():
        raise ValueError("residual-only grounding cache must not contain reliability.npy")


def validate_prototype_bank(
    config: Mapping[str, object], contract: InputContract
) -> None:
    directory = resolve_repo_path(
        REPO_ROOT, _string(_at(config, "artifacts.prototype_bank"))
    )
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    expected = {
        "version": 1,
        "baseline": "frozen_vit_prototype_refinement",
        "keypoint_source": _at(config, "prototype.keypoint_source"),
        "model_name": _at(config, "inputs.vision_model_name"),
        "selected_layers": _at(config, "inputs.vision_selected_layers"),
        "feature_level": _at(config, "prototype.feature_level"),
        "feature_layer": 11,
        "image_width": 224,
        "image_height": 224,
        "split_unit": "task-episode",
        "split_seed": _at(config, "prototype.split_seed"),
        "prototype_seed": _at(config, "prototype.seed"),
        "prototype_count": _at(config, "prototype.count_per_keypoint"),
        "num_keypoints": contract.num_keypoints,
        "feature_dim": contract.feature_dim,
        "training_tasks": ["button-press-v3"],
    }
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f"prototype metadata {key} must be {expected_value!r}, "
                f"got {metadata.get(key)!r}"
            )
    for key, expected_path in {
        "source_data": resolve_repo_path(REPO_ROOT, _string(_at(config, "inputs.data"))),
        "vision_cache": resolve_repo_path(
            REPO_ROOT, _string(_at(config, "inputs.vision_cache"))
        ),
    }.items():
        if not _same_path(metadata.get(key), expected_path):
            raise ValueError(f"prototype {key} path is incorrect")
    prototypes = np.load(directory / "prototypes.npy", mmap_mode="r")
    if prototypes.shape != (
        contract.num_keypoints,
        int(_at(config, "prototype.count_per_keypoint")),
        contract.feature_dim,
    ):
        raise ValueError("prototype array shape is incorrect")
    indices = np.load(directory / "source_indices.npy", mmap_mode="r")
    if indices.shape != (int(_at(config, "prototype.count_per_keypoint")),):
        raise ValueError("prototype source_indices shape is incorrect")


def validate_statistics_artifact(config: Mapping[str, object], job: StageJob) -> None:
    payload = json.loads(job.required_paths[0].read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("protocol") != PROTOCOL:
        raise ValueError("statistics output schema/protocol is incorrect")
    if payload.get("task") != _at(config, "task"):
        raise ValueError("statistics task is incorrect")
    if payload.get("policy_seeds") != _at(config, "seeds.policy"):
        raise ValueError("statistics policy seeds are incorrect")
    if payload.get("methods") != _at(config, "evaluation.methods"):
        raise ValueError("statistics methods are incorrect")
    design = payload.get("design")
    if not isinstance(design, Mapping):
        raise ValueError("statistics bootstrap design is missing")
    expected_design = {
        "task": "fixed_not_resampled",
        "resampled_dimensions": ["policy_seed", "shared_episode_identity"],
        "bootstrap": "two_way_crossed_cartesian_resampling",
        "bootstrap_samples": _at(config, "statistics.bootstrap_samples"),
        "bootstrap_seed": _at(config, "statistics.bootstrap_seed"),
        "confidence_level": _at(config, "statistics.confidence_level"),
    }
    for key, expected in expected_design.items():
        if design.get(key) != expected:
            raise ValueError(f"statistics design {key} is incorrect")
    if len(payload.get("severity_results", [])) != 6:
        raise ValueError("statistics must contain all six registered severity aliases")
    auc = payload.get("auc_results")
    if not isinstance(auc, Mapping) or set(auc) != {"rotation_deg", "translation_cm"}:
        raise ValueError("statistics AUC axes are incomplete")
    ni = payload.get("clean_noninferiority")
    if not isinstance(ni, Mapping) or ni.get("margin_success_probability") != -0.03:
        raise ValueError("statistics clean NI preregistration is missing")


def validate_job(
    config: Mapping[str, object], job: StageJob, contract: InputContract
) -> tuple[EpisodeRecord, ...] | None:
    if job.stage == "corrector":
        seed = int(job.metadata["seed"])
        for path in job.required_paths[:2]:
            validate_corrector_checkpoint_payload(_load_checkpoint(path), config, seed)
        validate_training_metrics(
            job.required_paths[2],
            expected_epochs=int(_at(config, "corrector.epochs")),
            architecture_key="corrector_arch",
            expected_architecture=_string(_at(config, "corrector.architecture")),
            label=f"corrector seed {seed}",
        )
        return None
    if job.stage == "cache":
        validate_grounding_cache(config, int(job.metadata["seed"]), contract)
        return None
    if job.stage == "prototype":
        validate_prototype_bank(config, contract)
        return None
    if job.stage == "policy":
        seed = int(job.metadata["seed"])
        family = str(job.metadata["family"])
        for path in job.required_paths[:2]:
            validate_policy_checkpoint_payload(_load_checkpoint(path), config, family, seed)
        validate_training_metrics(
            job.required_paths[2],
            expected_epochs=int(_at(config, "policy.epochs")),
            architecture_key="policy_arch",
            expected_architecture=_string(_at(config, "policy.architecture")),
            label=f"{family} policy seed {seed}",
        )
        return None
    if job.stage == "rollout":
        condition = job.metadata["condition"]
        method = job.metadata["method"]
        assert isinstance(condition, CameraCondition)
        assert isinstance(method, MethodSpec)
        command_payload = json.loads(
            (job.output_dir / "command.json").read_text(encoding="utf-8")
        )
        expected_command = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "stage": job.stage,
            "job": job.key,
            "command": list(job.command),
        }
        if command_payload != expected_command:
            raise ValueError(f"rollout command contract differs for {job.key}")
        return audit_rollout_artifact(
            config,
            REPO_ROOT,
            int(job.metadata["seed"]),
            condition,
            method,
            require_identity_contract=(job.output_dir / "identity_contract.json").exists(),
        )
    if job.stage == "statistics":
        validate_statistics_artifact(config, job)
        return None
    raise ValueError(f"unknown job stage: {job.stage}")


def artifact_presence(job: StageJob) -> str:
    exists = [path.exists() for path in job.required_paths]
    if all(exists):
        return "complete"
    if any(exists):
        missing = [str(path) for path, present in zip(job.required_paths, exists) if not present]
        raise RuntimeError(f"partial artifact for {job.key}; missing={missing}")
    if job.output_dir.exists() and any(job.output_dir.iterdir()):
        raise RuntimeError(
            f"output directory for {job.key} is non-empty without a complete artifact: "
            f"{job.output_dir}"
        )
    return "missing"


def _atomic_json_write(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _strict_json_sidecar(payload: object, path: Path) -> None:
    if path.exists():
        submitted = json.loads(path.read_text(encoding="utf-8"))
        if submitted != payload:
            raise ValueError(f"existing sidecar contract differs: {path}")
        return
    _atomic_json_write(payload, path)


def execute_job(
    config: Mapping[str, object],
    job: StageJob,
    contract: InputContract,
    *,
    dry_run: bool,
) -> tuple[str, tuple[EpisodeRecord, ...] | None]:
    presence = artifact_presence(job)
    if presence == "complete":
        result = validate_job(config, job, contract)
        print(f"[reuse] {job.stage}:{job.key}")
        return "reused", result
    if dry_run:
        print(f"[would-run] {job.stage}:{job.key} {shlex.join(job.command)}")
        return "planned", None

    job.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json_write(
        {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "stage": job.stage,
            "job": job.key,
            "command": list(job.command),
        },
        job.output_dir / "command.json",
    )
    environment = os.environ.copy()
    environment.setdefault("MUJOCO_GL", "egl")
    log_path = job.output_dir / "run.log"
    print(f"[run] {job.stage}:{job.key}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            job.command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"{job.stage}:{job.key} failed with exit code {completed.returncode}; "
            f"see {log_path}"
        )
    if artifact_presence(job) != "complete":
        raise RuntimeError(f"{job.stage}:{job.key} did not produce a complete artifact")
    result = validate_job(config, job, contract)
    return "completed", result


def _require_complete_jobs(
    config: Mapping[str, object],
    jobs: Sequence[StageJob],
    contract: InputContract,
) -> None:
    for job in jobs:
        if artifact_presence(job) != "complete":
            raise FileNotFoundError(f"required dependency is missing: {job.key}")
        validate_job(config, job, contract)


def _validate_stage_dependencies(
    config: Mapping[str, object],
    stage: str,
    jobs: Mapping[str, Sequence[StageJob]],
    contract: InputContract,
) -> None:
    dependencies = {
        "corrector": (),
        "cache": ("corrector",),
        "prototype": (),
        "policy": ("corrector", "cache"),
        "rollout": ("policy", "prototype"),
        "statistics": ("rollout",),
    }
    for dependency in dependencies[stage]:
        _require_complete_jobs(config, jobs[dependency], contract)


def write_rollout_identity_contracts(
    config: Mapping[str, object],
    rollout_jobs: Sequence[StageJob],
    audited: Mapping[str, tuple[EpisodeRecord, ...]],
) -> Path:
    all_records = []
    for job in rollout_jobs:
        records = audited.get(job.key)
        if records is None:
            raise RuntimeError(f"rollout audit result is missing for {job.key}")
        condition = job.metadata["condition"]
        method = job.metadata["method"]
        seed = int(job.metadata["seed"])
        assert isinstance(condition, CameraCondition)
        assert isinstance(method, MethodSpec)
        payload = identity_contract_payload(config, seed, condition, method, records)
        _strict_json_sidecar(payload, job.output_dir / "identity_contract.json")
        all_records.extend(records)
    manifest = protocol_identity_manifest(config, all_records)
    output_root = resolve_repo_path(
        REPO_ROOT, _string(_at(config, "artifacts.output_root"))
    )
    manifest_path = output_root / "episode_identity_manifest.json"
    _strict_json_sidecar(manifest, manifest_path)
    return manifest_path


def parse_stages(value: str) -> tuple[str, ...]:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    if not requested:
        raise argparse.ArgumentTypeError("--stages must not be empty")
    if requested == ["all"]:
        return STAGES
    if "all" in requested:
        raise argparse.ArgumentTypeError("all cannot be combined with named stages")
    unknown = sorted(set(requested).difference(STAGES))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown stages: {unknown}")
    if len(requested) != len(set(requested)):
        raise argparse.ArgumentTypeError("--stages contains duplicates")
    return tuple(stage for stage in STAGES if stage in requested)


def run_protocol(
    config: Mapping[str, object],
    *,
    stages: Sequence[str],
    device: str,
    dry_run: bool,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, int]:
    contract = validate_input_artifacts(config)
    jobs = build_stage_jobs(config, device, config_path=config_path)
    counts = {"planned": 0, "reused": 0, "completed": 0}
    for stage in stages:
        if stage not in STAGES:
            raise ValueError(f"unknown stage: {stage}")
        if not dry_run:
            _validate_stage_dependencies(config, stage, jobs, contract)
        rollout_audits: dict[str, tuple[EpisodeRecord, ...]] = {}
        for job in jobs[stage]:
            status, audit = execute_job(
                config, job, contract, dry_run=dry_run
            )
            counts[status] += 1
            if audit is not None:
                rollout_audits[job.key] = audit
        if stage == "rollout" and not dry_run:
            manifest_path = write_rollout_identity_contracts(
                config, jobs["rollout"], rollout_audits
            )
            print(f"[identity-manifest] {manifest_path}")
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stages",
        type=parse_stages,
        default=STAGES,
        metavar="STAGE[,STAGE...]",
        help=f"available stages: {', '.join(STAGES)}; default: all",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    config_path = resolve_repo_path(REPO_ROOT, args.config)
    config = load_protocol_config(config_path)
    counts = run_protocol(
        config,
        stages=args.stages,
        device=args.device,
        dry_run=args.dry_run,
        config_path=config_path,
    )
    print(json.dumps({"protocol": PROTOCOL, "stages": list(args.stages), **counts}, indent=2))
    return 0


__all__ = [
    "DEFAULT_CONFIG",
    "InputContract",
    "STAGES",
    "StageJob",
    "artifact_presence",
    "build_stage_jobs",
    "execute_job",
    "expected_corrector_args",
    "expected_policy_args",
    "parse_stages",
    "run_protocol",
    "validate_corrector_checkpoint_payload",
    "validate_grounding_cache",
    "validate_input_artifacts",
    "validate_policy_checkpoint_payload",
    "validate_prototype_bank",
    "write_rollout_identity_contracts",
]


if __name__ == "__main__":
    raise SystemExit(main())
