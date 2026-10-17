"""Bounded same-snapshot runtime for the exploratory post-L1 L2 pilot.

This module performs inference and simulator branching only. It deliberately
contains no optimizer, loss backward pass, or checkpoint-writing path.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import pickle
import platform
import random
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fcpg.lewm_counterfactual import (
    PILOT_ACTION_DIM,
    PILOT_CANDIDATE_NAMES,
    PILOT_HORIZONS,
    PILOT_NOISE_SCALES,
    PILOT_RAW_ACTION_STEPS,
    PILOT_TRANSITIONS,
    PILOT_TRANSITION_ACTION_DIM,
    ActionCandidates,
    build_action_candidates,
    counterfactual_ranking_metrics,
    paired_task_macro_bootstrap,
    reshape_action_chunks,
    task_macro_bootstrap,
)


PROTOCOL = "fcpg_grounded_lewm_post_l1_l2_counterfactual_pilot"
SCOPE = "exploratory_conditional_post_l1_not_retroactive_gate_l2"
EXPECTED_TASKS = (
    "pick-place-v3",
    "button-press-v3",
    "reach-v3",
    "push-v3",
)
EXPECTED_CONDITIONS = ("clean", "rotation5")
TRAIN_GROUNDINGS = ("nominal", "fcpg")
EVAL_GROUNDINGS = ("nominal", "fcpg")
FORBIDDEN_DEPLOYMENT_KEYS = frozenset(
    {
        "q_true",
        "valid_true",
        "oracle_target",
        "goal_oracle_target",
        "actual_oracle_target",
        "reward",
        "return",
        "info",
        "success",
    }
)
SUMMARY_METRICS = (
    "spearman",
    "top1_hit",
    "topk_hit",
    "expert_top1",
    "expert_topk",
    "expert_pairwise_accuracy",
    "top1_regret",
    "topk_regret",
    "normalized_top1_regret",
    "normalized_topk_regret",
    "return_spearman",
    "progress_spearman",
    "oracle_progress_spearman",
)
LOWER_IS_BETTER = frozenset(
    {
        "top1_regret",
        "topk_regret",
        "normalized_top1_regret",
        "normalized_topk_regret",
    }
)


@dataclass(frozen=True)
class FrameRecord:
    """One pre-action or branch-outcome frame."""

    rgb: np.ndarray
    proprio: np.ndarray
    keypoints_xyz: np.ndarray
    q_nominal: np.ndarray
    valid_nominal: np.ndarray
    q_true: np.ndarray
    valid_true: np.ndarray


@dataclass
class EpisodePlan:
    env: Any
    task: str
    condition: str
    episode: int
    env_seed: int
    nominal_extrinsic: np.ndarray
    intrinsic: np.ndarray
    snapshot: Any
    snapshot_step: int
    snapshot_sha256: str
    physical_snapshot_sha256: str
    records: list[FrameRecord]
    actions: np.ndarray
    goal_record: FrameRecord
    goal_info: dict[str, Any]


@dataclass(frozen=True)
class FrozenInputs:
    models: Mapping[str, Any]
    policy: Any
    policy_args: Mapping[str, Any]
    corrector: Any
    target_encoder: Any
    device: torch.device
    file_records: Mapping[str, Mapping[str, Any]]
    state_sha256: Mapping[str, str]
    checkpoint_metadata: Mapping[str, Any]


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence")
    return value


def resolve_path(value: object, repo_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def sha256_state_dict(module: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        array = value.numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(tuple(value.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _verify_loaded_corrector_binding(
    policy: Any,
    corrector: Any,
    *,
    checkpoint_path: Path,
    expected_file_sha256: str,
    state_sha256_before_binding: str,
) -> dict[str, str]:
    """Bind provenance to the corrector instance actually used by the policy."""

    if getattr(policy, "fcpg_adapter", None) is not corrector:
        raise RuntimeError(
            "action policy is not bound to the verified FCPG corrector instance"
        )
    state_sha256_after = sha256_state_dict(corrector)
    if state_sha256_after != state_sha256_before_binding:
        raise RuntimeError("FCPG corrector state changed while binding the action policy")
    file_sha256_after = sha256_file(checkpoint_path)
    if file_sha256_after != expected_file_sha256:
        raise RuntimeError(
            "FCPG corrector checkpoint changed while loading: "
            f"expected {expected_file_sha256}, got {file_sha256_after}"
        )
    return {
        "loaded_path": str(checkpoint_path),
        "loaded_file_sha256": file_sha256_after,
        "loaded_state_sha256": state_sha256_after,
    }


def snapshot_sha256(snapshot: Any) -> str:
    return hashlib.sha256(pickle.dumps(snapshot, protocol=5)).hexdigest()


def physical_snapshot_sha256(snapshot: Any) -> str:
    """Hash paired physical/task state while excluding camera and RNG state.

    RNG state remains in the full branch snapshot and is restored for every
    within-condition candidate replay. This hash only checks that clean and
    perturbed-camera plans share the same physical branch point.
    """

    values = (
        snapshot.integration_state,
        snapshot.warmstart_state,
        snapshot.model_arrays.body_pos,
        snapshot.model_arrays.site_pos,
        snapshot.task_fingerprint,
        snapshot.runtime_attributes,
    )
    return hashlib.sha256(pickle.dumps(values, protocol=5)).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("result contains a non-finite float")
    return value


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_pilot_config(config: Mapping[str, Any]) -> None:
    """Validate the immutable full-pilot contract before smoke overrides."""

    if config.get("schema_version") != 1 or config.get("protocol") != PROTOCOL:
        raise ValueError("unexpected counterfactual pilot schema or protocol")
    scope = _object(config.get("scope"), "scope")
    if scope.get("classification") != SCOPE:
        raise ValueError("pilot must be labeled exploratory and conditional")
    if scope.get("retroactive_formal_gate_l2") is not False:
        raise ValueError("pilot cannot be labeled as retroactive formal Gate L2")
    if scope.get("training_allowed") is not False:
        raise ValueError("counterfactual pilot must forbid training")
    if int(config.get("seed", -1)) != 4096:
        raise ValueError("full pilot seed must be 4096")
    if config.get("device") != "cuda:0":
        raise ValueError("full pilot device must be cuda:0")

    inputs = _object(config.get("inputs"), "inputs")
    checkpoints = _object(inputs.get("lewm_checkpoints"), "LeWM checkpoints")
    if tuple(checkpoints) != TRAIN_GROUNDINGS:
        raise ValueError("LeWM checkpoints must be ordered nominal then fcpg")
    for label, raw in (
        *checkpoints.items(),
        ("action_only_policy", inputs.get("action_only_policy")),
        ("fcpg_corrector", inputs.get("fcpg_corrector")),
        ("oracle_target_encoder", inputs.get("oracle_target_encoder")),
    ):
        entry = _object(raw, f"input {label}")
        digest = entry.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"input {label} must pin a lowercase SHA-256")
        if not isinstance(entry.get("path"), str) or not entry["path"]:
            raise ValueError(f"input {label} path is missing")

    evaluation = _object(config.get("evaluation"), "evaluation")
    exact = {
        "episodes_per_task": 2,
        "snapshots_per_episode": 1,
        "history_size": 3,
        "frame_skip": 5,
        "raw_action_steps": PILOT_RAW_ACTION_STEPS,
        "action_dim": PILOT_ACTION_DIM,
        "transitions": PILOT_TRANSITIONS,
        "transition_action_dim": PILOT_TRANSITION_ACTION_DIM,
        "random_count": 1,
        "top_k": 3,
    }
    for name, expected in exact.items():
        if evaluation.get(name) != expected:
            raise ValueError(f"evaluation {name} must be exactly {expected}")
    if tuple(evaluation.get("tasks", ())) != EXPECTED_TASKS:
        raise ValueError(f"full pilot tasks must be exactly {EXPECTED_TASKS!r}")
    if tuple(evaluation.get("horizons", ())) != PILOT_HORIZONS:
        raise ValueError(f"pilot horizons must be exactly {PILOT_HORIZONS!r}")
    if tuple(evaluation.get("noise_scales", ())) != PILOT_NOISE_SCALES:
        raise ValueError(f"pilot noise scales must be exactly {PILOT_NOISE_SCALES!r}")
    if tuple(evaluation.get("candidate_names", ())) != PILOT_CANDIDATE_NAMES:
        raise ValueError("full pilot candidate set or order changed")
    if int(evaluation.get("bootstrap_samples", 0)) <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if evaluation.get("progress_info_key") != "in_place_reward":
        raise ValueError("task progress must use MetaWorld in_place_reward")
    if evaluation.get("snapshot_include_process_rng") is not True:
        raise ValueError("formal snapshots must include process RNG state")
    if int(evaluation.get("renderer_replay_rgb_atol_lsb", -1)) != 1:
        raise ValueError("EGL replay tolerance must be exactly one RGB LSB")
    if not np.isclose(
        float(evaluation.get("renderer_replay_max_changed_fraction", -1.0)),
        0.001,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("EGL replay changed-value fraction must be exactly 0.001")

    conditions = _object(config.get("conditions"), "conditions")
    if tuple(conditions) != EXPECTED_CONDITIONS:
        raise ValueError("conditions must be ordered clean then rotation5")
    expected_conditions = {
        "clean": {
            "translation_world": [0.0, 0.0, 0.0],
            "rotation_axis_world": [0.0, 1.0, 0.0],
            "rotation_degrees": 0.0,
        },
        "rotation5": {
            "translation_world": [0.0, 0.0, 0.0],
            "rotation_axis_world": [0.0, 1.0, 0.0],
            "rotation_degrees": 5.0,
        },
    }
    for name, expected in expected_conditions.items():
        if _jsonable(_object(conditions[name], name)) != expected:
            raise ValueError(f"condition {name} changed from the frozen protocol")

    environment = _object(config.get("environment"), "environment")
    expected_environment = {
        "camera": "corner",
        "width": 224,
        "height": 224,
        "max_expert_steps": 120,
        "benchmark_seed": 4096,
        "env_seed_base": 20000,
        "task_index_offset": 0,
    }
    for name, expected in expected_environment.items():
        if environment.get(name) != expected:
            raise ValueError(f"environment {name} must be exactly {expected!r}")
    output = _object(config.get("output"), "output")
    result = Path(str(output.get("result", "")))
    if result.is_absolute() or not result.parts or result.parts[0] != "outputs":
        raise ValueError("pilot result must be repository-relative under outputs/")
    if output.get("write_mode") != "exclusive":
        raise ValueError("pilot output must use exclusive write mode")


def smoke_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the only allowed branch-count reduction, gated by CLI --smoke."""

    validate_pilot_config(config)
    result = copy.deepcopy(dict(config))
    smoke = _object(result.get("smoke"), "smoke")
    evaluation = result["evaluation"]
    evaluation["tasks"] = list(_sequence(smoke.get("tasks"), "smoke tasks"))
    evaluation["episodes_per_task"] = int(smoke.get("episodes_per_task", 1))
    evaluation["snapshots_per_episode"] = int(smoke.get("snapshots_per_episode", 1))
    evaluation["candidate_names"] = list(
        _sequence(smoke.get("candidate_names"), "smoke candidate names")
    )
    evaluation["bootstrap_samples"] = int(smoke.get("bootstrap_samples", 50))
    result["conditions"] = {
        name: result["conditions"][name]
        for name in _sequence(smoke.get("conditions"), "smoke conditions")
    }
    result["inputs"]["lewm_checkpoints"] = {
        name: result["inputs"]["lewm_checkpoints"][name]
        for name in _sequence(smoke.get("checkpoints"), "smoke checkpoints")
    }
    destination = Path(str(result["output"]["result"]))
    result["output"]["result"] = str(
        destination.with_name(f"{destination.stem}_smoke{destination.suffix}")
    )
    result["runtime_mode"] = "smoke"
    result["summary_interpretation"] = "pipeline_only"
    if tuple(evaluation["tasks"]) != ("reach-v3",):
        raise ValueError("smoke tasks changed from the frozen protocol")
    if tuple(result["conditions"]) != EXPECTED_CONDITIONS:
        raise ValueError("smoke must cover clean and rotation5")
    if tuple(result["inputs"]["lewm_checkpoints"]) != TRAIN_GROUNDINGS:
        raise ValueError("smoke must cover nominal and fcpg checkpoints")
    if evaluation["episodes_per_task"] != 2:
        raise ValueError("smoke must use exactly two episodes")
    if evaluation["snapshots_per_episode"] != 1:
        raise ValueError("smoke must use exactly one snapshot per episode")
    if tuple(evaluation["candidate_names"]) != PILOT_CANDIDATE_NAMES:
        raise ValueError("smoke must retain the complete candidate set")
    if evaluation["bootstrap_samples"] != 50:
        raise ValueError("smoke bootstrap_samples must be exactly 50")
    return result


def assert_deployment_batch(batch: Mapping[str, Any], label: str) -> None:
    forbidden = sorted(FORBIDDEN_DEPLOYMENT_KEYS.intersection(batch))
    if forbidden:
        raise RuntimeError(f"{label} received forbidden privileged keys: {forbidden}")


def _record_value(record: FrameRecord | Mapping[str, Any], name: str) -> np.ndarray:
    value = getattr(record, name) if isinstance(record, FrameRecord) else record[name]
    return np.asarray(value)


def pixel_points_inside(
    points: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Return per-point validity for finite pixel coordinates inside an image."""

    value = np.asarray(points)
    if value.ndim < 1 or value.shape[-1] != 2:
        raise ValueError("pixel points must have a final coordinate dimension of two")
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    return (
        np.isfinite(value).all(axis=-1)
        & (value[..., 0] >= 0.0)
        & (value[..., 0] <= float(width - 1))
        & (value[..., 1] >= 0.0)
        & (value[..., 1] <= float(height - 1))
    )


def grounded_valid_mask(
    valid_nominal: np.ndarray,
    grounded_points: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Match the LeWM training mask for nominal or corrected grounding points."""

    points = np.asarray(grounded_points)
    nominal = np.asarray(valid_nominal, dtype=bool)
    if nominal.shape != points.shape[:-1]:
        raise ValueError("valid_nominal must match grounded points without coordinates")
    return nominal & pixel_points_inside(points, width=width, height=height)


def build_lewm_history_batch(
    records: Sequence[FrameRecord | Mapping[str, Any]],
    *,
    device: torch.device,
    q_corrected: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    if len(records) != 3:
        raise ValueError("LeWM history requires exactly H=3 records")
    pixels = np.stack([_record_value(record, "rgb") for record in records])
    if pixels.ndim != 4 or pixels.shape[1] != 3:
        raise ValueError("LeWM history RGB must have shape [H, 3, height, width]")
    valid_nominal = np.stack(
        [_record_value(record, "valid_nominal") for record in records]
    ).astype(bool, copy=False)
    values = {
        "pixels": pixels,
        "proprio": np.stack([_record_value(record, "proprio") for record in records]),
        "q_nominal": np.stack(
            [_record_value(record, "q_nominal") for record in records]
        ),
        "keypoints_xyz": np.stack(
            [_record_value(record, "keypoints_xyz") for record in records]
        ),
    }
    grounded = values["q_nominal"]
    if q_corrected is not None:
        corrected = np.asarray(q_corrected, dtype=np.float32)
        if corrected.shape != values["q_nominal"].shape:
            raise ValueError("q_corrected must match the three-frame q_nominal shape")
        values["q_corrected"] = corrected
        grounded = corrected
    values["valid"] = grounded_valid_mask(
        valid_nominal,
        grounded,
        width=int(pixels.shape[-1]),
        height=int(pixels.shape[-2]),
    )
    batch = {
        name: torch.from_numpy(np.asarray(value)).unsqueeze(0).to(device)
        for name, value in values.items()
    }
    assert_deployment_batch(batch, "LeWM")
    return batch


def build_policy_history_batch(
    records: Sequence[FrameRecord | Mapping[str, Any]],
    *,
    device: torch.device,
    q_corrected: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    if not records:
        raise ValueError("policy history must not be empty")
    values = {
        "rgb": np.stack([_record_value(record, "rgb") for record in records]),
        "obs": np.stack([_record_value(record, "proprio") for record in records]),
        "q_nominal": np.stack(
            [_record_value(record, "q_nominal") for record in records]
        ),
        "keypoints_xyz": np.stack(
            [_record_value(record, "keypoints_xyz") for record in records]
        ),
    }
    if q_corrected is not None:
        corrected = np.asarray(q_corrected, dtype=np.float32)
        if corrected.shape != values["q_nominal"].shape:
            raise ValueError("policy q_corrected shape mismatch")
        values["fcpg_q_corrected"] = corrected
    batch = {
        name: torch.from_numpy(np.asarray(value)).unsqueeze(0).to(device)
        for name, value in values.items()
    }
    assert_deployment_batch(batch, "action-only policy")
    return batch


def capture_branch_snapshot(env: Any, *, include_process_rng: bool) -> Any:
    from fcpg.metaworld_snapshot import capture_metaworld_snapshot

    return capture_metaworld_snapshot(
        env,
        include_process_rng=include_process_rng,
    )


def restore_branch_snapshot(env: Any, snapshot: Any) -> None:
    from fcpg.metaworld_snapshot import restore_metaworld_snapshot

    restore_metaworld_snapshot(env, snapshot, restore_process_rng=True)


def validate_candidate_replay(
    candidate_names: Sequence[str],
    run_candidate: Callable[[str], Any],
    *,
    repeat_name: str,
    outcome_sha256: Callable[[Any], str] = canonical_sha256,
    outcome_comparator: Callable[[Any, Any], Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run original, repeated, and reversed orders and require exact outcomes."""

    names = tuple(candidate_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("candidate names must be non-empty and unique")
    if repeat_name not in names:
        raise ValueError("repeat candidate is absent")
    original = {name: run_candidate(name) for name in names}
    original_hash = {name: outcome_sha256(original[name]) for name in names}
    repeated = run_candidate(repeat_name)
    repeated_hash = outcome_sha256(repeated)
    if outcome_comparator is None:
        repeated_comparison = {
            "equal": repeated_hash == original_hash[repeat_name],
            "mode": "exact_hash",
        }
    else:
        repeated_comparison = dict(
            outcome_comparator(original[repeat_name], repeated)
        )
    if repeated_comparison.get("equal") is not True:
        raise RuntimeError(
            f"non-repeatable snapshot outcome for {repeat_name}: "
            f"{repeated_comparison}"
        )
    reversed_hash: dict[str, str] = {}
    reversed_comparison: dict[str, dict[str, Any]] = {}
    for name in reversed(names):
        outcome = run_candidate(name)
        digest = outcome_sha256(outcome)
        reversed_hash[name] = digest
        if outcome_comparator is None:
            comparison = {
                "equal": digest == original_hash[name],
                "mode": "exact_hash",
            }
        else:
            comparison = dict(outcome_comparator(original[name], outcome))
        reversed_comparison[name] = comparison
        if comparison.get("equal") is not True:
            raise RuntimeError(
                f"candidate order changed actual outcome for {name}: {comparison}"
            )
    return original, {
        "validated": True,
        "repeat_candidate": repeat_name,
        "original_outcome_sha256": original_hash,
        "repeated_outcome_sha256": repeated_hash,
        "reversed_outcome_sha256": reversed_hash,
        "repeat_comparison": repeated_comparison,
        "reversed_comparison": reversed_comparison,
    }


def _input_file_record(
    label: str,
    entry: Mapping[str, Any],
    repo_root: Path,
) -> tuple[Path, dict[str, Any]]:
    path = resolve_path(entry["path"], repo_root)
    if not path.is_file():
        raise FileNotFoundError(path)
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"input changed while hashing: {path}")
    if digest != entry["sha256"]:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {entry['sha256']}, got {digest}"
        )
    return path, {
        "path": str(path),
        "sha256": digest,
        "size_bytes": int(after.st_size),
    }


def _torch_payload(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint payload must be an object: {path}")
    return payload


def _matched_checkpoint_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    value["output_dir"] = "<paired-output>"
    value["provenance_manifest"] = "<paired-manifest>"
    value["data"]["grounding_mode"] = "<paired-grounding>"
    value["contract"]["training_grounding"] = "<paired-grounding>"
    return value


def _validate_lewm_payload(
    label: str,
    payload: Mapping[str, Any],
    target_sha256: str,
) -> dict[str, Any]:
    from fcpg.lewm_core import LEWM_UPSTREAM_COMMIT

    if payload.get("checkpoint_type") != "fcpg_grounded_lewm_v1":
        raise ValueError(f"{label} is not a grounded LeWM checkpoint")
    if payload.get("run_kind") != "formal" or int(payload.get("step", -1)) != 10_000:
        raise ValueError(f"{label} is not the frozen formal 10,000-step endpoint")
    commit = payload.get("upstream_commit", payload.get("lewm_upstream_commit"))
    if commit != LEWM_UPSTREAM_COMMIT:
        raise ValueError(f"{label} pins a mismatched LeWM upstream commit")
    resolved = _object(payload.get("resolved_config"), f"{label} resolved config")
    data = _object(resolved.get("data"), f"{label} data config")
    contract = _object(resolved.get("contract"), f"{label} contract")
    loss = _object(resolved.get("loss"), f"{label} loss")
    if data.get("target_mode") != "frozen_oracle":
        raise ValueError(f"{label} did not train against frozen oracle targets")
    if contract.get("target_encoder_sha256") != target_sha256:
        raise ValueError(f"{label} target encoder differs from the pilot input")
    if contract.get("target_q_true_usage") != "supervision_and_offline_evaluation_only":
        raise ValueError(f"{label} has an unexpected q_true contract")
    if float(loss.get("sigreg_weight", -1.0)) != 0.02:
        raise ValueError(f"{label} is not the frozen SIGReg=0.02 checkpoint")
    data_config = _object(payload.get("data_config"), f"{label} data metadata")
    if (
        int(data_config.get("history_size", -1)),
        int(data_config.get("frame_skip", -1)),
        int(data_config.get("action_dim", -1)),
    ) != (3, 5, 20):
        raise ValueError(f"{label} sequence dimensions do not match H=3/frame_skip=5")
    target_contract = _object(
        data_config.get("target_contract"),
        f"{label} target metadata",
    )
    if target_contract.get("encoder_checkpoint_sha256") != target_sha256:
        raise ValueError(f"{label} data target encoder metadata mismatches")
    world = _object(
        _object(payload.get("model_config"), f"{label} model config").get("world_model"),
        f"{label} world model config",
    )
    frame = _object(payload["model_config"].get("frame"), f"{label} frame config")
    if (
        int(world.get("history_size", -1)),
        int(world.get("action_dim", -1)),
        int(frame.get("obs_dim", -1)),
        int(frame.get("num_keypoints", -1)),
        int(frame.get("latent_dim", -1)),
    ) != (3, 20, 4, 3, 192):
        raise ValueError(f"{label} model dimensions mismatch the pilot")
    provenance = _object(payload.get("provenance"), f"{label} provenance")
    if not isinstance(provenance.get("sha256"), str):
        raise ValueError(f"{label} checkpoint provenance SHA-256 is missing")
    return {
        "step": 10_000,
        "epoch": int(payload.get("epoch", -1)),
        "training_grounding": data.get("grounding_mode"),
        "target_encoder_sha256": target_sha256,
        "upstream_commit": commit,
        "provenance_manifest_sha256": provenance["sha256"],
    }


def _validate_corrector_payload(payload: Mapping[str, Any]) -> None:
    from fcpg.lewm_core import LEWM_UPSTREAM_COMMIT

    args = _object(payload.get("args"), "corrector args")
    if (
        int(payload.get("schema_version", 0)) < 2
        or payload.get("corrector_arch") != "lewm"
        or bool(payload.get("has_reliability", True))
        or payload.get("lewm_upstream_commit") != LEWM_UPSTREAM_COMMIT
    ):
        raise ValueError("pilot corrector must be schema-v2 residual-only LeWM")
    if (
        int(payload.get("obs_dim", -1)),
        int(payload.get("num_keypoints", -1)),
        args.get("obs_mode"),
    ) != (4, 3, "robot"):
        raise ValueError("corrector observation dimensions mismatch the pilot")
    if bool(args.get("unfreeze_backbone", False)):
        raise ValueError("pilot corrector must use a frozen visual backbone")


def _validate_policy_payload(
    payload: Mapping[str, Any],
    corrector_path: Path,
    repo_root: Path,
) -> None:
    args = _object(payload.get("args"), "policy args")
    loss = _object(payload.get("loss_config"), "policy loss config")
    if (
        payload.get("checkpoint_type") != "fcpg_action_policy"
        or int(payload.get("schema_version", 0)) < 2
        or args.get("mode") != "fcpg"
        or args.get("obs_mode") != "robot"
    ):
        raise ValueError("pilot policy must be the shared schema-v2 FCPG policy")
    if float(args.get("latent_pred_weight", -1.0)) != 0.0:
        raise ValueError("pilot policy must be action-only (latent_pred_weight=0)")
    if float(loss.get("latent_pred_weight", -1.0)) != 0.0:
        raise ValueError("policy checkpoint loss metadata is not action-only")
    if (
        int(payload.get("obs_dim", -1)),
        int(payload.get("action_dim", -1)),
        int(payload.get("num_keypoints", -1)),
    ) != (4, 4, 3):
        raise ValueError("policy dimensions mismatch MetaWorld pilot dimensions")
    if int(args.get("action_horizon", 0)) < PILOT_RAW_ACTION_STEPS:
        raise ValueError("policy action horizon is shorter than 25 raw actions")
    if not bool(payload.get("balanced_sources")):
        raise ValueError("policy is not a balanced shared-task checkpoint")
    data_sources = _sequence(payload.get("data_sources"), "policy data_sources")
    if len(data_sources) != len(EXPECTED_TASKS):
        raise ValueError("policy is not shared across all four pilot tasks")
    embedded_corrector = resolve_path(args.get("fcpg_checkpoint"), repo_root)
    if embedded_corrector != corrector_path:
        raise ValueError("policy and pilot reference different FCPG correctors")


def load_frozen_inputs(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> FrozenInputs:
    """Load and freeze every model used by the no-training pilot."""

    inputs = _object(config.get("inputs"), "inputs")
    target_path, target_record = _input_file_record(
        "oracle_target_encoder",
        _object(inputs.get("oracle_target_encoder"), "oracle target encoder"),
        repo_root,
    )
    target_sha256 = target_record["sha256"]
    corrector_path, corrector_record = _input_file_record(
        "fcpg_corrector",
        _object(inputs.get("fcpg_corrector"), "FCPG corrector"),
        repo_root,
    )
    policy_path, policy_record = _input_file_record(
        "action_only_policy",
        _object(inputs.get("action_only_policy"), "action-only policy"),
        repo_root,
    )
    corrector_payload = _torch_payload(corrector_path)
    policy_payload = _torch_payload(policy_path)
    _validate_corrector_payload(corrector_payload)
    _validate_policy_payload(policy_payload, corrector_path, repo_root)

    checkpoint_paths: dict[str, Path] = {}
    file_records: dict[str, Mapping[str, Any]] = {
        "oracle_target_encoder": target_record,
        "fcpg_corrector": corrector_record,
        "action_only_policy": policy_record,
    }
    payloads: dict[str, Mapping[str, Any]] = {}
    checkpoint_metadata: dict[str, Any] = {}
    for label, raw in _object(
        inputs.get("lewm_checkpoints"),
        "LeWM checkpoints",
    ).items():
        path, record = _input_file_record(
            f"lewm_{label}",
            _object(raw, f"LeWM checkpoint {label}"),
            repo_root,
        )
        checkpoint_paths[label] = path
        file_records[f"lewm_{label}"] = record
        payload = _torch_payload(path)
        payloads[label] = payload
        metadata = _validate_lewm_payload(label, payload, target_sha256)
        if metadata["training_grounding"] != label:
            raise ValueError(f"{label} checkpoint has the wrong training grounding")
        checkpoint_metadata[label] = metadata

    if set(payloads) == set(TRAIN_GROUNDINGS):
        nominal = payloads["nominal"]
        fcpg = payloads["fcpg"]
        if _matched_checkpoint_config(nominal["resolved_config"]) != (
            _matched_checkpoint_config(fcpg["resolved_config"])
        ):
            raise ValueError("nominal/FCPG checkpoints violate the matched contract")
        if nominal["data_config"]["normalization"] != fcpg["data_config"]["normalization"]:
            raise ValueError("nominal/FCPG checkpoints use different normalization")
        if nominal["model_config"] != fcpg["model_config"]:
            raise ValueError("nominal/FCPG checkpoints use different architectures")

    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    from fcpg.action_chunk_policy import load_action_chunk_policy
    from fcpg.lewm_oracle_target import load_frozen_oracle_target_encoder
    from fcpg.lewm_runtime import load_lewm_grounded_checkpoint
    from fcpg.transformer_grounding import load_fcpg_vit_adapter

    models: dict[str, Any] = {}
    for label, path in checkpoint_paths.items():
        model, _, _ = load_lewm_grounded_checkpoint(path, device="cpu")
        model.to(device).requires_grad_(False).eval()
        models[label] = model
    corrector, _ = load_fcpg_vit_adapter(corrector_path, device)
    corrector.requires_grad_(False).eval()
    corrector_state_sha256 = sha256_state_dict(corrector)
    policy, policy_args = load_action_chunk_policy(
        policy_path,
        device,
        fcpg_adapter_override=corrector,
    )
    policy.requires_grad_(False).eval()
    corrector_binding = _verify_loaded_corrector_binding(
        policy,
        corrector,
        checkpoint_path=corrector_path,
        expected_file_sha256=corrector_record["sha256"],
        state_sha256_before_binding=corrector_state_sha256,
    )
    file_records["fcpg_corrector"] = {**corrector_record, **corrector_binding}
    target_encoder, _ = load_frozen_oracle_target_encoder(
        target_path,
        map_location="cpu",
    )
    target_encoder.to(device).requires_grad_(False).eval()
    if (target_encoder.num_keypoints, target_encoder.target_dim) != (3, 192):
        raise ValueError("oracle target encoder dimensions mismatch the LeWM pair")
    oracle_vision = _object(config.get("oracle_vision"), "oracle vision")
    vision = corrector.vision
    if (
        vision.model_name != oracle_vision.get("model_name")
        or tuple(vision.selected_layers)
        != (int(oracle_vision.get("selected_layer", -1)),)
        or int(vision.backbone.num_features) != int(target_encoder.visual_dim)
    ):
        raise ValueError("corrector backbone and oracle target vision contract mismatch")

    state_hashes = {
        **{f"lewm_{name}": sha256_state_dict(model) for name, model in models.items()},
        "action_only_policy": sha256_state_dict(policy),
        "fcpg_corrector": sha256_state_dict(corrector),
        "oracle_target_encoder": sha256_state_dict(target_encoder),
        "oracle_vision_backbone": sha256_state_dict(vision.backbone),
    }
    return FrozenInputs(
        models=models,
        policy=policy,
        policy_args=dict(policy_args),
        corrector=corrector,
        target_encoder=target_encoder,
        device=device,
        file_records=file_records,
        state_sha256=state_hashes,
        checkpoint_metadata=checkpoint_metadata,
    )


def _versions() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "torch", "timm", "mujoco", "metaworld", "gymnasium"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def _git_state(repo_root: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    tracked_diff = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout
    untracked_raw = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout
    content = hashlib.sha256()
    content.update(tracked_diff)
    untracked_files = []
    for raw_name in filter(None, untracked_raw.split(b"\0")):
        name = raw_name.decode("utf-8", errors="surrogateescape")
        path = repo_root / name
        if not path.is_file():
            continue
        digest = sha256_file(path)
        untracked_files.append({"path": name, "sha256": digest})
        content.update(raw_name)
        content.update(b"\0")
        content.update(digest.encode("ascii"))
        content.update(b"\0")
    return {
        "head": head,
        "dirty": bool(status),
        "status": status,
        "status_sha256": hashlib.sha256(
            ("\n".join(status) + ("\n" if status else "")).encode("utf-8")
        ).hexdigest(),
        "worktree_content_sha256": content.hexdigest(),
        "untracked_files": untracked_files,
    }


def _frame_sha256(record: FrameRecord, *, include_visual: bool = True) -> str:
    values = {
        "proprio": sha256_array(record.proprio),
        "keypoints_xyz": sha256_array(record.keypoints_xyz),
        "q_nominal": sha256_array(record.q_nominal),
        "valid_nominal": sha256_array(record.valid_nominal),
        "q_true": sha256_array(record.q_true),
        "valid_true": sha256_array(record.valid_true),
    }
    if include_visual:
        values["rgb"] = sha256_array(record.rgb)
    return canonical_sha256(values)


def _collect_frame_record(
    env: Any,
    observation: np.ndarray,
    *,
    camera_id: int,
    intrinsic: np.ndarray,
    nominal_extrinsic: np.ndarray,
    width: int,
    height: int,
) -> FrameRecord:
    from fcpg.geometry import Camera, project_points
    from fcpg.metaworld_utils import DEFAULT_KEYPOINT_SITES, get_site_keypoints
    from fcpg.mujoco_camera import camera_extrinsic_cv_from_mujoco
    from fcpg.observations import select_policy_observation

    image = np.asarray(env.render(), dtype=np.uint8)
    if image.shape != (height, width, 3):
        raise ValueError(f"rendered frame shape mismatch: {image.shape}")
    keypoints = get_site_keypoints(
        env.model,
        env.data,
        DEFAULT_KEYPOINT_SITES,
    ).xyz
    if keypoints.shape != (3, 3):
        raise ValueError(f"expected exactly three control keypoints, got {keypoints.shape}")
    true_extrinsic = camera_extrinsic_cv_from_mujoco(env.data, camera_id)
    true_camera = Camera(intrinsic, true_extrinsic, width, height)
    nominal_camera = Camera(intrinsic, nominal_extrinsic, width, height)
    q_true, valid_true = project_points(keypoints, true_camera)
    q_nominal, valid_nominal = project_points(keypoints, nominal_camera)
    record = FrameRecord(
        rgb=np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1)),
        proprio=select_policy_observation(observation, "robot").astype(np.float32),
        keypoints_xyz=np.asarray(keypoints, dtype=np.float32),
        q_nominal=np.asarray(q_nominal, dtype=np.float32),
        valid_nominal=np.asarray(valid_nominal, dtype=bool),
        q_true=np.asarray(q_true, dtype=np.float32),
        valid_true=np.asarray(valid_true, dtype=bool),
    )
    arrays = (
        record.rgb,
        record.proprio,
        record.keypoints_xyz,
        record.q_nominal,
        record.q_true,
    )
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("frame record contains non-finite deployment or oracle values")
    return record


def _history_indices(step: int, history_size: int, frame_skip: int) -> tuple[int, ...]:
    first = step - (history_size - 1) * frame_skip
    if first < 0:
        raise ValueError("snapshot does not have enough history")
    return tuple(first + index * frame_skip for index in range(history_size))


def _history_is_valid(
    records: Sequence[FrameRecord],
    indices: Sequence[int],
) -> bool:
    return all(
        bool(records[index].valid_nominal.all())
        for index in indices
    )


def collect_expert_plan(
    task: str,
    condition: str,
    condition_config: Mapping[str, Any],
    *,
    episode: int,
    config: Mapping[str, Any],
) -> EpisodePlan:
    """Collect one successful scripted trajectory and one exact branch point."""

    from fcpg.metaworld_utils import make_env, make_scripted_policy
    from fcpg.mujoco_camera import (
        camera_extrinsic_cv_from_mujoco,
        camera_intrinsic_from_fovy,
        get_camera_id,
        set_fixed_camera_pose_offset,
    )

    environment = _object(config.get("environment"), "environment")
    evaluation = _object(config.get("evaluation"), "evaluation")
    task_index = EXPECTED_TASKS.index(task)
    env_seed = (
        int(environment.get("env_seed_base", 20_000))
        + task_index * 100
        + int(episode)
    )
    random.seed(env_seed)
    np.random.seed(env_seed)
    torch.manual_seed(env_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(env_seed)
    width = int(environment["width"])
    height = int(environment["height"])
    camera_name = str(environment["camera"])
    env = make_env(
        task,
        camera_name,
        width,
        height,
        task_index=int(environment.get("task_index_offset", 0)) + episode,
        benchmark_seed=int(environment.get("benchmark_seed", config["seed"])),
    )
    observation, _ = env.reset(seed=env_seed)
    camera_id = get_camera_id(env.model, camera_name)
    intrinsic = camera_intrinsic_from_fovy(
        width,
        height,
        float(env.model.cam_fovy[camera_id]),
    )
    base_position = np.asarray(env.model.cam_pos[camera_id], dtype=np.float64).copy()
    base_quaternion = np.asarray(env.model.cam_quat[camera_id], dtype=np.float64).copy()
    nominal_extrinsic = camera_extrinsic_cv_from_mujoco(env.data, camera_id)
    translation = np.asarray(
        condition_config.get("translation_world", (0.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    axis = np.asarray(
        condition_config.get("rotation_axis_world", (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    if translation.shape != (3,) or axis.shape != (3,):
        raise ValueError("camera translation and rotation axis must have shape [3]")
    set_fixed_camera_pose_offset(
        env.model,
        env.data,
        camera_id,
        base_position,
        base_quaternion,
        translation,
        axis,
        float(condition_config["rotation_degrees"]),
    )

    expert = make_scripted_policy(task)
    records: list[FrameRecord] = []
    actions: list[np.ndarray] = []
    snapshot = None
    snapshot_step = -1
    history_size = int(evaluation["history_size"])
    frame_skip = int(evaluation["frame_skip"])
    minimum_step = (history_size - 1) * frame_skip
    max_steps = int(environment["max_expert_steps"])
    goal_record = None
    goal_info = None
    try:
        for step in range(max_steps):
            record = _collect_frame_record(
                env,
                observation,
                camera_id=camera_id,
                intrinsic=intrinsic,
                nominal_extrinsic=nominal_extrinsic,
                width=width,
                height=height,
            )
            records.append(record)
            if snapshot is None and step >= minimum_step:
                indices = _history_indices(step, history_size, frame_skip)
                if _history_is_valid(records, indices):
                    snapshot = capture_branch_snapshot(
                        env,
                        include_process_rng=bool(
                            evaluation["snapshot_include_process_rng"]
                        ),
                    )
                    snapshot_step = step
            action = np.asarray(expert.get_action(observation), dtype=np.float32)
            if action.shape != (PILOT_ACTION_DIM,) or not np.isfinite(action).all():
                raise ValueError("scripted expert produced an invalid 4D action")
            action = np.clip(action, env.action_space.low, env.action_space.high).astype(
                np.float32
            )
            actions.append(action)
            observation, _, terminated, truncated, info = env.step(action)
            if snapshot is not None and len(actions) >= snapshot_step + PILOT_RAW_ACTION_STEPS:
                if not isinstance(info, Mapping):
                    raise TypeError("MetaWorld step info must be a mapping")
                if float(info.get("success", 0.0)) > 0.0:
                    goal_record = _collect_frame_record(
                        env,
                        observation,
                        camera_id=camera_id,
                        intrinsic=intrinsic,
                        nominal_extrinsic=nominal_extrinsic,
                        width=width,
                        height=height,
                    )
                    goal_info = dict(_jsonable(info))
                    break
            if terminated or truncated:
                raise RuntimeError(
                    "expert episode ended before a successful goal and full candidate chunk"
                )
        if snapshot is None:
            raise RuntimeError("no valid H=3/frame_skip=5 snapshot was found")
        if goal_record is None or goal_info is None:
            raise RuntimeError("scripted expert did not produce a successful final goal")
        action_array = np.asarray(actions, dtype=np.float32)
        if action_array.shape[0] < snapshot_step + PILOT_RAW_ACTION_STEPS:
            raise RuntimeError("expert trajectory is too short for the 25-step chunk")
        return EpisodePlan(
            env=env,
            task=task,
            condition=condition,
            episode=episode,
            env_seed=env_seed,
            nominal_extrinsic=nominal_extrinsic,
            intrinsic=intrinsic,
            snapshot=snapshot,
            snapshot_step=snapshot_step,
            snapshot_sha256=snapshot_sha256(snapshot),
            physical_snapshot_sha256=physical_snapshot_sha256(snapshot),
            records=records,
            actions=action_array,
            goal_record=goal_record,
            goal_info=goal_info,
        )
    except Exception:
        if hasattr(env, "close"):
            env.close()
        raise


@torch.inference_mode()
def oracle_targets(
    records: Sequence[FrameRecord],
    frozen: FrozenInputs,
    *,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Construct targets; this is the only model path allowed to receive q_true."""

    if not records:
        raise ValueError("oracle target construction requires records")
    rgb = torch.from_numpy(np.stack([record.rgb for record in records])).to(
        frozen.device
    )
    q_true = torch.from_numpy(np.stack([record.q_true for record in records])).to(
        frozen.device
    )
    valid_true = torch.from_numpy(
        np.stack([record.valid_true for record in records])
    ).to(frozen.device)
    feature_maps, prefix = frozen.corrector.vision.extract_backbone_features(rgb)
    if feature_maps.ndim != 5 or feature_maps.shape[1] != 1:
        raise ValueError("oracle vision must expose exactly one ViT layer")
    target = frozen.target_encoder.forward_from_features(
        feature_maps[:, 0],
        prefix,
        q_true,
        valid_true,
        image_size=image_size,
    )
    if target.shape != (len(records), 192) or not bool(torch.isfinite(target).all()):
        raise ValueError("oracle target shape or finiteness mismatch")
    return target.float().cpu().numpy()


@torch.inference_mode()
def corrected_points(
    records: Sequence[FrameRecord],
    frozen: FrozenInputs,
    *,
    centers_px: np.ndarray | None = None,
) -> np.ndarray:
    if not records:
        raise ValueError("FCPG correction requires at least one frame")
    rgb = torch.from_numpy(np.stack([record.rgb for record in records])).to(
        frozen.device
    )
    nominal = np.stack([record.q_nominal for record in records]).astype(
        np.float32, copy=False
    )
    centers = nominal if centers_px is None else np.asarray(centers_px, dtype=np.float32)
    if centers.shape != nominal.shape:
        raise ValueError(
            f"FCPG centers must have shape {nominal.shape}, got {centers.shape}"
        )
    if not np.isfinite(centers).all():
        raise ValueError("FCPG centers contain non-finite values")
    q_nominal = torch.from_numpy(np.ascontiguousarray(centers)).to(frozen.device)
    xyz = torch.from_numpy(
        np.stack([record.keypoints_xyz for record in records])
    ).to(frozen.device)
    proprio = torch.from_numpy(
        np.stack([record.proprio for record in records])
    ).to(frozen.device)
    output = frozen.corrector(rgb, q_nominal, xyz, proprio)
    result = output.q_corrected_px.float().cpu().numpy()
    if result.shape != (len(records), 3, 2) or not np.isfinite(result).all():
        raise ValueError("FCPG corrector output shape or finiteness mismatch")
    return result


@torch.inference_mode()
def policy_action_chunk(plan: EpisodePlan, frozen: FrozenInputs) -> np.ndarray:
    horizon = int(frozen.policy_args.get("obs_horizon", 0))
    start = plan.snapshot_step - horizon + 1
    if horizon <= 0 or start < 0:
        raise ValueError("policy observation horizon is incompatible with snapshot")
    records = plan.records[start : plan.snapshot_step + 1]
    corrected = corrected_points(records, frozen)
    batch = build_policy_history_batch(
        records,
        device=frozen.device,
        q_corrected=corrected,
    )
    output = frozen.policy.predict_action_chunk(batch)
    actions = output.action[0].float().cpu().numpy()
    if actions.ndim != 2 or actions.shape[0] < PILOT_RAW_ACTION_STEPS:
        raise ValueError("policy did not produce at least 25 raw actions")
    result = actions[:PILOT_RAW_ACTION_STEPS].astype(np.float32, copy=False)
    if result.shape != (PILOT_RAW_ACTION_STEPS, PILOT_ACTION_DIM):
        raise ValueError("policy candidate shape mismatch")
    return result


@dataclass(frozen=True)
class BranchOutcome:
    """Observed endpoint data for one action candidate."""

    endpoint_records: tuple[FrameRecord, ...]
    endpoint_snapshot_sha256: tuple[str, ...]
    cumulative_return: np.ndarray
    progress: np.ndarray
    success: np.ndarray
    action_sha256: str


def branch_outcome_sha256(outcome: BranchOutcome) -> str:
    """Hash a branch without serializing environment or model objects."""

    return canonical_sha256(
        {
            "endpoint_frames": [
                _frame_sha256(record) for record in outcome.endpoint_records
            ],
            "cumulative_return": sha256_array(outcome.cumulative_return),
            "progress": sha256_array(outcome.progress),
            "success": sha256_array(outcome.success),
            "endpoint_snapshot_sha256": list(outcome.endpoint_snapshot_sha256),
            "action_sha256": outcome.action_sha256,
        }
    )


def compare_branch_outcomes(
    reference: BranchOutcome,
    candidate: BranchOutcome,
    *,
    rgb_atol_lsb: int,
    max_rgb_changed_fraction: float,
) -> dict[str, Any]:
    """Audit exact physics while tolerating sparse one-LSB EGL raster noise."""

    if rgb_atol_lsb < 0:
        raise ValueError("rgb_atol_lsb must be non-negative")
    if not 0.0 <= max_rgb_changed_fraction <= 1.0:
        raise ValueError("max_rgb_changed_fraction must be in [0, 1]")
    scalar_exact = all(
        np.array_equal(getattr(reference, name), getattr(candidate, name))
        for name in ("cumulative_return", "progress", "success")
    )
    action_exact = reference.action_sha256 == candidate.action_sha256
    endpoint_count_exact = len(reference.endpoint_records) == len(
        candidate.endpoint_records
    )
    nonvisual_exact = endpoint_count_exact
    endpoint_state_exact = (
        reference.endpoint_snapshot_sha256 == candidate.endpoint_snapshot_sha256
    )
    changed = 0
    total = 0
    max_abs = 0.0
    endpoint_rgb: list[dict[str, Any]] = []
    nonvisual_names = (
        "proprio",
        "keypoints_xyz",
        "q_nominal",
        "valid_nominal",
        "q_true",
        "valid_true",
    )
    for reference_record, candidate_record in zip(
        reference.endpoint_records,
        candidate.endpoint_records,
    ):
        nonvisual_exact = nonvisual_exact and all(
            np.array_equal(
                getattr(reference_record, name),
                getattr(candidate_record, name),
            )
            for name in nonvisual_names
        )
        difference = np.abs(
            np.asarray(reference_record.rgb, dtype=np.float64)
            - np.asarray(candidate_record.rgb, dtype=np.float64)
        )
        endpoint_changed = int(np.count_nonzero(difference))
        endpoint_total = int(difference.size)
        endpoint_max = float(difference.max(initial=0.0))
        changed += endpoint_changed
        total += endpoint_total
        max_abs = max(max_abs, endpoint_max)
        endpoint_rgb.append(
            {
                "changed_values": endpoint_changed,
                "total_values": endpoint_total,
                "changed_fraction": endpoint_changed / max(endpoint_total, 1),
                "max_abs": endpoint_max,
            }
        )
    changed_fraction = changed / max(total, 1)
    # Frames are stored as float32 RGB in [0, 1]. The epsilon only absorbs
    # float32 representation error around one integer image level.
    rgb_atol = float(rgb_atol_lsb) / 255.0 + 1e-7
    each_endpoint_within_bound = all(
        value["changed_fraction"] <= max_rgb_changed_fraction
        and value["max_abs"] <= rgb_atol
        for value in endpoint_rgb
    )
    rgb_within_bound = (
        endpoint_count_exact
        and each_endpoint_within_bound
    )
    equal = (
        action_exact
        and scalar_exact
        and nonvisual_exact
        and endpoint_state_exact
        and rgb_within_bound
    )
    return {
        "equal": bool(equal),
        "mode": "exact_endpoint_snapshot_sparse_lsb_rgb",
        "action_exact": bool(action_exact),
        "scalar_metrics_exact": bool(scalar_exact),
        "nonvisual_endpoints_exact": bool(nonvisual_exact),
        "endpoint_snapshot_exact": bool(endpoint_state_exact),
        "endpoint_count_exact": bool(endpoint_count_exact),
        "rgb_exact": changed == 0,
        "rgb_atol_lsb": int(rgb_atol_lsb),
        "max_rgb_changed_fraction": float(max_rgb_changed_fraction),
        "rgb_changed_values": int(changed),
        "rgb_total_values": int(total),
        "rgb_changed_fraction": float(changed_fraction),
        "rgb_max_abs": float(max_abs),
        "endpoint_rgb": endpoint_rgb,
    }


def _history_records_and_actions(
    plan: EpisodePlan,
    *,
    history_size: int,
    frame_skip: int,
) -> tuple[list[FrameRecord], np.ndarray]:
    indices = _history_indices(plan.snapshot_step, history_size, frame_skip)
    records = [plan.records[index] for index in indices]
    blocks = []
    for start, stop in zip(indices[:-1], indices[1:]):
        action = np.asarray(plan.actions[start:stop], dtype=np.float32)
        if action.shape != (frame_skip, PILOT_ACTION_DIM):
            raise ValueError("expert history action block has the wrong shape")
        blocks.append(action.reshape(-1))
    result = np.stack(blocks).astype(np.float32, copy=False)
    if result.shape != (history_size - 1, PILOT_TRANSITION_ACTION_DIM):
        raise ValueError("LeWM history action shape mismatch")
    return records, result


def _candidate_rng_seed(
    config: Mapping[str, Any],
    *,
    task: str,
    condition: str,
    episode: int,
) -> int:
    task_index = EXPECTED_TASKS.index(task)
    EXPECTED_CONDITIONS.index(condition)
    return (
        int(config["seed"])
        + 100 * task_index
        + int(episode)
    )


def build_plan_candidates(
    plan: EpisodePlan,
    frozen: FrozenInputs,
    config: Mapping[str, Any],
    *,
    donor: EpisodePlan | None,
) -> ActionCandidates:
    """Build and select the immutable candidate family for one branch point."""

    evaluation = _object(config.get("evaluation"), "evaluation")
    start = plan.snapshot_step
    expert = np.asarray(
        plan.actions[start : start + PILOT_RAW_ACTION_STEPS],
        dtype=np.float32,
    )
    if expert.shape != (PILOT_RAW_ACTION_STEPS, PILOT_ACTION_DIM):
        raise ValueError("expert candidate is shorter than 25 raw actions")
    policy = policy_action_chunk(plan, frozen)
    if donor is None:
        shuffled = np.roll(expert, shift=int(evaluation["frame_skip"]), axis=0)
    else:
        donor_start = donor.snapshot_step
        shuffled = np.asarray(
            donor.actions[donor_start : donor_start + PILOT_RAW_ACTION_STEPS],
            dtype=np.float32,
        )
        if shuffled.shape != expert.shape:
            raise ValueError("same-task donor candidate has the wrong shape")
    rng = np.random.default_rng(
        _candidate_rng_seed(
            config,
            task=plan.task,
            condition=plan.condition,
            episode=plan.episode,
        )
    )
    full = build_action_candidates(
        expert,
        policy_action=policy,
        shuffled_action=shuffled,
        rng=rng,
        noise_scales=tuple(float(value) for value in evaluation["noise_scales"]),
        random_count=int(evaluation["random_count"]),
    )
    selected_names = tuple(str(value) for value in evaluation["candidate_names"])
    lookup = {name: index for index, name in enumerate(full.names)}
    missing = [name for name in selected_names if name not in lookup]
    if missing:
        raise ValueError(f"unknown selected candidates: {missing}")
    selected = np.stack([full.actions[lookup[name]] for name in selected_names])
    if "expert" not in selected_names:
        raise ValueError("candidate set must retain the expert action")
    return ActionCandidates(
        names=selected_names,
        actions=selected.astype(np.float32, copy=False),
        expert_index=selected_names.index("expert"),
    )


def execute_candidate_branch(
    plan: EpisodePlan,
    actions: np.ndarray,
    config: Mapping[str, Any],
) -> BranchOutcome:
    """Execute one candidate from the exact saved simulator/RNG state."""

    from fcpg.mujoco_camera import get_camera_id

    evaluation = _object(config.get("evaluation"), "evaluation")
    environment = _object(config.get("environment"), "environment")
    value = np.asarray(actions, dtype=np.float32)
    if value.shape != (PILOT_RAW_ACTION_STEPS, PILOT_ACTION_DIM):
        raise ValueError("candidate branch requires exactly 25 4D raw actions")
    restore_branch_snapshot(plan.env, plan.snapshot)
    endpoints = {
        int(horizon) * int(evaluation["frame_skip"]): index
        for index, horizon in enumerate(evaluation["horizons"])
    }
    endpoint_records: list[FrameRecord | None] = [None] * len(endpoints)
    endpoint_state_hashes: list[str | None] = [None] * len(endpoints)
    returns = np.empty(len(endpoints), dtype=np.float64)
    progress = np.empty(len(endpoints), dtype=np.float64)
    success = np.empty(len(endpoints), dtype=np.float64)
    cumulative = 0.0
    camera_id = get_camera_id(plan.env.model, str(environment["camera"]))
    progress_key = str(evaluation["progress_info_key"])
    for raw_step, action in enumerate(value, start=1):
        clipped = np.clip(
            action,
            plan.env.action_space.low,
            plan.env.action_space.high,
        ).astype(np.float32)
        observation, reward, terminated, truncated, info = plan.env.step(clipped)
        cumulative += float(reward)
        if raw_step in endpoints:
            if not isinstance(info, Mapping):
                raise TypeError("MetaWorld branch info must be a mapping")
            if progress_key not in info:
                raise KeyError(f"MetaWorld info is missing {progress_key!r}")
            index = endpoints[raw_step]
            endpoint_records[index] = _collect_frame_record(
                plan.env,
                observation,
                camera_id=camera_id,
                intrinsic=plan.intrinsic,
                nominal_extrinsic=plan.nominal_extrinsic,
                width=int(environment["width"]),
                height=int(environment["height"]),
            )
            endpoint_state_hashes[index] = snapshot_sha256(
                capture_branch_snapshot(
                    plan.env, include_process_rng=False
                )
            )
            returns[index] = cumulative
            progress[index] = float(info[progress_key])
            success[index] = float(info.get("success", 0.0))
        if (terminated or truncated) and raw_step < PILOT_RAW_ACTION_STEPS:
            raise RuntimeError("candidate branch ended before all endpoints")
    if any(record is None for record in endpoint_records):
        raise RuntimeError("candidate branch did not produce every endpoint")
    if any(value is None for value in endpoint_state_hashes):
        raise RuntimeError("candidate branch did not snapshot every endpoint")
    return BranchOutcome(
        endpoint_records=tuple(
            record for record in endpoint_records if record is not None
        ),
        endpoint_snapshot_sha256=tuple(
            value for value in endpoint_state_hashes if value is not None
        ),
        cumulative_return=returns,
        progress=progress,
        success=success,
        action_sha256=sha256_array(value),
    )


def _actual_candidate_arrays(
    candidates: ActionCandidates,
    outcomes: Mapping[str, BranchOutcome],
    frozen: FrozenInputs,
    *,
    goal_target: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_rows = []
    return_rows = []
    progress_rows = []
    success_rows = []
    for name in candidates.names:
        outcome = outcomes[name]
        target_rows.append(
            oracle_targets(outcome.endpoint_records, frozen, image_size=image_size)
        )
        return_rows.append(outcome.cumulative_return)
        progress_rows.append(outcome.progress)
        success_rows.append(outcome.success)
    targets = np.stack(target_rows)
    actual_cost = np.square(targets - goal_target[None, None]).mean(axis=-1)
    return (
        actual_cost.astype(np.float64),
        np.stack(return_rows).astype(np.float64),
        np.stack(progress_rows).astype(np.float64),
        np.stack(success_rows).astype(np.float64),
    )


@torch.inference_mode()
def predict_candidate_costs(
    model: Any,
    plan: EpisodePlan,
    candidates: ActionCandidates,
    frozen: FrozenInputs,
    config: Mapping[str, Any],
    *,
    eval_grounding: str,
    goal_target: np.ndarray,
) -> np.ndarray:
    """Predict goal distances without exposing any privileged branch outcome."""

    if eval_grounding not in EVAL_GROUNDINGS:
        raise ValueError(f"unknown eval grounding: {eval_grounding}")
    evaluation = _object(config.get("evaluation"), "evaluation")
    records, history_actions = _history_records_and_actions(
        plan,
        history_size=int(evaluation["history_size"]),
        frame_skip=int(evaluation["frame_skip"]),
    )
    q_corrected = (
        corrected_points(records, frozen) if eval_grounding == "fcpg" else None
    )
    batch = build_lewm_history_batch(
        records,
        device=frozen.device,
        q_corrected=q_corrected,
    )
    assert_deployment_batch(batch, "LeWM counterfactual prediction")
    _, embedding = model.encode_frames(batch)
    future_actions = reshape_action_chunks(candidates.actions)
    predicted = model.rollout_embeddings(
        history_embedding=embedding,
        history_action=torch.from_numpy(history_actions)
        .unsqueeze(0)
        .to(frozen.device),
        future_action=torch.from_numpy(future_actions)
        .unsqueeze(0)
        .to(frozen.device),
    )
    horizon_indices = torch.tensor(
        [int(horizon) - 1 for horizon in evaluation["horizons"]],
        dtype=torch.long,
        device=frozen.device,
    )
    selected = predicted[0].index_select(1, horizon_indices)
    goal = torch.from_numpy(np.asarray(goal_target, dtype=np.float32)).to(
        frozen.device
    )
    cost = (selected - goal[None, None]).square().mean(dim=-1)
    result = cost.float().cpu().numpy()
    expected = (len(candidates.names), len(evaluation["horizons"]))
    if result.shape != expected or not np.isfinite(result).all():
        raise ValueError("predicted candidate cost shape or finiteness mismatch")
    return result


def evaluate_episode_plan(
    plan: EpisodePlan,
    candidates: ActionCandidates,
    frozen: FrozenInputs,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate all frozen LeWM/eval-grounding cells at one branch point."""

    environment = _object(config.get("environment"), "environment")
    evaluation = _object(config.get("evaluation"), "evaluation")
    candidate_lookup = {
        name: candidates.actions[index] for index, name in enumerate(candidates.names)
    }

    def run_candidate(name: str) -> BranchOutcome:
        return execute_candidate_branch(plan, candidate_lookup[name], config)

    outcomes, replay = validate_candidate_replay(
        candidates.names,
        run_candidate,
        repeat_name="expert",
        outcome_sha256=branch_outcome_sha256,
        outcome_comparator=lambda reference, candidate: compare_branch_outcomes(
            reference,
            candidate,
            rgb_atol_lsb=int(evaluation["renderer_replay_rgb_atol_lsb"]),
            max_rgb_changed_fraction=float(
                evaluation["renderer_replay_max_changed_fraction"]
            ),
        ),
    )
    image_size = (int(environment["width"]), int(environment["height"]))
    goal_target = oracle_targets(
        [plan.goal_record],
        frozen,
        image_size=image_size,
    )[0]
    actual_cost, actual_return, actual_progress, actual_success = (
        _actual_candidate_arrays(
            candidates,
            outcomes,
            frozen,
            goal_target=goal_target,
            image_size=image_size,
        )
    )
    rows: list[dict[str, Any]] = []
    candidate_action_set_sha256 = canonical_sha256(
        {
            "names": list(candidates.names),
            "actions": sha256_array(candidates.actions),
        }
    )
    for train_grounding, model in frozen.models.items():
        for eval_grounding in EVAL_GROUNDINGS:
            predicted_cost = predict_candidate_costs(
                model,
                plan,
                candidates,
                frozen,
                config,
                eval_grounding=eval_grounding,
                goal_target=goal_target,
            )
            for horizon_index, horizon in enumerate(evaluation["horizons"]):
                metrics = counterfactual_ranking_metrics(
                    predicted_cost[:, horizon_index],
                    actual_cost[:, horizon_index],
                    expert_index=candidates.expert_index,
                    top_k=min(int(evaluation["top_k"]), len(candidates.names)),
                    actual_return=actual_return[:, horizon_index],
                    actual_progress=actual_progress[:, horizon_index],
                    actual_oracle_progress=-actual_cost[:, horizon_index],
                )
                rows.append(
                    {
                        "task": plan.task,
                        "condition": plan.condition,
                        "episode": int(plan.episode),
                        "snapshot": plan.physical_snapshot_sha256,
                        "snapshot_sha256": plan.snapshot_sha256,
                        "snapshot_step": int(plan.snapshot_step),
                        "train_grounding": str(train_grounding),
                        "eval_grounding": eval_grounding,
                        "horizon": int(horizon),
                        "candidate_names": list(candidates.names),
                        "candidate_action_set_sha256": candidate_action_set_sha256,
                        "predicted_cost": predicted_cost[:, horizon_index].tolist(),
                        "actual_cost": actual_cost[:, horizon_index].tolist(),
                        "actual_return": actual_return[:, horizon_index].tolist(),
                        "actual_progress": actual_progress[:, horizon_index].tolist(),
                        "actual_success": actual_success[:, horizon_index].tolist(),
                        **metrics,
                    }
                )
    return rows, replay


def summarize_counterfactual_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Aggregate cells, eval interventions, and matched training diagonals."""

    pair_keys = (
        "task",
        "episode",
        "snapshot",
        "horizon",
        "candidate_action_set_sha256",
    )

    grouped: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["train_grounding"]),
            str(row["eval_grounding"]),
            str(row["condition"]),
            int(row["horizon"]),
        )
        grouped.setdefault(key, []).append(row)
    cells: dict[str, Any] = {}
    for index, (key, cell_rows) in enumerate(sorted(grouped.items())):
        train, evaluation, condition, horizon = key
        label = f"train={train}|eval={evaluation}|{condition}|H={horizon}"
        metrics = {}
        for metric in SUMMARY_METRICS:
            if all(metric in row for row in cell_rows):
                metrics[metric] = task_macro_bootstrap(
                    cell_rows,
                    metric,
                    bootstrap_samples=bootstrap_samples,
                    rng=np.random.default_rng(seed + index * 100 + len(metrics)),
                )
        cells[label] = metrics

    paired: dict[str, Any] = {}
    trains = sorted({str(row["train_grounding"]) for row in rows})
    conditions = sorted({str(row["condition"]) for row in rows})
    horizons = sorted({int(row["horizon"]) for row in rows})
    pair_index = 0
    for train in trains:
        for condition in conditions:
            for horizon in horizons:
                nominal = [
                    row
                    for row in rows
                    if row["train_grounding"] == train
                    and row["eval_grounding"] == "nominal"
                    and row["condition"] == condition
                    and int(row["horizon"]) == horizon
                ]
                fcpg = [
                    row
                    for row in rows
                    if row["train_grounding"] == train
                    and row["eval_grounding"] == "fcpg"
                    and row["condition"] == condition
                    and int(row["horizon"]) == horizon
                ]
                label = (
                    f"train={train}|fcpg-minus-nominal|{condition}|H={horizon}"
                )
                paired[label] = {}
                for metric in SUMMARY_METRICS:
                    if nominal and fcpg and all(
                        metric in row for row in (*nominal, *fcpg)
                    ):
                        paired[label][metric] = paired_task_macro_bootstrap(
                            nominal,
                            fcpg,
                            metric,
                            bootstrap_samples=bootstrap_samples,
                            rng=np.random.default_rng(
                                seed + 50_000 + pair_index
                            ),
                            higher_is_better=metric not in LOWER_IS_BETTER,
                            pair_keys=pair_keys,
                        )
                        pair_index += 1

    matched_diagonal: dict[str, Any] = {}
    for condition in conditions:
        for horizon in horizons:
            nominal = [
                row
                for row in rows
                if row["train_grounding"] == "nominal"
                and row["eval_grounding"] == "nominal"
                and row["condition"] == condition
                and int(row["horizon"]) == horizon
            ]
            fcpg = [
                row
                for row in rows
                if row["train_grounding"] == "fcpg"
                and row["eval_grounding"] == "fcpg"
                and row["condition"] == condition
                and int(row["horizon"]) == horizon
            ]
            label = f"fcpg-fcpg-minus-nominal-nominal|{condition}|H={horizon}"
            matched_diagonal[label] = {}
            for metric in SUMMARY_METRICS:
                if nominal and fcpg and all(
                    metric in row for row in (*nominal, *fcpg)
                ):
                    matched_diagonal[label][metric] = paired_task_macro_bootstrap(
                        nominal,
                        fcpg,
                        metric,
                        bootstrap_samples=bootstrap_samples,
                        rng=np.random.default_rng(seed + 100_000 + pair_index),
                        higher_is_better=metric not in LOWER_IS_BETTER,
                        pair_keys=pair_keys,
                    )
                    pair_index += 1
    return {
        "cells": cells,
        "paired_eval_grounding": paired,
        "paired_matched_diagonal": matched_diagonal,
    }


def run_counterfactual_pilot(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Run the frozen exploratory pilot and return a JSON-serializable report."""

    from fcpg.metaworld_snapshot import (
        capture_process_rng_snapshot,
        restore_process_rng_snapshot,
    )

    git_before = _git_state(repo_root)
    caller_rng = capture_process_rng_snapshot()
    plans: list[EpisodePlan] = []
    rows: list[dict[str, Any]] = []
    replay_audits: list[dict[str, Any]] = []
    try:
        frozen = load_frozen_inputs(config, repo_root=repo_root)
        initial_state = dict(frozen.state_sha256)
        evaluation = _object(config.get("evaluation"), "evaluation")
        for condition, condition_config in config["conditions"].items():
            for task in evaluation["tasks"]:
                for episode in range(int(evaluation["episodes_per_task"])):
                    plans.append(
                        collect_expert_plan(
                            str(task),
                            str(condition),
                            _object(condition_config, f"condition {condition}"),
                            episode=episode,
                            config=config,
                        )
                    )
        physical: dict[tuple[str, int], str] = {}
        for plan in plans:
            key = (plan.task, plan.episode)
            previous = physical.setdefault(key, plan.physical_snapshot_sha256)
            if previous != plan.physical_snapshot_sha256:
                raise RuntimeError(
                    "camera condition changed the paired physical snapshot"
                )

        reference_plans = {
            (plan.task, plan.episode): plan
            for plan in plans
            if plan.condition == "clean"
        }
        expected_plan_keys = {(plan.task, plan.episode) for plan in plans}
        if set(reference_plans) != expected_plan_keys:
            raise RuntimeError("every physical branch requires one clean reference plan")
        candidate_banks: dict[tuple[str, int], ActionCandidates] = {}
        for key, reference in reference_plans.items():
            donors = [
                candidate
                for candidate in reference_plans.values()
                if candidate.task == reference.task
                and candidate.episode != reference.episode
            ]
            candidate_banks[key] = build_plan_candidates(
                reference,
                frozen,
                config,
                donor=donors[0] if donors else None,
            )

        for plan in plans:
            candidates = candidate_banks[(plan.task, plan.episode)]
            plan_rows, replay = evaluate_episode_plan(
                plan,
                candidates,
                frozen,
                config,
            )
            rows.extend(plan_rows)
            replay_audits.append(
                {
                    "task": plan.task,
                    "condition": plan.condition,
                    "episode": plan.episode,
                    "snapshot": plan.physical_snapshot_sha256,
                    "candidate_action_set_sha256": canonical_sha256(
                        {
                            "names": list(candidates.names),
                            "actions": sha256_array(candidates.actions),
                        }
                    ),
                    **replay,
                }
            )
        final_state = {
            **{
                f"lewm_{name}": sha256_state_dict(model)
                for name, model in frozen.models.items()
            },
            "action_only_policy": sha256_state_dict(frozen.policy),
            "fcpg_corrector": sha256_state_dict(frozen.corrector),
            "oracle_target_encoder": sha256_state_dict(frozen.target_encoder),
            "oracle_vision_backbone": sha256_state_dict(
                frozen.corrector.vision.backbone
            ),
        }
        if final_state != initial_state:
            raise RuntimeError("a frozen model changed during the no-training pilot")
        summary = summarize_counterfactual_rows(
            rows,
            bootstrap_samples=int(evaluation["bootstrap_samples"]),
            seed=int(config["seed"]),
        )
        git_after = _git_state(repo_root)
        if git_after != git_before:
            raise RuntimeError("repository state changed during counterfactual evaluation")
        report = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "scope": dict(config["scope"]),
            "runtime_mode": str(config.get("runtime_mode", "full")),
            "summary_interpretation": str(
                config.get("summary_interpretation", "exploratory_only")
            ),
            "formal_gate_inference_allowed": False,
            "config_sha256": canonical_sha256(config),
            "git_before": git_before,
            "git_after": git_after,
            "versions": _versions(),
            "inputs": dict(frozen.file_records),
            "checkpoint_metadata": dict(frozen.checkpoint_metadata),
            "model_state_sha256_before": initial_state,
            "model_state_sha256_after": final_state,
            "no_training_verified": True,
            "row_count": len(rows),
            "rows": rows,
            "replay_audits": replay_audits,
            "summary": summary,
        }
        report["result_sha256"] = canonical_sha256(report)
        return _jsonable(report)
    finally:
        for plan in plans:
            if hasattr(plan.env, "close"):
                try:
                    plan.env.close()
                except Exception:
                    pass
        restore_process_rng_snapshot(caller_rng)
