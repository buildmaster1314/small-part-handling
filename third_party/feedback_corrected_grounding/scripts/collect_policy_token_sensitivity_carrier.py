#!/usr/bin/env python3
"""Collect fresh compact policy carriers and attach frozen P1 predictions."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import torch


os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_collection import make_metaworld_episode_runtime  # noqa: E402
from fcpg.lewm_counterfactual_runtime import sha256_file, sha256_state_dict  # noqa: E402
from fcpg.lewm_p1_native_complementary_evaluation import (  # noqa: E402
    COLLECTION_METADATA_REQUIRED_KEYS,
    FlattenedDeploymentInputs,
    LoadedFreshEvaluation,
    checkpoint_specs_from_config,
    evaluate_loaded_seed,
    load_fresh_evaluation_artifacts,
    load_one_frozen_seed,
    validate_formal_config,
    validate_and_load_p1_policy,
)
from fcpg.lewm_p1_native_dynamics import P1NativeDynamicsCache  # noqa: E402
from fcpg.lewm_known_drop_bridge_runtime import geometry_hold_policy_token  # noqa: E402
from fcpg.lewm_p1_native_fresh_collection import (  # noqa: E402
    BRANCH_COUNT,
    DEPLOYMENT_CACHE_KEYS,
    EPISODE_COUNT,
    LABEL_HORIZONS,
    FreshnessInventory,
    build_fresh_episode_plans,
    legacy_observed_state_signatures,
    validate_fresh_deployment_arrays,
)
from fcpg.policy_token_carrier_collection import (  # noqa: E402
    CLASSIFICATION,
    PROTOCOL,
    attach_aligned_predictions,
    attach_frozen_recursive_inputs,
    collect_policy_carrier_episode,
    policy_row_ids_for_contract,
    save_policy_carrier_bundle,
    stack_policy_carrier_episodes,
    verify_frozen_source_alignment,
)


DEFAULT_CONFIG = ROOT / "config/lewm/eval/p1_native_complementary_fresh.json"
DEFAULT_OUTPUT = ROOT / (
    "outputs/policy_token_sensitivity_carrier/"
    "exposed_p0_v2_common_prefix_scored_h3_h8"
)


@dataclass(frozen=True, slots=True)
class DeploymentFirstSource:
    """Fresh predictor inputs loaded without opening the offline-label archive."""

    config: dict[str, Any]
    training_config: dict[str, Any]
    collection_metadata: dict[str, Any]
    cache: P1NativeDynamicsCache
    deployment_path: Path
    deployment_sha256: str
    metadata_path: Path
    metadata_sha256: str
    recorded_label_path: Path
    recorded_label_sha256: str


def _resolve(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("artifact path must be a non-empty string")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_canonical_config(path: Path) -> dict[str, Any]:
    canonical = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload != canonical:
        raise ValueError("policy carrier requires the checked-in frozen P1 config")
    return payload


def _load_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def _load_npz_exact(
    path: Path, *, expected_keys: frozenset[str], label: str
) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        observed = set(archive.files)
        if observed != expected_keys:
            raise ValueError(
                f"{label} keys drifted: missing={sorted(expected_keys - observed)}, "
                f"extra={sorted(observed - expected_keys)}"
            )
        return {name: np.asarray(archive[name]) for name in archive.files}


def _metadata_artifact_record(
    metadata: Mapping[str, Any],
    name: str,
    *,
    expected_path: Path,
) -> tuple[Path, str]:
    record = metadata.get(name)
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise ValueError(f"fresh metadata {name} record is invalid")
    recorded_path = _resolve(record["path"])
    recorded_sha = str(record["sha256"])
    if recorded_path != expected_path:
        raise ValueError(f"fresh metadata {name} path mismatch")
    if len(recorded_sha) != 64 or any(char not in "0123456789abcdef" for char in recorded_sha):
        raise ValueError(f"fresh metadata {name} SHA-256 is invalid")
    return recorded_path, recorded_sha


def _deployment_only_native_cache(
    arrays: Mapping[str, np.ndarray],
    *,
    source_path: Path,
    metadata: Mapping[str, Any],
) -> P1NativeDynamicsCache:
    """Build the legacy inference view with a non-label shape sentinel.

    The legacy cache type requires ``target_token`` even though the baseline and
    frozen-predictor paths never read its values.  A NaN sentinel preserves only
    the public H3--H8 shape and makes accidental scoring fail loudly.
    """

    n = EPISODE_COUNT
    branches = BRANCH_COUNT
    physical = np.broadcast_to(
        np.asarray(arrays["physical_episode_id"]).astype(str)[:, None],
        (n, branches),
    )
    branch = np.broadcast_to(
        np.arange(branches, dtype=np.int64)[None], (n, branches)
    )
    source_row = np.broadcast_to(
        np.arange(n, dtype=np.int64)[:, None], (n, branches)
    )
    hazard = np.broadcast_to(
        np.asarray(arrays["hazard_id"], dtype=np.int64)[:, None], (n, branches)
    )
    task = np.broadcast_to(
        np.asarray(arrays["task_name"]).astype(str)[:, None], (n, branches)
    )
    split = np.full((n, branches), "fresh", dtype="U16")
    target_shape = (
        n,
        branches,
        len(LABEL_HORIZONS),
        int(np.asarray(arrays["history_token"]).shape[-2]),
        int(np.asarray(arrays["history_token"]).shape[-1]),
    )
    return P1NativeDynamicsCache(
        source_path=source_path,
        metadata={**dict(metadata), "target_token_is_nonlabel_shape_sentinel": True},
        history_token=np.ascontiguousarray(arrays["history_token"], dtype=np.float32),
        history_action=np.ascontiguousarray(arrays["history_action"], dtype=np.float32),
        future_action=np.ascontiguousarray(arrays["future_action"], dtype=np.float32),
        wrong_future_action=np.ascontiguousarray(
            arrays["wrong_future_action"], dtype=np.float32
        ),
        target_token=np.full(target_shape, np.nan, dtype=np.float32),
        history_sampled_visual=np.ascontiguousarray(
            arrays["history_sampled_visual"], dtype=np.float32
        ),
        history_xyz=np.ascontiguousarray(arrays["history_xyz"], dtype=np.float32),
        future_xyz=np.ascontiguousarray(arrays["future_xyz"], dtype=np.float32),
        source_row_index=np.ascontiguousarray(source_row),
        branch_index=np.ascontiguousarray(branch),
        task_name=np.ascontiguousarray(task),
        physical_episode_id=np.ascontiguousarray(physical),
        hazard_id=np.ascontiguousarray(hazard),
        split=split,
    )


def load_deployment_first_source(config_path: Path) -> DeploymentFirstSource:
    """Validate/load fresh deployment inputs without reading label bytes."""

    config_file = config_path.expanduser().resolve()
    config = _load_canonical_config(config_file)
    training_path, training_sha = validate_formal_config(config, repo_root=ROOT)
    if sha256_file(training_path) != training_sha:
        raise PermissionError("frozen predictor training config SHA-256 mismatch")
    training_config = _load_json_mapping(training_path, label="predictor training config")
    fresh = config["fresh_data"]
    deployment_path = _resolve(fresh["deployment_cache"])
    label_path = _resolve(fresh["label_cache"])
    metadata_path = _resolve(fresh["metadata"])
    metadata = _load_json_mapping(metadata_path, label="fresh collection metadata")
    if set(metadata) != COLLECTION_METADATA_REQUIRED_KEYS:
        raise ValueError("fresh collection metadata schema drifted")
    if (
        metadata.get("completed") is not True
        or metadata.get("config_sha256") != sha256_file(config_file)
    ):
        raise ValueError("fresh collection metadata identity/completion drifted")
    _, expected_deployment_sha = _metadata_artifact_record(
        metadata, "deployment_cache", expected_path=deployment_path
    )
    recorded_label_path, recorded_label_sha = _metadata_artifact_record(
        metadata, "label_cache", expected_path=label_path
    )
    deployment_sha = sha256_file(deployment_path)
    if deployment_sha != expected_deployment_sha:
        raise PermissionError("fresh deployment cache SHA-256 mismatch")
    arrays = _load_npz_exact(
        deployment_path,
        expected_keys=DEPLOYMENT_CACHE_KEYS,
        label="fresh deployment cache",
    )
    validate_fresh_deployment_arrays(arrays, contract=config)
    return DeploymentFirstSource(
        config=config,
        training_config=training_config,
        collection_metadata=metadata,
        cache=_deployment_only_native_cache(
            arrays, source_path=deployment_path, metadata=metadata
        ),
        deployment_path=deployment_path,
        deployment_sha256=deployment_sha,
        metadata_path=metadata_path,
        metadata_sha256=sha256_file(metadata_path),
        recorded_label_path=recorded_label_path,
        recorded_label_sha256=recorded_label_sha,
    )


def open_offline_labels_after_predictions(
    source: DeploymentFirstSource,
    config_path: Path,
    *,
    predictions_complete: bool,
) -> LoadedFreshEvaluation:
    """Open the bound offline labels only after every frozen prediction exists."""

    if predictions_complete is not True:
        raise PermissionError("offline labels cannot open before predictions complete")
    loaded = load_fresh_evaluation_artifacts(config_path, repo_root=ROOT)
    if (
        loaded.provenance.deployment_path != source.deployment_path
        or loaded.provenance.deployment_cache_sha256 != source.deployment_sha256
        or loaded.provenance.metadata_path != source.metadata_path
        or loaded.provenance.metadata_sha256 != source.metadata_sha256
        or loaded.provenance.label_path != source.recorded_label_path
        or loaded.provenance.label_cache_sha256 != source.recorded_label_sha256
    ):
        raise PermissionError("late-open source provenance differs from deployment phase")
    return loaded


@torch.inference_mode()
def _pre_drop_prior(
    predictor: torch.nn.Module,
    history: np.ndarray,
    action_context: np.ndarray,
    *,
    device: str | torch.device,
    batch_size: int,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if history.ndim != 4 or action_context.ndim != 3:
        raise ValueError("pre-drop history/action arrays have invalid rank")
    if history.shape[:2] != action_context.shape[:2]:
        raise ValueError("pre-drop history/action sequence identity mismatch")
    target_device = torch.device(device)
    outputs: list[np.ndarray] = []
    for start in range(0, int(history.shape[0]), batch_size):
        stop = min(start + batch_size, int(history.shape[0]))
        predicted = predictor.predict(
            torch.from_numpy(np.ascontiguousarray(history[start:stop])).to(target_device),
            torch.from_numpy(np.ascontiguousarray(action_context[start:stop])).to(
                target_device
            ),
        )
        if predicted.shape != torch.Size(history[start:stop].shape):
            raise RuntimeError("pre-drop predictor output shape differs from its context")
        outputs.append(predicted[:, -1].float().cpu().numpy())
    result = np.concatenate(outputs).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise RuntimeError("pre-drop prior contains non-finite values")
    return result


@torch.inference_mode()
def _same_replay_geometry_tokens(
    policy: torch.nn.Module,
    deployment: Mapping[str, np.ndarray],
    *,
    device: str | torch.device,
    batch_size: int,
) -> np.ndarray:
    """Build held-H2-visual/current-FK tokens from this carrier replay only."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    visual = np.asarray(deployment["pre_drop_visible_visual"], dtype=np.float32)
    xyz = np.asarray(deployment["policy_keypoints_xyz"], dtype=np.float32)[:, -1]
    if visual.shape[:2] != xyz.shape[:2]:
        raise ValueError("held visual and endpoint FK keypoints are misaligned")
    visual_panel = visual.reshape(
        EPISODE_COUNT, BRANCH_COUNT, len(LABEL_HORIZONS), *visual.shape[1:]
    )
    visual_reference = visual_panel[:, :1, :1]
    if not np.array_equal(
        visual_panel, np.broadcast_to(visual_reference, visual_panel.shape)
    ):
        raise PermissionError(
            "held H2 visual carrier differs across action branches or scored horizons"
        )
    target_device = torch.device(device)
    outputs: list[np.ndarray] = []
    for start in range(0, int(visual.shape[0]), batch_size):
        stop = min(start + batch_size, int(visual.shape[0]))
        token = geometry_hold_policy_token(
            policy,
            torch.from_numpy(np.ascontiguousarray(visual[start:stop])).to(target_device),
            torch.from_numpy(np.ascontiguousarray(xyz[start:stop])).to(target_device),
        )
        outputs.append(token.float().cpu().numpy())
    result = np.concatenate(outputs).astype(np.float32, copy=False)
    expected = (visual.shape[0], visual.shape[1], 256)
    if result.shape != expected or not np.isfinite(result).all():
        raise RuntimeError(f"same-replay geometry tokens must be finite {expected}")
    return np.ascontiguousarray(result)


def _same_replay_predictor_inputs(
    deployment: Mapping[str, np.ndarray],
    geometry_token: np.ndarray,
) -> FlattenedDeploymentInputs:
    """Construct common-state H0-H2 -> branch H3-H8 frozen-model inputs."""

    episodes, branches, horizons = EPISODE_COUNT, BRANCH_COUNT, len(LABEL_HORIZONS)
    history_rows = np.asarray(deployment["pre_drop_history_token"], dtype=np.float32)
    action_rows = np.asarray(deployment["pre_drop_action_context"], dtype=np.float32)
    visible_rows = np.asarray(deployment["pre_drop_visible_token"], dtype=np.float32)
    future_rows = np.asarray(deployment["policy_future_action"], dtype=np.float32)
    geometry_rows = np.asarray(geometry_token, dtype=np.float32)

    history_repeated = history_rows.reshape(episodes, branches, horizons, 3, 3, 256)
    action_repeated = action_rows.reshape(episodes, branches, horizons, 3, 20)
    visible_repeated = visible_rows.reshape(episodes, branches, horizons, 3, 256)
    for name, value in (
        ("history", history_repeated),
        ("action", action_repeated),
        ("visible", visible_repeated),
    ):
        if not np.array_equal(value, np.broadcast_to(value[:, :, :1], value.shape)):
            raise PermissionError(f"same-replay pre-drop {name} differs across H rows")

    common_history = history_repeated[:, :, 0]
    common_action = action_repeated[:, :, 0]
    final_visible = visible_repeated[:, :, 0]
    history = np.concatenate((common_history[:, :, 1:], final_visible[:, :, None]), axis=2)
    history_action = np.ascontiguousarray(common_action[:, :, 1:])
    if not np.array_equal(history, np.broadcast_to(history[:, :1], history.shape)):
        raise PermissionError("same-replay predictor history differs across action branches")
    if not np.array_equal(
        history_action, np.broadcast_to(history_action[:, :1], history_action.shape)
    ):
        raise PermissionError("same-replay predictor history actions differ across branches")

    future = future_rows.reshape(episodes, branches, horizons, 20)
    geometry = geometry_rows.reshape(episodes, branches, horizons, 3, 256)
    slots = np.arange(branches)
    wrong = np.stack(
        [future[:, (slots + shift) % branches] for shift in range(1, branches)], axis=1
    )
    total = episodes * branches
    wrong_flat = np.transpose(wrong, (0, 2, 1, 3, 4)).reshape(
        total, branches - 1, horizons, 20
    )
    return FlattenedDeploymentInputs(
        episode_count=episodes,
        branch_count=branches,
        history_token=np.ascontiguousarray(history.reshape(total, 3, 3, 256)),
        history_action=np.ascontiguousarray(history_action.reshape(total, 2, 20)),
        future_action=np.ascontiguousarray(future.reshape(total, 1, horizons, 20)),
        wrong_future_action=np.ascontiguousarray(
            wrong_flat[:, :, None], dtype=np.float32
        ),
        geometry_token=np.ascontiguousarray(
            geometry.reshape(total, 1, horizons, 3, 256), dtype=np.float32
        ),
    )


def collect_carrier(
    *,
    config_path: Path,
    device: str,
    output_dir: Path,
    batch_size: int,
    smoke: bool,
    alignment_debug_output: Path | None = None,
) -> dict[str, object]:
    config = _load_canonical_config(config_path)
    if output_dir.exists() and not smoke:
        raise FileExistsError(f"refusing to overwrite policy carrier: {output_dir}")
    source = load_deployment_first_source(config_path)
    if source.config != config:
        raise RuntimeError("deployment-first source config drifted")
    freshness = config["freshness"]
    legacy_source = _resolve(freshness["legacy_observation_source"])
    if sha256_file(legacy_source) != freshness["legacy_observation_source_sha256"]:
        raise ValueError("legacy observed-state source SHA-256 mismatch")
    inventory = FreshnessInventory(
        reset_digest_scopes={},
        observed_signature_scopes={
            "p1_fcpg_training": legacy_observed_state_signatures(legacy_source)
        },
    )
    policy = validate_and_load_p1_policy(config, repo_root=ROOT, device=device)
    policy_state = sha256_state_dict(policy)
    fresh = config["fresh_data"]
    runtime_factory = partial(
        make_metaworld_episode_runtime,
        camera=str(fresh["camera"]),
        width=int(fresh["width"]),
        height=int(fresh["height"]),
        benchmark_seed=int(fresh["benchmark_seed"]),
    )
    plans = build_fresh_episode_plans(config)
    if smoke:
        plans = plans[:1]
    episodes = []
    for index, plan in enumerate(plans, start=1):
        episodes.append(
            collect_policy_carrier_episode(
                plan,
                inventory=inventory,
                policy=policy,
                device=device,
                runtime_factory=runtime_factory,
                same_state_prefix=True,
            )
        )
        print(
            json.dumps(
                {
                    "event": "policy_carrier_episode_complete",
                    "episode": index,
                    "total": len(plans),
                    "episode_uid": plan.episode_uid,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if sha256_state_dict(policy) != policy_state:
        raise RuntimeError("frozen policy changed during policy-carrier collection")
    if smoke:
        row = episodes[0]
        return {
            "smoke": True,
            "endpoint_rows": int(row.endpoint.obs.shape[0]),
            "policy_obs_shape": list(row.endpoint.obs.shape),
            "pre_drop_history_shape": list(row.pre_drop_history_token.shape),
            "pre_drop_action_context_shape": list(row.pre_drop_action_context.shape),
            "same_state_prefix": bool(row.same_state_prefix),
        }

    # Keep native target tokens out of the deployment/prediction phase.  They
    # are materialized from this same frozen replay only after all predictors
    # and causal priors have been evaluated.
    deployment, labels = stack_policy_carrier_episodes(
        episodes, contract=config, include_labels=False
    )
    if labels is not None:
        raise RuntimeError("deployment-first stack unexpectedly materialized labels")
    same_state_receipt = np.asarray(deployment["same_state_prefix"])
    if same_state_receipt.tolist() != [True]:
        raise PermissionError("formal P0 carrier requires a same-state common prefix")
    geometry = _same_replay_geometry_tokens(
        policy, deployment, device=device, batch_size=batch_size
    )
    flattened = _same_replay_predictor_inputs(deployment, geometry)
    deployment = attach_frozen_recursive_inputs(
        deployment,
        source_row_id=policy_row_ids_for_contract(config),
        future_action=np.asarray(deployment["policy_future_action"], dtype=np.float32),
        geometry_token=geometry,
    )
    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    deployments: dict[int, Mapping[str, np.ndarray]] = {}
    checkpoint_records: dict[str, object] = {}
    for spec in checkpoint_specs_from_config(config, repo_root=ROOT):
        pair = load_one_frozen_seed(
            spec,
            training_config=source.training_config,
            device=device,
        )
        factual_state = sha256_state_dict(pair.factual)
        predictions = evaluate_loaded_seed(
            pair, flattened, device=device, batch_size=batch_size
        )
        if sha256_state_dict(pair.factual) != factual_state:
            raise RuntimeError(f"frozen predictor seed {spec.seed} changed")
        factual = predictions.factual.reshape(
            -1, predictions.factual.shape[-2], predictions.factual.shape[-1]
        )
        raw_factual = predictions.raw_factual.reshape(
            -1, predictions.raw_factual.shape[-2], predictions.raw_factual.shape[-1]
        )
        no_action = predictions.no_action.reshape(
            -1, predictions.no_action.shape[-2], predictions.no_action.shape[-1]
        )
        wrong_action = predictions.wrong_action.transpose(0, 2, 3, 1, 4, 5).reshape(
            -1,
            predictions.wrong_action.shape[1],
            predictions.wrong_action.shape[-2],
            predictions.wrong_action.shape[-1],
        )
        pre_drop_raw_prior = _pre_drop_prior(
            pair.factual,
            np.asarray(deployment["pre_drop_history_token"], dtype=np.float32),
            np.asarray(deployment["pre_drop_action_context"], dtype=np.float32),
            device=device,
            batch_size=batch_size,
        )
        attached = attach_aligned_predictions(
            deployment,
            factual,
            prediction_row_id=np.asarray(deployment["policy_row_id"]),
            pre_drop_raw_prior_token=pre_drop_raw_prior,
            raw_factual_predictions=raw_factual,
            matched_no_action_predictions=no_action,
            wrong_action_predictions=wrong_action,
            predictor_seed=spec.seed,
            predictor_checkpoint_sha256=pair.sha256,
        )
        deployments[spec.seed] = attached
        checkpoint_records[str(spec.seed)] = {
            "path": str(spec.path),
            "sha256": pair.sha256,
            "factual_state_sha256": factual_state,
        }
        del predictions, pair
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if set(deployments) != {15801, 15802, 15803}:
        raise RuntimeError("all three frozen predictor deployments are required")
    # Labels were recorded during the same physical replay, but remain isolated
    # from every predictor input. They are stacked and exposed only after all
    # frozen predictions and causal pre-drop priors have completed. The legacy
    # fresh-label archive is never opened or scored.
    _, labels = stack_policy_carrier_episodes(
        episodes, contract=config, include_labels=True
    )
    if labels is None:
        raise RuntimeError("late label stack unexpectedly returned no labels")
    expected_physical = np.repeat(
        source.cache.physical_episode_id[:, :, None],
        len(LABEL_HORIZONS),
        axis=2,
    ).reshape(-1).astype(str)
    if alignment_debug_output is not None:
        debug_path = alignment_debug_output.expanduser().resolve()
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            debug_path,
            same_replay_target_anchor=np.asarray(
                labels["policy_target_anchor"][:, -1], dtype=np.float32
            ),
            same_replay_keypoints=np.asarray(
                deployment["policy_keypoints_xyz"], dtype=np.float32
            ),
            policy_row_id=np.asarray(deployment["policy_row_id"]),
            policy_physical_episode_id=np.asarray(
                deployment["policy_physical_episode_id"]
            ),
        )
    verify_frozen_source_alignment(
        deployment,
        labels,
        expected_physical_episode_id=expected_physical,
    )

    destination = save_policy_carrier_bundle(
        output_dir,
        deployments,
        labels,
        metadata={
            "source_config": str(config_path.resolve()),
            "source_config_sha256": sha256_file(config_path),
            "reference_fresh_deployment_sha256": source.deployment_sha256,
            "source_deployment_loaded_before_offline_labels": True,
            "reference_fresh_labels_never_opened_or_scored": True,
            "offline_labels_recorded_with_same_replay_but_isolated_from_predictors": True,
            "offline_labels_stacked_and_exposed_after_predictions": True,
            "predictor_inputs_are_from_same_policy_carrier_replay": True,
            "source_deployment_cache_sha256": source.deployment_sha256,
            "reference_fresh_label_cache_sha256": source.recorded_label_sha256,
            "policy_state_sha256": policy_state,
            "predictors": checkpoint_records,
            "same_state_action_branch_contract": (
                "H-1,H0,H1,H2 share one expert prefix; action branches begin after H2"
            ),
            "pre_drop_prior_contract": (
                "predict visible H2 from causal H-1,H0,H1 plus executed "
                "H-1->H0,H0->H1,H1->H2 action blocks"
            ),
        },
    )
    return {
        "smoke": False,
        "protocol": PROTOCOL,
        "classification": CLASSIFICATION,
        "output_dir": str(destination),
        "row_count": int(np.asarray(labels["policy_row_id"]).size),
        "predictor_seeds": sorted(deployments),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--alignment-debug-output",
        type=Path,
        help="optional same-replay NPZ written before strict carrier alignment",
    )
    args = parser.parse_args(argv)
    result = collect_carrier(
        config_path=args.config,
        device=str(args.device),
        output_dir=args.output_dir.expanduser().resolve(),
        batch_size=int(args.batch_size),
        smoke=bool(args.smoke),
        alignment_debug_output=(
            args.alignment_debug_output.expanduser().resolve()
            if args.alignment_debug_output is not None
            else None
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
