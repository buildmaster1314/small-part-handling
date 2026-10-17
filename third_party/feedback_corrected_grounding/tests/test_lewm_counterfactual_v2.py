from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fcpg.lewm_counterfactual_v2 import (
    ACTION_ONLY_PROBE_KEYS,
    AUDIT_ARRAY_KEYS,
    DEPLOYMENT_KEYS,
    DIRECT_PROBE_KEYS,
    FULL_CANDIDATE_NAMES,
    GOAL_VISUAL_INPUT_DIM,
    GOAL_VISUAL_PROJECTION_SHA256,
    MODEL_NAMES,
    OFFLINE_LABEL_KEYS,
    NON_DEPLOYMENT_METADATA_KEYS,
    ORACLE_PROBE_KEYS,
    PHASE_STEPS,
    PREDICTED_PROBE_KEYS,
    PROBE_NAMES,
    PROTOCOL,
    SMOKE_CANDIDATE_NAMES,
    SCORER_INPUT_KEYS,
    TASK_NAMES,
    ActionOnlyProbeInputs,
    AuditArrays,
    DirectProbeInputs,
    DeploymentInputs,
    OracleProbeInputs,
    OfflineLabels,
    PredictedProbeInputs,
    assert_no_privileged_inputs,
    assign_physical_episode_splits,
    audit_camera_condition_extrinsics,
    audit_endpoint_rgb,
    audit_geometric_reprojection,
    audit_history_rgb,
    audit_ranking_informativeness,
    audit_episode_groups,
    build_branch_receipt_fields,
    build_g2_probe_pair,
    build_g3_probe_pair,
    build_g0_manifest,
    build_probe_inputs,
    canonical_sha256,
    expected_camera_condition_extrinsics,
    g0_eligibility,
    goal_visual_projection_seed,
    grounded_validity,
    physical_episode_sha256,
    sha256_array,
    smoke_config,
    validate_array_bundle,
    validate_full_config,
    validate_deployable_probe_inputs,
    validate_index_and_provenance,
    validate_manifest_hash,
    validate_smoke_config,
)


def _pinned(name: str) -> dict[str, str]:
    return {
        "path": f"outputs/{name}.pt",
        "sha256": hashlib.sha256(name.encode("ascii")).hexdigest(),
    }


def _full_config() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "scope": {
            "classification": "external_development",
            "upstream_training_allowed": False,
            "online_cem_allowed": False,
            "formal_claim_allowed": False,
        },
        "seed": 8192,
        "device": "cuda:0",
        "inputs": {
            "lewm_checkpoints": {
                "nominal": _pinned("nominal"),
                "fcpg": _pinned("fcpg"),
            },
            "action_only_policy": _pinned("action"),
            "fcpg_corrector": _pinned("corrector"),
            "oracle_target_encoder": _pinned("oracle"),
        },
        "oracle_vision": {
            "model_name": "vit_base_patch16_224.augreg_in1k",
            "selected_layer": 11,
            "usage": "offline_oracle_future_and_current_rgb_goal_context",
        },
        "evaluation": {
            "tasks": [
                "pick-place-v3",
                "button-press-v3",
                "reach-v3",
                "push-v3",
            ],
            "episodes_per_task": 10,
            "phase_steps": [10, 25, 35],
            "phase_names": ["early", "mid", "late"],
            "history_size": 3,
            "frame_skip": 5,
            "raw_action_steps": 25,
            "action_dim": 4,
            "transitions": 5,
            "transition_action_dim": 20,
            "horizons": [1, 2, 3, 4, 5],
            "noise_scales": [0.05, 0.15, 0.3],
            "random_count": 2,
            "candidate_names": list(FULL_CANDIDATE_NAMES),
            "progress_info_key": "in_place_reward",
            "snapshot_include_process_rng": True,
        },
        "conditions": {
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
            "translation4": {
                "translation_world": [0.04, 0.0, 0.0],
                "rotation_axis_world": [0.0, 1.0, 0.0],
                "rotation_degrees": 0.0,
            },
        },
        "environment": {
            "camera": "corner",
            "width": 224,
            "height": 224,
            "max_expert_steps": 120,
            "benchmark_seed": 8192,
            "env_seed_base": 30000,
            "task_index_offset": 0,
        },
        "split": {
            "unit": "physical_episode",
            "train_count_per_task": 7,
            "validation_count_per_task": 3,
            "assignment": "sha256_sorted",
            "upstream_non_overlap_required": True,
        },
        "goal_context": {
            "source": "current_rgb_and_task_id",
            "future_success_frame_allowed": False,
            "privileged_coordinates_allowed": False,
            "visual_dim": 48,
            "visual_input_dim": GOAL_VISUAL_INPUT_DIM,
            "projection_sha256": GOAL_VISUAL_PROJECTION_SHA256,
        },
        "smoke": {
            "episodes_per_task": 1,
            "candidate_names": list(SMOKE_CANDIDATE_NAMES),
            "output_suffix": "_smoke",
        },
        "output": {
            "directory": "outputs/lewm_counterfactual_v2/development_seed8192",
            "write_mode": "exclusive_atomic_directory",
        },
    }


def _smoke_arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    n, v, m, g, t, c, p, d, k = 12, 3, 2, 2, 3, 4, 5, 192, 3
    task_index = np.repeat(np.arange(4, dtype=np.uint8), 3)
    phase_index = np.tile(np.arange(3, dtype=np.uint8), 4)
    q_corrected = np.full((n, v, t, k, 2), 32.0, dtype=np.float32)
    valid_nominal = np.ones((n, t, k), dtype=bool)
    xyz_points = np.asarray(
        [[0.0, 0.0, 2.0], [0.1, 0.0, 2.0], [0.0, 0.1, 2.0]],
        dtype=np.float32,
    )
    keypoints_xyz = np.broadcast_to(xyz_points, (n, t, k, 3)).copy()
    projected_points = np.asarray(
        [[112.0, 112.0], [117.0, 112.0], [112.0, 117.0]],
        dtype=np.float32,
    )
    q_nominal = np.broadcast_to(projected_points, (n, t, k, 2)).copy()
    history_rgb = np.zeros((n, v, t, 224, 224, 3), dtype=np.uint8)
    for view in range(v):
        history_rgb[:, view] = 20 + 20 * view
        history_rgb[:, view, :, 0, 0, 0] += 1
    deployment = {
        "task_index": task_index,
        "history_rgb": history_rgb,
        "history_proprio": np.zeros((n, t, 4), dtype=np.float32),
        "history_keypoints_xyz": keypoints_xyz,
        "history_q_nominal": q_nominal,
        "history_valid_nominal": valid_nominal,
        "history_q_corrected": q_corrected,
        "history_valid_grounded": grounded_validity(q_corrected, valid_nominal),
        "history_action_raw": np.zeros((n, t - 1, 5, 4), dtype=np.float32),
        "candidate_action_raw": np.zeros((n, c, 25, 4), dtype=np.float32),
        "z_history": np.zeros((n, m, g, v, t, d), dtype=np.float32),
        "z_pred_future": np.zeros((n, m, g, v, c, p, d), dtype=np.float32),
        "goal_visual": np.zeros((n, v, 48), dtype=np.float32),
        "task_onehot": np.eye(4, dtype=np.float32)[task_index],
    }
    for candidate in range(c):
        deployment["candidate_action_raw"][:, candidate] = float(candidate + 1)
    labels = {
        "actual_return": np.zeros((n, c, p), dtype=np.float32),
        "actual_progress": np.zeros((n, c, p), dtype=np.float32),
        "actual_success": np.zeros((n, c, p), dtype=bool),
        "oracle_future_latent": np.zeros((n, v, c, p, d), dtype=np.float32),
        "history_q_true": np.empty((n, v, t, k, 2), dtype=np.float32),
        "history_valid_true": np.empty((n, v, t, k), dtype=bool),
        "goal_world": np.broadcast_to(
            np.asarray([0.0, 0.0, 2.0], dtype=np.float32), (n, 3)
        ).copy(),
        "goal_pixel": np.empty((n, v, 2), dtype=np.float32),
        "goal_valid": np.empty((n, v), dtype=bool),
    }
    horizon_scale = np.arange(1, p + 1, dtype=np.float32)[None, None, :]
    candidate_scale = np.arange(c, dtype=np.float32)[None, :, None]
    labels["actual_return"][:] = candidate_scale * horizon_scale
    labels["actual_progress"][:] = candidate_scale / float(max(c - 1, 1))
    labels["actual_success"][:, -1, -1] = True
    intrinsic_matrix = np.asarray(
        [[100.0, 0.0, 112.0], [0.0, 100.0, 112.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    intrinsic = np.broadcast_to(intrinsic_matrix, (n, v, 3, 3)).copy()
    nominal_extrinsic = np.broadcast_to(
        np.eye(4, dtype=np.float32), (n, 4, 4)
    ).copy()
    true_extrinsic = expected_camera_condition_extrinsics(
        nominal_extrinsic, _full_config()["conditions"]
    ).astype(np.float32)
    for sample in range(n):
        for view in range(v):
            homogeneous = np.concatenate(
                [keypoints_xyz[sample], np.ones((t, k, 1), dtype=np.float32)],
                axis=-1,
            )
            camera = np.einsum(
                "ij,...j->...i", true_extrinsic[sample, view], homogeneous
            )[..., :3]
            uvw = np.einsum("ij,...j->...i", intrinsic_matrix, camera)
            pixels = uvw[..., :2] / uvw[..., 2:3]
            labels["history_q_true"][sample, view] = pixels
            labels["history_valid_true"][sample, view] = (
                (camera[..., 2] > 0.0)
                & (pixels[..., 0] >= 0.0)
                & (pixels[..., 0] < 224.0)
                & (pixels[..., 1] >= 0.0)
                & (pixels[..., 1] < 224.0)
            )
            goal_h = np.concatenate(
                [labels["goal_world"][sample], np.ones(1, dtype=np.float32)]
            )
            goal_camera = (true_extrinsic[sample, view] @ goal_h)[:3]
            goal_uvw = intrinsic_matrix @ goal_camera
            goal_pixel = goal_uvw[:2] / goal_uvw[2]
            labels["goal_pixel"][sample, view] = goal_pixel
            labels["goal_valid"][sample, view] = bool(
                goal_camera[2] > 0.0
                and 0.0 <= goal_pixel[0] < 224.0
                and 0.0 <= goal_pixel[1] < 224.0
            )
    endpoint_dynamic_range = np.full((n, v, c, p), 200, dtype=np.uint16)
    endpoint_changed_pixels = np.full((n, v - 1, c, p), 100, dtype=np.uint32)
    endpoint_sha = np.empty((n, v, c, p, 32), dtype=np.uint8)
    for sample in range(n):
        for view in range(v):
            for candidate in range(c):
                for step in range(p):
                    endpoint_sha[sample, view, candidate, step] = np.frombuffer(
                        hashlib.sha256(
                            f"endpoint-rgb-{sample}-{view}-{candidate}-{step}".encode(
                                "ascii"
                            )
                        ).digest(),
                        dtype=np.uint8,
                    )
    audit = {
        "episode_id": np.repeat(np.arange(4, dtype=np.int32), 3),
        "env_seed": np.repeat(np.arange(100, 104, dtype=np.int64), 3),
        "phase_index": phase_index,
        "snapshot_step": np.take(PHASE_STEPS, phase_index).astype(np.int16),
        "camera_intrinsic": intrinsic,
        "true_extrinsic": true_extrinsic,
        "nominal_extrinsic": nominal_extrinsic,
        "endpoint_rgb_dynamic_range": endpoint_dynamic_range,
        "endpoint_rgb_changed_pixels": endpoint_changed_pixels,
        "endpoint_rgb_sha256": endpoint_sha,
    }
    return deployment, labels, audit


def _copy_arrays(value: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: array.copy() for name, array in value.items()}


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _smoke_index_rows(
    config: dict[str, Any],
    deployment: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    audit: dict[str, np.ndarray],
) -> tuple[dict[str, Any], ...]:
    candidate_family = tuple(config["evaluation"]["candidate_names"])
    rows = []
    for sample in range(len(deployment["task_index"])):
        shift = sample % len(candidate_family)
        permutation = tuple(
            list(range(len(candidate_family)))[shift:]
            + list(range(len(candidate_family)))[:shift]
        )
        names = tuple(candidate_family[index] for index in permutation)
        action_hashes = tuple(
            sha256_array(deployment["candidate_action_raw"][sample, slot])
            for slot in range(len(candidate_family))
        )
        history_physical = [
            [_digest(f"history-physical-{sample}-{step}") for step in range(3)]
            for _ in range(3)
        ]
        history_rng = [
            [_digest(f"history-rng-{sample}-{step}") for step in range(3)]
            for _ in range(3)
        ]
        endpoint_physical = [
            [
                [_digest(f"endpoint-physical-{sample}-{candidate}-{step}") for step in range(5)]
                for candidate in range(len(candidate_family))
            ]
            for _ in range(3)
        ]
        endpoint_rng = [
            [
                [_digest(f"endpoint-rng-{sample}-{candidate}-{step}") for step in range(5)]
                for candidate in range(len(candidate_family))
            ]
            for _ in range(3)
        ]
        receipt_fields = build_branch_receipt_fields(
            candidate_names=names,
            candidate_actions=deployment["candidate_action_raw"][sample],
            actual_return=labels["actual_return"][sample],
            actual_progress=labels["actual_progress"][sample],
            actual_success=labels["actual_success"][sample],
            oracle_future_latent=labels["oracle_future_latent"][sample],
            endpoint_physical_sha256_by_view=endpoint_physical,
            endpoint_rng_sha256_by_view=endpoint_rng,
            endpoint_rgb_sha256=audit["endpoint_rgb_sha256"][sample],
        )
        task_index = int(deployment["task_index"][sample])
        episode_id = int(audit["episode_id"][sample])
        env_seed = int(audit["env_seed"][sample])
        phase_index = int(audit["phase_index"][sample])
        rows.append(
            {
                "sample_index": sample,
                "task_index": task_index,
                "task_name": _full_config()["evaluation"]["tasks"][task_index],
                "episode_id": episode_id,
                "env_seed": env_seed,
                "phase_index": phase_index,
                "phase_name": _full_config()["evaluation"]["phase_names"][phase_index],
                "snapshot_step": int(audit["snapshot_step"][sample]),
                "split": "smoke",
                "physical_episode_sha256": physical_episode_sha256(
                    task_index, episode_id, env_seed
                ),
                "snapshot_sha256": _digest(f"snapshot-{sample}"),
                "history_physical_sha256_by_view": history_physical,
                "history_rng_sha256_by_view": history_rng,
                "view_names": ["clean", "rotation5", "translation4"],
                "candidate_names": list(names),
                "candidate_permutation": list(permutation),
                "candidate_permutation_sha256": canonical_sha256(list(permutation)),
                "candidate_action_sha256": list(action_hashes),
                "candidate_action_set_sha256": canonical_sha256(
                    {
                        "members": sorted(
                            (
                                {"name": name, "action_sha256": digest}
                                for name, digest in zip(names, action_hashes)
                            ),
                            key=lambda item: item["name"],
                        )
                    }
                ),
                **receipt_fields,
                "branch_execution_count": len(candidate_family),
                "endpoint_physical_sha256_by_view": endpoint_physical,
                "endpoint_rng_sha256_by_view": endpoint_rng,
            }
        )
    return tuple(rows)


def _provenance_inputs(value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, entry in value.items():
        if set(entry) == {"path", "sha256"}:
            result[name] = {
                "path": str(Path("/home/james/feedback-corrected-grounding") / entry["path"]),
                "expected_sha256": entry["sha256"],
                "actual_sha256": entry["sha256"],
            }
        else:
            result[name] = _provenance_inputs(entry)
    return result


def _smoke_provenance(
    config: dict[str, Any],
    *,
    upstream_status: str = "not_auditable",
) -> dict[str, Any]:
    n = 12
    c = len(config["evaluation"]["candidate_names"])
    config_hash = canonical_sha256(config)
    state = {
        name: _digest(f"model-{name}")
        for name in (
            "nominal",
            "fcpg",
            "action_only_policy",
            "fcpg_corrector",
            "oracle_target_encoder",
            "oracle_vision_backbone",
        )
    }
    git_state = {"head": _digest("head"), "dirty": False}
    return {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "scope": copy.deepcopy(config["scope"]),
        "config": copy.deepcopy(config),
        "config_sha256": config_hash,
        "git": {"before": git_state, "after": copy.deepcopy(git_state)},
        "versions": {"python": "3.test", "numpy": np.__version__},
        "inputs": _provenance_inputs(config["inputs"]),
        "model_state_before": state,
        "model_state_after": copy.deepcopy(state),
        "no_training_verified": True,
        "goal_visual_projection": {
            "seed": goal_visual_projection_seed(int(config["seed"])),
            "sha256": GOAL_VISUAL_PROJECTION_SHA256,
            "input_dim": GOAL_VISUAL_INPUT_DIM,
            "output_dim": 48,
            "source": "current_rgb_frozen_vit_prefix",
        },
        "sample_count": n,
        "physical_branch_count": n * c,
        "upstream_non_overlap": {
            "status": upstream_status,
            "reason": "pinned upstream artifacts do not expose a physical-episode seed ledger",
            "training_episode_ledger_sha256": (
                _digest("upstream-ledger") if upstream_status == "verified" else None
            ),
        },
        "result_contract_sha256": canonical_sha256(
            {
                "config_sha256": config_hash,
                "sample_count": n,
                "physical_branch_count": n * c,
            }
        ),
    }


def test_full_config_is_strict_and_smoke_override_is_bounded() -> None:
    full = _full_config()
    validate_full_config(full)
    smoke = smoke_config(full)
    validate_smoke_config(smoke)

    assert full["evaluation"]["episodes_per_task"] == 10
    assert smoke["evaluation"]["episodes_per_task"] == 1
    assert smoke["evaluation"]["candidate_names"] == list(SMOKE_CANDIDATE_NAMES)
    assert smoke["evaluation"]["noise_scales"] == [0.15]
    assert smoke["evaluation"]["random_count"] == 0
    assert smoke["evaluation"]["tasks"] == full["evaluation"]["tasks"]
    assert smoke["conditions"] == full["conditions"]
    assert smoke["inputs"] == full["inputs"]
    assert smoke["output"]["directory"].endswith("_smoke")


def test_config_validation_is_independent_of_canonical_json_object_order() -> None:
    smoke = smoke_config(_full_config())
    canonical_round_trip = json.loads(json.dumps(smoke, sort_keys=True))

    validate_smoke_config(canonical_round_trip)
    assert set(canonical_round_trip["inputs"]["lewm_checkpoints"]) == set(
        MODEL_NAMES
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("scope", "online_cem_allowed"), True),
        (("evaluation", "phase_steps"), [10, 20, 35]),
        (("split", "unit"), "snapshot"),
        (("goal_context", "future_success_frame_allowed"), True),
        (("conditions", "translation4", "translation_world"), [0.0, 0.0, 0.0]),
        (("inputs", "fcpg_corrector", "sha256"), "BAD"),
    ],
)
def test_full_config_rejects_contract_drift(path: tuple[str, ...], value: Any) -> None:
    config = _full_config()
    target: Any = config
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    with pytest.raises((TypeError, ValueError)):
        validate_full_config(config)


def test_exact_key_types_and_recursive_privileged_rejection() -> None:
    deployment, labels, audit = _smoke_arrays()
    assert set(deployment) == set(DEPLOYMENT_KEYS)
    assert set(labels) == set(OFFLINE_LABEL_KEYS)
    assert set(audit) == set(AUDIT_ARRAY_KEYS)
    assert SCORER_INPUT_KEYS == PREDICTED_PROBE_KEYS
    assert SCORER_INPUT_KEYS < DEPLOYMENT_KEYS
    assert NON_DEPLOYMENT_METADATA_KEYS.isdisjoint(SCORER_INPUT_KEYS)
    DeploymentInputs.from_mapping(deployment)
    OfflineLabels.from_mapping(labels)
    AuditArrays.from_mapping(audit)

    extra = dict(deployment, goal_pixel=labels["goal_pixel"])
    with pytest.raises(ValueError, match="privileged"):
        DeploymentInputs.from_mapping(extra)
    with pytest.raises(ValueError, match="privileged"):
        assert_no_privileged_inputs({"safe": [{"goal_info": {"success": 1}}]})
    for name in sorted(NON_DEPLOYMENT_METADATA_KEYS):
        with pytest.raises(ValueError, match="privileged"):
            assert_no_privileged_inputs({name: audit[name]})
        with pytest.raises(ValueError, match="privileged"):
            DeploymentInputs.from_mapping(dict(deployment, **{name: audit[name]}))

    legal = {"history_rgb": deployment["history_rgb"], "goal_visual": deployment["goal_visual"]}
    assert_no_privileged_inputs(legal)


def _distinguishable_probe_arrays() -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]
]:
    deployment, labels, audit = _smoke_arrays()
    deployment = _copy_arrays(deployment)
    labels = _copy_arrays(labels)
    for model in range(2):
        for grounding in range(2):
            for view in range(3):
                cell = float(100 * model + 10 * grounding + view)
                deployment["z_history"][:, model, grounding, view] = cell
                deployment["z_pred_future"][:, model, grounding, view] = cell + 0.5
    for view in range(3):
        deployment["goal_visual"][:, view] = float(view + 1)
        labels["oracle_future_latent"][:, view] = float(20 + view)
    deployment["history_action_raw"][:] = 3.0
    deployment["candidate_action_raw"][:] = 4.0
    return deployment, labels, audit


def test_probe_builders_expose_only_the_selected_minimal_feature_view() -> None:
    deployment, labels, _ = _distinguishable_probe_arrays()
    common = {"model": "fcpg", "grounding": "nominal", "view": "rotation5"}

    action = build_probe_inputs("action_only", deployment, **common)
    direct = build_probe_inputs("direct", deployment, **common)
    predicted = build_probe_inputs("predicted", deployment, **common)
    oracle = build_probe_inputs("oracle", deployment, labels=labels, **common)

    assert isinstance(action, ActionOnlyProbeInputs)
    assert isinstance(direct, DirectProbeInputs)
    assert isinstance(predicted, PredictedProbeInputs)
    assert isinstance(oracle, OracleProbeInputs)
    assert set(action.as_mapping()) == set(ACTION_ONLY_PROBE_KEYS)
    assert set(direct.as_mapping()) == set(DIRECT_PROBE_KEYS)
    assert set(predicted.as_mapping()) == set(PREDICTED_PROBE_KEYS)
    assert set(oracle.as_mapping()) == set(ORACLE_PROBE_KEYS)
    assert OracleProbeInputs.OFFLINE_ONLY is True

    np.testing.assert_array_equal(direct.z_history, 101.0)
    np.testing.assert_array_equal(predicted.z_history, direct.z_history)
    np.testing.assert_array_equal(predicted.z_pred_future, 101.5)
    np.testing.assert_array_equal(oracle.oracle_future_latent, 21.0)
    np.testing.assert_array_equal(action.goal_visual, 2.0)
    for record in (action, direct, predicted, oracle):
        for array in record.as_mapping().values():
            assert array.flags.writeable is False
    assert not np.shares_memory(direct.z_history, deployment["z_history"])

    action_other_cell = build_probe_inputs(
        "action_only", deployment, model="nominal", grounding="fcpg", view="rotation5"
    )
    for name in ACTION_ONLY_PROBE_KEYS:
        np.testing.assert_array_equal(
            action.as_mapping()[name], action_other_cell.as_mapping()[name]
        )

    selected_before = direct.z_history.copy()
    deployment["z_history"][:, 1, 0, 1] = -99.0
    np.testing.assert_array_equal(direct.z_history, selected_before)


def test_probe_selection_is_invariant_to_poisoned_other_cells() -> None:
    deployment, labels, _ = _distinguishable_probe_arrays()
    selectors = {"model": 0, "grounding": 0, "view": 0}
    before = {
        probe: build_probe_inputs(
            probe,
            deployment,
            labels=labels if probe == "oracle" else None,
            **selectors,
        ).as_mapping()
        for probe in PROBE_NAMES
    }

    poisoned = _copy_arrays(deployment)
    poisoned_labels = _copy_arrays(labels)
    poisoned["z_history"][:, 1, :, :, :, :] = 9_001.0
    poisoned["z_history"][:, 0, 1, :, :, :] = 9_002.0
    poisoned["z_history"][:, 0, 0, 1:, :, :] = 9_003.0
    poisoned["z_pred_future"][:, 1, :, :, :, :, :] = 9_011.0
    poisoned["z_pred_future"][:, 0, 1, :, :, :, :] = 9_012.0
    poisoned["z_pred_future"][:, 0, 0, 1:, :, :, :] = 9_013.0
    poisoned["goal_visual"][:, 1:] = 9_020.0
    poisoned_labels["oracle_future_latent"][:, 1:] = 9_030.0

    for probe in PROBE_NAMES:
        after = build_probe_inputs(
            probe,
            poisoned,
            labels=poisoned_labels if probe == "oracle" else None,
            **selectors,
        ).as_mapping()
        assert set(after) == set(before[probe])
        for name in after:
            np.testing.assert_array_equal(after[name], before[probe][name])


def test_probe_boundary_rejects_metadata_index_labels_and_invalid_selectors() -> None:
    deployment, labels, audit = _distinguishable_probe_arrays()
    selectors = {"model": "nominal", "grounding": "nominal", "view": "clean"}
    for forbidden in (
        {"episode_id": audit["episode_id"]},
        {"index_rows": [{"sample_index": 0}]},
        {"history_q_true": labels["history_q_true"]},
    ):
        with pytest.raises(ValueError):
            build_probe_inputs("direct", dict(deployment, **forbidden), **selectors)

    with pytest.raises(ValueError, match="offline labels may only"):
        build_probe_inputs("predicted", deployment, labels=labels, **selectors)
    with pytest.raises(ValueError, match="requires offline labels"):
        build_probe_inputs("oracle", deployment, **selectors)
    with pytest.raises(ValueError, match="unknown probe"):
        build_probe_inputs("mystery", deployment, **selectors)
    with pytest.raises(ValueError, match="unknown model"):
        build_probe_inputs("direct", deployment, model="other", grounding=0, view=0)
    with pytest.raises(IndexError, match="grounding selector"):
        build_probe_inputs("direct", deployment, model=0, grounding=2, view=0)
    with pytest.raises(IndexError, match="view selector"):
        build_probe_inputs("direct", deployment, model=0, grounding=0, view=-1)
    with pytest.raises(TypeError, match="model selector"):
        build_probe_inputs("direct", deployment, model=True, grounding=0, view=0)


def test_probe_views_enforce_g2_and_g3_matched_information_contracts() -> None:
    deployment, _, _ = _distinguishable_probe_arrays()
    nominal = build_probe_inputs(
        "predicted", deployment, model="nominal", grounding="nominal", view="clean"
    )
    fcpg = build_probe_inputs(
        "predicted", deployment, model="nominal", grounding="fcpg", view="clean"
    )
    direct = build_probe_inputs(
        "direct", deployment, model="nominal", grounding="nominal", view="clean"
    )

    # G2 adds only the selected predicted rollout to an otherwise identical
    # DirectRanker input; G3 freezes model/view and changes only grounding.
    assert set(nominal.as_mapping()) - set(direct.as_mapping()) == {"z_pred_future"}
    for name in DIRECT_PROBE_KEYS:
        np.testing.assert_array_equal(nominal.as_mapping()[name], direct.as_mapping()[name])
    for name in PREDICTED_PROBE_KEYS - {"z_history", "z_pred_future"}:
        np.testing.assert_array_equal(nominal.as_mapping()[name], fcpg.as_mapping()[name])
    assert not np.array_equal(nominal.z_history, fcpg.z_history)
    assert not np.array_equal(nominal.z_pred_future, fcpg.z_pred_future)


def test_matched_probe_builders_and_deployable_boundary_are_fail_closed() -> None:
    deployment, labels, _ = _distinguishable_probe_arrays()
    direct, predicted = build_g2_probe_pair(
        deployment, model="fcpg", grounding="nominal", view="rotation5"
    )
    nominal, corrected = build_g3_probe_pair(
        deployment, model="fcpg", view="rotation5"
    )

    for probe in (direct, predicted, nominal, corrected):
        validate_deployable_probe_inputs(probe)
    for name in DIRECT_PROBE_KEYS:
        np.testing.assert_array_equal(
            direct.as_mapping()[name], predicted.as_mapping()[name]
        )
    for name in PREDICTED_PROBE_KEYS - {"z_history", "z_pred_future"}:
        np.testing.assert_array_equal(
            nominal.as_mapping()[name], corrected.as_mapping()[name]
        )

    with pytest.raises(TypeError, match="pooled or oracle"):
        validate_deployable_probe_inputs(DeploymentInputs.from_mapping(deployment))
    oracle = build_probe_inputs(
        "oracle",
        deployment,
        labels=labels,
        model="fcpg",
        grounding="nominal",
        view="rotation5",
    )
    with pytest.raises(TypeError, match="pooled or oracle"):
        validate_deployable_probe_inputs(oracle)  # type: ignore[arg-type]
    writable = DirectProbeInputs(
        **{name: np.array(value, copy=True) for name, value in direct.as_mapping().items()}
    )
    with pytest.raises(ValueError, match="read-only"):
        validate_deployable_probe_inputs(writable)

    malformed_values = {
        name: np.array(value, copy=True) for name, value in direct.as_mapping().items()
    }
    malformed_values["goal_visual"] = malformed_values["goal_visual"][:, :-1]
    for array in malformed_values.values():
        array.setflags(write=False)
    malformed = DirectProbeInputs(**malformed_values)
    with pytest.raises(ValueError, match="goal_visual"):
        validate_deployable_probe_inputs(malformed)


def test_ranking_audit_reports_label_signal_and_rejects_duplicate_actions() -> None:
    deployment, labels, audit = _smoke_arrays()
    report = audit_ranking_informativeness(
        DeploymentInputs.from_mapping(deployment),
        OfflineLabels.from_mapping(labels),
        AuditArrays.from_mapping(audit),
    )
    assert report["passed"] is True
    assert report["candidate_actions_unique"] is True
    assert report["min_unique_candidate_count"] == 4
    assert report["min_pairwise_action_l2"] > 0.0
    assert report["ranking_informative"] is True
    assert report["labels"]["actual_return"]["informative"] is True

    duplicate = _copy_arrays(deployment)
    duplicate["candidate_action_raw"][0, 1] = duplicate["candidate_action_raw"][0, 0]
    with pytest.raises(ValueError, match="duplicate candidate action chunks"):
        audit_ranking_informativeness(
            DeploymentInputs.from_mapping(duplicate),
            OfflineLabels.from_mapping(labels),
            AuditArrays.from_mapping(audit),
        )

    tied = _copy_arrays(labels)
    tied["actual_return"][:] = 0.0
    tied["actual_progress"][:] = 0.0
    tied["actual_success"][:] = False
    tied_report = audit_ranking_informativeness(
        DeploymentInputs.from_mapping(deployment),
        OfflineLabels.from_mapping(tied),
        AuditArrays.from_mapping(audit),
    )
    assert tied_report["passed"] is True
    assert tied_report["ranking_informative"] is False


def test_smoke_array_bundle_closes_all_axes_and_camera_invariant_labels() -> None:
    deployment, labels, audit = _smoke_arrays()
    axes = validate_array_bundle(deployment, labels, audit, candidate_count=4)

    assert axes.sample_count == 12
    assert axes.candidate_count == 4
    assert labels["actual_return"].shape == (12, 4, 5)
    assert labels["actual_progress"].ndim == 3
    assert labels["actual_success"].ndim == 3
    assert labels["oracle_future_latent"].shape[1] == 3
    rgb_audit = audit_history_rgb(deployment["history_rgb"])
    assert rgb_audit["frame_count"] == 12 * 3 * 3
    assert rgb_audit["min_frame_dynamic_range"] == 1
    assert rgb_audit["min_perturbed_changed_pixels"] > 0
    endpoint_audit = audit_endpoint_rgb(AuditArrays.from_mapping(audit))
    assert endpoint_audit["frame_count"] == 12 * 3 * 4 * 5
    assert endpoint_audit["min_frame_dynamic_range"] == 200
    assert endpoint_audit["sha256_evidence_complete"] is True


def test_rgb_audit_rejects_constant_and_unapplied_camera_views() -> None:
    deployment, labels, audit = _smoke_arrays()
    constant = _copy_arrays(deployment)
    constant["history_rgb"][0, 1, 0] = 0
    with pytest.raises(ValueError, match="constant frame"):
        validate_array_bundle(constant, labels, audit, candidate_count=4)

    identical = _copy_arrays(deployment)
    identical["history_rgb"][0, 1, 0] = identical["history_rgb"][0, 0, 0]
    with pytest.raises(ValueError, match="pixel-identical"):
        validate_array_bundle(identical, labels, audit, candidate_count=4)


def test_endpoint_rgb_audit_rejects_constant_identical_and_empty_digest() -> None:
    deployment, labels, audit = _smoke_arrays()
    constant = _copy_arrays(audit)
    constant["endpoint_rgb_dynamic_range"][0, 1, 2, 3] = 0
    with pytest.raises(ValueError, match="endpoint RGB contains a constant frame"):
        validate_array_bundle(deployment, labels, constant, candidate_count=4)

    identical = _copy_arrays(audit)
    identical["endpoint_rgb_changed_pixels"][0, 0, 2, 3] = 0
    with pytest.raises(ValueError, match="pixel-identical to clean"):
        validate_array_bundle(deployment, labels, identical, candidate_count=4)

    empty = _copy_arrays(audit)
    empty["endpoint_rgb_sha256"][0, 2, 2, 3] = 0
    with pytest.raises(ValueError, match="SHA-256 evidence is empty"):
        validate_array_bundle(deployment, labels, empty, candidate_count=4)

    impossible = _copy_arrays(audit)
    impossible["endpoint_rgb_changed_pixels"][0, 1, 2, 3] = 224 * 224 + 1
    with pytest.raises(ValueError, match="exceeds image area"):
        validate_array_bundle(deployment, labels, impossible, candidate_count=4)


def test_corrected_validity_uses_bounds_and_not_oracle_validity() -> None:
    deployment, labels, audit = _smoke_arrays()
    deployment = _copy_arrays(deployment)
    deployment["history_q_corrected"][0, 0, 0] = np.asarray(
        [[0.0, 223.0], [-0.01, 10.0], [224.0, 10.0]], dtype=np.float32
    )
    expected = grounded_validity(
        deployment["history_q_corrected"], deployment["history_valid_nominal"]
    )
    deployment["history_valid_grounded"] = expected
    validate_array_bundle(deployment, labels, audit, candidate_count=4)
    assert bool(expected[0, 0, 0, 0])
    assert not bool(expected[0, 0, 0, 1])
    assert not bool(expected[0, 0, 0, 2])

    deployment["history_valid_grounded"][0, 0, 0, 1] = True
    with pytest.raises(ValueError, match="corrected-point bounds"):
        validate_array_bundle(deployment, labels, audit, candidate_count=4)


@pytest.mark.parametrize(
    "mutation",
    ["bad_dtype", "bad_candidate_shape", "nonfinite", "camera_axis_on_return"],
)
def test_array_schema_fails_closed(mutation: str) -> None:
    deployment, labels, audit = _smoke_arrays()
    if mutation == "bad_dtype":
        deployment["task_index"] = deployment["task_index"].astype(np.int64)
    elif mutation == "bad_candidate_shape":
        deployment["candidate_action_raw"] = deployment["candidate_action_raw"][:, :3]
    elif mutation == "nonfinite":
        deployment["goal_visual"][0, 0, 0] = np.nan
    elif mutation == "camera_axis_on_return":
        labels["actual_return"] = np.repeat(labels["actual_return"][:, None], 3, axis=1)
    with pytest.raises((TypeError, ValueError)):
        validate_array_bundle(deployment, labels, audit, candidate_count=4)


def test_physical_episode_split_is_deterministic_and_phase_grouped() -> None:
    task = np.repeat(np.arange(4, dtype=np.uint8), 10 * 3)
    episode = np.tile(np.repeat(np.arange(10, dtype=np.int32), 3), 4)
    seed = (task.astype(np.int64) * 1000 + episode).astype(np.int64)
    phases = np.tile(np.arange(3), 40)

    first, report = assign_physical_episode_splits(task, episode, seed)
    permutation = np.random.default_rng(7).permutation(len(task))
    second, second_report = assign_physical_episode_splits(
        task[permutation], episode[permutation], seed[permutation]
    )
    restored = np.empty_like(second)
    restored[permutation] = second

    np.testing.assert_array_equal(first, restored)
    assert report == second_report
    for task_index in range(4):
        rows = task == task_index
        train_episodes = set(episode[rows & (first == "train")].tolist())
        val_episodes = set(episode[rows & (first == "validation")].tolist())
        assert len(train_episodes) == 7
        assert len(val_episodes) == 3
        assert train_episodes.isdisjoint(val_episodes)
    for task_index in range(4):
        for episode_id in range(10):
            rows = (task == task_index) & (episode == episode_id)
            assert set(first[rows]) in ({"train"}, {"validation"})
            assert set(phases[rows]) == {0, 1, 2}


def test_episode_audit_requires_all_fixed_phases() -> None:
    deployment, _, audit = _smoke_arrays()
    typed = DeploymentInputs.from_mapping(deployment)
    typed_audit = AuditArrays.from_mapping(audit)
    report = audit_episode_groups(typed, typed_audit, smoke=True)
    assert report["passed"] is True
    assert report["g0_eligible"] is False

    bad = _copy_arrays(audit)
    bad["phase_index"][1] = 0
    bad["snapshot_step"][1] = 10
    with pytest.raises(ValueError, match="phase coverage"):
        audit_episode_groups(typed, AuditArrays.from_mapping(bad), smoke=True)


def _permute(value: dict[str, np.ndarray], order: np.ndarray) -> dict[str, np.ndarray]:
    return {name: array[order] for name, array in value.items()}


def test_manifest_is_order_invariant_self_hashed_and_tamper_evident() -> None:
    config = smoke_config(_full_config())
    deployment, labels, audit = _smoke_arrays()
    first = build_g0_manifest(config, deployment, labels, audit)
    validate_manifest_hash(first)
    assert first["g0_audit"]["passed"] is False
    assert first["g0_audit"]["index_and_provenance"]["passed"] is False

    permutation = np.random.default_rng(99).permutation(12)
    second = build_g0_manifest(
        config,
        _permute(deployment, permutation),
        _permute(labels, permutation),
        _permute(audit, permutation),
    )
    assert first == second
    assert first["axes"]["C"] == 4
    assert first["g0_audit"]["g0_eligible"] is False
    assert first["camera_invariant_label_keys"] == [
        "actual_progress",
        "actual_return",
        "actual_success",
    ]

    tampered = copy.deepcopy(first)
    tampered["axes"]["D"] = 191
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        validate_manifest_hash(tampered)


def test_manifest_hash_changes_when_one_deployment_value_changes() -> None:
    config = smoke_config(_full_config())
    deployment, labels, audit = _smoke_arrays()
    first = build_g0_manifest(config, deployment, labels, audit)
    changed = _copy_arrays(deployment)
    changed["candidate_action_raw"][0, 0, 0, 0] = 0.25
    second = build_g0_manifest(config, changed, labels, audit)

    assert first["manifest_sha256"] != second["manifest_sha256"]
    assert (
        first["deployment_arrays"]["candidate_action_raw"]["sha256"]
        != second["deployment_arrays"]["candidate_action_raw"]["sha256"]
    )


def test_geometric_reprojection_closes_nominal_true_goal_and_intrinsics() -> None:
    deployment, labels, audit = _smoke_arrays()
    report = audit_geometric_reprojection(
        DeploymentInputs.from_mapping(deployment),
        OfflineLabels.from_mapping(labels),
        AuditArrays.from_mapping(audit),
    )
    assert report["passed"] is True
    assert report["intrinsics_identical_across_views"] is True
    assert max(report["max_abs_error_pixels"].values()) <= 1e-5

    bad_nominal = _copy_arrays(deployment)
    bad_nominal["history_q_nominal"][0, 0, 0, 0] += 0.1
    with pytest.raises(ValueError, match="history_q_nominal"):
        validate_array_bundle(bad_nominal, labels, audit, candidate_count=4)

    bad_true = _copy_arrays(labels)
    bad_true["history_q_true"][0, 1, 0, 0, 1] += 0.1
    with pytest.raises(ValueError, match="history_q_true"):
        validate_array_bundle(deployment, bad_true, audit, candidate_count=4)

    bad_goal = _copy_arrays(labels)
    bad_goal["goal_valid"][0, 0] = False
    with pytest.raises(ValueError, match="goal_valid"):
        validate_array_bundle(deployment, bad_goal, audit, candidate_count=4)

    bad_intrinsic = _copy_arrays(audit)
    bad_intrinsic["camera_intrinsic"][0, 2, 0, 0] += 1.0
    with pytest.raises(ValueError, match="intrinsic K differs"):
        validate_array_bundle(deployment, labels, bad_intrinsic, candidate_count=4)


def test_camera_condition_audit_binds_view_order_rotation_and_translation() -> None:
    _, _, audit = _smoke_arrays()
    config = _full_config()
    typed = AuditArrays.from_mapping(audit)
    report = audit_camera_condition_extrinsics(typed, config["conditions"])
    assert report["passed"] is True
    assert report["condition_order"] == ["clean", "rotation5", "translation4"]
    np.testing.assert_allclose(
        audit["true_extrinsic"][:, 0], audit["nominal_extrinsic"], atol=1e-6
    )
    np.testing.assert_allclose(
        audit["true_extrinsic"][:, 2, :3, 3],
        np.broadcast_to(np.asarray([-0.04, 0.0, 0.0]), (12, 3)),
        atol=1e-6,
    )
    assert not np.allclose(
        audit["true_extrinsic"][:, 1], audit["nominal_extrinsic"]
    )

    swapped = _copy_arrays(audit)
    swapped["true_extrinsic"][:, [1, 2]] = swapped["true_extrinsic"][:, [2, 1]]
    with pytest.raises(ValueError, match="frozen camera condition"):
        audit_camera_condition_extrinsics(
            AuditArrays.from_mapping(swapped), config["conditions"]
        )

    wrong_conditions = copy.deepcopy(config["conditions"])
    wrong_conditions["rotation5"]["rotation_degrees"] = 4.0
    with pytest.raises(ValueError, match="camera conditions changed"):
        audit_camera_condition_extrinsics(typed, wrong_conditions)


def test_expected_camera_conditions_apply_world_delta_to_nontrivial_pose() -> None:
    angle = np.deg2rad(30.0)
    base_rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    base_position = np.asarray([0.3, -0.2, 1.7], dtype=np.float64)
    cv_from_mj = np.diag([1.0, -1.0, -1.0, 1.0])
    world_to_base = np.eye(4, dtype=np.float64)
    world_to_base[:3, :3] = base_rotation.T
    world_to_base[:3, 3] = -base_rotation.T @ base_position
    nominal = (cv_from_mj @ world_to_base)[None]

    actual = expected_camera_condition_extrinsics(
        nominal, _full_config()["conditions"]
    )[0]
    y = np.deg2rad(5.0)
    delta_y = np.asarray(
        [
            [np.cos(y), 0.0, np.sin(y)],
            [0.0, 1.0, 0.0],
            [-np.sin(y), 0.0, np.cos(y)],
        ],
        dtype=np.float64,
    )
    expected_rotation = delta_y @ base_rotation
    world_to_rotated = np.eye(4, dtype=np.float64)
    world_to_rotated[:3, :3] = expected_rotation.T
    world_to_rotated[:3, 3] = -expected_rotation.T @ base_position
    translated_position = base_position + np.asarray([0.04, 0.0, 0.0])
    world_to_translated = np.eye(4, dtype=np.float64)
    world_to_translated[:3, :3] = base_rotation.T
    world_to_translated[:3, 3] = -base_rotation.T @ translated_position

    np.testing.assert_allclose(actual[0], nominal[0], atol=1e-12)
    np.testing.assert_allclose(actual[1], cv_from_mj @ world_to_rotated, atol=1e-12)
    np.testing.assert_allclose(
        actual[2], cv_from_mj @ world_to_translated, atol=1e-12
    )


def test_exact_index_provenance_audit_binds_slots_views_and_upstream_status() -> None:
    config = smoke_config(_full_config())
    deployment, labels, audit = _smoke_arrays()
    rows = _smoke_index_rows(config, deployment, labels, audit)
    provenance = _smoke_provenance(config)
    report = validate_index_and_provenance(
        config,
        deployment,
        labels,
        audit,
        rows,
        provenance,
        smoke=True,
    )

    assert report["passed"] is True
    assert report["g0_eligible"] is False
    assert report["camera_pairing_verified"] is True
    assert report["candidate_slot_hashes_verified"] is True
    assert report["exact_branch_execution_count"] == 48
    assert report["upstream_non_overlap"]["status"] == "not_auditable"
    assert "upstream_episode_non_overlap_not_auditable" in report["ineligibility_reasons"]

    manifest = build_g0_manifest(
        config,
        deployment,
        labels,
        audit,
        index_rows=rows,
        provenance=provenance,
    )
    validate_manifest_hash(manifest)
    assert manifest["index_rows_sha256"] == report["index_rows_sha256"]
    assert manifest["provenance_sha256"] == report["provenance_sha256"]
    assert manifest["g0_audit"]["g0_eligible"] is False
    assert manifest["g0_audit"]["episodes_per_task"] == {
        task: 1 for task in TASK_NAMES
    }
    assert "candidate_family" in manifest["axis_names"]
    assert manifest["axis_names"]["candidate_slots"] == [
        "slot_0",
        "slot_1",
        "slot_2",
        "slot_3",
    ]


@pytest.mark.parametrize(
    ("field", "mutation", "match"),
    [
        ("candidate_action_sha256", lambda value: value.__setitem__(0, _digest("bad")), "slot hash"),
        ("candidate_permutation", lambda value: value.__setitem__(0, 0), "not bijective"),
        ("branch_execution_count", lambda _value: None, "exactly once"),
        (
            "history_physical_sha256_by_view",
            lambda value: value[1].__setitem__(0, _digest("different-view")),
            "differ across camera views",
        ),
        (
            "endpoint_rng_sha256_by_view",
            lambda value: value[2][0].__setitem__(0, _digest("different-endpoint")),
            "differ across camera views",
        ),
        (
            "branch_receipt_sha256",
            lambda value: value.__setitem__(0, _digest("tampered-receipt")),
            "branch receipt",
        ),
    ],
)
def test_exact_index_audit_rejects_tampering(
    field: str,
    mutation: Any,
    match: str,
) -> None:
    config = smoke_config(_full_config())
    deployment, labels, audit = _smoke_arrays()
    rows = [
        copy.deepcopy(row)
        for row in _smoke_index_rows(config, deployment, labels, audit)
    ]
    if field == "branch_execution_count":
        rows[0][field] = 3
    elif field == "candidate_permutation":
        rows[0][field][1] = rows[0][field][0]
    else:
        mutation(rows[0][field])
    with pytest.raises(ValueError, match=match):
        validate_index_and_provenance(
            config,
            deployment,
            labels,
            audit,
            rows,
            _smoke_provenance(config),
            smoke=True,
        )


def test_branch_receipt_rejects_offline_label_candidate_axis_swap() -> None:
    config = smoke_config(_full_config())
    deployment, labels, audit = _smoke_arrays()
    candidate_count = labels["actual_return"].shape[1]
    labels["actual_return"][0, :, 0] = np.arange(candidate_count)
    labels["actual_progress"][0, :, 1] = np.arange(candidate_count) + 10
    labels["actual_success"][0, 1::2, 2] = True
    rows = _smoke_index_rows(config, deployment, labels, audit)

    tampered = _copy_arrays(labels)
    permutation = np.asarray([1, 0, 2, 3])
    for name in ("actual_return", "actual_progress", "actual_success"):
        tampered[name][0] = tampered[name][0, permutation]
    with pytest.raises(ValueError, match="candidate label slot hash"):
        validate_index_and_provenance(
            config,
            deployment,
            tampered,
            audit,
            rows,
            _smoke_provenance(config),
            smoke=True,
        )


def test_candidate_slot_binding_rejects_action_candidate_axis_swap() -> None:
    config = smoke_config(_full_config())
    deployment, labels, audit = _smoke_arrays()
    candidate_count = deployment["candidate_action_raw"].shape[1]
    for slot in range(candidate_count):
        deployment["candidate_action_raw"][0, slot] = float(slot + 1)
    rows = _smoke_index_rows(config, deployment, labels, audit)

    tampered = _copy_arrays(deployment)
    tampered["candidate_action_raw"][0] = tampered["candidate_action_raw"][
        0, [1, 0, 2, 3]
    ]
    with pytest.raises(ValueError, match="candidate action slot hash"):
        validate_index_and_provenance(
            config,
            tampered,
            labels,
            audit,
            rows,
            _smoke_provenance(config),
            smoke=True,
        )


def test_branch_receipt_rejects_oracle_future_candidate_axis_swap() -> None:
    config = smoke_config(_full_config())
    deployment, labels, audit = _smoke_arrays()
    candidate_count = labels["oracle_future_latent"].shape[2]
    for slot in range(candidate_count):
        labels["oracle_future_latent"][0, :, slot, 0, 0] = float(slot + 1)
    rows = _smoke_index_rows(config, deployment, labels, audit)

    tampered = _copy_arrays(labels)
    permutation = np.asarray([1, 0, 2, 3])
    tampered["oracle_future_latent"][0] = np.take(
        tampered["oracle_future_latent"][0], permutation, axis=1
    )
    with pytest.raises(ValueError, match="oracle future latent slot hash"):
        validate_index_and_provenance(
            config,
            deployment,
            tampered,
            audit,
            rows,
            _smoke_provenance(config),
            smoke=True,
        )


def test_branch_receipt_rejects_endpoint_rgb_candidate_axis_swap() -> None:
    config = smoke_config(_full_config())
    deployment, labels, audit = _smoke_arrays()
    rows = _smoke_index_rows(config, deployment, labels, audit)

    tampered = _copy_arrays(audit)
    permutation = np.asarray([1, 0, 2, 3])
    tampered["endpoint_rgb_sha256"][0] = np.take(
        tampered["endpoint_rgb_sha256"][0], permutation, axis=1
    )
    with pytest.raises(ValueError, match="endpoint RGB slot hash"):
        validate_index_and_provenance(
            config,
            deployment,
            labels,
            tampered,
            rows,
            _smoke_provenance(config),
            smoke=True,
        )


def test_branch_receipts_follow_anonymous_candidate_slot_permutation() -> None:
    config = smoke_config(_full_config())
    deployment, labels, audit = _smoke_arrays()
    candidate_count = deployment["candidate_action_raw"].shape[1]
    for slot in range(candidate_count):
        deployment["candidate_action_raw"][0, slot] = float(slot + 1)
        deployment["z_pred_future"][0, :, :, :, slot] = float(slot + 1)
        labels["actual_return"][0, slot] = float(slot + 1)
        labels["actual_progress"][0, slot] = float(slot + 11)
        labels["actual_success"][0, slot, slot % 5] = True
        labels["oracle_future_latent"][0, :, slot] = float(slot + 21)
    rows = [
        copy.deepcopy(row)
        for row in _smoke_index_rows(config, deployment, labels, audit)
    ]

    moved_deployment = _copy_arrays(deployment)
    moved_labels = _copy_arrays(labels)
    moved_audit = _copy_arrays(audit)
    permutation = np.asarray([2, 0, 3, 1])
    moved_deployment["candidate_action_raw"][0] = moved_deployment[
        "candidate_action_raw"
    ][0, permutation]
    moved_deployment["z_pred_future"][0] = np.take(
        moved_deployment["z_pred_future"][0], permutation, axis=3
    )
    for name in ("actual_return", "actual_progress", "actual_success"):
        moved_labels[name][0] = moved_labels[name][0, permutation]
    moved_labels["oracle_future_latent"][0] = np.take(
        moved_labels["oracle_future_latent"][0], permutation, axis=1
    )
    for name in ("endpoint_rgb_dynamic_range", "endpoint_rgb_sha256"):
        moved_audit[name][0] = np.take(moved_audit[name][0], permutation, axis=1)
    moved_audit["endpoint_rgb_changed_pixels"][0] = np.take(
        moved_audit["endpoint_rgb_changed_pixels"][0], permutation, axis=1
    )

    row = rows[0]
    row["candidate_names"] = [row["candidate_names"][slot] for slot in permutation]
    row["candidate_permutation"] = [
        row["candidate_permutation"][slot] for slot in permutation
    ]
    row["candidate_permutation_sha256"] = canonical_sha256(
        row["candidate_permutation"]
    )
    for name in (
        "candidate_action_sha256",
        "candidate_label_sha256",
        "branch_receipt_sha256",
    ):
        row[name] = [row[name][slot] for slot in permutation]
    for name in (
        "oracle_future_latent_sha256_by_view",
        "endpoint_rgb_sha256_by_view",
        "endpoint_physical_sha256_by_view",
        "endpoint_rng_sha256_by_view",
    ):
        row[name] = [
            [view[slot] for slot in permutation] for view in row[name]
        ]

    report = validate_index_and_provenance(
        config,
        moved_deployment,
        moved_labels,
        moved_audit,
        rows,
        _smoke_provenance(config),
        smoke=True,
    )
    assert report["candidate_slot_hashes_verified"] is True


def test_provenance_audit_is_frozen_and_upstream_fail_closed() -> None:
    config = smoke_config(_full_config())
    deployment, labels, audit = _smoke_arrays()
    rows = _smoke_index_rows(config, deployment, labels, audit)

    verified = _smoke_provenance(config, upstream_status="verified")
    report = validate_index_and_provenance(
        config,
        deployment,
        labels,
        audit,
        rows,
        verified,
        smoke=True,
    )
    assert report["upstream_non_overlap"]["status"] == "verified"
    assert report["g0_eligible"] is False  # smoke can never become formal G0 evidence
    assert "upstream_episode_non_overlap_not_auditable" not in report["ineligibility_reasons"]

    changed = copy.deepcopy(verified)
    changed["model_state_after"]["fcpg"] = _digest("changed-model")
    with pytest.raises(ValueError, match="frozen model state"):
        validate_index_and_provenance(
            config,
            deployment,
            labels,
            audit,
            rows,
            changed,
            smoke=True,
        )

    overclaim = copy.deepcopy(_smoke_provenance(config))
    overclaim["upstream_non_overlap"]["training_episode_ledger_sha256"] = _digest(
        "invented-ledger"
    )
    with pytest.raises(ValueError, match="cannot claim"):
        validate_index_and_provenance(
            config,
            deployment,
            labels,
            audit,
            rows,
            overclaim,
            smoke=True,
        )

    projection_tamper = copy.deepcopy(_smoke_provenance(config))
    projection_tamper["goal_visual_projection"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="projection SHA-256 mismatch"):
        validate_index_and_provenance(
            config,
            deployment,
            labels,
            audit,
            rows,
            projection_tamper,
            smoke=True,
        )


def test_full_g0_eligibility_requires_clean_git_and_verified_upstream() -> None:
    assert g0_eligibility(
        smoke=False,
        upstream_status="verified",
        git_dirty=False,
    ) == (True, [])
    eligible, reasons = g0_eligibility(
        smoke=False,
        upstream_status="verified",
        git_dirty=True,
    )
    assert eligible is False
    assert reasons == ["dirty_worktree_full_collection"]
    eligible, reasons = g0_eligibility(
        smoke=False,
        upstream_status="not_auditable",
        git_dirty=False,
    )
    assert eligible is False
    assert reasons == ["upstream_episode_non_overlap_not_auditable"]
    audit_camera_condition_extrinsics,
    audit_endpoint_rgb,
