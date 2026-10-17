#!/usr/bin/env python3
"""Build the exposed control-sufficiency P0 seed-macro row panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.control_sufficient_p0_evaluation import (  # noqa: E402
    CONTROLLABILITY_BETA,
    EXPECTED_PRE_DROP_PRIOR_CONTRACT,
    HORIZONS,
    PREDICTOR_SEEDS,
    PROTOCOL,
    SeedPanel,
    build_seed_macro_rows,
    exact_carrier_held_fixed_action_error,
    fit_eq14_style_global_weights,
    load_frozen_observer_contract,
    recursive_analytic_feedback_complementary_rollout,
    repeat_endpoint_for_policy,
    validate_seed_invariant_carriers,
)
from fcpg.lewm_analytic_feedback_observer import (  # noqa: E402
    CAUSAL_PRIOR_FIELD,
    audit_feedback_observer_artifact,
)
from fcpg.lewm_counterfactual_runtime import sha256_file  # noqa: E402
from fcpg.lewm_p1_native_complementary_evaluation import (  # noqa: E402
    checkpoint_specs_from_config,
    load_one_frozen_seed,
    validate_and_load_p1_policy,
)
from fcpg.lewm_p1_native_dynamics import (  # noqa: E402
    load_p1_native_dynamics_cache,
    rollout_matched_p1_native,
)
from fcpg.policy_token_carrier_collection import (  # noqa: E402
    PROTOCOL as POLICY_CARRIER_PROTOCOL,
    validate_policy_carrier_arrays,
)


DEFAULT_CARRIER = ROOT / (
    "outputs/policy_token_sensitivity_carrier/"
    "exposed_p0_v2_common_prefix_scored_h3_h8"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/control_sufficient_p0/"
    "exposed_p0_v2_scored_h3_h8_row_panel.json"
)


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0float64\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _observed_state_signatures(value: np.ndarray) -> list[str]:
    states = np.asarray(value, dtype=np.float32)
    if states.ndim < 2 or states.shape[0] < 1 or not np.isfinite(states).all():
        raise ValueError("observed state history must be finite [states,...]")
    signatures: list[str] = []
    for state in states:
        array = np.ascontiguousarray(state)
        digest = hashlib.sha256()
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0float32\0")
        digest.update(array.tobytes())
        signatures.append(digest.hexdigest())
    return signatures


def _resolve_from_root(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("configured artifact path must be non-empty")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _validate_carrier_record(
    carrier_dir: Path,
    record: object,
    *,
    label: str,
) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError(f"carrier metadata {label} record is invalid")
    path = (carrier_dir / str(record["path"])).resolve()
    if path.parent != carrier_dir.resolve() or sha256_file(path) != record["sha256"]:
        raise PermissionError(f"carrier metadata {label} digest/path mismatch")
    return path


@torch.inference_mode()
def _same_state_weight_fit_predict_cache(
    factual: torch.nn.Module,
    no_action: torch.nn.Module,
    cache: Any,
    *,
    device: str | torch.device,
    batch_size: int,
) -> np.ndarray:
    """Roll out every fit action bank from one fixed causal state per episode.

    Historical P1 branches contain branch-specific H1/H2 histories because
    they were collected for dynamics training.  Eq.14-style controllability,
    however, requires action variation from the same observed state.  Branch 0
    is therefore a frozen, label-free reference history; all eight future
    action sequences are counterfactually rolled out from that one history.
    """

    history = np.asarray(cache.history_token, dtype=np.float32)
    history_action = np.asarray(cache.history_action, dtype=np.float32)
    future_action = np.asarray(cache.future_action, dtype=np.float32)
    branch_index = np.asarray(cache.branch_index)
    if (
        history.ndim != 5
        or history_action.ndim != 4
        or future_action.ndim != 4
        or history.shape[:2] != future_action.shape[:2]
        or history_action.shape[:2] != future_action.shape[:2]
        or branch_index.shape != future_action.shape[:2]
    ):
        raise ValueError("strict weight-fit cache has incompatible branch axes")
    expected_branch_index = np.broadcast_to(
        np.arange(future_action.shape[1], dtype=branch_index.dtype)[None],
        branch_index.shape,
    )
    if not np.array_equal(branch_index, expected_branch_index):
        raise PermissionError("strict weight-fit branch order is not canonical")
    reference_history = np.ascontiguousarray(history[:, 0])
    reference_action = np.ascontiguousarray(history_action[:, 0])
    if not all(
        np.isfinite(value).all()
        for value in (reference_history, reference_action, future_action)
    ):
        raise ValueError("strict weight-fit deployment inputs contain non-finite values")
    target_device = torch.device(device)
    outputs: list[np.ndarray] = []
    for start in range(0, int(reference_history.shape[0]), batch_size):
        stop = min(start + batch_size, int(reference_history.shape[0]))
        factual_prediction, _ = rollout_matched_p1_native(
            factual,
            no_action,
            torch.from_numpy(reference_history[start:stop]).to(target_device),
            torch.from_numpy(reference_action[start:stop]).to(target_device),
            torch.from_numpy(np.ascontiguousarray(future_action[start:stop])).to(
                target_device
            ),
        )
        outputs.append(factual_prediction.float().cpu().numpy())
    result = np.concatenate(outputs).astype(np.float32, copy=False)
    expected = (*future_action.shape[:3], history.shape[-2], history.shape[-1])
    if result.shape != expected or not np.isfinite(result).all():
        raise RuntimeError(f"same-state weight-fit predictions must be finite {expected}")
    return np.ascontiguousarray(result)


def _unique_pre_drop(
    value: np.ndarray,
    *,
    episodes: int,
    branches: int,
) -> np.ndarray:
    reshaped = np.asarray(value, dtype=np.float32).reshape(
        episodes, branches, len(HORIZONS), *value.shape[1:]
    )
    reference = reshaped[:, :, :1]
    if not np.array_equal(reshaped, np.broadcast_to(reference, reshaped.shape)):
        raise PermissionError("pre-drop causal state differs across repeated horizon rows")
    return np.ascontiguousarray(reference[:, :, 0].reshape(episodes * branches, *value.shape[1:]))


def _shared_pre_drop_state(
    value: np.ndarray,
    *,
    episodes: int,
    branches: int,
) -> np.ndarray:
    """Collapse repeated H rows and same-snapshot action branches."""

    per_branch = _unique_pre_drop(value, episodes=episodes, branches=branches)
    shaped = per_branch.reshape(episodes, branches, *per_branch.shape[1:])
    reference = shaped[:, :1]
    if not np.array_equal(shaped, np.broadcast_to(reference, shaped.shape)):
        raise PermissionError("causal pre-drop state differs across same-snapshot branches")
    return np.ascontiguousarray(reference[:, 0])


def _load_bound_training_config(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    record = config.get("frozen_predictor_training_contract")
    if not isinstance(record, dict) or set(record) != {"config", "config_sha256"}:
        raise ValueError("frozen predictor training-config binding is invalid")
    path = _resolve_from_root(record["config"])
    if sha256_file(path) != record["config_sha256"]:
        raise PermissionError("frozen predictor training-config SHA-256 mismatch")
    return path, _load_json(path)


def _weight_fit_ids(cache: Any) -> tuple[list[str], set[str]]:
    states, branches = cache.physical_episode_id.shape
    ids: list[str] = []
    physical: set[str] = set()
    for state in range(states):
        episode_values = np.asarray(cache.physical_episode_id[state]).astype(str)
        row_values = np.asarray(cache.source_row_index[state]).astype(int)
        hazard_values = np.asarray(cache.hazard_id[state]).astype(int)
        if (
            len(set(episode_values.tolist())) != 1
            or len(set(row_values.tolist())) != 1
            or len(set(hazard_values.tolist())) != 1
        ):
            raise PermissionError("strict weight-fit state identity differs across branches")
        episode = str(episode_values[0])
        physical.add(episode)
        for seed in PREDICTOR_SEEDS:
            for horizon in HORIZONS:
                ids.append(
                    f"seed={seed}|episode={episode}|row={row_values[0]}|"
                    f"hazard={hazard_values[0]}|H={horizon}"
                )
    if len(ids) != len(set(ids)) or len(ids) != states * len(PREDICTOR_SEEDS) * len(HORIZONS):
        raise RuntimeError("strict weight-fit state identities are not unique")
    return ids, physical


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def evaluate_p0(
    *,
    carrier_dir: Path,
    observer_contract_path: Path | None,
    observer_contract_sha256: str | None,
    device: str,
    model_batch_size: int,
    policy_batch_size: int,
) -> dict[str, object]:
    """Run deployment prediction first and open offline labels only afterwards."""

    carrier_dir = carrier_dir.expanduser().resolve()
    metadata_path = carrier_dir / "carrier.metadata.json"
    metadata = _load_json(metadata_path)
    if (
        metadata.get("completed") is not True
        or metadata.get("deployment_contains_no_q_true_or_target") is not True
        or metadata.get("row_count") != 30 * 8 * len(HORIZONS)
        or metadata.get("protocol") != POLICY_CARRIER_PROTOCOL
        or metadata.get("predictor_inputs_are_from_same_policy_carrier_replay") is not True
    ):
        raise PermissionError("policy carrier metadata is incomplete or ineligible")
    if metadata.get("pre_drop_prior_contract") != EXPECTED_PRE_DROP_PRIOR_CONTRACT:
        raise PermissionError("carrier causal pre-drop prior contract drifted")
    source_config = Path(str(metadata.get("source_config", ""))).expanduser().resolve()
    if not source_config.is_file() or sha256_file(source_config) != metadata.get(
        "source_config_sha256"
    ):
        raise PermissionError("carrier source-config binding is invalid")
    observer = load_frozen_observer_contract(
        observer_contract_path, expected_sha256=observer_contract_sha256
    )

    deployment_records = metadata.get("deployments")
    if not isinstance(deployment_records, dict) or tuple(
        sorted(int(seed) for seed in deployment_records)
    ) != PREDICTOR_SEEDS:
        raise ValueError("carrier must bind exactly three frozen predictor deployments")
    deployment_paths = {
        seed: _validate_carrier_record(
            carrier_dir, deployment_records[str(seed)], label=f"seed {seed}"
        )
        for seed in PREDICTOR_SEEDS
    }
    deployments = {seed: _load_npz(path) for seed, path in deployment_paths.items()}
    # Deployment artifacts are inspected before the label archive is opened.
    if any(
        any(name in arrays for name in ("policy_target_anchor", "target_token", "q_true"))
        for arrays in deployments.values()
    ):
        raise PermissionError("deployment carrier contains an offline target")
    validate_seed_invariant_carriers(deployments)
    for seed, deployment in deployments.items():
        audit = audit_feedback_observer_artifact(deployment)
        if not audit.scorable or audit.source != "policy_carrier_causal_pre_drop_prior":
            raise PermissionError(
                f"seed {seed} carrier lacks a complete causal pre-drop observer state: "
                f"{audit.reason}"
            )

    config = _load_json(source_config)
    _, training_config = _load_bound_training_config(config)
    policy = validate_and_load_p1_policy(config, repo_root=ROOT, device=device)
    episodes, branches = 30, 8
    reference_deployment = deployments[PREDICTOR_SEEDS[0]]
    same_state = np.asarray(reference_deployment["same_state_prefix"])
    if same_state.shape != (1,) or same_state.tolist() != [True]:
        raise PermissionError("P0 requires action branches after one shared H2 state")
    scored_episode_ids = set(
        np.asarray(reference_deployment["physical_episode_id"]).astype(str).tolist()
    )
    pre_history = _shared_pre_drop_state(
        np.asarray(reference_deployment["pre_drop_history_token"]),
        episodes=episodes,
        branches=branches,
    )
    pre_action = _shared_pre_drop_state(
        np.asarray(reference_deployment["pre_drop_action_context"]),
        episodes=episodes,
        branches=branches,
    )
    final_visible = _shared_pre_drop_state(
        np.asarray(reference_deployment["pre_drop_visible_token"]),
        episodes=episodes,
        branches=branches,
    )
    # H-1,H0,H1 plus the separately observed H2 become the predictor's causal
    # H0,H1,H2 history at dropout.  The corresponding actions are H0->H1 and
    # H1->H2; no future clean state is read.
    dropout_history = np.concatenate((pre_history[:, 1:], final_visible[:, None]), axis=1)
    dropout_history_action = np.ascontiguousarray(pre_action[:, 1:])
    future_action = np.asarray(
        reference_deployment["policy_future_action"], dtype=np.float32
    ).reshape(episodes, branches, len(HORIZONS), -1)
    geometry_token = np.asarray(
        reference_deployment["policy_geometry_token"], dtype=np.float32
    ).reshape(episodes, branches, len(HORIZONS), 3, 256)

    training_data = training_config["data"]
    strict_cache_path = _resolve_from_root(training_data["strict_cache"])
    strict_metadata_path = _resolve_from_root(training_data["strict_metadata"])
    if sha256_file(strict_cache_path) != training_data["strict_cache_sha256"]:
        raise PermissionError("strict weight-fit cache SHA-256 mismatch")
    if sha256_file(strict_metadata_path) != training_data["strict_metadata_sha256"]:
        raise PermissionError("strict weight-fit metadata SHA-256 mismatch")
    weight_fit_cache = load_p1_native_dynamics_cache(
        strict_cache_path,
        metadata_path=strict_metadata_path,
        split=("train", "development"),
    )
    weight_fit_ids, weight_fit_episode_ids = _weight_fit_ids(weight_fit_cache)
    overlap = sorted(scored_episode_ids & weight_fit_episode_ids)
    if overlap:
        raise PermissionError(
            f"strict weight fit overlaps scored physical episodes: {overlap[:3]}"
        )
    fit_history = np.asarray(weight_fit_cache.history_token, dtype=np.float32)[:, 0]
    fit_state_signatures = _observed_state_signatures(fit_history)
    scored_state_signatures = _observed_state_signatures(dropout_history)
    signature_overlap = set(fit_state_signatures) & set(scored_state_signatures)
    if signature_overlap:
        raise PermissionError("strict weight fit overlaps scored observed anchor histories")
    fit_state_signatures_sha = _sha256_json(sorted(fit_state_signatures))
    scored_state_signatures_sha = _sha256_json(sorted(scored_state_signatures))

    endpoint_predictions: dict[int, dict[str, np.ndarray | str]] = {}
    weight_fit_predictions: dict[int, np.ndarray] = {}
    specs = {spec.seed: spec for spec in checkpoint_specs_from_config(config, repo_root=ROOT)}
    for seed in PREDICTOR_SEEDS:
        pair = load_one_frozen_seed(
            specs[seed], training_config=training_config, device=device
        )
        raw = np.asarray(
            deployments[seed]["policy_predicted_anchor_raw_factual"], dtype=np.float32
        )[:, -1]
        complementary = np.asarray(
            deployments[seed]["policy_predicted_anchor_complementary_factual"],
            dtype=np.float32,
        )[:, -1]
        checkpoint_receipt = np.asarray(
            deployments[seed]["predictor_checkpoint_sha256"]
        )
        if checkpoint_receipt.shape != (1,) or str(checkpoint_receipt[0]) != pair.sha256:
            raise PermissionError(f"seed {seed} carrier checkpoint binding drifted")
        primary = np.asarray(deployments[seed]["policy_predicted_anchor"])
        complementary_pair = np.asarray(
            deployments[seed]["policy_predicted_anchor_complementary_factual"]
        )
        if not np.array_equal(primary, complementary_pair):
            raise PermissionError(f"seed {seed} carrier primary/complementary alias drifted")

        causal_prior = _shared_pre_drop_state(
            np.asarray(deployments[seed][CAUSAL_PRIOR_FIELD]),
            episodes=episodes,
            branches=branches,
        )
        analytic = recursive_analytic_feedback_complementary_rollout(
            pair.factual,
            torch.from_numpy(dropout_history).to(device),
            torch.from_numpy(dropout_history_action).to(device),
            torch.from_numpy(future_action).to(device),
            torch.from_numpy(geometry_token).to(device),
            torch.from_numpy(final_visible).to(device),
            torch.from_numpy(causal_prior).to(device),
            observer=observer,
        ).float().cpu().numpy().reshape(-1, 3, 256)
        weight_fit_predictions[seed] = _same_state_weight_fit_predict_cache(
            pair.factual,
            pair.no_action,
            weight_fit_cache,
            device=device,
            batch_size=model_batch_size,
        )
        endpoint_predictions[seed] = {
            "checkpoint_sha256": pair.sha256,
            "raw": raw,
            "complementary": complementary,
            "analytic": analytic,
        }
        del pair
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    weights = fit_eq14_style_global_weights(
        weight_fit_predictions,
        beta=CONTROLLABILITY_BETA,
    )
    weight_fit_source_id = (
        "frozen_p1_strict_train_development_cache:branch0_history_x_8_future_actions:"
        f"{sha256_file(strict_cache_path)}"
    )
    weight_fit_ids_sha = _sha256_json(weight_fit_ids)
    weight_fit_sha = _sha256_json(
        {
            "protocol": "project_specific_local_branch_eq14_style_v1",
            "strict_cache_sha256": sha256_file(strict_cache_path),
            "predictor_checkpoint_sha256": {
                str(seed): endpoint_predictions[seed]["checkpoint_sha256"]
                for seed in PREDICTOR_SEEDS
            },
            "beta": CONTROLLABILITY_BETA,
            "epsilon": 1.0e-8,
            "fit_source_id": weight_fit_source_id,
            "fit_ids_sha256": weight_fit_ids_sha,
            "fit_count": len(weight_fit_ids),
            "prediction_sha256": {
                str(seed): _array_sha256(weight_fit_predictions[seed])
                for seed in PREDICTOR_SEEDS
            },
            "weights_sha256": hashlib.sha256(
                np.ascontiguousarray(weights).tobytes()
            ).hexdigest(),
        }
    )

    # The scored carrier's offline labels are opened only after all deployment
    # predictions and the distinct strict-cache controllability fit are frozen.
    label_path = _validate_carrier_record(
        carrier_dir, metadata.get("offline_labels"), label="offline labels"
    )
    labels = _load_npz(label_path)
    target = np.asarray(labels["policy_target_anchor"], dtype=np.float32)
    seed_panels: list[SeedPanel] = []
    for seed in PREDICTOR_SEEDS:
        deployment = deployments[seed]
        validate_policy_carrier_arrays(deployment, labels, require_predictions=True)
        raw_pair = np.asarray(
            deployment["policy_predicted_anchor_raw_factual"], dtype=np.float32
        )
        complementary_pair = repeat_endpoint_for_policy(
            np.asarray(endpoint_predictions[seed]["complementary"])
        )
        analytic_pair = repeat_endpoint_for_policy(
            np.asarray(endpoint_predictions[seed]["analytic"])
        )
        raw_error = exact_carrier_held_fixed_action_error(
            policy,
            deployment,
            raw_pair,
            target,
            device=device,
            batch_size=policy_batch_size,
        )
        complementary_error = exact_carrier_held_fixed_action_error(
            policy,
            deployment,
            complementary_pair,
            target,
            device=device,
            batch_size=policy_batch_size,
        )
        analytic_error = exact_carrier_held_fixed_action_error(
            policy,
            deployment,
            analytic_pair,
            target,
            device=device,
            batch_size=policy_batch_size,
        )
        no_action_error = exact_carrier_held_fixed_action_error(
            policy,
            deployment,
            np.asarray(
                deployment["policy_predicted_anchor_matched_no_action"],
                dtype=np.float32,
            ),
            target,
            device=device,
            batch_size=policy_batch_size,
        )
        wrong_prediction = np.asarray(
            deployment["policy_predicted_anchor_wrong_action"], dtype=np.float32
        )
        wrong_error = np.stack(
            [
                exact_carrier_held_fixed_action_error(
                    policy,
                    deployment,
                    wrong_prediction[:, shift],
                    target,
                    device=device,
                    batch_size=policy_batch_size,
                )
                for shift in range(7)
            ],
            axis=1,
        )
        seed_panels.append(
            SeedPanel(
                seed=seed,
                checkpoint_sha256=str(endpoint_predictions[seed]["checkpoint_sha256"]),
                raw_prediction=raw_pair,
                complementary_prediction=complementary_pair,
                analytic_prediction=analytic_pair,
                raw_action_error=raw_error,
                complementary_action_error=complementary_error,
                analytic_action_error=analytic_error,
                matched_no_action_action_error=no_action_error,
                wrong_action_action_error=wrong_error,
            )
        )

    identity = {
        name: np.asarray(deployments[PREDICTOR_SEEDS[0]][name])
        for name in ("physical_episode_id", "branch_index", "horizon")
    }
    rows = build_seed_macro_rows(
        identity,
        target,
        seed_panels,
        frozen_controllability_weights=weights,
        weight_fit_sha256=weight_fit_sha,
        weight_fit_source_id=weight_fit_source_id,
        weight_fit_ids_sha256=weight_fit_ids_sha,
        weight_fit_count=len(weight_fit_ids),
        weight_fit_observed_state_signatures_sha256=fit_state_signatures_sha,
        scored_observed_state_signatures_sha256=scored_state_signatures_sha,
        weight_fit_scored_signature_overlap_count=len(signature_overlap),
        weight_fit_source_artifact_sha256=sha256_file(strict_cache_path),
        controllability_beta=CONTROLLABILITY_BETA,
    )
    rows_sha256 = _sha256_json(rows)
    observer_contract_metadata = _load_json(
        Path(observer_contract_path).expanduser().resolve()
    )
    observer_contract_metadata_sha256 = _sha256_json(observer_contract_metadata)
    weight_vector_sha256 = _array_sha256(weights)
    fit_source_artifact_sha256 = sha256_file(strict_cache_path)
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "classification": "exposed_zero_training_diagnostic",
        "completed": True,
        "row_count": len(rows),
        "physical_episode_count": len(set(identity["physical_episode_id"].astype(str))),
        "predictor_seeds": list(PREDICTOR_SEEDS),
        "seed_reduction": "macro_mean_not_independent_physical_episodes",
        "action_error_scope": "carrier_held_fixed_exact_frozen_policy",
        "scored_offline_labels_opened_after_all_deployment_predictions": True,
        "causal_action_controls_scored_with_exact_frozen_policy": True,
        "observer_contract_metadata": observer_contract_metadata,
        "observer_contract_metadata_sha256": observer_contract_metadata_sha256,
        "observer_contract_source_sha256": observer.sha256,
        "analytic_observer": {
            "contract_sha256": observer.sha256,
            "feedback_gain": observer.feedback_gain,
            "residual_decay": observer.residual_decay,
            "delta_t": observer.delta_t,
            "scope": "globally_frozen_missing_measurement_extension",
            "paper_equations": "feedback velocity observer",
            "project_extension": "held_or_decayed_residual_during_complete_visual_dropout",
            "target_tuned_per_row": False,
            "prediction_uses_offline_target": False,
            "known_missingness_routing_is_not_deployable": True,
            "paper_always_observed_convergence_theorem_inherited": False,
        },
        "controllability": {
            "beta": CONTROLLABILITY_BETA,
            "epsilon": 1.0e-8,
            "weight_fit_sha256": weight_fit_sha,
            "weight_vector_sha256": weight_vector_sha256,
            "weights": weights.tolist(),
            "fit_source": weight_fit_source_id,
            "fit_ids_sha256": weight_fit_ids_sha,
            "fit_count": len(weight_fit_ids),
            "fit_source_artifact_sha256": fit_source_artifact_sha256,
            "fit_observed_state_signatures_sha256": fit_state_signatures_sha,
            "scored_observed_state_signatures_sha256": scored_state_signatures_sha,
            "fit_scored_observed_state_overlap_count": len(signature_overlap),
            "observed_state_signature_contract": (
                "sha256_shape_float32_causal_H0_H1_H2_anchor_history"
            ),
            "estimator": "project_specific_local_branch_eq14_style_not_paper_action_pool",
            "same_state_fit_construction": (
                "frozen branch-0 causal history crossed with all eight future action banks"
            ),
            "fit_target_fields_read": False,
            "scored_fresh_panel_used_to_fit_weights": False,
        },
        "provenance": {
            "carrier_metadata": str(metadata_path),
            "carrier_metadata_sha256": sha256_file(metadata_path),
            "offline_labels_sha256": sha256_file(label_path),
            "strict_weight_fit_cache_sha256": sha256_file(strict_cache_path),
        },
        "rows_sha256": rows_sha256,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--carrier-dir", type=Path, default=DEFAULT_CARRIER)
    parser.add_argument("--observer-contract", type=Path)
    parser.add_argument("--observer-contract-sha256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-batch-size", type=int, default=32)
    parser.add_argument("--policy-batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        result = evaluate_p0(
            carrier_dir=args.carrier_dir,
            observer_contract_path=args.observer_contract,
            observer_contract_sha256=args.observer_contract_sha256,
            device=str(args.device),
            model_batch_size=int(args.model_batch_size),
            policy_batch_size=int(args.policy_batch_size),
        )
        status = 0
    except PermissionError as error:
        result = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "classification": "exposed_zero_training_diagnostic",
            "completed": False,
            "decision": "Fail-Closed-Unscorable",
            "reason": str(error),
            "required_design": (
                "one globally frozen scalar feedback_gain, one globally frozen "
                "dropout residual_decay, beta=1, and an expected contract SHA-256; "
                "per-row or per-horizon target-optimal selection is forbidden"
            ),
        }
        status = 2
    _atomic_json(args.output.expanduser().resolve(), result)
    print(json.dumps({key: result[key] for key in result if key != "rows"}, indent=2))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
