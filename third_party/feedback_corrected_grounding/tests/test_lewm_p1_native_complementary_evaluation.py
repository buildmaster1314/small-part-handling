from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn

import fcpg.lewm_p1_native_complementary_evaluation as evaluation_module
from fcpg.lewm_p1_native_complementary import LoadedP1NativePair
from fcpg.lewm_p1_native_complementary_evaluation import (
    FreshArtifactProvenance,
    build_seed_rows,
    episode_horizon_mse,
    evaluate_loaded_seed,
    flatten_branch_specific_deployment,
    load_fresh_evaluation_artifacts,
    validate_formal_config,
)
from fcpg.lewm_p1_native_complementary_statistics import REQUIRED_METHODS
from fcpg.lewm_p1_native_dynamics import (
    P1NativeBaselines,
    P1NativeDynamicsCache,
    P1NativePredictorConfig,
)
from fcpg.lewm_p1_native_fresh_collection import (
    DEPLOYMENT_CACHE_KEYS,
    OFFLINE_LABEL_CACHE_KEYS,
)


N, S, T, H, K, D, A = 2, 8, 3, 6, 3, 256, 20


def _cache() -> P1NativeDynamicsCache:
    history = np.zeros((N, S, T, K, D), dtype=np.float32)
    for episode in range(N):
        for branch in range(S):
            history[episode, branch, -1] = 10.0 * episode + branch
    future = np.zeros((N, S, H, A), dtype=np.float32)
    for branch in range(S):
        future[:, branch, :, 0] = branch + 1
    wrong = np.stack(
        [future[:, (np.arange(S) + shift) % S] for shift in range(1, 8)], axis=1
    )
    episodes = np.asarray(
        [[f"pick-place-v3:{episode}"] * S for episode in range(N)]
    )
    return P1NativeDynamicsCache(
        source_path=None,
        metadata=None,
        history_token=history,
        history_action=np.zeros((N, S, T - 1, A), dtype=np.float32),
        future_action=future,
        wrong_future_action=wrong,
        target_token=np.zeros((N, S, H, K, D), dtype=np.float32),
        history_sampled_visual=np.zeros((N, S, T, K, D), dtype=np.float32),
        history_xyz=np.zeros((N, S, T, K, 3), dtype=np.float32),
        future_xyz=np.zeros((N, S, H, K, 3), dtype=np.float32),
        source_row_index=np.broadcast_to(np.arange(N)[:, None], (N, S)),
        branch_index=np.broadcast_to(np.arange(S)[None], (N, S)),
        task_name=np.full((N, S), "pick-place-v3"),
        physical_episode_id=episodes,
        hazard_id=np.zeros((N, S), dtype=np.int64),
        split=np.full((N, S), "fresh_confirmation"),
    )


class _FakePredictor(nn.Module):
    history_size = T
    num_keypoints = K
    latent_dim = D
    action_dim = A

    def __init__(self, *, action_scale: float) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.action_scale = action_scale
        self.eval()

    def predict(self, embedding: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        output = embedding.clone()
        update = self.action_scale * action[:, -1, 0].reshape(-1, 1, 1)
        output[:, -1] = output[:, -1] + update
        return output


def _loaded_pair() -> LoadedP1NativePair:
    return LoadedP1NativePair(
        seed=15801,
        path=Path("fake.pt"),
        sha256="a" * 64,
        factual=_FakePredictor(action_scale=1.0),  # type: ignore[arg-type]
        no_action=_FakePredictor(action_scale=0.0),  # type: ignore[arg-type]
        predictor_config=P1NativePredictorConfig(),
        action_mean=torch.zeros(A),
        action_std=torch.ones(A),
    )


def _provenance() -> FreshArtifactProvenance:
    return FreshArtifactProvenance(
        config_path=Path("config.json"),
        deployment_path=Path("deployment.npz"),
        label_path=Path("labels.npz"),
        metadata_path=Path("metadata.json"),
        training_config_path=Path("training.json"),
        config_sha256="1" * 64,
        deployment_cache_sha256="2" * 64,
        label_cache_sha256="3" * 64,
        metadata_sha256="4" * 64,
        training_config_sha256="5" * 64,
        dataset_sha256="6" * 64,
    )


def test_flatten_keeps_each_branch_history_and_uses_runtime_s_one() -> None:
    cache = _cache()
    geometry = np.zeros((N, S, H, K, D), dtype=np.float32)
    flat = flatten_branch_specific_deployment(cache, geometry)
    assert flat.history_token.shape == (N * S, T, K, D)
    assert flat.future_action.shape == (N * S, 1, H, A)
    assert flat.wrong_future_action.shape == (N * S, 7, 1, H, A)
    assert flat.geometry_token.shape == (N * S, 1, H, K, D)
    np.testing.assert_array_equal(flat.history_token[S + 3, -1], 13.0)
    np.testing.assert_array_equal(
        flat.wrong_future_action[2, 0, 0, :, 0],
        cache.future_action[0, 3, :, 0],
    )


def test_seed_evaluation_preserves_branch_identity_and_wrong_controls() -> None:
    cache = _cache()
    flat = flatten_branch_specific_deployment(
        cache, np.zeros((N, S, H, K, D), dtype=np.float32)
    )
    result = evaluate_loaded_seed(_loaded_pair(), flat, device="cpu", batch_size=5)
    assert result.factual.shape == (N, S, H, K, D)
    assert result.no_action.shape == result.factual.shape
    assert result.wrong_action.shape == (N, 7, S, H, K, D)
    assert result.raw_factual.shape == result.factual.shape
    assert float(result.factual[0, 1, 0, 0, 0]) == pytest.approx(0.75)
    assert float(result.factual[0, 2, 0, 0, 0]) == pytest.approx(1.25)
    assert float(result.no_action[0, 2, 0, 0, 0]) == pytest.approx(0.5)
    assert float(result.wrong_action[0, 0, 2, 0, 0, 0]) == pytest.approx(1.5)
    assert float(result.raw_factual[0, 2, 0, 0, 0]) == pytest.approx(5.0)


def test_episode_horizon_mse_reduces_branch_anchor_and_token_only() -> None:
    target = np.zeros((N, S, H, K, D), dtype=np.float32)
    prediction = np.ones_like(target)
    prediction[1] *= 2.0
    metric = episode_horizon_mse(prediction, target)
    assert metric.shape == (N, H)
    np.testing.assert_array_equal(metric[0], 1.0)
    np.testing.assert_array_equal(metric[1], 4.0)


def test_build_rows_keeps_raw_lewm_out_of_formal_panel() -> None:
    cache = _cache()
    zero = np.zeros((N, S, H, K, D), dtype=np.float32)
    predictions = evaluate_loaded_seed(
        _loaded_pair(),
        flatten_branch_specific_deployment(cache, zero),
        device="cpu",
        batch_size=N * S,
    )
    baselines = P1NativeBaselines(
        hold_last_token=zero,
        geometry_hold_exact_fk_token=zero,
        visual_cv_exact_fk_token=zero,
    )
    formal, raw = build_seed_rows(
        seed=15801,
        checkpoint_sha256="a" * 64,
        predictions=predictions,
        baselines=baselines,
        target_token=zero,
        physical_episode_id=cache.physical_episode_id[:, 0],
        provenance=_provenance(),
    )
    assert len(formal) == len(REQUIRED_METHODS) * N * H
    assert len(raw) == N * H
    assert {row["method"] for row in formal} == set(REQUIRED_METHODS)
    assert {row["method"] for row in raw} == {"raw_lewm_factual"}
    assert all(row["collector_protocol"] == "fcpg_p1_native_complementary_fresh_v1" for row in formal)
    assert all(row["fresh_data_protocol"] == "p1-native-fresh-v1" for row in formal)


def test_checked_in_formal_config_binds_source_training_file() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "config/lewm/eval/p1_native_complementary_fresh.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_path, source_sha = validate_formal_config(config, repo_root=root)
    assert source_path == (root / "config/lewm/eval/p1_native_dynamics_dev.json").resolve()
    assert source_sha == config["frozen_predictor_training_contract"]["config_sha256"]


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_artifact_loader_keeps_labels_separate_until_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    frozen = json.loads(
        (root / "config/lewm/eval/p1_native_complementary_fresh.json").read_text(
            encoding="utf-8"
        )
    )
    deployment_path = tmp_path / "deployment.npz"
    label_path = tmp_path / "labels.npz"
    metadata_path = tmp_path / "metadata.json"
    inventory_path = tmp_path / "legacy-signatures.npz"
    config_path = tmp_path / "config.json"
    frozen["fresh_data"]["deployment_cache"] = str(deployment_path)
    frozen["fresh_data"]["label_cache"] = str(label_path)
    frozen["fresh_data"]["metadata"] = str(metadata_path)
    frozen["freshness"]["legacy_observed_state_inventory"] = str(inventory_path)
    config_path.write_text(json.dumps(frozen, sort_keys=True), encoding="utf-8")

    shared = DEPLOYMENT_CACHE_KEYS & OFFLINE_LABEL_CACHE_KEYS
    identity = {name: np.asarray([f"identity-{name}"]) for name in shared}
    deployment = {
        name: identity[name].copy() if name in identity else np.asarray([1.0])
        for name in DEPLOYMENT_CACHE_KEYS
    }
    labels = {
        name: identity[name].copy() if name in identity else np.asarray([7.0])
        for name in OFFLINE_LABEL_CACHE_KEYS
    }
    np.savez_compressed(deployment_path, **deployment)
    np.savez_compressed(label_path, **labels)
    np.savez_compressed(inventory_path, signatures=np.asarray(["f" * 64]))
    cache_contract = evaluation_module.cache_contract_json(frozen)
    collection_dataset_sha = hashlib.sha256(
        (
            cache_contract
            + "\0deployment\0"
            + _file_sha(deployment_path)
            + "\0offline-labels\0"
            + _file_sha(label_path)
        ).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema_version": 1,
        "protocol": "p1-native-fresh-v1",
        "completed": True,
        "config_sha256": _file_sha(config_path),
        "deployment_cache": {
            "path": str(deployment_path),
            "sha256": _file_sha(deployment_path),
        },
        "label_cache": {
            "path": str(label_path),
            "sha256": _file_sha(label_path),
        },
        "dataset_sha256": collection_dataset_sha,
        "policy_checkpoint": {
            "path": frozen["p1"]["policy_checkpoint"],
            "sha256": frozen["p1"]["policy_checkpoint_sha256"],
        },
        "corrector_checkpoint": {
            "path": frozen["p1"]["corrector_checkpoint"],
            "sha256": frozen["p1"]["corrector_checkpoint_sha256"],
        },
        "legacy_observation_source": {
            "path": frozen["freshness"]["legacy_observation_source"],
            "sha256": frozen["freshness"]["legacy_observation_source_sha256"],
        },
        "freshness_receipt": {
            "legacy_training_reset_digest_status": (
                "unverified_missing_legacy_reset_metadata"
            ),
            "reset_digest_scope_completeness_claimed": False,
            "known_optional_reset_scope_counts": {},
            "observed_signature_scope_counts": {"p1_fcpg_training": 1},
            "legacy_observed_state_inventory": {
                "path": str(inventory_path),
                "sha256": _file_sha(inventory_path),
                "signature_count": 1,
            },
            "fresh_observed_signature_count": 30,
            "exact_observed_signature_overlap_count": 0,
        },
        "cache_contract_json": cache_contract,
    }
    metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")

    observed: dict[str, np.ndarray] = {}

    def fake_convert(arrays, **_kwargs):
        observed.update(arrays)
        return _cache()

    def fake_combine(deployment_arrays, label_arrays, **_kwargs):
        return {**deployment_arrays, **label_arrays}

    monkeypatch.setattr(evaluation_module, "fresh_arrays_to_native_cache", fake_convert)
    monkeypatch.setattr(evaluation_module, "combine_fresh_cache_arrays", fake_combine)
    loaded = load_fresh_evaluation_artifacts(config_path, repo_root=root)
    assert "target_token" not in deployment
    assert "history_token" not in labels
    assert "target_token" in observed and "history_token" in observed
    np.testing.assert_array_equal(loaded.target_token, labels["target_token"].astype(np.float32))
    assert loaded.provenance.config_sha256 == _file_sha(config_path)
