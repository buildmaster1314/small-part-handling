from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from fcpg.lewm_formal_policy_bridge import (
    A3_FORMAL_CHECKPOINT_TYPE,
    A3_FORMAL_TRAINING_PROTOCOL,
    FORMAL_BRIDGE_CACHE_TYPE,
    FORMAL_BRIDGE_HORIZONS,
    FORMAL_BRIDGE_SEMANTICS,
    assert_formal_split_pair,
    build_formal_bridge_checkpoint,
    episode_horizon_row_weights,
    formal_episode_mse,
    formal_token_metrics,
    forward_bridge_replicates,
    load_formal_bridge_for_runtime,
    load_formal_bridge_split,
    make_formal_bridge_triplet,
    paired_episode_bootstrap,
    validate_a3_formal_checkpoint_header,
)
from fcpg.lewm_policy_token_bridge import sha256_file
from fcpg.lewm_known_drop_bridge_runtime import load_anchor_policy_bridges
from scripts.train_lewm_formal_policy_bridge import _moments, _report


def _file(path: Path, value: bytes) -> Path:
    path.write_bytes(value)
    return path


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _upstream(tmp_path: Path) -> dict[str, object]:
    policy = _file(tmp_path / "policy.pt", b"p1-policy")
    corrector = _file(tmp_path / "corrector.pt", b"p1-corrector")
    replicates = []
    for index, seed in enumerate((29301, 29302, 29303)):
        checkpoint = tmp_path / f"a3_{seed}.pt"
        torch.save(
            {
                "checkpoint_type": A3_FORMAL_CHECKPOINT_TYPE,
                "training_protocol": A3_FORMAL_TRAINING_PROTOCOL,
                "schema_version": 1,
                "step": 10_000,
                "completed": True,
                "selection": "fixed_latest",
                "seed": seed,
                "predictor_config": {
                    "latent_dim": 192,
                    "num_keypoints": 3,
                    "action_dim": 20,
                    "history_size": 3,
                },
                "factual_state": {"weight": torch.tensor([float(index)])},
                "no_action_state": {"weight": torch.tensor([float(index)])},
            },
            checkpoint,
        )
        replicates.append(
            {
                "seed": seed,
                "checkpoint": _artifact(checkpoint),
                "factual_state_key": "factual_state",
                "matched_no_action_state_key": "no_action_state",
            }
        )
    return {
        "p1_policy": _artifact(policy),
        "p1_corrector": _artifact(corrector),
        "a3_replicates": replicates,
        "collector": {
            "implementation_sha256": "1" * 64,
            "source_data_sha256": "2" * 64,
        },
    }


def test_real_a3_10k_checkpoint_header_matches_formal_contract() -> None:
    checkpoint = (
        Path(__file__).resolve().parents[1]
        / "outputs/lewm_action_branch_a3/train_seed29301_10k/latest.pt"
    )
    if not checkpoint.is_file():
        pytest.skip("real A3 10k checkpoint is not present")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_a3_formal_checkpoint_header(payload, expected_seed=29301)


def _cache(
    root: Path,
    *,
    split: str,
    upstream: dict[str, object],
    episodes: tuple[str, ...],
    duplicate_first: bool = False,
) -> Path:
    root.mkdir()
    rows: list[tuple[int, int]] = []
    for episode_index in range(len(episodes)):
        for horizon in FORMAL_BRIDGE_HORIZONS:
            rows.append((episode_index, horizon))
            if duplicate_first and episode_index == 0:
                rows.append((episode_index, horizon))
    count = len(rows)
    rng = np.random.default_rng(123 if split == "train" else 456)
    factual = rng.normal(size=(count, 3, 3, 192)).astype(np.float32)
    no_action = rng.normal(size=(count, 3, 3, 192)).astype(np.float32)
    oracle = rng.normal(size=(count, 3, 192)).astype(np.float32)
    last_embedding = rng.normal(size=(count, 3, 192)).astype(np.float32)
    last_token = rng.normal(size=(count, 3, 256)).astype(np.float32)
    target_token = last_token + rng.normal(scale=0.1, size=last_token.shape).astype(np.float32)
    geometry = last_token + rng.normal(scale=0.05, size=last_token.shape).astype(np.float32)
    last_xyz = rng.normal(size=(count, 3, 3)).astype(np.float32)
    current_xyz = last_xyz + rng.normal(scale=0.02, size=last_xyz.shape).astype(np.float32)
    tasks = np.asarray(
        ["pick-place-v3" if episode_index == 0 else "push-v3" for episode_index, _ in rows],
        dtype="U32",
    )
    episode_sha = np.asarray([episodes[index] for index, _ in rows], dtype="U64")
    horizons = np.asarray([horizon for _, horizon in rows], dtype=np.int64)
    arrays = root / "samples.npz"
    np.savez_compressed(
        arrays,
        factual_prediction=factual,
        matched_no_action_prediction=no_action,
        target_oracle_embedding=oracle,
        last_valid_embedding=last_embedding,
        last_valid_token=last_token,
        target_token=target_token,
        geometry_hold_token=geometry,
        last_valid_xyz=last_xyz,
        current_xyz=current_xyz,
        task_name=tasks,
        physical_episode_sha256=episode_sha,
        horizon=horizons,
    )
    metadata = {
        "schema_version": 1,
        "cache_type": FORMAL_BRIDGE_CACHE_TYPE,
        "bridge_semantics": FORMAL_BRIDGE_SEMANTICS,
        "split": split,
        "arrays": {"path": "samples.npz", "sha256": sha256_file(arrays)},
        "contract": {
            "num_keypoints": 3,
            "latent_dim": 192,
            "token_dim": 256,
            "replicate_count": 3,
            "replicate_semantics": "correlated_single_seed_predictions",
            "horizons": list(FORMAL_BRIDGE_HORIZONS),
            "target_alignment": "future_pre_action",
        },
        "provenance": upstream,
    }
    (root / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return root


def test_formal_reader_preserves_explicit_a3_replicate_axis(tmp_path: Path) -> None:
    upstream = _upstream(tmp_path)
    cache = _cache(
        tmp_path / "train",
        split="train",
        upstream=upstream,
        episodes=("a" * 64, "b" * 64),
    )
    split = load_formal_bridge_split(cache)
    assert split.factual_prediction.shape == (16, 3, 3, 192)
    assert split.matched_no_action_prediction.shape == (16, 3, 3, 192)
    assert split.replicate_seeds == (29301, 29302, 29303)


def test_formal_reader_rejects_upstream_sha_mismatch(tmp_path: Path) -> None:
    upstream = _upstream(tmp_path)
    cache = _cache(
        tmp_path / "train",
        split="train",
        upstream=upstream,
        episodes=("a" * 64,),
    )
    Path(upstream["p1_policy"]["path"]).write_bytes(b"changed")  # type: ignore[index]
    with pytest.raises(ValueError, match="p1_policy SHA-256 mismatch"):
        load_formal_bridge_split(cache)


def test_train_validation_require_physical_disjointness(tmp_path: Path) -> None:
    upstream = _upstream(tmp_path)
    train = load_formal_bridge_split(
        _cache(
            tmp_path / "train",
            split="train",
            upstream=upstream,
            episodes=("a" * 64,),
        )
    )
    validation = load_formal_bridge_split(
        _cache(
            tmp_path / "validation",
            split="validation",
            upstream=upstream,
            episodes=("a" * 64,),
        )
    )
    with pytest.raises(ValueError, match="physical episodes overlap"):
        assert_formal_split_pair(train, validation)


def test_train_validation_require_identical_source_data_provenance(
    tmp_path: Path,
) -> None:
    upstream = _upstream(tmp_path)
    train = load_formal_bridge_split(
        _cache(
            tmp_path / "train",
            split="train",
            upstream=upstream,
            episodes=("a" * 64,),
        )
    )
    validation = load_formal_bridge_split(
        _cache(
            tmp_path / "validation",
            split="validation",
            upstream=upstream,
            episodes=("b" * 64,),
        )
    )
    changed_metadata = json.loads(json.dumps(validation.metadata))
    changed_metadata["provenance"]["collector"]["source_data_sha256"] = "3" * 64
    changed = replace(validation, metadata=changed_metadata)
    with pytest.raises(ValueError, match="provenance mismatch for collector"):
        assert_formal_split_pair(train, changed)


def test_formal_reader_rejects_reordered_a3_seed_axis(tmp_path: Path) -> None:
    upstream = _upstream(tmp_path)
    upstream["a3_replicates"] = list(reversed(upstream["a3_replicates"]))  # type: ignore[index]
    cache = _cache(
        tmp_path / "train",
        split="train",
        upstream=upstream,
        episodes=("a" * 64,),
    )
    with pytest.raises(ValueError, match="seed axis must be exactly"):
        load_formal_bridge_split(cache)


def test_row_weights_equalize_episode_and_horizon_with_duplicate_rows(tmp_path: Path) -> None:
    split = load_formal_bridge_split(
        _cache(
            tmp_path / "train",
            split="train",
            upstream=_upstream(tmp_path),
            episodes=("a" * 64, "b" * 64),
            duplicate_first=True,
        )
    )
    weights = episode_horizon_row_weights(split)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[split.physical_episode_sha256 == "a" * 64].sum() == pytest.approx(0.5)
    assert weights[split.physical_episode_sha256 == "b" * 64].sum() == pytest.approx(0.5)
    for episode in ("a" * 64, "b" * 64):
        for horizon in FORMAL_BRIDGE_HORIZONS:
            mask = (split.physical_episode_sha256 == episode) & (split.horizon == horizon)
            assert weights[mask].sum() == pytest.approx(1.0 / 16.0)


def test_shared_bridge_forwards_each_seed_without_averaging() -> None:
    bridge, _, _ = make_formal_bridge_triplet(initialization_seed=9, hidden_dim=4)
    with torch.no_grad():
        bridge.input_projection.weight.zero_()
        bridge.input_projection.bias.zero_()
        bridge.input_projection.weight[0, 192] = 1.0
        bridge.output_projection.weight.zero_()
        bridge.output_projection.bias.zero_()
        bridge.output_projection.weight[:, 0] = 1.0
    predicted = torch.zeros(1, 3, 3, 192)
    predicted[:, 0, :, 0] = 1.0
    predicted[:, 1, :, 0] = 2.0
    predicted[:, 2, :, 0] = 3.0
    result = forward_bridge_replicates(
        bridge,
        predicted,
        torch.zeros(1, 3, 192),
        torch.zeros(1, 3, 256),
        torch.zeros(1, 3, 3),
        torch.zeros(1, 3, 3),
    )
    assert result.shape == (1, 3, 3, 256)
    assert not torch.equal(result[:, 0], result[:, 1])
    assert not torch.equal(result[:, 1], result[:, 2])


def test_metrics_keep_seed_axis_and_episode_balance(tmp_path: Path) -> None:
    split = load_formal_bridge_split(
        _cache(
            tmp_path / "train",
            split="train",
            upstream=_upstream(tmp_path),
            episodes=("a" * 64, "b" * 64),
            duplicate_first=True,
        )
    )
    prediction = np.repeat(split.target_token[:, None], 3, axis=1)
    prediction[:, 0] += 1.0
    prediction[:, 1] += 2.0
    prediction[:, 2] += 3.0
    metrics = formal_token_metrics(prediction, split)
    assert metrics["replicate_count"] == 3
    assert [row["task_macro_mse"] for row in metrics["replicates"]] == pytest.approx(
        [1.0, 4.0, 9.0]
    )
    assert set(metrics["replicates"][0]["per_horizon_task_macro_mse"]) == {
        f"H{value}" for value in FORMAL_BRIDGE_HORIZONS
    }

    episode = formal_episode_mse(prediction, split)
    assert episode["mse"].shape == (2, 3)
    assert episode["horizon_mse"].shape == (2, 8, 3)
    assert episode["mse"] == pytest.approx(np.asarray([[1.0, 4.0, 9.0]] * 2))


def test_bootstrap_counts_physical_episodes_not_a3_replicates() -> None:
    candidate_by_seed = np.asarray([[1.0, 1.2, 0.8], [2.0, 2.2, 1.8]])
    reference_by_seed = np.asarray([[2.0, 2.1, 1.9], [4.0, 4.1, 3.9]])
    report = paired_episode_bootstrap(
        candidate_by_seed.mean(axis=1),
        reference_by_seed.mean(axis=1),
        np.asarray(["pick-place-v3", "push-v3"]),
        seed=3,
        replicates=100,
    )
    assert report["episode_count"] == 2
    assert report["a3_replicates_averaged_within_episode"] is True
    assert report["relative_improvement"] == pytest.approx(0.5)


def test_normalization_moments_respect_balanced_row_weights() -> None:
    values = np.asarray([[[0.0]], [[100.0]], [[100.0]]])
    mean, std = _moments(values, row_weights=np.asarray([0.5, 0.25, 0.25]))
    assert mean == pytest.approx([50.0])
    assert std == pytest.approx([50.0])


def test_formal_report_gates_each_seed_horizon_and_episode_ci(tmp_path: Path) -> None:
    split = load_formal_bridge_split(
        _cache(
            tmp_path / "train",
            split="train",
            upstream=_upstream(tmp_path),
            episodes=("a" * 64, "b" * 64),
        )
    )
    split = replace(
        split,
        last_valid_token=split.target_token + 0.30,
        geometry_hold_token=split.target_token + 0.25,
    )
    factual = np.repeat((split.target_token + 0.10)[:, None], 3, axis=1)
    no_action = np.repeat((split.target_token + 0.20)[:, None], 3, axis=1)
    oracle = split.target_token.copy()
    report = _report(
        split,
        factual=factual,
        no_action=no_action,
        oracle=oracle,
        bootstrap_replicates=50,
    )
    assert len(report["per_a3_seed"]) == 3
    assert set(report["per_horizon_episode_statistics"]) == {
        f"H{value}" for value in FORMAL_BRIDGE_HORIZONS
    }
    assert all(report["screen"].values())
    assert (
        report["episode_statistics"]["factual_vs_matched_no_action"]
        ["scientific_unit"]
        == "physical_episode"
    )


def test_formal_checkpoint_consumer_binds_runtime_seed_and_all_shas(tmp_path: Path) -> None:
    upstream = _upstream(tmp_path)
    train = load_formal_bridge_split(
        _cache(
            tmp_path / "train",
            split="train",
            upstream=upstream,
            episodes=("a" * 64,),
        )
    )
    validation = load_formal_bridge_split(
        _cache(
            tmp_path / "validation",
            split="validation",
            upstream=upstream,
            episodes=("b" * 64,),
        )
    )
    models = make_formal_bridge_triplet(initialization_seed=11)
    payload = build_formal_bridge_checkpoint(
        factual=models[0],
        no_action=models[1],
        target_oracle=models[2],
        train=train,
        validation=validation,
        training_config={"steps": 2},
        report={"screen": "synthetic"},
    )
    checkpoint = tmp_path / "bridge.pt"
    torch.save(payload, checkpoint)
    runtime_bridges, runtime_payload = load_anchor_policy_bridges(checkpoint)
    assert set(runtime_bridges) == {"factual", "no_action"}
    assert runtime_payload["compatibility"]["single_seed_runtime"] is True
    selected = upstream["a3_replicates"][1]  # type: ignore[index]
    model, loaded = load_formal_bridge_for_runtime(
        checkpoint,
        bridge_name="factual",
        policy_checkpoint=upstream["p1_policy"]["path"],  # type: ignore[index]
        corrector_checkpoint=upstream["p1_corrector"]["path"],  # type: ignore[index]
        dynamics_checkpoint=selected["checkpoint"]["path"],
        dynamics_seed=selected["seed"],
    )
    assert isinstance(model, torch.nn.Module)
    assert loaded["bridge_semantics"] == FORMAL_BRIDGE_SEMANTICS

    wrong = _file(tmp_path / "wrong_a3.pt", b"wrong")
    with pytest.raises(ValueError, match="not bound"):
        load_formal_bridge_for_runtime(
            checkpoint,
            bridge_name="factual",
            policy_checkpoint=upstream["p1_policy"]["path"],  # type: ignore[index]
            corrector_checkpoint=upstream["p1_corrector"]["path"],  # type: ignore[index]
            dynamics_checkpoint=wrong,
            dynamics_seed=selected["seed"],
        )
