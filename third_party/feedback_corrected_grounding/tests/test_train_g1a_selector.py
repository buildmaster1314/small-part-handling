from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.train_g1a_selector import (
    DEPLOYMENT_ALLOWED_KEYS,
    OFFLINE_ALLOWED_KEYS,
    EquivariantG1ASelector,
    EXPECTED_CACHE_SCHEMA_VERSION,
    EXPECTED_CACHE_TYPE,
    SelectorCache,
    baseline_costs,
    evaluate_costs,
    fit_normalizer,
    load_selector_cache,
    listwise_kl_loss,
    paired_episode_bootstrap,
    parameter_count,
    per_policy_oracle_report,
    soft_label_distribution,
    train_selector,
    validate_selector_cache,
)


def _synthetic_partitions(
    *, samples: int = 36, candidates: int = 4, latent: int = 8
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(41)
    if samples % 12:
        raise ValueError("synthetic sample count must be divisible by 12")
    episodes = np.repeat(np.arange(12), samples // 12)
    splits = np.asarray(
        [
            "train"
            if value < 4
            else "validation"
            if value < 8
            else "test"
            for value in episodes
        ]
    )
    task = np.asarray(["pick-place-v3" if value % 2 == 0 else "button-press-v3" for value in episodes])
    frame = np.tile(np.arange(samples // 12), 12) + 20
    raw = np.arange(samples)
    anchor_count = 3
    history = 3
    geometry_dim = 5
    action_dim = 4
    valid = np.ones((samples, candidates, anchor_count), dtype=bool)
    valid[0, -1] = False
    geometry = rng.normal(
        size=(samples, candidates, anchor_count, geometry_dim)
    ).astype(np.float32)
    anchor = rng.normal(size=(samples, candidates, anchor_count, latent)).astype(np.float32)
    summary = rng.normal(size=(samples, candidates, latent)).astype(np.float32)
    visual_score = (
        -geometry[..., 0].mean(axis=-1)
        + 0.05 * rng.normal(size=(samples, candidates))
    ).astype(np.float32)
    deployment = {
        "task_name": task,
        "episode_id": episodes.astype(np.int64),
        "frame_id": frame.astype(np.int64),
        "raw_index": raw.astype(np.int64),
        "split": splits,
        "candidate_points_px": rng.normal(size=(samples, candidates, anchor_count, 2)).astype(np.float32),
        "candidate_valid": valid,
        "candidate_grid_indices": np.zeros((samples, candidates, anchor_count), dtype=np.int64),
        "candidate_joint_visual_score": visual_score,
        "candidate_visual_descriptor": rng.normal(size=(samples, candidates, anchor_count, 6)).astype(np.float32),
        "candidate_geometry": geometry,
        "candidate_summary_token": summary.copy(),
        "candidate_summary_embedding": summary,
        "candidate_anchor_tokens": anchor,
        "history_raw_indices": np.zeros((samples, history), dtype=np.int64),
        "history_frame_ids": np.zeros((samples, history), dtype=np.int64),
        "history_grounding_points_px": np.zeros((samples, history, anchor_count, 2), dtype=np.float32),
        "history_grounding_valid": np.ones((samples, history, anchor_count), dtype=bool),
        "history_action_blocks": rng.normal(size=(samples, history, action_dim)).astype(np.float32),
        "history_summary_token": rng.normal(size=(samples, history, latent)).astype(np.float32),
        "history_summary_embedding": rng.normal(size=(samples, history, latent)).astype(np.float32),
        "history_anchor_tokens": rng.normal(size=(samples, history, anchor_count, latent)).astype(np.float32),
        "predicted_current_summary_prior": rng.normal(size=(samples, latent)).astype(np.float32),
    }
    source = np.empty(samples, dtype=np.int64)
    for split_name in np.unique(splits):
        for task_name in np.unique(task):
            for frame_id in np.unique(frame):
                matched = np.flatnonzero(
                    (splits == split_name)
                    & (task == task_name)
                    & (frame == frame_id)
                )
                if len(matched):
                    source[matched] = np.roll(matched, 1)
    base_cost = geometry[..., 0].mean(axis=-1) - 0.2 * summary[..., 0]
    labels = np.stack(
        [base_cost - base_cost.min(axis=1, keepdims=True) + offset for offset in (0.0, 0.01, 0.02)],
        axis=1,
    ).astype(np.float32)
    offline = {
        "task_name": task.copy(),
        "episode_id": episodes.astype(np.int64),
        "frame_id": frame.astype(np.int64),
        "raw_index": raw.astype(np.int64),
        "policy_seed": np.asarray([0, 1, 2], dtype=np.int64),
        "action_xyz_normalized_mse_h8": labels,
        "shuffled_current_action_blocks": deployment["history_action_blocks"][
            source, -1
        ].copy(),
        "shuffled_action_source_index": source,
        "shuffled_predicted_current_summary_prior": deployment[
            "predicted_current_summary_prior"
        ][source].copy(),
    }
    assert set(deployment).issubset(DEPLOYMENT_ALLOWED_KEYS)
    assert set(offline).issubset(OFFLINE_ALLOWED_KEYS)
    return deployment, offline


def _cache() -> SelectorCache:
    return validate_selector_cache(*_synthetic_partitions())


def _model(mode: str) -> EquivariantG1ASelector:
    return EquivariantG1ASelector(
        mode=mode,
        anchor_count=3,
        latent_dim=8,
        geometry_dim=5,
        action_block_dim=4,
        model_dim=4,
        hidden_dim=8,
    )


def _inputs(cache: SelectorCache, model: EquivariantG1ASelector) -> dict[str, torch.Tensor]:
    return {
        key: torch.from_numpy(np.asarray(cache.deployment[key], dtype=np.float32).copy())
        for key in model.required_keys
    }


def test_cache_contract_and_physical_episode_split() -> None:
    cache = _cache()
    assert cache.policy_seeds == (0, 1, 2)
    assert cache.candidate_count == 4
    assert cache.anchor_count == 3
    deployment, offline = _synthetic_partitions()
    deployment["split"] = deployment["split"].copy()
    deployment["split"][1] = "validation"
    with pytest.raises(ValueError, match="episode crosses"):
        validate_selector_cache(deployment, offline)


def test_cache_rejects_nonwhitelisted_or_privileged_key() -> None:
    deployment, offline = _synthetic_partitions()
    deployment["q_true"] = np.zeros((len(deployment["split"]), 3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="non-whitelisted"):
        validate_selector_cache(deployment, offline)


def test_cache_rejects_identity_mismatch_and_bad_label_axes() -> None:
    deployment, offline = _synthetic_partitions()
    offline["frame_id"] = offline["frame_id"].copy()
    offline["frame_id"][0] += 1
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_selector_cache(deployment, offline)
    deployment, offline = _synthetic_partitions()
    offline["action_xyz_normalized_mse_h8"] = offline[
        "action_xyz_normalized_mse_h8"
    ][:, :2]
    with pytest.raises(ValueError, match=r"\[sample,policy,candidate\]"):
        validate_selector_cache(deployment, offline)


@pytest.mark.parametrize(
    ("mutation", "error_type", "message"),
    [
        ("shape", ValueError, r"shape \[sample,action_block_dim\]"),
        ("dtype", TypeError, "floating dtype"),
        ("history_dtype", TypeError, "dtype must match"),
        ("nonfinite", ValueError, "must be finite"),
    ],
)
def test_cache_rejects_invalid_shuffled_action_payload(
    mutation: str, error_type: type[Exception], message: str
) -> None:
    deployment, offline = _synthetic_partitions()
    if mutation == "shape":
        offline["shuffled_current_action_blocks"] = offline[
            "shuffled_current_action_blocks"
        ][:, :-1]
    elif mutation == "dtype":
        offline["shuffled_current_action_blocks"] = offline[
            "shuffled_current_action_blocks"
        ].astype(np.int64)
    elif mutation == "history_dtype":
        deployment["history_action_blocks"] = deployment[
            "history_action_blocks"
        ].astype(np.float64)
    else:
        offline["shuffled_current_action_blocks"] = offline[
            "shuffled_current_action_blocks"
        ].copy()
        offline["shuffled_current_action_blocks"][0, 0] = np.nan
    with pytest.raises(error_type, match=message):
        validate_selector_cache(deployment, offline)


@pytest.mark.parametrize(
    ("mutation", "error_type", "message"),
    [
        ("shape", ValueError, r"shape \[sample\]"),
        ("dtype", TypeError, "integer dtype"),
        ("range", ValueError, "outside the cache"),
        ("duplicate", ValueError, "must be a permutation"),
        ("fixed", ValueError, "must be a derangement"),
        ("support", ValueError, "preserve task and frame"),
        ("split", ValueError, "within its data split"),
        ("episode", ValueError, "another physical episode"),
        ("payload", ValueError, "exactly match source"),
    ],
)
def test_cache_rejects_invalid_shuffled_action_source_contract(
    mutation: str, error_type: type[Exception], message: str
) -> None:
    deployment, offline = _synthetic_partitions()
    source = offline["shuffled_action_source_index"].copy()
    if mutation == "shape":
        offline["shuffled_action_source_index"] = source[:, None]
    elif mutation == "dtype":
        offline["shuffled_action_source_index"] = source.astype(np.float32)
    elif mutation == "range":
        source[0] = len(source)
        offline["shuffled_action_source_index"] = source
    elif mutation == "duplicate":
        source[0] = source[1]
        offline["shuffled_action_source_index"] = source
    elif mutation == "fixed":
        # Swap the two source entries so the mapping stays bijective but gains
        # fixed points.
        zero_source = int(source[0])
        source[0], source[zero_source] = source[zero_source], source[0]
        offline["shuffled_action_source_index"] = source
    elif mutation == "support":
        source = np.roll(np.arange(len(source)), 1)
        offline["shuffled_action_source_index"] = source
    elif mutation == "split":
        left = int(
            np.flatnonzero(
                (deployment["split"] == "train")
                & (deployment["task_name"] == "pick-place-v3")
                & (deployment["frame_id"] == 20)
            )[0]
        )
        right = int(
            np.flatnonzero(
                (deployment["split"] == "validation")
                & (deployment["task_name"] == "pick-place-v3")
                & (deployment["frame_id"] == 20)
            )[0]
        )
        source[left], source[right] = source[right], source[left]
        offline["shuffled_action_source_index"] = source
    elif mutation == "episode":
        # Give each episode repeated matched-frame rows, then rotate strictly
        # within that episode.  This preserves task/frame and remains a
        # bijective derangement, isolating the cross-episode requirement.
        deployment["frame_id"] = np.full_like(deployment["frame_id"], 20)
        offline["frame_id"] = deployment["frame_id"].copy()
        source = np.arange(len(source))
        for episode_id in np.unique(deployment["episode_id"]):
            matched = np.flatnonzero(deployment["episode_id"] == episode_id)
            source[matched] = np.roll(matched, 1)
        offline["shuffled_action_source_index"] = source
    else:
        offline["shuffled_current_action_blocks"] = offline[
            "shuffled_current_action_blocks"
        ].copy()
        offline["shuffled_current_action_blocks"][0, 0] += 1.0
    with pytest.raises(error_type, match=message):
        validate_selector_cache(deployment, offline)


def test_cache_accepts_complete_history_diagnostics_and_rejects_partial() -> None:
    deployment, offline = _synthetic_partitions()
    stems = (
        "history_grounding_points_px",
        "history_grounding_valid",
        "history_summary_token",
        "history_summary_embedding",
        "history_anchor_tokens",
        "predicted_current_summary_prior",
    )
    for suffix in ("wide_top1", "nominal"):
        for stem in stems:
            deployment[f"{stem}_{suffix}"] = np.asarray(deployment[stem]).copy()
        offline[f"shuffled_predicted_current_summary_prior_{suffix}"] = np.asarray(
            offline["shuffled_predicted_current_summary_prior"]
        ).copy()
    cache = validate_selector_cache(deployment, offline)
    assert cache.sample_count == 36

    deployment.pop("history_anchor_tokens_nominal")
    with pytest.raises(KeyError, match="partial nominal"):
        validate_selector_cache(deployment, offline)


def test_cache_rejects_nonboolean_variant_history_validity() -> None:
    deployment, offline = _synthetic_partitions()
    stems = (
        "history_grounding_points_px",
        "history_grounding_valid",
        "history_summary_token",
        "history_summary_embedding",
        "history_anchor_tokens",
        "predicted_current_summary_prior",
    )
    for stem in stems:
        deployment[f"{stem}_nominal"] = np.asarray(deployment[stem]).copy()
    deployment["history_grounding_valid_nominal"] = deployment[
        "history_grounding_valid_nominal"
    ].astype(np.float32)
    offline["shuffled_predicted_current_summary_prior_nominal"] = np.asarray(
        offline["shuffled_predicted_current_summary_prior"]
    ).copy()
    with pytest.raises(TypeError, match="validity must be boolean"):
        validate_selector_cache(deployment, offline)


def _write_synthetic_cache(tmp_path: Path, contract: dict[str, object]) -> Path:
    deployment, offline = _synthetic_partitions()
    np.savez(tmp_path / "deployment_inputs.npz", **deployment)
    np.savez(tmp_path / "offline_labels.npz", **offline)
    metadata = {
        "schema_version": EXPECTED_CACHE_SCHEMA_VERSION,
        "cache_type": EXPECTED_CACHE_TYPE,
        "sample_count": len(deployment["task_name"]),
        "candidate_count": deployment["candidate_valid"].shape[1],
        "policy_seeds": offline["policy_seed"].tolist(),
        "contract": contract,
        "artifacts": {},
    }
    (tmp_path / "metadata.json").write_text(
        json.dumps(metadata) + "\n", encoding="utf-8"
    )
    return tmp_path


def _valid_metadata_contract() -> dict[str, object]:
    return {
        "history_grounding_primary": "native_fcpg",
        "label_history_source": "native_fcpg",
        "label_history_matches_primary": True,
        "history_shuffled_prior_partition": "offline_labels_only",
        "history_shuffled_source_scope": (
            "same_split_task_frame_different_physical_episode"
        ),
    }


def test_cache_loader_fails_closed_on_schema_type_and_label_history_contract(
    tmp_path: Path,
) -> None:
    cache_dir = _write_synthetic_cache(tmp_path, _valid_metadata_contract())
    assert load_selector_cache(cache_dir).sample_count == 36

    metadata_path = cache_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["schema_version"] += 1
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_selector_cache(cache_dir)

    metadata["schema_version"] = EXPECTED_CACHE_SCHEMA_VERSION
    metadata["cache_type"] = "stale_cache"
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cache_type"):
        load_selector_cache(cache_dir)

    metadata["cache_type"] = EXPECTED_CACHE_TYPE
    metadata["contract"]["label_history_source"] = "wide_top1"
    metadata["contract"]["label_history_matches_primary"] = False
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="label history differs"):
        load_selector_cache(cache_dir)


@pytest.mark.parametrize("failure", ["shape", "nonfinite"])
def test_cache_rejects_invalid_variant_shuffled_prior(failure: str) -> None:
    deployment, offline = _synthetic_partitions()
    stems = (
        "history_grounding_points_px",
        "history_grounding_valid",
        "history_summary_token",
        "history_summary_embedding",
        "history_anchor_tokens",
        "predicted_current_summary_prior",
    )
    for stem in stems:
        deployment[f"{stem}_wide_top1"] = np.asarray(deployment[stem]).copy()
    shuffled_key = "shuffled_predicted_current_summary_prior_wide_top1"
    offline[shuffled_key] = np.asarray(
        offline["shuffled_predicted_current_summary_prior"]
    ).copy()
    if failure == "shape":
        offline[shuffled_key] = offline[shuffled_key][:, :-1]
    else:
        offline[shuffled_key][0, 0] = np.nan
    with pytest.raises(ValueError, match="predicted-prior axes|non-finite"):
        validate_selector_cache(deployment, offline)


@pytest.mark.parametrize(
    "mode",
    ["structured_full", "structured_current_only", "summary_predictive"],
)
def test_selector_is_candidate_permutation_equivariant(mode: str) -> None:
    torch.manual_seed(5)
    cache = _cache()
    model = _model(mode).eval()
    inputs = _inputs(cache, model)
    valid = torch.from_numpy(
        np.all(np.asarray(cache.deployment["candidate_valid"]), axis=-1).copy()
    )
    permutation = torch.tensor([2, 0, 3, 1])
    candidate_keys = {
        "candidate_anchor_tokens",
        "candidate_geometry",
        "candidate_joint_visual_score",
        "candidate_summary_embedding",
    }
    permuted = {
        key: value[:, permutation] if key in candidate_keys else value
        for key, value in inputs.items()
    }
    with torch.no_grad():
        original = model(inputs, valid)
        changed = model(permuted, valid[:, permutation])
    torch.testing.assert_close(changed, original[:, permutation], rtol=1e-6, atol=1e-6)


def test_modes_are_parameter_matched_and_inputs_are_strict() -> None:
    models = [_model(mode) for mode in ("structured_full", "structured_current_only", "summary_predictive")]
    assert len({parameter_count(model) for model in models}) == 1
    cache = _cache()
    current = models[1]
    inputs = _inputs(cache, current)
    inputs["predicted_current_summary_prior"] = torch.zeros(
        cache.sample_count, 8
    )
    valid = torch.from_numpy(
        np.all(np.asarray(cache.deployment["candidate_valid"]), axis=-1).copy()
    )
    with pytest.raises(ValueError, match="input contract changed"):
        current(inputs, valid)


def test_soft_targets_and_listwise_loss_ignore_invalid_candidates() -> None:
    labels = torch.tensor([[[0.0, 1.0, 20.0], [0.0, 2.0, 30.0], [0.0, 3.0, 40.0]]])
    valid = torch.tensor([[True, True, False]])
    target = soft_label_distribution(labels, valid, temperature=0.25)
    assert target.shape == (1, 3)
    assert target[0, 2] == 0
    assert target[0, 0] > target[0, 1]
    costs = torch.tensor([[0.2, 0.8, torch.inf]], requires_grad=True)
    loss = listwise_kl_loss(costs, target, valid, temperature=1.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(costs.grad[:, :2]).all()


def test_train_selector_selects_validation_only_and_never_uses_test_labels() -> None:
    cache = _cache()
    trained = train_selector(
        cache,
        mode="structured_current_only",
        device=torch.device("cpu"),
        seed=7,
        epochs=3,
        batch_size=4,
        model_dim=4,
        hidden_dim=8,
    )
    labels = np.asarray(cache.offline["action_xyz_normalized_mse_h8"]).copy()
    test = np.asarray(cache.deployment["split"]).astype(str) == "test"
    labels[test] = labels[test, :, ::-1]
    poisoned = replace(cache, offline={**cache.offline, "action_xyz_normalized_mse_h8": labels})
    repeated = train_selector(
        poisoned,
        mode="structured_current_only",
        device=torch.device("cpu"),
        seed=7,
        epochs=3,
        batch_size=4,
        model_dim=4,
        hidden_dim=8,
    )
    assert trained.best_epoch == repeated.best_epoch
    assert trained.best_validation_kl == pytest.approx(repeated.best_validation_kl)
    for key, value in trained.model.state_dict().items():
        torch.testing.assert_close(value, repeated.model.state_dict()[key])


def test_metrics_report_every_task_and_policy_seed() -> None:
    cache = _cache()
    test = np.flatnonzero(np.asarray(cache.deployment["split"]).astype(str) == "test")
    visual = baseline_costs(cache, test)["frozen_visual_score"]
    report = evaluate_costs(visual, cache, test)
    assert set(report["by_policy_seed"]) == {"0", "1", "2"}
    assert set(report["by_task_and_policy_seed"]) == {
        "button-press-v3",
        "pick-place-v3",
    }
    assert report["macro"]["count"] == len(test) * 3
    assert 0.0 <= report["macro"]["mean_normalized_top1_regret"] <= 1.0


def test_per_policy_oracle_count_is_physical_snapshot_count() -> None:
    cache = _cache()
    test = np.flatnonzero(
        np.asarray(cache.deployment["split"]).astype(str) == "test"
    )
    report = per_policy_oracle_report(cache, test)
    for metrics in report.values():
        assert metrics["count"] == len(test)
        assert metrics["mean_spearman"] == pytest.approx(1.0)
        assert metrics["mean_normalized_top1_regret"] == pytest.approx(0.0)


def test_paired_bootstrap_clusters_physical_episodes_and_uses_one_direction() -> None:
    cache = _cache()
    test = np.flatnonzero(
        np.asarray(cache.deployment["split"]).astype(str) == "test"
    )
    labels = np.asarray(
        cache.offline["action_xyz_normalized_mse_h8"], dtype=np.float64
    )[test]
    first = labels.mean(axis=1)
    second = -first
    report = paired_episode_bootstrap(
        first,
        second,
        cache,
        test,
        first_name="first",
        second_name="second",
        resamples=500,
        seed=17,
    )
    repeated = paired_episode_bootstrap(
        first,
        second,
        cache,
        test,
        first_name="first",
        second_name="second",
        resamples=500,
        seed=17,
    )

    first_metrics = evaluate_costs(first, cache, test)["macro"]
    second_metrics = evaluate_costs(second, cache, test)["macro"]
    assert report == repeated
    assert report["physical_episode_count"] == 4
    assert "task+physical_episode" in report["cluster_unit"]
    assert report["spearman_delta"]["point_estimate"] == pytest.approx(
        first_metrics["mean_spearman"] - second_metrics["mean_spearman"]
    )
    assert report["normalized_top1_regret_improvement"][
        "point_estimate"
    ] == pytest.approx(
        second_metrics["mean_normalized_top1_regret"]
        - first_metrics["mean_normalized_top1_regret"]
    )
    assert report["spearman_delta"]["point_estimate"] > 0
    assert report["normalized_top1_regret_improvement"]["point_estimate"] > 0
    assert "positive favors first" in report["sign_convention"]


def test_normalizer_is_fit_only_on_training_rows() -> None:
    cache = _cache()
    train = np.flatnonzero(np.asarray(cache.deployment["split"]).astype(str) == "train")
    normalizer = fit_normalizer(cache.deployment, train, "summary_predictive")
    expected = np.mean(
        np.asarray(cache.deployment["candidate_summary_embedding"])[train],
        axis=(0, 1),
        keepdims=True,
    )
    np.testing.assert_allclose(
        normalizer["candidate_summary_embedding"].mean, expected, rtol=1e-6, atol=1e-6
    )


def test_real_development_cache_loads_and_forwards_when_available() -> None:
    cache_dir = Path(
        "outputs/g1a_selector_cache/"
        "pick_rot10_r64g17_l16m16_online10k_matched_history_native_labels_"
        "splitlocal_v3_development"
    )
    if not (cache_dir / "deployment_inputs.npz").is_file():
        pytest.skip("ignored real G1A cache is unavailable")
    cache = load_selector_cache(cache_dir)
    indices = np.asarray([0, 1], dtype=np.int64)
    valid = torch.from_numpy(
        np.all(np.asarray(cache.deployment["candidate_valid"])[indices], axis=-1).copy()
    )
    counts: set[int] = set()
    for mode in ("structured_full", "structured_current_only", "summary_predictive"):
        model = EquivariantG1ASelector(
            mode=mode,
            anchor_count=cache.anchor_count,
            latent_dim=cache.latent_dim,
            geometry_dim=cache.geometry_dim,
            action_block_dim=cache.action_block_dim,
        ).eval()
        counts.add(parameter_count(model))
        inputs = {
            key: torch.from_numpy(
                np.asarray(cache.deployment[key], dtype=np.float32)[indices].copy()
            )
            for key in model.required_keys
        }
        with torch.no_grad():
            costs = model(inputs, valid)
        assert costs.shape == (2, cache.candidate_count)
        assert torch.isfinite(costs[valid]).all()
    assert len(counts) == 1
