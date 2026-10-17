from __future__ import annotations

import json
import hashlib
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

import fcpg.lewm_counterfactual_v2_structured_cost as structured_cost

from fcpg.lewm_counterfactual_v2 import (
    PROTOCOL,
    TASK_NAMES,
    ActionOnlyProbeInputs,
    DirectProbeInputs,
    PredictedProbeInputs,
    canonical_sha256,
    physical_episode_sha256,
)
from fcpg.lewm_counterfactual_v2_scorer import (
    DEFAULT_TINY_PARAMETER_BUDGET,
    assemble_pooled_clean_training_data,
    assemble_grouped_scorer_data,
    audit_scorer_experiment_result,
    build_fixed_scorer,
    candidate_feature_array,
    cyclic_candidate_shuffled_predicted_inputs,
    episode_task_macro_bootstrap,
    fit_feature_scaler,
    fit_task_label_normalizer,
    paired_episode_task_macro_bootstrap,
    pairwise_ranking_huber_loss,
    pairwise_targets,
    parameter_match_report,
    publish_scorer_directory_exclusive,
    score_validation_split,
    shuffled_predicted_inputs,
    task_onehot_feature_indices,
    train_full_batch_candidate_scorer,
    validate_scorer_experiment_config,
    validate_source_result_sha256,
)
from fcpg.lewm_counterfactual_v2_structured_cost import (
    STRUCTURED_PROBES,
    assemble_structured_pooled_training_data,
    bind_shared_structured_scaler,
    cyclic_structured_future,
    fit_structured_feature_scaler,
    score_structured_validation_split,
    structured_inputs_from_probe,
    structured_model_from_config,
    summarize_structured_metrics,
    train_structured_cost_adapter,
    validate_structured_experiment_config,
)
from fcpg.lewm_counterfactual_v2_storage import CounterfactualV2Dataset


ROOT = Path(__file__).resolve().parents[1]


def _formal_dataset(*, mode: str = "full", eligible: bool = True) -> CounterfactualV2Dataset:
    config = json.loads(
        (ROOT / "config/lewm/eval/counterfactual_v2_development_seed8192.json").read_text(
            encoding="utf-8"
        )
    )
    n, c = 4 * 10 * 3, 8
    task_index = np.repeat(np.arange(4, dtype=np.uint8), 10 * 3)
    episode_id = np.tile(np.repeat(np.arange(10, dtype=np.int32), 3), 4)
    phase_index = np.tile(np.arange(3, dtype=np.uint8), 4 * 10)
    env_seed = 30_000 + task_index.astype(np.int64) * 1_000 + episode_id
    candidate_action = np.empty((n, c, 25, 4), dtype=np.float32)
    for candidate in range(c):
        candidate_action[:, candidate] = float(candidate)
    sample_values = np.arange(n, dtype=np.float32).reshape(n, 1, 1, 1, 1, 1)
    z_history = np.broadcast_to(sample_values, (n, 2, 2, 3, 3, 192))
    z_pred_sample = np.arange(n, dtype=np.float32).reshape(n, 1, 1, 1, 1, 1, 1)
    z_pred_future = np.broadcast_to(z_pred_sample, (n, 2, 2, 3, c, 5, 192))
    deployment = {
        "task_index": task_index,
        "history_rgb": np.zeros((n, 1), dtype=np.uint8),
        "history_proprio": np.zeros((n, 1), dtype=np.float32),
        "history_keypoints_xyz": np.zeros((n, 1), dtype=np.float32),
        "history_q_nominal": np.zeros((n, 1), dtype=np.float32),
        "history_valid_nominal": np.ones((n, 1), dtype=bool),
        "history_q_corrected": np.zeros((n, 1), dtype=np.float32),
        "history_valid_grounded": np.ones((n, 1), dtype=bool),
        "history_action_raw": np.zeros((n, 2, 5, 4), dtype=np.float32),
        "candidate_action_raw": candidate_action,
        "z_history": z_history,
        "z_pred_future": z_pred_future,
        "goal_visual": np.zeros((n, 3, 48), dtype=np.float32),
        "task_onehot": np.eye(4, dtype=np.float32)[task_index],
    }
    actual_return = np.empty((n, c, 5), dtype=np.float32)
    for sample in range(n):
        for candidate in range(c):
            actual_return[sample, candidate] = (
                10.0 * float(task_index[sample])
                + 0.1 * float(episode_id[sample])
                + float(candidate)
                + 0.01 * np.arange(5, dtype=np.float32)
            )
    oracle_sample = np.arange(n, dtype=np.float32).reshape(n, 1, 1, 1, 1)
    offline = {
        "actual_return": actual_return,
        "actual_progress": actual_return / 100.0,
        "actual_success": actual_return > 20.0,
        "oracle_future_latent": np.broadcast_to(
            oracle_sample, (n, 3, c, 5, 192)
        ),
        "history_q_true": np.zeros((n, 1), dtype=np.float32),
        "history_valid_true": np.ones((n, 1), dtype=bool),
        "goal_world": np.zeros((n, 3), dtype=np.float32),
        "goal_pixel": np.zeros((n, 3, 2), dtype=np.float32),
        "goal_valid": np.ones((n, 3), dtype=bool),
    }
    audit = {
        "episode_id": episode_id,
        "env_seed": env_seed,
        "phase_index": phase_index,
        "snapshot_step": np.take([10, 25, 35], phase_index).astype(np.int16),
        "camera_intrinsic": np.zeros((n, 1), dtype=np.float32),
        "true_extrinsic": np.zeros((n, 1), dtype=np.float32),
        "nominal_extrinsic": np.zeros((n, 1), dtype=np.float32),
        "endpoint_rgb_dynamic_range": np.ones((n, 1), dtype=np.uint16),
        "endpoint_rgb_changed_pixels": np.ones((n, 1), dtype=np.uint32),
        "endpoint_rgb_sha256": np.zeros((n, 1), dtype=np.uint8),
    }
    rows = []
    for sample in range(n):
        episode = int(episode_id[sample])
        rows.append(
            {
                "sample_index": sample,
                "task_index": int(task_index[sample]),
                "episode_id": episode,
                "env_seed": int(env_seed[sample]),
                "phase_index": int(phase_index[sample]),
                "split": "train" if episode < 7 else "validation",
                "physical_episode_sha256": physical_episode_sha256(
                    int(task_index[sample]), episode, int(env_seed[sample])
                ),
                "candidate_names": [
                    "expert",
                    "policy",
                    "expert_noise_0.05",
                    "expert_noise_0.15",
                    "expert_noise_0.3",
                    "same_task_shuffled",
                    "random_0",
                    "random_1",
                ],
            }
        )
    config_sha = canonical_sha256(config)
    g0 = {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "mode": mode,
        "config_sha256": config_sha,
        "g0_audit": {
            "passed": True,
            "g0_eligible": eligible,
            "index_and_provenance": {
                "passed": True,
                "g0_eligible": eligible,
                "git_clean": True,
                "ineligibility_reasons": (
                    [] if eligible else ["upstream_episode_non_overlap_not_auditable"]
                ),
                "upstream_non_overlap": {
                    "status": "verified" if eligible else "not_auditable"
                },
            },
        },
    }
    g0["manifest_sha256"] = canonical_sha256(g0)
    provenance = {
        "scientific_provenance": {"config": config, "config_sha256": config_sha},
        "g0_manifest": g0,
    }
    return CounterfactualV2Dataset(
        root=ROOT,
        deployment_inputs=deployment,
        offline_labels=offline,
        audit_arrays=audit,
        index_rows=tuple(rows),
        provenance=provenance,
        manifest={"result_sha256": "f" * 64},
    )


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _predicted_inputs(
    *, episodes_per_task: int = 2
) -> tuple[PredictedProbeInputs, tuple[dict[str, object], ...]]:
    n = 4 * episodes_per_task * 3
    c = 3
    task = np.repeat(np.arange(4), episodes_per_task * 3)
    episode = np.tile(np.repeat(np.arange(episodes_per_task), 3), 4)
    phase = np.tile(np.arange(3), 4 * episodes_per_task)
    values = np.arange(n, dtype=np.float32).reshape(n, 1, 1, 1)
    inputs = PredictedProbeInputs(
        z_history=_readonly(np.zeros((n, 3, 192), dtype=np.float32)),
        history_action_raw=_readonly(np.zeros((n, 2, 5, 4), dtype=np.float32)),
        candidate_action_raw=_readonly(np.zeros((n, c, 25, 4), dtype=np.float32)),
        goal_visual=_readonly(np.zeros((n, 48), dtype=np.float32)),
        task_onehot=_readonly(np.eye(4, dtype=np.float32)[task]),
        z_pred_future=_readonly(np.broadcast_to(values, (n, c, 5, 192))),
    )
    rows = tuple(
        {
            "task_index": int(task[index]),
            "phase_index": int(phase[index]),
            "physical_episode_sha256": f"task-{task[index]}-episode-{episode[index]}",
        }
        for index in range(n)
    )
    return inputs, rows


def test_formal_assembly_preserves_candidate_groups_and_episode_isolation() -> None:
    data = assemble_grouped_scorer_data(
        _formal_dataset(),
        probe="action_only",
        model="nominal",
        grounding="nominal",
        view="clean",
    )

    assert data.train.labels.shape == (4 * 7 * 3, 8)
    assert data.validation.labels.shape == (4 * 3 * 3, 8)
    assert data.train.inputs.candidate_action_raw.shape[:2] == data.train.labels.shape
    assert data.validation.inputs.candidate_action_raw.shape[:2] == data.validation.labels.shape
    assert not (
        set(data.train.physical_episode_sha256)
        & set(data.validation.physical_episode_sha256)
    )
    np.testing.assert_allclose(
        np.diff(data.train.raw_labels, axis=1),
        np.ones((84, 7), dtype=np.float32),
        atol=2e-6,
    )
    assert all(len(names) == 8 for names in data.train.candidate_names)


def test_label_normalization_is_train_only_and_validation_cannot_leak() -> None:
    first_dataset = _formal_dataset()
    first = assemble_grouped_scorer_data(
        first_dataset,
        probe="action_only",
        model=0,
        grounding=0,
        view=0,
    )
    poisoned_labels = dict(first_dataset.offline_labels)
    poisoned_return = poisoned_labels["actual_return"].copy()
    validation_rows = np.asarray(
        [row["split"] == "validation" for row in first_dataset.index_rows]
    )
    poisoned_return[validation_rows] += 1_000_000.0
    poisoned_labels["actual_return"] = poisoned_return
    second_dataset = CounterfactualV2Dataset(
        root=first_dataset.root,
        deployment_inputs=first_dataset.deployment_inputs,
        offline_labels=poisoned_labels,
        audit_arrays=first_dataset.audit_arrays,
        index_rows=first_dataset.index_rows,
        provenance=first_dataset.provenance,
        manifest=first_dataset.manifest,
    )
    second = assemble_grouped_scorer_data(
        second_dataset,
        probe="action_only",
        model=0,
        grounding=0,
        view=0,
    )

    np.testing.assert_array_equal(first.normalizer.mean, second.normalizer.mean)
    np.testing.assert_array_equal(first.normalizer.scale, second.normalizer.scale)
    np.testing.assert_array_equal(first.train.labels, second.train.labels)
    assert float(np.min(second.validation.labels - first.validation.labels)) > 100_000.0


def test_smoke_and_noneligible_g0_fail_closed() -> None:
    for dataset in (_formal_dataset(mode="smoke"), _formal_dataset(eligible=False)):
        with pytest.raises(ValueError, match="formal|full"):
            assemble_grouped_scorer_data(
                dataset,
                probe="action_only",
                model=0,
                grounding=0,
                view=0,
            )


def test_explicit_external_development_override_is_nonclaimable_and_recorded() -> None:
    data = assemble_grouped_scorer_data(
        _formal_dataset(eligible=False),
        probe="action_only",
        model=0,
        grounding=0,
        view=0,
        allow_ineligible_external_development=True,
    )

    assert data.source_eligibility.g0_eligible is False
    assert data.source_eligibility.external_development_override is True
    assert data.source_eligibility.ineligibility_reasons == (
        "upstream_episode_non_overlap_not_auditable",
    )


def test_feature_builder_excludes_privileged_poison() -> None:
    dataset = _formal_dataset()
    baseline = assemble_grouped_scorer_data(
        dataset, probe="direct", model=0, grounding=0, view=0
    )
    before = candidate_feature_array(baseline.train.inputs)
    labels = dict(dataset.offline_labels)
    labels["history_q_true"] = np.full_like(labels["history_q_true"], 9_999.0)
    labels["goal_world"] = np.full_like(labels["goal_world"], -9_999.0)
    poisoned = CounterfactualV2Dataset(
        root=dataset.root,
        deployment_inputs=dataset.deployment_inputs,
        offline_labels=labels,
        audit_arrays=dataset.audit_arrays,
        index_rows=dataset.index_rows,
        provenance=dataset.provenance,
        manifest=dataset.manifest,
    )
    after_data = assemble_grouped_scorer_data(
        poisoned, probe="direct", model=0, grounding=0, view=0
    )
    np.testing.assert_array_equal(before, candidate_feature_array(after_data.train.inputs))
    with pytest.raises(TypeError, match="exact"):
        candidate_feature_array(  # type: ignore[arg-type]
            dict(after_data.train.inputs.as_mapping(), q_true=np.zeros(1))
        )


def test_shuffled_prediction_is_episode_deranged_and_other_inputs_are_fixed() -> None:
    inputs, rows = _predicted_inputs()
    shuffled = shuffled_predicted_inputs(inputs, rows, seed=123)

    for name in inputs.as_mapping():
        if name != "z_pred_future":
            np.testing.assert_array_equal(
                inputs.as_mapping()[name], shuffled.as_mapping()[name]
            )
    assert not np.array_equal(inputs.z_pred_future, shuffled.z_pred_future)
    source_id = inputs.z_pred_future[:, 0, 0, 0]
    donor_id = shuffled.z_pred_future[:, 0, 0, 0]
    for index, row in enumerate(rows):
        donor_index = int(donor_id[index])
        assert rows[donor_index]["task_index"] == row["task_index"]
        assert rows[donor_index]["phase_index"] == row["phase_index"]
        assert (
            rows[donor_index]["physical_episode_sha256"]
            != row["physical_episode_sha256"]
        )
        assert donor_id[index] != source_id[index]


def test_candidate_feature_dimensions_and_fixed_model_parameter_matching() -> None:
    n, c = 2, 3
    action = ActionOnlyProbeInputs(
        candidate_action_raw=_readonly(np.zeros((n, c, 25, 4), dtype=np.float32)),
        goal_visual=_readonly(np.zeros((n, 48), dtype=np.float32)),
        task_onehot=_readonly(np.zeros((n, 4), dtype=np.float32)),
    )
    direct = DirectProbeInputs(
        z_history=_readonly(np.zeros((n, 3, 192), dtype=np.float32)),
        history_action_raw=_readonly(np.zeros((n, 2, 5, 4), dtype=np.float32)),
        candidate_action_raw=action.candidate_action_raw,
        goal_visual=action.goal_visual,
        task_onehot=action.task_onehot,
    )
    predicted = PredictedProbeInputs(
        **direct.as_mapping(),
        z_pred_future=_readonly(np.zeros((n, c, 5, 192), dtype=np.float32)),
    )
    features = {
        "action_only": candidate_feature_array(action),
        "direct": candidate_feature_array(direct),
        "predicted": candidate_feature_array(predicted),
    }
    assert {name: value.shape[-1] for name, value in features.items()} == {
        "action_only": 152,
        "direct": 768,
        "predicted": 1728,
    }
    scorers = {
        name: build_fixed_scorer(
            name,  # type: ignore[arg-type]
            input_dim=value.shape[-1],
            architecture="tiny",
            parameter_budget=DEFAULT_TINY_PARAMETER_BUDGET,
        )
        for name, value in features.items()
    }
    report = parameter_match_report(scorers, tolerance_fraction=0.01)
    assert report["passed"] is True
    assert report["counts"]["direct"] >= report["counts"]["predicted"]
    for name, model in scorers.items():
        output = model(torch.from_numpy(features[name].copy()))
        assert output.shape == (n, c)


def test_pairwise_masks_ties_and_ranking_huber_loss() -> None:
    labels = np.asarray([[1.0, 1.0, 0.0], [0.0, 2.0, 1.0]], dtype=np.float32)
    pairs = pairwise_targets(labels)
    assert pairs.preference.tolist() == [[0, 1, 1], [-1, -1, 1]]
    assert pairs.ties.tolist() == [[True, False, False], [False, False, False]]

    scores = torch.tensor([[2.0, 2.0, 0.0], [0.0, 2.0, 1.0]], requires_grad=True)
    target = torch.from_numpy(labels)
    result = pairwise_ranking_huber_loss(scores, target)
    assert result.informative_pair_count == 5
    assert result.tie_pair_count == 1
    expected_pairwise = 0.5 * (
        torch.nn.functional.softplus(torch.tensor(-2.0))
        + (
            torch.nn.functional.softplus(torch.tensor(-2.0))
            + 2.0 * torch.nn.functional.softplus(torch.tensor(-1.0))
        )
        / 3.0
    )
    assert torch.allclose(result.pairwise, expected_pairwise)
    assert torch.isfinite(result.total)
    result.total.backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


def test_normalizer_rejects_degenerate_task_labels() -> None:
    labels = np.zeros((8, 3), dtype=np.float32)
    task = np.repeat(np.arange(4), 2)
    with pytest.raises(ValueError, match="degenerate"):
        fit_task_label_normalizer(labels, task)


def test_feature_scaler_is_train_only_and_preserves_task_onehot() -> None:
    dataset = _formal_dataset()
    baseline = assemble_pooled_clean_training_data(
        dataset,
        probe="direct",
        model="fcpg",
        groundings=("nominal", "fcpg"),
        view="clean",
    )
    poisoned_deployment = dict(dataset.deployment_inputs)
    poisoned_history = poisoned_deployment["z_history"].copy()
    validation = np.asarray([row["split"] == "validation" for row in dataset.index_rows])
    poisoned_history[validation] = 1_000_000.0
    poisoned_deployment["z_history"] = poisoned_history
    poisoned = CounterfactualV2Dataset(
        root=dataset.root,
        deployment_inputs=poisoned_deployment,
        offline_labels=dataset.offline_labels,
        audit_arrays=dataset.audit_arrays,
        index_rows=dataset.index_rows,
        provenance=dataset.provenance,
        manifest=dataset.manifest,
    )
    after = assemble_pooled_clean_training_data(
        poisoned,
        probe="direct",
        model="fcpg",
        groundings=("nominal", "fcpg"),
        view="clean",
    )
    np.testing.assert_array_equal(baseline.scaler.mean, after.scaler.mean)
    np.testing.assert_array_equal(baseline.scaler.scale, after.scaler.scale)
    passthrough = baseline.scaler.passthrough_indices
    scaled = baseline.scaler.transform(baseline.features)
    np.testing.assert_array_equal(scaled[..., passthrough], baseline.features[..., passthrough])


def test_fixed_step_full_batch_training_is_deterministic() -> None:
    data = assemble_pooled_clean_training_data(
        _formal_dataset(),
        probe="direct",
        model="fcpg",
        groundings=("nominal", "fcpg"),
        view="clean",
    )
    kwargs = {
        "seed": 8201,
        "steps": 4,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
    }
    first = train_full_batch_candidate_scorer(data, **kwargs)
    second = train_full_batch_candidate_scorer(data, **kwargs)

    assert first.steps == second.steps == 4
    assert first.state_sha256 == second.state_sha256
    assert first.final_loss == pytest.approx(second.final_loss, abs=0.0)
    for name, tensor in first.model.state_dict().items():
        torch.testing.assert_close(tensor, second.model.state_dict()[name], rtol=0, atol=0)


def test_cyclic_candidate_shuffle_changes_only_future_candidate_axis() -> None:
    inputs, _rows = _predicted_inputs()
    future = inputs.z_pred_future.copy()
    for candidate in range(future.shape[1]):
        future[:, candidate] += 100.0 * candidate
    inputs = PredictedProbeInputs(
        **{
            **inputs.as_mapping(),
            "z_pred_future": _readonly(future),
        }
    )
    shuffled = cyclic_candidate_shuffled_predicted_inputs(inputs, offset=1)

    for name, value in inputs.as_mapping().items():
        if name != "z_pred_future":
            np.testing.assert_array_equal(value, shuffled.as_mapping()[name])
    np.testing.assert_array_equal(
        shuffled.z_pred_future,
        np.roll(inputs.z_pred_future, shift=1, axis=1),
    )
    assert not np.array_equal(shuffled.z_pred_future, inputs.z_pred_future)
    with pytest.raises(ValueError, match="identity"):
        cyclic_candidate_shuffled_predicted_inputs(inputs, offset=3)


def _bootstrap_rows(
    value: float, *, state: str = "state", scaler: str = "scaler"
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task in TASK_NAMES:
        for episode in range(2):
            for phase in range(3):
                rows.append(
                    {
                        "task": task,
                        "physical_episode_sha256": f"{task}-{episode}",
                        "phase_index": phase,
                        "view": "rotation5",
                        "seed": 8201,
                        "state_sha256": state,
                        "scaler_sha256": scaler,
                        "spearman": value,
                        "normalized_top1_regret": value,
                    }
                )
    return rows


def test_episode_bootstrap_averages_phase_and_seed_and_has_correct_orientation() -> None:
    left = _bootstrap_rows(0.6)
    right = _bootstrap_rows(0.8)
    right.extend(
        {
            **row,
            "seed": 8202,
        }
        for row in list(right)
    )
    left.extend(
        {
            **row,
            "seed": 8202,
        }
        for row in list(left)
    )
    aggregate = episode_task_macro_bootstrap(
        right, "spearman", bootstrap_samples=50, bootstrap_seed=8299
    )
    assert aggregate["estimate"] == pytest.approx(0.8)
    assert aggregate["resampling_unit"] == "physical_episode_within_task"
    higher = paired_episode_task_macro_bootstrap(
        left,
        right,
        "spearman",
        higher_is_better=True,
        bootstrap_samples=50,
        bootstrap_seed=8299,
        require_same_scorer=True,
    )
    assert higher["estimate"] == pytest.approx(0.2)
    lower = paired_episode_task_macro_bootstrap(
        right,
        left,
        "normalized_top1_regret",
        higher_is_better=False,
        bootstrap_samples=50,
        bootstrap_seed=8299,
        require_same_scorer=True,
    )
    assert lower["estimate"] == pytest.approx(0.2)
    mismatched = [dict(row, state_sha256="other") for row in right]
    with pytest.raises(ValueError, match="same scorer"):
        paired_episode_task_macro_bootstrap(
            left,
            mismatched,
            "spearman",
            higher_is_better=True,
            bootstrap_samples=10,
            bootstrap_seed=8299,
            require_same_scorer=True,
        )
    mismatched_scaler = [dict(row, scaler_sha256="other") for row in right]
    with pytest.raises(ValueError, match="same scorer.*scaler|scaler.*differs"):
        paired_episode_task_macro_bootstrap(
            left,
            mismatched_scaler,
            "spearman",
            higher_is_better=True,
            bootstrap_samples=10,
            bootstrap_seed=8299,
            require_same_scorer=True,
        )


def test_source_hash_and_config_contract_are_fail_closed() -> None:
    dataset = _formal_dataset()
    assert validate_source_result_sha256(dataset, "f" * 64) == "f" * 64
    with pytest.raises(ValueError, match="source result SHA mismatch"):
        validate_source_result_sha256(dataset, "e" * 64)
    config = json.loads(
        (ROOT / "config/lewm/eval/counterfactual_v2_scorer_e1_e3_seed8201.json").read_text(
            encoding="utf-8"
        )
    )
    validate_scorer_experiment_config(config)
    config["optimization"]["steps"] = 1999
    with pytest.raises(ValueError, match="optimization"):
        validate_scorer_experiment_config(config)


def test_result_audit_and_exclusive_atomic_publication(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    payload = staging / "metrics.json"
    payload.write_text('{"passed":true}\n', encoding="utf-8")
    files = {
        "metrics.json": {
            "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            "size_bytes": payload.stat().st_size,
        }
    }
    body: dict[str, object] = {"complete": True, "files": files}
    body["result_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    (staging / "manifest.json").write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    destination = tmp_path / "result"
    publish_scorer_directory_exclusive(staging, destination)
    audited = audit_scorer_experiment_result(destination)
    assert audited["result_sha256"] == body["result_sha256"]

    second = tmp_path / "second"
    second.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        publish_scorer_directory_exclusive(second, destination)
    shutil.rmtree(second)


def _structured_config() -> dict[str, object]:
    return json.loads(
        (
            ROOT
            / "config/lewm/eval/counterfactual_v2_structured_cost_e1_e3_seed8201.json"
        ).read_text(encoding="utf-8")
    )


def test_structured_config_and_exact_parameter_fairness() -> None:
    config = _structured_config()
    validate_structured_experiment_config(config)
    counts = {
        probe: sum(
            parameter.numel()
            for parameter in structured_model_from_config(  # type: ignore[arg-type]
                config["model"]
            ).parameters()
            if parameter.requires_grad
        )
        for probe in STRUCTURED_PROBES
    }
    assert counts == {"direct": 7181, "predicted": 7181, "oracle": 7181}


def test_structured_direct_only_replaces_future_with_repeated_z_last() -> None:
    dataset = _formal_dataset()
    direct_data = assemble_grouped_scorer_data(
        dataset, probe="direct", model="fcpg", grounding="fcpg", view="clean"
    )
    predicted_data = assemble_grouped_scorer_data(
        dataset, probe="predicted", model="fcpg", grounding="fcpg", view="clean"
    )
    direct = structured_inputs_from_probe(direct_data.train.inputs, probe="direct")
    predicted = structured_inputs_from_probe(
        predicted_data.train.inputs, probe="predicted"
    )

    for name in (
        "z_history",
        "history_action_raw",
        "candidate_action_raw",
        "goal_visual",
        "task_onehot",
    ):
        np.testing.assert_array_equal(getattr(direct, name), getattr(predicted, name))
    expected = np.broadcast_to(
        direct.z_history[:, -1, :][:, None, None, :], direct.future_state.shape
    )
    np.testing.assert_array_equal(direct.future_state, expected)

    shuffled = cyclic_structured_future(predicted, offset=1)
    for name in (
        "z_history",
        "history_action_raw",
        "candidate_action_raw",
        "goal_visual",
        "task_onehot",
    ):
        np.testing.assert_array_equal(getattr(predicted, name), getattr(shuffled, name))
    np.testing.assert_array_equal(
        shuffled.future_state, np.roll(predicted.future_state, shift=1, axis=1)
    )


def test_structured_training_is_deterministic_and_reports_both_candidate_sets() -> None:
    dataset = _formal_dataset()
    data = assemble_structured_pooled_training_data(
        dataset,
        probe="predicted",
        model="fcpg",
        groundings=("nominal", "fcpg"),
        view="clean",
    )
    config = _structured_config()
    kwargs = {
        "model_config": config["model"],
        "seed": 8201,
        "steps": 2,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
    }
    first = train_structured_cost_adapter(data, **kwargs)  # type: ignore[arg-type]
    second = train_structured_cost_adapter(data, **kwargs)  # type: ignore[arg-type]
    assert first.parameter_count == second.parameter_count == 7181
    assert first.state_sha256 == second.state_sha256
    assert first.final_loss == pytest.approx(second.final_loss, abs=0.0)

    evaluation = assemble_grouped_scorer_data(
        dataset, probe="predicted", model="fcpg", grounding="fcpg", view="clean"
    )
    rows = score_structured_validation_split(
        first, data.scaler, evaluation, view="clean", grounding="fcpg"
    )
    assert rows
    assert all(
        row["candidate_subset"] == "nonrandom_without_random_0_random_1"
        for row in rows
    )
    assert all(len(row["candidate_names"]) == 6 for row in rows)
    assert all(
        row["strict_near_candidate_subset"] == "expert_plus_three_expert_noise"
        for row in rows
    )
    assert all(len(row["strict_near_candidate_names"]) == 4 for row in rows)


def test_structured_scaler_ignores_disparate_future_distributions() -> None:
    dataset = _formal_dataset()
    pooled = {
        probe: assemble_structured_pooled_training_data(
            dataset,
            probe=probe,
            model="fcpg",
            groundings=("nominal", "fcpg"),
            view="clean",
        )
        for probe in STRUCTURED_PROBES
    }
    shifted = dict(pooled)
    for probe, offset in (("predicted", 1_000.0), ("oracle", -1_000.0)):
        original = pooled[probe]
        inputs = replace(
            original.inputs,
            future_state=_readonly(original.inputs.future_state + offset),
        )
        shifted[probe] = replace(
            original,
            inputs=inputs,
            scaler=fit_structured_feature_scaler(inputs),
        )

    rebound, receipt = bind_shared_structured_scaler(shifted)
    assert receipt["passed"] is True
    assert receipt["future_state_used_for_scaler_fit"] is False
    assert receipt["fit_scope"] == "pooled_clean_training_shared_across_probes"
    assert receipt["latent_fit_source"] == "raw_z_history_only"
    assert receipt["action_block_canonical_sha256"]
    assert receipt["goal_block_canonical_sha256"]
    assert len({id(value.scaler) for value in rebound.values()}) == 1
    transformed = {
        probe: value.scaler.transform(value.inputs)
        for probe, value in rebound.items()
    }
    for field in (
        "z_history",
        "history_action_raw",
        "candidate_action_raw",
        "goal_visual",
        "task_onehot",
    ):
        np.testing.assert_array_equal(
            getattr(transformed["direct"], field),
            getattr(transformed["predicted"], field),
        )
        np.testing.assert_array_equal(
            getattr(transformed["direct"], field),
            getattr(transformed["oracle"], field),
        )
    assert not np.array_equal(
        transformed["predicted"].future_state,
        transformed["oracle"].future_state,
    )


def test_structured_shared_scaler_is_train_only_under_validation_feature_poison() -> None:
    dataset = _formal_dataset()
    validation = np.asarray(
        [row["split"] == "validation" for row in dataset.index_rows], dtype=bool
    )
    deployment = dict(dataset.deployment_inputs)
    for name in (
        "z_history",
        "history_action_raw",
        "candidate_action_raw",
        "z_pred_future",
        "goal_visual",
    ):
        value = np.array(deployment[name], copy=True)
        value[validation] += np.asarray(100_000.0, dtype=value.dtype)
        deployment[name] = value
    offline = dict(dataset.offline_labels)
    oracle = np.array(offline["oracle_future_latent"], copy=True)
    oracle[validation] -= np.asarray(100_000.0, dtype=oracle.dtype)
    offline["oracle_future_latent"] = oracle
    poisoned = CounterfactualV2Dataset(
        root=dataset.root,
        deployment_inputs=deployment,
        offline_labels=offline,
        audit_arrays=dataset.audit_arrays,
        index_rows=dataset.index_rows,
        provenance=dataset.provenance,
        manifest=dataset.manifest,
    )

    def shared_scaler(value: CounterfactualV2Dataset):
        pooled = {
            probe: assemble_structured_pooled_training_data(
                value,
                probe=probe,
                model="fcpg",
                groundings=("nominal", "fcpg"),
                view="clean",
            )
            for probe in STRUCTURED_PROBES
        }
        rebound, receipt = bind_shared_structured_scaler(pooled)
        return rebound["direct"].scaler, receipt

    baseline_scaler, baseline_receipt = shared_scaler(dataset)
    poisoned_scaler, poisoned_receipt = shared_scaler(poisoned)
    assert poisoned_scaler.as_mapping() == baseline_scaler.as_mapping()
    assert (
        poisoned_receipt["shared_scaler_canonical_sha256"]
        == baseline_receipt["shared_scaler_canonical_sha256"]
    )
    assert poisoned_receipt["future_state_used_for_scaler_fit"] is False


def test_structured_probes_share_normalized_labels_tasks_and_sample_indices() -> None:
    grouped = {
        probe: assemble_grouped_scorer_data(
            _formal_dataset(),
            probe=probe,
            model="fcpg",
            grounding="fcpg",
            view="clean",
        )
        for probe in STRUCTURED_PROBES
    }
    reference = grouped["direct"]
    for probe in ("predicted", "oracle"):
        candidate = grouped[probe]
        for split in ("train", "validation"):
            expected = getattr(reference, split)
            actual = getattr(candidate, split)
            np.testing.assert_array_equal(actual.labels, expected.labels)
            np.testing.assert_array_equal(actual.raw_labels, expected.raw_labels)
            np.testing.assert_array_equal(actual.task_index, expected.task_index)
            np.testing.assert_array_equal(
                actual.sample_indices, expected.sample_indices
            )
        np.testing.assert_array_equal(candidate.normalizer.mean, reference.normalizer.mean)
        np.testing.assert_array_equal(
            candidate.normalizer.scale, reference.normalizer.scale
        )


def _structured_summary_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    conditions = (
        ("direct", "clean", "fcpg", "standard", 0.10, 0.80, "direct"),
        ("predicted", "clean", "fcpg", "standard", 0.20, 0.70, "predicted"),
        ("oracle", "clean", "fcpg", "standard", 0.40, 0.50, "oracle"),
        ("predicted", "clean", "nominal", "standard", 0.05, 0.90, "predicted"),
        ("direct", "rotation5", "fcpg", "standard", 0.10, 0.80, "direct"),
        ("predicted", "rotation5", "fcpg", "standard", 0.50, 0.40, "predicted"),
        (
            "predicted",
            "rotation5",
            "fcpg",
            "cyclic_candidate_future",
            0.00,
            0.90,
            "predicted",
        ),
        ("predicted", "rotation5", "nominal", "standard", 0.20, 0.70, "predicted"),
        ("direct", "translation4", "fcpg", "standard", 0.10, 0.80, "direct"),
        ("predicted", "translation4", "fcpg", "standard", 0.50, 0.40, "predicted"),
        (
            "predicted",
            "translation4",
            "fcpg",
            "cyclic_candidate_future",
            0.00,
            0.90,
            "predicted",
        ),
        ("predicted", "translation4", "nominal", "standard", 0.20, 0.70, "predicted"),
    )
    for task in TASK_NAMES:
        for probe, view, grounding, variant, spearman, regret, state in conditions:
            rows.append(
                {
                    "probe": probe,
                    "view": view,
                    "grounding": grounding,
                    "variant": variant,
                    "seed": 8201,
                    "state_sha256": state,
                    "scaler_sha256": "shared-scaler",
                    "task": task,
                    "physical_episode_sha256": f"{task}-episode",
                    "phase_index": 1,
                    "spearman": spearman,
                    "normalized_top1_regret": regret,
                    "strict_near_spearman": spearman,
                    "strict_near_normalized_top1_regret": regret,
                }
            )
    return rows


def test_structured_e1_e3_orientation_and_same_scorer_contract() -> None:
    rows = _structured_summary_rows()
    summary = summarize_structured_metrics(
        rows, bootstrap_samples=20, bootstrap_seed=8299
    )
    comparisons = summary["comparisons"]
    for name in (
        "E1_oracle_vs_direct_clean_fcpg",
        "E2_predicted_vs_direct_drift_fcpg",
        "E2_predicted_vs_cyclic_shuffle_drift_fcpg",
        "E3_fcpg_vs_nominal_grounding_drift",
    ):
        assert comparisons[name]["spearman"]["estimate"] > 0.0
        assert comparisons[name]["normalized_top1_regret"]["estimate"] > 0.0
        assert (
            comparisons[name]["spearman"]["orientation"]
            == "positive_means_right_better"
        )
        assert comparisons[name]["spearman"]["raw_delta"] == "right_minus_left"
        assert (
            comparisons[name]["normalized_top1_regret"]["raw_delta"]
            == "left_minus_right"
        )

    for field in ("state_sha256", "scaler_sha256"):
        mismatched = [dict(row) for row in rows]
        for row in mismatched:
            if (
                row["probe"] == "predicted"
                and row["grounding"] == "nominal"
                and row["view"] in {"rotation5", "translation4"}
            ):
                row[field] = "mismatched"
        with pytest.raises(ValueError, match="same scorer|same-scorer"):
            summarize_structured_metrics(
                mismatched, bootstrap_samples=10, bootstrap_seed=8299
            )


def test_structured_atomic_publication_and_audit_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _structured_config()
    config["source"] = {
        "directory": str(tmp_path / "synthetic-source"),
        "result_sha256": "f" * 64,
    }
    config["optimization"] = {
        **config["optimization"],
        "steps": 1,
        "seeds": [8201],
        "device": "cpu",
    }
    config["evaluation"] = {
        **config["evaluation"],
        "bootstrap_samples": 8,
    }
    config["output"] = {
        **config["output"],
        "directory": str(tmp_path / "published"),
    }
    monkeypatch.setattr(
        structured_cost, "validate_structured_experiment_config", lambda _config: None
    )
    monkeypatch.setattr(
        structured_cost,
        "load_counterfactual_v2_store",
        lambda _path: _formal_dataset(),
    )
    monkeypatch.setattr(
        structured_cost,
        "_git_provenance",
        lambda _root: {"commit": "a" * 40, "dirty": False, "status_sha256": "b" * 64},
    )

    published = structured_cost.run_structured_cost_experiment(
        config,
        repo_root=tmp_path,
        confirm_external_development=True,
    )
    audited = audit_scorer_experiment_result(tmp_path / "published")
    assert audited == published
    assert audited["complete"] is True
    assert audited["parameter_fairness"]["passed"] is True
    assert (tmp_path / "published" / "scalers" / "shared.json").is_file()
    with pytest.raises(FileExistsError, match="exists"):
        structured_cost.run_structured_cost_experiment(
            config,
            repo_root=tmp_path,
            confirm_external_development=True,
        )


def test_flat_scorer_candidate_subset_name_is_not_misleading() -> None:
    dataset = _formal_dataset()
    pooled = assemble_pooled_clean_training_data(
        dataset,
        probe="direct",
        model="fcpg",
        groundings=("nominal", "fcpg"),
        view="clean",
    )
    trained = train_full_batch_candidate_scorer(
        pooled,
        seed=8201,
        steps=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        gradient_clip_norm=1.0,
    )
    evaluation = assemble_grouped_scorer_data(
        dataset, probe="direct", model="fcpg", grounding="fcpg", view="clean"
    )
    rows = score_validation_split(
        trained, pooled.scaler, evaluation, view="clean", grounding="fcpg"
    )
    assert rows
    assert all(
        row["candidate_subset"] == "nonrandom_without_random_0_random_1"
        for row in rows
    )
    assert all(len(row["candidate_names"]) == 6 for row in rows)
