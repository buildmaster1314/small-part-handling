from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest
import torch

import fcpg.lewm_counterfactual_v3_gate as v3_gate
from fcpg.lewm_counterfactual_runtime import sha256_array
from fcpg.lewm_counterfactual_v3 import PHASE_NAMES, TASK_NAMES
from fcpg.lewm_counterfactual_v3_gate import (
    DIRECT_TRAINING_SEEDS,
    DirectRankerInputs,
    PrivilegedDirectRanker,
    StopGate,
    candidate_physical_score_spread_audit,
    candidate_return_spread_audit,
    deterministic_state_oracle_score_matrix,
    direct_ranker_inputs,
    direct_training_eligibility,
    evaluate_state_oracle_gate,
    metric_rows,
    paired_episode_within_task_bootstrap,
    prepare_gate_bundle,
    shortcut_lookup_audits,
    snapshot_ranking_metrics,
    train_direct_ranker_ensemble,
    validate_physical_score_contract,
)
from fcpg.lewm_counterfactual_v2_scorer import scorer_state_sha256
from fcpg.lewm_counterfactual_v3_state import (
    LEFT_CONTACT_INDEX,
    OBJECT_INITIAL_POSITION_SLICE,
    OBJECT_POSITION_SLICE,
    OBJECT_QUATERNION_SLICE,
    RAW_MOCAP_QUATERNION_SLICE,
    RAW_QPOS_SLICE,
    RIGHT_CONTACT_INDEX,
    STATE_DIM,
    TCP_SLICE,
    deterministic_oracle_scores,
)


def _sha(value: int) -> str:
    return f"{value:064x}"


def _valid_state(task: str = "pick-place-v3") -> np.ndarray:
    state = np.zeros(STATE_DIM, dtype=np.float32)
    state[OBJECT_QUATERNION_SLICE] = np.asarray([0.0, 0.0, 0.0, 1.0])
    state[OBJECT_INITIAL_POSITION_SLICE] = np.asarray([0.1, 0.1, 0.0])
    state[RAW_MOCAP_QUATERNION_SLICE] = np.asarray([1.0, 0.0, 0.0, 0.0])
    if task != "button-press-v3":
        # Pick/reach/push use the maximum 16-qpos schema with a free object joint.
        state[RAW_QPOS_SLICE.start + 12 : RAW_QPOS_SLICE.start + 16] = np.asarray(
            [1.0, 0.0, 0.0, 0.0]
        )
    return state


def _synthetic_bundle(*, snapshots_per_split: int = 8):
    tasks = len(TASK_NAMES)
    samples = tasks * 2 * snapshots_per_split
    rng = np.random.default_rng(123)
    center = np.zeros((samples, 25, 4), dtype=np.float32)
    candidates = rng.normal(0.0, 0.05, (samples, 8, 25, 4)).astype(np.float32)
    current = np.stack([_valid_state() for _ in range(samples)])
    future = np.empty((samples, 8, 5, STATE_DIM), dtype=np.float32)
    goal = np.zeros((samples, 3), dtype=np.float32)
    actual_return = np.empty((samples, 8, 5), dtype=np.float32)
    physical_task_score = np.empty((samples, 8), dtype=np.float32)
    progress = np.empty_like(actual_return)
    success = np.zeros((samples, 8, 5), dtype=np.bool_)
    task_index = np.empty(samples, dtype=np.int64)
    episode_id = np.empty(samples, dtype=np.int64)
    env_seed = np.empty(samples, dtype=np.int64)
    snapshot_step = np.full(samples, 25, dtype=np.int64)
    phase_index = np.full(samples, 1, dtype=np.int64)
    split_index = np.empty(samples, dtype=np.int64)
    axis = np.broadcast_to(np.repeat(np.arange(4), 2), (samples, 8)).copy()
    sign = np.broadcast_to(np.tile(np.asarray([-1, 1]), 4), (samples, 8)).copy()
    permutation = np.broadcast_to(np.arange(8), (samples, 8)).copy()
    delta_norm = np.full((samples, 8), 0.15, dtype=np.float32)
    rows = []

    sample = 0
    for task in range(tasks):
        for split in range(2):
            for local in range(snapshots_per_split):
                quality = np.roll(np.arange(8, dtype=np.float32), local)
                distance = (8.0 - quality) * 0.05
                current[sample] = _valid_state(TASK_NAMES[task])
                task_index[sample] = task
                split_index[sample] = split
                episode_id[sample] = 1000 * task + 100 * split + local
                env_seed[sample] = 50000 + sample
                for candidate in range(8):
                    state = _valid_state(TASK_NAMES[task])
                    state[LEFT_CONTACT_INDEX] = 1.0
                    state[RIGHT_CONTACT_INDEX] = 1.0
                    state[TCP_SLICE] = np.asarray(
                        [distance[candidate], 0.0, 0.0], dtype=np.float32
                    )
                    if TASK_NAMES[task] == "button-press-v3":
                        state[OBJECT_POSITION_SLICE] = np.asarray(
                            [0.0, distance[candidate], 0.0], dtype=np.float32
                        )
                    else:
                        state[OBJECT_POSITION_SLICE] = np.asarray(
                            [distance[candidate], 0.0, 0.0], dtype=np.float32
                        )
                    future[sample, candidate] = np.broadcast_to(state, (5, STATE_DIM))
                    actual_return[sample, candidate] = quality[candidate]
                    progress[sample, candidate] = quality[candidate] / 7.0
                    success[sample, candidate] = quality[candidate] == 7.0
                physical_task_score[sample] = deterministic_oracle_scores(
                    TASK_NAMES[task],
                    current[sample],
                    future[sample],
                    goal[sample],
                ).score
                split_name = "train" if split == 0 else "validation"
                rows.append(
                    {
                        "sample_index": sample,
                        "task": TASK_NAMES[task],
                        "episode_id": int(episode_id[sample]),
                        "env_seed": int(env_seed[sample]),
                        "phase": PHASE_NAMES[1],
                        "snapshot_step": 25,
                        "split": split_name,
                        "physical_episode_sha256": _sha(100000 + sample),
                        "physical_snapshot_sha256": _sha(200000 + sample),
                        "center_action_sha256": sha256_array(center[sample]),
                        "candidate_action_sha256": [
                            sha256_array(candidates[sample, candidate])
                            for candidate in range(8)
                        ],
                    }
                )
                sample += 1
    deployment = {
        "center_action_raw": center,
        "candidate_action_raw": candidates,
    }
    labels = {
        "current_state": current,
        "future_state": future,
        "goal_world": goal,
        "actual_return": actual_return,
        "physical_task_score": physical_task_score,
        "progress": progress,
        "success": success,
    }
    audit = {
        "task_index": task_index,
        "episode_id": episode_id,
        "env_seed": env_seed,
        "snapshot_step": snapshot_step,
        "phase_index": phase_index,
        "split_index": split_index,
        "candidate_axis": axis,
        "candidate_sign": sign,
        "candidate_permutation": permutation,
        "candidate_delta_rms": delta_norm,
    }
    return prepare_gate_bundle(deployment, labels, audit, rows)


def _gate_config() -> dict[str, object]:
    return {
        "minimum_return_spread_fraction": 0.8,
        "minimum_physical_score_spread": 0.1,
        "minimum_actual_return_spread": 0.01,
        "physical_return_alignment_required": True,
        "shortcut_abs_spearman_max": 0.1,
        "shortcut_regret_improvement_max": 0.02,
        "bootstrap_samples": 50,
        "bootstrap_seed": 9231,
        "direct_training_seeds": list(DIRECT_TRAINING_SEEDS),
        "direct_steps": 1,
        "direct_learning_rate": 0.001,
        "direct_weight_decay": 0.0001,
        "direct_gradient_clip_norm": 1.0,
        "direct_hidden_dim": 12,
        "direct_action_hidden_dim": 6,
        "direct_minimum_relative_loss_reduction": 0.1,
        "direct_minimum_train_spearman": 0.2,
        "pairwise_weight": 1.0,
        "huber_weight": 0.1,
        "huber_beta": 1.0,
        "tie_tolerance": 0.000001,
    }


def test_direct_ranker_schema_has_no_label_and_is_candidate_equivariant() -> None:
    assert {field.name for field in fields(DirectRankerInputs)} == {
        "current_state",
        "candidate_action_raw",
        "goal_world",
        "task_onehot",
    }
    torch.manual_seed(4)
    model = PrivilegedDirectRanker(hidden_dim=16, action_hidden_dim=8).eval()
    state = torch.randn(3, STATE_DIM)
    action = torch.randn(3, 8, 25, 4)
    goal = torch.randn(3, 3)
    task = torch.eye(4)[:3]
    permutation = torch.tensor([5, 1, 7, 0, 3, 6, 4, 2])
    with torch.no_grad():
        reference = model(state, action, goal, task)
        permuted = model(state, action[:, permutation], goal, task)
    assert torch.allclose(permuted, reference[:, permutation], atol=1e-6, rtol=0.0)


def test_bundle_alignment_and_direct_inputs_keep_return_out() -> None:
    bundle = _synthetic_bundle()
    inputs = direct_ranker_inputs(bundle)
    assert inputs.current_state.shape[-1] == STATE_DIM
    poisoned = dict(bundle.deployment_inputs)
    poisoned["actual_return"] = bundle.offline_labels["actual_return"]
    with pytest.raises(ValueError, match="deployment_inputs keys changed"):
        prepare_gate_bundle(
            poisoned,
            bundle.offline_labels,
            bundle.audit_arrays,
            bundle.index_rows,
        )


def test_formal_gate_recomputes_stored_physical_scores() -> None:
    bundle = _synthetic_bundle()
    validate_physical_score_contract(bundle)
    poisoned = {name: value.copy() for name, value in bundle.offline_labels.items()}
    poisoned["physical_task_score"][0, 0] += np.float32(0.01)
    changed = prepare_gate_bundle(
        bundle.deployment_inputs,
        poisoned,
        bundle.audit_arrays,
        bundle.index_rows,
    )
    with pytest.raises(ValueError, match="does not match the deterministic StateOracle"):
        validate_physical_score_contract(changed)


def test_tie_aware_snapshot_metrics_and_spread_audit() -> None:
    target = np.arange(8, dtype=np.float32)
    tied = snapshot_ranking_metrics(np.zeros(8), target)
    assert tied.spearman == pytest.approx(0.0)
    assert tied.normalized_top1_regret == pytest.approx(0.5)
    perfect = snapshot_ranking_metrics(target, target)
    assert perfect.spearman == pytest.approx(1.0)
    assert perfect.normalized_top1_regret == pytest.approx(0.0)
    report = candidate_return_spread_audit(_synthetic_bundle())
    assert report["passed"] is True
    assert report["fraction"] == pytest.approx(1.0)
    physical = candidate_physical_score_spread_audit(_synthetic_bundle())
    assert physical["passed"] is True
    assert all(physical["per_task_passed"].values())


def test_spread_gate_is_per_task_not_a_pooled_average() -> None:
    bundle = _synthetic_bundle()
    labels = {name: value.copy() for name, value in bundle.offline_labels.items()}
    validation_task_zero = np.flatnonzero(
        bundle.validation_mask & (bundle.audit_arrays["task_index"] == 0)
    )
    # Keep only two of eight task-0 snapshots informative. The pooled fraction
    # remains 26/32 > 0.8, but task 0 must still fail its own 80% requirement.
    labels["physical_task_score"][validation_task_zero[2:]] = 0.0
    changed = prepare_gate_bundle(
        bundle.deployment_inputs,
        labels,
        bundle.audit_arrays,
        bundle.index_rows,
    )
    report = candidate_physical_score_spread_audit(changed)
    assert report["fraction"] > 0.8
    assert report["per_task_fraction"][TASK_NAMES[0]] == pytest.approx(0.25)
    assert report["per_task_passed"][TASK_NAMES[0]] is False
    assert report["passed"] is False


def test_shortcut_lookups_use_train_only_and_detect_a_slot_shortcut() -> None:
    balanced = _synthetic_bundle()
    report = shortcut_lookup_audits(balanced)
    assert report["passed"] is True
    assert report["task_x_anonymous_slot"]["task_macro_spearman"] == pytest.approx(0.0)
    assert report["task_x_hidden_axis_sign"]["task_macro_spearman"] == pytest.approx(0.0)

    labels = {name: value.copy() for name, value in balanced.offline_labels.items()}
    slot_quality = np.arange(8, dtype=np.float32)
    labels["physical_task_score"][:] = slot_quality[None, :]
    shortcut = prepare_gate_bundle(
        balanced.deployment_inputs,
        labels,
        balanced.audit_arrays,
        balanced.index_rows,
    )
    failed = shortcut_lookup_audits(shortcut)
    assert failed["passed"] is False
    assert failed["task_x_anonymous_slot"]["task_macro_spearman"] == pytest.approx(1.0)
    assert (
        failed["task_x_anonymous_slot"][
            "spearman_ci_inside_equivalence_interval"
        ]
        is False
    )


def test_run_gate_stops_at_p0_before_direct_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    balanced = _synthetic_bundle()
    labels = {name: value.copy() for name, value in balanced.offline_labels.items()}
    # Build a genuine physical slot shortcut rather than poisoning the stored
    # primary labels.  The formal entry point must first prove those labels are
    # exactly reproducible from future state.
    for sample in range(balanced.axes.sample_count):
        task = TASK_NAMES[int(balanced.audit_arrays["task_index"][sample])]
        for candidate in range(8):
            distance = (8.0 - float(candidate)) * 0.05
            state = _valid_state(task)
            state[LEFT_CONTACT_INDEX] = 1.0
            state[RIGHT_CONTACT_INDEX] = 1.0
            state[TCP_SLICE] = np.asarray([distance, 0.0, 0.0], dtype=np.float32)
            if task == "button-press-v3":
                state[OBJECT_POSITION_SLICE] = np.asarray(
                    [0.0, distance, 0.0], dtype=np.float32
                )
            else:
                state[OBJECT_POSITION_SLICE] = np.asarray(
                    [distance, 0.0, 0.0], dtype=np.float32
                )
            labels["future_state"][sample, candidate] = np.broadcast_to(
                state, (5, STATE_DIM)
            )
        labels["physical_task_score"][sample] = deterministic_oracle_scores(
            task,
            labels["current_state"][sample],
            labels["future_state"][sample],
            labels["goal_world"][sample],
        ).score
    shortcut = prepare_gate_bundle(
        balanced.deployment_inputs,
        labels,
        balanced.audit_arrays,
        balanced.index_rows,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Direct training started before P0 passed")

    monkeypatch.setattr(v3_gate, "train_direct_ranker_ensemble", forbidden)
    with pytest.raises(StopGate) as caught:
        v3_gate.run_state_oracle_gate(shortcut, _gate_config(), device="cpu")
    assert caught.value.report["direct_training_started"] is False
    assert caught.value.report["shortcut_audits"]["passed"] is False


def test_oracle_and_cyclic_scores_are_separate_matched_controls() -> None:
    bundle = _synthetic_bundle()
    oracle = deterministic_state_oracle_score_matrix(bundle)
    cyclic = deterministic_state_oracle_score_matrix(bundle, cyclic_candidate_shift=1)
    assert oracle.shape == cyclic.shape == (32, 8)
    assert not np.array_equal(oracle, cyclic)
    oracle_rows = metric_rows(bundle, oracle, scorer="oracle")
    cyclic_rows = metric_rows(bundle, cyclic, scorer="cyclic")
    comparison = paired_episode_within_task_bootstrap(
        oracle_rows,
        cyclic_rows,
        metric="spearman",
        higher_is_better=True,
        bootstrap_samples=100,
        bootstrap_seed=9,
    )
    assert comparison["ci95"][0] > 0.0
    assert comparison["positive_task_count"] == 4


def test_cyclic_control_rotates_canonical_axis_not_anonymous_slot() -> None:
    bundle = _synthetic_bundle()
    audit = {name: value.copy() for name, value in bundle.audit_arrays.items()}
    permutation = np.asarray([3, 6, 1, 4, 7, 0, 5, 2], dtype=np.int16)
    audit["candidate_permutation"][:] = permutation
    audit["candidate_axis"][:] = permutation // 2
    audit["candidate_sign"][:] = np.where(
        permutation % 2 == 0, 1, -1
    ).astype(np.int8)
    permuted = prepare_gate_bundle(
        bundle.deployment_inputs,
        bundle.offline_labels,
        audit,
        bundle.index_rows,
    )
    cyclic = deterministic_state_oracle_score_matrix(
        permuted, cyclic_candidate_shift=1
    )
    sample = int(np.flatnonzero(permuted.validation_mask)[0])
    inverse = np.empty(8, dtype=np.int64)
    inverse[permutation] = np.arange(8)
    target_canonical = ((permutation // 2 + 1) % 4) * 2 + permutation % 2
    expected_future = permuted.offline_labels["future_state"][sample][
        inverse[target_canonical]
    ]
    expected = deterministic_oracle_scores(
        TASK_NAMES[int(permuted.audit_arrays["task_index"][sample])],
        permuted.offline_labels["current_state"][sample],
        expected_future,
        permuted.offline_labels["goal_world"][sample],
    ).score
    np.testing.assert_array_equal(cyclic[0], expected)


def test_gate_requires_the_same_three_tasks_positive_on_both_metrics() -> None:
    better = []
    baseline = []
    for task_index, task in enumerate(TASK_NAMES):
        common = {
            "task": task,
            "physical_episode_sha256": _sha(7000 + task_index),
            "physical_snapshot_sha256": _sha(8000 + task_index),
        }
        baseline.append(
            {**common, "spearman": 0.0, "normalized_top1_regret": 0.5}
        )
        better.append(
            {
                **common,
                "spearman": 0.1 if task_index in {0, 1, 2} else -0.01,
                "normalized_top1_regret": (
                    0.4 if task_index in {1, 2, 3} else 0.51
                ),
            }
        )
    report = v3_gate._comparison_report(
        better,
        baseline,
        bootstrap_samples=20,
        bootstrap_seed=5,
    )
    assert report["spearman"]["ci95"][0] > 0.0
    assert report["normalized_top1_regret"]["ci95"][0] > 0.0
    assert report["spearman"]["positive_task_count"] == 3
    assert report["normalized_top1_regret"]["positive_task_count"] == 3
    assert report["jointly_positive_tasks"] == [TASK_NAMES[1], TASK_NAMES[2]]
    assert report["jointly_positive_task_count"] == 2
    assert report["passed"] is False


def test_synthetic_state_oracle_gate_passes_all_predeclared_conditions() -> None:
    bundle = _synthetic_bundle()
    direct = np.zeros((int(np.sum(bundle.validation_mask)), 8), dtype=np.float32)
    report = evaluate_state_oracle_gate(
        bundle,
        direct,
        bootstrap_samples=200,
        bootstrap_seed=12,
    )
    assert report["passed"] is True
    assert report["candidate_physical_score_spread"]["fraction"] == pytest.approx(
        1.0
    )
    assert report["candidate_actual_return_spread"]["fraction"] == pytest.approx(
        1.0
    )
    assert report["shortcut_audits"]["passed"] is True
    assert report["physical_return_alignment"]["passed"] is True
    assert report["cross_baseline_jointly_positive_task_count"] == 4
    assert report["stage"] == "P1_state_oracle"
    assert report["direct_training_started"] is True
    for comparison in report["comparisons"].values():
        assert comparison["passed"] is True
        assert comparison["spearman"]["positive_task_count"] == 4
        assert comparison["normalized_top1_regret"]["positive_task_count"] == 4


def test_direct_ensemble_averages_fixed_seed_scores_before_metrics() -> None:
    bundle = _synthetic_bundle(snapshots_per_split=2)
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()
    first = train_direct_ranker_ensemble(
        bundle,
        seeds=DIRECT_TRAINING_SEEDS,
        steps=2,
        hidden_dim=12,
        action_hidden_dim=6,
        device="cpu",
    )
    second = train_direct_ranker_ensemble(
        bundle,
        seeds=DIRECT_TRAINING_SEEDS,
        steps=2,
        hidden_dim=12,
        action_hidden_dim=6,
        device="cpu",
    )
    assert first.training_seeds == DIRECT_TRAINING_SEEDS
    assert first.ensemble_rule == "average_seed_scores_before_metrics"
    assert first.validation_scores.shape == (8, 8)
    assert first.training_scores.shape == (8, 8)
    assert first.parameter_count > 0
    assert np.array_equal(first.validation_scores, second.validation_scores)
    assert len(first.state_dicts) == len(first.state_sha256) == 3
    assert len(first.initial_losses) == len(first.final_losses) == 3
    assert all(
        tensor.device.type == "cpu"
        for state in first.state_dicts
        for tensor in state.values()
    )
    assert first.input_scaler.state_mean.shape == (STATE_DIM,)
    assert first.input_scaler.action_mean.shape == (4,)
    assert first.label_normalizer.mean.shape == (4,)
    assert torch.are_deterministic_algorithms_enabled() is deterministic_before
    assert torch.is_deterministic_algorithms_warn_only_enabled() is warn_only_before

    restored = PrivilegedDirectRanker(**first.model_config)
    restored.load_state_dict(first.state_dicts[0], strict=True)
    assert scorer_state_sha256(restored) == first.state_sha256[0]
    assert first.state_sha256 == second.state_sha256


def test_direct_training_eligibility_rejects_one_undertrained_seed() -> None:
    bundle = _synthetic_bundle(snapshots_per_split=2)
    trained = train_direct_ranker_ensemble(
        bundle,
        seeds=DIRECT_TRAINING_SEEDS,
        steps=1,
        hidden_dim=12,
        action_hidden_dim=6,
        device="cpu",
    )
    perfect_train_scores = bundle.physical_task_score[bundle.train_mask]
    qualified = replace(
        trained,
        training_scores=perfect_train_scores,
        initial_losses=(1.0, 1.0, 1.0),
        final_losses=(0.5, 0.5, 0.5),
    )
    passed = direct_training_eligibility(bundle, qualified)
    assert passed["passed"] is True
    assert passed["direct_train_metrics"]["task_macro_spearman"] == pytest.approx(
        1.0
    )

    undertrained = replace(
        qualified,
        final_losses=(0.95, 0.5, 0.5),
    )
    failed = direct_training_eligibility(bundle, undertrained)
    assert failed["passed"] is False
    assert failed["all_seeds_loss_reduction_passed"] is False


def test_run_gate_preserves_rejected_direct_ensemble(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _synthetic_bundle()
    trained = train_direct_ranker_ensemble(
        bundle,
        seeds=DIRECT_TRAINING_SEEDS,
        steps=1,
        hidden_dim=12,
        action_hidden_dim=6,
        device="cpu",
    )
    rejected = replace(
        trained,
        training_scores=bundle.physical_task_score[bundle.train_mask],
        initial_losses=(1.0, 1.0, 1.0),
        final_losses=(0.95, 0.5, 0.5),
    )
    monkeypatch.setattr(
        v3_gate,
        "train_direct_ranker_ensemble",
        lambda *_args, **_kwargs: rejected,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("StateOracle comparison ran after Direct rejection")

    monkeypatch.setattr(v3_gate, "evaluate_state_oracle_gate", forbidden)
    with pytest.raises(StopGate) as caught:
        v3_gate.run_state_oracle_gate(bundle, _gate_config(), device="cpu")
    assert caught.value.report["stage"] == "P1_direct_training_invalid"
    assert caught.value.report["direct_training_started"] is True
    assert caught.value.direct_ensemble is rejected
