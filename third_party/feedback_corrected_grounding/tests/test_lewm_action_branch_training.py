from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from fcpg.lewm_action_branch_training import (
    ActionBranchCache,
    CachedActionBranchDataset,
    all_cyclic_branch_derangements,
    assert_matched_predictor_contract,
    budget_steps,
    fixed_branch_derangement,
    load_action_branch_cache,
    matched_action_branch_loss,
    matched_action_branch_rollout,
    permute_action_branches,
    rollout_action_branches,
    validate_frozen_base_checkpoint,
)
from fcpg.lewm_anchor_dynamics_pair import build_matched_predictor_pair
from scripts import train_lewm_action_branch_a1 as cli


def _checkpoint(*, dropout: float = 0.0) -> dict:
    return {
        "checkpoint_type": "fcpg_anchor_persistent_lewm_v1",
        "step": 2000,
        "training_budget": {
            "target_optimizer_steps": 2000,
            "checkpoint_optimizer_step": 2000,
            "run_completed": True,
        },
        "model_state": {},
        "resolved_config": {"loss": {"sigreg_weight": 0.02}},
        "normalization": {
            "action_mean": [0.0, 0.0, 0.0],
            "action_std": [1.0, 1.0, 1.0],
        },
        "model_config": {
            "frame": {"latent_dim": 8, "num_keypoints": 2},
            "world_model": {
                "action_dim": 3,
                "history_size": 3,
                "action_smoothing_dim": 2,
                "predictor_depth": 1,
                "predictor_heads": 1,
                "predictor_mlp_dim": 16,
                "predictor_dim_head": 8,
                "predictor_dropout": dropout,
                "projector_hidden_dim": 16,
            },
        },
    }


class _ActionProbe(nn.Module):
    history_size = 3
    num_keypoints = 1
    latent_dim = 1
    action_dim = 1

    def __init__(self) -> None:
        super().__init__()
        self.seen_actions: list[torch.Tensor] = []

    def predict(
        self, embedding: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        self.seen_actions.append(action.detach().clone())
        next_embedding = action[:, -1, 0].reshape(-1, 1, 1, 1)
        return torch.cat((embedding[:, 1:], next_embedding), dim=1)


class _TwoStepScale(nn.Module):
    history_size = 3
    num_keypoints = 1
    latent_dim = 1
    action_dim = 1

    def __init__(self) -> None:
        super().__init__()
        self.scales = nn.Parameter(torch.tensor([2.0, 3.0]))
        self.call_index = 0

    def predict(
        self, embedding: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        del action
        next_embedding = embedding[:, -1:] * self.scales[self.call_index]
        self.call_index += 1
        return torch.cat((embedding[:, 1:], next_embedding), dim=1)


class _A1EvaluationProbe(nn.Module):
    history_size = 3
    num_keypoints = 3
    latent_dim = 192
    action_dim = 20

    def __init__(self, *, condition_on_action: bool) -> None:
        super().__init__()
        self.condition_on_action = condition_on_action

    def predict(
        self, embedding: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        if self.condition_on_action:
            increment = action[:, -1].mean(dim=-1).reshape(-1, 1, 1, 1)
        else:
            increment = torch.zeros_like(action[:, -1, :1]).reshape(-1, 1, 1, 1)
        next_embedding = embedding[:, -1:] + increment
        return torch.cat((embedding[:, 1:], next_embedding), dim=1)


def test_action_shift_uses_current_action_then_shifts_both_histories() -> None:
    model = _ActionProbe()
    history = torch.zeros(1, 3, 1, 1)
    history_action = torch.tensor([[[1.0], [2.0]]])
    future_action = torch.tensor([[[[10.0], [20.0], [30.0]]]])

    prediction = rollout_action_branches(
        model, history, history_action, future_action
    )

    torch.testing.assert_close(prediction.flatten(), torch.tensor([10.0, 20.0, 30.0]))
    torch.testing.assert_close(
        model.seen_actions[0].flatten(), torch.tensor([1.0, 2.0, 10.0])
    )
    torch.testing.assert_close(
        model.seen_actions[1].flatten(), torch.tensor([2.0, 10.0, 20.0])
    )
    torch.testing.assert_close(
        model.seen_actions[2].flatten(), torch.tensor([10.0, 20.0, 30.0])
    )


def test_fixed_action_derangement_has_no_fixed_branch_and_is_stable() -> None:
    sample_index = torch.tensor([0, 1, 7, 9])
    first = fixed_branch_derangement(sample_index, branch_count=8)
    second = fixed_branch_derangement(sample_index, branch_count=8)
    slots = torch.arange(8).expand(4, -1)

    torch.testing.assert_close(first, second)
    assert not bool(torch.any(first == slots))
    torch.testing.assert_close(torch.sort(first, dim=1).values, slots)
    action = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8, 1, 1)
    shuffled = permute_action_branches(action, first)
    torch.testing.assert_close(
        shuffled[:, :, 0, 0], torch.gather(action[:, :, 0, 0], 1, first)
    )


def test_all_seven_cyclic_donor_shifts_are_complete_derangements() -> None:
    permutations = all_cyclic_branch_derangements(3, branch_count=8)
    slots = torch.arange(8)

    assert permutations.shape == (3, 7, 8)
    for shift_index in range(7):
        expected = torch.remainder(slots + shift_index + 1, 8)
        torch.testing.assert_close(
            permutations[:, shift_index], expected.expand(3, -1)
        )
    assert not bool(torch.any(permutations == slots.reshape(1, 1, 8)))
    torch.testing.assert_close(
        torch.sort(permutations, dim=-1).values,
        slots.reshape(1, 1, 8).expand(3, 7, -1),
    )


def test_no_action_closed_loop_is_invariant_to_all_future_actions() -> None:
    _, no_action = build_matched_predictor_pair(
        _checkpoint(), initialization_seed=21
    )
    no_action.eval()
    history = torch.randn(2, 3, 2, 8)
    history_action = torch.randn(2, 2, 3)
    first_action = torch.randn(2, 8, 4, 3)
    second_action = torch.roll(first_action * 17.0, shifts=1, dims=2)

    first = rollout_action_branches(
        no_action, history, history_action, first_action
    )
    second = rollout_action_branches(
        no_action, history, history_action, second_action
    )

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_later_horizon_gradient_reaches_an_earlier_prediction() -> None:
    model = _TwoStepScale()
    history = torch.ones(1, 3, 1, 1)
    history_action = torch.zeros(1, 2, 1)
    future_action = torch.zeros(1, 1, 2, 1)

    prediction = rollout_action_branches(
        model, history, history_action, future_action
    )
    prediction[:, :, -1].sum().backward()

    torch.testing.assert_close(prediction.flatten(), torch.tensor([2.0, 6.0]))
    assert model.scales.grad is not None
    assert float(model.scales.grad[0]) != 0.0
    assert float(model.scales.grad[1]) != 0.0


def test_matched_rng_covers_the_complete_dropout_rollout() -> None:
    factual, no_action = build_matched_predictor_pair(
        _checkpoint(dropout=0.5), initialization_seed=37
    )
    assert_matched_predictor_contract(factual, no_action)
    factual.train()
    no_action.train()
    history = torch.randn(3, 3, 2, 8)
    history_action = torch.zeros(3, 2, 3)
    future_action = torch.zeros(3, 4, 4, 3)

    factual_prediction, no_action_prediction = matched_action_branch_rollout(
        factual, no_action, history, history_action, future_action
    )

    torch.testing.assert_close(
        factual_prediction, no_action_prediction, rtol=0.0, atol=0.0
    )


def test_frozen_base_checkpoint_requires_exact_sigreg_002() -> None:
    valid = _checkpoint()
    assert validate_frozen_base_checkpoint(valid) is valid["resolved_config"]

    invalid = _checkpoint()
    invalid["resolved_config"]["loss"]["sigreg_weight"] = 0.0
    with pytest.raises(ValueError, match="exactly 0.02"):
        validate_frozen_base_checkpoint(invalid)

    with pytest.raises(ValueError, match="resolved_config"):
        invalid = _checkpoint()
        invalid.pop("resolved_config")
        validate_frozen_base_checkpoint(invalid)


def test_frozen_base_checkpoint_requires_type_step_and_completed_budget() -> None:
    invalid_type = _checkpoint()
    invalid_type["checkpoint_type"] = "wrong"
    with pytest.raises(ValueError, match="AnchorPersistent"):
        validate_frozen_base_checkpoint(invalid_type)

    invalid_step = _checkpoint()
    invalid_step["step"] = 17
    with pytest.raises(ValueError, match="step 2000"):
        validate_frozen_base_checkpoint(invalid_step)

    incomplete = _checkpoint()
    incomplete["training_budget"]["run_completed"] = False
    with pytest.raises(ValueError, match="incomplete"):
        validate_frozen_base_checkpoint(incomplete)


def test_all_horizon_raw_loss_and_output_shape() -> None:
    factual, no_action = build_matched_predictor_pair(
        _checkpoint(), initialization_seed=51
    )
    factual.eval()
    no_action.eval()
    history = torch.randn(2, 3, 2, 8)
    history_action = torch.randn(2, 2, 3)
    future_action = torch.randn(2, 8, 8, 3)
    target = torch.randn(2, 8, 8, 2, 8)

    output = matched_action_branch_loss(
        factual,
        no_action,
        history,
        history_action,
        future_action,
        target,
    )

    assert output.factual_prediction.shape == (2, 8, 8, 2, 8)
    assert output.no_action_prediction.shape == (2, 8, 8, 2, 8)
    expected = (output.factual_prediction.float() - target).square().mean()
    torch.testing.assert_close(output.factual_raw_mse, expected)
    torch.testing.assert_close(
        output.loss, output.factual_raw_mse + output.no_action_raw_mse
    )
    assert output.factual_branch_centered_mse.ndim == 0
    assert output.no_action_branch_centered_mse.ndim == 0


def test_fixed_budgets_use_bundle_batch_defaults_and_fresh_10k() -> None:
    assert budget_steps("smoke") == 100
    assert budget_steps("2k") == 2_000
    assert budget_steps("10k") == 10_000
    assert cli.DEFAULT_BUNDLE_BATCH_SIZE == {"smoke": 4, "2k": 16, "10k": 16}
    assert cli.DEFAULT_BASE_CHECKPOINT == Path(
        cli.ROOT,
        "outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k",
        "lewm_grounded_latest.pt",
    )
    args = cli.parse_args(
        [
            "--input-npz",
            "cache.npz",
            "--output-dir",
            "run",
            "--budget",
            "10k",
        ]
    )
    assert args.batch_size is None
    assert cli.DEFAULT_BUNDLE_BATCH_SIZE[args.budget] == 16
    assert cli._default_splits("2k", None, None) == ("train", "development")
    with pytest.raises(ValueError, match="freezes train/val splits"):
        cli._default_splits("10k", "test", "test")


def test_10k_requires_a_passed_screen_summary(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="passed A1 screen"):
        bad = tmp_path / "bad.json"
        bad.write_text(
            json.dumps(
                {
                    "protocol": "fcpg_action_branch_random_hazard_a1_v1",
                    "stage": "screen",
                    "decision": "No-Go",
                    "passed": False,
                }
            ),
            encoding="utf-8",
        )
        cli._validate_10k_authorization(bad)

    task_names = [f"task-{index}" for index in range(4)]
    good = tmp_path / "good.json"
    good.write_text(
        json.dumps(
            {
                "protocol": "fcpg_action_branch_random_hazard_a1_v1",
                "stage": "screen",
                "decision": "Go",
                "passed": True,
                "gate": {
                    "stage": "screen",
                    "scientific_decision": True,
                    "evaluated": True,
                    "decision": "Go",
                    "passed": True,
                    "failed_checks": [],
                    "checks": {
                        "informative_bundle_fraction_per_task": {
                            "passed": True
                        }
                    },
                },
                "coverage": {
                    "task_count": 4,
                    "task_names": task_names,
                    "training_seed_count": 3,
                    "training_seeds": [18401, 18402, 18403],
                },
                "informative_bundles": {
                    "all_tasks_passed": True,
                    "per_task": {
                        task: {"passed": True} for task in task_names
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    assert cli._validate_10k_authorization(good) is not None

    inconsistent = tmp_path / "inconsistent.json"
    payload = json.loads(good.read_text(encoding="utf-8"))
    payload["gate"]["decision"] = "No-Go"
    inconsistent.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="consistent nested A1 screen Go"):
        cli._validate_10k_authorization(inconsistent)

    failed_checks = tmp_path / "failed-checks.json"
    payload = json.loads(good.read_text(encoding="utf-8"))
    payload["gate"]["failed_checks"] = [
        "informative_bundle_fraction_per_task"
    ]
    failed_checks.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="consistent nested A1 screen Go"):
        cli._validate_10k_authorization(failed_checks)

    unsupported = tmp_path / "unsupported.json"
    payload = json.loads(good.read_text(encoding="utf-8"))
    payload["gate"]["checks"]["informative_bundle_fraction_per_task"][
        "passed"
    ] = False
    unsupported.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="informative support"):
        cli._validate_10k_authorization(unsupported)

    incomplete_tasks = tmp_path / "incomplete-tasks.json"
    payload = json.loads(good.read_text(encoding="utf-8"))
    payload["coverage"]["task_names"] = task_names[:3]
    incomplete_tasks.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly four-task"):
        cli._validate_10k_authorization(incomplete_tasks)

    incomplete_coverage = tmp_path / "incomplete-coverage.json"
    payload = json.loads(good.read_text(encoding="utf-8"))
    payload["coverage"]["training_seeds"] = [18401, 18402]
    incomplete_coverage.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly three-seed"):
        cli._validate_10k_authorization(incomplete_coverage)

    failed_task = tmp_path / "failed-task.json"
    payload = json.loads(good.read_text(encoding="utf-8"))
    payload["informative_bundles"]["per_task"][task_names[-1]][
        "passed"
    ] = False
    failed_task.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="informative support"):
        cli._validate_10k_authorization(failed_task)


def test_cache_reads_canonical_identity_fields(tmp_path: Path) -> None:
    path = tmp_path / "cache.npz"
    np.savez(
        path,
        history_embedding=np.zeros((2, 3, 1, 2), dtype=np.float32),
        history_action=np.zeros((2, 2, 1), dtype=np.float32),
        center_action=np.zeros((2, 2, 1), dtype=np.float32),
        future_action=np.zeros((2, 8, 2, 1), dtype=np.float32),
        target_embedding=np.zeros((2, 8, 2, 1, 2), dtype=np.float32),
        task_index=np.asarray([0, 0], dtype=np.int64),
        task_instance_index=np.asarray([0, 1], dtype=np.int64),
        source_episode_id=np.asarray(["episode-a", "episode-b"]),
        hazard_id=np.asarray([0, 1], dtype=np.int64),
        split=np.asarray(["development", "development"]),
    )

    cache = load_action_branch_cache(
        path,
        history_size=3,
        num_keypoints=1,
        latent_dim=2,
        action_dim=1,
        branch_count=8,
        minimum_horizon=2,
    )

    assert len(cache) == 2
    assert cache.indices_for_split("development").tolist() == [0, 1]
    assert cache.source_episode_id.tolist() == ["episode-a", "episode-b"]


def test_validation_persists_recursive_all_shift_scorer_schema(
    tmp_path: Path,
) -> None:
    sample_count = 2
    horizon = 8
    history = np.zeros((sample_count, 3, 3, 192), dtype=np.float32)
    history_action = np.zeros((sample_count, 2, 20), dtype=np.float32)
    future_action = np.empty(
        (sample_count, 8, horizon, 20), dtype=np.float32
    )
    for sample_index in range(sample_count):
        for branch_index in range(8):
            future_action[sample_index, branch_index] = (
                branch_index + 1 + sample_index * 0.25
            )
    cache = ActionBranchCache(
        path=tmp_path / "core.npz",
        history_embedding=history,
        history_action=history_action,
        center_action=np.zeros(
            (sample_count, horizon, 20), dtype=np.float32
        ),
        future_action=future_action,
        target_embedding=np.zeros(
            (sample_count, 8, horizon, 3, 192), dtype=np.float32
        ),
        split=np.asarray(["development", "development"]),
        task_index=np.asarray([0, 1], dtype=np.int64),
        task_instance_index=np.asarray([0, 1], dtype=np.int64),
        source_episode_id=np.asarray(["episode-a", "episode-b"]),
        hazard_id=np.asarray([3, 4], dtype=np.int64),
    )
    loader = DataLoader(
        CachedActionBranchDataset(cache, np.arange(sample_count)),
        batch_size=1,
        shuffle=False,
    )

    metrics, arrays = cli._evaluate_and_collect(
        _A1EvaluationProbe(condition_on_action=True),
        _A1EvaluationProbe(condition_on_action=False),
        loader,
        cache=cache,
        device=torch.device("cpu"),
        amp_enabled=False,
        amp_dtype=torch.bfloat16,
        step=100,
        training_seed=71,
    )

    assert arrays["factual_prediction"].shape == (2, 8, 8, 3, 192)
    assert arrays["no_action_prediction"].shape == (2, 8, 8, 3, 192)
    assert arrays["shuffled_action_prediction"].shape == (2, 7, 8, 8, 3, 192)
    assert arrays["shuffle_permutation"].shape == (2, 7, 8)
    assert arrays["prediction_heads_add_sigreg"].shape == ()
    assert not bool(arrays["prediction_heads_add_sigreg"])
    np.testing.assert_array_equal(
        arrays["shuffled_action_donor_shifts"], np.arange(1, 8)
    )
    np.testing.assert_allclose(
        arrays["shuffled_action_prediction"][0, 0, 0, :, 0, 0],
        2.0 * np.arange(1, horizon + 1),
    )
    np.testing.assert_allclose(
        arrays["shuffled_action_prediction"][0, :, 0, 0, 0, 0],
        np.arange(2.0, 9.0),
    )
    assert metrics["shuffled_action_raw_mse"] == pytest.approx(
        float(np.mean(arrays["shuffled_action_raw_mse_by_shift"]))
    )
