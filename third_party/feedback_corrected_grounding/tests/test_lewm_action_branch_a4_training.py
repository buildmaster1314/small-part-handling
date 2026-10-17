from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from torch import nn

import scripts.train_lewm_action_branch_a4 as training_cli

from fcpg.lewm_a4_checkpoint import validate_a4_checkpoint
from fcpg.lewm_action_branch_a4 import (
    BUNDLE_BATCH_SIZE,
    SIGREG_WEIGHT,
    TASKS,
)
from fcpg.lewm_action_branch_a4_training import (
    A4_FRESH_STATE_KEYS,
    A4_FROZEN_MODULES,
    A4_TRAINABLE_MODULES,
    assert_a4_matched_initialization,
    build_a4_latest_checkpoint,
    build_matched_a4_models,
    evaluate_a4_models,
    matched_a4_recursive_loss,
    resolve_a4_training_budget,
    validate_a4_training_batch,
)
from fcpg.lewm_anchor_persistent import (
    ANCHOR_PERSISTENT_CHECKPOINT_TYPE,
    ANCHOR_PERSISTENT_VARIANT,
    AnchorPersistentLeWM,
)
from fcpg.lewm_grounded_model import GroundedFrameEncoder
from fcpg.lewm_task_context_persistent import TaskContextPersistentLeWM


@dataclass(frozen=True)
class _VisionOutput:
    feature_map: torch.Tensor
    global_feature: torch.Tensor


class _TinyVision(nn.Module):
    output_dim = 8

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, self.output_dim, kernel_size=1)

    def forward(self, pixels: torch.Tensor) -> _VisionOutput:
        feature_map = self.projection(pixels)
        return _VisionOutput(
            feature_map=feature_map,
            global_feature=feature_map.mean(dim=(-1, -2)),
        )


def _frame_encoder() -> GroundedFrameEncoder:
    return GroundedFrameEncoder(
        vision_encoder=_TinyVision(),
        visual_dim=8,
        obs_dim=4,
        num_keypoints=3,
        latent_dim=192,
        depth=1,
        heads=4,
        dropout=0.0,
    )


def _anchor_model() -> AnchorPersistentLeWM:
    return AnchorPersistentLeWM(
        frame_encoder=_frame_encoder(),
        action_dim=20,
        history_size=3,
        predictor_depth=1,
        predictor_heads=4,
        predictor_mlp_dim=32,
        predictor_dim_head=8,
        predictor_dropout=0.0,
        projector_hidden_dim=32,
        sigreg_knots=5,
        sigreg_num_proj=8,
    )


def _task_model() -> TaskContextPersistentLeWM:
    return TaskContextPersistentLeWM(
        frame_encoder=_frame_encoder(),
        action_dim=20,
        num_tasks=7,
        history_size=3,
        predictor_depth=1,
        predictor_heads=4,
        predictor_mlp_dim=32,
        predictor_dim_head=8,
        predictor_dropout=0.0,
        projector_hidden_dim=32,
        sigreg_knots=5,
        sigreg_num_proj=8,
    )


def _base_checkpoint(*, sigreg_weight: float = 0.02) -> dict:
    anchor = _anchor_model()
    with torch.no_grad():
        for index, value in enumerate(anchor.state_dict().values(), start=1):
            if value.is_floating_point():
                value.fill_(index / 1000.0)
    return {
        "checkpoint_type": ANCHOR_PERSISTENT_CHECKPOINT_TYPE,
        "step": 2_000,
        "training_budget": {
            "target_optimizer_steps": 2_000,
            "checkpoint_optimizer_step": 2_000,
            "run_completed": True,
        },
        "resolved_config": {"loss": {"sigreg_weight": sigreg_weight}},
        "model_config": {
            "variant": ANCHOR_PERSISTENT_VARIANT,
            "frame": {"latent_dim": 192, "num_keypoints": 3},
            "world_model": {
                "action_dim": 20,
                "history_size": 3,
            },
        },
        "model_state": anchor.state_dict(),
    }


def _build_pair() -> tuple[TaskContextPersistentLeWM, TaskContextPersistentLeWM, object]:
    return build_matched_a4_models(
        _base_checkpoint(),
        seed=30_201,
        model_builder=lambda _: _task_model(),
    )


def _recursive_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "history_embedding": torch.randn(batch_size, 3, 4, 192),
        "history_action": torch.randn(batch_size, 2, 20),
        "future_action": torch.randn(batch_size, 8, 8, 20),
        "target_embedding": torch.randn(batch_size, 8, 8, 4, 192),
        "task_index": torch.arange(batch_size, dtype=torch.long) % len(TASKS),
    }


def test_budget_seed_and_anchor_sigreg_contract_are_exact() -> None:
    development = resolve_a4_training_budget("2k", seed=30_201)
    confirmation = resolve_a4_training_budget("10k", seed=30_303)
    assert development.optimizer_steps == 2_000
    assert development.evaluation_split == "development"
    assert confirmation.optimizer_steps == 10_000
    assert confirmation.train_split == "train"
    assert confirmation.evaluation_split is None
    with pytest.raises(ValueError, match="seed must be one of"):
        resolve_a4_training_budget("2k", seed=30_301)
    with pytest.raises(ValueError, match="sigreg_weight exactly 0.02"):
        build_matched_a4_models(
            _base_checkpoint(sigreg_weight=0.01),
            seed=30_201,
            model_builder=lambda _: _task_model(),
        )


def test_10k_never_constructs_an_inline_test_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = resolve_a4_training_budget("10k", seed=30_301)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("10k training must not build a reserved-test loader")

    monkeypatch.setattr(training_cli, "build_a4_evaluation_loader", forbidden)
    assert (
        training_cli._build_inline_evaluation_loader(
            {}, budget=budget, num_workers=0
        )
        is None
    )


def test_2k_keeps_the_existing_development_evaluation_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = resolve_a4_training_budget("2k", seed=30_201)
    sentinel = object()
    observed: dict[str, object] = {}

    def fake_builder(cache: object, **kwargs: object) -> object:
        observed["cache"] = cache
        observed.update(kwargs)
        return sentinel

    cache = {"split": np.asarray(["train", "development"])}
    monkeypatch.setattr(training_cli, "build_a4_evaluation_loader", fake_builder)
    result = training_cli._build_inline_evaluation_loader(
        cache, budget=budget, num_workers=3
    )

    assert result is sentinel
    assert observed == {
        "cache": cache,
        "split": "development",
        "batch_size": BUNDLE_BATCH_SIZE,
        "num_workers": 3,
        "pin_memory": True,
    }


def test_anchor_overlap_load_report_and_matched_state_are_exact() -> None:
    checkpoint = _base_checkpoint()
    factual, no_action, report = build_matched_a4_models(
        checkpoint,
        seed=30_201,
        model_builder=lambda _: _task_model(),
    )

    assert report.missing_keys == A4_FRESH_STATE_KEYS
    assert report.unexpected_keys == ()
    assert report.shape_mismatch_keys == ()
    assert set(report.loaded_keys) == set(checkpoint["model_state"])
    assert report.source_key_count == len(checkpoint["model_state"])
    assert report.target_key_count == report.source_key_count + 2
    for prefix in (
        "frame_encoder.",
        "predictor.",
        "action_encoder.",
        "projector.",
        "pred_proj.",
    ):
        assert any(name.startswith(prefix) for name in report.loaded_keys)
    for name in ("proprio_mean", "proprio_std", "action_mean", "action_std"):
        assert name in report.loaded_keys
    for name in report.loaded_keys:
        torch.testing.assert_close(
            factual.state_dict()[name], checkpoint["model_state"][name], rtol=0, atol=0
        )
    assert factual.num_tasks == no_action.num_tasks == len(TASKS)
    assert_a4_matched_initialization(factual, no_action)
    for module_name in A4_FROZEN_MODULES:
        assert not any(
            parameter.requires_grad
            for parameter in getattr(factual, module_name).parameters()
        )
    trainable_names = {
        name for name, parameter in factual.named_parameters() if parameter.requires_grad
    }
    assert trainable_names
    assert all(
        name.startswith(tuple(f"{module}." for module in A4_TRAINABLE_MODULES))
        for name in trainable_names
    )


def test_matched_recursive_loss_uses_no_predictor_head_sigreg() -> None:
    factual, no_action, _ = _build_pair()

    class _ForbiddenSIGReg(nn.Module):
        def forward(self, _: torch.Tensor) -> torch.Tensor:
            raise AssertionError("cached A4 dynamics must not call SIGReg")

    factual.sigreg = _ForbiddenSIGReg()
    no_action.sigreg = _ForbiddenSIGReg()
    batch = _recursive_batch(batch_size=2)
    output = matched_a4_recursive_loss(
        factual,
        no_action,
        batch["history_embedding"],
        batch["history_action"],
        batch["future_action"],
        batch["target_embedding"],
        batch["task_index"],
    )

    assert output.factual_prediction.shape == (2, 8, 8, 4, 192)
    assert output.no_action_prediction.shape == (2, 8, 8, 4, 192)
    torch.testing.assert_close(
        output.loss, output.factual_raw_mse + output.no_action_raw_mse
    )
    output.loss.backward()
    assert factual.task_embedding.weight.grad is not None
    assert no_action.task_embedding.weight.grad is not None
    assert all(parameter.grad is None for parameter in factual.frame_encoder.parameters())


def _strict_training_batch() -> dict[str, object]:
    task_index = torch.arange(len(TASKS), dtype=torch.long).repeat_interleave(4)
    return {
        "history_embedding": torch.zeros(BUNDLE_BATCH_SIZE, 3, 4, 192),
        "history_action": torch.zeros(BUNDLE_BATCH_SIZE, 2, 20),
        "future_action": torch.zeros(BUNDLE_BATCH_SIZE, 8, 8, 20),
        "target_embedding": torch.zeros(BUNDLE_BATCH_SIZE, 8, 8, 4, 192),
        "task_index": task_index,
        "task_name": [task for task in TASKS for _ in range(4)],
        "source_episode_id": [
            f"{task}:{episode}" for task in TASKS for episode in range(4)
        ],
    }


def test_training_batch_is_strictly_28_equals_7_by_4() -> None:
    batch = _strict_training_batch()
    validate_a4_training_batch(batch)

    wrong_order = dict(batch)
    wrong_order["task_index"] = torch.roll(batch["task_index"], 1)
    with pytest.raises(ValueError, match="four ordered rows"):
        validate_a4_training_batch(wrong_order)

    duplicate_episode = dict(batch)
    duplicate_episode["source_episode_id"] = list(batch["source_episode_id"])
    duplicate_episode["source_episode_id"][1] = duplicate_episode[
        "source_episode_id"
    ][0]
    with pytest.raises(ValueError, match="four distinct episodes"):
        validate_a4_training_batch(duplicate_episode)

    float_index = dict(batch)
    float_index["task_index"] = batch["task_index"].float()
    with pytest.raises(TypeError, match="integer dtype"):
        validate_a4_training_batch(float_index)


def test_evaluation_emits_development_comparison_predictions() -> None:
    factual, no_action, _ = _build_pair()
    batch = _recursive_batch(batch_size=2)
    batch.update(
        {
            "sample_index": torch.tensor([4, 9]),
            "task_name": [TASKS[0], TASKS[1]],
            "source_episode_id": ["episode-a", "episode-b"],
        }
    )

    result = evaluate_a4_models(
        factual,
        no_action,
        [batch],
        device="cpu",
        step=2_000,
        training_seed=30_201,
        amp_enabled=False,
    )

    assert result.arrays["factual_prediction"].shape == (2, 8, 8, 4, 192)
    assert result.arrays["no_action_prediction"].shape == (2, 8, 8, 4, 192)
    assert result.arrays["shuffled_action_prediction"].shape == (
        2,
        7,
        8,
        8,
        4,
        192,
    )
    assert result.arrays["hold_last_prediction"].shape == (2, 8, 8, 4, 192)
    assert result.arrays["constant_velocity_prediction"].shape == (
        2,
        8,
        8,
        4,
        192,
    )
    assert not bool(result.arrays["prediction_heads_add_sigreg"])
    assert float(result.arrays["anchor_encoder_sigreg"]) == SIGREG_WEIGHT
    np.testing.assert_allclose(
        result.arrays["no_action_prediction"],
        np.repeat(result.arrays["no_action_prediction"][:, :1], 8, axis=1),
        rtol=0,
        atol=1e-6,
    )
    slots = np.arange(8)
    for shift in range(1, 8):
        np.testing.assert_allclose(
            result.arrays["shuffled_action_prediction"][:, shift - 1],
            result.arrays["factual_prediction"][:, (slots + shift) % 8],
            rtol=0,
            atol=0,
        )


def test_fixed_latest_uses_existing_a4_checkpoint_schema() -> None:
    factual, no_action, _ = _build_pair()
    counts = {
        split: {task: 1 for task in TASKS}
        for split in ("train", "development", "test")
    }
    checkpoint = build_a4_latest_checkpoint(
        factual,
        no_action,
        seed=30_201,
        step=2_000,
        episode_counts=counts,
        bundle_counts=counts,
        cache_identifier="cache.npz",
        base_identifier="anchor_persistent@2000",
    )

    validated = validate_a4_checkpoint(checkpoint)
    assert validated["step"] == 2_000
    assert validated["seed"] == 30_201
    assert validated["bundle_batch_size"] == 28
    assert validated["sigreg_weight"] == 0.02
    assert set(validated["model_state"]) == set(factual.state_dict())
    assert set(validated["no_action_state"]) == set(no_action.state_dict())
