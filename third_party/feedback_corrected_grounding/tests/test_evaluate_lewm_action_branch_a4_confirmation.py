from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import scripts.evaluate_lewm_action_branch_a4_confirmation as evaluator
from fcpg.lewm_action_branch_a4 import CONFIRMATION_TRAINING_SEEDS, TASKS
from fcpg.lewm_action_branch_a4_training import (
    A4_LATEST_CHECKPOINT_NAME,
    A4_RESOLVED_CONFIG_NAME,
    build_a4_latest_checkpoint,
    build_matched_a4_models,
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


def _task_model(_: object | None = None) -> TaskContextPersistentLeWM:
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


def _base_checkpoint() -> dict:
    anchor = _anchor_model()
    return {
        "checkpoint_type": ANCHOR_PERSISTENT_CHECKPOINT_TYPE,
        "step": 2_000,
        "training_budget": {
            "target_optimizer_steps": 2_000,
            "checkpoint_optimizer_step": 2_000,
            "run_completed": True,
        },
        "resolved_config": {"loss": {"sigreg_weight": 0.02}},
        "model_config": {
            "variant": ANCHOR_PERSISTENT_VARIANT,
            "frame": {"latent_dim": 192, "num_keypoints": 3},
            "world_model": {"action_dim": 20, "history_size": 3},
        },
        "model_state": anchor.state_dict(),
    }


def _counts(train_count: int = 1) -> dict[str, dict[str, int]]:
    return {
        split: {task: train_count if split == "train" else 0 for task in TASKS}
        for split in ("train", "development", "test")
    }


def _write_endpoint_panel(tmp_path: Path) -> tuple[Path, Path, list[Path]]:
    base_path = tmp_path / "base.pt"
    base = _base_checkpoint()
    torch.save(base, base_path)
    train_cache = tmp_path / "formal_train_cache.npz"
    train_cache.write_bytes(b"bound-train-cache")
    endpoints: list[Path] = []
    for seed in CONFIRMATION_TRAINING_SEEDS:
        factual, no_action, _ = build_matched_a4_models(
            base,
            seed=seed,
            model_builder=_task_model,
        )
        run_dir = tmp_path / f"seed{seed}"
        run_dir.mkdir()
        checkpoint_path = run_dir / A4_LATEST_CHECKPOINT_NAME
        checkpoint = build_a4_latest_checkpoint(
            factual,
            no_action,
            seed=seed,
            step=10_000,
            episode_counts=_counts(),
            bundle_counts=_counts(),
            cache_identifier=str(train_cache.resolve()),
            base_identifier=str(base_path.resolve()),
        )
        torch.save(checkpoint, checkpoint_path)
        resolved = {
            "protocol": evaluator.PROTOCOL,
            "budget": "10k",
            "seed": seed,
            "optimizer_steps": 10_000,
            "train_split": "train",
            "evaluation_split": None,
            "selection": "fixed_latest",
            "input_npz": str(train_cache.resolve()),
            "base_checkpoint": str(base_path.resolve()),
        }
        (run_dir / A4_RESOLVED_CONFIG_NAME).write_text(
            json.dumps(resolved), encoding="utf-8"
        )
        endpoints.append(checkpoint_path)
    return base_path, train_cache, endpoints


def test_prepare_requires_three_real_fixed_latest_double_state_endpoints(
    tmp_path: Path,
) -> None:
    base, train_cache, endpoints = _write_endpoint_panel(tmp_path)

    prepared = evaluator.prepare_confirmation_checkpoints(
        endpoints[::-1],
        base_checkpoint_path=base,
        formal_train_cache_path=str(train_cache),
        model_builder=_task_model,
    )

    assert tuple(item.seed for item in prepared) == CONFIRMATION_TRAINING_SEEDS
    assert all(item.step == 10_000 for item in prepared)
    assert all(item.selection == "fixed_latest" for item in prepared)
    assert all(len(item.sha256) == 64 for item in prepared)
    assert all(item.factual.state_dict() for item in prepared)
    assert all(item.no_action.state_dict() for item in prepared)

    best_path = endpoints[0].with_name("best.pt")
    best_path.write_bytes(endpoints[0].read_bytes())
    with pytest.raises(ValueError, match="fixed-latest checkpoint filename"):
        evaluator.prepare_confirmation_checkpoints(
            [best_path, *endpoints[1:]],
            base_checkpoint_path=base,
            formal_train_cache_path=str(train_cache),
            model_builder=_task_model,
        )


def test_formal_train_cache_binding_rehashes_file_content(tmp_path: Path) -> None:
    cache = tmp_path / "formal.npz"
    cache.write_bytes(b"formal-cache")
    summary = {
        "formal_train_cache": {
            "path": str(cache.resolve()),
            "sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        }
    }
    assert evaluator._formal_train_cache_path(summary) == str(cache.resolve())

    cache.write_bytes(b"changed")
    with pytest.raises(PermissionError, match="no longer matches"):
        evaluator._formal_train_cache_path(summary)


def test_prediction_identity_is_derived_from_prepared_checkpoint(tmp_path: Path) -> None:
    model = _task_model()
    checkpoint = evaluator.PreparedCheckpoint(
        path=(tmp_path / A4_LATEST_CHECKPOINT_NAME).resolve(),
        sha256="a" * 64,
        seed=30_301,
        step=10_000,
        selection="fixed_latest",
        factual=model,
        no_action=_task_model(),
    )
    arrays = evaluator._attach_checkpoint_identity(
        {"protocol": np.asarray(evaluator.PROTOCOL)}, checkpoint
    )

    assert arrays["checkpoint_path"].item() == str(checkpoint.path)
    assert arrays["checkpoint_sha256"].item() == "a" * 64
    assert int(arrays["checkpoint_seed"]) == 30_301
    assert int(arrays["checkpoint_step"]) == 10_000
    assert arrays["selection"].item() == "fixed_latest"


def test_invalid_development_gate_stops_before_checkpoint_and_test_reads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(evaluator, "_load_json", lambda *args, **kwargs: {})

    def reject(*args: object, **kwargs: object) -> None:
        calls.append("authorization")
        raise PermissionError("not Development-Go")

    monkeypatch.setattr(evaluator, "validate_a4_development_go_authorization", reject)
    monkeypatch.setattr(
        evaluator,
        "prepare_confirmation_checkpoints",
        lambda *args, **kwargs: calls.append("checkpoint"),
    )
    monkeypatch.setattr(
        evaluator,
        "_load_test_only_cache",
        lambda *args, **kwargs: calls.append("test"),
    )

    with pytest.raises(PermissionError, match="not Development-Go"):
        evaluator.main(
            [
                "--development-go-summary",
                str(tmp_path / "go.json"),
                "--checkpoints",
                str(tmp_path / "one.pt"),
                str(tmp_path / "two.pt"),
                str(tmp_path / "three.pt"),
                "--test-cache",
                str(tmp_path / "test.npz"),
            ]
        )
    assert calls == ["authorization"]


def test_test_cache_must_be_test_only_and_cover_seven_tasks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        evaluator,
        "load_a4_state_cache",
        lambda _: {
            "split": np.asarray(["test"] * len(TASKS)),
            "task_name": np.asarray(TASKS),
        },
    )
    cache = evaluator._load_test_only_cache(tmp_path / "test.npz")
    assert set(cache["task_name"].tolist()) == set(TASKS)

    monkeypatch.setattr(
        evaluator,
        "load_a4_state_cache",
        lambda _: {
            "split": np.asarray(["development"] * len(TASKS)),
            "task_name": np.asarray(TASKS),
        },
    )
    with pytest.raises(PermissionError, match="development rows"):
        evaluator._load_test_only_cache(tmp_path / "test.npz")
