from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import fcpg.lewm_runtime as runtime
from fcpg.lewm_anchor_persistent import AnchorPersistentLeWM
from fcpg.lewm_grounded_model import GroundedLeWM
from fcpg.lewm_task_context_persistent import (
    TASK_CONTEXT_PERSISTENT_CHECKPOINT_TYPE,
    TASK_CONTEXT_PERSISTENT_SIGREG_WEIGHT,
    TASK_CONTEXT_PERSISTENT_VARIANT,
    TaskContextPersistentLeWM,
)


@dataclass(frozen=True)
class _VisionOutput:
    feature_map: torch.Tensor
    global_feature: torch.Tensor


class _TinyVision(nn.Module):
    output_dim = 8

    def __init__(self, **_: object) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, self.output_dim, kernel_size=1)

    def forward(self, pixels: torch.Tensor) -> _VisionOutput:
        feature_map = self.projection(pixels)
        return _VisionOutput(
            feature_map=feature_map,
            global_feature=feature_map.mean(dim=(-1, -2)),
        )


def _dataset() -> SimpleNamespace:
    return SimpleNamespace(
        proprio_dim=4,
        transition_action_dim=4,
        num_keypoints=3,
        history_size=3,
    )


def _runtime_config() -> dict[str, object]:
    return {
        "data": {
            "sources": ["unused.npz"],
            "history_size": 3,
            "rollout_steps": 1,
            "target_mode": "online",
        },
        "model": {
            "variant": TASK_CONTEXT_PERSISTENT_VARIANT,
            "vision": {
                "size": "tiny",
                "patch_size": 2,
                "image_size": 16,
                "pretrained": False,
                "use_mask_token": False,
            },
            "frame": {
                "latent_dim": 16,
                "depth": 1,
                "heads": 4,
                "mlp_ratio": 2,
                "dropout": 0.0,
            },
            "predictor": {
                "depth": 1,
                "heads": 4,
                "mlp_dim": 32,
                "dim_head": 4,
                "dropout": 0.0,
                "projector_hidden_dim": 32,
                "action_smoothing_dim": 4,
            },
            "sigreg": {"knots": 5, "num_proj": 8},
        },
        "loss": {
            "sigreg_weight": TASK_CONTEXT_PERSISTENT_SIGREG_WEIGHT,
            "per_source_sigreg": False,
            "alignment_weight": 0.0,
        },
    }


def _resolved_and_model_config() -> tuple[dict[str, object], dict[str, object]]:
    config = _runtime_config()
    resolved = runtime.resolve_runtime_config(config)
    model_config = runtime.resolve_model_config(resolved, _dataset())
    return resolved, model_config


def _build_task_model(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TaskContextPersistentLeWM, dict[str, object], dict[str, object]]:
    monkeypatch.setattr(runtime, "LeWMViTDenseEncoder", _TinyVision)
    resolved, model_config = _resolved_and_model_config()
    model = runtime._build_from_model_config(model_config)
    assert isinstance(model, TaskContextPersistentLeWM)
    return model, resolved, model_config


def _batch(batch_size: int = 4) -> dict[str, object]:
    frames = 4
    q_nominal = torch.rand(batch_size, frames, 3, 2) * 15.0
    return {
        "pixels": torch.rand(batch_size, frames, 3, 16, 16),
        "proprio": torch.randn(batch_size, frames, 4),
        "q_nominal": q_nominal,
        "q_corrected": q_nominal + 0.1 * torch.randn_like(q_nominal),
        "keypoints_xyz": torch.randn(batch_size, frames, 3, 3),
        "valid": torch.ones(batch_size, frames, 3, dtype=torch.bool),
        "action": torch.randn(batch_size, frames - 1, 4),
        "task_name": list(runtime.TASK_CONTEXT_PERSISTENT_TASK_NAMES[:batch_size]),
        "source_index": torch.arange(batch_size, dtype=torch.long) // 2,
        "episode_id": torch.arange(batch_size, dtype=torch.long),
    }


def test_model_config_records_frozen_a4_task_contract_and_builds_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, resolved, model_config = _build_task_model(monkeypatch)

    assert resolved["model"]["num_tasks"] == 7
    assert resolved["model"]["task_names"] == list(
        runtime.TASK_CONTEXT_PERSISTENT_TASK_NAMES
    )
    assert model_config["variant"] == TASK_CONTEXT_PERSISTENT_VARIANT
    assert model_config["world_model"]["num_tasks"] == 7
    assert model_config["task_names"] == list(
        runtime.TASK_CONTEXT_PERSISTENT_TASK_NAMES
    )
    assert model_config["sigreg_weight"] == 0.02
    assert model.num_tasks == 7


def test_legacy_summary_and_anchor_model_configs_stay_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "LeWMViTDenseEncoder", _TinyVision)
    _, task_config = _resolved_and_model_config()
    legacy = copy.deepcopy(task_config)
    legacy.pop("variant")
    legacy.pop("task_names")
    legacy.pop("sigreg_weight")
    legacy["world_model"].pop("num_tasks")

    summary = runtime._build_from_model_config(legacy)
    anchor = runtime._build_from_model_config(
        {**legacy, "variant": "anchor_persistent"}
    )

    assert isinstance(summary, GroundedLeWM)
    assert isinstance(anchor, AnchorPersistentLeWM)
    assert "task_names" not in summary.runtime_model_config
    assert "num_tasks" not in anchor.runtime_model_config["world_model"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("num_tasks", 6, "model.num_tasks must equal 7"),
        ("task_names", ["reach-v3"], "ordered A4 task panel"),
        ("sigreg_weight", 0.09, "loss.sigreg_weight must equal 0.02"),
    ],
)
def test_runtime_config_rejects_task_contract_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    config = _runtime_config()
    if field == "sigreg_weight":
        config["loss"][field] = value
    else:
        config["model"][field] = value

    with pytest.raises(ValueError, match=message):
        runtime.resolve_runtime_config(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("num_tasks", 8, "model_config.world_model.num_tasks must equal 7"),
        ("task_names", ["reach-v3"], "ordered A4 task panel"),
        ("sigreg_weight", 0.0, "model_config.sigreg_weight must equal 0.02"),
    ],
)
def test_model_config_rejects_task_contract_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    _, model_config = _resolved_and_model_config()
    if field == "num_tasks":
        model_config["world_model"][field] = value
    else:
        model_config[field] = value

    with pytest.raises(ValueError, match=message):
        runtime._validate_task_context_model_config(model_config)


def test_checkpoint_round_trip_saves_and_validates_task_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, resolved, model_config = _build_task_model(monkeypatch)
    checkpoint_path = tmp_path / "task_context.pt"
    checkpoint = runtime.save_lewm_checkpoint(
        checkpoint_path,
        model=model,
        resolved_config=resolved,
        data_config={"normalization": {}},
        model_config=model_config,
        step=1,
        epoch=1,
        run_kind="smoke",
    )

    assert checkpoint["checkpoint_type"] == TASK_CONTEXT_PERSISTENT_CHECKPOINT_TYPE
    assert checkpoint["model_config"]["world_model"]["num_tasks"] == 7
    assert checkpoint["model_config"]["task_names"] == list(
        runtime.TASK_CONTEXT_PERSISTENT_TASK_NAMES
    )
    assert checkpoint["model_config"]["sigreg_weight"] == 0.02

    loaded, loaded_config, loaded_checkpoint = runtime.load_lewm_grounded_checkpoint(
        checkpoint_path,
        device="cpu",
    )
    assert isinstance(loaded, TaskContextPersistentLeWM)
    assert loaded.training is False
    assert loaded.num_tasks == 7
    assert loaded_config["loss"]["sigreg_weight"] == 0.02
    assert loaded_checkpoint["checkpoint_type"] == (
        TASK_CONTEXT_PERSISTENT_CHECKPOINT_TYPE
    )

    bad_resolved = copy.deepcopy(resolved)
    bad_resolved["loss"]["sigreg_weight"] = 0.09
    with pytest.raises(ValueError, match="resolved_config.loss.sigreg_weight"):
        runtime.save_lewm_checkpoint(
            tmp_path / "bad_save.pt",
            model=model,
            resolved_config=bad_resolved,
            data_config={},
            model_config=model_config,
            step=1,
            epoch=1,
            run_kind="smoke",
        )

    tampered = dict(checkpoint)
    tampered["model_config"] = copy.deepcopy(checkpoint["model_config"])
    tampered["model_config"]["task_names"] = list(
        reversed(runtime.TASK_CONTEXT_PERSISTENT_TASK_NAMES)
    )
    tampered_path = tmp_path / "tampered.pt"
    torch.save(tampered, tampered_path)
    with pytest.raises(ValueError, match="ordered A4 task panel"):
        runtime.load_lewm_grounded_checkpoint(tampered_path, device="cpu")


def test_task_context_inference_maps_real_names_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _, _ = _build_task_model(monkeypatch)
    model.eval()
    task_names = runtime.TASK_CONTEXT_PERSISTENT_TASK_NAMES
    batch = {
        "action": torch.randn(2, 3, 4),
        "task_name": [task_names[0], task_names[-1]],
    }

    prepared = runtime._prepare_task_context_batch(model, batch)
    torch.testing.assert_close(prepared["task_index"], torch.tensor([0, 6]))
    embedding = torch.randn(2, 3, 4, 16)
    with torch.no_grad():
        prediction = runtime._predict_with_batch_context(
            model,
            embedding,
            batch["action"],
            {"task_name": batch["task_name"]},
        )
    assert prediction.shape == embedding.shape

    with torch.no_grad():
        explicit_prediction = runtime._predict_with_batch_context(
            model,
            embedding,
            batch["action"],
            {"task_index": torch.tensor([0, 6])},
        )
    torch.testing.assert_close(explicit_prediction, prediction)

    with pytest.raises(KeyError, match="requires task_index or exact A4 task_name"):
        runtime._predict_with_batch_context(
            model,
            embedding,
            batch["action"],
            {},
        )
    with pytest.raises(ValueError, match="outside the ordered A4 task panel"):
        runtime._prepare_task_context_batch(
            model,
            {"action": batch["action"], "task_name": [task_names[0], "unknown"]},
        )
    with pytest.raises(ValueError, match="conflicts with the ordered A4"):
        runtime._prepare_task_context_batch(
            model,
            {
                **batch,
                "task_index": torch.tensor([1, 6]),
            },
        )


def test_generic_evaluator_passes_bound_task_index_to_shuffled_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _, _ = _build_task_model(monkeypatch)

    metrics = runtime.evaluate_lewm(
        model,
        [_batch()],  # type: ignore[arg-type]
        torch.device("cpu"),
        sigreg_weight=0.02,
        per_source_sigreg=False,
        amp_enabled=False,
        amp_dtype=torch.bfloat16,
    )

    assert metrics["samples"] == 4.0
    assert metrics["action_shuffle_mse"] >= 0.0
