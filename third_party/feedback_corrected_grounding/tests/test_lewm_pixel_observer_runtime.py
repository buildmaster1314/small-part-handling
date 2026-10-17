from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import fcpg.lewm_pixel_observer_runtime as runtime
from fcpg.lewm_pixel_observer import (
    PIXEL_OBSERVER_CHECKPOINT_TYPE,
    LeWMPixelInnovationObserver,
    PixelResidualPriorAdapter,
)
from fcpg.lewm_provenance import sha256_file


def _base_payload(*, target_mode: str | None = "online") -> dict[str, object]:
    resolved_data: dict[str, object] = {"sources": []}
    data_config: dict[str, object] = {"sources": []}
    world_model: dict[str, object] = {}
    if target_mode is not None:
        resolved_data["target_mode"] = target_mode
        data_config["target_mode"] = target_mode
        world_model["target_mode"] = target_mode
    return {
        "resolved_config": {"data": resolved_data},
        "data_config": data_config,
        "model_config": {"world_model": world_model},
    }


def test_base_supervision_audit_accepts_explicit_and_oracle_free_legacy() -> None:
    explicit = runtime._base_supervision_audit(_base_payload(target_mode="online"))
    assert explicit["passed"] is True
    assert explicit["target"] == "online_self_target"
    assert explicit["evidence"] == "explicit"
    assert len(explicit["mode_records"]) == 3

    legacy = runtime._base_supervision_audit(_base_payload(target_mode=None))
    assert legacy["passed"] is True
    assert legacy["target"] == "online_self_target"
    assert legacy["evidence"] == "legacy_absence_of_oracle_artifacts"
    assert legacy["mode_records"] == []


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda payload: payload["model_config"]["world_model"].update(
                target_mode="frozen_oracle"
            ),
            "oracle-free",
        ),
        (
            lambda payload: payload["data_config"]["sources"].append(
                {"oracle_target_cache": "/tmp/oracle"}
            ),
            "oracle-free",
        ),
        (
            lambda payload: payload["data_config"].update(
                target_contract={"q_true_usage": "target_construction_only"}
            ),
            "oracle-free",
        ),
        (
            lambda payload: payload["resolved_config"]["data"].update(
                target_mode="unknown_target"
            ),
            "unsupported target modes",
        ),
    ],
)
def test_base_supervision_audit_fails_closed(mutate, match: str) -> None:
    payload = _base_payload(target_mode="online")
    mutate(payload)
    with pytest.raises(ValueError, match=match):
        runtime._base_supervision_audit(payload)


def _minimal_config(tmp_path: Path) -> dict[str, object]:
    return {
        "seed": 9,
        "output_dir": str(tmp_path / "pixel_observer"),
        "base_checkpoint": str(tmp_path / "base.pt"),
        "data": {
            "sources": [
                {
                    "name": "tiny",
                    "path": str(tmp_path / "tiny.npz"),
                    "correction_cache": str(tmp_path / "q_corrected.npy"),
                }
            ],
            "history_size": 2,
            "rollout_steps": 1,
            "frame_skip": 1,
            "grounding_mode": "fcpg",
            "target_mode": "online",
            "val_fraction": 0.25,
        },
        "loader": {
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        },
        "adapter": {
            "hidden_dim": 8,
            "max_delta_px": 4.0,
            "residual_scale_px": 4.0,
        },
        "observer": {"gain": 0.5},
        "trainer": {"epochs": 1, "max_steps": 1},
    }


def test_pixel_observer_config_is_oracle_free_and_output_is_independent(
    tmp_path: Path,
) -> None:
    config = _minimal_config(tmp_path)
    resolved = runtime.resolve_pixel_observer_config(config)
    assert resolved["data"]["target_mode"] == "online"
    assert resolved["data"]["grounding_mode"] == "fcpg"
    assert Path(resolved["output_dir"]).is_absolute()

    oracle = copy.deepcopy(config)
    oracle["data"]["sources"][0]["oracle_target_cache"] = str(tmp_path / "oracle")
    with pytest.raises(ValueError, match="oracle_target_cache is forbidden"):
        runtime.resolve_pixel_observer_config(oracle)

    q_true = copy.deepcopy(config)
    q_true["contract"] = {"q_true": "supervision"}
    with pytest.raises(ValueError, match="oracle field is forbidden"):
        runtime.resolve_pixel_observer_config(q_true)

    no_cache = copy.deepcopy(config)
    no_cache["data"]["sources"][0].pop("correction_cache")
    with pytest.raises(ValueError, match="requires correction_cache"):
        runtime.resolve_pixel_observer_config(no_cache)

    legacy_output = copy.deepcopy(config)
    legacy_output["output_dir"] = str(tmp_path / "outputs" / "gate_l1_fcpg")
    with pytest.raises(ValueError, match="legacy Gate L1"):
        runtime.resolve_pixel_observer_config(legacy_output)


def test_masked_mse_and_metrics_reject_empty_masks() -> None:
    prediction = torch.zeros(1, 2, 2)
    target = torch.ones_like(prediction)
    valid = torch.zeros(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="no valid target anchors"):
        runtime._masked_mse(prediction, target, valid)

    output = runtime.PixelObserverLoss(
        loss=torch.tensor(0.0),
        hold_last_loss=torch.tensor(0.0),
        prior_residual_px=prediction,
        target_residual_px=target,
        valid=valid,
    )
    with pytest.raises(ValueError, match="no valid target anchors"):
        runtime._observer_metrics(output)


def test_masked_mse_ignores_invalid_nan_but_rejects_selected_nan() -> None:
    prediction = torch.zeros(1, 2, 2)
    target = torch.tensor([[[1.0, 1.0], [float("nan"), float("nan")]]])
    valid = torch.tensor([[True, False]])
    assert float(runtime._masked_mse(prediction, target, valid)) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="selected target values must be finite"):
        runtime._masked_mse(prediction, target, torch.ones_like(valid))


class _LossBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.history_size = 2
        self.latent_dim = 2
        self.action_dim = 1
        self.frame_encoder = SimpleNamespace(num_keypoints=1)
        self.seen_q: torch.Tensor | None = None

    def encode_frames(self, batch):
        self.seen_q = batch["q_corrected"].detach().clone()
        embedding = batch["q_corrected"].mean(dim=2)
        return embedding, embedding

    def predict(self, embedding, actions):
        assert actions.shape[:2] == embedding.shape[:2]
        return embedding


def test_training_loss_uses_last_valid_history_posterior() -> None:
    base = _LossBase()
    adapter = PixelResidualPriorAdapter(
        latent_dim=2,
        num_keypoints=1,
        image_size=(8, 8),
        hidden_dim=4,
        max_delta_px=4.0,
        residual_scale_px=4.0,
    )
    observer = LeWMPixelInnovationObserver(base, adapter, freeze_base=True).eval()
    q_nominal = torch.zeros(1, 3, 1, 2)
    q_corrected = torch.tensor(
        [[[[2.0, 3.0]], [[float("nan"), float("nan")]], [[4.0, 5.0]]]]
    )
    batch = {
        "q_nominal": q_nominal,
        "q_corrected": q_corrected,
        "keypoints_xyz": torch.zeros(1, 3, 1, 3),
        "valid": torch.tensor([[[True], [False], [True]]]),
        "valid_nominal": torch.ones(1, 3, 1, dtype=torch.bool),
        "action": torch.zeros(1, 2, 1),
    }

    output = runtime.pixel_observer_loss(observer, batch)

    expected = torch.tensor([[[2.0, 3.0]]])
    torch.testing.assert_close(output.prior_residual_px, expected)
    assert base.seen_q is not None
    torch.testing.assert_close(base.seen_q[:, 1], expected)
    assert bool(torch.isfinite(output.loss))
    assert bool(torch.isfinite(output.hold_last_loss))


class _EvalOnlyObserver:
    def eval(self) -> "_EvalOnlyObserver":
        return self


def test_validation_metrics_weight_valid_coordinates_not_batch_size(monkeypatch) -> None:
    def fake_loss(_observer, batch):
        case = int(batch["case"].item())
        if case == 1:
            prediction = torch.full((1, 1, 2), 10.0)
            target = torch.zeros_like(prediction)
            valid = torch.ones(1, 1, dtype=torch.bool)
            hold_last_loss = torch.tensor(4.0)
        else:
            prediction = torch.zeros(4, 2, 2)
            target = torch.zeros_like(prediction)
            valid = torch.ones(4, 2, dtype=torch.bool)
            hold_last_loss = torch.tensor(1.0)
        return runtime.PixelObserverLoss(
            loss=runtime._masked_mse(prediction, target, valid),
            hold_last_loss=hold_last_loss,
            prior_residual_px=prediction,
            target_residual_px=target,
            valid=valid,
        )

    monkeypatch.setattr(runtime, "pixel_observer_loss", fake_loss)
    metrics = runtime.evaluate_pixel_observer(
        _EvalOnlyObserver(),
        [{"case": torch.tensor(1)}, {"case": torch.tensor(2)}],
        torch.device("cpu"),
        amp_enabled=False,
        amp_dtype=torch.bfloat16,
    )

    # Prior SSE is 200 over 18 valid coordinates.  A batch-size weighted mean
    # would instead report 20 and is intentionally ruled out by this fixture.
    assert metrics["prior_mse"] == pytest.approx(200.0 / 18.0)
    assert metrics["hold_last_mse"] == pytest.approx(24.0 / 18.0)
    assert metrics["prior_l2_px"] == pytest.approx((200.0**0.5) / 9.0)
    assert metrics["samples"] == 5.0
    assert metrics["valid_anchors"] == 9.0


class _BoundBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.history_size = 1
        self.latent_dim = 2
        self.action_dim = 1
        self.frame_encoder = SimpleNamespace(num_keypoints=1)


def _write_bound_adapter_checkpoint(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    resolved = {"observer": {"gain": 0.5}}
    snapshot = tmp_path / "resolved_config.json"
    snapshot.write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    origin = tmp_path / "origin.yaml"
    origin.write_text("observer:\n  gain: 0.5\n", encoding="utf-8")
    base_path = tmp_path / "base.pt"
    base_payload = _base_payload(target_mode="online")
    torch.save(base_payload, base_path)

    adapter = PixelResidualPriorAdapter(
        latent_dim=2,
        num_keypoints=1,
        image_size=(4, 4),
        hidden_dim=4,
        max_delta_px=2.0,
        residual_scale_px=2.0,
    )
    adapter_config = {
        "latent_dim": 2,
        "num_keypoints": 1,
        "image_size": [4, 4],
        "hidden_dim": 4,
        "max_delta_px": 2.0,
        "residual_scale_px": 2.0,
        "xyz_mean": [0.0, 0.0, 0.0],
        "xyz_std": [1.0, 1.0, 1.0],
    }
    data_config: dict[str, object] = {}
    checkpoint = {
        "schema_version": 1,
        "checkpoint_type": PIXEL_OBSERVER_CHECKPOINT_TYPE,
        "resolved_config": resolved,
        "config_binding": runtime._config_binding(resolved, snapshot, origin),
        "data_config": data_config,
        "data_config_sha256": runtime._canonical_sha256(data_config),
        "data_artifacts": [],
        "base_checkpoint": {
            "path": str(base_path.resolve()),
            "sha256": sha256_file(base_path),
            "checkpoint_type": base_payload.get("checkpoint_type"),
            "step": int(base_payload.get("step", 0)),
            "run_kind": base_payload.get("run_kind"),
            "supervision_audit": runtime._base_supervision_audit(base_payload),
            "training_budget": base_payload.get("training_budget"),
            "provenance": base_payload.get("provenance"),
        },
        "adapter_config": adapter_config,
        "adapter_state": adapter.state_dict(),
    }
    checkpoint_path = tmp_path / "adapter.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path, base_path, snapshot, origin


def test_checkpoint_binds_base_and_frozen_config_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint_path, base_path, snapshot, origin = _write_bound_adapter_checkpoint(
        tmp_path
    )
    base_payload = torch.load(base_path, map_location="cpu", weights_only=False)

    def load_base(*_args, **_kwargs):
        return _BoundBase(), {}, base_payload

    monkeypatch.setattr(runtime, "load_lewm_grounded_checkpoint", load_base)

    # The origin is provenance-at-start, not a mutable runtime dependency.
    origin.write_text("observer:\n  gain: 0.75\n", encoding="utf-8")
    observer, resolved, checkpoint = runtime.load_pixel_observer_checkpoint(
        checkpoint_path, device="cpu"
    )
    assert resolved["observer"]["gain"] == 0.5
    assert checkpoint["config_binding"]["origin_sha256_at_start"] != sha256_file(
        origin
    )
    assert all(not parameter.requires_grad for parameter in observer.base_model.parameters())

    adapter_values = dict(checkpoint["adapter_config"])
    xyz_mean = adapter_values.pop("xyz_mean")
    xyz_std = adapter_values.pop("xyz_std")
    adapter_values["image_size"] = tuple(adapter_values["image_size"])
    reference = PixelResidualPriorAdapter(**adapter_values)
    reference.set_geometry_normalization(xyz_mean=xyz_mean, xyz_std=xyz_std)
    reference.load_state_dict(checkpoint["adapter_state"], strict=True)
    inputs = (
        torch.tensor([[0.2, -0.1]]),
        torch.tensor([[[1.0, 2.0]]]),
        torch.tensor([[[2.0, 1.0]]]),
        torch.tensor([[[0.1, 0.2, 0.3]]]),
        torch.tensor([[True]]),
    )
    torch.testing.assert_close(reference(*inputs), observer.adapter(*inputs))

    snapshot.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot changed"):
        runtime.load_pixel_observer_checkpoint(checkpoint_path, device="cpu")

    snapshot.write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    torch.save({"changed": True}, base_path)
    with pytest.raises(ValueError, match="SHA-256 differs"):
        runtime.load_pixel_observer_checkpoint(checkpoint_path, device="cpu")


def test_checkpoint_rejects_tampered_embedded_config(tmp_path: Path) -> None:
    checkpoint_path, _, _, _ = _write_bound_adapter_checkpoint(tmp_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["resolved_config"]["observer"]["gain"] = 0.25
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="resolved config binding is invalid"):
        runtime.load_pixel_observer_checkpoint(
            checkpoint_path, device="cpu", verify_artifacts=False
        )


def test_checkpoint_rejects_forged_base_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path, base_path, _, _ = _write_bound_adapter_checkpoint(tmp_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["base_checkpoint"]["step"] = 10_000
    checkpoint["base_checkpoint"]["run_kind"] = "formal"
    checkpoint["base_checkpoint"]["training_budget"] = {
        "target_optimizer_steps": 10_000,
        "checkpoint_optimizer_step": 10_000,
        "run_completed": True,
        "final_optimizer_steps": 10_000,
    }
    torch.save(checkpoint, checkpoint_path)
    base_payload = torch.load(base_path, map_location="cpu", weights_only=False)
    monkeypatch.setattr(
        runtime,
        "load_lewm_grounded_checkpoint",
        lambda *_args, **_kwargs: (_BoundBase(), {}, base_payload),
    )

    with pytest.raises(ValueError, match="actual base.*differs"):
        runtime.load_pixel_observer_checkpoint(checkpoint_path, device="cpu")


def test_artifact_binding_detects_cache_or_data_mutation(tmp_path: Path) -> None:
    artifact = tmp_path / "q_corrected.npy"
    artifact.write_bytes(b"frozen-correction")
    records = [
        {
            "kind": "q_corrected",
            "source": "tiny",
            "path": str(artifact),
            "sha256": sha256_file(artifact),
        }
    ]
    runtime._verify_artifacts(records)
    artifact.write_bytes(b"mutated-correction")
    with pytest.raises(ValueError, match="bound artifact changed"):
        runtime._verify_artifacts(records)


def test_training_budget_certificate_rejects_incomplete_run(tmp_path: Path) -> None:
    checkpoint = tmp_path / "budget.pt"
    torch.save(
        {
            "training_budget": {
                "target_optimizer_steps": 2,
                "checkpoint_optimizer_step": 1,
                "run_completed": False,
                "final_optimizer_steps": None,
            }
        },
        checkpoint,
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        runtime._certify_completed_budget(
            checkpoint,
            final_optimizer_steps=1,
            target_optimizer_steps=2,
        )

    runtime._certify_completed_budget(
        checkpoint,
        final_optimizer_steps=2,
        target_optimizer_steps=2,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["training_budget"]["run_completed"] is True
    assert payload["training_budget"]["final_optimizer_steps"] == 2


def test_formal_base_contract_requires_online_completed_endpoint() -> None:
    resolved = {
        "contract": {
            "require_formal_base": True,
            "require_explicit_online_base": True,
            "require_completed_base_budget": True,
            "expected_base_step": 10_000,
        }
    }
    checkpoint = {
        "run_kind": "formal",
        "step": 10_000,
        "training_budget": {
            "target_optimizer_steps": 10_000,
            "checkpoint_optimizer_step": 10_000,
            "run_completed": True,
            "final_optimizer_steps": 10_000,
        },
    }
    supervision = {"evidence": "explicit"}
    runtime._validate_base_checkpoint_contract(resolved, checkpoint, supervision)

    for key, value, match in (
        ("run_kind", "smoke", "formal base"),
        ("step", 9_999, "step differs"),
    ):
        invalid = copy.deepcopy(checkpoint)
        invalid[key] = value
        with pytest.raises(ValueError, match=match):
            runtime._validate_base_checkpoint_contract(
                resolved, invalid, supervision
            )
    with pytest.raises(ValueError, match="explicit online"):
        runtime._validate_base_checkpoint_contract(
            resolved,
            checkpoint,
            {"evidence": "legacy_absence_of_oracle_artifacts"},
        )
    incomplete = copy.deepcopy(checkpoint)
    incomplete["training_budget"]["run_completed"] = False
    with pytest.raises(ValueError, match="valid completed training budget"):
        runtime._validate_base_checkpoint_contract(
            resolved, incomplete, supervision
        )

    malformed = copy.deepcopy(checkpoint)
    malformed["training_budget"].update(
        target_optimizer_steps=5_000,
        final_optimizer_steps=5_000,
    )
    with pytest.raises(ValueError, match="valid completed training budget"):
        runtime._validate_base_checkpoint_contract(
            resolved, malformed, supervision
        )

    missing_final = copy.deepcopy(checkpoint)
    missing_final["training_budget"]["final_optimizer_steps"] = None
    with pytest.raises(ValueError, match="valid completed training budget"):
        runtime._validate_base_checkpoint_contract(
            resolved, missing_final, supervision
        )
