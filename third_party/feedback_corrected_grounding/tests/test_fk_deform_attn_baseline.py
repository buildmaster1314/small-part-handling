from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from fcpg.action_chunk_policy import ActionChunkOutput
from fcpg.fk_deform_attn_baseline import (
    AUDITED_VISION_MODEL,
    AUDITED_VISION_SELECTED_LAYERS,
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_TYPE,
    DEVELOPMENT_CLAIM_SCOPE,
    LOSS_CONTRACT,
    SELECTION_POLICY,
    STRUCTURAL_ATTRIBUTION_CONTRACT,
    FKPolicyReadout,
    MatchedPolicyOnlyGrounding,
    attention_barycenter_epe,
    benchmark_interleaved_latency,
    make_fixed_stencil,
    validate_structural_pair,
)
from fcpg.modules import GroundedTokenAdapter
from scripts.train_fk_deform_attn_policy import (
    _single_item,
    policy_batch,
    validate_vision_cache_metadata,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


EXPECTED_LATTICE = torch.tensor(
    (
        (-1.0, -1.0),
        (0.0, -1.0),
        (1.0, -1.0),
        (-1.0, 0.0),
        (1.0, 0.0),
        (-1.0, 1.0),
        (0.0, 1.0),
        (1.0, 1.0),
    )
)


def make_readout(variant: str, *, dropout: float = 0.0) -> FKPolicyReadout:
    return FKPolicyReadout(
        variant=variant,
        visual_dim=4,
        obs_dim=5,
        num_keypoints=3,
        model_dim=16,
        num_samples=8,
        context_grid_size=3,
        context_radius_px=2.0,
        offset_scale_px=20.0,
        local_layers=1,
        global_layers=1,
        heads=2,
        dropout=dropout,
        lattice_step_px=1.0,
    )


@pytest.mark.parametrize("variant", ("policy_only_residual", "fk_deform_attn"))
def test_fixed_lattice_initialization_and_weighted_sampling(variant: str) -> None:
    torch.manual_seed(7)
    readout = make_readout(variant).eval()
    feature_map = torch.randn(2, 4, 8, 8)
    nominal = torch.tensor(
        [[[7.0, 7.0], [8.0, 8.0], [9.0, 9.0]]] * 2,
    )
    xyz = torch.randn(2, 3, 3)
    obs = torch.randn(2, 5)

    output = readout(feature_map, nominal, xyz, obs, (16, 16))

    expected = EXPECTED_LATTICE.view(1, 1, 8, 2).expand_as(output.offsets_px)
    torch.testing.assert_close(output.offsets_px, expected, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        output.attention_weights,
        torch.full_like(output.attention_weights, 1.0 / 8.0),
    )
    assert output.weighted_visual.shape == (2, 3, 4)
    assert output.sample_points_px.shape == (2, 3, 8, 2)
    torch.testing.assert_close(output.attention_barycenter_px, nominal)
    torch.testing.assert_close(
        attention_barycenter_epe(output, nominal),
        torch.zeros(()),
        atol=1e-6,
        rtol=0.0,
    )


def test_deform_head_is_independent_and_capacity_is_matched_without_padding() -> None:
    capacity_kwargs = {
        "visual_dim": 4,
        "obs_dim": 5,
        "num_keypoints": 3,
        "model_dim": 64,
        "num_samples": 8,
        "context_grid_size": 3,
        "context_radius_px": 2.0,
        "local_layers": 2,
        "global_layers": 1,
        "heads": 8,
        "dropout": 0.0,
    }
    residual = FKPolicyReadout(
        variant="policy_only_residual",
        **capacity_kwargs,
    )
    deform = FKPolicyReadout(
        variant="fk_deform_attn",
        **capacity_kwargs,
    )
    assert torch.equal(make_fixed_stencil(8), EXPECTED_LATTICE)
    assert deform.output_head[-1].out_features == 8 * 3
    assert residual.output_head[-1].out_features == 2 + 8
    residual_active = sum(parameter.numel() for parameter in residual.parameters())
    deform_active = sum(parameter.numel() for parameter in deform.parameters())
    relative_gap = abs(residual_active - deform_active) / max(
        residual_active,
        deform_active,
    )
    assert relative_gap < 0.01
    assert not any("padding" in name for name, _ in deform.named_parameters())


class TinyVision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Conv2d(3, 3, kernel_size=1)
        self.feature_projections = nn.ModuleList(
            [nn.Conv2d(3, 4, kernel_size=1)]
        )
        self.fusion = nn.Conv2d(4, 4, kernel_size=1)
        self.global_projection = nn.Linear(3, 4)

    def project_backbone_features(
        self,
        feature_maps: torch.Tensor,
        prefix: torch.Tensor,
    ) -> SimpleNamespace:
        feature_map = self.fusion(self.feature_projections[0](feature_maps[:, 0]))
        return SimpleNamespace(
            feature_map=feature_map,
            global_feature=self.global_projection(prefix),
        )

    def forward(self, rgb: torch.Tensor) -> SimpleNamespace:
        backbone_features = self.backbone(rgb)
        return self.project_backbone_features(
            backbone_features.unsqueeze(1),
            backbone_features.mean(dim=(-2, -1)),
        )


class TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_keypoints = 3
        self.obs_dim = 5
        self.vision_dim = 4
        self.action_dim = 4
        self.obs_horizon = 2
        self.action_horizon = 4
        self.model_dim = 8
        self.vision_model_name = "tiny"
        self.vision_selected_layers = (0,)
        self.policy_arch = "legacy"
        self.mode = "geoprop"
        self.fcpg_adapter = None
        self.next_latent_predictor = None
        self.vision_encoder = TinyVision()
        self.grounding_adapter = GroundedTokenAdapter(
            visual_dim=4,
            proprio_dim=6,
            token_dim=8,
            hidden_dim=8,
            gate_visual_only=True,
            include_reliability=True,
        )
        self.action_head = nn.Linear(8, self.action_horizon * self.action_dim)

    def set_normalization(self, values: dict[str, torch.Tensor]) -> None:
        del values

    def forward(self, batch: dict[str, torch.Tensor]) -> ActionChunkOutput:
        batch_size, horizon, keypoints, _ = batch["sampled_visual"].shape
        xyz = batch["keypoints_xyz"].reshape(batch_size * horizon, keypoints, 3)
        identity = torch.eye(keypoints, dtype=xyz.dtype, device=xyz.device)
        identity = identity.view(1, keypoints, keypoints).expand(
            batch_size * horizon,
            -1,
            -1,
        )
        proprio = torch.cat((xyz, identity), dim=-1)
        sampled = batch["sampled_visual"].reshape(
            batch_size * horizon,
            keypoints,
            self.vision_dim,
        )
        grounded = self.grounding_adapter.forward_sampled(
            sampled,
            proprio,
            reliability=torch.ones(
                batch_size * horizon,
                keypoints,
                dtype=sampled.dtype,
                device=sampled.device,
            ),
        )
        latent = grounded.reshape(batch_size, horizon * keypoints, -1).mean(dim=1)
        raw = self.action_head(latent).reshape(
            batch_size,
            self.action_horizon,
            self.action_dim,
        )
        motion = torch.tanh(raw[..., :3])
        gripper_logit = raw[..., 3]
        action = torch.cat((motion, torch.sigmoid(gripper_logit).unsqueeze(-1)), dim=-1)
        return ActionChunkOutput(
            action=action,
            motion=motion,
            gripper_logit=gripper_logit,
            latent=latent,
            grounding_points_px=batch["q_nominal"],
            reliability=torch.ones_like(batch["q_nominal"][..., 0]),
        )


def make_training_batch() -> dict[str, torch.Tensor]:
    nominal = torch.tensor(
        [[[[7.0, 7.0], [8.0, 8.0], [9.0, 9.0]]] * 2],
    )
    return {
        "rgb": torch.randn(1, 2, 3, 16, 16),
        "obs": torch.randn(1, 2, 5),
        "keypoints_xyz": torch.randn(1, 2, 3, 3),
        "q_nominal": nominal,
        "q_true": nominal + 100.0,
        "delta_q": torch.randn(1, 2, 3, 2),
        "action": torch.randn(1, 4, 4),
        "action_mask": torch.ones(1, 4),
    }


@pytest.mark.parametrize("variant", ("policy_only_residual", "fk_deform_attn"))
def test_projection_target_perturbation_cannot_change_bc_loss_or_gradients(
    variant: str,
) -> None:
    torch.manual_seed(11)
    model = MatchedPolicyOnlyGrounding(TinyPolicy(), make_readout(variant)).eval()
    batch = make_training_batch()

    def loss_and_gradients(target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        model.zero_grad(set_to_none=True)
        current = dict(batch)
        current["q_true"] = target.requires_grad_(True)
        loss, parts, _ = model.compute_bc_loss(current)
        assert set(parts) == {"motion_loss", "gripper_loss"}
        loss.backward()
        assert current["q_true"].grad is None
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        }
        return loss.detach().clone(), gradients

    first_loss, first_gradients = loss_and_gradients(batch["q_true"].clone())
    second_loss, second_gradients = loss_and_gradients(
        torch.randn_like(batch["q_true"]) * 10000.0,
    )
    torch.testing.assert_close(first_loss, second_loss, atol=0.0, rtol=0.0)
    assert first_gradients.keys() == second_gradients.keys()
    for name in first_gradients:
        torch.testing.assert_close(
            first_gradients[name],
            second_gradients[name],
            atol=0.0,
            rtol=0.0,
        )


def test_training_batch_filters_projection_targets() -> None:
    selected = policy_batch(make_training_batch(), torch.device("cpu"))
    assert "q_true" not in selected
    assert "delta_q" not in selected
    assert set(selected) == {
        "rgb",
        "obs",
        "keypoints_xyz",
        "q_nominal",
        "action",
        "action_mask",
    }


def test_single_item_slices_and_copies_on_cpu_before_device_transfer() -> None:
    one = make_training_batch()
    batch = {
        key: value.expand(4, *value.shape[1:]).clone()
        for key, value in one.items()
    }

    selected = _single_item(batch, torch.device("cpu"))

    assert "action" not in selected
    assert "action_mask" not in selected
    for value in selected.values():
        assert value.shape[0] == 1
        assert value.untyped_storage().nbytes() == value.numel() * value.element_size()


def _write_vision_cache(tmp_path, *, sample_count: int = 5):
    source = tmp_path / "source.npz"
    np.savez(source, obs=np.zeros((sample_count, 3), dtype=np.float32))
    cache = tmp_path / "vision"
    cache.mkdir()
    np.save(
        cache / "backbone_features.npy",
        np.zeros((sample_count, 1, 4, 2, 2), dtype=np.float16),
    )
    np.save(
        cache / "prefix.npy",
        np.zeros((sample_count, 4), dtype=np.float16),
    )
    metadata = {
        "source_data": str(source.resolve()),
        "sample_count": sample_count,
        "model_name": AUDITED_VISION_MODEL,
        "selected_layers": [11],
        "feature_shape": [1, 4, 2, 2],
        "prefix_dim": 4,
    }
    (cache / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return source, cache, metadata


def test_vision_cache_metadata_is_validated_against_audited_source(tmp_path) -> None:
    source, cache, _ = _write_vision_cache(tmp_path)

    summary = validate_vision_cache_metadata(source, cache)

    assert summary["model_name"] == AUDITED_VISION_MODEL
    assert summary["selected_layers"] == AUDITED_VISION_SELECTED_LAYERS
    assert summary["sample_count"] == 5
    assert len(str(summary["metadata_sha256"])) == 64


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("model_name", "vit_small_patch16_224", "model"),
        ("selected_layers", [10], "layers"),
        ("sample_count", 4, "sample_count"),
    ),
)
def test_vision_cache_metadata_rejects_contract_mismatch(
    tmp_path,
    field: str,
    value: object,
    message: str,
) -> None:
    source, cache, metadata = _write_vision_cache(tmp_path)
    metadata[field] = value
    (cache / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_vision_cache_metadata(source, cache)


def test_vision_cache_metadata_rejects_wrong_source(tmp_path) -> None:
    source, cache, metadata = _write_vision_cache(tmp_path)
    other = tmp_path / "other.npz"
    np.savez(other, obs=np.zeros((5, 3), dtype=np.float32))
    metadata["source_data"] = str(other.resolve())
    (cache / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="source"):
        validate_vision_cache_metadata(source, cache)


@pytest.mark.parametrize("variant", ("policy_only_residual", "fk_deform_attn"))
def test_cpu_float16_cached_vision_forward_backward(variant: str) -> None:
    torch.manual_seed(23)
    model = MatchedPolicyOnlyGrounding(TinyPolicy(), make_readout(variant)).train()
    batch = make_training_batch()
    batch.pop("rgb")
    batch["vision_backbone_features"] = torch.randn(
        1,
        2,
        1,
        3,
        8,
        8,
        dtype=torch.float16,
    )
    batch["vision_prefix"] = torch.randn(1, 2, 3, dtype=torch.float16)
    batch["image_size"] = torch.tensor([[16, 16]], dtype=torch.long)

    loss, _, output = model.compute_bc_loss(batch)
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert output.action.device.type == "cpu"
    assert output.action.dtype == torch.float32
    projection_grad = model.policy.vision_encoder.feature_projections[0].weight.grad
    readout_grad = model.readout.output_head[-1].weight.grad
    assert projection_grad is not None and torch.isfinite(projection_grad).all()
    assert readout_grad is not None and torch.isfinite(readout_grad).all()


class LatencyProbe(nn.Module):
    def __init__(self, variant: str, calls: list[str]) -> None:
        super().__init__()
        self.variant = variant
        self.calls = calls

    def predict_action_chunk(self, batch: object) -> None:
        del batch
        self.calls.append(self.variant)


def test_latency_benchmark_uses_fixed_equal_interleaved_calls_after_warmup() -> None:
    calls: list[str] = []
    models = {
        variant: LatencyProbe(variant, calls)
        for variant in ("policy_only_residual", "fk_deform_attn")
    }
    batch = {"obs": torch.zeros(1, 2, 3)}

    result = benchmark_interleaved_latency(
        models,  # type: ignore[arg-type]
        batch,
        torch.device("cpu"),
        warmup=2,
        iterations=3,
        amp=False,
        comparison_id="run-17:best",
    )

    assert calls == [
        "policy_only_residual",
        "fk_deform_attn",
        "fk_deform_attn",
        "policy_only_residual",
        "policy_only_residual",
        "fk_deform_attn",
        "fk_deform_attn",
        "policy_only_residual",
        "policy_only_residual",
        "fk_deform_attn",
    ]
    residual_contract = result["policy_only_residual"]["comparison_contract"]
    deform_contract = result["fk_deform_attn"]["comparison_contract"]
    assert residual_contract == deform_contract
    assert residual_contract["equal_call_count_per_variant"] is True
    assert residual_contract["rollout_call_counts_used"] is False
    assert result["policy_only_residual"]["measurement"]["timed_iterations"] == 3
    assert result["fk_deform_attn"]["measurement"]["timed_iterations"] == 3


def pair_checkpoint(
    variant: str,
    total_active: int,
    *,
    checkpoint_state: str = "best",
) -> dict[str, object]:
    maximum_budget = {"epochs": 10, "optimizer_steps_per_variant": 100}
    if checkpoint_state == "best":
        selected = {
            "epoch": 3 if variant == "policy_only_residual" else 7,
            "optimizer_steps": 30 if variant == "policy_only_residual" else 70,
            "val_loss": 0.1,
            "selection_role": "independent_validation_best",
        }
    else:
        selected = {
            "epoch": 10,
            "optimizer_steps": 100,
            "val_loss": 0.2,
            "selection_role": "final_at_matched_budget",
        }
    latency_contract = {
        "protocol": "fixed-input-equal-count-warmup-interleaved-v1",
        "comparison_id": f"run-17:{checkpoint_state}",
        "fixed_input_sha256": "0" * 64,
        "batch_size": 1,
        "warmup_iterations_per_variant": 2,
        "timed_iterations_per_variant": 5,
        "interleaved_order": "alternating-forward-reverse-by-iteration",
        "equal_call_count_per_variant": True,
        "rollout_call_counts_used": False,
    }
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_type": CHECKPOINT_TYPE,
        "variant": variant,
        "loss_contract": deepcopy(LOSS_CONTRACT),
        "structural_attribution_contract": deepcopy(STRUCTURAL_ATTRIBUTION_CONTRACT),
        "parameter_counts": {
            "frozen_backbone_active": 0,
            "vision_neck_active": 100,
            "readout_active": total_active - 1100,
            "adapter_active": total_active - 1000,
            "common_policy_active": 1000,
            "total_active": total_active,
        },
        "build_contract": {
            "policy_contract_checkpoint_sha256": "policy",
            "fcpg_corrector_checkpoint_sha256": "corrector",
            "policy_seed": 17,
            "policy_config": {"same": True},
            "active_parameter_reference": "fcpg_full_non_backbone_trainable",
            "active_parameter_tolerance": 0.01,
            "padding_parameters": 0,
            "backbone_state_sha256": "backbone",
            "sampling": {
                "policy_visual_samples_per_anchor": 8,
                "policy_token": "weighted_visual_then_shared_grounded_token_adapter",
                "offset_parameterization": "20*tanh",
            },
        },
        "data_contract": {"same": True},
        "training_contract": {"optimizer": "AdamW", "steps": 10},
        "selection_contract": {
            "policy": SELECTION_POLICY,
            "metric": "validation_behavior_cloning_loss",
            "maximum_budget": maximum_budget,
            "selected_step_relationship": "independent_by_variant",
        },
        "pair_run_contract": {
            "pair_run_id": "run-17",
            "selection_policy": SELECTION_POLICY,
            "maximum_budget": maximum_budget,
            "realized_budget": {
                "epochs": 10,
                "optimizer_steps_per_variant": 100,
            },
            "shared_training_run": True,
        },
        "vision_cache_validation": {
            "status": "validated_before_training",
            "required_model_name": AUDITED_VISION_MODEL,
            "required_selected_layers": AUDITED_VISION_SELECTED_LAYERS,
            "sources": ({"source_data": "source.npz", "sample_count": 5},),
        },
        "claim_contract": {
            "scope": DEVELOPMENT_CLAIM_SCOPE,
            "training_seed": 0,
            "development_only": True,
            "controlled_comparison_only": True,
            "cross_seed_generalization_supported": False,
            "general_claim_supported": False,
        },
        "normalization": {"obs_mean": torch.tensor([1.0, 2.0])},
        "runtime_budget": {
            "structural_latency": {
                "comparison_contract": latency_contract,
                "measurement": {
                    "variant": variant,
                    "timed_iterations": 5,
                    "mean_ms": 1.0,
                },
            }
        },
        "checkpoint_state": checkpoint_state,
        "selected": selected,
    }


def test_structural_pair_contract_accepts_only_matched_policy_only_pair() -> None:
    residual = pair_checkpoint("policy_only_residual", 100_000)
    deform = pair_checkpoint("fk_deform_attn", 100_069)
    result = validate_structural_pair(residual, deform)
    assert result["structural_attribution_allowed"] is True
    assert result["supervised_fcpg_comparison_scope"] == "system_only"
    assert result["selection_interpretation"] == SELECTION_POLICY
    assert result["selected_epochs"] == {
        "policy_only_residual": 3,
        "fk_deform_attn": 7,
    }
    assert result["same_selected_step_claim_supported"] is False

    mismatched = deepcopy(deform)
    mismatched["training_contract"] = {"optimizer": "SGD", "steps": 10}
    with pytest.raises(ValueError, match="training_contract"):
        validate_structural_pair(residual, mismatched)

    mismatched_normalization = deepcopy(deform)
    mismatched_normalization["normalization"]["obs_mean"] = torch.tensor([1.0, 3.0])
    with pytest.raises(ValueError, match="normalization mismatch"):
        validate_structural_pair(residual, mismatched_normalization)


def test_structural_pair_rejects_best_final_mixing_and_cross_run_pairing() -> None:
    residual = pair_checkpoint("policy_only_residual", 100_000)
    final_deform = pair_checkpoint(
        "fk_deform_attn",
        100_069,
        checkpoint_state="final",
    )
    with pytest.raises(ValueError, match="best/final mixing"):
        validate_structural_pair(residual, final_deform)

    deform = pair_checkpoint("fk_deform_attn", 100_069)
    mismatched_run = deepcopy(deform)
    mismatched_run["pair_run_contract"]["pair_run_id"] = "another-run"  # type: ignore[index]
    mismatched_run["runtime_budget"]["structural_latency"][  # type: ignore[index]
        "comparison_contract"
    ]["comparison_id"] = "another-run:best"
    with pytest.raises(ValueError, match="pair_run_contract"):
        validate_structural_pair(residual, mismatched_run)

    mismatched_input = deepcopy(deform)
    mismatched_input["runtime_budget"]["structural_latency"][  # type: ignore[index]
        "comparison_contract"
    ]["fixed_input_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="latency comparison contracts"):
        validate_structural_pair(residual, mismatched_input)


def test_seed0_config_declares_development_only_claim_scope() -> None:
    config = json.loads(
        (
            REPO_ROOT
            / "config/fk_deform_attn/matched_policy_only_pair.json"
        ).read_text(encoding="utf-8")
    )
    assert config["seed"] == 0
    assert config["evidence_scope"] == DEVELOPMENT_CLAIM_SCOPE
    assert config["cross_seed_generalization_claim_supported"] is False
    assert tuple(config["vision_selected_layers"]) == AUDITED_VISION_SELECTED_LAYERS
