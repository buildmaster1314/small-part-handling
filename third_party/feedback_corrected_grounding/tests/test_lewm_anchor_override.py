from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.evaluate_lewm_grounded import (
    ANCHOR_ABLATION_PAIR_ID,
    ANCHOR_ABLATION_PARAMETER_COUNT,
    ANCHOR_ABLATION_SEED,
    ANCHOR_ABLATION_SPLIT_SHA256,
    ANCHOR_ABLATION_TARGET_SHA256,
    apply_anchor_visual_mode_override,
    resolve_anchor_visual_ablation_contract,
    resolve_oracle_token_spec,
)


class _ParameterCount:
    def numel(self) -> int:
        return ANCHOR_ABLATION_PARAMETER_COUNT


class _Model:
    def __init__(self, mode: str) -> None:
        self.frame_encoder = SimpleNamespace(anchor_visual_mode=mode)

    def parameters(self):
        return [_ParameterCount()]


def _checkpoint(mode: str = "triplet_nominal") -> dict[str, object]:
    frame = {"anchor_visual_mode": mode}
    return {
        "run_kind": "formal",
        "step": 2000,
        "metrics": {"step": 2000},
        "model_config": {"frame": deepcopy(frame)},
        "resolved_config": {
            "seed": ANCHOR_ABLATION_SEED,
            "loss": {"sigreg_weight": 0.02},
            "model": {"frame": deepcopy(frame)},
            "contract": {
                "gate": "anchor_visual_input_ablation",
                "pair_id": ANCHOR_ABLATION_PAIR_ID,
                "anchor_visual_mode": mode,
                "development_only": True,
                "frozen_gate_l1_test_reuse_forbidden": True,
                "training_grounding": "fcpg",
                "optimizer_step_budget": 2000,
                "expected_parameter_count": ANCHOR_ABLATION_PARAMETER_COUNT,
                "target_encoder_sha256": ANCHOR_ABLATION_TARGET_SHA256,
            },
        },
        "provenance": {
            "manifest": {
                "episode_manifest": {"sha256": ANCHOR_ABLATION_SPLIT_SHA256}
            }
        },
    }


def test_override_contract_and_application_are_explicit() -> None:
    checkpoint = _checkpoint()
    contract = resolve_anchor_visual_ablation_contract(
        checkpoint,
        "lewm_grounded_latest.pt",
        "paired_innovation",
        split="validation",
        grounding_mode_request="checkpoint",
    )
    model = _Model("triplet_nominal")

    report = apply_anchor_visual_mode_override(model, checkpoint, contract)

    assert report["active"] is True
    assert report["train_mode"] == "triplet_nominal"
    assert report["eval_mode"] == "paired_innovation"
    assert report["parameter_count"] == ANCHOR_ABLATION_PARAMETER_COUNT
    assert report["override_applied"] is True
    assert model.frame_encoder.anchor_visual_mode == "paired_innovation"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(step=1999), "checkpoint.step"),
        (
            lambda value: value["resolved_config"]["contract"].update(
                gate="gate_l1"
            ),
            "contract mismatch",
        ),
        (
            lambda value: value["model_config"]["frame"].update(
                anchor_visual_mode="paired_visual"
            ),
            "disagree",
        ),
        (
            lambda value: value["resolved_config"]["contract"].update(
                expected_parameter_count=1
            ),
            "expected_parameter_count",
        ),
    ],
)
def test_override_contract_fails_closed(mutation, message: str) -> None:
    checkpoint = _checkpoint()
    mutation(checkpoint)

    with pytest.raises((TypeError, ValueError), match=message):
        resolve_anchor_visual_ablation_contract(
            checkpoint,
            "lewm_grounded_latest.pt",
            "triplet_corrected",
            split="validation",
            grounding_mode_request="checkpoint",
        )


def test_override_rejects_best_checkpoint_and_nonvalidation_use() -> None:
    checkpoint = _checkpoint()
    with pytest.raises(ValueError, match="latest"):
        resolve_anchor_visual_ablation_contract(
            checkpoint,
            "lewm_grounded_best.pt",
            "triplet_corrected",
            split="validation",
            grounding_mode_request="checkpoint",
        )
    with pytest.raises(ValueError, match="validation"):
        resolve_anchor_visual_ablation_contract(
            checkpoint,
            "lewm_grounded_latest.pt",
            "triplet_corrected",
            split="all",
            grounding_mode_request="checkpoint",
        )


def test_checkpoint_mode_does_not_activate_development_override() -> None:
    contract = resolve_anchor_visual_ablation_contract(
        {},
        "legacy.pt",
        "checkpoint",
        split="all",
        grounding_mode_request="nominal",
    )
    model = _Model("single_corrected")

    report = apply_anchor_visual_mode_override(model, {}, contract)

    assert report == {
        "active": False,
        "request": "checkpoint",
        "train_mode": "single_corrected",
        "eval_mode": "single_corrected",
    }


def test_legacy_oracle_cache_without_encoder_spec_remains_supported() -> None:
    assert (
        resolve_oracle_token_spec(
            [{"encoder_checkpoint_sha256": "a" * 64}],
            required=False,
        )
        is None
    )
    with pytest.raises(ValueError, match="encoder_spec"):
        resolve_oracle_token_spec(
            [{"encoder_checkpoint_sha256": "a" * 64}],
            required=True,
        )


def test_oracle_token_specs_must_match_for_anchor_ablation() -> None:
    spec = {"num_keypoints": 3, "target_dim": 192}
    assert resolve_oracle_token_spec(
        [{"encoder_spec": spec}, {"encoder_spec": deepcopy(spec)}],
        required=True,
    ) == spec
    with pytest.raises(ValueError, match="matching"):
        resolve_oracle_token_spec(
            [
                {"encoder_spec": spec},
                {"encoder_spec": {"num_keypoints": 2, "target_dim": 192}},
            ],
            required=True,
        )
