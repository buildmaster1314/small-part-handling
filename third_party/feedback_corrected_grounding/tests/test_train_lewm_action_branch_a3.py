from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import train_lewm_action_branch_a3 as module


def _contract() -> dict:
    return json.loads(module.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_a3_training_contract_is_frozen() -> None:
    contract = _contract()
    assert module._validate_training_contract(contract) == module.TASKS
    contract["training_if_gate_go"]["anchor_encoder_sigreg"] = 0.09
    with pytest.raises(ValueError, match="training contract drifted"):
        module._validate_training_contract(contract)


def test_a3_support_summary_requires_joint_go(tmp_path: Path) -> None:
    payload = {
        "protocol": module.PROTOCOL,
        "stage": "collection_only_physical_and_latent_support",
        "decision": "Go",
        "model_training_started": False,
        "input_contains_test_rows": False,
        "primary_horizons": [6, 7, 8],
        "physical_support_passed": True,
        "latent_support_passed": True,
        "joint_support_passed": True,
        "per_task": {
            task: {
                "passed": True,
                "physical": {"passed": True},
                "latent": {"passed": True},
                "joint": {"passed": True},
            }
            for task in module.TASKS
        },
    }
    path = tmp_path / "support.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    module._validate_support_summary(path, tasks=module.TASKS)
    payload["per_task"][module.TASKS[0]]["joint"]["passed"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=module.TASKS[0]):
        module._validate_support_summary(path, tasks=module.TASKS)


def test_a3_scope_sets_protocol_and_restores_hooks() -> None:
    original_protocol = module.a1_trainer.ACTION_BRANCH_TRAINING_PROTOCOL
    original_splits = module.a1_trainer._default_splits
    with module._a3_trainer_scope():
        assert module.a1_trainer.ACTION_BRANCH_TRAINING_PROTOCOL == module.PROTOCOL
        assert module.a1_trainer._default_splits("2k", None, None) == (
            "train",
            "development",
        )
        with pytest.raises(ValueError, match="only the 2k"):
            module.a1_trainer._default_splits("10k", None, None)
    assert module.a1_trainer.ACTION_BRANCH_TRAINING_PROTOCOL == original_protocol
    assert module.a1_trainer._default_splits is original_splits


def test_replace_option_normalizes_split_and_equals_forms() -> None:
    assert module._replace_option(
        ["--input-npz", "old", "--seed", "29201"],
        "--input-npz",
        "/absolute/data.npz",
    ) == ["--input-npz", "/absolute/data.npz", "--seed", "29201"]
    assert module._replace_option(
        ["--input-npz=old", "--seed", "29201"],
        "--input-npz",
        "/absolute/data.npz",
    ) == ["--input-npz", "/absolute/data.npz", "--seed", "29201"]


def test_a3_cli_rejects_non_frozen_seed() -> None:
    args = module.parse_args(
        [
            "--input-npz",
            "unused.npz",
            "--output-dir",
            "unused-output",
            "--budget",
            "2k",
            "--support-summary",
            "unused.json",
            "--seed",
            "1",
            "--device",
            "cuda:0",
            "--batch-size",
            "16",
            "--amp-dtype",
            "float16",
        ]
    )
    with pytest.raises(ValueError, match="29201"):
        module._validate_cli(args, _contract())
