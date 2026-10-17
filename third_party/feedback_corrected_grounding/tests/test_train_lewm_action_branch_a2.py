from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from fcpg.lewm_action_branch_a2 import PROTOCOL
from scripts import train_lewm_action_branch_a2 as cli


def _contract() -> dict:
    return json.loads(cli.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def _base_argv(budget: str) -> list[str]:
    return [
        "--input-npz",
        "cache.npz",
        "--output-dir",
        "run",
        "--budget",
        budget,
    ]


def _support_summary() -> dict:
    return {
        "protocol": PROTOCOL,
        "stage": "collection_only_intervention_support",
        "decision": "Development-Go",
        "model_training_started": False,
        "input_contains_test_rows": False,
        "late_horizon_development_only": True,
        "primary_horizons": [6, 7, 8],
        "per_task": {task: {"passed": True} for task in cli.FROZEN_TASKS},
    }


def _support_arrays(*, weak_drawer: bool = False) -> dict[str, np.ndarray]:
    rows = 4 * (24 + 12)
    target = np.zeros((rows, 8, 8, 1, 4), dtype=np.float32)
    split: list[str] = []
    task_index: list[int] = []
    episode: list[str] = []
    cursor = 0
    for task in range(4):
        for local in range(36):
            label = "train" if local < 24 else "development"
            split.append(label)
            task_index.append(task)
            episode.append(f"task-{task}-episode-{local}")
            scale = 1e-5 if weak_drawer and task == 1 and label == "development" else 0.2
            target[cursor] = scale * np.arange(8, dtype=np.float32)[:, None, None, None]
            cursor += 1
    return {
        "target_embedding": target,
        "split": np.asarray(split),
        "task_index": np.asarray(task_index, dtype=np.int64),
        "source_episode_id": np.asarray(episode),
    }


def test_checked_in_training_contract_is_frozen() -> None:
    contract = _contract()

    assert cli._validate_training_contract(contract) == cli.FROZEN_TASKS
    assert contract["authorized_after_support_go"] == {
        "matched_factual_and_no_action_screen_steps": 2000,
        "screen_training_seeds": [25801, 25802, 25803],
        "bundle_batch_size": 16,
        "amp_dtype": "float16",
        "anchor_encoder_sigreg": 0.02,
        "additional_sigreg_values_allowed": [],
        "fresh_10000_step_training_requires_2k_screen_go": True,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("anchor_encoder_sigreg", 0.0, "exactly 0.02"),
        ("anchor_encoder_sigreg", 0.09, "exactly 0.02"),
        ("additional_sigreg_values_allowed", [0.0], "additional SIGReg"),
        (
            "fresh_10000_step_training_requires_2k_screen_go",
            False,
            "require a 2k Screen Go",
        ),
        ("screen_training_seeds", [25801, 25802], "25801/25802/25803"),
    ],
)
def test_training_contract_rejects_sigreg_or_authorization_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    contract = deepcopy(_contract())
    contract["authorized_after_support_go"][field] = value

    with pytest.raises(ValueError, match=message):
        cli._validate_training_contract(contract)


def test_only_a_structurally_identical_config_copy_is_accepted(tmp_path: Path) -> None:
    copied = tmp_path / "a2.json"
    copied.write_text(
        json.dumps(_contract(), indent=2) + "\n",
        encoding="utf-8",
    )
    assert cli._load_fixed_config(copied) == _contract()

    changed = _contract()
    changed["status"] = "changed"
    copied.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed to the checked-in"):
        cli._load_fixed_config(copied)


def test_cli_supports_only_the_fixed_budgets_and_no_sigreg_override() -> None:
    for budget in ("smoke", "2k", "10k"):
        assert cli.parse_args(_base_argv(budget)).budget == budget

    with pytest.raises(SystemExit):
        cli.parse_args([*_base_argv("2k"), "--sigreg-weight", "0.0"])


def test_help_exposes_a2_gates(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--help"])
    output = capsys.readouterr().out
    assert "--support-summary" in output
    assert "Shared A1 trainer options" in output


def test_split_scope_is_frozen_and_test_needs_explicit_authorization() -> None:
    assert cli._default_splits("smoke", None, None) == ("smoke", "smoke")
    assert cli._default_splits("2k", None, None) == ("train", "development")
    with pytest.raises(ValueError, match="remains locked"):
        cli._default_splits("10k", None, None)

    with pytest.raises(ValueError, match="freezes train/val splits"):
        cli._default_splits("2k", "train", "test")
    with pytest.raises(ValueError, match="valid only for the 10k"):
        cli._default_splits("2k", None, None, test_authorized=True)


def test_support_authorization_requires_a_four_task_go(tmp_path: Path) -> None:
    path = tmp_path / "support.json"
    path.write_text(json.dumps(_support_summary()), encoding="utf-8")
    assert cli._validate_support_authorization(
        path, task_names=cli.FROZEN_TASKS
    )["decision"] == "Development-Go"

    failed = _support_summary()
    failed["per_task"][cli.FROZEN_TASKS[1]]["passed"] = False
    path.write_text(json.dumps(failed), encoding="utf-8")
    with pytest.raises(ValueError, match="every frozen task"):
        cli._validate_support_authorization(path, task_names=cli.FROZEN_TASKS)

    exposed = _support_summary()
    exposed["input_contains_test_rows"] = True
    path.write_text(json.dumps(exposed), encoding="utf-8")
    with pytest.raises(ValueError, match="unexposed-test"):
        cli._validate_support_authorization(path, task_names=cli.FROZEN_TASKS)


def test_support_is_recomputed_from_the_actual_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "load_a1_npz", lambda _path: _support_arrays())
    result = cli._recompute_support_authorization(
        "cache.npz",
        contract=_contract(),
        task_names=cli.FROZEN_TASKS,
    )
    assert result["decision"] == "Go"

    monkeypatch.setattr(
        cli,
        "load_a1_npz",
        lambda _path: _support_arrays(weak_drawer=True),
    )
    with pytest.raises(ValueError, match="does not reproduce"):
        cli._recompute_support_authorization(
            "cache.npz",
            contract=_contract(),
            task_names=cli.FROZEN_TASKS,
        )

def test_2k_requires_support_go_and_delegates_to_a1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = tmp_path / "support.json"
    summary.write_text(json.dumps(_support_summary()), encoding="utf-8")
    calls: list[list[str]] = []
    original_protocol = cli.a1_trainer.ACTION_BRANCH_TRAINING_PROTOCOL
    original_splits = cli.a1_trainer._default_splits

    def delegated(argv: list[str]) -> int:
        calls.append(argv)
        assert cli.a1_trainer.ACTION_BRANCH_TRAINING_PROTOCOL == PROTOCOL
        assert cli.a1_trainer._default_splits("2k", None, None) == (
            "train",
            "development",
        )
        assert cli.a1_trainer._validate_10k_authorization(None) is None
        return 7

    monkeypatch.setattr(cli.a1_trainer, "main", delegated)
    monkeypatch.setattr(
        cli,
        "_recompute_support_authorization",
        lambda *_args, **_kwargs: {"decision": "Go"},
    )
    with pytest.raises(ValueError, match="--support-summary is required"):
        cli.main(_base_argv("2k"))

    assert cli.main([
        *_base_argv("2k"),
        "--support-summary",
        str(summary),
        "--seed",
        "25801",
        "--device",
        "cuda:0",
        "--batch-size",
        "16",
        "--amp",
        "--amp-dtype",
        "float16",
    ]) == 7
    assert len(calls) == 1
    assert "--support-summary" not in calls[0]
    assert "--screen-summary" not in calls[0]
    assert cli.a1_trainer.ACTION_BRANCH_TRAINING_PROTOCOL == original_protocol
    assert cli.a1_trainer._default_splits is original_splits


def test_10k_remains_locked_until_a_fresh_confirmation_contract_exists() -> None:
    with pytest.raises(ValueError, match="remains locked"):
        cli.main(_base_argv("10k"))
