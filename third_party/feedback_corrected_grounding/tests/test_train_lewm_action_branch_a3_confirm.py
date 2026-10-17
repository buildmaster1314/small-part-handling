from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import train_lewm_action_branch_a3_confirm as module


def _contract() -> dict:
    return json.loads(module.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def _development_go() -> dict:
    tasks = list(module.TASKS)
    seeds = list(module.DEVELOPMENT_SEEDS)
    comparison = {
        "baseline_method": "",
        "factual_method": "factual",
        "effect": 0.20,
        "positive_task_count": 3,
        "task_count": 4,
        "positive_seed_count": 2,
        "seed_count": 3,
    }
    return {
        "protocol": module.PROTOCOL,
        "stage": "a3_2k_development",
        "decision": "Development-Go",
        "passed": True,
        "input_contains_test_rows": False,
        "checkpoint_step": module.DEVELOPMENT_CHECKPOINT_STEP,
        "primary_horizons": [6, 7, 8],
        "coverage": {
            "tasks": tasks,
            "training_seeds": seeds,
            "development_hazard_bundle_count": 48,
            "per_task": {
                task: {
                    "train_physical_episode_count": 24,
                    "development_physical_episode_count": 12,
                    "development_hazard_bundle_count": 12,
                }
                for task in tasks
            },
        },
        "whitening": {
            "source": "canonical_train_rows",
            "per_task": True,
            "episode_equal": True,
            "fit_horizons": [6, 7, 8],
            "tasks": {
                task: {
                    "train_physical_episode_count": 24,
                    "floored_dimensions": 0,
                }
                for task in tasks
            },
        },
        "comparisons": {
            "no_action": {**comparison, "baseline_method": "no_action"},
            "shuffled_action": {
                **comparison,
                "baseline_method": "shuffled_action",
            },
        },
        "shuffled_action_shifts": {
            "count": 7,
            "minimum_effect": 0.20,
            "per_shift": [
                {
                    "donor_shift": shift,
                    "effect": 0.20,
                    "positive_task_count": 3,
                    "positive_seed_count": 2,
                }
                for shift in range(1, 8)
            ],
        },
        "raw_h6_h8": {
            "methods": {
                "factual": {"task_macro_mse": 1.0},
                "no_action": {"task_macro_mse": 1.0},
                "hold_last": {"task_macro_mse": 2.0},
                "latent_constant_velocity": {"task_macro_mse": 3.0},
            }
        },
        "gate": {
            "minimum_relative_improvement": 0.05,
            "checks": {name: True for name in module.DEVELOPMENT_GATE_CHECKS},
            "failed_checks": [],
            "jointly_positive_tasks": tasks[:3],
            "jointly_positive_seeds": seeds[:2],
            "cross_replicating_seeds": seeds[:2],
            "cross_replicating_seed_task_counts": {
                str(seed): 3 if position < 2 else 0
                for position, seed in enumerate(seeds)
            },
            "passed": True,
            "decision": "Development-Go",
        },
    }


def _write_summary(path: Path, payload: dict | None = None) -> Path:
    path.write_text(json.dumps(payload or _development_go()), encoding="utf-8")
    return path


def _base_argv(summary: Path, *, seed: int = 29_301) -> list[str]:
    return [
        "--input-npz",
        "confirmation.npz",
        "--base-checkpoint",
        "frozen-base.pt",
        "--output-dir",
        "confirmation-run",
        "--budget",
        "10k",
        "--development-summary",
        str(summary),
        "--seed",
        str(seed),
        "--device",
        "cuda:0",
        "--train-split",
        "train",
        "--val-split",
        "test",
        "--batch-size",
        "16",
        "--amp",
        "--amp-dtype",
        "float16",
    ]


def test_checked_in_confirmation_contract_is_fully_frozen() -> None:
    contract = _contract()
    assert module._validate_confirmation_contract(contract) == module.TASKS
    assert contract["development_gate"] == {
        "evaluation_split": "development",
        "checkpoint_step": 2000,
        "seeds": [29201, 29202, 29203],
        "primary_horizons": [6, 7, 8],
        "minimum_relative_improvement": 0.05,
        "minimum_positive_tasks": 3,
        "minimum_positive_seeds": 2,
        "minimum_cross_replicating_seeds": 2,
        "minimum_positive_tasks_per_replicating_seed": 3,
        "every_shuffled_shift_must_reach_minimum": True,
        "raw_factual_not_worse_than_no_action": True,
        "raw_factual_better_than_hold_last_and_constant_velocity": True,
    }
    assert contract["confirmation_after_development_go"] == {
        "reserved_test_episodes_per_task": 12,
        "test_collection_requires_development_go": True,
        "optimizer_steps": 10000,
        "seeds": [29301, 29302, 29303],
        "fresh_start_from_frozen_base": True,
        "train_split": "train",
        "evaluation_split": "test",
        "bundle_batch_size": 16,
        "amp_dtype": "float16",
        "anchor_encoder_sigreg": 0.02,
        "additional_sigreg_values_allowed": [],
        "selection": "fixed_latest",
        "primary_horizons": [6, 7, 8],
        "minimum_relative_improvement": 0.05,
        "minimum_positive_tasks": 3,
        "minimum_positive_seeds": 2,
        "minimum_cross_replicating_seeds": 2,
        "minimum_positive_tasks_per_replicating_seed": 3,
        "every_shuffled_shift_must_reach_minimum": True,
        "raw_factual_not_worse_than_no_action": True,
        "raw_factual_better_than_hold_last_and_constant_velocity": True,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("optimizer_steps", 2000),
        ("seeds", [29201, 29202, 29203]),
        ("fresh_start_from_frozen_base", False),
        ("train_split", "development"),
        ("evaluation_split", "development"),
        ("bundle_batch_size", 8),
        ("amp_dtype", "bfloat16"),
        ("anchor_encoder_sigreg", 0.0),
        ("additional_sigreg_values_allowed", [0.02]),
        ("selection", "best"),
        ("minimum_cross_replicating_seeds", 1),
        ("minimum_positive_tasks_per_replicating_seed", 2),
        ("optimizer_steps", 10000.0),
        ("fresh_start_from_frozen_base", 1),
    ],
)
def test_confirmation_contract_rejects_any_training_drift(
    field: str, value: object
) -> None:
    contract = deepcopy(_contract())
    contract["confirmation_after_development_go"][field] = value
    with pytest.raises(ValueError, match="confirmation contract drifted"):
        module._validate_confirmation_contract(contract)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("minimum_cross_replicating_seeds", 1),
        ("minimum_positive_tasks_per_replicating_seed", 2),
    ],
)
def test_development_gate_contract_rejects_cross_replication_drift(
    field: str, value: object
) -> None:
    contract = deepcopy(_contract())
    contract["development_gate"][field] = value
    with pytest.raises(ValueError, match="development gate contract drifted"):
        module._validate_confirmation_contract(contract)


def test_development_summary_requires_complete_nested_go(tmp_path: Path) -> None:
    path = _write_summary(tmp_path / "development.json")
    result = module._validate_development_go_summary(
        path, contract=_contract(), tasks=module.TASKS
    )
    assert result["decision"] == "Development-Go"
    assert "sha256" not in json.dumps(result).lower()

    failed = _development_go()
    failed["gate"]["checks"]["late_effect_vs_every_shuffled_shift"] = False
    _write_summary(path, failed)
    with pytest.raises(ValueError, match="nested gate"):
        module._validate_development_go_summary(
            path, contract=_contract(), tasks=module.TASKS
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(decision="Development-No-Go"), "Development-Go"),
        (
            lambda value: value["coverage"].update(training_seeds=[29201, 29202]),
            "frozen tasks and seeds",
        ),
        (
            lambda value: value["shuffled_action_shifts"]["per_shift"][3].update(
                effect=0.01
            ),
            "shift 4",
        ),
        (
            lambda value: value["raw_h6_h8"]["methods"]["factual"].update(
                task_macro_mse=4.0
            ),
            "raw factual baselines",
        ),
        (lambda value: value.update(checkpoint_step=2000.0), "Development-Go"),
        (
            lambda value: value["gate"].pop("cross_replicating_seeds"),
            "cross-replicating A3 seeds",
        ),
        (
            lambda value: value["gate"]["checks"].update(
                cross_seed_task_replication=False
            ),
            "nested gate",
        ),
    ],
)
def test_development_summary_rejects_incomplete_evidence(
    tmp_path: Path, mutation, message: str
) -> None:
    payload = _development_go()
    mutation(payload)
    path = _write_summary(tmp_path / "development.json", payload)
    with pytest.raises(ValueError, match=message):
        module._validate_development_go_summary(
            path, contract=_contract(), tasks=module.TASKS
        )


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (("--budget", "2k"), "only the 10k"),
        (("--seed", "29201"), "29301"),
        (("--train-split", "development"), "train/test"),
        (("--val-split", "development"), "train/test"),
        (("--batch-size", "8"), "batch size"),
        (("--device", "cpu"), "CUDA"),
        (("--device", "cuda-not-a-device"), "CUDA"),
        (("--amp-dtype", "bfloat16"), "float16"),
        (("--lr", "0.001"), "frozen A1 optimizer"),
    ],
)
def test_cli_rejects_non_confirmation_training_options(
    tmp_path: Path, replacement: tuple[str, str], message: str
) -> None:
    summary = _write_summary(tmp_path / "development.json")
    argv = module._replace_option(_base_argv(summary), *replacement)
    args = module.parse_args(argv)
    with pytest.raises(ValueError, match=message):
        module._validate_cli(args, _contract())


def test_cli_has_no_resume_or_a1_screen_entrypoint(tmp_path: Path) -> None:
    summary = _write_summary(tmp_path / "development.json")
    for option, value in (
        ("--resume", "a3-2k/latest.pt"),
        ("--sigreg-weight", "0.0"),
        ("--selection", "best"),
    ):
        with pytest.raises(SystemExit):
            module.parse_args([*_base_argv(summary), option, value])
    with pytest.raises(ValueError, match="does not accept A1 --screen-summary"):
        module.parse_args(
            [*_base_argv(summary), "--screen-summary", "a3-2k-summary.json"]
        )


def test_frozen_base_validation_rejects_a3_2k_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "latest.pt"
    checkpoint.touch()
    monkeypatch.setattr(
        module.a1_trainer,
        "_torch_load",
        lambda _path: {"checkpoint_type": "fcpg_matched_anchor_dynamics_pair_v1"},
    )
    with pytest.raises(ValueError, match="AnchorPersistent"):
        module._validate_frozen_base(checkpoint)


def test_confirmation_input_requires_canonical_train_test_conformance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "confirmation.npz"
    arrays = {"split": object()}
    calls: list[tuple[object, object]] = []
    monkeypatch.setattr(module, "load_a1_npz", lambda path: arrays)
    monkeypatch.setattr(
        module,
        "validate_a3_confirmation_canonical_conformance",
        lambda value, contract: calls.append((value, contract)),
    )
    contract = _contract()

    assert module._validate_confirmation_input(
        source, contract=contract
    ) == source.resolve()
    assert calls == [(arrays, contract)]


def test_scope_freezes_test_predictions_and_restores_a1_hooks(tmp_path: Path) -> None:
    summary = _write_summary(tmp_path / "development.json")
    authorization = _development_go()
    original_protocol = module.a1_trainer.ACTION_BRANCH_TRAINING_PROTOCOL
    original_splits = module.a1_trainer._default_splits
    original_authorization = module.a1_trainer._validate_10k_authorization

    with module._a3_confirmation_trainer_scope(
        development_summary=summary,
        authorization=authorization,
    ):
        assert module.a1_trainer.ACTION_BRANCH_TRAINING_PROTOCOL == module.PROTOCOL
        assert module.a1_trainer._default_splits("10k", None, None) == (
            "train",
            "test",
        )
        assert (
            module.a1_trainer._validate_10k_authorization(summary)["decision"]
            == "Development-Go"
        )
        with pytest.raises(ValueError, match="only the 10k"):
            module.a1_trainer._default_splits("2k", None, None)
        with pytest.raises(ValueError, match="marker is invalid"):
            module.a1_trainer._validate_10k_authorization(tmp_path / "other.json")

    assert module.a1_trainer.ACTION_BRANCH_TRAINING_PROTOCOL == original_protocol
    assert module.a1_trainer._default_splits is original_splits
    assert module.a1_trainer._validate_10k_authorization is original_authorization


def test_main_delegates_fresh_train_test_run_to_a1_without_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = _write_summary(tmp_path / "development.json")
    base = tmp_path / "frozen-base.pt"
    base.touch()
    output = tmp_path / "run"
    calls: list[list[str]] = []

    monkeypatch.setattr(module, "_validate_frozen_base", lambda _path: base.resolve())
    monkeypatch.setattr(
        module,
        "_validate_confirmation_input",
        lambda path, *, contract: module._resolve(path),
    )

    def delegated(argv: list[str]) -> int:
        calls.append(argv)
        delegated_args = module.a1_trainer.parse_args(argv)
        assert delegated_args.budget == "10k"
        assert delegated_args.seed == 29301
        assert delegated_args.train_split == "train"
        assert delegated_args.val_split == "test"
        assert delegated_args.batch_size == 16
        assert delegated_args.amp is True
        assert delegated_args.amp_dtype == "float16"
        assert module.a1_trainer.ACTION_BRANCH_TRAINING_PROTOCOL == module.PROTOCOL
        assert module.a1_trainer._default_splits(
            delegated_args.budget,
            delegated_args.train_split,
            delegated_args.val_split,
        ) == ("train", "test")
        marker = module.a1_trainer._validate_10k_authorization(
            delegated_args.screen_summary
        )
        assert marker is not None and marker["decision"] == "Development-Go"
        return 17

    monkeypatch.setattr(module.a1_trainer, "main", delegated)
    argv = _base_argv(summary)
    argv = module._replace_option(argv, "--base-checkpoint", str(base))
    argv = module._replace_option(argv, "--output-dir", str(output))

    assert module.main(argv) == 17
    assert len(calls) == 1
    assert "--resume" not in calls[0]
    assert "--development-summary" not in calls[0]
