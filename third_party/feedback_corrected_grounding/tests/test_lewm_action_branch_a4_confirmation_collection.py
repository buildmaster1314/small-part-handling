from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fcpg.lewm_a4_storage import (
    ACTION_DIM,
    LATENT_DIM,
    STATE_CACHE_KEYS,
    STATE_SLOT_COUNT,
    load_a4_state_cache,
)
from fcpg.lewm_action_branch_a4 import (
    BRANCH_COUNT,
    DEVELOPMENT_TRAINING_SEEDS,
    HISTORY_STEPS,
    HORIZON,
    PROTOCOL,
    TASKS,
    assign_a4_splits,
)
from fcpg.lewm_action_branch_a4_confirmation_collection import (
    A4ConfirmationAuthorization,
    collect_a4_reserved_test_cache,
    save_a4_reserved_test_cache,
    validate_a4_development_go_summary,
)
from scripts import collect_lewm_action_branch_a4_confirmation as confirmation_cli


ROOT = Path(__file__).resolve().parents[1]


def _contract() -> dict:
    return json.loads(
        (ROOT / "config/lewm/eval/action_branch_active_control_a4.json").read_text(
            encoding="utf-8"
        )
    )


def _panel(contract: dict, split: str, count: int) -> dict[str, tuple[int, ...]]:
    assignments = assign_a4_splits(TASKS, contract)
    return {
        task: tuple(
            sorted(
                instance
                for instance in range(48)
                if assignments[(task, instance)] == split
            )[:count]
        )
        for task in TASKS
    }


def _cache(
    panel: dict[str, tuple[int, ...]], split: str, *, prefix: str
) -> dict[str, np.ndarray]:
    rows = [
        (task_index, task, instance)
        for task_index, task in enumerate(TASKS)
        for instance in panel[task]
    ]
    count = len(rows)
    semantic = np.broadcast_to(
        np.arange(BRANCH_COUNT, dtype=np.int64), (count, BRANCH_COUNT)
    ).copy()
    result = {
        "proposal_id": np.asarray(
            [f"{prefix}:{task}:instance-{instance:02d}" for _, task, instance in rows]
        ),
        "task_class_index": np.asarray([row[0] for row in rows], dtype=np.int64),
        "task_name": np.asarray([row[1] for row in rows]),
        "task_instance_index": np.asarray([row[2] for row in rows], dtype=np.int64),
        "source_episode_id": np.asarray(
            [f"{row[1]}:task-instance-{row[2]:02d}" for row in rows]
        ),
        "split": np.full(count, split),
        "history_state": np.zeros(
            (count, HISTORY_STEPS, STATE_SLOT_COUNT, LATENT_DIM), dtype=np.float16
        ),
        "history_action": np.zeros(
            (count, HISTORY_STEPS - 1, ACTION_DIM), dtype=np.float16
        ),
        "center_action": np.zeros((count, HORIZON, ACTION_DIM), dtype=np.float16),
        "future_action": np.zeros(
            (count, BRANCH_COUNT, HORIZON, ACTION_DIM), dtype=np.float16
        ),
        "target_state": np.zeros(
            (count, BRANCH_COUNT, HORIZON, STATE_SLOT_COUNT, LATENT_DIM),
            dtype=np.float16,
        ),
        "branch_semantic_index": semantic,
    }
    assert tuple(result) == STATE_CACHE_KEYS
    return result


def _formal_cache(contract: dict) -> dict[str, np.ndarray]:
    train = _cache(_panel(contract, "train", 4), "train", prefix="train")
    development = _cache(
        _panel(contract, "development", 2), "development", prefix="development"
    )
    return {
        name: np.concatenate((train[name], development[name]), axis=0)
        for name in STATE_CACHE_KEYS
    }


def _write_cache(path: Path, cache: dict[str, np.ndarray]) -> Path:
    np.savez_compressed(path, **cache)
    return path.resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _development_go(formal_cache: Path) -> dict:
    checks = {
        "minimum_effect_vs_no_action": True,
        "minimum_effect_vs_every_wrong_action_shift": True,
        "paired_ci_lower_exceeds_zero": True,
        "minimum_positive_tasks": True,
        "minimum_positive_seeds": True,
        "raw_factual_not_worse_than_no_action": True,
        "raw_factual_better_than_hold_last": True,
        "raw_factual_better_than_constant_velocity": True,
    }
    return {
        "protocol": PROTOCOL,
        "stage": "a4_2k_formal_development",
        "mode": "formal",
        "decision": "Development-Go",
        "passed": True,
        "development_go": True,
        "reserved_test_collection_authorized": True,
        "input_contains_test_rows": False,
        "checkpoint_step": 2_000,
        "primary_horizons": [6, 7, 8],
        "formal_train_cache": {
            "path": str(formal_cache.resolve()),
            "sha256": _sha256(formal_cache),
        },
        "coverage": {
            "tasks": list(TASKS),
            "training_seeds": list(DEVELOPMENT_TRAINING_SEEDS),
        },
        "gate": {
            "minimum_relative_improvement": 0.1,
            "checks": checks,
            "failed_checks": [],
            "jointly_positive_tasks": list(TASKS),
            "jointly_positive_seeds": list(DEVELOPMENT_TRAINING_SEEDS),
            "passed": True,
            "development_go": True,
        },
    }


def _authorized(tmp_path: Path) -> tuple[dict, Path, dict, A4ConfirmationAuthorization]:
    contract = _contract()
    cache_path = _write_cache(tmp_path / "formal.npz", _formal_cache(contract))
    summary = _development_go(cache_path)
    authorization = validate_a4_development_go_summary(
        summary, contract, formal_train_cache_path=cache_path
    )
    return contract, cache_path, summary, authorization


@pytest.mark.parametrize(
    "field,value",
    [
        ("decision", "Development-No-Go"),
        ("development_go", False),
        ("reserved_test_collection_authorized", False),
        ("input_contains_test_rows", True),
        ("checkpoint_step", 1_999),
    ],
)
def test_confirmation_requires_exact_development_go(
    tmp_path: Path, field: str, value: object
) -> None:
    contract = _contract()
    cache_path = _write_cache(tmp_path / "formal.npz", _formal_cache(contract))
    summary = _development_go(cache_path)
    summary[field] = value
    with pytest.raises(PermissionError, match="exact formal"):
        validate_a4_development_go_summary(
            summary, contract, formal_train_cache_path=cache_path
        )


def test_authorization_binds_explicit_cache_path_and_content(tmp_path: Path) -> None:
    contract, cache_path, summary, authorization = _authorized(tmp_path)
    assert authorization.formal_train_cache_path == cache_path
    assert authorization.formal_train_cache_sha256 == _sha256(cache_path)

    other = _write_cache(tmp_path / "other.npz", _formal_cache(contract))
    with pytest.raises(PermissionError, match="differs from Development-Go"):
        validate_a4_development_go_summary(
            summary, contract, formal_train_cache_path=other
        )

    with cache_path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(PermissionError, match="content differs"):
        validate_a4_development_go_summary(
            summary, contract, formal_train_cache_path=cache_path
        )


def test_formal_cache_rows_must_match_frozen_assignments(tmp_path: Path) -> None:
    contract = _contract()
    bad = _formal_cache(contract)
    bad["split"][0] = "development"
    cache_path = _write_cache(tmp_path / "bad-formal.npz", bad)
    summary = _development_go(cache_path)
    with pytest.raises(PermissionError, match="split assignment"):
        validate_a4_development_go_summary(
            summary, contract, formal_train_cache_path=cache_path
        )


def test_reserved_collector_directly_uses_test_split_and_frozen_seeds(
    tmp_path: Path,
) -> None:
    contract, _cache_path, _summary, authorization = _authorized(tmp_path)
    calls: list[tuple[str, int, int, int, str]] = []

    def episode_collector(**kwargs):
        calls.append(
            (
                kwargs["task"],
                kwargs["task_class_index"],
                kwargs["task_instance_index"],
                kwargs["environment_seed"],
                kwargs["split"],
            )
        )
        return [], []

    test_cache = _cache(_panel(contract, "test", 12), "test", prefix="test")
    result = collect_a4_reserved_test_cache(
        contract,
        authorization,
        encoder=SimpleNamespace(encode=lambda rows: None),
        runtime_factory=lambda *args: None,
        episode_collector=episode_collector,
        cache_stacker=lambda bundles: test_cache,
    )

    assert len(calls) == len(TASKS) * 12
    seed_base = int(contract["collection"]["environment_seed_base"])
    for task_index, task in enumerate(TASKS):
        task_calls = [row for row in calls if row[0] == task]
        expected_instances = authorization.test_instances_by_task[task]
        assert {row[2] for row in task_calls} == set(expected_instances)
        assert {row[4] for row in task_calls} == {"test"}
        assert {row[3] for row in task_calls} == {
            seed_base + task_index * 48 + instance for instance in expected_instances
        }
    assert set(result.cache["split"].astype(str)) == {"test"}
    assert result.accepted_instances_by_task == authorization.test_instances_by_task


def test_reserved_collector_requires_exactly_twelve_accepted_episodes_per_task(
    tmp_path: Path,
) -> None:
    contract, _cache_path, _summary, authorization = _authorized(tmp_path)
    too_small = _cache(_panel(contract, "test", 11), "test", prefix="test")
    with pytest.raises(RuntimeError, match="exactly 12 accepted"):
        collect_a4_reserved_test_cache(
            contract,
            authorization,
            encoder=SimpleNamespace(encode=lambda rows: None),
            runtime_factory=lambda *args: None,
            episode_collector=lambda **kwargs: ([], []),
            cache_stacker=lambda bundles: too_small,
        )


def test_reserved_output_rejects_non_test_instance(tmp_path: Path) -> None:
    contract, _cache_path, _summary, authorization = _authorized(tmp_path)
    bad = _cache(_panel(contract, "test", 12), "test", prefix="test")
    bad["task_instance_index"][0] = _panel(contract, "development", 1)[TASKS[0]][0]
    with pytest.raises(PermissionError, match="non-test"):
        collect_a4_reserved_test_cache(
            contract,
            authorization,
            encoder=SimpleNamespace(encode=lambda rows: None),
            runtime_factory=lambda *args: None,
            episode_collector=lambda **kwargs: ([], []),
            cache_stacker=lambda bundles: bad,
        )


def test_atomic_save_is_test_only_and_refuses_overwrite(tmp_path: Path) -> None:
    contract = _contract()
    test_cache = _cache(_panel(contract, "test", 8), "test", prefix="test")
    destination = tmp_path / "reserved-test.npz"
    save_a4_reserved_test_cache(destination, test_cache)
    loaded = load_a4_state_cache(destination)
    assert set(loaded["split"].astype(str)) == {"test"}
    original = destination.read_bytes()
    with pytest.raises(FileExistsError):
        save_a4_reserved_test_cache(destination, test_cache)
    assert destination.read_bytes() == original

    train_cache = _cache(_panel(contract, "train", 8), "train", prefix="train")
    with pytest.raises(ValueError, match="test-only"):
        save_a4_reserved_test_cache(tmp_path / "bad.npz", train_cache)


def test_cli_rejects_before_encoder_or_environment_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(contract), encoding="utf-8")
    cache_path = _write_cache(tmp_path / "formal.npz", _formal_cache(contract))
    summary = _development_go(cache_path)
    summary["decision"] = "Development-No-Go"
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    calls: list[str] = []

    def forbidden(*args, **kwargs):
        calls.append("forbidden")
        raise AssertionError("authorization must fail before encoder/environment access")

    monkeypatch.setattr(confirmation_cli, "load_frozen_a4_state_encoder", forbidden)
    monkeypatch.setattr(confirmation_cli, "make_a4_episode_runtime", forbidden)
    with pytest.raises(PermissionError, match="exact formal"):
        confirmation_cli.main(
            [
                "--config",
                str(config_path),
                "--development-go-summary",
                str(summary_path),
                "--formal-train-cache",
                str(cache_path),
                "--checkpoints",
                str(tmp_path / "seed30301" / "lewm_action_branch_a4_latest.pt"),
                str(tmp_path / "seed30302" / "lewm_action_branch_a4_latest.pt"),
                str(tmp_path / "seed30303" / "lewm_action_branch_a4_latest.pt"),
                "--output-cache",
                str(tmp_path / "test.npz"),
                "--device",
                "cpu",
            ]
        )
    assert calls == []


def test_cli_rejects_incomplete_checkpoint_panel_before_reserved_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, cache_path, summary, _authorization = _authorized(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(contract), encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    calls: list[str] = []

    def reject_checkpoints(*args, **kwargs):
        del args, kwargs
        calls.append("checkpoint")
        raise ValueError("checkpoint step must be exactly 10000")

    def forbidden(*args, **kwargs):
        del args, kwargs
        calls.append("reserved")
        raise AssertionError("reserved access must follow checkpoint validation")

    monkeypatch.setattr(
        confirmation_cli, "prepare_confirmation_checkpoints", reject_checkpoints
    )
    monkeypatch.setattr(confirmation_cli, "load_frozen_a4_state_encoder", forbidden)
    monkeypatch.setattr(confirmation_cli, "make_a4_episode_runtime", forbidden)
    with pytest.raises(ValueError, match="exactly 10000"):
        confirmation_cli.main(
            [
                "--config",
                str(config_path),
                "--development-go-summary",
                str(summary_path),
                "--formal-train-cache",
                str(cache_path),
                "--checkpoints",
                str(tmp_path / "one.pt"),
                str(tmp_path / "two.pt"),
                str(tmp_path / "three.pt"),
                "--output-cache",
                str(tmp_path / "test.npz"),
                "--device",
                "cpu",
            ]
        )
    assert calls == ["checkpoint"]


def test_cli_requires_explicit_formal_train_cache() -> None:
    with pytest.raises(SystemExit):
        confirmation_cli._parser().parse_args(
            ["--development-go-summary", "go.json"]
        )
