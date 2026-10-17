from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from fcpg.visual_trust_corruptions import CORRUPTION_TYPES


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_evaluate_visual_trust.py"
SPEC = importlib.util.spec_from_file_location("visual_trust_runner_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _method_summary(auroc: float) -> dict[str, object]:
    family = {
        name: {
            "auroc": auroc,
            "unsafe_recall": 0.90,
            "early_reopen_rate": 0.05,
        }
        for name in CORRUPTION_TYPES
    }
    return {
        "overall": {
            "auroc": auroc,
            "unsafe_recall": 0.90,
            "early_reopen_rate": 0.05,
        },
        "per_family": family,
    }


def _seed_result() -> dict[str, object]:
    recovery_group = {
        "event_count": 4,
        "detection_rate": 1.0,
        "median_delay": 1.0,
    }
    return {
        "classification": {
            "visual": _method_summary(0.90),
            "auxiliary_only": _method_summary(0.70),
            "geometry": _method_summary(0.60),
            "temporal": _method_summary(0.75),
        },
        "independent_clean_false_reject_rate": 0.04,
        "independent_clean_false_reject_rate_by_task": {
            "task_a": 0.04,
            "task_b": 0.04,
        },
        "recovery": {
            "eligible_rate": 1.0,
            "detection_rate": 1.0,
            "median_delay": 1.0,
            "per_family": {
                name: dict(recovery_group) for name in CORRUPTION_TYPES
            },
            "per_task": {
                "task_a": dict(recovery_group),
                "task_b": dict(recovery_group),
            },
        },
    }


def test_decision_rejects_missing_corruption_family() -> None:
    results = [_seed_result() for _ in range(3)]
    for result in results:
        result["classification"]["visual"]["per_family"].pop(
            CORRUPTION_TYPES[0]
        )
    decision = RUNNER._decision(results)
    assert not decision["go"]
    assert decision["discrimination_pass_count"] == 0


def test_decision_requires_margin_over_learned_auxiliary_control() -> None:
    results = [_seed_result() for _ in range(3)]
    for result in results:
        result["classification"]["auxiliary_only"]["overall"]["auroc"] = 0.88
    decision = RUNNER._decision(results)
    assert not decision["go"]
    assert decision["discrimination_pass_count"] == 0


def _write_identity_data(
    path: Path, *, task: str, env_seed: int, episode: int
) -> None:
    np.savez_compressed(
        path,
        task_name=np.asarray([task, task]),
        env_seed=np.asarray([env_seed, env_seed], dtype=np.int64),
        episode_id=np.asarray([episode, episode], dtype=np.int64),
    )


def test_physical_split_check_rejects_episode_overlap(tmp_path: Path) -> None:
    train = tmp_path / "train.npz"
    development = tmp_path / "development.npz"
    clean = tmp_path / "clean.npz"
    _write_identity_data(train, task="task", env_seed=10, episode=1)
    _write_identity_data(development, task="task", env_seed=10, episode=1)
    _write_identity_data(clean, task="task", env_seed=12, episode=3)

    try:
        RUNNER._assert_disjoint_physical_splits(train, development, clean)
    except ValueError as exc:
        assert "physical episode overlap" in str(exc)
    else:
        raise AssertionError("overlapping physical episodes were accepted")
