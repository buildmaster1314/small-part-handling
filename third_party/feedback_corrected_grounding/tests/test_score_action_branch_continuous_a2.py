from __future__ import annotations

import numpy as np
import pytest

from scripts.score_action_branch_continuous_a2 import (
    FROZEN_SEEDS,
    FROZEN_TASKS,
    _a2_gate,
    _map_task_names,
    _shuffle_shift_report,
    _validate_prediction_headers,
)
from fcpg.lewm_action_branch_metrics import fit_train_moments


def _comparison(effect: float) -> dict:
    tasks = FROZEN_TASKS
    seeds = (25801, 25802, 25803)
    return {
        "effect": effect,
        "per_task": {
            task: {
                "effect": effect,
                "per_seed": {
                    str(seed): {"effect": effect} for seed in seeds
                },
            }
            for task in tasks
        },
        "per_seed": {str(seed): {"effect": effect} for seed in seeds},
    }


def test_task_index_mapping_uses_a2_task_order() -> None:
    tasks = ("pick", "drawer", "reach", "push")
    mapped = _map_task_names(np.asarray([0, 1, 2, 3]), tasks)
    assert mapped.tolist() == list(tasks)


def test_a2_gate_uses_late_effects_and_raw_guards() -> None:
    metrics = {
        "late_window": {
            "comparisons": {
                "no_action": _comparison(0.10),
                "shuffled_action": _comparison(0.20),
            }
        }
    }
    support = {"per_task": {task: {"passed": True} for task in FROZEN_TASKS}}
    raw = {
        "factual": {"task_macro_mse": 0.5},
        "no_action": {"task_macro_mse": 0.6},
        "hold_last": {"task_macro_mse": 0.7},
        "latent_constant_velocity": {"task_macro_mse": 0.8},
    }
    shifts = {"minimum_effect": 0.15}
    result = _a2_gate(metrics, support=support, raw=raw, shuffle_shifts=shifts)
    assert result["decision"] == "Development-Go"
    raw["hold_last"]["task_macro_mse"] = 0.4
    result = _a2_gate(metrics, support=support, raw=raw, shuffle_shifts=shifts)
    assert result["decision"] == "Development-No-Go"
    assert "raw_better_than_hold_last" in result["failed_checks"]


def test_a2_gate_rejects_missing_tasks_and_a_weak_shuffle_shift() -> None:
    metrics = {
        "late_window": {
            "comparisons": {
                "no_action": _comparison(0.10),
                "shuffled_action": _comparison(0.20),
            }
        }
    }
    support = {"per_task": {task: {"passed": True} for task in FROZEN_TASKS}}
    raw = {
        "factual": {"task_macro_mse": 0.5},
        "no_action": {"task_macro_mse": 0.6},
        "hold_last": {"task_macro_mse": 0.7},
        "latent_constant_velocity": {"task_macro_mse": 0.8},
    }
    missing = {
        "late_window": {
            "comparisons": {
                name: {
                    **row,
                    "per_task": {
                        task: value
                        for task, value in row["per_task"].items()
                        if task != "drawer-open-v3"
                    },
                }
                for name, row in metrics["late_window"]["comparisons"].items()
            }
        }
    }
    with pytest.raises(ValueError, match="four frozen tasks"):
        _a2_gate(missing, support=support, raw=raw, shuffle_shifts={"minimum_effect": 0.2})

    result = _a2_gate(
        metrics,
        support=support,
        raw=raw,
        shuffle_shifts={"minimum_effect": 0.01},
    )
    assert result["decision"] == "Development-No-Go"
    assert "late_effect_vs_every_shuffled_shift" in result["failed_checks"]


def test_a2_gate_rejects_non_boolean_support_and_wrong_seed_panel() -> None:
    metrics = {
        "late_window": {
            "comparisons": {
                "no_action": _comparison(0.10),
                "shuffled_action": _comparison(0.20),
            }
        }
    }
    support = {
        "per_task": {task: {"passed": "false"} for task in FROZEN_TASKS}
    }
    raw = {
        "factual": {"task_macro_mse": 0.5},
        "no_action": {"task_macro_mse": 0.6},
        "hold_last": {"task_macro_mse": 0.7},
        "latent_constant_velocity": {"task_macro_mse": 0.8},
    }
    result = _a2_gate(
        metrics,
        support=support,
        raw=raw,
        shuffle_shifts={"minimum_effect": 0.2},
    )
    assert not result["checks"]["late_support_all_tasks"]

    bad = _comparison(0.10)
    bad["per_seed"].pop(str(FROZEN_SEEDS[-1]))
    for task in FROZEN_TASKS:
        bad["per_task"][task]["per_seed"].pop(str(FROZEN_SEEDS[-1]))
    metrics["late_window"]["comparisons"]["no_action"] = bad
    with pytest.raises(ValueError, match="three frozen seeds"):
        _a2_gate(
            metrics,
            support={
                "per_task": {task: {"passed": True} for task in FROZEN_TASKS}
            },
            raw=raw,
            shuffle_shifts={"minimum_effect": 0.2},
        )


def test_zero_loss_shuffle_shift_is_zero_effect_not_a_false_win() -> None:
    target = np.zeros((1, 8, 8, 1, 2), dtype=np.float32)
    shuffled = np.zeros((1, 7, 8, 8, 1, 2), dtype=np.float32)
    report = _shuffle_shift_report(
        target,
        target.copy(),
        shuffled,
        task=np.asarray([FROZEN_TASKS[0]]),
        episode=np.asarray(["episode-0"]),
        seed=np.asarray([FROZEN_SEEDS[0]], dtype=np.int64),
        moments=fit_train_moments(
            np.arange(1 * 8 * 8 * 1 * 2, dtype=np.float32).reshape(1, 8, 8, 1, 2)
        ),
    )
    assert report["minimum_effect"] == 0.0
    assert all(row["effect"] == 0.0 for row in report["per_shift"])


def _prediction_header(path, *, seed: int, step: int = 2000) -> None:
    np.savez_compressed(
        path,
        protocol=np.asarray("fcpg_action_branch_continuous_task_a2_v1"),
        checkpoint_step=np.asarray(step, dtype=np.int64),
        selection=np.asarray("fixed_latest"),
        prediction_heads_add_sigreg=np.asarray(False),
        training_seed=np.full(3, seed, dtype=np.int64),
    )


def test_prediction_headers_require_three_fixed_latest_endpoints(tmp_path) -> None:
    seeds = {25801, 25802, 25803}
    paths = []
    for seed in sorted(seeds):
        path = tmp_path / f"seed{seed}.npz"
        _prediction_header(path, seed=seed)
        paths.append(path)
    _validate_prediction_headers(paths, expected_seeds=seeds)

    _prediction_header(paths[0], seed=25801, step=1999)
    with pytest.raises(ValueError, match="step 2000"):
        _validate_prediction_headers(paths, expected_seeds=seeds)
