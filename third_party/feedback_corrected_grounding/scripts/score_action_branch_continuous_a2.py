#!/usr/bin/env python3
"""Score the A2 continuous-task 2k development screen."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a2 import PROTOCOL, validate_a2_contract  # noqa: E402
from fcpg.lewm_action_branch_a1 import load_a1_npz  # noqa: E402
from fcpg.lewm_action_branch_metrics import (  # noqa: E402
    FACTUAL_METHOD,
    HOLD_LAST_METHOD,
    LATENT_CONSTANT_VELOCITY_METHOD,
    NO_ACTION_METHOD,
    SHUFFLED_ACTION_METHOD,
    build_secondary_predictions,
    summarize_action_branch_metrics,
    whiten_latents,
)
from scripts.score_action_branch_random_hazard_a1 import (  # noqa: E402
    load_prediction_bundle,
    select_stage_rows,
)
from scripts.train_lewm_action_branch_a2 import (  # noqa: E402
    _recompute_support_authorization,
    _validate_support_authorization,
    _validate_training_contract,
)


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_continuous_task_a2.json"
DEFAULT_DATA = ROOT / "outputs/lewm_action_branch_a2/action_branch_a2_screen.npz"
DEFAULT_SUPPORT = ROOT / "outputs/lewm_action_branch_a2/support_gate/summary.json"
DEFAULT_OUTPUT = ROOT / "outputs/lewm_action_branch_a2/a2d_screen_gate"
MINIMUM_EFFECT = 0.05
FROZEN_TASKS = (
    "pick-place-v3",
    "drawer-open-v3",
    "reach-v3",
    "push-v3",
)
FROZEN_SEEDS = (25801, 25802, 25803)


def _resolve(path: Path) -> Path:
    source = path.expanduser()
    return source.resolve() if source.is_absolute() else (ROOT / source).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    source = _resolve(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {source}")
    return value


def _scalar(archive: Any, name: str) -> Any:
    if name not in archive.files:
        raise ValueError(f"A2 prediction NPZ is missing {name}")
    value = np.asarray(archive[name])
    if value.shape != ():
        raise ValueError(f"A2 prediction field {name} must be scalar")
    return value.item()


def _validate_prediction_headers(
    paths: Sequence[Path], *, expected_seeds: set[int]
) -> None:
    if len(paths) != len(expected_seeds):
        raise ValueError("A2 scoring requires exactly one NPZ per frozen seed")
    observed: list[int] = []
    for path in paths:
        source = _resolve(path)
        with np.load(source, allow_pickle=False) as archive:
            if str(_scalar(archive, "protocol")) != PROTOCOL:
                raise ValueError(f"{source}: prediction protocol is not A2")
            if int(_scalar(archive, "checkpoint_step")) != 2000:
                raise ValueError(f"{source}: A2 scoring requires latest step 2000")
            if str(_scalar(archive, "selection")) != "fixed_latest":
                raise ValueError(f"{source}: A2 scoring requires fixed_latest")
            if bool(_scalar(archive, "prediction_heads_add_sigreg")):
                raise ValueError(f"{source}: A2 prediction heads must not add SIGReg")
            if "training_seed" not in archive.files:
                raise ValueError(f"{source}: prediction NPZ is missing training_seed")
            seeds = np.unique(np.asarray(archive["training_seed"], dtype=np.int64))
            if seeds.size != 1:
                raise ValueError(f"{source}: prediction NPZ mixes training seeds")
            observed.append(int(seeds[0]))
    if len(set(observed)) != len(observed) or set(observed) != expected_seeds:
        raise ValueError(
            f"A2 prediction seeds must be {sorted(expected_seeds)}, got {sorted(observed)}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--support-summary", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _map_task_names(task_index: np.ndarray, task_names: Sequence[str]) -> np.ndarray:
    values = np.asarray(task_index)
    if values.ndim != 1 or values.dtype == np.bool_ or not np.issubdtype(
        values.dtype, np.integer
    ):
        raise TypeError("A2 task_index must be a one-dimensional integer array")
    if np.any(values < 0) or np.any(values >= len(task_names)):
        raise ValueError("A2 task_index is outside the configured task list")
    return np.asarray([task_names[int(value)] for value in values.tolist()])


def _task_macro_raw_mse(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    task: np.ndarray,
    episode: np.ndarray,
    seed: np.ndarray,
    moments: Any,
    horizons: Sequence[int] = (6, 7, 8),
) -> dict[str, Any]:
    indices = np.asarray([int(value) - 1 for value in horizons], dtype=np.int64)
    target_w = whiten_latents(target, moments, label="A2 target")[:, :, indices]
    prediction_w = whiten_latents(
        prediction, moments, label="A2 prediction"
    )[:, :, indices]
    row_loss = np.mean(
        np.square(target_w - prediction_w), axis=(1, 2, 3, 4), dtype=np.float64
    )
    tasks = tuple(dict.fromkeys(np.asarray(task).astype(str).tolist()))
    per_task: dict[str, float] = {}
    for name in tasks:
        task_rows = np.asarray(task).astype(str) == name
        episode_values: list[float] = []
        for episode_id in np.unique(np.asarray(episode)[task_rows].astype(str)):
            rows = task_rows & (np.asarray(episode).astype(str) == episode_id)
            seed_values = [
                float(np.mean(row_loss[rows & (np.asarray(seed) == seed_id)]))
                for seed_id in np.unique(np.asarray(seed)[rows])
            ]
            episode_values.append(float(np.mean(seed_values)))
        per_task[name] = float(np.mean(episode_values))
    return {"task_macro_mse": float(np.mean(list(per_task.values()))), "per_task": per_task}


def _grouped_task_seed_losses(
    row_loss: np.ndarray,
    *,
    task: np.ndarray,
    episode: np.ndarray,
    seed: np.ndarray,
) -> dict[str, dict[int, float]]:
    task_values = np.asarray(task).astype(str)
    episode_values = np.asarray(episode).astype(str)
    seed_values = np.asarray(seed, dtype=np.int64)
    result: dict[str, dict[int, float]] = {}
    for task_name in tuple(dict.fromkeys(task_values.tolist())):
        result[task_name] = {}
        for seed_id in np.unique(seed_values):
            selected = (task_values == task_name) & (seed_values == seed_id)
            losses = [
                float(np.mean(row_loss[selected & (episode_values == episode_id)]))
                for episode_id in np.unique(episode_values[selected])
            ]
            result[task_name][int(seed_id)] = float(np.mean(losses))
    return result


def _shuffle_shift_report(
    target: np.ndarray,
    factual: np.ndarray,
    shuffled: np.ndarray,
    *,
    task: np.ndarray,
    episode: np.ndarray,
    seed: np.ndarray,
    moments: Any,
) -> dict[str, Any]:
    indices = np.asarray([5, 6, 7], dtype=np.int64)
    target_w = whiten_latents(target, moments, label="A2 target")[:, :, indices]
    factual_w = whiten_latents(factual, moments, label="A2 factual")[:, :, indices]
    target_c = target_w - np.mean(target_w, axis=1, keepdims=True, dtype=np.float64)
    factual_c = factual_w - np.mean(factual_w, axis=1, keepdims=True, dtype=np.float64)
    factual_loss = np.mean(
        np.square(target_c - factual_c), axis=(1, 2, 3, 4), dtype=np.float64
    )
    factual_grouped = _grouped_task_seed_losses(
        factual_loss, task=task, episode=episode, seed=seed
    )
    if shuffled.ndim != 6 or shuffled.shape[1] != 7:
        raise ValueError("A2 requires all seven non-identity shuffled-action shifts")
    per_shift: list[dict[str, Any]] = []
    for shift in range(7):
        prediction_w = whiten_latents(
            shuffled[:, shift], moments, label=f"A2 shuffled shift {shift + 1}"
        )[:, :, indices]
        prediction_c = prediction_w - np.mean(
            prediction_w, axis=1, keepdims=True, dtype=np.float64
        )
        baseline_loss = np.mean(
            np.square(target_c - prediction_c),
            axis=(1, 2, 3, 4),
            dtype=np.float64,
        )
        baseline_grouped = _grouped_task_seed_losses(
            baseline_loss, task=task, episode=episode, seed=seed
        )
        task_effects: dict[str, float] = {}
        for task_name in factual_grouped:
            factual_task = float(np.mean(list(factual_grouped[task_name].values())))
            baseline_task = float(np.mean(list(baseline_grouped[task_name].values())))
            if baseline_task == 0.0:
                if factual_task == 0.0:
                    task_effects[task_name] = 0.0
                    continue
                raise ValueError(
                    f"shuffled shift {shift + 1}/{task_name} has zero baseline "
                    "loss with positive factual loss"
                )
            task_effects[task_name] = 1.0 - factual_task / baseline_task
        per_shift.append(
            {
                "donor_shift": shift + 1,
                "effect": float(np.mean(list(task_effects.values()))),
                "per_task": task_effects,
            }
        )
    return {
        "per_shift": per_shift,
        "minimum_effect": min(float(row["effect"]) for row in per_shift),
    }


def _validate_prediction_rows(
    arrays: Mapping[str, np.ndarray],
    *,
    train_data: Path,
    expected_seeds: set[int],
) -> None:
    canonical = load_a1_npz(_resolve(train_data))
    development = np.asarray(canonical["split"]).astype(str) == "development"
    canonical_keys = [
        (int(task), str(episode), int(hazard))
        for task, episode, hazard in zip(
            np.asarray(canonical["task_index"])[development],
            np.asarray(canonical["source_episode_id"])[development],
            np.asarray(canonical["hazard_id"])[development],
            strict=True,
        )
    ]
    if len(set(canonical_keys)) != len(canonical_keys):
        raise ValueError("canonical A2 development rows are not uniquely identified")
    canonical_index = {key: index for index, key in enumerate(canonical_keys)}
    target = np.asarray(canonical["target_embedding"])[development]
    history = np.asarray(canonical["history_embedding"])[development]
    task_instance = np.asarray(canonical["task_instance_index"])[development]
    observed_seed = np.asarray(arrays["seed"], dtype=np.int64)
    for seed_id in sorted(expected_seeds):
        rows = np.flatnonzero(observed_seed == seed_id)
        keys = [
            (int(arrays["task_index"][row]), str(arrays["source_episode_id"][row]), int(arrays["hazard_id"][row]))
            for row in rows.tolist()
        ]
        if len(set(keys)) != len(keys) or set(keys) != set(canonical_keys):
            raise ValueError(f"A2 seed {seed_id} does not cover canonical development rows")
        order = np.asarray([canonical_index[key] for key in keys], dtype=np.int64)
        if not np.array_equal(np.asarray(arrays["target"])[rows], target[order]):
            raise ValueError(f"A2 seed {seed_id} target differs from canonical data")
        if not np.array_equal(np.asarray(arrays["history"])[rows], history[order]):
            raise ValueError(f"A2 seed {seed_id} history differs from canonical data")
        if not np.array_equal(
            np.asarray(arrays["task_instance_index"])[rows], task_instance[order]
        ):
            raise ValueError(
                f"A2 seed {seed_id} task instance differs from canonical data"
            )


def _a2_gate(
    summary: Mapping[str, Any],
    *,
    support: Mapping[str, Any],
    raw: Mapping[str, Mapping[str, Any]],
    shuffle_shifts: Mapping[str, Any],
) -> dict[str, Any]:
    comparisons = summary["late_window"]["comparisons"]
    required = (NO_ACTION_METHOD, SHUFFLED_ACTION_METHOD)
    tasks = tuple(comparisons[NO_ACTION_METHOD]["per_task"])
    seeds = tuple(comparisons[NO_ACTION_METHOD]["per_seed"])
    if set(tasks) != set(FROZEN_TASKS):
        raise ValueError("A2 Gate requires exactly the four frozen tasks")
    if set(support.get("per_task", {})) != set(FROZEN_TASKS):
        raise ValueError("A2 support summary must contain exactly four frozen tasks")
    if set(int(value) for value in seeds) != set(FROZEN_SEEDS):
        raise ValueError("A2 Gate requires exactly the three frozen seeds")
    joint_tasks = [
        task
        for task in tasks
        if all(float(comparisons[name]["per_task"][task]["effect"]) > 0.0 for name in required)
    ]
    joint_seeds = [
        seed
        for seed in seeds
        if all(float(comparisons[name]["per_seed"][seed]["effect"]) > 0.0 for name in required)
    ]
    strong_seeds = [
        seed
        for seed in seeds
        if sum(
            all(
                float(comparisons[name]["per_task"][task]["per_seed"][seed]["effect"])
                > 0.0
                for name in required
            )
            for task in tasks
        )
        >= 3
    ]
    checks = {
        "late_effect_vs_no_action": float(comparisons[NO_ACTION_METHOD]["effect"]) >= MINIMUM_EFFECT,
        "late_effect_vs_shuffled_action": float(comparisons[SHUFFLED_ACTION_METHOD]["effect"]) >= MINIMUM_EFFECT,
        "late_effect_vs_every_shuffled_shift": float(shuffle_shifts["minimum_effect"])
        >= MINIMUM_EFFECT,
        "jointly_positive_tasks": len(joint_tasks) >= 3,
        "jointly_positive_seeds": len(joint_seeds) >= 2,
        "two_seeds_positive_on_three_tasks": len(strong_seeds) >= 2,
        "late_support_all_tasks": all(
            row.get("passed") is True for row in support["per_task"].values()
        ),
        "raw_not_worse_than_no_action": raw[FACTUAL_METHOD]["task_macro_mse"]
        <= raw[NO_ACTION_METHOD]["task_macro_mse"],
        "raw_better_than_hold_last": raw[FACTUAL_METHOD]["task_macro_mse"]
        < raw[HOLD_LAST_METHOD]["task_macro_mse"],
        "raw_better_than_constant_velocity": raw[FACTUAL_METHOD]["task_macro_mse"]
        < raw[LATENT_CONSTANT_VELOCITY_METHOD]["task_macro_mse"],
    }
    passed = all(checks.values())
    return {
        "stage": "a2_2k_development",
        "decision": "Development-Go" if passed else "Development-No-Go",
        "passed": passed,
        "minimum_relative_improvement": MINIMUM_EFFECT,
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "jointly_positive_tasks": joint_tasks,
        "jointly_positive_seeds": [int(value) for value in joint_seeds],
        "strong_seed_task_support": [int(value) for value in strong_seeds],
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    gate = payload["a2_gate"]
    comparisons = payload["metrics"]["late_window"]["comparisons"]
    raw = payload["raw_h6_h8"]
    lines = [
        "# A2 continuous-task 2k development screen",
        "",
        f"Decision: **{gate['decision']}**",
        "",
        "| comparison | H6-H8 relative improvement |",
        "| --- | ---: |",
        f"| factual vs matched no-action | {comparisons[NO_ACTION_METHOD]['effect']:.2%} |",
        f"| factual vs shuffled-action | {comparisons[SHUFFLED_ACTION_METHOD]['effect']:.2%} |",
        "",
        "| raw H6-H8 method | task-macro MSE |",
        "| --- | ---: |",
    ]
    for name, row in raw.items():
        lines.append(f"| {name} | {row['task_macro_mse']:.6f} |")
    if gate["failed_checks"]:
        lines.extend(["", "Failed: " + ", ".join(gate["failed_checks"])])
    lines.extend(
        [
            "",
            "This is a development-only decision. The original all-horizon support "
            "No-Go remains recorded; H6-H8 was selected to test accumulated action dynamics.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = _read_json(args.config)
    task_names = _validate_training_contract(contract)
    if task_names != FROZEN_TASKS:
        raise ValueError(f"A2 scoring tasks must remain fixed at {FROZEN_TASKS}")
    _validate_support_authorization(args.support_summary, task_names=task_names)
    support = _recompute_support_authorization(
        args.train_data,
        contract=contract,
        task_names=task_names,
    )
    arrays, predictions, moments = load_prediction_bundle(
        args.predictions, train_moments_path=args.train_data
    )
    arrays, predictions = select_stage_rows(arrays, predictions, stage="screen")
    arrays["task"] = _map_task_names(arrays["task_index"], task_names)
    expected_seeds = set(FROZEN_SEEDS)
    _validate_prediction_headers(args.predictions, expected_seeds=expected_seeds)
    observed_seeds = set(np.asarray(arrays["seed"], dtype=np.int64).tolist())
    if observed_seeds != expected_seeds:
        raise ValueError(
            f"A2 screen requires seeds {sorted(expected_seeds)}, got {sorted(observed_seeds)}"
        )
    metrics = summarize_action_branch_metrics(
        arrays["target"],
        predictions,
        arrays["history"],
        task=arrays["task"],
        task_instance_index=arrays["task_instance_index"],
        source_episode_id=arrays["source_episode_id"],
        hazard_id=arrays["hazard_id"],
        split=arrays["split"],
        training_seed=arrays["seed"],
        moments=moments,
        shuffle_permutation=arrays["shuffle_permutation"],
        stage="screen",
        bootstrap_samples=0,
        minimum_branch_variance=float(contract["support_gate"]["minimum_branch_variance"]),
    )
    if set(metrics["coverage"]["task_names"]) != set(FROZEN_TASKS):
        raise ValueError("A2 screen must cover exactly the four frozen tasks")
    if metrics["coverage"]["episodes_per_task"] != {
        task: 12 for task in metrics["coverage"]["task_names"]
    }:
        raise ValueError("A2 screen requires exactly 12 development episodes per task")
    _validate_prediction_rows(
        arrays,
        train_data=args.train_data,
        expected_seeds=expected_seeds,
    )
    generated = build_secondary_predictions(
        arrays["history"],
        branch_count=arrays["target"].shape[1],
        horizon_count=arrays["target"].shape[2],
    )
    raw_predictions = {
        FACTUAL_METHOD: predictions[FACTUAL_METHOD],
        NO_ACTION_METHOD: predictions[NO_ACTION_METHOD],
        HOLD_LAST_METHOD: generated[HOLD_LAST_METHOD],
        LATENT_CONSTANT_VELOCITY_METHOD: generated[LATENT_CONSTANT_VELOCITY_METHOD],
    }
    raw = {
        name: _task_macro_raw_mse(
            arrays["target"],
            prediction,
            task=arrays["task"],
            episode=arrays["source_episode_id"],
            seed=arrays["seed"],
            moments=moments,
        )
        for name, prediction in raw_predictions.items()
    }
    shuffle_shifts = _shuffle_shift_report(
        arrays["target"],
        predictions[FACTUAL_METHOD],
        predictions[SHUFFLED_ACTION_METHOD],
        task=arrays["task"],
        episode=arrays["source_episode_id"],
        seed=arrays["seed"],
        moments=moments,
    )
    gate = _a2_gate(
        metrics,
        support=support,
        raw=raw,
        shuffle_shifts=shuffle_shifts,
    )
    payload = {
        "protocol": PROTOCOL,
        "stage": "a2_2k_development",
        "decision": gate["decision"],
        "passed": gate["passed"],
        "a2_gate": gate,
        "raw_h6_h8": raw,
        "shuffled_action_h6_h8": shuffle_shifts,
        "metrics": metrics,
    }
    output = _resolve(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "summary.md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"decision": gate["decision"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
