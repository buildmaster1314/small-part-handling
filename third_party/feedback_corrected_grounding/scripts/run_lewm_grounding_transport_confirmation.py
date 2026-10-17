#!/usr/bin/env python3
"""Run and summarize the bounded A4c grounding-transport confirmation matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "scripts/render_lewm_action_prefix_joint_a4_accuracy_gallery.py"
DEFAULT_BASE = ROOT / (
    "outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k/"
    "lewm_grounded_latest.pt"
)
DEFAULT_FCPG = ROOT / (
    "outputs/transformer_residual/a4_multiview_projection_seed40601_2k/"
    "fcpg_a4_multiview_best.pt"
)
DEFAULT_LEWM = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "camera_action_metric3d_world_history_seed40583_10k/"
    "lewm_history_conditioned_latest.pt"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/lewm_action_prefix_a4/grounding_transport_confirmation_v1"
)
CONDITIONS = {
    "clean_nominal": {
        "grounding_mode": "nominal",
        "transport": "none",
        "rotation_degrees": 0.0,
    },
    "clean_gated": {
        "grounding_mode": "fcpg",
        "transport": "magnitude-gated",
        "rotation_degrees": 0.0,
    },
    "rot5_nominal": {
        "grounding_mode": "nominal",
        "transport": "none",
        "rotation_degrees": 5.0,
    },
    "rot5_current": {
        "grounding_mode": "fcpg",
        "transport": "none",
        "rotation_degrees": 5.0,
    },
    "rot5_gated": {
        "grounding_mode": "fcpg",
        "transport": "magnitude-gated",
        "rotation_degrees": 5.0,
    },
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--fcpg-checkpoint", type=Path, default=DEFAULT_FCPG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_LEWM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--test-instance-ranks", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--clock-hours", type=int, nargs="+", default=(1, 3, 5, 7, 9, 11))
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=40721)
    parser.add_argument("--force", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.device not in {"cuda", "cuda:0"}:
        raise ValueError("confirmation permits only CUDA device 0")
    ranks = tuple(int(value) for value in args.test_instance_ranks)
    clocks = tuple(int(value) for value in args.clock_hours)
    if not ranks or len(set(ranks)) != len(ranks) or any(value < 0 or value > 11 for value in ranks):
        raise ValueError("test instance ranks must be unique values in 0..11")
    if not clocks or len(set(clocks)) != len(clocks):
        raise ValueError("clock hours must be unique")
    if any(value not in {1, 3, 5, 7, 9, 11} for value in clocks):
        raise ValueError("confirmation clock hours must be odd values 1/3/5/7/9/11")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap samples must be positive")


def _run_dir(output_dir: Path, *, rank: int, clock: int, condition: str) -> Path:
    return output_dir / f"rank{rank:02d}_clock{clock:02d}_{condition}"


def _load_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "a4_joint_visual_action_prefix_full_episode_gallery_v2":
        raise ValueError(f"unexpected renderer protocol in {path}")
    return payload


def _matches_run(
    payload: dict[str, Any], *, rank: int, clock: int, condition: str
) -> bool:
    metadata = payload.get("metadata", {})
    expected = CONDITIONS[condition]
    drift = metadata.get("camera_drift", {})
    return bool(
        int(metadata.get("test_instance_rank", -1)) == rank
        and f"clock{clock:02d}" in str(metadata.get("camera", ""))
        and metadata.get("grounding_mode") == expected["grounding_mode"]
        and metadata.get("grounding_residual_transport") == expected["transport"]
        and float(drift.get("rotation_degrees", np.nan))
        == float(expected["rotation_degrees"])
    )


def _run_one(
    args: argparse.Namespace, *, rank: int, clock: int, condition: str
) -> Path:
    output_dir = _run_dir(args.output_dir, rank=rank, clock=clock, condition=condition)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        payload = _load_metrics(metrics_path)
        if not _matches_run(payload, rank=rank, clock=clock, condition=condition):
            raise ValueError(f"existing output does not match requested run: {metrics_path}")
        return metrics_path

    expected = CONDITIONS[condition]
    command = [
        sys.executable,
        str(RENDERER),
        "--base-checkpoint",
        str(args.base_checkpoint),
        "--fcpg-checkpoint",
        str(args.fcpg_checkpoint),
        "--checkpoint",
        str(args.checkpoint),
        "--output-dir",
        str(output_dir),
        "--device",
        "cuda:0",
        "--metrics-only",
        "--test-instance-rank",
        str(rank),
        "--orbit-clock-hour",
        str(clock),
        "--model-camera-calibration",
        "nominal",
        "--grounding-mode",
        str(expected["grounding_mode"]),
        "--grounding-residual-transport",
        str(expected["transport"]),
        "--camera-drift-rotation-degrees",
        str(expected["rotation_degrees"]),
    ]
    if expected["grounding_mode"] == "fcpg":
        command.append("--allow-orbit-fcpg")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"renderer failed for rank={rank} clock={clock} condition={condition}\n"
            f"{result.stdout}"
        )
    if not metrics_path.exists():
        raise RuntimeError(f"renderer did not create {metrics_path}")
    payload = _load_metrics(metrics_path)
    if not _matches_run(payload, rank=rank, clock=clock, condition=condition):
        raise ValueError(f"new output does not match requested run: {metrics_path}")
    return metrics_path


def _episode_values(
    records: dict[tuple[int, int, str], dict[str, Any]],
    *,
    ranks: tuple[int, ...],
    clocks: tuple[int, ...],
    condition: str,
) -> dict[str, np.ndarray]:
    task_names = tuple(next(iter(records.values()))["tasks"])
    result: dict[str, np.ndarray] = {}
    for task in task_names:
        values = []
        for rank in ranks:
            per_view = [
                float(
                    records[(rank, clock, condition)]["tasks"][task][
                        "stitched_anchor_mean_epe_px"
                    ]
                )
                for clock in clocks
            ]
            values.append(float(np.mean(per_view)))
        result[task] = np.asarray(values, dtype=np.float64)
    return result


def _task_macro(values: dict[str, np.ndarray]) -> float:
    return float(np.mean([array.mean() for array in values.values()]))


def _paired_bootstrap(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    if tuple(left) != tuple(right):
        raise ValueError("paired bootstrap task sets differ")
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=np.float64)
    relative = np.empty(samples, dtype=np.float64)
    tasks = tuple(left)
    for draw in range(samples):
        left_task = []
        right_task = []
        for task in tasks:
            if left[task].shape != right[task].shape:
                raise ValueError(f"paired episode shape differs for {task}")
            indices = rng.integers(0, left[task].size, size=left[task].size)
            left_task.append(float(left[task][indices].mean()))
            right_task.append(float(right[task][indices].mean()))
        left_macro = float(np.mean(left_task))
        right_macro = float(np.mean(right_task))
        differences[draw] = left_macro - right_macro
        relative[draw] = (left_macro - right_macro) / max(left_macro, 1.0e-12)
    point_left = _task_macro(left)
    point_right = _task_macro(right)
    return {
        "left_task_macro_epe_px": point_left,
        "right_task_macro_epe_px": point_right,
        "absolute_improvement_px": point_left - point_right,
        "absolute_ci_low_px": float(np.quantile(differences, 0.025)),
        "absolute_ci_high_px": float(np.quantile(differences, 0.975)),
        "relative_improvement": (point_left - point_right) / max(point_left, 1.0e-12),
        "relative_ci_low": float(np.quantile(relative, 0.025)),
        "relative_ci_high": float(np.quantile(relative, 0.975)),
        "positive_tasks": int(
            sum(right[task].mean() < left[task].mean() for task in tasks)
        ),
        "task_count": len(tasks),
    }


def _summarize(
    records: dict[tuple[int, int, str], dict[str, Any]],
    *,
    ranks: tuple[int, ...],
    clocks: tuple[int, ...],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    values = {
        condition: _episode_values(
            records, ranks=ranks, clocks=clocks, condition=condition
        )
        for condition in CONDITIONS
    }
    drift_gated = _paired_bootstrap(
        values["rot5_nominal"],
        values["rot5_gated"],
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    drift_current = _paired_bootstrap(
        values["rot5_nominal"],
        values["rot5_current"],
        samples=bootstrap_samples,
        seed=bootstrap_seed + 1,
    )
    clean = _paired_bootstrap(
        values["clean_nominal"],
        values["clean_gated"],
        samples=bootstrap_samples,
        seed=bootstrap_seed + 2,
    )
    return {
        "schema_version": 1,
        "protocol": "a4c_grounding_transport_confirmation_v1",
        "test_instance_ranks": list(ranks),
        "clock_hours": list(clocks),
        "physical_episode_count": len(ranks) * len(next(iter(values.values()))),
        "task_view_episode_cells": len(ranks)
        * len(clocks)
        * len(next(iter(values.values()))),
        "bootstrap_samples": bootstrap_samples,
        "drift_gated_vs_nominal": drift_gated,
        "drift_current_vs_nominal": drift_current,
        "clean_gated_vs_nominal": clean,
        "per_task": {
            task: {
                condition: float(values[condition][task].mean())
                for condition in CONDITIONS
            }
            for task in values["rot5_nominal"]
        },
    }


def _markdown(summary: dict[str, Any]) -> str:
    drift = summary["drift_gated_vs_nominal"]
    current = summary["drift_current_vs_nominal"]
    clean = summary["clean_gated_vs_nominal"]
    return "\n".join(
        [
            "# A4c Grounding Transport Confirmation",
            "",
            f"- Physical episodes: `{summary['physical_episode_count']}`",
            f"- Odd views: `{summary['clock_hours']}`",
            f"- Task-view episode cells: `{summary['task_view_episode_cells']}`",
            "",
            "| Comparison | Left EPE | Right EPE | Relative change (95% CI) | Positive tasks |",
            "| --- | ---: | ---: | ---: | ---: |",
            (
                "| rot5 gated vs nominal | "
                f"{drift['left_task_macro_epe_px']:.4f} | "
                f"{drift['right_task_macro_epe_px']:.4f} | "
                f"{100 * drift['relative_improvement']:.2f}% "
                f"[{100 * drift['relative_ci_low']:.2f}%, "
                f"{100 * drift['relative_ci_high']:.2f}%] | "
                f"{drift['positive_tasks']}/{drift['task_count']} |"
            ),
            (
                "| rot5 current-only vs nominal | "
                f"{current['left_task_macro_epe_px']:.4f} | "
                f"{current['right_task_macro_epe_px']:.4f} | "
                f"{100 * current['relative_improvement']:.2f}% "
                f"[{100 * current['relative_ci_low']:.2f}%, "
                f"{100 * current['relative_ci_high']:.2f}%] | "
                f"{current['positive_tasks']}/{current['task_count']} |"
            ),
            (
                "| clean gated vs nominal | "
                f"{clean['left_task_macro_epe_px']:.4f} | "
                f"{clean['right_task_macro_epe_px']:.4f} | "
                f"{100 * clean['relative_improvement']:.2f}% "
                f"[{100 * clean['relative_ci_low']:.2f}%, "
                f"{100 * clean['relative_ci_high']:.2f}%] | "
                f"{clean['positive_tasks']}/{clean['task_count']} |"
            ),
            "",
            "Positive change means the right-hand method has lower EPE. For the clean row, a negative value is a clean penalty.",
            "",
        ]
    )


def main() -> int:
    args = _parser().parse_args()
    _validate_args(args)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranks = tuple(int(value) for value in args.test_instance_ranks)
    clocks = tuple(int(value) for value in args.clock_hours)
    total = len(ranks) * len(clocks) * len(CONDITIONS)
    records: dict[tuple[int, int, str], dict[str, Any]] = {}
    completed = 0
    for rank in ranks:
        for clock in clocks:
            for condition in CONDITIONS:
                metrics_path = _run_one(
                    args, rank=rank, clock=clock, condition=condition
                )
                records[(rank, clock, condition)] = _load_metrics(metrics_path)
                completed += 1
                print(
                    f"[{completed:03d}/{total:03d}] rank={rank} "
                    f"clock={clock:02d} condition={condition}",
                    flush=True,
                )
    summary = _summarize(
        records,
        ranks=ranks,
        clocks=clocks,
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.md").write_text(
        _markdown(summary), encoding="utf-8"
    )
    print(_markdown(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
