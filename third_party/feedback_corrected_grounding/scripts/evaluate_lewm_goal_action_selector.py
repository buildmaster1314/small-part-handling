#!/usr/bin/env python3
"""Evaluate explicit-goal action selection from frozen LeWM forecasts.

This is a CPU-only development diagnostic.  It does not train a selector and
does not claim deployment readiness: the current goal is a phase-derived task
endpoint, while recorded future anchors are used only as an evaluation label.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_goal_action_selector import (  # noqa: E402
    goal_distance_cost,
    grouped_selector_metrics,
    selector_metrics,
)
from fcpg.lewm_goal_relative_sidecar import (  # noqa: E402
    GoalRelativeActionSidecar,
    load_sidecar_checkpoint,
)


DEFAULT_CACHE = ROOT / (
    "outputs/lewm_action_prefix_a4/goal_relative_towel_sidecar_seed40572_2k/"
    "towel_frozen_lewm_features.npz"
)
DEFAULT_RAW_INDEX = ROOT / (
    "outputs/lewm_action_prefix_a4/joint_orbit12_gripper_phaseaware_a4c_v4_48train_24dev/"
    "index.npz"
)
DEFAULT_OUTPUT = ROOT / "outputs/lewm_action_prefix_a4/goal_action_selector_dev"
IMAGE_SIZE = (224, 224)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _as_strings(value: np.ndarray) -> np.ndarray:
    return np.asarray(value).astype(str)


def _verify_alignment(
    cache: Mapping[str, np.ndarray], raw: Mapping[str, np.ndarray]
) -> np.ndarray:
    required_cache = {
        "base_q",
        "target_q",
        "current_q",
        "current_valid",
        "goal_px",
        "task_index",
        "split",
        "task_instance",
        "proposal_phase_name",
        "proposal_raw_step",
        "camera_clock_hour",
    }
    missing = sorted(required_cache - set(cache))
    if missing:
        raise ValueError(f"feature cache is missing {missing}")
    required_raw = {
        "task_name",
        "task_instance",
        "proposal_raw_step",
        "proposal_phase_name",
        "camera_clock_hour",
        "split",
        "future_action",
    }
    missing = sorted(required_raw - set(raw))
    if missing:
        raise ValueError(f"raw index is missing {missing}")
    towel_rows = np.flatnonzero(_as_strings(raw["task_name"]) == "towel-wipe-v0")
    if len(towel_rows) != len(cache["base_q"]):
        raise ValueError(
            "feature cache row count does not match the raw towel subset: "
            f"{len(cache['base_q'])} versus {len(towel_rows)}"
        )
    for cache_name, raw_name in (
        ("task_instance", "task_instance"),
        ("proposal_raw_step", "proposal_raw_step"),
        ("proposal_phase_name", "proposal_phase_name"),
        ("camera_clock_hour", "camera_clock_hour"),
        ("split", "split"),
    ):
        left = _as_strings(cache[cache_name])
        right = _as_strings(raw[raw_name][towel_rows])
        if not np.array_equal(left, right):
            raise ValueError(f"feature cache/raw index alignment failed for {cache_name}")
    actions = np.asarray(raw["future_action"][towel_rows], dtype=np.float32)
    if actions.ndim != 4 or actions.shape[1] != cache["base_q"].shape[1]:
        raise ValueError(
            "raw future_action must have shape [N,C,H,A] aligned with base_q"
        )
    return actions


def _sidecar_forecast(
    cache: Mapping[str, np.ndarray],
    actions: np.ndarray,
    checkpoint: Path,
    *,
    batch_size: int,
) -> np.ndarray:
    payload = __import__("torch").load(
        checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(payload, Mapping):
        raise TypeError("sidecar checkpoint must contain one mapping")
    model = load_sidecar_checkpoint(payload).cpu().eval()
    base = np.asarray(cache["base_q"], dtype=np.float32)
    rows, candidates, horizon, anchors, coords = base.shape
    flat_count = rows * candidates
    current = np.repeat(
        np.asarray(cache["current_q"], dtype=np.float32), candidates, axis=0
    )
    valid = np.repeat(
        np.asarray(cache["current_valid"], dtype=np.bool_), candidates, axis=0
    )
    goal = np.repeat(np.asarray(cache["goal_px"], dtype=np.float32), candidates, axis=0)
    task = np.repeat(np.asarray(cache["task_index"], dtype=np.int64), candidates)
    flat_base = base.reshape(flat_count, horizon, anchors, coords)
    flat_actions = np.asarray(actions, dtype=np.float32).reshape(
        flat_count, horizon, actions.shape[-1]
    )
    chunks: list[np.ndarray] = []
    import torch

    with torch.inference_mode():
        for start in range(0, flat_count, int(batch_size)):
            stop = min(flat_count, start + int(batch_size))
            kwargs: dict[str, Any] = {
                "image_size": IMAGE_SIZE,
                "valid": torch.from_numpy(valid[start:stop]),
            }
            if isinstance(model, GoalRelativeActionSidecar):
                kwargs["future_action"] = torch.from_numpy(flat_actions[start:stop])
            prediction = model(
                torch.from_numpy(flat_base[start:stop]),
                torch.from_numpy(current[start:stop]),
                torch.from_numpy(goal[start:stop]),
                torch.from_numpy(task[start:stop]),
                **kwargs,
            )
            chunks.append(prediction.numpy())
    return np.concatenate(chunks, axis=0).reshape(
        rows, candidates, horizon, anchors, coords
    )


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Frozen LeWM goal-action selector",
        "",
        "Decision: **development diagnostic only**",
        "",
        f"- split: `{report['split']}`",
        f"- rows/snapshots: `{report['row_count']}`",
        f"- candidate count: `{report['candidate_count']}`",
        f"- horizon: `H{int(report['horizon_index']) + 1}`",
        f"- scored anchor: `{report['anchor_description']}`",
        "- goal source: phase-derived task endpoint; not a deployable visual goal",
        "- recorded future anchors are evaluation labels only and never selector inputs",
        "",
        "## Overall",
        "",
        "| method | Spearman | top-1 | top-2 | top-3 | <=1 px | regret px | random top-1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, value in report["methods"].items():
        lines.append(
            f"| {name} | {value['spearman_mean']:.4f} | "
            f"{value['top1_accuracy']:.3f} | {value['top2_accuracy']:.3f} | "
            f"{value['top3_accuracy']:.3f} | {value['within_1px_rate']:.3f} | "
            f"{value['top1_regret_mean']:.4f} | "
            f"{value['random_top1_accuracy']:.3f} |"
        )
    lines.extend(["", "## Horizon sweep", ""])
    lines.append(
        "| method | horizon | Spearman | top-1 | top-2 | <=1 px | regret px |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for name, horizons in report["horizon_sweep"].items():
        for horizon, value in horizons.items():
            lines.append(
                f"| {name} | {horizon} | {value['spearman_mean']:.4f} | "
                f"{value['top1_accuracy']:.3f} | {value['top2_accuracy']:.3f} | "
                f"{value['within_1px_rate']:.3f} | "
                f"{value['top1_regret_mean']:.4f} |"
            )
    lines.extend(["", "## Group diagnostics", ""])
    lines.append(
        "| group | method | rows | Spearman | top-1 | top-2 | <=1 px | regret px |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for group, labels in report["groups"].items():
        for label, methods in labels.items():
            for name, value in methods.items():
                lines.append(
                    f"| {group}={label} | {name} | {value['row_count']} | "
                    f"{value['spearman_mean']:.4f} | {value['top1_accuracy']:.3f} | "
                    f"{value['top2_accuracy']:.3f} | {value['within_1px_rate']:.3f} | "
                    f"{value['top1_regret_mean']:.4f} |"
                )
    lines.extend(
        [
            "",
            "This report tests whether a frozen action-conditioned forecast can rank "
            "candidate actions under an explicit development goal. It does not train "
            "a CostAdapter, run CEM, or establish a deployment result.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--raw-index", type=Path, default=DEFAULT_RAW_INDEX)
    parser.add_argument("--sidecar-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", default="development")
    parser.add_argument("--horizon-index", type=int, default=7)
    parser.add_argument("--anchor-index", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args(argv)
    if int(args.batch_size) <= 0:
        raise ValueError("batch-size must be positive")
    cache = _load_npz(args.feature_cache)
    raw = _load_npz(args.raw_index)
    actions = _verify_alignment(cache, raw)
    split = _as_strings(cache["split"])
    selected = split == str(args.split)
    if not np.any(selected):
        raise ValueError(f"no rows for split {args.split!r}")
    base = np.asarray(cache["base_q"], dtype=np.float32)[selected]
    target = np.asarray(cache["target_q"], dtype=np.float32)[selected]
    goal = np.asarray(cache["goal_px"], dtype=np.float32)[selected]
    actions = actions[selected]
    metadata = {
        "task": _as_strings(cache["task_index"])[selected],
        "phase": _as_strings(cache["proposal_phase_name"])[selected],
        "clock": _as_strings(cache["camera_clock_hour"])[selected],
    }
    base_predicted = goal_distance_cost(
        base, goal, horizon_index=int(args.horizon_index), anchor_index=args.anchor_index
    )
    actual_cost = goal_distance_cost(
        target, goal, horizon_index=int(args.horizon_index), anchor_index=args.anchor_index
    )
    methods: dict[str, dict[str, Any]] = {
        "frozen_lewm": selector_metrics(base_predicted, actual_cost)
    }
    forecasts = {"frozen_lewm": base}
    if args.sidecar_checkpoint is not None:
        sidecar = _sidecar_forecast(
            {name: np.asarray(value)[selected] for name, value in cache.items()},
            actions,
            args.sidecar_checkpoint,
            batch_size=int(args.batch_size),
        )
        sidecar_cost = goal_distance_cost(
            sidecar,
            goal,
            horizon_index=int(args.horizon_index),
            anchor_index=args.anchor_index,
        )
        methods["goal_sidecar"] = selector_metrics(sidecar_cost, actual_cost)
        forecasts["goal_sidecar"] = sidecar
    horizon_sweep: dict[str, dict[str, dict[str, Any]]] = {}
    for method_name, forecast in forecasts.items():
        horizon_sweep[method_name] = {}
        for horizon in range(forecast.shape[2]):
            predicted_cost = goal_distance_cost(
                forecast, goal, horizon_index=horizon, anchor_index=args.anchor_index
            )
            target_cost = goal_distance_cost(
                target, goal, horizon_index=horizon, anchor_index=args.anchor_index
            )
            horizon_sweep[method_name][f"H{horizon + 1}"] = selector_metrics(
                predicted_cost, target_cost
            )
    groups: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for group_name, values in metadata.items():
        groups[group_name] = {}
        for method_name, forecast in forecasts.items():
            predicted_cost = goal_distance_cost(
                forecast,
                goal,
                horizon_index=int(args.horizon_index),
                anchor_index=args.anchor_index,
            )
            method_groups = grouped_selector_metrics(
                predicted_cost, actual_cost, {group_name: values}
            )
            for label, metrics in method_groups.items():
                label_name = label.split("=", 1)[1]
                groups[group_name].setdefault(label_name, {})[method_name] = metrics
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "frozen_lewm_goal_action_selector_v1",
        "claim_boundary": "development_only",
        "deployment_claim_allowed": False,
        "goal_source": "phase_derived_task_endpoint",
        "target_q_role": "evaluation_label_only",
        "split": str(args.split),
        "row_count": int(base.shape[0]),
        "candidate_count": int(base.shape[1]),
        "horizon_index": int(args.horizon_index),
        "anchor_index": int(args.anchor_index),
        "anchor_description": (
            "tool_center" if args.anchor_index == 2 else f"anchor_{args.anchor_index}"
        ),
        "candidate_slots_anonymous": True,
        "methods": methods,
        "horizon_sweep": horizon_sweep,
        "groups": groups,
        "source": {
            "feature_cache": str(args.feature_cache),
            "raw_index": str(args.raw_index),
            "sidecar_checkpoint": (
                None if args.sidecar_checkpoint is None else str(args.sidecar_checkpoint)
            ),
        },
    }
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "summary.md").write_text(_markdown(report), encoding="utf-8")
    np.savez_compressed(
        output / "selection_arrays.npz",
        goal_px=goal,
        actual_cost=actual_cost,
        frozen_lewm_cost=base_predicted,
        target_q=target,
        task=metadata["task"],
        phase=metadata["phase"],
        clock=metadata["clock"],
    )
    print(json.dumps({"output": str(output), "methods": list(methods)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
