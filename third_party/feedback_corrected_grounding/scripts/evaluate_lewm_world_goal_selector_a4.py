#!/usr/bin/env python3
"""Screen frozen LeWM action ranking against an explicit Towel world goal."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.custom_tasks.towel_wipe import (  # noqa: E402
    SawyerTowelWipeEnvV0,
    TOWEL_WIPE_TASK_NAME,
)
from fcpg.lewm_action_branch_a4 import HORIZON, TASKS  # noqa: E402
from fcpg.lewm_action_prefix_joint_data import open_raw_replay_arrays  # noqa: E402
from fcpg.lewm_goal_action_selector import (  # noqa: E402
    average_costs_by_physical_snapshot,
    integrated_cartesian_action_forecast,
    selector_metrics,
    world_goal_distance_cost,
)
from fcpg.lewm_goal_relative_sidecar import towel_goal_world  # noqa: E402
from scripts import render_lewm_action_prefix_joint_a4_accuracy_gallery as gallery  # noqa: E402
from scripts.train_lewm_action_prefix_joint_a4 import _move  # noqa: E402
from scripts.train_lewm_camera_action_joint_a4 import _validate_dataset  # noqa: E402
from scripts.train_lewm_history_conditioned_metric3d_a4 import (  # noqa: E402
    _batch_from_memmaps,
    _history_array,
)


DEFAULT_RAW = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "joint_orbit12_gripper_phaseaware_a4c_v4_48train_24dev"
)
DEFAULT_SOURCE = ROOT / "outputs/lewm_action_branch_a4c_v4/state_cache.npz"
DEFAULT_CHECKPOINT = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "camera_action_metric3d_world_history_seed40583_10k/"
    "lewm_history_conditioned_latest.pt"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/lewm_action_prefix_a4/world_goal_selector_towel_dev_h3"
)
BRANCH_COUNT = 8
PRIMARY_HORIZON_INDEX = 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", default="development")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _cuda0(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("world-goal selector evaluation permits only CUDA0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def _wipe_bundles(index: Mapping[str, np.ndarray], split: str) -> np.ndarray:
    task = np.asarray(index["task_name"]).astype(str)
    splits = np.asarray(index["split"]).astype(str)
    phase = np.char.lower(np.asarray(index["proposal_phase_name"]).astype(str))
    selected = (
        (task == TOWEL_WIPE_TASK_NAME)
        & (splits == str(split))
        & (np.char.find(phase, "wipe_") >= 0)
    )
    result = np.flatnonzero(selected).astype(np.int64)
    if result.size == 0:
        raise ValueError("no Towel wipe-phase bundles match the requested split")
    return result


def _goal_tool_centers(
    index: Mapping[str, np.ndarray], bundles: np.ndarray
) -> np.ndarray:
    instances = np.asarray(index["task_instance"], dtype=np.int64)[bundles]
    phases = np.asarray(index["proposal_phase_name"]).astype(str)[bundles]
    goals = np.stack(
        [
            towel_goal_world(int(instance), phase)
            + SawyerTowelWipeEnvV0.GRIP_OFFSET.astype(np.float32)
            for instance, phase in zip(instances, phases, strict=True)
        ]
    )
    return np.ascontiguousarray(goals, dtype=np.float32)


@torch.inference_mode()
def _forecast_batch(
    model: Any,
    index: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    history: np.ndarray,
    bundles: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    branches = np.zeros(len(bundles), dtype=np.int64)
    batch = _move(
        _batch_from_memmaps(
            index,
            arrays,
            history,
            bundles=bundles,
            branches=branches,
        ),
        device,
    )
    candidate_action = torch.from_numpy(
        np.ascontiguousarray(
            np.asarray(index["future_action"], dtype=np.float32)[bundles]
        )
    ).to(device)
    batch_size, candidate_count, horizon, action_dim = candidate_action.shape
    if candidate_count != BRANCH_COUNT or horizon != HORIZON:
        raise ValueError("candidate action bank must have shape [B,8,8,A]")
    cache_index = np.asarray(index["cache_index"], dtype=np.int64)[bundles]
    history_action = torch.from_numpy(
        np.ascontiguousarray(history[cache_index])
    ).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        current_state = model.encode_observations(
            rgb=batch["current_rgb"][:, None],
            proprio=batch["current_proprio"][:, None],
            q_nominal=batch["current_q_nominal"][:, None],
            q_corrected=batch["current_q_corrected"][:, None],
            keypoints_xyz=batch["current_keypoints_xyz"][:, None],
            valid=batch["current_valid"][:, None],
        )[:, 0]
        flat_state = current_state[:, None].expand(
            -1, candidate_count, -1, -1
        ).reshape(batch_size * candidate_count, *current_state.shape[1:])
        flat_action = candidate_action.reshape(
            batch_size * candidate_count, horizon, action_dim
        )
        flat_history = history_action[:, None].expand(
            -1, candidate_count, -1, -1
        ).reshape(batch_size * candidate_count, *history_action.shape[1:])
        flat_task = batch["task_index"][:, None].expand(
            -1, candidate_count
        ).reshape(-1)
        flat_extrinsic = batch["camera_extrinsic_cam_from_world"][:, None].expand(
            -1, candidate_count, -1, -1
        ).reshape(batch_size * candidate_count, 4, 4)
        flat_intrinsic = batch["camera_intrinsic"][:, None].expand(
            -1, candidate_count, -1, -1
        ).reshape(batch_size * candidate_count, 3, 3)
        flat_current_xyz = batch["current_keypoints_xyz"][:, None].expand(
            -1, candidate_count, -1, -1
        ).reshape(batch_size * candidate_count, 3, 3)

        def decode(actions: torch.Tensor) -> torch.Tensor:
            predicted_state = model.predict_with_history(
                current_state=flat_state,
                history_action=flat_history,
                future_action=actions,
                task_index=flat_task,
                batch={"camera_extrinsic_cam_from_world": flat_extrinsic},
            )
            return model.decode_3d(
                predicted_state=predicted_state,
                current_state=flat_state,
                current_keypoints_xyz=flat_current_xyz,
                future_action=actions,
                task_index=flat_task,
                camera_intrinsic=flat_intrinsic,
                extrinsic_cam_from_world=flat_extrinsic,
            ).predicted_world_xyz

        factual = decode(flat_action)
        wrong_action = torch.roll(candidate_action, shifts=1, dims=1).reshape_as(
            flat_action
        )
        wrong = decode(wrong_action)
    shape = (batch_size, candidate_count, horizon, 3, 3)
    actual = np.asarray(arrays["future_keypoints_xyz"])[bundles]
    return (
        factual.float().cpu().numpy().reshape(shape),
        wrong.float().cpu().numpy().reshape(shape),
        np.asarray(actual, dtype=np.float32),
        np.asarray(batch["current_keypoints_xyz"].float().cpu(), dtype=np.float32),
    )


def _scaled_metrics(predicted: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    result = selector_metrics(predicted, actual)
    for name in (
        "within_0_25px_rate",
        "within_0_5px_rate",
        "within_1px_rate",
        "within_2px_rate",
    ):
        result.pop(name, None)
    for name in (
        "selected_cost_mean",
        "oracle_best_cost_mean",
        "top1_regret_mean",
        "top1_regret_median",
        "actual_candidate_spread_mean",
    ):
        result[f"{name}_mm"] = 1000.0 * float(result.pop(name))
    return result


def _report(
    methods_by_horizon: Mapping[str, Mapping[str, Any]],
    *,
    view_methods_by_horizon: Mapping[str, Mapping[str, Any]],
    row_count: int,
    physical_snapshot_count: int,
    clocks: list[int],
) -> dict[str, Any]:
    primary = methods_by_horizon["H3"]
    factual = primary["factual_lewm"]
    kinematic = primary["kinematic_action"]
    wrong = primary["cyclic_wrong_action"]
    factual_regret = float(factual["top1_regret_mean_mm"])
    kinematic_regret = float(kinematic["top1_regret_mean_mm"])
    relative = (kinematic_regret - factual_regret) / max(kinematic_regret, 1.0e-12)
    go = bool(
        relative >= 0.10
        and float(factual["spearman_mean"]) > float(kinematic["spearman_mean"])
        and float(factual["spearman_mean"]) > float(wrong["spearman_mean"])
    )
    return {
        "schema_version": 1,
        "protocol": "a4_towel_world_goal_action_selector_v1",
        "decision": "development_go" if go else "development_no_go",
        "go_rule": (
            "H3 factual regret improves >=10% over analytic kinematics and "
            "factual Spearman exceeds both kinematic and cyclic-wrong-action"
        ),
        "row_count": int(row_count),
        "physical_snapshot_count": int(physical_snapshot_count),
        "camera_clock_hours": clocks,
        "primary_horizon": "H3",
        "primary_statistical_unit": "physical_snapshot_after_camera_view_average",
        "factual_vs_kinematic_regret_relative_improvement": relative,
        "methods_by_horizon": methods_by_horizon,
        "view_row_diagnostics_by_horizon": view_methods_by_horizon,
        "interpretation": "zero_training_development_screen_only",
        "uses_future_state_for_selection": False,
        "actual_future_state_is_label_only": True,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Towel world-goal action selector",
        "",
        f"Decision: **{report['decision']}**",
        "",
        f"- view rows: `{report['row_count']}`",
        f"- physical snapshots: `{report['physical_snapshot_count']}`",
        f"- camera clocks: `{report['camera_clock_hours']}`",
        "- goal: current task endpoint in world coordinates",
        "- actual future anchors: evaluation labels only",
        "- primary unit: physical snapshot after averaging six camera views",
        "",
        "| horizon | method | Spearman | top-1 | regret mm |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for horizon, methods in report["methods_by_horizon"].items():
        for method, values in methods.items():
            lines.append(
                f"| {horizon} | {method} | {values['spearman_mean']:.4f} | "
                f"{values['top1_accuracy']:.3f} | "
                f"{values['top1_regret_mean_mm']:.3f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parser().parse_args()
    if int(args.batch_size) <= 0:
        raise ValueError("batch size must be positive")
    device = _cuda0(args.device)
    torch.cuda.set_device(device)
    raw_path = args.raw_dataset.expanduser().resolve()
    index, arrays = open_raw_replay_arrays(raw_path)
    _validate_dataset(index, arrays)
    source = np.load(args.source_cache.expanduser().resolve(), allow_pickle=False)
    history = _history_array(source)
    model, checkpoint = gallery._load_checkpoint(
        args.checkpoint.expanduser().resolve(), device=device
    )
    if not callable(getattr(model, "predict_with_history", None)):
        raise TypeError("world-goal selector requires a history-conditioned model")
    bundles = _wipe_bundles(index, args.split)
    predicted_parts: list[np.ndarray] = []
    wrong_parts: list[np.ndarray] = []
    actual_parts: list[np.ndarray] = []
    current_parts: list[np.ndarray] = []
    try:
        for start in range(0, len(bundles), int(args.batch_size)):
            result = _forecast_batch(
                model,
                index,
                arrays,
                history,
                bundles[start : start + int(args.batch_size)],
                device=device,
            )
            predicted_parts.append(result[0])
            wrong_parts.append(result[1])
            actual_parts.append(result[2])
            current_parts.append(result[3])
    finally:
        del model
        gallery.legacy._clear_runtime_memory(device)
    predicted = np.concatenate(predicted_parts)
    wrong = np.concatenate(wrong_parts)
    actual = np.concatenate(actual_parts)
    current = np.concatenate(current_parts)
    actions = np.asarray(index["future_action"], dtype=np.float32)[bundles]
    kinematic = integrated_cartesian_action_forecast(current, actions)
    goals = _goal_tool_centers(index, bundles)
    methods_by_horizon: dict[str, dict[str, Any]] = {}
    view_methods_by_horizon: dict[str, dict[str, Any]] = {}
    physical = np.asarray(index["physical_bundle_index"], dtype=np.int64)[bundles]
    for horizon_index in range(HORIZON):
        actual_cost = world_goal_distance_cost(
            actual, goals, horizon_index=horizon_index
        )
        current_cost = np.repeat(
            np.linalg.norm(current[:, 2] - goals, axis=-1)[:, None],
            BRANCH_COUNT,
            axis=1,
        )
        method_forecasts = {
            "factual_lewm": predicted,
            "cyclic_wrong_action": wrong,
            "kinematic_action": kinematic,
        }
        predicted_costs = {
            name: world_goal_distance_cost(
                value, goals, horizon_index=horizon_index
            )
            for name, value in method_forecasts.items()
        }
        predicted_costs["current_only_first_slot"] = current_cost
        view_methods = {
            name: _scaled_metrics(value, actual_cost)
            for name, value in predicted_costs.items()
        }
        methods: dict[str, Any] = {}
        for name, value in predicted_costs.items():
            physical_prediction, physical_actual, _ = (
                average_costs_by_physical_snapshot(value, actual_cost, physical)
            )
            methods[name] = _scaled_metrics(physical_prediction, physical_actual)
        methods_by_horizon[f"H{horizon_index + 1}"] = methods
        view_methods_by_horizon[f"H{horizon_index + 1}"] = view_methods
    clocks = sorted(
        set(np.asarray(index["camera_clock_hour"], dtype=np.int64)[bundles].tolist())
    )
    report = _report(
        methods_by_horizon,
        view_methods_by_horizon=view_methods_by_horizon,
        row_count=len(bundles),
        physical_snapshot_count=len(np.unique(physical)),
        clocks=clocks,
    )
    report["checkpoint_protocol"] = str(checkpoint.get("protocol", ""))
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "summary.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
