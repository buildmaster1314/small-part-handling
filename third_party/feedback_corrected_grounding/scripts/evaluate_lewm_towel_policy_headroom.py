#!/usr/bin/env python3
"""Evaluate policy-centered Towel candidates with real branches and frozen LeWM."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import html
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch


os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.custom_tasks.towel_wipe import (  # noqa: E402
    SawyerTowelWipeEnvV0,
    TOWEL_WIPE_TASK_NAME,
)
from fcpg.lewm_action_branch_a4_collection import (  # noqa: E402
    make_a4_episode_runtime,
)
from fcpg.lewm_counterfactual_runtime import (  # noqa: E402
    capture_branch_snapshot,
    restore_branch_snapshot,
)
from fcpg.lewm_goal_action_selector import (  # noqa: E402
    integrated_cartesian_action_forecast,
    selector_metrics,
)
from fcpg.lewm_orbit_camera import OrbitCameraView  # noqa: E402
from fcpg.lewm_towel_receding_selector import (  # noqa: E402
    POLICY_CANDIDATE_NAMES,
    PRIMARY_HORIZON_INDEX,
    build_policy_centered_towel_candidates,
    build_policy_repeated_prefix,
    causal_history_chunks,
    center_calibrated_world_forecast,
    tool_xy_goal_reach_cost,
)
from fcpg.transformer_grounding import load_fcpg_vit_adapter  # noqa: E402
from scripts import evaluate_lewm_towel_receding_selector as online  # noqa: E402
from scripts import render_lewm_action_prefix_joint_a4_accuracy_gallery as gallery  # noqa: E402


DEFAULT_CHECKPOINT = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "camera_action_metric3d_world_history_seed40583_10k/"
    "lewm_history_conditioned_latest.pt"
)
DEFAULT_FCPG_CHECKPOINT = ROOT / (
    "outputs/transformer_residual/a4_multiview_projection_seed40601_2k/"
    "fcpg_a4_multiview_best.pt"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "towel_policy_centered_headroom_rot5_clock03"
)
BRANCH_COLORS = (
    (80, 220, 100),
    (235, 90, 80),
    (200, 200, 200),
    (255, 195, 70),
    (80, 180, 255),
    (190, 130, 255),
    (245, 100, 210),
    (80, 220, 220),
)
RAW_CHUNK = 5


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--fcpg-checkpoint", type=Path, default=DEFAULT_FCPG_CHECKPOINT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--task-instance-start", type=int, default=36)
    parser.add_argument("--environment-seed-base", type=int, default=162_000)
    parser.add_argument("--benchmark-seed", type=int, default=30_103)
    parser.add_argument("--clock-hour", type=int, default=3)
    parser.add_argument("--camera-rotation-degrees", type=float, default=5.0)
    parser.add_argument(
        "--camera-rotation-axis-world",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 1.0),
    )
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--max-raw-steps", type=int, default=500)
    parser.add_argument("--snapshots-per-half-cycle", type=int, default=1)
    parser.add_argument("--snapshot-stride", type=int, default=2)
    parser.add_argument("--minimum-goal-error-m", type=float, default=0.025)
    parser.add_argument("--maximum-goal-error-m", type=float, default=0.14)
    return parser


def _actual_branch_rollouts(
    runtime: Any,
    *,
    snapshot: Any,
    observation: np.ndarray,
    candidates: np.ndarray,
    goal_tool_world: np.ndarray,
    horizon_index: int = PRIMARY_HORIZON_INDEX,
) -> dict[str, np.ndarray]:
    """Execute every candidate from one state and restore the source state."""

    values = np.asarray(candidates, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (8, 20):
        raise ValueError("candidates must have shape [C,8,20]")
    horizon = int(horizon_index)
    if not 0 <= horizon < values.shape[1]:
        raise ValueError("horizon_index is outside the candidate horizon")
    goal = np.asarray(goal_tool_world, dtype=np.float32)
    if goal.shape != (3,) or not np.isfinite(goal).all():
        raise ValueError("goal_tool_world must be one finite 3-vector")

    model_steps = horizon + 1
    world = np.empty((len(values), model_steps, 3, 3), dtype=np.float32)
    pixels = np.empty((len(values), model_steps, 3, 2), dtype=np.float32)
    contact = np.zeros((len(values), model_steps), dtype=bool)
    grasp = np.zeros((len(values), model_steps), dtype=bool)
    half_cycle_delta = np.zeros(len(values), dtype=np.int64)
    start_half_cycles = int(runtime.env.wipe_half_cycles)
    try:
        for candidate_index, candidate in enumerate(values):
            restore_branch_snapshot(runtime.env, snapshot)
            branch_observation = np.asarray(observation, dtype=np.float32).copy()
            raw = candidate.reshape(8, RAW_CHUNK, 4)
            for model_step in range(model_steps):
                for raw_action in raw[model_step]:
                    (
                        branch_observation,
                        _reward,
                        terminated,
                        truncated,
                        _info,
                    ) = runtime.env.step(raw_action)
                    branch_observation = np.asarray(
                        branch_observation, dtype=np.float32
                    )
                    if terminated or truncated:
                        raise RuntimeError(
                            "policy-centered candidate ended before the H6 endpoint"
                        )
                record = runtime.record_current(branch_observation)
                world[candidate_index, model_step] = np.asarray(
                    record.keypoints_xyz, dtype=np.float32
                )
                pixels[candidate_index, model_step] = np.asarray(
                    record.q_true, dtype=np.float32
                )
                contact[candidate_index, model_step] = bool(
                    runtime.env.towel_surface_engaged
                )
                grasp[candidate_index, model_step] = bool(runtime.env.towel_grasped)
            half_cycle_delta[candidate_index] = (
                int(runtime.env.wipe_half_cycles) - start_half_cycles
            )
    finally:
        restore_branch_snapshot(runtime.env, snapshot)

    return {
        "world": world,
        "pixels": pixels,
        "cost": tool_xy_goal_reach_cost(world, goal, horizon_index=horizon),
        "contact_fraction": contact.mean(axis=1, dtype=np.float64),
        "grasp_fraction": grasp.mean(axis=1, dtype=np.float64),
        "half_cycle_delta": half_cycle_delta,
    }


def _method_row(cost: np.ndarray, actual_cost: np.ndarray) -> dict[str, Any]:
    predicted = np.asarray(cost, dtype=np.float64)
    actual = np.asarray(actual_cost, dtype=np.float64)
    selected = int(np.argmin(predicted))
    oracle = int(np.argmin(actual))
    spread = float(np.ptp(actual))
    # Equal minima can differ by a few ulps after JSON/float conversions.
    regret = max(0.0, float(actual[selected] - actual[oracle]))
    return {
        "selected_index": selected,
        "selected_name": POLICY_CANDIDATE_NAMES[selected],
        "oracle_index": oracle,
        "oracle_name": POLICY_CANDIDATE_NAMES[oracle],
        "top1": selected == oracle,
        "actual_selected_cost_m": float(actual[selected]),
        "actual_oracle_cost_m": float(actual[oracle]),
        "regret_m": regret,
        "normalized_regret": regret / max(spread, 1.0e-8),
    }


def _draw_snapshot(
    record: Any,
    *,
    row: Mapping[str, Any],
    output_path: Path,
) -> None:
    image = online._rgb_image(record)
    draw = ImageDraw.Draw(image)
    branch_pixels = np.asarray(row["actual_branch_pixels"], dtype=np.float32)
    for candidate_index, color in enumerate(BRANCH_COLORS):
        points = [
            tuple(float(value) for value in point)
            for point in branch_pixels[candidate_index, :, 2]
            if np.isfinite(point).all()
        ]
        if points:
            start = tuple(float(value) for value in np.asarray(record.q_true)[2])
            draw.line([start, *points], fill=color, width=2)
            x, y = points[-1]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    lines = (
        f"ep {row['episode_index']} step {row['raw_step']} "
        f"cycle {row['half_cycles']} {row['phase']}",
        f"goal {100.0 * float(row['goal_error_m']):.1f}cm | "
        f"oracle {row['methods']['oracle']['selected_name']}",
        f"geometry {row['methods']['geometry']['selected_name']} "
        f"r={100.0 * row['methods']['geometry']['regret_m']:.2f}cm",
        f"LeWM nominal {row['methods']['lewm_nominal']['selected_name']} "
        f"r={100.0 * row['methods']['lewm_nominal']['regret_m']:.2f}cm",
        f"LeWM FCPG {row['methods']['lewm_fcpg']['selected_name']} "
        f"r={100.0 * row['methods']['lewm_fcpg']['regret_m']:.2f}cm",
    )
    draw.rectangle((0, 0, image.width, 84), fill=(0, 0, 0))
    for index, line in enumerate(lines):
        draw.text((5, 4 + 16 * index), line, fill=(255, 255, 255))
    image.save(output_path, quality=90)


def _score_snapshot(
    model: Any,
    corrector: Any,
    runtime: Any,
    *,
    record: Any,
    observation: np.ndarray,
    executed_actions: Sequence[np.ndarray],
    episode_index: int,
    raw_step: int,
    phase: str,
    device: torch.device,
    width: int,
    height: int,
    frame_dir: Path,
) -> dict[str, Any]:
    expert_action = online._bounded_expert_action(runtime, observation)
    prefix = build_policy_repeated_prefix(
        expert_action,
        action_low=runtime.env.action_space.low,
        action_high=runtime.env.action_space.high,
    )
    candidates = build_policy_centered_towel_candidates(
        prefix,
        action_low=runtime.env.action_space.low,
        action_high=runtime.env.action_space.high,
    )
    goal_tool = np.asarray(runtime.env._target_pos, dtype=np.float32) + (
        SawyerTowelWipeEnvV0.GRIP_OFFSET.astype(np.float32)
    )
    snapshot = capture_branch_snapshot(runtime.env, include_process_rng=True)
    actual = _actual_branch_rollouts(
        runtime,
        snapshot=snapshot,
        observation=observation,
        candidates=candidates,
        goal_tool_world=goal_tool,
    )

    current_xyz = np.asarray(record.keypoints_xyz, dtype=np.float32)
    kinematic_world = integrated_cartesian_action_forecast(
        current_xyz[None], candidates[None]
    )[0]
    geometry_cost = tool_xy_goal_reach_cost(kinematic_world, goal_tool)
    history = causal_history_chunks(executed_actions)
    predicted: dict[str, np.ndarray] = {}
    grounding_epe: dict[str, float] = {}
    for mode in ("nominal", "fcpg"):
        grounded, valid, epe, _innovation = online._current_grounding(
            record,
            mode=mode,
            corrector=corrector,
            device=device,
            width=width,
            height=height,
        )
        _cost, world, _latent = online._lewm_costs(
            model,
            record,
            candidates,
            history,
            goal_tool,
            q_grounded=grounded,
            valid_grounded=valid,
            camera_intrinsic=runtime.camera_intrinsic,
            camera_extrinsic=runtime.camera_extrinsic_cam_from_world,
            device=device,
        )
        predicted[mode] = world
        grounding_epe[mode] = float(epe)

    calibrated = {
        mode: center_calibrated_world_forecast(
            world,
            current_xyz,
            candidates,
        )
        for mode, world in predicted.items()
    }
    costs = {
        "policy": np.arange(len(candidates), dtype=np.float64),
        "geometry": geometry_cost,
        "lewm_nominal": tool_xy_goal_reach_cost(predicted["nominal"], goal_tool),
        "lewm_fcpg": tool_xy_goal_reach_cost(predicted["fcpg"], goal_tool),
        "lewm_nominal_calibrated": tool_xy_goal_reach_cost(
            calibrated["nominal"], goal_tool
        ),
        "lewm_fcpg_calibrated": tool_xy_goal_reach_cost(
            calibrated["fcpg"], goal_tool
        ),
        "oracle": np.asarray(actual["cost"], dtype=np.float64),
    }
    # The policy method deterministically selects candidate zero.
    costs["policy"] = np.arange(len(candidates), dtype=np.float64)
    methods = {
        name: _method_row(cost, actual["cost"]) for name, cost in costs.items()
    }
    row: dict[str, Any] = {
        "episode_index": int(episode_index),
        "raw_step": int(raw_step),
        "phase": str(phase),
        "half_cycles": int(runtime.env.wipe_half_cycles),
        "goal_error_m": float(np.linalg.norm(current_xyz[2, :2] - goal_tool[:2])),
        "candidate_names": list(POLICY_CANDIDATE_NAMES),
        "expert_action": expert_action.astype(float).tolist(),
        "actual_cost_m": np.asarray(actual["cost"]).astype(float).tolist(),
        "geometry_cost_m": np.asarray(geometry_cost).astype(float).tolist(),
        "lewm_nominal_cost_m": np.asarray(costs["lewm_nominal"]).astype(float).tolist(),
        "lewm_fcpg_cost_m": np.asarray(costs["lewm_fcpg"]).astype(float).tolist(),
        "lewm_nominal_calibrated_cost_m": np.asarray(
            costs["lewm_nominal_calibrated"]
        ).astype(float).tolist(),
        "lewm_fcpg_calibrated_cost_m": np.asarray(
            costs["lewm_fcpg_calibrated"]
        ).astype(float).tolist(),
        "contact_fraction": np.asarray(actual["contact_fraction"]).astype(float).tolist(),
        "grasp_fraction": np.asarray(actual["grasp_fraction"]).astype(float).tolist(),
        "half_cycle_delta": np.asarray(actual["half_cycle_delta"]).astype(int).tolist(),
        "grounding_epe_px": grounding_epe,
        "methods": methods,
        "actual_branch_pixels": np.asarray(actual["pixels"]).astype(float).tolist(),
    }
    frame_name = f"ep{episode_index:02d}_step{raw_step:03d}.jpg"
    _draw_snapshot(record, row=row, output_path=frame_dir / frame_name)
    row["frame_path"] = f"frames/{frame_name}"
    return row


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("headroom evaluation produced no snapshots")
    actual = np.stack([np.asarray(row["actual_cost_m"]) for row in rows])
    method_keys = (
        "policy",
        "geometry",
        "lewm_nominal",
        "lewm_fcpg",
        "lewm_nominal_calibrated",
        "lewm_fcpg_calibrated",
        "oracle",
    )
    predicted = {
        "policy": np.broadcast_to(
            np.arange(len(POLICY_CANDIDATE_NAMES), dtype=np.float64), actual.shape
        ),
        "geometry": np.stack([np.asarray(row["geometry_cost_m"]) for row in rows]),
        "lewm_nominal": np.stack(
            [np.asarray(row["lewm_nominal_cost_m"]) for row in rows]
        ),
        "lewm_fcpg": np.stack(
            [np.asarray(row["lewm_fcpg_cost_m"]) for row in rows]
        ),
        "lewm_nominal_calibrated": np.stack(
            [np.asarray(row["lewm_nominal_calibrated_cost_m"]) for row in rows]
        ),
        "lewm_fcpg_calibrated": np.stack(
            [np.asarray(row["lewm_fcpg_calibrated_cost_m"]) for row in rows]
        ),
        "oracle": actual.copy(),
    }
    metrics = {
        name: selector_metrics(predicted[name], actual, expert_index=0)
        for name in method_keys
    }
    oracle = np.min(actual, axis=1)
    policy = actual[:, 0]
    return {
        "snapshots": len(rows),
        "episodes": len({int(row["episode_index"]) for row in rows}),
        "informative_snapshot_fraction_5mm": float(np.mean(np.ptp(actual, axis=1) >= 0.005)),
        "policy_oracle_headroom_m": float(np.mean(policy - oracle)),
        "policy_oracle_headroom_relative": float(
            np.mean((policy - oracle) / np.maximum(policy, 1.0e-8))
        ),
        "oracle_choice_counts": dict(
            sorted(
                Counter(
                    POLICY_CANDIDATE_NAMES[int(index)]
                    for index in np.argmin(actual, axis=1)
                ).items()
            )
        ),
        "metrics": metrics,
    }


def _write_html(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    cards: list[str] = []
    for row in rows:
        methods = row["methods"]
        costs = np.asarray(row["actual_cost_m"], dtype=np.float64)
        table_rows = []
        for index, name in enumerate(POLICY_CANDIDATE_NAMES):
            table_rows.append(
                "<tr>"
                f"<td>{html.escape(name)}</td>"
                f"<td>{100.0 * costs[index]:.2f}</td>"
                f"<td>{100.0 * float(row['geometry_cost_m'][index]):.2f}</td>"
                f"<td>{100.0 * float(row['lewm_nominal_cost_m'][index]):.2f}</td>"
                f"<td>{100.0 * float(row['lewm_fcpg_cost_m'][index]):.2f}</td>"
                "</tr>"
            )
        cards.append(
            "<section>"
            f"<h2>Episode {row['episode_index']} | step {row['raw_step']} | "
            f"cycle {row['half_cycles']}</h2>"
            f"<img src=\"{html.escape(str(row['frame_path']))}\" alt=\"snapshot\">"
            "<p>Selected: "
            f"oracle <b>{methods['oracle']['selected_name']}</b>; "
            f"geometry <b>{methods['geometry']['selected_name']}</b>; "
            f"LeWM nominal <b>{methods['lewm_nominal']['selected_name']}</b>; "
            f"LeWM FCPG <b>{methods['lewm_fcpg']['selected_name']}</b>.</p>"
            "<table><thead><tr><th>candidate</th><th>actual cm</th>"
            "<th>geometry cm</th><th>LeWM nominal cm</th>"
            "<th>LeWM FCPG cm</th></tr></thead><tbody>"
            + "".join(table_rows)
            + "</tbody></table></section>"
        )
    document = """<!doctype html><html><head><meta charset="utf-8">
<title>Towel policy-centered headroom</title><style>
body{font:14px sans-serif;background:#171717;color:#eee;margin:20px}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}
section{background:#252525;padding:10px;border-radius:6px}img{width:100%}
table{border-collapse:collapse;width:100%}th,td{padding:4px;border:1px solid #555;text-align:right}
th:first-child,td:first-child{text-align:left}h2{font-size:16px}
</style></head><body><h1>Policy-centered Towel headroom</h1>
<p>Candidates are generated only from the current policy action. The task goal is
used for scoring, never for proposal generation. Colored trails are real H6
same-snapshot branch outcomes.</p><main>""" + "".join(cards) + "</main></body></html>"
    (output / "index.html").write_text(document, encoding="utf-8")


def _run_episode(
    model: Any,
    corrector: Any,
    *,
    episode_index: int,
    task_instance: int,
    environment_seed: int,
    args: argparse.Namespace,
    device: torch.device,
    orbit: OrbitCameraView,
    frame_dir: Path,
) -> list[dict[str, Any]]:
    runtime = make_a4_episode_runtime(
        TOWEL_WIPE_TASK_NAME,
        task_instance,
        environment_seed,
        width=int(args.width),
        height=int(args.height),
        benchmark_seed=int(args.benchmark_seed),
        camera_name="towel_view",
        orbit_camera_view=orbit,
    )
    online._apply_camera_rotation(
        runtime,
        axis_world=np.asarray(args.camera_rotation_axis_world, dtype=np.float64),
        rotation_degrees=float(args.camera_rotation_degrees),
    )
    observation = np.asarray(runtime.observation, dtype=np.float32)
    executed: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    per_half_cycle: Counter[int] = Counter()
    wipe_decision = 0
    raw_step = 0
    success_streak = 0
    try:
        while raw_step < int(args.max_raw_steps):
            phase = online._phase(runtime.expert, observation)
            if phase in online.WIPE_PHASES:
                record = runtime.record_current(observation)
                goal_tool = np.asarray(runtime.env._target_pos, dtype=np.float32) + (
                    SawyerTowelWipeEnvV0.GRIP_OFFSET.astype(np.float32)
                )
                error = float(
                    np.linalg.norm(
                        np.asarray(record.keypoints_xyz[2, :2]) - goal_tool[:2]
                    )
                )
                half_cycle = int(runtime.env.wipe_half_cycles)
                eligible = (
                    wipe_decision % int(args.snapshot_stride) == 0
                    and per_half_cycle[half_cycle]
                    < int(args.snapshots_per_half_cycle)
                    and float(args.minimum_goal_error_m)
                    <= error
                    <= float(args.maximum_goal_error_m)
                )
                if eligible:
                    row = _score_snapshot(
                        model,
                        corrector,
                        runtime,
                        record=record,
                        observation=observation,
                        executed_actions=executed,
                        episode_index=episode_index,
                        raw_step=raw_step,
                        phase=phase,
                        device=device,
                        width=int(args.width),
                        height=int(args.height),
                        frame_dir=frame_dir,
                    )
                    rows.append(row)
                    per_half_cycle[half_cycle] += 1
                    print(
                        json.dumps(
                            {
                                "episode": episode_index,
                                "step": raw_step,
                                "cycle": half_cycle,
                                "oracle": row["methods"]["oracle"]["selected_name"],
                                "geometry": row["methods"]["geometry"]["selected_name"],
                                "lewm_fcpg": row["methods"]["lewm_fcpg"]["selected_name"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                wipe_decision += 1

            terminated = False
            truncated = False
            for _ in range(RAW_CHUNK):
                action = online._bounded_expert_action(runtime, observation)
                observation, _reward, terminated, truncated, info = runtime.env.step(
                    action
                )
                observation = np.asarray(observation, dtype=np.float32)
                executed.append(np.asarray(action, dtype=np.float32).copy())
                raw_step += 1
                if online._success(info, runtime.env):
                    success_streak += 1
                else:
                    success_streak = 0
                if success_streak >= 3 or terminated or truncated:
                    break
            if success_streak >= 3 or terminated or truncated:
                break
    finally:
        runtime.env.close()
    return rows


def main() -> int:
    args = _parser().parse_args()
    if int(args.episodes) <= 0:
        raise ValueError("episodes must be positive")
    if int(args.snapshots_per_half_cycle) <= 0 or int(args.snapshot_stride) <= 0:
        raise ValueError("snapshot counts and stride must be positive")
    if not (
        0.0 <= float(args.minimum_goal_error_m) < float(args.maximum_goal_error_m)
    ):
        raise ValueError("goal-error bounds must be ordered and non-negative")
    device = online._cuda0(args.device)
    torch.cuda.set_device(device)
    output = args.output_dir.expanduser().resolve()
    frame_dir = output / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    orbit = OrbitCameraView(
        clock_hour=int(args.clock_hour),
        elevation_degrees=35.0,
        radius_m=1.1,
        base_camera_name="towel_view",
    )
    model, checkpoint = gallery._load_checkpoint(
        args.checkpoint.expanduser().resolve(), device=device
    )
    corrector, _ = load_fcpg_vit_adapter(
        args.fcpg_checkpoint.expanduser().resolve(), device=device
    )
    rows: list[dict[str, Any]] = []
    try:
        for episode in range(int(args.episodes)):
            task_instance = int(args.task_instance_start) + episode
            if not 0 <= task_instance < 48:
                raise ValueError("task instances must remain in 0..47")
            rows.extend(
                _run_episode(
                    model,
                    corrector,
                    episode_index=episode,
                    task_instance=task_instance,
                    environment_seed=int(args.environment_seed_base) + episode,
                    args=args,
                    device=device,
                    orbit=orbit,
                    frame_dir=frame_dir,
                )
            )
    finally:
        del model
        del corrector
        gallery.legacy._clear_runtime_memory(device)
    report = {
        "schema_version": 1,
        "protocol": "towel_policy_centered_headroom_v1",
        "status": "development_zero_training_gate",
        "candidate_generation": "current_policy_action_only_goal_forbidden",
        "scoring_goal": "minimum_current_endpoint_world_xy_cost_over_H1_to_H6",
        "primary_horizon": f"H{PRIMARY_HORIZON_INDEX + 1}",
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "camera_clock_hour": int(args.clock_hour),
        "camera_rotation_degrees": float(args.camera_rotation_degrees),
        "summary": _aggregate(rows),
        "snapshots": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_html(output, rows)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
