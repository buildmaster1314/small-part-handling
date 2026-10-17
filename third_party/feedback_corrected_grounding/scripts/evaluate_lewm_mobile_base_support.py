#!/usr/bin/env python3
"""Check that moving-base Pick exposes a non-trivial odometry/FK prediction gap."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.geometry import Camera, project_points  # noqa: E402
from fcpg.lewm_action_branch_a4_collection import make_a4_episode_runtime  # noqa: E402
from fcpg.lewm_mobile_base import (  # noqa: E402
    PlanarBaseTracker,
    apply_planar_base_translation,
    replace_base_displacement_with_odometry,
)


DEFAULT_OUTPUT = ROOT / "outputs/lewm_mobile_base/pick_support_dev.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--max-steps", type=int, default=90)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    return parser.parse_args()


def _base_command(step: int, phase: float) -> np.ndarray:
    heading = phase + 0.028 * float(step)
    return np.asarray(
        [8.0e-4 * np.cos(heading), 6.0e-4 * np.sin(heading)],
        dtype=np.float64,
    )


def _epe(predicted: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    mask = np.asarray(valid, dtype=bool)
    if not bool(mask.any()):
        return float("nan")
    return float(np.linalg.norm(predicted[mask] - target[mask], axis=-1).mean())


def _episode(
    *,
    episode: int,
    seed: int,
    max_steps: int,
    width: int,
    height: int,
) -> dict[str, object]:
    rng = np.random.default_rng(int(seed) + 10_007 * int(episode))
    runtime = make_a4_episode_runtime(
        "pick-place-v3",
        int(episode),
        int(seed) + 100_000 + int(episode),
        width=int(width),
        height=int(height),
        benchmark_seed=int(seed),
        camera_name="corner",
    )
    camera = Camera(
        np.asarray(runtime.camera_intrinsic, dtype=np.float64),
        np.asarray(runtime.camera_extrinsic_cam_from_world, dtype=np.float64),
        int(width),
        int(height),
    )
    tracker = PlanarBaseTracker()
    phase = float(rng.uniform(-np.pi, np.pi))
    longitudinal_slip = float(rng.uniform(0.18, 0.34))
    lateral_slip = float(rng.uniform(-0.12, 0.12))
    onset = int(rng.integers(8, 15))
    duration = int(rng.integers(25, 36))
    hold_q: np.ndarray | None = None
    rows: list[dict[str, object]] = []
    success = False
    observation = np.asarray(runtime.observation, dtype=np.float32)
    try:
        for step in range(int(max_steps)):
            command = _base_command(step, phase)
            slip = np.asarray(
                [
                    longitudinal_slip * command[0] - lateral_slip * command[1],
                    lateral_slip * command[0] + longitudinal_slip * command[1],
                ],
                dtype=np.float64,
            )
            transition = tracker.step(command, slip_xy=slip)
            apply_planar_base_translation(runtime.env, transition.true_delta_xy)
            observation = np.asarray(runtime.env._get_obs(), dtype=np.float32)
            action = np.asarray(runtime.expert.get_action(observation), dtype=np.float32)
            observation, reward, terminated, truncated, info = runtime.env.step(action)
            record = runtime.record_current(observation)
            odometry_xyz = replace_base_displacement_with_odometry(
                record.keypoints_xyz,
                true_base_displacement_xy=transition.true_position_xy,
                odometry_base_displacement_xy=transition.odometry_position_xy,
            )
            q_odometry, valid_odometry = project_points(odometry_xyz, camera)
            common = np.asarray(record.valid_true) & np.asarray(valid_odometry)
            if step == onset - 1:
                hold_q = np.asarray(record.q_true, dtype=np.float64).copy()
            interruption = bool(onset <= step < onset + duration)
            row: dict[str, object] = {
                "step": step,
                "interruption": interruption,
                "reward": float(reward),
                "success": bool(info.get("success", False)),
                "true_base_xy": transition.true_position_xy.tolist(),
                "odometry_base_xy": transition.odometry_position_xy.tolist(),
                "odometry_epe_px": _epe(q_odometry, record.q_true, common),
            }
            if interruption and hold_q is not None:
                row["hold_epe_px"] = _epe(hold_q, record.q_true, common)
                row["true_anchor_motion_px"] = _epe(hold_q, record.q_true, common)
            rows.append(row)
            success = success or bool(info.get("success", False))
            if terminated or truncated:
                break
            if success and step >= max(onset + duration, 40):
                break
        interruption_rows = [row for row in rows if bool(row["interruption"])]
        odometry = np.asarray(
            [float(row["odometry_epe_px"]) for row in interruption_rows],
            dtype=np.float64,
        )
        hold = np.asarray(
            [float(row["hold_epe_px"]) for row in interruption_rows if "hold_epe_px" in row],
            dtype=np.float64,
        )
        return {
            "episode": episode,
            "onset": onset,
            "duration": duration,
            "longitudinal_slip": longitudinal_slip,
            "lateral_slip": lateral_slip,
            "success": success,
            "steps": len(rows),
            "interruption_steps": len(interruption_rows),
            "mean_odometry_epe_px": float(np.nanmean(odometry)),
            "final_odometry_epe_px": float(odometry[-1]),
            "mean_hold_epe_px": float(np.nanmean(hold)),
            "final_true_base_xy": tracker.true_position_xy.tolist(),
            "final_odometry_base_xy": tracker.odometry_position_xy.tolist(),
            "rows": rows,
        }
    finally:
        runtime.env.close()


def main() -> None:
    args = _parser()
    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError("episodes and max_steps must be positive")
    episodes = [
        _episode(
            episode=index,
            seed=int(args.seed),
            max_steps=int(args.max_steps),
            width=int(args.width),
            height=int(args.height),
        )
        for index in range(int(args.episodes))
    ]
    odometry = np.asarray(
        [float(row["mean_odometry_epe_px"]) for row in episodes], dtype=np.float64
    )
    hold = np.asarray(
        [float(row["mean_hold_epe_px"]) for row in episodes], dtype=np.float64
    )
    summary = {
        "protocol": "mobile_base_pick_support_development_v1",
        "task": "pick-place-v3",
        "episodes": int(args.episodes),
        "success_rate": float(np.mean([bool(row["success"]) for row in episodes])),
        "mean_odometry_epe_px": float(odometry.mean()),
        "median_odometry_epe_px": float(np.median(odometry)),
        "mean_hold_epe_px": float(hold.mean()),
        "median_hold_epe_px": float(np.median(hold)),
        "all_episodes_have_interruption_support": bool(
            all(int(row["interruption_steps"]) >= 20 for row in episodes)
        ),
        "episode_results": episodes,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "episode_results"},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
