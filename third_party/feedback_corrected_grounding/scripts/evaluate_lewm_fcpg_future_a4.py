#!/usr/bin/env python3
"""Evaluate FCPG's effect on frozen-LeWM prediction against actual future frames."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a4 import HORIZON  # noqa: E402
from fcpg.lewm_action_prefix_joint_a4 import load_joint_checkpoint  # noqa: E402
from scripts.evaluate_lewm_fcpg_intervention_a4 import (  # noqa: E402
    _paired_bootstrap,
    _summarize,
)
from scripts.train_lewm_fcpg_paired_invariance_a4 import (  # noqa: E402
    DEFAULT_SOURCE,
    DEFAULT_DEV_STEP_CACHE,
    DEFAULT_DEV_STEP_DATA,
    PairedFrames,
    _cuda0,
    _encode_frames,
    _frame_tensors,
    _load_paired_frames,
)


PROTOCOL = "fcpg_frozen_lewm_actual_future_v1"
MODES = ("nominal", "fcpg", "oracle")
METRICS = (
    "latent_mse",
    "anchor_latent_mse",
    "metric_epe_px",
    "metric_h1_epe_px",
    "metric_last_epe_px",
)
RAW_STEPS_PER_TOKEN = 5


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--data", type=Path, default=DEFAULT_DEV_STEP_DATA)
    parser.add_argument("--cache", type=Path, default=DEFAULT_DEV_STEP_CACHE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view", choices=("dynamic", "clean"), default="dynamic")
    parser.add_argument(
        "--fcpg-deadzone-px",
        type=float,
        default=0.0,
        help=(
            "Use nominal grounding when the mean FCPG correction magnitude across "
            "anchors is at or below this fixed pixel threshold."
        ),
    )
    parser.add_argument("--horizons", type=int, default=4)
    parser.add_argument("--post-step-only", action="store_true")
    parser.add_argument("--settle-frames", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=40_641)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _clean_view_frames(frames: PairedFrames, cache_path: Path) -> PairedFrames:
    """Use paired-clean RGB and the corrector's clean-view output as current state."""

    with np.load(cache_path, allow_pickle=False) as cache:
        required = {"q_clean_corrected", "task_name", "episode_id", "frame_id"}
        if not required.issubset(cache.files):
            raise ValueError("paired-view cache lacks clean corrected grounding")
        for name, expected in (
            ("task_name", frames.task_name),
            ("episode_id", frames.episode_id),
            ("frame_id", frames.frame_id),
        ):
            if not np.array_equal(cache[name], expected):
                raise ValueError(f"clean-view cache identity mismatch for {name}")
        q_clean = np.asarray(cache["q_clean_corrected"], dtype=np.float32)
    if q_clean.shape != frames.q_nominal.shape or not np.isfinite(q_clean).all():
        raise ValueError("clean corrected grounding has invalid shape or values")
    return replace(
        frames,
        dynamic_rgb=frames.clean_rgb,
        q_fcpg=q_clean.copy(),
        q_true=frames.q_nominal.copy(),
        valid_true=frames.valid_nominal.copy(),
    )


def _apply_fcpg_deadzone(
    frames: PairedFrames,
    threshold_px: float,
) -> tuple[PairedFrames, np.ndarray, np.ndarray]:
    """Suppress frame-level FCPG corrections whose aggregate magnitude is small."""

    if not np.isfinite(threshold_px) or threshold_px < 0.0:
        raise ValueError("FCPG dead-zone threshold must be finite and non-negative")
    delta = np.asarray(frames.q_fcpg - frames.q_nominal, dtype=np.float32)
    correction_magnitude = np.linalg.norm(delta, axis=-1).mean(axis=1)
    active = correction_magnitude > threshold_px
    gated = np.where(
        active[:, None, None],
        frames.q_fcpg,
        frames.q_nominal,
    ).astype(np.float32, copy=False)
    return replace(frames, q_fcpg=gated), active, correction_magnitude


def _deadzone_summary(
    frames: PairedFrames,
    active: np.ndarray,
    correction_magnitude: np.ndarray,
    *,
    threshold_px: float,
) -> dict[str, Any]:
    per_task = {}
    for task in np.unique(frames.task_name.astype(str)):
        selected = frames.task_name.astype(str) == task
        per_task[str(task)] = {
            "activation_rate": float(active[selected].mean()),
            "mean_correction_magnitude_px": float(
                correction_magnitude[selected].mean()
            ),
        }
    return {
        "threshold_px": float(threshold_px),
        "activation_rate": float(active.mean()),
        "active_frame_count": int(active.sum()),
        "frame_count": int(len(active)),
        "mean_correction_magnitude_px": float(correction_magnitude.mean()),
        "per_task": per_task,
    }


def build_future_windows(
    frames: PairedFrames,
    *,
    horizons: int,
    post_step_only: bool,
    settle_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build causal current/future/action row indices within physical episodes."""

    if not 1 <= horizons <= HORIZON:
        raise ValueError("horizons must lie in [1,HORIZON]")
    if settle_frames < 0:
        raise ValueError("settle frames must be non-negative")
    current_rows: list[int] = []
    future_rows: list[list[int]] = []
    action_rows: list[list[list[int]]] = []
    identities = np.stack(
        [frames.task_name.astype(str), frames.episode_id.astype(str)], axis=1
    )
    for task, episode in np.unique(identities, axis=0):
        episode_mask = (frames.task_name.astype(str) == task) & (
            frames.episode_id.astype(str) == episode
        )
        rows = np.flatnonzero(episode_mask)
        by_frame = {int(frames.frame_id[row]): int(row) for row in rows}
        for frame in sorted(by_frame):
            current = by_frame[frame]
            if post_step_only and frame < int(frames.camera_step_frame[current]) + settle_frames:
                continue
            future_frames = [
                frame + RAW_STEPS_PER_TOKEN * (index + 1)
                for index in range(horizons)
            ]
            action_frames = [
                [
                    frame + RAW_STEPS_PER_TOKEN * index + raw_step
                    for raw_step in range(RAW_STEPS_PER_TOKEN)
                ]
                for index in range(horizons)
            ]
            required = future_frames + [value for chunk in action_frames for value in chunk]
            if any(value not in by_frame for value in required):
                continue
            current_rows.append(current)
            future_rows.append([by_frame[value] for value in future_frames])
            action_rows.append(
                [[by_frame[value] for value in chunk] for chunk in action_frames]
            )
    return (
        np.asarray(current_rows, dtype=np.int64),
        np.asarray(future_rows, dtype=np.int64),
        np.asarray(action_rows, dtype=np.int64),
    )


def _masked_metric_epe(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    distance = torch.linalg.vector_norm(predicted.float() - target.float(), dim=-1)
    weights = valid.float()
    return (distance * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


@torch.inference_mode()
def _evaluate(
    model: torch.nn.Module,
    frames: PairedFrames,
    *,
    horizons: int,
    post_step_only: bool,
    settle_frames: int,
    device: torch.device,
    batch_size: int,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    current_rows, future_rows, action_rows = build_future_windows(
        frames,
        horizons=horizons,
        post_step_only=post_step_only,
        settle_frames=settle_frames,
    )
    if not len(current_rows):
        raise RuntimeError("future-window filter produced no examples")
    model.eval()
    collected = {mode: {metric: [] for metric in METRICS} for mode in MODES}
    for start in range(0, len(current_rows), batch_size):
        stop = min(start + batch_size, len(current_rows))
        current = current_rows[start:stop]
        future = future_rows[start:stop]
        action = action_rows[start:stop]
        task_index = torch.from_numpy(
            np.ascontiguousarray(frames.task_index[current])
        ).to(device)

        flat_future = future.reshape(-1)
        future_tensors = _frame_tensors(
            frames,
            flat_future,
            image_key="dynamic",
            grounding="oracle",
            device=device,
        )
        target_state = _encode_frames(model, future_tensors).reshape(
            len(current), horizons, 4, 192
        )
        target_q = torch.from_numpy(
            np.ascontiguousarray(frames.q_true[future])
        ).to(device)
        target_valid = torch.from_numpy(
            np.ascontiguousarray(frames.valid_true[future])
        ).to(device)

        action_prefix = torch.zeros(
            len(current), HORIZON, RAW_STEPS_PER_TOKEN * 4, device=device
        )
        selected_actions = torch.from_numpy(
            np.ascontiguousarray(frames.raw_action[action])
        ).to(device)
        action_prefix[:, :horizons] = selected_actions.reshape(
            len(current), horizons, RAW_STEPS_PER_TOKEN * 4
        )

        for mode in MODES:
            tensors = _frame_tensors(
                frames,
                current,
                image_key="dynamic",
                grounding=mode,
                device=device,
            )
            state = _encode_frames(model, tensors)
            full_forecast = model.dynamics(
                state,
                action_prefix.to(dtype=state.dtype),
                task_index=task_index,
            )
            full_metric = model.metric_readout(
                full_forecast,
                state,
                tensors["q_grounded"].float(),
                task_index,
                image_size=frames.image_size,
            )
            forecast = full_forecast[:, :horizons]
            metric = full_metric[:, :horizons]
            squared = (forecast.float() - target_state.float()).square()
            metric_by_horizon = _masked_metric_epe(metric, target_q, target_valid)
            batch_metrics = {
                "latent_mse": squared.mean(dim=(1, 2, 3)),
                "anchor_latent_mse": squared[:, :, 1:].mean(dim=(1, 2, 3)),
                "metric_epe_px": metric_by_horizon.mean(dim=1),
                "metric_h1_epe_px": metric_by_horizon[:, 0],
                "metric_last_epe_px": metric_by_horizon[:, -1],
            }
            for metric_name, values in batch_metrics.items():
                collected[mode][metric_name].append(values.float().cpu().numpy())

    arrays = {
        mode: {
            metric: np.concatenate(parts).astype(np.float64, copy=False)
            for metric, parts in mode_values.items()
        }
        for mode, mode_values in collected.items()
    }
    tasks = frames.task_name[current_rows]
    episodes = frames.episode_id[current_rows]
    summaries = {
        mode: {
            metric: _summarize(values, tasks, episodes)
            for metric, values in mode_values.items()
        }
        for mode, mode_values in arrays.items()
    }
    paired = {
        metric: _paired_bootstrap(
            arrays["nominal"][metric],
            arrays["fcpg"][metric],
            tasks,
            episodes,
            draws=bootstrap_draws,
            seed=seed + index,
        )
        for index, metric in enumerate(METRICS)
    }
    return {
        "window_count": int(len(current_rows)),
        "horizons": horizons,
        "raw_steps_per_horizon": RAW_STEPS_PER_TOKEN,
        "post_step_only": post_step_only,
        "settle_frames": settle_frames,
        "modes": summaries,
        "paired_improvement": paired,
    }


def main() -> int:
    args = _parser().parse_args()
    if args.view == "clean" and args.post_step_only:
        raise ValueError("clean-view evaluation does not use --post-step-only")
    device = _cuda0(str(args.device))
    torch.cuda.set_device(device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint must contain a mapping")
    model = load_joint_checkpoint(payload).to(device).eval()
    cache_path = args.cache.expanduser().resolve()
    frames = _load_paired_frames(args.data.expanduser().resolve(), cache_path)
    if args.view == "clean":
        frames = _clean_view_frames(frames, cache_path)
    frames, deadzone_active, correction_magnitude = _apply_fcpg_deadzone(
        frames,
        float(args.fcpg_deadzone_px),
    )
    evaluation = _evaluate(
        model,
        frames,
        horizons=int(args.horizons),
        post_step_only=bool(args.post_step_only),
        settle_frames=int(args.settle_frames),
        device=device,
        batch_size=int(args.batch_size),
        bootstrap_draws=int(args.bootstrap_draws),
        seed=int(args.seed),
    )
    result = {
        "protocol": PROTOCOL,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(payload.get("step", -1)),
        "data": str(args.data.expanduser().resolve()),
        "cache": str(cache_path),
        "view": str(args.view),
        "target": (
            "actual_future_dynamic_rgb_encoded_at_q_true"
            if args.view == "dynamic"
            else "paired_clean_future_rgb_encoded_at_q_nominal"
        ),
        "q_true_deployment_input_used": False,
        "fcpg_deadzone": _deadzone_summary(
            frames,
            deadzone_active,
            correction_magnitude,
            threshold_px=float(args.fcpg_deadzone_px),
        ),
        "evaluation": evaluation,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("output already exists")
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
