#!/usr/bin/env python3
"""Evaluate whether FCPG moves a frozen LeWM toward same-view oracle grounding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_prefix_joint_a4 import load_joint_checkpoint  # noqa: E402
from fcpg.lewm_fcpg_paired_invariance import repeated_action_prefix  # noqa: E402
from scripts.train_lewm_fcpg_paired_invariance_a4 import (  # noqa: E402
    DEFAULT_SOURCE,
    PairedFrames,
    _cuda0,
    _encode_frames,
    _frame_tensors,
    _load_paired_frames,
    _parse_eval_pairs,
)


PROTOCOL = "fcpg_same_view_lewm_oracle_intervention_v1"
MODES = ("nominal", "fcpg")
METRICS = (
    "input_epe_px",
    "state_gap",
    "state_anchor_gap",
    "forecast_gap",
    "forecast_anchor_gap",
    "metric_gap_px",
    "metric_h1_gap_px",
    "metric_h8_gap_px",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--eval-pair",
        action="append",
        default=[],
        help="Named evaluation pair as name=data.npz,cache.npz.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=40_631)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _masked_input_epe(
    grounded: torch.Tensor,
    oracle: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    distance = torch.linalg.vector_norm(grounded.float() - oracle.float(), dim=-1)
    weights = valid.float()
    return (distance * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _state_gap(candidate: torch.Tensor, oracle: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(candidate.float() - oracle.float(), dim=-1)


def _episode_values(
    values: np.ndarray,
    tasks: np.ndarray,
    episodes: np.ndarray,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for task in np.unique(tasks):
        selected_task = tasks == task
        task_values = []
        for episode in np.unique(episodes[selected_task]):
            selected = selected_task & (episodes == episode)
            task_values.append(float(np.mean(values[selected])))
        result[str(task)] = np.asarray(task_values, dtype=np.float64)
    return result


def _summarize(
    values: np.ndarray,
    tasks: np.ndarray,
    episodes: np.ndarray,
) -> dict[str, Any]:
    grouped = _episode_values(values, tasks, episodes)
    per_task = {task: float(task_values.mean()) for task, task_values in grouped.items()}
    return {
        "task_macro": float(np.mean(list(per_task.values()))),
        "per_task": per_task,
        "episode_count": int(sum(len(item) for item in grouped.values())),
    }


def _paired_bootstrap(
    nominal: np.ndarray,
    fcpg: np.ndarray,
    tasks: np.ndarray,
    episodes: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    grouped = _episode_values(nominal - fcpg, tasks, episodes)
    per_task = {task: float(values.mean()) for task, values in grouped.items()}
    rng = np.random.default_rng(seed)
    bootstrap_tasks = []
    for values in grouped.values():
        indices = rng.integers(0, len(values), size=(draws, len(values)))
        bootstrap_tasks.append(values[indices].mean(axis=1))
    samples = np.stack(bootstrap_tasks, axis=1).mean(axis=1)
    point = float(np.mean(list(per_task.values())))
    nominal_summary = _summarize(nominal, tasks, episodes)["task_macro"]
    return {
        "nominal_minus_fcpg": point,
        "relative_improvement": point / max(float(nominal_summary), 1e-12),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "positive_task_count": int(sum(value > 0.0 for value in per_task.values())),
        "task_count": len(per_task),
        "per_task": per_task,
    }


@torch.inference_mode()
def _evaluate(
    model: torch.nn.Module,
    frames: PairedFrames,
    *,
    device: torch.device,
    batch_size: int,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    collected = {mode: {metric: [] for metric in METRICS} for mode in MODES}
    for start in range(0, frames.count, batch_size):
        rows = np.arange(start, min(start + batch_size, frames.count))
        task_index = torch.from_numpy(
            np.ascontiguousarray(frames.task_index[rows])
        ).to(device)
        action = torch.from_numpy(
            np.ascontiguousarray(frames.raw_action[rows])
        ).to(device)
        prefix = repeated_action_prefix(action)

        oracle_tensors = _frame_tensors(
            frames,
            rows,
            image_key="dynamic",
            grounding="oracle",
            device=device,
        )
        oracle_state = _encode_frames(model, oracle_tensors)
        oracle_forecast = model.dynamics(
            oracle_state,
            prefix.to(dtype=oracle_state.dtype),
            task_index=task_index,
        )
        oracle_metric = model.metric_readout(
            oracle_forecast,
            oracle_state,
            oracle_tensors["q_grounded"].float(),
            task_index,
            image_size=frames.image_size,
        )

        for mode in MODES:
            tensors = _frame_tensors(
                frames,
                rows,
                image_key="dynamic",
                grounding=mode,
                device=device,
            )
            state = _encode_frames(model, tensors)
            forecast = model.dynamics(
                state,
                prefix.to(dtype=state.dtype),
                task_index=task_index,
            )
            metric = model.metric_readout(
                forecast,
                state,
                tensors["q_grounded"].float(),
                task_index,
                image_size=frames.image_size,
            )
            state_distance = _state_gap(state, oracle_state)
            forecast_distance = _state_gap(forecast, oracle_forecast)
            metric_distance = torch.linalg.vector_norm(
                metric.float() - oracle_metric.float(), dim=-1
            )
            valid = tensors["valid"] & oracle_tensors["valid"]
            batch_metrics = {
                "input_epe_px": _masked_input_epe(
                    tensors["q_grounded"], oracle_tensors["q_grounded"], valid
                ),
                "state_gap": state_distance.mean(dim=1),
                "state_anchor_gap": state_distance[:, 1:].mean(dim=1),
                "forecast_gap": forecast_distance.mean(dim=(1, 2)),
                "forecast_anchor_gap": forecast_distance[:, :, 1:].mean(dim=(1, 2)),
                "metric_gap_px": metric_distance.mean(dim=(1, 2)),
                "metric_h1_gap_px": metric_distance[:, 0].mean(dim=1),
                "metric_h8_gap_px": metric_distance[:, -1].mean(dim=1),
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
    summaries = {
        mode: {
            metric: _summarize(values, frames.task_name, frames.episode_id)
            for metric, values in mode_values.items()
        }
        for mode, mode_values in arrays.items()
    }
    paired = {
        metric: _paired_bootstrap(
            arrays["nominal"][metric],
            arrays["fcpg"][metric],
            frames.task_name,
            frames.episode_id,
            draws=bootstrap_draws,
            seed=seed + index,
        )
        for index, metric in enumerate(METRICS)
    }
    return {
        "frame_count": frames.count,
        "modes": summaries,
        "paired_improvement": paired,
    }


def main() -> int:
    args = _parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    device = _cuda0(str(args.device))
    torch.cuda.set_device(device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint must contain a mapping")
    model = load_joint_checkpoint(payload).to(device).eval()
    eval_paths = _parse_eval_pairs(list(args.eval_pair))
    evaluations = {}
    for index, (name, (data_path, cache_path)) in enumerate(eval_paths.items()):
        frames = _load_paired_frames(data_path.resolve(), cache_path.resolve())
        evaluations[name] = _evaluate(
            model,
            frames,
            device=device,
            batch_size=int(args.batch_size),
            bootstrap_draws=int(args.bootstrap_draws),
            seed=int(args.seed) + 100 * index,
        )
    result = {
        "protocol": PROTOCOL,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(payload.get("step", -1)),
        "bootstrap_unit": "physical_episode_within_task_then_task_macro",
        "oracle": "same_dynamic_rgb_with_q_true_grounding_only",
        "future_information_used": False,
        "evaluations": evaluations,
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
