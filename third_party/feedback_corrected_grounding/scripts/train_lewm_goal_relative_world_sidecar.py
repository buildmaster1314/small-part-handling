#!/usr/bin/env python3
"""Train and evaluate a small goal-relative sidecar on frozen LeWM outputs.

The parent seven-task LeWM is already evaluated and remains frozen.  This
script trains only the endpoint residual on moving-goal Reach/Push windows.
Episode identities, rather than individual overlapping windows, define the
development split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from fcpg.lewm_goal_relative_world_sidecar import (
    EEF_SLOT,
    GoalRelativeWorldSidecar,
    HORIZON,
    WORLD_SCALE_M,
    sidecar_payload,
)


TASK_TO_LOCAL_INDEX = {"reach-v3": 0, "push-v3": 1}
GROUNDINGS = ("nominal", "fcpg", "oracle")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/lewm_moving_goal_eval/reach_push_history10k_full.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lewm_moving_goal_sidecar_seed92001_2k"),
    )
    parser.add_argument("--train-grounding", choices=GROUNDINGS, default="fcpg")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=92001)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-residual-m", type=float, default=0.08)
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save corrected trajectories for HTML/diagnostic rendering.",
    )
    return parser


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and device.index == 1:
        raise ValueError("CUDA1 is disabled for this experiment")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    return device


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as archive:
        data = {name: np.array(archive[name], copy=True) for name in archive.files}
    required = {
        "task_name",
        "episode_id",
        "current_xyz",
        "current_goal",
        "current_goal_velocity",
        "target_xyz",
        "target_goal",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(
            "prediction cache is missing fields; rerun evaluate_moving_goal_lewm.py: "
            + ", ".join(missing)
        )
    for name in GROUNDINGS:
        if name not in data:
            raise ValueError(f"prediction cache is missing {name!r}")
    task_names = data["task_name"].astype(str)
    unknown = sorted(set(task_names.tolist()) - set(TASK_TO_LOCAL_INDEX))
    if unknown:
        raise ValueError(f"unsupported moving-goal task names: {unknown}")
    n = len(task_names)
    expected = {
        "current_xyz": (n, 3, 3),
        "current_goal": (n, 3),
        "current_goal_velocity": (n, 3),
        "target_xyz": (n, HORIZON, 3, 3),
        "target_goal": (n, HORIZON, 3),
    }
    for name, shape in expected.items():
        if data[name].shape != shape:
            raise ValueError(f"{name} has shape {data[name].shape}, expected {shape}")
    for name in GROUNDINGS:
        if data[name].shape != (n, HORIZON, 3, 3):
            raise ValueError(f"{name} has an unexpected trajectory shape")
    return data


def _split(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    task_names = data["task_name"].astype(str)
    episodes = data["episode_id"].astype(np.int64)
    train = np.zeros(len(task_names), dtype=bool)
    dev = np.zeros(len(task_names), dtype=bool)
    details: dict[str, Any] = {}
    for task in sorted(set(task_names.tolist())):
        task_episodes = sorted(set(episodes[task_names == task].tolist()))
        if len(task_episodes) < 2:
            raise ValueError(f"need at least two episodes for {task}")
        split_at = max(1, len(task_episodes) // 2)
        train_episodes = task_episodes[:split_at]
        dev_episodes = task_episodes[split_at:]
        task_mask = task_names == task
        train |= task_mask & np.isin(episodes, train_episodes)
        dev |= task_mask & np.isin(episodes, dev_episodes)
        details[task] = {
            "train_episodes": train_episodes,
            "development_episodes": dev_episodes,
            "train_rows": int((task_mask & train).sum()),
            "development_rows": int((task_mask & dev).sum()),
        }
    if not train.any() or not dev.any() or np.any(train & dev):
        raise ValueError("invalid episode split")
    return train, dev, details


def _local_tasks(task_names: np.ndarray, device: torch.device) -> torch.Tensor:
    values = np.asarray([TASK_TO_LOCAL_INDEX[str(name)] for name in task_names])
    return torch.from_numpy(values.astype(np.int64)).to(device)


def _tensors(
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    grounding: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    task_names = data["task_name"].astype(str)[indices]
    return {
        "base": torch.from_numpy(data[grounding][indices]).float().to(device),
        "current": torch.from_numpy(data["current_xyz"][indices]).float().to(device),
        "goal": torch.from_numpy(data["current_goal"][indices]).float().to(device),
        "goal_velocity": torch.from_numpy(
            data["current_goal_velocity"][indices]
        ).float().to(device),
        "task": _local_tasks(task_names, device),
        "target": torch.from_numpy(data["target_xyz"][indices]).float().to(device),
        "target_goal": torch.from_numpy(data["target_goal"][indices]).float().to(device),
    }


@torch.inference_mode()
def _evaluate(
    model: GoalRelativeWorldSidecar,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    grounding: str,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    predictions = []
    for start in range(0, len(indices), int(batch_size)):
        values = _tensors(
            data, indices[start : start + int(batch_size)], grounding, device
        )
        predictions.append(
            model(
                values["base"],
                values["current"],
                values["goal"],
                values["goal_velocity"],
                values["task"],
            ).cpu().numpy()
        )
    corrected = np.concatenate(predictions, axis=0)
    target = data["target_xyz"][indices]
    goal = data["target_goal"][indices]
    base = data[grounding][indices]
    base_eef = base[:, :, EEF_SLOT]
    corrected_eef = corrected[:, :, EEF_SLOT]
    target_eef = target[:, :, EEF_SLOT]
    base_epe = np.linalg.norm(base_eef - target_eef, axis=-1)
    corrected_epe = np.linalg.norm(corrected_eef - target_eef, axis=-1)
    base_goal = np.linalg.norm(base_eef - goal, axis=-1)
    corrected_goal = np.linalg.norm(corrected_eef - goal, axis=-1)
    task_names = data["task_name"].astype(str)[indices]
    result: dict[str, Any] = {
        "samples": int(len(indices)),
        "base_eef_epe_mm": float(base_epe.mean() * 1000.0),
        "sidecar_eef_epe_mm": float(corrected_epe.mean() * 1000.0),
        "base_h8_eef_epe_mm": float(base_epe[:, -1].mean() * 1000.0),
        "sidecar_h8_eef_epe_mm": float(corrected_epe[:, -1].mean() * 1000.0),
        "base_goal_distance_mm": float(base_goal.mean() * 1000.0),
        "sidecar_goal_distance_mm": float(corrected_goal.mean() * 1000.0),
        "base_h8_goal_distance_mm": float(base_goal[:, -1].mean() * 1000.0),
        "sidecar_h8_goal_distance_mm": float(corrected_goal[:, -1].mean() * 1000.0),
        "by_task": {},
    }
    for task in sorted(set(task_names.tolist())):
        mask = task_names == task
        result["by_task"][task] = {
            "samples": int(mask.sum()),
            "base_h8_eef_epe_mm": float(base_epe[mask, -1].mean() * 1000.0),
            "sidecar_h8_eef_epe_mm": float(
                corrected_epe[mask, -1].mean() * 1000.0
            ),
            "base_h8_goal_distance_mm": float(base_goal[mask, -1].mean() * 1000.0),
            "sidecar_h8_goal_distance_mm": float(
                corrected_goal[mask, -1].mean() * 1000.0
            ),
        }
    result["delta_sidecar_minus_base_h8_goal_mm"] = float(
        result["sidecar_h8_goal_distance_mm"]
        - result["base_h8_goal_distance_mm"]
    )
    return result, corrected


def _train(
    model: GoalRelativeWorldSidecar,
    data: dict[str, np.ndarray],
    train_indices: np.ndarray,
    grounding: str,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> list[float]:
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-4)
    horizon_weights = torch.linspace(1.0, 2.0, HORIZON, device=device)
    losses: list[float] = []
    model.train()
    for step in range(1, int(steps) + 1):
        count = min(int(batch_size), len(train_indices))
        selected = rng.choice(train_indices, size=count, replace=len(train_indices) < count)
        values = _tensors(data, selected, grounding, device)
        prediction = model(
            values["base"],
            values["current"],
            values["goal"],
            values["goal_velocity"],
            values["task"],
        )
        error = (prediction[:, :, EEF_SLOT] - values["target"][:, :, EEF_SLOT]) / WORLD_SCALE_M
        loss_map = F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none").mean(dim=-1)
        loss = (loss_map * horizon_weights[None]).mean()
        residual = (prediction[:, :, EEF_SLOT] - values["base"][:, :, EEF_SLOT]) / WORLD_SCALE_M
        loss = loss + 1.0e-3 * residual.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def main() -> int:
    args = _parser().parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch-size must be positive")
    device = _resolve_device(args.device)
    data = _load(args.predictions)
    train_mask, dev_mask, split_details = _split(data)
    train_indices = np.flatnonzero(train_mask)
    dev_indices = np.flatnonzero(dev_mask)
    model = GoalRelativeWorldSidecar(
        hidden_dim=int(args.hidden_dim), max_residual_m=float(args.max_residual_m)
    ).to(device)
    losses = _train(
        model,
        data,
        train_indices,
        args.train_grounding,
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        device=device,
    )
    metrics: dict[str, Any] = {
        "protocol": "moving_goal_goal_relative_world_sidecar_v1",
        "prediction_cache": str(args.predictions.expanduser().resolve()),
        "device": str(device),
        "steps": int(args.steps),
        "seed": int(args.seed),
        "train_grounding": args.train_grounding,
        "split": split_details,
        "train_rows": int(len(train_indices)),
        "development_rows": int(len(dev_indices)),
        "groundings": {},
    }
    corrected_predictions: dict[str, np.ndarray] = {}
    for grounding in GROUNDINGS:
        train_metrics, train_corrected = _evaluate(
            model, data, train_indices, grounding, device
        )
        dev_metrics, dev_corrected = _evaluate(
            model, data, dev_indices, grounding, device
        )
        metrics["groundings"][grounding] = {
            "train": train_metrics,
            "development": dev_metrics,
        }
        ordered = np.empty_like(data[grounding], dtype=np.float32)
        ordered[train_indices] = train_corrected
        ordered[dev_indices] = dev_corrected
        corrected_predictions[grounding] = ordered
    metrics["loss_first"] = losses[0]
    metrics["loss_last"] = losses[-1]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = sidecar_payload(
        model,
        step=int(args.steps),
        seed=int(args.seed),
        train_grounding=args.train_grounding,
        metrics=metrics,
    )
    torch.save(checkpoint, output_dir / "goal_relative_world_sidecar_latest.pt")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.save_predictions:
        np.savez_compressed(
            output_dir / "predictions.npz",
            task_name=data["task_name"],
            episode_id=data["episode_id"],
            current_row=data.get("current_row", np.arange(len(data["task_name"]))),
            target_xyz=data["target_xyz"],
            target_goal=data["target_goal"],
            nominal=data["nominal"],
            fcpg=data["fcpg"],
            oracle=data["oracle"],
            sidecar_nominal=corrected_predictions["nominal"],
            sidecar_fcpg=corrected_predictions["fcpg"],
            sidecar_oracle=corrected_predictions["oracle"],
        )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
