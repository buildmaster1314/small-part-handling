#!/usr/bin/env python3
"""Fit a low-order camera-equivariant correction to a frozen orbit LeWM."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a4 import HORIZON, TASKS  # noqa: E402
from fcpg.lewm_action_prefix_joint_a4 import ANCHOR_COUNT  # noqa: E402
from fcpg.lewm_action_prefix_joint_data import open_raw_replay_arrays  # noqa: E402
from fcpg.lewm_harmonic_camera_readout import (  # noqa: E402
    HARMONIC_DIM,
    INPUT_COMPONENT_DIM,
    HarmonicCameraReadout,
    harmonic_checkpoint,
    orbit_harmonic_features,
)
from scripts import train_lewm_pose_conditioned_readout as pose_train  # noqa: E402


DEFAULT_RAW = pose_train.DEFAULT_RAW
DEFAULT_BASE = pose_train.DEFAULT_BASE
BRANCH_COUNT = 8


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--cache-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if not np.isfinite(args.ridge) or args.ridge <= 0.0:
        parser.error("--ridge must be finite and positive")
    if args.cache_batch_size <= 0 or args.eval_batch_size <= 0:
        parser.error("batch sizes must be positive")
    return args


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(value), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fit(
    model: HarmonicCameraReadout,
    cache: Mapping[str, torch.Tensor],
    index: Mapping[str, np.ndarray],
    *,
    ridge: float,
    image_size: tuple[int, int],
) -> None:
    split = np.asarray(index["split"]).astype(str)
    train_bundles = np.flatnonzero(split == "train")
    task = np.asarray(index["task_index"], dtype=np.int64)
    hours = torch.from_numpy(
        np.asarray(index["camera_clock_hour"], dtype=np.int64)
    )
    harmonic = orbit_harmonic_features(hours).double().numpy()
    base_q = cache["base_q"].float().numpy()
    current_q = cache["current_q"].float().numpy()
    target_q = cache["target_q"].float().numpy()
    valid = cache["valid"].numpy().astype(bool)
    scale = np.asarray([image_size[0] - 1, image_size[1] - 1], dtype=np.float64)
    weights = np.zeros_like(model.weight.detach().cpu().numpy(), dtype=np.float64)
    regularizer = float(ridge) * np.eye(
        HARMONIC_DIM * INPUT_COMPONENT_DIM, dtype=np.float64
    )
    for task_index in range(len(TASKS)):
        bundles = train_bundles[task[train_bundles] == task_index]
        if bundles.size == 0:
            raise ValueError(f"train split misses {TASKS[task_index]}")
        repeated_harmonic = np.repeat(harmonic[bundles], BRANCH_COUNT, axis=0)
        repeated_current = np.repeat(current_q[bundles], BRANCH_COUNT, axis=0)
        task_base = base_q[bundles].reshape(-1, HORIZON, ANCHOR_COUNT, 2)
        task_target = target_q[bundles].reshape(-1, HORIZON, ANCHOR_COUNT, 2)
        task_valid = valid[bundles].reshape(-1, HORIZON, ANCHOR_COUNT)
        for horizon in range(HORIZON):
            for anchor in range(ANCHOR_COUNT):
                mask = task_valid[:, horizon, anchor]
                displacement = (
                    task_base[mask, horizon, anchor]
                    - repeated_current[mask, anchor]
                ) / scale
                augmented = np.concatenate(
                    [
                        displacement,
                        np.ones((int(mask.sum()), 1), dtype=np.float64),
                    ],
                    axis=1,
                )
                design = (
                    repeated_harmonic[mask, :, None] * augmented[:, None, :]
                ).reshape(-1, HARMONIC_DIM * INPUT_COMPONENT_DIM)
                target_residual = (
                    task_target[mask, horizon, anchor]
                    - task_base[mask, horizon, anchor]
                ) / scale
                gram = design.T @ design + regularizer
                solution = np.linalg.solve(gram, design.T @ target_residual)
                weights[task_index, horizon, anchor] = solution
    model.weight.data.copy_(torch.from_numpy(weights).float())


def _angle_error(
    prediction: np.ndarray,
    target: np.ndarray,
    current: np.ndarray,
) -> np.ndarray:
    predicted_delta = prediction.mean(axis=2) - current.mean(axis=1)[:, None]
    target_delta = target.mean(axis=2) - current.mean(axis=1)[:, None]
    predicted_norm = np.linalg.norm(predicted_delta, axis=-1)
    target_norm = np.linalg.norm(target_delta, axis=-1)
    cosine = np.sum(predicted_delta * target_delta, axis=-1) / np.maximum(
        predicted_norm * target_norm, 1.0e-6
    )
    angle = np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
    angle[(target_norm < 4.0) | (predicted_norm < 1.0e-3)] = np.nan
    return angle


def _task_macro(per_task: Mapping[str, Mapping[str, float]], name: str) -> float:
    return float(np.mean([float(per_task[task][name]) for task in TASKS]))


@torch.inference_mode()
def _evaluate(
    model: HarmonicCameraReadout,
    cache: Mapping[str, torch.Tensor],
    index: Mapping[str, np.ndarray],
    *,
    split_name: str,
    batch_size: int,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    split = np.asarray(index["split"]).astype(str)
    bundles = np.flatnonzero(split == split_name)
    pair_bundle = np.repeat(bundles, BRANCH_COUNT)
    pair_branch = np.tile(np.arange(BRANCH_COUNT, dtype=np.int64), bundles.size)
    base_all = cache["base_q"][pair_bundle, pair_branch].float()
    target_all = cache["target_q"][pair_bundle, pair_branch].float()
    current_all = cache["current_q"][pair_bundle].float()
    task_all = cache["task"][pair_bundle].long()
    hour_all = torch.from_numpy(
        np.asarray(index["camera_clock_hour"], dtype=np.int64)[pair_bundle]
    ).long()
    predictions = []
    cyclic_predictions = []
    for start in range(0, pair_bundle.size, int(batch_size)):
        stop = min(start + int(batch_size), pair_bundle.size)
        output = model(
            base_prediction_q_px=base_all[start:stop],
            current_q_px=current_all[start:stop],
            task_index=task_all[start:stop],
            clock_hour=hour_all[start:stop],
            image_size=image_size,
        )
        predictions.append(output.predicted_q_px.cpu())
        cyclic_branch = (pair_branch[start:stop] + 1) % BRANCH_COUNT
        cyclic_base = cache["base_q"][
            pair_bundle[start:stop], cyclic_branch
        ].float()
        cyclic_predictions.append(
            model(
                base_prediction_q_px=cyclic_base,
                current_q_px=current_all[start:stop],
                task_index=task_all[start:stop],
                clock_hour=hour_all[start:stop],
                image_size=image_size,
            ).predicted_q_px.cpu()
        )
    prediction = torch.cat(predictions).numpy()
    cyclic = torch.cat(cyclic_predictions).numpy()
    base = base_all.numpy()
    target = target_all.numpy()
    current = current_all.numpy()
    hold = np.repeat(current[:, None], HORIZON, axis=1)

    def epe(value: np.ndarray) -> np.ndarray:
        return np.linalg.norm(value - target, axis=-1).mean(axis=(1, 2))

    values = {
        "base": epe(base),
        "harmonic": epe(prediction),
        "cyclic": epe(cyclic),
        "hold": epe(hold),
        "base_angle": _angle_error(base, target, current),
        "harmonic_angle": _angle_error(prediction, target, current),
    }
    task_np = task_all.numpy()
    camera_names = np.asarray(index["camera_name"]).astype(str)[pair_bundle]
    per_task: dict[str, Any] = {}
    per_task_view: dict[str, Any] = {}
    for task_index, task_name in enumerate(TASKS):
        mask = task_np == task_index
        per_task[task_name] = {
            "pair_count": int(mask.sum()),
            "base_anchor_epe_px": float(values["base"][mask].mean()),
            "harmonic_anchor_epe_px": float(values["harmonic"][mask].mean()),
            "harmonic_cyclic_anchor_epe_px": float(values["cyclic"][mask].mean()),
            "hold_current_anchor_epe_px": float(values["hold"][mask].mean()),
            "base_direction_mae_deg": float(
                np.nanmean(values["base_angle"][mask])
            ),
            "harmonic_direction_mae_deg": float(
                np.nanmean(values["harmonic_angle"][mask])
            ),
        }
        for camera in sorted(set(camera_names[mask].tolist())):
            view_mask = mask & (camera_names == camera)
            per_task_view[f"{task_name}::{camera}"] = {
                "pair_count": int(view_mask.sum()),
                "base_anchor_epe_px": float(values["base"][view_mask].mean()),
                "harmonic_anchor_epe_px": float(
                    values["harmonic"][view_mask].mean()
                ),
                "base_direction_mae_deg": float(
                    np.nanmean(values["base_angle"][view_mask])
                ),
                "harmonic_direction_mae_deg": float(
                    np.nanmean(values["harmonic_angle"][view_mask])
                ),
            }
    result = {
        "split": split_name,
        "pair_count": int(pair_bundle.size),
        "task_macro_base_anchor_epe_px": _task_macro(
            per_task, "base_anchor_epe_px"
        ),
        "task_macro_harmonic_anchor_epe_px": _task_macro(
            per_task, "harmonic_anchor_epe_px"
        ),
        "task_macro_harmonic_cyclic_anchor_epe_px": _task_macro(
            per_task, "harmonic_cyclic_anchor_epe_px"
        ),
        "task_macro_hold_current_anchor_epe_px": _task_macro(
            per_task, "hold_current_anchor_epe_px"
        ),
        "task_macro_base_direction_mae_deg": _task_macro(
            per_task, "base_direction_mae_deg"
        ),
        "task_macro_harmonic_direction_mae_deg": _task_macro(
            per_task, "harmonic_direction_mae_deg"
        ),
        "per_task": per_task,
        "per_task_view": per_task_view,
    }
    result["task_macro_epe_improvement_fraction"] = 1.0 - (
        result["task_macro_harmonic_anchor_epe_px"]
        / result["task_macro_base_anchor_epe_px"]
    )
    result["task_macro_cyclic_penalty_px"] = (
        result["task_macro_harmonic_cyclic_anchor_epe_px"]
        - result["task_macro_harmonic_anchor_epe_px"]
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser(argv)
    device = pose_train._cuda0(args.device)
    torch.cuda.set_device(device)
    raw_path = args.raw_dataset.expanduser().resolve()
    base_path = args.base_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("harmonic readout output directory must be empty")
    index, arrays = open_raw_replay_arrays(raw_path)
    base_model, _base_payload = pose_train._load_base(base_path, device=device)
    started = time.perf_counter()
    cache = pose_train._build_cache(
        base_model,
        index,
        arrays,
        device=device,
        batch_size=int(args.cache_batch_size),
    )
    del base_model
    torch.cuda.empty_cache()
    image_size = (
        int(arrays["current_rgb"].shape[-1]),
        int(arrays["current_rgb"].shape[-2]),
    )
    model = HarmonicCameraReadout()
    _fit(model, cache, index, ridge=float(args.ridge), image_size=image_size)
    train = _evaluate(
        model,
        cache,
        index,
        split_name="train",
        batch_size=int(args.eval_batch_size),
        image_size=image_size,
    )
    development = _evaluate(
        model,
        cache,
        index,
        split_name="development",
        batch_size=int(args.eval_batch_size),
        image_size=image_size,
    )
    metrics = {
        "protocol": "a4_orbit_harmonic_camera_fit_v1",
        "ridge": float(args.ridge),
        "fit_seconds": time.perf_counter() - started,
        "train_evaluation": train,
        "development_evaluation": development,
    }
    _atomic_json(output_dir / "metrics.json", metrics)
    _atomic_json(
        output_dir / "resolved_config.json",
        {
            "raw_dataset": str(raw_path),
            "base_checkpoint": str(base_path),
            "ridge": float(args.ridge),
            "harmonic_orders": [0, 1, 2],
            "train_camera_hours": [2, 4, 6, 8, 10, 12],
            "heldout_development_camera_hours": [1, 3, 5, 7, 9, 11],
            "selection": "closed_form_fixed_ridge",
        },
    )
    _atomic_torch(
        output_dir / "lewm_harmonic_camera_readout.pt",
        harmonic_checkpoint(
            model,
            base_checkpoint=str(base_path),
            raw_dataset=str(raw_path),
            ridge=float(args.ridge),
        ),
    )
    print(json.dumps(metrics, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
