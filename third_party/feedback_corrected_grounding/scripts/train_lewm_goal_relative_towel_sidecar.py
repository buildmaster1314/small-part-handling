#!/usr/bin/env python3
"""Train a small goal-relative residual on the frozen seven-task LeWM.

This is a development experiment for the towel endpoint failure.  The parent
LeWM is never updated.  A feature cache is built from the current FCPG-grounded
observation and recorded action prefixes, then a bounded sidecar is trained on
the towel train episodes and evaluated on disjoint development episodes.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a4 import HORIZON, TASKS  # noqa: E402
from fcpg.lewm_camera_action_joint_a4 import (  # noqa: E402
    load_camera_action_joint_checkpoint,
)
from fcpg.lewm_goal_relative_sidecar import (  # noqa: E402
    GoalRelativeActionSidecar,
    GoalRelativePixelSidecar,
    sidecar_payload,
    towel_goal_projection,
)


RAW_DATASET = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "joint_orbit12_gripper_phaseaware_a4c_v4_48train_24dev"
)
PARENT_CHECKPOINT = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "camera_action_phaseaware_a4c_v4_seed40571_10k/"
    "lewm_camera_action_joint_latest.pt"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "goal_relative_towel_sidecar_seed40572_2k"
)
IMAGE_SIZE = (224, 224)
TOWEL_TASK = "towel-wipe-v0"


def _device(value: str) -> torch.device:
    result = torch.device(value)
    if result.type != "cuda" or result.index not in (None, 0):
        raise ValueError("goal sidecar permits only CUDA device 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--raw-dataset", type=Path, default=RAW_DATASET)
    parser.add_argument("--parent-checkpoint", type=Path, default=PARENT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=40572)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--skip-precompute", action="store_true")
    parser.add_argument(
        "--action-aware",
        action="store_true",
        help="condition the sidecar on the transformed H1-H8 action prefix",
    )
    return parser


def _load_raw(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    root = path.expanduser().resolve()
    with np.load(root / "index.npz", allow_pickle=False) as archive:
        index = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays: dict[str, np.ndarray] = {}
    for name in (
        "current_rgb",
        "current_proprio",
        "current_q_nominal",
        "current_q_corrected",
        "current_keypoints_xyz",
        "current_valid",
        "future_q_true",
        "camera_intrinsic",
        "camera_extrinsic_cam_from_world",
    ):
        arrays[name] = np.load(root / f"{name}.npy", mmap_mode="r")
    required_index = {
        "task_name",
        "task_index",
        "task_instance",
        "proposal_phase_name",
        "split",
        "future_action",
    }
    missing = sorted(required_index - set(index))
    if missing:
        raise ValueError(f"raw index is missing {missing}")
    return index, arrays


def _goal_context(
    index: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    goals = []
    depths = []
    valid = []
    for row in rows:
        pixel, depth, is_valid = towel_goal_projection(
            int(index["task_instance"][row]),
            str(index["proposal_phase_name"][row]),
            arrays["camera_intrinsic"][row],
            arrays["camera_extrinsic_cam_from_world"][row],
        )
        goals.append(pixel)
        depths.append(depth)
        valid.append(is_valid)
    goal = np.asarray(goals, dtype=np.float32)
    context = np.concatenate(
        [goal, np.asarray(depths, dtype=np.float32)[:, None], np.asarray(valid, dtype=np.float32)[:, None]],
        axis=1,
    )
    if not np.isfinite(context).all() or not np.isfinite(goal).all():
        raise ValueError("goal context is non-finite")
    return goal, context


@torch.inference_mode()
def _precompute_features(
    index: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    model: Any,
    *,
    rows: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    n_rows = int(rows.size)
    if n_rows <= 0:
        raise ValueError("no towel rows selected")
    goal_px, goal_context = _goal_context(index, arrays, rows)
    base_q = np.empty((n_rows, 8, HORIZON, 3, 2), dtype=np.float32)
    transformed_actions = np.empty(
        (n_rows, 8, HORIZON, 20), dtype=np.float32
    )
    target_q = np.asarray(arrays["future_q_true"][rows], dtype=np.float32).copy()
    current_q = np.asarray(arrays["current_q_corrected"][rows], dtype=np.float32).copy()
    current_valid = np.asarray(arrays["current_valid"][rows], dtype=np.bool_).copy()
    task_index = np.asarray(index["task_index"][rows], dtype=np.int64).copy()
    actions = np.asarray(index["future_action"][rows], dtype=np.float32)
    extrinsics = np.asarray(
        arrays["camera_extrinsic_cam_from_world"][rows], dtype=np.float32
    )
    model.eval()
    for offset in range(0, n_rows, int(batch_size)):
        local = rows[offset : offset + int(batch_size)]
        count = int(local.size)
        rgb = torch.from_numpy(np.ascontiguousarray(arrays["current_rgb"][local]))[:, None]
        proprio = torch.from_numpy(
            np.ascontiguousarray(arrays["current_proprio"][local])
        )[:, None]
        q_nominal = torch.from_numpy(
            np.ascontiguousarray(arrays["current_q_nominal"][local])
        )[:, None]
        q_corrected_t = torch.from_numpy(
            np.ascontiguousarray(arrays["current_q_corrected"][local])
        )[:, None]
        valid = torch.from_numpy(
            np.ascontiguousarray(arrays["current_valid"][local])
        )[:, None]
        xyz = torch.from_numpy(
            np.ascontiguousarray(arrays["current_keypoints_xyz"][local])
        )[:, None]
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            state = model.encode_observations(
                rgb=rgb.to(device),
                proprio=proprio.to(device),
                q_nominal=q_nominal.to(device),
                q_corrected=q_corrected_t.to(device),
                keypoints_xyz=xyz.to(device),
                valid=valid.to(device),
            )[:, 0]
            state_8 = state[:, None].expand(-1, 8, -1, -1).reshape(
                count * 8, state.shape[1], state.shape[2]
            )
            action_t = torch.from_numpy(
                np.ascontiguousarray(actions[offset : offset + count])
            ).to(device=device, dtype=state.dtype)
            action_t = action_t.reshape(count * 8, HORIZON, -1)
            ext = torch.from_numpy(
                np.ascontiguousarray(extrinsics[offset : offset + count])
            ).to(device=device, dtype=state.dtype)
            ext = ext[:, None].expand(-1, 8, -1, -1).reshape(count * 8, 4, 4)
            action_t = model.transform_future_action(
                action_t, {"camera_extrinsic_cam_from_world": ext}
            )
            transformed_actions[offset : offset + count] = (
                action_t.reshape(count, 8, HORIZON, 20).float().cpu().numpy()
            )
            task_t = torch.from_numpy(
                np.repeat(task_index[offset : offset + count], 8)
            ).to(device=device, dtype=torch.long)
            predicted_state = model.dynamics(state_8, action_t, task_index=task_t)
            current_q_t = q_corrected_t[:, 0].to(device).repeat_interleave(8, 0)
            predicted_q = model.metric_readout(
                predicted_state,
                state_8,
                current_q_t,
                task_t,
                image_size=IMAGE_SIZE,
            )
        base_q[offset : offset + count] = predicted_q.float().cpu().numpy().reshape(
            count, 8, HORIZON, 3, 2
        )
        print(json.dumps({"precompute_rows": int(offset + count), "total": n_rows}), flush=True)
    return {
        "base_q": base_q,
        "future_action": transformed_actions,
        "target_q": target_q,
        "current_q": current_q,
        "current_valid": current_valid,
        "goal_px": goal_px,
        "goal_context": goal_context,
        "task_index": task_index,
        "split": np.asarray(index["split"][rows]).astype("U16"),
        "task_instance": np.asarray(index["task_instance"][rows], dtype=np.int64),
        "proposal_phase_name": np.asarray(index["proposal_phase_name"][rows]).astype("U32"),
        "proposal_raw_step": np.asarray(index["proposal_raw_step"][rows], dtype=np.int64),
        "camera_clock_hour": np.asarray(index["camera_clock_hour"][rows], dtype=np.int64),
    }


def _save_features(path: Path, features: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **features)


def _flatten_features(
    features: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    rows = int(features["base_q"].shape[0])
    result = {
        "base_q": np.asarray(features["base_q"], dtype=np.float32).reshape(
            rows * 8, HORIZON, 3, 2
        ),
        "target_q": np.asarray(features["target_q"], dtype=np.float32).reshape(
            rows * 8, HORIZON, 3, 2
        ),
        "current_q": np.asarray(features["current_q"], dtype=np.float32)[:, None]
        .repeat(8, axis=1)
        .reshape(rows * 8, 3, 2),
        "current_valid": np.asarray(features["current_valid"], dtype=np.bool_)[:, None]
        .repeat(8, axis=1)
        .reshape(rows * 8, 3),
        "goal_px": np.asarray(features["goal_px"], dtype=np.float32)[:, None]
        .repeat(8, axis=1)
        .reshape(rows * 8, 2),
        "task_index": np.asarray(features["task_index"], dtype=np.int64)[:, None]
        .repeat(8, axis=1)
        .reshape(rows * 8),
    }
    if "future_action" in features:
        result["future_action"] = np.asarray(
            features["future_action"], dtype=np.float32
        ).reshape(rows * 8, HORIZON, 20)
    return result


@torch.inference_mode()
def _evaluate(
    model: Any,
    values: Mapping[str, np.ndarray],
    *,
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, float]:
    model.eval()
    base_errors = []
    sidecar_errors = []
    endpoint_base = []
    endpoint_sidecar = []
    scale = torch.tensor([223.0, 223.0], device=device)
    for start in range(0, len(values["base_q"]), batch_size):
        stop = min(len(values["base_q"]), start + batch_size)
        sl = slice(start, stop)
        base = torch.from_numpy(values["base_q"][sl]).to(device)
        current = torch.from_numpy(values["current_q"][sl]).to(device)
        goal = torch.from_numpy(values["goal_px"][sl]).to(device)
        task = torch.from_numpy(values["task_index"][sl]).to(device)
        valid = torch.from_numpy(values["current_valid"][sl]).to(device)
        target = torch.from_numpy(values["target_q"][sl]).to(device)
        kwargs = {"image_size": IMAGE_SIZE, "valid": valid}
        if isinstance(model, GoalRelativeActionSidecar):
            if "future_action" not in values:
                raise ValueError("action-aware model requires future_action values")
            kwargs["future_action"] = torch.from_numpy(
                values["future_action"][sl]
            ).to(device)
        pred = model(base, current, goal, task, **kwargs)
        base_errors.append(torch.linalg.vector_norm((base - target), dim=-1).mean().item())
        sidecar_errors.append(torch.linalg.vector_norm((pred - target), dim=-1).mean().item())
        endpoint_base.append(torch.linalg.vector_norm(base[:, -1].mean(1) - target[:, -1].mean(1), dim=-1).mean().item())
        endpoint_sidecar.append(torch.linalg.vector_norm(pred[:, -1].mean(1) - target[:, -1].mean(1), dim=-1).mean().item())
    return {
        "base_anchor_epe_px": float(np.mean(base_errors)),
        "sidecar_anchor_epe_px": float(np.mean(sidecar_errors)),
        "base_endpoint_center_epe_px": float(np.mean(endpoint_base)),
        "sidecar_endpoint_center_epe_px": float(np.mean(endpoint_sidecar)),
        "sample_count": int(len(values["base_q"])),
    }


def _train(
    train_values: Mapping[str, np.ndarray],
    dev_values: Mapping[str, np.ndarray],
    *,
    steps: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    model_class: Any,
) -> tuple[Any, dict[str, Any]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = model_class().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-4)
    rng = np.random.default_rng(int(seed))
    count = len(train_values["base_q"])
    if count == 0:
        raise ValueError("no training rows")
    horizon_weights = torch.linspace(1.0, 2.0, HORIZON, device=device)[None, :, None]
    for step in range(1, int(steps) + 1):
        indices = rng.integers(0, count, size=min(int(batch_size), count))
        base = torch.from_numpy(train_values["base_q"][indices]).to(device)
        current = torch.from_numpy(train_values["current_q"][indices]).to(device)
        goal = torch.from_numpy(train_values["goal_px"][indices]).to(device)
        task = torch.from_numpy(train_values["task_index"][indices]).to(device)
        valid = torch.from_numpy(train_values["current_valid"][indices]).to(device)
        target = torch.from_numpy(train_values["target_q"][indices]).to(device)
        kwargs = {"image_size": IMAGE_SIZE, "valid": valid}
        if isinstance(model, GoalRelativeActionSidecar):
            if "future_action" not in train_values:
                raise ValueError("action-aware model requires future_action values")
            kwargs["future_action"] = torch.from_numpy(
                train_values["future_action"][indices]
            ).to(device)
        prediction = model(base, current, goal, task, **kwargs)
        loss_map = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
        loss = (loss_map * horizon_weights).mean()
        residual = (prediction - base) / 223.0
        loss = loss + 1.0e-3 * residual.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 200 == 0 or step == int(steps):
            print(json.dumps({"step": step, "loss": float(loss.detach().cpu())}), flush=True)
    metrics = {
        "train": _evaluate(model, train_values, device=device),
        "development": _evaluate(model, dev_values, device=device),
    }
    return model, metrics


def main() -> int:
    args = _parser().parse_args()
    if int(args.steps) <= 0 or int(args.batch_size) <= 0:
        raise ValueError("steps and batch-size must be positive")
    device = _device(args.device)
    torch.cuda.set_device(device)
    raw_path = args.raw_dataset.expanduser().resolve()
    parent_path = args.parent_checkpoint.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    index, arrays = _load_raw(raw_path)
    towel_rows = np.flatnonzero(np.asarray(index["task_name"]).astype(str) == TOWEL_TASK)
    if towel_rows.size == 0:
        raise ValueError("raw dataset contains no towel rows")
    feature_path = output / "towel_frozen_lewm_features.npz"
    if args.skip_precompute and not feature_path.exists():
        raise FileNotFoundError(f"--skip-precompute requested but {feature_path} is absent")
    if not feature_path.exists() or not args.skip_precompute:
        payload = torch.load(parent_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise TypeError("parent checkpoint must be a mapping")
        model = load_camera_action_joint_checkpoint(payload).to(device).eval()
        features = _precompute_features(
            index,
            arrays,
            model,
            rows=towel_rows,
            device=device,
            batch_size=int(args.inference_batch_size),
        )
        _save_features(feature_path, features)
        del model
        torch.cuda.empty_cache()
    with np.load(feature_path, allow_pickle=False) as archive:
        features = {name: np.array(archive[name], copy=True) for name in archive.files}
    flat = _flatten_features(features)
    if args.action_aware and "future_action" not in flat:
        raise ValueError(
            "action-aware training requires a cache containing future_action"
        )
    row_split = np.asarray(features["split"]).astype(str)
    train_mask = np.repeat(row_split == "train", 8)
    dev_mask = np.repeat(row_split == "development", 8)
    train_values = {name: value[train_mask] for name, value in flat.items()}
    dev_values = {name: value[dev_mask] for name, value in flat.items()}
    model_class = GoalRelativeActionSidecar if args.action_aware else GoalRelativePixelSidecar
    sidecar, metrics = _train(
        train_values,
        dev_values,
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        device=device,
        model_class=model_class,
    )
    checkpoint = sidecar_payload(
        sidecar,
        step=int(args.steps),
        seed=int(args.seed),
        parent_checkpoint=str(parent_path),
        feature_cache=str(feature_path),
        metrics=metrics,
    )
    checkpoint_path = output / "goal_relative_sidecar_latest.pt"
    torch.save(checkpoint, checkpoint_path)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"checkpoint": str(checkpoint_path), **metrics}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
