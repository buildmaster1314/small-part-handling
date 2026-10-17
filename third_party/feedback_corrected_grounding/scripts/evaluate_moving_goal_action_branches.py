#!/usr/bin/env python3
"""Evaluate frozen FCPG/LeWM forecasts on same-state moving-goal branches.

This is a small development diagnostic.  It deliberately keeps the target
entity in the labels: the frozen seven-task LeWM predicts three robot anchors,
so Push entity-goal ranking is reported as unavailable rather than silently
using end-effector distance as a task metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fcpg.lewm_history_conditioned_metric3d_a4 import load_history_conditioned_checkpoint
from fcpg.metaworld_utils import DEFAULT_KEYPOINT_SITES
from fcpg.transformer_grounding import load_fcpg_vit_adapter


HORIZON = 8
CANDIDATE_COUNT = 8
ANCHOR_COUNT = 3
RAW_ACTION_DIM = 4
RAW_STEPS_PER_TOKEN = 5
TASK_INDEX = {"reach-v3": 2, "push-v3": 3}
EEF_SLOT = 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/metaworld_projection/moving_goal_reach_push_branches_dev.npz"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/lewm_action_prefix_a4/camera_action_metric3d_world_history_seed40583_10k/"
            "lewm_history_conditioned_latest.pt"
        ),
    )
    parser.add_argument(
        "--fcpg-checkpoint",
        type=Path,
        default=Path(
            "outputs/transformer_residual/lewm_stage1/shared_4task_seed0_sig002/"
            "fcpg_transformer_residual_best.pt"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/lewm_moving_goal_eval/branches.json"))
    parser.add_argument("--save-predictions", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    return parser


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and device.index == 1:
        raise ValueError("CUDA1 is disabled")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    return device


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as archive:
        data = {name: np.array(archive[name], copy=True) for name in archive.files}
    required = {
        "current_rgb", "current_obs", "current_xyz", "current_q_true",
        "current_q_nominal", "current_valid_true", "current_valid_nominal",
        "current_entity_xyz", "current_goal", "current_goal_velocity",
        "history_action", "candidate_action", "target_xyz", "target_entity_xyz",
        "target_goal", "camera_intrinsic", "true_extrinsic", "task_name",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"branch dataset is missing {missing}")
    n = len(data["task_name"])
    expected = {
        "current_rgb": (n, 224, 224, 3),
        "current_obs": (n, data["current_obs"].shape[1]),
        "current_xyz": (n, 3, 3),
        "current_q_true": (n, 3, 2),
        "current_q_nominal": (n, 3, 2),
        "current_valid_true": (n, 3),
        "current_valid_nominal": (n, 3),
        "current_entity_xyz": (n, 3),
        "current_goal": (n, 3),
        "current_goal_velocity": (n, 3),
        "history_action": (n, 2, 20),
        "candidate_action": (n, CANDIDATE_COUNT, HORIZON, 20),
        "target_xyz": (n, CANDIDATE_COUNT, HORIZON, 3, 3),
        "target_entity_xyz": (n, CANDIDATE_COUNT, HORIZON, 3),
        "target_goal": (n, CANDIDATE_COUNT, HORIZON, 3),
        "camera_intrinsic": (n, 3, 3),
        "true_extrinsic": (n, 4, 4),
    }
    for name, shape in expected.items():
        if data[name].shape != shape:
            raise ValueError(f"{name} has shape {data[name].shape}, expected {shape}")
    tasks = data["task_name"].astype(str)
    if set(tasks.tolist()) - set(TASK_INDEX):
        raise ValueError("branch dataset contains unsupported tasks")
    return data


def _rgb(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value)).float().permute(0, 3, 1, 2).to(device) / 255.0


@torch.inference_mode()
def _fcpg(
    corrector: Any,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    values = _rgb(data["current_rgb"][indices], device)
    q = torch.from_numpy(np.ascontiguousarray(data["current_q_nominal"][indices])).float().to(device)
    xyz = torch.from_numpy(np.ascontiguousarray(data["current_xyz"][indices])).float().to(device)
    proprio = torch.from_numpy(np.ascontiguousarray(data["current_obs"][indices, :4])).float().to(device)
    output = corrector(values, q, xyz, proprio)
    result = output.q_corrected_px.float().cpu().numpy()
    if result.shape != (len(indices), 3, 2) or not np.isfinite(result).all():
        raise RuntimeError("FCPG output has an invalid shape")
    return result


@torch.inference_mode()
def _forecast(
    model: Any,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    q_fcpg: np.ndarray,
    mode: str,
    device: torch.device,
) -> np.ndarray:
    if mode == "nominal":
        q = data["current_q_nominal"][indices]
        valid = data["current_valid_nominal"][indices]
    elif mode == "fcpg":
        q = q_fcpg[indices]
        valid = data["current_valid_nominal"][indices]
    elif mode == "oracle":
        q = data["current_q_true"][indices]
        valid = data["current_valid_true"][indices]
    else:
        raise ValueError(f"unknown grounding mode {mode!r}")
    rgb = _rgb(data["current_rgb"][indices], device)
    proprio = torch.from_numpy(np.ascontiguousarray(data["current_obs"][indices, :4])).float().to(device)
    q_nominal = torch.from_numpy(np.ascontiguousarray(data["current_q_nominal"][indices])).float().to(device)
    q_corrected = torch.from_numpy(np.ascontiguousarray(q)).float().to(device)
    xyz = torch.from_numpy(np.ascontiguousarray(data["current_xyz"][indices])).float().to(device)
    valid_tensor = torch.from_numpy(np.ascontiguousarray(valid)).bool().to(device)
    task = torch.from_numpy(np.asarray([TASK_INDEX[str(x)] for x in data["task_name"][indices]])).long().to(device)
    current = model.encode_observations(
        rgb=rgb[:, None], proprio=proprio[:, None], q_nominal=q_nominal[:, None],
        q_corrected=q_corrected[:, None], keypoints_xyz=xyz[:, None], valid=valid_tensor[:, None],
    )[:, 0]
    candidate_action = torch.from_numpy(np.ascontiguousarray(data["candidate_action"][indices])).float().to(device)
    batch_size = len(indices)
    current_expanded = current[:, None].expand(-1, CANDIDATE_COUNT, -1, -1).reshape(batch_size * CANDIDATE_COUNT, *current.shape[1:])
    actions = candidate_action.reshape(batch_size * CANDIDATE_COUNT, HORIZON, 20)
    history = torch.from_numpy(np.ascontiguousarray(data["history_action"][indices])).float().to(device)
    history = history[:, None].expand(-1, CANDIDATE_COUNT, -1, -1).reshape(batch_size * CANDIDATE_COUNT, 2, 20)
    task = task[:, None].expand(-1, CANDIDATE_COUNT).reshape(-1)
    extrinsic = torch.from_numpy(np.ascontiguousarray(data["true_extrinsic"][indices])).float().to(device)
    extrinsic = extrinsic[:, None].expand(-1, CANDIDATE_COUNT, -1, -1).reshape(batch_size * CANDIDATE_COUNT, 4, 4)
    xyz_expanded = xyz[:, None].expand(-1, CANDIDATE_COUNT, -1, -1).reshape(batch_size * CANDIDATE_COUNT, 3, 3)
    predicted_state = model.predict_with_history(
        current_state=current_expanded, history_action=history, future_action=actions,
        task_index=task, batch={"camera_extrinsic_cam_from_world": extrinsic},
    )
    intrinsic = torch.from_numpy(np.ascontiguousarray(data["camera_intrinsic"][indices])).float().to(device)
    intrinsic = intrinsic[:, None].expand(-1, CANDIDATE_COUNT, -1, -1).reshape(batch_size * CANDIDATE_COUNT, 3, 3)
    decoded = model.decode_3d(
        predicted_state=predicted_state, current_state=current_expanded,
        current_keypoints_xyz=xyz_expanded, future_action=actions, task_index=task,
        camera_intrinsic=intrinsic, extrinsic_cam_from_world=extrinsic,
    )
    return decoded.predicted_world_xyz.float().cpu().numpy().reshape(batch_size, CANDIDATE_COUNT, HORIZON, 3, 3)


def _cost_metrics(
    predicted: np.ndarray,
    target_xyz: np.ndarray,
    target_entity: np.ndarray,
    target_goal: np.ndarray,
    tasks: np.ndarray,
) -> dict[str, Any]:
    # The frozen model predicts robot anchors.  Reach can therefore use its
    # EEF forecast as a task score; Push must use an entity predictor later.
    eef_cost = np.linalg.norm(predicted[:, :, -1, EEF_SLOT] - target_goal[:, :, -1], axis=-1)
    actual_cost = np.linalg.norm(target_entity[:, :, -1] - target_goal[:, :, -1], axis=-1)
    anchor_epe = np.linalg.norm(predicted - target_xyz, axis=-1).mean(axis=(2, 3))
    result: dict[str, Any] = {
        "snapshots": int(len(tasks)),
        "candidate_count": CANDIDATE_COUNT,
        "robot_anchor_epe_mm": float(anchor_epe.mean() * 1000.0),
        "actual_entity_goal_h8_mm": float(actual_cost.mean() * 1000.0),
        "reach": {},
        "push": {
            "prediction_task_rank": "unavailable_without_entity_slot",
            "actual_entity_goal_h8_mm": float(actual_cost[tasks == "push-v3"].mean() * 1000.0),
        },
    }
    reach = tasks == "reach-v3"
    if bool(reach.any()):
        pred = eef_cost[reach]
        actual = actual_cost[reach]
        selected = np.argmin(pred, axis=1)
        rows = np.arange(len(selected))
        result["reach"] = {
            "snapshots": int(reach.sum()),
            "predicted_eef_goal_h8_mm": float(pred.mean() * 1000.0),
            "actual_best_entity_goal_h8_mm": float(actual.min(axis=1).mean() * 1000.0),
            "top1_rank_accuracy": float(np.mean(selected == np.argmin(actual, axis=1))),
            "selected_actual_entity_goal_h8_mm": float(actual[rows, selected].mean() * 1000.0),
            "oracle_actual_entity_goal_h8_mm": float(actual.min(axis=1).mean() * 1000.0),
            "spearman_mean": float(np.mean([_spearman(a, b) for a, b in zip(pred, actual)])),
        }
    return result


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = np.argsort(np.argsort(left, kind="mergesort"), kind="mergesort").astype(np.float64)
    right_rank = np.argsort(np.argsort(right, kind="mergesort"), kind="mergesort").astype(np.float64)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = float(np.linalg.norm(left_rank) * np.linalg.norm(right_rank))
    return float(np.dot(left_rank, right_rank) / denominator) if denominator else 0.0


def main() -> int:
    args = _parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    device = _device(args.device)
    data = _load(args.dataset)
    checkpoint = torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False)
    model = load_history_conditioned_checkpoint(checkpoint).to(device).eval()
    corrector, _ = load_fcpg_vit_adapter(args.fcpg_checkpoint.expanduser().resolve(), device=device)
    corrector.eval()
    indices = np.arange(len(data["task_name"]), dtype=np.int64)
    q_fcpg_parts = []
    for start in range(0, len(indices), int(args.batch_size)):
        q_fcpg_parts.append(_fcpg(corrector, data, indices[start : start + args.batch_size], device))
    q_fcpg = np.concatenate(q_fcpg_parts, axis=0)
    predictions: dict[str, list[np.ndarray]] = {name: [] for name in ("nominal", "fcpg", "oracle")}
    for start in range(0, len(indices), int(args.batch_size)):
        batch_indices = indices[start : start + args.batch_size]
        for mode in predictions:
            predictions[mode].append(_forecast(model, data, batch_indices, q_fcpg, mode, device))
    predicted = {name: np.concatenate(parts, axis=0) for name, parts in predictions.items()}
    tasks = data["task_name"].astype(str)
    report: dict[str, Any] = {
        "protocol": "moving_goal_same_state_branches_v1",
        "dataset": str(args.dataset.expanduser().resolve()),
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "fcpg_checkpoint": str(args.fcpg_checkpoint.expanduser().resolve()),
        "device": str(device),
        "snapshot_count": int(len(tasks)),
        "candidate_count": CANDIDATE_COUNT,
        "horizon": HORIZON,
        "prediction_inputs": ["current_rgb", "current_proprio", "current_grounding", "history_action", "candidate_action_prefix"],
        "future_rgb_used": False,
        "modes": {},
    }
    for mode, values in predicted.items():
        report["modes"][mode] = _cost_metrics(values, data["target_xyz"], data["target_entity_xyz"], data["target_goal"], tasks)
    # A valid action-dependence check: cyclically assign each same-state
    # candidate's action to another predicted trajectory and score the factual
    # forecast against its own branch target.
    shuffled = np.roll(predicted["fcpg"], 1, axis=1)
    factual_error = np.linalg.norm(predicted["fcpg"] - data["target_xyz"], axis=-1).mean()
    shuffled_error = np.linalg.norm(shuffled - data["target_xyz"], axis=-1).mean()
    report["fcpg_action_alignment"] = {
        "factual_anchor_epe_mm": float(factual_error * 1000.0),
        "cyclic_candidate_mismatch_epe_mm": float(shuffled_error * 1000.0),
        "mismatch_minus_factual_mm": float((shuffled_error - factual_error) * 1000.0),
    }
    report["fcpg_minus_nominal_anchor_epe_mm"] = float(
        np.linalg.norm(predicted["fcpg"] - data["target_xyz"], axis=-1).mean()
        * 1000.0
        - np.linalg.norm(predicted["nominal"] - data["target_xyz"], axis=-1).mean()
        * 1000.0
    )
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.save_predictions is not None:
        args.save_predictions.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_predictions.expanduser().resolve(), q_fcpg=q_fcpg, **predicted)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
