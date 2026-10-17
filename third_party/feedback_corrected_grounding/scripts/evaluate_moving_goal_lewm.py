#!/usr/bin/env python3
"""Evaluate the frozen seven-task LeWM on moving-goal Reach/Push windows.

This is an inference-only diagnostic.  The model receives the current RGB,
current grounded anchors, two previous action chunks, and the next eight
executed action chunks.  Future RGB is never passed to the predictor.  The
moving goal is used only to score the predicted end-effector trajectory.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fcpg.lewm_history_conditioned_metric3d_a4 import (
    load_history_conditioned_checkpoint,
)
from fcpg.lewm_metric3d_joint_a4 import command_integrated_world_xyz


TASK_INDEX = {"reach-v3": 2, "push-v3": 3}
RAW_ACTION_DIM = 4
RAW_STEPS_PER_TOKEN = 5
HORIZON = 8
ANCHOR_COUNT = 3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="data/metaworld_projection/moving_goal_reach_push_dev224.npz",
    )
    parser.add_argument(
        "--fcpg-cache",
        default="data/fcpg_cache/moving_goal_reach_push_dev224_sig002/q_corrected.npy",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "outputs/lewm_action_prefix_a4/"
            "camera_action_metric3d_world_history_seed40583_10k/"
            "lewm_history_conditioned_latest.pt"
        ),
    )
    parser.add_argument(
        "--output",
        default="outputs/lewm_moving_goal_eval/reach_push_history10k.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum windows; zero evaluates every available window.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=5,
        help="Raw-frame stride between adjacent decision windows.",
    )
    parser.add_argument(
        "--save-predictions",
        default="",
        help="Optional NPZ path for predicted and target trajectories.",
    )
    return parser


def _resolve_device(value: str) -> torch.device:
    if value.strip() == "cuda:1":
        raise ValueError("CUDA1 is disabled for this experiment")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    return device


def _episode_rows(data: dict[str, np.ndarray]) -> list[np.ndarray]:
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    task_name = np.asarray(data["task_name"]).astype(str)
    episode_id = np.asarray(data["episode_id"], dtype=np.int64)
    for row, key in enumerate(zip(task_name.tolist(), episode_id.tolist())):
        grouped[(str(key[0]), int(key[1]))].append(row)
    result = []
    frame_id = np.asarray(data["frame_id"], dtype=np.int64)
    for rows in grouped.values():
        ordered = np.asarray(sorted(rows, key=lambda row: int(frame_id[row])))
        result.append(ordered)
    result.sort(key=lambda rows: (str(task_name[rows[0]]), int(episode_id[rows[0]])))
    return result


def _window_rows(
    data: dict[str, np.ndarray], *, stride: int, max_samples: int
) -> np.ndarray:
    if stride <= 0:
        raise ValueError("window stride must be positive")
    windows: list[tuple[int, int, np.ndarray]] = []
    for episode_rows in _episode_rows(data):
        task_name = str(data["task_name"][episode_rows[0]])
        if task_name not in TASK_INDEX:
            continue
        if len(episode_rows) <= HORIZON * RAW_STEPS_PER_TOKEN:
            continue
        starts = range(
            0,
            len(episode_rows) - HORIZON * RAW_STEPS_PER_TOKEN,
            stride,
        )
        for start in starts:
            windows.append((int(data["task_index"][episode_rows[0]]), start, episode_rows))
    if max_samples > 0 and len(windows) > max_samples:
        # Keep task and episode coverage deterministic instead of taking only
        # the first task's windows.
        selected = np.linspace(0, len(windows) - 1, max_samples, dtype=np.int64)
        windows = [windows[int(index)] for index in selected]
    # One start offset, current + H future rows, then task/episode metadata.
    # Keep metadata after the complete trajectory table; H8 must not be
    # overwritten by the metadata columns.
    rows = np.zeros((len(windows), HORIZON + 4), dtype=np.int64)
    for index, (_, start, episode_rows) in enumerate(windows):
        rows[index, 0] = int(start)
        rows[index, 1 : 1 + HORIZON + 1] = episode_rows[
            start
            + np.arange(0, (HORIZON + 1) * RAW_STEPS_PER_TOKEN, RAW_STEPS_PER_TOKEN)
        ]
        rows[index, 2 + HORIZON : 4 + HORIZON] = [
            int(data["task_index"][episode_rows[0]]),
            int(data["episode_id"][episode_rows[0]]),
        ]
    return rows


def _window_indices(
    data: dict[str, np.ndarray], window: np.ndarray
) -> tuple[int, np.ndarray]:
    """Return the current row and eight future rows from a window record."""

    current = int(window[1])
    # The table stores current at slot 1, followed by H future states at
    # slots 2..H+1.  H8 must therefore be t+40, not the current row.
    future = np.asarray(window[2 : 2 + HORIZON], dtype=np.int64)
    return current, future


def _action_chunks(
    actions: np.ndarray, current: int, future: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    future_action = np.stack(
        [
            actions[current + h * RAW_STEPS_PER_TOKEN : current + (h + 1) * RAW_STEPS_PER_TOKEN]
            .reshape(-1)
            for h in range(HORIZON)
        ],
        axis=0,
    )
    history = np.zeros(
        (2, RAW_STEPS_PER_TOKEN * RAW_ACTION_DIM), dtype=np.float32
    )
    available = actions[max(0, current - 2 * RAW_STEPS_PER_TOKEN) : current]
    if available.size:
        flat = available.reshape(-1)
        history.reshape(-1)[-flat.size :] = flat
    return history, future_action


def _rgb_tensor(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    value = torch.from_numpy(np.ascontiguousarray(rgb)).to(device=device)
    value = value.float().permute(0, 3, 1, 2) / 255.0
    return value


def _make_batch(
    data: dict[str, np.ndarray],
    q_corrected: np.ndarray,
    windows: np.ndarray,
    mode: str,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray, np.ndarray]:
    current_rows = []
    future_rows = []
    history_actions = []
    future_actions = []
    for window in windows:
        current, future = _window_indices(data, window)
        current_rows.append(current)
        future_rows.append(future)
        history, future_action = _action_chunks(data["action"], current, future)
        history_actions.append(history)
        future_actions.append(future_action)
    current_rows_np = np.asarray(current_rows, dtype=np.int64)
    future_rows_np = np.asarray(future_rows, dtype=np.int64)
    task_names = np.asarray(data["task_name"])[current_rows_np].astype(str)
    task_indices = np.asarray([TASK_INDEX[name] for name in task_names], dtype=np.int64)

    q_nominal = np.asarray(data["q_nominal"], dtype=np.float32)
    valid_nominal = np.asarray(data["valid_nominal"], dtype=bool)
    q_true = np.asarray(data["q_true"], dtype=np.float32)
    valid_true = np.asarray(data["valid_true"], dtype=bool)
    if mode == "nominal":
        current_q = q_nominal[current_rows_np]
        future_q = q_nominal[future_rows_np]
        current_valid = valid_nominal[current_rows_np]
        future_valid = valid_nominal[future_rows_np]
    elif mode == "fcpg":
        current_q = q_corrected[current_rows_np]
        future_q = q_corrected[future_rows_np]
        current_valid = valid_nominal[current_rows_np]
        future_valid = valid_nominal[future_rows_np]
    elif mode == "oracle":
        current_q = q_true[current_rows_np]
        future_q = q_true[future_rows_np]
        current_valid = valid_true[current_rows_np]
        future_valid = valid_true[future_rows_np]
    else:
        raise ValueError(f"unknown grounding mode: {mode}")

    batch = {
        "current_rgb": _rgb_tensor(data["rgb"][current_rows_np], device),
        "current_proprio": torch.from_numpy(
            np.ascontiguousarray(data["obs"][current_rows_np, :4])
        ).float().to(device),
        "current_q_nominal": torch.from_numpy(
            np.ascontiguousarray(q_nominal[current_rows_np])
        ).float().to(device),
        "current_q_corrected": torch.from_numpy(
            np.ascontiguousarray(current_q)
        ).float().to(device),
        "current_keypoints_xyz": torch.from_numpy(
            np.ascontiguousarray(data["keypoints_xyz"][current_rows_np])
        ).float().to(device),
        "current_valid": torch.from_numpy(
            np.ascontiguousarray(current_valid)
        ).bool().to(device),
        "future_action": torch.from_numpy(
            np.ascontiguousarray(np.stack(future_actions))
        ).float().to(device),
        "history_action": torch.from_numpy(
            np.ascontiguousarray(np.stack(history_actions))
        ).float().to(device),
        "task_index": torch.from_numpy(task_indices).long().to(device),
        "camera_extrinsic_cam_from_world": torch.from_numpy(
            np.ascontiguousarray(data["true_extrinsic"][current_rows_np])
        ).float().to(device),
        "camera_intrinsic": torch.from_numpy(
            np.ascontiguousarray(data["camera_intrinsic"][current_rows_np])
        ).float().to(device),
    }
    return batch, current_rows_np, future_rows_np, task_indices


@torch.inference_mode()
def _predict(
    model: Any,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    current_state = model.encode_observations(
        rgb=batch["current_rgb"][:, None],
        proprio=batch["current_proprio"][:, None],
        q_nominal=batch["current_q_nominal"][:, None],
        q_corrected=batch["current_q_corrected"][:, None],
        keypoints_xyz=batch["current_keypoints_xyz"][:, None],
        valid=batch["current_valid"][:, None],
    )[:, 0]
    predicted_state = model.predict_with_history(
        current_state=current_state,
        history_action=batch["history_action"],
        future_action=batch["future_action"],
        task_index=batch["task_index"],
        batch=batch,
    )
    return current_state, predicted_state


@torch.inference_mode()
def _decode(
    model: Any,
    current_state: torch.Tensor,
    predicted_state: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    return model.decode_3d(
        predicted_state=predicted_state,
        current_state=current_state,
        current_keypoints_xyz=batch["current_keypoints_xyz"],
        future_action=batch["future_action"],
        task_index=batch["task_index"],
        camera_intrinsic=batch["camera_intrinsic"],
        extrinsic_cam_from_world=batch["camera_extrinsic_cam_from_world"],
    ).predicted_world_xyz


def _summarize(
    predicted: np.ndarray,
    target_xyz: np.ndarray,
    target_goal: np.ndarray,
    task_names: np.ndarray,
) -> dict[str, Any]:
    anchor_error = np.linalg.norm(predicted - target_xyz, axis=-1)
    goal_error = np.linalg.norm(predicted[:, :, 2] - target_goal, axis=-1)
    result: dict[str, Any] = {
        "samples": int(predicted.shape[0]),
        "anchor_world_epe_mm": float(anchor_error.mean() * 1000.0),
        "eef_world_epe_mm": float(anchor_error[:, :, 2].mean() * 1000.0),
        "eef_goal_distance_mm": float(goal_error.mean() * 1000.0),
        "h1_eef_goal_distance_mm": float(goal_error[:, 0].mean() * 1000.0),
        "h8_eef_goal_distance_mm": float(goal_error[:, -1].mean() * 1000.0),
        "by_task": {},
    }
    for task in sorted(set(task_names.tolist())):
        mask = task_names == task
        result["by_task"][str(task)] = {
            "samples": int(mask.sum()),
            "eef_world_epe_mm": float(anchor_error[mask, :, 2].mean() * 1000.0),
            "eef_goal_distance_mm": float(goal_error[mask].mean() * 1000.0),
            "h8_eef_goal_distance_mm": float(goal_error[mask, -1].mean() * 1000.0),
        }
    return result


def main() -> None:
    args = _parser().parse_args()
    device = _resolve_device(args.device)
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    dataset_path = Path(args.dataset).expanduser().resolve()
    q_path = Path(args.fcpg_cache).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    data_npz = np.load(dataset_path, allow_pickle=False)
    data = {name: data_npz[name] for name in data_npz.files}
    q_corrected = np.load(q_path, mmap_mode="r")
    if q_corrected.shape != (len(data["rgb"]), ANCHOR_COUNT, 2):
        raise ValueError("FCPG cache does not match the three-anchor dataset")
    window_table = _window_rows(
        data, stride=int(args.window_stride), max_samples=int(args.max_samples)
    )
    if len(window_table) == 0:
        raise ValueError("no Reach/Push windows were found")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = load_history_conditioned_checkpoint(checkpoint).to(device).eval()

    predictions: dict[str, np.ndarray] = {}
    target_xyz_parts = []
    target_goal_parts = []
    task_name_parts = []
    current_row_parts = []
    future_row_parts = []
    current_xyz_parts = []
    current_entity_parts = []
    current_goal_parts = []
    current_goal_velocity_parts = []
    future_action_parts = []
    target_entity_parts = []
    episode_id_parts = []
    task_index_parts = []
    for offset in range(0, len(window_table), int(args.batch_size)):
        window_batch = window_table[offset : offset + int(args.batch_size)]
        current_rows, future_rows = [], []
        for window in window_batch:
            current, future = _window_indices(data, window)
            current_rows.append(current)
            future_rows.append(future)
        current_rows_np = np.asarray(current_rows, dtype=np.int64)
        future_rows_np = np.asarray(future_rows, dtype=np.int64)
        target_xyz_parts.append(np.asarray(data["keypoints_xyz"][future_rows_np], dtype=np.float32))
        target_goal_parts.append(np.asarray(data["moving_goal_position"][future_rows_np], dtype=np.float32))
        task_name_parts.append(np.asarray(data["task_name"][current_rows_np]).astype(str))
        current_row_parts.append(current_rows_np)
        future_row_parts.append(future_rows_np)
        current_xyz_parts.append(
            np.asarray(data["keypoints_xyz"][current_rows_np], dtype=np.float32)
        )
        current_entity_parts.append(
            np.asarray(data["controlled_entity_xyz"][current_rows_np], dtype=np.float32)
        )
        current_goal_parts.append(
            np.asarray(data["moving_goal_position"][current_rows_np], dtype=np.float32)
        )
        current_goal_velocity_parts.append(
            np.asarray(
                data["moving_goal_next_position"][current_rows_np]
                - data["moving_goal_position"][current_rows_np],
                dtype=np.float32,
            )
        )
        future_action_parts.append(
            np.asarray(
                [
                    _action_chunks(data["action"], int(current), future)[1]
                    for current, future in zip(current_rows_np, future_rows_np)
                ],
                dtype=np.float32,
            )
        )
        target_entity_parts.append(
            np.asarray(
                data["controlled_entity_xyz"][future_rows_np], dtype=np.float32
            )
        )
        episode_id_parts.append(
            np.asarray(data["episode_id"][current_rows_np], dtype=np.int64)
        )
        task_index_parts.append(
            np.asarray(data["task_index"][current_rows_np], dtype=np.int64)
        )
        for mode in ("nominal", "fcpg", "oracle"):
            batch, _, _, _ = _make_batch(data, q_corrected, window_batch, mode, device)
            current_state, predicted_state = _predict(model, batch)
            predicted_world = _decode(model, current_state, predicted_state, batch)
            predictions.setdefault(mode, []).append(
                predicted_world.detach().cpu().numpy().astype(np.float32)
            )

    target_xyz = np.concatenate(target_xyz_parts, axis=0)
    target_goal = np.concatenate(target_goal_parts, axis=0)
    task_names = np.concatenate(task_name_parts, axis=0)
    current_rows = np.concatenate(current_row_parts, axis=0)
    future_rows = np.concatenate(future_row_parts, axis=0)
    current_xyz = np.concatenate(current_xyz_parts, axis=0)
    current_entity_xyz = np.concatenate(current_entity_parts, axis=0)
    current_goal = np.concatenate(current_goal_parts, axis=0)
    current_goal_velocity = np.concatenate(current_goal_velocity_parts, axis=0)
    future_action = np.concatenate(future_action_parts, axis=0)
    target_entity_xyz = np.concatenate(target_entity_parts, axis=0)
    episode_ids = np.concatenate(episode_id_parts, axis=0)
    task_indices = np.concatenate(task_index_parts, axis=0)
    predicted_arrays = {mode: np.concatenate(parts, axis=0) for mode, parts in predictions.items()}
    report: dict[str, Any] = {
        "protocol": "moving_goal_lewm_inference_v1",
        "dataset": str(dataset_path),
        "checkpoint": str(checkpoint_path),
        "fcpg_cache": str(q_path),
        "device": str(device),
        "sample_count": int(len(current_rows)),
        "tasks": sorted(set(task_names.tolist())),
        "window_stride_raw": int(args.window_stride),
        "raw_steps_per_token": RAW_STEPS_PER_TOKEN,
        "horizon": HORIZON,
        "prediction_inputs": [
            "current_rgb",
            "current_proprio",
            "current_grounding",
            "two_previous_action_chunks",
            "future_action_prefix",
        ],
        "future_rgb_used_by_predictor": False,
        "modes": {},
    }
    for mode, predicted in predicted_arrays.items():
        report["modes"][mode] = _summarize(
            predicted, target_xyz, target_goal, task_names
        )
    command = []
    hold = []
    for window in window_table:
        current, future = _window_indices(data, window)
        command_current_xyz = torch.from_numpy(
            np.asarray(data["keypoints_xyz"][current], dtype=np.float32)
        )[None]
        _, command_future_action = _action_chunks(data["action"], current, future)
        command.append(
            command_integrated_world_xyz(
                command_current_xyz,
                torch.from_numpy(command_future_action)[None],
            )[0].numpy()
        )
        hold.append(np.repeat(np.asarray(data["keypoints_xyz"][current])[None], HORIZON, axis=0))
    report["modes"]["command_integrated"] = _summarize(
        np.asarray(command), target_xyz, target_goal, task_names
    )
    report["modes"]["hold_last"] = _summarize(
        np.asarray(hold), target_xyz, target_goal, task_names
    )
    report["paired_delta_fcpg_minus_nominal"] = {
        "eef_goal_h8_mm": float(
            report["modes"]["fcpg"]["h8_eef_goal_distance_mm"]
            - report["modes"]["nominal"]["h8_eef_goal_distance_mm"]
        ),
        "eef_world_epe_mm": float(
            report["modes"]["fcpg"]["eef_world_epe_mm"]
            - report["modes"]["nominal"]["eef_world_epe_mm"]
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.save_predictions:
        prediction_path = Path(args.save_predictions).expanduser().resolve()
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            prediction_path,
            current_row=current_rows,
            future_rows=future_rows,
            task_name=task_names,
            episode_id=episode_ids,
            task_index=task_indices,
            current_xyz=current_xyz,
            current_entity_xyz=current_entity_xyz,
            current_goal=current_goal,
            current_goal_velocity=current_goal_velocity,
            future_action=future_action,
            target_xyz=target_xyz,
            target_entity_xyz=target_entity_xyz,
            target_goal=target_goal,
            nominal=predicted_arrays["nominal"],
            fcpg=predicted_arrays["fcpg"],
            oracle=predicted_arrays["oracle"],
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
