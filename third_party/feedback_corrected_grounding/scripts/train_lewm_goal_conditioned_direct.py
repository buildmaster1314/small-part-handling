#!/usr/bin/env python3
"""Train the direct goal/entity-conditioned LeWM development model.

The input is a same-state action-branch store.  The current image is encoded
once with nominal, FCPG, or oracle grounding; the model then predicts future
robot anchors and the task-controlled entity from the candidate action prefix.
No future image, future goal, or simulator state is used as an input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fcpg.lewm_goal_conditioned_metric3d import (
    GoalConditionedMetric3DLeWM,
    direct_checkpoint_payload,
)
from fcpg.lewm_history_conditioned_metric3d_a4 import (
    load_history_conditioned_checkpoint,
)
from fcpg.transformer_grounding import load_fcpg_vit_adapter


HORIZON = 8
CANDIDATE_COUNT = 8
ANCHOR_COUNT = 3
TASKS = ("pick-place-v3", "drawer-retrieve-v0", "reach-v3", "push-v3", "door-open-v3", "window-open-v3", "towel-wipe-v0")
GROUNDINGS = ("nominal", "fcpg", "oracle")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/metaworld_projection/moving_goal_reach_push_branches_train.npz"),
    )
    parser.add_argument(
        "--parent-checkpoint",
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lewm_goal_conditioned_direct_seed93001_2k"))
    parser.add_argument("--train-grounding", choices=GROUNDINGS, default="fcpg")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=93001)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--save-predictions", action="store_true")
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
        "episode_id", "split",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"dataset is missing {missing}")
    n = len(data["task_name"])
    expected = {
        "current_rgb": (n, 224, 224, 3),
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
    if set(tasks.tolist()) - {"reach-v3", "push-v3"}:
        raise ValueError("direct pilot currently supports reach-v3 and push-v3 only")
    if len(np.unique(data["episode_id"])) < 2:
        raise ValueError("need multiple physical episodes")
    return data


def _rgb(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value)).float().permute(0, 3, 1, 2).to(device) / 255.0


@torch.inference_mode()
def _fcpg_q(
    corrector: Any,
    data: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    output: list[np.ndarray] = []
    for start in range(0, len(data["task_name"]), int(batch_size)):
        end = min(len(data["task_name"]), start + int(batch_size))
        rgb = _rgb(data["current_rgb"][start:end], device)
        q = torch.from_numpy(np.ascontiguousarray(data["current_q_nominal"][start:end])).float().to(device)
        xyz = torch.from_numpy(np.ascontiguousarray(data["current_xyz"][start:end])).float().to(device)
        proprio = torch.from_numpy(np.ascontiguousarray(data["current_obs"][start:end, :4])).float().to(device)
        values = corrector(rgb, q, xyz, proprio).q_corrected_px.float().cpu().numpy()
        if values.shape != (end - start, 3, 2) or not np.isfinite(values).all():
            raise RuntimeError("FCPG returned invalid q_corrected")
        output.append(values)
    return np.concatenate(output, axis=0)


@torch.inference_mode()
def _encode_states(
    parent: Any,
    data: dict[str, np.ndarray],
    q_fcpg: np.ndarray,
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, np.ndarray]:
    parent.eval()
    result: dict[str, list[np.ndarray]] = {name: [] for name in GROUNDINGS}
    n = len(data["task_name"])
    for start in range(0, n, int(batch_size)):
        end = min(n, start + int(batch_size))
        rgb = _rgb(data["current_rgb"][start:end], device)[:, None]
        proprio = torch.from_numpy(np.ascontiguousarray(data["current_obs"][start:end, :4])).float().to(device)[:, None]
        xyz = torch.from_numpy(np.ascontiguousarray(data["current_xyz"][start:end])).float().to(device)[:, None]
        q_nominal = torch.from_numpy(np.ascontiguousarray(data["current_q_nominal"][start:end])).float().to(device)[:, None]
        q_fcpg_t = torch.from_numpy(np.ascontiguousarray(q_fcpg[start:end])).float().to(device)[:, None]
        q_true = torch.from_numpy(np.ascontiguousarray(data["current_q_true"][start:end])).float().to(device)[:, None]
        valid_nominal = torch.from_numpy(np.ascontiguousarray(data["current_valid_nominal"][start:end])).bool().to(device)[:, None]
        valid_true = torch.from_numpy(np.ascontiguousarray(data["current_valid_true"][start:end])).bool().to(device)[:, None]
        valid_fcpg = valid_nominal & (q_fcpg_t[..., 0] >= 0.0) & (q_fcpg_t[..., 1] >= 0.0) & (q_fcpg_t[..., 0] < 224.0) & (q_fcpg_t[..., 1] < 224.0)
        for name, q, valid in (
            ("nominal", q_nominal, valid_nominal),
            ("fcpg", q_fcpg_t, valid_fcpg),
            ("oracle", q_true, valid_true),
        ):
            state = parent.encode_observations(
                rgb=rgb,
                proprio=proprio,
                q_nominal=q_nominal,
                q_corrected=q,
                keypoints_xyz=xyz,
                valid=valid,
            )[:, 0]
            result[name].append(state.cpu().numpy())
    return {name: np.concatenate(parts, axis=0) for name, parts in result.items()}


def _task_index(values: np.ndarray, device: torch.device) -> torch.Tensor:
    mapping = {"reach-v3": 2, "push-v3": 3}
    return torch.from_numpy(np.asarray([mapping[str(value)] for value in values], dtype=np.int64)).to(device)


def _sample_indices(
    data: dict[str, np.ndarray],
    rows: np.ndarray,
    rng: np.random.Generator,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    task_names = data["task_name"].astype(str)
    available_tasks = sorted(set(task_names[rows].tolist()))
    selected: list[int] = []
    for _ in range(int(batch_size)):
        task = str(rng.choice(available_tasks))
        task_rows = rows[task_names[rows] == task]
        selected.append(int(rng.choice(task_rows)))
    selected_rows = np.asarray(selected, dtype=np.int64)
    selected_candidates = rng.integers(0, CANDIDATE_COUNT, size=int(batch_size))
    return selected_rows.astype(np.int64), selected_candidates.astype(np.int64)


def _batch(
    data: dict[str, np.ndarray],
    states: dict[str, np.ndarray],
    rows: np.ndarray,
    candidates: np.ndarray,
    grounding: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    task_names = data["task_name"].astype(str)[rows]
    ext = torch.from_numpy(np.ascontiguousarray(data["true_extrinsic"][rows])).float().to(device)
    return {
        "current_state": torch.from_numpy(np.ascontiguousarray(states[grounding][rows])).float().to(device),
        "history_action": torch.from_numpy(np.ascontiguousarray(data["history_action"][rows])).float().to(device),
        "future_action": torch.from_numpy(np.ascontiguousarray(data["candidate_action"][rows, candidates])).float().to(device),
        "current_entity_xyz": torch.from_numpy(np.ascontiguousarray(data["current_entity_xyz"][rows])).float().to(device),
        "current_goal_xyz": torch.from_numpy(np.ascontiguousarray(data["current_goal"][rows])).float().to(device),
        "current_goal_velocity": torch.from_numpy(np.ascontiguousarray(data["current_goal_velocity"][rows])).float().to(device),
        "task_index": _task_index(task_names, device),
        "target_xyz": torch.from_numpy(np.ascontiguousarray(data["target_xyz"][rows, candidates])).float().to(device),
        "target_entity_xyz": torch.from_numpy(np.ascontiguousarray(data["target_entity_xyz"][rows, candidates])).float().to(device),
        "target_goal_xyz": torch.from_numpy(np.ascontiguousarray(data["target_goal"][rows, candidates])).float().to(device),
        "batch": {
            "current_keypoints_xyz": torch.from_numpy(np.ascontiguousarray(data["current_xyz"][rows])).float().to(device),
            "camera_intrinsic": torch.from_numpy(np.ascontiguousarray(data["camera_intrinsic"][rows])).float().to(device),
            "camera_extrinsic_cam_from_world": ext,
        },
    }


@torch.inference_mode()
def _evaluate(
    model: GoalConditionedMetric3DLeWM,
    data: dict[str, np.ndarray],
    states: dict[str, np.ndarray],
    rows: np.ndarray,
    grounding: str,
    device: torch.device,
    batch_size: int,
    save: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    model.eval()
    robot_by_candidate: list[np.ndarray] = []
    entity_by_candidate: list[np.ndarray] = []
    for candidate in range(CANDIDATE_COUNT):
        robot_parts: list[np.ndarray] = []
        entity_parts: list[np.ndarray] = []
        for start in range(0, len(rows), int(batch_size)):
            block_rows = rows[start : start + int(batch_size)]
            block_candidates = np.full(len(block_rows), candidate, dtype=np.int64)
            values = _batch(data, states, block_rows, block_candidates, grounding, device)
            output = model(**values)
            robot_parts.append(output.predicted_world_xyz.cpu().numpy())
            entity_parts.append(output.predicted_entity_xyz.cpu().numpy())
        robot_by_candidate.append(np.concatenate(robot_parts, axis=0))
        entity_by_candidate.append(np.concatenate(entity_parts, axis=0))
    predicted_robot = np.stack(robot_by_candidate, axis=1)
    predicted_entity = np.stack(entity_by_candidate, axis=1)
    target_robot = data["target_xyz"][rows]
    target_entity = data["target_entity_xyz"][rows]
    target_goal = data["target_goal"][rows]
    robot_error = np.linalg.norm(predicted_robot - target_robot, axis=-1)
    entity_error = np.linalg.norm(predicted_entity - target_entity, axis=-1)
    predicted_cost = np.linalg.norm(
        predicted_entity[:, :, -1] - target_goal[:, :, -1], axis=-1
    )
    actual_cost = np.linalg.norm(
        target_entity[:, :, -1] - target_goal[:, :, -1], axis=-1
    )
    predicted_winner = np.argmin(predicted_cost, axis=1)
    actual_winner = np.argmin(actual_cost, axis=1)
    row_index = np.arange(len(rows))

    def average_rank(value: np.ndarray) -> np.ndarray:
        order = np.argsort(value, kind="mergesort")
        sorted_value = value[order]
        result = np.empty(len(value), dtype=np.float64)
        start = 0
        while start < len(value):
            end = start + 1
            while end < len(value) and np.isclose(
                sorted_value[end], sorted_value[start], rtol=1.0e-7, atol=1.0e-9
            ):
                end += 1
            result[order[start:end]] = 0.5 * float(start + end - 1)
            start = end
        return result

    def spearman(left: np.ndarray, right: np.ndarray) -> float:
        left_rank = average_rank(left)
        right_rank = average_rank(right)
        left_rank -= left_rank.mean()
        right_rank -= right_rank.mean()
        denominator = float(np.linalg.norm(left_rank) * np.linalg.norm(right_rank))
        return float(np.dot(left_rank, right_rank) / denominator) if denominator else 0.0

    report = {
        "rows": int(len(rows)),
        "branches": int(len(rows) * CANDIDATE_COUNT),
        "robot_anchor_epe_mm": float(robot_error.mean() * 1000.0),
        "entity_epe_mm": float(entity_error.mean() * 1000.0),
        "predicted_goal_distance_h8_mm": float(predicted_cost.mean() * 1000.0),
        "actual_goal_distance_h8_mm": float(actual_cost.mean() * 1000.0),
        "top1_action_accuracy": float(np.mean(predicted_winner == actual_winner)),
        "spearman_mean": float(
            np.mean([spearman(left, right) for left, right in zip(predicted_cost, actual_cost)])
        ),
        "selected_actual_goal_distance_h8_mm": float(
            actual_cost[row_index, predicted_winner].mean() * 1000.0
        ),
        "oracle_actual_goal_distance_h8_mm": float(
            actual_cost[row_index, actual_winner].mean() * 1000.0
        ),
    }
    saved = None
    if save:
        saved = {"robot": predicted_robot, "entity": predicted_entity}
    return report, saved


def main() -> int:
    args = _parser().parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch-size must be positive")
    device = _device(args.device)
    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    data = _load(args.dataset)
    parent_checkpoint = torch.load(args.parent_checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False)
    parent = load_history_conditioned_checkpoint(parent_checkpoint).to(device).eval()
    corrector, _ = load_fcpg_vit_adapter(args.fcpg_checkpoint.expanduser().resolve(), device=device)
    corrector.eval()
    q_fcpg = _fcpg_q(corrector, data, device)
    states = _encode_states(parent, data, q_fcpg, device)
    model = GoalConditionedMetric3DLeWM(parent, hidden_dim=int(args.hidden_dim)).to(device)
    train_rows = np.flatnonzero(data["split"].astype(str) == "train")
    dev_rows = np.flatnonzero(data["split"].astype(str) == "development")
    if not len(train_rows) or not len(dev_rows):
        raise ValueError("dataset must contain train and development episode rows")
    optimizer = torch.optim.AdamW(model.trainable_parameters, lr=3.0e-4, weight_decay=1.0e-4)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []

    def evaluate(step: int) -> None:
        metrics: dict[str, Any] = {"step": int(step)}
        for grounding in GROUNDINGS:
            metrics[grounding], _ = _evaluate(model, data, states, dev_rows, grounding, device, args.batch_size)
        history.append(metrics)
        print(json.dumps(metrics), flush=True)

    evaluate(0)
    model.train()
    for step in range(1, int(args.steps) + 1):
        rows, candidates = _sample_indices(data, train_rows, rng, args.batch_size)
        values = _batch(data, states, rows, candidates, args.train_grounding, device)
        output = model(**values)
        if output.loss is None:
            raise RuntimeError("direct model did not produce a training loss")
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters, 1.0)
        optimizer.step()
        if step % int(args.eval_interval) == 0 or step == int(args.steps):
            evaluate(step)
            model.train()
    final_metrics = history[-1]
    payload = direct_checkpoint_payload(
        model,
        step=int(args.steps),
        seed=int(args.seed),
        parent_checkpoint=str(args.parent_checkpoint.expanduser().resolve()),
        dataset=str(args.dataset.expanduser().resolve()),
        train_grounding=args.train_grounding,
        metrics=final_metrics,
    )
    torch.save(payload, output_dir / "goal_conditioned_direct_latest.pt")
    (output_dir / "metrics.json").write_text(json.dumps({"history": history}, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output_dir / "current_state_cache.npz", **states, q_fcpg=q_fcpg)
    print(json.dumps({"output_dir": str(output_dir), "final": final_metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
