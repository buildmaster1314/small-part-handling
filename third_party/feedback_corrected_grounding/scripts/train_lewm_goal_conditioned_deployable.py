#!/usr/bin/env python3
"""Train the deployable goal-conditioned head on a frozen LeWM backbone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fcpg.geometry import Camera, project_points
from fcpg.lewm_goal_conditioned_deployable import (
    DeployableGoalConditionedMetric3DLeWM,
    deployable_checkpoint_payload,
)
from fcpg.lewm_history_conditioned_metric3d_a4 import (
    load_history_conditioned_checkpoint,
)
from fcpg.transformer_grounding import load_fcpg_vit_adapter
from scripts.train_lewm_goal_conditioned_direct import (
    ANCHOR_COUNT,
    CANDIDATE_COUNT,
    GROUNDINGS,
    HORIZON,
    _encode_states,
    _fcpg_q,
    _load,
    _sample_indices,
    _task_index,
)


PROTOCOL = "goal_conditioned_deployable_training_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "data/metaworld_projection/moving_goal_reach_push_branches_train.npz"
        ),
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
    parser.add_argument(
        "--state-cache",
        type=Path,
        default=Path(
            "outputs/lewm_goal_conditioned_direct_seed93001_2k/current_state_cache.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lewm_goal_conditioned_deployable_seed93011_2k"),
    )
    parser.add_argument("--train-grounding", choices=GROUNDINGS, default="fcpg")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=93011)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=256)
    return parser


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and device.index == 1:
        raise ValueError("CUDA1 is disabled")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    return device


def infer_nominal_extrinsic(
    data: dict[str, np.ndarray],
    *,
    tolerance_px: float = 2.0e-3,
) -> np.ndarray:
    """Recover the dataset's known nominal calibration from a clean row."""

    q_error = np.linalg.norm(
        data["current_q_nominal"] - data["current_q_true"], axis=-1
    ).mean(axis=1)
    clean_index = int(np.argmin(q_error))
    if float(q_error[clean_index]) > float(tolerance_px):
        raise ValueError("dataset has no clean row from which to recover nominal extrinsic")
    nominal = np.asarray(data["true_extrinsic"][clean_index], dtype=np.float64)
    if nominal.shape != (4, 4) or not np.isfinite(nominal).all():
        raise ValueError("recovered nominal extrinsic is invalid")

    height, width = data["current_rgb"].shape[1:3]
    maximum_error = 0.0
    for xyz, intrinsic, expected_q in zip(
        data["current_xyz"], data["camera_intrinsic"], data["current_q_nominal"]
    ):
        q, _valid = project_points(
            xyz,
            Camera(intrinsic, nominal, int(width), int(height)),
        )
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(np.asarray(q) - np.asarray(expected_q)))),
        )
    if maximum_error > float(tolerance_px):
        raise ValueError(
            f"recovered nominal extrinsic does not reproduce q_nominal: {maximum_error:.6f}px"
        )
    return np.asarray(nominal, dtype=np.float32)


def _load_state_cache(
    path: Path,
    *,
    rows: int,
) -> tuple[dict[str, np.ndarray], np.ndarray] | None:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return None
    with np.load(resolved, allow_pickle=False) as archive:
        if not all(name in archive.files for name in (*GROUNDINGS, "q_fcpg")):
            raise ValueError("state cache is missing required arrays")
        states = {name: np.asarray(archive[name], dtype=np.float32) for name in GROUNDINGS}
        q_fcpg = np.asarray(archive["q_fcpg"], dtype=np.float32)
    for name, value in states.items():
        if value.shape != (rows, 4, 192) or not np.isfinite(value).all():
            raise ValueError(f"cached {name} state has an invalid shape or value")
    if q_fcpg.shape != (rows, ANCHOR_COUNT, 2) or not np.isfinite(q_fcpg).all():
        raise ValueError("cached q_fcpg has an invalid shape or value")
    return states, q_fcpg


def _batch(
    data: dict[str, np.ndarray],
    states: dict[str, np.ndarray],
    rows: np.ndarray,
    candidates: np.ndarray,
    grounding: str,
    nominal_extrinsic: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    task_names = data["task_name"].astype(str)[rows]
    extrinsic_source = np.asarray(nominal_extrinsic, dtype=np.float32)
    if extrinsic_source.shape == (4, 4):
        extrinsic = np.repeat(extrinsic_source[None], len(rows), axis=0)
    elif extrinsic_source.shape == (len(data["task_name"]), 4, 4):
        extrinsic = extrinsic_source[rows]
    else:
        raise ValueError(
            "nominal_extrinsic must have shape [4,4] or [dataset_rows,4,4]"
        )
    return {
        "current_state": torch.from_numpy(
            np.ascontiguousarray(states[grounding][rows])
        ).float().to(device),
        "history_action": torch.from_numpy(
            np.ascontiguousarray(data["history_action"][rows])
        ).float().to(device),
        "future_action": torch.from_numpy(
            np.ascontiguousarray(data["candidate_action"][rows, candidates])
        ).float().to(device),
        "current_goal_xyz": torch.from_numpy(
            np.ascontiguousarray(data["current_goal"][rows])
        ).float().to(device),
        "current_goal_velocity": torch.from_numpy(
            np.ascontiguousarray(data["current_goal_velocity"][rows])
        ).float().to(device),
        "task_index": _task_index(task_names, device),
        "target_xyz": torch.from_numpy(
            np.ascontiguousarray(data["target_xyz"][rows, candidates])
        ).float().to(device),
        "target_entity_xyz": torch.from_numpy(
            np.ascontiguousarray(data["target_entity_xyz"][rows, candidates])
        ).float().to(device),
        "target_goal_xyz": torch.from_numpy(
            np.ascontiguousarray(data["target_goal"][rows, candidates])
        ).float().to(device),
        "batch": {
            "current_keypoints_xyz": torch.from_numpy(
                np.ascontiguousarray(data["current_xyz"][rows])
            ).float().to(device),
            "camera_intrinsic": torch.from_numpy(
                np.ascontiguousarray(data["camera_intrinsic"][rows])
            ).float().to(device),
            "camera_extrinsic_cam_from_world": torch.from_numpy(
                np.ascontiguousarray(extrinsic)
            ).float().to(device),
        },
    }


def _average_rank(value: np.ndarray) -> np.ndarray:
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


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = _average_rank(left)
    right_rank = _average_rank(right)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = float(np.linalg.norm(left_rank) * np.linalg.norm(right_rank))
    return float(np.dot(left_rank, right_rank) / denominator) if denominator else 0.0


def _metric_block(
    predicted_robot: np.ndarray,
    predicted_entity: np.ndarray,
    target_robot: np.ndarray,
    target_entity: np.ndarray,
    target_goal: np.ndarray,
) -> dict[str, Any]:
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
    row_index = np.arange(len(predicted_robot))
    return {
        "rows": int(len(predicted_robot)),
        "branches": int(len(predicted_robot) * CANDIDATE_COUNT),
        "robot_anchor_epe_mm": float(robot_error.mean() * 1000.0),
        "entity_epe_mm": float(entity_error.mean() * 1000.0),
        "top1_action_accuracy": float(np.mean(predicted_winner == actual_winner)),
        "spearman_mean": float(
            np.mean(
                [
                    _spearman(left, right)
                    for left, right in zip(predicted_cost, actual_cost)
                ]
            )
        ),
        "selected_actual_goal_distance_h8_mm": float(
            actual_cost[row_index, predicted_winner].mean() * 1000.0
        ),
        "oracle_actual_goal_distance_h8_mm": float(
            actual_cost[row_index, actual_winner].mean() * 1000.0
        ),
    }


@torch.inference_mode()
def evaluate_model(
    model: DeployableGoalConditionedMetric3DLeWM,
    data: dict[str, np.ndarray],
    states: dict[str, np.ndarray],
    rows: np.ndarray,
    grounding: str,
    nominal_extrinsic: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    robot_by_candidate: list[np.ndarray] = []
    entity_by_candidate: list[np.ndarray] = []
    for candidate in range(CANDIDATE_COUNT):
        robot_parts: list[np.ndarray] = []
        entity_parts: list[np.ndarray] = []
        for start in range(0, len(rows), int(batch_size)):
            block_rows = rows[start : start + int(batch_size)]
            block_candidates = np.full(len(block_rows), candidate, dtype=np.int64)
            values = _batch(
                data,
                states,
                block_rows,
                block_candidates,
                grounding,
                nominal_extrinsic,
                device,
            )
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
    report = _metric_block(
        predicted_robot,
        predicted_entity,
        target_robot,
        target_entity,
        target_goal,
    )
    task_names = data["task_name"].astype(str)[rows]
    report["by_task"] = {}
    for task in sorted(set(task_names.tolist())):
        mask = task_names == task
        report["by_task"][task] = _metric_block(
            predicted_robot[mask],
            predicted_entity[mask],
            target_robot[mask],
            target_entity[mask],
            target_goal[mask],
        )
    return report


def main() -> int:
    args = _parser().parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.eval_interval <= 0:
        raise ValueError("steps, batch-size, and eval-interval must be positive")
    device = _device(args.device)
    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    data = _load(args.dataset)
    nominal_extrinsic = infer_nominal_extrinsic(data)
    parent_payload = torch.load(
        args.parent_checkpoint.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    parent = load_history_conditioned_checkpoint(parent_payload).to(device).eval()

    cached = _load_state_cache(args.state_cache, rows=len(data["task_name"]))
    if cached is None:
        corrector, _ = load_fcpg_vit_adapter(
            args.fcpg_checkpoint.expanduser().resolve(), device=device
        )
        corrector.eval()
        q_fcpg = _fcpg_q(corrector, data, device)
        states = _encode_states(parent, data, q_fcpg, device)
    else:
        states, q_fcpg = cached

    model = DeployableGoalConditionedMetric3DLeWM(
        parent,
        hidden_dim=int(args.hidden_dim),
    ).to(device)
    train_rows = np.flatnonzero(data["split"].astype(str) == "train")
    dev_rows = np.flatnonzero(data["split"].astype(str) == "development")
    if not len(train_rows) or not len(dev_rows):
        raise ValueError("dataset must contain train and development rows")
    optimizer = torch.optim.AdamW(
        model.trainable_parameters,
        lr=3.0e-4,
        weight_decay=1.0e-4,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []

    def evaluate(step: int) -> None:
        metrics: dict[str, Any] = {"step": int(step)}
        for grounding in GROUNDINGS:
            metrics[grounding] = evaluate_model(
                model,
                data,
                states,
                dev_rows,
                grounding,
                nominal_extrinsic,
                device,
                int(args.batch_size),
            )
        history.append(metrics)
        print(json.dumps(metrics), flush=True)

    evaluate(0)
    model.train()
    for step in range(1, int(args.steps) + 1):
        rows, candidates = _sample_indices(data, train_rows, rng, args.batch_size)
        values = _batch(
            data,
            states,
            rows,
            candidates,
            args.train_grounding,
            nominal_extrinsic,
            device,
        )
        output = model(**values)
        if output.loss is None:
            raise RuntimeError("deployable model did not produce a training loss")
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters, 1.0)
        optimizer.step()
        if step % int(args.eval_interval) == 0 or step == int(args.steps):
            evaluate(step)
            model.train()

    payload = deployable_checkpoint_payload(
        model,
        step=int(args.steps),
        seed=int(args.seed),
        parent_checkpoint=str(args.parent_checkpoint.expanduser().resolve()),
        dataset=str(args.dataset.expanduser().resolve()),
        train_grounding=args.train_grounding,
        metrics=history[-1],
    )
    payload["training_protocol"] = PROTOCOL
    payload["nominal_extrinsic_cam_from_world"] = nominal_extrinsic
    torch.save(payload, output_dir / "goal_conditioned_deployable_latest.pt")
    (output_dir / "metrics.json").write_text(
        json.dumps({"history": history}, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "current_state_cache.npz",
        **states,
        q_fcpg=q_fcpg,
        nominal_extrinsic=nominal_extrinsic,
    )
    print(json.dumps({"output_dir": str(output_dir), "final": history[-1]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
