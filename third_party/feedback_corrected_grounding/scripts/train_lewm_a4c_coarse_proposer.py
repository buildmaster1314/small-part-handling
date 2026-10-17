#!/usr/bin/env python3
"""Train the bounded seven-task A4c coarse action proposer on CUDA0."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_a4c_coarse_proposer import (  # noqa: E402
    A4CCoarseActionProposer,
    checkpoint_payload,
    weighted_action_loss,
)
from fcpg.lewm_action_branch_a4 import TASKS  # noqa: E402
from fcpg.lewm_action_prefix_joint_data import open_raw_replay_arrays  # noqa: E402
from fcpg.lewm_history_conditioned_metric3d_a4 import (  # noqa: E402
    load_history_conditioned_checkpoint,
)


DEFAULT_CONTROL_INDEX = ROOT / (
    "outputs/lewm_action_prefix_a4/a4c_control_index_v1/index.npz"
)
DEFAULT_RAW_DATASET = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "joint_orbit12_gripper_phaseaware_a4c_v4_48train_24dev"
)
DEFAULT_PARENT = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "camera_action_metric3d_world_history_seed40583_10k/"
    "lewm_history_conditioned_latest.pt"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/lewm_action_prefix_a4/a4c_coarse_proposer_seed40591_2k"
)
FIXED_STEPS = 2_000
INPUT_MODES = ("state_history", "state_only", "history_only")
LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 1.0e-4
WARMUP_STEPS = 100
GRAD_CLIP = 1.0


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--control-index", type=Path, default=DEFAULT_CONTROL_INDEX)
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_RAW_DATASET)
    parser.add_argument("--parent-checkpoint", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=FIXED_STEPS)
    parser.add_argument("--seed", type=int, default=40_591)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-mode", choices=INPUT_MODES, default="state_history")
    parser.add_argument("--state-cache", type=Path)
    parser.add_argument("--camera-conditioned", action="store_true")
    parser.add_argument("--reencode", action="store_true")
    args = parser.parse_args(argv)
    if int(args.steps) != FIXED_STEPS:
        parser.error(f"this bounded pilot requires exactly {FIXED_STEPS} steps")
    if int(args.seed) <= 0:
        parser.error("--seed must be positive")
    if min(
        int(args.batch_size),
        int(args.encode_batch_size),
        int(args.eval_batch_size),
        int(args.log_every),
    ) <= 0:
        parser.error("batch sizes and --log-every must be positive")
    return args


def _cuda0(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("A4c coarse proposer training permits only CUDA0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _validate_control_index(
    control: Mapping[str, np.ndarray],
    raw_index: Mapping[str, np.ndarray],
) -> int:
    required = {
        "raw_row_index",
        "task_index",
        "task_name",
        "split",
        "physical_bundle_index",
        "expert_action_prefix",
        "history_action",
        "sample_weight",
        "camera_clock_hour",
    }
    missing = sorted(required - set(control))
    if missing:
        raise ValueError(f"control index is missing fields: {missing}")
    rows = np.asarray(control["raw_row_index"], dtype=np.int64)
    count = int(rows.size)
    if count <= 0 or rows.shape != (count,):
        raise ValueError("control index is empty")
    raw_count = int(np.asarray(raw_index["task_index"]).shape[0])
    if int(rows.min()) < 0 or int(rows.max()) >= raw_count:
        raise ValueError("control raw_row_index is outside the raw dataset")
    for name in ("task_index", "task_name", "split", "camera_clock_hour"):
        if not np.array_equal(
            np.asarray(control[name]).astype(str),
            np.asarray(raw_index[name])[rows].astype(str),
        ):
            raise ValueError(f"control {name} differs from raw replay")
    if np.asarray(control["expert_action_prefix"]).shape != (count, 8, 20):
        raise ValueError("expert_action_prefix must have shape [N,8,20]")
    if np.asarray(control["history_action"]).shape != (count, 2, 20):
        raise ValueError("history_action must have shape [N,2,20]")
    if set(np.asarray(control["task_name"]).astype(str).tolist()) != set(TASKS):
        raise ValueError("control index does not contain all seven tasks")
    return count


@torch.inference_mode()
def _encode_current_states(
    *,
    parent_checkpoint: Path,
    raw_dataset: Path,
    raw_rows: np.ndarray,
    output_path: Path,
    device: torch.device,
    batch_size: int,
    reencode: bool,
) -> np.ndarray:
    if output_path.exists() and not reencode:
        cached = np.load(output_path, mmap_mode="r")
        if cached.shape != (len(raw_rows), 4, 192) or cached.dtype != np.float16:
            raise ValueError("existing current-state cache has an invalid contract")
        return cached

    payload = torch.load(
        parent_checkpoint.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, Mapping):
        raise TypeError("parent checkpoint must contain a mapping")
    parent = load_history_conditioned_checkpoint(payload).to(device).eval()
    _raw_index, arrays = open_raw_replay_arrays(raw_dataset)
    states = np.empty((len(raw_rows), 4, 192), dtype=np.float16)
    for start in range(0, len(raw_rows), int(batch_size)):
        stop = min(start + int(batch_size), len(raw_rows))
        rows = raw_rows[start:stop]
        batch = {}
        for name in (
            "current_rgb",
            "current_proprio",
            "current_q_nominal",
            "current_q_corrected",
            "current_keypoints_xyz",
            "current_valid",
        ):
            value = torch.from_numpy(
                np.ascontiguousarray(np.asarray(arrays[name])[rows])
            ).to(device)
            batch[name] = value[:, None]
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            encoded = parent.encode_observations(
                rgb=batch["current_rgb"],
                proprio=batch["current_proprio"],
                q_nominal=batch["current_q_nominal"],
                q_corrected=batch["current_q_corrected"],
                keypoints_xyz=batch["current_keypoints_xyz"],
                valid=batch["current_valid"],
            )[:, 0]
        states[start:stop] = encoded.float().cpu().numpy().astype(np.float16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, states)
    temporary.replace(output_path)
    del parent
    torch.cuda.empty_cache()
    return np.load(output_path, mmap_mode="r")


def _different_bundle_permutation(
    task_index: np.ndarray,
    physical_bundle: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    result = np.empty_like(rows)
    for task in range(len(TASKS)):
        positions = np.flatnonzero(task_index[rows] == task)
        task_rows = rows[positions]
        bundles = physical_bundle[task_rows]
        unique = np.unique(bundles)
        if unique.size < 2:
            raise ValueError(f"task {TASKS[task]!r} needs two development bundles")
        successor = {
            int(bundle): int(unique[(offset + 1) % unique.size])
            for offset, bundle in enumerate(unique)
        }
        replacement = []
        for row, bundle in zip(task_rows, bundles, strict=True):
            candidates = task_rows[bundles == successor[int(bundle)]]
            replacement.append(int(candidates[int(row) % len(candidates)]))
        result[positions] = np.asarray(replacement, dtype=np.int64)
    if np.any(physical_bundle[result] == physical_bundle[rows]):
        raise RuntimeError("diagnostic permutation reused a physical bundle")
    return result


def _same_bundle_different_view_permutation(
    physical_bundle: np.ndarray,
    camera_clock: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    result = np.empty_like(rows)
    row_to_position = {int(row): position for position, row in enumerate(rows)}
    for bundle in np.unique(physical_bundle[rows]):
        bundle_rows = rows[physical_bundle[rows] == bundle]
        order = np.argsort(camera_clock[bundle_rows])
        ordered = bundle_rows[order]
        if ordered.size < 2:
            raise ValueError("camera diagnostic requires two views per bundle")
        shifted = np.roll(ordered, -1)
        for source, destination in zip(ordered, shifted, strict=True):
            result[row_to_position[int(source)]] = int(destination)
    if np.any(camera_clock[result] == camera_clock[rows]):
        raise RuntimeError("camera diagnostic reused the factual view")
    return result


def _apply_input_mode(
    state: torch.Tensor,
    history: torch.Tensor,
    input_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if input_mode == "state_history":
        return state, history
    if input_mode == "state_only":
        return state, torch.zeros_like(history)
    if input_mode == "history_only":
        return torch.zeros_like(state), history
    raise ValueError(f"unsupported input mode {input_mode!r}")


def _row_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    squared = (prediction.float() - target.float()).square()
    xyz_columns = [index for index in range(20) if index % 4 != 3]
    return {
        "action_mse": squared.mean(dim=(1, 2)),
        "xyz_mse": squared[:, :, xyz_columns].mean(dim=(1, 2)),
        "gripper_mse": squared[:, :, 3::4].mean(dim=(1, 2)),
        "h1_mse": squared[:, 0].mean(dim=1),
        "h8_mse": squared[:, -1].mean(dim=1),
        "action_mae": squared.sqrt().mean(dim=(1, 2)),
    }


def _aggregate_metrics(
    rows: np.ndarray,
    task_index: np.ndarray,
    metrics: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    per_task: dict[str, dict[str, float]] = {}
    for task, task_name in enumerate(TASKS):
        selected = task_index[rows] == task
        if not bool(selected.any()):
            raise ValueError(f"metrics lack task {task_name!r}")
        per_task[task_name] = {
            name: float(np.asarray(value)[selected].mean())
            for name, value in metrics.items()
        }
    task_macro = {
        name: float(np.mean([per_task[task][name] for task in TASKS]))
        for name in metrics
    }
    return {"task_macro": task_macro, "per_task": per_task}


@torch.inference_mode()
def _evaluate(
    model: A4CCoarseActionProposer,
    current_state: np.ndarray,
    camera_feature: np.ndarray | None,
    control: Mapping[str, np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
    input_mode: str,
) -> dict[str, Any]:
    model.eval()
    split = np.asarray(control["split"]).astype(str)
    rows = np.flatnonzero(split == "development")
    task_index = np.asarray(control["task_index"], dtype=np.int64)
    physical_bundle = np.asarray(control["physical_bundle_index"], dtype=np.int64)
    camera_clock = np.asarray(control["camera_clock_hour"], dtype=np.int64)
    permutation = _different_bundle_permutation(
        task_index, physical_bundle, rows
    )
    camera_permutation = _same_bundle_different_view_permutation(
        physical_bundle, camera_clock, rows
    )
    target = torch.from_numpy(
        np.ascontiguousarray(np.asarray(control["expert_action_prefix"])[rows])
    ).float()
    history = torch.from_numpy(
        np.ascontiguousarray(np.asarray(control["history_action"])[rows])
    ).float()
    baseline = history[:, -1:, :].expand(-1, 8, -1)
    predictions: dict[str, list[torch.Tensor]] = {
        "factual": [],
        "state_shuffled": [],
        "history_shuffled": [],
    }
    if camera_feature is not None:
        predictions["camera_shuffled"] = []
    for start in range(0, len(rows), int(batch_size)):
        stop = min(start + int(batch_size), len(rows))
        batch_rows = rows[start:stop]
        shuffled_rows = permutation[start:stop]
        state = torch.from_numpy(
            np.ascontiguousarray(np.asarray(current_state)[batch_rows])
        ).float().to(device)
        shuffled_state = torch.from_numpy(
            np.ascontiguousarray(np.asarray(current_state)[shuffled_rows])
        ).float().to(device)
        factual_history = torch.from_numpy(
            np.ascontiguousarray(np.asarray(control["history_action"])[batch_rows])
        ).float().to(device)
        shuffled_history = torch.from_numpy(
            np.ascontiguousarray(np.asarray(control["history_action"])[shuffled_rows])
        ).float().to(device)
        task = torch.from_numpy(task_index[batch_rows]).long().to(device)
        camera = (
            torch.from_numpy(
                np.ascontiguousarray(np.asarray(camera_feature)[batch_rows])
            ).float().to(device)
            if camera_feature is not None
            else None
        )
        state_shuffled_camera = (
            torch.from_numpy(
                np.ascontiguousarray(np.asarray(camera_feature)[shuffled_rows])
            ).float().to(device)
            if camera_feature is not None
            else None
        )
        wrong_view_camera = (
            torch.from_numpy(
                np.ascontiguousarray(
                    np.asarray(camera_feature)[camera_permutation[start:stop]]
                )
            ).float().to(device)
            if camera_feature is not None
            else None
        )
        factual_state, factual_history_input = _apply_input_mode(
            state, factual_history, input_mode
        )
        shuffled_state_input, state_history_input = _apply_input_mode(
            shuffled_state, factual_history, input_mode
        )
        history_state_input, shuffled_history_input = _apply_input_mode(
            state, shuffled_history, input_mode
        )
        predictions["factual"].append(
            model(
                factual_state,
                factual_history_input,
                task,
                camera_condition=camera,
            ).cpu()
        )
        predictions["state_shuffled"].append(
            model(
                shuffled_state_input,
                state_history_input,
                task,
                camera_condition=state_shuffled_camera,
            ).cpu()
        )
        predictions["history_shuffled"].append(
            model(
                history_state_input,
                shuffled_history_input,
                task,
                camera_condition=camera,
            ).cpu()
        )
        if camera_feature is not None:
            predictions["camera_shuffled"].append(
                model(
                    factual_state,
                    factual_history_input,
                    task,
                    camera_condition=wrong_view_camera,
                ).cpu()
            )

    variants = {
        name: torch.cat(parts, dim=0) for name, parts in predictions.items()
    }
    variants["repeat_last"] = baseline
    report: dict[str, Any] = {}
    for name, prediction in variants.items():
        row_values = {
            metric: value.numpy()
            for metric, value in _row_metrics(prediction, target).items()
        }
        report[name] = _aggregate_metrics(rows, task_index, row_values)
    baseline_mse = report["repeat_last"]["task_macro"]["action_mse"]
    factual_mse = report["factual"]["task_macro"]["action_mse"]
    report["diagnostics"] = {
        "action_mse_reduction_vs_repeat_last": float(
            1.0 - factual_mse / max(baseline_mse, 1.0e-12)
        ),
        "state_shuffle_mse_increase": float(
            report["state_shuffled"]["task_macro"]["action_mse"] - factual_mse
        ),
        "history_shuffle_mse_increase": float(
            report["history_shuffled"]["task_macro"]["action_mse"] - factual_mse
        ),
        "development_rows": int(len(rows)),
        "development_physical_bundles": int(
            np.unique(physical_bundle[rows]).size
        ),
        "development_views": sorted(
            int(value)
            for value in np.unique(
                np.asarray(control["camera_clock_hour"], dtype=np.int64)[rows]
            )
        ),
        "input_mode": input_mode,
        "camera_conditioned": camera_feature is not None,
    }
    if camera_feature is not None:
        report["diagnostics"]["camera_shuffle_mse_increase"] = float(
            report["camera_shuffled"]["task_macro"]["action_mse"] - factual_mse
        )
    return report


def _learning_rate_lambda(step: int) -> float:
    if step < WARMUP_STEPS:
        return max(float(step + 1) / float(WARMUP_STEPS), 1.0e-3)
    progress = (step - WARMUP_STEPS) / max(FIXED_STEPS - WARMUP_STEPS, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser(argv)
    device = _cuda0(str(args.device))
    _seed_everything(int(args.seed))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_cache_path = (
        args.state_cache.expanduser().resolve()
        if args.state_cache is not None
        else output_dir / "current_state_fp16.npy"
    )

    control = _load_npz(args.control_index)
    raw_index, raw_arrays = open_raw_replay_arrays(args.raw_dataset)
    count = _validate_control_index(control, raw_index)
    raw_rows = np.asarray(control["raw_row_index"], dtype=np.int64)
    camera_feature = None
    if bool(args.camera_conditioned):
        extrinsic = np.asarray(raw_arrays["camera_extrinsic_cam_from_world"])[
            raw_rows, :3, :4
        ]
        camera_feature = np.ascontiguousarray(
            extrinsic.reshape(count, 12), dtype=np.float32
        )
    current_state = _encode_current_states(
        parent_checkpoint=args.parent_checkpoint,
        raw_dataset=args.raw_dataset,
        raw_rows=raw_rows,
        output_path=state_cache_path,
        device=device,
        batch_size=int(args.encode_batch_size),
        reencode=bool(args.reencode),
    )
    if current_state.shape[0] != count:
        raise RuntimeError("current-state cache count drifted")

    model = A4CCoarseActionProposer(
        camera_feature_dim=12 if camera_feature is not None else 0
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count > 1_000_000:
        raise RuntimeError("bounded coarse proposer exceeded one million parameters")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_learning_rate_lambda
    )

    split = np.asarray(control["split"]).astype(str)
    train_rows = np.flatnonzero(split == "train")
    if train_rows.size == 0:
        raise ValueError("control index has no training rows")
    sample_weight = np.asarray(control["sample_weight"], dtype=np.float64)[train_rows]
    probability = sample_weight / sample_weight.sum()
    rng = np.random.default_rng(int(args.seed))
    curve: list[dict[str, Any]] = []

    model.train()
    for step in range(1, FIXED_STEPS + 1):
        selected = rng.choice(
            train_rows,
            size=int(args.batch_size),
            replace=True,
            p=probability,
        )
        state = torch.from_numpy(
            np.ascontiguousarray(np.asarray(current_state)[selected])
        ).float().to(device)
        history = torch.from_numpy(
            np.ascontiguousarray(np.asarray(control["history_action"])[selected])
        ).float().to(device)
        task = torch.from_numpy(
            np.asarray(control["task_index"], dtype=np.int64)[selected]
        ).long().to(device)
        target = torch.from_numpy(
            np.ascontiguousarray(
                np.asarray(control["expert_action_prefix"])[selected]
            )
        ).float().to(device)
        weights = torch.from_numpy(
            np.asarray(control["sample_weight"], dtype=np.float32)[selected]
        ).to(device)
        camera = (
            torch.from_numpy(
                np.ascontiguousarray(np.asarray(camera_feature)[selected])
            ).float().to(device)
            if camera_feature is not None
            else None
        )

        optimizer.zero_grad(set_to_none=True)
        state, history = _apply_input_mode(state, history, str(args.input_mode))
        prediction = model(state, history, task, camera_condition=camera)
        loss, components = weighted_action_loss(prediction, target, weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % int(args.log_every) == 0 or step == FIXED_STEPS:
            evaluation = _evaluate(
                model,
                current_state,
                camera_feature,
                control,
                device=device,
                batch_size=int(args.eval_batch_size),
                input_mode=str(args.input_mode),
            )
            row = {
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                "train_xyz_loss": float(components["xyz_loss"].detach().cpu()),
                "train_gripper_loss": float(
                    components["gripper_loss"].detach().cpu()
                ),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "development_action_mse": evaluation["factual"]["task_macro"][
                    "action_mse"
                ],
                "repeat_last_action_mse": evaluation["repeat_last"]["task_macro"][
                    "action_mse"
                ],
                "action_mse_reduction_vs_repeat_last": evaluation["diagnostics"][
                    "action_mse_reduction_vs_repeat_last"
                ],
            }
            curve.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            model.train()

    final_evaluation = _evaluate(
        model,
        current_state,
        camera_feature,
        control,
        device=device,
        batch_size=int(args.eval_batch_size),
        input_mode=str(args.input_mode),
    )
    payload = checkpoint_payload(
        model,
        step=FIXED_STEPS,
        seed=int(args.seed),
        parent_checkpoint=str(args.parent_checkpoint.expanduser().resolve()),
        control_index=str(args.control_index.expanduser().resolve()),
        input_mode=str(args.input_mode),
    )
    _atomic_torch(output_dir / "coarse_proposer_latest.pt", payload)
    report = {
        "protocol": "lewm_a4c_coarse_proposer_pilot_v1",
        "status": "development_only",
        "interpretation": (
            "Tests whether a shared on-manifold action center generalizes across "
            "seven tasks, task phases, and held-out odd camera views. It is not a "
            "closed-loop control or LeWM-selection claim."
        ),
        "steps": FIXED_STEPS,
        "seed": int(args.seed),
        "parameters": int(parameter_count),
        "input_mode": str(args.input_mode),
        "camera_conditioned": camera_feature is not None,
        "parent_checkpoint": str(args.parent_checkpoint.expanduser().resolve()),
        "control_index": str(args.control_index.expanduser().resolve()),
        "raw_dataset": str(args.raw_dataset.expanduser().resolve()),
        "state_cache": str(state_cache_path),
        "curve": curve,
        "evaluation": final_evaluation,
        "limitations": [
            "expert-centered on-manifold training states",
            "no recovery actions after policy-induced drift",
            "coarse proposal only; LeWM candidate selection remains untested",
        ],
    }
    _atomic_json(output_dir / "metrics.json", report)
    print(json.dumps(final_evaluation["diagnostics"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
