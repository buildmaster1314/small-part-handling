#!/usr/bin/env python3
"""Adapt the frozen Stage-I FCPG head to clean/drift A4 multiview frames."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.transformer_grounding import load_fcpg_vit_adapter  # noqa: E402


DEFAULT_DATA = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "joint_orbit12_gripper_phaseaware_a4c_v4_48train_24dev"
)
DEFAULT_CHECKPOINT = ROOT / (
    "outputs/transformer_residual/lewm_stage1/shared_4task_seed0_sig0/"
    "fcpg_transformer_residual_best.pt"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/transformer_residual/a4_multiview_projection_seed40601_2k"
)
IMAGE_SIZE = (224, 224)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--clean-fraction", type=float, default=0.25)
    parser.add_argument("--translation-cm-max", type=float, default=4.0)
    parser.add_argument("--rotation-deg-max", type=float, default=5.0)
    parser.add_argument("--minimum-drift-epe-px", type=float, default=1.0)
    parser.add_argument("--maximum-drift-axis-px", type=float, default=19.0)
    parser.add_argument("--seed", type=int, default=40601)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _cuda0(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("A4 FCPG adaptation permits only CUDA device 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def _axis_angle_matrix(axis: np.ndarray, degrees: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = vector / norm
    skew = np.asarray(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64
    )
    radians = math.radians(float(degrees))
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(radians) * skew
        + (1.0 - math.cos(radians)) * (skew @ skew)
    )


def _project(
    xyz: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    camera = np.einsum("ij,kj->ki", extrinsic[:3, :3], xyz) + extrinsic[:3, 3]
    uvw = np.einsum("ij,kj->ki", intrinsic, camera)
    q = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1.0e-8)
    valid = (
        (camera[:, 2] > 1.0e-8)
        & (q[:, 0] >= 0.0)
        & (q[:, 0] < IMAGE_SIZE[0])
        & (q[:, 1] >= 0.0)
        & (q[:, 1] < IMAGE_SIZE[1])
    )
    return q.astype(np.float32), valid


def _perturbed_extrinsic(
    true_extrinsic: np.ndarray,
    translation_world: np.ndarray,
    rotation_axis_world: np.ndarray,
    rotation_degrees: float,
) -> np.ndarray:
    true = np.asarray(true_extrinsic, dtype=np.float64)
    rotation = true[:3, :3]
    center = -rotation.T @ true[:3, 3]
    delta = _axis_angle_matrix(rotation_axis_world, rotation_degrees)
    nominal_rotation = rotation @ delta.T
    nominal_center = center + np.asarray(translation_world, dtype=np.float64)
    nominal = np.eye(4, dtype=np.float64)
    nominal[:3, :3] = nominal_rotation
    nominal[:3, 3] = -nominal_rotation @ nominal_center
    return nominal


def _sample_direction(rng: np.random.Generator) -> np.ndarray:
    value = rng.normal(size=3)
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-12:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    return value / norm


def _sample_drift_nominal(
    xyz: np.ndarray,
    truth: np.ndarray,
    intrinsic: np.ndarray,
    true_extrinsic: np.ndarray,
    *,
    rng: np.random.Generator,
    translation_cm_max: float,
    rotation_deg_max: float,
    minimum_epe_px: float,
    maximum_axis_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(64):
        mode = int(rng.integers(0, 3))
        translation = np.zeros(3, dtype=np.float64)
        degrees = 0.0
        axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        if mode in (0, 2) and translation_cm_max > 0.0:
            magnitude = float(rng.uniform(0.25, translation_cm_max)) / 100.0
            translation = _sample_direction(rng) * magnitude
        if mode in (1, 2) and rotation_deg_max > 0.0:
            degrees = float(rng.uniform(0.25, rotation_deg_max))
            axis = _sample_direction(rng)
        nominal_extrinsic = _perturbed_extrinsic(
            true_extrinsic, translation, axis, degrees
        )
        nominal, valid = _project(xyz, intrinsic, nominal_extrinsic)
        residual = np.asarray(truth, dtype=np.float32) - nominal
        if (
            bool(valid.all())
            and float(np.linalg.norm(residual, axis=-1).mean()) >= minimum_epe_px
            and float(np.abs(residual).max()) <= maximum_axis_px
        ):
            return nominal, valid
    raise RuntimeError("failed to sample a recoverable A4 camera perturbation")


def _fixed_nominals(
    indices: np.ndarray,
    arrays: dict[str, np.ndarray],
    *,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    nominal = np.empty((indices.size, 3, 2), dtype=np.float32)
    valid = np.empty((indices.size, 3), dtype=bool)
    for output_index, source_index in enumerate(indices.tolist()):
        nominal[output_index], valid[output_index] = _sample_drift_nominal(
            arrays["xyz"][source_index],
            arrays["truth"][source_index],
            arrays["intrinsic"][source_index],
            arrays["extrinsic"][source_index],
            rng=rng,
            translation_cm_max=float(args.translation_cm_max),
            rotation_deg_max=float(args.rotation_deg_max),
            minimum_epe_px=float(args.minimum_drift_epe_px),
            maximum_axis_px=float(args.maximum_drift_axis_px),
        )
    return nominal, valid


@torch.inference_mode()
def _cache_backbone_features(
    model: torch.nn.Module,
    rgb: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    count = int(rgb.shape[0])
    cached_features: np.ndarray | None = None
    cached_prefix: np.ndarray | None = None
    model.vision.backbone.eval()
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        pixels = torch.from_numpy(
            np.array(rgb[start:stop], dtype=np.float32, copy=True)
        ).to(device)
        pixels.div_(255.0)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            features, prefix = model.vision.extract_backbone_features(pixels)
        feature_values = features.float().cpu().numpy().astype(np.float16)
        prefix_values = prefix.float().cpu().numpy().astype(np.float16)
        if cached_features is None:
            cached_features = np.empty(
                (count, *feature_values.shape[1:]), dtype=np.float16
            )
            cached_prefix = np.empty(
                (count, *prefix_values.shape[1:]), dtype=np.float16
            )
        cached_features[start:stop] = feature_values
        assert cached_prefix is not None
        cached_prefix[start:stop] = prefix_values
    if cached_features is None or cached_prefix is None:
        raise ValueError("cannot cache an empty A4 dataset")
    return cached_features, cached_prefix


def _metrics(
    prediction: np.ndarray,
    nominal: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    before = np.linalg.norm(nominal - truth, axis=-1)[valid]
    after = np.linalg.norm(prediction - truth, axis=-1)[valid]
    correction = np.linalg.norm(prediction - nominal, axis=-1)[valid]
    return {
        "anchors": int(valid.sum()),
        "before_epe_px": float(before.mean()),
        "after_epe_px": float(after.mean()),
        "correction_epe_px": float(correction.mean()),
        "ratio": float(after.mean() / max(before.mean(), 1.0e-12)),
    }


@torch.inference_mode()
def _evaluate(
    model: torch.nn.Module,
    backbone_features: np.ndarray,
    backbone_prefix: np.ndarray,
    indices: np.ndarray,
    arrays: dict[str, np.ndarray],
    drift_nominal: np.ndarray,
    drift_valid: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    truth = np.asarray(arrays["truth"][indices], dtype=np.float32)
    clean_predictions: list[np.ndarray] = []
    drift_predictions: list[np.ndarray] = []
    for start in range(0, indices.size, batch_size):
        stop = min(start + batch_size, indices.size)
        source = indices[start:stop]
        features = torch.from_numpy(
            np.array(backbone_features[source], dtype=np.float16, copy=True)
        ).to(device)
        prefix = torch.from_numpy(
            np.array(backbone_prefix[source], dtype=np.float16, copy=True)
        ).to(device)
        xyz = torch.from_numpy(
            np.array(arrays["xyz"][source], dtype=np.float32, copy=True)
        ).to(device)
        proprio = torch.from_numpy(
            np.array(arrays["proprio"][source], dtype=np.float32, copy=True)
        ).to(device)
        for nominal_source, predictions in (
            (truth[start:stop], clean_predictions),
            (drift_nominal[start:stop], drift_predictions),
        ):
            nominal = torch.from_numpy(
                np.array(nominal_source, dtype=np.float32, copy=True)
            ).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model.forward_from_backbone_features(
                    features,
                    prefix,
                    nominal,
                    xyz,
                    proprio,
                    image_size=IMAGE_SIZE,
                )
            predictions.append(output.q_corrected_px.float().cpu().numpy())
    clean_prediction = np.concatenate(clean_predictions, axis=0)
    drift_prediction = np.concatenate(drift_predictions, axis=0)
    clean_valid = np.ones(truth.shape[:2], dtype=bool)
    task = arrays["task"][indices]
    by_task: dict[str, dict[str, Any]] = {}
    for name in np.unique(task):
        mask = task == name
        by_task[str(name)] = {
            "clean": _metrics(
                clean_prediction[mask], truth[mask], truth[mask], clean_valid[mask]
            ),
            "drift": _metrics(
                drift_prediction[mask],
                drift_nominal[mask],
                truth[mask],
                drift_valid[mask],
            ),
        }
    return {
        "clean": _metrics(clean_prediction, truth, truth, clean_valid),
        "drift": _metrics(
            drift_prediction, drift_nominal, truth, drift_valid
        ),
        "by_task": by_task,
    }


def _training_batch(
    indices: np.ndarray,
    arrays: dict[str, np.ndarray],
    backbone_features: np.ndarray,
    backbone_prefix: np.ndarray,
    *,
    args: argparse.Namespace,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    source = rng.choice(indices, size=int(args.batch_size), replace=True)
    truth = np.array(arrays["truth"][source], dtype=np.float32, copy=True)
    nominal = truth.copy()
    valid = np.ones(truth.shape[:2], dtype=bool)
    drift_count = int(round(source.size * (1.0 - float(args.clean_fraction))))
    for item in range(drift_count):
        nominal[item], valid[item] = _sample_drift_nominal(
            arrays["xyz"][source[item]],
            truth[item],
            arrays["intrinsic"][source[item]],
            arrays["extrinsic"][source[item]],
            rng=rng,
            translation_cm_max=float(args.translation_cm_max),
            rotation_deg_max=float(args.rotation_deg_max),
            minimum_epe_px=float(args.minimum_drift_epe_px),
            maximum_axis_px=float(args.maximum_drift_axis_px),
        )
    permutation = rng.permutation(source.size)
    source = source[permutation]
    truth = truth[permutation]
    nominal = nominal[permutation]
    valid = valid[permutation]
    return (
        torch.from_numpy(
            np.array(backbone_features[source], dtype=np.float16, copy=True)
        ).to(device),
        torch.from_numpy(
            np.array(backbone_prefix[source], dtype=np.float16, copy=True)
        ).to(device),
        torch.from_numpy(nominal).to(device),
        torch.from_numpy(
            np.array(arrays["xyz"][source], dtype=np.float32, copy=True)
        ).to(device),
        torch.from_numpy(
            np.array(arrays["proprio"][source], dtype=np.float32, copy=True)
        ).to(device),
        torch.from_numpy(truth - nominal).to(device),
        torch.from_numpy(valid).to(device),
    )


def _save_checkpoint(
    path: Path,
    initial_payload: dict[str, Any],
    model: torch.nn.Module,
    *,
    step: int,
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    payload = dict(initial_payload)
    payload["model_state"] = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    payload["a4_multiview_adaptation"] = {
        "protocol": "a4_multiview_projection_drift_adaptation_v2",
        "step": int(step),
        "metrics": metrics,
        "training": vars(args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> int:
    args = _parser().parse_args()
    if (
        args.steps <= 0
        or args.batch_size <= 1
        or args.eval_batch_size <= 0
        or args.eval_every <= 0
    ):
        raise ValueError("training/evaluation sizes must be positive")
    if args.warmup_steps < 0 or args.warmup_steps >= args.steps:
        raise ValueError("--warmup-steps must be non-negative and less than --steps")
    if not 0.0 < args.clean_fraction < 1.0:
        raise ValueError("--clean-fraction must lie strictly inside (0,1)")
    device = _cuda0(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    data = args.data_dir.expanduser().resolve()
    arrays: dict[str, np.ndarray] = {
        "rgb": np.load(data / "current_rgb.npy", mmap_mode="r"),
        "truth": np.load(data / "current_q_true.npy", mmap_mode="r"),
        "xyz": np.load(data / "current_keypoints_xyz.npy", mmap_mode="r"),
        "proprio": np.load(data / "current_proprio.npy", mmap_mode="r"),
        "intrinsic": np.load(data / "camera_intrinsic.npy", mmap_mode="r"),
        "extrinsic": np.load(
            data / "camera_extrinsic_cam_from_world.npy", mmap_mode="r"
        ),
    }
    count = int(arrays["rgb"].shape[0])
    with np.load(data / "index.npz", allow_pickle=False) as archive:
        split = np.array(archive["split"], copy=True).astype(str)
        arrays["task"] = np.array(archive["task_name"], copy=True).astype(str)
        clock = np.array(archive["camera_clock_hour"], copy=True).astype(np.int64)
    train_indices = np.flatnonzero(split == "train")
    development_indices = np.flatnonzero(split == "development")
    if train_indices.size == 0 or development_indices.size == 0:
        raise ValueError("A4 data must contain train and development rows")
    if not bool(np.all(clock[train_indices] % 2 == 0)):
        raise ValueError("A4 training rows must use even clock views")
    if not bool(np.all(clock[development_indices] % 2 == 1)):
        raise ValueError("A4 development rows must use odd clock views")
    if any(int(value.shape[0]) != count for value in arrays.values()):
        raise ValueError("A4 arrays have inconsistent frame counts")

    checkpoint_path = args.init_checkpoint.expanduser().resolve()
    initial_payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model, _ = load_fcpg_vit_adapter(checkpoint_path, device=device)
    model.vision.backbone.requires_grad_(False)
    model.vision.backbone.eval()
    backbone_features, backbone_prefix = _cache_backbone_features(
        model,
        arrays["rgb"],
        batch_size=int(args.eval_batch_size),
        device=device,
    )
    development_nominal, development_valid = _fixed_nominals(
        development_indices,
        arrays,
        seed=int(args.seed) + 1,
        args=args,
    )
    baseline = _evaluate(
        model,
        backbone_features,
        backbone_prefix,
        development_indices,
        arrays,
        development_nominal,
        development_valid,
        batch_size=int(args.eval_batch_size),
        device=device,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_parameters = sum(parameter.numel() for parameter in trainable)
    if trainable_parameters <= 0:
        raise RuntimeError("A4 adaptation has no trainable parameters")
    print(
        f"cached_backbone={tuple(backbone_features.shape)} "
        f"trainable_parameters={trainable_parameters:,}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.95, 0.999),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history: list[dict[str, Any]] = []
    best_score = float("inf")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "fcpg_a4_multiview_best.pt"
    latest_path = output / "fcpg_a4_multiview_latest.pt"

    for step in range(1, int(args.steps) + 1):
        model.train()
        model.vision.backbone.eval()
        (
            features,
            prefix,
            nominal,
            xyz,
            proprio,
            target_delta,
            valid,
        ) = _training_batch(
            train_indices,
            arrays,
            backbone_features,
            backbone_prefix,
            args=args,
            rng=rng,
            device=device,
        )
        if int(args.warmup_steps) > 0 and step <= int(args.warmup_steps):
            lr_scale = float(step) / float(args.warmup_steps)
        else:
            decay_steps = max(int(args.steps) - int(args.warmup_steps), 1)
            progress = min(
                max(step - int(args.warmup_steps), 0) / decay_steps,
                1.0,
            )
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = float(args.lr) * lr_scale
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model.forward_from_backbone_features(
                features,
                prefix,
                nominal,
                xyz,
                proprio,
                image_size=IMAGE_SIZE,
            ).delta_px
            per_anchor = F.smooth_l1_loss(
                prediction, target_delta, reduction="none", beta=1.0
            ).mean(dim=-1)
            weight = valid.to(dtype=per_anchor.dtype)
            loss = (per_anchor * weight).sum() / weight.sum().clamp_min(1.0)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % int(args.eval_every) == 0 or step == int(args.steps):
            metrics = _evaluate(
                model,
                backbone_features,
                backbone_prefix,
                development_indices,
                arrays,
                development_nominal,
                development_valid,
                batch_size=int(args.eval_batch_size),
                device=device,
            )
            score = float(metrics["drift"]["after_epe_px"]) + 2.0 * float(
                metrics["clean"]["after_epe_px"]
            )
            row = {
                "step": int(step),
                "train_loss": float(loss.detach().cpu()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "selection_score": score,
                **metrics,
            }
            history.append(row)
            print(
                f"step={step} loss={row['train_loss']:.5f} "
                f"clean={metrics['clean']['after_epe_px']:.3f}px "
                f"drift={metrics['drift']['before_epe_px']:.3f}->"
                f"{metrics['drift']['after_epe_px']:.3f}px",
                flush=True,
            )
            if score < best_score:
                best_score = score
                _save_checkpoint(
                    best_path,
                    initial_payload,
                    model,
                    step=step,
                    metrics=metrics,
                    args=args,
                )

    final_metrics = history[-1]
    _save_checkpoint(
        latest_path,
        initial_payload,
        model,
        step=int(args.steps),
        metrics=final_metrics,
        args=args,
    )
    report = {
        "protocol": "a4_multiview_projection_drift_adaptation_v2",
        "data_dir": str(data),
        "init_checkpoint": str(checkpoint_path),
        "train_frames": int(train_indices.size),
        "development_frames": int(development_indices.size),
        "train_views": "even_clock_hours",
        "development_views": "odd_clock_hours",
        "trainable_parameters": int(trainable_parameters),
        "baseline": baseline,
        "history": history,
        "best_selection_score": best_score,
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "baseline": baseline, "final": final_metrics}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
