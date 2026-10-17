#!/usr/bin/env python3
"""Measure whether grounding correction changes a frozen policy as intended.

This is an offline diagnostic.  ``q_true`` and the oracle-grounded policy output
are used only as evaluation references; neither is a deployable policy input.
The evaluator never updates model parameters.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.action_chunk_policy import load_action_chunk_policy  # noqa: E402
from fcpg.lewm_counterfactual import (  # noqa: E402
    spearman_rank_correlation,
    task_macro_bootstrap,
)
from fcpg.lewm_counterfactual_runtime import pixel_points_inside  # noqa: E402
from fcpg.sequence_data import ProjectionSequenceDataset  # noqa: E402
from scripts.train_bc_policy import make_split  # noqa: E402


MODE_ORDER = ("nominal", "fcpg", "oracle")
MODEL_MODE = {"nominal": "geoprop", "fcpg": "fcpg", "oracle": "oracle"}
DEPLOYABLE_MODEL_KEYS = frozenset(
    {
        "obs",
        "history_action",
        "keypoints_xyz",
        "q_nominal",
        "valid",
        "image_size",
        "rgb",
        "vision_backbone_features",
        "vision_prefix",
        "sampled_visual",
        "global_visual",
        "fcpg_q_corrected",
        "fcpg_reliability",
        "source_index",
    }
)


@dataclass(frozen=True)
class ModeOutput:
    """CPU copy of one grounding intervention's relevant policy outputs."""

    action: np.ndarray
    points: np.ndarray


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def move_batch(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """Move a loader batch while making FP16 feature caches CPU-compatible."""

    result: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        moved = value.to(device, non_blocking=device.type == "cuda")
        if device.type == "cpu" and moved.dtype in (torch.float16, torch.bfloat16):
            moved = moved.float()
        result[key] = moved
    return result


def _checkpoint_path(value: object) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _input_path(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value).expanduser().resolve()


def input_contract_mismatches(
    *,
    data_path: str | Path,
    vision_cache: str | Path | None,
    grounding_cache: str | Path | None,
    checkpoint_args: Mapping[str, object],
) -> list[str]:
    """Return artifact mismatches relative to the policy's training inputs."""

    actual = {
        "data": _input_path(data_path),
        "vision_cache": _input_path(vision_cache),
        "grounding_cache": _input_path(grounding_cache),
    }
    expected = {
        name: _checkpoint_path(checkpoint_args.get(name)) for name in actual
    }
    mismatches = []
    for name in actual:
        if actual[name] != expected[name]:
            mismatches.append(
                f"{name}: evaluator={actual[name]} checkpoint={expected[name]}"
            )

    data = actual["data"]
    vision = actual["vision_cache"]
    grounding = actual["grounding_cache"]
    for label, cache in (("vision_cache", vision), ("grounding_cache", grounding)):
        if cache is None:
            continue
        metadata_path = cache / "metadata.json"
        if not metadata_path.is_file():
            mismatches.append(f"{label}: missing metadata.json at {metadata_path}")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_source = _checkpoint_path(metadata.get("source_data"))
        if metadata_source != data:
            mismatches.append(
                f"{label}.source_data: metadata={metadata_source} evaluator={data}"
            )
        if label == "grounding_cache":
            metadata_vision = _checkpoint_path(metadata.get("vision_cache"))
            if metadata_vision != vision:
                mismatches.append(
                    "grounding_cache.vision_cache: "
                    f"metadata={metadata_vision} evaluator={vision}"
                )
            metadata_corrector = _checkpoint_path(metadata.get("checkpoint"))
            expected_corrector = _checkpoint_path(
                checkpoint_args.get("fcpg_checkpoint")
            )
            if metadata_corrector != expected_corrector:
                mismatches.append(
                    "grounding_cache.corrector: "
                    f"metadata={metadata_corrector} checkpoint={expected_corrector}"
                )
    return mismatches


def _model_batch(
    batch: Mapping[str, torch.Tensor], label: str
) -> dict[str, torch.Tensor]:
    """Expose only deployable inputs, adding q_true solely for oracle scoring."""

    if label not in MODE_ORDER:
        raise ValueError(f"unknown grounding label {label!r}")
    model_batch = {
        key: value for key, value in batch.items() if key in DEPLOYABLE_MODEL_KEYS
    }
    if label == "oracle":
        if "q_true" not in batch:
            raise KeyError("oracle intervention requires q_true in the metric batch")
        model_batch["q_true"] = batch["q_true"]
    return model_batch


@torch.inference_mode()
def infer_grounding_modes(
    model: Any,
    batch: Mapping[str, torch.Tensor],
    *,
    device_type: str,
) -> dict[str, ModeOutput]:
    """Run nominal, corrected, and oracle grounding on one identical batch."""

    original_mode = str(model.mode)
    if original_mode != "fcpg":
        raise ValueError(
            "same-checkpoint sensitivity requires an FCPG policy checkpoint, "
            f"got mode={original_mode!r}"
        )
    outputs: dict[str, ModeOutput] = {}
    try:
        for label in MODE_ORDER:
            model.mode = MODEL_MODE[label]
            with torch.autocast(
                device_type=device_type,
                dtype=torch.float16,
                enabled=device_type == "cuda",
            ):
                result = model.predict_action_chunk(_model_batch(batch, label))
            outputs[label] = ModeOutput(
                action=result.action.detach().float().cpu().numpy().copy(),
                points=result.grounding_points_px.detach()
                .float()
                .cpu()
                .numpy()
                .copy(),
            )
    finally:
        model.mode = original_mode
    return outputs


def load_frame_identity(data_path: str | Path) -> dict[str, np.ndarray]:
    """Load lightweight per-frame identity fields indexed by ``anchor_index``."""

    with np.load(data_path, allow_pickle=False) as data:
        required = {"task_name", "episode_id", "frame_id"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"dataset is missing identity fields: {sorted(missing)}")
        identity = {
            "task_name": data["task_name"].astype(str),
            "episode_id": data["episode_id"].astype(np.int64),
            "frame_id": data["frame_id"].astype(np.int64),
        }
        for optional in (
            "actual_camera_rotation_deg",
            "actual_camera_rotation_axis",
            "actual_camera_translation_xyz",
            "actual_camera_clean_probability",
        ):
            if optional in data.files:
                identity[optional] = np.asarray(data[optional]).copy()
    return identity


def _motion_divergence(
    action: np.ndarray,
    oracle: np.ndarray,
    mask: np.ndarray,
    horizon: int,
) -> float:
    selected = np.asarray(mask[:horizon], dtype=bool)
    if selected.shape[0] != horizon or not bool(selected.all()):
        return float("nan")
    difference = np.asarray(action[:horizon, :3]) - np.asarray(oracle[:horizon, :3])
    return float(np.linalg.norm(difference, axis=-1)[selected].mean())


def _gripper_disagreement(
    action: np.ndarray,
    oracle: np.ndarray,
    mask: np.ndarray,
    horizon: int,
) -> float:
    selected = np.asarray(mask[:horizon], dtype=bool)
    if selected.shape[0] != horizon or not bool(selected.all()):
        return float("nan")
    disagreement = (action[:horizon, 3] >= 0.5) != (oracle[:horizon, 3] >= 0.5)
    return float(np.asarray(disagreement)[selected].mean())


def snapshot_rows(
    batch: Mapping[str, torch.Tensor],
    outputs: Mapping[str, ModeOutput],
    identity: Mapping[str, np.ndarray],
    *,
    h_exec: int,
    policy_seed: int,
) -> list[dict[str, Any]]:
    """Build one metric row per identical physical observation history."""

    if set(outputs) != set(MODE_ORDER):
        raise ValueError(f"mode outputs must be exactly {MODE_ORDER}")
    actions = {name: outputs[name].action for name in MODE_ORDER}
    points = {name: outputs[name].points[:, -1] for name in MODE_ORDER}
    batch_size = actions["oracle"].shape[0]
    if h_exec <= 0 or h_exec > actions["oracle"].shape[1]:
        raise ValueError(
            f"h_exec must be in [1, {actions['oracle'].shape[1]}], got {h_exec}"
        )
    if any(value.shape != actions["oracle"].shape for value in actions.values()):
        raise ValueError("all grounding modes must produce the same action shape")
    if any(value.shape != points["oracle"].shape for value in points.values()):
        raise ValueError("all grounding modes must produce the same point shape")

    anchor_indices = batch["anchor_index"].detach().cpu().numpy().astype(np.int64)
    valid = batch["valid"][:, -1].detach().cpu().numpy().astype(bool)
    action_mask = batch["action_mask"].detach().cpu().numpy().astype(bool)
    image_sizes = batch["image_size"].detach().cpu().numpy().astype(np.int64)
    q_nominal = batch["q_nominal"][:, -1].detach().float().cpu().numpy()
    q_true = batch["q_true"][:, -1].detach().float().cpu().numpy()

    if not np.allclose(points["nominal"], q_nominal, atol=1e-4, rtol=0.0):
        raise RuntimeError("nominal intervention did not use q_nominal")
    if not np.allclose(points["oracle"], q_true, atol=1e-4, rtol=0.0):
        raise RuntimeError("oracle intervention did not use q_true")

    rows: list[dict[str, Any]] = []
    for sample in range(batch_size):
        width, height = (int(value) for value in image_sizes[sample])
        nominal_inside = pixel_points_inside(
            points["nominal"][sample], width=width, height=height
        )
        oracle_inside = pixel_points_inside(
            points["oracle"][sample], width=width, height=height
        )
        fcpg_inside = pixel_points_inside(
            points["fcpg"][sample], width=width, height=height
        )
        base = valid[sample] & nominal_inside & oracle_inside
        common = base & fcpg_inside
        base_count = int(base.sum())
        common_count = int(common.sum())
        fcpg_oob_count = int((base & ~fcpg_inside).sum())
        coverage = (
            float(common_count / base_count) if base_count else float("nan")
        )
        if common_count:
            nominal_error = float(
                np.linalg.norm(
                    points["nominal"][sample] - points["oracle"][sample], axis=-1
                )[common].mean()
            )
            fcpg_error = float(
                np.linalg.norm(
                    points["fcpg"][sample] - points["oracle"][sample], axis=-1
                )[common].mean()
            )
        else:
            nominal_error = float("nan")
            fcpg_error = float("nan")

        motion: dict[tuple[str, int], float] = {}
        gripper: dict[tuple[str, int], float] = {}
        for label in ("nominal", "fcpg"):
            for horizon in (1, h_exec):
                motion[(label, horizon)] = _motion_divergence(
                    actions[label][sample],
                    actions["oracle"][sample],
                    action_mask[sample],
                    horizon,
                )
                gripper[(label, horizon)] = _gripper_disagreement(
                    actions[label][sample],
                    actions["oracle"][sample],
                    action_mask[sample],
                    horizon,
                )

        anchor = int(anchor_indices[sample])
        row: dict[str, Any] = {
            "task": str(identity["task_name"][anchor]),
            "episode": int(identity["episode_id"][anchor]),
            "frame": int(identity["frame_id"][anchor]),
            "anchor_index": anchor,
            "policy_seed": int(policy_seed),
            "base_valid_anchors": base_count,
            "common_valid_anchors": common_count,
            "fcpg_oob_anchors": fcpg_oob_count,
            "common_support_coverage": coverage,
            "nominal_epe": nominal_error,
            "fcpg_epe": fcpg_error,
            "delta_epe": nominal_error - fcpg_error,
            "nominal_oracle_xyz_h1": motion[("nominal", 1)],
            "fcpg_oracle_xyz_h1": motion[("fcpg", 1)],
            "delta_action_xyz_h1": motion[("nominal", 1)]
            - motion[("fcpg", 1)],
            "nominal_oracle_xyz_h8": motion[("nominal", h_exec)],
            "fcpg_oracle_xyz_h8": motion[("fcpg", h_exec)],
            "delta_action_xyz_h8": motion[("nominal", h_exec)]
            - motion[("fcpg", h_exec)],
            "nominal_oracle_gripper_h1": gripper[("nominal", 1)],
            "fcpg_oracle_gripper_h1": gripper[("fcpg", 1)],
            "delta_gripper_h1": gripper[("nominal", 1)]
            - gripper[("fcpg", 1)],
            "nominal_oracle_gripper_h8": gripper[("nominal", h_exec)],
            "fcpg_oracle_gripper_h8": gripper[("fcpg", h_exec)],
            "delta_gripper_h8": gripper[("nominal", h_exec)]
            - gripper[("fcpg", h_exec)],
        }
        if "actual_camera_rotation_deg" in identity:
            row["actual_camera_rotation_deg"] = float(
                identity["actual_camera_rotation_deg"][anchor]
            )
        if "actual_camera_rotation_axis" in identity:
            row["actual_camera_rotation_axis"] = (
                np.asarray(
                    identity["actual_camera_rotation_axis"][anchor],
                    dtype=np.float64,
                )
                .reshape(-1)
                .tolist()
            )
        if "actual_camera_translation_xyz" in identity:
            translation = np.asarray(
                identity["actual_camera_translation_xyz"][anchor],
                dtype=np.float64,
            ).reshape(-1)
            row["actual_camera_translation_xyz"] = translation.tolist()
            row["actual_camera_translation_norm"] = float(
                np.linalg.norm(translation)
            )
        rows.append(row)
    return rows


def _finite_rows(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> list[Mapping[str, Any]]:
    return [row for row in rows if np.isfinite(float(row[metric]))]


def _json_ready(value: Any) -> Any:
    """Replace non-finite diagnostic values with JSON null."""

    if isinstance(value, Mapping):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _task_means(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> dict[str, float]:
    episode_values: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in _finite_rows(rows, metric):
        episode_values[(str(row["task"]), int(row["episode"]))].append(
            float(row[metric])
        )
    task_values: dict[str, list[float]] = defaultdict(list)
    for (task, _), values in episode_values.items():
        task_values[task].append(float(np.mean(values)))
    return {
        task: float(np.mean(values)) for task, values in sorted(task_values.items())
    }


def _episode_spearman_rows(
    rows: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if np.isfinite(float(row[left])) and np.isfinite(float(row[right])):
            grouped[(str(row["task"]), int(row["episode"]))].append(row)
    result: list[dict[str, Any]] = []
    for (task, episode), samples in sorted(grouped.items()):
        if len(samples) < 2:
            continue
        left_values = np.asarray([float(row[left]) for row in samples])
        right_values = np.asarray([float(row[right]) for row in samples])
        result.append(
            {
                "task": task,
                "episode": episode,
                "spearman": spearman_rank_correlation(left_values, right_values),
            }
        )
    return result


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    inferential: bool = True,
) -> dict[str, Any]:
    """Aggregate overlapping windows at episode level before task macro means."""

    if not rows:
        raise ValueError("cannot summarize an empty row set")
    metrics = (
        "delta_epe",
        "delta_action_xyz_h1",
        "delta_action_xyz_h8",
        "delta_gripper_h1",
        "delta_gripper_h8",
        "common_support_coverage",
    )
    summaries: dict[str, Any] = {}
    for offset, metric in enumerate(metrics):
        selected = _finite_rows(rows, metric)
        if not selected:
            summaries[metric] = {"available": False}
            continue
        per_task = _task_means(selected, metric)
        metric_summary: dict[str, Any] = {
            "available": True,
            "estimate": float(np.mean(list(per_task.values()))),
            "per_task_episode_mean": per_task,
        }
        if inferential:
            metric_summary.update(
                task_macro_bootstrap(
                    selected,
                    metric,
                    bootstrap_samples=bootstrap_samples,
                    rng=np.random.default_rng(bootstrap_seed + offset),
                )
            )
        summaries[metric] = metric_summary

    correlation_rows = _episode_spearman_rows(
        rows, "delta_epe", "delta_action_xyz_h8"
    )
    if correlation_rows:
        per_task_correlation = _task_means(correlation_rows, "spearman")
        correlation: dict[str, Any] = {
            "available": True,
            "definition": "within-episode Spearman(delta_epe, delta_action_xyz_h8)",
            "estimate": float(np.mean(list(per_task_correlation.values()))),
            "per_task_episode_mean": per_task_correlation,
        }
        if inferential:
            correlation.update(
                task_macro_bootstrap(
                    correlation_rows,
                    "spearman",
                    bootstrap_samples=bootstrap_samples,
                    rng=np.random.default_rng(bootstrap_seed + len(metrics)),
                )
            )
    else:
        correlation = {
            "available": False,
            "definition": "within-episode Spearman(delta_epe, delta_action_xyz_h8)",
        }
    episodes = {(str(row["task"]), int(row["episode"])) for row in rows}
    base_anchors = int(sum(int(row["base_valid_anchors"]) for row in rows))
    common_anchors = int(sum(int(row["common_valid_anchors"]) for row in rows))
    fcpg_oob_anchors = int(sum(int(row["fcpg_oob_anchors"]) for row in rows))
    return {
        "inferential": bool(inferential),
        "snapshots": int(len(rows)),
        "physical_episodes": int(len(episodes)),
        "tasks": sorted({str(row["task"]) for row in rows}),
        "episode_clustered_task_macro": summaries,
        "support": {
            "base_valid_anchors": base_anchors,
            "common_valid_anchors": common_anchors,
            "fcpg_oob_anchors": fcpg_oob_anchors,
            "overall_common_support_coverage": (
                float(common_anchors / base_anchors)
                if base_anchors
                else None
            ),
        },
        "epe_action_association": correlation,
        "positive_delta_definition": (
            "positive means FCPG improves EPE or moves the frozen policy output "
            "closer to its oracle-grounded output"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vision-cache", default="")
    parser.add_argument("--grounding-cache", default="")
    parser.add_argument("--allow-cache-override", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--val-fraction", type=float)
    parser.add_argument("--h-exec", type=int, default=8)
    parser.add_argument("--policy-seed", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.max_batches < 0:
        parser.error("--max-batches must be non-negative")
    if args.h_exec != 8:
        parser.error("this frozen diagnostic requires --h-exec 8")
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    if args.vision_cache and not args.grounding_cache:
        parser.error(
            "--vision-cache requires --grounding-cache so FCPG mode has cached "
            "corrections without raw RGB"
        )

    device = choose_device(args.device)
    model, checkpoint_args = load_action_chunk_policy(args.checkpoint, device)
    if str(model.mode) != "fcpg":
        parser.error("--checkpoint must be an FCPG action policy")
    artifact_mismatches = input_contract_mismatches(
        data_path=args.data,
        vision_cache=args.vision_cache or None,
        grounding_cache=args.grounding_cache or None,
        checkpoint_args=checkpoint_args,
    )
    if artifact_mismatches and not args.allow_cache_override:
        parser.error(
            "input artifacts differ from the checkpoint contract; pass "
            "--allow-cache-override only for exploratory evaluation:\n  - "
            + "\n  - ".join(artifact_mismatches)
        )
    split_override = args.split_seed is not None or args.val_fraction is not None
    split_seed = int(
        args.split_seed
        if args.split_seed is not None
        else checkpoint_args.get("split_seed", 0)
    )
    val_fraction = float(
        args.val_fraction
        if args.val_fraction is not None
        else checkpoint_args.get("val_fraction", 0.2)
    )
    policy_seed = int(
        args.policy_seed
        if args.policy_seed is not None
        else checkpoint_args.get("seed", 0)
    )
    checkpoint_policy_seed = int(checkpoint_args.get("seed", 0))
    if policy_seed != checkpoint_policy_seed:
        parser.error(
            f"--policy-seed={policy_seed} does not match checkpoint seed "
            f"{checkpoint_policy_seed}"
        )
    exploratory = bool(args.allow_cache_override)
    smoke_only = bool(split_override or args.max_batches > 0)
    inferential = not exploratory and not smoke_only
    with np.load(args.data, allow_pickle=False) as raw:
        split = make_split(raw, val_fraction, split_seed, "task-episode")
    dataset = ProjectionSequenceDataset(
        args.data,
        split.test,
        obs_mode=str(checkpoint_args.get("obs_mode", "robot")),
        obs_horizon=int(checkpoint_args.get("obs_horizon", 2)),
        action_horizon=int(checkpoint_args.get("action_horizon", 32)),
        vision_cache=args.vision_cache or None,
        grounding_cache=args.grounding_cache or None,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    identity = load_frame_identity(args.data)
    rows: list[dict[str, Any]] = []
    for batch_index, host_batch in enumerate(loader):
        if args.max_batches > 0 and batch_index >= args.max_batches:
            break
        batch = move_batch(host_batch, device)
        outputs = infer_grounding_modes(model, batch, device_type=device.type)
        rows.extend(
            snapshot_rows(
                batch,
                outputs,
                identity,
                h_exec=args.h_exec,
                policy_seed=policy_seed,
            )
        )
    if not rows:
        raise RuntimeError("evaluation produced no snapshot rows")

    payload = {
        "contract": {
            "training_performed": False,
            "same_checkpoint_same_batch": True,
            "grounding_interventions": list(MODE_ORDER),
            "oracle_and_q_true_usage": "oracle_reference_only_absent_from_deployable_modes",
            "statistical_unit": (
                "complete_episode_within_task"
                if inferential
                else "noninferential_development_rows"
            ),
            "overlapping_windows_treated_as_independent": False,
            "exploratory": exploratory,
            "smoke_only": smoke_only,
            "inferential": inferential,
            "noninferential": not inferential,
            "confidence_intervals_reported": inferential,
            "artifact_contract_mismatches": artifact_mismatches,
        },
        "inputs": {
            "data": str(_input_path(args.data)),
            "checkpoint": str(_input_path(args.checkpoint)),
            "vision_cache": (
                str(_input_path(args.vision_cache)) if args.vision_cache else None
            ),
            "grounding_cache": (
                str(_input_path(args.grounding_cache))
                if args.grounding_cache
                else None
            ),
            "split_seed": split_seed,
            "val_fraction": val_fraction,
            "policy_seed": policy_seed,
            "h_exec": int(args.h_exec),
            "max_batches": int(args.max_batches),
            "allow_cache_override": bool(args.allow_cache_override),
        },
        "summary": summarize_rows(
            rows,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            inferential=inferential,
        ),
        "rows": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_payload = _json_ready(payload)
    output_path.write_text(
        json.dumps(serialized_payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2))
    print(f"output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
