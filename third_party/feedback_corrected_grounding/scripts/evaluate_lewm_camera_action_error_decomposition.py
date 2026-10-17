#!/usr/bin/env python3
"""Separate camera-action LeWM latent prediction from metric readout error."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a4 import HORIZON, SIGREG_WEIGHT, TASKS  # noqa: E402
from fcpg.lewm_action_prefix_joint_a4 import (  # noqa: E402
    joint_batch_from_memmaps,
)
from fcpg.lewm_action_prefix_joint_data import open_raw_replay_arrays  # noqa: E402
from fcpg.lewm_camera_action_joint_a4 import (  # noqa: E402
    CAMERA_ACTION_PROTOCOL,
    load_camera_action_joint_checkpoint,
)
from scripts.train_lewm_action_prefix_joint_a4 import (  # noqa: E402
    _decode_metric_for_model,
    _move,
)


DEFAULT_RAW = ROOT / (
    "outputs/lewm_action_prefix_a4/joint_orbit12_gripper_24train_8dev"
)


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="LABEL=PATH; repeat to compare frozen camera-action endpoints",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=84)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if int(args.batch_size) <= 0:
        parser.error("--batch-size must be positive")
    return args


def _cuda0(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("camera-action decomposition permits only CUDA0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def _parse_checkpoints(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = str(value).partition("=")
        if not separator or not label or not raw_path:
            raise ValueError("--checkpoint must use LABEL=PATH")
        if label in result:
            raise ValueError(f"duplicate checkpoint label {label!r}")
        result[label] = Path(raw_path).expanduser().resolve()
    return result


def _empty_accumulator() -> dict[str, Any]:
    return {
        "sequence_count": 0,
        "point_count": np.zeros(HORIZON, dtype=np.int64),
        "predicted_epe_sum": np.zeros(HORIZON, dtype=np.float64),
        "substitution_epe_sum": np.zeros(HORIZON, dtype=np.float64),
        "hold_epe_sum": np.zeros(HORIZON, dtype=np.float64),
        "latent_mse_sum": np.zeros(HORIZON, dtype=np.float64),
        "predicted_direction_sum": np.zeros(HORIZON, dtype=np.float64),
        "predicted_direction_count": np.zeros(HORIZON, dtype=np.int64),
        "substitution_direction_sum": np.zeros(HORIZON, dtype=np.float64),
        "substitution_direction_count": np.zeros(HORIZON, dtype=np.int64),
    }


def _direction_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_motion = target.float() - current[:, None].float()
    predicted_motion = prediction.float() - current[:, None].float()
    target_norm = torch.linalg.vector_norm(target_motion, dim=-1)
    predicted_norm = torch.linalg.vector_norm(predicted_motion, dim=-1)
    support = valid.bool() & (target_norm >= 1.0) & (predicted_norm >= 1.0e-4)
    cosine = (
        (target_motion * predicted_motion).sum(dim=-1)
        / (target_norm * predicted_norm).clamp_min(1.0e-6)
    ).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine)), support


def _accumulate(
    accumulator: dict[str, Any],
    *,
    predicted_q: torch.Tensor,
    substitution_q: torch.Tensor,
    target_q: torch.Tensor,
    current_q: torch.Tensor,
    valid: torch.Tensor,
    predicted_state: torch.Tensor,
    target_state: torch.Tensor,
) -> None:
    point_valid = valid.bool()
    predicted_epe = torch.linalg.vector_norm(
        predicted_q.float() - target_q.float(), dim=-1
    )
    substitution_epe = torch.linalg.vector_norm(
        substitution_q.float() - target_q.float(), dim=-1
    )
    hold_epe = torch.linalg.vector_norm(
        current_q[:, None].float() - target_q.float(), dim=-1
    )
    latent_difference = predicted_state.float() - target_state.float()
    latent_mse = (
        0.25 * latent_difference[:, :, 0].square().mean(dim=-1)
        + 0.75 * latent_difference[:, :, 1:].square().mean(dim=(-1, -2))
    )
    predicted_direction, predicted_direction_valid = _direction_error(
        predicted_q, target_q, current_q, point_valid
    )
    substitution_direction, substitution_direction_valid = _direction_error(
        substitution_q, target_q, current_q, point_valid
    )

    accumulator["sequence_count"] += int(target_q.shape[0])
    for horizon in range(HORIZON):
        horizon_valid = point_valid[:, horizon]
        point_count = int(horizon_valid.sum())
        accumulator["point_count"][horizon] += point_count
        if point_count:
            accumulator["predicted_epe_sum"][horizon] += float(
                predicted_epe[:, horizon][horizon_valid].sum()
            )
            accumulator["substitution_epe_sum"][horizon] += float(
                substitution_epe[:, horizon][horizon_valid].sum()
            )
            accumulator["hold_epe_sum"][horizon] += float(
                hold_epe[:, horizon][horizon_valid].sum()
            )
        accumulator["latent_mse_sum"][horizon] += float(
            latent_mse[:, horizon].sum()
        )
        predicted_support = predicted_direction_valid[:, horizon]
        predicted_count = int(predicted_support.sum())
        accumulator["predicted_direction_count"][horizon] += predicted_count
        if predicted_count:
            accumulator["predicted_direction_sum"][horizon] += float(
                predicted_direction[:, horizon][predicted_support].sum()
            )
        substitution_support = substitution_direction_valid[:, horizon]
        substitution_count = int(substitution_support.sum())
        accumulator["substitution_direction_count"][horizon] += substitution_count
        if substitution_count:
            accumulator["substitution_direction_sum"][horizon] += float(
                substitution_direction[:, horizon][substitution_support].sum()
            )


def _divide(sum_value: float, count: int) -> float | None:
    return float(sum_value) / int(count) if int(count) > 0 else None


def _finalize(accumulator: Mapping[str, Any]) -> dict[str, Any]:
    sequence_count = int(accumulator["sequence_count"])
    if sequence_count <= 0:
        raise RuntimeError("decomposition accumulator is empty")
    per_horizon = []
    for horizon in range(HORIZON):
        point_count = int(accumulator["point_count"][horizon])
        predicted = _divide(accumulator["predicted_epe_sum"][horizon], point_count)
        substitution = _divide(
            accumulator["substitution_epe_sum"][horizon], point_count
        )
        if predicted is None or substitution is None:
            raise RuntimeError("decomposition horizon has no valid anchor points")
        per_horizon.append(
            {
                "horizon": horizon + 1,
                "point_count": point_count,
                "latent_prediction_mse": float(
                    accumulator["latent_mse_sum"][horizon]
                )
                / sequence_count,
                "predicted_anchor_epe_px": predicted,
                "target_latent_substitution_anchor_epe_px": substitution,
                "dynamics_interface_excess_epe_px": predicted - substitution,
                "target_substitution_fraction_of_predicted_epe": (
                    substitution / predicted if predicted > 0.0 else None
                ),
                "hold_current_anchor_epe_px": _divide(
                    accumulator["hold_epe_sum"][horizon], point_count
                ),
                "predicted_motion_direction_mae_degrees": _divide(
                    accumulator["predicted_direction_sum"][horizon],
                    int(accumulator["predicted_direction_count"][horizon]),
                ),
                "target_latent_substitution_direction_mae_degrees": _divide(
                    accumulator["substitution_direction_sum"][horizon],
                    int(accumulator["substitution_direction_count"][horizon]),
                ),
            }
        )

    point_count = int(np.asarray(accumulator["point_count"]).sum())
    predicted = _divide(np.asarray(accumulator["predicted_epe_sum"]).sum(), point_count)
    substitution = _divide(
        np.asarray(accumulator["substitution_epe_sum"]).sum(), point_count
    )
    if predicted is None or substitution is None:
        raise RuntimeError("decomposition summary has no valid anchor points")
    predicted_direction_count = int(
        np.asarray(accumulator["predicted_direction_count"]).sum()
    )
    substitution_direction_count = int(
        np.asarray(accumulator["substitution_direction_count"]).sum()
    )
    return {
        "sequence_count": sequence_count,
        "point_count": point_count,
        "latent_prediction_mse": float(
            np.asarray(accumulator["latent_mse_sum"]).sum()
        )
        / (sequence_count * HORIZON),
        "predicted_anchor_epe_px": predicted,
        "target_latent_substitution_anchor_epe_px": substitution,
        "dynamics_interface_excess_epe_px": predicted - substitution,
        "target_substitution_fraction_of_predicted_epe": (
            substitution / predicted if predicted > 0.0 else None
        ),
        "hold_current_anchor_epe_px": float(
            np.asarray(accumulator["hold_epe_sum"]).sum()
        )
        / point_count,
        "predicted_motion_direction_mae_degrees": _divide(
            np.asarray(accumulator["predicted_direction_sum"]).sum(),
            predicted_direction_count,
        ),
        "target_latent_substitution_direction_mae_degrees": _divide(
            np.asarray(accumulator["substitution_direction_sum"]).sum(),
            substitution_direction_count,
        ),
        "per_horizon": per_horizon,
    }


@torch.no_grad()
def evaluate_checkpoint(
    model: torch.nn.Module,
    index: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    development = np.flatnonzero(
        np.asarray(index["split"]).astype(str) == "development"
    )
    pairs = [(int(bundle), branch) for bundle in development for branch in range(8)]
    overall = _empty_accumulator()
    by_task = {task: _empty_accumulator() for task in TASKS}
    by_task_view: dict[str, dict[str, Any]] = {}
    camera_names = np.asarray(index.get("camera_name", np.asarray([""] * len(index["split"]))))

    for start in range(0, len(pairs), int(batch_size)):
        selected = pairs[start : start + int(batch_size)]
        bundles = np.asarray([value[0] for value in selected], dtype=np.int64)
        branches = np.asarray([value[1] for value in selected], dtype=np.int64)
        batch = _move(
            joint_batch_from_memmaps(
                index,
                arrays,
                bundle_indices=bundles,
                branch_indices=branches,
            ),
            device,
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(batch)
            image_size = (
                int(batch["current_rgb"].shape[-1]),
                int(batch["current_rgb"].shape[-2]),
            )
            substitution_q = _decode_metric_for_model(
                model,
                predicted_state=output.target_state,
                current_state=output.current_state,
                batch=batch,
                image_size=image_size,
            )

        common = {
            "predicted_q": output.predicted_q_px,
            "substitution_q": substitution_q,
            "target_q": batch["future_q_true"],
            "current_q": batch["current_q_corrected"],
            "valid": batch["future_valid"],
            "predicted_state": output.predicted_state,
            "target_state": output.target_state,
        }
        _accumulate(overall, **common)
        task_indices = batch["task_index"].detach().cpu().numpy()
        for row, task_index in enumerate(task_indices.tolist()):
            task = TASKS[int(task_index)]
            row_values = {name: value[row : row + 1] for name, value in common.items()}
            _accumulate(by_task[task], **row_values)
            camera = str(camera_names[bundles[row]])
            if camera:
                key = f"{task}::{camera}"
                _accumulate(by_task_view.setdefault(key, _empty_accumulator()), **row_values)

    per_task = {task: _finalize(by_task[task]) for task in TASKS}
    per_task_view = {
        key: _finalize(values) for key, values in sorted(by_task_view.items())
    }
    task_macro = {}
    scalar_names = (
        "latent_prediction_mse",
        "predicted_anchor_epe_px",
        "target_latent_substitution_anchor_epe_px",
        "dynamics_interface_excess_epe_px",
        "target_substitution_fraction_of_predicted_epe",
        "hold_current_anchor_epe_px",
        "predicted_motion_direction_mae_degrees",
        "target_latent_substitution_direction_mae_degrees",
    )
    for name in scalar_names:
        values = [per_task[task][name] for task in TASKS]
        finite = [float(value) for value in values if value is not None]
        task_macro[name] = float(np.mean(finite)) if finite else None
    return {
        "overall": _finalize(overall),
        "task_macro": task_macro,
        "per_task": per_task,
        "per_task_view": per_task_view,
    }


def _markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Camera-action LeWM error decomposition",
        "",
        "> Development diagnostic only. `target latent substitution` feeds the encoded future latent into the frozen metric readout. It is not deployable and is not a strict oracle lower bound because the readout was trained on predicted states.",
        "",
        "| Model | Step | Predicted EPE | Target-latent substitution EPE | Excess after dynamics | Substitution fraction | Direction | Substitution direction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, model in result["models"].items():
        macro = model["evaluation"]["task_macro"]
        lines.append(
            "| {label} | {step} | {pred:.3f} px | {sub:.3f} px | {excess:.3f} px | {fraction:.1%} | {direction:.2f} deg | {sub_direction:.2f} deg |".format(
                label=label,
                step=int(model["checkpoint_step"]),
                pred=float(macro["predicted_anchor_epe_px"]),
                sub=float(macro["target_latent_substitution_anchor_epe_px"]),
                excess=float(macro["dynamics_interface_excess_epe_px"]),
                fraction=float(macro["target_substitution_fraction_of_predicted_epe"]),
                direction=float(macro["predicted_motion_direction_mae_degrees"]),
                sub_direction=float(
                    macro["target_latent_substitution_direction_mae_degrees"]
                ),
            )
        )
    lines.extend(["", "## Towel clock03", ""])
    lines.extend(
        [
            "| Model | Predicted EPE | Substitution EPE | Excess | H8 predicted | H8 substitution | Direction | Substitution direction |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    key = "towel-wipe-v0::clock03"
    for label, model in result["models"].items():
        view = model["evaluation"]["per_task_view"].get(key)
        if view is None:
            continue
        h8 = view["per_horizon"][-1]
        lines.append(
            "| {label} | {pred:.3f} px | {sub:.3f} px | {excess:.3f} px | {h8_pred:.3f} px | {h8_sub:.3f} px | {direction:.2f} deg | {sub_direction:.2f} deg |".format(
                label=label,
                pred=float(view["predicted_anchor_epe_px"]),
                sub=float(view["target_latent_substitution_anchor_epe_px"]),
                excess=float(view["dynamics_interface_excess_epe_px"]),
                h8_pred=float(h8["predicted_anchor_epe_px"]),
                h8_sub=float(h8["target_latent_substitution_anchor_epe_px"]),
                direction=float(view["predicted_motion_direction_mae_degrees"]),
                sub_direction=float(
                    view["target_latent_substitution_direction_mae_degrees"]
                ),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser(argv)
    device = _cuda0(str(args.device))
    torch.cuda.set_device(device)
    raw_path = args.raw_dataset.expanduser().resolve()
    checkpoints = _parse_checkpoints(args.checkpoint)
    index, arrays = open_raw_replay_arrays(raw_path)
    models: dict[str, Any] = {}

    for label, checkpoint_path in checkpoints.items():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise TypeError(f"checkpoint {label!r} must contain a mapping")
        if payload.get("protocol") != CAMERA_ACTION_PROTOCOL:
            raise ValueError(f"checkpoint {label!r} is not a camera-action model")
        if float(payload.get("sigreg_weight", -1.0)) != SIGREG_WEIGHT:
            raise ValueError(f"checkpoint {label!r} does not use SIGReg=0.02")
        checkpoint_raw = Path(str(payload.get("raw_dataset", ""))).resolve()
        if checkpoint_raw != raw_path:
            raise ValueError(f"checkpoint {label!r} uses a different raw dataset")
        model = load_camera_action_joint_checkpoint(payload).to(device).eval()
        try:
            evaluation = evaluate_checkpoint(
                model,
                index,
                arrays,
                device=device,
                batch_size=int(args.batch_size),
            )
        finally:
            del model
            torch.cuda.empty_cache()
        models[label] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": int(payload["step"]),
            "seed": int(payload["seed"]),
            "parameter_count": int(
                sum(value.numel() for value in payload["model_state"].values())
            ),
            "evaluation": evaluation,
        }

    result = {
        "schema_version": 1,
        "diagnostic_scope": "development_target_latent_substitution",
        "formal_claim_allowed": False,
        "warning": (
            "Target-latent substitution is not deployable and is not a strict "
            "oracle lower bound because the metric readout was trained on "
            "predicted states."
        ),
        "raw_dataset": str(raw_path),
        "sigreg_weight": SIGREG_WEIGHT,
        "models": models,
    }
    for value in _walk_floats(result):
        if not math.isfinite(value):
            raise RuntimeError("decomposition output contains a non-finite metric")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    output.with_suffix(".md").write_text(_markdown(result), encoding="utf-8")
    print(output)
    return 0


def _walk_floats(value: Any):
    if isinstance(value, Mapping):
        for nested in value.values():
            yield from _walk_floats(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _walk_floats(nested)
    elif isinstance(value, float):
        yield value


if __name__ == "__main__":
    raise SystemExit(main())
