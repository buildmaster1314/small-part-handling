#!/usr/bin/env python3
"""Train matched nominal/FCPG heads on randomized calibration drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fcpg.lewm_goal_conditioned_deployable import (
    DeployableGoalConditionedMetric3DLeWM,
    deployable_checkpoint_payload,
)
from fcpg.lewm_history_conditioned_metric3d_a4 import (
    load_history_conditioned_checkpoint,
)
from fcpg.transformer_grounding import load_fcpg_vit_adapter
from scripts.evaluate_lewm_goal_conditioned_deployable_robustness import (
    nominal_camera_for_rotation,
)
from scripts.train_lewm_goal_conditioned_deployable import (
    _batch,
    _device,
    evaluate_model,
)
from scripts.train_lewm_goal_conditioned_direct import (
    _encode_states,
    _fcpg_q,
    _load,
    _sample_indices,
)


PROTOCOL = "goal_conditioned_deployable_drift_matched_v1"
GROUNDINGS = ("nominal", "fcpg")


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
            "outputs/lewm_action_prefix_a4/"
            "camera_action_metric3d_world_history_seed40583_10k/"
            "lewm_history_conditioned_latest.pt"
        ),
    )
    parser.add_argument(
        "--fcpg-checkpoint",
        type=Path,
        default=Path(
            "outputs/transformer_residual/lewm_stage1/"
            "shared_4task_seed0_sig002/fcpg_transformer_residual_best.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/lewm_goal_conditioned_deployable_drift_matched_seed93021_2k"
        ),
    )
    parser.add_argument("--rotation-degrees", default="-10,-5,0,5,10")
    parser.add_argument("--rotation-axis-world", default="0,1,0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=93021)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--hidden-dim", type=int, default=256)
    return parser


def parse_signed_float_list(value: str, *, name: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite values")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def parse_axis(value: str) -> np.ndarray:
    axis = np.asarray(
        [float(item.strip()) for item in value.split(",") if item.strip()],
        dtype=np.float64,
    )
    if axis.shape != (3,) or not np.isfinite(axis).all():
        raise ValueError("rotation-axis-world must contain three finite values")
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-12:
        raise ValueError("rotation-axis-world must be non-zero")
    return axis / norm


def _condition_bank(
    *,
    data: dict[str, np.ndarray],
    parent: Any,
    corrector: Any,
    rotations: tuple[float, ...],
    axis_world: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[float, dict[str, Any]]:
    bank: dict[float, dict[str, Any]] = {}
    for rotation in rotations:
        q_nominal, valid_nominal, nominal_extrinsic = nominal_camera_for_rotation(
            data,
            axis_world=axis_world,
            degrees=float(rotation),
        )
        condition_data = dict(data)
        condition_data["current_q_nominal"] = q_nominal
        condition_data["current_valid_nominal"] = valid_nominal
        q_fcpg = _fcpg_q(
            corrector,
            condition_data,
            device,
            batch_size=int(batch_size),
        )
        states = _encode_states(
            parent,
            condition_data,
            q_fcpg,
            device,
            batch_size=max(1, int(batch_size) // 2),
        )
        bank[float(rotation)] = {
            "states": states,
            "nominal_extrinsic": nominal_extrinsic,
        }
        print(
            json.dumps(
                {
                    "cache_rotation_degrees": float(rotation),
                    "rows": int(len(data["task_name"])),
                }
            ),
            flush=True,
        )
    return bank


def _new_model(
    parent: Any,
    *,
    seed: int,
    hidden_dim: int,
    device: torch.device,
) -> DeployableGoalConditionedMetric3DLeWM:
    torch.manual_seed(int(seed))
    return DeployableGoalConditionedMetric3DLeWM(
        parent,
        hidden_dim=int(hidden_dim),
    ).to(device)


def _paired_summary(
    metrics: dict[str, Any],
    *,
    rotations: tuple[float, ...],
) -> dict[str, Any]:
    rows = []
    for rotation in rotations:
        key = str(float(rotation))
        nominal = metrics["models"]["nominal"][key]["nominal"]
        fcpg = metrics["models"]["fcpg"][key]["fcpg"]
        rows.append(
            {
                "rotation_degrees": float(rotation),
                "entity_epe_improvement_mm": float(nominal["entity_epe_mm"])
                - float(fcpg["entity_epe_mm"]),
                "spearman_delta": float(fcpg["spearman_mean"])
                - float(nominal["spearman_mean"]),
                "top1_delta": float(fcpg["top1_action_accuracy"])
                - float(nominal["top1_action_accuracy"]),
                "selected_h8_improvement_mm": float(
                    nominal["selected_actual_goal_distance_h8_mm"]
                )
                - float(fcpg["selected_actual_goal_distance_h8_mm"]),
                "by_task": {
                    task: {
                        "entity_epe_improvement_mm": float(
                            nominal["by_task"][task]["entity_epe_mm"]
                        )
                        - float(fcpg["by_task"][task]["entity_epe_mm"]),
                        "spearman_delta": float(
                            fcpg["by_task"][task]["spearman_mean"]
                        )
                        - float(nominal["by_task"][task]["spearman_mean"]),
                        "selected_h8_improvement_mm": float(
                            nominal["by_task"][task][
                                "selected_actual_goal_distance_h8_mm"
                            ]
                        )
                        - float(
                            fcpg["by_task"][task][
                                "selected_actual_goal_distance_h8_mm"
                            ]
                        ),
                    }
                    for task in ("reach-v3", "push-v3")
                },
            }
        )
    drifted = [row for row in rows if abs(float(row["rotation_degrees"])) > 0.0]

    def mean(name: str) -> float:
        return float(np.mean([float(row[name]) for row in drifted]))

    return {
        "by_rotation": rows,
        "drifted_mean_entity_epe_improvement_mm": mean(
            "entity_epe_improvement_mm"
        ),
        "drifted_mean_spearman_delta": mean("spearman_delta"),
        "drifted_mean_top1_delta": mean("top1_delta"),
        "drifted_mean_selected_h8_improvement_mm": mean(
            "selected_h8_improvement_mm"
        ),
        "by_task": {
            task: {
                "drifted_mean_entity_epe_improvement_mm": float(
                    np.mean(
                        [
                            row["by_task"][task]["entity_epe_improvement_mm"]
                            for row in drifted
                        ]
                    )
                ),
                "drifted_mean_spearman_delta": float(
                    np.mean(
                        [
                            row["by_task"][task]["spearman_delta"]
                            for row in drifted
                        ]
                    )
                ),
                "drifted_mean_selected_h8_improvement_mm": float(
                    np.mean(
                        [
                            row["by_task"][task]["selected_h8_improvement_mm"]
                            for row in drifted
                        ]
                    )
                ),
            }
            for task in ("reach-v3", "push-v3")
        },
    }


def main() -> int:
    args = _parser().parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.eval_interval <= 0:
        raise ValueError("steps, batch-size, and eval-interval must be positive")
    rotations = parse_signed_float_list(
        args.rotation_degrees,
        name="rotation-degrees",
    )
    axis_world = parse_axis(args.rotation_axis_world)
    device = _device(args.device)
    data = _load(args.dataset.expanduser().resolve())
    train_rows = np.flatnonzero(data["split"].astype(str) == "train")
    dev_rows = np.flatnonzero(data["split"].astype(str) == "development")
    if not len(train_rows) or not len(dev_rows):
        raise ValueError("dataset must contain train and development rows")

    parent_path = args.parent_checkpoint.expanduser().resolve()
    parent_payload = torch.load(parent_path, map_location="cpu", weights_only=False)
    parent = load_history_conditioned_checkpoint(parent_payload).to(device).eval()
    corrector, _ = load_fcpg_vit_adapter(
        args.fcpg_checkpoint.expanduser().resolve(),
        device=device,
    )
    corrector.eval()
    bank = _condition_bank(
        data=data,
        parent=parent,
        corrector=corrector,
        rotations=rotations,
        axis_world=axis_world,
        device=device,
        batch_size=int(args.batch_size),
    )

    models = {
        grounding: _new_model(
            parent,
            seed=int(args.seed),
            hidden_dim=int(args.hidden_dim),
            device=device,
        )
        for grounding in GROUNDINGS
    }
    optimizers = {
        grounding: torch.optim.AdamW(
            model.trainable_parameters,
            lr=3.0e-4,
            weight_decay=1.0e-4,
        )
        for grounding, model in models.items()
    }
    rng = np.random.default_rng(int(args.seed))
    history: list[dict[str, Any]] = []

    def evaluate(step: int) -> dict[str, Any]:
        report: dict[str, Any] = {"step": int(step), "models": {}}
        for model_name, model in models.items():
            report["models"][model_name] = {}
            for rotation in rotations:
                condition = bank[float(rotation)]
                report["models"][model_name][str(float(rotation))] = {
                    eval_grounding: evaluate_model(
                        model,
                        data,
                        condition["states"],
                        dev_rows,
                        eval_grounding,
                        condition["nominal_extrinsic"],
                        device,
                        int(args.batch_size),
                    )
                    for eval_grounding in GROUNDINGS
                }
        report["paired"] = _paired_summary(report, rotations=rotations)
        history.append(report)
        print(json.dumps(report), flush=True)
        return report

    evaluate(0)
    for model in models.values():
        model.train()
    for step in range(1, int(args.steps) + 1):
        rows, candidates = _sample_indices(data, train_rows, rng, args.batch_size)
        rotation = float(rng.choice(rotations))
        condition = bank[rotation]
        for grounding in GROUNDINGS:
            model = models[grounding]
            optimizer = optimizers[grounding]
            values = _batch(
                data,
                condition["states"],
                rows,
                candidates,
                grounding,
                condition["nominal_extrinsic"],
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
            for model in models.values():
                model.train()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final = history[-1]
    for grounding, model in models.items():
        model_dir = output_dir / grounding
        model_dir.mkdir(parents=True, exist_ok=True)
        payload = deployable_checkpoint_payload(
            model,
            step=int(args.steps),
            seed=int(args.seed),
            parent_checkpoint=str(parent_path),
            dataset=str(args.dataset.expanduser().resolve()),
            train_grounding=grounding,
            metrics=final["models"][grounding],
        )
        payload["training_protocol"] = PROTOCOL
        payload["drift_augmentation_degrees"] = [float(value) for value in rotations]
        payload["rotation_axis_world"] = axis_world.tolist()
        torch.save(payload, model_dir / "goal_conditioned_deployable_latest.pt")
    (output_dir / "metrics.json").write_text(
        json.dumps({"history": history}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "final": final["paired"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
