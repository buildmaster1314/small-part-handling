#!/usr/bin/env python3
"""Train a compact deployable H8 proposal head for LeWM MPC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_goal_conditioned_proposer import (  # noqa: E402
    GoalConditionedActionPrefixProposer,
    advantage_weighted_prefix_target,
    proposer_checkpoint_payload,
)
from fcpg.lewm_action_branch_a4 import TASKS  # noqa: E402
from fcpg.lewm_goal_conditioned_deployable import (  # noqa: E402
    load_deployable_checkpoint,
)
from fcpg.lewm_history_conditioned_metric3d_a4 import (  # noqa: E402
    load_history_conditioned_checkpoint,
)
from fcpg.transformer_grounding import load_fcpg_vit_adapter  # noqa: E402
from scripts.train_lewm_goal_conditioned_direct import _encode_states, _fcpg_q  # noqa: E402


DEFAULT_DATASET = ROOT / "data/metaworld_projection/moving_goal_reach_push_branches_train.npz"
DEFAULT_CACHE = ROOT / "outputs/lewm_goal_conditioned_deployable_seed93011_2k/current_state_cache.npz"
DEFAULT_OUTPUT = ROOT / "outputs/lewm_goal_conditioned_proposer_seed93031_2k"
DEFAULT_PARENT = ROOT / (
    "outputs/lewm_action_prefix_a4/camera_action_metric3d_world_history_seed40583_10k/"
    "lewm_history_conditioned_latest.pt"
)
DEFAULT_FCPG = ROOT / (
    "outputs/transformer_residual/lewm_stage1/shared_4task_seed0_sig002/"
    "fcpg_transformer_residual_best.pt"
)
DEFAULT_LEWM = ROOT / (
    "outputs/lewm_goal_conditioned_deployable_seed93011_2k/"
    "goal_conditioned_deployable_latest.pt"
)
GROUNDINGS = ("nominal", "fcpg", "oracle")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--state-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--parent-checkpoint", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--fcpg-checkpoint", type=Path, default=DEFAULT_FCPG)
    parser.add_argument("--lewm-checkpoint", type=Path, default=DEFAULT_LEWM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=93031)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--temperature-m", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--slot-dim", type=int, default=128)
    parser.add_argument("--use-lewm-entity-estimate", action="store_true")
    parser.add_argument(
        "--use-oracle-entity",
        action="store_true",
        help="Diagnostic only: condition the proposer on simulator entity XYZ.",
    )
    return parser


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("proposal training permits only CUDA0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _batch(
    data: dict[str, np.ndarray],
    states: dict[str, np.ndarray],
    target: np.ndarray,
    rows: np.ndarray,
    grounding_index: np.ndarray,
    device: torch.device,
    entity_estimates: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    selected_state = np.stack(
        [states[GROUNDINGS[int(g)]][int(row)] for row, g in zip(rows, grounding_index)]
    )
    task_index = np.asarray(data["task_index"][rows], dtype=np.int64)
    inputs = {
        "current_state": torch.from_numpy(np.ascontiguousarray(selected_state)).to(device),
        "history_action": torch.from_numpy(
            np.ascontiguousarray(data["history_action"][rows])
        ).to(device),
        "current_goal_xyz": torch.from_numpy(
            np.ascontiguousarray(data["current_goal"][rows])
        ).to(device),
        "current_goal_velocity": torch.from_numpy(
            np.ascontiguousarray(data["current_goal_velocity"][rows])
        ).to(device),
        "current_world_xyz": torch.from_numpy(
            np.ascontiguousarray(data["current_xyz"][rows])
        ).to(device),
        "task_index": torch.from_numpy(task_index).to(device),
    }
    if entity_estimates is not None:
        selected_entity = np.stack(
            [
                entity_estimates[GROUNDINGS[int(g)]][int(row)]
                for row, g in zip(rows, grounding_index)
            ]
        )
        inputs["current_entity_xyz"] = torch.from_numpy(
            np.ascontiguousarray(selected_entity)
        ).to(device)
    target_t = torch.from_numpy(np.ascontiguousarray(target[rows])).to(device)
    return inputs, target_t


@torch.inference_mode()
def _evaluate(
    model: GoalConditionedActionPrefixProposer,
    data: dict[str, np.ndarray],
    states: dict[str, np.ndarray],
    target: np.ndarray,
    rows: np.ndarray,
    device: torch.device,
    entity_estimates: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    model.eval()
    result: dict[str, Any] = {}
    for grounding_index, grounding in enumerate(GROUNDINGS):
        inputs, expected = _batch(
            data,
            states,
            target,
            rows,
            np.full(len(rows), grounding_index, dtype=np.int64),
            device,
            entity_estimates,
        )
        predicted = model(**inputs).action_prefix
        error = (predicted - expected).square()
        result[grounding] = {
            "action_mse": float(error.mean().cpu()),
            "first_token_mse": float(error[:, 0].mean().cpu()),
        }
    model.train()
    return result


@torch.inference_mode()
def _estimate_current_entities(
    model: Any,
    data: dict[str, np.ndarray],
    states: dict[str, np.ndarray],
    device: torch.device,
    *,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    model.eval()
    task_names = data["task_name"].astype(str)
    task_index = np.asarray([TASKS.index(name) for name in task_names], dtype=np.int64)
    estimates: dict[str, np.ndarray] = {}
    for grounding in GROUNDINGS:
        parts: list[np.ndarray] = []
        for start in range(0, len(task_names), int(batch_size)):
            rows = np.arange(start, min(start + int(batch_size), len(task_names)))
            count = len(rows)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(
                    current_state=torch.from_numpy(
                        np.ascontiguousarray(states[grounding][rows])
                    ).to(device),
                    history_action=torch.from_numpy(
                        np.ascontiguousarray(data["history_action"][rows])
                    ).to(device),
                    future_action=torch.zeros(
                        (count, 8, 20), dtype=torch.float32, device=device
                    ),
                    current_goal_xyz=torch.from_numpy(
                        np.ascontiguousarray(data["current_goal"][rows])
                    ).to(device),
                    current_goal_velocity=torch.from_numpy(
                        np.ascontiguousarray(data["current_goal_velocity"][rows])
                    ).to(device),
                    task_index=torch.from_numpy(task_index[rows]).to(device),
                    batch={
                        "current_keypoints_xyz": torch.from_numpy(
                            np.ascontiguousarray(data["current_xyz"][rows])
                        ).to(device),
                        "camera_intrinsic": torch.from_numpy(
                            np.ascontiguousarray(data["camera_intrinsic"][rows])
                        ).to(device),
                        "camera_extrinsic_cam_from_world": torch.from_numpy(
                            np.ascontiguousarray(data["true_extrinsic"][rows])
                        ).to(device),
                    },
                )
            parts.append(output.predicted_entity_xyz[:, 0].float().cpu().numpy())
        estimates[grounding] = np.ascontiguousarray(
            np.concatenate(parts, axis=0), dtype=np.float32
        )
    return estimates


def _oracle_current_entities(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    tasks = data["task_name"].astype(str)
    observation = np.asarray(data["current_obs"], dtype=np.float32)
    result = np.empty((len(tasks), 3), dtype=np.float32)
    for index, task in enumerate(tasks):
        start = 0 if task == "reach-v3" else 4
        result[index] = observation[index, start : start + 3]
    if not np.isfinite(result).all():
        raise ValueError("oracle current entity contains non-finite values")
    return {name: result.copy() for name in GROUNDINGS}


def main() -> int:
    args = _parser().parse_args()
    if int(args.steps) <= 0 or int(args.batch_size) <= 0 or int(args.eval_interval) <= 0:
        raise ValueError("steps, batch-size, and eval-interval must be positive")
    device = _device(args.device)
    if args.use_lewm_entity_estimate and args.use_oracle_entity:
        raise ValueError("choose only one entity source")
    torch.cuda.set_device(device)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    data = _load_npz(args.dataset)
    cache_path = args.state_cache.expanduser().resolve()
    parent = None
    if cache_path.exists():
        cache = _load_npz(cache_path)
    else:
        parent_payload = torch.load(
            args.parent_checkpoint.expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        parent = load_history_conditioned_checkpoint(parent_payload).to(device).eval()
        corrector, _ = load_fcpg_vit_adapter(
            args.fcpg_checkpoint.expanduser().resolve(), device=device
        )
        corrector.eval()
        q_fcpg = _fcpg_q(corrector, data, device)
        encoded = _encode_states(parent, data, q_fcpg, device)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **encoded, q_fcpg=q_fcpg)
        cache = {**encoded, "q_fcpg": q_fcpg}
    states = {name: np.asarray(cache[name], dtype=np.float32) for name in GROUNDINGS}
    rows = len(data["task_name"])
    if any(value.shape != (rows, 4, 192) for value in states.values()):
        raise ValueError("state cache does not match the proposal dataset")
    entity_estimates: dict[str, np.ndarray] | None = None
    if args.use_oracle_entity:
        entity_estimates = _oracle_current_entities(data)
    elif args.use_lewm_entity_estimate:
        entity_keys = {name: f"entity_{name}" for name in GROUNDINGS}
        if not all(key in cache for key in entity_keys.values()):
            if parent is None:
                parent_payload = torch.load(
                    args.parent_checkpoint.expanduser().resolve(),
                    map_location="cpu",
                    weights_only=False,
                )
                parent = load_history_conditioned_checkpoint(parent_payload).to(device).eval()
            lewm_payload = torch.load(
                args.lewm_checkpoint.expanduser().resolve(),
                map_location="cpu",
                weights_only=False,
            )
            lewm = load_deployable_checkpoint(lewm_payload, parent).to(device).eval()
            generated = _estimate_current_entities(lewm, data, states, device)
            cache = {
                **cache,
                **{entity_keys[name]: value for name, value in generated.items()},
            }
            np.savez_compressed(cache_path, **cache)
        entity_estimates = {
            name: np.asarray(cache[entity_keys[name]], dtype=np.float32)
            for name in GROUNDINGS
        }
        if any(value.shape != (rows, 3) for value in entity_estimates.values()):
            raise ValueError("cached LeWM entity estimates must have shape [N,3]")
    if "expert_action_prefix" in data:
        target = np.asarray(data["expert_action_prefix"], dtype=np.float32)
        if target.shape != (rows, 8, 20) or not np.isfinite(target).all():
            raise ValueError("expert_action_prefix must have finite shape [N,8,20]")
        weights = None
        target_source = "exact_expert_prefix"
    else:
        target, weights = advantage_weighted_prefix_target(
            data["candidate_action"],
            data["target_entity_xyz"],
            data["target_goal"],
            temperature_m=float(args.temperature_m),
        )
        target_source = "advantage_weighted_candidate_average"
    split = data["split"].astype(str)
    train_rows = np.flatnonzero(split == "train")
    development_rows = np.flatnonzero(split == "development")
    if len(train_rows) == 0 or len(development_rows) == 0:
        raise ValueError("dataset must contain train and development rows")

    model = GoalConditionedActionPrefixProposer(
        hidden_dim=int(args.hidden_dim),
        slot_dim=int(args.slot_dim),
        use_entity_estimate=bool(
            args.use_lewm_entity_estimate or args.use_oracle_entity
        ),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=1.0e-4
    )
    rng = np.random.default_rng(int(args.seed))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, Any]] = []
    latest_metrics: dict[str, Any] = {}
    best_score = float("inf")
    model.train()
    for step in range(1, int(args.steps) + 1):
        sampled_rows = rng.choice(train_rows, size=int(args.batch_size), replace=True)
        sampled_grounding = rng.integers(0, len(GROUNDINGS), size=len(sampled_rows))
        inputs, expected = _batch(
            data,
            states,
            target,
            sampled_rows,
            sampled_grounding,
            device,
            entity_estimates,
        )
        predicted = model(**inputs).action_prefix
        action_loss = F.smooth_l1_loss(predicted, expected, beta=0.05)
        first_loss = F.smooth_l1_loss(predicted[:, 0], expected[:, 0], beta=0.05)
        loss = action_loss + 0.5 * first_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % int(args.eval_interval) == 0 or step == int(args.steps):
            evaluation = _evaluate(
                model,
                data,
                states,
                target,
                development_rows,
                device,
                entity_estimates,
            )
            latest_metrics = {
                "step": int(step),
                "train_loss": float(loss.detach().cpu()),
                "development": evaluation,
            }
            metrics.append(latest_metrics)
            payload = proposer_checkpoint_payload(
                model,
                step=step,
                args=vars(args),
                metrics=latest_metrics,
            )
            torch.save(payload, output_dir / "goal_conditioned_proposer_latest.pt")
            score = float(
                np.mean(
                    [
                        float(values["action_mse"])
                        for values in evaluation.values()
                    ]
                )
            )
            if score < best_score:
                best_score = score
                torch.save(payload, output_dir / "goal_conditioned_proposer_best.pt")
            (output_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "protocol": "goal_conditioned_action_prefix_proposer_training_v1",
                        "train_rows": int(len(train_rows)),
                        "development_rows": int(len(development_rows)),
                        "target_source": target_source,
                        "target_effective_candidates": (
                            float(np.mean(1.0 / np.square(weights).sum(axis=1)))
                            if weights is not None
                            else None
                        ),
                        "best_development_action_mse": best_score,
                        "metrics": metrics,
                    },
                    indent=2,
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )
            print(json.dumps(latest_metrics), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
