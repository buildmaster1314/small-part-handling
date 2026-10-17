#!/usr/bin/env python3
"""Train matched expert-action residual heads and evaluate a Pick Gate."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_expert_action_residual import (  # noqa: E402
    ExpertActionResidualHead,
    ExpertResidualArrays,
    METHODS,
    expert_action_loss,
    load_expert_residual_arrays,
)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _json(path: str | Path) -> dict[str, object]:
    value = json.loads(_resolve(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _split(config: Mapping[str, object], name: str) -> Mapping[str, object]:
    splits = config.get("splits")
    if not isinstance(splits, Mapping) or not isinstance(splits.get(name), Mapping):
        raise ValueError(f"training config lacks split {name!r}")
    return splits[name]


def _training(config: Mapping[str, object]) -> Mapping[str, object]:
    value = config.get("training")
    if not isinstance(value, Mapping):
        raise ValueError("training config lacks training settings")
    return value


def _tensors(
    arrays: ExpertResidualArrays,
    method_index: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "state": torch.from_numpy(
            np.ascontiguousarray(arrays.state_latent[:, method_index])
        ).to(device),
        "latent": torch.from_numpy(
            np.ascontiguousarray(arrays.action_latent[:, method_index])
        ).to(device),
        "base": torch.from_numpy(
            np.ascontiguousarray(arrays.base_action[:, method_index])
        ).to(device),
        "age": torch.from_numpy(np.ascontiguousarray(arrays.missing_age)).long().to(
            device
        ),
        "target": torch.from_numpy(np.ascontiguousarray(arrays.expert_action)).to(
            device
        ),
    }


def _predict(
    model: ExpertActionResidualHead,
    tensors: Mapping[str, torch.Tensor],
    *,
    batch_size: int = 1024,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, int(tensors["base"].shape[0]), batch_size):
            stop = min(start + batch_size, int(tensors["base"].shape[0]))
            value = model(
                base_action=tensors["base"][start:stop],
                state_latent=tensors["state"][start:stop],
                action_latent=tensors["latent"][start:stop],
                missing_age=tensors["age"][start:stop],
            )
            outputs.append(value.float().cpu().numpy())
    return np.ascontiguousarray(np.concatenate(outputs), dtype=np.float32)


def _row_mse(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    motion_only: bool = True,
) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if motion_only:
        prediction = prediction[..., :3]
        target = target[..., :3]
    return np.mean(
        np.square(prediction - target), axis=(1, 2)
    )


def _episode_mean(
    values: np.ndarray, episode_id: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.unique(episode_id)
    return ids, np.asarray(
        [float(np.mean(values[episode_id == episode])) for episode in ids],
        dtype=np.float64,
    )


def _paired_interval(
    better: np.ndarray,
    baseline: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    better = np.asarray(better, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if better.shape != baseline.shape or better.ndim != 1 or better.size < 2:
        raise ValueError("paired interval requires aligned episode vectors")
    delta = baseline - better
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, delta.size, size=(samples, delta.size))
    sampled = np.mean(delta[draws], axis=1)
    baseline_mean = float(np.mean(baseline))
    return {
        "better_mse": float(np.mean(better)),
        "baseline_mse": baseline_mean,
        "absolute_improvement": float(np.mean(delta)),
        "relative_improvement": (
            float(np.mean(delta) / baseline_mean) if baseline_mean > 0 else 0.0
        ),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
    }


def _train_one(
    train: ExpertResidualArrays,
    development: ExpertResidualArrays,
    *,
    method_index: int,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
    device: torch.device,
) -> tuple[ExpertActionResidualHead, np.ndarray, dict[str, float]]:
    torch.manual_seed(seed)
    model = ExpertActionResidualHead(
        model_dim=train.model_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_tensors = _tensors(train, method_index, device)
    development_tensors = _tensors(development, method_index, device)
    rng = np.random.default_rng(seed)
    batches = rng.integers(
        0,
        train.sample_count,
        size=(steps, min(batch_size, train.sample_count)),
    )
    model.train()
    last_loss = float("nan")
    for step, rows in enumerate(batches, start=1):
        index = torch.from_numpy(rows).long().to(device)
        prediction = model(
            base_action=train_tensors["base"].index_select(0, index),
            state_latent=train_tensors["state"].index_select(0, index),
            action_latent=train_tensors["latent"].index_select(0, index),
            missing_age=train_tensors["age"].index_select(0, index),
        )
        target = train_tensors["target"].index_select(0, index)
        loss = expert_action_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    prediction = _predict(model, development_tensors)
    target = development.expert_action
    metrics = {
        "last_train_loss": last_loss,
        "development_motion_mse": float(np.mean(_row_mse(prediction, target))),
        "development_all_action_mse": float(
            np.mean(_row_mse(prediction, target, motion_only=False))
        ),
        "development_gripper_mse": float(
            np.mean(np.square(prediction[..., 3] - target[..., 3]))
        ),
    }
    return model, prediction, metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/lewm/eval/expert_action_residual_pick_dev.json",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)

    config = _json(args.config)
    training = _training(config)
    train = load_expert_residual_arrays(_resolve(str(_split(config, "train")["output"])))
    development = load_expert_residual_arrays(
        _resolve(str(_split(config, "development")["output"]))
    )
    if set(train.episode_id.tolist()) & set(development.episode_id.tolist()):
        raise ValueError("train and development physical episodes overlap")
    if train.model_dim != development.model_dim:
        raise ValueError("train and development latent dimensions differ")

    device = torch.device(args.device)
    steps = int(training["steps"] if args.steps is None else args.steps)
    if steps <= 0:
        parser.error("--steps must be positive")
    output_dir = _resolve(args.output_dir or str(training["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(value) for value in training["seeds"])
    predictions: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
    rows: list[dict[str, object]] = []

    for seed in seeds:
        for method_index, method in enumerate(METHODS):
            model, prediction, metrics = _train_one(
                train,
                development,
                method_index=method_index,
                seed=seed,
                steps=steps,
                batch_size=int(training["batch_size"]),
                learning_rate=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
                hidden_dim=int(training["hidden_dim"]),
                device=device,
            )
            checkpoint = output_dir / f"{method}_seed{seed}_latest.pt"
            torch.save(
                {
                    "schema_version": 1,
                    "protocol": str(config["protocol"]),
                    "method": method,
                    "seed": seed,
                    "steps": steps,
                    "model_dim": train.model_dim,
                    "hidden_dim": int(training["hidden_dim"]),
                    "model_state": model.state_dict(),
                },
                checkpoint,
            )
            predictions[method].append(prediction)
            rows.append({"method": method, "seed": seed, **metrics})
            print(
                f"seed={seed} method={method} "
                f"dev_xyz_mse={metrics['development_motion_mse']:.6f}",
                flush=True,
            )

    target = np.asarray(development.expert_action, dtype=np.float64)
    base_row_mse = {
        method: _row_mse(development.base_action[:, index], target)
        for index, method in enumerate(METHODS)
    }
    residual_row_mse = {
        method: np.mean(
            np.stack([_row_mse(value, target) for value in values]), axis=0
        )
        for method, values in predictions.items()
    }
    episode_id, factual_episode = _episode_mean(
        residual_row_mse["complementary_factual"], development.episode_id
    )
    comparisons: dict[str, dict[str, float]] = {}
    bootstrap_samples = int(training["bootstrap_samples"])
    bootstrap_seed = int(training["bootstrap_seed"])
    for offset, (name, baseline) in enumerate(
        (
            ("factual_residual_vs_factual_base", base_row_mse["complementary_factual"]),
            (
                "factual_residual_vs_no_action_residual",
                residual_row_mse["complementary_no_action"],
            ),
            (
                "factual_residual_vs_geometry_residual",
                residual_row_mse["geometry_hold"],
            ),
        )
    ):
        baseline_ids, baseline_episode = _episode_mean(baseline, development.episode_id)
        if not np.array_equal(episode_id, baseline_ids):
            raise RuntimeError("paired episode ordering drifted")
        comparisons[name] = _paired_interval(
            factual_episode,
            baseline_episode,
            samples=bootstrap_samples,
            seed=bootstrap_seed + offset,
        )

    per_seed_direction = []
    for seed_index, seed in enumerate(seeds):
        factual = float(np.mean(_row_mse(predictions["complementary_factual"][seed_index], target)))
        no_action = float(np.mean(_row_mse(predictions["complementary_no_action"][seed_index], target)))
        geometry = float(np.mean(_row_mse(predictions["geometry_hold"][seed_index], target)))
        per_seed_direction.append(
            {
                "seed": seed,
                "factual_mse": factual,
                "no_action_mse": no_action,
                "geometry_mse": geometry,
                "factual_beats_both": factual < no_action and factual < geometry,
            }
        )

    base_gate = comparisons["factual_residual_vs_factual_base"]
    no_action_gate = comparisons["factual_residual_vs_no_action_residual"]
    geometry_gate = comparisons["factual_residual_vs_geometry_residual"]
    passed = bool(
        base_gate["relative_improvement"] >= 0.10
        and base_gate["ci_low"] > 0.0
        and no_action_gate["ci_low"] > 0.0
        and geometry_gate["ci_low"] > 0.0
        and sum(bool(row["factual_beats_both"]) for row in per_seed_direction) >= 2
    )
    report = {
        "schema_version": 1,
        "protocol": str(config["protocol"]),
        "classification": "fresh_development_screen",
        "primary_metric": "scripted_expert_xyz_action_mse_over_next_5_raw_steps",
        "gripper_note": (
            "all collected Pick interruption labels have gripper=1; gripper MSE is "
            "supporting-only and cannot establish grasp-state correction"
        ),
        "steps": steps,
        "train_rows": train.sample_count,
        "train_episodes": int(np.unique(train.episode_id).size),
        "development_rows": development.sample_count,
        "development_episodes": int(np.unique(development.episode_id).size),
        "methods": list(METHODS),
        "seeds": list(seeds),
        "seed_metrics": rows,
        "comparisons": comparisons,
        "per_seed_direction": per_seed_direction,
        "gate": {
            "passed": passed,
            "minimum_factual_vs_base_relative_improvement": 0.10,
            "all_three_paired_ci_lowers_must_exceed_zero": True,
            "minimum_seeds_factual_beats_both": 2,
            "rollout_authorized": passed,
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["comparisons"], indent=2, sort_keys=True))
    print(f"Gate: {'GO' if passed else 'NO-GO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
