#!/usr/bin/env python3
"""Train the bounded H3-H8 native-P1-token dynamics development Gate."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.action_chunk_policy import load_action_chunk_policy  # noqa: E402
from fcpg.lewm_a3_p1_bridge_collection import (  # noqa: E402
    sha256_file,
    validate_strict_p1_checkpoint,
)
from fcpg.lewm_anchor_dynamics_pair import trainable_parameter_count  # noqa: E402
from fcpg.lewm_p1_native_dynamics import (  # noqa: E402
    P1_NATIVE_PROTOCOL,
    P1_NATIVE_TARGET_HORIZONS,
    P1_NATIVE_WRONG_ACTION_SHIFTS,
    P1NativeDynamicsCache,
    P1NativePredictorConfig,
    build_matched_p1_native_predictors,
    flatten_native_examples,
    load_p1_native_dynamics_cache,
    reconstruct_p1_native_baselines,
    rollout_matched_p1_native,
    validate_matched_p1_native_pair,
)
from fcpg.lewm_runtime import cosine_warmup_lambda  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/lewm/eval/p1_native_dynamics_dev.json"
CHECKPOINT_TYPE = "fcpg_p1_native_dynamics_pair_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=0,
        help="Run a shortened contract smoke; zero runs exactly 2,000 steps.",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _load_fixed_config(path: str | Path) -> dict[str, Any]:
    source = _resolve(path)
    payload = _read_json(source)
    _validate_config(payload)
    return payload


def _validate_config(config: Mapping[str, Any]) -> None:
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("P1-native config schema_version must be 1")
    if config.get("protocol") != P1_NATIVE_PROTOCOL:
        raise ValueError("P1-native protocol mismatch")
    scope = _mapping(config.get("scope"), "scope")
    if (
        scope.get("train_split") != "train"
        or scope.get("evaluation_split") != "development"
        or scope.get("independent_test_claim") is not False
        or scope.get("policy_rollout_authorized") is not False
        or scope.get("ten_thousand_step_training_authorized") is not False
    ):
        raise ValueError("P1-native scope must remain development-only")
    model = _mapping(config.get("model"), "model")
    expected_model = P1NativePredictorConfig().as_kwargs()
    observed_model = {name: model.get(name) for name in expected_model}
    if observed_model != expected_model or float(model.get("sigreg_weight", -1.0)) != 0.0:
        raise ValueError("P1-native model or no-SIGReg contract drifted")
    training = _mapping(config.get("training"), "training")
    observed_training = (
        tuple(int(x) for x in training.get("allowed_seeds", ())),
        int(training.get("optimizer_steps", -1)),
        int(training.get("batch_size", -1)),
        int(training.get("gradient_accumulation_steps", -1)),
        training.get("optimizer"),
        float(training.get("learning_rate", float("nan"))),
        float(training.get("weight_decay", float("nan"))),
        tuple(float(x) for x in training.get("betas", ())),
        float(training.get("epsilon", float("nan"))),
        training.get("scheduler"),
        int(training.get("warmup_steps", -1)),
        float(training.get("gradient_clip_norm", float("nan"))),
        training.get("amp"),
        training.get("amp_dtype"),
        training.get("selection"),
    )
    expected_training = (
        (15801, 15802, 15803),
        2000,
        8,
        2,
        "AdamW",
        3.0e-4,
        1.0e-4,
        (0.9, 0.999),
        1.0e-8,
        "cosine_with_linear_warmup",
        100,
        1.0,
        True,
        "float16",
        "fixed_latest",
    )
    if observed_training != expected_training:
        raise ValueError("P1-native optimizer/training contract drifted")
    data = _mapping(config.get("data"), "data")
    if tuple(int(x) for x in data.get("forecast_horizons", ())) != P1_NATIVE_TARGET_HORIZONS:
        raise ValueError("P1-native forecast must remain exactly H3-H8")
    evaluation = _mapping(config.get("evaluation"), "evaluation")
    if tuple(int(x) for x in evaluation.get("wrong_action_shifts", ())) != P1_NATIVE_WRONG_ACTION_SHIFTS:
        raise ValueError("P1-native evaluation requires all seven wrong-action shifts")
    if evaluation.get("statistical_unit") != "physical_episode":
        raise ValueError("P1-native statistical unit must be physical_episode")


def _validate_file_digest(path: Path, expected: object, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"{label} SHA-256 mismatch: {actual}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _exclusive_output(path: Path) -> Path:
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _action_moments(cache: P1NativeDynamicsCache) -> tuple[np.ndarray, np.ndarray]:
    actions = np.concatenate((cache.history_action, cache.future_action), axis=2)
    flat = actions.reshape(-1, actions.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    std = np.maximum(flat.std(axis=0), 1.0e-5)
    return mean.astype(np.float32), std.astype(np.float32)


def recursive_forecast(
    model: nn.Module,
    history_token: torch.Tensor,
    history_action: torch.Tensor,
    future_action: torch.Tensor,
) -> torch.Tensor:
    """Forecast H3-H8 without ever teacher-forcing an H3+ target token."""

    if history_token.ndim != 4 or history_token.shape[1] != 3:
        raise ValueError("history_token must be [B,3,K,D]")
    if history_action.ndim != 3 or history_action.shape[1] != 2:
        raise ValueError("history_action must be [B,2,A]")
    if future_action.ndim != 3 or future_action.shape[1] != len(P1_NATIVE_TARGET_HORIZONS):
        raise ValueError("future_action must contain exactly H2->H3 through H7->H8")
    if history_action.shape[0] != history_token.shape[0] or future_action.shape[0] != history_token.shape[0]:
        raise ValueError("P1-native forecast batch axes do not align")
    transitions = torch.cat((history_action, future_action), dim=1)
    context = history_token
    predictions: list[torch.Tensor] = []
    for offset in range(future_action.shape[1]):
        action_window = transitions[:, offset : offset + 3]
        predicted_sequence = model.predict(context, action_window)
        predicted_next = predicted_sequence[:, -1]
        predictions.append(predicted_next)
        context = torch.cat((context[:, 1:], predicted_next[:, None]), dim=1)
    return torch.stack(predictions, dim=1)


def _matched_recursive_forecast(
    factual: nn.Module,
    no_action: nn.Module,
    history_token: torch.Tensor,
    history_action: torch.Tensor,
    future_action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    factual_prediction, no_action_prediction = rollout_matched_p1_native(
        factual,
        no_action,
        history_token,
        history_action,
        future_action[:, None],
    )
    return factual_prediction[:, 0], no_action_prediction[:, 0]


def _sample_batch(
    examples: Any,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(
        torch.from_numpy(np.asarray(getattr(examples, name)[indices]))
        .to(device=device, dtype=torch.float32, non_blocking=True)
        for name in ("history_token", "history_action", "future_action", "target_token")
    )  # type: ignore[return-value]


@torch.inference_mode()
def _predict(
    model: nn.Module,
    cache: P1NativeDynamicsCache,
    *,
    device: torch.device,
    batch_size: int,
    amp: bool,
    wrong_shift_index: int | None = None,
) -> np.ndarray:
    model.eval()
    examples = flatten_native_examples(cache)
    rows: list[np.ndarray] = []
    for start in range(0, len(examples), batch_size):
        stop = min(start + batch_size, len(examples))
        history = torch.from_numpy(examples.history_token[start:stop]).to(device)
        history_action = torch.from_numpy(examples.history_action[start:stop]).to(device)
        if wrong_shift_index is None:
            future = examples.future_action[start:stop]
        else:
            future = examples.wrong_future_action[start:stop, wrong_shift_index]
        future_action = torch.from_numpy(future).to(device)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp
        ):
            prediction = recursive_forecast(
                model, history, history_action, future_action
            )
        rows.append(prediction.float().cpu().numpy())
    flat = np.concatenate(rows, axis=0)
    return flat.reshape(cache.target_token.shape)


def _episode_horizon_mse(
    prediction: np.ndarray,
    cache: P1NativeDynamicsCache,
) -> tuple[np.ndarray, np.ndarray]:
    target = cache.target_token
    if prediction.shape != target.shape:
        raise ValueError("prediction and P1-native target shapes differ")
    per_row_branch_horizon = np.mean(
        np.square(prediction.astype(np.float64) - target.astype(np.float64)),
        axis=(3, 4),
    )
    episode_ids = np.asarray(
        sorted(set(cache.physical_episode_id.reshape(-1).tolist())), dtype="U128"
    )
    values = np.empty((episode_ids.size, target.shape[2]), dtype=np.float64)
    for index, episode_id in enumerate(episode_ids.tolist()):
        mask = cache.physical_episode_id == episode_id
        values[index] = per_row_branch_horizon[mask].mean(axis=0)
    return episode_ids, values


def _metric(prediction: np.ndarray, cache: P1NativeDynamicsCache) -> dict[str, Any]:
    episodes, values = _episode_horizon_mse(prediction, cache)
    return {
        "physical_episode_count": int(episodes.size),
        "physical_episode_macro_mse": float(values.mean()),
        "per_horizon_mse": {
            f"H{horizon}": float(values[:, index].mean())
            for index, horizon in enumerate(P1_NATIVE_TARGET_HORIZONS)
        },
    }


def _relative(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float((reference.mean() - candidate.mean()) / max(reference.mean(), 1.0e-12))


def _paired_bootstrap(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size < 2:
        raise ValueError("paired bootstrap needs aligned physical-episode vectors")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, left.size, size=(samples, left.size))
    left_mean = left[draws].mean(axis=1)
    right_mean = right[draws].mean(axis=1)
    effects = (right_mean - left_mean) / np.maximum(right_mean, 1.0e-12)
    alpha = (1.0 - confidence) / 2.0
    return {
        "orientation": "positive_means_candidate_lower_MSE",
        "relative_improvement": _relative(right, left),
        "confidence": confidence,
        "confidence_interval": [
            float(np.quantile(effects, alpha)),
            float(np.quantile(effects, 1.0 - alpha)),
        ],
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "statistical_unit": "physical_episode",
    }


def _comparison(
    factual: np.ndarray,
    reference: np.ndarray,
    cache: P1NativeDynamicsCache,
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    factual_ids, factual_horizon = _episode_horizon_mse(factual, cache)
    reference_ids, reference_horizon = _episode_horizon_mse(reference, cache)
    if not np.array_equal(factual_ids, reference_ids):
        raise RuntimeError("physical-episode axes changed between paired methods")
    result = _paired_bootstrap(
        factual_horizon.mean(axis=1),
        reference_horizon.mean(axis=1),
        samples=samples,
        seed=seed,
        confidence=confidence,
    )
    result["per_horizon_relative_improvement"] = {
        f"H{horizon}": _relative(
            reference_horizon[:, index], factual_horizon[:, index]
        )
        for index, horizon in enumerate(P1_NATIVE_TARGET_HORIZONS)
    }
    return result


def _build_report(
    predictions: Mapping[str, np.ndarray],
    wrong_predictions: np.ndarray,
    cache: P1NativeDynamicsCache,
    config: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    evaluation = _mapping(config["evaluation"], "evaluation")
    gate = _mapping(config["gate"], "gate")
    samples = int(evaluation["bootstrap_samples"])
    bootstrap_seed = int(evaluation["bootstrap_seed"]) + (seed - 15801) * 100
    confidence = float(evaluation["confidence"])
    metrics = {name: _metric(value, cache) for name, value in predictions.items()}
    comparisons: dict[str, Any] = {}
    controls = (
        "matched_no_action",
        "hold_last",
        "geometry_hold_exact_fk",
        "visual_cv_exact_fk",
    )
    for index, name in enumerate(controls):
        comparisons[f"factual_vs_{name}"] = _comparison(
            predictions["factual"],
            predictions[name],
            cache,
            samples=samples,
            seed=bootstrap_seed + index,
            confidence=confidence,
        )
    wrong_metrics: dict[str, Any] = {}
    for index, shift in enumerate(P1_NATIVE_WRONG_ACTION_SHIFTS):
        name = f"cyclic_shift_{shift}"
        wrong_metrics[name] = _metric(wrong_predictions[index], cache)
        comparisons[f"factual_vs_wrong_action_{name}"] = _comparison(
            predictions["factual"],
            wrong_predictions[index],
            cache,
            samples=samples,
            seed=bootstrap_seed + len(controls) + index,
            confidence=confidence,
        )
    threshold = float(gate["paired_bootstrap_ci_lower_must_exceed"])
    minimum_effect = float(gate["minimum_relative_improvement"])
    primary_names = (
        "factual_vs_matched_no_action",
        "factual_vs_hold_last",
        "factual_vs_geometry_hold_exact_fk",
        "factual_vs_visual_cv_exact_fk",
    )
    primary_pass = {
        name: (
            float(comparisons[name]["relative_improvement"]) >= minimum_effect
            and float(comparisons[name]["confidence_interval"][0]) > threshold
        )
        for name in primary_names
    }
    wrong_names = tuple(
        f"factual_vs_wrong_action_cyclic_shift_{shift}"
        for shift in P1_NATIVE_WRONG_ACTION_SHIFTS
    )
    wrong_action_pass = {
        name: (
            float(comparisons[name]["relative_improvement"]) > 0.0
            and float(comparisons[name]["confidence_interval"][0]) > threshold
        )
        for name in wrong_names
    }
    nonlearning_names = (
        "hold_last",
        "geometry_hold_exact_fk",
        "visual_cv_exact_fk",
    )
    strongest_nonlearning_by_horizon: dict[str, float] = {}
    for horizon in P1_NATIVE_TARGET_HORIZONS:
        key = f"H{horizon}"
        factual_mse = float(metrics["factual"]["per_horizon_mse"][key])
        reference_mse = min(
            float(metrics[name]["per_horizon_mse"][key])
            for name in nonlearning_names
        )
        strongest_nonlearning_by_horizon[key] = (
            reference_mse - factual_mse
        ) / max(reference_mse, 1.0e-12)
    positive_horizons = sum(
        value > 0.0 for value in strongest_nonlearning_by_horizon.values()
    )
    horizon_pass = positive_horizons >= int(
        gate["minimum_horizons_beating_strongest_nonlearning"]
    )
    h3_noninferiority_pass = (
        strongest_nonlearning_by_horizon["H3"]
        >= -float(gate["h3_maximum_relative_regression"])
    )
    passed = (
        all(primary_pass.values())
        and all(wrong_action_pass.values())
        and horizon_pass
        and h3_noninferiority_pass
    )
    return {
        "protocol": P1_NATIVE_PROTOCOL,
        "seed": seed,
        "scope": "exposed_train_development_only",
        "forecast_horizons": list(P1_NATIVE_TARGET_HORIZONS),
        "statistical_unit": "physical_episode",
        "metrics": metrics,
        "wrong_action_metrics": wrong_metrics,
        "comparisons": comparisons,
        "screen": {
            "primary_pass": primary_pass,
            "wrong_action_pass": wrong_action_pass,
            "strongest_nonlearning_per_horizon_relative_improvement": (
                strongest_nonlearning_by_horizon
            ),
            "positive_horizon_count": positive_horizons,
            "horizon_pass": horizon_pass,
            "h3_noninferiority_pass": h3_noninferiority_pass,
            "passed": passed,
            "cross_seed_gate_not_evaluated_by_single_run": True,
            "passing_one_seed_does_not_authorize_a_claim": True,
        },
    }


def _train(
    factual: nn.Module,
    no_action: nn.Module,
    train_cache: P1NativeDynamicsCache,
    config: Mapping[str, Any],
    *,
    device: torch.device,
    seed: int,
    steps: int,
) -> list[dict[str, Any]]:
    training = _mapping(config["training"], "training")
    batch_size = int(training["batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    optimizer = torch.optim.AdamW(
        list(factual.parameters()) + list(no_action.parameters()),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=tuple(float(x) for x in training["betas"]),
        eps=float(training["epsilon"]),
    )
    warmup = min(int(training["warmup_steps"]), steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup_lambda(step, warmup, steps),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    generator = np.random.default_rng(seed + 1)
    episode_ids = np.asarray(
        sorted(set(train_cache.physical_episode_id[:, 0].tolist())), dtype="U128"
    )
    rows_by_episode = {
        episode_id: np.flatnonzero(
            train_cache.physical_episode_id[:, 0] == episode_id
        )
        for episode_id in episode_ids.tolist()
    }
    if episode_ids.size < batch_size:
        raise ValueError("P1-native training requires at least batch_size episodes")

    def sample_bundle_batch() -> tuple[torch.Tensor, ...]:
        chosen_episodes = generator.choice(
            episode_ids, size=batch_size, replace=False
        )
        row_indices = np.asarray(
            [generator.choice(rows_by_episode[str(value)]) for value in chosen_episodes],
            dtype=np.int64,
        )
        branches = int(train_cache.branch_count)

        def tensor(name: str) -> torch.Tensor:
            value = np.asarray(getattr(train_cache, name)[row_indices])
            value = value.reshape(batch_size * branches, *value.shape[2:])
            return torch.from_numpy(value).to(
                device=device, dtype=torch.float32, non_blocking=True
            )

        return tuple(
            tensor(name)
            for name in (
                "history_token",
                "history_action",
                "future_action",
                "target_token",
            )
        )

    trace: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        factual_loss_total = 0.0
        no_action_loss_total = 0.0
        for _ in range(accumulation):
            history, history_action, future_action, target = sample_bundle_batch()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                factual_prediction, no_action_prediction = _matched_recursive_forecast(
                    factual,
                    no_action,
                    history,
                    history_action,
                    future_action,
                )
                factual_loss = torch.mean(torch.square(factual_prediction - target))
                no_action_loss = torch.mean(torch.square(no_action_prediction - target))
                loss = (factual_loss + no_action_loss) / accumulation
            scaler.scale(loss).backward()
            factual_loss_total += float(factual_loss.detach()) / accumulation
            no_action_loss_total += float(no_action_loss.detach()) / accumulation
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(factual.parameters(), float(training["gradient_clip_norm"]))
        nn.utils.clip_grad_norm_(no_action.parameters(), float(training["gradient_clip_norm"]))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if step == 1 or step == steps or step % 100 == 0:
            row = {
                "step": step,
                "factual_recursive_mse": factual_loss_total,
                "matched_no_action_recursive_mse": no_action_loss_total,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            if not all(math.isfinite(float(value)) for value in row.values()):
                raise RuntimeError("P1-native training produced non-finite values")
            trace.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    return trace


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = _load_fixed_config(args.config)
    training = _mapping(config["training"], "training")
    if args.seed not in tuple(int(x) for x in training["allowed_seeds"]):
        raise ValueError("seed must be one of 15801, 15802, 15803")
    if args.smoke_steps < 0 or args.smoke_steps >= int(training["optimizer_steps"]):
        raise ValueError("smoke-steps must be zero or in [1,1999]")
    steps = args.smoke_steps or int(training["optimizer_steps"])
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("P1-native Gate requires CUDA float16 AMP")
    _seed_everything(args.seed)

    data = _mapping(config["data"], "data")
    p1 = _mapping(config["p1"], "p1")
    cache_path = _resolve(str(data["strict_cache"]))
    metadata_path = _resolve(str(data["strict_metadata"]))
    policy_path = _resolve(str(p1["policy_checkpoint"]))
    corrector_path = _resolve(str(p1["corrector_checkpoint"]))
    _validate_file_digest(cache_path, data["strict_cache_sha256"], "strict cache")
    _validate_file_digest(metadata_path, data["strict_metadata_sha256"], "strict metadata")
    _validate_file_digest(policy_path, p1["policy_checkpoint_sha256"], "P1 policy")
    _validate_file_digest(corrector_path, p1["corrector_checkpoint_sha256"], "P1 corrector")
    validate_strict_p1_checkpoint(policy_path, corrector_path)

    train_cache = load_p1_native_dynamics_cache(
        cache_path, metadata_path=metadata_path, split="train"
    )
    development_cache = load_p1_native_dynamics_cache(
        cache_path, metadata_path=metadata_path, split="development"
    )
    configured_model = _mapping(config["model"], "model")
    predictor_config = P1NativePredictorConfig(
        **{
            name: configured_model[name]
            for name in P1NativePredictorConfig().as_kwargs()
        }
    )
    # SIGReg is explicitly absent from the native dynamics head.
    model_kwargs = predictor_config.as_kwargs()
    factual, no_action = build_matched_p1_native_predictors(
        P1NativePredictorConfig(**model_kwargs), initialization_seed=args.seed
    )
    action_mean, action_std = _action_moments(train_cache)
    for model in (factual, no_action):
        model.set_action_normalization(action_mean=action_mean, action_std=action_std)
        model.to(device)
    validate_matched_p1_native_pair(factual, no_action, require_identical=True)

    template = str(config["output_dir_template"]).format(seed=args.seed)
    output = _exclusive_output(
        _resolve(args.output_dir) if args.output_dir is not None else _resolve(template + (f"_smoke{steps}" if args.smoke_steps else ""))
    )
    trace = _train(
        factual,
        no_action,
        train_cache,
        config,
        device=device,
        seed=args.seed,
        steps=steps,
    )
    validate_matched_p1_native_pair(factual, no_action)

    policy, _ = load_action_chunk_policy(policy_path, device)
    baselines = reconstruct_p1_native_baselines(
        development_cache,
        policy,
        device=device,
        batch_size=int(_mapping(config["evaluation"], "evaluation")["batch_size"]),
    )
    evaluation_batch = int(_mapping(config["evaluation"], "evaluation")["batch_size"])
    predictions = {
        "factual": _predict(
            factual, development_cache, device=device, batch_size=evaluation_batch, amp=True
        ),
        "matched_no_action": _predict(
            no_action, development_cache, device=device, batch_size=evaluation_batch, amp=True
        ),
        "hold_last": baselines.hold_last_token,
        "geometry_hold_exact_fk": baselines.geometry_hold_exact_fk_token,
        "visual_cv_exact_fk": baselines.visual_cv_exact_fk_token,
    }
    wrong_predictions = np.stack(
        [
            _predict(
                factual,
                development_cache,
                device=device,
                batch_size=evaluation_batch,
                amp=True,
                wrong_shift_index=index,
            )
            for index in range(len(P1_NATIVE_WRONG_ACTION_SHIFTS))
        ],
        axis=0,
    )
    report = _build_report(
        predictions,
        wrong_predictions,
        development_cache,
        config,
        seed=args.seed,
    )
    report.update(
        {
            "optimizer_steps": steps,
            "smoke": bool(args.smoke_steps),
            "selection": "fixed_latest",
            "parameter_count_per_head": trainable_parameter_count(factual),
            "train_physical_episode_count": len(
                set(train_cache.physical_episode_id.reshape(-1).tolist())
            ),
            "development_physical_episode_count": len(
                set(development_cache.physical_episode_id.reshape(-1).tolist())
            ),
        }
    )
    prediction_path = output / "development_predictions.npz"
    np.savez_compressed(
        prediction_path,
        target_token=development_cache.target_token,
        factual_prediction=predictions["factual"],
        matched_no_action_prediction=predictions["matched_no_action"],
        wrong_action_prediction=wrong_predictions,
        wrong_action_shift=np.asarray(P1_NATIVE_WRONG_ACTION_SHIFTS, dtype=np.int64),
        hold_last_prediction=predictions["hold_last"],
        geometry_hold_exact_fk_prediction=predictions["geometry_hold_exact_fk"],
        visual_cv_exact_fk_prediction=predictions["visual_cv_exact_fk"],
        physical_episode_id=development_cache.physical_episode_id,
        source_row_index=development_cache.source_row_index,
        branch_index=development_cache.branch_index,
        hazard_id=development_cache.hazard_id,
        horizon=np.asarray(P1_NATIVE_TARGET_HORIZONS, dtype=np.int64),
    )
    checkpoint_path = output / "p1_native_dynamics_latest.pt"
    _atomic_torch_save(
        {
            "checkpoint_type": CHECKPOINT_TYPE,
            "protocol": P1_NATIVE_PROTOCOL,
            "selection": "fixed_latest",
            "completed": steps == int(training["optimizer_steps"]),
            "step": steps,
            "seed": args.seed,
            "predictor_config": model_kwargs,
            "parameter_count_per_head": trainable_parameter_count(factual),
            "action_mean": action_mean,
            "action_std": action_std,
            "factual_state": factual.state_dict(),
            "no_action_state": no_action.state_dict(),
            "config": config,
            "strict_cache_sha256": sha256_file(cache_path),
            "p1_policy_checkpoint_sha256": sha256_file(policy_path),
            "report_screen": report["screen"],
        },
        checkpoint_path,
    )
    report["artifacts"] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "predictions": str(prediction_path),
        "predictions_sha256": sha256_file(prediction_path),
    }
    (output / "training_trace.json").write_text(
        json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "resolved_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["screen"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
