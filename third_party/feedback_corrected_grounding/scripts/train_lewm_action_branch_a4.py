#!/usr/bin/env python3
"""Train the shared seven-task A4 cached recursive dynamics pair."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_a4_storage import load_a4_state_cache  # noqa: E402
from fcpg.lewm_a4_training_data import (  # noqa: E402
    build_a4_state_cache_loader,
)
from fcpg.lewm_action_branch_a4 import (  # noqa: E402
    BUNDLE_BATCH_SIZE,
    PROTOCOL,
    SIGREG_WEIGHT,
    TASKS,
    validate_a4_contract,
)
from fcpg.lewm_action_branch_a4_confirmation_metrics import (  # noqa: E402
    validate_a4_development_go_authorization,
)
from fcpg.lewm_action_branch_a4_training import (  # noqa: E402
    A4_AMP_DTYPE,
    A4_BASE_INITIALIZATION_NAME,
    A4_BETAS,
    A4_EPS,
    A4_EVALUATION_PREDICTIONS_NAME,
    A4_GRAD_CLIP,
    A4_LATEST_CHECKPOINT_NAME,
    A4_LEARNING_RATE,
    A4_METRICS_NAME,
    A4_RESOLVED_CONFIG_NAME,
    A4_WARMUP_STEPS,
    A4_WEIGHT_DECAY,
    assert_a4_matched_initialization,
    atomic_save_a4_checkpoint,
    atomic_save_a4_predictions,
    attach_a4_cache_metadata,
    build_a4_evaluation_loader,
    build_a4_latest_checkpoint,
    build_matched_a4_models,
    evaluate_a4_models,
    matched_a4_recursive_loss,
    move_a4_batch_to_device,
    resolve_a4_training_budget,
    set_a4_model_training_mode,
    summarize_a4_cache_counts,
    validate_a4_training_batch,
)
from fcpg.lewm_runtime import cosine_warmup_lambda, seed_everything  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_active_control_a4.json"
DEFAULT_BASE_CHECKPOINT = (
    ROOT
    / "outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k"
    / "lewm_grounded_latest.pt"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument(
        "--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", choices=("2k", "10k"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--development-go-summary",
        type=Path,
        default=None,
        help="required formal A4 Development-Go summary for a fresh 10k run",
    )
    args = parser.parse_args(argv)
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.log_every <= 0:
        parser.error("--log-every must be positive")
    return args


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError("A4 base checkpoint must contain a mapping")
    return value


def _validate_confirmation_authorization(
    budget_name: str,
    path: Path | None,
    *,
    contract: Mapping[str, Any],
    input_npz: Path,
) -> None:
    if budget_name != "10k":
        if path is not None:
            raise ValueError("--development-go-summary is only valid for 10k")
        return
    if path is None:
        raise ValueError("A4 10k requires --development-go-summary")
    summary = _read_json(path, label="A4 development summary")
    validate_a4_development_go_authorization(summary, contract=contract)
    binding = summary.get("formal_train_cache")
    if not isinstance(binding, Mapping) or not isinstance(binding.get("path"), str):
        raise PermissionError(
            "A4 10k requires Development-Go to bind the formal train cache"
        )
    authorized_cache = Path(binding["path"]).expanduser().resolve()
    if authorized_cache != input_npz.expanduser().resolve():
        raise PermissionError(
            "A4 10k input must be the train cache bound by Development-Go"
        )


def _finite_training_tensors(batch: Mapping[str, Any]) -> None:
    for name in (
        "history_embedding",
        "history_action",
        "future_action",
        "target_embedding",
    ):
        value = batch[name]
        if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"A4 training {name} contains non-finite values")


def _training_metrics(output: Any) -> dict[str, float]:
    return {
        "loss": float(output.loss.detach()),
        "factual_raw_mse": float(output.factual_raw_mse.detach()),
        "no_action_raw_mse": float(output.no_action_raw_mse.detach()),
        "factual_branch_centered_mse": float(
            output.factual_branch_centered_mse.detach()
        ),
        "no_action_branch_centered_mse": float(
            output.no_action_branch_centered_mse.detach()
        ),
    }


def _build_inline_evaluation_loader(
    cache: Mapping[str, Any], *, budget: Any, num_workers: int
) -> Any | None:
    """Keep development evaluation inline and reserve test for its evaluator."""

    if budget.name == "10k":
        if budget.evaluation_split is not None:
            raise RuntimeError("A4 10k must defer reserved-test evaluation")
        return None
    if budget.name != "2k" or budget.evaluation_split != "development":
        raise RuntimeError("A4 2k evaluation split drifted")
    return build_a4_evaluation_loader(
        cache,
        split=budget.evaluation_split,
        batch_size=BUNDLE_BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    contract = _read_json(args.config, label="A4 contract")
    validate_a4_contract(contract)
    budget = resolve_a4_training_budget(args.budget, seed=args.seed)
    _validate_confirmation_authorization(
        budget.name,
        args.development_go_summary,
        contract=contract,
        input_npz=args.input_npz,
    )

    if not str(args.device).lower().startswith("cuda"):
        raise ValueError("formal A4 training requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("formal A4 training requires available CUDA")
    device = torch.device(args.device)
    seed_everything(int(args.seed))

    base_path = args.base_checkpoint.expanduser().resolve()
    cache_path = args.input_npz.expanduser().resolve()
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    checkpoint = _torch_load(base_path)
    cache = load_a4_state_cache(cache_path)
    split_values = cache["split"].astype(str, copy=False)
    if bool((split_values == "test").any()):
        raise ValueError("A4 training must never open reserved test rows")
    if not bool((split_values == "train").any()):
        raise ValueError("A4 training cache contains no train rows")

    train_loader = build_a4_state_cache_loader(
        cache,
        split=budget.train_split,
        seed=int(args.seed),
        dtype=torch.float32,
        batches_per_epoch=budget.optimizer_steps,
        num_workers=int(args.num_workers),
        pin_memory=True,
    )
    evaluation_loader = _build_inline_evaluation_loader(
        cache,
        budget=budget,
        num_workers=int(args.num_workers),
    )
    factual, no_action, initialization = build_matched_a4_models(
        checkpoint, seed=int(args.seed)
    )
    assert_a4_matched_initialization(factual, no_action)
    factual.to(device)
    no_action.to(device)

    factual_parameters = [
        parameter for parameter in factual.parameters() if parameter.requires_grad
    ]
    no_action_parameters = [
        parameter for parameter in no_action.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": factual_parameters, "name": "factual"},
            {"params": no_action_parameters, "name": "no_action"},
        ],
        lr=A4_LEARNING_RATE,
        weight_decay=A4_WEIGHT_DECAY,
        betas=A4_BETAS,
        eps=A4_EPS,
    )
    warmup_steps = min(A4_WARMUP_STEPS, budget.optimizer_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup_lambda(
            step, warmup_steps, budget.optimizer_steps
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(path.name for path in output_dir.iterdir())
    if existing:
        raise FileExistsError(
            "A4 training requires an empty output directory; "
            f"found {existing}"
        )
    report = initialization.to_dict()
    resolved = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "budget": budget.name,
        "seed": int(args.seed),
        "optimizer_steps": budget.optimizer_steps,
        "train_split": budget.train_split,
        "evaluation_split": budget.evaluation_split,
        "tasks": list(TASKS),
        "model_scope": "one_shared_model_across_all_seven_tasks",
        "bundle_batch_size": BUNDLE_BATCH_SIZE,
        "bundles_per_task_per_batch": 4,
        "base_checkpoint": str(base_path),
        "input_npz": str(cache_path),
        "base_initialization": report,
        "encoder_frozen": True,
        "anchor_encoder_sigreg": SIGREG_WEIGHT,
        "prediction_heads_add_sigreg": False,
        "objective": "matched_factual_plus_no_action_recursive_prediction_mse",
        "amp": True,
        "amp_dtype": "float16",
        "selection": "fixed_latest",
    }
    _atomic_json(output_dir / A4_BASE_INITIALIZATION_NAME, report)
    _atomic_json(output_dir / A4_RESOLVED_CONFIG_NAME, resolved)
    print(json.dumps({"base_initialization": report}, sort_keys=True), flush=True)

    set_a4_model_training_mode(factual, True)
    set_a4_model_training_mode(no_action, True)
    last_train_metrics: dict[str, float] = {}
    started = time.perf_counter()
    for step, host_batch in enumerate(train_loader, start=1):
        validate_a4_training_batch(host_batch)
        batch = move_a4_batch_to_device(host_batch, device)
        _finite_training_tensors(batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda", dtype=A4_AMP_DTYPE, enabled=True
        ):
            output = matched_a4_recursive_loss(
                factual,
                no_action,
                batch["history_embedding"],
                batch["history_action"],
                batch["future_action"],
                batch["target_embedding"],
                batch["task_index"],
            )
        if not bool(torch.isfinite(output.loss)):
            raise RuntimeError("A4 training produced a non-finite loss")
        scaler.scale(output.loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(factual_parameters, A4_GRAD_CLIP)
        nn.utils.clip_grad_norm_(no_action_parameters, A4_GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        last_train_metrics = _training_metrics(output)
        if step % int(args.log_every) == 0 or step == budget.optimizer_steps:
            record = {
                "step": step,
                "optimizer_steps": budget.optimizer_steps,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.perf_counter() - started,
                "train": last_train_metrics,
            }
            if not all(math.isfinite(value) for value in last_train_metrics.values()):
                raise RuntimeError("A4 training produced non-finite metrics")
            print(json.dumps(record, sort_keys=True), flush=True)
    if step != budget.optimizer_steps:
        raise RuntimeError("A4 loader ended before the fixed optimizer endpoint")

    evaluation = None
    if budget.name == "2k":
        if evaluation_loader is None:
            raise RuntimeError("A4 2k evaluation loader is missing")
        evaluation = evaluate_a4_models(
            factual,
            no_action,
            evaluation_loader,
            device=device,
            step=budget.optimizer_steps,
            training_seed=int(args.seed),
            amp_enabled=True,
            amp_dtype=A4_AMP_DTYPE,
        )
        prediction_arrays = attach_a4_cache_metadata(evaluation.arrays, cache)
        atomic_save_a4_predictions(
            prediction_arrays, output_dir / A4_EVALUATION_PREDICTIONS_NAME
        )
    episode_counts, bundle_counts = summarize_a4_cache_counts(cache)
    latest = build_a4_latest_checkpoint(
        factual,
        no_action,
        seed=int(args.seed),
        step=budget.optimizer_steps,
        episode_counts=episode_counts,
        bundle_counts=bundle_counts,
        cache_identifier=str(cache_path),
        base_identifier=str(base_path),
    )
    atomic_save_a4_checkpoint(latest, output_dir / A4_LATEST_CHECKPOINT_NAME)
    final_metrics: dict[str, Any] = {
        "protocol": PROTOCOL,
        "budget": budget.name,
        "seed": int(args.seed),
        "step": budget.optimizer_steps,
        "anchor_encoder_sigreg": SIGREG_WEIGHT,
        "prediction_heads_add_sigreg": False,
        "train": last_train_metrics,
    }
    if evaluation is not None:
        final_metrics["evaluation"] = evaluation.metrics
        _atomic_json(output_dir / A4_METRICS_NAME, final_metrics)
    else:
        final_metrics["evaluation"] = "deferred_to_independent_reserved_test_evaluator"
    print(json.dumps(final_metrics, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
