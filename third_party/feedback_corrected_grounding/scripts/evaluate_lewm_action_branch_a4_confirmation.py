#!/usr/bin/env python3
"""Evaluate three fixed A4 10k checkpoints on an authorized test-only cache."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_a4_checkpoint import (  # noqa: E402
    A4_CHECKPOINT_TYPE,
    validate_a4_checkpoint,
)
from fcpg.lewm_a4_storage import load_a4_state_cache  # noqa: E402
from fcpg.lewm_action_branch_a4 import (  # noqa: E402
    BUNDLE_BATCH_SIZE,
    CONFIRMATION_TRAINING_SEEDS,
    PROTOCOL,
    TASKS,
)
from fcpg.lewm_action_branch_a4_confirmation_metrics import (  # noqa: E402
    CONFIRMATION_CHECKPOINT_STEP,
    validate_a4_development_go_authorization,
)
from fcpg.lewm_action_branch_a4_training import (  # noqa: E402
    A4_AMP_DTYPE,
    A4_LATEST_CHECKPOINT_NAME,
    A4_RESOLVED_CONFIG_NAME,
    assert_a4_matched_capacity,
    atomic_save_a4_predictions,
    attach_a4_cache_metadata,
    build_a4_evaluation_loader,
    build_matched_a4_models,
    evaluate_a4_models,
)
from fcpg.lewm_task_context_persistent import TaskContextPersistentLeWM  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_active_control_a4.json"
DEFAULT_BASE_CHECKPOINT = (
    ROOT
    / "outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k"
    / "lewm_grounded_latest.pt"
)
DEFAULT_OUTPUT = ROOT / "outputs/lewm_action_branch_a4/confirmation_predictions"


@dataclass(frozen=True)
class PreparedCheckpoint:
    path: Path
    sha256: str
    seed: int
    step: int
    selection: str
    factual: TaskContextPersistentLeWM
    no_action: TaskContextPersistentLeWM


def _resolve(path: str | Path) -> Path:
    source = Path(path).expanduser()
    return source.resolve() if source.is_absolute() else (ROOT / source).resolve()


def _load_json(path: str | Path, *, label: str) -> dict[str, Any]:
    source = _resolve(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def _torch_load(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a mapping")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _formal_train_cache_path(summary: Mapping[str, Any]) -> str:
    binding = summary.get("formal_train_cache")
    if not isinstance(binding, Mapping):
        raise PermissionError("Development-Go is missing formal_train_cache binding")
    path = binding.get("path")
    digest = binding.get("sha256")
    if not isinstance(path, str) or not path:
        raise PermissionError("Development-Go formal_train_cache path is invalid")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise PermissionError("Development-Go formal_train_cache sha256 is invalid")
    source = _resolve(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if _sha256(source) != digest:
        raise PermissionError(
            "Development-Go formal_train_cache content no longer matches its binding"
        )
    return str(source)


def _validate_resolved_config(
    checkpoint_path: Path,
    *,
    seed: int,
    cache_identifier: str,
    base_identifier: str,
) -> None:
    config = _load_json(
        checkpoint_path.parent / A4_RESOLVED_CONFIG_NAME,
        label="A4 10k resolved config",
    )
    observed = (
        config.get("protocol"),
        config.get("budget"),
        config.get("seed"),
        config.get("optimizer_steps"),
        config.get("train_split"),
        config.get("evaluation_split"),
        config.get("selection"),
        str(_resolve(str(config.get("input_npz", "")))),
        str(_resolve(str(config.get("base_checkpoint", "")))),
    )
    expected = (
        PROTOCOL,
        "10k",
        seed,
        CONFIRMATION_CHECKPOINT_STEP,
        "train",
        None,
        "fixed_latest",
        str(_resolve(cache_identifier)),
        str(_resolve(base_identifier)),
    )
    if observed != expected:
        raise ValueError(
            "A4 10k resolved config does not identify a train-only fixed-latest run"
        )


def prepare_confirmation_checkpoints(
    checkpoint_paths: Sequence[str | Path],
    *,
    base_checkpoint_path: str | Path,
    formal_train_cache_path: str,
    model_builder: Callable[[Mapping[str, Any]], TaskContextPersistentLeWM]
    | None = None,
) -> tuple[PreparedCheckpoint, ...]:
    """Validate all three endpoints and load both model states before test access."""

    sources = tuple(_resolve(path) for path in checkpoint_paths)
    if len(sources) != len(CONFIRMATION_TRAINING_SEEDS):
        raise ValueError("A4 confirmation requires exactly three 10k checkpoints")
    if len(set(sources)) != len(sources):
        raise ValueError("A4 confirmation checkpoint paths must be distinct")

    base_path = _resolve(base_checkpoint_path)
    base_checkpoint = _torch_load(base_path, label="A4 frozen base checkpoint")
    prepared_by_seed: dict[int, PreparedCheckpoint] = {}
    for source in sources:
        if source.name != A4_LATEST_CHECKPOINT_NAME:
            raise ValueError(
                "A4 confirmation accepts only the fixed-latest checkpoint filename"
            )
        raw = _torch_load(source, label="A4 10k checkpoint")
        checkpoint = validate_a4_checkpoint(raw)
        seed = int(checkpoint["seed"])
        step = int(checkpoint["step"])
        if checkpoint["checkpoint_type"] != A4_CHECKPOINT_TYPE:
            raise ValueError("A4 confirmation checkpoint protocol drifted")
        if seed not in CONFIRMATION_TRAINING_SEEDS:
            raise ValueError(
                f"A4 confirmation seed must be one of {CONFIRMATION_TRAINING_SEEDS}"
            )
        if seed in prepared_by_seed:
            raise ValueError(f"duplicate A4 confirmation seed {seed}")
        if step != CONFIRMATION_CHECKPOINT_STEP:
            raise ValueError("A4 confirmation checkpoint step must be exactly 10000")
        if str(_resolve(checkpoint["cache_identifier"])) != str(
            _resolve(formal_train_cache_path)
        ):
            raise ValueError("A4 10k checkpoint was not trained from the authorized cache")
        if str(_resolve(checkpoint["base_identifier"])) != str(base_path):
            raise ValueError("A4 10k checkpoint base identity drifted")
        _validate_resolved_config(
            source,
            seed=seed,
            cache_identifier=checkpoint["cache_identifier"],
            base_identifier=checkpoint["base_identifier"],
        )

        factual, no_action, _ = build_matched_a4_models(
            base_checkpoint,
            seed=seed,
            model_builder=model_builder,
        )
        factual.load_state_dict(checkpoint["model_state"], strict=True)
        no_action.load_state_dict(checkpoint["no_action_state"], strict=True)
        assert_a4_matched_capacity(factual, no_action)
        prepared_by_seed[seed] = PreparedCheckpoint(
            path=source,
            sha256=_sha256(source),
            seed=seed,
            step=step,
            selection="fixed_latest",
            factual=factual,
            no_action=no_action,
        )

    if tuple(sorted(prepared_by_seed)) != CONFIRMATION_TRAINING_SEEDS:
        raise ValueError(
            f"A4 confirmation seeds must be exactly {CONFIRMATION_TRAINING_SEEDS}"
        )
    return tuple(prepared_by_seed[seed] for seed in CONFIRMATION_TRAINING_SEEDS)


def _load_test_only_cache(path: str | Path) -> dict[str, np.ndarray]:
    cache = load_a4_state_cache(_resolve(path))
    split = np.asarray(cache["split"]).astype(str, copy=False)
    if set(split.tolist()) != {"test"}:
        if np.any(split == "development"):
            raise PermissionError("A4 confirmation evaluator forbids development rows")
        raise ValueError("A4 confirmation evaluator requires a test-only cache")
    observed_tasks = set(np.asarray(cache["task_name"]).astype(str).tolist())
    if observed_tasks != set(TASKS):
        raise ValueError("A4 confirmation test cache must cover all seven tasks")
    return cache


def _attach_checkpoint_identity(
    arrays: Mapping[str, np.ndarray], checkpoint: PreparedCheckpoint
) -> dict[str, np.ndarray]:
    result = dict(arrays)
    result.update(
        {
            "checkpoint_path": np.asarray(str(checkpoint.path)),
            "checkpoint_sha256": np.asarray(checkpoint.sha256),
            "checkpoint_seed": np.asarray(checkpoint.seed, dtype=np.int64),
            "checkpoint_step": np.asarray(checkpoint.step, dtype=np.int64),
            "selection": np.asarray(checkpoint.selection),
            "checkpoint_type": np.asarray(A4_CHECKPOINT_TYPE),
        }
    )
    return result


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--development-go-summary", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")

    contract = _load_json(args.config, label="A4 config")
    development_go = _load_json(
        args.development_go_summary,
        label="A4 Development-Go summary",
    )
    # Authorization and all three checkpoint states are validated before the
    # first access to the reserved-test cache.
    validate_a4_development_go_authorization(development_go, contract=contract)
    formal_train_cache = _formal_train_cache_path(development_go)
    prepared = prepare_confirmation_checkpoints(
        args.checkpoints,
        base_checkpoint_path=args.base_checkpoint,
        formal_train_cache_path=formal_train_cache,
    )

    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A4 confirmation CUDA device is unavailable")

    test_cache = _load_test_only_cache(args.test_cache)
    loader = build_a4_evaluation_loader(
        test_cache,
        split="test",
        batch_size=BUNDLE_BATCH_SIZE,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs: list[dict[str, Any]] = []
    for endpoint in prepared:
        endpoint.factual.to(device)
        endpoint.no_action.to(device)
        evaluation = evaluate_a4_models(
            endpoint.factual,
            endpoint.no_action,
            loader,
            device=device,
            step=endpoint.step,
            training_seed=endpoint.seed,
            amp_enabled=device.type == "cuda",
            amp_dtype=A4_AMP_DTYPE,
        )
        arrays = attach_a4_cache_metadata(evaluation.arrays, test_cache)
        arrays = _attach_checkpoint_identity(arrays, endpoint)
        destination = output_dir / f"seed{endpoint.seed}_test_predictions.npz"
        atomic_save_a4_predictions(arrays, destination)
        outputs.append(
            {
                "seed": endpoint.seed,
                "checkpoint": str(endpoint.path),
                "checkpoint_sha256": endpoint.sha256,
                "predictions": str(destination),
                "metrics": evaluation.metrics,
            }
        )
        endpoint.factual.cpu()
        endpoint.no_action.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    manifest = {
        "protocol": PROTOCOL,
        "stage": "a4_10k_reserved_test_prediction",
        "authorization_validated_before_test_read": True,
        "selection": "fixed_latest",
        "checkpoint_step": CONFIRMATION_CHECKPOINT_STEP,
        "training_seeds": list(CONFIRMATION_TRAINING_SEEDS),
        "test_cache": str(_resolve(args.test_cache)),
        "outputs": outputs,
    }
    _atomic_json(output_dir / "evaluation_manifest.json", manifest)
    print(json.dumps({"output": str(output_dir), "seeds": list(CONFIRMATION_TRAINING_SEEDS)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
