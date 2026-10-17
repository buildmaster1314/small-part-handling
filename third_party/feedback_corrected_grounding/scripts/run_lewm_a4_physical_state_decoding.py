#!/usr/bin/env python3
"""Run the frozen A4 exposed-development physical-state diagnostic."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_a4_physical_state_decoding import (  # noqa: E402
    AUDIT_DESIGNATION,
    DEFAULT_BOOTSTRAP_SAMPLES,
    build_a4_physical_decoding_report,
    decode_a4_evaluation_arrays,
    fit_a4_physical_decoder,
    fit_fixed_a4_baselines,
    validate_a4_physical_decoding_inputs,
)
from fcpg.lewm_a4_storage import load_a4_state_cache  # noqa: E402
from fcpg.lewm_action_branch_a4 import (  # noqa: E402
    BUNDLE_BATCH_SIZE,
    CONFIRMATION_TRAINING_SEEDS,
    TASKS,
)
from fcpg.lewm_action_branch_a4_training import (  # noqa: E402
    A4_AMP_DTYPE,
    build_a4_evaluation_loader,
    evaluate_a4_models,
)
from scripts.evaluate_lewm_action_branch_a4_confirmation import (  # noqa: E402
    prepare_confirmation_checkpoints,
)


DEFAULT_STATE_CACHE = (
    ROOT / "outputs/lewm_action_branch_a4/formal_v2/state_cache.npz"
)
DEFAULT_PREFLIGHT_LABELS = (
    ROOT / "outputs/lewm_action_branch_a4/formal_v2/preflight_labels.npz"
)
DEFAULT_BASE_CHECKPOINT = (
    ROOT
    / "outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k"
    / "lewm_grounded_latest.pt"
)
DEFAULT_CHECKPOINTS = tuple(
    ROOT
    / f"outputs/lewm_action_branch_a4/shared_7task_seed{seed}_10k"
    / "lewm_action_branch_a4_latest.pt"
    for seed in CONFIRMATION_TRAINING_SEEDS
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/lewm_action_branch_a4/physical_state_decoding_exploratory"
)

FROZEN_STATE_CACHE_SHA256 = (
    "eeb0ff146a5a01d9acaeeb0c54a82fe4bedbe814a7475da3acf80366a3804de9"
)
FROZEN_PREFLIGHT_LABELS_SHA256 = (
    "f8b9e4c2e9bf58d74040e14d89d445c9a34e81183f835f863af3ee022882486a"
)
FROZEN_BASE_CHECKPOINT_SHA256 = (
    "a6c329d88692810e852685be1460569f6da28753b8d672468b20538811b9c092"
)
FROZEN_CHECKPOINT_SHA256 = {
    30_301: "69b0b9388db7b7dc4c07c203856d634c020858f060bf7962d2dfa900a74a79e0",
    30_302: "9de33a4dba0d16cb01179db8c9e6a85c9c31f4771d3ae5054af2809b1f415730",
    30_303: "c9ee1bee9a8ef4408b42c191dfba678a2c9b1f69963b0b180ad01b56ff686fd2",
}
SMOKE_TRAIN_EPISODES_PER_TASK = 2
SMOKE_DEVELOPMENT_EPISODES_PER_TASK = 1
SMOKE_BOOTSTRAP_SAMPLES = 64


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bind_file(path: str | Path, expected: str, *, role: str) -> dict[str, str]:
    source = _resolve(path)
    observed = _sha256(source)
    if observed != expected:
        raise PermissionError(
            f"frozen {role} sha256 drifted: expected {expected}, got {observed}"
        )
    return {"path": str(source), "sha256": observed}


def validate_frozen_artifacts(
    *,
    state_cache: str | Path,
    preflight_labels: str | Path,
    base_checkpoint: str | Path,
    checkpoints: Sequence[str | Path],
) -> dict[str, Any]:
    sources = tuple(checkpoints)
    if len(sources) != len(CONFIRMATION_TRAINING_SEEDS):
        raise ValueError("the diagnostic requires exactly three frozen 10k checkpoints")
    checkpoint_rows = []
    for seed, path in zip(CONFIRMATION_TRAINING_SEEDS, sources, strict=True):
        checkpoint_rows.append(
            {
                "seed": seed,
                **_bind_file(
                    path,
                    FROZEN_CHECKPOINT_SHA256[seed],
                    role=f"A4 seed {seed} 10k checkpoint",
                ),
            }
        )
    return {
        "state_cache": _bind_file(
            state_cache, FROZEN_STATE_CACHE_SHA256, role="formal_v2 state cache"
        ),
        "preflight_labels": _bind_file(
            preflight_labels,
            FROZEN_PREFLIGHT_LABELS_SHA256,
            role="formal_v2 preflight labels",
        ),
        "base_checkpoint": _bind_file(
            base_checkpoint,
            FROZEN_BASE_CHECKPOINT_SHA256,
            role="frozen base checkpoint",
        ),
        "checkpoints": checkpoint_rows,
    }


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    source = _resolve(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _smoke_subset(
    cache: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    task = np.asarray(cache["task_name"]).astype(str, copy=False)
    split = np.asarray(cache["split"]).astype(str, copy=False)
    episode = np.asarray(cache["source_episode_id"]).astype(str, copy=False)
    progress = np.asarray(labels["branch_outcome"], dtype=np.float64)[..., 0]
    position = np.asarray(labels["physical_state"], dtype=np.float64)[..., :3]
    selected = np.zeros(split.size, dtype=bool)
    limits = {
        "train": SMOKE_TRAIN_EPISODES_PER_TASK,
        "development": SMOKE_DEVELOPMENT_EPISODES_PER_TASK,
    }
    for task_name in TASKS:
        for split_name, limit in limits.items():
            candidates = np.flatnonzero((task == task_name) & (split == split_name))
            available = tuple(sorted(np.unique(episode[candidates]).tolist()))
            identities = list(available[:limit])
            if len(identities) != limit:
                raise ValueError(
                    f"smoke subset lacks {limit} {split_name} physical episodes "
                    f"for {task_name}"
                )
            if split_name == "train":
                def informative(candidate_ids: Sequence[str]) -> bool:
                    rows = (
                        (task == task_name)
                        & (split == split_name)
                        & np.isin(episode, np.asarray(candidate_ids))
                    )
                    return bool(
                        np.var(progress[rows]) > 1.0e-12
                        and np.any(
                            np.var(position[rows], axis=(0, 1, 2)) > 1.0e-12
                        )
                    )

                if not informative(identities):
                    for candidate in available[limit:]:
                        if informative((*identities, candidate)):
                            identities.append(candidate)
                            break
                if not informative(identities):
                    raise ValueError(
                        f"smoke train subset has no identifiable fixed target for {task_name}"
                    )
            selected |= (
                (task == task_name)
                & (split == split_name)
                & np.isin(episode, np.asarray(identities))
            )
    cache_subset = {
        name: np.asarray(value)[selected].copy() for name, value in cache.items()
    }
    label_subset = {
        name: np.asarray(value)[selected].copy() for name, value in labels.items()
    }
    validate_a4_physical_decoding_inputs(cache_subset, label_subset)
    return cache_subset, label_subset


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--state-cache", type=Path, default=DEFAULT_STATE_CACHE)
    parser.add_argument(
        "--preflight-labels", type=Path, default=DEFAULT_PREFLIGHT_LABELS
    )
    parser.add_argument(
        "--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT
    )
    parser.add_argument(
        "--checkpoints", type=Path, nargs=3, default=DEFAULT_CHECKPOINTS
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=BUNDLE_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    device = torch.device(args.device)
    if args.smoke and device.type != "cpu":
        raise ValueError("--smoke is CPU-only")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")

    bindings = validate_frozen_artifacts(
        state_cache=args.state_cache,
        preflight_labels=args.preflight_labels,
        base_checkpoint=args.base_checkpoint,
        checkpoints=args.checkpoints,
    )
    full_cache = load_a4_state_cache(bindings["state_cache"]["path"])
    full_labels = _load_npz(bindings["preflight_labels"]["path"])
    validate_a4_physical_decoding_inputs(full_cache, full_labels)
    cache, labels = (
        _smoke_subset(full_cache, full_labels)
        if args.smoke
        else (full_cache, full_labels)
    )

    decoder = fit_a4_physical_decoder(cache, labels)
    baselines = fit_fixed_a4_baselines(cache, labels, decoder)
    prepared = prepare_confirmation_checkpoints(
        tuple(row["path"] for row in bindings["checkpoints"]),
        base_checkpoint_path=bindings["base_checkpoint"]["path"],
        formal_train_cache_path=bindings["state_cache"]["path"],
    )
    loader = build_a4_evaluation_loader(
        cache,
        split="development",
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    decoded = []
    checkpoint_manifest = []
    binding_by_seed = {
        int(row["seed"]): row for row in bindings["checkpoints"]
    }
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
        decoded.append(
            decode_a4_evaluation_arrays(decoder, cache, evaluation.arrays)
        )
        checkpoint_manifest.append(
            {
                "seed": endpoint.seed,
                "step": endpoint.step,
                "selection": endpoint.selection,
                "path": str(endpoint.path),
                "sha256": binding_by_seed[endpoint.seed]["sha256"],
            }
        )
        endpoint.factual.cpu()
        endpoint.no_action.cpu()
        del evaluation
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    bootstrap_samples = (
        SMOKE_BOOTSTRAP_SAMPLES if args.smoke else DEFAULT_BOOTSTRAP_SAMPLES
    )
    report, archive = build_a4_physical_decoding_report(
        cache,
        labels,
        decoder,
        tuple(decoded),
        checkpoint_manifest=checkpoint_manifest,
        fixed_baselines=baselines,
        bootstrap_samples=bootstrap_samples,
    )
    report["run_mode"] = "smoke" if args.smoke else "formal"
    report["formal_result"] = not args.smoke
    report["artifact_bindings"] = bindings
    report["runtime"] = {
        "device": str(device),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "amp_enabled": device.type == "cuda",
    }
    if args.smoke:
        report["smoke_boundary"] = {
            "formal_result": False,
            "minimum_train_physical_episodes_per_task": (
                SMOKE_TRAIN_EPISODES_PER_TASK
            ),
            "train_episode_may_add_one_for_target_identifiability": True,
            "development_physical_episodes_per_task": (
                SMOKE_DEVELOPMENT_EPISODES_PER_TASK
            ),
            "device": "cpu",
        }

    output = _resolve(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=f".{output.name}."
    ) as temporary_name:
        staging = Path(temporary_name) / output.name
        staging.mkdir()
        report_path = staging / "report.json"
        decoded_path = staging / "decoded_predictions.npz"
        decoder_path = staging / "ridge_decoder.npz"
        _write_json(report_path, report)
        with decoded_path.open("wb") as handle:
            np.savez_compressed(handle, **archive)
        with decoder_path.open("wb") as handle:
            np.savez_compressed(handle, **decoder.arrays())
        manifest = {
            "audit_designation": AUDIT_DESIGNATION,
            "run_mode": report["run_mode"],
            "formal_result": report["formal_result"],
            "outputs": {
                path.name: {"sha256": _sha256(path)}
                for path in (report_path, decoded_path, decoder_path)
            },
        }
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, output)

    print(
        json.dumps(
            {
                "output": str(output),
                "run_mode": report["run_mode"],
                "training_seeds": list(CONFIRMATION_TRAINING_SEEDS),
                "audit_designation": AUDIT_DESIGNATION,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
