#!/usr/bin/env python3
"""Train and evaluate the bounded current-only VisualTrust probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.visual_trust_metrics import (  # noqa: E402
    clean_false_reject_rate,
    recovery_detection_delay,
    summarize_visual_trust,
)
from fcpg.visual_trust_corruptions import CORRUPTION_TYPES  # noqa: E402
from fcpg.visual_trust_training import (  # noqa: E402
    DEFAULT_UNSAFE_RMS_THRESHOLD,
    VisualTrustSplit,
    load_visual_trust_split,
    predict_visual_trust,
    temporal_feature_distance,
    threshold_for_clean_false_reject,
    train_visual_trust_head,
)


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be unique non-negative integers")
    return seeds


def _cache_file(path: Path, filename: str) -> Path:
    return path / filename if path.is_dir() else path


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _finite_or_none(float(value))
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _classification_summary(
    split: VisualTrustSplit,
    trust: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    mask = split.valid.reshape(-1)
    return summarize_visual_trust(
        trust.reshape(-1)[mask],
        split.unsafe.reshape(-1)[mask],
        split.clean.reshape(-1)[mask],
        split.early_reopen.reshape(-1)[mask],
        split.task.reshape(-1)[mask],
        split.family.reshape(-1)[mask],
        threshold=threshold,
    )


def _physical_identities(path: Path) -> set[tuple[str, int, int]]:
    """Read simple physical episode identities used to enforce split isolation."""

    with np.load(path, allow_pickle=False) as data:
        required = {"task_name", "env_seed", "episode_id"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"identity arrays missing from {path}: {missing}")
        task = data["task_name"].astype(str)
        env_seed = data["env_seed"].astype(np.int64)
        episode = data["episode_id"].astype(np.int64)
    if task.shape != env_seed.shape or task.shape != episode.shape:
        raise ValueError(f"identity arrays must have matching shapes in {path}")
    return {
        (str(task_name), int(seed), int(episode_id))
        for task_name, seed, episode_id in zip(
            task, env_seed, episode, strict=True
        )
    }


def _assert_disjoint_physical_splits(
    train_path: Path,
    development_path: Path,
    clean_path: Path,
) -> dict[str, int]:
    identities = {
        "train": _physical_identities(train_path),
        "development": _physical_identities(development_path),
        "clean": _physical_identities(clean_path),
    }
    names = tuple(identities)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = identities[left].intersection(identities[right])
            if overlap:
                sample = sorted(overlap)[:3]
                raise ValueError(
                    f"physical episode overlap between {left} and {right}: {sample}"
                )
    return {name: len(value) for name, value in identities.items()}


def _clean_frr_by_task(
    split: VisualTrustSplit,
    trust: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    result: dict[str, float] = {}
    tasks = split.task.reshape(-1).astype(str)
    valid = split.valid.reshape(-1)
    values = trust.reshape(-1)
    for task in sorted(set(tasks)):
        selected = valid & (tasks == task)
        if not bool(np.any(selected)):
            raise ValueError(f"clean split has no valid anchors for task {task}")
        result[task] = clean_false_reject_rate(
            values[selected],
            np.ones(int(selected.sum()), dtype=bool),
            threshold=threshold,
        )
    return result


def _recovery_summary(
    split: VisualTrustSplit,
    trust: np.ndarray,
    threshold: float,
    *,
    min_consecutive: int,
) -> dict[str, object]:
    tasks = split.task[:, 0].astype(str)
    episodes = split.episode[:, 0].astype(np.int64)
    frames = split.frame[:, 0].astype(np.int64)
    delays: list[float] = []
    event_tasks: list[str] = []
    event_families: list[str] = []
    total_event_count = 0
    ineligible_event_count = 0
    for task, episode in sorted(set(zip(tasks, episodes, strict=True))):
        rows = np.flatnonzero((tasks == task) & (episodes == episode))
        rows = rows[np.argsort(frames[rows], kind="stable")]
        event_positions = np.argwhere(split.recovery_onset[rows])
        if event_positions.shape[0] != 1:
            raise ValueError(
                f"episode {(task, int(episode))} must have one recovery onset"
            )
        recovery_position, anchor = (int(value) for value in event_positions[0])
        total_event_count += 1
        actually_unsafe = (
            split.active[rows, anchor]
            & split.unsafe[rows, anchor]
            & split.valid[rows, anchor]
        )
        if not bool(np.any(actually_unsafe)):
            ineligible_event_count += 1
            continue
        sequence = np.where(
            split.valid[rows, anchor], trust[rows, anchor], 0.0
        )
        delay = recovery_detection_delay(
            sequence,
            threshold=threshold,
            min_consecutive=min_consecutive,
            recovery_start=recovery_position,
        )
        delays.append(delay)
        event_tasks.append(str(task))
        event_families.append(str(split.family[rows[0], anchor]))

    delay_array = np.asarray(delays, dtype=np.float64)
    detected = np.isfinite(delay_array)
    finite = delay_array[detected]

    def grouped(labels: list[str]) -> dict[str, object]:
        result: dict[str, object] = {}
        label_array = np.asarray(labels)
        for label in sorted(set(labels)):
            selected = delay_array[label_array == label]
            selected_finite = selected[np.isfinite(selected)]
            result[label] = {
                "event_count": int(selected.size),
                "detection_rate": float(np.mean(np.isfinite(selected))),
                "median_delay": (
                    float(np.median(selected_finite))
                    if selected_finite.size
                    else None
                ),
            }
        return result

    return {
        "event_count": int(total_event_count),
        "eligible_event_count": int(delay_array.size),
        "ineligible_event_count": int(ineligible_event_count),
        "eligible_rate": (
            float(delay_array.size / total_event_count)
            if total_event_count
            else 0.0
        ),
        "detection_rate": float(np.mean(detected)),
        "median_delay": float(np.median(finite)) if finite.size else None,
        "p90_delay": float(np.quantile(finite, 0.9)) if finite.size else None,
        "per_task": grouped(event_tasks),
        "per_family": grouped(event_families),
    }


def _fit_ensemble(
    train: VisualTrustSplit,
    development: VisualTrustSplit,
    *,
    seeds: tuple[int, ...],
    steps: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    visual_enabled: bool,
    output_dir: Path,
    name: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    train_predictions: list[np.ndarray] = []
    development_predictions: list[np.ndarray] = []
    parameter_counts: list[int] = []
    for seed in seeds:
        model = train_visual_trust_head(
            train.visual,
            train.auxiliary,
            train.safe_target,
            train.valid,
            seed=seed,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            visual_enabled=visual_enabled,
        )
        train_predictions.append(
            predict_visual_trust(
                model,
                train.visual,
                train.auxiliary,
                batch_size=batch_size,
                device=device,
                visual_enabled=visual_enabled,
            )
        )
        development_predictions.append(
            predict_visual_trust(
                model,
                development.visual,
                development.auxiliary,
                batch_size=batch_size,
                device=device,
                visual_enabled=visual_enabled,
            )
        )
        parameter_counts.append(model.parameter_count)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "visual_dim": model.visual_dim,
                "auxiliary_dim": model.auxiliary_dim,
                "hidden_dim": model.hidden_dim,
                "visual_enabled": visual_enabled,
                "seed": seed,
            },
            output_dir / f"{name}_seed{seed}.pt",
        )
    return train_predictions, development_predictions, parameter_counts


def _temporal_trust(
    train: VisualTrustSplit, development: VisualTrustSplit
) -> tuple[np.ndarray, np.ndarray]:
    train_distance = temporal_feature_distance(train)
    development_distance = temporal_feature_distance(development)
    positive = train_distance[train_distance > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, 1e-6)
    return np.exp(-train_distance / scale), np.exp(-development_distance / scale)


def _decision(
    seed_results: list[dict[str, object]],
) -> dict[str, object]:
    discrimination_passes = 0
    safety_passes = 0
    for result in seed_results:
        summaries = result["classification"]
        visual = summaries["visual"]
        overall = visual["overall"]
        per_family = visual["per_family"]
        expected_families = set(CORRUPTION_TYPES)
        complete_family_coverage = set(per_family) == expected_families
        family_auc = [float(value["auroc"]) for value in per_family.values()]
        family_recall = [
            float(value["unsafe_recall"]) for value in per_family.values()
        ]
        baseline_auc = max(
            float(summaries["geometry"]["overall"]["auroc"]),
            float(summaries["temporal"]["overall"]["auroc"]),
            float(summaries["auxiliary_only"]["overall"]["auroc"]),
        )
        discrimination = (
            float(overall["auroc"]) >= 0.85
            and complete_family_coverage
            and len(family_auc) == len(expected_families)
            and all(np.isfinite(value) for value in family_auc + family_recall)
            and min(family_auc) >= 0.80
            and min(family_recall) >= 0.80
            and float(overall["unsafe_recall"]) >= 0.80
            and float(overall["auroc"]) >= baseline_auc + 0.03
        )
        recovery = result["recovery"]
        clean_by_task = result["independent_clean_false_reject_rate_by_task"]
        family_early_reopen = [
            float(value["early_reopen_rate"]) for value in per_family.values()
        ]
        recovery_families = recovery["per_family"]
        recovery_tasks = recovery["per_task"]
        recovery_groups = [
            *recovery_families.values(),
            *recovery_tasks.values(),
        ]
        safety = (
            float(result["independent_clean_false_reject_rate"]) <= 0.05
            and bool(clean_by_task)
            and max(float(value) for value in clean_by_task.values()) <= 0.05
            and float(overall["early_reopen_rate"]) <= 0.10
            and complete_family_coverage
            and bool(family_early_reopen)
            and max(family_early_reopen) <= 0.10
            and float(recovery["eligible_rate"]) >= 0.90
            and float(recovery["detection_rate"]) >= 0.90
            and recovery["median_delay"] is not None
            and float(recovery["median_delay"]) <= 2.0
            and set(recovery_families) == expected_families
            and bool(recovery_groups)
            and all(
                float(group["detection_rate"]) >= 0.90
                and group["median_delay"] is not None
                and float(group["median_delay"]) <= 2.0
                for group in recovery_groups
            )
        )
        discrimination_passes += int(discrimination)
        safety_passes += int(safety)
        result["discrimination_pass"] = discrimination
        result["safety_pass"] = safety
    checks = {
        "discrimination_passes_at_least_2_of_3": discrimination_passes >= 2,
        "safety_passes_all_3": safety_passes == len(seed_results),
        "exactly_3_training_seeds": len(seed_results) == 3,
    }
    return {
        "go": all(checks.values()),
        "status": "Development-Go" if all(checks.values()) else "Development-No-Go",
        "checks": checks,
        "discrimination_pass_count": discrimination_passes,
        "safety_pass_count": safety_passes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "development"):
        parser.add_argument(f"--{split}-data", type=Path, required=True)
        parser.add_argument(f"--{split}-observed-visual", type=Path, required=True)
        parser.add_argument(f"--{split}-canonical-visual", type=Path, required=True)
        parser.add_argument(f"--{split}-observed-grounding", type=Path, required=True)
    parser.add_argument("--clean-data", type=Path, required=True)
    parser.add_argument("--clean-visual", type=Path, required=True)
    parser.add_argument("--clean-grounding", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=_parse_seeds, default=(0, 1, 2))
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--unsafe-rms-threshold", type=float, default=DEFAULT_UNSAFE_RMS_THRESHOLD)
    parser.add_argument("--target-clean-false-reject", type=float, default=0.05)
    parser.add_argument("--min-recovery-consecutive", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.steps <= 1000 or args.batch_size <= 0:
        raise ValueError("steps must lie in [1,1000] and batch-size must be positive")
    if not 0.0 < args.target_clean_false_reject < 1.0:
        raise ValueError("target-clean-false-reject must lie in (0,1)")
    if args.min_recovery_consecutive <= 0:
        raise ValueError("min-recovery-consecutive must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    identity_counts = _assert_disjoint_physical_splits(
        args.train_data, args.development_data, args.clean_data
    )

    def load(split: str) -> VisualTrustSplit:
        return load_visual_trust_split(
            getattr(args, f"{split}_data"),
            _cache_file(getattr(args, f"{split}_observed_visual"), "sampled_visual.npy"),
            _cache_file(getattr(args, f"{split}_canonical_visual"), "sampled_visual.npy"),
            _cache_file(getattr(args, f"{split}_observed_grounding"), "q_corrected.npy"),
            unsafe_rms_threshold=args.unsafe_rms_threshold,
        )

    train = load("train")
    development = load("development")
    clean = load_visual_trust_split(
        args.clean_data,
        _cache_file(args.clean_visual, "sampled_visual.npy"),
        _cache_file(args.clean_visual, "sampled_visual.npy"),
        _cache_file(args.clean_grounding, "q_corrected.npy"),
        unsafe_rms_threshold=args.unsafe_rms_threshold,
        clean_only=True,
    )
    device = torch.device(args.device)

    train_predictions, development_predictions, parameter_counts = _fit_ensemble(
        train,
        development,
        seeds=args.seeds,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        visual_enabled=True,
        output_dir=args.output_dir,
        name="visual",
    )
    (
        auxiliary_train_predictions,
        auxiliary_development_predictions,
        auxiliary_parameter_counts,
    ) = _fit_ensemble(
        train,
        development,
        seeds=args.seeds,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        visual_enabled=False,
        output_dir=args.output_dir,
        name="auxiliary_only",
    )
    if auxiliary_parameter_counts != parameter_counts:
        raise ValueError("auxiliary-only control must be parameter matched")
    clean_predictions = []
    for seed in args.seeds:
        checkpoint = torch.load(
            args.output_dir / f"visual_seed{seed}.pt",
            map_location=device,
            weights_only=False,
        )
        from fcpg.visual_trust import VisualTrustHead

        model = VisualTrustHead(
            checkpoint["visual_dim"],
            checkpoint["auxiliary_dim"],
            checkpoint["hidden_dim"],
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        clean_predictions.append(
            predict_visual_trust(
                model,
                clean.visual,
                clean.auxiliary,
                batch_size=args.batch_size,
                device=device,
                visual_enabled=True,
            )
        )

    geometry_train = np.exp(-train.auxiliary[..., 4])
    geometry_development = np.exp(-development.auxiliary[..., 4])
    temporal_train, temporal_development = _temporal_trust(train, development)
    baseline_predictions = {
        "geometry": (geometry_train, geometry_development),
        "temporal": (temporal_train, temporal_development),
    }
    baseline_summaries: dict[str, dict[str, object]] = {}
    baseline_thresholds: dict[str, float] = {}
    for name, (train_trust, development_trust) in baseline_predictions.items():
        threshold = threshold_for_clean_false_reject(
            train_trust,
            train.clean,
            train.valid,
            target_rate=args.target_clean_false_reject,
        )
        baseline_thresholds[name] = threshold
        baseline_summaries[name] = _classification_summary(
            development, development_trust, threshold
        )

    seed_results: list[dict[str, object]] = []
    for index, seed in enumerate(args.seeds):
        train_trust = np.where(
            train.valid, train_predictions[index], 0.0
        )
        development_trust = development_predictions[index]
        clean_trust = clean_predictions[index]
        development_trust = np.where(
            development.valid, development_trust, 0.0
        )
        clean_trust = np.where(clean.valid, clean_trust, 0.0)
        auxiliary_train_trust = np.where(
            train.valid, auxiliary_train_predictions[index], 0.0
        )
        auxiliary_development_trust = np.where(
            development.valid, auxiliary_development_predictions[index], 0.0
        )
        threshold = threshold_for_clean_false_reject(
            train_trust,
            train.clean,
            train.valid,
            target_rate=args.target_clean_false_reject,
        )
        classification = {
            "visual": _classification_summary(
                development, development_trust, threshold
            ),
            "auxiliary_only": _classification_summary(
                development,
                auxiliary_development_trust,
                threshold_for_clean_false_reject(
                    auxiliary_train_trust,
                    train.clean,
                    train.valid,
                    target_rate=args.target_clean_false_reject,
                ),
            ),
            **baseline_summaries,
        }
        recovery = _recovery_summary(
            development,
            development_trust,
            threshold,
            min_consecutive=args.min_recovery_consecutive,
        )
        clean_mask = clean.valid.reshape(-1)
        clean_frr = clean_false_reject_rate(
            clean_trust.reshape(-1)[clean_mask],
            np.ones(int(clean_mask.sum()), dtype=bool),
            threshold=threshold,
        )
        clean_frr_by_task = _clean_frr_by_task(clean, clean_trust, threshold)
        seed_results.append(
            {
                "seed": seed,
                "threshold": threshold,
                "classification": classification,
                "independent_clean_false_reject_rate": clean_frr,
                "independent_clean_false_reject_rate_by_task": clean_frr_by_task,
                "recovery": recovery,
            }
        )

    decision = _decision(seed_results)
    result = {
        "decision": decision,
        "label": {
            "canonical_feature_rms_threshold": args.unsafe_rms_threshold,
            "canonical_is_label_only": True,
        },
        "training": {
            "seeds": list(args.seeds),
            "seed_role": "optimization_and_minibatch_sampling_only",
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "parameter_counts": parameter_counts,
            "auxiliary_only_parameter_counts": auxiliary_parameter_counts,
        },
        "physical_episode_counts": identity_counts,
        "baseline_operating_thresholds": baseline_thresholds,
        "per_seed": seed_results,
    }
    safe_result = _json_safe(result)
    (args.output_dir / "summary.json").write_text(
        json.dumps(safe_result, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "development_predictions.npz",
        visual=np.stack(development_predictions),
        auxiliary_only=np.stack(auxiliary_development_predictions),
        geometry=geometry_development,
        temporal=temporal_development,
        unsafe=development.unsafe,
        clean=development.clean,
        active=development.active,
        canonical_rms=development.canonical_rms,
    )
    print(json.dumps(safe_result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
