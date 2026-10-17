#!/usr/bin/env python3
"""Summarize external-grounding rollouts with seed-clustered paired inference."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_paired_rollouts import PAIR_KEYS, exact_mcnemar, row_key
from scripts.summarize_seeded_matrix import parse_checkpoint_seed


@dataclass(frozen=True)
class PairedCell:
    checkpoint_label: str
    policy_seed: int
    task: str
    perturbation: str
    severity: float
    keys: tuple[tuple[str, ...], ...]
    baseline_success: np.ndarray
    candidate_success: np.ndarray

    @property
    def differences(self) -> np.ndarray:
        return self.candidate_success - self.baseline_success


def _resolve_matrix_paths(matrix: Path) -> tuple[Path, Path, Path]:
    source = matrix.expanduser().resolve()
    if source.is_dir():
        root = source
        summary_path = root / "matrix_summary.json"
    elif source.name == "matrix_summary.json":
        root = source.parent
        summary_path = source
    else:
        raise ValueError("--matrix must be a matrix root directory or matrix_summary.json")
    manifest_path = root / "matrix_manifest.json"
    for label, path in (("summary", summary_path), ("manifest", manifest_path)):
        if not path.is_file():
            raise FileNotFoundError(f"matrix {label} does not exist: {path}")
    return root, summary_path, manifest_path


def _resolve_output_dir(raw_path: str, matrix_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    candidates = (REPO_ROOT / path, matrix_root / path)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return candidates[0].resolve()


def _rollout_csv(output_dir: Path) -> Path:
    paths = sorted(output_dir.glob("*_rollout.csv"))
    if len(paths) != 1:
        raise ValueError(
            f"expected exactly one rollout CSV in {output_dir}, found {len(paths)}"
        )
    return paths[0]


def _read_success_rows(path: Path) -> dict[tuple[str, ...], float]:
    rows: dict[tuple[str, ...], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "success" not in reader.fieldnames:
            raise KeyError(f"rollout CSV is missing success: {path}")
        for row in reader:
            key = row_key(row)
            if key in rows:
                raise ValueError(f"duplicate paired episode in {path}: {key}")
            value = float(row["success"])
            if not np.isfinite(value) or value not in (0.0, 1.0):
                raise ValueError(f"success must be binary in {path}: {value}")
            rows[key] = value
    if not rows:
        raise ValueError(f"rollout CSV is empty: {path}")
    return rows


def _summary_row_index(payload: Mapping[str, object]) -> dict[tuple[str, str, str, float, str], dict[str, object]]:
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError("matrix_summary.json does not contain external-grounding rows")
    result: dict[tuple[str, str, str, float, str], dict[str, object]] = {}
    for raw in raw_rows:
        if not isinstance(raw, dict):
            raise TypeError("matrix summary rows must be objects")
        key = (
            str(raw["checkpoint"]),
            str(raw["task"]),
            str(raw["perturbation"]),
            float(raw["severity"]),
            str(raw["method"]),
        )
        if key in result:
            raise ValueError(f"duplicate matrix summary row: {key}")
        result[key] = raw
    return result


def load_paired_cells(
    matrix: Path,
    *,
    baseline: str,
    candidate: str,
) -> tuple[Path, list[PairedCell]]:
    if baseline == candidate:
        raise ValueError("baseline and candidate must differ")
    matrix_root, summary_path, manifest_path = _resolve_matrix_paths(matrix)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary_rows = _summary_row_index(summary)
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("matrix_manifest.json does not contain jobs")

    raw_checkpoints = manifest.get("checkpoints")
    if not isinstance(raw_checkpoints, list) or not raw_checkpoints:
        raise ValueError("matrix manifest checkpoints must be a non-empty list")
    checkpoint_paths: dict[str, str] = {}
    for raw in raw_checkpoints:
        if not isinstance(raw, dict):
            raise TypeError("matrix manifest checkpoints must be objects")
        label = str(raw["label"])
        if not label or label in checkpoint_paths:
            raise ValueError(f"duplicate/empty matrix checkpoint label: {label!r}")
        checkpoint_paths[label] = str(raw["checkpoint"])

    raw_tasks = manifest.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("matrix manifest tasks must be a non-empty list")
    task_names: dict[str, str] = {}
    for raw in raw_tasks:
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError("matrix manifest tasks must use [label, task_name]")
        label, task_name = (str(raw[0]), str(raw[1]))
        if not label or label in task_names:
            raise ValueError(f"duplicate/empty matrix task label: {label!r}")
        task_names[label] = task_name

    raw_severities = manifest.get("severities")
    if not isinstance(raw_severities, list) or not raw_severities:
        raise ValueError("matrix manifest severities must be a non-empty list")
    severities = tuple(float(value) for value in raw_severities)
    if not all(np.isfinite(value) for value in severities) or len(set(severities)) != len(
        severities
    ):
        raise ValueError("matrix manifest severities must be finite and unique")

    raw_methods = manifest.get("methods")
    if not isinstance(raw_methods, list) or not raw_methods:
        raise ValueError("matrix manifest methods must be a non-empty list")
    method_names = tuple(str(value) for value in raw_methods)
    if any(not value for value in method_names) or len(set(method_names)) != len(method_names):
        raise ValueError("matrix manifest methods must be non-empty and unique")
    methods = set(method_names)
    for method in (baseline, candidate):
        if method not in methods:
            raise ValueError(f"method {method!r} is absent from matrix manifest")

    settings = manifest.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("matrix manifest settings must be an object")
    perturbation = str(settings.get("perturbation", ""))
    if not perturbation:
        raise ValueError("matrix manifest settings.perturbation must be non-empty")
    expected_episodes = settings.get("episodes")
    if (
        not isinstance(expected_episodes, int)
        or isinstance(expected_episodes, bool)
        or expected_episodes <= 0
    ):
        raise ValueError("matrix manifest settings.episodes must be a positive integer")
    expected_job_keys = {
        (checkpoint, task, perturbation, severity, method)
        for checkpoint in checkpoint_paths
        for task in task_names
        for severity in severities
        for method in method_names
    }

    job_index: dict[tuple[str, str, str, float, str], dict[str, object]] = {}
    seed_by_label: dict[str, int] = {}
    label_by_seed: dict[int, str] = {}
    for raw in jobs:
        if not isinstance(raw, dict):
            raise TypeError("matrix manifest jobs must be objects")
        checkpoint_label = str(raw["checkpoint_label"])
        if checkpoint_label not in checkpoint_paths:
            raise ValueError(f"job references undeclared checkpoint {checkpoint_label!r}")
        if str(raw["checkpoint"]) != checkpoint_paths[checkpoint_label]:
            raise ValueError(f"job checkpoint path disagrees for {checkpoint_label!r}")
        policy_seed = parse_checkpoint_seed(checkpoint_label)
        previous_seed = seed_by_label.setdefault(checkpoint_label, policy_seed)
        if previous_seed != policy_seed:
            raise ValueError(f"inconsistent policy seed for {checkpoint_label!r}")
        previous_label = label_by_seed.setdefault(policy_seed, checkpoint_label)
        if previous_label != checkpoint_label:
            raise ValueError(
                f"checkpoint labels {previous_label!r} and {checkpoint_label!r} map to seed {policy_seed}"
            )
        task_label = str(raw["task_label"])
        if task_label not in task_names:
            raise ValueError(f"job references undeclared task {task_label!r}")
        if str(raw["task_name"]) != task_names[task_label]:
            raise ValueError(f"job task name disagrees for {task_label!r}")
        key = (
            checkpoint_label,
            task_label,
            str(raw["perturbation"]),
            float(raw["severity"]),
            str(raw["method"]),
        )
        if key in job_index:
            raise ValueError(f"duplicate matrix manifest job: {key}")
        job_index[key] = raw

    actual_job_keys = set(job_index)
    if actual_job_keys != expected_job_keys:
        missing = sorted(expected_job_keys - actual_job_keys)[:5]
        extra = sorted(actual_job_keys - expected_job_keys)[:5]
        raise ValueError(
            "matrix manifest jobs do not form the declared Cartesian design: "
            f"missing={missing} extra={extra}"
        )
    if set(summary_rows) != expected_job_keys:
        missing = sorted(expected_job_keys - set(summary_rows))[:5]
        extra = sorted(set(summary_rows) - expected_job_keys)[:5]
        raise ValueError(
            "matrix summary rows do not match the declared Cartesian design: "
            f"missing={missing} extra={extra}"
        )

    success_rows_by_job: dict[
        tuple[str, str, str, float, str], dict[tuple[str, ...], float]
    ] = {}
    for key in sorted(expected_job_keys):
        raw_job = job_index[key]
        output_dir = _resolve_output_dir(str(raw_job["output_dir"]), matrix_root)
        rows = _read_success_rows(_rollout_csv(output_dir))
        if len(rows) != expected_episodes:
            raise ValueError(
                f"episode count mismatch for {'/'.join(map(str, key))}: "
                f"expected={expected_episodes} actual={len(rows)}"
            )
        summary_rate = float(summary_rows[key]["success_rate"])
        csv_rate = float(np.mean(list(rows.values())))
        if not np.isclose(csv_rate, summary_rate, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"CSV/summary success mismatch for {'/'.join(map(str, key))}: "
                f"csv={csv_rate:.12g} summary={summary_rate:.12g}"
            )
        success_rows_by_job[key] = rows

    cell_prefixes = sorted(
        {key[:4] for key in job_index if key[4] in {baseline, candidate}},
        key=lambda key: (key[2], key[3], key[1], parse_checkpoint_seed(key[0])),
    )
    cells: list[PairedCell] = []
    for checkpoint_label, task, perturbation, severity in cell_prefixes:
        for method in (baseline, candidate):
            key = (checkpoint_label, task, perturbation, severity, method)
            if key not in job_index:
                raise ValueError(f"missing {method!r} job for matrix cell {key[:4]}")
            if key not in summary_rows:
                raise ValueError(f"missing {method!r} summary row for matrix cell {key[:4]}")

        baseline_rows = success_rows_by_job[
            (checkpoint_label, task, perturbation, severity, baseline)
        ]
        candidate_rows = success_rows_by_job[
            (checkpoint_label, task, perturbation, severity, candidate)
        ]
        baseline_keys = set(baseline_rows)
        candidate_keys = set(candidate_rows)
        if baseline_keys != candidate_keys:
            raise ValueError(
                f"pair-key mismatch for {checkpoint_label}/{task}/{perturbation}/{severity:g}: "
                f"missing={len(baseline_keys - candidate_keys)} "
                f"extra={len(candidate_keys - baseline_keys)}"
            )
        keys = tuple(sorted(baseline_keys))
        baseline_values = np.asarray([baseline_rows[key] for key in keys], dtype=np.float64)
        candidate_values = np.asarray([candidate_rows[key] for key in keys], dtype=np.float64)

        cells.append(
            PairedCell(
                checkpoint_label=checkpoint_label,
                policy_seed=seed_by_label[checkpoint_label],
                task=task,
                perturbation=perturbation,
                severity=severity,
                keys=keys,
                baseline_success=baseline_values,
                candidate_success=candidate_values,
            )
        )
    if not cells:
        raise ValueError("matrix contains no comparable baseline/candidate cells")
    by_cell: dict[tuple[str, str, float], list[PairedCell]] = defaultdict(list)
    for cell in cells:
        by_cell[(cell.task, cell.perturbation, cell.severity)].append(cell)
    for key, group in by_cell.items():
        reference = group[0].keys
        for cell in group[1:]:
            if cell.keys != reference:
                raise ValueError(
                    f"cross-seed episode-key mismatch for {key}: "
                    f"reference_seed={group[0].policy_seed} seed={cell.policy_seed}"
                )
    return matrix_root, cells


def policy_seed_episode_bootstrap(
    differences: Mapping[int, Mapping[str, np.ndarray]],
    *,
    tasks: Sequence[str],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Crossed bootstrap of policy seeds and shared physical episodes."""
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    task_names = tuple(tasks)
    if not task_names or len(set(task_names)) != len(task_names):
        raise ValueError("tasks must be a non-empty unique sequence")
    policy_seeds = tuple(sorted(differences))
    if not policy_seeds:
        raise ValueError("at least one policy seed is required")
    arrays: dict[tuple[int, str], np.ndarray] = {}
    episode_counts: dict[str, int] = {}
    for policy_seed in policy_seeds:
        if set(differences[policy_seed]) != set(task_names):
            raise ValueError(
                f"policy seed {policy_seed} does not contain exactly tasks {task_names}"
            )
        for task in task_names:
            values = np.asarray(differences[policy_seed][task], dtype=np.float64)
            if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"bootstrap values must be a non-empty finite vector: seed={policy_seed} task={task}"
                )
            arrays[(policy_seed, task)] = values
            previous_count = episode_counts.setdefault(task, values.size)
            if previous_count != values.size:
                raise ValueError(
                    f"shared episode count mismatch for task={task}: "
                    f"expected={previous_count} seed={policy_seed} actual={values.size}"
                )

    rng = np.random.default_rng(seed)
    estimates: list[np.ndarray] = []
    batch_size = min(5_000, samples)
    num_seeds = len(policy_seeds)
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        selected = rng.integers(0, num_seeds, size=(count, num_seeds))
        occurrence_means = np.zeros((count, num_seeds), dtype=np.float64)
        for task in task_names:
            stacked = np.stack([arrays[(policy_seed, task)] for policy_seed in policy_seeds])
            episode_count = episode_counts[task]
            episode_indices = rng.integers(
                0, episode_count, size=(count, episode_count)
            )
            for position in range(num_seeds):
                selected_values = stacked[selected[:, position]]
                occurrence_means[:, position] += np.take_along_axis(
                    selected_values, episode_indices, axis=1
                ).mean(axis=1)
        occurrence_means /= len(task_names)
        estimates.append(occurrence_means.mean(axis=1))
    distribution = np.concatenate(estimates)
    return (
        float(np.quantile(distribution, 0.025)),
        float(np.quantile(distribution, 0.975)),
    )


def _cell_summary(
    cells: Sequence[PairedCell],
    *,
    baseline: str,
    candidate: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    reference_keys = cells[0].keys
    for cell in cells[1:]:
        if cell.keys != reference_keys:
            raise ValueError(
                f"cross-seed episode-key mismatch for {cells[0].task}/"
                f"{cells[0].perturbation}/{cells[0].severity:g}"
            )
    policy_seeds = [cell.policy_seed for cell in cells]
    if len(set(policy_seeds)) != len(policy_seeds):
        raise ValueError("duplicate policy seed within task/severity cell")
    per_seed = []
    for cell in sorted(cells, key=lambda value: value.policy_seed):
        test = exact_mcnemar(cell.baseline_success, cell.candidate_success)
        per_seed.append(
            {
                "checkpoint_label": cell.checkpoint_label,
                "policy_seed": cell.policy_seed,
                "episodes": len(cell.keys),
                "baseline_success_rate": float(cell.baseline_success.mean()),
                "candidate_success_rate": float(cell.candidate_success.mean()),
                "paired_success_difference": float(cell.differences.mean()),
                "mcnemar_exact": {
                    "baseline_only_success": test["left_only_success"],
                    "candidate_only_success": test["right_only_success"],
                    "discordant": test["discordant"],
                    "p_value_two_sided": test["p_value_two_sided"],
                },
            }
        )
    baseline_values = np.concatenate([cell.baseline_success for cell in cells])
    candidate_values = np.concatenate([cell.candidate_success for cell in cells])
    seed_macro_baseline = float(np.mean([item["baseline_success_rate"] for item in per_seed]))
    seed_macro_candidate = float(np.mean([item["candidate_success_rate"] for item in per_seed]))
    differences = {
        cell.policy_seed: {cell.task: cell.differences}
        for cell in cells
    }
    ci = policy_seed_episode_bootstrap(
        differences,
        tasks=(cells[0].task,),
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    return {
        "task": cells[0].task,
        "perturbation": cells[0].perturbation,
        "severity": cells[0].severity,
        "baseline": baseline,
        "candidate": candidate,
        "policy_seeds": sorted(policy_seeds),
        "per_seed": per_seed,
        "pooled_episode": {
            "episodes": int(baseline_values.size),
            "baseline_success_rate": float(baseline_values.mean()),
            "candidate_success_rate": float(candidate_values.mean()),
            "paired_success_difference": float((candidate_values - baseline_values).mean()),
        },
        "policy_seed_macro": {
            "baseline_success_rate": seed_macro_baseline,
            "candidate_success_rate": seed_macro_candidate,
            "paired_success_difference": seed_macro_candidate - seed_macro_baseline,
            "hierarchical_bootstrap_95_ci": list(ci),
        },
    }


def summarize_cells(
    cells: Sequence[PairedCell],
    *,
    baseline: str,
    candidate: str,
    bootstrap_samples: int = 100_000,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    if not cells:
        raise ValueError("at least one paired cell is required")
    grouped: dict[tuple[str, str, float], list[PairedCell]] = defaultdict(list)
    for cell in cells:
        grouped[(cell.task, cell.perturbation, cell.severity)].append(cell)
    task_cells = [
        _cell_summary(
            grouped[key],
            baseline=baseline,
            candidate=candidate,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + index,
        )
        for index, key in enumerate(sorted(grouped))
    ]

    macro_groups: dict[tuple[str, float], list[PairedCell]] = defaultdict(list)
    for cell in cells:
        macro_groups[(cell.perturbation, cell.severity)].append(cell)
    task_macro = []
    for index, ((perturbation, severity), group) in enumerate(sorted(macro_groups.items())):
        tasks = tuple(sorted({cell.task for cell in group}))
        seeds_by_task = {
            task: {cell.policy_seed for cell in group if cell.task == task}
            for task in tasks
        }
        seed_sets = list(seeds_by_task.values())
        if any(values != seed_sets[0] for values in seed_sets[1:]):
            raise ValueError(
                f"task-macro seed mismatch for {perturbation}/{severity:g}: {seeds_by_task}"
            )
        policy_seeds = sorted(seed_sets[0])
        by_seed_task = {(cell.policy_seed, cell.task): cell for cell in group}
        differences = {
            policy_seed: {
                task: by_seed_task[(policy_seed, task)].differences for task in tasks
            }
            for policy_seed in policy_seeds
        }
        per_seed = []
        for policy_seed in policy_seeds:
            task_differences = {
                task: float(by_seed_task[(policy_seed, task)].differences.mean())
                for task in tasks
            }
            per_seed.append(
                {
                    "policy_seed": policy_seed,
                    "task_differences": task_differences,
                    "task_macro_paired_success_difference": float(
                        np.mean(list(task_differences.values()))
                    ),
                }
            )
        task_pooled_differences = {
            task: float(
                np.concatenate(
                    [cell.differences for cell in group if cell.task == task]
                ).mean()
            )
            for task in tasks
        }
        ci = policy_seed_episode_bootstrap(
            differences,
            tasks=tasks,
            samples=bootstrap_samples,
            seed=bootstrap_seed + 10_000 + index,
        )
        task_macro.append(
            {
                "perturbation": perturbation,
                "severity": severity,
                "tasks": list(tasks),
                "policy_seeds": policy_seeds,
                "per_seed": per_seed,
                "equal_task_macro_of_pooled_episode_differences": float(
                    np.mean(list(task_pooled_differences.values()))
                ),
                "equal_task_equal_seed_macro_paired_success_difference": float(
                    np.mean(
                        [item["task_macro_paired_success_difference"] for item in per_seed]
                    )
                ),
                "hierarchical_bootstrap_95_ci": list(ci),
            }
        )
    return {
        "schema_version": 1,
        "comparison": {"baseline": baseline, "candidate": candidate},
        "pair_keys": list(PAIR_KEYS),
        "inference": {
            "primary_estimand": "equal-policy-seed mean of within-seed paired episode differences",
            "bootstrap": (
                "crossed resampling of policy seeds and shared paired physical-episode "
                "identities; each episode draw is reused across selected seeds"
            ),
            "task_macro": "fixed task set with equal task weight in every bootstrap replicate",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
        },
        "task_cells": task_cells,
        "task_macro": task_macro,
    }


def render_markdown(summary: Mapping[str, object]) -> str:
    comparison = summary["comparison"]
    baseline = str(comparison["baseline"])
    candidate = str(comparison["candidate"])
    lines = [
        "# External Grounding Paired Multi-Seed Summary",
        "",
        f"Comparison: `{candidate}` - `{baseline}`.",
        "",
        "| task | perturbation | severity | seeds | pooled episodes | baseline | candidate | delta | hierarchical 95% CI |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for cell in summary["task_cells"]:
        pooled = cell["pooled_episode"]
        macro = cell["policy_seed_macro"]
        ci = macro["hierarchical_bootstrap_95_ci"]
        lines.append(
            f"| {cell['task']} | {cell['perturbation']} | {float(cell['severity']):g} | "
            f"{len(cell['policy_seeds'])} | {pooled['episodes']} | "
            f"{macro['baseline_success_rate']:.4f} | {macro['candidate_success_rate']:.4f} | "
            f"{macro['paired_success_difference']:+.4f} | "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Equal-Task Macro",
            "",
            "| perturbation | severity | tasks | seeds | delta | hierarchical 95% CI |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for cell in summary["task_macro"]:
        ci = cell["hierarchical_bootstrap_95_ci"]
        lines.append(
            f"| {cell['perturbation']} | {float(cell['severity']):g} | {len(cell['tasks'])} | "
            f"{len(cell['policy_seeds'])} | "
            f"{cell['equal_task_equal_seed_macro_paired_success_difference']:+.4f} | "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "The primary interval uses crossed resampling of policy seeds and shared deterministic "
            "paired-episode identities. Tasks are fixed and equally weighted in the task-macro result.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        required=True,
        type=Path,
        help="external-grounding matrix root or its matrix_summary.json",
    )
    parser.add_argument("--baseline", default="native")
    parser.add_argument("--candidate", default="prototype")
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    matrix_root, cells = load_paired_cells(
        args.matrix,
        baseline=args.baseline,
        candidate=args.candidate,
    )
    summary = summarize_cells(
        cells,
        baseline=args.baseline,
        candidate=args.candidate,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    summary["source"] = {
        "matrix_root": str(matrix_root),
        "matrix_summary": str(matrix_root / "matrix_summary.json"),
        "matrix_manifest": str(matrix_root / "matrix_manifest.json"),
    }
    output_dir = args.output_dir or matrix_root / f"paired_seeds_{args.candidate}_vs_{args.baseline}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "paired_seed_summary.json"
    markdown_path = output_dir / "paired_seed_summary.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    print(markdown_path.read_text(encoding="utf-8"))
    print(f"json: {json_path}")
    print(f"markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
