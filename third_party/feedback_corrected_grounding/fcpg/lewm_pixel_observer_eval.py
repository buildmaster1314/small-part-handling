"""Independent offline evaluation for the pixel-innovation LeWM observer.

This module deliberately does not reuse the legacy grounded-LeWM gate.  That
gate measures latent prediction, whereas PixelObserver-G0 measures per-anchor
pixel endpoint error after real-observation filtering.  Oracle pixels are
attached by an evaluator-only dataset wrapper and are stripped before any
observer method is called.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from fcpg.lewm_pixel_observer import LeWMPixelInnovationObserver, fixed_gain_update
from fcpg.lewm_provenance import sha256_file


PROTOCOL = "lewm_pixel_observer_g0_filter_v1"
SCHEMA_VERSION = 1
DEVELOPMENT_ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
FUSION_ALPHA_GRID = (0.25, 0.5, 0.75)
ENDPOINT_HORIZONS = (1, 3, 5)
CONFIRMATORY_COHORTS_BY_SOURCE = {
    0: "rotation5",
    1: "translation4",
}
EVALUATOR_ONLY_KEYS = frozenset(
    {"q_true", "valid_true", "valid_nominal_eval"}
)
LEGACY_OUTPUT_ROOTS = (
    Path("outputs/lewm_grounded"),
    Path("outputs/lewm_grounded_eval"),
)


def _inside(points: torch.Tensor, *, width: int, height: int) -> torch.Tensor:
    return (
        torch.isfinite(points).all(dim=-1)
        & (points[..., 0] >= 0.0)
        & (points[..., 0] <= float(width - 1))
        & (points[..., 1] >= 0.0)
        & (points[..., 1] <= float(height - 1))
    )


def validate_protocol(
    *, phase: str, alphas: Sequence[float], horizons: Sequence[int]
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Validate the fixed development grid or a single frozen test gain."""

    normalized_phase = str(phase).lower()
    if normalized_phase not in {"development", "confirmatory"}:
        raise ValueError("phase must be 'development' or 'confirmatory'")
    selected = tuple(float(value) for value in alphas)
    if not selected or not all(math.isfinite(value) for value in selected):
        raise ValueError("alphas must be a non-empty finite sequence")
    if any(value < 0.0 or value > 1.0 for value in selected):
        raise ValueError("all alphas must lie in [0, 1]")
    if len(set(selected)) != len(selected):
        raise ValueError("alphas must not contain duplicates")
    if normalized_phase == "development" and selected != DEVELOPMENT_ALPHA_GRID:
        raise ValueError(
            f"development requires the frozen alpha grid {DEVELOPMENT_ALPHA_GRID}"
        )
    if normalized_phase == "confirmatory":
        if len(selected) != 1:
            raise ValueError("confirmatory evaluation requires exactly one frozen alpha")
        if selected[0] not in FUSION_ALPHA_GRID:
            raise ValueError(
                "confirmatory fusion alpha must be one of "
                f"{FUSION_ALPHA_GRID}; endpoints are diagnostic only"
            )
    endpoint_horizons = tuple(int(value) for value in horizons)
    if endpoint_horizons != ENDPOINT_HORIZONS:
        raise ValueError(f"PixelObserver-G0 requires horizons {ENDPOINT_HORIZONS}")
    return selected, endpoint_horizons


def reject_legacy_output_path(path: str | Path, repo_root: str | Path) -> Path:
    """Reject every destination inside either legacy grounded-LeWM output tree."""

    destination = Path(path).expanduser()
    root = Path(repo_root).expanduser().resolve()
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    if any("gate_l1" in part.lower() for part in destination.parts):
        raise ValueError(
            "PixelObserver-G0 output path must not contain a legacy Gate L1 "
            f"component: {destination}"
        )
    for relative in LEGACY_OUTPUT_ROOTS:
        legacy = (root / relative).resolve()
        if destination == legacy or legacy in destination.parents:
            raise ValueError(
                "PixelObserver-G0 output must not be written into the legacy "
                f"Gate L1 tree: {destination}"
            )
    return destination


def deployment_batch(
    batch: Mapping[str, object], device: torch.device | str
) -> dict[str, object]:
    """Move deployment tensors while removing all evaluator-only oracle fields."""

    target = torch.device(device)
    result = {
        key: value.to(target, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
        if key not in EVALUATOR_ONLY_KEYS
    }
    leaked = EVALUATOR_ONLY_KEYS.intersection(result)
    if leaked:
        raise AssertionError(f"evaluator-only fields leaked into deployment batch: {leaked}")
    return result


class EvaluatorOracleDataset(Dataset):
    """Attach raw NPZ oracle pixels using only the sample's recorded raw indices."""

    def __init__(self, dataset: Dataset, data_path: str | Path) -> None:
        self.dataset = dataset
        self.data_path = Path(data_path).expanduser().resolve()
        with np.load(self.data_path, allow_pickle=False) as raw:
            missing = {"q_true", "valid_true", "valid_nominal"}.difference(raw.files)
            if missing:
                raise KeyError(f"oracle evaluation source is missing {sorted(missing)}")
            self.q_true = np.asarray(raw["q_true"], dtype=np.float32)
            self.valid_true = np.asarray(raw["valid_true"], dtype=bool)
            self.valid_nominal = np.asarray(raw["valid_nominal"], dtype=bool)
        if self.q_true.ndim != 3 or self.q_true.shape[-1] != 2:
            raise ValueError("q_true must have shape [N, K, 2]")
        expected = self.q_true.shape[:-1]
        if self.valid_true.shape != expected or self.valid_nominal.shape != expected:
            raise ValueError("oracle validity arrays must match q_true anchors")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = dict(self.dataset[index])
        if any(key in sample for key in EVALUATOR_ONLY_KEYS):
            raise ValueError("base LeWM sample unexpectedly contains evaluator oracle fields")
        raw_index = sample.get("raw_index")
        if not torch.is_tensor(raw_index) or raw_index.ndim != 1:
            raise KeyError("oracle wrapper requires one-dimensional sample raw_index")
        indices = raw_index.detach().cpu().numpy().astype(np.int64, copy=False)
        if indices.size == 0 or indices.min() < 0 or indices.max() >= len(self.q_true):
            raise IndexError("sample raw_index is outside the oracle source")
        sample["q_true"] = torch.from_numpy(
            np.array(self.q_true[indices], dtype=np.float32, copy=True)
        )
        sample["valid_true"] = torch.from_numpy(
            np.array(self.valid_true[indices], dtype=bool, copy=True)
        )
        sample["valid_nominal_eval"] = torch.from_numpy(
            np.array(self.valid_nominal[indices], dtype=bool, copy=True)
        )
        return sample


def hold_last_filter(
    measurement_residual_px: torch.Tensor,
    measurement_valid: torch.Tensor,
    *,
    history_size: int,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a fixed-gain EMA whose process prior is the previous posterior."""

    if measurement_residual_px.ndim != 4 or measurement_residual_px.shape[-1] != 2:
        raise ValueError("measurement residual must have shape [B, T, K, 2]")
    if measurement_valid.shape != measurement_residual_px.shape[:-1]:
        raise ValueError("measurement_valid must have shape [B, T, K]")
    if history_size <= 0 or measurement_residual_px.shape[1] <= history_size:
        raise ValueError("filter requires at least history_size + 1 frames")
    posterior = torch.zeros_like(measurement_residual_px[:, 0])
    for index in range(history_size):
        _, posterior, _ = fixed_gain_update(
            posterior,
            measurement_residual_px[:, index],
            1.0,
            measurement_valid[:, index],
        )
    priors = []
    posteriors = []
    gains = []
    for index in range(history_size, measurement_residual_px.shape[1]):
        prior = posterior
        _, posterior, effective_gain = fixed_gain_update(
            prior,
            measurement_residual_px[:, index],
            alpha,
            measurement_valid[:, index],
        )
        priors.append(prior)
        posteriors.append(posterior)
        gains.append(effective_gain)
    return torch.stack(priors, dim=1), torch.stack(posteriors, dim=1), torch.stack(gains, dim=1)


@dataclass
class _EpisodeErrors:
    source_index: int
    task: str
    episode: int
    errors: list[float] = field(default_factory=list)
    squared_errors: list[float] = field(default_factory=list)
    measurement_valid: int = 0
    eligible: int = 0
    updates: int = 0
    window_count: int = 0
    possible_anchor_count: int = 0
    contributing_window_count: int = 0

    def update(
        self,
        errors: torch.Tensor,
        eligible: torch.Tensor,
        measurement_valid: torch.Tensor,
        effective_gain: torch.Tensor,
    ) -> None:
        selected = errors[eligible].detach().double().cpu().numpy()
        self.errors.extend(float(value) for value in selected.tolist())
        self.squared_errors.extend(float(value * value) for value in selected.tolist())
        self.eligible += int(eligible.sum().item())
        self.measurement_valid += int((eligible & measurement_valid).sum().item())
        update_mask = torch.linalg.vector_norm(effective_gain, dim=-1) > 0.0
        self.updates += int((eligible & update_mask).sum().item())
        self.window_count += 1
        self.possible_anchor_count += int(eligible.numel())
        self.contributing_window_count += int(bool(eligible.any()))

    def metrics(self) -> dict[str, object]:
        if not self.errors or self.eligible <= 0:
            raise ValueError(
                f"episode {self.task}:{self.episode} contains no eligible anchors"
            )
        values = np.asarray(self.errors, dtype=np.float64)
        squared = np.asarray(self.squared_errors, dtype=np.float64)
        return {
            "source_index": self.source_index,
            "task": self.task,
            "episode": self.episode,
            "window_count": self.window_count,
            "anchor_count": self.eligible,
            "mean_epe_px": float(values.mean()),
            "rmse_px": float(np.sqrt(squared.mean())),
            "possible_window_count": self.window_count,
            "contributing_window_count": self.contributing_window_count,
            "contributing_window_fraction": (
                self.contributing_window_count / self.window_count
            ),
            "possible_anchor_count": self.possible_anchor_count,
            "eligible_anchor_count": self.eligible,
            "eligible_anchor_fraction": self.eligible / self.possible_anchor_count,
            "median_epe_px": float(np.median(values)),
            "p95_epe_px": float(np.quantile(values, 0.95)),
            "measurement_valid_anchor_count": self.measurement_valid,
            "measurement_eligible_anchor_count": self.eligible,
            "measurement_coverage": self.measurement_valid / self.eligible,
            "update_fraction": self.updates / self.eligible,
        }


def _metadata_list(value: object, size: int, name: str) -> list[object]:
    if torch.is_tensor(value):
        items = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = list(value)
    else:
        items = [value] * size
    if len(items) != size:
        raise ValueError(f"{name} metadata length differs from batch size")
    return items


def _alpha_label(alpha: float) -> str:
    return format(float(alpha), ".12g")


def _cell_id(prior: str, measurement: str, alpha: float) -> str:
    return f"prior={prior}|measurement={measurement}|alpha={_alpha_label(alpha)}"


def _support_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate support from raw counts rather than averaging fractions."""

    possible_anchors = sum(int(row.get("possible_anchor_count", 0)) for row in rows)
    eligible_anchors = sum(int(row.get("eligible_anchor_count", 0)) for row in rows)
    possible_windows = sum(int(row.get("possible_window_count", 0)) for row in rows)
    contributing_windows = sum(
        int(row.get("contributing_window_count", 0)) for row in rows
    )
    measurement_eligible = sum(
        int(row.get("measurement_eligible_anchor_count", 0)) for row in rows
    )
    measurement_valid = sum(
        int(row.get("measurement_valid_anchor_count", 0)) for row in rows
    )

    def fraction(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator > 0 else None

    return {
        "possible_anchor_count": possible_anchors,
        "eligible_anchor_count": eligible_anchors,
        "eligible_anchor_fraction": fraction(eligible_anchors, possible_anchors),
        "possible_window_count": possible_windows,
        "contributing_window_count": contributing_windows,
        "contributing_window_fraction": fraction(
            contributing_windows, possible_windows
        ),
        "measurement_valid_anchor_count": measurement_valid,
        "measurement_eligible_anchor_count": measurement_eligible,
        "measurement_coverage": fraction(measurement_valid, measurement_eligible),
    }


def _summarize_episode_blocks(
    accumulators: Mapping[tuple[str, int, int, str, int], _EpisodeErrors]
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    blocks = []
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for (cell, horizon, _source, _task, _episode), accumulator in sorted(
        accumulators.items()
    ):
        metrics = accumulator.metrics()
        row = {"cell": cell, "horizon": horizon, **metrics}
        blocks.append(row)
        grouped[(cell, horizon)].append(row)

    per_task: dict[str, object] = {}
    macro: dict[str, object] = {}
    for (cell, horizon), rows in sorted(grouped.items()):
        tasks: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            tasks[str(row["task"])].append(row)
        task_summary = {}
        for task, task_rows in sorted(tasks.items()):
            task_summary[task] = {
                **_support_summary(task_rows),
                "num_episodes": len(task_rows),
                "num_windows": sum(int(row["window_count"]) for row in task_rows),
                "mean_epe_px": float(
                    np.mean([float(row["mean_epe_px"]) for row in task_rows])
                ),
                "rmse_px": float(np.mean([float(row["rmse_px"]) for row in task_rows])),
                "mean_episode_p95_epe_px": float(
                    np.mean([float(row["p95_epe_px"]) for row in task_rows])
                ),
                "update_fraction": float(
                    np.mean([float(row["update_fraction"]) for row in task_rows])
                ),
            }
        label = f"{cell}|H={horizon}"
        per_task[label] = task_summary
        macro[label] = {
            **_support_summary(rows),
            "num_tasks": len(task_summary),
            "num_episodes": len(rows),
            "mean_epe_px": float(
                np.mean([float(values["mean_epe_px"]) for values in task_summary.values()])
            ),
            "rmse_px": float(
                np.mean([float(values["rmse_px"]) for values in task_summary.values()])
            ),
            "mean_episode_p95_epe_px": float(
                np.mean(
                    [float(values["mean_episode_p95_epe_px"]) for values in task_summary.values()]
                )
            ),
            "update_fraction": float(
                np.mean([float(values["update_fraction"]) for values in task_summary.values()])
            ),
        }
    return blocks, per_task, macro


def paired_episode_task_bootstrap(
    candidate_rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    """Bootstrap whole episodes within source/task strata.

    Each source is reduced to an equal-task macro before sources are equally
    weighted. Development currently has one task per source, so this remains
    its equal-task macro. Confirmatory evaluation has two four-task camera
    cohorts; keeping those strata fixed prevents one cohort from changing the
    other's sampling weight or hiding its uncertainty.
    """

    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")

    def indexed(
        rows: Sequence[Mapping[str, object]], label: str
    ) -> dict[tuple[str, int, int], float]:
        result = {}
        for row in rows:
            key = (str(row["task"]), int(row["source_index"]), int(row["episode"]))
            if key in result:
                raise ValueError(f"duplicate {label} episode row: {key}")
            result[key] = float(row["mean_epe_px"])
        return result

    candidate = indexed(candidate_rows, "candidate")
    baseline = indexed(baseline_rows, "baseline")
    if set(candidate) != set(baseline) or not candidate:
        raise ValueError("paired candidate/baseline episode sets differ or are empty")
    grouped: dict[tuple[int, str], list[tuple[float, float]]] = defaultdict(list)
    for key in sorted(candidate):
        task, source, _episode = key
        grouped[(source, task)].append((candidate[key], baseline[key]))

    def effect(candidate_value: float, baseline_value: float) -> dict[str, float]:
        improvement = baseline_value - candidate_value
        return {
            "candidate": candidate_value,
            "baseline": baseline_value,
            "improvement_px": improvement,
            "relative_improvement": improvement / max(baseline_value, 1e-12),
        }

    cell_points = {
        cell: effect(
            float(np.mean([pair[0] for pair in pairs])),
            float(np.mean([pair[1] for pair in pairs])),
        )
        for cell, pairs in sorted(grouped.items())
    }

    sources = sorted({source for source, _task in grouped})
    tasks_by_source = {
        source: sorted(
            task for cell_source, task in grouped if cell_source == source
        )
        for source in sources
    }
    sources_by_task: dict[str, list[int]] = defaultdict(list)
    for source, task in grouped:
        sources_by_task[task].append(source)
    sources_by_task = {
        task: sorted(task_sources)
        for task, task_sources in sorted(sources_by_task.items())
    }

    point_source: dict[str, dict[str, object]] = {}
    for source in sources:
        tasks = tasks_by_source[source]
        summary: dict[str, object] = effect(
            float(
                np.mean(
                    [cell_points[(source, task)]["candidate"] for task in tasks]
                )
            ),
            float(
                np.mean(
                    [cell_points[(source, task)]["baseline"] for task in tasks]
                )
            ),
        )
        summary.update(
            {
                "source_index": source,
                "tasks": tasks,
                "num_tasks": len(tasks),
                "num_episodes": sum(
                    len(grouped[(source, task)]) for task in tasks
                ),
            }
        )
        point_source[f"source_{source}"] = summary

    point_task = {
        task: effect(
            float(
                np.mean(
                    [
                        cell_points[(source, task)]["candidate"]
                        for source in task_sources
                    ]
                )
            ),
            float(
                np.mean(
                    [
                        cell_points[(source, task)]["baseline"]
                        for source in task_sources
                    ]
                )
            ),
        )
        for task, task_sources in sources_by_task.items()
    }
    point = {
        "candidate_mean_epe_px": float(
            np.mean([float(value["candidate"]) for value in point_source.values()])
        ),
        "baseline_mean_epe_px": float(
            np.mean([float(value["baseline"]) for value in point_source.values()])
        ),
    }
    point["improvement_px"] = point["baseline_mean_epe_px"] - point["candidate_mean_epe_px"]
    point["relative_improvement"] = point["improvement_px"] / max(
        point["baseline_mean_epe_px"], 1e-12
    )

    rng = np.random.default_rng(seed)
    candidate_macro = np.zeros(samples, dtype=np.float64)
    baseline_macro = np.zeros(samples, dtype=np.float64)
    source_bootstrap: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    source_task_report: dict[str, dict[str, object]] = {
        f"source_{source}": {} for source in sources
    }
    for source in sources:
        source_candidate = np.zeros(samples, dtype=np.float64)
        source_baseline = np.zeros(samples, dtype=np.float64)
        for task in tasks_by_source[source]:
            values = np.asarray(grouped[(source, task)], dtype=np.float64)
            indices = rng.integers(0, len(values), size=(samples, len(values)))
            candidate_means = values[indices, 0].mean(axis=1)
            baseline_means = values[indices, 1].mean(axis=1)
            source_candidate += candidate_means / len(tasks_by_source[source])
            source_baseline += baseline_means / len(tasks_by_source[source])
            cell_difference = baseline_means - candidate_means
            cell_relative = cell_difference / np.maximum(baseline_means, 1e-12)
            candidate_cell_rows = [
                row
                for row in candidate_rows
                if int(row["source_index"]) == source and str(row["task"]) == task
            ]
            source_task_report[f"source_{source}"][task] = {
                **cell_points[(source, task)],
                "num_episodes": len(values),
                "candidate_support": _support_summary(candidate_cell_rows),
                "bootstrap": {
                    "difference_95_ci": [
                        float(np.quantile(cell_difference, 0.025)),
                        float(np.quantile(cell_difference, 0.975)),
                    ],
                    "relative_improvement_95_ci": [
                        float(np.quantile(cell_relative, 0.025)),
                        float(np.quantile(cell_relative, 0.975)),
                    ],
                },
            }
        source_bootstrap[source] = (source_candidate, source_baseline)
        candidate_macro += source_candidate / len(sources)
        baseline_macro += source_baseline / len(sources)

    for source in sources:
        source_candidate, source_baseline = source_bootstrap[source]
        source_difference = source_baseline - source_candidate
        source_relative = source_difference / np.maximum(source_baseline, 1e-12)
        point_source[f"source_{source}"]["bootstrap"] = {
            "difference_95_ci": [
                float(np.quantile(source_difference, 0.025)),
                float(np.quantile(source_difference, 0.975)),
            ],
            "relative_improvement_95_ci": [
                float(np.quantile(source_relative, 0.025)),
                float(np.quantile(source_relative, 0.975)),
            ],
        }

    differences = baseline_macro - candidate_macro
    relative = differences / np.maximum(baseline_macro, 1e-12)
    return {
        "orientation": "positive_means_candidate_lower_error",
        "point": point,
        "per_task": point_task,
        "per_source": point_source,
        "per_source_task": source_task_report,
        "bootstrap": {
            "method": (
                "paired_whole_episode_within_source_task_"
                "equal_task_within_source_equal_source_macro"
            ),
            "samples": samples,
            "seed": seed,
            "difference_95_ci": [
                float(np.quantile(differences, 0.025)),
                float(np.quantile(differences, 0.975)),
            ],
            "relative_improvement_95_ci": [
                float(np.quantile(relative, 0.025)),
                float(np.quantile(relative, 0.975)),
            ],
            "one_sided_p_candidate_better": float(
                (1 + np.count_nonzero(differences <= 0.0)) / (samples + 1)
            ),
        },
    }


def _rows_for(
    blocks: Sequence[Mapping[str, object]], cell: str, horizon: int
) -> list[Mapping[str, object]]:
    return [
        row
        for row in blocks
        if row["cell"] == cell and int(row["horizon"]) == horizon
    ]


def _build_comparisons(
    blocks: Sequence[Mapping[str, object]],
    alphas: Sequence[float],
    horizons: Sequence[int],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    comparisons = {}
    raw_fcpg = _cell_id("hold_last", "fcpg", 1.0)
    for alpha in alphas:
        candidate = _cell_id("lewm", "fcpg", alpha)
        same_alpha = _cell_id("hold_last", "fcpg", alpha)
        for horizon in horizons:
            for name, baseline in (
                ("lewm_vs_hold_last_same_alpha", same_alpha),
                ("lewm_vs_raw_fcpg", raw_fcpg),
            ):
                label = f"{name}|alpha={_alpha_label(alpha)}|H={horizon}"
                comparisons[label] = paired_episode_task_bootstrap(
                    _rows_for(blocks, candidate, horizon),
                    _rows_for(blocks, baseline, horizon),
                    samples=bootstrap_samples,
                    # Resetting to one seed gives every matched cell/H the same
                    # task/episode resampling indices.
                    seed=bootstrap_seed,
                )
    return comparisons


def select_development_alpha(
    *,
    alphas: Sequence[float],
    comparisons: Mapping[str, object],
    macro: Mapping[str, object],
    primary_horizon: int = 5,
    max_task_relative_degradation: float = 0.10,
) -> dict[str, object]:
    """Select alpha without consulting oracle cells or secondary horizons."""

    available = {float(value) for value in alphas}
    missing = set(FUSION_ALPHA_GRID).difference(available)
    if missing:
        raise ValueError(f"development matrix is missing fusion alphas: {sorted(missing)}")
    candidates = []
    for alpha in FUSION_ALPHA_GRID:
        alpha_value = float(alpha)
        comparison_label = (
            "lewm_vs_hold_last_same_alpha|"
            f"alpha={_alpha_label(alpha_value)}|H={primary_horizon}"
        )
        comparison = comparisons[comparison_label]
        relative_by_task = {
            str(task): float(values["relative_improvement"])
            for task, values in comparison["per_task"].items()
        }
        eligible = all(
            value >= -float(max_task_relative_degradation)
            for value in relative_by_task.values()
        )
        cell_label = (
            f"{_cell_id('lewm', 'fcpg', alpha_value)}|H={primary_horizon}"
        )
        candidate_epe = float(macro[cell_label]["mean_epe_px"])
        candidates.append(
            {
                "alpha": alpha_value,
                "eligible": eligible,
                "candidate_task_macro_mean_epe_px": candidate_epe,
                "relative_improvement_by_task": relative_by_task,
            }
        )
    eligible_candidates = [value for value in candidates if value["eligible"]]
    selected = (
        min(
            eligible_candidates,
            key=lambda value: (
                value["candidate_task_macro_mean_epe_px"],
                value["alpha"],
            ),
        )
        if eligible_candidates
        else None
    )
    return {
        "status": (
            "candidate_selected" if selected is not None else "no_eligible_alpha"
        ),
        "selected_alpha": selected["alpha"] if selected is not None else None,
        "selection_uses": f"H{primary_horizon}_FCPG_candidate_only",
        "objective": "minimum_task_macro_mean_epe_px",
        "eligibility": {
            "comparison": "same_alpha_hold_last",
            "minimum_per_task_relative_improvement": -float(
                max_task_relative_degradation
            ),
        },
        "tie_break": "smaller_alpha",
        "oracle_consulted": False,
        "secondary_horizons_consulted": False,
        "diagnostic_endpoint_alphas": [0.0, 1.0],
        "candidates": candidates,
    }


def build_g0_gate(
    *,
    phase: str,
    alphas: Sequence[float],
    comparisons: Mapping[str, object],
    sanity_checks: Mapping[str, object],
    primary_horizon: int = 5,
    max_task_relative_degradation: float = 0.10,
    max_raw_fcpg_relative_degradation: float = 0.05,
    min_eligible_anchor_fraction: float = 0.95,
    min_contributing_window_fraction: float = 0.95,
    min_measurement_coverage: float = 0.95,
    max_source_task_relative_degradation: float = 0.10,
) -> dict[str, object]:
    """Apply a confirmatory H=5 gate; development never emits a Go decision."""

    integrity_passed = all(bool(value.get("passed")) for value in sanity_checks.values())
    if phase == "development":
        return {
            "name": "PixelObserver-G0",
            "passed": None,
            "decision": "Development-Only",
            "primary_horizon": primary_horizon,
            "oracle_part_of_gate": False,
            "conditions": {"sanity_checks": {"passed": integrity_passed}},
            "reason": "alpha selection is exploratory; a fresh confirmatory split is required",
        }
    alpha = float(alphas[0])
    label = (
        f"lewm_vs_hold_last_same_alpha|alpha={_alpha_label(alpha)}|"
        f"H={primary_horizon}"
    )
    comparison = comparisons[label]
    raw_label = (
        f"lewm_vs_raw_fcpg|alpha={_alpha_label(alpha)}|"
        f"H={primary_horizon}"
    )
    raw_comparison = comparisons[raw_label]

    def finite_ci_lower(value: object, key: str) -> float | None:
        interval = value.get(key) if isinstance(value, Mapping) else None
        if not isinstance(interval, Sequence) or isinstance(interval, (str, bytes)):
            return None
        if len(interval) != 2 or not math.isfinite(float(interval[0])):
            return None
        return float(interval[0])

    ci_lower = float(comparison["bootstrap"]["difference_95_ci"][0])
    task_relative = {
        task: float(values["relative_improvement"])
        for task, values in comparison["per_task"].items()
    }
    per_source = comparison.get("per_source", {})
    cohort_ci_lower: dict[str, float | None] = {}
    for source_index, cohort_name in CONFIRMATORY_COHORTS_BY_SOURCE.items():
        source = (
            per_source.get(f"source_{source_index}", {})
            if isinstance(per_source, Mapping)
            else {}
        )
        bootstrap = (
            source.get("bootstrap", {}) if isinstance(source, Mapping) else {}
        )
        interval = (
            bootstrap.get("difference_95_ci")
            if isinstance(bootstrap, Mapping)
            else None
        )
        cohort_ci_lower[cohort_name] = (
            float(interval[0])
            if isinstance(interval, Sequence)
            and not isinstance(interval, (str, bytes))
            and len(interval) == 2
            else None
        )
    cohort_ci_passed = all(
        value is not None and value > 0.0 for value in cohort_ci_lower.values()
    )

    per_source_task = comparison.get("per_source_task", {})
    task_names = sorted(
        {
            str(task)
            for source_report in (
                per_source_task.values()
                if isinstance(per_source_task, Mapping)
                else ()
            )
            if isinstance(source_report, Mapping)
            for task in source_report
        }
    )
    expected_cells = [
        (source_index, task)
        for source_index in CONFIRMATORY_COHORTS_BY_SOURCE
        for task in task_names
    ]
    if not expected_cells:
        expected_cells = [
            (source_index, "<missing>")
            for source_index in CONFIRMATORY_COHORTS_BY_SOURCE
        ]

    def source_task_cell(source_index: int, task: str) -> Mapping[str, object]:
        source_report = (
            per_source_task.get(f"source_{source_index}", {})
            if isinstance(per_source_task, Mapping)
            else {}
        )
        cell = source_report.get(task, {}) if isinstance(source_report, Mapping) else {}
        return cell if isinstance(cell, Mapping) else {}

    def support_values(metric: str) -> dict[str, float | None]:
        values: dict[str, float | None] = {}
        for source_index, task in expected_cells:
            cohort = CONFIRMATORY_COHORTS_BY_SOURCE[source_index]
            support = source_task_cell(source_index, task).get(
                "candidate_support", {}
            )
            raw = support.get(metric) if isinstance(support, Mapping) else None
            value = float(raw) if raw is not None else None
            values[f"{cohort}|{task}"] = (
                value if value is not None and math.isfinite(value) else None
            )
        return values

    eligible_anchor_fraction = support_values("eligible_anchor_fraction")
    contributing_window_fraction = support_values("contributing_window_fraction")
    measurement_coverage = support_values("measurement_coverage")
    source_task_relative_ci_lower = {}
    for source_index, task in expected_cells:
        cohort = CONFIRMATORY_COHORTS_BY_SOURCE[source_index]
        bootstrap = source_task_cell(source_index, task).get("bootstrap", {})
        source_task_relative_ci_lower[f"{cohort}|{task}"] = finite_ci_lower(
            bootstrap, "relative_improvement_95_ci"
        )

    raw_bootstrap = raw_comparison.get("bootstrap", {})
    raw_relative_ci_lower = finite_ci_lower(
        raw_bootstrap, "relative_improvement_95_ci"
    )

    def every_at_least(values: Mapping[str, float | None], threshold: float) -> bool:
        return bool(values) and all(
            value is not None and value >= threshold for value in values.values()
        )

    conditions = {
        "sanity_checks": {"passed": integrity_passed},
        "h5_paired_ci_lower_positive": {
            "passed": ci_lower > 0.0,
            "value": ci_lower,
            "operator": ">",
            "threshold": 0.0,
        },
        "each_camera_cohort_h5_paired_ci_lower_positive": {
            "passed": cohort_ci_passed,
            "values": cohort_ci_lower,
            "operator": ">",
            "threshold": 0.0,
            "source_index_binding": {
                cohort_name: source_index
                for source_index, cohort_name in CONFIRMATORY_COHORTS_BY_SOURCE.items()
            },
            "combination": "intersection_union",
        },
        "no_task_harmful_degradation": {
            "passed": all(
                value >= -float(max_task_relative_degradation)
                for value in task_relative.values()
            ),
            "values": task_relative,
            "minimum_allowed": -float(max_task_relative_degradation),
        },
        "each_source_task_eligible_anchor_fraction": {
            "passed": every_at_least(
                eligible_anchor_fraction, float(min_eligible_anchor_fraction)
            ),
            "values": eligible_anchor_fraction,
            "operator": ">=",
            "minimum_allowed": float(min_eligible_anchor_fraction),
        },
        "each_source_task_contributing_window_fraction": {
            "passed": every_at_least(
                contributing_window_fraction,
                float(min_contributing_window_fraction),
            ),
            "values": contributing_window_fraction,
            "operator": ">=",
            "minimum_allowed": float(min_contributing_window_fraction),
        },
        "each_source_task_measurement_coverage": {
            "passed": every_at_least(
                measurement_coverage, float(min_measurement_coverage)
            ),
            "values": measurement_coverage,
            "operator": ">=",
            "minimum_allowed": float(min_measurement_coverage),
        },
        "each_source_task_relative_improvement_ci_noninferiority": {
            "passed": every_at_least(
                source_task_relative_ci_lower,
                -float(max_source_task_relative_degradation),
            ),
            "values": source_task_relative_ci_lower,
            "operator": ">=",
            "minimum_allowed": -float(max_source_task_relative_degradation),
            "statistic": "bootstrap_relative_improvement_95_ci_lower",
        },
        "raw_fcpg_practical_noninferiority": {
            "passed": raw_relative_ci_lower is not None
            and raw_relative_ci_lower >= -float(max_raw_fcpg_relative_degradation),
            "value": raw_relative_ci_lower,
            "operator": ">=",
            "minimum_allowed": -float(max_raw_fcpg_relative_degradation),
            "statistic": "bootstrap_relative_improvement_95_ci_lower",
        },
    }
    passed = all(bool(value["passed"]) for value in conditions.values())
    return {
        "name": "PixelObserver-G0",
        "passed": passed,
        "decision": "Go" if passed else "No-Go",
        "frozen_alpha": alpha,
        "primary_horizon": primary_horizon,
        "primary_comparison": label,
        "raw_fcpg_guard_comparison": raw_label,
        "oracle_part_of_gate": False,
        "conditions": conditions,
    }


def evaluate_filter_batches(
    observer: LeWMPixelInnovationObserver,
    batches: Iterable[Mapping[str, object]],
    *,
    alphas: Sequence[float] = DEVELOPMENT_ALPHA_GRID,
    horizons: Sequence[int] = ENDPOINT_HORIZONS,
    phase: str = "development",
    device: torch.device | str = "cpu",
    max_batches: int | None = None,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 6144,
    primary_horizon: int = 5,
    max_task_relative_degradation: float = 0.10,
    max_raw_fcpg_relative_degradation: float = 0.05,
    min_eligible_anchor_fraction: float = 0.95,
    min_contributing_window_fraction: float = 0.95,
    min_measurement_coverage: float = 0.95,
    max_source_task_relative_degradation: float = 0.10,
) -> dict[str, object]:
    """Evaluate hold-last and LeWM filters with FCPG and oracle measurements."""

    selected_alphas, selected_horizons = validate_protocol(
        phase=phase, alphas=alphas, horizons=horizons
    )
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    target_device = torch.device(device)
    observer.to(target_device)
    observer.eval()
    history = int(observer.history_size)
    accumulators: dict[tuple[str, int, int, str, int], _EpisodeErrors] = {}
    batch_count = 0
    window_count = 0
    fcpg_identity_max = 0.0
    oracle_identity_max = 0.0
    fcpg_identity_anchors = 0
    oracle_identity_anchors = 0
    oracle_keys_absent = True

    def accumulate_cell(
        *,
        cell: str,
        q_posterior: torch.Tensor,
        measurement_valid: torch.Tensor,
        effective_gain: torch.Tensor,
        q_true_target: torch.Tensor,
        eligible_target: torch.Tensor,
        sources: list[object],
        tasks: list[object],
        episodes: list[object],
    ) -> None:
        errors = torch.linalg.vector_norm(q_posterior - q_true_target, dim=-1)
        for horizon in selected_horizons:
            endpoint = horizon - 1
            for index in range(errors.shape[0]):
                key = (
                    cell,
                    horizon,
                    int(sources[index]),
                    str(tasks[index]),
                    int(episodes[index]),
                )
                accumulator = accumulators.setdefault(
                    key,
                    _EpisodeErrors(
                        source_index=int(sources[index]),
                        task=str(tasks[index]),
                        episode=int(episodes[index]),
                    ),
                )
                accumulator.update(
                    errors[index, endpoint],
                    eligible_target[index, endpoint],
                    measurement_valid[index, endpoint],
                    effective_gain[index, endpoint],
                )

    with torch.inference_mode():
        for raw_batch in batches:
            if max_batches is not None and batch_count >= max_batches:
                break
            q_true_value = raw_batch.get("q_true")
            valid_true_value = raw_batch.get("valid_true")
            valid_nominal_value = raw_batch.get("valid_nominal_eval")
            if not all(
                torch.is_tensor(value)
                for value in (q_true_value, valid_true_value, valid_nominal_value)
            ):
                raise KeyError(
                    "evaluator batch requires q_true, valid_true, and valid_nominal_eval"
                )
            q_true = q_true_value.to(target_device).float()
            valid_true = valid_true_value.to(target_device).bool()
            valid_nominal = valid_nominal_value.to(target_device).bool()
            model_batch = deployment_batch(raw_batch, target_device)
            oracle_keys_absent = oracle_keys_absent and not bool(
                EVALUATOR_ONLY_KEYS.intersection(model_batch)
            )
            q_nominal = model_batch["q_nominal"].float()
            q_corrected = model_batch["q_corrected"].float()
            fcpg_valid = model_batch["valid"].bool()
            if q_true.shape != q_nominal.shape or valid_true.shape != q_nominal.shape[:-1]:
                raise ValueError("oracle and deployment grounding shapes differ")
            _, frames, _, _ = q_nominal.shape
            if frames < history + max(selected_horizons):
                raise ValueError("batch does not contain H history plus five filter frames")
            width = int(model_batch["pixels"].shape[-1])
            height = int(model_batch["pixels"].shape[-2])
            common = (
                valid_true
                & valid_nominal
                & _inside(q_true, width=width, height=height)
                & _inside(q_nominal, width=width, height=height)
            )
            target_slice = slice(history, history + max(selected_horizons))
            q_true_target = q_true[:, target_slice]
            eligible_target = common[:, target_slice]
            sources = _metadata_list(raw_batch.get("source_index"), q_true.shape[0], "source_index")
            tasks = _metadata_list(raw_batch.get("task_name"), q_true.shape[0], "task_name")
            episodes = _metadata_list(raw_batch.get("episode_id"), q_true.shape[0], "episode_id")

            # Nominal is a shared, alpha-independent reference.
            zero_gain = torch.zeros_like(q_nominal[:, target_slice])
            accumulate_cell(
                cell="prior=nominal|measurement=none|alpha=0",
                q_posterior=q_nominal[:, target_slice],
                measurement_valid=torch.zeros_like(common[:, target_slice]),
                effective_gain=zero_gain,
                q_true_target=q_true_target,
                eligible_target=eligible_target,
                sources=sources,
                tasks=tasks,
                episodes=episodes,
            )

            measurements = {
                "fcpg": (q_corrected - q_nominal, fcpg_valid),
                "oracle": (
                    q_true - q_nominal,
                    valid_true & _inside(q_true, width=width, height=height),
                ),
            }
            for measurement_name, (measurement, measurement_valid) in measurements.items():
                observer_batch = dict(model_batch)
                # Oracle substitution exists only in this diagnostic cell. The
                # original q_true key never enters the observer API.
                observer_batch["q_corrected"] = q_nominal + measurement
                observer_batch["valid"] = measurement_valid
                # The raw-FCPG endpoint is a fixed reference, not a selected
                # observer gain. Confirmatory runs evaluate one frozen alpha,
                # which need not be 1, so materialize this reference explicitly.
                if measurement_name == "fcpg" and 1.0 not in selected_alphas:
                    _, raw_posterior, raw_gain = hold_last_filter(
                        measurement,
                        measurement_valid,
                        history_size=history,
                        alpha=1.0,
                    )
                    raw_posterior = raw_posterior[:, : max(selected_horizons)]
                    raw_gain = raw_gain[:, : max(selected_horizons)]
                    accumulate_cell(
                        cell=_cell_id("hold_last", "fcpg", 1.0),
                        q_posterior=q_nominal[:, target_slice] + raw_posterior,
                        measurement_valid=measurement_valid[:, target_slice],
                        effective_gain=raw_gain,
                        q_true_target=q_true_target,
                        eligible_target=eligible_target,
                        sources=sources,
                        tasks=tasks,
                        episodes=episodes,
                    )
                for alpha in selected_alphas:
                    _, hold_posterior, hold_gain = hold_last_filter(
                        measurement,
                        measurement_valid,
                        history_size=history,
                        alpha=alpha,
                    )
                    hold_posterior = hold_posterior[:, : max(selected_horizons)]
                    hold_gain = hold_gain[:, : max(selected_horizons)]
                    hold_q = q_nominal[:, target_slice] + hold_posterior
                    accumulate_cell(
                        cell=_cell_id("hold_last", measurement_name, alpha),
                        q_posterior=hold_q,
                        measurement_valid=measurement_valid[:, target_slice],
                        effective_gain=hold_gain,
                        q_true_target=q_true_target,
                        eligible_target=eligible_target,
                        sources=sources,
                        tasks=tasks,
                        episodes=episodes,
                    )

                    filtered = observer.filter_sequence(
                        observer_batch,
                        gain=alpha,
                    )
                    lewm_q = filtered.q_posterior_px[:, : max(selected_horizons)]
                    lewm_gain = filtered.effective_gain[:, : max(selected_horizons)]
                    accumulate_cell(
                        cell=_cell_id("lewm", measurement_name, alpha),
                        q_posterior=lewm_q,
                        measurement_valid=measurement_valid[:, target_slice],
                        effective_gain=lewm_gain,
                        q_true_target=q_true_target,
                        eligible_target=eligible_target,
                        sources=sources,
                        tasks=tasks,
                        episodes=episodes,
                    )
                    if alpha == 1.0:
                        valid_endpoint = measurement_valid[:, target_slice]
                        if bool(valid_endpoint.any()):
                            identity_error = torch.abs(
                                lewm_q[valid_endpoint]
                                - (q_nominal[:, target_slice] + measurement[:, target_slice])[
                                    valid_endpoint
                                ]
                            ).max()
                            if measurement_name == "fcpg":
                                fcpg_identity_max = max(fcpg_identity_max, float(identity_error))
                                fcpg_identity_anchors += int(valid_endpoint.sum().item())
                            else:
                                oracle_identity_max = max(
                                    oracle_identity_max, float(identity_error)
                                )
                                oracle_identity_anchors += int(valid_endpoint.sum().item())
                if 1.0 not in selected_alphas:
                    identity = observer.filter_sequence(
                        observer_batch,
                        gain=1.0,
                    )
                    identity_q = identity.q_posterior_px[:, : max(selected_horizons)]
                    valid_endpoint = measurement_valid[:, target_slice]
                    if bool(valid_endpoint.any()):
                        identity_error = torch.abs(
                            identity_q[valid_endpoint]
                            - (q_nominal[:, target_slice] + measurement[:, target_slice])[
                                valid_endpoint
                            ]
                        ).max()
                        if measurement_name == "fcpg":
                            fcpg_identity_max = max(
                                fcpg_identity_max, float(identity_error)
                            )
                            fcpg_identity_anchors += int(valid_endpoint.sum().item())
                        else:
                            oracle_identity_max = max(
                                oracle_identity_max, float(identity_error)
                            )
                            oracle_identity_anchors += int(valid_endpoint.sum().item())
            batch_count += 1
            window_count += int(q_true.shape[0])

    if batch_count == 0:
        raise ValueError("filter evaluation received no batches")
    blocks, per_task, macro = _summarize_episode_blocks(accumulators)
    comparisons = _build_comparisons(
        blocks,
        selected_alphas,
        selected_horizons,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    sanity = {
        "oracle_fields_absent_from_model_batch": {
            "passed": oracle_keys_absent,
        },
        "fcpg_alpha1_identity": {
            "passed": fcpg_identity_anchors > 0 and fcpg_identity_max <= 1e-5,
            "max_abs_px": fcpg_identity_max,
            "valid_anchor_count": fcpg_identity_anchors,
            "tolerance_px": 1e-5,
        },
        "oracle_alpha1_identity": {
            "passed": oracle_identity_anchors > 0 and oracle_identity_max <= 1e-5,
            "max_abs_px": oracle_identity_max,
            "valid_anchor_count": oracle_identity_anchors,
            "tolerance_px": 1e-5,
        },
    }
    gate = build_g0_gate(
        phase=phase,
        alphas=selected_alphas,
        comparisons=comparisons,
        sanity_checks=sanity,
        primary_horizon=primary_horizon,
        max_task_relative_degradation=max_task_relative_degradation,
        max_raw_fcpg_relative_degradation=max_raw_fcpg_relative_degradation,
        min_eligible_anchor_fraction=min_eligible_anchor_fraction,
        min_contributing_window_fraction=min_contributing_window_fraction,
        min_measurement_coverage=min_measurement_coverage,
        max_source_task_relative_degradation=max_source_task_relative_degradation,
    )
    development_selection = (
        select_development_alpha(
            alphas=selected_alphas,
            comparisons=comparisons,
            macro=macro,
            primary_horizon=primary_horizon,
            max_task_relative_degradation=max_task_relative_degradation,
        )
        if phase == "development"
        else None
    )
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "schedule": {
            "filter": {
                "implemented": True,
                "future_measurements": "one real observation before every update",
                "endpoint_semantics": "elapsed filtered steps after shared warmup",
            },
            "forecast": {
                "implemented": False,
                "reason": (
                    "the observer has a no-future-observation forecast API, but "
                    "future FK/nominal propagation is only zero-order hold and is "
                    "not yet admitted to the G0 scientific gate"
                ),
            },
        },
        "contract": {
            "state": "per_anchor_2d_pixel_residual",
            "update": "prior + alpha * (measurement - prior)",
            "invalid_update": "retain_prior",
            "warmup": (
                "first H valid measurements use unit gain; invalid measurements "
                "retain the previous posterior from a zero initialization"
            ),
            "alphas": list(selected_alphas),
            "horizons": list(selected_horizons),
            "q_true_usage": "offline_scoring_and_oracle_diagnostic_only",
            "oracle_part_of_gate": False,
            "common_support": "valid_true_and_valid_nominal_and_both_inside_image",
        },
        "num_batches": batch_count,
        "num_windows": window_count,
        "episode_blocks": blocks,
        "per_task": per_task,
        "macro": macro,
        "paired_comparisons": comparisons,
        "bootstrap": {
            "method": (
                "paired_whole_episode_within_source_task_"
                "equal_task_within_source_equal_source_macro"
            ),
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "overlapping_windows_treated_as_independent": False,
            "shared_resampling_seed_across_cells": True,
        },
        "sanity_checks": sanity,
        "development_selection": development_selection,
        "gate": gate,
    }


def input_file_record(path: str | Path) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def records_digest(records: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda value: str(value["path"])):
        digest.update(str(record["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()
