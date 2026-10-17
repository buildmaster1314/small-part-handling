#!/usr/bin/env python3
"""Train matched posterior DAgger residual heads and report offline diagnostics.

The collector owns the causal rollout contract.  This script only consumes its
per-method caches, keeps train/development physical episodes disjoint, balances
each minibatch across DAgger iterations, and evaluates fixed latest
checkpoints.  Cross-method intervals are paired only when complete development
decision identities agree.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib
import inspect
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_STEPS = 2_000
DEFAULT_METHODS = (
    "complementary_factual",
    "complementary_no_action",
    "geometry_hold",
)
ACTION_STEPS = 5
ACTION_DIM = 4
MOTION_DIM = 3


@dataclass(frozen=True, slots=True)
class DaggerRows:
    """Normalized, single-method rows loaded from one or more cache files."""

    task_name: np.ndarray
    physical_episode_id: np.ndarray
    decision_index: np.ndarray
    dagger_iteration: np.ndarray
    missing_age: np.ndarray
    sensor_valid: np.ndarray
    state_latent: np.ndarray
    action_latent: np.ndarray
    base_action: np.ndarray
    expert_action: np.ndarray
    posterior_token: np.ndarray | None = None

    @property
    def sample_count(self) -> int:
        return int(self.physical_episode_id.shape[0])

    @property
    def model_dim(self) -> int:
        return int(self.state_latent.shape[-1])

    @property
    def decision_identity(self) -> np.ndarray:
        return np.asarray(
            [
                f"{task}\x1f{episode}\x1f{iteration}\x1f{decision}"
                for task, episode, iteration, decision in zip(
                    self.task_name.tolist(),
                    self.physical_episode_id.tolist(),
                    self.dagger_iteration.tolist(),
                    self.decision_index.tolist(),
                )
            ],
            dtype=str,
        )


@dataclass(frozen=True, slots=True)
class MetricRows:
    """Decision-level metric values with episode and pairing identities."""

    value: np.ndarray
    task_name: np.ndarray
    physical_episode_id: np.ndarray
    dagger_iteration: np.ndarray
    decision_identity: np.ndarray


@dataclass(frozen=True, slots=True)
class CacheReceipt:
    """Validated collection sidecar for one split/method/DAgger round."""

    path: Path
    split: str
    method: str
    dagger_iteration: int
    rows: int
    episode_panel: tuple[tuple[str, str, str], ...]
    head_checkpoint: str | None
    head_iteration: int | None


def balanced_dagger_batch_indices(
    dagger_iteration: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    """Sample every minibatch approximately uniformly across DAgger rounds.

    Sampling is with replacement within each iteration.  Every batch contains
    every observed iteration and per-iteration counts differ by at most one.
    The remainder rotates randomly so later DAgger rounds cannot be drowned by
    a much larger round-zero cache.
    """

    iterations = np.asarray(dagger_iteration)
    if iterations.ndim != 1 or iterations.size == 0:
        raise ValueError("dagger_iteration must be a non-empty vector")
    if iterations.dtype.kind not in "iu":
        raise TypeError("dagger_iteration must contain integers")
    if isinstance(steps, bool) or int(steps) <= 0:
        raise ValueError("steps must be positive")
    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    groups = np.unique(iterations)
    if int(batch_size) < groups.size:
        raise ValueError(
            "batch_size must be at least the number of DAgger iterations"
        )
    members = {int(group): np.flatnonzero(iterations == group) for group in groups}
    rng = np.random.default_rng(int(seed))
    result = np.empty((int(steps), int(batch_size)), dtype=np.int64)
    quotient, remainder = divmod(int(batch_size), int(groups.size))
    for step in range(int(steps)):
        order = rng.permutation(groups)
        chunks: list[np.ndarray] = []
        for rank, group in enumerate(order.tolist()):
            count = quotient + int(rank < remainder)
            chunks.append(rng.choice(members[int(group)], size=count, replace=True))
        batch = np.concatenate(chunks).astype(np.int64, copy=False)
        rng.shuffle(batch)
        result[step] = batch
    return result


def assert_disjoint_physical_episodes(
    train_task: Sequence[object],
    train_episode: Sequence[object],
    development_task: Sequence[object],
    development_episode: Sequence[object],
) -> None:
    """Reject physical episodes shared across train and development."""

    train_task_array = np.asarray(train_task).astype(str)
    train_episode_array = np.asarray(train_episode).astype(str)
    development_task_array = np.asarray(development_task).astype(str)
    development_episode_array = np.asarray(development_episode).astype(str)
    if train_task_array.shape != train_episode_array.shape:
        raise ValueError("train task and episode arrays are not aligned")
    if development_task_array.shape != development_episode_array.shape:
        raise ValueError("development task and episode arrays are not aligned")
    train = set(train_episode_array.tolist())
    development = set(development_episode_array.tolist())
    overlap = train & development
    if overlap:
        example = sorted(overlap)[0]
        raise ValueError(
            "train and development physical episodes overlap: "
            f"episode={example!r}"
        )


def _validate_metric_rows(rows: MetricRows, *, label: str) -> MetricRows:
    value = np.asarray(rows.value, dtype=np.float64)
    task = np.asarray(rows.task_name).astype(str)
    episode = np.asarray(rows.physical_episode_id).astype(str)
    iteration = np.asarray(rows.dagger_iteration)
    decision = np.asarray(rows.decision_identity).astype(str)
    count = int(value.size)
    if value.shape != (count,) or count == 0:
        raise ValueError(f"{label} metric values must be a non-empty vector")
    if (
        task.shape != (count,)
        or episode.shape != (count,)
        or iteration.shape != (count,)
        or decision.shape != (count,)
    ):
        raise ValueError(f"{label} metric metadata is not row-aligned")
    if iteration.dtype.kind not in "iu" or np.any(iteration < 0):
        raise ValueError(f"{label} DAgger iterations must be non-negative integers")
    if not np.isfinite(value).all() or np.any(value < 0.0):
        raise ValueError(f"{label} metric values must be finite and non-negative")
    if np.unique(decision).size != count:
        raise ValueError(f"{label} decision identities must be unique")
    return MetricRows(value, task, episode, iteration.astype(np.int64), decision)


def _episode_values(
    rows: MetricRows,
) -> dict[str, dict[str, dict[int, float]]]:
    grouped: dict[str, dict[str, dict[int, list[float]]]] = {}
    for value, task, episode, iteration in zip(
        rows.value.tolist(),
        rows.task_name.tolist(),
        rows.physical_episode_id.tolist(),
        rows.dagger_iteration.tolist(),
    ):
        grouped.setdefault(str(task), {}).setdefault(str(episode), {}).setdefault(
            int(iteration), []
        ).append(float(value))
    return {
        task: {
            episode: {
                iteration: float(np.mean(values))
                for iteration, values in iteration_rows.items()
            }
            for episode, iteration_rows in episode_rows.items()
        }
        for task, episode_rows in grouped.items()
    }


def _task_macro_mean(
    episodes: Mapping[str, Mapping[str, Mapping[int, float]]],
) -> float:
    """Average rounds within an episode, episodes within a task, then tasks."""

    return float(
        np.mean(
            [
                np.mean(
                    [
                        np.mean(np.fromiter(episode_rows.values(), dtype=np.float64))
                        for episode_rows in task_rows.values()
                    ]
                )
                for task_rows in episodes.values()
            ]
        )
    )


def _assert_same_task_rounds(
    candidate: Mapping[str, Mapping[str, Mapping[int, float]]],
    baseline: Mapping[str, Mapping[str, Mapping[int, float]]],
) -> None:
    for task in candidate:
        candidate_rounds = {
            iteration
            for episode_rows in candidate[task].values()
            for iteration in episode_rows
        }
        baseline_rounds = {
            iteration
            for episode_rows in baseline[task].values()
            for iteration in episode_rows
        }
        if candidate_rounds != baseline_rounds:
            raise ValueError(
                f"candidate and baseline DAgger round sets differ for task {task!r}"
            )


def episode_comparison_interval(
    candidate: MetricRows,
    baseline: MetricRows,
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    """Return paired or explicitly unpaired task/episode bootstrap intervals.

    Positive improvement means the candidate has lower XYZ action MSE.  Exact
    decision-identity equality is required for pairing.  When identities differ,
    candidate and baseline physical episodes are resampled independently within
    each task; the report never labels that result paired.
    """

    candidate = _validate_metric_rows(candidate, label="candidate")
    baseline = _validate_metric_rows(baseline, label="baseline")
    if isinstance(samples, bool) or int(samples) <= 0:
        raise ValueError("bootstrap samples must be positive")
    candidate_tasks = set(candidate.task_name.tolist())
    baseline_tasks = set(baseline.task_name.tolist())
    if candidate_tasks != baseline_tasks:
        raise ValueError("candidate and baseline task sets differ")

    candidate_by_episode = _episode_values(candidate)
    baseline_by_episode = _episode_values(baseline)
    _assert_same_task_rounds(candidate_by_episode, baseline_by_episode)
    paired = set(candidate.decision_identity.tolist()) == set(
        baseline.decision_identity.tolist()
    )
    tasks = sorted(candidate_tasks)
    rng = np.random.default_rng(int(seed))
    absolute = np.empty(int(samples), dtype=np.float64)
    relative = np.empty(int(samples), dtype=np.float64)

    if paired:
        candidate_identity_metadata = {
            identity: (str(task), str(episode), int(iteration))
            for identity, task, episode, iteration in zip(
                candidate.decision_identity.tolist(),
                candidate.task_name.tolist(),
                candidate.physical_episode_id.tolist(),
                candidate.dagger_iteration.tolist(),
            )
        }
        baseline_identity_metadata = {
            identity: (str(task), str(episode), int(iteration))
            for identity, task, episode, iteration in zip(
                baseline.decision_identity.tolist(),
                baseline.task_name.tolist(),
                baseline.physical_episode_id.tolist(),
                baseline.dagger_iteration.tolist(),
            )
        }
        if candidate_identity_metadata != baseline_identity_metadata:
            raise ValueError(
                "aligned decision identities carry different task/episode/round metadata"
            )
        candidate_lookup = dict(
            zip(candidate.decision_identity.tolist(), candidate.value.tolist())
        )
        baseline_lookup = dict(
            zip(baseline.decision_identity.tolist(), baseline.value.tolist())
        )
        delta_rows = MetricRows(
            value=np.asarray(
                [
                    baseline_lookup[identity] - candidate_lookup[identity]
                    for identity in candidate.decision_identity.tolist()
                ],
                dtype=np.float64,
            ),
            task_name=candidate.task_name,
            physical_episode_id=candidate.physical_episode_id,
            dagger_iteration=candidate.dagger_iteration,
            decision_identity=candidate.decision_identity,
        )
        delta_by_episode = _episode_values(delta_rows)
        for task in tasks:
            if set(candidate_by_episode[task]) != set(baseline_by_episode[task]):
                raise RuntimeError("paired decisions produced different episode sets")
            for episode in candidate_by_episode[task]:
                if set(candidate_by_episode[task][episode]) != set(
                    baseline_by_episode[task][episode]
                ):
                    raise RuntimeError(
                        "paired decisions produced different episode/round sets"
                    )
        for draw in range(int(samples)):
            task_delta: list[float] = []
            task_baseline: list[float] = []
            for task in tasks:
                episode_ids = sorted(delta_by_episode[task])
                chosen = rng.integers(0, len(episode_ids), size=len(episode_ids))
                selected = [episode_ids[index] for index in chosen.tolist()]
                task_delta.append(
                    float(
                        np.mean(
                            [
                                np.mean(
                                    np.fromiter(
                                        delta_by_episode[task][item].values(),
                                        dtype=np.float64,
                                    )
                                )
                                for item in selected
                            ]
                        )
                    )
                )
                task_baseline.append(
                    float(
                        np.mean(
                            [
                                np.mean(
                                    np.fromiter(
                                        baseline_by_episode[task][item].values(),
                                        dtype=np.float64,
                                    )
                                )
                                for item in selected
                            ]
                        )
                    )
                )
            absolute[draw] = float(np.mean(task_delta))
            denominator = float(np.mean(task_baseline))
            relative[draw] = absolute[draw] / max(denominator, 1.0e-12)
        design = "paired_physical_episode_within_task"
        note = "complete development decision identities align"
    else:
        for draw in range(int(samples)):
            task_delta = []
            task_baseline = []
            for task in tasks:
                candidate_values = np.asarray(
                    [
                        np.mean(np.fromiter(values.values(), dtype=np.float64))
                        for values in candidate_by_episode[task].values()
                    ],
                    dtype=np.float64,
                )
                baseline_values = np.asarray(
                    [
                        np.mean(np.fromiter(values.values(), dtype=np.float64))
                        for values in baseline_by_episode[task].values()
                    ],
                    dtype=np.float64,
                )
                candidate_mean = float(
                    np.mean(
                        rng.choice(
                            candidate_values,
                            size=candidate_values.size,
                            replace=True,
                        )
                    )
                )
                baseline_mean = float(
                    np.mean(
                        rng.choice(
                            baseline_values,
                            size=baseline_values.size,
                            replace=True,
                        )
                    )
                )
                task_delta.append(baseline_mean - candidate_mean)
                task_baseline.append(baseline_mean)
            absolute[draw] = float(np.mean(task_delta))
            denominator = float(np.mean(task_baseline))
            relative[draw] = absolute[draw] / max(denominator, 1.0e-12)
        design = "unpaired_independent_physical_episode_within_task"
        note = (
            "development decision identities do not align; arms were resampled "
            "independently and no paired claim is made"
        )

    candidate_mean = _task_macro_mean(candidate_by_episode)
    baseline_mean = _task_macro_mean(baseline_by_episode)
    improvement = baseline_mean - candidate_mean
    return {
        "paired": paired,
        "bootstrap_design": design,
        "pairing_note": note,
        "unit": "physical_episode_within_task_and_dagger_round",
        "round_aggregation": "equal_weight_dagger_rounds",
        "task_aggregation": "episode_mean_then_equal_weight_task_macro",
        "samples": int(samples),
        "seed": int(seed),
        "candidate_xyz_mse": candidate_mean,
        "baseline_xyz_mse": baseline_mean,
        "absolute_improvement": improvement,
        "absolute_improvement_95_ci": [
            float(np.quantile(absolute, 0.025)),
            float(np.quantile(absolute, 0.975)),
        ],
        "relative_improvement": improvement / max(baseline_mean, 1.0e-12),
        "relative_improvement_95_ci": [
            float(np.quantile(relative, 0.025)),
            float(np.quantile(relative, 0.975)),
        ],
        "candidate_episode_count": sum(
            len(task_rows) for task_rows in candidate_by_episode.values()
        ),
        "baseline_episode_count": sum(
            len(task_rows) for task_rows in baseline_by_episode.values()
        ),
    }


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _read_json(path: str | Path) -> dict[str, object]:
    source = _resolve(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {source}")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _cache_paths(
    config: Mapping[str, object], split: str, method: str
) -> tuple[Path, ...]:
    caches = _mapping(config.get("caches"), label="caches")
    split_config = _mapping(caches.get(split), label=f"caches.{split}")
    value = split_config.get(method)
    if isinstance(value, (str, Path)):
        entries: Sequence[object] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        entries = value
    else:
        raise TypeError(f"caches.{split}.{method} must be a path or path list")
    paths = tuple(_resolve(str(entry)) for entry in entries)
    if not paths:
        raise ValueError(f"caches.{split}.{method} cannot be empty")
    return paths


def _required_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    return int(value)


def _load_cache_receipt(
    path: Path,
    *,
    expected_split: str,
    expected_method: str,
    expected_protocol: str,
    observed_rows: int,
    observed_iterations: np.ndarray,
) -> CacheReceipt:
    metadata_path = path.with_suffix(".metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"posterior DAgger cache requires sibling sidecar {metadata_path}"
        )
    metadata = _read_json(metadata_path)
    if str(metadata.get("protocol", "")) != expected_protocol:
        raise ValueError(f"cache sidecar protocol mismatch: {metadata_path}")
    split = str(metadata.get("split", ""))
    method = str(metadata.get("method", ""))
    if split != expected_split:
        raise ValueError(
            f"cache sidecar split mismatch for {path}: {split!r} != {expected_split!r}"
        )
    if method != expected_method:
        raise ValueError(
            f"cache sidecar method mismatch for {path}: "
            f"{method!r} != {expected_method!r}"
        )
    iteration = _required_int(
        metadata.get("dagger_iteration"),
        label=f"{metadata_path}: dagger_iteration",
    )
    rows = _required_int(metadata.get("rows"), label=f"{metadata_path}: rows")
    if rows != int(observed_rows):
        raise ValueError(
            f"cache sidecar row count mismatch for {path}: {rows} != {observed_rows}"
        )
    cached_iterations = np.unique(np.asarray(observed_iterations, dtype=np.int64))
    if cached_iterations.tolist() != [iteration]:
        raise ValueError(
            f"cache {path} must contain exactly sidecar DAgger round {iteration}"
        )

    episode_metrics = metadata.get("episode_metrics")
    if not isinstance(episode_metrics, Sequence) or isinstance(
        episode_metrics, (str, bytes)
    ):
        raise TypeError(f"{metadata_path}: episode_metrics must be a list")
    panel_rows: list[tuple[str, str, str]] = []
    for index, entry in enumerate(episode_metrics):
        item = _mapping(entry, label=f"{metadata_path}: episode_metrics[{index}]")
        missing = {
            key for key in ("episode_id", "task_index", "environment_seed") if key not in item
        }
        if missing:
            raise ValueError(
                f"{metadata_path}: episode_metrics[{index}] lacks {sorted(missing)}"
            )
        panel_rows.append(
            (
                str(item["episode_id"]),
                str(item["task_index"]),
                str(item["environment_seed"]),
            )
        )
    if not panel_rows or len(set(panel_rows)) != len(panel_rows):
        raise ValueError(
            f"{metadata_path}: initial physical episode panel is empty or duplicated"
        )
    episodes = _required_int(
        metadata.get("episodes"), label=f"{metadata_path}: episodes"
    )
    if episodes != len(panel_rows):
        raise ValueError(
            f"{metadata_path}: episodes does not match episode_metrics length"
        )

    head_checkpoint_value = metadata.get("head_checkpoint")
    head_metadata_value = metadata.get("head_metadata")
    head_checkpoint: str | None = None
    head_iteration: int | None = None
    if iteration == 0:
        if head_checkpoint_value is not None or head_metadata_value is not None:
            raise ValueError(f"round 0 cache {path} must not use a behavior head")
    else:
        if not isinstance(head_checkpoint_value, str) or not head_checkpoint_value:
            raise ValueError(f"round {iteration} cache {path} must record a behavior head")
        head_metadata = _mapping(
            head_metadata_value, label=f"{metadata_path}: head_metadata"
        )
        if str(head_metadata.get("method", "")) != expected_method:
            raise ValueError(f"behavior head method mismatch for {path}")
        head_iteration = _required_int(
            head_metadata.get("dagger_iteration"),
            label=f"{metadata_path}: head_metadata.dagger_iteration",
        )
        if head_iteration != iteration - 1:
            raise ValueError(
                f"round {iteration} cache {path} must use a round {iteration - 1} head"
            )
        head_checkpoint = head_checkpoint_value

    return CacheReceipt(
        path=path,
        split=split,
        method=method,
        dagger_iteration=iteration,
        rows=rows,
        episode_panel=tuple(sorted(panel_rows)),
        head_checkpoint=head_checkpoint,
        head_iteration=head_iteration,
    )


def _validate_matched_collection(
    receipts: Mapping[str, Sequence[CacheReceipt]],
    *,
    split: str,
    methods: Sequence[str],
) -> tuple[int, ...]:
    method_rounds: dict[str, tuple[int, ...]] = {}
    by_method_round: dict[str, dict[int, CacheReceipt]] = {}
    for method in methods:
        values = tuple(receipts[method])
        round_map = {receipt.dagger_iteration: receipt for receipt in values}
        if len(round_map) != len(values):
            raise ValueError(f"{split} method {method!r} has duplicate DAgger rounds")
        rounds = tuple(sorted(round_map))
        if not rounds or rounds != tuple(range(rounds[-1] + 1)):
            raise ValueError(
                f"{split} method {method!r} DAgger rounds must be continuous from zero"
            )
        method_rounds[method] = rounds
        by_method_round[method] = round_map
    reference_method = str(methods[0])
    common_rounds = method_rounds[reference_method]
    for method in methods[1:]:
        if method_rounds[str(method)] != common_rounds:
            raise ValueError(f"{split} matched methods use different DAgger round sets")
    for iteration in common_rounds:
        reference_panel = by_method_round[reference_method][iteration].episode_panel
        for method in methods[1:]:
            if by_method_round[str(method)][iteration].episode_panel != reference_panel:
                raise ValueError(
                    f"{split} round {iteration} matched methods use different "
                    "initial physical episode panels"
                )
    return common_rounds


def _field(source: object, *names: str, required: bool = True) -> np.ndarray | None:
    for name in names:
        if isinstance(source, Mapping) and name in source:
            return np.asarray(source[name])
        if hasattr(source, name):
            return np.asarray(getattr(source, name))
    if required:
        raise AttributeError(f"posterior DAgger cache lacks fields {names}")
    return None


def _load_api() -> tuple[Any, type[nn.Module], Any]:
    try:
        module = importlib.import_module("fcpg.lewm_posterior_dagger")
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "fcpg.lewm_posterior_dagger must be implemented before training"
        ) from error
    model = getattr(module, "PosteriorDaggerResidualHead", None)
    loader = getattr(module, "load_posterior_dagger_arrays", None)
    if not isinstance(model, type) or not callable(loader):
        raise AttributeError(
            "posterior DAgger API must expose PosteriorDaggerResidualHead and "
            "load_posterior_dagger_arrays"
        )
    return module, model, loader


def _normalize_cache(
    source: object,
    *,
    method: str,
    path: Path,
    method_order: Sequence[str],
) -> DaggerRows:
    episode = _field(source, "physical_episode_id", "episode_id")
    count = int(np.asarray(episode).shape[0])
    task = _field(source, "task_name", required=False)
    if task is None:
        task = np.full(count, "all_tasks", dtype=str)
    cached_method = _field(source, "method", required=False)
    if cached_method is not None:
        values = set(np.asarray(cached_method).astype(str).reshape(-1).tolist())
        if values != {str(method)}:
            raise ValueError(f"cache {path} does not contain only method {method!r}")
    method_index = _field(source, "method_index", required=False)
    if method_index is not None:
        expected_index = tuple(method_order).index(method)
        observed = set(np.asarray(method_index, dtype=np.int64).reshape(-1).tolist())
        if observed != {expected_index}:
            raise ValueError(
                f"cache {path} method_index does not match method {method!r}"
            )
    posterior = _field(source, "posterior_token", required=False)
    sensor_valid = _field(source, "sensor_valid", required=False)
    if sensor_valid is None:
        sensor_valid = np.zeros(count, dtype=bool)
    rows = DaggerRows(
        task_name=np.asarray(task).astype(str),
        physical_episode_id=np.asarray(episode).astype(str),
        decision_index=np.asarray(
            _field(source, "decision_index", "decision_id"), dtype=np.int64
        ),
        dagger_iteration=np.asarray(
            _field(source, "dagger_iteration", "iteration"), dtype=np.int64
        ),
        missing_age=np.asarray(_field(source, "missing_age"), dtype=np.int64),
        sensor_valid=np.asarray(sensor_valid, dtype=bool),
        state_latent=np.asarray(_field(source, "state_latent"), dtype=np.float32),
        action_latent=np.asarray(_field(source, "action_latent"), dtype=np.float32),
        base_action=np.asarray(_field(source, "base_action"), dtype=np.float32),
        expert_action=np.asarray(_field(source, "expert_action"), dtype=np.float32),
        posterior_token=None if posterior is None else np.asarray(posterior, dtype=np.float32),
    )
    _validate_rows(rows, label=str(path))
    return rows


def _validate_rows(rows: DaggerRows, *, label: str) -> None:
    count = rows.sample_count
    vectors = (
        ("task_name", rows.task_name),
        ("physical_episode_id", rows.physical_episode_id),
        ("decision_index", rows.decision_index),
        ("dagger_iteration", rows.dagger_iteration),
        ("missing_age", rows.missing_age),
        ("sensor_valid", rows.sensor_valid),
    )
    for name, value in vectors:
        if np.asarray(value).shape != (count,):
            raise ValueError(f"{label}: {name} must have shape [N]")
    if count <= 0 or np.any(rows.dagger_iteration < 0):
        raise ValueError(f"{label}: cache must contain non-negative DAgger rows")
    if rows.sensor_valid.dtype != np.bool_ or np.any(rows.sensor_valid):
        raise ValueError(f"{label}: DAgger training rows must be invalid observations")
    if rows.state_latent.ndim != 2 or rows.state_latent.shape[0] != count:
        raise ValueError(f"{label}: state_latent must have shape [N,D]")
    expected_action_latent = (count, ACTION_STEPS, rows.model_dim)
    if rows.action_latent.shape != expected_action_latent:
        raise ValueError(
            f"{label}: action_latent must have shape {expected_action_latent}"
        )
    if rows.base_action.shape != (count, ACTION_STEPS, ACTION_DIM):
        raise ValueError(f"{label}: base_action must have shape [N,5,4]")
    if rows.expert_action.shape != (count, ACTION_STEPS, ACTION_DIM):
        raise ValueError(f"{label}: expert_action must have shape [N,5,4]")
    if rows.posterior_token is not None and (
        rows.posterior_token.ndim != 3
        or rows.posterior_token.shape[0] != count
        or rows.posterior_token.shape[-1] != rows.model_dim
    ):
        raise ValueError(f"{label}: posterior_token must have shape [N,K,D]")
    for name, value in (
        ("state_latent", rows.state_latent),
        ("action_latent", rows.action_latent),
        ("base_action", rows.base_action),
        ("expert_action", rows.expert_action),
    ):
        if not np.isfinite(value).all():
            raise ValueError(f"{label}: {name} contains non-finite values")
    if np.unique(rows.decision_identity).size != count:
        raise ValueError(f"{label}: decision identities must be unique")


def _concatenate(rows: Sequence[DaggerRows]) -> DaggerRows:
    if not rows:
        raise ValueError("at least one DAgger cache is required")
    model_dims = {value.model_dim for value in rows}
    posterior_presence = {value.posterior_token is not None for value in rows}
    if len(model_dims) != 1 or len(posterior_presence) != 1:
        raise ValueError("aggregated DAgger caches have incompatible tensor contracts")
    combined = DaggerRows(
        task_name=np.concatenate([value.task_name for value in rows]),
        physical_episode_id=np.concatenate(
            [value.physical_episode_id for value in rows]
        ),
        decision_index=np.concatenate([value.decision_index for value in rows]),
        dagger_iteration=np.concatenate(
            [value.dagger_iteration for value in rows]
        ),
        missing_age=np.concatenate([value.missing_age for value in rows]),
        sensor_valid=np.concatenate([value.sensor_valid for value in rows]),
        state_latent=np.concatenate([value.state_latent for value in rows]),
        action_latent=np.concatenate([value.action_latent for value in rows]),
        base_action=np.concatenate([value.base_action for value in rows]),
        expert_action=np.concatenate([value.expert_action for value in rows]),
        posterior_token=(
            np.concatenate(
                [value.posterior_token for value in rows if value.posterior_token is not None]
            )
            if True in posterior_presence
            else None
        ),
    )
    _validate_rows(combined, label="aggregated caches")
    return combined


def _load_method_rows(
    paths: Sequence[Path],
    *,
    split: str,
    method: str,
    method_order: Sequence[str],
    protocol: str,
    loader: Any,
) -> tuple[DaggerRows, tuple[CacheReceipt, ...]]:
    loaded: list[DaggerRows] = []
    receipts: list[CacheReceipt] = []
    for path in paths:
        if "expected_method" in inspect.signature(loader).parameters:
            source = loader(path, expected_method=method)
        else:
            source = loader(path)
        rows = _normalize_cache(
            source,
            method=method,
            path=path,
            method_order=method_order,
        )
        loaded.append(rows)
        receipts.append(
            _load_cache_receipt(
                path,
                expected_split=split,
                expected_method=method,
                expected_protocol=protocol,
                observed_rows=rows.sample_count,
                observed_iterations=rows.dagger_iteration,
            )
        )
    return _concatenate(loaded), tuple(receipts)


def _make_model(
    model_type: type[nn.Module], rows: DaggerRows, *, hidden_dim: int
) -> nn.Module:
    signature = inspect.signature(model_type)
    kwargs: dict[str, object] = {
        "model_dim": rows.model_dim,
        "hidden_dim": int(hidden_dim),
    }
    if "posterior_token_count" in signature.parameters:
        kwargs["posterior_token_count"] = (
            0 if rows.posterior_token is None else int(rows.posterior_token.shape[1])
        )
    return model_type(**kwargs)


def _tensor_rows(rows: DaggerRows, device: torch.device) -> dict[str, torch.Tensor]:
    tensors = {
        "state_latent": torch.from_numpy(
            np.ascontiguousarray(rows.state_latent)
        ).to(device),
        "action_latent": torch.from_numpy(
            np.ascontiguousarray(rows.action_latent)
        ).to(device),
        "base_action": torch.from_numpy(
            np.ascontiguousarray(rows.base_action)
        ).to(device),
        "expert_action": torch.from_numpy(
            np.ascontiguousarray(rows.expert_action)
        ).to(device),
        "missing_age": torch.from_numpy(
            np.ascontiguousarray(rows.missing_age)
        ).long().to(device),
        "sensor_valid": torch.from_numpy(
            np.ascontiguousarray(rows.sensor_valid)
        ).bool().to(device),
    }
    if rows.posterior_token is not None:
        tensors["posterior_token"] = torch.from_numpy(
            np.ascontiguousarray(rows.posterior_token)
        ).to(device)
    return tensors


def _forward(model: nn.Module, tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    accepted = inspect.signature(model.forward).parameters
    kwargs = {
        name: value
        for name, value in tensors.items()
        if name != "expert_action" and name in accepted
    }
    prediction = model(**kwargs)
    if prediction.shape == tensors["base_action"][..., :MOTION_DIM].shape:
        prediction = torch.cat(
            (prediction, tensors["base_action"][..., MOTION_DIM:]), dim=-1
        )
    if prediction.shape != tensors["base_action"].shape:
        raise ValueError(
            "PosteriorDaggerResidualHead must return [B,5,4] or [B,5,3]"
        )
    return prediction


def _predict(
    model: nn.Module,
    rows: DaggerRows,
    *,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    tensors = _tensor_rows(rows, device)
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, rows.sample_count, int(batch_size)):
            stop = min(start + int(batch_size), rows.sample_count)
            batch = {name: value[start:stop] for name, value in tensors.items()}
            predictions.append(_forward(model, batch).float().cpu().numpy())
    return np.ascontiguousarray(np.concatenate(predictions), dtype=np.float32)


def _xyz_row_mse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.shape[-2:] != (
        ACTION_STEPS,
        ACTION_DIM,
    ):
        raise ValueError("action arrays must be aligned [N,5,4]")
    return np.mean(
        np.square(prediction[..., :MOTION_DIM] - target[..., :MOTION_DIM]),
        axis=(1, 2),
    )


def _metric_rows(rows: DaggerRows, values: np.ndarray) -> MetricRows:
    return MetricRows(
        value=np.asarray(values, dtype=np.float64),
        task_name=rows.task_name,
        physical_episode_id=rows.physical_episode_id,
        dagger_iteration=rows.dagger_iteration,
        decision_identity=rows.decision_identity,
    )


def _train_one(
    model_type: type[nn.Module],
    train: DaggerRows,
    development: DaggerRows,
    *,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
    device: torch.device,
) -> tuple[nn.Module, torch.optim.Optimizer, np.ndarray, dict[str, object]]:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = _make_model(model_type, train, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    tensors = _tensor_rows(train, device)
    batches = balanced_dagger_batch_indices(
        train.dagger_iteration,
        steps=int(steps),
        batch_size=int(batch_size),
        seed=int(seed),
    )
    model.train()
    last_loss = float("nan")
    for row_indices in batches:
        index = torch.from_numpy(row_indices).long().to(device)
        batch = {name: value.index_select(0, index) for name, value in tensors.items()}
        prediction = _forward(model, batch)
        target = batch["expert_action"]
        loss = F.smooth_l1_loss(
            prediction[..., :MOTION_DIM],
            target[..., :MOTION_DIM],
            beta=0.05,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    prediction = _predict(model, development, device=device)
    gripper_delta = np.abs(prediction[..., 3] - development.base_action[..., 3])
    if not np.array_equal(prediction[..., 3], development.base_action[..., 3]):
        raise RuntimeError("PosteriorDaggerResidualHead changed gripper actions")
    residual_mse = _task_macro_mean(
        _episode_values(
            _metric_rows(
                development,
                _xyz_row_mse(prediction, development.expert_action),
            )
        )
    )
    base_mse = _task_macro_mean(
        _episode_values(
            _metric_rows(
                development,
                _xyz_row_mse(development.base_action, development.expert_action),
            )
        )
    )
    metrics: dict[str, object] = {
        "last_train_loss": last_loss,
        "development_xyz_action_mse": residual_mse,
        "development_base_xyz_action_mse": base_mse,
        "gripper_unchanged": True,
        "gripper_max_abs_change_from_base": float(np.max(gripper_delta)),
        "gripper_mse_vs_expert_supporting_only": float(
            np.mean(np.square(prediction[..., 3] - development.expert_action[..., 3]))
        ),
    }
    return model, optimizer, prediction, metrics


def _atomic_torch_save(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _role_method(
    config: Mapping[str, object], role: str, default: str, methods: Sequence[str]
) -> str:
    roles = config.get("roles", {})
    if not isinstance(roles, Mapping):
        raise TypeError("roles must be an object")
    method = str(roles.get(role, default))
    if method not in methods:
        raise ValueError(f"roles.{role} refers to unknown method {method!r}")
    return method


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)

    config = _read_json(args.config)
    training = _mapping(config.get("training"), label="training")
    methods_value = config.get("methods", DEFAULT_METHODS)
    if not isinstance(methods_value, Sequence) or isinstance(methods_value, (str, bytes)):
        parser.error("methods must be a list")
    methods = tuple(str(value) for value in methods_value)
    if len(methods) != 3 or len(set(methods)) != len(methods):
        parser.error("methods must contain three unique matched methods")
    steps = int(training.get("steps", DEFAULT_STEPS) if args.steps is None else args.steps)
    seeds_value = training.get("seeds", (0,)) if args.seeds is None else args.seeds
    if not isinstance(seeds_value, Sequence) or isinstance(seeds_value, (str, bytes)):
        parser.error("training.seeds must be a list")
    seeds = tuple(int(value) for value in seeds_value)
    if steps <= 0 or not seeds or len(set(seeds)) != len(seeds):
        parser.error("steps and unique seeds must be positive/non-empty")
    batch_size = int(training.get("batch_size", 256))
    if batch_size <= 0:
        parser.error("training.batch_size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index not in (None, 0):
        parser.error("posterior DAgger training may use cuda:0 or cpu only")

    dagger_api, model_type, loader = _load_api()
    api_methods = tuple(str(value) for value in dagger_api.METHODS)
    if methods != api_methods:
        parser.error(
            "methods must exactly match fcpg.lewm_posterior_dagger.METHODS"
        )
    protocol = str(config.get("protocol", dagger_api.PROTOCOL))
    if protocol != str(dagger_api.PROTOCOL):
        parser.error("config protocol must match fcpg.lewm_posterior_dagger.PROTOCOL")
    train_rows: dict[str, DaggerRows] = {}
    development_rows: dict[str, DaggerRows] = {}
    train_receipts: dict[str, tuple[CacheReceipt, ...]] = {}
    development_receipts: dict[str, tuple[CacheReceipt, ...]] = {}
    for method in methods:
        train_rows[method], train_receipts[method] = _load_method_rows(
            _cache_paths(config, "train", method),
            split="train",
            method=method,
            method_order=api_methods,
            protocol=protocol,
            loader=loader,
        )
        development_rows[method], development_receipts[method] = _load_method_rows(
            _cache_paths(config, "development", method),
            split="development",
            method=method,
            method_order=api_methods,
            protocol=protocol,
            loader=loader,
        )
        assert_disjoint_physical_episodes(
            train_rows[method].task_name,
            train_rows[method].physical_episode_id,
            development_rows[method].task_name,
            development_rows[method].physical_episode_id,
        )
    train_rounds = _validate_matched_collection(
        train_receipts, split="train", methods=methods
    )
    development_rounds = _validate_matched_collection(
        development_receipts, split="development", methods=methods
    )
    if train_rounds != development_rounds:
        raise ValueError("train and development DAgger round sets must match")
    all_train_task = np.concatenate([train_rows[method].task_name for method in methods])
    all_train_episode = np.concatenate(
        [train_rows[method].physical_episode_id for method in methods]
    )
    all_development_task = np.concatenate(
        [development_rows[method].task_name for method in methods]
    )
    all_development_episode = np.concatenate(
        [development_rows[method].physical_episode_id for method in methods]
    )
    assert_disjoint_physical_episodes(
        all_train_task,
        all_train_episode,
        all_development_task,
        all_development_episode,
    )
    dimensions = {
        rows.model_dim for rows in (*train_rows.values(), *development_rows.values())
    }
    if len(dimensions) != 1:
        raise ValueError("matched methods use different latent dimensions")

    output_dir = _resolve(
        args.output_dir
        or str(training.get("output_dir", "outputs/lewm_posterior_dagger"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, list[np.ndarray]] = {method: [] for method in methods}
    seed_metrics: list[dict[str, object]] = []
    hidden_dim = int(training.get("hidden_dim", 128))
    learning_rate = float(training.get("learning_rate", 3.0e-4))
    weight_decay = float(training.get("weight_decay", 1.0e-6))

    for seed in seeds:
        for method in methods:
            model, optimizer, prediction, metrics = _train_one(
                model_type,
                train_rows[method],
                development_rows[method],
                seed=seed,
                steps=steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                hidden_dim=hidden_dim,
                device=device,
            )
            predictions[method].append(prediction)
            checkpoint_path = output_dir / f"{method}_seed{seed}_latest.pt"
            _atomic_torch_save(
                {
                    "schema_version": int(dagger_api.SCHEMA_VERSION),
                    "checkpoint_type": str(dagger_api.CHECKPOINT_TYPE),
                    "protocol": protocol,
                    "methods": list(api_methods),
                    "method": method,
                    "method_index": api_methods.index(method),
                    "seed": int(seed),
                    "steps": steps,
                    "dagger_iteration": int(train_rounds[-1]),
                    "optimizer_steps": steps,
                    "selection": "fixed_latest_only",
                    "architecture": model.checkpoint_config(),
                    "dagger_iterations": sorted(
                        np.unique(train_rows[method].dagger_iteration).astype(int).tolist()
                    ),
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                checkpoint_path,
            )
            seed_metrics.append(
                {
                    "method": method,
                    "seed": int(seed),
                    "checkpoint": str(checkpoint_path),
                    **metrics,
                }
            )
            print(
                f"seed={seed} method={method} "
                f"dev_xyz_mse={float(metrics['development_xyz_action_mse']):.6f}",
                flush=True,
            )

    residual_metrics: dict[str, MetricRows] = {}
    base_metrics: dict[str, MetricRows] = {}
    for method in methods:
        rows = development_rows[method]
        residual_loss = np.mean(
            np.stack(
                [
                    _xyz_row_mse(prediction, rows.expert_action)
                    for prediction in predictions[method]
                ]
            ),
            axis=0,
        )
        residual_metrics[method] = _metric_rows(rows, residual_loss)
        base_metrics[method] = _metric_rows(
            rows, _xyz_row_mse(rows.base_action, rows.expert_action)
        )

    factual = _role_method(
        config, "factual", "complementary_factual", methods
    )
    no_action = _role_method(
        config, "no_action", "complementary_no_action", methods
    )
    geometry = _role_method(config, "geometry", "geometry_hold", methods)
    bootstrap_samples = int(training.get("bootstrap_samples", 10_000))
    bootstrap_seed = int(training.get("bootstrap_seed", 73_101))
    comparisons = {
        "factual_residual_vs_factual_base": episode_comparison_interval(
            residual_metrics[factual],
            base_metrics[factual],
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "factual_residual_vs_no_action_residual": episode_comparison_interval(
            residual_metrics[factual],
            residual_metrics[no_action],
            samples=bootstrap_samples,
            seed=bootstrap_seed + 1,
        ),
        "factual_residual_vs_geometry_residual": episode_comparison_interval(
            residual_metrics[factual],
            residual_metrics[geometry],
            samples=bootstrap_samples,
            seed=bootstrap_seed + 2,
        ),
    }
    report = {
        "schema_version": 1,
        "protocol": protocol,
        "classification": "posterior_dagger_offline_development",
        "primary_metric": "physical_episode_task_macro_xyz_action_mse_vs_expert",
        "selection": "fixed_latest_checkpoints_only",
        "optimizer_steps": steps,
        "methods": list(methods),
        "roles": {"factual": factual, "no_action": no_action, "geometry": geometry},
        "seeds": list(seeds),
        "training": {
            method: {
                "rows": train_rows[method].sample_count,
                "physical_episodes": len(
                    set(
                        zip(
                            train_rows[method].task_name.tolist(),
                            train_rows[method].physical_episode_id.tolist(),
                        )
                    )
                ),
                "dagger_iterations": sorted(
                    np.unique(train_rows[method].dagger_iteration).astype(int).tolist()
                ),
                "minibatch_sampling": "equal_weight_dagger_iteration",
            }
            for method in methods
        },
        "development": {
            method: {
                "rows": development_rows[method].sample_count,
                "physical_episodes": len(
                    set(
                        zip(
                            development_rows[method].task_name.tolist(),
                            development_rows[method].physical_episode_id.tolist(),
                        )
                    )
                ),
                "dagger_iterations": list(development_rounds),
            }
            for method in methods
        },
        "collection_contract": {
            "sibling_metadata_required": True,
            "train_rounds": list(train_rounds),
            "development_rounds": list(development_rounds),
            "matched_initial_episode_panel_per_split_round": True,
            "round_zero_behavior_head": "forbidden",
            "later_round_behavior_head": "required_from_immediately_previous_round",
        },
        "seed_metrics": seed_metrics,
        "comparisons": comparisons,
        "gripper": {
            "changed_by_any_head": False,
            "contract": "all residual heads preserve the frozen base gripper exactly",
            "expert_gripper_mse_is_supporting_only": True,
        },
        "pairing_policy": (
            "paired bootstrap requires exact task/physical-episode/DAgger-iteration/"
            "decision identity alignment; otherwise independent task/episode "
            "bootstrap is reported; DAgger rounds are always equally weighted"
        ),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparisons, indent=2, sort_keys=True))
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
