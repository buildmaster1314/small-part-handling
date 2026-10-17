"""Validation-only orchestration for the bounded posterior-adapter Gate.

This module deliberately has no sealed-test argument.  It trains fixed final
endpoints on the preregistered train split, evaluates only the validation and
independent clean cohorts, and delegates every statistical decision to the
frozen statistics and selection modules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import numpy as np
import torch

from fcpg.lewm_anchor_persistent import AnchorPersistentLeWM
from fcpg.lewm_counterfactual_v2_storage import _publish_directory_no_replace
from fcpg.lewm_posterior_gate_cache import (
    PosteriorGateCache,
    concatenate_training_rows,
    load_posterior_gate_cache,
)
from fcpg.lewm_posterior_gate_runtime import (
    ACTION_BLIND_METHOD,
    ACTION_SHUFFLE_METHOD,
    ADAPTER_METHOD,
    FIXED_BETA_METHOD,
    MEASUREMENT_METHOD,
    NO_LEWM_METHOD,
    NO_WRITEBACK_METHOD,
    build_action_shuffle_indices,
    evaluate_posterior_gate_cache,
    posterior_gate_examples_from_cache,
)
from fcpg.lewm_posterior_gate_selection import (
    FIXED_BETA_CANDIDATES,
    OPTIMIZATION_SEEDS,
    TASKS,
    select_validation_only_posterior_gate,
)
from fcpg.lewm_posterior_gate_statistics import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    summarize_posterior_gate_statistics,
)
from fcpg.lewm_posterior_gate_training import (
    POSTERIOR_GATE_MODES,
    PosteriorGateSplitIndices,
    PosteriorGateTrainingConfig,
    load_posterior_gate_checkpoint,
    train_posterior_gate,
)
from fcpg.lewm_provenance import load_yaml_config, sha256_file
from fcpg.lewm_runtime import load_lewm_grounded_checkpoint


PROTOCOL_NAME = "fcpg_lewm_posterior_adapter_validation_gate_v1"
SCHEMA_VERSION = 1
EXPECTED_EXPERIMENT = "anchor_persistent_measurement_dominant_posterior_gate_v1"
EXPECTED_EVALUATION_VIEWS = {
    "train": "perturbed_observation",
    "validation": "perturbed_observation",
    "validation_clean": "canonical_unoccluded_only",
}
FIXED_METHODS = {
    0.25: "fixed_beta_025",
    0.50: "fixed_beta_050",
    0.75: "fixed_beta_075",
    1.00: "fixed_beta_100",
}
PRIMARY_RUNTIME_METHODS = (
    ADAPTER_METHOD,
    NO_LEWM_METHOD,
    ACTION_BLIND_METHOD,
    NO_WRITEBACK_METHOD,
    ACTION_SHUFFLE_METHOD,
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class PosteriorGateInputs:
    config_path: Path
    config: Mapping[str, Any]
    train: PosteriorGateCache
    validation: PosteriorGateCache
    validation_clean: PosteriorGateCache
    lewm_checkpoint: Path
    lewm_checkpoint_sha256: str
    fcpg_checkpoint: Path
    fcpg_checkpoint_sha256: str
    device: torch.device


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value  # type: ignore[return-value]


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    return int(value)


def _float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _absolute(path: str | os.PathLike[str], *, base: Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = base / value
    return value.resolve()


def _sha256_text(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def validate_posterior_gate_config(config: Mapping[str, Any]) -> None:
    """Fail closed if any preregistered training or decision constant changes."""

    if config.get("schema_version") != 1:
        raise ValueError("posterior Gate config schema_version must be 1")
    if config.get("experiment") != EXPECTED_EXPERIMENT:
        raise ValueError("posterior Gate experiment name changed")
    if config.get("status") != "preregistered_before_fresh_collection":
        raise ValueError("posterior Gate config is not preregistered")
    tasks = tuple(config.get("tasks", ()))
    if tasks != TASKS:
        raise ValueError(f"posterior Gate task order must be {TASKS!r}")

    sequence = _mapping(config.get("sequence"), "sequence")
    expected_sequence = {
        "history_size": 3,
        "frame_skip": 5,
        "events": ["o0", "o1", "o2", "r0", "r1"],
        "primary_events": ["o1", "o2"],
        "primary_estimand": "post_update",
        "recovery_safety_event": "r1",
        "directly_affected_anchor": 1,
        "canonical_target_role": "detached_label_only",
        "training_history": "deployment_measurement_forced",
        "recursive_evaluation_updates": 5,
    }
    if dict(sequence) != expected_sequence:
        raise ValueError("posterior Gate sequence contract changed")

    optimization = _mapping(config.get("optimization"), "optimization")
    if tuple(optimization.get("seeds", ())) != OPTIMIZATION_SEEDS:
        raise ValueError("posterior Gate optimization seeds changed")
    expected_optimization = {
        "steps": 2_000,
        "batch_size": 512,
        "optimizer": "adamw",
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-4,
        "gradient_clip_norm": 1.0,
        "checkpoint_selection": "validation_median_seed_after_safety_filter",
        "required_positive_validation_seeds": 4,
        "test_seed_selection_forbidden": True,
    }
    for key, expected in expected_optimization.items():
        actual = optimization.get(key)
        if isinstance(expected, float):
            if _float(actual, f"optimization.{key}") != expected:
                raise ValueError(f"optimization.{key} changed")
        elif actual != expected:
            raise ValueError(f"optimization.{key} changed")

    adapter = _mapping(config.get("adapter"), "adapter")
    if adapter.get("hidden_dim") != 32 or adapter.get("max_parameters") != 20_000:
        raise ValueError("posterior adapter architecture budget changed")
    if adapter.get("posterior") != "z_measurement + beta * (z_prior - z_measurement)":
        raise ValueError("posterior update equation changed")
    if adapter.get("action_summary_dim") != 20:
        raise ValueError("posterior adapter action summary dimension changed")

    expected_loss_masses = {
        "primary_direct_o0": 0.20,
        "primary_direct_o1": 0.20,
        "primary_direct_o2": 0.20,
        "safety_recovery": 0.20,
        "safety_indirect": 0.20,
    }
    if dict(_mapping(optimization.get("loss_group_masses"), "optimization.loss_group_masses")) != expected_loss_masses:
        raise ValueError("posterior Gate loss group masses changed")
    if optimization.get("loss_group_reduction") != (
        "equal_row_anchor_within_group_from_balanced_task_collection"
    ):
        raise ValueError("posterior Gate loss reduction changed")

    comparators = _mapping(config.get("comparators"), "comparators")
    if tuple(float(value) for value in comparators.get("fixed_beta_candidates", ())) != FIXED_BETA_CANDIDATES:
        raise ValueError("fixed beta candidate set changed")
    if tuple(comparators.get("required_sequential", ())) != (
        "selected_fixed_beta",
        "no_lewm",
    ):
        raise ValueError("required comparator order changed")
    if comparators.get("primary_baseline") != "measurement_writeback":
        raise ValueError("posterior Gate primary baseline changed")
    if tuple(comparators.get("secondary", ())) != (
        "action_blind",
        "action_shuffle",
        "no_writeback",
    ):
        raise ValueError("posterior Gate secondary diagnostic panel changed")

    statistics = _mapping(config.get("statistics"), "statistics")
    expected_statistics = {
        "statistical_unit": "physical_episode",
        "aggregation": "phase_then_event_then_episode_then_equal_task_macro",
        "bootstrap_repetitions": DEFAULT_BOOTSTRAP_SAMPLES,
        "bootstrap_seed": DEFAULT_BOOTSTRAP_SEED,
        "primary_relative_improvement_threshold": 0.10,
        "primary_two_sided_ci_lower_strictly_positive": True,
        "sequential_one_sided_alpha": 0.025,
        "per_task_point_degradation_max": 0.05,
        "per_task_ratio_upper_bound": 1.10,
        "zero_measurement_denominator_is_valid_for_additive_safety": True,
    }
    for key, expected in expected_statistics.items():
        if statistics.get(key) != expected:
            raise ValueError(f"statistics.{key} changed")
    expected_safety = {
        "recovery_safety": {
            "estimand": "post_update",
            "events": ["r1"],
            "normalization": "task_primary_measurement_mse",
            "maximum_normalized_excess_mse": 0.02,
            "one_sided_ci_upper_at_most": 0.02,
        },
        "indirectly_affected_anchor_safety": {
            "estimand": "post_update",
            "events": ["o0", "o1", "o2", "r0", "r1"],
            "normalization": "task_primary_measurement_mse",
            "maximum_normalized_excess_mse": 0.02,
            "one_sided_ci_upper_at_most": 0.02,
        },
        "clean_safety": {
            "estimand": "post_update",
            "events": ["o0", "o1", "o2", "r0", "r1"],
            "normalization": "task_primary_measurement_mse",
            "interpretation": "additive_clean_excess_as_fraction_of_perturbed_primary_measurement_mse",
            "maximum_normalized_excess_mse_strict": 0.02,
            "one_sided_ci_upper_strictly_below": 0.02,
        },
    }
    for key, expected in expected_safety.items():
        if dict(_mapping(statistics.get(key), f"statistics.{key}")) != expected:
            raise ValueError(f"statistics.{key} changed")
    stopping = _mapping(config.get("stopping"), "stopping")
    if dict(stopping) != {
        "one_architecture_only": True,
        "no_hyperparameter_sweep": True,
        "selected_endpoint_safety_failure_stops_route": True,
        "control_experiment_authorized_only_after_full_gate": True,
    }:
        raise ValueError("posterior Gate stopping rules changed")


def _cache_episode_ids(cache: PosteriorGateCache) -> frozenset[str]:
    field = cache.evaluation.get("physical_episode_sha256")
    if field is None:
        raise ValueError(f"cache {cache.path} lacks physical_episode_sha256")
    values = np.asarray(field).astype(str)
    if values.shape != (cache.sample_count,) or any(len(value) != 64 for value in values):
        raise ValueError(f"cache {cache.path} has invalid physical episode hashes")
    return frozenset(values.tolist())


def _cache_tasks(cache: PosteriorGateCache) -> tuple[str, ...]:
    values = np.asarray(cache.evaluation["task_name"]).astype(str)
    return tuple(sorted(set(values.tolist())))


def _cache_episode_counts(cache: PosteriorGateCache) -> dict[str, int]:
    tasks = np.asarray(cache.evaluation["task_name"]).astype(str)
    episode_ids = np.asarray(cache.evaluation["physical_episode_sha256"]).astype(str)
    if tasks.shape != (cache.sample_count,) or episode_ids.shape != tasks.shape:
        raise ValueError(f"cache {cache.path} has invalid episode identity fields")
    grouped: dict[str, set[str]] = {}
    for task, episode_id in zip(tasks.tolist(), episode_ids.tolist(), strict=True):
        grouped.setdefault(task, set()).add(episode_id)
    return {task: len(values) for task, values in grouped.items()}


def _validate_cache_role(
    cache: PosteriorGateCache,
    *,
    split: str,
    expected_checkpoint_sha256: str,
    expected_fcpg_checkpoint_sha256: str,
    expected_episodes_per_task: int,
) -> None:
    contract = _mapping(cache.metadata.get("contract"), f"{split}.contract")
    expected_view = EXPECTED_EVALUATION_VIEWS[split]
    if contract.get("evaluation_view", "perturbed_observation") != expected_view:
        raise ValueError(f"{split} cache evaluation_view must be {expected_view!r}")
    clean_only = bool(contract.get("clean_evaluation_only", False))
    if clean_only != (split == "validation_clean"):
        raise ValueError(f"{split} cache clean_evaluation_only marker is wrong")
    checkpoint = _mapping(
        _mapping(cache.metadata.get("inputs"), f"{split}.inputs").get("checkpoint"),
        f"{split}.inputs.checkpoint",
    )
    if checkpoint.get("sha256") != expected_checkpoint_sha256:
        raise ValueError(f"{split} cache uses a different frozen LeWM checkpoint")
    if checkpoint.get("step") != 2_000:
        raise ValueError(f"{split} cache is not bound to LeWM step 2000")
    inputs = _mapping(cache.metadata.get("inputs"), f"{split}.inputs")
    for name in (
        "observed_correction_cache",
        "measurement_correction_cache",
        "canonical_correction_cache",
    ):
        correction = _mapping(inputs.get(name), f"{split}.inputs.{name}")
        actual = correction.get("corrector_checkpoint_sha256")
        if actual != expected_fcpg_checkpoint_sha256:
            raise ValueError(
                f"{split} cache {name} uses a different frozen FCPG checkpoint"
            )
    if _cache_tasks(cache) != tuple(sorted(TASKS)):
        raise ValueError(f"{split} cache task set changed")
    expected_counts = {task: expected_episodes_per_task for task in TASKS}
    if _cache_episode_counts(cache) != expected_counts:
        raise ValueError(
            f"{split} cache must contain exactly {expected_episodes_per_task} "
            "physical episodes per task"
        )


def load_posterior_gate_inputs(
    config_path: str | os.PathLike[str],
    train_cache: str | os.PathLike[str],
    validation_cache: str | os.PathLike[str],
    validation_clean_cache: str | os.PathLike[str],
    *,
    repo_root: str | os.PathLike[str],
) -> PosteriorGateInputs:
    root = Path(repo_root).expanduser().resolve()
    config_file = _absolute(config_path, base=root)
    config = load_yaml_config(config_file)
    validate_posterior_gate_config(config)
    frozen = _mapping(config.get("frozen_upstream"), "frozen_upstream")
    checkpoint = _absolute(str(frozen.get("lewm_checkpoint")), base=root)
    expected_sha = str(frozen.get("lewm_checkpoint_sha256"))
    expected_sha = _sha256_text(expected_sha, "frozen_upstream.lewm_checkpoint_sha256")
    expected_fcpg_sha = _sha256_text(
        frozen.get("fcpg_checkpoint_sha256"),
        "frozen_upstream.fcpg_checkpoint_sha256",
    )
    fcpg_checkpoint = _absolute(str(frozen.get("fcpg_checkpoint")), base=root)
    if not checkpoint.is_file() or sha256_file(checkpoint) != expected_sha:
        raise ValueError("frozen LeWM checkpoint path/hash mismatch")
    if (
        not fcpg_checkpoint.is_file()
        or sha256_file(fcpg_checkpoint) != expected_fcpg_sha
    ):
        raise ValueError("frozen FCPG checkpoint path/hash mismatch")
    if _integer(frozen.get("expected_lewm_step"), "expected_lewm_step") != 2_000:
        raise ValueError("expected LeWM endpoint changed")

    caches = {
        "train": load_posterior_gate_cache(train_cache),
        "validation": load_posterior_gate_cache(validation_cache),
        "validation_clean": load_posterior_gate_cache(validation_clean_cache),
    }
    splits = _mapping(config.get("splits"), "splits")
    for split, cache in caches.items():
        split_config = _mapping(splits.get(split), f"splits.{split}")
        _validate_cache_role(
            cache,
            split=split,
            expected_checkpoint_sha256=expected_sha,
            expected_fcpg_checkpoint_sha256=expected_fcpg_sha,
            expected_episodes_per_task=_integer(
                split_config.get("episodes_per_task"),
                f"splits.{split}.episodes_per_task",
            ),
        )
    episode_sets = {name: _cache_episode_ids(cache) for name, cache in caches.items()}
    for left, right in (
        ("train", "validation"),
        ("train", "validation_clean"),
        ("validation", "validation_clean"),
    ):
        overlap = episode_sets[left].intersection(episode_sets[right])
        if overlap:
            raise ValueError(f"{left}/{right} physical episodes overlap")

    cache_config = _mapping(config.get("cache"), "cache")
    device = torch.device(str(cache_config.get("device")))
    return PosteriorGateInputs(
        config_path=config_file,
        config=config,
        train=caches["train"],
        validation=caches["validation"],
        validation_clean=caches["validation_clean"],
        lewm_checkpoint=checkpoint,
        lewm_checkpoint_sha256=expected_sha,
        fcpg_checkpoint=fcpg_checkpoint,
        fcpg_checkpoint_sha256=expected_fcpg_sha,
        device=device,
    )


def _rename_method(
    records: Sequence[Mapping[str, Any]], old: str, new: str
) -> list[dict[str, Any]]:
    renamed: list[dict[str, Any]] = []
    for source in records:
        row = dict(source)
        if row.get("method") != old:
            raise ValueError(f"expected only method {old!r} while renaming")
        row["method"] = new
        renamed.append(row)
    return renamed


def merge_posterior_gate_records(
    *groups: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge disjoint method panels and reject duplicate statistic cells."""

    result: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for group in groups:
        for source in group:
            row = dict(source)
            key = tuple(
                row[name]
                for name in (
                    "task",
                    "physical_episode_id",
                    "anchor",
                    "estimand",
                    "event",
                    "method",
                )
            )
            if key in seen:
                raise ValueError(f"duplicate posterior Gate statistic cell: {key}")
            seen.add(key)
            result.append(row)
    return result


def _training_config(
    config: Mapping[str, Any], *, mode: str, seed: int
) -> PosteriorGateTrainingConfig:
    optimization = _mapping(config.get("optimization"), "optimization")
    adapter = _mapping(config.get("adapter"), "adapter")
    return PosteriorGateTrainingConfig(
        mode=mode,  # type: ignore[arg-type]
        seed=seed,
        steps=_integer(optimization.get("steps"), "optimization.steps"),
        batch_size=_integer(
            optimization.get("batch_size"), "optimization.batch_size"
        ),
        learning_rate=_float(
            optimization.get("learning_rate"), "optimization.learning_rate"
        ),
        weight_decay=_float(
            optimization.get("weight_decay"), "optimization.weight_decay"
        ),
        gradient_clip_norm=_float(
            optimization.get("gradient_clip_norm"),
            "optimization.gradient_clip_norm",
        ),
        hidden_dim=_integer(adapter.get("hidden_dim"), "adapter.hidden_dim"),
        max_parameters=_integer(
            adapter.get("max_parameters"), "adapter.max_parameters"
        ),
    )


def _write_json(path: Path, value: object) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.write_text(payload, encoding="utf-8")


def _checkpoint_filename(mode: str, seed: int) -> str:
    return f"posterior_{mode}_seed{seed}_step2000.pt"


def _evaluate_common_records(
    model: AnchorPersistentLeWM,
    cache: PosteriorGateCache,
    adapter: torch.nn.Module,
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    groups: list[Sequence[Mapping[str, Any]]] = []
    measurement = evaluate_posterior_gate_cache(
        model,
        cache,
        adapter=adapter,  # unused by these methods
        methods=(MEASUREMENT_METHOD,),
        device=device,
    )
    groups.append(measurement.records)
    for beta in FIXED_BETA_CANDIDATES:
        evaluated = evaluate_posterior_gate_cache(
            model,
            cache,
            adapter=adapter,
            fixed_beta=beta,
            methods=(FIXED_BETA_METHOD,),
            device=device,
        )
        groups.append(
            _rename_method(evaluated.records, FIXED_BETA_METHOD, FIXED_METHODS[beta])
        )
    return merge_posterior_gate_records(*groups)


def _evaluate_seed_records(
    model: AnchorPersistentLeWM,
    cache: PosteriorGateCache,
    full_adapter: torch.nn.Module,
    action_blind_adapter: torch.nn.Module,
    no_lewm_adapter: torch.nn.Module,
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    examples = posterior_gate_examples_from_cache(cache)
    shuffle = build_action_shuffle_indices(examples.metadata)
    evaluated = evaluate_posterior_gate_cache(
        model,
        cache,
        adapter=full_adapter,  # type: ignore[arg-type]
        action_blind_adapter=action_blind_adapter,  # type: ignore[arg-type]
        no_lewm_adapter=no_lewm_adapter,  # type: ignore[arg-type]
        action_shuffle_indices=shuffle,
        methods=PRIMARY_RUNTIME_METHODS,
        device=device,
    )
    return list(evaluated.records)


def _summary(
    records: Sequence[Mapping[str, Any]],
    clean_records: Sequence[Mapping[str, Any]],
    *,
    adapter_method: str,
    fixed_methods: Sequence[str],
) -> dict[str, Any]:
    result = summarize_posterior_gate_statistics(
        records,
        clean_records,
        adapter_method=adapter_method,
        fixed_methods=fixed_methods,
        no_lewm_method=NO_LEWM_METHOD,
        bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
        bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
    )
    result["split"] = "validation"
    result["source_split"] = "validation"
    result["sealed_data_used"] = False
    return result


def _markdown(decision: Mapping[str, Any]) -> str:
    lines = [
        "# Posterior Adapter Validation Gate",
        "",
        f"Decision: **{decision['decision']}**",
        "",
        "This report uses train, validation, and an independent clean-validation "
        "cohort only. No sealed-test artifact was generated or read.",
        "",
    ]
    reasons = decision.get("reasons", [])
    if reasons:
        lines.extend(("## Failed Conditions", ""))
        for reason in reasons:
            lines.append(f"- `{reason.get('stage', 'gate')}/{reason['code']}`: {reason['message']}")
    return "\n".join(lines) + "\n"


def run_validation_posterior_gate(
    inputs: PosteriorGateInputs,
    output_dir: str | os.PathLike[str],
) -> Mapping[str, Any]:
    """Train/evaluate the one preregistered validation Gate and publish once."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if inputs.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configured CUDA device is unavailable")

    rows, ranges = concatenate_training_rows((inputs.train, inputs.validation))
    if len(ranges) != 2:
        raise RuntimeError("train/validation concatenation contract changed")
    split = PosteriorGateSplitIndices(
        train=tuple(range(*ranges[0])),
        validation=tuple(range(*ranges[1])),
        test=(),
    )
    model, _, checkpoint = load_lewm_grounded_checkpoint(
        inputs.lewm_checkpoint, device=inputs.device
    )
    if not isinstance(model, AnchorPersistentLeWM):
        raise TypeError("posterior Gate requires AnchorPersistentLeWM")
    if checkpoint.get("step") != 2_000:
        raise ValueError("loaded LeWM checkpoint is not step 2000")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    published = False
    try:
        checkpoint_dir = staging / "checkpoints"
        record_dir = staging / "records"
        summary_dir = staging / "summaries"
        checkpoint_dir.mkdir()
        record_dir.mkdir()
        summary_dir.mkdir()

        payloads: dict[tuple[str, int], Mapping[str, Any]] = {}
        checkpoint_records: list[dict[str, Any]] = []
        for seed in OPTIMIZATION_SEEDS:
            for mode in POSTERIOR_GATE_MODES:
                result = train_posterior_gate(
                    rows,
                    split,
                    config=_training_config(inputs.config, mode=mode, seed=seed),
                    device=inputs.device,
                )
                relative = Path("checkpoints") / _checkpoint_filename(mode, seed)
                path = staging / relative
                torch.save(result.checkpoint_payload, path)
                payloads[(mode, seed)] = result.checkpoint_payload
                checkpoint_records.append({
                    "mode": mode,
                    "seed": seed,
                    "path": relative.as_posix(),
                    "train_loss": result.train_loss,
                    "parameter_count": result.parameter_count,
                    "optimizer_steps": result.step,
                })

        seed_summaries: dict[int, dict[str, Any]] = {}
        first_validation_panel: list[dict[str, Any]] | None = None
        first_clean_panel: list[dict[str, Any]] | None = None
        common_validation: list[dict[str, Any]] | None = None
        common_clean: list[dict[str, Any]] | None = None
        for seed in OPTIMIZATION_SEEDS:
            full_adapter = load_posterior_gate_checkpoint(
                payloads[("full", seed)], expected_source_hashes=rows.source_hashes
            ).to(inputs.device)
            action_blind_adapter = load_posterior_gate_checkpoint(
                payloads[("action_blind", seed)],
                expected_source_hashes=rows.source_hashes,
            ).to(inputs.device)
            no_lewm_adapter = load_posterior_gate_checkpoint(
                payloads[("no_lewm", seed)], expected_source_hashes=rows.source_hashes
            ).to(inputs.device)
            full_adapter.eval()
            action_blind_adapter.eval()
            no_lewm_adapter.eval()
            if common_validation is None:
                common_validation = _evaluate_common_records(
                    model, inputs.validation, full_adapter, device=inputs.device
                )
                common_clean = _evaluate_common_records(
                    model, inputs.validation_clean, full_adapter, device=inputs.device
                )
            validation_panel = merge_posterior_gate_records(
                common_validation,
                _evaluate_seed_records(
                    model,
                    inputs.validation,
                    full_adapter,
                    action_blind_adapter,
                    no_lewm_adapter,
                    device=inputs.device,
                ),
            )
            clean_panel = merge_posterior_gate_records(
                common_clean,
                _evaluate_seed_records(
                    model,
                    inputs.validation_clean,
                    full_adapter,
                    action_blind_adapter,
                    no_lewm_adapter,
                    device=inputs.device,
                ),
            )
            summary = _summary(
                validation_panel,
                clean_panel,
                adapter_method=ADAPTER_METHOD,
                fixed_methods=tuple(FIXED_METHODS.values()),
            )
            seed_summaries[seed] = summary
            _write_json(record_dir / f"validation_seed{seed}.json", validation_panel)
            _write_json(record_dir / f"validation_clean_seed{seed}.json", clean_panel)
            _write_json(summary_dir / f"seed{seed}.json", summary)
            if first_validation_panel is None:
                first_validation_panel = validation_panel
                first_clean_panel = clean_panel
            del full_adapter, action_blind_adapter, no_lewm_adapter

        if first_validation_panel is None or first_clean_panel is None:
            raise RuntimeError("no optimization seed was evaluated")
        fixed_summaries: dict[float, dict[str, Any]] = {}
        fixed_names = tuple(FIXED_METHODS.values())
        for beta, method in FIXED_METHODS.items():
            nuisance_fixed = next(name for name in fixed_names if name != method)
            summary = _summary(
                first_validation_panel,
                first_clean_panel,
                adapter_method=method,
                fixed_methods=(nuisance_fixed,),
            )
            fixed_summaries[beta] = summary
            _write_json(summary_dir / f"fixed_beta_{beta:.2f}.json", summary)

        decision = select_validation_only_posterior_gate(
            fixed_summaries, seed_summaries, source_split="validation"
        )
        run_metadata = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL_NAME,
            "config": str(inputs.config_path),
            "frozen_lewm": {
                "path": str(inputs.lewm_checkpoint),
                "sha256": inputs.lewm_checkpoint_sha256,
                "step": 2_000,
            },
            "frozen_fcpg": {
                "path": str(inputs.fcpg_checkpoint),
                "sha256": inputs.fcpg_checkpoint_sha256,
            },
            "caches": {
                "train": str(inputs.train.path),
                "validation": str(inputs.validation.path),
                "validation_clean": str(inputs.validation_clean.path),
            },
            "adapter_checkpoints": checkpoint_records,
            "selection_scope": "validation_only",
            "sealed_data_used": False,
            "sealed_data_generated": False,
        }
        result = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL_NAME,
            "decision": decision,
            "fixed_beta_summaries": fixed_summaries,
            "seed_summaries": seed_summaries,
            "run_metadata": run_metadata,
        }
        _write_json(staging / "decision.json", result)
        (staging / "summary.md").write_text(_markdown(decision), encoding="utf-8")
        _publish_directory_no_replace(staging, destination)
        published = True
        return result
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "EXPECTED_EVALUATION_VIEWS",
    "FIXED_METHODS",
    "PosteriorGateInputs",
    "load_posterior_gate_inputs",
    "merge_posterior_gate_records",
    "run_validation_posterior_gate",
    "validate_posterior_gate_config",
]
