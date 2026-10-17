#!/usr/bin/env python3
"""Development rollout for posterior-aware policies trained with DAgger.

The primary condition combines a fixed five-degree camera rotation with a
random-onset local visual interruption.  Methods share the deterministic clean
prefix of each physical episode, then execute their own closed-loop actions.
Head seeds are averaged inside each physical episode before physical episodes
are resampled for paired intervals.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import importlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.action_chunk_policy import load_action_chunk_policy  # noqa: E402
from fcpg.lewm_p1_native_complementary import (  # noqa: E402
    FrozenP1NativeCheckpointSpec,
    load_frozen_p1_native_checkpoint,
)
from scripts.evaluate_p1_native_policy_rollout import (  # noqa: E402
    P1_POLICY_ACTION_STEPS,
    rollout_once,
)


PROTOCOL = "fcpg_lewm_posterior_dagger_v1"
PRIMARY_METRIC = "sustained_success"
DAGGER_METHODS = (
    "complementary_factual",
    "complementary_no_action",
    "geometry_hold",
)
METHOD_FAMILIES = (
    "factual_base",
    "factual_dagger",
    "no_action_dagger",
    "geometry_dagger",
)
REPORT_METRICS = (
    "success",
    "sustained_success",
    "final_success",
    "success_fraction",
    "reward_sum",
    "post_drop_success_fraction",
    "post_drop_maximum_success_streak",
    "recovery_exact_native_bypass",
)
PAIRING_FIELDS = (
    "task_index",
    "environment_seed",
    "drop_start",
    "drop_duration",
    "camera_rotation_deg",
    "camera_rotation_axis_world",
)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _json(path: str | Path) -> dict[str, object]:
    value = json.loads(_resolve(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _condition_name(condition: Mapping[str, object]) -> str:
    value = str(condition.get("name", "")).strip()
    if not value:
        raise ValueError("every rollout condition needs a non-empty name")
    return value


def rollout_conditions(config: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    """Normalize and validate the primary joint condition and optional controls."""

    rollout = _mapping(config.get("rollout"), label="rollout")
    configured = rollout.get("conditions")
    if configured is None:
        configured = (
            {
                "name": "joint_rotation5_local_dropout",
                "primary": True,
                "dropout_enabled": True,
                "camera_rotation_deg": rollout.get(
                    "camera_rotation_deg", config.get("camera_rotation_deg", 5.0)
                ),
                "camera_rotation_axis_world": rollout.get(
                    "camera_rotation_axis_world",
                    config.get("camera_rotation_axis_world", (0.0, 0.0, 1.0)),
                ),
            },
        )
    if not isinstance(configured, Sequence) or isinstance(configured, (str, bytes)):
        raise TypeError("rollout.conditions must be a sequence of mappings")

    normalized: list[dict[str, object]] = []
    names: set[str] = set()
    primary_count = 0
    for raw in configured:
        condition = dict(_mapping(raw, label="rollout condition"))
        name = _condition_name(condition)
        if name in names:
            raise ValueError(f"duplicate rollout condition {name!r}")
        names.add(name)
        primary = bool(condition.get("primary", False))
        dropout_enabled = bool(condition.get("dropout_enabled", True))
        degrees = float(condition.get("camera_rotation_deg", 0.0))
        axis = np.asarray(
            condition.get("camera_rotation_axis_world", (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) <= 0:
            raise ValueError(f"condition {name!r} has an invalid camera rotation axis")
        if not np.isfinite(degrees):
            raise ValueError(f"condition {name!r} has a non-finite camera rotation")
        if primary:
            primary_count += 1
            if not dropout_enabled or not np.isclose(abs(degrees), 5.0, atol=1.0e-8):
                raise ValueError(
                    "the primary condition must combine local dropout with a "
                    "five-degree camera rotation"
                )
        normalized.append(
            {
                **condition,
                "name": name,
                "primary": primary,
                "dropout_enabled": dropout_enabled,
                "camera_rotation_deg": degrees,
                "camera_rotation_axis_world": axis.astype(float).tolist(),
            }
        )
    if primary_count != 1:
        raise ValueError("rollout matrix requires exactly one primary condition")
    return tuple(normalized)


def _validate_config(config: Mapping[str, object]) -> None:
    if int(config.get("schema_version", -1)) != 1 or config.get("protocol") != PROTOCOL:
        raise ValueError("posterior DAgger rollout config has the wrong schema/protocol")
    if config.get("classification") not in {None, "development_only"}:
        raise ValueError("posterior DAgger rollout is development-only")
    if int(config.get("action_steps_per_decision", -1)) != P1_POLICY_ACTION_STEPS:
        raise ValueError("posterior DAgger rollout requires five-action decisions")
    duration = int(config.get("drop_duration_decisions", -1))
    start_min = int(config.get("drop_start_min_decision", -1))
    start_max = int(config.get("drop_start_max_decision", -1))
    if duration <= 0 or start_min < 3 or start_max < start_min:
        raise ValueError("invalid random local-dropout schedule")
    if int(config.get("occlusion_radius_px", 0)) <= 0:
        raise ValueError("local dropout requires a positive occlusion radius")
    if str(config.get("occlusion_center_key", "")) != "q_nominal":
        raise ValueError("posterior DAgger dropout center must be q_nominal")

    rollout = _mapping(config.get("rollout"), label="rollout")
    episodes = int(rollout.get("episodes", 0))
    horizon = int(rollout.get("horizon_raw_steps", 0))
    if episodes <= 0:
        raise ValueError("rollout requires physical episodes")
    if horizon <= 0 or horizon % P1_POLICY_ACTION_STEPS:
        raise ValueError("rollout horizon must contain complete action decisions")
    if start_max + duration >= horizon // P1_POLICY_ACTION_STEPS:
        raise ValueError("rollout must include at least one recovery decision")
    if int(rollout.get("bootstrap_samples", 0)) <= 0:
        raise ValueError("rollout requires bootstrap samples")

    training = _mapping(config.get("training"), label="training")
    seeds = tuple(int(value) for value in training.get("seeds", ()))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("training.seeds must contain unique head seeds")
    if int(training.get("steps", 0)) <= 0:
        raise ValueError("training.steps must be positive")
    if "dagger_iteration" in training and int(training["dagger_iteration"]) < 0:
        raise ValueError("training.dagger_iteration must be non-negative")
    rollout_conditions(config)


def condition_rollout_kwargs(condition: Mapping[str, object]) -> dict[str, object]:
    """Return the camera arguments passed verbatim to ``rollout_once``."""

    return {
        "camera_rotation_deg": float(condition["camera_rotation_deg"]),
        "camera_rotation_axis_world": tuple(
            float(value) for value in condition["camera_rotation_axis_world"]
        ),
    }


def _method_labels(head_seeds: Sequence[int]) -> tuple[str, ...]:
    seeds = tuple(int(value) for value in head_seeds)
    return (
        "factual_base",
        *(f"factual_dagger_s{seed}" for seed in seeds),
        *(f"no_action_dagger_s{seed}" for seed in seeds),
        *(f"geometry_dagger_s{seed}" for seed in seeds),
    )


def _family_methods(family: str, head_seeds: Sequence[int]) -> tuple[str, ...]:
    seeds = tuple(int(value) for value in head_seeds)
    if family == "factual_base":
        return ("factual_base",)
    prefix = {
        "factual_dagger": "factual_dagger",
        "no_action_dagger": "no_action_dagger",
        "geometry_dagger": "geometry_dagger",
    }.get(family)
    if prefix is None:
        raise ValueError(f"unknown method family {family!r}")
    return tuple(f"{prefix}_s{seed}" for seed in seeds)


def _episode_key(row: Mapping[str, object]) -> tuple[str, int]:
    return str(row.get("task", "")), int(row["physical_episode"])


def validate_closed_loop_panel(
    rows: Sequence[Mapping[str, object]],
    *,
    condition: str,
    head_seeds: Sequence[int],
    dropout_enabled: bool,
) -> tuple[tuple[str, int], ...]:
    """Require a complete paired prefix while allowing trajectories to diverge."""

    selected = [row for row in rows if str(row.get("condition")) == condition]
    if not selected:
        raise ValueError(f"rollout panel has no rows for condition {condition!r}")
    expected_methods = set(_method_labels(head_seeds))
    indexed: dict[tuple[tuple[str, int], str], Mapping[str, object]] = {}
    for row in selected:
        method = str(row["method"])
        if method not in expected_methods:
            raise ValueError(f"unexpected rollout method {method!r}")
        key = (_episode_key(row), method)
        if key in indexed:
            raise ValueError(f"duplicate rollout row for {key}")
        indexed[key] = row

        head_seed = row.get("head_seed")
        if method == "factual_base":
            if head_seed is not None:
                raise ValueError("factual_base must not carry a head seed")
        else:
            encoded = int(method.rsplit("_s", 1)[1])
            if int(head_seed) != encoded or encoded not in set(map(int, head_seeds)):
                raise ValueError(f"method/head-seed identity mismatch for {method!r}")

    episodes = tuple(sorted({_episode_key(row) for row in selected}))
    for episode in episodes:
        missing = [
            method for method in sorted(expected_methods) if (episode, method) not in indexed
        ]
        if missing:
            raise ValueError(f"incomplete rollout panel for {episode}: {missing}")
        panel = [indexed[(episode, method)] for method in sorted(expected_methods)]
        for field in PAIRING_FIELDS:
            values = {str(row[field]) for row in panel}
            if len(values) != 1:
                raise ValueError(
                    f"paired episode {episode} differs before intervention in {field}"
                )
        reference = np.asarray(panel[0]["pre_drop_action_values"], dtype=np.float32)
        for row in panel[1:]:
            current = np.asarray(row["pre_drop_action_values"], dtype=np.float32)
            if current.shape != reference.shape or not np.allclose(
                current, reference, rtol=0.0, atol=1.0e-6
            ):
                raise ValueError(f"paired episode {episode} has a divergent clean prefix")
        if dropout_enabled:
            if not all(bool(row["recovery_exact_native_bypass"]) for row in panel):
                raise ValueError(f"paired episode {episode} failed recovery bypass")
            active_counts = {int(row["active_decisions_observed"]) for row in panel}
            if len(active_counts) != 1 or next(iter(active_counts)) <= 0:
                raise ValueError(f"paired episode {episode} has inconsistent dropout support")

    actual_keys = set(indexed)
    expected_keys = {
        (episode, method) for episode in episodes for method in expected_methods
    }
    if actual_keys != expected_keys:
        raise ValueError("rollout panel contains unmatched physical episodes")
    return episodes


def _percentile(samples: np.ndarray) -> list[float]:
    low, high = np.quantile(np.asarray(samples, dtype=np.float64), (0.025, 0.975))
    return [float(low), float(high)]


def summarize_condition(
    rows: Sequence[Mapping[str, object]],
    *,
    condition: str,
    head_seeds: Sequence[int],
    dropout_enabled: bool,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    """Seed-macro first, then paired-bootstrap independent physical episodes."""

    if int(bootstrap_samples) <= 0:
        raise ValueError("bootstrap_samples must be positive")
    episodes = validate_closed_loop_panel(
        rows,
        condition=condition,
        head_seeds=head_seeds,
        dropout_enabled=dropout_enabled,
    )
    selected = [row for row in rows if str(row.get("condition")) == condition]
    indexed = {(_episode_key(row), str(row["method"])): row for row in selected}

    episode_values: dict[str, dict[str, np.ndarray]] = {}
    for family in METHOD_FAMILIES:
        methods = _family_methods(family, head_seeds)
        metric_values: dict[str, np.ndarray] = {}
        for metric in REPORT_METRICS:
            values = np.asarray(
                [
                    np.mean(
                        [float(indexed[(episode, method)][metric]) for method in methods]
                    )
                    for episode in episodes
                ],
                dtype=np.float64,
            )
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite {metric} in method family {family}")
            metric_values[metric] = values
        episode_values[family] = metric_values

    rng = np.random.default_rng(int(seed))
    draws = rng.integers(
        0, len(episodes), size=(int(bootstrap_samples), len(episodes))
    )
    method_summary = {
        family: {
            metric: float(np.mean(values[metric])) for metric in REPORT_METRICS
        }
        for family, values in episode_values.items()
    }
    comparisons: dict[str, object] = {}
    for baseline in ("factual_base", "no_action_dagger", "geometry_dagger"):
        name = f"factual_dagger_vs_{baseline}"
        metrics: dict[str, object] = {}
        for metric in REPORT_METRICS:
            delta = (
                episode_values["factual_dagger"][metric]
                - episode_values[baseline][metric]
            )
            samples = np.mean(delta[draws], axis=1)
            metrics[metric] = {
                "delta": float(np.mean(delta)),
                "ci": _percentile(samples),
                "one_sided_95_lower": float(np.quantile(samples, 0.05)),
            }
        comparisons[name] = {
            "candidate": "factual_dagger",
            "baseline": baseline,
            "metrics": metrics,
        }

    return {
        "condition": condition,
        "primary_metric": PRIMARY_METRIC,
        "independent_unit": "physical_episode",
        "physical_episode_count": len(episodes),
        "head_seed_count": len(tuple(head_seeds)),
        "aggregation": "mean_head_seeds_within_episode_then_paired_episode_bootstrap",
        "method_summary": method_summary,
        "comparisons": comparisons,
    }


def validate_checkpoint_metadata(
    metadata: Mapping[str, object],
    *,
    method: str,
    seed: int,
    steps: int,
    protocol: str,
    dagger_iteration: int | None,
) -> dict[str, object]:
    """Fail closed on exact DAgger checkpoint identity and fixed endpoint."""

    actual_method = str(metadata.get("method", ""))
    actual_seed = int(metadata.get("seed", -1))
    actual_protocol = metadata.get("protocol")
    step_values = [
        int(metadata[name])
        for name in ("steps", "optimizer_step", "step")
        if name in metadata
    ]
    if actual_method != method:
        raise ValueError(f"checkpoint method mismatch: {actual_method!r} != {method!r}")
    if actual_seed != int(seed):
        raise ValueError(f"checkpoint seed mismatch: {actual_seed} != {int(seed)}")
    # The public loader validates the protocol from the checkpoint payload but
    # intentionally omits it from returned metadata.  Validate it when present
    # without rejecting the loader's canonical metadata shape.
    if actual_protocol is not None and str(actual_protocol) != protocol:
        raise ValueError(
            f"checkpoint protocol mismatch: {actual_protocol!r} != {protocol!r}"
        )
    if not step_values or len(set(step_values)) != 1 or step_values[0] != int(steps):
        raise ValueError(
            f"checkpoint is not latest@{int(steps)}: recorded steps={step_values}"
        )
    if dagger_iteration is not None:
        actual_iteration = int(metadata.get("dagger_iteration", -1))
        if actual_iteration != int(dagger_iteration):
            raise ValueError(
                "checkpoint DAgger iteration mismatch: "
                f"{actual_iteration} != {dagger_iteration}"
            )
    return dict(metadata)


def _dagger_api() -> tuple[str, tuple[str, ...], Callable[..., object]]:
    try:
        module = importlib.import_module("fcpg.lewm_posterior_dagger")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "fcpg.lewm_posterior_dagger must be implemented before rollout"
        ) from error
    protocol = str(getattr(module, "PROTOCOL", ""))
    methods = tuple(
        str(value)
        for value in getattr(module, "METHODS", getattr(module, "DAGGER_METHODS", ()))
    )
    loader = getattr(module, "load_posterior_dagger_checkpoint", None)
    if not protocol or methods != DAGGER_METHODS or not callable(loader):
        raise RuntimeError(
            "posterior DAgger API must export PROTOCOL, matched METHODS, and "
            "load_posterior_dagger_checkpoint"
        )
    return protocol, methods, loader


def _checkpoint_paths(
    training: Mapping[str, object],
    *,
    seeds: Sequence[int],
    dagger_iterations: Mapping[str, int] | None = None,
) -> dict[tuple[str, int], Path]:
    expected = {(method, int(seed)) for method in DAGGER_METHODS for seed in seeds}
    records = training.get("checkpoints")
    paths: dict[tuple[str, int], Path] = {}
    if records is not None:
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise TypeError("training.checkpoints must be a sequence")
        for raw in records:
            record = _mapping(raw, label="checkpoint record")
            key = (str(record["method"]), int(record["seed"]))
            if key not in expected:
                if key[0] not in DAGGER_METHODS:
                    raise ValueError(f"unknown checkpoint method in record {key}")
                continue
            if key in paths:
                raise ValueError(f"duplicate checkpoint record for {key}")
            paths[key] = _resolve(str(record["path"]))
    else:
        output_dir = _resolve(str(training["output_dir"]))
        pattern = str(
            training.get("checkpoint_pattern", "{method}_seed{seed}_latest.pt")
        )
        steps = int(training["steps"])
        for method, head_seed in expected:
            dagger_iteration = int(
                (dagger_iterations or {}).get(
                    method, int(training.get("dagger_iteration", 0))
                )
            )
            paths[(method, head_seed)] = output_dir / pattern.format(
                method=method,
                seed=head_seed,
                steps=steps,
                iteration=dagger_iteration,
                dagger_iteration=dagger_iteration,
            )
    if set(paths) != expected:
        missing = sorted(expected - set(paths))
        extra = sorted(set(paths) - expected)
        raise ValueError(f"checkpoint panel mismatch; missing={missing}, extra={extra}")
    if len(set(paths.values())) != len(paths):
        raise ValueError("each method/seed must use a distinct checkpoint path")
    for (method, seed), path in paths.items():
        if not path.name.endswith("_latest.pt"):
            raise ValueError(
                "posterior DAgger rollout requires fixed latest checkpoints; "
                f"got {path.name!r} for {(method, seed)}"
            )
    return paths


def _expected_dagger_iterations(
    training: Mapping[str, object], *, protocol: str
) -> dict[str, int]:
    """Resolve the fixed latest DAgger iteration from config or trainer report."""

    if "dagger_iteration" in training:
        iteration = int(training["dagger_iteration"])
        if iteration < 0:
            raise ValueError("training.dagger_iteration must be non-negative")
        return {method: iteration for method in DAGGER_METHODS}

    report_path = _resolve(
        str(
            training.get(
                "report_path",
                _resolve(str(training["output_dir"])) / "report.json",
            )
        )
    )
    if not report_path.is_file():
        raise FileNotFoundError(
            "training.dagger_iteration is absent and the trainer report is missing: "
            f"{report_path}"
        )
    report = _json(report_path)
    if str(report.get("protocol", "")) != protocol:
        raise ValueError("posterior DAgger trainer report protocol mismatch")
    if int(report.get("optimizer_steps", -1)) != int(training["steps"]):
        raise ValueError("posterior DAgger trainer report endpoint mismatch")
    if tuple(str(value) for value in report.get("methods", ())) != DAGGER_METHODS:
        raise ValueError("posterior DAgger trainer report method ordering mismatch")
    configured_seeds = tuple(int(value) for value in training["seeds"])
    if tuple(int(value) for value in report.get("seeds", ())) != configured_seeds:
        raise ValueError("posterior DAgger trainer report seed panel mismatch")

    report_training = _mapping(report.get("training"), label="trainer report training")
    iterations: dict[str, int] = {}
    for method in DAGGER_METHODS:
        method_report = _mapping(
            report_training.get(method), label=f"trainer report {method}"
        )
        values = method_report.get("dagger_iterations")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"trainer report lacks DAgger iterations for {method}")
        parsed = tuple(int(value) for value in values)
        if not parsed or min(parsed) < 0 or tuple(sorted(set(parsed))) != parsed:
            raise ValueError(f"trainer report has invalid DAgger iterations for {method}")
        iterations[method] = parsed[-1]
    return iterations


def _call_head_loader(
    loader: Callable[..., object],
    path: Path,
    *,
    device: torch.device,
    method: str,
    seed: int,
    steps: int,
    dagger_iteration: int | None,
) -> tuple[torch.nn.Module, Mapping[str, object]]:
    parameters = inspect.signature(loader).parameters
    kwargs: dict[str, object] = {}
    for name, value in (
        ("device", device),
        ("expected_method", method),
        ("expected_seed", seed),
        ("expected_steps", steps),
        ("expected_dagger_iteration", dagger_iteration),
    ):
        if name in parameters and value is not None:
            kwargs[name] = value
    loaded = loader(path, **kwargs)
    if (
        not isinstance(loaded, tuple)
        or len(loaded) != 2
        or not isinstance(loaded[0], torch.nn.Module)
        or not isinstance(loaded[1], Mapping)
    ):
        raise TypeError("posterior DAgger loader must return (module, metadata)")
    return loaded[0], loaded[1]


def _load_heads(
    config: Mapping[str, object],
    *,
    seeds: Sequence[int],
    device: torch.device,
) -> tuple[dict[tuple[str, int], torch.nn.Module], list[dict[str, object]]]:
    training = _mapping(config["training"], label="training")
    protocol, _methods, loader = _dagger_api()
    expected_protocol = str(training.get("checkpoint_protocol", protocol))
    if expected_protocol != protocol:
        raise ValueError("configured checkpoint protocol differs from DAgger module")
    expected_iterations = _expected_dagger_iterations(training, protocol=protocol)
    paths = _checkpoint_paths(
        training, seeds=seeds, dagger_iterations=expected_iterations
    )
    expected_steps = int(training["steps"])
    heads: dict[tuple[str, int], torch.nn.Module] = {}
    records: list[dict[str, object]] = []
    for method in DAGGER_METHODS:
        expected_iteration = expected_iterations[method]
        for head_seed in seeds:
            path = paths[(method, int(head_seed))]
            if not path.is_file():
                raise FileNotFoundError(f"missing posterior DAgger checkpoint: {path}")
            head, raw_metadata = _call_head_loader(
                loader,
                path,
                device=device,
                method=method,
                seed=int(head_seed),
                steps=expected_steps,
                dagger_iteration=expected_iteration,
            )
            metadata = validate_checkpoint_metadata(
                raw_metadata,
                method=method,
                seed=int(head_seed),
                steps=expected_steps,
                protocol=protocol,
                dagger_iteration=expected_iteration,
            )
            if head.training or any(parameter.requires_grad for parameter in head.parameters()):
                raise ValueError("loaded posterior DAgger head must be frozen and eval-only")
            heads[(method, int(head_seed))] = head
            records.append({"path": str(path), **metadata})
    return heads, records


def _load_frozen_components(
    config: Mapping[str, object], device: torch.device
) -> tuple[torch.nn.Module, Mapping[str, object], object]:
    source = _json(str(config["source_fresh_gate_config"]))
    p1 = _mapping(source.get("p1"), label="source p1")
    policy, policy_args = load_action_chunk_policy(
        _resolve(str(p1["policy_checkpoint"])), device
    )
    policy.set_reliability_override("ones")
    policy.requires_grad_(False).eval()

    training = _mapping(
        source.get("frozen_predictor_training_contract"),
        label="predictor training contract",
    )
    expected_config = _json(str(training["config"]))
    records = source.get("frozen_predictors")
    if not isinstance(records, list):
        raise TypeError("source config lacks frozen predictor records")
    model_seed = int(config["model_seed"])
    matches = [record for record in records if int(record["seed"]) == model_seed]
    if len(matches) != 1:
        raise ValueError(f"frozen predictor seed {model_seed} is unavailable or ambiguous")
    record = matches[0]
    pair = load_frozen_p1_native_checkpoint(
        FrozenP1NativeCheckpointSpec(
            seed=model_seed,
            path=_resolve(str(record["checkpoint"])),
            sha256=str(record["sha256"]),
        ),
        expected_config=expected_config,
        device=device,
    )
    return policy, policy_args, pair


def _drop_starts(
    config: Mapping[str, object], *, episodes: int, condition_index: int
) -> np.ndarray:
    rollout = _mapping(config["rollout"], label="rollout")
    rng = np.random.default_rng(
        int(rollout["schedule_seed"]) + 1009 * int(condition_index)
    )
    return rng.integers(
        int(config["drop_start_min_decision"]),
        int(config["drop_start_max_decision"]) + 1,
        size=int(episodes),
        dtype=np.int64,
    )


def _specifications(
    heads: Mapping[tuple[str, int], torch.nn.Module],
    *,
    seeds: Sequence[int],
    dropout_enabled: bool,
) -> tuple[tuple[str, str, int | None, torch.nn.Module | None], ...]:
    factual_runtime = "complementary_factual" if dropout_enabled else "clean_native"
    no_action_runtime = "complementary_no_action" if dropout_enabled else "clean_native"
    geometry_runtime = "geometry_hold" if dropout_enabled else "clean_native"
    values: list[tuple[str, str, int | None, torch.nn.Module | None]] = [
        ("factual_base", factual_runtime, None, None)
    ]
    for seed in seeds:
        values.extend(
            (
                (
                    f"factual_dagger_s{seed}",
                    factual_runtime,
                    int(seed),
                    heads[("complementary_factual", int(seed))],
                ),
                (
                    f"no_action_dagger_s{seed}",
                    no_action_runtime,
                    int(seed),
                    heads[("complementary_no_action", int(seed))],
                ),
                (
                    f"geometry_dagger_s{seed}",
                    geometry_runtime,
                    int(seed),
                    heads[("geometry_hold", int(seed))],
                ),
            )
        )
    return tuple(values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/lewm/eval/posterior_dagger_pick_dev.json"
    )
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--head-seeds", type=int, nargs="+")
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.device not in {"cpu", "cuda:0"}:
        parser.error("posterior DAgger rollout may use only cpu or cuda:0")
    os.environ.setdefault("MUJOCO_GL", "egl")
    config = _json(args.config)
    _validate_config(config)
    rollout = _mapping(config["rollout"], label="rollout")
    training = _mapping(config["training"], label="training")
    all_seeds = tuple(int(value) for value in training["seeds"])
    seeds = all_seeds if args.head_seeds is None else tuple(args.head_seeds)
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds).issubset(all_seeds):
        parser.error("--head-seeds must be unique members of training.seeds")
    configured_episodes = int(rollout["episodes"])
    episodes = configured_episodes if args.episodes is None else int(args.episodes)
    if not 0 < episodes <= configured_episodes:
        parser.error(f"--episodes must lie in [1,{configured_episodes}]")

    all_conditions = rollout_conditions(config)
    selected_names = (
        {str(value) for value in args.conditions}
        if args.conditions is not None
        else {str(value["name"]) for value in all_conditions}
    )
    if not selected_names or not selected_names.issubset(
        {str(value["name"]) for value in all_conditions}
    ):
        parser.error("--conditions contains an unknown condition")
    conditions = tuple(
        value for value in all_conditions if str(value["name"]) in selected_names
    )
    primary = next(value for value in all_conditions if bool(value["primary"]))
    if str(primary["name"]) not in selected_names:
        parser.error("the primary joint drift+drop condition cannot be omitted")

    output_dir = _resolve(args.output_dir or str(rollout["output_dir"]))
    report_path = output_dir / "report.json"
    progress_path = output_dir / "progress.jsonl"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {report_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path.write_text("", encoding="utf-8")

    device = torch.device(args.device)
    policy, policy_args, pair = _load_frozen_components(config, device)
    heads, head_records = _load_heads(config, seeds=seeds, device=device)
    rows: list[dict[str, object]] = []

    for condition_index, condition in enumerate(conditions):
        condition_name = str(condition["name"])
        dropout_enabled = bool(condition["dropout_enabled"])
        drop_starts = _drop_starts(
            config, episodes=episodes, condition_index=condition_index
        )
        specifications = _specifications(
            heads, seeds=seeds, dropout_enabled=dropout_enabled
        )
        for episode in range(episodes):
            shared_actions: np.ndarray | None = None
            shared_tokens: np.ndarray | None = None
            shared_visual: np.ndarray | None = None
            for label, observer_method, head_seed, head in specifications:
                result = rollout_once(
                    policy,
                    policy_args,
                    pair=pair,
                    method=observer_method,
                    model_seed=int(config["model_seed"]),
                    task=str(config["task"]),
                    camera=str(config["camera"]),
                    width=int(config["width"]),
                    height=int(config["height"]),
                    benchmark_seed=int(rollout["benchmark_seed"]),
                    task_index=int(rollout["task_index_base"]) + episode,
                    environment_seed=int(rollout["environment_seed_base"]) + episode,
                    drop_start=int(drop_starts[episode]),
                    drop_duration=int(config["drop_duration_decisions"]),
                    horizon=int(rollout["horizon_raw_steps"]),
                    radius_px=int(config["occlusion_radius_px"]),
                    value_uint8=int(config["occlusion_value_uint8"]),
                    device=device,
                    forced_prefix_actions=shared_actions,
                    forced_prefix_tokens=shared_tokens,
                    forced_prefix_visual=shared_visual,
                    action_residual_head=head,
                    result_method=label,
                    occlusion_center_key=str(config["occlusion_center_key"]),
                    compute_clean_action_metric=False,
                    **condition_rollout_kwargs(condition),
                )
                prefix_tokens = np.asarray(result.pop("_prefix_native_tokens"))
                prefix_visual = np.asarray(result.pop("_prefix_sampled_visual"))
                result = {
                    key: value
                    for key, value in result.items()
                    if key != "reset_digest" and not key.endswith("sha256")
                }
                row = {
                    "condition": condition_name,
                    "condition_primary": bool(condition["primary"]),
                    "dropout_enabled": dropout_enabled,
                    "physical_episode": episode,
                    "head_seed": head_seed,
                    "observer_method": observer_method,
                    **result,
                }
                if shared_actions is None:
                    if label != "factual_base":
                        raise RuntimeError("factual_base must define the shared clean prefix")
                    shared_actions = np.asarray(
                        result["pre_drop_action_values"], dtype=np.float32
                    ).reshape(
                        int(drop_starts[episode]), P1_POLICY_ACTION_STEPS, 4
                    )
                    shared_tokens = prefix_tokens.astype(np.float32, copy=False)
                    shared_visual = prefix_visual.astype(np.float32, copy=False)
                rows.append(row)
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                print(
                    f"condition={condition_name} episode={episode:02d} "
                    f"method={label:30s} sustained={row['sustained_success']:.0f} "
                    f"reward={row['reward_sum']:.2f}",
                    flush=True,
                )

    summaries: dict[str, object] = {}
    for index, condition in enumerate(conditions):
        name = str(condition["name"])
        summaries[name] = summarize_condition(
            rows,
            condition=name,
            head_seeds=seeds,
            dropout_enabled=bool(condition["dropout_enabled"]),
            bootstrap_samples=int(rollout["bootstrap_samples"]),
            seed=int(rollout["bootstrap_seed"]) + index,
        )
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "classification": "development_only",
        "publication_claim_allowed": False,
        "primary_condition": str(primary["name"]),
        "primary_metric": PRIMARY_METRIC,
        "known_sensor_validity_only": True,
        "physical_episodes": episodes,
        "head_seeds": list(seeds),
        "head_checkpoints": head_records,
        "conditions": list(conditions),
        "condition_summaries": summaries,
        "claim_boundary": (
            "Development-only known-validity rollout. Methods share only the clean "
            "prefix and execute independent closed-loop trajectories after dropout."
        ),
        "rows": rows,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
