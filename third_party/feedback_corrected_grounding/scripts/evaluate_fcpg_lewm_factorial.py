#!/usr/bin/env python3
"""Paired FCPG x LeWM policy rollout on the existing frozen P1 system."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
import numpy as np

from fcpg.action_chunk_policy import load_action_chunk_policy
from fcpg.lewm_p1_native_complementary import (
    FrozenP1NativeCheckpointSpec,
    load_frozen_p1_native_checkpoint,
)
from fcpg.lewm_factorial_prefix_bundle import (
    PrefixBundle,
    load_prefix_bundle,
    write_prefix_bundle,
)
from fcpg.lewm_provenance import sha256_file
from scripts.evaluate_action_chunk_rollout import choose_device
from scripts.evaluate_p1_native_policy_rollout import (
    P1_POLICY_ACTION_STEPS,
    _load_json,
    _resolve,
    rollout_once,
)


PROTOCOL = "fcpg_lewm_factorial_pick_development_v2"
CELL_SPECS = {
    "f0_l0": ("nominal", "geometry_hold"),
    "f1_l0": ("fcpg", "geometry_hold"),
    "f0_l1": ("nominal", "complementary_factual"),
    "f1_l1": ("fcpg", "complementary_factual"),
}
RUN_ORDER = ("f1_l0", "f0_l0", "f0_l1", "f1_l1")
CANONICAL_BASELINE_CELLS = ("f1_l0", "f0_l0")
DYNAMICS_CELLS = ("f0_l1", "f1_l1")
PREFIX_HASH_FIELDS = (
    "prefix_action_sha256",
    "prefix_native_token_sha256",
    "prefix_sampled_visual_sha256",
    "prefix_bundle_sha256",
    "prefix_manifest_sha256",
)


def _moving_goal_contract(config: Mapping[str, object]) -> dict[str, object]:
    raw = config.get("moving_goal")
    if raw is None:
        return {"enabled": False}
    if not isinstance(raw, Mapping):
        raise TypeError("moving_goal must be a mapping")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("moving_goal.enabled must be boolean")
    if not enabled:
        return {"enabled": False}
    seed_base = raw.get("seed_base")
    if isinstance(seed_base, (bool, np.bool_)) or not isinstance(
        seed_base, (int, np.integer)
    ):
        raise TypeError("moving_goal.seed_base must be an integer")
    if int(seed_base) < 0:
        raise ValueError("moving_goal.seed_base must be non-negative")
    amplitude = raw.get("amplitude_xy")
    if (
        not isinstance(amplitude, Sequence)
        or isinstance(amplitude, (str, bytes))
        or len(amplitude) != 2
    ):
        raise ValueError("moving_goal.amplitude_xy must contain two values")
    amplitude_xy = [float(value) for value in amplitude]
    if not np.isfinite(amplitude_xy).all() or any(
        value <= 0.0 for value in amplitude_xy
    ):
        raise ValueError("moving_goal.amplitude_xy must be finite and positive")
    period = float(raw.get("period_raw_steps", 0.0))
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("moving_goal.period_raw_steps must be finite and positive")
    return {
        "enabled": True,
        "seed_base": int(seed_base),
        "amplitude_xy": amplitude_xy,
        "period_raw_steps": period,
    }


def _canonical_json_sha256(value: Mapping[str, object]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    value = result.stdout.strip()
    if len(value) != 40:
        raise RuntimeError("git returned an invalid HEAD")
    return value


def _common_provenance(
    *,
    config_path: Path,
    source_config_path: Path,
    policy_checkpoint: Path,
    corrector_checkpoint: Path,
    config: Mapping[str, object],
    device: str,
) -> dict[str, object]:
    evaluator_path = Path(__file__).resolve()
    rollout_path = evaluator_path.with_name("evaluate_p1_native_policy_rollout.py")
    repo_root = evaluator_path.parents[1]
    return {
        "protocol": PROTOCOL,
        "git_commit": _git_head(Path(__file__).resolve().parents[1]),
        "evaluator_sha256": sha256_file(evaluator_path),
        "rollout_evaluator_sha256": sha256_file(rollout_path),
        "p1_runtime_sha256": sha256_file(
            repo_root / "fcpg" / "lewm_p1_native_policy_runtime.py"
        ),
        "prefix_bundle_module_sha256": sha256_file(
            repo_root / "fcpg" / "lewm_factorial_prefix_bundle.py"
        ),
        "device": str(device),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source_config_path": str(source_config_path),
        "source_config_sha256": sha256_file(source_config_path),
        "policy_checkpoint": str(policy_checkpoint),
        "policy_checkpoint_sha256": sha256_file(policy_checkpoint),
        "corrector_checkpoint": str(corrector_checkpoint),
        "corrector_checkpoint_sha256": sha256_file(corrector_checkpoint),
        "task": str(config["task"]),
        "camera": str(config["camera"]),
        "physical_episode_count": int(config["physical_episode_count"]),
        "model_seeds": [
            int(value) for value in config.get("model_seeds", [config["model_seed"]])
        ],
        "canonical_model_seed": int(
            config.get("canonical_model_seed", config["model_seed"])
        ),
        "benchmark_seed": int(config["benchmark_seed"]),
        "environment_seed_base": int(config["environment_seed_base"]),
        "task_instance_indices": [
            int(value) for value in config["task_instance_indices"]
        ],
        "drop_start_decisions": [
            int(value) for value in config["drop_start_decisions"]
        ],
        "drop_duration_decisions": int(config["drop_duration_decisions"]),
        "horizon_raw_steps": int(config["horizon_raw_steps"]),
        "occlusion_radius_px": int(config["occlusion_radius_px"]),
        "occlusion_value_uint8": int(config["occlusion_value_uint8"]),
        "camera_rotation_axis_world": [
            float(value) for value in config["camera_rotation_axis_world"]
        ],
        "camera_rotation_deg": float(config["camera_rotation_deg"]),
        "moving_goal": _moving_goal_contract(config),
    }


def _episode_prefix_provenance(
    common: Mapping[str, object],
    *,
    episode: int,
    grounding: str,
    config: Mapping[str, object],
) -> dict[str, object]:
    return {
        **dict(common),
        "physical_episode": int(episode),
        "grounding": str(grounding),
        "task_instance_index": int(config["task_instance_indices"][episode]),
        "environment_seed": int(config["environment_seed_base"]) + episode,
        "drop_start_decision": int(config["drop_start_decisions"][episode]),
    }


def _percentile(samples: np.ndarray) -> list[float]:
    return [float(value) for value in np.percentile(samples, [2.5, 97.5])]


def factorial_effects(
    rows: Sequence[Mapping[str, object]],
    *,
    metric: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    """Compute paired main effects and interaction for one higher-is-better metric."""

    indexed = {
        (int(row["physical_episode"]), str(row["method"])): float(row[metric])
        for row in rows
    }
    episodes = sorted({episode for episode, _ in indexed})
    if not episodes:
        raise ValueError("factorial panel is empty")
    if any((episode, cell) not in indexed for episode in episodes for cell in CELL_SPECS):
        raise ValueError("factorial panel is incomplete")
    values = {
        cell: np.asarray([indexed[(episode, cell)] for episode in episodes])
        for cell in CELL_SPECS
    }
    contrasts = {
        "fcpg_main_effect": 0.5
        * ((values["f1_l0"] - values["f0_l0"]) + (values["f1_l1"] - values["f0_l1"])),
        "lewm_main_effect": 0.5
        * ((values["f0_l1"] - values["f0_l0"]) + (values["f1_l1"] - values["f1_l0"])),
        "interaction": values["f1_l1"]
        - values["f1_l0"]
        - values["f0_l1"]
        + values["f0_l0"],
        "full_vs_fcpg_only": values["f1_l1"] - values["f1_l0"],
        "full_vs_lewm_only": values["f1_l1"] - values["f0_l1"],
        "full_vs_neither": values["f1_l1"] - values["f0_l0"],
    }
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(episodes), size=(bootstrap_samples, len(episodes)))
    return {
        "metric": metric,
        "higher_is_better": True,
        "physical_episode_count": len(episodes),
        "cell_means": {cell: float(np.mean(value)) for cell, value in values.items()},
        "contrasts": {
            name: {
                "estimate": float(np.mean(value)),
                "ci95": _percentile(np.mean(value[draws], axis=1)),
            }
            for name, value in contrasts.items()
        },
    }


def crossed_seed_factorial_effects(
    rows: Sequence[Mapping[str, object]],
    *,
    metric: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    """Aggregate a shared episode panel without treating model seeds as episodes."""

    indexed = {
        (
            int(row["model_seed"]),
            int(row["physical_episode"]),
            str(row["method"]),
        ): float(row[metric])
        for row in rows
    }
    model_seeds = sorted({model_seed for model_seed, _, _ in indexed})
    episodes = sorted({episode for _, episode, _ in indexed})
    if not model_seeds or not episodes:
        raise ValueError("crossed factorial panel is empty")
    if any(
        (model_seed, episode, cell) not in indexed
        for model_seed in model_seeds
        for episode in episodes
        for cell in CELL_SPECS
    ):
        raise ValueError("crossed factorial panel is incomplete")
    values = {
        cell: np.asarray(
            [
                [indexed[(model_seed, episode, cell)] for episode in episodes]
                for model_seed in model_seeds
            ],
            dtype=np.float64,
        )
        for cell in CELL_SPECS
    }
    contrasts = {
        "fcpg_main_effect": 0.5
        * ((values["f1_l0"] - values["f0_l0"]) + (values["f1_l1"] - values["f0_l1"])),
        "lewm_main_effect": 0.5
        * ((values["f0_l1"] - values["f0_l0"]) + (values["f1_l1"] - values["f1_l0"])),
        "interaction": values["f1_l1"]
        - values["f1_l0"]
        - values["f0_l1"]
        + values["f0_l0"],
        "full_vs_fcpg_only": values["f1_l1"] - values["f1_l0"],
        "full_vs_lewm_only": values["f1_l1"] - values["f0_l1"],
        "full_vs_neither": values["f1_l1"] - values["f0_l0"],
    }
    rng = np.random.default_rng(seed)
    seed_draws = rng.integers(
        0, len(model_seeds), size=(bootstrap_samples, len(model_seeds))
    )
    episode_draws = rng.integers(
        0, len(episodes), size=(bootstrap_samples, len(episodes))
    )
    return {
        "metric": metric,
        "higher_is_better": True,
        "model_seeds": model_seeds,
        "physical_episode_count": len(episodes),
        "cell_means": {cell: float(np.mean(value)) for cell, value in values.items()},
        "contrasts": {
            name: {
                "estimate": float(np.mean(value)),
                "ci95": _percentile(
                    value[
                        seed_draws[:, :, None],
                        episode_draws[:, None, :],
                    ].mean(axis=(1, 2))
                ),
                "per_model_seed": {
                    str(model_seed): float(np.mean(value[index]))
                    for index, model_seed in enumerate(model_seeds)
                },
            }
            for name, value in contrasts.items()
        },
    }


def _validate_physical_pairing(rows: Sequence[Mapping[str, object]]) -> None:
    grouped: dict[int, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["physical_episode"]), []).append(row)
    for episode, panel in grouped.items():
        methods = [str(row["method"]) for row in panel]
        if len(panel) != len(CELL_SPECS) or len(set(methods)) != len(CELL_SPECS):
            raise RuntimeError(
                f"episode {episode} must contain exactly one row for each cell"
            )
        if set(methods) != set(CELL_SPECS):
            raise RuntimeError(f"episode {episode} does not contain all four cells")
        for field in ("reset_digest", "camera_sha256", "schedule_sha256", "drop_onset_state_sha256"):
            if len({str(row[field]) for row in panel}) != 1:
                raise RuntimeError(f"episode {episode} differs across cells in {field}")
        reference = np.asarray(panel[0]["pre_drop_action_values"], dtype=np.float32)
        for row in panel[1:]:
            current = np.asarray(row["pre_drop_action_values"], dtype=np.float32)
            if not np.array_equal(current, reference):
                raise RuntimeError(f"episode {episode} has unequal pre-drop actions")
        if len({str(row["prefix_action_sha256"]) for row in panel}) != 1:
            raise RuntimeError(f"episode {episode} has unequal action-prefix hashes")
        moving_enabled = {bool(row.get("moving_goal_enabled", False)) for row in panel}
        if len(moving_enabled) != 1:
            raise RuntimeError(f"episode {episode} differs across cells in moving-goal mode")
        if moving_enabled == {True}:
            for field in (
                "moving_goal_seed",
                "moving_goal_amplitude_xy",
                "moving_goal_period_raw_steps",
                "moving_goal_trajectory_sha256",
                "moving_goal_trajectory_points",
                "moving_goal_start_xyz",
                "moving_goal_final_xyz",
            ):
                values = {
                    json.dumps(row.get(field), sort_keys=True, separators=(",", ":"))
                    for row in panel
                }
                if len(values) != 1:
                    raise RuntimeError(
                        f"episode {episode} differs across cells in {field}"
                    )
            trajectory_hash = str(panel[0].get("moving_goal_trajectory_sha256"))
            if len(trajectory_hash) != 64:
                raise RuntimeError(
                    f"episode {episode} has an invalid moving-goal trajectory hash"
                )
            if any(
                float(row.get("moving_goal_max_displacement_m", 0.0)) <= 1.0e-6
                for row in panel
            ):
                raise RuntimeError(
                    f"episode {episode} moving goal did not measurably move"
                )
        indexed = {str(row["method"]): row for row in panel}
        for grounding, cells in {
            "nominal": ("f0_l0", "f0_l1"),
            "fcpg": ("f1_l0", "f1_l1"),
        }.items():
            left, right = (indexed[cell] for cell in cells)
            for field in PREFIX_HASH_FIELDS[1:]:
                if str(left[field]) != str(right[field]):
                    raise RuntimeError(
                        f"episode {episode} {grounding} L0/L1 differ in {field}"
                    )


def _predictor_specs_by_grounding(
    config: Mapping[str, object],
    source: Mapping[str, object],
    *,
    seed: int,
) -> dict[str, tuple[dict[str, object], FrozenP1NativeCheckpointSpec]]:
    configured = config.get("predictors_by_grounding")
    if configured is None:
        training = source["frozen_predictor_training_contract"]
        expected_config = _load_json(training["config"])
        record = next(
            (
                item
                for item in source["frozen_predictors"]
                if int(item["seed"]) == seed
            ),
            None,
        )
        if record is None:
            raise ValueError(f"unknown frozen model seed {seed}")
        shared = (
            expected_config,
            FrozenP1NativeCheckpointSpec(
                seed=seed,
                path=_resolve(record["checkpoint"]),
                sha256=str(record["sha256"]),
            ),
        )
        return {"nominal": shared, "fcpg": shared}

    if not isinstance(configured, Mapping) or set(configured) != {"nominal", "fcpg"}:
        raise ValueError("predictors_by_grounding must define nominal and fcpg")
    result: dict[str, tuple[dict[str, object], FrozenP1NativeCheckpointSpec]] = {}
    for grounding in ("nominal", "fcpg"):
        record = configured[grounding]
        if not isinstance(record, Mapping):
            raise TypeError(f"{grounding} predictor specification must be a mapping")
        checkpoints = record.get("checkpoints")
        if checkpoints is not None:
            if not isinstance(checkpoints, Mapping) or str(seed) not in checkpoints:
                raise ValueError(f"{grounding} predictor has no checkpoint for seed {seed}")
            checkpoint_record = checkpoints[str(seed)]
            if not isinstance(checkpoint_record, Mapping):
                raise TypeError(f"{grounding} checkpoint for seed {seed} must be a mapping")
            record = {**record, **checkpoint_record, "seed": seed}
        if int(record.get("seed", -1)) != seed:
            raise ValueError(f"{grounding} predictor seed does not match model_seed")
        expected_config = _load_json(record["training_config"])
        result[grounding] = (
            expected_config,
            FrozenP1NativeCheckpointSpec(
                seed=seed,
                path=_resolve(record["checkpoint"]),
                sha256=str(record["sha256"]),
            ),
        )
    return result


def _reference_prefix_actions(
    report: Mapping[str, object],
    *,
    episodes: int,
    drop_starts: Sequence[int],
) -> dict[int, np.ndarray]:
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise TypeError("reference prefix report must contain rows")
    selected: dict[int, np.ndarray] = {}
    for row in rows:
        if not isinstance(row, Mapping) or str(row.get("method")) != "f1_l0":
            continue
        episode = int(row["physical_episode"])
        if not 0 <= episode < episodes:
            continue
        value = np.asarray(row["pre_drop_action_values"], dtype=np.float32)
        expected = int(drop_starts[episode]) * P1_POLICY_ACTION_STEPS * 4
        if value.size != expected:
            raise ValueError(
                f"reference episode {episode} has {value.size} pre-drop values, "
                f"expected {expected}"
            )
        selected[episode] = value.reshape(
            int(drop_starts[episode]), P1_POLICY_ACTION_STEPS, 4
        )
    missing = sorted(set(range(episodes)) - set(selected))
    if missing:
        raise ValueError(f"reference prefix report is missing episodes {missing}")
    return selected


def _prefix_row_fields(bundle: PrefixBundle) -> dict[str, str]:
    arrays = bundle.manifest["arrays"]
    if not isinstance(arrays, Mapping):
        raise TypeError("prefix bundle manifest arrays must be a mapping")
    return {
        "prefix_action_sha256": str(arrays["actions"]["sha256"]),
        "prefix_native_token_sha256": str(arrays["native_tokens"]["sha256"]),
        "prefix_sampled_visual_sha256": str(arrays["sampled_visual"]["sha256"]),
        "prefix_bundle_sha256": bundle.bundle_sha256,
        "prefix_manifest_sha256": bundle.manifest_sha256,
    }


def _manifest_key(episode: int, grounding: str) -> str:
    return f"episode_{episode:03d}/{grounding}"


def _predictor_report_records(
    config: Mapping[str, object],
    source: Mapping[str, object],
    pair_specs: Mapping[
        str, tuple[dict[str, object], FrozenP1NativeCheckpointSpec]
    ],
    *,
    seed: int,
) -> dict[str, dict[str, object]]:
    configured = config.get("predictors_by_grounding")
    records: dict[str, dict[str, object]] = {}
    for grounding in ("nominal", "fcpg"):
        if isinstance(configured, Mapping):
            raw = configured[grounding]
            if not isinstance(raw, Mapping):
                raise TypeError(f"{grounding} predictor specification must be a mapping")
            training_config = _resolve(raw["training_config"])
        else:
            training_config = _resolve(
                source["frozen_predictor_training_contract"]["config"]
            )
        spec = pair_specs[grounding][1]
        records[grounding] = {
            "seed": seed,
            "training_config": str(training_config),
            "training_config_sha256": sha256_file(training_config),
            "strict_cache": str(pair_specs[grounding][0]["data"]["strict_cache"]),
            "checkpoint": str(spec.path),
            "checkpoint_sha256": spec.sha256,
        }
    return records


def _load_canonical_baseline_rows(
    path: Path,
    *,
    episodes: int,
    canonical_model_seed: int,
    expected_provenance: Mapping[str, object],
) -> tuple[dict[tuple[int, str], dict[str, object]], str]:
    report = _load_json(path)
    if report.get("protocol") != PROTOCOL or report.get("run_role") != "canonical":
        raise ValueError("canonical baseline report has the wrong protocol or role")
    if int(report.get("model_seed", -1)) != canonical_model_seed:
        raise ValueError("canonical baseline report has the wrong model seed")
    if int(report.get("episodes", -1)) != episodes:
        raise ValueError("canonical baseline report has the wrong episode count")
    if report.get("provenance") != dict(expected_provenance):
        raise ValueError("canonical baseline report provenance mismatch")
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise TypeError("canonical baseline report rows must be a list")
    _validate_physical_pairing(rows)
    indexed: dict[tuple[int, str], dict[str, object]] = {}
    for row in rows:
        key = (int(row["physical_episode"]), str(row["method"]))
        if key in indexed:
            raise ValueError(f"duplicate canonical row {key}")
        indexed[key] = dict(row)
    expected = {
        (episode, cell)
        for episode in range(episodes)
        for cell in CELL_SPECS
    }
    if set(indexed) != expected:
        raise ValueError("canonical baseline report is incomplete")
    return indexed, sha256_file(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/lewm/eval/fcpg_lewm_factorial_pick_matched_all3_dev30_v2.json",
    )
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir")
    parser.add_argument("--prefix-bundle-dir")
    parser.add_argument("--canonical-baseline-report")
    args = parser.parse_args(argv)

    os.environ.setdefault("MUJOCO_GL", "egl")
    config_path = _resolve(args.config)
    config = _load_json(config_path)
    if config.get("schema_version") != 2 or config.get("protocol") != PROTOCOL:
        raise ValueError("factorial config has the wrong schema or protocol")
    if str(args.device).startswith("cuda") and str(args.device) != "cuda:0":
        parser.error("the paired factorial is restricted to cuda:0")
    source_config_path = _resolve(config["source_fresh_gate_config"])
    source = _load_json(source_config_path)
    p1 = source["p1"]
    seed = int(config["model_seed"] if args.model_seed is None else args.model_seed)
    allowed_model_seeds = tuple(
        int(value) for value in config.get("model_seeds", [config["model_seed"]])
    )
    if seed not in allowed_model_seeds:
        parser.error(f"--model-seed must be one of {allowed_model_seeds}")
    canonical_model_seed = int(config["canonical_model_seed"])
    if canonical_model_seed not in allowed_model_seeds:
        raise ValueError("canonical_model_seed must be one of model_seeds")
    run_role = "canonical" if seed == canonical_model_seed else "dynamics_only"
    canonical_report_path = (
        None
        if args.canonical_baseline_report is None
        else _resolve(args.canonical_baseline_report)
    )
    if run_role == "canonical" and canonical_report_path is not None:
        parser.error("the canonical seed cannot reuse another baseline report")
    if run_role == "dynamics_only" and canonical_report_path is None:
        parser.error("non-canonical seeds require --canonical-baseline-report")

    device = choose_device(args.device)
    policy_checkpoint = _resolve(p1["policy_checkpoint"])
    corrector_checkpoint = _resolve(p1["corrector_checkpoint"])
    for label, path, expected in (
        ("policy", policy_checkpoint, p1["policy_checkpoint_sha256"]),
        ("corrector", corrector_checkpoint, p1["corrector_checkpoint_sha256"]),
    ):
        actual = sha256_file(path)
        if actual != str(expected):
            raise ValueError(f"{label} checkpoint SHA-256 mismatch")
    policy, policy_args = load_action_chunk_policy(policy_checkpoint, device)
    policy.set_reliability_override("ones")
    policy.requires_grad_(False).eval()
    pair_specs = _predictor_specs_by_grounding(config, source, seed=seed)
    pairs = {
        grounding: load_frozen_p1_native_checkpoint(
            spec,
            expected_config=expected_config,
            device=device,
        )
        for grounding, (expected_config, spec) in pair_specs.items()
    }

    total = int(config["physical_episode_count"])
    episodes = total if args.episodes is None else int(args.episodes)
    if not 0 < episodes <= total:
        parser.error(f"--episodes must lie in [1,{total}]")
    configured_output_dir = _resolve(config["output_dir"])
    if args.output_dir is not None:
        output_dir = _resolve(args.output_dir)
    elif run_role == "canonical":
        output_dir = configured_output_dir
    else:
        output_dir = configured_output_dir.parent / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix_bundle_dir = _resolve(
        args.prefix_bundle_dir or config["prefix_bundle_dir"]
    )
    common_provenance = _common_provenance(
        config_path=config_path,
        source_config_path=source_config_path,
        policy_checkpoint=policy_checkpoint,
        corrector_checkpoint=corrector_checkpoint,
        config=config,
        device=str(device),
    )
    moving_goal = _moving_goal_contract(config)
    canonical_rows: dict[tuple[int, str], dict[str, object]] = {}
    canonical_report_sha256: str | None = None
    if canonical_report_path is not None:
        canonical_rows, canonical_report_sha256 = _load_canonical_baseline_rows(
            canonical_report_path,
            episodes=episodes,
            canonical_model_seed=canonical_model_seed,
            expected_provenance=common_provenance,
        )

    rows: list[dict[str, object]] = []
    prefix_bundle_manifests: dict[str, str] = {}
    for episode in range(episodes):
        bundles: dict[str, PrefixBundle] = {}
        if run_role == "canonical":
            shared_actions: np.ndarray | None = None
            cells_to_run = RUN_ORDER
        else:
            shared_actions = None
            cells_to_run = DYNAMICS_CELLS
            for cell in CANONICAL_BASELINE_CELLS:
                copied = dict(canonical_rows[(episode, cell)])
                copied["model_seed"] = seed
                rows.append(copied)

        for grounding in ("nominal", "fcpg"):
            if run_role == "dynamics_only":
                expected = _episode_prefix_provenance(
                    common_provenance,
                    episode=episode,
                    grounding=grounding,
                    config=config,
                )
                bundle = load_prefix_bundle(
                    prefix_bundle_dir,
                    episode=episode,
                    grounding=grounding,
                    expected_provenance=expected,
                )
                bundles[grounding] = bundle
                prefix_bundle_manifests[_manifest_key(episode, grounding)] = (
                    bundle.manifest_sha256
                )
        if run_role == "dynamics_only":
            shared_actions = bundles["fcpg"].actions

        for cell in cells_to_run:
            grounding, method = CELL_SPECS[cell]
            bundle = bundles.get(grounding)
            result = rollout_once(
                policy,
                policy_args,
                pair=pairs[grounding],
                method=method,
                result_method=cell,
                visible_grounding_mode=grounding,
                model_seed=seed,
                task=str(config["task"]),
                camera=str(config["camera"]),
                width=int(config["width"]),
                height=int(config["height"]),
                benchmark_seed=int(config["benchmark_seed"]),
                task_index=int(config["task_instance_indices"][episode]),
                environment_seed=int(config["environment_seed_base"]) + episode,
                drop_start=int(config["drop_start_decisions"][episode]),
                drop_duration=int(config["drop_duration_decisions"]),
                horizon=int(config["horizon_raw_steps"]),
                radius_px=int(config["occlusion_radius_px"]),
                value_uint8=int(config["occlusion_value_uint8"]),
                device=device,
                camera_rotation_axis_world=config["camera_rotation_axis_world"],
                camera_rotation_deg=float(config["camera_rotation_deg"]),
                moving_goal_seed=(
                    int(moving_goal["seed_base"]) + episode
                    if moving_goal["enabled"]
                    else None
                ),
                moving_goal_amplitude_xy=moving_goal.get(
                    "amplitude_xy", (0.02, 0.01)
                ),
                moving_goal_period_raw_steps=float(
                    moving_goal.get("period_raw_steps", 220.0)
                ),
                forced_prefix_actions=shared_actions,
                forced_prefix_tokens=(None if bundle is None else bundle.native_tokens),
                forced_prefix_visual=(None if bundle is None else bundle.sampled_visual),
                action_residual_head=None,
                occlusion_center_key="q_true",
                compute_clean_action_metric=True,
                visual_validity_fsm=None,
            )
            returned_tokens = np.ascontiguousarray(
                result.pop("_prefix_native_tokens"), dtype=np.float32
            )
            returned_visual = np.ascontiguousarray(
                result.pop("_prefix_sampled_visual"), dtype=np.float32
            )
            if shared_actions is None:
                if cell != "f1_l0":
                    raise RuntimeError("f1_l0 must define the canonical action prefix")
                shared_actions = np.asarray(
                    result["pre_drop_action_values"], dtype=np.float32
                ).reshape(
                    int(config["drop_start_decisions"][episode]),
                    P1_POLICY_ACTION_STEPS,
                    4,
                )
            shared_actions = np.ascontiguousarray(shared_actions, dtype=np.float32)

            if run_role == "canonical" and cell in CANONICAL_BASELINE_CELLS:
                expected = _episode_prefix_provenance(
                    common_provenance,
                    episode=episode,
                    grounding=grounding,
                    config=config,
                )
                write_prefix_bundle(
                    prefix_bundle_dir,
                    episode=episode,
                    grounding=grounding,
                    actions=shared_actions,
                    native_tokens=returned_tokens,
                    sampled_visual=returned_visual,
                    provenance=expected,
                )
                bundle = load_prefix_bundle(
                    prefix_bundle_dir,
                    episode=episode,
                    grounding=grounding,
                    expected_provenance=expected,
                )
                bundles[grounding] = bundle
            if bundle is None:
                raise RuntimeError(f"missing canonical {grounding} prefix bundle")
            if not np.array_equal(shared_actions, bundle.actions):
                raise RuntimeError("rollout action prefix differs from canonical bundle")
            if not np.array_equal(returned_tokens, bundle.native_tokens):
                raise RuntimeError("rollout native-token prefix differs from canonical bundle")
            if not np.array_equal(returned_visual, bundle.sampled_visual):
                raise RuntimeError("rollout sampled-visual prefix differs from canonical bundle")
            prefix_fields = _prefix_row_fields(bundle)
            if str(result["pre_drop_action_sha256"]) != prefix_fields[
                "prefix_action_sha256"
            ]:
                raise RuntimeError("rollout action hash differs from canonical bundle")
            row = {
                "physical_episode": episode,
                **result,
                **prefix_fields,
            }
            rows.append(row)
            prefix_bundle_manifests[_manifest_key(episode, grounding)] = (
                bundle.manifest_sha256
            )
            print(
                f"episode={episode:02d} cell={cell} success={row['success']:.0f} "
                f"sustained={row['sustained_success']:.0f} reward={row['reward_sum']:.2f}",
                flush=True,
            )

    _validate_physical_pairing(rows)
    metrics = (
        "success",
        "sustained_success",
        "final_success",
        "post_drop_success_fraction",
        "post_drop_maximum_success_streak",
        "reward_sum",
    )
    report = {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "classification": str(
            config.get("classification", "exploratory_factorial_not_confirmation")
        ),
        "config": str(config_path),
        "run_role": run_role,
        "model_seed": seed,
        "canonical_model_seed": canonical_model_seed,
        "canonical_baseline_source": (
            None
            if canonical_report_path is None
            else {
                "path": str(canonical_report_path),
                "sha256": canonical_report_sha256,
            }
        ),
        "provenance": common_provenance,
        "predictor_matching": _predictor_report_records(
            config, source, pair_specs, seed=seed
        ),
        "prefix_bundle_root": str(prefix_bundle_dir),
        "prefix_bundle_manifests": dict(sorted(prefix_bundle_manifests.items())),
        "episodes": episodes,
        "cell_semantics": {
            cell: {"grounding": grounding, "dropout_state": method}
            for cell, (grounding, method) in CELL_SPECS.items()
        },
        "factorial": {
            metric: factorial_effects(
                rows,
                metric=metric,
                bootstrap_samples=int(config["bootstrap_samples"]),
                seed=int(config["bootstrap_seed"]) + offset,
            )
            for offset, metric in enumerate(metrics)
        },
        "pairing_audit": "passed",
        "rows": rows,
    }
    report_path = output_dir / "factorial_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
