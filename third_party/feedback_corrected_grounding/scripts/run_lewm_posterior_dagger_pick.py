#!/usr/bin/env python3
"""Run the two-round Pick posterior-DAgger development pipeline.

The runner owns orchestration only. Collection, head training, and fresh
closed-loop evaluation remain in their dedicated scripts. Every subprocess is
restricted to physical CUDA device 0, and completed stage outputs are reused
only when the complete expected artifact set is present.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/lewm/eval/posterior_dagger_pick_dev.json"
COLLECTOR = ROOT / "scripts/collect_lewm_posterior_dagger_pick.py"
TRAINER = ROOT / "scripts/train_evaluate_lewm_posterior_dagger.py"
EVALUATOR = ROOT / "scripts/evaluate_lewm_posterior_dagger_rollout.py"
PROTOCOL = "fcpg_lewm_posterior_dagger_v1"
METHODS = (
    "complementary_factual",
    "complementary_no_action",
    "geometry_hold",
)


@dataclass(frozen=True, slots=True)
class StageJob:
    """One resumable subprocess in the fixed DAgger stage order."""

    name: str
    command: tuple[str, ...]
    outputs: tuple[Path, ...]
    log_path: Path


@dataclass(frozen=True, slots=True)
class ProtocolPlan:
    """Expanded runtime configs and commands for one full or smoke run."""

    mode: str
    output_root: Path
    round0_config_path: Path
    final_config_path: Path
    round0_config: dict[str, Any]
    final_config: dict[str, Any]
    jobs: tuple[StageJob, ...]


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    source = _resolve(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"config root must be an object: {source}")
    return value


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result != value:
        raise ValueError(f"{label} must be an integer")
    return result


def _validate_source_contract(config: Mapping[str, object]) -> None:
    source_path = _resolve(str(config["source_fresh_gate_config"]))
    source = load_config(source_path)
    p1 = _mapping(source.get("p1"), label="source.p1")
    policy_path = str(p1.get("policy_checkpoint", ""))
    if "pick_fcpg_pureres_seed0" not in policy_path:
        raise ValueError("source policy must be the frozen Pick policy seed 0")
    if not _resolve(policy_path).is_file():
        raise FileNotFoundError(f"missing frozen policy checkpoint: {policy_path}")

    predictors = source.get("frozen_predictors")
    if not isinstance(predictors, Sequence) or isinstance(predictors, (str, bytes)):
        raise TypeError("source.frozen_predictors must be a list")
    model_seed = int(config["model_seed"])
    matches = [
        _mapping(record, label="frozen predictor")
        for record in predictors
        if isinstance(record, Mapping) and int(record.get("seed", -1)) == model_seed
    ]
    if len(matches) != 1:
        raise ValueError(f"source must bind exactly one LeWM seed {model_seed}")
    checkpoint = _resolve(str(matches[0]["checkpoint"]))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing frozen LeWM checkpoint: {checkpoint}")


def validate_config(config: Mapping[str, object]) -> None:
    """Fail closed on the fixed single-policy, single-LeWM protocol."""

    if _integer(config.get("schema_version", -1), label="schema_version") != 1:
        raise ValueError("unsupported posterior DAgger schema")
    if str(config.get("protocol", "")) != PROTOCOL:
        raise ValueError("posterior DAgger protocol mismatch")
    if str(config.get("classification", "")) != "development_only":
        raise ValueError("posterior DAgger orchestration is development-only")
    if str(config.get("task", "")) != "pick-place-v3":
        raise ValueError("this runner is fixed to pick-place-v3")
    if _integer(config.get("policy_seed", -1), label="policy_seed") != 0:
        raise ValueError("this gate is fixed to policy seed 0")
    if _integer(config.get("model_seed", -1), label="model_seed") != 15801:
        raise ValueError("this gate is fixed to LeWM seed 15801")
    if _integer(
        config.get("action_steps_per_decision", -1),
        label="action_steps_per_decision",
    ) != 5:
        raise ValueError("each policy decision must execute five raw actions")
    if _integer(
        config.get("drop_duration_decisions", -1),
        label="drop_duration_decisions",
    ) != 6:
        raise ValueError("dropout duration must be exactly six decisions")
    if (
        _integer(
            config.get("drop_start_min_decision", -1),
            label="drop_start_min_decision",
        ),
        _integer(
            config.get("drop_start_max_decision", -1),
            label="drop_start_max_decision",
        ),
    ) != (6, 9):
        raise ValueError("dropout start decisions must be exactly 6 through 9")
    if config.get("known_sensor_validity_only") is not True:
        raise ValueError("the development gate requires known sensor validity")
    if str(config.get("occlusion_center_key", "")) != "q_nominal":
        raise ValueError("local dropout must be centered on q_nominal")
    if str(config.get("camera_condition", "")) != "fixed_rotation":
        raise ValueError("collector camera condition must be fixed_rotation")
    if not np.isclose(float(config.get("camera_rotation_deg", 0.0)), 5.0):
        raise ValueError("collector camera rotation must be exactly five degrees")
    axis = np.asarray(config.get("camera_rotation_axis_world"), dtype=np.float64)
    if axis.shape != (3,) or not np.allclose(axis, [0.0, 0.0, 1.0]):
        raise ValueError("collector camera rotation axis must be world z")

    methods = tuple(str(value) for value in config.get("methods", ()))
    if methods != METHODS:
        raise ValueError(f"methods must exactly equal {METHODS}")
    training = _mapping(config.get("training"), label="training")
    seeds = tuple(int(value) for value in training.get("seeds", ()))
    if len(seeds) != 1:
        raise ValueError("the bounded pilot requires exactly one head seed")
    if int(training.get("steps", 0)) != 2_000:
        raise ValueError("the full development endpoint must be latest@2000")

    splits = _mapping(config.get("splits"), label="splits")
    episode_ranges: dict[str, set[int]] = {}
    for split_name in ("train", "development"):
        split = _mapping(splits.get(split_name), label=f"splits.{split_name}")
        episodes = int(split.get("episodes", 0))
        if episodes <= 0:
            raise ValueError(f"splits.{split_name}.episodes must be positive")
        base = int(split.get("environment_seed_base", -1))
        if base < 0:
            raise ValueError(f"splits.{split_name} needs non-negative environment seeds")
        episode_ranges[split_name] = set(range(base, base + episodes))
    if episode_ranges["train"].intersection(episode_ranges["development"]):
        raise ValueError("train and development physical episode seeds overlap")

    rollout = _mapping(config.get("rollout"), label="rollout")
    if int(rollout.get("horizon_raw_steps", 0)) != 335:
        raise ValueError("fresh rollout horizon must be exactly 335 raw steps")
    if int(rollout.get("episodes", 0)) <= 0:
        raise ValueError("fresh rollout requires physical episodes")
    conditions = rollout.get("conditions")
    if not isinstance(conditions, Sequence) or isinstance(conditions, (str, bytes)):
        raise TypeError("rollout.conditions must be a list")
    primary = [
        _mapping(value, label="rollout condition")
        for value in conditions
        if isinstance(value, Mapping) and bool(value.get("primary"))
    ]
    if len(primary) != 1:
        raise ValueError("rollout requires exactly one primary condition")
    condition = primary[0]
    if condition.get("dropout_enabled") is not True:
        raise ValueError("primary rollout must include local dropout")
    if not np.isclose(float(condition.get("camera_rotation_deg", 0.0)), 5.0):
        raise ValueError("primary rollout must include five-degree camera rotation")
    rollout_axis = np.asarray(
        condition.get("camera_rotation_axis_world"), dtype=np.float64
    )
    if rollout_axis.shape != (3,) or not np.allclose(rollout_axis, [0.0, 0.0, 1.0]):
        raise ValueError("primary rollout rotation axis must be world z")

    orchestration = _mapping(config.get("orchestration"), label="orchestration")
    if str(orchestration.get("device", "")) != "cuda:0":
        raise ValueError("orchestration device must be cuda:0")
    if str(orchestration.get("cuda_visible_devices", "")) != "0":
        raise ValueError("CUDA_VISIBLE_DEVICES must expose physical GPU 0 only")
    if int(orchestration.get("policy_seed", -1)) != 0:
        raise ValueError("orchestration policy seed must remain zero")
    if int(orchestration.get("model_seed", -1)) != 15801:
        raise ValueError("orchestration LeWM seed must remain 15801")
    smoke = _mapping(orchestration.get("smoke"), label="orchestration.smoke")
    for key in ("train_episodes", "development_episodes", "rollout_episodes", "steps"):
        if int(smoke.get(key, 0)) <= 0:
            raise ValueError(f"orchestration.smoke.{key} must be positive")

    for script in (COLLECTOR, TRAINER, EVALUATOR):
        if not script.is_file():
            raise FileNotFoundError(f"missing pipeline script: {script}")
    _validate_source_contract(config)


def _collection_path(root: Path, iteration: int, split: str, method: str) -> Path:
    return root / "collection" / f"round{iteration}" / f"{split}_{method}.npz"


def _head_dir(root: Path, stage: str) -> Path:
    return root / "heads" / stage


def _head_path(root: Path, stage: str, method: str, seed: int) -> Path:
    return _head_dir(root, stage) / f"{method}_seed{seed}_latest.pt"


def _runtime_configs(
    config: Mapping[str, object], *, smoke: bool
) -> tuple[Path, dict[str, Any], Path, dict[str, Any], Path]:
    orchestration = _mapping(config["orchestration"], label="orchestration")
    root = _resolve(str(orchestration["output_root"]))
    if smoke:
        root = root / "smoke"
    round0_path = root / "runtime" / "round0.json"
    final_path = root / "runtime" / "final.json"
    round0 = deepcopy(dict(config))
    final = deepcopy(dict(config))

    if smoke:
        smoke_budget = _mapping(orchestration["smoke"], label="orchestration.smoke")
        for candidate in (round0, final):
            candidate["splits"]["train"]["episodes"] = int(
                smoke_budget["train_episodes"]
            )
            candidate["splits"]["development"]["episodes"] = int(
                smoke_budget["development_episodes"]
            )
            candidate["training"]["steps"] = int(smoke_budget["steps"])
            candidate["training"]["bootstrap_samples"] = int(
                smoke_budget["bootstrap_samples"]
            )
            candidate["rollout"]["episodes"] = int(
                smoke_budget["rollout_episodes"]
            )
            candidate["rollout"]["bootstrap_samples"] = int(
                smoke_budget["bootstrap_samples"]
            )
            candidate["orchestration"]["active_mode"] = "smoke"
    else:
        round0["orchestration"]["active_mode"] = "full"
        final["orchestration"]["active_mode"] = "full"

    round0["caches"] = {
        split: {
            method: [str(_collection_path(root, 0, split, method))]
            for method in METHODS
        }
        for split in ("train", "development")
    }
    final["caches"] = {
        split: {
            method: [
                str(_collection_path(root, iteration, split, method))
                for iteration in (0, 1)
            ]
            for method in METHODS
        }
        for split in ("train", "development")
    }
    round0["training"]["output_dir"] = str(_head_dir(root, "round0"))
    final["training"]["output_dir"] = str(_head_dir(root, "final"))
    round0["rollout"]["output_dir"] = str(root / "rollout" / "unused_round0")
    final["rollout"]["output_dir"] = str(root / "rollout" / "fresh")
    return round0_path, round0, final_path, final, root


def _collector_job(
    *,
    root: Path,
    config_path: Path,
    iteration: int,
    split: str,
    method: str,
    episodes: int,
    head_checkpoint: Path | None,
) -> StageJob:
    output = _collection_path(root, iteration, split, method)
    command = [
        sys.executable,
        str(COLLECTOR),
        "--config",
        str(config_path),
        "--method",
        method,
        "--dagger-iteration",
        str(iteration),
        "--split",
        split,
        "--episodes",
        str(episodes),
        "--device",
        "cuda:0",
        "--output",
        str(output),
    ]
    if head_checkpoint is not None:
        command.extend(("--head-checkpoint", str(head_checkpoint)))
    return StageJob(
        name=f"collect_r{iteration}_{split}_{method}",
        command=tuple(command),
        outputs=(output, output.with_suffix(".metadata.json")),
        log_path=root / "logs" / f"collect_r{iteration}_{split}_{method}.log",
    )


def _trainer_job(
    *,
    root: Path,
    name: str,
    config_path: Path,
    output_dir: Path,
    methods: Sequence[str],
    seeds: Sequence[int],
    steps: int,
) -> StageJob:
    command = (
        sys.executable,
        str(TRAINER),
        "--config",
        str(config_path),
        "--device",
        "cuda:0",
        "--steps",
        str(steps),
        "--seeds",
        *(str(seed) for seed in seeds),
        "--output-dir",
        str(output_dir),
    )
    outputs = tuple(
        _head_path(root, name, method, seed)
        for method in methods
        for seed in seeds
    ) + (output_dir / "report.json",)
    return StageJob(
        name=f"train_{name}",
        command=tuple(command),
        outputs=outputs,
        log_path=root / "logs" / f"train_{name}.log",
    )


def build_plan(config: Mapping[str, object], *, smoke: bool) -> ProtocolPlan:
    validate_config(config)
    round0_path, round0, final_path, final, root = _runtime_configs(
        config, smoke=smoke
    )
    methods = tuple(str(value) for value in final["methods"])
    seeds = tuple(int(value) for value in final["training"]["seeds"])
    if len(seeds) != 1:
        raise ValueError("round-one behavior requires one unambiguous head seed")
    head_seed = seeds[0]
    steps = int(final["training"]["steps"])
    jobs: list[StageJob] = []

    for split in ("train", "development"):
        episodes = int(round0["splits"][split]["episodes"])
        for method in methods:
            jobs.append(
                _collector_job(
                    root=root,
                    config_path=round0_path,
                    iteration=0,
                    split=split,
                    method=method,
                    episodes=episodes,
                    head_checkpoint=None,
                )
            )
    jobs.append(
        _trainer_job(
            root=root,
            name="round0",
            config_path=round0_path,
            output_dir=_head_dir(root, "round0"),
            methods=methods,
            seeds=seeds,
            steps=steps,
        )
    )

    for split in ("train", "development"):
        episodes = int(final["splits"][split]["episodes"])
        for method in methods:
            jobs.append(
                _collector_job(
                    root=root,
                    config_path=final_path,
                    iteration=1,
                    split=split,
                    method=method,
                    episodes=episodes,
                    head_checkpoint=_head_path(
                        root, "round0", method, head_seed
                    ),
                )
            )
    jobs.append(
        _trainer_job(
            root=root,
            name="final",
            config_path=final_path,
            output_dir=_head_dir(root, "final"),
            methods=methods,
            seeds=seeds,
            steps=steps,
        )
    )

    rollout = final["rollout"]
    rollout_dir = Path(str(rollout["output_dir"]))
    rollout_command = (
        sys.executable,
        str(EVALUATOR),
        "--config",
        str(final_path),
        "--episodes",
        str(int(rollout["episodes"])),
        "--head-seeds",
        *(str(seed) for seed in seeds),
        "--device",
        "cuda:0",
        "--output-dir",
        str(rollout_dir),
    )
    jobs.append(
        StageJob(
            name="fresh_paired_rollout",
            command=tuple(rollout_command),
            outputs=(rollout_dir / "report.json",),
            log_path=root / "logs" / "fresh_paired_rollout.log",
        )
    )
    return ProtocolPlan(
        mode="smoke" if smoke else "full",
        output_root=root,
        round0_config_path=round0_path,
        final_config_path=final_path,
        round0_config=round0,
        final_config=final,
        jobs=tuple(jobs),
    )


def _write_json_idempotent(payload: object, path: Path) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"runtime config differs from existing file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def materialize_runtime_configs(plan: ProtocolPlan) -> None:
    _write_json_idempotent(plan.round0_config, plan.round0_config_path)
    _write_json_idempotent(plan.final_config, plan.final_config_path)


def execution_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["MUJOCO_GL"] = "egl"
    return environment


def _presence(job: StageJob) -> str:
    present = tuple(path.exists() for path in job.outputs)
    if all(present):
        return "complete"
    if any(present):
        missing = [str(path) for path, exists in zip(job.outputs, present) if not exists]
        raise RuntimeError(f"partial output for {job.name}; missing={missing}")
    return "absent"


def run_plan(plan: ProtocolPlan, *, dry_run: bool) -> dict[str, int]:
    counts = {"planned": 0, "completed": 0, "reused": 0}
    if dry_run:
        print(f"[would-materialize] {plan.round0_config_path}")
        print(f"[would-materialize] {plan.final_config_path}")
    else:
        materialize_runtime_configs(plan)

    environment = execution_environment()
    for job in plan.jobs:
        presence = _presence(job)
        if presence == "complete":
            print(f"[reuse] {job.name}")
            counts["reused"] += 1
            continue
        if dry_run:
            print(
                "[would-run] "
                f"CUDA_VISIBLE_DEVICES=0 {shlex.join(job.command)}"
            )
            counts["planned"] += 1
            continue
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[run] {job.name}", flush=True)
        with job.log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                job.command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode:
            raise RuntimeError(
                f"{job.name} failed with exit code {completed.returncode}; "
                f"see {job.log_path}"
            )
        if _presence(job) != "complete":
            raise RuntimeError(f"{job.name} did not create its complete output set")
        counts["completed"] += 1
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    plan = build_plan(config, smoke=bool(args.smoke))
    counts = run_plan(plan, dry_run=bool(args.dry_run))
    print(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "mode": plan.mode,
                "physical_cuda_device": 0,
                "jobs": len(plan.jobs),
                **counts,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
