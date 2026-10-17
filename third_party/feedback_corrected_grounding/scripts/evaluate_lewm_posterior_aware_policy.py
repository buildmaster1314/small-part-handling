#!/usr/bin/env python3
"""Historical development screen for posterior-trained policies.

The tested controller interface is frozen as No-Go. Do not use this script to
produce the primary FCPG/LeWM control claim; see docs/posterior_aware_policy_pilot.md.
"""

from __future__ import annotations

import argparse
import json
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
from scripts.evaluate_p1_native_policy_rollout import rollout_once  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/lewm/eval/p1_native_joint_drift_drop_screen.json"
DEFAULT_SOURCE = ROOT / "config/lewm/eval/p1_native_complementary_fresh.json"
DEFAULT_POLICIES = {
    "base": ROOT / "outputs/action_chunk/p1/pick_fcpg_pureres_seed0_retry1/fcpg_action_chunk_best.pt",
    "measurement": ROOT / "outputs/lewm_posterior_aware_policy/measurement_seed63201_2k_fixed/measurement_policy_latest.pt",
    "no_action": ROOT / "outputs/lewm_posterior_aware_policy/no_action_seed63201_2k_fixed/no_action_policy_latest.pt",
    "factual": ROOT / "outputs/lewm_posterior_aware_policy/factual_seed63201_2k_fixed/factual_policy_latest.pt",
}
DEFAULT_OUTPUT = ROOT / "outputs/lewm_posterior_aware_policy/rollout_screen_seed63301_fixed"
METRICS = ("success", "success_fraction", "maximum_success_streak", "reward_sum")


def _resolve(path: str | Path) -> Path:
    target = Path(path).expanduser()
    return target.resolve() if target.is_absolute() else (ROOT / target).resolve()


def _validate_sparse_frontend_checkpoint(path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise TypeError(f"policy checkpoint has no model_state mapping: {path}")
    materialized = [
        name
        for name in state
        if name.startswith("vision_encoder.backbone.")
        or name.startswith("fcpg_adapter.")
    ]
    if materialized:
        raise ValueError(
            "posterior policy materialized a frozen external frontend: "
            f"{path} ({materialized[0]})"
        )


def _bootstrap_delta(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int,
    samples: int = 10000,
) -> dict[str, float]:
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("paired bootstrap inputs must be aligned vectors")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, left.size, size=(samples, left.size))
    values = (left[draws] - right[draws]).mean(axis=1)
    return {
        "delta": float((left - right).mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
    }


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    methods = sorted({str(row["method"]) for row in rows})
    episodes = sorted({int(row["physical_episode"]) for row in rows})
    lookup = {
        (str(row["method"]), int(row["physical_episode"])): row for row in rows
    }
    summary: dict[str, object] = {"methods": {}}
    method_summary: dict[str, object] = {}
    for method in methods:
        method_rows = [lookup[(method, episode)] for episode in episodes]
        method_summary[method] = {
            metric: float(np.mean([float(row[metric]) for row in method_rows]))
            for metric in METRICS
        }
    summary["methods"] = method_summary
    comparisons: dict[str, object] = {}
    pairs = (
        ("joint_factual", "joint_no_action"),
        ("joint_factual", "joint_measurement"),
        ("joint_factual", "joint_base_factual"),
        ("clean_factual", "clean_base"),
        ("clean_factual", "clean_no_action"),
        ("clean_factual", "clean_measurement"),
    )
    for pair_index, (left_name, right_name) in enumerate(pairs):
        if left_name not in methods or right_name not in methods:
            continue
        item: dict[str, object] = {}
        for metric_index, metric in enumerate(METRICS):
            left = np.asarray(
                [float(lookup[(left_name, episode)][metric]) for episode in episodes]
            )
            right = np.asarray(
                [float(lookup[(right_name, episode)][metric]) for episode in episodes]
            )
            item[metric] = _bootstrap_delta(
                left,
                right,
                seed=63390 + pair_index * 10 + metric_index,
            )
        comparisons[f"{left_name}_vs_{right_name}"] = item
    summary["comparisons"] = comparisons
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--p1-seed", type=int, default=15801)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not str(args.device).startswith("cuda:0") and str(args.device) != "cpu":
        parser.error("this experiment may use cuda:0 or cpu only")
    config = json.loads(_resolve(args.config).read_text(encoding="utf-8"))
    source = json.loads(_resolve(args.source_config).read_text(encoding="utf-8"))
    episodes = int(args.episodes)
    if not 0 < episodes <= int(config["physical_episode_count"]):
        parser.error("episodes exceed the frozen screen panel")
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    progress_path = output_dir / "progress.jsonl"
    device = torch.device(args.device)

    training_path = _resolve(source["frozen_predictor_training_contract"]["config"])
    training_config = json.loads(training_path.read_text(encoding="utf-8"))
    records = {int(row["seed"]): row for row in source["frozen_predictors"]}
    record = records[int(args.p1_seed)]
    pair = load_frozen_p1_native_checkpoint(
        FrozenP1NativeCheckpointSpec(
            seed=int(args.p1_seed),
            path=_resolve(record["checkpoint"]),
            sha256=str(record["sha256"]),
        ),
        expected_config=training_config,
        device=device,
    )
    policies: dict[str, tuple[torch.nn.Module, dict[str, object]]] = {}
    for name, path in DEFAULT_POLICIES.items():
        resolved_path = _resolve(path)
        _validate_sparse_frontend_checkpoint(resolved_path)
        policy, policy_args = load_action_chunk_policy(resolved_path, device)
        policy.set_reliability_override("ones")
        policy.requires_grad_(False).eval()
        policies[name] = (policy, policy_args)

    conditions = (
        ("clean_base", "base", "clean_native", 0.0),
        ("clean_measurement", "measurement", "clean_native", 0.0),
        ("clean_no_action", "no_action", "clean_native", 0.0),
        ("clean_factual", "factual", "clean_native", 0.0),
        ("joint_base_factual", "base", f"complementary_factual_s{args.p1_seed}", float(config["camera_rotation_deg"])),
        ("joint_measurement", "measurement", "measurement", float(config["camera_rotation_deg"])),
        ("joint_no_action", "no_action", f"complementary_no_action_s{args.p1_seed}", float(config["camera_rotation_deg"])),
        ("joint_factual", "factual", f"complementary_factual_s{args.p1_seed}", float(config["camera_rotation_deg"])),
    )
    rows: list[dict[str, object]] = []
    with progress_path.open("w", encoding="utf-8") as progress:
        for episode in range(episodes):
            for result_name, policy_name, runtime_method, rotation in conditions:
                policy, policy_args = policies[policy_name]
                result = rollout_once(
                    policy,
                    policy_args,
                    pair=pair,
                    method=runtime_method,
                    model_seed=int(args.p1_seed),
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
                    camera_rotation_deg=rotation,
                )
                result["method"] = result_name
                result["physical_episode"] = episode
                for key in list(result):
                    if key.startswith("_"):
                        result.pop(key)
                rows.append(result)
                progress.write(json.dumps(result) + "\n")
                progress.flush()
                print(
                    json.dumps(
                        {
                            "episode": episode,
                            "method": result_name,
                            "success": result["success"],
                            "reward": result["reward_sum"],
                        }
                    ),
                    flush=True,
                )
    report = {
        "classification": "posterior_aware_policy_development_screen",
        "episodes": episodes,
        "p1_seed": int(args.p1_seed),
        "conditions": [row[0] for row in conditions],
        "summary": _summarize(rows),
        "rows": rows,
        "claim_boundary": "single-task single-seed development screen",
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
