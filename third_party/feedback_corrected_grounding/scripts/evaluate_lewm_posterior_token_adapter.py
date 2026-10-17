#!/usr/bin/env python3
"""Historical development screen for posterior-token adapters.

The tested adapter interface is frozen as No-Go. Do not use this script to
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
from fcpg.lewm_posterior_aware_policy import (  # noqa: E402
    PosteriorTokenAdapterPolicy,
    PosteriorTokenResidualAdapter,
)
from scripts.evaluate_p1_native_policy_rollout import rollout_once  # noqa: E402


CONFIG = ROOT / "config/lewm/eval/p1_native_joint_drift_drop_screen.json"
SOURCE = ROOT / "config/lewm/eval/p1_native_complementary_fresh.json"
BASE = ROOT / "outputs/action_chunk/p1/pick_fcpg_pureres_seed0_retry1/fcpg_action_chunk_best.pt"
ADAPTERS = {
    "factual": ROOT / "outputs/lewm_posterior_token_adapter/factual_seed63401_2k/adapter_latest.pt",
    "no_action": ROOT / "outputs/lewm_posterior_token_adapter/no_action_seed63401_2k/adapter_latest.pt",
}
OUTPUT = ROOT / "outputs/lewm_posterior_token_adapter/rollout_screen_seed63501"
METRICS = (
    "success",
    "success_fraction",
    "maximum_success_streak",
    "reward_sum",
)


def _resolve(path: str | Path) -> Path:
    target = Path(path).expanduser()
    return target.resolve() if target.is_absolute() else (ROOT / target).resolve()


def _load_adapter(path: Path, device: torch.device) -> PosteriorTokenResidualAdapter:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_type") != "lewm_posterior_token_residual_adapter":
        raise ValueError(f"wrong adapter checkpoint type: {path}")
    config = dict(checkpoint["adapter_config"])
    adapter = PosteriorTokenResidualAdapter(**config).to(device)
    adapter.load_state_dict(checkpoint["adapter_state"], strict=True)
    return adapter.requires_grad_(False).eval()


def _bootstrap(left: np.ndarray, right: np.ndarray, *, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, left.size, size=(10000, left.size))
    values = (left[draws] - right[draws]).mean(axis=1)
    return {
        "delta": float((left - right).mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
    }


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    methods = sorted({str(row["method"]) for row in rows})
    episodes = sorted({int(row["physical_episode"]) for row in rows})
    lookup = {(str(row["method"]), int(row["physical_episode"])): row for row in rows}
    means = {
        method: {
            metric: float(np.mean([float(lookup[(method, episode)][metric]) for episode in episodes]))
            for metric in METRICS
        }
        for method in methods
    }
    pairs = (
        ("joint_factual_adapter", "joint_no_action_adapter"),
        ("joint_factual_adapter", "joint_unadapted_factual"),
        ("joint_factual_adapter", "joint_measurement"),
        ("clean_factual_adapter", "clean_base"),
    )
    comparisons: dict[str, object] = {}
    for pair_index, (left_name, right_name) in enumerate(pairs):
        comparisons[f"{left_name}_vs_{right_name}"] = {
            metric: _bootstrap(
                np.asarray([float(lookup[(left_name, episode)][metric]) for episode in episodes]),
                np.asarray([float(lookup[(right_name, episode)][metric]) for episode in episodes]),
                seed=63590 + pair_index * 10 + metric_index,
            )
            for metric_index, metric in enumerate(METRICS)
        }
    return {"methods": means, "comparisons": comparisons}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--p1-seed", type=int, default=15801)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not str(args.device).startswith("cuda:0") and str(args.device) != "cpu":
        parser.error("this experiment may use cuda:0 or cpu only")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if not 0 < int(args.episodes) <= int(config["physical_episode_count"]):
        parser.error("episodes exceed the frozen development panel")
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)

    expected_path = _resolve(source["frozen_predictor_training_contract"]["config"])
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    record = {int(row["seed"]): row for row in source["frozen_predictors"]}[int(args.p1_seed)]
    pair = load_frozen_p1_native_checkpoint(
        FrozenP1NativeCheckpointSpec(
            seed=int(args.p1_seed),
            path=_resolve(record["checkpoint"]),
            sha256=str(record["sha256"]),
        ),
        expected_config=expected,
        device=device,
    )
    base, policy_args = load_action_chunk_policy(BASE, device)
    base.set_reliability_override("ones")
    base.requires_grad_(False).eval()
    factual_policy = PosteriorTokenAdapterPolicy(base, _load_adapter(ADAPTERS["factual"], device))
    no_action_policy = PosteriorTokenAdapterPolicy(base, _load_adapter(ADAPTERS["no_action"], device))
    factual_policy.requires_grad_(False).eval()
    no_action_policy.requires_grad_(False).eval()

    conditions = (
        ("clean_base", base, "clean_native", 0.0),
        ("clean_factual_adapter", factual_policy, "clean_native", 0.0),
        ("joint_measurement", base, "measurement", float(config["camera_rotation_deg"])),
        ("joint_unadapted_factual", base, f"complementary_factual_s{args.p1_seed}", float(config["camera_rotation_deg"])),
        ("joint_no_action_adapter", no_action_policy, f"complementary_no_action_s{args.p1_seed}", float(config["camera_rotation_deg"])),
        ("joint_factual_adapter", factual_policy, f"complementary_factual_s{args.p1_seed}", float(config["camera_rotation_deg"])),
    )
    rows: list[dict[str, object]] = []
    progress_path = output_dir / "progress.jsonl"
    with progress_path.open("w", encoding="utf-8") as progress:
        for episode in range(int(args.episodes)):
            for name, policy, method, rotation in conditions:
                result = rollout_once(
                    policy,
                    policy_args,
                    pair=pair,
                    method=method,
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
                result["method"] = name
                result["physical_episode"] = episode
                for key in list(result):
                    if key.startswith("_"):
                        result.pop(key)
                rows.append(result)
                progress.write(json.dumps(result) + "\n")
                progress.flush()
                print(json.dumps({"episode": episode, "method": name, "success": result["success"], "reward": result["reward_sum"]}), flush=True)
    report = {
        "classification": "frozen_policy_posterior_token_adapter_development_screen",
        "episodes": int(args.episodes),
        "p1_seed": int(args.p1_seed),
        "summary": _summarize(rows),
        "rows": rows,
        "claim_boundary": "single-task single-policy single-LeWM-seed development screen",
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
