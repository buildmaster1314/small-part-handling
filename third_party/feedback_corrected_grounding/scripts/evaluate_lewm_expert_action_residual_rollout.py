#!/usr/bin/env python3
"""Fresh paired Pick rollout for the causal expert-action residual screen."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_expert_action_residual import (  # noqa: E402
    METHODS,
    PROTOCOL,
    load_expert_action_residual_head,
)
from scripts.collect_lewm_expert_action_residual_pick import (  # noqa: E402
    _json,
    _load_frozen_components,
    _resolve,
)
from scripts.evaluate_action_chunk_rollout import choose_device  # noqa: E402
from scripts.evaluate_p1_native_policy_rollout import (  # noqa: E402
    P1_POLICY_ACTION_STEPS,
    _validate_pairing,
    rollout_once,
)


def _rollout_config(config: Mapping[str, object]) -> Mapping[str, object]:
    value = config.get("rollout")
    if not isinstance(value, Mapping):
        raise ValueError("expert residual config lacks rollout settings")
    return value


def _validate_rollout_contract(
    config: Mapping[str, object], rollout: Mapping[str, object]
) -> None:
    if int(config.get("action_steps_per_decision", -1)) != P1_POLICY_ACTION_STEPS:
        raise ValueError("expert residual rollout requires five-action decisions")
    if int(config.get("drop_duration_decisions", -1)) != 6:
        raise ValueError("expert residual rollout is fixed to six missing decisions")
    start_min = int(config.get("drop_start_min_decision", -1))
    start_max = int(config.get("drop_start_max_decision", -1))
    horizon = int(rollout.get("horizon_raw_steps", -1))
    if start_min < 3 or start_max < start_min:
        raise ValueError("drop schedule requires at least three warmup decisions")
    if horizon <= 0 or horizon % P1_POLICY_ACTION_STEPS:
        raise ValueError("rollout horizon must contain complete five-action decisions")
    if start_max + 6 >= horizon // P1_POLICY_ACTION_STEPS:
        raise ValueError("rollout must include at least one recovery decision")
    if int(rollout.get("episodes", 0)) <= 0:
        raise ValueError("rollout requires physical episodes")


def _percentile(samples: np.ndarray) -> list[float]:
    low, high = np.quantile(np.asarray(samples, dtype=np.float64), [0.025, 0.975])
    return [float(low), float(high)]


def seed_macro_comparison(
    rows: Sequence[Mapping[str, object]],
    *,
    candidate_methods: Sequence[str],
    baseline_methods: Sequence[str],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    """Average head seeds within episode, then bootstrap physical episodes."""

    candidates = tuple(str(value) for value in candidate_methods)
    baselines = tuple(str(value) for value in baseline_methods)
    if not candidates or len(candidates) != len(baselines):
        raise ValueError("seed-macro comparison requires aligned method lists")
    indexed = {
        (int(row["physical_episode"]), str(row["method"])): row for row in rows
    }
    episodes = sorted(
        episode for episode, method in indexed if method == candidates[0]
    )
    if not episodes:
        raise ValueError("seed-macro comparison has no physical episodes")
    for method in (*candidates, *baselines):
        if any((episode, method) not in indexed for episode in episodes):
            raise ValueError(f"incomplete rollout panel for {method}")

    metrics = (
        "success",
        "sustained_success",
        "final_success",
        "success_fraction",
        "maximum_success_streak",
        "post_drop_success_fraction",
        "post_drop_maximum_success_streak",
        "reward_sum",
    )
    result: dict[str, object] = {
        "candidate_methods": list(candidates),
        "baseline_methods": list(baselines),
        "head_seed_count": len(candidates),
        "physical_episode_count": len(episodes),
        "independent_unit": "physical_episode",
    }
    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0, len(episodes), size=(int(bootstrap_samples), len(episodes))
    )
    for metric in metrics:
        candidate = np.asarray(
            [
                [float(indexed[(episode, method)][metric]) for method in candidates]
                for episode in episodes
            ],
            dtype=np.float64,
        )
        baseline = np.asarray(
            [
                [float(indexed[(episode, method)][metric]) for method in baselines]
                for episode in episodes
            ],
            dtype=np.float64,
        )
        delta_by_episode = np.mean(candidate - baseline, axis=1)
        samples = np.mean(delta_by_episode[draws], axis=1)
        result[f"{metric}_delta"] = float(np.mean(delta_by_episode))
        result[f"{metric}_delta_ci"] = _percentile(samples)
        result[f"{metric}_delta_one_sided_95_lower"] = float(
            np.quantile(samples, 0.05)
        )

    per_seed = []
    for candidate_method, baseline_method in zip(candidates, baselines, strict=True):
        candidate_rows = [indexed[(episode, candidate_method)] for episode in episodes]
        baseline_rows = [indexed[(episode, baseline_method)] for episode in episodes]
        per_seed.append(
            {
                "candidate": candidate_method,
                "baseline": baseline_method,
                "sustained_success_delta": float(
                    np.mean(
                        [
                            float(candidate["sustained_success"])
                            - float(baseline["sustained_success"])
                            for candidate, baseline in zip(
                                candidate_rows, baseline_rows, strict=True
                            )
                        ]
                    )
                ),
                "reward_delta": float(
                    np.mean(
                        [
                            float(candidate["reward_sum"])
                            - float(baseline["reward_sum"])
                            for candidate, baseline in zip(
                                candidate_rows, baseline_rows, strict=True
                            )
                        ]
                    )
                ),
            }
        )
    result["per_seed"] = per_seed
    return result


def _method_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    methods = sorted({str(row["method"]) for row in rows})
    summary: dict[str, object] = {}
    for method in methods:
        selected = [row for row in rows if str(row["method"]) == method]
        summary[method] = {
            "success_rate": float(np.mean([row["success"] for row in selected])),
            "sustained_success_rate": float(
                np.mean([row["sustained_success"] for row in selected])
            ),
            "final_success_rate": float(
                np.mean([row["final_success"] for row in selected])
            ),
            "post_drop_success_fraction": float(
                np.mean([row["post_drop_success_fraction"] for row in selected])
            ),
            "reward_mean": float(np.mean([row["reward_sum"] for row in selected])),
            "action_distance_mean": float(
                np.mean([row["action_distance"] for row in selected])
            ),
        }
    return summary


def _gate(
    comparisons: Mapping[str, Mapping[str, object]],
    *,
    full_panel: bool,
) -> dict[str, object]:
    factual = comparisons["factual_residual_vs_factual_base"]
    no_action = comparisons["factual_residual_vs_no_action_residual"]
    geometry = comparisons["factual_residual_vs_geometry_residual"]
    direction_count = sum(
        all(
            float(comparison["per_seed"][index]["sustained_success_delta"]) > 0.0
            for comparison in (factual, no_action, geometry)
        )
        for index in range(len(factual["per_seed"]))
    )
    criteria = {
        "full_30_episode_three_seed_panel": bool(full_panel),
        "all_three_sustained_success_deltas_at_least_10pp": all(
            float(value["sustained_success_delta"]) >= 0.10
            for value in (factual, no_action, geometry)
        ),
        "all_three_sustained_success_one_sided_95_lowers_above_zero": all(
            float(value["sustained_success_delta_one_sided_95_lower"]) > 0.0
            for value in (factual, no_action, geometry)
        ),
        "at_least_two_head_seeds_positive_against_all_three_baselines": (
            direction_count >= 2
        ),
    }
    return {
        "passed": bool(all(criteria.values())),
        "criteria": criteria,
        "direction_count": int(direction_count),
        "scope": "Pick-only fresh development control screen",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/lewm/eval/expert_action_residual_pick_dev.json",
    )
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--head-seeds", type=int, nargs="+")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    os.environ.setdefault("MUJOCO_GL", "egl")
    config = _json(args.config)
    if config.get("schema_version") != 1 or config.get("protocol") != PROTOCOL:
        raise ValueError("expert residual rollout config is incompatible")
    rollout = _rollout_config(config)
    _validate_rollout_contract(config, rollout)
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("expert residual rollout lacks training settings")
    all_seeds = tuple(int(value) for value in training["seeds"])
    seeds = all_seeds if args.head_seeds is None else tuple(args.head_seeds)
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds).issubset(all_seeds):
        parser.error("--head-seeds must be unique members of the trained seed panel")
    total_episodes = int(rollout["episodes"])
    episodes = total_episodes if args.episodes is None else int(args.episodes)
    if not 0 < episodes <= total_episodes:
        parser.error(f"--episodes must lie in [1,{total_episodes}]")

    output_dir = _resolve(args.output_dir or str(rollout["output_dir"]))
    report_path = output_dir / "report.json"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {report_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    policy, policy_args, pair = _load_frozen_components(config, device)

    checkpoint_dir = _resolve(str(training["output_dir"]))
    training_report = _json(checkpoint_dir / "report.json")
    training_gate = training_report.get("gate")
    if (
        training_report.get("protocol") != PROTOCOL
        or training_report.get("primary_metric")
        != "scripted_expert_xyz_action_mse_over_next_5_raw_steps"
        or not isinstance(training_gate, Mapping)
        or training_gate.get("rollout_authorized") is not True
        or int(training_report.get("steps", -1)) != int(training["steps"])
        or tuple(int(value) for value in training_report.get("seeds", []))
        != all_seeds
    ):
        raise PermissionError("offline XYZ Gate does not authorize this rollout")
    heads: dict[tuple[str, int], torch.nn.Module] = {}
    head_metadata = []
    for seed in seeds:
        for method in METHODS:
            path = checkpoint_dir / f"{method}_seed{seed}_latest.pt"
            head, metadata = load_expert_action_residual_head(
                path,
                device=device,
                expected_method=method,
                expected_seed=seed,
            )
            if int(metadata["steps"]) != int(training["steps"]):
                raise ValueError(f"residual checkpoint is not at the fixed endpoint: {path}")
            heads[(method, seed)] = head
            head_metadata.append({"path": str(path), **metadata})

    rng = np.random.default_rng(int(rollout["schedule_seed"]))
    schedule_values = np.arange(
        int(config["drop_start_min_decision"]),
        int(config["drop_start_max_decision"]) + 1,
        dtype=np.int64,
    )
    drop_starts = np.resize(schedule_values, episodes)
    rng.shuffle(drop_starts)
    specifications: list[tuple[str, str, int | None, torch.nn.Module | None]] = [
        ("factual_base", "complementary_factual", None, None)
    ]
    for seed in seeds:
        specifications.extend(
            (
                (
                    f"factual_residual_s{seed}",
                    "complementary_factual",
                    seed,
                    heads[("complementary_factual", seed)],
                ),
                (
                    f"no_action_residual_s{seed}",
                    "complementary_no_action",
                    seed,
                    heads[("complementary_no_action", seed)],
                ),
                (
                    f"geometry_residual_s{seed}",
                    "geometry_hold",
                    seed,
                    heads[("geometry_hold", seed)],
                ),
            )
        )

    rows: list[dict[str, object]] = []
    progress_path = output_dir / "progress.jsonl"
    progress_path.write_text("", encoding="utf-8")
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
            )
            prefix_tokens = np.asarray(result.pop("_prefix_native_tokens"))
            prefix_visual = np.asarray(result.pop("_prefix_sampled_visual"))
            row = {
                "physical_episode": episode,
                "head_seed": head_seed,
                "observer_method": observer_method,
                **result,
            }
            if shared_actions is None:
                if label != "factual_base":
                    raise RuntimeError("factual base must define the shared prefix")
                shared_actions = np.asarray(result["pre_drop_action_values"], dtype=np.float32).reshape(
                    int(drop_starts[episode]), P1_POLICY_ACTION_STEPS, 4
                )
                shared_tokens = prefix_tokens.astype(np.float32, copy=False)
                shared_visual = prefix_visual.astype(np.float32, copy=False)
            rows.append(row)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            print(
                f"episode={episode:02d} method={label:31s} "
                f"sustained={row['sustained_success']:.0f} "
                f"reward={row['reward_sum']:.2f}",
                flush=True,
            )

    _validate_pairing(rows)
    factual = [f"factual_residual_s{seed}" for seed in seeds]
    comparisons = {
        "factual_residual_vs_factual_base": seed_macro_comparison(
            rows,
            candidate_methods=factual,
            baseline_methods=["factual_base"] * len(seeds),
            bootstrap_samples=int(rollout["bootstrap_samples"]),
            seed=int(rollout["bootstrap_seed"]),
        ),
        "factual_residual_vs_no_action_residual": seed_macro_comparison(
            rows,
            candidate_methods=factual,
            baseline_methods=[f"no_action_residual_s{seed}" for seed in seeds],
            bootstrap_samples=int(rollout["bootstrap_samples"]),
            seed=int(rollout["bootstrap_seed"]) + 1,
        ),
        "factual_residual_vs_geometry_residual": seed_macro_comparison(
            rows,
            candidate_methods=factual,
            baseline_methods=[f"geometry_residual_s{seed}" for seed in seeds],
            bootstrap_samples=int(rollout["bootstrap_samples"]),
            seed=int(rollout["bootstrap_seed"]) + 2,
        ),
    }
    full_panel = episodes == total_episodes and seeds == all_seeds
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "classification": str(rollout["classification"]),
        "known_sensor_validity_only": True,
        "episodes": episodes,
        "head_seeds": list(seeds),
        "methods": [value[0] for value in specifications],
        "head_checkpoints": head_metadata,
        "method_summary": _method_summary(rows),
        "comparisons": comparisons,
        "gate": _gate(comparisons, full_panel=full_panel),
        "claim_boundary": (
            "Pick-only development evidence; the scripted expert supervises only "
            "the output residual and is never available during rollout inference"
        ),
        "rows": rows,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"gate": report["gate"], "comparisons": comparisons}, indent=2))
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
