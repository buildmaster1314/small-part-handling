#!/usr/bin/env python3
"""Zero-training rollout for a deterministic FCPG/LeWM validity switch."""

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

from fcpg.lewm_rule_based_validity import RuleBasedVisualValidityFSM  # noqa: E402
from scripts.evaluate_lewm_posterior_dagger_rollout import (  # noqa: E402
    _drop_starts,
    _json,
    _load_frozen_components,
    _mapping,
)
from scripts.evaluate_p1_native_policy_rollout import (  # noqa: E402
    P1_POLICY_ACTION_STEPS,
    rollout_once,
)


PROTOCOL = "fcpg_lewm_rule_observer_rollout_v1"
METHODS = (
    "clean_reference",
    "measurement_only",
    "lewm_known_validity",
    "lewm_rule",
    "geometry_rule",
    "hold_rule",
)
REPORT_METRICS = (
    "sustained_success",
    "success",
    "final_success",
    "success_fraction",
    "reward_sum",
    "post_drop_success_fraction",
)


def method_specifications(model_seed: int) -> tuple[tuple[str, str, bool], ...]:
    trusted = f"complementary_factual_trusted_global_s{int(model_seed)}"
    return (
        ("clean_reference", "clean_native", False),
        ("measurement_only", "measurement", False),
        ("lewm_known_validity", trusted, False),
        ("lewm_rule", trusted, True),
        ("geometry_rule", "geometry_hold", True),
        ("hold_rule", "hold_last", True),
    )


def _percentile(values: np.ndarray) -> list[float]:
    low, high = np.quantile(np.asarray(values, dtype=np.float64), (0.025, 0.975))
    return [float(low), float(high)]


def summarize_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    episodes = tuple(sorted({int(row["physical_episode"]) for row in rows}))
    indexed: dict[tuple[int, str], Mapping[str, object]] = {}
    for row in rows:
        key = (int(row["physical_episode"]), str(row["method"]))
        if key in indexed:
            raise ValueError(f"duplicate rollout row {key}")
        indexed[key] = row
    expected = {(episode, method) for episode in episodes for method in METHODS}
    if set(indexed) != expected:
        raise ValueError("rule-observer rollout panel is incomplete")

    method_summary = {
        method: {
            metric: float(
                np.mean([float(indexed[(episode, method)][metric]) for episode in episodes])
            )
            for metric in REPORT_METRICS
        }
        for method in METHODS
    }
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(episodes), size=(bootstrap_samples, len(episodes)))
    comparisons: dict[str, object] = {}
    for candidate, baseline in (
        ("lewm_rule", "measurement_only"),
        ("lewm_rule", "geometry_rule"),
        ("lewm_rule", "hold_rule"),
        ("lewm_known_validity", "lewm_rule"),
    ):
        metric_rows: dict[str, object] = {}
        for metric in REPORT_METRICS:
            delta = np.asarray(
                [
                    float(indexed[(episode, candidate)][metric])
                    - float(indexed[(episode, baseline)][metric])
                    for episode in episodes
                ],
                dtype=np.float64,
            )
            metric_rows[metric] = {
                "delta": float(np.mean(delta)),
                "ci": _percentile(np.mean(delta[draws], axis=1)),
            }
        comparisons[f"{candidate}_vs_{baseline}"] = metric_rows

    selector_summary: dict[str, object] = {}
    for method in ("lewm_rule", "geometry_rule", "hold_rule"):
        selected = [indexed[(episode, method)] for episode in episodes]
        tp = sum(int(row["selector_true_positive"]) for row in selected)
        fp = sum(int(row["selector_false_positive"]) for row in selected)
        fn = sum(int(row["selector_false_negative"]) for row in selected)
        tn = sum(int(row["selector_true_negative"]) for row in selected)
        selector_summary[method] = {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        }
    return {
        "physical_episodes": len(episodes),
        "method_summary": method_summary,
        "comparisons": comparisons,
        "selector_summary": selector_summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/lewm/eval/posterior_dagger_pick_dev.json"
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir", default="outputs/lewm_rule_observer_pick/development_v1"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.device not in {"cpu", "cuda:0"}:
        parser.error("rule-observer rollout may use only cpu or cuda:0")
    if args.episodes <= 0 or args.episode_start < 0:
        parser.error("--episodes must be positive and --episode-start non-negative")

    os.environ.setdefault("MUJOCO_GL", "egl")
    config = _json(args.config)
    rollout = _mapping(config["rollout"], label="rollout")
    configured_episodes = int(rollout["episodes"])
    if args.episode_start + args.episodes > configured_episodes:
        parser.error(
            "the requested episode range cannot exceed "
            f"{configured_episodes} configured episodes"
        )
    output_dir = (ROOT / args.output_dir).resolve()
    report_path = output_dir / "report.json"
    progress_path = output_dir / "progress.jsonl"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {report_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path.write_text("", encoding="utf-8")

    device = torch.device(args.device)
    policy, policy_args, pair = _load_frozen_components(config, device)
    model_seed = int(config["model_seed"])
    all_drop_starts = _drop_starts(
        config, episodes=configured_episodes, condition_index=0
    )
    episode_indices = range(args.episode_start, args.episode_start + args.episodes)
    rows: list[dict[str, object]] = []
    for episode in episode_indices:
        shared_actions: np.ndarray | None = None
        shared_tokens: np.ndarray | None = None
        shared_visual: np.ndarray | None = None
        for label, runtime_method, use_rule in method_specifications(model_seed):
            selector = RuleBasedVisualValidityFSM() if use_rule else None
            result = rollout_once(
                policy,
                policy_args,
                pair=pair,
                method=runtime_method,
                model_seed=model_seed,
                task=str(config["task"]),
                camera=str(config["camera"]),
                width=int(config["width"]),
                height=int(config["height"]),
                benchmark_seed=int(rollout["benchmark_seed"]),
                task_index=int(rollout["task_index_base"]) + episode,
                environment_seed=int(rollout["environment_seed_base"]) + episode,
                drop_start=int(all_drop_starts[episode]),
                drop_duration=int(config["drop_duration_decisions"]),
                horizon=int(rollout["horizon_raw_steps"]),
                radius_px=int(config["occlusion_radius_px"]),
                value_uint8=int(config["occlusion_value_uint8"]),
                device=device,
                camera_rotation_deg=float(config["camera_rotation_deg"]),
                camera_rotation_axis_world=tuple(config["camera_rotation_axis_world"]),
                forced_prefix_actions=shared_actions,
                forced_prefix_tokens=shared_tokens,
                forced_prefix_visual=shared_visual,
                result_method=label,
                occlusion_center_key=str(config["occlusion_center_key"]),
                compute_clean_action_metric=False,
                visual_validity_fsm=selector,
            )
            prefix_tokens = np.asarray(result.pop("_prefix_native_tokens"))
            prefix_visual = np.asarray(result.pop("_prefix_sampled_visual"))
            result = {
                key: value
                for key, value in result.items()
                if key != "reset_digest" and not key.endswith("sha256")
            }
            row = {"physical_episode": episode, **result}
            if shared_actions is None:
                shared_actions = np.asarray(
                    result["pre_drop_action_values"], dtype=np.float32
                ).reshape(int(all_drop_starts[episode]), P1_POLICY_ACTION_STEPS, 4)
                shared_tokens = prefix_tokens.astype(np.float32, copy=False)
                shared_visual = prefix_visual.astype(np.float32, copy=False)
            rows.append(row)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            print(
                f"episode={episode:02d} method={label:20s} "
                f"success={row['sustained_success']:.0f} reward={row['reward_sum']:.2f} "
                f"prior={row['selector_prior_decisions']}",
                flush=True,
            )

    summary = summarize_rows(
        rows,
        bootstrap_samples=int(rollout["bootstrap_samples"]),
        seed=int(rollout["bootstrap_seed"]) + 71,
    )
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "classification": "development_only",
        "learned_selector": False,
        "oracle_inputs_to_selector": False,
        "task": str(config["task"]),
        "camera_rotation_deg": float(config["camera_rotation_deg"]),
        "occlusion_radius_px": int(config["occlusion_radius_px"]),
        "rows": rows,
        "summary": summary,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
