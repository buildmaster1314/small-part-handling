#!/usr/bin/env python3
"""Replay critical matched-factorial episodes with physical task traces."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from fcpg.action_chunk_policy import load_action_chunk_policy
from fcpg.lewm_factorial_physical_trace import (
    PhysicalTraceEnv,
    summarize_physical_trace,
)
from fcpg.lewm_p1_native_complementary import load_frozen_p1_native_checkpoint
from scripts import evaluate_p1_native_policy_rollout as rollout_module
from scripts.evaluate_action_chunk_rollout import choose_device
from scripts.evaluate_fcpg_lewm_factorial import (
    _load_json,
    _predictor_specs_by_grounding,
    _reference_prefix_actions,
    _resolve,
)


DEFAULT_CONFIG = "config/lewm/eval/fcpg_lewm_factorial_pick_matched_all3_dev30.json"
DEFAULT_EPISODES = (5, 7, 9, 10, 14, 24, 27)
DEFAULT_REPORTS = (
    "outputs/fcpg_lewm_factorial/pick_matched_dev30_seed990000/factorial_report.json",
    "outputs/fcpg_lewm_factorial/pick_matched_seed15802_sharedprefix_dev30_seed990000/factorial_report.json",
    "outputs/fcpg_lewm_factorial/pick_matched_seed15803_sharedprefix_dev30_seed990000/factorial_report.json",
)
DEFAULT_OUTPUT = (
    "outputs/fcpg_lewm_factorial/pick_matched_critical_physical_trace/report.json"
)
CELLS = {
    "f1_l0": "geometry_hold",
    "f1_l1": "complementary_factual",
}


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("expected unique comma-separated integers")
    return result


def _expected_rows(reports: list[dict[str, object]]) -> dict[tuple[int, int, str], dict[str, object]]:
    indexed: dict[tuple[int, int, str], dict[str, object]] = {}
    for report in reports:
        seed = int(report["model_seed"])
        rows = report.get("rows")
        if not isinstance(rows, list):
            raise TypeError("source factorial report is missing rows")
        for raw in rows:
            if not isinstance(raw, dict):
                raise TypeError("factorial row must be a mapping")
            key = (seed, int(raw["physical_episode"]), str(raw["method"]))
            if key in indexed:
                raise ValueError(f"duplicate source factorial row {key}")
            indexed[key] = raw
    return indexed


def _validate_replay(
    result: dict[str, object], expected: dict[str, object]
) -> None:
    for field in (
        "reset_digest",
        "camera_sha256",
        "schedule_sha256",
        "drop_onset_state_sha256",
        "pre_drop_action_round6_sha256",
        "all_action_sha256",
    ):
        if str(result[field]) != str(expected[field]):
            raise RuntimeError(f"physical replay drifted in {field}")
    for field in (
        "success",
        "final_success",
        "sustained_success",
        "post_drop_maximum_success_streak",
    ):
        if float(result[field]) != float(expected[field]):
            raise RuntimeError(f"physical replay drifted in {field}")
    if not np.isclose(
        float(result["reward_sum"]),
        float(expected["reward_sum"]),
        rtol=0.0,
        atol=1.0e-5,
    ):
        raise RuntimeError("physical replay drifted in reward_sum")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--report", action="append")
    parser.add_argument("--episodes", type=_parse_ints, default=DEFAULT_EPISODES)
    parser.add_argument("--model-seeds", type=_parse_ints)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if str(args.device).startswith("cuda") and str(args.device) != "cuda:0":
        parser.error("this diagnostic is restricted to CUDA0")

    os.environ.setdefault("MUJOCO_GL", "egl")
    config = _load_json(_resolve(args.config))
    reports = [_load_json(_resolve(path)) for path in (args.report or DEFAULT_REPORTS)]
    expected = _expected_rows(reports)
    report_by_seed = {int(report["model_seed"]): report for report in reports}
    configured_seeds = tuple(int(value) for value in config["model_seeds"])
    model_seeds = configured_seeds if args.model_seeds is None else args.model_seeds
    if not set(model_seeds).issubset(configured_seeds):
        parser.error(f"model seeds must be selected from {configured_seeds}")
    episodes = tuple(int(value) for value in args.episodes)
    if not set(episodes).issubset(range(int(config["physical_episode_count"]))):
        parser.error("episode selection lies outside the configured physical panel")

    source = _load_json(config["source_fresh_gate_config"])
    p1 = source["p1"]
    device = choose_device(args.device)
    policy, policy_args = load_action_chunk_policy(_resolve(p1["policy_checkpoint"]), device)
    policy.set_reliability_override("ones")
    policy.requires_grad_(False).eval()
    reference_report = report_by_seed[configured_seeds[0]]
    reference_actions = _reference_prefix_actions(
        reference_report,
        episodes=int(config["physical_episode_count"]),
        drop_starts=config["drop_start_decisions"],
    )

    traces: list[dict[str, object]] = []
    original_make_env = rollout_module.make_env
    try:
        for seed in model_seeds:
            pair_specs = _predictor_specs_by_grounding(config, source, seed=seed)
            expected_config, checkpoint_spec = pair_specs["fcpg"]
            pair = load_frozen_p1_native_checkpoint(
                checkpoint_spec,
                expected_config=expected_config,
                device=device,
            )
            for episode in episodes:
                for cell, method in CELLS.items():
                    step_rows: list[dict[str, object]] = []

                    def traced_make_env(*make_args: Any, **make_kwargs: Any):
                        base = original_make_env(*make_args, **make_kwargs)
                        return PhysicalTraceEnv(
                            base,
                            step_rows,
                            drop_start=int(config["drop_start_decisions"][episode]),
                            drop_duration=int(config["drop_duration_decisions"]),
                        )

                    rollout_module.make_env = traced_make_env
                    result = rollout_module.rollout_once(
                        policy,
                        policy_args,
                        pair=pair,
                        method=method,
                        result_method=cell,
                        visible_grounding_mode="fcpg",
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
                        forced_prefix_actions=reference_actions[episode],
                    )
                    result.pop("_prefix_native_tokens")
                    result.pop("_prefix_sampled_visual")
                    source_row = expected[(seed, episode, cell)]
                    _validate_replay(result, source_row)
                    traces.append(
                        {
                            "model_seed": seed,
                            "physical_episode": episode,
                            "cell": cell,
                            "rollout": result,
                            "summary": summarize_physical_trace(step_rows),
                            "steps": step_rows,
                        }
                    )
                    print(
                        f"seed={seed} episode={episode:02d} cell={cell} "
                        f"mechanism={traces[-1]['summary']['retention_mechanism']}",
                        flush=True,
                    )
    finally:
        rollout_module.make_env = original_make_env

    output = {
        "protocol": "fcpg_lewm_factorial_critical_physical_trace_v1",
        "classification": "matched_development_failure_diagnostic",
        "config": str(_resolve(args.config)),
        "model_seeds": list(model_seeds),
        "physical_episodes": list(episodes),
        "cells": CELLS,
        "traces": traces,
    }
    output_path = _resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
