#!/usr/bin/env python3
"""Replay exposed Button A1 rows and compare physical versus latent spread."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_collection import (  # noqa: E402
    FRAME_SKIP,
    capture_branch_snapshot,
    make_metaworld_episode_runtime,
)
from fcpg.lewm_action_branch_metrics import (  # noqa: E402
    center_whitened_branches,
    fit_train_moments,
)
from fcpg.lewm_button_branch_diagnostic import (  # noqa: E402
    BUTTON_TASK,
    BUTTON_TASK_INDEX,
    classify_button_phase,
    read_button_physical_state,
    replay_button_branches,
    summarize_button_diagnostic,
)
from fcpg.lewm_counterfactual_v2_runtime import _close_episode_env_safely  # noqa: E402


DEFAULT_INPUT = "outputs/lewm_action_branch_a1/action_branch_a1_formal.npz"
DEFAULT_OUTPUT_DIR = "outputs/lewm_action_branch_a1/button_phase_diagnostic"


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "development"),
        default=("train", "development"),
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--camera", default="corner")
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--benchmark-seed", type=int, default=18_301)
    parser.add_argument("--environment-seed-base", type=int, default=40_000)
    return parser


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Button action-branch phase diagnostic",
        "",
        "This is a disposable train/development diagnostic, not a new gate.",
        "",
        "| phase | bundles | latent median | informative | TCP endpoint spread | button endpoint spread | success disagreement |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for phase, row in summary["per_phase"].items():
        lines.append(
            "| {phase} | {count} | {latent:.6f} | {informative:.1%} | "
            "{tcp:.5f} m | {button:.5f} m | {success:.1%} |".format(
                phase=phase,
                count=row["bundle_count"],
                latent=row["latent_branch_variance"]["median"],
                informative=row["latent_informative_fraction"],
                tcp=row["tcp_endpoint_max_pairwise_m"]["median"],
                button=row["button_endpoint_max_pairwise_m"]["median"],
                success=row["success_disagreement_fraction"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive")

    source = _resolve(args.input)
    with np.load(source, allow_pickle=False) as store:
        split = store["split"].astype(str)
        task_index = np.asarray(store["task_index"], dtype=np.int64)
        selected = np.flatnonzero(
            (task_index == BUTTON_TASK_INDEX) & np.isin(split, tuple(args.splits))
        )
        if args.max_rows is not None:
            selected = selected[: int(args.max_rows)]
        if selected.size == 0:
            raise RuntimeError("no exposed Button rows matched the requested splits")

        train_target = np.asarray(store["target_embedding"])[split == "train"]
        moments = fit_train_moments(train_target)
        selected_target = np.asarray(store["target_embedding"])[selected]
        centered = center_whitened_branches(selected_target, moments)
        latent_variance = np.mean(
            np.square(centered), axis=(1, 2, 3, 4), dtype=np.float64
        )
        task_instance = np.asarray(store["task_instance_index"], dtype=np.int64)[selected]
        hazard_step = np.asarray(store["hazard_id"], dtype=np.int64)[selected]
        selected_split = split[selected]
        branch_chunks = np.asarray(store["future_action"], dtype=np.float32)[selected]

    order = np.lexsort((hazard_step, task_instance))
    selected = selected[order]
    task_instance = task_instance[order]
    hazard_step = hazard_step[order]
    selected_split = selected_split[order]
    latent_variance = latent_variance[order]
    branch_chunks = branch_chunks[order]

    phase: list[str] = []
    source_near: list[bool] = []
    source_success: list[bool] = []
    physical_rows: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "tcp",
            "qpos",
            "button_position",
            "button_joint",
            "object_to_target",
            "near_object",
            "success",
            "touching",
        )
    }

    for instance in np.unique(task_instance):
        row_positions = np.flatnonzero(task_instance == instance)
        environment_seed = (
            int(args.environment_seed_base) + BUTTON_TASK_INDEX * 10_000 + int(instance)
        )
        runtime = make_metaworld_episode_runtime(
            BUTTON_TASK,
            int(instance),
            environment_seed,
            camera=str(args.camera),
            width=int(args.width),
            height=int(args.height),
            benchmark_seed=int(args.benchmark_seed),
        )
        try:
            observation = np.asarray(runtime.observation, dtype=np.float32)
            positions_by_raw = {
                int(hazard_step[position]) * FRAME_SKIP: int(position)
                for position in row_positions
            }
            final_raw = max(positions_by_raw)
            snapshots: dict[int, Any] = {}
            source_states: dict[int, Any] = {}
            for raw_step in range(final_raw + 1):
                if raw_step in positions_by_raw:
                    position = positions_by_raw[raw_step]
                    snapshots[position] = capture_branch_snapshot(
                        runtime.env,
                        include_process_rng=True,
                    )
                    source_states[position] = read_button_physical_state(
                        runtime.env,
                        observation,
                    )
                if raw_step == final_raw:
                    break
                action = np.asarray(runtime.expert.get_action(observation), dtype=np.float32)
                action = np.clip(
                    action,
                    runtime.env.action_space.low,
                    runtime.env.action_space.high,
                ).astype(np.float32)
                observation, _, terminated, truncated, _ = runtime.env.step(action)
                observation = np.asarray(observation, dtype=np.float32)
                if (terminated or truncated) and raw_step + 1 < final_raw:
                    raise RuntimeError("Button source ended before a selected hazard")

            for position in row_positions:
                source_state = source_states[int(position)]
                phase.append(
                    classify_button_phase(
                        near_object=source_state.near_object,
                        success=source_state.success,
                        button_joint=source_state.button_joint,
                    )
                )
                source_near.append(source_state.near_object)
                source_success.append(source_state.success)
                raw_actions = branch_chunks[position].reshape(8, -1, 4)
                replay = replay_button_branches(
                    runtime,
                    snapshots[int(position)],
                    raw_actions,
                )
                for name, value in replay.items():
                    physical_rows[name].append(value)
        finally:
            _close_episode_env_safely(runtime.env)

    arrays = {
        "source_row": selected.astype(np.int64),
        "task_instance_index": task_instance.astype(np.int64),
        "hazard_step": hazard_step.astype(np.int64),
        "split": selected_split.astype(str),
        "phase": np.asarray(phase),
        "source_near_object": np.asarray(source_near, dtype=bool),
        "source_success": np.asarray(source_success, dtype=bool),
        "latent_branch_variance": latent_variance.astype(np.float64),
    }
    arrays.update(
        {name: np.stack(values) for name, values in physical_rows.items()}
    )
    summary = summarize_button_diagnostic(
        phase=arrays["phase"],
        latent_branch_variance=arrays["latent_branch_variance"],
        tcp=arrays["tcp"],
        qpos=arrays["qpos"],
        button_position=arrays["button_position"],
        button_joint=arrays["button_joint"],
        object_to_target=arrays["object_to_target"],
        success=arrays["success"],
        hazard_step=arrays["hazard_step"],
    )
    summary["source"] = str(source)
    summary["splits"] = list(args.splits)
    summary["test_rows_scored"] = 0

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "rows.npz", **arrays)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _write_markdown(output_dir / "summary.md", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
