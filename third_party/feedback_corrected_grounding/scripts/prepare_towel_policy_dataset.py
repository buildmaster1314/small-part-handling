#!/usr/bin/env python3
"""Trim complete Towel demonstrations to their active-control interval."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.render_towel_policy_dataset import (  # noqa: E402
    episode_indices,
    summarize_episode,
    validate_dataset,
)


def first_stable_success(success: np.ndarray, required_frames: int) -> int | None:
    """Return the first index starting a consecutive successful run."""

    values = np.asarray(success, dtype=np.float32).reshape(-1) > 0.5
    if required_frames <= 0:
        raise ValueError("required_frames must be positive")
    if values.size < required_frames:
        return None
    run = np.convolve(
        values.astype(np.int16),
        np.ones(required_frames, dtype=np.int16),
        mode="valid",
    )
    matches = np.flatnonzero(run == required_frames)
    return None if matches.size == 0 else int(matches[0])


def active_control_indices(
    data: Mapping[str, np.ndarray],
    *,
    stable_success_frames: int,
    post_success_tail: int,
) -> tuple[np.ndarray, dict[int, int]]:
    """Select complete prefixes ending shortly after stable task success."""

    if post_success_tail < 0:
        raise ValueError("post_success_tail must be non-negative")
    selected: list[np.ndarray] = []
    recorded_counts: dict[int, int] = {}
    success = np.asarray(data["success"], dtype=np.float32)
    episodes = np.asarray(data["episode_id"], dtype=np.int64)
    for indices in episode_indices(data):
        episode = int(episodes[indices[0]])
        stable_start = first_stable_success(
            success[indices], stable_success_frames
        )
        if stable_start is None:
            raise ValueError(
                f"episode {episode} never reaches {stable_success_frames} "
                "consecutive successful frames"
            )
        stop = min(
            int(indices.size),
            stable_start + stable_success_frames + post_success_tail,
        )
        kept = indices[:stop]
        selected.append(kept)
        recorded_counts[episode] = int(kept.size)
    return np.concatenate(selected), recorded_counts


def prepare_arrays(
    data: Mapping[str, np.ndarray],
    *,
    stable_success_frames: int,
    post_success_tail: int,
) -> dict[str, np.ndarray]:
    """Slice every frame-aligned field while retaining dataset metadata."""

    validate_dataset(data)
    sample_count = int(np.asarray(data["rgb"]).shape[0])
    selected, recorded_counts = active_control_indices(
        data,
        stable_success_frames=stable_success_frames,
        post_success_tail=post_success_tail,
    )
    result: dict[str, np.ndarray] = {}
    fields = data.files if hasattr(data, "files") else data.keys()
    for name in fields:
        value = np.asarray(data[name])
        if value.ndim > 0 and value.shape[0] == sample_count:
            result[name] = value[selected]
        else:
            result[name] = value

    trimmed_episodes = np.asarray(result["episode_id"], dtype=np.int64)
    result["episode_recorded_frame_count"] = np.asarray(
        [recorded_counts[int(episode)] for episode in trimmed_episodes],
        dtype=np.int32,
    )
    result["dataset_type"] = np.asarray("towel_policy_active_control_v1")
    result["stable_success_frames"] = np.asarray(
        stable_success_frames, dtype=np.int32
    )
    result["post_success_tail"] = np.asarray(post_success_tail, dtype=np.int32)
    return result


def validate_towel_semantics(data: Mapping[str, np.ndarray]) -> None:
    """Reject demonstrations that do not complete the intended three round trips."""

    for indices in episode_indices(data):
        summary = summarize_episode(data, indices)
        if int(summary["wipe_half_cycles"]) != 6:
            raise ValueError(
                f"episode {summary['episode']} has "
                f"{summary['wipe_half_cycles']} wipe half-cycles; expected 6"
            )
        if summary["wipe_order"] != [
            "wipe_right",
            "wipe_left",
            "wipe_right",
            "wipe_left",
            "wipe_right",
            "wipe_left",
        ]:
            raise ValueError(
                f"episode {summary['episode']} has unexpected wipe order: "
                f"{summary['wipe_order']}"
            )
        if summary["first_success_frame"] is None:
            raise ValueError(f"episode {summary['episode']} never succeeds")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stable-success-frames", type=int, default=3)
    parser.add_argument("--post-success-tail", type=int, default=12)
    args = parser.parse_args()
    if args.stable_success_frames <= 0:
        parser.error("--stable-success-frames must be positive")
    if args.post_success_tail < 0:
        parser.error("--post-success-tail must be non-negative")

    source = Path(args.data).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with np.load(source, allow_pickle=False) as data:
        validate_towel_semantics(data)
        arrays = prepare_arrays(
            data,
            stable_success_frames=args.stable_success_frames,
            post_success_tail=args.post_success_tail,
        )
    validate_towel_semantics(arrays)

    temporary = output.with_suffix(output.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, output)
    print(f"source_frames: {sample_count_from(source)}")
    print(f"prepared_frames: {int(arrays['rgb'].shape[0])}")
    print(f"episodes: {len(np.unique(arrays['episode_id']))}")
    print(f"output: {output}")
    return 0


def sample_count_from(path: Path) -> int:
    with np.load(path, allow_pickle=False) as data:
        return int(data["rgb"].shape[0])


if __name__ == "__main__":
    raise SystemExit(main())
