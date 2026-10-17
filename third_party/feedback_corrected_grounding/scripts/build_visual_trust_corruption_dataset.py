#!/usr/bin/env python3
"""Build a minimal MetaWorld dataset with local VisualTrust corruptions."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from numbers import Integral
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.visual_trust_corruptions import (  # noqa: E402
    CORRUPTION_TYPES,
    apply_visual_trust_corruption,
)


CACHE_REQUIRED_KEYS = (
    "obs",
    "q_true",
    "q_nominal",
    "keypoints_xyz",
    "valid_true",
    "valid_nominal",
    "episode_id",
    "frame_id",
)
EPISODE_METADATA_KEYS = (
    "task_name",
    "task_index",
    "episode_id",
    "env_seed",
    "frame_id",
    "episode_recorded_frame_count",
)
ANNOTATION_KEYS = (
    "corruption_active",
    "corruption_family",
    "corruption_target_anchor",
    "corruption_recovery_onset",
    "frames_since_recovery",
)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _seed(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
        raise TypeError("seed must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError("seed must be non-negative")
    return result


def _copy_array(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError(f"{name} must not use object storage")
    return np.array(array, copy=True)


def _validate_source(
    source: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray, int, int]:
    source_keys = set(source)
    required = set(CACHE_REQUIRED_KEYS) | {"task_name"}
    missing = sorted(required - source_keys)
    if missing:
        raise KeyError(f"input dataset is missing required arrays: {missing}")
    if "rgb_camera_only" in source_keys:
        clean_key = "rgb_camera_only"
    elif "rgb" in source_keys:
        clean_key = "rgb"
    else:
        raise KeyError("input dataset must contain rgb_camera_only or rgb")

    arrays = {
        name: _copy_array(source[name], name=name)
        for name in required
    }
    for name in EPISODE_METADATA_KEYS:
        if name in source_keys and name not in arrays:
            arrays[name] = _copy_array(source[name], name=name)

    clean = _copy_array(source[clean_key], name=clean_key)
    if (
        clean.ndim != 4
        or clean.shape[-1] != 3
        or any(int(size) <= 0 for size in clean.shape)
    ):
        raise ValueError(
            f"{clean_key} must have shape [N, H, W, 3] with positive "
            f"dimensions, got {clean.shape}"
        )
    if clean.dtype != np.uint8:
        raise TypeError(f"{clean_key} must have dtype uint8, got {clean.dtype}")

    sample_count, height, width, _ = (int(size) for size in clean.shape)
    q_true = arrays["q_true"]
    q_nominal = arrays["q_nominal"]
    if q_true.ndim != 3 or q_true.shape[0] != sample_count or q_true.shape[-1] != 2:
        raise ValueError(
            f"q_true must have shape [N, K, 2] for N={sample_count}, "
            f"got {q_true.shape}"
        )
    if q_nominal.shape != q_true.shape:
        raise ValueError(
            f"q_nominal must have shape {q_true.shape}, got {q_nominal.shape}"
        )
    anchor_count = int(q_true.shape[1])
    if anchor_count <= 0:
        raise ValueError("q_true must contain at least one anchor")

    expected_anchor_shape = (sample_count, anchor_count)
    for name in ("valid_true", "valid_nominal"):
        value = arrays[name]
        if value.shape != expected_anchor_shape or value.dtype != np.bool_:
            raise TypeError(
                f"{name} must be a boolean array with shape "
                f"{expected_anchor_shape}, got {value.shape} with dtype {value.dtype}"
            )
    keypoints_xyz = arrays["keypoints_xyz"]
    if keypoints_xyz.shape != (sample_count, anchor_count, 3):
        raise ValueError(
            "keypoints_xyz must have shape "
            f"{(sample_count, anchor_count, 3)}, got {keypoints_xyz.shape}"
        )
    obs = arrays["obs"]
    if obs.ndim != 2 or obs.shape[0] != sample_count or obs.shape[1] <= 0:
        raise ValueError(
            f"obs must have shape [N, D] with N={sample_count} and D > 0, "
            f"got {obs.shape}"
        )

    for name in ("q_true", "q_nominal", "keypoints_xyz", "obs"):
        value = arrays[name]
        if value.dtype.kind not in "fiu" or not bool(np.isfinite(value).all()):
            raise ValueError(f"{name} must be a finite numeric array")

    task_name = arrays["task_name"]
    episode_id = arrays["episode_id"]
    frame_id = arrays["frame_id"]
    for name, value in (
        ("task_name", task_name),
        ("episode_id", episode_id),
        ("frame_id", frame_id),
    ):
        if value.shape != (sample_count,):
            raise ValueError(
                f"{name} must have shape [{sample_count}], got {value.shape}"
            )
    if task_name.dtype.kind not in "US" or bool(np.any(task_name.astype(str) == "")):
        raise TypeError("task_name must be a non-empty string array")
    for name, value in (("episode_id", episode_id), ("frame_id", frame_id)):
        if value.dtype.kind not in "iu" or value.dtype == np.bool_:
            raise TypeError(f"{name} must be an integer array")
    if bool(np.any(frame_id < 0)):
        raise ValueError("frame_id must be non-negative")

    for name in EPISODE_METADATA_KEYS:
        if name in arrays and arrays[name].shape != (sample_count,):
            raise ValueError(
                f"episode metadata {name} must have shape [{sample_count}], "
                f"got {arrays[name].shape}"
            )

    return arrays, clean, height, width


def _episode_groups(
    arrays: Mapping[str, np.ndarray],
) -> list[tuple[tuple[str, int], np.ndarray]]:
    tasks = arrays["task_name"].astype(str)
    episodes = arrays["episode_id"]
    frames = arrays["frame_id"]
    groups: dict[tuple[str, int], list[int]] = {}
    for row, (task, episode) in enumerate(zip(tasks, episodes, strict=True)):
        groups.setdefault((str(task), int(episode)), []).append(row)

    result: list[tuple[tuple[str, int], np.ndarray]] = []
    for key in sorted(groups):
        indices = np.asarray(groups[key], dtype=np.int64)
        order = np.argsort(frames[indices], kind="stable")
        ordered = indices[order]
        ordered_frames = frames[ordered].astype(np.int64, copy=False)
        if ordered_frames.size > 1 and not bool(np.all(np.diff(ordered_frames) == 1)):
            raise ValueError(
                f"episode {key} must have unique, contiguous frame IDs; "
                f"got {ordered_frames.tolist()}"
            )
        result.append((key, ordered))
    return result


def _sample_event(
    episode_key: tuple[str, int],
    ordered_indices: np.ndarray,
    valid_anchor: np.ndarray,
    *,
    minimum_duration: int,
    maximum_duration: int,
    minimum_recovery_frames: int,
    rng: np.random.Generator,
) -> tuple[int, int, int]:
    frame_count = int(ordered_indices.size)
    if frame_count < minimum_duration + minimum_recovery_frames:
        raise ValueError(
            f"episode {episode_key} has {frame_count} frames but needs at least "
            f"{minimum_duration + minimum_recovery_frames} for corruption and "
            "clean recovery"
        )

    choices: list[tuple[int, int, int]] = []
    for onset in range(
        frame_count - minimum_duration - minimum_recovery_frames + 1
    ):
        largest_duration = min(
            maximum_duration,
            frame_count - onset - minimum_recovery_frames,
        )
        for duration in range(minimum_duration, largest_duration + 1):
            # The target anchor must remain observable throughout both the
            # corruption and the minimum recovery-confirmation window.
            rows = ordered_indices[
                onset : onset + duration + minimum_recovery_frames
            ]
            anchors = np.flatnonzero(np.all(valid_anchor[rows], axis=0))
            choices.extend(
                (onset, duration, int(anchor)) for anchor in anchors.tolist()
            )
    if not choices:
        raise ValueError(
            f"episode {episode_key} has no valid in-bounds anchor window for "
            f"durations [{minimum_duration}, {maximum_duration}]"
        )
    return choices[int(rng.integers(0, len(choices)))]


def build_visual_trust_arrays(
    source: Mapping[str, np.ndarray],
    *,
    seed: int = 0,
    minimum_duration: int = 1,
    maximum_duration: int = 4,
    minimum_recovery_frames: int = 1,
) -> dict[str, np.ndarray]:
    """Return a minimal derived dataset without modifying any source array."""

    resolved_seed = _seed(seed)
    minimum = _positive_integer(minimum_duration, name="minimum_duration")
    maximum = _positive_integer(maximum_duration, name="maximum_duration")
    recovery_frames = _positive_integer(
        minimum_recovery_frames, name="minimum_recovery_frames"
    )
    if maximum < minimum:
        raise ValueError("maximum_duration must be at least minimum_duration")

    arrays, clean, height, width = _validate_source(source)
    sample_count = int(clean.shape[0])
    anchor_count = int(arrays["q_true"].shape[1])
    rng = np.random.default_rng(resolved_seed)

    q_true = arrays["q_true"].astype(np.float64, copy=False)
    finite_coordinates = np.isfinite(q_true).all(axis=-1)
    in_bounds = (
        (q_true[..., 0] >= 0.0)
        & (q_true[..., 0] < width)
        & (q_true[..., 1] >= 0.0)
        & (q_true[..., 1] < height)
    )
    valid_anchor = (
        arrays["valid_true"]
        & arrays["valid_nominal"]
        & finite_coordinates
        & in_bounds
    )

    corrupted = np.array(clean, copy=True, order="C")
    active = np.zeros((sample_count, anchor_count), dtype=np.bool_)
    recovery_onset = np.zeros_like(active)
    frames_since_recovery = np.full(
        (sample_count, anchor_count), -1, dtype=np.int32
    )
    target_anchor = np.full(sample_count, -1, dtype=np.int32)
    family_width = max(len(name) for name in CORRUPTION_TYPES)
    family = np.full(sample_count, "", dtype=f"<U{family_width}")

    groups = _episode_groups(arrays)
    for episode_key, ordered in groups:
        onset, duration, anchor = _sample_event(
            episode_key,
            ordered,
            valid_anchor,
            minimum_duration=minimum,
            maximum_duration=maximum,
            minimum_recovery_frames=recovery_frames,
            rng=rng,
        )
        event_family = CORRUPTION_TYPES[
            int(rng.integers(0, len(CORRUPTION_TYPES)))
        ]
        active_rows = ordered[onset : onset + duration]
        recovery_position = onset + duration
        recovery_row = int(ordered[recovery_position])

        active[active_rows, anchor] = True
        family[active_rows] = event_family
        target_anchor[active_rows] = anchor
        recovery_onset[recovery_row, anchor] = True
        recovery_rows = ordered[recovery_position:]
        frames_since_recovery[recovery_rows, anchor] = np.arange(
            recovery_rows.size, dtype=np.int32
        )

        for row in active_rows.tolist():
            image, _, _ = apply_visual_trust_corruption(
                clean[row],
                q_true[row, anchor],
                rng,
                corruption_type=event_family,
            )
            corrupted[row] = image

    output: dict[str, np.ndarray] = {
        "rgb": corrupted,
        "rgb_camera_only": clean,
    }
    for name in CACHE_REQUIRED_KEYS + EPISODE_METADATA_KEYS:
        if name in arrays and name not in output:
            output[name] = arrays[name]
    output.update(
        {
            "corruption_active": active,
            "corruption_family": family,
            "corruption_target_anchor": target_anchor,
            "corruption_recovery_onset": recovery_onset,
            "frames_since_recovery": frames_since_recovery,
        }
    )
    return output


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp.npz", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    seed: int = 0,
    minimum_duration: int = 1,
    maximum_duration: int = 4,
    minimum_recovery_frames: int = 8,
) -> Path:
    source_path = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if source_path == destination:
        raise ValueError("input and output paths must differ")
    if not source_path.is_file():
        raise FileNotFoundError(f"input dataset does not exist: {source_path}")

    with np.load(source_path, allow_pickle=False) as raw:
        available = set(raw.files)
        selected_image = (
            "rgb_camera_only" if "rgb_camera_only" in available else "rgb"
        )
        selected_keys = (
            set(CACHE_REQUIRED_KEYS)
            | set(EPISODE_METADATA_KEYS)
            | {selected_image}
        )
        source = {
            name: np.array(raw[name], copy=True)
            for name in selected_keys.intersection(available)
        }
    output = build_visual_trust_arrays(
        source,
        seed=seed,
        minimum_duration=minimum_duration,
        maximum_duration=maximum_duration,
        minimum_recovery_frames=minimum_recovery_frames,
    )
    _write_npz(destination, output)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", "--data", dest="input_path", type=Path, required=True
    )
    parser.add_argument("--output", dest="output_path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--minimum-duration", "--min-duration", type=int, default=1
    )
    parser.add_argument(
        "--maximum-duration", "--max-duration", type=int, default=4
    )
    parser.add_argument("--minimum-recovery-frames", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = build_dataset(
        args.input_path,
        args.output_path,
        seed=args.seed,
        minimum_duration=args.minimum_duration,
        maximum_duration=args.maximum_duration,
        minimum_recovery_frames=args.minimum_recovery_frames,
    )
    with np.load(output, allow_pickle=False) as data:
        sample_count = int(data["rgb"].shape[0])
        episode_count = len(
            set(
                zip(
                    data["task_name"].astype(str).tolist(),
                    data["episode_id"].astype(np.int64).tolist(),
                    strict=True,
                )
            )
        )
    print(f"output: {output}")
    print(f"samples: {sample_count}")
    print(f"episodes: {episode_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
