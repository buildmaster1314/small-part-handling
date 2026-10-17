#!/usr/bin/env python3
"""Convert LeRobot SO-101 datasets into a compact LeWM training cache."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import imageio_ffmpeg
import numpy as np
import pyarrow.parquet as pq


CANONICAL_JOINTS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


@dataclass(frozen=True)
class DataVideoPair:
    data_path: Path
    video_path: Path
    row_start: int = 0
    row_stop: int | None = None
    frame_ranges: tuple[tuple[int, int], ...] = ()
    source_frames: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument(
        "--allow-tail-trim",
        action="store_true",
        help="Permit a parquet/video length mismatch known to be tail-only.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_info(root: Path) -> dict:
    path = root / "meta" / "info.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing LeRobot metadata: {path}")
    return json.loads(path.read_text())


def load_task_names(root: Path) -> dict[str, str]:
    jsonl_path = root / "meta" / "tasks.jsonl"
    if jsonl_path.is_file():
        rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
        return {str(int(row["task_index"])): str(row["task"]) for row in rows}

    parquet_path = root / "meta" / "tasks.parquet"
    if not parquet_path.is_file():
        return {}
    table = pq.read_table(parquet_path)
    if "task_index" not in table.column_names:
        return {}
    label_column = next(
        (name for name in ("task", "__index_level_0__") if name in table.column_names),
        None,
    )
    if label_column is None:
        return {}
    indices = table["task_index"].to_pylist()
    labels = table[label_column].to_pylist()
    return {str(int(index)): str(label) for index, label in zip(indices, labels)}


def canonical_order(info: dict, feature: str) -> list[int]:
    names = info["features"][feature].get("names")
    if not names:
        raise ValueError(f"{feature} has no joint names")
    aliases = {name.removesuffix(".pos"): i for i, name in enumerate(names)}
    order = []
    for canonical in CANONICAL_JOINTS:
        key = canonical.removesuffix(".pos")
        if key not in aliases:
            raise ValueError(f"{feature} missing canonical joint {canonical}: {names}")
        order.append(aliases[key])
    return order


def discover_pairs(root: Path, info: dict, camera: str) -> list[DataVideoPair]:
    if camera not in info["features"]:
        available = sorted(k for k in info["features"] if k.startswith("observation.images."))
        raise ValueError(f"camera {camera!r} unavailable; choose one of {available}")

    version = str(info.get("codebase_version", ""))
    if version.startswith("v2"):
        pairs = []
        for data_path in sorted(root.glob("data/chunk-*/episode_*.parquet")):
            chunk = data_path.parent.name
            video_path = root / "videos" / chunk / camera / f"{data_path.stem}.mp4"
            pairs.append(DataVideoPair(data_path, video_path))
        return pairs

    data_paths = sorted(root.glob("data/chunk-*/file-*.parquet"))
    video_paths = sorted((root / "videos" / camera).glob("chunk-*/file-*.mp4"))
    # LeRobot v3 shards tabular data and videos independently according to
    # separate size budgets. Some datasets therefore contain one parquet file
    # but several video files. The episode metadata records the authoritative
    # global parquet row range for every video shard.
    if len(data_paths) == 1:
        episode_paths = sorted(root.glob("meta/episodes/chunk-*/file-*.parquet"))
        if not episode_paths:
            raise FileNotFoundError("independently sharded v3 data has no episode metadata")
        chunk_column = f"videos/{camera}/chunk_index"
        file_column = f"videos/{camera}/file_index"
        from_column = f"videos/{camera}/from_timestamp"
        to_column = f"videos/{camera}/to_timestamp"
        columns = [
            "dataset_from_index",
            "dataset_to_index",
            chunk_column,
            file_column,
            from_column,
            to_column,
        ]
        rows = []
        for episode_path in episode_paths:
            table = pq.read_table(episode_path, columns=columns)
            rows.extend(zip(*(table[column].to_pylist() for column in columns)))

        rows.sort(key=lambda row: int(row[0]))
        expected_start = 0
        completed_shards: set[tuple[int, int]] = set()
        active_shard: tuple[int, int] | None = None
        fps = int(info.get("fps", 30))
        shard_frame_ranges: dict[tuple[int, int], list[tuple[int, int]]] = {}
        shard_source_frames: dict[tuple[int, int], int] = {}
        previous_video_stop: dict[tuple[int, int], int] = {}
        for row_start, row_stop, chunk_index, file_index, from_timestamp, to_timestamp in rows:
            row_start = int(row_start)
            row_stop = int(row_stop)
            shard = (int(chunk_index), int(file_index))
            if row_start != expected_start or row_stop <= row_start:
                raise ValueError(
                    "independently sharded v3 episode rows are not contiguous: "
                    f"expected row {expected_start}, found [{row_start}, {row_stop})"
                )
            if shard != active_shard:
                if shard in completed_shards:
                    raise ValueError(f"v3 video shard {shard} occupies disjoint row ranges")
                if active_shard is not None:
                    completed_shards.add(active_shard)
                active_shard = shard
            video_start = int(round(float(from_timestamp) * fps))
            video_stop = int(round(float(to_timestamp) * fps))
            if video_start != previous_video_stop.get(shard, 0):
                raise ValueError(
                    f"v3 video shard {shard} has a timestamp gap at frame {video_start}"
                )
            data_length = row_stop - row_start
            if video_stop - video_start < data_length:
                raise ValueError(
                    f"v3 video shard {shard} has fewer frames than data in "
                    f"[{row_start}, {row_stop})"
                )
            shard_frame_ranges.setdefault(shard, []).append(
                (video_start, video_start + data_length)
            )
            shard_source_frames[shard] = video_stop
            previous_video_stop[shard] = video_stop
            expected_start = row_stop

        ranges: dict[tuple[int, int], list[int]] = {}
        for row_start, row_stop, chunk_index, file_index, _, _ in rows:
            key = (int(chunk_index), int(file_index))
            bounds = ranges.setdefault(key, [int(row_start), int(row_stop)])
            bounds[0] = min(bounds[0], int(row_start))
            bounds[1] = max(bounds[1], int(row_stop))

        pairs = []
        expected_start = 0
        for (chunk_index, file_index), (row_start, row_stop) in sorted(ranges.items()):
            if row_start != expected_start:
                raise ValueError(
                    "independently sharded v3 metadata is not contiguous: "
                    f"expected row {expected_start}, found {row_start}"
                )
            video_path = (
                root
                / "videos"
                / camera
                / f"chunk-{chunk_index:03d}"
                / f"file-{file_index:03d}.mp4"
            )
            pairs.append(
                DataVideoPair(
                    data_paths[0],
                    video_path,
                    row_start,
                    row_stop,
                    tuple(shard_frame_ranges[(chunk_index, file_index)]),
                    shard_source_frames[(chunk_index, file_index)],
                )
            )
            expected_start = row_stop

        parquet_rows = pq.read_metadata(data_paths[0]).num_rows
        if expected_start != parquet_rows:
            raise ValueError(
                "independently sharded v3 metadata does not cover the parquet stream: "
                f"{parquet_rows} rows versus {expected_start} indexed rows"
            )
        return pairs

    if len(data_paths) == len(video_paths):
        return [
            DataVideoPair(data_path, video_path)
            for data_path, video_path in zip(data_paths, video_paths)
        ]

    raise ValueError(
        "unsupported independently sharded LeRobot v3 layout: "
        f"{len(data_paths)} parquet files versus {len(video_paths)} video files"
    )


def slice_pair_table(table, pair: DataVideoPair):
    if pair.row_stop is None:
        return table
    return table.slice(pair.row_start, pair.row_stop - pair.row_start)


def read_fixed_list(column, order: list[int]) -> np.ndarray:
    values = np.asarray(column.to_pylist(), dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected a matrix column, got shape {values.shape}")
    return values[:, order]


def build_frame_selection(
    source_frames: int, frame_ranges: tuple[tuple[int, int], ...]
) -> np.ndarray:
    selected = np.zeros(source_frames, dtype=bool)
    for start, stop in frame_ranges:
        if start < 0 or stop <= start or stop > source_frames:
            raise ValueError(f"invalid video frame range [{start}, {stop})/{source_frames}")
        if selected[start:stop].any():
            raise ValueError(f"overlapping video frame range [{start}, {stop})")
        selected[start:stop] = True
    return selected


def decode_video(
    path: Path,
    expected: int,
    image_size: int,
    *,
    source_frames: int | None = None,
    frame_ranges: tuple[tuple[int, int], ...] = (),
) -> Iterable[np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing video: {path}")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    vf = (
        f"scale={image_size}:{image_size}:force_original_aspect_ratio=decrease,"
        f"pad={image_size}:{image_size}:(ow-iw)/2:(oh-ih)/2"
    )
    decode_frames = source_frames if source_frames is not None else expected
    selected = (
        build_frame_selection(decode_frames, frame_ranges)
        if frame_ranges
        else np.ones(decode_frames, dtype=bool)
    )
    if int(selected.sum()) != expected:
        raise ValueError(
            f"video selection has {int(selected.sum())} frames, expected {expected}: {path}"
        )
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "2",
        "-i",
        str(path),
        "-frames:v",
        str(decode_frames),
        "-vf",
        vf,
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    frame_bytes = image_size * image_size * 3
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    try:
        yielded = 0
        for index in range(decode_frames):
            payload = process.stdout.read(frame_bytes)
            if len(payload) != frame_bytes:
                stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
                raise RuntimeError(
                    f"video ended at frame {index}/{decode_frames}: {path}\n{stderr[-2000:]}"
                )
            if selected[index]:
                yielded += 1
                yield np.frombuffer(payload, dtype=np.uint8).reshape(image_size, image_size, 3)
        if yielded != expected:
            raise RuntimeError(f"video selection yielded {yielded}/{expected} frames: {path}")

    finally:
        if process.stdout:
            process.stdout.close()
        return_code = process.wait()
        if return_code != 0:
            stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            raise RuntimeError(f"ffmpeg failed for {path}: {stderr[-2000:]}")


def selected_rows(
    episode_index: np.ndarray,
    max_episodes: int | None,
    selected_episode_ids: set[int],
) -> np.ndarray:
    if max_episodes is None:
        return np.ones(len(episode_index), dtype=bool)
    episodes = [int(value) for value in np.unique(episode_index)]
    existing = [episode for episode in episodes if episode in selected_episode_ids]
    remaining = max(0, max_episodes - len(selected_episode_ids))
    new = [episode for episode in episodes if episode not in selected_episode_ids][:remaining]
    selected_episode_ids.update(new)
    return np.isin(episode_index, existing + new)


def scan_pairs(
    pairs: list[DataVideoPair], max_episodes: int | None, allow_tail_trim: bool
) -> tuple[int, list[tuple[DataVideoPair, int, int]]]:
    total = 0
    counts = []
    seen_episodes: set[int] = set()
    for pair in pairs:
        table = pq.read_table(pair.data_path, columns=["episode_index"])
        table = slice_pair_table(table, pair)
        episodes = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        video_frames, _ = imageio_ffmpeg.count_frames_and_secs(str(pair.video_path))
        video_frame_count = int(video_frames)
        if pair.frame_ranges:
            if pair.source_frames != video_frame_count:
                raise ValueError(
                    f"video/metadata length mismatch for {pair.video_path}: "
                    f"{video_frame_count} frames versus {pair.source_frames} indexed frames"
                )
            selected_frame_count = sum(stop - start for start, stop in pair.frame_ranges)
            if selected_frame_count != len(episodes):
                raise ValueError(
                    f"video selection/data mismatch for {pair.video_path}: "
                    f"{selected_frame_count} selected frames versus {len(episodes)} rows"
                )
            usable_rows = len(episodes)
        else:
            usable_rows = min(len(episodes), video_frame_count)
        if not pair.frame_ranges and video_frame_count != len(episodes):
            if not allow_tail_trim:
                raise ValueError(
                    f"parquet/video length mismatch for {pair.data_path}: "
                    f"{len(episodes)} rows versus {video_frame_count} frames; "
                    "exclude the episode or use --allow-tail-trim only after verifying "
                    "the missing frames are a contiguous tail"
                )
            print(
                f"warning: trimming {pair.data_path.name} from {len(episodes)} "
                f"rows and {video_frame_count} video frames to {usable_rows} aligned rows"
            )
        episodes = episodes[:usable_rows]
        if max_episodes is None:
            count = len(episodes)
        else:
            keep = selected_rows(episodes, max_episodes, seen_episodes)
            count = int(keep.sum())
        counts.append((pair, count, usable_rows))
        total += count
        if max_episodes is not None and len(seen_episodes) >= max_episodes:
            # A v3 file may contain the selected episodes plus later episodes, so keep that pair.
            if str(pair.data_path.name).startswith("episode_"):
                break
    return total, counts


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        print(f"cache already exists: {output}")
        return

    info = load_info(root)
    action_order = canonical_order(info, "action")
    state_order = canonical_order(info, "observation.state")
    pairs = discover_pairs(root, info, args.camera)
    if not pairs:
        raise FileNotFoundError(f"no data/video pairs found below {root}")
    missing_pairs = [pair for pair in pairs if not pair.video_path.is_file()]
    if missing_pairs and not args.skip_missing:
        raise FileNotFoundError(f"dataset download is incomplete: {missing_pairs[0].video_path}")
    if missing_pairs:
        for pair in missing_pairs:
            print(f"warning: skipping missing video {pair.video_path}")
        pairs = [pair for pair in pairs if pair.video_path.is_file()]

    total, pair_counts = scan_pairs(pairs, args.max_episodes, args.allow_tail_trim)
    if total <= 0:
        raise ValueError("no selected frames")

    output.mkdir(parents=True, exist_ok=True)
    frames = np.lib.format.open_memmap(
        output / "frames.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(total, args.image_size, args.image_size, 3),
    )
    actions = np.lib.format.open_memmap(
        output / "actions.npy", mode="w+", dtype=np.float32, shape=(total, 6)
    )
    states = np.lib.format.open_memmap(
        output / "states.npy", mode="w+", dtype=np.float32, shape=(total, 6)
    )
    episodes_out = np.lib.format.open_memmap(
        output / "episodes.npy", mode="w+", dtype=np.int32, shape=(total,)
    )
    tasks_out = np.lib.format.open_memmap(
        output / "tasks.npy", mode="w+", dtype=np.int16, shape=(total,)
    )
    frame_indices_out = np.lib.format.open_memmap(
        output / "frame_indices.npy", mode="w+", dtype=np.int32, shape=(total,)
    )

    cursor = 0
    selected_episode_ids: set[int] = set()
    for pair_number, (pair, selected_count, usable_rows) in enumerate(pair_counts, start=1):
        if selected_count == 0:
            continue
        table = pq.read_table(
            pair.data_path,
            columns=[
                "action",
                "observation.state",
                "episode_index",
                "frame_index",
                "task_index",
            ],
        )
        table = slice_pair_table(table, pair)
        episode_index = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)[:usable_rows]
        frame_index = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)[:usable_rows]
        task_index = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)[:usable_rows]
        action = read_fixed_list(table["action"], action_order)[:usable_rows]
        state = read_fixed_list(table["observation.state"], state_order)[:usable_rows]

        keep = selected_rows(episode_index, args.max_episodes, selected_episode_ids)

        expected_video_frames = usable_rows
        write_at = cursor
        for row, frame in enumerate(
            decode_video(
                pair.video_path,
                expected_video_frames,
                args.image_size,
                source_frames=pair.source_frames,
                frame_ranges=pair.frame_ranges,
            )
        ):
            if not keep[row]:
                continue
            frames[write_at] = frame
            actions[write_at] = action[row]
            states[write_at] = state[row]
            episodes_out[write_at] = episode_index[row]
            tasks_out[write_at] = task_index[row]
            frame_indices_out[write_at] = frame_index[row]
            write_at += 1
        cursor = write_at
        print(
            f"[{pair_number}/{len(pair_counts)}] {pair.video_path.name}: "
            f"{cursor}/{total} cached frames"
        )

    if cursor != total:
        raise RuntimeError(f"cache row mismatch: wrote {cursor}, allocated {total}")
    for array in (frames, actions, states, episodes_out, tasks_out, frame_indices_out):
        array.flush()

    task_names = load_task_names(root)
    discarded_video_frames = sum(
        pair.source_frames - sum(stop - start for start, stop in pair.frame_ranges)
        for pair in pairs
        if pair.source_frames is not None and pair.frame_ranges
    )

    manifest = {
        "format": "so101-lewm-cache-v1",
        "name": args.name,
        "dataset_id": root.name,
        "source_root": str(root),
        "robot_type": info.get("robot_type"),
        "camera": args.camera,
        "image_size": args.image_size,
        "fps": info.get("fps", 30),
        "frames": total,
        "episodes": len(np.unique(np.asarray(episodes_out))),
        "tasks": len(np.unique(np.asarray(tasks_out))),
        "task_names": task_names,
        "skipped_missing_videos": len(missing_pairs),
        "discarded_video_frames": discarded_video_frames,
        "joint_names": list(CANONICAL_JOINTS),
        "action_mean": np.asarray(actions).mean(axis=0).tolist(),
        "action_std": np.maximum(np.asarray(actions).std(axis=0), 1e-6).tolist(),
        "state_mean": np.asarray(states).mean(axis=0).tolist(),
        "state_std": np.maximum(np.asarray(states).std(axis=0), 1e-6).tolist(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
