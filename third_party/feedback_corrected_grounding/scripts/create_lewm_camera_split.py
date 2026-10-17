#!/usr/bin/env python3
"""Create a camera-aware, episode-local LeWM train/validation manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.lewm_provenance import sha256_file  # noqa: E402


def _source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("source must use non-empty NAME=PATH")
    return name.strip(), Path(raw_path).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=_source, required=True)
    parser.add_argument("--max-rotation-deg", type=float, default=4.0)
    parser.add_argument("--max-translation-cm", type=float, default=3.2)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _episode_rows(raw: np.lib.npyio.NpzFile) -> np.ndarray:
    episode_id = np.asarray(raw["episode_id"])
    task_name = raw["task_name"].astype(str)
    first = np.r_[True, (episode_id[1:] != episode_id[:-1]) | (task_name[1:] != task_name[:-1])]
    return np.flatnonzero(first)


def _split_group(
    labels: list[str],
    *,
    val_fraction: float,
    rng: np.random.Generator,
) -> tuple[list[str], list[str]]:
    if len(labels) < 2:
        raise ValueError("each camera stratum needs at least two eligible episodes")
    ordered = sorted(labels)
    permutation = rng.permutation(len(ordered))
    val_count = max(1, int(round(len(ordered) * val_fraction)))
    val_count = min(val_count, len(ordered) - 1)
    val_indices = {int(index) for index in permutation[:val_count]}
    train = [label for index, label in enumerate(ordered) if index not in val_indices]
    val = [label for index, label in enumerate(ordered) if index in val_indices]
    return train, val


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be strictly between zero and one")
    if args.max_rotation_deg < 0.0 or args.max_translation_cm < 0.0:
        raise ValueError("camera limits must be non-negative")
    names = [name for name, _ in args.source]
    if len(set(names)) != len(names):
        raise ValueError("source names must be unique")

    sources: dict[str, object] = {}
    for source_index, (name, path) in enumerate(args.source):
        with np.load(path, allow_pickle=False) as raw:
            required = {
                "episode_id",
                "task_name",
                "actual_camera_translation_xyz",
                "actual_camera_rotation_deg",
            }
            missing = required.difference(raw.files)
            if missing:
                raise KeyError(f"{path} is missing camera split fields: {sorted(missing)}")
            rows = _episode_rows(raw)
            episode_id = np.asarray(raw["episode_id"])[rows]
            task_name = raw["task_name"].astype(str)[rows]
            translation_cm = np.linalg.norm(
                np.asarray(raw["actual_camera_translation_xyz"], dtype=np.float64)[rows],
                axis=1,
            ) * 100.0
            rotation_deg = np.abs(
                np.asarray(raw["actual_camera_rotation_deg"], dtype=np.float64)[rows]
            )

        labels = [
            f"{task}:{episode}" for task, episode in zip(task_name, episode_id, strict=True)
        ]
        clean = (translation_cm <= 1e-8) & (rotation_deg <= 1e-8)
        eligible = (
            (translation_cm <= args.max_translation_cm)
            & (rotation_deg <= args.max_rotation_deg)
        )
        rng = np.random.default_rng(args.seed + source_index * 1009)
        train: list[str] = []
        val: list[str] = []
        for stratum in (clean & eligible, ~clean & eligible):
            group = [labels[index] for index in np.flatnonzero(stratum)]
            group_train, group_val = _split_group(
                group, val_fraction=args.val_fraction, rng=rng
            )
            train.extend(group_train)
            val.extend(group_val)
        excluded = [labels[index] for index in np.flatnonzero(~eligible)]
        if set(train) & set(val) or set(train) & set(excluded) or set(val) & set(excluded):
            raise RuntimeError("camera split contains overlapping episode sets")
        if len(train) + len(val) + len(excluded) != len(labels):
            raise RuntimeError("camera split does not partition all episodes")
        sources[name] = {
            "path": str(path),
            "data_sha256": sha256_file(path),
            "train_episodes": sorted(train),
            "val_episodes": sorted(val),
            "excluded_episodes": sorted(excluded),
            "counts": {
                "train": len(train),
                "validation": len(val),
                "excluded_stronger_camera": len(excluded),
                "total": len(labels),
            },
            "observed_camera": {
                "max_translation_cm": float(translation_cm.max()),
                "max_abs_rotation_deg": float(rotation_deg.max()),
            },
        }

    manifest = {
        "schema_version": 1,
        "protocol": "lewm_gate_l1_camera_aware_episode_split",
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "training_camera_limits": {
            "max_translation_cm": args.max_translation_cm,
            "max_abs_rotation_deg": args.max_rotation_deg,
        },
        "sources": sources,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
