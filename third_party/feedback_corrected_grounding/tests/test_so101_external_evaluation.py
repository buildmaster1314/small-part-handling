import json
from pathlib import Path

import numpy as np

from scripts.evaluate_so101_external_lewm import (
    all_validation_assignments,
    relative_gain,
)
from scripts.train_so101_lewm import Cache, SO101Sequences


def _write_cache(root: Path) -> None:
    root.mkdir()
    frames = 26
    manifest = {
        "format": "so101-lewm-cache-v1",
        "name": "external-test",
        "dataset_id": "external-test",
        "source_root": str(root),
        "robot_type": "so_follower",
        "camera": "observation.images.topview",
        "frames": frames,
        "episodes": 2,
        "tasks": 1,
        "task_names": {"0": "external task"},
        "joint_names": [f"joint-{index}" for index in range(6)],
        "action_mean": [0.0] * 6,
        "action_std": [1.0] * 6,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    np.save(root / "frames.npy", np.zeros((frames, 8, 8, 3), dtype=np.uint8))
    np.save(root / "actions.npy", np.zeros((frames, 6), dtype=np.float32))
    np.save(root / "states.npy", np.zeros((frames, 6), dtype=np.float32))
    np.save(root / "episodes.npy", np.repeat([3, 7], frames // 2).astype(np.int32))
    np.save(root / "tasks.npy", np.zeros(frames, dtype=np.int16))
    np.save(
        root / "frame_indices.npy",
        np.tile(np.arange(frames // 2), 2).astype(np.int32),
    )


def test_external_sequence_dataset_uses_every_episode(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    _write_cache(cache_root)
    assignments = all_validation_assignments([Cache(cache_root)])

    dataset = SO101Sequences(
        [cache_root],
        "val",
        frameskip=2,
        history=2,
        exclusions=set(),
        split_seed=1,
        val_fraction=0.1,
        joint_scale=100.0,
        motion_threshold=0.01,
        episode_assignments=assignments,
    )

    assert assignments == {("external-test", 3): "val", ("external-test", 7): "val"}
    assert len(dataset) == 18
    assert {int(dataset[index]["episode"]) for index in range(len(dataset))} == {3, 7}


def test_relative_gain_uses_baseline_as_denominator() -> None:
    assert relative_gain(0.75, 1.0) == 0.25
