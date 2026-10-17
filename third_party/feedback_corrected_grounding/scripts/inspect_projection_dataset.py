#!/usr/bin/env python3
"""Inspect an FCPG projection dataset `.npz` file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    path = Path(args.path)
    data = np.load(path, allow_pickle=False)
    print(f"path: {path}")
    print("keys:")
    for key in sorted(data.files):
        arr = data[key]
        print(f"  {key}: shape={arr.shape} dtype={arr.dtype}")

    if "pixel_error" in data.files:
        err = data["pixel_error"]
        print("pixel_error:")
        print(f"  mean={float(err.mean()):.4f}")
        print(f"  p50={float(np.percentile(err, 50)):.4f}")
        print(f"  p90={float(np.percentile(err, 90)):.4f}")
        print(f"  max={float(err.max()):.4f}")
    if "q_true_inside_patch" in data.files:
        inside = data["q_true_inside_patch"]
        print(f"inside_patch_rate: {float(inside.mean()):.4f}")
    if "task_name" in data.files:
        tasks, counts = np.unique(data["task_name"], return_counts=True)
        print("task_counts:")
        for task, count in zip(tasks, counts):
            print(f"  {task}: {int(count)}")
    if "task_index" in data.files:
        task_indices, counts = np.unique(data["task_index"], return_counts=True)
        print("task_index_counts:")
        for task_index, count in zip(task_indices, counts):
            print(f"  {int(task_index)}: {int(count)}")
    if "actual_camera_translation_xyz" in data.files:
        translations_cm = np.linalg.norm(data["actual_camera_translation_xyz"], axis=1) * 100.0
        print("actual_camera_translation_cm:")
        print(f"  mean={float(translations_cm.mean()):.4f}")
        print(f"  p90={float(np.percentile(translations_cm, 90)):.4f}")
        print(f"  max={float(translations_cm.max()):.4f}")
        if "actual_camera_rotation_deg" in data.files:
            clean = (translations_cm <= 1e-6) & (np.abs(data["actual_camera_rotation_deg"]) <= 1e-6)
            print(f"actual_camera_clean_rate: {float(clean.mean()):.4f}")
    if "actual_camera_rotation_deg" in data.files:
        rotations = np.abs(data["actual_camera_rotation_deg"].astype(np.float32))
        print("actual_camera_rotation_deg_abs:")
        print(f"  mean={float(rotations.mean()):.4f}")
        print(f"  p90={float(np.percentile(rotations, 90)):.4f}")
        print(f"  max={float(rotations.max()):.4f}")
    if "actual_camera_randomization" in data.files:
        value = data["actual_camera_randomization"]
        print(f"actual_camera_randomization: {str(value.item() if value.shape == () else value.tolist())}")
    if "action_source" in data.files:
        sources, counts = np.unique(data["action_source"], return_counts=True)
        print("action_source_counts:")
        for source, count in zip(sources, counts):
            print(f"  {source}: {int(count)}")
    if "observation_action_alignment" in data.files:
        value = data["observation_action_alignment"]
        print(f"observation_action_alignment: {str(value.item() if value.shape == () else value.tolist())}")
    else:
        print("observation_action_alignment: MISSING")
    if "success" in data.files:
        success = data["success"].astype(np.float32)
        print("success:")
        print(f"  mean={float(success.mean()):.4f}")
        print(f"  max={float(success.max()):.4f}")
    if "keypoint_site" in data.files:
        print(f"keypoint_site: {data['keypoint_site'].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
