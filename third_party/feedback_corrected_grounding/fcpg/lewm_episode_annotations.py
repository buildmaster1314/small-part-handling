"""Episode-level camera and corrector-exposure annotations for LeWM dev gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


EpisodeKey = tuple[int, str, int]


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence")
    return value


def _resolve(path: object, *, base: Path, label: str) -> Path:
    if not isinstance(path, str) or not path:
        raise TypeError(f"{label} must be a non-empty path string")
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def _episode_conditions(
    *,
    source_index: int,
    data_path: Path,
    validation_labels: set[str],
) -> dict[EpisodeKey, str]:
    required = {
        "task_name",
        "episode_id",
        "actual_camera_rotation_deg",
        "actual_camera_translation_xyz",
    }
    with np.load(data_path, allow_pickle=False) as raw:
        missing = required.difference(raw.files)
        if missing:
            raise KeyError(f"camera annotation data is missing fields: {sorted(missing)}")
        tasks = np.asarray(raw["task_name"]).astype(str).reshape(-1)
        episodes = np.asarray(raw["episode_id"]).reshape(-1)
        rotations = np.asarray(raw["actual_camera_rotation_deg"], dtype=np.float64).reshape(-1)
        translations = np.asarray(
            raw["actual_camera_translation_xyz"], dtype=np.float64
        )
    if translations.ndim != 2 or translations.shape[1] != 3:
        raise ValueError("actual_camera_translation_xyz must have shape [N, 3]")
    count = tasks.size
    if not (episodes.size == rotations.size == translations.shape[0] == count):
        raise ValueError("camera annotation arrays must have equal first dimensions")
    if not bool(np.isfinite(rotations).all() and np.isfinite(translations).all()):
        raise ValueError("camera annotation arrays contain non-finite values")

    annotations: dict[EpisodeKey, str] = {}
    observed_labels: set[str] = set()
    for index in range(count):
        task = str(tasks[index])
        episode = int(episodes[index])
        label = f"{task}:{episode}"
        if label not in validation_labels:
            continue
        observed_labels.add(label)
        clean = bool(
            math.isclose(float(rotations[index]), 0.0, abs_tol=1e-12)
            and np.allclose(translations[index], 0.0, rtol=0.0, atol=1e-12)
        )
        condition = "clean" if clean else "mild"
        key = (source_index, task, episode)
        previous = annotations.setdefault(key, condition)
        if previous != condition:
            raise ValueError(f"camera condition changes within episode {label}")
    missing_labels = validation_labels.difference(observed_labels)
    if missing_labels:
        raise ValueError(
            "validation manifest references missing camera episodes: "
            f"{sorted(missing_labels)[:5]}"
        )
    return annotations


def load_validation_episode_annotations(
    data_config: Mapping[str, Any],
    *,
    episode_manifest_path: str | Path,
    corrector_slice_path: str | Path,
    repo_root: str | Path,
    expected_counts: Mapping[str, Any] | None = None,
) -> dict[EpisodeKey, dict[str, Any]]:
    """Build auditable clean/mild and corrector-gradient-exposure labels."""

    root = Path(repo_root).resolve()
    manifest_path = _resolve(
        str(episode_manifest_path), base=root, label="episode_manifest_path"
    )
    slice_path = _resolve(
        str(corrector_slice_path), base=root, label="corrector_slice_path"
    )
    manifest = _load_json(manifest_path, "episode manifest")
    if manifest.get("protocol") != "lewm_gate_l1_camera_aware_episode_split":
        raise ValueError("unexpected episode manifest protocol")
    manifest_sources = _mapping(manifest.get("sources"), "episode manifest sources")
    sources = _sequence(data_config.get("sources"), "checkpoint data sources")
    annotations: dict[EpisodeKey, dict[str, Any]] = {}
    for source_index, raw_source in enumerate(sources):
        source = _mapping(raw_source, "checkpoint source")
        name = source.get("name")
        if not isinstance(name, str) or name not in manifest_sources:
            raise ValueError(f"checkpoint source has no matching manifest entry: {name!r}")
        declared = _mapping(manifest_sources[name], f"manifest source {name}")
        labels = {str(item) for item in _sequence(declared.get("val_episodes"), "val_episodes")}
        if not labels:
            raise ValueError(f"manifest source {name} has no validation episodes")
        data_path = _resolve(source.get("path"), base=root, label=f"source {name} path")
        declared_path = _resolve(
            declared.get("path"), base=root, label=f"manifest source {name} path"
        )
        if data_path != declared_path:
            raise ValueError(f"checkpoint and manifest paths differ for source {name}")
        for key, condition in _episode_conditions(
            source_index=source_index,
            data_path=data_path,
            validation_labels=labels,
        ).items():
            if key in annotations:
                raise ValueError(f"duplicate validation episode annotation: {key}")
            annotations[key] = {
                "camera_condition": condition,
                "corrector_train_unseen": False,
            }

    corrector_slice = _load_json(slice_path, "corrector exposure slice")
    if (
        corrector_slice.get("protocol")
        != "fcpg_anchor_visual_corrector_train_unseen_slice_v1"
    ):
        raise ValueError("unexpected corrector exposure slice protocol")
    if corrector_slice.get("lewm_validation_manifest_sha256") != data_config.get(
        "episode_manifest", {}
    ).get("sha256"):
        raise ValueError("corrector slice is bound to a different LeWM split")
    slice_tasks = _mapping(corrector_slice.get("tasks"), "corrector slice tasks")
    selected = 0
    condition_counts = {"clean": 0, "mild": 0}
    by_task_name = {(key[1], key[2]): key for key in annotations}
    for task, raw_conditions in slice_tasks.items():
        conditions = _mapping(raw_conditions, f"corrector slice task {task}")
        for condition in ("clean", "mild"):
            for raw_episode in _sequence(
                conditions.get(condition), f"corrector slice {task}/{condition}"
            ):
                episode = int(raw_episode)
                key = by_task_name.get((str(task), episode))
                if key is None:
                    raise ValueError(
                        f"corrector slice episode is absent from validation: {task}:{episode}"
                    )
                if annotations[key]["camera_condition"] != condition:
                    raise ValueError(
                        f"corrector slice condition mismatch for {task}:{episode}"
                    )
                if annotations[key]["corrector_train_unseen"]:
                    raise ValueError(f"duplicate corrector slice episode: {task}:{episode}")
                annotations[key]["corrector_train_unseen"] = True
                selected += 1
                condition_counts[condition] += 1
    if selected != int(corrector_slice.get("episode_count", -1)):
        raise ValueError("corrector slice episode_count does not match its entries")

    actual_counts = {
        "clean": sum(
            row["camera_condition"] == "clean" for row in annotations.values()
        ),
        "mild": sum(
            row["camera_condition"] == "mild" for row in annotations.values()
        ),
        "total": len(annotations),
    }
    if expected_counts is not None:
        expected = {name: int(expected_counts[name]) for name in actual_counts}
        if actual_counts != expected:
            raise ValueError(
                f"validation condition counts differ: expected {expected}, got {actual_counts}"
            )
    return annotations


def annotate_episode_blocks(
    blocks: Sequence[Mapping[str, Any]],
    annotations: Mapping[EpisodeKey, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach immutable episode labels and require exact set coverage."""

    output: list[dict[str, Any]] = []
    seen: set[EpisodeKey] = set()
    for raw_block in blocks:
        block = dict(_mapping(raw_block, "episode block"))
        key = (
            int(block.get("source_index")),
            str(block.get("task_name")),
            int(block.get("episode_id")),
        )
        if key in seen:
            raise ValueError(f"duplicate episode block: {key}")
        if key not in annotations:
            raise ValueError(f"episode block has no annotation: {key}")
        seen.add(key)
        block.update(dict(annotations[key]))
        output.append(block)
    if seen != set(annotations):
        missing = sorted(set(annotations).difference(seen))
        raise ValueError(f"evaluation omitted annotated episodes: {missing[:5]}")
    return output


__all__ = [
    "EpisodeKey",
    "annotate_episode_blocks",
    "load_validation_episode_annotations",
]
