from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fcpg.lewm_episode_annotations import (
    annotate_episode_blocks,
    load_validation_episode_annotations,
)


_MANIFEST_PROTOCOL = "lewm_gate_l1_camera_aware_episode_split"
_SLICE_PROTOCOL = "fcpg_anchor_visual_corrector_train_unseen_slice_v1"
_SPLIT_SHA256 = "tiny-validation-manifest-sha256"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_inputs(tmp_path: Path) -> tuple[dict[str, Any], Path, Path]:
    data_path = tmp_path / "camera.npz"
    np.savez(
        data_path,
        task_name=np.asarray(
            ["pick", "pick", "pick", "push", "push", "push"]
        ),
        episode_id=np.asarray([10, 10, 11, 20, 21, 21], dtype=np.int32),
        actual_camera_rotation_deg=np.asarray(
            [0.0, 0.0, 5.0, 0.0, 0.0, 0.0], dtype=np.float64
        ),
        actual_camera_translation_xyz=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.01, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )

    manifest_path = tmp_path / "episodes.json"
    _write_json(
        manifest_path,
        {
            "protocol": _MANIFEST_PROTOCOL,
            "sources": {
                "tiny": {
                    "path": data_path.name,
                    "val_episodes": [
                        "pick:10",
                        "pick:11",
                        "push:20",
                        "push:21",
                    ],
                }
            },
        },
    )

    slice_path = tmp_path / "corrector-slice.json"
    _write_json(
        slice_path,
        {
            "protocol": _SLICE_PROTOCOL,
            "lewm_validation_manifest_sha256": _SPLIT_SHA256,
            "episode_count": 2,
            "tasks": {
                "pick": {"clean": [10], "mild": []},
                "push": {"clean": [], "mild": [21]},
            },
        },
    )

    data_config = {
        "sources": [{"name": "tiny", "path": data_path.name}],
        "episode_manifest": {"sha256": _SPLIT_SHA256},
    }
    return data_config, manifest_path, slice_path


def _load(
    tmp_path: Path,
    data_config: dict[str, Any],
    manifest_path: Path,
    slice_path: Path,
    *,
    expected_counts: dict[str, int] | None = None,
) -> dict[tuple[int, str, int], dict[str, Any]]:
    return load_validation_episode_annotations(
        data_config,
        episode_manifest_path=manifest_path.name,
        corrector_slice_path=slice_path.name,
        repo_root=tmp_path,
        expected_counts=expected_counts,
    )


def test_load_classifies_counts_and_marks_corrector_train_unseen(
    tmp_path: Path,
) -> None:
    data_config, manifest_path, slice_path = _write_inputs(tmp_path)

    annotations = _load(
        tmp_path,
        data_config,
        manifest_path,
        slice_path,
        expected_counts={"clean": 2, "mild": 2, "total": 4},
    )

    assert annotations == {
        (0, "pick", 10): {
            "camera_condition": "clean",
            "corrector_train_unseen": True,
        },
        (0, "pick", 11): {
            "camera_condition": "mild",
            "corrector_train_unseen": False,
        },
        (0, "push", 20): {
            "camera_condition": "clean",
            "corrector_train_unseen": False,
        },
        (0, "push", 21): {
            "camera_condition": "mild",
            "corrector_train_unseen": True,
        },
    }


def test_load_rejects_incomplete_expected_counts(tmp_path: Path) -> None:
    data_config, manifest_path, slice_path = _write_inputs(tmp_path)

    with pytest.raises(ValueError, match="validation condition counts differ"):
        _load(
            tmp_path,
            data_config,
            manifest_path,
            slice_path,
            expected_counts={"clean": 2, "mild": 2, "total": 5},
        )


@pytest.mark.parametrize(
    ("declared_path", "error", "message"),
    [
        pytest.param(
            None,
            TypeError,
            "manifest source tiny path must be a non-empty path string",
            id="missing",
        ),
        pytest.param(
            "different-camera.npz",
            ValueError,
            "checkpoint and manifest paths differ",
            id="different",
        ),
    ],
)
def test_load_requires_matching_manifest_source_path(
    tmp_path: Path,
    declared_path: str | None,
    error: type[Exception],
    message: str,
) -> None:
    data_config, manifest_path, slice_path = _write_inputs(tmp_path)
    manifest = _read_json(manifest_path)
    if declared_path is None:
        manifest["sources"]["tiny"].pop("path")
    else:
        manifest["sources"]["tiny"]["path"] = declared_path
    _write_json(manifest_path, manifest)

    with pytest.raises(error, match=message):
        _load(tmp_path, data_config, manifest_path, slice_path)


def test_load_rejects_manifest_episode_missing_from_npz(tmp_path: Path) -> None:
    data_config, manifest_path, slice_path = _write_inputs(tmp_path)
    manifest = _read_json(manifest_path)
    manifest["sources"]["tiny"]["val_episodes"].append("pick:999")
    _write_json(manifest_path, manifest)

    with pytest.raises(
        ValueError, match="validation manifest references missing camera episodes"
    ):
        _load(tmp_path, data_config, manifest_path, slice_path)


def test_load_rejects_corrector_slice_condition_mismatch(tmp_path: Path) -> None:
    data_config, manifest_path, slice_path = _write_inputs(tmp_path)
    corrector_slice = _read_json(slice_path)
    corrector_slice["episode_count"] = 1
    corrector_slice["tasks"] = {
        "pick": {"clean": [], "mild": [10]},
    }
    _write_json(slice_path, corrector_slice)

    with pytest.raises(ValueError, match="corrector slice condition mismatch for pick:10"):
        _load(tmp_path, data_config, manifest_path, slice_path)


def test_annotate_episode_blocks_requires_exact_unique_coverage(
    tmp_path: Path,
) -> None:
    data_config, manifest_path, slice_path = _write_inputs(tmp_path)
    annotations = _load(tmp_path, data_config, manifest_path, slice_path)
    blocks = [
        {
            "source_index": source_index,
            "task_name": task_name,
            "episode_id": episode_id,
            "num_sequences": index + 1,
        }
        for index, (source_index, task_name, episode_id) in enumerate(
            sorted(annotations)
        )
    ]

    output = annotate_episode_blocks(blocks, annotations)

    assert [row["camera_condition"] for row in output] == [
        "clean",
        "mild",
        "clean",
        "mild",
    ]
    assert [row["corrector_train_unseen"] for row in output] == [
        True,
        False,
        False,
        True,
    ]
    assert "camera_condition" not in blocks[0]

    with pytest.raises(ValueError, match="duplicate episode block"):
        annotate_episode_blocks([*blocks, dict(blocks[0])], annotations)

    with pytest.raises(ValueError, match="evaluation omitted annotated episodes"):
        annotate_episode_blocks(blocks[:-1], annotations)
