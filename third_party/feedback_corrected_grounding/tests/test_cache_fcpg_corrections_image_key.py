from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.lewm_provenance import sha256_file
from scripts import cache_fcpg_corrections as cache_corrections


class _FakeAdapter:
    def forward_from_backbone_features(
        self,
        _feature_maps: torch.Tensor,
        _prefix: torch.Tensor,
        q_nominal: torch.Tensor,
        _keypoints_xyz: torch.Tensor,
        _obs: torch.Tensor,
        *,
        image_size: tuple[int, int],
    ) -> SimpleNamespace:
        assert image_size == (4, 2)
        return SimpleNamespace(q_corrected_px=q_nominal + 0.5, reliability=None)


def _content_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _write_dataset(
    path: Path,
    *,
    rgb: np.ndarray,
    rgb_camera_only: np.ndarray | None,
    sample_count: int | None = None,
) -> None:
    count = int(rgb.shape[0] if sample_count is None else sample_count)
    q_nominal = np.zeros((count, 2, 2), dtype=np.float32)
    arrays: dict[str, np.ndarray] = {
        "rgb": rgb,
        "q_nominal": q_nominal,
        "q_true": q_nominal + 1.0,
        "keypoints_xyz": np.zeros((count, 2, 3), dtype=np.float32),
        "valid_true": np.ones((count, 2), dtype=bool),
        "valid_nominal": np.ones((count, 2), dtype=bool),
        "obs": np.zeros((count, 4), dtype=np.float32),
        "episode_id": np.zeros(count, dtype=np.int64),
        "frame_id": np.arange(count, dtype=np.int64),
    }
    if rgb_camera_only is not None:
        arrays["rgb_camera_only"] = rgb_camera_only
    np.savez(path, **arrays)


def _write_vision_cache(
    path: Path,
    *,
    data_path: Path,
    selected: np.ndarray,
    image_key: str,
    metadata_updates: dict[str, object] | None = None,
) -> None:
    path.mkdir()
    with np.load(data_path, allow_pickle=False) as data:
        sample_count = int(data["q_nominal"].shape[0])
    features = np.zeros((sample_count, 1, 2, 1, 1), dtype=np.float16)
    prefix = np.zeros((sample_count, 2), dtype=np.float16)
    feature_path = path / "backbone_features.npy"
    prefix_path = path / "prefix.npy"
    np.save(feature_path, features)
    np.save(prefix_path, prefix)
    view_role = (
        "deployment_observation" if image_key == "rgb" else "target_only"
    )
    metadata: dict[str, object] = {
        "source_data": str(data_path.resolve()),
        "source_data_sha256": sha256_file(data_path),
        "image_key": image_key,
        "selected_image_sha256": _content_sha256(selected),
        "view_role": view_role,
        "deployment_cache_eligible": view_role == "deployment_observation",
        "sample_count": int(selected.shape[0]),
        "image_width": 4,
        "image_height": 2,
        "selected_layers": [11],
        "backbone_features_sha256": sha256_file(feature_path),
        "prefix_sha256": sha256_file(prefix_path),
    }
    if metadata_updates:
        metadata.update(metadata_updates)
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _run_correction(
    *,
    data_path: Path,
    vision_cache: Path,
    checkpoint: Path,
    output: Path,
    image_key: str | None = None,
) -> int:
    argv = [
        "--data",
        str(data_path),
        "--vision-cache",
        str(vision_cache),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output),
        "--batch-size",
        "2",
        "--device",
        "cpu",
    ]
    if image_key is not None:
        argv.extend(["--image-key", image_key])
    return cache_corrections.main(argv)


@pytest.fixture
def fake_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    checkpoint = tmp_path / "corrector.pt"
    checkpoint.write_bytes(b"corrector")
    monkeypatch.setattr(
        cache_corrections,
        "load_fcpg_vit_adapter",
        lambda *_args, **_kwargs: (
            _FakeAdapter(),
            {
                "vision_selected_layers": [11],
                "obs_mode": "robot",
                "schema_version": 2,
                "corrector_arch": "fake",
            },
        ),
    )
    return checkpoint


def test_default_rgb_is_compatible_and_image_arrays_split_lineage(
    tmp_path: Path,
    fake_checkpoint: Path,
) -> None:
    rgb = np.zeros((3, 2, 4, 3), dtype=np.uint8)
    rgb_camera_only = np.full_like(rgb, 17)
    data_path = tmp_path / "images.npz"
    _write_dataset(
        data_path, rgb=rgb, rgb_camera_only=rgb_camera_only
    )
    rgb_vision = tmp_path / "vision_rgb"
    target_vision = tmp_path / "vision_target"
    _write_vision_cache(
        rgb_vision, data_path=data_path, selected=rgb, image_key="rgb"
    )
    _write_vision_cache(
        target_vision,
        data_path=data_path,
        selected=rgb_camera_only,
        image_key="rgb_camera_only",
    )

    rgb_output = tmp_path / "correction_rgb"
    target_output = tmp_path / "correction_target"
    assert (
        _run_correction(
            data_path=data_path,
            vision_cache=rgb_vision,
            checkpoint=fake_checkpoint,
            output=rgb_output,
        )
        == 0
    )
    assert (
        _run_correction(
            data_path=data_path,
            vision_cache=target_vision,
            checkpoint=fake_checkpoint,
            output=target_output,
            image_key="rgb_camera_only",
        )
        == 0
    )

    rgb_metadata = json.loads(
        (rgb_output / "metadata.json").read_text(encoding="utf-8")
    )
    target_metadata = json.loads(
        (target_output / "metadata.json").read_text(encoding="utf-8")
    )
    assert rgb_metadata["image_key"] == "rgb"
    assert rgb_metadata["view_role"] == "deployment_observation"
    assert rgb_metadata["deployment_cache_eligible"] is True
    assert rgb_metadata["vision_image_lineage_verified"] is True
    assert rgb_metadata["selected_image_sha256"] == _content_sha256(rgb)
    assert target_metadata["image_key"] == "rgb_camera_only"
    assert target_metadata["view_role"] == "target_only"
    assert target_metadata["deployment_cache_eligible"] is False
    assert target_metadata["vision_image_lineage_verified"] is True
    assert target_metadata["selected_image_sha256"] == _content_sha256(
        rgb_camera_only
    )
    assert (
        rgb_metadata["source_data_sha256"]
        == target_metadata["source_data_sha256"]
        == sha256_file(data_path)
    )
    assert (
        rgb_metadata["selected_image_sha256"]
        != target_metadata["selected_image_sha256"]
    )
    expected = np.full((3, 2, 2), 0.5, dtype=np.float32)
    np.testing.assert_array_equal(
        np.load(rgb_output / "q_corrected.npy", allow_pickle=False), expected
    )


def test_default_rgb_accepts_legacy_vision_metadata_but_target_requires_lineage(
    tmp_path: Path,
    fake_checkpoint: Path,
) -> None:
    rgb = np.full((3, 2, 4, 3), 9, dtype=np.uint8)
    data_path = tmp_path / "images.npz"
    _write_dataset(data_path, rgb=rgb, rgb_camera_only=np.full_like(rgb, 11))
    vision = tmp_path / "legacy_vision"
    _write_vision_cache(
        vision, data_path=data_path, selected=rgb, image_key="rgb"
    )
    metadata_path = vision / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for field in ("image_key", "selected_image_sha256"):
        metadata.pop(field)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    output = tmp_path / "legacy_correction"
    assert (
        _run_correction(
            data_path=data_path,
            vision_cache=vision,
            checkpoint=fake_checkpoint,
            output=output,
        )
        == 0
    )
    correction_metadata = json.loads(
        (output / "metadata.json").read_text(encoding="utf-8")
    )
    assert correction_metadata["image_key"] == "rgb"
    assert correction_metadata["selected_image_sha256"] == _content_sha256(rgb)
    assert correction_metadata["view_role"] == "deployment_observation"
    assert correction_metadata["deployment_cache_eligible"] is True
    assert correction_metadata["vision_image_lineage_verified"] is False

    with pytest.raises(ValueError, match="explicit image lineage"):
        _run_correction(
            data_path=data_path,
            vision_cache=vision,
            checkpoint=fake_checkpoint,
            output=tmp_path / "target_correction",
            image_key="rgb_camera_only",
        )


@pytest.mark.parametrize(
    ("image_key", "metadata_updates", "message"),
    [
        ("rgb_camera_only", {}, "image_key"),
        ("rgb", {"view_role": "target_only"}, "view_role"),
        (
            "rgb",
            {"deployment_cache_eligible": False},
            "deployment_cache_eligible",
        ),
        ("rgb", {"selected_image_sha256": "0" * 64}, "selected_image_sha256"),
    ],
)
def test_correction_cache_rejects_mismatched_vision_lineage(
    tmp_path: Path,
    fake_checkpoint: Path,
    image_key: str,
    metadata_updates: dict[str, object],
    message: str,
) -> None:
    rgb = np.zeros((3, 2, 4, 3), dtype=np.uint8)
    data_path = tmp_path / "images.npz"
    _write_dataset(data_path, rgb=rgb, rgb_camera_only=np.ones_like(rgb))
    vision = tmp_path / "vision"
    _write_vision_cache(
        vision,
        data_path=data_path,
        selected=rgb,
        image_key="rgb",
        metadata_updates=metadata_updates,
    )

    with pytest.raises(ValueError, match=message):
        _run_correction(
            data_path=data_path,
            vision_cache=vision,
            checkpoint=fake_checkpoint,
            output=tmp_path / "correction",
            image_key=image_key,
        )


@pytest.mark.parametrize(
    ("selected", "sample_count", "message"),
    [
        (np.zeros((3, 2, 4, 1), dtype=np.uint8), 3, "shape"),
        (np.zeros((3, 2, 4, 3), dtype=np.float32), 3, "dtype uint8"),
        (np.zeros((2, 2, 4, 3), dtype=np.uint8), 3, "samples"),
    ],
)
def test_correction_cache_rejects_invalid_selected_image_array(
    tmp_path: Path,
    fake_checkpoint: Path,
    selected: np.ndarray,
    sample_count: int,
    message: str,
) -> None:
    rgb = np.zeros((sample_count, 2, 4, 3), dtype=np.uint8)
    data_path = tmp_path / "images.npz"
    _write_dataset(
        data_path,
        rgb=rgb,
        rgb_camera_only=selected,
        sample_count=sample_count,
    )
    vision = tmp_path / "vision"
    _write_vision_cache(
        vision,
        data_path=data_path,
        selected=selected,
        image_key="rgb_camera_only",
    )

    with pytest.raises(ValueError, match=message):
        _run_correction(
            data_path=data_path,
            vision_cache=vision,
            checkpoint=fake_checkpoint,
            output=tmp_path / "correction",
            image_key="rgb_camera_only",
        )
