from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import cache_vit_backbone_features as cache_vit


class _FakeVisionEncoder:
    last_kwargs: dict[str, object] | None = None
    loaded_state: dict[str, torch.Tensor] | None = None

    def __init__(self, **_kwargs: object) -> None:
        type(self).last_kwargs = dict(_kwargs)

    def load_state_dict(
        self, state: dict[str, torch.Tensor], *, strict: bool
    ) -> object:
        assert strict is True
        type(self).loaded_state = dict(state)
        return object()

    def to(self, _device: torch.device) -> _FakeVisionEncoder:
        return self

    def eval(self) -> _FakeVisionEncoder:
        return self

    def extract_backbone_features(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return image[:, None], image.mean(dim=(2, 3))


def _content_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _write_dataset(
    path: Path,
    *,
    rgb: np.ndarray,
    rgb_camera_only: np.ndarray | None = None,
    sample_count: int | None = None,
) -> None:
    count = int(rgb.shape[0] if sample_count is None else sample_count)
    arrays: dict[str, np.ndarray] = {
        "rgb": rgb,
        "q_nominal": np.zeros((count, 2, 2), dtype=np.float32),
        "episode_id": np.zeros(count, dtype=np.int64),
        "frame_id": np.arange(count, dtype=np.int64),
    }
    if rgb_camera_only is not None:
        arrays["rgb_camera_only"] = rgb_camera_only
    np.savez(path, **arrays)


@pytest.mark.parametrize(
    ("image_key", "view_role", "deployment_eligible"),
    [
        ("rgb", "deployment_observation", True),
        ("rgb_camera_only", "target_only", False),
    ],
)
def test_vit_cache_uses_selected_image_and_records_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image_key: str,
    view_role: str,
    deployment_eligible: bool,
) -> None:
    rgb = np.zeros((3, 2, 4, 3), dtype=np.uint8)
    rgb_camera_only = np.full_like(rgb, 255)
    data_path = tmp_path / "images.npz"
    _write_dataset(data_path, rgb=rgb, rgb_camera_only=rgb_camera_only)
    output = tmp_path / image_key
    monkeypatch.setattr(cache_vit, "ViTDenseEncoder", _FakeVisionEncoder)

    argv = [
        "--data",
        str(data_path),
        "--output-dir",
        str(output),
        "--batch-size",
        "2",
        "--device",
        "cpu",
    ]
    if image_key != "rgb":
        argv.extend(["--image-key", image_key])
    assert cache_vit.main(argv) == 0

    selected = rgb if image_key == "rgb" else rgb_camera_only
    features = np.load(output / "backbone_features.npy", allow_pickle=False)
    expected = selected.transpose(0, 3, 1, 2)[:, None].astype(np.float16) / 255.0
    np.testing.assert_array_equal(features, expected)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["image_key"] == image_key
    assert metadata["view_role"] == view_role
    assert metadata["deployment_cache_eligible"] is deployment_eligible
    assert metadata["selected_image_sha256"] == _content_sha256(selected)
    assert metadata["sample_count"] == selected.shape[0]


def test_vit_cache_rejects_missing_and_unsupported_image_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_path = tmp_path / "images.npz"
    _write_dataset(data_path, rgb=np.zeros((2, 2, 2, 3), dtype=np.uint8))
    monkeypatch.setattr(cache_vit, "ViTDenseEncoder", _FakeVisionEncoder)

    with pytest.raises(KeyError, match="rgb_camera_only"):
        cache_vit.main(
            [
                "--data",
                str(data_path),
                "--output-dir",
                str(tmp_path / "missing"),
                "--image-key",
                "rgb_camera_only",
                "--device",
                "cpu",
            ]
        )

    with pytest.raises(SystemExit):
        cache_vit.main(
            [
                "--data",
                str(data_path),
                "--output-dir",
                str(tmp_path / "unsupported"),
                "--image-key",
                "q_true",
                "--device",
                "cpu",
            ]
        )


def test_vit_cache_can_load_embedded_checkpoint_weights_without_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_path = tmp_path / "images.npz"
    _write_dataset(data_path, rgb=np.zeros((2, 2, 2, 3), dtype=np.uint8))
    checkpoint = tmp_path / "fcpg.pt"
    torch.save(
        {"model_state": {"vision.backbone.weight": torch.ones(1), "other": 3}},
        checkpoint,
    )
    _FakeVisionEncoder.last_kwargs = None
    _FakeVisionEncoder.loaded_state = None
    monkeypatch.setattr(cache_vit, "ViTDenseEncoder", _FakeVisionEncoder)

    assert cache_vit.main(
        [
            "--data",
            str(data_path),
            "--output-dir",
            str(tmp_path / "cache"),
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
        ]
    ) == 0
    assert _FakeVisionEncoder.last_kwargs is not None
    assert _FakeVisionEncoder.last_kwargs["pretrained"] is False
    assert _FakeVisionEncoder.loaded_state is not None
    assert set(_FakeVisionEncoder.loaded_state) == {"backbone.weight"}
    metadata = json.loads((tmp_path / "cache" / "metadata.json").read_text())
    assert metadata["weight_source"] == str(checkpoint.resolve())


@pytest.mark.parametrize(
    ("selected", "sample_count", "message"),
    [
        (np.zeros((2, 2, 2, 1), dtype=np.uint8), 2, "shape"),
        (np.zeros((2, 2, 2, 3), dtype=np.float32), 2, "dtype uint8"),
        (np.zeros((2, 2, 2, 3), dtype=np.uint8), 3, "q_nominal"),
    ],
)
def test_vit_cache_rejects_invalid_image_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected: np.ndarray,
    sample_count: int,
    message: str,
) -> None:
    data_path = tmp_path / "invalid.npz"
    _write_dataset(data_path, rgb=selected, sample_count=sample_count)
    monkeypatch.setattr(cache_vit, "ViTDenseEncoder", _FakeVisionEncoder)

    with pytest.raises(ValueError, match=message):
        cache_vit.main(
            [
                "--data",
                str(data_path),
                "--output-dir",
                str(tmp_path / "cache"),
                "--device",
                "cpu",
            ]
        )
