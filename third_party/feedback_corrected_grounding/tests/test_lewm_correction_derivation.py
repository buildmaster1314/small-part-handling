from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.lewm_correction_derivation import (
    CORRECTION_ATOL_PX,
    validate_fcpg_correction_derivations,
)
from fcpg.lewm_provenance import sha256_file


class _RawRGBFakeCorrector(torch.nn.Module):
    def forward(
        self,
        rgb: torch.Tensor,
        q_nominal: torch.Tensor,
        keypoints_xyz: torch.Tensor,
        robot_obs: torch.Tensor,
    ) -> SimpleNamespace:
        del keypoints_xyz, robot_obs
        delta = rgb.mean(dim=(1, 2, 3)).reshape(-1, 1, 1)
        return SimpleNamespace(q_corrected_px=q_nominal + delta)


def _expected(rgb: np.ndarray, q_nominal: np.ndarray) -> np.ndarray:
    delta = rgb.astype(np.float32).mean(axis=(1, 2, 3)) / 255.0
    return q_nominal + delta[:, None, None]


def _fixture(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    sample_count = 5
    rgb = np.arange(sample_count * 4 * 4 * 3, dtype=np.uint8).reshape(
        sample_count, 4, 4, 3
    )
    q_nominal = np.arange(sample_count * 2 * 2, dtype=np.float32).reshape(
        sample_count, 2, 2
    )
    data_path = tmp_path / "source.npz"
    np.savez(
        data_path,
        rgb=rgb,
        q_nominal=q_nominal,
        keypoints_xyz=np.ones((sample_count, 2, 3), dtype=np.float32),
        obs=np.ones((sample_count, 12), dtype=np.float32),
        q_true=np.full((sample_count, 2, 2), 999.0, dtype=np.float32),
    )
    checkpoint = tmp_path / "corrector.pt"
    checkpoint.write_bytes(b"frozen corrector")
    cache = tmp_path / "cache"
    cache.mkdir()
    corrected_path = cache / "q_corrected.npy"
    np.save(corrected_path, _expected(rgb, q_nominal).astype(np.float32))
    metadata = {
        "schema_version": 3,
        "source_data": str(data_path),
        "source_data_sha256": sha256_file(data_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "q_corrected_sha256": sha256_file(corrected_path),
        "q_true_usage": "offline_metrics_only",
        "obs_mode": "robot",
    }
    metadata_path = cache / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    job = {
        "name": "rotation5",
        "data": str(data_path),
        "correction_cache": str(cache),
        "frame_indices": np.asarray([0, 1, 3, 4], dtype=np.int64),
    }
    return job, corrected_path, metadata_path


def _loader(
    _path: Path,
    _device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object]]:
    return _RawRGBFakeCorrector(), {"obs_mode": "robot"}


def _rewrite_q_hash(metadata_path: Path, corrected_path: Path) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["q_corrected_sha256"] = sha256_file(corrected_path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def test_raw_rgb_replay_accepts_derived_scored_corrections(
    tmp_path: Path,
) -> None:
    job, _corrected_path, _metadata_path = _fixture(tmp_path)

    audit, bound = validate_fcpg_correction_derivations(
        [job],
        device="cpu",
        batch_size=2,
        require_schema3=True,
        model_loader=_loader,
    )

    source = audit["sources"]["rotation5"]
    assert audit["passed"] is True
    assert source["scored_frames"] == 4
    assert source["oracle_fields_consumed"] == []
    assert source["consumed_fields"] == [
        "rgb",
        "q_nominal",
        "keypoints_xyz",
        "obs",
    ]
    assert source["max_abs_error_px"] <= CORRECTION_ATOL_PX
    assert Path(job["data"]).resolve() in bound



def test_raw_rgb_replay_emits_canonical_scoring_coordinates(
    tmp_path: Path,
) -> None:
    job, _corrected_path, _metadata_path = _fixture(tmp_path)
    emitted: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    audit, _bound = validate_fcpg_correction_derivations(
        [job],
        device="cpu",
        batch_size=2,
        require_schema3=True,
        model_loader=_loader,
        canonical_sink=lambda name, indices, values: emitted.setdefault(
            name, (indices, values)
        ),
    )

    indices, values = emitted["rotation5"]
    assert np.array_equal(indices, job["frame_indices"])
    with np.load(job["data"], allow_pickle=False) as raw:
        expected = _expected(raw["rgb"][indices], raw["q_nominal"][indices])
    assert np.allclose(values, expected, rtol=0.0, atol=2e-6)
    assert audit["sources"]["rotation5"]["canonical_scoring_output_emitted"] is True

def test_raw_rgb_replay_rejects_oracle_injected_cache_even_with_new_hash(
    tmp_path: Path,
) -> None:
    job, corrected_path, metadata_path = _fixture(tmp_path)
    corrected = np.load(corrected_path)
    corrected[1] += 0.5
    np.save(corrected_path, corrected)
    _rewrite_q_hash(metadata_path, corrected_path)

    with pytest.raises(ValueError, match="not derived from raw RGB"):
        validate_fcpg_correction_derivations(
            [job],
            device="cpu",
            require_schema3=True,
            model_loader=_loader,
        )


def test_raw_rgb_replay_rejects_changed_rgb_even_with_new_source_hash(
    tmp_path: Path,
) -> None:
    job, _corrected_path, metadata_path = _fixture(tmp_path)
    data_path = Path(job["data"])
    with np.load(data_path, allow_pickle=False) as raw:
        values = {name: np.asarray(raw[name]) for name in raw.files}
    values["rgb"] = values["rgb"].copy()
    values["rgb"][0] = 0
    np.savez(data_path, **values)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["source_data_sha256"] = sha256_file(data_path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="not derived from raw RGB"):
        validate_fcpg_correction_derivations(
            [job],
            device="cpu",
            require_schema3=True,
            model_loader=_loader,
        )


def test_raw_rgb_replay_ignores_unscored_cache_rows(
    tmp_path: Path,
) -> None:
    job, corrected_path, metadata_path = _fixture(tmp_path)
    corrected = np.load(corrected_path)
    corrected[2] += 10.0
    np.save(corrected_path, corrected)
    _rewrite_q_hash(metadata_path, corrected_path)

    audit, _bound = validate_fcpg_correction_derivations(
        [job],
        device="cpu",
        require_schema3=True,
        model_loader=_loader,
    )

    assert audit["passed"] is True


def test_raw_rgb_replay_rejects_checkpoint_metadata_rewrite(
    tmp_path: Path,
) -> None:
    job, _corrected_path, metadata_path = _fixture(tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["checkpoint_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint hash"):
        validate_fcpg_correction_derivations(
            [job],
            device="cpu",
            require_schema3=True,
            model_loader=_loader,
        )


def test_raw_rgb_replay_rejects_full_state_corrector(
    tmp_path: Path,
) -> None:
    job, _corrected_path, metadata_path = _fixture(tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["obs_mode"] = "full"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    def full_state_loader(
        _path: Path,
        _device: torch.device,
    ) -> tuple[torch.nn.Module, dict[str, object]]:
        return _RawRGBFakeCorrector(), {"obs_mode": "full"}

    with pytest.raises(ValueError, match="robot-only"):
        validate_fcpg_correction_derivations(
            [job],
            device="cpu",
            require_schema3=True,
            model_loader=full_state_loader,
        )


def test_raw_rgb_replay_rejects_nonfinite_scored_cache(
    tmp_path: Path,
) -> None:
    job, corrected_path, metadata_path = _fixture(tmp_path)
    corrected = np.load(corrected_path)
    corrected[1, 0, 0] = np.nan
    np.save(corrected_path, corrected)
    _rewrite_q_hash(metadata_path, corrected_path)

    with pytest.raises(ValueError, match="replayed correction is invalid"):
        validate_fcpg_correction_derivations(
            [job],
            device="cpu",
            require_schema3=True,
            model_loader=_loader,
        )
