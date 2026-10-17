from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.lewm_anchor_persistent import ANCHOR_PERSISTENT_CHECKPOINT_TYPE
from fcpg.lewm_provenance import sha256_file
from fcpg.lewm_sequence_data import frame_identity_sha256
from scripts import build_lewm_posterior_gate_cache as builder


FRAME_COUNT = 45
ANCHORS = 2
LATENT_DIM = 3


@dataclass
class _Artifacts:
    data: Path
    observed_correction: Path
    canonical_correction: Path
    checkpoint_path: Path
    output_dir: Path
    action: np.ndarray
    checkpoint: dict[str, object]
    model: torch.nn.Module


class _FakeAnchorModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.history_size = 3
        self.num_keypoints = ANCHORS
        self.latent_dim = LATENT_DIM
        self.action_dim = 20
        self.frame_encoder = SimpleNamespace(obs_dim=3)

    def encode_frames(self, _batch: object) -> None:
        raise AssertionError("mock encoder boundary should be used")

    def predict(self, _history: object, _action: object) -> None:
        raise AssertionError("mock prior boundary should be used")


def _write_dataset(path: Path) -> np.ndarray:
    rgb = np.zeros((FRAME_COUNT, 4, 5, 3), dtype=np.uint8)
    for frame in range(FRAME_COUNT):
        rgb[frame] = frame
    canonical = rgb + np.uint8(50)
    action = np.arange(FRAME_COUNT * 4, dtype=np.float32).reshape(FRAME_COUNT, 4)
    q_nominal = np.empty((FRAME_COUNT, ANCHORS, 2), dtype=np.float32)
    q_nominal[:, 0] = [1.0, 1.0]
    q_nominal[:, 1] = [3.75, 1.0]
    dropout = np.zeros((FRAME_COUNT, ANCHORS), dtype=np.bool_)
    dropout[20:35, 1] = True
    recovery = np.zeros_like(dropout)
    recovery[35:45, 1] = True
    recovery_onset = np.zeros_like(dropout)
    recovery_onset[35, 1] = True
    measurement_valid = np.ones_like(dropout)
    measurement_valid[dropout] = False
    np.savez_compressed(
        path,
        rgb=rgb,
        rgb_camera_only=canonical,
        obs=np.arange(FRAME_COUNT * 3, dtype=np.float32).reshape(FRAME_COUNT, 3),
        action=action,
        task_name=np.asarray(["pick-place-v3"] * FRAME_COUNT),
        task_index=np.full(FRAME_COUNT, 10001, dtype=np.int64),
        episode_id=np.full(FRAME_COUNT, 7, dtype=np.int64),
        env_seed=np.full(FRAME_COUNT, 1200107, dtype=np.int64),
        frame_id=np.arange(FRAME_COUNT, dtype=np.int64),
        task_generation_seed=np.asarray(12001, dtype=np.int64),
        keypoints_xyz=np.zeros((FRAME_COUNT, ANCHORS, 3), dtype=np.float32),
        q_true=q_nominal + 0.2,
        q_nominal=q_nominal,
        valid_true=np.ones((FRAME_COUNT, ANCHORS), dtype=np.bool_),
        valid_nominal=np.ones((FRAME_COUNT, ANCHORS), dtype=np.bool_),
        anchor_dropout_active=dropout,
        anchor_recovery_onset=recovery_onset,
        anchor_recovery_segment=recovery,
        measurement_valid=measurement_valid,
        camera_step_frame=np.full(FRAME_COUNT, 5, dtype=np.int64),
        observation_action_alignment=np.asarray("pre_action"),
    )
    return action


def _write_correction_cache(
    path: Path,
    *,
    data_path: Path,
    image_key: str,
    offset: float,
) -> None:
    path.mkdir()
    with np.load(data_path, allow_pickle=False) as raw:
        q_nominal = np.asarray(raw["q_nominal"], dtype=np.float32)
        images = np.asarray(raw[image_key])
        episodes = np.asarray(raw["episode_id"])
        frames = np.asarray(raw["frame_id"], dtype=np.int64)
    corrected_path = path / "q_corrected.npy"
    np.save(corrected_path, q_nominal + offset)
    role = builder.IMAGE_VIEW_ROLES[image_key]
    metadata = {
        "version": 3,
        "schema_version": 3,
        "has_reliability": False,
        "source_data": str(data_path.resolve()),
        "source_data_sha256": sha256_file(data_path),
        "image_key": image_key,
        "selected_image_sha256": builder._selected_image_sha256(images),
        "view_role": role,
        "deployment_cache_eligible": role == "deployment_observation",
        "vision_image_lineage_verified": True,
        "checkpoint": "/frozen/fcpg.pt",
        "checkpoint_sha256": "a" * 64,
        "sample_count": FRAME_COUNT,
        "num_keypoints": ANCHORS,
        "frame_identity_sha256": frame_identity_sha256(episodes, frames),
        "q_corrected_sha256": sha256_file(corrected_path),
        "q_true_usage": "offline_metrics_only",
    }
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


@pytest.fixture
def artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Artifacts:
    data_path = tmp_path / "dataset.npz"
    action = _write_dataset(data_path)
    observed = tmp_path / "observed_correction"
    canonical = tmp_path / "canonical_correction"
    _write_correction_cache(
        observed, data_path=data_path, image_key="rgb", offset=0.1
    )
    _write_correction_cache(
        canonical,
        data_path=data_path,
        image_key="rgb_camera_only",
        offset=0.5,
    )
    checkpoint_path = tmp_path / "anchor_persistent_step2000.pt"
    checkpoint_path.write_bytes(b"frozen-anchor-persistent-step-2000")
    checkpoint: dict[str, object] = {
        "checkpoint_type": ANCHOR_PERSISTENT_CHECKPOINT_TYPE,
        "step": 2000,
        "data_config": {
            "history_size": 3,
            "frame_skip": 5,
            "num_keypoints": ANCHORS,
            "raw_action_dim": 4,
            "action_dim": 20,
            "image_size": [5, 4],
        },
        "model_config": {
            "world_model": {"history_size": 3, "action_dim": 20},
            "frame": {"num_keypoints": ANCHORS, "obs_dim": 3},
        },
    }
    model = _FakeAnchorModel()
    monkeypatch.setattr(
        builder,
        "load_lewm_grounded_checkpoint",
        lambda *_args, **_kwargs: (model, {}, checkpoint),
    )

    def fake_encode(
        _model: object,
        *,
        images: np.ndarray,
        obs: np.ndarray,
        q_nominal: np.ndarray,
        q_corrected: np.ndarray,
        keypoints_xyz: np.ndarray,
        valid_nominal: np.ndarray,
        batch_size: int,
        device: torch.device,
    ) -> np.ndarray:
        del q_nominal, keypoints_xyz, valid_nominal, batch_size, device
        result = np.empty((len(images), ANCHORS, LATENT_DIM), dtype=np.float32)
        result[..., 0] = images[:, None, 0, 0, 0]
        result[..., 1] = q_corrected[..., 0]
        result[..., 2] = obs[:, None, 0]
        return result

    def fake_prior(
        _model: object,
        histories: np.ndarray,
        actions: np.ndarray,
        *,
        device: torch.device,
        batch_size: int,
    ) -> np.ndarray:
        del device, batch_size
        assert histories.shape[1:] == (3, ANCHORS, LATENT_DIM)
        assert actions.shape[1:] == (3, 20)
        return histories[:, -1] + actions[:, -1, 0, None, None]

    monkeypatch.setattr(builder, "_encode_anchor_states", fake_encode)
    monkeypatch.setattr(builder, "_predict_factual_prior", fake_prior)
    return _Artifacts(
        data=data_path,
        observed_correction=observed,
        canonical_correction=canonical,
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "posterior_gate_cache",
        action=action,
        checkpoint=checkpoint,
        model=model,
    )


def _argv(artifacts: _Artifacts, *extra: str) -> list[str]:
    return [
        "--dataset",
        str(artifacts.data),
        "--observed-correction-cache",
        str(artifacts.observed_correction),
        "--canonical-correction-cache",
        str(artifacts.canonical_correction),
        "--checkpoint",
        str(artifacts.checkpoint_path),
        "--device",
        "cpu",
        "--batch-size",
        "8",
        "--output-dir",
        str(artifacts.output_dir),
        *extra,
    ]


def test_help_exposes_required_inputs_without_requiring_gpu_or_feature_cache() -> None:
    help_text = builder.build_parser().format_help()
    for option in (
        "--dataset",
        "--observed-correction-cache",
        "--canonical-correction-cache",
        "--checkpoint",
        "--device",
        "--evaluation-view",
        "--output-dir",
    ):
        assert option in help_text
    assert "--observed-feature-cache" not in help_text
    assert "--canonical-feature-cache" not in help_text


def test_cli_builds_partitioned_fresh_cache_with_target_only_canonical_history(
    artifacts: _Artifacts,
) -> None:
    assert builder.main(_argv(artifacts)) == 0
    assert {path.name for path in artifacts.output_dir.iterdir()} == {
        builder.DEPLOYMENT_FILENAME,
        builder.LABEL_FILENAME,
        builder.EVALUATION_FILENAME,
        builder.METADATA_FILENAME,
    }
    with np.load(
        artifacts.output_dir / builder.DEPLOYMENT_FILENAME, allow_pickle=False
    ) as deployment, np.load(
        artifacts.output_dir / builder.LABEL_FILENAME, allow_pickle=False
    ) as labels, np.load(
        artifacts.output_dir / builder.EVALUATION_FILENAME, allow_pickle=False
    ) as evaluation:
        assert set(deployment.files) == set(builder.DEPLOYABLE_TENSOR_KEYS)
        assert set(labels.files) == set(builder.OFFLINE_LABEL_KEYS)
        assert set(deployment.files).isdisjoint(labels.files)
        assert set(builder.DEPLOYABLE_TENSOR_KEYS).isdisjoint(
            builder.OFFLINE_LABEL_KEYS
        )
        assert deployment["measurement_frame_embedding"].shape == (
            FRAME_COUNT,
            ANCHORS,
            LATENT_DIM,
        )
        assert deployment["measurement_history"].shape == (
            25,
            3,
            ANCHORS,
            LATENT_DIM,
        )
        assert deployment["action_chunks"].shape == (25, 3, 20)
        assert deployment["teacher_forced_factual_prior"].shape == (
            25,
            ANCHORS,
            LATENT_DIM,
        )
        assert labels["canonical_measurement_history"].shape == (
            25,
            3,
            ANCHORS,
            LATENT_DIM,
        )
        assert labels["canonical_target"].shape == (25, ANCHORS, LATENT_DIM)
        assert labels["canonical_q_corrected_history"].shape == (
            25,
            3,
            ANCHORS,
            2,
        )
        assert labels["canonical_q_corrected"].shape == (25, ANCHORS, 2)
        assert labels["canonical_valid_history"].shape == (25, 3, ANCHORS)
        assert labels["canonical_valid_current"].shape == (25, ANCHORS)
        np.testing.assert_array_equal(
            labels["canonical_target"], labels["canonical_current"]
        )
        np.testing.assert_array_equal(
            labels["canonical_measurement_history"][0, :, 0, 0],
            [55.0, 60.0, 65.0],
        )
        assert labels["canonical_target"][0, 0, 0] == 70.0
        assert bool(labels["canonical_valid_history"][:, :, 0].all())
        assert not bool(labels["canonical_valid_history"][:, :, 1].any())
        assert not bool(labels["canonical_valid_current"][:, 1].any())

        np.testing.assert_array_equal(evaluation["history_frames"][0], [5, 10, 15])
        np.testing.assert_array_equal(
            deployment["action_chunks"][0, 0], artifacts.action[5:10].reshape(-1)
        )
        assert set(evaluation["event"].astype(str)) == set(
            builder.FRESH_EVENT_NAMES
        )
        assert {name: int(np.count_nonzero(evaluation["event"] == name)) for name in builder.FRESH_EVENT_NAMES} == {
            name: 5 for name in builder.FRESH_EVENT_NAMES
        }
        for forbidden in (
            "q_true",
            "measurement_valid",
            "dropout",
            "recovery",
            "event",
            "task",
            "episode",
            "canonical",
        ):
            assert not any(forbidden in key.lower() for key in deployment.files)
        assert "source_q_true" in evaluation.files
        assert "source_measurement_valid" in evaluation.files
        assert "source_anchor_dropout_active" in evaluation.files
        assert "source_anchor_recovery_segment" in evaluation.files

    metadata = json.loads(
        (artifacts.output_dir / builder.METADATA_FILENAME).read_text(encoding="utf-8")
    )
    assert metadata["contract"]["checkpoint_step"] == 2000
    assert metadata["contract"]["action_dim"] == 20
    assert metadata["contract"]["canonical_arrays_role"] == (
        "target_only_clean_safety_diagnostic_and_offline_label"
    )
    assert set(metadata["deployable_tensor_keys"]).isdisjoint(
        metadata["offline_label_keys"]
    )
    assert metadata["inputs"]["dataset"]["sha256"] == sha256_file(artifacts.data)
    for name in (
        "observed_correction_cache",
        "measurement_correction_cache",
        "canonical_correction_cache",
    ):
        assert (
            metadata["inputs"][name]["corrector_checkpoint_sha256"]
            == "a" * 64
        )
    assert "source_files" not in metadata
    assert "execution_files" not in metadata


def test_clean_evaluation_view_uses_canonical_as_deployment_observation(
    artifacts: _Artifacts,
) -> None:
    assert (
        builder.main(
            _argv(
                artifacts,
                "--evaluation-view",
                "canonical_unoccluded_only",
            )
        )
        == 0
    )
    with np.load(
        artifacts.output_dir / builder.DEPLOYMENT_FILENAME, allow_pickle=False
    ) as deployment, np.load(
        artifacts.output_dir / builder.LABEL_FILENAME, allow_pickle=False
    ) as labels:
        np.testing.assert_array_equal(
            deployment["measurement_current"], labels["canonical_current"]
        )
        np.testing.assert_array_equal(
            deployment["measurement_history"],
            labels["canonical_measurement_history"],
        )
        np.testing.assert_array_equal(
            deployment["q_corrected_current"],
            labels["canonical_q_corrected"],
        )
        np.testing.assert_array_equal(
            deployment["q_corrected_history"],
            labels["canonical_q_corrected_history"],
        )

    metadata = json.loads(
        (artifacts.output_dir / builder.METADATA_FILENAME).read_text(encoding="utf-8")
    )
    assert metadata["contract"]["evaluation_view"] == "canonical_unoccluded_only"
    assert metadata["contract"]["clean_evaluation_only"] is True
    assert metadata["contract"]["measurement_image_key"] == "rgb_camera_only"
    assert metadata["contract"]["measurement_view_role"] == (
        "independent_clean_deployment_observation"
    )
    assert metadata["inputs"]["measurement_correction_cache"]["path"] == str(
        artifacts.canonical_correction.resolve()
    )
    assert "source_files" not in metadata
    assert "execution_files" not in metadata


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--expected-step", "1999", "checkpoint step 2000"),
        ("--history-size", "4", "history_size 3"),
        ("--frame-skip", "4", "frame_skip 5"),
    ],
)
def test_cli_rejects_nonfrozen_temporal_contract(
    artifacts: _Artifacts, option: str, value: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        builder.main(_argv(artifacts, option, value))
    assert not artifacts.output_dir.exists()


def test_correction_frame_identity_mismatch_fails_before_output(
    artifacts: _Artifacts,
) -> None:
    metadata_path = artifacts.observed_correction / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["frame_identity_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="frame identity"):
        builder.main(_argv(artifacts))
    assert not artifacts.output_dir.exists()


def test_checkpoint_requires_action_dim_20(artifacts: _Artifacts) -> None:
    dataset = builder._load_gate_dataset(
        artifacts.data,
        observed_image_key="rgb",
        canonical_image_key="rgb_camera_only",
    )
    checkpoint = deepcopy(artifacts.checkpoint)
    checkpoint["data_config"]["action_dim"] = 19  # type: ignore[index]
    with pytest.raises(ValueError, match="action_dim"):
        builder._validate_checkpoint_contract(
            artifacts.model,
            checkpoint,
            dataset=dataset,
            expected_step=2000,
            history_size=3,
            frame_skip=5,
        )


def test_atomic_directory_cleanup_on_partition_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "cache"
    calls = 0
    original = builder._atomic_write_npz

    def fail_second(path: Path, arrays: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected write failure")
        original(path, arrays)  # type: ignore[arg-type]

    monkeypatch.setattr(builder, "_atomic_write_npz", fail_second)
    with pytest.raises(RuntimeError, match="injected"):
        builder._publish_cache_directory(
            output,
            deployment={"value": np.zeros(1, dtype=np.float32)},
            labels={"value": np.zeros(1, dtype=np.float32)},
            evaluation={"value": np.zeros(1, dtype=np.float32)},
            metadata={},
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".cache.tmp-*"))
