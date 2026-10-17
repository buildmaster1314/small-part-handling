from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from fcpg.lewm_posterior_gate_data import (
    DEPLOYMENT_TENSOR_KEYS,
    FRESH_EVENT_NAMES,
    PhysicalEpisodeIdentity,
    build_disjoint_episode_sha_manifests,
    build_fresh_posterior_gate_examples,
    build_physical_episode_sha_manifest,
    physical_episode_sha256,
    validate_disjoint_episode_sha_manifests,
)


def _torch_frames(
    *,
    count: int = 40,
    dropout_start: int = 15,
) -> dict[str, object]:
    anchors = 2
    latent_dim = 3
    action_dim = 2
    frame = torch.arange(count, dtype=torch.float32)
    measurement = (
        frame[:, None, None]
        + torch.arange(anchors, dtype=torch.float32)[None, :, None] * 0.1
        + torch.arange(latent_dim, dtype=torch.float32)[None, None, :] * 0.01
    )
    canonical = (measurement + 1_000.0).clone().requires_grad_(True)
    action = (
        frame[:, None] * 10.0
        + torch.arange(action_dim, dtype=torch.float32)[None, :]
    )
    q_nominal = torch.stack((frame, frame + 0.5), dim=-1)
    q_nominal = q_nominal[:, None, :].expand(-1, anchors, -1).clone()
    q_corrected = q_nominal + 2.0
    valid_nominal = torch.ones((count, anchors), dtype=torch.bool)
    dropout = torch.zeros((count, anchors), dtype=torch.bool)
    recovery = torch.zeros_like(dropout)
    dropout[dropout_start : dropout_start + 15, 1] = True
    recovery[dropout_start + 15 : dropout_start + 25, 1] = True
    return {
        "task": ["pick-place-v3"] * count,
        "episode": [7] * count,
        "frame": torch.arange(count, dtype=torch.int64),
        "action": action,
        "measurement_embedding": measurement,
        "canonical_label": canonical,
        "q_nominal": q_nominal,
        "q_corrected": q_corrected,
        "valid_nominal": valid_nominal,
        "anchor_dropout_active": dropout,
        "anchor_recovery_segment": recovery,
        "observation_action_alignment": "pre_action",
    }


def _build(payload: dict[str, object]):
    return build_fresh_posterior_gate_examples(**payload)  # type: ignore[arg-type]


def test_builds_five_exact_phase_streams_with_measurement_only_histories() -> None:
    payload = _torch_frames()
    result = _build(payload)

    assert result.sample_count == 25
    assert result.phase_stream_count == 5
    assert len(result.stream_slices) == 5
    assert tuple(result.deployment) == DEPLOYMENT_TENSOR_KEYS
    assert set(result.labels) == {"canonical_current"}
    assert not (
        {"task", "task_id", "episode", "event", "phase", "canonical"}
        & set(result.deployment)
    )
    assert not (
        {"dropout_active", "recovery_active", "measurement_valid"}
        & set(result.deployment)
    )

    first_stream = result.metadata[result.stream_slices[0]]
    last_stream = result.metadata[result.stream_slices[-1]]
    assert tuple(row.event for row in first_stream) == FRESH_EVENT_NAMES
    assert [row.target_frame for row in first_stream] == [15, 20, 25, 30, 35]
    assert [row.target_frame for row in last_stream] == [19, 24, 29, 34, 39]
    assert [row.phase_index for row in first_stream] == [0] * 5
    assert [row.phase_index for row in last_stream] == [4] * 5
    assert [row.stream_offset for row in first_stream] == [0] * 5
    assert [row.stream_offset for row in last_stream] == [4] * 5
    assert [row.frame_phase for row in first_stream] == [0] * 5
    assert all(row.physical_episode == ("pick-place-v3", 7) for row in result.metadata)

    measurement = payload["measurement_embedding"]
    canonical = payload["canonical_label"]
    assert isinstance(measurement, torch.Tensor)
    assert isinstance(canonical, torch.Tensor)
    torch.testing.assert_close(
        result.deployment.measurement_history[0], measurement[[0, 5, 10]]
    )
    # o1 is forced from the deployment measurement at o0, never canonical o0.
    torch.testing.assert_close(
        result.deployment.measurement_history[1], measurement[[5, 10, 15]]
    )
    assert not torch.equal(
        result.deployment.measurement_history[1, -1], canonical[15]
    )
    assert not hasattr(result.labels, "canonical_history")

    torch.testing.assert_close(result.labels.canonical_current[0], canonical[15])
    assert not result.labels.canonical_current.requires_grad
    assert result.labels.canonical_current.data_ptr() != canonical.data_ptr()
    assert result.deployment.measurement_history.shape == (25, 3, 2, 3)
    assert result.deployment.action_chunks.shape == (25, 3, 10)
    assert result.deployment.q_nominal_history.shape == (25, 3, 2, 2)
    assert result.deployment.valid_nominal_current.shape == (25, 2)


def test_action_chunks_map_history_states_to_target_without_target_action() -> None:
    payload = _torch_frames()
    result = _build(payload)
    action = payload["action"]
    assert isinstance(action, torch.Tensor)

    expected = torch.stack(
        (
            action[0:5].reshape(-1),
            action[5:10].reshape(-1),
            action[10:15].reshape(-1),
        )
    )
    torch.testing.assert_close(result.deployment.action_chunks[0], expected)
    assert result.metadata[0].history_frames == (0, 5, 10)
    assert result.metadata[0].action_frames == (
        (0, 1, 2, 3, 4),
        (5, 6, 7, 8, 9),
        (10, 11, 12, 13, 14),
    )
    target_action = action[15]
    assert not bool(
        (result.deployment.action_chunks[0].reshape(-1, 2) == target_action)
        .all(dim=1)
        .any()
    )


def test_annotations_are_evaluation_only_and_follow_event_rows() -> None:
    result = _build(_torch_frames())
    expected_dropout = torch.tensor(
        [[False, True]] * 3 + [[False, False]] * 2
    )
    expected_recovery = torch.tensor(
        [[False, False]] * 3 + [[False, True]] * 2
    )
    torch.testing.assert_close(result.evaluation.dropout_active[:5], expected_dropout)
    torch.testing.assert_close(
        result.evaluation.recovery_active[:5], expected_recovery
    )
    assert set(result.evaluation) == {"dropout_active", "recovery_active"}


def test_phase_index_is_raw_frame_modulo_not_relative_stream_offset() -> None:
    result = _build(_torch_frames(count=41, dropout_start=16))
    first_stream = result.metadata[result.stream_slices[0]]
    last_stream = result.metadata[result.stream_slices[-1]]
    assert [row.phase_index for row in first_stream] == [1] * 5
    assert [row.stream_offset for row in first_stream] == [0] * 5
    assert [row.phase_index for row in last_stream] == [0] * 5
    assert [row.stream_offset for row in last_stream] == [4] * 5


def test_numpy_backend_is_preserved_and_clean_episodes_are_not_selected() -> None:
    selected = _torch_frames()
    clean = _torch_frames()
    count = 40
    payload: dict[str, object] = {
        "task": ["clean-v3"] * count + list(selected["task"]),
        "episode": [2] * count + list(selected["episode"]),
        "frame": np.concatenate((np.arange(count), np.arange(count))),
        "observation_action_alignment": "pre_action",
    }
    for name in (
        "action",
        "measurement_embedding",
        "canonical_label",
        "q_nominal",
        "q_corrected",
        "valid_nominal",
        "anchor_dropout_active",
        "anchor_recovery_segment",
    ):
        left = clean[name]
        right = selected[name]
        assert isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
        if name in {"anchor_dropout_active", "anchor_recovery_segment"}:
            left = torch.zeros_like(left)
        payload[name] = torch.cat((left.detach(), right.detach())).cpu().numpy()

    result = _build(payload)
    assert isinstance(result.deployment.measurement_history, np.ndarray)
    assert isinstance(result.labels.canonical_current, np.ndarray)
    assert result.sample_count == 25
    assert {row.task for row in result.metadata} == {"pick-place-v3"}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("observation_action_alignment", "post_action", "pre_action"),
        ("frame_skip", 4, "frame_skip must be 5"),
        ("history_size", 2, "history_size must be 3"),
    ],
)
def test_rejects_non_fresh_alignment_and_sequence_constants(
    field: str, value: object, message: str
) -> None:
    payload = _torch_frames()
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        _build(payload)


def test_requires_exact_dropout_and_recovery_lengths() -> None:
    short_dropout = _torch_frames()
    dropout = short_dropout["anchor_dropout_active"]
    assert isinstance(dropout, torch.Tensor)
    dropout[29, 1] = False
    with pytest.raises(ValueError, match="exactly 15 dropout"):
        _build(short_dropout)

    short_recovery = _torch_frames()
    recovery = short_recovery["anchor_recovery_segment"]
    assert isinstance(recovery, torch.Tensor)
    recovery[39, 1] = False
    with pytest.raises(ValueError, match="exactly 10 finite recovery"):
        _build(short_recovery)


def test_requires_consecutive_masks_and_one_fixed_anchor_set() -> None:
    gapped = _torch_frames(count=41)
    recovery = gapped["anchor_recovery_segment"]
    assert isinstance(recovery, torch.Tensor)
    recovery[34, 1] = False
    recovery[40, 1] = True
    with pytest.raises(ValueError, match="recovery frames must be consecutive"):
        _build(gapped)

    changing_anchor = _torch_frames()
    dropout = changing_anchor["anchor_dropout_active"]
    assert isinstance(dropout, torch.Tensor)
    dropout[20, 0] = True
    with pytest.raises(ValueError, match="one fixed anchor set"):
        _build(changing_anchor)


def test_requires_contiguous_physical_frames_and_full_prehistory() -> None:
    gap = _torch_frames()
    gap["frame"] = torch.cat(
        (torch.arange(20), torch.arange(21, 41))
    ).to(torch.int64)
    with pytest.raises(ValueError, match="strictly contiguous"):
        _build(gap)

    early = _torch_frames(dropout_start=14)
    with pytest.raises(ValueError, match="15 contiguous pre-dropout"):
        _build(early)


def test_rejects_nonfinite_recovery_data() -> None:
    payload = _torch_frames()
    measurement = payload["measurement_embedding"]
    assert isinstance(measurement, torch.Tensor)
    measurement[35, 1, 0] = torch.nan
    with pytest.raises(ValueError, match="measurement_embedding must be finite"):
        _build(payload)


def test_deployment_floating_dtypes_must_match() -> None:
    action_mismatch = _torch_frames()
    action = action_mismatch["action"]
    assert isinstance(action, torch.Tensor)
    action_mismatch["action"] = action.double()
    with pytest.raises(TypeError, match="action dtype"):
        _build(action_mismatch)

    coordinate_mismatch = _torch_frames()
    nominal = coordinate_mismatch["q_nominal"]
    corrected = coordinate_mismatch["q_corrected"]
    assert isinstance(nominal, torch.Tensor)
    assert isinstance(corrected, torch.Tensor)
    coordinate_mismatch["q_nominal"] = nominal.double()
    coordinate_mismatch["q_corrected"] = corrected.double()
    with pytest.raises(TypeError, match="q_nominal and q_corrected dtype"):
        _build(coordinate_mismatch)


def _identity(seed: int) -> dict[str, object]:
    return {
        "task": "pick-place-v3",
        "task_index": seed % 3,
        "env_seed": 10_000 + seed,
        "task_generation_seed": 9517,
    }


def test_physical_episode_sha_binds_all_identity_fields_canonically() -> None:
    record = _identity(1)
    digest = physical_episode_sha256(**record)  # type: ignore[arg-type]
    expected_payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    assert digest == hashlib.sha256(expected_payload).hexdigest()
    assert len(
        {
            physical_episode_sha256(
                **{**record, field: value}  # type: ignore[arg-type]
            )
            for field, value in (
                ("task", "push-v3"),
                ("task_index", 9),
                ("env_seed", 99),
                ("task_generation_seed", 99),
            )
        }
    ) == 4
    identity = PhysicalEpisodeIdentity(**record)  # type: ignore[arg-type]
    assert identity.sha256 == digest


def test_builds_sorted_disjoint_train_dev_sealed_manifests() -> None:
    manifests = build_disjoint_episode_sha_manifests(
        train=[_identity(2), _identity(1)],
        dev=[_identity(3)],
        sealed=[_identity(4)],
    )
    assert tuple(manifests) == ("train", "dev", "sealed")
    assert manifests.train == tuple(sorted(manifests.train))
    assert manifests.train == build_physical_episode_sha_manifest(
        [_identity(1), _identity(2)]
    )
    validate_disjoint_episode_sha_manifests(manifests)


def test_manifest_validation_rejects_overlap_duplicates_and_bad_schema() -> None:
    with pytest.raises(ValueError, match="overlap between train and sealed"):
        build_disjoint_episode_sha_manifests(
            train=[_identity(1)],
            dev=[_identity(2)],
            sealed=[_identity(1)],
        )
    digest = physical_episode_sha256(**_identity(1))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="contains duplicates"):
        validate_disjoint_episode_sha_manifests(
            {"train": [digest, digest], "dev": ["b" * 64], "sealed": ["c" * 64]}
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        validate_disjoint_episode_sha_manifests(
            {"train": [digest.upper()], "dev": ["b" * 64], "sealed": ["c" * 64]}
        )
    with pytest.raises(ValueError, match="identity keys changed"):
        build_physical_episode_sha_manifest([{**_identity(1), "episode_id": 0}])
