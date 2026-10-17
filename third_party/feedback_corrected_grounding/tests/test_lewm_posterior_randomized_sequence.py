from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from fcpg.lewm_posterior_gate_data import DEPLOYMENT_TENSOR_KEYS
from fcpg.lewm_posterior_randomized_sequence import (
    PosteriorRandomizedSequenceExamples,
    build_fresh_posterior_randomized_sequence,
)


def _torch_frames(
    *,
    active_frames: int = 15,
    recovery_frames: int = 10,
    onset: int = 17,
    anchor: int = 1,
    anchors: int = 3,
    tail: int = 0,
    frame_base: int = 0,
) -> dict[str, object]:
    count = onset + active_frames + recovery_frames + tail
    latent_dim = 3
    action_dim = 2
    index = torch.arange(count, dtype=torch.float32)
    measurement = (
        index[:, None, None]
        + torch.arange(anchors, dtype=torch.float32)[None, :, None] * 0.1
        + torch.arange(latent_dim, dtype=torch.float32)[None, None, :] * 0.01
    )
    canonical = (measurement + 1_000.0).clone().requires_grad_(True)
    action = (
        index[:, None] * 10.0
        + torch.arange(action_dim, dtype=torch.float32)[None, :]
    )
    q_nominal = torch.stack((index, index + 0.5), dim=-1)
    q_nominal = q_nominal[:, None, :].expand(-1, anchors, -1).clone()
    q_corrected = q_nominal + 2.0
    valid_nominal = torch.ones((count, anchors), dtype=torch.bool)
    dropout = torch.zeros((count, anchors), dtype=torch.bool)
    recovery = torch.zeros_like(dropout)
    dropout[onset : onset + active_frames, anchor] = True
    recovery[
        onset + active_frames : onset + active_frames + recovery_frames, anchor
    ] = True
    return {
        "task": ["pick-place-v3"] * count,
        "episode": [7] * count,
        "frame": torch.arange(frame_base, frame_base + count, dtype=torch.int64),
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


def _build(payload: dict[str, object]) -> PosteriorRandomizedSequenceExamples:
    return build_fresh_posterior_randomized_sequence(  # type: ignore[arg-type]
        **payload
    )


@pytest.mark.parametrize("duration", [10, 15, 20])
def test_variable_durations_define_exact_widths_slices_and_events(
    duration: int,
) -> None:
    result = _build(
        _torch_frames(active_frames=duration, recovery_frames=duration, onset=15)
    )
    events_per_interval = duration // 5
    width = events_per_interval * 2

    assert result.sample_count == 5 * width
    assert result.phase_stream_count == 5
    assert result.stream_slices == tuple(
        slice(phase * width, (phase + 1) * width) for phase in range(5)
    )
    assert tuple(result.deployment) == DEPLOYMENT_TENSOR_KEYS
    assert set(result.labels) == {"canonical_current"}
    assert set(result.evaluation) == {"dropout_active", "recovery_active"}
    assert "canonical_current" not in result.deployment
    assert "dropout_active" not in result.deployment

    expected_events = tuple(
        [f"o{index}" for index in range(events_per_interval)]
        + [f"r{index}" for index in range(events_per_interval)]
    )
    for phase, stream_slice in enumerate(result.stream_slices):
        stream = result.metadata[stream_slice]
        assert tuple(row.event for row in stream) == expected_events
        assert tuple(row.event_index for row in stream) == tuple(range(width))
        assert {row.phase_index for row in stream} == {phase}
        assert {row.frame_phase for row in stream} == {phase}
        assert all(
            right.target_frame - left.target_frame == 5
            for left, right in zip(stream, stream[1:])
        )


def test_shifted_onset_orders_raw_phases_and_preserves_random_anchor() -> None:
    result = _build(
        _torch_frames(
            active_frames=15,
            recovery_frames=20,
            onset=17,
            anchor=2,
        )
    )
    assert [
        result.metadata[item.start].phase_index for item in result.stream_slices
    ] == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert [
        result.metadata[item.start].stream_offset for item in result.stream_slices
    ] == [
        3,
        4,
        0,
        1,
        2,
    ]

    phase_zero = result.metadata[result.stream_slices[0]]
    assert [row.target_frame for row in phase_zero] == [20, 25, 30, 35, 40, 45, 50]
    assert [row.event for row in phase_zero] == [
        "o0",
        "o1",
        "o2",
        "r0",
        "r1",
        "r2",
        "r3",
    ]
    expected_dropout = torch.tensor([[False, False, True]] * 3)
    expected_recovery = torch.tensor([[False, False, True]] * 4)
    torch.testing.assert_close(
        result.evaluation.dropout_active[result.stream_slices[0]],
        torch.cat((expected_dropout, torch.zeros_like(expected_recovery))),
    )
    torch.testing.assert_close(
        result.evaluation.recovery_active[result.stream_slices[0]],
        torch.cat((torch.zeros_like(expected_dropout), expected_recovery)),
    )


def test_all_phases_preserve_boundary_causality_and_masks() -> None:
    payload = _torch_frames(
        active_frames=15,
        recovery_frames=20,
        onset=17,
        anchor=2,
        frame_base=101,
    )
    result = _build(payload)
    frames = payload["frame"]
    action = payload["action"]
    measurement = payload["measurement_embedding"]
    assert isinstance(frames, torch.Tensor)
    assert isinstance(action, torch.Tensor)
    assert isinstance(measurement, torch.Tensor)
    raw_index_by_frame = {
        int(frame): index for index, frame in enumerate(frames.tolist())
    }
    expected_anchor_mask = torch.tensor([False, False, True])

    assert {
        result.metadata[stream_slice.start].phase_index
        for stream_slice in result.stream_slices
    } == set(range(5))
    for stream_slice in result.stream_slices:
        stream = result.metadata[stream_slice]
        recovery_start = next(
            index for index, row in enumerate(stream) if row.event == "r0"
        )
        assert stream[recovery_start - 1].event == "o2"
        assert (
            stream[recovery_start].target_frame
            - stream[recovery_start - 1].target_frame
            == 5
        )

        for local_index, row in enumerate(stream):
            expected_history = tuple(
                row.target_frame - offset for offset in (15, 10, 5)
            )
            expected_actions = tuple(
                tuple(range(start, start + 5)) for start in expected_history
            )
            assert row.history_frames == expected_history
            assert row.action_frames == expected_actions
            assert row.target_frame not in {
                frame for chunk in row.action_frames for frame in chunk
            }

            row_index = stream_slice.start + local_index
            history_indices = [
                raw_index_by_frame[frame] for frame in expected_history
            ]
            torch.testing.assert_close(
                result.deployment.measurement_history[row_index],
                measurement[history_indices],
            )
            expected_action_chunks = torch.stack(
                tuple(
                    action[
                        raw_index_by_frame[chunk[0]] : raw_index_by_frame[chunk[-1]]
                        + 1
                    ].reshape(-1)
                    for chunk in expected_actions
                )
            )
            torch.testing.assert_close(
                result.deployment.action_chunks[row_index], expected_action_chunks
            )

            active = row.event.startswith("o")
            empty_mask = torch.zeros(3, dtype=torch.bool)
            expected_dropout = expected_anchor_mask if active else empty_mask
            expected_recovery = expected_anchor_mask if not active else empty_mask
            torch.testing.assert_close(
                result.evaluation.dropout_active[row_index], expected_dropout
            )
            torch.testing.assert_close(
                result.evaluation.recovery_active[row_index], expected_recovery
            )

        for left, right in zip(stream, stream[1:]):
            assert right.history_frames[:-1] == left.history_frames[1:]
            assert right.history_frames[-1] == left.target_frame


def test_rejects_one_slice_that_combines_multiple_phase_streams() -> None:
    result = _build(_torch_frames(onset=15))

    with pytest.raises(ValueError, match="exactly one task, episode, phase"):
        replace(result, stream_slices=(slice(0, result.sample_count),))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task", "other-task"),
        ("episode_id", 8),
        ("phase_index", 1),
        ("stream_offset", 1),
    ],
)
def test_rejects_mixed_stream_identity(field: str, value: object) -> None:
    result = _build(_torch_frames(onset=15))
    metadata = list(result.metadata)
    metadata[1] = replace(metadata[1], **{field: value})

    with pytest.raises(ValueError, match="exactly one task, episode, phase"):
        replace(result, metadata=tuple(metadata))


@pytest.mark.parametrize(
    ("row_index", "changes", "message"),
    [
        (3, {"event": "r1"}, "ordered o0"),
        (3, {"event_index": 0}, "event_index"),
        (1, {"event_index": True}, "event_index must be an integer"),
        (1, {"target_frame": 25}, "five frames apart"),
        (1, {"frame_phase": 1}, "frame_phase must match"),
        (0, {"frame_phase": False}, "frame_phase must be an integer"),
    ],
)
def test_rejects_malformed_stream_rows(
    row_index: int, changes: dict[str, object], message: str
) -> None:
    result = _build(_torch_frames(onset=15))
    metadata = list(result.metadata)
    metadata[row_index] = replace(metadata[row_index], **changes)

    with pytest.raises((TypeError, ValueError), match=message):
        replace(result, metadata=tuple(metadata))


def test_rejects_duplicate_or_missing_episode_phase() -> None:
    result = _build(_torch_frames(onset=15))
    metadata = list(result.metadata)
    last_stream = result.stream_slices[-1]
    for index in range(last_stream.start, last_stream.stop):
        metadata[index] = replace(
            metadata[index],
            phase_index=3,
            frame_phase=3,
            target_frame=metadata[index].target_frame - 1,
        )

    with pytest.raises(ValueError, match="duplicate phase 3"):
        replace(result, metadata=tuple(metadata))


def test_history_and_actions_are_strictly_causal() -> None:
    payload = _torch_frames(active_frames=10, recovery_frames=10, onset=15)
    result = _build(payload)
    action = payload["action"]
    measurement = payload["measurement_embedding"]
    canonical = payload["canonical_label"]
    assert isinstance(action, torch.Tensor)
    assert isinstance(measurement, torch.Tensor)
    assert isinstance(canonical, torch.Tensor)

    first = result.metadata[0]
    assert first.target_frame == 15
    assert first.history_frames == (0, 5, 10)
    assert first.action_frames == (
        (0, 1, 2, 3, 4),
        (5, 6, 7, 8, 9),
        (10, 11, 12, 13, 14),
    )
    torch.testing.assert_close(
        result.deployment.measurement_history[0], measurement[[0, 5, 10]]
    )
    torch.testing.assert_close(
        result.deployment.action_chunks[0],
        torch.stack(
            (
                action[0:5].reshape(-1),
                action[5:10].reshape(-1),
                action[10:15].reshape(-1),
            )
        ),
    )
    assert not bool(
        (result.deployment.action_chunks[0].reshape(-1, 2) == action[15])
        .all(dim=1)
        .any()
    )
    torch.testing.assert_close(result.labels.canonical_current[0], canonical[15])
    assert not result.labels.canonical_current.requires_grad
    assert result.labels.canonical_current.data_ptr() != canonical.data_ptr()


def test_multiple_episodes_are_sorted_deterministically_with_explicit_slices() -> None:
    later = _torch_frames(
        active_frames=20, recovery_frames=10, onset=15, anchor=0
    )
    earlier = _torch_frames(
        active_frames=10, recovery_frames=15, onset=15, anchor=2
    )
    later_count = len(later["task"])  # type: ignore[arg-type]
    earlier_count = len(earlier["task"])  # type: ignore[arg-type]
    payload: dict[str, object] = {
        "task": ["z-task"] * later_count + ["a-task"] * earlier_count,
        "episode": [9] * later_count + [3] * earlier_count,
        "frame": torch.cat((later["frame"], earlier["frame"])),
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
        left = later[name]
        right = earlier[name]
        assert isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
        payload[name] = torch.cat((left.detach(), right.detach()))

    result = _build(payload)
    assert [
        result.metadata[item.start].physical_episode for item in result.stream_slices
    ] == [("a-task", 3)] * 5 + [("z-task", 9)] * 5
    assert [(item.start, item.stop) for item in result.stream_slices] == [
        (0, 5),
        (5, 10),
        (10, 15),
        (15, 20),
        (20, 25),
        (25, 31),
        (31, 37),
        (37, 43),
        (43, 49),
        (49, 55),
    ]


def test_requires_full_history_and_recovery_tail() -> None:
    with pytest.raises(ValueError, match="15 contiguous pre-active"):
        _build(_torch_frames(onset=14))

    short_tail = _torch_frames(
        active_frames=10, recovery_frames=10, onset=15
    )
    recovery = short_tail["anchor_recovery_segment"]
    assert isinstance(recovery, torch.Tensor)
    recovery[-1, 1] = False
    with pytest.raises(ValueError, match="recovery duration"):
        _build(short_tail)


@pytest.mark.parametrize(
    ("active_frames", "recovery_frames", "message"),
    [
        (5, 10, "active duration"),
        (12, 10, "active duration"),
        (10, 5, "recovery duration"),
        (10, 12, "recovery duration"),
    ],
)
def test_rejects_short_or_non_multiple_durations(
    active_frames: int, recovery_frames: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _build(
            _torch_frames(
                active_frames=active_frames,
                recovery_frames=recovery_frames,
                onset=15,
            )
        )


def test_rejects_partial_gapped_and_multi_anchor_schedules() -> None:
    partial = _torch_frames(onset=15)
    recovery = partial["anchor_recovery_segment"]
    assert isinstance(recovery, torch.Tensor)
    recovery.zero_()
    with pytest.raises(ValueError, match="partial active/recovery"):
        _build(partial)

    gapped = _torch_frames(onset=15, tail=1)
    recovery = gapped["anchor_recovery_segment"]
    assert isinstance(recovery, torch.Tensor)
    recovery[32, 1] = False
    recovery[-1, 1] = True
    with pytest.raises(ValueError, match="recovery interval must be contiguous"):
        _build(gapped)

    multi_anchor = _torch_frames(onset=15)
    dropout = multi_anchor["anchor_dropout_active"]
    assert isinstance(dropout, torch.Tensor)
    dropout[20, 0] = True
    with pytest.raises(ValueError, match="exactly one active anchor"):
        _build(multi_anchor)

    wrong_recovery_anchor = _torch_frames(onset=15)
    recovery = wrong_recovery_anchor["anchor_recovery_segment"]
    assert isinstance(recovery, torch.Tensor)
    recovery[:, 0] = recovery[:, 1]
    recovery[:, 1] = False
    with pytest.raises(ValueError, match="single active anchor"):
        _build(wrong_recovery_anchor)


def test_rejects_bad_alignment_shapes_dtypes_nonfinite_and_frame_gaps() -> None:
    alignment = _torch_frames(onset=15)
    alignment["observation_action_alignment"] = "post_action"
    with pytest.raises(ValueError, match="pre_action"):
        _build(alignment)

    bad_dtype = _torch_frames(onset=15)
    action = bad_dtype["action"]
    assert isinstance(action, torch.Tensor)
    bad_dtype["action"] = action.double()
    with pytest.raises(TypeError, match="action dtype"):
        _build(bad_dtype)

    nonfinite = _torch_frames(onset=15)
    measurement = nonfinite["measurement_embedding"]
    assert isinstance(measurement, torch.Tensor)
    measurement[20, 1, 0] = torch.nan
    with pytest.raises(ValueError, match="measurement_embedding must be finite"):
        _build(nonfinite)

    gap = _torch_frames(onset=15)
    frames = gap["frame"]
    assert isinstance(frames, torch.Tensor)
    gap["frame"] = torch.cat((frames[:20], frames[20:] + 1))
    with pytest.raises(ValueError, match="strictly contiguous"):
        _build(gap)


def test_numpy_and_torch_builders_are_exactly_equivalent() -> None:
    torch_payload = _torch_frames(
        active_frames=20,
        recovery_frames=15,
        onset=18,
        anchor=2,
        frame_base=101,
    )
    numpy_payload: dict[str, object] = {}
    for name, value in torch_payload.items():
        if isinstance(value, torch.Tensor):
            numpy_payload[name] = value.detach().cpu().numpy()
        else:
            numpy_payload[name] = value

    torch_result = _build(torch_payload)
    numpy_result = _build(numpy_payload)
    assert numpy_result.metadata == torch_result.metadata
    assert numpy_result.stream_slices == torch_result.stream_slices
    for torch_container, numpy_container in (
        (torch_result.deployment, numpy_result.deployment),
        (torch_result.labels, numpy_result.labels),
        (torch_result.evaluation, numpy_result.evaluation),
    ):
        assert tuple(torch_container) == tuple(numpy_container)
        for name in torch_container:
            numpy_value = numpy_container[name]
            torch_value = torch_container[name]
            assert isinstance(numpy_value, np.ndarray)
            assert isinstance(torch_value, torch.Tensor)
            np.testing.assert_array_equal(
                numpy_value, torch_value.detach().cpu().numpy()
            )
