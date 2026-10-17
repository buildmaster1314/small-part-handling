from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fcpg import lewm_posterior_randomized_dataset as randomized
from fcpg.visual_trust_corruptions import CORRUPTION_TYPES
from scripts import build_lewm_posterior_randomized_dataset as cli
from scripts.build_lewm_posterior_gate_cache import _load_gate_dataset


def _source(
    *,
    env_seed_start: int,
    mode: str,
    episode_count: int = 2,
    frame_count: int = 30,
    include_camera_only: bool = True,
) -> dict[str, np.ndarray]:
    rows = [
        (episode, frame)
        for frame in range(frame_count)
        for episode in range(episode_count)
    ]
    count = len(rows)
    height, width, anchors = 24, 26, 3
    image_rng = np.random.default_rng(7000 + env_seed_start)
    clean = image_rng.integers(
        0, 256, size=(count, height, width, 3), dtype=np.uint8
    )
    q_true = np.empty((count, anchors, 2), dtype=np.float32)
    for row, (_, frame) in enumerate(rows):
        q_true[row] = np.asarray(
            ((4 + frame % 2, 5), (12, 10 + frame % 3), (21, 18)),
            dtype=np.float32,
        )
    valid = np.ones((count, anchors), dtype=bool)
    arrays: dict[str, np.ndarray] = {
        "rgb": np.zeros_like(clean),
        "obs": image_rng.normal(size=(count, 7)).astype(np.float32),
        "action": image_rng.normal(size=(count, 4)).astype(np.float32),
        "task_name": np.full(count, "assembly-v3"),
        "task_index": np.full(count, 3, dtype=np.int32),
        "episode_id": np.asarray([episode for episode, _ in rows], dtype=np.int32),
        "env_seed": np.asarray(
            [env_seed_start + episode for episode, _ in rows], dtype=np.int64
        ),
        "frame_id": np.asarray([frame for _, frame in rows], dtype=np.int32),
        "task_generation_seed": np.asarray(91, dtype=np.int64),
        "keypoints_xyz": image_rng.normal(size=(count, anchors, 3)).astype(
            np.float32
        ),
        "q_true": q_true,
        "q_nominal": q_true + np.float32(0.5),
        "valid_true": valid,
        "valid_nominal": valid.copy(),
        "camera_step_frame": np.full(count, 2, dtype=np.int32),
        "observation_action_alignment": np.asarray("pre_action"),
        "actual_camera_trajectory_mode": np.asarray(mode),
        "actual_camera_trajectory_phase": np.tile(
            np.linspace(0.0, 1.0, frame_count, dtype=np.float32), episode_count
        ).reshape(frame_count, episode_count).ravel(),
        "anchor_dropout_active": np.ones((count, anchors), dtype=bool),
        "anchor_recovery_onset": np.ones((count, anchors), dtype=bool),
        "anchor_recovery_segment": np.ones((count, anchors), dtype=bool),
        "measurement_valid": np.zeros((count, anchors), dtype=bool),
        "shared_scalar": np.asarray("preserved"),
    }
    if include_camera_only:
        arrays["rgb_camera_only"] = clean
    else:
        arrays["rgb"] = clean
    return arrays


def _permute_rows(
    source: dict[str, np.ndarray], permutation: np.ndarray
) -> dict[str, np.ndarray]:
    count = int(source["rgb"].shape[0])
    return {
        name: value[permutation].copy()
        if value.ndim > 0 and value.shape[0] == count
        else value.copy()
        for name, value in source.items()
    }


def _build(
    sources: list[dict[str, np.ndarray]], seed: int = 14
) -> dict[str, np.ndarray]:
    return randomized.build_posterior_randomized_arrays(
        sources,
        seed=seed,
        minimum_duration=10,
        maximum_duration=15,
        recovery_length=10,
        onset_phase_min=0.1,
        onset_phase_max=0.3,
    )


def test_deterministic_and_independent_of_source_and_row_order() -> None:
    step = _source(env_seed_start=100, mode="step")
    drift = _source(
        env_seed_start=200,
        mode="drift",
        include_camera_only=False,
    )
    first = _build([step, drift])
    second = _build([step, drift])
    reordered = _build(
        [
            _permute_rows(drift, np.arange(drift["rgb"].shape[0])[::-1]),
            _permute_rows(step, np.roll(np.arange(step["rgb"].shape[0]), 7)),
        ]
    )

    assert set(first) == set(second) == set(reordered)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
        np.testing.assert_array_equal(first[name], reordered[name])

    assert len(np.unique(first["episode_id"])) == 4
    assert set(first["source_episode_id"].tolist()) == {0, 1}
    assert set(first["randomized_source_trajectory_mode"].tolist()) == {
        "step",
        "drift",
    }
    assert first["observation_action_alignment"].shape == ()
    assert first["shared_scalar"].shape == ()
    assert first["actual_camera_trajectory_mode"].shape == (120,)


def test_output_satisfies_gate_loader_and_schedule_contract(tmp_path: Path) -> None:
    result = _build(
        [
            _source(env_seed_start=100, mode="step"),
            _source(
                env_seed_start=200,
                mode="drift",
                include_camera_only=False,
            ),
        ]
    )
    required = {
        *randomized.GATE_FRAME_KEYS,
        *randomized.GATE_SCALAR_KEYS,
        *randomized.SCHEDULE_KEYS,
        *randomized.GENERATION_ANNOTATION_KEYS,
        "rgb",
        "rgb_camera_only",
        "source_episode_id",
    }
    assert required <= set(result)
    path = tmp_path / "randomized.npz"
    np.savez_compressed(path, **result)
    loaded = _load_gate_dataset(
        path,
        observed_image_key="rgb",
        canonical_image_key="rgb_camera_only",
    )
    assert loaded.observed_images.shape == result["rgb"].shape

    active = result["anchor_dropout_active"]
    inactive_rows = ~np.any(active, axis=1)
    active_rows = np.any(active, axis=1)
    np.testing.assert_array_equal(
        result["rgb"][inactive_rows], result["rgb_camera_only"][inactive_rows]
    )
    assert bool(
        np.all(
            np.any(
                result["rgb"][active_rows]
                != result["rgb_camera_only"][active_rows],
                axis=(1, 2, 3),
            )
        )
    )
    np.testing.assert_array_equal(
        result["measurement_valid"], result["valid_nominal"] & ~active
    )

    for episode in np.unique(result["episode_id"]):
        rows = np.flatnonzero(result["episode_id"] == episode)
        rows = rows[np.argsort(result["frame_id"][rows])]
        anchor = int(result["randomized_anchor_index"][rows[0]])
        duration = int(result["randomized_duration"][rows[0]])
        assert np.all(result["randomized_anchor_index"][rows] == anchor)
        assert np.all(result["randomized_duration"][rows] == duration)
        positions = np.flatnonzero(active[rows, anchor])
        np.testing.assert_array_equal(
            positions, np.arange(positions[0], positions[0] + duration)
        )
        recovery = np.flatnonzero(result["anchor_recovery_segment"][rows, anchor])
        np.testing.assert_array_equal(
            recovery, np.arange(positions[-1] + 1, positions[-1] + 11)
        )
        onset = np.flatnonzero(result["anchor_recovery_onset"][rows, anchor])
        np.testing.assert_array_equal(onset, recovery[:1])
        event_rows = rows[np.concatenate((positions, recovery))]
        assert bool(np.all(result["valid_true"][event_rows, anchor]))
        assert bool(np.all(result["valid_nominal"][event_rows, anchor]))
        q_true = result["q_true"][event_rows, anchor]
        assert bool(np.all((q_true[:, 0] >= 0) & (q_true[:, 0] < 26)))
        assert bool(np.all((q_true[:, 1] >= 0) & (q_true[:, 1] < 24)))
        assert not bool(
            np.any(
                result["anchor_dropout_active"][rows]
                & result["anchor_recovery_segment"][rows]
            )
        )


def test_randomizes_all_requested_axes_across_many_physical_episodes() -> None:
    source = _source(
        env_seed_start=1000,
        mode="drift",
        episode_count=80,
        frame_count=30,
    )
    result = _build([source], seed=8)
    first_rows = np.asarray(
        [
            np.flatnonzero(result["episode_id"] == episode)[0]
            for episode in np.unique(result["episode_id"])
        ]
    )
    assert set(result["randomized_corruption_family"][first_rows]) == set(
        CORRUPTION_TYPES
    )
    assert len(set(result["randomized_anchor_index"][first_rows].tolist())) > 1
    assert len(set(result["randomized_onset_frame"][first_rows].tolist())) > 1
    assert len(set(result["randomized_duration"][first_rows].tolist())) > 1
    assert set(result["randomized_severity_tier"][first_rows]) == set(
        randomized.SEVERITY_TIERS
    )
    phase = result["randomized_onset_relative_phase"][first_rows]
    assert bool(np.all((phase >= 0.1) & (phase <= 0.3)))


def test_family_severity_and_event_rng_are_fixed_with_moving_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(
        env_seed_start=500,
        mode="step",
        episode_count=1,
        frame_count=25,
    )
    records: list[tuple[str, dict[str, object], int, tuple[float, float]]] = []

    def fake_corruption(
        image: np.ndarray,
        anchor_xy: np.ndarray,
        rng: np.random.Generator,
        *,
        corruption_type: str,
        **parameters: object,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        draw = int(rng.integers(0, 2**31))
        records.append(
            (
                corruption_type,
                parameters,
                draw,
                (float(anchor_xy[0]), float(anchor_xy[1])),
            )
        )
        changed = image.copy()
        changed[0, 0, 0] ^= np.uint8(255)
        return changed, np.ones(image.shape[:2], dtype=bool), {}

    monkeypatch.setattr(randomized, "apply_visual_trust_corruption", fake_corruption)
    randomized.build_posterior_randomized_arrays(
        source,
        seed=4,
        minimum_duration=10,
        maximum_duration=10,
        recovery_length=10,
        onset_phase_min=0.0,
        onset_phase_max=0.25,
    )
    assert len(records) == 10
    assert len({record[0] for record in records}) == 1
    assert all(record[1] == records[0][1] for record in records)
    assert len({record[2] for record in records}) == 1


def test_rejects_no_op_and_overlapping_source_physical_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(env_seed_start=300, mode="step")

    def no_op(
        image: np.ndarray,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        return image.copy(), np.zeros(image.shape[:2], dtype=bool), {}

    monkeypatch.setattr(randomized, "apply_visual_trust_corruption", no_op)
    with pytest.raises(ValueError, match="no pixel change"):
        _build([source])

    monkeypatch.undo()
    duplicate = {name: value.copy() for name, value in source.items()}
    duplicate["actual_camera_trajectory_mode"] = np.asarray("drift")
    with pytest.raises(ValueError, match="physical identities must be disjoint"):
        _build([source, duplicate])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"minimum_duration": 5}, "at least 10"),
        ({"minimum_duration": 11}, "multiples of 5"),
        ({"maximum_duration": 16}, "multiples of 5"),
        ({"recovery_length": 5}, "at least 10"),
        ({"recovery_length": 11}, "multiple of 5"),
    ],
)
def test_rejects_schedule_values_outside_five_frame_lattice(
    kwargs: dict[str, int], message: str
) -> None:
    source = _source(env_seed_start=700, mode="step")
    defaults = {
        "minimum_duration": 10,
        "maximum_duration": 20,
        "recovery_length": 10,
    }
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=message):
        randomized.build_posterior_randomized_arrays(source, **defaults)


def test_cli_defaults_obey_schedule_contract() -> None:
    args = cli.build_parser().parse_args(
        ["--input", "source.npz", "--output", "output.npz"]
    )
    assert (args.minimum_duration, args.maximum_duration) == (10, 20)
    assert args.recovery_length == 10


def test_cli_supports_repeated_inputs_and_writes_output(tmp_path: Path) -> None:
    first_path = tmp_path / "step.npz"
    second_path = tmp_path / "drift.npz"
    output_path = tmp_path / "derived" / "posterior.npz"
    np.savez_compressed(first_path, **_source(env_seed_start=10, mode="step"))
    np.savez_compressed(
        second_path,
        **_source(
            env_seed_start=20,
            mode="drift",
            include_camera_only=False,
        ),
    )

    assert (
        cli.main(
            [
                "--input",
                str(first_path),
                "--input",
                str(second_path),
                "--output",
                str(output_path),
                "--seed",
                "7",
                "--min-duration",
                "10",
                "--max-duration",
                "15",
                "--recovery-length",
                "10",
                "--onset-phase-min",
                "0.0",
                "--onset-phase-max",
                "0.5",
            ]
        )
        == 0
    )
    with np.load(output_path, allow_pickle=False) as data:
        assert data["rgb"].shape[0] == 120
        assert len(np.unique(data["episode_id"])) == 4
        assert set(data["source_episode_id"].tolist()) == {0, 1}
        assert set(data["randomized_source_trajectory_mode"].tolist()) == {
            "step",
            "drift",
        }
