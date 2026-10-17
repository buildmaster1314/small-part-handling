from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fcpg.lewm_provenance import sha256_file
from fcpg.lewm_sequence_data import LeWMSequenceDataset, frame_identity_sha256
from fcpg.paired_camera_step import compose_paired_camera_step


def _dataset_arrays(*, intervention: bool) -> dict[str, np.ndarray]:
    frames = np.tile(np.arange(4, dtype=np.int32), 2)
    episodes = np.repeat(np.arange(2, dtype=np.int32), 4)
    count = len(frames)
    keypoints = 2
    control_points = 1
    values = np.arange(count, dtype=np.float32)
    camera_value = 2.0 if intervention else 1.0
    translation = 0.04 if intervention else 0.0
    rotation = 5.0 if intervention else 0.0

    identity = np.broadcast_to(np.eye(4, dtype=np.float32), (count, 4, 4)).copy()
    true_extrinsic = identity.copy()
    true_extrinsic[:, 0, 3] = translation
    q_nominal = np.full((count, keypoints, 2), 0.5, dtype=np.float32)
    control_q_nominal = np.full(
        (count, control_points, 2), 0.5, dtype=np.float32
    )
    arrays: dict[str, np.ndarray] = {
        "task_name": np.asarray(["pick-place-v3"] * count),
        "task_index": np.zeros(count, dtype=np.int32),
        "episode_id": episodes,
        "env_seed": 1000 + episodes,
        "frame_id": frames,
        "obs": np.broadcast_to(values[:, None], (count, 6)).copy(),
        "action": np.broadcast_to(values[:, None], (count, 4)).copy(),
        "reward": values.copy(),
        "success": frames == 3,
        "keypoints_xyz": np.broadcast_to(
            values[:, None, None], (count, keypoints, 3)
        ).copy(),
        "control_keypoints_xyz": np.broadcast_to(
            values[:, None, None], (count, control_points, 3)
        ).copy(),
        "q_nominal": q_nominal,
        "control_q_nominal": control_q_nominal,
        "valid_nominal": np.ones((count, keypoints), dtype=bool),
        "control_valid_nominal": np.ones((count, control_points), dtype=bool),
        "camera_intrinsic": np.broadcast_to(
            np.eye(3, dtype=np.float32), (count, 3, 3)
        ).copy(),
        "base_extrinsic": identity.copy(),
        "nominal_extrinsic": identity.copy(),
        "translation_cm": np.full(count, 4.0, dtype=np.float32),
        "translation_direction": np.asarray(["x"] * count),
        "yaw_deg": np.full(count, 5.0, dtype=np.float32),
        "action_source": np.asarray(["scripted"] * count),
        "observation_action_alignment": np.asarray("pre_action"),
        "task_generation_seed": np.asarray(11, dtype=np.int64),
        "rng_seed": np.asarray(12, dtype=np.int64),
        "independent_rng_streams": np.asarray(True),
        "rgb": np.full((count, 4, 4, 3), int(camera_value * 50), dtype=np.uint8),
        "q_true": np.full(
            (count, keypoints, 2), camera_value, dtype=np.float32
        ),
        "control_q_true": np.full(
            (count, control_points, 2), camera_value, dtype=np.float32
        ),
        "true_extrinsic": true_extrinsic,
        "actual_camera_translation_xyz": np.broadcast_to(
            np.asarray([translation, 0.0, 0.0], dtype=np.float32), (count, 3)
        ).copy(),
        "actual_camera_rotation_axis": np.broadcast_to(
            np.asarray([0.0, 1.0, 0.0], dtype=np.float32), (count, 3)
        ).copy(),
        "actual_camera_rotation_deg": np.full(count, rotation, dtype=np.float32),
        "valid_true": np.ones((count, keypoints), dtype=bool),
        "control_valid_true": np.ones((count, control_points), dtype=bool),
        "observer_visibility": np.full(
            (count, keypoints), camera_value, dtype=np.float32
        ),
        "pixel_error": np.full(
            (count, keypoints), camera_value, dtype=np.float32
        ),
        "q_true_inside_patch": np.full(
            (count, keypoints), not intervention, dtype=bool
        ),
        "actual_camera_translation_cm_min": np.asarray(
            translation * 100.0, dtype=np.float32
        ),
        "actual_camera_translation_cm_max": np.asarray(
            translation * 100.0, dtype=np.float32
        ),
        "actual_camera_rotation_deg_min": np.asarray(rotation, dtype=np.float32),
        "actual_camera_rotation_deg_max": np.asarray(rotation, dtype=np.float32),
        "actual_camera_randomization": np.asarray(
            "fixed_drift" if intervention else "clean"
        ),
        "actual_camera_clean_probability": np.asarray(
            0.0 if intervention else 1.0, dtype=np.float32
        ),
    }
    return arrays


def _write_dataset(path: Path, *, intervention: bool) -> dict[str, np.ndarray]:
    arrays = _dataset_arrays(intervention=intervention)
    np.savez(path, **arrays)
    return arrays


def _write_cache(
    cache: Path,
    *,
    data_path: Path,
    arrays: dict[str, np.ndarray],
    checkpoint: Path,
    correction_value: float,
) -> np.ndarray:
    cache.mkdir()
    q_corrected = np.full_like(
        arrays["q_nominal"], correction_value, dtype=np.float32
    )
    q_path = cache / "q_corrected.npy"
    np.save(q_path, q_corrected, allow_pickle=False)
    metadata = {
        "version": 3,
        "schema_version": 3,
        "corrector_arch": "synthetic-residual-only",
        "checkpoint_schema_version": 1,
        "obs_mode": "rgb",
        "lewm_upstream_commit": "synthetic",
        "source_data": str(data_path.resolve()),
        "source_data_sha256": sha256_file(data_path),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "sample_count": int(arrays["obs"].shape[0]),
        "num_keypoints": int(arrays["q_nominal"].shape[1]),
        "frame_identity_sha256": frame_identity_sha256(
            arrays["episode_id"], arrays["frame_id"]
        ),
        "q_corrected_sha256": sha256_file(q_path),
        "q_true_usage": "offline_metrics_only",
        "has_reliability": False,
    }
    (cache / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )
    return q_corrected


@pytest.fixture
def paired_inputs(tmp_path: Path) -> dict[str, object]:
    reference = tmp_path / "reference.npz"
    intervention = tmp_path / "intervention.npz"
    reference_arrays = _write_dataset(reference, intervention=False)
    intervention_arrays = _write_dataset(intervention, intervention=True)
    checkpoint = tmp_path / "corrector.pt"
    checkpoint.write_bytes(b"same-frozen-corrector")
    reference_cache = tmp_path / "reference_cache"
    intervention_cache = tmp_path / "intervention_cache"
    reference_q = _write_cache(
        reference_cache,
        data_path=reference,
        arrays=reference_arrays,
        checkpoint=checkpoint,
        correction_value=0.75,
    )
    intervention_q = _write_cache(
        intervention_cache,
        data_path=intervention,
        arrays=intervention_arrays,
        checkpoint=checkpoint,
        correction_value=2.25,
    )
    return {
        "reference": reference,
        "intervention": intervention,
        "reference_arrays": reference_arrays,
        "intervention_arrays": intervention_arrays,
        "reference_cache": reference_cache,
        "intervention_cache": intervention_cache,
        "reference_q": reference_q,
        "intervention_q": intervention_q,
    }


def _compose(
    tmp_path: Path,
    inputs: dict[str, object],
    *,
    name: str,
    switch_frame: int | None = 2,
    switch_frame_min: int | None = None,
    switch_frame_max: int | None = None,
    switch_seed: int = 0,
):
    return compose_paired_camera_step(
        reference_data=inputs["reference"],
        intervention_data=inputs["intervention"],
        reference_correction_cache=inputs["reference_cache"],
        intervention_correction_cache=inputs["intervention_cache"],
        **({} if switch_frame is None else {"switch_frame": switch_frame}),
        switch_frame_min=switch_frame_min,
        switch_frame_max=switch_frame_max,
        switch_seed=switch_seed,
        output_data=tmp_path / f"{name}.npz",
        output_correction_cache=tmp_path / f"{name}_cache",
    )


def test_fixed_switch_selects_camera_and_correction_at_boundary(
    tmp_path: Path, paired_inputs: dict[str, object]
) -> None:
    result = _compose(tmp_path, paired_inputs, name="fixed")
    with np.load(result.data_path, allow_pickle=False) as output:
        frame_id = output["frame_id"]
        use_intervention = frame_id >= 2
        np.testing.assert_array_equal(
            output["q_true"][:, 0, 0],
            np.where(use_intervention, 2.0, 1.0),
        )
        np.testing.assert_array_equal(
            output["camera_step_event"], frame_id == 2
        )
        assert output["camera_step_event"].dtype == np.bool_
        np.testing.assert_array_equal(
            output["camera_step_frame"], np.full(8, 2, dtype=np.int32)
        )
        assert output["camera_step_frame"].dtype == np.int32
        assert output["camera_step_schedule_mode"].item() == "fixed"
        assert int(output["camera_step_switch_frame"]) == 2
        assert output["actual_camera_randomization"].item() == "paired_step"
        assert float(output["actual_camera_translation_cm_min"]) == pytest.approx(0.0)
        assert float(output["actual_camera_translation_cm_max"]) == pytest.approx(4.0)
        assert float(output["actual_camera_rotation_deg_min"]) == pytest.approx(0.0)
        assert float(output["actual_camera_rotation_deg_max"]) == pytest.approx(5.0)
        assert float(output["actual_camera_clean_probability"]) == pytest.approx(0.5)

    corrected = np.load(
        result.correction_cache / "q_corrected.npy", allow_pickle=False
    )
    expected = np.where(
        (np.tile(np.arange(4), 2) >= 2)[:, None, None], 2.25, 0.75
    )
    np.testing.assert_array_equal(corrected, np.broadcast_to(expected, corrected.shape))

    metadata = json.loads(result.data_metadata_path.read_text(encoding="utf-8"))
    assert metadata["selection"]["schedule"]["mode"] == "fixed"
    assert metadata["selection"]["schedule"]["fixed_frame"] == 2
    assert len(metadata["selection"]["schedule"]["per_episode"]) == 2
    cache_metadata = json.loads(
        (result.correction_cache / "metadata.json").read_text(encoding="utf-8")
    )
    assert cache_metadata["source_data_sha256"] == sha256_file(result.data_path)
    assert cache_metadata["q_corrected_sha256"] == sha256_file(
        result.correction_cache / "q_corrected.npy"
    )

    dataset = LeWMSequenceDataset(
        result.data_path,
        history_size=2,
        rollout_steps=2,
        correction_cache=result.correction_cache,
    )
    assert len(dataset) == 2
    assert "q_corrected" in dataset[0]


def test_seeded_range_is_deterministic_and_records_exact_events(
    tmp_path: Path, paired_inputs: dict[str, object]
) -> None:
    first = _compose(
        tmp_path,
        paired_inputs,
        name="range_a",
        switch_frame=None,
        switch_frame_min=1,
        switch_frame_max=2,
        switch_seed=731,
    )
    second = _compose(
        tmp_path,
        paired_inputs,
        name="range_b",
        switch_frame=None,
        switch_frame_min=1,
        switch_frame_max=2,
        switch_seed=731,
    )
    with np.load(first.data_path, allow_pickle=False) as left, np.load(
        second.data_path, allow_pickle=False
    ) as right:
        np.testing.assert_array_equal(left["camera_step_frame"], right["camera_step_frame"])
        np.testing.assert_array_equal(left["camera_step_event"], right["camera_step_event"])
        assert int(left["camera_step_switch_frame"]) == -1
        assert left["camera_step_schedule_mode"].item() == "range"
        assert bool(
            ((left["camera_step_frame"] >= 1) & (left["camera_step_frame"] <= 2)).all()
        )
        np.testing.assert_array_equal(
            left["camera_step_event"],
            left["frame_id"] == left["camera_step_frame"],
        )
        for episode in np.unique(left["episode_id"]):
            rows = left["episode_id"] == episode
            assert int(left["camera_step_event"][rows].sum()) == 1

    metadata = json.loads(first.data_metadata_path.read_text(encoding="utf-8"))
    schedule = metadata["selection"]["schedule"]
    assert schedule["frame_min"] == 1
    assert schedule["frame_max"] == 2
    assert schedule["seed"] == 731
    assert all(1 <= item["switch_frame"] <= 2 for item in schedule["per_episode"])


def test_rejects_physical_pairing_mismatch(
    tmp_path: Path, paired_inputs: dict[str, object]
) -> None:
    intervention = paired_inputs["intervention"]
    with np.load(intervention, allow_pickle=False) as raw:
        arrays = {key: np.array(raw[key], copy=True) for key in raw.files}
    arrays["action"][0, 0] += 1.0
    np.savez(intervention, **arrays)

    with pytest.raises(ValueError, match="mismatched keys: action"):
        _compose(tmp_path, paired_inputs, name="bad_pair")


def test_rejects_different_correction_checkpoints(
    tmp_path: Path, paired_inputs: dict[str, object]
) -> None:
    other_checkpoint = tmp_path / "other_corrector.pt"
    other_checkpoint.write_bytes(b"different-frozen-corrector")
    _write_cache(
        tmp_path / "other_intervention_cache",
        data_path=paired_inputs["intervention"],
        arrays=paired_inputs["intervention_arrays"],
        checkpoint=other_checkpoint,
        correction_value=2.25,
    )
    inputs = dict(paired_inputs)
    inputs["intervention_cache"] = tmp_path / "other_intervention_cache"

    with pytest.raises(ValueError, match="different checkpoint paths"):
        _compose(tmp_path, inputs, name="bad_checkpoint")


@pytest.mark.parametrize("switch_frame", [0, 4])
def test_rejects_boundary_without_both_camera_branches(
    tmp_path: Path,
    paired_inputs: dict[str, object],
    switch_frame: int,
) -> None:
    with pytest.raises(
        ValueError, match="must leave reference and intervention frames"
    ):
        _compose(
            tmp_path,
            paired_inputs,
            name=f"boundary_{switch_frame}",
            switch_frame=switch_frame,
        )


def test_refuses_to_overwrite_any_output(
    tmp_path: Path, paired_inputs: dict[str, object]
) -> None:
    _compose(tmp_path, paired_inputs, name="sealed")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _compose(tmp_path, paired_inputs, name="sealed")
