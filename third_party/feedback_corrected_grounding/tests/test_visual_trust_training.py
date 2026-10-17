from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fcpg.visual_trust_training import (
    VisualTrustSplit,
    build_auxiliary_descriptors,
    canonical_feature_rms,
    load_visual_trust_split,
    predict_visual_trust,
    temporal_feature_distance,
    threshold_for_clean_false_reject,
    train_visual_trust_head,
)


def test_build_auxiliary_descriptors_has_expected_current_only_values() -> None:
    nominal = np.asarray([[[10.0, 20.0]]], dtype=np.float32)
    corrected = np.asarray([[[13.0, 16.0]]], dtype=np.float32)
    result = build_auxiliary_descriptors(
        nominal,
        corrected,
        np.asarray([[True]]),
        image_width=101,
        image_height=51,
        patch_radius=5.0,
    )

    assert result.shape == (1, 1, 7)
    np.testing.assert_allclose(
        result[0, 0],
        [-0.74, -0.36, 0.6, -0.8, 1.0, 1.0, 1.0],
        atol=1e-6,
    )


def test_canonical_feature_rms_is_per_anchor() -> None:
    observed = np.zeros((2, 2, 4), dtype=np.float32)
    canonical = observed.copy()
    canonical[1, 0] = 2.0
    error = canonical_feature_rms(observed, canonical)
    np.testing.assert_array_equal(error, [[0.0, 0.0], [2.0, 0.0]])


def _write_split(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    count, anchors, channels = 6, 2, 4
    observed = np.zeros((count, anchors, channels), dtype=np.float16)
    canonical = observed.copy()
    observed[1:3, 0] = 1.0
    observed_path = tmp_path / "observed.npy"
    canonical_path = tmp_path / "canonical.npy"
    grounding_path = tmp_path / "q_corrected.npy"
    np.save(observed_path, observed)
    np.save(canonical_path, canonical)
    q = np.full((count, anchors, 2), 4.0, dtype=np.float32)
    np.save(grounding_path, q)

    active = np.zeros((count, anchors), dtype=bool)
    active[1:3, 0] = True
    recovery = np.zeros_like(active)
    recovery[3, 0] = True
    data_path = tmp_path / "data.npz"
    np.savez_compressed(
        data_path,
        rgb=np.zeros((count, 9, 9, 3), dtype=np.uint8),
        q_nominal=q,
        valid_nominal=np.ones((count, anchors), dtype=bool),
        task_name=np.asarray(["task"] * count),
        episode_id=np.zeros(count, dtype=np.int64),
        frame_id=np.arange(count, dtype=np.int64),
        corruption_active=active,
        corruption_family=np.asarray(
            ["", "square_occlusion", "square_occlusion", "", "", ""]
        ),
        corruption_recovery_onset=recovery,
    )
    return data_path, observed_path, canonical_path, grounding_path


def test_load_split_keeps_canonical_label_only_and_expands_family(tmp_path: Path) -> None:
    paths = _write_split(tmp_path)
    split = load_visual_trust_split(*paths, unsafe_rms_threshold=0.1)

    assert split.visual.shape == (6, 2, 4)
    assert split.auxiliary.shape == (6, 2, 7)
    assert split.unsafe[1, 0] and split.unsafe[2, 0]
    assert not split.safe_target[1, 0]
    assert split.clean[3, 0]
    assert set(split.family.reshape(-1)) == {"square_occlusion"}
    assert split.recovery_onset[3, 0]


def test_clean_only_loader_needs_no_corruption_annotations(tmp_path: Path) -> None:
    data_path, observed_path, canonical_path, grounding_path = _write_split(tmp_path)
    with np.load(data_path, allow_pickle=False) as raw:
        arrays = {
            name: raw[name]
            for name in raw.files
            if not name.startswith("corruption_")
        }
    clean_path = tmp_path / "clean.npz"
    np.savez_compressed(clean_path, **arrays)

    split = load_visual_trust_split(
        clean_path,
        observed_path,
        canonical_path,
        grounding_path,
        clean_only=True,
    )
    assert not bool(split.active.any())
    assert not bool(split.recovery_onset.any())
    assert set(split.family.reshape(-1)) == {"clean"}


def test_temporal_distance_resets_at_episode_boundaries() -> None:
    visual = np.asarray([0.0, 2.0, 10.0, 13.0], dtype=np.float32).reshape(4, 1, 1)
    split = VisualTrustSplit(
        visual=visual,
        auxiliary=np.zeros((4, 1, 1), dtype=np.float32),
        safe_target=np.ones((4, 1), dtype=np.float32),
        unsafe=np.zeros((4, 1), dtype=bool),
        clean=np.ones((4, 1), dtype=bool),
        early_reopen=np.zeros((4, 1), dtype=bool),
        valid=np.ones((4, 1), dtype=bool),
        active=np.zeros((4, 1), dtype=bool),
        recovery_onset=np.zeros((4, 1), dtype=bool),
        task=np.asarray([["t"], ["t"], ["t"], ["t"]]),
        episode=np.asarray([[0], [0], [1], [1]]),
        frame=np.asarray([[0], [1], [0], [1]]),
        family=np.asarray([["a"], ["a"], ["b"], ["b"]]),
        canonical_rms=np.zeros((4, 1), dtype=np.float32),
    )
    np.testing.assert_allclose(temporal_feature_distance(split)[:, 0], [0, 2, 0, 3])


def test_threshold_calibrates_lower_clean_tail() -> None:
    trust = np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    clean = np.ones_like(trust, dtype=bool)
    valid = np.ones_like(trust, dtype=bool)
    assert threshold_for_clean_false_reject(
        trust, clean, valid, target_rate=0.25
    ) == pytest.approx(0.2)
    assert float(np.mean(trust < 0.2)) == pytest.approx(0.25)


def test_tiny_head_training_uses_visual_signal() -> None:
    rng = np.random.default_rng(7)
    safe = rng.normal(1.0, 0.05, size=(128, 1, 3)).astype(np.float32)
    unsafe = rng.normal(-1.0, 0.05, size=(128, 1, 3)).astype(np.float32)
    visual = np.concatenate((safe, unsafe), axis=0)
    target = np.concatenate(
        (np.ones((128, 1)), np.zeros((128, 1))), axis=0
    ).astype(np.float32)
    auxiliary = np.zeros((256, 1, 2), dtype=np.float32)
    valid = np.ones((256, 1), dtype=bool)
    device = torch.device("cpu")

    model = train_visual_trust_head(
        visual,
        auxiliary,
        target,
        valid,
        seed=11,
        steps=100,
        batch_size=64,
        learning_rate=3e-3,
        device=device,
        visual_enabled=True,
    )
    prediction = predict_visual_trust(
        model,
        visual,
        auxiliary,
        batch_size=64,
        device=device,
        visual_enabled=True,
    )
    assert float(prediction[:128].mean()) > 0.8
    assert float(prediction[128:].mean()) < 0.2


def test_auxiliary_only_control_retains_parameter_matched_visual_width() -> None:
    visual = np.ones((8, 1, 5), dtype=np.float32)
    auxiliary = np.zeros((8, 1, 2), dtype=np.float32)
    target = np.asarray([[0.0], [1.0]] * 4, dtype=np.float32)
    valid = np.ones((8, 1), dtype=bool)
    model = train_visual_trust_head(
        visual,
        auxiliary,
        target,
        valid,
        seed=3,
        steps=1,
        batch_size=4,
        learning_rate=1e-3,
        device=torch.device("cpu"),
        visual_enabled=False,
    )
    assert model.visual_dim == visual.shape[-1]
    prediction = predict_visual_trust(
        model,
        visual,
        auxiliary,
        batch_size=4,
        device=torch.device("cpu"),
        visual_enabled=False,
    )
    assert prediction.shape == target.shape


@pytest.mark.parametrize("bad_threshold", [0.0, -1.0, float("inf")])
def test_load_split_rejects_bad_label_threshold(
    tmp_path: Path, bad_threshold: float
) -> None:
    paths = _write_split(tmp_path)
    with pytest.raises(ValueError, match="unsafe_rms_threshold"):
        load_visual_trust_split(*paths, unsafe_rms_threshold=bad_threshold)
