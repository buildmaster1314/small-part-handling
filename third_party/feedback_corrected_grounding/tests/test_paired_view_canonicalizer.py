from __future__ import annotations

import numpy as np
import pytest
import torch

from fcpg.paired_view_canonicalizer import (
    CONTEXT_DIM,
    PairedViewAnchorCache,
    PairedViewCanonicalizer,
    build_anchor_context,
    load_paired_view_anchor_cache,
    relative_camera_features,
)


def _cache() -> PairedViewAnchorCache:
    frames, anchors, latent = 2, 3, 4
    q_nominal = np.full((frames, anchors, 2), 8.0, dtype=np.float32)
    q_dynamic = q_nominal + 2.0
    eye = np.eye(4, dtype=np.float32)[None].repeat(frames, axis=0)
    true = eye.copy()
    true[:, 0, 3] = 0.1
    values = np.arange(frames * anchors * latent, dtype=np.float32).reshape(
        frames, anchors, latent
    )
    return PairedViewAnchorCache(
        task_name=np.asarray(["pick", "reach"]),
        episode_id=np.asarray([0, 0]),
        frame_id=np.asarray([0, 1]),
        q_nominal=q_nominal,
        q_true=q_dynamic,
        q_dynamic_corrected=q_dynamic,
        q_clean_corrected=q_nominal,
        valid_nominal=np.ones((frames, anchors), dtype=bool),
        valid_true=np.ones((frames, anchors), dtype=bool),
        keypoints_xyz=np.ones((frames, anchors, 3), dtype=np.float32),
        true_extrinsic=true,
        base_extrinsic=eye,
        trajectory_phase=np.ones(frames, dtype=np.float32),
        camera_intervention_onset=np.zeros(frames, dtype=bool),
        latent_dynamic_corrected=values,
        latent_clean_corrected=values,
        latent_clean_nominal=values,
        latent_dynamic_oracle=values,
        image_size=(17, 17),
        trajectory_mode="step",
        source_data_sha256="abc",
    )


def test_context_separates_deployment_and_privileged_camera_features() -> None:
    cache = _cache()
    latent_only = build_anchor_context(cache, mode="latent_only", clean=False)
    residual = build_anchor_context(cache, mode="residual_context", clean=False)
    oracle = build_anchor_context(cache, mode="camera_oracle", clean=False)
    clean_oracle = build_anchor_context(cache, mode="camera_oracle", clean=True)
    assert latent_only.shape == (2, 3, CONTEXT_DIM)
    assert not latent_only.any()
    assert np.any(residual[..., :15])
    assert not residual[..., 15:].any()
    assert np.any(oracle[..., 15:27])
    assert not clean_oracle[..., 15:27].any()


def test_cache_round_trip_preserves_phase_contract(tmp_path) -> None:
    cache = _cache()
    path = tmp_path / "cache.npz"
    payload = {
        name: getattr(cache, name)
        for name in (
            "task_name",
            "episode_id",
            "frame_id",
            "q_nominal",
            "q_true",
            "q_dynamic_corrected",
            "q_clean_corrected",
            "valid_nominal",
            "valid_true",
            "keypoints_xyz",
            "true_extrinsic",
            "base_extrinsic",
            "trajectory_phase",
            "camera_intervention_onset",
            "latent_dynamic_corrected",
            "latent_clean_corrected",
            "latent_clean_nominal",
            "latent_dynamic_oracle",
        )
    }
    np.savez_compressed(
        path,
        schema_version=np.asarray(1),
        protocol=np.asarray("fcpg_paired_camera_anchor_cache_v1"),
        image_width=np.asarray(cache.image_size[0]),
        image_height=np.asarray(cache.image_size[1]),
        trajectory_mode=np.asarray(cache.trajectory_mode),
        source_data_sha256=np.asarray(cache.source_data_sha256),
        **payload,
    )
    loaded = load_paired_view_anchor_cache(path)
    assert np.array_equal(loaded.trajectory_phase, cache.trajectory_phase)
    assert np.array_equal(
        loaded.camera_intervention_onset, cache.camera_intervention_onset
    )


def test_relative_camera_features_are_zero_at_identity() -> None:
    eye = np.eye(4)[None]
    assert np.array_equal(relative_camera_features(eye, eye), np.zeros((1, 12)))


def test_canonicalizer_starts_as_exact_identity() -> None:
    model = PairedViewCanonicalizer(latent_dim=4, hidden_dim=8)
    latent = torch.randn(5, 4)
    context = torch.randn(5, CONTEXT_DIM)
    assert torch.equal(model(latent, context), latent)


def test_canonicalizer_rejects_bad_context_shape() -> None:
    model = PairedViewCanonicalizer(latent_dim=4, hidden_dim=8)
    with pytest.raises(ValueError, match="context shape"):
        model(torch.zeros(2, 4), torch.zeros(2, CONTEXT_DIM - 1))
