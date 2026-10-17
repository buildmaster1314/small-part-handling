"""Core regression tests for FCPG geometry and data plumbing."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import fcpg.action_chunk_policy as action_chunk_policy

from fcpg.modules import sample_image_patches_at_pixels
from fcpg.mujoco_camera import sample_camera_pose_offset
from fcpg.observations import apply_keypoint_square_occlusion, select_policy_observation
from fcpg.sequence_data import ProjectionSequenceDataset, compute_normalization_stats
from fcpg.transformer_grounding import (
    AnchorConditionedResidualTransformer,
    sample_local_feature_tokens,
)
from scripts.train_bc_policy import make_split


def _legacy_crop(image: np.ndarray, center_xy: np.ndarray, radius: int) -> np.ndarray:
    cx, cy = (int(round(float(value))) for value in center_xy)
    side = 2 * radius + 1
    patch = np.zeros((side, side, image.shape[-1]), dtype=image.dtype)
    x0, y0 = cx - radius, cy - radius
    src_x0, src_y0 = max(x0, 0), max(y0, 0)
    src_x1 = min(cx + radius + 1, image.shape[1])
    src_y1 = min(cy + radius + 1, image.shape[0])
    dst_x0, dst_y0 = src_x0 - x0, src_y0 - y0
    patch[dst_y0 : dst_y0 + src_y1 - src_y0, dst_x0 : dst_x0 + src_x1 - src_x0] = image[
        src_y0:src_y1, src_x0:src_x1
    ]
    return patch


def test_batched_patch_sampler_matches_legacy_integer_crop() -> None:
    rng = np.random.default_rng(7)
    image_np = rng.random((2, 13, 17, 3), dtype=np.float32)
    points_np = np.asarray(
        [[[0.2, 0.4], [8.1, 6.2]], [[16.0, 12.0], [4.7, 9.4]]],
        dtype=np.float32,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = torch.from_numpy(image_np).permute(0, 3, 1, 2).to(device)
    points = torch.from_numpy(points_np).to(device)

    actual = (
        sample_image_patches_at_pixels(image, points, radius=2)
        .permute(0, 1, 3, 4, 2)
        .cpu()
        .numpy()
    )
    expected = np.stack(
        [
            [
                _legacy_crop(image_np[b], points_np[b, k], 2)
                for k in range(points_np.shape[1])
            ]
            for b in range(2)
        ]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_camera_pose_sampler_respects_fixed_magnitudes() -> None:
    translation, axis, rotation = sample_camera_pose_offset(
        np.random.default_rng(3),
        translation_cm_max=4.0,
        rotation_deg_max=5.0,
        translation_cm_min=4.0,
        rotation_deg_min=5.0,
    )
    np.testing.assert_allclose(np.linalg.norm(translation) * 100.0, 4.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(axis), 1.0, atol=1e-12)
    np.testing.assert_allclose(abs(rotation), 5.0, atol=1e-12)


def test_camera_pose_sampler_clean_probability() -> None:
    translation, axis, rotation = sample_camera_pose_offset(
        np.random.default_rng(5),
        translation_cm_max=4.0,
        rotation_deg_max=5.0,
        clean_probability=1.0,
    )
    np.testing.assert_array_equal(translation, np.zeros(3))
    np.testing.assert_array_equal(axis, np.asarray([0.0, 0.0, 1.0]))
    assert rotation == 0.0


def test_task_episode_split_has_no_group_leakage(tmp_path) -> None:
    task_names = np.repeat(np.asarray(["reach-v3", "push-v3"]), 12)
    episode_ids = np.tile(np.repeat(np.arange(4), 3), 2)
    path = tmp_path / "split.npz"
    np.savez(
        path,
        task_name=task_names,
        episode_id=episode_ids,
    )
    with np.load(path, allow_pickle=False) as raw:
        split = make_split(raw, val_fraction=0.25, seed=9, split_unit="task-episode")
        groups = np.char.add(
            np.char.add(raw["task_name"].astype(str), ":"),
            raw["episode_id"].astype(str),
        )
        assert set(groups[split.train]).isdisjoint(set(groups[split.test]))
        assert set(raw["task_name"][split.test]) == {"reach-v3", "push-v3"}


def test_robot_observation_excludes_metaworld_object_goal_state() -> None:
    observation = np.arange(78, dtype=np.float32).reshape(2, 39)

    full = select_policy_observation(observation, "full")
    robot = select_policy_observation(observation, "robot")
    robot_goal = select_policy_observation(observation, "robot_goal")

    assert full.shape == (2, 39)
    assert robot.shape == (2, 4)
    assert robot_goal.shape == (2, 7)
    np.testing.assert_array_equal(robot, observation[:, :4])
    np.testing.assert_array_equal(robot_goal[:, :4], observation[:, :4])
    np.testing.assert_array_equal(robot_goal[:, 4:], observation[:, -3:])
    with pytest.raises(ValueError, match="unknown observation mode"):
        select_policy_observation(observation, "invalid")


def test_keypoint_square_occlusion_masks_valid_regions() -> None:
    image = np.full((8, 10, 3), 255, dtype=np.uint8)
    points = np.asarray([[2.0, 3.0], [9.0, 7.0], [np.nan, 1.0]], dtype=np.float32)

    masked = apply_keypoint_square_occlusion(image, points, radius_px=1, value=7)

    np.testing.assert_array_equal(masked[2:5, 1:4], np.full((3, 3, 3), 7, dtype=np.uint8))
    np.testing.assert_array_equal(masked[6:8, 8:10], np.full((2, 2, 3), 7, dtype=np.uint8))
    assert bool(np.all(image == 255))


def test_action_chunk_runtime_reliability_override(monkeypatch) -> None:
    class FakeVisionEncoder(nn.Module):
        def __init__(self, output_dim: int, **_: object) -> None:
            super().__init__()
            self.output_dim = output_dim

        def forward(self, rgb: torch.Tensor):
            batch = rgb.shape[0]
            feature_map = torch.ones(batch, self.output_dim, 8, 8, device=rgb.device)
            return SimpleNamespace(
                feature_map=feature_map,
                global_feature=feature_map.mean(dim=(-2, -1)),
            )

    class FakeFCPGAdapter(nn.Module):
        def forward(
            self,
            _rgb: torch.Tensor,
            q_nominal: torch.Tensor,
            _xyz: torch.Tensor,
            _obs: torch.Tensor,
        ) -> SimpleNamespace:
            return SimpleNamespace(
                q_corrected_px=q_nominal + 2.0,
                reliability=torch.full(q_nominal.shape[:2], 0.25, device=q_nominal.device),
            )

    monkeypatch.setattr(action_chunk_policy, "ViTDenseEncoder", FakeVisionEncoder)
    model = action_chunk_policy.GroundedActionChunkTransformer(
        mode="fcpg",
        obs_dim=4,
        action_dim=4,
        num_keypoints=3,
        obs_horizon=2,
        action_horizon=4,
        model_dim=32,
        vision_dim=16,
        encoder_layers=1,
        decoder_layers=1,
        heads=4,
        dropout=0.0,
        pretrained_vision=False,
        fcpg_adapter=FakeFCPGAdapter(),
    ).eval()
    q_nominal = torch.rand(1, 2, 3, 2) * 31.0
    batch = {
        "rgb": torch.randn(1, 2, 3, 32, 32),
        "obs": torch.randn(1, 2, 4),
        "q_nominal": q_nominal,
        "q_true": q_nominal,
        "keypoints_xyz": torch.randn(1, 2, 3, 3),
    }

    native = model.predict_action_chunk(batch)
    model.set_reliability_override("ones")
    overridden = model.predict_action_chunk(batch)

    torch.testing.assert_close(native.grounding_points_px, q_nominal + 2.0)
    torch.testing.assert_close(overridden.grounding_points_px, native.grounding_points_px)
    torch.testing.assert_close(native.reliability, torch.full_like(native.reliability, 0.25))
    torch.testing.assert_close(overridden.reliability, torch.ones_like(overridden.reliability))
    model.set_reliability_override("native")
    assert model.runtime_reliability_override is None
    with pytest.raises(ValueError, match="unknown reliability override"):
        model.set_reliability_override("invalid")


def test_robot_goal_policy_sends_only_robot_state_to_fcpg(monkeypatch) -> None:
    class FakeVisionEncoder(nn.Module):
        def __init__(self, output_dim: int, **_: object) -> None:
            super().__init__()
            self.output_dim = output_dim

        def forward(self, rgb: torch.Tensor):
            feature_map = torch.ones(
                rgb.shape[0], self.output_dim, 8, 8, device=rgb.device
            )
            return SimpleNamespace(
                feature_map=feature_map,
                global_feature=feature_map.mean(dim=(-2, -1)),
            )

    class FakeFCPGAdapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.grounding = SimpleNamespace(obs_dim=4)
            self.received_obs: torch.Tensor | None = None

        def forward(
            self,
            _rgb: torch.Tensor,
            q_nominal: torch.Tensor,
            _xyz: torch.Tensor,
            obs: torch.Tensor,
        ) -> SimpleNamespace:
            self.received_obs = obs.detach().clone()
            return SimpleNamespace(q_corrected_px=q_nominal, reliability=None)

    monkeypatch.setattr(action_chunk_policy, "ViTDenseEncoder", FakeVisionEncoder)
    adapter = FakeFCPGAdapter()
    model = action_chunk_policy.GroundedActionChunkTransformer(
        mode="fcpg",
        obs_dim=7,
        action_dim=4,
        num_keypoints=3,
        obs_horizon=2,
        action_horizon=4,
        model_dim=32,
        vision_dim=16,
        encoder_layers=1,
        decoder_layers=1,
        heads=4,
        dropout=0.0,
        pretrained_vision=False,
        fcpg_adapter=adapter,
    ).eval()
    policy_obs = torch.randn(1, 2, 7)
    q_nominal = torch.rand(1, 2, 3, 2) * 31.0
    model.predict_action_chunk(
        {
            "rgb": torch.randn(1, 2, 3, 32, 32),
            "obs": policy_obs,
            "q_nominal": q_nominal,
            "q_true": q_nominal,
            "keypoints_xyz": torch.randn(1, 2, 3, 3),
        }
    )

    assert adapter.received_obs is not None
    assert adapter.received_obs.shape == (2, 4)
    torch.testing.assert_close(adapter.received_obs, policy_obs.reshape(2, 7)[:, :4])


def test_fcpg_robot_observation_rejects_unknown_dimension_mismatch() -> None:
    adapter = SimpleNamespace(grounding=SimpleNamespace(obs_dim=5))
    with pytest.raises(ValueError, match="FCPG observation mismatch"):
        action_chunk_policy.fcpg_robot_observation(torch.zeros(2, 7), adapter)


def test_local_feature_tokens_preserve_anchor_grid_shape() -> None:
    feature_map = torch.arange(2 * 4 * 8 * 8, dtype=torch.float32).reshape(2, 4, 8, 8)
    points = torch.tensor(
        [
            [[16.0, 16.0], [24.0, 24.0]],
            [[20.0, 12.0], [8.0, 28.0]],
        ]
    )

    tokens, offsets = sample_local_feature_tokens(
        feature_map,
        points,
        image_size=(32, 32),
        radius_px=8.0,
        grid_size=3,
    )

    assert tokens.shape == (2, 2, 9, 4)
    assert offsets.shape == (9, 2)
    torch.testing.assert_close(offsets.min(dim=0).values, torch.tensor([-1.0, -1.0]))
    torch.testing.assert_close(offsets.max(dim=0).values, torch.tensor([1.0, 1.0]))


def test_anchor_transformer_outputs_bounded_residual_and_reliability() -> None:
    torch.manual_seed(0)
    module = AnchorConditionedResidualTransformer(
        visual_dim=16,
        obs_dim=4,
        num_keypoints=3,
        model_dim=32,
        local_grid_size=3,
        local_radius_px=8.0,
        local_layers=1,
        global_layers=1,
        heads=4,
        dropout=0.0,
        max_residual_px=12.0,
    ).eval()
    feature_map = torch.randn(2, 16, 14, 14)
    q_nominal = torch.tensor(
        [
            [[32.0, 32.0], [48.0, 36.0], [40.0, 56.0]],
            [[30.0, 34.0], [46.0, 38.0], [42.0, 54.0]],
        ]
    )
    keypoints = torch.randn(2, 3, 3)
    obs = torch.randn(2, 4)

    with torch.no_grad():
        output = module(feature_map, q_nominal, keypoints, obs, image_size=(128, 128))

    assert output.delta_px.shape == (2, 3, 2)
    assert output.reliability.shape == (2, 3)
    assert output.keypoint_latent.shape == (2, 3, 32)
    torch.testing.assert_close(output.q_corrected_px, q_nominal + output.delta_px)
    assert float(output.delta_px.abs().max()) <= 12.0
    assert bool(torch.all((output.reliability >= 0.0) & (output.reliability <= 1.0)))


def test_action_chunk_transformer_preserves_chunk_shape_and_backprop(monkeypatch) -> None:
    class FakeVisionEncoder(nn.Module):
        def __init__(self, output_dim: int, **_: object) -> None:
            super().__init__()
            self.output_dim = output_dim
            self.backbone_fully_frozen = True

        def forward(self, rgb: torch.Tensor):
            batch = rgb.shape[0]
            feature_map = torch.randn(batch, self.output_dim, 8, 8, device=rgb.device)
            global_feature = feature_map.mean(dim=(-2, -1))
            return SimpleNamespace(feature_map=feature_map, global_feature=global_feature)

    monkeypatch.setattr(action_chunk_policy, "ViTDenseEncoder", FakeVisionEncoder)
    model = action_chunk_policy.GroundedActionChunkTransformer(
        mode="geoprop",
        obs_dim=4,
        action_dim=4,
        num_keypoints=3,
        obs_horizon=2,
        action_horizon=8,
        model_dim=32,
        vision_dim=16,
        encoder_layers=1,
        decoder_layers=1,
        heads=4,
        dropout=0.0,
        pretrained_vision=False,
    )
    batch = {
        "rgb": torch.randn(2, 2, 3, 32, 32),
        "obs": torch.randn(2, 2, 4),
        "q_nominal": torch.rand(2, 2, 3, 2) * 31.0,
        "q_true": torch.rand(2, 2, 3, 2) * 31.0,
        "keypoints_xyz": torch.randn(2, 2, 3, 3),
        "action": torch.cat(
            [
                torch.rand(2, 8, 3) * 2.0 - 1.0,
                torch.randint(0, 2, (2, 8, 1)).float(),
            ],
            dim=-1,
        ),
        "action_mask": torch.ones(2, 8, dtype=torch.bool),
    }

    loss, metrics, output = model.compute_loss(batch)

    assert output.action.shape == (2, 8, 4)
    assert output.grounding_points_px.shape == (2, 2, 3, 2)
    assert set(metrics) == {"motion_loss", "gripper_loss"}
    loss.backward()
    assert model.action_queries.grad is not None


def test_projection_sequence_dataset_pads_within_episode(tmp_path) -> None:
    frames_per_episode = 5
    count = 2 * frames_per_episode
    episode_ids = np.repeat(np.arange(2), frames_per_episode)
    frame_ids = np.tile(np.arange(frames_per_episode), 2)
    action = np.repeat(frame_ids[:, None].astype(np.float32), 4, axis=1)
    path = tmp_path / "sequence.npz"
    np.savez(
        path,
        rgb=np.zeros((count, 4, 4, 3), dtype=np.uint8),
        obs=np.arange(count * 39, dtype=np.float32).reshape(count, 39),
        action=action,
        task_name=np.repeat(np.asarray(["pick-place-v3"]), count),
        episode_id=episode_ids,
        frame_id=frame_ids,
        keypoints_xyz=np.zeros((count, 3, 3), dtype=np.float32),
        q_nominal=np.zeros((count, 3, 2), dtype=np.float32),
        q_true=np.zeros((count, 3, 2), dtype=np.float32),
        valid_true=np.ones((count, 3), dtype=bool),
        valid_nominal=np.ones((count, 3), dtype=bool),
    )
    dataset = ProjectionSequenceDataset(
        path,
        np.arange(count),
        obs_mode="robot",
        obs_horizon=2,
        action_horizon=4,
    )

    first = dataset[0]
    last = dataset[4]
    assert len(dataset) == count
    assert first["rgb"].shape == (2, 3, 4, 4)
    assert first["obs"].shape == (2, 4)
    torch.testing.assert_close(first["obs"][0], first["obs"][1])
    torch.testing.assert_close(first["action"][:, 0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    torch.testing.assert_close(last["action"][:, 0], torch.tensor([4.0, 4.0, 4.0, 4.0]))
    torch.testing.assert_close(last["action_mask"], torch.tensor([True, False, False, False]))
    stats = compute_normalization_stats(path, np.arange(count), "robot")
    normalized_action = (action - stats.action_mean) / stats.action_std
    np.testing.assert_allclose(normalized_action.min(axis=0), -1.0)
    np.testing.assert_allclose(normalized_action.max(axis=0), 1.0)


def test_projection_sequence_dataset_reads_memmapped_vit_cache(tmp_path) -> None:
    count = 4
    data_path = tmp_path / "cached_sequence.npz"
    np.savez(
        data_path,
        rgb=np.zeros((count, 8, 8, 3), dtype=np.uint8),
        obs=np.zeros((count, 39), dtype=np.float32),
        action=np.zeros((count, 4), dtype=np.float32),
        task_name=np.repeat(np.asarray(["pick-place-v3"]), count),
        episode_id=np.zeros(count, dtype=np.int64),
        frame_id=np.arange(count),
        keypoints_xyz=np.zeros((count, 3, 3), dtype=np.float32),
        q_nominal=np.zeros((count, 3, 2), dtype=np.float32),
        q_true=np.zeros((count, 3, 2), dtype=np.float32),
        valid_true=np.ones((count, 3), dtype=bool),
        valid_nominal=np.ones((count, 3), dtype=bool),
    )
    cache = tmp_path / "vit_cache"
    cache.mkdir()
    np.save(cache / "backbone_features.npy", np.ones((count, 1, 8, 2, 2), dtype=np.float16))
    np.save(cache / "prefix.npy", np.ones((count, 8), dtype=np.float16))
    (cache / "metadata.json").write_text(
        json.dumps({"image_width": 8, "image_height": 8}),
        encoding="utf-8",
    )

    grounding_cache = tmp_path / "grounding_cache"
    grounding_cache.mkdir()
    np.save(
        grounding_cache / "q_corrected.npy",
        np.full((count, 3, 2), 2.0, dtype=np.float32),
    )
    np.save(
        grounding_cache / "reliability.npy",
        np.full((count, 3), 0.75, dtype=np.float32),
    )

    dataset = ProjectionSequenceDataset(
        data_path,
        np.arange(count),
        obs_mode="robot",
        obs_horizon=2,
        action_horizon=2,
        vision_cache=cache,
        grounding_cache=grounding_cache,
    )
    sample = dataset[1]

    assert "rgb" not in sample
    assert sample["vision_backbone_features"].shape == (2, 1, 8, 2, 2)
    assert sample["vision_prefix"].shape == (2, 8)
    torch.testing.assert_close(sample["image_size"], torch.tensor([8, 8]))
    torch.testing.assert_close(sample["fcpg_q_corrected"], torch.full((2, 3, 2), 2.0))
    torch.testing.assert_close(sample["fcpg_reliability"], torch.full((2, 3), 0.75))


def test_projection_sequence_dataset_rejects_mismatched_vision_cache(tmp_path) -> None:
    count = 2
    data_path = tmp_path / "sequence.npz"
    np.savez(
        data_path,
        rgb=np.zeros((count, 8, 8, 3), dtype=np.uint8),
        obs=np.zeros((count, 39), dtype=np.float32),
        action=np.zeros((count, 4), dtype=np.float32),
        task_name=np.repeat(np.asarray(["pick-place-v3"]), count),
        episode_id=np.zeros(count, dtype=np.int64),
        frame_id=np.arange(count),
        keypoints_xyz=np.zeros((count, 3, 3), dtype=np.float32),
        q_nominal=np.zeros((count, 3, 2), dtype=np.float32),
        q_true=np.zeros((count, 3, 2), dtype=np.float32),
        valid_true=np.ones((count, 3), dtype=bool),
        valid_nominal=np.ones((count, 3), dtype=bool),
    )
    cache = tmp_path / "vit_cache"
    cache.mkdir()
    np.save(cache / "backbone_features.npy", np.ones((count, 1, 8, 2, 2), dtype=np.float16))
    np.save(cache / "prefix.npy", np.ones((count, 8), dtype=np.float16))
    (cache / "metadata.json").write_text(
        json.dumps(
            {
                "source_data": str(tmp_path / "other.npz"),
                "image_width": 8,
                "image_height": 8,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="vision cache source"):
        ProjectionSequenceDataset(
            data_path,
            np.arange(count),
            obs_mode="robot",
            vision_cache=cache,
        )
