from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from fcpg.lewm_policy_token_bridge import (
    BridgeAlignedSplit,
    PolicyTokenBridge,
    assert_disjoint_physical_episodes,
    bridge_training_loss,
    load_bridge_source_split,
    matched_geometry_hidden_dim,
    reconstruct_native_policy_targets,
    token_metrics,
    trainable_parameter_count,
)
from fcpg.lewm_horizon_oracle_bridge import build_horizon_pairs
from fcpg.lewm_residual_policy_token_bridge import (
    GeometryResidualPolicyBridge,
    load_residual_bridge_source_split,
    matched_geometry_residual_hidden_dim,
    reconstruct_residual_policy_tokens,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_source_fixture(tmp_path: Path) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    source = tmp_path / "source.npz"
    source_tasks = np.asarray(
        [
            "pick-place-v3",
            "button-press-v3",
            "reach-v3",
            "push-v3",
            "pick-place-v3",
            "reach-v3",
            "push-v3",
            "button-press-v3",
        ]
    )
    xyz = rng.normal(size=(8, 3, 3)).astype(np.float32)
    np.savez(source, task_name=source_tasks, keypoints_xyz=xyz)
    corrector = tmp_path / "corrector.pt"
    corrector.write_bytes(b"corrector-checkpoint")

    policy_cache = tmp_path / "policy_cache"
    policy_cache.mkdir()
    sampled = rng.normal(size=(8, 3, 256)).astype(np.float32)
    np.save(policy_cache / "sampled_visual.npy", sampled)
    (policy_cache / "metadata.json").write_text(
        json.dumps(
            {
                "source_data": str(source),
                "checkpoint": str(corrector),
                "sample_count": 8,
                "num_keypoints": 3,
                "visual_dim": 256,
                "image_width": 224,
                "image_height": 224,
            }
        ),
        encoding="utf-8",
    )

    posterior = tmp_path / "posterior"
    posterior.mkdir()
    target_index = np.asarray([5, 2, 7, 3, 4, 0], dtype=np.int64)
    state = rng.normal(size=(6, 3, 192)).astype(np.float32)
    np.savez(posterior / "deployment_inputs.npz", teacher_forced_factual_prior=state)
    np.savez(
        posterior / "evaluation_metadata.npz",
        task_name=source_tasks[target_index],
        physical_episode_sha256=np.asarray(
            [f"{index:064x}" for index in range(6)]
        ),
        source_episode_id=np.arange(6, dtype=np.int64),
        source_frame_id=np.arange(10, 16, dtype=np.int64),
        target_raw_index=target_index,
    )
    (posterior / "metadata.json").write_text(
        json.dumps(
            {
                "contract": {
                    "frame_skip": 5,
                    "observation_action_alignment": "pre_action",
                },
                "inputs": {
                    "dataset": {"path": str(source), "sha256": _sha256(source)},
                    "canonical_correction_cache": {
                        "corrector_checkpoint_sha256": _sha256(corrector)
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return posterior, policy_cache, sampled, xyz


def test_load_bridge_source_split_filters_and_joins_by_raw_index(
    tmp_path: Path,
) -> None:
    posterior, policy_cache, sampled, xyz = _build_source_fixture(tmp_path)
    split = load_bridge_source_split(
        posterior,
        policy_cache,
        tasks=("pick-place-v3", "reach-v3", "push-v3"),
    )
    # Row 2 of the posterior metadata is a button example and disappears.
    assert split.target_raw_index.tolist() == [5, 2, 3, 4, 0]
    assert split.state.shape == (5, 3, 192)
    assert split.geometry().shape == (5, 3, 6)
    np.testing.assert_array_equal(split.sampled_visual, sampled[[5, 2, 3, 4, 0]])
    np.testing.assert_array_equal(split.keypoints_xyz, xyz[[5, 2, 3, 4, 0]])
    assert split.provenance["physical_episode_count"] == 5


class _FakeGroundingAdapter(nn.Module):
    def forward_sampled(
        self,
        sampled: torch.Tensor,
        proprio: torch.Tensor,
        *,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        return sampled + reliability.unsqueeze(-1) * proprio.mean(
            dim=-1, keepdim=True
        )


class _FakePolicy(nn.Module):
    num_keypoints = 3
    vision_dim = 256
    model_dim = 256

    def __init__(self) -> None:
        super().__init__()
        self.grounding_adapter = _FakeGroundingAdapter()

    def _proprio(self, xyz: torch.Tensor) -> torch.Tensor:
        identity = torch.eye(3, dtype=xyz.dtype, device=xyz.device)[None]
        return torch.cat((xyz, identity.expand(xyz.shape[0], -1, -1)), dim=-1)


def test_reconstruct_native_policy_targets_uses_actual_adapter_contract(
    tmp_path: Path,
) -> None:
    posterior, policy_cache, _, _ = _build_source_fixture(tmp_path)
    source = load_bridge_source_split(
        posterior,
        policy_cache,
        tasks=("pick-place-v3", "reach-v3", "push-v3"),
    )
    policy = _FakePolicy()
    aligned = reconstruct_native_policy_targets(
        policy,
        source,
        device=torch.device("cpu"),
        batch_size=2,
    )
    with torch.no_grad():
        expected = policy.grounding_adapter.forward_sampled(
            torch.from_numpy(source.sampled_visual),
            policy._proprio(torch.from_numpy(source.keypoints_xyz)),
            reliability=torch.ones(source.sample_count, 3),
        ).numpy()
    np.testing.assert_allclose(aligned.target, expected, atol=0.0, rtol=0.0)
    assert aligned.target.shape == (5, 3, 256)
    assert aligned.provenance["target_role"] == "native_pre_memory_grounded_policy_token"


def _aligned(physical: list[str]) -> BridgeAlignedSplit:
    count = len(physical)
    return BridgeAlignedSplit(
        state=np.zeros((count, 3, 192), dtype=np.float32),
        geometry=np.zeros((count, 3, 6), dtype=np.float32),
        target=np.zeros((count, 3, 256), dtype=np.float32),
        task_name=np.asarray(["pick-place-v3"] * count),
        physical_episode_sha256=np.asarray(physical),
        source_episode_id=np.asarray([str(index) for index in range(count)]),
        source_frame_id=np.arange(count),
        target_raw_index=np.arange(count),
        provenance={},
    )


def test_physical_episode_split_overlap_is_rejected() -> None:
    shared = "a" * 64
    with pytest.raises(ValueError, match="share physical episodes"):
        assert_disjoint_physical_episodes(
            _aligned([shared]),
            _aligned([shared]),
        )


def test_bridge_loss_metrics_and_parameter_matching() -> None:
    rng = np.random.default_rng(19)
    inputs = rng.normal(size=(12, 3, 192)).astype(np.float32)
    targets = rng.normal(size=(12, 3, 256)).astype(np.float32)
    model = PolicyTokenBridge(input_dim=192, output_dim=256, hidden_dim=64)
    model.set_normalization(inputs, targets)
    input_tensor = torch.from_numpy(inputs)
    target_tensor = torch.from_numpy(targets)
    prediction = model(input_tensor)
    assert prediction.shape == target_tensor.shape
    loss, parts = bridge_training_loss(
        model, input_tensor, target_tensor, cosine_weight=0.1
    )
    assert torch.isfinite(loss)
    assert set(parts) == {"normalized_mse", "cosine_distance"}
    loss.backward()

    main_count = trainable_parameter_count(model)
    geometry_hidden = matched_geometry_hidden_dim(
        target_parameter_count=main_count,
        output_dim=256,
    )
    geometry = PolicyTokenBridge(
        input_dim=6, output_dim=256, hidden_dim=geometry_hidden
    )
    geometry_count = trainable_parameter_count(geometry)
    assert abs(main_count - geometry_count) / main_count < 0.01

    metrics = token_metrics(targets, targets, np.asarray(["a", "b"] * 6))
    assert metrics["token_mse"] == pytest.approx(0.0)
    assert metrics["cosine_similarity"] == pytest.approx(1.0)
    assert metrics["task_macro_token_mse"] == pytest.approx(0.0)


def _build_residual_fixture(tmp_path: Path) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(31)
    source = tmp_path / "residual_source.npz"
    source_tasks = np.asarray(
        [
            "pick-place-v3",
            "button-press-v3",
            "reach-v3",
            "push-v3",
            "pick-place-v3",
            "reach-v3",
            "push-v3",
            "button-press-v3",
        ]
    )
    xyz = rng.normal(size=(8, 3, 3)).astype(np.float32)
    np.savez(source, task_name=source_tasks, keypoints_xyz=xyz)
    corrector = tmp_path / "residual_corrector.pt"
    corrector.write_bytes(b"residual-corrector")
    policy_cache = tmp_path / "residual_policy_cache"
    policy_cache.mkdir()
    sampled = rng.normal(size=(8, 3, 256)).astype(np.float32)
    np.save(policy_cache / "sampled_visual.npy", sampled)
    (policy_cache / "metadata.json").write_text(
        json.dumps(
            {
                "source_data": str(source),
                "checkpoint": str(corrector),
                "sample_count": 8,
                "num_keypoints": 3,
                "visual_dim": 256,
            }
        ),
        encoding="utf-8",
    )

    posterior = tmp_path / "residual_posterior"
    posterior.mkdir()
    target = np.asarray([4, 5, 6, 7], dtype=np.int64)
    history = np.asarray([[0, 0, 0], [2, 2, 2], [3, 3, 3], [1, 1, 1]])
    np.savez(
        posterior / "deployment_inputs.npz",
        teacher_forced_factual_prior=rng.normal(size=(4, 3, 192)).astype(np.float32),
    )
    np.savez(
        posterior / "offline_labels.npz",
        canonical_measurement_history=rng.normal(size=(4, 3, 3, 192)).astype(np.float32),
        canonical_current=rng.normal(size=(4, 3, 192)).astype(np.float32),
    )
    raw_episode = np.asarray(["p", "b", "r", "u", "p", "r", "u", "b"])
    np.savez(
        posterior / "evaluation_metadata.npz",
        task_name=source_tasks[target],
        physical_episode_sha256=np.asarray([f"{index + 20:064x}" for index in range(4)]),
        source_episode_id=raw_episode,
        source_frame_id=np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
        history_raw_indices=history,
        target_raw_index=target,
    )
    (posterior / "metadata.json").write_text(
        json.dumps(
            {
                "contract": {
                    "frame_skip": 5,
                    "observation_action_alignment": "pre_action",
                },
                "inputs": {
                    "dataset": {"path": str(source), "sha256": _sha256(source)},
                    "canonical_correction_cache": {
                        "corrector_checkpoint_sha256": _sha256(corrector)
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return posterior, policy_cache, sampled, xyz


def test_residual_bridge_loader_and_native_token_pair(tmp_path: Path) -> None:
    posterior, policy_cache, sampled, xyz = _build_residual_fixture(tmp_path)
    source = load_residual_bridge_source_split(
        posterior,
        policy_cache,
        tasks=("pick-place-v3", "reach-v3", "push-v3"),
    )
    assert source.history_raw_index.tolist() == [0, 2, 3]
    assert source.target_raw_index.tolist() == [4, 5, 6]
    np.testing.assert_array_equal(source.last_sampled_visual, sampled[[0, 2, 3]])
    np.testing.assert_array_equal(source.target_sampled_visual, sampled[[4, 5, 6]])
    np.testing.assert_array_equal(source.last_valid_xyz, xyz[[0, 2, 3]])
    np.testing.assert_array_equal(source.current_xyz, xyz[[4, 5, 6]])

    policy = _FakePolicy()
    aligned = reconstruct_residual_policy_tokens(
        policy, source, device=torch.device("cpu"), batch_size=2
    )
    assert aligned.last_valid_token.shape == (3, 3, 256)
    assert aligned.geometry_hold_exact_token.shape == (3, 3, 256)
    assert aligned.target_token.shape == (3, 3, 256)
    assert not np.array_equal(aligned.last_valid_token, aligned.target_token)
    assert not np.array_equal(
        aligned.geometry_hold_exact_token, aligned.last_valid_token
    )

    oracle = load_residual_bridge_source_split(
        posterior,
        policy_cache,
        tasks=("pick-place-v3", "reach-v3", "push-v3"),
        embedding_source="oracle_current",
    )
    assert oracle.provenance["predicted_embedding"] == "oracle_current"
    assert oracle.provenance["privileged_oracle_embedding"] is True
    assert not np.array_equal(oracle.predicted_embedding, source.predicted_embedding)


def test_geometry_residual_bridge_is_hold_last_at_initialization() -> None:
    target_parameters = 41664
    hidden = matched_geometry_residual_hidden_dim(
        target_parameter_count=target_parameters,
        token_dim=256,
        num_keypoints=3,
    )
    model = GeometryResidualPolicyBridge(hidden_dim=hidden)
    assert abs(model.trainable_parameter_count - target_parameters) / target_parameters < 0.01
    rng = np.random.default_rng(41)
    last_xyz = rng.normal(size=(6, 3, 3)).astype(np.float32)
    current_xyz = rng.normal(size=(6, 3, 3)).astype(np.float32)
    last_token = rng.normal(size=(6, 3, 256)).astype(np.float32)
    target = rng.normal(size=(6, 3, 256)).astype(np.float32)
    model.set_normalization(
        last_xyz=last_xyz,
        current_xyz=current_xyz,
        token_delta=target - last_token,
    )
    prediction = model(
        torch.from_numpy(last_token),
        torch.from_numpy(current_xyz),
        torch.from_numpy(last_xyz),
    )
    np.testing.assert_allclose(prediction.detach().numpy(), last_token)


def test_horizon_pairs_use_exact_offsets_without_crossing_episodes() -> None:
    task = np.asarray(["pick"] * 10 + ["pick"] * 10 + ["reach"] * 10)
    episode = np.asarray([0] * 10 + [1] * 10 + [0] * 10)
    frame = np.tile(np.arange(10), 3)
    start, target, horizon, pair_task = build_horizon_pairs(
        task,
        episode,
        frame,
        tasks=("pick", "reach"),
        horizons=(1, 2),
        raw_steps_per_decision=2,
    )
    assert start.shape == target.shape == horizon.shape == pair_task.shape
    assert np.array_equal(frame[target] - frame[start], horizon * 2)
    assert np.array_equal(task[start], task[target])
    assert np.array_equal(episode[start], episode[target])
    # Per episode: 8 H1 pairs and 6 H2 pairs; there are three episodes.
    assert int(np.sum(horizon == 1)) == 24
    assert int(np.sum(horizon == 2)) == 18
