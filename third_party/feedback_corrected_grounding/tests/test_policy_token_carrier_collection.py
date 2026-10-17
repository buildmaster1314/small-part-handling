from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import fcpg.policy_token_carrier_collection as carrier
from fcpg.lewm_counterfactual_runtime import FrameRecord
from fcpg.lewm_p1_native_fresh_collection import FreshnessInventory


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _contract() -> dict[str, object]:
    return json.loads(
        Path("config/lewm/eval/p1_native_complementary_fresh.json").read_text(
            encoding="utf-8"
        )
    )


def _inventory() -> FreshnessInventory:
    return FreshnessInventory(
        {}, {"p1_fcpg_training": {_sha("legacy-observed")}}
    )


class _FrozenPolicy(nn.Module):
    obs_horizon = 2

    def __init__(self) -> None:
        super().__init__()
        self.eval()


class _IdentityGrounding(nn.Module):
    def forward_sampled(
        self,
        sampled: torch.Tensor,
        proprio: torch.Tensor,
        *,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        del proprio, reliability
        return sampled


class _ConstantVision(nn.Module):
    def forward(self, rgb: torch.Tensor) -> SimpleNamespace:
        count = int(rgb.shape[0])
        marker = rgb[:, :1].mean(dim=(2, 3), keepdim=True)
        feature = marker.expand(count, 256, 2, 2).contiguous()
        return SimpleNamespace(
            feature_map=feature,
            global_feature=marker.reshape(count, 1).expand(count, 256).contiguous(),
        )


class _CompactPolicy(nn.Module):
    obs_horizon = 2
    num_keypoints = 3
    vision_dim = 256

    def __init__(self) -> None:
        super().__init__()
        self.vision_encoder = _ConstantVision()
        self.grounding_adapter = _IdentityGrounding()
        self.eval()

    def _proprio(self, xyz: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*xyz.shape[:2], 6), dtype=xyz.dtype, device=xyz.device)

    def _observation_tokens(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        count, frames = batch["rgb"].shape[:2]
        flat = batch["rgb"].reshape(count * frames, 3, 4, 4)
        vision = self.vision_encoder(flat)
        marker = vision.global_feature.reshape(count, frames, 256)
        grounded = marker[:, :, None].expand(count, frames, 3, 256)
        memory = torch.cat(
            (
                torch.zeros(count, frames, 2, 256),
                grounded,
            ),
            dim=2,
        )
        points = batch["q_nominal"]
        reliability = torch.ones(count, frames, 3)
        return memory, points, reliability


class _FakeEnv:
    def __init__(self) -> None:
        self.data = SimpleNamespace(
            qpos=np.asarray([0.0, 1.0], dtype=np.float64),
            qvel=np.asarray([0.0, 0.0], dtype=np.float64),
        )
        self._target_pos = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
        self._last_rand_vec = np.asarray([0.4, 0.5, 0.6], dtype=np.float64)
        self.action_space = SimpleNamespace(
            low=np.full(4, -1.0, dtype=np.float32),
            high=np.full(4, 1.0, dtype=np.float32),
        )
        self.step_index = 0
        self.closed = False

    def step(self, action: np.ndarray):
        del action
        self.step_index += 1
        return np.full(39, self.step_index, np.float32), 0.0, False, False, {}

    def close(self) -> None:
        self.closed = True


class _Runtime:
    def __init__(self) -> None:
        self.env = _FakeEnv()
        self.expert = SimpleNamespace(
            get_action=lambda observation: np.asarray(
                [float(np.asarray(observation)[0]) * 0.001, 0.0, 0.0, 1.0],
                dtype=np.float32,
            )
        )
        self.observation = np.zeros(39, dtype=np.float32)

    def record_current(self, observation: np.ndarray) -> FrameRecord:
        marker = float(np.asarray(observation)[0])
        return FrameRecord(
            rgb=np.full((3, 4, 4), marker, np.float32),
            proprio=np.full(4, marker, np.float32),
            keypoints_xyz=np.full((3, 3), marker + 0.2, np.float32),
            q_nominal=np.full((3, 2), marker + 0.3, np.float32),
            valid_nominal=np.ones(3, dtype=bool),
            q_true=np.full((3, 2), marker + 0.4, np.float32),
            valid_true=np.ones(3, dtype=bool),
        )


def _compact_from_pairs(
    policy: nn.Module,
    pairs: list[tuple[FrameRecord, FrameRecord]]
    | tuple[tuple[FrameRecord, FrameRecord], ...],
    *,
    device: object,
) -> carrier.CompactPolicyPairs:
    del policy, device
    count = len(pairs)
    marker = np.asarray(
        [[record.proprio[0] for record in pair] for pair in pairs], np.float32
    )

    def expand(tail: tuple[int, ...], offset: float = 0.0) -> np.ndarray:
        return np.broadcast_to(
            marker.reshape(count, 2, *([1] * len(tail))) + np.float32(offset),
            (count, 2, *tail),
        ).copy()

    return carrier.CompactPolicyPairs(
        obs=expand((4,)),
        q_nominal=expand((3, 2), 0.3),
        keypoints_xyz=expand((3, 3), 0.2),
        sampled_visual=expand((3, 256), 0.4),
        global_visual=expand((256,), 0.5),
        image_size=np.full((count, 2), 4, dtype=np.int64),
        fcpg_q_corrected=expand((3, 2), 0.6),
        fcpg_reliability=np.ones((count, 2, 3), np.float32),
        native_anchor=expand((3, 256), 0.7),
    )


def test_real_compact_extractor_emits_policy_compatible_image_size() -> None:
    runtime = _Runtime()
    first = runtime.record_current(np.zeros(39, np.float32))
    second = runtime.record_current(np.ones(39, np.float32))
    second = FrameRecord(
        rgb=np.ones((3, 4, 4), np.float32),
        proprio=second.proprio,
        keypoints_xyz=second.keypoints_xyz,
        q_nominal=second.q_nominal,
        valid_nominal=second.valid_nominal,
        q_true=second.q_true,
        valid_true=second.valid_true,
    )
    compact = carrier.extract_compact_policy_pairs(
        _CompactPolicy(), ((first, second),), device="cpu"
    )
    assert compact.image_size.shape == (1, 2)
    assert compact.image_size.tolist() == [[4, 4]]
    assert compact.native_anchor.shape == (1, 2, 3, 256)
    assert compact.native_anchor[0, 0, 0, 0] == pytest.approx(0.0)
    assert compact.native_anchor[0, 1, 0, 0] == pytest.approx(1.0)

def test_collector_binds_h3_h8_and_builds_causal_pre_drop_context() -> None:
    plan = carrier.build_fresh_episode_plans(_contract())[0]
    runtime = _Runtime()

    def branch_builder(center: np.ndarray, **_: object) -> SimpleNamespace:
        actions = np.broadcast_to(center[None], (8, *center.shape)).copy()
        for branch in range(8):
            actions[branch, :, 1] = branch * 0.01
        return SimpleNamespace(actions=actions)

    row = carrier.collect_policy_carrier_episode(
        plan,
        inventory=_inventory(),
        policy=_FrozenPolicy(),
        device="cpu",
        runtime_factory=lambda *_: runtime,
        branch_bank_builder=branch_builder,
        snapshot_capture=lambda env, **_: env.step_index,
        snapshot_restore=lambda env, snapshot: setattr(env, "step_index", snapshot),
        pair_extractor=_compact_from_pairs,
    )

    assert runtime.env.closed
    assert row.endpoint.obs.shape == (8 * 6, 2, 4)
    # Branch-major H3..H8 endpoint markers.  The two slots are raw-adjacent.
    assert row.endpoint.obs[0, :, 0].tolist() == pytest.approx([39.0, 40.0])
    assert row.endpoint.obs[5, :, 0].tolist() == pytest.approx([64.0, 65.0])
    assert row.endpoint.native_anchor[0, :, 0, 0].tolist() == pytest.approx(
        [39.7, 40.7]
    )
    # H-1,H0,H1 are visible states.  H2 is a separate visible measurement and
    # never enters the history used to predict itself.
    assert row.pre_drop_history_token[0, :, 0, 0].tolist() == pytest.approx(
        [20.7, 25.7, 30.7]
    )
    assert row.pre_drop_visible_token[0, 0, 0] == pytest.approx(35.7)
    assert row.pre_drop_action_context.shape == (8, 3, 20)
    # First action context is source H-1->H0; the other two are branch H0->H2.
    assert row.pre_drop_action_context[0, 0, 0] == pytest.approx(0.020)
    assert row.pre_drop_action_context[0, 1, 0] == pytest.approx(0.025)
    assert row.pre_drop_action_context[0, 2, 0] == pytest.approx(0.030)
    assert row.pre_drop_visible_visual.shape == (8, 3, 256)
    assert row.branch_future_action.shape == (8, 6, 20)
    assert row.same_state_prefix is False


def test_same_state_collector_branches_only_after_common_h2() -> None:
    plan = carrier.build_fresh_episode_plans(_contract())[0]
    runtime = _Runtime()

    def branch_builder(center: np.ndarray, **_: object) -> SimpleNamespace:
        actions = np.broadcast_to(center[None], (8, *center.shape)).copy()
        for branch in range(8):
            actions[branch, :, 1] = branch * 0.01
        return SimpleNamespace(actions=actions)

    row = carrier.collect_policy_carrier_episode(
        plan,
        inventory=_inventory(),
        policy=_FrozenPolicy(),
        device="cpu",
        runtime_factory=lambda *_: runtime,
        branch_bank_builder=branch_builder,
        snapshot_capture=lambda env, **_: env.step_index,
        snapshot_restore=lambda env, snapshot: setattr(env, "step_index", snapshot),
        pair_extractor=_compact_from_pairs,
        same_state_prefix=True,
    )

    assert row.same_state_prefix is True
    assert row.endpoint.obs[0, :, 0].tolist() == pytest.approx([39.0, 40.0])
    assert row.endpoint.obs[5, :, 0].tolist() == pytest.approx([64.0, 65.0])
    assert row.pre_drop_history_token[0, :, 0, 0].tolist() == pytest.approx(
        [20.7, 25.7, 30.7]
    )
    assert row.pre_drop_visible_token[0, 0, 0] == pytest.approx(35.7)
    np.testing.assert_array_equal(
        row.pre_drop_history_token,
        np.broadcast_to(row.pre_drop_history_token[:1], row.pre_drop_history_token.shape),
    )
    np.testing.assert_array_equal(
        row.pre_drop_action_context,
        np.broadcast_to(row.pre_drop_action_context[:1], row.pre_drop_action_context.shape),
    )
    np.testing.assert_array_equal(
        row.pre_drop_visible_token,
        np.broadcast_to(row.pre_drop_visible_token[:1], row.pre_drop_visible_token.shape),
    )
    assert row.branch_future_action[0, 0, 1] == pytest.approx(0.0)
    assert row.branch_future_action[7, 0, 1] == pytest.approx(0.07)


def _minimal_bundle() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    count = 30 * 8 * 6
    episode = np.asarray([f"episode:{index // 48}" for index in range(count)])
    physical = np.asarray([f"physical:{index // 48}" for index in range(count)])
    branch = np.tile(np.repeat(np.arange(8), 6), 30)
    horizon = np.tile(np.arange(3, 9), 30 * 8)
    row_id = np.asarray(
        [
            carrier._row_id(str(uid), int(branch_value), int(horizon_value))
            for uid, branch_value, horizon_value in zip(
                episode.tolist(), branch.tolist(), horizon.tolist(), strict=True
            )
        ]
    )
    identity = {
        "policy_row_id": row_id,
        "episode_uid": episode,
        "physical_episode_id": physical,
        "reset_physical_state_goal_digest": np.asarray(
            [_sha(f"reset:{index // 48}") for index in range(count)]
        ),
        "observed_state_signature": np.asarray(
            [_sha(f"observed:{index // 48}") for index in range(count)]
        ),
        "branch_index": branch,
        "horizon": horizon,
    }
    deployment = {
        **identity,
        "policy_obs": np.zeros((count, 2, 4), np.float32),
        "policy_q_nominal": np.zeros((count, 2, 3, 2), np.float32),
        "policy_keypoints_xyz": np.zeros((count, 2, 3, 3), np.float32),
        "policy_sampled_visual": np.zeros((count, 2, 3, 256), np.float32),
        "policy_global_visual": np.zeros((count, 2, 256), np.float32),
        "policy_image_size": np.full((count, 2), 224, np.int64),
        "policy_fcpg_q_corrected": np.zeros((count, 2, 3, 2), np.float32),
        "policy_fcpg_reliability": np.ones((count, 2, 3), np.float32),
        "pre_drop_history_token": np.zeros((count, 3, 3, 256), np.float32),
        "pre_drop_action_context": np.zeros((count, 3, 20), np.float32),
        "pre_drop_visible_token": np.zeros((count, 3, 256), np.float32),
        "pre_drop_visible_visual": np.zeros((count, 3, 256), np.float32),
        "policy_future_action": np.zeros((count, 20), np.float32),
        "same_state_prefix": np.asarray([True], dtype=bool),
    }
    labels = {
        **identity,
        "policy_target_anchor": np.zeros((count, 2, 3, 256), np.float32),
    }
    return deployment, labels


def test_prediction_attachment_requires_exact_identity_and_separates_target() -> None:
    deployment, labels = _minimal_bundle()
    count = len(deployment["policy_row_id"])
    deployment = carrier.attach_frozen_recursive_inputs(
        deployment,
        source_row_id=deployment["policy_row_id"],
        future_action=np.zeros((count, 20), np.float32),
        geometry_token=np.zeros((count, 3, 256), np.float32),
    )
    prediction = np.ones((count, 3, 256), np.float32)
    attached = carrier.attach_aligned_predictions(
        deployment,
        prediction,
        prediction_row_id=deployment["policy_row_id"],
        pre_drop_raw_prior_token=np.zeros((count, 3, 256), np.float32),
        raw_factual_predictions=prediction + 1.0,
        matched_no_action_predictions=prediction + 2.0,
        wrong_action_predictions=np.zeros((count, 7, 3, 256), np.float32),
        predictor_seed=15801,
        predictor_checkpoint_sha256=_sha("predictor"),
    )
    assert "policy_target_anchor" not in attached
    assert attached["policy_predicted_anchor"].shape == (count, 2, 3, 256)
    np.testing.assert_array_equal(
        attached["policy_predicted_anchor"][:, 0],
        attached["policy_predicted_anchor"][:, 1],
    )
    np.testing.assert_array_equal(
        attached["policy_predicted_anchor"],
        attached["policy_predicted_anchor_complementary_factual"],
    )
    assert attached["policy_predicted_anchor_wrong_action"].shape == (
        count,
        7,
        2,
        3,
        256,
    )
    assert attached["pre_drop_prior_is_causal"].tolist() == [True]
    carrier.validate_policy_carrier_arrays(attached, labels, require_predictions=True)

    with pytest.raises(PermissionError, match="same-replay collected action"):
        carrier.attach_frozen_recursive_inputs(
            deployment,
            source_row_id=deployment["policy_row_id"],
            future_action=np.ones((count, 20), np.float32),
            geometry_token=np.zeros((count, 3, 256), np.float32),
        )

    wrong = np.asarray(deployment["policy_row_id"]).copy()
    wrong[[0, 1]] = wrong[[1, 0]]
    with pytest.raises(PermissionError, match="identities/order"):
        carrier.attach_aligned_predictions(
            deployment,
            prediction,
            prediction_row_id=wrong,
            pre_drop_raw_prior_token=np.zeros((count, 3, 256), np.float32),
            raw_factual_predictions=prediction,
            matched_no_action_predictions=prediction,
            wrong_action_predictions=np.zeros((count, 7, 3, 256), np.float32),
            predictor_seed=15801,
            predictor_checkpoint_sha256=_sha("predictor"),
        )

    leaked = dict(attached)
    leaked["q_true"] = np.zeros((count, 2, 3, 2), np.float32)
    with pytest.raises(ValueError, match="forbidden"):
        carrier.validate_policy_carrier_arrays(
            leaked, labels, require_predictions=True
        )


def test_validation_fails_closed_on_label_row_mismatch() -> None:
    deployment, labels = _minimal_bundle()
    count = 30 * 8 * 6
    deployment = carrier.attach_frozen_recursive_inputs(
        deployment,
        source_row_id=deployment["policy_row_id"],
        future_action=np.zeros((count, 20), np.float32),
        geometry_token=np.zeros((count, 3, 256), np.float32),
    )
    attached = carrier.attach_aligned_predictions(
        deployment,
        np.zeros((count, 3, 256), np.float32),
        prediction_row_id=deployment["policy_row_id"],
        pre_drop_raw_prior_token=np.zeros((count, 3, 256), np.float32),
        raw_factual_predictions=np.zeros((count, 3, 256), np.float32),
        matched_no_action_predictions=np.zeros((count, 3, 256), np.float32),
        wrong_action_predictions=np.zeros((count, 7, 3, 256), np.float32),
        predictor_seed=15801,
        predictor_checkpoint_sha256=_sha("predictor"),
    )
    bad_labels = dict(labels)
    bad_labels["horizon"] = np.asarray(labels["horizon"]).copy()
    bad_labels["horizon"][0] = 99
    with pytest.raises(PermissionError, match="identity mismatch"):
        carrier.validate_policy_carrier_arrays(
            attached, bad_labels, require_predictions=True
        )


def test_frozen_source_alignment_rejects_physical_order_or_current_target_drift() -> None:
    deployment, labels = _minimal_bundle()
    physical = np.asarray(deployment["physical_episode_id"]).copy()
    current = np.asarray(labels["policy_target_anchor"])[:, -1].copy()
    carrier.verify_frozen_source_alignment(
        deployment,
        labels,
        expected_physical_episode_id=physical,
        expected_current_target=current,
    )

    wrong_order = physical.copy()
    wrong_order[[0, 48]] = wrong_order[[48, 0]]
    with pytest.raises(PermissionError, match="identity/order"):
        carrier.verify_frozen_source_alignment(
            deployment,
            labels,
            expected_physical_episode_id=wrong_order,
            expected_current_target=current,
        )
    wrong_target = current.copy()
    wrong_target[0, 0, 0] = 1.0
    with pytest.raises(PermissionError, match="current anchors"):
        carrier.verify_frozen_source_alignment(
            deployment,
            labels,
            expected_physical_episode_id=physical,
            expected_current_target=wrong_target,
        )
