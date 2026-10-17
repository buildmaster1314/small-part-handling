from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.build_g1a_selector_cache import (
    CANONICAL_CONTROL_KEYPOINT_SITES,
    DEPLOYMENT_FORBIDDEN_SUBSTRINGS,
    StrictHistory,
    assert_label_history_contract,
    assert_deployment_partition,
    candidate_geometry,
    episode_splits,
    executed_action_derangement,
    grounding_valid_mask,
    infer_canonical_fcpg_history_points,
    select_smoke_snapshots,
    strict_history_from_data,
    subset_history,
    validate_selected_policy_checkpoint_tasks,
)


def _control_data(episodes: int = 3, frames: int = 20) -> SimpleNamespace:
    count = episodes * frames
    episode = np.repeat(np.arange(episodes), frames)
    frame = np.tile(np.arange(frames), episodes)
    action = np.stack(
        [
            episode.astype(np.float32),
            frame.astype(np.float32),
            np.ones(count, dtype=np.float32),
            np.zeros(count, dtype=np.float32),
        ],
        axis=1,
    )
    return SimpleNamespace(
        task=np.full(count, "pick-place-v3"),
        episode=episode,
        frame=frame,
        action=action,
        q_nominal=np.full((count, 3, 2), 112.0, dtype=np.float32),
        xyz=np.arange(count * 9, dtype=np.float32).reshape(count, 3, 3),
        obs=np.zeros((count, 39), dtype=np.float32),
        valid_nominal=np.ones((count, 3), dtype=bool),
        image_size=(224, 224),
    )


def test_strict_history_uses_only_actions_before_current() -> None:
    data = _control_data(episodes=2, frames=20)
    history = strict_history_from_data(data)
    assert len(history.current_indices) == 10
    first = 15
    position = int(np.flatnonzero(history.current_indices == first)[0])
    np.testing.assert_array_equal(history.state_indices[position], [0, 5, 10])
    np.testing.assert_array_equal(
        history.action_indices[position],
        np.asarray([range(0, 5), range(5, 10), range(10, 15)]),
    )
    assert 15 not in history.action_indices[position]
    np.testing.assert_array_equal(
        history.action_blocks[position, 2].reshape(5, 4), data.action[10:15]
    )


def test_smoke_selection_spans_episodes_and_subsets_history() -> None:
    data = _control_data(episodes=3, frames=20)
    history = strict_history_from_data(data)
    positions = select_smoke_snapshots(history, data, max_snapshots=3)
    selected = subset_history(history, positions)
    np.testing.assert_array_equal(data.frame[selected.current_indices], [15, 15, 15])
    assert set(data.episode[selected.current_indices].tolist()) == {0, 1, 2}


def test_selected_tasks_must_be_present_in_every_policy_training_source(
    tmp_path,
) -> None:
    pick = tmp_path / "pick.npz"
    shared = tmp_path / "shared.npz"
    np.savez(pick, task_name=np.asarray(["pick-place-v3"]))
    np.savez(
        shared,
        task_name=np.asarray(["pick-place-v3", "button-press-v3"]),
    )

    with pytest.raises(ValueError, match="was not trained on 'button-press-v3'"):
        validate_selected_policy_checkpoint_tasks(
            [{"data": str(pick), "extra_data": []}],
            np.asarray(["pick-place-v3", "button-press-v3"]),
        )

    validate_selected_policy_checkpoint_tasks(
        [{"data": str(pick), "extra_data": [str(shared)]}],
        np.asarray(["pick-place-v3", "button-press-v3"]),
    )


def test_executed_action_derangement_preserves_frame_and_changes_episode() -> None:
    data = _control_data(episodes=3, frames=20)
    history = strict_history_from_data(data)
    selected_positions = select_smoke_snapshots(history, data, max_snapshots=3)
    history = subset_history(history, selected_positions)
    current = history.current_indices
    split = np.full(len(current), "smoke")
    shuffled, permutation = executed_action_derangement(
        data.task[current],
        data.episode[current],
        data.frame[current],
        split,
        history.action_blocks,
        seed=7,
    )
    assert not np.any(permutation == np.arange(3))
    np.testing.assert_array_equal(data.frame[current][permutation], data.frame[current])
    assert not np.any(data.episode[current][permutation] == data.episode[current])
    np.testing.assert_array_equal(shuffled[:, :2], history.action_blocks[:, :2])
    np.testing.assert_array_equal(
        shuffled[:, -1], history.action_blocks[permutation, -1]
    )


def test_executed_action_derangement_never_crosses_data_split() -> None:
    data = _control_data(episodes=6, frames=20)
    history = strict_history_from_data(data)
    selected_positions = select_smoke_snapshots(history, data, max_snapshots=6)
    history = subset_history(history, selected_positions)
    current = history.current_indices
    split = np.asarray(
        ["train", "train", "validation", "validation", "test", "test"]
    )
    shuffled, permutation = executed_action_derangement(
        data.task[current],
        data.episode[current],
        data.frame[current],
        split,
        history.action_blocks,
        seed=11,
    )
    np.testing.assert_array_equal(split[permutation], split)
    assert not np.any(data.episode[current][permutation] == data.episode[current])
    np.testing.assert_array_equal(
        shuffled[:, -1], history.action_blocks[permutation, -1]
    )


def test_candidate_geometry_matches_frame_encoder_contract() -> None:
    data = _control_data(episodes=1, frames=20)
    rows = np.asarray([15, 16])
    points = np.broadcast_to(data.q_nominal[rows, None], (2, 4, 3, 2)).copy()
    points[0, 1, 2] += np.asarray([22.3, -11.15], dtype=np.float32)
    valid = np.ones((2, 4, 3), dtype=bool)
    valid[1, 3, 1] = False
    geometry = candidate_geometry(data, raw_indices=rows, points=points, valid=valid)
    assert geometry.shape == (2, 4, 3, 11)
    np.testing.assert_allclose(
        geometry[0, 1, 2, 2:4], [0.1, -0.05], atol=1e-7
    )
    np.testing.assert_array_equal(geometry[0, 0, :, 7:10], np.eye(3))
    assert geometry[1, 3, 1, -1] == 0.0


def test_grounding_valid_mask_combines_nominal_finite_and_image_bounds() -> None:
    points = np.asarray(
        [
            [[0.0, 0.0], [223.0, 223.0], [224.0, 20.0]],
            [[10.0, -0.1], [np.nan, 5.0], [30.0, 40.0]],
        ],
        dtype=np.float32,
    )
    valid_nominal = np.asarray([[True, False, True], [True, True, True]])
    np.testing.assert_array_equal(
        grounding_valid_mask(points, valid_nominal, image_size=(224, 224)),
        [[True, False, False], [False, False, True]],
    )


def test_native_history_inference_has_no_privileged_array_dependency() -> None:
    class DummyAdapter(torch.nn.Module):
        def forward_from_backbone_features(
            self,
            feature: torch.Tensor,
            prefix: torch.Tensor,
            q_nominal: torch.Tensor,
            xyz: torch.Tensor,
            robot_obs: torch.Tensor,
            *,
            image_size: tuple[int, int],
        ) -> SimpleNamespace:
            del feature, prefix, xyz, robot_obs, image_size
            return SimpleNamespace(q_corrected_px=q_nominal + 0.5)

    data = _control_data(episodes=1, frames=4)
    assert not hasattr(data, "q_true")
    assert not hasattr(data, "valid_true")
    result = infer_canonical_fcpg_history_points(
        DummyAdapter(),
        obs_mode="robot",
        backbone=np.zeros((4, 2, 2, 2), dtype=np.float16),
        prefix=np.zeros((4, 2), dtype=np.float16),
        data=data,
        raw_indices=np.asarray([1, 3]),
        device=torch.device("cpu"),
        batch_size=1,
    )
    np.testing.assert_allclose(result, data.q_nominal[[1, 3]] + 0.5)


def test_offline_label_history_must_match_primary_history() -> None:
    assert_label_history_contract(
        {
            "history_grounding_primary": "native_fcpg",
            "label_history_source": "native_fcpg",
            "label_history_matches_primary": True,
        }
    )
    with pytest.raises(ValueError, match="differs from deployment-primary"):
        assert_label_history_contract(
            {
                "history_grounding_primary": "native_fcpg",
                "label_history_source": "wide_top1",
                "label_history_matches_primary": False,
            }
        )


@pytest.mark.parametrize(
    "contract",
    [
        {"history_grounding_primary": "native_fcpg"},
        {"label_history_source": "native_fcpg"},
    ],
)
def test_offline_label_history_contract_requires_both_sources(
    contract: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="must name primary and label"):
        assert_label_history_contract(contract)


def test_offline_label_history_contract_requires_explicit_match_flag() -> None:
    with pytest.raises(ValueError, match="must explicitly certify"):
        assert_label_history_contract(
            {
                "history_grounding_primary": "native_fcpg",
                "label_history_source": "native_fcpg",
                "label_history_matches_primary": False,
            }
        )


def test_canonical_control_anchor_order_is_explicit() -> None:
    assert CANONICAL_CONTROL_KEYPOINT_SITES == (
        "leftEndEffector",
        "rightEndEffector",
        "endEffector",
    )


def test_episode_splits_never_cross_physical_episode() -> None:
    data = _control_data(episodes=10, frames=20)
    selected = data.frame >= 15
    split, groups = episode_splits(data.task[selected], data.episode[selected], seed=5)
    assert set(split.tolist()) == {"train", "validation", "test"}
    for episode in range(10):
        assert len(set(split[data.episode[selected] == episode].tolist())) == 1
    flattened = [key for values in groups.values() for key in values]
    assert len(flattened) == len(set(flattened)) == 10


@pytest.mark.parametrize("forbidden", DEPLOYMENT_FORBIDDEN_SUBSTRINGS)
def test_deployment_partition_rejects_offline_names(forbidden: str) -> None:
    with pytest.raises(ValueError, match="offline information leaked"):
        assert_deployment_partition(
            {
                "episode_id": np.asarray([0]),
                f"bad_{forbidden}": np.asarray([0.0], dtype=np.float32),
            }
        )


def test_deployment_partition_accepts_minimal_schema() -> None:
    assert_deployment_partition(
        {
            "episode_id": np.asarray([0, 1]),
            "candidate_points_px": np.zeros((2, 16, 3, 2), dtype=np.float32),
            "candidate_anchor_tokens": np.zeros(
                (2, 16, 3, 192), dtype=np.float16
            ),
            "history_action_blocks": np.zeros((2, 3, 20), dtype=np.float32),
            "predicted_current_summary_prior": np.zeros((2, 192), dtype=np.float16),
        }
    )


def test_deployment_partition_rejects_nonfinite_and_mismatched_counts() -> None:
    with pytest.raises(ValueError, match="sample count"):
        assert_deployment_partition(
            {"episode_id": np.asarray([0, 1]), "points": np.zeros((1, 2))}
        )
    with pytest.raises(ValueError, match="non-finite"):
        assert_deployment_partition(
            {"episode_id": np.asarray([0]), "score": np.asarray([np.nan])}
        )
