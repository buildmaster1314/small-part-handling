from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from fcpg.sequence_data import (
    HomogeneousConcatDataset,
    ProjectionSequenceDataset,
    compute_multi_source_normalization_stats,
    compute_normalization_stats,
    make_balanced_source_sampler,
    resolve_projection_sources,
    select_episode_subset,
)
from scripts.cache_fcpg_corrections import _extract_grounding_output


def _write_sequence(path, frames_per_episode: int = 4) -> None:
    count = 2 * frames_per_episode
    frame_ids = np.tile(np.arange(frames_per_episode), 2)
    values = np.arange(count, dtype=np.float32)
    np.savez(
        path,
        rgb=np.broadcast_to(values[:, None, None, None], (count, 2, 3, 3)).astype(np.uint8),
        obs=np.broadcast_to(values[:, None], (count, 39)).copy(),
        action=np.broadcast_to((values + 100.0)[:, None], (count, 4)).copy(),
        task_name=np.repeat(np.asarray(["pick-place-v3"]), count),
        episode_id=np.repeat(np.arange(2), frames_per_episode),
        frame_id=frame_ids,
        keypoints_xyz=np.broadcast_to(values[:, None, None], (count, 3, 3)).copy(),
        q_nominal=np.broadcast_to(values[:, None, None], (count, 3, 2)).copy(),
        q_true=np.broadcast_to(values[:, None, None], (count, 3, 2)).copy(),
        valid_true=np.ones((count, 3), dtype=bool),
        valid_nominal=np.ones((count, 3), dtype=bool),
    )


def test_sequence_dataset_returns_episode_safe_next_history_and_actions(tmp_path) -> None:
    data_path = tmp_path / "sequence.npz"
    _write_sequence(data_path)
    dataset = ProjectionSequenceDataset(
        data_path,
        np.arange(8),
        obs_mode="robot",
        obs_horizon=2,
        action_horizon=3,
    )

    first = dataset[0]
    middle = dataset[2]
    terminal = dataset[3]

    torch.testing.assert_close(first["obs"][:, 0], torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(first["history_action"][:, 0], torch.tensor([100.0, 100.0]))
    torch.testing.assert_close(first["next_obs"][:, 0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(middle["next_obs"][:, 0], torch.tensor([2.0, 3.0]))
    assert bool(first["next_mask"])
    assert not bool(terminal["next_mask"])
    torch.testing.assert_close(terminal["next_obs"], terminal["obs"])
    torch.testing.assert_close(terminal["next_rgb"], terminal["rgb"])
    # The terminal placeholder must not cross into the next episode.
    assert float(terminal["next_obs"].max()) == 3.0


def test_sequence_dataset_optionally_tags_source_index(tmp_path) -> None:
    data_path = tmp_path / "sequence.npz"
    _write_sequence(data_path)
    sample = ProjectionSequenceDataset(
        data_path,
        np.arange(8),
        source_index=3,
    )[0]

    assert int(sample["source_index"]) == 3


@pytest.mark.parametrize("legacy_reliability", [False, True])
def test_sequence_dataset_loads_residual_only_and_legacy_grounding_caches(
    tmp_path,
    legacy_reliability: bool,
) -> None:
    data_path = tmp_path / "sequence.npz"
    _write_sequence(data_path)
    cache = tmp_path / "grounding"
    cache.mkdir()
    np.save(cache / "q_corrected.npy", np.full((8, 3, 2), 9.0, dtype=np.float32))
    if legacy_reliability:
        np.save(cache / "reliability.npy", np.full((8, 3), 0.75, dtype=np.float32))
        (cache / "metadata.json").write_text(
            json.dumps({"version": 1, "source_data": str(data_path.resolve())}),
            encoding="utf-8",
        )
    else:
        (cache / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "has_reliability": False,
                    "source_data": str(data_path.resolve()),
                }
            ),
            encoding="utf-8",
        )

    sample = ProjectionSequenceDataset(
        data_path,
        np.arange(8),
        obs_mode="robot",
        obs_horizon=2,
        grounding_cache=cache,
    )[1]

    torch.testing.assert_close(sample["fcpg_q_corrected"], torch.full((2, 3, 2), 9.0))
    torch.testing.assert_close(sample["next_fcpg_q_corrected"], torch.full((2, 3, 2), 9.0))
    if legacy_reliability:
        torch.testing.assert_close(sample["fcpg_reliability"], torch.full((2, 3), 0.75))
        torch.testing.assert_close(
            sample["next_fcpg_reliability"], torch.full((2, 3), 0.75)
        )
    else:
        assert "fcpg_reliability" not in sample
        assert "next_fcpg_reliability" not in sample


def test_cache_output_extraction_accepts_residual_only_adapter() -> None:
    nominal = torch.ones(2, 3, 2)
    corrected, reliability = _extract_grounding_output(
        SimpleNamespace(delta_px=torch.full_like(nominal, 2.0)), nominal
    )

    torch.testing.assert_close(corrected, torch.full_like(nominal, 3.0))
    assert reliability is None

    corrected, reliability = _extract_grounding_output(
        SimpleNamespace(
            q_corrected_px=torch.full_like(nominal, 4.0),
            reliability=torch.full((2, 3), 0.5),
        ),
        nominal,
    )
    torch.testing.assert_close(corrected, torch.full_like(nominal, 4.0))
    torch.testing.assert_close(reliability, torch.full((2, 3), 0.5))


def test_multi_source_normalization_matches_combined_frames(tmp_path) -> None:
    data_path = tmp_path / "sequence.npz"
    _write_sequence(data_path)
    combined = compute_normalization_stats(data_path, np.arange(8), "robot")
    multi = compute_multi_source_normalization_stats(
        [data_path, data_path],
        [np.arange(4), np.arange(4, 8)],
        "robot",
    )

    np.testing.assert_allclose(multi.obs_mean, combined.obs_mean)
    np.testing.assert_allclose(multi.obs_std, combined.obs_std)
    np.testing.assert_allclose(multi.action_mean, combined.action_mean)
    np.testing.assert_allclose(multi.action_std, combined.action_std)


def test_homogeneous_concat_and_balanced_sampler_enforce_source_contract() -> None:
    first = TensorDataset(torch.arange(2))
    second = TensorDataset(torch.arange(5))
    first.obs_dim = second.obs_dim = 4
    combined = HomogeneousConcatDataset([first, second], ("obs_dim",))

    assert len(combined) == 7
    assert combined.obs_dim == 4
    assert combined.source_lengths == (2, 5)
    sampler = make_balanced_source_sampler([first, second], seed=7)
    np.testing.assert_allclose(sampler.weights[:2].sum().item(), 1.0)
    np.testing.assert_allclose(sampler.weights[2:].sum().item(), 1.0)
    assert list(sampler) == list(make_balanced_source_sampler([first, second], seed=7))

    second.obs_dim = 5
    with pytest.raises(ValueError, match="disagree on obs_dim"):
        HomogeneousConcatDataset([first, second], ("obs_dim",))


def test_episode_subset_is_complete_deterministic_and_nested(tmp_path) -> None:
    data_path = tmp_path / "sequence.npz"
    _write_sequence(data_path)
    with np.load(data_path, allow_pickle=False) as data:
        all_indices = np.arange(8)
        one = select_episode_subset(data, all_indices, max_episodes=1, seed=7)
        one_again = select_episode_subset(data, all_indices, max_episodes=1, seed=7)
        two = select_episode_subset(data, all_indices, max_episodes=2, seed=7)

        np.testing.assert_array_equal(one, one_again)
        assert len(one) == 4
        assert set(one).issubset(set(two))
        assert len(np.unique(data["episode_id"][one])) == 1
        with pytest.raises(ValueError, match="only 2 are available"):
            select_episode_subset(data, all_indices, max_episodes=3, seed=7)


def test_projection_sources_require_explicit_cache_pairing() -> None:
    sources = resolve_projection_sources(
        "pick.npz",
        "pick_vision",
        ["button.npz"],
        ["button_vision"],
        "pick_grounding",
        ["button_grounding"],
    )

    assert [source.data.name for source in sources] == ["pick.npz", "button.npz"]
    assert [source.vision_cache.name for source in sources] == [
        "pick_vision",
        "button_vision",
    ]
    with pytest.raises(ValueError, match="extra-vision-cache"):
        resolve_projection_sources("pick.npz", "pick_vision", ["button.npz"])


def test_projection_sources_pair_compact_policy_caches() -> None:
    compact = resolve_projection_sources(
        data="pick.npz",
        vision_cache=None,
        extra_data=["button.npz"],
        grounding_cache="pick_grounding",
        extra_grounding_cache=["button_grounding"],
        policy_cache="pick_policy",
        extra_policy_cache=["button_policy"],
    )

    assert [source.policy_cache.name for source in compact] == [
        "pick_policy",
        "button_policy",
    ]
    with pytest.raises(ValueError, match="extra-policy-cache"):
        resolve_projection_sources(
            data="pick.npz",
            vision_cache=None,
            extra_data=["button.npz"],
            policy_cache="pick_policy",
        )


def test_sequence_dataset_loads_compact_policy_visual_cache(tmp_path) -> None:
    data_path = tmp_path / "sequence.npz"
    _write_sequence(data_path)
    policy_cache = tmp_path / "policy"
    policy_cache.mkdir()
    np.save(
        policy_cache / "sampled_visual.npy",
        np.arange(8 * 3 * 6, dtype=np.float16).reshape(8, 3, 6),
    )
    np.save(
        policy_cache / "global_visual.npy",
        np.arange(8 * 6, dtype=np.float16).reshape(8, 6),
    )
    (policy_cache / "metadata.json").write_text(
        json.dumps(
            {
                "source_data": str(data_path.resolve()),
                "image_width": 8,
                "image_height": 8,
                "visual_dim": 6,
            }
        ),
        encoding="utf-8",
    )

    sample = ProjectionSequenceDataset(
        data_path,
        np.arange(8),
        obs_horizon=2,
        action_horizon=2,
        policy_cache=policy_cache,
    )[1]

    assert "rgb" not in sample
    assert "vision_backbone_features" not in sample
    assert sample["sampled_visual"].shape == (2, 3, 6)
    assert sample["global_visual"].shape == (2, 6)
    assert sample["next_sampled_visual"].shape == (2, 3, 6)
    assert sample["next_global_visual"].shape == (2, 6)
