from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from fcpg.lewm_action_branch_collection import ActionBranchBank
from fcpg.lewm_action_branch_a3 import (
    build_diverse_action_branch_bank,
    summarize_a3_support,
)


def _center(*, zero_motion: bool = False) -> np.ndarray:
    center = np.zeros((40, 4), dtype=np.float32)
    if not zero_motion:
        step = np.arange(40, dtype=np.float32)
        center[:, 0] = 0.2 + 0.1 * np.sin(step)
        center[:, 1] = -0.3 + 0.05 * np.cos(step)
        center[:, 2] = 0.1
    center[:, 3] = np.where(np.arange(40) < 17, -1.0, 1.0)
    return center


def _matching_indices(actions: np.ndarray, expected: np.ndarray) -> list[int]:
    return [
        index
        for index, branch in enumerate(actions[:, :, :3])
        if np.array_equal(branch, expected)
    ]


def test_bank_contains_absolute_semantics_before_anonymous_slots() -> None:
    center = _center()
    bank = build_diverse_action_branch_bank(
        center,
        direction_seed=11,
        permutation_seed=19,
    )

    assert isinstance(bank, ActionBranchBank)
    assert bank.names == tuple(f"candidate_{index}" for index in range(8))
    assert bank.actions.shape == (8, 40, 4)
    assert bank.actions.dtype == np.float32
    np.testing.assert_array_equal(
        bank.actions[:, :, 3],
        np.broadcast_to(center[None, :, 3], (8, 40)),
    )
    for expected in (
        center[:, :3],
        -center[:, :3],
        np.zeros((40, 3), dtype=np.float32),
        0.5 * center[:, :3],
    ):
        assert len(_matching_indices(bank.actions, expected)) == 1

    simple = {
        _matching_indices(bank.actions, expected)[0]
        for expected in (
            center[:, :3],
            -center[:, :3],
            np.zeros((40, 3), dtype=np.float32),
            0.5 * center[:, :3],
        )
    }
    transverse = bank.actions[
        [index for index in range(8) if index not in simple], :, :3
    ]
    first = transverse[0]
    first_pair = next(
        index for index in range(1, 4) if np.array_equal(transverse[index], -first)
    )
    second = transverse[next(index for index in range(1, 4) if index != first_pair)]
    assert any(np.array_equal(candidate, -second) for candidate in transverse)
    np.testing.assert_allclose(np.sum(first * center[:, :3], axis=1), 0.0, atol=1e-7)
    np.testing.assert_allclose(np.sum(second * center[:, :3], axis=1), 0.0, atol=1e-7)
    np.testing.assert_allclose(np.sum(first * second, axis=1), 0.0, atol=1e-7)


def test_bank_is_seed_deterministic_and_direction_seed_is_fallback_only() -> None:
    center = _center()
    left = build_diverse_action_branch_bank(
        center,
        direction_seed=3,
        permutation_seed=5,
    )
    repeated = build_diverse_action_branch_bank(
        center,
        direction_seed=3,
        permutation_seed=5,
    )
    ignored = build_diverse_action_branch_bank(
        center,
        direction_seed=300,
        permutation_seed=5,
    )
    permuted = build_diverse_action_branch_bank(
        center,
        direction_seed=3,
        permutation_seed=6,
    )

    np.testing.assert_array_equal(left.actions, repeated.actions)
    np.testing.assert_array_equal(left.actions, ignored.actions)
    assert not np.array_equal(left.actions, permuted.actions)
    assert sorted(branch.tobytes() for branch in left.actions) == sorted(
        branch.tobytes() for branch in permuted.actions
    )


def test_zero_motion_uses_finite_seeded_transverse_fallback() -> None:
    center = _center(zero_motion=True)
    left = build_diverse_action_branch_bank(
        center,
        direction_seed=23,
        permutation_seed=29,
    ).actions
    repeated = build_diverse_action_branch_bank(
        center,
        direction_seed=23,
        permutation_seed=29,
    ).actions
    changed = build_diverse_action_branch_bank(
        center,
        direction_seed=24,
        permutation_seed=29,
    ).actions

    assert np.isfinite(left).all()
    np.testing.assert_array_equal(left, repeated)
    assert not np.array_equal(left, changed)
    nonzero = left[np.any(left[:, :, :3] != 0.0, axis=(1, 2)), :, :3]
    assert nonzero.shape[0] == 4
    assert all(
        any(np.array_equal(other, -branch) for other in nonzero)
        for branch in nonzero
    )


def test_transverse_clipping_scales_each_step_without_rotating_sign_pairs() -> None:
    center = _center()
    center[:, :3] *= 0.5
    bank = build_diverse_action_branch_bank(
        center,
        direction_seed=7,
        permutation_seed=31,
        action_low=np.asarray([-0.25, -0.25, -0.25, -1.0]),
        action_high=np.asarray([0.25, 0.25, 0.25, 1.0]),
    )
    assert np.all(bank.actions[:, :, :3] >= -0.25)
    assert np.all(bank.actions[:, :, :3] <= 0.25)

    simple = set()
    for expected in (
        center[:, :3],
        -center[:, :3],
        np.zeros((40, 3), dtype=np.float32),
        0.5 * center[:, :3],
    ):
        simple.add(_matching_indices(bank.actions, expected)[0])
    transverse = bank.actions[
        [index for index in range(8) if index not in simple], :, :3
    ]
    assert all(
        any(np.array_equal(other, -branch) for other in transverse)
        for branch in transverse
    )
    assert np.all(np.abs(transverse) <= 0.25)
    expected_norm = np.linalg.norm(center[:, :3], axis=1)
    np.testing.assert_allclose(
        np.linalg.norm(transverse, axis=2),
        np.broadcast_to(expected_norm, (4, 40)),
        atol=1e-7,
    )


@pytest.mark.parametrize(
    ("center", "kwargs", "message"),
    [
        (np.zeros((39, 4), dtype=np.float32), {}, "shape"),
        (np.zeros((40, 3), dtype=np.float32), {}, "shape"),
        (
            np.full((40, 4), np.nan, dtype=np.float32),
            {},
            "finite",
        ),
        (
            np.zeros((40, 4), dtype=np.float32),
            {"motion_delta_rms": 0.15},
            "must be None",
        ),
    ],
)
def test_bank_rejects_invalid_shapes_values_and_a1_radius_semantics(
    center: np.ndarray,
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        build_diverse_action_branch_bank(
            center,
            direction_seed=1,
            permutation_seed=2,
            **kwargs,
        )


def _support_stores(
    *,
    train_scales: Sequence[float],
    development_episodes: int,
    bad_long_episode_rows: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    task_count = len(train_scales)
    split: list[str] = []
    task_index: list[int] = []
    episode: list[str] = []
    target_rows: list[np.ndarray] = []
    tcp_rows: list[np.ndarray] = []
    object_rows: list[np.ndarray] = []

    branch_sign = np.asarray([-1.0] * 4 + [1.0] * 4)[:, None, None, None]
    for task, train_scale in enumerate(train_scales):
        split.append("train")
        task_index.append(task)
        episode.append(f"task-{task}-train")
        target_rows.append(
            np.broadcast_to(branch_sign * train_scale, (8, 8, 1, 1)).copy()
        )
        tcp_rows.append(np.zeros((8, 8, 3), dtype=np.float32))
        object_rows.append(np.zeros((8, 8, 3), dtype=np.float32))

        for development_episode in range(development_episodes):
            repeats = (
                bad_long_episode_rows
                if development_episode == 0 and bad_long_episode_rows
                else 1
            )
            informative = development_episode != 0 or not bad_long_episode_rows
            for row in range(repeats):
                split.append("development")
                task_index.append(task)
                episode.append(f"task-{task}-development-{development_episode}")
                amplitude = 1.0 if informative else 0.0
                target_rows.append(
                    np.broadcast_to(
                        branch_sign * amplitude,
                        (8, 8, 1, 1),
                    ).copy()
                )
                tcp = np.zeros((8, 8, 3), dtype=np.float32)
                if informative:
                    tcp[1, 5:, 0] = 0.02
                tcp_rows.append(tcp)
                object_rows.append(np.zeros((8, 8, 3), dtype=np.float32))

    identity = {
        "split": np.asarray(split),
        "task_index": np.asarray(task_index, dtype=np.int64),
        "source_episode_id": np.asarray(episode),
    }
    latent = {
        **identity,
        "target_embedding": np.asarray(target_rows, dtype=np.float32),
    }
    physical = {
        **{key: value.copy() for key, value in identity.items()},
        "tcp": np.asarray(tcp_rows, dtype=np.float32),
        "object_position": np.asarray(object_rows, dtype=np.float32),
    }
    return latent, physical


def test_support_uses_task_specific_train_moments_without_scale_leakage() -> None:
    latent, physical = _support_stores(
        train_scales=(1.0, 100.0),
        development_episodes=1,
    )
    summary = summarize_a3_support(
        latent,
        physical,
        task_names=("small-scale", "large-scale"),
        expected_train_episodes=1,
        expected_development_episodes=1,
    )

    assert summary["decision"] == "No-Go"
    assert summary["per_task"]["small-scale"]["latent"]["passed"]
    assert not summary["per_task"]["large-scale"]["latent"]["passed"]
    assert summary["per_task"]["small-scale"]["latent"][
        "median_branch_variance"
    ] == pytest.approx(1.0)
    assert summary["per_task"]["large-scale"]["latent"][
        "median_branch_variance"
    ] == pytest.approx(1.0e-4)
    assert summary["physical_support_passed"]


def test_support_weights_episodes_equally_instead_of_bundles() -> None:
    latent, physical = _support_stores(
        train_scales=(1.0,),
        development_episodes=5,
        bad_long_episode_rows=100,
    )
    summary = summarize_a3_support(
        latent,
        physical,
        task_names=("task",),
        expected_train_episodes=1,
        expected_development_episodes=5,
    )
    task = summary["per_task"]["task"]

    assert summary["decision"] == "Go"
    assert task["physical_episode_macro_informative_fraction"] == pytest.approx(0.8)
    assert task["latent_episode_macro_informative_fraction"] == pytest.approx(0.8)
    assert task["latent"]["bundle_fraction"] == pytest.approx(4.0 / 104.0)
    assert task["physical"]["bundle_fraction"] == pytest.approx(4.0 / 104.0)
    assert task["latent"]["episode_macro_informative_fraction"] == pytest.approx(0.8)
    assert task["physical"]["episode_macro_informative_fraction"] == pytest.approx(0.8)


def test_support_accepts_object_or_tcp_and_rejects_identity_misalignment() -> None:
    latent, physical = _support_stores(
        train_scales=(1.0,),
        development_episodes=1,
    )
    development = physical["split"] == "development"
    physical["tcp"][development] = 0.0
    physical["object_position"][development, 2, 5:, 1] = 0.006
    summary = summarize_a3_support(
        latent,
        physical,
        task_names=("task",),
        expected_train_episodes=1,
        expected_development_episodes=1,
    )
    assert summary["decision"] == "Go"

    physical["source_episode_id"] = physical["source_episode_id"][::-1]
    with pytest.raises(ValueError, match="not aligned"):
        summarize_a3_support(
            latent,
            physical,
            task_names=("task",),
            expected_train_episodes=1,
            expected_development_episodes=1,
        )


def test_h1_to_h5_cannot_change_h6_to_h8_whitening_or_gate() -> None:
    latent, physical = _support_stores(
        train_scales=(1.0,),
        development_episodes=1,
    )
    baseline = summarize_a3_support(
        latent,
        physical,
        task_names=("task",),
        expected_train_episodes=1,
        expected_development_episodes=1,
    )
    changed = {name: value.copy() for name, value in latent.items()}
    changed["target_embedding"][:, :, :5] *= 1.0e6
    repeated = summarize_a3_support(
        changed,
        physical,
        task_names=("task",),
        expected_train_episodes=1,
        expected_development_episodes=1,
    )

    assert repeated["decision"] == baseline["decision"]
    assert repeated["per_task"]["task"]["median_latent_branch_variance"] == (
        baseline["per_task"]["task"]["median_latent_branch_variance"]
    )


def test_physical_and_latent_support_must_overlap_on_the_same_episodes() -> None:
    latent, physical = _support_stores(
        train_scales=(1.0,),
        development_episodes=12,
    )
    episodes = latent["source_episode_id"].astype(str)
    for index in (0, 1):
        identity = f"task-0-development-{index}"
        latent["target_embedding"][episodes == identity] = 0.0
    for index in (10, 11):
        identity = f"task-0-development-{index}"
        physical["tcp"][episodes == identity] = 0.0

    summary = summarize_a3_support(
        latent,
        physical,
        task_names=("task",),
        expected_train_episodes=1,
        expected_development_episodes=12,
    )
    task = summary["per_task"]["task"]
    assert task["physical_episode_macro_informative_fraction"] == pytest.approx(10 / 12)
    assert task["latent_episode_macro_informative_fraction"] == pytest.approx(10 / 12)
    assert task["joint_episode_macro_informative_fraction"] == pytest.approx(8 / 12)
    assert summary["decision"] == "No-Go"
