from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fcpg.lewm_action_branch_a4 import (
    BRANCH_COUNT,
    BRANCH_HORIZON_RAW_STEPS,
    HORIZON,
    RAW_STEPS_PER_MODEL_STEP,
)
from fcpg.lewm_action_branch_a4_collection import RAW_ACTION_DIM
from scripts import collect_lewm_action_branch_a4_direction_targets as cli


def test_expert_branch_is_selected_by_semantic_identity_and_unpacked() -> None:
    action_dim = RAW_STEPS_PER_MODEL_STEP * RAW_ACTION_DIM
    center = np.arange(HORIZON * action_dim, dtype=np.float32).reshape(
        HORIZON, action_dim
    )
    future = np.zeros((BRANCH_COUNT, HORIZON, action_dim), dtype=np.float32)
    semantic = np.asarray([4, 2, 7, 1, 5, 0, 6, 3], dtype=np.int64)
    future[5] = center

    slot, actions = cli._expert_raw_actions(
        future, center, semantic, expert_semantic_id=0
    )

    assert slot == 5
    assert actions.shape == (BRANCH_HORIZON_RAW_STEPS, RAW_ACTION_DIM)
    np.testing.assert_array_equal(actions, center.reshape(-1, RAW_ACTION_DIM))

    future[5, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="center_action"):
        cli._expert_raw_actions(future, center, semantic, expert_semantic_id=0)


def test_display_coordinates_flip_only_standard_tasks() -> None:
    points = np.asarray([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])

    standard = cli._display_coordinates(points, task="pick-place-v3", height=224)
    custom = cli._display_coordinates(points, task="drawer-retrieve-v0", height=224)

    np.testing.assert_array_equal(standard[:, 0], points[:, 0])
    np.testing.assert_array_equal(standard[:, 1], 223.0 - points[:, 1])
    np.testing.assert_array_equal(custom, points)
    np.testing.assert_array_equal(
        cli._display_coordinates(standard, task="pick-place-v3", height=224),
        points,
    )


def test_stack_preserves_raw_model_targets_and_rejects_test_split() -> None:
    row = cli.ReplayRow(
        cache_row_index=7,
        proposal_id="pick-place-v3:task-instance-01:proposal-020",
        split="train",
        task_name="pick-place-v3",
        task_instance_index=1,
        source_episode_id="pick-place-v3:task-instance-01",
        environment_seed=620001,
        proposal_raw_step=20,
        expert_slot=3,
        expert_actions=np.zeros((BRANCH_HORIZON_RAW_STEPS, RAW_ACTION_DIM), dtype=np.float32),
        expected_physical=np.zeros((HORIZON, 7), dtype=np.float32),
    )
    raw = np.stack(
        (
            np.arange(BRANCH_HORIZON_RAW_STEPS + 1, dtype=np.float32),
            np.arange(BRANCH_HORIZON_RAW_STEPS + 1, dtype=np.float32) + 10.0,
        ),
        axis=-1,
    )
    display = raw.copy()
    display[:, 1] = 223.0 - display[:, 1]
    result = cli.ReplayResult(
        q_raw=raw,
        q_display=display,
        all_anchors_valid=np.ones(BRANCH_HORIZON_RAW_STEPS + 1, dtype=bool),
        physical_validation_max_abs=0.0,
    )

    arrays = cli.stack_direction_targets(
        [row], {row.cache_row_index: result}, width=224, height=224
    )

    assert tuple(arrays) == cli.OUTPUT_KEYS
    assert arrays["q_center_raw_coords_raw_steps"].shape == (1, 41, 2)
    assert arrays["q_center_model_raw"].shape == (1, 9, 2)
    np.testing.assert_array_equal(
        arrays["q_center_model_raw"][0], raw[cli.MODEL_STEP_OFFSETS]
    )
    np.testing.assert_array_equal(
        arrays["q_center_display"][0], display[cli.MODEL_STEP_OFFSETS]
    )

    with pytest.raises(PermissionError, match="train/development"):
        cli.stack_direction_targets(
            [replace(row, split="test")],
            {row.cache_row_index: result},
            width=224,
            height=224,
        )
