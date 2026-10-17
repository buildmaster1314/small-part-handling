from __future__ import annotations
import numpy as np

from scripts.evaluate_moving_goal_lewm import (
    HORIZON,
    _window_indices,
    _window_rows,
)


def test_window_metadata_does_not_overwrite_h8() -> None:
    count = 50
    data = {
        "task_name": np.asarray(["reach-v3"] * count),
        "episode_id": np.zeros(count, dtype=np.int64),
        "task_index": np.zeros(count, dtype=np.int64),
        "frame_id": np.arange(count, dtype=np.int64),
    }
    table = _window_rows(data, stride=5, max_samples=0)
    assert table.shape[1] == HORIZON + 4
    current, future = _window_indices(data, table[0])
    assert current == 0
    np.testing.assert_array_equal(future, np.arange(5, 45, 5))
    assert int(table[0, 2 + HORIZON]) == 0
    assert int(table[0, 3 + HORIZON]) == 0
