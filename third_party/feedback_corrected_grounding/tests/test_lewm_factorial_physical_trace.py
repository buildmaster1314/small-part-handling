from __future__ import annotations

import numpy as np

from fcpg.lewm_factorial_physical_trace import (
    PhysicalTraceEnv,
    summarize_physical_trace,
)


class _FakeEnv:
    def __init__(self, distances: list[float]) -> None:
        self.distances = distances
        self.index = 0
        self._target_pos = np.zeros(3, dtype=np.float64)
        self.tcp_center = np.asarray([0.0, 0.0, 0.1])
        self.action_space = object()

    def _get_pos_objects(self) -> np.ndarray:
        return np.asarray([self.distances[self.index], 0.0, 0.0])

    def step(self, action: np.ndarray):
        self.index = min(self.index + 1, len(self.distances) - 1)
        distance = self.distances[self.index]
        info = {"success": float(distance <= 0.07), "obj_to_target": distance}
        return np.zeros(2), 1.0, False, False, info


def test_trace_records_phases_and_escaped_success_shell() -> None:
    distances = [0.10, 0.06, 0.06, 0.08]
    rows: list[dict[str, object]] = []
    env = PhysicalTraceEnv(_FakeEnv(distances), rows, drop_start=0, drop_duration=1)
    action = np.asarray([0.1, 0.0, 0.0, 1.0], dtype=np.float32)
    for _ in range(3):
        env.step(action)
    assert [row["phase"] for row in rows] == ["dropout", "dropout", "dropout"]
    summary = summarize_physical_trace(rows)
    assert summary["success_steps"] == 2
    assert summary["maximum_success_streak"] == 2
    assert summary["success_shell_steps"] == 2
    assert summary["retention_mechanism"] == "escaped_success_shell"


def test_trace_summary_rejects_nonconsecutive_steps() -> None:
    rows = [
        {
            "raw_step": 2,
            "success": 0.0,
            "object_to_target": 0.1,
            "within_terminal_5cm": False,
            "within_success_7cm": False,
            "object_position": [0.1, 0.0, 0.0],
            "phase": "prefix",
        }
    ]
    try:
        summarize_physical_trace(rows)
    except ValueError as error:
        assert "consecutive" in str(error)
    else:
        raise AssertionError("trace summary accepted nonconsecutive steps")
