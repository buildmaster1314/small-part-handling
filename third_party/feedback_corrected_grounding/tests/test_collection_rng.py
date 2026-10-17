from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from fcpg.metaworld_utils import make_env
from scripts.collect_metaworld_projection_dataset import make_rng_streams


def test_independent_camera_draws_do_not_change_action_stream() -> None:
    action_a, camera_a, _ = make_rng_streams(9100, independent=True)
    action_b, camera_b, _ = make_rng_streams(9100, independent=True)

    camera_a.normal(size=100)
    camera_b.normal(size=3)

    np.testing.assert_array_equal(action_a.normal(size=20), action_b.normal(size=20))


def test_legacy_shared_stream_contract_is_preserved() -> None:
    action, camera, projection = make_rng_streams(3, independent=False)

    assert action is camera is projection


def test_make_env_seeds_metaworld_task_generation(monkeypatch) -> None:
    calls: list[tuple[str, int | None]] = []

    class FakeEnv:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.task = None

        def set_task(self, task) -> None:
            self.task = task

    def fake_mt1(task_name: str, seed: int | None = None):
        calls.append((task_name, seed))
        return SimpleNamespace(
            train_classes={task_name: FakeEnv},
            train_tasks=["task-0", "task-1"],
        )

    monkeypatch.setitem(sys.modules, "metaworld", SimpleNamespace(MT1=fake_mt1))

    env = make_env(
        "reach-v3",
        "corner",
        224,
        224,
        task_index=3,
        benchmark_seed=9100,
    )

    assert calls == [("reach-v3", 9100)]
    assert env.task == "task-1"
