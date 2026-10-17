from __future__ import annotations

import random
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import fcpg.metaworld_snapshot as metaworld_snapshot
from fcpg.metaworld_snapshot import (
    MetaWorldSnapshot,
    SnapshotMismatchError,
    capture_metaworld_snapshot,
    capture_process_rng_snapshot,
    restore_metaworld_snapshot,
    restore_process_rng_snapshot,
)


class _FakeModel:
    def __init__(self) -> None:
        self.nq = 4
        self.nv = 3
        self.na = 1
        self.nu = 2
        self.nbody = 3
        self.nsite = 2
        self.ncam = 1
        self.nmocap = 1
        self.neq = 0
        self.npluginstate = 0
        self.nuserdata = 0
        self.integration_size = 8
        self.warmstart_size = 3
        self.body_pos = np.arange(9, dtype=np.float64).reshape(3, 3)
        self.site_pos = np.arange(6, dtype=np.float64).reshape(2, 3) + 10.0
        self.cam_pos = np.arange(3, dtype=np.float64).reshape(1, 3) + 20.0
        self.cam_quat = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)


class _FakeData:
    def __init__(self, model: _FakeModel) -> None:
        self.integration = np.linspace(0.25, 2.0, model.integration_size)
        self.warmstart = np.linspace(-0.5, 0.5, model.warmstart_size)


class _FakeSpace:
    def __init__(self, seed: int) -> None:
        self.np_random = np.random.default_rng(seed)


class _FakeEnv:
    def __init__(self) -> None:
        self.model = _FakeModel()
        self.data = _FakeData(self.model)

        self.curr_path_length = 7
        self._prev_obs = np.asarray([1.5, 2.5, 3.5], dtype=np.float64)
        self._last_stable_obs = np.asarray([4.5, 5.5], dtype=np.float64)
        self._did_see_sim_exception = False

        self._last_rand_vec = np.asarray(
            [0.7, 0.8, 0.9, 0.4, 0.5, 0.6], dtype=np.float64
        )
        self.obj_init_pos = self._last_rand_vec[:3]
        self._target_pos = self._last_rand_vec[3:]
        self._freeze_rand_vec = True
        self._partially_observable = False
        self._set_task_called = True
        self.maxReachDist = np.float64(1.25)
        self.heightTarget = np.float32(0.125)
        self._obj_to_target_init = np.asarray(
            [0.025, -0.5, 1.75], dtype=np.float32
        )
        self.maxDist = 2.5

        self.np_random = np.random.default_rng(11)
        self.action_space = _FakeSpace(12)
        self.observation_space = _FakeSpace(13)

        # This stands in for MetaWorld fields that alias MjData/MjModel storage.
        self.view_backed_field = self.model.body_pos[0]


class _FakeMujoco:
    mjtState = SimpleNamespace(
        mjSTATE_INTEGRATION=1,
        mjSTATE_WARMSTART=2,
    )

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    @staticmethod
    def mj_stateSize(model: _FakeModel, spec: int) -> int:
        if spec == _FakeMujoco.mjtState.mjSTATE_INTEGRATION:
            return model.integration_size
        if spec == _FakeMujoco.mjtState.mjSTATE_WARMSTART:
            return model.warmstart_size
        raise AssertionError(f"unexpected state spec: {spec}")

    @staticmethod
    def mj_getState(
        _model: _FakeModel,
        data: _FakeData,
        destination: np.ndarray,
        spec: int,
    ) -> None:
        source = (
            data.integration
            if spec == _FakeMujoco.mjtState.mjSTATE_INTEGRATION
            else data.warmstart
        )
        destination[:] = source

    def mj_setState(
        self,
        model: _FakeModel,
        data: _FakeData,
        state: np.ndarray,
        spec: int,
    ) -> None:
        if spec == self.mjtState.mjSTATE_INTEGRATION:
            self.events.append(("integration", model.body_pos.copy()))
            data.integration = state.copy()
        elif spec == self.mjtState.mjSTATE_WARMSTART:
            self.events.append(("warmstart", None))
            data.warmstart = state.copy()
        else:  # pragma: no cover - defensive fake contract
            raise AssertionError(f"unexpected state spec: {spec}")

    def mj_forward(self, _model: _FakeModel, data: _FakeData) -> None:
        self.events.append(("forward", None))
        data.warmstart[:] = -999.0


@pytest.fixture
def fake_env(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_FakeEnv, _FakeMujoco]:
    fake_mujoco = _FakeMujoco()
    monkeypatch.setattr(
        metaworld_snapshot, "_require_mujoco", lambda: fake_mujoco
    )
    return _FakeEnv(), fake_mujoco


def test_fake_roundtrip_restores_exact_state_and_rngs_in_required_order(
    fake_env: tuple[_FakeEnv, _FakeMujoco],
) -> None:
    env, fake_mujoco = fake_env
    model_values = {
        name: np.asarray(getattr(env.model, name)).copy()
        for name in ("body_pos", "site_pos", "cam_pos", "cam_quat")
    }
    integration = env.data.integration.copy()
    warmstart = env.data.warmstart.copy()
    view_backed_field = env.view_backed_field

    snapshot = capture_metaworld_snapshot(env)
    expected_rng_draws = (
        env.np_random.integers(0, 2**31, size=4),
        env.action_space.np_random.integers(0, 2**31, size=4),
        env.observation_space.np_random.integers(0, 2**31, size=4),
    )

    assert snapshot.process_rng is None
    assert not snapshot.integration_state.flags.writeable
    assert not snapshot.model_arrays.body_pos.flags.writeable
    assert not np.shares_memory(snapshot.model_arrays.body_pos, env.model.body_pos)
    with pytest.raises(FrozenInstanceError):
        snapshot.environment_id = 0  # type: ignore[misc]

    for name in model_values:
        getattr(env.model, name)[:] = -100.0
    env.data.integration[:] = 100.0
    env.data.warmstart[:] = 200.0
    env.curr_path_length = 99
    env._prev_obs[:] = -1.0
    env._last_stable_obs = None
    env._did_see_sim_exception = True
    env.np_random.random(20)
    env.action_space.np_random.random(20)
    env.observation_space.np_random.random(20)

    restore_metaworld_snapshot(env, snapshot)

    assert [event[0] for event in fake_mujoco.events] == [
        "integration",
        "forward",
        "warmstart",
    ]
    np.testing.assert_array_equal(fake_mujoco.events[0][1], model_values["body_pos"])
    np.testing.assert_array_equal(env.data.integration, integration)
    np.testing.assert_array_equal(env.data.warmstart, warmstart)
    for name, expected in model_values.items():
        np.testing.assert_array_equal(getattr(env.model, name), expected)

    assert env.curr_path_length == 7
    np.testing.assert_array_equal(env._prev_obs, [1.5, 2.5, 3.5])
    np.testing.assert_array_equal(env._last_stable_obs, [4.5, 5.5])
    assert env._did_see_sim_exception is False
    assert env.view_backed_field is view_backed_field
    assert np.shares_memory(env.view_backed_field, env.model.body_pos)
    assert env._prev_obs.flags.writeable

    actual_rng_draws = (
        env.np_random.integers(0, 2**31, size=4),
        env.action_space.np_random.integers(0, 2**31, size=4),
        env.observation_space.np_random.integers(0, 2**31, size=4),
    )
    for actual, expected in zip(actual_rng_draws, expected_rng_draws):
        np.testing.assert_array_equal(actual, expected)


def test_restore_restores_mutated_task_arrays_and_scalars_bit_exactly(
    fake_env: tuple[_FakeEnv, _FakeMujoco],
) -> None:
    env, fake_mujoco = fake_env
    array_names = (
        "_last_rand_vec",
        "_target_pos",
        "obj_init_pos",
        "_obj_to_target_init",
    )
    scalar_names = (
        "_freeze_rand_vec",
        "_partially_observable",
        "_set_task_called",
        "maxReachDist",
        "heightTarget",
        "maxDist",
    )
    expected_arrays = {
        name: (
            getattr(env, name),
            getattr(env, name).dtype.str,
            getattr(env, name).shape,
            getattr(env, name).tobytes(order="C"),
        )
        for name in array_names
    }
    expected_scalars = {
        name: (
            np.asarray(getattr(env, name)).dtype.str,
            np.asarray(getattr(env, name)).shape,
            np.asarray(getattr(env, name)).tobytes(),
        )
        for name in scalar_names
    }
    snapshot = capture_metaworld_snapshot(env)

    env._last_rand_vec[:] = [-9.0, -8.0, -7.0, 9.0, 8.0, 7.0]
    env._obj_to_target_init[:] = np.asarray(
        [10.0, 20.0, 30.0], dtype=np.float32
    )
    env._freeze_rand_vec = False
    env._partially_observable = True
    env._set_task_called = False
    env.maxReachDist = np.float64(-0.0)
    env.heightTarget = np.float32(-12.5)
    env.maxDist = -0.0

    restore_metaworld_snapshot(env, snapshot)

    assert [event[0] for event in fake_mujoco.events] == [
        "integration",
        "forward",
        "warmstart",
    ]
    for name, (original, dtype, shape, payload) in expected_arrays.items():
        actual = getattr(env, name)
        assert actual is original
        assert actual.dtype.str == dtype
        assert actual.shape == shape
        assert actual.tobytes(order="C") == payload
    assert np.shares_memory(env.obj_init_pos, env._last_rand_vec)
    assert np.shares_memory(env._target_pos, env._last_rand_vec)
    for name, (dtype, shape, payload) in expected_scalars.items():
        actual = np.asarray(getattr(env, name))
        assert actual.dtype.str == dtype
        assert actual.shape == shape
        assert actual.tobytes() == payload


def test_restore_rejects_changed_task_attribute_presence_before_writing(
    fake_env: tuple[_FakeEnv, _FakeMujoco],
) -> None:
    env, fake_mujoco = fake_env
    snapshot = capture_metaworld_snapshot(env)
    del env._target_pos

    with pytest.raises(
        SnapshotMismatchError,
        match=r"task fingerprint attribute presence mismatch: _target_pos",
    ):
        restore_metaworld_snapshot(env, snapshot)

    assert fake_mujoco.events == []


def test_restore_rejects_different_environment_instance(
    fake_env: tuple[_FakeEnv, _FakeMujoco],
) -> None:
    env, fake_mujoco = fake_env
    snapshot = capture_metaworld_snapshot(env)

    with pytest.raises(SnapshotMismatchError, match="different environment instance"):
        restore_metaworld_snapshot(_FakeEnv(), snapshot)

    assert fake_mujoco.events == []


def test_restore_rejects_model_array_shape_mismatch_before_writing(
    fake_env: tuple[_FakeEnv, _FakeMujoco],
) -> None:
    env, fake_mujoco = fake_env
    snapshot = capture_metaworld_snapshot(env)
    env.model.body_pos = np.zeros((4, 3), dtype=np.float64)

    with pytest.raises(SnapshotMismatchError, match="array shape mismatch for body_pos"):
        restore_metaworld_snapshot(env, snapshot)

    assert fake_mujoco.events == []


def test_restore_rejects_mujoco_state_size_mismatch_before_writing(
    fake_env: tuple[_FakeEnv, _FakeMujoco],
) -> None:
    env, fake_mujoco = fake_env
    snapshot = capture_metaworld_snapshot(env)
    env.model.integration_size += 1

    with pytest.raises(SnapshotMismatchError, match="INTEGRATION size mismatch"):
        restore_metaworld_snapshot(env, snapshot)

    assert fake_mujoco.events == []


def test_process_rng_snapshot_restores_python_numpy_and_torch() -> None:
    torch = pytest.importorskip("torch")
    original = capture_process_rng_snapshot()
    try:
        random.seed(101)
        np.random.seed(102)
        torch.manual_seed(103)
        snapshot = capture_process_rng_snapshot()
        assert snapshot.torch_available
        assert snapshot.torch_cpu_state is not None
        assert not snapshot.torch_cpu_state.flags.writeable

        expected = (
            random.random(),
            np.random.random(5),
            torch.rand(5),
        )
        random.random()
        np.random.random(10)
        torch.rand(10)

        restore_process_rng_snapshot(snapshot)
        actual = (
            random.random(),
            np.random.random(5),
            torch.rand(5),
        )

        assert actual[0] == expected[0]
        np.testing.assert_array_equal(actual[1], expected[1])
        assert torch.equal(actual[2], expected[2])
    finally:
        restore_process_rng_snapshot(original)


def test_reach_v3_repeated_action_roundtrip_is_identical() -> None:
    metaworld = pytest.importorskip("metaworld")
    pytest.importorskip("mujoco")

    benchmark = metaworld.MT1("reach-v3", seed=7123)
    env_cls = benchmark.train_classes["reach-v3"]
    env = env_cls()
    try:
        env.set_task(benchmark.train_tasks[0])
        env.reset()
        env.step(np.asarray([0.1, -0.2, 0.05, 0.25], dtype=np.float32))
        snapshot = capture_metaworld_snapshot(env)
        action = np.asarray([-0.15, 0.05, 0.2, -0.4], dtype=np.float32)

        first = env.step(action)
        restore_metaworld_snapshot(env, snapshot)
        second = env.step(action)

        np.testing.assert_array_equal(first[0], second[0])
        assert first[1] == second[1]
        assert first[2:4] == second[2:4]
        assert first[4].keys() == second[4].keys()
        for key in first[4]:
            np.testing.assert_array_equal(np.asarray(first[4][key]), np.asarray(second[4][key]))
    finally:
        env.close()
