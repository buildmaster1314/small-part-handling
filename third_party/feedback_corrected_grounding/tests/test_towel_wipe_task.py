from __future__ import annotations

import os

import mujoco
import numpy as np
import pytest


os.environ.setdefault("MUJOCO_GL", "egl")
pytest.importorskip("metaworld")

from fcpg.custom_tasks import (  # noqa: E402
    CUSTOM_TASK_REGISTRY,
    TOWEL_WIPE_TASK_NAME,
    SawyerTowelWipeEnvV0,
    TowelWipePhysicalStateExpert,
    make_towel_wipe_env,
)
from fcpg.metaworld_snapshot import (  # noqa: E402
    capture_metaworld_snapshot,
    restore_metaworld_snapshot,
)


@pytest.fixture
def env() -> SawyerTowelWipeEnvV0:
    environment = make_towel_wipe_env(
        render_mode="rgb_array",
        camera_name="towel_view",
        width=96,
        height=80,
        seed=7,
    )
    yield environment
    environment.close()


def test_registry_exposes_towel_environment_and_expert_contract() -> None:
    assert CUSTOM_TASK_REGISTRY[TOWEL_WIPE_TASK_NAME] is SawyerTowelWipeEnvV0
    expert = TowelWipePhysicalStateExpert()
    assert callable(expert.get_action)


def test_all_48_task_instances_define_unique_safe_wipe_lanes() -> None:
    pairs: set[tuple[float, ...]] = set()
    for task_instance in range(SawyerTowelWipeEnvV0.TASK_INSTANCE_COUNT):
        left, right = SawyerTowelWipeEnvV0.task_instance_endpoints(task_instance)
        assert left[0] < 0 < right[0]
        assert left[0] == pytest.approx(-right[0])
        assert left[1] == pytest.approx(right[1])
        assert left[2] == pytest.approx(SawyerTowelWipeEnvV0.WIPE_HEIGHT)
        assert right[2] == pytest.approx(SawyerTowelWipeEnvV0.WIPE_HEIGHT)
        assert 0.15 <= abs(left[0]) <= 0.22
        assert 0.50 <= left[1] <= 0.70
        pairs.add(tuple(left) + tuple(right))
    assert len(pairs) == 48


@pytest.mark.parametrize("task_instance", (-1, 48))
def test_task_instance_rejects_out_of_range_ids(task_instance: int) -> None:
    with pytest.raises(ValueError, match="0..47"):
        make_towel_wipe_env(task_instance=task_instance)


@pytest.mark.parametrize("task_instance", (True, 1.5, "1"))
def test_task_instance_rejects_non_integer_ids(task_instance: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        make_towel_wipe_env(task_instance=task_instance)  # type: ignore[arg-type]


def test_reset_render_and_physical_tool_geometry(env: SawyerTowelWipeEnvV0) -> None:
    observation, info = env.reset(seed=11)
    assert observation.shape == (39,)
    assert env.observation_space.contains(observation)
    np.testing.assert_allclose(observation[4:7], env.towel_position)
    np.testing.assert_allclose(observation[-3:][:2], env.left_endpoint[:2])
    assert observation[-1] > env.INITIAL_GRASP_LIFT_HEIGHT
    assert info["success"] == 0.0
    assert info["wipe_half_cycles"] == 0
    assert info["towel_grasped"] == 0.0
    assert not env.towel_surface_engaged

    towel = env.model.geom("towel_pad")
    handle = env.model.geom("objGeom")
    assert towel.type == mujoco.mjtGeom.mjGEOM_BOX
    assert handle.type == mujoco.mjtGeom.mjGEOM_BOX
    assert env.model.joint("towel_free").type == mujoco.mjtJoint.mjJNT_FREE
    assert towel.contype != 0 and towel.conaffinity != 0
    assert handle.contype != 0 and handle.conaffinity != 0
    np.testing.assert_allclose(env.data.body("towel").xmat.reshape(3, 3)[:, 2], [0, 0, 1])

    frame = env.render()
    assert frame.shape == (80, 96, 3)
    assert frame.dtype == np.uint8
    assert float(frame.std()) > 1.0


def test_ungrasped_endpoint_motion_cannot_increment_wipe_progress(
    env: SawyerTowelWipeEnvV0,
) -> None:
    env.reset(seed=13)
    env._grasp_lifted = True
    env._target_pos = env.left_endpoint
    env._set_towel_pose(env.left_endpoint)
    env._advance_wipe_state()
    assert not env._wipe_started
    assert env.wipe_half_cycles == 0
    assert not env.towel_grasped
    assert not env.towel_surface_engaged


def test_expert_completes_exactly_three_round_trips_and_lifts_tool(
    env: SawyerTowelWipeEnvV0,
) -> None:
    observation, _ = env.reset(seed=17)
    expert = TowelWipePhysicalStateExpert()
    first_success: int | None = None
    observed_sides: list[str] = []
    observed_phases: set[str] = set()
    previous_half_cycles = 0
    for step in range(350):
        observed_phases.add(expert.get_phase(observation).value)
        action = np.clip(
            expert.get_action(observation), env.action_space.low, env.action_space.high
        ).astype(np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        observed_sides.append(str(info["current_target_side"]))
        assert not terminated and not truncated
        if int(info["wipe_half_cycles"]) > previous_half_cycles:
            assert info["towel_grasped"] == 1.0
            assert info["towel_surface_engaged"] == 1.0
            previous_half_cycles = int(info["wipe_half_cycles"])
        if info["success"]:
            first_success = step + 1
            assert reward == 10.0
            break

    assert first_success is not None
    assert env.wipe_half_cycles == 6
    assert env.wipe_round_trips == pytest.approx(3.0)
    assert env.towel_position[2] >= env.SUCCESS_LIFT_HEIGHT
    assert env.towel_grasped
    assert {
        "grasp_towel",
        "lift_towel",
        "lower_to_start",
        "wipe_right",
        "wipe_left",
        "lift_after_wipe",
    }.issubset(observed_phases)
    assert "left" in observed_sides and "right" in observed_sides
    assert observed_sides[-1] == "finish"


def test_snapshot_restore_replays_wipe_progress_and_same_step(
    env: SawyerTowelWipeEnvV0,
) -> None:
    observation, _ = env.reset(seed=19)
    expert = TowelWipePhysicalStateExpert()
    for _ in range(70):
        action = np.clip(
            expert.get_action(observation), env.action_space.low, env.action_space.high
        ).astype(np.float32)
        observation, *_ = env.step(action)
    snapshot = capture_metaworld_snapshot(env)
    action = np.clip(
        expert.get_action(observation), env.action_space.low, env.action_space.high
    ).astype(np.float32)

    first = env.step(action)
    first_progress = (env._wipe_started, env.wipe_half_cycles, env.current_target_side)
    restore_metaworld_snapshot(env, snapshot)
    second = env.step(action)
    second_progress = (env._wipe_started, env.wipe_half_cycles, env.current_target_side)

    np.testing.assert_array_equal(first[0], second[0])
    assert first[1:4] == second[1:4]
    assert first[4] == second[4]
    assert first_progress == second_progress
