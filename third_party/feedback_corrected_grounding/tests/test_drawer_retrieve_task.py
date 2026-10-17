from __future__ import annotations

import os
from dataclasses import FrozenInstanceError

import numpy as np
import pytest


os.environ.setdefault("MUJOCO_GL", "egl")
metaworld = pytest.importorskip("metaworld")
mujoco = pytest.importorskip("mujoco")

from fcpg.custom_tasks import (  # noqa: E402
    CUSTOM_TASK_REGISTRY,
    DRAWER_RETRIEVE_TASK_NAME,
    DrawerRetrievePhase,
    DrawerRetrievePhysicalStateExpert,
    SawyerDrawerRetrieveEnvV0,
    available_custom_tasks,
    make_custom_task,
    make_drawer_retrieve_env,
)
from fcpg.metaworld_snapshot import (  # noqa: E402
    capture_metaworld_snapshot,
    restore_metaworld_snapshot,
)


@pytest.fixture
def env() -> SawyerDrawerRetrieveEnvV0:
    environment = make_drawer_retrieve_env(
        render_mode="rgb_array",
        width=96,
        height=80,
        seed=7,
    )
    yield environment
    environment.close()


def _expert_observation(
    *,
    hand: tuple[float, float, float],
    aperture: float,
    block: tuple[float, float, float],
    handle: tuple[float, float, float] = (0.0, 0.46, 0.112),
    target: tuple[float, float, float] = (0.0, 0.78, 0.228),
) -> np.ndarray:
    observation = np.zeros(39, dtype=np.float64)
    observation[:3] = hand
    observation[3] = aperture
    observation[4:7] = block
    observation[7] = 1.0
    observation[11:14] = handle
    observation[14] = 1.0
    observation[-3:] = target
    return observation


def test_local_registry_constructs_ready_to_reset_environment() -> None:
    assert DRAWER_RETRIEVE_TASK_NAME in available_custom_tasks()
    assert CUSTOM_TASK_REGISTRY[DRAWER_RETRIEVE_TASK_NAME] is SawyerDrawerRetrieveEnvV0

    environment = make_custom_task(
        DRAWER_RETRIEVE_TASK_NAME, task_instance=9, seed=3
    )
    try:
        assert isinstance(environment, SawyerDrawerRetrieveEnvV0)
        assert environment.task_name == DRAWER_RETRIEVE_TASK_NAME
        assert environment.task_instance == 9
        observation, info = environment.reset(seed=3)
        assert environment.observation_space.contains(observation)
        assert info["task_instance"] == 9
        assert info["success"] == 0.0
    finally:
        environment.close()

    with pytest.raises(KeyError, match="Unknown custom task"):
        make_custom_task("not-a-local-task")


@pytest.mark.parametrize("task_instance", (-1, 48))
def test_task_instance_rejects_out_of_range_ids(task_instance: int) -> None:
    with pytest.raises(ValueError, match="0..47"):
        make_drawer_retrieve_env(task_instance=task_instance)


@pytest.mark.parametrize("task_instance", (True, 1.5, "1"))
def test_task_instance_rejects_non_integer_ids(task_instance: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        make_drawer_retrieve_env(task_instance=task_instance)  # type: ignore[arg-type]


def test_all_48_task_instances_have_unique_safe_reproducible_resets() -> None:
    reset_pairs: set[tuple[float, ...]] = set()
    block_positions: set[tuple[float, ...]] = set()
    target_positions: set[tuple[float, ...]] = set()

    assert SawyerDrawerRetrieveEnvV0.TASK_INSTANCE_COUNT == 48
    outer_top_center = np.asarray(SawyerDrawerRetrieveEnvV0.OUTER_TOP_CENTER)
    outer_top_half_size = np.asarray(SawyerDrawerRetrieveEnvV0.OUTER_TOP_HALF_SIZE)
    target_half_size = np.asarray(SawyerDrawerRetrieveEnvV0.TARGET_HALF_SIZE)
    for task_instance in range(SawyerDrawerRetrieveEnvV0.TASK_INSTANCE_COUNT):
        expected_block, expected_target = (
            SawyerDrawerRetrieveEnvV0.task_instance_reset_parameters(task_instance)
        )
        environment = make_drawer_retrieve_env(
            task_instance=task_instance, seed=task_instance
        )
        try:
            first_observation, first_info = environment.reset(seed=task_instance)
            first_block = environment.block_position
            first_target = first_observation[-3:].copy()

            np.testing.assert_allclose(first_block, expected_block, atol=1e-12)
            np.testing.assert_allclose(first_target, expected_target, atol=1e-12)
            assert environment.observation_space.contains(first_observation)
            assert first_info["task_instance"] == task_instance
            assert first_info["success"] == 0.0
            assert not environment.is_success()
            assert environment.drawer_open_amount == pytest.approx(0.0, abs=1e-12)

            assert -0.056 <= first_block[0] <= 0.056
            assert 0.795 <= first_block[1] <= 0.835
            assert first_block[2] == pytest.approx(0.052, abs=1e-12)
            target_offset = np.abs(first_target[:2] - outer_top_center[:2])
            assert bool(
                np.all(
                    target_offset + target_half_size[:2]
                    < outer_top_half_size[:2]
                )
            )
            assert first_target[2] == pytest.approx(
                SawyerDrawerRetrieveEnvV0.OUTER_TOP_SURFACE_Z
                + SawyerDrawerRetrieveEnvV0.BLOCK_HALF_EXTENT,
                abs=1e-12,
            )
            np.testing.assert_allclose(
                environment.data.site("goal").xpos,
                environment.goal_marker_position,
                atol=1e-12,
            )

            second_observation, second_info = environment.reset(
                seed=10_000 - task_instance
            )
            np.testing.assert_array_equal(second_observation, first_observation)
            assert second_info["task_instance"] == task_instance
            assert second_info["success"] == 0.0
        finally:
            environment.close()

        block_key = tuple(float(value) for value in first_block)
        target_key = tuple(float(value) for value in first_target)
        block_positions.add(block_key)
        target_positions.add(target_key)
        reset_pairs.add(block_key + target_key)

    assert len(block_positions) == 48
    assert len(target_positions) == 48
    assert len(reset_pairs) == 48


def test_model_uses_matched_primitives_and_flat_top_goal_geometry(
    env: SawyerDrawerRetrieveEnvV0,
) -> None:
    env.reset()
    green_names = ("outer_left", "outer_right", "outer_back", "outer_top")
    red_names = (
        "drawer_bottom",
        "drawer_front",
        "drawer_left",
        "drawer_right",
        "drawer_back",
        "drawer_handle_left_support",
        "drawer_handle_right_support",
        "drawer_handle_geom",
    )

    for name in green_names + red_names + ("objGeom",):
        geom = env.model.geom(name)
        assert geom.type == mujoco.mjtGeom.mjGEOM_BOX
        assert geom.group == 0
        assert geom.contype != 0
        assert geom.conaffinity != 0

    for name in green_names:
        np.testing.assert_allclose(env.model.geom(name).rgba, [0.08, 0.62, 0.22, 1])
    for name in red_names:
        np.testing.assert_allclose(env.model.geom(name).rgba, [0.82, 0.08, 0.08, 1])

    assert env.model.joint("drawer_slide").type == mujoco.mjtJoint.mjJNT_SLIDE
    assert env.model.joint("block_free").type == mujoco.mjtJoint.mjJNT_FREE
    goal = env.model.site("goal")
    assert int(goal.type[0]) == mujoco.mjtGeom.mjGEOM_BOX
    np.testing.assert_allclose(goal.size, env.TARGET_HALF_SIZE)
    np.testing.assert_allclose(goal.rgba, [0.12, 0.35, 0.95, 0.45])

    outer_top = env.data.geom("outer_top")
    top_surface_z = outer_top.xpos[2] + env.model.geom("outer_top").size[2]
    goal_position = env.data.site("goal").xpos
    assert goal_position[2] - goal.size[2] == pytest.approx(top_surface_z)
    np.testing.assert_allclose(goal_position[:2], env.TARGET_POS[:2])
    assert env.TARGET_POS[2] == pytest.approx(
        top_surface_z + env.model.geom("objGeom").size[2]
    )


def test_reset_render_step_and_success_contract(
    env: SawyerDrawerRetrieveEnvV0,
) -> None:
    observation, reset_info = env.reset(seed=11)
    assert observation.shape == (39,)
    assert observation.dtype == np.float64
    assert env.observation_space.contains(observation)
    np.testing.assert_allclose(observation[4:7], env.block_position)
    np.testing.assert_allclose(observation[11:14], env.drawer_handle_position)
    np.testing.assert_allclose(observation[-3:], env.TARGET_POS)
    assert reset_info["task_instance"] == 0
    assert reset_info["success"] == 0.0
    assert env.drawer_open_amount == pytest.approx(0.0, abs=1e-12)
    assert not env.success

    frame = env.render()
    assert frame.shape == (80, 96, 3)
    assert frame.dtype == np.uint8
    assert float(frame.std()) > 1.0

    next_obs, reward, terminated, truncated, info = env.step(
        np.zeros(4, dtype=np.float32)
    )
    assert next_obs.shape == (39,)
    assert np.isfinite(reward)
    assert terminated is False
    assert truncated is False
    assert info["task_instance"] == 0
    assert info["success"] == 0.0
    assert info["drawer_open_amount"] >= 0.0

    _, reset_info = env.reset()
    assert reset_info["success"] == 0.0
    assert env.drawer_open_amount == pytest.approx(0.0, abs=1e-12)


def test_grasped_block_at_supported_target_is_not_success(
    env: SawyerDrawerRetrieveEnvV0,
) -> None:
    observation, _ = env.reset(seed=23)
    expert = DrawerRetrievePhysicalStateExpert()

    for _ in range(300):
        if expert.phase(observation) is DrawerRetrievePhase.RELEASE:
            env._set_block_pose(observation[-3:])
            mujoco.mj_forward(env.model, env.data)

            assert env.drawer_open_amount >= env.DRAWER_OPEN_THRESHOLD
            assert env.is_block_in_target
            assert env.is_block_supported_by_outer_top
            assert env.is_block_grasped
            assert not env.is_block_released
            assert not env.is_success()

            _, reward, _, _, info = env.step(
                np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            )
            assert env.is_block_grasped
            assert info["block_in_target"] == 1.0
            assert info["block_supported"] == 1.0
            assert info["block_released"] == 0.0
            assert info["success"] == 0.0
            assert reward < 10.0
            break

        observation, _, _, _, _ = env.step(expert.get_action(observation))
    else:
        pytest.fail("expert never reached the held-at-target release phase")


def test_released_block_at_supported_target_is_success(
    env: SawyerDrawerRetrieveEnvV0,
) -> None:
    observation, _ = env.reset(seed=29)
    env.data.joint("drawer_slide").qpos[0] = 0.18
    env.data.joint("drawer_slide").qvel[0] = 0.0
    env._set_block_pose(observation[-3:])
    mujoco.mj_forward(env.model, env.data)

    assert env.is_block_in_target
    assert env.is_block_supported_by_outer_top
    assert env.is_block_released
    assert not env.is_block_grasped
    assert env.is_success()

    _, reward, _, _, info = env.step(
        np.asarray([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    )
    assert reward == 10.0
    assert info["block_in_target"] == 1.0
    assert info["block_supported"] == 1.0
    assert info["block_released"] == 1.0
    assert info["success"] == 1.0


def test_snapshot_restore_replays_the_same_physical_step(
    env: SawyerDrawerRetrieveEnvV0,
) -> None:
    env.reset(seed=19)
    snapshot = capture_metaworld_snapshot(env)
    action = np.asarray([0.2, -0.1, 0.15, -0.5], dtype=np.float32)

    first = env.step(action)
    restore_metaworld_snapshot(env, snapshot)
    second = env.step(action)

    np.testing.assert_array_equal(first[0], second[0])
    assert first[1:4] == second[1:4]
    assert first[4] == second[4]


@pytest.mark.parametrize(
    ("observation", "expected"),
    [
        (
            _expert_observation(
                hand=(0.0, 0.60, 0.20),
                aperture=1.0,
                block=(0.0, 0.82, 0.052),
                handle=(0.0, 0.66, 0.112),
            ),
            DrawerRetrievePhase.PULL_OPEN,
        ),
        (
            _expert_observation(
                hand=(0.0, 0.47, 0.15),
                aperture=1.0,
                block=(0.0, 0.64, 0.052),
            ),
            DrawerRetrievePhase.LIFT_AFTER_OPEN,
        ),
        (
            _expert_observation(
                hand=(0.20, 0.40, 0.28),
                aperture=1.0,
                block=(0.0, 0.64, 0.052),
            ),
            DrawerRetrievePhase.MOVE_ABOVE_BLOCK,
        ),
        (
            _expert_observation(
                hand=(0.0, 0.64, 0.18),
                aperture=1.0,
                block=(0.0, 0.64, 0.052),
            ),
            DrawerRetrievePhase.DESCEND,
        ),
        (
            _expert_observation(
                hand=(0.0, 0.64, 0.06),
                aperture=1.0,
                block=(0.0, 0.64, 0.052),
            ),
            DrawerRetrievePhase.GRASP,
        ),
        (
            _expert_observation(
                hand=(0.0, 0.64, 0.06),
                aperture=0.5,
                block=(0.0, 0.64, 0.052),
            ),
            DrawerRetrievePhase.LIFT_BLOCK,
        ),
        (
            _expert_observation(
                hand=(0.0, 0.64, 0.15),
                aperture=0.5,
                block=(0.0, 0.64, 0.13),
            ),
            DrawerRetrievePhase.LIFT_BLOCK,
        ),
        (
            _expert_observation(
                hand=(0.0, 0.64, 0.36),
                aperture=0.5,
                block=(0.0, 0.64, 0.33),
            ),
            DrawerRetrievePhase.TRANSPORT_ABOVE,
        ),
        (
            _expert_observation(
                hand=(0.0, 0.78, 0.35),
                aperture=0.5,
                block=(0.0, 0.78, 0.32),
            ),
            DrawerRetrievePhase.LOWER,
        ),
        (
            _expert_observation(
                hand=(0.0, 0.78, 0.258),
                aperture=0.5,
                block=(0.0, 0.78, 0.228),
            ),
            DrawerRetrievePhase.RELEASE,
        ),
        (
            _expert_observation(
                hand=(0.0, 0.78, 0.27),
                aperture=0.9,
                block=(0.0, 0.78, 0.228),
            ),
            DrawerRetrievePhase.RETREAT,
        ),
    ],
)
def test_expert_phase_is_derived_only_from_current_physical_state(
    observation: np.ndarray,
    expected: DrawerRetrievePhase,
) -> None:
    expert = DrawerRetrievePhysicalStateExpert()
    assert expert.phase(observation) is expected
    assert expert.phase(observation.copy()) is expected

    action = expert.get_action(observation)
    assert action.shape == (4,)
    assert action.dtype == np.float32
    assert np.isfinite(action).all()
    assert bool((np.abs(action) <= 1.0).all())
    expected_gripper = 1.0 if expected in {
        DrawerRetrievePhase.GRASP,
        DrawerRetrievePhase.LIFT_BLOCK,
        DrawerRetrievePhase.TRANSPORT_ABOVE,
        DrawerRetrievePhase.LOWER,
    } else -1.0
    assert action[3] == expected_gripper


def test_expert_configuration_is_immutable_and_rejects_nonphysical_input() -> None:
    expert = DrawerRetrievePhysicalStateExpert()
    with pytest.raises(FrozenInstanceError):
        expert.open_handle_y = 0.0  # type: ignore[misc]
    with pytest.raises(ValueError, match="shape"):
        expert.phase(np.zeros(38))
    invalid = np.zeros(39)
    invalid[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        expert.phase(invalid)


@pytest.mark.parametrize("task_instance", (0, 7, 8, 18, 37, 47))
def test_expert_completes_representative_physical_task_instances(
    task_instance: int,
) -> None:
    env = make_drawer_retrieve_env(task_instance=task_instance, seed=23)
    observation, _ = env.reset(seed=23)
    expert = DrawerRetrievePhysicalStateExpert()
    phases: set[DrawerRetrievePhase] = set()
    first_open_step: int | None = None
    first_lift_after_open_step: int | None = None
    shell_geom_names = {
        "outer_left",
        "outer_right",
        "outer_back",
        "outer_top",
        "drawer_bottom",
        "drawer_front",
        "drawer_left",
        "drawer_right",
        "drawer_back",
        "drawer_handle_left_support",
        "drawer_handle_right_support",
        "drawer_handle_geom",
    }
    shell_geom_ids = {int(env.model.geom(name).id) for name in shell_geom_names}
    worst_robot_shell_penetration = 0.0

    try:
        for step in range(500):
            phase = expert.phase(observation)
            phases.add(phase)
            if env.drawer_open_amount >= env.DRAWER_OPEN_THRESHOLD:
                first_open_step = step if first_open_step is None else first_open_step
            if phase is DrawerRetrievePhase.LIFT_AFTER_OPEN:
                first_lift_after_open_step = (
                    step
                    if first_lift_after_open_step is None
                    else first_lift_after_open_step
                )
            if phase is DrawerRetrievePhase.RELEASE:
                assert not env.is_success()
            observation, _, _, _, info = env.step(expert.get_action(observation))
            for contact_index in range(env.data.ncon):
                contact = env.data.contact[contact_index]
                geom1 = int(np.asarray(contact.geom1).item())
                geom2 = int(np.asarray(contact.geom2).item())
                if (geom1 in shell_geom_ids) == (geom2 in shell_geom_ids):
                    continue
                other_geom = geom2 if geom1 in shell_geom_ids else geom1
                if int(env.model.geom_group[other_geom]) == 4:
                    worst_robot_shell_penetration = min(
                        worst_robot_shell_penetration,
                        float(contact.dist),
                    )
            assert info["task_instance"] == task_instance
            if info["success"]:
                break
        else:
            pytest.fail(
                "drawer-retrieve expert did not complete within 500 steps: "
                f"instance={task_instance}, drawer={env.drawer_open_amount:.4f}, "
                f"block_to_target={env.block_to_target:.4f}"
            )

        assert first_open_step is not None
        assert first_lift_after_open_step is not None
        assert first_lift_after_open_step >= first_open_step
        assert DrawerRetrievePhase.PULL_OPEN in phases
        assert DrawerRetrievePhase.GRASP in phases
        assert DrawerRetrievePhase.LIFT_BLOCK in phases
        assert DrawerRetrievePhase.TRANSPORT_ABOVE in phases
        assert DrawerRetrievePhase.LOWER in phases
        assert DrawerRetrievePhase.RELEASE in phases
        assert DrawerRetrievePhase.RETREAT in phases
        assert env.drawer_open_amount >= env.DRAWER_OPEN_THRESHOLD
        assert env.is_block_in_target
        assert env.is_block_supported_by_outer_top
        assert env.is_block_released
        assert not env.is_block_grasped
        assert worst_robot_shell_penetration >= -0.002
    finally:
        env.close()
