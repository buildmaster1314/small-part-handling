from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from fcpg.lewm_a4_task_signals import (
    A4DeploymentSignals,
    A4SignalError,
    POSE_DIM,
    TASKS,
    extract_a4_task_signals,
)


class FakeModel:
    def __init__(self) -> None:
        self._geoms = [
            SimpleNamespace(name="objGeom"),
            SimpleNamespace(name="rightpad_geom"),
            SimpleNamespace(name="table_top"),
            SimpleNamespace(name=""),
            SimpleNamespace(name=""),
            SimpleNamespace(name="drawer_handle_geom"),
            SimpleNamespace(name="towel_pad"),
        ]
        self._bodies = [
            SimpleNamespace(name="obj"),
            SimpleNamespace(name="rightpad"),
            SimpleNamespace(name="table"),
            SimpleNamespace(name="door_link"),
            SimpleNamespace(name="windowb_a"),
            SimpleNamespace(name="drawer"),
            SimpleNamespace(name="towel"),
        ]
        self.geom_bodyid = np.arange(len(self._geoms), dtype=np.int32)
        self._joints = {
            "drawer_slide": SimpleNamespace(range=np.asarray([0.0, 0.2])),
            "doorjoint": SimpleNamespace(range=np.asarray([-2.0, 0.0])),
            "window_slide": SimpleNamespace(range=np.asarray([0.0, 0.2])),
        }

    def geom(self, index: int) -> Any:
        return self._geoms[index]

    def body(self, index: int) -> Any:
        return self._bodies[index]

    def joint(self, name: str) -> Any:
        return self._joints[name]


class FakeData:
    def __init__(self) -> None:
        pose = SimpleNamespace(
            xpos=np.asarray([0.5, 0.0, 0.5]),
            xquat=np.asarray([1.0, 0.0, 0.0, 0.0]),
        )
        self._bodies = {
            "block": pose,
            "towel": pose,
            "windowb_a": pose,
        }
        self._joints = {
            "drawer_slide": SimpleNamespace(qpos=np.asarray([0.1])),
            "doorjoint": SimpleNamespace(qpos=np.asarray([-1.0])),
            "window_slide": SimpleNamespace(qpos=np.asarray([0.1])),
        }
        self.contact: list[Any] = []
        self.ncon = 0
        self.mocap_pos = np.asarray([[0.5, 0.0, 0.5]], dtype=np.float64)

    def body(self, name: str) -> Any:
        return self._bodies[name]

    def joint(self, name: str) -> Any:
        return self._joints[name]


class FakeEnv:
    def __init__(self) -> None:
        self.model = FakeModel()
        self.data = FakeData()
        self.tcp_center = np.asarray([0.5, 0.0, 0.5])
        self.hand_low = np.asarray([-0.5, -1.0, 0.0])
        self.hand_high = np.asarray([1.5, 1.0, 1.0])
        self.mocap_low = self.hand_low.copy()
        self.mocap_high = self.hand_high.copy()
        self._target_pos = np.asarray([1.0, 0.0, 0.5])
        self.obj_init_pos = np.asarray([0.0, 0.0, 0.5])
        self.maxReachDist = 1.0
        self.maxPullDist = 1.0
        self.window_handle_pos_init = self.obj_init_pos.copy()
        self.drawer_open_fraction = 1.0
        self.is_block_grasped = False
        self.towel_grasped = False
        self.touching_main_object = False
        self._did_see_sim_exception = False
        self.action_space = SimpleNamespace(low=np.full(4, -1.0, dtype=np.float32))

    def _get_pos_objects(self) -> np.ndarray:
        return np.asarray([0.5, 0.0, 0.5])

    def _get_quat_objects(self) -> np.ndarray:
        return np.asarray([0.0, 0.0, 0.0, 1.0])

    def evaluate_state(
        self, _observation: np.ndarray, action: np.ndarray
    ) -> tuple[float, dict[str, float]]:
        assert np.array_equal(action, np.zeros(4, dtype=np.float32))
        return 0.0, {"success": 0.0, "in_place_reward": 0.4}


def _observation() -> np.ndarray:
    result = np.zeros(39, dtype=np.float64)
    result[:4] = [0.5, 0.0, 0.5, 0.75]
    return result


def _info(*, success: float = 0.0) -> dict[str, float]:
    return {"success": success, "in_place_reward": 0.4}


def _contact(env: FakeEnv, geom1: int, geom2: int, distance: float) -> None:
    env.data.contact = [
        SimpleNamespace(geom1=geom1, geom2=geom2, dist=distance)
    ]
    env.data.ncon = 1


@pytest.mark.parametrize("task_name", TASKS)
def test_all_seven_tasks_return_fixed_shapes_and_dtypes(task_name: str) -> None:
    result = extract_a4_task_signals(task_name, FakeEnv(), _observation(), _info())

    assert result.deployment.task_onehot.shape == (7,)
    assert result.deployment.proprio.shape == (4,)
    assert result.deployment.task_onehot.dtype == np.float32
    assert result.deployment.proprio.dtype == np.float32
    assert result.deployment.task_onehot[TASKS.index(task_name)] == 1.0
    assert result.deployment.task_onehot.sum() == 1.0
    assert not result.deployment.task_onehot.flags.writeable
    assert not result.deployment.proprio.flags.writeable

    diagnostics = result.diagnostics
    assert diagnostics.object_pose.shape == (POSE_DIM,)
    assert diagnostics.reference_pose.shape == (POSE_DIM,)
    assert diagnostics.branch_outcome.shape == (2,)
    assert diagnostics.object_pose.dtype == np.float32
    assert diagnostics.reference_pose.dtype == np.float32
    assert diagnostics.branch_outcome.dtype == np.float32
    np.testing.assert_allclose(diagnostics.object_pose, [0.5, 0.0, 0.5, 0, 0, 0, 1])
    np.testing.assert_allclose(diagnostics.reference_pose, [1.0, 0.0, 0.5, 0, 0, 0, 1])
    assert isinstance(diagnostics.active_stage, np.bool_)
    assert isinstance(diagnostics.task_progress, np.float32)
    assert isinstance(diagnostics.goal_relative_progress, np.float32)
    assert isinstance(diagnostics.contact_or_grasp_state, np.bool_)
    assert isinstance(diagnostics.success, np.bool_)
    assert isinstance(diagnostics.workspace_and_joint_margin, np.float32)
    assert isinstance(diagnostics.unexpected_collision, np.bool_)
    assert isinstance(diagnostics.severe_penetration, np.bool_)
    assert diagnostics.task_progress == pytest.approx(
        0.55 if task_name == "drawer-retrieve-v0" else 0.4
    )
    assert diagnostics.goal_relative_progress == pytest.approx(
        0.4 if task_name == "towel-wipe-v0" else 0.5
    )
    assert diagnostics.workspace_and_joint_margin == pytest.approx(0.5)
    assert result.eligibility_fields() == {
        "success": False,
        "active_stage": True,
        "workspace_and_joint_margin": pytest.approx(0.5),
        "unexpected_collision": False,
    }


def test_deployment_record_cannot_contain_privileged_fields() -> None:
    names = {field.name for field in fields(A4DeploymentSignals)}

    assert names == {"task_onehot", "proprio"}
    assert names.isdisjoint(
        {
            "object_pose",
            "reference_pose",
            "task_progress",
            "goal_relative_progress",
            "contact_or_grasp_state",
            "success",
            "branch_outcome",
            "unexpected_collision",
            "severe_penetration",
        }
    )


def test_success_is_branch_outcome_and_closes_active_stage() -> None:
    result = extract_a4_task_signals(
        "pick-place-v3", FakeEnv(), _observation(), _info(success=1.0)
    )

    assert result.diagnostics.success
    assert not result.diagnostics.active_stage
    np.testing.assert_allclose(result.diagnostics.branch_outcome, [0.4, 1.0])


@pytest.mark.parametrize(
    ("task_name", "geom_id", "attribute"),
    [
        ("pick-place-v3", 0, "touching_main_object"),
        ("push-v3", 0, "touching_main_object"),
        ("door-open-v3", 3, None),
        ("window-open-v3", 4, None),
        ("drawer-retrieve-v0", 5, "is_block_grasped"),
        ("towel-wipe-v0", 6, "towel_grasped"),
    ],
)
def test_expected_task_contact_or_grasp_is_not_a_collision(
    task_name: str, geom_id: int, attribute: str | None
) -> None:
    env = FakeEnv()
    _contact(env, 1, geom_id, -0.01)
    if attribute is not None:
        setattr(env, attribute, True)

    diagnostics = extract_a4_task_signals(
        task_name, env, _observation(), _info()
    ).diagnostics

    assert diagnostics.contact_or_grasp_state
    assert not diagnostics.unexpected_collision
    assert not diagnostics.severe_penetration


def test_reach_has_no_synthetic_contact_indicator() -> None:
    env = FakeEnv()
    _contact(env, 1, 0, -0.01)

    diagnostics = extract_a4_task_signals(
        "reach-v3", env, _observation(), _info()
    ).diagnostics

    assert not diagnostics.contact_or_grasp_state
    assert diagnostics.unexpected_collision
    assert diagnostics.severe_penetration


@pytest.mark.parametrize(
    ("distance", "unexpected", "severe"),
    [(-0.001, True, False), (-0.003, True, True)],
)
def test_unexpected_robot_workspace_contact_tracks_severity(
    distance: float, unexpected: bool, severe: bool
) -> None:
    env = FakeEnv()
    _contact(env, 1, 2, distance)

    diagnostics = extract_a4_task_signals(
        "pick-place-v3", env, _observation(), _info()
    ).diagnostics

    assert bool(diagnostics.unexpected_collision) is unexpected
    assert bool(diagnostics.severe_penetration) is severe


def test_simulation_exception_sets_both_safety_flags() -> None:
    env = FakeEnv()
    env._did_see_sim_exception = True

    diagnostics = extract_a4_task_signals(
        "reach-v3", env, _observation(), _info()
    ).diagnostics

    assert diagnostics.unexpected_collision
    assert diagnostics.severe_penetration


def test_articulated_task_joint_endpoint_does_not_reject_safe_workspace() -> None:
    env = FakeEnv()
    env.data.joint("window_slide").qpos[0] = 0.01

    diagnostics = extract_a4_task_signals(
        "window-open-v3", env, _observation(), _info()
    ).diagnostics

    assert diagnostics.workspace_and_joint_margin == pytest.approx(0.5)


def test_one_saturated_mocap_direction_preserves_planar_control_margin() -> None:
    env = FakeEnv()
    env.data.mocap_pos[0, 2] = env.mocap_low[2]

    diagnostics = extract_a4_task_signals(
        "push-v3", env, _observation(), _info()
    ).diagnostics

    assert diagnostics.workspace_and_joint_margin == pytest.approx(0.5)


def test_two_saturated_mocap_directions_fail_control_margin() -> None:
    env = FakeEnv()
    env.data.mocap_pos[0, 0] = env.mocap_low[0]
    env.data.mocap_pos[0, 2] = env.mocap_low[2]

    diagnostics = extract_a4_task_signals(
        "push-v3", env, _observation(), _info()
    ).diagnostics

    assert diagnostics.workspace_and_joint_margin == pytest.approx(0.0)


def test_true_out_of_bounds_mocap_position_fails_control_margin() -> None:
    env = FakeEnv()
    env.data.mocap_pos[0, 0] = env.mocap_low[0] - 1e-3

    diagnostics = extract_a4_task_signals(
        "push-v3", env, _observation(), _info()
    ).diagnostics

    assert diagnostics.workspace_and_joint_margin == pytest.approx(0.0)


def test_tcp_offset_does_not_change_mocap_control_margin() -> None:
    env = FakeEnv()
    env.tcp_center[:] = [-10.0, -10.0, -10.0]

    diagnostics = extract_a4_task_signals(
        "towel-wipe-v0", env, _observation(), _info()
    ).diagnostics

    assert diagnostics.workspace_and_joint_margin == pytest.approx(0.5)


def test_missing_step_values_use_environment_evaluation() -> None:
    result = extract_a4_task_signals("push-v3", FakeEnv(), _observation(), {})

    assert result.diagnostics.task_progress == pytest.approx(0.4)
    assert not result.diagnostics.success


@pytest.mark.parametrize("task_name", ("drawer-open-v3", "sweep-v3", ""))
def test_only_exact_frozen_task_panel_is_accepted(task_name: str) -> None:
    with pytest.raises(ValueError, match="unsupported A4 task"):
        extract_a4_task_signals(task_name, FakeEnv(), _observation(), _info())


def test_missing_required_signal_fails_clearly() -> None:
    env = FakeEnv()
    del env._target_pos

    with pytest.raises(A4SignalError, match="_target_pos"):
        extract_a4_task_signals("pick-place-v3", env, _observation(), _info())


def test_missing_info_and_evaluator_fails_clearly() -> None:
    env = FakeEnv()
    env.evaluate_state = None  # type: ignore[method-assign]

    with pytest.raises(A4SignalError, match="in_place_reward"):
        extract_a4_task_signals("push-v3", env, _observation(), {})


def test_malformed_observation_fails_before_environment_access() -> None:
    with pytest.raises(A4SignalError, match=r"shape \(39,\)"):
        extract_a4_task_signals("reach-v3", object(), np.zeros(38), _info())
