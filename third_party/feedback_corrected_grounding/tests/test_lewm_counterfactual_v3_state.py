from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fcpg.lewm_counterfactual_v3_state import (
    LEFT_CONTACT_INDEX,
    LEFT_FORCE_INDEX,
    OBJECT_INITIAL_POSITION_SLICE,
    OBJECT_POSITION_SLICE,
    OBJECT_QUATERNION_SLICE,
    ORACLE_HORIZON,
    RAW_CTRL_SLICE,
    RAW_MOCAP_POSITION_SLICE,
    RAW_MOCAP_QUATERNION_SLICE,
    RAW_QPOS_SLICE,
    RAW_QVEL_SLICE,
    RIGHT_CONTACT_INDEX,
    RIGHT_FORCE_INDEX,
    STATE_DIM,
    STATE_FIELD_NAMES,
    TCP_SLICE,
    deterministic_oracle_scores,
    extract_true_simulator_state,
    normalize_object_quaternion,
    query_pad_object_contacts,
    stack_state_oracle_inputs,
    validate_state_oracle_inputs,
)


class FakeMujoco:
    mjtObj = SimpleNamespace(mjOBJ_GEOM=object())

    @staticmethod
    def mj_name2id(model, object_type, name):
        assert object_type is FakeMujoco.mjtObj.mjOBJ_GEOM
        return model.geom_ids.get(name, -1)

    @staticmethod
    def mj_contactForce(model, data, index, output):
        del model
        output[:] = data.contact_forces[index]


def _contact(geom1: int, geom2: int) -> SimpleNamespace:
    return SimpleNamespace(geom1=geom1, geom2=geom2)


def _fake_env(*, task: str = "button-press-v3") -> tuple[SimpleNamespace, np.ndarray]:
    object_name = "btnGeom" if task == "button-press-v3" else "objGeom"
    nq, nv = (10, 10) if task == "button-press-v3" else (16, 15)
    model = SimpleNamespace(
        geom_ids={"leftpad_geom": 3, "rightpad_geom": 7, object_name: 11},
        nq=nq,
        nv=nv,
        nu=2,
        nmocap=1,
    )
    contacts = [
        _contact(3, 11),
        _contact(11, 7),
        _contact(3, 99),
    ]
    data = SimpleNamespace(
        ncon=len(contacts),
        contact=contacts,
        contact_forces=[
            np.asarray([3.0, 4.0, 0.0, 0.0, 0.0, 0.0]),
            np.asarray([0.0, 0.0, 12.0, 0.0, 0.0, 0.0]),
            np.asarray([50.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ],
        qpos=np.linspace(0.01, 0.16, nq, dtype=np.float64),
        qvel=np.linspace(-0.15, -0.01, nv, dtype=np.float64),
        ctrl=np.asarray([0.25, -0.25], dtype=np.float64),
        mocap_pos=np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64),
        mocap_quat=np.asarray([[1.0, 0.0, 1.0, 0.0]], dtype=np.float64),
    )
    if nq == 16:
        data.qpos[12:16] = np.asarray([1.0, 0.0, 0.0, 0.0])
    quaternion = (
        np.asarray([2.0, 0.0, 0.0, 0.0])
        if task == "button-press-v3"
        else np.asarray([0.0, 0.0, 0.0, 2.0])
    )
    env = SimpleNamespace(
        model=model,
        data=data,
        tcp_center=np.asarray([0.1, 0.2, 0.3]),
        obj_init_pos=np.asarray([-0.1, 0.5, 0.02]),
        _target_pos=np.asarray([0.0, 0.8, 0.12]),
        _get_pos_objects=lambda: np.asarray([0.0, 0.7, 0.12]),
        _get_quat_objects=lambda: quaternion,
    )
    observation = np.asarray([0.1, 0.2, 0.3, 0.75, 99.0], dtype=np.float64)
    return env, observation


def _state(
    *,
    task: str = "reach-v3",
    tcp=(0.0, 0.0, 0.0),
    obj=(0.0, 0.0, 0.0),
    obj_init=(0.0, 0.0, 0.0),
    left=False,
    right=False,
) -> np.ndarray:
    value = np.zeros(STATE_DIM, dtype=np.float32)
    value[TCP_SLICE] = tcp
    value[3] = 0.5
    value[OBJECT_POSITION_SLICE] = obj
    value[OBJECT_QUATERNION_SLICE] = np.asarray([0.0, 0.0, 0.0, 1.0])
    value[LEFT_CONTACT_INDEX] = float(left)
    value[RIGHT_CONTACT_INDEX] = float(right)
    value[LEFT_FORCE_INDEX] = 1.0 if left else 0.0
    value[RIGHT_FORCE_INDEX] = 1.0 if right else 0.0
    value[OBJECT_INITIAL_POSITION_SLICE] = obj_init
    if task != "button-press-v3":
        value[RAW_QPOS_SLICE.start + 12 : RAW_QPOS_SLICE.start + 16] = (
            np.asarray([1.0, 0.0, 0.0, 0.0])
        )
    value[RAW_MOCAP_QUATERNION_SLICE] = np.asarray([1.0, 0.0, 1.0, 0.0])
    return value


def _trajectory(final: np.ndarray, start: np.ndarray | None = None) -> np.ndarray:
    if start is None:
        start = final
    values = [
        start + (final - start) * np.float32(index / (ORACLE_HORIZON - 1))
        for index in range(ORACLE_HORIZON)
    ]
    result = np.stack(values).astype(np.float32)
    # Linear interpolation preserves the constant unit quaternion in our fixtures.
    return result


def test_contact_query_uses_named_geoms_and_accumulates_force_norms() -> None:
    env, _ = _fake_env()
    result = query_pad_object_contacts(
        env.model,
        env.data,
        "btnGeom",
        mujoco_module=FakeMujoco,
    )

    assert result.left is True
    assert result.right is True
    assert result.left_force == pytest.approx(5.0)
    assert result.right_force == pytest.approx(12.0)

    with pytest.raises(ValueError, match="required MuJoCo geom"):
        query_pad_object_contacts(
            env.model,
            env.data,
            "missing",
            mujoco_module=FakeMujoco,
        )


def test_extract_state_has_fixed_semantics_and_keeps_goal_separate() -> None:
    env, observation = _fake_env()
    extracted = extract_true_simulator_state(
        env,
        observation,
        "button-press-v3",
        mujoco_module=FakeMujoco,
    )

    assert extracted.state.shape == (STATE_DIM,)
    assert extracted.state.dtype == np.float32
    assert len(STATE_FIELD_NAMES) == STATE_DIM
    np.testing.assert_allclose(extracted.state[TCP_SLICE], [0.1, 0.2, 0.3])
    assert extracted.state[3] == pytest.approx(0.75)
    np.testing.assert_allclose(
        extracted.state[OBJECT_POSITION_SLICE], [0.0, 0.7, 0.12]
    )
    # Button exposes MuJoCo wxyz; the v3 state contract stores unit xyzw.
    np.testing.assert_array_equal(
        extracted.state[OBJECT_QUATERNION_SLICE], [0.0, 0.0, 0.0, 1.0]
    )
    np.testing.assert_array_equal(
        extracted.state[[LEFT_CONTACT_INDEX, RIGHT_CONTACT_INDEX]], [1.0, 1.0]
    )
    np.testing.assert_allclose(
        extracted.state[[LEFT_FORCE_INDEX, RIGHT_FORCE_INDEX]], [5.0, 12.0]
    )
    np.testing.assert_allclose(
        extracted.state[OBJECT_INITIAL_POSITION_SLICE], [-0.1, 0.5, 0.02]
    )
    np.testing.assert_allclose(
        extracted.state[RAW_QPOS_SLICE][:10], env.data.qpos
    )
    np.testing.assert_array_equal(extracted.state[RAW_QPOS_SLICE][10:], 0.0)
    np.testing.assert_allclose(
        extracted.state[RAW_QVEL_SLICE][:10], env.data.qvel
    )
    np.testing.assert_array_equal(extracted.state[RAW_QVEL_SLICE][10:], 0.0)
    np.testing.assert_allclose(extracted.state[RAW_CTRL_SLICE], env.data.ctrl)
    np.testing.assert_allclose(
        extracted.state[RAW_MOCAP_POSITION_SLICE], env.data.mocap_pos[0]
    )
    np.testing.assert_allclose(
        extracted.state[RAW_MOCAP_QUATERNION_SLICE], env.data.mocap_quat[0]
    )
    np.testing.assert_allclose(extracted.goal_world, [0.0, 0.8, 0.12])
    assert not any("goal" in name for name in STATE_FIELD_NAMES)


def test_quaternion_normalization_handles_task_conventions_and_sign() -> None:
    np.testing.assert_allclose(
        normalize_object_quaternion("button-press-v3", [-2.0, 0.0, 0.0, 0.0]),
        [0.0, 0.0, 0.0, 1.0],
    )
    np.testing.assert_allclose(
        normalize_object_quaternion("push-v3", [0.0, 0.0, 0.0, -3.0]),
        [0.0, 0.0, 0.0, 1.0],
    )
    with pytest.raises(ValueError, match="non-zero norm"):
        normalize_object_quaternion("reach-v3", np.zeros(4))


@pytest.mark.parametrize(
    "forbidden",
    [
        "reward",
        "future_return",
        "progress",
        "success",
        "rgb",
        "visual_latent",
        "q_true",
        "offline_labels",
    ],
)
def test_state_oracle_contract_rejects_labels_and_visual_inputs(forbidden: str) -> None:
    current = _state()
    future = np.stack([_trajectory(_state())])
    values = {
        "task": "reach-v3",
        "current_state": current,
        "future_state": future,
        "goal_world": np.zeros(3, dtype=np.float32),
        forbidden: np.zeros(1, dtype=np.float32),
    }
    with pytest.raises(ValueError, match="forbidden"):
        validate_state_oracle_inputs(values)


def test_state_oracle_contract_requires_exact_h5_float32_physical_states() -> None:
    values = {
        "task": "reach-v3",
        "current_state": _state(),
        "future_state": np.stack([_trajectory(_state())]),
        "goal_world": np.zeros(3, dtype=np.float32),
    }
    validate_state_oracle_inputs(values)

    wrong_horizon = dict(values)
    wrong_horizon["future_state"] = values["future_state"][:, :-1]
    with pytest.raises(ValueError, match="future_state must have shape"):
        validate_state_oracle_inputs(wrong_horizon)

    wrong_dtype = dict(values)
    wrong_dtype["future_state"] = values["future_state"].astype(np.float64)
    with pytest.raises(TypeError, match="dtype float32"):
        validate_state_oracle_inputs(wrong_dtype)


def test_task_specific_padding_and_raw_quaternions_are_strict() -> None:
    button = _state(task="button-press-v3")
    button_future = np.stack([_trajectory(button)])
    inputs = {
        "task": "button-press-v3",
        "current_state": button,
        "future_state": button_future,
        "goal_world": np.zeros(3, dtype=np.float32),
    }
    validate_state_oracle_inputs(inputs)

    bad_padding = button_future.copy()
    bad_padding[0, 0, RAW_QPOS_SLICE.start + 10] = 1.0
    with pytest.raises(ValueError, match="qpos padding"):
        validate_state_oracle_inputs({**inputs, "future_state": bad_padding})

    reach = _state(task="reach-v3")
    bad_free_quaternion = np.stack([_trajectory(reach)])
    free_quaternion = slice(RAW_QPOS_SLICE.start + 12, RAW_QPOS_SLICE.start + 16)
    bad_free_quaternion[..., free_quaternion] = 0.0
    with pytest.raises(ValueError, match="free-joint quaternion"):
        validate_state_oracle_inputs(
            {
                "task": "reach-v3",
                "current_state": reach,
                "future_state": bad_free_quaternion,
                "goal_world": np.zeros(3, dtype=np.float32),
            }
        )

    bad_mocap_quaternion = reach.copy()
    bad_mocap_quaternion[RAW_MOCAP_QUATERNION_SLICE] = 0.0
    with pytest.raises(ValueError, match="mocap quaternions"):
        validate_state_oracle_inputs(
            {
                "task": "reach-v3",
                "current_state": bad_mocap_quaternion,
                "future_state": np.stack([_trajectory(reach)]),
                "goal_world": np.zeros(3, dtype=np.float32),
            }
        )


@pytest.mark.parametrize(
    ("task", "goal", "closer_state", "farther_state"),
    [
        (
            "reach-v3",
            (0.0, 0.0, 0.0),
            _state(task="reach-v3", tcp=(0.01, 0.0, 0.0), obj=(1.0, 1.0, 1.0)),
            _state(task="reach-v3", tcp=(0.2, 0.0, 0.0), obj=(0.0, 0.0, 0.0)),
        ),
        (
            "button-press-v3",
            (0.0, 0.0, 0.0),
            _state(task="button-press-v3", tcp=(0.0, 0.02, 0.0), obj=(0.0, 0.01, 0.0)),
            _state(task="button-press-v3", tcp=(0.0, 0.02, 0.0), obj=(0.0, 0.1, 0.0)),
        ),
        (
            "push-v3",
            (0.0, 0.0, 0.0),
            _state(task="push-v3", obj=(0.01, 0.0, 0.0), obj_init=(0.3, 0.0, 0.0)),
            _state(task="push-v3", obj=(0.2, 0.0, 0.0), obj_init=(0.3, 0.0, 0.0)),
        ),
        (
            "pick-place-v3",
            (0.0, 0.0, 0.1),
            _state(
                task="pick-place-v3",
                obj=(0.01, 0.0, 0.1),
                obj_init=(0.3, 0.0, 0.0),
            ),
            _state(task="pick-place-v3", obj=(0.2, 0.0, 0.1), obj_init=(0.3, 0.0, 0.0)),
        ),
    ],
)
def test_task_distance_is_primary_for_all_four_tasks(
    task: str,
    goal: tuple[float, float, float],
    closer_state: np.ndarray,
    farther_state: np.ndarray,
) -> None:
    # Give the farther candidate contact and the closer candidate none: task
    # distance must still remain the primary ranking component.
    farther_state = farther_state.copy()
    farther_state[LEFT_CONTACT_INDEX] = 1.0
    farther_state[LEFT_FORCE_INDEX] = 1.0
    future = np.stack([_trajectory(farther_state), _trajectory(closer_state)])

    result = deterministic_oracle_scores(
        task,
        _state(task=task),
        future,
        np.asarray(goal, dtype=np.float32),
    )

    np.testing.assert_array_equal(result.order, [1, 0])
    assert result.task_distance[1] < result.task_distance[0]
    assert result.score[1] > result.score[0]


def test_contact_and_reach_are_deterministic_tie_breaks() -> None:
    goal = np.asarray([0.0, 0.0, 0.1], dtype=np.float32)
    no_contact = _state(
        task="pick-place-v3",
        tcp=(0.0, 0.0, 0.1),
        obj=(0.1, 0.0, 0.1),
        obj_init=(0.1, 0.0, 0.0),
    )
    contact = no_contact.copy()
    contact[LEFT_CONTACT_INDEX] = 1.0
    contact[LEFT_FORCE_INDEX] = 2.0
    future = np.stack([_trajectory(no_contact), _trajectory(contact)])

    result = deterministic_oracle_scores(
        "pick-place-v3", _state(task="pick-place-v3"), future, goal
    )

    np.testing.assert_allclose(result.task_distance, [0.1, 0.1])
    np.testing.assert_array_equal(result.contact_penalty, [1.0, 0.0])
    np.testing.assert_array_equal(result.order, [1, 0])


def test_identical_physical_trajectories_have_identical_scores_and_ranks() -> None:
    goal = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    close = _trajectory(_state(task="push-v3", obj=(0.02, 0.0, 0.0)))
    far = _trajectory(_state(task="push-v3", obj=(0.2, 0.0, 0.0)))
    future = np.stack([close, far, close.copy()])

    result = deterministic_oracle_scores(
        "push-v3", _state(task="push-v3"), future, goal
    )

    assert result.physical_cost[0] == result.physical_cost[2]
    assert result.score[0] == result.score[2]
    assert result.rank[0] == result.rank[2]
    assert result.score[0] > result.score[1]

    changed_current = _state(task="push-v3", tcp=(0.8, -0.8, 0.4))
    repeated = deterministic_oracle_scores(
        "push-v3", changed_current, future, goal
    )
    np.testing.assert_array_equal(result.score, repeated.score)


@pytest.mark.parametrize(
    "task",
    ["pick-place-v3", "button-press-v3", "reach-v3", "push-v3"],
)
def test_real_metaworld_state_smoke_has_full_markov_contract(task: str) -> None:
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("metaworld")

    env = gym.make("Meta-World/MT1", env_name=task, seed=7321)
    try:
        observation, _ = env.reset()
        unwrapped = env.unwrapped
        extracted = extract_true_simulator_state(unwrapped, observation, task)

        assert extracted.state.shape == (58,)
        assert np.isfinite(extracted.state).all()
        np.testing.assert_allclose(
            extracted.state[RAW_QPOS_SLICE][: unwrapped.model.nq],
            np.asarray(unwrapped.data.qpos, dtype=np.float32),
        )
        np.testing.assert_allclose(
            extracted.state[RAW_QVEL_SLICE][: unwrapped.model.nv],
            np.asarray(unwrapped.data.qvel, dtype=np.float32),
        )
        if task == "button-press-v3":
            np.testing.assert_array_equal(
                extracted.state[RAW_QPOS_SLICE][10:], 0.0
            )
            np.testing.assert_array_equal(
                extracted.state[RAW_QVEL_SLICE][10:], 0.0
            )
        else:
            assert np.linalg.norm(
                extracted.state[
                    RAW_QPOS_SLICE.start + 12 : RAW_QPOS_SLICE.start + 16
                ]
            ) == pytest.approx(1.0, abs=1e-5)

        future = np.stack(
            [np.repeat(extracted.state[None], ORACLE_HORIZON, axis=0)]
        )
        validate_state_oracle_inputs(
            {
                "task": task,
                "current_state": extracted.state,
                "future_state": future,
                "goal_world": extracted.goal_world,
            }
        )
    finally:
        env.close()


def test_stack_keeps_state_and_goal_in_distinct_arrays() -> None:
    first_env, observation = _fake_env(task="push-v3")
    first = extract_true_simulator_state(
        first_env,
        observation,
        "push-v3",
        mujoco_module=FakeMujoco,
    )
    second_env, observation = _fake_env(task="push-v3")
    second_env._target_pos = np.asarray([0.3, 0.4, 0.5])
    second = extract_true_simulator_state(
        second_env,
        observation,
        "push-v3",
        mujoco_module=FakeMujoco,
    )

    states, goals = stack_state_oracle_inputs([first, second])

    assert states.shape == (2, STATE_DIM)
    assert goals.shape == (2, 3)
    np.testing.assert_array_equal(states[0], states[1])
    assert not np.array_equal(goals[0], goals[1])
