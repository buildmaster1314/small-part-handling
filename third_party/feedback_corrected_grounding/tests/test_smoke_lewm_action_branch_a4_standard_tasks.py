from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.smoke_lewm_action_branch_a4_standard_tasks import (
    AVAILABILITY_ONLY_MODE,
    DIAGNOSTIC_SUBSET,
    FULL_TRAJECTORY_MODE,
    STANDARD_FIVE_TASK_SUBPANEL,
    STANDARD_METAWORLD_TASKS,
    build_report,
    collect_task_rows,
    parse_tasks,
    reset_distance_fields,
    run_full_source_trajectory,
    summarize_rows,
)


class MockPolicy:
    def __init__(self, actions: Sequence[Sequence[float]]) -> None:
        self.actions = [np.asarray(action, dtype=np.float32) for action in actions]
        self.calls = 0

    def get_action(self, _observation: np.ndarray) -> np.ndarray:
        action = self.actions[min(self.calls, len(self.actions) - 1)]
        self.calls += 1
        return action.copy()


class MockEnv:
    def __init__(
        self,
        step_successes: Sequence[bool],
        *,
        reset_success: bool = False,
        object_xyz: Sequence[float] = (0.0, 3.0, 4.0),
        goal_xyz: Sequence[float] = (0.0, 0.0, 2.0),
    ) -> None:
        self.step_successes = list(step_successes)
        self.reset_success = reset_success
        self.object_xyz = np.asarray(object_xyz, dtype=np.float64)
        self._target_pos = np.asarray(goal_xyz, dtype=np.float64)
        self.tcp_center = np.zeros(3, dtype=np.float64)
        self.action_space = SimpleNamespace(
            low=np.full(4, -1.0, dtype=np.float32),
            high=np.full(4, 1.0, dtype=np.float32),
        )
        self.step_index = 0
        self.closed = False

    def reset(self, seed: int | None = None):
        self.seed = seed
        self.step_index = 0
        return np.zeros(39, dtype=np.float64), {"success": float(self.reset_success)}

    def step(self, _action: np.ndarray):
        success = self.step_successes[self.step_index]
        self.step_index += 1
        return (
            np.full(39, self.step_index, dtype=np.float64),
            0.0,
            False,
            False,
            {"success": float(success)},
        )

    def _get_pos_objects(self) -> np.ndarray:
        return self.object_xyz.copy()

    def close(self) -> None:
        self.closed = True


def _policy(action: Sequence[float] = (1.0, 0.0, 0.0, 0.0)) -> MockPolicy:
    return MockPolicy([action])


def _successful_row(task: str) -> dict[str, object]:
    return run_full_source_trajectory(
        task,
        MockEnv([True, True, True]),
        _policy(),
        max_steps=20,
        env_seed=7,
    )


def test_mock_trajectory_can_succeed_after_more_than_eight_steps() -> None:
    successes = [False] * 9 + [True, True, True]
    actions = [
        [1.0, 0.0, 0.0, 0.0] if index % 2 else [0.0, 0.0, 0.0, 0.0]
        for index in range(len(successes))
    ]
    env = MockEnv(successes)
    policy = MockPolicy(actions)

    row = run_full_source_trajectory(
        "pick-place-v3", env, policy, max_steps=20, env_seed=17
    )

    assert env.seed == 17
    assert row["reset_success"] is False
    assert row["steps_executed"] == 12
    assert row["first_success_step"] == 10
    assert row["final_success"] is True
    assert row["success_streak_reached"] is True
    assert row["max_success_streak"] == 3
    assert row["nonzero_action_fraction"] == pytest.approx(0.5)
    assert row["stop_reason"] == "success_streak"
    assert policy.calls == 12


def test_reset_success_is_not_a_streak_frame_and_forces_task_failure() -> None:
    env = MockEnv([True, True, True], reset_success=True)
    row = run_full_source_trajectory("push-v3", env, _policy(), max_steps=10)

    assert row["reset_success"] is True
    assert row["first_success_step"] == 1
    assert row["steps_executed"] == 3
    assert row["success_streak_reached"] is True

    report = build_report(
        [row],
        tasks=("push-v3",),
        availability_only=False,
        max_steps=10,
        nonzero_action_epsilon=1e-8,
        diagnostic_subset=True,
    )
    assert report["summary"]["per_task"]["push-v3"]["passed"] is False
    assert report["diagnostic_subset_pass"] is False
    assert report["run_pass"] is False


@pytest.mark.parametrize(
    ("task", "reference", "distance"),
    [
        ("pick-place-v3", "object", 5.0),
        ("reach-v3", "goal", 2.0),
        ("push-v3", "object", 5.0),
        ("door-open-v3", "object", 5.0),
        ("window-open-v3", "object", 5.0),
    ],
)
def test_each_standard_task_uses_its_required_reset_reference(
    task: str, reference: str, distance: float
) -> None:
    fields = reset_distance_fields(MockEnv([]), task)

    assert fields["reset_distance_reference"] == reference
    assert fields["reset_tcp_reference_distance_m"] == pytest.approx(distance)
    assert fields[f"reset_tcp_{reference}_distance_m"] == pytest.approx(distance)
    other_reference = "goal" if reference == "object" else "object"
    assert fields[f"reset_tcp_{other_reference}_distance_m"] is None


def test_missing_reset_geometry_remains_explicitly_optional() -> None:
    env = MockEnv([])
    env._target_pos = None

    fields = reset_distance_fields(env, "reach-v3")

    assert fields == {
        "reset_distance_reference": None,
        "reset_tcp_reference_distance_m": None,
        "reset_tcp_object_distance_m": None,
        "reset_tcp_goal_distance_m": None,
    }


def test_standard_report_requires_and_passes_exact_five_task_subpanel() -> None:
    rows = [_successful_row(task) for task in STANDARD_METAWORLD_TASKS]

    summary = summarize_rows(rows)
    report = build_report(
        rows,
        tasks=STANDARD_METAWORLD_TASKS,
        availability_only=False,
        max_steps=20,
        nonzero_action_epsilon=1e-8,
    )

    assert summary["task_count"] == 5
    assert summary["is_complete_standard_task_subpanel"] is True
    assert summary["all_tasks_available"] is True
    assert summary["valid_reset_task_count"] == 5
    assert summary["reset_distance_available_count"] == 5
    assert summary["success_streak_reached_count"] == 5
    assert summary["passed_task_count"] == 5
    assert summary["standard_subpanel_pass"] is True
    assert summary["mean_nonzero_action_fraction"] == pytest.approx(1.0)
    assert report["mode"] == FULL_TRAJECTORY_MODE
    assert report["panel_kind"] == STANDARD_FIVE_TASK_SUBPANEL
    assert report["is_complete_standard_task_subpanel_smoke"] is True
    assert report["reset_frame_counts_toward_success_streak"] is False
    assert report["required_reset_success"] is False
    assert report["standard_subpanel_pass"] is True
    assert report["run_pass"] is True


def test_subset_requires_diagnostic_label_and_never_claims_subpanel_pass() -> None:
    tasks = ("reach-v3", "push-v3")
    rows = [_successful_row(task) for task in tasks]

    with pytest.raises(ValueError, match="exact ordered five-task"):
        build_report(
            rows,
            tasks=tasks,
            availability_only=False,
            max_steps=20,
            nonzero_action_epsilon=1e-8,
        )

    report = build_report(
        rows,
        tasks=tasks,
        availability_only=False,
        max_steps=20,
        nonzero_action_epsilon=1e-8,
        diagnostic_subset=True,
    )
    assert report["panel_kind"] == DIAGNOSTIC_SUBSET
    assert report["is_complete_standard_task_subpanel_smoke"] is False
    assert report["summary"]["is_complete_standard_task_subpanel"] is False
    assert report["standard_subpanel_pass"] is False
    assert report["diagnostic_subset_pass"] is True
    assert report["run_pass"] is True


def test_reordered_or_diagnostic_standard_subpanel_is_rejected() -> None:
    reordered_tasks = (*STANDARD_METAWORLD_TASKS[1:], STANDARD_METAWORLD_TASKS[0])
    reordered_rows = [_successful_row(task) for task in reordered_tasks]

    with pytest.raises(ValueError, match="exact ordered five-task"):
        build_report(
            reordered_rows,
            tasks=reordered_tasks,
            availability_only=False,
            max_steps=20,
            nonzero_action_epsilon=1e-8,
        )
    with pytest.raises(ValueError, match="proper subset"):
        build_report(
            [_successful_row(task) for task in STANDARD_METAWORLD_TASKS],
            tasks=STANDARD_METAWORLD_TASKS,
            availability_only=False,
            max_steps=20,
            nonzero_action_epsilon=1e-8,
            diagnostic_subset=True,
        )


@pytest.mark.parametrize(
    "missing_field",
    [
        "mode",
        "available",
        "reset_success",
        "reset_tcp_reference_distance_m",
        "success_streak_reached",
        "close_error_type",
        "max_steps",
    ],
)
def test_aggregation_fails_closed_when_required_field_is_missing(
    missing_field: str,
) -> None:
    row = _successful_row("reach-v3")
    del row[missing_field]

    with pytest.raises(ValueError, match="missing required"):
        summarize_rows([row], expected_tasks=("reach-v3",))


def test_availability_probe_is_explicit_and_cannot_pass_full_smoke() -> None:
    env = MockEnv([])

    class NoActionPolicy:
        def get_action(self, _observation: np.ndarray) -> np.ndarray:
            raise AssertionError("availability probe must not execute the policy")

    rows = collect_task_rows(
        ("reach-v3",),
        camera="corner",
        width=32,
        height=32,
        task_index=0,
        benchmark_seed=3,
        env_seed=4,
        max_steps=5,
        nonzero_action_epsilon=1e-8,
        availability_only=True,
        env_factory=lambda *_args, **_kwargs: env,
        policy_factory=lambda _task: NoActionPolicy(),
    )
    report = build_report(
        rows,
        tasks=("reach-v3",),
        availability_only=True,
        max_steps=5,
        nonzero_action_epsilon=1e-8,
        diagnostic_subset=True,
    )

    assert rows[0]["mode"] == AVAILABILITY_ONLY_MODE
    assert rows[0]["nonzero_action_fraction"] is None
    assert env.closed is True
    assert report["mode"] == AVAILABILITY_ONLY_MODE
    assert report["standard_subpanel_pass"] is False
    assert report["diagnostic_subset_pass"] is None
    assert report["availability_pass"] is True
    assert report["run_pass"] is True


def test_availability_probe_fails_closed_when_reset_success_is_unknown() -> None:
    class UnknownResetSuccessEnv(MockEnv):
        def reset(self, seed: int | None = None):
            self.seed = seed
            self.step_index = 0
            return np.zeros(39, dtype=np.float64), {}

    rows = collect_task_rows(
        ("reach-v3",),
        camera="corner",
        width=32,
        height=32,
        task_index=0,
        benchmark_seed=3,
        env_seed=4,
        max_steps=5,
        nonzero_action_epsilon=1e-8,
        availability_only=True,
        env_factory=lambda *_args, **_kwargs: UnknownResetSuccessEnv([]),
        policy_factory=lambda _task: _policy(),
    )
    report = build_report(
        rows,
        tasks=("reach-v3",),
        availability_only=True,
        max_steps=5,
        nonzero_action_epsilon=1e-8,
        diagnostic_subset=True,
    )

    assert rows[0]["available"] is True
    assert rows[0]["reset_success"] is None
    assert report["availability_pass"] is False
    assert report["run_pass"] is False


def test_report_rejects_rows_from_an_implicit_different_mode() -> None:
    with pytest.raises(ValueError, match="does not match expected mode"):
        build_report(
            [_successful_row("reach-v3")],
            tasks=("reach-v3",),
            availability_only=True,
            max_steps=20,
            nonzero_action_epsilon=1e-8,
            diagnostic_subset=True,
        )


def test_close_failure_preserves_primary_error_and_collection_continues() -> None:
    class ResetAndCloseFailureEnv(MockEnv):
        def reset(self, seed: int | None = None):
            self.seed = seed
            raise RuntimeError("primary reset failure")

        def close(self) -> None:
            self.closed = True
            raise OSError("cleanup failure")

    first_env = ResetAndCloseFailureEnv([])
    second_env = MockEnv([True, True, True])
    environments = {
        "pick-place-v3": first_env,
        "reach-v3": second_env,
    }
    tasks = tuple(environments)

    rows = collect_task_rows(
        tasks,
        camera="corner",
        width=32,
        height=32,
        task_index=0,
        benchmark_seed=3,
        env_seed=4,
        max_steps=10,
        nonzero_action_epsilon=1e-8,
        availability_only=False,
        env_factory=lambda task, *_args, **_kwargs: environments[task],
        policy_factory=lambda _task: _policy(),
    )
    summary = summarize_rows(rows, expected_tasks=tasks)

    assert len(rows) == 2
    assert rows[0]["error_type"] == "RuntimeError"
    assert rows[0]["error"] == "primary reset failure"
    assert rows[0]["close_error_type"] == "OSError"
    assert rows[0]["close_error"] == "cleanup failure"
    assert first_env.closed is True
    assert rows[1]["available"] is True
    assert rows[1]["success_streak_reached"] is True
    assert second_env.closed is True
    assert summary["primary_error_count"] == 1
    assert summary["close_error_count"] == 1
    assert summary["per_task"]["pick-place-v3"]["passed"] is False
    assert summary["per_task"]["reach-v3"]["passed"] is True
    assert summary["all_tasks_passed"] is False


def test_task_parser_keeps_the_standard_metaworld_subpanel_closed() -> None:
    assert parse_tasks(",".join(STANDARD_METAWORLD_TASKS)) == STANDARD_METAWORLD_TASKS
    with pytest.raises(Exception, match="duplicates"):
        parse_tasks("reach-v3,reach-v3")
    with pytest.raises(Exception, match="unknown"):
        parse_tasks("drawer-open-v3")
