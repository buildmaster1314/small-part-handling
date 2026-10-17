import numpy as np
import pytest

from fcpg.so101_receding_control import (
    FreshObservationRequired,
    SO101ObservationIdentity,
    SO101RecedingHorizonGate,
)


def _observation(sequence: int) -> SO101ObservationIdentity:
    return SO101ObservationIdentity(
        session_id="live-session",
        wrist_sequence=sequence,
        column_sequence=sequence,
        grounding_sequence=sequence,
        wrist_timestamp_s=sequence / 60.0,
        column_timestamp_s=sequence / 30.0,
    )


def _prefix(offset: float = 0.0) -> np.ndarray:
    return np.arange(40 * 6, dtype=np.float32).reshape(40, 6) + offset


def test_gate_authorizes_exactly_one_five_command_chunk() -> None:
    gate = SO101RecedingHorizonGate(commands_per_round=5)
    observation = _observation(10)
    prefix = _prefix()
    plan = gate.build_plan(
        observation=observation,
        target_kind="white_adapter",
        target_xyz=np.asarray((0.1, 0.2, 0.03)),
        candidate_name="direct_low",
        diagnostic_prefix=prefix,
    )

    chunk = gate.authorize(plan, current_observation=observation)

    assert chunk.shape == (5, 6)
    np.testing.assert_array_equal(chunk, prefix[:5])
    chunk[0, 0] = -999.0
    assert plan.diagnostic_prefix[0, 0] != -999.0
    assert plan.diagnostic_horizon_commands == 40


def test_gate_rejects_reusing_an_observation() -> None:
    gate = SO101RecedingHorizonGate()
    observation = _observation(10)
    plan = gate.build_plan(
        observation=observation,
        target_kind="fish",
        target_xyz=np.zeros(3),
        candidate_name="direct_low",
        diagnostic_prefix=_prefix(),
    )
    gate.authorize(plan, current_observation=observation)

    with pytest.raises(FreshObservationRequired, match="must advance"):
        gate.build_plan(
            observation=observation,
            target_kind="fish",
            target_xyz=np.zeros(3),
            candidate_name="direct_low",
            diagnostic_prefix=_prefix(),
        )
    with pytest.raises(FreshObservationRequired, match="must advance"):
        gate.authorize(plan, current_observation=observation)


def test_gate_requires_both_cameras_and_grounding_to_advance() -> None:
    gate = SO101RecedingHorizonGate()
    first = _observation(10)
    plan = gate.build_plan(
        observation=first,
        target_kind="fish",
        target_xyz=np.zeros(3),
        candidate_name="direct_low",
        diagnostic_prefix=_prefix(),
    )
    gate.authorize(plan, current_observation=first)
    stale_column = SO101ObservationIdentity(
        session_id="live-session",
        wrist_sequence=11,
        column_sequence=10,
        grounding_sequence=11,
        wrist_timestamp_s=11 / 60.0,
        column_timestamp_s=10 / 30.0,
    )

    with pytest.raises(FreshObservationRequired, match="both camera frames"):
        gate.build_plan(
            observation=stale_column,
            target_kind="fish",
            target_xyz=np.zeros(3),
            candidate_name="direct_low",
            diagnostic_prefix=_prefix(),
        )


def test_fresh_round_can_reassociate_a_moved_target() -> None:
    gate = SO101RecedingHorizonGate()
    first = _observation(10)
    first_plan = gate.build_plan(
        observation=first,
        target_kind="fish",
        target_xyz=np.asarray((0.10, 0.20, 0.03)),
        candidate_name="direct_low",
        diagnostic_prefix=_prefix(),
    )
    gate.authorize(first_plan, current_observation=first)

    second = _observation(11)
    moved_target = np.asarray((0.13, 0.17, 0.03))
    second_plan = gate.build_plan(
        observation=second,
        target_kind="fish",
        target_xyz=moved_target,
        candidate_name="arc_left",
        diagnostic_prefix=_prefix(1.0),
    )
    second_chunk = gate.authorize(second_plan, current_observation=second)

    np.testing.assert_allclose(second_plan.target_xyz, moved_target)
    np.testing.assert_array_equal(second_chunk, _prefix(1.0)[:5])


def test_gate_rejects_a_plan_from_a_different_current_observation() -> None:
    gate = SO101RecedingHorizonGate()
    plan = gate.build_plan(
        observation=_observation(10),
        target_kind="fish",
        target_xyz=np.zeros(3),
        candidate_name="direct_low",
        diagnostic_prefix=_prefix(),
    )

    with pytest.raises(FreshObservationRequired, match="plan is stale"):
        gate.authorize(plan, current_observation=_observation(11))


@pytest.mark.parametrize(
    "prefix",
    (
        np.zeros((4, 6), dtype=np.float32),
        np.zeros((40, 5), dtype=np.float32),
        np.full((40, 6), np.nan, dtype=np.float32),
    ),
)
def test_plan_rejects_invalid_diagnostic_prefix(prefix: np.ndarray) -> None:
    gate = SO101RecedingHorizonGate()
    with pytest.raises(ValueError, match="diagnostic_prefix"):
        gate.build_plan(
            observation=_observation(10),
            target_kind="fish",
            target_xyz=np.zeros(3),
            candidate_name="direct_low",
            diagnostic_prefix=prefix,
        )
