from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from fcpg.lewm_counterfactual_v3_candidates import (
    CANDIDATE_COUNT,
    MAX_SLOT_PRIOR_ABS_SPEARMAN,
    MAX_SLOT_PRIOR_NORMALIZED_REGRET_IMPROVEMENT,
    TARGET_MOTION_RMS,
    AnonymousCandidateBank,
    build_equal_norm_candidate_bank,
    motion_delta_rms,
    public_candidate_fields,
    transition_axes,
    validate_equal_norm_candidate_bank,
    validate_slot_prior_audit,
)


def _center() -> np.ndarray:
    rng = np.random.default_rng(17)
    result = rng.uniform(-0.72, 0.72, size=(25, 4)).astype(np.float32)
    result[:, 3] = np.where(np.arange(25) < 12, -1.0, 1.0)
    return result


def test_transition_axes_are_deterministic_and_orthonormal() -> None:
    left = transition_axes(73)
    right = transition_axes(73)
    different = transition_axes(74)

    np.testing.assert_array_equal(left, right)
    assert not np.array_equal(left, different)
    flattened = left.reshape(4, 15)
    np.testing.assert_allclose(flattened @ flattened.T, np.eye(4), atol=1.0e-12)
    assert not left.flags.writeable


def test_bank_is_deterministic_equal_norm_and_executor_ready() -> None:
    center = _center()
    left, left_audit = build_equal_norm_candidate_bank(
        center,
        direction_seed=103,
        permutation_seed=211,
    )
    right, right_audit = build_equal_norm_candidate_bank(
        center,
        direction_seed=103,
        permutation_seed=211,
    )

    assert left.names == tuple(f"candidate_{index}" for index in range(8))
    np.testing.assert_array_equal(left.actions, right.actions)
    assert left_audit == right_audit
    np.testing.assert_array_equal(left.actions, np.clip(left.actions, -1.0, 1.0))
    np.testing.assert_array_equal(
        left.actions[:, :, 3],
        np.broadcast_to(center[None, :, 3], (CANDIDATE_COUNT, 25)),
    )
    np.testing.assert_allclose(
        motion_delta_rms(left.actions, center),
        TARGET_MOTION_RMS,
        rtol=0.0,
        atol=2.0e-6,
    )
    validation = validate_equal_norm_candidate_bank(left, center)
    assert validation.executor_clip_identity
    assert validation.minimum_pairwise_motion_rms > 0.0
    assert not left.actions.flags.writeable


def test_boundary_center_is_solved_after_per_direction_clipping() -> None:
    center = np.zeros((25, 4), dtype=np.float32)
    center[:, 0] = 1.0
    center[:, 1] = -1.0
    center[::2, 2] = 0.98
    center[1::2, 2] = -0.98
    center[:, 3] = 1.0

    bank, _ = build_equal_norm_candidate_bank(
        center,
        direction_seed=19,
        permutation_seed=23,
    )

    np.testing.assert_array_equal(bank.actions, np.clip(bank.actions, -1.0, 1.0))
    np.testing.assert_allclose(
        motion_delta_rms(bank.actions, center),
        TARGET_MOTION_RMS,
        rtol=0.0,
        atol=2.0e-6,
    )


def test_slot_permutation_changes_rows_but_not_candidate_set() -> None:
    center = _center()
    left, left_audit = build_equal_norm_candidate_bank(
        center,
        direction_seed=31,
        permutation_seed=41,
    )
    right, right_audit = build_equal_norm_candidate_bank(
        center,
        direction_seed=31,
        permutation_seed=43,
    )

    assert (
        left_audit.slot_to_canonical_direction
        != right_audit.slot_to_canonical_direction
    )
    assert not np.array_equal(left.actions, right.actions)
    left_rows = {row.tobytes() for row in left.actions}
    right_rows = {row.tobytes() for row in right.actions}
    assert left_rows == right_rows


def test_public_bank_does_not_expose_semantic_or_audit_ids() -> None:
    bank, audit = build_equal_norm_candidate_bank(
        _center(),
        direction_seed=5,
        permutation_seed=7,
    )

    assert public_candidate_fields() == ("names", "actions")
    assert tuple(field.name for field in fields(bank)) == ("names", "actions")
    assert all(name.startswith("candidate_") for name in bank.names)
    assert not any(
        "axis" in name or "pos" in name or "neg" in name for name in bank.names
    )
    assert not hasattr(bank, "slot_to_canonical_direction")
    assert sorted(audit.slot_to_canonical_direction) == list(range(8))
    assert len(set(audit.canonical_direction_sha256)) == 8


def test_poison_semantic_names_are_rejected_by_public_bank() -> None:
    with pytest.raises(ValueError, match="anonymous"):
        AnonymousCandidateBank(
            names=("axis_0_positive",) + tuple(
                f"candidate_{index}" for index in range(1, 8)
            ),
            actions=np.zeros((8, 25, 4), dtype=np.float32),
        )


def test_unreachable_fixed_radius_fails_closed() -> None:
    center = np.zeros((25, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="unreachable"):
        build_equal_norm_candidate_bank(
            center,
            direction_seed=3,
            permutation_seed=4,
            action_low=np.array([-0.05, -0.05, -0.05, -1.0]),
            action_high=np.array([0.05, 0.05, 0.05, 1.0]),
        )


def test_out_of_bounds_center_and_tampered_candidate_fail_closed() -> None:
    center = _center()
    poisoned_center = center.copy()
    poisoned_center[0, 0] = 1.01
    with pytest.raises(ValueError, match="executor clipping"):
        build_equal_norm_candidate_bank(
            poisoned_center,
            direction_seed=3,
            permutation_seed=4,
        )

    bank, _ = build_equal_norm_candidate_bank(
        center,
        direction_seed=3,
        permutation_seed=4,
    )
    tampered = bank.actions.copy()
    tampered[0, 0, 0] = 1.1
    poisoned_bank = AnonymousCandidateBank(names=bank.names, actions=tampered)
    with pytest.raises(ValueError, match="executor clipping"):
        validate_equal_norm_candidate_bank(poisoned_bank, center)


def test_slot_prior_gate_is_fixed_and_fail_closed() -> None:
    validate_slot_prior_audit(
        task_macro_spearman=MAX_SLOT_PRIOR_ABS_SPEARMAN,
        normalized_regret_improvement=MAX_SLOT_PRIOR_NORMALIZED_REGRET_IMPROVEMENT,
        confidence_interval=(-0.03, 0.12),
    )
    with pytest.raises(ValueError, match="Spearman"):
        validate_slot_prior_audit(
            task_macro_spearman=MAX_SLOT_PRIOR_ABS_SPEARMAN + 1.0e-3,
            normalized_regret_improvement=0.0,
            confidence_interval=(-0.02, 0.15),
        )
    with pytest.raises(ValueError, match="excludes zero"):
        validate_slot_prior_audit(
            task_macro_spearman=0.04,
            normalized_regret_improvement=0.0,
            confidence_interval=(0.01, 0.09),
        )
    with pytest.raises(ValueError, match="normalized-regret"):
        validate_slot_prior_audit(
            task_macro_spearman=0.04,
            normalized_regret_improvement=(
                MAX_SLOT_PRIOR_NORMALIZED_REGRET_IMPROVEMENT + 1.0e-3
            ),
            confidence_interval=(-0.01, 0.09),
        )
