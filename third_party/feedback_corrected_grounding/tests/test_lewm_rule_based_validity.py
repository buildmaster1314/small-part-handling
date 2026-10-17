from __future__ import annotations

import numpy as np
import pytest

from fcpg.lewm_rule_based_validity import (
    RuleBasedValidityConfig,
    RuleBasedVisualValidityFSM,
)


def _checkerboard(size: int = 64) -> np.ndarray:
    yy, xx = np.indices((size, size))
    gray = ((xx + yy) % 2).astype(np.float32)
    return np.repeat(gray[..., None], 3, axis=-1)


def _inputs(image: np.ndarray | None) -> dict[str, object]:
    return {
        "image": image,
        "frame_present": image is not None,
        "frame_fresh": True,
        "q_nominal": np.asarray([[24.0, 24.0], [32.0, 32.0], [40.0, 40.0]], dtype=np.float32),
        "q_corrected": np.asarray([[25.0, 24.0], [32.0, 31.0], [39.0, 40.0]], dtype=np.float32),
        "valid_nominal": np.ones(3, dtype=bool),
        "executed_action_rms": 0.0,
    }


def test_healthy_frames_use_measurement_and_initialize_adaptive_baseline() -> None:
    selector = RuleBasedVisualValidityFSM()
    first = selector.update(**_inputs(_checkerboard()))
    second = selector.update(**_inputs(_checkerboard()))
    assert first.mode == "measurement"
    assert second.mode == "measurement"
    assert second.reasons == ()


@pytest.mark.parametrize(
    ("field", "reason"),
    (("frame_present", "frame_missing"), ("frame_fresh", "frame_stale")),
)
def test_hard_camera_failures_switch_to_prior_immediately(
    field: str, reason: str
) -> None:
    selector = RuleBasedVisualValidityFSM()
    selector.update(**_inputs(_checkerboard()))
    values = _inputs(_checkerboard())
    values[field] = False
    if field == "frame_present":
        values["image"] = None
    decision = selector.update(**values)
    assert decision.use_prior
    assert decision.hard_failure
    assert reason in decision.reasons


def test_local_blur_requires_two_bad_frames_and_one_good_frame_recovers() -> None:
    selector = RuleBasedVisualValidityFSM()
    sharp = _checkerboard()
    selector.update(**_inputs(sharp))
    blurred = np.full_like(sharp, 0.5)
    first_bad = selector.update(**_inputs(blurred))
    second_bad = selector.update(**_inputs(blurred))
    recovered = selector.update(**_inputs(sharp))
    assert not first_bad.use_prior
    assert second_bad.use_prior
    assert "global_blur" in second_bad.reasons
    assert "global_contrast_loss" in second_bad.reasons
    assert recovered.mode == "measurement"


def test_saturated_fcpg_residual_only_corrobates_visual_degradation() -> None:
    selector = RuleBasedVisualValidityFSM()
    sharp = _checkerboard()
    selector.update(**_inputs(sharp))
    values = _inputs(sharp)
    values["q_corrected"] = np.asarray(values["q_nominal"]) + np.asarray(
        [[20.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32
    )
    saturated_only = selector.update(**values)
    assert not saturated_only.soft_failure
    values["image"] = np.full_like(sharp, 0.5)
    first = selector.update(**values)
    second = selector.update(**values)
    assert not first.use_prior
    assert second.use_prior
    assert "fcpg_residual_saturated" in second.reasons


def test_blackout_at_all_nominal_anchors_switches_immediately() -> None:
    selector = RuleBasedVisualValidityFSM()
    image = _checkerboard()
    selector.update(**_inputs(image))
    blacked_out = image.copy()
    for x, y in np.asarray(_inputs(image)["q_nominal"], dtype=np.int64):
        blacked_out[y - 12 : y + 13, x - 12 : x + 13] = 0.0
    values = _inputs(blacked_out)
    decision = selector.update(**values)
    assert decision.use_prior
    assert decision.hard_failure
    assert "local_blackout" in decision.reasons
    assert decision.maximum_nominal_patch_mean == 0.0


def test_unchanged_frame_is_only_frozen_when_nominal_projection_moves() -> None:
    selector = RuleBasedVisualValidityFSM()
    image = _checkerboard()
    selector.update(**_inputs(image))
    stationary = selector.update(**_inputs(image))
    moving_values = _inputs(image)
    moving_values["executed_action_rms"] = 0.2
    action_only = selector.update(**moving_values)
    moving_values["q_nominal"] = np.asarray(moving_values["q_nominal"]) + np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32
    )
    moving_values["q_corrected"] = np.asarray(moving_values["q_corrected"]) + np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32
    )
    moving = selector.update(**moving_values)
    assert stationary.mode == "measurement"
    assert action_only.mode == "measurement"
    assert moving.use_prior
    assert "frame_frozen_while_moving" in moving.reasons
    assert moving.maximum_nominal_motion_px == pytest.approx(1.0)


def test_corrected_anchor_out_of_bounds_switches_to_prior_immediately() -> None:
    selector = RuleBasedVisualValidityFSM()
    selector.update(**_inputs(_checkerboard()))
    values = _inputs(_checkerboard())
    values["q_corrected"] = np.asarray(values["q_corrected"]).copy()
    values["q_corrected"][0, 0] = 80.0
    decision = selector.update(**values)
    assert decision.use_prior
    assert decision.hard_failure
    assert "corrected_anchor_out_of_bounds" in decision.reasons


def test_nominal_invalidity_is_a_hard_failure() -> None:
    selector = RuleBasedVisualValidityFSM()
    selector.update(**_inputs(_checkerboard()))
    values = _inputs(_checkerboard())
    values["valid_nominal"] = np.asarray([True, False, True])
    decision = selector.update(**values)
    assert decision.use_prior
    assert decision.hard_failure
    assert "nominal_anchor_invalid" in decision.reasons


def test_invalid_anchor_shapes_fail_closed() -> None:
    selector = RuleBasedVisualValidityFSM()
    values = _inputs(_checkerboard())
    values["valid_nominal"] = np.ones((3, 1), dtype=bool)
    with pytest.raises(ValueError, match="valid_nominal"):
        selector.update(**values)


def test_configuration_rejects_nonpositive_hysteresis() -> None:
    with pytest.raises(ValueError, match="soft_failures_to_prior"):
        RuleBasedValidityConfig(soft_failures_to_prior=0)
