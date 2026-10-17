"""Analytic latent observer primitives for a frozen P1-native world model.

The implementation follows the feedback-state equations from *Feedback
World Model Enables Precise Guidance of Diffusion Policy*.  It deliberately
separates two operations that are easy to conflate:

1. a temporal feedback residual formed from a causal one-step prior and the
   subsequently observed latent state; and
2. the existing geometry-complementary update, which treats the deployable
   exact-FK geometry reconstruction as a partial measurement during a known
   visual dropout.

No function in this module accepts a future clean target.  Offline targets may
be used by a caller after an observer rollout has completed, but they cannot be
used to initialize or update the observer state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Final

import numpy as np
import torch


Tensor = torch.Tensor

P1_HISTORY_SIZE: Final = 3
P1_ANCHOR_COUNT: Final = 3
P1_TOKEN_DIM: Final = 256
P1_ACTION_DIM: Final = 20

CAUSAL_PRIOR_FIELD: Final = "pre_drop_raw_prior_token"
CAUSAL_PRIOR_FLAG: Final = "pre_drop_prior_is_causal"
CARRIER_HISTORY_FIELD: Final = "pre_drop_history_token"
CARRIER_ACTION_FIELD: Final = "pre_drop_action_context"
CARRIER_VISIBLE_FIELD: Final = "pre_drop_visible_token"
CARRIER_HISTORY_HORIZONS_FIELD: Final = "pre_drop_history_horizons"
CARRIER_TARGET_HORIZON_FIELD: Final = "pre_drop_target_horizon"
FOUR_STATE_HISTORY_FIELD: Final = "observer_history_token"
THREE_ACTION_HISTORY_FIELD: Final = "observer_history_action"

_CARRIER_CAUSAL_FIELDS: Final = frozenset(
    {
        CARRIER_HISTORY_FIELD,
        CARRIER_ACTION_FIELD,
        CARRIER_VISIBLE_FIELD,
        CAUSAL_PRIOR_FIELD,
        CAUSAL_PRIOR_FLAG,
    }
)

_FORBIDDEN_DEPLOYMENT_FIELDS: Final = frozenset(
    {
        "target_token",
        "future_clean_token",
        "canonical_target",
        "q_true",
        "hidden_clean_token",
    }
)


@dataclass(frozen=True)
class FeedbackWorldModelStep:
    """One exact analytic feedback-world-model update.

    ``corrected_prediction`` is the prediction exposed to a downstream
    consumer. ``next_belief`` is the auxiliary feedback state propagated by
    the observer.  They are distinct whenever the observed state differs from
    the auxiliary belief.
    """

    innovation: Tensor
    raw_velocity: Tensor
    corrected_velocity: Tensor
    corrected_prediction: Tensor
    next_belief: Tensor


@dataclass(frozen=True)
class FeedbackObserverArtifactAudit:
    """Whether an archive can form a causal pre-drop feedback residual."""

    scorable: bool
    source: str | None
    reason: str
    visible_state_count: int | None
    visible_action_count: int | None
    required_fields: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "scorable": self.scorable,
            "source": self.source,
            "reason": self.reason,
            "visible_state_count": self.visible_state_count,
            "visible_action_count": self.visible_action_count,
            "required_fields": list(self.required_fields),
            "uses_future_clean_target": False,
        }


def _finite_float_tensor(value: Tensor, *, name: str) -> Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be floating point")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")
    return value


def _shared_state_tensors(**values: Tensor) -> tuple[tuple[int, ...], torch.device]:
    shape: tuple[int, ...] | None = None
    device: torch.device | None = None
    for name, value in values.items():
        tensor = _finite_float_tensor(value, name=name)
        if shape is None:
            shape = tuple(tensor.shape)
            device = tensor.device
        elif tuple(tensor.shape) != shape:
            raise ValueError("all observer states must have the same shape")
        elif tensor.device != device:
            raise ValueError("all observer states must be on the same device")
    assert shape is not None and device is not None
    return shape, device


def _positive_scalar(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _feedback_gain(value: float, *, delta_t: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("feedback_gain must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("feedback_gain must be finite and positive")
    if not 0.0 < delta_t * result < 2.0:
        raise ValueError(
            "scalar feedback gain violates the discrete stability condition "
            "0 < delta_t * feedback_gain < 2"
        )
    return result


def feedback_world_model_step(
    observed_state: Tensor,
    feedback_belief: Tensor,
    raw_next_prediction: Tensor,
    *,
    feedback_gain: float,
    delta_t: float = 1.0,
) -> FeedbackWorldModelStep:
    """Apply the paper's scalar-gain latent observer equations exactly.

    The raw transition is interpreted in velocity form,

    ``v = (raw_next_prediction - observed_state) / delta_t``.

    The real-time innovation ``observed_state - feedback_belief`` corrects
    that velocity.  This function is only valid when ``observed_state`` is a
    deployment-time measurement, never an offline future target.
    """

    _shared_state_tensors(
        observed_state=observed_state,
        feedback_belief=feedback_belief,
        raw_next_prediction=raw_next_prediction,
    )
    step = _positive_scalar(delta_t, name="delta_t")
    gain = _feedback_gain(feedback_gain, delta_t=step)
    innovation = observed_state - feedback_belief
    raw_velocity = (raw_next_prediction - observed_state) / step
    corrected_velocity = raw_velocity + gain * innovation
    corrected_prediction = observed_state + step * corrected_velocity
    next_belief = feedback_belief + step * corrected_velocity
    return FeedbackWorldModelStep(
        innovation=innovation,
        raw_velocity=raw_velocity,
        corrected_velocity=corrected_velocity,
        corrected_prediction=corrected_prediction,
        next_belief=next_belief,
    )


def propagate_held_feedback_residual(
    current_state: Tensor,
    feedback_belief: Tensor,
    raw_next_prediction: Tensor,
    last_observed_innovation: Tensor,
    *,
    feedback_gain: float,
    missing_age: int,
    residual_decay: float = 1.0,
    delta_t: float = 1.0,
) -> FeedbackWorldModelStep:
    """Propagate briefly while vision is missing using the last valid residual.

    The original Feedback-WM receives a new observation after every action.
    Known visual dropout removes that update, so this project-specific
    extension holds (or causally decays) the last innovation that was formed
    *before* dropout.  It never estimates a residual from an offline clean
    target.  Recovery must call :func:`hard_measurement_reanchor` and form a
    fresh innovation from the newly available measurement.
    """

    _shared_state_tensors(
        current_state=current_state,
        feedback_belief=feedback_belief,
        raw_next_prediction=raw_next_prediction,
        last_observed_innovation=last_observed_innovation,
    )
    if type(missing_age) is not int or missing_age <= 0:
        raise ValueError("missing_age must be a positive integer")
    if isinstance(residual_decay, bool) or not isinstance(
        residual_decay, (int, float)
    ):
        raise TypeError("residual_decay must be numeric")
    decay = float(residual_decay)
    if not math.isfinite(decay) or not 0.0 <= decay <= 1.0:
        raise ValueError("residual_decay must be finite and in [0,1]")
    step = _positive_scalar(delta_t, name="delta_t")
    gain = _feedback_gain(feedback_gain, delta_t=step)
    effective_innovation = (decay ** (missing_age - 1)) * last_observed_innovation
    raw_velocity = (raw_next_prediction - current_state) / step
    corrected_velocity = raw_velocity + gain * effective_innovation
    corrected_prediction = current_state + step * corrected_velocity
    next_belief = feedback_belief + step * corrected_velocity
    return FeedbackWorldModelStep(
        innovation=effective_innovation,
        raw_velocity=raw_velocity,
        corrected_velocity=corrected_velocity,
        corrected_prediction=corrected_prediction,
        next_belief=next_belief,
    )


def partial_measurement_innovation_update(
    raw_prior: Tensor,
    partial_measurement: Tensor,
    *,
    measurement_gain: float,
) -> Tensor:
    """Correct a prior toward a deployable partial measurement.

    This is the static innovation form of the repository's complementary
    observer.  If its prior weight is ``K``, then

    ``geometry + K * (prior - geometry)``

    is exactly this function with ``measurement_gain = 1 - K``.
    """

    _shared_state_tensors(
        raw_prior=raw_prior,
        partial_measurement=partial_measurement,
    )
    if isinstance(measurement_gain, bool) or not isinstance(
        measurement_gain, (int, float)
    ):
        raise TypeError("measurement_gain must be numeric")
    gain = float(measurement_gain)
    if not math.isfinite(gain) or not 0.0 <= gain <= 1.0:
        raise ValueError("measurement_gain must be finite and in [0,1]")
    if gain == 0.0:
        return raw_prior
    if gain == 1.0:
        return partial_measurement
    return raw_prior + gain * (partial_measurement - raw_prior)


def hard_measurement_reanchor(measurement: Tensor) -> Tensor:
    """Return a bitwise measurement bypass for visible/recovery steps."""

    _finite_float_tensor(measurement, name="measurement")
    return measurement


def audit_feedback_observer_artifact(
    deployment_arrays: Mapping[str, np.ndarray],
) -> FeedbackObserverArtifactAudit:
    """Fail closed unless a causal pre-drop innovation can be constructed.

    A three-state P1 history is sufficient to predict the *next missing*
    state, but not to predict its own last visible state.  Feedback-WM needs
    the latter prediction followed by the real measurement to initialize the
    innovation before dropout.  Therefore an artifact must contain either:

    * four visible states plus their three executed transition actions; or
    * an explicitly stored causal one-step prior for the final visible state.

    Hidden future labels are rejected even if the archive contains one of the
    valid causal routes.
    """

    forbidden = sorted(_FORBIDDEN_DEPLOYMENT_FIELDS & set(deployment_arrays))
    if forbidden:
        raise PermissionError(
            "deployment archive contains forbidden offline fields: "
            f"{forbidden}"
        )

    history = np.asarray(deployment_arrays.get("history_token"))
    actions = np.asarray(deployment_arrays.get("history_action"))
    visible_states = int(history.shape[-3]) if history.ndim >= 3 else None
    visible_actions = int(actions.shape[-2]) if actions.ndim >= 2 else None

    # The exposed policy-carrier schema stores the causal context row-wise,
    # rather than under the legacy history_token/history_action names.  Its
    # three history states predict a separately stored fourth, still-visible
    # state.  The raw prior for that state must be explicitly marked causal.
    # Validate the complete bundle before considering the more permissive
    # legacy stored-prior route below; accepting only the prior and its flag
    # would silently discard row/context misalignment in the carrier artifact.
    present_carrier_fields = _CARRIER_CAUSAL_FIELDS & set(deployment_arrays)
    carrier_only_fields = {
        CARRIER_HISTORY_FIELD,
        CARRIER_ACTION_FIELD,
        CARRIER_VISIBLE_FIELD,
    }
    if present_carrier_fields & carrier_only_fields:
        missing = sorted(_CARRIER_CAUSAL_FIELDS - set(deployment_arrays))
        if missing:
            return FeedbackObserverArtifactAudit(
                scorable=False,
                source=None,
                reason=(
                    "policy-carrier causal pre-drop context is incomplete; "
                    f"missing {missing}"
                ),
                visible_state_count=None,
                visible_action_count=None,
                required_fields=tuple(missing),
            )

        carrier_history = np.asarray(deployment_arrays[CARRIER_HISTORY_FIELD])
        carrier_actions = np.asarray(deployment_arrays[CARRIER_ACTION_FIELD])
        carrier_visible = np.asarray(deployment_arrays[CARRIER_VISIBLE_FIELD])
        carrier_prior = np.asarray(deployment_arrays[CAUSAL_PRIOR_FIELD])
        causal = np.asarray(deployment_arrays[CAUSAL_PRIOR_FLAG])

        expected_history_tail = (
            P1_HISTORY_SIZE,
            P1_ANCHOR_COUNT,
            P1_TOKEN_DIM,
        )
        if (
            carrier_history.ndim != 4
            or carrier_history.shape[0] <= 0
            or carrier_history.shape[-3:] != expected_history_tail
        ):
            raise ValueError(
                f"{CARRIER_HISTORY_FIELD} must have shape [N,3,3,256] with N>0"
            )
        row_count = int(carrier_history.shape[0])
        expected_action_shape = (row_count, P1_HISTORY_SIZE, P1_ACTION_DIM)
        expected_state_shape = (row_count, P1_ANCHOR_COUNT, P1_TOKEN_DIM)
        if carrier_actions.shape != expected_action_shape:
            raise ValueError(
                f"{CARRIER_ACTION_FIELD} must align with history as "
                f"{expected_action_shape}, got {carrier_actions.shape}"
            )
        if carrier_visible.shape != expected_state_shape:
            raise ValueError(
                f"{CARRIER_VISIBLE_FIELD} must align with history as "
                f"{expected_state_shape}, got {carrier_visible.shape}"
            )
        if carrier_prior.shape != expected_state_shape:
            raise ValueError(
                f"{CAUSAL_PRIOR_FIELD} must align with the final visible token as "
                f"{expected_state_shape}, got {carrier_prior.shape}"
            )
        for name, value in (
            (CARRIER_HISTORY_FIELD, carrier_history),
            (CARRIER_ACTION_FIELD, carrier_actions),
            (CARRIER_VISIBLE_FIELD, carrier_visible),
            (CAUSAL_PRIOR_FIELD, carrier_prior),
        ):
            if value.dtype.kind != "f":
                raise TypeError(f"{name} must be floating point")
            if not bool(np.isfinite(value).all()):
                raise ValueError(f"{name} contains non-finite values")
        if causal.shape != (1,) or causal.dtype != np.bool_ or not bool(causal[0]):
            return FeedbackObserverArtifactAudit(
                scorable=False,
                source=None,
                reason=(
                    "policy-carrier prior lacks the scalar true "
                    f"{CAUSAL_PRIOR_FLAG} provenance flag"
                ),
                visible_state_count=P1_HISTORY_SIZE + 1,
                visible_action_count=P1_HISTORY_SIZE,
                required_fields=(CAUSAL_PRIOR_FLAG,),
            )

        # These receipts are emitted by the collector.  They are optional for
        # backwards compatibility, but if present they must encode the exact
        # H-1,H0,H1 -> H2 causal alignment.
        if CARRIER_HISTORY_HORIZONS_FIELD in deployment_arrays:
            horizons = np.asarray(
                deployment_arrays[CARRIER_HISTORY_HORIZONS_FIELD]
            )
            if (
                horizons.shape != (P1_HISTORY_SIZE,)
                or horizons.dtype.kind not in "iu"
                or not np.array_equal(horizons, np.asarray([-1, 0, 1]))
            ):
                raise ValueError(
                    f"{CARRIER_HISTORY_HORIZONS_FIELD} must equal [-1,0,1]"
                )
        if CARRIER_TARGET_HORIZON_FIELD in deployment_arrays:
            target_horizon = np.asarray(
                deployment_arrays[CARRIER_TARGET_HORIZON_FIELD]
            )
            if (
                target_horizon.shape != (1,)
                or target_horizon.dtype.kind not in "iu"
                or int(target_horizon[0]) != 2
            ):
                raise ValueError(f"{CARRIER_TARGET_HORIZON_FIELD} must equal [2]")

        if "policy_row_id" in deployment_arrays:
            row_id = np.asarray(deployment_arrays["policy_row_id"])
            if row_id.shape != (row_count,):
                raise ValueError(
                    "policy_row_id must have one identity per pre-drop context row"
                )
            if len(set(row_id.astype(str).tolist())) != row_count:
                raise ValueError("policy_row_id values must be unique")

        return FeedbackObserverArtifactAudit(
            scorable=True,
            source="policy_carrier_causal_pre_drop_prior",
            reason=(
                "three causal history states/actions, the separately observed "
                "final visible state, and its causal one-step prior are aligned"
            ),
            visible_state_count=P1_HISTORY_SIZE + 1,
            visible_action_count=P1_HISTORY_SIZE,
            required_fields=(),
        )

    if CAUSAL_PRIOR_FIELD in deployment_arrays:
        causal = np.asarray(deployment_arrays.get(CAUSAL_PRIOR_FLAG))
        if causal.shape != (1,) or causal.dtype != np.bool_ or not bool(causal[0]):
            return FeedbackObserverArtifactAudit(
                scorable=False,
                source=None,
                reason=(
                    f"{CAUSAL_PRIOR_FIELD} is present without a scalar true "
                    f"{CAUSAL_PRIOR_FLAG} provenance flag"
                ),
                visible_state_count=visible_states,
                visible_action_count=visible_actions,
                required_fields=(CAUSAL_PRIOR_FIELD, CAUSAL_PRIOR_FLAG),
            )
        prior = np.asarray(deployment_arrays[CAUSAL_PRIOR_FIELD])
        if history.ndim < 3 or prior.shape != history.shape[:-3] + history.shape[-2:]:
            return FeedbackObserverArtifactAudit(
                scorable=False,
                source=None,
                reason="stored causal pre-drop prior shape differs from the last visible token",
                visible_state_count=visible_states,
                visible_action_count=visible_actions,
                required_fields=(CAUSAL_PRIOR_FIELD, CAUSAL_PRIOR_FLAG),
            )
        if prior.dtype.kind != "f" or not bool(np.isfinite(prior).all()):
            raise ValueError("stored causal pre-drop prior must be finite floating point")
        return FeedbackObserverArtifactAudit(
            scorable=True,
            source="stored_causal_pre_drop_prior",
            reason="causal pre-drop prior and final visible measurement are available",
            visible_state_count=visible_states,
            visible_action_count=visible_actions,
            required_fields=(),
        )

    extended_history = deployment_arrays.get(FOUR_STATE_HISTORY_FIELD)
    extended_actions = deployment_arrays.get(THREE_ACTION_HISTORY_FIELD)
    if extended_history is not None or extended_actions is not None:
        tokens = np.asarray(extended_history)
        transition_actions = np.asarray(extended_actions)
        if tokens.ndim < 3 or tokens.shape[-3:] != (
            P1_HISTORY_SIZE + 1,
            P1_ANCHOR_COUNT,
            P1_TOKEN_DIM,
        ):
            raise ValueError(
                f"{FOUR_STATE_HISTORY_FIELD} must end in shape [4,3,256]"
            )
        if transition_actions.shape != tokens.shape[:-3] + (
            P1_HISTORY_SIZE,
            P1_ACTION_DIM,
        ):
            raise ValueError(
                f"{THREE_ACTION_HISTORY_FIELD} must align one action with each "
                "of the final three visible transitions"
            )
        if tokens.dtype.kind != "f" or transition_actions.dtype.kind != "f":
            raise TypeError("extended observer history must be floating point")
        if not bool(np.isfinite(tokens).all()) or not bool(
            np.isfinite(transition_actions).all()
        ):
            raise ValueError("extended observer history contains non-finite values")
        return FeedbackObserverArtifactAudit(
            scorable=True,
            source="four_visible_states",
            reason="the final visible state can be causally predicted then observed",
            visible_state_count=P1_HISTORY_SIZE + 1,
            visible_action_count=P1_HISTORY_SIZE,
            required_fields=(),
        )

    return FeedbackObserverArtifactAudit(
        scorable=False,
        source=None,
        reason=(
            "the archive has only three visible states for a three-state predictor; "
            "it can forecast the first missing state but cannot form a causal "
            "prediction error on the final visible state before dropout"
        ),
        visible_state_count=visible_states,
        visible_action_count=visible_actions,
        required_fields=(
            FOUR_STATE_HISTORY_FIELD,
            THREE_ACTION_HISTORY_FIELD,
            f"or {CAUSAL_PRIOR_FIELD}",
            CAUSAL_PRIOR_FLAG,
        ),
    )


__all__ = [
    "CAUSAL_PRIOR_FIELD",
    "CAUSAL_PRIOR_FLAG",
    "CARRIER_ACTION_FIELD",
    "CARRIER_HISTORY_FIELD",
    "CARRIER_HISTORY_HORIZONS_FIELD",
    "CARRIER_TARGET_HORIZON_FIELD",
    "CARRIER_VISIBLE_FIELD",
    "FOUR_STATE_HISTORY_FIELD",
    "THREE_ACTION_HISTORY_FIELD",
    "FeedbackObserverArtifactAudit",
    "FeedbackWorldModelStep",
    "audit_feedback_observer_artifact",
    "feedback_world_model_step",
    "hard_measurement_reanchor",
    "partial_measurement_innovation_update",
    "propagate_held_feedback_residual",
]
