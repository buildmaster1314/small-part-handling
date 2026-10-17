"""Causal policy runtime for the frozen P1-native complementary observer.

The runtime keeps the representation Gate's exact timing contract: three
decision-spaced anchor-token states and three actually executed five-action
blocks predict the current missing observation.  During a known visual drop,
the frozen LeWM prior is fused with held visual evidence reconstructed at the
current exact FK.  The posterior is written back before the next prediction.

No hidden-clean token, ``q_true``, or corruption schedule is accepted by this
module.  Those values may be used by an external evaluator only to synthesize
and score a known sensor interruption.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import torch
from torch import nn

from fcpg.lewm_known_drop_bridge_runtime import geometry_hold_policy_token
from fcpg.lewm_p1_native_complementary import P1_NATIVE_COMPLEMENTARY_GAINS


P1_POLICY_ACTION_STEPS: Final = 5
P1_POLICY_ACTION_DIM: Final = 20
P1_POLICY_HISTORY_STEPS: Final = 2
P1_POLICY_RUNTIME_METHODS: Final = (
    "measurement",
    "hold_last",
    "geometry_hold",
    "visual_cv",
    "complementary_factual",
    "complementary_no_action",
)


@dataclass(frozen=True)
class P1NativePolicyDecision:
    """One causal state choice and its optional policy-token override."""

    source: str
    sensor_valid: bool
    missing_age: int
    policy_token: torch.Tensor
    override_tokens: torch.Tensor
    override_mask: torch.Tensor
    global_visual_override: torch.Tensor | None = None
    global_visual_override_mask: torch.Tensor | None = None

    @property
    def override_required(self) -> bool:
        return bool(self.override_mask.any())

    def attach(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Attach selected anchor/global replacements without changing other inputs."""

        result = dict(batch)
        if self.override_required:
            result["grounded_token_override"] = self.override_tokens
            result["grounded_token_override_mask"] = self.override_mask
        has_global = self.global_visual_override is not None
        has_global_mask = self.global_visual_override_mask is not None
        if has_global != has_global_mask:
            raise RuntimeError("global visual override and mask must be provided together")
        if has_global:
            assert self.global_visual_override is not None
            assert self.global_visual_override_mask is not None
            if bool(self.global_visual_override_mask.any()):
                result["global_visual_token_override"] = self.global_visual_override
                result["global_visual_token_override_mask"] = (
                    self.global_visual_override_mask
                )
        return result


def _frozen_eval_model(model: nn.Module, *, name: str) -> None:
    if model.training:
        raise ValueError(f"{name} must be in eval mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError(f"{name} must be frozen")


def _finite_float(
    value: torch.Tensor,
    *,
    shape: tuple[int, ...],
    name: str,
    device: torch.device | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be floating point")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")
    return value.detach().float()


class P1NativeComplementaryPolicyRuntime:
    """Maintain one online P1-native policy state under known visual dropout."""

    def __init__(
        self,
        *,
        method: str,
        policy: nn.Module,
        factual_predictor: nn.Module | None = None,
        no_action_predictor: nn.Module | None = None,
        gains: Sequence[float] = P1_NATIVE_COMPLEMENTARY_GAINS,
        policy_history_steps: int = P1_POLICY_HISTORY_STEPS,
        freeze_global_visual_during_drop: bool = False,
    ) -> None:
        if method not in P1_POLICY_RUNTIME_METHODS:
            raise ValueError(f"unsupported P1 policy method {method!r}")
        if policy_history_steps != P1_POLICY_HISTORY_STEPS:
            raise ValueError("the frozen P1 policy requires exactly two history slots")
        if len(gains) != len(P1_NATIVE_COMPLEMENTARY_GAINS):
            raise ValueError("the frozen P1 observer requires gains for H3-H8")
        normalized_gains = tuple(float(value) for value in gains)
        if any(not 0.0 <= value <= 1.0 for value in normalized_gains):
            raise ValueError("all complementary gains must lie in [0,1]")
        _frozen_eval_model(policy, name="policy")
        if method == "complementary_factual":
            if factual_predictor is None:
                raise ValueError("complementary_factual requires a factual predictor")
            _frozen_eval_model(factual_predictor, name="factual predictor")
        if method == "complementary_no_action":
            if no_action_predictor is None:
                raise ValueError("complementary_no_action requires a no-action predictor")
            _frozen_eval_model(no_action_predictor, name="no-action predictor")

        self.method = method
        self.policy = policy
        self.factual_predictor = factual_predictor
        self.no_action_predictor = no_action_predictor
        self.gains = normalized_gains
        self.policy_history_steps = int(policy_history_steps)
        self.freeze_global_visual_during_drop = bool(
            freeze_global_visual_during_drop
        )
        self._tokens: deque[torch.Tensor] = deque(maxlen=3)
        self._actions: deque[torch.Tensor] = deque(maxlen=3)
        self._pending_action = False
        self._missing_age = 0
        self._last_valid_token: torch.Tensor | None = None
        self._last_valid_visual: torch.Tensor | None = None
        self._last_valid_global_visual_token: torch.Tensor | None = None
        self._valid_visuals: deque[torch.Tensor] = deque(maxlen=2)
        self._batch_size: int | None = None
        self._num_keypoints: int | None = None
        self._token_dim: int | None = None
        self._device: torch.device | None = None

    @property
    def ready_for_drop(self) -> bool:
        return (
            not self._pending_action
            and len(self._tokens) == 3
            and len(self._actions) == 3
        )

    @property
    def missing_age(self) -> int:
        return self._missing_age

    def _require_decision_phase(self) -> None:
        if self._pending_action:
            raise RuntimeError("register the executed action before the next decision")

    def _state_token(self, value: torch.Tensor, *, name: str) -> torch.Tensor:
        if self._batch_size is None:
            if value.ndim != 3:
                raise ValueError(f"{name} must have shape [B,K,D]")
            self._batch_size, self._num_keypoints, self._token_dim = map(
                int, value.shape
            )
            self._device = value.device
        assert self._batch_size is not None
        assert self._num_keypoints is not None
        assert self._token_dim is not None
        assert self._device is not None
        return _finite_float(
            value,
            shape=(self._batch_size, self._num_keypoints, self._token_dim),
            name=name,
            device=self._device,
        )

    def _xyz(self, value: torch.Tensor) -> torch.Tensor:
        if self._batch_size is None or self._num_keypoints is None:
            raise RuntimeError("a valid observation must initialize the runtime")
        assert self._device is not None
        return _finite_float(
            value,
            shape=(self._batch_size, self._num_keypoints, 3),
            name="current_xyz",
            device=self._device,
        )

    def _visual(self, value: torch.Tensor) -> torch.Tensor:
        if self._batch_size is None or self._num_keypoints is None:
            raise RuntimeError("a native token must initialize the runtime")
        assert self._device is not None
        visual_dim = int(getattr(self.policy, "vision_dim", -1))
        return _finite_float(
            value,
            shape=(self._batch_size, self._num_keypoints, visual_dim),
            name="sampled_visual",
            device=self._device,
        )

    def _global_visual_token(self, value: torch.Tensor) -> torch.Tensor:
        if self._batch_size is None or self._token_dim is None:
            raise RuntimeError("a native token must initialize the runtime")
        assert self._device is not None
        return _finite_float(
            value,
            shape=(self._batch_size, self._token_dim),
            name="global_visual_token",
            device=self._device,
        )

    def _decision(
        self,
        token: torch.Tensor,
        *,
        source: str,
        sensor_valid: bool,
        override_required: bool,
        missing_age: int,
    ) -> P1NativePolicyDecision:
        self._tokens.append(token.detach())
        self._pending_action = True
        override = token.unsqueeze(1).expand(
            -1, self.policy_history_steps, -1, -1
        )
        assert self._batch_size is not None and self._num_keypoints is not None
        mask = torch.full(
            (self._batch_size, self.policy_history_steps, self._num_keypoints),
            override_required,
            dtype=torch.bool,
            device=token.device,
        )
        global_override = None
        global_mask = None
        if (
            override_required
            and self.freeze_global_visual_during_drop
            and self._last_valid_global_visual_token is not None
        ):
            global_override = self._last_valid_global_visual_token.unsqueeze(1).expand(
                -1, self.policy_history_steps, -1
            )
            global_mask = torch.ones(
                (self._batch_size, self.policy_history_steps),
                dtype=torch.bool,
                device=token.device,
            )
        return P1NativePolicyDecision(
            source=source,
            sensor_valid=sensor_valid,
            missing_age=missing_age,
            policy_token=token.detach(),
            override_tokens=override.detach(),
            override_mask=mask,
            global_visual_override=(
                None if global_override is None else global_override.detach()
            ),
            global_visual_override_mask=global_mask,
        )

    @torch.inference_mode()
    def begin_valid(
        self,
        native_token: torch.Tensor,
        sampled_visual: torch.Tensor,
        global_visual_token: torch.Tensor | None = None,
    ) -> P1NativePolicyDecision:
        """Consume a visible native token and hard-reset missing-age state."""

        self._require_decision_phase()
        token = self._state_token(native_token, name="native_token")
        visual = self._visual(sampled_visual)
        self._last_valid_token = token
        self._last_valid_visual = visual
        self._last_valid_global_visual_token = (
            None
            if global_visual_token is None
            else self._global_visual_token(global_visual_token)
        )
        self._valid_visuals.append(visual)
        self._missing_age = 0
        return self._decision(
            token,
            source="native_measurement",
            sensor_valid=True,
            override_required=False,
            missing_age=0,
        )

    def _predict_prior(self, predictor: nn.Module) -> torch.Tensor:
        assert self._batch_size is not None
        assert self._num_keypoints is not None
        assert self._token_dim is not None
        history = torch.stack(tuple(self._tokens), dim=1)
        actions = torch.stack(tuple(self._actions), dim=1)
        prediction = predictor.predict(history, actions)
        expected = (self._batch_size, 3, self._num_keypoints, self._token_dim)
        if tuple(prediction.shape) != expected:
            raise ValueError(
                f"P1 predictor returned {tuple(prediction.shape)}, expected {expected}"
            )
        if not bool(torch.isfinite(prediction).all()):
            raise ValueError("P1 predictor returned non-finite values")
        return prediction[:, -1].detach().float()

    @torch.inference_mode()
    def begin_invalid(
        self,
        current_xyz: torch.Tensor,
        *,
        corrupted_token: torch.Tensor | None = None,
        allow_geometry_fallback: bool = False,
    ) -> P1NativePolicyDecision:
        """Choose the current state without accepting any hidden-clean input."""

        self._require_decision_phase()
        if not self.ready_for_drop:
            raise RuntimeError(
                "known dropout requires three prior decisions and executed action blocks"
            )
        horizon_exhausted = self._missing_age >= len(self.gains)
        if horizon_exhausted and not allow_geometry_fallback:
            raise RuntimeError("known dropout exceeds the frozen H3-H8 horizon")
        xyz = self._xyz(current_xyz)
        assert self._last_valid_token is not None
        assert self._last_valid_visual is not None
        age = self._missing_age + 1

        if horizon_exhausted:
            if corrupted_token is not None:
                raise ValueError("geometry fallback must not consume a corrupted token")
            token = geometry_hold_policy_token(
                self.policy,
                self._last_valid_visual,
                xyz,
            ).detach().float()
            source = "geometry_horizon_fallback"
            override_required = True
        elif self.method == "measurement":
            if corrupted_token is None:
                raise ValueError("measurement baseline requires the observed corrupted token")
            token = self._state_token(corrupted_token, name="corrupted_token")
            source = "corrupted_measurement"
            override_required = False
        elif self.method == "hold_last":
            if corrupted_token is not None:
                raise ValueError("hold_last must not consume a corrupted token")
            token = self._last_valid_token
            source = "hold_last"
            override_required = True
        else:
            if corrupted_token is not None:
                raise ValueError(f"{self.method} must not consume a corrupted token")
            geometry = geometry_hold_policy_token(
                self.policy,
                self._last_valid_visual,
                xyz,
            ).detach().float()
            if self.method == "geometry_hold":
                token = geometry
                source = "geometry_hold"
            elif self.method == "visual_cv":
                if len(self._valid_visuals) != 2:
                    raise RuntimeError("visual_cv requires two visible descriptor states")
                previous, current = tuple(self._valid_visuals)
                extrapolated = current + float(age) * (current - previous)
                token = geometry_hold_policy_token(
                    self.policy,
                    extrapolated,
                    xyz,
                ).detach().float()
                source = "visual_cv"
            else:
                predictor = (
                    self.factual_predictor
                    if self.method == "complementary_factual"
                    else self.no_action_predictor
                )
                assert predictor is not None
                prior = self._predict_prior(predictor)
                gain = self.gains[self._missing_age]
                token = geometry + gain * (prior - geometry)
                source = self.method
            override_required = True

        self._missing_age = age
        return self._decision(
            token,
            source=source,
            sensor_valid=False,
            override_required=override_required,
            missing_age=age,
        )

    def register_executed_action(self, action_block: torch.Tensor) -> None:
        """Append the actual clipped 5x4 action block for the next prior."""

        if not self._pending_action:
            raise RuntimeError("begin a decision before registering its action")
        if self._batch_size is None or self._device is None:
            raise RuntimeError("runtime is not initialized")
        if action_block.ndim == 3:
            if tuple(action_block.shape) != (
                self._batch_size,
                P1_POLICY_ACTION_STEPS,
                4,
            ):
                raise ValueError("action_block must have shape [B,5,4]")
            action_block = action_block.reshape(self._batch_size, P1_POLICY_ACTION_DIM)
        action = _finite_float(
            action_block,
            shape=(self._batch_size, P1_POLICY_ACTION_DIM),
            name="action_block",
            device=self._device,
        )
        self._actions.append(action)
        self._pending_action = False


__all__ = [
    "P1_POLICY_ACTION_DIM",
    "P1_POLICY_ACTION_STEPS",
    "P1_POLICY_HISTORY_STEPS",
    "P1_POLICY_RUNTIME_METHODS",
    "P1NativeComplementaryPolicyRuntime",
    "P1NativePolicyDecision",
]
