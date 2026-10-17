"""Causal expert-action residuals for known visual interruptions.

The frozen policy and observer produce the deployment features.  A small head
only corrects the next five policy actions while the external sensor-valid bit
is false.  Valid observations bypass the head exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


PROTOCOL: Final = "lewm-expert-action-residual-pick-v1"
METHODS: Final = (
    "complementary_factual",
    "complementary_no_action",
    "geometry_hold",
)
ACTION_STEPS: Final = 5
ACTION_DIM: Final = 4
MOTION_DIM: Final = 3
MAX_MISSING_AGE: Final = 6


@dataclass(frozen=True, slots=True)
class ExpertResidualArrays:
    """Aligned deployment features and expert action labels."""

    episode_id: np.ndarray
    decision_index: np.ndarray
    missing_age: np.ndarray
    sensor_valid: np.ndarray
    state_latent: np.ndarray
    action_latent: np.ndarray
    base_action: np.ndarray
    posterior_token: np.ndarray
    expert_action: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.episode_id.shape[0])

    @property
    def method_count(self) -> int:
        return int(self.state_latent.shape[1])

    @property
    def model_dim(self) -> int:
        return int(self.state_latent.shape[-1])


def validate_expert_residual_arrays(
    arrays: ExpertResidualArrays,
    *,
    methods: tuple[str, ...] = METHODS,
) -> ExpertResidualArrays:
    """Fail closed on shape, ordering, finiteness, or causal-routing drift."""

    count = arrays.sample_count
    method_count = len(methods)
    if count <= 0 or len(set(methods)) != method_count:
        raise ValueError("expert residual arrays require samples and unique methods")
    if arrays.episode_id.shape != (count,) or arrays.episode_id.dtype.kind not in "iu":
        raise ValueError("episode_id must be an integer vector")
    if arrays.decision_index.shape != (count,) or arrays.decision_index.dtype.kind not in "iu":
        raise ValueError("decision_index must be an integer vector")
    if arrays.missing_age.shape != (count,) or arrays.missing_age.dtype.kind not in "iu":
        raise ValueError("missing_age must be an integer vector")
    if arrays.sensor_valid.shape != (count,) or arrays.sensor_valid.dtype != np.bool_:
        raise ValueError("sensor_valid must be a boolean vector")
    if np.any(arrays.sensor_valid):
        raise PermissionError("training rows must come only from invalid observations")
    if np.any(arrays.missing_age < 1) or np.any(arrays.missing_age > MAX_MISSING_AGE):
        raise ValueError("missing_age lies outside the frozen H3-H8 interval")
    if len(np.unique(np.stack((arrays.episode_id, arrays.decision_index), axis=1), axis=0)) != count:
        raise ValueError("episode/decision rows must be unique")

    state = np.asarray(arrays.state_latent)
    action = np.asarray(arrays.action_latent)
    base = np.asarray(arrays.base_action)
    posterior = np.asarray(arrays.posterior_token)
    expert = np.asarray(arrays.expert_action)
    if state.ndim != 3 or state.shape[:2] != (count, method_count):
        raise ValueError("state_latent must have shape [N,M,D]")
    model_dim = int(state.shape[-1])
    if action.shape != (count, method_count, ACTION_STEPS, model_dim):
        raise ValueError("action_latent must have shape [N,M,5,D]")
    if base.shape != (count, method_count, ACTION_STEPS, ACTION_DIM):
        raise ValueError("base_action must have shape [N,M,5,4]")
    if posterior.ndim != 4 or posterior.shape[:2] != (count, method_count):
        raise ValueError("posterior_token must have shape [N,M,K,D]")
    if posterior.shape[-1] != model_dim:
        raise ValueError("posterior and policy latent dimensions differ")
    if expert.shape != (count, ACTION_STEPS, ACTION_DIM):
        raise ValueError("expert_action must have shape [N,5,4]")
    for name, value in (
        ("state_latent", state),
        ("action_latent", action),
        ("base_action", base),
        ("posterior_token", posterior),
        ("expert_action", expert),
    ):
        if value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite floating point")
    if np.any(base < -1.00001) or np.any(base > 1.00001):
        raise ValueError("base actions exceed the environment bounds")
    if np.any(expert < -1.00001) or np.any(expert > 1.00001):
        raise ValueError("expert actions exceed the environment bounds")
    return arrays


def save_expert_residual_arrays(
    path: str | Path,
    arrays: ExpertResidualArrays,
    *,
    methods: tuple[str, ...] = METHODS,
) -> None:
    validated = validate_expert_residual_arrays(arrays, methods=methods)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        schema_version=np.asarray(1, dtype=np.int64),
        protocol=np.asarray(PROTOCOL),
        methods=np.asarray(methods),
        episode_id=np.asarray(validated.episode_id, dtype=np.int64),
        decision_index=np.asarray(validated.decision_index, dtype=np.int64),
        missing_age=np.asarray(validated.missing_age, dtype=np.int64),
        sensor_valid=np.asarray(validated.sensor_valid, dtype=bool),
        state_latent=np.asarray(validated.state_latent, dtype=np.float32),
        action_latent=np.asarray(validated.action_latent, dtype=np.float32),
        base_action=np.asarray(validated.base_action, dtype=np.float32),
        posterior_token=np.asarray(validated.posterior_token, dtype=np.float32),
        expert_action=np.asarray(validated.expert_action, dtype=np.float32),
    )


def load_expert_residual_arrays(
    path: str | Path,
    *,
    methods: tuple[str, ...] = METHODS,
) -> ExpertResidualArrays:
    with np.load(Path(path), allow_pickle=False) as archive:
        if int(np.asarray(archive["schema_version"]).reshape(-1)[0]) != 1:
            raise ValueError("expert residual cache schema is unsupported")
        if str(np.asarray(archive["protocol"]).reshape(-1)[0]) != PROTOCOL:
            raise ValueError("expert residual cache protocol is unsupported")
        cached_methods = tuple(np.asarray(archive["methods"]).astype(str).tolist())
        if cached_methods != methods:
            raise ValueError("expert residual method ordering drifted")
        arrays = ExpertResidualArrays(
            episode_id=np.array(archive["episode_id"], copy=True),
            decision_index=np.array(archive["decision_index"], copy=True),
            missing_age=np.array(archive["missing_age"], copy=True),
            sensor_valid=np.array(archive["sensor_valid"], copy=True),
            state_latent=np.array(archive["state_latent"], copy=True),
            action_latent=np.array(archive["action_latent"], copy=True),
            base_action=np.array(archive["base_action"], copy=True),
            posterior_token=np.array(archive["posterior_token"], copy=True),
            expert_action=np.array(archive["expert_action"], copy=True),
        )
    return validate_expert_residual_arrays(arrays, methods=methods)


class ExpertActionResidualHead(nn.Module):
    """Small zero-initialized correction of a frozen policy's five actions."""

    def __init__(
        self,
        *,
        model_dim: int,
        hidden_dim: int = 128,
        age_dim: int = 16,
        action_steps: int = ACTION_STEPS,
    ) -> None:
        super().__init__()
        if min(model_dim, hidden_dim, age_dim, action_steps) <= 0:
            raise ValueError("residual-head dimensions must be positive")
        if action_steps != ACTION_STEPS:
            raise ValueError(f"residual head is fixed to {ACTION_STEPS} actions")
        self.model_dim = int(model_dim)
        self.hidden_dim = int(hidden_dim)
        self.age_embedding = nn.Embedding(MAX_MISSING_AGE + 1, int(age_dim))
        self.step_embedding = nn.Embedding(ACTION_STEPS, int(age_dim))
        self.state_norm = nn.LayerNorm(self.model_dim)
        self.action_norm = nn.LayerNorm(self.model_dim)
        input_dim = 2 * self.model_dim + ACTION_DIM + 2 * int(age_dim)
        self.input_projection = nn.Linear(input_dim, self.hidden_dim)
        self.hidden_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, MOTION_DIM)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.register_buffer("action_low", torch.full((MOTION_DIM,), -1.0))
        self.register_buffer("action_high", torch.full((MOTION_DIM,), 1.0))

    def forward(
        self,
        *,
        base_action: torch.Tensor,
        state_latent: torch.Tensor,
        action_latent: torch.Tensor,
        missing_age: torch.Tensor,
        sensor_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = int(base_action.shape[0])
        if base_action.shape != (batch, ACTION_STEPS, ACTION_DIM):
            raise ValueError("base_action must have shape [B,5,4]")
        if state_latent.shape != (batch, self.model_dim):
            raise ValueError("state_latent must have shape [B,D]")
        if action_latent.shape != (batch, ACTION_STEPS, self.model_dim):
            raise ValueError("action_latent must have shape [B,5,D]")
        if missing_age.shape != (batch,) or missing_age.dtype != torch.long:
            raise ValueError("missing_age must be a long tensor with shape [B]")
        if bool(torch.any((missing_age < 0) | (missing_age > MAX_MISSING_AGE))):
            raise ValueError("missing_age is outside the supported range")
        if not all(
            bool(torch.isfinite(value).all())
            for value in (base_action, state_latent, action_latent)
        ):
            raise ValueError("residual-head inputs contain non-finite values")

        state = self.state_norm(state_latent).unsqueeze(1).expand(-1, ACTION_STEPS, -1)
        action = self.action_norm(action_latent)
        age = self.age_embedding(missing_age).unsqueeze(1).expand(-1, ACTION_STEPS, -1)
        steps = torch.arange(ACTION_STEPS, device=base_action.device)
        step = self.step_embedding(steps).unsqueeze(0).expand(batch, -1, -1)
        features = torch.cat((state, action, base_action, age, step), dim=-1)
        hidden = F.gelu(self.input_projection(features))
        hidden = hidden + F.gelu(self.hidden_projection(hidden))
        direction = torch.tanh(self.output_projection(hidden))
        base_motion = base_action[..., :MOTION_DIM]
        low = self.action_low.to(base_action).view(1, 1, MOTION_DIM)
        high = self.action_high.to(base_action).view(1, 1, MOTION_DIM)
        room = torch.where(direction >= 0, high - base_motion, base_motion - low)
        corrected_motion = base_motion + room * direction
        corrected = torch.cat((corrected_motion, base_action[..., MOTION_DIM:]), dim=-1)
        if sensor_valid is None:
            return corrected
        if sensor_valid.shape != (batch,) or sensor_valid.dtype != torch.bool:
            raise ValueError("sensor_valid must be a bool tensor with shape [B]")
        return torch.where(sensor_valid[:, None, None], base_action, corrected)


def expert_action_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float = 0.05,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape[-2:] != (
        ACTION_STEPS,
        ACTION_DIM,
    ):
        raise ValueError("expert action loss expects aligned [B,5,4] tensors")
    motion = F.smooth_l1_loss(
        prediction[..., :3], target[..., :3], reduction="mean", beta=float(beta)
    )
    return motion


def method_index(method: str, *, methods: tuple[str, ...] = METHODS) -> int:
    try:
        return methods.index(str(method))
    except ValueError as error:
        raise ValueError(f"unknown expert residual method {method!r}") from error


def load_expert_action_residual_head(
    path: str | Path,
    *,
    device: torch.device,
    expected_method: str | None = None,
    expected_seed: int | None = None,
) -> tuple[ExpertActionResidualHead, dict[str, object]]:
    """Load one frozen residual head and validate its experiment identity."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("expert residual checkpoint must contain a mapping")
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("expert residual checkpoint schema is unsupported")
    if str(payload.get("protocol", "")) != PROTOCOL:
        raise ValueError("expert residual checkpoint protocol is unsupported")
    method = str(payload.get("method", ""))
    if method not in METHODS:
        raise ValueError(f"expert residual checkpoint has unknown method {method!r}")
    seed = int(payload.get("seed", -1))
    if expected_method is not None and method != expected_method:
        raise ValueError(
            f"expert residual method mismatch: {method!r} != {expected_method!r}"
        )
    if expected_seed is not None and seed != int(expected_seed):
        raise ValueError(
            f"expert residual seed mismatch: {seed} != {int(expected_seed)}"
        )
    model_dim = int(payload.get("model_dim", 0))
    hidden_dim = int(payload.get("hidden_dim", 0))
    state = payload.get("model_state")
    if model_dim <= 0 or hidden_dim <= 0 or not isinstance(state, dict):
        raise ValueError("expert residual checkpoint architecture is incomplete")
    model = ExpertActionResidualHead(
        model_dim=model_dim,
        hidden_dim=hidden_dim,
    )
    model.load_state_dict(state, strict=True)
    model.to(device).requires_grad_(False).eval()
    metadata: dict[str, object] = {
        "method": method,
        "seed": seed,
        "steps": int(payload.get("steps", -1)),
        "model_dim": model_dim,
        "hidden_dim": hidden_dim,
    }
    return model, metadata


__all__ = [
    "ACTION_DIM",
    "ACTION_STEPS",
    "ExpertActionResidualHead",
    "ExpertResidualArrays",
    "MAX_MISSING_AGE",
    "MOTION_DIM",
    "METHODS",
    "PROTOCOL",
    "expert_action_loss",
    "load_expert_residual_arrays",
    "load_expert_action_residual_head",
    "method_index",
    "save_expert_residual_arrays",
    "validate_expert_residual_arrays",
]
