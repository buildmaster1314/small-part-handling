"""Data and model contracts for on-policy posterior DAgger.

Each stored row belongs to one observer method and one learner-visited state.
Expert and executed action chunks are labels and receipts for that state; rows
are deliberately not aligned along a method axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SCHEMA_VERSION: Final = 1
PROTOCOL: Final = "fcpg_lewm_posterior_dagger_v1"
CHECKPOINT_TYPE: Final = "posterior_dagger_residual_head"
METHODS: Final = (
    "complementary_factual",
    "complementary_no_action",
    "geometry_hold",
)
ACTION_STEPS: Final = 5
ACTION_DIM: Final = 4
MOTION_DIM: Final = 3
MAX_MISSING_AGE: Final = 6

_ARRAY_FIELDS: Final = (
    "episode_id",
    "decision_index",
    "dagger_iteration",
    "method_index",
    "missing_age",
    "sensor_valid",
    "state_latent",
    "action_latent",
    "base_action",
    "posterior_token",
    "expert_action",
    "executed_action",
)
_ARCHITECTURE_FIELDS: Final = frozenset(
    {"model_dim", "hidden_dim", "age_dim", "max_xyz_correction"}
)


@dataclass(frozen=True, slots=True)
class PosteriorDaggerArrays:
    """Unpaired per-method rows collected from learner-visited states."""

    episode_id: np.ndarray
    decision_index: np.ndarray
    dagger_iteration: np.ndarray
    method_index: np.ndarray
    missing_age: np.ndarray
    sensor_valid: np.ndarray
    state_latent: np.ndarray
    action_latent: np.ndarray
    base_action: np.ndarray
    posterior_token: np.ndarray
    expert_action: np.ndarray
    executed_action: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(np.asarray(self.episode_id).shape[0])

    @property
    def model_dim(self) -> int:
        state = np.asarray(self.state_latent)
        return int(state.shape[-1]) if state.ndim == 2 else 0

    @property
    def posterior_token_count(self) -> int:
        posterior = np.asarray(self.posterior_token)
        return int(posterior.shape[1]) if posterior.ndim == 3 else 0


def _integer_vector(value: np.ndarray, *, count: int, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (count,) or array.dtype.kind not in "iu":
        raise ValueError(f"{label} must be an integer vector")
    return array


def _finite_float(value: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind != "f" or not bool(np.isfinite(array).all()):
        raise ValueError(f"{label} must be finite floating point")
    return array


def _bounded_actions(value: np.ndarray, *, label: str) -> None:
    if bool(np.any(value < -1.000001)) or bool(np.any(value > 1.000001)):
        raise ValueError(f"{label} exceeds the environment action bounds")


def validate_posterior_dagger_arrays(
    arrays: PosteriorDaggerArrays,
) -> PosteriorDaggerArrays:
    """Fail closed on row identity, causality, shape, and numeric drift."""

    count = arrays.sample_count
    if count <= 0:
        raise ValueError("posterior DAgger arrays require at least one row")

    episode = _integer_vector(arrays.episode_id, count=count, label="episode_id")
    decision = _integer_vector(
        arrays.decision_index, count=count, label="decision_index"
    )
    iteration = _integer_vector(
        arrays.dagger_iteration, count=count, label="dagger_iteration"
    )
    method = _integer_vector(arrays.method_index, count=count, label="method_index")
    missing_age = _integer_vector(
        arrays.missing_age, count=count, label="missing_age"
    )
    if any(bool(np.any(value < 0)) for value in (episode, decision, iteration)):
        raise ValueError("episode, decision, and DAgger iteration must be non-negative")
    if bool(np.any(method < 0)) or bool(np.any(method >= len(METHODS))):
        raise ValueError("method_index lies outside the frozen method ordering")
    if bool(np.any(missing_age < 1)) or bool(
        np.any(missing_age > MAX_MISSING_AGE)
    ):
        raise ValueError("missing_age lies outside the invalid-observation range")

    sensor_valid = np.asarray(arrays.sensor_valid)
    if sensor_valid.shape != (count,) or sensor_valid.dtype != np.dtype(np.bool_):
        raise ValueError("sensor_valid must be a boolean vector")
    if bool(np.any(sensor_valid)):
        raise PermissionError("training rows must contain invalid observations only")

    row_identity = np.stack((iteration, method, episode, decision), axis=1)
    if int(np.unique(row_identity, axis=0).shape[0]) != count:
        raise ValueError(
            "(iteration, method, episode, decision) identities must be unique"
        )

    state = _finite_float(arrays.state_latent, label="state_latent")
    if state.ndim != 2 or state.shape[0] != count or state.shape[1] <= 0:
        raise ValueError("state_latent must have shape [N,D] with D > 0")
    model_dim = int(state.shape[1])

    action = _finite_float(arrays.action_latent, label="action_latent")
    if action.shape != (count, ACTION_STEPS, model_dim):
        raise ValueError("action_latent must have shape [N,5,D]")

    posterior = _finite_float(arrays.posterior_token, label="posterior_token")
    if (
        posterior.ndim != 3
        or posterior.shape[0] != count
        or posterior.shape[1] <= 0
        or posterior.shape[2] != model_dim
    ):
        raise ValueError("posterior_token must have shape [N,K,D] with K > 0")

    action_arrays = (
        ("base_action", arrays.base_action),
        ("expert_action", arrays.expert_action),
        ("executed_action", arrays.executed_action),
    )
    for label, value in action_arrays:
        action_value = _finite_float(value, label=label)
        if action_value.shape != (count, ACTION_STEPS, ACTION_DIM):
            raise ValueError(f"{label} must have shape [N,5,4]")
        _bounded_actions(action_value, label=label)
    return arrays


def method_index(method: str) -> int:
    """Return the immutable index used in stored rows and checkpoints."""

    try:
        return METHODS.index(str(method))
    except ValueError as error:
        raise ValueError(f"unknown posterior DAgger method {method!r}") from error


def save_posterior_dagger_arrays(
    path: str | Path,
    arrays: PosteriorDaggerArrays,
) -> None:
    validated = validate_posterior_dagger_arrays(arrays)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int64),
        protocol=np.asarray(PROTOCOL),
        methods=np.asarray(METHODS),
        action_steps=np.asarray(ACTION_STEPS, dtype=np.int64),
        action_dim=np.asarray(ACTION_DIM, dtype=np.int64),
        motion_dim=np.asarray(MOTION_DIM, dtype=np.int64),
        max_missing_age=np.asarray(MAX_MISSING_AGE, dtype=np.int64),
        episode_id=np.asarray(validated.episode_id, dtype=np.int64),
        decision_index=np.asarray(validated.decision_index, dtype=np.int64),
        dagger_iteration=np.asarray(validated.dagger_iteration, dtype=np.int64),
        method_index=np.asarray(validated.method_index, dtype=np.int64),
        missing_age=np.asarray(validated.missing_age, dtype=np.int64),
        sensor_valid=np.asarray(validated.sensor_valid, dtype=bool),
        state_latent=np.asarray(validated.state_latent, dtype=np.float32),
        action_latent=np.asarray(validated.action_latent, dtype=np.float32),
        base_action=np.asarray(validated.base_action, dtype=np.float32),
        posterior_token=np.asarray(validated.posterior_token, dtype=np.float32),
        expert_action=np.asarray(validated.expert_action, dtype=np.float32),
        executed_action=np.asarray(validated.executed_action, dtype=np.float32),
    )


def _scalar(archive: Mapping[str, np.ndarray], key: str) -> object:
    value = np.asarray(archive[key])
    if value.shape != ():
        raise ValueError(f"posterior DAgger metadata {key!r} must be scalar")
    return value.item()


def load_posterior_dagger_arrays(path: str | Path) -> PosteriorDaggerArrays:
    metadata_keys = {
        "schema_version",
        "protocol",
        "methods",
        "action_steps",
        "action_dim",
        "motion_dim",
        "max_missing_age",
    }
    expected_keys = metadata_keys | set(_ARRAY_FIELDS)
    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise ValueError("posterior DAgger archive fields do not match the schema")
        if int(_scalar(archive, "schema_version")) != SCHEMA_VERSION:
            raise ValueError("posterior DAgger schema is unsupported")
        if str(_scalar(archive, "protocol")) != PROTOCOL:
            raise ValueError("posterior DAgger protocol is unsupported")
        cached_methods = tuple(np.asarray(archive["methods"]).astype(str).tolist())
        if cached_methods != METHODS:
            raise ValueError("posterior DAgger method ordering drifted")
        constants = {
            "action_steps": ACTION_STEPS,
            "action_dim": ACTION_DIM,
            "motion_dim": MOTION_DIM,
            "max_missing_age": MAX_MISSING_AGE,
        }
        for key, expected in constants.items():
            if int(_scalar(archive, key)) != expected:
                raise ValueError(f"posterior DAgger constant {key!r} drifted")
        arrays = PosteriorDaggerArrays(
            **{name: np.array(archive[name], copy=True) for name in _ARRAY_FIELDS}
        )
    return validate_posterior_dagger_arrays(arrays)


class PosteriorDaggerResidualHead(nn.Module):
    """Bounded action residual conditioned on an on-policy posterior state."""

    def __init__(
        self,
        *,
        model_dim: int,
        hidden_dim: int = 128,
        age_dim: int = 16,
        max_xyz_correction: float = 0.25,
    ) -> None:
        super().__init__()
        if min(model_dim, hidden_dim, age_dim) <= 0:
            raise ValueError("posterior DAgger head dimensions must be positive")
        if not 0.0 < float(max_xyz_correction) <= 2.0:
            raise ValueError("max_xyz_correction must lie in (0, 2]")
        self.model_dim = int(model_dim)
        self.hidden_dim = int(hidden_dim)
        self.age_dim = int(age_dim)
        self.max_xyz_correction = float(max_xyz_correction)
        self.state_norm = nn.LayerNorm(self.model_dim)
        self.action_norm = nn.LayerNorm(self.model_dim)
        self.posterior_norm = nn.LayerNorm(self.model_dim)
        self.age_embedding = nn.Embedding(MAX_MISSING_AGE + 1, self.age_dim)
        self.step_embedding = nn.Embedding(ACTION_STEPS, self.age_dim)
        input_dim = 3 * self.model_dim + ACTION_DIM + 2 * self.age_dim
        self.input_projection = nn.Linear(input_dim, self.hidden_dim)
        self.hidden_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, MOTION_DIM)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.register_buffer("action_low", torch.full((MOTION_DIM,), -1.0))
        self.register_buffer("action_high", torch.full((MOTION_DIM,), 1.0))

    def checkpoint_config(self) -> dict[str, int | float]:
        return {
            "model_dim": self.model_dim,
            "hidden_dim": self.hidden_dim,
            "age_dim": self.age_dim,
            "max_xyz_correction": self.max_xyz_correction,
        }

    @classmethod
    def from_checkpoint_config(
        cls, config: Mapping[str, object]
    ) -> "PosteriorDaggerResidualHead":
        if set(config) != _ARCHITECTURE_FIELDS:
            raise ValueError("posterior DAgger checkpoint architecture fields mismatch")
        return cls(
            model_dim=int(config["model_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            age_dim=int(config["age_dim"]),
            max_xyz_correction=float(config["max_xyz_correction"]),
        )

    def _validate_inputs(
        self,
        *,
        base_action: torch.Tensor,
        state_latent: torch.Tensor,
        action_latent: torch.Tensor,
        posterior_token: torch.Tensor,
        missing_age: torch.Tensor,
        sensor_valid: torch.Tensor,
    ) -> int:
        if base_action.ndim != 3:
            raise ValueError("base_action must have shape [B,5,4]")
        batch = int(base_action.shape[0])
        if base_action.shape != (batch, ACTION_STEPS, ACTION_DIM):
            raise ValueError("base_action must have shape [B,5,4]")
        if state_latent.shape != (batch, self.model_dim):
            raise ValueError("state_latent must have shape [B,D]")
        if action_latent.shape != (batch, ACTION_STEPS, self.model_dim):
            raise ValueError("action_latent must have shape [B,5,D]")
        if (
            posterior_token.ndim != 3
            or posterior_token.shape[0] != batch
            or posterior_token.shape[1] <= 0
            or posterior_token.shape[2] != self.model_dim
        ):
            raise ValueError("posterior_token must have shape [B,K,D] with K > 0")
        if missing_age.shape != (batch,) or missing_age.dtype != torch.long:
            raise ValueError("missing_age must be a long tensor with shape [B]")
        if bool(torch.any((missing_age < 0) | (missing_age > MAX_MISSING_AGE))):
            raise ValueError("missing_age is outside the supported range")
        if sensor_valid.shape != (batch,) or sensor_valid.dtype != torch.bool:
            raise ValueError("sensor_valid must be a bool tensor with shape [B]")
        floating = (base_action, state_latent, action_latent, posterior_token)
        if not all(value.is_floating_point() for value in floating):
            raise TypeError("posterior DAgger head inputs must be floating point")
        if not all(bool(torch.isfinite(value).all()) for value in floating):
            raise ValueError("posterior DAgger head inputs contain non-finite values")
        if bool(torch.any(base_action < -1.000001)) or bool(
            torch.any(base_action > 1.000001)
        ):
            raise ValueError("base_action exceeds the environment action bounds")
        return batch

    def forward(
        self,
        *,
        base_action: torch.Tensor,
        state_latent: torch.Tensor,
        action_latent: torch.Tensor,
        posterior_token: torch.Tensor,
        missing_age: torch.Tensor,
        sensor_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch = self._validate_inputs(
            base_action=base_action,
            state_latent=state_latent,
            action_latent=action_latent,
            posterior_token=posterior_token,
            missing_age=missing_age,
            sensor_valid=sensor_valid,
        )
        state = self.state_norm(state_latent).unsqueeze(1).expand(-1, ACTION_STEPS, -1)
        action = self.action_norm(action_latent)
        posterior = self.posterior_norm(posterior_token).mean(dim=1)
        posterior = posterior.unsqueeze(1).expand(-1, ACTION_STEPS, -1)
        age = self.age_embedding(missing_age).unsqueeze(1).expand(-1, ACTION_STEPS, -1)
        steps = torch.arange(ACTION_STEPS, device=base_action.device)
        step = self.step_embedding(steps).unsqueeze(0).expand(batch, -1, -1)
        features = torch.cat(
            (state, action, posterior, base_action, age, step), dim=-1
        )
        hidden = F.gelu(self.input_projection(features))
        hidden = hidden + F.gelu(self.hidden_projection(hidden))
        direction = torch.tanh(self.output_projection(hidden))

        base_motion = base_action[..., :MOTION_DIM]
        low = self.action_low.to(base_action).view(1, 1, MOTION_DIM)
        high = self.action_high.to(base_action).view(1, 1, MOTION_DIM)
        cap = torch.full_like(base_motion, self.max_xyz_correction)
        positive_room = torch.minimum(high - base_motion, cap)
        negative_room = torch.minimum(base_motion - low, cap)
        room = torch.where(direction >= 0, positive_room, negative_room)
        corrected_motion = base_motion + room * direction
        corrected = torch.cat(
            (corrected_motion, base_action[..., MOTION_DIM:]), dim=-1
        )
        return torch.where(sensor_valid[:, None, None], base_action, corrected)


def posterior_dagger_xyz_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float = 0.05,
) -> torch.Tensor:
    """Smooth-L1 imitation loss over XYZ only; gripper is never supervised."""

    if prediction.shape != target.shape or prediction.shape[-2:] != (
        ACTION_STEPS,
        ACTION_DIM,
    ):
        raise ValueError("posterior DAgger loss expects aligned [B,5,4] tensors")
    if not float(beta) > 0.0:
        raise ValueError("posterior DAgger loss beta must be positive")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("posterior DAgger loss inputs must be floating point")
    if not bool(torch.isfinite(prediction).all()) or not bool(
        torch.isfinite(target).all()
    ):
        raise ValueError("posterior DAgger loss inputs must be finite")
    return F.smooth_l1_loss(
        prediction[..., :MOTION_DIM],
        target[..., :MOTION_DIM],
        beta=float(beta),
        reduction="mean",
    )


def save_posterior_dagger_checkpoint(
    path: str | Path,
    model: PosteriorDaggerResidualHead,
    *,
    method: str,
    seed: int,
    steps: int,
    dagger_iteration: int,
) -> None:
    index = method_index(method)
    if min(int(seed), int(steps), int(dagger_iteration)) < 0:
        raise ValueError("checkpoint seed, steps, and DAgger iteration must be non-negative")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "checkpoint_type": CHECKPOINT_TYPE,
            "methods": list(METHODS),
            "method": METHODS[index],
            "method_index": index,
            "seed": int(seed),
            "steps": int(steps),
            "dagger_iteration": int(dagger_iteration),
            "architecture": model.checkpoint_config(),
            "model_state": model.state_dict(),
        },
        target,
    )


def load_posterior_dagger_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    expected_method: str | None = None,
    expected_seed: int | None = None,
    expected_steps: int | None = None,
    expected_dagger_iteration: int | None = None,
) -> tuple[PosteriorDaggerResidualHead, dict[str, object]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("posterior DAgger checkpoint must contain a mapping")
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("posterior DAgger checkpoint schema is unsupported")
    if str(payload.get("protocol", "")) != PROTOCOL:
        raise ValueError("posterior DAgger checkpoint protocol is unsupported")
    if str(payload.get("checkpoint_type", "")) != CHECKPOINT_TYPE:
        raise ValueError("posterior DAgger checkpoint type is unsupported")
    if tuple(payload.get("methods", ())) != METHODS:
        raise ValueError("posterior DAgger checkpoint method ordering drifted")

    method = str(payload.get("method", ""))
    index = int(payload.get("method_index", -1))
    if method not in METHODS or index != method_index(method):
        raise ValueError("posterior DAgger checkpoint method identity is inconsistent")
    seed = int(payload.get("seed", -1))
    steps = int(payload.get("steps", -1))
    iteration = int(payload.get("dagger_iteration", -1))
    if min(seed, steps, iteration) < 0:
        raise ValueError("posterior DAgger checkpoint training metadata is invalid")
    expected = (
        ("method", method, expected_method),
        ("seed", seed, expected_seed),
        ("steps", steps, expected_steps),
        ("DAgger iteration", iteration, expected_dagger_iteration),
    )
    for label, actual, wanted in expected:
        if wanted is not None and actual != wanted:
            raise ValueError(
                f"posterior DAgger checkpoint {label} mismatch: "
                f"{actual!r} != {wanted!r}"
            )

    architecture = payload.get("architecture")
    state = payload.get("model_state")
    if not isinstance(architecture, Mapping) or not isinstance(state, Mapping):
        raise ValueError("posterior DAgger checkpoint architecture is incomplete")
    model = PosteriorDaggerResidualHead.from_checkpoint_config(architecture)
    model.load_state_dict(state, strict=True)
    model.to(device).requires_grad_(False).eval()
    metadata: dict[str, object] = {
        "method": method,
        "method_index": index,
        "seed": seed,
        "steps": steps,
        "dagger_iteration": iteration,
        "architecture": model.checkpoint_config(),
    }
    return model, metadata


__all__ = [
    "ACTION_DIM",
    "ACTION_STEPS",
    "CHECKPOINT_TYPE",
    "MAX_MISSING_AGE",
    "METHODS",
    "MOTION_DIM",
    "PROTOCOL",
    "PosteriorDaggerArrays",
    "PosteriorDaggerResidualHead",
    "load_posterior_dagger_arrays",
    "load_posterior_dagger_checkpoint",
    "method_index",
    "posterior_dagger_xyz_loss",
    "save_posterior_dagger_arrays",
    "save_posterior_dagger_checkpoint",
    "validate_posterior_dagger_arrays",
]
