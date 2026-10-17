"""Deterministic, leakage-isolated training for the LeWM posterior gate.

The kernel consumes precomputed teacher-forced rows.  Deployment features are
rebuilt inside each batch through :func:`build_lewm_posterior_features`; the
canonical state and annotation-derived loss masks are never model inputs.
Validation and test indices are accepted only so their disjointness from the
training indices can be certified.  They are not evaluated and cannot select
a seed, checkpoint, or stopping step.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral
import re
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Sampler

from fcpg.lewm_posterior_adapter import (
    LeWMPosteriorAdapter,
    LeWMPosteriorAdapterOutput,
    MAX_POSTERIOR_ADAPTER_HIDDEN_DIM,
)
from fcpg.lewm_posterior_features import (
    LATENT_INNOVATION_CLIP,
    build_lewm_posterior_features,
)


PosteriorGateMode = Literal["full", "action_blind", "no_lewm"]

POSTERIOR_GATE_TRAINING_CHECKPOINT_TYPE = "fcpg_lewm_posterior_gate_training_v1"
POSTERIOR_GATE_TRAINING_SCHEMA_VERSION = 1
POSTERIOR_GATE_MODES = ("full", "action_blind", "no_lewm")
POSTERIOR_GATE_LOSS_GROUPS = (
    "primary_direct_o0",
    "primary_direct_o1",
    "primary_direct_o2",
    "safety_recovery",
    "safety_indirect",
)
POSTERIOR_GATE_MODEL_INPUTS = (
    "z_measurement",
    "z_prior",
    "innovation",
    "confidence",
    "mask_history",
    "action_summary",
)
POSTERIOR_GATE_FORBIDDEN_MODEL_INPUTS = (
    "canonical_target",
    "q_true",
    "annotation",
    "annotations",
    "event",
    "event_name",
    "task_id",
    "dropout_active",
    "recovery_active",
    "direct_anchor",
    "loss_masks",
)
POSTERIOR_GATE_SELECTION_CONTRACT = "fixed_seed_final_train_endpoint"
DEFAULT_POSTERIOR_GATE_SEED = 12011
DEFAULT_POSTERIOR_GATE_SEEDS = (DEFAULT_POSTERIOR_GATE_SEED,)
HARD_MAX_POSTERIOR_GATE_PARAMETERS = 20_000

_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")


def _strict_integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_float(value: object, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return result


def _mode(value: object) -> PosteriorGateMode:
    if not isinstance(value, str) or value not in POSTERIOR_GATE_MODES:
        raise ValueError(
            f"mode must be one of {POSTERIOR_GATE_MODES}, got {value!r}"
        )
    return value  # type: ignore[return-value]


@dataclass(frozen=True)
class PosteriorGateTrainingConfig:
    """A single pre-registered AdamW endpoint and its fixed budget."""

    mode: PosteriorGateMode = "full"
    seed: int = DEFAULT_POSTERIOR_GATE_SEED
    steps: int = 2_000
    batch_size: int = 512
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1.0e-8
    gradient_clip_norm: float = 1.0
    hidden_dim: int = MAX_POSTERIOR_ADAPTER_HIDDEN_DIM
    max_parameters: int = HARD_MAX_POSTERIOR_GATE_PARAMETERS

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", _mode(self.mode))
        object.__setattr__(
            self, "seed", _strict_integer(self.seed, "seed", minimum=0)
        )
        object.__setattr__(
            self, "steps", _strict_integer(self.steps, "steps", minimum=1)
        )
        object.__setattr__(
            self,
            "batch_size",
            _strict_integer(self.batch_size, "batch_size", minimum=1),
        )
        hidden_dim = _strict_integer(self.hidden_dim, "hidden_dim", minimum=0)
        if hidden_dim > MAX_POSTERIOR_ADAPTER_HIDDEN_DIM:
            raise ValueError(
                f"hidden_dim must be at most {MAX_POSTERIOR_ADAPTER_HIDDEN_DIM}"
            )
        object.__setattr__(self, "hidden_dim", hidden_dim)
        budget = _strict_integer(
            self.max_parameters, "max_parameters", minimum=1
        )
        if budget > HARD_MAX_POSTERIOR_GATE_PARAMETERS:
            raise ValueError(
                "max_parameters cannot exceed the 20000-parameter protocol budget"
            )
        object.__setattr__(self, "max_parameters", budget)

        learning_rate = _finite_float(self.learning_rate, "learning_rate")
        if learning_rate == 0.0:
            raise ValueError("learning_rate must be positive")
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(
            self,
            "weight_decay",
            _finite_float(self.weight_decay, "weight_decay"),
        )
        if not isinstance(self.adam_betas, tuple) or len(self.adam_betas) != 2:
            raise TypeError("adam_betas must be a two-item tuple")
        betas = tuple(
            _finite_float(value, f"adam_betas[{index}]")
            for index, value in enumerate(self.adam_betas)
        )
        if any(value >= 1.0 for value in betas):
            raise ValueError("adam_betas values must lie in [0, 1)")
        object.__setattr__(self, "adam_betas", betas)
        eps = _finite_float(self.adam_eps, "adam_eps")
        if eps == 0.0:
            raise ValueError("adam_eps must be positive")
        object.__setattr__(self, "adam_eps", eps)
        clip = _finite_float(self.gradient_clip_norm, "gradient_clip_norm")
        if clip == 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        object.__setattr__(self, "gradient_clip_norm", clip)

    def to_checkpoint_config(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "seed": self.seed,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "optimizer": "AdamW",
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "adam_betas": list(self.adam_betas),
            "adam_eps": self.adam_eps,
            "gradient_clip_norm": self.gradient_clip_norm,
            "hidden_dim": self.hidden_dim,
            "max_parameters": self.max_parameters,
        }

    @classmethod
    def from_checkpoint_config(
        cls, value: Mapping[str, object]
    ) -> "PosteriorGateTrainingConfig":
        if not isinstance(value, Mapping):
            raise TypeError("training config must be a mapping")
        expected = {
            "mode",
            "seed",
            "steps",
            "batch_size",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "adam_betas",
            "adam_eps",
            "gradient_clip_norm",
            "hidden_dim",
            "max_parameters",
        }
        if set(value) != expected:
            raise ValueError("posterior gate training config fields changed")
        if value["optimizer"] != "AdamW":
            raise ValueError("posterior gate optimizer must be AdamW")
        raw_betas = value["adam_betas"]
        if (
            isinstance(raw_betas, (str, bytes))
            or not isinstance(raw_betas, Sequence)
            or len(raw_betas) != 2
        ):
            raise TypeError("checkpoint adam_betas must be a pair")
        return cls(
            mode=value["mode"],  # type: ignore[arg-type]
            seed=value["seed"],  # type: ignore[arg-type]
            steps=value["steps"],  # type: ignore[arg-type]
            batch_size=value["batch_size"],  # type: ignore[arg-type]
            learning_rate=value["learning_rate"],  # type: ignore[arg-type]
            weight_decay=value["weight_decay"],  # type: ignore[arg-type]
            adam_betas=(raw_betas[0], raw_betas[1]),  # type: ignore[arg-type]
            adam_eps=value["adam_eps"],  # type: ignore[arg-type]
            gradient_clip_norm=value["gradient_clip_norm"],  # type: ignore[arg-type]
            hidden_dim=value["hidden_dim"],  # type: ignore[arg-type]
            max_parameters=value["max_parameters"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class PosteriorGateLossMasks:
    """Offline-only masks for the five equally weighted loss groups."""

    primary_direct_o0: Tensor
    primary_direct_o1: Tensor
    primary_direct_o2: Tensor
    safety_recovery: Tensor
    safety_indirect: Tensor

    def as_mapping(self) -> dict[str, Tensor]:
        return {
            name: getattr(self, name) for name in POSTERIOR_GATE_LOSS_GROUPS
        }


@dataclass(frozen=True)
class PosteriorGateTrainingRows:
    """Precomputed teacher-forced rows with supervision kept out of features."""

    z_measurement: Tensor
    z_prior: Tensor
    z_hold_last: Tensor
    canonical_target: Tensor
    q_nominal: Tensor
    q_corrected: Tensor
    valid_nominal: Tensor
    validity_history: Tensor
    last_causal_action_chunk: Tensor
    loss_masks: PosteriorGateLossMasks | Mapping[str, Tensor]
    source_hashes: Mapping[str, str]
    image_width: int = 224
    image_height: int = 224

    @property
    def sample_count(self) -> int:
        if not torch.is_tensor(self.z_measurement) or self.z_measurement.ndim == 0:
            return 0
        return int(self.z_measurement.shape[0])

    @property
    def measurement(self) -> Tensor:
        return self.z_measurement

    @property
    def prior(self) -> Tensor:
        return self.z_prior

    @property
    def target(self) -> Tensor:
        return self.canonical_target

    @property
    def online_validity_history(self) -> Tensor:
        return self.validity_history

    @property
    def action_summary(self) -> Tensor:
        return self.last_causal_action_chunk


# Retain the conventional short name for call sites that treat rows as data.
PosteriorGateTrainingData = PosteriorGateTrainingRows


def _normalize_indices(value: object, name: str) -> tuple[int, ...]:
    if torch.is_tensor(value):
        if value.ndim != 1 or value.dtype == torch.bool:
            raise TypeError(f"{name} must be a one-dimensional integer tensor")
        if value.is_floating_point() or value.is_complex():
            raise TypeError(f"{name} must contain integers")
        raw = value.detach().cpu().tolist()
    else:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"{name} must be an integer sequence")
        raw = list(value)
    indices = tuple(
        _strict_integer(item, f"{name}[{position}]", minimum=0)
        for position, item in enumerate(raw)
    )
    if len(indices) != len(set(indices)):
        raise ValueError(f"{name} contains duplicate indices")
    return tuple(sorted(indices))


def _batch_indices(value: object, name: str) -> tuple[int, ...]:
    """Normalize a sampled batch while preserving order and replacement."""

    if torch.is_tensor(value):
        if value.ndim != 1 or value.dtype == torch.bool:
            raise TypeError(f"{name} must be a one-dimensional integer tensor")
        if value.is_floating_point() or value.is_complex():
            raise TypeError(f"{name} must contain integers")
        raw = value.detach().cpu().tolist()
    else:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"{name} must be an integer sequence")
        raw = list(value)
    return tuple(
        _strict_integer(item, f"{name}[{position}]", minimum=0)
        for position, item in enumerate(raw)
    )


@dataclass(frozen=True)
class PosteriorGateSplitIndices:
    """Explicit split membership; only ``train`` may be dereferenced."""

    train: Sequence[int] | Tensor
    validation: Sequence[int] | Tensor = ()
    test: Sequence[int] | Tensor = ()

    def __post_init__(self) -> None:
        train = _normalize_indices(self.train, "train")
        validation = _normalize_indices(self.validation, "validation")
        test = _normalize_indices(self.test, "test")
        if not train:
            raise ValueError("train indices must not be empty")
        memberships = {
            "train": set(train),
            "validation": set(validation),
            "test": set(test),
        }
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            overlap = memberships[left].intersection(memberships[right])
            if overlap:
                raise ValueError(
                    f"{left} and {right} indices overlap: {sorted(overlap)}"
                )
        object.__setattr__(self, "train", train)
        object.__setattr__(self, "validation", validation)
        object.__setattr__(self, "test", test)

    @property
    def sealed_test(self) -> tuple[int, ...]:
        return tuple(self.test)


@dataclass(frozen=True)
class PosteriorGateForwardInputs:
    """The exact six tensors that may cross the adapter boundary."""

    z_measurement: Tensor
    z_prior: Tensor
    innovation: Tensor
    confidence: Tensor
    mask_history: Tensor
    action_summary: Tensor

    def as_args(self) -> tuple[Tensor, ...]:
        return tuple(getattr(self, name) for name in POSTERIOR_GATE_MODEL_INPUTS)


@dataclass(frozen=True)
class PosteriorGateSeedResult:
    """The sole fixed-seed, fixed-step endpoint."""

    seed: int
    optimizer_steps: int
    train_loss: float
    initial_state_sha256: str
    adapter_state_sha256: str


@dataclass(frozen=True)
class PosteriorGateTrainingResult:
    adapter: LeWMPosteriorAdapter
    config: PosteriorGateTrainingConfig
    seed: int
    step: int
    train_loss: float
    parameter_count: int
    initial_state_sha256: str
    adapter_state_sha256: str
    sampled_train_indices: tuple[int, ...]
    sampled_index_trace_sha256: str
    group_sample_counts: Mapping[str, int]
    checkpoint_payload: Mapping[str, Any]

    @property
    def checkpoint(self) -> Mapping[str, Any]:
        return self.checkpoint_payload

    @property
    def selected_seed(self) -> int:
        return self.seed

    @property
    def optimizer_steps(self) -> int:
        return self.step

    @property
    def seed_results(self) -> tuple[PosteriorGateSeedResult, ...]:
        return (
            PosteriorGateSeedResult(
                seed=self.seed,
                optimizer_steps=self.step,
                train_loss=self.train_loss,
                initial_state_sha256=self.initial_state_sha256,
                adapter_state_sha256=self.adapter_state_sha256,
            ),
        )

    @property
    def selected_result(self) -> PosteriorGateSeedResult:
        return self.seed_results[0]


def _plain_metadata(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_metadata(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata cannot contain non-finite floats")
        return value
    raise TypeError("metadata must contain only JSON-compatible values")


def canonical_metadata_sha256(value: object) -> str:
    encoded = json.dumps(
        _plain_metadata(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_sha256(value: Tensor) -> str:
    if not torch.is_tensor(value):
        raise TypeError("tensor_sha256 expects a tensor")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _cpu_state_dict(value: nn.Module | Mapping[str, Tensor]) -> dict[str, Tensor]:
    source = value.state_dict() if isinstance(value, nn.Module) else value
    state: dict[str, Tensor] = {}
    for name, tensor in sorted(source.items()):
        if not isinstance(name, str) or not torch.is_tensor(tensor):
            raise TypeError("adapter state must map strings to tensors")
        state[name] = tensor.detach().cpu().contiguous().clone()
    return state


def posterior_adapter_state_sha256(
    value: nn.Module | Mapping[str, Tensor],
) -> str:
    digest = hashlib.sha256()
    for name, tensor in _cpu_state_dict(value).items():
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor_sha256(tensor).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def posterior_adapter_parameter_count(adapter: nn.Module) -> int:
    return sum(parameter.numel() for parameter in adapter.parameters())


def _normalized_source_hashes(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("source_hashes must be a non-empty mapping")
    result: dict[str, str] = {}
    for raw_name, raw_hash in sorted(value.items()):
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError("source hash names must be non-empty strings")
        if not isinstance(raw_hash, str) or _SHA256_RE.fullmatch(raw_hash) is None:
            raise ValueError(f"source_hashes[{raw_name!r}] must be a SHA-256 hex digest")
        result[raw_name] = raw_hash.lower()
    return result


def _loss_masks(
    value: PosteriorGateLossMasks | Mapping[str, Tensor],
) -> dict[str, Tensor]:
    if isinstance(value, PosteriorGateLossMasks):
        return value.as_mapping()
    if not isinstance(value, Mapping) or set(value) != set(POSTERIOR_GATE_LOSS_GROUPS):
        raise ValueError(
            "loss_masks must contain exactly " f"{POSTERIOR_GATE_LOSS_GROUPS}"
        )
    return {name: value[name] for name in POSTERIOR_GATE_LOSS_GROUPS}


def _index_tensor(indices: Sequence[int], device: torch.device) -> Tensor:
    return torch.tensor(tuple(indices), dtype=torch.long, device=device)


def _take(value: Tensor, indices: Tensor, device: torch.device) -> Tensor:
    selected = value.detach().index_select(0, indices.to(device=value.device))
    return selected.to(device=device)


def validate_posterior_gate_training_contract(
    rows: PosteriorGateTrainingRows,
    split_indices: PosteriorGateSplitIndices,
) -> dict[str, Tensor]:
    """Validate shapes and only dereference values selected for training."""

    if not isinstance(rows, PosteriorGateTrainingRows):
        raise TypeError("rows must be PosteriorGateTrainingRows")
    if not isinstance(split_indices, PosteriorGateSplitIndices):
        raise TypeError("split_indices must be PosteriorGateSplitIndices")
    if rows.sample_count <= 0:
        raise ValueError("posterior gate rows must not be empty")
    for split_name in ("train", "validation", "test"):
        values = getattr(split_indices, split_name)
        if values and max(values) >= rows.sample_count:
            raise IndexError(f"{split_name} index is outside the row table")
    _normalized_source_hashes(rows.source_hashes)
    width = _strict_integer(rows.image_width, "image_width", minimum=2)
    height = _strict_integer(rows.image_height, "image_height", minimum=2)
    del width, height

    tensor_fields = {
        "z_measurement": rows.z_measurement,
        "z_prior": rows.z_prior,
        "z_hold_last": rows.z_hold_last,
        "canonical_target": rows.canonical_target,
        "q_nominal": rows.q_nominal,
        "q_corrected": rows.q_corrected,
        "valid_nominal": rows.valid_nominal,
        "validity_history": rows.validity_history,
        "last_causal_action_chunk": rows.last_causal_action_chunk,
    }
    for name, value in tensor_fields.items():
        if not torch.is_tensor(value):
            raise TypeError(f"rows.{name} must be a tensor")
    measurement = rows.z_measurement
    if measurement.ndim not in (3, 4) or any(size <= 0 for size in measurement.shape):
        raise ValueError("z_measurement must be nonempty [N,K,D] or [N,T,K,D]")
    if not measurement.is_floating_point():
        raise TypeError("z_measurement must have a floating dtype")
    for name in ("z_prior", "z_hold_last", "canonical_target"):
        value = tensor_fields[name]
        if value.shape != measurement.shape:
            raise ValueError(f"{name} must match z_measurement shape")
        if value.dtype != measurement.dtype or value.device != measurement.device:
            raise TypeError(f"{name} must share z_measurement dtype and device")

    coordinate_shape = (*measurement.shape[:-1], 2)
    for name in ("q_nominal", "q_corrected"):
        value = tensor_fields[name]
        if value.shape != coordinate_shape:
            raise ValueError(f"{name} must have shape {coordinate_shape}")
        if value.dtype != measurement.dtype or value.device != measurement.device:
            raise TypeError(f"{name} must share z_measurement dtype and device")
    if rows.valid_nominal.shape != measurement.shape[:-1]:
        raise ValueError("valid_nominal must match row/time/anchor axes")
    if rows.valid_nominal.dtype != torch.bool:
        raise TypeError("valid_nominal must have boolean dtype")
    if rows.valid_nominal.device != measurement.device:
        raise ValueError("valid_nominal must share z_measurement device")
    if (
        rows.validity_history.ndim != measurement.ndim
        or rows.validity_history.shape[:-1] != measurement.shape[:-1]
        or rows.validity_history.shape[-1] <= 0
    ):
        raise ValueError("validity_history must match row/time/anchor axes plus H")
    if rows.validity_history.dtype != torch.bool:
        raise TypeError("validity_history must have boolean dtype")
    if rows.validity_history.device != measurement.device:
        raise ValueError("validity_history must share z_measurement device")
    action_shape = measurement.shape[:-2]
    action = rows.last_causal_action_chunk
    if (
        action.ndim != len(action_shape) + 1
        or action.shape[:-1] != action_shape
        or action.shape[-1] <= 0
    ):
        raise ValueError("last_causal_action_chunk must match row/time axes plus A")
    if action.dtype != measurement.dtype or action.device != measurement.device:
        raise TypeError(
            "last_causal_action_chunk must share z_measurement dtype and device"
        )

    masks = _loss_masks(rows.loss_masks)
    for name, mask in masks.items():
        if not torch.is_tensor(mask):
            raise TypeError(f"loss mask {name!r} must be a tensor")
        if mask.shape != measurement.shape[:-1]:
            raise ValueError(
                f"loss mask {name!r} must have shape {measurement.shape[:-1]}"
            )
        if mask.dtype != torch.bool:
            raise TypeError(f"loss mask {name!r} must have boolean dtype")
        if mask.device != measurement.device:
            raise ValueError(f"loss mask {name!r} must share measurement device")

    train_index = _index_tensor(split_indices.train, measurement.device)
    finite_fields = (
        rows.z_measurement,
        rows.z_prior,
        rows.z_hold_last,
        rows.canonical_target,
        rows.q_nominal,
        rows.q_corrected,
        rows.last_causal_action_chunk,
    )
    if any(
        not bool(value.index_select(0, train_index).isfinite().all())
        for value in finite_fields
    ):
        raise ValueError("training rows must contain only finite values")
    for name, mask in masks.items():
        if not bool(mask.index_select(0, train_index).any()):
            raise ValueError(f"training loss group {name!r} is empty")
    return masks


def build_posterior_gate_forward_inputs(
    rows: PosteriorGateTrainingRows,
    indices: Tensor | Sequence[int],
    *,
    mode: PosteriorGateMode = "full",
    device: torch.device | str = "cpu",
) -> PosteriorGateForwardInputs:
    """Build one batch without exposing supervision or annotations."""

    actual_mode = _mode(mode)
    target_device = torch.device(device)
    raw_indices = _batch_indices(indices, "batch_indices")
    if not raw_indices:
        raise ValueError("batch_indices must not be empty")
    if max(raw_indices) >= rows.sample_count:
        raise IndexError("batch index is outside the row table")
    source_index = _index_tensor(raw_indices, rows.z_measurement.device)
    measurement = _take(rows.z_measurement, source_index, target_device)
    if actual_mode == "no_lewm":
        prior = _take(rows.z_hold_last, source_index, target_device)
    else:
        prior = _take(rows.z_prior, source_index, target_device)
    action = _take(rows.last_causal_action_chunk, source_index, target_device)
    if actual_mode == "action_blind":
        action = torch.zeros_like(action)
    features = build_lewm_posterior_features(
        z_measurement=measurement,
        z_prior=prior,
        q_nominal=_take(rows.q_nominal, source_index, target_device),
        q_corrected=_take(rows.q_corrected, source_index, target_device),
        valid_nominal=_take(rows.valid_nominal, source_index, target_device),
        image_width=rows.image_width,
        image_height=rows.image_height,
        online_validity_history=_take(
            rows.validity_history, source_index, target_device
        ),
        causal_action_summary=action,
    )
    return PosteriorGateForwardInputs(
        z_measurement=measurement,
        z_prior=prior,
        innovation=features.innovation,
        confidence=features.confidence,
        mask_history=features.mask_history,
        action_summary=features.action_summary,
    )


def forward_posterior_gate(
    adapter: LeWMPosteriorAdapter,
    rows: PosteriorGateTrainingRows,
    indices: Tensor | Sequence[int],
    *,
    mode: PosteriorGateMode = "full",
    device: torch.device | str | None = None,
) -> LeWMPosteriorAdapterOutput:
    """Forward only the allowlisted deployment tensors through ``adapter``."""

    if not isinstance(adapter, LeWMPosteriorAdapter):
        raise TypeError("adapter must be LeWMPosteriorAdapter")
    if device is None:
        parameter = next(adapter.parameters())
        target_device = parameter.device
    else:
        target_device = torch.device(device)
    inputs = build_posterior_gate_forward_inputs(
        rows, indices, mode=mode, device=target_device
    )
    return adapter(*inputs.as_args())


def posterior_adapter_architecture(
    rows: PosteriorGateTrainingRows,
    *,
    hidden_dim: int = MAX_POSTERIOR_ADAPTER_HIDDEN_DIM,
) -> dict[str, int]:
    """Return the mode-invariant adapter architecture for these rows."""

    return {
        "latent_dim": int(rows.z_measurement.shape[-1]),
        "innovation_dim": int(rows.z_measurement.shape[-1]),
        "confidence_dim": 6,
        "mask_history_dim": int(rows.validity_history.shape[-1]),
        "action_summary_dim": int(rows.last_causal_action_chunk.shape[-1]),
        "hidden_dim": _strict_integer(hidden_dim, "hidden_dim", minimum=0),
    }


def balanced_posterior_gate_loss(
    prediction: Tensor,
    canonical_target: Tensor,
    loss_masks: PosteriorGateLossMasks | Mapping[str, Tensor],
) -> Tensor:
    """Average the five masked MSE group means with equal group weight."""

    if prediction.shape != canonical_target.shape or prediction.ndim < 3:
        raise ValueError("prediction and canonical_target must match [...,K,D]")
    if prediction.dtype != canonical_target.dtype:
        raise TypeError("prediction and canonical_target dtypes must match")
    per_position = torch.square(prediction - canonical_target).mean(dim=-1)
    group_losses: list[Tensor] = []
    for name, mask in _loss_masks(loss_masks).items():
        if mask.shape != per_position.shape or mask.dtype != torch.bool:
            raise ValueError(f"loss mask {name!r} must be boolean {per_position.shape}")
        if mask.device != prediction.device:
            raise ValueError(f"loss mask {name!r} must share prediction device")
        if not bool(mask.any()):
            raise ValueError(f"loss group {name!r} is empty")
        group_losses.append(per_position.masked_select(mask).mean())
    loss = torch.stack(group_losses).mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("posterior gate loss became non-finite")
    return loss


class _LossElementDataset(Dataset[dict[str, int]]):
    def __init__(
        self,
        masks: Mapping[str, Tensor],
        train_indices: Sequence[int],
    ) -> None:
        train = _index_tensor(train_indices, next(iter(masks.values())).device)
        row_indices: list[int] = []
        position_indices: list[int] = []
        group_indices: list[int] = []
        group_tokens: list[Tensor] = []
        for group_index, name in enumerate(POSTERIOR_GATE_LOSS_GROUPS):
            selected = masks[name].index_select(0, train).reshape(len(train_indices), -1)
            nonzero = selected.nonzero(as_tuple=False).detach().cpu()
            start = len(row_indices)
            for local_row, flat_position in nonzero.tolist():
                row_indices.append(int(train_indices[local_row]))
                position_indices.append(int(flat_position))
                group_indices.append(group_index)
            stop = len(row_indices)
            group_tokens.append(torch.arange(start, stop, dtype=torch.long))
        self.row_indices = tuple(row_indices)
        self.position_indices = tuple(position_indices)
        self.group_indices = tuple(group_indices)
        self.group_tokens = tuple(group_tokens)

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, index: int) -> dict[str, int]:
        return {
            "row_index": self.row_indices[index],
            "position_index": self.position_indices[index],
            "group_index": self.group_indices[index],
        }


class _DeterministicBalancedSampler(Sampler[int]):
    def __init__(
        self,
        group_tokens: Sequence[Tensor],
        *,
        sample_count: int,
        seed: int,
    ) -> None:
        self._groups = tuple(group.detach().cpu() for group in group_tokens)
        self._sample_count = sample_count
        self._seed = seed
        if any(group.numel() == 0 for group in self._groups):
            raise ValueError("balanced sampler received an empty loss group")

    def __len__(self) -> int:
        return self._sample_count

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator(device="cpu").manual_seed(self._seed)
        for position in range(self._sample_count):
            group = self._groups[position % len(self._groups)]
            offset = int(
                torch.randint(group.numel(), (1,), generator=generator).item()
            )
            yield int(group[offset].item())


def build_posterior_gate_train_loader(
    rows: PosteriorGateTrainingRows,
    split_indices: PosteriorGateSplitIndices,
    *,
    config: PosteriorGateTrainingConfig,
) -> DataLoader[dict[str, Tensor]]:
    """Return a deterministic, group-balanced, train-only fixed-budget loader."""

    if not isinstance(config, PosteriorGateTrainingConfig):
        raise TypeError("config must be PosteriorGateTrainingConfig")
    masks = validate_posterior_gate_training_contract(rows, split_indices)
    dataset = _LossElementDataset(masks, split_indices.train)
    sampler = _DeterministicBalancedSampler(
        dataset.group_tokens,
        sample_count=config.steps * config.batch_size,
        seed=config.seed,
    )
    loader_generator = torch.Generator(device="cpu").manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=0,
        generator=loader_generator,
    )


def _initial_identity_check(
    adapter: LeWMPosteriorAdapter,
    rows: PosteriorGateTrainingRows,
    indices: Sequence[int],
    *,
    mode: PosteriorGateMode,
    device: torch.device,
) -> None:
    probe_indices = tuple(indices[: min(2, len(indices))])
    inputs = build_posterior_gate_forward_inputs(
        rows, probe_indices, mode=mode, device=device
    )
    with torch.no_grad():
        output = adapter(*inputs.as_args())
    if torch.count_nonzero(adapter.output_layer.weight).item() != 0:
        raise RuntimeError("posterior adapter output weights are not zero-initialized")
    if torch.count_nonzero(adapter.output_layer.bias).item() != 0:
        raise RuntimeError("posterior adapter output bias is not zero-initialized")
    if torch.count_nonzero(output.beta).item() != 0 or not torch.equal(
        output.z_post, inputs.z_measurement
    ):
        raise RuntimeError("zero-initialized posterior adapter is not identity")


def _full_train_loss(
    adapter: LeWMPosteriorAdapter,
    rows: PosteriorGateTrainingRows,
    split_indices: PosteriorGateSplitIndices,
    masks: Mapping[str, Tensor],
    *,
    config: PosteriorGateTrainingConfig,
    device: torch.device,
) -> float:
    sums = {name: 0.0 for name in POSTERIOR_GATE_LOSS_GROUPS}
    counts = {name: 0 for name in POSTERIOR_GATE_LOSS_GROUPS}
    adapter.eval()
    with torch.no_grad():
        train = tuple(split_indices.train)
        for start in range(0, len(train), config.batch_size):
            batch_indices = train[start : start + config.batch_size]
            inputs = build_posterior_gate_forward_inputs(
                rows, batch_indices, mode=config.mode, device=device
            )
            output = adapter(*inputs.as_args())
            source = _index_tensor(batch_indices, rows.canonical_target.device)
            canonical = _take(rows.canonical_target, source, device)
            error = torch.square(output.z_post - canonical).mean(dim=-1).double()
            for name in POSTERIOR_GATE_LOSS_GROUPS:
                mask = _take(masks[name], source, device)
                sums[name] += float(error.masked_select(mask).sum().cpu())
                counts[name] += int(mask.sum().cpu())
    return sum(sums[name] / counts[name] for name in POSTERIOR_GATE_LOSS_GROUPS) / len(
        POSTERIOR_GATE_LOSS_GROUPS
    )


def _indices_sha256(values: Sequence[int]) -> str:
    return tensor_sha256(torch.tensor(tuple(values), dtype=torch.int64))


def _checkpoint_integrity_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_type": payload["checkpoint_type"],
        "schema_version": payload["schema_version"],
        "config": payload["config"],
        "state_sha256": payload["state_sha256"],
        "seed": payload["seed"],
        "step": payload["step"],
        "source_hashes": payload["source_hashes"],
        "parameter_count": payload["parameter_count"],
        "sampling_sha256": payload["sampling_sha256"],
        "split_contract": payload["split_contract"],
        "model_inputs": payload["model_inputs"],
        "loss_groups": payload["loss_groups"],
    }


def _build_checkpoint(
    adapter: LeWMPosteriorAdapter,
    rows: PosteriorGateTrainingRows,
    split_indices: PosteriorGateSplitIndices,
    config: PosteriorGateTrainingConfig,
    *,
    parameter_count: int,
    sampling_sha256: str,
    group_sample_counts: Mapping[str, int],
) -> dict[str, Any]:
    state = _cpu_state_dict(adapter)
    payload: dict[str, Any] = {
        "checkpoint_type": POSTERIOR_GATE_TRAINING_CHECKPOINT_TYPE,
        "schema_version": POSTERIOR_GATE_TRAINING_SCHEMA_VERSION,
        "config": {
            "training": config.to_checkpoint_config(),
            "adapter": adapter.checkpoint_config(),
        },
        "state": state,
        "state_sha256": posterior_adapter_state_sha256(state),
        "seed": config.seed,
        "step": config.steps,
        "source_hashes": _normalized_source_hashes(rows.source_hashes),
        "parameter_count": parameter_count,
        "sampling_sha256": sampling_sha256,
        "split_contract": {
            "selection": POSTERIOR_GATE_SELECTION_CONTRACT,
            "optimizer_split": "train",
            "validation_used_for_optimization": False,
            "validation_used_for_seed_selection": False,
            "test_used_for_optimization": False,
            "test_used_for_seed_selection": False,
            "train_indices_sha256": _indices_sha256(split_indices.train),
            "validation_indices_sha256": _indices_sha256(split_indices.validation),
            "test_indices_sha256": _indices_sha256(split_indices.test),
            "train_count": len(split_indices.train),
            "validation_count": len(split_indices.validation),
            "test_count": len(split_indices.test),
            "group_sample_counts": dict(group_sample_counts),
        },
        "model_inputs": list(POSTERIOR_GATE_MODEL_INPUTS),
        "loss_groups": list(POSTERIOR_GATE_LOSS_GROUPS),
    }
    payload["integrity_sha256"] = canonical_metadata_sha256(
        _checkpoint_integrity_fields(payload)
    )
    return payload


def load_posterior_gate_checkpoint(
    payload: Mapping[str, Any],
    *,
    expected_source_hashes: Mapping[str, str] | None = None,
) -> LeWMPosteriorAdapter:
    """Validate a checkpoint and reconstruct its adapter without RNG effects."""

    if not isinstance(payload, Mapping):
        raise TypeError("posterior gate checkpoint must be a mapping")
    expected_fields = {
        "checkpoint_type",
        "schema_version",
        "config",
        "state",
        "state_sha256",
        "seed",
        "step",
        "source_hashes",
        "parameter_count",
        "sampling_sha256",
        "split_contract",
        "model_inputs",
        "loss_groups",
        "integrity_sha256",
    }
    if set(payload) != expected_fields:
        raise ValueError("posterior gate checkpoint fields changed")
    if payload["checkpoint_type"] != POSTERIOR_GATE_TRAINING_CHECKPOINT_TYPE:
        raise ValueError("posterior gate checkpoint_type mismatch")
    if payload["schema_version"] != POSTERIOR_GATE_TRAINING_SCHEMA_VERSION:
        raise ValueError("posterior gate checkpoint schema_version mismatch")
    if payload["model_inputs"] != list(POSTERIOR_GATE_MODEL_INPUTS):
        raise ValueError("posterior gate model input contract changed")
    if payload["loss_groups"] != list(POSTERIOR_GATE_LOSS_GROUPS):
        raise ValueError("posterior gate loss contract changed")
    if payload["integrity_sha256"] != canonical_metadata_sha256(
        _checkpoint_integrity_fields(payload)
    ):
        raise ValueError("posterior gate checkpoint integrity mismatch")

    source_hashes = _normalized_source_hashes(payload["source_hashes"])
    if expected_source_hashes is not None and source_hashes != _normalized_source_hashes(
        expected_source_hashes
    ):
        raise ValueError("posterior gate source hashes mismatch")
    config_value = payload["config"]
    if not isinstance(config_value, Mapping) or set(config_value) != {
        "training",
        "adapter",
    }:
        raise ValueError("posterior gate checkpoint config is invalid")
    training_config = PosteriorGateTrainingConfig.from_checkpoint_config(
        config_value["training"]
    )
    if payload["seed"] != training_config.seed or payload["step"] != training_config.steps:
        raise ValueError("posterior gate seed/step binding mismatch")
    split_contract = payload["split_contract"]
    if not isinstance(split_contract, Mapping) or any(
        split_contract.get(name) is not False
        for name in (
            "validation_used_for_optimization",
            "validation_used_for_seed_selection",
            "test_used_for_optimization",
            "test_used_for_seed_selection",
        )
    ):
        raise ValueError("posterior gate split contract is unsafe")
    if split_contract.get("selection") != POSTERIOR_GATE_SELECTION_CONTRACT:
        raise ValueError("posterior gate seed selection contract changed")

    state = payload["state"]
    if not isinstance(state, Mapping):
        raise TypeError("posterior gate state must be a mapping")
    state_hash = posterior_adapter_state_sha256(state)  # type: ignore[arg-type]
    if payload["state_sha256"] != state_hash:
        raise ValueError("posterior gate state hash mismatch")
    state_tensors = list(state.values())
    if not state_tensors or any(not torch.is_tensor(value) for value in state_tensors):
        raise TypeError("posterior gate state contains a non-tensor")
    dtype = state_tensors[0].dtype
    if not dtype.is_floating_point or any(value.dtype != dtype for value in state_tensors):
        raise TypeError("posterior gate state must use one floating dtype")
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(0)
        adapter = LeWMPosteriorAdapter.from_checkpoint_config(
            config_value["adapter"]
        ).to(dtype=dtype)
    adapter.load_state_dict(state, strict=True)
    count = posterior_adapter_parameter_count(adapter)
    if count != payload["parameter_count"] or count > HARD_MAX_POSTERIOR_GATE_PARAMETERS:
        raise ValueError("posterior gate checkpoint parameter budget mismatch")
    if posterior_adapter_state_sha256(adapter) != state_hash:
        raise ValueError("posterior gate state reconstruction mismatch")
    adapter.eval()
    return adapter


def train_posterior_gate(
    rows: PosteriorGateTrainingRows,
    split_indices: PosteriorGateSplitIndices,
    *,
    config: PosteriorGateTrainingConfig | None = None,
    device: torch.device | str = "cpu",
) -> PosteriorGateTrainingResult:
    """Train exactly one fixed seed for exactly ``steps`` train-only updates."""

    actual_config = config or PosteriorGateTrainingConfig()
    if not isinstance(actual_config, PosteriorGateTrainingConfig):
        raise TypeError("config must be PosteriorGateTrainingConfig")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA training requested but CUDA is unavailable")
    masks = validate_posterior_gate_training_contract(rows, split_indices)
    architecture = posterior_adapter_architecture(
        rows, hidden_dim=actual_config.hidden_dim
    )
    cuda_devices: list[int] = []
    if target_device.type == "cuda":
        cuda_devices = [
            torch.cuda.current_device()
            if target_device.index is None
            else target_device.index
        ]

    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(actual_config.seed)
        if target_device.type == "cuda":
            torch.cuda.manual_seed_all(actual_config.seed)
        adapter = LeWMPosteriorAdapter(**architecture).to(
            device=target_device, dtype=rows.z_measurement.dtype
        )
        parameter_count = posterior_adapter_parameter_count(adapter)
        if (
            parameter_count > actual_config.max_parameters
            or parameter_count > HARD_MAX_POSTERIOR_GATE_PARAMETERS
        ):
            raise ValueError(
                "posterior adapter parameter budget exceeded: "
                f"{parameter_count} > {actual_config.max_parameters}"
            )
        _initial_identity_check(
            adapter,
            rows,
            split_indices.train,
            mode=actual_config.mode,
            device=target_device,
        )
        initial_hash = posterior_adapter_state_sha256(adapter)
        optimizer = torch.optim.AdamW(
            adapter.parameters(),
            lr=actual_config.learning_rate,
            weight_decay=actual_config.weight_decay,
            betas=actual_config.adam_betas,
            eps=actual_config.adam_eps,
        )
        loader = build_posterior_gate_train_loader(
            rows, split_indices, config=actual_config
        )
        trace = hashlib.sha256()
        sampled_rows: set[int] = set()
        group_counts = {name: 0 for name in POSTERIOR_GATE_LOSS_GROUPS}
        completed_steps = 0
        adapter.train()
        for batch in loader:
            row_index = batch["row_index"].to(dtype=torch.long)
            position_index = batch["position_index"].to(
                device=target_device, dtype=torch.long
            )
            group_index = batch["group_index"].to(dtype=torch.long)
            if row_index.numel() != actual_config.batch_size:
                raise RuntimeError("posterior gate loader emitted a partial batch")
            trace.update(row_index.contiguous().numpy().tobytes())
            sampled_rows.update(int(value) for value in row_index.tolist())
            for index, name in enumerate(POSTERIOR_GATE_LOSS_GROUPS):
                group_counts[name] += int((group_index == index).sum())

            inputs = build_posterior_gate_forward_inputs(
                rows,
                row_index,
                mode=actual_config.mode,
                device=target_device,
            )
            output = adapter(*inputs.as_args())
            source = row_index.to(device=rows.canonical_target.device)
            canonical = _take(rows.canonical_target, source, target_device)
            per_position = torch.square(output.z_post - canonical).mean(dim=-1)
            selected_loss = per_position.reshape(per_position.shape[0], -1).gather(
                1, position_index[:, None]
            )
            loss = selected_loss.mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("posterior gate loss became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                adapter.parameters(), actual_config.gradient_clip_norm
            )
            optimizer.step()
            completed_steps += 1
        if completed_steps != actual_config.steps:
            raise RuntimeError(
                "posterior gate fixed-step budget was not completed: "
                f"{completed_steps}/{actual_config.steps}"
            )
        train_loss = _full_train_loss(
            adapter,
            rows,
            split_indices,
            masks,
            config=actual_config,
            device=target_device,
        )
        sampling_hash = trace.hexdigest()
        checkpoint = _build_checkpoint(
            adapter,
            rows,
            split_indices,
            actual_config,
            parameter_count=parameter_count,
            sampling_sha256=sampling_hash,
            group_sample_counts=group_counts,
        )
        final_hash = str(checkpoint["state_sha256"])

    restored = load_posterior_gate_checkpoint(
        checkpoint, expected_source_hashes=rows.source_hashes
    )
    return PosteriorGateTrainingResult(
        adapter=restored,
        config=actual_config,
        seed=actual_config.seed,
        step=completed_steps,
        train_loss=train_loss,
        parameter_count=parameter_count,
        initial_state_sha256=initial_hash,
        adapter_state_sha256=final_hash,
        sampled_train_indices=tuple(sorted(sampled_rows)),
        sampled_index_trace_sha256=sampling_hash,
        group_sample_counts=dict(group_counts),
        checkpoint_payload=checkpoint,
    )


def measurement_comparator(z_measurement: Tensor) -> LeWMPosteriorAdapterOutput:
    if not torch.is_tensor(z_measurement) or not z_measurement.is_floating_point():
        raise TypeError("z_measurement must be a floating tensor")
    beta = torch.zeros(
        *z_measurement.shape[:-1],
        1,
        dtype=z_measurement.dtype,
        device=z_measurement.device,
    )
    return LeWMPosteriorAdapterOutput(z_post=z_measurement, beta=beta)


def fixed_beta_comparator(
    z_measurement: Tensor,
    z_prior: Tensor,
    *,
    beta: float,
) -> LeWMPosteriorAdapterOutput:
    gain = _finite_float(beta, "beta")
    if gain > 1.0:
        raise ValueError("beta must lie in [0, 1]")
    if z_prior.shape != z_measurement.shape:
        raise ValueError("z_prior must match z_measurement")
    beta_tensor = torch.full(
        (*z_measurement.shape[:-1], 1),
        gain,
        dtype=z_measurement.dtype,
        device=z_measurement.device,
    )
    return LeWMPosteriorAdapterOutput(
        z_post=z_measurement + beta_tensor * (z_prior - z_measurement),
        beta=beta_tensor,
    )


def action_blind_comparator(
    adapter: LeWMPosteriorAdapter,
    z_measurement: Tensor,
    z_prior: Tensor,
    innovation: Tensor,
    confidence: Tensor,
    mask_history: Tensor,
    action_summary: Tensor,
) -> LeWMPosteriorAdapterOutput:
    return adapter(
        z_measurement,
        z_prior,
        innovation,
        confidence,
        mask_history,
        torch.zeros_like(action_summary),
    )


def no_lewm_comparator(
    adapter: LeWMPosteriorAdapter,
    z_measurement: Tensor,
    z_hold_last: Tensor,
    innovation: Tensor,
    confidence: Tensor,
    mask_history: Tensor,
    action_summary: Tensor,
) -> LeWMPosteriorAdapterOutput:
    del innovation
    hold_last_innovation = (z_hold_last - z_measurement).clamp(
        min=-LATENT_INNOVATION_CLIP,
        max=LATENT_INNOVATION_CLIP,
    ) / LATENT_INNOVATION_CLIP
    return adapter(
        z_measurement,
        z_hold_last,
        hold_last_innovation,
        confidence,
        mask_history,
        action_summary,
    )


__all__ = [
    "DEFAULT_POSTERIOR_GATE_SEED",
    "DEFAULT_POSTERIOR_GATE_SEEDS",
    "HARD_MAX_POSTERIOR_GATE_PARAMETERS",
    "POSTERIOR_GATE_FORBIDDEN_MODEL_INPUTS",
    "POSTERIOR_GATE_LOSS_GROUPS",
    "POSTERIOR_GATE_MODEL_INPUTS",
    "POSTERIOR_GATE_MODES",
    "POSTERIOR_GATE_SELECTION_CONTRACT",
    "POSTERIOR_GATE_TRAINING_CHECKPOINT_TYPE",
    "POSTERIOR_GATE_TRAINING_SCHEMA_VERSION",
    "PosteriorGateForwardInputs",
    "PosteriorGateLossMasks",
    "PosteriorGateMode",
    "PosteriorGateSeedResult",
    "PosteriorGateSplitIndices",
    "PosteriorGateTrainingConfig",
    "PosteriorGateTrainingData",
    "PosteriorGateTrainingResult",
    "PosteriorGateTrainingRows",
    "action_blind_comparator",
    "balanced_posterior_gate_loss",
    "build_posterior_gate_forward_inputs",
    "build_posterior_gate_train_loader",
    "canonical_metadata_sha256",
    "fixed_beta_comparator",
    "forward_posterior_gate",
    "load_posterior_gate_checkpoint",
    "measurement_comparator",
    "no_lewm_comparator",
    "posterior_adapter_architecture",
    "posterior_adapter_parameter_count",
    "posterior_adapter_state_sha256",
    "tensor_sha256",
    "train_posterior_gate",
    "validate_posterior_gate_training_contract",
]
