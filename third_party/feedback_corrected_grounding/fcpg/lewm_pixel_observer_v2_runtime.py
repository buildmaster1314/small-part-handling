"""Causal runtime and provenance binding for pixel observer v2.

The runtime owns GroundedLeWM embedding construction. It accepts only
historical observations, transition actions, and current known kinematics.
Current RGB/corrections/oracle coordinates and external embeddings cannot enter.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import uuid

import torch
from torch import nn

from fcpg.lewm_pixel_observer_v2 import (
    ORACLE_CEILING_TARGET_MODE,
    PAIRED_DEPLOYMENT_TARGET_MODE,
    PIXEL_OBSERVER_V2_CHECKPOINT_TYPE,
    SINGLE_MEASUREMENT_BASELINE_TARGET_MODE,
    PixelObserverV2Prior,
    PixelResidualVelocityInnovationAdapter,
    pixel_observer_v2_checkpoint_metadata,
)


_SCHEMA_VERSION = 2
_DEPLOYMENT_CONTRACT = "independent_a_b_mutual_supervision"
_ALLOWED_HISTORY = frozenset(
    {
        "pixels",
        "rgb",
        "proprio",
        "obs",
        "q_nominal",
        "q_corrected",
        "q_grounded",
        "keypoints_xyz",
        "valid",
        "valid_nominal",
    }
)
_REQUIRED_ALIASES = (
    ("pixels", "rgb"),
    ("proprio", "obs"),
    ("q_nominal",),
    ("keypoints_xyz",),
)
_OPTIONAL_ALIASES = (
    ("q_corrected", "q_grounded"),
    ("valid", "valid_nominal"),
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"provenance must be JSON-compatible, got {type(value)!r}")


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def state_dict_sha256(module: nn.Module) -> str:
    """Hash state tensor names, metadata, and bytes deterministically."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if not torch.is_tensor(tensor):
            raise TypeError(f"state_dict entry {name!r} is not a tensor")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    for value in tuple(module.parameters()) + tuple(module.buffers()):
        if value.is_floating_point():
            return value.device, value.dtype
    return torch.device("cpu"), torch.float32


def _validate_history(
    history_batch: Mapping[str, torch.Tensor],
    *,
    history_size: int,
    num_keypoints: int,
) -> tuple[int, torch.device]:
    if not isinstance(history_batch, Mapping):
        raise TypeError("history_batch must be a mapping")
    names = frozenset(history_batch)
    unknown = sorted(names - _ALLOWED_HISTORY)
    if unknown:
        raise ValueError(
            "history_batch contains non-historical or forbidden fields: "
            f"{unknown}"
        )
    for group in _REQUIRED_ALIASES:
        if sum(name in names for name in group) != 1:
            raise ValueError(f"history_batch requires exactly one of {group}")
    for group in _OPTIONAL_ALIASES:
        if sum(name in names for name in group) > 1:
            raise ValueError(f"history_batch aliases are ambiguous: {group}")
    q_nominal = history_batch["q_nominal"]
    expected_tail = (history_size, num_keypoints, 2)
    if not torch.is_tensor(q_nominal) or q_nominal.shape[1:] != expected_tail:
        raise ValueError(f"history q_nominal must have shape [B, {expected_tail}]")
    batch_size = int(q_nominal.shape[0])
    device = q_nominal.device
    for name, value in history_batch.items():
        if not torch.is_tensor(value):
            raise TypeError(f"history field {name!r} must be a tensor")
        if value.ndim < 2 or value.shape[:2] != (batch_size, history_size):
            raise ValueError(
                f"history field {name!r} must begin with "
                f"[B, H]=[{batch_size}, {history_size}]"
            )
        if value.device != device:
            raise ValueError("all history tensors must use the same device")
    return batch_size, device


def _validate_current(
    values: Mapping[str, torch.Tensor],
    *,
    batch_size: int,
    num_keypoints: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    shapes = {
        "posterior_residual_px": (batch_size, num_keypoints, 2),
        "residual_velocity_px": (batch_size, num_keypoints, 2),
        "previous_q_nominal_px": (batch_size, num_keypoints, 2),
        "current_q_nominal_px": (batch_size, num_keypoints, 2),
        "previous_keypoints_xyz": (batch_size, num_keypoints, 3),
        "current_keypoints_xyz": (batch_size, num_keypoints, 3),
    }
    for name, shape in shapes.items():
        value = values[name]
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not value.is_floating_point() or value.dtype != dtype:
            raise TypeError(f"{name} must use runtime floating dtype {dtype}")
        if value.device != device:
            raise ValueError(f"{name} must use runtime device {device}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
    age = values["measurement_age_steps"]
    if age.shape != (batch_size, num_keypoints):
        raise ValueError("measurement_age_steps has an invalid shape")
    if not age.is_floating_point() or age.dtype != dtype:
        raise TypeError(f"measurement_age_steps must use runtime floating dtype {dtype}")
    if age.device != device or not bool(torch.isfinite(age).all()):
        raise ValueError("measurement_age_steps must be finite on the runtime device")
    if bool(((age < 0.0) | (age > 1.0)).any()):
        raise ValueError("measurement_age_steps must be normalized to [0,1]")
    for name in (
        "previous_nominal_valid",
        "current_nominal_valid",
        "previous_measurement_valid",
    ):
        value = values[name]
        if value.shape != (batch_size, num_keypoints):
            raise ValueError(f"{name} has an invalid shape")
        if value.dtype != torch.bool:
            raise TypeError(f"{name} must have boolean dtype")
        if value.device != device:
            raise ValueError(f"{name} must use runtime device {device}")


class CausalLeWMPixelObserverV2Runtime(nn.Module):
    """Frozen GroundedLeWM plus a trainable second-order pixel adapter."""

    def __init__(
        self,
        base_model: nn.Module,
        adapter: PixelResidualVelocityInnovationAdapter,
    ) -> None:
        super().__init__()
        required = ("history_size", "latent_dim", "action_dim", "frame_encoder")
        missing = [name for name in required if not hasattr(base_model, name)]
        if missing:
            raise TypeError(f"base_model is missing GroundedLeWM fields: {missing}")
        if not hasattr(base_model, "encode_frames") or not hasattr(base_model, "predict"):
            raise TypeError("base_model must expose encode_frames and predict")
        num_keypoints = int(base_model.frame_encoder.num_keypoints)
        if adapter.latent_dim != int(base_model.latent_dim):
            raise ValueError("adapter latent_dim differs from GroundedLeWM")
        if adapter.num_keypoints != num_keypoints:
            raise ValueError("adapter num_keypoints differs from GroundedLeWM")
        self.base_model = base_model
        self.adapter = adapter
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()

    @property
    def history_size(self) -> int:
        return int(self.base_model.history_size)

    @property
    def num_keypoints(self) -> int:
        return int(self.base_model.frame_encoder.num_keypoints)

    def train(self, mode: bool = True) -> "CausalLeWMPixelObserverV2Runtime":
        super().train(mode)
        self.base_model.eval()
        return self

    def predict_prior(
        self,
        *,
        history_batch: Mapping[str, torch.Tensor],
        transition_actions: torch.Tensor,
        posterior_residual_px: torch.Tensor,
        residual_velocity_px: torch.Tensor,
        previous_q_nominal_px: torch.Tensor,
        current_q_nominal_px: torch.Tensor,
        previous_keypoints_xyz: torch.Tensor,
        current_keypoints_xyz: torch.Tensor,
        previous_nominal_valid: torch.Tensor,
        current_nominal_valid: torch.Tensor,
        previous_measurement_valid: torch.Tensor,
        measurement_age_steps: torch.Tensor,
    ) -> PixelObserverV2Prior:
        """Construct the LeWM prior internally, then apply known geometry."""

        batch_size, history_device = _validate_history(
            history_batch,
            history_size=self.history_size,
            num_keypoints=self.num_keypoints,
        )
        base_device, base_dtype = _module_device_dtype(self.base_model)
        adapter_device, adapter_dtype = _module_device_dtype(self.adapter)
        if history_device != base_device or adapter_device != base_device:
            raise ValueError("history, GroundedLeWM, and adapter devices must match")
        if adapter_dtype != base_dtype:
            raise TypeError("GroundedLeWM and adapter dtypes must match")
        expected_actions = (
            batch_size,
            self.history_size,
            int(self.base_model.action_dim),
        )
        if transition_actions.shape != expected_actions:
            raise ValueError(f"transition_actions must have shape {expected_actions}")
        if (
            not transition_actions.is_floating_point()
            or transition_actions.dtype != base_dtype
        ):
            raise TypeError(f"transition_actions must use runtime dtype {base_dtype}")
        if transition_actions.device != base_device:
            raise ValueError(f"transition_actions must use runtime device {base_device}")
        current_values = {
            "posterior_residual_px": posterior_residual_px,
            "residual_velocity_px": residual_velocity_px,
            "previous_q_nominal_px": previous_q_nominal_px,
            "current_q_nominal_px": current_q_nominal_px,
            "previous_keypoints_xyz": previous_keypoints_xyz,
            "current_keypoints_xyz": current_keypoints_xyz,
            "previous_nominal_valid": previous_nominal_valid,
            "current_nominal_valid": current_nominal_valid,
            "previous_measurement_valid": previous_measurement_valid,
            "measurement_age_steps": measurement_age_steps,
        }
        _validate_current(
            current_values,
            batch_size=batch_size,
            num_keypoints=self.num_keypoints,
            device=base_device,
            dtype=base_dtype,
        )
        if self.adapter.use_predicted_latent:
            with torch.no_grad():
                _, history_embedding = self.base_model.encode_frames(history_batch)
                predicted = self.base_model.predict(
                    history_embedding, transition_actions
                )[:, -1].detach()
        else:
            # The matched no-latent ablation has the same adapter parameters,
            # but its latent input is identically zero. Avoiding the frozen
            # backbone forward changes only compute, not the ablation output.
            predicted = torch.zeros(
                batch_size,
                int(self.base_model.latent_dim),
                device=base_device,
                dtype=base_dtype,
            )
        return self.adapter.predict_prior(
            {"predicted_embedding": predicted, **current_values}
        )

    forward = predict_prior


def _validate_provenance(
    target_mode: str, data_provenance: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(data_provenance, Mapping) or not data_provenance:
        raise ValueError("data_provenance must be a non-empty mapping")
    result = _plain(data_provenance)
    if target_mode == PAIRED_DEPLOYMENT_TARGET_MODE:
        if result.get("target_contract") != _DEPLOYMENT_CONTRACT:
            raise ValueError("paired deployment requires the A/B target contract")
        first = result.get("measurement_a_sha256")
        second = result.get("measurement_b_sha256")
        if not isinstance(first, str) or not isinstance(second, str):
            raise ValueError("paired deployment provenance requires A/B SHA-256 values")
        if len(first) != 64 or len(second) != 64 or first == second:
            raise ValueError("A/B measurement SHA-256 values must be distinct")
    return result


def _atomic_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise TypeError("pixel observer v2 checkpoint must be a mapping")
    return payload


def _binding(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": payload["schema_version"],
        "checkpoint_type": payload["checkpoint_type"],
        "target_mode": payload["target_mode"],
        "adapter_config_sha256": payload["adapter_config_sha256"],
        "adapter_state_sha256": payload["adapter_state_sha256"],
        "base_state_sha256": payload["base_state_sha256"],
        "data_provenance_sha256": payload["data_provenance_sha256"],
        "supervision_contract": payload["metadata"]["supervision_contract"],
    }


def save_pixel_observer_v2_checkpoint(
    path: str | Path,
    runtime: CausalLeWMPixelObserverV2Runtime,
    *,
    target_mode: str,
    data_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    allowed = {
        PAIRED_DEPLOYMENT_TARGET_MODE,
        SINGLE_MEASUREMENT_BASELINE_TARGET_MODE,
        ORACLE_CEILING_TARGET_MODE,
    }
    if target_mode not in allowed:
        raise ValueError("unknown pixel observer v2 target_mode")
    provenance = _validate_provenance(target_mode, data_provenance)
    config = runtime.adapter.export_config()
    payload: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "checkpoint_type": PIXEL_OBSERVER_V2_CHECKPOINT_TYPE,
        "target_mode": target_mode,
        "adapter_config": config,
        "adapter_config_sha256": canonical_sha256(config),
        "adapter_state": runtime.adapter.state_dict(),
        "adapter_state_sha256": state_dict_sha256(runtime.adapter),
        "base_state_sha256": state_dict_sha256(runtime.base_model),
        "data_provenance": provenance,
        "data_provenance_sha256": canonical_sha256(provenance),
        "metadata": pixel_observer_v2_checkpoint_metadata(
            runtime.adapter, target_mode=target_mode
        ),
    }
    payload["binding_sha256"] = canonical_sha256(_binding(payload))
    _atomic_save(payload, Path(path).expanduser().resolve())
    return payload


def _adapter_from_config(config: Mapping[str, Any]) -> PixelResidualVelocityInnovationAdapter:
    names = (
        "latent_dim",
        "num_keypoints",
        "image_size",
        "hidden_dim",
        "max_velocity_innovation_px",
        "residual_scale_px",
        "velocity_scale_px",
        "use_predicted_latent",
    )
    values = {name: config[name] for name in names}
    values["image_size"] = tuple(values["image_size"])
    return PixelResidualVelocityInnovationAdapter(**values)


def load_pixel_observer_v2_checkpoint(
    path: str | Path,
    *,
    base_model: nn.Module,
    expected_data_provenance: Mapping[str, Any],
    for_deployment: bool,
) -> tuple[CausalLeWMPixelObserverV2Runtime, dict[str, Any]]:
    """Verify target, config, states, and data before constructing a runtime."""

    base_device, base_dtype = _module_device_dtype(base_model)
    payload = _load(Path(path).expanduser().resolve(), base_device)
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unsupported pixel observer v2 checkpoint schema")
    if payload.get("checkpoint_type") != PIXEL_OBSERVER_V2_CHECKPOINT_TYPE:
        raise ValueError("wrong pixel observer v2 checkpoint type")
    target_mode = payload.get("target_mode")
    allowed = {
        PAIRED_DEPLOYMENT_TARGET_MODE,
        SINGLE_MEASUREMENT_BASELINE_TARGET_MODE,
        ORACLE_CEILING_TARGET_MODE,
    }
    if target_mode not in allowed:
        raise ValueError("checkpoint target mode is invalid")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint metadata must be a mapping")
    expected_metadata = pixel_observer_v2_checkpoint_metadata(
        _adapter_from_config(payload["adapter_config"]), target_mode=target_mode
    )
    if metadata != expected_metadata:
        raise ValueError("checkpoint supervision metadata is inconsistent")
    if payload.get("adapter_config_sha256") != canonical_sha256(
        payload["adapter_config"]
    ):
        raise ValueError("adapter config binding is invalid")
    if payload.get("data_provenance_sha256") != canonical_sha256(
        payload["data_provenance"]
    ):
        raise ValueError("data provenance binding is invalid")
    expected = _validate_provenance(target_mode, expected_data_provenance)
    if canonical_sha256(expected) != payload["data_provenance_sha256"]:
        raise ValueError("expected data provenance differs from checkpoint binding")
    if payload.get("base_state_sha256") != state_dict_sha256(base_model):
        raise ValueError("GroundedLeWM state differs from checkpoint binding")
    if payload.get("binding_sha256") != canonical_sha256(_binding(payload)):
        raise ValueError("checkpoint target/config/state binding is invalid")
    adapter = _adapter_from_config(payload["adapter_config"]).to(
        device=base_device, dtype=base_dtype
    )
    adapter.load_state_dict(payload["adapter_state"], strict=True)
    if payload.get("adapter_state_sha256") != state_dict_sha256(adapter):
        raise ValueError("adapter state binding is invalid")
    contract = metadata["supervision_contract"]
    if for_deployment and (
        target_mode != PAIRED_DEPLOYMENT_TARGET_MODE
        or not bool(contract.get("deployment_allowed"))
        or not bool(contract.get("formal_deployment_claim_allowed"))
        or contract.get("target_contract") != _DEPLOYMENT_CONTRACT
    ):
        raise ValueError(
            "only independent A/B paired-supervision checkpoints are deployable"
        )
    runtime = CausalLeWMPixelObserverV2Runtime(base_model, adapter)
    runtime.eval()
    return runtime, payload
