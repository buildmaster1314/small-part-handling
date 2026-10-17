"""Frozen complementary observer for native P1 anchor dynamics.

This module intentionally adds no trainable component.  It combines the
deployable geometry-hold baseline with a frozen P1-native LeWM prior and writes
the fused posterior back into the recurrent history at every missing-vision
step::

    posterior_h = geometry_h + gain_h * (prior_h - geometry_h)

The loader is deliberately fail-closed.  A model is accepted only when its
checkpoint identity, training protocol, fixed endpoint, configuration, and
action normalization all match the pre-declared experiment contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import nn

from fcpg.lewm_anchor_dynamics_pair import MatchedAnchorDynamicsPredictor
from fcpg.lewm_p1_native_dynamics import (
    P1_NATIVE_ACTION_DIM,
    P1_NATIVE_PROTOCOL,
    P1_NATIVE_WRONG_ACTION_SHIFTS,
    P1NativePredictorConfig,
    build_matched_p1_native_predictors,
    validate_matched_p1_native_pair,
)


P1_NATIVE_PAIR_CHECKPOINT_TYPE: Final = "fcpg_p1_native_dynamics_pair_v1"
P1_NATIVE_FROZEN_SEEDS: Final = (15801, 15802, 15803)
P1_NATIVE_FROZEN_STEP: Final = 2000
P1_NATIVE_FROZEN_SELECTION: Final = "fixed_latest"
P1_NATIVE_COMPLEMENTARY_GAINS: Final = (0.25, 0.50, 0.75, 0.75, 0.75, 0.75)

_CHECKPOINT_KEYS: Final = frozenset(
    {
        "checkpoint_type",
        "protocol",
        "selection",
        "completed",
        "step",
        "seed",
        "predictor_config",
        "parameter_count_per_head",
        "action_mean",
        "action_std",
        "factual_state",
        "no_action_state",
        "config",
        "strict_cache_sha256",
        "p1_policy_checkpoint_sha256",
        "report_screen",
    }
)


@dataclass(frozen=True)
class FrozenP1NativeCheckpointSpec:
    """Expected identity of one immutable P1-native 2k checkpoint."""

    seed: int
    path: Path
    sha256: str


@dataclass(frozen=True)
class LoadedP1NativePair:
    """Frozen factual/no-action pair with verified provenance."""

    seed: int
    path: Path
    sha256: str
    factual: MatchedAnchorDynamicsPredictor
    no_action: MatchedAnchorDynamicsPredictor
    predictor_config: P1NativePredictorConfig
    action_mean: torch.Tensor
    action_std: torch.Tensor


@dataclass(frozen=True)
class ComplementaryMatchedRollout:
    """Recursive factual, no-action, and all wrong-action posteriors."""

    factual: torch.Tensor
    no_action: torch.Tensor
    wrong_action: torch.Tensor


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    if value != value.lower() or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _typed_equal(left: object, right: object) -> bool:
    """Compare JSON-like contracts without accepting ``True == 1``."""

    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return left.keys() == right.keys() and all(
            _typed_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _typed_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _required_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _validate_expected_config(
    config: Mapping[str, Any],
) -> tuple[P1NativePredictorConfig, str, str]:
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("expected config schema_version must be exactly 1")
    if config.get("protocol") != P1_NATIVE_PROTOCOL:
        raise ValueError("expected config has the wrong P1-native protocol")

    data = _required_mapping(config.get("data"), label="expected config data")
    p1 = _required_mapping(config.get("p1"), label="expected config p1")
    model = _required_mapping(config.get("model"), label="expected config model")
    training = _required_mapping(config.get("training"), label="expected config training")

    cache_sha = _validate_sha256(
        data.get("strict_cache_sha256"), label="strict cache SHA-256"
    )
    policy_sha = _validate_sha256(
        p1.get("policy_checkpoint_sha256"), label="P1 policy SHA-256"
    )
    if type(model.get("sigreg_weight")) not in {int, float}:
        raise TypeError("model sigreg_weight must be numeric")
    if float(model["sigreg_weight"]) != 0.0:
        raise ValueError("frozen P1-native checkpoints require sigreg_weight=0")

    predictor_fields = set(P1NativePredictorConfig().as_kwargs())
    if set(model) != predictor_fields | {"sigreg_weight"}:
        raise ValueError("expected model config has unexpected or missing fields")
    predictor_values = {name: model[name] for name in predictor_fields}
    predictor_config = P1NativePredictorConfig(**predictor_values)

    if training.get("allowed_seeds") != list(P1_NATIVE_FROZEN_SEEDS):
        raise ValueError("expected config must bind exactly the three frozen seeds")
    if (
        type(training.get("optimizer_steps")) is not int
        or training["optimizer_steps"] != P1_NATIVE_FROZEN_STEP
    ):
        raise ValueError("expected config must bind the fixed 2k endpoint")
    if training.get("selection") != P1_NATIVE_FROZEN_SELECTION:
        raise ValueError("expected config must use fixed_latest selection")
    return predictor_config, cache_sha, policy_sha


def _normalization_tensor(
    value: object,
    *,
    label: str,
    positive: bool,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.shape != (P1_NATIVE_ACTION_DIM,):
        raise ValueError(
            f"{label} must have shape {(P1_NATIVE_ACTION_DIM,)}, got {array.shape}"
        )
    if array.dtype.kind != "f":
        raise TypeError(f"{label} must be floating point")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{label} contains non-finite values")
    if positive and not bool(np.all(array > 0.0)):
        raise ValueError(f"{label} must be strictly positive")
    return torch.from_numpy(np.asarray(array, dtype=np.float32).copy())


def _freeze_model(model: nn.Module, *, device: torch.device) -> None:
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def load_frozen_p1_native_checkpoint(
    spec: FrozenP1NativeCheckpointSpec,
    *,
    expected_config: Mapping[str, Any],
    device: str | torch.device = "cpu",
) -> LoadedP1NativePair:
    """Load one checkpoint only after verifying its complete experiment identity."""

    if type(spec.seed) is not int or spec.seed not in P1_NATIVE_FROZEN_SEEDS:
        raise ValueError("checkpoint seed is not one of the three frozen P1 seeds")
    expected_sha = _validate_sha256(spec.sha256, label="checkpoint SHA-256")
    path = Path(spec.path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = _sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"checkpoint SHA-256 mismatch for seed {spec.seed}: "
            f"expected {expected_sha}, got {actual_sha}"
        )

    predictor_config, cache_sha, policy_sha = _validate_expected_config(expected_config)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("P1-native checkpoint must contain a mapping")
    if set(payload) != _CHECKPOINT_KEYS:
        missing = sorted(_CHECKPOINT_KEYS - set(payload))
        extra = sorted(set(payload) - _CHECKPOINT_KEYS)
        raise ValueError(f"checkpoint schema mismatch; missing={missing}, extra={extra}")
    if payload["checkpoint_type"] != P1_NATIVE_PAIR_CHECKPOINT_TYPE:
        raise ValueError("checkpoint_type does not identify a P1-native pair")
    if payload["protocol"] != P1_NATIVE_PROTOCOL:
        raise ValueError("checkpoint protocol mismatch")
    if payload["selection"] != P1_NATIVE_FROZEN_SELECTION:
        raise ValueError("checkpoint is not the fixed_latest endpoint")
    if payload["completed"] is not True:
        raise ValueError("checkpoint training is not marked complete")
    if type(payload["step"]) is not int or payload["step"] != P1_NATIVE_FROZEN_STEP:
        raise ValueError("checkpoint is not the frozen step=2000 endpoint")
    if type(payload["seed"]) is not int or payload["seed"] != spec.seed:
        raise ValueError("checkpoint seed does not match its declared identity")
    if not _typed_equal(payload["config"], expected_config):
        raise ValueError("checkpoint embeds a different experiment config")
    if payload["strict_cache_sha256"] != cache_sha:
        raise ValueError("checkpoint strict-cache identity mismatch")
    if payload["p1_policy_checkpoint_sha256"] != policy_sha:
        raise ValueError("checkpoint P1-policy identity mismatch")
    if not isinstance(payload["report_screen"], Mapping):
        raise TypeError("checkpoint report_screen must be a mapping")

    expected_predictor = predictor_config.as_kwargs()
    if not _typed_equal(payload["predictor_config"], expected_predictor):
        raise ValueError("checkpoint predictor config mismatch")
    action_mean = _normalization_tensor(
        payload["action_mean"], label="action_mean", positive=False
    )
    action_std = _normalization_tensor(
        payload["action_std"], label="action_std", positive=True
    )

    factual, no_action = build_matched_p1_native_predictors(
        predictor_config, initialization_seed=spec.seed
    )
    factual_state = _required_mapping(payload["factual_state"], label="factual_state")
    no_action_state = _required_mapping(payload["no_action_state"], label="no_action_state")
    factual.load_state_dict(factual_state, strict=True)
    no_action.load_state_dict(no_action_state, strict=True)
    validate_matched_p1_native_pair(factual, no_action)

    factual_count = sum(parameter.numel() for parameter in factual.parameters())
    no_action_count = sum(parameter.numel() for parameter in no_action.parameters())
    count = payload["parameter_count_per_head"]
    if type(count) is not int or count <= 0:
        raise ValueError("parameter_count_per_head must be a positive integer")
    if factual_count != count or no_action_count != count:
        raise ValueError("checkpoint parameter count does not match reconstructed heads")
    for name, model in (("factual", factual), ("no_action", no_action)):
        if not torch.equal(model.action_mean.detach().cpu(), action_mean):
            raise ValueError(f"{name} action_mean differs from checkpoint metadata")
        if not torch.equal(model.action_std.detach().cpu(), action_std):
            raise ValueError(f"{name} action_std differs from checkpoint metadata")

    target_device = torch.device(device)
    _freeze_model(factual, device=target_device)
    _freeze_model(no_action, device=target_device)
    return LoadedP1NativePair(
        seed=spec.seed,
        path=path,
        sha256=actual_sha,
        factual=factual,
        no_action=no_action,
        predictor_config=predictor_config,
        action_mean=action_mean,
        action_std=action_std,
    )


def load_frozen_p1_native_ensemble(
    specs: Sequence[FrozenP1NativeCheckpointSpec],
    *,
    expected_config: Mapping[str, Any],
    device: str | torch.device = "cpu",
) -> tuple[LoadedP1NativePair, ...]:
    """Load exactly the three frozen seeds and reject normalization drift."""

    if len(specs) != len(P1_NATIVE_FROZEN_SEEDS):
        raise ValueError("the frozen ensemble must contain exactly three checkpoints")
    by_seed = {spec.seed: spec for spec in specs}
    if len(by_seed) != len(specs) or tuple(sorted(by_seed)) != P1_NATIVE_FROZEN_SEEDS:
        raise ValueError("the frozen ensemble must contain seeds 15801, 15802, 15803 once")
    loaded = tuple(
        load_frozen_p1_native_checkpoint(
            by_seed[seed], expected_config=expected_config, device=device
        )
        for seed in P1_NATIVE_FROZEN_SEEDS
    )
    reference = loaded[0]
    for item in loaded[1:]:
        if item.predictor_config != reference.predictor_config:
            raise ValueError("frozen ensemble predictor configs differ")
        if not torch.equal(item.action_mean, reference.action_mean):
            raise ValueError("frozen ensemble action_mean values differ")
        if not torch.equal(item.action_std, reference.action_std):
            raise ValueError("frozen ensemble action_std values differ")
    return loaded


def _validate_locked_model(model: MatchedAnchorDynamicsPredictor) -> None:
    if model.training:
        raise ValueError("complementary rollout requires an eval-mode predictor")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("complementary rollout requires a frozen predictor")


def _validate_complementary_inputs(
    model: MatchedAnchorDynamicsPredictor,
    history_token: torch.Tensor,
    history_action: torch.Tensor,
    future_action: torch.Tensor,
    geometry_token: torch.Tensor,
    gains: Sequence[float],
) -> tuple[int, int, int, int, int]:
    _validate_locked_model(model)
    if history_token.ndim != 4:
        raise ValueError("history_token must have shape [B,T,K,D]")
    batch, history, anchors, latent = map(int, history_token.shape)
    if (history, anchors, latent) != (
        int(model.history_size),
        int(model.num_keypoints),
        int(model.latent_dim),
    ):
        raise ValueError("history_token shape does not match the predictor")
    if history_action.shape != (batch, history - 1, int(model.action_dim)):
        raise ValueError("history_action must contain the observed transition actions")
    if (
        future_action.ndim != 4
        or future_action.shape[0] != batch
        or future_action.shape[-1] != int(model.action_dim)
    ):
        raise ValueError("future_action must have shape [B,S,P,A]")
    branches, horizon = map(int, future_action.shape[1:3])
    if branches <= 0 or horizon <= 0:
        raise ValueError("future_action must contain branches and horizons")
    if geometry_token.shape != (batch, branches, horizon, anchors, latent):
        raise ValueError("geometry_token must have shape [B,S,P,K,D]")
    if len(gains) != horizon:
        raise ValueError(f"gains must contain exactly {horizon} horizon values")
    for gain in gains:
        if isinstance(gain, bool) or not isinstance(gain, (int, float)):
            raise TypeError("all complementary gains must be numeric")
        if not math.isfinite(float(gain)) or not 0.0 <= float(gain) <= 1.0:
            raise ValueError("all complementary gains must be finite and in [0,1]")
    tensors = (history_token, history_action, future_action, geometry_token)
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all complementary rollout tensors must share one device")
    if any(not torch.is_floating_point(tensor) for tensor in tensors):
        raise TypeError("all complementary rollout tensors must be floating point")
    if any(not bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise ValueError("complementary rollout inputs contain non-finite values")
    return batch, branches, horizon, anchors, latent


@torch.inference_mode()
def recursive_complementary_rollout(
    model: MatchedAnchorDynamicsPredictor,
    history_token: torch.Tensor,
    history_action: torch.Tensor,
    future_action: torch.Tensor,
    geometry_token: torch.Tensor,
    *,
    gains: Sequence[float] = P1_NATIVE_COMPLEMENTARY_GAINS,
) -> torch.Tensor:
    """Fuse each prior with geometry and recursively write back the posterior."""

    batch, branches, horizon, anchors, latent = _validate_complementary_inputs(
        model,
        history_token,
        history_action,
        future_action,
        geometry_token,
        gains,
    )
    history = int(model.history_size)
    embeddings = history_token[:, None].expand(-1, branches, -1, -1, -1)
    actions = history_action[:, None].expand(-1, branches, -1, -1)
    outputs: list[torch.Tensor] = []
    for step, raw_gain in enumerate(gains):
        current_action = future_action[:, :, step : step + 1].float()
        action_context = torch.cat((actions.float(), current_action), dim=2)
        flat_embedding = embeddings.reshape(
            batch * branches, history, anchors, latent
        )
        flat_action = action_context.reshape(
            batch * branches, history, int(model.action_dim)
        )
        predicted = model.predict(flat_embedding, flat_action)
        if predicted.shape != flat_embedding.shape:
            raise ValueError("predictor output shape differs from its recurrent context")
        prior = predicted[:, -1].reshape(batch, branches, anchors, latent)
        if not bool(torch.isfinite(prior).all()):
            raise ValueError("predictor emitted non-finite prior values")
        baseline = geometry_token[:, :, step]
        gain = float(raw_gain)
        if gain == 0.0:
            posterior = baseline
        elif gain == 1.0:
            posterior = prior
        else:
            posterior = baseline + gain * (prior - baseline)
        outputs.append(posterior)
        embeddings = torch.cat(
            (embeddings[:, :, 1:], posterior[:, :, None]), dim=2
        )
        actions = action_context[:, :, 1:]
    return torch.stack(outputs, dim=2)


@torch.inference_mode()
def recursive_complementary_matched_rollout(
    factual: MatchedAnchorDynamicsPredictor,
    no_action: MatchedAnchorDynamicsPredictor,
    history_token: torch.Tensor,
    history_action: torch.Tensor,
    future_action: torch.Tensor,
    wrong_future_action: torch.Tensor,
    geometry_token: torch.Tensor,
    *,
    gains: Sequence[float] = P1_NATIVE_COMPLEMENTARY_GAINS,
) -> ComplementaryMatchedRollout:
    """Evaluate factual, matched no-action, and all seven wrong-action controls."""

    if wrong_future_action.ndim != 5:
        raise ValueError("wrong_future_action must have shape [B,W,S,P,A]")
    expected_tail = tuple(future_action.shape[1:])
    if (
        tuple(wrong_future_action.shape[2:]) != expected_tail
        or wrong_future_action.shape[0] != future_action.shape[0]
    ):
        raise ValueError("wrong_future_action is not aligned with future_action")
    if int(wrong_future_action.shape[1]) != len(P1_NATIVE_WRONG_ACTION_SHIFTS):
        raise ValueError("all seven cyclic wrong-action controls are required")
    factual_rollout = recursive_complementary_rollout(
        factual,
        history_token,
        history_action,
        future_action,
        geometry_token,
        gains=gains,
    )
    no_action_rollout = recursive_complementary_rollout(
        no_action,
        history_token,
        history_action,
        future_action,
        geometry_token,
        gains=gains,
    )
    wrong = torch.stack(
        [
            recursive_complementary_rollout(
                factual,
                history_token,
                history_action,
                wrong_future_action[:, index],
                geometry_token,
                gains=gains,
            )
            for index in range(int(wrong_future_action.shape[1]))
        ],
        dim=1,
    )
    return ComplementaryMatchedRollout(
        factual=factual_rollout,
        no_action=no_action_rollout,
        wrong_action=wrong,
    )


def hard_measurement_bypass(
    measurement: torch.Tensor,
    predicted: torch.Tensor,
    sensor_valid: torch.Tensor,
) -> torch.Tensor:
    """Return measurement bits unchanged wherever vision is valid or recovered."""

    if measurement.shape != predicted.shape:
        raise ValueError("measurement and predicted tensors must have identical shape")
    if measurement.dtype != predicted.dtype or measurement.device != predicted.device:
        raise ValueError("measurement and predicted tensors must share dtype and device")
    if sensor_valid.dtype != torch.bool:
        raise TypeError("sensor_valid must be boolean")
    if sensor_valid.device != measurement.device:
        raise ValueError("sensor_valid must be on the measurement device")
    if not torch.is_floating_point(measurement):
        raise TypeError("measurement and predicted tensors must be floating point")
    if not bool(torch.isfinite(measurement).all()) or not bool(torch.isfinite(predicted).all()):
        raise ValueError("measurement and predicted tensors must be finite")

    mask = sensor_valid
    while mask.ndim < measurement.ndim:
        mask = mask.unsqueeze(-1)
    try:
        mask = torch.broadcast_to(mask, measurement.shape)
    except RuntimeError as error:
        raise ValueError("sensor_valid cannot broadcast to the token shape") from error
    if bool(mask.all()):
        return measurement
    if not bool(mask.any()):
        return predicted
    return torch.where(mask, measurement, predicted)
