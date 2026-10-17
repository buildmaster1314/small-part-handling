"""Minimal policy plumbing for the causal known-drop bridge runtime.

This module deliberately leaves simulator scheduling outside the controller.
The privileged synthetic drop schedule and ``q_true`` may be used by an
external renderer to produce an observed record, but they are stripped before
the policy, A3 encoder, or learned bridge sees an input.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from fcpg.lewm_known_drop_bridge_runtime import (
    CAUSAL_KNOWN_DROP_METHODS,
    CausalDecision,
    CausalKnownDropBridgeRuntime,
)
from fcpg.lewm_known_drop_policy import grounded_tokens_from_memory
from fcpg.modules import sample_feature_map_at_pixels


ACTION_STEPS = 5
PRIVILEGED_BATCH_KEYS = frozenset(
    (
        "q_true",
        "canonical_target",
        "measurement_valid",
        "sensor_valid",
        "drop_active",
        "drop_schedule",
    )
)
OVERRIDE_KEYS = frozenset(
    (
        "grounded_token_override",
        "grounded_token_override_mask",
        "global_visual_token_override",
        "global_visual_token_override_mask",
    )
)
A3_RECORD_KEYS = (
    "rgb",
    "proprio",
    "q_nominal",
    "keypoints_xyz",
    "valid_nominal",
)


@dataclass(frozen=True)
class PolicyAnchorObservation:
    """Exact native policy anchor representation before temporal memory."""

    native_token: torch.Tensor
    global_visual_token: torch.Tensor
    sampled_visual: torch.Tensor
    points: torch.Tensor
    reliability: torch.Tensor
    xyz: torch.Tensor


@dataclass(frozen=True)
class CausalPolicyAction:
    """Selected action block and causal state used to select it."""

    action: torch.Tensor
    decision: CausalDecision


@dataclass(frozen=True)
class PairedEpisodeContract:
    """Frozen physical seed and privileged synthetic routing schedule."""

    env_seed: int
    task_index: int
    drop_start: int
    drop_duration: int

    def __post_init__(self) -> None:
        if self.env_seed < 0 or self.task_index < 0:
            raise ValueError("episode and task seeds must be non-negative")
        if self.drop_start < 3:
            raise ValueError("causal A3 evaluation requires drop_start >= 3")
        if self.drop_duration <= 0:
            raise ValueError("drop_duration must be positive")

    def drop_is_active(self, decision_index: int) -> bool:
        if decision_index < 0:
            raise ValueError("decision_index must be non-negative")
        return self.drop_start <= decision_index < self.drop_start + self.drop_duration


def deployment_policy_batch(
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remove all privileged routing/label fields from a policy batch."""

    if not isinstance(batch, Mapping):
        raise TypeError("policy batch must be a mapping")
    return {
        str(name): value
        for name, value in batch.items()
        if name not in PRIVILEGED_BATCH_KEYS
    }


def deployment_a3_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only fields consumed by the deployable FCPG/A3 encoder."""

    if not isinstance(record, Mapping):
        raise TypeError("A3 record must be a mapping")
    missing = [name for name in A3_RECORD_KEYS if name not in record]
    if missing:
        raise KeyError(f"A3 record is missing deployable fields: {missing}")
    return {name: record[name] for name in A3_RECORD_KEYS}


@torch.inference_mode()
def extract_policy_anchor_observation(
    policy: nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    atol: float = 1.0e-5,
    rtol: float = 1.0e-5,
) -> PolicyAnchorObservation:
    """Extract the exact sampled visual and native token from a frozen policy.

    Native token reconstruction is checked through the same frozen
    ``GroundedTokenAdapter``.  A batch that already contains an override is
    rejected so a predicted token can never be mistaken for a measurement.
    """

    if any(name in batch for name in OVERRIDE_KEYS):
        raise ValueError("native policy extraction rejects pre-existing overrides")
    if policy.training or any(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("policy anchor extraction requires a frozen eval policy")
    deployable = deployment_policy_batch(batch)
    memory, points, reliability = policy._observation_tokens(deployable)
    native_token = grounded_tokens_from_memory(
        memory,
        num_keypoints=int(getattr(policy, "num_keypoints")),
    )
    global_visual_token = memory[:, :, 0, :]
    xyz = deployable["keypoints_xyz"]
    if xyz.ndim != 4:
        raise ValueError("keypoints_xyz must have shape [B,H,K,3]")
    batch_size, history_steps, keypoints = (int(value) for value in xyz.shape[:3])
    visual_dim = int(getattr(policy, "vision_dim"))

    if "sampled_visual" in deployable:
        sampled_visual = deployable["sampled_visual"]
    elif "rgb" in deployable:
        rgb = deployable["rgb"]
        if rgb.ndim != 5 or tuple(rgb.shape[:2]) != (batch_size, history_steps):
            raise ValueError("rgb must have shape [B,H,3,Y,X]")
        height, width = int(rgb.shape[-2]), int(rgb.shape[-1])
        flat_rgb = rgb.reshape(batch_size * history_steps, 3, height, width)
        vision = policy.vision_encoder(flat_rgb)
        sampled = sample_feature_map_at_pixels(
            vision.feature_map,
            points.reshape(batch_size * history_steps, keypoints, 2),
            image_size=(width, height),
        )
        sampled_visual = sampled.reshape(
            batch_size,
            history_steps,
            keypoints,
            visual_dim,
        )
    else:
        raise KeyError(
            "exact sampled-visual extraction currently requires rgb or sampled_visual"
        )
    expected_visual = (batch_size, history_steps, keypoints, visual_dim)
    if tuple(sampled_visual.shape) != expected_visual:
        raise ValueError(
            f"sampled_visual must have shape {expected_visual}, got "
            f"{tuple(sampled_visual.shape)}"
        )
    flat_visual = sampled_visual.reshape(
        batch_size * history_steps,
        keypoints,
        visual_dim,
    )
    flat_xyz = xyz.reshape(batch_size * history_steps, keypoints, 3)
    reconstructed = policy.grounding_adapter.forward_sampled(
        flat_visual,
        policy._proprio(flat_xyz),
        reliability=reliability.reshape(batch_size * history_steps, keypoints),
    ).reshape_as(native_token)
    if not torch.allclose(reconstructed, native_token, atol=atol, rtol=rtol):
        error = torch.max(torch.abs(reconstructed.float() - native_token.float()))
        raise RuntimeError(
            "sampled visual does not reconstruct the native policy token; "
            f"max error={float(error.cpu()):.8f}"
        )
    return PolicyAnchorObservation(
        native_token=native_token.detach(),
        global_visual_token=global_visual_token.detach(),
        sampled_visual=sampled_visual.detach(),
        points=points.detach(),
        reliability=reliability.detach(),
        xyz=xyz.detach(),
    )


def _encode_current_a3(
    anchor_encoder: Any,
    record: Mapping[str, Any],
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    deployable = deployment_a3_record(record)
    encoded = anchor_encoder.encode([deployable])
    if isinstance(encoded, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(encoded)).to(
            device=reference.device,
            dtype=reference.dtype,
        )
    elif isinstance(encoded, torch.Tensor):
        tensor = encoded.to(device=reference.device, dtype=reference.dtype)
    else:
        raise TypeError("A3 encoder must return a numpy array or torch.Tensor")
    expected = (1, int(reference.shape[-2]), 192)
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"A3 current embedding must have shape {expected}, got {tuple(tensor.shape)}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("A3 current embedding contains non-finite values")
    return tensor


class CausalKnownDropPolicyController:
    """Join exact policy observations to one causal runtime state machine."""

    def __init__(
        self,
        *,
        policy: nn.Module,
        anchor_encoder: Any,
        runtime: CausalKnownDropBridgeRuntime,
        action_steps: int = ACTION_STEPS,
    ) -> None:
        if runtime.method not in CAUSAL_KNOWN_DROP_METHODS:
            raise ValueError("runtime has an unsupported method")
        if action_steps != ACTION_STEPS:
            raise ValueError(f"known-drop policy control is fixed to {ACTION_STEPS} actions")
        if policy.training or any(parameter.requires_grad for parameter in policy.parameters()):
            raise RuntimeError("controller requires a frozen eval policy")
        if int(getattr(policy, "obs_horizon", -1)) != runtime.policy_history_steps:
            raise ValueError("policy and causal runtime history lengths differ")
        self.policy = policy
        self.anchor_encoder = anchor_encoder
        self.runtime = runtime
        self.action_steps = int(action_steps)

    def _predict(self, batch: Mapping[str, torch.Tensor], decision: CausalDecision) -> CausalPolicyAction:
        deployable = deployment_policy_batch(batch)
        policy_batch = decision.override.attach(deployable)
        output = self.policy.predict_action_chunk(policy_batch)
        action = output.action[:, : self.action_steps]
        expected = (decision.embedding.shape[0], self.action_steps, 4)
        if tuple(action.shape) != expected:
            raise ValueError(
                f"policy action must have shape {expected}, got {tuple(action.shape)}"
            )
        if not bool(torch.isfinite(action).all()):
            raise ValueError("policy returned non-finite action")
        return CausalPolicyAction(action=action.detach(), decision=decision)

    @torch.inference_mode()
    def select_valid(
        self,
        observed_batch: Mapping[str, torch.Tensor],
        observed_current_record: Mapping[str, Any],
    ) -> CausalPolicyAction:
        """Consume one actually visible measurement and refresh both spaces."""

        observation = extract_policy_anchor_observation(self.policy, observed_batch)
        if int(observation.native_token.shape[0]) != 1:
            raise ValueError("minimal environment controller currently requires batch size one")
        current_token = observation.native_token[:, -1].float()
        current_xyz = observation.xyz[:, -1].float()
        embedding = _encode_current_a3(
            self.anchor_encoder,
            observed_current_record,
            reference=current_token,
        )
        sampled = observation.sampled_visual[:, -1].float()
        decision = self.runtime.begin_valid(
            embedding,
            current_token,
            current_xyz,
            sampled_visual=(sampled if self.runtime.method == "bridge_geometry" else None),
        )
        return self._predict(observed_batch, decision)

    @torch.inference_mode()
    def select_invalid(
        self,
        observed_batch: Mapping[str, torch.Tensor],
        observed_current_record: Mapping[str, Any],
    ) -> CausalPolicyAction:
        """Select an invalid learned/baseline action with no hidden-clean input."""

        if self.runtime.method == "oracle":
            raise RuntimeError("oracle invalid routing requires select_invalid_oracle")
        deployable = deployment_policy_batch(observed_batch)
        current_xyz = deployable["keypoints_xyz"][:, -1].float()
        if self.runtime.method == "measurement":
            observation = extract_policy_anchor_observation(self.policy, deployable)
            token = observation.native_token[:, -1].float()
            embedding = _encode_current_a3(
                self.anchor_encoder,
                observed_current_record,
                reference=token,
            )
            decision = self.runtime.begin_invalid_measurement(
                embedding,
                token,
                current_xyz,
            )
        else:
            # No current image, clean latent, or native token reaches this call.
            decision = self.runtime.begin_invalid_predicted(current_xyz)
        return self._predict(deployable, decision)

    @torch.inference_mode()
    def select_invalid_oracle(
        self,
        observed_batch: Mapping[str, torch.Tensor],
        hidden_clean_batch: Mapping[str, torch.Tensor],
        hidden_clean_current_record: Mapping[str, Any],
    ) -> CausalPolicyAction:
        """Explicitly privileged oracle entry point, isolated from learned code."""

        if self.runtime.method != "oracle":
            raise RuntimeError("hidden-clean routing is reserved for the oracle method")
        clean = extract_policy_anchor_observation(self.policy, hidden_clean_batch)
        token = clean.native_token[:, -1].float()
        xyz = deployment_policy_batch(observed_batch)["keypoints_xyz"][:, -1].float()
        embedding = _encode_current_a3(
            self.anchor_encoder,
            hidden_clean_current_record,
            reference=token,
        )
        decision = self.runtime.begin_invalid_oracle(embedding, token, xyz)
        return self._predict(observed_batch, decision)

    def register_executed_action(self, clipped_action: torch.Tensor) -> None:
        self.runtime.register_executed_action(clipped_action)


def evaluate_paired_methods(
    methods: Sequence[str],
    contract: PairedEpisodeContract,
    evaluator: Callable[[str, PairedEpisodeContract], Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    """Run every method against one immutable seed/drop contract."""

    labels = tuple(str(method) for method in methods)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("methods must be non-empty and unique")
    unknown = set(labels).difference(CAUSAL_KNOWN_DROP_METHODS)
    if unknown:
        raise ValueError(f"unsupported paired methods: {sorted(unknown)}")
    return {method: evaluator(method, contract) for method in labels}


__all__ = [
    "ACTION_STEPS",
    "A3_RECORD_KEYS",
    "CausalKnownDropPolicyController",
    "CausalPolicyAction",
    "PairedEpisodeContract",
    "PolicyAnchorObservation",
    "deployment_a3_record",
    "deployment_policy_batch",
    "evaluate_paired_methods",
    "extract_policy_anchor_observation",
]
