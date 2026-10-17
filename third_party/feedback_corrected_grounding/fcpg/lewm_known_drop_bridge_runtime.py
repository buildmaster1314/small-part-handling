"""Causal decision-level runtime for known-drop policy-token bridging.

The runtime keeps camera measurement routing separate from dynamics.  A valid
decision appends native A3 and policy-anchor states.  An invalid decision can
either use the corrupted measurement baseline, a privileged oracle diagnostic,
hold-last, or a recursively predicted A3 state translated into policy-token
space by :class:`ResidualAnchorPolicyBridge`.

Learned bridge methods deliberately expose no argument for a hidden clean
embedding or token.  Synthetic ``q_true`` and the drop schedule belong to the
external evaluator that chooses which public method to call; neither is stored
or consumed here.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from fcpg.lewm_anchor_dynamics_pair import (
    MatchedAnchorDynamicsPredictor,
    load_matched_predictor_pair_checkpoint,
)
from fcpg.lewm_anchor_policy_bridge import (
    ANCHOR_POLICY_BRIDGE_CHECKPOINT_TYPE,
    ResidualAnchorPolicyBridge,
)
from fcpg.lewm_policy_token_bridge import sha256_file


CAUSAL_KNOWN_DROP_METHODS = (
    "measurement",
    "hold_last",
    "oracle",
    "bridge_factual",
    "bridge_noaction",
    "bridge_geometry",
)
ANCHOR_POLICY_BRIDGE_SEMANTICS = "last_valid_residual_v1"
MIN_VALID_WARMUP_DECISIONS = 3
POLICY_HISTORY_STEPS = 2
EXECUTED_ACTION_DIM = 20


@dataclass(frozen=True)
class PolicyTokenOverride:
    """Raw-adjacent policy-token replacement window.

    A privileged policy-clock diagnostic established that the frozen P1 policy
    expects its two grounded slots to be raw-adjacent observations, not two
    decision-spaced states.  During a sensor-valid decision the native policy
    batch therefore remains untouched.  During an invalid decision, where no
    raw-adjacent measurements exist, the current bridge prediction is repeated
    into both slots.
    """

    tokens: torch.Tensor
    mask: torch.Tensor

    @property
    def any(self) -> bool:
        return bool(self.mask.any())

    def attach(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return a batch copy with only explicitly masked slots replaceable."""

        result = dict(batch)
        if self.any:
            result["grounded_token_override"] = self.tokens
            result["grounded_token_override_mask"] = self.mask
        return result


@dataclass(frozen=True)
class CausalDecision:
    """State chosen for one policy decision and its two-slot override."""

    decision_index: int
    source: str
    sensor_valid: bool
    embedding: torch.Tensor
    policy_token: torch.Tensor
    current_xyz: torch.Tensor
    override: PolicyTokenOverride


@dataclass(frozen=True)
class LoadedKnownDropBridgeModels:
    """Frozen models required by a known-drop bridge evaluator."""

    policy: nn.Module
    policy_args: dict[str, object]
    factual_predictor: MatchedAnchorDynamicsPredictor
    no_action_predictor: MatchedAnchorDynamicsPredictor
    dynamics_payload: dict[str, object]
    bridges: dict[str, ResidualAnchorPolicyBridge]
    bridge_payload: dict[str, object]

    @property
    def methods(self) -> tuple[str, ...]:
        methods = ["measurement", "hold_last", "oracle", "bridge_factual"]
        if "no_action" in self.bridges:
            methods.append("bridge_noaction")
        if hasattr(self.policy, "grounding_adapter") and hasattr(self.policy, "_proprio"):
            methods.append("bridge_geometry")
        return tuple(methods)

    def make_runtime(self, method: str) -> CausalKnownDropBridgeRuntime:
        """Construct one method-specific state machine from this frozen bundle."""

        if method not in self.methods:
            raise ValueError(f"method {method!r} is unavailable; choose from {self.methods}")
        return CausalKnownDropBridgeRuntime(
            method=method,
            factual_predictor=self.factual_predictor,
            no_action_predictor=self.no_action_predictor,
            factual_bridge=self.bridges.get("factual"),
            no_action_bridge=self.bridges.get("no_action"),
            geometry_policy=self.policy,
        )


@dataclass(frozen=True)
class _PolicySlot:
    token: torch.Tensor
    override_required: bool


def _finite_float_tensor(
    value: torch.Tensor,
    *,
    shape: tuple[int, ...],
    name: str,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be floating point")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must use {dtype}, got {value.dtype}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")
    return value.detach()


@torch.inference_mode()
def geometry_hold_policy_token(
    policy: nn.Module,
    last_valid_sampled_visual: torch.Tensor,
    current_xyz: torch.Tensor,
) -> torch.Tensor:
    """Rebuild a token from held visual evidence and current deployable FK.

    The helper has no current-image or hidden-clean input by construction.  It
    holds the last valid sampled visual descriptor fixed, updates proprioception
    from current XYZ, fixes reliability to one, and runs the *frozen* native
    policy ``GroundedTokenAdapter``.
    """

    if not hasattr(policy, "grounding_adapter") or not hasattr(policy, "_proprio"):
        raise TypeError("policy must expose grounding_adapter and _proprio")
    adapter = policy.grounding_adapter
    if not isinstance(adapter, nn.Module):
        raise TypeError("policy grounding_adapter must be an nn.Module")
    if adapter.training or any(parameter.requires_grad for parameter in adapter.parameters()):
        raise RuntimeError("geometry hold requires a frozen grounding adapter")
    if not isinstance(last_valid_sampled_visual, torch.Tensor):
        raise TypeError("last_valid_sampled_visual must be a torch.Tensor")
    if last_valid_sampled_visual.ndim != 3:
        raise ValueError("last_valid_sampled_visual must have shape [B,K,C]")
    batch_size, keypoints, visual_dim = (
        int(value) for value in last_valid_sampled_visual.shape
    )
    expected_keypoints = int(getattr(policy, "num_keypoints", -1))
    expected_visual_dim = int(getattr(policy, "vision_dim", -1))
    if (keypoints, visual_dim) != (expected_keypoints, expected_visual_dim):
        raise ValueError(
            "held visual dimensions do not match the frozen policy: "
            f"got {(keypoints, visual_dim)}, expected "
            f"{(expected_keypoints, expected_visual_dim)}"
        )
    visual = _finite_float_tensor(
        last_valid_sampled_visual,
        shape=(batch_size, keypoints, visual_dim),
        name="last_valid_sampled_visual",
    )
    xyz = _finite_float_tensor(
        current_xyz,
        shape=(batch_size, keypoints, 3),
        name="current_xyz",
        device=visual.device,
    ).to(dtype=visual.dtype)
    proprio = policy._proprio(xyz)
    reliability = torch.ones(
        (batch_size, keypoints),
        device=visual.device,
        dtype=visual.dtype,
    )
    token = adapter.forward_sampled(
        visual,
        proprio,
        reliability=reliability,
    )
    expected = (batch_size, keypoints, int(getattr(policy, "model_dim", -1)))
    if tuple(token.shape) != expected:
        raise ValueError(
            f"frozen grounding adapter returned {tuple(token.shape)}, expected {expected}"
        )
    if not bool(torch.isfinite(token).all()):
        raise ValueError("frozen grounding adapter returned non-finite tokens")
    return token


def load_anchor_policy_bridges(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[dict[str, ResidualAnchorPolicyBridge], dict[str, object]]:
    """Load the canonical multi-bridge checkpoint used by this evaluator.

    Expected payload fields are ``checkpoint_type``, ``bridge_config`` and a
    ``bridge_states`` mapping.  The factual state is mandatory; matched
    no-action control is optional.  The geometry-hold control is reconstructed
    directly through the frozen policy adapter and therefore has no learned
    state in this checkpoint.
    """

    target = torch.device(device)
    payload = torch.load(
        Path(checkpoint_path).expanduser().resolve(),
        map_location=target,
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise TypeError("bridge checkpoint must contain a mapping")
    if payload.get("checkpoint_type") != ANCHOR_POLICY_BRIDGE_CHECKPOINT_TYPE:
        raise ValueError("checkpoint is not an anchor-policy token bridge")
    if payload.get("bridge_semantics") != ANCHOR_POLICY_BRIDGE_SEMANTICS:
        raise ValueError(
            "bridge checkpoint does not use last-valid residual semantics"
        )
    config = payload.get("bridge_config")
    if not isinstance(config, Mapping):
        raise ValueError("bridge checkpoint is missing bridge_config")
    states = payload.get("bridge_states")
    if not isinstance(states, Mapping) or "factual" not in states:
        raise ValueError("bridge checkpoint requires bridge_states['factual']")
    allowed = frozenset(("factual", "no_action"))
    unknown = set(states).difference(allowed)
    if unknown:
        raise ValueError(f"bridge checkpoint contains unsupported states: {sorted(unknown)}")

    bridges: dict[str, ResidualAnchorPolicyBridge] = {}
    for name, state in states.items():
        if not isinstance(state, Mapping):
            raise TypeError(f"bridge state {name!r} must be a mapping")
        bridge = ResidualAnchorPolicyBridge(**dict(config)).to(target)
        bridge.load_state_dict(dict(state), strict=True)
        bridge.requires_grad_(False).eval()
        bridges[str(name)] = bridge
    return bridges, payload


def load_known_drop_bridge_models(
    *,
    policy_checkpoint: str | Path,
    corrector_checkpoint: str | Path,
    dynamics_checkpoint: str | Path,
    bridge_checkpoint: str | Path,
    device: str | torch.device = "cpu",
) -> LoadedKnownDropBridgeModels:
    """Strictly load and dimension-check all frozen bridge-evaluation models."""

    from fcpg.action_chunk_policy import load_action_chunk_policy

    target = torch.device(device)
    policy_path = Path(policy_checkpoint).expanduser().resolve()
    corrector_path = Path(corrector_checkpoint).expanduser().resolve()
    dynamics_path = Path(dynamics_checkpoint).expanduser().resolve()
    policy, policy_args = load_action_chunk_policy(policy_path, target)
    factual, no_action, dynamics_payload = load_matched_predictor_pair_checkpoint(
        dynamics_path,
        device=target,
    )
    bridges, bridge_payload = load_anchor_policy_bridges(
        bridge_checkpoint,
        device=target,
    )
    compatibility = bridge_payload.get("compatibility")
    if not isinstance(compatibility, Mapping):
        raise ValueError("formal bridge checkpoint is missing compatibility metadata")
    if compatibility.get("single_seed_runtime") is not True:
        raise ValueError("formal bridge does not authorize a single-seed runtime")
    expected_artifacts = {
        "p1_policy_checkpoint_sha256": sha256_file(policy_path),
        "p1_corrector_checkpoint_sha256": sha256_file(corrector_path),
    }
    for key, expected_digest in expected_artifacts.items():
        if compatibility.get(key) != expected_digest:
            raise ValueError(f"formal bridge {key} mismatch")
    dynamics_seed = dynamics_payload.get("seed")
    if isinstance(dynamics_seed, bool) or not isinstance(dynamics_seed, int):
        raise ValueError("dynamics checkpoint is missing its integer training seed")
    replicate_records = compatibility.get("a3_replicates")
    if isinstance(replicate_records, (str, bytes)) or not isinstance(
        replicate_records, (list, tuple)
    ):
        raise TypeError("formal bridge a3_replicates must be an ordered sequence")
    dynamics_digest = sha256_file(dynamics_path)
    matching_replicates = [
        item
        for item in replicate_records
        if isinstance(item, Mapping) and item.get("seed") == dynamics_seed
    ]
    if (
        len(matching_replicates) != 1
        or matching_replicates[0].get("checkpoint_sha256") != dynamics_digest
    ):
        raise ValueError("runtime A3 seed/checkpoint is not bound by formal bridge")
    factual_bridge = bridges["factual"]
    expected = (
        int(factual.num_keypoints),
        int(factual.latent_dim),
        int(getattr(policy, "num_keypoints")),
        int(getattr(policy, "model_dim")),
    )
    actual = (
        factual_bridge.num_keypoints,
        factual_bridge.latent_dim,
        factual_bridge.num_keypoints,
        factual_bridge.token_dim,
    )
    if actual != expected:
        raise ValueError(
            "policy, dynamics, and bridge dimensions are inconsistent: "
            f"bridge={actual}, expected={expected}"
        )
    if (
        no_action.num_keypoints != factual.num_keypoints
        or no_action.latent_dim != factual.latent_dim
        or no_action.action_dim != factual.action_dim
        or no_action.history_size != factual.history_size
    ):
        raise ValueError("factual and no-action dynamics heads are not matched")
    for name, bridge in bridges.items():
        if (
            bridge.num_keypoints,
            bridge.latent_dim,
            bridge.token_dim,
        ) != (
            factual.num_keypoints,
            factual.latent_dim,
            int(getattr(policy, "model_dim")),
        ):
            raise ValueError(f"bridge {name!r} dimensions are inconsistent")

    policy.requires_grad_(False).eval()
    factual.requires_grad_(False).eval()
    no_action.requires_grad_(False).eval()
    for bridge in bridges.values():
        bridge.requires_grad_(False).eval()
    return LoadedKnownDropBridgeModels(
        policy=policy,
        policy_args=dict(policy_args),
        factual_predictor=factual,
        no_action_predictor=no_action,
        dynamics_payload=dynamics_payload,
        bridges=bridges,
        bridge_payload=bridge_payload,
    )


class CausalKnownDropBridgeRuntime:
    """Maintain causal A3 and policy-token histories for one rollout method.

    The caller must follow a two-phase cadence at each decision:

    1. call one ``begin_*`` method to choose the current state/token;
    2. execute five raw actions and call :meth:`register_executed_action` with
       the actual clipped 20D action block.

    A first invalid decision is rejected until three valid decisions and their
    executed action blocks have initialized the A3 history.
    """

    def __init__(
        self,
        *,
        method: str,
        factual_predictor: MatchedAnchorDynamicsPredictor | nn.Module | None = None,
        no_action_predictor: MatchedAnchorDynamicsPredictor | nn.Module | None = None,
        factual_bridge: ResidualAnchorPolicyBridge | nn.Module | None = None,
        no_action_bridge: ResidualAnchorPolicyBridge | nn.Module | None = None,
        geometry_policy: nn.Module | None = None,
        policy_history_steps: int = POLICY_HISTORY_STEPS,
        min_valid_warmup: int = MIN_VALID_WARMUP_DECISIONS,
    ) -> None:
        if method not in CAUSAL_KNOWN_DROP_METHODS:
            raise ValueError(f"unknown causal known-drop method {method!r}")
        if policy_history_steps <= 0:
            raise ValueError("policy_history_steps must be positive")
        if min_valid_warmup < 3:
            raise ValueError("A3 requires at least three valid warmup decisions")
        self.method = method
        self.policy_history_steps = int(policy_history_steps)
        self.min_valid_warmup = int(min_valid_warmup)
        self.factual_predictor = factual_predictor
        self.no_action_predictor = no_action_predictor
        self.factual_bridge = factual_bridge
        self.no_action_bridge = no_action_bridge
        self.geometry_policy = geometry_policy

        if method == "bridge_factual" and (
            factual_predictor is None or factual_bridge is None
        ):
            raise ValueError("bridge_factual requires a factual predictor and bridge")
        if method == "bridge_noaction" and (
            no_action_predictor is None or no_action_bridge is None
        ):
            raise ValueError("bridge_noaction requires a no-action predictor and bridge")
        if method == "bridge_geometry" and geometry_policy is None:
            raise ValueError("bridge_geometry requires the frozen policy adapter")

        self._embeddings: deque[torch.Tensor] = deque(maxlen=3)
        self._actions: deque[torch.Tensor] = deque(maxlen=3)
        self._policy_slots: deque[_PolicySlot] = deque(maxlen=self.policy_history_steps)
        self._pending_action = False
        self._decision_count = 0
        self._valid_before_first_drop = 0
        self._seen_invalid = False
        self._batch_size: int | None = None
        self._num_keypoints: int | None = None
        self._latent_dim: int | None = None
        self._token_dim: int | None = None
        self._action_dim: int | None = None
        self._device: torch.device | None = None
        self._dtype: torch.dtype | None = None
        self._last_valid_embedding: torch.Tensor | None = None
        self._last_valid_token: torch.Tensor | None = None
        self._last_valid_xyz: torch.Tensor | None = None
        self._last_valid_sampled_visual: torch.Tensor | None = None

    @property
    def decision_count(self) -> int:
        return self._decision_count

    @property
    def ready_for_invalid(self) -> bool:
        return (
            not self._pending_action
            and self._valid_before_first_drop >= self.min_valid_warmup
            and len(self._embeddings) == 3
            and len(self._actions) == 3
        )

    def _require_decision_phase(self) -> None:
        if self._pending_action:
            raise RuntimeError("register the executed action before beginning another decision")

    def _initialize_or_validate_state(
        self,
        embedding: torch.Tensor,
        token: torch.Tensor,
        xyz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._batch_size is None:
            if embedding.ndim != 3 or token.ndim != 3 or xyz.ndim != 3:
                raise ValueError("state tensors must have shapes [B,K,D], [B,K,G], [B,K,3]")
            batch_size, keypoints, latent_dim = (int(v) for v in embedding.shape)
            if tuple(token.shape[:2]) != (batch_size, keypoints) or xyz.shape != (
                batch_size,
                keypoints,
                3,
            ):
                raise ValueError("embedding, policy token, and XYZ anchor axes must align")
            self._batch_size = batch_size
            self._num_keypoints = keypoints
            self._latent_dim = latent_dim
            self._token_dim = int(token.shape[-1])
            self._device = embedding.device
            self._dtype = embedding.dtype
        assert self._batch_size is not None
        assert self._num_keypoints is not None
        assert self._latent_dim is not None
        assert self._token_dim is not None
        assert self._device is not None
        assert self._dtype is not None
        embedding = _finite_float_tensor(
            embedding,
            shape=(self._batch_size, self._num_keypoints, self._latent_dim),
            name="embedding",
            device=self._device,
            dtype=self._dtype,
        )
        token = _finite_float_tensor(
            token,
            shape=(self._batch_size, self._num_keypoints, self._token_dim),
            name="policy_token",
            device=self._device,
            dtype=self._dtype,
        )
        xyz = _finite_float_tensor(
            xyz,
            shape=(self._batch_size, self._num_keypoints, 3),
            name="current_xyz",
            device=self._device,
            dtype=self._dtype,
        )
        return embedding, token, xyz

    def _validate_xyz(self, xyz: torch.Tensor) -> torch.Tensor:
        if self._batch_size is None:
            raise RuntimeError("a valid decision must initialize the runtime")
        assert self._num_keypoints is not None
        assert self._device is not None
        assert self._dtype is not None
        return _finite_float_tensor(
            xyz,
            shape=(self._batch_size, self._num_keypoints, 3),
            name="current_xyz",
            device=self._device,
            dtype=self._dtype,
        )

    def _policy_override(self) -> PolicyTokenOverride:
        if self._batch_size is None or self._num_keypoints is None or self._token_dim is None:
            raise RuntimeError("runtime has no policy-token state")
        if not self._policy_slots:
            raise RuntimeError("runtime has no policy-token slots")
        current = self._policy_slots[-1]
        # The P1 policy was trained on raw-adjacent slots [t-1,t].  A3 runs at
        # the five-raw-step decision clock, so stacking previous/current A3
        # predictions would silently feed [t-5,t].  For an invalid decision we
        # instead represent the single current bridge estimate as [pred_t,
        # pred_t].  A valid decision attaches an all-false mask, leaving the
        # native raw-adjacent history (including recovery) exactly unchanged.
        tokens = current.token.unsqueeze(1).expand(
            -1,
            self.policy_history_steps,
            -1,
            -1,
        )
        mask = torch.full(
            (
                self._batch_size,
                self.policy_history_steps,
                self._num_keypoints,
            ),
            bool(current.override_required),
            device=tokens.device,
            dtype=torch.bool,
        )
        return PolicyTokenOverride(tokens=tokens, mask=mask)

    def _append_decision(
        self,
        *,
        embedding: torch.Tensor,
        token: torch.Tensor,
        xyz: torch.Tensor,
        source: str,
        sensor_valid: bool,
        override_required: bool,
    ) -> CausalDecision:
        self._embeddings.append(embedding.detach())
        self._policy_slots.append(_PolicySlot(token.detach(), override_required))
        self._pending_action = True
        decision = CausalDecision(
            decision_index=self._decision_count,
            source=source,
            sensor_valid=sensor_valid,
            embedding=embedding.detach(),
            policy_token=token.detach(),
            current_xyz=xyz.detach(),
            override=self._policy_override(),
        )
        self._decision_count += 1
        return decision

    @torch.inference_mode()
    def begin_valid(
        self,
        native_embedding: torch.Tensor,
        native_policy_token: torch.Tensor,
        current_xyz: torch.Tensor,
        *,
        sampled_visual: torch.Tensor | None = None,
    ) -> CausalDecision:
        """Use native current measurement and immediately re-anchor A3."""

        self._require_decision_phase()
        embedding, token, xyz = self._initialize_or_validate_state(
            native_embedding,
            native_policy_token,
            current_xyz,
        )
        held_visual: torch.Tensor | None = None
        if self.method == "bridge_geometry":
            assert self.geometry_policy is not None
            if sampled_visual is None:
                raise ValueError(
                    "bridge_geometry valid decisions require sampled_visual"
                )
            visual_dim = int(getattr(self.geometry_policy, "vision_dim", -1))
            held_visual = _finite_float_tensor(
                sampled_visual,
                shape=(
                    int(embedding.shape[0]),
                    int(embedding.shape[1]),
                    visual_dim,
                ),
                name="sampled_visual",
                device=embedding.device,
                dtype=embedding.dtype,
            )
        if not self._seen_invalid:
            self._valid_before_first_drop += 1
        self._last_valid_embedding = embedding
        self._last_valid_token = token
        self._last_valid_xyz = xyz
        if held_visual is not None:
            self._last_valid_sampled_visual = held_visual
        return self._append_decision(
            embedding=embedding,
            token=token,
            xyz=xyz,
            source="measurement",
            sensor_valid=True,
            override_required=False,
        )

    def _require_invalid_ready(self) -> None:
        self._require_decision_phase()
        if not self.ready_for_invalid:
            raise RuntimeError(
                "invalid routing requires at least three valid decisions and "
                "their three executed 20D action blocks"
            )
        self._seen_invalid = True

    @torch.inference_mode()
    def begin_invalid_measurement(
        self,
        corrupted_embedding: torch.Tensor,
        corrupted_policy_token: torch.Tensor,
        current_xyz: torch.Tensor,
    ) -> CausalDecision:
        """Use only the actually observed corrupted measurement baseline."""

        if self.method != "measurement":
            raise RuntimeError("invalid measurement input is reserved for method='measurement'")
        self._require_invalid_ready()
        embedding, token, xyz = self._initialize_or_validate_state(
            corrupted_embedding,
            corrupted_policy_token,
            current_xyz,
        )
        return self._append_decision(
            embedding=embedding,
            token=token,
            xyz=xyz,
            source="corrupted_measurement",
            sensor_valid=False,
            override_required=False,
        )

    @torch.inference_mode()
    def begin_invalid_oracle(
        self,
        hidden_clean_embedding: torch.Tensor,
        hidden_clean_policy_token: torch.Tensor,
        current_xyz: torch.Tensor,
    ) -> CausalDecision:
        """Privileged diagnostic; learned methods cannot call this entry point."""

        if self.method != "oracle":
            raise RuntimeError("hidden clean state is reserved for method='oracle'")
        self._require_invalid_ready()
        embedding, token, xyz = self._initialize_or_validate_state(
            hidden_clean_embedding,
            hidden_clean_policy_token,
            current_xyz,
        )
        return self._append_decision(
            embedding=embedding,
            token=token,
            xyz=xyz,
            source="oracle",
            sensor_valid=False,
            override_required=True,
        )

    def _predict_next_embedding(self, predictor: nn.Module) -> torch.Tensor:
        assert self._batch_size is not None
        assert self._num_keypoints is not None
        assert self._latent_dim is not None
        assert self._action_dim is not None
        history = torch.stack(tuple(self._embeddings), dim=1)
        actions = torch.stack(tuple(self._actions), dim=1)
        predicted_sequence = predictor.predict(history, actions)
        expected = (
            self._batch_size,
            3,
            self._num_keypoints,
            self._latent_dim,
        )
        if tuple(predicted_sequence.shape) != expected:
            raise ValueError(
                f"dynamics predictor must return {expected}, got "
                f"{tuple(predicted_sequence.shape)}"
            )
        if not bool(torch.isfinite(predicted_sequence).all()):
            raise ValueError("dynamics predictor returned non-finite state")
        return predicted_sequence[:, -1]

    @torch.inference_mode()
    def begin_invalid_predicted(self, current_xyz: torch.Tensor) -> CausalDecision:
        """Predict an invalid decision without accepting any hidden clean input."""

        if self.method not in (
            "hold_last",
            "bridge_factual",
            "bridge_noaction",
            "bridge_geometry",
        ):
            raise RuntimeError(f"method={self.method!r} does not use predicted routing")
        self._require_invalid_ready()
        xyz = self._validate_xyz(current_xyz)
        assert self._last_valid_embedding is not None
        assert self._last_valid_token is not None
        assert self._last_valid_xyz is not None

        if self.method == "hold_last":
            embedding = self._last_valid_embedding
            token = self._last_valid_token
            source = "hold_last"
        elif self.method == "bridge_geometry":
            assert self.geometry_policy is not None
            assert self._last_valid_sampled_visual is not None
            embedding = self._last_valid_embedding
            token = geometry_hold_policy_token(
                self.geometry_policy,
                self._last_valid_sampled_visual,
                xyz,
            )
            source = "bridge_geometry"
        else:
            if self.method == "bridge_factual":
                predictor = self.factual_predictor
                bridge = self.factual_bridge
            else:
                predictor = self.no_action_predictor
                bridge = self.no_action_bridge
            assert predictor is not None and bridge is not None
            embedding = self._predict_next_embedding(predictor)
            token = bridge(
                embedding,
                self._last_valid_embedding,
                self._last_valid_token,
                xyz,
                self._last_valid_xyz,
            )
            source = self.method
        return self._append_decision(
            embedding=embedding,
            token=token,
            xyz=xyz,
            source=source,
            sensor_valid=False,
            override_required=True,
        )

    def register_executed_action(self, action_block: torch.Tensor) -> None:
        """Append the actual clipped five-action block after policy execution."""

        if not self._pending_action:
            raise RuntimeError("begin a decision before registering its executed action")
        if self._batch_size is None or self._device is None or self._dtype is None:
            raise RuntimeError("runtime is not initialized")
        if not isinstance(action_block, torch.Tensor):
            raise TypeError("action_block must be a torch.Tensor")
        if action_block.ndim == 3:
            if tuple(action_block.shape[1:]) != (5, 4):
                raise ValueError("rank-3 action_block must have shape [B,5,4]")
            action_block = action_block.reshape(self._batch_size, 20)
        if action_block.ndim != 2 or int(action_block.shape[0]) != self._batch_size:
            raise ValueError("action_block must have shape [B,A] or [B,5,4]")
        if int(action_block.shape[-1]) != EXECUTED_ACTION_DIM:
            raise ValueError(
                f"action_block must contain exactly {EXECUTED_ACTION_DIM} clipped "
                "action values"
            )
        if self._action_dim is None:
            self._action_dim = EXECUTED_ACTION_DIM
        expected = (self._batch_size, self._action_dim)
        action = _finite_float_tensor(
            action_block,
            shape=expected,
            name="action_block",
            device=self._device,
            dtype=self._dtype,
        )
        for predictor in (self.factual_predictor, self.no_action_predictor):
            if predictor is not None and int(getattr(predictor, "action_dim")) != self._action_dim:
                raise ValueError("executed action dimension does not match A3 predictor")
        self._actions.append(action)
        self._pending_action = False


__all__ = [
    "ANCHOR_POLICY_BRIDGE_SEMANTICS",
    "CAUSAL_KNOWN_DROP_METHODS",
    "MIN_VALID_WARMUP_DECISIONS",
    "POLICY_HISTORY_STEPS",
    "CausalDecision",
    "CausalKnownDropBridgeRuntime",
    "LoadedKnownDropBridgeModels",
    "PolicyTokenOverride",
    "geometry_hold_policy_token",
    "load_anchor_policy_bridges",
    "load_known_drop_bridge_models",
]
