"""Training data for policies that consume a P1 complementary posterior.

The frozen P1 observer was previously injected into a policy trained only on
clean measurements.  This module builds matched behavior-cloning examples in
which the controller sees either the native measurement, the factual P1
posterior, or the matched no-action posterior.  The expert action target and
all non-anchor policy inputs stay identical across the three variants.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Final

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from fcpg.lewm_known_drop_bridge_runtime import geometry_hold_policy_token
from fcpg.lewm_p1_native_complementary import P1_NATIVE_COMPLEMENTARY_GAINS


POSTERIOR_POLICY_METHODS: Final = ("measurement", "no_action", "factual")
POSTERIOR_POLICY_RAW_STEPS: Final = 5
POSTERIOR_POLICY_HISTORY: Final = 3
POSTERIOR_POLICY_MAX_MISSING: Final = len(P1_NATIVE_COMPLEMENTARY_GAINS)
POSTERIOR_POLICY_FROZEN_PREFIXES: Final = (
    "vision_encoder.",
    "grounding_adapter.",
    "global_visual_projection.",
    "robot_projection.",
    "fcpg_adapter.",
)


class PosteriorTokenResidualAdapter(nn.Module):
    """Bounded per-anchor residual that preserves a frozen controller."""

    def __init__(
        self,
        *,
        token_dim: int,
        num_keypoints: int,
        hidden_dim: int = 128,
        identity_dim: int = 16,
        residual_cap: float = 0.25,
    ) -> None:
        super().__init__()
        if token_dim <= 0 or num_keypoints <= 0 or hidden_dim <= 0 or identity_dim <= 0:
            raise ValueError("adapter dimensions must be positive")
        if not 0.0 < residual_cap <= 1.0:
            raise ValueError("residual_cap must lie in (0,1]")
        self.token_dim = int(token_dim)
        self.num_keypoints = int(num_keypoints)
        self.residual_cap = float(residual_cap)
        self.normalization = nn.LayerNorm(self.token_dim)
        self.anchor_identity = nn.Embedding(self.num_keypoints, int(identity_dim))
        self.residual = nn.Sequential(
            nn.Linear(self.token_dim + int(identity_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.token_dim),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        token: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token.ndim != 4:
            raise ValueError("posterior token must have shape [B,T,K,D]")
        if token.shape[-2:] != (self.num_keypoints, self.token_dim):
            raise ValueError("posterior token anchor or feature dimension is wrong")
        if not torch.is_floating_point(token):
            raise TypeError("posterior token must be floating point")
        if mask is not None:
            if mask.shape != token.shape[:-1] or mask.dtype != torch.bool:
                raise ValueError("posterior mask must be bool with shape [B,T,K]")
        anchor = torch.arange(self.num_keypoints, device=token.device)
        identity = self.anchor_identity(anchor).to(dtype=token.dtype)
        identity = identity.view(1, 1, self.num_keypoints, -1).expand(
            *token.shape[:-1], -1
        )
        features = torch.cat((self.normalization(token), identity), dim=-1)
        raw = self.residual(features)
        delta = self.residual_cap * torch.tanh(raw / self.residual_cap)
        if mask is not None:
            delta = delta * mask.unsqueeze(-1)
        return token + delta


class PosteriorTokenAdapterPolicy(nn.Module):
    """Deploy a token adapter while delegating all policy geometry APIs."""

    def __init__(self, base_policy: nn.Module, adapter: PosteriorTokenResidualAdapter) -> None:
        super().__init__()
        self.base_policy = base_policy
        self.adapter = adapter

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = super().__getattr__("base_policy")
            return getattr(base, name)

    def _adapt(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result = dict(batch)
        token = result.get("grounded_token_override")
        mask = result.get("grounded_token_override_mask")
        if token is not None or mask is not None:
            if token is None or mask is None:
                raise KeyError("grounded token override requires token and mask")
            result["grounded_token_override"] = self.adapter(token, mask)
        return result

    def forward(self, batch: Mapping[str, torch.Tensor]):
        return self.base_policy(self._adapt(batch))

    @torch.inference_mode()
    def predict_action_chunk(self, batch: Mapping[str, torch.Tensor]):
        return self.base_policy.predict_action_chunk(self._adapt(batch))


def merge_posterior_trained_state(
    base_state: Mapping[str, torch.Tensor],
    trained_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Serialize trained controller weights without materializing frozen vision.

    Legacy action-policy checkpoints intentionally omit the frozen ViT and FCPG
    modules.  Reconstructing a policy with a temporary backbone and serializing
    its full ``state_dict`` would silently make that temporary backbone part of
    the new checkpoint.  Preserve the base checkpoint's sparse state contract
    and replace only modules that posterior-aware training is allowed to update.
    """

    missing = [name for name in base_state if name not in trained_state]
    if missing:
        raise ValueError(f"trained state is missing base keys: {missing[:5]}")
    result: dict[str, torch.Tensor] = {}
    for name, base_value in base_state.items():
        frozen = any(name.startswith(prefix) for prefix in POSTERIOR_POLICY_FROZEN_PREFIXES)
        value = base_value if frozen else trained_state[name]
        result[name] = value.detach().cpu().clone()
    return result


@dataclass(frozen=True)
class PosteriorPolicyCache:
    """Decision-level posterior tokens aligned to source frame indices."""

    anchor_index: np.ndarray
    episode_id: np.ndarray
    schedule_id: np.ndarray
    decision_index: np.ndarray
    drop_active: np.ndarray
    missing_age: np.ndarray
    factual_token: np.ndarray
    no_action_token: np.ndarray

    def __len__(self) -> int:
        return int(self.anchor_index.size)

    def validate(self) -> None:
        count = len(self)
        vectors = {
            "episode_id": self.episode_id,
            "schedule_id": self.schedule_id,
            "decision_index": self.decision_index,
            "drop_active": self.drop_active,
            "missing_age": self.missing_age,
        }
        for name, value in vectors.items():
            if np.asarray(value).shape != (count,):
                raise ValueError(f"{name} must have shape {(count,)}")
        if self.anchor_index.dtype.kind not in "iu":
            raise TypeError("anchor_index must be integral")
        if self.drop_active.dtype != np.bool_:
            raise TypeError("drop_active must be boolean")
        if bool(np.any(self.missing_age < 0)):
            raise ValueError("missing_age cannot be negative")
        if not np.array_equal(self.drop_active, self.missing_age > 0):
            raise ValueError("drop_active and missing_age disagree")
        if bool(np.any(self.missing_age > POSTERIOR_POLICY_MAX_MISSING)):
            raise ValueError("cache exceeds the frozen P1 prediction horizon")
        if self.factual_token.shape != self.no_action_token.shape:
            raise ValueError("factual and no-action token shapes differ")
        if self.factual_token.ndim != 3 or self.factual_token.shape[0] != count:
            raise ValueError("posterior tokens must have shape [N,K,D]")
        if self.factual_token.dtype.kind != "f" or self.no_action_token.dtype.kind != "f":
            raise TypeError("posterior tokens must be floating point")
        if not bool(
            np.isfinite(self.factual_token).all()
            and np.isfinite(self.no_action_token).all()
        ):
            raise ValueError("posterior cache contains non-finite tokens")

    def save(self, path: str | Path) -> None:
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            schema_version=np.asarray([1], dtype=np.int64),
            anchor_index=np.asarray(self.anchor_index, dtype=np.int64),
            episode_id=np.asarray(self.episode_id, dtype=np.int64),
            schedule_id=np.asarray(self.schedule_id, dtype=np.int64),
            decision_index=np.asarray(self.decision_index, dtype=np.int64),
            drop_active=np.asarray(self.drop_active, dtype=np.bool_),
            missing_age=np.asarray(self.missing_age, dtype=np.int64),
            factual_token=np.asarray(self.factual_token, dtype=np.float16),
            no_action_token=np.asarray(self.no_action_token, dtype=np.float16),
        )


def load_posterior_policy_cache(path: str | Path) -> PosteriorPolicyCache:
    with np.load(Path(path), allow_pickle=False) as archive:
        if (
            "schema_version" not in archive.files
            or archive["schema_version"].shape != (1,)
            or int(archive["schema_version"][0]) != 1
        ):
            raise ValueError("unsupported posterior-policy cache schema")
        cache = PosteriorPolicyCache(
            anchor_index=np.asarray(archive["anchor_index"], dtype=np.int64),
            episode_id=np.asarray(archive["episode_id"], dtype=np.int64),
            schedule_id=np.asarray(archive["schedule_id"], dtype=np.int64),
            decision_index=np.asarray(archive["decision_index"], dtype=np.int64),
            drop_active=np.asarray(archive["drop_active"], dtype=np.bool_),
            missing_age=np.asarray(archive["missing_age"], dtype=np.int64),
            factual_token=np.asarray(archive["factual_token"], dtype=np.float16),
            no_action_token=np.asarray(archive["no_action_token"], dtype=np.float16),
        )
    cache.validate()
    return cache


def sample_drop_schedules(
    *,
    episode_count: int,
    decision_count: int,
    schedules_per_episode: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample bounded, task-time-independent drop windows."""

    if episode_count <= 0 or schedules_per_episode <= 0:
        raise ValueError("episode and schedule counts must be positive")
    if decision_count < POSTERIOR_POLICY_HISTORY + 3:
        raise ValueError("episodes are too short for a causal P1 interruption")
    rng = np.random.default_rng(int(seed))
    duration = rng.integers(
        2,
        POSTERIOR_POLICY_MAX_MISSING + 1,
        size=(episode_count, schedules_per_episode),
        dtype=np.int64,
    )
    onset = np.empty_like(duration)
    for episode in range(episode_count):
        for schedule in range(schedules_per_episode):
            latest = decision_count - int(duration[episode, schedule]) - 1
            if latest < POSTERIOR_POLICY_HISTORY:
                raise ValueError("sampled duration leaves no valid recovery decision")
            onset[episode, schedule] = rng.integers(
                POSTERIOR_POLICY_HISTORY,
                latest + 1,
            )
    return onset, duration


def _predict_last(
    model: nn.Module,
    history: torch.Tensor,
    action: torch.Tensor,
    *,
    batch_size: int,
    amp: bool,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for start in range(0, history.shape[0], batch_size):
        stop = min(start + batch_size, history.shape[0])
        with torch.autocast(
            device_type=history.device.type,
            dtype=torch.float16,
            enabled=amp and history.device.type == "cuda",
        ):
            predicted = model.predict(history[start:stop], action[start:stop])
        rows.append(predicted[:, -1].float())
    return torch.cat(rows, dim=0)


@torch.inference_mode()
def build_posterior_policy_cache(
    *,
    policy: nn.Module,
    factual_predictor: nn.Module,
    no_action_predictor: nn.Module,
    source_indices: np.ndarray,
    episode_ids: np.ndarray,
    frame_ids: np.ndarray,
    sampled_visual: np.ndarray,
    xyz: np.ndarray,
    actions: np.ndarray,
    schedules_per_episode: int,
    schedule_seed: int,
    device: str | torch.device,
    inference_batch_size: int = 256,
    amp: bool = True,
) -> PosteriorPolicyCache:
    """Roll out matched recursive posteriors over complete expert episodes."""

    target_device = torch.device(device)
    for name, model in (
        ("policy", policy),
        ("factual predictor", factual_predictor),
        ("no-action predictor", no_action_predictor),
    ):
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError(f"{name} must be frozen and in eval mode")
    source_indices = np.asarray(source_indices, dtype=np.int64)
    episode_ids = np.asarray(episode_ids, dtype=np.int64)
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    sampled_visual = np.asarray(sampled_visual, dtype=np.float32)
    xyz = np.asarray(xyz, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    count = source_indices.size
    if any(value.shape[0] != count for value in (episode_ids, frame_ids, sampled_visual, xyz, actions)):
        raise ValueError("source arrays must share their leading frame axis")
    if actions.shape[1:] != (4,):
        raise ValueError("actions must have shape [N,4]")

    episodes = np.unique(episode_ids)
    decision_sources: list[np.ndarray] = []
    for episode in episodes.tolist():
        selected = np.flatnonzero(episode_ids == episode)
        order = selected[np.argsort(frame_ids[selected])]
        ordered_frames = frame_ids[order]
        if not np.array_equal(ordered_frames, np.arange(ordered_frames.size)):
            raise ValueError(f"episode {episode} does not contain contiguous frames from zero")
        if ordered_frames.size % POSTERIOR_POLICY_RAW_STEPS:
            raise ValueError("episode length must be divisible by five raw steps")
        decision_sources.append(order[::POSTERIOR_POLICY_RAW_STEPS])
    decision_counts = {value.size for value in decision_sources}
    if len(decision_counts) != 1:
        raise ValueError("all episodes must have equal decision counts")
    decision_count = decision_counts.pop()
    episode_count = len(decision_sources)
    decision_rows = np.stack(decision_sources)

    action_blocks = np.empty((episode_count, decision_count, 20), dtype=np.float32)
    for episode_index, decision_row in enumerate(decision_rows):
        first = int(decision_row[0])
        episode_frames = np.flatnonzero(episode_ids == episodes[episode_index])
        ordered = episode_frames[np.argsort(frame_ids[episode_frames])]
        del first
        action_blocks[episode_index] = actions[ordered].reshape(decision_count, 20)

    decision_visual = sampled_visual[decision_rows]
    decision_xyz = xyz[decision_rows]
    flat_visual = torch.from_numpy(decision_visual.reshape(-1, *decision_visual.shape[2:])).to(target_device)
    flat_xyz = torch.from_numpy(decision_xyz.reshape(-1, *decision_xyz.shape[2:])).to(target_device)
    native_chunks: list[torch.Tensor] = []
    for start in range(0, flat_visual.shape[0], inference_batch_size):
        stop = min(start + inference_batch_size, flat_visual.shape[0])
        with torch.autocast(
            device_type=target_device.type,
            dtype=torch.float16,
            enabled=amp and target_device.type == "cuda",
        ):
            token = geometry_hold_policy_token(
                policy,
                flat_visual[start:stop],
                flat_xyz[start:stop],
            )
        native_chunks.append(token.float())
    native = torch.cat(native_chunks).reshape(
        episode_count,
        decision_count,
        decision_visual.shape[2],
        -1,
    )

    onset, duration = sample_drop_schedules(
        episode_count=episode_count,
        decision_count=decision_count,
        schedules_per_episode=schedules_per_episode,
        seed=schedule_seed,
    )
    repeats = schedules_per_episode
    native = native.repeat_interleave(repeats, dim=0)
    visual = torch.from_numpy(decision_visual).to(target_device).repeat_interleave(repeats, dim=0)
    current_xyz = torch.from_numpy(decision_xyz).to(target_device).repeat_interleave(repeats, dim=0)
    transition = torch.from_numpy(action_blocks).to(target_device).repeat_interleave(repeats, dim=0)
    flat_onset = torch.from_numpy(onset.reshape(-1)).to(target_device)
    flat_duration = torch.from_numpy(duration.reshape(-1)).to(target_device)
    row_count = int(native.shape[0])

    factual_history = torch.empty(
        row_count, POSTERIOR_POLICY_HISTORY, native.shape[2], native.shape[3],
        device=target_device, dtype=torch.float32,
    )
    no_action_history = torch.empty_like(factual_history)
    last_valid_visual = visual[:, 0].clone()
    factual_output: list[torch.Tensor] = []
    no_action_output: list[torch.Tensor] = []
    active_output: list[torch.Tensor] = []
    age_output: list[torch.Tensor] = []
    gains = torch.tensor(P1_NATIVE_COMPLEMENTARY_GAINS, device=target_device)

    for decision in range(decision_count):
        active = (decision >= flat_onset) & (decision < flat_onset + flat_duration)
        age = torch.where(active, decision - flat_onset + 1, torch.zeros_like(flat_onset))
        factual_current = native[:, decision].float().clone()
        no_action_current = factual_current.clone()
        if decision >= POSTERIOR_POLICY_HISTORY and bool(active.any()):
            selected = torch.nonzero(active, as_tuple=False).flatten()
            action_window = transition[selected, decision - 3 : decision]
            factual_prior = _predict_last(
                factual_predictor,
                factual_history[selected],
                action_window,
                batch_size=inference_batch_size,
                amp=amp,
            )
            no_action_prior = _predict_last(
                no_action_predictor,
                no_action_history[selected],
                action_window,
                batch_size=inference_batch_size,
                amp=amp,
            )
            geometry_chunks: list[torch.Tensor] = []
            for start in range(0, selected.numel(), inference_batch_size):
                subset = selected[start : start + inference_batch_size]
                with torch.autocast(
                    device_type=target_device.type,
                    dtype=torch.float16,
                    enabled=amp and target_device.type == "cuda",
                ):
                    geometry = geometry_hold_policy_token(
                        policy,
                        last_valid_visual[subset],
                        current_xyz[subset, decision],
                    )
                geometry_chunks.append(geometry.float())
            geometry = torch.cat(geometry_chunks)
            gain = gains[age[selected] - 1].view(-1, 1, 1)
            factual_current[selected] = geometry + gain * (factual_prior - geometry)
            no_action_current[selected] = geometry + gain * (no_action_prior - geometry)

        valid = ~active
        if bool(valid.any()):
            last_valid_visual[valid] = visual[valid, decision]
        if decision < POSTERIOR_POLICY_HISTORY:
            factual_history[:, decision] = factual_current
            no_action_history[:, decision] = no_action_current
        else:
            factual_history = torch.cat(
                (factual_history[:, 1:], factual_current[:, None]), dim=1
            )
            no_action_history = torch.cat(
                (no_action_history[:, 1:], no_action_current[:, None]), dim=1
            )
        factual_output.append(factual_current.cpu().to(torch.float16))
        no_action_output.append(no_action_current.cpu().to(torch.float16))
        active_output.append(active.cpu())
        age_output.append(age.cpu())

    factual_array = torch.stack(factual_output, dim=1).numpy()
    no_action_array = torch.stack(no_action_output, dim=1).numpy()
    active_array = torch.stack(active_output, dim=1).numpy()
    age_array = torch.stack(age_output, dim=1).numpy()
    source_decisions = source_indices[decision_rows]
    source_repeated = np.repeat(source_decisions, repeats, axis=0)
    episode_repeated = np.repeat(episodes, repeats)
    schedule_repeated = np.tile(np.arange(repeats, dtype=np.int64), episode_count)
    decision_repeated = np.broadcast_to(
        np.arange(decision_count, dtype=np.int64),
        (row_count, decision_count),
    )
    cache = PosteriorPolicyCache(
        anchor_index=source_repeated.reshape(-1),
        episode_id=np.repeat(episode_repeated[:, None], decision_count, axis=1).reshape(-1),
        schedule_id=np.repeat(schedule_repeated[:, None], decision_count, axis=1).reshape(-1),
        decision_index=decision_repeated.reshape(-1),
        drop_active=active_array.reshape(-1),
        missing_age=age_array.reshape(-1),
        factual_token=factual_array.reshape(-1, factual_array.shape[-2], factual_array.shape[-1]),
        no_action_token=no_action_array.reshape(-1, no_action_array.shape[-2], no_action_array.shape[-1]),
    )
    cache.validate()
    return cache


class PosteriorAwarePolicyDataset(Dataset[dict[str, torch.Tensor]]):
    """Wrap a policy dataset with method-specific posterior token overrides."""

    def __init__(
        self,
        base: Dataset,
        cache: PosteriorPolicyCache,
        *,
        method: str,
        policy_history_steps: int = 2,
    ) -> None:
        if method not in POSTERIOR_POLICY_METHODS:
            raise ValueError(f"unknown posterior-policy method {method!r}")
        if policy_history_steps <= 0:
            raise ValueError("policy_history_steps must be positive")
        cache.validate()
        if not hasattr(base, "anchors") or not hasattr(base, "episodes"):
            raise TypeError("base dataset must expose episode-aware anchors")
        source_to_item: dict[int, int] = {}
        for item, (episode_index, position) in enumerate(base.anchors):
            source = int(base.episodes[episode_index][position])
            source_to_item[source] = item
        selected = [
            row for row, source in enumerate(cache.anchor_index.tolist())
            if int(source) in source_to_item
        ]
        if not selected:
            raise ValueError("posterior cache has no rows in the base split")
        self.base = base
        self.cache = cache
        self.method = method
        self.policy_history_steps = int(policy_history_steps)
        self.rows = np.asarray(selected, dtype=np.int64)
        self.base_items = np.asarray(
            [source_to_item[int(cache.anchor_index[row])] for row in selected],
            dtype=np.int64,
        )
        for name in ("obs_dim", "action_dim", "num_keypoints", "image_size", "policy_visual_dim"):
            if hasattr(base, name):
                setattr(self, name, getattr(base, name))

    def __len__(self) -> int:
        return int(self.rows.size)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row = int(self.rows[int(item)])
        result = dict(self.base[int(self.base_items[int(item)])])
        result["posterior_drop_active"] = torch.tensor(
            bool(self.cache.drop_active[row]), dtype=torch.bool
        )
        result["posterior_missing_age"] = torch.tensor(
            int(self.cache.missing_age[row]), dtype=torch.long
        )
        if self.method != "measurement":
            token = (
                self.cache.factual_token[row]
                if self.method == "factual"
                else self.cache.no_action_token[row]
            )
            repeated = np.broadcast_to(
                token[None],
                (self.policy_history_steps, *token.shape),
            ).copy()
            result["grounded_token_override"] = torch.from_numpy(repeated)
            result["grounded_token_override_mask"] = torch.full(
                (self.policy_history_steps, token.shape[0]),
                bool(self.cache.drop_active[row]),
                dtype=torch.bool,
            )
        return result


__all__ = [
    "POSTERIOR_POLICY_METHODS",
    "POSTERIOR_POLICY_FROZEN_PREFIXES",
    "PosteriorTokenAdapterPolicy",
    "PosteriorTokenResidualAdapter",
    "PosteriorAwarePolicyDataset",
    "PosteriorPolicyCache",
    "build_posterior_policy_cache",
    "load_posterior_policy_cache",
    "merge_posterior_trained_state",
    "sample_drop_schedules",
]
