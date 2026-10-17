#!/usr/bin/env python3
"""Train the fixed-capacity G1A candidate selector development probes.

This script consumes the partitioned cache produced by
``build_g1a_selector_cache.py``.  Deployment inputs and offline labels are
loaded from separate files and validated against fixed key whitelists before
any model is constructed.  The three learned modes share exactly the same
parameterization; mode gates only determine which legal deployment fields are
read.

The GroundedLeWM prior remains a 192-D summary prediction.  Structured anchor
tokens are current/history observations, not predicted structured dynamics.
Consequently, this development probe must not be described as a structured
future-anchor world model or as a rollout/control result.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.lewm_counterfactual import spearman_rank_correlation  # noqa: E402


DEFAULT_CACHE_DIR = (
    "outputs/g1a_selector_cache/"
    "pick_rot10_r64g17_l16m16_online10k_matched_history_native_labels_"
    "splitlocal_v3_development"
)
DEFAULT_OUTPUT_DIR = (
    "outputs/g1a_selector/"
    "pick_rot10_matched_history_native_labels_splitlocal_v3_development"
)
MODES = ("structured_full", "structured_current_only", "summary_predictive")
HISTORY_VARIANTS = ("wide_top1", "nominal")
PAIRED_BOOTSTRAP_RESAMPLES = 10_000
PAIRED_BOOTSTRAP_SEED = 20260715
EXPECTED_CACHE_SCHEMA_VERSION = 3
EXPECTED_CACHE_TYPE = "fcpg_g1a_selector_development_v3"


def _history_variant_deployment_keys(suffix: str) -> frozenset[str]:
    return frozenset(
        {
            f"history_grounding_points_px_{suffix}",
            f"history_grounding_valid_{suffix}",
            f"history_summary_token_{suffix}",
            f"history_summary_embedding_{suffix}",
            f"history_anchor_tokens_{suffix}",
            f"predicted_current_summary_prior_{suffix}",
        }
    )

DEPLOYMENT_ALLOWED_KEYS = frozenset(
    {
        "task_name",
        "episode_id",
        "frame_id",
        "raw_index",
        "split",
        "candidate_points_px",
        "candidate_valid",
        "candidate_grid_indices",
        "candidate_joint_visual_score",
        "candidate_visual_descriptor",
        "candidate_geometry",
        "candidate_summary_token",
        "candidate_summary_embedding",
        "candidate_anchor_tokens",
        "history_raw_indices",
        "history_frame_ids",
        "history_grounding_points_px",
        "history_grounding_valid",
        "history_action_blocks",
        "history_summary_token",
        "history_summary_embedding",
        "history_anchor_tokens",
        "predicted_current_summary_prior",
    }
).union(*(_history_variant_deployment_keys(value) for value in HISTORY_VARIANTS))
OFFLINE_ALLOWED_KEYS = frozenset(
    {
        "task_name",
        "episode_id",
        "frame_id",
        "raw_index",
        "policy_seed",
        "action_xyz_normalized_mse_h8",
        "shuffled_current_action_blocks",
        "shuffled_action_source_index",
        "shuffled_predicted_current_summary_prior",
        "shuffled_predicted_current_summary_prior_wide_top1",
        "shuffled_predicted_current_summary_prior_nominal",
    }
)
COMMON_REQUIRED_KEYS = frozenset(
    {
        "task_name",
        "episode_id",
        "frame_id",
        "raw_index",
        "split",
        "candidate_valid",
        "candidate_joint_visual_score",
    }
)
MODE_INPUT_KEYS = {
    "structured_full": frozenset(
        {
            "candidate_anchor_tokens",
            "candidate_geometry",
            "candidate_joint_visual_score",
            "history_anchor_tokens",
            "history_action_blocks",
            "predicted_current_summary_prior",
        }
    ),
    "structured_current_only": frozenset(
        {
            "candidate_anchor_tokens",
            "candidate_geometry",
            "candidate_joint_visual_score",
        }
    ),
    "summary_predictive": frozenset(
        {
            "candidate_summary_embedding",
            "predicted_current_summary_prior",
        }
    ),
}
OFFLINE_REQUIRED_KEYS = frozenset(
    {
        "task_name",
        "episode_id",
        "frame_id",
        "raw_index",
        "policy_seed",
        "action_xyz_normalized_mse_h8",
        "shuffled_current_action_blocks",
        "shuffled_action_source_index",
        "shuffled_predicted_current_summary_prior",
    }
)
FORBIDDEN_MODEL_INPUT_SUBSTRINGS = (
    "q_true",
    "qtrue",
    "oracle",
    "teacher",
    "privileged",
    "ground_truth",
    "reward",
    "success",
    "label",
    "future",
    "target",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as raw:
        return {key: np.asarray(raw[key]) for key in raw.files}


def _exact_whitelist(
    values: Mapping[str, np.ndarray], allowed: frozenset[str], partition: str
) -> None:
    unexpected = sorted(set(values) - allowed)
    if unexpected:
        raise ValueError(f"{partition} contains non-whitelisted keys: {unexpected}")


def _assert_legal_model_keys(keys: Sequence[str]) -> None:
    for key in keys:
        lowered = key.lower()
        if any(value in lowered for value in FORBIDDEN_MODEL_INPUT_SUBSTRINGS):
            raise ValueError(f"privileged or future field cannot enter selector: {key}")


@dataclass(frozen=True)
class SelectorCache:
    deployment: Mapping[str, np.ndarray]
    offline: Mapping[str, np.ndarray]
    policy_seeds: tuple[int, ...]
    sample_count: int
    candidate_count: int
    anchor_count: int
    latent_dim: int
    geometry_dim: int
    action_block_dim: int
    metadata: Mapping[str, Any] | None = None


def candidate_mask(deployment: Mapping[str, np.ndarray]) -> np.ndarray:
    """Reduce per-anchor validity to one strict hypothesis-validity mask."""

    value = np.asarray(deployment["candidate_valid"])
    if value.ndim != 3:
        raise ValueError("candidate_valid must have shape [sample,candidate,anchor]")
    return np.all(value, axis=-1)


def validate_selector_cache(
    deployment: Mapping[str, np.ndarray], offline: Mapping[str, np.ndarray]
) -> SelectorCache:
    """Validate partitioning, axes, episode splits, and input legality."""

    _exact_whitelist(deployment, DEPLOYMENT_ALLOWED_KEYS, "deployment_inputs")
    _exact_whitelist(offline, OFFLINE_ALLOWED_KEYS, "offline_labels")
    missing_deployment = sorted(COMMON_REQUIRED_KEYS - set(deployment))
    missing_offline = sorted(OFFLINE_REQUIRED_KEYS - set(offline))
    if missing_deployment or missing_offline:
        raise KeyError(
            f"missing selector cache keys: deployment={missing_deployment}, "
            f"offline={missing_offline}"
        )
    for mode in MODES:
        missing = sorted(MODE_INPUT_KEYS[mode] - set(deployment))
        if missing:
            raise KeyError(f"{mode} is missing deployment fields: {missing}")
        _assert_legal_model_keys(sorted(MODE_INPUT_KEYS[mode]))

    task = np.asarray(deployment["task_name"])
    episode = np.asarray(deployment["episode_id"])
    frame = np.asarray(deployment["frame_id"])
    raw_index = np.asarray(deployment["raw_index"])
    split = np.asarray(deployment["split"])
    candidate_valid = np.asarray(deployment["candidate_valid"])
    if task.ndim != 1:
        raise ValueError("task_name must be one-dimensional")
    sample_count = len(task)
    if sample_count == 0:
        raise ValueError("selector cache is empty")
    for name, value in (
        ("episode_id", episode),
        ("frame_id", frame),
        ("raw_index", raw_index),
        ("split", split),
    ):
        if value.shape != (sample_count,):
            raise ValueError(f"{name} must have shape [sample]")
    if set(str(value) for value in np.unique(split)) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("selector cache requires train/validation/test splits")
    if candidate_valid.ndim != 3 or candidate_valid.shape[0] != sample_count:
        raise ValueError("candidate_valid must have shape [sample,candidate,anchor]")
    if candidate_valid.dtype != np.bool_:
        raise TypeError("candidate_valid must be boolean")
    candidate_count = int(candidate_valid.shape[1])
    hypothesis_valid = np.all(candidate_valid, axis=-1)
    if candidate_count < 2 or np.any(hypothesis_valid.sum(axis=1) < 2):
        raise ValueError("each snapshot requires at least two valid candidates")

    identities = np.stack((task.astype(str), episode.astype(str)), axis=1)
    split_by_episode: dict[tuple[str, str], str] = {}
    for identity, split_name in zip(identities, split.astype(str), strict=True):
        key = (str(identity[0]), str(identity[1]))
        previous = split_by_episode.setdefault(key, str(split_name))
        if previous != str(split_name):
            raise ValueError("one physical episode crosses selector splits")

    offline_identity_keys = ("task_name", "episode_id", "frame_id", "raw_index")
    for key in offline_identity_keys:
        if not np.array_equal(np.asarray(deployment[key]), np.asarray(offline[key])):
            raise ValueError(f"deployment/offline identity mismatch for {key}")

    seeds = np.asarray(offline["policy_seed"])
    labels = np.asarray(offline["action_xyz_normalized_mse_h8"])
    if seeds.ndim != 1 or len(seeds) != 3 or len(np.unique(seeds)) != 3:
        raise ValueError("G1A requires exactly three unique policy seeds")
    if labels.shape != (sample_count, len(seeds), candidate_count):
        raise ValueError("action divergence labels must be [sample,policy,candidate]")
    if not np.isfinite(labels).all() or np.any(labels < 0):
        raise ValueError("action divergence labels must be finite and non-negative")

    anchors = np.asarray(deployment["candidate_anchor_tokens"])
    geometry = np.asarray(deployment["candidate_geometry"])
    summary = np.asarray(deployment["candidate_summary_embedding"])
    history_anchor = np.asarray(deployment["history_anchor_tokens"])
    actions = np.asarray(deployment["history_action_blocks"])
    prior = np.asarray(deployment["predicted_current_summary_prior"])
    shuffled_prior = np.asarray(offline["shuffled_predicted_current_summary_prior"])
    if anchors.ndim != 4 or anchors.shape[:2] != (sample_count, candidate_count):
        raise ValueError("candidate_anchor_tokens must be [sample,candidate,anchor,latent]")
    anchor_count, latent_dim = int(anchors.shape[2]), int(anchors.shape[3])
    if anchor_count <= 0 or latent_dim <= 0:
        raise ValueError("anchor/latent dimensions must be positive")
    if candidate_valid.shape[2] != anchor_count:
        raise ValueError("candidate validity and anchor-token axes differ")
    if geometry.ndim != 4 or geometry.shape[:3] != (
        sample_count,
        candidate_count,
        anchor_count,
    ):
        raise ValueError("candidate_geometry must be [sample,candidate,anchor,feature]")
    if summary.shape != (sample_count, candidate_count, latent_dim):
        raise ValueError("candidate summary embedding shape changed")
    if history_anchor.ndim != 4 or history_anchor.shape[0] != sample_count:
        raise ValueError("history_anchor_tokens must be [sample,history,anchor,latent]")
    if history_anchor.shape[2:] != (anchor_count, latent_dim):
        raise ValueError("history/current anchor axes differ")
    if actions.ndim != 3 or actions.shape[:2] != history_anchor.shape[:2]:
        raise ValueError("history action and state axes differ")
    shuffled_actions = np.asarray(offline["shuffled_current_action_blocks"])
    if shuffled_actions.shape != (sample_count, actions.shape[-1]):
        raise ValueError(
            "shuffled_current_action_blocks must have shape [sample,action_block_dim]"
        )
    if not np.issubdtype(shuffled_actions.dtype, np.floating):
        raise TypeError("shuffled_current_action_blocks must have floating dtype")
    if shuffled_actions.dtype != actions.dtype:
        raise TypeError(
            "shuffled_current_action_blocks dtype must match history_action_blocks"
        )
    if not np.isfinite(shuffled_actions).all():
        raise ValueError("shuffled_current_action_blocks must be finite")

    shuffled_source = np.asarray(offline["shuffled_action_source_index"])
    if shuffled_source.shape != (sample_count,):
        raise ValueError("shuffled_action_source_index must have shape [sample]")
    if not np.issubdtype(shuffled_source.dtype, np.integer):
        raise TypeError("shuffled_action_source_index must have integer dtype")
    shuffled_source = shuffled_source.astype(np.int64, copy=False)
    if np.any(shuffled_source < 0) or np.any(shuffled_source >= sample_count):
        raise ValueError("shuffled_action_source_index lies outside the cache")
    if len(np.unique(shuffled_source)) != sample_count:
        raise ValueError("shuffled_action_source_index must be a permutation")
    if np.any(shuffled_source == np.arange(sample_count)):
        raise ValueError("shuffled_action_source_index must be a derangement")
    if not np.array_equal(task[shuffled_source], task) or not np.array_equal(
        frame[shuffled_source], frame
    ):
        raise ValueError("shuffled action source must preserve task and frame")
    if not np.array_equal(split[shuffled_source], split):
        raise ValueError("shuffled action source must remain within its data split")
    if np.any(episode[shuffled_source] == episode):
        raise ValueError("shuffled action source must use another physical episode")
    source_current_actions = actions[shuffled_source, -1]
    if not np.array_equal(shuffled_actions, source_current_actions):
        raise ValueError(
            "shuffled_current_action_blocks must exactly match source final history actions"
        )
    if prior.shape != (sample_count, latent_dim) or shuffled_prior.shape != prior.shape:
        raise ValueError("predicted prior must have shape [sample,latent]")
    visual_score = np.asarray(deployment["candidate_joint_visual_score"])
    if visual_score.shape != (sample_count, candidate_count):
        raise ValueError("candidate visual score must be [sample,candidate]")

    float_fields = (
        anchors,
        geometry,
        summary,
        history_anchor,
        actions,
        prior,
        shuffled_prior,
        visual_score,
    )
    for suffix in HISTORY_VARIANTS:
        variant_keys = _history_variant_deployment_keys(suffix)
        present = variant_keys.intersection(deployment)
        shuffled_key = f"shuffled_predicted_current_summary_prior_{suffix}"
        if present and present != variant_keys:
            raise KeyError(f"partial {suffix} history diagnostic fields are forbidden")
        if present and shuffled_key not in offline:
            raise KeyError(f"{suffix} history requires its shuffled prior diagnostic")
        if not present and shuffled_key in offline:
            raise KeyError(f"orphaned {suffix} shuffled prior diagnostic")
        if not present:
            continue
        history_length = int(history_anchor.shape[1])
        if np.asarray(deployment[f"history_grounding_points_px_{suffix}"]).shape != (
            sample_count,
            history_length,
            anchor_count,
            2,
        ):
            raise ValueError(f"{suffix} history grounding-point axes changed")
        if np.asarray(deployment[f"history_grounding_valid_{suffix}"]).shape != (
            sample_count,
            history_length,
            anchor_count,
        ):
            raise ValueError(f"{suffix} history validity axes changed")
        if np.asarray(deployment[f"history_grounding_valid_{suffix}"]).dtype != np.bool_:
            raise TypeError(f"{suffix} history grounding validity must be boolean")
        for stem in ("history_summary_token", "history_summary_embedding"):
            if np.asarray(deployment[f"{stem}_{suffix}"]).shape != (
                sample_count,
                history_length,
                latent_dim,
            ):
                raise ValueError(f"{suffix} {stem} axes changed")
        if np.asarray(deployment[f"history_anchor_tokens_{suffix}"]).shape != (
            sample_count,
            history_length,
            anchor_count,
            latent_dim,
        ):
            raise ValueError(f"{suffix} history anchor-token axes changed")
        variant_prior = np.asarray(
            deployment[f"predicted_current_summary_prior_{suffix}"]
        )
        variant_shuffled = np.asarray(offline[shuffled_key])
        if variant_prior.shape != prior.shape or variant_shuffled.shape != prior.shape:
            raise ValueError(f"{suffix} predicted-prior axes changed")
        float_fields = (*float_fields, variant_prior, variant_shuffled)
    if any(not np.isfinite(value).all() for value in float_fields):
        raise ValueError("selector cache contains non-finite model fields")
    return SelectorCache(
        deployment=deployment,
        offline=offline,
        policy_seeds=tuple(int(value) for value in seeds),
        sample_count=sample_count,
        candidate_count=candidate_count,
        anchor_count=anchor_count,
        latent_dim=latent_dim,
        geometry_dim=int(geometry.shape[-1]),
        action_block_dim=int(actions.shape[-1]),
    )


def load_selector_cache(cache_dir: Path) -> SelectorCache:
    deployment_path = cache_dir / "deployment_inputs.npz"
    offline_path = cache_dir / "offline_labels.npz"
    metadata_path = cache_dir / "metadata.json"
    if (
        not deployment_path.is_file()
        or not offline_path.is_file()
        or not metadata_path.is_file()
    ):
        raise FileNotFoundError(
            "G1A cache requires deployment_inputs.npz, offline_labels.npz, and metadata.json"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise TypeError("G1A cache metadata must be a mapping")
    if int(metadata.get("schema_version", -1)) != EXPECTED_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"G1A cache schema_version must be {EXPECTED_CACHE_SCHEMA_VERSION}"
        )
    if str(metadata.get("cache_type", "")) != EXPECTED_CACHE_TYPE:
        raise ValueError(f"G1A cache_type must be {EXPECTED_CACHE_TYPE!r}")
    artifacts = metadata.get("artifacts", {})
    for name, path in (
        ("deployment_inputs", deployment_path),
        ("offline_labels", offline_path),
    ):
        expected = artifacts.get(name, {}).get("sha256")
        if expected is not None and str(expected) != sha256_file(path):
            raise ValueError(f"G1A cache {name} SHA256 does not match metadata")
    validated = validate_selector_cache(
        _load_npz(deployment_path), _load_npz(offline_path)
    )
    if int(metadata.get("sample_count", -1)) != validated.sample_count:
        raise ValueError("G1A cache metadata sample_count changed")
    if int(metadata.get("candidate_count", -1)) != validated.candidate_count:
        raise ValueError("G1A cache metadata candidate_count changed")
    metadata_seeds = tuple(int(value) for value in metadata.get("policy_seeds", ()))
    if metadata_seeds and metadata_seeds != validated.policy_seeds:
        raise ValueError("G1A cache metadata policy_seeds changed")
    contract = metadata.get("contract", {})
    if not isinstance(contract, Mapping):
        raise TypeError("G1A cache metadata contract must be a mapping")
    primary_history = str(contract.get("history_grounding_primary", ""))
    label_history = str(contract.get("label_history_source", ""))
    if not primary_history or not label_history:
        raise ValueError("G1A cache contract must name primary and label history sources")
    if label_history != primary_history:
        raise ValueError(
            "G1A cache label history differs from deployment-primary history"
        )
    if contract.get("label_history_matches_primary") is not True:
        raise ValueError(
            "G1A cache must explicitly certify label_history_matches_primary"
        )
    shuffled_partition = contract.get("history_shuffled_prior_partition")
    if shuffled_partition is not None and shuffled_partition != "offline_labels_only":
        raise ValueError("shuffled predicted priors must remain offline-label diagnostics")
    shuffled_scope = contract.get("history_shuffled_source_scope")
    if shuffled_scope is not None and shuffled_scope != (
        "same_split_task_frame_different_physical_episode"
    ):
        raise ValueError("shuffled action sources must remain split-local")
    return replace(validated, metadata=metadata)


@dataclass(frozen=True)
class FieldStatistics:
    mean: np.ndarray
    std: np.ndarray


NORMALIZATION_REDUCTION_AXES = {
    "candidate_anchor_tokens": (0, 1, 2),
    "candidate_geometry": (0, 1, 2),
    "candidate_joint_visual_score": (0, 1),
    "candidate_summary_embedding": (0, 1),
    "history_anchor_tokens": (0, 1, 2),
    "history_action_blocks": (0, 1),
    "predicted_current_summary_prior": (0,),
}


def fit_normalizer(
    deployment: Mapping[str, np.ndarray], train_indices: np.ndarray, mode: str
) -> dict[str, FieldStatistics]:
    if mode not in MODES:
        raise ValueError(f"unknown selector mode {mode!r}")
    result: dict[str, FieldStatistics] = {}
    for key in sorted(MODE_INPUT_KEYS[mode]):
        value = np.asarray(deployment[key], dtype=np.float32)[train_indices]
        axes = NORMALIZATION_REDUCTION_AXES[key]
        mean = np.mean(value, axis=axes, keepdims=True, dtype=np.float64).astype(np.float32)
        std = np.std(value, axis=axes, keepdims=True, dtype=np.float64).astype(np.float32)
        std = np.maximum(std, np.float32(1e-6))
        result[key] = FieldStatistics(mean=mean, std=std)
    return result


def _normalized_slice(
    deployment: Mapping[str, np.ndarray],
    indices: np.ndarray,
    keys: Sequence[str],
    normalizer: Mapping[str, FieldStatistics],
) -> dict[str, Tensor]:
    _assert_legal_model_keys(keys)
    result: dict[str, Tensor] = {}
    for key in keys:
        value = np.asarray(deployment[key], dtype=np.float32)[indices]
        stats = normalizer[key]
        result[key] = torch.from_numpy((value - stats.mean) / stats.std)
    return result


class EquivariantG1ASelector(nn.Module):
    """Fixed-capacity shared candidate scorer with permutation equivariance."""

    def __init__(
        self,
        *,
        mode: str,
        anchor_count: int,
        latent_dim: int,
        geometry_dim: int,
        action_block_dim: int,
        model_dim: int = 16,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown selector mode {mode!r}")
        if min(anchor_count, latent_dim, geometry_dim, action_block_dim, model_dim, hidden_dim) <= 0:
            raise ValueError("selector dimensions must be positive")
        self.mode = mode
        self.anchor_count = int(anchor_count)
        self.latent_dim = int(latent_dim)
        self.model_dim = int(model_dim)

        # Every mode owns this same complete module set.  Mode-specific forward
        # gates leave irrelevant branches dormant, keeping parameter counts exact.
        self.anchor_projection = nn.Linear(latent_dim, model_dim)
        self.anchor_identity = nn.Parameter(torch.zeros(anchor_count, model_dim))
        self.anchor_fusion = nn.Linear(anchor_count * model_dim, model_dim)
        self.geometry_projection = nn.Linear(geometry_dim, model_dim)
        self.geometry_fusion = nn.Linear(anchor_count * model_dim, model_dim)
        self.visual_projection = nn.Linear(1, model_dim)
        self.summary_projection = nn.Linear(latent_dim, model_dim)
        self.prior_projection = nn.Linear(latent_dim, model_dim)
        self.action_projection = nn.Linear(action_block_dim, model_dim)
        self.local_fusion = nn.Linear(4 * model_dim, model_dim)
        self.context_fusion = nn.Linear(3 * model_dim, model_dim)
        self.scorer = nn.Sequential(
            nn.Linear(4 * model_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def required_keys(self) -> frozenset[str]:
        return MODE_INPUT_KEYS[self.mode]

    def _anchor_features(self, value: Tensor) -> Tensor:
        if value.shape[-2:] != (self.anchor_count, self.latent_dim):
            raise ValueError("anchor token axes changed")
        projected = self.anchor_projection(value)
        identity_shape = (1,) * (projected.ndim - 2) + self.anchor_identity.shape
        projected = projected + self.anchor_identity.reshape(identity_shape)
        return self.anchor_fusion(projected.flatten(start_dim=-2))

    def forward(self, inputs: Mapping[str, Tensor], valid: Tensor) -> Tensor:
        if set(inputs) != set(self.required_keys):
            raise ValueError(
                f"{self.mode} input contract changed: expected "
                f"{sorted(self.required_keys)}, got {sorted(inputs)}"
            )
        _assert_legal_model_keys(sorted(inputs))
        if valid.ndim != 2 or valid.dtype != torch.bool:
            raise ValueError("candidate validity must be boolean [batch,candidate]")
        batch, candidates = valid.shape
        device = valid.device
        dtype = next(self.parameters()).dtype
        zeros_candidate = torch.zeros(
            batch, candidates, self.model_dim, device=device, dtype=dtype
        )
        zeros_context = torch.zeros(batch, self.model_dim, device=device, dtype=dtype)

        if self.mode in {"structured_full", "structured_current_only"}:
            anchor = self._anchor_features(inputs["candidate_anchor_tokens"])
            geometry_projected = self.geometry_projection(inputs["candidate_geometry"])
            identity_shape = (
                (1,) * (geometry_projected.ndim - 2) + self.anchor_identity.shape
            )
            geometry_projected = geometry_projected + self.anchor_identity.reshape(
                identity_shape
            )
            geometry = self.geometry_fusion(geometry_projected.flatten(start_dim=-2))
            visual = self.visual_projection(
                inputs["candidate_joint_visual_score"].unsqueeze(-1)
            )
            summary = zeros_candidate
        else:
            anchor = zeros_candidate
            geometry = zeros_candidate
            visual = zeros_candidate
            summary = self.summary_projection(inputs["candidate_summary_embedding"])
        local = self.local_fusion(torch.cat((anchor, geometry, visual, summary), dim=-1))

        if self.mode == "structured_full":
            history = self._anchor_features(inputs["history_anchor_tokens"]).mean(dim=1)
            action = self.action_projection(inputs["history_action_blocks"]).mean(dim=1)
            prior = self.prior_projection(inputs["predicted_current_summary_prior"])
        elif self.mode == "summary_predictive":
            history = zeros_context
            action = zeros_context
            prior = self.prior_projection(inputs["predicted_current_summary_prior"])
        else:
            history = zeros_context
            action = zeros_context
            prior = zeros_context
        context = self.context_fusion(torch.cat((history, action, prior), dim=-1))
        valid_weight = valid.to(local.dtype).unsqueeze(-1)
        set_mean = (local * valid_weight).sum(dim=1) / valid_weight.sum(dim=1).clamp_min(1.0)
        context_expanded = context[:, None, :].expand(-1, candidates, -1)
        set_expanded = set_mean[:, None, :].expand(-1, candidates, -1)
        score_features = torch.cat(
            (local, set_expanded, context_expanded, local * context_expanded), dim=-1
        )
        costs = self.scorer(score_features).squeeze(-1)
        return costs.masked_fill(~valid, torch.inf)


def parameter_count(model: nn.Module) -> int:
    return int(sum(value.numel() for value in model.parameters()))


def module_path_parameter_count(model: EquivariantG1ASelector) -> int:
    """Report parameters in modules invoked by the selected mode path."""

    modules: list[nn.Module] = [model.local_fusion, model.context_fusion, model.scorer]
    standalone: list[nn.Parameter] = []
    if model.mode in {"structured_full", "structured_current_only"}:
        modules.extend(
            [
                model.anchor_projection,
                model.anchor_fusion,
                model.geometry_projection,
                model.geometry_fusion,
                model.visual_projection,
            ]
        )
        standalone.append(model.anchor_identity)
    else:
        modules.append(model.summary_projection)
    if model.mode == "structured_full":
        modules.extend([model.prior_projection, model.action_projection])
    elif model.mode == "summary_predictive":
        modules.append(model.prior_projection)
    unique = {id(value): value for module in modules for value in module.parameters()}
    unique.update({id(value): value for value in standalone})
    return int(sum(value.numel() for value in unique.values()))


def _split_indices(split: np.ndarray, name: str) -> np.ndarray:
    indices = np.flatnonzero(np.asarray(split).astype(str) == name).astype(np.int64)
    if len(indices) == 0:
        raise ValueError(f"selector split {name!r} is empty")
    return indices


def soft_label_distribution(
    labels: Tensor, valid: Tensor, *, temperature: float
) -> Tensor:
    """Average per-policy soft ranks after snapshot-wise range normalization."""

    if temperature <= 0:
        raise ValueError("label temperature must be positive")
    if labels.ndim != 3 or valid.shape != (labels.shape[0], labels.shape[2]):
        raise ValueError("labels/valid axes changed")
    mask = valid[:, None, :]
    masked_min = labels.masked_fill(~mask, torch.inf).amin(dim=-1, keepdim=True)
    masked_max = labels.masked_fill(~mask, -torch.inf).amax(dim=-1, keepdim=True)
    span = (masked_max - masked_min).clamp_min(1e-6)
    normalized = (labels - masked_min) / span
    logits = (-normalized / temperature).masked_fill(~mask, -torch.inf)
    probabilities = torch.softmax(logits, dim=-1).mean(dim=1)
    return probabilities.masked_fill(~valid, 0.0)


def listwise_kl_loss(
    predicted_cost: Tensor,
    target_probability: Tensor,
    valid: Tensor,
    *,
    temperature: float,
) -> Tensor:
    if temperature <= 0:
        raise ValueError("prediction temperature must be positive")
    logits = (-predicted_cost / temperature).masked_fill(~valid, -torch.inf)
    log_probability = torch.log_softmax(logits, dim=-1).masked_fill(~valid, 0.0)
    terms = torch.where(
        target_probability > 0,
        target_probability * (torch.log(target_probability.clamp_min(1e-12)) - log_probability),
        torch.zeros_like(target_probability),
    )
    return terms.sum(dim=-1).mean()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class TrainedSelector:
    mode: str
    model: EquivariantG1ASelector
    normalizer: Mapping[str, FieldStatistics]
    best_epoch: int
    best_validation_kl: float
    parameter_count: int
    module_path_parameter_count: int
    train_history: tuple[Mapping[str, float], ...]


def _predict_costs(
    model: EquivariantG1ASelector,
    deployment: Mapping[str, np.ndarray],
    indices: np.ndarray,
    normalizer: Mapping[str, FieldStatistics],
    *,
    device: torch.device,
    batch_size: int,
    input_overrides: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    model.eval()
    overrides = dict(input_overrides or {})
    if not set(overrides).issubset(model.required_keys):
        raise ValueError("selector control tries to override an unavailable model input")
    _assert_legal_model_keys(sorted(overrides))
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            inputs = _normalized_slice(
                deployment,
                batch_indices,
                sorted(model.required_keys),
                normalizer,
            )
            for key, override in overrides.items():
                stats = normalizer[key]
                raw = np.asarray(override, dtype=np.float32)[batch_indices]
                expected = np.asarray(deployment[key])[batch_indices].shape
                if raw.shape != expected:
                    raise ValueError(f"selector control override shape changed for {key}")
                inputs[key] = torch.from_numpy((raw - stats.mean) / stats.std)
            inputs = {key: value.to(device=device) for key, value in inputs.items()}
            valid = torch.from_numpy(candidate_mask(deployment)[batch_indices].copy()).to(
                device=device
            )
            rows.append(model(inputs, valid).cpu().numpy())
    return np.concatenate(rows, axis=0)


def _validation_kl(
    model: EquivariantG1ASelector,
    cache: SelectorCache,
    indices: np.ndarray,
    normalizer: Mapping[str, FieldStatistics],
    *,
    device: torch.device,
    batch_size: int,
    label_temperature: float,
    prediction_temperature: float,
) -> float:
    costs = _predict_costs(
        model,
        cache.deployment,
        indices,
        normalizer,
        device=device,
        batch_size=batch_size,
    )
    labels = torch.from_numpy(
        np.asarray(cache.offline["action_xyz_normalized_mse_h8"], dtype=np.float32)[indices]
    )
    valid = torch.from_numpy(candidate_mask(cache.deployment)[indices].copy())
    target = soft_label_distribution(labels, valid, temperature=label_temperature)
    return float(
        listwise_kl_loss(
            torch.from_numpy(costs),
            target,
            valid,
            temperature=prediction_temperature,
        )
    )


def train_selector(
    cache: SelectorCache,
    *,
    mode: str,
    device: torch.device,
    seed: int = 0,
    epochs: int = 80,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    model_dim: int = 16,
    hidden_dim: int = 32,
    label_temperature: float = 0.25,
    prediction_temperature: float = 1.0,
) -> TrainedSelector:
    """Train once and select the checkpoint exclusively by validation KL."""

    if mode not in MODES or epochs <= 0 or batch_size <= 0:
        raise ValueError("invalid selector mode or training budget")
    _seed_everything(seed)
    train_indices = _split_indices(cache.deployment["split"], "train")
    validation_indices = _split_indices(cache.deployment["split"], "validation")
    normalizer = fit_normalizer(cache.deployment, train_indices, mode)
    model = EquivariantG1ASelector(
        mode=mode,
        anchor_count=cache.anchor_count,
        latent_dim=cache.latent_dim,
        geometry_dim=cache.geometry_dim,
        action_block_dim=cache.action_block_dim,
        model_dim=model_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    rng = np.random.default_rng(seed)
    best_state: dict[str, Tensor] | None = None
    best_validation = float("inf")
    best_epoch = -1
    history: list[Mapping[str, float]] = []
    labels_all = np.asarray(cache.offline["action_xyz_normalized_mse_h8"], dtype=np.float32)
    valid_all = candidate_mask(cache.deployment)

    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = train_indices[rng.permutation(len(train_indices))]
        epoch_losses: list[float] = []
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start : start + batch_size]
            inputs = _normalized_slice(
                cache.deployment, indices, sorted(model.required_keys), normalizer
            )
            inputs = {key: value.to(device=device) for key, value in inputs.items()}
            valid = torch.from_numpy(valid_all[indices].copy()).to(device=device)
            labels = torch.from_numpy(labels_all[indices].copy()).to(device=device)
            target = soft_label_distribution(labels, valid, temperature=label_temperature)
            optimizer.zero_grad(set_to_none=True)
            costs = model(inputs, valid)
            loss = listwise_kl_loss(
                costs,
                target,
                valid,
                temperature=prediction_temperature,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("selector loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        validation = _validation_kl(
            model,
            cache,
            validation_indices,
            normalizer,
            device=device,
            batch_size=batch_size,
            label_temperature=label_temperature,
            prediction_temperature=prediction_temperature,
        )
        train_loss = float(np.mean(epoch_losses))
        history.append(
            {"epoch": float(epoch), "train_kl": train_loss, "validation_kl": validation}
        )
        if validation < best_validation:
            best_validation = validation
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("selector training produced no validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return TrainedSelector(
        mode=mode,
        model=model.cpu(),
        normalizer=normalizer,
        best_epoch=best_epoch,
        best_validation_kl=best_validation,
        parameter_count=parameter_count(model),
        module_path_parameter_count=module_path_parameter_count(model),
        train_history=tuple(history),
    )


def _ranking_row(predicted_cost: np.ndarray, actual_cost: np.ndarray) -> tuple[float, float]:
    if len(predicted_cost) < 2:
        raise ValueError("ranking requires at least two candidates")
    spearman = float(spearman_rank_correlation(predicted_cost, actual_cost))
    best = np.flatnonzero(np.isclose(predicted_cost, np.min(predicted_cost), atol=1e-12))
    selected_cost = float(np.mean(actual_cost[best]))
    low = float(np.min(actual_cost))
    high = float(np.max(actual_cost))
    regret = 0.0 if high - low <= 1e-12 else (selected_cost - low) / (high - low)
    return spearman, float(regret)


def evaluate_costs(
    predicted_cost: np.ndarray,
    cache: SelectorCache,
    indices: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(cache.offline["action_xyz_normalized_mse_h8"], dtype=np.float64)[
        indices
    ]
    valid = candidate_mask(cache.deployment)[indices]
    task = np.asarray(cache.deployment["task_name"]).astype(str)[indices]
    if predicted_cost.shape != (len(indices), cache.candidate_count):
        raise ValueError("predicted cost axes changed")
    rows: list[dict[str, Any]] = []
    for local, raw in enumerate(indices):
        kept = np.flatnonzero(valid[local])
        for policy_position, seed in enumerate(cache.policy_seeds):
            spearman, regret = _ranking_row(
                predicted_cost[local, kept], labels[local, policy_position, kept]
            )
            rows.append(
                {
                    "sample_index": int(raw),
                    "task": str(task[local]),
                    "policy_seed": int(seed),
                    "spearman": spearman,
                    "normalized_top1_regret": regret,
                }
            )

    def aggregate(selected: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        return {
            "count": int(len(selected)),
            "mean_spearman": float(np.mean([row["spearman"] for row in selected])),
            "mean_normalized_top1_regret": float(
                np.mean([row["normalized_top1_regret"] for row in selected])
            ),
        }

    by_task_policy: dict[str, Any] = {}
    for task_name in sorted(set(row["task"] for row in rows)):
        by_task_policy[task_name] = {}
        for seed in cache.policy_seeds:
            selected = [
                row
                for row in rows
                if row["task"] == task_name and row["policy_seed"] == seed
            ]
            by_task_policy[task_name][str(seed)] = aggregate(selected)
    by_policy = {
        str(seed): aggregate([row for row in rows if row["policy_seed"] == seed])
        for seed in cache.policy_seeds
    }
    return {
        "macro": aggregate(rows),
        "by_policy_seed": by_policy,
        "by_task_and_policy_seed": by_task_policy,
    }


def _ranking_metric_arrays(
    predicted_cost: np.ndarray,
    cache: SelectorCache,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-snapshot, per-policy ranking metrics for paired analyses."""

    labels = np.asarray(
        cache.offline["action_xyz_normalized_mse_h8"], dtype=np.float64
    )[indices]
    valid = candidate_mask(cache.deployment)[indices]
    if predicted_cost.shape != (len(indices), cache.candidate_count):
        raise ValueError("predicted cost axes changed")
    spearman = np.empty((len(indices), len(cache.policy_seeds)), dtype=np.float64)
    regret = np.empty_like(spearman)
    for local in range(len(indices)):
        kept = np.flatnonzero(valid[local])
        for policy_position in range(len(cache.policy_seeds)):
            spearman[local, policy_position], regret[local, policy_position] = (
                _ranking_row(
                    predicted_cost[local, kept],
                    labels[local, policy_position, kept],
                )
            )
    return spearman, regret


def paired_episode_bootstrap(
    first_cost: np.ndarray,
    second_cost: np.ndarray,
    cache: SelectorCache,
    indices: np.ndarray,
    *,
    first_name: str,
    second_name: str,
    resamples: int = PAIRED_BOOTSTRAP_RESAMPLES,
    seed: int = PAIRED_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Cluster-bootstrap paired ranking deltas over physical episodes.

    Episode clusters are sampled with replacement.  Every selected cluster
    contributes all of its frames and all three policy-label rows.  Both
    reported deltas use a positive-means-``first_name`` convention.
    """

    if resamples <= 0 or seed < 0:
        raise ValueError("paired bootstrap resamples/seed are invalid")
    selected = np.asarray(indices, dtype=np.int64)
    first_spearman, first_regret = _ranking_metric_arrays(
        first_cost, cache, selected
    )
    second_spearman, second_regret = _ranking_metric_arrays(
        second_cost, cache, selected
    )
    spearman_delta = first_spearman - second_spearman
    regret_improvement = second_regret - first_regret

    task = np.asarray(cache.deployment["task_name"]).astype(str)[selected]
    episode = np.asarray(cache.deployment["episode_id"])[selected]
    cluster_keys = sorted(
        {(str(task_name), str(episode_id)) for task_name, episode_id in zip(task, episode, strict=True)}
    )
    if len(cluster_keys) < 2:
        raise ValueError("paired episode bootstrap requires at least two episodes")
    cluster_spearman_sum: list[float] = []
    cluster_regret_sum: list[float] = []
    cluster_count: list[int] = []
    for task_name, episode_id in cluster_keys:
        mask = (task == task_name) & (episode.astype(str) == episode_id)
        # Flattening preserves every frame and every policy-label row in the
        # physical-episode cluster.
        cluster_spearman_sum.append(float(spearman_delta[mask].sum()))
        cluster_regret_sum.append(float(regret_improvement[mask].sum()))
        cluster_count.append(int(spearman_delta[mask].size))
    spearman_sum = np.asarray(cluster_spearman_sum, dtype=np.float64)
    regret_sum = np.asarray(cluster_regret_sum, dtype=np.float64)
    counts = np.asarray(cluster_count, dtype=np.float64)
    if np.any(counts <= 0):
        raise RuntimeError("paired bootstrap constructed an empty episode cluster")

    rng = np.random.default_rng(int(seed))
    draws = rng.integers(
        0, len(cluster_keys), size=(int(resamples), len(cluster_keys))
    )
    sampled_count = counts[draws].sum(axis=1)
    sampled_spearman = spearman_sum[draws].sum(axis=1) / sampled_count
    sampled_regret = regret_sum[draws].sum(axis=1) / sampled_count

    def statistic(values: np.ndarray, samples: np.ndarray) -> dict[str, Any]:
        return {
            "point_estimate": float(values.mean()),
            "ci95_percentile": [
                float(value) for value in np.quantile(samples, (0.025, 0.975))
            ],
        }

    return {
        "first": str(first_name),
        "second": str(second_name),
        "sign_convention": (
            "positive favors first: Spearman=first-second; "
            "normalized_top1_regret_improvement=second-first"
        ),
        "physical_episode_count": int(len(cluster_keys)),
        "bootstrap_resamples": int(resamples),
        "bootstrap_seed": int(seed),
        "cluster_unit": (
            "task+physical_episode; each resampled cluster preserves all frames "
            "and all policy-label rows"
        ),
        "development_only_disclaimer": (
            "Descriptive development-set interval with few physical episodes; "
            "it is not independent confirmatory evidence and scorer seeds do not "
            "increase the physical-episode sample size."
        ),
        "spearman_delta": statistic(spearman_delta, sampled_spearman),
        "normalized_top1_regret_improvement": statistic(
            regret_improvement, sampled_regret
        ),
    }


def baseline_costs(cache: SelectorCache, indices: np.ndarray) -> dict[str, np.ndarray]:
    labels = np.asarray(cache.offline["action_xyz_normalized_mse_h8"], dtype=np.float32)[
        indices
    ]
    visual = -np.asarray(
        cache.deployment["candidate_joint_visual_score"], dtype=np.float32
    )[indices]
    return {
        "frozen_visual_score": visual,
        "offline_shared_mean_label_reference": labels.mean(axis=1),
    }


def per_policy_oracle_report(cache: SelectorCache, indices: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(cache.offline["action_xyz_normalized_mse_h8"], dtype=np.float32)
    result: dict[str, Any] = {}
    for policy_position, seed in enumerate(cache.policy_seeds):
        single_offline = dict(cache.offline)
        single_offline["action_xyz_normalized_mse_h8"] = np.repeat(
            labels[:, policy_position : policy_position + 1], 3, axis=1
        )
        single_cache = SelectorCache(
            **{**cache.__dict__, "offline": single_offline}
        )
        metrics = evaluate_costs(
            labels[indices, policy_position], single_cache, indices
        )["macro"]
        # ``single_cache`` repeats one policy label across the fixed three-seed
        # cache only to reuse the common metric implementation.  Report the
        # number of physical snapshots represented by this per-policy row,
        # rather than the internal threefold repetition.
        result[str(seed)] = {**metrics, "count": int(len(indices))}
    return result


def _normalizer_payload(
    normalizer: Mapping[str, FieldStatistics]
) -> dict[str, dict[str, np.ndarray]]:
    return {
        key: {"mean": value.mean, "std": value.std}
        for key, value in normalizer.items()
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", action="append", choices=MODES, default=[])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--label-temperature", type=float, default=0.25)
    parser.add_argument("--prediction-temperature", type=float, default=1.0)
    return parser


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA selector training requested but unavailable")
    return device


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seed < 0 or min(args.epochs, args.batch_size, args.model_dim, args.hidden_dim) <= 0:
        raise ValueError("training seed/dimensions/budget are invalid")
    cache_dir = (REPO_ROOT / args.cache_dir).resolve() if not Path(args.cache_dir).is_absolute() else Path(args.cache_dir)
    output_dir = (REPO_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty selector output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = load_selector_cache(cache_dir)
    metadata_contract = dict((cache.metadata or {}).get("contract", {}))
    history_primary = str(
        metadata_contract.get("history_grounding_primary", "legacy_unspecified")
    )
    history_source = str(
        metadata_contract.get("history_grounding_source", "legacy_unspecified")
    )
    canonical_native_history = history_primary == "native_fcpg"
    default_cache = (REPO_ROOT / DEFAULT_CACHE_DIR).resolve()
    if cache_dir.resolve() == default_cache and not canonical_native_history:
        raise ValueError(
            "default matched-history training requires metadata primary=native_fcpg"
        )
    modes = tuple(dict.fromkeys(args.mode or list(MODES)))
    device = _resolve_device(args.device)
    test_indices = _split_indices(cache.deployment["split"], "test")
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "posthoc_development_not_confirmatory",
        "claim_boundary": (
            "Candidate-selection development only. The prior is a predicted 192-D "
            "summary; no structured future-anchor dynamics, rollout, CEM, or control "
            "benefit is established."
        ),
        "history_contract": {
            "generic_primary": history_primary,
            "generic_source": history_source,
            "generic_is_canonical_native_fcpg": canonical_native_history,
            "diagnostics": metadata_contract.get(
                "history_grounding_diagnostics", []
            ),
            "inference_q_true_usage": metadata_contract.get(
                "history_inference_q_true_usage", "legacy_unspecified"
            ),
            "metadata_provenance_not_inferred_from_path": True,
        },
        "cache": {
            "path": str(cache_dir),
            "deployment_sha256": sha256_file(cache_dir / "deployment_inputs.npz"),
            "offline_labels_sha256": sha256_file(cache_dir / "offline_labels.npz"),
            "metadata_sha256": sha256_file(cache_dir / "metadata.json"),
        },
        "selection_contract": "checkpoint selected only by validation KL",
        "training_config": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "model_dim": args.model_dim,
            "hidden_dim": args.hidden_dim,
            "label_temperature": args.label_temperature,
            "prediction_temperature": args.prediction_temperature,
            "hyperparameter_sweep": False,
        },
        "baselines": {},
        "models": {},
        "paired_episode_bootstrap": {
            "status": "development_only_small_n_descriptive",
            "comparisons": {},
        },
    }
    for name, costs in baseline_costs(cache, test_indices).items():
        report["baselines"][name] = evaluate_costs(costs, cache, test_indices)
    report["baselines"]["offline_per_policy_oracle_upper_bound"] = per_policy_oracle_report(
        cache, test_indices
    )

    counts: set[int] = set()
    predicted_costs_by_mode: dict[str, np.ndarray] = {}
    shuffled_costs_by_mode: dict[str, np.ndarray] = {}
    for mode in modes:
        trained = train_selector(
            cache,
            mode=mode,
            device=device,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            model_dim=args.model_dim,
            hidden_dim=args.hidden_dim,
            label_temperature=args.label_temperature,
            prediction_temperature=args.prediction_temperature,
        )
        counts.add(trained.parameter_count)
        costs = _predict_costs(
            trained.model.to(device),
            cache.deployment,
            test_indices,
            trained.normalizer,
            device=device,
            batch_size=args.batch_size,
        )
        predicted_costs_by_mode[mode] = costs
        mode_report: dict[str, Any] = {
            "allocated_parameter_count": trained.parameter_count,
            "module_path_parameter_count": trained.module_path_parameter_count,
            "capacity_note": (
                "Allocated counts are identical because the implementation keeps "
                "dormant branches. Module-path counts differ and are reported; the "
                "three probes are controlled small models, not exactly capacity matched."
            ),
            "required_deployment_keys": sorted(MODE_INPUT_KEYS[mode]),
            "best_epoch": trained.best_epoch,
            "best_validation_kl": trained.best_validation_kl,
            "development_test": evaluate_costs(costs, cache, test_indices),
            "training_history": list(trained.train_history),
            "structured_future_dynamics_claim": False,
        }
        if "predicted_current_summary_prior" in MODE_INPUT_KEYS[mode]:
            shuffled = np.asarray(
                cache.offline["shuffled_predicted_current_summary_prior"],
                dtype=np.float32,
            )
            control_overrides: dict[str, np.ndarray] = {
                "predicted_current_summary_prior": shuffled
            }
            control_name = "offline_shuffled_prior_control"
            if mode == "structured_full":
                shuffled_history = np.asarray(
                    cache.deployment["history_action_blocks"], dtype=np.float32
                ).copy()
                shuffled_history[:, -1] = np.asarray(
                    cache.offline["shuffled_current_action_blocks"], dtype=np.float32
                )
                control_overrides["history_action_blocks"] = shuffled_history
                control_name = "offline_shuffled_executed_action_control"
            shuffled_costs = _predict_costs(
                trained.model,
                cache.deployment,
                test_indices,
                trained.normalizer,
                device=device,
                batch_size=args.batch_size,
                input_overrides=control_overrides,
            )
            shuffled_costs_by_mode[mode] = shuffled_costs
            mode_report[control_name] = {
                "source_partition": "offline_labels_only",
                "used_for_training": False,
                "overridden_model_inputs": sorted(control_overrides),
                "metrics": evaluate_costs(shuffled_costs, cache, test_indices),
            }
        history_diagnostics: dict[str, Any] = {}
        if mode in {"structured_full", "summary_predictive"}:
            for suffix in HISTORY_VARIANTS:
                prior_key = f"predicted_current_summary_prior_{suffix}"
                if prior_key not in cache.deployment:
                    continue
                variant_overrides: dict[str, np.ndarray] = {
                    "predicted_current_summary_prior": np.asarray(
                        cache.deployment[prior_key], dtype=np.float32
                    )
                }
                if mode == "structured_full":
                    variant_overrides["history_anchor_tokens"] = np.asarray(
                        cache.deployment[f"history_anchor_tokens_{suffix}"],
                        dtype=np.float32,
                    )
                variant_costs = _predict_costs(
                    trained.model,
                    cache.deployment,
                    test_indices,
                    trained.normalizer,
                    device=device,
                    batch_size=args.batch_size,
                    input_overrides=variant_overrides,
                )
                shuffled_variant_overrides = dict(variant_overrides)
                shuffled_variant_overrides[
                    "predicted_current_summary_prior"
                ] = np.asarray(
                    cache.offline[
                        f"shuffled_predicted_current_summary_prior_{suffix}"
                    ],
                    dtype=np.float32,
                )
                if mode == "structured_full":
                    shuffled_history = np.asarray(
                        cache.deployment["history_action_blocks"], dtype=np.float32
                    ).copy()
                    shuffled_history[:, -1] = np.asarray(
                        cache.offline["shuffled_current_action_blocks"],
                        dtype=np.float32,
                    )
                    shuffled_variant_overrides[
                        "history_action_blocks"
                    ] = shuffled_history
                shuffled_variant_costs = _predict_costs(
                    trained.model,
                    cache.deployment,
                    test_indices,
                    trained.normalizer,
                    device=device,
                    batch_size=args.batch_size,
                    input_overrides=shuffled_variant_overrides,
                )
                history_diagnostics[suffix] = {
                    "used_for_training": False,
                    "used_for_checkpoint_selection": False,
                    "overridden_model_inputs": sorted(variant_overrides),
                    "factual_metrics": evaluate_costs(
                        variant_costs, cache, test_indices
                    ),
                    "offline_shuffled_action_condition_control": {
                        "source_partition": "offline_labels_only",
                        "overridden_model_inputs": sorted(
                            shuffled_variant_overrides
                        ),
                        "metrics": evaluate_costs(
                            shuffled_variant_costs, cache, test_indices
                        ),
                    },
                }
        if history_diagnostics:
            mode_report["history_source_diagnostics"] = history_diagnostics
        checkpoint = {
            "schema_version": 1,
            "mode": mode,
            "model_config": {
                "anchor_count": cache.anchor_count,
                "latent_dim": cache.latent_dim,
                "geometry_dim": cache.geometry_dim,
                "action_block_dim": cache.action_block_dim,
                "model_dim": args.model_dim,
                "hidden_dim": args.hidden_dim,
            },
            "allocated_parameter_count": trained.parameter_count,
            "module_path_parameter_count": trained.module_path_parameter_count,
            "required_deployment_keys": sorted(MODE_INPUT_KEYS[mode]),
            "best_epoch": trained.best_epoch,
            "best_validation_kl": trained.best_validation_kl,
            "model_state_dict": trained.model.cpu().state_dict(),
            "normalizer": _normalizer_payload(trained.normalizer),
            "cache_sha256": report["cache"],
        }
        torch.save(checkpoint, output_dir / f"{mode}_best.pt")
        report["models"][mode] = mode_report

    paired_comparisons = report["paired_episode_bootstrap"]["comparisons"]

    def add_paired_comparison(
        key: str,
        first_name: str,
        first_cost: np.ndarray,
        second_name: str,
        second_cost: np.ndarray,
    ) -> None:
        paired_comparisons[key] = paired_episode_bootstrap(
            first_cost,
            second_cost,
            cache,
            test_indices,
            first_name=first_name,
            second_name=second_name,
        )

    if {"structured_full", "structured_current_only"}.issubset(
        predicted_costs_by_mode
    ):
        add_paired_comparison(
            "structured_full_vs_structured_current_only",
            "structured_full",
            predicted_costs_by_mode["structured_full"],
            "structured_current_only",
            predicted_costs_by_mode["structured_current_only"],
        )
    if {"structured_full", "summary_predictive"}.issubset(predicted_costs_by_mode):
        add_paired_comparison(
            "structured_full_vs_summary_predictive",
            "structured_full",
            predicted_costs_by_mode["structured_full"],
            "summary_predictive",
            predicted_costs_by_mode["summary_predictive"],
        )
    if "structured_current_only" in predicted_costs_by_mode:
        add_paired_comparison(
            "structured_current_only_vs_frozen_visual_score",
            "structured_current_only",
            predicted_costs_by_mode["structured_current_only"],
            "frozen_visual_score",
            baseline_costs(cache, test_indices)["frozen_visual_score"],
        )
    if "structured_full" in shuffled_costs_by_mode:
        add_paired_comparison(
            "structured_full_factual_vs_combined_shuffled_action_prior",
            "structured_full_factual",
            predicted_costs_by_mode["structured_full"],
            "structured_full_combined_shuffled_action_prior",
            shuffled_costs_by_mode["structured_full"],
        )
    if len(counts) != 1:
        raise RuntimeError("selector modes are not parameter matched")
    report["shared_allocated_parameter_count"] = int(next(iter(counts)))
    report["module_path_parameter_counts"] = {
        mode: int(value["module_path_parameter_count"])
        for mode, value in report["models"].items()
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "modes": list(modes),
                "shared_allocated_parameter_count": report[
                    "shared_allocated_parameter_count"
                ],
                "module_path_parameter_counts": report[
                    "module_path_parameter_counts"
                ],
                "report_sha256": sha256_file(report_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
