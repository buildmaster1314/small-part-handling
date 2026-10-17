"""Native P1-token dynamics over strict A3/P1 replay bundles.

This module removes the failed ``192D A3 -> 256D P1`` bridge from the
experiment.  It reconstructs a causal three-state P1 history directly from
the strict replay cache::

    history = [H0 last-valid token, H1 observed token, H2 observed token]
    target  = [H3, H4, H5, H6, H7, H8]

The visual interruption starts *after* H2.  Consequently H1/H2 are available
measurements at deployment time, while H3-H8 remain training/evaluation
labels.  Actions are aligned with transitions: history actions are H0->H1 and
H1->H2; future actions are H2->H3 through H7->H8.

The source is exposed A3 train/development replay, not an independent test.
Physical episodes, rather than branches or hazards, remain the statistical
unit.  Exact-FK baselines use deployable robot kinematics but never future
images.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from fcpg.lewm_a3_p1_bridge_collection import (
    BRANCH_COUNT,
    HORIZON,
    PICK_TASK,
    validate_bridge_arrays,
)
from fcpg.lewm_action_branch_training import (
    matched_action_branch_rollout,
    rollout_action_branches,
)
from fcpg.lewm_anchor_dynamics_pair import MatchedAnchorDynamicsPredictor
from fcpg.lewm_formal_bridge_conversion import validate_stacked_collection_metadata
from fcpg.lewm_known_drop_bridge_runtime import geometry_hold_policy_token


P1_NATIVE_PROTOCOL: Final = "fcpg_p1_native_dynamics_h2_dropout_v1"
P1_NATIVE_HISTORY_SIZE: Final = 3
P1_NATIVE_DROPOUT_ONSET_HORIZON: Final = 2
P1_NATIVE_TARGET_HORIZONS: Final = tuple(range(3, HORIZON + 1))
P1_NATIVE_FUTURE_STEPS: Final = len(P1_NATIVE_TARGET_HORIZONS)
P1_NATIVE_ANCHORS: Final = 3
P1_NATIVE_TOKEN_DIM: Final = 256
P1_NATIVE_VISUAL_DIM: Final = 256
P1_NATIVE_ACTION_DIM: Final = 20
P1_NATIVE_WRONG_ACTION_SHIFTS: Final = tuple(range(1, BRANCH_COUNT))


def _finite_float_array(
    value: object,
    *,
    shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {array.shape}")
    if array.dtype.kind != "f":
        raise TypeError(f"{label} must be floating point")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{label} contains non-finite values")
    return np.asarray(array, dtype=np.float32)


def _text_vector(value: object, *, count: int, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (count,):
        raise ValueError(f"{label} must have shape {(count,)}, got {array.shape}")
    if array.dtype.kind not in {"S", "U"}:
        raise TypeError(f"{label} must contain stable strings")
    result = array.astype("U", copy=False)
    if any(not item for item in result.tolist()):
        raise ValueError(f"{label} contains an empty value")
    return result


def _validate_episode_split_integrity(
    episode: np.ndarray,
    split: np.ndarray,
) -> None:
    assignments: dict[str, str] = {}
    for episode_id, split_name in zip(episode.tolist(), split.tolist(), strict=True):
        previous = assignments.setdefault(str(episode_id), str(split_name))
        if previous != str(split_name):
            raise ValueError(
                "one physical source episode crosses train/development splits"
            )


def _all_cyclic_wrong_actions(future_action: np.ndarray) -> np.ndarray:
    """Return seven deterministic branch-deranged future-action banks.

    The output is ``[N, shift, recipient_branch, H, A]``.  History actions are
    intentionally not permuted: H0/H1/H2 have already been observed, so only
    the action plan after the interruption may be counterfactually wrong.
    """

    if future_action.ndim != 4 or future_action.shape[1] != BRANCH_COUNT:
        raise ValueError("future_action must have shape [N,8,H,A]")
    slots = np.arange(BRANCH_COUNT)
    result = np.stack(
        [future_action[:, (slots + shift) % BRANCH_COUNT] for shift in P1_NATIVE_WRONG_ACTION_SHIFTS],
        axis=1,
    )
    if any(
        np.shares_memory(result[:, index], future_action)
        for index in range(result.shape[1])
    ):
        result = result.copy()
    return np.ascontiguousarray(result, dtype=np.float32)


@dataclass(frozen=True)
class P1NativeDynamicsCache:
    """Bundle-preserving causal P1 examples.

    Leading axes are ``[source_row, branch]``.  They are deliberately retained
    so a trainer cannot silently treat eight counterfactual branches as eight
    independent physical episodes.
    """

    source_path: Path | None
    metadata: Mapping[str, Any] | None
    history_token: np.ndarray
    history_action: np.ndarray
    future_action: np.ndarray
    wrong_future_action: np.ndarray
    target_token: np.ndarray
    history_sampled_visual: np.ndarray
    history_xyz: np.ndarray
    future_xyz: np.ndarray
    source_row_index: np.ndarray
    branch_index: np.ndarray
    task_name: np.ndarray
    physical_episode_id: np.ndarray
    hazard_id: np.ndarray
    split: np.ndarray

    @property
    def bundle_count(self) -> int:
        return int(self.history_token.shape[0])

    @property
    def branch_count(self) -> int:
        return int(self.history_token.shape[1])

    @property
    def sample_count(self) -> int:
        return self.bundle_count * self.branch_count

    @property
    def source_episode_id(self) -> np.ndarray:
        """Compatibility alias; values are physical episode identities."""

        return self.physical_episode_id


@dataclass(frozen=True)
class P1NativeFlatExamples:
    """Branch-flattened training view with episode identity retained."""

    history_token: np.ndarray
    history_action: np.ndarray
    future_action: np.ndarray
    wrong_future_action: np.ndarray
    target_token: np.ndarray
    history_sampled_visual: np.ndarray
    history_xyz: np.ndarray
    future_xyz: np.ndarray
    source_row_index: np.ndarray
    branch_index: np.ndarray
    task_name: np.ndarray
    physical_episode_id: np.ndarray
    hazard_id: np.ndarray
    split: np.ndarray

    def __len__(self) -> int:
        return int(self.history_token.shape[0])

    @property
    def source_episode_id(self) -> np.ndarray:
        return self.physical_episode_id


@dataclass(frozen=True)
class P1NativeBaselines:
    """No-future-image P1-token controls aligned to H3-H8."""

    hold_last_token: np.ndarray
    geometry_hold_exact_fk_token: np.ndarray
    visual_cv_exact_fk_token: np.ndarray

    @property
    def hold_last(self) -> np.ndarray:
        return self.hold_last_token

    @property
    def geometry_hold(self) -> np.ndarray:
        return self.geometry_hold_exact_fk_token

    @property
    def visual_constant_velocity(self) -> np.ndarray:
        return self.visual_cv_exact_fk_token


@dataclass(frozen=True)
class P1NativePredictorConfig:
    """Architecture shared exactly by factual and no-action predictors."""

    latent_dim: int = P1_NATIVE_TOKEN_DIM
    num_keypoints: int = P1_NATIVE_ANCHORS
    action_dim: int = P1_NATIVE_ACTION_DIM
    history_size: int = P1_NATIVE_HISTORY_SIZE
    action_smoothing_dim: int = 10
    predictor_depth: int = 6
    predictor_heads: int = 16
    predictor_mlp_dim: int = 2048
    predictor_dim_head: int = 64
    predictor_dropout: float = 0.1
    projector_hidden_dim: int = 2048

    def as_kwargs(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "latent_dim",
                "num_keypoints",
                "action_dim",
                "history_size",
                "action_smoothing_dim",
                "predictor_depth",
                "predictor_heads",
                "predictor_mlp_dim",
                "predictor_dim_head",
                "predictor_dropout",
                "projector_hidden_dim",
            )
        }


def build_native_examples(
    arrays: Mapping[str, np.ndarray],
    *,
    split: str | Sequence[str] | None = None,
    source_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> P1NativeDynamicsCache:
    """Reconstruct causal H0-H2 histories and H3-H8 targets.

    Full-source split integrity is checked before applying ``split`` so a
    caller cannot hide physical-episode leakage by filtering first.
    """

    validate_bridge_arrays(arrays)
    rows = int(np.asarray(arrays["source_row_index"]).size)
    branches = int(np.asarray(arrays["branch_index"]).size)
    if branches != BRANCH_COUNT:
        raise ValueError("P1-native dynamics requires all eight action branches")
    branch_index = np.asarray(arrays["branch_index"], dtype=np.int64)
    if not np.array_equal(branch_index, np.arange(BRANCH_COUNT, dtype=np.int64)):
        raise ValueError("P1-native branch axis must be exactly [0,1,...,7]")

    source_episode = _text_vector(
        arrays["source_episode_id"], count=rows, label="source_episode_id"
    )
    source_split = _text_vector(arrays["split"], count=rows, label="split")
    if not set(source_split.tolist()).issubset({"train", "development"}):
        raise ValueError("P1-native cache accepts exposed train/development only")
    _validate_episode_split_integrity(source_episode, source_split)

    if split is None:
        selected = np.arange(rows, dtype=np.int64)
    else:
        requested = {str(split)} if isinstance(split, str) else {str(x) for x in split}
        if not requested or not requested.issubset({"train", "development"}):
            raise ValueError("split must select train and/or development")
        selected = np.flatnonzero(np.isin(source_split, sorted(requested))).astype(np.int64)
        if selected.size == 0:
            raise ValueError(f"strict cache has no rows for split {sorted(requested)}")

    n = int(selected.size)
    hidden_token = _finite_float_array(
        np.asarray(arrays["hidden_clean_policy_token"])[selected],
        shape=(n, branches, HORIZON, P1_NATIVE_ANCHORS, P1_NATIVE_TOKEN_DIM),
        label="hidden_clean_policy_token",
    )
    hidden_visual = _finite_float_array(
        np.asarray(arrays["hidden_clean_sampled_visual"])[selected],
        shape=(n, branches, HORIZON, P1_NATIVE_ANCHORS, P1_NATIVE_VISUAL_DIM),
        label="hidden_clean_sampled_visual",
    )
    hidden_xyz = _finite_float_array(
        np.asarray(arrays["current_xyz"])[selected],
        shape=(n, branches, HORIZON, P1_NATIVE_ANCHORS, 3),
        label="current_xyz",
    )
    branch_action = _finite_float_array(
        np.asarray(arrays["branch_action"])[selected],
        shape=(n, branches, HORIZON, P1_NATIVE_ACTION_DIM),
        label="branch_action",
    )
    h0_token = _finite_float_array(
        np.asarray(arrays["last_policy_token"])[selected],
        shape=(n, P1_NATIVE_ANCHORS, P1_NATIVE_TOKEN_DIM),
        label="last_policy_token",
    )
    h0_visual = _finite_float_array(
        np.asarray(arrays["last_sampled_visual"])[selected],
        shape=(n, P1_NATIVE_ANCHORS, P1_NATIVE_VISUAL_DIM),
        label="last_sampled_visual",
    )
    h0_xyz = _finite_float_array(
        np.asarray(arrays["last_xyz"])[selected],
        shape=(n, P1_NATIVE_ANCHORS, 3),
        label="last_xyz",
    )

    def repeat_h0(value: np.ndarray) -> np.ndarray:
        return np.broadcast_to(value[:, None], (n, branches, *value.shape[1:]))

    history_token = np.stack(
        (repeat_h0(h0_token), hidden_token[:, :, 0], hidden_token[:, :, 1]),
        axis=2,
    )
    history_visual = np.stack(
        (repeat_h0(h0_visual), hidden_visual[:, :, 0], hidden_visual[:, :, 1]),
        axis=2,
    )
    history_xyz = np.stack(
        (repeat_h0(h0_xyz), hidden_xyz[:, :, 0], hidden_xyz[:, :, 1]),
        axis=2,
    )
    history_action = branch_action[:, :, :P1_NATIVE_DROPOUT_ONSET_HORIZON]
    future_action = branch_action[:, :, P1_NATIVE_DROPOUT_ONSET_HORIZON:]
    target_token = hidden_token[:, :, P1_NATIVE_DROPOUT_ONSET_HORIZON:]
    future_xyz = hidden_xyz[:, :, P1_NATIVE_DROPOUT_ONSET_HORIZON:]

    expected_future = (n, branches, P1_NATIVE_FUTURE_STEPS)
    if future_action.shape[:3] != expected_future or target_token.shape[:3] != expected_future:
        raise RuntimeError("H2 interruption did not yield exact H3-H8 supervision")

    expanded_episode = np.broadcast_to(source_episode[selected, None], (n, branches))
    expanded_split = np.broadcast_to(source_split[selected, None], (n, branches))
    expanded_task = np.full((n, branches), PICK_TASK, dtype="U32")
    expanded_row = np.broadcast_to(
        np.asarray(arrays["source_row_index"], dtype=np.int64)[selected, None],
        (n, branches),
    )
    expanded_branch = np.broadcast_to(branch_index[None], (n, branches))
    expanded_hazard = np.broadcast_to(
        np.asarray(arrays["hazard_id"], dtype=np.int64)[selected, None],
        (n, branches),
    )
    return P1NativeDynamicsCache(
        source_path=(
            Path(source_path).expanduser().resolve() if source_path is not None else None
        ),
        metadata=dict(metadata) if metadata is not None else None,
        history_token=np.ascontiguousarray(history_token, dtype=np.float32),
        history_action=np.ascontiguousarray(history_action, dtype=np.float32),
        future_action=np.ascontiguousarray(future_action, dtype=np.float32),
        wrong_future_action=_all_cyclic_wrong_actions(future_action),
        target_token=np.ascontiguousarray(target_token, dtype=np.float32),
        history_sampled_visual=np.ascontiguousarray(history_visual, dtype=np.float32),
        history_xyz=np.ascontiguousarray(history_xyz, dtype=np.float32),
        future_xyz=np.ascontiguousarray(future_xyz, dtype=np.float32),
        source_row_index=np.ascontiguousarray(expanded_row),
        branch_index=np.ascontiguousarray(expanded_branch),
        task_name=np.ascontiguousarray(expanded_task),
        physical_episode_id=np.ascontiguousarray(expanded_episode),
        hazard_id=np.ascontiguousarray(expanded_hazard),
        split=np.ascontiguousarray(expanded_split),
    )


def load_p1_native_dynamics_cache(
    source: str | Path,
    *,
    metadata_path: str | Path | None = None,
    split: str | Sequence[str] | None = None,
    verify_provenance: bool = True,
) -> P1NativeDynamicsCache:
    """Load strict stacked replay and, by default, verify full provenance."""

    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    raw_metadata: Mapping[str, Any] | None = None
    if verify_provenance:
        receipt = (
            Path(metadata_path).expanduser().resolve()
            if metadata_path is not None
            else source_path.with_suffix(".metadata.json")
        )
        if not receipt.is_file():
            raise FileNotFoundError(f"strict stacked metadata is missing: {receipt}")
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("strict stacked metadata must contain a mapping")
        validate_stacked_collection_metadata(payload, source_path=source_path)
        raw_metadata = payload
    elif metadata_path is not None:
        raise ValueError("metadata_path is meaningless when verify_provenance=False")

    with np.load(source_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return build_native_examples(
        arrays,
        split=split,
        source_path=source_path,
        metadata=raw_metadata,
    )


def flatten_native_examples(cache: P1NativeDynamicsCache) -> P1NativeFlatExamples:
    """Flatten row/branch for optimization without dropping group identity."""

    n, branches = cache.history_token.shape[:2]

    def flat(value: np.ndarray, *, wrong: bool = False) -> np.ndarray:
        if wrong:
            # [N,shift,B,...] -> [N,B,shift,...] -> [N*B,shift,...]
            return np.ascontiguousarray(
                np.swapaxes(value, 1, 2).reshape(n * branches, *value.shape[1:2], *value.shape[3:])
            )
        return np.ascontiguousarray(value.reshape(n * branches, *value.shape[2:]))

    return P1NativeFlatExamples(
        history_token=flat(cache.history_token),
        history_action=flat(cache.history_action),
        future_action=flat(cache.future_action),
        wrong_future_action=flat(cache.wrong_future_action, wrong=True),
        target_token=flat(cache.target_token),
        history_sampled_visual=flat(cache.history_sampled_visual),
        history_xyz=flat(cache.history_xyz),
        future_xyz=flat(cache.future_xyz),
        source_row_index=cache.source_row_index.reshape(-1).copy(),
        branch_index=cache.branch_index.reshape(-1).copy(),
        task_name=cache.task_name.reshape(-1).copy(),
        physical_episode_id=cache.physical_episode_id.reshape(-1).copy(),
        hazard_id=cache.hazard_id.reshape(-1).copy(),
        split=cache.split.reshape(-1).copy(),
    )


def p1_native_episode_mse(
    prediction: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    cache: P1NativeDynamicsCache,
) -> dict[str, float]:
    """Aggregate token MSE with physical episode as the statistical unit.

    Rows from multiple hazards and all counterfactual branches/horizons are
    averaged *within* an episode.  The returned mapping contains exactly one
    value per physical episode and must be used for paired confidence
    intervals; branch-level values are not independent evidence.
    """

    left = (
        prediction.detach().float().cpu().numpy()
        if torch.is_tensor(prediction)
        else np.asarray(prediction, dtype=np.float32)
    )
    right = (
        target.detach().float().cpu().numpy()
        if torch.is_tensor(target)
        else np.asarray(target, dtype=np.float32)
    )
    expected = cache.target_token.shape
    if left.shape != expected or right.shape != expected:
        raise ValueError(
            f"prediction and target must both have shape {expected}; "
            f"got {left.shape} and {right.shape}"
        )
    if not bool(np.isfinite(left).all() and np.isfinite(right).all()):
        raise ValueError("prediction and target must be finite")
    per_row_branch = np.mean(
        np.square(left.astype(np.float64) - right.astype(np.float64)),
        axis=(2, 3, 4),
    )
    episode = cache.physical_episode_id
    result: dict[str, float] = {}
    for episode_id in sorted(set(episode.reshape(-1).tolist())):
        mask = episode == episode_id
        if not bool(mask.any()):
            raise RuntimeError("episode aggregation produced an empty group")
        result[str(episode_id)] = float(np.mean(per_row_branch[mask]))
    return result


class P1NativeTorchDataset(Dataset[dict[str, Any]]):
    """Torch view over branch examples; grouping identifiers remain explicit."""

    def __init__(self, examples: P1NativeFlatExamples) -> None:
        if len(examples) <= 0:
            raise ValueError("P1-native dataset cannot be empty")
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        i = int(index)
        if i < 0 or i >= len(self):
            raise IndexError(i)
        return {
            "sample_index": i,
            "history_token": torch.from_numpy(self.examples.history_token[i]),
            "history_action": torch.from_numpy(self.examples.history_action[i]),
            "future_action": torch.from_numpy(self.examples.future_action[i]),
            "wrong_future_action": torch.from_numpy(self.examples.wrong_future_action[i]),
            "target_token": torch.from_numpy(self.examples.target_token[i]),
            "source_row_index": int(self.examples.source_row_index[i]),
            "branch_index": int(self.examples.branch_index[i]),
            "physical_episode_id": str(self.examples.physical_episode_id[i]),
            "hazard_id": int(self.examples.hazard_id[i]),
            "split": str(self.examples.split[i]),
        }


def build_matched_p1_native_predictors(
    config: P1NativePredictorConfig,
    *,
    initialization_seed: int,
) -> tuple[MatchedAnchorDynamicsPredictor, MatchedAnchorDynamicsPredictor]:
    """Build matched factual/no-action heads from byte-identical state."""

    expected = {
        "latent_dim": P1_NATIVE_TOKEN_DIM,
        "num_keypoints": P1_NATIVE_ANCHORS,
        "action_dim": P1_NATIVE_ACTION_DIM,
        "history_size": P1_NATIVE_HISTORY_SIZE,
    }
    for name, value in expected.items():
        if getattr(config, name) != value:
            raise ValueError(f"P1-native {name} must remain exactly {value}")
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(initialization_seed))
        factual = MatchedAnchorDynamicsPredictor(
            **config.as_kwargs(), condition_on_action=True
        )
    no_action = deepcopy(factual)
    no_action.condition_on_action = False
    validate_matched_p1_native_pair(factual, no_action, require_identical=True)
    return factual, no_action


def validate_matched_p1_native_pair(
    factual: MatchedAnchorDynamicsPredictor,
    no_action: MatchedAnchorDynamicsPredictor,
    *,
    require_identical: bool = False,
) -> None:
    """Validate capacity and, before training, exact matched initialization."""

    expected = {
        "latent_dim": P1_NATIVE_TOKEN_DIM,
        "num_keypoints": P1_NATIVE_ANCHORS,
        "action_dim": P1_NATIVE_ACTION_DIM,
        "history_size": P1_NATIVE_HISTORY_SIZE,
    }
    if factual.condition_on_action is not True or no_action.condition_on_action is not False:
        raise ValueError("P1-native pair must contain factual and no-action heads")
    for model in (factual, no_action):
        for name, value in expected.items():
            if getattr(model, name, None) != value:
                raise ValueError(f"P1-native predictor {name} must be {value}")
    factual_count = sum(parameter.numel() for parameter in factual.parameters())
    no_action_count = sum(parameter.numel() for parameter in no_action.parameters())
    if factual_count != no_action_count:
        raise ValueError("P1-native factual/no-action capacities differ")
    if require_identical:
        left = factual.state_dict()
        right = no_action.state_dict()
        if left.keys() != right.keys() or any(
            not torch.equal(left[name], right[name]) for name in left
        ):
            raise ValueError("P1-native matched heads do not share initialization")


def set_p1_native_action_normalization(
    factual: MatchedAnchorDynamicsPredictor,
    no_action: MatchedAnchorDynamicsPredictor,
    training_cache: P1NativeDynamicsCache,
    *,
    minimum_std: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Set identical train-split-only action normalization on matched heads."""

    validate_matched_p1_native_pair(factual, no_action)
    if minimum_std <= 0.0:
        raise ValueError("minimum_std must be positive")
    if set(training_cache.split.reshape(-1).tolist()) != {"train"}:
        raise ValueError("action normalization must be computed from train split only")
    actions = np.concatenate(
        (
            training_cache.history_action.reshape(-1, P1_NATIVE_ACTION_DIM),
            training_cache.future_action.reshape(-1, P1_NATIVE_ACTION_DIM),
        ),
        axis=0,
    ).astype(np.float64)
    mean = actions.mean(axis=0).astype(np.float32)
    std = actions.std(axis=0).astype(np.float32)
    std = np.maximum(std, np.float32(minimum_std))
    for model in (factual, no_action):
        model.set_action_normalization(action_mean=mean, action_std=std)
    return mean, std


def _validate_native_rollout_tensors(
    history_token: torch.Tensor,
    history_action: torch.Tensor,
    future_action: torch.Tensor,
) -> None:
    if history_token.ndim != 4:
        raise ValueError("history_token must have shape [B,3,3,256]")
    expected_history = (
        int(history_token.shape[0]),
        P1_NATIVE_HISTORY_SIZE,
        P1_NATIVE_ANCHORS,
        P1_NATIVE_TOKEN_DIM,
    )
    if tuple(history_token.shape) != expected_history:
        raise ValueError(f"history_token must have shape {expected_history}")
    expected_actions = (
        expected_history[0],
        P1_NATIVE_HISTORY_SIZE - 1,
        P1_NATIVE_ACTION_DIM,
    )
    if tuple(history_action.shape) != expected_actions:
        raise ValueError(f"history_action must have shape {expected_actions}")
    if (
        future_action.ndim != 4
        or int(future_action.shape[0]) != expected_history[0]
        or int(future_action.shape[2]) != P1_NATIVE_FUTURE_STEPS
        or int(future_action.shape[3]) != P1_NATIVE_ACTION_DIM
    ):
        raise ValueError(
            "future_action must have shape [B,branches,6,20]"
        )


def rollout_p1_native(
    model: MatchedAnchorDynamicsPredictor,
    history_token: torch.Tensor,
    history_action: torch.Tensor,
    future_action: torch.Tensor,
    *,
    detach_between_steps: bool = False,
) -> torch.Tensor:
    """Recursively predict H3-H8 in the native P1 token space."""

    _validate_native_rollout_tensors(history_token, history_action, future_action)
    return rollout_action_branches(
        model,
        history_token,
        history_action,
        future_action,
        detach_between_steps=detach_between_steps,
    )


def rollout_matched_p1_native(
    factual: MatchedAnchorDynamicsPredictor,
    no_action: MatchedAnchorDynamicsPredictor,
    history_token: torch.Tensor,
    history_action: torch.Tensor,
    future_action: torch.Tensor,
    *,
    detach_between_steps: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run matched heads with the shared stochastic-draw contract."""

    validate_matched_p1_native_pair(factual, no_action)
    _validate_native_rollout_tensors(history_token, history_action, future_action)
    return matched_action_branch_rollout(
        factual,
        no_action,
        history_token,
        history_action,
        future_action,
        detach_between_steps=detach_between_steps,
    )


@torch.inference_mode()
def reconstruct_p1_native_baselines(
    cache: P1NativeDynamicsCache,
    policy: nn.Module,
    *,
    device: str | torch.device,
    batch_size: int = 1024,
) -> P1NativeBaselines:
    """Build hold-last, geometry-hold, and visual-CV controls.

    Geometry-hold keeps the last available H2 visual descriptor.  Visual-CV
    extrapolates that descriptor from H1/H2 with constant velocity.  Both use
    exact current FK at H3-H8 and the same frozen native P1 grounding adapter.
    Neither reads H3-H8 images or clean target tokens.
    """

    if batch_size <= 0:
        raise ValueError("baseline batch_size must be positive")
    n, branches, future_steps = cache.target_token.shape[:3]
    if future_steps != P1_NATIVE_FUTURE_STEPS:
        raise ValueError("P1-native targets must cover exactly H3-H8")
    h2_token = cache.history_token[:, :, -1]
    hold = np.broadcast_to(
        h2_token[:, :, None],
        (n, branches, future_steps, P1_NATIVE_ANCHORS, P1_NATIVE_TOKEN_DIM),
    ).copy()
    h1_visual = cache.history_sampled_visual[:, :, -2]
    h2_visual = cache.history_sampled_visual[:, :, -1]
    held_visual = np.broadcast_to(
        h2_visual[:, :, None],
        (n, branches, future_steps, P1_NATIVE_ANCHORS, P1_NATIVE_VISUAL_DIM),
    )
    steps = np.arange(1, future_steps + 1, dtype=np.float32).reshape(1, 1, -1, 1, 1)
    cv_visual = h2_visual[:, :, None] + steps * (
        h2_visual[:, :, None] - h1_visual[:, :, None]
    )
    xyz = cache.future_xyz
    target_device = torch.device(device)
    policy.requires_grad_(False).eval()

    def reconstruct(visual: np.ndarray) -> np.ndarray:
        flat_visual = np.ascontiguousarray(
            visual.reshape(-1, P1_NATIVE_ANCHORS, P1_NATIVE_VISUAL_DIM),
            dtype=np.float32,
        )
        flat_xyz = np.ascontiguousarray(
            xyz.reshape(-1, P1_NATIVE_ANCHORS, 3), dtype=np.float32
        )
        output: list[np.ndarray] = []
        for start in range(0, flat_visual.shape[0], batch_size):
            stop = min(start + batch_size, flat_visual.shape[0])
            token = geometry_hold_policy_token(
                policy,
                torch.from_numpy(flat_visual[start:stop]).to(target_device),
                torch.from_numpy(flat_xyz[start:stop]).to(target_device),
            )
            output.append(token.float().cpu().numpy())
        return np.concatenate(output, axis=0).reshape(
            n,
            branches,
            future_steps,
            P1_NATIVE_ANCHORS,
            P1_NATIVE_TOKEN_DIM,
        )

    return P1NativeBaselines(
        hold_last_token=np.ascontiguousarray(hold, dtype=np.float32),
        geometry_hold_exact_fk_token=np.ascontiguousarray(reconstruct(held_visual)),
        visual_cv_exact_fk_token=np.ascontiguousarray(reconstruct(cv_visual)),
    )


__all__ = [
    "P1_NATIVE_ACTION_DIM",
    "P1_NATIVE_ANCHORS",
    "P1_NATIVE_DROPOUT_ONSET_HORIZON",
    "P1_NATIVE_FUTURE_STEPS",
    "P1_NATIVE_HISTORY_SIZE",
    "P1_NATIVE_PROTOCOL",
    "P1_NATIVE_TARGET_HORIZONS",
    "P1_NATIVE_TOKEN_DIM",
    "P1_NATIVE_VISUAL_DIM",
    "P1_NATIVE_WRONG_ACTION_SHIFTS",
    "P1NativeBaselines",
    "P1NativeDynamicsCache",
    "P1NativeFlatExamples",
    "P1NativePredictorConfig",
    "P1NativeTorchDataset",
    "build_matched_p1_native_predictors",
    "build_native_examples",
    "flatten_native_examples",
    "load_p1_native_dynamics_cache",
    "reconstruct_p1_native_baselines",
    "rollout_matched_p1_native",
    "rollout_p1_native",
    "set_p1_native_action_normalization",
    "p1_native_episode_mse",
    "validate_matched_p1_native_pair",
]
