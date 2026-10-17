"""Closed-loop matched factual/no-action training over cached LeWM latents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from fcpg.lewm_action_branch_a1 import (
    ACTION_DIM as A1_ACTION_DIM,
    CANDIDATE_COUNT as A1_BRANCH_COUNT,
    EMBEDDING_DIM as A1_LATENT_DIM,
    HISTORY_STEPS as A1_HISTORY_SIZE,
    KEYPOINT_COUNT as A1_NUM_KEYPOINTS,
    MIN_HORIZON as A1_MIN_HORIZON,
    NPZ_KEYS as CORE_NPZ_KEYS,
    validate_a1_arrays,
)
from fcpg.lewm_anchor_dynamics_pair import MatchedAnchorDynamicsPredictor


ACTION_BRANCH_TRAINING_PROTOCOL = "fcpg_action_branch_random_hazard_a1_v1"
BASE_SIGREG_WEIGHT = 0.02
FROZEN_BASE_CHECKPOINT_TYPE = "fcpg_anchor_persistent_lewm_v1"
FROZEN_BASE_STEP = 2_000
BUDGET_STEPS = {"smoke": 100, "2k": 2_000, "10k": 10_000}

REQUIRED_NPZ_FIELDS = frozenset(CORE_NPZ_KEYS)


@dataclass(frozen=True)
class MatchedActionBranchOutput:
    """Predictions and unwhitened losses for both matched heads."""

    factual_prediction: torch.Tensor
    no_action_prediction: torch.Tensor
    factual_raw_mse: torch.Tensor
    no_action_raw_mse: torch.Tensor
    factual_branch_centered_mse: torch.Tensor
    no_action_branch_centered_mse: torch.Tensor

    @property
    def loss(self) -> torch.Tensor:
        """Training objective; SIGReg is intentionally absent."""

        return self.factual_raw_mse + self.no_action_raw_mse


@dataclass(frozen=True)
class ActionBranchCache:
    """Validated arrays loaded from one collector-produced NPZ."""

    path: Path
    history_embedding: np.ndarray
    history_action: np.ndarray
    center_action: np.ndarray
    future_action: np.ndarray
    target_embedding: np.ndarray
    split: np.ndarray
    task_index: np.ndarray
    task_instance_index: np.ndarray
    source_episode_id: np.ndarray
    hazard_id: np.ndarray

    def __len__(self) -> int:
        return int(self.history_embedding.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.future_action.shape[2])

    def indices_for_split(self, names: str | Sequence[str]) -> np.ndarray:
        requested = {str(names)} if isinstance(names, str) else {str(x) for x in names}
        if not requested:
            raise ValueError("at least one split name is required")
        indices = np.flatnonzero(np.isin(self.split, sorted(requested))).astype(
            np.int64, copy=False
        )
        if indices.size == 0:
            available = sorted(str(value) for value in np.unique(self.split))
            raise ValueError(
                f"NPZ has no samples for split {sorted(requested)}; available={available}"
            )
        return indices


class CachedActionBranchDataset(Dataset[dict[str, Any]]):
    """Index view over an :class:`ActionBranchCache`."""

    def __init__(self, cache: ActionBranchCache, indices: np.ndarray) -> None:
        values = np.asarray(indices, dtype=np.int64)
        if values.ndim != 1:
            raise ValueError("dataset indices must be one-dimensional")
        if values.size == 0:
            raise ValueError("cached action-branch dataset cannot be empty")
        if int(values.min()) < 0 or int(values.max()) >= len(cache):
            raise IndexError("dataset index is outside the cache")
        self.cache = cache
        self.indices = values

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.indices[item])
        return {
            "sample_index": index,
            "history_embedding": torch.from_numpy(self.cache.history_embedding[index]),
            "history_action": torch.from_numpy(self.cache.history_action[index]),
            "future_action": torch.from_numpy(self.cache.future_action[index]),
            "target_embedding": torch.from_numpy(self.cache.target_embedding[index]),
        }


def budget_steps(name: str) -> int:
    """Resolve one of the fixed A1 optimizer-step budgets."""

    try:
        return BUDGET_STEPS[str(name)]
    except KeyError as error:
        raise ValueError(f"unknown A1 budget {name!r}; expected {sorted(BUDGET_STEPS)}") from error


def validate_frozen_base_checkpoint(
    checkpoint: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Require the cached latent source to be the frozen SIGReg=0.02 run."""

    if checkpoint.get("checkpoint_type") != FROZEN_BASE_CHECKPOINT_TYPE:
        raise ValueError(
            "frozen base checkpoint must be an AnchorPersistent LeWM checkpoint"
        )
    if int(checkpoint.get("step", -1)) != FROZEN_BASE_STEP:
        raise ValueError(
            f"frozen base checkpoint must be fixed at step {FROZEN_BASE_STEP}"
        )
    training_budget = checkpoint.get("training_budget")
    if not isinstance(training_budget, Mapping):
        raise ValueError("frozen base checkpoint is missing training_budget")
    if not bool(training_budget.get("run_completed", False)):
        raise ValueError("frozen base checkpoint training run is incomplete")
    for name in ("target_optimizer_steps", "checkpoint_optimizer_step"):
        if int(training_budget.get(name, -1)) != FROZEN_BASE_STEP:
            raise ValueError(
                f"frozen base checkpoint {name} must equal {FROZEN_BASE_STEP}"
            )
    if not isinstance(checkpoint.get("model_state"), Mapping):
        raise ValueError("frozen base checkpoint is missing model_state")

    resolved = checkpoint.get("resolved_config")
    if not isinstance(resolved, Mapping):
        raise ValueError("frozen base checkpoint is missing resolved_config")
    loss = resolved.get("loss")
    if not isinstance(loss, Mapping):
        raise ValueError("frozen base checkpoint is missing resolved_config.loss")
    try:
        sigreg_weight = float(loss["sigreg_weight"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "frozen base checkpoint is missing a numeric SIGReg weight"
        ) from error
    if sigreg_weight != BASE_SIGREG_WEIGHT:
        raise ValueError(
            "frozen base checkpoint must have resolved_config.loss."
            f"sigreg_weight exactly {BASE_SIGREG_WEIGHT}, got {sigreg_weight}"
        )
    return resolved


def action_normalization_from_base_checkpoint(
    checkpoint: Mapping[str, Any], *, action_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the exact action normalization used by the frozen base model."""

    normalization = checkpoint.get("normalization")
    if not isinstance(normalization, Mapping):
        data_config = checkpoint.get("data_config")
        if isinstance(data_config, Mapping):
            normalization = data_config.get("normalization")

    mean: Any = None
    std: Any = None
    if isinstance(normalization, Mapping):
        mean = normalization.get("action_mean")
        std = normalization.get("action_std")

    model_state = checkpoint.get("model_state")
    if (mean is None or std is None) and isinstance(model_state, Mapping):
        mean = model_state.get("action_mean", mean)
        std = model_state.get("action_std", std)

    if mean is None or std is None:
        raise ValueError("frozen base checkpoint is missing action normalization")
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32).detach().cpu()
    std_tensor = torch.as_tensor(std, dtype=torch.float32).detach().cpu()
    expected = (int(action_dim),)
    if mean_tensor.shape != expected or std_tensor.shape != expected:
        raise ValueError(f"base action normalization must have shape {expected}")
    if not bool(torch.isfinite(mean_tensor).all() and torch.isfinite(std_tensor).all()):
        raise ValueError("base action normalization must be finite")
    if bool(torch.any(std_tensor <= 0.0)):
        raise ValueError("base action normalization std must be strictly positive")
    return mean_tensor, std_tensor


def assert_matched_predictor_contract(
    factual: MatchedAnchorDynamicsPredictor,
    no_action: MatchedAnchorDynamicsPredictor,
) -> None:
    """Check capacity/initialization matching and the no-SIGReg head contract."""

    if factual.condition_on_action is not True:
        raise ValueError("factual predictor must condition on actions")
    if no_action.condition_on_action is not False:
        raise ValueError("no-action predictor must use a constant-zero action channel")
    factual_parameters = sum(parameter.numel() for parameter in factual.parameters())
    no_action_parameters = sum(parameter.numel() for parameter in no_action.parameters())
    if factual_parameters != no_action_parameters:
        raise ValueError("matched predictor capacities differ")
    left_state = factual.state_dict()
    right_state = no_action.state_dict()
    if left_state.keys() != right_state.keys() or any(
        not torch.equal(left_state[name], right_state[name]) for name in left_state
    ):
        raise ValueError("matched predictors must start from identical state")
    for model in (factual, no_action):
        for module_name, module in model.named_modules():
            if "sigreg" in module_name.lower() or "sigreg" in type(module).__name__.lower():
                raise ValueError("A1 predictor heads must not contain SIGReg")


def _model_dimension(model: nn.Module, name: str) -> int:
    value = getattr(model, name, None)
    if not isinstance(value, int) or value <= 0:
        raise TypeError(f"predictor must expose positive integer {name}")
    return value


def _validate_rollout_inputs(
    model: nn.Module,
    history_embedding: torch.Tensor,
    history_action: torch.Tensor,
    future_action: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    history_size = _model_dimension(model, "history_size")
    num_keypoints = _model_dimension(model, "num_keypoints")
    latent_dim = _model_dimension(model, "latent_dim")
    action_dim = _model_dimension(model, "action_dim")
    if history_embedding.ndim != 4:
        raise ValueError("history_embedding must have shape [B,H,K,D]")
    batch_size = int(history_embedding.shape[0])
    expected_history = (batch_size, history_size, num_keypoints, latent_dim)
    if tuple(history_embedding.shape) != expected_history:
        raise ValueError(
            f"history_embedding must have shape {expected_history}, "
            f"got {tuple(history_embedding.shape)}"
        )
    expected_history_action = (batch_size, history_size - 1, action_dim)
    if tuple(history_action.shape) != expected_history_action:
        raise ValueError(
            f"history_action must have shape {expected_history_action}, "
            f"got {tuple(history_action.shape)}"
        )
    if (
        future_action.ndim != 4
        or int(future_action.shape[0]) != batch_size
        or int(future_action.shape[-1]) != action_dim
    ):
        raise ValueError(
            f"future_action must have shape [B,S,P,{action_dim}], "
            f"got {tuple(future_action.shape)}"
        )
    if not (
        history_embedding.device == history_action.device == future_action.device
    ):
        raise ValueError("all rollout tensors must be on the same device")
    branches = int(future_action.shape[1])
    horizon = int(future_action.shape[2])
    if branches <= 0:
        raise ValueError("future_action must contain at least one branch")
    return batch_size, branches, horizon, num_keypoints, latent_dim


def rollout_action_branches(
    model: MatchedAnchorDynamicsPredictor,
    history_embedding: torch.Tensor,
    history_action: torch.Tensor,
    future_action: torch.Tensor,
    *,
    detach_between_steps: bool = False,
) -> torch.Tensor:
    """Recursively predict ``[B,S,P,K,D]`` and feed predictions back as history.

    The default preserves the graph through the entire horizon.  The explicit
    detach option exists only for diagnostic comparisons.
    """

    batch_size, branches, horizon, num_keypoints, latent_dim = _validate_rollout_inputs(
        model, history_embedding, history_action, future_action
    )
    history_size = int(model.history_size)
    embeddings = history_embedding[:, None].expand(-1, branches, -1, -1, -1)
    actions = history_action[:, None].expand(-1, branches, -1, -1)
    outputs: list[torch.Tensor] = []
    for step in range(horizon):
        current_action = future_action[:, :, step : step + 1].float()
        action_context = torch.cat((actions.float(), current_action), dim=2)
        flat_embedding = embeddings.reshape(
            batch_size * branches,
            history_size,
            num_keypoints,
            latent_dim,
        )
        flat_action = action_context.reshape(
            batch_size * branches, history_size, int(model.action_dim)
        )
        predicted_sequence = model.predict(flat_embedding, flat_action)
        if predicted_sequence.shape != flat_embedding.shape:
            raise ValueError(
                "predictor output must match its embedding context shape, got "
                f"{tuple(predicted_sequence.shape)}"
            )
        next_embedding = predicted_sequence[:, -1].reshape(
            batch_size, branches, num_keypoints, latent_dim
        )
        outputs.append(next_embedding)
        recurrent_embedding = (
            next_embedding.detach() if detach_between_steps else next_embedding
        )
        embeddings = torch.cat(
            (embeddings[:, :, 1:], recurrent_embedding[:, :, None]), dim=2
        )
        actions = action_context[:, :, 1:]
    if not outputs:
        return history_embedding.new_empty(
            batch_size, branches, 0, num_keypoints, latent_dim
        )
    return torch.stack(outputs, dim=2)


@dataclass(frozen=True)
class _TorchRNGState:
    cpu: torch.Tensor
    cuda: torch.Tensor | None
    cuda_device: torch.device | None


def _capture_torch_rng(device: torch.device) -> _TorchRNGState:
    cuda_device = device if device.type == "cuda" else None
    cuda_state = (
        torch.cuda.get_rng_state(cuda_device).clone()
        if cuda_device is not None
        else None
    )
    return _TorchRNGState(
        cpu=torch.random.get_rng_state().clone(),
        cuda=cuda_state,
        cuda_device=cuda_device,
    )


def _restore_torch_rng(state: _TorchRNGState) -> None:
    torch.random.set_rng_state(state.cpu)
    if state.cuda is not None and state.cuda_device is not None:
        torch.cuda.set_rng_state(state.cuda, state.cuda_device)


def matched_action_branch_rollout(
    factual: MatchedAnchorDynamicsPredictor,
    no_action: MatchedAnchorDynamicsPredictor,
    history_embedding: torch.Tensor,
    history_action: torch.Tensor,
    future_action: torch.Tensor,
    *,
    detach_between_steps: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run both complete rollouts from the same stochastic-layer RNG state."""

    starting_rng = _capture_torch_rng(history_embedding.device)
    factual_prediction = rollout_action_branches(
        factual,
        history_embedding,
        history_action,
        future_action,
        detach_between_steps=detach_between_steps,
    )
    ending_rng = _capture_torch_rng(history_embedding.device)
    _restore_torch_rng(starting_rng)
    try:
        no_action_prediction = rollout_action_branches(
            no_action,
            history_embedding,
            history_action,
            future_action,
            detach_between_steps=detach_between_steps,
        )
    finally:
        # One matched pair consumes one rollout's random draws globally.
        _restore_torch_rng(ending_rng)
    return factual_prediction, no_action_prediction


def raw_and_branch_centered_mse(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return all-horizon raw MSE and the optional branch-centered diagnostic."""

    if prediction.ndim != 5:
        raise ValueError("prediction must have shape [B,S,P,K,D]")
    if target.shape != prediction.shape:
        raise ValueError(
            f"target must have shape {tuple(prediction.shape)}, got {tuple(target.shape)}"
        )
    prediction_float = prediction.float()
    target_float = target.float()
    raw_mse = (prediction_float - target_float).square().mean()
    centered_prediction = prediction_float - prediction_float.mean(dim=1, keepdim=True)
    centered_target = target_float - target_float.mean(dim=1, keepdim=True)
    centered_mse = (centered_prediction - centered_target).square().mean()
    return raw_mse, centered_mse


def matched_action_branch_loss(
    factual: MatchedAnchorDynamicsPredictor,
    no_action: MatchedAnchorDynamicsPredictor,
    history_embedding: torch.Tensor,
    history_action: torch.Tensor,
    future_action: torch.Tensor,
    target_embedding: torch.Tensor,
    *,
    detach_between_steps: bool = False,
) -> MatchedActionBranchOutput:
    """Compute the matched closed-loop objective without predictor-head SIGReg."""

    factual_prediction, no_action_prediction = matched_action_branch_rollout(
        factual,
        no_action,
        history_embedding,
        history_action,
        future_action,
        detach_between_steps=detach_between_steps,
    )
    factual_raw, factual_centered = raw_and_branch_centered_mse(
        factual_prediction, target_embedding
    )
    no_action_raw, no_action_centered = raw_and_branch_centered_mse(
        no_action_prediction, target_embedding
    )
    return MatchedActionBranchOutput(
        factual_prediction=factual_prediction,
        no_action_prediction=no_action_prediction,
        factual_raw_mse=factual_raw,
        no_action_raw_mse=no_action_raw,
        factual_branch_centered_mse=factual_centered,
        no_action_branch_centered_mse=no_action_centered,
    )


def fixed_branch_derangement(
    sample_index: torch.Tensor, *, branch_count: int = A1_BRANCH_COUNT
) -> torch.Tensor:
    """Return a stable per-bundle cyclic derangement with shape ``[B,S]``."""

    indices = torch.as_tensor(sample_index, dtype=torch.int64)
    if indices.ndim != 1:
        raise ValueError("sample_index must have shape [B]")
    if branch_count < 2:
        raise ValueError("a branch derangement requires at least two branches")
    offsets = 1 + torch.remainder(indices, branch_count - 1)
    slots = torch.arange(branch_count, device=indices.device, dtype=torch.int64)
    permutation = torch.remainder(slots[None] + offsets[:, None], branch_count)
    if bool(torch.any(permutation == slots[None])):
        raise RuntimeError("constructed branch permutation is not a derangement")
    return permutation


def all_cyclic_branch_derangements(
    bundle_count: int,
    *,
    branch_count: int = A1_BRANCH_COUNT,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return all ``S - 1`` non-identity cyclic donor shifts as ``[B,S-1,S]``."""

    if bundle_count <= 0:
        raise ValueError("bundle_count must be positive")
    if branch_count < 2:
        raise ValueError("cyclic derangements require at least two branches")
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    slots = torch.arange(branch_count, device=target_device, dtype=torch.int64)
    shifts = torch.arange(1, branch_count, device=target_device, dtype=torch.int64)
    permutations = torch.remainder(
        slots[None, None] + shifts[None, :, None], branch_count
    ).expand(bundle_count, -1, -1)
    if bool(torch.any(permutations == slots[None, None])):
        raise RuntimeError("constructed cyclic permutations include fixed points")
    return permutations


def permute_action_branches(
    future_action: torch.Tensor, permutation: torch.Tensor
) -> torch.Tensor:
    """Apply one explicit branch permutation to each bundle."""

    if future_action.ndim != 4:
        raise ValueError("future_action must have shape [B,S,P,A]")
    batch_size, branches, horizon, action_dim = future_action.shape
    expected = (batch_size, branches)
    if tuple(permutation.shape) != expected:
        raise ValueError(
            f"permutation must have shape {expected}, got {tuple(permutation.shape)}"
        )
    permutation = permutation.to(device=future_action.device, dtype=torch.int64)
    expected_slots = torch.arange(branches, device=future_action.device).expand(
        batch_size, -1
    )
    if not torch.equal(torch.sort(permutation, dim=1).values, expected_slots):
        raise ValueError("every row must be a permutation of the branch slots")
    gather_index = permutation[:, :, None, None].expand(
        -1, -1, horizon, action_dim
    )
    return torch.gather(future_action, dim=1, index=gather_index)


def _text_vector(value: np.ndarray, name: str, sample_count: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (sample_count,):
        raise ValueError(f"{name} must have shape ({sample_count},)")
    if array.dtype.kind not in {"S", "U"}:
        raise TypeError(f"{name} must be a string array")
    return array.astype(str, copy=False)


def _identifier_vector(
    value: np.ndarray, name: str, sample_count: int
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (sample_count,):
        raise ValueError(f"{name} must have shape ({sample_count},)")
    if array.dtype.kind == "O":
        raise TypeError(f"{name} must not require pickle/object decoding")
    if array.dtype.kind not in {"S", "U", "i", "u"}:
        raise TypeError(f"{name} must contain strings or integers")
    if array.dtype.kind == "S":
        return array.astype(str)
    return array


def _validate_finite_in_chunks(name: str, array: np.ndarray) -> None:
    for start in range(0, int(array.shape[0]), 64):
        if not bool(np.isfinite(array[start : start + 64]).all()):
            raise ValueError(f"{name} contains non-finite values")


def load_action_branch_cache(
    path: str | Path,
    *,
    history_size: int = A1_HISTORY_SIZE,
    num_keypoints: int = A1_NUM_KEYPOINTS,
    latent_dim: int = A1_LATENT_DIM,
    action_dim: int = A1_ACTION_DIM,
    branch_count: int = A1_BRANCH_COUNT,
    minimum_horizon: int = A1_MIN_HORIZON,
    validate_finite: bool = True,
) -> ActionBranchCache:
    """Load and strictly validate the collector's cached-latent NPZ schema."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as archive:
        missing = REQUIRED_NPZ_FIELDS.difference(archive.files)
        if missing:
            raise ValueError(f"action-branch NPZ is missing {sorted(missing)}")
        extra = set(archive.files).difference(REQUIRED_NPZ_FIELDS)
        if extra:
            raise ValueError(f"action-branch NPZ has unexpected fields {sorted(extra)}")
        arrays = {name: np.asarray(archive[name]) for name in REQUIRED_NPZ_FIELDS}

    core_dimensions = (
        history_size == A1_HISTORY_SIZE
        and num_keypoints == A1_NUM_KEYPOINTS
        and latent_dim == A1_LATENT_DIM
        and action_dim == A1_ACTION_DIM
        and branch_count == A1_BRANCH_COUNT
        and minimum_horizon == A1_MIN_HORIZON
    )
    if core_dimensions:
        validate_a1_arrays(arrays)

    history_embedding = arrays["history_embedding"]
    history_action = arrays["history_action"]
    center_action = arrays["center_action"]
    future_action = arrays["future_action"]
    target_embedding = arrays["target_embedding"]
    if history_embedding.ndim != 4:
        raise ValueError("history_embedding must have shape [N,H,K,D]")
    sample_count = int(history_embedding.shape[0])
    if sample_count <= 0:
        raise ValueError("action-branch NPZ cannot be empty")
    expected_history = (sample_count, history_size, num_keypoints, latent_dim)
    if tuple(history_embedding.shape) != expected_history:
        raise ValueError(
            f"history_embedding must have shape {expected_history}, "
            f"got {tuple(history_embedding.shape)}"
        )
    expected_history_action = (sample_count, history_size - 1, action_dim)
    if tuple(history_action.shape) != expected_history_action:
        raise ValueError(
            f"history_action must have shape {expected_history_action}, "
            f"got {tuple(history_action.shape)}"
        )
    if (
        future_action.ndim != 4
        or tuple(future_action.shape[:2]) != (sample_count, branch_count)
        or int(future_action.shape[-1]) != action_dim
    ):
        raise ValueError(
            "future_action must have shape "
            f"[N,{branch_count},P,{action_dim}], got {tuple(future_action.shape)}"
        )
    horizon = int(future_action.shape[2])
    if horizon < minimum_horizon:
        raise ValueError(
            f"future_action horizon must be at least {minimum_horizon}, got {horizon}"
        )
    expected_center = (sample_count, horizon, action_dim)
    if tuple(center_action.shape) != expected_center:
        raise ValueError(
            f"center_action must have shape {expected_center}, "
            f"got {tuple(center_action.shape)}"
        )
    expected_target = (
        sample_count,
        branch_count,
        horizon,
        num_keypoints,
        latent_dim,
    )
    if tuple(target_embedding.shape) != expected_target:
        raise ValueError(
            f"target_embedding must have shape {expected_target}, "
            f"got {tuple(target_embedding.shape)}"
        )
    for name, array in (
        ("history_embedding", history_embedding),
        ("history_action", history_action),
        ("center_action", center_action),
        ("future_action", future_action),
        ("target_embedding", target_embedding),
    ):
        if array.dtype.kind != "f":
            raise TypeError(f"{name} must use a floating dtype")
        if validate_finite and not core_dimensions:
            _validate_finite_in_chunks(name, array)

    split = _text_vector(arrays["split"], "split", sample_count)
    task_index = _identifier_vector(
        arrays["task_index"], "task_index", sample_count
    )
    task_instance_index = _identifier_vector(
        arrays["task_instance_index"], "task_instance_index", sample_count
    )
    source_episode_id = _identifier_vector(
        arrays["source_episode_id"], "source_episode_id", sample_count
    )
    hazard_id = _identifier_vector(arrays["hazard_id"], "hazard_id", sample_count)
    episode_split: dict[tuple[str, str], str] = {}
    for task_value, episode_value, split_value in zip(
        task_index, source_episode_id, split, strict=True
    ):
        key = (str(task_value), str(episode_value))
        previous = episode_split.setdefault(key, str(split_value))
        if previous != str(split_value):
            raise ValueError(
                "one task/episode identity appears in multiple splits: "
                f"task={key[0]!r} episode={key[1]!r}"
            )

    return ActionBranchCache(
        path=source,
        history_embedding=history_embedding,
        history_action=history_action,
        center_action=center_action,
        future_action=future_action,
        target_embedding=target_embedding,
        split=split,
        task_index=task_index,
        task_instance_index=task_instance_index,
        source_episode_id=source_episode_id,
        hazard_id=hazard_id,
    )


__all__ = [
    "ACTION_BRANCH_TRAINING_PROTOCOL",
    "A1_ACTION_DIM",
    "A1_BRANCH_COUNT",
    "A1_HISTORY_SIZE",
    "A1_LATENT_DIM",
    "A1_MIN_HORIZON",
    "A1_NUM_KEYPOINTS",
    "BASE_SIGREG_WEIGHT",
    "BUDGET_STEPS",
    "CachedActionBranchDataset",
    "ActionBranchCache",
    "MatchedActionBranchOutput",
    "action_normalization_from_base_checkpoint",
    "all_cyclic_branch_derangements",
    "assert_matched_predictor_contract",
    "budget_steps",
    "fixed_branch_derangement",
    "load_action_branch_cache",
    "matched_action_branch_loss",
    "matched_action_branch_rollout",
    "raw_and_branch_centered_mse",
    "permute_action_branches",
    "rollout_action_branches",
    "validate_frozen_base_checkpoint",
]
