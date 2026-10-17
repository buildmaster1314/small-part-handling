"""Statistics for the preregistered random-hazard action-branch A1 gate.

The scorer consumes complete action-branch bundles.  Target and ordinary
prediction axes are ``[bundle, branch, horizon, anchor, latent]``.  The
shuffled-action control adds a cyclic-shift axis and has shape
``[bundle, shift, branch, horizon, anchor, latent]``.  These axes are never
flattened into independent observations.  The independent sampling unit is
the source physical episode; hazards, shifts, branches, horizons, and training
seeds remain nested inside it for every bootstrap draw.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any, TypeAlias

import numpy as np


DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 18_431
BRANCH_COUNT = 8
CYCLIC_SHIFT_COUNT = BRANCH_COUNT - 1
LATE_WINDOW_HORIZONS = (6, 7, 8)
TASK_NAMES = (
    "pick-place-v3",
    "button-press-v3",
    "reach-v3",
    "push-v3",
)
FACTUAL_METHOD = "factual"
NO_ACTION_METHOD = "no_action"
SHUFFLED_ACTION_METHOD = "shuffled_action"
HOLD_LAST_METHOD = "hold_last"
LATENT_CONSTANT_VELOCITY_METHOD = "latent_constant_velocity"
REQUIRED_PREDICTION_METHODS = (
    FACTUAL_METHOD,
    NO_ACTION_METHOD,
    SHUFFLED_ACTION_METHOD,
)
PRIMARY_BASELINES = (NO_ACTION_METHOD, SHUFFLED_ACTION_METHOD)
MINIMUM_RELATIVE_IMPROVEMENT = 0.05
MINIMUM_INFORMATIVE_BRANCH_VARIANCE = 0.01
MINIMUM_INFORMATIVE_BUNDLE_FRACTION = 0.80
MINIMUM_TASK_COUNT = 4
MINIMUM_SEED_COUNT = 3
MINIMUM_POSITIVE_TASK_FRACTION = 0.75
MINIMUM_POSITIVE_SEED_FRACTION = 2.0 / 3.0

Identity: TypeAlias = int | str
BundleKey: TypeAlias = tuple[str, Identity, Identity]
EpisodeKey: TypeAlias = tuple[str, Identity]


def _readonly(value: np.ndarray, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = np.array(value, copy=True, order="C", dtype=dtype)
    result.setflags(write=False)
    return result


def _numeric_array(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{label} must be numeric")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    return array


def _strict_integer(value: Any, *, label: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _finite_number(value: Any, *, label: str, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _identity_order(value: Identity) -> tuple[int, int | str]:
    return (0, value) if isinstance(value, int) else (1, value)


def _identity_item(value: Any, *, label: str) -> Identity:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be an integer or non-empty string")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str) and value:
        return value
    raise TypeError(f"{label} must be an integer or non-empty string")


def _identity_vector(value: Any, *, label: str, count: int) -> tuple[Identity, ...]:
    array = np.asarray(value)
    if array.shape != (count,):
        raise ValueError(f"{label} must have shape [N]")
    return tuple(
        _identity_item(item, label=f"{label}[{index}]")
        for index, item in enumerate(array.tolist())
    )


def _task_vector(value: Any, *, count: int) -> tuple[str, ...]:
    raw = _identity_vector(value, label="task_index or task", count=count)
    tasks: list[str] = []
    for item in raw:
        if isinstance(item, int):
            if not 0 <= item < len(TASK_NAMES):
                raise ValueError("task_index must be in the range 0..3")
            tasks.append(TASK_NAMES[item])
        else:
            tasks.append(item)
    return tuple(tasks)


def _seed_vector(value: Any, *, count: int) -> tuple[int, ...]:
    array = np.asarray(value)
    if array.shape != (count,):
        raise ValueError("seed must have shape [N]")
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.integer):
        raise TypeError("seed must be a one-dimensional integer array")
    return tuple(int(item) for item in array.tolist())


def _integer_vector(value: Any, *, label: str, count: int) -> tuple[int, ...]:
    array = np.asarray(value)
    if array.shape != (count,):
        raise ValueError(f"{label} must have shape [N]")
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{label} must be a one-dimensional integer array")
    return tuple(int(item) for item in array.tolist())


def _split_vector(value: Any, *, count: int) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.shape != (count,):
        raise ValueError("split must have shape [N]")
    allowed = {"train", "development", "test", "smoke"}
    result = tuple(str(item) for item in array.tolist())
    if any(item not in allowed for item in result):
        raise ValueError(
            "split values must be train, development, test, or smoke"
        )
    return result


@dataclass(frozen=True)
class TrainMoments:
    """Per-anchor/per-dimension moments fitted from training targets only."""

    mean: np.ndarray
    scale: np.ndarray
    count: int
    epsilon: float = 1e-8
    floored_dimensions: int = 0

    def __post_init__(self) -> None:
        mean = _numeric_array(self.mean, label="train moment mean")
        scale = _numeric_array(self.scale, label="train moment scale")
        if mean.ndim != 2 or scale.shape != mean.shape:
            raise ValueError("train moment mean and scale must have shape [K,D]")
        if mean.size == 0:
            raise ValueError("train moments must not be empty")
        if np.any(scale <= 0.0):
            raise ValueError("train moment scale must be strictly positive")
        count = _strict_integer(self.count, label="train moment count", minimum=1)
        epsilon = _finite_number(self.epsilon, label="epsilon", positive=True)
        floored = _strict_integer(
            self.floored_dimensions,
            label="floored_dimensions",
            minimum=0,
        )
        if floored > mean.size:
            raise ValueError("floored_dimensions exceeds the moment dimension count")
        object.__setattr__(self, "mean", _readonly(mean, dtype=np.dtype(np.float64)))
        object.__setattr__(
            self, "scale", _readonly(scale, dtype=np.dtype(np.float64))
        )
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "floored_dimensions", floored)

    @property
    def std(self) -> np.ndarray:
        """Alias used by NPZ producers that call the scale a standard deviation."""

        return self.scale

    def transform(self, value: np.ndarray, *, label: str = "latent array") -> np.ndarray:
        return whiten_latents(value, self, label=label)


def fit_train_moments(
    train_target: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> TrainMoments:
    """Fit population moments over training bundle/branch/horizon axes."""

    values = _numeric_array(train_target, label="train_target")
    if values.ndim != 5:
        raise ValueError("train_target must have shape [N,C,P,K,D]")
    if any(size <= 0 for size in values.shape):
        raise ValueError("train_target axes must be non-empty")
    if values.shape[1] != BRANCH_COUNT:
        raise ValueError(f"train_target must contain exactly {BRANCH_COUNT} branches")
    threshold = _finite_number(epsilon, label="epsilon", positive=True)
    work = values.astype(np.float64, copy=False)
    mean = np.mean(work, axis=(0, 1, 2), dtype=np.float64)
    raw_scale = np.std(work, axis=(0, 1, 2), ddof=0, dtype=np.float64)
    floored = raw_scale <= threshold
    scale = np.where(floored, 1.0, raw_scale)
    return TrainMoments(
        mean=mean,
        scale=scale,
        count=int(np.prod(values.shape[:3], dtype=np.int64)),
        epsilon=threshold,
        floored_dimensions=int(np.count_nonzero(floored)),
    )


fit_train_whitening_moments = fit_train_moments


def train_moments_from_arrays(
    mean: np.ndarray,
    scale: np.ndarray,
    *,
    count: int = 1,
    epsilon: float = 1e-8,
) -> TrainMoments:
    """Construct validated train moments supplied by an upstream trainer."""

    return TrainMoments(mean=mean, scale=scale, count=count, epsilon=epsilon)


def whiten_latents(
    value: np.ndarray,
    moments: TrainMoments,
    *,
    label: str = "latent array",
) -> np.ndarray:
    """Whiten an array whose final two axes are ``[anchor, latent]``."""

    if not isinstance(moments, TrainMoments):
        raise TypeError("moments must be TrainMoments fitted on the train split")
    array = _numeric_array(value, label=label)
    if array.ndim < 2 or array.shape[-2:] != moments.mean.shape:
        raise ValueError(
            f"{label} trailing dimensions must match train moments "
            f"{moments.mean.shape}"
        )
    shape = (1,) * (array.ndim - 2) + moments.mean.shape
    result = (
        array.astype(np.float64, copy=False) - moments.mean.reshape(shape)
    ) / moments.scale.reshape(shape)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} whitening produced non-finite values")
    return result


def center_whitened_branches(
    value: np.ndarray,
    moments: TrainMoments,
    *,
    label: str = "latent bundle",
) -> np.ndarray:
    """Whiten, then independently center each bundle/horizon over branches."""

    array = _numeric_array(value, label=label)
    if array.ndim != 5:
        raise ValueError(f"{label} must have shape [N,C,P,K,D]")
    if array.shape[1] != BRANCH_COUNT:
        raise ValueError(f"{label} must contain exactly {BRANCH_COUNT} branches")
    whitened = whiten_latents(array, moments, label=label)
    centered = whitened - np.mean(whitened, axis=1, keepdims=True, dtype=np.float64)
    if not np.isfinite(centered).all():
        raise ValueError(f"{label} branch centering produced non-finite values")
    return centered


def branch_centered_whitened_mse(
    target: np.ndarray,
    prediction: np.ndarray,
    moments: TrainMoments,
) -> np.ndarray:
    """Return one centered/whitened MSE per complete action-branch bundle."""

    target_array = _numeric_array(target, label="target")
    prediction_array = _numeric_array(prediction, label="prediction")
    if target_array.ndim != 5:
        raise ValueError("target must have shape [N,C,P,K,D]")
    target_centered = center_whitened_branches(
        target_array, moments, label="target"
    )
    if prediction_array.ndim == 5:
        predictions = (prediction_array,)
    elif prediction_array.ndim == 6 and prediction_array.shape == (
        target_array.shape[0],
        target_array.shape[1] - 1,
        *target_array.shape[1:],
    ):
        predictions = tuple(
            prediction_array[:, shift]
            for shift in range(prediction_array.shape[1])
        )
    else:
        raise ValueError(
            "prediction must match [N,C,P,K,D] or have shuffled-control "
            "shape [N,C-1,C,P,K,D]"
        )
    losses: list[np.ndarray] = []
    for candidate in predictions:
        if candidate.shape != target_array.shape:
            raise ValueError("prediction must match target [N,C,P,K,D] axes")
        prediction_centered = center_whitened_branches(
            candidate, moments, label="prediction"
        )
        squared = np.square(target_centered - prediction_centered)
        if not np.isfinite(squared).all():
            raise ValueError("centered/whitened squared error is non-finite")
        losses.append(
            np.mean(squared, axis=(1, 2, 3, 4), dtype=np.float64)
        )
    return np.mean(np.stack(losses, axis=1), axis=1, dtype=np.float64)


centered_whitened_bundle_mse = branch_centered_whitened_mse


def raw_whitened_bundle_mse(
    target: np.ndarray,
    prediction: np.ndarray,
    moments: TrainMoments,
) -> np.ndarray:
    """Return raw trajectory MSE as a secondary, non-gating diagnostic."""

    target_array = _numeric_array(target, label="target")
    prediction_array = _numeric_array(prediction, label="prediction")
    if target_array.ndim != 5:
        raise ValueError("target must have shape [N,C,P,K,D]")
    if prediction_array.ndim == 5:
        predictions = (prediction_array,)
    elif prediction_array.ndim == 6 and prediction_array.shape == (
        target_array.shape[0],
        target_array.shape[1] - 1,
        *target_array.shape[1:],
    ):
        predictions = tuple(
            prediction_array[:, shift]
            for shift in range(prediction_array.shape[1])
        )
    else:
        raise ValueError(
            "prediction must match [N,C,P,K,D] or have shuffled-control "
            "shape [N,C-1,C,P,K,D]"
        )
    target_whitened = whiten_latents(target_array, moments, label="target")
    losses: list[np.ndarray] = []
    for candidate in predictions:
        if candidate.shape != target_array.shape:
            raise ValueError("prediction must match target [N,C,P,K,D] axes")
        residual = target_whitened - whiten_latents(
            candidate, moments, label="prediction"
        )
        squared = np.square(residual)
        if not np.isfinite(squared).all():
            raise ValueError("raw whitened squared error is non-finite")
        losses.append(
            np.mean(squared, axis=(1, 2, 3, 4), dtype=np.float64)
        )
    return np.mean(np.stack(losses, axis=1), axis=1, dtype=np.float64)


def validate_within_bundle_derangement(
    permutation: np.ndarray | Sequence[int],
    *,
    bundle_count: int | None = None,
    branch_count: int = BRANCH_COUNT,
) -> np.ndarray:
    """Validate donor indices where ``p[target_branch]`` is the action donor."""

    branches = _strict_integer(branch_count, label="branch_count", minimum=2)
    array = np.asarray(permutation)
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.integer):
        raise TypeError("shuffle_permutation must use an integer dtype")
    if array.ndim == 1:
        if array.shape != (branches,):
            raise ValueError("one derangement must have shape [C]")
        rows = array[None, :]
    elif array.ndim == 2 and array.shape[1] == branches:
        rows = array
    else:
        raise ValueError("shuffle_permutation must have shape [C] or [N,C]")
    if bundle_count is not None:
        count = _strict_integer(bundle_count, label="bundle_count", minimum=1)
        if rows.shape[0] == 1 and count != 1:
            rows = np.broadcast_to(rows, (count, branches))
        elif rows.shape[0] != count:
            raise ValueError("shuffle_permutation row count must match N")
    expected = np.arange(branches, dtype=np.int64)
    normalized = rows.astype(np.int64, copy=False)
    for index, row in enumerate(normalized):
        if not np.array_equal(np.sort(row), expected):
            raise ValueError(
                f"shuffle_permutation row {index} is not a branch permutation"
            )
        if np.any(row == expected):
            raise ValueError(
                f"shuffle_permutation row {index} is not a derangement"
            )
    return _readonly(normalized, dtype=np.dtype(np.int64))


def validate_all_cyclic_branch_shifts(
    permutation: np.ndarray | Sequence[int],
    *,
    bundle_count: int | None = None,
    branch_count: int = BRANCH_COUNT,
) -> np.ndarray:
    """Validate all non-identity cyclic donor shifts for every bundle."""

    branches = _strict_integer(branch_count, label="branch_count", minimum=2)
    array = np.asarray(permutation)
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.integer):
        raise TypeError("shuffle_permutation must use an integer dtype")
    expected_shape = (branches - 1, branches)
    if array.ndim == 2:
        if array.shape != expected_shape:
            raise ValueError(
                "shared shuffle_permutation must have shape [C-1,C]"
            )
        rows = array[None, :, :]
    elif array.ndim == 3 and array.shape[1:] == expected_shape:
        rows = array
    else:
        raise ValueError(
            "shuffle_permutation must have shape [C-1,C] or [N,C-1,C]"
        )
    if bundle_count is not None:
        count = _strict_integer(bundle_count, label="bundle_count", minimum=1)
        if rows.shape[0] == 1 and count != 1:
            rows = np.broadcast_to(rows, (count, *expected_shape))
        elif rows.shape[0] != count:
            raise ValueError("shuffle_permutation row count must match N")

    slots = np.arange(branches, dtype=np.int64)
    expected_shifts = set(range(1, branches))
    normalized = rows.astype(np.int64, copy=False)
    for bundle_index, bundle in enumerate(normalized):
        observed_shifts: list[int] = []
        for shift_index, row in enumerate(bundle):
            offsets = np.remainder(row - slots, branches)
            if not np.all(offsets == offsets[0]) or int(offsets[0]) == 0:
                raise ValueError(
                    "shuffle_permutation must contain only non-identity "
                    f"cyclic shifts; bundle={bundle_index}, shift={shift_index}"
                )
            observed_shifts.append(int(offsets[0]))
        if set(observed_shifts) != expected_shifts or len(set(observed_shifts)) != (
            branches - 1
        ):
            raise ValueError(
                "shuffle_permutation must cover each non-identity cyclic "
                f"shift exactly once; bundle={bundle_index}"
            )
    return _readonly(normalized, dtype=np.dtype(np.int64))


def make_all_cyclic_branch_shifts(
    bundle_count: int,
    *,
    branch_count: int = BRANCH_COUNT,
) -> np.ndarray:
    """Return canonical donor indices for every non-identity cyclic shift."""

    count = _strict_integer(bundle_count, label="bundle_count", minimum=1)
    branches = _strict_integer(branch_count, label="branch_count", minimum=2)
    slots = np.arange(branches, dtype=np.int64)
    shifts = np.arange(1, branches, dtype=np.int64)
    permutations = np.remainder(
        slots[None, None, :] + shifts[None, :, None], branches
    )
    return _readonly(
        np.broadcast_to(permutations, (count, branches - 1, branches)),
        dtype=np.dtype(np.int64),
    )


def make_within_bundle_derangement(
    bundle_count: int,
    *,
    branch_count: int = BRANCH_COUNT,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> np.ndarray:
    """Create reproducible within-bundle branch derangements."""

    count = _strict_integer(bundle_count, label="bundle_count", minimum=1)
    branches = _strict_integer(branch_count, label="branch_count", minimum=2)
    random_seed = _strict_integer(seed, label="seed", minimum=0)
    rng = np.random.default_rng(random_seed)
    output = np.empty((count, branches), dtype=np.int64)
    expected = np.arange(branches, dtype=np.int64)
    for index in range(count):
        while True:
            candidate = rng.permutation(branches)
            if np.all(candidate != expected):
                output[index] = candidate
                break
    return _readonly(output)


within_bundle_derangement = make_within_bundle_derangement


def build_secondary_predictions(
    history: np.ndarray,
    *,
    branch_count: int,
    horizon_count: int,
) -> dict[str, np.ndarray]:
    """Build branch-invariant hold-last and latent constant-velocity baselines."""

    values = _numeric_array(history, label="history")
    if values.ndim != 4:
        raise ValueError("history must have shape [N,H,K,D]")
    if values.shape[1] < 2:
        raise ValueError("history needs at least two steps for latent CV")
    branches = _strict_integer(branch_count, label="branch_count", minimum=1)
    horizons = _strict_integer(horizon_count, label="horizon_count", minimum=1)
    last = values[:, -1].astype(np.float64, copy=False)
    velocity = last - values[:, -2].astype(np.float64, copy=False)
    factors = np.arange(1, horizons + 1, dtype=np.float64).reshape(1, horizons, 1, 1)
    hold_horizon = np.broadcast_to(last[:, None], (values.shape[0], horizons, *last.shape[1:]))
    cv_horizon = last[:, None] + factors * velocity[:, None]
    output_shape = (values.shape[0], branches, horizons, *last.shape[1:])
    return {
        HOLD_LAST_METHOD: np.broadcast_to(hold_horizon[:, None], output_shape),
        LATENT_CONSTANT_VELOCITY_METHOD: np.broadcast_to(
            cv_horizon[:, None], output_shape
        ),
    }


@dataclass(frozen=True)
class _ValidatedPanel:
    target: np.ndarray
    predictions: Mapping[str, np.ndarray]
    history: np.ndarray
    tasks: tuple[str, ...]
    episodes: tuple[Identity, ...]
    hazards: tuple[Identity, ...]
    seeds: tuple[int, ...]
    task_instances: tuple[int, ...] | None
    splits: tuple[str, ...] | None
    permutations: np.ndarray
    no_action_max_branch_deviation: float


_PREDICTION_ALIASES = {
    FACTUAL_METHOD: ("factual", "factual_action", "factual_prediction"),
    NO_ACTION_METHOD: (
        "no_action",
        "matched_no_action",
        "no_action_prediction",
    ),
    SHUFFLED_ACTION_METHOD: (
        "shuffled_action",
        "within_bundle_shuffled",
        "shuffled_action_prediction",
    ),
}


def _canonical_predictions(
    predictions: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if not isinstance(predictions, Mapping):
        raise TypeError("predictions must be a mapping of method name to array")
    if not predictions:
        raise ValueError("predictions must not be empty")
    result: dict[str, np.ndarray] = {}
    consumed: set[str] = set()
    for role, aliases in _PREDICTION_ALIASES.items():
        present = [name for name in aliases if name in predictions]
        if not present:
            raise ValueError(f"required prediction method {role!r} is absent")
        reference = np.asarray(predictions[present[0]])
        for name in present[1:]:
            if not np.array_equal(reference, np.asarray(predictions[name])):
                raise ValueError(f"conflicting aliases were supplied for {role!r}")
        result[role] = reference
        consumed.update(present)
    for name, value in predictions.items():
        if not isinstance(name, str) or not name:
            raise TypeError("prediction method names must be non-empty strings")
        if name in consumed:
            continue
        if name in {HOLD_LAST_METHOD, LATENT_CONSTANT_VELOCITY_METHOD}:
            raise ValueError(f"{name} is generated from history and cannot be supplied")
        result[name] = np.asarray(value)
    return result


def _validate_panel(
    target: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    history: np.ndarray,
    task: np.ndarray | Sequence[Identity],
    episode: np.ndarray | Sequence[Identity],
    hazard: np.ndarray | Sequence[Identity],
    seed: np.ndarray | Sequence[int],
    shuffle_permutation: np.ndarray | Sequence[int],
    *,
    task_instance_index: np.ndarray | Sequence[int] | None,
    split: np.ndarray | Sequence[str] | None,
    stage: str,
    no_action_invariance_tolerance: float,
) -> _ValidatedPanel:
    target_array = _numeric_array(target, label="target")
    if target_array.ndim != 5:
        raise ValueError("target must have shape [N,C,P,K,D]")
    if any(size <= 0 for size in target_array.shape):
        raise ValueError("target axes must be non-empty")
    n, branches, _horizons, anchors, latent = target_array.shape
    if branches != BRANCH_COUNT:
        raise ValueError(f"target must contain exactly {BRANCH_COUNT} branches")
    if _horizons < max(LATE_WINDOW_HORIZONS):
        raise ValueError("target must contain horizons H1 through H8")

    canonical = _canonical_predictions(predictions)
    checked_predictions: dict[str, np.ndarray] = {}
    for name, value in canonical.items():
        array = _numeric_array(value, label=f"prediction {name!r}")
        expected_shape = (
            (n, branches - 1, branches, _horizons, anchors, latent)
            if name == SHUFFLED_ACTION_METHOD
            else target_array.shape
        )
        if array.shape != expected_shape:
            raise ValueError(
                f"prediction {name!r} must have shape {expected_shape}"
            )
        checked_predictions[name] = array

    history_array = _numeric_array(history, label="history")
    if history_array.ndim != 4 or history_array.shape[0] != n:
        raise ValueError("history must have shape [N,H,K,D]")
    if history_array.shape[1] < 2 or history_array.shape[2:] != (anchors, latent):
        raise ValueError(
            "history must have H >= 2 and the same [K,D] axes as target"
        )

    tasks = _task_vector(task, count=n)
    episodes = _identity_vector(episode, label="episode", count=n)
    hazards = _identity_vector(hazard, label="hazard", count=n)
    seeds = _seed_vector(seed, count=n)
    task_instances = (
        _integer_vector(
            task_instance_index,
            label="task_instance_index",
            count=n,
        )
        if task_instance_index is not None
        else None
    )
    splits = _split_vector(split, count=n) if split is not None else None
    if splits is not None:
        expected_split = {
            "smoke": "smoke",
            "screen": "development",
            "final": "test",
        }[stage]
        observed_splits = set(splits)
        if observed_splits != {expected_split}:
            raise ValueError(
                f"stage {stage!r} requires split={expected_split!r}; "
                f"observed={sorted(observed_splits)!r}"
            )
    permutations = validate_all_cyclic_branch_shifts(
        shuffle_permutation,
        bundle_count=n,
        branch_count=branches,
    )

    tolerance = _finite_number(
        no_action_invariance_tolerance,
        label="no_action_invariance_tolerance",
    )
    if tolerance < 0.0:
        raise ValueError("no_action_invariance_tolerance must be non-negative")
    no_action = checked_predictions[NO_ACTION_METHOD].astype(np.float64, copy=False)
    branch_deviation = np.abs(no_action - no_action[:, :1])
    max_deviation = float(np.max(branch_deviation))
    if not np.allclose(no_action, no_action[:, :1], rtol=0.0, atol=tolerance):
        raise ValueError(
            "no_action prediction must be branch-invariant within every "
            "bundle/horizon"
        )

    row_keys: set[tuple[str, Identity, Identity, int]] = set()
    bundle_rows: dict[BundleKey, list[int]] = {}
    for index, key in enumerate(zip(tasks, episodes, hazards, seeds, strict=True)):
        if key in row_keys:
            raise ValueError(f"duplicate task/episode/hazard/seed row: {key!r}")
        row_keys.add(key)
        bundle_rows.setdefault(key[:3], []).append(index)
    expected_seeds = set(seeds)
    for key, rows in bundle_rows.items():
        observed = {seeds[index] for index in rows}
        if observed != expected_seeds or len(rows) != len(expected_seeds):
            raise ValueError(
                f"bundle {key!r} does not contain the complete training-seed panel"
            )
        reference = rows[0]
        for index in rows[1:]:
            if not np.array_equal(target_array[reference], target_array[index]):
                raise ValueError("target must be identical across seeds in one bundle")
            if not np.array_equal(history_array[reference], history_array[index]):
                raise ValueError("history must be identical across seeds in one bundle")
            if (
                task_instances is not None
                and task_instances[reference] != task_instances[index]
            ):
                raise ValueError(
                    "task_instance_index must be identical across seeds in one bundle"
                )
    if task_instances is not None:
        instance_by_episode: dict[EpisodeKey, int] = {}
        episode_by_instance: dict[tuple[str, int], Identity] = {}
        for task_name, episode_id, instance in zip(
            tasks, episodes, task_instances, strict=True
        ):
            key = (task_name, episode_id)
            previous = instance_by_episode.setdefault(key, instance)
            if previous != instance:
                raise ValueError(
                    "one source_episode_id changes task_instance_index"
                )
            reverse_key = (task_name, instance)
            previous_episode = episode_by_instance.setdefault(
                reverse_key, episode_id
            )
            if previous_episode != episode_id:
                raise ValueError(
                    "one task_instance_index maps to multiple source episodes"
                )

    return _ValidatedPanel(
        target=target_array,
        predictions=checked_predictions,
        history=history_array,
        tasks=tasks,
        episodes=episodes,
        hazards=hazards,
        seeds=seeds,
        task_instances=task_instances,
        splits=splits,
        permutations=permutations,
        no_action_max_branch_deviation=max_deviation,
    )


@dataclass(frozen=True)
class _PanelLayout:
    tasks: tuple[str, ...]
    seeds: tuple[int, ...]
    episodes_by_task: Mapping[str, tuple[Identity, ...]]
    rows_by_task_episode_seed: Mapping[tuple[str, Identity, int], tuple[int, ...]]
    bundle_rows: Mapping[BundleKey, tuple[int, ...]]


def _panel_layout(panel: _ValidatedPanel) -> _PanelLayout:
    present_tasks = set(panel.tasks)
    tasks = tuple(task for task in TASK_NAMES if task in present_tasks) + tuple(
        sorted(present_tasks.difference(TASK_NAMES))
    )
    seeds = tuple(sorted(set(panel.seeds)))
    episode_sets: dict[str, set[Identity]] = {task: set() for task in tasks}
    rows: dict[tuple[str, Identity, int], list[int]] = {}
    bundles: dict[BundleKey, list[int]] = {}
    for index, (task, episode, hazard, seed) in enumerate(
        zip(
            panel.tasks,
            panel.episodes,
            panel.hazards,
            panel.seeds,
            strict=True,
        )
    ):
        episode_sets[task].add(episode)
        rows.setdefault((task, episode, seed), []).append(index)
        bundles.setdefault((task, episode, hazard), []).append(index)
    episodes_by_task = {
        task: tuple(sorted(values, key=_identity_order))
        for task, values in episode_sets.items()
    }
    for task in tasks:
        for episode in episodes_by_task[task]:
            for seed in seeds:
                key = (task, episode, seed)
                if key not in rows:
                    raise ValueError(
                        f"episode {(task, episode)!r} is missing training seed {seed}"
                    )
    immutable_rows = {
        key: tuple(
            sorted(indices, key=lambda index: _identity_order(panel.hazards[index]))
        )
        for key, indices in rows.items()
    }
    immutable_bundles = {
        key: tuple(sorted(indices, key=lambda index: panel.seeds[index]))
        for key, indices in bundles.items()
    }
    return _PanelLayout(
        tasks=tasks,
        seeds=seeds,
        episodes_by_task=episodes_by_task,
        rows_by_task_episode_seed=immutable_rows,
        bundle_rows=immutable_bundles,
    )


@dataclass(frozen=True)
class _MethodReduction:
    episode_values: Mapping[str, np.ndarray]
    task_values: Mapping[str, float]
    task_seed_values: Mapping[str, Mapping[int, float]]


def _reduce_bundle_losses(
    bundle_losses: np.ndarray,
    *,
    layout: _PanelLayout,
) -> _MethodReduction:
    losses = _numeric_array(bundle_losses, label="bundle losses").astype(
        np.float64, copy=False
    )
    expected_count = sum(
        len(indices) for indices in layout.rows_by_task_episode_seed.values()
    )
    if losses.shape != (expected_count,):
        raise ValueError("bundle losses must have shape [N]")
    if np.any(losses < 0.0):
        raise ValueError("bundle losses must be non-negative")

    episode_values: dict[str, np.ndarray] = {}
    task_values: dict[str, float] = {}
    task_seed_values: dict[str, dict[int, float]] = {}
    for task in layout.tasks:
        episodes = layout.episodes_by_task[task]
        by_seed: dict[int, np.ndarray] = {}
        for seed in layout.seeds:
            by_seed[seed] = np.asarray(
                [
                    np.mean(
                        losses[
                            list(layout.rows_by_task_episode_seed[(task, episode, seed)])
                        ],
                        dtype=np.float64,
                    )
                    for episode in episodes
                ],
                dtype=np.float64,
            )
        stacked = np.stack([by_seed[seed] for seed in layout.seeds], axis=0)
        episode_values[task] = np.mean(stacked, axis=0, dtype=np.float64)
        task_seed_values[task] = {
            seed: float(np.mean(by_seed[seed], dtype=np.float64))
            for seed in layout.seeds
        }
        task_values[task] = float(
            np.mean(episode_values[task], dtype=np.float64)
        )
    return _MethodReduction(episode_values, task_values, task_seed_values)


def _method_report(
    reduction: _MethodReduction,
    *,
    layout: _PanelLayout,
) -> dict[str, Any]:
    return {
        "task_macro_mse": float(
            np.mean([reduction.task_values[task] for task in layout.tasks])
        ),
        "per_task": {
            task: {
                "mse": reduction.task_values[task],
                "source_physical_episode_count": len(
                    layout.episodes_by_task[task]
                ),
                "per_seed": {
                    str(seed): {
                        "mse": reduction.task_seed_values[task][seed]
                    }
                    for seed in layout.seeds
                },
            }
            for task in layout.tasks
        },
        "per_seed": {
            str(seed): {
                "task_macro_mse": float(
                    np.mean(
                        [
                            reduction.task_seed_values[task][seed]
                            for task in layout.tasks
                        ]
                    )
                )
            }
            for seed in layout.seeds
        },
    }


def _relative_effect(factual: float, baseline: float, *, label: str) -> float:
    if baseline < 0.0 or factual < 0.0:
        raise ValueError(f"{label} losses must be non-negative")
    if baseline == 0.0:
        if factual == 0.0:
            return 0.0
        raise ValueError(f"{label} baseline loss is zero while factual loss is positive")
    result = 1.0 - factual / baseline
    if not math.isfinite(result):
        raise ValueError(f"{label} effect must remain finite")
    return float(result)


def _relative_effect_array(
    factual: np.ndarray,
    baseline: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    if factual.shape != baseline.shape:
        raise RuntimeError(f"{label} bootstrap arrays differ")
    positive = baseline > 0.0
    invalid = (~positive) & (factual > 0.0)
    if np.any(invalid):
        raise ValueError(
            f"{label} bootstrap produced zero baseline loss with positive factual loss"
        )
    output = np.zeros_like(factual, dtype=np.float64)
    output[positive] = 1.0 - factual[positive] / baseline[positive]
    if not np.isfinite(output).all():
        raise ValueError(f"{label} bootstrap effect must remain finite")
    return output


def _bootstrap_draws(
    layout: _PanelLayout,
    *,
    samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        task: rng.integers(
            0,
            len(layout.episodes_by_task[task]),
            size=(samples, len(layout.episodes_by_task[task])),
        )
        for task in layout.tasks
    }


def _comparison_report(
    factual: _MethodReduction,
    baseline: _MethodReduction,
    *,
    factual_method: str,
    baseline_method: str,
    layout: _PanelLayout,
    draws: Mapping[str, np.ndarray] | None,
) -> dict[str, Any]:
    per_task: dict[str, dict[str, Any]] = {}
    task_effects: list[float] = []
    task_bootstrap: dict[str, np.ndarray] = {}
    for task in layout.tasks:
        effect = _relative_effect(
            factual.task_values[task],
            baseline.task_values[task],
            label=f"{task}/{baseline_method}",
        )
        task_effects.append(effect)
        per_seed = {
            str(seed): {
                "factual_mse": factual.task_seed_values[task][seed],
                "baseline_mse": baseline.task_seed_values[task][seed],
                "effect": _relative_effect(
                    factual.task_seed_values[task][seed],
                    baseline.task_seed_values[task][seed],
                    label=f"{task}/{seed}/{baseline_method}",
                ),
            }
            for seed in layout.seeds
        }
        row: dict[str, Any] = {
            "factual_mse": factual.task_values[task],
            "baseline_mse": baseline.task_values[task],
            "effect": effect,
            "relative_improvement": effect,
            "positive": effect > 0.0,
            "source_physical_episode_count": len(layout.episodes_by_task[task]),
            "per_seed": per_seed,
        }
        if draws is not None:
            indices = draws[task]
            factual_boot = np.mean(
                factual.episode_values[task][indices], axis=1, dtype=np.float64
            )
            baseline_boot = np.mean(
                baseline.episode_values[task][indices], axis=1, dtype=np.float64
            )
            values = _relative_effect_array(
                factual_boot,
                baseline_boot,
                label=f"{task}/{baseline_method}",
            )
            task_bootstrap[task] = values
            row["effect_95_ci"] = [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ]
        else:
            row["effect_95_ci"] = None
        per_task[task] = row

    per_seed: dict[str, dict[str, Any]] = {}
    for seed in layout.seeds:
        effects = {
            task: float(per_task[task]["per_seed"][str(seed)]["effect"])
            for task in layout.tasks
        }
        macro = float(np.mean(list(effects.values())))
        per_seed[str(seed)] = {
            "effect": macro,
            "relative_improvement": macro,
            "positive": macro > 0.0,
            "per_task_effect": effects,
        }

    macro_effect = float(np.mean(task_effects))
    if draws is not None:
        bootstrap_effect = np.mean(
            np.stack([task_bootstrap[task] for task in layout.tasks], axis=0),
            axis=0,
            dtype=np.float64,
        )
        ci: list[float] | None = [
            float(np.quantile(bootstrap_effect, 0.025)),
            float(np.quantile(bootstrap_effect, 0.975)),
        ]
    else:
        ci = None
    positive_tasks = sum(bool(row["positive"]) for row in per_task.values())
    positive_seeds = sum(bool(row["positive"]) for row in per_seed.values())
    return {
        "factual_method": factual_method,
        "baseline_method": baseline_method,
        "orientation": "positive_means_factual_has_lower_mse",
        "effect": macro_effect,
        "relative_improvement": macro_effect,
        "effect_95_ci": ci,
        "relative_improvement_95_ci": ci,
        "factual_task_macro_mse": float(
            np.mean([factual.task_values[task] for task in layout.tasks])
        ),
        "baseline_task_macro_mse": float(
            np.mean([baseline.task_values[task] for task in layout.tasks])
        ),
        "positive_task_count": positive_tasks,
        "task_count": len(layout.tasks),
        "positive_seed_count": positive_seeds,
        "seed_count": len(layout.seeds),
        "per_task": per_task,
        "per_seed": per_seed,
    }


def _raw_mse_sanity_report(
    factual: _MethodReduction,
    no_action: _MethodReduction,
    *,
    layout: _PanelLayout,
) -> dict[str, Any]:
    """Check that factual rollout does not trade raw accuracy for centered fit."""

    factual_macro = float(
        np.mean([factual.task_values[task] for task in layout.tasks])
    )
    no_action_macro = float(
        np.mean([no_action.task_values[task] for task in layout.tasks])
    )
    return {
        "metric": "raw_whitened_trajectory_mse",
        "aggregation": "task_macro",
        "factual_mse": factual_macro,
        "no_action_mse": no_action_macro,
        "margin_no_action_minus_factual": no_action_macro - factual_macro,
        "operator": "<=",
        "passed": factual_macro <= no_action_macro,
        "per_task": {
            task: {
                "factual_mse": factual.task_values[task],
                "no_action_mse": no_action.task_values[task],
                "passed": factual.task_values[task] <= no_action.task_values[task],
            }
            for task in layout.tasks
        },
    }


def _informative_report(
    target_centered: np.ndarray,
    *,
    layout: _PanelLayout,
    threshold: float,
) -> dict[str, Any]:
    variance = np.mean(
        np.square(target_centered), axis=(1, 2, 3, 4), dtype=np.float64
    )
    per_task_counts = {
        task: {"informative": 0, "total": 0} for task in layout.tasks
    }
    unique_values: list[float] = []
    for key, rows in layout.bundle_rows.items():
        task = key[0]
        value = float(variance[rows[0]])
        if not np.allclose(variance[list(rows)], value, rtol=0.0, atol=1e-12):
            raise ValueError("target branch variance changes across training seeds")
        unique_values.append(value)
        per_task_counts[task]["total"] += 1
        per_task_counts[task]["informative"] += int(value >= threshold)
    per_task = {}
    for task, counts in per_task_counts.items():
        fraction = counts["informative"] / counts["total"]
        per_task[task] = {
            "informative_bundle_count": counts["informative"],
            "bundle_count": counts["total"],
            "fraction": fraction,
            "passed": fraction >= MINIMUM_INFORMATIVE_BUNDLE_FRACTION,
        }
    return {
        "definition": (
            "mean population variance across whitened target branches, then "
            "mean over horizon/anchor/latent dimensions"
        ),
        "minimum_branch_variance": threshold,
        "minimum_fraction_per_task": MINIMUM_INFORMATIVE_BUNDLE_FRACTION,
        "informative_bundle_count": int(
            sum(row["informative_bundle_count"] for row in per_task.values())
        ),
        "bundle_count": len(unique_values),
        "fraction": float(np.mean(np.asarray(unique_values) >= threshold)),
        "per_task": per_task,
        "all_tasks_passed": all(bool(row["passed"]) for row in per_task.values()),
    }


def action_branch_gate_decision(
    comparisons: Mapping[str, Mapping[str, Any]],
    informative_bundles: Mapping[str, Any],
    *,
    stage: str,
    raw_mse_sanity: Mapping[str, Any] | None = None,
    late_comparisons: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply the preregistered screen/final A1 decision thresholds."""

    if stage not in {"smoke", "screen", "final"}:
        raise ValueError("stage must be one of smoke, screen, or final")
    if stage == "smoke":
        return {
            "schema_version": 1,
            "stage": stage,
            "scientific_decision": False,
            "evaluated": False,
            "passed": True,
            "decision": "Smoke-Pass",
            "checks": {},
            "failed_checks": [],
        }
    if raw_mse_sanity is None:
        raise ValueError("screen/final gate requires factual raw-MSE sanity")
    if late_comparisons is None:
        raise ValueError("screen/final gate requires H6-H8 comparisons")
    missing = sorted(set(PRIMARY_BASELINES).difference(comparisons))
    if missing:
        raise ValueError(f"required primary comparisons are absent: {missing}")
    late_missing = sorted(set(PRIMARY_BASELINES).difference(late_comparisons))
    if late_missing:
        raise ValueError(
            f"required H6-H8 comparisons are absent: {late_missing}"
        )
    task_names = tuple(comparisons[NO_ACTION_METHOD]["per_task"])
    seed_names = tuple(comparisons[NO_ACTION_METHOD]["per_seed"])
    if any(tuple(comparisons[name]["per_task"]) != task_names for name in PRIMARY_BASELINES):
        raise ValueError("primary comparisons do not share the same tasks")
    if any(tuple(comparisons[name]["per_seed"]) != seed_names for name in PRIMARY_BASELINES):
        raise ValueError("primary comparisons do not share the same seeds")

    task_count = len(task_names)
    seed_count = len(seed_names)
    required_tasks = math.ceil(MINIMUM_POSITIVE_TASK_FRACTION * task_count)
    required_seeds = math.ceil(MINIMUM_POSITIVE_SEED_FRACTION * seed_count)
    jointly_positive_tasks = [
        task
        for task in task_names
        if all(
            float(comparisons[name]["per_task"][task]["effect"]) > 0.0
            for name in PRIMARY_BASELINES
        )
    ]
    jointly_positive_seeds = [
        seed
        for seed in seed_names
        if all(
            float(comparisons[name]["per_seed"][seed]["effect"]) > 0.0
            for name in PRIMARY_BASELINES
        )
    ]

    checks: dict[str, dict[str, Any]] = {
        "minimum_task_count": {
            "passed": task_count >= MINIMUM_TASK_COUNT,
            "observed": task_count,
            "operator": ">=",
            "threshold": MINIMUM_TASK_COUNT,
        },
        "minimum_seed_count": {
            "passed": seed_count >= MINIMUM_SEED_COUNT,
            "observed": seed_count,
            "operator": ">=",
            "threshold": MINIMUM_SEED_COUNT,
        },
        "jointly_positive_tasks": {
            "passed": len(jointly_positive_tasks) >= required_tasks,
            "observed": len(jointly_positive_tasks),
            "operator": ">=",
            "threshold": required_tasks,
        },
        "jointly_positive_seeds": {
            "passed": len(jointly_positive_seeds) >= required_seeds,
            "observed": len(jointly_positive_seeds),
            "operator": ">=",
            "threshold": required_seeds,
        },
        "informative_bundle_fraction_per_task": {
            "passed": bool(informative_bundles.get("all_tasks_passed")),
            "observed": {
                task: row["fraction"]
                for task, row in informative_bundles["per_task"].items()
            },
            "operator": ">= for every task",
            "threshold": MINIMUM_INFORMATIVE_BUNDLE_FRACTION,
        },
        "factual_raw_mse_not_worse_than_no_action": {
            "passed": bool(raw_mse_sanity.get("passed")),
            "observed": {
                "factual_mse": raw_mse_sanity.get("factual_mse"),
                "no_action_mse": raw_mse_sanity.get("no_action_mse"),
            },
            "operator": "factual <= no_action",
            "threshold": 0.0,
        },
    }
    for baseline in PRIMARY_BASELINES:
        row = comparisons[baseline]
        effect = float(row["effect"])
        checks[f"factual_over_{baseline}_at_least_five_percent"] = {
            "passed": effect >= MINIMUM_RELATIVE_IMPROVEMENT,
            "observed": effect,
            "operator": ">=",
            "threshold": MINIMUM_RELATIVE_IMPROVEMENT,
        }
        if stage == "final":
            ci = row.get("effect_95_ci")
            if not isinstance(ci, Sequence) or len(ci) != 2:
                raise ValueError(f"final stage requires a two-sided CI for {baseline}")
            lower = float(ci[0])
            checks[f"factual_over_{baseline}_ci_lower_positive"] = {
                "passed": lower > 0.0,
                "observed": lower,
                "operator": ">",
                "threshold": 0.0,
            }
        late_effect = float(late_comparisons[baseline]["effect"])
        checks[f"h6_h8_factual_over_{baseline}_positive"] = {
            "passed": late_effect > 0.0,
            "observed": late_effect,
            "operator": ">",
            "threshold": 0.0,
        }
    passed = all(bool(check["passed"]) for check in checks.values())
    return {
        "schema_version": 1,
        "stage": stage,
        "scientific_decision": True,
        "evaluated": True,
        "thresholds": {
            "minimum_relative_improvement": MINIMUM_RELATIVE_IMPROVEMENT,
            "ci_lower_strictly_positive": stage == "final",
            "minimum_task_count": MINIMUM_TASK_COUNT,
            "minimum_positive_task_fraction": MINIMUM_POSITIVE_TASK_FRACTION,
            "required_positive_tasks": required_tasks,
            "minimum_seed_count": MINIMUM_SEED_COUNT,
            "minimum_positive_seed_fraction": MINIMUM_POSITIVE_SEED_FRACTION,
            "required_positive_seeds": required_seeds,
            "minimum_informative_bundle_fraction_per_task": (
                MINIMUM_INFORMATIVE_BUNDLE_FRACTION
            ),
        },
        "jointly_positive_tasks": jointly_positive_tasks,
        "jointly_positive_task_count": len(jointly_positive_tasks),
        "jointly_positive_seeds": jointly_positive_seeds,
        "jointly_positive_seed_count": len(jointly_positive_seeds),
        "checks": checks,
        "passed": passed,
        "decision": "Go" if passed else "No-Go",
        "failed_checks": [
            name for name, check in checks.items() if not bool(check["passed"])
        ],
    }


def _resolve_moments(
    *,
    moments: TrainMoments | None,
    train_moments: TrainMoments | None,
    train_target: np.ndarray | None,
) -> TrainMoments:
    supplied = sum(value is not None for value in (moments, train_moments, train_target))
    if supplied != 1:
        raise ValueError(
            "supply exactly one of moments, train_moments, or train_target"
        )
    selected = moments if moments is not None else train_moments
    if selected is not None:
        if not isinstance(selected, TrainMoments):
            raise TypeError("moments must be a TrainMoments instance")
        return selected
    assert train_target is not None
    return fit_train_moments(train_target)


def _prediction_loss_vectors(
    target_centered: np.ndarray,
    target_whitened: np.ndarray,
    prediction: np.ndarray,
    moments: TrainMoments,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return centered, raw, and H6-H8 centered bundle loss vectors."""

    target_shape = target_centered.shape
    if prediction.ndim == 5:
        candidates = (prediction,)
    elif prediction.ndim == 6 and prediction.shape == (
        target_shape[0],
        CYCLIC_SHIFT_COUNT,
        *target_shape[1:],
    ):
        candidates = tuple(
            prediction[:, shift] for shift in range(CYCLIC_SHIFT_COUNT)
        )
    else:
        raise ValueError(
            f"{label} must have shape {target_shape} or "
            f"{(target_shape[0], CYCLIC_SHIFT_COUNT, *target_shape[1:])}"
        )

    centered_by_shift: list[np.ndarray] = []
    raw_by_shift: list[np.ndarray] = []
    late_by_shift: list[np.ndarray] = []
    late_slice = slice(
        min(LATE_WINDOW_HORIZONS) - 1,
        max(LATE_WINDOW_HORIZONS),
    )
    for shift_index, candidate in enumerate(candidates):
        prediction_whitened = whiten_latents(
            candidate,
            moments,
            label=f"{label} shift {shift_index + 1}",
        )
        prediction_centered = prediction_whitened - np.mean(
            prediction_whitened,
            axis=1,
            keepdims=True,
            dtype=np.float64,
        )
        centered_squared = np.square(target_centered - prediction_centered)
        raw_squared = np.square(target_whitened - prediction_whitened)
        if not np.isfinite(centered_squared).all() or not np.isfinite(
            raw_squared
        ).all():
            raise ValueError(f"{label} produced non-finite loss")
        centered_by_shift.append(
            np.mean(centered_squared, axis=(1, 2, 3, 4), dtype=np.float64)
        )
        raw_by_shift.append(
            np.mean(raw_squared, axis=(1, 2, 3, 4), dtype=np.float64)
        )
        late_by_shift.append(
            np.mean(
                centered_squared[:, :, late_slice],
                axis=(1, 2, 3, 4),
                dtype=np.float64,
            )
        )
    return tuple(
        np.mean(np.stack(values, axis=1), axis=1, dtype=np.float64)
        for values in (centered_by_shift, raw_by_shift, late_by_shift)
    )


def summarize_action_branch_metrics(
    target: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    history: np.ndarray,
    task: np.ndarray | Sequence[Identity] | None = None,
    episode: np.ndarray | Sequence[Identity] | None = None,
    hazard: np.ndarray | Sequence[Identity] | None = None,
    seed: np.ndarray | Sequence[int] | None = None,
    *,
    task_index: np.ndarray | Sequence[int] | None = None,
    task_instance_index: np.ndarray | Sequence[int] | None = None,
    source_episode_id: np.ndarray | Sequence[Identity] | None = None,
    hazard_id: np.ndarray | Sequence[Identity] | None = None,
    split: np.ndarray | Sequence[str] | None = None,
    training_seed: np.ndarray | Sequence[int] | None = None,
    moments: TrainMoments | None = None,
    train_moments: TrainMoments | None = None,
    train_target: np.ndarray | None = None,
    shuffle_permutation: np.ndarray | Sequence[int] | None = None,
    stage: str = "final",
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    minimum_branch_variance: float = MINIMUM_INFORMATIVE_BRANCH_VARIANCE,
    no_action_invariance_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Score one complete A1 panel and apply the bounded stage decision."""

    if seed is not None and training_seed is not None:
        if not np.array_equal(np.asarray(seed), np.asarray(training_seed)):
            raise ValueError("seed and training_seed aliases conflict")
    seed_value = seed if seed is not None else training_seed
    if seed_value is None:
        raise ValueError("seed is required")
    if task is not None and task_index is not None:
        if not np.array_equal(np.asarray(task), np.asarray(task_index)):
            raise ValueError("task and task_index aliases conflict")
    task_value = task if task is not None else task_index
    if task_value is None:
        raise ValueError("task_index is required")
    if episode is not None and source_episode_id is not None:
        if not np.array_equal(np.asarray(episode), np.asarray(source_episode_id)):
            raise ValueError("episode and source_episode_id aliases conflict")
    episode_value = episode if episode is not None else source_episode_id
    if episode_value is None:
        raise ValueError("source_episode_id is required")
    if hazard is not None and hazard_id is not None:
        if not np.array_equal(np.asarray(hazard), np.asarray(hazard_id)):
            raise ValueError("hazard and hazard_id aliases conflict")
    hazard_value = hazard if hazard is not None else hazard_id
    if hazard_value is None:
        raise ValueError("hazard_id is required")
    if shuffle_permutation is None:
        raise ValueError("shuffle_permutation is required")
    if stage not in {"smoke", "screen", "final"}:
        raise ValueError("stage must be one of smoke, screen, or final")
    samples = _strict_integer(
        bootstrap_samples,
        label="bootstrap_samples",
        minimum=1 if stage == "final" else 0,
    )
    random_seed = _strict_integer(
        bootstrap_seed, label="bootstrap_seed", minimum=0
    )
    threshold = _finite_number(
        minimum_branch_variance,
        label="minimum_branch_variance",
    )
    if threshold < 0.0:
        raise ValueError("minimum_branch_variance must be non-negative")
    fitted = _resolve_moments(
        moments=moments,
        train_moments=train_moments,
        train_target=train_target,
    )
    panel = _validate_panel(
        target,
        predictions,
        history,
        task_value,
        episode_value,
        hazard_value,
        seed_value,
        shuffle_permutation,
        task_instance_index=task_instance_index,
        split=split,
        stage=stage,
        no_action_invariance_tolerance=no_action_invariance_tolerance,
    )
    if panel.target.shape[-2:] != fitted.mean.shape:
        raise ValueError("train moments [K,D] do not match target")
    layout = _panel_layout(panel)

    generated = build_secondary_predictions(
        panel.history,
        branch_count=panel.target.shape[1],
        horizon_count=panel.target.shape[2],
    )
    method_predictions = {**panel.predictions, **generated}
    target_centered = center_whitened_branches(
        panel.target, fitted, label="target"
    )
    target_whitened = whiten_latents(panel.target, fitted, label="target")
    centered_losses: dict[str, np.ndarray] = {}
    raw_losses: dict[str, np.ndarray] = {}
    late_centered_losses: dict[str, np.ndarray] = {}
    for name, prediction in method_predictions.items():
        centered, raw, late = _prediction_loss_vectors(
            target_centered,
            target_whitened,
            prediction,
            fitted,
            label=f"prediction {name!r}",
        )
        centered_losses[name] = centered
        raw_losses[name] = raw
        if name in REQUIRED_PREDICTION_METHODS:
            late_centered_losses[name] = late

    centered_reductions = {
        name: _reduce_bundle_losses(loss, layout=layout)
        for name, loss in centered_losses.items()
    }
    raw_reductions = {
        name: _reduce_bundle_losses(loss, layout=layout)
        for name, loss in raw_losses.items()
    }
    late_centered_reductions = {
        name: _reduce_bundle_losses(loss, layout=layout)
        for name, loss in late_centered_losses.items()
    }
    effective_samples = samples if stage == "final" else 0
    draws = (
        _bootstrap_draws(layout, samples=effective_samples, seed=random_seed)
        if effective_samples
        else None
    )
    comparisons = {
        baseline: _comparison_report(
            centered_reductions[FACTUAL_METHOD],
            centered_reductions[baseline],
            factual_method=FACTUAL_METHOD,
            baseline_method=baseline,
            layout=layout,
            draws=draws,
        )
        for baseline in PRIMARY_BASELINES
    }
    late_comparisons = {
        baseline: _comparison_report(
            late_centered_reductions[FACTUAL_METHOD],
            late_centered_reductions[baseline],
            factual_method=FACTUAL_METHOD,
            baseline_method=baseline,
            layout=layout,
            draws=draws,
        )
        for baseline in PRIMARY_BASELINES
    }
    raw_sanity = _raw_mse_sanity_report(
        raw_reductions[FACTUAL_METHOD],
        raw_reductions[NO_ACTION_METHOD],
        layout=layout,
    )
    informative = _informative_report(
        target_centered,
        layout=layout,
        threshold=threshold,
    )
    gate = action_branch_gate_decision(
        comparisons,
        informative,
        stage=stage,
        raw_mse_sanity=raw_sanity,
        late_comparisons=late_comparisons,
    )
    primary_methods = {
        name: _method_report(reduction, layout=layout)
        for name, reduction in centered_reductions.items()
    }
    secondary_methods = {
        name: _method_report(reduction, layout=layout)
        for name, reduction in raw_reductions.items()
    }
    bundle_count = len(layout.bundle_rows)
    episode_count = sum(len(values) for values in layout.episodes_by_task.values())
    summary = {
        "schema_version": 1,
        "protocol": "fcpg_action_branch_random_hazard_a1_v1",
        "stage": stage,
        "metric": "within_state_branch_centered_whitened_mse",
        "prediction_mode": "closed_loop_recursive_rollout",
        "lower_is_better": True,
        "whitening": {
            "source": "train_split_only",
            "axes": ["anchor", "latent_dimension"],
            "moment_shape": list(fitted.mean.shape),
            "fit_observation_count": fitted.count,
            "epsilon": fitted.epsilon,
            "floored_dimension_count": fitted.floored_dimensions,
            "minimum_scale": float(np.min(fitted.scale)),
            "maximum_scale": float(np.max(fitted.scale)),
        },
        "aggregation": {
            "branch_centering": (
                "target_and_each_prediction_centered_separately_within_"
                "bundle_and_horizon"
            ),
            "latent_reduction": "equal_mean",
            "branch_reduction": "equal_mean_of_all_eight_branches",
            "horizon_reduction": "equal_mean_within_hazard_bundle",
            "shuffle_reduction": (
                "bundle_mse_per_non_identity_cyclic_shift_then_equal_mean_"
                "over_all_seven_shifts"
            ),
            "seed_reduction": "equal_mean_within_source_episode",
            "hazard_reduction": "equal_mean_within_source_episode_and_seed",
            "episode_reduction": "equal_mean_within_task",
            "task_reduction": "equal_mean_of_task_effects",
        },
        "bootstrap": {
            "method": "task_stratified_paired_cluster_percentile",
            "unit": ["task", "source_physical_episode"],
            "nested_not_resampled": [
                "hazard",
                "cyclic_shift",
                "branch",
                "horizon",
                "seed",
            ],
            "shared_draws_across_methods": True,
            "samples": effective_samples,
            "requested_samples": samples,
            "seed": random_seed,
            "confidence": 0.95 if effective_samples else None,
        },
        "coverage": {
            "row_count_including_training_seeds": panel.target.shape[0],
            "hazard_bundle_count": bundle_count,
            "source_physical_episode_count": episode_count,
            "independent_sampling_unit_count": episode_count,
            "task_count": len(layout.tasks),
            "training_seed_count": len(layout.seeds),
            "training_seeds": list(layout.seeds),
            "task_names": list(layout.tasks),
            "branch_count": panel.target.shape[1],
            "horizon_count": panel.target.shape[2],
            "anchor_count": panel.target.shape[3],
            "latent_dimension": panel.target.shape[4],
            "episodes_per_task": {
                task: len(layout.episodes_by_task[task]) for task in layout.tasks
            },
            "bundles_per_task": {
                task: sum(key[0] == task for key in layout.bundle_rows)
                for task in layout.tasks
            },
            "no_action_branch_invariant": True,
            "no_action_max_branch_deviation": (
                panel.no_action_max_branch_deviation
            ),
            "shuffle_is_all_non_identity_cyclic_shifts": True,
            "shuffle_shift_count": panel.permutations.shape[1],
            "task_instance_count": (
                len(set(zip(panel.tasks, panel.task_instances, strict=True)))
                if panel.task_instances is not None
                else None
            ),
            "split": panel.splits[0] if panel.splits is not None else None,
        },
        "primary": {
            "metric": "within_state_branch_centered_whitened_mse",
            "methods": primary_methods,
            "method_task_macro_mse": {
                name: row["task_macro_mse"] for name, row in primary_methods.items()
            },
            "comparisons": comparisons,
        },
        "comparisons": comparisons,
        "late_window": {
            "gating": True,
            "metric": "within_state_branch_centered_whitened_mse",
            "horizons_1_based": list(LATE_WINDOW_HORIZONS),
            "comparisons": late_comparisons,
        },
        "secondary": {
            "gating": False,
            "metric": "raw_whitened_trajectory_mse",
            "generated_methods": [
                HOLD_LAST_METHOD,
                LATENT_CONSTANT_VELOCITY_METHOD,
            ],
            "methods": secondary_methods,
            "method_task_macro_mse": {
                name: row["task_macro_mse"]
                for name, row in secondary_methods.items()
            },
        },
        "sanity": {
            "gating": stage != "smoke",
            "factual_raw_mse_not_worse_than_no_action": raw_sanity,
        },
        "informative_bundles": informative,
        "gate": gate,
        "decision": gate["decision"],
        "passed": gate["passed"],
    }
    return summary


def score_action_branch_metrics(
    target: np.ndarray,
    factual_prediction: np.ndarray,
    no_action_prediction: np.ndarray,
    shuffled_action_prediction: np.ndarray,
    history: np.ndarray,
    task: np.ndarray | Sequence[Identity],
    episode: np.ndarray | Sequence[Identity],
    hazard: np.ndarray | Sequence[Identity],
    seed: np.ndarray | Sequence[int],
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience wrapper accepting the three required prediction arrays."""

    return summarize_action_branch_metrics(
        target,
        {
            FACTUAL_METHOD: factual_prediction,
            NO_ACTION_METHOD: no_action_prediction,
            SHUFFLED_ACTION_METHOD: shuffled_action_prediction,
        },
        history,
        task,
        episode,
        hazard,
        seed,
        **kwargs,
    )


summarize_action_branch_a1 = summarize_action_branch_metrics


__all__ = [
    "BRANCH_COUNT",
    "CYCLIC_SHIFT_COUNT",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "FACTUAL_METHOD",
    "HOLD_LAST_METHOD",
    "LATENT_CONSTANT_VELOCITY_METHOD",
    "LATE_WINDOW_HORIZONS",
    "MINIMUM_INFORMATIVE_BRANCH_VARIANCE",
    "MINIMUM_INFORMATIVE_BUNDLE_FRACTION",
    "MINIMUM_RELATIVE_IMPROVEMENT",
    "NO_ACTION_METHOD",
    "PRIMARY_BASELINES",
    "REQUIRED_PREDICTION_METHODS",
    "SHUFFLED_ACTION_METHOD",
    "TASK_NAMES",
    "TrainMoments",
    "action_branch_gate_decision",
    "branch_centered_whitened_mse",
    "build_secondary_predictions",
    "center_whitened_branches",
    "centered_whitened_bundle_mse",
    "fit_train_moments",
    "fit_train_whitening_moments",
    "make_all_cyclic_branch_shifts",
    "make_within_bundle_derangement",
    "raw_whitened_bundle_mse",
    "score_action_branch_metrics",
    "summarize_action_branch_a1",
    "summarize_action_branch_metrics",
    "train_moments_from_arrays",
    "validate_all_cyclic_branch_shifts",
    "validate_within_bundle_derangement",
    "whiten_latents",
    "within_bundle_derangement",
]
