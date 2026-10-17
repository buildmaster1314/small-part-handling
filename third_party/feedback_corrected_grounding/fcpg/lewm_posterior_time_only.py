"""Task-agnostic time-only switching baseline for the posterior Gate."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np


DEFAULT_N_TIME_BINS = 10
DEFAULT_MIN_COUNT = 5
DEFAULT_DEAD_ZONE = 0.0
DEFAULT_EPS = 1e-12
N_FRAME_PHASES = 5


def _integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _numeric_rows(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be a real numeric array")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must be a real numeric array")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _integer_rows(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must be an integer array")
    return np.asarray(array, dtype=np.int64)


def _group_rows(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if array.dtype.kind not in "USiu" or array.dtype == np.bool_:
        raise TypeError(f"{name} must contain string or integer group identities")
    result = array.astype(str)
    if np.any(result == ""):
        raise ValueError(f"{name} must contain non-empty group identities")
    return result


def _stable_group_mean(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("group values must be a nonempty finite vector")
    scale = float(np.max(np.abs(array)))
    if scale == 0.0:
        return 0.0
    return scale * float(np.mean(np.sort(array / scale), dtype=np.float64))


def _time_bins(normalized_time: np.ndarray, n_time_bins: int) -> np.ndarray:
    # Clipping only handles the closed endpoint t=1, already range-validated.
    return np.minimum(
        np.floor(normalized_time * n_time_bins).astype(np.int64),
        n_time_bins - 1,
    )


@dataclass(frozen=True, eq=False)
class TimeOnlyPolicy:
    """Immutable table indexed by time bin, frame phase, and anchor."""

    n_time_bins: int
    n_anchors: int
    min_count: int
    dead_zone: float
    eps: float
    mean_advantage: np.ndarray
    count: np.ndarray
    choose_prior: np.ndarray

    def __post_init__(self) -> None:
        n_time_bins = _integer(self.n_time_bins, name="n_time_bins", minimum=1)
        n_anchors = _integer(self.n_anchors, name="n_anchors", minimum=1)
        min_count = _integer(self.min_count, name="min_count", minimum=1)
        dead_zone = _finite_real(self.dead_zone, name="dead_zone")
        if not 0.0 <= dead_zone <= 1.0:
            raise ValueError("dead_zone must lie in [0, 1]")
        eps = _finite_real(self.eps, name="eps")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        shape = (n_time_bins, N_FRAME_PHASES, n_anchors)
        mean_advantage = np.asarray(self.mean_advantage)
        count = np.asarray(self.count)
        choose_prior = np.asarray(self.choose_prior)
        if mean_advantage.shape != shape:
            raise ValueError(f"mean_advantage must have shape {shape}")
        if count.shape != shape:
            raise ValueError(f"count must have shape {shape}")
        if choose_prior.shape != shape:
            raise ValueError(f"choose_prior must have shape {shape}")
        if not np.issubdtype(mean_advantage.dtype, np.floating):
            raise TypeError("mean_advantage must have a floating dtype")
        if not np.isfinite(mean_advantage).all():
            raise ValueError("mean_advantage must be finite")
        if not np.issubdtype(count.dtype, np.integer) or count.dtype == np.bool_:
            raise TypeError("count must have an integer dtype")
        if np.any(count < 0):
            raise ValueError("count must be non-negative")
        if choose_prior.dtype != np.bool_:
            raise TypeError("choose_prior must have boolean dtype")

        means_copy = np.array(mean_advantage, dtype=np.float64, copy=True)
        count_copy = np.array(count, dtype=np.int64, copy=True)
        choice_copy = np.array(choose_prior, dtype=np.bool_, copy=True)
        expected_choice = (count_copy >= min_count) & (means_copy > dead_zone)
        if not np.array_equal(choice_copy, expected_choice):
            raise ValueError(
                "choose_prior must match count >= min_count and "
                "mean_advantage > dead_zone"
            )
        means_copy.setflags(write=False)
        count_copy.setflags(write=False)
        choice_copy.setflags(write=False)

        object.__setattr__(self, "n_time_bins", n_time_bins)
        object.__setattr__(self, "n_anchors", n_anchors)
        object.__setattr__(self, "min_count", min_count)
        object.__setattr__(self, "dead_zone", dead_zone)
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "mean_advantage", means_copy)
        object.__setattr__(self, "count", count_copy)
        object.__setattr__(self, "choose_prior", choice_copy)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimeOnlyPolicy):
            return NotImplemented
        return (
            self.n_time_bins == other.n_time_bins
            and self.n_anchors == other.n_anchors
            and self.min_count == other.min_count
            and self.dead_zone == other.dead_zone
            and self.eps == other.eps
            and np.array_equal(self.mean_advantage, other.mean_advantage)
            and np.array_equal(self.count, other.count)
            and np.array_equal(self.choose_prior, other.choose_prior)
        )

    __hash__ = None


def fit_time_only_policy(
    normalized_time: object,
    frame_phase: object,
    anchor_index: object,
    measurement_error: object,
    prior_error: object,
    *,
    sample_weight: object | None = None,
    sample_group: object | None = None,
    sample_subgroup: object | None = None,
    n_anchors: object | None = None,
    n_time_bins: object = DEFAULT_N_TIME_BINS,
    min_count: object = DEFAULT_MIN_COUNT,
    dead_zone: object = DEFAULT_DEAD_ZONE,
    eps: object = DEFAULT_EPS,
) -> TimeOnlyPolicy:
    """Fit independent switching decisions for fixed time/phase/anchor cells.

    ``sample_weight`` affects cell means only; counts remain integer row counts.
    ``sample_group`` instead averages rows within each group first and makes the
    cell count equal the number of independent groups. When ``sample_subgroup``
    is provided, rows are first averaged within each subgroup, subgroup means
    are then averaged within each group, and groups remain the counted
    independent units. This gives paired views equal weight before their parent
    trajectory is averaged. ``sample_weight`` is mutually exclusive with both
    grouping options.
    """

    times = _numeric_rows(normalized_time, name="normalized_time")
    phases = _integer_rows(frame_phase, name="frame_phase")
    anchors = _integer_rows(anchor_index, name="anchor_index")
    measurement = _numeric_rows(measurement_error, name="measurement_error")
    prior = _numeric_rows(prior_error, name="prior_error")
    weights = (
        None
        if sample_weight is None
        else _numeric_rows(sample_weight, name="sample_weight")
    )
    groups = (
        None if sample_group is None else _group_rows(sample_group, name="sample_group")
    )
    subgroups = (
        None
        if sample_subgroup is None
        else _group_rows(sample_subgroup, name="sample_subgroup")
    )
    arrays = (times, phases, anchors, measurement, prior)
    if times.size == 0:
        raise ValueError("calibration arrays must not be empty")
    if any(array.shape != times.shape for array in arrays[1:]):
        raise ValueError("calibration arrays must have identical shapes")
    if weights is not None and weights.shape != times.shape:
        raise ValueError(
            "sample_weight must have an identical shape to calibration arrays"
        )
    if groups is not None and groups.shape != times.shape:
        raise ValueError("sample_group must have an identical shape to calibration arrays")
    if subgroups is not None and subgroups.shape != times.shape:
        raise ValueError(
            "sample_subgroup must have an identical shape to calibration arrays"
        )
    if subgroups is not None and groups is None:
        raise ValueError("sample_subgroup requires sample_group")
    if weights is not None and (groups is not None or subgroups is not None):
        raise ValueError(
            "sample_weight and sample_group/sample_subgroup are mutually exclusive"
        )
    if subgroups is not None:
        subgroup_parent: dict[str, str] = {}
        for subgroup, group in zip(subgroups.tolist(), groups.tolist(), strict=True):
            previous = subgroup_parent.setdefault(subgroup, group)
            if previous != group:
                raise ValueError("each sample_subgroup must belong to exactly one sample_group")
    if weights is not None and np.any(weights <= 0.0):
        raise ValueError("sample_weight must be positive")
    if np.any((times < 0.0) | (times > 1.0)):
        raise ValueError("normalized_time must lie in [0, 1]")
    if np.any((phases < 0) | (phases >= N_FRAME_PHASES)):
        raise ValueError("frame_phase must lie in [0, 4]")
    if np.any(anchors < 0):
        raise ValueError("anchor_index must be non-negative")
    if np.any(measurement < 0.0):
        raise ValueError("measurement_error must be non-negative")
    if np.any(prior < 0.0):
        raise ValueError("prior_error must be non-negative")

    bins = _integer(n_time_bins, name="n_time_bins", minimum=1)
    required_count = _integer(min_count, name="min_count", minimum=1)
    threshold = _finite_real(dead_zone, name="dead_zone")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("dead_zone must lie in [0, 1]")
    epsilon = _finite_real(eps, name="eps")
    if epsilon <= 0.0:
        raise ValueError("eps must be positive")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        advantage = (measurement - prior) / (measurement + prior + epsilon)
    if not np.isfinite(advantage).all():
        raise ValueError("normalized advantages must remain finite")

    observed_anchors = int(anchors.max()) + 1
    anchor_capacity = (
        observed_anchors
        if n_anchors is None
        else _integer(n_anchors, name="n_anchors", minimum=observed_anchors)
    )
    shape = (bins, N_FRAME_PHASES, anchor_capacity)
    counts = np.zeros(shape, dtype=np.int64)
    means = np.zeros(shape, dtype=np.float64)
    time_bin = _time_bins(times, bins)

    # Sorting fixes the floating-point reduction order within every cell.
    for bin_index in range(bins):
        for phase_index in range(N_FRAME_PHASES):
            for anchor in range(anchor_capacity):
                selected = (
                    (time_bin == bin_index)
                    & (phases == phase_index)
                    & (anchors == anchor)
                )
                cell_values = advantage[selected]
                cell_groups = None if groups is None else groups[selected]
                cell_subgroups = None if subgroups is None else subgroups[selected]
                if cell_subgroups is not None:
                    subgroup_values: list[float] = []
                    subgroup_groups: list[str] = []
                    for subgroup in sorted(set(cell_subgroups.tolist())):
                        subgroup_rows = cell_subgroups == subgroup
                        subgroup_values.append(
                            _stable_group_mean(cell_values[subgroup_rows])
                        )
                        subgroup_parents = set(cell_groups[subgroup_rows].tolist())
                        if len(subgroup_parents) != 1:
                            raise AssertionError(
                                "validated subgroup maps to multiple parent groups"
                            )
                        subgroup_groups.append(next(iter(subgroup_parents)))
                    subgroup_value_array = np.asarray(
                        subgroup_values, dtype=np.float64
                    )
                    subgroup_group_array = np.asarray(subgroup_groups)
                    grouped_values = [
                        _stable_group_mean(
                            subgroup_value_array[subgroup_group_array == group]
                        )
                        for group in sorted(set(subgroup_groups))
                    ]
                    cell_values = np.asarray(grouped_values, dtype=np.float64)
                elif cell_groups is not None:
                    grouped_values = [
                        _stable_group_mean(cell_values[cell_groups == group])
                        for group in sorted(set(cell_groups.tolist()))
                    ]
                    cell_values = np.asarray(grouped_values, dtype=np.float64)
                counts[bin_index, phase_index, anchor] = cell_values.size
                if cell_values.size:
                    if weights is None:
                        cell_values = np.sort(cell_values)
                        means[bin_index, phase_index, anchor] = np.mean(
                            cell_values, dtype=np.float64
                        )
                    else:
                        cell_weights = weights[selected]
                        order = np.lexsort((cell_weights, cell_values))
                        cell_values = cell_values[order]
                        cell_weights = cell_weights[order]
                        scaled_weights = cell_weights / np.max(cell_weights)
                        weighted_sum = np.sum(
                            cell_values * scaled_weights, dtype=np.float64
                        )
                        weight_sum = np.sum(scaled_weights, dtype=np.float64)
                        means[bin_index, phase_index, anchor] = (
                            weighted_sum / weight_sum
                        )

    choose_prior = (counts >= required_count) & (means > threshold)
    return TimeOnlyPolicy(
        n_time_bins=bins,
        n_anchors=anchor_capacity,
        min_count=required_count,
        dead_zone=threshold,
        eps=epsilon,
        mean_advantage=means,
        count=counts,
        choose_prior=choose_prior,
    )


def predict_beta(
    policy: TimeOnlyPolicy,
    normalized_time: object,
    frame_phase: object,
    anchor_index: object,
) -> np.ndarray:
    """Return float32 binary prior gains using deployment-safe inputs only."""

    if not isinstance(policy, TimeOnlyPolicy):
        raise TypeError("policy must be a TimeOnlyPolicy")
    times = _numeric_rows(normalized_time, name="normalized_time")
    phases = _integer_rows(frame_phase, name="frame_phase")
    anchors = _integer_rows(anchor_index, name="anchor_index")
    if phases.shape != times.shape or anchors.shape != times.shape:
        raise ValueError("prediction arrays must have identical shapes")
    if np.any((times < 0.0) | (times > 1.0)):
        raise ValueError("normalized_time must lie in [0, 1]")
    if np.any((phases < 0) | (phases >= N_FRAME_PHASES)):
        raise ValueError("frame_phase must lie in [0, 4]")
    if np.any((anchors < 0) | (anchors >= policy.n_anchors)):
        raise ValueError(
            f"anchor_index must lie in [0, {policy.n_anchors - 1}]"
        )

    bins = _time_bins(times, policy.n_time_bins)
    return policy.choose_prior[bins, phases, anchors].astype(np.float32, copy=True)


__all__ = [
    "DEFAULT_DEAD_ZONE",
    "DEFAULT_EPS",
    "DEFAULT_MIN_COUNT",
    "DEFAULT_N_TIME_BINS",
    "N_FRAME_PHASES",
    "TimeOnlyPolicy",
    "fit_time_only_policy",
    "predict_beta",
]
