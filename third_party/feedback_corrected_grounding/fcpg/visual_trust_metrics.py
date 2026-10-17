"""Pure NumPy metrics for evaluating per-sample visual trust."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


_CLASSIFICATION_METRICS = (
    "auroc",
    "unsafe_recall",
    "clean_false_reject_rate",
    "early_reopen_rate",
)


def _real_array(
    value: ArrayLike,
    *,
    name: str,
    ndim: int,
    allow_empty: bool,
) -> NDArray[np.float64]:
    array = np.asarray(value)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got shape {array.shape}")
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numbers")
    result = array.astype(np.float64, copy=False)
    if not allow_empty and result.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not bool(np.isfinite(result).all()):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _trust_array(
    value: ArrayLike,
    *,
    ndim: int,
    allow_empty: bool,
) -> NDArray[np.float64]:
    trust = _real_array(value, name="trust", ndim=ndim, allow_empty=allow_empty)
    if bool(np.any((trust < 0.0) | (trust > 1.0))):
        raise ValueError("trust values must lie in [0, 1]")
    return trust


def _binary_vector(value: ArrayLike, *, name: str) -> NDArray[np.bool_]:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape {array.shape}")
    if array.dtype.kind == "b":
        return array.astype(np.bool_, copy=False)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain binary values")
    numeric = array.astype(np.float64, copy=False)
    if not bool(np.isfinite(numeric).all()):
        raise ValueError(f"{name} must contain only finite values")
    if not bool(np.all((numeric == 0.0) | (numeric == 1.0))):
        raise ValueError(f"{name} must contain only 0, 1, or boolean values")
    return numeric.astype(np.bool_)


def _threshold(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("threshold must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("threshold must be finite")
    if not 0.0 <= result <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _check_aligned(reference: np.ndarray, other: np.ndarray, *, name: str) -> None:
    if reference.shape != other.shape:
        raise ValueError(
            f"trust and {name} must have identical shapes, got "
            f"{reference.shape} and {other.shape}"
        )


def _binary_auroc(labels: NDArray[np.bool_], scores: NDArray[np.float64]) -> float:
    positive_count = int(np.count_nonzero(labels))
    negative_count = int(labels.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    _, inverse, counts = np.unique(
        scores, return_inverse=True, return_counts=True
    )
    rank_high = np.cumsum(counts, dtype=np.float64)
    rank_low = rank_high - counts + 1.0
    average_ranks = (rank_low + rank_high) / 2.0
    positive_rank_sum = float(np.sum(average_ranks[inverse][labels]))
    auc = (
        positive_rank_sum - positive_count * (positive_count + 1.0) / 2.0
    ) / (positive_count * negative_count)
    return float(np.clip(auc, 0.0, 1.0))


def binary_auroc(labels: ArrayLike, scores: ArrayLike) -> float:
    """Return binary AUROC with average ranks for tied scores.

    ``labels == 1`` denotes the positive class and larger scores favor that
    class. The result is ``NaN`` when either class is absent.
    """

    binary_labels = _binary_vector(labels, name="labels")
    numeric_scores = _real_array(
        scores, name="scores", ndim=1, allow_empty=True
    )
    if binary_labels.shape != numeric_scores.shape:
        raise ValueError(
            "labels and scores must have identical shapes, got "
            f"{binary_labels.shape} and {numeric_scores.shape}"
        )
    return _binary_auroc(binary_labels, numeric_scores)


def _masked_rate(mask: NDArray[np.bool_], event: NDArray[np.bool_]) -> float:
    denominator = int(np.count_nonzero(mask))
    if denominator == 0:
        return float("nan")
    return float(np.count_nonzero(mask & event) / denominator)


def unsafe_recall(
    trust: ArrayLike,
    unsafe_mask: ArrayLike,
    threshold: float = 0.5,
) -> float:
    """Fraction of unsafe samples rejected by ``trust < threshold``."""

    values = _trust_array(trust, ndim=1, allow_empty=True)
    mask = _binary_vector(unsafe_mask, name="unsafe_mask")
    _check_aligned(values, mask, name="unsafe_mask")
    cutoff = _threshold(threshold)
    return _masked_rate(mask, values < cutoff)


def clean_false_reject_rate(
    trust: ArrayLike,
    clean_mask: ArrayLike,
    threshold: float = 0.5,
) -> float:
    """Fraction of clean samples incorrectly rejected."""

    values = _trust_array(trust, ndim=1, allow_empty=True)
    mask = _binary_vector(clean_mask, name="clean_mask")
    _check_aligned(values, mask, name="clean_mask")
    cutoff = _threshold(threshold)
    return _masked_rate(mask, values < cutoff)


def early_reopen_rate(
    trust: ArrayLike,
    early_reopen_mask: ArrayLike,
    threshold: float = 0.5,
) -> float:
    """Fraction of pre-recovery opportunities accepted too early.

    The caller supplies the opportunities through ``early_reopen_mask``;
    acceptance follows the deployment rule ``trust >= threshold``.
    """

    values = _trust_array(trust, ndim=1, allow_empty=True)
    mask = _binary_vector(early_reopen_mask, name="early_reopen_mask")
    _check_aligned(values, mask, name="early_reopen_mask")
    cutoff = _threshold(threshold)
    return _masked_rate(mask, values >= cutoff)


def _classification_values(
    trust: NDArray[np.float64],
    unsafe: NDArray[np.bool_],
    clean: NDArray[np.bool_],
    early: NDArray[np.bool_],
    *,
    threshold: float,
) -> dict[str, float | int]:
    classified = unsafe | clean
    return {
        "auroc": _binary_auroc(clean[classified], trust[classified]),
        "unsafe_recall": _masked_rate(unsafe, trust < threshold),
        "clean_false_reject_rate": _masked_rate(clean, trust < threshold),
        "early_reopen_rate": _masked_rate(early, trust >= threshold),
        "sample_count": int(trust.size),
        "unsafe_count": int(np.count_nonzero(unsafe)),
        "clean_count": int(np.count_nonzero(clean)),
        "early_reopen_count": int(np.count_nonzero(early)),
    }


def _identifier_vector(
    value: ArrayLike,
    *,
    name: str,
    length: int,
) -> tuple[str | bytes | int | float, ...]:
    array = np.asarray(value)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")

    identifiers: list[str | bytes | int | float] = []
    for item in array.tolist():
        if isinstance(item, (bool, np.bool_)):
            raise TypeError(f"{name} must contain string or numeric identifiers")
        if isinstance(item, (str, bytes, Integral)):
            identifiers.append(item)
            continue
        if isinstance(item, Real):
            numeric = float(item)
            if not np.isfinite(numeric):
                raise ValueError(f"{name} must contain only finite identifiers")
            identifiers.append(numeric)
            continue
        raise TypeError(f"{name} must contain string or numeric identifiers")
    return tuple(identifiers)


def _group_positions(
    identifiers: tuple[str | bytes | int | float, ...],
) -> dict[str | bytes | int | float, NDArray[np.int64]]:
    positions: dict[str | bytes | int | float, list[int]] = {}
    for index, identifier in enumerate(identifiers):
        positions.setdefault(identifier, []).append(index)
    return {
        identifier: np.asarray(indices, dtype=np.int64)
        for identifier, indices in positions.items()
    }


def _classification_groups(
    identifiers: tuple[str | bytes | int | float, ...],
    trust: NDArray[np.float64],
    unsafe: NDArray[np.bool_],
    clean: NDArray[np.bool_],
    early: NDArray[np.bool_],
    *,
    threshold: float,
) -> dict[str | bytes | int | float, dict[str, float | int]]:
    return {
        identifier: _classification_values(
            trust[indices],
            unsafe[indices],
            clean[indices],
            early[indices],
            threshold=threshold,
        )
        for identifier, indices in _group_positions(identifiers).items()
    }


def _classification_macro(
    groups: dict[Any, dict[str, float | int]],
) -> dict[str, object]:
    result: dict[str, object] = {"group_count": len(groups)}
    defined: dict[str, int] = {}
    for metric in _CLASSIFICATION_METRICS:
        values = np.asarray(
            [float(group[metric]) for group in groups.values()], dtype=np.float64
        )
        usable = np.isfinite(values)
        defined[metric] = int(np.count_nonzero(usable))
        result[metric] = (
            float(np.mean(values[usable], dtype=np.float64))
            if bool(np.any(usable))
            else float("nan")
        )
    result["defined_group_count"] = defined
    return result


def summarize_visual_trust(
    trust: ArrayLike,
    unsafe_mask: ArrayLike,
    clean_mask: ArrayLike,
    early_reopen_mask: ArrayLike,
    task_ids: ArrayLike,
    family_ids: ArrayLike,
    threshold: float = 0.5,
) -> dict[str, object]:
    """Return overall, equal-task, and equal-family classification metrics.

    AUROC treats clean samples as the positive class and ignores samples that
    are in neither the clean nor unsafe mask. Undefined group metrics are
    omitted from that metric's macro mean; ``defined_group_count`` makes this
    explicit. Clean and unsafe masks must not overlap.
    """

    values = _trust_array(trust, ndim=1, allow_empty=False)
    unsafe = _binary_vector(unsafe_mask, name="unsafe_mask")
    clean = _binary_vector(clean_mask, name="clean_mask")
    early = _binary_vector(early_reopen_mask, name="early_reopen_mask")
    for name, mask in (
        ("unsafe_mask", unsafe),
        ("clean_mask", clean),
        ("early_reopen_mask", early),
    ):
        _check_aligned(values, mask, name=name)
    if bool(np.any(unsafe & clean)):
        raise ValueError("unsafe_mask and clean_mask must not overlap")

    cutoff = _threshold(threshold)
    tasks = _identifier_vector(task_ids, name="task_ids", length=values.size)
    families = _identifier_vector(
        family_ids, name="family_ids", length=values.size
    )
    per_task = _classification_groups(
        tasks, values, unsafe, clean, early, threshold=cutoff
    )
    per_family = _classification_groups(
        families, values, unsafe, clean, early, threshold=cutoff
    )
    return {
        "threshold": cutoff,
        "overall": _classification_values(
            values, unsafe, clean, early, threshold=cutoff
        ),
        "per_task": per_task,
        "task_macro": _classification_macro(per_task),
        "per_family": per_family,
        "family_macro": _classification_macro(per_family),
    }


def _recovery_starts(
    value: int | ArrayLike,
    *,
    episode_count: int,
    time_count: int,
) -> NDArray[np.int64]:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("recovery_start must contain integers")
    if isinstance(value, Integral):
        starts = np.full(episode_count, int(value), dtype=np.int64)
    else:
        array = np.asarray(value)
        if array.shape != (episode_count,):
            raise ValueError(
                "recovery_start must be a scalar or have shape "
                f"({episode_count},), got {array.shape}"
            )
        if array.dtype.kind in "f" and not bool(np.isfinite(array).all()):
            raise ValueError("recovery_start must contain only finite values")
        if array.dtype.kind not in "iu":
            raise TypeError("recovery_start must contain integers")
        starts = array.astype(np.int64, copy=False)
    if bool(np.any((starts < 0) | (starts > time_count))):
        raise ValueError(
            f"recovery_start values must lie in [0, {time_count}]"
        )
    return starts


def _first_run_delay(
    trust: NDArray[np.float64],
    *,
    recovery_start: int,
    threshold: float,
    min_consecutive: int,
) -> float:
    accepted = trust[recovery_start:] >= threshold
    if accepted.size < min_consecutive:
        return float("nan")
    run_counts = np.convolve(
        accepted.astype(np.int64),
        np.ones(min_consecutive, dtype=np.int64),
        mode="valid",
    )
    matches = np.flatnonzero(run_counts == min_consecutive)
    # A causal observer can announce recovery only after seeing the final
    # sample in the qualifying run, not at the run's retrospective start.
    return (
        float(matches[0] + min_consecutive - 1)
        if matches.size
        else float("nan")
    )


def recovery_detection_delay(
    trust: ArrayLike,
    threshold: float = 0.5,
    min_consecutive: int = 1,
    recovery_start: int = 0,
) -> float:
    """Return the causal detection time relative to recovery onset.

    A qualifying run contains ``min_consecutive`` values satisfying
    ``trust >= threshold``. The delay points to the final sample needed to
    confirm that run. ``NaN`` means no such run was observed.
    """

    values = _trust_array(trust, ndim=1, allow_empty=False)
    cutoff = _threshold(threshold)
    run_length = _positive_integer(min_consecutive, name="min_consecutive")
    start = int(
        _recovery_starts(
            recovery_start, episode_count=1, time_count=values.size
        )[0]
    )
    return _first_run_delay(
        values,
        recovery_start=start,
        threshold=cutoff,
        min_consecutive=run_length,
    )


def recovery_detection_delays(
    trust: ArrayLike,
    threshold: float = 0.5,
    min_consecutive: int = 1,
    recovery_start: int | ArrayLike = 0,
) -> NDArray[np.float64]:
    """Return one recovery detection delay per row of ``trust[E, T]``."""

    values = _trust_array(trust, ndim=2, allow_empty=False)
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("trust must have positive episode and time dimensions")
    cutoff = _threshold(threshold)
    run_length = _positive_integer(min_consecutive, name="min_consecutive")
    starts = _recovery_starts(
        recovery_start,
        episode_count=values.shape[0],
        time_count=values.shape[1],
    )
    return np.asarray(
        [
            _first_run_delay(
                sequence,
                recovery_start=int(start),
                threshold=cutoff,
                min_consecutive=run_length,
            )
            for sequence, start in zip(values, starts, strict=True)
        ],
        dtype=np.float64,
    )


def _recovery_groups(
    delays: NDArray[np.float64],
    identifiers: tuple[str | bytes | int | float, ...],
) -> dict[str | bytes | int | float, dict[str, float | int]]:
    result: dict[str | bytes | int | float, dict[str, float | int]] = {}
    for identifier, indices in _group_positions(identifiers).items():
        group_delays = delays[indices]
        detected = np.isfinite(group_delays)
        detected_count = int(np.count_nonzero(detected))
        result[identifier] = {
            "mean_delay": (
                float(np.mean(group_delays[detected], dtype=np.float64))
                if detected_count
                else float("nan")
            ),
            "detection_rate": float(detected_count / group_delays.size),
            "episode_count": int(group_delays.size),
            "detected_count": detected_count,
        }
    return result


def _recovery_macro(groups: dict[Any, dict[str, float | int]]) -> dict[str, object]:
    mean_delays = np.asarray(
        [float(group["mean_delay"]) for group in groups.values()], dtype=np.float64
    )
    defined = np.isfinite(mean_delays)
    detection_rates = np.asarray(
        [float(group["detection_rate"]) for group in groups.values()],
        dtype=np.float64,
    )
    return {
        "mean_delay": (
            float(np.mean(mean_delays[defined], dtype=np.float64))
            if bool(np.any(defined))
            else float("nan")
        ),
        "detection_rate": float(np.mean(detection_rates, dtype=np.float64)),
        "group_count": len(groups),
        "defined_delay_group_count": int(np.count_nonzero(defined)),
    }


def summarize_recovery_detection(
    trust: ArrayLike,
    episode_ids: ArrayLike,
    family_ids: ArrayLike,
    task_ids: ArrayLike,
    threshold: float = 0.5,
    min_consecutive: int = 1,
    recovery_start: int | ArrayLike = 0,
) -> dict[str, object]:
    """Summarize recovery delay per episode, family, and task.

    Group mean delays exclude never-detected episodes, whose delays are
    ``NaN``. Detection rates include every episode. Family and task macros
    weight their groups equally rather than pooling episodes.
    """

    values = _trust_array(trust, ndim=2, allow_empty=False)
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("trust must have positive episode and time dimensions")
    cutoff = _threshold(threshold)
    run_length = _positive_integer(min_consecutive, name="min_consecutive")
    starts = _recovery_starts(
        recovery_start,
        episode_count=values.shape[0],
        time_count=values.shape[1],
    )
    episodes = _identifier_vector(
        episode_ids, name="episode_ids", length=values.shape[0]
    )
    if len(set(episodes)) != len(episodes):
        raise ValueError("episode_ids must be unique")
    families = _identifier_vector(
        family_ids, name="family_ids", length=values.shape[0]
    )
    tasks = _identifier_vector(task_ids, name="task_ids", length=values.shape[0])
    delays = np.asarray(
        [
            _first_run_delay(
                sequence,
                recovery_start=int(start),
                threshold=cutoff,
                min_consecutive=run_length,
            )
            for sequence, start in zip(values, starts, strict=True)
        ],
        dtype=np.float64,
    )

    per_episode = {
        episode: {
            "family": family,
            "task": task,
            "recovery_start": int(start),
            "delay": float(delay),
            "detected": bool(np.isfinite(delay)),
        }
        for episode, family, task, start, delay in zip(
            episodes, families, tasks, starts, delays, strict=True
        )
    }
    per_family = _recovery_groups(delays, families)
    per_task = _recovery_groups(delays, tasks)
    return {
        "threshold": cutoff,
        "min_consecutive": run_length,
        "per_episode": per_episode,
        "per_family": per_family,
        "family_macro": _recovery_macro(per_family),
        "per_task": per_task,
        "task_macro": _recovery_macro(per_task),
    }


auroc = binary_auroc


__all__ = [
    "auroc",
    "binary_auroc",
    "clean_false_reject_rate",
    "early_reopen_rate",
    "recovery_detection_delay",
    "recovery_detection_delays",
    "summarize_recovery_detection",
    "summarize_visual_trust",
    "unsafe_recall",
]
