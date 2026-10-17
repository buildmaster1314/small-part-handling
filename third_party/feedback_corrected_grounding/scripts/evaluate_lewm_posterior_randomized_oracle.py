#!/usr/bin/env python3
"""Evaluate the fresh randomized recursive posterior oracle Gate."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from numbers import Integral
from pathlib import Path
import sys
from typing import Any, TypeAlias

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.lewm_anchor_persistent import AnchorPersistentLeWM  # noqa: E402
from fcpg.lewm_posterior_gate_runtime import (  # noqa: E402
    MEASUREMENT_METHOD,
    infer_direct_anchors,
)
from fcpg.lewm_posterior_randomized_oracle import (  # noqa: E402
    FSM_ORACLE_METHOD,
    RANDOMIZED_ORACLE_METHODS,
    TIME_ONLY_METHOD,
    build_randomized_oracle_records,
    fit_time_only_from_teacher_forced,
    normalized_episode_time,
    run_randomized_recursive_oracle,
)
from fcpg.lewm_posterior_randomized_sequence import (  # noqa: E402
    RANDOMIZED_FRAME_SKIP,
    RANDOMIZED_HISTORY_SIZE,
    PosteriorRandomizedSequenceExamples,
    build_fresh_posterior_randomized_sequence,
)
from fcpg.lewm_posterior_randomized_statistics import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    randomized_oracle_decision,
    summarize_randomized_oracle,
)
from fcpg.lewm_runtime import (  # noqa: E402
    choose_device,
    load_lewm_grounded_checkpoint,
)


DEFAULT_CALIBRATION_TRAJECTORY_MODE = "step"
DEFAULT_EVALUATION_TRAJECTORY_MODE = "linear_drift"
DEFAULT_BATCH_SIZE = 64
DEFAULT_TIME_BINS = 10
DEFAULT_MIN_COUNT = 5
EXPECTED_CHECKPOINT_STEP = 2_000
EVALUATION_METHODS = RANDOMIZED_ORACLE_METHODS

PhysicalIdentity: TypeAlias = tuple[str, int, int, int]
EpisodeKey: TypeAlias = tuple[str, str]

_FRAME_KEYS = (
    "rgb",
    "rgb_camera_only",
    "obs",
    "action",
    "task_name",
    "task_index",
    "episode_id",
    "source_episode_id",
    "env_seed",
    "frame_id",
    "keypoints_xyz",
    "q_nominal",
    "valid_nominal",
    "anchor_dropout_active",
    "anchor_recovery_segment",
    "randomized_source_trajectory_mode",
)
_SCALAR_OR_FRAME_KEYS = ("task_generation_seed",)
_SCALAR_KEYS = ("observation_action_alignment",)
_REQUIRED_DATA_KEYS = frozenset(
    (*_FRAME_KEYS, *_SCALAR_OR_FRAME_KEYS, *_SCALAR_KEYS)
)


@dataclass(frozen=True)
class SplitSpec:
    label: str
    data_path: Path
    observed_q_path: Path
    canonical_q_path: Path
    trajectory_mode: str | None


@dataclass(frozen=True)
class InputLayout:
    name: str
    calibration: SplitSpec
    evaluation: SplitSpec


@dataclass(frozen=True)
class EpisodeGroup:
    key: EpisodeKey
    physical_identity: PhysicalIdentity
    trajectory_mode: str
    indices: np.ndarray
    first_frame: int
    last_frame: int


@dataclass
class LoadedSplit:
    spec: SplitSpec
    arrays: dict[str, np.ndarray]
    observed_q: np.ndarray
    canonical_q: np.ndarray
    indices: np.ndarray


def _positive_integer(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _nonnegative_integer(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    single = parser.add_argument_group("single-dataset bounded smoke")
    single.add_argument(
        "--data",
        "--dataset",
        "--randomized-data",
        "--randomized-npz",
        dest="data",
        type=Path,
        help="Randomized NPZ containing both trajectory partitions.",
    )
    single.add_argument(
        "--observed-q",
        "--observed-q-corrected",
        dest="observed_q",
        type=Path,
        help="Observed-view q_corrected.npy for --data.",
    )
    single.add_argument(
        "--canonical-q",
        "--canonical-q-corrected",
        dest="canonical_q",
        type=Path,
        help="Canonical-view q_corrected.npy for --data.",
    )
    single.add_argument(
        "--calibration-trajectory-mode",
        default=DEFAULT_CALIBRATION_TRAJECTORY_MODE,
        help="Whole-episode trajectory partition used to fit time-only.",
    )
    single.add_argument(
        "--evaluation-trajectory-mode",
        default=DEFAULT_EVALUATION_TRAJECTORY_MODE,
        help="Whole-episode trajectory partition used for evaluation.",
    )

    separate = parser.add_argument_group("separate development datasets")
    separate.add_argument("--calibration-data", type=Path)
    separate.add_argument(
        "--calibration-observed-q",
        "--calibration-observed-q-corrected",
        dest="calibration_observed_q",
        type=Path,
    )
    separate.add_argument(
        "--calibration-canonical-q",
        "--calibration-canonical-q-corrected",
        dest="calibration_canonical_q",
        type=Path,
    )
    separate.add_argument("--evaluation-data", type=Path)
    separate.add_argument(
        "--evaluation-observed-q",
        "--evaluation-observed-q-corrected",
        dest="evaluation_observed_q",
        type=Path,
    )
    separate.add_argument(
        "--evaluation-canonical-q",
        "--evaluation-canonical-q-corrected",
        dest="evaluation_canonical_q",
        type=Path,
    )

    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir", "--output", dest="output_dir", type=Path, required=True
    )
    parser.add_argument(
        "--batch-size", type=_positive_integer, default=DEFAULT_BATCH_SIZE
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--time-bins",
        "--n-time-bins",
        dest="time_bins",
        type=_positive_integer,
        default=DEFAULT_TIME_BINS,
    )
    parser.add_argument(
        "--min-count", type=_positive_integer, default=DEFAULT_MIN_COUNT
    )
    parser.add_argument(
        "--bootstrap-samples",
        "--bootstrap",
        dest="bootstrap_samples",
        type=_positive_integer,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=_nonnegative_integer,
        default=DEFAULT_BOOTSTRAP_SEED,
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _resolve_input_layout(args: argparse.Namespace) -> InputLayout:
    single_names = ("data", "observed_q", "canonical_q")
    separate_names = (
        "calibration_data",
        "calibration_observed_q",
        "calibration_canonical_q",
        "evaluation_data",
        "evaluation_observed_q",
        "evaluation_canonical_q",
    )
    single_present = [getattr(args, name) is not None for name in single_names]
    separate_present = [getattr(args, name) is not None for name in separate_names]
    if any(single_present) and any(separate_present):
        raise ValueError(
            "single-dataset and separate-dataset input arguments are mutually exclusive"
        )
    if any(single_present):
        missing = [
            name
            for name, present in zip(single_names, single_present)
            if not present
        ]
        if missing:
            raise ValueError(f"single-dataset input is missing arguments: {missing}")
        data = Path(args.data)
        observed = Path(args.observed_q)
        canonical = Path(args.canonical_q)
        calibration_mode = str(args.calibration_trajectory_mode)
        evaluation_mode = str(args.evaluation_trajectory_mode)
        if not calibration_mode or not evaluation_mode:
            raise ValueError("trajectory modes must be non-empty strings")
        return InputLayout(
            name="single_dataset_trajectory_partition",
            calibration=SplitSpec(
                "calibration", data, observed, canonical, calibration_mode
            ),
            evaluation=SplitSpec(
                "evaluation", data, observed, canonical, evaluation_mode
            ),
        )
    if any(separate_present):
        missing = [
            name
            for name, present in zip(separate_names, separate_present)
            if not present
        ]
        if missing:
            raise ValueError(f"separate-dataset input is missing arguments: {missing}")
        return InputLayout(
            name="separate_datasets",
            calibration=SplitSpec(
                "calibration",
                Path(args.calibration_data),
                Path(args.calibration_observed_q),
                Path(args.calibration_canonical_q),
                None,
            ),
            evaluation=SplitSpec(
                "evaluation",
                Path(args.evaluation_data),
                Path(args.evaluation_observed_q),
                Path(args.evaluation_canonical_q),
                None,
            ),
        )
    raise ValueError(
        "provide either --data with observed/canonical q arrays or both "
        "separate datasets"
    )


def _resolve_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _correction_file(path: Path, *, label: str) -> Path:
    requested = path.expanduser().resolve()
    resolved = requested / "q_corrected.npy" if requested.is_dir() else requested
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _integer_rows(value: np.ndarray, *, count: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != (count,)
        or array.dtype.kind not in "iu"
        or array.dtype == np.bool_
    ):
        raise TypeError(f"{name} must be an integer [{count}] array")
    return np.asarray(array, dtype=np.int64)


def _broadcast_integer_rows(
    value: np.ndarray, *, count: int, name: str
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape == ():
        if array.dtype.kind not in "iu" or array.dtype == np.bool_:
            raise TypeError(f"{name} must be a scalar or [{count}] integer array")
        return np.full(count, int(array.item()), dtype=np.int64)
    return _integer_rows(array, count=count, name=name)


def _string_rows(
    value: np.ndarray, *, count: int, name: str, allow_scalar: bool = False
) -> np.ndarray:
    array = np.asarray(value)
    if allow_scalar and array.shape == ():
        if array.dtype.kind not in "US":
            raise TypeError(f"{name} must use string storage")
        item = str(array.item())
        if not item:
            raise ValueError(f"{name} must not be empty")
        return np.full(count, item, dtype=f"<U{len(item)}")
    if array.shape != (count,) or array.dtype.kind not in "US":
        raise TypeError(f"{name} must be a string [{count}] array")
    result = array.astype(str)
    if bool(np.any(result == "")):
        raise ValueError(f"{name} must contain non-empty strings")
    return result


def _episode_rows(
    value: np.ndarray, *, count: int, name: str = "episode_id"
) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != (count,)
        or array.dtype.kind not in "USiu"
        or array.dtype == np.bool_
    ):
        raise TypeError(f"{name} must be a string or integer [{count}] array")
    result = array.astype(str)
    if bool(np.any(result == "")):
        raise ValueError(f"{name} must contain non-empty values")
    return result


def _finite_float_array(
    value: np.ndarray, *, shape: tuple[int, ...], name: str
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must be a floating array with shape {shape}")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(array, dtype=np.float32)


def _load_randomized_data(path: Path) -> dict[str, np.ndarray]:
    source = _resolve_file(path, label="randomized dataset")
    with np.load(source, allow_pickle=False) as raw:
        missing = sorted(_REQUIRED_DATA_KEYS.difference(raw.files))
        if missing:
            raise KeyError(f"randomized dataset is missing fields: {missing}")
        arrays = {
            name: np.array(raw[name], copy=True) for name in _REQUIRED_DATA_KEYS
        }

    rgb = np.asarray(arrays["rgb"])
    if (
        rgb.ndim != 4
        or rgb.shape[-1] != 3
        or rgb.dtype != np.uint8
        or any(int(size) <= 0 for size in rgb.shape)
    ):
        raise ValueError("rgb must be nonempty uint8 [N,H,W,3]")
    count = int(rgb.shape[0])
    canonical = np.asarray(arrays["rgb_camera_only"])
    if canonical.shape != rgb.shape or canonical.dtype != np.uint8:
        raise ValueError("rgb_camera_only must match rgb as uint8 [N,H,W,3]")

    arrays["task_name"] = _string_rows(
        arrays["task_name"], count=count, name="task_name"
    )
    arrays["episode_id"] = _episode_rows(arrays["episode_id"], count=count)
    arrays["source_episode_id"] = _episode_rows(
        arrays["source_episode_id"], count=count, name="source_episode_id"
    )
    arrays["randomized_source_trajectory_mode"] = _string_rows(
        arrays["randomized_source_trajectory_mode"],
        count=count,
        name="randomized_source_trajectory_mode",
        allow_scalar=True,
    )
    for name in ("task_index", "env_seed", "frame_id"):
        arrays[name] = _integer_rows(arrays[name], count=count, name=name)
    arrays["task_generation_seed"] = _broadcast_integer_rows(
        arrays["task_generation_seed"], count=count, name="task_generation_seed"
    )

    alignment = np.asarray(arrays["observation_action_alignment"])
    if alignment.shape != () or alignment.dtype.kind not in "US":
        raise TypeError("observation_action_alignment must be a scalar string")
    if str(alignment.item()) != "pre_action":
        raise ValueError("observation_action_alignment must be 'pre_action'")
    arrays["observation_action_alignment"] = np.asarray("pre_action")

    obs = np.asarray(arrays["obs"])
    action = np.asarray(arrays["action"])
    if obs.ndim != 2 or obs.shape[0] != count or obs.shape[1] <= 0:
        raise ValueError("obs must have nonempty shape [N,D]")
    if action.ndim != 2 or action.shape[0] != count or action.shape[1] <= 0:
        raise ValueError("action must have nonempty shape [N,A]")
    arrays["obs"] = _finite_float_array(
        obs, shape=tuple(obs.shape), name="obs"
    )
    arrays["action"] = _finite_float_array(
        action, shape=tuple(action.shape), name="action"
    )

    q_nominal = np.asarray(arrays["q_nominal"])
    if q_nominal.ndim != 3 or q_nominal.shape[0] != count or q_nominal.shape[-1] != 2:
        raise ValueError("q_nominal must have nonempty shape [N,K,2]")
    anchors = int(q_nominal.shape[1])
    if anchors <= 0:
        raise ValueError("q_nominal must contain at least one anchor")
    arrays["q_nominal"] = _finite_float_array(
        q_nominal, shape=(count, anchors, 2), name="q_nominal"
    )
    arrays["keypoints_xyz"] = _finite_float_array(
        arrays["keypoints_xyz"],
        shape=(count, anchors, 3),
        name="keypoints_xyz",
    )
    for name in (
        "valid_nominal",
        "anchor_dropout_active",
        "anchor_recovery_segment",
    ):
        value = np.asarray(arrays[name])
        if value.shape != (count, anchors) or value.dtype != np.bool_:
            raise TypeError(f"{name} must be boolean [{count},{anchors}]")
    return arrays


def _load_q_corrected(
    path: Path, *, count: int, anchors: int, label: str
) -> np.ndarray:
    source = _correction_file(path, label=label)
    value = np.load(source, allow_pickle=False)
    return np.array(
        _finite_float_array(
            value, shape=(count, anchors, 2), name=f"{label} q_corrected"
        ),
        copy=True,
    )


def _episode_groups(arrays: Mapping[str, np.ndarray]) -> tuple[EpisodeGroup, ...]:
    count = int(arrays["rgb"].shape[0])
    task = np.asarray(arrays["task_name"]).astype(str)
    episode = np.asarray(arrays["episode_id"]).astype(str)
    frame = np.asarray(arrays["frame_id"], dtype=np.int64)
    mode = _string_rows(
        arrays["randomized_source_trajectory_mode"],
        count=count,
        name="randomized_source_trajectory_mode",
        allow_scalar=True,
    )
    task_index = np.asarray(arrays["task_index"], dtype=np.int64)
    env_seed = np.asarray(arrays["env_seed"], dtype=np.int64)
    generation = _broadcast_integer_rows(
        arrays["task_generation_seed"],
        count=count,
        name="task_generation_seed",
    )
    for name, value in (
        ("task_name", task),
        ("episode_id", episode),
        ("frame_id", frame),
        ("randomized_source_trajectory_mode", mode),
        ("task_index", task_index),
        ("env_seed", env_seed),
        ("task_generation_seed", generation),
    ):
        if value.shape != (count,):
            raise ValueError(f"{name} must contain one value per frame")

    grouped: dict[EpisodeKey, list[int]] = {}
    for index, key in enumerate(zip(task.tolist(), episode.tolist(), strict=True)):
        grouped.setdefault((str(key[0]), str(key[1])), []).append(index)

    result: list[EpisodeGroup] = []
    for key in sorted(grouped):
        raw_indices = np.asarray(grouped[key], dtype=np.int64)
        order = np.argsort(frame[raw_indices], kind="stable")
        indices = raw_indices[order]
        frames = frame[indices]
        if frames.size > 1 and not bool(np.all(np.diff(frames) == 1)):
            raise ValueError(f"episode {key!r} must have unique contiguous frame IDs")
        modes = np.unique(mode[indices])
        if modes.size != 1:
            raise ValueError(f"trajectory mode changes within physical episode {key!r}")
        identities = {
            (
                str(task[index]),
                int(task_index[index]),
                int(env_seed[index]),
                int(generation[index]),
            )
            for index in indices.tolist()
        }
        if len(identities) != 1:
            raise ValueError(
                f"simulator physical identity changes within episode {key!r}"
            )
        identity = next(iter(identities))
        result.append(
            EpisodeGroup(
                key=key,
                physical_identity=identity,
                trajectory_mode=str(modes[0]),
                indices=indices,
                first_frame=int(frames[0]),
                last_frame=int(frames[-1]),
            )
        )
    if not result:
        raise ValueError("randomized dataset contains no physical episodes")
    return tuple(result)


def _selected_episode_groups(
    arrays: Mapping[str, np.ndarray], indices: np.ndarray
) -> tuple[EpisodeGroup, ...]:
    selected = set(np.asarray(indices, dtype=np.int64).tolist())
    result: list[EpisodeGroup] = []
    for group in _episode_groups(arrays):
        group_rows = set(group.indices.tolist())
        intersection = selected.intersection(group_rows)
        if not intersection:
            continue
        if intersection != group_rows:
            raise ValueError(f"selected rows split physical view episode {group.key!r}")
        result.append(group)
    if not result:
        raise ValueError("selected rows contain no complete view episode")
    return tuple(result)


def _resampling_units(
    arrays: Mapping[str, np.ndarray], indices: np.ndarray
) -> dict[EpisodeKey, str]:
    """Map paired camera views onto independent source trajectories.

    The randomized dataset contains step and linear-drift views of the same
    robot trajectory.  Those views are useful conditions, but they are not
    independent bootstrap units.
    """

    groups = _selected_episode_groups(arrays, indices)
    source = np.asarray(arrays["source_episode_id"]).astype(str)
    parents: dict[tuple[str, str], list[EpisodeGroup]] = {}
    mapping: dict[EpisodeKey, str] = {}
    for group in groups:
        source_ids = np.unique(source[group.indices])
        if source_ids.size != 1:
            raise ValueError(
                f"source_episode_id changes within view episode {group.key!r}"
            )
        parent = (group.key[0], str(source_ids[0]))
        parents.setdefault(parent, []).append(group)
        mapping[group.key] = parent[1]

    physical_fields = ("frame_id", "obs", "action", "keypoints_xyz")
    for parent, views in parents.items():
        modes = [view.trajectory_mode for view in views]
        if len(modes) != len(set(modes)):
            raise ValueError(
                f"source trajectory {parent!r} has duplicate camera modes {modes!r}"
            )
        reference = views[0]
        for view in views[1:]:
            for name in physical_fields:
                left = np.asarray(arrays[name])[reference.indices]
                right = np.asarray(arrays[name])[view.indices]
                if not np.array_equal(left, right):
                    raise ValueError(
                        f"paired camera views for source trajectory {parent!r} "
                        f"differ in physical field {name!r}"
                    )
    return mapping


def _assert_disjoint_physical_identities(
    calibration: set[PhysicalIdentity],
    evaluation: set[PhysicalIdentity],
) -> None:
    overlap = calibration.intersection(evaluation)
    if overlap:
        sample = sorted(overlap)[:3]
        raise ValueError(
            "physical episode identity leakage between calibration and evaluation: "
            f"{sample}"
        )


def _split_by_trajectory_mode(
    arrays: Mapping[str, np.ndarray],
    calibration_mode: str,
    evaluation_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not calibration_mode or not evaluation_mode:
        raise ValueError("trajectory modes must be non-empty strings")
    if calibration_mode == evaluation_mode:
        raise ValueError("calibration and evaluation trajectory modes must differ")
    groups = _episode_groups(arrays)
    calibration_groups = [
        group for group in groups if group.trajectory_mode == calibration_mode
    ]
    evaluation_groups = [
        group for group in groups if group.trajectory_mode == evaluation_mode
    ]
    if not calibration_groups:
        raise ValueError(
            f"no calibration episodes use trajectory mode {calibration_mode!r}"
        )
    if not evaluation_groups:
        raise ValueError(
            f"no evaluation episodes use trajectory mode {evaluation_mode!r}"
        )
    _assert_disjoint_physical_identities(
        {group.physical_identity for group in calibration_groups},
        {group.physical_identity for group in evaluation_groups},
    )
    calibration = np.sort(
        np.concatenate([group.indices for group in calibration_groups])
    )
    evaluation = np.sort(
        np.concatenate([group.indices for group in evaluation_groups])
    )
    calibration_units = {
        (task, unit)
        for (task, _), unit in _resampling_units(arrays, calibration).items()
    }
    evaluation_units = {
        (task, unit)
        for (task, _), unit in _resampling_units(arrays, evaluation).items()
    }
    overlap = calibration_units.intersection(evaluation_units)
    if overlap:
        raise ValueError(
            "source trajectory leakage between calibration and evaluation: "
            f"{sorted(overlap)[:3]}"
        )
    return calibration, evaluation


def _all_indices_and_identities(
    arrays: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, set[PhysicalIdentity]]:
    groups = _episode_groups(arrays)
    indices = np.sort(np.concatenate([group.indices for group in groups]))
    return indices, {group.physical_identity for group in groups}


def _assert_disjoint_datasets(
    calibration: Mapping[str, np.ndarray],
    evaluation: Mapping[str, np.ndarray],
) -> None:
    _, calibration_identities = _all_indices_and_identities(calibration)
    _, evaluation_identities = _all_indices_and_identities(evaluation)
    _assert_disjoint_physical_identities(
        calibration_identities, evaluation_identities
    )


def _episode_frame_bounds(
    arrays: Mapping[str, np.ndarray], indices: np.ndarray
) -> dict[EpisodeKey, tuple[int, int]]:
    bounds: dict[EpisodeKey, tuple[int, int]] = {}
    for group in _selected_episode_groups(arrays, indices):
        if group.last_frame <= group.first_frame:
            raise ValueError(f"episode {group.key!r} must span at least two frames")
        bounds[group.key] = (group.first_frame, group.last_frame)
    return bounds


def _inside_image(q: np.ndarray, *, width: int, height: int) -> np.ndarray:
    return (
        np.isfinite(q).all(axis=-1)
        & (q[..., 0] >= 0.0)
        & (q[..., 0] <= float(width - 1))
        & (q[..., 1] >= 0.0)
        & (q[..., 1] <= float(height - 1))
    )


def _encode_frames(
    model: AnchorPersistentLeWM,
    *,
    images: np.ndarray,
    obs: np.ndarray,
    q_nominal: np.ndarray,
    q_corrected: np.ndarray,
    keypoints_xyz: np.ndarray,
    valid_nominal: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Encode per-anchor frames without exposing offline truth to the model."""

    count, height, width = images.shape[:3]
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if q_corrected.shape != q_nominal.shape:
        raise ValueError("q_corrected must match q_nominal")
    valid = valid_nominal & _inside_image(q_corrected, width=width, height=height)
    obs_dim = int(model.frame_encoder.obs_dim)
    if obs.ndim != 2 or obs.shape[0] != count or obs.shape[1] < obs_dim:
        raise ValueError(f"obs must provide at least {obs_dim} model dimensions")

    chunks: list[np.ndarray] = []
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            batch = {
                "pixels": torch.from_numpy(
                    np.ascontiguousarray(images[start:stop].transpose(0, 3, 1, 2))
                )
                .to(device=device, dtype=torch.float32, non_blocking=amp_enabled)
                .div_(255.0)
                .unsqueeze(1),
                "proprio": torch.from_numpy(
                    np.ascontiguousarray(obs[start:stop, :obs_dim])
                )
                .to(device=device, dtype=torch.float32, non_blocking=amp_enabled)
                .unsqueeze(1),
                "q_nominal": torch.from_numpy(
                    np.ascontiguousarray(q_nominal[start:stop])
                )
                .to(device=device, dtype=torch.float32, non_blocking=amp_enabled)
                .unsqueeze(1),
                "q_corrected": torch.from_numpy(
                    np.ascontiguousarray(q_corrected[start:stop])
                )
                .to(device=device, dtype=torch.float32, non_blocking=amp_enabled)
                .unsqueeze(1),
                "keypoints_xyz": torch.from_numpy(
                    np.ascontiguousarray(keypoints_xyz[start:stop])
                )
                .to(device=device, dtype=torch.float32, non_blocking=amp_enabled)
                .unsqueeze(1),
                "valid_nominal": torch.from_numpy(
                    np.ascontiguousarray(valid[start:stop])
                )
                .to(device=device, dtype=torch.bool, non_blocking=amp_enabled)
                .unsqueeze(1),
            }
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                _, embedding = model.encode_frames(batch)
            if embedding.ndim != 4 or embedding.shape[:2] != (stop - start, 1):
                raise ValueError("model.encode_frames must return embedding [B,1,K,D]")
            chunks.append(embedding[:, 0].detach().float().cpu().numpy())
    output = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    expected = (count, int(model.num_keypoints), int(model.latent_dim))
    if output.shape != expected or not bool(np.isfinite(output).all()):
        raise ValueError(f"encoded embeddings must be finite {expected}")
    return output


def _validate_frozen_model(
    model: object,
    checkpoint: Mapping[str, Any],
    datasets: Sequence[Mapping[str, np.ndarray]],
) -> AnchorPersistentLeWM:
    if not isinstance(model, AnchorPersistentLeWM):
        raise TypeError("randomized oracle requires AnchorPersistentLeWM")
    step = checkpoint.get("step")
    if (
        isinstance(step, bool)
        or not isinstance(step, Integral)
        or int(step) != EXPECTED_CHECKPOINT_STEP
    ):
        raise ValueError("randomized oracle requires the frozen step-2000 LeWM")
    if int(model.history_size) != RANDOMIZED_HISTORY_SIZE:
        raise ValueError(
            f"AnchorPersistentLeWM history_size must be {RANDOMIZED_HISTORY_SIZE}"
        )
    for arrays in datasets:
        anchors = int(arrays["q_nominal"].shape[1])
        if anchors != int(model.num_keypoints):
            raise ValueError("dataset anchor count differs from checkpoint")
        action_width = int(arrays["action"].shape[1]) * RANDOMIZED_FRAME_SKIP
        if action_width != int(model.action_dim):
            raise ValueError(
                "dataset raw action width times frame skip differs from checkpoint"
            )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def _build_sequence(
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    measurement_embedding: np.ndarray,
    canonical_embedding: np.ndarray,
    q_corrected: np.ndarray,
) -> PosteriorRandomizedSequenceExamples:
    rows = np.asarray(indices, dtype=np.int64)
    return build_fresh_posterior_randomized_sequence(
        task=np.asarray(arrays["task_name"])[rows],
        episode=np.asarray(arrays["episode_id"])[rows],
        frame=np.asarray(arrays["frame_id"])[rows],
        action=np.asarray(arrays["action"])[rows],
        measurement_embedding=measurement_embedding[rows],
        canonical_label=canonical_embedding[rows],
        q_nominal=np.asarray(arrays["q_nominal"])[rows],
        q_corrected=q_corrected[rows],
        valid_nominal=np.asarray(arrays["valid_nominal"])[rows],
        anchor_dropout_active=np.asarray(arrays["anchor_dropout_active"])[rows],
        anchor_recovery_segment=np.asarray(arrays["anchor_recovery_segment"])[rows],
        observation_action_alignment=str(
            np.asarray(arrays["observation_action_alignment"]).item()
        ),
    )


def _direct_anchors(
    examples: PosteriorRandomizedSequenceExamples,
) -> dict[EpisodeKey, int]:
    inferred = infer_direct_anchors(
        examples.metadata, examples.evaluation.dropout_active
    )
    return {
        (str(task), str(episode)): int(anchor)
        for (task, episode), anchor in inferred.items()
    }


def _oracle_records(
    result: Any,
    examples: PosteriorRandomizedSequenceExamples,
    direct_anchors: Mapping[EpisodeKey, int],
    resampling_units: Mapping[EpisodeKey, str],
    *,
    clean: bool,
) -> tuple[dict[str, Any], ...]:
    records = build_randomized_oracle_records(
        result,
        examples.labels.canonical_current,
        direct_anchors,
        clean=clean,
    )
    expected = {
        (str(row.task), str(row.episode_id)) for row in examples.metadata
    }
    if set(resampling_units) != expected:
        raise ValueError("resampling_units must cover exactly the evaluated views")
    output: list[dict[str, Any]] = []
    for record in records:
        key = (str(record["task"]), str(record["physical_episode_id"]))
        updated = dict(record)
        updated["resampling_unit_id"] = resampling_units[key]
        output.append(updated)
    return tuple(output)


def _trajectory_summary(
    arrays: Mapping[str, np.ndarray], indices: np.ndarray
) -> dict[str, Any]:
    selected = set(np.asarray(indices, dtype=np.int64).tolist())
    groups = [
        group
        for group in _episode_groups(arrays)
        if selected.intersection(group.indices.tolist())
    ]
    episode_counts = Counter(group.trajectory_mode for group in groups)
    row_counts = Counter(
        str(value)
        for value in np.asarray(arrays["randomized_source_trajectory_mode"])[indices]
    )
    return {
        "episode_counts": dict(sorted(episode_counts.items())),
        "frame_counts": dict(sorted(row_counts.items())),
    }


def _split_report(split: LoadedSplit) -> dict[str, Any]:
    groups = _selected_episode_groups(split.arrays, split.indices)
    units = _resampling_units(split.arrays, split.indices)
    return {
        "data": str(split.spec.data_path.expanduser().resolve()),
        "observed_q_corrected": str(
            _correction_file(split.spec.observed_q_path, label="observed correction")
        ),
        "canonical_q_corrected": str(
            _correction_file(split.spec.canonical_q_path, label="canonical correction")
        ),
        "trajectory_filter": split.spec.trajectory_mode,
        "frame_count": int(split.indices.size),
        "view_episode_count": len(groups),
        "physical_trajectory_count": len(
            {(task, unit) for (task, _), unit in units.items()}
        ),
        # Retained as a compatibility alias for downstream report readers.
        "physical_episode_count": len(
            {(task, unit) for (task, _), unit in units.items()}
        ),
        "task_count": len({group.key[0] for group in groups}),
        "trajectories": _trajectory_summary(split.arrays, split.indices),
    }


def _percent(value: object) -> str:
    return f"{float(value):.2%}"


def _markdown(payload: Mapping[str, Any]) -> str:
    statistics = payload["statistics"]
    primary = statistics["primary"]
    decision = payload["decision"]
    inputs = payload["inputs"]
    calibration = inputs["calibration"]
    evaluation = inputs["evaluation"]
    candidate_ci = primary["candidate_relative_improvement_95_ci"]
    incremental_ci = primary[
        "candidate_incremental_improvement_over_time_only_95_ci"
    ]
    lines = [
        "# Fresh Randomized Posterior Oracle",
        "",
        f"Decision: **{decision['decision']}**",
        "",
        "The candidate is the privileged cautious-enter/fast-exit FSM. "
        "The event-aligned time-only shortcut is fit on calibration trajectories "
        "only. Canonical frames directly define privileged oracle decisions and "
        "offline scores; canonical data are not deployment inputs, and q_true is "
        "never a model input.",
        "",
        "## Inputs",
        "",
        f"Layout: `{inputs['layout']}`",
        "",
        "| Split | Frames | View episodes | Independent trajectories | Modes |",
        "| --- | ---: | ---: | ---: | --- |",
        "| Calibration | {frames} | {views} | {trajectories} | {modes} |".format(
            frames=int(calibration["frame_count"]),
            views=int(calibration["view_episode_count"]),
            trajectories=int(calibration["physical_trajectory_count"]),
            modes=", ".join(
                f"{name}: {count}"
                for name, count in calibration["trajectories"][
                    "episode_counts"
                ].items()
            ),
        ),
        "| Evaluation | {frames} | {views} | {trajectories} | {modes} |".format(
            frames=int(evaluation["frame_count"]),
            views=int(evaluation["view_episode_count"]),
            trajectories=int(evaluation["physical_trajectory_count"]),
            modes=", ".join(
                f"{name}: {count}"
                for name, count in evaluation["trajectories"][
                    "episode_counts"
                ].items()
            ),
        ),
        "",
        "## Primary",
        "",
        "Active event 0 is excluded before equal view, source-trajectory, and "
        "task aggregation.",
        "",
        "| Comparison | Effect | 95% CI |",
        "| --- | ---: | ---: |",
        "| FSM vs measurement | {effect} | [{low}, {high}] |".format(
            effect=_percent(primary["candidate_relative_improvement"]),
            low=_percent(candidate_ci[0]),
            high=_percent(candidate_ci[1]),
        ),
        "| Time-only vs measurement | {effect} | [{low}, {high}] |".format(
            effect=_percent(primary["time_only_relative_improvement"]),
            low=_percent(primary["time_only_relative_improvement_95_ci"][0]),
            high=_percent(primary["time_only_relative_improvement_95_ci"][1]),
        ),
        "| FSM incremental over time-only | {effect} | [{low}, {high}] |".format(
            effect=_percent(
                primary["candidate_incremental_improvement_over_time_only"]
            ),
            low=_percent(incremental_ci[0]),
            high=_percent(incremental_ci[1]),
        ),
        "",
        "## Per Task",
        "",
        "| Task | FSM vs measurement | FSM over time-only | Trajectories |",
        "| --- | ---: | ---: | ---: |",
    ]
    for task, row in primary["per_task"].items():
        lines.append(
            "| {task} | {candidate} | {incremental} | {episodes} |".format(
                task=task,
                candidate=_percent(row["candidate_relative_improvement"]),
                incremental=_percent(
                    row["candidate_incremental_improvement_over_time_only"]
                ),
                episodes=int(row["physical_episode_count"]),
            )
        )
    lines.extend(
        [
            "",
            "## Coverage And Diagnostics",
            "",
            f"Evaluated {statistics['coverage']['physical_trajectory_count']} "
            f"independent source trajectories from "
            f"{statistics['coverage']['view_episode_count']} camera-view episodes "
            f"across {statistics['coverage']['task_count']} tasks. Minimum "
            f"active/recovery event counts per view are "
            f"{statistics['coverage']['minimum_active_events_per_episode']}/"
            f"{statistics['coverage']['minimum_recovery_events_per_episode']}.",
            "",
            "Recovery and clean one-sided upper bounds are hard feasibility "
            "checks, but remain privileged and are not deployable recovery evidence.",
            "",
            "| Window | Normalized excess MSE | 95% CI | One-sided 95% upper |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name in ("recovery", "clean"):
        row = statistics[name]
        interval = row["normalized_excess_mse_95_ci"]
        lines.append(
            "| {name} | {effect} | [{low}, {high}] | {upper} |".format(
                name=name.title(),
                effect=_percent(row["normalized_excess_mse"]),
                low=_percent(interval[0]),
                high=_percent(interval[1]),
                upper=_percent(
                    row["normalized_excess_mse_one_sided_95_upper"]
                ),
            )
        )
    if decision["failed_checks"]:
        lines.extend(
            [
                "",
                "Failed checks: "
                + ", ".join(f"`{name}`" for name in decision["failed_checks"]),
            ]
        )
    return "\n".join(lines) + "\n"


def _load_split_arrays(
    spec: SplitSpec,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    arrays = _load_randomized_data(spec.data_path)
    count = int(arrays["rgb"].shape[0])
    anchors = int(arrays["q_nominal"].shape[1])
    observed_q = _load_q_corrected(
        spec.observed_q_path,
        count=count,
        anchors=anchors,
        label=f"{spec.label} observed correction",
    )
    canonical_q = _load_q_corrected(
        spec.canonical_q_path,
        count=count,
        anchors=anchors,
        label=f"{spec.label} canonical correction",
    )
    return arrays, observed_q, canonical_q


def _load_inputs(layout: InputLayout) -> tuple[LoadedSplit, LoadedSplit]:
    if layout.name == "single_dataset_trajectory_partition":
        arrays, observed_q, canonical_q = _load_split_arrays(layout.calibration)
        calibration_indices, evaluation_indices = _split_by_trajectory_mode(
            arrays,
            str(layout.calibration.trajectory_mode),
            str(layout.evaluation.trajectory_mode),
        )
        return (
            LoadedSplit(
                layout.calibration,
                arrays,
                observed_q,
                canonical_q,
                calibration_indices,
            ),
            LoadedSplit(
                layout.evaluation,
                arrays,
                observed_q,
                canonical_q,
                evaluation_indices,
            ),
        )

    calibration_arrays, calibration_observed, calibration_canonical = (
        _load_split_arrays(layout.calibration)
    )
    evaluation_arrays, evaluation_observed, evaluation_canonical = (
        _load_split_arrays(layout.evaluation)
    )
    _assert_disjoint_datasets(calibration_arrays, evaluation_arrays)
    calibration_indices, _ = _all_indices_and_identities(calibration_arrays)
    evaluation_indices, _ = _all_indices_and_identities(evaluation_arrays)
    return (
        LoadedSplit(
            layout.calibration,
            calibration_arrays,
            calibration_observed,
            calibration_canonical,
            calibration_indices,
        ),
        LoadedSplit(
            layout.evaluation,
            evaluation_arrays,
            evaluation_observed,
            evaluation_canonical,
            evaluation_indices,
        ),
    )


def _encode_split(
    model: AnchorPersistentLeWM,
    split: LoadedSplit,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = split.arrays
    print(f"encoding {split.spec.label} observed rgb", flush=True)
    observed = _encode_frames(
        model,
        images=arrays["rgb"],
        obs=arrays["obs"],
        q_nominal=arrays["q_nominal"],
        q_corrected=split.observed_q,
        keypoints_xyz=arrays["keypoints_xyz"],
        valid_nominal=arrays["valid_nominal"],
        batch_size=batch_size,
        device=device,
    )
    print(f"encoding {split.spec.label} canonical rgb_camera_only", flush=True)
    canonical = _encode_frames(
        model,
        images=arrays["rgb_camera_only"],
        obs=arrays["obs"],
        q_nominal=arrays["q_nominal"],
        q_corrected=split.canonical_q,
        keypoints_xyz=arrays["keypoints_xyz"],
        valid_nominal=arrays["valid_nominal"],
        batch_size=batch_size,
        device=device,
    )
    return observed, canonical


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    layout = _resolve_input_layout(args)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)

    calibration, evaluation = _load_inputs(layout)
    checkpoint_path = _resolve_file(args.checkpoint, label="LeWM checkpoint")
    device = choose_device(args.device)
    model, _, checkpoint = load_lewm_grounded_checkpoint(
        checkpoint_path, device=device
    )
    unique_datasets = [calibration.arrays]
    if evaluation.arrays is not calibration.arrays:
        unique_datasets.append(evaluation.arrays)
    model = _validate_frozen_model(model, checkpoint, unique_datasets)

    calibration_observed, calibration_canonical = _encode_split(
        model, calibration, batch_size=args.batch_size, device=device
    )
    if evaluation.arrays is calibration.arrays:
        evaluation_observed = calibration_observed
        evaluation_canonical = calibration_canonical
    else:
        evaluation_observed, evaluation_canonical = _encode_split(
            model, evaluation, batch_size=args.batch_size, device=device
        )

    calibration_examples = _build_sequence(
        calibration.arrays,
        calibration.indices,
        measurement_embedding=calibration_observed,
        canonical_embedding=calibration_canonical,
        q_corrected=calibration.observed_q,
    )
    evaluation_examples = _build_sequence(
        evaluation.arrays,
        evaluation.indices,
        measurement_embedding=evaluation_observed,
        canonical_embedding=evaluation_canonical,
        q_corrected=evaluation.observed_q,
    )
    clean_examples = _build_sequence(
        evaluation.arrays,
        evaluation.indices,
        measurement_embedding=evaluation_canonical,
        canonical_embedding=evaluation_canonical,
        q_corrected=evaluation.canonical_q,
    )
    calibration_resampling_units = _resampling_units(
        calibration.arrays, calibration.indices
    )
    evaluation_resampling_units = _resampling_units(
        evaluation.arrays, evaluation.indices
    )

    calibration_bounds = _episode_frame_bounds(
        calibration.arrays, calibration.indices
    )
    evaluation_bounds = _episode_frame_bounds(evaluation.arrays, evaluation.indices)
    calibration_time = normalized_episode_time(
        calibration_examples.metadata, calibration_bounds
    )
    evaluation_time = normalized_episode_time(
        evaluation_examples.metadata, evaluation_bounds
    )
    clean_time = normalized_episode_time(clean_examples.metadata, evaluation_bounds)
    time_only_policy = fit_time_only_from_teacher_forced(
        model,
        calibration_examples,
        calibration_time,
        resampling_units=calibration_resampling_units,
        n_time_bins=args.time_bins,
        min_count=args.min_count,
        device=device,
    )

    observed_result = run_randomized_recursive_oracle(
        model,
        evaluation_examples,
        methods=EVALUATION_METHODS,
        time_only_policy=time_only_policy,
        normalized_time=evaluation_time,
        device=device,
    )
    clean_result = run_randomized_recursive_oracle(
        model,
        clean_examples,
        methods=EVALUATION_METHODS,
        time_only_policy=time_only_policy,
        normalized_time=clean_time,
        device=device,
    )
    direct = _direct_anchors(evaluation_examples)
    observed_records = _oracle_records(
        observed_result,
        evaluation_examples,
        direct,
        evaluation_resampling_units,
        clean=False,
    )
    clean_records = _oracle_records(
        clean_result,
        clean_examples,
        direct,
        evaluation_resampling_units,
        clean=True,
    )
    records = (*observed_records, *clean_records)
    statistics = summarize_randomized_oracle(
        records,
        measurement_method=MEASUREMENT_METHOD,
        candidate_method=FSM_ORACLE_METHOD,
        time_only_method=TIME_ONLY_METHOD,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    decision = randomized_oracle_decision(statistics)

    payload = {
        "schema_version": 1,
        "status": "fresh_randomized_privileged_oracle",
        "decision": decision,
        "inputs": {
            "layout": layout.name,
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": int(checkpoint["step"]),
            "device": str(device),
            "calibration": _split_report(calibration),
            "evaluation": _split_report(evaluation),
        },
        "protocol": {
            "model": "AnchorPersistentLeWM",
            "model_frozen": True,
            "frame_skip": RANDOMIZED_FRAME_SKIP,
            "history_size": RANDOMIZED_HISTORY_SIZE,
            "recursive_methods": list(EVALUATION_METHODS),
            "candidate_method": FSM_ORACLE_METHOD,
            "time_only_fit_split": "calibration_only",
            "time_only_role": "privileged_event_aligned_schedule_shortcut",
            "paired_camera_views_reduced_within_source_trajectory": True,
            "physical_identity_fields": [
                "task_name",
                "task_index",
                "env_seed",
                "task_generation_seed",
            ],
            "resampling_unit_fields": ["task_name", "source_episode_id"],
            "calibration_evaluation_physical_identities_disjoint": True,
            "normalized_time_uses_full_episode_bounds": True,
            "clean_evaluation_uses": "rgb_camera_only_and_canonical_q_corrected",
            "q_true_model_input": False,
        },
        "time_only": {
            "n_time_bins": int(time_only_policy.n_time_bins),
            "min_count": int(time_only_policy.min_count),
            "calibration_row_count": int(calibration_examples.sample_count),
            "occupied_cell_count": int(np.count_nonzero(time_only_policy.count)),
            "prior_cell_count": int(np.count_nonzero(time_only_policy.choose_prior)),
        },
        "evaluation": {
            "observed_sequence_rows": int(evaluation_examples.sample_count),
            "clean_sequence_rows": int(clean_examples.sample_count),
            "phase_stream_count": int(evaluation_examples.phase_stream_count),
            "record_count": len(records),
        },
        "statistics": statistics,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        _markdown(payload), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "decision": decision["decision"],
                "physical_episode_count": statistics["coverage"][
                    "physical_episode_count"
                ],
                "view_episode_count": statistics["coverage"][
                    "view_episode_count"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
