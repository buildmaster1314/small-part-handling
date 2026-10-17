"""Derive randomized visual-corruption data for the LeWM posterior gate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
from typing import TypeAlias

import numpy as np

from fcpg.visual_trust_corruptions import (
    CORRUPTION_TYPES,
    apply_visual_trust_corruption,
)


ArraySource: TypeAlias = Mapping[str, np.ndarray]

GATE_FRAME_KEYS = (
    "obs",
    "action",
    "task_name",
    "task_index",
    "episode_id",
    "env_seed",
    "frame_id",
    "keypoints_xyz",
    "q_true",
    "q_nominal",
    "valid_true",
    "valid_nominal",
    "camera_step_frame",
)
GATE_SCALAR_KEYS = (
    "task_generation_seed",
    "observation_action_alignment",
)
SCHEDULE_KEYS = (
    "anchor_dropout_active",
    "anchor_recovery_onset",
    "anchor_recovery_segment",
    "measurement_valid",
)
GENERATION_ANNOTATION_KEYS = (
    "randomized_corruption_family",
    "randomized_anchor_index",
    "randomized_onset_frame",
    "randomized_duration",
    "randomized_severity_tier",
    "randomized_onset_relative_phase",
    "randomized_source_trajectory_mode",
)
SEVERITY_TIERS = (1, 2, 3)
_TRAJECTORY_MODE_KEYS = (
    "actual_camera_trajectory_mode",
    "actual_camera_randomization",
)
_REPLACED_KEYS = {
    "rgb",
    "rgb_camera_only",
    "source_episode_id",
    *SCHEDULE_KEYS,
    *GENERATION_ANNOTATION_KEYS,
}


def _integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        relation = "non-negative" if minimum == 0 else f"at least {minimum}"
        raise ValueError(f"{name} must be {relation}")
    return result


def _phase(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _copy_array(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError(f"{name} must not use object storage")
    return np.array(array, copy=True)


def _scalar_string(value: np.ndarray, *, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "US":
        raise TypeError(f"{name} must be a scalar string")
    result = str(array.item())
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _source_mode(source: ArraySource, count: int) -> np.ndarray:
    key = next((name for name in _TRAJECTORY_MODE_KEYS if name in source), None)
    if key is None:
        raise KeyError(
            "input dataset must contain actual_camera_trajectory_mode or "
            "actual_camera_randomization"
        )
    value = _copy_array(source[key], name=key)
    if value.shape == ():
        mode = _scalar_string(value, name=key)
        return np.full(count, mode, dtype=f"<U{len(mode)}")
    if value.shape != (count,) or value.dtype.kind not in "US":
        raise TypeError(f"{key} must be a scalar string or [{count}] string array")
    modes = value.astype(str)
    if bool(np.any(modes == "")):
        raise ValueError(f"{key} must contain non-empty values")
    return modes


def _validate_source(
    source: ArraySource,
    *,
    source_index: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    required = {*GATE_FRAME_KEYS, *GATE_SCALAR_KEYS}
    missing = sorted(required.difference(source))
    if missing:
        raise KeyError(
            f"input source {source_index} is missing gate-required fields: {missing}"
        )
    clean_key = "rgb_camera_only" if "rgb_camera_only" in source else "rgb"
    if clean_key not in source:
        raise KeyError(
            f"input source {source_index} must contain rgb_camera_only or rgb"
        )

    arrays = {
        name: _copy_array(value, name=name) for name, value in source.items()
    }
    clean = arrays[clean_key]
    if (
        clean.ndim != 4
        or clean.shape[-1] != 3
        or clean.dtype != np.uint8
        or any(int(size) <= 0 for size in clean.shape)
    ):
        raise ValueError(
            f"source {source_index} {clean_key} must be nonempty uint8 [N,H,W,3]"
        )
    count = int(clean.shape[0])

    for name in GATE_FRAME_KEYS:
        value = arrays[name]
        if value.ndim == 0 or int(value.shape[0]) != count:
            raise ValueError(
                f"source {source_index} {name} must contain {count} frame rows"
            )
    for name in GATE_SCALAR_KEYS:
        if arrays[name].shape != ():
            raise ValueError(f"source {source_index} {name} must be scalar")

    if _scalar_string(
        arrays["observation_action_alignment"],
        name="observation_action_alignment",
    ) != "pre_action":
        raise ValueError("observation_action_alignment must be 'pre_action'")
    generation_seed = arrays["task_generation_seed"]
    if (
        generation_seed.dtype.kind not in "iu"
        or generation_seed.dtype == np.bool_
    ):
        raise TypeError("task_generation_seed must be a scalar integer")

    for name in ("obs", "action"):
        value = arrays[name]
        if value.ndim != 2 or value.shape[1] <= 0 or value.dtype.kind != "f":
            raise TypeError(f"{name} must be a nonempty floating [N,D] array")
        if not bool(np.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")

    q_true = arrays["q_true"]
    q_nominal = arrays["q_nominal"]
    if (
        q_true.ndim != 3
        or q_true.shape[0] != count
        or q_true.shape[-1] != 2
        or q_nominal.shape != q_true.shape
        or q_true.shape[1] <= 0
    ):
        raise ValueError("q_true and q_nominal must match nonempty [N,K,2]")
    anchor_count = int(q_true.shape[1])
    if arrays["keypoints_xyz"].shape != (count, anchor_count, 3):
        raise ValueError("keypoints_xyz must have shape [N,K,3]")
    for name in ("q_true", "q_nominal", "keypoints_xyz"):
        value = arrays[name]
        if value.dtype.kind != "f" or not bool(np.isfinite(value).all()):
            raise TypeError(f"{name} must be a finite floating array")
    for name in ("valid_true", "valid_nominal"):
        value = arrays[name]
        if value.shape != (count, anchor_count) or value.dtype != np.bool_:
            raise TypeError(f"{name} must be boolean [N,K]")

    for name in ("task_index", "env_seed", "frame_id", "camera_step_frame"):
        value = arrays[name]
        if value.dtype.kind not in "iu" or value.dtype == np.bool_:
            raise TypeError(f"{name} must use integer storage")
    for name in ("task_name", "episode_id"):
        value = arrays[name].astype(str)
        if value.shape != (count,) or bool(np.any(value == "")):
            raise ValueError(f"{name} must contain {count} non-empty values")

    return arrays, np.array(clean, copy=True, order="C"), _source_mode(source, count)


def _same_array(left: np.ndarray, right: np.ndarray) -> bool:
    return left.dtype == right.dtype and np.array_equal(left, right)


def _combine_preserved(
    sources: Sequence[dict[str, np.ndarray]],
    counts: Sequence[int],
) -> dict[str, np.ndarray]:
    common = set(sources[0])
    union = set(sources[0])
    for source in sources[1:]:
        common.intersection_update(source)
        union.update(source)
    asymmetric = sorted((union - common) - {"rgb_camera_only"})
    if asymmetric:
        raise KeyError(
            "optional source fields must be present in every input to preserve "
            f"them safely: {asymmetric}"
        )

    output: dict[str, np.ndarray] = {}
    for name in sorted(common - _REPLACED_KEYS):
        values = [source[name] for source in sources]
        first = values[0]
        is_frame = first.ndim > 0 and int(first.shape[0]) == counts[0]
        if is_frame:
            for source_index, (value, count) in enumerate(
                zip(values, counts, strict=True)
            ):
                if (
                    value.ndim == 0
                    or int(value.shape[0]) != count
                    or value.shape[1:] != first.shape[1:]
                    or value.dtype != first.dtype
                ):
                    raise ValueError(
                        f"field {name} is not consistently frame-aligned at "
                        f"source {source_index}"
                    )
            output[name] = np.concatenate(values, axis=0)
            continue

        if first.shape == ():
            if all(_same_array(first, value) for value in values[1:]):
                output[name] = np.array(first, copy=True)
                continue
            if name not in GATE_SCALAR_KEYS and all(
                value.shape == () for value in values
            ):
                output[name] = np.concatenate(
                    [
                        np.full(count, value.item(), dtype=value.dtype)
                        for value, count in zip(values, counts, strict=True)
                    ]
                )
                continue
            raise ValueError(f"gate scalar field {name} differs across input sources")
        if any(
            value.shape != first.shape or value.dtype != first.dtype
            for value in values
        ):
            raise ValueError(
                f"non-frame field {name} has incompatible source storage"
            )
        if all(_same_array(first, value) for value in values[1:]):
            output[name] = np.array(first, copy=True)
            continue
        raise ValueError(f"non-frame field {name} differs across input sources")
    return output


def _identity_components(
    arrays: Mapping[str, np.ndarray], row: int
) -> tuple[str, str, int, int]:
    return (
        str(arrays["task_name"][row]),
        str(arrays["source_episode_id"][row]),
        int(arrays["task_index"][row]),
        int(arrays["env_seed"][row]),
    )


def _episode_seed(
    global_seed: int, identity: tuple[str, str, int, int]
) -> np.random.SeedSequence:
    payload = json.dumps(
        [global_seed, identity[0], identity[1], identity[2], identity[3]],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    entropy = np.frombuffer(digest, dtype="<u4").astype(np.uint32).tolist()
    return np.random.SeedSequence(entropy)


def _canonical_order(
    arrays: Mapping[str, np.ndarray],
    source_ids: np.ndarray,
) -> tuple[np.ndarray, list[tuple[tuple[str, str, int, int], np.ndarray]]]:
    groups: dict[tuple[str, str, int, int], list[int]] = {}
    source_owners: dict[tuple[str, str, int, int], int] = {}
    count = int(arrays["frame_id"].shape[0])
    for row in range(count):
        identity = _identity_components(arrays, row)
        owner = int(source_ids[row])
        previous_owner = source_owners.setdefault(identity, owner)
        if previous_owner != owner:
            raise ValueError(
                f"source physical identities must be disjoint; duplicate {identity}"
            )
        groups.setdefault(identity, []).append(row)

    ordered_groups: list[tuple[tuple[str, str, int, int], np.ndarray]] = []
    seen_frames: set[tuple[str, str, int]] = set()
    for identity in sorted(groups):
        rows = np.asarray(groups[identity], dtype=np.int64)
        frames = arrays["frame_id"][rows].astype(np.int64, copy=False)
        order = np.argsort(frames, kind="stable")
        rows = rows[order]
        frames = frames[order]
        if frames.size > 1 and not bool(np.all(np.diff(frames) == 1)):
            raise ValueError(
                f"physical episode {identity} must have unique contiguous frame IDs"
            )
        output_episode = str(arrays["episode_id"][int(rows[0])])
        for frame in frames.tolist():
            gate_identity = (identity[0], output_episode, int(frame))
            if gate_identity in seen_frames:
                raise ValueError(
                    f"duplicate task/episode/frame identity {gate_identity}"
                )
            seen_frames.add(gate_identity)
        ordered_groups.append((identity, rows))
    canonical = np.concatenate([rows for _, rows in ordered_groups])
    return canonical, ordered_groups


def _severity_parameters(
    family: str,
    tier: int,
    image_shape: tuple[int, int],
) -> dict[str, object]:
    height, width = image_shape
    scale = (0.06, 0.10, 0.16)[tier - 1]
    radius = max(tier, int(round(min(height, width) * scale)))
    if family == "square_occlusion":
        return {"radius_px": radius}
    if family == "irregular_occlusion":
        return {"radius_px": radius, "vertex_count": 6 + 2 * tier}
    if family == "texture_distractor":
        return {
            "radius_px": radius,
            "tile_size_px": max(1, radius // 3),
            "opacity": (0.55, 0.75, 0.95)[tier - 1],
        }
    if family == "local_gaussian_blur":
        return {
            "radius_px": radius,
            "sigma": (0.8, 1.4, 2.2)[tier - 1],
        }
    raise ValueError(f"unsupported corruption family {family!r}")


def _event_choices(
    ordered: np.ndarray,
    usable_anchor: np.ndarray,
    *,
    minimum_duration: int,
    maximum_duration: int,
    recovery_length: int,
    onset_phase_min: float,
    onset_phase_max: float,
) -> list[tuple[int, int, int]]:
    frame_count = int(ordered.size)
    denominator = max(frame_count - 1, 1)
    choices: list[tuple[int, int, int]] = []
    for onset in range(frame_count):
        relative_phase = onset / denominator
        if not onset_phase_min <= relative_phase <= onset_phase_max:
            continue
        largest_duration = min(
            maximum_duration,
            frame_count - onset - recovery_length,
        )
        for duration in range(minimum_duration, largest_duration + 1, 5):
            window = ordered[onset : onset + duration + recovery_length]
            anchors = np.flatnonzero(np.all(usable_anchor[window], axis=0))
            choices.extend(
                (onset, duration, int(anchor)) for anchor in anchors.tolist()
            )
    return choices


def build_posterior_randomized_arrays(
    sources: Sequence[ArraySource] | ArraySource,
    *,
    seed: int = 0,
    minimum_duration: int = 10,
    maximum_duration: int = 20,
    recovery_length: int = 10,
    onset_phase_min: float = 0.2,
    onset_phase_max: float = 0.7,
) -> dict[str, np.ndarray]:
    """Return one gate-compatible randomized dataset from one or more sources."""

    resolved_seed = _integer(seed, name="seed", minimum=0)
    minimum = _integer(minimum_duration, name="minimum_duration", minimum=10)
    maximum = _integer(maximum_duration, name="maximum_duration", minimum=10)
    recovery = _integer(recovery_length, name="recovery_length", minimum=10)
    phase_min = _phase(onset_phase_min, name="onset_phase_min")
    phase_max = _phase(onset_phase_max, name="onset_phase_max")
    if maximum < minimum:
        raise ValueError("maximum_duration must be at least minimum_duration")
    if minimum % 5 or maximum % 5:
        raise ValueError("duration bounds must be exact multiples of 5")
    if recovery % 5:
        raise ValueError("recovery_length must be an exact multiple of 5")
    if phase_max < phase_min:
        raise ValueError("onset_phase_max must be at least onset_phase_min")

    if isinstance(sources, Mapping):
        source_list = [sources]
    else:
        source_list = list(sources)
    if not source_list:
        raise ValueError("at least one input source is required")

    validated: list[dict[str, np.ndarray]] = []
    clean_parts: list[np.ndarray] = []
    mode_parts: list[np.ndarray] = []
    usable_parts: list[np.ndarray] = []
    counts: list[int] = []
    for source_index, source in enumerate(source_list):
        arrays, clean, mode = _validate_source(source, source_index=source_index)
        q_true = arrays["q_true"]
        height, width = clean.shape[1:3]
        usable = (
            arrays["valid_true"]
            & arrays["valid_nominal"]
            & np.isfinite(q_true).all(axis=-1)
            & (q_true[..., 0] >= 0.0)
            & (q_true[..., 0] < width)
            & (q_true[..., 1] >= 0.0)
            & (q_true[..., 1] < height)
        )
        validated.append(arrays)
        clean_parts.append(clean)
        mode_parts.append(mode)
        usable_parts.append(usable)
        counts.append(int(clean.shape[0]))

    image_shapes = {tuple(part.shape[1:]) for part in clean_parts}
    if len(image_shapes) != 1:
        raise ValueError("all source RGB arrays must have the same image shape")
    anchor_counts = {int(part.shape[1]) for part in usable_parts}
    if len(anchor_counts) != 1:
        raise ValueError("all sources must have the same anchor count")

    output = _combine_preserved(validated, counts)
    source_episode_id = np.array(output["episode_id"], copy=True)
    stable_episode_ids = np.asarray(
        [
            json.dumps(
                [
                    str(output["task_name"][row]),
                    str(source_episode_id[row]),
                    int(output["task_index"][row]),
                    int(output["env_seed"][row]),
                ],
                ensure_ascii=True,
                separators=(",", ":"),
            )
            for row in range(int(source_episode_id.shape[0]))
        ]
    )
    output["source_episode_id"] = source_episode_id
    output["episode_id"] = stable_episode_ids
    clean = np.concatenate(clean_parts, axis=0)
    source_mode = np.concatenate(mode_parts, axis=0)
    usable_anchor = np.concatenate(usable_parts, axis=0)
    source_ids = np.concatenate(
        [np.full(count, index, dtype=np.int32) for index, count in enumerate(counts)]
    )
    canonical, source_groups = _canonical_order(output, source_ids)
    for name, value in list(output.items()):
        if value.ndim > 0 and value.shape[0] == canonical.size:
            output[name] = value[canonical]
    clean = clean[canonical]
    source_mode = source_mode[canonical]
    usable_anchor = usable_anchor[canonical]

    remapped_groups: list[tuple[tuple[str, str, int, int], np.ndarray]] = []
    cursor = 0
    for identity, old_rows in source_groups:
        size = int(old_rows.size)
        remapped_groups.append(
            (identity, np.arange(cursor, cursor + size, dtype=np.int64))
        )
        cursor += size

    count = int(clean.shape[0])
    anchor_count = int(usable_anchor.shape[1])
    corrupted = np.array(clean, copy=True, order="C")
    active = np.zeros((count, anchor_count), dtype=np.bool_)
    recovery_onset = np.zeros_like(active)
    recovery_segment = np.zeros_like(active)
    family_width = max(len(name) for name in CORRUPTION_TYPES)
    family_annotation = np.empty(count, dtype=f"<U{family_width}")
    anchor_annotation = np.empty(count, dtype=np.int32)
    onset_annotation = np.empty(count, dtype=output["frame_id"].dtype)
    duration_annotation = np.empty(count, dtype=np.int32)
    severity_annotation = np.empty(count, dtype=np.int8)
    phase_annotation = np.empty(count, dtype=np.float64)

    for identity, ordered in remapped_groups:
        choices = _event_choices(
            ordered,
            usable_anchor,
            minimum_duration=minimum,
            maximum_duration=maximum,
            recovery_length=recovery,
            onset_phase_min=phase_min,
            onset_phase_max=phase_max,
        )
        if not choices:
            raise ValueError(
                f"physical episode {identity} has no valid in-bounds anchor/event "
                "window for the requested duration, recovery, and onset phase bounds"
            )
        rng = np.random.default_rng(_episode_seed(resolved_seed, identity))
        onset, duration, anchor = choices[int(rng.integers(0, len(choices)))]
        family = CORRUPTION_TYPES[int(rng.integers(0, len(CORRUPTION_TYPES)))]
        tier = SEVERITY_TIERS[int(rng.integers(0, len(SEVERITY_TIERS)))]
        event_seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
        active_rows = ordered[onset : onset + duration]
        recovery_rows = ordered[
            onset + duration : onset + duration + recovery
        ]
        active[active_rows, anchor] = True
        recovery_onset[int(recovery_rows[0]), anchor] = True
        recovery_segment[recovery_rows, anchor] = True

        parameters = _severity_parameters(
            family,
            tier,
            (int(clean.shape[1]), int(clean.shape[2])),
        )
        for row in active_rows.tolist():
            frame_rng = np.random.default_rng(event_seed)
            image, _, _ = apply_visual_trust_corruption(
                clean[row],
                output["q_true"][row, anchor],
                frame_rng,
                corruption_type=family,
                **parameters,
            )
            if np.array_equal(image, clean[row]):
                frame = int(output["frame_id"][row])
                raise ValueError(
                    f"corruption produced no pixel change for physical episode "
                    f"{identity}, frame {frame}, family {family}, severity {tier}"
                )
            corrupted[row] = image

        onset_frame = output["frame_id"][int(ordered[onset])]
        relative_phase = onset / max(int(ordered.size) - 1, 1)
        family_annotation[ordered] = family
        anchor_annotation[ordered] = anchor
        onset_annotation[ordered] = onset_frame
        duration_annotation[ordered] = duration
        severity_annotation[ordered] = tier
        phase_annotation[ordered] = relative_phase

    output.update(
        {
            "rgb": corrupted,
            "rgb_camera_only": clean,
            "anchor_dropout_active": active,
            "anchor_recovery_onset": recovery_onset,
            "anchor_recovery_segment": recovery_segment,
            "measurement_valid": output["valid_nominal"] & ~active,
            "randomized_corruption_family": family_annotation,
            "randomized_anchor_index": anchor_annotation,
            "randomized_onset_frame": onset_annotation,
            "randomized_duration": duration_annotation,
            "randomized_severity_tier": severity_annotation,
            "randomized_onset_relative_phase": phase_annotation,
            "randomized_source_trajectory_mode": source_mode,
        }
    )
    inactive = ~np.any(active, axis=1)
    if not np.array_equal(output["rgb"][inactive], clean[inactive]):
        raise AssertionError("inactive RGB frames must exactly match rgb_camera_only")
    changed = np.any(output["rgb"] != clean, axis=(1, 2, 3))
    if not bool(np.all(changed[np.any(active, axis=1)])):
        raise AssertionError("every active RGB frame must differ from rgb_camera_only")
    return output


def load_npz_source(path: str | Path) -> dict[str, np.ndarray]:
    """Load a non-pickle NPZ source into owned arrays."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"input dataset does not exist: {source_path}")
    with np.load(source_path, allow_pickle=False) as raw:
        return {name: np.array(raw[name], copy=True) for name in raw.files}


__all__ = [
    "GENERATION_ANNOTATION_KEYS",
    "GATE_FRAME_KEYS",
    "GATE_SCALAR_KEYS",
    "SCHEDULE_KEYS",
    "SEVERITY_TIERS",
    "build_posterior_randomized_arrays",
    "load_npz_source",
]
