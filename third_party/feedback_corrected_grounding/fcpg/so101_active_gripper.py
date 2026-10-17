"""Active column-camera gripper localization from a short open/close pulse."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


DEFAULT_MAXIMUM_LIVE_ROTATION_DEG = 60.0


@dataclass(frozen=True)
class ActiveGripperLocalization:
    center_px: tuple[float, float]
    open_tip_px: tuple[float, float]
    moving_tip_track_candidates_px: tuple[
        tuple[tuple[float, float], tuple[float, float]],
        tuple[tuple[float, float], tuple[float, float]],
    ]
    closed_frame_index: int
    open_frame_index: int
    opening_span: float
    max_arm_motion: float
    moving_area_px: int
    bbox_xyxy: tuple[int, int, int, int]
    contrast: float
    elongation: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def next_gripper_position(current: float, target: float, maximum_step: float) -> float:
    """Move one bounded normalized-position step toward a gripper target."""

    values = np.asarray([current, target, maximum_step], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("gripper positions and maximum step must be finite")
    if not 0.0 <= current <= 100.0 or not 0.0 <= target <= 100.0:
        raise ValueError("gripper positions must lie in [0,100]")
    if maximum_step <= 0.0:
        raise ValueError("maximum gripper step must be positive")
    delta = float(np.clip(target - current, -maximum_step, maximum_step))
    return float(current + delta)


def _validate_pulse(frames: np.ndarray, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    images = np.asarray(frames)
    joints = np.asarray(states, dtype=np.float64)
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError("frames must be uint8 RGB [T,H,W,3]")
    if joints.shape != (len(images), 6) or not np.all(np.isfinite(joints)):
        raise ValueError("states must be finite [T,6] and align with frames")
    if len(images) < 5:
        raise ValueError("active localization requires at least five frames")
    return images, joints


def _component_elongation(component_xy: np.ndarray) -> float:
    centered = component_xy - component_xy.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return float(
        np.sqrt(max(float(eigenvalues[-1]), 1.0e-9) / max(float(eigenvalues[0]), 1.0e-9))
    )


def _principal_endpoint_centers(
    component_xy: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Estimate both physical ends of one elongated signed-difference region."""

    if len(component_xy) < 8:
        raise ValueError("signed jaw region is too small for endpoint estimation")
    center = component_xy.mean(axis=0)
    centered = component_xy - center[None]
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    minor_axis = eigenvectors[:, 0]
    major_axis = eigenvectors[:, -1]
    major = centered @ major_axis
    minor = centered @ minor_axis
    low, high = np.quantile(major, [0.02, 0.98])
    span = max(float(high - low), 1.0)
    endpoints = []
    for endpoint, band in (
        (low, major <= low + 0.12 * span),
        (high, major >= high - 0.12 * span),
    ):
        band_minor = minor[band]
        transverse = float(np.median(band_minor)) if len(band_minor) else 0.0
        point = (
            center
            + float(endpoint) * major_axis
            + transverse * minor_axis
        )
        endpoints.append((float(point[0]), float(point[1])))
    return endpoints[0], endpoints[1]


def _moving_tip_track_candidates(
    signed_difference: np.ndarray,
    component_mask: np.ndarray,
    threshold: float,
) -> tuple[
    tuple[tuple[float, float], tuple[float, float]],
    tuple[tuple[float, float], tuple[float, float]],
]:
    """Return candidate tracks for the same moving-jaw point in open/closed frames."""

    signed = np.asarray(signed_difference, dtype=np.float32)
    selected = np.asarray(component_mask, dtype=bool)
    if signed.shape != selected.shape:
        raise ValueError("signed difference and component mask must align")
    # signed = open - closed. The dark jaw is negative where it exists only in
    # the open frame and positive where it exists only in the closed frame.
    regions = (
        selected & (signed <= -threshold),
        selected & (signed >= threshold),
    )
    endpoint_sets = []
    for region in regions:
        rows, columns = np.nonzero(region)
        points = np.column_stack((columns, rows)).astype(np.float64)
        endpoint_sets.append(_principal_endpoint_centers(points))
    first = np.asarray(endpoint_sets[0], dtype=np.float64)
    second = np.asarray(endpoint_sets[1], dtype=np.float64)
    aligned_cost = float(np.linalg.norm(first[0] - second[0])) + float(
        np.linalg.norm(first[1] - second[1])
    )
    reversed_cost = float(np.linalg.norm(first[0] - second[1])) + float(
        np.linalg.norm(first[1] - second[0])
    )
    if reversed_cost < aligned_cost:
        second = second[::-1]
    pairs = [(first[index], second[index]) for index in range(2)]
    pairs.sort(
        key=lambda pair: float(np.linalg.norm(pair[0] - pair[1])), reverse=True
    )
    return tuple(tuple(point.tolist()) for point in pairs[0]), tuple(
        tuple(point.tolist()) for point in pairs[1]
    )


def fit_point_track_similarity_transform(
    nominal_point_track_px: np.ndarray,
    observed_point_track_px: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Fit the exact 2-D similarity mapping from one two-frame point track."""

    nominal = np.asarray(nominal_point_track_px, dtype=np.float64)
    observed = np.asarray(observed_point_track_px, dtype=np.float64)
    if nominal.shape != (2, 2) or observed.shape != (2, 2):
        raise ValueError("point tracks must both have shape [2,2]")
    if not np.all(np.isfinite(nominal)) or not np.all(np.isfinite(observed)):
        raise ValueError("point tracks must be finite")
    source = nominal[1] - nominal[0]
    target = observed[1] - observed[0]
    source_length = float(np.linalg.norm(source))
    target_length = float(np.linalg.norm(target))
    if source_length < 1.0 or target_length < 1.0:
        raise ValueError("point tracks must have at least one pixel of displacement")

    source_angle = float(np.arctan2(source[1], source[0]))
    target_angle = float(np.arctan2(target[1], target[0]))
    angle_delta = target_angle - source_angle
    rotation = float(np.arctan2(np.sin(angle_delta), np.cos(angle_delta)))
    scale = target_length / source_length
    cosine = float(np.cos(rotation))
    sine = float(np.sin(rotation))
    linear = scale * np.asarray([[cosine, -sine], [sine, cosine]])
    translation = observed[0] - linear @ nominal[0]
    transform = np.concatenate((linear, translation[:, None]), axis=1)
    return transform.astype(np.float32), float(scale), float(np.degrees(rotation))


def apply_live_similarity_transform(
    q_nominal_px: np.ndarray,
    transform_px: np.ndarray,
) -> np.ndarray:
    """Apply one image-plane similarity transform to any anchor array."""

    nominal = np.asarray(q_nominal_px, dtype=np.float64)
    transform = np.asarray(transform_px, dtype=np.float64)
    if nominal.ndim < 2 or nominal.shape[-1] != 2:
        raise ValueError("q_nominal_px must end in an XY coordinate")
    if transform.shape != (2, 3) or not np.all(np.isfinite(transform)):
        raise ValueError("transform_px must be one finite 2x3 matrix")
    linear = transform[:, :2]
    translation = transform[:, 2]
    return (nominal @ linear.T + translation).astype(np.float32)


def invert_live_similarity_transform(
    q_live_px: np.ndarray,
    transform_px: np.ndarray,
    *,
    max_condition_number: float = 1.0e4,
) -> np.ndarray:
    """Map live-camera pixels back into the nominal calibration image frame."""

    live = np.asarray(q_live_px, dtype=np.float64)
    transform = np.asarray(transform_px, dtype=np.float64)
    if live.ndim < 1 or live.shape[-1] != 2:
        raise ValueError("q_live_px must end in an XY coordinate")
    if not np.all(np.isfinite(live)):
        raise ValueError("q_live_px must be finite")
    if transform.shape != (2, 3) or not np.all(np.isfinite(transform)):
        raise ValueError("transform_px must be one finite 2x3 matrix")
    if max_condition_number <= 1.0 or not np.isfinite(max_condition_number):
        raise ValueError("max_condition_number must be finite and greater than one")
    linear = transform[:, :2]
    condition = float(np.linalg.cond(linear))
    if not np.isfinite(condition) or condition > max_condition_number:
        raise ValueError(
            "live similarity transform is not safely invertible: "
            f"condition_number={condition:.3g}"
        )
    translation = transform[:, 2]
    flat = live.reshape(-1, 2)
    nominal = np.linalg.solve(linear, (flat - translation[None]).T).T
    return nominal.reshape(live.shape).astype(np.float32)


def select_live_moving_tip_similarity_transform(
    nominal_anchors_px: np.ndarray,
    moving_tip_track_candidates_px: np.ndarray,
    *,
    center_anchor_index: int = 0,
    fixed_tip_anchor_index: int = 1,
    moving_tip_anchor_index: int = 2,
    minimum_scale: float = 0.5,
    maximum_scale: float = 1.5,
    maximum_rotation_deg: float = DEFAULT_MAXIMUM_LIVE_ROTATION_DEG,
    maximum_center_offset_px: float = 35.0,
) -> dict[str, Any]:
    """Select the physically local image similarity fit for an open/close pulse."""

    nominal = np.asarray(nominal_anchors_px, dtype=np.float32)
    candidates = np.asarray(moving_tip_track_candidates_px, dtype=np.float32)
    if nominal.shape != (2, 3, 2) or not np.all(np.isfinite(nominal)):
        raise ValueError("nominal calibration anchors must be finite [2,3,2]")
    if (
        candidates.ndim != 3
        or candidates.shape[1:] != (2, 2)
        or len(candidates) == 0
        or not np.all(np.isfinite(candidates))
    ):
        raise ValueError("moving-tip track candidates must be finite [N,2,2]")
    indices = (center_anchor_index, fixed_tip_anchor_index, moving_tip_anchor_index)
    if len(set(indices)) != 3 or any(index not in range(3) for index in indices):
        raise ValueError("center, fixed-tip, and moving-tip indices must be distinct")
    bounds = np.asarray(
        [
            minimum_scale,
            maximum_scale,
            maximum_rotation_deg,
            maximum_center_offset_px,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(bounds)) or np.any(bounds <= 0.0):
        raise ValueError("similarity calibration bounds must be finite and positive")
    if minimum_scale >= maximum_scale:
        raise ValueError("minimum_scale must be below maximum_scale")

    nominal_track = nominal[:, moving_tip_anchor_index]
    valid_fits: list[tuple[float, dict[str, Any]]] = []
    rejected: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        transform, scale, rotation_deg = fit_point_track_similarity_transform(
            nominal_track, candidate
        )
        transformed = apply_live_similarity_transform(nominal, transform)
        center_offsets = (
            transformed[:, center_anchor_index] - nominal[:, center_anchor_index]
        )
        center_distances = np.linalg.norm(center_offsets, axis=1)
        maximum_center_distance = float(np.max(center_distances))
        score = (
            float(np.mean(center_distances))
            + 0.2 * abs(rotation_deg)
            + 15.0 * abs(float(np.log(scale)))
        )
        row = {
            "candidate_index": candidate_index,
            "observed_moving_tip_track_px": candidate.copy(),
            "nominal_moving_tip_track_px": nominal_track.copy(),
            "transform_px": transform.copy(),
            "scale": scale,
            "rotation_deg": rotation_deg,
            "center_offsets_px": center_offsets.copy(),
            "maximum_center_offset_norm_px": maximum_center_distance,
            "transformed_anchors_px": transformed.copy(),
            "selection_score": score,
        }
        admissible = (
            minimum_scale <= scale <= maximum_scale
            and abs(rotation_deg) <= maximum_rotation_deg
            and maximum_center_distance <= maximum_center_offset_px
        )
        if admissible:
            valid_fits.append((score, row))
        else:
            rejected.append(row)
    if not valid_fits:
        diagnostics = [
            {
                "candidate_index": row["candidate_index"],
                "scale": row["scale"],
                "rotation_deg": row["rotation_deg"],
                "maximum_center_offset_norm_px": row[
                    "maximum_center_offset_norm_px"
                ],
            }
            for row in rejected
        ]
        raise ValueError(
            "no automatic moving-tip track fit passed the calibration bounds: "
            f"{diagnostics}"
        )

    _, selected = min(valid_fits, key=lambda item: item[0])
    return {
        **selected,
        "nominal_anchors_px": nominal.copy(),
        "center_anchor_index": center_anchor_index,
        "fixed_tip_anchor_index": fixed_tip_anchor_index,
        "moving_tip_anchor_index": moving_tip_anchor_index,
        "rejected_candidate_count": len(rejected),
    }


def localize_gripper_from_pulse(
    frames: np.ndarray,
    states: np.ndarray,
    *,
    minimum_opening_span: float = 20.0,
    maximum_arm_motion: float = 2.0,
    minimum_component_area: int = 24,
    maximum_component_fraction: float = 0.18,
) -> ActiveGripperLocalization:
    """Locate the closed moving fingertip while the five arm joints stay fixed.

    Only image changes caused by the commanded gripper aperture are used.  The
    scene background and target may be anywhere, but they must remain still
    during the short pulse.
    """

    images, joints = _validate_pulse(frames, states)
    opening = joints[:, 5]
    closed_index = int(np.argmin(opening))
    open_index = int(np.argmax(opening))
    opening_span = float(opening[open_index] - opening[closed_index])
    if opening_span < minimum_opening_span:
        raise ValueError(
            f"gripper opening span {opening_span:.3f} is below {minimum_opening_span:.3f}"
        )
    arm_motion = float(np.max(np.ptp(joints[:, :5], axis=0)))
    if arm_motion > maximum_arm_motion:
        raise ValueError(
            f"arm moved {arm_motion:.3f} during gripper pulse; maximum is {maximum_arm_motion:.3f}"
        )

    open_gray = cv2.cvtColor(images[open_index], cv2.COLOR_RGB2GRAY).astype(np.float32)
    closed_gray = cv2.cvtColor(images[closed_index], cv2.COLOR_RGB2GRAY).astype(np.float32)
    open_gray = cv2.GaussianBlur(open_gray, (5, 5), 0)
    closed_gray = cv2.GaussianBlur(closed_gray, (5, 5), 0)
    signed = open_gray - closed_gray
    absolute = np.abs(signed)
    threshold = max(12.0, float(np.quantile(absolute, 0.96)))
    mask = (absolute >= threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    image_area = int(images.shape[1] * images.shape[2])
    maximum_area = int(round(maximum_component_fraction * image_area))
    candidates: list[tuple[float, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_component_area or area > maximum_area:
            continue
        component = labels == label
        contrast = float(np.mean(absolute[component]))
        candidates.append((area * contrast, label))
    if not candidates:
        raise ValueError("no reliable moving-finger component was found")
    _, selected = max(candidates)
    rows, columns = np.nonzero(labels == selected)
    component_xy = np.column_stack((columns, rows)).astype(np.float64)
    flow = cv2.calcOpticalFlowFarneback(
        open_gray.astype(np.uint8),
        closed_gray.astype(np.uint8),
        None,
        0.5,
        5,
        21,
        5,
        7,
        1.5,
        0,
    )
    magnitude = np.linalg.norm(flow, axis=-1)
    component_magnitude = magnitude[rows, columns]
    moving = component_magnitude >= np.quantile(component_magnitude, 0.85)
    sources = component_xy[moving]
    destinations = sources + flow[rows[moving], columns[moving]]
    open_tip = np.median(sources, axis=0)
    closed_tip = np.median(destinations, axis=0)
    elongation = _component_elongation(component_xy)
    moving_tip_track_candidates = _moving_tip_track_candidates(
        signed,
        labels == selected,
        threshold,
    )
    x, y, width, height, area = stats[selected].tolist()
    contrast = float(np.mean(absolute[labels == selected]))
    return ActiveGripperLocalization(
        # ``center_px`` is retained for existing offline artifacts. It is the
        # closed moving fingertip, not the midpoint between both jaws.
        center_px=(float(closed_tip[0]), float(closed_tip[1])),
        open_tip_px=(float(open_tip[0]), float(open_tip[1])),
        moving_tip_track_candidates_px=moving_tip_track_candidates,
        closed_frame_index=closed_index,
        open_frame_index=open_index,
        opening_span=opening_span,
        max_arm_motion=arm_motion,
        moving_area_px=int(area),
        bbox_xyxy=(int(x), int(y), int(x + width), int(y + height)),
        contrast=contrast,
        elongation=elongation,
    )


def apply_live_center_translation(
    q_nominal_px: np.ndarray,
    live_center_px: np.ndarray,
    *,
    anchor_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate all anchor priors so one selected anchor matches observation."""

    nominal = np.asarray(q_nominal_px, dtype=np.float64)
    center = np.asarray(live_center_px, dtype=np.float64)
    if nominal.ndim != 3 or nominal.shape[-2:] != (3, 2):
        raise ValueError("q_nominal_px must have shape [T,3,2]")
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("live_center_px must be one finite XY coordinate")
    if anchor_index not in range(nominal.shape[-2]):
        raise ValueError("anchor_index is outside the nominal anchor axis")
    offset = center - nominal[-1, anchor_index]
    return (nominal + offset[None, None, :]).astype(np.float32), offset.astype(np.float32)
