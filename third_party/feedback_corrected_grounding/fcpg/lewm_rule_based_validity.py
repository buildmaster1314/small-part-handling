"""Deterministic visual-validity switch for FCPG/LeWM observers.

The switch deliberately contains no learned parameters.  It accepts only
signals available to a deployed camera and grounding stack, then selects the
current FCPG measurement or the action-conditioned LeWM prior.  Synthetic
corruption schedules and oracle keypoints are not part of the interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


ObserverMode = Literal["measurement", "prior"]


@dataclass(frozen=True, slots=True)
class RuleBasedValidityConfig:
    """Fixed thresholds for a camera-adaptive reliability state machine."""

    patch_radius_px: int = 12
    max_fcpg_residual_px: float = 20.0
    residual_saturation_fraction: float = 0.95
    minimum_sharpness_ratio: float = 0.25
    minimum_contrast_ratio: float = 0.25
    maximum_black_patch_mean: float = 0.03
    frozen_frame_rms: float = 1.0e-4
    moving_action_rms: float = 0.05
    minimum_nominal_motion_px: float = 0.5
    soft_failures_to_prior: int = 2
    healthy_frames_to_measurement: int = 1
    baseline_ema: float = 0.95
    quality_epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.patch_radius_px <= 0:
            raise ValueError("patch_radius_px must be positive")
        if self.max_fcpg_residual_px <= 0.0:
            raise ValueError("max_fcpg_residual_px must be positive")
        if not 0.0 < self.residual_saturation_fraction <= 1.0:
            raise ValueError("residual_saturation_fraction must lie in (0,1]")
        for name, value in (
            ("minimum_sharpness_ratio", self.minimum_sharpness_ratio),
            ("minimum_contrast_ratio", self.minimum_contrast_ratio),
            ("baseline_ema", self.baseline_ema),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0,1)")
        if not 0.0 <= self.maximum_black_patch_mean < 1.0:
            raise ValueError("maximum_black_patch_mean must lie in [0,1)")
        if (
            self.frozen_frame_rms < 0.0
            or self.moving_action_rms < 0.0
            or self.minimum_nominal_motion_px < 0.0
        ):
            raise ValueError("frame, action, and motion thresholds must be non-negative")
        if self.soft_failures_to_prior <= 0:
            raise ValueError("soft_failures_to_prior must be positive")
        if self.healthy_frames_to_measurement <= 0:
            raise ValueError("healthy_frames_to_measurement must be positive")
        if self.quality_epsilon <= 0.0:
            raise ValueError("quality_epsilon must be positive")


@dataclass(frozen=True, slots=True)
class RuleBasedValidityDecision:
    """One causal switch decision and its deployment diagnostics."""

    mode: ObserverMode
    use_prior: bool
    hard_failure: bool
    soft_failure: bool
    reasons: tuple[str, ...]
    minimum_sharpness_ratio: float
    minimum_contrast_ratio: float
    frame_delta_rms: float | None
    maximum_residual_px: float
    maximum_nominal_motion_px: float | None
    maximum_nominal_patch_mean: float | None


def _image_float_hwc(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 3 or 3 not in (value.shape[0], value.shape[-1]):
        raise ValueError("image must have shape [H,W,3] or [3,H,W]")
    if value.shape[0] == 3 and value.shape[-1] != 3:
        value = np.transpose(value, (1, 2, 0))
    if value.shape[-1] != 3 or min(value.shape[:2]) < 3:
        raise ValueError("image must contain a non-empty RGB plane")
    result = value.astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("image must be finite")
    if result.max(initial=0.0) > 1.0:
        result = result / 255.0
    return np.clip(result, 0.0, 1.0)


def _anchors(value: np.ndarray, *, name: str, columns: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape [K,{columns}]")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite numeric values")
    return array.astype(np.float32, copy=False)


def _local_quality(
    image: np.ndarray,
    centers: np.ndarray,
    *,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    gray = (
        0.2989 * image[..., 0]
        + 0.5870 * image[..., 1]
        + 0.1140 * image[..., 2]
    )
    height, width = gray.shape
    sharpness: list[float] = []
    contrast: list[float] = []
    for x_float, y_float in centers.tolist():
        x = int(round(float(x_float)))
        y = int(round(float(y_float)))
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        patch = gray[y0:y1, x0:x1]
        if min(patch.shape, default=0) < 3:
            sharpness.append(0.0)
            contrast.append(0.0)
            continue
        center = patch[1:-1, 1:-1]
        laplacian = (
            patch[:-2, 1:-1]
            + patch[2:, 1:-1]
            + patch[1:-1, :-2]
            + patch[1:-1, 2:]
            - 4.0 * center
        )
        sharpness.append(float(np.sqrt(np.mean(np.square(laplacian)))))
        contrast.append(float(np.std(patch)))
    return np.asarray(sharpness, dtype=np.float32), np.asarray(
        contrast, dtype=np.float32
    )


def _local_patch_means(
    image: np.ndarray,
    centers: np.ndarray,
    *,
    radius: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    means: list[float] = []
    for x_float, y_float in centers.tolist():
        x = int(round(float(x_float)))
        y = int(round(float(y_float)))
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        patch = image[y0:y1, x0:x1]
        means.append(float(np.mean(patch)) if patch.size else 0.0)
    return np.asarray(means, dtype=np.float32)


class RuleBasedVisualValidityFSM:
    """Select FCPG measurement or LeWM prior using causal quality checks."""

    def __init__(self, config: RuleBasedValidityConfig | None = None) -> None:
        self.config = RuleBasedValidityConfig() if config is None else config
        self.mode: ObserverMode = "measurement"
        self._soft_failure_count = 0
        self._healthy_count = 0
        self._sharpness_baseline: np.ndarray | None = None
        self._contrast_baseline: np.ndarray | None = None
        self._previous_image: np.ndarray | None = None
        self._previous_nominal: np.ndarray | None = None

    def reset(self) -> None:
        self.mode = "measurement"
        self._soft_failure_count = 0
        self._healthy_count = 0
        self._sharpness_baseline = None
        self._contrast_baseline = None
        self._previous_image = None
        self._previous_nominal = None

    def _update_baseline(
        self, sharpness: np.ndarray, contrast: np.ndarray
    ) -> None:
        decay = self.config.baseline_ema
        if self._sharpness_baseline is None:
            self._sharpness_baseline = sharpness.copy()
            self._contrast_baseline = contrast.copy()
            return
        assert self._contrast_baseline is not None
        self._sharpness_baseline = (
            decay * self._sharpness_baseline + (1.0 - decay) * sharpness
        )
        self._contrast_baseline = (
            decay * self._contrast_baseline + (1.0 - decay) * contrast
        )

    def update(
        self,
        *,
        image: np.ndarray | None,
        frame_present: bool,
        frame_fresh: bool,
        q_nominal: np.ndarray,
        q_corrected: np.ndarray,
        valid_nominal: np.ndarray,
        executed_action_rms: float,
    ) -> RuleBasedValidityDecision:
        """Consume one camera event without any oracle validity annotation."""

        nominal = _anchors(q_nominal, name="q_nominal", columns=2)
        corrected = _anchors(q_corrected, name="q_corrected", columns=2)
        if corrected.shape != nominal.shape:
            raise ValueError("q_corrected must match q_nominal")
        validity = np.asarray(valid_nominal)
        if validity.shape != (nominal.shape[0],) or validity.dtype != np.bool_:
            raise ValueError("valid_nominal must be a boolean vector with K entries")
        action_rms = float(executed_action_rms)
        if not np.isfinite(action_rms) or action_rms < 0.0:
            raise ValueError("executed_action_rms must be finite and non-negative")

        residual = np.linalg.norm(corrected - nominal, axis=-1)
        maximum_residual = float(np.max(residual))
        hard_reasons: list[str] = []
        soft_reasons: list[str] = []
        if not frame_present:
            hard_reasons.append("frame_missing")
        elif not frame_fresh:
            hard_reasons.append("frame_stale")
        if not bool(np.all(validity)):
            hard_reasons.append("nominal_anchor_invalid")
        image_height: int | None = None
        image_width: int | None = None
        residual_saturated = maximum_residual >= (
            self.config.max_fcpg_residual_px
            * self.config.residual_saturation_fraction
        )

        minimum_sharpness_ratio = 1.0
        minimum_contrast_ratio = 1.0
        frame_delta: float | None = None
        maximum_nominal_motion: float | None = None
        maximum_nominal_patch_mean: float | None = None
        current_image: np.ndarray | None = None
        sharpness: np.ndarray | None = None
        contrast: np.ndarray | None = None
        if image is not None:
            current_image = _image_float_hwc(image)
            image_height, image_width = current_image.shape[:2]
            corrected_valid = (
                (corrected[:, 0] >= 0.0)
                & (corrected[:, 0] <= float(image_width - 1))
                & (corrected[:, 1] >= 0.0)
                & (corrected[:, 1] <= float(image_height - 1))
            )
            if not bool(np.all(corrected_valid)):
                hard_reasons.append("corrected_anchor_out_of_bounds")
            global_center = np.asarray(
                [[0.5 * (image_width - 1), 0.5 * (image_height - 1)]],
                dtype=np.float32,
            )
            sharpness, contrast = _local_quality(
                current_image,
                global_center,
                radius=max(image_height, image_width),
            )
            nominal_patch_means = _local_patch_means(
                current_image,
                nominal,
                radius=self.config.patch_radius_px,
            )
            maximum_nominal_patch_mean = float(np.max(nominal_patch_means))
            if (
                maximum_nominal_patch_mean
                <= self.config.maximum_black_patch_mean
            ):
                hard_reasons.append("local_blackout")
            if self._sharpness_baseline is not None:
                assert self._contrast_baseline is not None
                epsilon = self.config.quality_epsilon
                sharpness_ratio = sharpness / np.maximum(
                    self._sharpness_baseline, epsilon
                )
                contrast_ratio = contrast / np.maximum(
                    self._contrast_baseline, epsilon
                )
                minimum_sharpness_ratio = float(np.min(sharpness_ratio))
                minimum_contrast_ratio = float(np.min(contrast_ratio))
                if minimum_sharpness_ratio < self.config.minimum_sharpness_ratio:
                    soft_reasons.append("global_blur")
                if minimum_contrast_ratio < self.config.minimum_contrast_ratio:
                    soft_reasons.append("global_contrast_loss")
                if residual_saturated and soft_reasons:
                    soft_reasons.append("fcpg_residual_saturated")
            if self._previous_image is not None:
                if self._previous_image.shape != current_image.shape:
                    raise ValueError("camera image shape changed without reset")
                frame_delta = float(
                    np.sqrt(np.mean(np.square(current_image - self._previous_image)))
                )
                if self._previous_nominal is None:
                    raise RuntimeError("previous image exists without nominal anchors")
                maximum_nominal_motion = float(
                    np.max(np.linalg.norm(nominal - self._previous_nominal, axis=-1))
                )
                if (
                    frame_delta <= self.config.frozen_frame_rms
                    and maximum_nominal_motion
                    >= self.config.minimum_nominal_motion_px
                ):
                    hard_reasons.append("frame_frozen_while_moving")
        elif frame_present:
            raise ValueError("a present frame requires image data")

        hard_failure = bool(hard_reasons)
        soft_failure = bool(soft_reasons)
        if hard_failure:
            self.mode = "prior"
            self._soft_failure_count = 0
            self._healthy_count = 0
        elif soft_failure:
            self._soft_failure_count += 1
            self._healthy_count = 0
            if self._soft_failure_count >= self.config.soft_failures_to_prior:
                self.mode = "prior"
        else:
            self._soft_failure_count = 0
            self._healthy_count += 1
            if self._healthy_count >= self.config.healthy_frames_to_measurement:
                self.mode = "measurement"
            if (
                self.mode == "measurement"
                and sharpness is not None
                and contrast is not None
            ):
                self._update_baseline(sharpness, contrast)

        if current_image is not None and frame_fresh:
            self._previous_image = current_image.copy()
            self._previous_nominal = nominal.copy()
        reasons = tuple(hard_reasons + soft_reasons)
        return RuleBasedValidityDecision(
            mode=self.mode,
            use_prior=self.mode == "prior",
            hard_failure=hard_failure,
            soft_failure=soft_failure,
            reasons=reasons,
            minimum_sharpness_ratio=minimum_sharpness_ratio,
            minimum_contrast_ratio=minimum_contrast_ratio,
            frame_delta_rms=frame_delta,
            maximum_residual_px=maximum_residual,
            maximum_nominal_motion_px=maximum_nominal_motion,
            maximum_nominal_patch_mean=maximum_nominal_patch_mean,
        )


__all__ = [
    "ObserverMode",
    "RuleBasedValidityConfig",
    "RuleBasedValidityDecision",
    "RuleBasedVisualValidityFSM",
]
