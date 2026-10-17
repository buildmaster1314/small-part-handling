"""Observation selection utilities for fair visuomotor policy evaluation.

MetaWorld's default 39-D state observation contains object and goal state in
addition to robot state.  FCPG's policy experiments must not consume those
privileged values when evaluating visual grounding.
"""

from __future__ import annotations

import numpy as np


ROBOT_PROPRIO_DIM = 4
TASK_GOAL_DIM = 3
OBSERVATION_MODES = ("full", "robot", "robot_goal")


def apply_keypoint_square_occlusion(
    image: np.ndarray,
    points_xy: np.ndarray,
    radius_px: int,
    value: int = 0,
) -> np.ndarray:
    """Mask square image regions centered on valid keypoint projections."""

    if radius_px < 0:
        raise ValueError("radius_px must be non-negative")
    if not 0 <= value <= 255:
        raise ValueError("value must be in [0, 255]")
    result = np.asarray(image).copy()
    if radius_px == 0:
        return result
    if result.ndim != 3:
        raise ValueError(f"expected HWC image, got shape {result.shape}")
    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError(f"expected points with shape [N, 2], got {points.shape}")
    height, width = result.shape[:2]
    for point in points:
        if not bool(np.isfinite(point).all()):
            continue
        center_x, center_y = (int(round(float(coordinate))) for coordinate in point)
        x0 = max(0, center_x - radius_px)
        x1 = min(width, center_x + radius_px + 1)
        y0 = max(0, center_y - radius_px)
        y1 = min(height, center_y + radius_px + 1)
        if x0 < x1 and y0 < y1:
            result[y0:y1, x0:x1] = value
    return result


def select_policy_observation(observation: np.ndarray, mode: str) -> np.ndarray:
    """Return a policy-safe subset of the MetaWorld observation.

    MetaWorld places end-effector XYZ and gripper opening in the first four
    entries of its current observation and the task goal in the final three.
    ``robot`` intentionally excludes object state, the previous frame, and
    the goal. ``robot_goal`` additionally exposes only the deployable task
    command, while continuing to exclude object and previous-frame state.
    """

    values = np.asarray(observation, dtype=np.float32)
    if values.shape[-1] < ROBOT_PROPRIO_DIM:
        raise ValueError(
            f"expected at least {ROBOT_PROPRIO_DIM} observation values, got shape {values.shape}"
        )
    if mode == "full":
        return values
    if mode == "robot":
        return values[..., :ROBOT_PROPRIO_DIM]
    if mode == "robot_goal":
        if values.shape[-1] < ROBOT_PROPRIO_DIM + TASK_GOAL_DIM:
            raise ValueError(
                "robot_goal requires four robot values and a three-value task goal"
            )
        return np.concatenate(
            (
                values[..., :ROBOT_PROPRIO_DIM],
                values[..., -TASK_GOAL_DIM:],
            ),
            axis=-1,
        )
    raise ValueError(f"unknown observation mode {mode!r}; expected one of {OBSERVATION_MODES}")
