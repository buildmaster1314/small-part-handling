"""Parametric clock-face camera views for multi-view A4 experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class OrbitCameraView:
    """A fixed camera placed around a task focus using clock-face azimuth."""

    clock_hour: int
    elevation_degrees: float = 35.0
    radius_m: float = 1.1
    base_camera_name: str = "corner"

    def __post_init__(self) -> None:
        if not 1 <= int(self.clock_hour) <= 12:
            raise ValueError("clock_hour must be in [1,12]")
        if not 5.0 <= float(self.elevation_degrees) <= 80.0:
            raise ValueError("camera elevation must be in [5,80] degrees")
        if not 0.3 <= float(self.radius_m) <= 5.0:
            raise ValueError("camera radius must be in [0.3,5.0] metres")
        if not str(self.base_camera_name):
            raise ValueError("base_camera_name must be non-empty")

    @property
    def label(self) -> str:
        elevation = int(round(float(self.elevation_degrees)))
        return f"clock{int(self.clock_hour):02d}_e{elevation:02d}"


ORBIT_TRAIN_VIEWS = tuple(OrbitCameraView(hour) for hour in (12, 2, 4, 6, 8, 10))
ORBIT_DEVELOPMENT_VIEWS = tuple(
    OrbitCameraView(hour) for hour in (1, 3, 5, 7, 9, 11)
)
ORBIT_ALL_VIEWS = tuple(
    sorted(ORBIT_TRAIN_VIEWS + ORBIT_DEVELOPMENT_VIEWS, key=lambda view: view.clock_hour)
)


def orbit_camera_position(focus_world: np.ndarray, view: OrbitCameraView) -> np.ndarray:
    """Return a camera position around ``focus_world``.

    Clock 12 is in front of the robot along negative world Y, clock 3 is on
    positive world X, and the remaining hours follow clockwise order.
    ``radius_m`` is the horizontal radius, while elevation controls height.
    """

    focus = np.asarray(focus_world, dtype=np.float64)
    if focus.shape != (3,) or not np.isfinite(focus).all():
        raise ValueError("focus_world must be a finite 3-vector")
    theta = math.radians(float(view.clock_hour % 12) * 30.0)
    radius = float(view.radius_m)
    offset = np.asarray(
        [
            radius * math.sin(theta),
            -radius * math.cos(theta),
            radius * math.tan(math.radians(float(view.elevation_degrees))),
        ],
        dtype=np.float64,
    )
    return focus + offset


def camera_look_at_quaternion_mujoco(
    position_world: np.ndarray,
    target_world: np.ndarray,
    *,
    up_world: np.ndarray | None = None,
) -> np.ndarray:
    """Return a MuJoCo ``wxyz`` quaternion looking from position to target."""

    import mujoco

    position = np.asarray(position_world, dtype=np.float64)
    target = np.asarray(target_world, dtype=np.float64)
    up = (
        np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        if up_world is None
        else np.asarray(up_world, dtype=np.float64)
    )
    if position.shape != (3,) or target.shape != (3,) or up.shape != (3,):
        raise ValueError("look-at position, target, and up must be 3-vectors")
    if not (
        np.isfinite(position).all()
        and np.isfinite(target).all()
        and np.isfinite(up).all()
    ):
        raise ValueError("look-at inputs must be finite")
    forward = target - position
    forward_norm = float(np.linalg.norm(forward))
    up_norm = float(np.linalg.norm(up))
    if forward_norm <= 1.0e-9 or up_norm <= 1.0e-9:
        raise ValueError("look-at direction and up vector must be non-zero")
    forward /= forward_norm
    up /= up_norm
    right = np.cross(forward, up)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1.0e-9:
        raise ValueError("look-at direction must not be parallel to up")
    right /= right_norm
    camera_up = np.cross(right, forward)
    camera_up /= np.linalg.norm(camera_up)
    rotation = np.column_stack([right, camera_up, -forward])
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(rotation.reshape(-1)))
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def set_fixed_camera_look_at(
    model,
    data,
    camera_id: int,
    position_world: np.ndarray,
    target_world: np.ndarray,
) -> None:
    """Place a world-attached fixed camera and orient it toward a target."""

    import mujoco

    if int(model.cam_bodyid[camera_id]) != 0:
        raise ValueError("look-at placement requires a world-attached fixed camera")
    position = np.asarray(position_world, dtype=np.float64)
    model.cam_pos[camera_id] = position
    model.cam_quat[camera_id] = camera_look_at_quaternion_mujoco(
        position, np.asarray(target_world, dtype=np.float64)
    )
    mujoco.mj_forward(model, data)


def a4_orbit_focus(observation: np.ndarray, target_position: np.ndarray | None) -> np.ndarray:
    """Choose a stable operation-region focus from deployable reset state."""

    values = np.asarray(observation, dtype=np.float64)
    if values.ndim != 1 or values.size < 3 or not np.isfinite(values[:3]).all():
        raise ValueError("A4 observation must begin with a finite TCP position")
    start = values[:3]
    target = start
    if target_position is not None:
        candidate = np.asarray(target_position, dtype=np.float64)
        if candidate.shape == (3,) and np.isfinite(candidate).all():
            target = candidate
    focus = 0.5 * (start + target)
    focus[2] = max(float(focus[2]), 0.10)
    return focus


def orbit_view_from_values(
    clock_hour: int,
    elevation_degrees: float,
    radius_m: float,
    base_camera_name: str,
) -> OrbitCameraView:
    return OrbitCameraView(
        clock_hour=int(clock_hour),
        elevation_degrees=float(elevation_degrees),
        radius_m=float(radius_m),
        base_camera_name=str(base_camera_name),
    )


__all__ = [
    "ORBIT_ALL_VIEWS",
    "ORBIT_DEVELOPMENT_VIEWS",
    "ORBIT_TRAIN_VIEWS",
    "OrbitCameraView",
    "a4_orbit_focus",
    "camera_look_at_quaternion_mujoco",
    "orbit_camera_position",
    "orbit_view_from_values",
    "set_fixed_camera_look_at",
]
