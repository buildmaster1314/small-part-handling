from __future__ import annotations

import mujoco
import numpy as np

from fcpg.lewm_orbit_camera import (
    OrbitCameraView,
    a4_orbit_focus,
    orbit_camera_position,
    set_fixed_camera_look_at,
)
from fcpg.mujoco_camera import (
    camera_extrinsic_cv_from_mujoco,
    get_camera_id,
)


def test_clock_camera_positions_follow_documented_axes() -> None:
    focus = np.asarray([0.2, -0.1, 0.3])
    radius = 1.4
    elevation = 25.0
    height = radius * np.tan(np.deg2rad(elevation))

    clock12 = orbit_camera_position(
        focus, OrbitCameraView(12, elevation_degrees=elevation, radius_m=radius)
    )
    clock3 = orbit_camera_position(
        focus, OrbitCameraView(3, elevation_degrees=elevation, radius_m=radius)
    )

    np.testing.assert_allclose(clock12, focus + [0.0, -radius, height], atol=1.0e-7)
    np.testing.assert_allclose(clock3, focus + [radius, 0.0, height], atol=1.0e-7)


def test_a4_orbit_focus_uses_reset_tcp_and_goal_midpoint() -> None:
    observation = np.asarray([0.0, 0.2, 0.05, 1.0], dtype=np.float64)
    target = np.asarray([0.4, -0.2, 0.15], dtype=np.float64)

    focus = a4_orbit_focus(observation, target)

    np.testing.assert_allclose(focus, [0.2, 0.0, 0.1])


def test_fixed_camera_look_at_centres_target_in_cv_frame() -> None:
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><camera name="cam" pos="0 0 1"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    camera_id = get_camera_id(model, "cam")
    position = np.asarray([1.0, -2.0, 1.5])
    target = np.asarray([0.1, 0.2, 0.3])

    set_fixed_camera_look_at(model, data, camera_id, position, target)
    target_camera = camera_extrinsic_cv_from_mujoco(data, camera_id) @ np.r_[target, 1.0]

    np.testing.assert_allclose(target_camera[:2], 0.0, atol=1.0e-8)
    assert float(target_camera[2]) > 0.0
