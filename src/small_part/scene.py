"""Workcell scene built on the public SO-101 MuJoCo model."""

from __future__ import annotations

from pathlib import Path

import mujoco

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROBOT_XML = (
    PACKAGE_ROOT
    / "third_party"
    / "so_arm100"
    / "Simulation"
    / "SO101"
    / "so101_new_calib.xml"
)

HOME_QPOS = (0.0, -0.6, 0.8, 0.4, 0.0, 0.0)
ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
GRIPPER_JOINT = "gripper"
GRIPPER_OPEN = 0.2
GRIPPER_CLOSED = 1.4

SCENE_XML = """
<mujoco model="small_part_cell">
  <include file="{robot}"/>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="140" elevation="-25"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="5 5" reflectance="0.1"/>
    <material name="part" rgba="0.85 0.2 0.15 1"/>
    <material name="bin" rgba="0.2 0.45 0.75 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="table" type="box" size="0.45 0.35 0.02" pos="0.18 0 0.08" material="grid"/>
    <body name="part" pos="0.22 -0.08 0.115">
      <freejoint name="part_free"/>
      <geom name="part_geom" type="box" size="0.012 0.012 0.008" mass="0.02" material="part" friction="1.2 0.05 0.01"/>
    </body>
    <body name="place_bin" pos="0.22 0.12 0.10">
      <geom name="bin_floor" type="box" size="0.035 0.035 0.004" material="bin" contype="0" conaffinity="0"/>
      <site name="place_target" pos="0 0 0.02" size="0.008" rgba="0.1 0.8 0.2 0.6"/>
    </body>
    <site name="grasp_hover" pos="0.22 -0.08 0.18" size="0.006" rgba="1 0.6 0.1 0.5"/>
  </worldbody>
</mujoco>
"""


def scene_xml() -> str:
    if not ROBOT_XML.is_file():
        raise FileNotFoundError(f"SO-101 model is missing: {ROBOT_XML}")
    return SCENE_XML.format(robot=ROBOT_XML)


def load_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(scene_xml(), assets=_mesh_assets())
    data = mujoco.MjData(model)
    return model, data


def _mesh_assets() -> dict[str, bytes]:
    asset_dir = ROBOT_XML.parent / "assets"
    return {path.name: path.read_bytes() for path in asset_dir.glob("*.stl")}
