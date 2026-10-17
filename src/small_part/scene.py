"""Workcell scene built on the public SO-101 MuJoCo model.

The task object is a small ball in a recessed block. The yellow gripper meshes collide
as a rigid non-convex shell, so the tooth gap stays open. The upstream robot
file is not edited.
"""

from __future__ import annotations

import struct
from pathlib import Path

import mujoco
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROBOT_XML = (
    PACKAGE_ROOT
    / "third_party"
    / "so_arm100"
    / "Simulation"
    / "SO101"
    / "so101_new_calib.xml"
)

# Raised ready pose. The last value leaves the gripper open.
HOME_QPOS = (0.0, -0.55, 1.15, -0.55, 0.0, 0.15)
ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
GRIPPER_JOINT = "gripper"
# On this hinge a larger command opens the yellow jaw. 0.40 is the pinch.
GRIPPER_OPEN = 1.20
GRIPPER_CLOSED = 0.40

# The ball starts in the shallow recess of the square pick pedestal.
# The orange bin is the place target. This location is reachable by both tips.
PICK_BIN = (0.466, 0.228)
PLACE_BIN = (0.325, -0.080)
BALL_X = (PICK_BIN[0] - 0.0015, PICK_BIN[0] + 0.0015)
BALL_Y = (PICK_BIN[1] - 0.0015, PICK_BIN[1] + 0.0015)
BALL_Z = 0.140
BALL_RADIUS = 0.008

SCENE_XML = """
<mujoco model="small_part_cell">
  <include file="{robot}"/>
  <option gravity="0 0 -9.81" timestep="0.002" integrator="implicitfast" noslip_iterations="20"/>
  <default>
    <geom friction="1.0 0.02 0.002" solref="0.004 1" solimp="0.95 0.99 0.001"/>
  </default>
  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.35 0.35 0.35" specular="0.05 0.05 0.05"/>
    <global azimuth="135" elevation="-25" offwidth="1280" offheight="960"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.78 0.76 0.70" rgb2="0.62 0.60 0.55" width="300" height="300"/>
    <material name="table" texture="grid" texrepeat="6 5" reflectance="0.05"/>
    <material name="ball" rgba="0.15 0.72 0.28 1"/>
    <material name="bin" rgba="0.20 0.35 0.75 1"/>
    <mesh name="pick_pedestal_mesh" file="pick_pedestal.stl"/>
  </asset>
  <worldbody>
    <light pos="0.2 -0.2 1.4" dir="0 0.1 -1" directional="true"/>
    <light pos="0.4 0.4 1.0" dir="-0.2 -0.2 -1" directional="true" diffuse="0.25 0.25 0.25"/>
    <camera name="overhead" pos="0.02 0.00 0.78" quat="0.707107 0 0 0.707107" fovy="55"/>
    <camera name="front" pos="0.55 -0.62 0.55" xyaxes="0.75 0.66 0 -0.22 0.25 0.94" fovy="48"/>
    <geom name="floor" type="plane" size="0 0 0.01" pos="0 0 0" rgba="0.85 0.85 0.85 1"/>
    <body name="table_body" pos="0.18 0 0.08">
      <geom name="table" type="box" size="0.42 0.32 0.02" material="table" friction="0.9 0.01 0.001"/>
    </body>
    <body name="pedestal" pos="0.19 0.15 0.05">
      <geom name="pedestal" type="box" size="0.045 0.045 0.05" rgba="0.25 0.25 0.25 1"/>
    </body>
    <body name="pick_bin" pos="0.466 0.228 0.116">
      <!-- The original 50 mm square block with a 20 mm round, 6 mm deep cup. -->
      <geom name="pick_pedestal" type="mesh" mesh="pick_pedestal_mesh" rgba="0.20 0.35 0.75 1" contype="0" conaffinity="0"/>
      <!-- Invisible convex pieces give the ball a floor and a retaining rim. -->
      <geom name="pick_floor" type="box" size="0.025 0.025 0.016" rgba="0 0 0 0" friction="0.8 0.005 0.001"/>
      <geom name="socket_x_plus" type="box" size="0.0075 0.025 0.003" pos="0.0175 0 0.019" rgba="0 0 0 0"/>
      <geom name="socket_x_minus" type="box" size="0.0075 0.025 0.003" pos="-0.0175 0 0.019" rgba="0 0 0 0"/>
      <geom name="socket_y_plus" type="box" size="0.010 0.0075 0.003" pos="0 0.0175 0.019" rgba="0 0 0 0"/>
      <geom name="socket_y_minus" type="box" size="0.010 0.0075 0.003" pos="0 -0.0175 0.019" rgba="0 0 0 0"/>
    </body>
    <body name="place_bin" pos="0.325 -0.080 0.10">
      <geom name="place_floor" type="box" size="0.045 0.045 0.004" pos="0 0 0" rgba="0.75 0.45 0.15 1" friction="1.4 0.01 0.001"/>
      <geom name="place_wall_x0" type="box" size="0.004 0.045 0.006" pos="-0.045 0 0.006" rgba="0.75 0.45 0.15 1"/>
      <geom name="place_wall_x1" type="box" size="0.004 0.045 0.006" pos="0.045 0 0.006" rgba="0.75 0.45 0.15 1"/>
      <geom name="place_wall_y0" type="box" size="0.045 0.004 0.006" pos="0 -0.045 0.006" rgba="0.75 0.45 0.15 1"/>
      <geom name="place_wall_y1" type="box" size="0.045 0.004 0.006" pos="0 0.045 0.006" rgba="0.75 0.45 0.15 1"/>
    </body>
    <body name="ball" pos="0.466 0.228 0.26">
      <freejoint name="ball_free"/>
      <geom name="ball_geom" type="sphere" size="0.008" mass="0.008" material="ball" friction="2.4 0.02 0.002" condim="6" solref="0.01 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def scene_xml() -> str:
    if not ROBOT_XML.is_file():
        raise FileNotFoundError(f"SO-101 model is missing: {ROBOT_XML}")
    return SCENE_XML.format(robot=ROBOT_XML)


# Pose the servos can hold with the yellow fingertips level, clear of the table.
# The original hinge opens the moving yellow tooth toward command 1.2.
JAW_QPOS = (0.00, 0.60, -0.22, 0.62, 1.60, 0.45)


def load_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    spec = mujoco.MjSpec.from_string(scene_xml())
    spec.assets = _mesh_assets()
    # The included position class allows more torque than the per-joint
    # forcerange. Cap the arm so the wrist cannot out-pull the shoulder.
    for actuator in spec.actuators:
        actuator.forcerange = [-3.35, 3.35]
    _use_jaw_shells(spec)
    model = spec.compile()
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    model.body_pos[base_id] = (0.19, 0.15, 0.10)
    data = mujoco.MjData(model)
    return model, data


# Mesh-geom collision is a convex hull, which fills the tooth gap. A rigid flex
# uses the same STL faces, so the ball can sit between the yellow teeth.
# Full yellow teeth. The roots of the two meshes pass within about a millimetre
# at every opening; that pair is excluded from the contact below, instead of
# cutting the tooth down until the ball no longer fits between the tips.
_JAW_SHELLS = (
    ("fixed_jaw", "gripper", "wrist_roll_follower_so101_v1"),
    ("moving_jaw", "moving_jaw_so101_v1", "moving_jaw_so101_v1"),
)


def _use_jaw_shells(spec: mujoco.MjSpec) -> None:
    """Replace the yellow jaw convex hulls with the mesh surface itself.

    The shell is built from the compiled mesh, in the same frame as the visual
    geom. Copying the STL with the raw XML pose misses the body's own rotation,
    so the teeth would sit away from the yellow mesh.
    """

    for name, body, mesh in _JAW_SHELLS:
        verts, faces = _compiled_jaw_surface(mesh)
        _add_rigid_shell(spec, name, body, verts, faces)
    _drop_jaw_self_contact(spec)
    # The wrist servo's convex hull reaches past the yellow teeth and holds
    # the ball before the moving jaw can close. The teeth themselves collide
    # through the rigid shells below.
    # Every convex hull on the wrist and the yellow fingers reaches past the
    # teeth. The ball is only allowed to meet the rigid shells of those teeth.
    for body in spec.bodies:
        if body.name not in ("wrist", "gripper", "moving_jaw_so101_v1"):
            continue
        for geom in body.geoms:
            if geom.contype:
                geom.contype = 0
                geom.conaffinity = 0


def _compiled_jaw_surface(mesh: str, *, cell: float = 0.0035) -> tuple[np.ndarray, np.ndarray]:
    """Every triangle of one yellow mesh, posed as its collision geom."""

    spec = mujoco.MjSpec.from_string(scene_xml())
    spec.assets = _mesh_assets()
    model = spec.compile()
    mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, mesh)
    vert_adr = model.mesh_vertadr[mesh_id]
    vertices = np.asarray(model.mesh_vert[vert_adr : vert_adr + model.mesh_vertnum[mesh_id]])
    face_adr = model.mesh_faceadr[mesh_id]
    faces = np.asarray(model.mesh_face[face_adr : face_adr + model.mesh_facenum[mesh_id]])
    geom_id = next(
        index
        for index in range(model.ngeom)
        if model.geom_dataid[index] == mesh_id and model.geom_contype[index] != 0
    )
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
    posed = (rotation.reshape(3, 3) @ vertices.T).T + model.geom_pos[geom_id]
    return _cluster_triangles(posed[faces], cell)


def _drop_jaw_self_contact(spec: mujoco.MjSpec) -> None:
    """Keep the teeth from colliding with each other.

    The roots of the two yellow meshes occupy the same space at every opening,
    which fills the arena with contacts and stops the jaw moving. The tips are
    centimetres apart, so nothing that could hold the ball is removed. Each
    tooth still collides with the ball.
    """

    for flex in spec.flexes:
        if flex.name in ("fixed_jaw", "moving_jaw"):
            flex.conaffinity = 2


def _add_rigid_shell(
    spec: mujoco.MjSpec,
    name: str,
    body: str,
    verts: np.ndarray,
    faces: np.ndarray,
) -> None:
    flex = spec.add_flex()
    flex.name = name
    flex.dim = 2
    flex.radius = 0.0005
    flex.vert = verts.reshape(-1)
    flex.vertbody = [body]
    flex.elem = faces.reshape(-1)
    flex.selfcollide = mujoco.mjtFlexSelf.mjFLEXSELF_NONE
    flex.contype = 1
    flex.conaffinity = 1
    flex.condim = 6
    # Same grip as the neck pads. A slicker shell is what actually touches the
    # tube once the jaw seats, and it lets go as soon as the hand yaws.
    flex.friction = [2.4, 0.1, 0.02]
    flex.solref = [0.01, 1]
    flex.solimp = [0.95, 0.99, 0.001, 0.5, 2]
    flex.rgba = [1.0, 0.82, 0.12, 0.35]


def _cluster_triangles(triangles: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    flat = np.round(triangles.reshape(-1, 3), 5)
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    quantized = np.round(np.round(unique / cell) * cell, 5)
    _, first, mapped = np.unique(quantized, axis=0, return_index=True, return_inverse=True)
    vertices = unique[first]
    faces = mapped[faces]
    valid = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces = faces[valid]
    _, index = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    faces = faces[np.sort(index)]
    used = np.unique(faces)
    remap = -np.ones(len(vertices), dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return vertices[used], remap[faces]


def _mesh_assets() -> dict[str, bytes]:
    asset_dir = ROBOT_XML.parent / "assets"
    assets = {path.name: path.read_bytes() for path in asset_dir.glob("*.stl")}
    assets["pick_pedestal.stl"] = _pick_pedestal_mesh()
    return assets


def _pick_pedestal_mesh() -> bytes:
    """Closed 50 mm square block with a 20 mm vertical-sided, 6 mm cup."""

    sides = 48
    outer = 0.025
    opening = 0.010
    bottom = -0.016  # z = 0.100, flush with the table
    cup_bottom = 0.016  # z = 0.132, under the 16 mm ball
    cup_top = 0.022  # z = 0.138
    angles = np.arange(sides) * (2 * np.pi / sides)
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    vertices: list[tuple[float, float, float]] = [(0.0, 0.0, cup_bottom)]
    rings: list[list[int]] = []

    # A shallow flat-bottomed well keeps the sphere exposed to the original jaws.
    for radius in np.linspace(opening / 6, opening, 6):
        ring = []
        for direction in directions:
            ring.append(len(vertices))
            vertices.append((float(radius * direction[0]), float(radius * direction[1]), cup_bottom))
        rings.append(ring)
    mouth_ring = []
    for direction in directions:
        mouth_ring.append(len(vertices))
        vertices.append((float(opening * direction[0]), float(opening * direction[1]), cup_top))

    outer_ring = []
    lower_ring = []
    for direction in directions:
        point = outer * direction / np.max(np.abs(direction))
        outer_ring.append(len(vertices))
        vertices.append((float(point[0]), float(point[1]), cup_top))
        lower_ring.append(len(vertices))
        vertices.append((float(point[0]), float(point[1]), bottom))
    vertices.append((0.0, 0.0, bottom))
    bottom_center = len(vertices) - 1

    triangles: list[tuple[int, int, int]] = []
    for index in range(sides):
        following = (index + 1) % sides
        triangles.append((0, rings[0][index], rings[0][following]))
        for inner, outer_points in zip(rings[:-1], rings[1:], strict=True):
            triangles.extend(((inner[index], outer_points[index], outer_points[following]),
                              (inner[index], outer_points[following], inner[following])))
        triangles.extend(((rings[-1][index], mouth_ring[index], mouth_ring[following]),
                          (rings[-1][index], mouth_ring[following], rings[-1][following]),
                          (mouth_ring[index], outer_ring[index], outer_ring[following]),
                          (mouth_ring[index], outer_ring[following], mouth_ring[following]),
                          (outer_ring[index], lower_ring[index], lower_ring[following]),
                          (outer_ring[index], lower_ring[following], outer_ring[following]),
                          (bottom_center, lower_ring[following], lower_ring[index])))

    mesh = bytearray(80)
    mesh.extend(struct.pack("<I", len(triangles)))
    points = np.asarray(vertices, dtype=np.float32)
    for triangle in triangles:
        p0, p1, p2 = points[list(triangle)]
        normal = np.cross(p1 - p0, p2 - p0)
        length = float(np.linalg.norm(normal))
        normal = normal / length if length else normal
        mesh.extend(struct.pack("<12fH", *normal, *p0, *p1, *p2, 0))
    return bytes(mesh)
