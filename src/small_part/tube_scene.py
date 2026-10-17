"""Move one upright tube from a source rack into an empty target rack.

The original ball scene is not loaded. Each rack has four round holes in one
row, on an upper plate and a lower plate, with the side left open. The tube
has a round bottom and stays mouth-up: the jaws close from the side, around
the upper wall, and the wrist does not roll the tube over. Liquid is not
simulated. A tilt past a few degrees is treated as a spill.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, asdict

import mujoco
import numpy as np

from small_part.scene import ROBOT_XML, _mesh_assets, _use_jaw_shells, _add_rigid_shell, _cluster_triangles

# Shoulder of the posed SO-101. Both racks sit inside the upright-pinch reach.
SHOULDER_XY = np.array([0.229, 0.150])
# Four holes, one straight row. Kartell 370 is a 74 mm two-tier rack. The
# opening is only a little wider than the tube, or the two plates cannot stop
# a lean. Hole 1 is the recorded pinch. The other holes of the same plate sit
# one spacing either side of it, so both racks stay one piece.
HOLE_SPACING = 0.034
SOURCE_ROW = np.array([0.466, 0.228])
# The target row is on the far side of the table, 268 mm from the source.
# The side-on native-jaw grasp keeps both rows within the reachable workspace.
TARGET_ROW = np.array([0.466, -0.040])
# Table top. A two-tier polypropylene rack for 16–20 mm tubes is about 74 mm
# tall (Kartell 370, BRAND). Two round holes, one above the other, keep the
# tube upright. The side between them is open.
RACK_Z = 0.100
RACK_HEIGHT = 0.074
TUBE_RADIUS = 0.011
TUBE_HALF_LENGTH = 0.055
# The round bottom sits on a 6 mm shelf. The shelf only carries the weight.
TUBE_CENTER_Z = RACK_Z + 0.006 + TUBE_HALF_LENGTH
# 26 mm opening, 2 mm of play around the 22 mm tube.
HOLE_RADIUS = 0.013
# The mouth stands 42 mm above the upper lip, allowing a side-on pinch.
# Thin two-tier plates guide the tube during insertion.
DECK = 0.005
COLLAR = 16
# A few degrees. Past this the mouth is no longer up and the liquid would leave.
MAX_TILT = 0.10
# Hemisphere centre and the straight wall above it, in the tube frame.
_SPHERE_Z = -TUBE_HALF_LENGTH + TUBE_RADIUS
_CYL_BOTTOM = _SPHERE_Z - 0.001
_CYL_HALF = (TUBE_HALF_LENGTH - _CYL_BOTTOM) / 2
_CYL_Z = (TUBE_HALF_LENGTH + _CYL_BOTTOM) / 2


def hole_xy(row: np.ndarray, index: int) -> np.ndarray:
    """One of four holes in a straight row. Index 1 is the recorded pinch."""

    if index not in range(4):
        raise IndexError(index)
    return np.array([float(row[0]) + (index - 1) * HOLE_SPACING, float(row[1])])


def _stl(vertices: list[tuple[float, float, float]], triangles: list[tuple[int, int, int]]) -> bytes:
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


def _hemisphere_stl() -> bytes:
    """Round bottom of a culture tube. The flat face is the convex hull, not the mesh."""

    segments = 24
    rings = 8
    vertices: list[tuple[float, float, float]] = []
    ring_ids: list[list[int]] = []
    for index in range(rings):
        alpha = (np.pi / 2) * index / rings
        height = -TUBE_RADIUS * np.sin(alpha)
        radius = TUBE_RADIUS * np.cos(alpha)
        ids = []
        for step in range(segments):
            angle = 2 * np.pi * step / segments
            ids.append(len(vertices))
            vertices.append((radius * np.cos(angle), radius * np.sin(angle), height))
        ring_ids.append(ids)
    pole = len(vertices)
    vertices.append((0.0, 0.0, -TUBE_RADIUS))
    triangles: list[tuple[int, int, int]] = []
    for index in range(rings - 1):
        for step in range(segments):
            following = (step + 1) % segments
            a, b = ring_ids[index][step], ring_ids[index][following]
            c, d = ring_ids[index + 1][step], ring_ids[index + 1][following]
            triangles.extend(((a, c, b), (b, c, d)))
    for step in range(segments):
        following = (step + 1) % segments
        triangles.append((ring_ids[-1][step], pole, ring_ids[-1][following]))
    return _stl(vertices, triangles)


def _annulus_stl() -> bytes:
    """One circular hole in a thin plate. Visual only; collision uses the collar."""

    segments = 48
    inner = HOLE_RADIUS
    outer = HOLE_RADIUS + 0.0028
    half = DECK / 2
    vertices: list[tuple[float, float, float]] = []

    def ring(radius: float, height: float) -> list[int]:
        ids = []
        for step in range(segments):
            angle = 2 * np.pi * step / segments
            ids.append(len(vertices))
            vertices.append((radius * np.cos(angle), radius * np.sin(angle), height))
        return ids

    bottom_inner, bottom_outer = ring(inner, -half), ring(outer, -half)
    top_inner, top_outer = ring(inner, half), ring(outer, half)
    triangles: list[tuple[int, int, int]] = []
    for step in range(segments):
        following = (step + 1) % segments
        triangles.extend(
            (
                (top_inner[step], top_outer[step], top_outer[following]),
                (top_inner[step], top_outer[following], top_inner[following]),
                (bottom_inner[step], bottom_outer[following], bottom_outer[step]),
                (bottom_inner[step], bottom_inner[following], bottom_outer[following]),
                (bottom_inner[step], top_inner[step], top_inner[following]),
                (bottom_inner[step], top_inner[following], bottom_inner[following]),
                (bottom_outer[step], top_outer[following], top_outer[step]),
                (bottom_outer[step], bottom_outer[following], top_outer[following]),
            )
        )
    return _stl(vertices, triangles)


def _hole_xs() -> list[float]:
    """Hole centres in the rack frame. Index 1 is the origin, the recorded pinch."""

    return [float((index - 1) * HOLE_SPACING) for index in range(4)]


def _yaw_quat(yaw: float) -> str:
    half = 0.5 * yaw
    return f"{np.cos(half):.6f} 0 0 {np.sin(half):.6f}"


def _rack(name: str, row: np.ndarray, rgba: str) -> str:
    """One two-tier rack: four round holes, open between the plates.

    The upper and lower plates stop a lean. A thin shelf under each hole only
    holds the round bottom. The sides between the plates stay open.
    """

    slip = "0.2 0.005 0.001"
    shelf = 0.006
    radial = 0.0014
    upper_z = RACK_HEIGHT - DECK / 2
    lower_z = 0.022
    xs = _hole_xs()
    mid_x = 0.5 * (xs[0] + xs[-1])
    half_x = 0.5 * (xs[-1] - xs[0]) + HOLE_RADIUS + 0.010
    half_y = HOLE_RADIUS + 0.010
    parts = []
    for hole, x in enumerate(_hole_xs()):
        parts.append(
            f'<geom name="{name}_shelf_{hole}" type="cylinder" size="{HOLE_RADIUS + 0.008} {shelf / 2}" '
            f'pos="{x} 0 {shelf / 2}" rgba="{rgba}" friction="1.2 0.01 0.001"/>'
        )
    post_half = (RACK_HEIGHT - shelf) / 2
    post_z = (RACK_HEIGHT + shelf) / 2
    for index, (x, y) in enumerate(
        (
            (mid_x - half_x + 0.004, -half_y + 0.004),
            (mid_x + half_x - 0.004, -half_y + 0.004),
            (mid_x - half_x + 0.004, half_y - 0.004),
            (mid_x + half_x - 0.004, half_y - 0.004),
        )
    ):
        parts.append(
            f'<geom name="{name}_post_{index}" type="cylinder" size="0.003 {post_half}" '
            f'pos="{x:.4f} {y:.4f} {post_z}" rgba="{rgba}" friction="{slip}"/>'
        )
    for level, height in (("upper", upper_z), ("lower", lower_z)):
        for hole, x in enumerate(_hole_xs()):
            parts.append(
                f'<geom name="{name}_{level}_ring_{hole}" type="mesh" mesh="rack_hole" '
                f'pos="{x} 0 {height}" rgba="{rgba}" contype="0" conaffinity="0"/>'
            )
            centre = HOLE_RADIUS + radial
            tangent = centre * np.tan(np.pi / COLLAR) * 1.08
            for step in range(COLLAR):
                angle = 2 * np.pi * step / COLLAR
                px = x + centre * np.cos(angle)
                py = centre * np.sin(angle)
                parts.append(
                    f'<geom name="{name}_{level}_wall_{hole}_{step}" type="box" '
                    f'size="{tangent:.5f} {radial} {DECK / 2}" pos="{px:.5f} {py:.5f} {height}" '
                    f'quat="{_yaw_quat(angle + np.pi / 2)}" rgba="{rgba}" friction="{slip}" group="3" '
                    "/>"
                )
            if level == "upper":
                lip = HOLE_RADIUS + 0.0025 + radial
                lip_tangent = lip * np.tan(np.pi / COLLAR) * 1.08
                lip_z = height + DECK / 2 + 0.001
                for step in range(COLLAR):
                    angle = 2 * np.pi * step / COLLAR
                    px = x + lip * np.cos(angle)
                    py = lip * np.sin(angle)
                    parts.append(
                        f'<geom name="{name}_upper_lip_{hole}_{step}" type="box" '
                        f'size="{lip_tangent:.5f} {radial} 0.001" pos="{px:.5f} {py:.5f} {lip_z}" '
                        f'quat="{_yaw_quat(angle + np.pi / 2)}" rgba="{rgba}" friction="{slip}" group="3"/>'
                    )
    return f'<body name="{name}_rack" pos="{row[0]} {row[1]} {RACK_Z}">' + "".join(parts) + "</body>"


def _rack_geoms(name: str) -> tuple[str, ...]:
    names = [f"{name}_shelf_{hole}" for hole in range(4)]
    names.extend(f"{name}_post_{index}" for index in range(4))
    for level in ("upper", "lower"):
        for hole in range(4):
            names.extend(f"{name}_{level}_wall_{hole}_{step}" for step in range(COLLAR))
    for hole in range(4):
        names.extend(f"{name}_upper_lip_{hole}_{step}" for step in range(COLLAR))
    return tuple(names)


def _tube_contacts() -> str:
    """Slick hole walls, a shelf that can hold the round bottom.

    Geom friction is the element-wise max of the two sides, so the tube's
    grip friction would also glue it to the rack. An explicit pair keeps the
    polypropylene slick without loosening the pinch.
    """

    lines = []
    for rack in ("source", "target"):
        for geom in _rack_geoms(rack):
            # Bottom shelves use 0.9 sliding friction; the guide walls use 0.01.
            # Shelf names include a hole suffix, e.g. source_shelf_0.
            # Pair coefficients are tangent1, tangent2, spin, roll1, roll2.
            friction = "0.9 0.9 0.01 0.001 0.001" if "_shelf_" in geom else "0.01 0.01 0.001 0.0001 0.0001"
            for tube in ("tube_wall", "tube_bottom"):
                lines.append(
                    f'<pair geom1="{tube}" geom2="{geom}" condim="3" friction="{friction}"/>'
                )
    return "<contact>" + "".join(lines) + "</contact>"


def scene_xml() -> str:
    if not ROBOT_XML.is_file():
        raise FileNotFoundError(f"SO-101 model is missing: {ROBOT_XML}")
    source = _rack("source", SOURCE_ROW, "0.20 0.35 0.72 1")
    target = _rack("target", TARGET_ROW, "0.72 0.42 0.16 1")
    start = hole_xy(SOURCE_ROW, 0)
    return f"""
<mujoco model="tube_racks">
  <size memory="256M"/>
  <include file="{ROBOT_XML}"/>
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
    <material name="tube" rgba="0.75 0.88 0.92 0.45"/>
    <material name="liquid" rgba="0.15 0.55 0.85 0.9"/>
    <mesh name="tube_bottom" file="tube_bottom.stl"/>
    <mesh name="rack_hole" file="rack_hole.stl"/>
  </asset>
  <worldbody>
    <light pos="0.2 -0.2 1.4" dir="0 0.1 -1" directional="true"/>
    <light pos="0.4 0.4 1.0" dir="-0.2 -0.2 -1" directional="true" diffuse="0.25 0.25 0.25"/>
    <camera name="overhead" pos="0.760 -0.621 0.725" xyaxes="0.86603 0.50000 0.00000 -0.28679 0.49673 0.81915" fovy="50"/>
    <geom name="floor" type="plane" size="0 0 0.01" pos="0 0 0" rgba="0.85 0.85 0.85 1"/>
    <body name="table_body" pos="0.18 0 0.08">
      <geom name="table" type="box" size="0.42 0.32 0.02" material="table" friction="0.9 0.01 0.001"/>
    </body>
    <body name="pedestal" pos="0.19 0.15 0.05">
      <geom name="pedestal" type="box" size="0.045 0.045 0.05" rgba="0.25 0.25 0.25 1"/>
    </body>
    {source}
    {target}
    <body name="tube" pos="{start[0]} {start[1]} {TUBE_CENTER_Z}">
      <freejoint name="tube_free"/>
      <geom name="tube_wall" type="cylinder" size="{TUBE_RADIUS} {_CYL_HALF}" pos="0 0 {_CYL_Z}" mass="0.012"
            material="tube" friction="2.4 0.05 0.004" condim="6" solref="0.008 1" contype="3" conaffinity="3"/>
      <geom name="tube_bottom" type="mesh" mesh="tube_bottom" pos="0 0 {_SPHERE_Z}" mass="0.002"
            material="tube" friction="2.4 0.05 0.004" condim="3" solref="0.008 1" contype="3" conaffinity="3"/>
      <geom name="tube_liquid" type="sphere" size="{TUBE_RADIUS - 0.003}" pos="0 0 {_SPHERE_Z + 0.001}"
            mass="0.003" material="liquid" contype="0" conaffinity="0" group="1"/>
      <geom name="tube_liquid_fill" type="cylinder" size="{TUBE_RADIUS - 0.003} 0.012" pos="0 0 {_SPHERE_Z + 0.014}"
            mass="0.001" material="liquid" contype="0" conaffinity="0" group="1"/>
    </body>
  </worldbody>
  {_tube_contacts()}
</mujoco>
"""


def _attach_wrist_camera(spec: mujoco.MjSpec) -> None:
    """SO-101 wrist camera on the gripper, aimed at the fingertips.

    The housing uses the stock mount and unchanged camera placement. Its visual
    geom has no convex-hull collision; the configured loader adds the original
    mesh surface as a rigid collision shell, keeping the gap open.
    """

    mount_mesh = spec.add_mesh()
    mount_mesh.name = "wrist_camera_mount_so101_v1"
    mount_mesh.file = "wrist_camera_mount_so101_v1.stl"
    mount_mesh.scale[:] = (0.001, 0.001, 0.001)
    camera_mesh = spec.add_mesh()
    camera_mesh.name = "wrist_camera_so101_v1"
    camera_mesh.file = "wrist_camera_so101_v1.stl"
    mount = spec.body("gripper").add_body(
        name="wrist_camera_mount",
        pos=(-0.0150, 0.0240, -0.0315),
        quat=(
            0.4997981226682881,
            -0.5001999599912927,
            0.4998017973423795,
            -0.5001999599912927,
        ),
    )
    mount.add_geom(
        name="wrist_camera_mount_geom",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="wrist_camera_mount_so101_v1",
        rgba=(1.0, 0.82, 0.12, 1.0),
        contype=0,
        conaffinity=0,
        mass=0,
        density=0,
    )
    lens = mount.add_body(
        name="wrist_camera_body",
        pos=(-0.04941, -0.03813, 0.0175),
        quat=(
            0.37978693649389456,
            -0.5964584309361936,
            -0.37978833153177,
            -0.596456240026065,
        ),
    )
    lens.add_geom(
        name="wrist_camera_geom",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="wrist_camera_so101_v1",
        rgba=(0.12, 0.12, 0.12, 1.0),
        contype=0,
        conaffinity=0,
        mass=0,
        density=0,
    )
    # The housing's +Z points at the fingertips. A MuJoCo camera looks along
    # its own -Z, so it is turned around and set just in front of the lens.
    lens.add_camera(name="wrist", pos=(0.0, 0.0, 0.034), quat=(0.0, 1.0, 0.0, 0.0), fovy=70)



@dataclass(frozen=True)
class PhysicsConfig:
    """Explicit experimental parameters; not measurements of real materials."""
    wall_friction: float = 0.2
    shelf_friction: float = 0.9
    tube_friction: float = 2.4
    jaw_friction: float = 1.5
    noslip_iterations: int = 0
    housing_collision: bool = True

    def __post_init__(self):
        for name in ('wall_friction','shelf_friction','tube_friction','jaw_friction'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<0:
                raise ValueError(name)
        if self.noslip_iterations<0: raise ValueError('noslip_iterations')

LEGACY_PHYSICS = PhysicsConfig(wall_friction=.01, noslip_iterations=20, housing_collision=False)

def camera_signature(model):
    def mount_chain(body_id):
        chain=[]
        while body_id:
            chain.append({'name':model.body(body_id).name,
                          'pos':model.body_pos[body_id].tolist(),
                          'quat':model.body_quat[body_id].tolist()})
            body_id=int(model.body_parentid[body_id])
        return chain
    return {name: {'pos': model.cam_pos[model.camera(name).id].tolist(),
                   'quat': model.cam_quat[model.camera(name).id].tolist(),
                   'fovy': float(model.cam_fovy[model.camera(name).id]),
                   'body_name': model.body(int(model.cam_bodyid[model.camera(name).id])).name,
                   'mount_chain':mount_chain(int(model.cam_bodyid[model.camera(name).id]))}
            for name in ('overhead','wrist')}

def _restore_housing_shells(spec):
    """Restore stock surfaces without the convex hull filling finger/camera gaps.

    Rigid shell vertices come from the compiled source mesh in its body frame.
    Like the existing jaw shells, surfaces on the same assembly do not collide
    with one another; all collide with the tube and rack structural geometries.
    """
    compiled = spec.compile()
    selected = {'wrist','gripper','wrist_camera_mount','wrist_camera_body'}
    jaw_meshes = {'wrist_roll_follower_so101_v1','moving_jaw_so101_v1'}
    seen=set()
    for gid in range(compiled.ngeom):
        bid=int(compiled.geom_bodyid[gid]); body=compiled.body(bid).name
        mid=int(compiled.geom_dataid[gid])
        if body not in selected or compiled.geom_type[gid]!=mujoco.mjtGeom.mjGEOM_MESH:continue
        if compiled.mesh(mid).name in jaw_meshes:continue
        key=(bid,mid,tuple(compiled.geom_pos[gid]),tuple(compiled.geom_quat[gid]))
        if key in seen:continue
        seen.add(key)
        va=int(compiled.mesh_vertadr[mid]);vn=int(compiled.mesh_vertnum[mid])
        fa=int(compiled.mesh_faceadr[mid]);fn=int(compiled.mesh_facenum[mid])
        verts=compiled.mesh_vert[va:va+vn];faces=compiled.mesh_face[fa:fa+fn]
        rotation=np.zeros(9);mujoco.mju_quat2Mat(rotation,compiled.geom_quat[gid])
        posed=verts@rotation.reshape(3,3).T+compiled.geom_pos[gid]
        vertices,triangles=_cluster_triangles(posed[faces],.002)
        name=f'housing_{len(seen)}'
        _add_rigid_shell(spec,name,body,vertices,triangles)
        shell=next(f for f in spec.flexes if f.name==name);shell.contype=4 << (len(seen)-1);shell.conaffinity=2;shell.friction=[.5,.005,.0001]
        shell.rgba=[0,0,0,0]


def load_tube_model(physics: PhysicsConfig | None = None) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Apply explicit experimental physics after compilation; no attachments.

    XML contact coefficients are legacy defaults. PhysicsConfig controls the
    compiled model used for execution and is recorded with every new trial.
    """
    physics = physics or PhysicsConfig()
    spec = mujoco.MjSpec.from_string(scene_xml())
    spec.assets = {**_mesh_assets(), "tube_bottom.stl": _hemisphere_stl(), "rack_hole.stl": _annulus_stl()}
    for actuator in spec.actuators:
        actuator.forcerange = [-3.35, 3.35]
    _use_jaw_shells(spec)
    _attach_wrist_camera(spec)
    if physics.housing_collision: _restore_housing_shells(spec)
    model = spec.compile()
    # Flex/geom contacts do not apply the usual adjacent-body geom filtering.
    # Give each restored housing a bit and add that bit to non-adjacent bodies.
    # Tube/rack/table contacts remain enabled through the housing affinity bit 2.
    for name in ('floor','table','pedestal'):
        model.geom_contype[model.geom(name).id] = 3
    for f in range(model.nflex):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_FLEX,f) or ''
        if not name.startswith('housing_'): continue
        body=int(model.flex_vertbodyid[int(model.flex_vertadr[f])])
        welded=int(model.body_weldid[body])
        parent=int(model.body_weldid[int(model.body_parentid[welded])])
        bit=int(model.flex_contype[f])
        for gid in range(model.ngeom):
            if not model.geom_contype[gid]: continue
            other=int(model.body_weldid[int(model.geom_bodyid[gid])])
            other_parent=int(model.body_weldid[int(model.body_parentid[other])])
            if other not in (welded,parent) and other_parent != welded:
                model.geom_conaffinity[gid] |= bit
    model.opt.noslip_iterations = physics.noslip_iterations
    for i in range(model.npair):
        names=[model.geom(int(g)).name or "" for g in (model.pair_geom1[i],model.pair_geom2[i])]
        mu=physics.shelf_friction if any("_shelf_" in n for n in names) else physics.wall_friction
        model.pair_friction[i,:2] = mu
    model.body_pos[model.body("base").id] = (0.19, 0.15, 0.10)
    # Native jaw mesh shells replace convex hulls; no additional gripping pads.
    # Every structural rack collider participates for both tube and robot.
    for index in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
        if name.startswith(("source_", "target_")) and "_ring_" not in name:
            model.geom_contype[index] = 3
            model.geom_conaffinity[index] = 3
    for index in range(model.nflex):
        if mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_FLEX,index) in ("fixed_jaw", "moving_jaw"):
            model.flex_friction[index] = [physics.jaw_friction, 0.005, 0.0001]
    for name in ("tube_wall", "tube_bottom"):
        model.geom_friction[model.geom(name).id] = [physics.tube_friction, 0.05, 0.004]
    return model, mujoco.MjData(model)


def tube_tilt(data: mujoco.MjData) -> float:
    """Angle between the tube axis and the world up direction, in radians."""

    joint = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_JOINT, "tube_free")
    address = int(data.model.jnt_qposadr[joint])
    quat = np.asarray(data.qpos[address + 3 : address + 7], dtype=float)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    axis = mat.reshape(3, 3)[:, 2]
    return float(np.arccos(np.clip(axis[2], -1.0, 1.0)))


def tube_position(data: mujoco.MjData) -> np.ndarray:
    joint = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_JOINT, "tube_free")
    address = int(data.model.jnt_qposadr[joint])
    return np.asarray(data.qpos[address : address + 3], dtype=float).copy()
