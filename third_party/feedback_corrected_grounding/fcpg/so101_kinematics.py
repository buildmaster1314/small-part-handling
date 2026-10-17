"""Forward kinematics for LeRobot's normalized SO-101 joint convention."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


SO101_JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
SO101_ARM_JOINT_NAMES = tuple(name.removesuffix(".pos") for name in SO101_JOINT_NAMES[:5])
SO101_MUJOCO_JOINT_NAMES = tuple(
    name.removesuffix(".pos") for name in SO101_JOINT_NAMES
)
SO101_ANCHOR_NAMES = (
    "gripper_center",
    "fixed_finger_tip",
    "moving_finger_tip",
)


@dataclass(frozen=True)
class SO101IKResult:
    """Result of one position-only inverse-kinematics solve."""

    state: np.ndarray
    achieved_xyz: np.ndarray
    target_xyz: np.ndarray
    error_m: float
    converged: bool
    iterations: int


@dataclass(frozen=True)
class SO101AnchorIKResult:
    """Result of fitting all three physical gripper anchors."""

    state: np.ndarray
    achieved_anchors: np.ndarray
    target_anchors: np.ndarray
    rms_error_m: float
    max_error_m: float
    converged: bool
    iterations: int


def normalized_arm_positions_to_radians(
    positions: np.ndarray,
    joint_ranges: np.ndarray,
) -> np.ndarray:
    """Map LeRobot SO-101 ``[-100, 100]`` arm positions to joint radians.

    The post-PR-777 SO-101 calibration places zero at the middle of each
    measured joint range. The gripper is intentionally excluded because
    LeRobot represents it as a linear ``[0, 100]`` command while the upstream
    MuJoCo model still uses a revolute approximation.
    """

    values = np.asarray(positions, dtype=np.float64)
    ranges = np.asarray(joint_ranges, dtype=np.float64)
    if values.shape[-1:] != (5,):
        raise ValueError(f"SO-101 arm positions must end in five joints, got {values.shape}")
    if ranges.shape != (5, 2):
        raise ValueError(f"SO-101 joint ranges must have shape (5, 2), got {ranges.shape}")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(ranges)):
        raise ValueError("SO-101 positions and joint ranges must be finite")
    if np.any(ranges[:, 1] <= ranges[:, 0]):
        raise ValueError("SO-101 joint range maxima must exceed minima")

    clipped = np.clip(values, -100.0, 100.0)
    fraction = (clipped + 100.0) / 200.0
    return ranges[:, 0] + fraction * (ranges[:, 1] - ranges[:, 0])


def radians_to_normalized_arm_positions(
    radians: np.ndarray,
    joint_ranges: np.ndarray,
) -> np.ndarray:
    """Map five SO-101 hinge angles to LeRobot's ``[-100, 100]`` convention."""

    values = np.asarray(radians, dtype=np.float64)
    ranges = np.asarray(joint_ranges, dtype=np.float64)
    if values.shape[-1:] != (5,):
        raise ValueError(f"SO-101 arm radians must end in five joints, got {values.shape}")
    if ranges.shape != (5, 2):
        raise ValueError(f"SO-101 joint ranges must have shape (5, 2), got {ranges.shape}")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(ranges)):
        raise ValueError("SO-101 radians and joint ranges must be finite")
    span = ranges[:, 1] - ranges[:, 0]
    if np.any(span <= 0):
        raise ValueError("SO-101 joint range maxima must exceed minima")
    clipped = np.clip(values, ranges[:, 0], ranges[:, 1])
    return 200.0 * (clipped - ranges[:, 0]) / span - 100.0


def normalized_positions_to_mujoco_qpos(
    positions: np.ndarray,
    joint_ranges: np.ndarray,
) -> np.ndarray:
    """Map six LeRobot SO-101 positions to the official MuJoCo hinges.

    The five arm joints use LeRobot's ``[-100, 100]`` calibration. The public
    SO-101 MuJoCo model still represents the gripper as a hinge, so its
    ``[0, 100]`` linear command is mapped across that hinge range only for
    pose visualization. End-effector XYZ does not depend on this approximation.
    """

    values = np.asarray(positions, dtype=np.float64)
    ranges = np.asarray(joint_ranges, dtype=np.float64)
    if values.shape[-1:] != (6,):
        raise ValueError(f"SO-101 positions must end in six joints, got {values.shape}")
    if ranges.shape != (6, 2):
        raise ValueError(f"SO-101 joint ranges must have shape (6, 2), got {ranges.shape}")
    result = np.empty_like(values, dtype=np.float64)
    result[..., :5] = normalized_arm_positions_to_radians(values[..., :5], ranges[:5])
    gripper_fraction = np.clip(values[..., 5], 0.0, 100.0) / 100.0
    result[..., 5] = ranges[5, 0] + gripper_fraction * (
        ranges[5, 1] - ranges[5, 0]
    )
    return result


class SO101MuJoCoKinematics:
    """Evaluate the official SO-101 ``gripperframe`` site in base coordinates."""

    def __init__(self, model_path: str | Path) -> None:
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - environment-specific dependency
            raise RuntimeError("SO-101 FK requires the mujoco Python package") from exc

        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SO-101 MuJoCo model not found: {path}")
        self.model_path = path
        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)

        all_joint_ids = []
        all_qpos_addresses = []
        for name in SO101_MUJOCO_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"SO-101 MuJoCo model is missing joint {name!r}")
            all_joint_ids.append(joint_id)
            all_qpos_addresses.append(int(self.model.jnt_qposadr[joint_id]))
        self.all_joint_ids = np.asarray(all_joint_ids, dtype=np.int32)
        self.all_qpos_addresses = np.asarray(all_qpos_addresses, dtype=np.int32)
        self.joint_ids = self.all_joint_ids[:5]
        self.qpos_addresses = self.all_qpos_addresses[:5]
        self.dof_addresses = np.asarray(
            self.model.jnt_dofadr[self.joint_ids], dtype=np.int32
        ).copy()
        self.all_joint_ranges = np.asarray(
            self.model.jnt_range[self.all_joint_ids], dtype=np.float64
        ).copy()
        self.joint_ranges = np.asarray(
            self.model.jnt_range[self.joint_ids], dtype=np.float64
        ).copy()
        self.gripperframe_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"
        )
        if self.gripperframe_site_id < 0:
            raise ValueError("SO-101 MuJoCo model is missing site 'gripperframe'")
        self.gripper_body_id = self._body_id("gripper")
        self.moving_jaw_body_id = self._body_id("moving_jaw_so101_v1")
        mujoco.mj_forward(self.model, self.data)
        tool_direction = (
            np.asarray(self.data.site_xpos[self.gripperframe_site_id], dtype=np.float64)
            - np.asarray(self.data.xpos[self.gripper_body_id], dtype=np.float64)
        )
        tool_direction /= np.linalg.norm(tool_direction)
        self.fixed_tip_local = self._mesh_tip_local_offset(
            "wrist_roll_follower_so101_v1",
            self.gripper_body_id,
            tool_direction,
        )
        self.moving_tip_local = self._mesh_tip_local_offset(
            "moving_jaw_so101_v1",
            self.moving_jaw_body_id,
            tool_direction,
        )

    def _body_id(self, name: str) -> int:
        body_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_BODY, name
        )
        if body_id < 0:
            raise ValueError(f"SO-101 MuJoCo model is missing body {name!r}")
        return int(body_id)

    def _mesh_tip_local_offset(
        self,
        mesh_name: str,
        body_id: int,
        tool_direction_world: np.ndarray,
    ) -> np.ndarray:
        """Locate a jaw tip from the distal surface of its official visual mesh."""

        mesh_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_MESH, mesh_name
        )
        if mesh_id < 0:
            raise ValueError(f"SO-101 MuJoCo model is missing mesh {mesh_name!r}")
        candidates = np.flatnonzero(
            (np.asarray(self.model.geom_dataid) == mesh_id)
            & (np.asarray(self.model.geom_bodyid) == body_id)
            & (np.asarray(self.model.geom_group) == 2)
        )
        if len(candidates) != 1:
            raise ValueError(
                f"expected one visual geom for mesh {mesh_name!r}, got {len(candidates)}"
            )
        geom_id = int(candidates[0])
        vertex_start = int(self.model.mesh_vertadr[mesh_id])
        vertex_stop = vertex_start + int(self.model.mesh_vertnum[mesh_id])
        vertices = np.asarray(
            self.model.mesh_vert[vertex_start:vertex_stop], dtype=np.float64
        )
        geom_rotation = np.asarray(self.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        world_vertices = (
            vertices @ geom_rotation.T
            + np.asarray(self.data.geom_xpos[geom_id], dtype=np.float64)
        )
        scores = world_vertices @ np.asarray(tool_direction_world, dtype=np.float64)
        distal = world_vertices[scores >= np.quantile(scores, 0.995)]
        if len(distal) < 3:
            raise ValueError(f"mesh {mesh_name!r} has insufficient distal vertices")
        tip_world = np.median(distal, axis=0)
        body_rotation = np.asarray(self.data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        return body_rotation.T @ (
            tip_world - np.asarray(self.data.xpos[body_id], dtype=np.float64)
        )

    def _set_state(self, qpos: np.ndarray) -> None:
        self.data.qpos[self.all_qpos_addresses] = qpos
        self._mujoco.mj_forward(self.model, self.data)

    def _body_local_to_world(self, body_id: int, local_point: np.ndarray) -> np.ndarray:
        rotation = np.asarray(self.data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        return (
            np.asarray(self.data.xpos[body_id], dtype=np.float64)
            + rotation @ np.asarray(local_point, dtype=np.float64)
        )

    def forward(self, states: np.ndarray) -> np.ndarray:
        """Return base-frame gripper-center XYZ for normalized six-joint states."""

        values = np.asarray(states, dtype=np.float64)
        if values.shape[-1:] != (6,):
            raise ValueError(f"SO-101 states must end in six joints, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("SO-101 states must be finite")
        leading_shape = values.shape[:-1]
        flat = values.reshape(-1, 6)
        all_qpos = normalized_positions_to_mujoco_qpos(flat, self.all_joint_ranges)
        xyz = np.empty((len(flat), 3), dtype=np.float64)
        for index, qpos in enumerate(all_qpos):
            self._set_state(qpos)
            xyz[index] = self.data.site_xpos[self.gripperframe_site_id]
        return xyz.reshape(*leading_shape, 3)

    def forward_anchors(self, states: np.ndarray) -> np.ndarray:
        """Return center, fixed-tip and moving-tip XYZ in the MuJoCo base frame."""

        values = np.asarray(states, dtype=np.float64)
        if values.shape[-1:] != (6,):
            raise ValueError(f"SO-101 states must end in six joints, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("SO-101 states must be finite")
        leading_shape = values.shape[:-1]
        flat = values.reshape(-1, 6)
        all_qpos = normalized_positions_to_mujoco_qpos(flat, self.all_joint_ranges)
        anchors = np.empty((len(flat), len(SO101_ANCHOR_NAMES), 3), dtype=np.float64)
        for index, qpos in enumerate(all_qpos):
            self._set_state(qpos)
            fixed_tip = self._body_local_to_world(
                self.gripper_body_id, self.fixed_tip_local
            )
            moving_tip = self._body_local_to_world(
                self.moving_jaw_body_id, self.moving_tip_local
            )
            anchors[index, 0] = 0.5 * (fixed_tip + moving_tip)
            anchors[index, 1] = fixed_tip
            anchors[index, 2] = moving_tip
        return anchors.reshape(*leading_shape, len(SO101_ANCHOR_NAMES), 3)

    def forward_anchor_frames(
        self, states: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return base-frame and gripperframe-local anchor coordinates.

        Local coordinates are the correct nominal geometry for an eye-in-hand
        camera. The moving jaw still changes with the gripper command.
        """

        values = np.asarray(states, dtype=np.float64)
        if values.shape[-1:] != (6,):
            raise ValueError(f"SO-101 states must end in six joints, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("SO-101 states must be finite")
        leading_shape = values.shape[:-1]
        flat = values.reshape(-1, 6)
        all_qpos = normalized_positions_to_mujoco_qpos(flat, self.all_joint_ranges)
        base = np.empty((len(flat), len(SO101_ANCHOR_NAMES), 3), dtype=np.float64)
        local = np.empty_like(base)
        for index, qpos in enumerate(all_qpos):
            self._set_state(qpos)
            fixed_tip = self._body_local_to_world(
                self.gripper_body_id, self.fixed_tip_local
            )
            moving_tip = self._body_local_to_world(
                self.moving_jaw_body_id, self.moving_tip_local
            )
            anchors = np.stack(
                (0.5 * (fixed_tip + moving_tip), fixed_tip, moving_tip), axis=0
            )
            site_position = np.asarray(
                self.data.site_xpos[self.gripperframe_site_id], dtype=np.float64
            )
            site_rotation = np.asarray(
                self.data.site_xmat[self.gripperframe_site_id], dtype=np.float64
            ).reshape(3, 3)
            base[index] = anchors
            local[index] = (anchors - site_position[None]) @ site_rotation
        return (
            base.reshape(*leading_shape, len(SO101_ANCHOR_NAMES), 3),
            local.reshape(*leading_shape, len(SO101_ANCHOR_NAMES), 3),
        )

    def clipping_fraction(self, states: np.ndarray) -> float:
        values = np.asarray(states, dtype=np.float64)
        if values.shape[-1:] != (6,):
            raise ValueError(f"SO-101 states must end in six joints, got {values.shape}")
        return float(np.mean(np.any(np.abs(values[..., :5]) > 100.0, axis=-1)))

    def inverse_position(
        self,
        target_xyz: np.ndarray,
        initial_state: np.ndarray,
        *,
        max_iterations: int = 100,
        damping: float = 1e-3,
        tolerance_m: float = 1e-3,
        max_step_normalized: float = 8.0,
        posture_weight: float = 1e-4,
    ) -> SO101IKResult:
        """Solve a bounded position-only IK problem for ``gripperframe``.

        The first five normalized LeRobot joints are optimized. The gripper
        command is copied bit-for-bit from ``initial_state`` because arm IK
        cannot decide when an object should be grasped or released.
        """

        target = np.asarray(target_xyz, dtype=np.float64)
        initial = np.asarray(initial_state, dtype=np.float64)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target_xyz must be one finite XYZ vector")
        if initial.shape != (6,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial_state must be one finite six-joint vector")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if damping <= 0 or not np.isfinite(damping):
            raise ValueError("damping must be finite and positive")
        if tolerance_m <= 0 or not np.isfinite(tolerance_m):
            raise ValueError("tolerance_m must be finite and positive")
        if max_step_normalized <= 0 or not np.isfinite(max_step_normalized):
            raise ValueError("max_step_normalized must be finite and positive")
        if posture_weight < 0 or not np.isfinite(posture_weight):
            raise ValueError("posture_weight must be finite and non-negative")

        state = initial.copy()
        state[:5] = np.clip(state[:5], -100.0, 100.0)
        reference_radians = normalized_arm_positions_to_radians(
            state[:5], self.joint_ranges
        )
        best_state = state.copy()
        best_xyz = self.forward(state)
        best_error = float(np.linalg.norm(target - best_xyz))
        iterations = 0

        for iteration in range(max_iterations):
            iterations = iteration + 1
            qpos = normalized_positions_to_mujoco_qpos(
                state, self.all_joint_ranges
            )
            self._set_state(qpos)
            current_xyz = np.asarray(
                self.data.site_xpos[self.gripperframe_site_id], dtype=np.float64
            ).copy()
            residual = target - current_xyz
            error = float(np.linalg.norm(residual))
            if error < best_error:
                best_error = error
                best_xyz = current_xyz
                best_state = state.copy()
            if error <= tolerance_m:
                break

            jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
            jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
            self._mujoco.mj_jacSite(
                self.model,
                self.data,
                jacobian_position,
                jacobian_rotation,
                self.gripperframe_site_id,
            )
            jacobian = jacobian_position[:, self.dof_addresses]
            current_radians = qpos[:5]
            regularizer = damping * damping + posture_weight
            system = jacobian.T @ jacobian + regularizer * np.eye(5)
            rhs = jacobian.T @ residual
            if posture_weight:
                rhs += posture_weight * (reference_radians - current_radians)
            try:
                delta_radians = np.linalg.solve(system, rhs)
            except np.linalg.LinAlgError:
                delta_radians = np.linalg.lstsq(system, rhs, rcond=None)[0]

            normalized_scale = 200.0 / (
                self.joint_ranges[:, 1] - self.joint_ranges[:, 0]
            )
            delta_normalized = delta_radians * normalized_scale
            largest = float(np.max(np.abs(delta_normalized)))
            if largest > max_step_normalized:
                delta_normalized *= max_step_normalized / largest
            state[:5] = np.clip(state[:5] + delta_normalized, -100.0, 100.0)

        final_error = float(np.linalg.norm(target - best_xyz))
        return SO101IKResult(
            state=best_state.astype(np.float32),
            achieved_xyz=np.asarray(best_xyz, dtype=np.float64),
            target_xyz=target.copy(),
            error_m=final_error,
            converged=bool(final_error <= tolerance_m),
            iterations=iterations,
        )

    def inverse_anchors(
        self,
        target_anchors: np.ndarray,
        initial_state: np.ndarray,
        *,
        max_iterations: int = 80,
        damping: float = 1e-4,
        tolerance_m: float = 1e-3,
        finite_difference_step: float = 0.25,
        max_step_normalized: float = 8.0,
        posture_weight: float = 0.0,
        position_lower: np.ndarray | None = None,
        position_upper: np.ndarray | None = None,
    ) -> SO101AnchorIKResult:
        """Fit center and both fingertip targets in normalized joint space.

        The sixth coordinate is optimized from fingertip separation, so the
        output retains the LeWM-predicted open/close trajectory without a
        hand-written grasp or release phase.
        """

        target = np.asarray(target_anchors, dtype=np.float64)
        initial = np.asarray(initial_state, dtype=np.float64)
        if target.shape != (len(SO101_ANCHOR_NAMES), 3) or not np.all(
            np.isfinite(target)
        ):
            raise ValueError("target_anchors must be finite [3,3]")
        if initial.shape != (6,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial_state must be one finite six-joint vector")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        positive = (
            damping,
            tolerance_m,
            finite_difference_step,
            max_step_normalized,
        )
        if any(not np.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("anchor IK values must be finite and positive")
        if not np.isfinite(posture_weight) or posture_weight < 0:
            raise ValueError("posture_weight must be finite and non-negative")

        lower = np.asarray(
            [-100.0] * 5 + [0.0]
            if position_lower is None
            else position_lower,
            dtype=np.float64,
        )
        upper = np.asarray(
            [100.0] * 6 if position_upper is None else position_upper,
            dtype=np.float64,
        )
        if lower.shape != (6,) or upper.shape != (6,):
            raise ValueError("anchor IK position bounds must be six-vectors")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("anchor IK position bounds must be finite")
        if np.any(upper < lower):
            raise ValueError("anchor IK upper bounds cannot be below lower bounds")
        state = np.clip(initial, lower, upper)
        reference = state.copy()
        best_state = state.copy()
        best_anchors = np.asarray(self.forward_anchors(state), dtype=np.float64)
        best_errors = np.linalg.norm(best_anchors - target, axis=-1)
        best_max_error = float(np.max(best_errors))
        best_rms_error = float(np.sqrt(np.mean(np.square(best_anchors - target))))
        iterations = 0

        for iteration in range(max_iterations):
            iterations = iteration + 1
            current = np.asarray(self.forward_anchors(state), dtype=np.float64)
            errors = np.linalg.norm(current - target, axis=-1)
            max_error = float(np.max(errors))
            rms_error = float(np.sqrt(np.mean(np.square(current - target))))
            if max_error < best_max_error:
                best_state = state.copy()
                best_anchors = current.copy()
                best_max_error = max_error
                best_rms_error = rms_error
            if max_error <= tolerance_m:
                break

            jacobian = np.empty((target.size, 6), dtype=np.float64)
            for joint in range(6):
                plus = state.copy()
                minus = state.copy()
                plus[joint] = min(upper[joint], plus[joint] + finite_difference_step)
                minus[joint] = max(lower[joint], minus[joint] - finite_difference_step)
                denominator = plus[joint] - minus[joint]
                if denominator <= 0:
                    jacobian[:, joint] = 0.0
                    continue
                plus_anchors = np.asarray(
                    self.forward_anchors(plus), dtype=np.float64
                )
                minus_anchors = np.asarray(
                    self.forward_anchors(minus), dtype=np.float64
                )
                jacobian[:, joint] = (
                    (plus_anchors - minus_anchors) / denominator
                ).reshape(-1)

            residual = (target - current).reshape(-1)
            regularizer = damping * damping + posture_weight
            system = jacobian.T @ jacobian + regularizer * np.eye(6)
            rhs = jacobian.T @ residual
            if posture_weight:
                rhs += posture_weight * (reference - state)
            try:
                delta = np.linalg.solve(system, rhs)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(system, rhs, rcond=None)[0]
            largest = float(np.max(np.abs(delta)))
            if largest > max_step_normalized:
                delta *= max_step_normalized / largest
            state = np.clip(state + delta, lower, upper)

        return SO101AnchorIKResult(
            state=best_state.astype(np.float32),
            achieved_anchors=best_anchors,
            target_anchors=target.copy(),
            rms_error_m=best_rms_error,
            max_error_m=best_max_error,
            converged=bool(best_max_error <= tolerance_m),
            iterations=iterations,
        )
