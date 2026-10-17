"""Repo-local MetaWorld drawer retrieval task."""

from __future__ import annotations

import hashlib
import operator
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from metaworld.asset_path_utils import ENV_ASSET_DIR_V3
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.utils import reward_utils


DRAWER_RETRIEVE_TASK_NAME = "drawer-retrieve-v0"
DRAWER_RETRIEVE_TASK_INSTANCE_COUNT = 48
_BLOCK_X_LEVELS = (0.0, -0.048, 0.048, -0.032, 0.032, -0.016, 0.016, 0.056)
_BLOCK_Y_LEVELS = (0.82, 0.795, 0.835, 0.805, 0.825, 0.815)
_TARGET_X_LEVELS = (0.0, -0.096, 0.096, -0.064, 0.064, -0.032, 0.032, 0.104)
_TARGET_Y_LEVELS = (0.78, 0.812, 0.844, 0.876, 0.908, 0.94)
_TARGET_PERMUTATION_MULTIPLIER = 17
_ASSET_DIR_TOKEN = "__METAWORLD_ASSET_DIR__"
_MODEL_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "drawer_retrieve"
    / "drawer_retrieve.xml"
)


def _normalize_task_instance(task_instance: int) -> int:
    if isinstance(task_instance, (bool, np.bool_)):
        raise TypeError("task_instance must be an integer, not bool")
    try:
        normalized = operator.index(task_instance)
    except TypeError as exc:
        raise TypeError("task_instance must be an integer") from exc
    if not 0 <= normalized < DRAWER_RETRIEVE_TASK_INSTANCE_COUNT:
        raise ValueError(
            "task_instance must be in the inclusive range "
            f"0..{DRAWER_RETRIEVE_TASK_INSTANCE_COUNT - 1}"
        )
    return normalized


@lru_cache(maxsize=1)
def resolved_drawer_retrieve_model_path() -> str:
    """Return a loadable MJCF copy with MetaWorld asset paths resolved."""

    template = _MODEL_TEMPLATE.read_text(encoding="utf-8")
    if _ASSET_DIR_TOKEN not in template:
        raise RuntimeError(f"Missing asset token in {_MODEL_TEMPLATE}")

    asset_dir = escape(str(Path(ENV_ASSET_DIR_V3).resolve()), {'"': "&quot;"})
    resolved = template.replace(_ASSET_DIR_TOKEN, asset_dir)
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    cache_dir = Path(tempfile.gettempdir()) / "fcpg_drawer_retrieve" / digest
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "drawer_retrieve.xml"
    if not model_path.exists():
        temporary_path = cache_dir / f"drawer_retrieve.{os.getpid()}.tmp"
        temporary_path.write_text(resolved, encoding="utf-8")
        os.replace(temporary_path, model_path)
    return str(model_path)


class SawyerDrawerRetrieveEnvV0(SawyerXYZEnv):
    """Open a drawer and place its free block on the outer shell."""

    TASK_NAME = DRAWER_RETRIEVE_TASK_NAME
    TASK_INSTANCE_COUNT = DRAWER_RETRIEVE_TASK_INSTANCE_COUNT
    BLOCK_HALF_EXTENT = 0.018
    BLOCK_INIT_POS = (0.0, 0.82, 0.052)
    OUTER_TOP_CENTER = (0.0, 0.87, 0.198)
    OUTER_TOP_HALF_SIZE = (0.152, 0.14, 0.012)
    OUTER_TOP_SURFACE_Z = OUTER_TOP_CENTER[2] + OUTER_TOP_HALF_SIZE[2]
    TARGET_HALF_SIZE = (0.04, 0.04, 0.002)
    TARGET_POS = (0.0, 0.78, OUTER_TOP_SURFACE_Z + BLOCK_HALF_EXTENT)
    HAND_INIT_POS = (0.0, 0.60, 0.25)
    DRAWER_HANDLE_CLOSED_Y = 0.615
    DRAWER_OPEN_THRESHOLD = 0.14
    DRAWER_MAX_TRAVEL = 0.20
    TARGET_XY_TOLERANCE = TARGET_HALF_SIZE[0] - BLOCK_HALF_EXTENT
    TARGET_Z_TOLERANCE = 0.008
    RELEASED_GRIPPER_APERTURE = 0.8
    SUCCESS_RADIUS = 0.032

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = "drawer_view",
        camera_id: int | None = None,
        height: int = 480,
        width: int = 480,
        seed: int | None = None,
        task_instance: int = 0,
    ) -> None:
        normalized_task_instance = _normalize_task_instance(task_instance)
        block_position, target_position = self.task_instance_reset_parameters(
            normalized_task_instance
        )
        super().__init__(
            hand_low=(-0.45, 0.35, 0.05),
            hand_high=(0.45, 1.02, 0.55),
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
            height=height,
            width=width,
        )

        self.task_name = self.TASK_NAME
        self._task_instance = normalized_task_instance
        self.hand_init_pos = np.asarray(self.HAND_INIT_POS, dtype=np.float64)
        self.obj_init_pos = block_position
        self._target_pos = target_position
        reset_vector = np.hstack((self.obj_init_pos, self._target_pos))
        self._random_reset_space = Box(
            reset_vector.copy(), reset_vector.copy(), dtype=np.float64
        )
        self.goal_space = Box(
            target_position - 1e-6, target_position + 1e-6, dtype=np.float64
        )

        # Local constructors are immediately usable; no benchmark/global task
        # registration is required.
        self._partially_observable = False
        self._set_task_called = True
        self._freeze_rand_vec = True
        self._last_rand_vec = reset_vector.copy()
        self.__dict__.pop("sawyer_observation_space", None)
        self.observation_space = self.sawyer_observation_space
        if seed is not None:
            self.seed(seed)

    @classmethod
    def task_instance_reset_parameters(
        cls, task_instance: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the deterministic block and goal positions for one instance."""

        normalized = _normalize_task_instance(task_instance)
        block_x_index = normalized % len(_BLOCK_X_LEVELS)
        block_y_index = normalized // len(_BLOCK_X_LEVELS)

        # 17 is coprime to 48, so this decorrelates the goal lattice while
        # visiting each goal exactly once.
        target_index = (
            _TARGET_PERMUTATION_MULTIPLIER * normalized
        ) % DRAWER_RETRIEVE_TASK_INSTANCE_COUNT
        target_x_index = target_index % len(_TARGET_X_LEVELS)
        target_y_index = target_index // len(_TARGET_X_LEVELS)
        block = np.asarray(
            (
                _BLOCK_X_LEVELS[block_x_index],
                _BLOCK_Y_LEVELS[block_y_index],
                cls.BLOCK_INIT_POS[2],
            ),
            dtype=np.float64,
        )
        target = np.asarray(
            (
                _TARGET_X_LEVELS[target_x_index],
                _TARGET_Y_LEVELS[target_y_index],
                cls.TARGET_POS[2],
            ),
            dtype=np.float64,
        )
        return block, target

    @property
    def task_instance(self) -> int:
        return self._task_instance

    @property
    def model_name(self) -> str:
        return resolved_drawer_retrieve_model_path()

    @property
    def model_template_path(self) -> Path:
        return _MODEL_TEMPLATE

    @property
    def block_position(self) -> npt.NDArray[np.float64]:
        return self.data.body("block").xpos.copy()

    @property
    def drawer_handle_position(self) -> npt.NDArray[np.float64]:
        return self.data.site("drawer_handle").xpos.copy()

    @property
    def drawer_open_amount(self) -> float:
        return float(self.data.joint("drawer_slide").qpos[0])

    @property
    def drawer_open_fraction(self) -> float:
        return float(np.clip(self.drawer_open_amount / self.DRAWER_OPEN_THRESHOLD, 0, 1))

    @property
    def gripper_aperture(self) -> float:
        left = self.data.body("leftclaw").xpos
        right = self.data.body("rightclaw").xpos
        return float(np.clip(np.linalg.norm(left - right) / 0.1, 0, 1))

    @property
    def block_to_target(self) -> float:
        assert self._target_pos is not None
        return float(np.linalg.norm(self.block_position - self._target_pos))

    @property
    def goal_marker_position(self) -> npt.NDArray[np.float64]:
        """Position the thin visual target flush with the shell top."""

        assert self._target_pos is not None
        marker = self._target_pos.copy()
        marker[2] = self.OUTER_TOP_SURFACE_Z + self.TARGET_HALF_SIZE[2]
        return marker

    @property
    def is_block_in_target(self) -> bool:
        assert self._target_pos is not None
        offset = np.abs(self.block_position - self._target_pos)
        return bool(
            np.all(offset[:2] <= self.TARGET_XY_TOLERANCE)
            and offset[2] <= self.TARGET_Z_TOLERANCE
        )

    @property
    def is_block_supported_by_outer_top(self) -> bool:
        object_geom_id = self._get_id_main_object()
        outer_top_geom_id = int(self.data.geom("outer_top").id)
        for contact in self.data.contact:
            geom_ids = {int(contact.geom1), int(contact.geom2)}
            if geom_ids == {object_geom_id, outer_top_geom_id}:
                return True
        return False

    @property
    def is_block_grasped(self) -> bool:
        return bool(
            self.gripper_aperture < self.RELEASED_GRIPPER_APERTURE
            and self.touching_main_object
        )

    @property
    def is_block_released(self) -> bool:
        return bool(
            self.gripper_aperture >= self.RELEASED_GRIPPER_APERTURE
            and not self.touching_main_object
        )

    def is_success(self) -> bool:
        return bool(
            self.drawer_open_amount >= self.DRAWER_OPEN_THRESHOLD
            and self.is_block_in_target
            and self.is_block_supported_by_outer_top
            and self.is_block_released
        )

    @property
    def success(self) -> bool:
        return self.is_success()

    def _get_id_main_object(self) -> int:
        return int(self.data.geom("objGeom").id)

    def _get_pos_objects(self) -> npt.NDArray[np.float64]:
        return np.hstack((self.block_position, self.drawer_handle_position))

    def _get_quat_objects(self) -> npt.NDArray[np.float64]:
        return np.hstack(
            (
                self.data.body("block").xquat.copy(),
                self.data.body("drawer").xquat.copy(),
            )
        )

    @property
    def _target_site_config(self) -> list[tuple[str, npt.NDArray[Any]]]:
        return [("goal", self.goal_marker_position)]

    def _set_block_pose(self, position: npt.ArrayLike) -> None:
        joint = self.data.joint("block_free")
        joint.qpos[:] = np.hstack(
            (np.asarray(position, dtype=np.float64), [1.0, 0.0, 0.0, 0.0])
        )
        joint.qvel[:] = 0.0

    def reset_model(self) -> npt.NDArray[np.float64]:
        self.set_state(self.init_qpos.copy(), self.init_qvel.copy())
        self._reset_hand()

        reset_vector = np.asarray(self._get_state_rand_vec(), dtype=np.float64)
        if reset_vector.shape != (6,):
            raise ValueError("drawer-retrieve reset vector must have shape (6,)")
        self.obj_init_pos = reset_vector[:3].copy()
        self._target_pos = reset_vector[3:].copy()

        drawer_joint = self.data.joint("drawer_slide")
        drawer_joint.qpos[0] = 0.0
        drawer_joint.qvel[0] = 0.0
        self._set_block_pose(self.obj_init_pos)
        self.model.site("goal").pos = self.goal_marker_position
        mujoco.mj_forward(self.model, self.data)

        self.init_tcp = self.tcp_center.copy()
        self.init_left_pad = self.get_body_com("leftpad").copy()
        self.init_right_pad = self.get_body_com("rightpad").copy()
        self._prev_obs = self._get_curr_obs_combined_no_goal()
        return self._get_obs()

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
        if seed is not None:
            self.seed(seed)
        observation, info = super().reset(seed=seed, options=options)
        reset_info = dict(info)
        reset_info.update(self._task_info())
        return observation, reset_info

    def _reward_components(self) -> tuple[float, float, float, float, float]:
        assert self.obj_init_pos is not None and self._target_pos is not None
        tcp_to_block = float(np.linalg.norm(self.tcp_center - self.block_position))
        block_to_target = self.block_to_target
        reach = float(
            reward_utils.tolerance(
                tcp_to_block,
                bounds=(0, 0.025),
                margin=max(float(np.linalg.norm(self.init_tcp - self.obj_init_pos)), 0.025),
                sigmoid="long_tail",
            )
        )
        in_place = float(
            reward_utils.tolerance(
                block_to_target,
                bounds=(0, self.SUCCESS_RADIUS),
                margin=max(
                    float(np.linalg.norm(self.obj_init_pos - self._target_pos)),
                    self.SUCCESS_RADIUS,
                ),
                sigmoid="long_tail",
            )
        )
        lift = float(np.clip((self.block_position[2] - self.obj_init_pos[2]) / 0.08, 0, 1))
        grasp = float(self.is_block_grasped)
        return tcp_to_block, block_to_target, reach, in_place, max(lift, grasp)

    def compute_reward(
        self,
        action: npt.NDArray[Any],
        obs: npt.NDArray[np.float64],
    ) -> tuple[float, float, float, float, float, float]:
        del action, obs
        tcp_to_block, block_to_target, reach, in_place, lift_or_grasp = (
            self._reward_components()
        )
        drawer = self.drawer_open_fraction
        reward = 2.0 * drawer
        if drawer >= 0.95:
            reward += 2.0 * reach + 2.0 * lift_or_grasp
            if lift_or_grasp >= 0.5:
                reward += 4.0 * in_place
        reward = float(np.clip(reward, 0, 9))
        if self.is_success():
            reward = 10.0
        return (
            reward,
            tcp_to_block,
            self.gripper_aperture,
            block_to_target,
            float(self.is_block_grasped),
            in_place,
        )

    def _task_info(self) -> dict[str, float | int]:
        return {
            "task_instance": self.task_instance,
            "success": float(self.is_success()),
            "drawer_open_amount": self.drawer_open_amount,
            "drawer_open_fraction": self.drawer_open_fraction,
            "block_to_target": self.block_to_target,
            "block_in_target": float(self.is_block_in_target),
            "block_supported": float(self.is_block_supported_by_outer_top),
            "block_released": float(self.is_block_released),
        }

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self,
        obs: npt.NDArray[np.float64],
        action: npt.NDArray[np.float32],
    ) -> tuple[float, dict[str, Any]]:
        (
            reward,
            tcp_to_block,
            gripper_aperture,
            block_to_target,
            grasp_reward,
            in_place_reward,
        ) = self.compute_reward(action, obs)
        info: dict[str, Any] = {
            "task_instance": self.task_instance,
            "success": float(self.is_success()),
            "near_object": float(tcp_to_block <= 0.04),
            "grasp_success": grasp_reward,
            "grasp_reward": grasp_reward,
            "in_place_reward": in_place_reward,
            "obj_to_target": block_to_target,
            "unscaled_reward": reward,
            "gripper_aperture": gripper_aperture,
            "drawer_open_amount": self.drawer_open_amount,
            "drawer_open_fraction": self.drawer_open_fraction,
            "block_to_target": block_to_target,
            "block_in_target": float(self.is_block_in_target),
            "block_supported": float(self.is_block_supported_by_outer_top),
            "block_released": float(self.is_block_released),
        }
        return reward, info


DrawerRetrieveEnv = SawyerDrawerRetrieveEnvV0
