"""Repo-local persistent towel wiping task for the A4 panel."""

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


TOWEL_WIPE_TASK_NAME = "towel-wipe-v0"
TOWEL_WIPE_TASK_INSTANCE_COUNT = 48
_HALF_SPANS = (0.15, 0.17, 0.19, 0.21, 0.16, 0.18, 0.20, 0.22)
_LANE_Y_LEVELS = (0.50, 0.54, 0.58, 0.62, 0.66, 0.70)
_ASSET_DIR_TOKEN = "__METAWORLD_ASSET_DIR__"
_MODEL_TEMPLATE = (
    Path(__file__).resolve().parents[1] / "assets" / "towel_wipe" / "towel_wipe.xml"
)


def _normalize_task_instance(task_instance: int) -> int:
    if isinstance(task_instance, (bool, np.bool_)):
        raise TypeError("task_instance must be an integer, not bool")
    try:
        normalized = operator.index(task_instance)
    except TypeError as exc:
        raise TypeError("task_instance must be an integer") from exc
    if not 0 <= normalized < TOWEL_WIPE_TASK_INSTANCE_COUNT:
        raise ValueError(
            "task_instance must be in the inclusive range "
            f"0..{TOWEL_WIPE_TASK_INSTANCE_COUNT - 1}"
        )
    return normalized


@lru_cache(maxsize=1)
def resolved_towel_wipe_model_path() -> str:
    """Return a loadable MJCF copy with MetaWorld asset paths resolved."""

    template = _MODEL_TEMPLATE.read_text(encoding="utf-8")
    if _ASSET_DIR_TOKEN not in template:
        raise RuntimeError(f"Missing asset token in {_MODEL_TEMPLATE}")
    asset_dir = escape(str(Path(ENV_ASSET_DIR_V3).resolve()), {'"': "&quot;"})
    resolved = template.replace(_ASSET_DIR_TOKEN, asset_dir)
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    cache_dir = Path(tempfile.gettempdir()) / "fcpg_towel_wipe" / digest
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "towel_wipe.xml"
    if not model_path.exists():
        temporary_path = cache_dir / f"towel_wipe.{os.getpid()}.tmp"
        temporary_path.write_text(resolved, encoding="utf-8")
        os.replace(temporary_path, model_path)
    return str(model_path)


class SawyerTowelWipeEnvV0(SawyerXYZEnv):
    """Wipe a table with three complete left-right towel round trips."""

    TASK_NAME = TOWEL_WIPE_TASK_NAME
    TASK_INSTANCE_COUNT = TOWEL_WIPE_TASK_INSTANCE_COUNT
    REQUIRED_HALF_CYCLES = 6
    WIPE_HEIGHT = 0.012
    ENDPOINT_RADIUS = 0.035
    SURFACE_MAX_HEIGHT = 0.035
    FINISH_HEIGHT = 0.16
    SUCCESS_LIFT_HEIGHT = 0.10
    INITIAL_GRASP_LIFT_HEIGHT = 0.08
    GRIP_OFFSET = np.asarray((0.0, 0.0, 0.027), dtype=np.float64)
    GRASPED_APERTURE = 0.82

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = "towel_view",
        camera_id: int | None = None,
        height: int = 480,
        width: int = 480,
        seed: int | None = None,
        task_instance: int = 0,
    ) -> None:
        normalized = _normalize_task_instance(task_instance)
        left, right = self.task_instance_endpoints(normalized)
        super().__init__(
            hand_low=(-0.45, 0.34, 0.045),
            hand_high=(0.45, 0.92, 0.48),
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
            height=height,
            width=width,
        )
        self.task_name = self.TASK_NAME
        self._task_instance = normalized
        self._left_endpoint = left
        self._right_endpoint = right
        self.hand_init_pos = left + np.asarray((0.0, 0.0, 0.20))
        self.obj_init_pos = left.copy()
        self._target_pos = self._initial_lift_target()
        reset_vector = np.hstack((left, right))
        self._random_reset_space = Box(
            reset_vector.copy(), reset_vector.copy(), dtype=np.float64
        )
        self.goal_space = Box(
            np.asarray((-0.24, 0.45, 0.005), dtype=np.float64),
            np.asarray((0.24, 0.75, self.FINISH_HEIGHT), dtype=np.float64),
            dtype=np.float64,
        )
        self._wipe_started = False
        self._wipe_half_cycles = 0
        self._grasp_lifted = False
        self._partially_observable = False
        self._set_task_called = True
        self._freeze_rand_vec = True
        self._last_rand_vec = reset_vector.copy()
        self.__dict__.pop("sawyer_observation_space", None)
        self.observation_space = self.sawyer_observation_space
        if seed is not None:
            self.seed(seed)

    @classmethod
    def task_instance_endpoints(
        cls, task_instance: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        normalized = _normalize_task_instance(task_instance)
        half_span = _HALF_SPANS[normalized % len(_HALF_SPANS)]
        lane_y = _LANE_Y_LEVELS[normalized // len(_HALF_SPANS)]
        left = np.asarray((-half_span, lane_y, cls.WIPE_HEIGHT), dtype=np.float64)
        right = np.asarray((half_span, lane_y, cls.WIPE_HEIGHT), dtype=np.float64)
        return left, right

    @property
    def task_instance(self) -> int:
        return self._task_instance

    @property
    def model_name(self) -> str:
        return resolved_towel_wipe_model_path()

    @property
    def model_template_path(self) -> Path:
        return _MODEL_TEMPLATE

    @property
    def towel_position(self) -> npt.NDArray[np.float64]:
        return self.data.body("towel").xpos.copy()

    @property
    def towel_grip_position(self) -> npt.NDArray[np.float64]:
        return self.data.geom("objGeom").xpos.copy()

    @property
    def gripper_aperture(self) -> float:
        left = self.data.body("leftclaw").xpos
        right = self.data.body("rightclaw").xpos
        return float(np.clip(np.linalg.norm(left - right) / 0.1, 0, 1))

    @property
    def towel_grasped(self) -> bool:
        return bool(
            self.gripper_aperture <= self.GRASPED_APERTURE
            and self.touching_main_object
        )

    @property
    def left_endpoint(self) -> npt.NDArray[np.float64]:
        return self._left_endpoint.copy()

    @property
    def right_endpoint(self) -> npt.NDArray[np.float64]:
        return self._right_endpoint.copy()

    @property
    def wipe_half_cycles(self) -> int:
        return int(self._wipe_half_cycles)

    @property
    def wipe_round_trips(self) -> float:
        return float(self._wipe_half_cycles / 2.0)

    @property
    def towel_surface_engaged(self) -> bool:
        rotation = self.data.body("towel").xmat.reshape(3, 3)
        upright = float(rotation[2, 2]) >= 0.75
        lane_error = abs(float(self.towel_position[1] - self._left_endpoint[1]))
        return bool(
            self.towel_position[2] <= self.SURFACE_MAX_HEIGHT
            and lane_error <= 0.09
            and upright
            and self.towel_grasped
        )

    @property
    def current_target_side(self) -> str:
        if self._wipe_half_cycles >= self.REQUIRED_HALF_CYCLES:
            return "finish"
        if np.allclose(self._target_pos, self._left_endpoint):
            return "left"
        return "right"

    def is_success(self) -> bool:
        return bool(
            self._wipe_half_cycles >= self.REQUIRED_HALF_CYCLES
            and self.towel_position[2] >= self.SUCCESS_LIFT_HEIGHT
            and self.towel_grasped
        )

    @property
    def success(self) -> bool:
        return self.is_success()

    def _get_id_main_object(self) -> int:
        return int(self.data.geom("objGeom").id)

    def _get_pos_objects(self) -> npt.NDArray[np.float64]:
        return self.towel_position

    def _get_quat_objects(self) -> npt.NDArray[np.float64]:
        return self.data.body("towel").xquat.copy()

    def _set_visual_markers(self) -> None:
        self.model.site("wipe_left").pos = self._left_endpoint
        self.model.site("wipe_right").pos = self._right_endpoint
        assert self._target_pos is not None
        self.model.site("goal").pos = self._target_pos

    def _set_towel_pose(self, position: npt.ArrayLike) -> None:
        joint = self.data.joint("towel_free")
        joint.qpos[:] = np.hstack(
            (np.asarray(position, dtype=np.float64), [1.0, 0.0, 0.0, 0.0])
        )
        joint.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _finish_target(self) -> npt.NDArray[np.float64]:
        return np.asarray(
            (0.0, float(self._left_endpoint[1]), self.FINISH_HEIGHT),
            dtype=np.float64,
        )

    def _initial_lift_target(self) -> npt.NDArray[np.float64]:
        target = self._left_endpoint.copy()
        target[2] = self.INITIAL_GRASP_LIFT_HEIGHT + 0.03
        return target

    def _advance_wipe_state(self) -> None:
        if (
            not self._grasp_lifted
            and self.towel_grasped
            and self.towel_position[2] >= self.INITIAL_GRASP_LIFT_HEIGHT
        ):
            self._grasp_lifted = True
            self._target_pos = self._left_endpoint.copy()
            self._set_visual_markers()
        if self._wipe_half_cycles >= self.REQUIRED_HALF_CYCLES:
            self._target_pos = self._finish_target()
            return
        assert self._target_pos is not None
        target_error = float(np.linalg.norm(self.towel_position[:2] - self._target_pos[:2]))
        if (
            not self._grasp_lifted
            or not self.towel_surface_engaged
            or target_error > self.ENDPOINT_RADIUS
        ):
            return
        if not self._wipe_started:
            self._wipe_started = True
            self._target_pos = self._right_endpoint.copy()
            return
        self._wipe_half_cycles += 1
        if self._wipe_half_cycles >= self.REQUIRED_HALF_CYCLES:
            self._target_pos = self._finish_target()
        elif self._wipe_half_cycles % 2:
            self._target_pos = self._left_endpoint.copy()
        else:
            self._target_pos = self._right_endpoint.copy()
        self._set_visual_markers()

    def reset_model(self) -> npt.NDArray[np.float64]:
        self.set_state(self.init_qpos.copy(), self.init_qvel.copy())
        reset_vector = np.asarray(self._get_state_rand_vec(), dtype=np.float64)
        if reset_vector.shape != (6,):
            raise ValueError("towel-wipe reset vector must have shape (6,)")
        self._left_endpoint = reset_vector[:3].copy()
        self._right_endpoint = reset_vector[3:].copy()
        self.hand_init_pos = self._left_endpoint + np.asarray((0.0, 0.0, 0.20))
        self._target_pos = self._initial_lift_target()
        self._wipe_started = False
        self._wipe_half_cycles = 0
        self._grasp_lifted = False
        self._set_visual_markers()
        self._reset_hand()
        self._set_towel_pose(self._left_endpoint)
        self.obj_init_pos = self.towel_position
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

    def _target_distance(self) -> float:
        assert self._target_pos is not None
        dimensions = slice(None) if self.current_target_side == "finish" else slice(0, 2)
        return float(np.linalg.norm(self.towel_position[dimensions] - self._target_pos[dimensions]))

    def compute_reward(
        self,
        action: npt.NDArray[Any],
        obs: npt.NDArray[np.float64],
    ) -> tuple[float, float, float]:
        del action, obs
        target_distance = self._target_distance()
        target_score = float(
            reward_utils.tolerance(
                target_distance,
                bounds=(0, self.ENDPOINT_RADIUS),
                margin=0.45,
                sigmoid="long_tail",
            )
        )
        progress = float(self._wipe_half_cycles / self.REQUIRED_HALF_CYCLES)
        reward = 1.5 * float(self._wipe_started) + 6.5 * progress + 2.0 * target_score
        reward = float(np.clip(reward, 0, 10))
        if self.is_success():
            reward = 10.0
        return reward, target_distance, target_score

    def _task_info(self) -> dict[str, float | int | str]:
        return {
            "task_instance": self.task_instance,
            "success": float(self.is_success()),
            "wipe_started": int(self._wipe_started),
            "wipe_half_cycles": self.wipe_half_cycles,
            "wipe_round_trips": self.wipe_round_trips,
            "towel_grasped": float(self.towel_grasped),
            "grasp_lifted": float(self._grasp_lifted),
            "towel_surface_engaged": float(self.towel_surface_engaged),
            "current_target_side": self.current_target_side,
        }

    def _evaluation_info(
        self, reward: float, target_distance: float, target_score: float
    ) -> dict[str, Any]:
        return {
            **self._task_info(),
            "near_object": float(
                np.linalg.norm(self.tcp_center - self.towel_grip_position) <= 0.06
            ),
            "grasp_success": float(self.towel_grasped),
            "grasp_reward": float(self.towel_grasped),
            "in_place_reward": float(
                self._wipe_half_cycles / self.REQUIRED_HALF_CYCLES
            ),
            "obj_to_target": target_distance,
            "unscaled_reward": reward,
            "target_score": target_score,
        }

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self,
        obs: npt.NDArray[np.float64],
        action: npt.NDArray[np.float32],
    ) -> tuple[float, dict[str, Any]]:
        reward, target_distance, target_score = self.compute_reward(action, obs)
        return reward, self._evaluation_info(reward, target_distance, target_score)

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float64], float, bool, bool, dict[str, Any]]:
        observation, _reward, terminated, truncated, _info = super().step(action)
        self._advance_wipe_state()
        self._set_visual_markers()
        assert self._target_pos is not None
        observation = np.asarray(observation, dtype=np.float64).copy()
        observation[4:7] = self.towel_position
        observation[7:11] = self._get_quat_objects()
        observation[-3:] = self._target_pos
        reward, target_distance, target_score = self.compute_reward(action, observation)
        info = self._evaluation_info(reward, target_distance, target_score)
        if self._last_stable_obs is not None:
            self._last_stable_obs = observation.copy()
        return observation, float(reward), terminated, truncated, info


TowelWipeEnv = SawyerTowelWipeEnvV0
