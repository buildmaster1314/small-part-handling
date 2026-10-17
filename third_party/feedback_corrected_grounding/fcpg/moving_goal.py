"""Smooth, task-independent moving-goal support for manipulation pilots.

The wrapper deliberately owns only the exogenous goal trajectory.  Robot state,
actions, rewards, and observations remain owned by the underlying environment.
This makes it possible to compare a fixed goal and a moving goal without
changing the policy or the MuJoCo dynamics implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from numbers import Integral, Real
from typing import Any

import numpy as np


MOVING_GOAL_TASKS = ("reach-v3", "push-v3", "pick-place-v3")
DEFAULT_AMPLITUDE_XY = (0.02, 0.01)
DEFAULT_PERIOD_RAW_STEPS = 220.0


def _finite_vector(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must have finite shape {shape}")
    return result.copy()


def _positive_real(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


@dataclass(frozen=True)
class MovingGoalConfig:
    """Configuration for one deterministic smooth target trajectory."""

    task: str
    seed: int
    amplitude_xy: tuple[float, float] = DEFAULT_AMPLITUDE_XY
    period_raw_steps: float = DEFAULT_PERIOD_RAW_STEPS
    workspace_xy: tuple[tuple[float, float], tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        if self.task not in MOVING_GOAL_TASKS:
            raise ValueError(
                f"moving goals support only {MOVING_GOAL_TASKS}, got {self.task!r}"
            )
        object.__setattr__(self, "seed", _nonnegative_int(self.seed, "seed"))
        if len(self.amplitude_xy) != 2:
            raise ValueError("amplitude_xy must contain two values")
        amplitude = tuple(
            _positive_real(value, f"amplitude_xy[{index}]")
            for index, value in enumerate(self.amplitude_xy)
        )
        object.__setattr__(self, "amplitude_xy", amplitude)
        object.__setattr__(
            self,
            "period_raw_steps",
            _positive_real(self.period_raw_steps, "period_raw_steps"),
        )
        if self.workspace_xy is not None:
            if len(self.workspace_xy) != 2:
                raise ValueError("workspace_xy must contain x and y bounds")
            bounds: list[tuple[float, float]] = []
            for index, bound in enumerate(self.workspace_xy):
                if len(bound) != 2:
                    raise ValueError(f"workspace_xy[{index}] must contain two values")
                low = float(bound[0])
                high = float(bound[1])
                if not np.isfinite([low, high]).all() or low >= high:
                    raise ValueError(f"workspace_xy[{index}] must be finite low < high")
                bounds.append((low, high))
            object.__setattr__(self, "workspace_xy", tuple(bounds))


class SmoothMovingGoal:
    """Generate a bounded, phase-randomized, smooth target trajectory."""

    def __init__(self, config: MovingGoalConfig) -> None:
        if not isinstance(config, MovingGoalConfig):
            raise TypeError("config must be MovingGoalConfig")
        self.config = config
        self._base_target: np.ndarray | None = None
        self._phase = 0.0
        self._initial_xy = np.zeros(2, dtype=np.float64)
        self._push_direction_xy = np.asarray([0.0, 1.0], dtype=np.float64)

    def reset(self, base_target: Any, *, reference_xy: Any | None = None) -> np.ndarray:
        self._base_target = _finite_vector(base_target, (3,), "base_target")
        rng = np.random.default_rng(self.config.seed)
        self._phase = float(rng.uniform(0.0, 2.0 * pi))
        self._initial_xy = np.asarray(
            [sin(self._phase), cos(self._phase)], dtype=np.float64
        )
        if self.config.task == "push-v3" and reference_xy is not None:
            reference = _finite_vector(reference_xy, (2,), "reference_xy")
            direction = self._base_target[:2] - reference
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-6:
                raise ValueError("push reference must differ from the base target")
            self._push_direction_xy = direction / norm
        return self.position(0)

    @property
    def base_target(self) -> np.ndarray:
        if self._base_target is None:
            raise RuntimeError("moving goal has not been reset")
        return self._base_target.copy()

    @property
    def phase(self) -> float:
        return self._phase

    def position(self, raw_step: int) -> np.ndarray:
        step = _nonnegative_int(raw_step, "raw_step")
        if self._base_target is None:
            raise RuntimeError("moving goal has not been reset")
        if self.config.task == "push-v3":
            progress = min(float(step) / self.config.period_raw_steps, 1.0)
            smooth = progress * progress * (3.0 - 2.0 * progress)
            tangent = self._push_direction_xy
            normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
            offset = (
                tangent * self.config.amplitude_xy[0] * smooth
                + normal * self.config.amplitude_xy[1] * sin(pi * smooth)
            )
        else:
            theta = (
                self._phase
                + 2.0 * pi * float(step) / self.config.period_raw_steps
            )
            # Subtract the reset phase so position(0) is the original target.
            offset = np.asarray(
                [
                    self.config.amplitude_xy[0]
                    * (sin(theta) - self._initial_xy[0]),
                    self.config.amplitude_xy[1]
                    * (cos(theta) - self._initial_xy[1]),
                ],
                dtype=np.float64,
            )
        target = self._base_target.copy()
        target[:2] += offset
        if self.config.workspace_xy is not None:
            target[0] = np.clip(
                target[0],
                self.config.workspace_xy[0][0],
                self.config.workspace_xy[0][1],
            )
            target[1] = np.clip(
                target[1],
                self.config.workspace_xy[1][0],
                self.config.workspace_xy[1][1],
            )
        return target

    def state_dict(self) -> dict[str, Any]:
        if self._base_target is None:
            raise RuntimeError("moving goal has not been reset")
        return {
            "config": self.config,
            "base_target": self._base_target.copy(),
            "phase": float(self._phase),
            "initial_xy": self._initial_xy.copy(),
            "push_direction_xy": self._push_direction_xy.copy(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("moving goal state must be a dict")
        if state.get("config") != self.config:
            raise ValueError("moving goal state config does not match")
        self._base_target = _finite_vector(state["base_target"], (3,), "base_target")
        self._phase = float(state["phase"])
        if not np.isfinite(self._phase):
            raise ValueError("moving goal phase must be finite")
        self._initial_xy = _finite_vector(state["initial_xy"], (2,), "initial_xy")
        self._push_direction_xy = _finite_vector(
            state.get("push_direction_xy", [0.0, 1.0]),
            (2,),
            "push_direction_xy",
        )
        direction_norm = float(np.linalg.norm(self._push_direction_xy))
        if direction_norm <= 1e-6:
            raise ValueError("push direction must be non-zero")
        self._push_direction_xy /= direction_norm


class MovingGoalEnv:
    """Gym-style proxy that updates a MetaWorld goal before each transition.

    The action, reward, and success for transition ``t`` all use ``goal_t``.
    After the transition, the wrapper installs ``goal_{t+1}`` and patches only
    the goal tail of the returned observation.  It never calls ``_get_obs``
    merely to refresh the goal because doing so mutates MetaWorld's frame stack.
    """

    def __init__(
        self,
        env: Any,
        *,
        task: str,
        seed: int,
        amplitude_xy: tuple[float, float] = DEFAULT_AMPLITUDE_XY,
        period_raw_steps: float = DEFAULT_PERIOD_RAW_STEPS,
    ) -> None:
        if task not in MOVING_GOAL_TASKS:
            raise ValueError(f"unsupported moving-goal task {task!r}")
        if env is None:
            raise TypeError("env must not be None")
        self.env = env
        self.unwrapped = getattr(env, "unwrapped", env)
        self.task = task
        self.config = MovingGoalConfig(
            task=task,
            seed=seed,
            amplitude_xy=amplitude_xy,
            period_raw_steps=period_raw_steps,
        )
        self.controller = SmoothMovingGoal(self.config)
        self.raw_step = 0
        self._last_observation: np.ndarray | None = None

    def _refresh_observation(self, observation: Any) -> np.ndarray:
        result = np.asarray(observation, dtype=np.float32).copy()
        if result.ndim != 1 or result.size < 3:
            raise ValueError("moving-goal environment observation must be a 1D vector")
        result[-3:] = np.asarray(self.env._target_pos, dtype=np.float32)
        self._last_observation = result.copy()
        return result

    def _apply_target(self, target: np.ndarray) -> None:
        target = _finite_vector(target, (3,), "target")
        current = np.asarray(getattr(self.env, "_target_pos", None))
        if current.shape != (3,):
            raise ValueError("underlying _target_pos must have shape [3]")
        # Reach aliases _target_pos into _last_rand_vec after reset.  Break that
        # alias so moving the runtime target does not rewrite task identity.
        self.env._target_pos = target.copy()
        model = getattr(self.env, "model", None)
        if model is not None:
            try:
                model.site("goal").pos = target
            except (AttributeError, KeyError, TypeError, ValueError):
                pass
        set_site = getattr(self.env, "_set_pos_site", None)
        if callable(set_site):
            try:
                set_site("goal", target)
            except (AttributeError, KeyError, TypeError, ValueError):
                pass

    def reset(self, *, seed: int | None = None, options: Any = None):
        kwargs: dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = int(seed)
        if options is not None:
            kwargs["options"] = options
        try:
            result = self.env.reset(**kwargs)
        except TypeError:
            kwargs.pop("options", None)
            result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            observation, info = result
        else:
            observation, info = result, {}
        base_target = _finite_vector(self.env._target_pos, (3,), "_target_pos")
        values = np.asarray(observation, dtype=np.float64)
        reference_xy = values[4:6] if self.task == "push-v3" and values.size >= 7 else None
        self.controller.reset(base_target, reference_xy=reference_xy)
        self.raw_step = 0
        self._apply_target(self.controller.position(0))
        observation = self._refresh_observation(observation)
        info = dict(info) if isinstance(info, dict) else {}
        info.update(self.moving_goal_info())
        return observation, info

    def step(self, action: Any):
        reward_target = np.asarray(self.env._target_pos, dtype=np.float32).copy()
        result = self.env.step(action)
        if not isinstance(result, tuple) or len(result) != 5:
            raise ValueError("underlying moving-goal environment must return a 5-tuple")
        observation, reward, terminated, truncated, info = result
        self.raw_step += 1
        self._apply_target(self.controller.position(self.raw_step))
        observation = self._refresh_observation(observation)
        info = dict(info) if isinstance(info, dict) else {}
        info.update(self.moving_goal_info())
        info["moving_goal_reward_position"] = reward_target
        return observation, reward, terminated, truncated, info

    def moving_goal_info(self) -> dict[str, Any]:
        return {
            "moving_goal": True,
            "moving_goal_task": self.task,
            "moving_goal_raw_step": int(self.raw_step),
            "moving_goal_position": np.asarray(
                self.env._target_pos, dtype=np.float32
            ).copy(),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "raw_step": int(self.raw_step),
            "controller": self.controller.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or state.get("task") != self.task:
            raise ValueError("moving-goal environment state task mismatch")
        self.controller.load_state_dict(state["controller"])
        self.raw_step = _nonnegative_int(state["raw_step"], "raw_step")
        self._apply_target(self.controller.position(self.raw_step))
        if self._last_observation is not None:
            self._refresh_observation(self._last_observation)

    def close(self) -> Any:
        return self.env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


class MovingPushExpert:
    """Reactive demonstration controller that can re-approach a moving puck."""

    def __init__(
        self,
        *,
        success_radius: float = 0.04,
        behind_distance: float = 0.045,
    ) -> None:
        self.success_radius = _positive_real(success_radius, "success_radius")
        self.behind_distance = _positive_real(
            behind_distance, "behind_distance"
        )

    def get_action(self, observation: Any) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float64)
        if values.ndim != 1 or values.size < 10 or not np.isfinite(values).all():
            raise ValueError("moving-push observation must be a finite 1D vector")
        hand = values[:3]
        puck = values[4:7]
        goal = values[-3:]
        displacement = goal[:2] - puck[:2]
        distance = float(np.linalg.norm(displacement))
        direction = (
            displacement / distance
            if distance > 1e-6
            else np.asarray([0.0, 1.0], dtype=np.float64)
        )
        behind = puck[:2] - self.behind_distance * direction
        lateral_error = float(np.linalg.norm(hand[:2] - behind))
        hover_z = float(puck[2] + 0.13)
        contact_z = float(puck[2] + 0.025)

        if distance <= self.success_radius:
            desired = np.asarray([behind[0], behind[1], hover_z])
        elif lateral_error > 0.018:
            if hand[2] < puck[2] + 0.09:
                desired = np.asarray([hand[0], hand[1], hover_z])
            else:
                desired = np.asarray([behind[0], behind[1], hover_z])
        elif abs(float(hand[2]) - contact_z) > 0.018:
            desired = np.asarray([behind[0], behind[1], contact_z])
        else:
            lead = goal[:2] + 0.02 * direction
            desired = np.asarray([lead[0], lead[1], contact_z])

        action = np.empty(4, dtype=np.float32)
        action[:3] = np.clip(10.0 * (desired - hand), -1.0, 1.0)
        action[3] = 0.6
        return action


__all__ = [
    "DEFAULT_AMPLITUDE_XY",
    "DEFAULT_PERIOD_RAW_STEPS",
    "MOVING_GOAL_TASKS",
    "MovingGoalConfig",
    "MovingGoalEnv",
    "MovingPushExpert",
    "SmoothMovingGoal",
]
