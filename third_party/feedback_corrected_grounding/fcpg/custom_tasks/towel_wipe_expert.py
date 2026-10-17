"""Physical-state expert for three-round-trip towel wiping."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt


class TowelWipePhase(str, Enum):
    APPROACH_TOWEL = "approach_towel"
    DESCEND_TO_GRIP = "descend_to_grip"
    GRASP_TOWEL = "grasp_towel"
    LIFT_TOWEL = "lift_towel"
    LOWER_TO_START = "lower_to_start"
    WIPE_RIGHT = "wipe_right"
    WIPE_LEFT = "wipe_left"
    LIFT_AFTER_WIPE = "lift_after_wipe"


@dataclass(frozen=True, slots=True)
class TowelWipePhysicalState:
    hand_pos: tuple[float, float, float]
    gripper_aperture: float
    towel_pos: tuple[float, float, float]
    target_pos: tuple[float, float, float]

    @classmethod
    def from_observation(cls, observation: npt.ArrayLike) -> "TowelWipePhysicalState":
        obs = np.asarray(observation, dtype=np.float64)
        if obs.shape != (39,):
            raise ValueError(f"towel-wipe observation must have shape (39,), got {obs.shape}")
        if not np.isfinite(obs).all():
            raise ValueError("towel-wipe observation must contain only finite values")
        return cls(
            hand_pos=tuple(float(value) for value in obs[0:3]),
            gripper_aperture=float(obs[3]),
            towel_pos=tuple(float(value) for value in obs[4:7]),
            target_pos=tuple(float(value) for value in obs[-3:]),
        )


@dataclass(frozen=True, slots=True)
class TowelWipePhysicalStateExpert:
    """Physically grasp the towel, wipe three round trips, then lift it."""

    finish_height_threshold: float = 0.08
    endpoint_tolerance: float = 0.025
    grip_offset_z: float = 0.027
    xy_tolerance: float = 0.025
    grasp_z_tolerance: float = 0.030
    closed_aperture: float = 0.78
    initial_lift_height: float = 0.09

    @staticmethod
    def parse_observation(observation: npt.ArrayLike) -> TowelWipePhysicalState:
        return TowelWipePhysicalState.from_observation(observation)

    def phase(
        self, observation: npt.ArrayLike | TowelWipePhysicalState
    ) -> TowelWipePhase:
        state = (
            observation
            if isinstance(observation, TowelWipePhysicalState)
            else self.parse_observation(observation)
        )
        towel = np.asarray(state.towel_pos)
        target = np.asarray(state.target_pos)
        hand = np.asarray(state.hand_pos)
        grip = towel + np.asarray((0.0, 0.0, self.grip_offset_z))
        hand_to_grip = float(np.linalg.norm(hand - grip))
        inferred_grasp = bool(
            state.gripper_aperture <= self.closed_aperture and hand_to_grip <= 0.09
        )

        if not inferred_grasp:
            if np.linalg.norm(hand[:2] - grip[:2]) > self.xy_tolerance:
                return TowelWipePhase.APPROACH_TOWEL
            if hand[2] - grip[2] > self.grasp_z_tolerance:
                return TowelWipePhase.DESCEND_TO_GRIP
            return TowelWipePhase.GRASP_TOWEL
        if target[2] >= self.finish_height_threshold:
            if abs(target[0]) > 0.08:
                return TowelWipePhase.LIFT_TOWEL
            return TowelWipePhase.LIFT_AFTER_WIPE
        if np.allclose(target[:2], towel[:2], atol=self.endpoint_tolerance):
            return TowelWipePhase.LOWER_TO_START
        if target[0] > towel[0] + self.endpoint_tolerance:
            return TowelWipePhase.WIPE_RIGHT
        if target[0] < towel[0] - self.endpoint_tolerance:
            return TowelWipePhase.WIPE_LEFT
        return TowelWipePhase.LOWER_TO_START

    def get_phase(self, observation: npt.ArrayLike) -> TowelWipePhase:
        return self.phase(observation)

    def get_action(self, observation: npt.ArrayLike) -> npt.NDArray[np.float32]:
        state = self.parse_observation(observation)
        hand = np.asarray(state.hand_pos, dtype=np.float64)
        towel = np.asarray(state.towel_pos, dtype=np.float64)
        target = np.asarray(state.target_pos, dtype=np.float64)
        grip = towel + np.asarray((0.0, 0.0, self.grip_offset_z))
        hand_to_towel = hand - towel
        phase = self.phase(state)
        gripper = 1.0
        gain = 12.0

        if phase is TowelWipePhase.APPROACH_TOWEL:
            desired_hand = grip + np.asarray((0.0, 0.0, 0.10))
            gripper = -1.0
        elif phase is TowelWipePhase.DESCEND_TO_GRIP:
            desired_hand = grip.copy()
            gripper = -1.0
            gain = 8.0
        elif phase is TowelWipePhase.GRASP_TOWEL:
            desired_hand = grip.copy()
            gain = 6.0
        elif phase is TowelWipePhase.LIFT_TOWEL:
            desired_towel = towel.copy()
            desired_towel[2] = self.initial_lift_height + 0.03
            desired_hand = desired_towel + hand_to_towel
        elif phase is TowelWipePhase.LOWER_TO_START:
            desired_towel = target.copy()
            desired_hand = desired_towel + hand_to_towel
            gain = 8.0
        else:
            desired_hand = target + hand_to_towel
            gain = 18.0 if phase in {
                TowelWipePhase.WIPE_RIGHT,
                TowelWipePhase.WIPE_LEFT,
            } else 12.0
        delta = np.clip(gain * (desired_hand - hand), -1, 1)
        action = np.empty(4, dtype=np.float32)
        action[:3] = delta.astype(np.float32)
        action[3] = gripper
        return action


TowelWipeExpert = TowelWipePhysicalStateExpert


def infer_towel_wipe_phase(observation: npt.ArrayLike) -> TowelWipePhase:
    return TowelWipePhysicalStateExpert().phase(observation)
