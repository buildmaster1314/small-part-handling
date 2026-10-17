"""Stateless physical-state expert for drawer-retrieve-v0."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt


class DrawerRetrievePhase(str, Enum):
    PULL_OPEN = "pull_open"
    LIFT_AFTER_OPEN = "lift_after_open"
    MOVE_ABOVE_BLOCK = "move_above_block"
    DESCEND = "descend"
    GRASP = "grasp"
    LIFT_BLOCK = "lift_block"
    TRANSPORT_ABOVE = "transport_above"
    LOWER = "lower"
    RELEASE = "release"
    RETREAT = "retreat"


@dataclass(frozen=True, slots=True)
class DrawerRetrievePhysicalState:
    """Physical quantities encoded in one standard MetaWorld observation."""

    hand_pos: tuple[float, float, float]
    gripper_aperture: float
    block_pos: tuple[float, float, float]
    handle_pos: tuple[float, float, float]
    target_pos: tuple[float, float, float]

    @classmethod
    def from_observation(
        cls, observation: npt.ArrayLike
    ) -> "DrawerRetrievePhysicalState":
        obs = np.asarray(observation, dtype=np.float64)
        if obs.shape != (39,):
            raise ValueError(f"drawer-retrieve observation must have shape (39,), got {obs.shape}")
        if not np.isfinite(obs).all():
            raise ValueError("drawer-retrieve observation must contain only finite values")
        return cls(
            hand_pos=tuple(float(value) for value in obs[0:3]),
            gripper_aperture=float(obs[3]),
            block_pos=tuple(float(value) for value in obs[4:7]),
            handle_pos=tuple(float(value) for value in obs[11:14]),
            target_pos=tuple(float(value) for value in obs[-3:]),
        )


@dataclass(frozen=True, slots=True)
class DrawerRetrievePhysicalStateExpert:
    """Policy whose phase is recomputed only from current physical state."""

    open_handle_y: float = 0.475
    handle_lift_height: float = 0.26
    transport_clearance: float = 0.10
    xy_alignment_tolerance: float = 0.025
    transport_xy_tolerance: float = 0.016
    placement_xy_tolerance: float = 0.020
    placement_height_tolerance: float = 0.006
    grasp_height_tolerance: float = 0.04
    closed_aperture: float = 0.72

    @staticmethod
    def parse_observation(observation: npt.ArrayLike) -> DrawerRetrievePhysicalState:
        return DrawerRetrievePhysicalState.from_observation(observation)

    def phase(
        self, observation: npt.ArrayLike | DrawerRetrievePhysicalState
    ) -> DrawerRetrievePhase:
        state = (
            observation
            if isinstance(observation, DrawerRetrievePhysicalState)
            else self.parse_observation(observation)
        )
        hand = np.asarray(state.hand_pos)
        block = np.asarray(state.block_pos)
        handle = np.asarray(state.handle_pos)
        target = np.asarray(state.target_pos)

        if handle[1] > self.open_handle_y:
            return DrawerRetrievePhase.PULL_OPEN

        hand_to_handle_xy = float(np.linalg.norm(hand[:2] - handle[:2]))
        if hand_to_handle_xy <= 0.05 and hand[2] < self.handle_lift_height:
            return DrawerRetrievePhase.LIFT_AFTER_OPEN

        block_to_target_xy = float(np.linalg.norm(block[:2] - target[:2]))
        block_at_placement_height = bool(
            abs(block[2] - target[2]) <= self.placement_height_tolerance
        )
        block_at_placement = bool(
            block_to_target_xy <= self.placement_xy_tolerance
            and block_at_placement_height
        )
        if block_at_placement and state.gripper_aperture > self.closed_aperture:
            return DrawerRetrievePhase.RETREAT

        hand_to_block = float(np.linalg.norm(hand - block))
        inferred_grasp = (
            state.gripper_aperture <= self.closed_aperture and hand_to_block <= 0.10
        )
        if inferred_grasp:
            if block_at_placement:
                return DrawerRetrievePhase.RELEASE
            minimum_transport_height = target[2] + 0.75 * self.transport_clearance
            if (
                block[2] < minimum_transport_height
                and block_to_target_xy > self.transport_xy_tolerance
            ):
                return DrawerRetrievePhase.LIFT_BLOCK
            if block_to_target_xy > self.transport_xy_tolerance:
                return DrawerRetrievePhase.TRANSPORT_ABOVE
            if not block_at_placement_height:
                return DrawerRetrievePhase.LOWER
            return DrawerRetrievePhase.RELEASE

        if np.linalg.norm(hand[:2] - block[:2]) > self.xy_alignment_tolerance:
            return DrawerRetrievePhase.MOVE_ABOVE_BLOCK
        if hand[2] - block[2] > self.grasp_height_tolerance:
            return DrawerRetrievePhase.DESCEND
        return DrawerRetrievePhase.GRASP

    def get_phase(self, observation: npt.ArrayLike) -> DrawerRetrievePhase:
        return self.phase(observation)

    @staticmethod
    def _move(
        current: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
        gain: float,
    ) -> npt.NDArray[np.float32]:
        return np.clip(gain * (target - current), -1, 1).astype(np.float32)

    def get_action(self, observation: npt.ArrayLike) -> npt.NDArray[np.float32]:
        state = self.parse_observation(observation)
        phase = self.phase(state)
        hand = np.asarray(state.hand_pos, dtype=np.float64)
        block = np.asarray(state.block_pos, dtype=np.float64)
        handle = np.asarray(state.handle_pos, dtype=np.float64)
        target = np.asarray(state.target_pos, dtype=np.float64)
        hand_to_block = hand - block
        gripper = -1.0
        gain = 10.0

        if phase is DrawerRetrievePhase.PULL_OPEN:
            pull_height = handle[2] + 0.055
            if np.linalg.norm(hand[:2] - handle[:2]) > 0.035:
                desired = handle + np.array([0.0, 0.0, 0.15])
            elif abs(hand[2] - pull_height) > 0.015:
                desired = handle.copy()
                desired[2] = pull_height
            else:
                desired = handle + np.array([0.0, -0.09, 0.055])
                gain = 40.0
        elif phase is DrawerRetrievePhase.LIFT_AFTER_OPEN:
            desired = hand.copy()
            desired[2] = 0.30
        elif phase is DrawerRetrievePhase.MOVE_ABOVE_BLOCK:
            desired = block.copy()
            desired[2] = 0.28
        elif phase is DrawerRetrievePhase.DESCEND:
            desired = block.copy()
        elif phase is DrawerRetrievePhase.GRASP:
            desired = block.copy()
            gripper = 1.0
        elif phase is DrawerRetrievePhase.LIFT_BLOCK:
            desired_block = block.copy()
            desired_block[2] = target[2] + self.transport_clearance
            desired = desired_block + hand_to_block
            gripper = 1.0
        elif phase is DrawerRetrievePhase.TRANSPORT_ABOVE:
            desired_block = target.copy()
            desired_block[2] += self.transport_clearance
            desired = desired_block + hand_to_block
            gripper = 1.0
        elif phase is DrawerRetrievePhase.LOWER:
            desired = target + hand_to_block
            gripper = 1.0
            gain = 8.0
        elif phase is DrawerRetrievePhase.RELEASE:
            desired = target + hand_to_block
            gain = 8.0
        else:
            desired = target + np.array([0.0, 0.0, 0.18])

        action = np.empty(4, dtype=np.float32)
        action[:3] = self._move(hand, desired, gain)
        action[3] = gripper
        return action


DrawerRetrieveExpert = DrawerRetrievePhysicalStateExpert


def infer_drawer_retrieve_phase(observation: npt.ArrayLike) -> DrawerRetrievePhase:
    return DrawerRetrievePhysicalStateExpert().phase(observation)
