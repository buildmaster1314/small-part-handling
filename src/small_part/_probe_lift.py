"""Search for a friction-only lift. Not part of the task."""

from __future__ import annotations

import mujoco
import numpy as np

from small_part.control import actuator_index, joint_index, set_home
from small_part.tube_cycle import JOINTS, OPEN, _ranges, _set_friction, _tooth_midpoint
from small_part.tube_scene import (
    RACK_Z,
    SOURCE_ROW,
    TUBE_CENTER_Z,
    TUBE_HALF_LENGTH,
    hole_xy,
    load_tube_model,
    tube_position,
    tube_tilt,
)


def jaw_contact_count(model, data) -> int:
    n = 0
    for i in range(data.ncon):
        c = data.contact[i]
        if int(c.flex[0]) >= 0 or int(c.flex[1]) >= 0:
            n += 1
    return n


def run(label, *, pinch_lift, grip, friction, lift_style, approach_z_bias=0.0, close_steps=600):
    model, data = load_tube_model()
    actuators = [actuator_index(model, name) for name in JOINTS]
    arm = [joint_index(model, name) for name in JOINTS]
    tube = joint_index(model, "tube_free")
    source = hole_xy(SOURCE_ROW, 1)
    ranges = _ranges(model)
    slide, spin, roll, condim = friction
    _set_friction(model, slide, spin, roll, condim)

    mujoco.mj_resetData(model, data)
    set_home(model, data)
    data.qpos[tube : tube + 7] = (*source, TUBE_CENTER_Z, 1.0, 0.0, 0.0, 0.0)
    gi = joint_index(model, "gripper")
    data.qpos[gi] = OPEN
    data.ctrl[actuator_index(model, "gripper")] = OPEN
    mujoco.mj_forward(model, data)
    for _ in range(80):
        mujoco.mj_step(model, data)

    # seed from the recorded side pinch, but lower the hand by pinch_lift delta
    from small_part.tube_cycle import IN_SOCKET, PINCH_PAN, PICK_LIFT, ROLL

    seed = IN_SOCKET.copy()
    seed[0] = PINCH_PAN
    seed[1] = PICK_LIFT + pinch_lift
    seed[2] = IN_SOCKET[2] - pinch_lift  # keep pitch: flex cancels
    seed[3] = IN_SOCKET[3]
    # pitch of IN_SOCKET is lift+elbow+flex. We changed lift and elbow equally opposite? 
    # original pitch = IN_SOCKET[1]+IN_SOCKET[2]+IN_SOCKET[3]
    # want seed pitch unchanged, and lift = PICK_LIFT + pinch_lift
    # The recorded IN_SOCKET already is the pinch. PICK_LIFT is 0.18, IN_SOCKET[1] is 0.472.
    # Don't mix them. Use IN_SOCKET and only shift lift/elbow/flex to change height while keeping pitch.
    seed = IN_SOCKET.copy()
    seed[0] = PINCH_PAN
    seed[4] = ROLL
    seed[5] = OPEN
    seed[1] = float(IN_SOCKET[1] + pinch_lift)
    # To lower the hand, increase shoulder_lift (more positive folds down on this arm? probe said
    # lift goal[1] -= raises the tube, so decreasing shoulder_lift raises. Increasing lowers.)
    # Keep pitch: flex compensates. pitch = q1+q2+q3. If q1 increases, q3 decreases by the same.
    seed[3] = float(IN_SOCKET[3] - pinch_lift)

    start = np.array([data.ctrl[a] for a in actuators], float)
    goal = seed.copy()
    for i in range(1, 301):
        data.ctrl[actuators] = start + (goal - start) * i / 300
        mujoco.mj_step(model, data)

    pre_z = float(tube_position(data)[2])
    pre_tilt = tube_tilt(data)
    tooth = _tooth_midpoint(model, data)
    gap = tooth - tube_position(data)

    start = np.array([data.ctrl[a] for a in actuators], float)
    goal = start.copy()
    goal[5] = grip
    worst = pre_tilt
    for i in range(1, close_steps + 1):
        data.ctrl[actuators] = start + (goal - start) * i / close_steps
        mujoco.mj_step(model, data)
        worst = max(worst, tube_tilt(data))
    closed_z = float(tube_position(data)[2])
    closed_tilt = tube_tilt(data)
    njaw = jaw_contact_count(model, data)

    # lift
    start = np.array([data.ctrl[a] for a in actuators], float)
    lifted = False
    clear_z = RACK_Z + 0.074 + TUBE_HALF_LENGTH + 0.01  # bottom above the rack top, roughly
    # tube center needs to be above rack top + half length
    need = RACK_Z + 0.074 + TUBE_HALF_LENGTH + 0.008
    final_z = closed_z
    final_tilt = closed_tilt
    for step_i in range(1, 25):
        goal = np.array([data.ctrl[a] for a in actuators], float)
        if lift_style == "pitch_locked":
            goal[1] -= 0.018
            goal[2] -= 0.018
            goal[3] += 0.018
        elif lift_style == "shoulder":
            goal[1] -= 0.025
        elif lift_style == "elbow":
            goal[2] -= 0.03
            goal[3] += 0.03
        goal[5] = grip
        s0 = np.array([data.ctrl[a] for a in actuators], float)
        for i in range(1, 31):
            data.ctrl[actuators] = s0 + (goal - s0) * i / 30
            mujoco.mj_step(model, data)
            worst = max(worst, tube_tilt(data))
        final_z = float(tube_position(data)[2])
        final_tilt = tube_tilt(data)
        if final_z >= need and final_tilt < 0.10:
            lifted = True
            break
        if final_tilt > 0.35:
            break
        # if the tube stopped rising and is still in the hole, give up
        if step_i > 6 and final_z < pre_z + 0.01:
            break
    print(
        f"{label:22s} gap_z={gap[2]:+.3f} gap_x={gap[0]:+.3f} njaw={njaw:2d} "
        f"pre_z={pre_z:.3f} closed_z={closed_z:.3f} tilt_c={closed_tilt:.3f} "
        f"final_z={final_z:.3f} tilt={final_tilt:.3f} worst={worst:.3f} "
        f"need={need:.3f} LIFT={lifted}"
    )


def main():
    frictions = {
        "sticky": (2.4, 0.08, 0.02, 6),
        "mid": (1.2, 0.02, 0.002, 4),
        "rubber": (1.6, 0.05, 0.01, 6),
    }
    # pinch_lift > 0 lowers the hand (increases shoulder_lift command)
    configs = []
    for dz in (0.0, 0.08, 0.14, 0.20, 0.26):
        for grip in (0.02, -0.05, -0.12):
            for fname, fr in frictions.items():
                for style in ("pitch_locked",):
                    configs.append((dz, grip, fname, fr, style))
    # a smaller sweep first
    configs = []
    for dz in (0.0, 0.10, 0.18, 0.26):
        for grip in (0.02, -0.10):
            for fname in ("sticky", "rubber"):
                configs.append((dz, grip, fname, frictions[fname], "pitch_locked"))
    for dz, grip, fname, fr, style in configs:
        run(
            f"dz={dz:.2f} g={grip:+.2f} {fname}",
            pinch_lift=dz,
            grip=grip,
            friction=fr,
            lift_style=style,
        )


if __name__ == "__main__":
    main()
