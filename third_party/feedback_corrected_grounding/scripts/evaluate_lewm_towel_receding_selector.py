#!/usr/bin/env python3
"""Run a causal H6 Towel selector with a frozen history-conditioned LeWM."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
import html
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.custom_tasks.towel_wipe import (  # noqa: E402
    SawyerTowelWipeEnvV0,
    TOWEL_WIPE_TASK_NAME,
)
from fcpg.custom_tasks.towel_wipe_expert import TowelWipePhase  # noqa: E402
from fcpg.lewm_action_branch_a4 import TASKS  # noqa: E402
from fcpg.lewm_action_branch_a4_collection import (  # noqa: E402
    make_a4_episode_runtime,
)
from fcpg.lewm_counterfactual_runtime import (  # noqa: E402
    corrected_points,
    grounded_valid_mask,
)
from fcpg.lewm_orbit_camera import OrbitCameraView  # noqa: E402
from fcpg.lewm_random_hazard_a4 import (  # noqa: E402
    RandomHazardSchedule,
    sample_random_hazard_schedule,
)
from fcpg.lewm_towel_receding_selector import (  # noqa: E402
    CANDIDATE_NAMES,
    POLICY_CANDIDATE_NAMES,
    PRIMARY_HORIZON_INDEX,
    build_goal_centered_prefix,
    build_policy_centered_towel_candidates,
    build_policy_repeated_prefix,
    build_towel_candidate_prefixes,
    causal_history_chunks,
    center_calibrated_world_forecast,
    integrated_cartesian_action_forecast,
    kinematic_world_goal_cost,
    lewm_world_goal_cost,
    selected_raw_chunk,
    tool_xy_goal_reach_cost,
)
from fcpg.mujoco_camera import (  # noqa: E402
    get_camera_id,
    set_fixed_camera_pose_offset,
)
from fcpg.transformer_grounding import load_fcpg_vit_adapter  # noqa: E402
from scripts import render_lewm_action_prefix_joint_a4_accuracy_gallery as gallery  # noqa: E402


DEFAULT_CHECKPOINT = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "camera_action_metric3d_world_history_seed40583_10k/"
    "lewm_history_conditioned_latest.pt"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "towel_h6_receding_selector_smoke_clock03"
)
DEFAULT_FCPG_CHECKPOINT = ROOT / (
    "outputs/transformer_residual/a4_multiview_projection_seed40601_2k/"
    "fcpg_a4_multiview_best.pt"
)
CONTROLLERS = (
    "center",
    "kinematic",
    "lewm",
    "lewm_residual",
    "lewm_nominal",
    "lewm_fcpg",
    "lewm_fcpg_corrupt",
    "lewm_fcpg_hold",
    "lewm_fcpg_prior",
)
INTERRUPTION_CONTROLLERS = frozenset(
    {"lewm_fcpg_corrupt", "lewm_fcpg_hold", "lewm_fcpg_prior"}
)
FCPG_CONTROLLERS = frozenset({"lewm_fcpg", *INTERRUPTION_CONTROLLERS})
WIPE_PHASES = {TowelWipePhase.WIPE_LEFT.value, TowelWipePhase.WIPE_RIGHT.value}
RAW_CHUNK = 5


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--fcpg-checkpoint", type=Path, default=DEFAULT_FCPG_CHECKPOINT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--task-instance-start", type=int, default=36)
    parser.add_argument("--environment-seed-base", type=int, default=160_000)
    parser.add_argument("--benchmark-seed", type=int, default=30_103)
    parser.add_argument("--clock-hour", type=int, default=3)
    parser.add_argument("--camera-rotation-degrees", type=float, default=0.0)
    parser.add_argument(
        "--camera-rotation-axis-world",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 1.0),
    )
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--max-raw-steps", type=int, default=500)
    parser.add_argument(
        "--candidate-source",
        choices=("goal-centered", "policy-centered"),
        default="goal-centered",
    )
    parser.add_argument("--controllers", nargs="+", choices=CONTROLLERS, default=CONTROLLERS)
    parser.add_argument("--frame-duration-ms", type=int, default=350)
    parser.add_argument("--hazard-seed", type=int, default=176_021)
    parser.add_argument("--hazard-probability", type=float, default=0.12)
    parser.add_argument("--hazard-min-duration", type=int, default=2)
    parser.add_argument("--hazard-max-duration", type=int, default=8)
    parser.add_argument("--hazard-max-events", type=int, default=3)
    parser.add_argument("--hazard-schedule-length", type=int, default=96)
    parser.add_argument("--occluder-radius-px", type=int, default=22)
    parser.add_argument("--occluder-blur-radius", type=float, default=4.0)
    return parser


def _cuda0(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("Towel selector permits only CUDA0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def _phase(expert: Any, observation: np.ndarray) -> str:
    value = expert.get_phase(observation)
    return str(getattr(value, "value", value))


def _bounded_expert_action(runtime: Any, observation: np.ndarray) -> np.ndarray:
    action = np.asarray(runtime.expert.get_action(observation), dtype=np.float32)
    low = np.asarray(runtime.env.action_space.low, dtype=np.float32)
    high = np.asarray(runtime.env.action_space.high, dtype=np.float32)
    if action.shape != (4,) or not np.isfinite(action).all():
        raise ValueError("Towel expert must emit one finite 4D action")
    return np.clip(action, low, high).astype(np.float32, copy=False)


def _success(info: Mapping[str, Any], env: Any) -> bool:
    value = info.get("success", getattr(env, "success", False))
    value = value() if callable(value) else value
    return bool(float(value) > 0.0)


def _rgb_image(record: Any) -> Image.Image:
    rgb = np.asarray(record.rgb, dtype=np.float32)
    if rgb.shape[0] != 3:
        raise ValueError("record RGB must be CHW")
    values = np.clip(np.transpose(rgb, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(values, mode="RGB")


def _anchor_occluded_record(
    record: Any,
    *,
    radius_px: int,
    blur_radius: float,
) -> Any:
    """Return a diagnostic frame with deployment-known anchor regions hidden."""

    radius = int(radius_px)
    blur = float(blur_radius)
    if radius <= 0 or not np.isfinite(blur) or blur < 0.0:
        raise ValueError("occluder radius must be positive and blur non-negative")
    image = _rgb_image(record)
    if blur > 0.0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur))
    draw = ImageDraw.Draw(image)
    for point in np.asarray(record.q_nominal, dtype=np.float32):
        x, y = (float(value) for value in point)
        draw.rectangle(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(18, 18, 18),
        )
    rgb = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return replace(record, rgb=np.ascontiguousarray(rgb, dtype=np.float32))


def _route_interruption_state(
    controller: str,
    *,
    hazard_active: bool,
    measurement_state: np.ndarray | None,
    last_valid_state: np.ndarray | None,
    next_prior_state: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, str, bool]:
    """Route current state without exposing hidden measurements to priors."""

    def checked(value: np.ndarray | None, label: str) -> np.ndarray:
        if value is None:
            raise RuntimeError(f"{label} state is unavailable")
        result = np.asarray(value, dtype=np.float32)
        if result.shape != (4, 192) or not np.isfinite(result).all():
            raise ValueError(f"{label} state must be one finite [4,192] array")
        return np.ascontiguousarray(result, dtype=np.float32)

    if controller not in INTERRUPTION_CONTROLLERS:
        raise ValueError(f"unsupported interruption controller: {controller}")
    if not hazard_active or controller == "lewm_fcpg_corrupt":
        current = checked(measurement_state, "measurement")
        trusted = current.copy() if not hazard_active else last_valid_state
        source = "corrupt_measurement" if hazard_active else "clean_measurement"
        return current, trusted, source, True
    if controller == "lewm_fcpg_hold":
        return checked(last_valid_state, "hold-last"), last_valid_state, "hold_last", False
    return checked(next_prior_state, "LeWM prior"), last_valid_state, "lewm_prior", False


@torch.inference_mode()
def _encode_current_state(
    model: Any,
    record: Any,
    *,
    q_grounded: np.ndarray,
    valid_grounded: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    rgb = torch.from_numpy(np.ascontiguousarray(record.rgb))[None, None].to(device)
    proprio = torch.from_numpy(np.ascontiguousarray(record.proprio))[None, None].to(device)
    q_nominal = torch.from_numpy(np.ascontiguousarray(record.q_nominal))[None, None].to(device)
    q_corrected = torch.from_numpy(np.ascontiguousarray(q_grounded))[None, None].to(device)
    xyz = torch.from_numpy(np.ascontiguousarray(record.keypoints_xyz))[None, None].to(device)
    valid = torch.from_numpy(np.ascontiguousarray(valid_grounded))[None, None].to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        current = model.encode_observations(
            rgb=rgb,
            proprio=proprio,
            q_nominal=q_nominal,
            q_corrected=q_corrected,
            keypoints_xyz=xyz,
            valid=valid,
        )[:, 0]
    result = current[0].float().cpu().numpy()
    if result.shape != (4, 192) or not np.isfinite(result).all():
        raise RuntimeError("encoded current state must be one finite [4,192] array")
    return np.ascontiguousarray(result, dtype=np.float32)


@torch.inference_mode()
def _lewm_costs_from_state(
    model: Any,
    current_state: np.ndarray,
    candidate_action: np.ndarray,
    history_action: np.ndarray,
    goal_tool_world: np.ndarray,
    *,
    current_keypoints_xyz: np.ndarray,
    camera_intrinsic: np.ndarray,
    camera_extrinsic: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate = np.asarray(candidate_action, dtype=np.float32)
    count = int(candidate.shape[0])
    state_array = np.asarray(current_state, dtype=np.float32)
    if state_array.shape != (4, 192) or not np.isfinite(state_array).all():
        raise ValueError("current_state must be one finite [4,192] array")
    state = torch.from_numpy(np.ascontiguousarray(state_array))[None].expand(
        count, -1, -1
    ).to(device)
    xyz = torch.from_numpy(
        np.ascontiguousarray(current_keypoints_xyz, dtype=np.float32)
    )[None].to(device)
    extrinsic = torch.from_numpy(np.ascontiguousarray(camera_extrinsic))[None].to(device)
    intrinsic = torch.from_numpy(np.ascontiguousarray(camera_intrinsic))[None].to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        future = torch.from_numpy(candidate).to(device=device, dtype=state.dtype)
        history = torch.from_numpy(np.ascontiguousarray(history_action))[None].expand(
            count, -1, -1
        ).to(device=device, dtype=state.dtype)
        task = torch.full(
            (count,), TASKS.index(TOWEL_WIPE_TASK_NAME), dtype=torch.long, device=device
        )
        extrinsic_batch = extrinsic.expand(count, -1, -1)
        predicted_state = model.predict_with_history(
            current_state=state,
            history_action=history,
            future_action=future,
            task_index=task,
            batch={"camera_extrinsic_cam_from_world": extrinsic_batch},
        )
        decoded = model.decode_3d(
            predicted_state=predicted_state,
            current_state=state,
            current_keypoints_xyz=xyz.expand(count, -1, -1),
            future_action=future,
            task_index=task,
            camera_intrinsic=intrinsic.expand(count, -1, -1),
            extrinsic_cam_from_world=extrinsic_batch,
        )
    predicted = decoded.predicted_world_xyz.float().cpu().numpy()
    predicted_latent = predicted_state.float().cpu().numpy()
    return (
        lewm_world_goal_cost(predicted, goal_tool_world),
        predicted,
        predicted_latent,
    )


def _lewm_costs(
    model: Any,
    record: Any,
    candidate_action: np.ndarray,
    history_action: np.ndarray,
    goal_tool_world: np.ndarray,
    *,
    q_grounded: np.ndarray,
    valid_grounded: np.ndarray,
    camera_intrinsic: np.ndarray,
    camera_extrinsic: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current = _encode_current_state(
        model,
        record,
        q_grounded=q_grounded,
        valid_grounded=valid_grounded,
        device=device,
    )
    return _lewm_costs_from_state(
        model,
        current,
        candidate_action,
        history_action,
        goal_tool_world,
        current_keypoints_xyz=np.asarray(record.keypoints_xyz),
        camera_intrinsic=camera_intrinsic,
        camera_extrinsic=camera_extrinsic,
        device=device,
    )


def _current_grounding(
    record: Any,
    *,
    mode: str,
    corrector: Any | None,
    device: torch.device,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    nominal = np.asarray(record.q_nominal, dtype=np.float32)
    if mode == "nominal":
        grounded = nominal.copy()
    elif mode == "fcpg":
        if corrector is None:
            raise ValueError("FCPG grounding requires a loaded corrector")
        grounded = corrected_points(
            [record], SimpleNamespace(corrector=corrector, device=device)
        )[0]
    else:
        raise ValueError(f"unsupported grounding mode: {mode}")
    valid = grounded_valid_mask(
        np.asarray(record.valid_nominal, dtype=bool),
        grounded,
        width=int(width),
        height=int(height),
    )
    truth = np.asarray(record.q_true, dtype=np.float32)
    truth_valid = np.asarray(record.valid_true, dtype=bool)
    support = valid & truth_valid
    epe = (
        float(np.linalg.norm(grounded[support] - truth[support], axis=-1).mean())
        if np.any(support)
        else float("nan")
    )
    innovation = float(np.linalg.norm(grounded - nominal, axis=-1).mean())
    return grounded, valid, epe, innovation


def _apply_camera_rotation(
    runtime: Any,
    *,
    axis_world: np.ndarray,
    rotation_degrees: float,
) -> None:
    degrees = float(rotation_degrees)
    if abs(degrees) <= 1.0e-12:
        return
    axis = np.asarray(axis_world, dtype=np.float64)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) <= 1.0e-12:
        raise ValueError("camera rotation axis must be one finite non-zero 3-vector")
    camera_id = get_camera_id(runtime.env.model, runtime.camera_name)
    base_position = np.asarray(
        runtime.env.model.cam_pos[camera_id], dtype=np.float64
    ).copy()
    base_quaternion = np.asarray(
        runtime.env.model.cam_quat[camera_id], dtype=np.float64
    ).copy()
    set_fixed_camera_pose_offset(
        runtime.env.model,
        runtime.env.data,
        camera_id,
        base_position,
        base_quaternion,
        np.zeros(3, dtype=np.float64),
        axis,
        degrees,
    )
    # Keep runtime.camera_extrinsic_cam_from_world nominal. Deployment does not
    # receive the unknown actual camera pose; record_current still measures
    # q_true from MuJoCo solely for offline evaluation.


def _draw_anchor_marker(
    draw: ImageDraw.ImageDraw,
    point: np.ndarray,
    *,
    color: tuple[int, int, int],
    diamond: bool,
) -> None:
    x, y = (float(value) for value in point)
    radius = 4.0
    if diamond:
        draw.polygon(
            ((x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)),
            fill=color,
        )
    else:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)


def _annotated_frame(
    record: Any,
    *,
    controller: str,
    raw_step: int,
    phase: str,
    half_cycles: int,
    choice: str,
    goal_error_m: float,
    q_grounded: np.ndarray,
    q_true: np.ndarray,
    grounding_epe_px: float,
    state_source: str = "measurement",
    hazard_active: bool = False,
    hazard_event_id: int = -1,
    hazard_event_age: int = 0,
    grounding_used: bool = True,
) -> Image.Image:
    image = _rgb_image(record)
    draw = ImageDraw.Draw(image)
    lines = (
        f"{controller} | step {raw_step}",
        f"{phase} | half-cycles {half_cycles}/6",
        f"choice {choice} | goal {100.0 * goal_error_m:.1f} cm",
        f"state {state_source} | hazard {int(hazard_active)} e{hazard_event_id} a{hazard_event_age}",
        f"ground EPE {grounding_epe_px:.2f} px | cyan {'used' if grounding_used else 'diagnostic'} / green true",
    )
    box_height = 15 * len(lines) + 8
    draw.rectangle((0, 0, image.width, box_height), fill=(0, 0, 0))
    for index, line in enumerate(lines):
        draw.text((5, 4 + 15 * index), line, fill=(255, 255, 255))
    for point in np.asarray(q_true, dtype=np.float32):
        _draw_anchor_marker(draw, point, color=(80, 220, 100), diamond=False)
    for point in np.asarray(q_grounded, dtype=np.float32):
        _draw_anchor_marker(draw, point, color=(60, 220, 235), diamond=True)
    return image


def _run_episode(
    model: Any,
    *,
    controller: str,
    task_instance: int,
    environment_seed: int,
    benchmark_seed: int,
    orbit_view: OrbitCameraView,
    width: int,
    height: int,
    max_raw_steps: int,
    frame_dir: Path,
    episode_index: int,
    device: torch.device,
    corrector: Any | None,
    camera_rotation_axis_world: np.ndarray,
    camera_rotation_degrees: float,
    hazard_schedule: RandomHazardSchedule | None,
    occluder_radius_px: int,
    occluder_blur_radius: float,
    candidate_source: str = "goal-centered",
) -> dict[str, Any]:
    runtime = make_a4_episode_runtime(
        TOWEL_WIPE_TASK_NAME,
        int(task_instance),
        int(environment_seed),
        width=int(width),
        height=int(height),
        benchmark_seed=int(benchmark_seed),
        camera_name="towel_view",
        orbit_camera_view=orbit_view,
    )
    _apply_camera_rotation(
        runtime,
        axis_world=camera_rotation_axis_world,
        rotation_degrees=camera_rotation_degrees,
    )
    observation = np.asarray(runtime.observation, dtype=np.float32)
    info = dict(runtime.reset_info)
    executed: list[np.ndarray] = []
    decisions: list[dict[str, Any]] = []
    frame_paths: list[str] = []
    reward_sum = 0.0
    success_streak = 0
    stable_success_step: int | None = None
    terminated_reason = "max_raw_steps"
    raw_step = 0
    wipe_decision_index = 0
    last_valid_state: np.ndarray | None = None
    next_prior_state: np.ndarray | None = None
    try:
        while raw_step < int(max_raw_steps):
            record = runtime.record_current(observation)
            phase = _phase(runtime.expert, observation)
            hazard_active = bool(
                controller in INTERRUPTION_CONTROLLERS
                and phase in WIPE_PHASES
                and hazard_schedule is not None
                and wipe_decision_index < len(hazard_schedule.active)
                and hazard_schedule.active[wipe_decision_index]
            )
            hazard_event_id = (
                int(hazard_schedule.event_id[wipe_decision_index])
                if hazard_active and hazard_schedule is not None
                else -1
            )
            hazard_event_age = (
                int(hazard_schedule.event_age[wipe_decision_index])
                if hazard_active and hazard_schedule is not None
                else 0
            )
            display_record = (
                _anchor_occluded_record(
                    record,
                    radius_px=occluder_radius_px,
                    blur_radius=occluder_blur_radius,
                )
                if hazard_active
                else record
            )
            target = np.asarray(runtime.env._target_pos, dtype=np.float32)
            goal_tool = target + SawyerTowelWipeEnvV0.GRIP_OFFSET.astype(np.float32)
            goal_error = float(
                np.linalg.norm(np.asarray(record.keypoints_xyz[2]) - goal_tool)
            )
            choice = "expert"
            lewm_cost: list[float] | None = None
            kinematic_cost: list[float] | None = None
            selected_index: int | None = None
            grounding_mode = "fcpg" if controller in FCPG_CONTROLLERS else "nominal"
            q_grounded, valid_grounded, grounding_epe, grounding_innovation = (
                _current_grounding(
                    display_record,
                    mode=grounding_mode,
                    corrector=corrector,
                    device=device,
                    width=width,
                    height=height,
                )
            )
            current_state_override: np.ndarray | None = None
            state_source = "measurement"
            grounding_used = True
            if controller in INTERRUPTION_CONTROLLERS and phase in WIPE_PHASES:
                measurement_state = None
                if not hazard_active or controller == "lewm_fcpg_corrupt":
                    measurement_state = _encode_current_state(
                        model,
                        display_record,
                        q_grounded=q_grounded,
                        valid_grounded=valid_grounded,
                        device=device,
                    )
                (
                    current_state_override,
                    last_valid_state,
                    state_source,
                    grounding_used,
                ) = _route_interruption_state(
                    controller,
                    hazard_active=hazard_active,
                    measurement_state=measurement_state,
                    last_valid_state=last_valid_state,
                    next_prior_state=next_prior_state,
                )

            if phase in WIPE_PHASES:
                expert_now = _bounded_expert_action(runtime, observation)
                if candidate_source == "policy-centered":
                    center = build_policy_repeated_prefix(
                        expert_now,
                        action_low=runtime.env.action_space.low,
                        action_high=runtime.env.action_space.high,
                    )
                    candidates = build_policy_centered_towel_candidates(
                        center,
                        action_low=runtime.env.action_space.low,
                        action_high=runtime.env.action_space.high,
                    )
                    kinematic_world = integrated_cartesian_action_forecast(
                        np.asarray(record.keypoints_xyz)[None], candidates[None]
                    )[0]
                    kinematic = tool_xy_goal_reach_cost(kinematic_world, goal_tool)
                    candidate_names = POLICY_CANDIDATE_NAMES
                elif candidate_source == "goal-centered":
                    center = build_goal_centered_prefix(
                        np.asarray(record.keypoints_xyz[2]),
                        goal_tool,
                        gripper_command=float(expert_now[3]),
                        action_low=runtime.env.action_space.low,
                        action_high=runtime.env.action_space.high,
                    )
                    candidates = build_towel_candidate_prefixes(
                        center,
                        action_low=runtime.env.action_space.low,
                        action_high=runtime.env.action_space.high,
                    )
                    kinematic = kinematic_world_goal_cost(
                        np.asarray(record.keypoints_xyz), candidates, goal_tool
                    )
                    candidate_names = CANDIDATE_NAMES
                else:
                    raise ValueError(f"unsupported candidate source: {candidate_source}")
                kinematic_cost = kinematic.astype(float).tolist()
                if controller == "center":
                    selected_index = 0
                elif controller == "kinematic":
                    selected_index = int(np.argmin(kinematic))
                else:
                    if current_state_override is None:
                        learned, predicted, predicted_latent = _lewm_costs(
                            model,
                            record,
                            candidates,
                            causal_history_chunks(executed),
                            goal_tool,
                            q_grounded=q_grounded,
                            valid_grounded=valid_grounded,
                            camera_intrinsic=runtime.camera_intrinsic,
                            camera_extrinsic=runtime.camera_extrinsic_cam_from_world,
                            device=device,
                        )
                    else:
                        learned, predicted, predicted_latent = _lewm_costs_from_state(
                            model,
                            current_state_override,
                            candidates,
                            causal_history_chunks(executed),
                            goal_tool,
                            current_keypoints_xyz=np.asarray(record.keypoints_xyz),
                            camera_intrinsic=runtime.camera_intrinsic,
                            camera_extrinsic=runtime.camera_extrinsic_cam_from_world,
                            device=device,
                        )
                    if candidate_source == "policy-centered":
                        learned = tool_xy_goal_reach_cost(predicted, goal_tool)
                    if controller == "lewm_residual":
                        calibrated = center_calibrated_world_forecast(
                            predicted,
                            np.asarray(record.keypoints_xyz),
                            candidates,
                        )
                        learned = (
                            tool_xy_goal_reach_cost(calibrated, goal_tool)
                            if candidate_source == "policy-centered"
                            else lewm_world_goal_cost(calibrated, goal_tool)
                        )
                    lewm_cost = learned.astype(float).tolist()
                    selected_index = int(np.argmin(learned))
                    if controller in INTERRUPTION_CONTROLLERS:
                        next_prior_state = np.ascontiguousarray(
                            predicted_latent[selected_index, 0], dtype=np.float32
                        )
                choice = candidate_names[selected_index]
                action_chunk = selected_raw_chunk(candidates, selected_index)
                wipe_decision_index += 1
            else:
                action_chunk = np.empty((RAW_CHUNK, 4), dtype=np.float32)
                virtual_observation = observation
                # Outside wiping, retain the task's tested closed-loop controller.
                action_chunk[0] = _bounded_expert_action(runtime, virtual_observation)

            image = _annotated_frame(
                display_record,
                controller=controller,
                raw_step=raw_step,
                phase=phase,
                half_cycles=int(runtime.env.wipe_half_cycles),
                choice=choice,
                goal_error_m=goal_error,
                q_grounded=q_grounded,
                q_true=np.asarray(record.q_true),
                grounding_epe_px=grounding_epe,
                state_source=state_source,
                hazard_active=hazard_active,
                hazard_event_id=hazard_event_id,
                hazard_event_age=hazard_event_age,
                grounding_used=grounding_used,
            )
            frame_name = f"ep{episode_index:02d}_{controller}_{len(frame_paths):03d}.jpg"
            image.save(frame_dir / frame_name, quality=88)
            frame_paths.append(f"frames/{frame_name}")
            decisions.append(
                {
                    "raw_step": int(raw_step),
                    "phase": phase,
                    "half_cycles": int(runtime.env.wipe_half_cycles),
                    "goal_error_m": goal_error,
                    "choice": choice,
                    "selected_index": selected_index,
                    "kinematic_cost_m": kinematic_cost,
                    "lewm_cost_m": lewm_cost,
                    "grounding_mode": grounding_mode,
                    "grounding_epe_px": grounding_epe,
                    "grounding_innovation_px": grounding_innovation,
                    "state_source": state_source,
                    "hazard_active": hazard_active,
                    "hazard_event_id": hazard_event_id,
                    "hazard_event_age": hazard_event_age,
                    "grounding_used": grounding_used,
                }
            )

            for chunk_step in range(RAW_CHUNK):
                if phase not in WIPE_PHASES and chunk_step > 0:
                    action = _bounded_expert_action(runtime, observation)
                else:
                    action = action_chunk[chunk_step]
                observation, reward, terminated, truncated, step_info = runtime.env.step(action)
                observation = np.asarray(observation, dtype=np.float32)
                info = dict(step_info) if isinstance(step_info, Mapping) else {}
                executed.append(np.asarray(action, dtype=np.float32).copy())
                reward_sum += float(reward)
                raw_step += 1
                if _success(info, runtime.env):
                    success_streak += 1
                    if success_streak >= 3:
                        stable_success_step = raw_step
                        terminated_reason = "stable_success"
                        break
                else:
                    success_streak = 0
                if terminated or truncated:
                    terminated_reason = "terminated" if terminated else "truncated"
                    break
                if raw_step >= int(max_raw_steps):
                    break
            if stable_success_step is not None or terminated or truncated:
                break
    finally:
        final_half_cycles = int(runtime.env.wipe_half_cycles)
        final_success = bool(runtime.env.is_success())
        runtime.env.close()

    wipe_rows = [row for row in decisions if row["phase"] in WIPE_PHASES]
    choice_counts = Counter(row["choice"] for row in wipe_rows)
    return {
        "controller": controller,
        "candidate_source": candidate_source,
        "episode_index": int(episode_index),
        "task_instance": int(task_instance),
        "environment_seed": int(environment_seed),
        "success": final_success,
        "stable_success_step": stable_success_step,
        "raw_steps": int(raw_step),
        "terminated_reason": terminated_reason,
        "wipe_half_cycles": final_half_cycles,
        "cumulative_reward": reward_sum,
        "wipe_decisions": len(wipe_rows),
        "mean_wipe_goal_error_m": (
            float(np.mean([row["goal_error_m"] for row in wipe_rows]))
            if wipe_rows
            else None
        ),
        "mean_wipe_grounding_epe_px": (
            float(np.nanmean([row["grounding_epe_px"] for row in wipe_rows]))
            if wipe_rows
            else None
        ),
        "choice_counts": dict(sorted(choice_counts.items())),
        "hazard_wipe_decisions": int(
            sum(bool(row["hazard_active"]) for row in wipe_rows)
        ),
        "state_source_counts": dict(
            sorted(Counter(str(row["state_source"]) for row in wipe_rows).items())
        ),
        "hazard_events": list(hazard_schedule.events) if hazard_schedule else [],
        "frame_paths": frame_paths,
        "decisions": decisions,
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_controller: dict[str, Any] = {}
    for controller in CONTROLLERS:
        selected = [row for row in rows if row["controller"] == controller]
        if not selected:
            continue
        success = np.asarray([bool(row["success"]) for row in selected])
        raw_steps = np.asarray([int(row["raw_steps"]) for row in selected])
        per_controller[controller] = {
            "episodes": len(selected),
            "success_rate": float(np.mean(success)),
            "mean_raw_steps": float(np.mean(raw_steps)),
            "mean_raw_steps_success_only": (
                float(np.mean(raw_steps[success])) if np.any(success) else None
            ),
            "mean_wipe_half_cycles": float(
                np.mean([int(row["wipe_half_cycles"]) for row in selected])
            ),
            "mean_cumulative_reward": float(
                np.mean([float(row["cumulative_reward"]) for row in selected])
            ),
            "mean_wipe_grounding_epe_px": float(
                np.mean(
                    [float(row["mean_wipe_grounding_epe_px"]) for row in selected]
                )
            ),
            "mean_hazard_wipe_decisions": float(
                np.mean([int(row["hazard_wipe_decisions"]) for row in selected])
            ),
        }
    return per_controller


def _paired_nominal_fcpg_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    by_key = {
        (int(row["episode_index"]), str(row["controller"])): row for row in rows
    }
    episodes = sorted(
        episode
        for episode, controller in by_key
        if controller == "lewm_nominal"
        and (episode, "lewm_fcpg") in by_key
    )
    if not episodes:
        return None
    nominal_rows = [by_key[(episode, "lewm_nominal")] for episode in episodes]
    fcpg_rows = [by_key[(episode, "lewm_fcpg")] for episode in episodes]
    disagreements = 0
    aligned = 0
    changed_episodes = 0
    for nominal, fcpg in zip(nominal_rows, fcpg_rows):
        nominal_choices = [
            row["choice"] for row in nominal["decisions"] if row["phase"] in WIPE_PHASES
        ]
        fcpg_choices = [
            row["choice"] for row in fcpg["decisions"] if row["phase"] in WIPE_PHASES
        ]
        shared = min(len(nominal_choices), len(fcpg_choices))
        changed = sum(
            left != right
            for left, right in zip(nominal_choices[:shared], fcpg_choices[:shared])
        )
        disagreements += changed
        aligned += shared
        changed_episodes += int(changed > 0 or len(nominal_choices) != len(fcpg_choices))
    nominal_epe = float(
        np.mean([float(row["mean_wipe_grounding_epe_px"]) for row in nominal_rows])
    )
    fcpg_epe = float(
        np.mean([float(row["mean_wipe_grounding_epe_px"]) for row in fcpg_rows])
    )
    return {
        "episodes": len(episodes),
        "both_success": int(
            sum(
                bool(nominal["success"]) and bool(fcpg["success"])
                for nominal, fcpg in zip(nominal_rows, fcpg_rows)
            )
        ),
        "nominal_only_success": int(
            sum(
                bool(nominal["success"]) and not bool(fcpg["success"])
                for nominal, fcpg in zip(nominal_rows, fcpg_rows)
            )
        ),
        "fcpg_only_success": int(
            sum(
                not bool(nominal["success"]) and bool(fcpg["success"])
                for nominal, fcpg in zip(nominal_rows, fcpg_rows)
            )
        ),
        "neither_success": int(
            sum(
                not bool(nominal["success"]) and not bool(fcpg["success"])
                for nominal, fcpg in zip(nominal_rows, fcpg_rows)
            )
        ),
        "mean_raw_step_delta_fcpg_minus_nominal": float(
            np.mean(
                [
                    int(fcpg["raw_steps"]) - int(nominal["raw_steps"])
                    for nominal, fcpg in zip(nominal_rows, fcpg_rows)
                ]
            )
        ),
        "nominal_mean_wipe_grounding_epe_px": nominal_epe,
        "fcpg_mean_wipe_grounding_epe_px": fcpg_epe,
        "relative_grounding_epe_reduction": 1.0
        - fcpg_epe / max(nominal_epe, np.finfo(np.float64).eps),
        "aligned_wipe_decisions": aligned,
        "candidate_choice_disagreements": disagreements,
        "candidate_choice_disagreement_fraction": disagreements / max(aligned, 1),
        "episodes_with_changed_choice_sequence": changed_episodes,
    }


def _write_html(output: Path, rows: Sequence[Mapping[str, Any]], duration_ms: int) -> None:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["episode_index"]), []).append(row)
    cards: list[str] = []
    for episode, episode_rows in sorted(grouped.items()):
        panels = []
        ordered_rows = sorted(
            episode_rows,
            key=lambda value: CONTROLLERS.index(str(value["controller"])),
        )
        for row in ordered_rows:
            paths = json.dumps(row["frame_paths"])
            panels.append(
                "<section><h3>"
                + html.escape(str(row["controller"]))
                + f" | success={row['success']} | steps={row['raw_steps']}</h3>"
                + f'<img class="player" data-frames=\'{html.escape(paths)}\' alt="rollout">'
                + "</section>"
            )
        cards.append(f"<h2>Episode {episode}</h2><div class=\"row\">{''.join(panels)}</div>")
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Towel H6 receding selector</title>
<style>
body{{font:14px sans-serif;background:#171717;color:#eee;margin:20px}}
.row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}}
section{{background:#252525;padding:8px;border-radius:6px}}
img{{width:100%;image-rendering:auto}}h2{{margin-top:28px}}h3{{font-size:14px}}
</style></head>
<body><h1>Towel H6 receding-horizon smoke</h1>
<p>All controllers share the same reset. LeWM predicts H6 but executes only
the first five raw actions before re-observing. LeWM selects actions only during
the wipe phases; the existing expert controls grasp and lift. Text overlays are
visualization only.</p>{''.join(cards)}
<script>
const players=[...document.querySelectorAll('.player')];
players.forEach(p=>{{
  p.frames=JSON.parse(p.dataset.frames);p.i=0;
  if(p.frames.length)p.src=p.frames[0];
}});
setInterval(()=>players.forEach(p=>{{
  if(p.frames.length){{p.i=(p.i+1)%p.frames.length;p.src=p.frames[p.i];}}
}}),{int(duration_ms)});
</script></body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")


def main() -> int:
    args = _parser().parse_args()
    if int(args.episodes) <= 0:
        raise ValueError("episodes must be positive")
    if not 1 <= int(args.clock_hour) <= 12:
        raise ValueError("clock-hour must be in 1..12")
    if int(args.max_raw_steps) <= 0:
        raise ValueError("max-raw-steps must be positive")
    if not 0.0 <= float(args.hazard_probability) <= 1.0:
        raise ValueError("hazard probability must be in [0,1]")
    if not (
        1
        <= int(args.hazard_min_duration)
        <= int(args.hazard_max_duration)
        <= int(args.hazard_schedule_length)
    ):
        raise ValueError("hazard duration must fit the positive schedule length")
    if int(args.hazard_max_events) <= 0:
        raise ValueError("hazard-max-events must be positive")
    if int(args.occluder_radius_px) <= 0 or float(args.occluder_blur_radius) < 0.0:
        raise ValueError("occluder radius/blur must be positive/non-negative")
    camera_rotation_degrees = float(args.camera_rotation_degrees)
    camera_rotation_axis_world = np.asarray(
        args.camera_rotation_axis_world, dtype=np.float64
    )
    if not np.isfinite(camera_rotation_degrees):
        raise ValueError("camera rotation degrees must be finite")
    if (
        camera_rotation_axis_world.shape != (3,)
        or not np.isfinite(camera_rotation_axis_world).all()
    ):
        raise ValueError("camera rotation axis must be one finite 3-vector")
    if abs(camera_rotation_degrees) > 0.0 and np.linalg.norm(camera_rotation_axis_world) <= 1.0e-12:
        raise ValueError("non-zero camera rotation requires a non-zero axis")
    device = _cuda0(args.device)
    torch.cuda.set_device(device)
    output = args.output_dir.expanduser().resolve()
    frame_dir = output / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    orbit = OrbitCameraView(
        clock_hour=int(args.clock_hour),
        elevation_degrees=35.0,
        radius_m=1.1,
        base_camera_name="towel_view",
    )
    model, checkpoint = gallery._load_checkpoint(
        args.checkpoint.expanduser().resolve(), device=device
    )
    if not callable(getattr(model, "predict_with_history", None)):
        raise TypeError("selector requires a history-conditioned checkpoint")
    corrector = None
    if any(controller in FCPG_CONTROLLERS for controller in args.controllers):
        corrector, _ = load_fcpg_vit_adapter(
            args.fcpg_checkpoint.expanduser().resolve(), device=device
        )
    rows: list[dict[str, Any]] = []
    try:
        for episode in range(int(args.episodes)):
            task_instance = int(args.task_instance_start) + episode
            if not 0 <= task_instance < 48:
                raise ValueError("task instances must remain in 0..47")
            environment_seed = int(args.environment_seed_base) + episode
            hazard_schedule = (
                sample_random_hazard_schedule(
                    int(args.hazard_schedule_length),
                    seed=int(args.hazard_seed) + episode,
                    trigger_probability=float(args.hazard_probability),
                    minimum_duration=int(args.hazard_min_duration),
                    maximum_duration=int(args.hazard_max_duration),
                    maximum_events=int(args.hazard_max_events),
                    warmup_steps=2,
                )
                if any(
                    controller in INTERRUPTION_CONTROLLERS
                    for controller in args.controllers
                )
                else None
            )
            for controller in args.controllers:
                row = _run_episode(
                    model,
                    controller=str(controller),
                    task_instance=task_instance,
                    environment_seed=environment_seed,
                    benchmark_seed=int(args.benchmark_seed),
                    orbit_view=orbit,
                    width=int(args.width),
                    height=int(args.height),
                    max_raw_steps=int(args.max_raw_steps),
                    frame_dir=frame_dir,
                    episode_index=episode,
                    device=device,
                    corrector=corrector,
                    camera_rotation_axis_world=camera_rotation_axis_world,
                    camera_rotation_degrees=camera_rotation_degrees,
                    hazard_schedule=hazard_schedule,
                    occluder_radius_px=int(args.occluder_radius_px),
                    occluder_blur_radius=float(args.occluder_blur_radius),
                    candidate_source=str(args.candidate_source),
                )
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "episode": episode,
                            "controller": controller,
                            "success": row["success"],
                            "raw_steps": row["raw_steps"],
                            "half_cycles": row["wipe_half_cycles"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        del model
        if corrector is not None:
            del corrector
        gallery.legacy._clear_runtime_memory(device)
    report = {
        "schema_version": 2,
        "protocol": "a4_towel_h6_receding_selector_smoke_v2",
        "status": "development_smoke_only",
        "checkpoint_protocol": str(checkpoint.get("protocol", "")),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "camera_clock_hour": int(args.clock_hour),
        "grounding": "controller_specific_nominal_or_fcpg",
        "fcpg_checkpoint": (
            str(args.fcpg_checkpoint.expanduser().resolve())
            if any(controller in FCPG_CONTROLLERS for controller in args.controllers)
            else None
        ),
        "camera_rotation_degrees": camera_rotation_degrees,
        "camera_rotation_axis_world": camera_rotation_axis_world.tolist(),
        "goal_input": "current_task_endpoint_world_coordinate",
        "candidate_inputs": (
            "current_policy_action_only_goal_used_for_scoring"
            if args.candidate_source == "policy-centered"
            else "current_tool_current_goal_only"
        ),
        "candidate_source": str(args.candidate_source),
        "primary_horizon": f"H{PRIMARY_HORIZON_INDEX + 1}",
        "execution": "first_5_raw_actions_then_reobserve",
        "controller_scope": "lewm_selects_wipe_phases_expert_controls_other_phases",
        "interruption": {
            "enabled": any(
                controller in INTERRUPTION_CONTROLLERS
                for controller in args.controllers
            ),
            "schedule_axis": "wipe_decision_index",
            "schedule_reads_task_state": False,
            "sensor_validity": "privileged_routing_only",
            "hazard_seed": int(args.hazard_seed),
            "trigger_probability": float(args.hazard_probability),
            "minimum_duration": int(args.hazard_min_duration),
            "maximum_duration": int(args.hazard_max_duration),
            "maximum_events": int(args.hazard_max_events),
            "occlusion": "gaussian_blur_plus_nominal_anchor_local_blocks",
            "occluder_radius_px": int(args.occluder_radius_px),
            "occluder_blur_radius": float(args.occluder_blur_radius),
        },
        "summary": _summary(rows),
        "paired_nominal_fcpg": _paired_nominal_fcpg_summary(rows),
        "episodes": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_html(output, rows, int(args.frame_duration_ms))
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
