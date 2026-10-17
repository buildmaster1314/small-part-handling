#!/usr/bin/env python3
"""Generate SO-101 IK candidates and rank their predicted outcomes with LeWM."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from fcpg.so101_active_gripper import (
    apply_live_center_translation,
    apply_live_similarity_transform,
    invert_live_similarity_transform,
    select_live_moving_tip_similarity_transform,
)
from fcpg.so101_camera_calibration import approximate_column_backprojection_on_plane
from fcpg.so101_fcpg_lewm_bridge import (
    SO101FCPGConditionedLeWM,
    build_so101_fcpg_corrector,
    infer_so101_fcpg_corrector_variant,
)
from fcpg.so101_ik_candidates import generate_ik_candidate_prefixes
from fcpg.so101_kinematics import SO101MuJoCoKinematics
from fcpg.so101_multiscale_memory import (
    MultiScaleHistoryLeWM,
    rollout_multiscale_candidates,
)
from fcpg.so101_v9 import V6_PROTOCOL, V9_PROTOCOL
from fcpg.so101_receding_control import (
    SO101ObservationIdentity,
    SO101RecedingHorizonGate,
)
from fcpg.so101_retrieval_mpc import (
    SO101_POSITION_LOWER,
    SO101_POSITION_UPPER,
    candidate_safety_checks,
    chunk_and_normalize_actions,
)
from fcpg.so101_state_residual_lewm import StateResidualLeWM
from fcpg.so101_trajectory_memory import DiskBackedLFUTrajectoryStore
from scripts.evaluate_so101_real_shadow_fcpg_lewm import (
    VIEWS,
    brown_box_descriptor,
    draw_grounding,
    encode_fcpg_view,
    load_paired_session,
    load_projection_fits,
    project_history,
    red_object_descriptor,
)
from scripts.evaluate_so101_retrieval_mpc import _rollout_candidates
from scripts.render_so101_mujoco_prediction import SO101ReplayRenderer
from scripts.visualize_so101_lewm import (
    build_checkpoint_model,
    checkpoint_action_normalizer,
    checkpoint_contract,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "third_party/so_arm100/Simulation/SO101/so101_new_calib.xml"
DEFAULT_SCENE = ROOT / "third_party/so_arm100/Simulation/SO101/scene.xml"
DEFAULT_TRAJECTORY_STORE = (
    ROOT / "outputs/lerobot/so101/receding_trajectory_memory"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--bridge-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-checkpoint",
        type=Path,
        help="Optional compatible dynamics checkpoint override for matched shadow comparison.",
    )
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument(
        "--column-gripper-localization",
        type=Path,
        help="Legacy one-point translation bootstrap; V9 does not accept this contract.",
    )
    parser.add_argument(
        "--live-anchor-calibration",
        type=Path,
        help=(
            "Open/closed moving-tip calibration used to fit the column similarity "
            "transform. Required by V9."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trajectory-store",
        type=Path,
        default=DEFAULT_TRAJECTORY_STORE,
        help="Append every offline planning round to this disk-backed trajectory store.",
    )
    parser.add_argument("--trajectory-memory-capacity", type=int, default=32)
    parser.add_argument("--target", choices=("fish", "box"), default="fish")
    parser.add_argument("--table-z", type=float, default=-0.01711812394)
    parser.add_argument("--goal-clearance", type=float, default=0.025)
    parser.add_argument("--history-stride", type=int, default=5)
    parser.add_argument(
        "--freeze-recent-frames",
        type=int,
        choices=(0, 1, 2, 3),
        default=0,
        help="Causally freeze this many latest visual contexts; proprio remains current.",
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-command-step", type=float, default=2.5)
    parser.add_argument("--prediction-bound-margin", type=float, default=3.0)
    parser.add_argument(
        "--residual-scales",
        default="0,0.25,0.5,0.75,1",
        help="Comma-separated recursive LeWM residual scales used for diagnosis.",
    )
    parser.add_argument(
        "--render-residual-scale",
        type=float,
        default=1.0,
        help="Residual scale whose prediction is rendered and used in the candidate table.",
    )
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-25.0)
    parser.add_argument("--distance", type=float, default=0.72)
    args = parser.parse_args()
    if args.device.startswith("cuda:1"):
        parser.error("CUDA1 is forbidden for this project")
    if (
        args.column_gripper_localization is not None
        and args.live_anchor_calibration is not None
    ):
        parser.error(
            "--column-gripper-localization and --live-anchor-calibration are exclusive"
        )
    if args.history_stride <= 0 or args.horizon <= 0:
        parser.error("history-stride and horizon must be positive")
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("fps must be finite and positive")
    if args.trajectory_memory_capacity < 0:
        parser.error("trajectory-memory-capacity must be non-negative")
    try:
        args.residual_scales = tuple(
            float(value.strip())
            for value in args.residual_scales.split(",")
            if value.strip()
        )
    except ValueError as exc:
        parser.error(f"invalid --residual-scales: {exc}")
    if not args.residual_scales:
        parser.error("--residual-scales must contain at least one value")
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in args.residual_scales):
        parser.error("all residual scales must lie in [0, 1]")
    if not np.isfinite(args.render_residual_scale) or not 0.0 <= args.render_residual_scale <= 1.0:
        parser.error("--render-residual-scale must lie in [0, 1]")
    if not any(
        np.isclose(args.render_residual_scale, value, rtol=0.0, atol=1e-9)
        for value in args.residual_scales
    ):
        parser.error("--render-residual-scale must be listed in --residual-scales")
    return args


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _target_descriptor(frame: np.ndarray, kind: str) -> dict[str, Any]:
    descriptor = (
        red_object_descriptor(frame) if kind == "fish" else brown_box_descriptor(frame)
    )
    if not descriptor["visible"]:
        raise ValueError(f"current column frame does not contain a detectable {kind}")
    return descriptor


def _causal_freeze_history(
    frames: np.ndarray,
    fallback_frame: np.ndarray,
    suffix_frames: int,
) -> np.ndarray:
    values = np.asarray(frames)
    fallback = np.asarray(fallback_frame)
    if values.ndim != 4 or len(values) != 3 or fallback.shape != values.shape[1:]:
        raise ValueError("visual freeze requires three aligned RGB history frames")
    if suffix_frames not in (0, 1, 2, 3):
        raise ValueError("suffix_frames must be in [0,3]")
    frozen = values.copy()
    if suffix_frames == 0:
        return frozen
    first_missing = 3 - suffix_frames
    source = fallback if first_missing == 0 else frozen[first_missing - 1]
    frozen[first_missing:] = source
    return frozen


def _normalized_session_action_chunk(
    session,
    start: int,
    frameskip: int,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    source = session.states if session.actions is None else session.actions
    raw = np.asarray(source[start : start + frameskip], dtype=np.float32)
    if raw.shape != (frameskip, 6):
        raise ValueError("recorded action chunk is incomplete")
    return ((raw - center) / scale).reshape(-1).astype(np.float32)


def _draw_column_target(
    frame: np.ndarray,
    target_px: np.ndarray,
    current_gripper_px: np.ndarray,
    fcpg_gripper_px: np.ndarray | None = None,
) -> Image.Image:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    tx, ty = map(float, target_px)
    gx, gy = map(float, current_gripper_px)
    draw.line((gx, gy, tx, ty), fill=(56, 139, 194), width=3)
    draw.ellipse((tx - 7, ty - 7, tx + 7, ty + 7), outline=(245, 190, 45), width=4)
    draw.ellipse((gx - 6, gy - 6, gx + 6, gy + 6), outline=(250, 250, 250), width=4)
    if fcpg_gripper_px is not None:
        fx, fy = map(float, fcpg_gripper_px)
        draw.ellipse((fx - 5, fy - 5, fx + 5, fy + 5), outline=(40, 190, 215), width=3)
    return image


def _load_column_gripper_localization(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("format") != "so101-active-gripper-localization-v1":
        raise ValueError("column gripper localization has an unsupported format")
    if payload.get("camera") != "column":
        raise ValueError("gripper localization must come from the column camera")
    center = np.asarray(payload.get("center_px"), dtype=np.float64)
    if (
        center.shape != (2,)
        or not np.all(np.isfinite(center))
        or np.any(center < 0.0)
        or np.any(center >= 224.0)
    ):
        raise ValueError("column gripper localization has no finite center_px")
    required_quality = {
        "opening_span": (20.0, np.inf),
        "max_arm_motion": (0.0, 2.0),
        "moving_area_px": (24.0, 224.0 * 224.0 * 0.18),
        "contrast": (12.0, np.inf),
        "elongation": (1.0, np.inf),
    }
    for key, (lower, upper) in required_quality.items():
        value = payload.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"column gripper localization is missing quality field {key}")
        if not lower <= float(value) <= upper:
            raise ValueError(f"column gripper localization failed quality field {key}")
    payload["center_px"] = center
    return payload


def _load_live_anchor_calibration(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("version") != 1 or payload.get("view") != "column":
        raise ValueError("unsupported live-anchor calibration contract")
    candidates = np.asarray(
        payload.get("moving_tip_track_candidates_px"), dtype=np.float32
    )
    open_state = np.asarray(payload.get("open_state"), dtype=np.float32)
    closed_state = np.asarray(payload.get("closed_state"), dtype=np.float32)
    if (
        candidates.ndim != 3
        or candidates.shape[1:] != (2, 2)
        or len(candidates) == 0
        or not np.all(np.isfinite(candidates))
    ):
        raise ValueError("live-anchor calibration has invalid moving-tip tracks")
    if (
        open_state.shape != (6,)
        or closed_state.shape != (6,)
        or not np.all(np.isfinite(open_state))
        or not np.all(np.isfinite(closed_state))
    ):
        raise ValueError("live-anchor calibration has invalid joint states")
    return {
        **payload,
        "moving_tip_track_candidates_px": candidates,
        "open_state": open_state,
        "closed_state": closed_state,
        "source": str(path.expanduser().resolve()),
    }


def _render_selected(
    output: Path,
    current_state: np.ndarray,
    command_states: np.ndarray,
    predicted_states: np.ndarray,
    *,
    azimuth: float,
    elevation: float,
    distance: float,
) -> tuple[list[str], list[str]]:
    frames = output / "frames"
    frames.mkdir()
    renderer = SO101ReplayRenderer(
        DEFAULT_SCENE,
        azimuth=azimuth,
        elevation=elevation,
        distance=distance,
    )
    command_paths: list[str] = []
    prediction_paths: list[str] = []
    try:
        command_sequence = np.concatenate((current_state[None], command_states), axis=0)
        prediction_sequence = np.concatenate((current_state[None], predicted_states), axis=0)
        for horizon, (command, prediction) in enumerate(
            zip(command_sequence, prediction_sequence, strict=True)
        ):
            command_rgb, _, _ = renderer.render(command)
            prediction_rgb, _, _ = renderer.render(prediction)
            command_name = f"frames/command_h{horizon}.jpg"
            prediction_name = f"frames/prediction_h{horizon}.jpg"
            Image.fromarray(command_rgb).save(output / command_name, quality=92)
            Image.fromarray(prediction_rgb).save(output / prediction_name, quality=92)
            command_paths.append(command_name)
            prediction_paths.append(prediction_name)
    finally:
        renderer.close()
    return command_paths, prediction_paths


def _html_page(report: dict[str, Any]) -> str:
    candidates = "".join(
        "<tr><td>{index}</td><td>{name}</td><td>{safe}</td><td>{prediction}</td>"
        "<td>{pred_terminal:.1f}</td><td>{pred_tail:.1f}</td>"
        "<td>{command_terminal:.1f}</td><td>{consistency:.1f}</td>"
        "<td>{ik:.1f}</td></tr>".format(
            index=row["index"],
            name=html.escape(row["name"]),
            safe="yes" if row["safe"] else "no",
            prediction="yes" if row["prediction_plausible"] else "no",
            pred_terminal=1000.0 * row["predicted_terminal_goal_distance_m"],
            pred_tail=1000.0 * row["predicted_tail_goal_distance_m"],
            command_terminal=1000.0 * row["command_terminal_goal_distance_m"],
            consistency=1000.0 * row["prediction_command_disagreement_m"],
            ik=1000.0 * row["maximum_ik_error_m"],
        )
        for row in report["candidates"]
    )
    scale_rows = "".join(
        "<tr><td>{scale:.2f}</td><td>{selected}</td><td>{terminal:.1f}</td>"
        "<td>{tail:.1f}</td><td>{gap:.1f}</td><td>{joint:.2f}</td></tr>".format(
            scale=row["residual_scale"],
            selected=html.escape(row["selected_name"]),
            terminal=1000.0 * row["geometry_candidate_terminal_goal_distance_m"],
            tail=1000.0 * row["geometry_candidate_tail_goal_distance_m"],
            gap=1000.0 * row["geometry_candidate_prediction_command_disagreement_m"],
            joint=row["geometry_candidate_terminal_max_joint_residual"],
        )
        for row in report["residual_scale_diagnostics"]
    )
    command_frames = json.dumps(report["media"]["command_frames"])
    prediction_frames = json.dumps(report["media"]["prediction_frames"])
    interval_ms = int(round(1000.0 / report["media"]["fps"]))
    has_live_bootstrap = (
        report["column_live_calibration"] is not None
        or report["column_gripper_localization"] is not None
    )
    column_legend = (
        "黄色目标 / 白色主动定位 / 青色 FCPG"
        if has_live_bootstrap
        else "黄色目标 / 白色未实时校准估计"
    )
    bootstrap_note = (
        "column 已由开合脉冲拟合相似变换；两个指尖独立校正，中点由两端计算。"
        if has_live_bootstrap
        else "column 尚无开合脉冲定位，白点与目标 3D 不具备执行资格。"
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SO-101 IK + LeWM Shadow</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#edf1f3;color:#182126;font:15px system-ui}}header{{background:#1f2d33;color:#fff;padding:22px 4%}}main{{max-width:1280px;margin:auto;padding:18px}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}article{{background:#fff;border:1px solid #c7d0d4;padding:12px}}img{{width:100%;height:auto;display:block}}h2{{font-size:16px;margin:0 0 10px}}.controls{{display:flex;gap:10px;align-items:center;margin:12px 0}}input[type=range]{{flex:1}}table{{width:100%;border-collapse:collapse;background:#fff;margin-top:16px}}th,td{{padding:8px;border-bottom:1px solid #d9dfe2;text-align:right}}th:nth-child(2),td:nth-child(2){{text-align:left}}.note{{background:#fff;border-left:5px solid #2876a7;padding:14px;margin-bottom:14px;line-height:1.55}}.execution{{border-left-color:#b15b22}}#label{{min-width:330px}}@media(max-width:760px){{.grid{{grid-template-columns:1fr}}#label{{min-width:0}}}}</style></head><body>
<header><h1>SO-101：FCPG 当前接地 + IK 候选 + LeWM 后果预测</h1><p>纯离线 shadow；没有发送机械臂命令。左侧是 IK/FK 命令姿态，右侧是所选 LeWM 变体对同一动作的未来状态预测。</p></header><main>
<div class="note">{bootstrap_note} 目标 {html.escape(report['target']['kind'])}：临时桌面平面 XYZ = {report['target']['surface_xyz_m']}。模型 = {html.escape(report['selection']['deployed_variant'])}；冻结最近视觉帧数 = {report['freeze_recent_frames']}。Geometry winner = {report['selection']['geometry_name']}；parent winner = {report['selection']['parent_name']}；当前模型 winner = {report['selection']['lewm_name']}。当前 3D 是 RGB-only 临时逆解，不是深度相机测量。</div>
<div class="note execution"><strong>执行边界：</strong>H1 只对应接下来 {report['execution_contract']['commands_per_round']} 条 raw command。H2-H8 全部只是 LeWM 诊断预测，不得连续发送。H1 执行后必须重新读取 wrist + column、重新运行 FCPG、重新检测目标并重新规划。</div>
<div class="grid"><article><h2>当前 wrist + FCPG</h2><img src="wrist_grounding.jpg"></article><article><h2>当前 column：{column_legend}</h2><img src="column_target.jpg"></article><article><h2>IK/FK 命令轨迹</h2><img id="command" src="{report['media']['command_frames'][0]}"></article><article><h2>LeWM 预测轨迹</h2><img id="prediction" src="{report['media']['prediction_frames'][0]}"></article></div>
<div class="controls"><button id="play">暂停</button><input id="step" type="range" min="0" max="{len(report['media']['command_frames']) - 1}" value="0"><span id="label">H0：当前观测</span></div>
<table><thead><tr><th>residual alpha</th><th>selected path</th><th>geometry path H8→goal mm</th><th>geometry path H6:H8→goal mm</th><th>LeWM/FK gap mm</th><th>H8 max joint residual</th></tr></thead><tbody>{scale_rows}</tbody></table>
<table><thead><tr><th>#</th><th>IK path</th><th>command safe</th><th>LeWM bounds</th><th>LeWM H8→goal mm</th><th>LeWM H6:H8→goal mm</th><th>FK H8→goal mm</th><th>LeWM/FK gap mm</th><th>max IK err mm</th></tr></thead><tbody>{candidates}</tbody></table>
<script>const commands={command_frames};const predictions={prediction_frames};const slider=document.getElementById('step');const label=document.getElementById('label');const play=document.getElementById('play');let running=true;function stage(i){{if(i===0)return'H0：当前观测';if(i===1)return'H1：唯一可执行边界（5 commands）';return'H'+i+'：仅诊断，必须重新观测后才能执行';}}function show(i){{slider.value=i;document.getElementById('command').src=commands[i];document.getElementById('prediction').src=predictions[i];label.textContent=stage(i);}}slider.oninput=()=>show(Number(slider.value));play.onclick=()=>{{running=!running;play.textContent=running?'暂停':'播放';}};setInterval(()=>{{if(running)show((Number(slider.value)+1)%commands.length);}}, {interval_ms});</script>
</main></body></html>"""


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)

    sessions = load_paired_session(args.session)
    final_index = sessions["wrist"].frame_count - 1
    history_indices = final_index - np.arange(2, -1, -1) * args.history_stride
    if history_indices[0] < 0:
        raise ValueError("shadow session is too short for three history frames")
    states = np.asarray(sessions["wrist"].states[history_indices], dtype=np.float32)
    fallback_index = int(history_indices[0] - args.history_stride)
    if args.freeze_recent_frames == 3 and fallback_index < 0:
        raise ValueError("shadow session has no causal frame before the recent history")
    view_frames = {
        view: _causal_freeze_history(
            np.asarray(sessions[view].frames[history_indices]),
            np.asarray(
                sessions[view].frames[
                    fallback_index if fallback_index >= 0 else history_indices[0]
                ]
            ),
            args.freeze_recent_frames,
        )
        for view in VIEWS
    }

    device = torch.device(args.device)
    bridge_payload = torch.load(
        args.bridge_checkpoint.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_path = (
        args.model_checkpoint.expanduser().resolve()
        if args.model_checkpoint is not None
        else Path(bridge_payload["args"]["checkpoint"]).resolve()
    )
    checkpoint, embedded_args, _, frameskip, history, checkpoint_sha = checkpoint_contract(
        checkpoint_path, None
    )
    if frameskip != args.history_stride or history != 3 or args.horizon != 8:
        raise ValueError("IK shadow must match the checkpoint 3x5 history and H8 contract")
    loaded_model = build_checkpoint_model(checkpoint, frameskip, history, device)
    memory_model: MultiScaleHistoryLeWM | None = None
    protocol = checkpoint["run_config"]["contract"].get("protocol")
    if isinstance(loaded_model, MultiScaleHistoryLeWM):
        memory_model = loaded_model
        model = memory_model.backbone
        if protocol not in {V6_PROTOCOL, V9_PROTOCOL}:
            raise ValueError("IK shadow deploys a V6 or V9 memory checkpoint only")
    elif isinstance(loaded_model, StateResidualLeWM):
        model = loaded_model
    else:
        raise TypeError("IK shadow requires state-residual, V6, or V9 LeWM")
    corrector = build_so101_fcpg_corrector(
        model,
        camera_context_dim=5,
        variant=infer_so101_fcpg_corrector_variant(bridge_payload),
    )
    system = SO101FCPGConditionedLeWM(
        model,
        corrector=corrector,
        camera_context_dim=5,
        freeze_model=True,
    ).to(device)
    system.corrector.load_state_dict(bridge_payload["corrector"], strict=True)
    system.bridge.load_state_dict(bridge_payload["bridge"], strict=True)
    system.eval()

    contract = checkpoint["run_config"]["contract"]
    state_center = np.asarray(contract["normalization_center"], dtype=np.float32)
    state_scale = np.asarray(contract["normalization_scale"], dtype=np.float32)
    normalized_states = (states - state_center) / state_scale
    kinematics = SO101MuJoCoKinematics(DEFAULT_MODEL)
    fits = load_projection_fits(args.calibration)
    anchors_xyz, nominal, valid = project_history(
        states, fits, kinematics, image_size=(224, 224)
    )
    column_localization: dict[str, Any] | None = None
    column_live_calibration: dict[str, Any] | None = None
    column_transform: np.ndarray | None = None
    column_offset = np.zeros(2, dtype=np.float32)
    if args.live_anchor_calibration is not None:
        calibration = _load_live_anchor_calibration(args.live_anchor_calibration)
        calibration_states = np.stack(
            (calibration["open_state"], calibration["closed_state"])
        )
        _, calibration_nominal, _ = project_history(
            calibration_states, fits, kinematics, image_size=(224, 224)
        )
        fit_result = select_live_moving_tip_similarity_transform(
            calibration_nominal["column"],
            calibration["moving_tip_track_candidates_px"],
        )
        column_transform = np.asarray(fit_result["transform_px"], dtype=np.float32)
        column_offset = column_transform[:, 2].copy()
        nominal["column"] = apply_live_similarity_transform(
            nominal["column"], column_transform
        )
        valid["column"] = np.asarray(valid["column"], dtype=bool) & np.all(
            (nominal["column"] >= 0.0) & (nominal["column"] < 224.0), axis=-1
        )
        column_live_calibration = {
            **fit_result,
            "source": calibration["source"],
            "open_state": calibration["open_state"],
            "closed_state": calibration["closed_state"],
        }
    elif args.column_gripper_localization is not None:
        column_localization = _load_column_gripper_localization(
            args.column_gripper_localization
        )
        nominal["column"], column_offset = apply_live_center_translation(
            nominal["column"], column_localization["center_px"]
        )
    if protocol == V9_PROTOCOL and column_transform is None:
        raise ValueError(
            "V9 release shadows require --live-anchor-calibration; a one-point "
            "translation or uncalibrated column projection is not publishable"
        )

    contexts: dict[str, np.ndarray] = {}
    corrected: dict[str, np.ndarray] = {}
    for view_index, view in enumerate(VIEWS):
        camera_context = np.zeros((history, 5), dtype=np.float32)
        camera_context[:, view_index] = 1.0
        _, context, points = encode_fcpg_view(
            system,
            view_frames[view],
            normalized_states,
            anchors_xyz,
            nominal[view],
            valid[view],
            camera_context,
            device,
            derive_center_from_tips=True,
        )
        contexts[view] = context
        corrected[view] = points
        draw_grounding(
            view_frames[view][-1], nominal[view][-1], points[-1], valid[view][-1]
        ).save(output / f"{view}_grounding.jpg", quality=92)

    memory_data: dict[str, np.ndarray] | None = None
    if memory_model is not None:
        memory_offsets = np.asarray(contract["memory_offsets"], dtype=np.int64)
        memory_indices = int(history_indices[0]) - memory_offsets
        if np.any(memory_indices < 0):
            raise ValueError(
                "shadow session is too short for the memory offsets "
                f"{memory_offsets.tolist()}"
            )
        memory_states = np.asarray(
            sessions["wrist"].states[memory_indices], dtype=np.float32
        )
        normalized_memory_states = (memory_states - state_center) / state_scale
        memory_anchors_xyz, memory_nominal, memory_valid = project_history(
            memory_states, fits, kinematics, image_size=(224, 224)
        )
        memory_camera_context = np.zeros((len(memory_indices), 5), dtype=np.float32)
        memory_camera_context[:, 0] = 1.0
        _, memory_context, memory_corrected = encode_fcpg_view(
            system,
            np.asarray(sessions["wrist"].frames[memory_indices]).copy(),
            normalized_memory_states,
            memory_anchors_xyz,
            memory_nominal["wrist"],
            memory_valid["wrist"],
            memory_camera_context,
            device,
            derive_center_from_tips=True,
        )
        memory_data = {
            "indices": memory_indices,
            "offsets": memory_offsets,
            "states": memory_states,
            "normalized_states": normalized_memory_states.astype(np.float32),
            "context": memory_context,
            "corrected": memory_corrected,
        }

    target_descriptor = _target_descriptor(view_frames["column"][-1], args.target)
    target_px = np.asarray(target_descriptor["centroid_px"], dtype=np.float64)
    target_calibration_px = (
        invert_live_similarity_transform(target_px, column_transform)
        if column_transform is not None
        else target_px - column_offset.astype(np.float64)
    )
    target_surface = approximate_column_backprojection_on_plane(
        fits["column"], target_calibration_px, z_base=args.table_z
    )
    current_state = states[-1]
    candidates = generate_ik_candidate_prefixes(
        kinematics,
        current_state,
        target_surface,
        frameskip=frameskip,
        horizon=args.horizon,
    )

    action_center, action_scale, action_normalization = checkpoint_action_normalizer(
        checkpoint, embedded_args
    )
    chunks = chunk_and_normalize_actions(
        candidates.raw_prefixes,
        frameskip=frameskip,
        center=action_center,
        scale=action_scale,
    )
    memory_actions = None
    if memory_data is not None:
        memory_actions = np.stack(
            [
                _normalized_session_action_chunk(
                    sessions["wrist"],
                    int(index),
                    frameskip,
                    action_center,
                    action_scale,
                )
                for index in memory_data["indices"]
            ]
        )
    hold_chunk = np.tile((current_state - action_center) / action_scale, frameskip)
    past_actions = np.stack((hold_chunk, hold_chunk)).astype(np.float32)
    goal_xyz = target_surface.copy()
    goal_xyz[2] += args.goal_clearance
    command_terminal_goal_distance = np.linalg.norm(
        candidates.horizon_xyz[:, -1] - goal_xyz[None], axis=-1
    )
    command_tail_goal_distance = np.mean(
        np.linalg.norm(
            candidates.horizon_xyz[:, -3:] - goal_xyz[None, None], axis=-1
        ),
        axis=1,
    )
    predictions_by_scale: list[dict[str, Any]] = []
    for residual_scale in args.residual_scales:
        _, normalized_prediction = _rollout_candidates(
            model,
            contexts["wrist"],
            normalized_states,
            past_actions,
            chunks,
            residual_scale=residual_scale,
        )
        scale_states = (
            normalized_prediction * state_scale[None, None]
            + state_center[None, None]
        )
        scale_xyz = kinematics.forward(scale_states.reshape(-1, 6)).reshape(
            len(candidates.names), args.horizon, 3
        )
        terminal_goal_distance = np.linalg.norm(
            scale_xyz[:, -1] - goal_xyz[None], axis=-1
        )
        tail_goal_distance = np.mean(
            np.linalg.norm(scale_xyz[:, -3:] - goal_xyz[None, None], axis=-1),
            axis=1,
        )
        disagreement = np.mean(
            np.linalg.norm(scale_xyz - candidates.horizon_xyz, axis=-1), axis=1
        )
        costs = (
            tail_goal_distance
            + 0.25 * command_tail_goal_distance
            + 0.10 * disagreement
        )
        predictions_by_scale.append(
            {
                "residual_scale": float(residual_scale),
                "states": scale_states,
                "xyz": scale_xyz,
                "terminal_goal_distance": terminal_goal_distance,
                "tail_goal_distance": tail_goal_distance,
                "disagreement": disagreement,
                "costs": costs,
            }
        )
    parent_render_prediction = next(
        row
        for row in predictions_by_scale
        if np.isclose(
            row["residual_scale"], args.render_residual_scale, rtol=0.0, atol=1e-9
        )
    )
    memory_prediction = None
    if memory_model is not None:
        if memory_data is None or memory_actions is None:
            raise AssertionError("memory inputs were not constructed")
        tensor = lambda value: torch.as_tensor(value, device=device, dtype=torch.float32)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            _, memory_state_prediction = rollout_multiscale_candidates(
                memory_model,
                tensor(contexts["wrist"]),
                tensor(normalized_states),
                tensor(past_actions),
                tensor(chunks),
                memory_embedding=tensor(memory_data["context"]),
                memory_actions=tensor(memory_actions),
                memory_state=tensor(memory_data["normalized_states"]),
                memory_age=tensor(
                    memory_data["offsets"].astype(np.float32) / float(frameskip)
                ),
                memory_valid=torch.ones(
                    len(memory_data["offsets"]), dtype=torch.bool, device=device
                ),
            )
        memory_states = (
            memory_state_prediction.float().cpu().numpy()
            * state_scale[None, None]
            + state_center[None, None]
        )
        memory_xyz = kinematics.forward(memory_states.reshape(-1, 6)).reshape(
            len(candidates.names), args.horizon, 3
        )
        memory_terminal_goal_distance = np.linalg.norm(
            memory_xyz[:, -1] - goal_xyz[None], axis=-1
        )
        memory_tail_goal_distance = np.mean(
            np.linalg.norm(memory_xyz[:, -3:] - goal_xyz[None, None], axis=-1),
            axis=1,
        )
        memory_disagreement = np.mean(
            np.linalg.norm(memory_xyz - candidates.horizon_xyz, axis=-1), axis=1
        )
        memory_prediction = {
            "variant": (
                "v9-local-multiscale-memory"
                if protocol == V9_PROTOCOL
                else "v6-freeze75-multiscale-memory"
            ),
            "states": memory_states,
            "xyz": memory_xyz,
            "terminal_goal_distance": memory_terminal_goal_distance,
            "tail_goal_distance": memory_tail_goal_distance,
            "disagreement": memory_disagreement,
            "costs": (
                memory_tail_goal_distance
                + 0.25 * command_tail_goal_distance
                + 0.10 * memory_disagreement
            ),
        }
    render_prediction = (
        parent_render_prediction if memory_prediction is None else memory_prediction
    )
    predicted_states = render_prediction["states"]
    predicted_xyz = render_prediction["xyz"]
    predicted_terminal_goal_distance = render_prediction["terminal_goal_distance"]
    predicted_tail_goal_distance = render_prediction["tail_goal_distance"]
    prediction_command_disagreement = render_prediction["disagreement"]
    costs = render_prediction["costs"]
    safety = candidate_safety_checks(
        candidates.raw_prefixes[:, 0],
        current_state,
        max_joint_jump=args.max_command_step,
        raw_prefixes=candidates.raw_prefixes,
        predicted_states=predicted_states,
        position_lower=SO101_POSITION_LOWER,
        position_upper=SO101_POSITION_UPPER,
        max_prefix_step_jump=args.max_command_step,
        prediction_bound_margin=args.prediction_bound_margin,
    )
    ik_safe = candidates.converged
    # A learned state forecast is a ranking signal, not an actuator command.
    # Keep its bound check visible, but never let a biased gripper forecast
    # invalidate an otherwise bounded IK command.
    safe = safety.hard_command_safe & ik_safe
    if not np.any(safe):
        diagnostics = {
            "names": candidates.names,
            "ik_converged": ik_safe,
            "command_safe": safety.hard_command_safe,
            "prediction_plausible": safety.prediction_plausible,
            "first_command_jump": safety.first_command_jump,
            "prefix_max_step_jump": safety.prefix_max_step_jump,
            "command_bounds_ok": safety.command_bounds_ok,
            "prediction_bounds_ok": safety.prediction_bounds_ok,
            "predicted_state_min": np.min(predicted_states, axis=(0, 1)),
            "predicted_state_max": np.max(predicted_states, axis=(0, 1)),
        }
        (output / "safety_failure.json").write_text(
            json.dumps(_jsonable(diagnostics), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            "no IK candidate passed the offline safety contract; see "
            f"{output / 'safety_failure.json'}"
        )
    parent_index = int(
        np.argmin(np.where(safe, parent_render_prediction["costs"], np.inf))
    )
    geometry_index = int(
        np.argmin(np.where(safe, command_terminal_goal_distance, np.inf))
    )
    lewm_index = int(np.argmin(np.where(safe, costs, np.inf)))

    observation_identity = SO101ObservationIdentity(
        session_id=str(args.session.expanduser().resolve()),
        wrist_sequence=final_index,
        column_sequence=final_index,
        grounding_sequence=final_index,
        wrist_timestamp_s=float(sessions["wrist"].timestamps_s[final_index]),
        column_timestamp_s=float(sessions["column"].timestamps_s[final_index]),
    )
    execution_gate = SO101RecedingHorizonGate(commands_per_round=frameskip)
    selected_plan = execution_gate.build_plan(
        observation=observation_identity,
        target_kind=args.target,
        target_xyz=target_surface,
        candidate_name=candidates.names[lewm_index],
        diagnostic_prefix=candidates.raw_prefixes[lewm_index],
    )
    authorized_chunk = execution_gate.authorize(
        selected_plan,
        current_observation=observation_identity,
    )

    trajectory_store = DiskBackedLFUTrajectoryStore(
        args.trajectory_store,
        memory_capacity=args.trajectory_memory_capacity,
    )
    safe_output_name = re.sub(r"[^A-Za-z0-9._-]+", "-", output.name).strip("-.")
    if not safe_output_name:
        safe_output_name = "shadow"
    trajectory_id = (
        f"{safe_output_name}-f{final_index}-"
        f"t{int(round(observation_identity.wrist_timestamp_s * 1_000_000))}"
    )
    trajectory_arrays = {
        "history_states": states,
        "history_timestamps_s": np.asarray(
            sessions["wrist"].timestamps_s[history_indices], dtype=np.float64
        ),
        "nominal_wrist": nominal["wrist"],
        "corrected_wrist": corrected["wrist"],
        "nominal_column": nominal["column"],
        "corrected_column": corrected["column"],
        "target_surface_xyz": target_surface,
        "authorized_commands": authorized_chunk,
        "diagnostic_prefix": selected_plan.diagnostic_prefix,
        "command_horizon_states": candidates.horizon_states[lewm_index],
        "predicted_horizon_states": predicted_states[lewm_index],
        "command_horizon_xyz": candidates.horizon_xyz[lewm_index],
        "predicted_horizon_xyz": predicted_xyz[lewm_index],
    }
    if memory_data is not None:
        trajectory_arrays.update(
            {
                "memory_indices": memory_data["indices"],
                "memory_offsets": memory_data["offsets"],
                "memory_states": memory_data["states"],
                "memory_corrected_wrist": memory_data["corrected"],
            }
        )
    trajectory_record = trajectory_store.append(
        trajectory_id,
        arrays=trajectory_arrays,
        metadata={
            "session": str(args.session.expanduser().resolve()),
            "frame_index": final_index,
            "target_kind": args.target,
            "candidate_name": candidates.names[lewm_index],
            "residual_scale": args.render_residual_scale,
            "model_variant": render_prediction.get("variant", "parent-lewm"),
            "freeze_recent_frames": args.freeze_recent_frames,
            "commands_per_round": frameskip,
            "hardware_commands_sent": False,
        },
    )

    residual_scale_diagnostics = []
    for scale_prediction in predictions_by_scale:
        selected_index = int(
            np.argmin(np.where(safe, scale_prediction["costs"], np.inf))
        )
        state_residual = (
            scale_prediction["states"] - candidates.horizon_states
        )
        residual_scale_diagnostics.append(
            {
                "residual_scale": scale_prediction["residual_scale"],
                "selected_index": selected_index,
                "selected_name": candidates.names[selected_index],
                "geometry_candidate_terminal_goal_distance_m": float(
                    scale_prediction["terminal_goal_distance"][geometry_index]
                ),
                "geometry_candidate_tail_goal_distance_m": float(
                    scale_prediction["tail_goal_distance"][geometry_index]
                ),
                "geometry_candidate_prediction_command_disagreement_m": float(
                    scale_prediction["disagreement"][geometry_index]
                ),
                "geometry_candidate_terminal_joint_residual": state_residual[
                    geometry_index, -1
                ],
                "geometry_candidate_terminal_max_joint_residual": float(
                    np.max(np.abs(state_residual[geometry_index, -1]))
                ),
                "geometry_candidate_per_horizon_goal_distance_m": np.linalg.norm(
                    scale_prediction["xyz"][geometry_index] - goal_xyz[None], axis=-1
                ),
                "geometry_candidate_per_horizon_command_gap_m": np.linalg.norm(
                    scale_prediction["xyz"][geometry_index]
                    - candidates.horizon_xyz[geometry_index],
                    axis=-1,
                ),
            }
        )

    displayed_gripper = nominal["column"][-1, 0]
    _draw_column_target(
        view_frames["column"][-1],
        target_px,
        displayed_gripper,
        (
            corrected["column"][-1, 0]
            if column_live_calibration is not None or column_localization is not None
            else None
        ),
    ).save(output / "column_target.jpg", quality=92)
    command_frames, prediction_frames = _render_selected(
        output,
        current_state,
        candidates.horizon_states[lewm_index],
        predicted_states[lewm_index],
        azimuth=args.azimuth,
        elevation=args.elevation,
        distance=args.distance,
    )

    candidate_rows = []
    for index, name in enumerate(candidates.names):
        candidate_rows.append(
            {
                "index": index,
                "name": name,
                "safe": bool(safe[index]),
                "prediction_plausible": bool(safety.prediction_plausible[index]),
                "predicted_terminal_goal_distance_m": float(
                    predicted_terminal_goal_distance[index]
                ),
                "predicted_tail_goal_distance_m": float(
                    predicted_tail_goal_distance[index]
                ),
                "command_terminal_goal_distance_m": float(
                    command_terminal_goal_distance[index]
                ),
                "command_tail_goal_distance_m": float(
                    command_tail_goal_distance[index]
                ),
                "prediction_command_disagreement_m": float(
                    prediction_command_disagreement[index]
                ),
                "maximum_ik_error_m": float(candidates.terminal_ik_error_m[index]),
                "ik_converged_fraction": float(candidates.converged[index]),
                "first_command_jump": float(safety.first_command_jump[index]),
                "prefix_max_step_jump": float(safety.prefix_max_step_jump[index]),
                "cost": float(costs[index]),
            }
        )
    report = {
        "format": "so101-ik-lewm-shadow-v4",
        "hardware_commands_sent": False,
        "session": str(args.session.resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_protocol": contract.get("protocol"),
        "bridge_checkpoint": str(args.bridge_checkpoint.resolve()),
        "freeze_recent_frames": args.freeze_recent_frames,
        "action_normalization": action_normalization,
        "current_state": current_state,
        "current_gripper_xyz_m": kinematics.forward(current_state),
        "target": {
            "kind": args.target,
            "centroid_px": target_px,
            "calibration_px": target_calibration_px,
            "column_translation_px": column_offset,
            "column_similarity_transform_px": column_transform,
            "surface_xyz_m": target_surface,
            "table_z_m": args.table_z,
            "provider": (
                "live-gripper-similarity-plus-column-affine-inverse"
                if column_live_calibration is not None
                else "live-gripper-translation-plus-column-affine-inverse"
                if column_localization is not None
                else "temporary-column-affine-inverse-without-live-bootstrap"
            ),
            "metric_depth": False,
        },
        "column_gripper_localization": (
            None
            if column_localization is None
            else {
                **column_localization,
                "source": str(
                    args.column_gripper_localization.expanduser().resolve()
                ),
            }
        ),
        "column_live_calibration": column_live_calibration,
        "selection": {
            "geometry_index": geometry_index,
            "geometry_name": candidates.names[geometry_index],
            "parent_index": parent_index,
            "parent_name": candidates.names[parent_index],
            "lewm_index": lewm_index,
            "lewm_name": candidates.names[lewm_index],
            "deployed_variant": render_prediction.get("variant", "parent-lewm"),
            "winners_agree": geometry_index == lewm_index,
            "parent_memory_agree": parent_index == lewm_index,
            "safe_candidates": int(np.sum(safe)),
            "render_residual_scale": args.render_residual_scale,
        },
        "memory_comparison": (
            None
            if memory_prediction is None
            else {
                "memory_offsets_raw_frames": memory_data["offsets"],
                "memory_indices": memory_data["indices"],
                "parent_selected_cost": float(
                    parent_render_prediction["costs"][parent_index]
                ),
                "memory_selected_cost": float(memory_prediction["costs"][lewm_index]),
                "parent_memory_mean_state_difference": float(
                    np.mean(
                        np.abs(
                            parent_render_prediction["states"]
                            - memory_prediction["states"]
                        )
                    )
                ),
                "parent_memory_mean_tcp_difference_m": float(
                    np.mean(
                        np.linalg.norm(
                            parent_render_prediction["xyz"]
                            - memory_prediction["xyz"],
                            axis=-1,
                        )
                    )
                ),
            }
        ),
        "execution_contract": {
            "mode": "five-command-receding-horizon",
            "commands_per_round": frameskip,
            "authorized_commands": authorized_chunk,
            "diagnostic_prefix_commands": selected_plan.diagnostic_horizon_commands,
            "requires_new_wrist_frame": True,
            "requires_new_column_frame": True,
            "requires_new_fcpg_grounding": True,
            "requires_target_redetection": True,
            "h2_to_h8_executable_without_reobservation": False,
            "observation": {
                "session_id": observation_identity.session_id,
                "wrist_sequence": observation_identity.wrist_sequence,
                "column_sequence": observation_identity.column_sequence,
                "grounding_sequence": observation_identity.grounding_sequence,
                "wrist_timestamp_s": observation_identity.wrist_timestamp_s,
                "column_timestamp_s": observation_identity.column_timestamp_s,
            },
        },
        "trajectory_memory": {
            "store": str(trajectory_store.root),
            "trajectory_id": trajectory_record.trajectory_id,
            "record_bytes_uncompressed": trajectory_record.byte_size,
            "memory_policy": "LFU; oldest access breaks equal-frequency ties",
            "memory_capacity_records": args.trajectory_memory_capacity,
            "cached_ids_after_append": trajectory_store.cached_ids,
            "persistent_records_after_append": len(trajectory_store.trajectory_ids),
            "recent_model_context": "latest three contiguous sampled observations",
            "long_history_role": (
                (
                    "v9 task-local causal inference plus persistent replay"
                    if protocol == V9_PROTOCOL
                    else "v6 causal inference plus persistent replay"
                )
                if memory_model is not None
                else "persistent replay only"
            ),
        },
        "candidates": candidate_rows,
        "residual_scale_diagnostics": residual_scale_diagnostics,
        "grounding": {
            view: {
                "nominal": nominal[view][-1],
                "corrected": corrected[view][-1],
                "mean_correction_px": float(
                    np.mean(np.linalg.norm(corrected[view][-1] - nominal[view][-1], axis=-1))
                ),
            }
            for view in VIEWS
        },
        "media": {
            "fps": args.fps,
            "command_frames": command_frames,
            "prediction_frames": prediction_frames,
        },
    }
    (output / "report.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "index.html").write_text(_html_page(_jsonable(report)), encoding="utf-8")
    print(json.dumps(_jsonable(report["selection"]), indent=2, sort_keys=True))
    print(f"html={output / 'index.html'}")


if __name__ == "__main__":
    main()
