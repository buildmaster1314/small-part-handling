"""Prepared FCPG + LeWM planner for interactive SO-101 control."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import uuid
from typing import Any

import numpy as np
import torch

from fcpg.so101_fcpg_lewm_bridge import (
    SO101FCPGConditionedLeWM,
    build_so101_fcpg_corrector,
    infer_so101_fcpg_corrector_variant,
)
from fcpg.modules import sample_feature_map_at_pixels
from fcpg.so101_active_gripper import (
    DEFAULT_MAXIMUM_LIVE_ROTATION_DEG,
    apply_live_center_translation,
    apply_live_similarity_transform,
    invert_live_similarity_transform,
    select_live_moving_tip_similarity_transform,
)
from fcpg.so101_camera_calibration import (
    ProjectionFit,
    approximate_column_backprojection_on_plane,
)
from fcpg.so101_goal_directed_candidates import (
    DEFAULT_APPROACH_VARIANTS,
    DEFAULT_TRANSPORT_VARIANTS,
    generate_fcpg_goal_prefixes,
    generate_gripper_phase_prefixes,
)
from fcpg.so101_ik_candidates import (
    REALTIME_CONTACT_PATH_SPECS,
    generate_ik_candidate_prefixes,
    track_anchor_trajectory,
)
from fcpg.so101_kinematics import SO101MuJoCoKinematics
from fcpg.so101_multiscale_memory import (
    MultiScaleHistoryLeWM,
    rollout_multiscale_candidates,
)
from fcpg.so101_realtime_mpc import (
    ManipulationPhase,
    PhaseEvidence,
    PhaseThresholds,
    PhaseTracker,
    PlanningRoundGuard,
    candidate_phase_mask,
    phase_prediction_bounds_mask,
    target_role_for_phase,
    validate_command_chunk,
)
from fcpg.so101_retrieval_mpc import (
    SO101_POSITION_LOWER,
    SO101_POSITION_UPPER,
    candidate_safety_checks,
    candidate_start_rows,
    chunk_and_normalize_actions,
    raw_action_prefixes,
    rebase_action_prefixes,
    retrieve_diverse_candidates,
    validate_paired_camera_data,
)
from fcpg.so101_state_residual_lewm import StateResidualLeWM
from fcpg.so101_v9 import (
    V6_PROTOCOL,
    V9_PROTOCOL,
    assert_v9_backbone_matches,
    should_use_memory,
)
from fcpg.so101_visual_targets import TARGET_ROLES, TARGET_VIEWS, VisualTargetTracker
from scripts.evaluate_so101_real_shadow_fcpg_lewm import (
    VIEWS,
    current_pixel_goal_cost,
    encode_fcpg_view,
    load_projection_fits,
    project_future_anchor_pixels,
    project_history,
    red_object_phase_distances as demonstration_pickup_phase_distances,
)
from scripts.evaluate_so101_retrieval_mpc import _rollout_candidates, _training_episodes
from scripts.train_so101_lewm import episode_splits, parse_exclusions
from scripts.train_so101_synthetic_fcpg import normalize_rgb
from scripts.visualize_so101_lewm import (
    CacheView,
    build_checkpoint_model,
    checkpoint_action_normalizer,
    checkpoint_contract,
    encode_rows,
    load_caches,
)


@dataclass(frozen=True)
class PreparedPlannerConfig:
    bridge_checkpoint: Path
    calibration: Path
    secondary_cache: Path
    model_checkpoint: Path | None = None
    device: str = "cuda:0"
    batch_size: int = 64
    candidates_per_episode: int = 8
    candidate_min_gap: int = 20
    state_retrieval_weight: float = 1.0
    retrieval_cost_weight: float = 0.05
    object_phase_weight: float = 4.0
    max_object_phase_distance: float = 0.08
    goal_final_steps: int = 3
    cartesian_goal_scale_m: float = 0.20
    pixel_goal_weight: float = 0.10
    max_first_command_jump: float = 8.0
    max_prefix_step_jump: float = 8.0
    prediction_bound_margin: float = 3.0
    fcpg_max_correction_px: float = 18.0
    workspace_margin_m: float = 0.040
    # Extra height above the demonstration transport level while carrying to
    # the drop zone: the carry must clear container walls the planner cannot
    # see (no obstacle model exists), and the release then drops the object
    # in from above.
    release_clearance_m: float = 0.07
    max_column_servo_step_m: float = 0.040
    # Per-round vertical travel cap. The planar hop has always been bounded;
    # without this, height changes (descent, the carry lift to release
    # height) command their full extent in a single round.
    max_column_z_step_m: float = 0.030
    column_descent_radius_m: float = 0.050
    command_chunk: int = 5
    ik_arrival_raw_steps: int = 20
    target_profile: Path | None = None
    recursive_grounding: bool = False
    memory_policy: str = "auto"
    require_live_column_calibration: bool = False
    # Target-conditioned control is the deployable path. The demonstration
    # trajectory mode remains available for replay and temporal-contract tests.
    control_mode: str = "phase_target"

    def __post_init__(self) -> None:
        if self.device.startswith("cuda:1"):
            raise ValueError("CUDA1 is forbidden for this project")
        positive = (
            self.batch_size,
            self.candidates_per_episode,
            self.goal_final_steps,
            self.cartesian_goal_scale_m,
            self.max_first_command_jump,
            self.max_prefix_step_jump,
            self.fcpg_max_correction_px,
            self.workspace_margin_m,
            self.max_column_servo_step_m,
            self.max_column_z_step_m,
            self.column_descent_radius_m,
            self.command_chunk,
            self.ik_arrival_raw_steps,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("planner configuration values must be positive")
        if not np.isfinite(self.pixel_goal_weight) or self.pixel_goal_weight < 0.0:
            raise ValueError("pixel_goal_weight must be finite and non-negative")
        if self.memory_policy not in {"auto", "off", "always"}:
            raise ValueError("memory_policy must be auto, off, or always")
        if self.control_mode not in {"anchor_trajectory", "phase_target"}:
            raise ValueError("control_mode must be anchor_trajectory or phase_target")
        if self.ik_arrival_raw_steps % self.command_chunk:
            raise ValueError("ik_arrival_raw_steps must align to command_chunk")


@dataclass(frozen=True)
class CausalPlanningMemory:
    """Older decision-boundary observations aligned with executed action chunks."""

    frames: dict[str, np.ndarray]
    states: np.ndarray
    raw_actions: np.ndarray
    offsets_raw: np.ndarray

    def __post_init__(self) -> None:
        states = np.asarray(self.states, dtype=np.float32)
        raw_actions = np.asarray(self.raw_actions, dtype=np.float32)
        offsets = np.asarray(self.offsets_raw, dtype=np.int64)
        if states.ndim != 2 or states.shape[1:] != (6,) or len(states) == 0:
            raise ValueError("memory states must have shape [M,6]")
        if raw_actions.shape != (len(states), 5, 6):
            raise ValueError("memory raw_actions must have shape [M,5,6]")
        if offsets.shape != (len(states),):
            raise ValueError("memory offsets_raw must have shape [M]")
        if np.any(offsets <= 0) or np.any(np.diff(offsets) <= 0):
            raise ValueError("memory offsets_raw must be positive and increasing")
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(raw_actions)):
            raise ValueError("memory states and actions must be finite")
        if set(self.frames) != set(VIEWS):
            raise ValueError("memory frames must contain wrist and column views")
        for view in VIEWS:
            values = np.asarray(self.frames[view], dtype=np.uint8)
            if values.shape != (len(states), 224, 224, 3):
                raise ValueError(f"{view} memory frames have an incompatible shape")


@dataclass(frozen=True)
class DemonstrationProgress:
    """Next action-prefix location after a completed real command chunk."""

    episode: int
    next_frame: int
    completed_chunks: int


@dataclass
class RealtimePlan:
    guard: PlanningRoundGuard
    phase: ManipulationPhase
    phase_status: dict[str, Any]
    current_state: np.ndarray
    frames: dict[str, np.ndarray]
    nominal: dict[str, np.ndarray]
    correction_seed: dict[str, np.ndarray]
    corrected: dict[str, np.ndarray]
    valid: dict[str, np.ndarray]
    grounding_diagnostics: dict[str, dict[str, Any]]
    target_descriptors: dict[str, dict[str, Any]]
    target_profile: dict[str, Any]
    target: dict[str, Any]
    selected_index: int | None
    selected_chunk: np.ndarray | None
    selected_prefix: np.ndarray | None
    predicted_states: np.ndarray | None
    commanded_states: np.ndarray | None
    predicted_gripper_px: np.ndarray | None
    commanded_gripper_px: np.ndarray | None
    predicted_anchor_px: np.ndarray | None
    commanded_anchor_px: np.ndarray | None
    candidate_rows: list[dict[str, Any]]
    executable: bool
    blocked_reasons: tuple[str, ...]
    model_contract: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "round_id": self.guard.round_id,
            "phase": self.phase.value,
            "phase_status": self.phase_status,
            "current_state": self.current_state,
            "target": self.target,
            "selected_index": self.selected_index,
            "selected_chunk": self.selected_chunk,
            "selected_prefix": self.selected_prefix,
            "predicted_states": self.predicted_states,
            "commanded_states": self.commanded_states,
            "predicted_gripper_px": self.predicted_gripper_px,
            "commanded_gripper_px": self.commanded_gripper_px,
            "predicted_anchor_px": self.predicted_anchor_px,
            "commanded_anchor_px": self.commanded_anchor_px,
            "candidates": self.candidate_rows,
            "executable": self.executable,
            "blocked_reasons": self.blocked_reasons,
            "model_contract": self.model_contract,
            "grounding": {
                view: {
                    "nominal": self.nominal[view][-1],
                    "correction_seed": self.correction_seed[view][-1],
                    "corrected": self.corrected[view][-1],
                    "valid": self.valid[view][-1],
                    "prior_source": self.grounding_diagnostics[view][
                        "prior_source"
                    ],
                    "mean_local_correction_px": float(
                        np.mean(
                            np.linalg.norm(
                                self.corrected[view][-1]
                                - self.correction_seed[view][-1],
                                axis=-1,
                            )
                        )
                    ),
                    "mean_total_correction_px": float(
                        np.mean(
                            np.linalg.norm(
                                self.corrected[view][-1] - self.nominal[view][-1],
                                axis=-1,
                            )
                        )
                    ),
                    "mean_correction_px": float(
                        np.mean(
                            np.linalg.norm(
                                self.corrected[view][-1] - self.nominal[view][-1],
                                axis=-1,
                            )
                        )
                    ),
                    "fk_delta_px": self.grounding_diagnostics[view]["fk_delta_px"],
                    "posterior_offset_px": self.grounding_diagnostics[view][
                        "posterior_offset_px"
                    ],
                }
                for view in VIEWS
            },
            "targets": self.target_descriptors,
            "target_profile": self.target_profile,
        }


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
    if isinstance(value, ManipulationPhase):
        return value.value
    return value


def plan_json(plan: RealtimePlan) -> dict[str, Any]:
    return _jsonable(plan.summary())


def column_live_pixel_to_base_xyz(
    fit: ProjectionFit,
    target_live_px: np.ndarray,
    *,
    z_base: float,
    live_transform_px: np.ndarray | None = None,
) -> np.ndarray:
    """Backproject one live column pixel after undoing live image calibration."""

    live = np.asarray(target_live_px, dtype=np.float32)
    if live.shape != (2,) or not np.all(np.isfinite(live)):
        raise ValueError("target_live_px must be one finite pixel")
    calibration_px = (
        live
        if live_transform_px is None
        else invert_live_similarity_transform(live, live_transform_px)
    )
    return np.asarray(
        approximate_column_backprojection_on_plane(
            fit, calibration_px, z_base=float(z_base)
        ),
        dtype=np.float32,
    )


def bounded_column_servo_target_xyz(
    fit: ProjectionFit,
    current_live_px: np.ndarray,
    target_live_px: np.ndarray,
    current_base_xyz: np.ndarray,
    *,
    target_z_base: float,
    max_planar_step_m: float,
    descent_radius_m: float,
    standoff_z_base: float | None,
    live_transform_px: np.ndarray | None = None,
    max_z_step_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert a column-image error into one bounded Cartesian servo target.

    The absolute affine backprojection is poorly conditioned far from the
    locally calibrated gripper. Subtracting backprojections of the target and
    current gripper cancels the affine offset; bounding that relative motion
    keeps each closed-loop IK request inside the locally observed region.
    """

    current_xyz = np.asarray(current_base_xyz, dtype=np.float32)
    if current_xyz.shape != (3,) or not np.all(np.isfinite(current_xyz)):
        raise ValueError("current_base_xyz must be one finite XYZ vector")
    if (
        not np.isfinite(target_z_base)
        or not np.isfinite(max_planar_step_m)
        or max_planar_step_m <= 0.0
        or not np.isfinite(descent_radius_m)
        or descent_radius_m <= 0.0
    ):
        raise ValueError("column servo distances must be finite and positive")
    if standoff_z_base is not None and not np.isfinite(standoff_z_base):
        raise ValueError("standoff_z_base must be finite when provided")

    current_on_plane = column_live_pixel_to_base_xyz(
        fit,
        current_live_px,
        z_base=float(target_z_base),
        live_transform_px=live_transform_px,
    )
    target_on_plane = column_live_pixel_to_base_xyz(
        fit,
        target_live_px,
        z_base=float(target_z_base),
        live_transform_px=live_transform_px,
    )
    planar_delta = np.asarray(
        target_on_plane[:2] - current_on_plane[:2], dtype=np.float32
    )
    planar_error_m = float(np.linalg.norm(planar_delta))
    step_delta = planar_delta.copy()
    if planar_error_m > max_planar_step_m:
        step_delta *= float(max_planar_step_m / planar_error_m)

    relative_target = current_xyz.copy()
    relative_target[:2] += planar_delta
    relative_target[2] = float(target_z_base)
    planning_target = current_xyz.copy()
    planning_target[:2] += step_delta
    if standoff_z_base is not None and planar_error_m > descent_radius_m:
        planning_target[2] = max(
            float(current_xyz[2]), float(standoff_z_base)
        )
    else:
        planning_target[2] = float(target_z_base)
    if max_z_step_m is not None:
        if not np.isfinite(max_z_step_m) or max_z_step_m <= 0.0:
            raise ValueError("max_z_step_m must be finite and positive")
        z_delta = float(
            np.clip(
                planning_target[2] - current_xyz[2],
                -max_z_step_m,
                max_z_step_m,
            )
        )
        planning_target[2] = float(current_xyz[2]) + z_delta
    return planning_target, relative_target, planar_error_m


def cartesian_terminal_goal_cost(
    predicted_xyz: np.ndarray,
    target_xyz: np.ndarray,
    *,
    final_steps: int,
    scale_m: float,
) -> np.ndarray:
    """Score the LeWM-predicted terminal tool position in base-frame metres."""

    predicted = np.asarray(predicted_xyz, dtype=np.float32)
    target = np.asarray(target_xyz, dtype=np.float32)
    if predicted.ndim != 3 or predicted.shape[-1] != 3:
        raise ValueError("predicted_xyz must have shape [candidates,horizon,3]")
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("target_xyz must be one finite XYZ vector")
    if final_steps <= 0 or final_steps > predicted.shape[1]:
        raise ValueError("final_steps is outside the prediction horizon")
    if not np.isfinite(scale_m) or scale_m <= 0.0:
        raise ValueError("scale_m must be finite and positive")
    error = (predicted[:, -final_steps:] - target[None, None]) / float(scale_m)
    return np.mean(np.sum(np.square(error), axis=-1), axis=-1).astype(np.float32)


@dataclass
class _AnchorPosterior:
    state_latest: np.ndarray
    frame_latest: np.ndarray
    nominal_latest_px: np.ndarray
    seed_history_px: np.ndarray
    corrected_latest_px: np.ndarray


@dataclass
class GroundingObservation:
    anchors_xyz: np.ndarray
    nominal: dict[str, np.ndarray]
    correction_seed: dict[str, np.ndarray]
    corrected: dict[str, np.ndarray]
    valid: dict[str, np.ndarray]
    contexts: dict[str, np.ndarray]
    diagnostics: dict[str, dict[str, Any]]
    blocked_reasons: tuple[str, ...]


def recursive_anchor_seed(
    nominal_history_px: np.ndarray,
    current_state: np.ndarray,
    current_frame: np.ndarray,
    posterior: _AnchorPosterior | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Propagate one calibrated anchor posterior with measured FK displacement."""

    nominal = np.asarray(nominal_history_px, dtype=np.float32)
    state = np.asarray(current_state, dtype=np.float32)
    frame = np.asarray(current_frame, dtype=np.uint8)
    if nominal.ndim != 3 or nominal.shape[-2:] != (3, 2):
        raise ValueError("nominal_history_px must have shape [T,3,2]")
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("current_state must be a finite six-vector")
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError("current_frame must be an RGB image")
    if not np.all(np.isfinite(nominal)):
        raise ValueError("nominal_history_px must be finite")

    if posterior is None:
        seed = nominal.copy()
        source = "calibrated_nominal"
        fk_delta = np.zeros((3, 2), dtype=np.float32)
        posterior_offset = np.zeros((3, 2), dtype=np.float32)
    elif np.array_equal(state, posterior.state_latest) and np.array_equal(
        frame, posterior.frame_latest
    ):
        seed = posterior.seed_history_px.copy()
        source = "same_observation_replay"
        fk_delta = nominal[-1] - posterior.nominal_latest_px
        posterior_offset = seed[-1] - nominal[-1]
    else:
        fk_delta = nominal[-1] - posterior.nominal_latest_px
        posterior_offset = (
            posterior.corrected_latest_px - posterior.nominal_latest_px
        )
        seed = nominal + posterior_offset[None]
        source = "posterior_plus_fk_delta"

    return seed.astype(np.float32), {
        "prior_source": source,
        "fk_delta_px": fk_delta.astype(np.float32),
        "posterior_offset_px": posterior_offset.astype(np.float32),
    }


class PreparedSO101RealtimePlanner:
    """Load model/data once and plan repeatedly from fresh real observations."""

    def __init__(self, config: PreparedPlannerConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        bridge_path = config.bridge_checkpoint.expanduser().resolve()
        bridge_payload = torch.load(bridge_path, map_location="cpu", weights_only=False)
        source_checkpoint_path = Path(
            bridge_payload["args"]["checkpoint"]
        ).resolve()
        (
            source_checkpoint,
            source_embedded_args,
            cache_paths,
            source_frameskip,
            source_history,
            source_checkpoint_sha,
        ) = checkpoint_contract(source_checkpoint_path, None)
        checkpoint_path = (
            config.model_checkpoint.expanduser().resolve()
            if config.model_checkpoint is not None
            else source_checkpoint_path
        )
        (
            checkpoint,
            embedded_args,
            _,
            frameskip,
            history,
            checkpoint_sha,
        ) = checkpoint_contract(checkpoint_path, None)
        if (source_frameskip, source_history) != (frameskip, history):
            raise ValueError(
                "bridge source and dynamics checkpoint sequence contracts differ"
            )
        if frameskip != 5 or history != 3:
            raise ValueError("real-time planner requires frameskip=5 and history=3")
        if config.command_chunk != frameskip:
            raise ValueError("command_chunk must match the checkpoint frameskip")
        grounding_model = build_checkpoint_model(
            source_checkpoint, source_frameskip, source_history, self.device
        )
        if not isinstance(grounding_model, StateResidualLeWM):
            raise TypeError("FCPG source checkpoint must be a state-residual LeWM")
        loaded_model = build_checkpoint_model(
            checkpoint, frameskip, history, self.device
        )
        self.memory_model: MultiScaleHistoryLeWM | None = None
        self.memory_offsets_raw: tuple[int, ...] = ()
        selected_protocol = checkpoint["run_config"]["contract"].get("protocol")
        if isinstance(loaded_model, MultiScaleHistoryLeWM):
            if selected_protocol not in {V6_PROTOCOL, V9_PROTOCOL}:
                raise ValueError("real-time planner supports V6 or V9 memory models")
            if selected_protocol == V9_PROTOCOL:
                assert_v9_backbone_matches(
                    checkpoint["model"], source_checkpoint["model"]
                )
                # V9's embedded backbone is exactly the task-local model used
                # by the FCPG bridge. Use that one object for grounding, bank
                # encoding, recent rollout, and memory-augmented rollout.
                grounding_model = loaded_model.backbone
            self.memory_model = loaded_model
            self.memory_offsets_raw = tuple(
                int(value)
                for value in checkpoint["run_config"]["contract"]["memory_offsets"]
            )
            model = loaded_model.backbone
        else:
            model = loaded_model
        if not isinstance(model, StateResidualLeWM):
            raise TypeError("real-time planner requires state-residual LeWM dynamics")
        corrector = build_so101_fcpg_corrector(
            grounding_model,
            camera_context_dim=5,
            variant=infer_so101_fcpg_corrector_variant(bridge_payload),
        )
        self.system = SO101FCPGConditionedLeWM(
            grounding_model,
            corrector=corrector,
            camera_context_dim=5,
            freeze_model=True,
        ).to(self.device)
        self.system.corrector.load_state_dict(bridge_payload["corrector"], strict=True)
        self.system.bridge.load_state_dict(bridge_payload["bridge"], strict=True)
        self.system.eval()
        self.grounding_model = grounding_model
        self.model = model
        self.frameskip = frameskip
        self.history = history
        self.horizon = 8
        self.horizon_raw = frameskip * self.horizon
        if config.ik_arrival_raw_steps > self.horizon_raw:
            raise ValueError("ik_arrival_raw_steps exceeds the prediction horizon")

        caches = load_caches(cache_paths)
        if len(caches) != 1:
            raise ValueError("planner checkpoint must bind exactly one primary cache")
        self.primary: CacheView = caches[0]
        self.secondary: CacheView = load_caches(
            [config.secondary_cache.expanduser().resolve()]
        )[0]
        validate_paired_camera_data(self.primary.data, self.secondary.data)

        contract = checkpoint["run_config"]["contract"]
        self.state_center = np.asarray(
            contract["normalization_center"], dtype=np.float32
        )
        self.state_scale = np.asarray(
            contract["normalization_scale"], dtype=np.float32
        )
        (
            self.action_center,
            self.action_scale,
            self.action_normalization,
        ) = checkpoint_action_normalizer(checkpoint, embedded_args)
        self.kinematics = SO101MuJoCoKinematics(
            Path(__file__).resolve().parents[1]
            / "third_party/so_arm100/Simulation/SO101/so101_new_calib.xml"
        )
        self.fits = load_projection_fits(config.calibration)

        assignments = episode_splits(
            [self.primary.data],
            parse_exclusions(source_embedded_args.get("exclude_episode", [])),
            split_seed=int(source_embedded_args["split_seed"]),
            val_fraction=float(source_embedded_args["val_fraction"]),
        )
        self.train_episodes = _training_episodes(self.primary, assignments)
        self.bank_rows = candidate_start_rows(
            self.primary.data.episodes,
            self.primary.data.frame_indices,
            self.train_episodes,
            horizon_raw=self.horizon_raw,
            stride=self.frameskip,
        )
        self.bank_states = np.asarray(
            self.primary.data.states[self.bank_rows], dtype=np.float32
        )
        self.bank_episodes = np.asarray(
            self.primary.data.episodes[self.bank_rows], dtype=np.int64
        )
        self.bank_frames = np.asarray(
            self.primary.data.frame_indices[self.bank_rows], dtype=np.int64
        )
        self._bank_index_by_episode_frame: dict[tuple[int, int], int] = {}
        for index, (episode, frame) in enumerate(
            zip(self.bank_episodes, self.bank_frames, strict=True)
        ):
            key = (int(episode), int(frame))
            if key in self._bank_index_by_episode_frame:
                raise ValueError(
                    "demonstration bank contains duplicate episode/frame candidates"
                )
            self._bank_index_by_episode_frame[key] = index
        self.bank_prefixes = raw_action_prefixes(
            self.primary.data.actions,
            self.bank_rows,
            horizon_raw=self.horizon_raw,
        )
        future_offsets = np.arange(
            self.frameskip,
            self.horizon_raw + 1,
            self.frameskip,
            dtype=np.int64,
        )
        self.bank_future_rows = self.bank_rows[:, None] + future_offsets[None]
        if not np.all(
            np.asarray(self.primary.data.episodes)[self.bank_future_rows]
            == self.bank_episodes[:, None]
        ):
            raise ValueError("demonstration future trajectory crosses an episode")
        frame_delta = (
            np.asarray(self.primary.data.frame_indices)[self.bank_future_rows]
            - np.asarray(self.primary.data.frame_indices)[self.bank_rows, None]
        )
        if not np.array_equal(
            frame_delta,
            np.broadcast_to(future_offsets, frame_delta.shape),
        ):
            raise ValueError("demonstration future trajectory is not frame-aligned")
        self.bank_future_states = np.asarray(
            self.primary.data.states[self.bank_future_rows], dtype=np.float32
        )
        self.primary_bank = encode_rows(
            self.model, self.primary, self.bank_rows, config.batch_size
        )
        self.secondary_bank = encode_rows(
            self.model, self.secondary, self.bank_rows, config.batch_size
        )
        _, bank_nominal, _ = project_history(
            self.bank_states,
            self.fits,
            self.kinematics,
            image_size=(224, 224),
        )
        self.bank_wrist_gripper_px = np.asarray(
            bank_nominal["wrist"][:, 0], dtype=np.float32
        )

        train_rows = np.concatenate(
            [
                np.flatnonzero(
                    np.asarray(self.primary.data.episodes) == episode
                )
                for episode in sorted(self.train_episodes)
            ]
        )
        train_states = np.asarray(
            self.primary.data.states[train_rows], dtype=np.float32
        )
        workspace = self.kinematics.forward(train_states)
        train_anchor_centers = self.kinematics.forward_anchors(train_states)[:, 0]
        self.contact_height_m = float(np.percentile(train_anchor_centers[:, 2], 10))
        self.transport_height_m = float(np.percentile(train_anchor_centers[:, 2], 65))
        if (
            not np.isfinite(config.release_clearance_m)
            or config.release_clearance_m < 0
        ):
            raise ValueError("release_clearance_m must be finite and >= 0")
        self.release_height_m = (
            self.transport_height_m + config.release_clearance_m
        )
        self.workspace_lower = np.min(workspace, axis=0) - config.workspace_margin_m
        self.workspace_upper = np.max(workspace, axis=0) + config.workspace_margin_m
        gripper = train_states[:, -1]
        self.gripper_calibration_closed = float(np.percentile(gripper, 1))
        self.gripper_calibration_open = float(np.percentile(gripper, 99))
        if self.gripper_calibration_open - self.gripper_calibration_closed < 20.0:
            raise ValueError(
                "training data does not contain a safe, observable gripper pulse"
            )
        self.thresholds = PhaseThresholds(
            pickup_near_px=28.0,
            drop_near_px=35.0,
            pickup_near_m=0.018,
            drop_near_m=0.025,
            gripper_closed=float(np.percentile(gripper, 25)),
            gripper_open=float(np.percentile(gripper, 80)),
            grasp_roi_px=30.0,
            confirmations=2,
            gripper_empty_closed=self.gripper_calibration_closed + 2.0,
        )
        self.phase_tracker = PhaseTracker(self.thresholds)
        self._transport_squeeze_chunks = 0
        self._grasp_empty_close_events = 0
        self._grasp_recovery_engaged = False
        self.target_tracker = VisualTargetTracker(config.target_profile)
        self.live_anchor_offsets_px: dict[str, np.ndarray] = {}
        self.live_anchor_transforms_px: dict[str, np.ndarray] = {}
        self._anchor_posteriors: dict[str, _AnchorPosterior] = {}
        self._demonstration_progress: DemonstrationProgress | None = None
        self.model_contract = {
            "bridge_checkpoint": bridge_path,
            "bridge_source_checkpoint": source_checkpoint_path,
            "bridge_source_checkpoint_sha256": source_checkpoint_sha,
            "bridge_source_endpoint": int(source_checkpoint.get("step", -1)),
            "dynamics_checkpoint": checkpoint_path,
            "dynamics_checkpoint_sha256": checkpoint_sha,
            "dynamics_endpoint": int(checkpoint.get("step", -1)),
            "dynamics_protocol": selected_protocol,
            "dynamics_variant": (
                "v9_local_multiscale"
                if selected_protocol == V9_PROTOCOL
                else "v6_multiscale_memory"
                if selected_protocol == V6_PROTOCOL
                else "state_residual_backbone"
            ),
            "grounding_checkpoint": source_checkpoint_path,
            "grounding_checkpoint_sha256": source_checkpoint_sha,
            "grounding_contract": "legacy_fcpq_visual_semantics_for_current_anchor_measurement",
            "dynamics_context": (
                "shared_task_local_fcpg_backbone"
                if selected_protocol == V9_PROTOCOL
                else "separate_v6_backbone_with_fcpg_bridge_context"
            ),
            "memory_policy": config.memory_policy,
            "memory_offsets_raw": list(self.memory_offsets_raw),
            "control_mode": config.control_mode,
            "frameskip": frameskip,
            "history": history,
            "horizon": self.horizon,
            "action_normalization": self.action_normalization,
            "phase_thresholds": {
                "pickup_near_px": self.thresholds.pickup_near_px,
                "drop_near_px": self.thresholds.drop_near_px,
                "pickup_near_m": self.thresholds.pickup_near_m,
                "drop_near_m": self.thresholds.drop_near_m,
                "gripper_closed": self.thresholds.gripper_closed,
                "gripper_open": self.thresholds.gripper_open,
                "grasp_roi_px": self.thresholds.grasp_roi_px,
            },
            "gripper_calibration_positions": {
                "closed": self.gripper_calibration_closed,
                "open": self.gripper_calibration_open,
                "source": "train-only gripper position percentiles 1/99",
            },
            "workspace_lower_m": self.workspace_lower,
            "workspace_upper_m": self.workspace_upper,
            "workspace_margin_m": config.workspace_margin_m,
            "max_column_servo_step_m": config.max_column_servo_step_m,
            "column_descent_radius_m": config.column_descent_radius_m,
            "approach_candidates": (
                "bounded relative column-image servo, MuJoCo IK, and retrieved "
                "demonstrations"
            ),
            "contact_height_m": self.contact_height_m,
            "transport_height_m": self.transport_height_m,
            "release_height_m": self.release_height_m,
            "cartesian_goal_scale_m": config.cartesian_goal_scale_m,
            "pixel_goal_weight": config.pixel_goal_weight,
            "target_contract": "operator-configured pickup/drop_zone appearance references",
            "live_anchor_offsets_px": {},
            "live_anchor_transforms_px": {},
            "require_live_column_calibration": bool(
                config.require_live_column_calibration
            ),
            "recursive_grounding": {
                "enabled": bool(config.recursive_grounding),
                "propagation": (
                    "previous corrected anchor plus measured calibrated-FK "
                    "pixel displacement"
                ),
                "measurement_update": "current-frame local FCPG",
                "manual_label_supervision": False,
            },
        }

    def reset_grounding_posterior(self) -> None:
        """Forget recursive image anchors after a discontinuous robot reset."""

        self._anchor_posteriors = {}

    def reset_task_progress(self) -> None:
        """Forget demonstration alignment after a discontinuous task reset."""

        self._demonstration_progress = None

    def commit_executed_plan(
        self,
        plan: RealtimePlan,
        *,
        command_count: int,
    ) -> dict[str, int] | None:
        """Advance demonstration time only after commands completed on the robot."""

        if plan.phase == ManipulationPhase.TRANSPORT:
            self._transport_squeeze_chunks += 1
        if plan.phase != ManipulationPhase.TRAJECTORY:
            return None
        if command_count != self.config.command_chunk:
            raise ValueError(
                "trajectory progress requires one complete planner command chunk"
            )
        if plan.selected_index is None:
            raise ValueError("cannot advance an unselected trajectory plan")
        if not 0 <= plan.selected_index < len(plan.candidate_rows):
            raise ValueError("selected trajectory candidate index is invalid")
        selected = plan.candidate_rows[plan.selected_index]
        if selected.get("source") not in {
            "implicit_successful_demonstration",
            "locked_demonstration_continuation",
        }:
            raise ValueError("selected trajectory candidate has no demonstration source")
        episode = int(selected["episode"])
        frame = int(selected["frame"])
        if self._demonstration_progress is not None and (
            episode != self._demonstration_progress.episode
            or frame != self._demonstration_progress.next_frame
        ):
            raise ValueError(
                "executed trajectory candidate does not match committed progress"
            )
        next_frame = frame + command_count
        completed_chunks = (
            1
            if self._demonstration_progress is None
            else self._demonstration_progress.completed_chunks + 1
        )
        self._demonstration_progress = DemonstrationProgress(
            episode=episode,
            next_frame=next_frame,
            completed_chunks=completed_chunks,
        )
        return {
            "episode": episode,
            "executed_source_frame": frame,
            "next_source_frame": next_frame,
            "completed_chunks": completed_chunks,
        }

    def set_live_anchor_center(
        self,
        view: str,
        center_px: np.ndarray,
        state: np.ndarray,
        *,
        anchor_index: int = 0,
    ) -> np.ndarray:
        """Bind a local projection offset from one observed physical anchor."""

        _, offset = self.set_live_anchor_candidates(
            view,
            np.asarray(center_px, dtype=np.float32)[None],
            state,
            anchor_index=anchor_index,
        )
        return offset

    def set_live_anchor_candidates(
        self,
        view: str,
        candidates_px: np.ndarray,
        state: np.ndarray,
        *,
        anchor_index: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select the observed envelope endpoint nearest one nominal anchor."""

        if view not in VIEWS:
            raise ValueError(f"unsupported live-anchor view: {view}")
        candidates = np.asarray(candidates_px, dtype=np.float32)
        current_state = np.asarray(state, dtype=np.float32)
        if (
            candidates.ndim != 2
            or candidates.shape[1:] != (2,)
            or len(candidates) == 0
            or not np.all(np.isfinite(candidates))
        ):
            raise ValueError("live anchor candidates must be finite [N,2]")
        if current_state.shape != (6,) or not np.all(np.isfinite(current_state)):
            raise ValueError("live anchor calibration state must be a finite six-vector")
        _, nominal, _ = project_history(
            current_state[None],
            self.fits,
            self.kinematics,
            image_size=(224, 224),
        )
        anchor = np.asarray(nominal[view][-1, anchor_index], dtype=np.float32)
        offsets = candidates - anchor[None]
        selected_index = int(np.argmin(np.linalg.norm(offsets, axis=1)))
        live_center = candidates[selected_index]
        _, offset = apply_live_center_translation(
            nominal[view], live_center, anchor_index=anchor_index
        )
        if float(np.linalg.norm(offset)) > 35.0:
            raise ValueError(
                "live anchor offset exceeds the 35 px local calibration envelope: "
                f"view={view}, anchor_index={anchor_index}, offset={offset.tolist()}"
            )
        self.live_anchor_offsets_px[view] = offset.copy()
        transform = np.asarray(
            [[1.0, 0.0, offset[0]], [0.0, 1.0, offset[1]]],
            dtype=np.float32,
        )
        self.live_anchor_transforms_px[view] = transform
        self.reset_grounding_posterior()
        self.model_contract["live_anchor_offsets_px"] = {
            name: values.tolist()
            for name, values in sorted(self.live_anchor_offsets_px.items())
        }
        self.model_contract["live_anchor_transforms_px"] = {
            name: values.tolist()
            for name, values in sorted(self.live_anchor_transforms_px.items())
        }
        return live_center.copy(), offset.copy()

    def set_live_moving_tip_track(
        self,
        view: str,
        moving_tip_track_candidates_px: np.ndarray,
        open_state: np.ndarray,
        closed_state: np.ndarray,
        *,
        center_anchor_index: int = 0,
        fixed_tip_anchor_index: int = 1,
        moving_tip_anchor_index: int = 2,
        minimum_scale: float = 0.5,
        maximum_scale: float = 1.5,
        maximum_rotation_deg: float = DEFAULT_MAXIMUM_LIVE_ROTATION_DEG,
        maximum_center_offset_px: float = 35.0,
    ) -> dict[str, Any]:
        """Fit image geometry from one moving fingertip observed open and closed."""

        if view not in VIEWS:
            raise ValueError(f"unsupported live-anchor view: {view}")
        candidates = np.asarray(moving_tip_track_candidates_px, dtype=np.float32)
        open_joint_state = np.asarray(open_state, dtype=np.float32)
        closed_joint_state = np.asarray(closed_state, dtype=np.float32)
        if (
            open_joint_state.shape != (6,)
            or closed_joint_state.shape != (6,)
            or not np.all(np.isfinite(open_joint_state))
            or not np.all(np.isfinite(closed_joint_state))
        ):
            raise ValueError(
                "open and closed calibration states must be finite six-vectors"
            )
        calibration_states = np.stack((open_joint_state, closed_joint_state))
        _, nominal, _ = project_history(
            calibration_states,
            self.fits,
            self.kinematics,
            image_size=(224, 224),
        )
        nominal_anchors = np.asarray(nominal[view], dtype=np.float32)
        selected = select_live_moving_tip_similarity_transform(
            nominal_anchors,
            candidates,
            center_anchor_index=center_anchor_index,
            fixed_tip_anchor_index=fixed_tip_anchor_index,
            moving_tip_anchor_index=moving_tip_anchor_index,
            minimum_scale=minimum_scale,
            maximum_scale=maximum_scale,
            maximum_rotation_deg=maximum_rotation_deg,
            maximum_center_offset_px=maximum_center_offset_px,
        )
        transform = np.asarray(selected["transform_px"], dtype=np.float32)
        self.live_anchor_transforms_px[view] = transform.copy()
        self.reset_grounding_posterior()
        self.live_anchor_offsets_px.pop(view, None)
        self.model_contract["live_anchor_offsets_px"] = {
            name: values.tolist()
            for name, values in sorted(self.live_anchor_offsets_px.items())
        }
        self.model_contract["live_anchor_transforms_px"] = {
            name: values.tolist()
            for name, values in sorted(self.live_anchor_transforms_px.items())
        }
        return {
            **selected,
            "open_state": open_joint_state.copy(),
            "closed_state": closed_joint_state.copy(),
        }

    def set_target_reference(
        self,
        role: str,
        view: str,
        frame: np.ndarray,
        centroid_px: np.ndarray,
        *,
        radius_px: float,
    ) -> dict[str, Any]:
        if role not in TARGET_ROLES or view not in TARGET_VIEWS:
            raise ValueError("unsupported target role or view")
        return self.target_tracker.register(
            role,
            view,
            frame,
            centroid_px,
            radius_px=radius_px,
        )

    def clear_target_reference(self, role: str, view: str | None = None) -> None:
        self.target_tracker.clear(role, view)

    def target_status(self) -> dict[str, Any]:
        return self.target_tracker.status()

    @torch.inference_mode()
    def observe_grounding(
        self,
        frames: dict[str, np.ndarray],
        states: np.ndarray,
        *,
        commit: bool = True,
    ) -> GroundingObservation:
        """Encode one dual-view observation with recursive local FCPG updates."""

        state_history = np.asarray(states, dtype=np.float32)
        if state_history.shape != (self.history, 6):
            raise ValueError("states must have shape [3,6]")
        prepared_frames: dict[str, np.ndarray] = {}
        for view in VIEWS:
            values = np.asarray(frames[view], dtype=np.uint8)
            if values.shape != (self.history, 224, 224, 3):
                raise ValueError(f"{view} frames must have shape [3,224,224,3]")
            prepared_frames[view] = values

        current_state = state_history[-1]
        normalized_states = (state_history - self.state_center) / self.state_scale
        anchors_xyz, nominal, nominal_valid = project_history(
            state_history,
            self.fits,
            self.kinematics,
            image_size=(224, 224),
        )
        for view, transform in self.live_anchor_transforms_px.items():
            raw_nominal = nominal[view]
            nominal[view] = apply_live_similarity_transform(
                raw_nominal, transform
            )
            # The projection fit's image bounds belong to the ORIGINAL
            # calibration session's camera aim. Once a live transform exists
            # the camera has been re-aimed, and a fingertip a few pixels past
            # the stale frame edge can be fully visible live (the affine fit
            # extrapolates linearly). Validity is therefore finiteness plus
            # the LIVE-frame bounds; the seed and corrected checks downstream
            # already operate in the live frame.
            nominal_valid[view] = np.all(
                np.isfinite(raw_nominal), axis=-1
            ) & np.all(
                (nominal[view] >= 0.0) & (nominal[view] < 224.0), axis=-1
            )

        contexts: dict[str, np.ndarray] = {}
        correction_seed: dict[str, np.ndarray] = {}
        corrected: dict[str, np.ndarray] = {}
        valid: dict[str, np.ndarray] = {}
        diagnostics: dict[str, dict[str, Any]] = {}
        blocked: list[str] = []
        pending: dict[str, _AnchorPosterior] = {}
        safe_to_commit: dict[str, bool] = {}
        posteriors = (
            getattr(self, "_anchor_posteriors", {})
            if self.config.recursive_grounding
            else {}
        )

        for view_index, view in enumerate(VIEWS):
            seed, row_diagnostics = recursive_anchor_seed(
                nominal[view],
                current_state,
                prepared_frames[view][-1],
                posteriors.get(view),
            )
            seed_valid = np.asarray(nominal_valid[view], dtype=bool) & np.all(
                (seed >= 0.0) & (seed < 224.0), axis=-1
            )
            camera_context = np.zeros((self.history, 5), dtype=np.float32)
            camera_context[:, view_index] = 1.0
            _, bridged, points = encode_fcpg_view(
                self.system,
                prepared_frames[view],
                normalized_states,
                anchors_xyz,
                nominal[view],
                seed_valid,
                camera_context,
                self.device,
                correction_seed_px=seed,
                derive_center_from_tips=True,
            )
            corrected_valid = seed_valid & np.all(
                (points >= 0.0) & (points < 224.0), axis=-1
            )
            corrected_valid[:, 0] = (
                corrected_valid[:, 1] & corrected_valid[:, 2]
            )
            local_correction = np.linalg.norm(points[-1] - seed[-1], axis=-1)
            view_safe = True
            if not np.all(corrected_valid[-1]):
                blocked.append(f"{view}_anchor_out_of_bounds")
                view_safe = False
            if np.any(local_correction > self.config.fcpg_max_correction_px):
                blocked.append(f"{view}_fcpg_outside_training_envelope")
                view_safe = False

            contexts[view] = bridged
            correction_seed[view] = seed
            corrected[view] = points
            valid[view] = corrected_valid
            diagnostics[view] = row_diagnostics
            safe_to_commit[view] = view_safe
            pending[view] = _AnchorPosterior(
                state_latest=current_state.copy(),
                frame_latest=prepared_frames[view][-1].copy(),
                nominal_latest_px=nominal[view][-1].copy(),
                seed_history_px=seed.copy(),
                corrected_latest_px=points[-1].copy(),
            )

        if commit and self.config.recursive_grounding:
            for view, posterior in pending.items():
                if safe_to_commit[view]:
                    self._anchor_posteriors[view] = posterior

        return GroundingObservation(
            anchors_xyz=anchors_xyz,
            nominal=nominal,
            correction_seed=correction_seed,
            corrected=corrected,
            valid=valid,
            contexts=contexts,
            diagnostics=diagnostics,
            blocked_reasons=tuple(sorted(set(blocked))),
        )

    def _past_action_chunks(
        self, current_state: np.ndarray, past_raw_actions: np.ndarray | None
    ) -> np.ndarray:
        hold = np.repeat(
            np.asarray(current_state, dtype=np.float32)[None],
            self.frameskip,
            axis=0,
        )
        if past_raw_actions is None:
            raw = np.stack((hold, hold))
        else:
            values = np.asarray(past_raw_actions, dtype=np.float32)
            if values.ndim != 3 or values.shape[1:] != (self.frameskip, 6):
                raise ValueError("past_raw_actions must have shape [N,5,6]")
            if len(values) > 2 or not np.all(np.isfinite(values)):
                raise ValueError("at most two finite past action chunks are allowed")
            padding = [hold] * (2 - len(values))
            raw = np.stack([*padding, *list(values)])
        normalized = (
            raw - self.action_center[None, None]
        ) / self.action_scale[None, None]
        return normalized.reshape(2, self.frameskip * 6).astype(np.float32)

    def _prepare_causal_memory(
        self, memory: CausalPlanningMemory
    ) -> dict[str, np.ndarray]:
        if self.memory_model is None:
            raise RuntimeError("causal memory requires a v6 memory checkpoint")
        states = np.asarray(memory.states, dtype=np.float32)
        offsets = np.asarray(memory.offsets_raw, dtype=np.int64)
        allowed_offsets = set(self.memory_offsets_raw)
        if any(int(value) not in allowed_offsets for value in offsets):
            raise ValueError("runtime memory offsets do not match the v6 contract")
        if np.any(offsets % self.frameskip):
            raise ValueError("runtime memory offsets must align to frameskip")

        normalized_states = (states - self.state_center) / self.state_scale
        anchors_xyz, nominal, valid = project_history(
            states,
            self.fits,
            self.kinematics,
            image_size=(224, 224),
        )
        transform = self.live_anchor_transforms_px.get("wrist")
        if transform is not None:
            nominal["wrist"] = apply_live_similarity_transform(
                nominal["wrist"], transform
            )
            valid["wrist"] = np.asarray(valid["wrist"], dtype=bool) & np.all(
                (nominal["wrist"] >= 0.0) & (nominal["wrist"] < 224.0),
                axis=-1,
            )
        camera_context = np.zeros((len(states), 5), dtype=np.float32)
        camera_context[:, 0] = 1.0
        _, context, corrected = encode_fcpg_view(
            self.system,
            np.asarray(memory.frames["wrist"], dtype=np.uint8),
            normalized_states,
            anchors_xyz,
            nominal["wrist"],
            valid["wrist"],
            camera_context,
            self.device,
            derive_center_from_tips=True,
        )
        normalized_actions = (
            np.asarray(memory.raw_actions, dtype=np.float32)
            - self.action_center[None, None]
        ) / self.action_scale[None, None]
        return {
            "context": context.astype(np.float32),
            "states": normalized_states.astype(np.float32),
            "actions": normalized_actions.reshape(len(states), -1).astype(np.float32),
            "ages": (offsets.astype(np.float32) / float(self.frameskip)),
            "corrected": corrected.astype(np.float32),
        }

    def _predict_candidates(
        self,
        context: np.ndarray,
        normalized_states: np.ndarray,
        past_chunks: np.ndarray,
        candidate_chunks: np.ndarray,
        causal_memory: CausalPlanningMemory | None,
        *,
        measurement_reliable: bool,
    ) -> tuple[np.ndarray, np.ndarray, str, int]:
        use_memory = should_use_memory(
            self.config.memory_policy,
            model_available=self.memory_model is not None,
            memory_available=causal_memory is not None,
            measurement_reliable=measurement_reliable,
        )
        if not use_memory:
            latent, state = _rollout_candidates(
                self.model,
                context,
                normalized_states,
                past_chunks,
                candidate_chunks,
            )
            variant = (
                "v9_local_backbone"
                if self.model_contract["dynamics_protocol"] == V9_PROTOCOL
                else "v6_backbone"
                if self.model_contract["dynamics_protocol"] == V6_PROTOCOL
                else "state_residual"
            )
            return latent, state, variant, 0

        memory = self._prepare_causal_memory(causal_memory)

        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(value, device=self.device, dtype=torch.float32)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            latent_tensor, state_tensor = rollout_multiscale_candidates(
                self.memory_model,
                tensor(context),
                tensor(normalized_states),
                tensor(past_chunks),
                tensor(candidate_chunks),
                memory_embedding=tensor(memory["context"]),
                memory_actions=tensor(memory["actions"]),
                memory_state=tensor(memory["states"]),
                memory_age=tensor(memory["ages"]),
                memory_valid=torch.ones(
                    len(memory["ages"]), dtype=torch.bool, device=self.device
                ),
            )
        return (
            latent_tensor.float().cpu().numpy(),
            state_tensor.float().cpu().numpy(),
            (
                "v9_memory_fallback"
                if self.model_contract["dynamics_protocol"] == V9_PROTOCOL
                else "v6_multiscale_memory"
            ),
            len(memory["ages"]),
        )

    def _plan_anchor_trajectory(
        self,
        prepared_frames: dict[str, np.ndarray],
        state_history: np.ndarray,
        grounding: GroundingObservation,
        *,
        past_raw_actions: np.ndarray | None,
        causal_memory: CausalPlanningMemory | None,
        measurement_reliable: bool,
    ) -> RealtimePlan:
        """Follow an implicit successful demonstration via LeWM anchor forecasts."""

        current_state = np.asarray(state_history[-1], dtype=np.float32)
        normalized_states = (
            np.asarray(state_history, dtype=np.float32) - self.state_center
        ) / self.state_scale
        contexts = grounding.contexts
        blocked = list(grounding.blocked_reasons)
        query_embedding = np.concatenate(
            (contexts["wrist"][-1], contexts["column"][-1])
        )
        bank_embeddings = np.concatenate(
            (self.primary_bank, self.secondary_bank), axis=1
        )
        continuation = self._demonstration_progress
        continuation_missing = False
        continuation_bank_index: int | None = None
        if continuation is not None:
            continuation_bank_index = self._bank_index_by_episode_frame.get(
                (continuation.episode, continuation.next_frame)
            )
            continuation_missing = continuation_bank_index is None
            if continuation_missing:
                blocked.append("demonstration_trajectory_complete")

        if continuation_bank_index is None:
            retrieval = retrieve_diverse_candidates(
                query_embedding,
                current_state,
                bank_embeddings,
                self.bank_states,
                self.bank_rows,
                self.bank_episodes,
                state_scale=self.state_scale,
                per_episode=self.config.candidates_per_episode,
                minimum_row_gap=self.config.candidate_min_gap,
                state_weight=self.config.state_retrieval_weight,
            )
        else:
            candidate_slice = np.asarray([continuation_bank_index], dtype=np.int64)
            retrieval = retrieve_diverse_candidates(
                query_embedding,
                current_state,
                bank_embeddings[candidate_slice],
                self.bank_states[candidate_slice],
                self.bank_rows[candidate_slice],
                self.bank_episodes[candidate_slice],
                state_scale=self.state_scale,
                per_episode=1,
                minimum_row_gap=0,
                state_weight=self.config.state_retrieval_weight,
            )
        bank_index = {int(row): index for index, row in enumerate(self.bank_rows)}
        source_indices = np.asarray(
            [bank_index[int(row)] for row in retrieval.rows], dtype=np.int64
        )
        prefixes = rebase_action_prefixes(
            self.bank_prefixes[source_indices],
            self.bank_states[source_indices],
            current_state,
        )
        candidate_chunks = chunk_and_normalize_actions(
            prefixes,
            frameskip=self.frameskip,
            center=self.action_center,
            scale=self.action_scale,
        )
        past_chunks = self._past_action_chunks(current_state, past_raw_actions)
        (
            _latent_prediction,
            state_prediction,
            rollout_variant,
            memory_token_count,
        ) = self._predict_candidates(
            contexts["wrist"],
            normalized_states,
            past_chunks,
            candidate_chunks,
            causal_memory,
            measurement_reliable=measurement_reliable,
        )
        predicted_states = (
            state_prediction * self.state_scale[None, None]
            + self.state_center[None, None]
        ).astype(np.float32)
        predicted_anchors = np.asarray(
            self.kinematics.forward_anchors(predicted_states), dtype=np.float64
        )
        commanded_states = prefixes[:, self.frameskip - 1 :: self.frameskip]
        rebase_delta = (
            current_state[None] - self.bank_states[source_indices]
        )
        demonstrated_future = (
            self.bank_future_states[source_indices] + rebase_delta[:, None]
        )
        demonstration_cost = np.mean(
            np.square(
                (predicted_states - demonstrated_future)
                / self.state_scale[None, None]
            ),
            axis=(1, 2),
        )
        model_command_cost = np.mean(
            np.square(
                (predicted_states - commanded_states)
                / self.state_scale[None, None]
            ),
            axis=(1, 2),
        )

        def unit_cost(values: np.ndarray) -> np.ndarray:
            vector = np.asarray(values, dtype=np.float64)
            minimum = float(np.min(vector))
            spread = float(np.max(vector) - minimum)
            if spread <= 1e-12:
                return np.zeros_like(vector)
            return (vector - minimum) / spread

        costs = (
            unit_cost(retrieval.combined_distance)
            + unit_cost(demonstration_cost)
            + 0.25 * unit_cost(model_command_cost)
        )
        input_safety = candidate_safety_checks(
            prefixes[:, 0],
            current_state,
            max_joint_jump=self.config.max_first_command_jump,
            raw_prefixes=prefixes,
            predicted_states=predicted_states,
            position_lower=SO101_POSITION_LOWER,
            position_upper=SO101_POSITION_UPPER,
            max_prefix_step_jump=self.config.max_prefix_step_jump,
            prediction_bound_margin=self.config.prediction_bound_margin,
        )
        predicted_xyz = np.asarray(
            self.kinematics.forward(predicted_states), dtype=np.float64
        )
        workspace_safe = np.all(
            (predicted_xyz >= self.workspace_lower[None, None])
            & (predicted_xyz <= self.workspace_upper[None, None]),
            axis=(1, 2),
        )
        pre_safe = input_safety.safe & workspace_safe
        if continuation_missing:
            pre_safe[:] = False
        candidate_rows: list[dict[str, Any]] = []
        for index, row in enumerate(retrieval.rows):
            reasons: list[str] = []
            if not input_safety.hard_command_safe[index]:
                reasons.append("demonstration_prefix_safety")
            if not input_safety.prediction_plausible[index]:
                reasons.append("lewm_prediction_bounds")
            if not workspace_safe[index]:
                reasons.append("lewm_prediction_workspace")
            candidate_rows.append(
                {
                    "source": (
                        "locked_demonstration_continuation"
                        if continuation_bank_index is not None
                        else "implicit_successful_demonstration"
                    ),
                    "index": index,
                    "row": int(row),
                    "episode": int(retrieval.episodes[index]),
                    "frame": int(self.primary.data.frame_indices[int(row)]),
                    "cost": float(costs[index]),
                    "retrieval_cost": float(retrieval.combined_distance[index]),
                    "demonstration_future_cost": float(demonstration_cost[index]),
                    "model_command_cost": float(model_command_cost[index]),
                    "safe": False,
                    "reasons": reasons,
                    "anchor_ik_attempted": False,
                    "continuation_required": continuation is not None,
                    "continuation_match": bool(
                        continuation is not None
                        and int(retrieval.episodes[index]) == continuation.episode
                        and int(self.primary.data.frame_indices[int(row)])
                        == continuation.next_frame
                    ),
                }
            )

        selected_index: int | None = None
        selected_prefix: np.ndarray | None = None
        selected_prediction: np.ndarray | None = None
        selected_commanded: np.ndarray | None = None
        selected_ik_error: float | None = None
        for index in np.argsort(costs, kind="stable"):
            index = int(index)
            if not pre_safe[index]:
                continue
            trajectory = track_anchor_trajectory(
                self.kinematics,
                current_state,
                predicted_anchors[index],
                frameskip=self.frameskip,
                tolerance_m=3e-3,
                max_step_normalized=self.config.max_prefix_step_jump,
            )
            candidate_rows[index]["anchor_ik_attempted"] = True
            candidate_rows[index]["anchor_ik_max_error_m"] = float(
                np.max(trajectory.per_horizon_max_error_m)
            )
            if not trajectory.converged:
                candidate_rows[index]["reasons"].append("anchor_ik_not_converged")
                continue
            command_safety = candidate_safety_checks(
                trajectory.raw_prefix[None, 0],
                current_state,
                max_joint_jump=self.config.max_first_command_jump,
                raw_prefixes=trajectory.raw_prefix[None],
                predicted_states=trajectory.horizon_states[None],
                position_lower=SO101_POSITION_LOWER,
                position_upper=SO101_POSITION_UPPER,
                max_prefix_step_jump=self.config.max_prefix_step_jump,
                prediction_bound_margin=0.0,
            )
            chunk_check = validate_command_chunk(
                trajectory.raw_prefix[: self.config.command_chunk],
                current_state,
                expected_commands=self.config.command_chunk,
                max_first_jump=self.config.max_first_command_jump,
                max_step_jump=self.config.max_prefix_step_jump,
            )
            command_xyz = np.asarray(
                self.kinematics.forward(trajectory.horizon_states), dtype=np.float64
            )
            command_workspace_safe = bool(
                np.all(command_xyz >= self.workspace_lower[None])
                and np.all(command_xyz <= self.workspace_upper[None])
            )
            if not bool(command_safety.safe[0]):
                candidate_rows[index]["reasons"].append("anchor_ik_command_safety")
                continue
            if not chunk_check.safe:
                candidate_rows[index]["reasons"].extend(chunk_check.reasons)
                continue
            if not command_workspace_safe:
                candidate_rows[index]["reasons"].append("anchor_ik_workspace")
                continue
            candidate_rows[index]["safe"] = True
            selected_index = index
            selected_prefix = trajectory.raw_prefix.copy()
            selected_prediction = predicted_states[index].copy()
            selected_commanded = trajectory.horizon_states.copy()
            selected_ik_error = float(np.max(trajectory.per_horizon_max_error_m))
            break

        if selected_index is None:
            blocked.append("no_safe_anchor_trajectory_candidate")

        selected_chunk = (
            None
            if selected_prefix is None
            else selected_prefix[: self.config.command_chunk].copy()
        )
        prediction_anchor_px: np.ndarray | None = None
        command_anchor_px: np.ndarray | None = None
        if selected_prediction is not None and selected_commanded is not None:
            correction = (
                grounding.corrected["column"][-1]
                - grounding.nominal["column"][-1]
            )
            transform = self.live_anchor_transforms_px.get("column")
            prediction_anchor_px = project_future_anchor_pixels(
                selected_prediction[None],
                self.fits["column"],
                self.kinematics,
                current_correction_px=correction,
                live_transform_px=transform,
            )[0]
            command_anchor_px = project_future_anchor_pixels(
                selected_commanded[None],
                self.fits["column"],
                self.kinematics,
                current_correction_px=correction,
                live_transform_px=transform,
            )[0]

        round_id = f"r-{time.time_ns()}-{uuid.uuid4().hex[:8]}"
        guard_point = np.asarray(
            grounding.corrected["column"][-1, 0], dtype=np.float32
        )
        guard = PlanningRoundGuard(
            round_id=round_id,
            created_monotonic_s=time.monotonic(),
            state=current_state.copy(),
            target_px=guard_point.copy(),
            target_kind="grounded_trajectory",
        )
        model_contract = {
            **self.model_contract,
            "control_contract": (
                "FCPG current three anchors -> action-conditioned LeWM three-anchor "
                "forecast -> three-anchor IK -> execute one chunk -> reobserve"
            ),
            "task_progress_source": (
                "initial observation alignment, then committed same-episode "
                "five-frame demonstration continuation"
            ),
            "demonstration_progress": (
                None
                if continuation is None
                else {
                    "episode": continuation.episode,
                    "required_source_frame": continuation.next_frame,
                    "completed_chunks": continuation.completed_chunks,
                    "continuation_available": not continuation_missing,
                }
            ),
            "handwritten_grasp_release": False,
            "rollout_variant": rollout_variant,
            "active_memory_tokens": int(memory_token_count),
            "measurement_reliable": bool(measurement_reliable),
            "selected_anchor_ik_max_error_m": selected_ik_error,
        }
        empty_target = {"visible": False, "reason": "not_used_for_control"}
        return RealtimePlan(
            guard=guard,
            phase=ManipulationPhase.TRAJECTORY,
            phase_status={
                "phase": ManipulationPhase.TRAJECTORY.value,
                "pending_phase": None,
                "pending_confirmations": 0,
                "required_confirmations": 0,
            },
            current_state=current_state.copy(),
            frames={view: prepared_frames[view].copy() for view in VIEWS},
            nominal=grounding.nominal,
            correction_seed=grounding.correction_seed,
            corrected=grounding.corrected,
            valid=grounding.valid,
            grounding_diagnostics=grounding.diagnostics,
            target_descriptors={
                "pickup_column": dict(empty_target),
                "pickup_wrist": dict(empty_target),
                "drop_zone_column": dict(empty_target),
            },
            target_profile=self.target_tracker.status(),
            target={
                "kind": "implicit_demonstration_trajectory",
                "visible": False,
                "centroid_px": guard_point,
                "base_xyz": None,
                "gripper_distance_m": None,
            },
            selected_index=selected_index,
            selected_chunk=selected_chunk,
            selected_prefix=selected_prefix,
            predicted_states=selected_prediction,
            commanded_states=selected_commanded,
            predicted_gripper_px=(
                None if prediction_anchor_px is None else prediction_anchor_px[:, 0]
            ),
            commanded_gripper_px=(
                None if command_anchor_px is None else command_anchor_px[:, 0]
            ),
            predicted_anchor_px=prediction_anchor_px,
            commanded_anchor_px=command_anchor_px,
            candidate_rows=candidate_rows,
            executable=selected_index is not None and not blocked,
            blocked_reasons=tuple(sorted(set(blocked))),
            model_contract=model_contract,
        )

    def detect_live_target(
        self,
        column_frame: np.ndarray,
        phase: ManipulationPhase,
        *,
        update_track: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        role = target_role_for_phase(phase)
        if role is None:
            raise RuntimeError(f"phase {phase.value} has no executable target")
        descriptor = self.target_tracker.track(
            role,
            "column",
            column_frame,
            update=update_track,
        )
        if not descriptor.get("visible", False):
            reason = descriptor.get("reason", "not reliably visible")
            raise RuntimeError(
                f"current {role} is not reliably visible: {reason}"
            )
        return role, descriptor

    @torch.inference_mode()
    def plan(
        self,
        frames: dict[str, np.ndarray],
        states: np.ndarray,
        *,
        past_raw_actions: np.ndarray | None = None,
        causal_memory: CausalPlanningMemory | None = None,
    ) -> RealtimePlan:
        state_history = np.asarray(states, dtype=np.float32)
        if state_history.shape != (self.history, 6):
            raise ValueError("states must have shape [3,6]")
        prepared_frames = {
            view: np.asarray(frames[view], dtype=np.uint8) for view in VIEWS
        }
        current_state = state_history[-1]
        normalized_states = (state_history - self.state_center) / self.state_scale
        grounding = self.observe_grounding(prepared_frames, state_history)
        nominal = grounding.nominal
        correction_seed = grounding.correction_seed
        corrected = grounding.corrected
        valid = grounding.valid
        contexts = grounding.contexts
        blocked = list(grounding.blocked_reasons)
        measurement_reliable = not grounding.blocked_reasons and all(
            bool(np.all(valid[view][-1])) for view in VIEWS
        )
        if self.config.control_mode == "anchor_trajectory":
            return self._plan_anchor_trajectory(
                prepared_frames,
                state_history,
                grounding,
                past_raw_actions=past_raw_actions,
                causal_memory=causal_memory,
                measurement_reliable=measurement_reliable,
            )

        column_pickup = self.target_tracker.track(
            "pickup",
            "column",
            prepared_frames["column"][-1],
            gripper_center_px=corrected["column"][-1, 0],
        )
        wrist_pickup = self.target_tracker.track(
            "pickup",
            "wrist",
            prepared_frames["wrist"][-1],
            gripper_center_px=corrected["wrist"][-1, 0],
        )
        drop_zone = self.target_tracker.track(
            "drop_zone",
            "column",
            prepared_frames["column"][-1],
            gripper_center_px=corrected["column"][-1, 0],
        )

        gripper_column = corrected["column"][-1, 0]
        current_gripper_xyz = np.asarray(
            self.kinematics.forward(current_state), dtype=np.float32
        )
        pickup_distance = (
            float(
                np.linalg.norm(
                    gripper_column
                    - np.asarray(column_pickup["centroid_px"], dtype=np.float32)
                )
            )
            if column_pickup.get("visible", False)
            else None
        )
        column_transform = self.live_anchor_transforms_px.get("column")
        if self.config.require_live_column_calibration and column_transform is None:
            blocked.append("column_live_calibration_required")
        pickup_raw_backprojection_xyz: np.ndarray | None = None
        drop_raw_backprojection_xyz: np.ndarray | None = None
        pickup_relative_target_xyz: np.ndarray | None = None
        drop_relative_target_xyz: np.ndarray | None = None
        pickup_target_xyz: np.ndarray | None = None
        drop_target_xyz: np.ndarray | None = None
        pickup_planar_error_m: float | None = None
        drop_planar_error_m: float | None = None
        if column_pickup.get("visible", False):
            try:
                pickup_raw_backprojection_xyz = column_live_pixel_to_base_xyz(
                    self.fits["column"],
                    np.asarray(column_pickup["centroid_px"], dtype=np.float32),
                    z_base=self.contact_height_m,
                    live_transform_px=column_transform,
                )
                (
                    pickup_target_xyz,
                    pickup_relative_target_xyz,
                    pickup_planar_error_m,
                ) = bounded_column_servo_target_xyz(
                    self.fits["column"],
                    gripper_column,
                    np.asarray(column_pickup["centroid_px"], dtype=np.float32),
                    current_gripper_xyz,
                    target_z_base=self.contact_height_m,
                    max_planar_step_m=self.config.max_column_servo_step_m,
                    descent_radius_m=self.config.column_descent_radius_m,
                    standoff_z_base=self.transport_height_m,
                    live_transform_px=column_transform,
                    max_z_step_m=self.config.max_column_z_step_m,
                )
            except ValueError:
                blocked.append("pickup_backprojection_invalid")
        if drop_zone.get("visible", False):
            try:
                drop_raw_backprojection_xyz = column_live_pixel_to_base_xyz(
                    self.fits["column"],
                    np.asarray(drop_zone["centroid_px"], dtype=np.float32),
                    z_base=self.release_height_m,
                    live_transform_px=column_transform,
                )
                (
                    drop_target_xyz,
                    drop_relative_target_xyz,
                    drop_planar_error_m,
                ) = bounded_column_servo_target_xyz(
                    self.fits["column"],
                    gripper_column,
                    np.asarray(drop_zone["centroid_px"], dtype=np.float32),
                    current_gripper_xyz,
                    target_z_base=self.release_height_m,
                    max_planar_step_m=self.config.max_column_servo_step_m,
                    descent_radius_m=self.config.column_descent_radius_m,
                    standoff_z_base=None,
                    live_transform_px=column_transform,
                    max_z_step_m=self.config.max_column_z_step_m,
                )
            except ValueError:
                blocked.append("drop_zone_backprojection_invalid")
        pickup_distance_m = (
            float(np.linalg.norm(current_gripper_xyz - pickup_relative_target_xyz))
            if pickup_relative_target_xyz is not None
            else None
        )
        drop_distance_m = (
            float(np.linalg.norm(current_gripper_xyz - drop_relative_target_xyz))
            if drop_relative_target_xyz is not None
            else None
        )
        drop_distance = (
            float(
                np.linalg.norm(
                    gripper_column
                    - np.asarray(drop_zone["centroid_px"], dtype=np.float32)
                )
            )
            if drop_zone.get("visible", False)
            else None
        )
        wrist_distance = (
            float(
                np.linalg.norm(
                    np.asarray(wrist_pickup["relative_centroid_px"], dtype=np.float32)
                )
            )
            if wrist_pickup.get("visible", False)
            else None
        )
        pickup_inside_drop = False
        if column_pickup.get("visible", False) and drop_zone.get("visible", False):
            x0, y0, x1, y1 = map(float, drop_zone["bbox_xyxy"])
            x, y = map(float, column_pickup["centroid_px"])
            pickup_inside_drop = x0 <= x <= x1 and y0 <= y <= y1
        wrist_required = self.target_tracker.configured("pickup", "wrist")
        attachment_distance = (
            wrist_distance if wrist_distance is not None else pickup_distance
        )
        pickup_visible = bool(column_pickup.get("visible", False)) and (
            bool(wrist_pickup.get("visible", False)) or not wrist_required
        )
        phase = self.phase_tracker.update(
            PhaseEvidence(
                gripper_position=float(current_state[-1]),
                gripper_to_pickup_px=pickup_distance,
                gripper_to_drop_px=drop_distance,
                gripper_to_pickup_m=pickup_distance_m,
                gripper_to_drop_m=drop_distance_m,
                wrist_pickup_distance_px=attachment_distance,
                pickup_visible=pickup_visible,
                drop_zone_visible=bool(drop_zone.get("visible", False)),
                pickup_inside_drop_zone=pickup_inside_drop,
            )
        )
        if phase != ManipulationPhase.TRANSPORT:
            self._transport_squeeze_chunks = 0
        if phase in {ManipulationPhase.TRANSPORT, ManipulationPhase.DONE}:
            # Success clears the grasp-retry budget; APPROACH does not, so
            # regression retries stay counted across re-aim cycles.
            self._grasp_empty_close_events = 0
            self._grasp_recovery_engaged = False
        if (
            phase == ManipulationPhase.GRASP
            and float(current_state[5]) >= self.thresholds.gripper_open
            and pickup_planar_error_m is not None
            and pickup_planar_error_m > self.thresholds.pickup_near_m
        ):
            # Open jaws but no longer aligned over the pickup (the object
            # moved, or a reopen retry needs re-aiming): closing here would
            # hit air. Go back to APPROACH and re-servo instead of retrying
            # the close at a frozen, misaligned pose.
            self.phase_tracker.regress_to_approach()
            phase = ManipulationPhase.APPROACH
        if phase in {ManipulationPhase.DONE, ManipulationPhase.FAULT}:
            blocked.append(f"phase_{phase.value}")
        if (
            phase == ManipulationPhase.APPROACH
            and float(current_state[5]) <= self.thresholds.gripper_closed
        ):
            # Approaching with closed jaws cannot end in a grasp, and the
            # gripper calibration deliberately parks the jaws closed. Fail
            # closed with an actionable reason instead of executing a
            # pointless approach (return home restores the task-ready
            # aperture).
            blocked.append("gripper_not_open_for_approach")

        target_kind = target_role_for_phase(phase)
        target_descriptor: dict[str, Any] | None = None
        if target_kind == "pickup":
            target_descriptor = column_pickup
            target_xyz = pickup_target_xyz
            relative_target_xyz = pickup_relative_target_xyz
            raw_backprojection_xyz = pickup_raw_backprojection_xyz
            target_planar_error_m = pickup_planar_error_m
        elif target_kind == "drop_zone":
            target_descriptor = drop_zone
            target_xyz = drop_target_xyz
            relative_target_xyz = drop_relative_target_xyz
            raw_backprojection_xyz = drop_raw_backprojection_xyz
            target_planar_error_m = drop_planar_error_m
        else:
            target_xyz = None
            relative_target_xyz = None
            raw_backprojection_xyz = None
            target_planar_error_m = None
        if target_descriptor is None or not target_descriptor.get("visible", False):
            blocked.append(f"{target_kind or 'target'}_not_visible")
            target_px = np.asarray([np.nan, np.nan], dtype=np.float32)
        else:
            target_px = np.asarray(target_descriptor["centroid_px"], dtype=np.float32)
        if target_kind is not None and target_xyz is None:
            blocked.append(f"{target_kind}_target_xyz_unavailable")

        candidate_reports: list[dict[str, Any]] = []
        selected_index: int | None = None
        selected_chunk: np.ndarray | None = None
        selected_prefix: np.ndarray | None = None
        selected_prediction: np.ndarray | None = None
        selected_commanded: np.ndarray | None = None
        selected_prediction_px: np.ndarray | None = None
        selected_command_px: np.ndarray | None = None
        selected_prediction_anchor_px: np.ndarray | None = None
        selected_command_anchor_px: np.ndarray | None = None
        rollout_variant = "not_executed"
        memory_token_count = 0

        if (
            np.all(np.isfinite(target_px))
            and target_xyz is not None
            and np.all(np.isfinite(target_xyz))
            and phase not in {
                ManipulationPhase.DONE,
                ManipulationPhase.FAULT,
            }
        ):
            if wrist_pickup.get("visible", False):
                bank_object_distance = demonstration_pickup_phase_distances(
                    wrist_pickup,
                    np.asarray(self.primary.data.frames[self.bank_rows]),
                    gripper_centers_px=self.bank_wrist_gripper_px,
                )
                extra_weight = self.config.object_phase_weight
            else:
                bank_object_distance = np.zeros(len(self.bank_rows), dtype=np.float32)
                extra_weight = 0.0
            retrieval = retrieve_diverse_candidates(
                np.concatenate((contexts["wrist"][-1], contexts["column"][-1])),
                current_state,
                np.concatenate((self.primary_bank, self.secondary_bank), axis=1),
                self.bank_states,
                self.bank_rows,
                self.bank_episodes,
                state_scale=self.state_scale,
                per_episode=self.config.candidates_per_episode,
                minimum_row_gap=self.config.candidate_min_gap,
                state_weight=self.config.state_retrieval_weight,
                extra_distance=bank_object_distance,
                extra_weight=extra_weight,
            )
            bank_index = {int(row): index for index, row in enumerate(self.bank_rows)}
            source_indices = np.asarray(
                [bank_index[int(row)] for row in retrieval.rows], dtype=np.int64
            )
            retrieval_prefixes = rebase_action_prefixes(
                self.bank_prefixes[source_indices],
                self.bank_states[source_indices],
                current_state,
            )
            if phase in {
                ManipulationPhase.APPROACH,
                ManipulationPhase.TRANSPORT,
            }:
                # Demonstration prefixes carry the demo's gripper column; a
                # demo may open and fully re-close inside one prefix, which
                # the endpoint-delta phase mask cannot see (this is how the
                # arm arrived at the object with closed jaws). Neither phase's
                # job includes moving the gripper — hold the live aperture.
                retrieval_prefixes = retrieval_prefixes.copy()
                retrieval_prefixes[:, :, 5] = current_state[5]
            retrieval_cost = np.asarray(
                retrieval.combined_distance, dtype=np.float32
            )
            selected_phase_distance = (
                bank_object_distance[source_indices]
                if phase in {ManipulationPhase.APPROACH, ManipulationPhase.GRASP}
                else np.zeros(len(retrieval_prefixes), dtype=np.float32)
            )
            candidate_metadata: list[dict[str, Any]] = [
                {
                    "source": "retrieved_demonstration",
                    "row": int(row),
                    "episode": int(retrieval.episodes[index]),
                    "frame": int(self.primary.data.frame_indices[int(row)]),
                }
                for index, row in enumerate(retrieval.rows)
            ]
            prefixes = retrieval_prefixes
            candidate_ik_safe = np.ones(len(prefixes), dtype=bool)
            role_prefixes: np.ndarray | None = None
            role_metadata: list[dict[str, Any]] = []
            role_ik_safe: np.ndarray | None = None
            if phase == ManipulationPhase.APPROACH:
                servo_prefixes, servo_metadata = generate_fcpg_goal_prefixes(
                    current_state,
                    gripper_column,
                    target_px,
                    self.fits["column"],
                    self.kinematics,
                    contact_height_m=self.contact_height_m,
                    position_lower=SO101_POSITION_LOWER,
                    position_upper=SO101_POSITION_UPPER,
                    horizon_raw=self.horizon_raw,
                    near_target_px=self.thresholds.pickup_near_px,
                    variants=DEFAULT_APPROACH_VARIANTS,
                    source="pickup",
                    transit_height_m=self.transport_height_m,
                    live_transform_px=column_transform,
                )
                ik_candidates = generate_ik_candidate_prefixes(
                    self.kinematics,
                    current_state,
                    target_xyz,
                    frameskip=self.frameskip,
                    horizon=self.horizon,
                    arrival_raw_steps=self.config.ik_arrival_raw_steps,
                    # 0.7 leaves headroom for allowed live state drift below
                    # the execution-time first-jump guard.
                    max_arrival_step=0.7 * self.config.max_prefix_step_jump,
                    path_specs=REALTIME_CONTACT_PATH_SPECS,
                )
                role_prefixes = np.concatenate(
                    (servo_prefixes, ik_candidates.raw_prefixes), axis=0
                )
                role_ik_safe = np.concatenate(
                    (
                        np.ones(len(servo_prefixes), dtype=bool),
                        ik_candidates.converged.copy(),
                    )
                )
                ik_metadata = [
                    {
                        "source": "calibrated_cartesian_ik",
                        "goal_role": "pickup",
                        "variant": name,
                        "target_surface_xyz": ik_candidates.target_surface_xyz,
                        "target_approach_xyz": ik_candidates.target_approach_xyz[index],
                        "terminal_ik_error_m": float(
                            ik_candidates.terminal_ik_error_m[index]
                        ),
                        "ik_converged": bool(ik_candidates.converged[index]),
                    }
                    for index, name in enumerate(ik_candidates.names)
                ]
                role_metadata = [*servo_metadata, *ik_metadata]
            elif phase == ManipulationPhase.GRASP:
                aperture = float(current_state[-1])
                # Recovery threshold must be EMPTY-closed (near the physical
                # stop), not the phase threshold: a successful grasp stalls
                # against the object well below gripper_closed, and opening
                # there would drop it.
                if aperture <= self.thresholds.gripper_empty_closed:
                    # The gripper arrived truly closed; no candidate can
                    # demonstrate further closing (targets would clip at the
                    # physical stop and deadlock the round). Open first so a
                    # real grasp can follow — a bounded number of times: the
                    # arm is frozen in GRASP, so a repeated air-close proves
                    # an aim error that only the operator can correct.
                    if not self._grasp_recovery_engaged:
                        self._grasp_recovery_engaged = True
                        self._grasp_empty_close_events += 1
                    if self._grasp_empty_close_events > 2:
                        blocked.append("grasp_empty_close_retry_exhausted")
                        self.phase_tracker.fault()
                    close_targets: tuple[float, ...] = (
                        self.gripper_calibration_open,
                        self.thresholds.gripper_open,
                    )
                else:
                    if aperture >= self.thresholds.gripper_open:
                        self._grasp_recovery_engaged = False
                    # Close targets must never command past the calibrated
                    # physical stop.
                    close = max(
                        min(self.thresholds.gripper_closed, aperture - 2.0),
                        self.gripper_calibration_closed,
                    )
                    close_targets = (
                        close,
                        max(close - 4.0, self.gripper_calibration_closed),
                        max(
                            float(SO101_POSITION_LOWER[-1] + 2.0),
                            self.gripper_calibration_closed,
                        ),
                    )
                role_prefixes, role_metadata = generate_gripper_phase_prefixes(
                    current_state,
                    close_targets,
                    position_lower=SO101_POSITION_LOWER,
                    position_upper=SO101_POSITION_UPPER,
                    horizon_raw=self.horizon_raw,
                    source="grasp",
                )
                role_ik_safe = np.ones(len(role_prefixes), dtype=bool)
            elif phase == ManipulationPhase.TRANSPORT:
                servo_prefixes, servo_metadata = generate_fcpg_goal_prefixes(
                    current_state,
                    gripper_column,
                    target_px,
                    self.fits["column"],
                    self.kinematics,
                    # Carry at the release height: the planner has no obstacle
                    # model, so the carry must clear container walls by
                    # construction and drop the object in from above.
                    contact_height_m=self.release_height_m,
                    position_lower=SO101_POSITION_LOWER,
                    position_upper=SO101_POSITION_UPPER,
                    horizon_raw=self.horizon_raw,
                    near_target_px=self.thresholds.drop_near_px,
                    variants=DEFAULT_TRANSPORT_VARIANTS,
                    source="drop_zone",
                    transit_height_m=self.release_height_m,
                    live_transform_px=column_transform,
                )
                ik_candidates = generate_ik_candidate_prefixes(
                    self.kinematics,
                    current_state,
                    target_xyz,
                    frameskip=self.frameskip,
                    horizon=self.horizon,
                    arrival_raw_steps=self.config.ik_arrival_raw_steps,
                    # 0.7 leaves headroom for allowed live state drift below
                    # the execution-time first-jump guard.
                    max_arrival_step=0.7 * self.config.max_prefix_step_jump,
                    path_specs=REALTIME_CONTACT_PATH_SPECS,
                )
                role_prefixes = np.concatenate(
                    (servo_prefixes, ik_candidates.raw_prefixes), axis=0
                )
                role_ik_safe = np.concatenate(
                    (
                        np.ones(len(servo_prefixes), dtype=bool),
                        ik_candidates.converged.copy(),
                    )
                )
                ik_metadata = [
                    {
                        "source": "calibrated_cartesian_ik",
                        "goal_role": "drop_zone",
                        "variant": name,
                        "target_surface_xyz": ik_candidates.target_surface_xyz,
                        "target_approach_xyz": ik_candidates.target_approach_xyz[index],
                        "terminal_ik_error_m": float(
                            ik_candidates.terminal_ik_error_m[index]
                        ),
                        "ik_converged": bool(ik_candidates.converged[index]),
                    }
                    for index, name in enumerate(ik_candidates.names)
                ]
                role_metadata = [*servo_metadata, *ik_metadata]
            elif phase == ManipulationPhase.RELEASE:
                opened = max(
                    self.thresholds.gripper_open,
                    float(current_state[-1]) + 2.0,
                )
                role_prefixes, role_metadata = generate_gripper_phase_prefixes(
                    current_state,
                    (
                        opened,
                        opened + 4.0,
                        float(SO101_POSITION_UPPER[-1] - 2.0),
                    ),
                    position_lower=SO101_POSITION_LOWER,
                    position_upper=SO101_POSITION_UPPER,
                    horizon_raw=self.horizon_raw,
                    source="release",
                )
                role_ik_safe = np.ones(len(role_prefixes), dtype=bool)

            if role_prefixes is not None:
                if role_ik_safe is None or role_ik_safe.shape != (len(role_prefixes),):
                    raise RuntimeError("role candidate IK safety mask is invalid")
                prefixes = np.concatenate((role_prefixes, prefixes), axis=0)
                candidate_ik_safe = np.concatenate(
                    (role_ik_safe, candidate_ik_safe)
                )
                retrieval_cost = np.concatenate(
                    (
                        np.zeros(len(role_prefixes), dtype=np.float32),
                        retrieval_cost,
                    )
                )
                selected_phase_distance = np.concatenate(
                    (
                        np.zeros(len(role_prefixes), dtype=np.float32),
                        selected_phase_distance,
                    )
                )
                candidate_metadata = [*role_metadata, *candidate_metadata]
            if phase == ManipulationPhase.TRANSPORT:
                # Maintain squeeze while carrying: position-hold alone loses
                # grip as the object shifts under motion (observed live: a
                # stalled grasp slipped out during the lift). Every carry
                # candidate ramps the gripper gently below the current
                # aperture — bounded per step, floored above the calibrated
                # stop, and capped so the transport phase mask (|delta|<=5)
                # still holds. Harmless when the jaws are empty.
                # The floor must clear the plan-time joint-limit guard band
                # (limit_guard 3.0 above the hard lower bound): a ramp that
                # enters the band synthesizes candidates validate_command_
                # chunk must reject, and a sole rejected candidate deadlocks
                # the phase (observed live at aperture 2.37).
                squeeze_floor = max(
                    self.gripper_calibration_closed + 1.0,
                    float(SO101_POSITION_LOWER[-1]) + 3.0 + 0.1,
                )
                aperture = float(current_state[5])
                if aperture > squeeze_floor + 0.3:
                    steps = prefixes.shape[1]
                    ramp = np.maximum(
                        aperture - 0.3 * np.arange(1, steps + 1),
                        max(squeeze_floor, aperture - 4.5),
                    ).astype(prefixes.dtype)
                    if self._transport_squeeze_chunks < 2:
                        # Load the grip BEFORE moving: lifting with an
                        # unloaded position-hold is how the object slipped
                        # out. The first transport chunks squeeze in place
                        # with the arm frozen; motion begins once the grip
                        # carries sustained position error.
                        hold = np.repeat(
                            current_state[None, :].astype(prefixes.dtype),
                            steps,
                            axis=0,
                        )[None]
                        hold[0, :, 5] = ramp
                        prefixes = hold
                        candidate_ik_safe = np.ones(1, dtype=bool)
                        retrieval_cost = np.zeros(1, dtype=np.float32)
                        selected_phase_distance = np.zeros(
                            1, dtype=np.float32
                        )
                        candidate_metadata = [
                            {
                                "source": "transport_squeeze",
                                "variant": "hold_and_load",
                            }
                        ]
                    else:
                        prefixes = prefixes.copy()
                        prefixes[:, :, 5] = ramp[None, :]
            candidate_chunks = chunk_and_normalize_actions(
                prefixes,
                frameskip=self.frameskip,
                center=self.action_center,
                scale=self.action_scale,
            )
            past_chunks = self._past_action_chunks(current_state, past_raw_actions)
            (
                primary_prediction,
                state_prediction,
                rollout_variant,
                memory_token_count,
            ) = self._predict_candidates(
                contexts["wrist"],
                normalized_states,
                past_chunks,
                candidate_chunks,
                causal_memory,
                measurement_reliable=measurement_reliable,
            )
            del primary_prediction
            predicted_states = (
                state_prediction * self.state_scale[None, None]
                + self.state_center[None, None]
            )
            predicted_anchor_pixels = project_future_anchor_pixels(
                predicted_states,
                self.fits["column"],
                self.kinematics,
                current_correction_px=(
                    corrected["column"][-1] - nominal["column"][-1]
                ),
                live_transform_px=column_transform,
            )
            predicted_pixels = predicted_anchor_pixels[:, :, 0]
            commanded_states = prefixes[:, self.frameskip - 1 :: self.frameskip]
            commanded_anchor_pixels = project_future_anchor_pixels(
                commanded_states,
                self.fits["column"],
                self.kinematics,
                current_correction_px=(
                    corrected["column"][-1] - nominal["column"][-1]
                ),
                live_transform_px=column_transform,
            )
            commanded_pixels = commanded_anchor_pixels[:, :, 0]
            pixel_goal_cost = current_pixel_goal_cost(
                predicted_pixels,
                target_px,
                final_steps=self.config.goal_final_steps,
                image_size=(224, 224),
            )
            predicted_xyz = self.kinematics.forward(predicted_states)
            commanded_xyz = self.kinematics.forward(commanded_states)
            initial_target_error_m = float(
                np.linalg.norm(current_gripper_xyz - target_xyz)
            )
            first_chunk_target_error_m = np.linalg.norm(
                commanded_xyz[:, 0] - target_xyz[None], axis=-1
            )
            terminal_target_error_m = np.linalg.norm(
                commanded_xyz[:, -1] - target_xyz[None], axis=-1
            )
            for index, metadata in enumerate(candidate_metadata):
                if metadata.get("source") != "calibrated_cartesian_ik":
                    continue
                first_progress_m = float(
                    initial_target_error_m - first_chunk_target_error_m[index]
                )
                terminal_progress_m = float(
                    initial_target_error_m - terminal_target_error_m[index]
                )
                receding_horizon_qualified = bool(
                    first_progress_m >= 1.0e-3 and terminal_progress_m > 0.0
                )
                metadata.update(
                    {
                        "ik_receding_horizon_qualified": receding_horizon_qualified,
                        "first_chunk_target_error_m": float(
                            first_chunk_target_error_m[index]
                        ),
                        "terminal_command_target_error_m": float(
                            terminal_target_error_m[index]
                        ),
                        "first_chunk_target_progress_m": first_progress_m,
                        "terminal_target_progress_m": terminal_progress_m,
                    }
                )
            cartesian_goal_cost = cartesian_terminal_goal_cost(
                predicted_xyz,
                target_xyz,
                final_steps=self.config.goal_final_steps,
                scale_m=self.config.cartesian_goal_scale_m,
            )
            model_command_cost = np.mean(
                np.square(
                    (predicted_states - commanded_states)
                    / self.state_scale[None, None]
                ),
                axis=(1, 2),
            )
            costs = (
                cartesian_goal_cost
                + self.config.pixel_goal_weight * pixel_goal_cost
                + self.config.retrieval_cost_weight * retrieval_cost
                + 0.05 * model_command_cost
            )
            safety = candidate_safety_checks(
                prefixes[:, 0],
                current_state,
                max_joint_jump=self.config.max_first_command_jump,
                max_prefix_step_jump=self.config.max_prefix_step_jump,
                raw_prefixes=prefixes,
                predicted_states=predicted_states,
                position_lower=SO101_POSITION_LOWER,
                position_upper=SO101_POSITION_UPPER,
                prediction_bound_margin=self.config.prediction_bound_margin,
            )
            grasp_recovery_aperture = (
                self.gripper_calibration_closed + 2.0
                if phase == ManipulationPhase.GRASP
                else None
            )
            phase_safe = candidate_phase_mask(
                prefixes,
                phase,
                self.thresholds,
                current_gripper=float(current_state[5]),
                recovery_below=grasp_recovery_aperture,
            )
            phase_prediction_safe = phase_prediction_bounds_mask(
                predicted_states,
                phase,
                margin=self.config.prediction_bound_margin,
            )
            if phase in {ManipulationPhase.APPROACH, ManipulationPhase.GRASP}:
                phase_distance_safe = (
                    selected_phase_distance <= self.config.max_object_phase_distance
                )
            else:
                selected_phase_distance = np.zeros(len(prefixes), dtype=np.float32)
                phase_distance_safe = np.ones(len(prefixes), dtype=bool)
            prediction_lower_excess = np.max(
                SO101_POSITION_LOWER[None, None]
                - self.config.prediction_bound_margin
                - predicted_states,
                axis=(1, 2),
            )
            prediction_upper_excess = np.max(
                predicted_states
                - SO101_POSITION_UPPER[None, None]
                - self.config.prediction_bound_margin,
                axis=(1, 2),
            )
            prediction_bound_excess = np.maximum(
                0.0,
                np.maximum(prediction_lower_excess, prediction_upper_excess),
            )
            prediction_bound_dimension = np.argmax(
                np.maximum(
                    SO101_POSITION_LOWER[None, None]
                    - self.config.prediction_bound_margin
                    - predicted_states,
                    predicted_states
                    - SO101_POSITION_UPPER[None, None]
                    - self.config.prediction_bound_margin,
                ).max(axis=1),
                axis=1,
            )
            # The training-FK box must cover the operator-selected target: a
            # pickup at the box edge otherwise rejects every candidate that
            # descends to it — and every grasp-phase hold at it. Growth is
            # directional (only toward the target) and hard-capped.
            growth_cap = 0.06
            effective_workspace_lower = np.maximum(
                np.minimum(self.workspace_lower, target_xyz - 0.01),
                self.workspace_lower - growth_cap,
            )
            effective_workspace_upper = np.minimum(
                np.maximum(self.workspace_upper, target_xyz + 0.01),
                self.workspace_upper + growth_cap,
            )
            workspace_safe = np.all(
                (predicted_xyz >= effective_workspace_lower[None, None])
                & (predicted_xyz <= effective_workspace_upper[None, None]),
                axis=(1, 2),
            ) & np.all(
                (commanded_xyz >= effective_workspace_lower[None, None])
                & (commanded_xyz <= effective_workspace_upper[None, None]),
                axis=(1, 2),
            )
            predicted_workspace_excess = np.maximum(
                0.0,
                np.maximum(
                    effective_workspace_lower[None, None] - predicted_xyz,
                    predicted_xyz - effective_workspace_upper[None, None],
                ),
            ).max(axis=(1, 2))
            commanded_workspace_excess = np.maximum(
                0.0,
                np.maximum(
                    effective_workspace_lower[None, None] - commanded_xyz,
                    commanded_xyz - effective_workspace_upper[None, None],
                ),
            ).max(axis=(1, 2))
            model_command_xyz_gap = np.max(
                np.linalg.norm(predicted_xyz - commanded_xyz, axis=-1), axis=1
            )
            # Uniform commanded tool-travel bound, family-independent. The
            # per-round caps applied to the servo TARGET do not bound IK
            # clearance/arc offsets or rebased demonstration trajectories;
            # enforce the operator's caps on every candidate's actual
            # commanded travel. Step 0 is the executed chunk's terminal
            # displacement from the current tool position. The 3mm allowance
            # covers the IK terminal tolerance.
            tool_step_xyz = np.diff(
                np.concatenate(
                    (
                        np.broadcast_to(
                            np.asarray(
                                current_gripper_xyz, dtype=commanded_xyz.dtype
                            )[None, None],
                            (len(prefixes), 1, 3),
                        ),
                        commanded_xyz,
                    ),
                    axis=1,
                ),
                axis=1,
            )
            tool_planar_step_m = np.linalg.norm(
                tool_step_xyz[:, :, :2], axis=-1
            )
            tool_z_step_m = np.abs(tool_step_xyz[:, :, 2])
            tool_travel_safe = np.all(
                (
                    tool_planar_step_m
                    <= self.config.max_column_servo_step_m + 3.0e-3
                )
                & (tool_z_step_m <= self.config.max_column_z_step_m + 3.0e-3),
                axis=1,
            )
            chunk_safe = np.zeros(len(prefixes), dtype=bool)
            chunk_reasons: list[tuple[str, ...]] = []
            for index in range(len(prefixes)):
                check = validate_command_chunk(
                    prefixes[index, : self.config.command_chunk],
                    current_state,
                    expected_commands=self.config.command_chunk,
                    max_first_jump=self.config.max_first_command_jump,
                    max_step_jump=self.config.max_prefix_step_jump,
                )
                chunk_safe[index] = check.safe
                chunk_reasons.append(check.reasons)
            safe = (
                safety.hard_command_safe
                & candidate_ik_safe
                & phase_prediction_safe
                & phase_safe
                & phase_distance_safe
                & workspace_safe
                & tool_travel_safe
                & chunk_safe
            )
            for index, metadata in enumerate(candidate_metadata):
                reasons = list(chunk_reasons[index])
                if not safety.hard_command_safe[index]:
                    reasons.append("raw_prefix_safety")
                if not candidate_ik_safe[index]:
                    reasons.append("ik_not_converged")
                if not phase_prediction_safe[index]:
                    reasons.append("lewm_prediction_bounds")
                if not phase_safe[index]:
                    reasons.append("gripper_phase")
                if not phase_distance_safe[index]:
                    reasons.append("object_phase_distance")
                if not workspace_safe[index]:
                    reasons.append("mujoco_workspace")
                if not tool_travel_safe[index]:
                    reasons.append("tool_travel_exceeded")
                candidate_reports.append(
                    {
                        **metadata,
                        "index": index,
                        "cost": float(costs[index]),
                        "goal_cost": float(cartesian_goal_cost[index]),
                        "cartesian_goal_cost": float(cartesian_goal_cost[index]),
                        "pixel_goal_cost": float(pixel_goal_cost[index]),
                        "retrieval_cost": float(retrieval_cost[index]),
                        "model_command_cost": float(model_command_cost[index]),
                        "model_command_max_xyz_gap_m": float(
                            model_command_xyz_gap[index]
                        ),
                        "predicted_workspace_excess_m": float(
                            predicted_workspace_excess[index]
                        ),
                        "commanded_workspace_excess_m": float(
                            commanded_workspace_excess[index]
                        ),
                        "first_chunk_joint_delta": (
                            prefixes[index, self.config.command_chunk - 1]
                            - current_state
                        ),
                        "raw_prefix_max_joint_step": float(
                            np.max(
                                np.abs(
                                    np.diff(
                                        np.concatenate(
                                            (current_state[None], prefixes[index]), axis=0
                                        ),
                                        axis=0,
                                    )
                                )
                            )
                        ),
                        "prediction_bound_excess": float(
                            prediction_bound_excess[index]
                        ),
                        "prediction_bound_dimension": int(
                            prediction_bound_dimension[index]
                        ),
                        "phase_distance": float(selected_phase_distance[index]),
                        "safe": bool(safe[index]),
                        "reasons": sorted(set(reasons)),
                    }
                )
            if np.any(safe):
                selected_index = int(np.argmin(np.where(safe, costs, np.inf)))
                selected_prefix = prefixes[selected_index].copy()
                selected_chunk = selected_prefix[: self.config.command_chunk].copy()
                selected_prediction = predicted_states[selected_index].copy()
                selected_commanded = commanded_states[selected_index].copy()
                selected_prediction_px = predicted_pixels[selected_index].copy()
                selected_command_px = commanded_pixels[selected_index].copy()
                selected_prediction_anchor_px = predicted_anchor_pixels[
                    selected_index
                ].copy()
                selected_command_anchor_px = commanded_anchor_pixels[
                    selected_index
                ].copy()
            else:
                blocked.append("no_safe_phase_compatible_candidate")

        round_id = f"r-{time.time_ns()}-{uuid.uuid4().hex[:8]}"
        safe_target = (
            target_px
            if np.all(np.isfinite(target_px))
            else np.asarray([-1.0, -1.0], dtype=np.float32)
        )
        guard = PlanningRoundGuard(
            round_id=round_id,
            created_monotonic_s=time.monotonic(),
            state=current_state.copy(),
            target_px=safe_target.copy(),
            target_kind=target_kind or "none",
        )
        plan_model_contract = {
            **self.model_contract,
            "rollout_variant": rollout_variant,
            "active_memory_tokens": int(memory_token_count),
            "measurement_reliable": bool(measurement_reliable),
        }
        return RealtimePlan(
            guard=guard,
            phase=phase,
            phase_status=self.phase_tracker.status(),
            current_state=current_state.copy(),
            frames={view: prepared_frames[view].copy() for view in VIEWS},
            nominal=nominal,
            correction_seed=correction_seed,
            corrected=corrected,
            valid=valid,
            grounding_diagnostics=grounding.diagnostics,
            target_descriptors={
                "pickup_column": column_pickup,
                "pickup_wrist": wrist_pickup,
                "drop_zone_column": drop_zone,
            },
            target_profile=self.target_tracker.status(),
            target={
                "kind": target_kind,
                "visible": bool(target_descriptor and target_descriptor.get("visible")),
                "centroid_px": safe_target,
                "base_xyz": (
                    np.asarray(target_xyz, dtype=np.float32)
                    if target_xyz is not None
                    else None
                ),
                "relative_goal_xyz": (
                    np.asarray(relative_target_xyz, dtype=np.float32)
                    if relative_target_xyz is not None
                    else None
                ),
                "raw_backprojection_xyz": (
                    np.asarray(raw_backprojection_xyz, dtype=np.float32)
                    if raw_backprojection_xyz is not None
                    else None
                ),
                "planar_error_m": target_planar_error_m,
                "bounded_column_servo": bool(
                    target_planar_error_m is not None
                    and target_planar_error_m > self.config.max_column_servo_step_m
                ),
                "gripper_distance_m": (
                    float(np.linalg.norm(current_gripper_xyz - relative_target_xyz))
                    if relative_target_xyz is not None
                    else None
                ),
            },
            selected_index=selected_index,
            selected_chunk=selected_chunk,
            selected_prefix=selected_prefix,
            predicted_states=selected_prediction,
            commanded_states=selected_commanded,
            predicted_gripper_px=selected_prediction_px,
            commanded_gripper_px=selected_command_px,
            predicted_anchor_px=selected_prediction_anchor_px,
            commanded_anchor_px=selected_command_anchor_px,
            candidate_rows=candidate_reports,
            executable=selected_index is not None and not blocked,
            blocked_reasons=tuple(sorted(set(blocked))),
            model_contract=plan_model_contract,
        )


__all__ = [
    "CausalPlanningMemory",
    "GroundingObservation",
    "PreparedPlannerConfig",
    "PreparedSO101RealtimePlanner",
    "RealtimePlan",
    "cartesian_terminal_goal_cost",
    "bounded_column_servo_target_xyz",
    "column_live_pixel_to_base_xyz",
    "plan_json",
    "recursive_anchor_seed",
]
