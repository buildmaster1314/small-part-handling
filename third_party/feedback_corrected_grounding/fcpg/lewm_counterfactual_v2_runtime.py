"""Leakage-safe and auditable same-snapshot collection for Counterfactual v2.

The collector deliberately does not train or score a policy.  Each open-loop
candidate is executed once from one physical MetaWorld snapshot.  Camera
conditions are then rendered from the saved history and endpoint snapshots,
so return/progress/success labels have no camera axis.
"""

from __future__ import annotations

import copy
import hashlib
import pickle
import random
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from fcpg.lewm_counterfactual import ActionCandidates, build_action_candidates, reshape_action_chunks
from fcpg.lewm_counterfactual_runtime import (
    EVAL_GROUNDINGS,
    EXPECTED_TASKS,
    FrameRecord,
    FrozenInputs,
    TRAIN_GROUNDINGS,
    _collect_frame_record,
    _git_state,
    _history_indices,
    _jsonable,
    _object,
    _versions,
    build_lewm_history_batch,
    canonical_sha256,
    capture_branch_snapshot,
    corrected_points,
    grounded_valid_mask,
    load_frozen_inputs,
    oracle_targets,
    physical_snapshot_sha256,
    policy_action_chunk,
    restore_branch_snapshot,
    sha256_array,
    sha256_file,
    sha256_state_dict,
    snapshot_sha256,
)
from fcpg.lewm_counterfactual_v2 import (
    AuditArrays,
    DeploymentInputs,
    GROUNDING_NAMES,
    MODEL_NAMES,
    OfflineLabels,
    TASK_NAMES,
    VIEW_NAMES,
    assert_no_privileged_inputs,
    assign_physical_episode_splits,
    audit_episode_groups,
    build_branch_receipt_fields,
    canonical_row_order,
    goal_visual_projection_seed,
    physical_episode_sha256,
    sha256_array as schema_sha256_array,
    validate_array_bundle,
    validate_full_config,
    validate_index_and_provenance,
    validate_smoke_config,
)


COLLECTOR_PROTOCOL = "fcpg_lewm_counterfactual_v2_collection"
CONDITION_NAMES = ("clean", "rotation5", "translation4")
PHASE_NAMES = ("early", "mid", "late")
PHASE_STEPS = (10, 25, 35)
HISTORY_SIZE = 3
FRAME_SKIP = 5
RAW_ACTION_STEPS = 25
ACTION_DIM = 4
TRANSITIONS = 5
TRANSITION_ACTION_DIM = 20
LATENT_DIM = 192
NUM_KEYPOINTS = 3


@dataclass
class PhysicalEpisode:
    """One camera-independent expert trajectory and its exact state snapshots."""

    env: Any
    task: str
    episode: int
    env_seed: int
    camera_id: int
    base_position: np.ndarray
    base_quaternion: np.ndarray
    intrinsic: np.ndarray
    nominal_extrinsic: np.ndarray
    records_clean: list[FrameRecord]
    actions: np.ndarray
    snapshots: dict[int, Any]
    observations: dict[int, np.ndarray]
    goal_world: np.ndarray


@dataclass(frozen=True)
class PhysicalBranchOutcome:
    """Camera-independent labels and endpoint states for one fixed action chunk."""

    endpoint_snapshots: tuple[Any, ...]
    endpoint_observations: tuple[np.ndarray, ...]
    cumulative_return: np.ndarray
    progress: np.ndarray
    success: np.ndarray
    action_sha256: str


@dataclass(frozen=True)
class CollectionPayload:
    deployment_inputs: Mapping[str, np.ndarray]
    offline_labels: Mapping[str, np.ndarray]
    audit_arrays: Mapping[str, np.ndarray]
    index_rows: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any]


def rng_snapshot_sha256(snapshot: Any) -> str:
    """Bind both environment-owned and process-wide RNG state."""

    values = (snapshot.environment_rng, snapshot.process_rng)
    return hashlib.sha256(pickle.dumps(values, protocol=5)).hexdigest()


def _rgb_uint8(record: FrameRecord) -> np.ndarray:
    rgb = np.transpose(np.asarray(record.rgb, dtype=np.float32), (1, 2, 0))
    return np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _episode_seed(config: Mapping[str, Any], task: str, episode: int) -> int:
    environment = _object(config.get("environment"), "environment")
    return (
        int(environment["env_seed_base"])
        + EXPECTED_TASKS.index(task) * 1_000
        + int(episode)
    )


def _required_state_steps(phase_steps: Sequence[int]) -> tuple[int, ...]:
    result: set[int] = set(int(step) for step in phase_steps)
    for step in phase_steps:
        result.update(_history_indices(int(step), HISTORY_SIZE, FRAME_SKIP))
    return tuple(sorted(result))


def collect_physical_episode(
    task: str,
    episode: int,
    config: Mapping[str, Any],
) -> PhysicalEpisode:
    """Collect one expert trajectory without selecting phases from future success."""

    from fcpg.metaworld_utils import make_env, make_scripted_policy
    from fcpg.mujoco_camera import (
        camera_extrinsic_cv_from_mujoco,
        camera_intrinsic_from_fovy,
        get_camera_id,
    )

    environment = _object(config.get("environment"), "environment")
    evaluation = _object(config.get("evaluation"), "evaluation")
    phase_steps = tuple(int(value) for value in evaluation["phase_steps"])
    if phase_steps != PHASE_STEPS:
        raise ValueError(f"v2 phase steps must be exactly {PHASE_STEPS}")
    env_seed = _episode_seed(config, task, episode)
    _set_seed(env_seed)
    width = int(environment["width"])
    height = int(environment["height"])
    camera_name = str(environment["camera"])
    env = make_env(
        task,
        camera_name,
        width,
        height,
        task_index=int(environment["task_index_offset"]) + int(episode),
        benchmark_seed=int(environment["benchmark_seed"]),
    )
    observation, _ = env.reset(seed=env_seed)
    camera_id = get_camera_id(env.model, camera_name)
    intrinsic = camera_intrinsic_from_fovy(
        width,
        height,
        float(env.model.cam_fovy[camera_id]),
    )
    base_position = np.asarray(env.model.cam_pos[camera_id], dtype=np.float64).copy()
    base_quaternion = np.asarray(env.model.cam_quat[camera_id], dtype=np.float64).copy()
    nominal_extrinsic = camera_extrinsic_cv_from_mujoco(env.data, camera_id)
    goal_world = np.asarray(getattr(env, "_target_pos"), dtype=np.float32).copy()
    if goal_world.shape != (3,) or not np.isfinite(goal_world).all():
        raise ValueError("MetaWorld _target_pos must be one finite 3D goal label")

    required_steps = _required_state_steps(phase_steps)
    final_step = max(phase_steps) + RAW_ACTION_STEPS
    snapshots: dict[int, Any] = {}
    observations: dict[int, np.ndarray] = {}
    records: list[FrameRecord] = []
    actions: list[np.ndarray] = []
    expert = make_scripted_policy(task)
    try:
        for step in range(final_step + 1):
            record = _collect_frame_record(
                env,
                observation,
                camera_id=camera_id,
                intrinsic=intrinsic,
                nominal_extrinsic=nominal_extrinsic,
                width=width,
                height=height,
            )
            records.append(record)
            if step in required_steps:
                snapshots[step] = capture_branch_snapshot(
                    env,
                    include_process_rng=bool(
                        evaluation["snapshot_include_process_rng"]
                    ),
                )
                observations[step] = np.asarray(observation, dtype=np.float32).copy()
            if step == final_step:
                break
            action = np.asarray(expert.get_action(observation), dtype=np.float32)
            if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
                raise ValueError("scripted expert produced an invalid 4D action")
            action = np.clip(action, env.action_space.low, env.action_space.high).astype(
                np.float32
            )
            actions.append(action)
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                raise RuntimeError(
                    f"{task} episode {episode} ended before fixed v2 step {final_step}"
                )
        if set(snapshots) != set(required_steps):
            raise RuntimeError("fixed phase/history snapshots are incomplete")
        action_array = np.asarray(actions, dtype=np.float32)
        if action_array.shape != (final_step, ACTION_DIM):
            raise RuntimeError("expert action trajectory has an unexpected shape")
        return PhysicalEpisode(
            env=env,
            task=task,
            episode=int(episode),
            env_seed=env_seed,
            camera_id=camera_id,
            base_position=base_position,
            base_quaternion=base_quaternion,
            intrinsic=np.asarray(intrinsic, dtype=np.float32),
            nominal_extrinsic=np.asarray(nominal_extrinsic, dtype=np.float32),
            records_clean=records,
            actions=action_array,
            snapshots=snapshots,
            observations=observations,
            goal_world=goal_world,
        )
    except Exception:
        if hasattr(env, "close"):
            env.close()
        raise


def _condition_pose(
    episode: PhysicalEpisode,
    condition: Mapping[str, Any],
) -> None:
    from fcpg.mujoco_camera import set_fixed_camera_pose_offset

    set_fixed_camera_pose_offset(
        episode.env.model,
        episode.env.data,
        episode.camera_id,
        episode.base_position,
        episode.base_quaternion,
        np.asarray(condition["translation_world"], dtype=np.float64),
        np.asarray(condition["rotation_axis_world"], dtype=np.float64),
        float(condition["rotation_degrees"]),
    )


def _restore_render_warmstart(episode: PhysicalEpisode, snapshot: Any) -> None:
    """Undo solver roundoff from ``mj_forward`` without resetting the camera.

    Updating a fixed camera requires ``mj_forward`` so MuJoCo refreshes
    ``cam_xpos``/``cam_xmat``.  ``mj_forward`` also recomputes
    ``qacc_warmstart`` and can change it by a few ULPs even though no physical
    state advanced.  Restore only the saved warm-start component afterwards;
    the camera-derived arrays remain current while the branch state is again
    bit-exact.
    """

    import mujoco

    warmstart = np.asarray(snapshot.warmstart_state, dtype=np.float64)
    spec = mujoco.mjtState.mjSTATE_WARMSTART
    expected_size = int(mujoco.mj_stateSize(episode.env.model, spec))
    if warmstart.shape != (expected_size,):
        raise ValueError(
            "snapshot warm-start shape disagrees with the active MuJoCo model: "
            f"expected {(expected_size,)}, got {warmstart.shape}"
        )
    mujoco.mj_setState(
        episode.env.model,
        episode.env.data,
        np.array(warmstart, copy=True),
        spec,
    )


def _close_episode_env_safely(env: Any) -> None:
    """Release offscreen viewers without terminating process-global GLFW."""

    renderer = getattr(env, "mujoco_renderer", None)
    viewers = getattr(renderer, "_viewers", None)
    if isinstance(viewers, MutableMapping):
        for viewer in tuple(viewers.values()):
            context = getattr(viewer, "con", None)
            if (
                not hasattr(viewer, "make_context_current")
                or not hasattr(context, "free")
                or not hasattr(viewer, "free")
            ):
                raise RuntimeError("collector encountered a non-offscreen MuJoCo viewer")
            viewer.make_context_current()
            context.free()
            viewer.free()
        viewers.clear()
        return
    if hasattr(env, "close"):
        env.close()


def render_state_record(
    episode: PhysicalEpisode,
    step: int,
    condition: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[FrameRecord, np.ndarray, str, str]:
    """Render one camera view from an exact saved physical/RNG state."""

    from fcpg.mujoco_camera import camera_extrinsic_cv_from_mujoco

    environment = _object(config.get("environment"), "environment")
    snapshot = episode.snapshots[int(step)]
    restore_branch_snapshot(episode.env, snapshot)
    before_physical = physical_snapshot_sha256(snapshot)
    before_rng = rng_snapshot_sha256(snapshot)
    _condition_pose(episode, condition)
    record = _collect_frame_record(
        episode.env,
        episode.observations[int(step)],
        camera_id=episode.camera_id,
        intrinsic=episode.intrinsic,
        nominal_extrinsic=episode.nominal_extrinsic,
        width=int(environment["width"]),
        height=int(environment["height"]),
    )
    _restore_render_warmstart(episode, snapshot)
    true_extrinsic = camera_extrinsic_cv_from_mujoco(
        episode.env.data,
        episode.camera_id,
    )
    current = capture_branch_snapshot(episode.env, include_process_rng=True)
    current_physical = physical_snapshot_sha256(current)
    current_rng = rng_snapshot_sha256(current)
    if current_physical != before_physical:
        raise RuntimeError("camera-only rendering changed the physical state")
    if current_rng != before_rng:
        raise RuntimeError("camera-only rendering changed RNG state")
    return (
        record,
        np.asarray(true_extrinsic, dtype=np.float32),
        current_physical,
        current_rng,
    )


def execute_physical_branch(
    episode: PhysicalEpisode,
    snapshot_step: int,
    actions: np.ndarray,
    config: Mapping[str, Any],
) -> PhysicalBranchOutcome:
    """Execute one fixed action chunk once; camera views are rendered later."""

    evaluation = _object(config.get("evaluation"), "evaluation")
    value = np.asarray(actions, dtype=np.float32)
    if value.shape != (RAW_ACTION_STEPS, ACTION_DIM):
        raise ValueError("v2 candidate must have shape [25, 4]")
    restore_branch_snapshot(episode.env, episode.snapshots[int(snapshot_step)])
    endpoints: list[Any] = []
    observations: list[np.ndarray] = []
    returns: list[float] = []
    progress: list[float] = []
    successes: list[bool] = []
    cumulative = 0.0
    progress_key = str(evaluation["progress_info_key"])
    for raw_step, action in enumerate(value, start=1):
        clipped = np.clip(
            action,
            episode.env.action_space.low,
            episode.env.action_space.high,
        ).astype(np.float32)
        observation, reward, terminated, truncated, info = episode.env.step(clipped)
        cumulative += float(reward)
        if raw_step % FRAME_SKIP == 0:
            if not isinstance(info, Mapping) or progress_key not in info:
                raise KeyError(f"MetaWorld branch info is missing {progress_key!r}")
            endpoints.append(
                capture_branch_snapshot(episode.env, include_process_rng=True)
            )
            observations.append(np.asarray(observation, dtype=np.float32).copy())
            returns.append(cumulative)
            progress.append(float(info[progress_key]))
            successes.append(bool(float(info.get("success", 0.0)) > 0.0))
        if (terminated or truncated) and raw_step < RAW_ACTION_STEPS:
            raise RuntimeError("candidate branch ended before all five endpoints")
    if len(endpoints) != TRANSITIONS:
        raise RuntimeError("candidate branch did not produce five endpoints")
    return PhysicalBranchOutcome(
        endpoint_snapshots=tuple(endpoints),
        endpoint_observations=tuple(observations),
        cumulative_return=np.asarray(returns, dtype=np.float32),
        progress=np.asarray(progress, dtype=np.float32),
        success=np.asarray(successes, dtype=bool),
        action_sha256=sha256_array(value),
    )


def render_endpoint_record(
    episode: PhysicalEpisode,
    outcome: PhysicalBranchOutcome,
    endpoint_index: int,
    condition: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[FrameRecord, str, str]:
    """Render a camera view of an already-executed physical endpoint."""

    environment = _object(config.get("environment"), "environment")
    snapshot = outcome.endpoint_snapshots[int(endpoint_index)]
    restore_branch_snapshot(episode.env, snapshot)
    physical_before = physical_snapshot_sha256(snapshot)
    rng_before = rng_snapshot_sha256(snapshot)
    _condition_pose(episode, condition)
    record = _collect_frame_record(
        episode.env,
        outcome.endpoint_observations[int(endpoint_index)],
        camera_id=episode.camera_id,
        intrinsic=episode.intrinsic,
        nominal_extrinsic=episode.nominal_extrinsic,
        width=int(environment["width"]),
        height=int(environment["height"]),
    )
    _restore_render_warmstart(episode, snapshot)
    current = capture_branch_snapshot(episode.env, include_process_rng=True)
    current_physical = physical_snapshot_sha256(current)
    current_rng = rng_snapshot_sha256(current)
    if current_physical != physical_before:
        raise RuntimeError("endpoint re-render changed physical state")
    if current_rng != rng_before:
        raise RuntimeError("endpoint re-render changed RNG state")
    return record, current_physical, current_rng


def _namespace_seed(
    config: Mapping[str, Any],
    namespace: str,
    *,
    task: str,
    episode: int,
    snapshot_step: int,
) -> int:
    """Derive one camera-independent candidate RNG seed."""

    if task not in TASK_NAMES:
        raise ValueError(f"unknown v2 task: {task}")
    if not namespace or not namespace.isascii():
        raise ValueError("RNG namespace must be non-empty ASCII")
    payload = (
        f"{COLLECTOR_PROTOCOL}|{int(config['seed'])}|{namespace}|"
        f"{TASK_NAMES.index(task)}|"
        f"{int(episode)}|{int(snapshot_step)}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _history_raw_actions(
    episode: PhysicalEpisode,
    snapshot_step: int,
) -> np.ndarray:
    """Return the two physical action blocks connecting the fixed history."""

    indices = _history_indices(int(snapshot_step), HISTORY_SIZE, FRAME_SKIP)
    blocks = []
    for start, stop in zip(indices[:-1], indices[1:]):
        block = np.asarray(episode.actions[start:stop], dtype=np.float32)
        if block.shape != (FRAME_SKIP, ACTION_DIM):
            raise ValueError("v2 history action block must have shape [5, 4]")
        blocks.append(block)
    actions = np.stack(blocks).astype(np.float32, copy=False)
    if actions.shape != (HISTORY_SIZE - 1, FRAME_SKIP, ACTION_DIM):
        raise ValueError("v2 history actions have an unexpected shape")
    return actions


def build_episode_candidate_bank(
    episode: PhysicalEpisode,
    snapshot_step: int,
    frozen: FrozenInputs,
    config: Mapping[str, Any],
) -> ActionCandidates:
    """Build one immutable action bank shared by every camera view."""

    evaluation = _object(config.get("evaluation"), "evaluation")
    start = int(snapshot_step)
    expert = np.asarray(
        episode.actions[start : start + RAW_ACTION_STEPS],
        dtype=np.float32,
    )
    if expert.shape != (RAW_ACTION_STEPS, ACTION_DIM):
        raise ValueError("expert candidate is shorter than the fixed 25 actions")
    plan = SimpleNamespace(
        snapshot_step=start,
        records=episode.records_clean,
        actions=episode.actions,
    )
    policy = policy_action_chunk(plan, frozen)
    shuffle_rng = np.random.default_rng(
        _namespace_seed(
            config,
            "same_episode_temporal_shuffle",
            task=episode.task,
            episode=episode.episode,
            snapshot_step=start,
        )
    )
    temporal_order = shuffle_rng.permutation(RAW_ACTION_STEPS)
    if np.array_equal(temporal_order, np.arange(RAW_ACTION_STEPS)):
        temporal_order = np.roll(temporal_order, FRAME_SKIP)
    shuffled = expert[temporal_order]
    rng = np.random.default_rng(
        _namespace_seed(
            config,
            "candidate_content",
            task=episode.task,
            episode=episode.episode,
            snapshot_step=start,
        )
    )
    full = build_action_candidates(
        expert,
        policy_action=policy,
        shuffled_action=shuffled,
        rng=rng,
        noise_scales=tuple(float(value) for value in evaluation["noise_scales"]),
        random_count=int(evaluation["random_count"]),
        action_low=np.asarray(episode.env.action_space.low, dtype=np.float32),
        action_high=np.asarray(episode.env.action_space.high, dtype=np.float32),
    )
    family_names = tuple(str(value) for value in evaluation["candidate_names"])
    lookup = {name: index for index, name in enumerate(full.names)}
    missing = [name for name in family_names if name not in lookup]
    if missing:
        raise ValueError(f"v2 candidate selection contains unknown names: {missing}")
    actions = np.stack([full.actions[lookup[name]] for name in family_names]).astype(
        np.float32,
        copy=False,
    )
    order_rng = np.random.default_rng(
        _namespace_seed(
            config,
            "candidate_slot_permutation",
            task=episode.task,
            episode=episode.episode,
            snapshot_step=start,
        )
    )
    order = order_rng.permutation(len(family_names))
    names = tuple(family_names[index] for index in order.tolist())
    actions = actions[order]
    result = ActionCandidates(
        names=names,
        actions=actions,
        expert_index=names.index("expert"),
    )
    if result.actions.shape != (len(names), RAW_ACTION_STEPS, ACTION_DIM):
        raise RuntimeError("v2 candidate bank shape mismatch")
    return result


def execute_candidate_bank_once(
    episode: PhysicalEpisode,
    snapshot_step: int,
    candidates: ActionCandidates,
    config: Mapping[str, Any],
    *,
    executor: Callable[
        [PhysicalEpisode, int, np.ndarray, Mapping[str, Any]],
        PhysicalBranchOutcome,
    ] = execute_physical_branch,
) -> dict[str, PhysicalBranchOutcome]:
    """Execute every physical candidate exactly once, without replay validation."""

    outcomes: dict[str, PhysicalBranchOutcome] = {}
    for index, name in enumerate(candidates.names):
        if name in outcomes:
            raise RuntimeError(f"duplicate candidate execution requested: {name}")
        action = np.asarray(candidates.actions[index], dtype=np.float32)
        outcome = executor(episode, int(snapshot_step), action, config)
        expected = sha256_array(action)
        if outcome.action_sha256 != expected:
            raise RuntimeError(f"candidate {name} outcome is bound to another action")
        outcomes[name] = outcome
    if tuple(outcomes) != candidates.names:
        raise RuntimeError("candidate execution order differs from the frozen bank")
    return outcomes


def camera_invariant_branch_labels(
    candidates: ActionCandidates,
    outcomes: Mapping[str, PhysicalBranchOutcome],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return [C,P] physical labels with no camera/view axis."""

    returns = np.stack(
        [outcomes[name].cumulative_return for name in candidates.names]
    ).astype(np.float32, copy=False)
    progress = np.stack([outcomes[name].progress for name in candidates.names]).astype(
        np.float32,
        copy=False,
    )
    success = np.stack([outcomes[name].success for name in candidates.names]).astype(
        bool,
        copy=False,
    )
    expected = (len(candidates.names), TRANSITIONS)
    if returns.shape != expected or progress.shape != expected or success.shape != expected:
        raise ValueError("physical candidate labels must have shape [C, 5]")
    if not np.isfinite(returns).all() or not np.isfinite(progress).all():
        raise ValueError("physical candidate labels must be finite")
    return returns, progress, success


@lru_cache(maxsize=4)
def _goal_projection(
    input_dim: int,
    output_dim: int,
    seed: int,
) -> torch.Tensor:
    """Create the protocol-fixed orthonormal projection used by goal context."""

    if input_dim < output_dim or output_dim <= 0:
        raise ValueError("goal projection requires input_dim >= output_dim > 0")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(
        input_dim,
        output_dim,
        generator=generator,
        dtype=torch.float64,
    )
    basis, triangular = torch.linalg.qr(matrix, mode="reduced")
    signs = torch.sign(torch.diagonal(triangular))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return (basis * signs.unsqueeze(0)).to(dtype=torch.float32)


@torch.inference_mode()
def goal_visual_from_current_rgb(
    rgb: np.ndarray,
    frozen: FrozenInputs,
    config: Mapping[str, Any],
) -> np.ndarray:
    """Encode current RGB with the frozen ViT and a fixed 48D projection."""

    value = np.asarray(rgb, dtype=np.float32)
    if value.ndim != 3 or value.shape[0] != 3 or not np.isfinite(value).all():
        raise ValueError("goal visual input must be finite CHW RGB")
    tensor = torch.from_numpy(value).unsqueeze(0).to(frozen.device)
    _, prefix = frozen.corrector.vision.extract_backbone_features(tensor)
    if prefix.ndim != 2 or prefix.shape[0] != 1:
        raise ValueError("frozen ViT prefix must have shape [1, visual_dim]")
    projection = _goal_projection(
        int(prefix.shape[-1]),
        48,
        goal_visual_projection_seed(int(config["seed"])),
    ).to(device=prefix.device, dtype=torch.float32)
    normalized = F.layer_norm(prefix.float(), (int(prefix.shape[-1]),))
    result = (normalized @ projection)[0].cpu().numpy().astype(np.float32)
    if result.shape != (48,) or not np.isfinite(result).all():
        raise RuntimeError("frozen-ViT goal visual must have exactly 48 values")
    return result


def _project_goal_label(
    goal_world: np.ndarray,
    intrinsic: np.ndarray,
    true_extrinsic: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, bool]:
    from fcpg.geometry import Camera, project_points

    pixel, valid = project_points(
        np.asarray(goal_world, dtype=np.float64)[None],
        Camera(
            intrinsic=np.asarray(intrinsic, dtype=np.float64),
            extrinsic_cam_from_world=np.asarray(true_extrinsic, dtype=np.float64),
            width=int(width),
            height=int(height),
        ),
    )
    point = np.asarray(pixel[0], dtype=np.float32)
    is_valid = bool(valid[0]) and bool(np.isfinite(point).all())
    if not np.isfinite(point).all():
        point = np.zeros(2, dtype=np.float32)
    return point, is_valid


def render_history_views(
    episode: PhysicalEpisode,
    snapshot_step: int,
    config: Mapping[str, Any],
) -> tuple[
    dict[str, list[FrameRecord]],
    np.ndarray,
    tuple[tuple[str, ...], ...],
    tuple[tuple[str, ...], ...],
]:
    """Re-render one physical history under every fixed camera condition."""

    if tuple(config["conditions"]) != VIEW_NAMES:
        raise ValueError("v2 camera view order differs from the schema")
    steps = _history_indices(int(snapshot_step), HISTORY_SIZE, FRAME_SKIP)
    records_by_view: dict[str, list[FrameRecord]] = {}
    extrinsics: list[np.ndarray] = []
    physical_by_view: list[tuple[str, ...]] = []
    rng_by_view: list[tuple[str, ...]] = []
    physical_reference: tuple[str, ...] | None = None
    rng_reference: tuple[str, ...] | None = None
    for view in VIEW_NAMES:
        condition = _object(config["conditions"][view], f"condition {view}")
        view_records: list[FrameRecord] = []
        view_extrinsics: list[np.ndarray] = []
        physical_hashes: list[str] = []
        rng_hashes: list[str] = []
        for step in steps:
            record, extrinsic, physical_hash, rng_hash = render_state_record(
                episode,
                int(step),
                condition,
                config,
            )
            view_records.append(record)
            view_extrinsics.append(extrinsic)
            physical_hashes.append(physical_hash)
            rng_hashes.append(rng_hash)
        if not all(
            np.array_equal(view_extrinsics[0], value)
            for value in view_extrinsics[1:]
        ):
            raise RuntimeError("fixed camera extrinsic changed within one history")
        physical_tuple = tuple(physical_hashes)
        rng_tuple = tuple(rng_hashes)
        if physical_reference is None:
            physical_reference = physical_tuple
            rng_reference = rng_tuple
        elif physical_tuple != physical_reference or rng_tuple != rng_reference:
            raise RuntimeError("camera views do not share the same physical/RNG history")
        records_by_view[view] = view_records
        extrinsics.append(view_extrinsics[-1])
        physical_by_view.append(physical_tuple)
        rng_by_view.append(rng_tuple)
    assert physical_reference is not None and rng_reference is not None
    return (
        records_by_view,
        np.stack(extrinsics).astype(np.float32, copy=False),
        tuple(physical_by_view),
        tuple(rng_by_view),
    )


def _camera_invariant_history(
    records_by_view: Mapping[str, Sequence[FrameRecord]],
) -> tuple[list[FrameRecord], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract physical history fields and prove they do not depend on view."""

    reference = list(records_by_view[VIEW_NAMES[0]])
    if len(reference) != HISTORY_SIZE:
        raise ValueError("v2 history must contain exactly three records")
    field_names = ("proprio", "keypoints_xyz", "q_nominal", "valid_nominal")
    for view in VIEW_NAMES[1:]:
        records = records_by_view[view]
        if len(records) != HISTORY_SIZE:
            raise ValueError("all camera views must contain three history records")
        for left, right in zip(reference, records):
            for name in field_names:
                if not np.array_equal(getattr(left, name), getattr(right, name)):
                    raise RuntimeError(
                        f"camera-only rendering changed physical history field {name}"
                    )
    return (
        reference,
        np.stack([record.proprio for record in reference]).astype(
            np.float32, copy=False
        ),
        np.stack([record.keypoints_xyz for record in reference]).astype(
            np.float32, copy=False
        ),
        np.stack([record.q_nominal for record in reference]).astype(
            np.float32, copy=False
        ),
        np.stack([record.valid_nominal for record in reference]).astype(
            bool, copy=False
        ),
    )


@torch.inference_mode()
def extract_lewm_feature_grid(
    records_by_view: Mapping[str, Sequence[FrameRecord]],
    history_actions_raw: np.ndarray,
    candidates: ActionCandidates,
    frozen: FrozenInputs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract the frozen MxGxV history and action-conditioned future latents."""

    raw_history = np.asarray(history_actions_raw, dtype=np.float32)
    if raw_history.shape != (HISTORY_SIZE - 1, FRAME_SKIP, ACTION_DIM):
        raise ValueError("history action tensor must have shape [2, 5, 4]")
    history_actions = raw_history.reshape(HISTORY_SIZE - 1, TRANSITION_ACTION_DIM)
    future_actions = reshape_action_chunks(candidates.actions)
    if future_actions.shape != (
        len(candidates.names),
        TRANSITIONS,
        TRANSITION_ACTION_DIM,
    ):
        raise ValueError("future action tensor must have shape [C, 5, 20]")

    corrected_by_view: list[np.ndarray] = []
    valid_by_view: list[np.ndarray] = []
    for view in VIEW_NAMES:
        records = list(records_by_view[view])
        corrected = corrected_points(records, frozen).astype(np.float32, copy=False)
        valid_nominal = np.stack([record.valid_nominal for record in records]).astype(
            bool, copy=False
        )
        valid = grounded_valid_mask(
            valid_nominal,
            corrected,
            width=int(records[0].rgb.shape[-1]),
            height=int(records[0].rgb.shape[-2]),
        )
        corrected_by_view.append(corrected)
        valid_by_view.append(valid)

    history_rows: list[list[list[np.ndarray]]] = []
    future_rows: list[list[list[np.ndarray]]] = []
    history_action_tensor = (
        torch.from_numpy(history_actions).unsqueeze(0).to(frozen.device)
    )
    future_action_tensor = (
        torch.from_numpy(future_actions).unsqueeze(0).to(frozen.device)
    )
    for model_name in MODEL_NAMES:
        if model_name not in frozen.models:
            raise KeyError(f"frozen LeWM model is missing {model_name!r}")
        model = frozen.models[model_name]
        model_history: list[list[np.ndarray]] = []
        model_future: list[list[np.ndarray]] = []
        for grounding in GROUNDING_NAMES:
            grounding_history: list[np.ndarray] = []
            grounding_future: list[np.ndarray] = []
            for view_index, view in enumerate(VIEW_NAMES):
                records = list(records_by_view[view])
                q_corrected = (
                    corrected_by_view[view_index] if grounding == "fcpg" else None
                )
                batch = build_lewm_history_batch(
                    records,
                    device=frozen.device,
                    q_corrected=q_corrected,
                )
                assert_no_privileged_inputs(
                    batch,
                    "Counterfactual v2 LeWM deployment batch",
                )
                _, embedding = model.encode_frames(batch)
                predicted = model.rollout_embeddings(
                    history_embedding=embedding,
                    history_action=history_action_tensor,
                    future_action=future_action_tensor,
                )
                history = embedding[0].float().cpu().numpy().astype(
                    np.float32, copy=False
                )
                future = predicted[0].float().cpu().numpy().astype(
                    np.float32, copy=False
                )
                if history.shape != (HISTORY_SIZE, LATENT_DIM):
                    raise ValueError("LeWM history latent shape mismatch")
                if future.shape != (
                    len(candidates.names),
                    TRANSITIONS,
                    LATENT_DIM,
                ):
                    raise ValueError("LeWM future latent shape mismatch")
                if not np.isfinite(history).all() or not np.isfinite(future).all():
                    raise ValueError("LeWM emitted non-finite latent values")
                grounding_history.append(history)
                grounding_future.append(future)
            model_history.append(grounding_history)
            model_future.append(grounding_future)
        history_rows.append(model_history)
        future_rows.append(model_future)

    return (
        np.stack(corrected_by_view).astype(np.float32, copy=False),
        np.stack(valid_by_view).astype(bool, copy=False),
        np.stack(history_rows).astype(np.float32, copy=False),
        np.stack(future_rows).astype(np.float32, copy=False),
    )


def render_oracle_future_grid(
    episode: PhysicalEpisode,
    candidates: ActionCandidates,
    outcomes: Mapping[str, PhysicalBranchOutcome],
    frozen: FrozenInputs,
    config: Mapping[str, Any],
) -> tuple[
    np.ndarray,
    tuple[tuple[tuple[str, ...], ...], ...],
    tuple[tuple[tuple[str, ...], ...], ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Re-render endpoints, fail closed on bad pixels, and encode oracle targets.

    Full endpoint RGB is deliberately not persisted because it is an offline
    label source.  Instead, the audit partition retains per-frame dynamic
    range, camera-difference counts, and exact SHA-256 digests.
    """

    environment = _object(config.get("environment"), "environment")
    image_size = (int(environment["width"]), int(environment["height"]))
    rows: list[np.ndarray] = []
    physical_by_view: list[tuple[tuple[str, ...], ...]] = []
    rng_by_view: list[tuple[tuple[str, ...], ...]] = []
    rgb_by_view: list[np.ndarray] = []
    for view in VIEW_NAMES:
        condition = _object(config["conditions"][view], f"condition {view}")
        records: list[FrameRecord] = []
        view_physical: list[tuple[str, ...]] = []
        view_rng: list[tuple[str, ...]] = []
        for name in candidates.names:
            outcome = outcomes[name]
            candidate_physical: list[str] = []
            candidate_rng: list[str] = []
            for endpoint_index in range(TRANSITIONS):
                record, physical_hash, rng_hash = render_endpoint_record(
                    episode,
                    outcome,
                    endpoint_index,
                    condition,
                    config,
                )
                records.append(record)
                candidate_physical.append(physical_hash)
                candidate_rng.append(rng_hash)
            view_physical.append(tuple(candidate_physical))
            view_rng.append(tuple(candidate_rng))
        targets = oracle_targets(records, frozen, image_size=image_size)
        expected = (len(candidates.names) * TRANSITIONS, LATENT_DIM)
        if targets.shape != expected or not np.isfinite(targets).all():
            raise ValueError("oracle endpoint target shape or finiteness mismatch")
        endpoint_rgb = np.stack([_rgb_uint8(record) for record in records]).reshape(
            len(candidates.names),
            TRANSITIONS,
            image_size[1],
            image_size[0],
            3,
        )
        rgb_by_view.append(endpoint_rgb)
        rows.append(targets.reshape(len(candidates.names), TRANSITIONS, LATENT_DIM))
        physical_by_view.append(tuple(view_physical))
        rng_by_view.append(tuple(view_rng))

    if any(value != physical_by_view[0] for value in physical_by_view[1:]):
        raise RuntimeError("endpoint camera views do not share exact physical states")
    if any(value != rng_by_view[0] for value in rng_by_view[1:]):
        raise RuntimeError("endpoint camera views do not share exact RNG states")

    endpoint_rgb = np.stack(rgb_by_view).astype(np.uint8, copy=False)
    frame_min = endpoint_rgb.min(axis=(-3, -2, -1)).astype(np.uint16)
    frame_max = endpoint_rgb.max(axis=(-3, -2, -1)).astype(np.uint16)
    dynamic_range = frame_max - frame_min
    constant = np.argwhere(dynamic_range == 0)
    if constant.size:
        view_index, candidate_index, endpoint_index = (
            int(value) for value in constant[0]
        )
        raise RuntimeError(
            "oracle endpoint RGB contains a constant frame at "
            f"view={VIEW_NAMES[view_index]}, candidate={candidates.names[candidate_index]}, "
            f"endpoint={endpoint_index}"
        )

    changed_pixels = np.count_nonzero(
        np.any(endpoint_rgb[1:] != endpoint_rgb[:1], axis=-1),
        axis=(-2, -1),
    ).astype(np.uint32, copy=False)
    identical = np.argwhere(changed_pixels == 0)
    if identical.size:
        perturbed_index, candidate_index, endpoint_index = (
            int(value) for value in identical[0]
        )
        raise RuntimeError(
            "oracle endpoint RGB perturbed view is pixel-identical to clean at "
            f"view={VIEW_NAMES[perturbed_index + 1]}, "
            f"candidate={candidates.names[candidate_index]}, endpoint={endpoint_index}"
        )

    digest_bytes = np.empty(
        (len(VIEW_NAMES), len(candidates.names), TRANSITIONS, 32),
        dtype=np.uint8,
    )
    for view_index in range(len(VIEW_NAMES)):
        for candidate_index in range(len(candidates.names)):
            for endpoint_index in range(TRANSITIONS):
                digest_bytes[view_index, candidate_index, endpoint_index] = (
                    np.frombuffer(
                        bytes.fromhex(
                            schema_sha256_array(
                                endpoint_rgb[
                                    view_index,
                                    candidate_index,
                                    endpoint_index,
                                ]
                            )
                        ),
                        dtype=np.uint8,
                    )
                )
    return (
        np.stack(rows).astype(np.float32, copy=False),
        tuple(physical_by_view),
        tuple(rng_by_view),
        dynamic_range,
        changed_pixels,
        digest_bytes,
    )


def _sample_index_row(
    episode: PhysicalEpisode,
    *,
    phase_index: int,
    snapshot_step: int,
    candidates: ActionCandidates,
    outcomes: Mapping[str, PhysicalBranchOutcome],
    history_physical_hashes_by_view: Sequence[Sequence[str]],
    history_rng_hashes_by_view: Sequence[Sequence[str]],
    endpoint_physical_hashes_by_view: Sequence[Sequence[Sequence[str]]],
    endpoint_rng_hashes_by_view: Sequence[Sequence[Sequence[str]]],
    actual_return: np.ndarray,
    actual_progress: np.ndarray,
    actual_success: np.ndarray,
    oracle_future_latent: np.ndarray,
    endpoint_rgb_sha256: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    family = tuple(str(value) for value in config["evaluation"]["candidate_names"])
    permutation = [family.index(name) for name in candidates.names]
    action_hashes = [schema_sha256_array(row) for row in candidates.actions]
    expected_endpoint_physical = [
        [
            physical_snapshot_sha256(snapshot)
            for snapshot in outcomes[name].endpoint_snapshots
        ]
        for name in candidates.names
    ]
    expected_endpoint_rng = [
        [rng_snapshot_sha256(snapshot) for snapshot in outcomes[name].endpoint_snapshots]
        for name in candidates.names
    ]
    history_steps = _history_indices(int(snapshot_step), HISTORY_SIZE, FRAME_SKIP)
    expected_history_physical = [
        physical_snapshot_sha256(episode.snapshots[step]) for step in history_steps
    ]
    expected_history_rng = [
        rng_snapshot_sha256(episode.snapshots[step]) for step in history_steps
    ]
    history_physical = [list(row) for row in history_physical_hashes_by_view]
    history_rng = [list(row) for row in history_rng_hashes_by_view]
    endpoint_physical = [
        [list(endpoint) for endpoint in view]
        for view in endpoint_physical_hashes_by_view
    ]
    endpoint_rng = [
        [list(endpoint) for endpoint in view]
        for view in endpoint_rng_hashes_by_view
    ]
    for view_index, view in enumerate(VIEW_NAMES):
        if history_physical[view_index] != expected_history_physical:
            raise RuntimeError(f"{view} measured history physical hashes mismatch sources")
        if history_rng[view_index] != expected_history_rng:
            raise RuntimeError(f"{view} measured history RNG hashes mismatch sources")
        if endpoint_physical[view_index] != expected_endpoint_physical:
            raise RuntimeError(f"{view} measured endpoint physical hashes mismatch sources")
        if endpoint_rng[view_index] != expected_endpoint_rng:
            raise RuntimeError(f"{view} measured endpoint RNG hashes mismatch sources")
    receipt_fields = build_branch_receipt_fields(
        candidate_names=candidates.names,
        candidate_actions=candidates.actions,
        actual_return=actual_return,
        actual_progress=actual_progress,
        actual_success=actual_success,
        oracle_future_latent=oracle_future_latent,
        endpoint_physical_sha256_by_view=endpoint_physical,
        endpoint_rng_sha256_by_view=endpoint_rng,
        endpoint_rgb_sha256=endpoint_rgb_sha256,
    )
    return {
        "sample_index": -1,
        "task_index": TASK_NAMES.index(episode.task),
        "task_name": episode.task,
        "episode_id": int(episode.episode),
        "env_seed": int(episode.env_seed),
        "phase_index": int(phase_index),
        "phase_name": PHASE_NAMES[int(phase_index)],
        "snapshot_step": int(snapshot_step),
        "split": "pending",
        "physical_episode_sha256": physical_episode_sha256(
            TASK_NAMES.index(episode.task),
            episode.episode,
            episode.env_seed,
        ),
        "snapshot_sha256": snapshot_sha256(episode.snapshots[int(snapshot_step)]),
        "history_physical_sha256_by_view": history_physical,
        "history_rng_sha256_by_view": history_rng,
        "candidate_names": list(candidates.names),
        "candidate_permutation": permutation,
        "candidate_permutation_sha256": canonical_sha256(permutation),
        "candidate_action_sha256": action_hashes,
        "candidate_action_set_sha256": canonical_sha256(
            {
                "members": sorted(
                    [
                        {"name": name, "action_sha256": digest}
                        for name, digest in zip(candidates.names, action_hashes)
                    ],
                    key=lambda item: item["name"],
                )
            }
        ),
        **receipt_fields,
        "branch_execution_count": len(candidates.names),
        "endpoint_physical_sha256_by_view": endpoint_physical,
        "endpoint_rng_sha256_by_view": endpoint_rng,
        "view_names": list(VIEW_NAMES),
    }


def collect_phase_row(
    episode: PhysicalEpisode,
    phase_index: int,
    frozen: FrozenInputs,
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Collect one task/episode/phase row without any hindsight input path."""

    snapshot_step = PHASE_STEPS[int(phase_index)]
    records_by_view, true_extrinsic, history_physical, history_rng = (
        render_history_views(episode, snapshot_step, config)
    )
    reference, proprio, xyz, q_nominal, valid_nominal = _camera_invariant_history(
        records_by_view
    )
    history_actions = _history_raw_actions(
        episode,
        snapshot_step,
    )
    candidates = build_episode_candidate_bank(
        episode,
        snapshot_step,
        frozen,
        config,
    )
    outcomes = execute_candidate_bank_once(
        episode,
        snapshot_step,
        candidates,
        config,
    )
    actual_return, actual_progress, actual_success = camera_invariant_branch_labels(
        candidates,
        outcomes,
    )
    q_corrected, valid_grounded, z_history, z_pred_future = (
        extract_lewm_feature_grid(
            records_by_view,
            history_actions,
            candidates,
            frozen,
        )
    )
    (
        oracle_future,
        endpoint_physical,
        endpoint_rng,
        endpoint_rgb_dynamic_range,
        endpoint_rgb_changed_pixels,
        endpoint_rgb_sha256,
    ) = render_oracle_future_grid(episode, candidates, outcomes, frozen, config)
    history_rgb = np.stack(
        [
            np.stack([_rgb_uint8(record) for record in records_by_view[view]])
            for view in VIEW_NAMES
        ]
    ).astype(np.uint8, copy=False)
    goal_visual = np.stack(
        [
            goal_visual_from_current_rgb(records_by_view[view][-1].rgb, frozen, config)
            for view in VIEW_NAMES
        ]
    ).astype(np.float32, copy=False)
    q_true = np.stack(
        [
            np.stack([record.q_true for record in records_by_view[view]])
            for view in VIEW_NAMES
        ]
    ).astype(np.float32, copy=False)
    valid_true = np.stack(
        [
            np.stack([record.valid_true for record in records_by_view[view]])
            for view in VIEW_NAMES
        ]
    ).astype(bool, copy=False)
    environment = _object(config.get("environment"), "environment")
    goal_rows = [
        _project_goal_label(
            episode.goal_world,
            episode.intrinsic,
            true_extrinsic[index],
            width=int(environment["width"]),
            height=int(environment["height"]),
        )
        for index in range(len(VIEW_NAMES))
    ]
    goal_pixel = np.stack([row[0] for row in goal_rows]).astype(
        np.float32, copy=False
    )
    goal_valid = np.asarray([row[1] for row in goal_rows], dtype=bool)
    task_index = TASK_NAMES.index(episode.task)
    deployment = {
        "task_index": np.asarray(task_index, dtype=np.uint8),
        "history_rgb": history_rgb,
        "history_proprio": proprio,
        "history_keypoints_xyz": xyz,
        "history_q_nominal": q_nominal,
        "history_valid_nominal": valid_nominal,
        "history_q_corrected": q_corrected,
        "history_valid_grounded": valid_grounded,
        "history_action_raw": history_actions,
        "candidate_action_raw": candidates.actions.astype(np.float32, copy=False),
        "z_history": z_history,
        "z_pred_future": z_pred_future,
        "goal_visual": goal_visual,
        "task_onehot": np.eye(len(TASK_NAMES), dtype=np.float32)[task_index],
    }
    labels = {
        "actual_return": actual_return,
        "actual_progress": actual_progress,
        "actual_success": actual_success,
        "oracle_future_latent": oracle_future,
        "history_q_true": q_true,
        "history_valid_true": valid_true,
        "goal_world": episode.goal_world.astype(np.float32, copy=False),
        "goal_pixel": goal_pixel,
        "goal_valid": goal_valid,
    }
    audit = {
        "episode_id": np.asarray(episode.episode, dtype=np.int32),
        "env_seed": np.asarray(episode.env_seed, dtype=np.int64),
        "phase_index": np.asarray(phase_index, dtype=np.uint8),
        "snapshot_step": np.asarray(snapshot_step, dtype=np.int16),
        "camera_intrinsic": np.repeat(
            episode.intrinsic[None], len(VIEW_NAMES), axis=0
        ).astype(np.float32, copy=False),
        "true_extrinsic": true_extrinsic.astype(np.float32, copy=False),
        "nominal_extrinsic": episode.nominal_extrinsic.astype(
            np.float32, copy=False
        ),
        "endpoint_rgb_dynamic_range": endpoint_rgb_dynamic_range,
        "endpoint_rgb_changed_pixels": endpoint_rgb_changed_pixels,
        "endpoint_rgb_sha256": endpoint_rgb_sha256,
    }
    assert_no_privileged_inputs(deployment)
    index = _sample_index_row(
        episode,
        phase_index=phase_index,
        snapshot_step=snapshot_step,
        candidates=candidates,
        outcomes=outcomes,
        history_physical_hashes_by_view=history_physical,
        history_rng_hashes_by_view=history_rng,
        endpoint_physical_hashes_by_view=endpoint_physical,
        endpoint_rng_hashes_by_view=endpoint_rng,
        actual_return=actual_return,
        actual_progress=actual_progress,
        actual_success=actual_success,
        oracle_future_latent=oracle_future,
        endpoint_rgb_sha256=endpoint_rgb_sha256,
        config=config,
    )
    return deployment, labels, audit, index


def _stack_sample_mappings(rows: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("Counterfactual v2 collection produced no sample rows")
    keys = set(rows[0])
    if any(set(row) != keys for row in rows[1:]):
        raise ValueError("sample rows have inconsistent array keys")
    return {name: np.stack([row[name] for row in rows]) for name in sorted(keys)}


def _reorder_arrays(
    values: Mapping[str, np.ndarray],
    order: np.ndarray,
) -> dict[str, np.ndarray]:
    return {name: np.asarray(value)[order] for name, value in values.items()}


def _current_frozen_state(frozen: FrozenInputs) -> dict[str, str]:
    return {
        "nominal": sha256_state_dict(frozen.models["nominal"]),
        "fcpg": sha256_state_dict(frozen.models["fcpg"]),
        "action_only_policy": sha256_state_dict(frozen.policy),
        "fcpg_corrector": sha256_state_dict(frozen.corrector),
        "oracle_target_encoder": sha256_state_dict(frozen.target_encoder),
        "oracle_vision_backbone": sha256_state_dict(
            frozen.corrector.vision.backbone
        ),
    }


def _input_provenance(
    config: Mapping[str, Any],
    frozen: FrozenInputs,
) -> dict[str, Any]:
    def leaf(config_leaf: Mapping[str, Any], record_key: str) -> dict[str, str]:
        record = frozen.file_records[record_key]
        path = Path(str(record["path"]))
        actual = sha256_file(path)
        expected = str(config_leaf["sha256"])
        if actual != expected or actual != str(record["sha256"]):
            raise RuntimeError(f"frozen input changed during collection: {path}")
        return {
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
        }

    raw = _object(config.get("inputs"), "inputs")
    lewm = _object(raw.get("lewm_checkpoints"), "lewm checkpoints")
    return {
        "lewm_checkpoints": {
            name: leaf(_object(lewm[name], name), f"lewm_{name}")
            for name in MODEL_NAMES
        },
        "action_only_policy": leaf(
            _object(raw["action_only_policy"], "action policy"),
            "action_only_policy",
        ),
        "fcpg_corrector": leaf(
            _object(raw["fcpg_corrector"], "FCPG corrector"),
            "fcpg_corrector",
        ),
        "oracle_target_encoder": leaf(
            _object(raw["oracle_target_encoder"], "oracle target"),
            "oracle_target_encoder",
        ),
    }


def _goal_projection_provenance(
    frozen: FrozenInputs,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    input_dim = int(frozen.target_encoder.visual_dim)
    projection_seed = goal_visual_projection_seed(int(config["seed"]))
    projection = _goal_projection(input_dim, 48, projection_seed).numpy()
    return {
        "seed": projection_seed,
        "sha256": schema_sha256_array(projection),
        "input_dim": input_dim,
        "output_dim": 48,
        "source": "current_rgb_frozen_vit_prefix",
    }


def run_counterfactual_v2_collection(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> CollectionPayload:
    """Collect the immutable v2 feature/label bundle without upstream training."""

    from fcpg.metaworld_snapshot import (
        capture_process_rng_snapshot,
        restore_process_rng_snapshot,
    )

    is_smoke = int(config["evaluation"]["episodes_per_task"]) == 1
    if is_smoke:
        validate_smoke_config(config)
    else:
        validate_full_config(config)
    git_before = _git_state(repo_root)
    caller_rng = capture_process_rng_snapshot()
    deployment_rows: list[dict[str, np.ndarray]] = []
    label_rows: list[dict[str, np.ndarray]] = []
    audit_rows: list[dict[str, np.ndarray]] = []
    index_rows: list[dict[str, Any]] = []
    try:
        _set_seed(int(config["seed"]))
        frozen = load_frozen_inputs(config, repo_root=repo_root)
        state_before = _current_frozen_state(frozen)
        evaluation = _object(config.get("evaluation"), "evaluation")
        for task in TASK_NAMES:
            for episode_id in range(int(evaluation["episodes_per_task"])):
                episode = collect_physical_episode(task, episode_id, config)
                try:
                    for phase_index in range(len(PHASE_STEPS)):
                        deployment, labels, audit, index = collect_phase_row(
                            episode,
                            phase_index,
                            frozen,
                            config,
                        )
                        deployment_rows.append(deployment)
                        label_rows.append(labels)
                        audit_rows.append(audit)
                        index_rows.append(index)
                finally:
                    try:
                        _close_episode_env_safely(episode.env)
                    finally:
                        # The MetaWorld renderer owns an EGL context beyond its
                        # viewer map. Drop the environment before constructing
                        # the next task; overlapping renderer lifetimes can make
                        # the new context return constant black frames.
                        del episode

        deployment = _stack_sample_mappings(deployment_rows)
        labels = _stack_sample_mappings(label_rows)
        audit = _stack_sample_mappings(audit_rows)
        typed = DeploymentInputs.from_mapping(deployment)
        typed_audit = AuditArrays.from_mapping(audit)
        order = canonical_row_order(typed, typed_audit)
        deployment = _reorder_arrays(deployment, order)
        labels = _reorder_arrays(labels, order)
        audit = _reorder_arrays(audit, order)
        index_rows = [index_rows[index] for index in order.tolist()]

        if is_smoke:
            splits = np.full(len(index_rows), "smoke", dtype="<U10")
        else:
            splits, _ = assign_physical_episode_splits(
                deployment["task_index"],
                audit["episode_id"],
                audit["env_seed"],
                train_count_per_task=int(config["split"]["train_count_per_task"]),
                validation_count_per_task=int(
                    config["split"]["validation_count_per_task"]
                ),
            )
        for sample_index, (row, split) in enumerate(zip(index_rows, splits.tolist())):
            row["sample_index"] = sample_index
            row["split"] = str(split)

        candidate_count = len(evaluation["candidate_names"])
        axes = validate_array_bundle(
            deployment,
            labels,
            audit,
            candidate_count=candidate_count,
        )
        assert_no_privileged_inputs(deployment)
        state_after = _current_frozen_state(frozen)
        if state_after != state_before:
            raise RuntimeError("a frozen model changed during v2 collection")
        inputs = _input_provenance(config, frozen)
        git_after = _git_state(repo_root)
        if git_after != git_before:
            raise RuntimeError("repository state changed during v2 collection")
        config_sha256 = canonical_sha256(config)
        physical_branch_count = axes.sample_count * axes.candidate_count
        result_contract = {
            "config_sha256": config_sha256,
            "sample_count": axes.sample_count,
            "physical_branch_count": physical_branch_count,
        }
        provenance = {
            "schema_version": 2,
            "protocol": str(config["protocol"]),
            "scope": _jsonable(config["scope"]),
            "config": _jsonable(config),
            "config_sha256": config_sha256,
            "git": {"before": git_before, "after": git_after},
            "versions": _versions(),
            "inputs": inputs,
            "model_state_before": state_before,
            "model_state_after": state_after,
            "no_training_verified": True,
            "goal_visual_projection": _goal_projection_provenance(frozen, config),
            "sample_count": axes.sample_count,
            "physical_branch_count": physical_branch_count,
            "upstream_non_overlap": {
                "status": "not_auditable",
                "reason": (
                    "pinned upstream artifacts do not expose a physical-episode "
                    "seed ledger"
                ),
                "training_episode_ledger_sha256": None,
            },
            "result_contract_sha256": canonical_sha256(result_contract),
        }
        validate_index_and_provenance(
            config,
            deployment,
            labels,
            audit,
            index_rows,
            provenance,
            smoke=is_smoke,
        )
        return CollectionPayload(
            deployment_inputs=deployment,
            offline_labels=labels,
            audit_arrays=audit,
            index_rows=tuple(index_rows),
            provenance=provenance,
        )
    finally:
        restore_process_rng_snapshot(caller_rng)
