"""Physical same-snapshot action-branch collection for the A1 gate.

This module only collects factual action effects.  The center trajectory is
the source scripted expert's actual future action sequence; it is not a
deployment policy proposal or a CEM candidate set.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from fcpg.lewm_action_branch_a1 import (
    ACTION_DIM as CHUNK_ACTION_DIM,
    CANDIDATE_COUNT,
    EMBEDDING_DIM,
    HISTORY_STEPS,
    KEYPOINT_COUNT,
    MIN_HORIZON,
    RAW_ACTION_DIM,
    RAW_STEPS_PER_HORIZON,
    TARGET_MOTION_RMS,
    build_anonymous_action_branches,
    reshape_action_branches,
    validate_a1_arrays,
)
from fcpg.lewm_counterfactual_runtime import (
    FrameRecord,
    _collect_frame_record,
    capture_branch_snapshot,
    corrected_points,
    grounded_valid_mask,
    restore_branch_snapshot,
)
from fcpg.lewm_counterfactual_v2_runtime import _close_episode_env_safely


HISTORY_SIZE = HISTORY_STEPS
FRAME_SKIP = RAW_STEPS_PER_HORIZON
ACTION_DIM = RAW_ACTION_DIM
ANCHOR_COUNT = KEYPOINT_COUNT
LATENT_DIM = EMBEDDING_DIM
MIN_HORIZON_STEPS = MIN_HORIZON
CENTER_SOURCE = "source_scripted_expert_future"
COLLECTION_PURPOSE = "same_state_action_effect_identification"
FORMAL_TASK_INSTANCES = tuple(range(48))
SMOKE_TASK_INSTANCES = (48, 49)
FORMAL_SPLIT_COUNTS = (("train", 24), ("development", 12), ("test", 12))


def _readonly_float32(value: Any) -> np.ndarray:
    result = np.array(value, dtype=np.float32, copy=True, order="C")
    result.setflags(write=False)
    return result


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _record_value(record: FrameRecord | Mapping[str, Any], name: str) -> np.ndarray:
    value = record[name] if isinstance(record, Mapping) else getattr(record, name)
    return np.asarray(value)


@dataclass(frozen=True, slots=True)
class HazardScheduleConfig:
    decision_steps: int
    eligible_first_step: int
    eligible_last_step: int
    probability: float
    refractory_steps: int
    maximum_events: int
    condition_on_at_least_one_event: bool

    def __post_init__(self) -> None:
        if self.decision_steps <= 0:
            raise ValueError("decision_steps must be positive")
        if not 0 <= self.eligible_first_step <= self.eligible_last_step:
            raise ValueError("hazard eligible step range is invalid")
        if self.eligible_last_step >= self.decision_steps:
            raise ValueError("hazard eligible_last_step must be below decision_steps")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("hazard probability must be in [0,1]")
        if self.refractory_steps < 0:
            raise ValueError("hazard refractory_steps must be non-negative")
        if self.maximum_events <= 0:
            raise ValueError("hazard maximum_events must be positive")
        if self.condition_on_at_least_one_event and self.probability <= 0.0:
            raise ValueError("a conditioned hazard schedule requires positive probability")

    @classmethod
    def from_contract(cls, config: Mapping[str, Any]) -> "HazardScheduleConfig":
        hazard = _mapping(config.get("hazard"), "hazard")
        if int(hazard.get("history_steps", -1)) != HISTORY_SIZE:
            raise ValueError(f"A1 history_steps must be {HISTORY_SIZE}")
        if not bool(hazard.get("schedule_drawn_before_environment_reset", False)):
            raise ValueError("A1 hazards must be drawn before environment reset")
        if not bool(
            hazard.get("schedule_independent_of_state_action_reward_and_task_phase", False)
        ):
            raise ValueError("A1 hazard schedule must be independent of episode signals")
        return cls(
            decision_steps=int(hazard["decision_steps_per_episode"]),
            eligible_first_step=int(hazard["eligible_first_step"]),
            eligible_last_step=int(hazard["eligible_last_step"]),
            probability=float(hazard["bernoulli_probability"]),
            refractory_steps=int(hazard["refractory_steps"]),
            maximum_events=int(hazard["maximum_events_per_episode"]),
            condition_on_at_least_one_event=bool(
                hazard["condition_on_at_least_one_event"]
            ),
        )


def sample_hazard_schedule(
    config: HazardScheduleConfig,
    *,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Draw a state/action/reward-independent random hazard schedule.

    Rejection sampling implements the contract's conditioning on at least one
    event.  It happens before an episode runtime (and therefore reset) exists.
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    eligible = range(config.eligible_first_step, config.eligible_last_step + 1)
    for _ in range(100_000):
        events: list[int] = []
        for step in eligible:
            if len(events) >= config.maximum_events:
                break
            if events and step - events[-1] < config.refractory_steps:
                continue
            if float(rng.random()) < config.probability:
                events.append(step)
        if events or not config.condition_on_at_least_one_event:
            return tuple(events)
    raise RuntimeError("failed to draw a non-empty conditioned hazard schedule")


@dataclass(frozen=True, slots=True)
class ActionBranchBank:
    """Eight anonymous equal-RMS branches around one factual center."""

    names: tuple[str, ...]
    center_action: np.ndarray
    actions: np.ndarray

    def __post_init__(self) -> None:
        expected_names = tuple(f"candidate_{index}" for index in range(CANDIDATE_COUNT))
        if tuple(self.names) != expected_names:
            raise ValueError("action branches require eight fixed anonymous slot names")
        center = np.asarray(self.center_action)
        actions = np.asarray(self.actions)
        if center.ndim != 2 or center.shape[1] != ACTION_DIM:
            raise ValueError("center_action must have shape [raw_steps,4]")
        if center.shape[0] < MIN_HORIZON_STEPS * FRAME_SKIP:
            raise ValueError("A1 center_action must contain at least 40 raw actions")
        if center.shape[0] % FRAME_SKIP:
            raise ValueError("A1 raw action count must be divisible by frame_skip=5")
        if actions.shape != (CANDIDATE_COUNT, *center.shape):
            raise ValueError("branch actions must have shape [8,raw_steps,4]")
        if not np.isfinite(center).all() or not np.isfinite(actions).all():
            raise ValueError("center and branch actions must be finite")
        object.__setattr__(self, "center_action", _readonly_float32(center))
        object.__setattr__(self, "actions", _readonly_float32(actions))


def build_action_branch_bank(
    center_action: np.ndarray,
    *,
    direction_seed: int,
    permutation_seed: int,
    action_low: Any = -1.0,
    action_high: Any = 1.0,
    motion_delta_rms: float = TARGET_MOTION_RMS,
) -> ActionBranchBank:
    """Generalize the v3 four-antithetic-axis bank to ``P >= 8``.

    The public v3 builder is fixed to 25 raw actions.  A1 reuses its fixed
    radius and clipping solver in a configurable ``P x 3`` transition space.
    No action-policy horizon is consulted and no center action is padded.
    """

    if float(motion_delta_rms) != TARGET_MOTION_RMS:
        raise ValueError(f"A1 motion_delta_rms must remain fixed at {TARGET_MOTION_RMS}")
    center32 = np.asarray(center_action, dtype=np.float32)
    horizon_steps = center32.shape[0] // FRAME_SKIP if center32.ndim == 2 else None
    actions = build_anonymous_action_branches(
        center32,
        direction_seed=int(direction_seed),
        permutation_seed=int(permutation_seed),
        horizon=horizon_steps,
        action_low=action_low,
        action_high=action_high,
    )
    return ActionBranchBank(
        names=tuple(f"candidate_{index}" for index in range(CANDIDATE_COUNT)),
        center_action=center32,
        actions=actions,
    )


@torch.inference_mode()
def encode_fcpg_anchor_records(
    records: Sequence[FrameRecord],
    *,
    model: Any,
    corrector: Any,
    device: torch.device,
) -> np.ndarray:
    """Encode current rendered records using FCPG ``q_corrected`` only.

    ``q_true`` and canonical-camera images are deliberately absent from the
    model batch.  Each returned frame retains the three-anchor axis.
    """

    rows = list(records)
    if not rows:
        raise ValueError("at least one current-image record is required")
    corrected = corrected_points(
        rows,
        SimpleNamespace(corrector=corrector, device=device),
    ).astype(np.float32, copy=False)
    pixels = np.stack([_record_value(record, "rgb") for record in rows]).astype(
        np.float32, copy=False
    )
    proprio = np.stack([_record_value(record, "proprio") for record in rows]).astype(
        np.float32, copy=False
    )
    q_nominal = np.stack(
        [_record_value(record, "q_nominal") for record in rows]
    ).astype(np.float32, copy=False)
    keypoints_xyz = np.stack(
        [_record_value(record, "keypoints_xyz") for record in rows]
    ).astype(np.float32, copy=False)
    valid_nominal = np.stack(
        [_record_value(record, "valid_nominal") for record in rows]
    ).astype(bool, copy=False)
    if pixels.ndim != 4 or pixels.shape[1] != 3:
        raise ValueError("current RGB records must have shape [T,3,H,W]")
    valid = grounded_valid_mask(
        valid_nominal,
        corrected,
        width=int(pixels.shape[-1]),
        height=int(pixels.shape[-2]),
    )
    batch = {
        "pixels": torch.from_numpy(np.ascontiguousarray(pixels)).unsqueeze(0).to(device),
        "proprio": torch.from_numpy(np.ascontiguousarray(proprio)).unsqueeze(0).to(
            device
        ),
        "q_nominal": torch.from_numpy(np.ascontiguousarray(q_nominal)).unsqueeze(0).to(
            device
        ),
        "q_corrected": torch.from_numpy(np.ascontiguousarray(corrected)).unsqueeze(0).to(
            device
        ),
        "keypoints_xyz": torch.from_numpy(
            np.ascontiguousarray(keypoints_xyz)
        ).unsqueeze(0).to(device),
        "valid": torch.from_numpy(np.ascontiguousarray(valid)).unsqueeze(0).to(device),
    }
    _, embedding = model.encode_frames(batch)
    expected = (1, len(rows), ANCHOR_COUNT, LATENT_DIM)
    if tuple(embedding.shape) != expected or not bool(torch.isfinite(embedding).all()):
        raise ValueError(f"AnchorPersistent embedding must be finite {expected}")
    return embedding[0].float().cpu().numpy().astype(np.float32, copy=False)


@dataclass(frozen=True)
class FrozenFCPGAnchorEncoder:
    model: Any
    corrector: Any
    device: torch.device

    def encode(self, records: Sequence[FrameRecord]) -> np.ndarray:
        return encode_fcpg_anchor_records(
            records,
            model=self.model,
            corrector=self.corrector,
            device=self.device,
        )


def load_frozen_fcpg_anchor_encoder(
    anchor_checkpoint: str | Path,
    fcpg_checkpoint: str | Path,
    *,
    device: str | torch.device,
) -> FrozenFCPGAnchorEncoder:
    """Load and freeze the existing FCPG and AnchorPersistent checkpoints."""

    from fcpg.lewm_anchor_persistent import AnchorPersistentLeWM
    from fcpg.lewm_runtime import load_lewm_grounded_checkpoint
    from fcpg.transformer_grounding import load_fcpg_vit_adapter

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {target_device}")
    model, _, checkpoint = load_lewm_grounded_checkpoint(
        anchor_checkpoint,
        device=target_device,
    )
    if not isinstance(model, AnchorPersistentLeWM):
        raise TypeError("A1 requires a frozen AnchorPersistentLeWM checkpoint")
    if (
        int(model.history_size),
        int(model.num_keypoints),
        int(model.latent_dim),
    ) != (HISTORY_SIZE, ANCHOR_COUNT, LATENT_DIM):
        raise ValueError("AnchorPersistent checkpoint dimensions violate A1")
    data_config = checkpoint.get("data_config", {})
    if isinstance(data_config, Mapping) and "frame_skip" in data_config:
        if int(data_config["frame_skip"]) != FRAME_SKIP:
            raise ValueError("AnchorPersistent checkpoint frame_skip must be 5")
    corrector, _ = load_fcpg_vit_adapter(fcpg_checkpoint, target_device)
    model.requires_grad_(False).eval()
    corrector.requires_grad_(False).eval()
    return FrozenFCPGAnchorEncoder(model, corrector, target_device)


@dataclass
class EpisodeRuntime:
    env: Any
    expert: Any
    observation: np.ndarray
    record_current: Callable[[np.ndarray], FrameRecord]


def make_metaworld_episode_runtime(
    task: str,
    task_index_mod50: int,
    environment_seed: int,
    *,
    camera: str,
    width: int,
    height: int,
    benchmark_seed: int,
) -> EpisodeRuntime:
    """Construct and reset the real MetaWorld source episode."""

    from fcpg.metaworld_utils import make_env, make_scripted_policy
    from fcpg.mujoco_camera import (
        camera_extrinsic_cv_from_mujoco,
        camera_intrinsic_from_fovy,
        get_camera_id,
    )

    instance = int(task_index_mod50)
    if not 0 <= instance < 50:
        raise ValueError("MetaWorld task_index_mod50 must be in [0,49]")
    env = make_env(
        task,
        camera,
        int(width),
        int(height),
        task_index=instance,
        benchmark_seed=int(benchmark_seed),
    )
    try:
        observation, _ = env.reset(seed=int(environment_seed))
        camera_id = get_camera_id(env.model, camera)
        intrinsic = camera_intrinsic_from_fovy(
            int(width),
            int(height),
            float(env.model.cam_fovy[camera_id]),
        )
        nominal_extrinsic = camera_extrinsic_cv_from_mujoco(env.data, camera_id)

        def record_current(current_observation: np.ndarray) -> FrameRecord:
            return _collect_frame_record(
                env,
                current_observation,
                camera_id=camera_id,
                intrinsic=intrinsic,
                nominal_extrinsic=nominal_extrinsic,
                width=int(width),
                height=int(height),
            )

        return EpisodeRuntime(
            env=env,
            expert=make_scripted_policy(task),
            observation=np.asarray(observation, dtype=np.float32),
            record_current=record_current,
        )
    except Exception:
        _close_episode_env_safely(env)
        raise


def execute_action_branches(
    runtime: EpisodeRuntime,
    snapshot: Any,
    branch_actions: np.ndarray,
    *,
    encode_records: Callable[[Sequence[FrameRecord]], np.ndarray],
    restore_snapshot: Callable[[Any, Any], None] = restore_branch_snapshot,
) -> np.ndarray:
    """Restore one snapshot for every branch and encode each actual endpoint."""

    actions = np.asarray(branch_actions, dtype=np.float32)
    if actions.ndim != 3 or actions.shape[0] != CANDIDATE_COUNT:
        raise ValueError("branch_actions must have shape [8,raw_steps,4]")
    if actions.shape[2] != ACTION_DIM or actions.shape[1] % FRAME_SKIP:
        raise ValueError("branch_actions raw/action dimensions violate A1")
    horizon_steps = actions.shape[1] // FRAME_SKIP
    if horizon_steps < MIN_HORIZON_STEPS:
        raise ValueError("A1 branch horizon must contain at least eight endpoints")
    low = np.asarray(runtime.env.action_space.low, dtype=np.float32)
    high = np.asarray(runtime.env.action_space.high, dtype=np.float32)
    if not np.array_equal(actions, np.clip(actions, low, high)):
        raise ValueError("branch executor refuses actions that require clipping")

    branch_embeddings: list[np.ndarray] = []
    for branch in actions:
        restore_snapshot(runtime.env, snapshot)
        endpoint_records: list[FrameRecord] = []
        for raw_index, action in enumerate(branch, start=1):
            observation, _, terminated, truncated, _ = runtime.env.step(action)
            if raw_index % FRAME_SKIP == 0:
                endpoint_records.append(
                    runtime.record_current(np.asarray(observation, dtype=np.float32))
                )
            if (terminated or truncated) and raw_index < branch.shape[0]:
                raise RuntimeError("action branch ended before the configured horizon")
        if len(endpoint_records) != horizon_steps:
            raise RuntimeError("action branch endpoint count is incomplete")
        embedding = np.asarray(encode_records(endpoint_records), dtype=np.float32)
        expected = (horizon_steps, ANCHOR_COUNT, LATENT_DIM)
        if embedding.shape != expected or not np.isfinite(embedding).all():
            raise ValueError(f"future endpoint embedding must be finite {expected}")
        branch_embeddings.append(embedding)
    return np.stack(branch_embeddings).astype(np.float32, copy=False)


@dataclass(frozen=True, slots=True)
class ActionBranchRow:
    task: str
    task_index: int
    task_instance_index: int
    source_episode_id: str
    environment_seed: int
    hazard_step: int
    split: str
    center_action: np.ndarray
    history_action: np.ndarray
    branch_action: np.ndarray
    history_embedding: np.ndarray
    future_embedding: np.ndarray

    def __post_init__(self) -> None:
        center = np.asarray(self.center_action)
        horizon_steps = center.shape[0] // FRAME_SKIP if center.ndim == 2 else -1
        expected = {
            "center_action": (horizon_steps * FRAME_SKIP, ACTION_DIM),
            "history_action": (HISTORY_SIZE - 1, FRAME_SKIP, ACTION_DIM),
            "branch_action": (
                CANDIDATE_COUNT,
                horizon_steps * FRAME_SKIP,
                ACTION_DIM,
            ),
            "history_embedding": (HISTORY_SIZE, ANCHOR_COUNT, LATENT_DIM),
            "future_embedding": (
                CANDIDATE_COUNT,
                horizon_steps,
                ANCHOR_COUNT,
                LATENT_DIM,
            ),
        }
        values = {
            "center_action": center,
            "history_action": np.asarray(self.history_action),
            "branch_action": np.asarray(self.branch_action),
            "history_embedding": np.asarray(self.history_embedding),
            "future_embedding": np.asarray(self.future_embedding),
        }
        if horizon_steps < MIN_HORIZON_STEPS:
            raise ValueError("row horizon must be at least eight")
        for name, value in values.items():
            if value.shape != expected[name] or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape {expected[name]}")
            object.__setattr__(self, name, _readonly_float32(value))
        if (
            not self.task
            or not self.source_episode_id
            or int(self.task_index) < 0
            or not 0 <= int(self.task_instance_index) < 50
            or int(self.hazard_step) < 0
        ):
            raise ValueError("row task/source_episode/hazard identity is invalid")
        if self.split not in {"smoke", "train", "development", "test"}:
            raise ValueError(f"unknown A1 split {self.split!r}")


def _seeded_rng(
    protocol_seed: int,
    namespace: str,
    task_index: int,
    source_episode: int,
    stream: int,
) -> np.random.Generator:
    namespace_id = zlib.crc32(namespace.encode("utf-8"))
    sequence = np.random.SeedSequence(
        [
            int(protocol_seed),
            int(namespace_id),
            int(task_index),
            int(source_episode),
            int(stream),
        ]
    )
    return np.random.default_rng(sequence)


def _candidate_seeds(
    protocol_seed: int,
    task_index: int,
    source_episode: int,
    hazard_step: int,
) -> tuple[int, int]:
    values = np.random.SeedSequence(
        [int(protocol_seed), int(task_index), int(source_episode), int(hazard_step)]
    ).generate_state(2, dtype=np.uint32)
    return int(values[0]), int(values[1])


def collect_source_episode(
    *,
    task: str,
    task_seed_index: int,
    task_index_mod50: int,
    environment_seed: int,
    split: str,
    contract: Mapping[str, Any],
    horizon_steps: int,
    runtime_factory: Callable[[str, int, int], EpisodeRuntime],
    encode_records: Callable[[Sequence[FrameRecord]], np.ndarray],
    max_events: int | None = None,
    branch_bank_builder: Callable[..., ActionBranchBank] = build_action_branch_bank,
    capture_snapshot: Callable[..., Any] = capture_branch_snapshot,
    restore_snapshot: Callable[[Any, Any], None] = restore_branch_snapshot,
    schedule_sampler: Callable[..., tuple[int, ...]] = sample_hazard_schedule,
) -> list[ActionBranchRow]:
    """Collect every selected hazard in one scripted source episode.

    The hazard draw is deliberately the first episode-dependent operation.  In
    particular, ``runtime_factory`` (which performs reset) is called later.
    """

    if int(horizon_steps) < MIN_HORIZON_STEPS:
        raise ValueError(f"A1 horizon_steps must be at least {MIN_HORIZON_STEPS}")
    hazard_config = HazardScheduleConfig.from_contract(contract)
    protocol_seed = int(_mapping(contract.get("split"), "split")["protocol_seed"])
    namespace = str(
        _mapping(contract.get("split"), "split").get(
            "smoke_episode_namespace" if split == "smoke" else "assignment",
            split,
        )
    )
    schedule_rng = _seeded_rng(
        protocol_seed,
        namespace,
        int(task_seed_index),
        int(task_index_mod50),
        0,
    )
    hazards = tuple(schedule_sampler(hazard_config, rng=schedule_rng))
    if max_events is not None:
        if int(max_events) <= 0:
            raise ValueError("max_events must be positive when provided")
        hazards = hazards[: int(max_events)]

    runtime = runtime_factory(task, int(task_index_mod50), int(environment_seed))
    try:
        hazard_raw_steps = {hazard: hazard * FRAME_SKIP for hazard in hazards}
        required_record_steps: set[int] = set()
        for raw_step in hazard_raw_steps.values():
            required_record_steps.update(
                (raw_step - 2 * FRAME_SKIP, raw_step - FRAME_SKIP, raw_step)
            )
        if required_record_steps and min(required_record_steps) < 0:
            raise ValueError("hazard occurs before the complete H=3 history")
        final_raw_step = (
            max(hazard_raw_steps.values()) + int(horizon_steps) * FRAME_SKIP
            if hazard_raw_steps
            else 0
        )
        records: dict[int, FrameRecord] = {}
        snapshots: dict[int, Any] = {}
        source_actions: list[np.ndarray] = []
        observation = np.asarray(runtime.observation, dtype=np.float32)
        raw_to_hazard = {raw: hazard for hazard, raw in hazard_raw_steps.items()}
        for raw_step in range(final_raw_step + 1):
            if raw_step in raw_to_hazard:
                snapshots[raw_to_hazard[raw_step]] = capture_snapshot(
                    runtime.env,
                    include_process_rng=True,
                )
            if raw_step in required_record_steps:
                records[raw_step] = runtime.record_current(observation)
            if raw_step == final_raw_step:
                break
            action = np.asarray(runtime.expert.get_action(observation), dtype=np.float32)
            if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
                raise ValueError("scripted expert produced an invalid 4D action")
            action = np.clip(
                action,
                runtime.env.action_space.low,
                runtime.env.action_space.high,
            ).astype(np.float32)
            source_actions.append(action)
            observation, _, terminated, truncated, _ = runtime.env.step(action)
            observation = np.asarray(observation, dtype=np.float32)
            if (terminated or truncated) and raw_step + 1 < final_raw_step:
                raise RuntimeError("scripted source episode ended before future centers")

        action_array = np.asarray(source_actions, dtype=np.float32)
        rows: list[ActionBranchRow] = []
        branch_config = _mapping(
            contract.get("matched_state_branches"), "matched_state_branches"
        )
        if int(branch_config.get("branch_count", -1)) != CANDIDATE_COUNT:
            raise ValueError("A1 matched_state_branches.branch_count must be 8")
        if not bool(branch_config.get("preserve_policy_gripper_commands", False)):
            raise ValueError("A1 branches must preserve center gripper commands")
        radius_value = branch_config.get("motion_delta_rms", TARGET_MOTION_RMS)
        radius = None if radius_value is None else float(radius_value)
        for hazard_step in hazards:
            raw_step = hazard_raw_steps[hazard_step]
            stop = raw_step + int(horizon_steps) * FRAME_SKIP
            center_action = action_array[raw_step:stop]
            expected_center = (int(horizon_steps) * FRAME_SKIP, ACTION_DIM)
            if center_action.shape != expected_center:
                raise RuntimeError(
                    "scripted source did not provide the exact configured future center"
                )
            history_records = [
                records[raw_step - 2 * FRAME_SKIP],
                records[raw_step - FRAME_SKIP],
                records[raw_step],
            ]
            history_action = np.stack(
                (
                    action_array[raw_step - 2 * FRAME_SKIP : raw_step - FRAME_SKIP],
                    action_array[raw_step - FRAME_SKIP : raw_step],
                )
            )
            history_embedding = np.asarray(
                encode_records(history_records), dtype=np.float32
            )
            if history_embedding.shape != (HISTORY_SIZE, ANCHOR_COUNT, LATENT_DIM):
                raise ValueError("history embedding must have shape [3,3,192]")
            direction_seed, permutation_seed = _candidate_seeds(
                protocol_seed,
                int(task_seed_index),
                int(task_index_mod50),
                int(hazard_step),
            )
            bank = branch_bank_builder(
                center_action,
                direction_seed=direction_seed,
                permutation_seed=permutation_seed,
                action_low=runtime.env.action_space.low,
                action_high=runtime.env.action_space.high,
                motion_delta_rms=radius,
            )
            future_embedding = execute_action_branches(
                runtime,
                snapshots[hazard_step],
                bank.actions,
                encode_records=encode_records,
                restore_snapshot=restore_snapshot,
            )
            rows.append(
                ActionBranchRow(
                    task=task,
                    task_index=int(task_seed_index),
                    task_instance_index=int(task_index_mod50),
                    source_episode_id=(
                        f"{task}:task-instance-{int(task_index_mod50):02d}"
                    ),
                    environment_seed=int(environment_seed),
                    hazard_step=int(hazard_step),
                    split=split,
                    center_action=bank.center_action,
                    history_action=history_action,
                    branch_action=bank.actions,
                    history_embedding=history_embedding,
                    future_embedding=future_embedding,
                )
            )
        return rows
    finally:
        _close_episode_env_safely(runtime.env)


def assign_source_episode_splits(
    tasks: Sequence[str],
    task_instances: Mapping[str, Sequence[int]],
    contract: Mapping[str, Any],
    *,
    smoke: bool,
) -> dict[tuple[str, int], str]:
    """Assign all hazards from a source episode to one preregistered split."""

    result: dict[tuple[str, int], str] = {}
    if smoke:
        for task in tasks:
            instances = tuple(int(value) for value in task_instances[task])
            if not instances or not set(instances).issubset(SMOKE_TASK_INSTANCES):
                raise ValueError("smoke must use only MetaWorld task instances 48 and 49")
            for instance in instances:
                result[(task, instance)] = "smoke"
        return result
    split = _mapping(contract.get("split"), "split")
    protocol_seed = int(split["protocol_seed"])
    for task in tasks:
        instances = tuple(int(value) for value in task_instances[task])
        if tuple(sorted(instances)) != FORMAL_TASK_INSTANCES:
            raise ValueError(f"{task} formal collection must cover task instances 0..47")
        task_seed = np.random.SeedSequence(
            [protocol_seed, zlib.crc32(task.encode("utf-8"))]
        )
        ranked = np.random.default_rng(task_seed).permutation(instances).tolist()
        offset = 0
        for name, count in FORMAL_SPLIT_COUNTS:
            for instance in ranked[offset : offset + count]:
                result[(task, int(instance))] = name
            offset += count
    return result


@dataclass(frozen=True)
class CollectionPayload:
    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]


def stack_action_branch_rows(
    rows: Sequence[ActionBranchRow],
) -> dict[str, np.ndarray]:
    """Stack event rows into training-ready arrays without RGB or snapshots."""

    values = list(rows)
    if not values:
        raise ValueError("A1 collection produced no hazard rows")
    horizon_steps = values[0].center_action.shape[0] // FRAME_SKIP
    if any(row.center_action.shape[0] != horizon_steps * FRAME_SKIP for row in values):
        raise ValueError("all A1 rows must share one configured horizon")
    episode_splits: dict[tuple[str, int], str] = {}
    for row in values:
        key = (row.task, int(row.task_instance_index))
        previous = episode_splits.setdefault(key, row.split)
        if previous != row.split:
            raise ValueError(
                f"MetaWorld task instance crosses splits: {key} {previous}/{row.split}"
            )
    arrays = {
        "history_embedding": np.stack(
            [row.history_embedding for row in values]
        ).astype(np.float32, copy=False),
        "history_action": np.stack(
            [row.history_action for row in values]
        ).reshape(len(values), HISTORY_SIZE - 1, CHUNK_ACTION_DIM).astype(
            np.float32, copy=False
        ),
        "center_action": np.stack(
            [row.center_action for row in values]
        ).reshape(len(values), horizon_steps, CHUNK_ACTION_DIM).astype(
            np.float32, copy=False
        ),
        "future_action": np.stack(
            [reshape_action_branches(row.branch_action) for row in values]
        ).astype(np.float32, copy=False),
        "target_embedding": np.stack(
            [row.future_embedding for row in values]
        ).astype(np.float32, copy=False),
        "task_index": np.asarray(
            [int(row.task_index) for row in values], dtype=np.int64
        ),
        "task_instance_index": np.asarray(
            [int(row.task_instance_index) for row in values], dtype=np.int64
        ),
        "source_episode_id": np.asarray(
            [row.source_episode_id for row in values]
        ),
        "hazard_id": np.asarray(
            [row.hazard_step for row in values], dtype=np.int64
        ),
        "split": np.asarray([row.split for row in values]),
    }
    forbidden = {"rgb", "snapshot", "q_true", "canonical"}
    if any(any(token in name for token in forbidden) for name in arrays):
        raise RuntimeError("A1 persisted arrays contain a forbidden target input")
    validate_a1_arrays(arrays)
    return arrays


def run_action_branch_collection(
    contract: Mapping[str, Any],
    *,
    encoder: FrozenFCPGAnchorEncoder | Any,
    runtime_factory: Callable[[str, int, int], EpisodeRuntime],
    horizon_steps: int = MIN_HORIZON_STEPS,
    environment_seed_base: int = 40_000,
    smoke: bool,
    max_tasks: int | None = None,
    max_episodes_per_task: int | None = None,
    max_events_per_episode: int | None = None,
) -> CollectionPayload:
    """Run the bounded A1 collection; this function never starts training."""

    configured_tasks = tuple(str(value) for value in contract.get("tasks", ()))
    if not configured_tasks:
        raise ValueError("A1 contract has no tasks")
    if max_tasks is not None:
        if not smoke:
            raise ValueError("max_tasks is a smoke-only bound")
        if int(max_tasks) <= 0:
            raise ValueError("max_tasks must be positive")
        tasks = configured_tasks[: int(max_tasks)]
    else:
        tasks = configured_tasks
    episode_count = len(SMOKE_TASK_INSTANCES) if smoke else len(FORMAL_TASK_INSTANCES)
    if max_episodes_per_task is not None:
        if not smoke:
            raise ValueError("max_episodes_per_task is a smoke-only bound")
        if int(max_episodes_per_task) <= 0:
            raise ValueError("max_episodes_per_task must be positive")
        if int(max_episodes_per_task) > len(SMOKE_TASK_INSTANCES):
            raise ValueError("smoke has exactly task instances 48 and 49 available")
        episode_count = int(max_episodes_per_task)
    if max_events_per_episode is not None and not smoke:
        raise ValueError("max_events_per_episode is a smoke-only bound")

    selected_instances = (
        SMOKE_TASK_INSTANCES[:episode_count] if smoke else FORMAL_TASK_INSTANCES
    )
    namespace_offset = 1_000_000 if smoke else 0
    task_instances = {task: selected_instances for task in tasks}
    environment_seeds = {
        task: tuple(
            int(environment_seed_base)
            + namespace_offset
            + configured_tasks.index(task) * 10_000
            + instance
            for instance in selected_instances
        )
        for task in tasks
    }
    assignments = assign_source_episode_splits(
        tasks,
        task_instances,
        contract,
        smoke=smoke,
    )
    rows: list[ActionBranchRow] = []
    for task in tasks:
        task_seed_index = configured_tasks.index(task)
        for position, task_index_mod50 in enumerate(selected_instances):
            environment_seed = environment_seeds[task][position]
            rows.extend(
                collect_source_episode(
                    task=task,
                    task_seed_index=task_seed_index,
                    task_index_mod50=task_index_mod50,
                    environment_seed=environment_seed,
                    split=assignments[(task, task_index_mod50)],
                    contract=contract,
                    horizon_steps=int(horizon_steps),
                    runtime_factory=runtime_factory,
                    encode_records=encoder.encode,
                    max_events=max_events_per_episode,
                )
            )
    arrays = stack_action_branch_rows(rows)
    return CollectionPayload(
        arrays=arrays,
        metadata={
            "history_size": HISTORY_SIZE,
            "frame_skip": FRAME_SKIP,
            "horizon_steps": int(horizon_steps),
            "raw_action_steps": int(horizon_steps) * FRAME_SKIP,
            "branch_count": CANDIDATE_COUNT,
            "anchor_count": ANCHOR_COUNT,
            "latent_dim": LATENT_DIM,
            "center_source": CENTER_SOURCE,
            "purpose": COLLECTION_PURPOSE,
            "smoke": bool(smoke),
        },
    )


__all__ = [
    "ACTION_DIM",
    "ANCHOR_COUNT",
    "ActionBranchBank",
    "ActionBranchRow",
    "CENTER_SOURCE",
    "COLLECTION_PURPOSE",
    "CollectionPayload",
    "EpisodeRuntime",
    "FRAME_SKIP",
    "FrozenFCPGAnchorEncoder",
    "HISTORY_SIZE",
    "HazardScheduleConfig",
    "LATENT_DIM",
    "MIN_HORIZON_STEPS",
    "assign_source_episode_splits",
    "build_action_branch_bank",
    "collect_source_episode",
    "encode_fcpg_anchor_records",
    "execute_action_branches",
    "load_frozen_fcpg_anchor_encoder",
    "make_metaworld_episode_runtime",
    "run_action_branch_collection",
    "sample_hazard_schedule",
    "stack_action_branch_rows",
]
