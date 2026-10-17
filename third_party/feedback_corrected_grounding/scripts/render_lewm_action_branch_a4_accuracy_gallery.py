#!/usr/bin/env python3
"""Render causal sliding-window A4 q-center accuracy GIFs and HTML.

Each GIF advances through complete five-raw-step decision windows in one
reserved-test episode.  A frame shows actual future RGB at H1, H3, H5, and H8
with evaluation-only GT q-center and a q-center decoded from the ensemble LeWM
latent forecast.  The forecast receives only three current/history observation
records and the recorded history/future action chunks.  Future RGB and future
GT are never included in encoder or world-model inputs.
"""

from __future__ import annotations

import argparse
import gc
import html
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image, ImageDraw
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_a4_checkpoint import validate_a4_checkpoint  # noqa: E402
from fcpg.lewm_a4_storage import (  # noqa: E402
    ACTION_DIM,
    LATENT_DIM,
    STATE_SLOT_COUNT,
)
from fcpg.lewm_a4_task_factory import CUSTOM_TASKS  # noqa: E402
from fcpg.lewm_action_branch_a4 import (  # noqa: E402
    CONFIRMATION_TRAINING_SEEDS,
    FORMAL_TASK_INSTANCES,
    HISTORY_STEPS,
    HORIZON,
    RAW_STEPS_PER_MODEL_STEP,
    TASKS,
    assign_a4_splits,
    validate_a4_contract,
)
from fcpg.lewm_action_branch_a4_collection import (  # noqa: E402
    load_frozen_a4_state_encoder,
)
from fcpg.lewm_action_branch_a4_training import (  # noqa: E402
    initialize_a4_model_from_anchor_base,
)
from fcpg.lewm_counterfactual_runtime import corrected_points  # noqa: E402
from scripts.render_lewm_action_branch_a4_trajectory_comparison import (  # noqa: E402
    LEWM_COLOR,
    GROUND_TRUTH_COLOR,
    TASK_LABELS,
    _chunk_actions as _safe_chunk_actions,
    _collect_full_episode as _safe_collect_full_episode,
    _decode_open_loop as _safe_decode_open_loop,
    _font as _safe_font,
    _load_decoder as _safe_load_decoder,
    _read_json as _safe_read_json,
    _torch_load as _safe_torch_load,
)


ALL_HORIZONS = tuple(range(1, HORIZON + 1))
DISPLAY_HORIZONS = (1, 3, 5, 8)
HISTORY_RAW_SPAN = (HISTORY_STEPS - 1) * RAW_STEPS_PER_MODEL_STEP
FUTURE_RAW_SPAN = HORIZON * RAW_STEPS_PER_MODEL_STEP

CAUSAL_OBSERVATION_FIELDS = (
    "rgb",
    "proprio",
    "keypoints_xyz",
    "q_nominal",
    "valid_nominal",
)
FORBIDDEN_MODEL_FIELDS = frozenset(
    {
        "q_true",
        "valid_true",
        "future_rgb",
        "future_q_center",
        "ground_truth",
        "target_state",
    }
)

DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_active_control_a4.json"
DEFAULT_BASE_CHECKPOINT = ROOT / (
    "outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k/"
    "lewm_grounded_latest.pt"
)
DEFAULT_FCPG_CHECKPOINT = ROOT / (
    "outputs/transformer_residual/lewm_stage1/shared_4task_seed0_sig0/"
    "fcpg_transformer_residual_best.pt"
)
DEFAULT_A4_CHECKPOINTS = tuple(
    ROOT
    / f"outputs/lewm_action_branch_a4/shared_7task_seed{seed}_10k/"
    "lewm_action_branch_a4_latest.pt"
    for seed in CONFIRMATION_TRAINING_SEEDS
)
DEFAULT_DECODER = ROOT / (
    "outputs/lewm_action_branch_a4/task_conditioned_direction_readout/decoder.npz"
)
DEFAULT_OUTPUT = ROOT / "outputs/lewm_action_branch_a4/accuracy_gallery_10k_test"

DEFAULT_MAX_RAW_STEPS = 500
DEFAULT_POST_SUCCESS_RAW_STEPS = FUTURE_RAW_SPAN
DEFAULT_FPS = 4.0

HTML_RISK_HEADLINE = (
    "latent forecast decoded by train-only probe, not RGB generation"
)
PROTOCOL_NAME = "a4_10k_test_sliding_qcenter_accuracy_gallery_v1"
DECODER_FLOOR_COLOR = (224, 138, 44)
TEACHER_FORCED_COLOR = (137, 83, 174)
METRIC_METHODS = ("decoder_floor", "teacher_forced", "recursive")


@dataclass(frozen=True)
class WindowAlignment:
    """Raw-step alignment metadata kept outside the model input payload."""

    decision_raw_step: int
    history_frame_indices: tuple[int, ...]
    history_action_starts: tuple[int, ...]
    future_action_starts: tuple[int, ...]
    future_frame_indices: tuple[int, ...]


@dataclass(frozen=True)
class CausalModelInputs:
    """The complete allowlisted input payload before visual-state encoding."""

    history_observations: tuple[Mapping[str, np.ndarray], ...]
    history_action: np.ndarray
    future_action: np.ndarray
    task_index: int


@dataclass(frozen=True)
class ConfirmationEndpoint:
    path: Path
    seed: int
    model_state: Mapping[str, Any]


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT
    )
    parser.add_argument(
        "--fcpg-checkpoint", type=Path, default=DEFAULT_FCPG_CHECKPOINT
    )
    parser.add_argument(
        "--a4-checkpoint",
        type=Path,
        action="append",
        dest="a4_checkpoints",
        help=(
            "fixed-latest A4 step-10000 checkpoint; repeat exactly three times "
            "to override the seed30301/30302/30303 defaults"
        ),
    )
    parser.add_argument(
        "--expected-step",
        type=_positive_integer,
        default=10_000,
        help=(
            "required optimizer step for every supplied checkpoint; use 2000 "
            "only for an explicitly labelled development sanity gallery"
        ),
    )
    parser.add_argument("--decoder", type=Path, default=DEFAULT_DECODER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="inference device; CUDA execution is restricted to cuda:0",
    )
    parser.add_argument(
        "--task",
        choices=TASKS,
        action="append",
        help="render one task; repeat for a task subset",
    )
    parser.add_argument(
        "--max-decision-windows",
        type=_positive_integer,
        help="maximum complete H8 windows per task; default renders all",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument(
        "--max-raw-steps",
        type=_positive_integer,
        default=DEFAULT_MAX_RAW_STEPS,
    )
    parser.add_argument(
        "--post-success-steps",
        type=_nonnegative_integer,
        default=DEFAULT_POST_SUCCESS_RAW_STEPS,
        help="expert tail retained after stable success",
    )
    return parser


def _canonical_device(value: str | torch.device) -> torch.device:
    """Normalize CUDA to index zero and reject every other CUDA index."""

    target = torch.device(value)
    if target.type != "cuda":
        return target
    index = 0 if target.index is None else int(target.index)
    if index != 0:
        raise ValueError("A4 accuracy rendering permits only CUDA device index 0")
    return torch.device("cuda:0")


def _runtime_device(value: str | torch.device) -> torch.device:
    target = _canonical_device(value)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {target}")
    return target


def _clear_runtime_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _validated_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    if isinstance(horizons, (str, bytes)):
        raise TypeError("horizons must be an integer sequence")
    result = tuple(
        _integer(value, label="horizon", minimum=1) for value in horizons
    )
    if not result:
        raise ValueError("at least one horizon is required")
    if len(set(result)) != len(result):
        raise ValueError("horizons must be unique")
    if any(value > HORIZON for value in result):
        raise ValueError(f"accuracy gallery supports only H1-H{HORIZON}")
    return result


def _horizon_frame_indices(
    decision_raw_step: int,
    horizons: Sequence[int] = ALL_HORIZONS,
) -> np.ndarray:
    """Map Hk to the exact future raw frame ``decision + 5*k``."""

    decision = _integer(
        decision_raw_step, label="decision_raw_step", minimum=0
    )
    if decision % RAW_STEPS_PER_MODEL_STEP:
        raise ValueError("decision_raw_step must lie on the five-step model grid")
    selected = _validated_horizons(horizons)
    return decision + RAW_STEPS_PER_MODEL_STEP * np.asarray(
        selected, dtype=np.int64
    )


def _window_alignment(
    decision_raw_step: int,
    *,
    raw_action_count: int,
) -> WindowAlignment:
    decision = _integer(
        decision_raw_step, label="decision_raw_step", minimum=HISTORY_RAW_SPAN
    )
    action_count = _integer(
        raw_action_count, label="raw_action_count", minimum=0
    )
    if decision % RAW_STEPS_PER_MODEL_STEP:
        raise ValueError("decision_raw_step must lie on the five-step model grid")
    future_frames = tuple(
        int(value) for value in _horizon_frame_indices(decision, ALL_HORIZONS)
    )
    if future_frames[-1] > action_count:
        raise ValueError("decision window does not contain a complete H8 future")
    history_frames = tuple(
        decision - offset
        for offset in range(
            HISTORY_RAW_SPAN,
            -1,
            -RAW_STEPS_PER_MODEL_STEP,
        )
    )
    if len(history_frames) != HISTORY_STEPS:
        raise RuntimeError("A4 history alignment did not produce three frames")
    history_action_starts = history_frames[:-1]
    future_action_starts = tuple(
        decision + index * RAW_STEPS_PER_MODEL_STEP
        for index in range(HORIZON)
    )
    return WindowAlignment(
        decision_raw_step=decision,
        history_frame_indices=history_frames,
        history_action_starts=history_action_starts,
        future_action_starts=future_action_starts,
        future_frame_indices=future_frames,
    )


def _decision_window_starts(
    raw_action_count: int,
    *,
    max_decision_windows: int | None = None,
) -> tuple[int, ...]:
    """Return every contiguous five-step decision with a complete H8 target."""

    action_count = _integer(
        raw_action_count, label="raw_action_count", minimum=0
    )
    if max_decision_windows is not None:
        limit = _integer(
            max_decision_windows,
            label="max_decision_windows",
            minimum=1,
        )
    else:
        limit = None
    last = action_count - FUTURE_RAW_SPAN
    if last < HISTORY_RAW_SPAN:
        return ()
    result = tuple(
        range(
            HISTORY_RAW_SPAN,
            last + 1,
            RAW_STEPS_PER_MODEL_STEP,
        )
    )
    return result if limit is None else result[:limit]


def _record_field(record: Any, name: str) -> np.ndarray:
    value = record[name] if isinstance(record, Mapping) else getattr(record, name)
    return np.array(value, copy=True)


def _causal_observation_view(record: Any) -> dict[str, np.ndarray]:
    """Strip GT fields before a record can reach the frozen visual encoder."""

    return {name: _record_field(record, name) for name in CAUSAL_OBSERVATION_FIELDS}


def _encoder_observation_records(
    inputs: CausalModelInputs,
) -> tuple[SimpleNamespace, ...]:
    """Adapt the allowlisted mappings to the encoder's attribute API."""

    _validate_causal_input_contract(inputs)
    return tuple(
        SimpleNamespace(**dict(observation))
        for observation in inputs.history_observations
    )


def _coerce_raw_actions(actions: np.ndarray) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] * RAW_STEPS_PER_MODEL_STEP != ACTION_DIM:
        raise ValueError("raw actions must have shape [T,4]")
    if not np.isfinite(values).all():
        raise ValueError("raw actions must be finite")
    return np.ascontiguousarray(values)


def _validate_causal_input_contract(inputs: CausalModelInputs) -> dict[str, Any]:
    """Fail closed if RGB/GT leakage enters the model payload."""

    if not isinstance(inputs, CausalModelInputs):
        raise TypeError("causal model input must be CausalModelInputs")
    if len(inputs.history_observations) != HISTORY_STEPS:
        raise ValueError("causal input requires exactly three history observations")
    for observation in inputs.history_observations:
        if not isinstance(observation, Mapping):
            raise TypeError("history observations must be mappings")
        keys = set(observation)
        if keys != set(CAUSAL_OBSERVATION_FIELDS):
            raise PermissionError(
                "history observation fields differ from the causal allowlist"
            )
        forbidden = keys & FORBIDDEN_MODEL_FIELDS
        if forbidden:
            raise PermissionError(
                f"future/GT fields are forbidden model inputs: {sorted(forbidden)}"
            )
    history_action = np.asarray(inputs.history_action)
    future_action = np.asarray(inputs.future_action)
    if history_action.shape != (HISTORY_STEPS - 1, ACTION_DIM):
        raise ValueError("history_action must have shape [2,20]")
    if future_action.shape != (HORIZON, ACTION_DIM):
        raise ValueError("future_action must have shape [8,20]")
    if not np.isfinite(history_action).all() or not np.isfinite(future_action).all():
        raise ValueError("causal action chunks must be finite")
    task_index = _integer(inputs.task_index, label="task_index", minimum=0)
    if task_index >= len(TASKS):
        raise ValueError("task_index lies outside the frozen seven-task panel")
    return {
        "model_input_fields": [
            "history_observations",
            "history_action",
            "future_action",
            "task_index",
        ],
        "history_observation_fields": list(CAUSAL_OBSERVATION_FIELDS),
        "history_observation_count": HISTORY_STEPS,
        "future_action_chunks": HORIZON,
        "future_rgb": "forbidden",
        "future_gt_q_center": "forbidden",
    }


def _build_causal_model_inputs(
    records: Sequence[Any],
    actions: np.ndarray,
    *,
    decision_raw_step: int,
    task_index: int,
) -> tuple[WindowAlignment, CausalModelInputs]:
    """Build a window without reading any post-decision observation record."""

    raw_actions = _coerce_raw_actions(actions)
    if len(records) != raw_actions.shape[0] + 1:
        raise ValueError("episode must contain T+1 records for T raw actions")
    alignment = _window_alignment(
        decision_raw_step,
        raw_action_count=raw_actions.shape[0],
    )
    history_observations = tuple(
        _causal_observation_view(records[index])
        for index in alignment.history_frame_indices
    )
    model_inputs = CausalModelInputs(
        history_observations=history_observations,
        history_action=_safe_chunk_actions(
            raw_actions, alignment.history_action_starts
        ),
        future_action=_safe_chunk_actions(
            raw_actions, alignment.future_action_starts
        ),
        task_index=_integer(task_index, label="task_index", minimum=0),
    )
    _validate_causal_input_contract(model_inputs)
    return alignment, model_inputs


def _test_instances_by_rank(
    contract: Mapping[str, Any], *, rank: int
) -> dict[str, int]:
    configured_tasks = validate_a4_contract(contract)
    assignments = assign_a4_splits(configured_tasks, contract)
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError("test instance rank must be an integer")
    result: dict[str, int] = {}
    for task in TASKS:
        candidates = sorted(
            instance
            for instance in FORMAL_TASK_INSTANCES
            if assignments[(task, instance)] == "test"
        )
        if rank < 0 or rank >= len(candidates):
            raise ValueError(
                f"test instance rank must be in 0..{len(candidates) - 1} for {task}"
            )
        result[task] = int(candidates[rank])
    return result


def _first_test_instances(contract: Mapping[str, Any]) -> dict[str, int]:
    return _test_instances_by_rank(contract, rank=0)


def _environment_seed(
    contract: Mapping[str, Any], *, task: str, task_instance: int
) -> int:
    task_index = TASKS.index(task)
    seed_base = int(contract["collection"]["environment_seed_base"])
    return seed_base + task_index * len(FORMAL_TASK_INSTANCES) + int(task_instance)


def _load_confirmation_endpoints(
    paths: Sequence[Path],
    *,
    expected_step: int = 10_000,
) -> tuple[ConfirmationEndpoint, ...]:
    sources = tuple(path.expanduser().resolve() for path in paths)
    if len(sources) != len(CONFIRMATION_TRAINING_SEEDS):
        raise ValueError("accuracy gallery requires exactly three A4 checkpoints")
    if len(set(sources)) != len(sources):
        raise ValueError("accuracy gallery checkpoint paths must be distinct")
    by_seed: dict[int, ConfirmationEndpoint] = {}
    for path in sources:
        checkpoint = validate_a4_checkpoint(_safe_torch_load(path))
        seed = int(checkpoint["seed"])
        step = int(checkpoint["step"])
        if step != int(expected_step):
            raise ValueError(
                f"accuracy gallery requires fixed step-{int(expected_step)} endpoints"
            )
        if int(expected_step) == 10_000 and seed not in CONFIRMATION_TRAINING_SEEDS:
            raise ValueError(
                "accuracy gallery requires seed30301/30302/30303 checkpoints"
            )
        if seed in by_seed:
            raise ValueError(f"duplicate A4 confirmation seed {seed}")
        by_seed[seed] = ConfirmationEndpoint(
            path=path,
            seed=seed,
            model_state=checkpoint["model_state"],
        )
    ordered_seeds = tuple(sorted(by_seed))
    if int(expected_step) == 10_000 and ordered_seeds != CONFIRMATION_TRAINING_SEEDS:
        raise ValueError("A4 confirmation checkpoint seed panel is incomplete")
    return tuple(by_seed[seed] for seed in ordered_seeds)


def _load_train_only_decoder(path: Path) -> dict[str, np.ndarray]:
    decoder = _safe_load_decoder(path)
    horizons = np.asarray(decoder.get("horizons", ALL_HORIZONS), dtype=np.int64)
    if not np.array_equal(horizons, np.asarray(ALL_HORIZONS, dtype=np.int64)):
        raise ValueError("train-only decoder must cover exactly H1-H8")
    coordinate = decoder.get("coordinate_contract")
    if coordinate is not None and str(np.asarray(coordinate).item()) != (
        "raw_runtime_model_coordinates_no_display_transform"
    ):
        raise ValueError("decoder coordinate contract differs from A4 q-center")
    return decoder


def _display_q_center(
    points: np.ndarray, *, task: str, image_height: int
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    if values.shape != (3, 2) or not np.isfinite(values).all():
        raise ValueError("current observed q points must be finite [3,2]")
    center = np.mean(values, axis=0).astype(np.float32)
    if task not in CUSTOM_TASKS:
        center[1] = float(image_height - 1) - center[1]
    return center


def _current_observed_q_center(
    inputs: CausalModelInputs,
    *,
    encoder: Any,
    task: str,
    image_height: int,
) -> np.ndarray:
    """Estimate q0 from the current observation, never from simulator GT."""

    current_record = _encoder_observation_records(inputs)[-1]
    points = corrected_points([current_record], encoder)[0]
    return _display_q_center(points, task=task, image_height=image_height)


def _encode_history(inputs: CausalModelInputs, *, encoder: Any) -> np.ndarray:
    records = _encoder_observation_records(inputs)
    state = np.asarray(encoder.encode(records), dtype=np.float32)
    expected = (HISTORY_STEPS, STATE_SLOT_COUNT, LATENT_DIM)
    if state.shape != expected or not np.isfinite(state).all():
        raise RuntimeError(f"frozen A4 history encoder must emit {expected}")
    return np.ascontiguousarray(state)


def _encode_evaluation_future(
    records: Sequence[Any],
    alignment: WindowAlignment,
    *,
    encoder: Any,
) -> np.ndarray:
    """Encode future observations only for targets and decoder-floor analysis."""

    observations = tuple(
        SimpleNamespace(**_causal_observation_view(records[index]))
        for index in alignment.future_frame_indices
    )
    state = np.asarray(encoder.encode(observations), dtype=np.float32)
    expected = (HORIZON, STATE_SLOT_COUNT, LATENT_DIM)
    if state.shape != expected or not np.isfinite(state).all():
        raise RuntimeError(f"frozen A4 future encoder must emit {expected}")
    return np.ascontiguousarray(state)


@torch.inference_mode()
def _rollout_encoded_window(
    model: Any,
    *,
    history_embedding: np.ndarray,
    history_action: np.ndarray,
    future_action: np.ndarray,
    task_index: int,
    device: torch.device,
) -> np.ndarray:
    history = np.asarray(history_embedding, dtype=np.float32)
    past_action = np.asarray(history_action, dtype=np.float32)
    planned_action = np.asarray(future_action, dtype=np.float32)
    if history.shape != (HISTORY_STEPS, STATE_SLOT_COUNT, LATENT_DIM):
        raise ValueError("history_embedding must have shape [3,4,192]")
    if past_action.shape != (HISTORY_STEPS - 1, ACTION_DIM):
        raise ValueError("history_action must have shape [2,20]")
    if planned_action.shape != (HORIZON, ACTION_DIM):
        raise ValueError("future_action must have shape [8,20]")
    prediction = model.rollout_state_embeddings(
        torch.from_numpy(np.ascontiguousarray(history))[None].to(device),
        torch.from_numpy(np.ascontiguousarray(past_action))[None].to(device),
        torch.from_numpy(np.ascontiguousarray(planned_action))[None, None].to(
            device
        ),
        torch.tensor([int(task_index)], dtype=torch.long, device=device),
    )
    expected = (1, 1, HORIZON, STATE_SLOT_COUNT, LATENT_DIM)
    if tuple(prediction.shape) != expected:
        raise RuntimeError(
            f"A4 sliding rollout shape drifted: {tuple(prediction.shape)} != {expected}"
        )
    result = prediction[0, 0].float().cpu().numpy()
    if not np.isfinite(result).all():
        raise RuntimeError("A4 sliding rollout produced non-finite latent values")
    return result


@torch.inference_mode()
def _teacher_forced_encoded_window(
    model: Any,
    *,
    history_embedding: np.ndarray,
    history_action: np.ndarray,
    future_action: np.ndarray,
    target_embedding: np.ndarray,
    task_index: int,
    device: torch.device,
) -> np.ndarray:
    """Predict each horizon one step ahead from the real preceding latent states."""

    history = np.asarray(history_embedding, dtype=np.float32)
    past_action = np.asarray(history_action, dtype=np.float32)
    planned_action = np.asarray(future_action, dtype=np.float32)
    target = np.asarray(target_embedding, dtype=np.float32)
    if history.shape != (HISTORY_STEPS, STATE_SLOT_COUNT, LATENT_DIM):
        raise ValueError("history_embedding must have shape [3,4,192]")
    if past_action.shape != (HISTORY_STEPS - 1, ACTION_DIM):
        raise ValueError("history_action must have shape [2,20]")
    if planned_action.shape != (HORIZON, ACTION_DIM):
        raise ValueError("future_action must have shape [8,20]")
    if target.shape != (HORIZON, STATE_SLOT_COUNT, LATENT_DIM):
        raise ValueError("target_embedding must have shape [8,4,192]")

    state_sequence = np.concatenate((history, target), axis=0)
    action_sequence = np.concatenate((past_action, planned_action), axis=0)
    task = torch.tensor([int(task_index)], dtype=torch.long, device=device)
    outputs = []
    for horizon_index in range(HORIZON):
        context = torch.from_numpy(
            np.ascontiguousarray(
                state_sequence[horizon_index : horizon_index + HISTORY_STEPS]
            )
        )[None].to(device)
        action = torch.from_numpy(
            np.ascontiguousarray(
                action_sequence[horizon_index : horizon_index + HISTORY_STEPS]
            )
        )[None].to(device)
        prediction = model.predict(context, action, task)
        expected = (1, HISTORY_STEPS, STATE_SLOT_COUNT, LATENT_DIM)
        if tuple(prediction.shape) != expected:
            raise RuntimeError(
                "A4 teacher-forced prediction shape drifted: "
                f"{tuple(prediction.shape)} != {expected}"
            )
        outputs.append(prediction[0, -1].float().cpu().numpy())
    result = np.stack(outputs, axis=0).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise RuntimeError("A4 teacher-forced prediction produced non-finite values")
    return result


def _decode_window_q_centers(
    prediction: np.ndarray,
    history_embedding: np.ndarray,
    current_q_center_display: np.ndarray,
    *,
    task: str,
    image_height: int,
    decoder: Mapping[str, np.ndarray],
) -> np.ndarray:
    latent = np.asarray(prediction, dtype=np.float32)
    if latent.shape != (HORIZON, STATE_SLOT_COUNT, LATENT_DIM):
        raise ValueError("window prediction must have shape [8,4,192]")
    decoded = _safe_decode_open_loop(
        latent,
        np.asarray(history_embedding, dtype=np.float32),
        np.asarray(current_q_center_display, dtype=np.float32),
        task=task,
        height=image_height,
        decoder=decoder,
    )
    expected = (HORIZON + 1, 2)
    if decoded.shape != expected or not np.isfinite(decoded).all():
        raise RuntimeError(f"decoded A4 q-center path must have shape {expected}")
    return decoded[1:].astype(np.float32, copy=False)


def _pixel_epe(
    ground_truth: np.ndarray, predicted: np.ndarray
) -> np.ndarray:
    truth = np.asarray(ground_truth, dtype=np.float64)
    estimate = np.asarray(predicted, dtype=np.float64)
    if truth.shape != estimate.shape or truth.ndim == 0 or truth.shape[-1] != 2:
        raise ValueError("pixel EPE requires aligned [...,2] q-center arrays")
    if not np.isfinite(truth).all() or not np.isfinite(estimate).all():
        raise ValueError("pixel EPE inputs must be finite")
    return np.linalg.norm(estimate - truth, axis=-1)


def _summarize_horizon_metrics(
    ground_truth: np.ndarray,
    predicted: np.ndarray,
    *,
    horizons: Sequence[int] = ALL_HORIZONS,
) -> dict[str, dict[str, float | int]]:
    selected = _validated_horizons(horizons)
    truth = np.asarray(ground_truth, dtype=np.float64)
    estimate = np.asarray(predicted, dtype=np.float64)
    if truth.ndim == 2:
        truth = truth[None]
        estimate = estimate[None]
    expected_tail = (len(selected), 2)
    if (
        truth.shape != estimate.shape
        or truth.ndim != 3
        or truth.shape[1:] != expected_tail
        or truth.shape[0] == 0
    ):
        raise ValueError(
            f"horizon metrics require aligned [N,{len(selected)},2] arrays"
        )
    errors = _pixel_epe(truth, estimate)
    result: dict[str, dict[str, float | int]] = {}
    for column, horizon in enumerate(selected):
        values = errors[:, column]
        result[f"H{horizon}"] = {
            "horizon": int(horizon),
            "sample_count": int(values.size),
            "mean_epe_px": float(np.mean(values)),
            "median_epe_px": float(np.median(values)),
            "p95_epe_px": float(np.quantile(values, 0.95)),
            "pck_at_4": float(np.mean(values <= 4.0)),
            "pck_at_8": float(np.mean(values <= 8.0)),
        }
    return result


def _draw_marker(
    draw: ImageDraw.ImageDraw,
    point: np.ndarray,
    *,
    fill: tuple[int, int, int],
    radius: int,
    diamond: bool,
) -> None:
    x, y = (float(value) for value in np.asarray(point, dtype=np.float64))
    if diamond:
        draw.polygon(
            ((x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)),
            fill=fill,
            outline=(255, 255, 255),
        )
    else:
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=fill,
            outline=(255, 255, 255),
            width=2,
        )


def _render_accuracy_frame(
    actual_future_frames: Sequence[Image.Image],
    ground_truth: np.ndarray,
    decoder_floor: np.ndarray,
    teacher_forced: np.ndarray,
    recursive: np.ndarray,
    *,
    task_label: str,
    decision_raw_step: int,
    window_index: int,
    window_count: int,
    horizons: Sequence[int] = DISPLAY_HORIZONS,
) -> Image.Image:
    selected = _validated_horizons(horizons)
    if len(actual_future_frames) != len(selected):
        raise ValueError("one actual future RGB frame is required per display horizon")
    truth = np.asarray(ground_truth, dtype=np.float32)
    floor_estimate = np.asarray(decoder_floor, dtype=np.float32)
    teacher_estimate = np.asarray(teacher_forced, dtype=np.float32)
    recursive_estimate = np.asarray(recursive, dtype=np.float32)
    if truth.shape != (len(selected), 2) or any(
        estimate.shape != truth.shape
        for estimate in (floor_estimate, teacher_estimate, recursive_estimate)
    ):
        raise ValueError("display q-centers must have shape [panel_count,2]")
    floor_errors = _pixel_epe(truth, floor_estimate)
    teacher_errors = _pixel_epe(truth, teacher_estimate)
    recursive_errors = _pixel_epe(truth, recursive_estimate)
    frame_sizes = {frame.size for frame in actual_future_frames}
    if len(frame_sizes) != 1:
        raise ValueError("all actual future RGB panels must share one size")
    source_width, source_height = next(iter(frame_sizes))
    panel_width, panel_height = source_width, source_height
    margin = 12
    gap = 8
    header_height = 62
    footer_height = 38
    canvas = Image.new(
        "RGB",
        (
            margin * 2 + len(selected) * panel_width + (len(selected) - 1) * gap,
            header_height + panel_height + footer_height,
        ),
        (235, 239, 241),
    )
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, header_height), fill=(25, 37, 46))
    draw.text(
        (margin, 8),
        task_label,
        fill=(255, 255, 255),
        font=_safe_font(17),
    )
    draw.text(
        (margin, 34),
        (
            f"decision raw {decision_raw_step:03d} | "
            f"window {window_index + 1:02d}/{window_count:02d}"
        ),
        fill=(194, 205, 212),
        font=_safe_font(11),
    )
    legend_x = max(margin, canvas.width - 515)
    _draw_marker(
        draw,
        np.asarray([legend_x + 5, 18], dtype=np.float32),
        fill=GROUND_TRUTH_COLOR,
        radius=5,
        diamond=False,
    )
    draw.text(
        (legend_x + 16, 10),
        "GT",
        fill=(230, 235, 238),
        font=_safe_font(10),
    )
    _draw_marker(
        draw,
        np.asarray([legend_x + 62, 18], dtype=np.float32),
        fill=DECODER_FLOOR_COLOR,
        radius=6,
        diamond=True,
    )
    draw.text(
        (legend_x + 74, 10),
        "target decode",
        fill=(230, 235, 238),
        font=_safe_font(10),
    )
    _draw_marker(
        draw,
        np.asarray([legend_x + 176, 18], dtype=np.float32),
        fill=TEACHER_FORCED_COLOR,
        radius=6,
        diamond=True,
    )
    draw.text(
        (legend_x + 188, 10),
        "teacher-forced",
        fill=(230, 235, 238),
        font=_safe_font(10),
    )
    _draw_marker(
        draw,
        np.asarray([legend_x + 300, 18], dtype=np.float32),
        fill=LEWM_COLOR,
        radius=6,
        diamond=True,
    )
    draw.text(
        (legend_x + 312, 10),
        "recursive",
        fill=(230, 235, 238),
        font=_safe_font(10),
    )

    target_steps = _horizon_frame_indices(decision_raw_step, selected)
    for index, (horizon, source_frame) in enumerate(
        zip(selected, actual_future_frames, strict=True)
    ):
        panel = source_frame.convert("RGB").copy()
        panel_draw = ImageDraw.Draw(panel)
        scale = np.asarray(
            [panel_width / source_width, panel_height / source_height],
            dtype=np.float32,
        )
        gt_point = truth[index] * scale
        floor_point = floor_estimate[index] * scale
        teacher_point = teacher_estimate[index] * scale
        recursive_point = recursive_estimate[index] * scale
        _draw_marker(
            panel_draw,
            gt_point,
            fill=GROUND_TRUTH_COLOR,
            radius=6,
            diamond=False,
        )
        _draw_marker(
            panel_draw,
            floor_point,
            fill=DECODER_FLOOR_COLOR,
            radius=6,
            diamond=True,
        )
        _draw_marker(
            panel_draw,
            teacher_point,
            fill=TEACHER_FORCED_COLOR,
            radius=6,
            diamond=True,
        )
        _draw_marker(
            panel_draw,
            recursive_point,
            fill=LEWM_COLOR,
            radius=7,
            diamond=True,
        )
        panel_draw.rectangle((0, 0, panel_width, 30), fill=(247, 249, 250))
        panel_draw.text(
            (7, 6),
            f"H{horizon} | actual RGB raw {int(target_steps[index]):03d}",
            fill=(31, 42, 50),
            font=_safe_font(11),
        )
        panel_draw.rectangle(
            (0, panel_height - 30, panel_width, panel_height),
            fill=(247, 249, 250),
        )
        panel_draw.text(
            (7, panel_height - 24),
            (
                f"EPE floor {float(floor_errors[index]):.1f} | "
                f"TF {float(teacher_errors[index]):.1f} | "
                f"rec {float(recursive_errors[index]):.1f} px"
            ),
            fill=(31, 42, 50),
            font=_safe_font(9),
        )
        x = margin + index * (panel_width + gap)
        canvas.paste(panel, (x, header_height))

    footer_y = header_height + panel_height + 10
    draw.text(
        (margin, footer_y),
        "Actual future RGB is evaluation-only. Orange is the probe floor; purple isolates one-step error; blue includes recursive drift.",
        fill=(55, 66, 75),
        font=_safe_font(11),
    )
    return canvas


def _save_gallery_gif(
    output_dir: Path,
    *,
    task: str,
    frames: Sequence[Image.Image],
    fps: float,
) -> str:
    animation = list(frames)
    if not animation:
        raise ValueError("accuracy GIF requires at least one decision frame")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be positive and finite")
    name = f"{task.replace('-', '_')}_h1_h3_h5_h8_accuracy.gif"
    duration = max(40, int(round(1000.0 / float(fps))))
    animation[0].save(
        output_dir / name,
        format="GIF",
        save_all=True,
        append_images=animation[1:],
        duration=duration,
        loop=0,
        disposal=2,
    )
    return name


def _metric_row_html(
    metrics: Mapping[str, Mapping[str, Any]], horizon: int
) -> str:
    key = f"H{int(horizon)}"
    floor = metrics["decoder_floor"][key]
    teacher = metrics["teacher_forced"][key]
    recursive = metrics["recursive"][key]
    return (
        "<tr>"
        f"<td>H{int(horizon)}</td>"
        f"<td>{float(floor['mean_epe_px']):.2f}</td>"
        f"<td>{float(teacher['mean_epe_px']):.2f}</td>"
        f"<td>{float(recursive['mean_epe_px']):.2f}</td>"
        f"<td>{float(recursive['p95_epe_px']):.2f}</td>"
        f"<td>{100.0 * float(recursive['pck_at_4']):.1f}%</td>"
        f"<td>{100.0 * float(recursive['pck_at_8']):.1f}%</td>"
        "</tr>"
    )


def _render_html(
    entries: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Render a static report with the causal and non-generation risks visible."""

    report_metadata = {} if metadata is None else dict(metadata)
    checkpoint_step = int(report_metadata.get("checkpoint_step", 10_000))
    step_label = (
        f"{checkpoint_step // 1000}k"
        if checkpoint_step % 1000 == 0
        else f"step {checkpoint_step}"
    )
    report_title = str(
        report_metadata.get(
            "report_title", f"A4 {step_label} sliding-window q-center accuracy"
        )
    )
    report_scope = str(
        report_metadata.get(
            "report_scope",
            "Reserved-test episodes use the first numeric instance assigned by the frozen original A4 split.",
        )
    )
    sections = []
    for entry in entries:
        task = str(entry["task"])
        task_label = html.escape(str(entry.get("task_label", TASK_LABELS.get(task, task))))
        gif = html.escape(str(entry["gif"]), quote=True)
        metrics = entry["metrics"]
        if not isinstance(metrics, Mapping) or tuple(metrics) != METRIC_METHODS:
            raise TypeError("entry metrics must contain floor/teacher/recursive mappings")
        rows = []
        for horizon in ALL_HORIZONS:
            key = f"H{horizon}"
            if any(key not in metrics[method] for method in METRIC_METHODS):
                raise ValueError(f"entry metrics are missing {key}")
            rows.append(_metric_row_html(metrics, horizon))
        sections.append(
            f"""
      <section>
        <div class="section-head">
          <div><h2>{task_label}</h2><p>reserved-test instance {int(entry['task_instance'])} &middot; environment seed {int(entry['environment_seed'])} &middot; {int(entry['decision_window_count'])} complete decision windows</p></div>
          <div class="scope">H1/H3/H5/H8 GIF<br>H1-H8 metrics</div>
        </div>
        <img src="{gif}" alt="{task_label} A4 latent q-center forecast accuracy">
        <table><thead><tr><th>Horizon</th><th>Target-decode floor mean</th><th>Teacher-forced mean</th><th>Recursive mean</th><th>Recursive P95</th><th>Recursive PCK@4</th><th>Recursive PCK@8</th></tr></thead>
        <tbody>{''.join(rows)}</tbody></table>
      </section>"""
        )
    seed_text = ", ".join(
        str(int(value))
        for value in report_metadata.get(
            "checkpoint_seeds", CONFIRMATION_TRAINING_SEEDS
        )
    )
    decoder_name = html.escape(
        str(report_metadata.get("decoder", "frozen train-only decoder")),
        quote=True,
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(report_title)}</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#edf1f3;color:#17212a;font-family:Inter,Arial,sans-serif;letter-spacing:0}}header{{background:#17252e;color:white;padding:24px max(20px,calc((100vw - 1020px)/2))}}h1,h2,p{{margin:0}}header p{{margin-top:8px;color:#c4cfd5;line-height:1.5}}main{{max-width:1020px;margin:auto;padding:20px 20px 60px}}.risk{{background:#fff0ee;border-left:5px solid #bd3d33;padding:16px 18px;line-height:1.55;margin-bottom:18px}}.risk strong{{display:block;font-size:18px;text-transform:uppercase;margin-bottom:5px}}.contract{{background:white;border:1px solid #cbd3d8;padding:14px 16px;line-height:1.55}}section{{padding:28px 0 34px;border-bottom:1px solid #c8d0d5}}.section-head{{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:12px}}.section-head p{{margin-top:5px;color:#66737d;font-size:13px}}.scope{{text-align:right;color:#52616b;font-size:12px;line-height:1.4}}img{{display:block;width:100%;height:auto;border:1px solid #bbc5cb;background:#747c82}}table{{width:100%;border-collapse:collapse;background:white;margin-top:14px}}th,td{{padding:9px 10px;text-align:right;border-bottom:1px solid #dde3e6;font-variant-numeric:tabular-nums}}th:first-child,td:first-child{{text-align:left}}th{{color:#60707a;font-size:12px}}@media(max-width:700px){{.section-head{{display:block}}.scope{{text-align:left;margin-top:8px}}table{{font-size:12px}}th,td{{padding:7px 5px}}}}
</style></head><body><header><h1>{html.escape(report_title)}</h1><p>Seven-task fixed-latest ensemble seeds {html.escape(seed_text)}; fixed checkpoint endpoint {checkpoint_step} ({html.escape(step_label)}). {html.escape(report_scope)}</p></header><main>
<div class="risk"><strong>{HTML_RISK_HEADLINE}</strong>Actual future RGB and GT q-center are evaluation-only overlays and are never model inputs. Orange target-decode error is the probe/representation floor, purple teacher-forced error isolates one-step dynamics, and blue recursive error additionally includes rollout drift.</div>
<div class="contract"><b>Causal input contract.</b> Each forecast uses only three current/history observations, two recorded history action chunks, the eight actual future action chunks, and task identity. Absolute coordinates are anchored by the frozen current-frame FCPG correction derived from the current observation. No future observation or future GT is read before latent rollout. Only H1-H8 are evaluated or displayed.<br><b>Decoder:</b> {decoder_name}</div>
{''.join(sections)}</main></body></html>"""


def _prepare_task_episode(
    task: str,
    *,
    contract: Mapping[str, Any],
    task_instance: int,
    encoder: Any,
    max_decision_windows: int | None,
    max_raw_steps: int,
    post_success_steps: int,
) -> dict[str, Any]:
    collection = contract["collection"]
    task_index = TASKS.index(task)
    environment_seed = _environment_seed(
        contract, task=task, task_instance=task_instance
    )
    episode = _safe_collect_full_episode(
        task,
        task_instance=task_instance,
        environment_seed=environment_seed,
        width=int(collection["width"]),
        height=int(collection["height"]),
        benchmark_seed=int(collection["benchmark_seed"]),
        max_raw_steps=max_raw_steps,
        post_success_steps=post_success_steps,
    )
    starts = _decision_window_starts(
        len(episode["actions"]),
        max_decision_windows=max_decision_windows,
    )
    if not starts:
        raise RuntimeError(
            f"{task} episode has no complete three-frame-history plus H8 window"
        )
    prepared_windows = []
    for decision_step in starts:
        alignment, causal_inputs = _build_causal_model_inputs(
            episode["records"],
            episode["actions"],
            decision_raw_step=decision_step,
            task_index=task_index,
        )
        history_embedding = _encode_history(causal_inputs, encoder=encoder)
        target_embedding = _encode_evaluation_future(
            episode["records"], alignment, encoder=encoder
        )
        current_q_center = _current_observed_q_center(
            causal_inputs,
            encoder=encoder,
            task=task,
            image_height=int(collection["height"]),
        )
        prepared_windows.append(
            {
                "alignment": alignment,
                "history_embedding": history_embedding,
                "target_embedding": target_embedding,
                "history_action": np.array(
                    causal_inputs.history_action, dtype=np.float32, copy=True
                ),
                "future_action": np.array(
                    causal_inputs.future_action, dtype=np.float32, copy=True
                ),
                "task_index": task_index,
                "current_q_center_display": current_q_center,
                "latent_sum": np.zeros(
                    (HORIZON, STATE_SLOT_COUNT, LATENT_DIM), dtype=np.float64
                ),
                "teacher_forced_latent_sum": np.zeros(
                    (HORIZON, STATE_SLOT_COUNT, LATENT_DIM), dtype=np.float64
                ),
            }
        )
    return {
        "task": task,
        "task_instance": int(task_instance),
        "environment_seed": int(environment_seed),
        "frames": episode["frames"],
        "q_center_display": np.asarray(
            episode["q_center_display"], dtype=np.float32
        ),
        "frame_count": len(episode["frames"]),
        "raw_action_count": len(episode["actions"]),
        "first_success_step": episode["first_success_step"],
        "stable_success_step": episode["stable_success_step"],
        "stop_reason": episode["stop_reason"],
        "windows": prepared_windows,
    }


def _add_endpoint_predictions(
    prepared_tasks: Sequence[dict[str, Any]],
    endpoint: ConfirmationEndpoint,
    *,
    base_checkpoint: Mapping[str, Any],
    device: torch.device,
) -> None:
    model, _ = initialize_a4_model_from_anchor_base(
        base_checkpoint, seed=endpoint.seed
    )
    model.load_state_dict(endpoint.model_state, strict=True)
    model.to(device).eval()
    try:
        for task_entry in prepared_tasks:
            for window in task_entry["windows"]:
                window["latent_sum"] += _rollout_encoded_window(
                    model,
                    history_embedding=window["history_embedding"],
                    history_action=window["history_action"],
                    future_action=window["future_action"],
                    task_index=window["task_index"],
                    device=device,
                )
                window["teacher_forced_latent_sum"] += (
                    _teacher_forced_encoded_window(
                        model,
                        history_embedding=window["history_embedding"],
                        history_action=window["history_action"],
                        future_action=window["future_action"],
                        target_embedding=window["target_embedding"],
                        task_index=window["task_index"],
                        device=device,
                    )
                )
    finally:
        del model
        _clear_runtime_memory(device)


def _finish_task_entry(
    task_entry: dict[str, Any],
    *,
    endpoint_count: int,
    decoder: Mapping[str, np.ndarray],
    output_dir: Path,
    image_height: int,
    fps: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task = str(task_entry["task"])
    window_count = len(task_entry["windows"])
    animation = []
    truths = []
    decoder_floors = []
    teacher_forced_predictions = []
    recursive_predictions = []
    window_manifest = []
    for window_index, window in enumerate(task_entry["windows"]):
        alignment: WindowAlignment = window["alignment"]
        recursive_latent = np.asarray(window["latent_sum"], dtype=np.float64) / float(
            endpoint_count
        )
        teacher_latent = np.asarray(
            window["teacher_forced_latent_sum"], dtype=np.float64
        ) / float(endpoint_count)
        decoder_floor = _decode_window_q_centers(
            window["target_embedding"],
            window["history_embedding"],
            window["current_q_center_display"],
            task=task,
            image_height=image_height,
            decoder=decoder,
        )
        teacher_forced = _decode_window_q_centers(
            teacher_latent.astype(np.float32),
            window["history_embedding"],
            window["current_q_center_display"],
            task=task,
            image_height=image_height,
            decoder=decoder,
        )
        recursive = _decode_window_q_centers(
            recursive_latent.astype(np.float32),
            window["history_embedding"],
            window["current_q_center_display"],
            task=task,
            image_height=image_height,
            decoder=decoder,
        )
        frame_indices = np.asarray(
            alignment.future_frame_indices, dtype=np.int64
        )
        ground_truth = task_entry["q_center_display"][frame_indices]
        floor_errors = _pixel_epe(ground_truth, decoder_floor)
        teacher_errors = _pixel_epe(ground_truth, teacher_forced)
        recursive_errors = _pixel_epe(ground_truth, recursive)
        truths.append(ground_truth)
        decoder_floors.append(decoder_floor)
        teacher_forced_predictions.append(teacher_forced)
        recursive_predictions.append(recursive)
        display_columns = [horizon - 1 for horizon in DISPLAY_HORIZONS]
        animation.append(
            _render_accuracy_frame(
                [
                    task_entry["frames"][alignment.future_frame_indices[column]]
                    for column in display_columns
                ],
                ground_truth[display_columns],
                decoder_floor[display_columns],
                teacher_forced[display_columns],
                recursive[display_columns],
                task_label=TASK_LABELS[task],
                decision_raw_step=alignment.decision_raw_step,
                window_index=window_index,
                window_count=window_count,
            )
        )
        window_manifest.append(
            {
                "decision_raw_step": alignment.decision_raw_step,
                "history_frame_indices": list(alignment.history_frame_indices),
                "future_action_starts": list(alignment.future_action_starts),
                "future_frame_indices": list(alignment.future_frame_indices),
                "epe_px": {
                    method: {
                        f"H{horizon}": float(values[horizon - 1])
                        for horizon in ALL_HORIZONS
                    }
                    for method, values in (
                        ("decoder_floor", floor_errors),
                        ("teacher_forced", teacher_errors),
                        ("recursive", recursive_errors),
                    )
                },
            }
        )
    truth_array = np.stack(truths, axis=0)
    metrics = {
        "decoder_floor": _summarize_horizon_metrics(
            truth_array, np.stack(decoder_floors, axis=0)
        ),
        "teacher_forced": _summarize_horizon_metrics(
            truth_array, np.stack(teacher_forced_predictions, axis=0)
        ),
        "recursive": _summarize_horizon_metrics(
            truth_array, np.stack(recursive_predictions, axis=0)
        ),
    }
    gif = _save_gallery_gif(output_dir, task=task, frames=animation, fps=fps)
    html_entry = {
        "task": task,
        "task_label": TASK_LABELS[task],
        "task_instance": task_entry["task_instance"],
        "environment_seed": task_entry["environment_seed"],
        "decision_window_count": window_count,
        "gif": gif,
        "metrics": metrics,
    }
    manifest_entry = {
        **html_entry,
        "frame_count": task_entry["frame_count"],
        "raw_action_count": task_entry["raw_action_count"],
        "first_success_step": task_entry["first_success_step"],
        "stable_success_step": task_entry["stable_success_step"],
        "stop_reason": task_entry["stop_reason"],
        "windows": window_manifest,
    }
    return html_entry, manifest_entry


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not np.isfinite(args.fps) or args.fps <= 0.0:
        raise ValueError("fps must be positive and finite")
    device = _runtime_device(args.device)
    contract = _safe_read_json(args.config)
    validate_a4_contract(contract)
    selected_tasks = tuple(args.task) if args.task else TASKS
    if len(set(selected_tasks)) != len(selected_tasks):
        raise ValueError("--task values must be unique")
    instances = _first_test_instances(contract)
    checkpoint_paths = tuple(args.a4_checkpoints or DEFAULT_A4_CHECKPOINTS)
    endpoints = _load_confirmation_endpoints(
        checkpoint_paths, expected_step=int(args.expected_step)
    )
    decoder = _load_train_only_decoder(args.decoder.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder = load_frozen_a4_state_encoder(
        args.base_checkpoint.expanduser().resolve(),
        args.fcpg_checkpoint.expanduser().resolve(),
        device=device,
    )
    prepared_tasks = []
    try:
        for task in selected_tasks:
            prepared = _prepare_task_episode(
                task,
                contract=contract,
                task_instance=instances[task],
                encoder=encoder,
                max_decision_windows=args.max_decision_windows,
                max_raw_steps=int(args.max_raw_steps),
                post_success_steps=int(args.post_success_steps),
            )
            prepared_tasks.append(prepared)
            print(
                json.dumps(
                    {
                        "task": task,
                        "test_instance": instances[task],
                        "decision_windows": len(prepared["windows"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        del encoder
        _clear_runtime_memory(device)

    base_checkpoint = _safe_torch_load(args.base_checkpoint)
    for endpoint in endpoints:
        _add_endpoint_predictions(
            prepared_tasks,
            endpoint,
            base_checkpoint=base_checkpoint,
            device=device,
        )
        print(
            json.dumps(
                {"checkpoint_seed": endpoint.seed, "step": int(args.expected_step)},
                sort_keys=True,
            ),
            flush=True,
        )
    del base_checkpoint

    html_entries = []
    manifest_entries = []
    image_height = int(contract["collection"]["height"])
    for prepared in prepared_tasks:
        html_entry, manifest_entry = _finish_task_entry(
            prepared,
            endpoint_count=len(endpoints),
            decoder=decoder,
            output_dir=output_dir,
            image_height=image_height,
            fps=float(args.fps),
        )
        html_entries.append(html_entry)
        manifest_entries.append(manifest_entry)

    metadata = {
        "checkpoint_seeds": [endpoint.seed for endpoint in endpoints],
        "checkpoint_step": int(args.expected_step),
        "decoder": str(args.decoder.expanduser().resolve()),
    }
    if int(args.expected_step) != 10_000:
        metadata.update(
            {
                "report_title": (
                    f"A4c v4 {int(args.expected_step) // 1000}k development "
                    "sliding-window q-center sanity check"
                ),
                "report_scope": (
                    "Visual sanity check only; this page is not a sealed test "
                    "result or a 10k endpoint claim."
                ),
            }
        )
    (output_dir / "index.html").write_text(
        _render_html(html_entries, metadata), encoding="utf-8"
    )
    metrics_payload = {
        "schema_version": 1,
        "protocol": (
            PROTOCOL_NAME
            if int(args.expected_step) == 10_000
            else f"a4c_v4_step{int(args.expected_step)}_development_visual_sanity_v1"
        ),
        "split": "test",
        "horizons": list(ALL_HORIZONS),
        "tasks": {
            entry["task"]: entry["metrics"] for entry in html_entries
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(
            metrics_payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    input_contract = {
        "model_input_fields": [
            "history_observations",
            "history_action",
            "future_action",
            "task_index",
        ],
        "history_observation_fields": list(CAUSAL_OBSERVATION_FIELDS),
        "future_rgb": "evaluation_only_for_rendering",
        "future_gt_q_center": "evaluation_only_for_metrics_and_overlay",
        "current_q_center_anchor": (
            "frozen_current_frame_fcpg_correction_from_current_observation"
        ),
    }
    manifest = {
        "schema_version": 1,
        "protocol": (
            PROTOCOL_NAME
            if int(args.expected_step) == 10_000
            else f"a4c_v4_step{int(args.expected_step)}_development_visual_sanity_v1"
        ),
        "split": "test",
        "test_instance_selection": (
            "first_numeric_instance_per_task_in_frozen_original_a4_test_split"
        ),
        "display_horizons": list(DISPLAY_HORIZONS),
        "metric_horizons": list(ALL_HORIZONS),
        "device": str(device),
        "checkpoints": [
            {
                "path": str(endpoint.path),
                "seed": endpoint.seed,
                "step": int(args.expected_step),
            }
            for endpoint in endpoints
        ],
        "decoder": {
            "path": str(args.decoder.expanduser().resolve()),
            "fit_split": "train",
            "role": "latent_forecast_to_q_center_probe_not_rgb_generation",
        },
        "model_input_contract": input_contract,
        "entries": manifest_entries,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output_dir / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
