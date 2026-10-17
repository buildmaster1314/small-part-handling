"""Pure contracts and audits for the Counterfactual v2 development dataset.

This module deliberately contains no simulator, model, or runtime imports.  It
defines the fail-closed boundary between deployable scorer inputs and labels
that are available only after a counterfactual branch has been executed.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, ClassVar

import numpy as np


SCHEMA_VERSION = 2
PROTOCOL = "fcpg_lewm_counterfactual_v2_development"
COLLECTOR_PROTOCOL = "fcpg_lewm_counterfactual_v2_collection"
SEED = 8192

TASK_NAMES = (
    "pick-place-v3",
    "button-press-v3",
    "reach-v3",
    "push-v3",
)
MODEL_NAMES = ("nominal", "fcpg")
GROUNDING_NAMES = ("nominal", "fcpg")
VIEW_NAMES = ("clean", "rotation5", "translation4")
PHASE_NAMES = ("early", "mid", "late")
PHASE_STEPS = (10, 25, 35)

NUM_VIEWS = 3
NUM_MODELS = 2
NUM_GROUNDINGS = 2
HISTORY_SIZE = 3
PREDICTION_STEPS = 5
LATENT_DIM = 192
NUM_KEYPOINTS = 3
IMAGE_SIZE = 224
GOAL_VISUAL_DIM = 48
GOAL_VISUAL_INPUT_DIM = 768
GOAL_VISUAL_PROJECTION_SHA256 = (
    "a7c785c4e4a3c4e7736af913686318d857f46760bdbbb60233f243b44a96f365"
)
PROPRIO_DIM = 4
ACTION_DIM = 4
RAW_ACTION_STEPS = 25
RAW_ACTIONS_PER_TRANSITION = 5
TRANSITION_ACTION_DIM = 20
FULL_CANDIDATE_COUNT = 8
SMOKE_CANDIDATE_COUNT = 4

FULL_CANDIDATE_NAMES = (
    "expert",
    "policy",
    "expert_noise_0.05",
    "expert_noise_0.15",
    "expert_noise_0.3",
    "same_task_shuffled",
    "random_0",
    "random_1",
)
SMOKE_CANDIDATE_NAMES = (
    "expert",
    "policy",
    "expert_noise_0.15",
    "same_task_shuffled",
)

DEPLOYMENT_KEYS = frozenset(
    {
        "task_index",
        "history_rgb",
        "history_proprio",
        "history_keypoints_xyz",
        "history_q_nominal",
        "history_valid_nominal",
        "history_q_corrected",
        "history_valid_grounded",
        "history_action_raw",
        "candidate_action_raw",
        "z_history",
        "z_pred_future",
        "goal_visual",
        "task_onehot",
    }
)
ACTION_ONLY_PROBE_KEYS = frozenset(
    {"candidate_action_raw", "goal_visual", "task_onehot"}
)
DIRECT_PROBE_KEYS = frozenset(
    ACTION_ONLY_PROBE_KEYS | {"z_history", "history_action_raw"}
)
PREDICTED_PROBE_KEYS = frozenset(DIRECT_PROBE_KEYS | {"z_pred_future"})
ORACLE_PROBE_KEYS = frozenset(DIRECT_PROBE_KEYS | {"oracle_future_latent"})
PROBE_NAMES = ("action_only", "direct", "predicted", "oracle")

# This is the universe of fields that a deployable scorer may receive, not the
# much larger DeploymentInputs storage pool.  Each concrete probe is further
# restricted to its exact key set above.
SCORER_INPUT_KEYS = PREDICTED_PROBE_KEYS
NON_DEPLOYMENT_METADATA_KEYS = frozenset(
    {"episode_id", "env_seed", "phase_index", "snapshot_step"}
)
OFFLINE_LABEL_KEYS = frozenset(
    {
        "actual_return",
        "actual_progress",
        "actual_success",
        "oracle_future_latent",
        "history_q_true",
        "history_valid_true",
        "goal_world",
        "goal_pixel",
        "goal_valid",
    }
)
AUDIT_ARRAY_KEYS = frozenset(
    NON_DEPLOYMENT_METADATA_KEYS
    | {
        "camera_intrinsic",
        "true_extrinsic",
        "nominal_extrinsic",
        "endpoint_rgb_dynamic_range",
        "endpoint_rgb_changed_pixels",
        "endpoint_rgb_sha256",
    }
)

INDEX_ROW_KEYS = frozenset(
    {
        "sample_index",
        "task_index",
        "task_name",
        "episode_id",
        "env_seed",
        "phase_index",
        "phase_name",
        "snapshot_step",
        "split",
        "physical_episode_sha256",
        "snapshot_sha256",
        "history_physical_sha256_by_view",
        "history_rng_sha256_by_view",
        "view_names",
        "candidate_names",
        "candidate_permutation",
        "candidate_permutation_sha256",
        "candidate_action_sha256",
        "candidate_action_set_sha256",
        "candidate_label_sha256",
        "oracle_future_latent_sha256_by_view",
        "endpoint_rgb_sha256_by_view",
        "branch_receipt_sha256",
        "branch_execution_count",
        "endpoint_physical_sha256_by_view",
        "endpoint_rng_sha256_by_view",
    }
)

PROVENANCE_KEYS = frozenset(
    {
        "schema_version",
        "protocol",
        "scope",
        "config",
        "config_sha256",
        "git",
        "versions",
        "inputs",
        "model_state_before",
        "model_state_after",
        "no_training_verified",
        "goal_visual_projection",
        "sample_count",
        "physical_branch_count",
        "upstream_non_overlap",
        "result_contract_sha256",
    }
)

MODEL_STATE_KEYS = frozenset(
    {
        "nominal",
        "fcpg",
        "action_only_policy",
        "fcpg_corrector",
        "oracle_target_encoder",
        "oracle_vision_backbone",
    }
)

# These names are forbidden recursively at a scorer/deployment boundary.  The
# predicted future latent is legal, while an actual/oracle future is not.
PRIVILEGED_INPUT_KEYS = frozenset(
    OFFLINE_LABEL_KEYS
    | NON_DEPLOYMENT_METADATA_KEYS
    | {
        "audit_arrays",
        "offline_labels",
        "q_true",
        "valid_true",
        "q_oracle",
        "reward",
        "rewards",
        "return",
        "returns",
        "progress",
        "success",
        "info",
        "actual_future_latent",
        "future_rgb",
        "future_observation",
        "future_success_frame",
        "goal_record",
        "goal_info",
        "goal_world_coordinate",
        "goal_pixel_coordinate",
        "oracle_target",
        "goal_oracle_target",
    }
)


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = set(value)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise ValueError(f"{label} keys mismatch: missing={missing}, extra={extra}")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        raise TypeError("arrays must be represented by sha256_array records")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("canonical JSON cannot contain non-finite floats")
    return value


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def goal_visual_projection_seed(seed: int) -> int:
    payload = f"{COLLECTOR_PROTOCOL}|{int(seed)}|goal_visual".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def assert_no_privileged_inputs(value: Any, label: str = "deployment inputs") -> None:
    """Recursively reject label/oracle fields at a deployment boundary."""

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                name = str(key).lower()
                if name in PRIVILEGED_INPUT_KEYS:
                    raise ValueError(f"{label} contains privileged field at {path}.{key}")
                visit(child, f"{path}.{key}")
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
            return
        if hasattr(item, "__dataclass_fields__"):
            for field in fields(item):
                name = field.name.lower()
                if name in PRIVILEGED_INPUT_KEYS:
                    raise ValueError(
                        f"{label} contains privileged field at {path}.{field.name}"
                    )
                visit(getattr(item, field.name), f"{path}.{field.name}")

    visit(value, label)


@dataclass(frozen=True)
class _ExactArrayRecord:
    KEY_SET: ClassVar[frozenset[str]]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Any:
        mapping = _mapping(value, cls.__name__)
        _exact_keys(mapping, cls.KEY_SET, cls.__name__)
        return cls(**{name: mapping[name] for name in cls.KEY_SET})

    def as_mapping(self) -> dict[str, np.ndarray]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class DeploymentInputs(_ExactArrayRecord):
    """Safe data pool from which a minimal scorer view must be built.

    Passing this record directly to a scorer is forbidden: it contains several
    alternative model/grounding/view cells used by the paired evaluation.
    """

    KEY_SET: ClassVar[frozenset[str]] = DEPLOYMENT_KEYS

    task_index: np.ndarray
    history_rgb: np.ndarray
    history_proprio: np.ndarray
    history_keypoints_xyz: np.ndarray
    history_q_nominal: np.ndarray
    history_valid_nominal: np.ndarray
    history_q_corrected: np.ndarray
    history_valid_grounded: np.ndarray
    history_action_raw: np.ndarray
    candidate_action_raw: np.ndarray
    z_history: np.ndarray
    z_pred_future: np.ndarray
    goal_visual: np.ndarray
    task_onehot: np.ndarray

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DeploymentInputs":
        assert_no_privileged_inputs(value)
        return super().from_mapping(value)


@dataclass(frozen=True)
class OfflineLabels(_ExactArrayRecord):
    """Post-branch labels; this object must never enter a deployable scorer."""

    KEY_SET: ClassVar[frozenset[str]] = OFFLINE_LABEL_KEYS

    actual_return: np.ndarray
    actual_progress: np.ndarray
    actual_success: np.ndarray
    oracle_future_latent: np.ndarray
    history_q_true: np.ndarray
    history_valid_true: np.ndarray
    goal_world: np.ndarray
    goal_pixel: np.ndarray
    goal_valid: np.ndarray


@dataclass(frozen=True)
class AuditArrays(_ExactArrayRecord):
    """Non-deployable metadata and evidence retained only for audits."""

    KEY_SET: ClassVar[frozenset[str]] = AUDIT_ARRAY_KEYS

    episode_id: np.ndarray
    env_seed: np.ndarray
    phase_index: np.ndarray
    snapshot_step: np.ndarray
    camera_intrinsic: np.ndarray
    true_extrinsic: np.ndarray
    nominal_extrinsic: np.ndarray
    endpoint_rgb_dynamic_range: np.ndarray
    endpoint_rgb_changed_pixels: np.ndarray
    endpoint_rgb_sha256: np.ndarray


@dataclass(frozen=True)
class ActionOnlyProbeInputs(_ExactArrayRecord):
    """Minimal deployable action-only baseline features for one camera view."""

    KEY_SET: ClassVar[frozenset[str]] = ACTION_ONLY_PROBE_KEYS

    candidate_action_raw: np.ndarray
    goal_visual: np.ndarray
    task_onehot: np.ndarray


@dataclass(frozen=True)
class DirectProbeInputs(_ExactArrayRecord):
    """DirectRanker features for exactly one model/grounding/view cell."""

    KEY_SET: ClassVar[frozenset[str]] = DIRECT_PROBE_KEYS

    z_history: np.ndarray
    history_action_raw: np.ndarray
    candidate_action_raw: np.ndarray
    goal_visual: np.ndarray
    task_onehot: np.ndarray


@dataclass(frozen=True)
class PredictedProbeInputs(_ExactArrayRecord):
    """DirectRanker features plus the matching predicted-future latent."""

    KEY_SET: ClassVar[frozenset[str]] = PREDICTED_PROBE_KEYS

    z_history: np.ndarray
    history_action_raw: np.ndarray
    candidate_action_raw: np.ndarray
    goal_visual: np.ndarray
    task_onehot: np.ndarray
    z_pred_future: np.ndarray


@dataclass(frozen=True)
class OracleProbeInputs(_ExactArrayRecord):
    """Offline-only upper-bound view using the matching actual future latent."""

    KEY_SET: ClassVar[frozenset[str]] = ORACLE_PROBE_KEYS
    OFFLINE_ONLY: ClassVar[bool] = True

    z_history: np.ndarray
    history_action_raw: np.ndarray
    candidate_action_raw: np.ndarray
    goal_visual: np.ndarray
    task_onehot: np.ndarray
    oracle_future_latent: np.ndarray


@dataclass(frozen=True)
class ArrayAxes:
    sample_count: int
    candidate_count: int
    view_count: int = NUM_VIEWS
    model_count: int = NUM_MODELS
    grounding_count: int = NUM_GROUNDINGS
    history_size: int = HISTORY_SIZE
    prediction_steps: int = PREDICTION_STEPS
    latent_dim: int = LATENT_DIM
    keypoint_count: int = NUM_KEYPOINTS


def _validate_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any] | type[np.generic],
    finite: bool = False,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    expected_dtype = np.dtype(dtype)
    if value.dtype != expected_dtype:
        raise TypeError(f"{name} dtype must be {expected_dtype}, got {value.dtype}")
    if value.shape != shape:
        raise ValueError(f"{name} shape must be {shape}, got {value.shape}")
    if finite and not bool(np.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")
    return value


def _resolve_probe_selector(
    selector: str | int,
    names: tuple[str, ...],
    *,
    axis: str,
) -> int:
    """Resolve one frozen protocol selector without accepting aliases."""

    if isinstance(selector, str):
        if selector not in names:
            raise ValueError(
                f"unknown {axis} selector {selector!r}; expected one of {list(names)}"
            )
        return names.index(selector)
    if isinstance(selector, (int, np.integer)) and not isinstance(selector, (bool, np.bool_)):
        index = int(selector)
        if 0 <= index < len(names):
            return index
        raise IndexError(
            f"{axis} selector index {index} is outside [0, {len(names)})"
        )
    raise TypeError(f"{axis} selector must be an exact name or integer index")


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    """Detach a probe feature from the storage pool and make it immutable."""

    result = np.array(value, copy=True, order="C")
    result.setflags(write=False)
    return result


def _validate_probe_source(deployment: DeploymentInputs) -> tuple[int, int]:
    """Validate all arrays that can cross one of the probe boundaries."""

    if not isinstance(deployment.task_index, np.ndarray) or deployment.task_index.ndim != 1:
        raise ValueError("task_index must have shape [N] before building probe inputs")
    n = int(deployment.task_index.shape[0])
    candidate = deployment.candidate_action_raw
    if not isinstance(candidate, np.ndarray) or candidate.ndim != 4:
        raise ValueError("candidate_action_raw must have shape [N,C,25,4]")
    c = int(candidate.shape[1])
    if n <= 0 or c <= 0:
        raise ValueError("probe source must contain at least one sample and candidate")

    float_specs = {
        "history_action_raw": (
            deployment.history_action_raw,
            (n, HISTORY_SIZE - 1, RAW_ACTIONS_PER_TRANSITION, ACTION_DIM),
        ),
        "candidate_action_raw": (
            deployment.candidate_action_raw,
            (n, c, RAW_ACTION_STEPS, ACTION_DIM),
        ),
        "z_history": (
            deployment.z_history,
            (
                n,
                NUM_MODELS,
                NUM_GROUNDINGS,
                NUM_VIEWS,
                HISTORY_SIZE,
                LATENT_DIM,
            ),
        ),
        "z_pred_future": (
            deployment.z_pred_future,
            (
                n,
                NUM_MODELS,
                NUM_GROUNDINGS,
                NUM_VIEWS,
                c,
                PREDICTION_STEPS,
                LATENT_DIM,
            ),
        ),
        "goal_visual": (
            deployment.goal_visual,
            (n, NUM_VIEWS, GOAL_VISUAL_DIM),
        ),
        "task_onehot": (
            deployment.task_onehot,
            (n, len(TASK_NAMES)),
        ),
    }
    for name, (value, shape) in float_specs.items():
        _validate_array(value, name=name, shape=shape, dtype=np.float32, finite=True)
    return n, c


def build_probe_inputs(
    probe: str,
    deployment: DeploymentInputs | Mapping[str, Any],
    *,
    model: str | int,
    grounding: str | int,
    view: str | int,
    labels: OfflineLabels | Mapping[str, Any] | None = None,
) -> (
    ActionOnlyProbeInputs
    | DirectProbeInputs
    | PredictedProbeInputs
    | OracleProbeInputs
):
    """Build an exact, detached feature view for one Counterfactual v2 probe.

    ``model``, ``grounding``, and ``view`` are frozen protocol selectors.  The
    first two are intentionally unused by ``action_only`` but are still
    validated so one matrix iterator can construct all four probe families.
    Only ``oracle`` accepts :class:`OfflineLabels`; its returned record is
    explicitly marked ``OFFLINE_ONLY``.
    """

    if probe not in PROBE_NAMES:
        raise ValueError(f"unknown probe {probe!r}; expected one of {list(PROBE_NAMES)}")
    if isinstance(deployment, Mapping):
        deployment = DeploymentInputs.from_mapping(deployment)
    elif not isinstance(deployment, DeploymentInputs):
        raise TypeError("deployment must be DeploymentInputs or its exact mapping")

    model_index = _resolve_probe_selector(model, MODEL_NAMES, axis="model")
    grounding_index = _resolve_probe_selector(
        grounding, GROUNDING_NAMES, axis="grounding"
    )
    view_index = _resolve_probe_selector(view, VIEW_NAMES, axis="view")
    _validate_probe_source(deployment)

    if probe != "oracle" and labels is not None:
        raise ValueError("offline labels may only be supplied to the oracle probe")

    shared = {
        "candidate_action_raw": _readonly_copy(deployment.candidate_action_raw),
        "goal_visual": _readonly_copy(deployment.goal_visual[:, view_index]),
        "task_onehot": _readonly_copy(deployment.task_onehot),
    }
    if probe == "action_only":
        result: (
            ActionOnlyProbeInputs
            | DirectProbeInputs
            | PredictedProbeInputs
            | OracleProbeInputs
        ) = ActionOnlyProbeInputs(**shared)
        assert_no_privileged_inputs(result)
        return result

    direct = {
        "z_history": _readonly_copy(
            deployment.z_history[
                :, model_index, grounding_index, view_index, :, :
            ]
        ),
        "history_action_raw": _readonly_copy(deployment.history_action_raw),
        **shared,
    }
    if probe == "direct":
        result = DirectProbeInputs(**direct)
        assert_no_privileged_inputs(result)
        return result
    if probe == "predicted":
        result = PredictedProbeInputs(
            **direct,
            z_pred_future=_readonly_copy(
                deployment.z_pred_future[
                    :, model_index, grounding_index, view_index, :, :, :
                ]
            ),
        )
        assert_no_privileged_inputs(result)
        return result

    if labels is None:
        raise ValueError("oracle probe requires offline labels")
    if isinstance(labels, Mapping):
        labels = OfflineLabels.from_mapping(labels)
    elif not isinstance(labels, OfflineLabels):
        raise TypeError("labels must be OfflineLabels or its exact mapping")
    _validate_array(
        labels.oracle_future_latent,
        name="oracle_future_latent",
        shape=(
            deployment.task_index.shape[0],
            NUM_VIEWS,
            deployment.candidate_action_raw.shape[1],
            PREDICTION_STEPS,
            LATENT_DIM,
        ),
        dtype=np.float32,
        finite=True,
    )
    return OracleProbeInputs(
        **direct,
        oracle_future_latent=_readonly_copy(
            labels.oracle_future_latent[:, view_index, :, :, :]
        ),
    )


def validate_deployable_probe_inputs(
    value: ActionOnlyProbeInputs | DirectProbeInputs | PredictedProbeInputs,
) -> None:
    """Fail closed at a deployable scorer boundary.

    A scorer must consume one detached minimal probe view, never the full
    collection pool and never the offline-only oracle view.
    """

    if isinstance(value, (DeploymentInputs, OracleProbeInputs)):
        raise TypeError("deployable scorers cannot consume pooled or oracle inputs")
    if not isinstance(
        value, (ActionOnlyProbeInputs, DirectProbeInputs, PredictedProbeInputs)
    ):
        raise TypeError("deployable scorer input must be a concrete probe view")
    assert_no_privileged_inputs(value, label="deployable scorer input")
    arrays = value.as_mapping()
    candidate = arrays["candidate_action_raw"]
    if not isinstance(candidate, np.ndarray) or candidate.ndim != 4:
        raise ValueError("candidate_action_raw must have shape [N,C,25,4]")
    sample_count, candidate_count = candidate.shape[:2]
    common_shapes = {
        "candidate_action_raw": (
            sample_count,
            candidate_count,
            RAW_ACTION_STEPS,
            ACTION_DIM,
        ),
        "goal_visual": (sample_count, GOAL_VISUAL_DIM),
        "task_onehot": (sample_count, len(TASK_NAMES)),
    }
    if isinstance(value, (DirectProbeInputs, PredictedProbeInputs)):
        common_shapes.update(
            {
                "z_history": (sample_count, HISTORY_SIZE, LATENT_DIM),
                "history_action_raw": (
                    sample_count,
                    HISTORY_SIZE - 1,
                    RAW_ACTIONS_PER_TRANSITION,
                    ACTION_DIM,
                ),
            }
        )
    if isinstance(value, PredictedProbeInputs):
        common_shapes["z_pred_future"] = (
            sample_count,
            candidate_count,
            PREDICTION_STEPS,
            LATENT_DIM,
        )
    for name, array in arrays.items():
        if not isinstance(array, np.ndarray):
            raise TypeError(f"deployable scorer field {name} must be an array")
        _validate_array(
            array,
            name=f"deployable scorer field {name}",
            shape=common_shapes[name],
            dtype=np.float32,
            finite=True,
        )
        if array.flags.writeable:
            raise ValueError(
                f"deployable scorer field {name} must be detached and read-only"
            )


def build_g2_probe_pair(
    deployment: DeploymentInputs | Mapping[str, Any],
    *,
    model: str | int,
    grounding: str | int,
    view: str | int,
) -> tuple[DirectProbeInputs, PredictedProbeInputs]:
    """Build the matched DirectRanker/predicted-latent G2 comparison."""

    direct = build_probe_inputs(
        "direct", deployment, model=model, grounding=grounding, view=view
    )
    predicted = build_probe_inputs(
        "predicted", deployment, model=model, grounding=grounding, view=view
    )
    assert isinstance(direct, DirectProbeInputs)
    assert isinstance(predicted, PredictedProbeInputs)
    for name in DIRECT_PROBE_KEYS:
        if not np.array_equal(direct.as_mapping()[name], predicted.as_mapping()[name]):
            raise RuntimeError(f"G2 control variable changed unexpectedly: {name}")
    validate_deployable_probe_inputs(direct)
    validate_deployable_probe_inputs(predicted)
    return direct, predicted


def build_g3_probe_pair(
    deployment: DeploymentInputs | Mapping[str, Any],
    *,
    model: str | int,
    view: str | int,
) -> tuple[PredictedProbeInputs, PredictedProbeInputs]:
    """Build the matched nominal/FCPG-grounding G3 comparison.

    Model, view, action bank, history actions, goal context, and task context
    are fixed mechanically.  Only the selected grounding cell may change.
    """

    nominal = build_probe_inputs(
        "predicted", deployment, model=model, grounding="nominal", view=view
    )
    corrected = build_probe_inputs(
        "predicted", deployment, model=model, grounding="fcpg", view=view
    )
    assert isinstance(nominal, PredictedProbeInputs)
    assert isinstance(corrected, PredictedProbeInputs)
    controlled = PREDICTED_PROBE_KEYS - {"z_history", "z_pred_future"}
    for name in controlled:
        if not np.array_equal(nominal.as_mapping()[name], corrected.as_mapping()[name]):
            raise RuntimeError(f"G3 control variable changed unexpectedly: {name}")
    validate_deployable_probe_inputs(nominal)
    validate_deployable_probe_inputs(corrected)
    return nominal, corrected


def grounded_validity(
    q_corrected: np.ndarray,
    valid_nominal: np.ndarray,
    *,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Compute deployable validity from corrected points and nominal geometry."""

    corrected = np.asarray(q_corrected)
    nominal = np.asarray(valid_nominal)
    if corrected.ndim != 5 or corrected.shape[-2:] != (NUM_KEYPOINTS, 2):
        raise ValueError("q_corrected must have shape [N,V,T,K,2]")
    if nominal.shape != (corrected.shape[0], corrected.shape[2], NUM_KEYPOINTS):
        raise ValueError("valid_nominal must have shape [N,T,K]")
    if nominal.dtype != np.bool_:
        raise TypeError("valid_nominal must be boolean")
    finite = np.isfinite(corrected).all(axis=-1)
    inside = (
        (corrected[..., 0] >= 0.0)
        & (corrected[..., 0] <= float(image_size - 1))
        & (corrected[..., 1] >= 0.0)
        & (corrected[..., 1] <= float(image_size - 1))
    )
    return nominal[:, None, :, :] & finite & inside


def _project_points_for_audit(
    points_xyz: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project one point set with the same pinhole convention as collection."""

    points = np.asarray(points_xyz, dtype=np.float64)
    homogeneous = np.concatenate(
        [points, np.ones((*points.shape[:-1], 1), dtype=np.float64)], axis=-1
    )
    camera = np.einsum("ij,...j->...i", np.asarray(extrinsic, dtype=np.float64), homogeneous)
    xyz = camera[..., :3]
    uvw = np.einsum("ij,...j->...i", np.asarray(intrinsic, dtype=np.float64), xyz)
    pixels = uvw[..., :2] / np.maximum(uvw[..., 2:3], 1e-8)
    valid = (
        (xyz[..., 2] > 1e-8)
        & (pixels[..., 0] >= 0.0)
        & (pixels[..., 0] < IMAGE_SIZE)
        & (pixels[..., 1] >= 0.0)
        & (pixels[..., 1] < IMAGE_SIZE)
    )
    return pixels.astype(np.float32), valid


def audit_geometric_reprojection(
    deployment: DeploymentInputs,
    labels: OfflineLabels,
    audit: AuditArrays,
    *,
    atol: float = 2e-3,
) -> dict[str, Any]:
    """Recompute every stored geometric projection from XYZ and calibration.

    This is intentionally independent of learned correction.  It binds the
    nominal projection, per-view oracle projection, validity masks, and goal
    labels to the calibration arrays that are persisted beside the features.
    """

    n = int(deployment.task_index.shape[0])
    intrinsic = np.asarray(audit.camera_intrinsic)
    if not np.array_equal(
        intrinsic,
        np.broadcast_to(intrinsic[:, :1], intrinsic.shape),
    ):
        raise ValueError("camera intrinsic K differs across camera views")
    expected_intrinsic_tail = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    if not np.allclose(intrinsic[..., 2, :], expected_intrinsic_tail, atol=atol, rtol=0.0):
        raise ValueError("camera intrinsic K has an invalid homogeneous row")

    expected_nominal = np.empty_like(deployment.history_q_nominal)
    expected_valid_nominal = np.empty_like(deployment.history_valid_nominal)
    expected_true = np.empty_like(labels.history_q_true)
    expected_valid_true = np.empty_like(labels.history_valid_true)
    expected_goal = np.empty_like(labels.goal_pixel)
    expected_goal_valid = np.empty_like(labels.goal_valid)
    for sample in range(n):
        expected_nominal[sample], expected_valid_nominal[sample] = (
            _project_points_for_audit(
                deployment.history_keypoints_xyz[sample],
                intrinsic[sample, 0],
                audit.nominal_extrinsic[sample],
            )
        )
        for view in range(NUM_VIEWS):
            expected_true[sample, view], expected_valid_true[sample, view] = (
                _project_points_for_audit(
                    deployment.history_keypoints_xyz[sample],
                    intrinsic[sample, view],
                    audit.true_extrinsic[sample, view],
                )
            )
            goal, goal_valid = _project_points_for_audit(
                labels.goal_world[sample : sample + 1],
                intrinsic[sample, view],
                audit.true_extrinsic[sample, view],
            )
            expected_goal[sample, view] = goal[0]
            expected_goal_valid[sample, view] = goal_valid[0]

    comparisons = (
        ("history_q_nominal", deployment.history_q_nominal, expected_nominal),
        ("history_q_true", labels.history_q_true, expected_true),
        ("goal_pixel", labels.goal_pixel, expected_goal),
    )
    max_error: dict[str, float] = {}
    for name, stored, expected in comparisons:
        error = float(np.max(np.abs(stored.astype(np.float64) - expected.astype(np.float64))))
        max_error[name] = error
        if not np.allclose(stored, expected, atol=atol, rtol=0.0):
            raise ValueError(f"{name} disagrees with geometric reprojection (max error {error:.6g})")
    if not np.array_equal(deployment.history_valid_nominal, expected_valid_nominal):
        raise ValueError("history_valid_nominal disagrees with geometric reprojection")
    if not np.array_equal(labels.history_valid_true, expected_valid_true):
        raise ValueError("history_valid_true disagrees with geometric reprojection")
    if not np.array_equal(labels.goal_valid, expected_goal_valid):
        raise ValueError("goal_valid disagrees with geometric reprojection")
    return {
        "passed": True,
        "intrinsics_identical_across_views": True,
        "atol_pixels": float(atol),
        "max_abs_error_pixels": max_error,
    }


def audit_history_rgb(history_rgb: np.ndarray) -> dict[str, Any]:
    """Reject black/constant frames and camera conditions with identical pixels."""

    rgb = np.asarray(history_rgb)
    expected_tail = (
        NUM_VIEWS,
        HISTORY_SIZE,
        IMAGE_SIZE,
        IMAGE_SIZE,
        3,
    )
    if rgb.ndim != 6 or rgb.shape[1:] != expected_tail or rgb.dtype != np.uint8:
        raise ValueError("history_rgb must have canonical [N,V,T,H,W,3] uint8 shape")
    frame_min = rgb.min(axis=(-3, -2, -1)).astype(np.int16)
    frame_max = rgb.max(axis=(-3, -2, -1)).astype(np.int16)
    dynamic_range = frame_max - frame_min
    constant = np.argwhere(dynamic_range <= 0)
    if constant.size:
        sample, view, step = (int(value) for value in constant[0])
        raise ValueError(
            "history_rgb contains a constant frame at "
            f"sample={sample}, view={view}, history_step={step}"
        )

    changed_pixels = np.count_nonzero(
        np.any(rgb[:, 1:] != rgb[:, :1], axis=-1),
        axis=(-2, -1),
    )
    identical = np.argwhere(changed_pixels <= 0)
    if identical.size:
        sample, perturbed_view, step = (int(value) for value in identical[0])
        raise ValueError(
            "history_rgb perturbed view is pixel-identical to clean at "
            f"sample={sample}, view={perturbed_view + 1}, history_step={step}"
        )
    return {
        "passed": True,
        "frame_count": int(np.prod(dynamic_range.shape)),
        "min_frame_dynamic_range": int(dynamic_range.min()),
        "min_perturbed_changed_pixels": int(changed_pixels.min()),
        "global_min": int(rgb.min()),
        "global_max": int(rgb.max()),
    }


def audit_endpoint_rgb(audit: AuditArrays) -> dict[str, Any]:
    """Validate persisted endpoint-render evidence without storing full frames."""

    dynamic_range = audit.endpoint_rgb_dynamic_range
    changed_pixels = audit.endpoint_rgb_changed_pixels
    digests = audit.endpoint_rgb_sha256
    constant = np.argwhere(dynamic_range <= 0)
    if constant.size:
        sample, view, candidate, step = (int(value) for value in constant[0])
        raise ValueError(
            "endpoint RGB contains a constant frame at "
            f"sample={sample}, view={view}, candidate={candidate}, step={step}"
        )
    impossible_range = np.argwhere(dynamic_range > 255)
    if impossible_range.size:
        sample, view, candidate, step = (
            int(value) for value in impossible_range[0]
        )
        raise ValueError(
            "endpoint RGB dynamic range exceeds uint8 image range at "
            f"sample={sample}, view={view}, candidate={candidate}, step={step}"
        )
    identical = np.argwhere(changed_pixels <= 0)
    if identical.size:
        sample, perturbed_view, candidate, step = (
            int(value) for value in identical[0]
        )
        raise ValueError(
            "endpoint RGB perturbed view is pixel-identical to clean at "
            f"sample={sample}, view={perturbed_view + 1}, "
            f"candidate={candidate}, step={step}"
        )
    impossible_changed = np.argwhere(changed_pixels > IMAGE_SIZE * IMAGE_SIZE)
    if impossible_changed.size:
        sample, perturbed_view, candidate, step = (
            int(value) for value in impossible_changed[0]
        )
        raise ValueError(
            "endpoint RGB changed-pixel count exceeds image area at "
            f"sample={sample}, view={perturbed_view + 1}, "
            f"candidate={candidate}, step={step}"
        )
    empty_digest = np.argwhere(np.all(digests == 0, axis=-1))
    if empty_digest.size:
        sample, view, candidate, step = (
            int(value) for value in empty_digest[0]
        )
        raise ValueError(
            "endpoint RGB SHA-256 evidence is empty at "
            f"sample={sample}, view={view}, candidate={candidate}, step={step}"
        )
    digest_identical = np.argwhere(
        np.all(digests[:, 1:] == digests[:, :1], axis=-1)
    )
    if digest_identical.size:
        sample, perturbed_view, candidate, step = (
            int(value) for value in digest_identical[0]
        )
        raise ValueError(
            "endpoint RGB digest for perturbed view equals clean at "
            f"sample={sample}, view={perturbed_view + 1}, "
            f"candidate={candidate}, step={step}"
        )
    return {
        "passed": True,
        "frame_count": int(np.prod(dynamic_range.shape)),
        "min_frame_dynamic_range": int(dynamic_range.min()),
        "min_perturbed_changed_pixels": int(changed_pixels.min()),
        "sha256_evidence_complete": True,
    }


def validate_array_bundle(
    deployment: DeploymentInputs | Mapping[str, Any],
    labels: OfflineLabels | Mapping[str, Any],
    audit: AuditArrays | Mapping[str, Any],
    *,
    candidate_count: int,
) -> ArrayAxes:
    """Validate exact keys, shapes, dtypes, finiteness, and semantic parity."""

    if isinstance(deployment, Mapping):
        deployment = DeploymentInputs.from_mapping(deployment)
    if isinstance(labels, Mapping):
        labels = OfflineLabels.from_mapping(labels)
    if isinstance(audit, Mapping):
        audit = AuditArrays.from_mapping(audit)
    assert_no_privileged_inputs(deployment)

    n = int(np.asarray(deployment.task_index).shape[0])
    c = int(candidate_count)
    if n <= 0 or c <= 0:
        raise ValueError("sample_count and candidate_count must be positive")
    axes = ArrayAxes(sample_count=n, candidate_count=c)
    d = deployment
    l = labels
    a = audit

    _validate_array(
        d.task_index, name="task_index", shape=(n,), dtype=np.uint8
    )
    audit_integer_specs = {
        "episode_id": ((n,), np.int32),
        "env_seed": ((n,), np.int64),
        "phase_index": ((n,), np.uint8),
        "snapshot_step": ((n,), np.int16),
    }
    for name, (shape, dtype) in audit_integer_specs.items():
        _validate_array(getattr(a, name), name=name, shape=shape, dtype=dtype)
    _validate_array(
        d.history_rgb,
        name="history_rgb",
        shape=(n, NUM_VIEWS, HISTORY_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=np.uint8,
    )
    audit_history_rgb(d.history_rgb)
    deployment_float_specs = {
        "history_proprio": (n, HISTORY_SIZE, PROPRIO_DIM),
        "history_keypoints_xyz": (n, HISTORY_SIZE, NUM_KEYPOINTS, 3),
        "history_q_nominal": (n, HISTORY_SIZE, NUM_KEYPOINTS, 2),
        "history_q_corrected": (
            n,
            NUM_VIEWS,
            HISTORY_SIZE,
            NUM_KEYPOINTS,
            2,
        ),
        "history_action_raw": (
            n,
            HISTORY_SIZE - 1,
            RAW_ACTIONS_PER_TRANSITION,
            ACTION_DIM,
        ),
        "candidate_action_raw": (n, c, RAW_ACTION_STEPS, ACTION_DIM),
        "z_history": (
            n,
            NUM_MODELS,
            NUM_GROUNDINGS,
            NUM_VIEWS,
            HISTORY_SIZE,
            LATENT_DIM,
        ),
        "z_pred_future": (
            n,
            NUM_MODELS,
            NUM_GROUNDINGS,
            NUM_VIEWS,
            c,
            PREDICTION_STEPS,
            LATENT_DIM,
        ),
        "goal_visual": (n, NUM_VIEWS, GOAL_VISUAL_DIM),
        "task_onehot": (n, len(TASK_NAMES)),
    }
    for name, shape in deployment_float_specs.items():
        _validate_array(
            getattr(d, name), name=name, shape=shape, dtype=np.float32, finite=True
        )
    _validate_array(
        d.history_valid_nominal,
        name="history_valid_nominal",
        shape=(n, HISTORY_SIZE, NUM_KEYPOINTS),
        dtype=np.bool_,
    )
    _validate_array(
        d.history_valid_grounded,
        name="history_valid_grounded",
        shape=(n, NUM_VIEWS, HISTORY_SIZE, NUM_KEYPOINTS),
        dtype=np.bool_,
    )

    camera_invariant_shapes = {
        "actual_return": (n, c, PREDICTION_STEPS),
        "actual_progress": (n, c, PREDICTION_STEPS),
    }
    for name, shape in camera_invariant_shapes.items():
        _validate_array(
            getattr(l, name), name=name, shape=shape, dtype=np.float32, finite=True
        )
    _validate_array(
        l.actual_success,
        name="actual_success",
        shape=(n, c, PREDICTION_STEPS),
        dtype=np.bool_,
    )
    label_float_specs = {
        "oracle_future_latent": (
            n,
            NUM_VIEWS,
            c,
            PREDICTION_STEPS,
            LATENT_DIM,
        ),
        "history_q_true": (
            n,
            NUM_VIEWS,
            HISTORY_SIZE,
            NUM_KEYPOINTS,
            2,
        ),
        "goal_world": (n, 3),
        "goal_pixel": (n, NUM_VIEWS, 2),
    }
    for name, shape in label_float_specs.items():
        _validate_array(
            getattr(l, name), name=name, shape=shape, dtype=np.float32, finite=True
        )
    _validate_array(
        l.history_valid_true,
        name="history_valid_true",
        shape=(n, NUM_VIEWS, HISTORY_SIZE, NUM_KEYPOINTS),
        dtype=np.bool_,
    )
    _validate_array(
        l.goal_valid,
        name="goal_valid",
        shape=(n, NUM_VIEWS),
        dtype=np.bool_,
    )
    audit_float_specs = {
        "camera_intrinsic": (n, NUM_VIEWS, 3, 3),
        "true_extrinsic": (n, NUM_VIEWS, 4, 4),
        "nominal_extrinsic": (n, 4, 4),
    }
    for name, shape in audit_float_specs.items():
        _validate_array(
            getattr(a, name), name=name, shape=shape, dtype=np.float32, finite=True
        )
    _validate_array(
        a.endpoint_rgb_dynamic_range,
        name="endpoint_rgb_dynamic_range",
        shape=(n, NUM_VIEWS, c, PREDICTION_STEPS),
        dtype=np.uint16,
    )
    _validate_array(
        a.endpoint_rgb_changed_pixels,
        name="endpoint_rgb_changed_pixels",
        shape=(n, NUM_VIEWS - 1, c, PREDICTION_STEPS),
        dtype=np.uint32,
    )
    _validate_array(
        a.endpoint_rgb_sha256,
        name="endpoint_rgb_sha256",
        shape=(n, NUM_VIEWS, c, PREDICTION_STEPS, 32),
        dtype=np.uint8,
    )
    audit_endpoint_rgb(a)

    expected_valid = grounded_validity(
        d.history_q_corrected, d.history_valid_nominal, image_size=IMAGE_SIZE
    )
    if not np.array_equal(expected_valid, d.history_valid_grounded):
        raise ValueError("history_valid_grounded disagrees with corrected-point bounds")
    if np.any(a.phase_index >= len(PHASE_STEPS)):
        raise ValueError("phase_index is outside the fixed phase set")
    if not np.array_equal(a.snapshot_step, np.take(PHASE_STEPS, a.phase_index)):
        raise ValueError("snapshot_step disagrees with fixed phase_index mapping")
    if np.any(d.task_index >= len(TASK_NAMES)):
        raise ValueError("task_index is outside the canonical four-task suite")
    expected_onehot = np.eye(len(TASK_NAMES), dtype=np.float32)[d.task_index]
    if not np.array_equal(d.task_onehot, expected_onehot):
        raise ValueError("task_onehot disagrees with task_index")
    audit_geometric_reprojection(d, l, a)
    return axes


CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "protocol",
        "scope",
        "seed",
        "device",
        "inputs",
        "oracle_vision",
        "evaluation",
        "conditions",
        "environment",
        "split",
        "goal_context",
        "smoke",
        "output",
    }
)


def _validate_pinned_input(value: object, label: str) -> None:
    entry = _mapping(value, label)
    _exact_keys(entry, frozenset({"path", "sha256"}), label)
    if not isinstance(entry["path"], str) or not entry["path"]:
        raise ValueError(f"{label}.path must be non-empty")
    digest = entry["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{label}.sha256 must be lowercase SHA-256")


def _expected_conditions() -> dict[str, dict[str, Any]]:
    return {
        "clean": {
            "translation_world": [0.0, 0.0, 0.0],
            "rotation_axis_world": [0.0, 1.0, 0.0],
            "rotation_degrees": 0.0,
        },
        "rotation5": {
            "translation_world": [0.0, 0.0, 0.0],
            "rotation_axis_world": [0.0, 1.0, 0.0],
            "rotation_degrees": 5.0,
        },
        "translation4": {
            "translation_world": [0.04, 0.0, 0.0],
            "rotation_axis_world": [0.0, 1.0, 0.0],
            "rotation_degrees": 0.0,
        },
    }


def _axis_angle_rotation_matrix(axis: Sequence[float], degrees: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        if abs(float(degrees)) <= 1e-12:
            return np.eye(3, dtype=np.float64)
        raise ValueError("rotation axis must be non-zero for a non-zero angle")
    x, y, z = vector / norm
    angle = np.deg2rad(float(degrees))
    skew = np.asarray(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * (skew @ skew)
    )


def expected_camera_condition_extrinsics(
    nominal_extrinsic: np.ndarray,
    conditions: Mapping[str, Any],
) -> np.ndarray:
    """Reconstruct fixed condition extrinsics independently from the runtime."""

    nominal = np.asarray(nominal_extrinsic, dtype=np.float64)
    if nominal.ndim != 3 or nominal.shape[1:] != (4, 4):
        raise ValueError("nominal_extrinsic must have shape [N,4,4]")
    if tuple(conditions) != VIEW_NAMES or dict(conditions) != _expected_conditions():
        raise ValueError("camera conditions changed from clean/rotation5/translation4")
    homogeneous_row = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if not np.allclose(nominal[:, 3], homogeneous_row, atol=1e-6, rtol=0.0):
        raise ValueError("nominal_extrinsic has an invalid homogeneous row")

    mjcam_to_cv = np.diag([1.0, -1.0, -1.0, 1.0])
    expected = np.empty((nominal.shape[0], NUM_VIEWS, 4, 4), dtype=np.float64)
    for sample in range(nominal.shape[0]):
        world_to_mjcam = mjcam_to_cv @ nominal[sample]
        world_to_camera_rotation = world_to_mjcam[:3, :3]
        camera_to_world_rotation = world_to_camera_rotation.T
        if not np.allclose(
            camera_to_world_rotation.T @ camera_to_world_rotation,
            np.eye(3),
            atol=2e-5,
            rtol=0.0,
        ) or not np.isclose(
            np.linalg.det(camera_to_world_rotation), 1.0, atol=2e-5, rtol=0.0
        ):
            raise ValueError("nominal_extrinsic rotation is not a proper rotation")
        camera_position = (
            -camera_to_world_rotation @ world_to_mjcam[:3, 3]
        )
        for view, name in enumerate(VIEW_NAMES):
            condition = _mapping(conditions[name], f"conditions.{name}")
            delta_rotation = _axis_angle_rotation_matrix(
                condition["rotation_axis_world"],
                float(condition["rotation_degrees"]),
            )
            position = camera_position + np.asarray(
                condition["translation_world"], dtype=np.float64
            )
            camera_to_world = delta_rotation @ camera_to_world_rotation
            world_to_condition = np.eye(4, dtype=np.float64)
            world_to_condition[:3, :3] = camera_to_world.T
            world_to_condition[:3, 3] = -camera_to_world.T @ position
            expected[sample, view] = mjcam_to_cv @ world_to_condition
    return expected


def audit_camera_condition_extrinsics(
    audit: AuditArrays,
    conditions: Mapping[str, Any],
    *,
    atol: float = 5e-5,
) -> dict[str, Any]:
    """Bind stored true camera matrices to the frozen condition definitions."""

    expected = expected_camera_condition_extrinsics(
        audit.nominal_extrinsic, conditions
    )
    actual = np.asarray(audit.true_extrinsic, dtype=np.float64)
    if actual.shape != expected.shape:
        raise ValueError(
            f"true_extrinsic shape must be {expected.shape}, got {actual.shape}"
        )
    max_errors = np.max(np.abs(actual - expected), axis=(0, 2, 3))
    if not np.allclose(actual, expected, atol=atol, rtol=0.0):
        worst_view = int(np.argmax(max_errors))
        raise ValueError(
            "true_extrinsic disagrees with frozen camera condition "
            f"{VIEW_NAMES[worst_view]!r} (max error {max_errors[worst_view]:.6g})"
        )
    return {
        "passed": True,
        "condition_order": list(VIEW_NAMES),
        "atol_matrix": float(atol),
        "max_abs_error_by_view": {
            name: float(max_errors[index])
            for index, name in enumerate(VIEW_NAMES)
        },
    }


def _validate_common_config(config: Mapping[str, Any]) -> None:
    _exact_keys(config, CONFIG_KEYS, "config")
    if config["schema_version"] != SCHEMA_VERSION or config["protocol"] != PROTOCOL:
        raise ValueError("unexpected Counterfactual v2 schema or protocol")
    if config["seed"] != SEED or config["device"] != "cuda:0":
        raise ValueError("Counterfactual v2 seed/device contract changed")

    scope = _mapping(config["scope"], "scope")
    expected_scope = {
        "classification": "external_development",
        "upstream_training_allowed": False,
        "online_cem_allowed": False,
        "formal_claim_allowed": False,
    }
    if dict(scope) != expected_scope:
        raise ValueError("scope must remain external development with no upstream training/CEM")

    inputs = _mapping(config["inputs"], "inputs")
    _exact_keys(
        inputs,
        frozenset(
            {
                "lewm_checkpoints",
                "action_only_policy",
                "fcpg_corrector",
                "oracle_target_encoder",
            }
        ),
        "inputs",
    )
    checkpoints = _mapping(inputs["lewm_checkpoints"], "lewm_checkpoints")
    _exact_keys(checkpoints, frozenset(MODEL_NAMES), "lewm_checkpoints")
    for name in MODEL_NAMES:
        value = checkpoints[name]
        _validate_pinned_input(value, f"lewm_checkpoints.{name}")
    for name in ("action_only_policy", "fcpg_corrector", "oracle_target_encoder"):
        _validate_pinned_input(inputs[name], f"inputs.{name}")

    vision = _mapping(config["oracle_vision"], "oracle_vision")
    expected_vision = {
        "model_name": "vit_base_patch16_224.augreg_in1k",
        "selected_layer": 11,
        "usage": "offline_oracle_future_and_current_rgb_goal_context",
    }
    if dict(vision) != expected_vision:
        raise ValueError("oracle vision contract changed")

    if dict(_mapping(config["conditions"], "conditions")) != _expected_conditions():
        raise ValueError("camera conditions changed from clean/rotation5/translation4")

    environment = _mapping(config["environment"], "environment")
    _exact_keys(
        environment,
        frozenset(
            {
                "camera",
                "width",
                "height",
                "max_expert_steps",
                "benchmark_seed",
                "env_seed_base",
                "task_index_offset",
            }
        ),
        "environment",
    )
    fixed_environment = {
        "camera": "corner",
        "width": IMAGE_SIZE,
        "height": IMAGE_SIZE,
        "max_expert_steps": 120,
        "benchmark_seed": SEED,
        "task_index_offset": 0,
    }
    for name, expected in fixed_environment.items():
        if environment[name] != expected:
            raise ValueError(f"environment.{name} must be {expected!r}")
    if not isinstance(environment["env_seed_base"], int) or environment["env_seed_base"] < 0:
        raise ValueError("environment.env_seed_base must be a non-negative integer")

    split = _mapping(config["split"], "split")
    expected_split = {
        "unit": "physical_episode",
        "train_count_per_task": 7,
        "validation_count_per_task": 3,
        "assignment": "sha256_sorted",
        "upstream_non_overlap_required": True,
    }
    if dict(split) != expected_split:
        raise ValueError("physical-episode split contract changed")

    goal = _mapping(config["goal_context"], "goal_context")
    expected_goal = {
        "source": "current_rgb_and_task_id",
        "future_success_frame_allowed": False,
        "privileged_coordinates_allowed": False,
        "visual_dim": GOAL_VISUAL_DIM,
        "visual_input_dim": GOAL_VISUAL_INPUT_DIM,
        "projection_sha256": GOAL_VISUAL_PROJECTION_SHA256,
    }
    if dict(goal) != expected_goal:
        raise ValueError("goal context must be current-RGB/task-only and non-privileged")

    smoke = _mapping(config["smoke"], "smoke")
    expected_smoke = {
        "episodes_per_task": 1,
        "candidate_names": list(SMOKE_CANDIDATE_NAMES),
        "output_suffix": "_smoke",
    }
    if dict(smoke) != expected_smoke:
        raise ValueError("smoke override contract changed")

    output = _mapping(config["output"], "output")
    expected_output = {
        "directory": "outputs/lewm_counterfactual_v2/development_seed8192",
        "write_mode": "exclusive_atomic_directory",
    }
    if dict(output) not in (
        expected_output,
        {
            **expected_output,
            "directory": expected_output["directory"] + "_smoke",
        },
    ):
        raise ValueError("Counterfactual v2 output contract changed")


def _expected_evaluation(*, smoke: bool) -> dict[str, Any]:
    return {
        "tasks": list(TASK_NAMES),
        "episodes_per_task": 1 if smoke else 10,
        "phase_steps": list(PHASE_STEPS),
        "phase_names": list(PHASE_NAMES),
        "history_size": HISTORY_SIZE,
        "frame_skip": RAW_ACTIONS_PER_TRANSITION,
        "raw_action_steps": RAW_ACTION_STEPS,
        "action_dim": ACTION_DIM,
        "transitions": PREDICTION_STEPS,
        "transition_action_dim": TRANSITION_ACTION_DIM,
        "horizons": list(range(1, PREDICTION_STEPS + 1)),
        "noise_scales": [0.15] if smoke else [0.05, 0.15, 0.3],
        "random_count": 0 if smoke else 2,
        "candidate_names": list(
            SMOKE_CANDIDATE_NAMES if smoke else FULL_CANDIDATE_NAMES
        ),
        "progress_info_key": "in_place_reward",
        "snapshot_include_process_rng": True,
    }


def validate_full_config(config: Mapping[str, Any]) -> None:
    _validate_common_config(config)
    evaluation = _mapping(config["evaluation"], "evaluation")
    if dict(evaluation) != _expected_evaluation(smoke=False):
        raise ValueError("full Counterfactual v2 evaluation contract changed")
    if config["output"]["directory"].endswith("_smoke"):
        raise ValueError("full config cannot use the smoke output directory")


def validate_smoke_config(config: Mapping[str, Any]) -> None:
    _validate_common_config(config)
    evaluation = _mapping(config["evaluation"], "evaluation")
    if dict(evaluation) != _expected_evaluation(smoke=True):
        raise ValueError("smoke Counterfactual v2 evaluation contract changed")
    if config["output"]["directory"] != (
        "outputs/lewm_counterfactual_v2/development_seed8192_smoke"
    ):
        raise ValueError("smoke output directory suffix is missing")


def smoke_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the only allowed bounded smoke override to a valid full config."""

    validate_full_config(config)
    result = copy.deepcopy(dict(config))
    result["evaluation"] = _expected_evaluation(smoke=True)
    result["output"]["directory"] += str(result["smoke"]["output_suffix"])
    validate_smoke_config(result)
    return result


def physical_episode_sha256(task_index: int, episode_id: int, env_seed: int) -> str:
    return canonical_sha256(
        {
            "task_index": int(task_index),
            "episode_id": int(episode_id),
            "env_seed": int(env_seed),
        }
    )


def canonical_row_order(
    deployment: DeploymentInputs,
    audit: AuditArrays,
) -> np.ndarray:
    """Return the deterministic semantic row order, independent of collection order."""

    return np.lexsort(
        (
            audit.snapshot_step.astype(np.int64),
            audit.phase_index.astype(np.int64),
            audit.env_seed,
            audit.episode_id.astype(np.int64),
            deployment.task_index.astype(np.int64),
        )
    ).astype(np.int64)


def assign_physical_episode_splits(
    task_index: np.ndarray,
    episode_id: np.ndarray,
    env_seed: np.ndarray,
    *,
    train_count_per_task: int = 7,
    validation_count_per_task: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Assign complete physical episodes by task using sorted identity hashes."""

    task = np.asarray(task_index)
    episode = np.asarray(episode_id)
    seed = np.asarray(env_seed)
    if task.ndim != 1 or episode.shape != task.shape or seed.shape != task.shape:
        raise ValueError("physical episode identity arrays must be aligned vectors")
    assignments = np.empty(task.shape, dtype="<U10")
    records: list[dict[str, Any]] = []
    for task_value in range(len(TASK_NAMES)):
        rows = np.flatnonzero(task == task_value)
        if not len(rows):
            raise ValueError(f"task {task_value} has no physical episodes")
        identities = {
            (int(episode[index]), int(seed[index])) for index in rows.tolist()
        }
        expected = train_count_per_task + validation_count_per_task
        if len(identities) != expected:
            raise ValueError(
                f"task {task_value} must contain {expected} physical episodes, "
                f"got {len(identities)}"
            )
        ranked = sorted(
            (
                physical_episode_sha256(task_value, episode_value, seed_value),
                episode_value,
                seed_value,
            )
            for episode_value, seed_value in identities
        )
        train = {(episode_value, seed_value) for _, episode_value, seed_value in ranked[:train_count_per_task]}
        for digest, episode_value, seed_value in ranked:
            split = "train" if (episode_value, seed_value) in train else "validation"
            mask = (
                (task == task_value)
                & (episode == episode_value)
                & (seed == seed_value)
            )
            assignments[mask] = split
            records.append(
                {
                    "task_index": task_value,
                    "episode_id": episode_value,
                    "env_seed": seed_value,
                    "physical_episode_sha256": digest,
                    "split": split,
                }
            )
    return assignments, {
        "unit": "physical_episode",
        "assignment": "sha256_sorted",
        "records": sorted(
            records,
            key=lambda item: (
                item["task_index"],
                item["physical_episode_sha256"],
            ),
        ),
    }


def audit_episode_groups(
    deployment: DeploymentInputs,
    audit: AuditArrays,
    *,
    smoke: bool,
) -> dict[str, Any]:
    """Close phase coverage and physical-episode split isolation."""

    groups: dict[tuple[int, int, int], list[int]] = {}
    for index in range(len(deployment.task_index)):
        key = (
            int(deployment.task_index[index]),
            int(audit.episode_id[index]),
            int(audit.env_seed[index]),
        )
        groups.setdefault(key, []).append(index)
    for key, indices in groups.items():
        phases = sorted(int(audit.phase_index[index]) for index in indices)
        steps = sorted(int(audit.snapshot_step[index]) for index in indices)
        if phases != list(range(len(PHASE_STEPS))) or steps != sorted(PHASE_STEPS):
            raise ValueError(f"physical episode {key} lacks exact fixed phase coverage")
    expected_episodes = 1 if smoke else 10
    per_task = {
        TASK_NAMES[task]: sum(1 for key in groups if key[0] == task)
        for task in range(len(TASK_NAMES))
    }
    if any(count != expected_episodes for count in per_task.values()):
        raise ValueError(f"physical episode counts per task mismatch: {per_task}")
    if smoke:
        return {
            "passed": True,
            "g0_eligible": False,
            "mode": "smoke",
            "physical_episode_count": len(groups),
            "episodes_per_task": per_task,
            "split": None,
        }
    assignments, split = assign_physical_episode_splits(
        deployment.task_index,
        audit.episode_id,
        audit.env_seed,
    )
    for indices in groups.values():
        if len(set(assignments[indices].tolist())) != 1:
            raise RuntimeError("one physical episode straddles train and validation")
    return {
        "passed": True,
        "g0_eligible": True,
        "mode": "full",
        "physical_episode_count": len(groups),
        "episodes_per_task": per_task,
        "split": split,
    }


def _ranking_label_summary(
    values: np.ndarray,
    task_index: np.ndarray,
    phase_index: np.ndarray,
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Describe candidate-order signal without treating legitimate ties as errors."""

    numeric = np.asarray(values, dtype=np.float64)
    if numeric.ndim != 3 or numeric.shape[1] < 2:
        raise ValueError("ranking labels must have shape [N,C,P] with C >= 2")
    sample_count, candidate_count, horizon_count = numeric.shape
    pair_left, pair_right = np.triu_indices(candidate_count, k=1)
    candidate_spread = np.max(numeric, axis=1) - np.min(numeric, axis=1)
    informative = candidate_spread > tolerance
    pairwise_ties = np.isclose(
        numeric[:, pair_left, :],
        numeric[:, pair_right, :],
        atol=tolerance,
        rtol=0.0,
    )

    by_task_phase: dict[str, dict[str, Any]] = {}
    for task, task_name in enumerate(TASK_NAMES):
        for phase, phase_name in enumerate(PHASE_NAMES):
            selected = (task_index == task) & (phase_index == phase)
            selected_count = int(np.count_nonzero(selected))
            selected_info = informative[selected]
            selected_ties = pairwise_ties[selected]
            by_task_phase[f"{task_name}:{phase_name}"] = {
                "sample_count": selected_count,
                "informative_snapshot_count": int(
                    np.count_nonzero(np.any(selected_info, axis=1))
                ),
                "informative_snapshot_horizon_count": int(
                    np.count_nonzero(selected_info)
                ),
                "snapshot_horizon_count": int(selected_count * horizon_count),
                "pairwise_tie_fraction": (
                    float(np.mean(selected_ties)) if selected_ties.size else None
                ),
                "max_candidate_spread": (
                    float(np.max(candidate_spread[selected]))
                    if selected_count
                    else None
                ),
            }

    informative_snapshot_count = int(
        np.count_nonzero(np.any(informative, axis=1))
    )
    return {
        "tolerance": float(tolerance),
        "informative": bool(np.any(informative)),
        "informative_snapshot_count": informative_snapshot_count,
        "snapshot_count": int(sample_count),
        "informative_snapshot_horizon_count": int(np.count_nonzero(informative)),
        "snapshot_horizon_count": int(sample_count * horizon_count),
        "pairwise_tie_fraction": float(np.mean(pairwise_ties)),
        "max_candidate_spread": float(np.max(candidate_spread)),
        "by_task_phase": by_task_phase,
    }


def audit_ranking_informativeness(
    deployment: DeploymentInputs,
    labels: OfflineLabels,
    audit: AuditArrays,
) -> dict[str, Any]:
    """Audit whether the persisted branches form a meaningful ranking dataset.

    Duplicate action chunks invalidate the candidate intervention and fail
    closed.  Tied returns/progress/success are valid scientific outcomes, so
    they are reported as evaluability diagnostics rather than collection
    failures.
    """

    actions = np.asarray(deployment.candidate_action_raw)
    if actions.ndim != 4 or actions.shape[1] < 2:
        raise ValueError("candidate actions must have shape [N,C,25,4] with C >= 2")
    sample_count, candidate_count = actions.shape[:2]
    if audit.phase_index.shape != (sample_count,):
        raise ValueError("phase_index disagrees with candidate-action sample count")
    if deployment.task_index.shape != (sample_count,):
        raise ValueError("task_index disagrees with candidate-action sample count")

    unique_counts: list[int] = []
    pairwise_distances: list[float] = []
    pair_left, pair_right = np.triu_indices(candidate_count, k=1)
    flattened = actions.astype(np.float64, copy=False).reshape(sample_count, candidate_count, -1)
    for sample in range(sample_count):
        digests = [sha256_array(actions[sample, slot]) for slot in range(candidate_count)]
        unique_count = len(set(digests))
        unique_counts.append(unique_count)
        if unique_count != candidate_count:
            raise ValueError(
                f"sample {sample} contains duplicate candidate action chunks "
                f"({unique_count}/{candidate_count} unique)"
            )
        differences = flattened[sample, pair_left] - flattened[sample, pair_right]
        pairwise_distances.extend(np.linalg.norm(differences, axis=1).tolist())

    label_summaries = {
        "actual_return": _ranking_label_summary(
            labels.actual_return,
            deployment.task_index,
            audit.phase_index,
            tolerance=1e-8,
        ),
        "actual_progress": _ranking_label_summary(
            labels.actual_progress,
            deployment.task_index,
            audit.phase_index,
            tolerance=1e-8,
        ),
        "actual_success": _ranking_label_summary(
            labels.actual_success,
            deployment.task_index,
            audit.phase_index,
            tolerance=0.0,
        ),
    }
    return {
        "passed": True,
        "candidate_actions_unique": True,
        "candidate_count": int(candidate_count),
        "min_unique_candidate_count": int(min(unique_counts)),
        "max_unique_candidate_count": int(max(unique_counts)),
        "min_pairwise_action_l2": float(min(pairwise_distances)),
        "labels": label_summaries,
        "ranking_informative": bool(
            any(summary["informative"] for summary in label_summaries.values())
        ),
    }


def _canonical_array_hashes(
    value: _ExactArrayRecord,
    order: np.ndarray,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name, array in sorted(value.as_mapping().items()):
        canonical = np.asarray(array)[order]
        records[name] = {
            "shape": list(canonical.shape),
            "dtype": canonical.dtype.str,
            "sha256": sha256_array(canonical),
        }
    return records


def _validate_sha256(value: object, label: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _digest_tensor(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=object)
    if array.shape != shape:
        raise ValueError(f"{label} shape must be {shape}, got {array.shape}")
    for index, digest in np.ndenumerate(array):
        _validate_sha256(digest, f"{label}{index}")
    return array


def build_branch_receipt_fields(
    *,
    candidate_names: Sequence[str],
    candidate_actions: np.ndarray,
    actual_return: np.ndarray,
    actual_progress: np.ndarray,
    actual_success: np.ndarray,
    oracle_future_latent: np.ndarray,
    endpoint_physical_sha256_by_view: object,
    endpoint_rng_sha256_by_view: object,
    endpoint_rgb_sha256: np.ndarray,
) -> dict[str, list[Any]]:
    """Bind every candidate slot to its physical branch evidence and labels.

    Receipts follow a branch when candidate slots are anonymously permuted.  A
    receipt deliberately contains no sample-row index, so canonical row sorting
    also leaves its scientific identity unchanged.
    """

    names = tuple(str(value) for value in candidate_names)
    candidate_count = len(names)
    if candidate_count <= 0 or len(set(names)) != candidate_count:
        raise ValueError("branch receipts require unique candidate names")

    actions = np.asarray(candidate_actions)
    returns = np.asarray(actual_return)
    progress = np.asarray(actual_progress)
    success = np.asarray(actual_success)
    oracle = np.asarray(oracle_future_latent)
    endpoint_rgb = np.asarray(endpoint_rgb_sha256)
    expected_shapes = {
        "candidate_actions": (candidate_count, RAW_ACTION_STEPS, ACTION_DIM),
        "actual_return": (candidate_count, PREDICTION_STEPS),
        "actual_progress": (candidate_count, PREDICTION_STEPS),
        "actual_success": (candidate_count, PREDICTION_STEPS),
        "oracle_future_latent": (
            NUM_VIEWS,
            candidate_count,
            PREDICTION_STEPS,
            LATENT_DIM,
        ),
        "endpoint_rgb_sha256": (
            NUM_VIEWS,
            candidate_count,
            PREDICTION_STEPS,
            32,
        ),
    }
    arrays = {
        "candidate_actions": actions,
        "actual_return": returns,
        "actual_progress": progress,
        "actual_success": success,
        "oracle_future_latent": oracle,
        "endpoint_rgb_sha256": endpoint_rgb,
    }
    for name, expected_shape in expected_shapes.items():
        if arrays[name].shape != expected_shape:
            raise ValueError(
                f"{name} shape must be {expected_shape}, got {arrays[name].shape}"
            )
    expected_dtypes = {
        "candidate_actions": np.dtype(np.float32),
        "actual_return": np.dtype(np.float32),
        "actual_progress": np.dtype(np.float32),
        "actual_success": np.dtype(np.bool_),
        "oracle_future_latent": np.dtype(np.float32),
        "endpoint_rgb_sha256": np.dtype(np.uint8),
    }
    for name, expected_dtype in expected_dtypes.items():
        if arrays[name].dtype != expected_dtype:
            raise TypeError(
                f"{name} dtype must be {expected_dtype}, got {arrays[name].dtype}"
            )

    endpoint_physical = _digest_tensor(
        endpoint_physical_sha256_by_view,
        (NUM_VIEWS, candidate_count, PREDICTION_STEPS),
        "endpoint_physical_sha256_by_view",
    )
    endpoint_rng = _digest_tensor(
        endpoint_rng_sha256_by_view,
        (NUM_VIEWS, candidate_count, PREDICTION_STEPS),
        "endpoint_rng_sha256_by_view",
    )
    action_hashes = [sha256_array(actions[slot]) for slot in range(candidate_count)]
    label_hashes = [
        canonical_sha256(
            {
                "actual_return_sha256": sha256_array(returns[slot]),
                "actual_progress_sha256": sha256_array(progress[slot]),
                "actual_success_sha256": sha256_array(success[slot]),
            }
        )
        for slot in range(candidate_count)
    ]
    oracle_hashes = [
        [sha256_array(oracle[view, slot]) for slot in range(candidate_count)]
        for view in range(NUM_VIEWS)
    ]
    endpoint_rgb_hashes = [
        [
            [
                bytes(endpoint_rgb[view, slot, step].tolist()).hex()
                for step in range(PREDICTION_STEPS)
            ]
            for slot in range(candidate_count)
        ]
        for view in range(NUM_VIEWS)
    ]

    receipts: list[str] = []
    for slot, candidate_name in enumerate(names):
        receipts.append(
            canonical_sha256(
                {
                    "protocol": f"{COLLECTOR_PROTOCOL}:branch_receipt:v1",
                    "candidate_name": candidate_name,
                    "candidate_action_sha256": action_hashes[slot],
                    "candidate_label_sha256": label_hashes[slot],
                    "oracle_future_latent_sha256_by_view": [
                        oracle_hashes[view][slot] for view in range(NUM_VIEWS)
                    ],
                    "endpoint_physical_sha256_by_view": [
                        endpoint_physical[view, slot].tolist()
                        for view in range(NUM_VIEWS)
                    ],
                    "endpoint_rng_sha256_by_view": [
                        endpoint_rng[view, slot].tolist()
                        for view in range(NUM_VIEWS)
                    ],
                    "endpoint_rgb_sha256_by_view": [
                        endpoint_rgb_hashes[view][slot]
                        for view in range(NUM_VIEWS)
                    ],
                }
            )
        )
    return {
        "candidate_label_sha256": label_hashes,
        "oracle_future_latent_sha256_by_view": oracle_hashes,
        "endpoint_rgb_sha256_by_view": endpoint_rgb_hashes,
        "branch_receipt_sha256": receipts,
    }


def _validate_provenance_inputs(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    label: str = "provenance.inputs",
) -> None:
    _exact_keys(actual, frozenset(expected), label)
    for name, expected_value in expected.items():
        actual_value = _mapping(actual[name], f"{label}.{name}")
        expected_mapping = _mapping(expected_value, f"config.inputs.{name}")
        if set(expected_mapping) == {"path", "sha256"}:
            _exact_keys(
                actual_value,
                frozenset({"path", "expected_sha256", "actual_sha256"}),
                f"{label}.{name}",
            )
            path = actual_value["path"]
            if not isinstance(path, str) or not path:
                raise ValueError(f"{label}.{name}.path must be non-empty")
            configured = Path(str(expected_mapping["path"])).as_posix()
            if not Path(path).as_posix().endswith(configured):
                raise ValueError(f"{label}.{name}.path disagrees with pinned config")
            expected_digest = str(expected_mapping["sha256"])
            if (
                actual_value["expected_sha256"] != expected_digest
                or actual_value["actual_sha256"] != expected_digest
            ):
                raise ValueError(f"{label}.{name} failed pinned-file SHA-256 binding")
        else:
            _validate_provenance_inputs(
                expected_mapping,
                actual_value,
                label=f"{label}.{name}",
            )


def g0_eligibility(
    *,
    smoke: bool,
    upstream_status: str,
    git_dirty: bool,
) -> tuple[bool, list[str]]:
    """Return the fail-closed formal-G0 decision and explicit reasons."""

    if upstream_status not in {"verified", "not_auditable"}:
        raise ValueError("unknown upstream non-overlap status")
    reasons: list[str] = []
    if smoke:
        reasons.append("bounded_smoke_is_not_g0_eligible")
    if upstream_status != "verified":
        reasons.append("upstream_episode_non_overlap_not_auditable")
    if not smoke and git_dirty:
        reasons.append("dirty_worktree_full_collection")
    return not reasons, reasons


def validate_index_and_provenance(
    config: Mapping[str, Any],
    deployment: DeploymentInputs | Mapping[str, Any],
    labels: OfflineLabels | Mapping[str, Any],
    audit: AuditArrays | Mapping[str, Any],
    index_rows: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    *,
    smoke: bool,
) -> dict[str, Any]:
    """Close exactly-once, camera-pairing, permutation, and frozen-input audit."""

    if isinstance(deployment, Mapping):
        deployment = DeploymentInputs.from_mapping(deployment)
    if isinstance(labels, Mapping):
        labels = OfflineLabels.from_mapping(labels)
    if isinstance(audit, Mapping):
        audit = AuditArrays.from_mapping(audit)
    candidate_family = tuple(str(value) for value in config["evaluation"]["candidate_names"])
    candidate_count = len(candidate_family)
    axes = validate_array_bundle(
        deployment,
        labels,
        audit,
        candidate_count=candidate_count,
    )
    audit_camera_condition_extrinsics(
        audit, _mapping(config["conditions"], "conditions")
    )
    rows = _sequence(index_rows, "index_rows")
    if len(rows) != axes.sample_count:
        raise ValueError("index row count disagrees with array sample count")
    if len(set(candidate_family)) != candidate_count:
        raise ValueError("configured candidate family contains duplicate names")

    expected_splits: np.ndarray | None = None
    if not smoke:
        expected_splits, _ = assign_physical_episode_splits(
            deployment.task_index,
            audit.episode_id,
            audit.env_seed,
        )
    for sample, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"index_rows[{sample}]")
        _exact_keys(row, INDEX_ROW_KEYS, f"index_rows[{sample}]")
        scalar_expected = {
            "sample_index": sample,
            "task_index": int(deployment.task_index[sample]),
            "task_name": TASK_NAMES[int(deployment.task_index[sample])],
            "episode_id": int(audit.episode_id[sample]),
            "env_seed": int(audit.env_seed[sample]),
            "phase_index": int(audit.phase_index[sample]),
            "phase_name": PHASE_NAMES[int(audit.phase_index[sample])],
            "snapshot_step": int(audit.snapshot_step[sample]),
        }
        for name, expected in scalar_expected.items():
            if row[name] != expected:
                raise ValueError(f"index_rows[{sample}].{name} disagrees with arrays")
        expected_split = "smoke" if smoke else str(expected_splits[sample])
        if row["split"] != expected_split:
            raise ValueError(f"index_rows[{sample}].split disagrees with episode split")
        expected_episode_hash = physical_episode_sha256(
            scalar_expected["task_index"],
            scalar_expected["episode_id"],
            scalar_expected["env_seed"],
        )
        if row["physical_episode_sha256"] != expected_episode_hash:
            raise ValueError(
                f"index_rows[{sample}].physical_episode_sha256 disagrees with identity"
            )
        _validate_sha256(row["snapshot_sha256"], f"index_rows[{sample}].snapshot_sha256")
        if tuple(row["view_names"]) != VIEW_NAMES:
            raise ValueError(f"index_rows[{sample}].view_names changed canonical view order")

        names = tuple(str(value) for value in _sequence(row["candidate_names"], "candidate_names"))
        permutation = tuple(
            int(value) for value in _sequence(row["candidate_permutation"], "candidate_permutation")
        )
        if len(names) != candidate_count or set(names) != set(candidate_family):
            raise ValueError(f"index_rows[{sample}].candidate_names is not the candidate-family permutation")
        if sorted(permutation) != list(range(candidate_count)):
            raise ValueError(f"index_rows[{sample}].candidate_permutation is not bijective")
        expected_names = tuple(candidate_family[index] for index in permutation)
        if names != expected_names:
            raise ValueError(f"index_rows[{sample}] candidate names and permutation disagree")
        expected_permutation_hash = canonical_sha256(list(permutation))
        if row["candidate_permutation_sha256"] != expected_permutation_hash:
            raise ValueError(f"index_rows[{sample}] candidate permutation hash mismatch")

        action_hashes = tuple(
            str(value) for value in _sequence(row["candidate_action_sha256"], "candidate_action_sha256")
        )
        if len(action_hashes) != candidate_count:
            raise ValueError(f"index_rows[{sample}] candidate action hash count mismatch")
        expected_action_hashes = tuple(
            sha256_array(deployment.candidate_action_raw[sample, slot])
            for slot in range(candidate_count)
        )
        if action_hashes != expected_action_hashes:
            raise ValueError(f"index_rows[{sample}] candidate action slot hash mismatch")
        expected_set_hash = canonical_sha256(
            {
                "members": sorted(
                    (
                        {"name": name, "action_sha256": digest}
                        for name, digest in zip(names, action_hashes)
                    ),
                    key=lambda item: item["name"],
                )
            }
        )
        if row["candidate_action_set_sha256"] != expected_set_hash:
            raise ValueError(f"index_rows[{sample}] candidate action-set hash mismatch")
        if row["branch_execution_count"] != candidate_count:
            raise ValueError(f"index_rows[{sample}] must execute each candidate branch exactly once")

        history_physical = _digest_tensor(
            row["history_physical_sha256_by_view"],
            (NUM_VIEWS, HISTORY_SIZE),
            f"index_rows[{sample}].history_physical_sha256_by_view",
        )
        history_rng = _digest_tensor(
            row["history_rng_sha256_by_view"],
            (NUM_VIEWS, HISTORY_SIZE),
            f"index_rows[{sample}].history_rng_sha256_by_view",
        )
        endpoint_physical = _digest_tensor(
            row["endpoint_physical_sha256_by_view"],
            (NUM_VIEWS, candidate_count, PREDICTION_STEPS),
            f"index_rows[{sample}].endpoint_physical_sha256_by_view",
        )
        endpoint_rng = _digest_tensor(
            row["endpoint_rng_sha256_by_view"],
            (NUM_VIEWS, candidate_count, PREDICTION_STEPS),
            f"index_rows[{sample}].endpoint_rng_sha256_by_view",
        )
        parity_values = (
            ("history physical", history_physical),
            ("history RNG", history_rng),
            ("endpoint physical", endpoint_physical),
            ("endpoint RNG", endpoint_rng),
        )
        for label, value in parity_values:
            if not np.array_equal(value, np.broadcast_to(value[:1], value.shape)):
                raise ValueError(f"index_rows[{sample}] {label} hashes differ across camera views")

        candidate_label_hashes = _digest_tensor(
            row["candidate_label_sha256"],
            (candidate_count,),
            f"index_rows[{sample}].candidate_label_sha256",
        )
        oracle_hashes = _digest_tensor(
            row["oracle_future_latent_sha256_by_view"],
            (NUM_VIEWS, candidate_count),
            f"index_rows[{sample}].oracle_future_latent_sha256_by_view",
        )
        endpoint_rgb_hashes = _digest_tensor(
            row["endpoint_rgb_sha256_by_view"],
            (NUM_VIEWS, candidate_count, PREDICTION_STEPS),
            f"index_rows[{sample}].endpoint_rgb_sha256_by_view",
        )
        branch_receipts = _digest_tensor(
            row["branch_receipt_sha256"],
            (candidate_count,),
            f"index_rows[{sample}].branch_receipt_sha256",
        )
        expected_receipt_fields = build_branch_receipt_fields(
            candidate_names=names,
            candidate_actions=deployment.candidate_action_raw[sample],
            actual_return=labels.actual_return[sample],
            actual_progress=labels.actual_progress[sample],
            actual_success=labels.actual_success[sample],
            oracle_future_latent=labels.oracle_future_latent[sample],
            endpoint_physical_sha256_by_view=endpoint_physical,
            endpoint_rng_sha256_by_view=endpoint_rng,
            endpoint_rgb_sha256=audit.endpoint_rgb_sha256[sample],
        )
        receipt_comparisons = (
            (
                "candidate label slot hash",
                candidate_label_hashes.tolist(),
                expected_receipt_fields["candidate_label_sha256"],
            ),
            (
                "oracle future latent slot hash",
                oracle_hashes.tolist(),
                expected_receipt_fields["oracle_future_latent_sha256_by_view"],
            ),
            (
                "endpoint RGB slot hash",
                endpoint_rgb_hashes.tolist(),
                expected_receipt_fields["endpoint_rgb_sha256_by_view"],
            ),
            (
                "branch receipt",
                branch_receipts.tolist(),
                expected_receipt_fields["branch_receipt_sha256"],
            ),
        )
        for label, actual_value, expected_value in receipt_comparisons:
            if actual_value != expected_value:
                raise ValueError(f"index_rows[{sample}] {label} mismatch")

    provenance_mapping = _mapping(provenance, "provenance")
    _exact_keys(provenance_mapping, PROVENANCE_KEYS, "provenance")
    if provenance_mapping["schema_version"] != SCHEMA_VERSION:
        raise ValueError("provenance schema version mismatch")
    if provenance_mapping["protocol"] != PROTOCOL:
        raise ValueError("provenance protocol mismatch")
    if provenance_mapping["scope"] != config["scope"]:
        raise ValueError("provenance scope disagrees with config")
    if provenance_mapping["config"] != config:
        raise ValueError("provenance config is not the resolved collection config")
    config_hash = canonical_sha256(config)
    if provenance_mapping["config_sha256"] != config_hash:
        raise ValueError("provenance config SHA-256 mismatch")
    git = _mapping(provenance_mapping["git"], "provenance.git")
    _exact_keys(git, frozenset({"before", "after"}), "provenance.git")
    if not git["before"] or git["before"] != git["after"]:
        raise ValueError("git state changed during collection")
    versions = _mapping(provenance_mapping["versions"], "provenance.versions")
    if not versions:
        raise ValueError("provenance versions must be non-empty")
    _validate_provenance_inputs(
        _mapping(config["inputs"], "config.inputs"),
        _mapping(provenance_mapping["inputs"], "provenance.inputs"),
    )
    before = _mapping(provenance_mapping["model_state_before"], "model_state_before")
    after = _mapping(provenance_mapping["model_state_after"], "model_state_after")
    _exact_keys(before, MODEL_STATE_KEYS, "model_state_before")
    _exact_keys(after, MODEL_STATE_KEYS, "model_state_after")
    for name, digest in before.items():
        _validate_sha256(digest, f"model_state_before.{name}")
    for name, digest in after.items():
        _validate_sha256(digest, f"model_state_after.{name}")
    if before != after or provenance_mapping["no_training_verified"] is not True:
        raise ValueError("frozen model state changed or no-training audit is false")
    projection = _mapping(
        provenance_mapping["goal_visual_projection"],
        "goal_visual_projection",
    )
    _exact_keys(
        projection,
        frozenset({"seed", "sha256", "input_dim", "output_dim", "source"}),
        "goal_visual_projection",
    )
    if projection["seed"] != goal_visual_projection_seed(int(config["seed"])):
        raise ValueError("goal visual projection seed mismatch")
    projection_sha256 = _validate_sha256(
        projection["sha256"], "goal_visual_projection.sha256"
    )
    goal_context = _mapping(config["goal_context"], "config.goal_context")
    if projection["input_dim"] != goal_context["visual_input_dim"]:
        raise ValueError("goal visual projection input_dim mismatch")
    if projection["output_dim"] != goal_context["visual_dim"]:
        raise ValueError("goal visual projection output_dim mismatch")
    if projection_sha256 != goal_context["projection_sha256"]:
        raise ValueError("goal visual projection SHA-256 mismatch")
    if projection["source"] != "current_rgb_frozen_vit_prefix":
        raise ValueError("goal visual projection source mismatch")
    expected_branches = axes.sample_count * candidate_count
    if provenance_mapping["sample_count"] != axes.sample_count:
        raise ValueError("provenance sample count mismatch")
    if provenance_mapping["physical_branch_count"] != expected_branches:
        raise ValueError("provenance physical branch count mismatch")
    expected_contract_hash = canonical_sha256(
        {
            "config_sha256": config_hash,
            "sample_count": axes.sample_count,
            "physical_branch_count": expected_branches,
        }
    )
    if provenance_mapping["result_contract_sha256"] != expected_contract_hash:
        raise ValueError("provenance result contract SHA-256 mismatch")

    upstream = _mapping(provenance_mapping["upstream_non_overlap"], "upstream_non_overlap")
    _exact_keys(
        upstream,
        frozenset({"status", "reason", "training_episode_ledger_sha256"}),
        "upstream_non_overlap",
    )
    if upstream["status"] not in {"verified", "not_auditable"}:
        raise ValueError("upstream non-overlap status must be verified or not_auditable")
    if not isinstance(upstream["reason"], str) or not upstream["reason"]:
        raise ValueError("upstream non-overlap audit needs an explicit reason")
    ledger = upstream["training_episode_ledger_sha256"]
    if upstream["status"] == "verified":
        _validate_sha256(ledger, "upstream training episode ledger")
    elif ledger is not None:
        raise ValueError("not_auditable upstream status cannot claim a training ledger")

    order = canonical_row_order(deployment, audit)
    canonical_rows = [dict(rows[index]) for index in order.tolist()]
    git_before = _mapping(git["before"], "provenance.git.before")
    if not isinstance(git_before.get("dirty"), bool):
        raise ValueError("provenance git state must include a boolean dirty flag")
    git_clean = not bool(git_before["dirty"])
    eligible, reasons = g0_eligibility(
        smoke=smoke,
        upstream_status=str(upstream["status"]),
        git_dirty=not git_clean,
    )
    return {
        "passed": True,
        "g0_eligible": eligible,
        "ineligibility_reasons": reasons,
        "index_rows_sha256": canonical_sha256(canonical_rows),
        "provenance_sha256": canonical_sha256(provenance_mapping),
        "upstream_non_overlap": dict(upstream),
        "exact_branch_execution_count": expected_branches,
        "camera_pairing_verified": True,
        "candidate_slot_hashes_verified": True,
        "git_clean": git_clean,
    }


def build_g0_manifest(
    config: Mapping[str, Any],
    deployment: DeploymentInputs | Mapping[str, Any],
    labels: OfflineLabels | Mapping[str, Any],
    audit: AuditArrays | Mapping[str, Any],
    *,
    index_rows: Sequence[Mapping[str, Any]] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a bundle and return a deterministic, self-hashed G0 manifest."""

    is_smoke = config.get("evaluation", {}).get("episodes_per_task") == 1
    if is_smoke:
        validate_smoke_config(config)
    else:
        validate_full_config(config)
    if isinstance(deployment, Mapping):
        deployment = DeploymentInputs.from_mapping(deployment)
    if isinstance(labels, Mapping):
        labels = OfflineLabels.from_mapping(labels)
    if isinstance(audit, Mapping):
        audit = AuditArrays.from_mapping(audit)
    candidate_count = len(config["evaluation"]["candidate_names"])
    axes = validate_array_bundle(
        deployment, labels, audit, candidate_count=candidate_count
    )
    episode_audit = audit_episode_groups(deployment, audit, smoke=is_smoke)
    geometry_audit = audit_geometric_reprojection(deployment, labels, audit)
    rgb_audit = audit_history_rgb(deployment.history_rgb)
    endpoint_rgb_audit = audit_endpoint_rgb(audit)
    camera_condition_audit = audit_camera_condition_extrinsics(
        audit, _mapping(config["conditions"], "conditions")
    )
    ranking_audit = audit_ranking_informativeness(deployment, labels, audit)
    if (index_rows is None) != (provenance is None):
        raise ValueError("index_rows and provenance must be supplied together")
    if index_rows is None:
        contract_audit = {
            "passed": False,
            "g0_eligible": False,
            "ineligibility_reasons": ["index_and_provenance_audit_missing"],
            "index_rows_sha256": None,
            "provenance_sha256": None,
            "upstream_non_overlap": None,
            "exact_branch_execution_count": None,
            "camera_pairing_verified": False,
            "candidate_slot_hashes_verified": False,
        }
    else:
        assert provenance is not None
        contract_audit = validate_index_and_provenance(
            config,
            deployment,
            labels,
            audit,
            index_rows,
            provenance,
            smoke=is_smoke,
        )
    g0_eligible = bool(
        episode_audit["g0_eligible"] and contract_audit["g0_eligible"]
    )
    overall_passed = bool(
        episode_audit["passed"]
        and contract_audit["passed"]
        and geometry_audit["passed"]
        and camera_condition_audit["passed"]
        and rgb_audit["passed"]
        and endpoint_rgb_audit["passed"]
        and ranking_audit["passed"]
    )
    g0_audit = {
        **episode_audit,
        "passed": overall_passed,
        "g0_eligible": g0_eligible,
        "index_and_provenance": contract_audit,
        "geometric_reprojection": geometry_audit,
        "camera_conditions": camera_condition_audit,
        "history_rgb_observability": rgb_audit,
        "endpoint_rgb_observability": endpoint_rgb_audit,
        "ranking_informativeness": ranking_audit,
    }
    order = canonical_row_order(deployment, audit)
    identities = [
        {
            "task_index": int(deployment.task_index[index]),
            "episode_id": int(audit.episode_id[index]),
            "env_seed": int(audit.env_seed[index]),
            "phase_index": int(audit.phase_index[index]),
            "snapshot_step": int(audit.snapshot_step[index]),
        }
        for index in order.tolist()
    ]
    if len({canonical_sha256(item) for item in identities}) != len(identities):
        raise ValueError("duplicate task/episode/phase rows in v2 bundle")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "mode": "smoke" if is_smoke else "full",
        "config_sha256": canonical_sha256(config),
        "axes": {
            "N": axes.sample_count,
            "V": NUM_VIEWS,
            "M": NUM_MODELS,
            "G": NUM_GROUNDINGS,
            "T": HISTORY_SIZE,
            "C": axes.candidate_count,
            "P": PREDICTION_STEPS,
            "D": LATENT_DIM,
            "K": NUM_KEYPOINTS,
        },
        "axis_names": {
            "views": list(VIEW_NAMES),
            "models": list(MODEL_NAMES),
            "groundings": list(GROUNDING_NAMES),
            "phases": list(PHASE_NAMES),
            "candidate_family": list(config["evaluation"]["candidate_names"]),
            "candidate_slots": [f"slot_{index}" for index in range(candidate_count)],
        },
        "goal_context": dict(config["goal_context"]),
        "camera_invariant_label_keys": [
            "actual_progress",
            "actual_return",
            "actual_success",
        ],
        "canonical_row_identities": identities,
        "deployment_arrays": _canonical_array_hashes(deployment, order),
        "offline_label_arrays": _canonical_array_hashes(labels, order),
        "audit_arrays": _canonical_array_hashes(audit, order),
        "index_rows_sha256": contract_audit["index_rows_sha256"],
        "provenance_sha256": contract_audit["provenance_sha256"],
        "g0_audit": g0_audit,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def validate_manifest_hash(manifest: Mapping[str, Any]) -> None:
    expected = manifest.get("manifest_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("manifest_sha256 is missing")
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    actual = canonical_sha256(payload)
    if actual != expected:
        raise ValueError(f"manifest SHA-256 mismatch: expected {expected}, got {actual}")


__all__ = [
    "ACTION_DIM",
    "ACTION_ONLY_PROBE_KEYS",
    "AUDIT_ARRAY_KEYS",
    "ActionOnlyProbeInputs",
    "ArrayAxes",
    "AuditArrays",
    "DEPLOYMENT_KEYS",
    "DIRECT_PROBE_KEYS",
    "DirectProbeInputs",
    "DeploymentInputs",
    "FULL_CANDIDATE_NAMES",
    "GROUNDING_NAMES",
    "GOAL_VISUAL_INPUT_DIM",
    "GOAL_VISUAL_PROJECTION_SHA256",
    "IMAGE_SIZE",
    "INDEX_ROW_KEYS",
    "LATENT_DIM",
    "MODEL_NAMES",
    "MODEL_STATE_KEYS",
    "NON_DEPLOYMENT_METADATA_KEYS",
    "NUM_KEYPOINTS",
    "OFFLINE_LABEL_KEYS",
    "ORACLE_PROBE_KEYS",
    "OracleProbeInputs",
    "OfflineLabels",
    "PHASE_NAMES",
    "PHASE_STEPS",
    "PREDICTION_STEPS",
    "PRIVILEGED_INPUT_KEYS",
    "PROVENANCE_KEYS",
    "PROTOCOL",
    "PREDICTED_PROBE_KEYS",
    "PROBE_NAMES",
    "PredictedProbeInputs",
    "SCHEMA_VERSION",
    "SCORER_INPUT_KEYS",
    "SMOKE_CANDIDATE_NAMES",
    "TASK_NAMES",
    "VIEW_NAMES",
    "assert_no_privileged_inputs",
    "assign_physical_episode_splits",
    "audit_camera_condition_extrinsics",
    "audit_endpoint_rgb",
    "audit_geometric_reprojection",
    "audit_ranking_informativeness",
    "audit_episode_groups",
    "build_g2_probe_pair",
    "build_g3_probe_pair",
    "build_probe_inputs",
    "build_g0_manifest",
    "canonical_row_order",
    "canonical_sha256",
    "expected_camera_condition_extrinsics",
    "goal_visual_projection_seed",
    "g0_eligibility",
    "grounded_validity",
    "physical_episode_sha256",
    "sha256_array",
    "smoke_config",
    "validate_array_bundle",
    "validate_full_config",
    "validate_manifest_hash",
    "validate_index_and_provenance",
    "validate_deployable_probe_inputs",
    "validate_smoke_config",
]
