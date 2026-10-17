"""Exact same-environment snapshots for MetaWorld on modern MuJoCo."""

from __future__ import annotations

import copy
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


_MODEL_ARRAY_NAMES = ("body_pos", "site_pos", "cam_pos", "cam_quat")
_MODEL_DIMENSION_NAMES = (
    "nq",
    "nv",
    "na",
    "nu",
    "nbody",
    "nsite",
    "ncam",
    "nmocap",
    "neq",
    "npluginstate",
    "nuserdata",
)
_REQUIRED_TASK_ATTRIBUTES = (
    "_last_rand_vec",
    "_target_pos",
    "obj_init_pos",
    "_freeze_rand_vec",
    "_partially_observable",
    "_set_task_called",
)
_OPTIONAL_TASK_ATTRIBUTES = (
    "maxReachDist",
    "objHeight",
    "heightTarget",
    "maxPushDist",
    "maxPlacingDist",
    "_obj_to_target_init",
    "maxDist",
    "_wipe_started",
    "_wipe_half_cycles",
    "_grasp_lifted",
)
_RUNTIME_ATTRIBUTES = (
    "curr_path_length",
    "_prev_obs",
    "_last_stable_obs",
    "_did_see_sim_exception",
)


class SnapshotMismatchError(RuntimeError):
    """Raised when a snapshot no longer matches its source environment."""


@dataclass(frozen=True)
class _FrozenMapping:
    items: tuple[tuple[Any, Any], ...]


@dataclass(frozen=True)
class _FrozenList:
    items: tuple[Any, ...]


def _readonly_array(value: Any, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _freeze_value(value: Any) -> Any:
    """Deep-copy structured RNG/attribute state into immutable containers."""

    if isinstance(value, np.ndarray):
        return _readonly_array(value)
    if isinstance(value, np.generic):
        return copy.deepcopy(value)
    if isinstance(value, Mapping):
        return _FrozenMapping(
            tuple((_freeze_value(key), _freeze_value(item)) for key, item in value.items())
        )
    if isinstance(value, list):
        return _FrozenList(tuple(_freeze_value(item) for item in value))
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    return copy.deepcopy(value)


def _thaw_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True, order="C")
    if isinstance(value, _FrozenMapping):
        return {_thaw_value(key): _thaw_value(item) for key, item in value.items}
    if isinstance(value, _FrozenList):
        return [_thaw_value(item) for item in value.items]
    if isinstance(value, tuple):
        return tuple(_thaw_value(item) for item in value)
    return copy.deepcopy(value)


def _array_bytes_equal(left: np.ndarray, right: Any) -> bool:
    if not isinstance(right, np.ndarray):
        return False
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _frozen_value_equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, np.ndarray):
        return _array_bytes_equal(expected, actual)
    if isinstance(expected, _FrozenMapping):
        if not isinstance(actual, Mapping) or len(actual) != len(expected.items):
            return False
        for key, item in expected.items:
            thawed_key = _thaw_value(key)
            if thawed_key not in actual or not _frozen_value_equal(item, actual[thawed_key]):
                return False
        return True
    if isinstance(expected, _FrozenList):
        return isinstance(actual, list) and len(actual) == len(expected.items) and all(
            _frozen_value_equal(item, current)
            for item, current in zip(expected.items, actual)
        )
    if isinstance(expected, tuple):
        return isinstance(actual, tuple) and len(actual) == len(expected) and all(
            _frozen_value_equal(item, current)
            for item, current in zip(expected, actual)
        )
    if isinstance(expected, np.generic):
        current = np.asarray(actual)
        saved = np.asarray(expected)
        return (
            current.shape == saved.shape
            and current.dtype == saved.dtype
            and current.tobytes() == saved.tobytes()
        )
    if isinstance(expected, float) and isinstance(actual, float):
        return np.asarray(expected).tobytes() == np.asarray(actual).tobytes()
    try:
        result = expected == actual
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _class_fingerprint(value: Any) -> tuple[str, str]:
    cls = type(value)
    return cls.__module__, cls.__qualname__


@dataclass(frozen=True, eq=False)
class AttributeSnapshot:
    """One explicitly selected Python attribute in a snapshot."""

    name: str
    present: bool
    value: Any = None


@dataclass(frozen=True, eq=False)
class TaskFingerprint:
    """Captured task-defining values and their environment contract."""

    env_class_module: str
    env_class_qualname: str
    attributes: tuple[AttributeSnapshot, ...]


@dataclass(frozen=True)
class ModelSignature:
    """Fixed MuJoCo model dimensions and state-vector sizes."""

    model_class_module: str
    model_class_qualname: str
    dimensions: tuple[tuple[str, int], ...]
    integration_state_size: int
    warmstart_state_size: int
    array_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    array_dtypes: tuple[tuple[str, str], ...]


@dataclass(frozen=True, eq=False)
class ModelArraysSnapshot:
    """Mutable model arrays used by MetaWorld task and camera setup."""

    body_pos: np.ndarray
    site_pos: np.ndarray
    cam_pos: np.ndarray
    cam_quat: np.ndarray


@dataclass(frozen=True, eq=False)
class BitGeneratorSnapshot:
    """A deep copy of one NumPy Generator bit-generator state."""

    bit_generator_module: str
    bit_generator_qualname: str
    state: Any


@dataclass(frozen=True, eq=False)
class EnvironmentRNGSnapshot:
    """RNGs owned by MetaWorld and its public action/observation spaces."""

    env: BitGeneratorSnapshot | None
    action_space: BitGeneratorSnapshot | None
    observation_space: BitGeneratorSnapshot | None


@dataclass(frozen=True, eq=False)
class ProcessRNGSnapshot:
    """Optional process-wide Python, NumPy, and Torch RNG state."""

    python_state: Any
    numpy_state: Any
    torch_available: bool
    torch_cpu_state: np.ndarray | None
    torch_cuda_states: tuple[np.ndarray, ...]


@dataclass(frozen=True, eq=False)
class MetaWorldSnapshot:
    """State needed to replay a branch on the same MetaWorld env instance."""

    environment_id: int
    model_id: int
    data_id: int
    model_signature: ModelSignature
    model_arrays: ModelArraysSnapshot
    integration_state: np.ndarray
    warmstart_state: np.ndarray
    task_fingerprint: TaskFingerprint
    runtime_attributes: tuple[AttributeSnapshot, ...]
    environment_rng: EnvironmentRNGSnapshot
    process_rng: ProcessRNGSnapshot | None


def _require_mujoco() -> Any:
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - exercised only without dependency
        raise RuntimeError("MuJoCo is required for MetaWorld snapshots") from exc
    return mujoco


def _state_specs(mujoco: Any) -> tuple[Any, Any]:
    try:
        return (
            mujoco.mjtState.mjSTATE_INTEGRATION,
            mujoco.mjtState.mjSTATE_WARMSTART,
        )
    except AttributeError as exc:
        raise RuntimeError(
            "MuJoCo does not expose mjSTATE_INTEGRATION/mjSTATE_WARMSTART"
        ) from exc


def _capture_mujoco_state(mujoco: Any, model: Any, data: Any, spec: Any) -> np.ndarray:
    size = int(mujoco.mj_stateSize(model, spec))
    if size < 0:
        raise RuntimeError(f"MuJoCo returned an invalid state size: {size}")
    state = np.empty(size, dtype=np.float64)
    mujoco.mj_getState(model, data, state, spec)
    state.setflags(write=False)
    return state


def _capture_model_signature(
    model: Any,
    *,
    integration_state_size: int,
    warmstart_state_size: int,
) -> ModelSignature:
    module, qualname = _class_fingerprint(model)
    dimensions = tuple(
        (name, int(getattr(model, name)))
        for name in _MODEL_DIMENSION_NAMES
        if hasattr(model, name)
    )
    arrays = {name: np.asarray(getattr(model, name)) for name in _MODEL_ARRAY_NAMES}
    return ModelSignature(
        model_class_module=module,
        model_class_qualname=qualname,
        dimensions=dimensions,
        integration_state_size=integration_state_size,
        warmstart_state_size=warmstart_state_size,
        array_shapes=tuple((name, arrays[name].shape) for name in _MODEL_ARRAY_NAMES),
        array_dtypes=tuple((name, arrays[name].dtype.str) for name in _MODEL_ARRAY_NAMES),
    )


def _capture_task_fingerprint(env: Any) -> TaskFingerprint:
    attributes: list[AttributeSnapshot] = []
    for name in _REQUIRED_TASK_ATTRIBUTES:
        if not hasattr(env, name):
            raise AttributeError(
                f"MetaWorld environment is missing required task attribute {name!r}"
            )
        attributes.append(AttributeSnapshot(name, True, _freeze_value(getattr(env, name))))
    for name in _OPTIONAL_TASK_ATTRIBUTES:
        present = hasattr(env, name)
        value = _freeze_value(getattr(env, name)) if present else None
        attributes.append(AttributeSnapshot(name, present, value))
    module, qualname = _class_fingerprint(env)
    return TaskFingerprint(module, qualname, tuple(attributes))


def _capture_runtime_attributes(env: Any) -> tuple[AttributeSnapshot, ...]:
    result = []
    for name in _RUNTIME_ATTRIBUTES:
        if not hasattr(env, name):
            raise AttributeError(
                f"MetaWorld environment is missing required runtime attribute {name!r}"
            )
        result.append(AttributeSnapshot(name, True, _freeze_value(getattr(env, name))))
    return tuple(result)


def _capture_bit_generator(owner: Any) -> BitGeneratorSnapshot | None:
    if owner is None:
        return None
    try:
        rng = getattr(owner, "np_random")
    except AttributeError:
        return None
    if rng is None:
        return None
    bit_generator = getattr(rng, "bit_generator", None)
    if bit_generator is None:
        raise TypeError("np_random must expose a NumPy bit_generator")
    module, qualname = _class_fingerprint(bit_generator)
    return BitGeneratorSnapshot(module, qualname, _freeze_value(bit_generator.state))


def _capture_environment_rng(env: Any) -> EnvironmentRNGSnapshot:
    return EnvironmentRNGSnapshot(
        env=_capture_bit_generator(env),
        action_space=_capture_bit_generator(getattr(env, "action_space", None)),
        observation_space=_capture_bit_generator(
            getattr(env, "observation_space", None)
        ),
    )


def capture_process_rng_snapshot() -> ProcessRNGSnapshot:
    """Capture process-wide RNG state without requiring Torch at module import."""

    python_state = _freeze_value(random.getstate())
    numpy_state = _freeze_value(np.random.get_state())
    try:
        import torch
    except ImportError:
        return ProcessRNGSnapshot(python_state, numpy_state, False, None, ())

    cpu_state = _readonly_array(torch.get_rng_state().detach().cpu().numpy())
    cuda_states: tuple[np.ndarray, ...] = ()
    if torch.cuda.is_available():
        cuda_states = tuple(
            _readonly_array(state.detach().cpu().numpy())
            for state in torch.cuda.get_rng_state_all()
        )
    return ProcessRNGSnapshot(
        python_state=python_state,
        numpy_state=numpy_state,
        torch_available=True,
        torch_cpu_state=cpu_state,
        torch_cuda_states=cuda_states,
    )


def capture_metaworld_snapshot(
    env: Any,
    *,
    include_process_rng: bool = False,
) -> MetaWorldSnapshot:
    """Capture an exact branch point for later restore on ``env`` itself."""

    if not hasattr(env, "model") or not hasattr(env, "data"):
        raise TypeError("env must expose MuJoCo model and data attributes")
    model = env.model
    data = env.data
    mujoco = _require_mujoco()
    integration_spec, warmstart_spec = _state_specs(mujoco)
    integration_state = _capture_mujoco_state(
        mujoco, model, data, integration_spec
    )
    warmstart_state = _capture_mujoco_state(mujoco, model, data, warmstart_spec)
    signature = _capture_model_signature(
        model,
        integration_state_size=integration_state.size,
        warmstart_state_size=warmstart_state.size,
    )
    model_arrays = ModelArraysSnapshot(
        **{
            name: _readonly_array(getattr(model, name))
            for name in _MODEL_ARRAY_NAMES
        }
    )
    return MetaWorldSnapshot(
        environment_id=id(env),
        model_id=id(model),
        data_id=id(data),
        model_signature=signature,
        model_arrays=model_arrays,
        integration_state=integration_state,
        warmstart_state=warmstart_state,
        task_fingerprint=_capture_task_fingerprint(env),
        runtime_attributes=_capture_runtime_attributes(env),
        environment_rng=_capture_environment_rng(env),
        process_rng=(capture_process_rng_snapshot() if include_process_rng else None),
    )


def _validate_task_fingerprint(env: Any, fingerprint: TaskFingerprint) -> None:
    current_class = _class_fingerprint(env)
    expected_class = (
        fingerprint.env_class_module,
        fingerprint.env_class_qualname,
    )
    if current_class != expected_class:
        raise SnapshotMismatchError(
            f"environment class mismatch: snapshot {expected_class}, current {current_class}"
        )
    for attribute in fingerprint.attributes:
        present = hasattr(env, attribute.name)
        if present != attribute.present:
            raise SnapshotMismatchError(
                f"task fingerprint attribute presence mismatch: {attribute.name}"
            )


def _restore_task_attributes(
    env: Any,
    values: tuple[tuple[AttributeSnapshot, Any], ...],
) -> None:
    for attribute, saved in values:
        current = getattr(env, attribute.name)
        if (
            isinstance(current, np.ndarray)
            and isinstance(saved, np.ndarray)
            and current.shape == saved.shape
            and current.dtype == saved.dtype
            and current.flags.writeable
        ):
            np.copyto(current, saved, casting="no")
        else:
            setattr(env, attribute.name, saved)

    for attribute, _saved in values:
        if not _frozen_value_equal(attribute.value, getattr(env, attribute.name)):
            raise SnapshotMismatchError(
                f"task attribute {attribute.name} was not restored bit-exactly"
            )


def _validate_model(
    mujoco: Any,
    model: Any,
    snapshot: MetaWorldSnapshot,
    integration_spec: Any,
    warmstart_spec: Any,
) -> None:
    signature = snapshot.model_signature
    current_class = _class_fingerprint(model)
    expected_class = (
        signature.model_class_module,
        signature.model_class_qualname,
    )
    if current_class != expected_class:
        raise SnapshotMismatchError(
            f"MuJoCo model class mismatch: snapshot {expected_class}, current {current_class}"
        )
    for name, expected in signature.dimensions:
        if not hasattr(model, name) or int(getattr(model, name)) != expected:
            current = getattr(model, name, "<missing>")
            raise SnapshotMismatchError(
                f"MuJoCo model dimension mismatch for {name}: "
                f"snapshot {expected}, current {current}"
            )

    current_integration_size = int(mujoco.mj_stateSize(model, integration_spec))
    current_warmstart_size = int(mujoco.mj_stateSize(model, warmstart_spec))
    if current_integration_size != signature.integration_state_size:
        raise SnapshotMismatchError(
            "mjSTATE_INTEGRATION size mismatch: "
            f"snapshot {signature.integration_state_size}, current {current_integration_size}"
        )
    if current_warmstart_size != signature.warmstart_state_size:
        raise SnapshotMismatchError(
            "mjSTATE_WARMSTART size mismatch: "
            f"snapshot {signature.warmstart_state_size}, current {current_warmstart_size}"
        )
    if snapshot.integration_state.shape != (signature.integration_state_size,):
        raise SnapshotMismatchError(
            f"snapshot integration state shape mismatch: {snapshot.integration_state.shape}"
        )
    if snapshot.warmstart_state.shape != (signature.warmstart_state_size,):
        raise SnapshotMismatchError(
            f"snapshot warm-start state shape mismatch: {snapshot.warmstart_state.shape}"
        )
    if snapshot.integration_state.dtype != np.dtype(np.float64):
        raise SnapshotMismatchError("snapshot integration state dtype must be float64")
    if snapshot.warmstart_state.dtype != np.dtype(np.float64):
        raise SnapshotMismatchError("snapshot warm-start state dtype must be float64")

    expected_shapes = dict(signature.array_shapes)
    expected_dtypes = dict(signature.array_dtypes)
    for name in _MODEL_ARRAY_NAMES:
        current = np.asarray(getattr(model, name))
        saved = getattr(snapshot.model_arrays, name)
        if current.shape != expected_shapes[name] or saved.shape != expected_shapes[name]:
            raise SnapshotMismatchError(
                f"MuJoCo model array shape mismatch for {name}: "
                f"snapshot {saved.shape}, current {current.shape}, "
                f"expected {expected_shapes[name]}"
            )
        if current.dtype.str != expected_dtypes[name] or saved.dtype.str != expected_dtypes[name]:
            raise SnapshotMismatchError(
                f"MuJoCo model array dtype mismatch for {name}: "
                f"snapshot {saved.dtype.str}, current {current.dtype.str}, "
                f"expected {expected_dtypes[name]}"
            )


def _validate_bit_generator(
    owner: Any,
    snapshot: BitGeneratorSnapshot | None,
    owner_name: str,
) -> Any | None:
    if snapshot is None:
        return None
    if owner is None:
        raise SnapshotMismatchError(f"missing RNG owner: {owner_name}")
    try:
        rng = getattr(owner, "np_random")
    except AttributeError as exc:
        raise SnapshotMismatchError(f"missing RNG on {owner_name}") from exc
    bit_generator = getattr(rng, "bit_generator", None)
    if bit_generator is None:
        raise SnapshotMismatchError(f"invalid RNG on {owner_name}: no bit_generator")
    current_class = _class_fingerprint(bit_generator)
    expected_class = (
        snapshot.bit_generator_module,
        snapshot.bit_generator_qualname,
    )
    if current_class != expected_class:
        raise SnapshotMismatchError(
            f"bit-generator mismatch on {owner_name}: "
            f"snapshot {expected_class}, current {current_class}"
        )
    return bit_generator


def _validate_environment_rng(
    env: Any,
    snapshot: EnvironmentRNGSnapshot,
) -> tuple[Any | None, Any | None, Any | None]:
    return (
        _validate_bit_generator(env, snapshot.env, "env"),
        _validate_bit_generator(
            getattr(env, "action_space", None), snapshot.action_space, "action_space"
        ),
        _validate_bit_generator(
            getattr(env, "observation_space", None),
            snapshot.observation_space,
            "observation_space",
        ),
    )


def _validate_process_rng(snapshot: ProcessRNGSnapshot | None) -> Any | None:
    if snapshot is None or not snapshot.torch_available:
        return None
    try:
        import torch
    except ImportError as exc:
        raise SnapshotMismatchError(
            "Torch was available at capture but is unavailable at restore"
        ) from exc
    if snapshot.torch_cpu_state is None:
        raise SnapshotMismatchError("captured Torch CPU RNG state is missing")
    current_cpu_state = torch.get_rng_state()
    if tuple(current_cpu_state.shape) != snapshot.torch_cpu_state.shape:
        raise SnapshotMismatchError(
            "Torch CPU RNG state shape mismatch: "
            f"snapshot {snapshot.torch_cpu_state.shape}, "
            f"current {tuple(current_cpu_state.shape)}"
        )
    if snapshot.torch_cuda_states:
        if not torch.cuda.is_available():
            raise SnapshotMismatchError(
                "CUDA RNG states were captured but CUDA is unavailable at restore"
            )
        current_count = torch.cuda.device_count()
        if current_count != len(snapshot.torch_cuda_states):
            raise SnapshotMismatchError(
                "CUDA device count mismatch: "
                f"snapshot {len(snapshot.torch_cuda_states)}, current {current_count}"
            )
    return torch


def _restore_process_rng(snapshot: ProcessRNGSnapshot, torch: Any | None) -> None:
    random.setstate(_thaw_value(snapshot.python_state))
    np.random.set_state(_thaw_value(snapshot.numpy_state))
    if not snapshot.torch_available:
        return
    assert torch is not None and snapshot.torch_cpu_state is not None
    torch.set_rng_state(torch.from_numpy(np.array(snapshot.torch_cpu_state, copy=True)))
    if snapshot.torch_cuda_states:
        torch.cuda.set_rng_state_all(
            [torch.from_numpy(np.array(state, copy=True)) for state in snapshot.torch_cuda_states]
        )


def restore_process_rng_snapshot(snapshot: ProcessRNGSnapshot) -> None:
    """Restore a standalone process RNG snapshot."""

    if not isinstance(snapshot, ProcessRNGSnapshot):
        raise TypeError("snapshot must be a ProcessRNGSnapshot")
    torch = _validate_process_rng(snapshot)
    _restore_process_rng(snapshot, torch)


def _assert_restored_array(name: str, expected: np.ndarray, actual: np.ndarray) -> None:
    if not _array_bytes_equal(expected, actual):
        raise SnapshotMismatchError(f"{name} was not restored bit-exactly")


def restore_metaworld_snapshot(
    env: Any,
    snapshot: MetaWorldSnapshot,
    *,
    restore_process_rng: bool = True,
) -> None:
    """Restore ``snapshot`` onto the exact environment instance it came from."""

    if not isinstance(snapshot, MetaWorldSnapshot):
        raise TypeError("snapshot must be a MetaWorldSnapshot")
    if id(env) != snapshot.environment_id:
        raise SnapshotMismatchError("snapshot belongs to a different environment instance")
    if not hasattr(env, "model") or not hasattr(env, "data"):
        raise SnapshotMismatchError("environment no longer exposes MuJoCo model/data")
    model = env.model
    data = env.data
    if id(model) != snapshot.model_id:
        raise SnapshotMismatchError("environment MuJoCo model instance changed")
    if id(data) != snapshot.data_id:
        raise SnapshotMismatchError("environment MuJoCo data instance changed")

    mujoco = _require_mujoco()
    integration_spec, warmstart_spec = _state_specs(mujoco)
    _validate_task_fingerprint(env, snapshot.task_fingerprint)
    task_values = tuple(
        (attribute, _thaw_value(attribute.value))
        for attribute in snapshot.task_fingerprint.attributes
        if attribute.present
    )
    _validate_model(mujoco, model, snapshot, integration_spec, warmstart_spec)
    for attribute in snapshot.runtime_attributes:
        if not hasattr(env, attribute.name):
            raise SnapshotMismatchError(
                f"missing runtime attribute at restore: {attribute.name}"
            )
    rng_targets = _validate_environment_rng(env, snapshot.environment_rng)
    process_torch = (
        _validate_process_rng(snapshot.process_rng)
        if restore_process_rng and snapshot.process_rng is not None
        else None
    )

    # MuJoCo restore order is significant: forward overwrites qacc_warmstart.
    for name in _MODEL_ARRAY_NAMES:
        np.copyto(
            np.asarray(getattr(model, name)),
            getattr(snapshot.model_arrays, name),
            casting="no",
        )
    mujoco.mj_setState(
        model,
        data,
        np.array(snapshot.integration_state, copy=True),
        integration_spec,
    )
    mujoco.mj_forward(model, data)
    mujoco.mj_setState(
        model,
        data,
        np.array(snapshot.warmstart_state, copy=True),
        warmstart_spec,
    )

    restored_integration = _capture_mujoco_state(
        mujoco, model, data, integration_spec
    )
    restored_warmstart = _capture_mujoco_state(
        mujoco, model, data, warmstart_spec
    )
    _assert_restored_array(
        "mjSTATE_INTEGRATION", snapshot.integration_state, restored_integration
    )
    _assert_restored_array(
        "mjSTATE_WARMSTART", snapshot.warmstart_state, restored_warmstart
    )
    for name in _MODEL_ARRAY_NAMES:
        _assert_restored_array(
            f"model.{name}",
            getattr(snapshot.model_arrays, name),
            np.asarray(getattr(model, name)),
        )

    _restore_task_attributes(env, task_values)
    for attribute in snapshot.runtime_attributes:
        setattr(env, attribute.name, _thaw_value(attribute.value))
    for target, saved in zip(
        rng_targets,
        (
            snapshot.environment_rng.env,
            snapshot.environment_rng.action_space,
            snapshot.environment_rng.observation_space,
        ),
    ):
        if target is not None and saved is not None:
            target.state = _thaw_value(saved.state)
    if restore_process_rng and snapshot.process_rng is not None:
        _restore_process_rng(snapshot.process_rng, process_torch)


# ``save`` is a convenient spelling for callers that pair save/restore explicitly.
save_metaworld_snapshot = capture_metaworld_snapshot


__all__ = [
    "AttributeSnapshot",
    "BitGeneratorSnapshot",
    "EnvironmentRNGSnapshot",
    "MetaWorldSnapshot",
    "ModelArraysSnapshot",
    "ModelSignature",
    "ProcessRNGSnapshot",
    "SnapshotMismatchError",
    "TaskFingerprint",
    "capture_metaworld_snapshot",
    "capture_process_rng_snapshot",
    "restore_metaworld_snapshot",
    "restore_process_rng_snapshot",
    "save_metaworld_snapshot",
]
