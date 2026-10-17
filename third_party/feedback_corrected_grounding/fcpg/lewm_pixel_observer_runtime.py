"""Independent training and loading runtime for the LeWM pixel observer adapter.

The base :class:`GroundedLeWM` is immutable.  Training sees only cached FCPG
measurements and minimizes the masked error of the next pixel residual,
``q_corrected - q_nominal``.  Oracle coordinates and oracle latent targets are
rejected at the configuration and batch boundaries.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from fcpg.lewm_pixel_observer import (
    PIXEL_OBSERVER_CHECKPOINT_TYPE,
    LeWMPixelInnovationObserver,
    PixelResidualPriorAdapter,
    fixed_gain_update,
)
from fcpg.lewm_provenance import sha256_file
from fcpg.lewm_runtime import (
    LeWMDataBundle,
    build_lewm_data,
    choose_device,
    cosine_warmup_lambda,
    load_lewm_grounded_checkpoint,
    move_batch_to_device,
    resolve_runtime_config,
    seed_everything,
)


DEFAULT_PIXEL_OBSERVER_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "seed": 6144,
    "output_dir": "outputs/lewm_pixel_observer/adapter_dev_seed6144",
    "base_checkpoint": None,
    "data": {
        "sources": [],
        "history_size": 3,
        "rollout_steps": 1,
        "frame_skip": 5,
        "grounding_mode": "fcpg",
        "target_mode": "online",
        "split_seed": 3072,
        "split_seed_stride": 0,
        "episode_manifest": None,
        "require_corrector_split_match": False,
        "normalize_inputs": True,
        "val_fraction": 0.2,
        "balanced_sources": True,
        "require_pre_action_alignment": True,
    },
    "loader": {
        "batch_size": 128,
        "num_workers": 0,
        "pin_memory": True,
        "persistent_workers": False,
        "prefetch_factor": 2,
    },
    "adapter": {
        "hidden_dim": 256,
        "max_delta_px": 16.0,
        "residual_scale_px": 16.0,
    },
    "observer": {"gain": 0.5},
    "optimizer": {
        "lr": 3.0e-4,
        "weight_decay": 1.0e-4,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "warmup_steps": 100,
    },
    "trainer": {
        "epochs": 100,
        "max_steps": 2000,
        "max_train_batches": 0,
        "max_val_batches": 0,
        "grad_clip": 1.0,
        "amp": True,
        "amp_dtype": "bfloat16",
    },
}


@dataclass(frozen=True)
class PixelObserverLoss:
    loss: torch.Tensor
    hold_last_loss: torch.Tensor
    prior_residual_px: torch.Tensor
    target_residual_px: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class PixelObserverTrainingResult:
    output_dir: Path
    latest_checkpoint: Path
    best_checkpoint: Path
    metrics_path: Path
    global_step: int
    epoch: int
    best_val_prior_mse: float


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _assert_oracle_free_config(config: Mapping[str, Any]) -> None:
    data = _require_mapping(config.get("data"), "data")
    if str(data.get("target_mode", "online")).lower() != "online":
        raise ValueError("pixel observer training requires data.target_mode='online'")
    if str(data.get("grounding_mode", "")).lower() != "fcpg":
        raise ValueError("pixel observer training requires cached FCPG grounding")
    for index, source in enumerate(data.get("sources", [])):
        source = _require_mapping(source, f"data.sources[{index}]")
        if source.get("oracle_target_cache") not in (None, ""):
            raise ValueError("oracle_target_cache is forbidden for pixel observer training")
        if not source.get("correction_cache"):
            raise ValueError(
                f"data.sources[{index}] requires correction_cache with q_corrected"
            )
    forbidden_keys = {"q_true", "q_oracle", "oracle_grounding"}

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                key_text = str(key).lower()
                if key_text in forbidden_keys:
                    raise ValueError(f"oracle field is forbidden in config: {path}.{key}")
                visit(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(config, "config")


def _base_supervision_audit(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Reject a base LeWM whose representation used privileged oracle targets."""

    mode_records: list[dict[str, str]] = []
    for path, container in (
        ("resolved_config.data", checkpoint.get("resolved_config", {}).get("data", {})),
        ("data_config", checkpoint.get("data_config", {})),
        (
            "model_config.world_model",
            checkpoint.get("model_config", {}).get("world_model", {}),
        ),
    ):
        if isinstance(container, Mapping):
            value = container.get("target_mode")
            if value not in (None, ""):
                mode_records.append({"path": path, "value": str(value).lower()})

    oracle_artifacts: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                child = f"{path}.{key}"
                normalized = str(key).lower()
                if normalized == "oracle_target_cache" and item not in (None, ""):
                    oracle_artifacts.append(child)
                if normalized == "q_true_usage" and item not in (None, ""):
                    oracle_artifacts.append(child)
                visit(item, child)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    for name in ("resolved_config", "data_config", "model_config"):
        visit(checkpoint.get(name, {}), name)

    modes = {record["value"] for record in mode_records}
    if "frozen_oracle" in modes or oracle_artifacts:
        raise ValueError(
            "pixel observer base must be oracle-free; found privileged target "
            f"evidence modes={sorted(modes)} artifacts={sorted(set(oracle_artifacts))}"
        )
    unexpected = modes.difference({"online"})
    if unexpected:
        raise ValueError(
            f"pixel observer base has unsupported target modes: {sorted(unexpected)}"
        )
    return {
        "passed": True,
        "target": "online_self_target",
        "evidence": "explicit" if mode_records else "legacy_absence_of_oracle_artifacts",
        "mode_records": mode_records,
        "oracle_artifacts": [],
    }


def validate_completed_training_budget(
    checkpoint: Mapping[str, Any],
    *,
    label: str,
    expected_target_steps: int | None = None,
    require_endpoint: bool = False,
) -> dict[str, Any]:
    budget = dict(
        _require_mapping(checkpoint.get("training_budget"), f"{label} training_budget")
    )
    try:
        top_level_step = int(checkpoint.get("step", -1))
        target = int(budget.get("target_optimizer_steps", -1))
        checkpoint_step = int(budget.get("checkpoint_optimizer_step", -1))
        final = int(budget.get("final_optimizer_steps", -1))
    except (TypeError, ValueError):
        raise ValueError(
            f"{label} lacks a valid completed training budget"
        ) from None
    valid = (
        budget.get("run_completed") is True
        and target > 0
        and 0 <= checkpoint_step <= final
        and final == target
        and checkpoint_step == top_level_step
    )
    if require_endpoint:
        valid = valid and checkpoint_step == final
    if expected_target_steps is not None:
        valid = valid and target == int(expected_target_steps)
    if not valid:
        raise ValueError(f"{label} lacks a valid completed training budget")
    return budget


def _validate_base_checkpoint_contract(
    resolved: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    supervision: Mapping[str, Any],
) -> None:
    contract = _require_mapping(resolved.get("contract", {}), "contract")
    if contract.get("require_formal_base") is True:
        if checkpoint.get("run_kind") != "formal":
            raise ValueError("pixel observer formal run requires a formal base checkpoint")
    if contract.get("require_explicit_online_base") is True:
        if supervision.get("evidence") != "explicit":
            raise ValueError("pixel observer formal run requires explicit online base metadata")
    expected_step = contract.get("expected_base_step")
    if expected_step is not None and int(checkpoint.get("step", -1)) != int(expected_step):
        raise ValueError(
            "pixel observer base checkpoint step differs from its frozen contract"
        )
    if contract.get("require_completed_base_budget") is True:
        validate_completed_training_budget(
            checkpoint,
            label="pixel observer base",
            expected_target_steps=(
                int(expected_step) if expected_step is not None else None
            ),
            require_endpoint=True,
        )


def _assert_independent_output_dir(path: Path) -> None:
    offending = [part for part in path.parts if "gate_l1" in part.lower()]
    if offending:
        raise ValueError(
            "pixel observer checkpoints must not be written into legacy Gate L1 "
            f"directories: {path}"
        )


def resolve_pixel_observer_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve defaults and enforce the adapter-only supervision contract."""

    merged = _merge(DEFAULT_PIXEL_OBSERVER_CONFIG, _plain(config))
    # Reuse the exact LeWM data validation/defaults consumed by build_lewm_data.
    resolved = resolve_runtime_config(merged)
    if not resolved.get("base_checkpoint"):
        raise ValueError("base_checkpoint is required")
    _assert_oracle_free_config(resolved)
    output_dir = Path(str(resolved["output_dir"])).expanduser().resolve()
    _assert_independent_output_dir(output_dir)
    resolved["output_dir"] = str(output_dir)
    resolved["base_checkpoint"] = str(
        Path(str(resolved["base_checkpoint"])).expanduser().resolve()
    )

    adapter = _require_mapping(resolved.get("adapter"), "adapter")
    for name in ("hidden_dim", "max_delta_px", "residual_scale_px"):
        if float(adapter[name]) <= 0:
            raise ValueError(f"adapter.{name} must be positive")
    gain = float(_require_mapping(resolved.get("observer"), "observer")["gain"])
    if not 0.0 <= gain <= 1.0:
        raise ValueError("observer.gain must lie in [0, 1]")
    trainer = _require_mapping(resolved.get("trainer"), "trainer")
    if int(trainer["epochs"]) <= 0 or int(trainer.get("max_steps", 0)) < 0:
        raise ValueError("trainer epochs must be positive and max_steps non-negative")
    return resolved


def _amp_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"unsupported trainer.amp_dtype: {name}")


def _forbid_oracle_batch(batch: Mapping[str, Any]) -> None:
    forbidden = sorted(
        name
        for name in ("q_true", "q_oracle", "oracle_grounding", "oracle_target")
        if name in batch
    )
    if forbidden:
        raise ValueError(
            "pixel observer supervision is restricted to q_corrected-q_nominal; "
            f"forbidden batch fields: {forbidden}"
        )


def _history_batch(
    batch: Mapping[str, Any], history_size: int
) -> dict[str, torch.Tensor]:
    frame_names = {
        "pixels",
        "rgb",
        "obs",
        "proprio",
        "q_nominal",
        "q_corrected",
        "q_grounded",
        "keypoints_xyz",
        "valid",
        "valid_nominal",
    }
    return {
        name: value[:, :history_size]
        for name, value in batch.items()
        if name in frame_names and torch.is_tensor(value)
    }


def _masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    if prediction.shape != target.shape or valid.shape != prediction.shape[:-1]:
        raise ValueError("pixel observer target/mask shapes are inconsistent")
    selected = valid.bool().unsqueeze(-1).expand_as(prediction)
    difference = torch.where(
        selected, prediction - target, torch.zeros_like(prediction)
    )
    if not bool(torch.isfinite(difference).all()):
        raise ValueError("pixel observer selected target values must be finite")
    weights = selected.to(dtype=prediction.dtype)
    denominator = weights.sum()
    if int(denominator.detach().item()) == 0:
        raise ValueError("pixel observer batch contains no valid target anchors")
    return difference.square().sum() / denominator


def pixel_observer_loss(
    observer: LeWMPixelInnovationObserver,
    batch: Mapping[str, Any],
) -> PixelObserverLoss:
    """Compute the adapter-only next-residual objective.

    The target frame contributes only ``q_corrected-q_nominal`` and its validity
    mask.  Its RGB, proprioception, and any oracle value cannot enter the prior.
    """

    _forbid_oracle_batch(batch)
    q_nominal = batch.get("q_nominal")
    q_corrected = batch.get("q_corrected")
    keypoints_xyz = batch.get("keypoints_xyz")
    valid = batch.get("valid")
    actions = batch.get("action", batch.get("actions"))
    tensors = {
        "q_nominal": q_nominal,
        "q_corrected": q_corrected,
        "keypoints_xyz": keypoints_xyz,
        "valid": valid,
        "action": actions,
    }
    missing = [name for name, value in tensors.items() if not torch.is_tensor(value)]
    if missing:
        raise KeyError(f"pixel observer batch is missing tensor fields: {missing}")
    assert isinstance(q_nominal, torch.Tensor)
    assert isinstance(q_corrected, torch.Tensor)
    assert isinstance(keypoints_xyz, torch.Tensor)
    assert isinstance(valid, torch.Tensor)
    assert isinstance(actions, torch.Tensor)
    history = observer.history_size
    batch_size = q_nominal.shape[0]
    expected_q = (batch_size, history + 1, observer.num_keypoints, 2)
    if q_nominal.shape != expected_q or q_corrected.shape != expected_q:
        raise ValueError(
            f"q_nominal/q_corrected must have shape {expected_q}; got "
            f"{tuple(q_nominal.shape)} and {tuple(q_corrected.shape)}"
        )
    if keypoints_xyz.shape != (batch_size, history + 1, observer.num_keypoints, 3):
        raise ValueError("keypoints_xyz has an incompatible observer sequence shape")
    if valid.shape != expected_q[:-1]:
        raise ValueError("valid has an incompatible observer sequence shape")
    if actions.shape != (batch_size, history, observer.base_model.action_dim):
        raise ValueError("action must contain exactly one transition per frame pair")

    measurement = q_corrected.float() - q_nominal.float()
    history_input = _history_batch(batch, history)
    history_posteriors: list[torch.Tensor] = []
    history_posterior = torch.zeros_like(measurement[:, 0])
    for history_index in range(history):
        _, history_posterior, _ = fixed_gain_update(
            history_posterior,
            measurement[:, history_index],
            1.0,
            valid[:, history_index].bool(),
        )
        history_posteriors.append(history_posterior)
    history_posterior_sequence = torch.stack(history_posteriors, dim=1)
    q_history_posterior = q_nominal[:, :history].float() + history_posterior_sequence
    history_input["q_corrected"] = q_history_posterior
    if "q_grounded" in history_input:
        history_input["q_grounded"] = q_history_posterior
    posterior_valid = observer._posterior_valid(history_input, q_history_posterior)
    history_input["valid"] = posterior_valid
    # The frozen model produces a constant conditioning signal.  Keeping it out
    # of autograd is both cheaper and an enforceable guarantee that only the
    # adapter can update.
    with torch.no_grad():
        _, history_embedding = observer.base_model.encode_frames(history_input)
        predicted_embedding = observer.base_model.predict(
            history_embedding,
            actions[:, :history].float(),
        )[:, -1]
    prior = observer.adapter(
        predicted_embedding,
        history_posterior_sequence[:, -1],
        q_nominal[:, history - 1].float(),
        keypoints_xyz[:, history - 1].float(),
        posterior_valid[:, -1],
    )
    target = measurement[:, history]
    target_valid = valid[:, history].bool()
    loss = _masked_mse(prior, target, target_valid)
    hold_last_loss = _masked_mse(
        history_posterior_sequence[:, -1], target, target_valid
    )
    return PixelObserverLoss(
        loss=loss,
        hold_last_loss=hold_last_loss,
        prior_residual_px=prior,
        target_residual_px=target,
        valid=target_valid,
    )


def _geometry_stats(bundle: LeWMDataBundle) -> tuple[list[float], list[float]]:
    total = np.zeros(3, dtype=np.float64)
    square = np.zeros(3, dtype=np.float64)
    count = 0
    for subset in bundle.train_sources:
        dataset = subset.dataset
        raw_indices: set[int] = set()
        for window_index in subset.indices:
            episode_index, start = dataset.windows[int(window_index)]
            sequence = dataset.episodes[episode_index]
            positions = start + np.arange(dataset.num_state_frames) * dataset.frame_skip
            raw_indices.update(int(value) for value in sequence[positions])
        indices = np.asarray(sorted(raw_indices), dtype=np.int64)
        xyz = np.asarray(dataset.keypoints_xyz[indices], dtype=np.float64).reshape(-1, 3)
        finite = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite]
        total += xyz.sum(axis=0)
        square += np.square(xyz).sum(axis=0)
        count += int(xyz.shape[0])
    if count == 0:
        raise ValueError("cannot compute adapter geometry normalization")
    mean = total / count
    variance = np.maximum(square / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    return mean.tolist(), std.tolist()


def _artifact_record(path: Path, kind: str, source: str) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "kind": kind,
        "source": source,
        "path": str(resolved),
        "sha256": sha256_file(resolved),
    }


def _data_artifacts(bundle: LeWMDataBundle) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(path: Path, kind: str, source: str) -> None:
        key = (str(path.expanduser().resolve()), kind)
        if key not in seen:
            records.append(_artifact_record(path, kind, source))
            seen.add(key)

    for dataset in bundle.base_datasets:
        source = str(dataset.source_name)
        add(dataset.path, "source_data", source)
        if dataset.correction_cache_path is None:
            raise ValueError(f"source {source!r} has no FCPG correction cache")
        cache_array = Path(dataset.correction_cache_path)
        add(cache_array, "q_corrected", source)
        metadata_path = cache_array.parent / "metadata.json"
        if dataset.correction_metadata and metadata_path.is_file():
            add(metadata_path, "correction_metadata", source)
        checkpoint_value = dataset.correction_metadata.get("checkpoint")
        if checkpoint_value:
            checkpoint_path = Path(str(checkpoint_value)).expanduser()
            if not checkpoint_path.is_absolute():
                checkpoint_path = metadata_path.parent / checkpoint_path
            if checkpoint_path.is_file():
                add(checkpoint_path, "corrector_checkpoint", source)
    episode = bundle.data_config.get("episode_manifest")
    if isinstance(episode, Mapping) and episode.get("path"):
        add(Path(str(episode["path"])), "episode_manifest", "shared")
    return records


def _verify_artifacts(records: Sequence[Mapping[str, Any]]) -> None:
    for record in records:
        path = Path(str(record.get("path", "")))
        expected = str(record.get("sha256", ""))
        if not path.is_file():
            raise FileNotFoundError(f"bound artifact is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"bound artifact changed ({record.get('kind')}): {path}; "
                f"expected {expected}, got {actual}"
            )


def _config_binding(
    resolved: Mapping[str, Any],
    snapshot_path: str | Path,
    origin_path: str | Path | None,
) -> dict[str, Any]:
    snapshot = Path(snapshot_path).expanduser().resolve()
    binding: dict[str, Any] = {
        "resolved_config_sha256": _canonical_sha256(resolved),
        "snapshot_path": str(snapshot),
        "snapshot_sha256": sha256_file(snapshot),
        "origin_path": None,
        "origin_sha256_at_start": None,
    }
    if origin_path is not None:
        path = Path(origin_path).expanduser().resolve()
        binding["origin_path"] = str(path)
        binding["origin_sha256_at_start"] = sha256_file(path)
    return binding


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _run_lock(output_dir: Path):
    import fcntl

    lock_path = output_dir / ".pixel_observer.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another pixel observer run holds {lock_path}") from exc
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _observer_metrics(
    output: PixelObserverLoss,
) -> tuple[float, float, float, int]:
    valid = output.valid
    count = int(valid.sum().item())
    if count == 0:
        raise ValueError("pixel observer metrics contain no valid target anchors")
    weights = valid.unsqueeze(-1).expand_as(output.prior_residual_px)
    prior_sse = float(
        (
            (output.prior_residual_px - output.target_residual_px).square()
            * weights
        ).sum().detach()
    )
    previous = output.target_residual_px.new_tensor(float(output.hold_last_loss))
    hold_sse = float((previous * (2 * count)).detach())
    error = torch.linalg.vector_norm(
        output.prior_residual_px - output.target_residual_px, dim=-1
    )
    l2 = float(error[valid].sum().detach())
    return prior_sse, hold_sse, l2, count


@torch.no_grad()
def evaluate_pixel_observer(
    observer: LeWMPixelInnovationObserver,
    loader: Any,
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    max_batches: int = 0,
) -> dict[str, float]:
    observer.eval()
    prior_sse = 0.0
    hold_sse = 0.0
    l2_sum = 0.0
    samples = 0
    valid_anchors = 0
    batches = 0
    for batch_index, host_batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        batch = move_batch_to_device(host_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            output = pixel_observer_loss(observer, batch)
        batch_size = int(output.prior_residual_px.shape[0])
        batch_prior_sse, batch_hold_sse, l2, count = _observer_metrics(output)
        prior_sse += batch_prior_sse
        hold_sse += batch_hold_sse
        l2_sum += l2
        samples += batch_size
        valid_anchors += count
        batches += 1
    if samples == 0:
        raise ValueError("pixel observer validation loader produced no samples")
    return {
        "prior_mse": prior_sse / (2 * valid_anchors),
        "hold_last_mse": hold_sse / (2 * valid_anchors),
        "prior_l2_px": l2_sum / max(valid_anchors, 1),
        "samples": float(samples),
        "valid_anchors": float(valid_anchors),
        "batches": float(batches),
    }


def _checkpoint_payload(
    *,
    observer: LeWMPixelInnovationObserver,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    resolved: Mapping[str, Any],
    data_config: Mapping[str, Any],
    base_binding: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
    config_binding: Mapping[str, Any],
    adapter_config: Mapping[str, Any],
    step: int,
    epoch: int,
    run_kind: str,
    metrics: Mapping[str, Any],
    best_val_prior_mse: float,
    target_optimizer_steps: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "checkpoint_type": PIXEL_OBSERVER_CHECKPOINT_TYPE,
        "objective": "masked_mse(next_residual_prior, q_corrected-q_nominal)",
        "supervision_contract": {
            "target": "q_corrected-q_nominal",
            "oracle_target_forbidden": True,
            "q_true_forbidden": True,
            "base_grounded_lewm_frozen": True,
        },
        "step": int(step),
        "epoch": int(epoch),
        "run_kind": str(run_kind),
        "training_budget": {
            "target_optimizer_steps": int(target_optimizer_steps),
            "checkpoint_optimizer_step": int(step),
            "run_completed": False,
            "final_optimizer_steps": None,
        },
        "base_checkpoint": _plain(base_binding),
        "data_artifacts": _plain(artifacts),
        "data_config": _plain(data_config),
        "data_config_sha256": _canonical_sha256(data_config),
        "config_binding": _plain(config_binding),
        "resolved_config": _plain(resolved),
        "adapter_config": _plain(adapter_config),
        "adapter_state": observer.adapter.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "metrics": _plain(metrics),
        "best_val_prior_mse": float(best_val_prior_mse),
    }


def _check_base_frozen(observer: LeWMPixelInnovationObserver) -> None:
    trainable = [
        name
        for name, parameter in observer.base_model.named_parameters()
        if parameter.requires_grad
    ]
    if trainable:
        raise RuntimeError(f"base GroundedLeWM is not frozen: {trainable[:5]}")


def _certify_completed_budget(
    checkpoint_path: Path,
    *,
    final_optimizer_steps: int,
    target_optimizer_steps: int,
) -> None:
    if final_optimizer_steps != target_optimizer_steps:
        raise RuntimeError(
            "cannot certify an incomplete pixel observer training budget: "
            f"{final_optimizer_steps}/{target_optimizer_steps} optimizer steps"
        )
    payload = _torch_load(checkpoint_path, map_location="cpu")
    budget = dict(_require_mapping(payload.get("training_budget"), "training_budget"))
    if int(budget.get("target_optimizer_steps", -1)) != target_optimizer_steps:
        raise ValueError("checkpoint optimizer-step budget differs from the run")
    budget["run_completed"] = True
    budget["final_optimizer_steps"] = int(final_optimizer_steps)
    payload["training_budget"] = budget
    _atomic_torch_save(payload, checkpoint_path)


def train_pixel_observer_adapter(
    config: Mapping[str, Any],
    *,
    device: str | torch.device = "auto",
    smoke_steps: int = 0,
    config_path: str | Path | None = None,
) -> PixelObserverTrainingResult:
    """Train only the pixel residual adapter and emit a bound checkpoint."""

    if smoke_steps < 0:
        raise ValueError("smoke_steps must be non-negative")
    resolved = resolve_pixel_observer_config(config)
    output_dir = Path(str(resolved["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with _run_lock(output_dir):
        existing = {
            path.name for path in output_dir.iterdir() if path.name != ".pixel_observer.lock"
        }
        if existing:
            raise FileExistsError(
                f"refusing to train in non-empty output directory {output_dir}: "
                f"{sorted(existing)}"
            )
        return _train_pixel_observer_locked(
            resolved,
            device=device,
            smoke_steps=smoke_steps,
            config_path=config_path,
        )


def _train_pixel_observer_locked(
    resolved: Mapping[str, Any],
    *,
    device: str | torch.device,
    smoke_steps: int,
    config_path: str | Path | None,
) -> PixelObserverTrainingResult:
    seed_everything(int(resolved["seed"]))
    target_device = choose_device(device)
    bundle = build_lewm_data(resolved)
    base_path = Path(str(resolved["base_checkpoint"])).resolve()
    base_sha = sha256_file(base_path)
    base_raw = _torch_load(base_path, map_location="cpu")
    base_supervision = _base_supervision_audit(base_raw)
    _validate_base_checkpoint_contract(resolved, base_raw, base_supervision)
    base_model, _, base_checkpoint = load_lewm_grounded_checkpoint(
        base_path, device=target_device, dataset=bundle.reference_dataset
    )
    if _base_supervision_audit(base_checkpoint) != base_supervision:
        raise ValueError("base supervision audit changed while loading the model")
    base_binding = {
        "path": str(base_path),
        "sha256": base_sha,
        "checkpoint_type": base_checkpoint.get("checkpoint_type"),
        "step": int(base_checkpoint.get("step", 0)),
        "run_kind": base_checkpoint.get("run_kind"),
        "supervision_audit": base_supervision,
        "training_budget": _plain(base_checkpoint.get("training_budget")),
        "provenance": _plain(base_checkpoint.get("provenance")),
    }
    adapter_values = _require_mapping(resolved["adapter"], "adapter")
    adapter_config = {
        "latent_dim": int(base_model.latent_dim),
        "num_keypoints": int(bundle.reference_dataset.num_keypoints),
        "image_size": list(bundle.reference_dataset.image_size),
        "hidden_dim": int(adapter_values["hidden_dim"]),
        "max_delta_px": float(adapter_values["max_delta_px"]),
        "residual_scale_px": float(adapter_values["residual_scale_px"]),
    }
    adapter = PixelResidualPriorAdapter(
        latent_dim=adapter_config["latent_dim"],
        num_keypoints=adapter_config["num_keypoints"],
        image_size=tuple(adapter_config["image_size"]),
        hidden_dim=adapter_config["hidden_dim"],
        max_delta_px=adapter_config["max_delta_px"],
        residual_scale_px=adapter_config["residual_scale_px"],
    )
    xyz_mean, xyz_std = _geometry_stats(bundle)
    adapter.set_geometry_normalization(xyz_mean=xyz_mean, xyz_std=xyz_std)
    adapter_config["xyz_mean"] = xyz_mean
    adapter_config["xyz_std"] = xyz_std
    observer = LeWMPixelInnovationObserver(
        base_model,
        adapter,
        gain=float(_require_mapping(resolved["observer"], "observer")["gain"]),
        freeze_base=True,
    ).to(target_device)
    _check_base_frozen(observer)

    optimizer_values = _require_mapping(resolved["optimizer"], "optimizer")
    optimizer = torch.optim.AdamW(
        observer.adapter.parameters(),
        lr=float(optimizer_values["lr"]),
        weight_decay=float(optimizer_values["weight_decay"]),
        betas=tuple(float(value) for value in optimizer_values["betas"]),
        eps=float(optimizer_values["eps"]),
    )
    trainer = _require_mapping(resolved["trainer"], "trainer")
    max_train_batches = int(trainer.get("max_train_batches", 0))
    batches_per_epoch = len(bundle.train_loader)
    if max_train_batches > 0:
        batches_per_epoch = min(batches_per_epoch, max_train_batches)
    configured_steps = int(trainer.get("max_steps", 0))
    effective_steps = int(smoke_steps) if smoke_steps > 0 else configured_steps
    total_steps = effective_steps or max(1, int(trainer["epochs"]) * batches_per_epoch)
    warmup = min(int(optimizer_values.get("warmup_steps", 0)), total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_warmup_lambda(step, warmup, total_steps)
    )
    amp_dtype = _amp_dtype(str(trainer.get("amp_dtype", "bfloat16")))
    amp_enabled = bool(trainer.get("amp", True)) and target_device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16
    )
    max_val_batches = int(trainer.get("max_val_batches", 0))
    if smoke_steps > 0:
        max_val_batches = 1 if max_val_batches <= 0 else min(1, max_val_batches)
    grad_clip = float(trainer.get("grad_clip", 1.0))

    artifacts = _data_artifacts(bundle)
    output_dir = Path(str(resolved["output_dir"])).resolve()
    latest_path = output_dir / "lewm_pixel_observer_latest.pt"
    best_path = output_dir / "lewm_pixel_observer_best.pt"
    metrics_path = output_dir / "metrics.jsonl"
    resolved_snapshot = output_dir / "resolved_config.json"
    _atomic_write_text(
        resolved_snapshot,
        json.dumps(_plain(resolved), indent=2, sort_keys=True) + "\n",
    )
    config_record = _config_binding(resolved, resolved_snapshot, config_path)

    global_step = 0
    best_val = math.inf
    last_epoch = 0
    records: list[dict[str, Any]] = []
    run_kind = "smoke" if smoke_steps > 0 else "formal"
    for epoch in range(1, int(trainer["epochs"]) + 1):
        if effective_steps > 0 and global_step >= effective_steps:
            break
        observer.train()
        _check_base_frozen(observer)
        train_prior_sse = 0.0
        train_hold_sse = 0.0
        train_samples = 0
        train_valid_anchors = 0
        started = time.perf_counter()
        for batch_index, host_batch in enumerate(bundle.train_loader):
            if max_train_batches > 0 and batch_index >= max_train_batches:
                break
            if effective_steps > 0 and global_step >= effective_steps:
                break
            batch = move_batch_to_device(host_batch, target_device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=target_device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                output = pixel_observer_loss(observer, batch)
            scaler.scale(output.loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(observer.adapter.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            batch_size = int(output.prior_residual_px.shape[0])
            prior_sse, hold_sse, _, valid_count = _observer_metrics(output)
            train_prior_sse += prior_sse
            train_hold_sse += hold_sse
            train_samples += batch_size
            train_valid_anchors += valid_count
        if train_samples == 0:
            break
        last_epoch = epoch
        val = evaluate_pixel_observer(
            observer,
            bundle.val_loader,
            target_device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            max_batches=max_val_batches,
        )
        train_metrics = {
            "prior_mse": train_prior_sse / (2 * train_valid_anchors),
            "hold_last_mse": train_hold_sse / (2 * train_valid_anchors),
            "samples": float(train_samples),
            "valid_anchors": float(train_valid_anchors),
            "seconds": time.perf_counter() - started,
        }
        record = {
            "epoch": epoch,
            "step": global_step,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "objective": "q_corrected_minus_q_nominal_only",
            "train": train_metrics,
            "val": val,
        }
        if not all(
            math.isfinite(float(value))
            for value in (train_metrics["prior_mse"], val["prior_mse"])
        ):
            raise FloatingPointError("pixel observer training produced non-finite loss")
        improved = float(val["prior_mse"]) < best_val
        if improved:
            best_val = float(val["prior_mse"])
        _verify_artifacts(artifacts)
        if sha256_file(base_path) != base_sha:
            raise ValueError("base GroundedLeWM checkpoint changed during adapter training")
        payload = _checkpoint_payload(
            observer=observer,
            optimizer=optimizer,
            scheduler=scheduler,
            resolved=resolved,
            data_config=bundle.data_config,
            base_binding=base_binding,
            artifacts=artifacts,
            config_binding=config_record,
            adapter_config=adapter_config,
            step=global_step,
            epoch=epoch,
            run_kind=run_kind,
            metrics=record,
            best_val_prior_mse=best_val,
            target_optimizer_steps=total_steps,
        )
        _atomic_torch_save(payload, latest_path)
        if improved:
            _atomic_torch_save(payload, best_path)
        records.append(record)
        _atomic_write_text(
            metrics_path,
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
        )
    if global_step == 0 or last_epoch == 0:
        raise RuntimeError("pixel observer training completed no optimizer steps")
    if global_step != total_steps:
        raise RuntimeError(
            "pixel observer training exhausted its epoch budget before the frozen "
            f"optimizer-step budget: {global_step}/{total_steps}"
        )
    for checkpoint_path in (latest_path, best_path):
        _certify_completed_budget(
            checkpoint_path,
            final_optimizer_steps=global_step,
            target_optimizer_steps=total_steps,
        )
    return PixelObserverTrainingResult(
        output_dir=output_dir,
        latest_checkpoint=latest_path,
        best_checkpoint=best_path,
        metrics_path=metrics_path,
        global_step=global_step,
        epoch=last_epoch,
        best_val_prior_mse=best_val,
    )


def _torch_load(path: Path, map_location: Any) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise TypeError(f"pixel observer checkpoint must be a mapping: {path}")
    return payload


def load_pixel_observer_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    base_checkpoint: str | Path | None = None,
    dataset: Any | None = None,
    verify_artifacts: bool = True,
) -> tuple[LeWMPixelInnovationObserver, dict[str, Any], dict[str, Any]]:
    """Load an adapter and verify its immutable base/data/cache/config bindings."""

    target_device = choose_device(device)
    checkpoint_path = Path(path).expanduser().resolve()
    checkpoint = _torch_load(checkpoint_path, map_location=target_device)
    if checkpoint.get("checkpoint_type") != PIXEL_OBSERVER_CHECKPOINT_TYPE:
        raise ValueError(
            f"expected checkpoint_type={PIXEL_OBSERVER_CHECKPOINT_TYPE!r}, got "
            f"{checkpoint.get('checkpoint_type')!r}"
        )
    resolved = _require_mapping(checkpoint.get("resolved_config"), "resolved_config")
    binding = _require_mapping(checkpoint.get("config_binding"), "config_binding")
    if binding.get("resolved_config_sha256") != _canonical_sha256(resolved):
        raise ValueError("pixel observer resolved config binding is invalid")
    snapshot_path = binding.get("snapshot_path")
    if not snapshot_path:
        raise ValueError("pixel observer checkpoint is missing its config snapshot")
    snapshot = Path(str(snapshot_path))
    if (
        not snapshot.is_file()
        or sha256_file(snapshot) != binding.get("snapshot_sha256")
    ):
        raise ValueError("pixel observer resolved-config snapshot changed after training")
    data_config = _require_mapping(checkpoint.get("data_config"), "data_config")
    if checkpoint.get("data_config_sha256") != _canonical_sha256(data_config):
        raise ValueError("pixel observer data config binding is invalid")
    if verify_artifacts:
        records = checkpoint.get("data_artifacts")
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise TypeError("pixel observer data_artifacts must be an array")
        _verify_artifacts(records)

    base_binding = _require_mapping(checkpoint.get("base_checkpoint"), "base_checkpoint")
    base_path = (
        Path(base_checkpoint).expanduser().resolve()
        if base_checkpoint is not None
        else Path(str(base_binding.get("path", ""))).expanduser().resolve()
    )
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    base_sha = sha256_file(base_path)
    if base_sha != base_binding.get("sha256"):
        raise ValueError(
            "base GroundedLeWM checkpoint SHA-256 differs from adapter binding"
        )
    base_raw = _torch_load(base_path, map_location="cpu")
    current_supervision = _base_supervision_audit(base_raw)
    base_model, _, base_payload = load_lewm_grounded_checkpoint(
        base_path, device=target_device, dataset=dataset
    )
    if _base_supervision_audit(base_payload) != current_supervision:
        raise ValueError("base supervision audit changed while loading the model")
    if base_binding.get("supervision_audit") != current_supervision:
        raise ValueError("base GroundedLeWM supervision audit differs from binding")
    _validate_base_checkpoint_contract(resolved, base_payload, current_supervision)
    actual_binding = {
        "checkpoint_type": base_payload.get("checkpoint_type"),
        "step": int(base_payload.get("step", 0)),
        "run_kind": base_payload.get("run_kind"),
        "supervision_audit": current_supervision,
        "training_budget": _plain(base_payload.get("training_budget")),
        "provenance": _plain(base_payload.get("provenance")),
    }
    bound_contract = {
        key: _plain(base_binding.get(key)) for key in actual_binding
    }
    if actual_binding != bound_contract:
        raise ValueError(
            "actual base GroundedLeWM contract differs from adapter binding"
        )
    adapter_config = dict(
        _require_mapping(checkpoint.get("adapter_config"), "adapter_config")
    )
    xyz_mean = adapter_config.pop("xyz_mean")
    xyz_std = adapter_config.pop("xyz_std")
    adapter_config["image_size"] = tuple(adapter_config["image_size"])
    adapter = PixelResidualPriorAdapter(**adapter_config)
    adapter.set_geometry_normalization(xyz_mean=xyz_mean, xyz_std=xyz_std)
    adapter.load_state_dict(checkpoint["adapter_state"], strict=True)
    observer = LeWMPixelInnovationObserver(
        base_model,
        adapter,
        gain=float(_require_mapping(resolved["observer"], "observer")["gain"]),
        freeze_base=True,
    ).to(target_device)
    observer.eval()
    _check_base_frozen(observer)
    return observer, dict(resolved), checkpoint
