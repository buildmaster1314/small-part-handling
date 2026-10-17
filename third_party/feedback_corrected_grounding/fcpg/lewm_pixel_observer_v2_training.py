"""Train only the PixelObserver v2 adapter around a frozen GroundedLeWM."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping
import uuid

import numpy as np
import torch

from fcpg.lewm_pixel_observer_v2 import (
    ORACLE_CEILING_TARGET_MODE,
    PAIRED_DEPLOYMENT_TARGET_MODE,
    SINGLE_MEASUREMENT_BASELINE_TARGET_MODE,
    PixelObserverV2Supervision,
    PixelResidualVelocityInnovationAdapter,
    build_pixel_observer_v2_paired_supervision,
    build_pixel_observer_v2_supervision,
    masked_huber_pixel_loss,
    paired_mutual_huber_pixel_loss,
)
from fcpg.lewm_pixel_observer_v2_data import (
    PixelObserverV2DataBundle,
    build_pixel_observer_v2_data,
)
from fcpg.lewm_pixel_observer_v2_runtime import (
    CausalLeWMPixelObserverV2Runtime,
    canonical_sha256,
    load_pixel_observer_v2_checkpoint,
    save_pixel_observer_v2_checkpoint,
)
from fcpg.lewm_provenance import sha256_file
from fcpg.lewm_runtime import (
    choose_device,
    cosine_warmup_lambda,
    load_lewm_grounded_checkpoint,
    move_batch_to_device,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "seed": 7168,
    "target_mode": SINGLE_MEASUREMENT_BASELINE_TARGET_MODE,
    "base_checkpoint": None,
    "output_dir": "outputs/lewm_pixel_observer_v2/dev",
    "data": {
        "sources": [],
        "history_size": 3,
        "frame_skip": 5,
        "val_fraction": 0.2,
        "split_seed": 3072,
        "history_tail_mask": {
            "enabled": False,
            "probability": 0.0,
            "max_steps": 0,
            "seed": 0,
        },
    },
    "loader": {"batch_size": 128, "num_workers": 0, "pin_memory": True},
    "adapter": {
        "hidden_dim": 256,
        "max_velocity_innovation_px": 24.0,
        "residual_scale_px": 8.0,
        "velocity_scale_px": 0.5,
        "use_predicted_latent": True,
    },
    "loss": {"huber_delta_px": 0.25},
    "optimizer": {
        "lr": 3.0e-4,
        "weight_decay": 1.0e-4,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "warmup_steps": 100,
    },
    "trainer": {"epochs": 100, "max_steps": 2000, "grad_clip": 1.0},
}


@dataclass(frozen=True)
class V2LossOutput:
    loss: torch.Tensor
    predictions: tuple[torch.Tensor, ...]
    target_residual_px: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class V2TrainingResult:
    output_dir: Path
    latest_checkpoint: Path
    best_checkpoint: Path
    metrics_path: Path
    global_step: int
    best_val_loss: float


def _merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_v2_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    resolved = _merge(DEFAULT_CONFIG, config)
    mode = str(resolved["target_mode"])
    if mode not in {
        ORACLE_CEILING_TARGET_MODE,
        SINGLE_MEASUREMENT_BASELINE_TARGET_MODE,
        PAIRED_DEPLOYMENT_TARGET_MODE,
    }:
        raise ValueError("invalid target_mode")
    if not resolved.get("base_checkpoint"):
        raise ValueError("base_checkpoint is required")
    resolved["base_checkpoint"] = str(
        Path(str(resolved["base_checkpoint"])).expanduser().resolve()
    )
    resolved["output_dir"] = str(
        Path(str(resolved["output_dir"])).expanduser().resolve()
    )
    if int(resolved["data"]["history_size"]) < 2:
        raise ValueError("data.history_size must be >= 2")
    if int(resolved["data"]["frame_skip"]) <= 0:
        raise ValueError("data.frame_skip must be positive")
    if int(resolved["trainer"]["max_steps"]) <= 0:
        raise ValueError("trainer.max_steps must be positive")
    if int(resolved["trainer"]["epochs"]) <= 0:
        raise ValueError("trainer.epochs must be positive")
    tail = resolved["data"].setdefault("history_tail_mask", {})
    tail["enabled"] = bool(tail.get("enabled", False))
    tail["probability"] = float(tail.get("probability", 0.0))
    tail["max_steps"] = int(tail.get("max_steps", 0))
    tail["seed"] = int(tail.get("seed", 0))
    if not 0.0 <= tail["probability"] <= 1.0:
        raise ValueError("history-tail probability must lie in [0,1]")
    if tail["enabled"] and not 1 <= tail["max_steps"] < int(resolved["data"]["history_size"]):
        raise ValueError("enabled history-tail max_steps must lie in [1,H-1]")
    for source in resolved["data"]["sources"]:
        if mode == PAIRED_DEPLOYMENT_TARGET_MODE and not source.get(
            "correction_cache_b"
        ):
            raise ValueError("paired deployment requires two correction caches")
        if mode != PAIRED_DEPLOYMENT_TARGET_MODE and source.get(
            "correction_cache_b"
        ):
            raise ValueError("second correction cache is only valid for paired mode")
    return resolved


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _history_state(
    q_nominal: torch.Tensor,
    q_corrected: torch.Tensor,
    measurement_valid: torch.Tensor,
    history_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    residual = q_corrected[:, :history_size].float() - q_nominal[:, :history_size].float()
    valid = measurement_valid[:, :history_size].bool()
    posterior = torch.zeros_like(residual[:, 0])
    velocity = torch.zeros_like(posterior)
    has_previous = torch.zeros_like(valid[:, 0])
    steps_since_valid = torch.zeros_like(valid[:, 0], dtype=torch.long)
    last_valid_residual = torch.zeros_like(posterior)
    posterior_sequence: list[torch.Tensor] = []
    valid_sequence: list[torch.Tensor] = []
    for index in range(history_size):
        current_valid = valid[:, index]
        propagated = torch.where(
            has_previous.unsqueeze(-1), posterior + velocity, posterior
        )
        elapsed = (steps_since_valid + 1).clamp_min(1)
        observed_velocity = (
            residual[:, index] - last_valid_residual
        ) / elapsed.unsqueeze(-1)
        can_update_velocity = current_valid & has_previous
        velocity = torch.where(
            can_update_velocity.unsqueeze(-1), observed_velocity, velocity
        )
        posterior = torch.where(
            current_valid.unsqueeze(-1), residual[:, index], propagated
        )
        last_valid_residual = torch.where(
            current_valid.unsqueeze(-1), residual[:, index], last_valid_residual
        )
        steps_since_valid = torch.where(
            current_valid,
            torch.zeros_like(steps_since_valid),
            torch.where(has_previous, elapsed, steps_since_valid),
        )
        has_previous = has_previous | current_valid
        posterior_sequence.append(posterior)
        valid_sequence.append(has_previous)
    return (
        posterior,
        velocity,
        torch.stack(posterior_sequence, dim=1),
        torch.stack(valid_sequence, dim=1),
    )


def _measurement_observability(
    measurement_valid: torch.Tensor, history_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = measurement_valid[:, :history_size].bool()
    age = torch.zeros_like(valid[:, 0], dtype=torch.long)
    has = torch.zeros_like(valid[:, 0])
    for index in range(history_size):
        current = valid[:, index]
        age = torch.where(current, torch.zeros_like(age), age + 1)
        has = has | current
    age = torch.where(has, age, torch.full_like(age, history_size))
    normalized = age.clamp(max=history_size).to(dtype=torch.float32) / float(history_size)
    return valid[:, history_size - 1], normalized


def _causal_prior(
    runtime: CausalLeWMPixelObserverV2Runtime,
    batch: Mapping[str, Any],
    *,
    measurement_name: str,
    measurement_valid_name: str,
) -> torch.Tensor:
    history = runtime.history_size
    q_nominal = batch["q_nominal"].float()
    q_corrected = batch[measurement_name].float()
    posterior, velocity, posterior_sequence, posterior_valid = _history_state(
        q_nominal,
        q_corrected,
        batch[measurement_valid_name],
        history,
    )
    previous_measurement_valid, measurement_age_steps = _measurement_observability(
        batch[measurement_valid_name], history
    )
    measurement_age_steps = measurement_age_steps.to(dtype=q_nominal.dtype)
    q_grounded_history = q_nominal[:, :history] + posterior_sequence
    history_batch = {
        "pixels": batch["pixels"][:, :history].float(),
        "proprio": batch["proprio"][:, :history].float(),
        "q_nominal": q_nominal[:, :history],
        "q_corrected": q_grounded_history,
        "keypoints_xyz": batch["keypoints_xyz"][:, :history].float(),
        "valid": posterior_valid & batch["valid_nominal"][:, :history].bool(),
    }
    result = runtime.predict_prior(
        history_batch=history_batch,
        transition_actions=batch["action"][:, :history].float(),
        posterior_residual_px=posterior,
        residual_velocity_px=velocity,
        previous_q_nominal_px=q_nominal[:, history - 1],
        current_q_nominal_px=q_nominal[:, history],
        previous_keypoints_xyz=batch["keypoints_xyz"][:, history - 1].float(),
        current_keypoints_xyz=batch["keypoints_xyz"][:, history].float(),
        previous_nominal_valid=batch["valid_nominal"][:, history - 1].bool(),
        current_nominal_valid=batch["valid_nominal"][:, history].bool(),
        previous_measurement_valid=previous_measurement_valid,
        measurement_age_steps=measurement_age_steps,
    )
    return result.residual_px


def pixel_observer_v2_loss(
    runtime: CausalLeWMPixelObserverV2Runtime,
    batch: Mapping[str, Any],
    *,
    target_mode: str,
    huber_delta_px: float,
) -> V2LossOutput:
    """Compute a target-isolated prior loss.

    Only history measurements are passed to :func:`_causal_prior`; target
    measurements and q_true are admitted after the prior has been formed.
    """

    history = runtime.history_size
    valid_a = batch.get("measurement_valid_a")
    if target_mode == ORACLE_CEILING_TARGET_MODE:
        prediction_a = _causal_prior(
            runtime,
            batch,
            measurement_name="q_true_history",
            measurement_valid_name="valid_true_history",
        )
    else:
        if not torch.is_tensor(valid_a) or valid_a.dtype != torch.bool:
            raise TypeError("measurement_valid_a must be a boolean tensor")
        prediction_a = _causal_prior(
            runtime,
            batch,
            measurement_name="q_corrected_a",
            measurement_valid_name="measurement_valid_a",
        )
    nominal = batch["q_nominal"][:, history].float()
    nominal_valid = (
        batch["valid_nominal"][:, history].bool()
        & batch["prior_supervision_valid_target"].bool()
    )
    if target_mode != ORACLE_CEILING_TARGET_MODE:
        nominal_valid &= valid_a[:, history]
    if target_mode == PAIRED_DEPLOYMENT_TARGET_MODE:
        valid_b = batch.get("measurement_valid_b")
        if not torch.is_tensor(valid_b) or valid_b.dtype != torch.bool:
            raise TypeError("paired deployment requires boolean measurement_valid_b")
        if valid_a.shape != valid_b.shape or not torch.equal(valid_a, valid_b):
            raise ValueError(
                "paired deployment requires one identical source-derived "
                "measurement-validity mask for A and B"
            )
        prediction_b = _causal_prior(
            runtime,
            batch,
            measurement_name="q_corrected_b",
            measurement_valid_name="measurement_valid_b",
        )
        supervision = build_pixel_observer_v2_paired_supervision(
            q_nominal_px=nominal,
            nominal_valid=nominal_valid,
            q_corrected_a_px=batch["q_corrected_a"][:, history].float(),
            q_corrected_b_px=batch["q_corrected_b"][:, history].float(),
        )
        loss = paired_mutual_huber_pixel_loss(
            prediction_a,
            prediction_b,
            supervision,
            delta_px=huber_delta_px,
        )
        return V2LossOutput(
            loss=loss,
            predictions=(prediction_a, prediction_b),
            target_residual_px=supervision.consensus_target_px,
            valid=supervision.valid,
        )
    if target_mode == ORACLE_CEILING_TARGET_MODE:
        target_valid = nominal_valid & batch["valid_true_target"].bool()
        supervision = build_pixel_observer_v2_supervision(
            q_nominal_px=nominal,
            nominal_valid=target_valid,
            target_mode=target_mode,
            q_true_px=batch["q_true_target"].float(),
        )
    elif target_mode == SINGLE_MEASUREMENT_BASELINE_TARGET_MODE:
        supervision = build_pixel_observer_v2_supervision(
            q_nominal_px=nominal,
            nominal_valid=nominal_valid,
            target_mode=target_mode,
            q_corrected_px=batch["q_corrected_a"][:, history].float(),
        )
    else:
        raise ValueError("unknown target_mode")
    loss = masked_huber_pixel_loss(
        prediction_a, supervision, delta_px=huber_delta_px
    )
    return V2LossOutput(
        loss=loss,
        predictions=(prediction_a,),
        target_residual_px=supervision.target_residual_px,
        valid=supervision.valid,
    )


def _metrics(output: V2LossOutput) -> tuple[float, int, float, int]:
    """Return pooled EPE and robust-loss sufficient statistics."""

    prediction = torch.stack(output.predictions, dim=0).mean(dim=0)
    valid = output.valid
    error = torch.linalg.vector_norm(prediction - output.target_residual_px, dim=-1)
    anchor_count = int(valid.sum().item())
    coordinate_count = anchor_count * int(prediction.shape[-1])
    if anchor_count == 0:
        return 0.0, 0, 0.0, 0
    return (
        float(error[valid].sum().detach()),
        anchor_count,
        float(output.loss.detach()) * coordinate_count,
        coordinate_count,
    )


@torch.no_grad()
def evaluate_v2_loader(
    runtime: CausalLeWMPixelObserverV2Runtime,
    loader: Any,
    *,
    device: torch.device,
    target_mode: str,
    huber_delta_px: float,
    max_batches: int = 0,
) -> dict[str, float]:
    runtime.eval()
    epe_sum = 0.0
    count = 0
    loss_sum = 0.0
    loss_coordinates = 0
    batches = 0
    for batch_index, host_batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        batch = move_batch_to_device(host_batch, device)
        output = pixel_observer_v2_loss(
            runtime,
            batch,
            target_mode=target_mode,
            huber_delta_px=huber_delta_px,
        )
        epe, anchors, loss_numerator, coordinates = _metrics(output)
        if anchors == 0:
            continue
        epe_sum += epe
        count += anchors
        loss_sum += loss_numerator
        loss_coordinates += coordinates
        batches += 1
    if batches == 0:
        raise ValueError("validation has no unmasked target anchors")
    return {
        "loss": loss_sum / loss_coordinates,
        "epe_px": epe_sum / count,
        "valid_anchors": float(count),
        "valid_coordinates": float(loss_coordinates),
        "batches": float(batches),
    }


def _training_binding(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "resolved_config_sha256": payload["resolved_config_sha256"],
        "base_checkpoint_sha256": payload["base_checkpoint_sha256"],
        "adapter_state_sha256": payload["adapter_state_sha256"],
        "data_provenance_sha256": payload["data_provenance_sha256"],
        "target_mode": payload["target_mode"],
        "step": payload["step"],
        "epoch": payload["epoch"],
        "training_budget": payload["training_budget"],
    }


def validate_v2_completed_training_budget(
    payload: Mapping[str, Any], *, require_endpoint: bool
) -> dict[str, Any]:
    budget_value = payload.get("training_budget")
    if not isinstance(budget_value, Mapping) or budget_value.get("run_completed") is not True:
        raise ValueError("checkpoint training budget is incomplete")
    budget = dict(budget_value)
    try:
        target_steps = int(budget.get("target_optimizer_steps", -1))
        checkpoint_steps = int(budget.get("checkpoint_optimizer_step", -1))
        final_steps = int(budget.get("final_optimizer_steps", -1))
        top_level_steps = int(payload.get("step", -1))
    except (TypeError, ValueError):
        raise ValueError("checkpoint training budget is malformed") from None
    valid = (
        target_steps > 0
        and 0 <= checkpoint_steps <= final_steps
        and final_steps == target_steps
        and checkpoint_steps == top_level_steps
    )
    if require_endpoint:
        valid = valid and checkpoint_steps == final_steps
    if not valid:
        raise ValueError("checkpoint training budget cross-fields are inconsistent")
    return budget


def validate_v2_base_history_binding(
    resolved_config: Mapping[str, Any], base_model: torch.nn.Module
) -> int:
    """Fail closed when observer windows disagree with the frozen LeWM history."""

    try:
        configured = int(resolved_config["data"]["history_size"])
        frozen = int(base_model.history_size)
    except (AttributeError, KeyError, TypeError, ValueError):
        raise ValueError("cannot validate observer/base history_size binding") from None
    if configured != frozen:
        raise ValueError(
            "data.history_size must match frozen GroundedLeWM history_size "
            f"({configured} != {frozen})"
        )
    return frozen


def _augment_checkpoint(
    path: Path,
    *,
    resolved: Mapping[str, Any],
    base_path: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    target_steps: int,
    epoch: int,
    metrics: Mapping[str, Any],
    completed: bool,
) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload.update(
        {
            "resolved_config": copy.deepcopy(dict(resolved)),
            "resolved_config_sha256": canonical_sha256(resolved),
            "base_checkpoint_path": str(base_path),
            "base_checkpoint_sha256": sha256_file(base_path),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "step": int(step),
            "epoch": int(epoch),
            "training_budget": {
                "target_optimizer_steps": int(target_steps),
                "checkpoint_optimizer_step": int(step),
                "run_completed": bool(completed),
                "final_optimizer_steps": int(step) if completed else None,
            },
            "metrics": copy.deepcopy(dict(metrics)),
        }
    )
    payload["training_binding_sha256"] = canonical_sha256(
        _training_binding(payload)
    )
    _atomic_torch_save(payload, path)


def _certify_finished_run(path: Path, final_steps: int) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    budget = dict(payload["training_budget"])
    if int(budget["target_optimizer_steps"]) != int(final_steps):
        raise ValueError("checkpoint target budget differs from finished run")
    checkpoint_steps = int(budget["checkpoint_optimizer_step"])
    if checkpoint_steps != int(payload["step"]):
        raise ValueError("checkpoint step differs from its optimizer-step record")
    if not 0 <= checkpoint_steps <= int(final_steps):
        raise ValueError("checkpoint step lies outside the finished run")
    # A best checkpoint can predate the endpoint. Its own step remains intact;
    # final_optimizer_steps certifies that the complete matched run finished.
    budget["run_completed"] = True
    budget["final_optimizer_steps"] = int(final_steps)
    payload["training_budget"] = budget
    payload["training_binding_sha256"] = canonical_sha256(
        _training_binding(payload)
    )
    _atomic_torch_save(payload, path)
    validate_v2_completed_training_budget(
        payload, require_endpoint=checkpoint_steps == int(final_steps)
    )


def _new_runtime(
    base_model: torch.nn.Module,
    bundle: PixelObserverV2DataBundle,
    config: Mapping[str, Any],
    device: torch.device,
) -> CausalLeWMPixelObserverV2Runtime:
    adapter_config = dict(config["adapter"])
    adapter = PixelResidualVelocityInnovationAdapter(
        latent_dim=int(base_model.latent_dim),
        num_keypoints=int(base_model.frame_encoder.num_keypoints),
        image_size=tuple(bundle.reference_dataset.image_size),
        **adapter_config,
    ).to(device)
    adapter.set_geometry_normalization(
        xyz_mean=bundle.geometry_mean, xyz_std=bundle.geometry_std
    )
    return CausalLeWMPixelObserverV2Runtime(base_model, adapter).to(device)


def train_pixel_observer_v2(
    config: Mapping[str, Any],
    *,
    device: str | torch.device = "auto",
    smoke_steps: int = 0,
) -> V2TrainingResult:
    resolved = resolve_v2_training_config(config)
    target_steps = (
        int(smoke_steps)
        if smoke_steps > 0
        else int(resolved["trainer"]["max_steps"])
    )
    resolved["run_contract"] = {
        "kind": "smoke" if smoke_steps > 0 else "formal",
        "effective_target_optimizer_steps": target_steps,
        "configured_target_optimizer_steps": int(resolved["trainer"]["max_steps"]),
    }
    _seed_everything(int(resolved["seed"]))
    target_device = choose_device(device)
    output_dir = Path(resolved["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.iterdir())
    if existing:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    bundle = build_pixel_observer_v2_data(resolved)
    maximum_batches = len(bundle.train_loader) * int(resolved["trainer"]["epochs"])
    if maximum_batches < target_steps:
        raise ValueError(
            "training capacity cannot reach target_optimizer_steps: "
            f"{maximum_batches} < {target_steps}"
        )
    base_path = Path(resolved["base_checkpoint"])
    base_model, _, _ = load_lewm_grounded_checkpoint(
        base_path, device=target_device, dataset=bundle.reference_dataset
    )
    validate_v2_base_history_binding(resolved, base_model)
    runtime = _new_runtime(base_model, bundle, resolved, target_device)
    trainable = [parameter for parameter in runtime.adapter.parameters() if parameter.requires_grad]
    if any(parameter.requires_grad for parameter in runtime.base_model.parameters()):
        raise RuntimeError("GroundedLeWM must remain frozen")
    optimizer_config = resolved["optimizer"]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(optimizer_config["lr"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["eps"]),
    )
    warmup = min(int(optimizer_config["warmup_steps"]), target_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_warmup_lambda(step, warmup, target_steps)
    )
    mode = str(resolved["target_mode"])
    delta = float(resolved["loss"]["huber_delta_px"])
    latest = output_dir / "pixel_observer_v2_latest.pt"
    best = output_dir / "pixel_observer_v2_best.pt"
    metrics_path = output_dir / "metrics.jsonl"
    snapshot = output_dir / "resolved_config.json"
    _atomic_text(snapshot, json.dumps(resolved, indent=2, sort_keys=True) + "\n")
    records: list[dict[str, Any]] = []
    global_step = 0
    best_val = math.inf
    final_epoch = 0
    for epoch in range(1, int(resolved["trainer"]["epochs"]) + 1):
        runtime.train()
        train_loss_numerator = 0.0
        train_loss_coordinates = 0
        train_batches = 0
        for host_batch in bundle.train_loader:
            batch = move_batch_to_device(host_batch, target_device)
            optimizer.zero_grad(set_to_none=True)
            output = pixel_observer_v2_loss(
                runtime, batch, target_mode=mode, huber_delta_px=delta
            )
            if not bool(output.valid.any()):
                continue
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable, float(resolved["trainer"]["grad_clip"])
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            train_batches += 1
            _, _, loss_numerator, coordinates = _metrics(output)
            train_loss_numerator += loss_numerator
            train_loss_coordinates += coordinates
            if global_step >= target_steps:
                break
        val = evaluate_v2_loader(
            runtime,
            bundle.val_loader,
            device=target_device,
            target_mode=mode,
            huber_delta_px=delta,
            max_batches=1 if smoke_steps > 0 else 0,
        )
        record = {
            "epoch": epoch,
            "step": global_step,
            "train_loss": train_loss_numerator / max(train_loss_coordinates, 1),
            "train_valid_coordinates": train_loss_coordinates,
            "train_batches": train_batches,
            "val": val,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        records.append(record)
        improved = float(val["loss"]) < best_val
        if improved:
            best_val = float(val["loss"])
        save_pixel_observer_v2_checkpoint(
            latest,
            runtime,
            target_mode=mode,
            data_provenance=bundle.provenance,
        )
        _augment_checkpoint(
            latest,
            resolved=resolved,
            base_path=base_path,
            optimizer=optimizer,
            scheduler=scheduler,
            step=global_step,
            target_steps=target_steps,
            epoch=epoch,
            metrics=record,
            completed=global_step == target_steps,
        )
        if improved:
            _atomic_torch_save(
                torch.load(latest, map_location="cpu", weights_only=False), best
            )
        _atomic_text(
            metrics_path,
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
        )
        final_epoch = epoch
        if global_step >= target_steps:
            break
    if global_step != target_steps:
        raise RuntimeError(f"training exhausted epochs at {global_step}/{target_steps}")
    _certify_finished_run(latest, global_step)
    _certify_finished_run(best, global_step)
    return V2TrainingResult(
        output_dir=output_dir,
        latest_checkpoint=latest,
        best_checkpoint=best,
        metrics_path=metrics_path,
        global_step=global_step,
        best_val_loss=best_val,
    )


def load_trained_pixel_observer_v2(
    checkpoint_path: str | Path,
    *,
    base_model: torch.nn.Module,
    expected_data_provenance: Mapping[str, Any],
    for_deployment: bool,
    allow_smoke: bool = False,
) -> tuple[CausalLeWMPixelObserverV2Runtime, dict[str, Any]]:
    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    resolved = payload.get("resolved_config")
    if not isinstance(resolved, Mapping):
        raise ValueError("trained checkpoint lacks resolved_config")
    if payload.get("resolved_config_sha256") != canonical_sha256(resolved):
        raise ValueError("resolved config binding is invalid")
    validate_v2_base_history_binding(resolved, base_model)
    base_path = Path(str(payload.get("base_checkpoint_path", "")))
    if not base_path.is_file() or payload.get("base_checkpoint_sha256") != sha256_file(base_path):
        raise ValueError("base checkpoint file binding is invalid")
    if payload.get("training_binding_sha256") != canonical_sha256(
        _training_binding(payload)
    ):
        raise ValueError("training completion binding is invalid")
    budget = validate_v2_completed_training_budget(
        payload, require_endpoint=path.name.endswith("_latest.pt")
    )
    run_contract = resolved.get("run_contract")
    if not isinstance(run_contract, Mapping):
        raise ValueError("trained checkpoint lacks run_contract")
    kind = str(run_contract.get("kind"))
    effective = int(run_contract.get("effective_target_optimizer_steps", -1))
    configured = int(run_contract.get("configured_target_optimizer_steps", -1))
    if effective != int(budget["target_optimizer_steps"]):
        raise ValueError("run contract and training budget disagree")
    if configured != int(resolved["trainer"]["max_steps"]):
        raise ValueError("run contract and resolved trainer budget disagree")
    if kind == "formal" and effective != configured:
        raise ValueError("formal run did not complete its configured budget")
    if kind == "smoke" and not allow_smoke:
        raise ValueError("smoke checkpoint is diagnostic-only")
    if kind not in {"formal", "smoke"}:
        raise ValueError("unknown run contract kind")
    return load_pixel_observer_v2_checkpoint(
        path,
        base_model=base_model,
        expected_data_provenance=expected_data_provenance,
        for_deployment=for_deployment,
    )


__all__ = [
    "DEFAULT_CONFIG",
    "V2LossOutput",
    "V2TrainingResult",
    "evaluate_v2_loader",
    "load_trained_pixel_observer_v2",
    "pixel_observer_v2_loss",
    "resolve_v2_training_config",
    "train_pixel_observer_v2",
    "validate_v2_base_history_binding",
    "validate_v2_completed_training_budget",
]
