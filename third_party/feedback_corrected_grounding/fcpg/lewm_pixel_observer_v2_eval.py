"""Sequential causal evaluation for PixelObserver v2 feedback modes."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping

import numpy as np
import torch

from fcpg.lewm_pixel_observer_baselines import (
    ConstantVelocityResidualBaseline,
    HoldLastResidualBaseline,
    LinearConstantVelocityKalmanBaseline,
    fcpg_measurement,
)
from fcpg.lewm_pixel_observer_v2_data import PixelObserverV2WindowDataset
from fcpg.lewm_pixel_observer_v2_runtime import CausalLeWMPixelObserverV2Runtime


@dataclass(frozen=True)
class V2Sequence:
    pixels: torch.Tensor
    proprio: torch.Tensor
    actions: torch.Tensor
    q_nominal_px: torch.Tensor
    q_corrected_px: torch.Tensor
    q_true_px: torch.Tensor
    keypoints_xyz: torch.Tensor
    nominal_valid: torch.Tensor
    measurement_valid: torch.Tensor
    true_valid: torch.Tensor
    step_index: int | None = None
    active_drift: torch.Tensor | None = None
    drift_phase: torch.Tensor | None = None


def oracle_diagnostic_sequence(sequence: V2Sequence) -> V2Sequence:
    """Substitute true measurements without reviving evaluator-censored frames."""

    return replace(
        sequence,
        q_corrected_px=sequence.q_true_px,
        measurement_valid=(
            sequence.measurement_valid.bool()
            & sequence.true_valid.bool()
            & sequence.nominal_valid.bool()
        ),
    )


@dataclass(frozen=True)
class SequentialModeResult:
    name: str
    residual_px: torch.Tensor
    prior_residual_px: torch.Tensor
    measurement_valid: torch.Tensor
    warmup_frames: int
    prior_available: bool = True


def _validate_sequence(sequence: V2Sequence, history_size: int) -> tuple[int, int]:
    q = sequence.q_nominal_px
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != 2:
        raise ValueError("sequence q arrays must have shape [1,T,K,2]")
    batch, frames, anchors, _ = q.shape
    if frames <= history_size:
        raise ValueError("sequence is too short for observer history")
    for name in ("q_corrected_px", "q_true_px"):
        if getattr(sequence, name).shape != q.shape:
            raise ValueError(f"{name} must match q_nominal_px")
    if sequence.keypoints_xyz.shape != (batch, frames, anchors, 3):
        raise ValueError("keypoints_xyz has an invalid shape")
    for name in ("nominal_valid", "measurement_valid", "true_valid"):
        if getattr(sequence, name).shape != (batch, frames, anchors):
            raise ValueError(f"{name} has an invalid shape")
    if sequence.actions.shape[1:] != (frames - 1, sequence.actions.shape[-1]):
        raise ValueError("actions must have shape [1,T-1,A]")
    if sequence.pixels.shape[:2] != (batch, frames):
        raise ValueError("pixels must begin with [1,T]")
    if sequence.proprio.shape[:2] != (batch, frames):
        raise ValueError("proprio must begin with [1,T]")
    if sequence.active_drift is not None and sequence.active_drift.shape != (
        batch, frames
    ):
        raise ValueError("active_drift must have shape [1,T]")
    if sequence.drift_phase is not None:
        if sequence.drift_phase.shape != (batch, frames):
            raise ValueError("drift_phase must have shape [1,T]")
        if not sequence.drift_phase.is_floating_point():
            raise TypeError("drift_phase must be floating point")
        if not bool(torch.isfinite(sequence.drift_phase).all()):
            raise ValueError("drift_phase must be finite")
    return frames, anchors


def _measurement_history_state(
    residual: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    posterior = torch.zeros_like(residual[:, 0])
    velocity = torch.zeros_like(posterior)
    has = torch.zeros_like(valid[:, 0])
    steps_since_valid = torch.zeros_like(valid[:, 0], dtype=torch.long)
    last_valid_residual = torch.zeros_like(posterior)
    states: list[torch.Tensor] = []
    valid_states: list[torch.Tensor] = []
    for index in range(residual.shape[1]):
        propagated = torch.where(has.unsqueeze(-1), posterior + velocity, posterior)
        elapsed = (steps_since_valid + 1).clamp_min(1)
        observed = (residual[:, index] - last_valid_residual) / elapsed.unsqueeze(-1)
        velocity = torch.where((valid[:, index] & has).unsqueeze(-1), observed, velocity)
        posterior = torch.where(
            valid[:, index].unsqueeze(-1), residual[:, index], propagated
        )
        last_valid_residual = torch.where(
            valid[:, index].unsqueeze(-1), residual[:, index], last_valid_residual
        )
        steps_since_valid = torch.where(
            valid[:, index],
            torch.zeros_like(steps_since_valid),
            torch.where(has, elapsed, steps_since_valid),
        )
        has = has | valid[:, index]
        states.append(posterior)
        valid_states.append(has)
    return posterior, velocity, torch.stack(states, 1), torch.stack(valid_states, 1)


def _measurement_observability_before(
    measurement_valid: torch.Tensor, index: int, history_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    past = measurement_valid[:, :index].bool()
    if past.shape[1] == 0:
        raise ValueError("measurement observability requires non-empty history")
    age = torch.zeros_like(past[:, 0], dtype=torch.long)
    has = torch.zeros_like(past[:, 0])
    for position in range(past.shape[1]):
        current = past[:, position]
        age = torch.where(current, torch.zeros_like(age), age + 1)
        has = has | current
    age = torch.where(has, age, torch.full_like(age, history_size))
    normalized = age.clamp(max=history_size).to(dtype=torch.float32) / float(history_size)
    return past[:, -1], normalized


@torch.no_grad()
def run_v2_feedback_mode(
    runtime: CausalLeWMPixelObserverV2Runtime,
    sequence: V2Sequence,
    *,
    mode: str,
    gain: float = 1.0,
    velocity_gain: float = 1.0,
) -> SequentialModeResult:
    """Run one-shot, no-writeback, or full posterior writeback causally."""

    allowed = {
        "one_shot_fcpg",
        "one_shot_fcpg_nominal_fallback",
        "one_shot_fcpg_hold_fallback",
        "posterior_reindex_no_writeback",
        "full_writeback",
    }
    if mode not in allowed:
        raise ValueError(f"unsupported feedback mode: {mode!r}")
    if not math.isfinite(float(gain)) or not 0.0 <= float(gain) <= 1.0:
        raise ValueError("gain must lie in [0,1]")
    if (
        not math.isfinite(float(velocity_gain))
        or not 0.0 <= float(velocity_gain) <= 1.0
    ):
        raise ValueError("velocity_gain must lie in [0,1]")
    history = runtime.history_size
    frames, _ = _validate_sequence(sequence, history)
    nominal = sequence.q_nominal_px.float()
    measurement = sequence.q_corrected_px.float() - nominal
    measurement_valid = sequence.measurement_valid.bool() & sequence.nominal_valid.bool()
    if mode in {
        "one_shot_fcpg",
        "one_shot_fcpg_nominal_fallback",
        "one_shot_fcpg_hold_fallback",
    }:
        if mode == "one_shot_fcpg_hold_fallback":
            posterior = torch.zeros_like(measurement[:, 0])
            steps = []
            for index in range(frames):
                posterior = torch.where(
                    measurement_valid[:, index].unsqueeze(-1),
                    measurement[:, index],
                    posterior,
                )
                if index >= history:
                    steps.append(posterior)
            output = torch.stack(steps, dim=1)
        else:
            output = torch.where(
                measurement_valid[:, history:].unsqueeze(-1),
                measurement[:, history:],
                torch.zeros_like(measurement[:, history:]),
            )
        return SequentialModeResult(
            name=mode,
            residual_px=output,
            prior_residual_px=torch.zeros_like(output),
            measurement_valid=measurement_valid[:, history:],
            warmup_frames=history,
            prior_available=False,
        )

    _, initial_velocity, initial_states, initial_valid_states = _measurement_history_state(
        measurement[:, :history], measurement_valid[:, :history]
    )
    full_states = [initial_states[:, index] for index in range(history)]
    full_valid_states = [
        initial_valid_states[:, index] for index in range(history)
    ]
    full_velocity = initial_velocity
    output_steps: list[torch.Tensor] = []
    prior_steps: list[torch.Tensor] = []
    for index in range(history, frames):
        if mode == "posterior_reindex_no_writeback":
            posterior, velocity, history_states, history_valid = _measurement_history_state(
                measurement[:, index - history : index],
                measurement_valid[:, index - history : index],
            )
        else:
            history_tensor = torch.stack(full_states[-history:], dim=1)
            posterior = history_tensor[:, -1]
            velocity = full_velocity
            history_states = history_tensor
            history_valid = torch.stack(full_valid_states[-history:], dim=1)
        history_batch = {
            "pixels": sequence.pixels[:, index - history : index].float(),
            "proprio": sequence.proprio[:, index - history : index].float(),
            "q_nominal": nominal[:, index - history : index],
            "q_corrected": nominal[:, index - history : index] + history_states,
            "keypoints_xyz": sequence.keypoints_xyz[:, index - history : index].float(),
            "valid": history_valid
            & sequence.nominal_valid[:, index - history : index].bool(),
        }
        previous_measurement_valid, measurement_age_steps = (
            _measurement_observability_before(measurement_valid, index, history)
        )
        prior = runtime.predict_prior(
            history_batch=history_batch,
            transition_actions=sequence.actions[:, index - history : index].float(),
            posterior_residual_px=posterior,
            residual_velocity_px=velocity,
            previous_q_nominal_px=nominal[:, index - 1],
            current_q_nominal_px=nominal[:, index],
            previous_keypoints_xyz=sequence.keypoints_xyz[:, index - 1].float(),
            current_keypoints_xyz=sequence.keypoints_xyz[:, index].float(),
            previous_nominal_valid=sequence.nominal_valid[:, index - 1].bool(),
            current_nominal_valid=sequence.nominal_valid[:, index].bool(),
            previous_measurement_valid=previous_measurement_valid,
            measurement_age_steps=measurement_age_steps.to(dtype=nominal.dtype),
        )
        current_valid = measurement_valid[:, index]
        # Explicit alpha-beta measurement update: alpha corrects position,
        # while beta updates the velocity carried into the next prior.
        posterior_current = torch.where(
            current_valid.unsqueeze(-1),
            prior.residual_px
            + float(gain) * (measurement[:, index] - prior.residual_px),
            prior.residual_px,
        )
        posterior_velocity = torch.where(
            current_valid.unsqueeze(-1),
            prior.velocity_px
            + float(velocity_gain) * (measurement[:, index] - prior.residual_px),
            prior.velocity_px,
        )
        output_steps.append(posterior_current)
        prior_steps.append(prior.residual_px)
        if mode == "full_writeback":
            full_states.append(posterior_current)
            full_velocity = posterior_velocity
            full_valid_states.append(full_valid_states[-1] | current_valid)
    return SequentialModeResult(
        name=mode,
        residual_px=torch.stack(output_steps, dim=1),
        prior_residual_px=torch.stack(prior_steps, dim=1),
        measurement_valid=measurement_valid[:, history:],
        warmup_frames=history,
    )


def run_classic_baselines(
    sequence: V2Sequence,
    *,
    warmup_frames: int,
) -> dict[str, SequentialModeResult]:
    residual = sequence.q_corrected_px.float() - sequence.q_nominal_px.float()
    measurement = fcpg_measurement(residual, sequence.measurement_valid.bool())
    baselines = {
        "hold_last": HoldLastResidualBaseline(gain=1.0),
        "constant_velocity": ConstantVelocityResidualBaseline(gain=1.0),
        "linear_kalman": LinearConstantVelocityKalmanBaseline(),
    }
    results: dict[str, SequentialModeResult] = {
        "nominal_open_loop": SequentialModeResult(
            name="nominal_open_loop",
            residual_px=torch.zeros_like(residual[:, warmup_frames:]),
            prior_residual_px=torch.zeros_like(residual[:, warmup_frames:]),
            measurement_valid=sequence.measurement_valid[:, warmup_frames:].bool(),
            warmup_frames=warmup_frames,
        )
    }
    for name, baseline in baselines.items():
        state = baseline.initialize(residual[:, 0])
        priors: list[torch.Tensor] = []
        posteriors: list[torch.Tensor] = []
        for index in range(residual.shape[1]):
            prior = baseline.prior(state)
            current = fcpg_measurement(
                residual[:, index], sequence.measurement_valid[:, index].bool()
            )
            update = baseline.update(prior, current)
            state = update.state
            if index >= warmup_frames:
                priors.append(update.prior_residual_px)
                posteriors.append(update.posterior_residual_px)
        results[name] = SequentialModeResult(
            name=name,
            residual_px=torch.stack(posteriors, dim=1),
            prior_residual_px=torch.stack(priors, dim=1),
            measurement_valid=sequence.measurement_valid[:, warmup_frames:].bool(),
            warmup_frames=warmup_frames,
        )
    return results


def _record_error_family(
    metrics: dict[str, Any],
    *,
    prefix: str,
    error: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    selected = error[mask]
    metrics[f"_{prefix}_errors_px"] = selected.detach().cpu().tolist()
    metrics[f"{prefix}_anchor_frames"] = int(selected.numel())
    metrics[f"{prefix}_sum_px"] = float(selected.sum()) if selected.numel() else 0.0
    metrics[f"{prefix}_epe_px"] = (
        float(selected.mean()) if selected.numel() else None
    )
    metrics[f"{prefix}_p95_px"] = (
        float(torch.quantile(selected, 0.95)) if selected.numel() else None
    )


def _record_step_settling(
    metrics: dict[str, Any],
    *,
    prefix: str,
    error: torch.Tensor,
    valid: torch.Tensor,
    relative_step: int,
    threshold_px: float,
    consecutive: int,
) -> None:
    valid_count = valid.sum(dim=-1)
    per_frame = (
        torch.where(valid, error, torch.zeros_like(error)).sum(dim=-1)
        / valid_count.clamp_min(1)
    )
    per_frame = torch.where(
        valid_count > 0, per_frame, torch.full_like(per_frame, torch.inf)
    )
    settling = None
    for index in range(relative_step, per_frame.shape[1] - consecutive + 1):
        window = per_frame[0, index : index + consecutive]
        if bool((window <= float(threshold_px)).all()):
            settling = index - relative_step
            break
    horizon = max(0, int(per_frame.shape[1]) - relative_step)
    metrics[f"{prefix}_step_settled"] = int(settling is not None)
    metrics[f"{prefix}_step_time_sampled_steps"] = (
        int(settling) if settling is not None else int(horizon)
    )
    metrics[f"{prefix}_step_followup_sampled_steps"] = int(horizon)
    metrics[f"{prefix}_step_settling_sampled_steps"] = (
        int(settling) if settling is not None else None
    )
    metrics[f"{prefix}_step_event_epe_px"] = float(per_frame[0, relative_step])


def sequential_metrics(
    result: SequentialModeResult,
    sequence: V2Sequence,
    *,
    settling_threshold_px: float = 1.0,
    settling_consecutive: int = 2,
) -> dict[str, Any]:
    """Score measurement-free priors separately from updated posteriors."""

    start = result.warmup_frames
    truth = sequence.q_true_px[:, start:].float() - sequence.q_nominal_px[:, start:].float()
    valid = sequence.true_valid[:, start:].bool() & sequence.nominal_valid[:, start:].bool()
    posterior_error = torch.linalg.vector_norm(result.residual_px - truth, dim=-1)
    prior_error = torch.linalg.vector_norm(result.prior_residual_px - truth, dim=-1)
    if not bool(valid.any()):
        raise ValueError("sequence contains no valid q_true anchors")
    visible = result.measurement_valid & valid
    dropout = ~result.measurement_valid & valid
    metrics: dict[str, Any] = {}
    metrics["prior_available"] = bool(result.prior_available)
    error_families = [("post_update", posterior_error)]
    if result.prior_available:
        error_families.append(("prior", prior_error))
    for stem, error in error_families:
        _record_error_family(metrics, prefix=f"{stem}_qtrue", error=error, mask=valid)
        _record_error_family(metrics, prefix=f"{stem}_qtrue_visible", error=error, mask=visible)
        _record_error_family(metrics, prefix=f"{stem}_qtrue_dropout", error=error, mask=dropout)

    metrics["qtrue_epe_px"] = metrics["post_update_qtrue_epe_px"]
    metrics["qtrue_p95_px"] = metrics["post_update_qtrue_p95_px"]
    metrics["dropout_auc_sum_px"] = metrics["post_update_qtrue_dropout_sum_px"]
    metrics["dropout_auc_mean_px"] = metrics["post_update_qtrue_dropout_epe_px"]
    metrics["dropout_anchor_frames"] = metrics["post_update_qtrue_dropout_anchor_frames"]

    if sequence.drift_phase is not None:
        phase = sequence.drift_phase[:, start:].float()
        pre_frames = torch.isclose(phase, torch.zeros_like(phase), atol=1e-6, rtol=0.0)
        active_frames = (phase > 0.0) & (phase < 1.0)
        settled_frames = phase >= 1.0
        metrics["drift_phase_mask_source"] = "actual_camera_trajectory_phase"
        for stem, error in error_families:
            for label, frame_mask in (
                ("pre_drift_clean", pre_frames),
                ("active_drift", active_frames),
                ("settled_drift", settled_frames),
            ):
                _record_error_family(
                    metrics,
                    prefix=f"{stem}_{label}",
                    error=error,
                    mask=frame_mask.unsqueeze(-1) & valid,
                )
    elif sequence.step_index is None:
        active = (
            sequence.active_drift[:, start:].bool().unsqueeze(-1) & valid
            if sequence.active_drift is not None
            else valid
        )
        metrics["drift_phase_mask_source"] = (
            "legacy_active_drift" if sequence.active_drift is not None else "all_scored_frames_no_phase_metadata"
        )
        for stem, error in error_families:
            _record_error_family(
                metrics, prefix=f"{stem}_active_drift", error=error, mask=active
            )

    if sequence.step_index is not None and sequence.step_index >= start:
        relative = int(sequence.step_index - start)
        frame_index = torch.arange(valid.shape[1], device=valid.device).view(1, -1, 1)
        pre_step = (frame_index < relative) & valid
        post_excluding_event = (frame_index > relative) & valid
        for stem, error in error_families:
            _record_error_family(
                metrics, prefix=f"{stem}_pre_step_clean", error=error, mask=pre_step
            )
            _record_error_family(
                metrics,
                prefix=f"{stem}_post_step_excluding_event",
                error=error,
                mask=post_excluding_event,
            )
            _record_step_settling(
                metrics,
                prefix=stem,
                error=error,
                valid=valid,
                relative_step=relative,
                threshold_px=settling_threshold_px,
                consecutive=settling_consecutive,
            )

    visible_frames = result.measurement_valid.any(dim=-1)[0]
    invalid_frames = ~visible_frames
    reacquisition = torch.nonzero(visible_frames[1:] & invalid_frames[:-1]).flatten() + 1
    if reacquisition.numel():
        index = int(reacquisition[0])
        valid_frame = valid[0, index]
        metrics["reacquisition_sample_index"] = index + start
        metrics["post_update_reacquisition_epe_px"] = float(
            posterior_error[0, index][valid_frame].mean()
        )
        metrics["prior_reacquisition_epe_px"] = (
            float(prior_error[0, index][valid_frame].mean())
            if result.prior_available
            else None
        )
    else:
        metrics["reacquisition_sample_index"] = None
        metrics["post_update_reacquisition_epe_px"] = None
        metrics["prior_reacquisition_epe_px"] = None
    if result.name in {"one_shot_fcpg", "one_shot_fcpg_nominal_fallback"}:
        metrics["missing_measurement_fallback"] = "nominal_residual_zero"
    elif result.name == "one_shot_fcpg_hold_fallback":
        metrics["missing_measurement_fallback"] = "last_valid_fcpg"
    return metrics


def _right_censored_summary(
    times: list[int], events: list[bool]
) -> dict[str, Any]:
    if len(times) != len(events):
        raise ValueError("settling times/events length mismatch")
    if not times:
        return {}
    ordered = sorted(set(times))
    survival = 1.0
    previous = 0
    area = 0.0
    median = None
    for value in ordered:
        area += survival * max(0, value - previous)
        at_risk = sum(time >= value for time in times)
        observed = sum(time == value and event for time, event in zip(times, events))
        if at_risk > 0 and observed > 0:
            survival *= 1.0 - observed / at_risk
            if median is None and survival <= 0.5:
                median = int(value)
        previous = value
    return {
        "episodes": len(times),
        "settled": int(sum(events)),
        "right_censored": int(len(times) - sum(events)),
        "settled_fraction": float(np.mean(events)),
        "event_only_mean_sampled_steps": (
            float(np.mean([time for time, event in zip(times, events) if event]))
            if any(events)
            else None
        ),
        "kaplan_meier_median_sampled_steps": median,
        "restricted_mean_sampled_steps": float(area),
        "restriction_horizon_sampled_steps": int(max(times)),
    }


def aggregate_sequential_metrics(
    records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Pool anchor errors globally and preserve settling censoring."""

    result: dict[str, Any] = {"episodes": len(records)}
    private_error_keys = sorted(
        {key for record in records for key in record if key.startswith("_") and key.endswith("_errors_px")}
    )
    for private in private_error_keys:
        prefix = private[1:-10]
        values = np.asarray(
            [value for record in records for value in record.get(private, [])],
            dtype=np.float64,
        )
        result[f"{prefix}_anchor_frames"] = int(values.size)
        result[f"{prefix}_sum_px"] = float(values.sum())
        result[f"{prefix}_epe_px"] = float(values.mean()) if values.size else None
        result[f"{prefix}_p95_px"] = (
            float(np.quantile(values, 0.95)) if values.size else None
        )
    for stem in ("prior", "post_update"):
        rows = [
            record
            for record in records
            if f"{stem}_step_time_sampled_steps" in record
        ]
        if rows:
            summary = _right_censored_summary(
                [int(record[f"{stem}_step_time_sampled_steps"]) for record in rows],
                [bool(record[f"{stem}_step_settled"]) for record in rows],
            )
            for name, value in summary.items():
                result[f"{stem}_step_{name}"] = value
            event_values = [float(record[f"{stem}_step_event_epe_px"]) for record in rows]
            result[f"{stem}_step_event_epe_px"] = float(np.mean(event_values))
    for name in ("prior_reacquisition_epe_px", "post_update_reacquisition_epe_px"):
        values = [float(record[name]) for record in records if record.get(name) is not None]
        result[f"{name}_episode_macro_mean"] = float(np.mean(values)) if values else None
        result[f"{name}_episodes"] = len(values)
    for key in sorted({key for record in records for key in record}):
        if key.startswith("_") or key in result:
            continue
        labels = sorted({str(record[key]) for record in records if isinstance(record.get(key), str)})
        if labels:
            result[key] = labels[0] if len(labels) == 1 else labels
    return result


def evaluate_v2_sequence(
    runtime: CausalLeWMPixelObserverV2Runtime,
    sequence: V2Sequence,
    *,
    gain: float = 1.0,
    velocity_gain: float = 1.0,
) -> dict[str, dict[str, Any]]:
    outputs = run_classic_baselines(sequence, warmup_frames=runtime.history_size)
    for mode in (
        "one_shot_fcpg_nominal_fallback",
        "one_shot_fcpg_hold_fallback",
        "posterior_reindex_no_writeback",
        "full_writeback",
    ):
        outputs[mode] = run_v2_feedback_mode(
            runtime, sequence, mode=mode, gain=gain, velocity_gain=velocity_gain
        )
    return {
        name: sequential_metrics(output, sequence)
        for name, output in outputs.items()
    }


def sequence_from_dataset(
    dataset: PixelObserverV2WindowDataset,
    episode_index: int,
    *,
    device: torch.device,
    dropout_start_offset: int | None = None,
    dropout_length: int = 0,
    hidden_step_length: int = 0,
    require_step_event: bool = False,
) -> V2Sequence:
    """Materialize one frame-skip sequence and an evaluator-only dropout mask."""

    if not 0 <= int(episode_index) < len(dataset.episodes):
        raise IndexError("episode_index is out of range")
    raw_episode = dataset.episodes[int(episode_index)]
    positions = np.arange(0, len(raw_episode), dataset.frame_skip, dtype=np.int64)
    if len(positions) <= dataset.history_size:
        raise ValueError("episode is too short after frame_skip sampling")
    raw_indices = raw_episode[positions]
    base = dataset.base_a
    rgb = np.array(base.rgb[raw_indices], dtype=np.uint8, copy=True)
    pixels = torch.from_numpy(
        np.transpose(rgb.astype(np.float32) / 255.0, (0, 3, 1, 2))
    ).unsqueeze(0)
    proprio = torch.from_numpy(
        np.array(base.obs[raw_indices, : base.proprio_dim], dtype=np.float32, copy=True)
    ).unsqueeze(0)
    q_nominal = torch.from_numpy(
        np.array(base.q_nominal[raw_indices], dtype=np.float32, copy=True)
    ).unsqueeze(0)
    q_corrected = torch.from_numpy(
        np.array(base.q_corrected[raw_indices], dtype=np.float32, copy=True)
    ).unsqueeze(0)
    xyz = torch.from_numpy(
        np.array(base.keypoints_xyz[raw_indices], dtype=np.float32, copy=True)
    ).unsqueeze(0)
    nominal_valid = torch.from_numpy(
        np.array(base.valid_nominal[raw_indices], dtype=bool, copy=True)
    ).unsqueeze(0)
    if "q_true" not in dataset.extras or "valid_true" not in dataset.extras:
        raise ValueError("sequential evaluator requires q_true/valid_true for scoring")
    q_true = torch.from_numpy(
        np.array(dataset.extras["q_true"][raw_indices], dtype=np.float32, copy=True)
    ).unsqueeze(0)
    true_valid = torch.from_numpy(
        np.array(dataset.extras["valid_true"][raw_indices], dtype=bool, copy=True)
    ).unsqueeze(0)
    blocks = []
    for position in positions[:-1]:
        transition = raw_episode[position : position + dataset.frame_skip]
        blocks.append(base.action[transition].reshape(-1))
    actions = torch.from_numpy(
        np.stack(blocks).astype(np.float32, copy=False)
    ).unsqueeze(0)
    measurement_valid = torch.from_numpy(
        np.array(dataset.shared_measurement_valid[raw_indices], dtype=bool, copy=True)
    ).unsqueeze(0)
    active_drift = None
    drift_phase = None
    if "actual_camera_trajectory_phase" in dataset.extras:
        phase = np.asarray(dataset.extras["actual_camera_trajectory_phase"])[raw_indices]
        drift_phase = torch.from_numpy(
            np.array(phase, dtype=np.float32, copy=True)
        ).unsqueeze(0)
        active_drift = (drift_phase > 0.0) & (drift_phase < 1.0)
    step_index = None
    if dataset.camera_step_event is not None:
        events = dataset.camera_step_event[raw_episode]
        event_positions = np.flatnonzero(events)
        if event_positions.size:
            candidates = np.flatnonzero(positions >= int(event_positions[0]))
            if candidates.size:
                step_index = int(candidates[0])
    if require_step_event:
        if dataset.camera_step_event is None:
            raise ValueError("evaluation condition requires a camera step event")
        event_count = int(
            np.asarray(dataset.camera_step_event[raw_episode], dtype=bool).sum()
        )
        if event_count != 1:
            raise ValueError(
                "evaluation condition requires exactly one camera step event"
            )
        if step_index is None or step_index < dataset.history_size:
            raise ValueError(
                "camera step event must map into the scored sequence region"
            )
    if hidden_step_length < 0:
        raise ValueError("hidden_step_length must be non-negative")
    if hidden_step_length > 0:
        if step_index is None:
            raise ValueError("hidden-step control requires a camera step event")
        hidden_end = min(step_index + hidden_step_length, measurement_valid.shape[1])
        measurement_valid[:, step_index:hidden_end] = False
    if dropout_start_offset is not None:
        if dropout_length <= 0:
            raise ValueError("dropout_length must be positive when dropout is enabled")
        origin = step_index if step_index is not None else dataset.history_size
        start = origin + int(dropout_start_offset)
        end = min(start + int(dropout_length), measurement_valid.shape[1])
        if start < dataset.history_size or start >= end:
            raise ValueError("dropout interval must overlap scored sequence frames")
        measurement_valid[:, start:end] = False
    return V2Sequence(
        pixels=pixels.to(device),
        proprio=proprio.to(device),
        actions=actions.to(device),
        q_nominal_px=q_nominal.to(device),
        q_corrected_px=q_corrected.to(device),
        q_true_px=q_true.to(device),
        keypoints_xyz=xyz.to(device),
        nominal_valid=nominal_valid.to(device),
        measurement_valid=measurement_valid.to(device),
        true_valid=true_valid.to(device),
        step_index=step_index,
        active_drift=(
            active_drift.to(device) if active_drift is not None else None
        ),
        drift_phase=(drift_phase.to(device) if drift_phase is not None else None),
    )


__all__ = [
    "SequentialModeResult",
    "aggregate_sequential_metrics",
    "V2Sequence",
    "evaluate_v2_sequence",
    "oracle_diagnostic_sequence",
    "run_classic_baselines",
    "run_v2_feedback_mode",
    "sequence_from_dataset",
    "sequential_metrics",
]
