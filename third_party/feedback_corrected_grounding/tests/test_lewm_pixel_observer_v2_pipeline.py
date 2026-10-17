from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from fcpg.lewm_pixel_observer_v2 import PixelResidualVelocityInnovationAdapter
from fcpg.lewm_pixel_observer_v2_data import PixelObserverV2WindowDataset
from fcpg.lewm_pixel_observer_v2_eval import (
    _measurement_observability_before,
    aggregate_sequential_metrics,
    oracle_diagnostic_sequence,
    V2Sequence,
    run_v2_feedback_mode,
    sequence_from_dataset,
    sequential_metrics,
)
from fcpg.lewm_pixel_observer_v2_runtime import CausalLeWMPixelObserverV2Runtime
from fcpg.lewm_pixel_observer_v2_training import _certify_finished_run
from fcpg.lewm_pixel_observer_v2_training import _history_state
from fcpg.lewm_pixel_observer_v2_training import pixel_observer_v2_loss
from fcpg.lewm_pixel_observer_v2_training import validate_v2_base_history_binding
from fcpg.lewm_pixel_observer_v2_training import validate_v2_completed_training_budget
from fcpg.lewm_provenance import sha256_file


class _FrameEncoder(nn.Module):
    def __init__(self, num_keypoints: int) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints


class _Base(nn.Module):
    def __init__(self, *, history_size: int = 2, latent_dim: int = 4) -> None:
        super().__init__()
        self.history_size = history_size
        self.latent_dim = latent_dim
        self.action_dim = 2
        self.frame_encoder = _FrameEncoder(2)
        self.weight = nn.Parameter(torch.ones(1))

    def encode_frames(self, batch):
        reference = batch["q_nominal"]
        embedding = reference.new_zeros(
            reference.shape[0], reference.shape[1], self.latent_dim
        )
        embedding[..., 0] = batch["q_corrected"].mean(dim=(-1, -2))
        return embedding, embedding

    def predict(self, embedding, actions):
        result = embedding.clone()
        result[..., :2] += actions[..., :2]
        return result


def _runtime(*, use_latent: bool = True) -> CausalLeWMPixelObserverV2Runtime:
    base = _Base()
    adapter = PixelResidualVelocityInnovationAdapter(
        latent_dim=4,
        num_keypoints=2,
        image_size=(16, 16),
        hidden_dim=8,
        max_velocity_innovation_px=1.0,
        residual_scale_px=8.0,
        velocity_scale_px=1.0,
        use_predicted_latent=use_latent,
    )
    return CausalLeWMPixelObserverV2Runtime(base, adapter)


def _batch() -> dict[str, torch.Tensor]:
    batch, frames, anchors = 2, 3, 2
    q_nominal = torch.full((batch, frames, anchors, 2), 4.0)
    correction = torch.tensor([0.2, -0.1]).view(1, 1, 1, 2)
    q_corrected = q_nominal + correction
    return {
        "pixels": torch.zeros(batch, frames, 3, 8, 8),
        "proprio": torch.zeros(batch, frames, 4),
        "action": torch.zeros(batch, frames - 1, 2),
        "q_nominal": q_nominal,
        "q_corrected_a": q_corrected,
        "keypoints_xyz": torch.zeros(batch, frames, anchors, 3),
        "valid_nominal": torch.ones(batch, frames, anchors, dtype=torch.bool),
        "measurement_valid": torch.ones(batch, frames, anchors, dtype=torch.bool),
        "measurement_valid_a": torch.ones(batch, frames, anchors, dtype=torch.bool),
        "prior_supervision_valid_target": torch.ones(
            batch, anchors, dtype=torch.bool
        ),
        "q_true_history": q_nominal[:, :-1] + 0.25,
        "valid_true_history": torch.ones(batch, frames - 1, anchors, dtype=torch.bool),
        "q_true_target": q_nominal[:, -1] + 0.5,
        "valid_true_target": torch.ones(batch, anchors, dtype=torch.bool),
    }


def test_target_frame_measurement_and_oracle_cannot_change_prior() -> None:
    runtime = _runtime()
    first = _batch()
    output_first = pixel_observer_v2_loss(
        runtime,
        first,
        target_mode="oracle_ceiling",
        huber_delta_px=0.25,
    )
    second = {name: value.clone() for name, value in first.items()}
    second["q_corrected_a"][:, -1] += 100.0
    second["q_true_target"] -= 50.0
    output_second = pixel_observer_v2_loss(
        runtime,
        second,
        target_mode="oracle_ceiling",
        huber_delta_px=0.25,
    )
    torch.testing.assert_close(
        output_first.predictions[0], output_second.predictions[0]
    )
    assert not torch.equal(
        output_first.target_residual_px, output_second.target_residual_px
    )
    third = {name: value.clone() for name, value in first.items()}
    third["q_true_history"][:, -1] += 3.0
    output_third = pixel_observer_v2_loss(
        runtime, third, target_mode="oracle_ceiling", huber_delta_px=0.25
    )
    assert not torch.equal(output_first.predictions[0], output_third.predictions[0])


def _write_dataset(
    tmp_path, *, step_frame: int = 4, step_annotation: str = "frame"
):
    count, anchors = 12, 2
    data = tmp_path / "source.npz"
    q_nominal = np.zeros((count, anchors, 2), dtype=np.float32)
    q_corrected = q_nominal + np.arange(count, dtype=np.float32)[:, None, None] * 0.1
    payload = dict(
        rgb=np.zeros((count, 8, 8, 3), dtype=np.uint8),
        obs=np.zeros((count, 8), dtype=np.float32),
        action=np.zeros((count, 2), dtype=np.float32),
        task_name=np.full(count, "task"),
        episode_id=np.zeros(count, dtype=np.int64),
        frame_id=np.arange(count, dtype=np.int64),
        keypoints_xyz=np.zeros((count, anchors, 3), dtype=np.float32),
        q_nominal=q_nominal,
        valid_nominal=np.ones((count, anchors), dtype=bool),
        q_true=q_corrected,
        valid_true=np.ones((count, anchors), dtype=bool),
        observation_action_alignment=np.asarray("pre_action"),
    )
    event = np.arange(count, dtype=np.int64) == step_frame
    if step_annotation in {"frame", "all", "mismatch"}:
        payload["camera_step_frame"] = np.full(
            count, step_frame, dtype=np.int32
        )
    if step_annotation in {"event", "all", "multiple"}:
        payload["camera_step_event"] = (
            event | (np.arange(count) == step_frame + 1)
            if step_annotation == "multiple"
            else event
        )
    if step_annotation in {"onset", "all", "mismatch"}:
        payload["camera_intervention_onset"] = (
            np.arange(count, dtype=np.int64)
            == step_frame + (1 if step_annotation == "mismatch" else 0)
        )
    if step_annotation not in {
        "none", "frame", "event", "onset", "all", "multiple", "mismatch"
    }:
        raise ValueError("unsupported test step annotation")
    np.savez_compressed(data, **payload)
    cache_a = tmp_path / "a.npy"
    cache_b = tmp_path / "b.npy"
    np.save(cache_a, q_corrected)
    np.save(cache_b, q_corrected + 0.01)
    return data, cache_a, cache_b


def test_oracle_diagnostic_preserves_evaluator_censoring(tmp_path) -> None:
    data, cache_a, _ = _write_dataset(tmp_path, step_frame=4)
    dataset = PixelObserverV2WindowDataset(
        data_path=data,
        correction_cache_a=cache_a,
        correction_cache_b=None,
        target_mode="oracle_ceiling",
        history_size=2,
        frame_skip=1,
    )
    sequence = sequence_from_dataset(
        dataset, 0, device=torch.device("cpu"), hidden_step_length=2
    )
    censored = ~sequence.measurement_valid.clone()
    assert bool(censored[:, 4:6].all())
    oracle = oracle_diagnostic_sequence(sequence)
    torch.testing.assert_close(oracle.q_corrected_px, sequence.q_true_px)
    torch.testing.assert_close(oracle.measurement_valid, sequence.measurement_valid)
    assert bool((~oracle.measurement_valid[:, 4:6]).all())


def test_frame_skip_transition_crossing_step_is_masked(tmp_path) -> None:
    data, cache, _ = _write_dataset(tmp_path)
    dataset = PixelObserverV2WindowDataset(
        data_path=data,
        correction_cache_a=cache,
        correction_cache_b=None,
        target_mode="oracle_ceiling",
        history_size=2,
        frame_skip=3,
    )
    crossing = [
        sample
        for sample in (dataset[index] for index in range(len(dataset)))
        if bool(sample["camera_step_crossed_target"])
    ]
    assert crossing
    assert all(not bool(sample["prior_supervision_valid_target"].any()) for sample in crossing)
    later = [
        dataset[index]
        for index in range(len(dataset))
        if int(dataset[index]["frame_id"][-2]) >= 4
    ]
    assert later
    assert all(bool(sample["prior_supervision_valid_target"].all()) for sample in later)


def test_hidden_step_control_masks_measurement_from_step(tmp_path) -> None:
    data, cache, _ = _write_dataset(tmp_path)
    dataset = PixelObserverV2WindowDataset(
        data_path=data,
        correction_cache_a=cache,
        correction_cache_b=None,
        target_mode="oracle_ceiling",
        history_size=2,
        frame_skip=1,
    )
    sequence = sequence_from_dataset(
        dataset, 0, device=torch.device("cpu"), hidden_step_length=2
    )
    assert sequence.step_index == 4
    assert not bool(sequence.measurement_valid[:, 4:6].any())
    assert bool(sequence.measurement_valid[:, 6:].all())


@pytest.mark.parametrize("step_annotation", ["event", "onset", "all"])
def test_camera_event_aliases_drive_training_and_evaluation_masks(
    tmp_path, step_annotation
) -> None:
    data, cache, _ = _write_dataset(
        tmp_path, step_annotation=step_annotation
    )
    dataset = PixelObserverV2WindowDataset(
        data_path=data,
        correction_cache_a=cache,
        correction_cache_b=None,
        target_mode="oracle_ceiling",
        history_size=2,
        frame_skip=1,
    )
    sequence = sequence_from_dataset(
        dataset,
        0,
        device=torch.device("cpu"),
        hidden_step_length=2,
        require_step_event=True,
    )
    assert sequence.step_index == 4
    assert not bool(sequence.measurement_valid[:, 4:6].any())
    crossing = [
        dataset[index]
        for index in range(len(dataset))
        if bool(dataset[index]["camera_step_crossed_target"])
    ]
    assert crossing
    assert all(
        not bool(sample["prior_supervision_valid_target"].any())
        for sample in crossing
    )


def test_camera_step_between_sampled_frames_maps_to_next_sample(tmp_path) -> None:
    data, cache, _ = _write_dataset(
        tmp_path, step_frame=4, step_annotation="onset"
    )
    dataset = PixelObserverV2WindowDataset(
        data_path=data,
        correction_cache_a=cache,
        correction_cache_b=None,
        target_mode="oracle_ceiling",
        history_size=2,
        frame_skip=3,
    )
    sequence = sequence_from_dataset(
        dataset,
        0,
        device=torch.device("cpu"),
        require_step_event=True,
    )
    assert sequence.step_index == 2


def test_camera_step_annotations_must_agree(tmp_path) -> None:
    data, cache, _ = _write_dataset(tmp_path, step_annotation="mismatch")
    with pytest.raises(ValueError, match="camera step annotations disagree"):
        PixelObserverV2WindowDataset(
            data_path=data,
            correction_cache_a=cache,
            correction_cache_b=None,
            target_mode="oracle_ceiling",
            history_size=2,
            frame_skip=1,
        )


@pytest.mark.parametrize(
    ("step_annotation", "step_frame", "message"),
    [
        ("none", 4, "requires a camera step event"),
        ("multiple", 4, "requires exactly one camera step event"),
        ("event", 1, "scored sequence region"),
    ],
)
def test_required_camera_step_rejects_invalid_episode(
    tmp_path, step_annotation, step_frame, message
) -> None:
    data, cache, _ = _write_dataset(
        tmp_path, step_frame=step_frame, step_annotation=step_annotation
    )
    dataset = PixelObserverV2WindowDataset(
        data_path=data,
        correction_cache_a=cache,
        correction_cache_b=None,
        target_mode="oracle_ceiling",
        history_size=2,
        frame_skip=1,
    )
    with pytest.raises(ValueError, match=message):
        sequence_from_dataset(
            dataset,
            0,
            device=torch.device("cpu"),
            require_step_event=True,
        )


def test_paired_caches_must_be_independent(tmp_path) -> None:
    data, raw_a, raw_b = _write_dataset(tmp_path)

    def cache(name: str, source, view: str, seed: int):
        path = tmp_path / name
        path.mkdir()
        array_path = path / "q_corrected.npy"
        np.save(array_path, np.load(source))
        (path / "metadata.json").write_text(
            json.dumps(
                {
                    "independent_measurement_contract": "independent_fcpg_measurement_v1",
                    "measurement_view_id": view,
                    "measurement_generation_seed": seed,
                    "source_data_sha256": sha256_file(data),
                    "q_corrected_sha256": sha256_file(array_path),
                }
            )
        )
        return path

    cache_a = cache("cache_a", raw_a, "view_a", 10)
    cache_b = cache("cache_b", raw_b, "view_b", 11)
    paired = PixelObserverV2WindowDataset(
        data_path=data,
        correction_cache_a=cache_a,
        correction_cache_b=cache_b,
        target_mode="paired_deployment",
        history_size=2,
        frame_skip=1,
        task_names=["task"],
    )
    assert paired.provenance()["task_names"] == ["task"]
    assert paired.provenance()["measurement_a_sha256"] != paired.provenance()[
        "measurement_b_sha256"
    ]
    with pytest.raises(ValueError, match="independent"):
        PixelObserverV2WindowDataset(
            data_path=data,
            correction_cache_a=cache_a,
            correction_cache_b=cache_a,
            target_mode="paired_deployment",
            history_size=2,
            frame_skip=1,
        )
    same_lineage = cache("same_lineage", raw_b, "view_a", 12)
    with pytest.raises(ValueError, match="lineage"):
        PixelObserverV2WindowDataset(
            data_path=data,
            correction_cache_a=cache_a,
            correction_cache_b=same_lineage,
            target_mode="paired_deployment",
            history_size=2,
            frame_skip=1,
        )


def test_invalid_single_measurement_target_is_not_supervised() -> None:
    runtime = _runtime()
    batch = _batch()
    batch["measurement_valid_a"][:, -1] = False
    output = pixel_observer_v2_loss(
        runtime,
        batch,
        target_mode="single_measurement_baseline",
        huber_delta_px=0.25,
    )
    assert not bool(output.valid.any())
    assert float(output.loss) == 0.0
    output.loss.backward()


@pytest.mark.parametrize(
    "change",
    [
        {"step": 1},
        {"checkpoint_optimizer_step": 3},
        {"target_optimizer_steps": 0, "final_optimizer_steps": 0},
        {"final_optimizer_steps": 3},
    ],
)
def test_completed_budget_rejects_cross_field_tampering(change) -> None:
    payload = {
        "step": 2,
        "training_budget": {
            "target_optimizer_steps": 2,
            "checkpoint_optimizer_step": 2,
            "run_completed": True,
            "final_optimizer_steps": 2,
        },
    }
    for name, value in change.items():
        if name == "step":
            payload["step"] = value
        else:
            payload["training_budget"][name] = value
    with pytest.raises(ValueError, match="cross-fields"):
        validate_v2_completed_training_budget(payload, require_endpoint=True)


def test_pre_final_best_retains_checkpoint_step_and_records_run_endpoint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pixel_observer_v2_best.pt"
    payload = {
        "step": 3,
        "epoch": 1,
        "resolved_config_sha256": "config",
        "base_checkpoint_sha256": "base",
        "adapter_state_sha256": "adapter",
        "data_provenance_sha256": "data",
        "target_mode": "paired_deployment",
        "training_budget": {
            "target_optimizer_steps": 10,
            "checkpoint_optimizer_step": 3,
            "run_completed": False,
            "final_optimizer_steps": None,
        },
    }
    torch.save(payload, path)
    _certify_finished_run(path, 10)
    certified = torch.load(path, map_location="cpu", weights_only=False)
    assert certified["step"] == 3
    assert certified["training_budget"]["checkpoint_optimizer_step"] == 3
    assert certified["training_budget"]["final_optimizer_steps"] == 10
    validate_v2_completed_training_budget(certified, require_endpoint=False)
    with pytest.raises(ValueError, match="cross-fields"):
        validate_v2_completed_training_budget(certified, require_endpoint=True)


def test_config_history_must_match_frozen_base() -> None:
    base = _Base(history_size=2)
    assert validate_v2_base_history_binding(
        {"data": {"history_size": 2}}, base
    ) == 2
    with pytest.raises(ValueError, match="must match"):
        validate_v2_base_history_binding(
            {"data": {"history_size": 3}}, base
        )


def test_history_dropout_propagates_constant_velocity() -> None:
    nominal = torch.zeros(1, 3, 1, 2)
    corrected = nominal.clone()
    corrected[:, 1] = 1.0
    corrected[:, 2] = 999.0
    valid = torch.tensor([[[True], [True], [False]]])
    posterior, velocity, states, _ = _history_state(
        nominal, corrected, valid, history_size=3
    )
    torch.testing.assert_close(velocity, torch.ones_like(velocity))
    torch.testing.assert_close(posterior, torch.full_like(posterior, 2.0))
    torch.testing.assert_close(states[:, 2], torch.full_like(posterior, 2.0))


def test_24px_velocity_innovation_can_damp_18px_step() -> None:
    adapter = PixelResidualVelocityInnovationAdapter(
        latent_dim=4,
        num_keypoints=2,
        image_size=(16, 16),
        hidden_dim=8,
        max_velocity_innovation_px=24.0,
        residual_scale_px=8.0,
        velocity_scale_px=8.0,
    )
    with torch.no_grad():
        adapter.net[-1].weight.zero_()
        adapter.net[-1].bias.fill_(torch.atanh(torch.tensor(-18.0 / 24.0)))
    valid = torch.ones(1, 2, dtype=torch.bool)
    output = adapter.predict_prior(
        {
            "predicted_embedding": torch.zeros(1, 4),
            "posterior_residual_px": torch.zeros(1, 2, 2),
            "residual_velocity_px": torch.full((1, 2, 2), 18.0),
            "previous_q_nominal_px": torch.zeros(1, 2, 2),
            "current_q_nominal_px": torch.zeros(1, 2, 2),
            "previous_keypoints_xyz": torch.zeros(1, 2, 3),
            "current_keypoints_xyz": torch.zeros(1, 2, 3),
            "previous_nominal_valid": valid,
            "current_nominal_valid": valid,
            "previous_measurement_valid": valid,
            "measurement_age_steps": torch.zeros(1, 2),
        }
    )
    torch.testing.assert_close(
        output.velocity_px, torch.zeros_like(output.velocity_px), atol=1e-5, rtol=0.0
    )


def test_no_latent_preserves_exact_parameter_count() -> None:
    with_latent = _runtime(use_latent=True)
    without_latent = _runtime(use_latent=False)
    assert sum(parameter.numel() for parameter in with_latent.adapter.parameters()) == sum(
        parameter.numel() for parameter in without_latent.adapter.parameters()
    )


class _StatePriorRuntime:
    history_size = 2

    def predict_prior(self, **values):
        residual = values["posterior_residual_px"] + values["residual_velocity_px"]
        return SimpleNamespace(residual_px=residual, velocity_px=values["residual_velocity_px"])


def test_full_writeback_and_no_writeback_have_distinct_state_paths() -> None:
    frames, anchors = 7, 2
    nominal = torch.zeros(1, frames, anchors, 2)
    residual = torch.arange(frames, dtype=torch.float32).square().view(1, frames, 1, 1)
    residual = residual.expand(1, frames, anchors, 2)
    measurement_valid = torch.ones(1, frames, anchors, dtype=torch.bool)
    measurement_valid[:, 4:6] = False
    sequence = V2Sequence(
        pixels=torch.zeros(1, frames, 3, 8, 8),
        proprio=torch.zeros(1, frames, 4),
        actions=torch.zeros(1, frames - 1, 2),
        q_nominal_px=nominal,
        q_corrected_px=nominal + residual,
        q_true_px=nominal + residual,
        keypoints_xyz=torch.zeros(1, frames, anchors, 3),
        nominal_valid=torch.ones(1, frames, anchors, dtype=torch.bool),
        measurement_valid=measurement_valid,
        true_valid=torch.ones(1, frames, anchors, dtype=torch.bool),
    )
    no_writeback = run_v2_feedback_mode(
        _StatePriorRuntime(), sequence, mode="posterior_reindex_no_writeback", gain=0.5
    )
    full = run_v2_feedback_mode(
        _StatePriorRuntime(), sequence, mode="full_writeback", gain=0.5
    )
    assert not torch.equal(no_writeback.residual_px, full.residual_px)
    drift_metrics = sequential_metrics(full, sequence)
    assert "post_update_active_drift_epe_px" in drift_metrics
    assert "prior_active_drift_epe_px" in drift_metrics


def test_measurement_age_is_causal_reset_and_clipped() -> None:
    valid = torch.tensor([[[True], [False], [False], [False]]])
    previous, age = _measurement_observability_before(valid, 4, 3)
    assert not bool(previous.any())
    torch.testing.assert_close(age, torch.ones_like(age))
    valid[:, -1] = True
    previous, age = _measurement_observability_before(valid, 4, 3)
    assert bool(previous.all())
    torch.testing.assert_close(age, torch.zeros_like(age))


def test_aggregate_pools_errors_and_preserves_right_censoring() -> None:
    records = [
        {
            "_prior_qtrue_errors_px": [0.0, 10.0],
            "prior_step_time_sampled_steps": 2,
            "prior_step_settled": 1,
            "prior_step_event_epe_px": 4.0,
        },
        {
            "_prior_qtrue_errors_px": [1.0],
            "prior_step_time_sampled_steps": 5,
            "prior_step_settled": 0,
            "prior_step_event_epe_px": 6.0,
        },
    ]
    summary = aggregate_sequential_metrics(records)
    assert summary["prior_qtrue_anchor_frames"] == 3
    assert summary["prior_qtrue_epe_px"] == pytest.approx(11.0 / 3.0)
    assert summary["prior_qtrue_p95_px"] == pytest.approx(9.1)
    assert summary["prior_step_settled"] == 1
    assert summary["prior_step_right_censored"] == 1
    assert summary["prior_step_settled_fraction"] == pytest.approx(0.5)


def test_paired_validity_masks_must_be_identical() -> None:
    runtime = _runtime()
    batch = _batch()
    batch["q_corrected_b"] = batch["q_corrected_a"].clone() + 0.01
    batch["measurement_valid_b"] = batch["measurement_valid_a"].clone()
    batch["measurement_valid_b"][:, 0, 0] = False
    with pytest.raises(ValueError, match="identical source-derived"):
        pixel_observer_v2_loss(
            runtime, batch, target_mode="paired_deployment", huber_delta_px=0.25
        )


def test_history_tail_mask_is_deterministic_and_keeps_target(tmp_path) -> None:
    data, cache, _ = _write_dataset(tmp_path)
    kwargs = {
        "data_path": data,
        "correction_cache_a": cache,
        "correction_cache_b": None,
        "target_mode": "oracle_ceiling",
        "history_size": 2,
        "frame_skip": 1,
        "history_tail_mask": {
            "enabled": True,
            "probability": 1.0,
            "max_steps": 1,
            "seed": 9,
        },
    }
    first = PixelObserverV2WindowDataset(**kwargs)
    second = PixelObserverV2WindowDataset(**kwargs)
    np.testing.assert_array_equal(first.history_tail_lengths, second.history_tail_lengths)
    assert first.provenance()["history_tail_assignment_sha256"] == second.provenance()["history_tail_assignment_sha256"]
    sample = first[0]
    assert not bool(sample["measurement_valid_a"][1].any())
    assert bool(sample["measurement_valid_a"][2].all())
    assert not bool(sample["valid_true_history"][1].any())
