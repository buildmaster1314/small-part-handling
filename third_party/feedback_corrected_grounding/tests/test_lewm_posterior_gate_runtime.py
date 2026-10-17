from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from fcpg.lewm_posterior_adapter import (
    LeWMPosteriorAdapter,
    LeWMPosteriorAdapterOutput,
)
from fcpg.lewm_posterior_gate_data import build_fresh_posterior_gate_examples
from fcpg.lewm_posterior_gate_cache import PosteriorGateCache
from fcpg.lewm_posterior_gate_runtime import (
    ACTION_BLIND_METHOD,
    ACTION_SHUFFLE_METHOD,
    ADAPTER_METHOD,
    FIXED_BETA_METHOD,
    MEASUREMENT_METHOD,
    NO_LEWM_METHOD,
    NO_WRITEBACK_METHOD,
    POSTERIOR_GATE_RUNTIME_FORBIDDEN_MODEL_INPUTS,
    build_action_shuffle_indices,
    evaluate_posterior_gate_cache,
    evaluate_recursive_posterior_gate,
    run_recursive_posterior_gate,
)
from fcpg.lewm_posterior_gate_statistics import (
    summarize_posterior_gate_statistics,
)


ANCHORS = 2
LATENT_DIM = 2
RAW_ACTION_DIM = 1
FRAMES = 40


class _IncrementLeWM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def predict(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        self.calls.append((history.detach().clone(), action.detach().clone()))
        return history + 1.0


class _PriorAdapter(LeWMPosteriorAdapter):
    def __init__(self) -> None:
        super().__init__(
            latent_dim=LATENT_DIM,
            innovation_dim=LATENT_DIM,
            confidence_dim=6,
            mask_history_dim=3,
            action_summary_dim=5,
            hidden_dim=0,
        )
        self.seen: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(
        self,
        z_measurement: torch.Tensor,
        z_prior: torch.Tensor,
        innovation: torch.Tensor,
        confidence: torch.Tensor,
        mask_history: torch.Tensor,
        action_summary: torch.Tensor,
    ) -> LeWMPosteriorAdapterOutput:
        self.seen.append(
            (z_prior.detach().clone(), action_summary.detach().clone())
        )
        del innovation, confidence, mask_history
        beta = torch.ones(
            *z_measurement.shape[:-1],
            1,
            dtype=z_measurement.dtype,
            device=z_measurement.device,
        )
        return LeWMPosteriorAdapterOutput(z_post=z_prior, beta=beta)


def _examples(*, episodes: int = 2):
    task: list[str] = []
    episode_ids: list[int] = []
    frames: list[int] = []
    actions: list[torch.Tensor] = []
    measurements: list[torch.Tensor] = []
    canonicals: list[torch.Tensor] = []
    q_nominal: list[torch.Tensor] = []
    q_corrected: list[torch.Tensor] = []
    valid: list[torch.Tensor] = []
    dropout: list[torch.Tensor] = []
    recovery: list[torch.Tensor] = []
    anchor_offsets = torch.arange(ANCHORS, dtype=torch.float32)[:, None] * 10.0
    latent_offsets = torch.arange(LATENT_DIM, dtype=torch.float32)[None, :] * 0.1

    for episode in range(episodes):
        for frame in range(FRAMES):
            base = float(episode * 100 + frame)
            state = base + anchor_offsets + latent_offsets
            nominal = torch.tensor(
                [[30.0 + frame, 40.0], [50.0 + frame, 60.0]],
                dtype=torch.float32,
            )
            active = torch.zeros(ANCHORS, dtype=torch.bool)
            recovering = torch.zeros(ANCHORS, dtype=torch.bool)
            if 15 <= frame < 30:
                active[1] = True
            if 30 <= frame < 40:
                recovering[1] = True
            task.append("pick-place-v3")
            episode_ids.append(episode)
            frames.append(frame)
            actions.append(torch.tensor([episode * 1_000 + frame], dtype=torch.float32))
            measurements.append(state)
            canonicals.append(state + 7.0)
            q_nominal.append(nominal)
            q_corrected.append(nominal + 1.0)
            valid.append(torch.ones(ANCHORS, dtype=torch.bool))
            dropout.append(active)
            recovery.append(recovering)

    return build_fresh_posterior_gate_examples(
        task=task,
        episode=episode_ids,
        frame=frames,
        action=torch.stack(actions),
        measurement_embedding=torch.stack(measurements),
        canonical_label=torch.stack(canonicals),
        q_nominal=torch.stack(q_nominal),
        q_corrected=torch.stack(q_corrected),
        valid_nominal=torch.stack(valid),
        anchor_dropout_active=torch.stack(dropout),
        anchor_recovery_segment=torch.stack(recovery),
        observation_action_alignment="pre_action",
    )


def _run(examples, methods):
    model = _IncrementLeWM()
    result = run_recursive_posterior_gate(
        model,
        examples.deployment,
        examples.metadata,
        adapter=_PriorAdapter(),
        image_width=224,
        image_height=224,
        methods=methods,
    )
    return model, result


def test_recursive_writeback_and_no_writeback_follow_distinct_histories() -> None:
    examples = _examples(episodes=1)
    _, result = _run(examples, (ADAPTER_METHOD, NO_WRITEBACK_METHOD))
    adapter = result.trajectories[ADAPTER_METHOD]
    no_writeback = result.trajectories[NO_WRITEBACK_METHOD]

    first, second = 0, 1
    torch.testing.assert_close(
        adapter.pre_prior[first], examples.deployment.measurement_history[first, -1] + 1.0
    )
    torch.testing.assert_close(
        adapter.pre_prior[second], adapter.post_update[first] + 1.0
    )
    torch.testing.assert_close(
        no_writeback.pre_prior[second],
        examples.deployment.measurement_current[first] + 1.0,
    )
    assert not torch.equal(adapter.pre_prior[second], no_writeback.pre_prior[second])
    torch.testing.assert_close(adapter.writeback, adapter.post_update)
    torch.testing.assert_close(
        no_writeback.writeback, examples.deployment.measurement_current
    )


def test_every_anchor_is_written_back_including_the_indirect_anchor() -> None:
    examples = _examples(episodes=1)
    _, result = _run(examples, (ADAPTER_METHOD,))
    trajectory = result.trajectories[ADAPTER_METHOD]

    # Anchor 1 is directly occluded. Anchor 0 must still take the posterior path.
    assert bool(examples.evaluation.dropout_active[0, 1])
    assert not bool(examples.evaluation.dropout_active[0, 0])
    torch.testing.assert_close(trajectory.writeback[0, 0], trajectory.post_update[0, 0])
    torch.testing.assert_close(
        trajectory.pre_prior[1, 0], trajectory.post_update[0, 0] + 1.0
    )
    assert not torch.equal(
        trajectory.writeback[0, 0], examples.deployment.measurement_current[0, 0]
    )


def test_phase_streams_start_from_their_own_measurement_history() -> None:
    examples = _examples(episodes=1)
    _, result = _run(examples, (ADAPTER_METHOD,))
    prior = result.trajectories[ADAPTER_METHOD].pre_prior

    for stream_slice in examples.stream_slices:
        start = stream_slice.start
        torch.testing.assert_close(
            prior[start], examples.deployment.measurement_history[start, -1] + 1.0
        )
    starts = [stream_slice.start for stream_slice in examples.stream_slices]
    assert len({float(prior[index, 0, 0]) for index in starts}) == 5


def test_action_blind_and_no_lewm_are_distinct_comparators() -> None:
    examples = _examples(episodes=1)
    model = _IncrementLeWM()
    action_blind_adapter = _PriorAdapter()
    no_lewm_adapter = _PriorAdapter()
    result = run_recursive_posterior_gate(
        model,
        examples.deployment,
        examples.metadata,
        adapter=_PriorAdapter(),
        action_blind_adapter=action_blind_adapter,
        no_lewm_adapter=no_lewm_adapter,
        image_width=224,
        image_height=224,
        methods=(ACTION_BLIND_METHOD, NO_LEWM_METHOD),
    )
    action_blind = result.trajectories[ACTION_BLIND_METHOD]
    no_lewm = result.trajectories[NO_LEWM_METHOD]

    assert ACTION_BLIND_METHOD == "action_blind"
    assert NO_LEWM_METHOD == "no_lewm"
    assert len(model.calls) == examples.sample_count
    assert action_blind_adapter.seen
    assert no_lewm_adapter.seen
    assert all(
        torch.count_nonzero(action).item() == 0
        for _, action in action_blind_adapter.seen
    )
    assert all(
        torch.count_nonzero(action).item() > 0
        for _, action in no_lewm_adapter.seen
    )
    torch.testing.assert_close(
        action_blind.pre_prior[0],
        examples.deployment.measurement_history[0, -1] + 1.0,
    )
    torch.testing.assert_close(
        action_blind.pre_prior[1], action_blind.post_update[0] + 1.0
    )
    torch.testing.assert_close(
        no_lewm.pre_prior[0], examples.deployment.measurement_history[0, -1]
    )
    torch.testing.assert_close(
        no_lewm.pre_prior[1], no_lewm.post_update[0]
    )


def test_target_frame_action_is_rejected_before_model_forward() -> None:
    examples = _examples(episodes=1)
    first = examples.metadata[0]
    invalid_first = replace(
        first,
        action_frames=(
            *first.action_frames[:-1],
            (*first.action_frames[-1], first.target_frame),
        ),
    )
    invalid = replace(
        examples, metadata=(invalid_first, *examples.metadata[1:])
    )
    model = _IncrementLeWM()

    with pytest.raises(ValueError, match="action_frames|target-frame action"):
        run_recursive_posterior_gate(
            model,
            invalid.deployment,
            invalid.metadata,
            adapter=_PriorAdapter(),
            image_width=224,
            image_height=224,
            methods=(ADAPTER_METHOD,),
        )
    assert model.calls == []

    valid_model, _ = _run(examples, (ADAPTER_METHOD,))
    first_actions = valid_model.calls[0][1][0].reshape(3, 5, RAW_ACTION_DIM)
    target_action = torch.tensor(float(first.target_frame))
    assert not bool((first_actions[..., 0] == target_action).any())


def test_all_modes_emit_statistics_compatible_phase_reduced_rows() -> None:
    examples = _examples(episodes=2)
    shuffle = build_action_shuffle_indices(examples.metadata)
    model = _IncrementLeWM()
    result = evaluate_recursive_posterior_gate(
        model,
        examples,
        adapter=_PriorAdapter(),
        image_width=224,
        image_height=224,
        fixed_beta=0.5,
        action_shuffle_indices=shuffle,
    )
    expected_methods = {
        MEASUREMENT_METHOD,
        FIXED_BETA_METHOD,
        ADAPTER_METHOD,
        NO_LEWM_METHOD,
        ACTION_BLIND_METHOD,
        NO_WRITEBACK_METHOD,
        ACTION_SHUFFLE_METHOD,
    }
    assert set(result.trajectories) == expected_methods
    assert len(result.records) == 2 * ANCHORS * 2 * 5 * len(expected_methods)
    assert set(result.records[0]) == {
        "task",
        "physical_episode_id",
        "anchor",
        "estimand",
        "event",
        "method",
        "mse",
        "is_direct_anchor",
    }
    assert {row["estimand"] for row in result.records} == {
        "pre_prior",
        "post_update",
    }
    assert all("phase" not in row for row in result.records)
    assert all(math.isfinite(float(row["mse"])) for row in result.records)

    clean_records = []
    for row in result.records:
        clean = dict(row)
        clean["physical_episode_id"] = f"clean-{row['physical_episode_id']}"
        clean_records.append(clean)
    summary = summarize_posterior_gate_statistics(
        result.records,
        clean_records,
        bootstrap_samples=10,
        bootstrap_seed=3,
    )
    assert summary["coverage"]["complete_paired_support"] is True
    assert summary["coverage"]["record_count"] == len(result.records)


def test_canonical_changes_scores_only_and_forbidden_fields_are_not_model_inputs() -> None:
    examples = _examples(episodes=1)
    model_a = _IncrementLeWM()
    first = evaluate_recursive_posterior_gate(
        model_a,
        examples,
        adapter=_PriorAdapter(),
        image_width=224,
        image_height=224,
        methods=(ADAPTER_METHOD,),
    )
    changed_labels = replace(
        examples.labels, canonical_current=examples.labels.canonical_current + 100.0
    )
    changed = replace(examples, labels=changed_labels)
    model_b = _IncrementLeWM()
    second = evaluate_recursive_posterior_gate(
        model_b,
        changed,
        adapter=_PriorAdapter(),
        image_width=224,
        image_height=224,
        methods=(ADAPTER_METHOD,),
    )

    torch.testing.assert_close(
        first.trajectories[ADAPTER_METHOD].post_update,
        second.trajectories[ADAPTER_METHOD].post_update,
    )
    assert all(
        torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        for a, b in zip(model_a.calls, model_b.calls, strict=True)
    )
    assert [row["mse"] for row in first.records] != [
        row["mse"] for row in second.records
    ]
    assert set(POSTERIOR_GATE_RUNTIME_FORBIDDEN_MODEL_INPUTS) == {
        "q_true",
        "canonical",
        "schedule",
        "measurement_valid",
    }


def test_verified_cache_bridge_selects_only_partitioned_runtime_fields() -> None:
    examples = _examples(episodes=1)
    evaluation = {
        "task_name": np.asarray([row.task for row in examples.metadata]),
        "episode_id": np.asarray([row.episode_id for row in examples.metadata]),
        "task_index": np.full(examples.sample_count, 500, dtype=np.int64),
        "env_seed": np.full(examples.sample_count, 600, dtype=np.int64),
        "phase_index": np.asarray([row.phase_index for row in examples.metadata]),
        "stream_offset": np.asarray([row.stream_offset for row in examples.metadata]),
        "frame_phase": np.asarray([row.frame_phase for row in examples.metadata]),
        "event": np.asarray([row.event for row in examples.metadata]),
        "event_index": np.asarray([row.event_index for row in examples.metadata]),
        "target_frame": np.asarray([row.target_frame for row in examples.metadata]),
        "target_raw_index": np.asarray(
            [row.target_raw_index for row in examples.metadata]
        ),
        "history_frames": np.asarray(
            [row.history_frames for row in examples.metadata]
        ),
        "action_frames": np.asarray([row.action_frames for row in examples.metadata]),
        "dropout_active": examples.evaluation.dropout_active.numpy(),
        "recovery_active": examples.evaluation.recovery_active.numpy(),
        # These cache fields must never cross the runtime model boundary.
        "q_true_current": np.full((examples.sample_count, ANCHORS, 2), np.nan),
        "measurement_valid_current": np.zeros(
            (examples.sample_count, ANCHORS), dtype=np.bool_
        ),
    }
    cache = PosteriorGateCache(
        path=Path("/unused/posterior-cache"),
        metadata={"image_width": 224, "image_height": 224},
        deployment={
            name: value.detach().cpu().numpy()
            for name, value in examples.deployment.items()
        },
        labels={
            "canonical_current": examples.labels.canonical_current.numpy()
        },
        evaluation=evaluation,
        file_hashes={},
    )
    model = _IncrementLeWM()
    result = evaluate_posterior_gate_cache(
        model,
        cache,
        adapter=_PriorAdapter(),
        methods=(ADAPTER_METHOD,),
    )

    assert result.records
    assert {
        row["physical_episode_id"] for row in result.records
    } == {"task_index=500|env_seed=600|episode_id=0"}
    assert model.calls
