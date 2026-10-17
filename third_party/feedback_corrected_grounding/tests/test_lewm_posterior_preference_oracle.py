from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from fcpg.lewm_posterior_gate_data import build_fresh_posterior_gate_examples
from fcpg.lewm_posterior_gate_runtime import MEASUREMENT_METHOD
from fcpg.lewm_posterior_preference_oracle import (
    CONVEX_ORACLE_METHOD,
    FSM_ORACLE_METHOD,
    HARD_ORACLE_METHOD,
    PRIOR_METHOD,
    SCHEDULE_ORACLE_METHOD,
    SEQUENCE_ORACLE_METHOD,
    _normalize_methods,
    _oracle_beta,
    run_recursive_posterior_preference_oracle,
)


ANCHORS = 2
LATENT_DIM = 2
FRAMES = 40


class _IncrementLeWM(nn.Module):
    def predict(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        del action
        return history + 1.0


class _NonFiniteLeWM(nn.Module):
    def predict(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        del action
        return torch.full_like(history, float("nan"))


def _examples():
    task: list[str] = []
    episodes: list[int] = []
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

    for frame in range(FRAMES):
        state = float(frame) + anchor_offsets + latent_offsets
        nominal = torch.tensor(
            [[30.0 + frame, 40.0], [50.0 + frame, 60.0]], dtype=torch.float32
        )
        active = torch.zeros(ANCHORS, dtype=torch.bool)
        recovering = torch.zeros(ANCHORS, dtype=torch.bool)
        if 15 <= frame < 30:
            active[1] = True
        if 30 <= frame < 40:
            recovering[1] = True
        task.append("pick-place-v3")
        episodes.append(0)
        frames.append(frame)
        actions.append(torch.tensor([frame], dtype=torch.float32))
        measurements.append(state)
        canonicals.append(state.clone())
        q_nominal.append(nominal)
        q_corrected.append(nominal + 1.0)
        valid.append(torch.ones(ANCHORS, dtype=torch.bool))
        dropout.append(active)
        recovery.append(recovering)

    return build_fresh_posterior_gate_examples(
        task=task,
        episode=episodes,
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


def test_hard_and_convex_beta_use_one_source_weight_per_anchor() -> None:
    measurement = torch.tensor([[[0.0, 0.0], [3.0, 3.0]]])
    prior = torch.tensor([[[2.0, 0.0], [3.0, 3.0]]])
    target = torch.tensor([[[1.5, 0.0], [9.0, 9.0]]])
    valid = torch.tensor([[True, True]])

    hard = _oracle_beta(
        HARD_ORACLE_METHOD,
        measurement=measurement,
        prior=prior,
        target=target,
        target_valid=valid,
        event="o1",
    )
    convex = _oracle_beta(
        CONVEX_ORACLE_METHOD,
        measurement=measurement,
        prior=prior,
        target=target,
        target_valid=valid,
        event="o1",
    )

    torch.testing.assert_close(hard[:, 0], torch.ones_like(hard[:, 0]))
    torch.testing.assert_close(convex[:, 0], torch.full_like(convex[:, 0], 0.75))
    # A zero-length source segment fails closed to measurement.
    torch.testing.assert_close(convex[:, 1], torch.zeros_like(convex[:, 1]))

    tiny = torch.tensor([[[0.0, 0.0]]])
    tiny_prior = torch.tensor([[[1.0e-4, 0.0]]])
    tiny_beta = _oracle_beta(
        CONVEX_ORACLE_METHOD,
        measurement=tiny,
        prior=tiny_prior,
        target=tiny_prior,
        target_valid=torch.ones(1, 1, dtype=torch.bool),
        event="o1",
    )
    torch.testing.assert_close(tiny_beta, torch.ones_like(tiny_beta))


def test_invalid_oracle_target_defaults_to_measurement_but_baselines_do_not() -> None:
    value = torch.zeros(1, 1, LATENT_DIM)
    invalid = torch.zeros(1, 1, dtype=torch.bool)
    for method in (HARD_ORACLE_METHOD, CONVEX_ORACLE_METHOD):
        beta = _oracle_beta(
            method,
            measurement=value,
            prior=value + 1.0,
            target=value,
            target_valid=invalid,
            event="o1",
        )
        assert float(beta.item()) == 0.0
    prior = _oracle_beta(
        PRIOR_METHOD,
        measurement=value,
        prior=value + 1.0,
        target=value,
        target_valid=invalid,
        event="o1",
    )
    schedule = _oracle_beta(
        SCHEDULE_ORACLE_METHOD,
        measurement=value,
        prior=value + 1.0,
        target=value,
        target_valid=invalid,
        event="o1",
    )
    assert float(prior.item()) == 1.0
    assert float(schedule.item()) == 1.0


def test_methods_have_independent_recursive_histories_and_only_direct_anchor_switches() -> None:
    examples = _examples()
    target = examples.labels.canonical_current - 100.0
    result = run_recursive_posterior_preference_oracle(
        _IncrementLeWM(),
        examples.deployment,
        examples.metadata,
        target,
        methods=(MEASUREMENT_METHOD, HARD_ORACLE_METHOD, PRIOR_METHOD),
        direct_anchor=1,
    )
    measurement = result.trajectories[MEASUREMENT_METHOD]
    hard = result.trajectories[HARD_ORACLE_METHOD]
    prior = result.trajectories[PRIOR_METHOD]

    torch.testing.assert_close(
        measurement.pre_prior[1], measurement.post_update[0] + 1.0
    )
    torch.testing.assert_close(hard.pre_prior[1], hard.post_update[0] + 1.0)
    assert not torch.equal(measurement.pre_prior[1], hard.pre_prior[1])
    assert torch.count_nonzero(hard.beta[:, 0]).item() == 0
    assert torch.count_nonzero(prior.beta[:, 0]).item() == 0
    assert torch.count_nonzero(prior.beta[:, 1]).item() == prior.beta.shape[0]


def test_cautious_enter_fast_exit_fsm_has_expected_event_pattern() -> None:
    examples = _examples()
    target = examples.labels.canonical_current.clone()
    for index, row in enumerate(examples.metadata):
        if row.event in {"o0", "o1", "o2"}:
            target[index, 1] -= 100.0
    result = run_recursive_posterior_preference_oracle(
        _IncrementLeWM(),
        examples.deployment,
        examples.metadata,
        target,
        methods=(FSM_ORACLE_METHOD,),
        direct_anchor=1,
    )
    beta = result.trajectories[FSM_ORACLE_METHOD].beta[:, 1, 0]
    for start in range(0, beta.numel(), 5):
        torch.testing.assert_close(
            beta[start : start + 5], torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0])
        )


def test_exact_sequence_oracle_minimizes_recursive_stream_cost() -> None:
    examples = _examples()
    target = examples.labels.canonical_current.clone()
    # Alternate endpoint preferences to make writeback consequences relevant.
    for index, row in enumerate(examples.metadata):
        target[index, 1] += -8.0 if row.event in {"o0", "o2"} else 2.0
    result = run_recursive_posterior_preference_oracle(
        _IncrementLeWM(),
        examples.deployment,
        examples.metadata,
        target,
        methods=(HARD_ORACLE_METHOD, SEQUENCE_ORACLE_METHOD),
        direct_anchor=1,
    )
    greedy = result.trajectories[HARD_ORACLE_METHOD]
    sequence = result.trajectories[SEQUENCE_ORACLE_METHOD]

    for start in range(0, len(examples.metadata), 5):
        stop = start + 5
        greedy_mse = torch.square(
            greedy.post_update[start:stop, 1] - target[start:stop, 1]
        ).mean()
        sequence_mse = torch.square(
            sequence.post_update[start:stop, 1] - target[start:stop, 1]
        ).mean()
        assert float(sequence_mse) <= float(greedy_mse) + 1.0e-6
    assert torch.count_nonzero(sequence.beta[:, 0]).item() == 0


def test_exact_sequence_requires_one_selectable_anchor() -> None:
    examples = _examples()
    try:
        run_recursive_posterior_preference_oracle(
            _IncrementLeWM(),
            examples.deployment,
            examples.metadata,
            examples.labels.canonical_current,
            methods=(SEQUENCE_ORACLE_METHOD,),
        )
    except ValueError as error:
        assert "exactly one direct anchor" in str(error)
    else:
        raise AssertionError("sequence oracle accepted multiple selectable anchors")


def test_default_methods_exclude_hindsight_sequence_search() -> None:
    examples = _examples()
    result = run_recursive_posterior_preference_oracle(
        _IncrementLeWM(),
        examples.deployment,
        examples.metadata,
        examples.labels.canonical_current,
    )
    assert SEQUENCE_ORACLE_METHOD not in result.methods
    assert SEQUENCE_ORACLE_METHOD not in _normalize_methods(None)


def test_recursive_rejects_nonfinite_canonical_and_measurement() -> None:
    examples = _examples()
    canonical = examples.labels.canonical_current.clone()
    canonical[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="canonical_current must be finite"):
        run_recursive_posterior_preference_oracle(
            _IncrementLeWM(),
            examples.deployment,
            examples.metadata,
            canonical,
        )

    measurement = examples.deployment.measurement_current.clone()
    measurement[0, 0, 0] = float("nan")
    deployment = replace(examples.deployment, measurement_current=measurement)
    with pytest.raises(ValueError, match="measurement_current must be finite"):
        run_recursive_posterior_preference_oracle(
            _IncrementLeWM(),
            deployment,
            examples.metadata,
            examples.labels.canonical_current,
        )


def test_recursive_rejects_nonfinite_lewm_prior() -> None:
    examples = _examples()
    with pytest.raises(ValueError, match="lewm prior must be finite"):
        run_recursive_posterior_preference_oracle(
            _NonFiniteLeWM(),
            examples.deployment,
            examples.metadata,
            examples.labels.canonical_current,
        )
