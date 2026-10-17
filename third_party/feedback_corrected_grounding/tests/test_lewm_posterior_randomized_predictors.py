from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
import torch

from fcpg.lewm_posterior_randomized_predictors import (
    FACTUAL_LEWM_METHOD,
    HOLD_LAST_METHOD,
    LATENT_CONSTANT_VELOCITY_METHOD,
    MATCHED_NO_ACTION_METHOD,
    SHUFFLED_ACTION_LEWM_METHOD,
    ZERO_ACTION_LEWM_METHOD,
    run_randomized_predictor_isolation,
    validate_shuffled_action_row_mapping,
)
from fcpg.lewm_posterior_randomized_sequence import (
    PosteriorRandomizedSequenceExamples,
    build_fresh_posterior_randomized_sequence,
)


class ActionSensitiveLeWM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("dtype_reference", torch.tensor(1.0))
        self.call_training: list[bool] = []
        self.call_grad_enabled: list[bool] = []

    def predict(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        self.call_training.append(self.training)
        self.call_grad_enabled.append(torch.is_grad_enabled())
        delta = action.mean(dim=-1, keepdim=True).unsqueeze(-2)
        return history + delta


class NonFiniteLeWM(ActionSensitiveLeWM):
    def predict(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.full_like(history, float("nan"))


class BadShapeLeWM(ActionSensitiveLeWM):
    def predict(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return history[:, -1]


def _examples(
    *,
    tasks: Sequence[str] = ("task",),
    episodes_per_task: int = 2,
    future_delta: float = 0.0,
) -> PosteriorRandomizedSequenceExamples:
    frames_per_episode = 60
    anchors = 2
    latent_dim = 3
    action_dim = 2
    task_rows: list[str] = []
    episode_rows: list[str] = []
    frame_rows: list[int] = []
    actions: list[np.ndarray] = []
    measurements: list[np.ndarray] = []
    canonicals: list[np.ndarray] = []
    dropout_rows: list[np.ndarray] = []
    recovery_rows: list[np.ndarray] = []

    for task_index, task in enumerate(tasks):
        for episode_index in range(episodes_per_task):
            episode = f"{task}-episode-{episode_index}"
            episode_bias = 20.0 * task_index + 5.0 * episode_index
            for frame in range(frames_per_episode):
                task_rows.append(task)
                episode_rows.append(episode)
                frame_rows.append(frame)
                action = np.asarray(
                    [1.0 + episode_index + frame / 100.0, -0.25 - task_index],
                    dtype=np.float32,
                )
                canonical = np.asarray(
                    [
                        [episode_bias + frame / 10.0 + anchor + dim / 100.0 for dim in range(latent_dim)]
                        for anchor in range(anchors)
                    ],
                    dtype=np.float32,
                )
                measurement = canonical.copy()
                if 20 <= frame < 30:
                    measurement[1] += 50.0
                if future_delta and frame >= 35:
                    action += future_delta
                    measurement += future_delta
                dropout = np.zeros(anchors, dtype=np.bool_)
                recovery = np.zeros(anchors, dtype=np.bool_)
                if 20 <= frame < 30:
                    dropout[1] = True
                if 30 <= frame < 40:
                    recovery[1] = True
                actions.append(action)
                measurements.append(measurement)
                canonicals.append(canonical)
                dropout_rows.append(dropout)
                recovery_rows.append(recovery)

    count = len(task_rows)
    return build_fresh_posterior_randomized_sequence(
        task=np.asarray(task_rows),
        episode=np.asarray(episode_rows),
        frame=np.asarray(frame_rows, dtype=np.int64),
        action=np.stack(actions),
        measurement_embedding=np.stack(measurements),
        canonical_label=np.stack(canonicals),
        q_nominal=np.zeros((count, anchors, 2), dtype=np.float32),
        q_corrected=np.zeros((count, anchors, 2), dtype=np.float32),
        valid_nominal=np.ones((count, anchors), dtype=np.bool_),
        anchor_dropout_active=np.stack(dropout_rows),
        anchor_recovery_segment=np.stack(recovery_rows),
        observation_action_alignment="pre_action",
    )


def _cross_episode_mapping(
    examples: PosteriorRandomizedSequenceExamples,
) -> tuple[int, ...]:
    rows = examples.metadata
    grouped: dict[tuple[str, int, str], list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(
            (row.task, int(row.phase_index), row.event), []
        ).append(index)
    result = [-1] * len(rows)
    for indices in grouped.values():
        assert len(indices) >= 2
        donors = indices[1:] + indices[:1]
        for target, donor in zip(indices, donors, strict=True):
            assert rows[target].episode_id != rows[donor].episode_id
            result[target] = donor
    assert all(index >= 0 for index in result)
    return tuple(result)


def _row_index(
    examples: PosteriorRandomizedSequenceExamples,
    *,
    episode_suffix: str,
    phase: int,
    event: str,
) -> int:
    matches = [
        index
        for index, row in enumerate(examples.metadata)
        if str(row.episode_id).endswith(episode_suffix)
        and int(row.phase_index) == phase
        and row.event == event
    ]
    assert len(matches) == 1
    return matches[0]


def test_predictors_route_only_interrupted_anchor_and_reanchor_on_recovery() -> None:
    examples = _examples()
    model = ActionSensitiveLeWM()
    result = run_randomized_predictor_isolation(model, examples)

    assert tuple(result.trajectories) == (
        FACTUAL_LEWM_METHOD,
        HOLD_LAST_METHOD,
        LATENT_CONSTANT_VELOCITY_METHOD,
        ZERO_ACTION_LEWM_METHOD,
    )
    active = _row_index(
        examples, episode_suffix="episode-0", phase=0, event="o0"
    )
    recovery = _row_index(
        examples, episode_suffix="episode-0", phase=0, event="r0"
    )
    measurement = torch.as_tensor(examples.deployment.measurement_current)

    for trajectory in result.trajectories.values():
        assert trajectory.pre_prior.shape == measurement.shape
        assert trajectory.post_update.shape == measurement.shape
        assert trajectory.writeback.shape == measurement.shape
        assert trajectory.beta.shape == (*measurement.shape[:2], 1)
        assert torch.isfinite(trajectory.pre_prior).all()
        assert torch.isfinite(trajectory.post_update).all()
        torch.testing.assert_close(
            trajectory.post_update[active, 0], measurement[active, 0]
        )
        torch.testing.assert_close(
            trajectory.post_update[active, 1], trajectory.pre_prior[active, 1]
        )
        torch.testing.assert_close(
            trajectory.post_update[recovery], measurement[recovery]
        )
        assert trajectory.beta[active, 0, 0] == 0
        assert trajectory.beta[active, 1, 0] == 1
        assert torch.count_nonzero(trajectory.beta[recovery]) == 0

    history = torch.as_tensor(examples.deployment.measurement_history[active])
    torch.testing.assert_close(
        result.trajectories[HOLD_LAST_METHOD].pre_prior[active], history[-1]
    )
    torch.testing.assert_close(
        result.trajectories[LATENT_CONSTANT_VELOCITY_METHOD].pre_prior[active],
        2.0 * history[-1] - history[-2],
    )
    assert not torch.equal(
        result.trajectories[FACTUAL_LEWM_METHOD].pre_prior[active],
        result.trajectories[ZERO_ACTION_LEWM_METHOD].pre_prior[active],
    )


def test_methods_have_independent_recursive_histories() -> None:
    examples = _examples()
    methods = (
        FACTUAL_LEWM_METHOD,
        HOLD_LAST_METHOD,
        LATENT_CONSTANT_VELOCITY_METHOD,
        ZERO_ACTION_LEWM_METHOD,
    )
    together = run_randomized_predictor_isolation(
        ActionSensitiveLeWM(), examples, methods=methods
    )
    for method in methods:
        alone = run_randomized_predictor_isolation(
            ActionSensitiveLeWM(), examples, methods=(method,)
        )
        joint = together.trajectories[method]
        single = alone.trajectories[method]
        torch.testing.assert_close(joint.pre_prior, single.pre_prior)
        torch.testing.assert_close(joint.post_update, single.post_update)
        torch.testing.assert_close(joint.writeback, single.writeback)


def test_matched_no_action_model_uses_its_own_predictor_and_history() -> None:
    examples = _examples()
    factual = ActionSensitiveLeWM()
    no_action = ActionSensitiveLeWM()
    result = run_randomized_predictor_isolation(
        factual,
        examples,
        methods=(FACTUAL_LEWM_METHOD, MATCHED_NO_ACTION_METHOD),
        matched_no_action_lewm=no_action,
    )
    assert tuple(result.trajectories) == (
        FACTUAL_LEWM_METHOD,
        MATCHED_NO_ACTION_METHOD,
    )
    assert factual.call_training and no_action.call_training
    assert not any(factual.call_training) and not any(no_action.call_training)


def test_matched_no_action_model_and_method_must_be_selected_together() -> None:
    examples = _examples()
    with pytest.raises(ValueError, match="requires matched_no_action_lewm"):
        run_randomized_predictor_isolation(
            ActionSensitiveLeWM(),
            examples,
            methods=(MATCHED_NO_ACTION_METHOD,),
        )
    with pytest.raises(ValueError, match="was provided"):
        run_randomized_predictor_isolation(
            ActionSensitiveLeWM(),
            examples,
            methods=(FACTUAL_LEWM_METHOD,),
            matched_no_action_lewm=ActionSensitiveLeWM(),
        )


def test_future_inputs_cannot_change_earlier_predictions() -> None:
    original = _examples(future_delta=0.0)
    changed = _examples(future_delta=100.0)
    first = run_randomized_predictor_isolation(ActionSensitiveLeWM(), original)
    second = run_randomized_predictor_isolation(ActionSensitiveLeWM(), changed)
    early = torch.tensor(
        [row.target_frame < 35 for row in original.metadata], dtype=torch.bool
    )
    assert tuple(original.metadata) == tuple(changed.metadata)
    for method in first.trajectories:
        torch.testing.assert_close(
            first.trajectories[method].pre_prior[early],
            second.trajectories[method].pre_prior[early],
        )
        torch.testing.assert_close(
            first.trajectories[method].post_update[early],
            second.trajectories[method].post_update[early],
        )


def test_frozen_lewm_runs_in_eval_inference_mode_and_restores_state() -> None:
    model = ActionSensitiveLeWM().train()
    run_randomized_predictor_isolation(
        model, _examples(), methods=(FACTUAL_LEWM_METHOD,)
    )
    assert model.training
    assert model.call_training
    assert not any(model.call_training)
    assert not any(model.call_grad_enabled)


def test_shuffled_action_mapping_is_applied_to_actions_only() -> None:
    examples = _examples()
    mapping = _cross_episode_mapping(examples)
    model = ActionSensitiveLeWM()
    result = run_randomized_predictor_isolation(
        model,
        examples,
        methods=(FACTUAL_LEWM_METHOD, SHUFFLED_ACTION_LEWM_METHOD),
        shuffled_action_row_mapping=mapping,
    )
    shuffled = result.trajectories[SHUFFLED_ACTION_LEWM_METHOD]
    assert shuffled.action_source_index.tolist() == list(mapping)
    assert result.trajectories[FACTUAL_LEWM_METHOD].action_source_index.tolist() == list(
        range(len(examples.metadata))
    )
    active = _row_index(
        examples, episode_suffix="episode-0", phase=0, event="o0"
    )
    assert not torch.equal(
        result.trajectories[FACTUAL_LEWM_METHOD].pre_prior[active],
        shuffled.pre_prior[active],
    )
    measurement = torch.as_tensor(examples.deployment.measurement_current)
    torch.testing.assert_close(shuffled.post_update[active, 0], measurement[active, 0])


def test_shuffled_action_mapping_fails_closed_on_contract_violations() -> None:
    examples = _examples(tasks=("task-a", "task-b"))
    mapping = list(_cross_episode_mapping(examples))
    assert validate_shuffled_action_row_mapping(mapping, examples.metadata) == tuple(
        mapping
    )

    same_episode = list(range(len(mapping)))
    with pytest.raises(ValueError, match="cross physical episodes"):
        validate_shuffled_action_row_mapping(same_episode, examples.metadata)

    wrong_event = mapping.copy()
    wrong_event[0], wrong_event[1] = wrong_event[1], wrong_event[0]
    with pytest.raises(ValueError, match="preserve phase and event"):
        validate_shuffled_action_row_mapping(wrong_event, examples.metadata)

    rows = examples.metadata
    source_a = next(
        index
        for index, row in enumerate(rows)
        if row.task == "task-a" and row.event == "o0" and row.phase_index == 0
    )
    source_b = next(
        index
        for index, row in enumerate(rows)
        if row.task == "task-b" and row.event == "o0" and row.phase_index == 0
    )
    wrong_task = mapping.copy()
    target_a = mapping.index(source_a)
    target_b = mapping.index(source_b)
    wrong_task[target_a], wrong_task[target_b] = (
        wrong_task[target_b],
        wrong_task[target_a],
    )
    with pytest.raises(ValueError, match="within task"):
        validate_shuffled_action_row_mapping(wrong_task, examples.metadata)

    duplicate = mapping.copy()
    duplicate[0] = duplicate[1]
    with pytest.raises(ValueError, match="permutation"):
        validate_shuffled_action_row_mapping(duplicate, examples.metadata)


def test_shuffle_method_and_mapping_must_be_selected_together() -> None:
    examples = _examples()
    mapping = _cross_episode_mapping(examples)
    with pytest.raises(ValueError, match="requires shuffled_action_row_mapping"):
        run_randomized_predictor_isolation(
            ActionSensitiveLeWM(),
            examples,
            methods=(SHUFFLED_ACTION_LEWM_METHOD,),
        )
    with pytest.raises(ValueError, match="was provided"):
        run_randomized_predictor_isolation(
            ActionSensitiveLeWM(),
            examples,
            methods=(FACTUAL_LEWM_METHOD,),
            shuffled_action_row_mapping=mapping,
        )


@pytest.mark.parametrize("model", [NonFiniteLeWM(), BadShapeLeWM()])
def test_invalid_lewm_predictions_fail_closed(model: torch.nn.Module) -> None:
    expected = "finite" if isinstance(model, NonFiniteLeWM) else "one prediction"
    with pytest.raises((ValueError, TypeError), match=expected):
        run_randomized_predictor_isolation(
            model,
            _examples(),
            methods=(FACTUAL_LEWM_METHOD,),
        )
