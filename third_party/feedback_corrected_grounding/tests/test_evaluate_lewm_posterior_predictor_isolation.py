from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.lewm_posterior_gate_runtime import (
    PosteriorGateMethodTrajectory,
    PosteriorGateRecursiveResult,
)
from fcpg.lewm_posterior_randomized_predictors import FACTUAL_LEWM_METHOD
from fcpg.lewm_posterior_randomized_sequence import (
    build_fresh_posterior_randomized_sequence,
)
from scripts import evaluate_lewm_posterior_predictor_isolation as cli


def _examples():
    frames = 60
    tasks = []
    episodes = []
    frame_ids = []
    measurement = []
    canonical = []
    dropout = []
    recovery = []
    actions = []
    for episode in ("a-step", "a-linear", "b-step", "b-linear"):
        for frame in range(frames):
            tasks.append("task")
            episodes.append(episode)
            frame_ids.append(frame)
            value = np.full((2, 3), float(frame), dtype=np.float32)
            canonical.append(value)
            observed = value.copy()
            if 20 <= frame < 40:
                observed[1] += 10.0
            measurement.append(observed)
            active = np.zeros(2, dtype=np.bool_)
            recovered = np.zeros(2, dtype=np.bool_)
            active[1] = 20 <= frame < 40
            recovered[1] = 40 <= frame < 50
            dropout.append(active)
            recovery.append(recovered)
            actions.append(np.asarray([frame, -frame], dtype=np.float32))
    count = len(tasks)
    return build_fresh_posterior_randomized_sequence(
        task=np.asarray(tasks),
        episode=np.asarray(episodes),
        frame=np.asarray(frame_ids),
        action=np.asarray(actions),
        measurement_embedding=np.asarray(measurement),
        canonical_label=np.asarray(canonical),
        q_nominal=np.zeros((count, 2, 2), dtype=np.float32),
        q_corrected=np.zeros((count, 2, 2), dtype=np.float32),
        valid_nominal=np.ones((count, 2), dtype=np.bool_),
        anchor_dropout_active=np.asarray(dropout),
        anchor_recovery_segment=np.asarray(recovery),
        observation_action_alignment="pre_action",
    )


def _units():
    return {
        ("task", "a-step"): "source-a",
        ("task", "a-linear"): "source-a",
        ("task", "b-step"): "source-b",
        ("task", "b-linear"): "source-b",
    }


def test_cross_source_shuffle_is_permutation_and_never_uses_paired_view() -> None:
    examples = _examples()
    mapping = cli.build_cross_source_action_shuffle(examples.metadata, _units())
    assert sorted(mapping) == list(range(len(mapping)))
    for target, donor in enumerate(mapping):
        target_row = examples.metadata[target]
        donor_row = examples.metadata[donor]
        assert target_row.task == donor_row.task
        assert target_row.phase_index == donor_row.phase_index
        assert target_row.event == donor_row.event
        assert _units()[cli._episode_key(target_row)] != _units()[
            cli._episode_key(donor_row)
        ]


def test_cross_source_shuffle_rejects_uncovered_and_impossible_groups() -> None:
    examples = _examples()
    missing = dict(_units())
    missing.pop(("task", "b-linear"))
    with pytest.raises(ValueError, match="cover exactly"):
        cli.build_cross_source_action_shuffle(examples.metadata, missing)

    one_source = {key: "same" for key in _units()}
    with pytest.raises(ValueError, match="two source trajectories"):
        cli.build_cross_source_action_shuffle(examples.metadata, one_source)


def test_predictor_records_use_only_post_onset_active_anchor() -> None:
    examples = _examples()
    target = torch.as_tensor(examples.labels.canonical_current)
    prediction = target.clone()
    prediction[:, 1] += 2.0
    trajectory = PosteriorGateMethodTrajectory(
        pre_prior=prediction.clone(),
        post_update=prediction,
        writeback=prediction.clone(),
        beta=torch.ones((*prediction.shape[:2], 1)),
        action_source_index=torch.arange(prediction.shape[0]),
    )
    result = PosteriorGateRecursiveResult(
        trajectories={FACTUAL_LEWM_METHOD: trajectory},
        metadata=examples.metadata,
    )
    views = {
        ("task", episode): "step" if episode.endswith("step") else "linear_drift"
        for episode in ("a-step", "a-linear", "b-step", "b-linear")
    }
    records = cli.build_predictor_records(
        result,
        examples,
        direct_anchors={key: 1 for key in _units()},
        resampling_units=_units(),
        camera_views=views,
    )
    assert records
    assert {record["event_id"] for record in records}.isdisjoint({"o0", "r0"})
    assert {record["squared_error"] for record in records} == {4.0}
    assert {record["source_trajectory_id"] for record in records} == {
        "source-a",
        "source-b",
    }


def test_markdown_marks_scope_and_method_limitations() -> None:
    comparison = {
        "relative_improvement": 0.2,
        "relative_improvement_95_ci": [0.1, 0.3],
        "positive_task_count": 4,
        "task_count": 4,
    }
    payload = {
        "scope": {"matched_no_action_is_independently_trained": False},
        "statistics": {
            "decision": "Go",
            "comparisons": {
                name: dict(comparison)
                for name in (
                    "hold_last",
                    "constant_velocity",
                    "zero_action",
                    "shuffled_action",
                )
            },
            "gate": {"failed_checks": []},
        }
    }
    markdown = cli._markdown(payload)
    assert "privileged sensor-valid mask" in markdown
    assert "not a deployable selector" in markdown
    assert "not a separately trained no-action" in markdown


def test_markdown_describes_matched_no_action_training() -> None:
    comparison = {
        "relative_improvement": 0.2,
        "relative_improvement_95_ci": [0.1, 0.3],
        "positive_task_count": 4,
        "task_count": 4,
    }
    payload = {
        "scope": {"matched_no_action_is_independently_trained": True},
        "statistics": {
            "decision": "Go",
            "comparisons": {
                name: dict(comparison)
                for name in (
                    "hold_last",
                    "constant_velocity",
                    "zero_action",
                    "shuffled_action",
                )
            },
            "gate": {"failed_checks": []},
        },
    }
    markdown = cli._markdown(payload)
    assert "Matched no-action recurrent" in markdown
    assert "identical initialization" in markdown
