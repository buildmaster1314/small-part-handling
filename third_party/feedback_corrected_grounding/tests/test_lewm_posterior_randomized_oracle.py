from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from fcpg.lewm_posterior_gate_runtime import MEASUREMENT_METHOD
from fcpg.lewm_posterior_gate_data import PosteriorGateEvaluationMasks
from fcpg.lewm_posterior_randomized_oracle import (
    CONVEX_ORACLE_METHOD,
    FSM_ORACLE_METHOD,
    HARD_ORACLE_METHOD,
    PRIOR_METHOD,
    SCHEDULE_SHORTCUT_METHOD,
    TIME_ONLY_METHOD,
    build_randomized_oracle_records,
    fit_time_only_from_teacher_forced,
    normalized_episode_time,
    run_randomized_recursive_oracle,
)
from fcpg.lewm_posterior_randomized_sequence import (
    build_fresh_posterior_randomized_sequence,
)


class ZeroPrior(torch.nn.Module):
    def predict(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        assert history.shape[:2] == action.shape[:2]
        return torch.zeros_like(history)


def _examples(*, clean: bool = False):
    frames = 60
    anchors = 2
    latent = 3
    canonical = np.zeros((frames, anchors, latent), dtype=np.float32)
    measurement = canonical.copy()
    if not clean:
        measurement[20:30, 1] = 4.0
    dropout = np.zeros((frames, anchors), dtype=np.bool_)
    recovery = np.zeros_like(dropout)
    dropout[20:30, 1] = True
    recovery[30:40, 1] = True
    return build_fresh_posterior_randomized_sequence(
        task=np.asarray(["task"] * frames),
        episode=np.asarray(["episode"] * frames),
        frame=np.arange(frames, dtype=np.int64),
        action=np.zeros((frames, 1), dtype=np.float32),
        measurement_embedding=measurement,
        canonical_label=canonical,
        q_nominal=np.zeros((frames, anchors, 2), dtype=np.float32),
        q_corrected=np.zeros((frames, anchors, 2), dtype=np.float32),
        valid_nominal=np.ones((frames, anchors), dtype=np.bool_),
        anchor_dropout_active=dropout,
        anchor_recovery_segment=recovery,
        observation_action_alignment="pre_action",
    )


def test_randomized_recursive_oracle_and_time_only_contract() -> None:
    model = ZeroPrior()
    calibration = _examples()
    times = normalized_episode_time(
        calibration.metadata, {("task", "episode"): (0, 59)}
    )
    policy = fit_time_only_from_teacher_forced(
        model,
        calibration,
        times,
        resampling_units={("task", "episode"): "source-episode"},
        n_time_bins=10,
        min_count=1,
    )
    result = run_randomized_recursive_oracle(
        model,
        calibration,
        time_only_policy=policy,
        normalized_time=times,
    )

    assert tuple(result.trajectories) == (
        MEASUREMENT_METHOD,
        PRIOR_METHOD,
        HARD_ORACLE_METHOD,
        CONVEX_ORACLE_METHOD,
        SCHEDULE_SHORTCUT_METHOD,
        FSM_ORACLE_METHOD,
        TIME_ONLY_METHOD,
    )
    rows = result.metadata
    o0 = [index for index, row in enumerate(rows) if row.event == "o0"]
    o1 = [index for index, row in enumerate(rows) if row.event == "o1"]
    r0 = [index for index, row in enumerate(rows) if row.event == "r0"]
    assert len(o0) == len(o1) == len(r0) == 5
    assert torch.all(result.trajectories[FSM_ORACLE_METHOD].beta[o0, 1] == 0)
    assert torch.all(result.trajectories[FSM_ORACLE_METHOD].beta[o1, 1] == 1)
    assert torch.all(result.trajectories[FSM_ORACLE_METHOD].beta[r0, 1] == 0)
    assert torch.all(
        result.trajectories[SCHEDULE_SHORTCUT_METHOD].beta[o0 + o1, 1] == 1
    )

    direct = {("task", "episode"): 1}
    records = build_randomized_oracle_records(
        result, calibration.labels.canonical_current, direct
    )
    assert len(records) == 7 * 5 * 4
    assert {row["window"] for row in records} == {"active", "recovery"}
    assert {row["event_index"] for row in records} == {0, 1}


def test_clean_records_are_explicitly_labeled_and_oracle_is_identity() -> None:
    model = ZeroPrior()
    examples = _examples(clean=True)
    result = run_randomized_recursive_oracle(
        model,
        examples,
        methods=(MEASUREMENT_METHOD, HARD_ORACLE_METHOD, FSM_ORACLE_METHOD),
    )
    assert torch.count_nonzero(result.trajectories[HARD_ORACLE_METHOD].beta) == 0
    assert torch.count_nonzero(result.trajectories[FSM_ORACLE_METHOD].beta) == 0
    records = build_randomized_oracle_records(
        result,
        examples.labels.canonical_current,
        {("task", "episode"): 1},
        clean=True,
    )
    assert {row["window"] for row in records} == {"clean"}
    assert all(row["mse"] == 0.0 for row in records)
    assert {row["event_index"] for row in records} == {0, 1, 2, 3}


def test_schedule_shortcut_reads_evaluation_mask_not_event_name() -> None:
    examples = _examples()
    dropout = np.asarray(examples.evaluation.dropout_active).copy()
    o1 = [index for index, row in enumerate(examples.metadata) if row.event == "o1"]
    dropout[o1, 1] = False
    modified = replace(
        examples,
        evaluation=PosteriorGateEvaluationMasks(
            dropout_active=dropout,
            recovery_active=examples.evaluation.recovery_active,
        ),
    )
    result = run_randomized_recursive_oracle(
        ZeroPrior(),
        modified,
        methods=(SCHEDULE_SHORTCUT_METHOD,),
    )
    assert torch.all(
        result.trajectories[SCHEDULE_SHORTCUT_METHOD].beta[o1, 1] == 0
    )


def test_time_only_inputs_and_episode_bounds_fail_closed() -> None:
    model = ZeroPrior()
    examples = _examples()
    with pytest.raises(KeyError, match="missing raw frame bounds"):
        normalized_episode_time(examples.metadata, {})
    with pytest.raises(ValueError, match="cover exactly"):
        normalized_episode_time(
            examples.metadata,
            {("task", "episode"): (0, 59), ("extra", "episode"): (0, 59)},
        )
    with pytest.raises(ValueError, match="fitted policy"):
        run_randomized_recursive_oracle(
            model,
            examples,
            methods=(TIME_ONLY_METHOD,),
        )
    times = normalized_episode_time(
        examples.metadata, {("task", "episode"): (0, 59)}
    )
    with pytest.raises(ValueError, match="cover exactly"):
        fit_time_only_from_teacher_forced(
            model,
            examples,
            times,
            resampling_units={},
        )


def test_methods_use_independent_recursive_histories() -> None:
    model = ZeroPrior()
    examples = _examples()
    times = normalized_episode_time(
        examples.metadata, {("task", "episode"): (0, 59)}
    )
    time_only_policy = fit_time_only_from_teacher_forced(
        model,
        examples,
        times,
        n_time_bins=1,
        min_count=1,
    )
    methods = (
        MEASUREMENT_METHOD,
        PRIOR_METHOD,
        HARD_ORACLE_METHOD,
        CONVEX_ORACLE_METHOD,
        FSM_ORACLE_METHOD,
        SCHEDULE_SHORTCUT_METHOD,
        TIME_ONLY_METHOD,
    )
    together = run_randomized_recursive_oracle(
        model,
        examples,
        methods=methods,
        time_only_policy=time_only_policy,
        normalized_time=times,
    )
    for method in methods:
        alone = run_randomized_recursive_oracle(
            model,
            examples,
            methods=(method,),
            time_only_policy=time_only_policy if method == TIME_ONLY_METHOD else None,
            normalized_time=times if method == TIME_ONLY_METHOD else None,
        )
        assert torch.equal(
            together.trajectories[method].post_update,
            alone.trajectories[method].post_update,
        )
        assert torch.equal(
            together.trajectories[method].pre_prior,
            alone.trajectories[method].pre_prior,
        )
