from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from fcpg.lewm_action_branch_a4 import FORMAL_TASK_INSTANCES
from fcpg.lewm_action_branch_a4b import (
    BENCHMARK_SEED,
    FINAL_CANDIDATES_PER_TASK,
    POOL_FINAL,
    POOL_PREFLIGHT,
    PREFLIGHT_CANDIDATES_PER_TASK,
    SOURCE_A4_PROTOCOL,
    SPLIT_MANIFEST_ID,
    TARGET_ELIGIBLE_CANDIDATES_PER_TASK,
    TASK_IDENTITY_NAMESPACE,
    TASKS,
    candidate_environment_seed,
    candidate_episode_uid,
    candidate_order,
    sample_a4b_proposal_schedule,
    validate_a4b_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def _json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _contracts() -> tuple[dict, dict, dict]:
    return (
        _json("config/lewm/eval/action_branch_a4b_fresh_confirmation.json"),
        _json("config/lewm/eval/action_branch_active_control_a4.json"),
        _json(
            "config/lewm/splits/"
            "action_branch_a4b_fresh_confirmation_seed31401.json"
        ),
    )


def test_a4b_contract_freezes_independent_finite_candidate_pools() -> None:
    contract, source, manifest = _contracts()

    assert validate_a4b_contract(
        contract, source_contract=source, split_manifest=manifest
    ) == TASKS
    assert contract["protocol"] != SOURCE_A4_PROTOCOL
    assert contract["split"]["manifest_id"] == SPLIT_MANIFEST_ID
    assert contract["collection"]["benchmark_seed"] == BENCHMARK_SEED
    assert contract["collection"]["target_eligible_candidates_per_task"] == (
        TARGET_ELIGIBLE_CANDIDATES_PER_TASK
    )
    for task in TASKS:
        preflight = candidate_order(manifest, pool=POOL_PREFLIGHT, task=task)
        final = candidate_order(manifest, pool=POOL_FINAL, task=task)
        assert len(preflight) == PREFLIGHT_CANDIDATES_PER_TASK
        assert len(final) == FINAL_CANDIDATES_PER_TASK
        assert set(preflight).isdisjoint(final)
        assert set(preflight + final) == set(FORMAL_TASK_INSTANCES)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("eligibility_source", "minimum_recent_action_rms", 0.049),
        ("eligibility_source", "minimum_workspace_and_joint_margin", 0.049),
        ("confirmation_gate", "minimum_relative_improvement", 0.09),
        ("confirmation_gate", "minimum_positive_tasks", 5),
        ("confirmation_gate", "bootstrap_samples", 9_999),
    ],
)
def test_a4b_rejects_lowered_v2_thresholds(
    section: str, field: str, value: object
) -> None:
    contract, source, manifest = _contracts()
    drifted = deepcopy(contract)
    drifted[section][field] = value

    with pytest.raises(ValueError, match="drifted"):
        validate_a4b_contract(
            drifted, source_contract=source, split_manifest=manifest
        )


def test_a4b_episode_uid_namespaces_raw_index_and_fresh_seed() -> None:
    _contract, _source, manifest = _contracts()
    task = TASKS[0]
    raw_index = candidate_order(manifest, pool=POOL_FINAL, task=task)[0]
    environment_seed = candidate_environment_seed(
        pool=POOL_FINAL, task_index=0, rank=0
    )
    uid = candidate_episode_uid(
        pool=POOL_FINAL,
        task=task,
        rank=0,
        raw_task_index=raw_index,
        environment_seed=environment_seed,
    )

    assert uid.startswith(f"{TASK_IDENTITY_NAMESPACE}:benchmark-{BENCHMARK_SEED}:")
    assert f":raw-{raw_index:02d}:" in uid
    assert uid != f"{task}:task-instance-{raw_index:02d}"
    assert environment_seed >= 720_000


def test_a4b_schedule_is_prereset_deterministic_and_manifest_bound() -> None:
    contract, source, manifest = _contracts()
    task = TASKS[2]
    raw_index = candidate_order(manifest, pool=POOL_PREFLIGHT, task=task)[0]
    kwargs = {
        "pool": POOL_PREFLIGHT,
        "candidate_rank": 0,
        "contract": contract,
        "source_contract": source,
        "split_manifest": manifest,
    }
    first = sample_a4b_proposal_schedule(task, raw_index, **kwargs)

    assert first == sample_a4b_proposal_schedule(task, raw_index, **kwargs)
    assert len(first) == 12
    assert first == tuple(sorted(first))
    assert first[0] >= 3 and first[-1] <= 160
    assert all(right - left >= 5 for left, right in zip(first, first[1:]))

    drifted = deepcopy(manifest)
    drifted["candidate_order_by_task"][task][POOL_PREFLIGHT][0:2] = reversed(
        drifted["candidate_order_by_task"][task][POOL_PREFLIGHT][0:2]
    )
    with pytest.raises(ValueError, match="candidate order"):
        sample_a4b_proposal_schedule(
            task, raw_index, **{**kwargs, "split_manifest": drifted}
        )
