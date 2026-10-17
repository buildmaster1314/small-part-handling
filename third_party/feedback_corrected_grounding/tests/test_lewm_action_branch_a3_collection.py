from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from fcpg.lewm_action_branch_a3_collection import (
    TASKS,
    assign_a3_splits,
    validate_a3_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/lewm/eval/action_branch_physical_support_a3.json"


def _contract() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_a3_contract_and_split_are_fresh_task_local_and_leave_test_reserved() -> None:
    contract = _contract()
    assert validate_a3_contract(contract) == TASKS
    first = assign_a3_splits(TASKS, contract)
    second = assign_a3_splits(TASKS, contract)
    assert first == second
    for task in TASKS:
        labels = [first[(task, instance)] for instance in range(48)]
        assert labels.count("train") == 24
        assert labels.count("development") == 12
        assert labels.count("test") == 12


@pytest.mark.parametrize(
    ("section", "key", "value", "match"),
    (
        ("split", "protocol_seed", 29104, "split"),
        ("collection", "horizon_steps", 7, "H=8"),
        ("hazard", "eligible_last_step", 30, "hazard"),
        ("support_gate", "minimum_tcp_max_pairwise_m", 0.0, "support"),
    ),
)
def test_a3_contract_rejects_protocol_drift(
    section: str, key: str, value: object, match: str
) -> None:
    contract = deepcopy(_contract())
    contract[section][key] = value
    with pytest.raises(ValueError, match=match):
        validate_a3_contract(contract)


def test_a3_contract_rejects_semantic_branch_reordering() -> None:
    contract = deepcopy(_contract())
    branches = contract["matched_state_branches"]["semantic_bank_before_permutation"]
    branches[0], branches[1] = branches[1], branches[0]
    with pytest.raises(ValueError, match="action-bank"):
        validate_a3_contract(contract)
