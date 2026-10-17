from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fcpg.lewm_counterfactual_v3 import PROTOCOL, canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_lewm_counterfactual_v3_state_oracle_gate.py"
CONFIG = ROOT / "config/lewm/eval/counterfactual_v3_state_oracle_seed9216.json"


def _module():
    spec = importlib.util.spec_from_file_location("v3_gate_runner_test_module", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    file_records = {
        "action_only_policy": {
            "path": "/verified/policy.pt",
            "sha256": "1" * 64,
            "size_bytes": 11,
        },
        "fcpg_corrector": {
            "path": "/verified/corrector.pt",
            "sha256": "2" * 64,
            "size_bytes": 22,
            "loaded_path": "/verified/corrector.pt",
            "loaded_file_sha256": "2" * 64,
            "loaded_state_sha256": "3" * 64,
        },
    }
    state_sha = {
        "action_only_policy": "4" * 64,
        "fcpg_corrector": "3" * 64,
    }
    names = {f"candidate_{index}": f"{100 + index:064x}" for index in range(8)}
    receipts = []
    for task in config["evaluation"]["tasks"]:
        receipts.append(
            {
                "task": task,
                "episode_id": 0,
                "snapshot_step": 10,
                "validated": True,
                "repeat_candidate": "candidate_0",
                "original_outcome_sha256": dict(names),
                "repeated_outcome_sha256": names["candidate_0"],
                "reversed_outcome_sha256": dict(names),
                "repeat_comparison": {},
                "reversed_comparison": {},
            }
        )
    scientific = {
        "config": config,
        "config_sha256": canonical_sha256(config),
        "input_files": file_records,
        "loaded_state_sha256": state_sha,
        "lewm_loaded": False,
        "known_v2_non_overlap": {"status": "passed"},
        "upstream_checkpoint_episode_non_overlap": {
            "formal_claim_allowed": False
        },
        "candidate_replay_audit": {
            "scope": "episode_0_early_snapshot_per_task",
            "tasks": 4,
            "extra_branch_executions": 36,
            "receipts": receipts,
        },
        "git": {"dirty": False, "head": "a" * 40},
    }
    dataset = SimpleNamespace(
        provenance={
            "scientific_provenance": scientific,
            "scientific_protocol": PROTOCOL,
            "scientific_config_sha256": canonical_sha256(config),
        },
        manifest={"result_sha256": "5" * 64},
    )
    frozen = SimpleNamespace(file_records=file_records, state_sha256=state_sha)
    return config, dataset, frozen


def test_dataset_contract_binds_loaded_models_and_replay_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    config, dataset, frozen = _fixture()
    monkeypatch.setattr(module, "load_frozen_policy_inputs", lambda *_args, **_kwargs: frozen)
    assert module._dataset_contract(dataset, config) == "5" * 64

    dataset.provenance["scientific_provenance"]["input_files"] = {
        **frozen.file_records,
        "action_only_policy": {
            **frozen.file_records["action_only_policy"],
            "path": "/wrong/policy.pt",
        },
    }
    with pytest.raises(ValueError, match="input path/file"):
        module._dataset_contract(dataset, config)


def test_dataset_contract_rejects_forged_state_and_missing_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    config, dataset, frozen = _fixture()
    monkeypatch.setattr(module, "load_frozen_policy_inputs", lambda *_args, **_kwargs: frozen)
    dataset.provenance["scientific_provenance"]["loaded_state_sha256"] = {
        **frozen.state_sha256,
        "action_only_policy": "0" * 64,
    }
    with pytest.raises(ValueError, match="model-state"):
        module._dataset_contract(dataset, config)

    config, dataset, frozen = _fixture()
    monkeypatch.setattr(module, "load_frozen_policy_inputs", lambda *_args, **_kwargs: frozen)
    del dataset.provenance["scientific_provenance"]["candidate_replay_audit"]
    with pytest.raises(ValueError, match="replay audit"):
        module._dataset_contract(dataset, config)
