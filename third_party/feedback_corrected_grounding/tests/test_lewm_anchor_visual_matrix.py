from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.evaluate_lewm_grounded import (
    ANCHOR_ABLATION_PARAMETER_COUNT,
    ANCHOR_ABLATION_SEED,
    ANCHOR_ABLATION_SPLIT_SHA256,
    ANCHOR_ABLATION_TARGET_SHA256,
)
from scripts.run_lewm_anchor_visual_matrix import (
    DEFAULT_DEV_PROTOCOL,
    EXPECTED_FULL_VALIDATION_COUNTS,
    EXPECTED_TRAINING_CONTRACT,
    EXPECTED_UNSEEN_COUNTS,
    TRAINING_SOURCE_FILES,
    _acquire_result_lock,
    _cell_completion_path,
    _cell_contract,
    _cell_contract_path,
    _completion_record,
    _episode_membership_sha256,
    _load_metrics_tail,
    _manifest_artifact_hash_snapshot,
    _load_reusable_cell,
    _paired_report_sections,
    _seal_cell_output,
    _sha256,
    _validate_annotation_counts,
    _validate_dev_protocol,
    _validate_frozen_training_contract,
    _validate_hash_snapshot,
    _validate_corrector_slice,
    _validate_legacy_corrector_checkpoint_binding,
    _write_text_idempotent,
    build_evaluator_command,
    build_parser,
    normalize_cell_report,
    replay_legacy_corrector_split,
    validate_training_blob_signatures,
)
from scripts.train_bc_policy import make_split


def _contract(path: Path) -> dict[str, object]:
    return {
        "schema_version": 4,
        "protocol": "fcpg_lewm_anchor_visual_cell_contract_v4",
        "repository_head": "1" * 40,
        "repository_worktree_sha256": "2" * 64,
        "frozen_input_artifacts_sha256": "3" * 64,
        "checkpoint_ledger_path": str(path.parent / "ledger.json"),
        "checkpoint_ledger_sha256": "a" * 64,
        "development_only": True,
        "train_mode": "triplet_nominal",
        "eval_mode": "paired_visual",
        "checkpoint": str(path.parent / "lewm_grounded_latest.pt"),
        "checkpoint_sha256": "c" * 64,
        "checkpoint_step": 2000,
        "metrics_path": str(path.parent / "metrics.jsonl"),
        "metrics_sha256": "m" * 64,
        "input_manifest_path": str(path.parent / "input_manifest.json"),
        "input_manifest_sha256": "i" * 64,
        "dev_protocol_path": str(path.parent / "protocol.json"),
        "dev_protocol_sha256": "p" * 64,
        "corrector_slice_path": str(path.parent / "slice.json"),
        "corrector_slice_sha256": "s" * 64,
        "evaluator_sha256": "e" * 64,
        "execution_code_sha256": "x" * 64,
        "evaluator_parameters": {
            "split": "validation",
            "grounding_mode": "checkpoint",
            "rollout_steps": 5,
            "batch_size": 64,
            "num_workers": 0,
            "bootstrap_samples": 10_000,
            "bootstrap_seed": 5120,
            "device": "cpu",
            "max_batches": None,
        },
        "output": str(path),
        "target_sha256": ANCHOR_ABLATION_TARGET_SHA256,
        "data_split_sha256": ANCHOR_ABLATION_SPLIT_SHA256,
        "command": ["python", "evaluate.py"],
    }


def _report(path: Path) -> dict[str, object]:
    path.write_text("{}" + chr(10), encoding="utf-8")
    per_task = {
        "pick-place-v3": {
            "num_sources": 1,
            "num_episodes": 1,
            "num_sequences": 1,
        }
    }
    macro = {
        "num_tasks": 1,
        "num_episodes": 1,
        "num_sequences": 1,
        "endpoints": {},
    }
    return {
        "schema_version": 1,
        "protocol": "grounded_lewm_offline_diagnostics",
        "grounding_mode_request": "checkpoint",
        "checkpoint": str(path.parent / "lewm_grounded_latest.pt"),
        "checkpoint_sha256": "c" * 64,
        "checkpoint_step": 2000,
        "provenance": {
            "validated": True,
            "git_state_validated": False,
            "manifest_sha256": "i" * 64,
            "run_sidecar": str(path.parent / "input_manifest.json"),
        },
        "execution_code": {"aggregate_sha256": "x" * 64},
        "anchor_visual_ablation": {
            "active": True,
            "train_mode": "triplet_nominal",
            "eval_mode": "paired_visual",
            "step": 2000,
            "parameter_count": ANCHOR_ABLATION_PARAMETER_COUNT,
            "seed": ANCHOR_ABLATION_SEED,
            "target_sha256": ANCHOR_ABLATION_TARGET_SHA256,
            "data_split_sha256": ANCHOR_ABLATION_SPLIT_SHA256,
            "development_only": True,
            "formal_gate_eligible": False,
        },
        "evaluation": {
            "split": "validation",
            "max_batches": None,
            "evaluator_sha256": "e" * 64,
            "batch_size": 64,
            "num_workers": 0,
            "device": "cpu",
            "eligible_windows": 1,
            "source_windows": [1],
            "action_shuffle_pairing": {
                "same_source": True,
                "same_task": True,
                "different_episode": True,
                "coverage": 1.0,
            },
        },
        "sequence": {
            "history_size": 3,
            "rollout_steps": 5,
            "frame_skip": 5,
            "endpoint_horizons": [1, 3, 5],
            "shared_eligible_windows": True,
        },
        "bootstrap": {
            "method": "episode_block_within_task_equal_task_macro",
            "samples": 10_000,
            "seed": 5120,
            "confidence": 0.95,
        },
        "metrics": {
            "bootstrap": {
                "method": "episode_block_within_task_equal_task_macro",
                "samples": 10_000,
                "seed": 5120,
                "confidence": 0.95,
            },
            "teacher_forced_pred_mse": 0.1,
            "identity_copy_mse": 0.2,
            "oracle_alignment_mse": 0.3,
            "action_shuffle_mse": 0.5,
            "action_shuffle_delta_mse": 0.4,
            "action_sensitivity_mse": 0.5,
            "rollout_steps": 5,
            "num_batches": 1,
            "num_sequences": 1,
            "num_eligible_sequences": 1,
            "num_action_shuffle_sequences": 1,
            "action_shuffle_coverage": 1.0,
            "endpoint_horizons": [1, 3, 5],
            "endpoint_metrics": {},
            "aggregation": {
                "error_accumulation": "float64_sse_then_element_count",
                "window_to_episode": "element_weighted_mse",
                "episode_to_task": "equal_episode_mean",
                "task_to_macro": "equal_task_mean",
                "ratio_macro": "equal_mean_of_task_ratios",
            },
            "oracle_token_mse": {
                "prediction": {"anchor_mse": [0.1, 0.2, 0.3]},
                "alignment": {"anchor_mse": [0.4, 0.5, 0.6]},
            },
            "episode_blocks": [
                {
                    "source_index": 0,
                    "task_name": "pick-place-v3",
                    "episode_id": 7,
                    "num_sequences": 1,
                    "num_action_shuffle_sequences": 1,
                    "teacher_forced": {
                        "element_count": 2,
                        "predicted_mse": 0.1,
                        "identity_copy_mse": 0.2,
                        "prediction_vs_copy_ratio": 0.5,
                        "action_shuffle_mse": 0.5,
                        "action_shuffle_delta_mse": 0.4,
                        "action_sensitivity_mse": 0.5,
                    },
                    "oracle_alignment": {
                        "element_count": 2,
                        "mse": 0.3,
                    },
                    "recursive_h1_action_shuffle": {
                        "element_count": 2,
                        "mse": 0.9,
                        "delta_mse": 0.5,
                    },
                    "endpoints": {
                        "1": {
                            "element_count": 2,
                            "predicted_mse": 0.4,
                            "hold_last_identity_mse": 0.8,
                            "ratio": 0.5,
                        },
                        "3": {
                            "element_count": 2,
                            "predicted_mse": 0.6,
                            "hold_last_identity_mse": 1.2,
                            "ratio": 0.5,
                        },
                        "5": {
                            "element_count": 2,
                            "predicted_mse": 0.8,
                            "hold_last_identity_mse": 1.6,
                            "ratio": 0.5,
                        },
                    },
                }
            ],
            "per_task": per_task,
            "macro": macro,
        },
        "per_task": deepcopy(per_task),
        "macro": deepcopy(macro),
    }


def test_cell_report_normalizes_strict_development_contract(tmp_path: Path) -> None:
    output = tmp_path / "cell.json"
    normalized = normalize_cell_report(_report(output), _contract(output))

    assert normalized["train_mode"] == "triplet_nominal"
    assert normalized["eval_mode"] == "paired_visual"
    assert normalized["metrics"]["teacher_forced_pred_mse"] == pytest.approx(0.1)
    assert normalized["episode_blocks"][0]["episode_id"] == 7
    assert len(normalized["report_sha256"]) == 64


def test_normalize_cell_report_attaches_frozen_episode_annotations(
    tmp_path: Path,
) -> None:
    output = tmp_path / "cell.json"
    report = _report(output)
    annotations = {
        (0, "pick-place-v3", 7): {
            "camera_condition": "mild",
            "corrector_train_unseen": True,
        }
    }

    normalized = normalize_cell_report(
        report,
        _contract(output),
        annotations=annotations,
    )

    block = normalized["episode_blocks"][0]
    assert block["camera_condition"] == "mild"
    assert block["corrector_train_unseen"] is True
    assert normalized["metrics"]["episode_blocks"] == normalized["episode_blocks"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("step", 1999, "step mismatch"),
        ("eval_mode", "triplet_corrected", "eval_mode mismatch"),
        ("parameter_count", 1, "parameter_count mismatch"),
    ],
)
def test_cell_report_rejects_contract_drift(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    output = tmp_path / "cell.json"
    report = _report(output)
    report["anchor_visual_ablation"][field] = value

    with pytest.raises(ValueError, match=message):
        normalize_cell_report(report, _contract(output))


def test_cell_report_rejects_partial_action_pairing(tmp_path: Path) -> None:
    output = tmp_path / "cell.json"
    report = _report(output)
    report["evaluation"]["action_shuffle_pairing"]["coverage"] = 0.99

    with pytest.raises(ValueError, match="coverage"):
        normalize_cell_report(report, _contract(output))



def test_cell_report_rejects_missing_pairing_invariant(tmp_path: Path) -> None:
    output = tmp_path / "cell.json"
    report = _report(output)
    report["evaluation"]["action_shuffle_pairing"].pop("same_task")

    with pytest.raises(ValueError, match="same_task"):
        normalize_cell_report(report, _contract(output))


def test_cell_report_recomputes_pooled_metrics_from_episode_blocks(
    tmp_path: Path,
) -> None:
    output = tmp_path / "cell.json"
    report = _report(output)
    report["metrics"]["teacher_forced_pred_mse"] = 0.09

    with pytest.raises(ValueError, match="episode-block recomputation"):
        normalize_cell_report(report, _contract(output))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["evaluation"].__setitem__(
                "source_windows", [2]
            ),
            "sum",
        ),
        (
            lambda report: report["metrics"].__setitem__(
                "num_eligible_sequences", 2
            ),
            "num_eligible_sequences",
        ),
        (
            lambda report: report["metrics"]["episode_blocks"][0].__setitem__(
                "source_index", True
            ),
            "source_index",
        ),
        (
            lambda report: report["metrics"]["episode_blocks"][0].__setitem__(
                "source_index", 1
            ),
            "source_index",
        ),
        (
            lambda report: report["metrics"]["episode_blocks"][0].__setitem__(
                "num_sequences", True
            ),
            "num_sequences",
        ),
        (
            lambda report: report["metrics"]["episode_blocks"][0].__setitem__(
                "task_name", ""
            ),
            "task_name",
        ),
        (
            lambda report: report["metrics"]["episode_blocks"][0].__setitem__(
                "episode_id", False
            ),
            "episode_id",
        ),
        (
            lambda report: report["metrics"]["episode_blocks"][0][
                "endpoints"
            ]["3"].__setitem__("element_count", 4),
            "endpoint",
        ),
    ],
)
def test_cell_report_closes_window_identity_and_endpoint_counts(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    output = tmp_path / "cell.json"
    report = _report(output)
    mutation(report)  # type: ignore[operator]

    with pytest.raises((TypeError, ValueError), match=message):
        normalize_cell_report(report, _contract(output))


def test_cell_report_recomputes_each_source_window(tmp_path: Path) -> None:
    output = tmp_path / "cell.json"
    report = _report(output)
    second = deepcopy(report["metrics"]["episode_blocks"][0])
    second["task_name"] = "second-task"
    second["episode_id"] = 8
    report["metrics"]["episode_blocks"].append(second)
    report["evaluation"]["eligible_windows"] = 2
    report["evaluation"]["source_windows"] = [1, 1]
    for field in (
        "num_sequences",
        "num_eligible_sequences",
        "num_action_shuffle_sequences",
    ):
        report["metrics"][field] = 2
    report["metrics"]["per_task"]["second-task"] = {
        "num_sources": 1,
        "num_episodes": 1,
        "num_sequences": 1,
    }
    report["per_task"] = deepcopy(report["metrics"]["per_task"])
    for field, value in (
        ("num_tasks", 2),
        ("num_episodes", 2),
        ("num_sequences", 2),
    ):
        report["metrics"]["macro"][field] = value
    report["macro"] = deepcopy(report["metrics"]["macro"])

    with pytest.raises(ValueError, match="per-source sequence counts"):
        normalize_cell_report(report, _contract(output))


def test_cell_report_rejects_duplicate_episode_keys(tmp_path: Path) -> None:
    output = tmp_path / "cell.json"
    report = _report(output)
    report["metrics"]["episode_blocks"].append(
        deepcopy(report["metrics"]["episode_blocks"][0])
    )

    with pytest.raises(ValueError, match="duplicate episode block key"):
        normalize_cell_report(report, _contract(output))


def _frozen_checkpoint() -> dict[str, object]:
    return {
        "step": 2000,
        "metrics": {"step": 2000, "loss": 1.0},
        "resolved_config": {
            "seed": 5120,
            "loader": {"batch_size": 192},
            "optimizer": {"warmup_steps": 127},
            "trainer": {"max_steps": 2000},
            "loss": {"sigreg_weight": 0.02},
            "contract": {
                "optimizer_step_budget": 2000,
                "expected_parameter_count": 19_152_816,
            },
        },
    }


def test_frozen_training_constants_are_explicit_and_enforced() -> None:
    assert EXPECTED_TRAINING_CONTRACT == {
        "seed": 5120,
        "batch_size": 192,
        "warmup_steps": 127,
        "max_steps": 2000,
        "parameter_count": 19_152_816,
        "sigreg_weight": 0.02,
    }
    _validate_frozen_training_contract(_frozen_checkpoint(), "triplet_nominal")


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        (None, "step", 1999, "checkpoint step"),
        ("resolved_config", "seed", 1, "seed"),
        ("loader", "batch_size", 64, "batch_size"),
        ("optimizer", "warmup_steps", 126, "warmup_steps"),
        ("trainer", "max_steps", 1999, "max_steps"),
        ("loss", "sigreg_weight", 0.0, "sigreg_weight"),
        ("contract", "expected_parameter_count", 1, "parameter_count"),
    ],
)
def test_frozen_training_contract_rejects_drift(
    section: str | None,
    field: str,
    value: object,
    message: str,
) -> None:
    checkpoint = _frozen_checkpoint()
    if section is None:
        checkpoint[field] = value
    elif section == "resolved_config":
        checkpoint[section][field] = value  # type: ignore[index]
    else:
        checkpoint["resolved_config"][section][field] = value  # type: ignore[index]

    with pytest.raises(ValueError, match=message):
        _validate_frozen_training_contract(checkpoint, "triplet_nominal")


def test_metrics_tail_must_be_step_2000_and_equal_checkpoint_metrics(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        '{"step": 1000, "loss": 2.0}\n'
        '{"step": 2000, "loss": 1.0}\n',
        encoding="utf-8",
    )
    assert _load_metrics_tail(metrics_path, {"step": 2000, "loss": 1.0}) == {
        "step": 2000,
        "loss": 1.0,
    }

    with pytest.raises(ValueError, match="checkpoint metrics"):
        _load_metrics_tail(metrics_path, {"step": 2000, "loss": 1.1})


def _training_signature(blob: str = "a" * 40) -> dict[str, str]:
    return {
        **{path: f"100644:{blob}" for path in TRAINING_SOURCE_FILES},
        "third_party/lewm/module.py": f"100644:{blob}",
    }


def test_training_source_blob_mismatch_fails_even_when_commits_may_differ() -> None:
    modes = (
        "triplet_nominal",
        "triplet_corrected",
        "paired_visual",
        "paired_innovation",
    )
    signatures = {mode: _training_signature() for mode in modes}
    signatures["paired_innovation"]["fcpg/lewm_runtime.py"] = "100644:" + "b" * 40

    with pytest.raises(ValueError, match="training source blob mismatch"):
        validate_training_blob_signatures(signatures)


def _write_complete_cell(output: Path, contract: dict[str, object]) -> None:
    report = _report(output)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    contract_path = _cell_contract_path(output)
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    completion = _completion_record(output, contract_path)
    _cell_completion_path(output).write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_partial_cell_artifacts_fail_closed(tmp_path: Path) -> None:
    output = tmp_path / "cell.json"
    output.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="partial cell artifacts"):
        _load_reusable_cell(output, _contract(output))


def test_complete_matching_cell_is_reused_and_hash_checked(tmp_path: Path) -> None:
    output = tmp_path / "cell.json"
    contract = _contract(output)
    _write_complete_cell(output, contract)

    normalized = _load_reusable_cell(output, contract)
    assert normalized is not None
    assert normalized["report_sha256"] == _sha256(output)

    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="output SHA-256"):
        _load_reusable_cell(output, contract)


def test_reuse_rejects_completion_with_another_ledger(tmp_path: Path) -> None:
    output = tmp_path / "cell.json"
    contract = _contract(output)
    _write_complete_cell(output, contract)
    completion_path = _cell_completion_path(output)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["checkpoint_ledger_sha256"] = "b" * 64
    completion_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + chr(10),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checkpoint-ledger binding"):
        _load_reusable_cell(output, contract)


def test_new_cell_writes_completion_only_after_validation(tmp_path: Path) -> None:
    output = tmp_path / "cell.json"
    contract = _contract(output)
    report = _report(output)
    output.write_text(json.dumps(report) + "\n", encoding="utf-8")
    _cell_contract_path(output).write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    normalized = _seal_cell_output(output, contract)

    assert normalized["report_sha256"] == _sha256(output)
    assert _cell_completion_path(output).is_file()


def test_evaluator_command_passes_expected_checkpoint_hash(tmp_path: Path) -> None:
    sources = []
    for source_index in range(4):
        source: dict[str, str] = {}
        for field in ("path", "correction_cache", "oracle_target_cache"):
            artifact = tmp_path / f"{source_index}-{field}.bin"
            artifact.write_bytes(b"fixture")
            source[field] = str(artifact)
        sources.append(source)

    command = build_evaluator_command(
        checkpoint=tmp_path / "checkpoint.pt",
        expected_checkpoint_sha256="b" * 64,
        eval_mode="paired_visual",
        data_config={"sources": sources},
        output=tmp_path / "cell.json",
        device="cpu",
        batch_size=8,
        num_workers=0,
        bootstrap_samples=100,
        bootstrap_seed=5120,
    )

    hash_flag = command.index("--expected-checkpoint-sha256")
    assert command[hash_flag + 1] == "b" * 64
    assert command[command.index("--device") + 1] == "cpu"


def test_cell_contract_binds_protocol_slice_inputs_and_evaluator_parameters(
    tmp_path: Path,
) -> None:
    output = tmp_path / "cell.json"
    evaluator_parameters = {
        "split": "validation",
        "grounding_mode": "checkpoint",
        "rollout_steps": 5,
        "batch_size": 32,
        "num_workers": 2,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 5120,
        "device": "cpu",
        "max_batches": None,
    }
    contract = _cell_contract(
        train_mode="triplet_nominal",
        eval_mode="paired_visual",
        ledger_path=tmp_path / "ledger.json",
        ledger_sha256="a" * 64,
        checkpoint={
            "path": str(tmp_path / "lewm_grounded_latest.pt"),
            "sha256": "c" * 64,
            "metrics_path": str(tmp_path / "metrics.jsonl"),
            "metrics_sha256": "m" * 64,
            "input_manifest_path": str(tmp_path / "input_manifest.json"),
            "input_manifest_sha256": "i" * 64,
        },
        output=output,
        dev_protocol_path=tmp_path / "protocol.json",
        dev_protocol_sha256="p" * 64,
        corrector_slice_path=tmp_path / "slice.json",
        corrector_slice_sha256="s" * 64,
        evaluator_sha256="e" * 64,
        execution_code_sha256="x" * 64,
        matrix_runner_sha256="r" * 64,
        repository_state={
            "head": "1" * 40,
            "worktree_sha256": "2" * 64,
        },
        frozen_input_artifacts_sha256="3" * 64,
        evaluator_parameters=evaluator_parameters,
        command=["python", "evaluate.py"],
    )

    assert contract["checkpoint_ledger_sha256"] == "a" * 64
    assert contract["dev_protocol_sha256"] == "p" * 64
    assert contract["corrector_slice_sha256"] == "s" * 64
    assert contract["metrics_sha256"] == "m" * 64
    assert contract["input_manifest_sha256"] == "i" * 64
    assert contract["matrix_runner_sha256"] == "r" * 64
    assert contract["repository_head"] == "1" * 40
    assert contract["repository_worktree_sha256"] == "2" * 64
    assert contract["frozen_input_artifacts_sha256"] == "3" * 64
    assert contract["evaluator_parameters"] == evaluator_parameters


def test_default_protocol_and_declared_counts_are_frozen() -> None:
    parsed = build_parser().parse_args(["--ledger", "ledger.json"])
    assert parsed.protocol == str(DEFAULT_DEV_PROTOCOL)
    assert parsed.device is None
    assert parsed.ledger == "ledger.json"
    protocol = json.loads(DEFAULT_DEV_PROTOCOL.read_text(encoding="utf-8"))
    _validate_dev_protocol(protocol)
    assert protocol["data"]["full_validation_counts"] == {
        "clean": 56,
        "mild": 112,
        "total": 168,
    }

    changed = deepcopy(protocol)
    changed["data"]["full_validation_counts"]["total"] = 167
    with pytest.raises(ValueError, match="full validation counts"):
        _validate_dev_protocol(changed)

    changed = deepcopy(protocol)
    changed["claim_boundary"]["development_only"] = False
    with pytest.raises(ValueError, match="claim boundary"):
        _validate_dev_protocol(changed)



def test_corrector_slice_consumes_checkpoint_split_overlap_and_disclosure() -> None:
    protocol = json.loads(DEFAULT_DEV_PROTOCOL.read_text(encoding="utf-8"))
    slice_path = Path(protocol["data"]["corrector_train_unseen_slice"])
    corrector_slice = json.loads(slice_path.read_text(encoding="utf-8"))
    _validate_corrector_slice(corrector_slice, protocol["data"])

    for field in (
        "corrector_checkpoint_sha256",
        "corrector_split",
        "overlap_set_sha256",
        "corrector_selection_disclosure",
    ):
        changed = deepcopy(corrector_slice)
        changed[field] = {} if field == "corrector_split" else "tampered"
        with pytest.raises(ValueError):
            _validate_corrector_slice(changed, protocol["data"])


def test_legacy_corrector_binding_checks_actual_hash_and_split(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "corrector.pt"
    torch.save(
        {
            "args": {
                "split_seed": 0,
                "split_unit": "task-episode",
                "val_fraction": 0.2,
            }
        },
        checkpoint_path,
    )
    checkpoint_sha256 = _sha256(checkpoint_path)
    source = {
        "correction_cache": {
            "corrector_checkpoint": {
                "absolute_path": str(checkpoint_path),
                "sha256": checkpoint_sha256,
            }
        }
    }
    records = {
        mode: {"manifest": {"sources": [source]}}
        for mode in (
            "triplet_nominal",
            "triplet_corrected",
            "paired_visual",
            "paired_innovation",
        )
    }
    corrector_slice = {
        "corrector_checkpoint_sha256": checkpoint_sha256,
        "corrector_split": {
            "split_seed": 0,
            "split_unit": "task-episode",
            "val_fraction": 0.2,
        },
    }

    binding = _validate_legacy_corrector_checkpoint_binding(
        records, corrector_slice
    )
    assert binding["sha256"] == checkpoint_sha256
    assert binding["evidence_level"] == (
        "legacy_split_replay_not_historical_provenance"
    )

    changed = deepcopy(corrector_slice)
    changed["corrector_split"]["split_seed"] = 1
    with pytest.raises(ValueError, match="split arguments"):
        _validate_legacy_corrector_checkpoint_binding(records, changed)


def test_legacy_split_replay_closes_membership_and_hashes(tmp_path: Path) -> None:
    modes = (
        "triplet_nominal",
        "triplet_corrected",
        "paired_visual",
        "paired_innovation",
    )
    task_names = ("button", "pick", "push", "reach")
    validation_counts = (11, 11, 10, 9)
    train_counts = (31, 31, 32, 33)
    manifest_sources: dict[str, object] = {}
    frozen_sources: list[dict[str, object]] = []
    slice_tasks: dict[str, object] = {}
    validation_overlap: set[tuple[str, int]] = set()
    train_overlap: set[tuple[str, int]] = set()

    for index, (task_name, validation_count, train_count) in enumerate(
        zip(task_names, validation_counts, train_counts, strict=True)
    ):
        source_name = f"source-{index}"
        source_path = tmp_path / f"{source_name}.npz"
        episode_ids = np.arange(300, dtype=np.int64)
        np.savez(
            source_path,
            task_name=np.asarray([task_name] * 300),
            episode_id=episode_ids,
        )
        with np.load(source_path, allow_pickle=False) as raw:
            split = make_split(raw, 0.2, 0, "task-episode")
        train_episodes = set(episode_ids[split.train].tolist())
        validation_episodes = set(episode_ids[split.test].tolist())
        assert len(train_episodes) == 240
        assert len(validation_episodes) == 60

        selected_validation = set(
            sorted(validation_episodes)[:validation_count]
        )
        selected_train = set(sorted(train_episodes)[:train_count])
        lewm_episodes = selected_validation | selected_train
        validation_overlap.update(
            (task_name, episode) for episode in selected_validation
        )
        train_overlap.update((task_name, episode) for episode in selected_train)

        digest = _sha256(source_path)
        manifest_sources[source_name] = {
            "path": str(source_path),
            "data_sha256": digest,
            "val_episodes": [
                f"{task_name}:{episode}" for episode in sorted(lewm_episodes)
            ],
        }
        frozen_sources.append(
            {
                "name": source_name,
                "dataset": {
                    "absolute_path": str(source_path),
                    "sha256": digest,
                },
            }
        )
        slice_tasks[task_name] = {
            "clean": sorted(selected_validation),
            "mild": [],
        }

    corrector_slice = {
        "corrector_split": {
            "split_seed": 0,
            "split_unit": "task-episode",
            "val_fraction": 0.2,
        },
        "episode_set_sha256": _episode_membership_sha256(validation_overlap),
        "overlap_set_sha256": _episode_membership_sha256(train_overlap),
        "tasks": slice_tasks,
    }
    episode_manifest = {
        "protocol": "lewm_gate_l1_camera_aware_episode_split",
        "sources": manifest_sources,
    }
    records = {
        mode: {"manifest": {"sources": deepcopy(frozen_sources)}}
        for mode in modes
    }

    replay = replay_legacy_corrector_split(
        records,
        corrector_slice,
        episode_manifest,
    )

    assert replay["evidence_level"] == (
        "legacy_split_replay_not_historical_provenance"
    )
    assert replay["corrector_episode_counts"] == {
        "train": 960,
        "validation": 240,
    }
    assert replay["lewm_validation_overlap_counts"] == {
        "train": 127,
        "validation": 41,
    }
    assert replay["episode_set_sha256"] == corrector_slice["episode_set_sha256"]
    assert replay["overlap_set_sha256"] == corrector_slice["overlap_set_sha256"]
    for task_name, validation_count, train_count in zip(
        task_names, validation_counts, train_counts, strict=True
    ):
        assert len(
            replay["tasks"][task_name]["corrector_validation_overlap"]
        ) == validation_count
        assert len(replay["tasks"][task_name]["corrector_train_overlap"]) == (
            train_count
        )

    tampered = deepcopy(corrector_slice)
    tampered["episode_set_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="41-episode membership SHA-256"):
        replay_legacy_corrector_split(records, tampered, episode_manifest)


def test_annotation_counts_bind_full_and_corrector_unseen_slices() -> None:
    annotations = {}
    for episode in range(168):
        condition = "clean" if episode < 56 else "mild"
        unseen = episode < 12 or 56 <= episode < 85
        annotations[(0, "task", episode)] = {
            "camera_condition": condition,
            "corrector_train_unseen": unseen,
        }

    counts = _validate_annotation_counts(
        annotations,
        EXPECTED_FULL_VALIDATION_COUNTS,
    )

    assert counts["full"] == EXPECTED_FULL_VALIDATION_COUNTS
    assert counts["corrector_train_unseen"] == EXPECTED_UNSEEN_COUNTS

    annotations[(0, "task", 12)]["corrector_train_unseen"] = True
    with pytest.raises(ValueError, match="corrector-train-unseen counts"):
        _validate_annotation_counts(annotations, EXPECTED_FULL_VALIDATION_COUNTS)



def test_result_lock_prevents_concurrent_matrix_writers(tmp_path: Path) -> None:
    first = _acquire_result_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="another anchor-visual matrix runner"):
            _acquire_result_lock(tmp_path)
    finally:
        first.close()

    second = _acquire_result_lock(tmp_path)
    second.close()


def test_manifest_artifact_snapshot_deduplicates_and_detects_mutation(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.npz"
    target = tmp_path / "target.npy"
    dataset.write_bytes(b"dataset")
    target.write_bytes(b"target")
    dataset_record = {
        "absolute_path": str(dataset),
        "sha256": _sha256(dataset),
    }
    manifests = {
        "triplet_nominal": {
            "sources": [
                {
                    "dataset": dataset_record,
                    "oracle": {
                        "absolute_path": str(target),
                        "sha256": _sha256(target),
                    },
                }
            ]
        },
        "paired_innovation": {
            "sources": [{"dataset": deepcopy(dataset_record)}]
        },
    }

    snapshot = _manifest_artifact_hash_snapshot(manifests)
    assert len(snapshot) == 2
    _validate_hash_snapshot(snapshot)

    target.write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="changed after preflight"):
        _validate_hash_snapshot(snapshot)


def test_manifest_artifact_snapshot_rejects_conflicting_hashes(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "dataset.npz"
    artifact.write_bytes(b"dataset")
    manifests = {
        "triplet_nominal": {
            "dataset": {
                "absolute_path": str(artifact),
                "sha256": _sha256(artifact),
            }
        },
        "paired_innovation": {
            "dataset": {
                "absolute_path": str(artifact),
                "sha256": "f" * 64,
            }
        },
    }

    with pytest.raises(ValueError, match="disagree on artifact hash"):
        _manifest_artifact_hash_snapshot(manifests)


def test_final_hash_snapshot_detects_post_evaluation_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"before")
    snapshot = {
        "triplet_nominal checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
        }
    }
    _validate_hash_snapshot(snapshot)

    checkpoint.write_bytes(b"after")
    with pytest.raises(RuntimeError, match="changed after preflight"):
        _validate_hash_snapshot(snapshot)


def test_paired_analysis_converts_to_report_schema_with_nn_scaling() -> None:
    modes = (
        "triplet_nominal",
        "triplet_corrected",
        "paired_visual",
        "paired_innovation",
    )
    paired = {
        "baseline": {
            "train_mode": "triplet_nominal",
            "eval_mode": "triplet_nominal",
        },
        "cells": [
            {
                "train_mode": train_mode,
                "eval_mode": eval_mode,
                "macro_relative_improvement": 0.1,
                "absolute_delta": 1.0,
                "relative_improvement_95_ci": [0.05, 0.15],
                "probability_improvement_gt_zero": 0.99,
                "task_count": 4,
                "episode_count": 100,
            }
            for train_mode in modes
            for eval_mode in modes
        ],
        "nominal_corrected_2x2": {
            "cells": {
                "nominal_train_nominal_eval": 10.0,
                "nominal_train_corrected_eval": 8.0,
                "corrected_train_nominal_eval": 9.0,
                "corrected_train_corrected_eval": 6.0,
            },
            "fixed_train_input_effect": {
                "triplet_nominal": 2.0,
                "triplet_corrected": 3.0,
            },
            "fixed_eval_training_effect": {
                "triplet_nominal": 1.0,
                "triplet_corrected": 2.0,
            },
            "matched_gain": 4.0,
            "raw_interaction": -1.0,
        },
    }

    sections = _paired_report_sections(paired)

    assert len(sections["paired_episode_bootstrap"]["cells"]) == 16
    factor = sections["factor_decomposition"]
    assert factor["cells"] == {"NN": 10.0, "NC": 8.0, "CN": 9.0, "CC": 6.0}
    assert factor["effects"]["eval_input_at_corrected_train"] == {
        "raw_mse_delta": 3.0,
        "relative_improvement": pytest.approx(0.3),
    }
    assert factor["effects"]["interaction"] == {"raw_mse_delta": -1.0}


def test_text_output_is_idempotent_and_refuses_different_content(
    tmp_path: Path,
) -> None:
    output = tmp_path / "summary.md"
    _write_text_idempotent(output, "same\n")
    _write_text_idempotent(output, "same\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _write_text_idempotent(output, "different\n")
