from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from fcpg.lewm_provenance import sha256_file
import scripts.run_lewm_gate_l1_matrix as runner
from scripts.run_lewm_gate_l1_matrix import (
    EXPECTED_TASKS,
    REPO_ROOT,
    _assert_manifest_unchanged,
    _build_cell_command,
    _cell_contract_path,
    _cell_sidecar,
    _gate_exit_code,
    _open_attempt_log,
    _run_cell,
    _validate_cell_artifacts,
    _validate_checkpoint_provenance,
    _write_json_exclusive,
    _write_json_idempotent,
    main,
    validate_matrix_manifest,
    validate_result_dir_is_ignored,
)


MANIFEST_PATH = REPO_ROOT / "config/lewm/eval/gate_l1_matrix_seed3072.json"


def _contract(tmp_path: Path, grounding: str = "nominal") -> dict[str, object]:
    root = tmp_path / grounding
    root.mkdir(parents=True)
    manifest = root / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    checkpoint = root / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    provenance_manifest = root / "frozen-input-manifest.json"
    provenance_manifest.write_text("{}\n", encoding="utf-8")
    provenance_sidecar = root / "input_manifest.json"
    provenance_sidecar.write_text("{}\n", encoding="utf-8")
    data = root / "data.npz"
    data.write_bytes(b"data")

    correction = root / "correction"
    correction.mkdir()
    correction_array = correction / "q_corrected.npy"
    correction_array.write_bytes(b"correction")
    correction_metadata = correction / "metadata.json"
    correction_metadata.write_text("{}\n", encoding="utf-8")
    correction_checkpoint = root / "correction-checkpoint.pt"
    correction_checkpoint.write_bytes(b"correction-checkpoint")

    target = root / "target"
    target.mkdir()
    target_array = target / "oracle_target.npy"
    target_array.write_bytes(b"target")
    target_metadata = target / "metadata.json"
    target_metadata.write_text("{}\n", encoding="utf-8")
    target_encoder = root / "target-encoder.pt"
    target_encoder.write_bytes(b"target-encoder")
    pairing_report = root / "pairing.json"
    pairing_report.write_text("{}\n", encoding="utf-8")
    return {
        "schema_version": 1,
        "protocol": "fcpg_grounded_lewm_gate_l1_cell_contract",
        "manifest_path": str(manifest),
        "manifest_initial_sha256": sha256_file(manifest),
        "condition": "clean",
        "train_grounding": "nominal",
        "evaluation_grounding": grounding,
        "device": "cpu",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_provenance_manifest_path": str(provenance_manifest),
        "checkpoint_provenance_manifest_sha256": sha256_file(provenance_manifest),
        "checkpoint_provenance_run_sidecar_path": str(provenance_sidecar),
        "checkpoint_provenance_run_sidecar_sha256": sha256_file(
            provenance_sidecar
        ),
        "data_path": str(data),
        "data_sha256": sha256_file(data),
        "correction_cache_path": str(correction),
        "correction_cache_sha256": sha256_file(correction_array),
        "correction_metadata_path": str(correction_metadata),
        "correction_metadata_sha256": sha256_file(correction_metadata),
        "correction_checkpoint_path": str(correction_checkpoint),
        "correction_checkpoint_sha256": sha256_file(correction_checkpoint),
        "oracle_target_cache_path": str(target),
        "oracle_target_cache_sha256": sha256_file(target_array),
        "oracle_target_metadata_path": str(target_metadata),
        "oracle_target_metadata_sha256": sha256_file(target_metadata),
        "target_encoder_checkpoint_path": str(target_encoder),
        "target_encoder_sha256": sha256_file(target_encoder),
        "pairing_report_path": str(pairing_report),
        "pairing_report_sha256": sha256_file(pairing_report),
        "split": "all",
        "rollout_steps": 5,
        "batch_size": 64,
        "num_workers": 0,
        "max_batches": None,
        "bootstrap_samples": 1000,
        "bootstrap_seed": 3072,
        "expected_windows": 120,
        "expected_tasks": list(EXPECTED_TASKS),
        "expected_episodes_per_task": 30,
    }


def _report(contract: dict[str, object]) -> dict[str, object]:
    grounding = contract["evaluation_grounding"]
    correction = (
        [
            {
                "input": contract["correction_cache_path"],
                "array_sha256": contract["correction_cache_sha256"],
                "metadata": contract["correction_metadata_path"],
                "metadata_sha256": contract["correction_metadata_sha256"],
            }
        ]
        if grounding == "fcpg"
        else []
    )
    blocks = [
        {
            "task_name": task,
            "episode_id": episode,
            "num_sequences": 1,
        }
        for task in EXPECTED_TASKS
        for episode in range(30)
    ]
    return {
        "schema_version": 1,
        "protocol": "grounded_lewm_offline_diagnostics",
        "checkpoint": contract["checkpoint_path"],
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "provenance": {
            "validated": True,
            "manifest": contract["checkpoint_provenance_manifest_path"],
            "manifest_sha256": contract[
                "checkpoint_provenance_manifest_sha256"
            ],
            "run_sidecar": contract[
                "checkpoint_provenance_run_sidecar_path"
            ],
        },
        "grounding_mode": grounding,
        "grounding_mode_request": grounding,
        "data": [contract["data_path"]],
        "correction_cache": correction,
        "oracle_target_cache": [
            {
                "input": contract["oracle_target_cache_path"],
                "array_sha256": contract["oracle_target_cache_sha256"],
                "metadata": contract["oracle_target_metadata_path"],
                "metadata_sha256": contract[
                    "oracle_target_metadata_sha256"
                ],
                "encoder_checkpoint_sha256": contract["target_encoder_sha256"],
            }
        ],
        "evaluation": {
            "split": contract["split"],
            "batch_size": contract["batch_size"],
            "num_workers": contract["num_workers"],
            "max_batches": contract["max_batches"],
            "device": contract["device"],
            "source_windows": [contract["expected_windows"]],
            "eligible_windows": contract["expected_windows"],
        },
        "sequence": {"rollout_steps": contract["rollout_steps"]},
        "bootstrap": {
            "samples": contract["bootstrap_samples"],
            "seed": contract["bootstrap_seed"],
        },
        "metrics": {
            "num_sequences": contract["expected_windows"],
            "num_eligible_sequences": contract["expected_windows"],
            "episode_blocks": blocks,
        },
    }


def _write_cell(
    tmp_path: Path, contract: dict[str, object], report: dict[str, object]
) -> Path:
    output = tmp_path / "cell.json"
    output.write_text(json.dumps(report), encoding="utf-8")
    _write_json_exclusive(
        _cell_contract_path(output),
        _cell_sidecar(contract, sha256_file(output)),
    )
    return output


def test_manifest_fixes_ignored_output_and_full_evaluation_shape() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    validate_matrix_manifest(manifest)

    evaluation = manifest["evaluation"]
    assert evaluation["result_dir"] == (
        "outputs/lewm_grounded/gate_l1_matrix_seed3072"
    )
    assert evaluation["expected_tasks"] == list(EXPECTED_TASKS)
    assert evaluation["expected_episodes_per_task"] == 30
    assert evaluation["max_batches"] is None
    assert {
        name: value["expected_windows"]
        for name, value in manifest["conditions"].items()
    } == {
        "clean": 7800,
        "rotation5": 7800,
        "translation4": 7800,
        "rotation10": 4800,
    }
    validate_result_dir_is_ignored(REPO_ROOT / evaluation["result_dir"])


def test_cell_command_passes_correction_cache_only_for_fcpg(tmp_path: Path) -> None:
    nominal = _build_cell_command(
        _contract(tmp_path, "nominal"),
        output=tmp_path / "nominal.json",
        device="cpu",
    )
    fcpg = _build_cell_command(
        _contract(tmp_path, "fcpg"),
        output=tmp_path / "fcpg.json",
        device="cpu",
    )

    assert nominal.count("--correction-cache") == 0
    assert fcpg.count("--correction-cache") == 1
    assert "--max-batches" not in nominal
    assert "--max-batches" not in fcpg
    with pytest.raises(ValueError, match="device differs"):
        _build_cell_command(
            _contract(tmp_path / "mismatch"),
            output=tmp_path / "mismatch.json",
            device="cuda:0",
        )


@pytest.mark.parametrize("grounding", ["nominal", "fcpg"])
def test_cell_artifacts_enforce_shared_contract(
    tmp_path: Path, grounding: str
) -> None:
    contract = _contract(tmp_path, grounding)
    report = _report(contract)
    output = _write_cell(tmp_path, contract, report)
    assert _validate_cell_artifacts(output, contract) == report

    Path(str(contract["data_path"])).write_bytes(b"changed")
    with pytest.raises(ValueError, match="data SHA-256 changed"):
        _validate_cell_artifacts(output, contract)

    Path(str(contract["data_path"])).write_bytes(b"data")
    Path(str(contract["checkpoint_path"])).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint SHA-256 changed"):
        _validate_cell_artifacts(output, contract)


def test_cell_sidecar_seals_exact_report_bytes(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    output = _write_cell(tmp_path, contract, _report(contract))
    sidecar = json.loads(_cell_contract_path(output).read_text(encoding="utf-8"))
    assert sidecar["report_sha256"] == sha256_file(output)

    output.write_bytes(output.read_bytes() + b" ")
    with pytest.raises(ValueError, match="report SHA-256 differs"):
        _validate_cell_artifacts(output, contract)


def test_cell_artifacts_reject_device_mismatch(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    report = _report(contract)
    report["evaluation"]["device"] = "cuda:0"
    output = _write_cell(tmp_path, contract, report)

    with pytest.raises(ValueError, match="device mismatch"):
        _validate_cell_artifacts(output, contract)


def test_cell_artifacts_reject_legacy_report_without_sidecar(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    output = tmp_path / "cell.json"
    output.write_text(json.dumps(_report(contract)), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="contract sidecar"):
        _validate_cell_artifacts(output, contract)


def test_cell_artifacts_require_exact_episode_ids(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    report = _report(contract)
    block = report["metrics"]["episode_blocks"][-1]
    assert block["episode_id"] == 29
    block["episode_id"] = 30
    output = _write_cell(tmp_path, contract, report)

    with pytest.raises(ValueError, match="episode IDs must be exactly 0..29"):
        _validate_cell_artifacts(output, contract)


def test_cell_artifacts_reject_partial_evaluation(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    report = _report(contract)
    report["evaluation"]["max_batches"] = 1
    output = _write_cell(tmp_path, contract, report)

    with pytest.raises(ValueError, match="max_batches mismatch"):
        _validate_cell_artifacts(output, contract)


def test_cell_artifacts_reject_different_manifest_contract(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    output = _write_cell(tmp_path, contract, _report(contract))
    changed = deepcopy(contract)
    changed["manifest_initial_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="contract differs"):
        _validate_cell_artifacts(output, changed)


def test_manifest_bytes_are_frozen(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    initial = b'{"value": 1}\n'
    manifest.write_bytes(initial)
    _assert_manifest_unchanged(manifest, initial)

    manifest.write_bytes(b'{"value": 2}\n')
    with pytest.raises(RuntimeError, match="manifest changed"):
        _assert_manifest_unchanged(manifest, initial)


def test_checkpoint_provenance_binds_external_and_run_sidecar_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint_path = run_dir / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    manifest = {"schema_version": 1, "sources": [], "repository": {}}
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    external = tmp_path / "frozen-input.json"
    external.write_text(payload, encoding="utf-8")
    sidecar = run_dir / "input_manifest.json"
    sidecar.write_text(payload, encoding="utf-8")
    digest = sha256_file(external)
    checkpoint = {
        "provenance": {
            "path": str(external),
            "sha256": digest,
            "manifest": manifest,
        }
    }
    calls = []
    monkeypatch.setattr(
        runner,
        "validate_manifest",
        lambda value, check_git: calls.append((value, check_git)),
    )

    record = _validate_checkpoint_provenance(checkpoint, checkpoint_path)
    assert record["manifest_sha256"] == digest
    assert record["run_sidecar_sha256"] == digest
    assert calls == [(manifest, True)]

    sidecar.write_text(payload + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar SHA-256 differs"):
        _validate_checkpoint_provenance(checkpoint, checkpoint_path)

    sidecar.write_text(payload, encoding="utf-8")
    external.write_text(payload + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="external frozen input manifest SHA-256"):
        _validate_checkpoint_provenance(checkpoint, checkpoint_path)


def test_attempt_logs_are_retry_safe(tmp_path: Path) -> None:
    output = tmp_path / "cell.json"
    first_path, first_log = _open_attempt_log(output)
    with first_log:
        first_log.write("first failure\n")
    second_path, second_log = _open_attempt_log(output)
    with second_log:
        second_log.write("retry\n")

    assert first_path.name == "cell.attempt-001.log"
    assert second_path.name == "cell.attempt-002.log"
    assert first_path.read_text(encoding="utf-8") == "first failure\n"
    assert second_path.read_text(encoding="utf-8") == "retry\n"


def test_run_cell_preserves_failed_attempt_log_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract(tmp_path / "inputs")
    result_dir = tmp_path / "results"
    result_dir.mkdir()

    def fail(*args: object, **kwargs: object) -> None:
        raise runner.subprocess.CalledProcessError(2, args[0])

    monkeypatch.setattr(runner.subprocess, "run", fail)
    with pytest.raises(runner.subprocess.CalledProcessError):
        _run_cell(
            contract=contract,
            condition="clean",
            train="nominal",
            evaluation="nominal",
            result_dir=result_dir,
            device="cpu",
        )

    def succeed(command: list[str], **kwargs: object) -> object:
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(_report(contract)), encoding="utf-8")
        return runner.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", succeed)
    assert _run_cell(
        contract=contract,
        condition="clean",
        train="nominal",
        evaluation="nominal",
        result_dir=result_dir,
        device="cpu",
    ) == _report(contract)

    logs = sorted(result_dir.glob("*.attempt-*.log"))
    assert [path.name for path in logs] == [
        "clean__train-nominal__eval-nominal.attempt-001.log",
        "clean__train-nominal__eval-nominal.attempt-002.log",
    ]


def test_runtime_failure_returns_exit_three(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--manifest", str(tmp_path / "missing.json")]) == 3
    assert "Gate L1 runtime failure" in capsys.readouterr().err


def test_summary_write_and_exit_code_contract(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    assert _write_json_idempotent(output, {"decision": "Go"}) is True
    assert _write_json_idempotent(output, {"decision": "Go"}) is False
    with pytest.raises(FileExistsError, match="different Gate L1 summary"):
        _write_json_idempotent(output, {"decision": "No-Go"})
    assert _gate_exit_code(True) == 0
    assert _gate_exit_code(False) == 1
