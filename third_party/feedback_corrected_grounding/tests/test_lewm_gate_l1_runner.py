from __future__ import annotations

import json
from pathlib import Path

import pytest

from fcpg.lewm_provenance import sha256_file
from scripts.run_lewm_gate_l1_matrix import (
    _build_cell_contract,
    _validate_contract_inputs,
    validate_condition_inputs,
)


def _write_condition(root: Path, name: str) -> dict[str, str | None]:
    data = root / f"{name}.npz"
    data.write_bytes(f"dataset-{name}".encode("ascii"))

    correction_checkpoint = root / "correction-checkpoint.pt"
    correction_checkpoint.write_bytes(b"correction-checkpoint")
    correction = root / f"{name}-correction"
    correction.mkdir()
    correction_array = correction / "q_corrected.npy"
    correction_array.write_bytes(b"corrected")
    (correction / "metadata.json").write_text(
        json.dumps(
            {
                "source_data": str(data),
                "source_data_sha256": sha256_file(data),
                "q_corrected_sha256": sha256_file(correction_array),
                "checkpoint": str(correction_checkpoint),
                "checkpoint_sha256": sha256_file(correction_checkpoint),
            }
        ),
        encoding="utf-8",
    )

    target_encoder = root / "target-encoder.pt"
    target_encoder.write_bytes(b"target-encoder")
    target = root / f"{name}-target"
    target.mkdir()
    target_array = target / "oracle_target.npy"
    target_array.write_bytes(b"target")
    (target / "metadata.json").write_text(
        json.dumps(
            {
                "source_data": str(data),
                "source_data_sha256": sha256_file(data),
                "oracle_target_sha256": sha256_file(target_array),
                "encoder_checkpoint": str(target_encoder),
                "encoder_checkpoint_sha256": sha256_file(target_encoder),
            }
        ),
        encoding="utf-8",
    )
    return {
        "data": str(data),
        "correction_cache": str(correction),
        "oracle_target_cache": str(target),
        "pairing_report": None,
    }


def _conditions(tmp_path: Path) -> dict[str, dict[str, str | None]]:
    conditions = {
        name: _write_condition(tmp_path, name)
        for name in ("clean", "rotation5", "translation4", "rotation10")
    }
    clean = Path(str(conditions["clean"]["data"]))
    for name in ("rotation5", "translation4", "rotation10"):
        intervention = Path(str(conditions[name]["data"]))
        report = tmp_path / f"{name}-pairing.json"
        report.write_text(
            json.dumps(
                {
                    "validated": True,
                    "reference_path": str(clean),
                    "reference_sha256": sha256_file(clean),
                    "intervention_path": str(intervention),
                    "intervention_sha256": sha256_file(intervention),
                    "physical_identity_sha256": "f" * 64,
                    "sample_count": 12000,
                }
            ),
            encoding="utf-8",
        )
        conditions[name]["pairing_report"] = str(report)
    return conditions


def _contract_for_record(
    tmp_path: Path, record: dict[str, object]
) -> dict[str, object]:
    manifest = tmp_path / "matrix.json"
    manifest.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    provenance_manifest = tmp_path / "checkpoint-manifest.json"
    provenance_manifest.write_text("{}\n", encoding="utf-8")
    run_sidecar = tmp_path / "input_manifest.json"
    run_sidecar.write_text("{}\n", encoding="utf-8")
    return _build_cell_contract(
        manifest_path=manifest,
        manifest_sha256=sha256_file(manifest),
        checkpoint={
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "target_encoder_sha256": record["target_encoder_sha256"],
            "provenance": {
                "manifest_path": str(provenance_manifest),
                "manifest_sha256": sha256_file(provenance_manifest),
                "run_sidecar_path": str(run_sidecar),
                "run_sidecar_sha256": sha256_file(run_sidecar),
            },
        },
        condition={"rollout_steps": 5, "expected_windows": 120},
        condition_record=record,
        condition_name="rotation5",
        train="nominal",
        evaluation="nominal",
        eval_config={
            "split": "all",
            "batch_size": 64,
            "num_workers": 0,
            "max_batches": None,
            "cell_bootstrap_samples": 1000,
            "bootstrap_seed": 3072,
            "expected_tasks": [
                "pick-place-v3",
                "button-press-v3",
                "reach-v3",
                "push-v3",
            ],
            "expected_episodes_per_task": 30,
        },
        device="cpu",
    )


def test_condition_validation_requires_one_clean_reference(tmp_path: Path) -> None:
    records = validate_condition_inputs(_conditions(tmp_path))

    assert set(records) == {"clean", "rotation5", "translation4", "rotation10"}
    assert len({record["target_encoder_sha256"] for record in records.values()}) == 1


@pytest.mark.parametrize(
    ("path_key", "message"),
    [
        ("correction_metadata_path", "correction metadata"),
        ("oracle_target_metadata_path", "oracle target metadata"),
        ("pairing_report_path", "pairing report"),
        ("target_encoder_checkpoint_path", "target encoder checkpoint"),
    ],
)
def test_condition_records_bind_metadata_pairing_and_encoder(
    tmp_path: Path, path_key: str, message: str
) -> None:
    records = validate_condition_inputs(_conditions(tmp_path))
    contract = _contract_for_record(tmp_path, records["rotation5"])
    path = Path(str(records["rotation5"][path_key]))
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match=message):
        _validate_contract_inputs(contract)


def test_condition_validation_rejects_other_clean_reference(tmp_path: Path) -> None:
    conditions = _conditions(tmp_path)
    report_path = Path(str(conditions["rotation5"]["pairing_report"]))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    other = tmp_path / "other-clean.npz"
    other.write_bytes(b"other-clean")
    report["reference_path"] = str(other)
    report["reference_sha256"] = sha256_file(other)
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="different clean reference"):
        validate_condition_inputs(conditions)


def test_condition_validation_requires_every_non_clean_pairing_report(
    tmp_path: Path,
) -> None:
    conditions = _conditions(tmp_path)
    conditions["rotation5"]["pairing_report"] = None

    with pytest.raises(ValueError, match="requires a pairing report"):
        validate_condition_inputs(conditions)


def test_condition_validation_requires_full_pairing_sample_count(
    tmp_path: Path,
) -> None:
    conditions = _conditions(tmp_path)
    report_path = Path(str(conditions["rotation5"]["pairing_report"]))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["sample_count"] = 11999
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="sample_count must be 12000"):
        validate_condition_inputs(conditions)


def test_condition_validation_rejects_path_escape_name(tmp_path: Path) -> None:
    conditions = _conditions(tmp_path)
    conditions["../escape"] = conditions.pop("rotation5")

    with pytest.raises(ValueError, match="unsafe Gate L1 condition name"):
        validate_condition_inputs(conditions)
