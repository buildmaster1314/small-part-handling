"""Tests for the fail-closed fresh complementary-observer summary CLI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fcpg.lewm_p1_native_complementary_statistics import (
    HORIZONS,
    MODEL_SEEDS,
    PROTOCOL,
    REQUIRED_METHODS,
    TASK,
)
from scripts import summarize_lewm_p1_native_complementary_fresh as cli


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifact(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha(path)


def _formal_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, dict[str, object]]:
    source = json.loads(cli.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    data_root = tmp_path / "data"
    deployment = data_root / "deployment.npz"
    labels = data_root / "labels.npz"
    metadata = data_root / "metadata.json"
    _write_artifact(deployment, b"fresh-deployment")
    _write_artifact(labels, b"fresh-labels")
    _write_artifact(metadata, b'{"fresh":true}\n')
    source["fresh_data"]["deployment_cache"] = str(deployment)
    source["fresh_data"]["label_cache"] = str(labels)
    source["fresh_data"]["metadata"] = str(metadata)

    policy = tmp_path / "checkpoints" / "policy.pt"
    corrector = tmp_path / "checkpoints" / "corrector.pt"
    source["p1"]["policy_checkpoint"] = str(policy)
    source["p1"]["policy_checkpoint_sha256"] = _write_artifact(policy, b"policy")
    source["p1"]["corrector_checkpoint"] = str(corrector)
    source["p1"]["corrector_checkpoint_sha256"] = _write_artifact(
        corrector, b"corrector"
    )

    checkpoint_sha: dict[int, str] = {}
    for item in source["frozen_predictors"]:
        seed = int(item["seed"])
        checkpoint = tmp_path / "checkpoints" / f"predictor-{seed}.pt"
        digest = _write_artifact(checkpoint, f"predictor-{seed}".encode("ascii"))
        item["checkpoint"] = str(checkpoint)
        item["sha256"] = digest
        checkpoint_sha[seed] = digest

    config = tmp_path / "frozen.json"
    config.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(cli, "DEFAULT_CONFIG", config)

    config_sha = _sha(config)
    artifact_hashes = {
        "deployment_cache_sha256": _sha(deployment),
        "label_cache_sha256": _sha(labels),
        "metadata_sha256": _sha(metadata),
    }
    dataset_sha = cli._canonical_digest(
        {"config_sha256": config_sha, **artifact_hashes}
    )
    rows: list[dict[str, object]] = []
    method_factor = {
        REQUIRED_METHODS[0]: 0.50,
        REQUIRED_METHODS[1]: 1.00,
        REQUIRED_METHODS[2]: 1.20,
        REQUIRED_METHODS[3]: 1.10,
        REQUIRED_METHODS[4]: 1.05,
        **{method: 1.30 for method in REQUIRED_METHODS[5:]},
    }
    for seed in MODEL_SEEDS:
        for method in REQUIRED_METHODS:
            for episode in range(30):
                for horizon in HORIZONS:
                    base = 1.0 + 0.002 * episode + 0.003 * (horizon - 3)
                    rows.append(
                        {
                            "schema_version": 1,
                            "classification": "formal",
                            "collector_protocol": cli.COLLECTOR_PROTOCOL,
                            "config_sha256": config_sha,
                            **artifact_hashes,
                            "protocol": PROTOCOL,
                            "task": TASK,
                            "model_seed": seed,
                            "method": method,
                            "physical_episode_id": f"episode-{episode:02d}",
                            "horizon": horizon,
                            "episode_horizon_mse": base * method_factor[method],
                            "statistical_unit": "physical_episode",
                            "dataset_sha256": dataset_sha,
                            "checkpoint_sha256": checkpoint_sha[seed],
                        }
                    )
    rows_path = tmp_path / "formal_rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return config, rows_path, source


def _rewrite_rows(path: Path, transform) -> None:  # type: ignore[no-untyped-def]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    transform(rows)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_cli_writes_formal_go_summary_and_markdown_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, rows, _ = _formal_fixture(tmp_path, monkeypatch)
    output = tmp_path / "formal-summary"

    assert cli.main(["--config", str(config), "--rows", str(rows), "--output-dir", str(output)]) == 0

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    markdown = (output / "summary.md").read_text(encoding="utf-8")
    assert summary["cross_seed_gate"]["decision"] == "Fresh-Go"
    assert summary["formal_input"]["classification"] == "formal"
    assert summary["formal_input"]["row_count"] == 6480
    assert summary["bootstrap"]["samples"] == 10_000
    assert "**Decision: Fresh-Go** (3/3 model seeds passed)" in markdown
    assert "known visual dropout only" in markdown
    assert not list(tmp_path.glob(".formal-summary.tmp-*"))


def test_refuses_to_overwrite_existing_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, rows, _ = _formal_fixture(tmp_path, monkeypatch)
    output = tmp_path / "summary"
    summary = cli.summarize_formal_rows(rows, config_path=config)
    cli.write_summary_atomic(summary, output)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        cli.write_summary_atomic(summary, output)


def test_rejects_nonformal_rows_without_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, rows, _ = _formal_fixture(tmp_path, monkeypatch)
    _rewrite_rows(rows, lambda values: values[0].__setitem__("classification", "diagnostic"))
    output = tmp_path / "summary"

    with pytest.raises(ValueError, match="not a formal fresh evaluation row"):
        cli.main(["--config", str(config), "--rows", str(rows), "--output-dir", str(output)])
    assert not output.exists()


def test_rejects_incomplete_panel_without_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, rows, _ = _formal_fixture(tmp_path, monkeypatch)
    lines = rows.read_text(encoding="utf-8").splitlines()
    rows.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    output = tmp_path / "summary"

    with pytest.raises(ValueError, match="exactly 6480 rows"):
        cli.main(["--config", str(config), "--rows", str(rows), "--output-dir", str(output)])
    assert not output.exists()


@pytest.mark.parametrize(
    "field",
    [
        "config_sha256",
        "deployment_cache_sha256",
        "label_cache_sha256",
        "metadata_sha256",
        "dataset_sha256",
        "checkpoint_sha256",
    ],
)
def test_rejects_row_provenance_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    config, rows, _ = _formal_fixture(tmp_path, monkeypatch)
    _rewrite_rows(rows, lambda values: values[0].__setitem__(field, "0" * 64))

    with pytest.raises(ValueError, match="provenance mismatch"):
        cli.summarize_formal_rows(rows, config_path=config)


def test_rejects_live_dataset_and_checkpoint_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, rows, source = _formal_fixture(tmp_path, monkeypatch)
    deployment = Path(source["fresh_data"]["deployment_cache"])
    deployment.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="deployment_cache_sha256 provenance mismatch"):
        cli.summarize_formal_rows(rows, config_path=config)

    deployment.write_bytes(b"fresh-deployment")
    checkpoint = Path(source["frozen_predictors"][0]["checkpoint"])
    checkpoint.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        cli.summarize_formal_rows(rows, config_path=config)


def test_rejects_alternate_config_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, rows, source = _formal_fixture(tmp_path, monkeypatch)
    drifted = tmp_path / "drifted.json"
    source["evaluation"]["bootstrap_seed"] += 1
    drifted.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="checked-in frozen config"):
        cli.summarize_formal_rows(rows, config_path=drifted)
