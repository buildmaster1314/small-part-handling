from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import scripts.collect_lewm_counterfactual_v2 as cli
from fcpg.lewm_counterfactual_v2 import (
    FULL_CANDIDATE_NAMES,
    SMOKE_CANDIDATE_NAMES,
    smoke_config,
    validate_full_config,
    validate_smoke_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/lewm/eval/counterfactual_v2_development_seed8192.json"


def _read_config() -> dict[str, Any]:
    with CONFIG.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def test_checked_in_development_config_and_smoke_override_are_exact() -> None:
    full = _read_config()
    original = copy.deepcopy(full)
    validate_full_config(full)

    smoke = smoke_config(full)
    validate_smoke_config(smoke)

    assert full == original
    assert full["evaluation"]["candidate_names"] == list(FULL_CANDIDATE_NAMES)
    assert smoke["evaluation"]["tasks"] == full["evaluation"]["tasks"]
    assert smoke["evaluation"]["phase_steps"] == [10, 25, 35]
    assert smoke["evaluation"]["candidate_names"] == list(SMOKE_CANDIDATE_NAMES)
    assert smoke["conditions"] == full["conditions"]
    assert smoke["inputs"] == full["inputs"]
    assert smoke["output"]["directory"].endswith("_smoke")


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--smoke", "--confirm-development"],
    ],
)
def test_cli_requires_exactly_one_explicit_mode(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        cli._parser().parse_args(arguments)
    assert error.value.code == 2


def test_resolved_config_binding_persists_object_and_matching_hash() -> None:
    source = _read_config()
    resolved = smoke_config(source)

    provenance = cli._bind_resolved_config(
        {"collector": "unit-test"},
        resolved_config=resolved,
    )

    assert provenance["collector"] == "unit-test"
    assert provenance["config"] == resolved
    assert provenance["config"] is not resolved
    assert provenance["config_sha256"] == cli.canonical_sha256(resolved)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("config", {"wrong": True}, "config disagrees"),
        ("config_sha256", "0" * 64, "config_sha256 disagrees"),
    ],
)
def test_resolved_config_binding_rejects_collector_disagreement(
    tmp_path: Path,
    field: str,
    value: Any,
    match: str,
) -> None:
    config = _read_config()
    with pytest.raises(ValueError, match=match):
        cli._bind_resolved_config(
            {field: value},
            resolved_config=config,
        )


@pytest.mark.parametrize(
    ("provenance", "expected"),
    [
        ({}, "not_audited"),
        ({"upstream_non_overlap": {"status": "not_auditable"}}, "not_auditable"),
        ({"upstream_non_overlap": {"status": "verified"}}, "verified"),
    ],
)
def test_upstream_non_overlap_status_is_fail_closed(
    provenance: dict[str, Any], expected: str
) -> None:
    assert cli._upstream_non_overlap_status(provenance) == expected


def test_smoke_cli_validates_collects_publishes_and_audits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _read_config()
    config["output"]["directory"] = "store"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    calls: list[str] = []
    payload = SimpleNamespace(
        deployment_inputs={"deployment": object()},
        offline_labels={"labels": object()},
        audit_arrays={"audit": object()},
        index_rows=tuple({"sample": index} for index in range(12)),
        provenance={"collector": "unit-test"},
    )

    def validate_full(value: dict[str, Any]) -> None:
        calls.append("validate_full")
        assert value["output"]["directory"] == "store"

    def make_smoke(value: dict[str, Any]) -> dict[str, Any]:
        calls.append("smoke_config")
        result = copy.deepcopy(value)
        result["evaluation"]["candidate_names"] = list(SMOKE_CANDIDATE_NAMES)
        result["output"]["directory"] = "store_smoke"
        return result

    def validate_smoke(value: dict[str, Any]) -> None:
        calls.append("validate_smoke")
        assert value["output"]["directory"] == "store_smoke"

    def collect(value: dict[str, Any], *, repo_root: Path) -> Any:
        calls.append("collect")
        assert repo_root == tmp_path
        assert value["evaluation"]["candidate_names"] == list(
            SMOKE_CANDIDATE_NAMES
        )
        return payload

    array_axes = SimpleNamespace(sample_count=12, candidate_count=4)

    def validate_arrays(*_args: Any, candidate_count: int) -> Any:
        calls.append("validate_arrays")
        assert candidate_count == 4
        return array_axes

    def build_manifest(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append("build_g0")
        assert kwargs["index_rows"] == payload.index_rows
        assert kwargs["provenance"]["collector"] == "unit-test"
        assert kwargs["provenance"]["config"]["output"]["directory"] == (
            "store_smoke"
        )
        resolved_hash = kwargs["provenance"]["config_sha256"]
        return {
            "manifest_sha256": "a" * 64,
            "config_sha256": resolved_hash,
            "g0_audit": {
                "g0_eligible": False,
                "index_and_provenance": {
                    "ineligibility_reasons": [
                        "bounded_smoke_is_not_g0_eligible",
                        "upstream_episode_non_overlap_not_auditable",
                    ]
                },
            },
        }

    def write_store(destination: Path, **kwargs: Any) -> dict[str, Any]:
        calls.append("write")
        assert destination == tmp_path / "store_smoke"
        assert kwargs["index_rows"] == payload.index_rows
        scientific = kwargs["provenance"]["scientific_provenance"]
        assert scientific["collector"] == "unit-test"
        assert scientific["config"]["output"]["directory"] == "store_smoke"
        assert scientific["config_sha256"] == cli.canonical_sha256(
            scientific["config"]
        )
        assert kwargs["provenance"]["g0_manifest"]["manifest_sha256"] == (
            "a" * 64
        )
        destination.mkdir()
        return {"result_sha256": "b" * 64}

    def audit_store(destination: Path) -> Any:
        calls.append("audit")
        assert destination == tmp_path / "store_smoke"
        return SimpleNamespace(result_sha256="b" * 64)

    def audit_scientific(
        destination: Path,
        *,
        config: dict[str, Any],
        expected_g0_manifest: dict[str, Any],
    ) -> tuple[Any, Any]:
        calls.append("audit_scientific")
        assert destination == tmp_path / "store_smoke"
        assert config["output"]["directory"] == "store_smoke"
        assert expected_g0_manifest["manifest_sha256"] == "a" * 64
        return SimpleNamespace(), array_axes

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "validate_full_config", validate_full)
    monkeypatch.setattr(cli, "smoke_config", make_smoke)
    monkeypatch.setattr(cli, "validate_smoke_config", validate_smoke)
    monkeypatch.setattr(cli, "run_counterfactual_v2_collection", collect)
    monkeypatch.setattr(cli, "validate_array_bundle", validate_arrays)
    monkeypatch.setattr(cli, "build_g0_manifest", build_manifest)
    monkeypatch.setattr(cli, "write_counterfactual_v2_store", write_store)
    monkeypatch.setattr(cli, "audit_counterfactual_v2_store", audit_store)
    monkeypatch.setattr(
        cli,
        "_audit_published_scientific_contract",
        audit_scientific,
    )

    assert cli.main(["--config", str(config_path), "--smoke"]) == 0
    assert calls == [
        "validate_full",
        "smoke_config",
        "validate_smoke",
        "collect",
        "validate_arrays",
        "build_g0",
        "write",
        "audit",
        "audit_scientific",
    ]
    output = capsys.readouterr().out
    assert f"destination: {tmp_path / 'store_smoke'}" in output
    assert f"result_sha256: {'b' * 64}" in output
    assert "samples: 12" in output
    assert "physical_branches: 48" in output
    assert "candidate_families: 4 (per-snapshot anonymous slot permutation)" in output
    assert "scope: external_development" in output
    assert "g0_eligible: false" in output
    assert "upstream_non_overlap: not_audited" in output
    assert "bounded_smoke_is_not_g0_eligible" in output


def _published_dataset(
    *,
    embedded_g0: dict[str, Any],
    config: dict[str, Any],
    scientific: dict[str, Any] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        deployment_inputs={"deployment": object()},
        offline_labels={"labels": object()},
        audit_arrays={"audit": object()},
        index_rows=({"sample_index": 0},),
        provenance={
            "scientific_provenance": scientific
            or {"collector": "unit-test", "config": copy.deepcopy(config)},
            "g0_manifest": embedded_g0,
        },
    )


def test_post_publication_audit_reloads_and_rebuilds_scientific_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _read_config()
    config = smoke_config(config)
    g0 = {"manifest_sha256": "a" * 64, "proof": {"semantic": "b" * 64}}
    dataset = _published_dataset(embedded_g0=g0, config=config)
    axes = SimpleNamespace(sample_count=1, candidate_count=4)
    calls: list[str] = []

    monkeypatch.setattr(
        cli,
        "load_counterfactual_v2_store",
        lambda destination: calls.append(f"load:{destination.name}") or dataset,
    )
    monkeypatch.setattr(
        cli,
        "validate_array_bundle",
        lambda *_args, candidate_count: (
            calls.append(f"arrays:{candidate_count}") or axes
        ),
    )

    def validate_contract(*_args: Any, smoke: bool) -> dict[str, Any]:
        calls.append(f"contract:{smoke}")
        return {"passed": True}

    monkeypatch.setattr(cli, "validate_index_and_provenance", validate_contract)
    monkeypatch.setattr(
        cli,
        "validate_manifest_hash",
        lambda value: calls.append(f"manifest:{value['manifest_sha256'][0]}"),
    )
    monkeypatch.setattr(
        cli,
        "build_g0_manifest",
        lambda *_args, **_kwargs: calls.append("rebuild") or copy.deepcopy(g0),
    )

    loaded, published_axes = cli._audit_published_scientific_contract(
        tmp_path / "store",
        config=config,
        expected_g0_manifest=copy.deepcopy(g0),
    )

    assert loaded is dataset
    assert published_axes is axes
    assert calls == [
        "load:store",
        "arrays:4",
        "contract:True",
        "manifest:a",
        "rebuild",
        "manifest:a",
    ]


@pytest.mark.parametrize(
    ("embedded", "rebuilt", "match"),
    [
        (
            {"manifest_sha256": "a" * 64, "value": 2},
            {"manifest_sha256": "a" * 64, "value": 2},
            "pre-publication",
        ),
        (
            {"manifest_sha256": "a" * 64, "value": 1},
            {"manifest_sha256": "a" * 64, "value": 2},
            "rebuilt published",
        ),
    ],
)
def test_post_publication_audit_rejects_g0_semantic_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    embedded: dict[str, Any],
    rebuilt: dict[str, Any],
    match: str,
) -> None:
    config = smoke_config(_read_config())
    expected = {"manifest_sha256": "a" * 64, "value": 1}
    dataset = _published_dataset(embedded_g0=embedded, config=config)
    monkeypatch.setattr(cli, "load_counterfactual_v2_store", lambda _path: dataset)
    monkeypatch.setattr(
        cli,
        "validate_array_bundle",
        lambda *_args, **_kwargs: SimpleNamespace(sample_count=1, candidate_count=4),
    )
    monkeypatch.setattr(
        cli, "validate_index_and_provenance", lambda *_args, **_kwargs: {"passed": True}
    )
    monkeypatch.setattr(cli, "validate_manifest_hash", lambda _value: None)
    monkeypatch.setattr(cli, "build_g0_manifest", lambda *_args, **_kwargs: rebuilt)

    with pytest.raises(RuntimeError, match=match):
        cli._audit_published_scientific_contract(
            tmp_path / "store",
            config=config,
            expected_g0_manifest=expected,
        )


def test_post_publication_audit_propagates_scientific_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = smoke_config(_read_config())
    g0 = {"manifest_sha256": "a" * 64}
    dataset = _published_dataset(embedded_g0=g0, config=config)
    monkeypatch.setattr(cli, "load_counterfactual_v2_store", lambda _path: dataset)
    monkeypatch.setattr(
        cli,
        "validate_array_bundle",
        lambda *_args, **_kwargs: SimpleNamespace(sample_count=1, candidate_count=4),
    )

    def reject_contract(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("published provenance mismatch")

    monkeypatch.setattr(cli, "validate_index_and_provenance", reject_contract)

    with pytest.raises(ValueError, match="published provenance mismatch"):
        cli._audit_published_scientific_contract(
            tmp_path / "store",
            config=config,
            expected_g0_manifest=g0,
        )


def test_post_publication_audit_rejects_published_config_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = smoke_config(_read_config())
    published_config = copy.deepcopy(config)
    published_config["seed"] += 1
    g0 = {"manifest_sha256": "a" * 64}
    dataset = _published_dataset(
        embedded_g0=g0,
        config=config,
        scientific={"config": published_config},
    )
    monkeypatch.setattr(cli, "load_counterfactual_v2_store", lambda _path: dataset)

    with pytest.raises(RuntimeError, match="scientific config disagrees"):
        cli._audit_published_scientific_contract(
            tmp_path / "store",
            config=config,
            expected_g0_manifest=g0,
        )


def test_existing_destination_fails_before_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _read_config()
    config["output"]["directory"] = "already-there"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "already-there").mkdir()

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "validate_full_config", lambda _value: None)
    called = False

    def collect(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("collector must not run")

    monkeypatch.setattr(cli, "run_counterfactual_v2_collection", collect)
    with pytest.raises(FileExistsError, match="already exists"):
        cli.main(
            [
                "--config",
                str(config_path),
                "--confirm-development",
            ]
        )
    assert called is False
