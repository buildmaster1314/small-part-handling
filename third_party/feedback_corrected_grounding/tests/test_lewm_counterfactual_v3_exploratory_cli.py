from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_lewm_counterfactual_v3_exploratory.py"


def _module():
    spec = importlib.util.spec_from_file_location("v3_exploratory_cli_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_help_is_available(capsys: pytest.CaptureFixture[str]) -> None:
    module = _module()
    with pytest.raises(SystemExit) as stopped:
        module.main(["--help"])
    assert stopped.value.code == 0
    assert "--confirm-exploratory" in capsys.readouterr().out


def test_synthetic_smoke_writes_bounded_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    destination = tmp_path / "exploratory"
    dataset = SimpleNamespace(
        deployment_inputs={"candidate_action_raw": object()},
        offline_labels={"actual_return": object()},
        audit_arrays={"split_index": object()},
        index_rows=({"physical_episode_sha256": "episode"},),
    )
    bundle = object()
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        module,
        "load_counterfactual_v2_store",
        lambda path: dataset if path == dataset_path else None,
    )

    def prepare(*args: object) -> object:
        calls["prepare"] = args
        return bundle

    def run_model(value: object, **kwargs: object) -> object:
        calls["run"] = (value, kwargs)
        return SimpleNamespace(
            report={
                "target": "actual_return_h5",
                "scorers": {
                    "anonymous_slot": {},
                    "axis_sign": {},
                    "direct": {},
                    "direct_plus_state_oracle_residual": {},
                },
            }
        )

    monkeypatch.setattr(module, "prepare_gate_bundle", prepare)
    monkeypatch.setattr(module, "_run_model", run_model)

    assert module.main(
        [
            "--dataset",
            str(dataset_path),
            "--output",
            str(destination),
            "--confirm-exploratory",
        ]
    ) == 0

    report = json.loads((destination / "report.json").read_text(encoding="utf-8"))
    assert report["classification"] == "post_hoc_exploratory"
    assert report["post_hoc"] is True
    assert report["exposed_validation"] is True
    assert report["descriptive_only"] is True
    assert report["formal_claim_allowed"] is False
    assert report["target"] == "actual_return_h5"
    assert set(report["scorers"]) == {
        "anonymous_slot",
        "axis_sign",
        "direct",
        "direct_plus_state_oracle_residual",
    }
    assert calls["run"] == (
        bundle,
        {"device": "cuda:0", "direct_steps": 2000, "residual_steps": 2000},
    )

    with pytest.raises(FileExistsError, match="already exists"):
        module.main(
            [
                "--dataset",
                str(dataset_path),
                "--output",
                str(destination),
                "--confirm-exploratory",
            ]
        )


def test_confirmation_is_required() -> None:
    module = _module()
    with pytest.raises(SystemExit, match="--confirm-exploratory"):
        module.main([])
