from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_lewm_counterfactual_v3_exploratory_crossfit.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "v3_exploratory_crossfit_cli_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_help_has_no_training_or_device_controls(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    with pytest.raises(SystemExit) as stopped:
        module.main(["--help"])
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    assert "--confirm-exploratory-crossfit" in help_text
    assert "--device" not in help_text
    assert "--direct-steps" not in help_text
    assert "--residual-steps" not in help_text
    assert "--folds" not in help_text


def test_synthetic_crossfit_smoke_is_isolated_and_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    naive_output = tmp_path / "state_oracle_residual_seed9216"
    naive_output.mkdir()
    destination = tmp_path / "state_oracle_residual_crossfit_seed9216"
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
    monkeypatch.setattr(module, "prepare_gate_bundle", lambda *_args: bundle)

    def run_model(value: object) -> object:
        calls["bundle"] = value
        return SimpleNamespace(
            report={
                "target": "actual_return_h5",
                "scorers": {
                    "anonymous_slot": {},
                    "axis_sign": {},
                    "direct_crossfit": {},
                    "direct_plus_state_oracle_residual_crossfit": {},
                },
            }
        )

    monkeypatch.setattr(module, "_run_model", run_model)

    assert module.main(
        [
            "--dataset",
            str(dataset_path),
            "--output",
            str(destination),
            "--confirm-exploratory-crossfit",
        ]
    ) == 0

    report = json.loads((destination / "report.json").read_text(encoding="utf-8"))
    assert calls["bundle"] is bundle
    assert report["classification"] == "post_hoc_exploratory"
    assert report["evaluation_design"] == "four_fold_physical_episode_crossfit"
    assert report["post_hoc"] is True
    assert report["exposed_validation"] is True
    assert report["descriptive_only"] is True
    assert report["formal_claim_allowed"] is False
    assert report["crossfit_folds"] == 4
    assert report["device"] == "cuda:0"
    assert report["direct_steps_per_fold"] == 2000
    assert report["residual_steps"] == 2000
    assert report["cublas_workspace_config"] == ":4096:8"
    assert naive_output.is_dir()
    assert not (naive_output / "report.json").exists()

    with pytest.raises(FileExistsError, match="already exists"):
        module.main(
            [
                "--dataset",
                str(dataset_path),
                "--output",
                str(destination),
                "--confirm-exploratory-crossfit",
            ]
        )


def test_confirmation_is_required_and_cublas_is_set() -> None:
    module = _module()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    with pytest.raises(SystemExit, match="--confirm-exploratory-crossfit"):
        module.main([])
