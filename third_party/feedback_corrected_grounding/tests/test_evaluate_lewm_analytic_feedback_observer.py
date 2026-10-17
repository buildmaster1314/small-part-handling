from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import evaluate_lewm_analytic_feedback_observer as cli


def _write_current_artifact(path: Path) -> None:
    np.savez_compressed(
        path,
        history_token=np.zeros((1, 8, 3, 3, 256), dtype=np.float32),
        history_action=np.zeros((1, 8, 2, 20), dtype=np.float32),
    )


def test_cli_writes_fail_closed_receipt_for_current_contract(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "deployment.npz"
    output = tmp_path / "audit.json"
    _write_current_artifact(cache)
    code = cli.main(
        ["--deployment-cache", str(cache), "--output", str(output)]
    )
    assert code == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["decision"] == "Fail-Closed-Unscorable"
    assert report["uses_future_clean_target"] is False


def test_cli_accepts_four_state_causal_contract(tmp_path: Path) -> None:
    cache = tmp_path / "deployment.npz"
    np.savez_compressed(
        cache,
        history_token=np.zeros((1, 8, 3, 3, 256), dtype=np.float32),
        history_action=np.zeros((1, 8, 2, 20), dtype=np.float32),
        observer_history_token=np.zeros((1, 8, 4, 3, 256), dtype=np.float32),
        observer_history_action=np.zeros((1, 8, 3, 20), dtype=np.float32),
    )
    assert cli.main(["--deployment-cache", str(cache)]) == 0
