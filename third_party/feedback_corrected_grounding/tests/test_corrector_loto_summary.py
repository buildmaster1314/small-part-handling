from __future__ import annotations

import json

import pytest

from scripts.summarize_corrector_loto import summarize_evaluations


def _write_evaluation(path, ratio: float, after: float) -> None:
    path.write_text(
        json.dumps(
            {
                "validation_frames": 6000,
                "results": {
                    "normal": {
                        "e_before_mean_px": 6.0,
                        "e_after_mean_px": after,
                        "e_after_p90_px": after * 2.0,
                        "ratio": ratio,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_loto_summary_reports_all_tasks_including_failures(tmp_path) -> None:
    pick = tmp_path / "pick.json"
    button = tmp_path / "button.json"
    _write_evaluation(pick, ratio=0.10, after=0.6)
    _write_evaluation(button, ratio=0.30, after=1.8)

    summary = summarize_evaluations(
        [("pick", pick), ("button", button)], ratio_threshold=0.15
    )

    assert [row["pass"] for row in summary["tasks"]] == [True, False]
    assert summary["aggregate"]["pass_count"] == 1
    assert not summary["aggregate"]["all_pass"]
    assert summary["aggregate"]["ratio_mean"] == pytest.approx(0.20)
