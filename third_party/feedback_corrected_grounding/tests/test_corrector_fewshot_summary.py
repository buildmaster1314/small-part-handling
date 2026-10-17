from __future__ import annotations

import json

import pytest

from scripts.summarize_corrector_fewshot import summarize_fewshot


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


def test_fewshot_summary_sorts_budgets_and_keeps_failures(tmp_path) -> None:
    zero = tmp_path / "zero.json"
    ten = tmp_path / "ten.json"
    _write_evaluation(zero, ratio=0.30, after=1.8)
    _write_evaluation(ten, ratio=0.14, after=0.84)

    summary = summarize_fewshot(
        [(10, ten), (0, zero)], ratio_threshold=0.15
    )

    assert [row["episodes"] for row in summary["points"]] == [0, 10]
    assert [row["pass"] for row in summary["points"]] == [False, True]
    with pytest.raises(ValueError, match="must be unique"):
        summarize_fewshot([(0, zero), (0, ten)], ratio_threshold=0.15)
