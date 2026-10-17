from __future__ import annotations

from scripts import summarize_lewm_towel_grounding_comparison as summary


def _result(mode: str, value: float) -> dict:
    metric = {
        f"H{h}": {"anchor_mean_epe_px": value + h}
        for h in range(1, 9)
    }
    return {
        "root": {"checkpoint": "/tmp/model.pt"},
        "metadata": {"grounding_mode": mode},
        "task": {
            "environment_seed": 7,
            "raw_frame_count": 100,
            "stable_success_step": 90,
            "task_instance": 3,
            "video": "prediction.mp4",
            "current_grounding_anchor_mean_epe_px": value,
            "stitched_anchor_mean_epe_px": value + 10,
            "horizon_metrics": metric,
        },
    }


def test_matched_identity_rejects_different_physical_replay() -> None:
    results = {mode: _result(mode, float(i)) for i, mode in enumerate(summary.MODES)}
    results["oracle"]["task"]["environment_seed"] = 8
    try:
        summary._matched_identity(results)
    except ValueError as error:
        assert "physical replay" in str(error)
    else:
        raise AssertionError("different physical replays must be rejected")


def test_render_reports_all_grounding_modes(tmp_path) -> None:
    results = {mode: _result(mode, float(i)) for i, mode in enumerate(summary.MODES)}
    roots = {mode: tmp_path / mode for mode in summary.MODES}
    identity = summary._matched_identity(results)
    document = summary._render(results, roots, tmp_path / "report", identity)
    assert "Nominal" in document
    assert "Fcpg" in document
    assert "Oracle" in document
    assert "Current EPE" in document
