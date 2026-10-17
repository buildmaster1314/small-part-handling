from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from scripts.render_lewm_action_branch_a4_accuracy_gallery import (
    ALL_HORIZONS,
    CAUSAL_OBSERVATION_FIELDS,
    DEFAULT_A4_CHECKPOINTS,
    DISPLAY_HORIZONS,
    _build_causal_model_inputs,
    _canonical_device,
    _decision_window_starts,
    _encoder_observation_records,
    _horizon_frame_indices,
    _parser,
    _pixel_epe,
    _render_html,
    _summarize_horizon_metrics,
    _validate_causal_input_contract,
)


def _record(value: float) -> dict[str, np.ndarray]:
    return {
        "rgb": np.full((3, 4, 4), value, dtype=np.float32),
        "proprio": np.full(4, value, dtype=np.float32),
        "keypoints_xyz": np.full((3, 3), value, dtype=np.float32),
        "q_nominal": np.full((3, 2), value, dtype=np.float32),
        "valid_nominal": np.ones(3, dtype=bool),
        "q_true": np.full((3, 2), value + 1_000.0, dtype=np.float32),
        "valid_true": np.ones(3, dtype=bool),
    }


def _metrics() -> dict[str, dict[str, dict[str, float | int]]]:
    horizon_metrics = {
        f"H{horizon}": {
            "horizon": horizon,
            "sample_count": 2,
            "mean_epe_px": float(horizon),
            "median_epe_px": float(horizon) + 0.1,
            "p95_epe_px": float(horizon) + 0.2,
            "pck_at_4": 0.5,
            "pck_at_8": 1.0,
        }
        for horizon in ALL_HORIZONS
    }
    return {
        "decoder_floor": deepcopy(horizon_metrics),
        "teacher_forced": deepcopy(horizon_metrics),
        "recursive": deepcopy(horizon_metrics),
    }


def test_horizon_alignment_uses_exact_five_step_targets_and_stops_at_h8() -> None:
    np.testing.assert_array_equal(
        _horizon_frame_indices(20, DISPLAY_HORIZONS),
        np.asarray([25, 35, 45, 60]),
    )
    np.testing.assert_array_equal(
        _horizon_frame_indices(20),
        np.arange(25, 61, 5),
    )
    assert _decision_window_starts(60) == (10, 15, 20)
    assert _decision_window_starts(60, max_decision_windows=2) == (10, 15)
    with pytest.raises(ValueError, match="H1-H8"):
        _horizon_frame_indices(20, (9,))


def test_pixel_epe_and_per_horizon_distribution_metrics() -> None:
    truth = np.zeros((2, 8, 2), dtype=np.float32)
    predicted = np.zeros_like(truth)
    predicted[0, :, 0] = 0.0
    predicted[1, :, 0] = 10.0
    errors = _pixel_epe(truth, predicted)
    np.testing.assert_array_equal(errors[:, 0], np.asarray([0.0, 10.0]))

    metrics = _summarize_horizon_metrics(truth, predicted)
    for horizon in ALL_HORIZONS:
        row = metrics[f"H{horizon}"]
        assert row["mean_epe_px"] == pytest.approx(5.0)
        assert row["median_epe_px"] == pytest.approx(5.0)
        assert row["p95_epe_px"] == pytest.approx(9.5)
        assert row["pck_at_4"] == pytest.approx(0.5)
        assert row["pck_at_8"] == pytest.approx(0.5)


def test_html_prominently_states_probe_and_future_rgb_risks() -> None:
    page = _render_html(
        [
            {
                "task": "push-v3",
                "task_label": "Push",
                "task_instance": 10,
                "environment_seed": 620_154,
                "decision_window_count": 3,
                "gif": 'push"><script>alert(1)</script>.gif',
                "metrics": _metrics(),
            }
        ],
        {"checkpoint_seeds": [30_301, 30_302, 30_303], "decoder": "decoder.npz"},
    )
    assert "latent forecast decoded by train-only probe, not RGB generation" in page
    assert "Actual future RGB and GT q-center are evaluation-only overlays" in page
    assert "never model inputs" in page
    assert "target-decode error is the probe/representation floor" in page
    assert "teacher-forced error isolates one-step dynamics" in page
    assert "recursive error additionally includes rollout drift" in page
    assert "frozen current-frame FCPG correction" in page
    assert "PCK@4" in page
    assert "PCK@8" in page
    assert "H8" in page
    assert "H9" not in page
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_causal_model_input_contract_ignores_all_future_rgb_and_gt() -> None:
    records = [_record(float(index)) for index in range(61)]
    actions = np.arange(60 * 4, dtype=np.float32).reshape(60, 4)
    alignment, inputs = _build_causal_model_inputs(
        records,
        actions,
        decision_raw_step=10,
        task_index=3,
    )
    assert alignment.history_frame_indices == (0, 5, 10)
    assert alignment.future_frame_indices == (15, 20, 25, 30, 35, 40, 45, 50)
    assert all(
        tuple(observation) == CAUSAL_OBSERVATION_FIELDS
        for observation in inputs.history_observations
    )
    assert all("q_true" not in observation for observation in inputs.history_observations)
    assert inputs.history_action.shape == (2, 20)
    assert inputs.future_action.shape == (8, 20)
    np.testing.assert_array_equal(inputs.history_action.reshape(10, 4), actions[:10])
    np.testing.assert_array_equal(inputs.future_action.reshape(40, 4), actions[10:50])
    contract = _validate_causal_input_contract(inputs)
    assert contract["future_rgb"] == "forbidden"
    assert contract["future_gt_q_center"] == "forbidden"
    encoder_records = _encoder_observation_records(inputs)
    assert all(
        set(vars(record)) == set(CAUSAL_OBSERVATION_FIELDS)
        for record in encoder_records
    )
    assert all(not hasattr(record, "q_true") for record in encoder_records)
    assert all(not hasattr(record, "valid_true") for record in encoder_records)

    changed = deepcopy(records)
    for index in range(11, len(changed)):
        changed[index]["rgb"].fill(99_000.0 + index)
        changed[index]["q_true"].fill(-99_000.0 - index)
    changed_alignment, changed_inputs = _build_causal_model_inputs(
        changed,
        actions,
        decision_raw_step=10,
        task_index=3,
    )
    assert changed_alignment == alignment
    for original, observed in zip(
        inputs.history_observations,
        changed_inputs.history_observations,
        strict=True,
    ):
        for field in CAUSAL_OBSERVATION_FIELDS:
            np.testing.assert_array_equal(original[field], observed[field])
    np.testing.assert_array_equal(inputs.history_action, changed_inputs.history_action)
    np.testing.assert_array_equal(inputs.future_action, changed_inputs.future_action)


def test_cli_defaults_to_three_10k_endpoints_and_never_cuda_index_one() -> None:
    args = _parser().parse_args([])
    assert args.device == "cuda:0"
    assert args.max_decision_windows is None
    assert [path.parent.name for path in DEFAULT_A4_CHECKPOINTS] == [
        "shared_7task_seed30301_10k",
        "shared_7task_seed30302_10k",
        "shared_7task_seed30303_10k",
    ]
    assert str(_canonical_device("cuda")) == "cuda:0"
    assert str(_canonical_device("cpu")) == "cpu"
    with pytest.raises(ValueError, match="index 0"):
        _canonical_device("cuda:1")
