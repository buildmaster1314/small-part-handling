import pytest

from scripts.summarize_seeded_matrix import aggregate, normalize_rows, parse_seeded_policy


def test_normalize_main_matrix_and_aggregate_same_seed_differences() -> None:
    payload = {
        "native_policy": [
            {
                "policy": "jitter_s0",
                "task": "pick",
                "perturbation": "rotation",
                "severity_deg": 5,
                "success": 0.20,
                "grounding_error": 10.0,
            },
            {
                "policy": "fcpg_s0",
                "task": "pick",
                "perturbation": "rotation",
                "severity_deg": 5,
                "success": 0.40,
                "grounding_error": 2.0,
            },
            {
                "policy": "jitter_s1",
                "task": "pick",
                "perturbation": "rotation",
                "severity_deg": 5,
                "success": 0.30,
                "grounding_error": 11.0,
            },
            {
                "policy": "fcpg_s1",
                "task": "pick",
                "perturbation": "rotation",
                "severity_deg": 5,
                "success": 0.40,
                "grounding_error": 3.0,
            },
        ],
        "auc": [
            {"policy": "jitter_s0", "task": "pick", "native_success_auc": 0.2},
            {"policy": "jitter_s1", "task": "pick", "native_success_auc": 0.4},
            {"policy": "fcpg_s0", "task": "pick", "native_success_auc": 0.5},
            {"policy": "fcpg_s1", "task": "pick", "native_success_auc": 0.7},
        ],
    }

    rows, auc_rows = normalize_rows(payload)
    result = aggregate(rows, auc_rows, baseline="jitter")

    fcpg = next(row for row in result["summary"] if row["method"] == "fcpg")
    assert fcpg["seeds"] == [0, 1]
    assert fcpg["success_mean"] == pytest.approx(0.4)
    difference = result["paired_seed_differences"][0]
    assert difference["seed_differences"] == pytest.approx([0.2, 0.1])
    assert difference["difference_mean"] == pytest.approx(0.15)
    fcpg_auc = next(row for row in result["auc"] if row["method"] == "fcpg")
    assert fcpg_auc["auc_mean"] == pytest.approx(0.6)


def test_normalize_external_matrix_keeps_checkpoint_seed_pairing() -> None:
    payload = {
        "rows": [
            {
                "checkpoint": "seed0",
                "method": "nominal",
                "task": "pick",
                "perturbation": "translation",
                "severity": 4,
                "success_rate": 0.1,
                "corrected_pixel_error_mean": 5.0,
            },
            {
                "checkpoint": "seed0",
                "method": "pnp",
                "task": "pick",
                "perturbation": "translation",
                "severity": 4,
                "success_rate": 0.3,
                "corrected_pixel_error_mean": 2.0,
            },
            {
                "checkpoint": "seed1",
                "method": "nominal",
                "task": "pick",
                "perturbation": "translation",
                "severity": 4,
                "success_rate": 0.2,
                "corrected_pixel_error_mean": 6.0,
            },
            {
                "checkpoint": "seed1",
                "method": "pnp",
                "task": "pick",
                "perturbation": "translation",
                "severity": 4,
                "success_rate": 0.5,
                "corrected_pixel_error_mean": 2.5,
            },
        ],
        "auc": [
            {"checkpoint": "seed0", "method": "nominal", "task": "pick", "success_auc": 0.1},
            {"checkpoint": "seed1", "method": "nominal", "task": "pick", "success_auc": 0.2},
            {"checkpoint": "seed0", "method": "pnp", "task": "pick", "success_auc": 0.3},
            {"checkpoint": "seed1", "method": "pnp", "task": "pick", "success_auc": 0.5},
        ],
    }

    rows, auc_rows = normalize_rows(payload)
    result = aggregate(rows, auc_rows, baseline="nominal")

    difference = result["paired_seed_differences"][0]
    assert difference["candidate"] == "pnp"
    assert difference["seed_differences"] == pytest.approx([0.2, 0.3])
    assert difference["difference_mean"] == pytest.approx(0.25)


def test_policy_labels_require_explicit_seed_suffix() -> None:
    assert parse_seeded_policy("fcpg_s12") == ("fcpg", 12)
    with pytest.raises(ValueError):
        parse_seeded_policy("fcpg")
