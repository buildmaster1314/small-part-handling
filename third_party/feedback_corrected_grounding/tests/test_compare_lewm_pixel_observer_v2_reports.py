from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from fcpg.lewm_pixel_observer_v2_runtime import canonical_sha256
from fcpg.lewm_pixel_observer_v2_report import (
    EVALUATION_IMPLEMENTATION_RELATIVE_PATHS,
    episode_coverage,
)
from fcpg.lewm_provenance import sha256_file
from scripts.compare_lewm_pixel_observer_v2_reports import compare_reports
from scripts.evaluate_lewm_pixel_observer_v2 import (
    _completed_report_matches,
    _evaluation_implementation_manifest,
    _matched_training_protocol,
    _matrix_conditions,
    _select_evaluation_episodes,
)

def _resolved_config(*, use_latent: bool) -> dict[str, object]:
    output_name = "latent" if use_latent else "no_latent"
    return {
        "schema_version": 1,
        "seed": 7168,
        "target_mode": "paired_deployment",
        "base_checkpoint": "/models/base.pt",
        "output_dir": f"/outputs/{output_name}",
        "data": {"history_size": 3, "frame_skip": 5, "split_seed": 3072},
        "loader": {"batch_size": 128, "num_workers": 0, "pin_memory": True},
        "adapter": {"hidden_dim": 256, "use_predicted_latent": use_latent},
        "loss": {"huber_delta_px": 0.25},
        "optimizer": {
            "lr": 0.0003,
            "weight_decay": 0.0001,
            "betas": [0.9, 0.999],
            "eps": 0.00000001, "warmup_steps": 100,
        },
        "trainer": {"epochs": 200, "max_steps": 2000, "grad_clip": 1.0},
        "run_contract": {
            "kind": "formal",
            "effective_target_optimizer_steps": 2000,
            "configured_target_optimizer_steps": 2000,
        },
    }


def _report(*, use_latent: bool, offset: float = 0.0) -> dict[str, object]:
    rows = []
    for task, base in (("pick", 2.0), ("button", 4.0)):
        for episode in range(3):
            value = base + episode + offset
            rows.append(
                {
                    "source": "external",
                    "task": task,
                    "episode": str(episode),
                    "modes": {
                        "full_writeback": {
                            "prior_qtrue_dropout_epe_px": value,
                            "prior_qtrue_epe_px": value + 0.5,
                            "post_update_qtrue_dropout_epe_px": value + 1.0,
                            "post_update_qtrue_epe_px": value + 1.5,
                        }
                    },
                }
            )
    resolved = _resolved_config(use_latent=use_latent)
    implementation_files = {
        name: "7" * 64 for name in EVALUATION_IMPLEMENTATION_RELATIVE_PATHS
    }
    selection = episode_coverage(rows)
    return {
        "target_mode": "paired_deployment",
        "checkpoint": f"{'latent' if use_latent else 'no_latent'}.pt",
        "checkpoint_sha256": ("1" if use_latent else "2") * 64,
        "checkpoint_adapter_config": {"hidden_dim": 256, "use_predicted_latent": use_latent},
        "checkpoint_resolved_config": resolved,
        "checkpoint_resolved_config_sha256": canonical_sha256(resolved),
        "checkpoint_matched_training_protocol_sha256": canonical_sha256(_matched_training_protocol(resolved)),
        "base_checkpoint_sha256": "3" * 64,
        "checkpoint_training_budget": {"target_optimizer_steps": 2000},
        "training_data_provenance_sha256": "4" * 64,
        "implementation_files": implementation_files,
        "implementation_sha256": canonical_sha256(implementation_files),
        "evaluation_config_sha256": "5" * 64,
        "evaluation_data_provenance_sha256": "6" * 64,
        "evaluation_condition": "drop3",
        "evaluation_matrix_profile": "drift",
        "max_episodes_per_source": 0,
        "evaluation_episode_selection": selection,
        "evaluation_episode_selection_sha256": canonical_sha256(selection),
        "split": "all",
        "observer_gains": {"position_alpha": 1.0, "velocity_beta": 1.0},
        "classic_baselines": {"hold_last_gain": 1.0},
        "synthetic_measurement_dropout_contract": {
            "type": "oracle_known_measurement_dropout", "rgb_occluded": False,
            "autonomous_detector": False, "start_offset": 7, "length": 3,
        },
        "hidden_step_negative_control": {"measurement_hidden_from_step": False, "sampled_steps": 0},
        "primary_control_metric_family": "pre_measurement_prior_qtrue",
        "primary_control_metric": "prior_qtrue_dropout_epe_px",
        "episode_rows": rows,
    }


def _compare(
    latent: dict[str, object],
    no_latent: dict[str, object],
) -> dict[str, object]:
    return compare_reports(
        latent,
        no_latent,
        latent_report_sha256="a" * 64,
        no_latent_report_sha256="b" * 64,
        bootstrap_samples=5,
    )


def _rebind_training_protocol(report: dict[str, object]) -> None:
    resolved = report["checkpoint_resolved_config"]
    report["checkpoint_resolved_config_sha256"] = canonical_sha256(resolved)
    report["checkpoint_matched_training_protocol_sha256"] = canonical_sha256(
        _matched_training_protocol(resolved)
    )


def test_matched_comparison_reports_task_stratified_latent_gain() -> None:
    latent = _report(use_latent=True, offset=-0.5)
    no_latent = _report(use_latent=False)
    result = _compare(latent, no_latent)
    metric = result["metrics"]["prior_qtrue_dropout_epe_px"]
    assert metric["episodes"] == 6
    assert metric["task_macro_difference_latent_minus_no_latent_px"] == pytest.approx(
        -0.5
    )
    assert metric["task_stratified_bootstrap95_ci_px"] == pytest.approx([-0.5, -0.5])
    assert metric["negative_favors_latent"] is True
    assert result["primary_control_metric"] == "prior_qtrue_dropout_epe_px"
    assert result["primary_metric"] == metric
    assert result["latent_report_sha256"] == "a" * 64
    assert result["no_latent_report_sha256"] == "b" * 64
    assert result["comparison_script_sha256"] == sha256_file(
        result["comparison_script"]
    )


def test_comparison_rejects_unmatched_protocol_and_adapter() -> None:
    latent = _report(use_latent=True)
    no_latent = _report(use_latent=False)
    no_latent["split"] = "val"
    with pytest.raises(ValueError, match="split"):
        _compare(latent, no_latent)

    no_latent = _report(use_latent=False)
    no_latent["checkpoint_adapter_config"]["hidden_dim"] = 128
    with pytest.raises(ValueError, match="beyond use_predicted_latent"):
        _compare(latent, no_latent)

    no_latent = _report(use_latent=False)
    no_latent["implementation_sha256"] = "8" * 64
    with pytest.raises(ValueError, match="implementation"):
        _compare(latent, no_latent)


def test_comparison_rejects_episode_mismatch_and_wrong_direction() -> None:
    latent = _report(use_latent=True)
    no_latent = _report(use_latent=False)
    no_latent["episode_rows"] = no_latent["episode_rows"][:-1]
    with pytest.raises(ValueError, match="episode sets differ"):
        _compare(latent, no_latent)

    wrong = deepcopy(_report(use_latent=False))
    wrong["checkpoint_adapter_config"]["use_predicted_latent"] = True
    with pytest.raises(ValueError, match="disable predicted latent"):
        _compare(latent, wrong)


def test_comparison_rejects_missing_or_changed_training_protocol() -> None:
    latent = _report(use_latent=True)
    no_latent = _report(use_latent=False)
    no_latent["checkpoint_resolved_config"]["optimizer"]["lr"] = 0.01
    _rebind_training_protocol(no_latent)
    with pytest.raises(ValueError, match="training protocols differ"):
        _compare(latent, no_latent)

    latent = _report(use_latent=True)
    no_latent = _report(use_latent=False)
    for report in (latent, no_latent):
        report.pop("checkpoint_matched_training_protocol_sha256")
    with pytest.raises(ValueError, match="matched_training_protocol"):
        _compare(latent, no_latent)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_comparison_rejects_missing_or_nonfinite_primary(value) -> None:
    latent = _report(use_latent=True)
    no_latent = _report(use_latent=False)
    for report in (latent, no_latent):
        report["episode_rows"][0]["modes"]["full_writeback"][
            "prior_qtrue_dropout_epe_px"
        ] = value
    with pytest.raises(ValueError, match="primary metric|not finite"):
        _compare(latent, no_latent)


def test_comparison_rejects_missing_or_wrong_primary_family() -> None:
    latent = _report(use_latent=True)
    no_latent = _report(use_latent=False)
    for report in (latent, no_latent):
        report.pop("primary_control_metric_family")
    with pytest.raises(ValueError, match="primary_control_metric_family"):
        _compare(latent, no_latent)

    latent = _report(use_latent=True)
    no_latent = _report(use_latent=False)
    for report in (latent, no_latent):
        report["primary_control_metric_family"] = "post_update_qtrue"
    with pytest.raises(ValueError, match="primary_control_metric_family"):
        _compare(latent, no_latent)


def test_comparison_rejects_capped_or_tampered_coverage() -> None:
    latent = _report(use_latent=True)
    no_latent = _report(use_latent=False)
    for report in (latent, no_latent):
        report["max_episodes_per_source"] = 1
    with pytest.raises(ValueError, match="max_episodes_per_source=0"):
        _compare(latent, no_latent)

    latent = _report(use_latent=True)
    no_latent = _report(use_latent=False)
    for report in (latent, no_latent):
        selection = report["evaluation_episode_selection"]
        selection["selected_identity_sha256"] = "0" * 64
        report["evaluation_episode_selection_sha256"] = canonical_sha256(
            selection
        )
    with pytest.raises(ValueError, match="episode coverage"):
        _compare(latent, no_latent)


def test_episode_cap_preserves_prefix_and_manifest_is_complete() -> None:
    dataset = SimpleNamespace(
        source_name="external",
        episodes=[np.asarray([index]) for index in range(6)],
        task_name=np.asarray(["pick"] * 3 + ["button"] * 3),
        episode_id=np.asarray([str(index) for index in range(6)]),
    )
    bundle = SimpleNamespace(datasets=[dataset], provenance={"splits": {}})
    selections, contract = _select_evaluation_episodes(
        bundle, split="all", max_episodes_per_source=4
    )
    assert selections[0][1] == (0, 1, 2, 3)
    assert contract["selected_counts_by_source_task"] == {
        "external": {"button": 1, "pick": 3}
    }
    with pytest.raises(ValueError, match="non-negative"):
        _select_evaluation_episodes(
            bundle, split="all", max_episodes_per_source=-1
        )

    manifest = _evaluation_implementation_manifest()
    assert set(EVALUATION_IMPLEMENTATION_RELATIVE_PATHS) <= manifest.keys()
    assert all(len(value) == 64 for value in manifest.values())


def test_standard_matrix_profiles_freeze_dropout_and_hidden_step_protocol() -> None:
    assert _matrix_conditions("drift") == (
        ("clean", None, 0, 0),
        ("drop1", 7, 1, 0),
        ("drop3", 7, 3, 0),
        ("drop5", 7, 5, 0),
    )
    assert _matrix_conditions("step") == (
        ("clean", None, 0, 0),
        ("drop1", 1, 1, 0),
        ("drop3", 1, 3, 0),
        ("drop5", 1, 5, 0),
        ("hidden5", None, 0, 5),
    )
    with pytest.raises(ValueError, match="unknown matrix profile"):
        _matrix_conditions("other")


def test_matrix_resume_accepts_only_exact_frozen_report(tmp_path) -> None:
    path = tmp_path / "drop3.json"
    report = _report(use_latent=True)
    report["evaluator_sha256"] = "2" * 64
    path.write_text(json.dumps(report), encoding="utf-8")
    kwargs = {
        "checkpoint_sha256": report["checkpoint_sha256"],
        "evaluator_sha256": "2" * 64,
        "implementation_files": deepcopy(report["implementation_files"]),
        "implementation_sha256": report["implementation_sha256"],
        "evaluation_config_sha256": report["evaluation_config_sha256"],
        "evaluation_data_provenance_sha256": report[
            "evaluation_data_provenance_sha256"
        ],
        "evaluation_episode_selection": deepcopy(
            report["evaluation_episode_selection"]
        ),
        "max_episodes_per_source": 0,
        "split": "all",
        "condition_name": "drop3",
        "matrix_profile": "drift",
        "gain": 1.0,
        "velocity_gain": 1.0,
        "dropout_start_offset": 7,
        "dropout_length": 3,
        "hidden_step_length": 0,
    }
    assert _completed_report_matches(path, **kwargs)
    kwargs["dropout_length"] = 5
    assert not _completed_report_matches(path, **kwargs)
    kwargs["dropout_length"] = 3
    kwargs["implementation_sha256"] = "6" * 64
    assert not _completed_report_matches(path, **kwargs)
    kwargs["implementation_sha256"] = report["implementation_sha256"]

    capped_kwargs = dict(kwargs)
    capped_kwargs["max_episodes_per_source"] = 1
    assert not _completed_report_matches(path, **capped_kwargs)

    tampered = deepcopy(report)
    tampered["episode_rows"] = tampered["episode_rows"][:-1]
    path.write_text(json.dumps(tampered), encoding="utf-8")
    assert not _completed_report_matches(path, **kwargs)

    tampered = deepcopy(report)
    first = sorted(tampered["implementation_files"])[0]
    tampered["implementation_files"][first] = "8" * 64
    path.write_text(json.dumps(tampered), encoding="utf-8")
    assert not _completed_report_matches(path, **kwargs)
