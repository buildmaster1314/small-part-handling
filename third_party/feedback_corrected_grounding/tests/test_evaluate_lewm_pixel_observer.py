from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

import scripts.evaluate_lewm_pixel_observer as evaluator_script
from fcpg.lewm_pixel_observer import fixed_gain_update
from fcpg.lewm_pixel_observer_eval import (
    DEVELOPMENT_ALPHA_GRID,
    ENDPOINT_HORIZONS,
    EvaluatorOracleDataset,
    build_g0_gate,
    deployment_batch,
    evaluate_filter_batches,
    hold_last_filter,
    input_file_record,
    records_digest,
    reject_legacy_output_path,
    select_development_alpha,
    validate_protocol,
)
from fcpg.lewm_provenance import sha256_file
from scripts.evaluate_lewm_pixel_observer import (
    _assert_evaluation_state_unchanged,
    _checkpoint_source_paths,
    _expected_hash,
    _load_observer,
    _require_clean_confirmatory_git,
    _resolve_settings,
    _validate_confirmatory_checkpoint_report,
    _validate_manifest_partitions,
    build_parser,
    confirmatory_source_exposure_audit,
    qualify_development_selection,
    validate_confirmatory_coverage,
)


class _RawIndexDataset(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, object]:
        assert index == 0
        return {
            "raw_index": torch.tensor([1, 3]),
            "q_nominal": torch.zeros(2, 1, 2),
        }


def test_oracle_wrapper_uses_raw_index_and_deployment_batch_strips_targets(
    tmp_path: Path,
) -> None:
    path = tmp_path / "oracle.npz"
    q_true = np.arange(8, dtype=np.float32).reshape(4, 1, 2)
    valid = np.ones((4, 1), dtype=bool)
    np.savez(path, q_true=q_true, valid_true=valid, valid_nominal=valid)
    wrapped = EvaluatorOracleDataset(_RawIndexDataset(), path)

    sample = wrapped[0]
    torch.testing.assert_close(
        sample["q_true"], torch.from_numpy(q_true[[1, 3]])
    )
    assert sample["valid_true"].all()
    sanitized = deployment_batch(sample, "cpu")
    assert "q_true" not in sanitized
    assert "valid_true" not in sanitized
    assert "valid_nominal_eval" not in sanitized
    assert "q_nominal" in sanitized


def test_protocol_freezes_development_grid_and_confirmatory_gain() -> None:
    alphas, horizons = validate_protocol(
        phase="development",
        alphas=DEVELOPMENT_ALPHA_GRID,
        horizons=ENDPOINT_HORIZONS,
    )
    assert alphas == DEVELOPMENT_ALPHA_GRID
    assert horizons == ENDPOINT_HORIZONS
    with pytest.raises(ValueError, match="frozen alpha grid"):
        validate_protocol(
            phase="development", alphas=(0.5,), horizons=ENDPOINT_HORIZONS
        )
    assert validate_protocol(
        phase="confirmatory", alphas=(0.5,), horizons=ENDPOINT_HORIZONS
    )[0] == (0.5,)
    with pytest.raises(ValueError, match="diagnostic only"):
        validate_protocol(
            phase="confirmatory", alphas=(1.0,), horizons=ENDPOINT_HORIZONS
        )
    with pytest.raises(ValueError, match="exactly one"):
        validate_protocol(
            phase="confirmatory",
            alphas=(0.25, 0.5),
            horizons=ENDPOINT_HORIZONS,
        )


def test_development_alpha_selection_uses_only_h5_candidate_and_harm_guard() -> None:
    alphas = DEVELOPMENT_ALPHA_GRID
    comparisons = {}
    macro = {}
    candidate_epe = {0.25: 0.8, 0.5: 0.7, 0.75: 0.6}
    task_relative = {
        0.25: {"pick": 0.01, "push": 0.00},
        0.5: {"pick": 0.02, "push": -0.05},
        0.75: {"pick": 0.03, "push": -0.20},
    }
    for alpha in (0.25, 0.5, 0.75):
        label = f"lewm_vs_hold_last_same_alpha|alpha={alpha:g}|H=5"
        comparisons[label] = {
            "per_task": {
                task: {"relative_improvement": value}
                for task, value in task_relative[alpha].items()
            }
        }
        cell = f"prior=lewm|measurement=fcpg|alpha={alpha:g}|H=5"
        macro[cell] = {"mean_epe_px": candidate_epe[alpha]}

    selection = select_development_alpha(
        alphas=alphas,
        comparisons=comparisons,
        macro=macro,
        primary_horizon=5,
        max_task_relative_degradation=0.1,
    )

    assert selection["selected_alpha"] == pytest.approx(0.5)
    assert selection["oracle_consulted"] is False
    assert selection["secondary_horizons_consulted"] is False
    assert selection["candidates"][2]["eligible"] is False
    assert selection["diagnostic_endpoint_alphas"] == [0.0, 1.0]


def test_hold_last_filter_is_masked_fixed_gain_ema() -> None:
    measurement = torch.tensor(
        [[[[0.0, 0.0]], [[2.0, 0.0]], [[4.0, 0.0]], [[8.0, 0.0]]]]
    )
    valid = torch.tensor([[[True], [True], [True], [False]]])
    priors, posteriors, gains = hold_last_filter(
        measurement, valid, history_size=2, alpha=0.5
    )

    torch.testing.assert_close(priors[0, 0, 0], torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(posteriors[0, 0, 0], torch.tensor([3.0, 0.0]))
    torch.testing.assert_close(priors[0, 1, 0], torch.tensor([3.0, 0.0]))
    torch.testing.assert_close(posteriors[0, 1, 0], torch.tensor([3.0, 0.0]))
    assert float(gains[0, 0, 0, 0]) == pytest.approx(0.5)
    assert float(gains[0, 1, 0, 0]) == pytest.approx(0.0)


class _FakeObserver(torch.nn.Module):
    history_size = 2

    def __init__(self) -> None:
        super().__init__()
        self.seen_keys: list[set[str]] = []

    def filter_sequence(
        self,
        batch,
        *,
        gain,
    ):
        self.seen_keys.append(set(batch))
        assert "q_true" not in batch
        assert "valid_true" not in batch
        q_nominal = batch["q_nominal"].float()
        q_corrected = batch["q_corrected"].float()
        measurement = q_corrected - q_nominal
        _, posterior, effective_gain = hold_last_filter(
            measurement,
            batch["valid"].bool(),
            history_size=self.history_size,
            alpha=float(gain),
        )
        return SimpleNamespace(
            q_posterior_px=q_nominal[:, self.history_size :] + posterior,
            effective_gain=effective_gain,
        )


def _filter_batch() -> dict[str, object]:
    batch, frames, keypoints = 2, 7, 1
    q_nominal = torch.full((batch, frames, keypoints, 2), 2.0)
    residual = torch.arange(frames, dtype=torch.float32).view(1, frames, 1, 1)
    residual = residual.expand(batch, -1, keypoints, 2) * 0.1
    q_true = q_nominal + residual
    return {
        "pixels": torch.zeros(batch, frames, 3, 8, 8),
        "q_nominal": q_nominal,
        "q_corrected": q_true.clone(),
        "q_true": q_true,
        "valid_true": torch.ones(batch, frames, keypoints, dtype=torch.bool),
        "valid_nominal_eval": torch.ones(
            batch, frames, keypoints, dtype=torch.bool
        ),
        "valid": torch.ones(batch, frames, keypoints, dtype=torch.bool),
        "actions": torch.zeros(batch, frames - 1, 1),
        "action": torch.zeros(batch, frames - 1, 1),
        "source_index": torch.zeros(batch, dtype=torch.long),
        "task_name": ["reach-v3", "reach-v3"],
        "episode_id": torch.tensor([10, 11]),
    }


def test_filter_matrix_reports_independent_schema_and_no_oracle_key_leak() -> None:
    observer = _FakeObserver()
    report = evaluate_filter_batches(
        observer,
        [_filter_batch()],
        phase="development",
        alphas=DEVELOPMENT_ALPHA_GRID,
        horizons=ENDPOINT_HORIZONS,
        device="cpu",
        bootstrap_samples=100,
        bootstrap_seed=7,
    )

    assert report["protocol"] == "lewm_pixel_observer_g0_filter_v1"
    assert report["schedule"]["filter"]["implemented"] is True
    assert report["schedule"]["forecast"]["implemented"] is False
    assert report["gate"]["decision"] == "Development-Only"
    assert report["gate"]["oracle_part_of_gate"] is False
    assert report["sanity_checks"]["fcpg_alpha1_identity"]["passed"] is True
    assert report["sanity_checks"]["oracle_alpha1_identity"]["passed"] is True
    assert observer.seen_keys
    assert all("q_true" not in keys and "valid_true" not in keys for keys in observer.seen_keys)
    assert any(
        key.startswith("lewm_vs_hold_last_same_alpha|alpha=0.5|H=5")
        for key in report["paired_comparisons"]
    )
    assert report["bootstrap"]["overlapping_windows_treated_as_independent"] is False


def test_confirmatory_nonunit_alpha_keeps_raw_fcpg_reference() -> None:
    observer = _FakeObserver()
    report = evaluate_filter_batches(
        observer,
        [_filter_batch()],
        phase="confirmatory",
        alphas=(0.5,),
        horizons=ENDPOINT_HORIZONS,
        device="cpu",
        bootstrap_samples=100,
        bootstrap_seed=7,
    )

    assert "lewm_vs_raw_fcpg|alpha=0.5|H=5" in report["paired_comparisons"]
    assert report["sanity_checks"]["fcpg_alpha1_identity"]["passed"] is True
    assert report["sanity_checks"]["oracle_alpha1_identity"]["passed"] is True


def test_confirmatory_settings_forbid_batch_truncation() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            "unused.json",
            "--phase",
            "confirmatory",
            "--max-batches",
            "1",
        ]
    )
    config = {
        "phase": "confirmatory",
        "evaluation": {
            "alphas": [0.5],
            "horizons": list(ENDPOINT_HORIZONS),
        },
    }
    with pytest.raises(ValueError, match="protocol CLI overrides.*--max-batches"):
        _resolve_settings(args, config)


def test_observer_bound_inputs_uses_relocated_base_path(
    tmp_path: Path,
) -> None:
    observer_path = tmp_path / "observer.pt"
    original = tmp_path / "missing-original.pt"
    relocated = tmp_path / "relocated.pt"
    relocated.write_bytes(b"relocated base")
    torch.save(
        {
            "base_checkpoint": {"path": str(original)},
            "data_artifacts": [],
            "config_binding": {},
        },
        observer_path,
    )

    actual, paths = evaluator_script._observer_bound_inputs(
        observer_path,
        base_checkpoint_override=relocated,
    )

    assert actual == relocated
    assert paths == [relocated]
@pytest.mark.parametrize(
    "override",
    [
        ["--alpha", "0.75"],
        ["--bootstrap-samples", "20000"],
        ["--bootstrap-seed", "9"],
        ["--expected-observer-sha256", "0" * 64],
        ["--expected-checkpoint-sha256", "1" * 64],
        ["--batch-size", "8"],
        ["--num-workers", "2"],
        ["--output", "alternate.json"],
    ],
)
def test_confirmatory_settings_forbid_statistical_overrides(override) -> None:
    args = build_parser().parse_args(
        ["--config", "unused.json", "--phase", "confirmatory", *override]
    )
    config = {
        "phase": "confirmatory",
        "evaluation": {
            "alphas": [0.5],
            "horizons": list(ENDPOINT_HORIZONS),
            "bootstrap_samples": 10000,
        },
    }
    with pytest.raises(ValueError, match="protocol CLI overrides"):
        _resolve_settings(args, config)


def test_confirmatory_settings_reject_config_level_protocol_tampering() -> None:
    args = build_parser().parse_args(
        ["--config", "unused.json", "--phase", "confirmatory"]
    )
    evaluation = {
        "alphas": [0.5],
        "horizons": [1, 3, 5],
        "history_size": 3,
        "rollout_steps": 5,
        "frame_skip": 5,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 6144,
        "primary_horizon": 3,
        "max_task_relative_degradation": 0.1,
        "max_raw_fcpg_relative_degradation": 0.05,
    }
    with pytest.raises(ValueError, match="frozen constants"):
        _resolve_settings(
            args,
            {"phase": "confirmatory", "evaluation": evaluation},
        )


def test_manifest_partitions_must_be_pairwise_disjoint() -> None:
    source = {
        "train_episodes": ["reach-v3:1"],
        "val_episodes": ["reach-v3:2"],
        "test_episodes": ["reach-v3:1"],
        "excluded_episodes": [],
    }
    with pytest.raises(ValueError, match="partitions overlap"):
        _validate_manifest_partitions(source, "reach")


def test_hold_last_filter_ignores_invalid_nan_during_warmup() -> None:
    measurement = torch.tensor(
        [[[[3.0, 4.0]], [[float("nan"), float("nan")]], [[8.0, 9.0]]]]
    )
    valid = torch.tensor([[[True], [False], [True]]])
    priors, _, _ = hold_last_filter(
        measurement, valid, history_size=2, alpha=0.5
    )

    torch.testing.assert_close(priors[:, 0], measurement[:, 0])
    assert bool(torch.isfinite(priors).all())


def test_confirmatory_exposure_audit_requires_new_source_file_and_hash(
    tmp_path: Path,
) -> None:
    adapter_source = tmp_path / "adapter.npz"
    base_source = tmp_path / "base.npz"
    corrector_source = tmp_path / "corrector.npz"
    evaluation_source = tmp_path / "fresh.npz"
    adapter_source.write_bytes(b"adapter-train")
    base_source.write_bytes(b"base-train")
    corrector_source.write_bytes(b"corrector-train")
    evaluation_source.write_bytes(b"fresh-test")

    observer_path = tmp_path / "observer.pt"
    torch.save(
        {
            "data_artifacts": [
                {
                    "kind": "source_data",
                    "path": str(adapter_source),
                    "sha256": sha256_file(adapter_source),
                }
            ]
        },
        observer_path,
    )
    base_path = tmp_path / "base.pt"
    torch.save(
        {
            "data_config": {"sources": [{"path": str(base_source)}]},
            "resolved_config": {},
        },
        base_path,
    )
    corrector_path = tmp_path / "corrector.pt"
    torch.save(
        {
            "source_manifest": [{"data": str(corrector_source)}],
            "resolved_config": None,
        },
        corrector_path,
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    correction_array = cache / "q_corrected.npy"
    correction_array.write_bytes(b"bound-correction-array")
    metadata = {
        "schema_version": 3,
        "source_data": str(evaluation_source),
        "source_data_sha256": sha256_file(evaluation_source),
        "frame_identity_sha256": "0" * 64,
        "checkpoint": str(corrector_path),
        "checkpoint_sha256": sha256_file(corrector_path),
        "q_corrected_sha256": sha256_file(correction_array),
        "q_true_usage": "offline_metrics_only",
    }
    (cache / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    audit, paths = confirmatory_source_exposure_audit(
        observer_path=observer_path,
        base_path=base_path,
        evaluation_sources=[evaluation_source],
        correction_caches=[cache],
    )
    assert audit["passed"] is True
    assert evaluation_source.resolve() in paths

    metadata["schema_version"] = 2
    (cache / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="schema-3"):
        confirmatory_source_exposure_audit(
            observer_path=observer_path,
            base_path=base_path,
            evaluation_sources=[evaluation_source],
            correction_caches=[cache],
        )

    metadata["schema_version"] = 3
    evaluation_source.write_bytes(corrector_source.read_bytes())
    metadata["source_data_sha256"] = sha256_file(evaluation_source)
    (cache / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="exposed to a learned component"):
        confirmatory_source_exposure_audit(
            observer_path=observer_path,
            base_path=base_path,
            evaluation_sources=[evaluation_source],
            correction_caches=[cache],
        )


def test_checkpoint_source_paths_accepts_mapping_shaped_args_data(
    tmp_path: Path,
) -> None:
    direct = tmp_path / "direct.npz"
    nested = tmp_path / "nested.npz"
    direct.write_bytes(b"direct")
    nested.write_bytes(b"nested")

    paths = _checkpoint_source_paths(
        {
            "data_config": {"sources": [{"path": str(direct)}]},
            "args": {"data": {"sources": [{"path": str(nested)}]}},
        }
    )

    assert paths == {direct.resolve(), nested.resolve()}


def test_confirmatory_coverage_is_exact_and_not_single_episode() -> None:
    split = {
        "expected_sources": ["fresh"],
        "expected_tasks": [
            "pick-place-v3",
            "button-press-v3",
            "reach-v3",
            "push-v3",
        ],
        "expected_episode_counts": {"fresh": 120},
        "expected_window_counts": {"fresh": 1_200},
    }
    report = [
        {
            "name": "fresh",
            "eligible_episodes": 120,
            "eligible_windows": 1_200,
            "episodes_per_task": {
                "pick-place-v3": 30,
                "button-press-v3": 30,
                "reach-v3": 30,
                "push-v3": 30,
            },
        }
    ]
    assert validate_confirmatory_coverage(split, report)["passed"] is True

    one_task_split = dict(split, expected_tasks=["reach-v3"])
    one_task_report = [
        dict(report[0], episodes_per_task={"reach-v3": 120})
    ]
    with pytest.raises(ValueError, match="frozen four-task suite"):
        validate_confirmatory_coverage(one_task_split, one_task_report)


def test_physical_episode_audit_rejects_cross_cohort_reuse(
    tmp_path: Path,
) -> None:
    first = tmp_path / "rotation5.npz"
    second = tmp_path / "translation4.npz"
    frames = np.arange(100, dtype=np.int32)

    def write(path: Path, offset: float) -> None:
        np.savez(
            path,
            task_name=np.asarray(["pick-place-v3"] * 100),
            task_index=np.zeros(100, dtype=np.int32),
            episode_id=np.zeros(100, dtype=np.int32),
            frame_id=frames,
            obs=np.full((100, 3), offset, dtype=np.float32),
            action=np.full((100, 2), offset, dtype=np.float32),
            keypoints_xyz=np.full((100, 3, 3), offset, dtype=np.float32),
        )

    write(first, 1.0)
    write(second, 2.0)
    report = [
        {"name": "rotation5", "data": str(first)},
        {"name": "translation4", "data": str(second)},
    ]
    assert (
        evaluator_script.validate_physical_episode_independence(report)["passed"]
        is True
    )

    write(second, 1.0)
    with pytest.raises(ValueError, match="reuse physical episode"):
        evaluator_script.validate_physical_episode_independence(report)


def test_confirmatory_rejects_repacked_development_episode(
    tmp_path: Path,
) -> None:
    development = tmp_path / "development.npz"
    confirmatory = tmp_path / "confirmatory.npz"
    frames = np.arange(100, dtype=np.int32)

    def write(
        path: Path,
        *,
        episode: int,
        task_index: int,
        offset: float,
        task_name: str = "pick-place-v3",
    ) -> None:
        np.savez(
            path,
            task_name=np.asarray([task_name] * 100),
            task_index=np.full(100, task_index, dtype=np.int32),
            episode_id=np.full(100, episode, dtype=np.int32),
            frame_id=frames,
            obs=np.full((100, 3), offset, dtype=np.float32),
            action=np.full((100, 2), offset, dtype=np.float32),
            keypoints_xyz=np.full((100, 3, 3), offset, dtype=np.float32),
        )

    write(development, episode=0, task_index=0, offset=1.0)
    write(
        confirmatory,
        episode=99,
        task_index=77,
        offset=1.0,
        task_name="renamed-task-v3",
    )
    development_sources = [
        {
            "name": "pick",
            "path": str(development),
            "sha256": sha256_file(development),
        }
    ]
    confirmatory_sources = [
        {
            "name": "rotation5",
            "data": str(confirmatory),
            "data_sha256": sha256_file(confirmatory),
        }
    ]
    with pytest.raises(ValueError, match="reuses development physical content"):
        evaluator_script.validate_development_confirmatory_independence(
            development_sources,
            confirmatory_sources,
        )

    write(confirmatory, episode=99, task_index=77, offset=2.0)
    confirmatory_sources[0]["data_sha256"] = sha256_file(confirmatory)
    assert evaluator_script.validate_development_confirmatory_independence(
        development_sources,
        confirmatory_sources,
    )["passed"] is True


def test_legacy_gate_output_is_rejected_and_cli_requires_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="legacy Gate L1"):
        reject_legacy_output_path(
            tmp_path / "outputs/lewm_grounded/gate_l1/report.json", tmp_path
        )
    accepted = reject_legacy_output_path(
        tmp_path / "outputs/lewm_pixel_observer/report.json", tmp_path
    )
    assert accepted.name == "report.json"
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_fixed_gain_alpha_endpoints_are_exact() -> None:
    prior = torch.tensor([[[1.0, 2.0]]])
    measurement = torch.tensor([[[4.0, 8.0]]])
    _, posterior_zero, _ = fixed_gain_update(prior, measurement, 0.0)
    _, posterior_one, _ = fixed_gain_update(prior, measurement, 1.0)
    torch.testing.assert_close(posterior_zero, prior)
    torch.testing.assert_close(posterior_one, measurement)


def _formal_checkpoint_report() -> dict[str, object]:
    formal_contract = {
        "development_only": False,
        "optimizer_step_budget": 2_000,
        "base_grounded_lewm_frozen": True,
        "require_formal_base": True,
        "require_explicit_online_base": True,
        "require_completed_base_budget": True,
        "expected_base_step": 10_000,
        "base_target": "online_self_target",
        "supervision": "q_corrected_minus_q_nominal_only",
        "q_true_forbidden": True,
        "oracle_target_forbidden": True,
        "old_gate_l1_test_reuse_forbidden": True,
    }
    return {
        "run_kind": "formal",
        "step": 2_000,
        "training_budget": {
            "checkpoint_optimizer_step": 2_000,
            "target_optimizer_steps": 2_000,
            "final_optimizer_steps": 2_000,
            "run_completed": True,
        },
        "base_checkpoint": {
            "run_kind": "formal",
            "supervision_audit": {"evidence": "explicit"},
            "step": 10_000,
            "training_budget": {
                "target_optimizer_steps": 10_000,
                "checkpoint_optimizer_step": 10_000,
                "run_completed": True,
                "final_optimizer_steps": 10_000,
            },
            "provenance": {
                "manifest": {
                    "config": {
                        "absolute_path": str(
                            evaluator_script.FORMAL_BASE_CONFIG_PATH
                        ),
                        "sha256": sha256_file(
                            evaluator_script.FORMAL_BASE_CONFIG_PATH
                        ),
                    }
                }
            },
        },
        "config_binding": {
            "origin_path": str(evaluator_script.FORMAL_ADAPTER_CONFIG_PATH),
            "origin_sha256_at_start": sha256_file(
                evaluator_script.FORMAL_ADAPTER_CONFIG_PATH
            ),
        },
        "resolved_observer_config": {"contract": formal_contract},
    }


def test_development_selection_requires_full_formal_untruncated_matrix() -> None:
    selection = {
        "status": "candidate_selected",
        "selected_alpha": 0.5,
    }
    qualified = qualify_development_selection(
        selection,
        checkpoint_report=_formal_checkpoint_report(),
        coverage_audit={"passed": True},
        correction_derivation_audit={"passed": True},
        max_batches=None,
        git_state={"dirty": False},
    )
    assert qualified["status"] == "selected"
    assert qualified["selected_alpha"] == pytest.approx(0.5)

    truncated = qualify_development_selection(
        selection,
        checkpoint_report=_formal_checkpoint_report(),
        coverage_audit={"passed": True},
        correction_derivation_audit={"passed": True},
        max_batches=1,
        git_state={"dirty": False},
    )
    assert truncated["status"] == "ineligible"
    assert truncated["selected_alpha"] is None
    assert "batch-truncated" in truncated["freeze_eligibility"]["failures"][0]


def test_g0_requires_raw_fcpg_practical_noninferiority() -> None:
    same_alpha = {
        "bootstrap": {"difference_95_ci": [0.01, 0.02]},
        "per_task": {
            task: {"relative_improvement": 0.01}
            for task in (
                "pick-place-v3",
                "button-press-v3",
                "reach-v3",
                "push-v3",
            )
        },
    }
    raw_fcpg = {
        "point": {"relative_improvement": -0.051},
    }
    gate = build_g0_gate(
        phase="confirmatory",
        alphas=(0.5,),
        comparisons={
            "lewm_vs_hold_last_same_alpha|alpha=0.5|H=5": same_alpha,
            "lewm_vs_raw_fcpg|alpha=0.5|H=5": raw_fcpg,
        },
        sanity_checks={"integrity": {"passed": True}},
    )

    assert gate["decision"] == "No-Go"
    assert (
        gate["conditions"]["raw_fcpg_practical_noninferiority"]["passed"]
        is False
    )


def test_loaded_observer_report_preserves_completed_base_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observer_path = tmp_path / "observer.pt"
    base_path = tmp_path / "base.pt"
    observer_path.write_bytes(b"formal observer")
    base_path.write_bytes(b"formal online base")
    base_budget = {
        "target_optimizer_steps": 10_000,
        "checkpoint_optimizer_step": 10_000,
        "run_completed": True,
        "final_optimizer_steps": 10_000,
    }
    payload = {
        "checkpoint_type": "lewm_pixel_innovation_observer",
        "step": 2_000,
        "epoch": 20,
        "run_kind": "formal",
        "training_budget": {
            "target_optimizer_steps": 2_000,
            "checkpoint_optimizer_step": 2_000,
            "run_completed": True,
            "final_optimizer_steps": 2_000,
        },
        "supervision_contract": {
            "q_true_forbidden": True,
            "oracle_target_forbidden": True,
            "base_grounded_lewm_frozen": True,
        },
        "base_checkpoint": {
            "path": str(tmp_path / "relocated-original-missing.pt"),
            "sha256": sha256_file(base_path),
            "step": 10_000,
            "run_kind": "formal",
            "supervision_audit": {"evidence": "explicit"},
            "training_budget": base_budget,
            "provenance": {
                "manifest": {
                    "config": {
                        "absolute_path": str(
                            evaluator_script.FORMAL_BASE_CONFIG_PATH
                        ),
                        "sha256": sha256_file(
                            evaluator_script.FORMAL_BASE_CONFIG_PATH
                        ),
                    }
                }
            },
        },
    }
    formal_report = _formal_checkpoint_report()
    monkeypatch.setattr(
        evaluator_script,
        "load_pixel_observer_checkpoint",
        lambda *_args, **_kwargs: (
            torch.nn.Identity(),
            formal_report["resolved_observer_config"],
            payload,
        ),
    )

    _, report = _load_observer(
        observer_path=observer_path,
        base_checkpoint_override=base_path,
        expected_observer_sha256=sha256_file(observer_path),
        expected_base_sha256=sha256_file(base_path),
        device=torch.device("cpu"),
        sample_dataset=object(),
    )

    assert report["base_checkpoint"]["training_budget"] == base_budget
    assert report["base_checkpoint"]["path"] == str(base_path)
    report["config_binding"] = formal_report["config_binding"]
    _validate_confirmatory_checkpoint_report(report)


@pytest.mark.parametrize(
    ("budget_update", "checkpoint_step"),
    [
        ({"run_completed": False}, 1),
        ({"final_optimizer_steps": 2_001}, 1),
        ({"checkpoint_optimizer_step": 2}, 1),
        ({"checkpoint_optimizer_step": 1_000}, 1_000),
        ({"checkpoint_optimizer_step": 2_001}, 2_001),
    ],
)
def test_confirmatory_checkpoint_budget_fails_closed(
    budget_update: dict[str, object], checkpoint_step: int
) -> None:
    report = _formal_checkpoint_report()
    report["step"] = checkpoint_step
    report["training_budget"].update(budget_update)
    with pytest.raises(ValueError, match="completed training budget"):
        _validate_confirmatory_checkpoint_report(report)


def test_expected_hash_accepts_only_the_pinned_file(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.pt"
    artifact.write_bytes(b"pinned checkpoint")
    expected = sha256_file(artifact)
    assert _expected_hash(artifact, expected, "test artifact") == expected
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _expected_hash(artifact, "0" * 64, "test artifact")
    with pytest.raises(ValueError, match="expected SHA-256 is invalid"):
        _expected_hash(artifact, "not-a-sha", "test artifact")


def test_confirmatory_git_and_input_immutability_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean = {"head": "frozen", "dirty": False, "status": []}
    _require_clean_confirmatory_git("development", {"dirty": True})
    with pytest.raises(ValueError, match="clean git worktree"):
        _require_clean_confirmatory_git(
            "confirmatory", {"head": "frozen", "dirty": True, "status": [" M x"]}
        )

    artifact = tmp_path / "input.bin"
    artifact.write_bytes(b"before")
    records = [input_file_record(artifact)]
    digest = records_digest(records)
    monkeypatch.setattr(evaluator_script, "_git_state", lambda: clean)
    _assert_evaluation_state_unchanged(records, digest, clean)

    artifact.write_bytes(b"after")
    with pytest.raises(RuntimeError, match="input changed"):
        _assert_evaluation_state_unchanged(records, digest, clean)

    records = [input_file_record(artifact)]
    digest = records_digest(records)
    changed = {"head": "different", "dirty": False, "status": []}
    monkeypatch.setattr(evaluator_script, "_git_state", lambda: changed)
    with pytest.raises(RuntimeError, match="repository state changed"):
        _assert_evaluation_state_unchanged(records, digest, clean)


def test_canonical_correction_sink_replaces_only_replayed_scoring_frames() -> None:
    original = np.zeros((4, 2, 2), dtype=np.float32)
    dataset = SimpleNamespace(q_corrected=original)
    applications: dict[str, dict[str, object]] = {}
    sink = evaluator_script._canonical_correction_sink(
        {"rotation5": dataset}, applications
    )
    replayed = np.asarray(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        dtype=np.float32,
    )

    sink("rotation5", np.asarray([1, 3]), replayed)

    np.testing.assert_array_equal(dataset.q_corrected[[1, 3]], replayed)
    np.testing.assert_array_equal(dataset.q_corrected[[0, 2]], 0.0)
    np.testing.assert_array_equal(original, 0.0)
    assert applications["rotation5"]["scored_frames_replaced"] == 2
    with pytest.raises(ValueError, match="emitted twice"):
        sink("rotation5", np.asarray([1, 3]), replayed)


class _ConfirmatoryMainDataset(Dataset):
    def __init__(self, path: str | Path, *args, **kwargs) -> None:
        del args, kwargs
        self.path = Path(path).resolve()
        self.image_size = (224, 224)
        self.windows = [
            (episode, start) for episode in range(120) for start in range(65)
        ]
        self.episodes = [np.asarray([index]) for index in range(120)]
        self.task_name = np.repeat(
            np.asarray(
                [
                    "pick-place-v3",
                    "button-press-v3",
                    "reach-v3",
                    "push-v3",
                ]
            ),
            30,
        )
        self.episode_id = np.tile(np.arange(30), 4)
        self.q_corrected = np.zeros((120, 3, 2), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"index": torch.tensor(index)}


def _bound_development_report(
    tmp_path: Path,
    observer_sha: str,
    base_sha: str,
    corrector_sha: str,
) -> tuple[dict[str, object], Path]:
    macro = {}
    comparisons = {}
    candidate_epe = {0.25: 0.8, 0.5: 0.6, 0.75: 0.7}
    for alpha in (0.25, 0.5, 0.75):
        alpha_label = f"{alpha:g}"
        macro[
            f"prior=lewm|measurement=fcpg|alpha={alpha_label}|H=5"
        ] = {"mean_epe_px": candidate_epe[alpha]}
        comparisons[
            "lewm_vs_hold_last_same_alpha|"
            f"alpha={alpha_label}|H=5"
        ] = {
            "per_task": {
                task: {"relative_improvement": 0.01}
                for task in evaluator_script.REQUIRED_TASK_NAMES
            }
        }
    candidate = select_development_alpha(
        alphas=DEVELOPMENT_ALPHA_GRID,
        comparisons=comparisons,
        macro=macro,
    )
    checkpoint = _formal_checkpoint_report()
    checkpoint["sha256"] = observer_sha
    checkpoint["base_checkpoint"]["sha256"] = base_sha

    manifest_path = tmp_path / "development-manifest.json"
    manifest_path.write_text('{"sources": {}}\n', encoding="utf-8")
    names = ("pick", "button", "reach", "push")
    source_paths = []
    cache_paths = []
    for name in names:
        data_path = tmp_path / f"development-{name}.npz"
        data_path.write_bytes(f"development-{name}".encode("ascii"))
        source_paths.append(data_path)
        cache_paths.append(tmp_path / f"development-{name}-cache")
    config_path = tmp_path / "canonical-development.json"
    canonical_config = {
        "protocol": "lewm_pixel_observer_g0_fixed_alpha",
        "phase": "development",
        "confirmatory_design_binding": {
            "path": str(evaluator_script.FORMAL_CONFIRMATORY_DESIGN_PATH),
            "sha256": sha256_file(
                evaluator_script.FORMAL_CONFIRMATORY_DESIGN_PATH
            ),
        },
        "observer_checkpoint_sha256": observer_sha,
        "base_checkpoint_sha256": base_sha,
        "split": {"episode_manifest": str(manifest_path)},
        "sources": [
            {
                "name": name,
                "path": str(data_path),
                "correction_cache": str(cache_path),
            }
            for name, data_path, cache_path in zip(
                names, source_paths, cache_paths, strict=True
            )
        ],
    }
    config_path.write_text(json.dumps(canonical_config), encoding="utf-8")
    sources = [
        {
            "name": name,
            "data": str(data_path),
            "data_sha256": sha256_file(data_path),
            "correction_cache": str(cache_path),
            "eligible_episodes": 42,
            "eligible_windows": 2_730,
            "episodes_per_task": {task: 42},
            "corrector_checkpoint": {
                "path": "/frozen/corrector.pt",
                "sha256": corrector_sha,
            },
        }
        for name, task, data_path, cache_path in zip(
            names,
            evaluator_script.REQUIRED_TASK_NAMES,
            source_paths,
            cache_paths,
            strict=True,
        )
    ]
    replay_sources = {
        name: {
            "passed": True,
            "scored_frames": 4_200,
            "canonical_scoring_output_emitted": True,
        }
        for name in names
    }
    replay_applications = {
        name: {
            "passed": True,
            "scored_frames_replaced": 4_200,
            "source_array_frames": 6_000,
        }
        for name in names
    }
    correction_derivation_audit = {
        "passed": True,
        "protocol": "fcpg_raw_rgb_correction_derivation_v1",
        "all_scored_frames_replayed": True,
        "canonical_scoring_outputs_emitted": True,
        "sources": replay_sources,
        "canonical_scoring": {
            "passed": True,
            "required_sources": sorted(names),
            "applications": replay_applications,
        },
    }

    coverage = validate_confirmatory_coverage(
        {
            "expected_sources": list(names),
            "expected_tasks": list(evaluator_script.REQUIRED_TASK_NAMES),
            "expected_episode_counts": {name: 42 for name in names},
            "expected_window_counts": {name: 2_730 for name in names},
        },
        sources,
    )
    selected = qualify_development_selection(
        candidate,
        checkpoint_report=checkpoint,
        coverage_audit=coverage,
        correction_derivation_audit={"passed": True},
        max_batches=None,
        git_state={"dirty": False},
    )
    design_records, _design_paths = (
        evaluator_script._confirmatory_design_input_records(canonical_config)
    )
    input_records = [
        input_file_record(config_path),
        *design_records,
        *evaluator_script._protocol_bundle_records(),
        input_file_record(manifest_path),
        *(input_file_record(path) for path in source_paths),
    ]
    report = {
        "protocol": "lewm_pixel_observer_g0_report_v1",
        "phase": "development",
        "checkpoint": checkpoint,
        "sources": sources,
        "split": {
            "coverage_audit": coverage,
            "correction_derivation_audit": correction_derivation_audit,
        },
        "runtime": {"max_batches": None},
        "provenance": {
            "input_records": input_records,
            "input_digest": records_digest(input_records),
            "inputs_unchanged": True,
            "git": {"dirty": False},
            "config": input_file_record(config_path),
            "episode_manifest": input_file_record(manifest_path),
        },
        "evaluation": {
            "protocol": "lewm_pixel_observer_g0_filter_v1",
            "contract": {
                "alphas": list(DEVELOPMENT_ALPHA_GRID),
                "horizons": list(ENDPOINT_HORIZONS),
            },
            "macro": macro,
            "paired_comparisons": comparisons,
            "development_selection": selected,
        },
    }
    return report, config_path


def test_confirmatory_binding_recomputes_development_selector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer_sha = "a" * 64
    base_sha = "b" * 64
    report, development_config = _bound_development_report(
        tmp_path, observer_sha, base_sha, "c" * 64
    )
    monkeypatch.setattr(
        evaluator_script,
        "FORMAL_DEVELOPMENT_CONFIG_PATH",
        development_config,
    )
    report["evaluation"]["macro"][
        "prior=lewm|measurement=fcpg|alpha=0.5|H=5"
    ]["mean_epe_px"] = 0.9
    report_path = tmp_path / "tampered-development.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="differs from recomputation"):
        evaluator_script.validate_confirmatory_development_binding(
            {
                "observer_checkpoint_sha256": observer_sha,
                "base_checkpoint_sha256": base_sha,
                "confirmatory_design_binding": {
                    "path": str(evaluator_script.FORMAL_CONFIRMATORY_DESIGN_PATH),
                    "sha256": sha256_file(
                        evaluator_script.FORMAL_CONFIRMATORY_DESIGN_PATH
                    ),
                },
                "development_binding": {
                    "report": str(report_path),
                    "report_sha256": sha256_file(report_path),
                },
            },
            {"alphas": (0.5,)},
        )


@pytest.mark.parametrize(
    "tamper",
    ("audit_failed", "canonical_failed", "missing_source", "wrong_count"),
)
def test_confirmatory_binding_rechecks_development_canonical_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    observer_sha = "a" * 64
    base_sha = "b" * 64
    report, development_config = _bound_development_report(
        tmp_path, observer_sha, base_sha, "c" * 64
    )
    monkeypatch.setattr(
        evaluator_script,
        "FORMAL_DEVELOPMENT_CONFIG_PATH",
        development_config,
    )
    audit = report["split"]["correction_derivation_audit"]
    if tamper == "audit_failed":
        audit["passed"] = False
    elif tamper == "canonical_failed":
        audit["canonical_scoring"]["passed"] = False
    elif tamper == "missing_source":
        audit["sources"].pop("pick")
    else:
        audit["canonical_scoring"]["applications"]["pick"][
            "scored_frames_replaced"
        ] = 4_199
    report_path = tmp_path / f"development-replay-{tamper}.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="development .*scor|development correction"):
        evaluator_script.validate_confirmatory_development_binding(
            {
                "observer_checkpoint_sha256": observer_sha,
                "base_checkpoint_sha256": base_sha,
                "confirmatory_design_binding": {
                    "path": str(evaluator_script.FORMAL_CONFIRMATORY_DESIGN_PATH),
                    "sha256": sha256_file(
                        evaluator_script.FORMAL_CONFIRMATORY_DESIGN_PATH
                    ),
                },
                "development_binding": {
                    "report": str(report_path),
                    "report_sha256": sha256_file(report_path),
                },
            },
            {"alphas": (0.5,)},
        )


@pytest.mark.parametrize("missing_record_index", [0, 1, 2])
def test_confirmatory_binding_rejects_missing_preregistered_input_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_record_index: int,
) -> None:
    observer_sha = "a" * 64
    base_sha = "b" * 64
    report, development_config = _bound_development_report(
        tmp_path, observer_sha, base_sha, "c" * 64
    )
    monkeypatch.setattr(
        evaluator_script,
        "FORMAL_DEVELOPMENT_CONFIG_PATH",
        development_config,
    )
    canonical_config = json.loads(
        development_config.read_text(encoding="utf-8")
    )
    design_records, _ = evaluator_script._confirmatory_design_input_records(
        canonical_config
    )
    missing_path = str(design_records[missing_record_index]["path"])
    remaining = [
        record
        for record in report["provenance"]["input_records"]
        if str(record["path"]) != missing_path
    ]
    report["provenance"]["input_records"] = remaining
    report["provenance"]["input_digest"] = records_digest(remaining)
    report_path = tmp_path / f"missing-prereg-{missing_record_index}.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="lacks the frozen confirmatory design/tooling record",
    ):
        evaluator_script.validate_confirmatory_development_binding(
            {
                "observer_checkpoint_sha256": observer_sha,
                "base_checkpoint_sha256": base_sha,
                "confirmatory_design_binding": canonical_config[
                    "confirmatory_design_binding"
                ],
                "development_binding": {
                    "report": str(report_path),
                    "report_sha256": sha256_file(report_path),
                },
            },
            {"alphas": (0.5,)},
        )


@pytest.mark.parametrize("tamper", ["missing", "replacement_hash"])
def test_confirmatory_binding_rejects_changed_cross_stage_protocol_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    observer_sha = "a" * 64
    base_sha = "b" * 64
    report, development_config = _bound_development_report(
        tmp_path, observer_sha, base_sha, "c" * 64
    )
    monkeypatch.setattr(
        evaluator_script,
        "FORMAL_DEVELOPMENT_CONFIG_PATH",
        development_config,
    )
    canonical_config = json.loads(
        development_config.read_text(encoding="utf-8")
    )
    bundle_record = evaluator_script._protocol_bundle_records()[0]
    records = list(report["provenance"]["input_records"])
    index = next(
        index
        for index, record in enumerate(records)
        if record["path"] == bundle_record["path"]
    )
    if tamper == "missing":
        records.pop(index)
    else:
        records[index] = {**records[index], "sha256": "0" * 64}
    report["provenance"]["input_records"] = records
    report["provenance"]["input_digest"] = records_digest(records)
    report_path = tmp_path / f"bundle-{tamper}.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="cross-stage protocol bundle record"):
        evaluator_script.validate_confirmatory_development_binding(
            {
                "observer_checkpoint_sha256": observer_sha,
                "base_checkpoint_sha256": base_sha,
                "confirmatory_design_binding": canonical_config[
                    "confirmatory_design_binding"
                ],
                "development_binding": {
                    "report": str(report_path),
                    "report_sha256": sha256_file(report_path),
                },
            },
            {"alphas": (0.5,)},
        )


def test_preload_dependency_graph_change_fails_before_dataset_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"frozen")
    calls = 0
    dataset_loaded = False

    def discover(_args, _config_path):
        nonlocal calls
        calls += 1
        return {
            "settings": {"phase": "development"},
            "input_paths": (artifact,),
            "graph": {"generation": calls},
        }

    class ForbiddenDataset:
        def __init__(self, *_args, **_kwargs):
            nonlocal dataset_loaded
            dataset_loaded = True

    monkeypatch.setattr(
        evaluator_script, "_discover_preload_dependencies", discover
    )
    monkeypatch.setattr(
        evaluator_script,
        "_git_state",
        lambda: {"head": "frozen", "dirty": False, "status": []},
    )
    monkeypatch.setattr(evaluator_script, "LeWMSequenceDataset", ForbiddenDataset)

    with pytest.raises(RuntimeError, match="dependency graph changed"):
        evaluator_script.main(["--config", str(config), "--device", "cpu"])
    assert calls == 2
    assert dataset_loaded is False


@pytest.mark.parametrize(
    "failed_audit",
    [
        None,
        "correction_derivation_audit",
        "camera_intervention_audit",
        "source_exposure_audit",
    ],
)
def test_confirmatory_main_returns_nogo_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_audit: str | None,
) -> None:
    rotation_source = tmp_path / "rotation5-test.npz"
    translation_source = tmp_path / "translation4-test.npz"
    observer = tmp_path / "observer.pt"
    base = tmp_path / "base.pt"
    cache = tmp_path / "cache"
    manifest = tmp_path / "manifest.json"
    config_path = tmp_path / "confirmatory.json"
    report_path = tmp_path / "pixel-observer-report.json"
    development_path = tmp_path / "development-report.json"
    rotation_source.write_bytes(b"fresh-rotation5-source")
    translation_source.write_bytes(b"fresh-translation4-source")
    observer.write_bytes(b"formal-observer")
    base.write_bytes(b"formal-base")
    manifest.write_text('{"sources": {}}\n', encoding="utf-8")
    observer_sha = sha256_file(observer)
    base_sha = sha256_file(base)
    corrector_sha = "c" * 64
    development_report, development_config = _bound_development_report(
        tmp_path,
        observer_sha,
        base_sha,
        corrector_sha,
    )
    monkeypatch.setattr(
        evaluator_script,
        "FORMAL_DEVELOPMENT_CONFIG_PATH",
        development_config,
    )
    development_path.write_text(
        json.dumps(development_report), encoding="utf-8"
    )
    config = {
        "protocol": "lewm_pixel_observer_g0_fixed_alpha",
        "phase": "confirmatory",
        "observer_checkpoint": str(observer),
        "observer_checkpoint_sha256": observer_sha,
        "base_checkpoint": str(base),
        "base_checkpoint_sha256": base_sha,
        "confirmatory_design_binding": {
            "path": str(evaluator_script.FORMAL_CONFIRMATORY_DESIGN_PATH),
            "sha256": sha256_file(
                evaluator_script.FORMAL_CONFIRMATORY_DESIGN_PATH
            ),
        },
        "development_binding": {
            "report": str(development_path),
            "report_sha256": sha256_file(development_path),
        },
        "evaluation": {
            "alphas": [0.5],
            "horizons": [1, 3, 5],
            "history_size": 3,
            "rollout_steps": 5,
            "frame_skip": 5,
            "batch_size": 4,
            "num_workers": 0,
            "bootstrap_samples": 10_000,
            "bootstrap_seed": 6144,
            "primary_horizon": 5,
            "max_task_relative_degradation": 0.1,
            "max_raw_fcpg_relative_degradation": 0.05,
            "partition": "test",
        },
        "split": {
            "episode_manifest": str(manifest),
            "fresh_non_exposure_audited": True,
            "expected_sources": ["rotation5", "translation4"],
            "expected_tasks": [
                "pick-place-v3",
                "button-press-v3",
                "reach-v3",
                "push-v3",
            ],
            "expected_episode_counts": {
                "rotation5": 120,
                "translation4": 120,
            },
            "expected_window_counts": {
                "rotation5": 7_800,
                "translation4": 7_800,
            },
            "disclosure": "fresh held-out test episodes",
        },
        "sources": [
            {
                "name": "rotation5",
                "path": str(rotation_source),
                "correction_cache": str(cache),
            },
            {
                "name": "translation4",
                "path": str(translation_source),
                "correction_cache": str(cache),
            }
        ],
        "output": {"report": str(report_path)},
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    checkpoint_report = _formal_checkpoint_report()
    monkeypatch.setattr(
        evaluator_script, "LeWMSequenceDataset", _ConfirmatoryMainDataset
    )
    monkeypatch.setattr(
        evaluator_script,
        "select_manifest_episodes",
        lambda dataset, **_kwargs: evaluator_script.Subset(
            dataset, range(7_800)
        ),
    )
    monkeypatch.setattr(
        evaluator_script,
        "EvaluatorOracleDataset",
        lambda dataset, _path: dataset,
    )
    monkeypatch.setattr(evaluator_script, "_correction_input_paths", lambda _path: [])
    monkeypatch.setattr(
        evaluator_script,
        "_correction_checkpoint_record",
        lambda _path: {
            "path": "/frozen/corrector.pt",
            "sha256": corrector_sha,
        },
    )
    monkeypatch.setattr(
        evaluator_script, "_observer_bound_inputs", lambda _path, **_kwargs: (base, [])
    )
    monkeypatch.setattr(
        evaluator_script,
        "confirmatory_source_exposure_audit",
        lambda **_kwargs: (
            {"passed": failed_audit != "source_exposure_audit"},
            [],
        ),
    )
    monkeypatch.setattr(
        evaluator_script,
        "validate_physical_episode_independence",
        lambda _sources: {
            "passed": True,
            "cross_source_overlap_count": 0,
        },
    )
    monkeypatch.setattr(
        evaluator_script,
        "validate_development_confirmatory_independence",
        lambda *_args: {"passed": True, "cross_split_overlap_count": 0},
    )
    monkeypatch.setattr(
        evaluator_script,
        "validate_confirmatory_camera_interventions",
        lambda *_args, **_kwargs: (
            {"passed": failed_audit != "camera_intervention_audit"},
            [],
        ),
    )
    def correction_derivation(jobs, *, canonical_sink, **_kwargs):
        reports = {}
        for job in jobs:
            name = str(job["name"])
            indices = np.asarray(job["frame_indices"], dtype=np.int64)
            canonical_sink(
                name,
                indices,
                np.full((len(indices), 3, 2), 7.0, dtype=np.float32),
            )
            reports[name] = {
                "scored_frames": 12_000,
                "canonical_scoring_output_emitted": True,
            }
        return (
            {
                "passed": failed_audit != "correction_derivation_audit",
                "canonical_scoring_outputs_emitted": True,
                "sources": reports,
            },
            [],
        )

    monkeypatch.setattr(
        evaluator_script,
        "validate_fcpg_correction_derivations",
        correction_derivation,
    )
    monkeypatch.setattr(
        evaluator_script,
        "_git_state",
        lambda: {"head": "frozen-head", "dirty": False, "status": []},
    )
    monkeypatch.setattr(
        evaluator_script,
        "_load_observer",
        lambda **_kwargs: (torch.nn.Identity(), checkpoint_report),
    )
    monkeypatch.setattr(
        evaluator_script,
        "evaluate_filter_batches",
        lambda *_args, **_kwargs: {
            "protocol": "fake-confirmatory-filter",
            "schema_version": 1,
            "gate": {
                "passed": failed_audit is not None,
                "decision": "Go" if failed_audit is not None else "No-Go",
                "conditions": {},
            },
        },
    )

    assert evaluator_script.main(
        ["--config", str(config_path), "--device", "cpu"]
    ) == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["split"]["confirmatory_eligible"] is (failed_audit is None)
    assert report["split"]["development_binding_audit"]["passed"] is True
    assert report["split"]["physical_independence_audit"]["passed"] is True
    assert (
        report["split"]["correction_derivation_audit"]["canonical_scoring"][
            "passed"
        ]
        is True
    )
    if failed_audit is not None:
        assert (
            report["split"]["confirmatory_eligibility_conditions"][
                failed_audit
            ]
            is False
        )
    assert report["gate"]["decision"] == "No-Go"
    assert report["gate"]["passed"] is False
    assert report["gate"]["scientific_effect_passed"] is (
        failed_audit is not None
    )
    assert (
        report["gate"]["conditions"]["confirmatory_provenance_conjunction"][
            "passed"
        ]
        is (failed_audit is None)
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evaluator_script.main(["--config", str(config_path), "--device", "cpu"])
