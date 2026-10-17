from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.evaluate_fcpg_lewm_factorial import CELL_SPECS, DYNAMICS_CELLS, PROTOCOL
from scripts.summarize_fcpg_lewm_factorial import (
    METRICS,
    validate_cross_report_contract,
)


MODEL_SEEDS = (15801, 15802, 15803)
CANONICAL_SEED = MODEL_SEEDS[0]


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_reports(paths: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    return [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in paths
    ]


def _provenance(episodes: int) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "git_commit": _digest("git")[:40],
        "evaluator_sha256": _digest("factorial-evaluator"),
        "rollout_evaluator_sha256": _digest("rollout-evaluator"),
        "p1_runtime_sha256": _digest("p1-runtime"),
        "prefix_bundle_module_sha256": _digest("prefix-bundle-module"),
        "device": "cuda:0",
        "config_sha256": _digest("config"),
        "source_config_sha256": _digest("source-config"),
        "policy_checkpoint_sha256": _digest("policy"),
        "corrector_checkpoint_sha256": _digest("corrector"),
        "task": "pick-place-v3",
        "camera": "corner",
        "physical_episode_count": episodes,
        "model_seeds": list(MODEL_SEEDS),
        "canonical_model_seed": CANONICAL_SEED,
        "benchmark_seed": 54901,
        "environment_seed_base": 990000,
        "task_instance_indices": list(range(episodes)),
        "drop_start_decisions": [2 + episode for episode in range(episodes)],
        "drop_duration_decisions": 2,
        "horizon_raw_steps": 20,
        "occlusion_radius_px": 24,
        "occlusion_value_uint8": 0,
        "camera_rotation_axis_world": [0.0, 0.0, 1.0],
        "camera_rotation_deg": 5.0,
    }


def _canonical_rows(episodes: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in range(episodes):
        action_sha = _digest(f"episode-{episode}-actions")
        identity = {
            "reset_digest": _digest(f"episode-{episode}-reset"),
            "camera_sha256": _digest(f"episode-{episode}-camera"),
            "schedule_sha256": _digest(f"episode-{episode}-schedule"),
            "drop_onset_state_sha256": _digest(f"episode-{episode}-drop-state"),
            "pre_drop_action_values": [float(episode), 0.25, -0.5, 1.0],
        }
        for index, (cell, (grounding, _)) in enumerate(CELL_SPECS.items()):
            metric_value = float(episode + index)
            row = {
                "model_seed": CANONICAL_SEED,
                "physical_episode": episode,
                "method": cell,
                "drop_start": 2 + episode,
                **identity,
                "prefix_action_sha256": action_sha,
                "prefix_native_token_sha256": _digest(
                    f"episode-{episode}-{grounding}-native"
                ),
                "prefix_sampled_visual_sha256": _digest(
                    f"episode-{episode}-{grounding}-visual"
                ),
                "prefix_bundle_sha256": _digest(
                    f"episode-{episode}-{grounding}-bundle"
                ),
                "prefix_manifest_sha256": _digest(
                    f"episode-{episode}-{grounding}-manifest"
                ),
            }
            row.update({metric: metric_value for metric in METRICS})
            rows.append(row)
    return rows


def _predictors(seed: int) -> dict[str, dict[str, Any]]:
    return {
        grounding: {
            "seed": seed,
            "training_config_sha256": _digest(f"{grounding}-training-config"),
            "checkpoint": f"/checkpoints/{grounding}-seed{seed}.pt",
            "checkpoint_sha256": _digest(f"{grounding}-checkpoint-{seed}"),
        }
        for grounding in ("nominal", "fcpg")
    }


def _report(
    *,
    seed: int,
    episodes: int,
    provenance: dict[str, Any],
    canonical_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = copy.deepcopy(canonical_rows)
    for row in rows:
        row["model_seed"] = seed
        if seed != CANONICAL_SEED and row["method"] in DYNAMICS_CELLS:
            row["reward_sum"] = float(row["reward_sum"]) + (seed - CANONICAL_SEED)

    return {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "classification": "paired_prefix_crossed_seed_development",
        "run_role": "canonical" if seed == CANONICAL_SEED else "dynamics_only",
        "model_seed": seed,
        "canonical_model_seed": CANONICAL_SEED,
        "canonical_baseline_source": None,
        "provenance": copy.deepcopy(provenance),
        "predictor_matching": _predictors(seed),
        "prefix_bundle_manifests": {
            f"episode_{episode:03d}/{grounding}": _digest(
                f"episode-{episode}-{grounding}-manifest"
            )
            for episode in range(episodes)
            for grounding in ("nominal", "fcpg")
        },
        "episodes": episodes,
        "cell_semantics": {
            cell: {"grounding": grounding, "dropout_state": dropout_state}
            for cell, (grounding, dropout_state) in CELL_SPECS.items()
        },
        "pairing_audit": "passed",
        "rows": rows,
    }


def _write_report_set(tmp_path: Path, *, episodes: int) -> list[Path]:
    provenance = _provenance(episodes)
    canonical_rows = _canonical_rows(episodes)
    canonical = _report(
        seed=CANONICAL_SEED,
        episodes=episodes,
        provenance=provenance,
        canonical_rows=canonical_rows,
    )
    canonical_path = tmp_path / f"seed{CANONICAL_SEED}.json"
    _write_json(canonical_path, canonical)
    canonical_sha = _file_sha256(canonical_path)

    paths = [canonical_path]
    for seed in MODEL_SEEDS[1:]:
        report = _report(
            seed=seed,
            episodes=episodes,
            provenance=provenance,
            canonical_rows=canonical_rows,
        )
        report["canonical_baseline_source"] = {
            "path": str(canonical_path),
            "sha256": canonical_sha,
        }
        path = tmp_path / f"seed{seed}.json"
        _write_json(path, report)
        paths.append(path)
    return paths


def _row(report: dict[str, Any], cell: str, *, episode: int = 0) -> dict[str, Any]:
    return next(
        row
        for row in report["rows"]
        if row["physical_episode"] == episode and row["method"] == cell
    )


@pytest.mark.parametrize("episodes", [1, 2])
def test_cross_report_contract_accepts_canonical_and_two_dynamics_reports(
    tmp_path: Path,
    episodes: int,
) -> None:
    paths = _write_report_set(tmp_path, episodes=episodes)

    contract = validate_cross_report_contract(_load_reports(paths))

    assert contract["canonical_report"] == str(paths[0])
    assert contract["canonical_report_sha256"] == _file_sha256(paths[0])
    assert len(contract["prefix_bundle_sha256"]) == episodes * 2
    assert len(contract["prefix_bundle_manifests"]) == episodes * 2


TAMPER_CASES = (
    pytest.param(
        "action_hash",
        ValueError,
        r"cross-seed prefix mismatch.*prefix_action_sha256",
        id="action-hash",
    ),
    pytest.param(
        "nominal_native_token",
        ValueError,
        r"cross-seed prefix mismatch.*prefix_native_token_sha256",
        id="nominal-native-token",
    ),
    pytest.param(
        "fcpg_sampled_visual",
        ValueError,
        r"cross-seed prefix mismatch.*prefix_sampled_visual_sha256",
        id="fcpg-sampled-visual",
    ),
    pytest.param(
        "canonical_l0_field",
        ValueError,
        r"canonical L0 row changed",
        id="canonical-l0-non-seed-field",
    ),
    pytest.param(
        "config_provenance_hash",
        ValueError,
        r"report provenance mismatch",
        id="config-provenance-hash",
    ),
    pytest.param(
        "episode_identity",
        RuntimeError,
        r"differs across cells in reset_digest",
        id="episode-identity",
    ),
    pytest.param(
        "row_model_seed",
        ValueError,
        r"row model_seed mismatch",
        id="row-model-seed",
    ),
    pytest.param(
        "canonical_seed_provenance",
        ValueError,
        r"canonical model seed disagrees with provenance",
        id="canonical-seed-provenance",
    ),
    pytest.param(
        "canonical_source_path",
        ValueError,
        r"canonical baseline source path mismatch",
        id="canonical-source-path",
    ),
    pytest.param(
        "missing_seed",
        ValueError,
        r"source reports must cover configured model_seeds",
        id="missing-seed",
    ),
    pytest.param(
        "duplicate_cell",
        ValueError,
        r"duplicate factorial row",
        id="duplicate-cell",
    ),
    pytest.param(
        "nan_metric",
        ValueError,
        r"non-finite reward_sum",
        id="nan-metric",
    ),
)


@pytest.mark.parametrize(("tamper", "error_type", "message"), TAMPER_CASES)
def test_cross_report_contract_rejects_tampering(
    tmp_path: Path,
    tamper: str,
    error_type: type[Exception],
    message: str,
) -> None:
    paths = _write_report_set(tmp_path, episodes=2)
    if tamper == "missing_seed":
        paths = paths[:-1]
    else:
        dynamics_path, dynamics = _load_reports(paths)[1]
        if tamper == "action_hash":
            for row in dynamics["rows"]:
                if row["physical_episode"] == 0:
                    row["prefix_action_sha256"] = _digest("tampered-actions")
        elif tamper == "nominal_native_token":
            for cell in ("f0_l0", "f0_l1"):
                _row(dynamics, cell)["prefix_native_token_sha256"] = _digest(
                    "tampered-nominal-native"
                )
        elif tamper == "fcpg_sampled_visual":
            for cell in ("f1_l0", "f1_l1"):
                _row(dynamics, cell)["prefix_sampled_visual_sha256"] = _digest(
                    "tampered-fcpg-visual"
                )
        elif tamper == "canonical_l0_field":
            baseline = _row(dynamics, "f1_l0")
            baseline["reward_sum"] = float(baseline["reward_sum"]) + 0.5
        elif tamper == "config_provenance_hash":
            dynamics["provenance"]["config_sha256"] = _digest("tampered-config")
        elif tamper == "episode_identity":
            _row(dynamics, "f1_l1")["reset_digest"] = _digest("tampered-reset")
        elif tamper == "row_model_seed":
            _row(dynamics, "f1_l1")["model_seed"] = MODEL_SEEDS[2]
        elif tamper == "canonical_seed_provenance":
            dynamics["canonical_model_seed"] = MODEL_SEEDS[1]
        elif tamper == "canonical_source_path":
            dynamics["canonical_baseline_source"]["path"] = str(
                tmp_path / "wrong-canonical.json"
            )
        elif tamper == "duplicate_cell":
            _row(dynamics, "f1_l1")["method"] = "f0_l1"
        elif tamper == "nan_metric":
            _row(dynamics, "f1_l1")["reward_sum"] = float("nan")
        else:  # pragma: no cover - the parameter list is exhaustive.
            raise AssertionError(f"unknown tamper case: {tamper}")
        _write_json(dynamics_path, dynamics)

    with pytest.raises(error_type, match=message):
        validate_cross_report_contract(_load_reports(paths))
