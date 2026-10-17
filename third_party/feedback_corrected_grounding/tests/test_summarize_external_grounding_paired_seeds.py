from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.summarize_external_grounding_paired_seeds import (
    load_paired_cells,
    policy_seed_episode_bootstrap,
    summarize_cells,
)


FIELDNAMES = [
    "task",
    "task_index",
    "env_seed",
    "success",
    "actual_camera_translation_x_cm",
    "actual_camera_translation_y_cm",
    "actual_camera_translation_z_cm",
    "actual_camera_rotation_axis_x",
    "actual_camera_rotation_axis_y",
    "actual_camera_rotation_axis_z",
    "actual_camera_rotation_deg",
]


def _write_rollout(path: Path, task: str, success: list[int], *, duplicate: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for index, value in enumerate(success):
            row = {
                "task": f"{task}-v3",
                "task_index": index,
                "env_seed": 50000 + index,
                "success": value,
                "actual_camera_translation_x_cm": 0.0,
                "actual_camera_translation_y_cm": 0.0,
                "actual_camera_translation_z_cm": 0.0,
                "actual_camera_rotation_axis_x": 0.0,
                "actual_camera_rotation_axis_y": 0.0,
                "actual_camera_rotation_axis_z": 1.0,
                "actual_camera_rotation_deg": 5.0,
            }
            writer.writerow(row)
            if duplicate and index == 0:
                writer.writerow(row)


def _matrix_fixture(tmp_path: Path, *, duplicate: bool = False) -> Path:
    root = tmp_path / "matrix"
    jobs = []
    rows = []
    for seed in (0, 1):
        for task in ("pick", "button"):
            for method in ("nominal", "native", "prototype"):
                output_dir = root / f"seed{seed}" / task / "rot5" / method
                if task == "pick":
                    success = [1, 1] if method == "prototype" else [0, 0]
                else:
                    success = [0, 0] if method == "prototype" else [1, 1]
                _write_rollout(
                    output_dir / f"{method}_rollout.csv",
                    task,
                    success,
                    duplicate=duplicate and seed == 0 and task == "pick" and method == "native",
                )
                jobs.append(
                    {
                        "checkpoint_label": f"seed{seed}",
                        "checkpoint": f"checkpoint_{seed}.pt",
                        "task_label": task,
                        "task_name": f"{task}-v3",
                        "severity": 5.0,
                        "method": method,
                        "output_dir": str(output_dir),
                        "perturbation": "rotation",
                    }
                )
                rows.append(
                    {
                        "checkpoint": f"seed{seed}",
                        "task": task,
                        "perturbation": "rotation",
                        "severity": 5.0,
                        "method": method,
                        "success_rate": sum(success) / len(success),
                        "corrected_pixel_error_mean": 1.0,
                    }
                )
    root.mkdir(parents=True, exist_ok=True)
    (root / "matrix_manifest.json").write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "label": f"seed{seed}",
                        "checkpoint": f"checkpoint_{seed}.pt",
                        "source_mode": "fcpg",
                    }
                    for seed in (0, 1)
                ],
                "methods": ["nominal", "native", "prototype"],
                "tasks": [["pick", "pick-v3"], ["button", "button-v3"]],
                "severities": [5.0],
                "settings": {"perturbation": "rotation", "episodes": 2},
                "jobs": jobs,
            }
        ),
        encoding="utf-8",
    )
    (root / "matrix_summary.json").write_text(
        json.dumps({"rows": rows, "auc": []}), encoding="utf-8"
    )
    return root


def test_multi_seed_pairing_and_fixed_equal_task_macro(tmp_path: Path) -> None:
    root = _matrix_fixture(tmp_path)
    matrix_root, cells = load_paired_cells(
        root / "matrix_summary.json", baseline="native", candidate="prototype"
    )
    assert matrix_root == root.resolve()
    summary = summarize_cells(
        cells,
        baseline="native",
        candidate="prototype",
        bootstrap_samples=2_000,
        bootstrap_seed=7,
    )

    pick = next(item for item in summary["task_cells"] if item["task"] == "pick")
    button = next(item for item in summary["task_cells"] if item["task"] == "button")
    assert pick["pooled_episode"]["episodes"] == 4
    assert pick["policy_seed_macro"]["paired_success_difference"] == pytest.approx(1.0)
    assert pick["policy_seed_macro"]["hierarchical_bootstrap_95_ci"] == pytest.approx(
        [1.0, 1.0]
    )
    assert button["policy_seed_macro"]["paired_success_difference"] == pytest.approx(-1.0)
    macro = summary["task_macro"][0]
    assert macro["tasks"] == ["button", "pick"]
    assert macro["equal_task_equal_seed_macro_paired_success_difference"] == pytest.approx(0.0)
    assert macro["hierarchical_bootstrap_95_ci"] == pytest.approx([0.0, 0.0])


def test_crossed_bootstrap_is_reproducible_and_uses_shared_episode_draws() -> None:
    shared = np.concatenate([np.zeros(10), np.ones(10)])
    values = {
        0: {"pick": shared},
        1: {"pick": shared},
        2: {"pick": shared},
    }
    first = policy_seed_episode_bootstrap(values, tasks=("pick",), samples=20_000, seed=11)
    second = policy_seed_episode_bootstrap(values, tasks=("pick",), samples=20_000, seed=11)
    assert first == second
    # A nested-within-seed bootstrap narrows this to roughly [.37, .63].
    assert first[0] <= 0.3
    assert first[1] == pytest.approx(0.7)


def test_crossed_bootstrap_rejects_unequal_shared_episode_counts() -> None:
    values = {
        0: {"pick": np.asarray([0.0, 1.0])},
        1: {"pick": np.asarray([1.0, 1.0, 1.0])},
    }
    with pytest.raises(ValueError, match="shared episode count mismatch"):
        policy_seed_episode_bootstrap(values, tasks=("pick",), samples=100, seed=11)


def test_pair_key_mismatch_is_rejected(tmp_path: Path) -> None:
    root = _matrix_fixture(tmp_path)
    candidate_path = root / "seed0" / "pick" / "rot5" / "prototype" / "prototype_rollout.csv"
    text = candidate_path.read_text(encoding="utf-8").replace("50001", "59999")
    candidate_path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="pair-key mismatch"):
        load_paired_cells(root, baseline="native", candidate="prototype")


def test_duplicate_episode_is_rejected(tmp_path: Path) -> None:
    root = _matrix_fixture(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="duplicate paired episode"):
        load_paired_cells(root, baseline="native", candidate="prototype")


def test_matrix_summary_success_must_match_raw_csv(tmp_path: Path) -> None:
    root = _matrix_fixture(tmp_path)
    summary_path = root / "matrix_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["rows"][0]["success_rate"] = 0.5
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="CSV/summary success mismatch"):
        load_paired_cells(root, baseline="native", candidate="prototype")


def test_manifest_cartesian_job_and_summary_completeness_are_required(tmp_path: Path) -> None:
    root = _matrix_fixture(tmp_path)
    manifest_path = root / "matrix_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    removed = manifest["jobs"].pop()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="declared Cartesian design"):
        load_paired_cells(root, baseline="native", candidate="prototype")

    root = _matrix_fixture(tmp_path / "summary")
    summary_path = root / "matrix_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["rows"].pop()
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="summary rows do not match"):
        load_paired_cells(root, baseline="native", candidate="prototype")


def test_manifest_episode_count_is_enforced(tmp_path: Path) -> None:
    root = _matrix_fixture(tmp_path)
    manifest_path = root / "matrix_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["settings"]["episodes"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="episode count mismatch"):
        load_paired_cells(root, baseline="native", candidate="prototype")


def test_unselected_manifest_method_csv_is_also_validated(tmp_path: Path) -> None:
    root = _matrix_fixture(tmp_path)
    path = root / "seed0" / "pick" / "rot5" / "nominal" / "nominal_rollout.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="episode count mismatch"):
        load_paired_cells(root, baseline="native", candidate="prototype")


def test_cross_seed_episode_keys_must_match(tmp_path: Path) -> None:
    root = _matrix_fixture(tmp_path)
    for method in ("native", "prototype"):
        path = root / "seed1" / "pick" / "rot5" / method / f"{method}_rollout.csv"
        path.write_text(
            path.read_text(encoding="utf-8").replace("50001", "59999"),
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match="cross-seed episode-key mismatch"):
        load_paired_cells(root, baseline="native", candidate="prototype")
