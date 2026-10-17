from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fcpg.lewm_goal_disjointness_audit import (
    AUDIT_PROTOCOL,
    PROOF_SCOPE,
    GoalDisjointnessAuditError,
    canonical_json_bytes,
    compare_task_goal_signatures,
    episode_goal_signatures,
    run_goal_disjointness_audit,
    sha256_file,
    write_atomic_json,
)


def _groups(values: list[list[float]], *, label: str) -> dict[bytes, dict[str, object]]:
    array = np.asarray(values, dtype=np.float32)
    groups, _summary = episode_goal_signatures(
        array,
        [f"episode-{index}" for index in range(len(values))],
        label=label,
    )
    return groups


def _prefix_config(source: Path, *, source_sha256: str) -> dict[str, object]:
    return {
        "protocol": AUDIT_PROTOCOL,
        "proof_scope": PROOF_SCOPE,
        "upgrade_g0_eligibility": False,
        "metaworld": {
            "version": "test-version",
            "sawyer_source_size_bytes": source.stat().st_size,
            "sawyer_source_sha256": source_sha256,
        },
    }


def test_detects_exact_overlap_within_task() -> None:
    shared = [0.125, -0.25, 0.5]
    report = compare_task_goal_signatures(
        {"pick-place-v3": _groups([shared, [1.0, 2.0, 3.0]], label="old")},
        {"pick-place-v3": _groups([shared, [4.0, 5.0, 6.0]], label="new")},
    )

    task = report["task_reports"]["pick-place-v3"]
    assert report["task_scoped_exact_disjoint"] is False
    assert report["total_exact_intersection_count"] == 1
    assert task["exact_intersection_count"] == 1
    assert task["minimum_l2_distance"] == 0.0
    assert task["exact_intersections"][0]["values"] == shared


def test_rejects_goal_that_changes_within_episode() -> None:
    goals = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    with pytest.raises(GoalDisjointnessAuditError, match="not exactly constant"):
        episode_goal_signatures(goals, ["same", "same"], label="synthetic")


def test_exact_comparison_is_task_scoped() -> None:
    shared_across_different_tasks = [0.1, 0.2, 0.3]
    old = {
        "button-press-v3": _groups([[9.0, 9.0, 9.0]], label="old-button"),
        "pick-place-v3": _groups([shared_across_different_tasks], label="old-pick"),
    }
    new = {
        "button-press-v3": _groups([shared_across_different_tasks], label="new-button"),
        "pick-place-v3": _groups([[8.0, 8.0, 8.0]], label="new-pick"),
    }

    report = compare_task_goal_signatures(old, new)

    assert report["task_scoped_exact_disjoint"] is True
    assert report["total_exact_intersection_count"] == 0


def test_run_fails_closed_on_sawyer_source_sha_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "sawyer_xyz_env.py"
    source.write_text("current source\n", encoding="utf-8")
    config = _prefix_config(source, source_sha256="0" * 64)

    with pytest.raises(GoalDisjointnessAuditError, match="sawyer source SHA-256 mismatch"):
        run_goal_disjointness_audit(
            config,
            repo_root=tmp_path,
            metaworld_version="test-version",
            sawyer_source_path=source,
        )


def test_run_fails_closed_on_bound_manifest_sha_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "sawyer_xyz_env.py"
    source.write_text("current source\n", encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text("{}\n", encoding="utf-8")
    config = _prefix_config(source, source_sha256=sha256_file(source))
    config["gate_l1_split_manifest"] = {
        "path": "split.json",
        "size_bytes": split.stat().st_size,
        "sha256": "f" * 64,
    }

    with pytest.raises(GoalDisjointnessAuditError, match="split manifest SHA-256 mismatch"):
        run_goal_disjointness_audit(
            config,
            repo_root=tmp_path,
            metaworld_version="test-version",
            sawyer_source_path=source,
        )


def test_atomic_json_is_canonical_and_never_replaces(tmp_path: Path) -> None:
    destination = tmp_path / "audit.json"
    payload = {"proof_scope": PROOF_SCOPE, "full_bundle_formal_verified": False}

    write_atomic_json(destination, payload)

    assert destination.read_bytes() == canonical_json_bytes(payload) + b"\n"
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == sha256_file(destination)
    with pytest.raises(FileExistsError):
        write_atomic_json(destination, {"replacement": True})
    assert json.loads(destination.read_text(encoding="utf-8")) == payload
