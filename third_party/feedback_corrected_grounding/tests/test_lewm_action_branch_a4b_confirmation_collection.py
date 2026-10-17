from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest

import fcpg.lewm_action_branch_a4b_confirmation_collection as collection_module
from fcpg.lewm_a4_storage import ACTION_DIM, LATENT_DIM, STATE_SLOT_COUNT
from fcpg.lewm_a4b_task_identity import (
    LEGACY_IDENTITY_SCOPES,
    ResetPhysicalIdentity,
)
from fcpg.lewm_action_branch_a4 import (
    BRANCH_COUNT,
    HISTORY_STEPS,
    HORIZON,
    assign_a4_splits,
)
from fcpg.lewm_action_branch_a4_collection import ProposalRow
from fcpg.lewm_action_branch_a4b import (
    BENCHMARK_SEED,
    CONFIRMATION_TRAINING_SEEDS,
    POOL_FINAL,
    POOL_PREFLIGHT,
    TARGET_ELIGIBLE_CANDIDATES_PER_TASK,
    TASKS,
    candidate_episode_uid,
)
from fcpg.lewm_action_branch_a4b_confirmation_collection import (
    A4bCandidateEpisode,
    a4b_artifact_sha256,
    build_a4b_collection_result,
    collect_a4b_final_pool,
    collect_a4b_preflight_pool,
    load_a4b_preflight_receipt,
    publish_a4b_final_bundle_exclusive,
    save_a4b_result_exclusive,
    validate_a4b_collection_result,
)
from fcpg.lewm_action_branch_a4b_confirmation_metrics import (
    validate_a4b_result_cache_path,
)


ROOT = Path(__file__).resolve().parents[1]


def _json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _contracts() -> tuple[dict, dict, dict]:
    return (
        _json("config/lewm/eval/action_branch_a4b_fresh_confirmation.json"),
        _json("config/lewm/eval/action_branch_active_control_a4.json"),
        _json(
            "config/lewm/splits/"
            "action_branch_a4b_fresh_confirmation_seed31401.json"
        ),
    )


def _collector(
    eligible_rank: Callable[[str, int], bool],
    *,
    forced_digest: tuple[str, int, str] | None = None,
):
    def collect(**kwargs):
        task = kwargs["task"]
        task_index = int(kwargs["task_class_index"])
        raw_index = int(kwargs["raw_task_index"])
        rank = int(kwargs["candidate_rank"])
        environment_seed = int(kwargs["environment_seed"])
        pool = str(kwargs["pool"])
        uid = candidate_episode_uid(
            pool=pool,
            task=task,
            rank=rank,
            raw_task_index=raw_index,
            environment_seed=environment_seed,
        )
        is_eligible = bool(eligible_rank(task, rank))
        proposals = tuple(
            ProposalRow(
                proposal_id=f"{uid}:proposal-{3 + proposal_index * 5:03d}",
                task_class_index=task_index,
                task_name=task,
                task_instance_index=raw_index,
                source_episode_id=uid,
                environment_seed=environment_seed,
                proposal_raw_step=3 + proposal_index * 5,
                split="test",
                inputs={},
                eligible=is_eligible and proposal_index == 0,
                rejection_reasons=(
                    () if is_eligible and proposal_index == 0 else ("inactive_stage",)
                ),
            )
            for proposal_index in range(12)
        )
        digest = hashlib.sha256(f"{pool}:{task}:{rank}".encode()).hexdigest()
        if forced_digest == (task, rank, pool):
            digest = "f" * 64
        identity = ResetPhysicalIdentity(
            task_name=task,
            raw_task_index=raw_index,
            benchmark_seed=BENCHMARK_SEED,
            environment_seed=environment_seed,
            camera_name="corner",
            digest=digest,
        )
        bundles = (
            (SimpleNamespace(proposal=proposals[0]),)
            if is_eligible and kwargs["materialize_branch_bundles"]
            else ()
        )
        return A4bCandidateEpisode(proposals, bundles, identity)

    return collect


def _legacy_inventory(source_contract: dict) -> dict[str, list[dict]]:
    scopes = {
        "train": "a4_v2_formal_train",
        "development": "a4_v2_formal_development",
        "test": "a4_v2_attempted_test",
    }
    result = {scope: [] for scope in scopes.values()}
    assignments = assign_a4_splits(TASKS, source_contract)
    for task in TASKS:
        for raw_index in range(48):
            scope = scopes[assignments[(task, raw_index)]]
            result[scope].append(
                {
                    "task_name": task,
                    "raw_task_index": raw_index,
                    "digest": hashlib.sha256(
                        f"legacy:{task}:{raw_index}".encode()
                    ).hexdigest(),
                }
            )
    return result


def _write_preflight_receipt(
    tmp_path: Path,
    contract: dict,
    source: dict,
    manifest: dict,
    *,
    forced_digest: tuple[str, int, str] | None = None,
    name: str = "preflight.json",
) -> Path:
    preflight = collect_a4b_preflight_pool(
        contract,
        source,
        manifest,
        legacy_physical_identities=set(),
        runtime_factory=lambda *args: None,
        episode_collector=_collector(
            lambda _task, _rank: False, forced_digest=forced_digest
        ),
    )
    result = build_a4b_collection_result(
        preflight, legacy_reset_identities=_legacy_inventory(source)
    )
    return save_a4b_result_exclusive(tmp_path / name, result)


def _final_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    contract: dict,
    source: dict,
    manifest: dict,
    *,
    forced_digest: tuple[str, int, str] | None = None,
) -> tuple[dict[str, Path], list[Path]]:
    receipt = _write_preflight_receipt(
        tmp_path,
        contract,
        source,
        manifest,
        forced_digest=forced_digest,
    )
    development_go = tmp_path / "development-go.json"
    development_go.write_text("{}\n", encoding="utf-8")
    formal_cache = tmp_path / "formal-train-cache.npz"
    formal_cache.write_bytes(b"formal-cache")
    validated_paths: list[Path] = []

    def validate_authorization(
        summary: object,
        source_contract: object,
        *,
        formal_train_cache_path: str | Path,
    ) -> SimpleNamespace:
        assert summary == {}
        assert source_contract is source
        validated = Path(formal_train_cache_path).resolve()
        validated_paths.append(validated)
        return SimpleNamespace(formal_train_cache_path=validated)

    monkeypatch.setattr(
        collection_module,
        "validate_a4_development_go_summary",
        validate_authorization,
    )
    return (
        {
            "preflight_receipt_path": receipt,
            "development_go_summary_path": development_go,
            "formal_train_cache_path": formal_cache,
        },
        validated_paths,
    )


def _artifact_pair() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    ledger = {
        "proposal_id": np.asarray(["p0"]),
        "task_class_index": np.asarray([0], dtype=np.int64),
        "task_name": np.asarray([TASKS[0]]),
        "task_instance_index": np.asarray([0], dtype=np.int64),
        "source_episode_id": np.asarray(["e0"]),
        "environment_seed": np.asarray([720_000], dtype=np.int64),
        "proposal_raw_step": np.asarray([3], dtype=np.int64),
        "split": np.asarray(["test"]),
        "proposal": np.ones(1, dtype=bool),
        "complete_history": np.ones(1, dtype=bool),
        "success": np.zeros(1, dtype=bool),
        "ever_success_before": np.zeros(1, dtype=bool),
        "active_stage": np.ones(1, dtype=bool),
        "expert_xyz_norm": np.asarray([0.2], dtype=np.float32),
        "recent_action_rms": np.asarray([0.1], dtype=np.float32),
        "workspace_and_joint_margin": np.asarray([0.1], dtype=np.float32),
        "unexpected_collision": np.zeros(1, dtype=bool),
        "remaining_raw_steps": np.asarray([100], dtype=np.int64),
        "eligible": np.ones(1, dtype=bool),
        "rejection_reasons": np.asarray([""]),
        "accepted_bundle_index": np.asarray([0], dtype=np.int64),
    }
    cache = {
        "proposal_id": np.asarray(["p0"]),
        "task_class_index": np.asarray([0], dtype=np.int64),
        "task_name": np.asarray([TASKS[0]]),
        "task_instance_index": np.asarray([0], dtype=np.int64),
        "source_episode_id": np.asarray(["e0"]),
        "split": np.asarray(["test"]),
        "history_state": np.zeros(
            (1, HISTORY_STEPS, STATE_SLOT_COUNT, LATENT_DIM), dtype=np.float32
        ),
        "history_action": np.zeros(
            (1, HISTORY_STEPS - 1, ACTION_DIM), dtype=np.float32
        ),
        "center_action": np.zeros((1, HORIZON, ACTION_DIM), dtype=np.float32),
        "future_action": np.zeros(
            (1, BRANCH_COUNT, HORIZON, ACTION_DIM), dtype=np.float32
        ),
        "target_state": np.zeros(
            (1, BRANCH_COUNT, HORIZON, STATE_SLOT_COUNT, LATENT_DIM),
            dtype=np.float32,
        ),
        "branch_semantic_index": np.arange(BRANCH_COUNT, dtype=np.int64)[
            None, :
        ],
    }
    return ledger, cache


def _successful_final(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> collection_module.A4bCandidatePoolCollection:
    contract, source, manifest = _contracts()
    final_access, _validated_paths = _final_access(
        monkeypatch, tmp_path, contract, source, manifest
    )
    ledger, cache = _artifact_pair()
    monkeypatch.setattr(
        collection_module,
        "_stack_final_artifacts",
        lambda proposals, bundles: (ledger, cache),
    )
    return collect_a4b_final_pool(
        contract,
        source,
        manifest,
        **final_access,
        encoder=SimpleNamespace(encode=lambda records: None),
        runtime_factory=lambda *args: None,
        episode_collector=_collector(lambda _task, rank: rank < 12),
    )


def test_final_selects_first_twelve_eligible_and_stops(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract, source, manifest = _contracts()
    final_access, validated_paths = _final_access(
        monkeypatch, tmp_path, contract, source, manifest
    )
    sentinel_ledger = {"kind": "ledger"}
    sentinel_cache = {"kind": "cache"}
    monkeypatch.setattr(
        collection_module,
        "_stack_final_artifacts",
        lambda proposals, bundles: (sentinel_ledger, sentinel_cache),
    )
    final = collect_a4b_final_pool(
        contract,
        source,
        manifest,
        **final_access,
        encoder=SimpleNamespace(encode=lambda records: None),
        runtime_factory=lambda *args: None,
        episode_collector=_collector(lambda _task, rank: 2 <= rank <= 13),
    )

    assert final.passed is True
    assert validated_paths == [final_access["formal_train_cache_path"].resolve()]
    assert final.proposal_ledger is sentinel_ledger
    assert final.state_cache is sentinel_cache
    for task in TASKS:
        assert len(final.attempts_by_task[task]) == 14
        assert [row.candidate_rank for row in final.attempts_by_task[task] if row.selected] == list(
            range(2, 14)
        )
        assert len(final.selected_episode_uids_by_task[task]) == (
            TARGET_ELIGIBLE_CANDIDATES_PER_TASK
        )

    result = build_a4b_collection_result(
        final,
        output={
            "state_cache": str((tmp_path / "cache.npz").resolve()),
            "state_cache_sha256": "a" * 64,
            "proposal_ledger": str((tmp_path / "ledger.npz").resolve()),
            "proposal_ledger_sha256": "b" * 64,
            "validated_endpoint_seeds": list(CONFIRMATION_TRAINING_SEEDS),
        },
    )
    validate_a4b_collection_result(result)


def test_insufficient_final_pool_writes_only_nonoverwrite_failure_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract, source, manifest = _contracts()
    final_access, _validated_paths = _final_access(
        monkeypatch, tmp_path, contract, source, manifest
    )
    final = collect_a4b_final_pool(
        contract,
        source,
        manifest,
        **final_access,
        encoder=SimpleNamespace(encode=lambda records: None),
        runtime_factory=lambda *args: None,
        episode_collector=_collector(lambda task, rank: task == TASKS[0] and rank < 11),
    )

    assert final.passed is False
    assert final.failure_task == TASKS[0]
    assert final.state_cache is None and final.proposal_ledger is None
    output_dir = tmp_path / "failure"
    result_path = output_dir / "result.json"
    cache_path = output_dir / "cache.npz"
    ledger_path = output_dir / "ledger.npz"
    result = publish_a4b_final_bundle_exclusive(
        result_path=result_path,
        cache_path=cache_path,
        ledger_path=ledger_path,
        collection=final,
        validated_endpoint_seeds=CONFIRMATION_TRAINING_SEEDS,
    )

    assert result["failure"] == {
        "code": "insufficient_eligible_candidates",
        "task": TASKS[0],
        "candidate_limit": 40,
        "required": 12,
        "available": 11,
    }
    assert result_path.is_file()
    assert not cache_path.exists() and not ledger_path.exists()
    with pytest.raises(FileExistsError):
        publish_a4b_final_bundle_exclusive(
            result_path=result_path,
            cache_path=cache_path,
            ledger_path=ledger_path,
            collection=final,
            validated_endpoint_seeds=CONFIRMATION_TRAINING_SEEDS,
        )


def test_preflight_attempts_complete_pool_and_rejects_legacy_digest_overlap() -> None:
    contract, source, manifest = _contracts()
    with pytest.raises(PermissionError, match="overlaps"):
        collect_a4b_preflight_pool(
            contract,
            source,
            manifest,
            legacy_physical_identities={(TASKS[0], "f" * 64)},
            runtime_factory=lambda *args: None,
            episode_collector=_collector(
                lambda _task, _rank: False,
                forced_digest=(TASKS[0], 0, POOL_PREFLIGHT),
            ),
        )

    preflight = collect_a4b_preflight_pool(
        contract,
        source,
        manifest,
        legacy_physical_identities=set(),
        runtime_factory=lambda *args: None,
        episode_collector=_collector(lambda _task, rank: rank % 2 == 0),
    )
    assert preflight.passed is True
    assert all(len(preflight.attempts_by_task[task]) == 8 for task in TASKS)
    assert all(not preflight.selected_episode_uids_by_task[task] for task in TASKS)
    result = build_a4b_collection_result(
        preflight, legacy_reset_identities=_legacy_inventory(source)
    )
    validate_a4b_collection_result(result)

    contaminated = deepcopy(result)
    contaminated["attempts_by_task"][TASKS[0]][0][
        "reset_physical_state_goal_digest"
    ] = contaminated["legacy_reset_identities"]["a4_v2_formal_train"][0]["digest"]
    with pytest.raises(PermissionError, match="overlap"):
        validate_a4b_collection_result(contaminated)


def test_preflight_receipt_survives_real_sorted_json_roundtrip(
    tmp_path: Path,
) -> None:
    contract, source, manifest = _contracts()
    receipt_path = _write_preflight_receipt(
        tmp_path, contract, source, manifest
    )

    persisted = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert tuple(persisted["attempts_by_task"]) == tuple(sorted(TASKS))
    assert tuple(persisted["attempts_by_task"]) != TASKS
    assert tuple(persisted["selected_episode_uids_by_task"]) == tuple(
        sorted(TASKS)
    )
    assert tuple(persisted["legacy_reset_identities"]) == tuple(
        sorted(LEGACY_IDENTITY_SCOPES)
    )
    validate_a4b_collection_result(persisted)
    assert load_a4b_preflight_receipt(receipt_path) == persisted

    extra_task = deepcopy(persisted)
    extra_task["attempts_by_task"]["not-an-a4b-task"] = []
    with pytest.raises(ValueError, match="task keys"):
        validate_a4b_collection_result(extra_task)


def test_final_api_owns_receipt_authorization_and_forbidden_identities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parameters = inspect.signature(collect_a4b_final_pool).parameters
    assert "authorization" not in parameters
    assert "forbidden_physical_identities" not in parameters

    contract, source, manifest = _contracts()
    final_access, validated_paths = _final_access(
        monkeypatch,
        tmp_path,
        contract,
        source,
        manifest,
        forced_digest=(TASKS[0], 0, POOL_PREFLIGHT),
    )
    with pytest.raises(PermissionError, match="overlaps"):
        collect_a4b_final_pool(
            contract,
            source,
            manifest,
            **final_access,
            encoder=SimpleNamespace(encode=lambda records: None),
            runtime_factory=lambda *args: None,
            episode_collector=_collector(
                lambda _task, _rank: False,
                forced_digest=(TASKS[0], 0, POOL_FINAL),
            ),
        )
    assert validated_paths == [final_access["formal_train_cache_path"].resolve()]


def test_final_api_revalidates_persisted_development_go_before_collection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract, source, manifest = _contracts()
    receipt = _write_preflight_receipt(tmp_path, contract, source, manifest)
    development_go = tmp_path / "development-go.json"
    development_go.write_text("{}\n", encoding="utf-8")

    def reject_authorization(*args: object, **kwargs: object) -> None:
        raise PermissionError("unvalidated Development-Go")

    monkeypatch.setattr(
        collection_module,
        "validate_a4_development_go_summary",
        reject_authorization,
    )
    with pytest.raises(PermissionError, match="Development-Go"):
        collect_a4b_final_pool(
            contract,
            source,
            manifest,
            preflight_receipt_path=receipt,
            development_go_summary_path=development_go,
            formal_train_cache_path=tmp_path / "formal-cache.npz",
            encoder=SimpleNamespace(encode=lambda records: None),
            runtime_factory=lambda *args: None,
            episode_collector=lambda **kwargs: pytest.fail(
                "collection started before authorization"
            ),
        )


def test_final_bundle_is_published_together_and_binds_plain_sha256(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    final = _successful_final(monkeypatch, tmp_path)
    output_dir = tmp_path / "published-final"
    result_path = output_dir / "result.json"
    cache_path = output_dir / "cache.npz"
    ledger_path = output_dir / "ledger.npz"

    result = publish_a4b_final_bundle_exclusive(
        result_path=result_path,
        cache_path=cache_path,
        ledger_path=ledger_path,
        collection=final,
        validated_endpoint_seeds=CONFIRMATION_TRAINING_SEEDS,
    )

    assert {path.name for path in output_dir.iterdir()} == {
        "result.json",
        "cache.npz",
        "ledger.npz",
    }
    assert json.loads(result_path.read_text(encoding="utf-8")) == result
    assert result["output"]["state_cache_sha256"] == a4b_artifact_sha256(
        cache_path
    )
    assert result["output"]["proposal_ledger_sha256"] == a4b_artifact_sha256(
        ledger_path
    )
    assert validate_a4b_result_cache_path(result, cache_path) == cache_path

    with ledger_path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(PermissionError, match="ledger content"):
        validate_a4b_result_cache_path(result, cache_path)


def test_final_bundle_write_failure_leaves_no_partial_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    final = _successful_final(monkeypatch, tmp_path)
    output_dir = tmp_path / "failed-final"

    def fail_ledger_write(*args: object, **kwargs: object) -> None:
        raise RuntimeError("ledger write failed")

    monkeypatch.setattr(
        collection_module,
        "save_a4b_final_ledger_exclusive",
        fail_ledger_write,
    )
    with pytest.raises(RuntimeError, match="ledger write failed"):
        publish_a4b_final_bundle_exclusive(
            result_path=output_dir / "result.json",
            cache_path=output_dir / "cache.npz",
            ledger_path=output_dir / "ledger.npz",
            collection=final,
            validated_endpoint_seeds=CONFIRMATION_TRAINING_SEEDS,
        )

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".failed-final.*.tmp"))
