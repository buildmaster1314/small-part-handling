from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fcpg.lewm_counterfactual_v3 import (
    CANDIDATE_COUNT,
    HORIZON,
    PHASE_NAMES,
    PHASE_STEPS,
    TASK_NAMES,
    canonical_sha256,
    validate_array_bundle,
    validate_action_hash_bindings,
    validate_config,
    validate_episode_split_bindings,
    validate_index_rows,
)
from fcpg.lewm_counterfactual_v3_runtime import (
    build_v3_candidate_bank,
    execute_v3_branch,
    validate_loaded_candidate_artifacts,
)
from fcpg.lewm_counterfactual_v3_state import (
    RAW_MOCAP_QUATERNION_SLICE,
    RAW_QPOS_SLICE,
    STATE_DIM,
    StateOracleInput,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/lewm/eval/counterfactual_v3_state_oracle_seed9216.json"


def _config() -> dict[str, object]:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _state() -> np.ndarray:
    value = np.zeros(STATE_DIM, dtype=np.float32)
    value[10] = 1.0
    value[RAW_QPOS_SLICE.start + 12] = 1.0
    value[RAW_MOCAP_QUATERNION_SLICE.start] = 1.0
    return value


def _bundle(n: int = 2) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    deployment = {
        "center_action_raw": np.zeros((n, 25, 4), dtype=np.float32),
        "candidate_action_raw": np.zeros((n, 8, 25, 4), dtype=np.float32),
    }
    current = np.stack([_state()] * n)
    labels = {
        "current_state": current,
        "future_state": np.broadcast_to(
            current[:, None, None, :], (n, 8, 5, STATE_DIM)
        ).copy(),
        "goal_world": np.zeros((n, 3), dtype=np.float32),
        "physical_task_score": np.zeros((n, 8), dtype=np.float32),
        "actual_return": np.zeros((n, 8, 5), dtype=np.float32),
        "progress": np.zeros((n, 8, 5), dtype=np.float32),
        "success": np.zeros((n, 8, 5), dtype=bool),
    }
    audit = {
        "task_index": np.zeros(n, dtype=np.int16),
        "episode_id": np.arange(n, dtype=np.int32),
        "env_seed": np.arange(n, dtype=np.int64) + 60_000,
        "snapshot_step": np.full(n, 10, dtype=np.int16),
        "phase_index": np.zeros(n, dtype=np.int8),
        "split_index": np.arange(n, dtype=np.int8) % 2,
        "candidate_axis": np.tile(np.repeat(np.arange(4), 2), (n, 1)).astype(np.int16),
        "candidate_sign": np.tile([1, -1] * 4, (n, 1)).astype(np.int8),
        "candidate_permutation": np.tile(np.arange(8), (n, 1)).astype(np.int16),
        "candidate_delta_rms": np.full((n, 8), 0.15, dtype=np.float32),
    }
    return deployment, labels, audit


def test_frozen_v3_config_and_privileged_array_split() -> None:
    config = _config()
    validate_config(config)
    deployment, labels, audit = _bundle()
    axes = validate_array_bundle(deployment, labels, audit)
    assert (axes.sample_count, axes.candidate_count, axes.horizon, axes.state_dim) == (
        2,
        CANDIDATE_COUNT,
        HORIZON,
        STATE_DIM,
    )

    poisoned = dict(deployment)
    poisoned["goal_world"] = labels["goal_world"]
    with pytest.raises(ValueError, match="keys changed|privileged"):
        validate_array_bundle(poisoned, labels, audit)

    changed = json.loads(json.dumps(config))
    changed["seed"] = 9217
    with pytest.raises(ValueError, match="pre-registered config changed"):
        validate_config(changed)


def test_index_rows_require_unique_candidate_hashes() -> None:
    deployment, labels, audit = _bundle(n=1)
    axes = validate_array_bundle(deployment, labels, audit)
    digest = "a" * 64
    row = {
        "sample_index": 0,
        "task": TASK_NAMES[0],
        "episode_id": 0,
        "env_seed": 60_000,
        "phase": "early",
        "snapshot_step": 10,
        "split": "train",
        "physical_episode_sha256": "b" * 64,
        "physical_snapshot_sha256": "c" * 64,
        "center_action_sha256": "d" * 64,
        "candidate_action_sha256": [f"{index:064x}" for index in range(8)],
    }
    validate_index_rows((row,), axes=axes)
    row["candidate_action_sha256"] = [digest] * 8
    with pytest.raises(ValueError, match="non-unique"):
        validate_index_rows((row,), axes=axes)


def test_action_hash_binding_recomputes_array_content() -> None:
    deployment, labels, audit = _bundle(n=1)
    for slot in range(8):
        deployment["candidate_action_raw"][0, slot, 0, 0] = float(slot + 1) / 10.0
    axes = validate_array_bundle(deployment, labels, audit)
    from fcpg.lewm_counterfactual_runtime import sha256_array

    row = {
        "sample_index": 0,
        "task": TASK_NAMES[0],
        "episode_id": 0,
        "env_seed": 60_000,
        "phase": "early",
        "snapshot_step": 10,
        "split": "train",
        "physical_episode_sha256": "b" * 64,
        "physical_snapshot_sha256": "c" * 64,
        "center_action_sha256": sha256_array(deployment["center_action_raw"][0]),
        "candidate_action_sha256": [
            sha256_array(deployment["candidate_action_raw"][0, slot])
            for slot in range(8)
        ],
    }
    validate_index_rows((row,), axes=axes)
    validate_action_hash_bindings(deployment, (row,))
    deployment["candidate_action_raw"][0, 0, 0, 0] = 0.25
    with pytest.raises(ValueError, match="candidate action hash mismatch"):
        validate_action_hash_bindings(deployment, (row,))


def _formal_episode_split_fixture() -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    values = {name: [] for name in (
        "task_index",
        "episode_id",
        "env_seed",
        "snapshot_step",
        "phase_index",
        "split_index",
    )}
    rows: list[dict[str, object]] = []
    sample = 0
    for task_index, task in enumerate(TASK_NAMES):
        identities = []
        for episode_id in range(12):
            env_seed = 70_000 + task_index * 100 + episode_id
            digest = canonical_sha256(
                {
                    "task_index": task_index,
                    "episode_id": episode_id,
                    "env_seed": env_seed,
                }
            )
            identities.append((digest, episode_id, env_seed))
        train = {episode_id for _, episode_id, _ in sorted(identities)[:8]}
        for digest, episode_id, env_seed in identities:
            split_index = 0 if episode_id in train else 1
            for phase_index, (phase, snapshot_step) in enumerate(
                zip(PHASE_NAMES, PHASE_STEPS, strict=True)
            ):
                values["task_index"].append(task_index)
                values["episode_id"].append(episode_id)
                values["env_seed"].append(env_seed)
                values["snapshot_step"].append(snapshot_step)
                values["phase_index"].append(phase_index)
                values["split_index"].append(split_index)
                rows.append(
                    {
                        "task": task,
                        "episode_id": episode_id,
                        "env_seed": env_seed,
                        "phase": phase,
                        "snapshot_step": snapshot_step,
                        "split": "train" if split_index == 0 else "validation",
                        "physical_episode_sha256": digest,
                        "physical_snapshot_sha256": f"{sample + 1:064x}",
                    }
                )
                sample += 1
    arrays = {
        name: np.asarray(value, dtype=np.int64) for name, value in values.items()
    }
    return arrays, rows


def test_episode_split_binding_reconstructs_identity_phases_and_sha_order() -> None:
    audit, rows = _formal_episode_split_fixture()
    validate_episode_split_bindings(
        audit,
        rows,
        train_count_per_task=8,
        validation_count_per_task=4,
    )

    poisoned_rows = [dict(row) for row in rows]
    poisoned_rows[0]["physical_episode_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="does not match its identity"):
        validate_episode_split_bindings(
            audit,
            poisoned_rows,
            train_count_per_task=8,
            validation_count_per_task=4,
        )

    poisoned_audit = {name: value.copy() for name, value in audit.items()}
    poisoned_audit["split_index"][1] = 1 - poisoned_audit["split_index"][1]
    poisoned_rows = [dict(row) for row in rows]
    poisoned_rows[1]["split"] = (
        "train" if poisoned_audit["split_index"][1] == 0 else "validation"
    )
    with pytest.raises(ValueError, match="crosses train/validation"):
        validate_episode_split_bindings(
            poisoned_audit,
            poisoned_rows,
            train_count_per_task=8,
            validation_count_per_task=4,
        )


def test_candidate_builder_uses_policy_center_and_hidden_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    center = np.zeros((25, 4), dtype=np.float32)
    center[:, 3] = 0.75
    monkeypatch.setattr(
        "fcpg.lewm_counterfactual_v3_runtime.policy_action_chunk",
        lambda _plan, _frozen: center,
    )
    episode = SimpleNamespace(
        env=SimpleNamespace(
            action_space=SimpleNamespace(
                low=np.full(4, -1.0, dtype=np.float32),
                high=np.full(4, 1.0, dtype=np.float32),
            )
        ),
        records_clean=[],
        actions=np.zeros((60, 4), dtype=np.float32),
        task=TASK_NAMES[0],
        episode=0,
    )
    actual_center, bank, hidden = build_v3_candidate_bank(
        episode, 10, SimpleNamespace(), _config()
    )
    np.testing.assert_array_equal(actual_center, center)
    np.testing.assert_array_equal(
        bank.actions[:, :, 3],
        np.broadcast_to(center[None, :, 3], (8, 25)),
    )
    assert bank.names == tuple(f"candidate_{index}" for index in range(8))
    assert sorted(hidden.slot_to_canonical_direction) == list(range(8))

    canonical = np.asarray(hidden.slot_to_canonical_direction, dtype=np.int16)
    deployment = {
        "center_action_raw": actual_center[None],
        "candidate_action_raw": bank.actions[None],
    }
    audit = {
        "task_index": np.asarray([0], dtype=np.int16),
        "episode_id": np.asarray([0], dtype=np.int32),
        "snapshot_step": np.asarray([10], dtype=np.int16),
        "candidate_axis": (canonical // 2)[None],
        "candidate_sign": np.where(canonical % 2 == 0, 1, -1).astype(np.int8)[None],
        "candidate_permutation": canonical[None],
        "candidate_delta_rms": np.asarray(
            hidden.realized_motion_rms, dtype=np.float32
        )[None],
    }
    validate_loaded_candidate_artifacts(_config(), deployment, audit)
    audit["candidate_sign"] = audit["candidate_sign"].copy()
    audit["candidate_sign"][0, 0] *= -1
    with pytest.raises(ValueError, match="candidate sign"):
        validate_loaded_candidate_artifacts(_config(), deployment, audit)


def test_executor_rejects_reclipping_and_collects_exact_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.action_space = SimpleNamespace(
                low=np.full(4, -1.0, dtype=np.float32),
                high=np.full(4, 1.0, dtype=np.float32),
            )
            self.steps = 0

        def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, float]]:
            assert np.array_equal(action, np.zeros(4, dtype=np.float32))
            self.steps += 1
            return (
                np.asarray([0.0, 0.0, 0.0, 0.5], dtype=np.float32),
                1.0,
                False,
                False,
                {"in_place_reward": float(self.steps), "success": 0.0},
            )

    env = FakeEnv()
    episode = SimpleNamespace(
        env=env,
        snapshots={10: object()},
        observations={10: np.zeros(4, dtype=np.float32)},
        task=TASK_NAMES[0],
        goal_world=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
    )
    monkeypatch.setattr(
        "fcpg.lewm_counterfactual_v3_runtime.restore_branch_snapshot",
        lambda _env, _snapshot: None,
    )
    monkeypatch.setattr(
        "fcpg.lewm_counterfactual_v3_runtime.extract_true_simulator_state",
        lambda _env, _observation, _task: StateOracleInput(
            state=_state(), goal_world=episode.goal_world.copy()
        ),
    )
    outcome = execute_v3_branch(
        episode,
        10,
        np.zeros((25, 4), dtype=np.float32),
        _config(),
    )
    assert env.steps == 25
    assert outcome.future_state.shape == (5, STATE_DIM)
    np.testing.assert_array_equal(outcome.cumulative_return, [5, 10, 15, 20, 25])

    bad = np.zeros((25, 4), dtype=np.float32)
    bad[0, 0] = 1.1
    with pytest.raises(RuntimeError, match="executor clipping"):
        execute_v3_branch(episode, 10, bad, _config())
