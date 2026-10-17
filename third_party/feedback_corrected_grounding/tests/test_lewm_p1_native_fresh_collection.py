from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import fcpg.lewm_p1_native_fresh_collection as fresh
from fcpg.lewm_counterfactual_runtime import FrameRecord


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _contract() -> dict[str, object]:
    path = Path("config/lewm/eval/p1_native_complementary_fresh.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _inventory(**overrides: object) -> fresh.FreshnessInventory:
    reset: dict[str, object] = {}
    observed = {
        scope: frozenset({_sha(f"observed:{scope}")})
        for scope in fresh.REQUIRED_OBSERVED_SIGNATURE_SCOPES
    }
    reset.update(overrides.get("reset", {}))  # type: ignore[arg-type]
    observed.update(overrides.get("observed", {}))  # type: ignore[arg-type]
    return fresh.FreshnessInventory(reset, observed)


def test_fresh_plan_is_exactly_30_pick_episodes_and_deterministic() -> None:
    contract = _contract()
    first = fresh.build_fresh_episode_plans(contract)
    second = fresh.build_fresh_episode_plans(contract)

    assert first == second
    assert len(first) == 30
    assert [plan.raw_task_index for plan in first] == list(range(30))
    assert [plan.candidate_rank for plan in first] == list(range(30))
    assert [plan.environment_seed for plan in first] == list(range(840_000, 840_030))
    assert [plan.hazard_id for plan in first] == contract["fresh_data"]["hazard_model_steps"]
    assert all(plan.episode_uid.startswith("p1-native-fresh-v1:benchmark-41801") for plan in first)


def test_freshness_inventory_fails_closed_on_missing_or_empty_observed_scope() -> None:
    reset: dict[str, set[tuple[str, str]]] = {}
    observed = {
        scope: {_sha(scope)} for scope in fresh.REQUIRED_OBSERVED_SIGNATURE_SCOPES
    }
    observed.pop(fresh.REQUIRED_OBSERVED_SIGNATURE_SCOPES[0])
    with pytest.raises(PermissionError, match="lacks|required"):
        fresh.FreshnessInventory(reset, observed)

    observed[fresh.REQUIRED_OBSERVED_SIGNATURE_SCOPES[0]] = set()
    with pytest.raises(PermissionError, match="empty"):
        fresh.FreshnessInventory(reset, observed)

    valid_observed = {
        fresh.REQUIRED_OBSERVED_SIGNATURE_SCOPES[0]: {_sha("legacy-observed")}
    }
    inventory = fresh.FreshnessInventory({}, valid_observed)
    assert inventory.reset_digest_scopes == {}
    assert inventory.legacy_training_reset_digest_status == (
        fresh.LEGACY_TRAINING_RESET_STATUS
    )


def test_freshness_inventory_rejects_reset_or_observation_overlap() -> None:
    reset_collision = _sha("new-reset")
    observed_collision = _sha("new-observed")
    reset = {"known-a4b": {(fresh.TASK, reset_collision)}}
    observed = {
        scope: {_sha(scope)} for scope in fresh.REQUIRED_OBSERVED_SIGNATURE_SCOPES
    }
    inventory = fresh.FreshnessInventory(reset, observed)
    with pytest.raises(PermissionError, match="overlaps"):
        inventory.assert_fresh(
            task_name=fresh.TASK,
            reset_digest=reset_collision,
            observed_signature=_sha("unused"),
        )

    observed[fresh.REQUIRED_OBSERVED_SIGNATURE_SCOPES[0]] = {observed_collision}
    inventory = fresh.FreshnessInventory(reset, observed)
    with pytest.raises(PermissionError, match="overlaps"):
        inventory.assert_fresh(
            task_name=fresh.TASK,
            reset_digest=_sha("different-reset"),
            observed_signature=observed_collision,
        )


def test_legacy_signature_inventory_hashes_only_requested_pick_rows(tmp_path) -> None:
    source = tmp_path / "legacy.npz"
    tasks = np.asarray([fresh.TASK, "push-v3", fresh.TASK])
    obs = np.arange(117, dtype=np.float32).reshape(3, 39)
    xyz = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    np.savez(source, task_name=tasks, obs=obs, keypoints_xyz=xyz)

    inventory = fresh.legacy_observed_state_signatures(source)

    assert inventory == {
        fresh.observed_state_signature(fresh.TASK, obs[0], xyz[0]),
        fresh.observed_state_signature(fresh.TASK, obs[2], xyz[2]),
    }


class _FrozenPolicy(nn.Module):
    obs_horizon = 2

    def __init__(self) -> None:
        super().__init__()
        self.eval()


class _FakeEnv:
    def __init__(self) -> None:
        self.data = SimpleNamespace(
            qpos=np.asarray([0.0, 1.0], dtype=np.float64),
            qvel=np.asarray([0.0, 0.0], dtype=np.float64),
        )
        self._target_pos = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
        self._last_rand_vec = np.asarray([0.4, 0.5, 0.6], dtype=np.float64)
        self.action_space = SimpleNamespace(
            low=np.full(4, -1.0, dtype=np.float32),
            high=np.full(4, 1.0, dtype=np.float32),
        )
        self.step_index = 0
        self.closed = False

    def step(self, action: np.ndarray):
        self.step_index += 1
        observation = np.full(39, self.step_index, dtype=np.float32)
        return observation, 0.0, False, False, {}

    def close(self) -> None:
        self.closed = True


class _Expert:
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        marker = float(np.asarray(observation)[0])
        return np.asarray([0.001 * marker, 0.0, 0.0, 1.0], dtype=np.float32)


class _Runtime:
    def __init__(self) -> None:
        self.env = _FakeEnv()
        self.expert = _Expert()
        self.observation = np.zeros(39, dtype=np.float32)

    def record_current(self, observation: np.ndarray) -> FrameRecord:
        marker = float(np.asarray(observation)[0])
        return FrameRecord(
            rgb=np.full((3, 4, 4), marker, dtype=np.float32),
            proprio=np.asarray(observation[:4], dtype=np.float32).copy(),
            keypoints_xyz=np.full((3, 3), marker + 0.2, dtype=np.float32),
            q_nominal=np.full((3, 2), marker + 0.3, dtype=np.float32),
            valid_nominal=np.ones(3, dtype=bool),
            q_true=np.full((3, 2), marker + 0.4, dtype=np.float32),
            valid_true=np.ones(3, dtype=bool),
        )


def test_native_collector_uses_prebuilt_hazard_and_emits_h0_h2_then_h3_h8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = fresh.build_fresh_episode_plans(_contract())[0]
    runtime = _Runtime()
    factory_calls: list[tuple[str, int, int, int]] = []

    def runtime_factory(task: str, raw_index: int, env_seed: int) -> _Runtime:
        # The randomized hazard is already immutable before reset/factory call.
        factory_calls.append((task, raw_index, env_seed, plan.hazard_id))
        return runtime

    def fake_branch_builder(center: np.ndarray, **_: object) -> SimpleNamespace:
        branches = np.broadcast_to(center[None], (8, *center.shape)).copy()
        for branch in range(8):
            branches[branch, :, 1] = branch * 0.01
        return SimpleNamespace(actions=branches)

    def fake_extract(
        policy: nn.Module,
        pairs: list[tuple[FrameRecord, FrameRecord]] | tuple[tuple[FrameRecord, FrameRecord], ...],
        *,
        device: torch.device,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del policy, device
        marker = np.asarray([pair[-1].proprio[0] for pair in pairs], dtype=np.float32)
        token = np.broadcast_to(marker[:, None, None], (len(marker), 3, 256)).copy()
        visual = token + 0.25
        xyz = np.broadcast_to(marker[:, None, None], (len(marker), 3, 3)).copy() + 0.5
        return token, visual, xyz

    monkeypatch.setattr(fresh, "_extract_pair_endpoints", fake_extract)

    def capture(env: _FakeEnv, *, include_process_rng: bool) -> int:
        assert include_process_rng
        return env.step_index

    def restore(env: _FakeEnv, snapshot: int) -> None:
        env.step_index = snapshot

    row = fresh.collect_fresh_p1_native_episode(
        plan,
        inventory=_inventory(),
        policy=_FrozenPolicy(),
        device="cpu",
        runtime_factory=runtime_factory,
        branch_bank_builder=fake_branch_builder,
        snapshot_capture=capture,
        snapshot_restore=restore,
    )

    assert factory_calls == [
        (fresh.TASK, plan.raw_task_index, plan.environment_seed, plan.hazard_id)
    ]
    initial_xyz = np.full((3, 3), 0.2, dtype=np.float32)
    assert row.observed_signature == fresh.observed_state_signature(
        fresh.TASK, np.zeros(39, dtype=np.float32), initial_xyz
    )
    with pytest.raises(ValueError, match="full 39-D"):
        fresh.observed_state_signature(
            fresh.TASK, np.zeros(4, dtype=np.float32), initial_xyz
        )
    assert runtime.env.closed
    assert row.history_token.shape == (8, 3, 3, 256)
    assert row.future_action.shape == (8, 6, 20)
    assert row.target_token.shape == (8, 6, 3, 256)
    hazard_raw = plan.hazard_id * 5
    assert row.history_token[0, :, 0, 0].tolist() == pytest.approx(
        [hazard_raw, hazard_raw + 5, hazard_raw + 10]
    )
    assert row.target_token[0, :, 0, 0].tolist() == pytest.approx(
        [hazard_raw + 15, hazard_raw + 20, hazard_raw + 25,
         hazard_raw + 30, hazard_raw + 35, hazard_raw + 40]
    )


def _fresh_row(plan: fresh.FreshEpisodePlan) -> fresh.FreshP1NativeEpisode:
    marker = np.float32(plan.candidate_rank)
    return fresh.FreshP1NativeEpisode(
        plan=plan,
        reset_digest=_sha(f"fresh-reset:{plan.candidate_rank}"),
        observed_signature=_sha(f"fresh-observed:{plan.candidate_rank}"),
        history_token=np.full((8, 3, 3, 256), marker, np.float32),
        history_action=np.full((8, 2, 20), marker, np.float32),
        future_action=np.stack(
            [np.full((6, 20), marker + branch, np.float32) for branch in range(8)]
        ),
        target_token=np.full((8, 6, 3, 256), marker, np.float32),
        history_sampled_visual=np.full((8, 3, 3, 256), marker, np.float32),
        history_xyz=np.full((8, 3, 3, 3), marker, np.float32),
        future_xyz=np.full((8, 6, 3, 3), marker, np.float32),
    )


def test_direct_fresh_schema_and_native_cache_preserve_episode_nesting() -> None:
    contract = _contract()
    rows = [_fresh_row(plan) for plan in fresh.build_fresh_episode_plans(contract)]
    arrays = fresh.stack_fresh_p1_native_episodes(rows, contract=contract)
    deployment, labels = fresh.split_fresh_cache_arrays(arrays, contract=contract)
    fresh.validate_fresh_deployment_arrays(deployment, contract=contract)
    fresh.validate_fresh_offline_label_arrays(labels, contract=contract)
    combined = fresh.combine_fresh_cache_arrays(
        deployment, labels, contract=contract
    )
    cache = fresh.fresh_arrays_to_native_cache(arrays, contract=contract)

    assert set(arrays) == fresh.FRESH_CACHE_KEYS
    assert arrays["history_sensor_valid"].all()
    assert not arrays["future_sensor_valid"].any()
    assert arrays["history_token"].shape == (30, 8, 3, 3, 256)
    assert arrays["target_token"].shape == (30, 8, 6, 3, 256)
    assert set(deployment) == fresh.DEPLOYMENT_CACHE_KEYS
    assert set(labels) == fresh.OFFLINE_LABEL_CACHE_KEYS
    assert "target_token" not in deployment
    assert "history_token" not in labels
    assert set(combined) == fresh.FRESH_CACHE_KEYS
    for name in fresh.FRESH_CACHE_KEYS:
        np.testing.assert_array_equal(combined[name], arrays[name])
    assert cache.bundle_count == 30
    assert cache.branch_count == 8
    physical = set(cache.physical_episode_id.reshape(-1).tolist())
    assert len(physical) == 30
    assert all(value.startswith(f"{fresh.TASK}:") for value in physical)
    assert set(cache.split.reshape(-1).tolist()) == {"fresh_confirmation"}
    for shift in range(1, 8):
        donor = (np.arange(8) + shift) % 8
        np.testing.assert_array_equal(
            cache.wrong_future_action[:, shift - 1],
            cache.future_action[:, donor],
        )

    mismatched_labels = {name: value.copy() for name, value in labels.items()}
    mismatched_labels["physical_episode_id"][0] = (
        f"{fresh.TASK}:{_sha('different-physical-episode')}"
    )
    with pytest.raises(PermissionError, match="identity mismatch"):
        fresh.combine_fresh_cache_arrays(
            deployment, mismatched_labels, contract=contract
        )


def test_save_bundle_writes_separate_validated_caches_and_receipt(tmp_path) -> None:
    root = tmp_path
    contract = json.loads(json.dumps(_contract()))
    config_path = root / "config/lewm/eval/p1_native_complementary_fresh.json"
    policy_path = root / "artifacts/policy.pt"
    corrector_path = root / "artifacts/corrector.pt"
    legacy_source_path = root / "data/legacy.npz"
    inventory_path = root / "receipts/legacy_signatures.npz"
    deployment_path = root / "outputs/fresh_deployment_inputs.npz"
    label_path = root / "outputs/fresh_offline_labels.npz"
    metadata_path = root / "outputs/fresh_collection.metadata.json"
    for path, payload in (
        (policy_path, b"frozen-policy"),
        (corrector_path, b"frozen-corrector"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    legacy_source_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_obs = np.arange(78, dtype=np.float32).reshape(2, 39)
    legacy_xyz = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    np.savez(
        legacy_source_path,
        task_name=np.asarray([fresh.TASK, "push-v3"]),
        obs=legacy_obs,
        keypoints_xyz=legacy_xyz,
    )
    legacy_signatures = fresh.legacy_observed_state_signatures(legacy_source_path)
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        inventory_path,
        observed_state_signature=np.asarray(sorted(legacy_signatures)),
    )

    contract["fresh_data"]["deployment_cache"] = str(
        deployment_path.relative_to(root)
    )
    contract["fresh_data"]["label_cache"] = str(label_path.relative_to(root))
    contract["fresh_data"]["metadata"] = str(metadata_path.relative_to(root))
    contract["freshness"]["legacy_observation_source"] = str(
        legacy_source_path.relative_to(root)
    )
    contract["freshness"]["legacy_observation_source_sha256"] = (
        fresh.sha256_file(legacy_source_path)
    )
    contract["freshness"]["legacy_observed_state_inventory"] = str(
        inventory_path.relative_to(root)
    )
    contract["p1"]["policy_checkpoint"] = str(policy_path.relative_to(root))
    contract["p1"]["policy_checkpoint_sha256"] = fresh.sha256_file(policy_path)
    contract["p1"]["corrector_checkpoint"] = str(
        corrector_path.relative_to(root)
    )
    contract["p1"]["corrector_checkpoint_sha256"] = fresh.sha256_file(
        corrector_path
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(contract), encoding="utf-8")

    rows = [
        _fresh_row(plan) for plan in fresh.build_fresh_episode_plans(contract)
    ]
    arrays = fresh.stack_fresh_p1_native_episodes(rows, contract=contract)
    inventory = fresh.FreshnessInventory(
        {},
        {
            fresh.REQUIRED_OBSERVED_SIGNATURE_SCOPES[0]: legacy_signatures
        },
    )
    metadata = fresh.save_fresh_cache_bundle(
        arrays,
        contract=contract,
        config_path=config_path,
        inventory=inventory,
    )

    assert metadata["completed"] is True
    assert metadata["schema_version"] == 1
    assert metadata["protocol"] == fresh.PROTOCOL
    assert metadata["freshness_receipt"][
        "legacy_training_reset_digest_status"
    ] == fresh.LEGACY_TRAINING_RESET_STATUS
    assert metadata["freshness_receipt"][
        "reset_digest_scope_completeness_claimed"
    ] is False
    assert metadata["freshness_receipt"][
        "exact_observed_signature_overlap_count"
    ] == 0
    assert fresh.sha256_file(deployment_path) == metadata["deployment_cache"][
        "sha256"
    ]
    assert fresh.sha256_file(label_path) == metadata["label_cache"]["sha256"]
    with np.load(deployment_path, allow_pickle=False) as archive:
        deployment = {name: archive[name] for name in archive.files}
    with np.load(label_path, allow_pickle=False) as archive:
        labels = {name: archive[name] for name in archive.files}
    fresh.validate_fresh_deployment_arrays(deployment, contract=contract)
    fresh.validate_fresh_offline_label_arrays(labels, contract=contract)
    combined = fresh.combine_fresh_cache_arrays(
        deployment, labels, contract=contract
    )
    np.testing.assert_array_equal(combined["target_token"], arrays["target_token"])
    receipt = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert receipt["dataset_sha256"] == metadata["dataset_sha256"]

    with pytest.raises(FileExistsError, match="already exists"):
        fresh.save_fresh_cache_bundle(
            arrays,
            contract=contract,
            config_path=config_path,
            inventory=inventory,
        )


def test_formal_stack_rejects_missing_episode_or_duplicate_identity() -> None:
    contract = _contract()
    plans = fresh.build_fresh_episode_plans(contract)
    rows = [_fresh_row(plan) for plan in plans]
    with pytest.raises(ValueError, match="exactly 30"):
        fresh.stack_fresh_p1_native_episodes(rows[:-1], contract=contract)

    duplicate = list(rows)
    second = duplicate[1]
    duplicate[1] = fresh.FreshP1NativeEpisode(
        plan=second.plan,
        reset_digest=duplicate[0].reset_digest,
        observed_signature=second.observed_signature,
        history_token=second.history_token,
        history_action=second.history_action,
        future_action=second.future_action,
        target_token=second.target_token,
        history_sampled_visual=second.history_sampled_visual,
        history_xyz=second.history_xyz,
        future_xyz=second.future_xyz,
    )
    with pytest.raises(PermissionError, match="repeated physical"):
        fresh.stack_fresh_p1_native_episodes(duplicate, contract=contract)
