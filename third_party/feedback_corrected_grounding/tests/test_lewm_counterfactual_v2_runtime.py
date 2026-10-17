from __future__ import annotations

import copy
import json
import sys
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from fcpg.lewm_counterfactual import ActionCandidates
from fcpg.lewm_counterfactual_runtime import (
    FrameRecord,
    physical_snapshot_sha256,
    sha256_array,
)
from fcpg.lewm_counterfactual_v2 import (
    GOAL_VISUAL_INPUT_DIM,
    INDEX_ROW_KEYS,
    PROVENANCE_KEYS,
    TASK_NAMES,
    VIEW_NAMES,
    assert_no_privileged_inputs,
    build_branch_receipt_fields,
    canonical_sha256,
    expected_camera_condition_extrinsics,
    physical_episode_sha256,
    sha256_array as schema_sha256_array,
    smoke_config,
    validate_array_bundle,
    validate_index_and_provenance,
)
from fcpg.lewm_counterfactual_v2_runtime import (
    PHASE_STEPS,
    PhysicalBranchOutcome,
    PhysicalEpisode,
    _camera_invariant_history,
    _close_episode_env_safely,
    _history_raw_actions,
    _rgb_uint8,
    _restore_render_warmstart,
    _sample_index_row,
    build_episode_candidate_bank,
    camera_invariant_branch_labels,
    execute_candidate_bank_once,
    extract_lewm_feature_grid,
    goal_visual_from_current_rgb,
    render_oracle_future_grid,
    rng_snapshot_sha256,
    run_counterfactual_v2_collection,
)


ROOT = Path(__file__).resolve().parents[1]


def _config(*, smoke: bool = True) -> dict[str, Any]:
    with (ROOT / "config/lewm/eval/counterfactual_v2_development_seed8192.json").open(
        "r", encoding="utf-8"
    ) as stream:
        full = json.load(stream)
    return smoke_config(full) if smoke else full


def _record(value: float = 0.0) -> FrameRecord:
    q = np.full((3, 2), 32.0, dtype=np.float32)
    return FrameRecord(
        rgb=np.full((3, 224, 224), value, dtype=np.float32),
        proprio=np.full(4, value, dtype=np.float32),
        keypoints_xyz=np.full((3, 3), value, dtype=np.float32),
        q_nominal=q.copy(),
        valid_nominal=np.ones(3, dtype=bool),
        q_true=q.copy(),
        valid_true=np.ones(3, dtype=bool),
    )


def _snapshot(value: int) -> SimpleNamespace:
    return SimpleNamespace(
        integration_state=np.asarray([value], dtype=np.float64),
        warmstart_state=np.asarray([value + 1], dtype=np.float64),
        model_arrays=SimpleNamespace(
            body_pos=np.asarray([[value, 0.0, 0.0]], dtype=np.float64),
            site_pos=np.asarray([[0.0, value, 0.0]], dtype=np.float64),
        ),
        task_fingerprint=("task", value),
        runtime_attributes={"value": value},
        environment_rng=("env", value),
        process_rng=("process", value),
    )


def _episode(actions: np.ndarray | None = None) -> PhysicalEpisode:
    if actions is None:
        actions = np.arange(60 * 4, dtype=np.float32).reshape(60, 4) / 100.0
    return PhysicalEpisode(
        env=SimpleNamespace(
            action_space=SimpleNamespace(
                low=np.full(4, -10.0, dtype=np.float32),
                high=np.full(4, 10.0, dtype=np.float32),
            )
        ),
        task=TASK_NAMES[0],
        episode=2,
        env_seed=30_002,
        camera_id=0,
        base_position=np.zeros(3),
        base_quaternion=np.asarray([1.0, 0.0, 0.0, 0.0]),
        intrinsic=np.eye(3, dtype=np.float32),
        nominal_extrinsic=np.eye(4, dtype=np.float32),
        records_clean=[_record(float(index) / 100.0) for index in range(61)],
        actions=np.asarray(actions, dtype=np.float32),
        snapshots={step: _snapshot(step) for step in range(61)},
        observations={step: np.zeros(4, dtype=np.float32) for step in range(61)},
        goal_world=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
    )


def test_render_warmstart_restore_preserves_camera_forward_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Any, Any, np.ndarray, Any]] = []
    state = SimpleNamespace(mjSTATE_WARMSTART=object())
    fake_mujoco = SimpleNamespace(
        mjtState=state,
        mj_stateSize=lambda model, spec: 3,
        mj_setState=lambda model, data, value, spec: calls.append(
            (model, data, value, spec)
        ),
    )
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    model = object()
    data = object()
    episode = SimpleNamespace(env=SimpleNamespace(model=model, data=data))
    saved = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    snapshot = SimpleNamespace(warmstart_state=saved)

    _restore_render_warmstart(episode, snapshot)

    assert len(calls) == 1
    actual_model, actual_data, restored, spec = calls[0]
    assert actual_model is model
    assert actual_data is data
    assert spec is state.mjSTATE_WARMSTART
    np.testing.assert_array_equal(restored, saved)
    assert restored is not saved

    bad = SimpleNamespace(warmstart_state=np.zeros(2, dtype=np.float64))
    with pytest.raises(ValueError, match="warm-start shape"):
        _restore_render_warmstart(episode, bad)


def test_episode_close_skips_process_global_glfw_terminate_for_offscreen_viewer() -> None:
    events: list[str] = []
    closed: list[bool] = []

    class ViewerMap(dict[str, Any]):
        def clear(self) -> None:
            events.append("clear")
            super().clear()

    viewers = ViewerMap(
        rgb_array=SimpleNamespace(
            backend="egl",
            make_context_current=lambda: events.append("make_context_current"),
            con=SimpleNamespace(free=lambda: events.append("con.free")),
            free=lambda: events.append("viewer.free"),
        )
    )
    env = SimpleNamespace(
        mujoco_renderer=SimpleNamespace(_viewers=viewers),
        close=lambda: closed.append(True),
    )

    _close_episode_env_safely(env)

    assert events == ["make_context_current", "con.free", "viewer.free", "clear"]
    assert viewers == {}
    assert closed == []

    fallback = SimpleNamespace(close=lambda: closed.append(True))
    _close_episode_env_safely(fallback)
    assert closed == [True]


@pytest.mark.parametrize("snapshot_step", PHASE_STEPS)
def test_history_raw_actions_use_global_episode_steps(snapshot_step: int) -> None:
    episode = _episode()
    result = _history_raw_actions(episode, snapshot_step)
    indices = (snapshot_step - 10, snapshot_step - 5, snapshot_step)
    expected = np.stack(
        [episode.actions[start:stop] for start, stop in zip(indices[:-1], indices[1:])]
    )
    assert result.shape == (2, 5, 4)
    np.testing.assert_array_equal(result, expected)


def test_candidate_bank_is_namespaced_same_episode_and_slot_permuted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode = _episode()
    config = _config()
    monkeypatch.setattr(
        "fcpg.lewm_counterfactual_v2_runtime.policy_action_chunk",
        lambda plan, frozen: np.zeros((25, 4), dtype=np.float32),
    )

    first = build_episode_candidate_bank(episode, 10, SimpleNamespace(), config)
    second = build_episode_candidate_bank(episode, 10, SimpleNamespace(), config)
    np.testing.assert_array_equal(first.actions, second.actions)
    assert first.names == second.names
    assert set(first.names) == set(config["evaluation"]["candidate_names"])
    assert first.expert_index == first.names.index("expert")

    expert = episode.actions[10:35]
    shuffled = first.actions[first.names.index("same_task_shuffled")]
    assert not np.array_equal(shuffled, expert)
    np.testing.assert_array_equal(
        np.sort(shuffled, axis=0),
        np.sort(expert, axis=0),
    )
    with pytest.raises(TypeError, match="donor"):
        build_episode_candidate_bank(
            episode,
            10,
            SimpleNamespace(),
            config,
            donor=_episode(),
        )


def test_candidate_execution_is_exactly_once_and_labels_have_no_view_axis() -> None:
    candidates = ActionCandidates(
        names=("b", "a", "c"),
        actions=np.arange(3 * 25 * 4, dtype=np.float32).reshape(3, 25, 4),
        expert_index=1,
    )
    calls: list[str] = []

    def executor(episode: Any, step: int, action: np.ndarray, config: Any) -> PhysicalBranchOutcome:
        calls.append(schema_sha256_array(action))
        value = float(len(calls))
        return PhysicalBranchOutcome(
            endpoint_snapshots=tuple(_snapshot(index) for index in range(5)),
            endpoint_observations=tuple(np.zeros(4, dtype=np.float32) for _ in range(5)),
            cumulative_return=np.full(5, value, dtype=np.float32),
            progress=np.full(5, value / 10.0, dtype=np.float32),
            success=np.full(5, value > 2, dtype=bool),
            action_sha256=sha256_array(action),
        )

    outcomes = execute_candidate_bank_once(
        SimpleNamespace(), 10, candidates, {}, executor=executor
    )
    assert len(calls) == len(candidates.names)
    assert len(set(calls)) == len(candidates.names)
    assert tuple(outcomes) == candidates.names
    returns, progress, success = camera_invariant_branch_labels(candidates, outcomes)
    assert returns.shape == progress.shape == success.shape == (3, 5)
    assert returns.ndim == 2


class _FakeVision:
    model_name = "fake"
    selected_layers = (11,)

    def __init__(self) -> None:
        self.backbone = torch.nn.Linear(1, 1, bias=False)

    def extract_backbone_features(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = pixels.float().mean(dim=(1, 2, 3))
        index = torch.arange(64, device=pixels.device, dtype=torch.float32)
        prefix = (
            torch.cos(index)[None]
            + mean[:, None] * index[None] / 64.0
            + mean[:, None].square() * (index.remainder(3))[None]
        )
        features = torch.zeros(pixels.shape[0], 1, 64, 2, 2, device=pixels.device)
        return features, prefix


def test_goal_visual_uses_only_current_rgb_frozen_vit_and_is_deterministic() -> None:
    frozen = SimpleNamespace(
        device=torch.device("cpu"),
        corrector=SimpleNamespace(vision=_FakeVision()),
    )
    config = _config()
    current = np.linspace(0.0, 1.0, 3 * 224 * 224, dtype=np.float32).reshape(
        3, 224, 224
    )
    first = goal_visual_from_current_rgb(current, frozen, config)
    second = goal_visual_from_current_rgb(current.copy(), frozen, config)
    changed = goal_visual_from_current_rgb(current * current, frozen, config)
    assert first.shape == (48,)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, changed)


class _FakeLeWM:
    def __init__(self, offset: float) -> None:
        self.offset = float(offset)

    def encode_frames(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        assert "candidate_names" not in batch
        assert all(torch.is_tensor(value) for value in batch.values())
        grounded = 1.0 if "q_corrected" in batch else 0.0
        view = float(batch["pixels"].float().mean())
        time = torch.arange(3, dtype=torch.float32).view(1, 3, 1)
        embedding = torch.full((1, 3, 192), self.offset + grounded + view) + time
        return embedding, embedding

    def rollout_embeddings(
        self,
        history_embedding: torch.Tensor,
        history_action: torch.Tensor,
        future_action: torch.Tensor,
    ) -> torch.Tensor:
        base = history_embedding[:, -1:, :].unsqueeze(1)
        signal = future_action.mean(dim=-1, keepdim=True)
        return base + signal


def test_lewm_feature_grid_closes_model_grounding_view_candidate_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = {
        view: [_record(index * 0.1 + view_index) for index in range(3)]
        for view_index, view in enumerate(VIEW_NAMES)
    }
    monkeypatch.setattr(
        "fcpg.lewm_counterfactual_v2_runtime.corrected_points",
        lambda rows, frozen: np.stack([row.q_nominal + 1.0 for row in rows]),
    )
    candidates = ActionCandidates(
        names=("one", "two"),
        actions=np.stack(
            [np.zeros((25, 4), dtype=np.float32), np.ones((25, 4), dtype=np.float32)]
        ),
        expert_index=0,
    )
    frozen = SimpleNamespace(
        device=torch.device("cpu"),
        models={"nominal": _FakeLeWM(10.0), "fcpg": _FakeLeWM(20.0)},
    )
    corrected, valid, history, future = extract_lewm_feature_grid(
        records,
        np.zeros((2, 5, 4), dtype=np.float32),
        candidates,
        frozen,
    )
    assert corrected.shape == (3, 3, 3, 2)
    assert valid.shape == (3, 3, 3)
    assert history.shape == (2, 2, 3, 3, 192)
    assert future.shape == (2, 2, 3, 2, 5, 192)
    assert not np.array_equal(history[0], history[1])
    assert not np.array_equal(history[:, 0], history[:, 1])
    assert not np.array_equal(future[..., 0, :, :], future[..., 1, :, :])


def _endpoint_rgb_record(view_index: int, candidate_index: int, endpoint_index: int) -> FrameRecord:
    record = _record()
    ramp = np.linspace(0.0, 0.4, 224, dtype=np.float32)
    rgb = np.broadcast_to(ramp[None, None, :], (3, 224, 224)).copy()
    rgb += 0.1 * view_index + 0.01 * candidate_index + 0.001 * endpoint_index
    return replace(record, rgb=rgb)


def test_oracle_endpoint_grid_persists_observability_and_exact_pairing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    candidates = ActionCandidates(
        names=("a", "b"),
        actions=np.zeros((2, 25, 4), dtype=np.float32),
        expert_index=0,
    )
    outcomes = {name: SimpleNamespace(name=name) for name in candidates.names}
    candidate_index = {name: index for index, name in enumerate(candidates.names)}

    def render(
        episode: Any,
        outcome: Any,
        endpoint_index: int,
        condition: Any,
        config_value: Any,
    ) -> tuple[FrameRecord, str, str]:
        view_index = next(
            index
            for index, view in enumerate(VIEW_NAMES)
            if condition is config["conditions"][view]
        )
        slot = candidate_index[outcome.name]
        return (
            _endpoint_rgb_record(view_index, slot, endpoint_index),
            f"physical-{slot}-{endpoint_index}",
            f"rng-{slot}-{endpoint_index}",
        )

    monkeypatch.setattr(
        "fcpg.lewm_counterfactual_v2_runtime.render_endpoint_record",
        render,
    )
    monkeypatch.setattr(
        "fcpg.lewm_counterfactual_v2_runtime.oracle_targets",
        lambda records, frozen, image_size: np.zeros((len(records), 192), dtype=np.float32),
    )

    oracle, physical, rng, dynamic_range, changed_pixels, digests = (
        render_oracle_future_grid(
            SimpleNamespace(),
            candidates,
            outcomes,
            SimpleNamespace(),
            config,
        )
    )

    assert oracle.shape == (3, 2, 5, 192)
    assert physical[0] == physical[1] == physical[2]
    assert rng[0] == rng[1] == rng[2]
    assert dynamic_range.shape == (3, 2, 5)
    assert dynamic_range.dtype == np.uint16
    assert np.all(dynamic_range > 0)
    assert changed_pixels.shape == (2, 2, 5)
    assert changed_pixels.dtype == np.uint32
    assert np.all(changed_pixels > 0)
    assert digests.shape == (3, 2, 5, 32)
    assert digests.dtype == np.uint8
    assert np.all(np.any(digests != 0, axis=-1))
    expected_digest = np.frombuffer(
        bytes.fromhex(
            schema_sha256_array(_rgb_uint8(_endpoint_rgb_record(2, 1, 4)))
        ),
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(digests[2, 1, 4], expected_digest)


@pytest.mark.parametrize("failure", ("constant", "identical", "physical", "rng"))
def test_oracle_endpoint_grid_fails_closed_on_bad_rerender(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    config = _config()
    candidates = ActionCandidates(
        names=("a",),
        actions=np.zeros((1, 25, 4), dtype=np.float32),
        expert_index=0,
    )
    outcomes = {"a": SimpleNamespace(name="a")}

    def render(
        episode: Any,
        outcome: Any,
        endpoint_index: int,
        condition: Any,
        config_value: Any,
    ) -> tuple[FrameRecord, str, str]:
        view_index = next(
            index
            for index, view in enumerate(VIEW_NAMES)
            if condition is config["conditions"][view]
        )
        if failure == "constant":
            record = _record(0.0)
        else:
            record = _endpoint_rgb_record(
                0 if failure == "identical" else view_index,
                0,
                endpoint_index,
            )
        physical = f"physical-{endpoint_index}"
        if failure == "physical" and view_index == 1 and endpoint_index == 0:
            physical += "-changed"
        rng = f"rng-{endpoint_index}"
        if failure == "rng" and view_index == 1 and endpoint_index == 0:
            rng += "-changed"
        return record, physical, rng

    monkeypatch.setattr(
        "fcpg.lewm_counterfactual_v2_runtime.render_endpoint_record",
        render,
    )
    monkeypatch.setattr(
        "fcpg.lewm_counterfactual_v2_runtime.oracle_targets",
        lambda records, frozen, image_size: np.zeros((len(records), 192), dtype=np.float32),
    )

    match = {
        "constant": "constant frame",
        "identical": "pixel-identical",
        "physical": "exact physical states",
        "rng": "exact RNG states",
    }[failure]
    with pytest.raises(RuntimeError, match=match):
        render_oracle_future_grid(
            SimpleNamespace(),
            candidates,
            outcomes,
            SimpleNamespace(),
            config,
        )


def test_camera_invariant_history_fails_closed_on_view_dependent_geometry() -> None:
    records = {view: [_record(float(index)) for index in range(3)] for view in VIEW_NAMES}
    reference, proprio, xyz, q_nominal, valid = _camera_invariant_history(records)
    assert len(reference) == 3
    assert proprio.shape == (3, 4)
    assert xyz.shape == (3, 3, 3)
    assert q_nominal.shape == (3, 3, 2)
    assert valid.shape == (3, 3)
    records["rotation5"][1].q_nominal[0, 0] += 1.0
    with pytest.raises(RuntimeError, match="q_nominal"):
        _camera_invariant_history(records)


def test_index_row_binds_actual_slots_and_measured_per_view_state() -> None:
    episode = _episode()
    config = _config()
    names = tuple(reversed(config["evaluation"]["candidate_names"]))
    actions = np.arange(len(names) * 25 * 4, dtype=np.float32).reshape(
        len(names), 25, 4
    )
    candidates = ActionCandidates(
        names=names,
        actions=actions,
        expert_index=names.index("expert"),
    )
    outcomes = {
        name: PhysicalBranchOutcome(
            endpoint_snapshots=tuple(_snapshot(index + slot * 10) for index in range(5)),
            endpoint_observations=tuple(np.zeros(4, dtype=np.float32) for _ in range(5)),
            cumulative_return=np.zeros(5, dtype=np.float32),
            progress=np.zeros(5, dtype=np.float32),
            success=np.zeros(5, dtype=bool),
            action_sha256=sha256_array(actions[slot]),
        )
        for slot, name in enumerate(names)
    }
    history_steps = (0, 5, 10)
    history_physical = [
        [physical_snapshot_sha256(episode.snapshots[step]) for step in history_steps]
        for _ in VIEW_NAMES
    ]
    history_rng = [
        [rng_snapshot_sha256(episode.snapshots[step]) for step in history_steps]
        for _ in VIEW_NAMES
    ]
    endpoint_physical = [
        [
            [physical_snapshot_sha256(snapshot) for snapshot in outcomes[name].endpoint_snapshots]
            for name in names
        ]
        for _ in VIEW_NAMES
    ]
    endpoint_rng = [
        [
            [rng_snapshot_sha256(snapshot) for snapshot in outcomes[name].endpoint_snapshots]
            for name in names
        ]
        for _ in VIEW_NAMES
    ]
    candidate_count = len(names)
    actual_return = np.zeros((candidate_count, 5), dtype=np.float32)
    actual_progress = np.zeros((candidate_count, 5), dtype=np.float32)
    actual_success = np.zeros((candidate_count, 5), dtype=bool)
    oracle_future_latent = np.zeros(
        (len(VIEW_NAMES), candidate_count, 5, 192), dtype=np.float32
    )
    endpoint_rgb_sha256 = np.ones(
        (len(VIEW_NAMES), candidate_count, 5, 32), dtype=np.uint8
    )
    row = _sample_index_row(
        episode,
        phase_index=0,
        snapshot_step=10,
        candidates=candidates,
        outcomes=outcomes,
        history_physical_hashes_by_view=history_physical,
        history_rng_hashes_by_view=history_rng,
        endpoint_physical_hashes_by_view=endpoint_physical,
        endpoint_rng_hashes_by_view=endpoint_rng,
        actual_return=actual_return,
        actual_progress=actual_progress,
        actual_success=actual_success,
        oracle_future_latent=oracle_future_latent,
        endpoint_rgb_sha256=endpoint_rgb_sha256,
        config=config,
    )
    assert set(row) == set(INDEX_ROW_KEYS)
    assert row["candidate_names"] == list(names)
    assert row["candidate_permutation"] == [
        config["evaluation"]["candidate_names"].index(name) for name in names
    ]
    assert row["candidate_action_sha256"] == [
        schema_sha256_array(action) for action in actions
    ]
    assert row["branch_execution_count"] == len(names)
    assert row["endpoint_physical_sha256_by_view"][0] == row[
        "endpoint_physical_sha256_by_view"
    ][2]
    assert row["endpoint_rng_sha256_by_view"][0] == row[
        "endpoint_rng_sha256_by_view"
    ][1]

    tampered = copy.deepcopy(endpoint_physical)
    tampered[1][0][0] = "0" * 64
    with pytest.raises(RuntimeError, match="measured endpoint physical"):
        _sample_index_row(
            episode,
            phase_index=0,
            snapshot_step=10,
            candidates=candidates,
            outcomes=outcomes,
            history_physical_hashes_by_view=history_physical,
            history_rng_hashes_by_view=history_rng,
            endpoint_physical_hashes_by_view=tampered,
            endpoint_rng_hashes_by_view=endpoint_rng,
            actual_return=actual_return,
            actual_progress=actual_progress,
            actual_success=actual_success,
            oracle_future_latent=oracle_future_latent,
            endpoint_rgb_sha256=endpoint_rgb_sha256,
            config=config,
        )


def _synthetic_phase_row(
    episode: Any,
    phase_index: int,
    frozen: Any,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    c = len(config["evaluation"]["candidate_names"])
    task_index = TASK_NAMES.index(episode.task)
    q = np.full((3, 3, 3, 2), 32.0, dtype=np.float32)
    xyz = np.broadcast_to(
        np.asarray([32.0, 32.0, 1.0], dtype=np.float32),
        (3, 3, 3),
    ).copy()
    nominal_extrinsic = np.eye(4, dtype=np.float32)
    true_extrinsic = expected_camera_condition_extrinsics(
        nominal_extrinsic[None],
        config["conditions"],
    )[0].astype(np.float32)

    def project(points: np.ndarray, extrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        homogeneous = np.concatenate(
            [points, np.ones((*points.shape[:-1], 1), dtype=np.float32)],
            axis=-1,
        )
        camera = np.einsum("ij,...j->...i", extrinsic[:3], homogeneous)
        pixel = camera[..., :2] / camera[..., 2:3]
        valid = (
            (camera[..., 2] > 1e-6)
            & (pixel[..., 0] >= 0.0)
            & (pixel[..., 0] < 224.0)
            & (pixel[..., 1] >= 0.0)
            & (pixel[..., 1] < 224.0)
        )
        return pixel.astype(np.float32), valid

    q_true_rows = [project(xyz, extrinsic) for extrinsic in true_extrinsic]
    q_true = np.stack([row[0] for row in q_true_rows])
    valid_true = np.stack([row[1] for row in q_true_rows])
    goal_world = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    goal_rows = [project(goal_world[None], extrinsic) for extrinsic in true_extrinsic]
    goal_pixel = np.stack([row[0][0] for row in goal_rows])
    goal_valid = np.asarray([row[1][0] for row in goal_rows], dtype=bool)
    history_rgb = np.zeros((3, 3, 224, 224, 3), dtype=np.uint8)
    for view in range(3):
        history_rgb[view] = 20 + 20 * view
        history_rgb[view, :, 0, 0, 0] += 1
    deployment = {
        "task_index": np.asarray(task_index, dtype=np.uint8),
        "history_rgb": history_rgb,
        "history_proprio": np.zeros((3, 4), dtype=np.float32),
        "history_keypoints_xyz": xyz,
        "history_q_nominal": np.full((3, 3, 2), 32.0, dtype=np.float32),
        "history_valid_nominal": np.ones((3, 3), dtype=bool),
        "history_q_corrected": q,
        "history_valid_grounded": np.ones((3, 3, 3), dtype=bool),
        "history_action_raw": np.zeros((2, 5, 4), dtype=np.float32),
        "candidate_action_raw": np.zeros((c, 25, 4), dtype=np.float32),
        "z_history": np.zeros((2, 2, 3, 3, 192), dtype=np.float32),
        "z_pred_future": np.zeros((2, 2, 3, c, 5, 192), dtype=np.float32),
        "goal_visual": np.zeros((3, 48), dtype=np.float32),
        "task_onehot": np.eye(4, dtype=np.float32)[task_index],
    }
    labels = {
        "actual_return": np.zeros((c, 5), dtype=np.float32),
        "actual_progress": np.zeros((c, 5), dtype=np.float32),
        "actual_success": np.zeros((c, 5), dtype=bool),
        "oracle_future_latent": np.zeros((3, c, 5, 192), dtype=np.float32),
        "history_q_true": q_true,
        "history_valid_true": valid_true,
        "goal_world": goal_world,
        "goal_pixel": goal_pixel,
        "goal_valid": goal_valid,
    }
    audit = {
        "episode_id": np.asarray(episode.episode, dtype=np.int32),
        "env_seed": np.asarray(episode.env_seed, dtype=np.int64),
        "phase_index": np.asarray(phase_index, dtype=np.uint8),
        "snapshot_step": np.asarray(PHASE_STEPS[phase_index], dtype=np.int16),
        "camera_intrinsic": np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0),
        "true_extrinsic": true_extrinsic,
        "nominal_extrinsic": nominal_extrinsic,
        "endpoint_rgb_dynamic_range": np.full((3, c, 5), 100, dtype=np.uint16),
        "endpoint_rgb_changed_pixels": np.full((2, c, 5), 100, dtype=np.uint32),
        "endpoint_rgb_sha256": np.broadcast_to(
            np.arange(1, 4, dtype=np.uint8)[:, None, None, None],
            (3, c, 5, 32),
        ).copy(),
    }
    family = list(config["evaluation"]["candidate_names"])
    endpoint_physical = [
        [["6" * 64 for _ in range(5)] for _ in range(c)] for _ in range(3)
    ]
    endpoint_rng = [
        [["7" * 64 for _ in range(5)] for _ in range(c)] for _ in range(3)
    ]
    receipt_fields = build_branch_receipt_fields(
        candidate_names=family,
        candidate_actions=deployment["candidate_action_raw"],
        actual_return=labels["actual_return"],
        actual_progress=labels["actual_progress"],
        actual_success=labels["actual_success"],
        oracle_future_latent=labels["oracle_future_latent"],
        endpoint_physical_sha256_by_view=endpoint_physical,
        endpoint_rng_sha256_by_view=endpoint_rng,
        endpoint_rgb_sha256=audit["endpoint_rgb_sha256"],
    )
    index = {
        name: value
        for name, value in zip(
            INDEX_ROW_KEYS,
            [None] * len(INDEX_ROW_KEYS),
        )
    }
    index.update(
        {
            "sample_index": -1,
            "task_index": task_index,
            "task_name": episode.task,
            "episode_id": episode.episode,
            "env_seed": episode.env_seed,
            "phase_index": phase_index,
            "phase_name": ("early", "mid", "late")[phase_index],
            "snapshot_step": PHASE_STEPS[phase_index],
            "split": "pending",
            "physical_episode_sha256": physical_episode_sha256(
                task_index, episode.episode, episode.env_seed
            ),
            "snapshot_sha256": "1" * 64,
            "history_physical_sha256_by_view": [["2" * 64] * 3] * 3,
            "history_rng_sha256_by_view": [["3" * 64] * 3] * 3,
            "view_names": list(VIEW_NAMES),
            "candidate_names": family,
            "candidate_permutation": list(range(c)),
            "candidate_permutation_sha256": canonical_sha256(list(range(c))),
            "candidate_action_sha256": [schema_sha256_array(row) for row in deployment["candidate_action_raw"]],
            "candidate_action_set_sha256": canonical_sha256(
                {
                    "members": sorted(
                        [
                            {"name": name, "action_sha256": schema_sha256_array(action)}
                            for name, action in zip(family, deployment["candidate_action_raw"])
                        ],
                        key=lambda item: item["name"],
                    )
                }
            ),
            **receipt_fields,
            "branch_execution_count": c,
            "endpoint_physical_sha256_by_view": endpoint_physical,
            "endpoint_rng_sha256_by_view": endpoint_rng,
        }
    )
    return deployment, labels, audit, index


def test_run_collection_stacks_canonical_payload_and_exact_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config()
    state = {
        "nominal": "a" * 64,
        "fcpg": "b" * 64,
        "action_only_policy": "c" * 64,
        "fcpg_corrector": "d" * 64,
        "oracle_target_encoder": "e" * 64,
        "oracle_vision_backbone": "f" * 64,
    }
    frozen = SimpleNamespace(
        target_encoder=SimpleNamespace(visual_dim=GOAL_VISUAL_INPUT_DIM)
    )
    monkeypatch.setattr(
        "fcpg.lewm_counterfactual_v2_runtime.load_frozen_inputs",
        lambda config, repo_root: frozen,
    )
    monkeypatch.setattr(
        "fcpg.lewm_counterfactual_v2_runtime._current_frozen_state",
        lambda frozen: dict(state),
    )
    def input_provenance(config: dict[str, Any], frozen: Any) -> dict[str, Any]:
        def leaf(value: dict[str, str]) -> dict[str, str]:
            return {
                "path": value["path"],
                "expected_sha256": value["sha256"],
                "actual_sha256": value["sha256"],
            }

        return {
            "lewm_checkpoints": {
                name: leaf(value)
                for name, value in config["inputs"]["lewm_checkpoints"].items()
            },
            "action_only_policy": leaf(config["inputs"]["action_only_policy"]),
            "fcpg_corrector": leaf(config["inputs"]["fcpg_corrector"]),
            "oracle_target_encoder": leaf(config["inputs"]["oracle_target_encoder"]),
        }

    monkeypatch.setattr(
        "fcpg.lewm_counterfactual_v2_runtime._input_provenance",
        input_provenance,
    )
    monkeypatch.setattr(
        "fcpg.lewm_counterfactual_v2_runtime._git_state",
        lambda root: {"head": "abc", "dirty": False},
    )
    monkeypatch.setattr(
        "fcpg.lewm_counterfactual_v2_runtime._versions",
        lambda: {"python": "test"},
    )
    class WeakEpisode:
        pass

    previous_episode: weakref.ReferenceType[Any] | None = None

    def collect_episode(task: str, episode: int, config: Any) -> WeakEpisode:
        nonlocal previous_episode
        if previous_episode is not None:
            assert previous_episode() is None
        result = WeakEpisode()
        result.task = task
        result.episode = episode
        result.env_seed = 30_000 + TASK_NAMES.index(task) * 1_000 + episode
        result.env = SimpleNamespace(close=lambda: None)
        previous_episode = weakref.ref(result)
        return result

    monkeypatch.setattr(
        "fcpg.lewm_counterfactual_v2_runtime.collect_physical_episode",
        collect_episode,
    )
    monkeypatch.setattr(
        "fcpg.lewm_counterfactual_v2_runtime.collect_phase_row",
        _synthetic_phase_row,
    )

    payload = run_counterfactual_v2_collection(config, repo_root=tmp_path)
    axes = validate_array_bundle(
        payload.deployment_inputs,
        payload.offline_labels,
        payload.audit_arrays,
        candidate_count=4,
    )
    assert axes.sample_count == 12
    for metadata_name in ("episode_id", "env_seed", "phase_index", "snapshot_step"):
        assert metadata_name not in payload.deployment_inputs
        assert metadata_name in payload.audit_arrays
    assert len(payload.index_rows) == 12
    assert [row["sample_index"] for row in payload.index_rows] == list(range(12))
    assert {row["split"] for row in payload.index_rows} == {"smoke"}
    assert set(payload.provenance) == set(PROVENANCE_KEYS)
    assert payload.provenance["model_state_before"] == state
    assert payload.provenance["model_state_after"] == state
    assert payload.provenance["upstream_non_overlap"]["status"] == "not_auditable"
    assert previous_episode is not None and previous_episode() is None
    audit = validate_index_and_provenance(
        config,
        payload.deployment_inputs,
        payload.offline_labels,
        payload.audit_arrays,
        payload.index_rows,
        payload.provenance,
        smoke=True,
    )
    assert audit["passed"] is True
    assert audit["g0_eligible"] is False
    assert_no_privileged_inputs(payload.deployment_inputs)
