from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.lewm_a4_storage import (
    PROPOSAL_LEDGER_KEYS,
    STATE_CACHE_KEYS,
    validate_a4_ledger_cache_pair,
)
from fcpg.lewm_action_branch_a4 import BRANCH_COUNT, TASKS
from fcpg.lewm_action_branch_a4_collection import (
    A4EpisodeRuntime,
    DIAGNOSTIC_KEYS,
    PhysicalTarget,
    _physical_support_pose,
    a4_camera_name,
    collect_a4_source_episode,
    encode_a4_state_records,
    load_frozen_a4_state_encoder,
    recent_xyz_action_rms,
    save_a4_diagnostics,
    stack_a4_collection,
)
from fcpg.lewm_counterfactual_runtime import FrameRecord
from scripts import collect_lewm_action_branch_a4 as cli


ROOT = Path(__file__).resolve().parents[1]


def test_recent_xyz_action_rms_uses_vector_magnitude_not_coordinate_mean() -> None:
    actions = np.full((5, 3), 1.0 / np.sqrt(3.0), dtype=np.float32)

    assert recent_xyz_action_rms(np.linalg.norm(actions, axis=1)) == pytest.approx(1.0)
    coordinate_rms = float(np.sqrt(np.mean(np.square(actions))))
    assert coordinate_rms == pytest.approx(1.0 / np.sqrt(3.0))


@pytest.mark.parametrize("values", [[], [0.1, float("nan")], [0.1, -0.1]])
def test_recent_xyz_action_rms_rejects_invalid_windows(values) -> None:
    with pytest.raises(ValueError):
        recent_xyz_action_rms(values)


def test_reach_physical_support_uses_tcp_instead_of_static_object() -> None:
    target = PhysicalTarget(
        object_pose=np.asarray([9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        reference_pose=np.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        goal_progress=0.25,
        contact_or_grasp=False,
        branch_outcome=np.asarray([0.25, 0.0], dtype=np.float32),
        tcp=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        qpos=np.zeros(64, dtype=np.float32),
        qpos_size=9,
        gripper_aperture=0.04,
    )

    np.testing.assert_allclose(
        _physical_support_pose("reach-v3", target),
        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
    )
    np.testing.assert_allclose(
        _physical_support_pose("pick-place-v3", target), target.object_pose
    )


def _contract() -> dict:
    return json.loads(
        (ROOT / "config/lewm/eval/action_branch_active_control_a4.json").read_text(
            encoding="utf-8"
        )
    )


class _ActionSpace:
    low = np.full(4, -1.0, dtype=np.float32)
    high = np.full(4, 1.0, dtype=np.float32)


class _Env:
    def __init__(self) -> None:
        self.action_space = _ActionSpace()
        self.t = 0
        self.position = np.zeros(3, dtype=np.float32)
        self.tcp_center = np.zeros(3, dtype=np.float32)
        self.data = SimpleNamespace(qpos=np.zeros(6, dtype=np.float32))
        self.closed = False

    def observation(self) -> np.ndarray:
        result = np.zeros(39, dtype=np.float32)
        result[:3] = self.position
        result[3] = float(self.t) / 100.0
        return result

    def step(self, action: np.ndarray):
        self.t += 1
        self.position += np.asarray(action[:3], dtype=np.float32) * 0.01
        self.tcp_center = self.position.copy()
        self.data.qpos[:3] = self.position
        return self.observation(), float(self.position[0]), False, self.t >= 60, {
            "success": 0.0,
            "in_place_reward": float(np.clip(self.position[0], 0.0, 1.0)),
        }

    def close(self) -> None:
        self.closed = True


class _Expert:
    def get_action(self, _observation: np.ndarray) -> np.ndarray:
        return np.asarray([0.2, 0.1, 0.0, -0.5], dtype=np.float32)


def _record(env: _Env, observation: np.ndarray) -> FrameRecord:
    nominal = np.asarray([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], dtype=np.float32)
    return FrameRecord(
        rgb=np.zeros((3, 8, 8), dtype=np.uint8),
        proprio=np.asarray(observation[:4], dtype=np.float32),
        keypoints_xyz=np.zeros((3, 3), dtype=np.float32),
        q_nominal=nominal,
        valid_nominal=np.ones(3, dtype=bool),
        q_true=nominal.copy(),
        valid_true=np.ones(3, dtype=bool),
    )


def _signals(_task: str, env: _Env, observation, info):
    success = bool(float(info.get("success", 0.0)))
    progress = float(np.clip(env.position[0], 0.0, 1.0))
    diagnostics = SimpleNamespace(
        object_pose=np.asarray([*env.position, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        reference_pose=np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        goal_relative_progress=np.float32(progress),
        contact_or_grasp_state=np.bool_(False),
        branch_outcome=np.asarray([progress, float(success)], dtype=np.float32),
    )
    fields = {
        "success": success,
        "active_stage": not success,
        "workspace_and_joint_margin": 0.5,
        "unexpected_collision": False,
    }
    return SimpleNamespace(
        diagnostics=diagnostics,
        deployment=SimpleNamespace(
            proprio=np.asarray(observation[:4], dtype=np.float32)
        ),
        eligibility_fields=lambda: fields,
    )


def _encode(records) -> np.ndarray:
    result = np.zeros((len(records), 4, 192), dtype=np.float32)
    for index, record in enumerate(records):
        result[index] = float(record.proprio[3]) + np.arange(4)[:, None]
    return result


def _snapshot(env: _Env, *, include_process_rng: bool):
    assert include_process_rng
    return env.t, env.position.copy(), env.data.qpos.copy()


def _restore(env: _Env, snapshot) -> None:
    env.t, position, qpos = snapshot
    env.position = position.copy()
    env.tcp_center = position.copy()
    env.data.qpos[:] = qpos


def test_one_scheduled_proposal_executes_all_branches_and_packs_exact_schemas(
    tmp_path: Path,
) -> None:
    order: list[str] = []
    environments: list[_Env] = []

    def schedule(task, instance, *, contract):
        assert task == TASKS[0]
        assert instance == 0
        assert contract["tasks"] == list(TASKS)
        order.append("schedule")
        return (10,)

    def runtime_factory(task, instance, seed):
        assert (task, instance, seed) == (TASKS[0], 0, 620000)
        order.append("runtime")
        env = _Env()
        environments.append(env)
        observation = env.observation()
        return A4EpisodeRuntime(
            env=env,
            expert=_Expert(),
            observation=observation,
            reset_info={"success": 0.0, "in_place_reward": 0.0},
            record_current=lambda current: _record(env, current),
            max_raw_steps=60,
            camera_name="corner",
            camera_intrinsic=np.eye(3, dtype=np.float32),
            camera_extrinsic_cam_from_world=np.eye(4, dtype=np.float32),
        )

    proposals, bundles = collect_a4_source_episode(
        task=TASKS[0],
        task_class_index=0,
        task_instance_index=0,
        environment_seed=620000,
        split="train",
        contract=_contract(),
        runtime_factory=runtime_factory,
        encode_records=_encode,
        schedule_sampler=schedule,
        snapshot_capture=_snapshot,
        snapshot_restore=_restore,
        signal_extractor=_signals,
    )

    assert order == ["schedule", "runtime"]
    assert environments[0].closed
    assert len(proposals) == len(bundles) == 1
    assert proposals[0].eligible
    assert bundles[0].target_state.shape == (8, 8, 4, 192)
    assert bundles[0].future_action.shape == (8, 8, 20)
    assert sorted(bundles[0].branch_semantic_index.tolist()) == list(range(8))

    ledger, cache, diagnostics = stack_a4_collection(proposals, bundles)
    assert tuple(ledger) == PROPOSAL_LEDGER_KEYS
    assert tuple(cache) == STATE_CACHE_KEYS
    assert tuple(diagnostics) == DIAGNOSTIC_KEYS
    assert cache["history_state"].shape == (1, 3, 4, 192)
    assert cache["target_state"].shape == (1, 8, 8, 4, 192)
    assert diagnostics["physical_state"].shape == (1, 8, 8, 7)
    assert diagnostics["branch_gripper_aperture"].shape == (1, 8, 8)
    assert diagnostics["branch_contact_or_grasp"].shape == (1, 8, 8)
    assert diagnostics["branch_outcome"].shape == (1, 8, 8, 2)
    assert diagnostics["contact_or_grasp_state"].shape == (1, 1)
    validate_a4_ledger_cache_pair(ledger, cache)

    output = save_a4_diagnostics(tmp_path / "preflight_labels.npz", diagnostics)
    with np.load(output, allow_pickle=False) as archive:
        assert tuple(archive.files) == DIAGNOSTIC_KEYS
        np.testing.assert_array_equal(archive["proposal_id"], cache["proposal_id"])


@pytest.mark.parametrize(
    ("task", "camera"),
    [
        ("pick-place-v3", "corner"),
        ("drawer-retrieve-v0", "drawer_view"),
        ("reach-v3", "corner"),
        ("push-v3", "corner"),
        ("door-open-v3", "corner"),
        ("window-open-v3", "corner"),
        ("towel-wipe-v0", "towel_view"),
    ],
)
def test_formal_camera_mapping(task: str, camera: str) -> None:
    assert a4_camera_name(task) == camera


def test_frozen_encoder_selects_summary_and_three_anchor_tokens() -> None:
    records = [_record(_Env(), np.zeros(39, dtype=np.float32)) for _ in range(2)]

    class Corrector:
        def __call__(self, _rgb, nominal, _xyz, _proprio):
            return SimpleNamespace(q_corrected_px=nominal)

    class FrameEncoder:
        def forward_token_sequence(self, **kwargs):
            count = kwargs["pixels"].shape[0]
            values = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
            return values.expand(count, 6, 192).clone()

    encoded = encode_a4_state_records(
        records,
        frame_encoder=FrameEncoder(),
        projector=torch.nn.Identity(),
        proprio_mean=torch.zeros(4),
        proprio_std=torch.ones(4),
        corrector=Corrector(),
        device=torch.device("cpu"),
    )
    assert encoded.shape == (2, 4, 192)
    np.testing.assert_array_equal(encoded[0, :, 0], [0.0, 3.0, 4.0, 5.0])


def test_source_checkpoint_must_have_sigreg_002(monkeypatch: pytest.MonkeyPatch) -> None:
    frame_encoder = torch.nn.Linear(1, 1)
    frame_encoder.num_keypoints = 3
    model = SimpleNamespace(
        latent_dim=192,
        history_size=3,
        action_dim=20,
        frame_encoder=frame_encoder,
        projector=torch.nn.Linear(192, 192),
        proprio_mean=torch.zeros(4),
        proprio_std=torch.ones(4),
    )
    monkeypatch.setattr(
        "fcpg.lewm_runtime.load_lewm_grounded_checkpoint",
        lambda *_args, **_kwargs: (
            model,
            {"loss": {"sigreg_weight": 0.0}},
            {"resolved_config": {}},
        ),
    )
    with pytest.raises(ValueError, match="SIGReg=0.02"):
        load_frozen_a4_state_encoder("base.pt", "fcpg.pt", device="cpu")


def test_formal_cli_rejects_smoke_bounds_before_loading_files() -> None:
    with pytest.raises(ValueError, match="smoke-only"):
        cli.main(["--max-tasks", "1"])
