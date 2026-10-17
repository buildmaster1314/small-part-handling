from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.lewm_counterfactual_runtime import (
    BranchOutcome,
    SUMMARY_METRICS,
    FrameRecord,
    _candidate_rng_seed,
    _git_state,
    _history_records_and_actions,
    _verify_loaded_corrector_binding,
    assert_deployment_batch,
    build_lewm_history_batch,
    compare_branch_outcomes,
    corrected_points,
    grounded_valid_mask,
    physical_snapshot_sha256,
    pixel_points_inside,
    sha256_file,
    smoke_config,
    sha256_state_dict,
    summarize_counterfactual_rows,
    validate_candidate_replay,
    validate_pilot_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/lewm/eval/gate_l2_pilot_seed4096.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _record(value: float) -> FrameRecord:
    return FrameRecord(
        rgb=np.full((3, 8, 8), value, dtype=np.float32),
        proprio=np.full(4, value, dtype=np.float32),
        keypoints_xyz=np.full((3, 3), value, dtype=np.float32),
        q_nominal=np.full((3, 2), value, dtype=np.float32),
        valid_nominal=np.ones(3, dtype=bool),
        q_true=np.full((3, 2), value + 1.0, dtype=np.float32),
        valid_true=np.ones(3, dtype=bool),
    )


class _CenterEchoCorrector:
    def __init__(self) -> None:
        self.centers: torch.Tensor | None = None

    def __call__(
        self,
        rgb: torch.Tensor,
        centers: torch.Tensor,
        xyz: torch.Tensor,
        proprio: torch.Tensor,
    ) -> SimpleNamespace:
        del rgb, xyz, proprio
        self.centers = centers.detach().cpu().clone()
        return SimpleNamespace(q_corrected_px=centers + 0.5)


def test_corrected_points_accepts_explicit_search_centers() -> None:
    corrector = _CenterEchoCorrector()
    records = [_record(1.0), _record(2.0)]
    centers = np.stack([record.q_nominal + 3.0 for record in records])

    result = corrected_points(
        records,
        SimpleNamespace(corrector=corrector, device=torch.device("cpu")),
        centers_px=centers,
    )

    np.testing.assert_allclose(result, centers + 0.5)
    assert corrector.centers is not None
    np.testing.assert_allclose(corrector.centers.numpy(), centers)


def test_corrected_points_defaults_to_nominal_and_validates_centers() -> None:
    corrector = _CenterEchoCorrector()
    records = [_record(1.0)]
    frozen = SimpleNamespace(corrector=corrector, device=torch.device("cpu"))

    result = corrected_points(records, frozen)
    np.testing.assert_allclose(result, records[0].q_nominal[None] + 0.5)

    with pytest.raises(ValueError, match="centers must have shape"):
        corrected_points(records, frozen, centers_px=np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        corrected_points(
            records,
            frozen,
            centers_px=np.full((1, 3, 2), np.nan, dtype=np.float32),
        )


def test_frozen_config_and_smoke_subset_are_valid() -> None:
    config = _config()
    validate_pilot_config(config)
    smoke = smoke_config(config)

    assert smoke["runtime_mode"] == "smoke"
    assert smoke["summary_interpretation"] == "pipeline_only"
    assert smoke["evaluation"]["tasks"] == ["reach-v3"]
    assert smoke["evaluation"]["episodes_per_task"] == 2
    assert smoke["evaluation"]["candidate_names"] == config["evaluation"][
        "candidate_names"
    ]
    assert list(smoke["conditions"]) == ["clean", "rotation5"]
    assert list(smoke["inputs"]["lewm_checkpoints"]) == ["nominal", "fcpg"]
    assert smoke["output"]["result"].endswith("_smoke.json")


def test_candidate_seed_is_canonical_and_camera_condition_independent() -> None:
    full = _config()
    smoke = smoke_config(full)

    expected = _candidate_rng_seed(
        full, task="reach-v3", condition="clean", episode=1
    )
    assert _candidate_rng_seed(
        full, task="reach-v3", condition="rotation5", episode=1
    ) == expected
    assert _candidate_rng_seed(
        smoke, task="reach-v3", condition="clean", episode=1
    ) == expected


def test_config_rejects_training_or_unpinned_input() -> None:
    config = _config()
    config["scope"]["training_allowed"] = True
    with pytest.raises(ValueError, match="forbid training"):
        validate_pilot_config(config)

    config = _config()
    config["inputs"]["fcpg_corrector"]["sha256"] = "bad"
    with pytest.raises(ValueError, match="SHA-256"):
        validate_pilot_config(config)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("environment", "camera", "corner2"),
        ("environment", "benchmark_seed", 1),
        ("environment", "env_seed_base", 1),
        ("environment", "task_index_offset", 1),
        ("clean", "translation_world", [0.1, 0.0, 0.0]),
        ("rotation5", "rotation_axis_world", [1.0, 0.0, 0.0]),
    ],
)
def test_config_rejects_changed_environment_or_camera_protocol(
    section: str, field: str, value: object
) -> None:
    config = _config()
    target = (
        config["environment"]
        if section == "environment"
        else config["conditions"][section]
    )
    target[field] = value
    with pytest.raises(ValueError, match="changed|environment"):
        validate_pilot_config(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tasks", []),
        ("conditions", []),
        ("checkpoints", []),
        ("episodes_per_task", 0),
        ("snapshots_per_episode", 0),
        ("candidate_names", []),
        ("bootstrap_samples", 0),
    ],
)
def test_smoke_rejects_protocol_reductions(field: str, value: object) -> None:
    config = _config()
    config["smoke"][field] = value
    with pytest.raises(ValueError, match="smoke"):
        smoke_config(config)


def test_git_state_binds_dirty_worktree_content() -> None:
    state = _git_state(ROOT)
    assert len(state["worktree_content_sha256"]) == 64
    assert isinstance(state["untracked_files"], list)


def test_state_dict_hash_supports_scalar_integer_buffers() -> None:
    module = torch.nn.Module()
    module.register_buffer("num_batches_tracked", torch.tensor(7, dtype=torch.long))

    first = sha256_state_dict(module)
    second = sha256_state_dict(module)

    assert first == second
    assert len(first) == 64


def test_pixel_points_inside_and_grounded_mask_match_training_boundaries() -> None:
    points = np.asarray(
        [
            [0.0, 0.0],
            [7.0, 7.0],
            [-1e-6, 1.0],
            [8.0, 1.0],
            [1.0, -1e-6],
            [1.0, 8.0],
            [np.nan, 1.0],
            [1.0, np.inf],
        ],
        dtype=np.float32,
    )
    inside = pixel_points_inside(points, width=8, height=8)
    np.testing.assert_array_equal(
        inside,
        np.asarray([True, True, False, False, False, False, False, False]),
    )

    nominal = np.asarray([True, False, True, True, True, True, True, True])
    np.testing.assert_array_equal(
        grounded_valid_mask(nominal, points, width=8, height=8),
        nominal & inside,
    )


def test_history_batch_masks_corrected_points_outside_the_image() -> None:
    records = [_record(0.0), _record(1.0), _record(2.0)]
    records[1].valid_nominal[1] = False
    corrected = np.asarray(
        [
            [[0.0, 0.0], [7.0, 7.0], [8.0, 3.0]],
            [[1.0, 1.0], [2.0, 2.0], [3.0, -0.1]],
            [[7.0, 7.0], [np.nan, 1.0], [4.0, 4.0]],
        ],
        dtype=np.float32,
    )
    batch = build_lewm_history_batch(
        records,
        device=torch.device("cpu"),
        q_corrected=corrected,
    )

    np.testing.assert_array_equal(
        batch["valid"].numpy(),
        np.asarray(
            [
                [
                    [True, True, False],
                    [True, False, False],
                    [True, False, True],
                ]
            ]
        ),
    )


def test_verified_corrector_binding_records_actual_file_and_state(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "corrector.pt"
    checkpoint.write_bytes(b"verified-corrector")
    corrector = torch.nn.Linear(2, 2)
    policy = SimpleNamespace(fcpg_adapter=corrector)
    state_sha256 = sha256_state_dict(corrector)

    record = _verify_loaded_corrector_binding(
        policy,
        corrector,
        checkpoint_path=checkpoint,
        expected_file_sha256=sha256_file(checkpoint),
        state_sha256_before_binding=state_sha256,
    )

    assert record == {
        "loaded_path": str(checkpoint),
        "loaded_file_sha256": sha256_file(checkpoint),
        "loaded_state_sha256": state_sha256,
    }


def test_verified_corrector_binding_rejects_wrong_instance_or_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "corrector.pt"
    checkpoint.write_bytes(b"verified-corrector")
    corrector = torch.nn.Linear(2, 2)
    expected_file_sha256 = sha256_file(checkpoint)
    expected_state_sha256 = sha256_state_dict(corrector)

    with pytest.raises(RuntimeError, match="not bound"):
        _verify_loaded_corrector_binding(
            SimpleNamespace(fcpg_adapter=torch.nn.Linear(2, 2)),
            corrector,
            checkpoint_path=checkpoint,
            expected_file_sha256=expected_file_sha256,
            state_sha256_before_binding=expected_state_sha256,
        )

    with torch.no_grad():
        corrector.weight.add_(1.0)
    with pytest.raises(RuntimeError, match="state changed"):
        _verify_loaded_corrector_binding(
            SimpleNamespace(fcpg_adapter=corrector),
            corrector,
            checkpoint_path=checkpoint,
            expected_file_sha256=expected_file_sha256,
            state_sha256_before_binding=expected_state_sha256,
        )

    expected_state_sha256 = sha256_state_dict(corrector)
    checkpoint.write_bytes(b"changed-corrector")
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        _verify_loaded_corrector_binding(
            SimpleNamespace(fcpg_adapter=corrector),
            corrector,
            checkpoint_path=checkpoint,
            expected_file_sha256=expected_file_sha256,
            state_sha256_before_binding=expected_state_sha256,
        )


def test_paired_physical_hash_excludes_camera_and_rng_state() -> None:
    base = SimpleNamespace(
        integration_state=np.asarray([1.0, 2.0]),
        warmstart_state=np.asarray([3.0]),
        model_arrays=SimpleNamespace(
            body_pos=np.asarray([[1.0, 2.0, 3.0]]),
            site_pos=np.asarray([[4.0, 5.0, 6.0]]),
            cam_pos=np.asarray([[0.0, 0.0, 0.0]]),
            cam_quat=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        ),
        task_fingerprint={"task": "same"},
        runtime_attributes={"step": 10},
        environment_rng={"seed": 1},
        process_rng={"seed": 2},
    )
    changed = copy.deepcopy(base)
    changed.model_arrays.cam_quat[0, 1] = 0.1
    changed.environment_rng["seed"] = 99
    changed.process_rng["seed"] = 100

    assert physical_snapshot_sha256(base) == physical_snapshot_sha256(changed)

    changed.integration_state[0] += 1e-12
    assert physical_snapshot_sha256(base) != physical_snapshot_sha256(changed)


def test_deployment_batch_rejects_privileged_targets() -> None:
    assert_deployment_batch({"pixels": torch.zeros(1)}, "clean")
    for key in ("q_true", "oracle_target", "reward", "success"):
        with pytest.raises(RuntimeError, match="forbidden privileged"):
            assert_deployment_batch({key: torch.zeros(1)}, key)


def test_history_batch_contains_no_oracle_values_and_has_expected_shapes() -> None:
    records = [_record(0.0), _record(1.0), _record(2.0)]
    corrected = np.stack([record.q_nominal + 0.25 for record in records])
    batch = build_lewm_history_batch(
        records,
        device=torch.device("cpu"),
        q_corrected=corrected,
    )

    assert set(batch) == {
        "pixels",
        "proprio",
        "q_nominal",
        "keypoints_xyz",
        "valid",
        "q_corrected",
    }
    assert batch["pixels"].shape == (1, 3, 3, 8, 8)
    assert batch["proprio"].shape == (1, 3, 4)
    assert batch["q_corrected"].shape == (1, 3, 3, 2)


def test_history_action_blocks_match_training_frame_skip_contract() -> None:
    records = [_record(float(index)) for index in range(11)]
    actions = np.arange(11 * 4, dtype=np.float32).reshape(11, 4)
    plan = SimpleNamespace(
        snapshot_step=10,
        records=records,
        actions=actions,
    )
    selected_records, history_action = _history_records_and_actions(
        plan,
        history_size=3,
        frame_skip=5,
    )

    assert selected_records == [records[0], records[5], records[10]]
    assert history_action.shape == (2, 20)
    np.testing.assert_array_equal(history_action[0], actions[0:5].reshape(-1))
    np.testing.assert_array_equal(history_action[1], actions[5:10].reshape(-1))


def test_candidate_replay_checks_repeat_and_order_independence() -> None:
    values = {"a": 1, "b": 2, "c": 3}

    def run(name: str) -> dict[str, int]:
        return {"value": values[name]}

    outcomes, audit = validate_candidate_replay(
        ("a", "b", "c"),
        run,
        repeat_name="a",
    )
    assert outcomes["b"] == {"value": 2}
    assert audit["validated"] is True
    assert audit["repeated_outcome_sha256"] == (
        audit["original_outcome_sha256"]["a"]
    )


def test_branch_replay_requires_exact_physics_but_allows_sparse_one_lsb_rgb() -> None:
    reference_record = _record(0.0)
    candidate_record = _record(0.0)
    candidate_record.rgb[0, 0, 0] = np.float32(1.0 / 255.0)
    values = dict(
        endpoint_snapshot_sha256=("state",),
        cumulative_return=np.asarray([1.0, 2.0, 3.0]),
        progress=np.asarray([0.1, 0.2, 0.3]),
        success=np.asarray([0.0, 0.0, 1.0]),
        action_sha256="same",
    )
    reference = BranchOutcome(endpoint_records=(reference_record,), **values)
    candidate = BranchOutcome(endpoint_records=(candidate_record,), **values)

    comparison = compare_branch_outcomes(
        reference,
        candidate,
        rgb_atol_lsb=1,
        max_rgb_changed_fraction=0.01,
    )

    assert comparison["equal"] is True
    assert comparison["rgb_exact"] is False
    assert comparison["nonvisual_endpoints_exact"] is True

    candidate_record.rgb[0, 0, 0] = np.float32(2.0 / 255.0)
    too_large = compare_branch_outcomes(
        reference,
        candidate,
        rgb_atol_lsb=1,
        max_rgb_changed_fraction=0.01,
    )
    assert too_large["equal"] is False

    state_mismatch = BranchOutcome(
        endpoint_records=(candidate_record,),
        endpoint_snapshot_sha256=("different-state",),
        cumulative_return=values["cumulative_return"],
        progress=values["progress"],
        success=values["success"],
        action_sha256=values["action_sha256"],
    )
    state_comparison = compare_branch_outcomes(
        reference, state_mismatch, rgb_atol_lsb=1, max_rgb_changed_fraction=0.01
    )
    assert state_comparison["equal"] is False
    assert state_comparison["endpoint_snapshot_exact"] is False

    candidate_record.rgb[0, 0, 0] = 0.0
    candidate_record.q_true[0, 0] += 1e-6
    physical_mismatch = compare_branch_outcomes(
        reference,
        candidate,
        rgb_atol_lsb=1,
        max_rgb_changed_fraction=0.01,
    )
    assert physical_mismatch["equal"] is False
    assert physical_mismatch["nonvisual_endpoints_exact"] is False


def test_counterfactual_summary_orients_lower_regret_as_positive_fcpg_gain() -> None:
    base = {
        "task": "reach-v3",
        "condition": "rotation5",
        "episode": 0,
        "snapshot": "same",
        "snapshot_sha256": "condition-specific",
        "snapshot_step": 10,
        "train_grounding": "fcpg",
        "horizon": 1,
        "candidate_action_set_sha256": "same-action-bank",
        "candidate_names": ["expert", "random"],
        "predicted_cost": [0.0, 1.0],
        "actual_cost": [0.0, 2.0],
        "actual_return": [2.0, 0.0],
        "actual_progress": [1.0, 0.0],
        "actual_success": [1.0, 0.0],
        "spearman": 1.0,
        "top1_hit": 1.0,
        "topk_hit": 1.0,
        "expert_top1": 1.0,
        "expert_topk": 1.0,
        "expert_pairwise_accuracy": 1.0,
        "top1_regret": 0.3,
        "topk_regret": 0.2,
        "normalized_top1_regret": 0.3,
        "normalized_topk_regret": 0.2,
        "return_spearman": 1.0,
        "progress_spearman": 1.0,
        "oracle_progress_spearman": 1.0,
    }
    nominal = {**base, "eval_grounding": "nominal"}
    fcpg = {
        **copy.deepcopy(base),
        "eval_grounding": "fcpg",
        "top1_regret": 0.1,
        "topk_regret": 0.05,
        "normalized_top1_regret": 0.1,
        "normalized_topk_regret": 0.05,
    }
    summary = summarize_counterfactual_rows(
        [nominal, fcpg],
        bootstrap_samples=20,
        seed=4,
    )
    paired = summary["paired_eval_grounding"][
        "train=fcpg|fcpg-minus-nominal|rotation5|H=1"
    ]
    assert paired["top1_regret"]["estimate"] == pytest.approx(0.2)
    assert paired["spearman"]["estimate"] == pytest.approx(0.0)


def test_counterfactual_summary_reports_matched_training_diagonal() -> None:
    base = {
        "task": "reach-v3",
        "condition": "rotation5",
        "episode": 0,
        "snapshot": "same",
        "snapshot_sha256": "condition-specific",
        "snapshot_step": 10,
        "horizon": 1,
        "candidate_names": ["expert", "random"],
        "candidate_action_set_sha256": "same-action-bank",
        **{metric: 0.0 for metric in SUMMARY_METRICS},
    }
    nominal = {
        **base,
        "train_grounding": "nominal",
        "eval_grounding": "nominal",
        "spearman": 0.1,
        "normalized_top1_regret": 0.4,
    }
    fcpg = {
        **base,
        "train_grounding": "fcpg",
        "eval_grounding": "fcpg",
        "spearman": 0.3,
        "normalized_top1_regret": 0.25,
    }

    summary = summarize_counterfactual_rows(
        [nominal, fcpg], bootstrap_samples=20, seed=4
    )
    paired = summary["paired_matched_diagonal"][
        "fcpg-fcpg-minus-nominal-nominal|rotation5|H=1"
    ]
    assert paired["spearman"]["estimate"] == pytest.approx(0.2)
    assert paired["normalized_top1_regret"]["estimate"] == pytest.approx(0.15)


def test_counterfactual_summary_rejects_mismatched_candidate_banks() -> None:
    base = {
        "task": "reach-v3",
        "condition": "rotation5",
        "episode": 0,
        "snapshot": "same",
        "horizon": 1,
        "train_grounding": "fcpg",
        **{metric: 0.0 for metric in SUMMARY_METRICS},
    }
    nominal = {
        **base,
        "eval_grounding": "nominal",
        "candidate_action_set_sha256": "bank-a",
    }
    fcpg = {
        **base,
        "eval_grounding": "fcpg",
        "candidate_action_set_sha256": "bank-b",
    }
    with pytest.raises(ValueError, match="paired rows differ"):
        summarize_counterfactual_rows(
            [nominal, fcpg], bootstrap_samples=20, seed=4
        )
