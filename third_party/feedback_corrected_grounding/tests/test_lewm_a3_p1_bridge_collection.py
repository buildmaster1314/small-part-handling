from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import fcpg.lewm_a3_p1_bridge_collection as bridge_collection
from fcpg.lewm_a3_p1_bridge_collection import (
    ReplayedBridgeRow,
    STRICT_A3_10K_SEEDS,
    build_raw_adjacent_policy_batch,
    select_p1_visible_grounding,
    select_pick_bridge_rows,
    sha256_file,
    stack_replayed_bridge_rows,
    validate_bridge_arrays,
    validate_strict_p1_checkpoint,
)
from fcpg.lewm_counterfactual_runtime import FrameRecord


def _record(value: float) -> FrameRecord:
    return FrameRecord(
        rgb=np.full((3, 4, 4), value, dtype=np.float32),
        proprio=np.full((9,), value + 0.1, dtype=np.float32),
        keypoints_xyz=np.full((3, 3), value + 0.2, dtype=np.float32),
        q_nominal=np.full((3, 2), value + 0.3, dtype=np.float32),
        valid_nominal=np.ones(3, dtype=bool),
        q_true=np.full((3, 2), value + 100.0, dtype=np.float32),
        valid_true=np.ones(3, dtype=bool),
    )


def _a1_arrays(sample_count: int = 4) -> dict[str, np.ndarray]:
    return {
        "history_embedding": np.zeros((sample_count, 3, 3, 192), np.float32),
        "history_action": np.zeros((sample_count, 2, 20), np.float32),
        "center_action": np.zeros((sample_count, 8, 20), np.float32),
        "future_action": np.zeros((sample_count, 8, 8, 20), np.float32),
        "target_embedding": np.zeros((sample_count, 8, 8, 3, 192), np.float32),
        "task_index": np.asarray([0, 0, 1, 0], dtype=np.int64),
        "task_instance_index": np.asarray([4, 2, 1, 5], dtype=np.int64),
        "source_episode_id": np.asarray(["pick:4", "pick:2", "drawer:1", "pick:5"]),
        "hazard_id": np.asarray([7, 9, 8, 6], dtype=np.int64),
        "split": np.asarray(["development", "train", "train", "test"]),
    }


def _bridge_row(source_row: int = 0) -> ReplayedBridgeRow:
    seeds = len(STRICT_A3_10K_SEEDS)
    branches = 1
    return ReplayedBridgeRow(
        source_row_index=source_row,
        branch_index=np.asarray([2], dtype=np.int64),
        history_embedding=np.zeros((3, 3, 192), np.float32),
        history_action=np.zeros((2, 20), np.float32),
        branch_action=np.zeros((branches, 8, 20), np.float32),
        factual_prediction=np.zeros((seeds, branches, 8, 3, 192), np.float32),
        no_action_prediction=np.zeros((seeds, branches, 8, 3, 192), np.float32),
        hidden_clean_a3_target=np.zeros((branches, 8, 3, 192), np.float32),
        last_policy_token=np.zeros((3, 256), np.float32),
        last_sampled_visual=np.zeros((3, 256), np.float32),
        last_xyz=np.zeros((3, 3), np.float32),
        current_xyz=np.zeros((branches, 8, 3, 3), np.float32),
        hidden_clean_policy_token=np.zeros((branches, 8, 3, 256), np.float32),
        hidden_clean_sampled_visual=np.zeros((branches, 8, 3, 256), np.float32),
        hidden_clean_q_corrected=np.zeros((branches, 8, 3, 2), np.float32),
        hidden_clean_reliability=np.ones((branches, 8, 3), np.float32),
        max_a3_history_reconstruction_error=0.0,
        max_a3_target_reconstruction_error=0.0,
    )


def test_policy_batch_uses_raw_adjacent_frames_and_excludes_oracle_fields() -> None:
    batch = build_raw_adjacent_policy_batch(
        ((_record(4.0), _record(5.0)), (_record(9.0), _record(10.0))),
        device=torch.device("cpu"),
    )

    assert set(batch) == {"rgb", "obs", "q_nominal", "keypoints_xyz"}
    assert tuple(batch["rgb"].shape) == (2, 2, 3, 4, 4)
    assert batch["rgb"][:, :, 0, 0, 0].tolist() == [[4.0, 5.0], [9.0, 10.0]]
    np.testing.assert_allclose(
        batch["obs"][:, :, 0].numpy(),
        np.asarray([[4.1, 5.1], [9.1, 10.1]], dtype=np.float32),
    )
    assert "q_true" not in batch


def test_nominal_visible_grounding_only_overrides_corrected_points() -> None:
    batch = build_raw_adjacent_policy_batch(
        ((_record(4.0), _record(5.0)),),
        device=torch.device("cpu"),
    )

    selected = select_p1_visible_grounding(batch, mode="nominal")

    assert set(selected) == set(batch) | {
        "fcpg_q_corrected",
        "fcpg_reliability",
    }
    assert selected["fcpg_q_corrected"] is batch["q_nominal"]
    assert torch.equal(
        selected["fcpg_reliability"],
        torch.ones_like(batch["q_nominal"][..., 0]),
    )
    assert "q_true" not in selected
    assert "fcpg_q_corrected" not in batch


def test_visible_grounding_rejects_unknown_mode() -> None:
    batch = build_raw_adjacent_policy_batch(
        ((_record(4.0), _record(5.0)),),
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="visible grounding mode"):
        select_p1_visible_grounding(batch, mode="oracle")


def test_pick_row_selection_is_split_safe_and_deterministic() -> None:
    rows = select_pick_bridge_rows(_a1_arrays(), max_instances=1, max_rows=1)
    # Instance 2 sorts before instance 4; task 1 and test are never eligible.
    np.testing.assert_array_equal(rows, np.asarray([1], dtype=np.int64))


def test_stacked_bridge_schema_keeps_predictor_seed_before_row_axis() -> None:
    arrays = _a1_arrays()
    stacked = stack_replayed_bridge_rows(
        [_bridge_row(0)], arrays, predictor_seeds=STRICT_A3_10K_SEEDS
    )
    validate_bridge_arrays(stacked)

    assert stacked["factual_prediction"].shape == (3, 1, 1, 8, 3, 192)
    assert stacked["hidden_clean_policy_token"].shape == (1, 1, 8, 3, 256)
    assert stacked["last_sampled_visual"].shape == (1, 3, 256)
    assert stacked["current_xyz"].shape == (1, 1, 8, 3, 3)


def test_strict_p1_provenance_binds_embedded_corrector(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "fcpg").mkdir(parents=True)
    corrector = repo / "outputs" / "corrector.pt"
    corrector.parent.mkdir(parents=True)
    corrector.write_bytes(b"corrector")
    policy = repo / "outputs" / "policy.pt"
    torch.save(
        {
            "checkpoint_type": "fcpg_action_policy",
            "num_keypoints": 3,
            "args": {
                "mode": "fcpg",
                "policy_arch": "legacy",
                "obs_horizon": 2,
                "model_dim": 256,
                "vision_dim": 256,
                "fcpg_checkpoint": "outputs/corrector.pt",
            },
        },
        policy,
    )

    report = validate_strict_p1_checkpoint(
        policy,
        corrector,
        expected_policy_sha256=sha256_file(policy),
        expected_corrector_sha256=sha256_file(corrector),
        expected_policy_suffix="",
        expected_corrector_suffix="",
    )
    assert report["history_alignment"] == "raw-adjacent-pre-action-[t-1,t]"

    other = repo / "outputs" / "other.pt"
    other.write_bytes(b"different")
    with pytest.raises(ValueError, match="embedded corrector path"):
        validate_strict_p1_checkpoint(
            policy,
            other,
            expected_policy_sha256=sha256_file(policy),
            expected_corrector_sha256=sha256_file(other),
            expected_policy_suffix="",
            expected_corrector_suffix="",
        )


def test_wrong_endpoint_is_far_outside_fp16_reconstruction_tolerance() -> None:
    expected = np.zeros((8, 3, 192), dtype=np.float32)
    wrong_endpoint = expected.copy()
    wrong_endpoint[3] = 0.01

    with pytest.raises(RuntimeError, match="reconstruction failed"):
        bridge_collection._require_close(
            wrong_endpoint, expected, label="deliberately wrong endpoint"
        )
