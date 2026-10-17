from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.evaluate_action_grounding_sensitivity import (
    ModeOutput,
    infer_grounding_modes,
    input_contract_mismatches,
    load_frame_identity,
    move_batch,
    snapshot_rows,
    summarize_rows,
)


class _FakePolicy:
    def __init__(self, *, fail_on: str = "") -> None:
        self.mode = "fcpg"
        self.fail_on = fail_on
        self.calls: list[tuple[str, int, frozenset[str]]] = []

    def predict_action_chunk(self, batch):
        self.calls.append((self.mode, id(batch["q_nominal"]), frozenset(batch)))
        if self.mode == self.fail_on:
            raise RuntimeError("injected failure")
        q_nominal = batch["q_nominal"]
        batch_size = q_nominal.shape[0]
        action = torch.zeros(batch_size, 8, 4)
        if self.mode == "geoprop":
            action[..., :3] = 2.0
            action[..., 3] = 1.0
            points = q_nominal
        elif self.mode == "fcpg":
            action[..., :3] = 0.5
            points = batch["fcpg_q_corrected"]
        elif self.mode == "oracle":
            points = batch["q_true"]
        else:
            raise AssertionError(self.mode)
        return SimpleNamespace(action=action, grounding_points_px=points)


def _batch() -> dict[str, torch.Tensor]:
    q_true = torch.tensor(
        [[[[4.0, 4.0], [8.0, 8.0]], [[5.0, 5.0], [10.0, 10.0]]]]
    )
    q_nominal = q_true + 4.0
    corrected = q_true + 1.0
    return {
        "q_true": q_true,
        "q_nominal": q_nominal,
        "fcpg_q_corrected": corrected,
        "valid": torch.ones(1, 2, 2, dtype=torch.bool),
        "action_mask": torch.ones(1, 8, dtype=torch.bool),
        "anchor_index": torch.tensor([1]),
        "image_size": torch.tensor([[32, 32]]),
        "action": torch.ones(1, 8, 4),
        "next_obs": torch.ones(1, 2, 4),
        "next_q_true": q_true.clone(),
    }


def _identity() -> dict[str, np.ndarray]:
    return {
        "task_name": np.asarray(["pick", "pick"]),
        "episode_id": np.asarray([7, 7]),
        "frame_id": np.asarray([0, 1]),
        "actual_camera_rotation_deg": np.asarray([0.0, 5.0]),
        "actual_camera_rotation_axis": np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        ),
        "actual_camera_translation_xyz": np.asarray(
            [[0.0, 0.0, 0.0], [0.01, -0.02, 0.03]], dtype=np.float32
        ),
    }


def test_three_modes_share_batch_and_restore_fcpg_mode() -> None:
    model = _FakePolicy()
    batch = _batch()

    outputs = infer_grounding_modes(model, batch, device_type="cpu")

    assert model.mode == "fcpg"
    assert [mode for mode, _, _ in model.calls] == ["geoprop", "fcpg", "oracle"]
    assert len({batch_id for _, batch_id, _ in model.calls}) == 1
    for mode, _, keys in model.calls:
        assert "action" not in keys and "next_obs" not in keys
        assert "next_q_true" not in keys
        assert ("q_true" in keys) == (mode == "oracle")
    assert set(outputs) == {"nominal", "fcpg", "oracle"}
    np.testing.assert_allclose(outputs["fcpg"].action[..., :3], 0.5)


def test_mode_is_restored_when_inference_raises() -> None:
    model = _FakePolicy(fail_on="oracle")

    with pytest.raises(RuntimeError, match="injected failure"):
        infer_grounding_modes(model, _batch(), device_type="cpu")

    assert model.mode == "fcpg"


def test_cpu_batch_promotes_half_precision_feature_cache() -> None:
    moved = move_batch(
        {
            "features": torch.ones(2, dtype=torch.float16),
            "indices": torch.ones(2, dtype=torch.long),
        },
        torch.device("cpu"),
    )

    assert moved["features"].dtype == torch.float32
    assert moved["indices"].dtype == torch.long


def test_snapshot_metrics_use_common_support_and_separate_gripper() -> None:
    model = _FakePolicy()
    batch = _batch()
    outputs = infer_grounding_modes(model, batch, device_type="cpu")
    # The second corrected point is invalid, so EPE uses one common anchor.
    corrected = outputs["fcpg"].points.copy()
    corrected[0, -1, 1] = np.asarray([40.0, 40.0])
    outputs["fcpg"] = ModeOutput(outputs["fcpg"].action, corrected)

    rows = snapshot_rows(
        batch,
        outputs,
        _identity(),
        h_exec=8,
        policy_seed=3,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["episode"] == 7 and row["frame"] == 1
    assert row["common_valid_anchors"] == 1
    assert row["base_valid_anchors"] == 2
    assert row["fcpg_oob_anchors"] == 1
    assert row["common_support_coverage"] == pytest.approx(0.5)
    assert row["nominal_epe"] == pytest.approx(np.sqrt(32.0))
    assert row["fcpg_epe"] == pytest.approx(np.sqrt(2.0))
    assert row["delta_action_xyz_h1"] == pytest.approx(1.5 * np.sqrt(3.0))
    assert row["delta_action_xyz_h8"] == pytest.approx(1.5 * np.sqrt(3.0))
    assert row["delta_gripper_h8"] == pytest.approx(1.0)
    assert row["actual_camera_rotation_deg"] == pytest.approx(5.0)
    assert row["actual_camera_rotation_axis"] == pytest.approx([0.0, 1.0, 0.0])
    assert row["actual_camera_translation_xyz"] == pytest.approx(
        [0.01, -0.02, 0.03]
    )
    assert row["actual_camera_translation_norm"] == pytest.approx(
        np.linalg.norm([0.01, -0.02, 0.03])
    )
    # Vector-valued identity fields must already be standard JSON values.
    assert json.loads(json.dumps(row))["actual_camera_rotation_axis"] == [
        0.0,
        1.0,
        0.0,
    ]


def test_load_frame_identity_preserves_camera_vectors(tmp_path) -> None:
    data_path = tmp_path / "identity.npz"
    np.savez(
        data_path,
        task_name=np.asarray(["pick-place-v3"]),
        episode_id=np.asarray([4], dtype=np.int64),
        frame_id=np.asarray([9], dtype=np.int64),
        actual_camera_rotation_deg=np.asarray([7.5], dtype=np.float32),
        actual_camera_rotation_axis=np.asarray(
            [[0.0, 0.0, 1.0]], dtype=np.float32
        ),
        actual_camera_translation_xyz=np.asarray(
            [[0.04, -0.01, 0.02]], dtype=np.float32
        ),
    )

    identity = load_frame_identity(data_path)

    assert identity["task_name"].tolist() == ["pick-place-v3"]
    assert identity["episode_id"].tolist() == [4]
    assert identity["frame_id"].tolist() == [9]
    np.testing.assert_allclose(
        identity["actual_camera_rotation_axis"], [[0.0, 0.0, 1.0]]
    )
    np.testing.assert_allclose(
        identity["actual_camera_translation_xyz"], [[0.04, -0.01, 0.02]]
    )


def test_h8_requires_all_eight_real_action_steps() -> None:
    model = _FakePolicy()
    batch = _batch()
    batch["action_mask"][0, -1] = False
    outputs = infer_grounding_modes(model, batch, device_type="cpu")

    row = snapshot_rows(
        batch,
        outputs,
        _identity(),
        h_exec=8,
        policy_seed=3,
    )[0]

    assert np.isfinite(row["delta_action_xyz_h1"])
    assert np.isnan(row["delta_action_xyz_h8"])
    assert np.isnan(row["delta_gripper_h8"])


def test_summary_clusters_windows_within_episode_before_task_macro() -> None:
    rows = []
    for task, episode, values in (
        ("pick", 1, (1.0, 3.0)),
        ("pick", 2, (5.0, 5.0)),
        ("button", 3, (2.0, 2.0)),
    ):
        for frame, value in enumerate(values):
            rows.append(
                {
                    "task": task,
                    "episode": episode,
                    "frame": frame,
                    "delta_epe": value,
                    "delta_action_xyz_h1": value,
                    "delta_action_xyz_h8": value,
                    "delta_gripper_h1": 0.0,
                    "delta_gripper_h8": 0.0,
                    "base_valid_anchors": 2,
                    "common_valid_anchors": 1,
                    "fcpg_oob_anchors": 1,
                    "common_support_coverage": 0.5,
                }
            )

    summary = summarize_rows(rows, bootstrap_samples=100, bootstrap_seed=9)

    # pick episode means are 2 and 5 -> 3.5; button is 2; task macro is 2.75.
    delta = summary["episode_clustered_task_macro"]["delta_action_xyz_h8"]
    assert delta["estimate"] == pytest.approx(2.75)
    assert len(delta["ci95"]) == 2
    assert delta["per_task_episode_mean"] == {"button": 2.0, "pick": 3.5}
    assert summary["physical_episodes"] == 3
    assert summary["epe_action_association"]["estimate"] == pytest.approx(0.25)
    assert summary["support"] == {
        "base_valid_anchors": 12,
        "common_valid_anchors": 6,
        "fcpg_oob_anchors": 6,
        "overall_common_support_coverage": 0.5,
    }


def test_noninferential_summary_omits_confidence_intervals() -> None:
    rows = [
        {
            "task": "pick",
            "episode": 1,
            "delta_epe": 1.0,
            "delta_action_xyz_h1": 1.0,
            "delta_action_xyz_h8": 1.0,
            "delta_gripper_h1": 0.0,
            "delta_gripper_h8": 0.0,
            "base_valid_anchors": 3,
            "common_valid_anchors": 3,
            "fcpg_oob_anchors": 0,
            "common_support_coverage": 1.0,
        }
    ]

    summary = summarize_rows(
        rows,
        bootstrap_samples=100,
        bootstrap_seed=9,
        inferential=False,
    )

    assert summary["inferential"] is False
    assert "ci95" not in json.dumps(summary)


def test_artifact_contract_rejects_wrong_corrector_cache(tmp_path) -> None:
    data = tmp_path / "data.npz"
    data.touch()
    vision = tmp_path / "vision"
    vision.mkdir()
    (vision / "metadata.json").write_text(
        json.dumps({"source_data": str(data)}), encoding="utf-8"
    )
    grounding = tmp_path / "grounding"
    grounding.mkdir()
    wrong_corrector = tmp_path / "wrong.pt"
    expected_corrector = tmp_path / "expected.pt"
    (grounding / "metadata.json").write_text(
        json.dumps(
            {
                "source_data": str(data),
                "vision_cache": str(vision),
                "checkpoint": str(expected_corrector),
            }
        ),
        encoding="utf-8",
    )

    matching = input_contract_mismatches(
        data_path=data,
        vision_cache=vision,
        grounding_cache=grounding,
        checkpoint_args={
            "data": str(data),
            "vision_cache": str(vision),
            "grounding_cache": str(grounding),
            "fcpg_checkpoint": str(expected_corrector),
        },
    )
    assert matching == []

    (grounding / "metadata.json").write_text(
        json.dumps(
            {
                "source_data": str(data),
                "vision_cache": str(vision),
                "checkpoint": str(wrong_corrector),
            }
        ),
        encoding="utf-8",
    )

    mismatches = input_contract_mismatches(
        data_path=data,
        vision_cache=vision,
        grounding_cache=grounding,
        checkpoint_args={
            "data": str(data),
            "vision_cache": str(vision),
            "grounding_cache": str(grounding),
            "fcpg_checkpoint": str(expected_corrector),
        },
    )

    assert len(mismatches) == 1
    assert mismatches[0].startswith("grounding_cache.corrector:")
