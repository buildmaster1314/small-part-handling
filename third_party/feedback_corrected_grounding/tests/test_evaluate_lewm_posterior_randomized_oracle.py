from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import evaluate_lewm_posterior_randomized_oracle as cli


def _arrays(
    episodes: list[tuple[str, str, int, int]],
    *,
    task_generation_seed: int = 73,
) -> dict[str, np.ndarray]:
    """Return complete 60-frame randomized episodes for pure evaluator tests."""

    frame_count = 60
    anchors = 2
    rows = [
        (episode, mode, task_index, env_seed, frame)
        for episode, mode, task_index, env_seed in episodes
        for frame in range(frame_count)
    ]
    count = len(rows)
    rgb_camera_only = np.zeros((count, 8, 8, 3), dtype=np.uint8)
    rgb = rgb_camera_only.copy()
    dropout = np.zeros((count, anchors), dtype=np.bool_)
    recovery = np.zeros_like(dropout)
    for episode_index in range(len(episodes)):
        start = episode_index * frame_count
        rgb[start + 20 : start + 30] = 255
        dropout[start + 20 : start + 30, 1] = True
        recovery[start + 30 : start + 40, 1] = True
    q = np.full((count, anchors, 2), 3.0, dtype=np.float32)
    return {
        "rgb": rgb,
        "rgb_camera_only": rgb_camera_only,
        "obs": np.zeros((count, 3), dtype=np.float32),
        "action": np.zeros((count, 1), dtype=np.float32),
        "task_name": np.full(count, "task"),
        "task_index": np.asarray([row[2] for row in rows], dtype=np.int32),
        "episode_id": np.asarray([row[0] for row in rows]),
        "source_episode_id": np.asarray([row[0] for row in rows]),
        "env_seed": np.asarray([row[3] for row in rows], dtype=np.int64),
        "frame_id": np.asarray([row[4] for row in rows], dtype=np.int32),
        "keypoints_xyz": np.zeros((count, anchors, 3), dtype=np.float32),
        "q_nominal": q.copy(),
        "valid_nominal": np.ones((count, anchors), dtype=np.bool_),
        "anchor_dropout_active": dropout,
        "anchor_recovery_segment": recovery,
        "randomized_source_trajectory_mode": np.asarray(
            [row[1] for row in rows]
        ),
        "task_generation_seed": np.asarray(task_generation_seed, dtype=np.int64),
        "observation_action_alignment": np.asarray("pre_action"),
        # This offline field is intentionally ignored by the evaluator loader.
        "q_true": q + np.float32(100.0),
    }


def _write_dataset(path: Path, arrays: dict[str, np.ndarray]) -> tuple[Path, Path]:
    np.savez_compressed(path, **arrays)
    shape = arrays["q_nominal"].shape
    observed = path.with_name(f"{path.stem}_observed_q.npy")
    canonical = path.with_name(f"{path.stem}_canonical_q.npy")
    np.save(observed, np.zeros(shape, dtype=np.float32))
    np.save(canonical, np.zeros(shape, dtype=np.float32))
    return observed, canonical


def _base_cli_args() -> list[str]:
    return ["--checkpoint", "checkpoint.pt", "--output-dir", "output"]


def test_cli_resolves_single_and_separate_input_layouts() -> None:
    single = cli.parse_args(
        [
            "--data",
            "randomized.npz",
            "--observed-q",
            "observed.npy",
            "--canonical-q",
            "canonical.npy",
            *_base_cli_args(),
        ]
    )
    single_layout = cli._resolve_input_layout(single)
    assert single_layout.name == "single_dataset_trajectory_partition"
    assert single_layout.calibration.trajectory_mode == "step"
    assert single_layout.evaluation.trajectory_mode == "linear_drift"

    separate = cli.parse_args(
        [
            "--calibration-data",
            "calibration.npz",
            "--calibration-observed-q",
            "calibration-observed.npy",
            "--calibration-canonical-q",
            "calibration-canonical.npy",
            "--evaluation-data",
            "evaluation.npz",
            "--evaluation-observed-q",
            "evaluation-observed.npy",
            "--evaluation-canonical-q",
            "evaluation-canonical.npy",
            *_base_cli_args(),
        ]
    )
    separate_layout = cli._resolve_input_layout(separate)
    assert separate_layout.name == "separate_datasets"
    assert separate_layout.calibration.trajectory_mode is None
    assert separate_layout.evaluation.trajectory_mode is None


def test_cli_rejects_mixed_or_incomplete_input_layouts() -> None:
    mixed = cli.parse_args(
        [
            "--data",
            "one.npz",
            "--observed-q",
            "one-observed.npy",
            "--canonical-q",
            "one-canonical.npy",
            "--calibration-data",
            "calibration.npz",
            *_base_cli_args(),
        ]
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        cli._resolve_input_layout(mixed)

    incomplete = cli.parse_args(
        ["--calibration-data", "calibration.npz", *_base_cli_args()]
    )
    with pytest.raises(ValueError, match="missing arguments"):
        cli._resolve_input_layout(incomplete)


def test_single_dataset_split_selects_whole_episodes() -> None:
    arrays = _arrays(
        [
            ("step-0", "step", 1, 10),
            ("step-1", "step", 1, 11),
            ("drift-0", "linear_drift", 2, 20),
        ]
    )
    calibration, evaluation = cli._split_by_trajectory_mode(
        arrays, "step", "linear_drift"
    )

    assert calibration.size == 120
    assert evaluation.size == 60
    assert set(arrays["episode_id"][calibration]) == {"step-0", "step-1"}
    assert set(arrays["episode_id"][evaluation]) == {"drift-0"}
    assert cli._episode_frame_bounds(arrays, calibration) == {
        ("task", "step-0"): (0, 59),
        ("task", "step-1"): (0, 59),
    }


def test_single_dataset_split_rejects_physical_identity_leakage() -> None:
    arrays = _arrays(
        [
            ("step-view", "step", 4, 99),
            ("drift-view", "linear_drift", 4, 99),
        ]
    )
    with pytest.raises(ValueError, match="physical episode identity leakage"):
        cli._split_by_trajectory_mode(arrays, "step", "linear_drift")


def test_separate_datasets_allow_mixed_modes_when_identities_are_disjoint() -> None:
    calibration = _arrays(
        [
            ("cal-step", "step", 1, 10),
            ("cal-drift", "linear_drift", 2, 11),
        ]
    )
    evaluation = _arrays(
        [
            ("eval-step", "step", 3, 20),
            ("eval-drift", "linear_drift", 4, 21),
        ]
    )

    cli._assert_disjoint_datasets(calibration, evaluation)
    calibration_indices, _ = cli._all_indices_and_identities(calibration)
    evaluation_indices, _ = cli._all_indices_and_identities(evaluation)
    assert calibration_indices.size == evaluation_indices.size == 120
    assert cli._trajectory_summary(calibration, calibration_indices)[
        "episode_counts"
    ] == {"linear_drift": 1, "step": 1}


def test_separate_dataset_split_rejects_physical_identity_leakage() -> None:
    calibration = _arrays([("calibration", "step", 7, 31)])
    evaluation = _arrays([("evaluation", "linear_drift", 7, 31)])

    with pytest.raises(ValueError, match="physical episode identity leakage"):
        cli._assert_disjoint_datasets(calibration, evaluation)


def test_resampling_units_pair_camera_views_and_validate_physics(
    tmp_path: Path,
) -> None:
    arrays = _arrays(
        [
            ("step-view", "step", 1, 10),
            ("drift-view", "linear_drift", 2, 20),
        ]
    )
    arrays["source_episode_id"][:] = "source-7"
    indices, _ = cli._all_indices_and_identities(arrays)

    units = cli._resampling_units(arrays, indices)

    assert units == {
        ("task", "step-view"): "source-7",
        ("task", "drift-view"): "source-7",
    }
    observed_path = tmp_path / "observed.npy"
    canonical_path = tmp_path / "canonical.npy"
    np.save(observed_path, np.zeros_like(arrays["q_nominal"]))
    np.save(canonical_path, np.zeros_like(arrays["q_nominal"]))
    report = cli._split_report(
        cli.LoadedSplit(
            spec=cli.SplitSpec(
                label="test",
                data_path=Path("data.npz"),
                observed_q_path=observed_path,
                canonical_q_path=canonical_path,
                trajectory_mode=None,
            ),
            arrays=arrays,
            indices=indices,
            observed_q=np.zeros_like(arrays["q_nominal"]),
            canonical_q=np.zeros_like(arrays["q_nominal"]),
        )
    )
    assert report["view_episode_count"] == 2
    assert report["physical_trajectory_count"] == 1

    arrays["action"][60, 0] = 1.0
    with pytest.raises(ValueError, match="physical field 'action'"):
        cli._resampling_units(arrays, indices)


def test_single_dataset_split_rejects_source_trajectory_leakage() -> None:
    arrays = _arrays(
        [
            ("step-view", "step", 1, 10),
            ("drift-view", "linear_drift", 2, 20),
        ]
    )
    arrays["source_episode_id"][:] = "shared-source"

    with pytest.raises(ValueError, match="source trajectory leakage"):
        cli._split_by_trajectory_mode(arrays, "step", "linear_drift")


class _EncodingModel:
    num_keypoints = 2
    latent_dim = 3
    frame_encoder = SimpleNamespace(obs_dim=2)

    def __init__(self) -> None:
        self.batch_keys: list[set[str]] = []

    def encode_frames(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.batch_keys.append(set(batch))
        value = batch["pixels"].mean(dim=(2, 3, 4))
        embedding = value[..., None, None].expand(-1, -1, 2, 3).clone()
        return embedding, embedding


def test_encode_frames_uses_allowlisted_inputs_without_q_true() -> None:
    arrays = _arrays([("episode", "step", 1, 10)])
    model = _EncodingModel()
    output = cli._encode_frames(  # type: ignore[arg-type]
        model,
        images=arrays["rgb"],
        obs=arrays["obs"],
        q_nominal=arrays["q_nominal"],
        q_corrected=np.zeros_like(arrays["q_nominal"]),
        keypoints_xyz=arrays["keypoints_xyz"],
        valid_nominal=arrays["valid_nominal"],
        batch_size=17,
        device=torch.device("cpu"),
    )

    assert output.shape == (60, 2, 3)
    assert model.batch_keys
    assert all(
        keys
        == {
            "pixels",
            "proprio",
            "q_nominal",
            "q_corrected",
            "keypoints_xyz",
            "valid_nominal",
        }
        for keys in model.batch_keys
    )
    assert all("q_true" not in keys for keys in model.batch_keys)


def _markdown_payload() -> dict[str, object]:
    per_task = {
        "task": {
            "candidate_relative_improvement": 0.2,
            "candidate_incremental_improvement_over_time_only": 0.1,
            "physical_episode_count": 2,
        }
    }
    return {
        "decision": {"decision": "Go", "failed_checks": []},
        "inputs": {
            "layout": "separate_datasets",
            "calibration": {
                "frame_count": 120,
                "view_episode_count": 2,
                "physical_trajectory_count": 1,
                "physical_episode_count": 2,
                "trajectories": {
                    "episode_counts": {"linear_drift": 1, "step": 1}
                },
            },
            "evaluation": {
                "frame_count": 120,
                "view_episode_count": 2,
                "physical_trajectory_count": 1,
                "physical_episode_count": 2,
                "trajectories": {
                    "episode_counts": {"linear_drift": 1, "step": 1}
                },
            },
        },
        "statistics": {
            "primary": {
                "candidate_relative_improvement": 0.2,
                "candidate_relative_improvement_95_ci": [0.1, 0.3],
                "time_only_relative_improvement": 0.1,
                "time_only_relative_improvement_95_ci": [0.05, 0.15],
                "candidate_incremental_improvement_over_time_only": 0.1,
                "candidate_incremental_improvement_over_time_only_95_ci": [
                    0.02,
                    0.18,
                ],
                "per_task": per_task,
            },
            "coverage": {
                "physical_episode_count": 2,
                "view_episode_count": 4,
                "physical_trajectory_count": 2,
                "task_count": 1,
                "minimum_active_events_per_episode": 2,
                "minimum_recovery_events_per_episode": 2,
            },
            "recovery": {
                "normalized_excess_mse": 0.01,
                "normalized_excess_mse_95_ci": [0.0, 0.02],
                "normalized_excess_mse_one_sided_95_upper": 0.018,
            },
            "clean": {
                "normalized_excess_mse": 0.0,
                "normalized_excess_mse_95_ci": [0.0, 0.0],
                "normalized_excess_mse_one_sided_95_upper": 0.0,
            },
        },
    }


def test_markdown_is_human_readable_and_marks_diagnostics() -> None:
    markdown = cli._markdown(_markdown_payload())

    assert markdown.startswith("# Fresh Randomized Posterior Oracle\n")
    assert "Decision: **Go**" in markdown
    assert "FSM incremental over time-only" in markdown
    assert "q_true is never a model input" in markdown
    assert "hard feasibility checks" in markdown
    assert "20.00%" in markdown


class _FakeAnchorPersistentLeWM(torch.nn.Module):
    num_keypoints = 2
    latent_dim = 3
    history_size = 3
    action_dim = 5

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.frame_encoder = SimpleNamespace(obs_dim=2)
        self.seen_batch_keys: list[set[str]] = []

    def encode_frames(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.seen_batch_keys.append(set(batch))
        assert "q_true" not in batch
        value = batch["pixels"].mean(dim=(2, 3, 4))
        embedding = value[..., None, None].expand(-1, -1, 2, 3).clone()
        return embedding, embedding

    def predict(
        self, history: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        assert action.shape[:2] == history.shape[:2]
        return torch.zeros_like(history)


def test_separate_dataset_cli_runs_end_to_end_with_frozen_fake_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calibration_path = tmp_path / "calibration.npz"
    evaluation_path = tmp_path / "evaluation.npz"
    calibration_q = _write_dataset(
        calibration_path,
        _arrays([("calibration", "step", 1, 100)]),
    )
    evaluation_q = _write_dataset(
        evaluation_path,
        _arrays([("evaluation", "linear_drift", 2, 200)]),
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"mock checkpoint")
    output = tmp_path / "report"
    model = _FakeAnchorPersistentLeWM()

    monkeypatch.setattr(cli, "AnchorPersistentLeWM", _FakeAnchorPersistentLeWM)
    monkeypatch.setattr(
        cli,
        "load_lewm_grounded_checkpoint",
        lambda path, device: (model.to(device), {}, {"step": 2_000}),
    )

    result = cli.main(
        [
            "--calibration-data",
            str(calibration_path),
            "--calibration-observed-q",
            str(calibration_q[0]),
            "--calibration-canonical-q",
            str(calibration_q[1]),
            "--evaluation-data",
            str(evaluation_path),
            "--evaluation-observed-q",
            str(evaluation_q[0]),
            "--evaluation-canonical-q",
            str(evaluation_q[1]),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--batch-size",
            "13",
            "--time-bins",
            "2",
            "--min-count",
            "1",
            "--bootstrap-samples",
            "20",
            "--bootstrap-seed",
            "9",
        ]
    )

    assert result == 0
    payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    markdown = (output / "summary.md").read_text(encoding="utf-8")
    assert payload["inputs"]["layout"] == "separate_datasets"
    assert payload["inputs"]["checkpoint_step"] == 2_000
    assert payload["protocol"]["q_true_model_input"] is False
    assert payload["time_only"]["calibration_row_count"] == 20
    assert payload["evaluation"]["observed_sequence_rows"] == 20
    assert payload["statistics"]["methods"] == {
        "measurement": cli.MEASUREMENT_METHOD,
        "candidate": cli.FSM_ORACLE_METHOD,
        "time_only": cli.TIME_ONLY_METHOD,
    }
    assert "# Fresh Randomized Posterior Oracle" in markdown
    assert all("q_true" not in keys for keys in model.seen_batch_keys)
    assert not model.weight.requires_grad
