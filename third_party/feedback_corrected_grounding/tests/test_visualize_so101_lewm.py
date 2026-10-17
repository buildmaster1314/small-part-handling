from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.visualize_so101_lewm import (
    BankGroup,
    PredictionResult,
    build_gallery_index,
    checkpoint_action_normalizer,
    normalized_action_chunk,
    prediction_cycle_origins,
    retrieval_expected_joint_states,
    retrieve_neighbors,
    resolve_embedded_cache,
)


def test_visualization_uses_checkpoint_physical_action_scale() -> None:
    actions = np.arange(24, dtype=np.float32).reshape(4, 6)
    cache = SimpleNamespace(data=SimpleNamespace(actions=actions))

    chunk = normalized_action_chunk(
        cache,
        global_start=1,
        frameskip=2,
        action_center=np.zeros(6, dtype=np.float32),
        action_scale=np.full(6, 100.0, dtype=np.float32),
    )

    np.testing.assert_allclose(chunk, actions[1:3].reshape(-1) / 100.0)


def test_visualization_rejects_invalid_action_scale() -> None:
    cache = SimpleNamespace(
        data=SimpleNamespace(actions=np.zeros((2, 6), dtype=np.float32))
    )

    for scale in (0.0, -1.0, float("nan")):
        try:
            normalized_action_chunk(
                cache,
                0,
                1,
                np.zeros(6, dtype=np.float32),
                np.full(6, scale, dtype=np.float32),
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid joint scale {scale!r} was accepted")


def test_state_residual_checkpoint_uses_shared_six_state_normalizer() -> None:
    checkpoint = {
        "run_config": {
            "contract": {
                "protocol": "so101-state-residual-lewm-v4",
                "normalization": "shared zscore",
                "normalization_center": [1, 2, 3, 4, 5, 6],
                "normalization_scale": [2, 3, 4, 5, 6, 7],
            }
        }
    }

    center, scale, description = checkpoint_action_normalizer(checkpoint, {})

    np.testing.assert_array_equal(center, np.arange(1, 7, dtype=np.float32))
    np.testing.assert_array_equal(scale, np.arange(2, 8, dtype=np.float32))
    assert description == "shared zscore"


def test_direct_state_prediction_drives_retrieval_without_future_state() -> None:
    predicted_states = np.arange(12, dtype=np.float32).reshape(2, 6)
    predictions = PredictionResult(
        mode="teacher-forced",
        destination_local=np.asarray([5, 6]),
        display_input_local=np.asarray([0, 1]),
        baseline_local=np.asarray([0, 1]),
        tasks=np.zeros(2, dtype=np.int64),
        context_sizes=np.full(2, 3, dtype=np.int16),
        predicted_latents=np.zeros((2, 4), dtype=np.float32),
        predicted_states=predicted_states,
    )
    choice = SimpleNamespace(
        rows=np.arange(10, dtype=np.int64),
        cache=SimpleNamespace(
            data=SimpleNamespace(actions=np.full((10, 6), 999.0, dtype=np.float32))
        ),
    )

    expected, source = retrieval_expected_joint_states(choice, predictions)

    np.testing.assert_array_equal(expected, predicted_states)
    assert source == "direct six-state residual-head prediction"


def test_absolute_checkpoint_cache_can_be_relocated(tmp_path: Path) -> None:
    cache_base = tmp_path / "cache"
    relocated = cache_base / "so101_fruit_front_224"
    relocated.mkdir(parents=True)
    (relocated / "manifest.json").write_text("{}")

    resolved = resolve_embedded_cache(
        "/home/james/lewm/data/so101_cache/so101_fruit_front_224",
        tmp_path / "checkpoint.pt",
        cache_base,
    )

    assert resolved == relocated.resolve()


def test_relative_checkpoint_cache_can_be_relocated(tmp_path: Path) -> None:
    cache_base = tmp_path / "cache"
    relocated = cache_base / "so101_fruit_front_224"
    relocated.mkdir(parents=True)
    (relocated / "manifest.json").write_text("{}")

    resolved = resolve_embedded_cache(
        "data/so101_cache/so101_fruit_front_224",
        tmp_path / "checkpoint.pt",
        cache_base,
    )

    assert resolved == relocated.resolve()


def test_prediction_cycle_origins_preserve_history_and_future_horizon() -> None:
    origins = prediction_cycle_origins(
        frame_count=80,
        frameskip=5,
        history=3,
        horizon=8,
    )

    np.testing.assert_array_equal(origins, np.arange(10, 40, 5))
    assert np.all(origins - 2 * 5 >= 0)
    assert np.all(origins + 8 * 5 < 80)


def test_prediction_cycle_rejects_episode_without_full_future() -> None:
    origins = prediction_cycle_origins(
        frame_count=50,
        frameskip=5,
        history=3,
        horizon=8,
    )

    assert origins.size == 0


def test_retrieval_rejects_latent_neighbor_outside_command_pose_gate() -> None:
    cache = SimpleNamespace(
        cache_id=0,
        source="fruit",
        dataset_id="fruit-dataset",
        data=SimpleNamespace(
            states=np.asarray(
                [np.zeros(6), np.full(6, 10.0), np.full(6, 20.0)],
                dtype=np.float32,
            ),
            episodes=np.asarray([1, 2, 3], dtype=np.int64),
            tasks=np.zeros(3, dtype=np.int64),
        ),
    )
    predictions = PredictionResult(
        mode="prediction-cycle",
        destination_local=np.asarray([5, 10]),
        display_input_local=np.asarray([0, 5]),
        baseline_local=np.asarray([0, 5]),
        tasks=np.zeros(2, dtype=np.int64),
        context_sizes=np.full(2, 3, dtype=np.int16),
        predicted_latents=np.asarray([[0.1], [0.1]], dtype=np.float32),
    )
    group = BankGroup(
        cache_id=0,
        source="fruit",
        task=0,
        rows=np.arange(3, dtype=np.int64),
        latents=np.asarray([[0.0], [10.0], [1.0]], dtype=np.float32),
        available_train_frames=3,
        sampling="all",
    )

    retrieval = retrieve_neighbors(
        predictions,
        cache,
        {(0, 0): group},
        {("fruit-dataset", episode): "train" for episode in (1, 2, 3)},
        expected_joint_states=np.asarray(
            [np.full(6, 10.0), np.full(6, 100.0)], dtype=np.float32
        ),
        max_command_joint_rmse=0.5,
        max_arm_joint_error=0.5,
        max_gripper_error=0.5,
    )

    assert retrieval.rows.tolist() == [1, -1]
    assert retrieval.valid.tolist() == [True, False]
    assert retrieval.command_joint_rmse[0] == 0.0
    assert retrieval.command_joint_rmse[1] == 80.0
    assert np.isfinite(retrieval.latent_mse[0])
    assert np.isnan(retrieval.latent_mse[1])


def test_retrieval_rejects_single_joint_or_gripper_mismatch_hidden_by_rms() -> None:
    cache = SimpleNamespace(
        cache_id=0,
        source="fruit",
        dataset_id="fruit-dataset",
        data=SimpleNamespace(
            states=np.asarray(
                [
                    [0.0, 0.0, 0.0, 0.0, 20.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 20.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                ],
                dtype=np.float32,
            ),
            episodes=np.asarray([1, 2, 3], dtype=np.int64),
            tasks=np.zeros(3, dtype=np.int64),
        ),
    )
    predictions = PredictionResult(
        mode="prediction-cycle",
        destination_local=np.asarray([5]),
        display_input_local=np.asarray([0]),
        baseline_local=np.asarray([0]),
        tasks=np.zeros(1, dtype=np.int64),
        context_sizes=np.full(1, 3, dtype=np.int16),
        predicted_latents=np.asarray([[0.0]], dtype=np.float32),
    )
    group = BankGroup(
        cache_id=0,
        source="fruit",
        task=0,
        rows=np.arange(3, dtype=np.int64),
        latents=np.asarray([[0.0], [0.1], [1.0]], dtype=np.float32),
        available_train_frames=3,
        sampling="all",
    )

    retrieval = retrieve_neighbors(
        predictions,
        cache,
        {(0, 0): group},
        {("fruit-dataset", episode): "train" for episode in (1, 2, 3)},
        expected_joint_states=np.zeros((1, 6), dtype=np.float32),
        max_command_joint_rmse=10.0,
        max_arm_joint_error=5.0,
        max_gripper_error=5.0,
    )

    assert retrieval.rows.tolist() == [2]
    assert retrieval.command_arm_max_error.tolist() == [2.0]
    assert retrieval.command_gripper_error.tolist() == [2.0]


def test_gallery_embeds_prediction_cycle_media_and_honest_label() -> None:
    markup = build_gallery_index(
        [
            {
                "task": "draw line",
                "source": "so101_drawing_front",
                "episode": 4,
                "directory": "draw-line-ep4",
                "prediction_mse": 0.1,
                "copy_baseline_mse": 0.2,
                "media": ["prediction_cycle.mp4", "prediction_cycle.gif"],
            }
        ],
        "prediction-cycle",
    )

    assert 'src="draw-line-ep4/prediction_cycle.mp4"' in markup
    assert 'href="draw-line-ep4/prediction_cycle.gif"' in markup
    assert "pauses while H1-H8 predictions play" in markup
    assert "never generated RGB" in markup
    assert "command-pose-gated" in markup
