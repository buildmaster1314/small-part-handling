import json
from pathlib import Path

import numpy as np
import torch

from scripts.train_so101_lewm import (
    Cache,
    SO101Sequences,
    _shuffle_actions_within_group,
    episode_splits,
    evaluate,
    global_train_action_stats,
    parse_exclusions,
    resolve_cache_source_root,
)


def _write_cache(
    root: Path,
    *,
    dataset_id: str,
    name: str,
    episodes: list[tuple[int, int]],
    frames_per_episode: int = 20,
) -> Path:
    root.mkdir(parents=True)
    rows = [
        (episode, task, frame)
        for episode, task in episodes
        for frame in range(frames_per_episode)
    ]
    frame_count = len(rows)
    np.save(root / "frames.npy", np.zeros((frame_count, 4, 4, 3), dtype=np.uint8))
    np.save(root / "actions.npy", np.full((frame_count, 6), 50.0, dtype=np.float32))
    state = np.arange(frame_count, dtype=np.float32)[:, None].repeat(6, axis=1)
    np.save(root / "states.npy", state)
    np.save(root / "episodes.npy", np.asarray([row[0] for row in rows], dtype=np.int32))
    np.save(root / "tasks.npy", np.asarray([row[1] for row in rows], dtype=np.int16))
    np.save(root / "frame_indices.npy", np.asarray([row[2] for row in rows], dtype=np.int32))
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": "so101-lewm-cache-v1",
                "name": name,
                "dataset_id": dataset_id,
                "robot_type": "so101_follower",
                "camera": "observation.images.front",
                "frames": frame_count,
                "task_names": {"0": "Grab a orange", "1": "Grab an orange"},
                "action_mean": [50.0] * 6,
                "action_std": [1.0] * 6,
            }
        )
    )
    return root


def _dataset(cache_roots: list[Path], split: str) -> SO101Sequences:
    return SO101Sequences(
        cache_roots,
        split,
        frameskip=5,
        history=3,
        exclusions=set(),
        split_seed=7,
        val_fraction=0.25,
        joint_scale=100.0,
        motion_threshold=0.01,
    )


def test_split_is_episode_level_stratified_and_camera_stable(tmp_path: Path) -> None:
    episodes = [(episode, 0 if episode < 4 else 1) for episode in range(8)]
    front = _write_cache(tmp_path / "front", dataset_id="robot_run", name="front", episodes=episodes)
    wrist = _write_cache(tmp_path / "wrist", dataset_id="robot_run", name="wrist", episodes=episodes)

    train = _dataset([front, wrist], "train")
    validation = _dataset([front, wrist], "val")
    train_episodes = {
        (train.caches[cache_id].dataset_id, int(train.caches[cache_id].episodes[start]))
        for cache_id, start in train.samples
    }
    val_episodes = {
        (validation.caches[cache_id].dataset_id, int(validation.caches[cache_id].episodes[start]))
        for cache_id, start in validation.samples
    }
    assert train_episodes.isdisjoint(val_episodes)
    assert {task for _, task in train.group_keys} == {"grab orange"}
    assert {task for _, task in validation.group_keys} == {"grab orange"}
    assert {cache_id for cache_id, _ in train.samples} == {0, 1}
    assert {cache_id for cache_id, _ in validation.samples} == {0, 1}


def test_cache_source_root_follows_relocated_standard_layout(tmp_path: Path) -> None:
    cache_root = tmp_path / "data" / "lerobot" / "so101" / "cache" / "drawing"
    raw_root = tmp_path / "data" / "lerobot" / "so101" / "raw" / "so101_drawing"
    cache_root.mkdir(parents=True)
    raw_root.mkdir(parents=True)

    resolved = resolve_cache_source_root(
        cache_root,
        "/old/worktree/data/so101_public/so101_drawing",
    )

    assert resolved == raw_root.resolve()


def test_actions_use_shared_physical_scale_and_exclusions(tmp_path: Path) -> None:
    root = _write_cache(
        tmp_path / "cache",
        dataset_id="source",
        name="source_front",
        episodes=[(0, 0), (1, 0), (2, 0), (3, 0)],
    )
    dataset = SO101Sequences(
        [root],
        "train",
        frameskip=5,
        history=3,
        exclusions=parse_exclusions(["source:2"]),
        split_seed=7,
        val_fraction=0.25,
        joint_scale=100.0,
        motion_threshold=0.01,
    )
    assert torch.allclose(dataset[0]["action"], torch.full((3, 30), 0.5))
    assert all(int(dataset.caches[cache_id].episodes[start]) != 2 for cache_id, start in dataset.samples)


def test_multiscale_memory_uses_only_older_rows_from_the_same_episode(
    tmp_path: Path,
) -> None:
    root = _write_cache(
        tmp_path / "cache",
        dataset_id="source",
        name="source_front",
        episodes=[(episode, 0) for episode in range(4)],
        frames_per_episode=90,
    )
    dataset = SO101Sequences(
        [root],
        "train",
        frameskip=5,
        history=3,
        exclusions=set(),
        split_seed=7,
        val_fraction=0.25,
        joint_scale=100.0,
        motion_threshold=0.01,
        memory_offsets=(5, 15, 30, 60),
    )

    sample = dataset[0]
    cache_id, start = dataset.samples[0]
    cache = dataset.caches[cache_id]
    memory_rows = start - np.asarray((5, 15, 30, 60))

    assert sample["memory_pixels"].shape == (4, 3, 4, 4)
    assert sample["memory_action"].shape == (4, 30)
    assert sample["memory_state"].shape == (4, 6)
    assert sample["memory_age"].tolist() == [1.0, 3.0, 6.0, 12.0]
    assert sample["memory_valid"].all()
    np.testing.assert_array_equal(
        sample["memory_state"].numpy(), np.asarray(cache.states[memory_rows])
    )
    assert np.all(cache.episodes[memory_rows] == cache.episodes[start])


def test_global_action_stats_use_training_episodes_only(tmp_path: Path) -> None:
    root = _write_cache(
        tmp_path / "cache",
        dataset_id="source",
        name="source_front",
        episodes=[(episode, 0) for episode in range(8)],
    )
    actions = np.load(root / "actions.npy")
    episodes = np.load(root / "episodes.npy")
    for episode in range(8):
        actions[episodes == episode] = episode * 10.0
    np.save(root / "actions.npy", actions)
    cache = Cache(root)
    assignments = episode_splits([cache], set(), split_seed=7, val_fraction=0.25)

    center, scale, count = global_train_action_stats([cache], assignments, set())
    train_rows = np.asarray(
        [
            row
            for row, episode in enumerate(cache.episodes)
            if assignments[(cache.dataset_id, int(episode))] == "train"
        ]
    )
    expected = np.asarray(cache.actions[train_rows], dtype=np.float32)

    np.testing.assert_allclose(center, expected.mean(axis=0))
    np.testing.assert_allclose(scale, expected.std(axis=0), rtol=1e-5)
    assert count == len(expected)


def test_natural_sampling_preserves_within_task_frequency(tmp_path: Path) -> None:
    root = _write_cache(
        tmp_path / "cache",
        dataset_id="source",
        name="source_front",
        episodes=[(episode, 0) for episode in range(8)],
    )
    dataset = _dataset([root], "train")

    weights = dataset.balanced_weights("source-task-natural")

    assert torch.allclose(weights, torch.full_like(weights, weights[0]))


def test_action_shuffle_never_crosses_source_task_group() -> None:
    actions = torch.arange(8, dtype=torch.float32).reshape(4, 1, 2)
    groups = torch.tensor([0, 0, 1, 1])
    shuffled = _shuffle_actions_within_group(actions, groups)
    assert torch.equal(shuffled[0], actions[1])
    assert torch.equal(shuffled[1], actions[0])
    assert torch.equal(shuffled[2], actions[3])
    assert torch.equal(shuffled[3], actions[2])


def test_no_action_zeros_embedding_not_zero_command() -> None:
    class BiasedActionModel:
        def eval(self):
            return self

        def action_encoder(self, action: torch.Tensor) -> torch.Tensor:
            return action + 1.0

        def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            batch_size = len(batch["pixels"])
            embedding = torch.zeros((batch_size, 4, 1), dtype=torch.float32)
            return {"emb": embedding, "act_emb": self.action_encoder(batch["action"])}

        def predict(
            self, embedding: torch.Tensor, action_embedding: torch.Tensor
        ) -> torch.Tensor:
            return action_embedding

    batch = {
        "pixels": torch.zeros((1, 4, 3, 2, 2)),
        "action": torch.zeros((1, 3, 1)),
        "group": torch.tensor([0]),
        "moving": torch.tensor([True]),
    }
    metrics = evaluate(BiasedActionModel(), [batch], torch.device("cpu"), {0: "task"})

    assert metrics["no_action_mse"] == 0.0
    assert metrics["zero_command_mse"] == 1.0
