from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from fcpg.lewm_a4_sampling import A4EpisodeBalancedBatchSampler
from fcpg.lewm_a4_storage import ACTION_DIM, LATENT_DIM, STATE_SLOT_COUNT
from fcpg.lewm_a4_training_data import (
    A4StateCacheDataset,
    build_a4_state_cache_dataloader,
    build_a4_state_cache_loader,
)
from fcpg.lewm_action_branch_a4 import (
    BRANCH_COUNT,
    BUNDLE_BATCH_SIZE,
    BUNDLES_PER_TASK_PER_BATCH,
    HISTORY_STEPS,
    HORIZON,
    TASKS,
)


SAMPLE_KEYS = {
    "history_embedding",
    "history_action",
    "future_action",
    "target_embedding",
    "task_index",
    "task_name",
    "source_episode_id",
}


def _cache(
    *,
    episodes_per_task: int = 4,
    splits: tuple[str, ...] = ("train",),
) -> dict[str, np.ndarray]:
    rows = [
        (task_index, split, episode)
        for task_index in range(len(TASKS))
        for split in splits
        for episode in range(episodes_per_task)
    ]
    count = len(rows)
    history_state = np.zeros(
        (count, HISTORY_STEPS, STATE_SLOT_COUNT, LATENT_DIM), dtype=np.float32
    )
    history_action = np.zeros(
        (count, HISTORY_STEPS - 1, ACTION_DIM), dtype=np.float32
    )
    center_action = np.zeros((count, HORIZON, ACTION_DIM), dtype=np.float32)
    future_action = np.zeros(
        (count, BRANCH_COUNT, HORIZON, ACTION_DIM), dtype=np.float32
    )
    target_state = np.zeros(
        (count, BRANCH_COUNT, HORIZON, STATE_SLOT_COUNT, LATENT_DIM),
        dtype=np.float32,
    )
    markers = np.arange(count, dtype=np.float32)
    history_state[:, 0, 0, 0] = markers
    history_action[:, 0, 0] = markers + 100
    future_action[:, 0, 0, 0] = markers + 200
    target_state[:, 0, 0, 0, 0] = markers + 300
    return {
        "proposal_id": np.asarray([f"proposal-{index}" for index in range(count)]),
        "task_class_index": np.asarray(
            [row[0] for row in rows], dtype=np.int64
        ),
        "task_name": np.asarray([TASKS[row[0]] for row in rows]),
        "task_instance_index": np.asarray(
            [row[2] for row in rows], dtype=np.int64
        ),
        "source_episode_id": np.asarray(
            [f"{row[1]}:{TASKS[row[0]]}:episode-{row[2]}" for row in rows]
        ),
        "split": np.asarray([row[1] for row in rows]),
        "history_state": history_state,
        "history_action": history_action,
        "center_action": center_action,
        "future_action": future_action,
        "target_state": target_state,
        "branch_semantic_index": np.tile(
            np.arange(BRANCH_COUNT, dtype=np.int64), (count, 1)
        ),
    }


def _filter_rows(
    cache: dict[str, np.ndarray], keep: np.ndarray
) -> dict[str, np.ndarray]:
    return {name: np.array(value[keep], copy=True) for name, value in cache.items()}


def test_dataset_filters_split_and_preserves_task_class_mapping() -> None:
    cache = _cache(splits=("train", "development"))
    dataset = A4StateCacheDataset(
        cache,
        split="development",
        dtype=torch.float64,
    )

    assert len(dataset) == len(TASKS) * 4
    assert set(dataset[0]) == SAMPLE_KEYS
    assert dataset.source_indices[0] == 4
    assert dataset.task_class_index == tuple(
        task_index for task_index in range(len(TASKS)) for _ in range(4)
    )

    first = dataset[0]
    assert first["history_embedding"].dtype == torch.float64
    assert first["history_action"].dtype == torch.float64
    assert first["future_action"].dtype == torch.float64
    assert first["target_embedding"].dtype == torch.float64
    assert first["task_index"].dtype == torch.long
    assert first["task_index"].item() == 0
    assert first["task_name"] == TASKS[0]
    assert first["source_episode_id"].startswith("development:")
    assert first["history_embedding"][0, 0, 0].item() == 4.0


def test_dataset_owns_cache_and_returned_tensor_storage() -> None:
    cache = _cache()
    dataset = A4StateCacheDataset(cache, split="train")
    expected = dataset[0]

    cache["history_state"][0, 0, 0, 0] = 999.0
    cache["task_class_index"][0] = 6
    cache["task_name"] = np.full_like(cache["task_name"], TASKS[-1])
    cache.clear()
    after_input_mutation = dataset[0]
    torch.testing.assert_close(
        after_input_mutation["history_embedding"], expected["history_embedding"]
    )
    assert after_input_mutation["task_index"].item() == 0
    assert after_input_mutation["task_name"] == TASKS[0]

    expected["history_embedding"].fill_(123.0)
    expected["task_index"].fill_(6)
    after_output_mutation = dataset[0]
    assert after_output_mutation["history_embedding"][0, 0, 0].item() == 0.0
    assert after_output_mutation["task_index"].item() == 0


def test_dataset_requires_one_unified_seven_task_cache_and_split() -> None:
    cache = _cache(splits=("train", "development"))
    keep = cache["task_class_index"] != len(TASKS) - 1
    with pytest.raises(ValueError, match="seven A4 tasks exactly"):
        A4StateCacheDataset(_filter_rows(cache, keep), split="train")

    task_six_train = (cache["task_class_index"] == len(TASKS) - 1) & (
        cache["split"] == "train"
    )
    cache["split"][task_six_train] = "development"
    with pytest.raises(ValueError, match="split 'train'.*seven A4 tasks"):
        A4StateCacheDataset(cache, split="train")

    with pytest.raises(ValueError, match="split must be one of"):
        A4StateCacheDataset(_cache(), split="validation")


def test_loader_uses_only_strict_seven_by_four_episode_batches() -> None:
    cache = _cache(episodes_per_task=5)
    loader = build_a4_state_cache_loader(
        cache,
        split="train",
        seed=83,
        batches_per_epoch=2,
    )

    assert isinstance(loader.dataset, A4StateCacheDataset)
    assert isinstance(loader.batch_sampler, A4EpisodeBalancedBatchSampler)
    assert loader.batch_size is None
    assert len(loader) == 2
    expected_tasks = [
        task_index
        for task_index in range(len(TASKS))
        for _ in range(BUNDLES_PER_TASK_PER_BATCH)
    ]
    first_pass = []
    for batch in loader:
        assert set(batch) == SAMPLE_KEYS
        assert batch["history_embedding"].shape[0] == BUNDLE_BATCH_SIZE
        assert batch["task_index"].tolist() == expected_tasks
        assert batch["task_name"] == [
            task
            for task in TASKS
            for _ in range(BUNDLES_PER_TASK_PER_BATCH)
        ]
        episodes = list(batch["source_episode_id"])
        first_pass.append(episodes)
        for task_offset in range(len(TASKS)):
            start = task_offset * BUNDLES_PER_TASK_PER_BATCH
            selected = episodes[start : start + BUNDLES_PER_TASK_PER_BATCH]
            assert len(set(selected)) == BUNDLES_PER_TASK_PER_BATCH
    assert [list(batch["source_episode_id"]) for batch in loader] == first_pass


def test_loader_rejects_insufficient_episodes_and_forbidden_controls() -> None:
    cache = _cache(episodes_per_task=3)
    with pytest.raises(ValueError, match="at least 4 distinct source episodes"):
        build_a4_state_cache_loader(cache, split="train", seed=1)

    parameters = inspect.signature(build_a4_state_cache_loader).parameters
    assert "shuffle" not in parameters
    assert "task_names" not in parameters
    with pytest.raises(TypeError, match="unexpected keyword argument 'shuffle'"):
        build_a4_state_cache_loader(  # type: ignore[call-arg]
            _cache(), split="train", seed=1, shuffle=True
        )
    with pytest.raises(TypeError, match="unexpected keyword argument 'task_subset'"):
        build_a4_state_cache_loader(  # type: ignore[call-arg]
            _cache(), split="train", seed=1, task_subset=TASKS[:-1]
        )


def test_torch_dtype_mapping_and_loader_option_validation() -> None:
    cache = _cache()
    with pytest.raises(TypeError, match="dtype must be a torch.dtype"):
        A4StateCacheDataset(
            cache,
            split="train",
            dtype=np.float32,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="floating torch.dtype"):
        A4StateCacheDataset(cache, split="train", dtype=torch.long)
    with pytest.raises(TypeError, match="numpy ndarray"):
        invalid = dict(cache)
        invalid["history_state"] = torch.zeros(1)  # type: ignore[assignment]
        A4StateCacheDataset(invalid, split="train")
    with pytest.raises(TypeError, match="num_workers must be an integer"):
        build_a4_state_cache_loader(cache, split="train", seed=1, num_workers=True)
    with pytest.raises(ValueError, match="num_workers must be non-negative"):
        build_a4_state_cache_loader(cache, split="train", seed=1, num_workers=-1)
    with pytest.raises(TypeError, match="pin_memory must be boolean"):
        build_a4_state_cache_loader(
            cache,
            split="train",
            seed=1,
            pin_memory=1,  # type: ignore[arg-type]
        )

    assert build_a4_state_cache_dataloader is build_a4_state_cache_loader
