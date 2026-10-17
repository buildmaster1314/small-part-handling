from __future__ import annotations

from copy import deepcopy

import pytest

from fcpg.lewm_a4_checkpoint import (
    A4_CHECKPOINT_TYPE,
    A4_SPLITS,
    A4_STATE_KEYS,
    A4_TASK_TO_INDEX,
    TASKS,
    build_a4_checkpoint,
    build_a4_checkpoint_metadata,
    validate_a4_checkpoint,
    validate_a4_checkpoint_metadata,
)


def _episode_counts() -> dict[str, dict[str, int]]:
    per_split = {"train": 24, "development": 12, "test": 0}
    return {
        split: {task: per_split[split] for task in TASKS}
        for split in A4_SPLITS
    }


def _bundle_counts() -> dict[str, dict[str, int]]:
    return {
        split: {
            task: (48 + task_index if split == "train" else 16 + task_index)
            if split != "test"
            else 0
            for task_index, task in enumerate(TASKS)
        }
        for split in A4_SPLITS
    }


def _metadata() -> dict[str, object]:
    return build_a4_checkpoint_metadata(
        seed=30_201,
        step=2_000,
        episode_counts=_episode_counts(),
        bundle_counts=_bundle_counts(),
        cache_identifier="a4-active-control-cache-v1",
        base_identifier="a4-frozen-base-v1",
    )


def test_builds_exact_shared_seven_task_metadata() -> None:
    metadata = _metadata()

    assert metadata["checkpoint_type"] == A4_CHECKPOINT_TYPE
    assert metadata["variant"] == "task_context_persistent"
    assert metadata["num_tasks"] == len(TASKS) == 7
    assert metadata["tasks"] == list(TASKS)
    assert metadata["task_to_index"] == A4_TASK_TO_INDEX
    assert list(metadata["task_to_index"]) == list(TASKS)
    assert metadata["models_per_seed"] == 1
    assert metadata["bundle_batch_size"] == 28
    assert metadata["bundles_per_task_per_batch"] == 4
    assert metadata["sigreg_weight"] == 0.02
    assert metadata["seed"] == 30_201
    assert metadata["step"] == 2_000
    assert metadata["state_keys"] == list(A4_STATE_KEYS)
    assert validate_a4_checkpoint_metadata(metadata) == metadata


def test_complete_checkpoint_requires_both_shared_state_keys() -> None:
    checkpoint = build_a4_checkpoint(
        _metadata(),
        model_state={"predictor.weight": "factual"},
        no_action_state={"predictor.weight": "no-action"},
    )

    assert validate_a4_checkpoint(checkpoint) == checkpoint
    missing = deepcopy(checkpoint)
    del missing["no_action_state"]
    with pytest.raises(ValueError, match="no_action_state"):
        validate_a4_checkpoint(missing)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tasks": TASKS[:-1]}, "complete seven-task panel"),
        ({"tasks": (*TASKS[1:], TASKS[0])}, "frozen order"),
        ({"num_tasks": 6}, "num_tasks must be exactly 7"),
        ({"models_per_seed": 7}, "models_per_seed must be exactly 1"),
        ({"bundle_batch_size": 196}, "bundle_batch_size must be exactly 28"),
        ({"bundles_per_task_per_batch": 3}, "must be exactly 4"),
        ({"variant": "anchor_persistent"}, "variant must be exactly"),
        ({"sigreg_weight": 0.0}, "SIGReg weight must be exactly 0.02"),
        ({"sigreg_weight": 0.09}, "SIGReg weight must be exactly 0.02"),
    ],
)
def test_constructor_rejects_non_shared_or_drifted_contract(
    overrides: dict[str, object], message: str
) -> None:
    arguments = {
        "seed": 30_201,
        "step": 2_000,
        "episode_counts": _episode_counts(),
        "bundle_counts": _bundle_counts(),
        "cache_identifier": "cache-v1",
        "base_identifier": "base-v1",
        **overrides,
    }

    with pytest.raises(ValueError, match=message):
        build_a4_checkpoint_metadata(**arguments)  # type: ignore[arg-type]


def test_validator_rejects_missing_reordered_or_misindexed_tasks() -> None:
    metadata = _metadata()

    missing = deepcopy(metadata)
    del missing["task_to_index"][TASKS[-1]]  # type: ignore[index]
    with pytest.raises(ValueError, match="task_to_index.*missing"):
        validate_a4_checkpoint_metadata(missing)

    reordered = deepcopy(metadata)
    reordered["task_to_index"] = dict(
        reversed(list(reordered["task_to_index"].items()))  # type: ignore[union-attr]
    )
    with pytest.raises(ValueError, match="frozen A4 order"):
        validate_a4_checkpoint_metadata(reordered)

    misindexed = deepcopy(metadata)
    misindexed["task_to_index"][TASKS[0]] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="must match the frozen TASKS order"):
        validate_a4_checkpoint_metadata(misindexed)


@pytest.mark.parametrize("count_name", ["episode_counts", "bundle_counts"])
def test_counts_require_every_split_and_every_ordered_task(count_name: str) -> None:
    metadata = _metadata()
    missing_task = deepcopy(metadata)
    del missing_task[count_name]["train"][TASKS[-1]]  # type: ignore[index]
    with pytest.raises(ValueError, match=rf"{count_name}.*missing"):
        validate_a4_checkpoint_metadata(missing_task)

    missing_split = deepcopy(metadata)
    del missing_split[count_name]["development"]  # type: ignore[index]
    with pytest.raises(ValueError, match=rf"{count_name}.*missing"):
        validate_a4_checkpoint_metadata(missing_split)

    reordered_tasks = deepcopy(metadata)
    table = reordered_tasks[count_name]["train"]  # type: ignore[index]
    reordered_tasks[count_name]["train"] = dict(  # type: ignore[index]
        reversed(list(table.items()))
    )
    with pytest.raises(ValueError, match="frozen A4 order"):
        validate_a4_checkpoint_metadata(reordered_tasks)


def test_counts_and_identifiers_are_simple_but_fail_closed() -> None:
    metadata = _metadata()
    bundle_without_episode = deepcopy(metadata)
    bundle_without_episode["bundle_counts"]["test"][TASKS[0]] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="zero episodes"):
        validate_a4_checkpoint_metadata(bundle_without_episode)

    negative = deepcopy(metadata)
    negative["episode_counts"]["train"][TASKS[0]] = -1  # type: ignore[index]
    with pytest.raises(ValueError, match="at least 0"):
        validate_a4_checkpoint_metadata(negative)

    for field in ("cache_identifier", "base_identifier"):
        invalid = deepcopy(metadata)
        invalid[field] = " "
        with pytest.raises(ValueError, match="stable identifier"):
            validate_a4_checkpoint_metadata(invalid)


def test_seven_task_specific_state_maps_are_rejected() -> None:
    task_specific = {task: {"weight": task} for task in TASKS}

    with pytest.raises(ValueError, match="seven task-specific models"):
        build_a4_checkpoint(
            _metadata(),
            model_state=task_specific,
            no_action_state={"weight": "shared"},
        )


def test_metadata_and_payload_reject_unregistered_keys() -> None:
    metadata = _metadata()
    metadata["optimizer_variant"] = "extra"
    with pytest.raises(ValueError, match="unexpected"):
        validate_a4_checkpoint_metadata(metadata)

    checkpoint = build_a4_checkpoint(
        _metadata(), model_state={"weight": 1}, no_action_state={"weight": 1}
    )
    checkpoint["task_models"] = {}
    with pytest.raises(ValueError, match="unexpected"):
        validate_a4_checkpoint(checkpoint)
