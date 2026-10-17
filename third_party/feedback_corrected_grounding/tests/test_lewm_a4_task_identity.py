from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from fcpg.lewm_a4_task_identity import (
    A4TaskIdentity,
    IDENTITY_KEYS,
    TASKS,
    TASK_CLASS_COUNT,
    TASK_CLASS_INDICES,
    TASK_CLASS_INDEX_TO_NAME,
    TASK_INSTANCE_COUNT,
    TASK_INSTANCE_INDICES,
    TASK_NAME_TO_CLASS_INDEX,
    VALID_SPLITS,
    task_class_index_to_name,
    task_name_to_class_index,
    validate_a4_task_identity,
    validate_a4_task_identity_arrays,
    validate_a4_task_identity_vectors,
    validate_source_episode_id,
    validate_split,
    validate_task_class_index,
    validate_task_instance_index,
    validate_task_name,
)


EXPECTED_TASKS = (
    "pick-place-v3",
    "drawer-retrieve-v0",
    "reach-v3",
    "push-v3",
    "door-open-v3",
    "window-open-v3",
    "towel-wipe-v0",
)


def _valid_vectors() -> dict[str, np.ndarray]:
    return {
        "task_name": np.asarray(
            [TASKS[0], TASKS[0], TASKS[1], TASKS[1], TASKS[1]]
        ),
        "task_class_index": np.asarray([0, 0, 1, 1, 1], dtype=np.int16),
        "task_instance_index": np.asarray([7, 7, 7, 7, 8], dtype=np.int16),
        # Reuse across tasks is intentional: source identity is task-scoped.
        "source_episode_id": np.asarray(
            ["episode-7", "episode-7", "episode-7", "episode-7", "episode-8"]
        ),
        "split": np.asarray(
            ["train", "train", "development", "development", "test"]
        ),
    }


def test_frozen_order_and_bidirectional_mapping_are_exact() -> None:
    assert TASKS == EXPECTED_TASKS
    assert TASK_CLASS_COUNT == 7
    assert TASK_INSTANCE_COUNT == 48
    assert TASK_CLASS_INDICES == tuple(range(7))
    assert TASK_INSTANCE_INDICES == tuple(range(48))
    assert VALID_SPLITS == ("train", "development", "test")
    assert tuple(TASK_NAME_TO_CLASS_INDEX) == TASKS

    for index, name in enumerate(TASKS):
        assert TASK_NAME_TO_CLASS_INDEX[name] == index
        assert TASK_CLASS_INDEX_TO_NAME[index] == name
        assert task_name_to_class_index(np.str_(name)) == index
        assert task_class_index_to_name(np.int64(index)) == name

    with pytest.raises(TypeError):
        TASK_NAME_TO_CLASS_INDEX[TASKS[0]] = 6  # type: ignore[index]


@pytest.mark.parametrize("value", [True, np.bool_(False)])
def test_boolean_values_are_never_integer_identity_indices(value: object) -> None:
    with pytest.raises(TypeError, match="task_class_index.*not bool"):
        validate_task_class_index(value)
    with pytest.raises(TypeError, match="task_instance_index.*not bool"):
        validate_task_instance_index(value)
    with pytest.raises(TypeError, match="source_episode_id.*bool"):
        validate_source_episode_id(value)


def test_scalar_validators_support_numpy_scalars_and_reject_drift() -> None:
    assert validate_task_name(np.str_(TASKS[2])) == TASKS[2]
    assert validate_task_name(np.bytes_(TASKS[2])) == TASKS[2]
    assert validate_task_class_index(np.uint8(6)) == 6
    assert validate_task_instance_index(np.int32(0)) == 0
    assert validate_task_instance_index(np.int64(47)) == 47
    assert validate_source_episode_id(np.str_("episode-a")) == "episode-a"
    assert validate_source_episode_id(np.bytes_(b"episode-b")) == "episode-b"
    assert validate_source_episode_id(np.int64(19)) == 19
    assert validate_split(np.str_("development")) == "development"
    assert validate_split(np.bytes_(b"test")) == "test"

    for value in (-1, 7, 1.0, "1"):
        with pytest.raises((TypeError, ValueError), match="task_class_index"):
            validate_task_class_index(value)
    for value in (-1, 48, 1.0, "1"):
        with pytest.raises((TypeError, ValueError), match="task_instance_index"):
            validate_task_instance_index(value)
    for value in ("", "   ", -1, 1.0, object()):
        with pytest.raises((TypeError, ValueError), match="source_episode_id"):
            validate_source_episode_id(value)
    with pytest.raises(ValueError, match="valid UTF-8"):
        validate_source_episode_id(b"\xff")
    with pytest.raises(ValueError, match="unknown A4 task_name"):
        validate_task_name("drawer-open-v3")
    with pytest.raises(ValueError, match="unsupported A4 split"):
        validate_split("validation")


def test_one_row_identity_normalizes_and_binds_name_to_class() -> None:
    identity = validate_a4_task_identity(
        np.str_(TASKS[4]),
        np.int64(4),
        np.uint8(47),
        np.int32(91),
        np.str_("test"),
    )

    assert identity == A4TaskIdentity(TASKS[4], 4, 47, 91, "test")
    assert isinstance(identity.task_class_index, int)
    assert isinstance(identity.task_instance_index, int)
    assert isinstance(identity.source_episode_id, int)
    with pytest.raises(FrozenInstanceError):
        identity.task_class_index = 1  # type: ignore[misc]

    with pytest.raises(ValueError, match="not MetaWorld task_index/task instance"):
        validate_a4_task_identity(TASKS[0], 4, 4, "episode", "train")


def test_valid_vectors_are_normalized_read_only_and_mapping_reusable() -> None:
    arrays = _valid_vectors()
    validated = validate_a4_task_identity_vectors(
        *(arrays[name] for name in IDENTITY_KEYS)
    )

    assert validated.sample_count == 5
    assert validated.task_class_index.dtype == np.int64
    assert validated.task_instance_index.dtype == np.int64
    assert validated.source_episode_id.dtype.kind == "U"
    assert tuple(validated.task_name) == tuple(arrays["task_name"])
    assert all(not value.flags.writeable for value in validated.as_arrays().values())

    arrays["task_name"][0] = TASKS[-1]
    assert validated.task_name[0] == TASKS[0]
    with pytest.raises(ValueError, match="read-only"):
        validated.split[0] = "test"

    copied = validated.as_arrays(copy=True)
    assert tuple(copied) == IDENTITY_KEYS
    assert all(value.flags.writeable for value in copied.values())
    copied["split"][0] = "test"
    assert validated.split[0] == "train"

    larger_mapping = {**_valid_vectors(), "history_state": np.zeros((5, 1))}
    from_mapping = validate_a4_task_identity_arrays(larger_mapping)
    assert from_mapping.sample_count == 5


def test_integer_episode_vectors_and_numpy_boolean_rejection() -> None:
    validated = validate_a4_task_identity_vectors(
        np.asarray([TASKS[2], TASKS[2]]),
        np.asarray([2, 2], dtype=np.uint8),
        np.asarray([0, 1], dtype=np.uint8),
        np.asarray([10, 11], dtype=np.uint32),
        np.asarray(["train", "development"]),
    )
    assert validated.source_episode_id.dtype == np.int64
    np.testing.assert_array_equal(validated.source_episode_id, [10, 11])

    with pytest.raises(TypeError, match="row 0.*not bool"):
        validate_a4_task_identity_vectors(
            [TASKS[0]], [np.bool_(False)], [0], ["episode"], ["train"]
        )
    with pytest.raises(TypeError, match="row 0.*not bool"):
        validate_a4_task_identity_vectors(
            [TASKS[0]], [0], [True], ["episode"], ["train"]
        )
    with pytest.raises(TypeError, match="row 0.*bool"):
        validate_a4_task_identity_vectors(
            [TASKS[0]], [0], [0], [np.bool_(True)], ["train"]
        )


def test_vector_shapes_lengths_and_types_fail_closed() -> None:
    arrays = _valid_vectors()
    with pytest.raises(ValueError, match="equal length"):
        validate_a4_task_identity_vectors(
            arrays["task_name"][:-1],
            arrays["task_class_index"],
            arrays["task_instance_index"],
            arrays["source_episode_id"],
            arrays["split"],
        )
    with pytest.raises(ValueError, match="must not be empty"):
        validate_a4_task_identity_vectors([], [], [], [], [])
    with pytest.raises(ValueError, match=r"task_name must have shape \[N\]"):
        validate_a4_task_identity_vectors(
            np.asarray([[TASKS[0]]]), [0], [0], ["episode"], ["train"]
        )
    with pytest.raises(TypeError, match="one-dimensional vector"):
        validate_a4_task_identity_vectors(
            TASKS[0], [0], [0], ["episode"], ["train"]  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="must not mix"):
        validate_a4_task_identity_vectors(
            [TASKS[0], TASKS[0]],
            [0, 0],
            [0, 1],
            ["episode", 2],
            ["train", "train"],
        )


def test_name_class_instance_and_split_cross_row_drift_is_rejected() -> None:
    arrays = _valid_vectors()
    arrays["task_class_index"][0] = 1
    with pytest.raises(ValueError, match="task_name/task_class_index mismatch"):
        validate_a4_task_identity_arrays(arrays)

    arrays = _valid_vectors()
    arrays["task_instance_index"][0] = 48
    with pytest.raises(ValueError, match="inclusive range 0..47"):
        validate_a4_task_identity_arrays(arrays)

    arrays = _valid_vectors()
    arrays["split"][0] = "validation"
    with pytest.raises(ValueError, match="unsupported A4 split"):
        validate_a4_task_identity_arrays(arrays)


def test_source_episode_and_instance_are_one_to_one_within_task() -> None:
    with pytest.raises(ValueError, match="source_episode_id changes"):
        validate_a4_task_identity_vectors(
            [TASKS[0], TASKS[0]],
            [0, 0],
            [2, 3],
            ["same", "same"],
            ["train", "train"],
        )

    with pytest.raises(ValueError, match="multiple source_episode_id"):
        validate_a4_task_identity_vectors(
            [TASKS[0], TASKS[0]],
            [0, 0],
            [2, 2],
            ["first", "second"],
            ["train", "train"],
        )


def test_source_physical_episode_cannot_cross_splits() -> None:
    with pytest.raises(ValueError, match="source_episode_id.*multiple splits"):
        validate_a4_task_identity_vectors(
            [TASKS[3], TASKS[3]],
            [3, 3],
            [12, 12],
            ["episode-12", "episode-12"],
            ["train", "development"],
        )


def test_ambiguous_metaworld_task_index_field_is_explicitly_forbidden() -> None:
    arrays = _valid_vectors()
    arrays["task_index"] = arrays.pop("task_class_index")
    with pytest.raises(ValueError, match="ambiguous task_index is forbidden"):
        validate_a4_task_identity_arrays(arrays)

    arrays = _valid_vectors()
    arrays.pop("source_episode_id")
    with pytest.raises(ValueError, match="missing keys.*source_episode_id"):
        validate_a4_task_identity_arrays(arrays)
    with pytest.raises(TypeError, match="must be a mapping"):
        validate_a4_task_identity_arrays([])  # type: ignore[arg-type]
