from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from fcpg.lewm_a4_storage import (
    ACTION_DIM,
    LATENT_DIM,
    STATE_SLOT_COUNT,
)
from fcpg.lewm_action_branch_a4 import (
    BRANCH_COUNT,
    CONFIRMATION_TRAINING_SEEDS,
    HISTORY_STEPS,
    HORIZON,
    TASKS,
)
from fcpg.lewm_a4_physical_state_decoding import (
    AUDIT_DESIGNATION,
    MATCHED_DIRECT_ACTION_FEATURE_DIM,
    STAGE,
    build_a4_physical_decoding_report,
    decode_a4_evaluation_arrays,
    fit_a4_physical_decoder,
    fit_fixed_a4_baselines,
)
from fcpg.lewm_action_branch_a4_metrics import RIDGE_ALPHA
from scripts import run_lewm_a4_physical_state_decoding as cli


def _formal_panel() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rows: list[tuple[int, int, str]] = []
    for task_index in range(len(TASKS)):
        for local_episode in range(4):
            rows.append(
                (
                    task_index,
                    local_episode,
                    "train" if local_episode < 2 else "development",
                )
            )
    count = len(rows)
    task = np.asarray([row[0] for row in rows], dtype=np.int64)
    local_episode = np.asarray([row[1] for row in rows], dtype=np.int64)
    split = np.asarray([row[2] for row in rows])
    proposal = np.asarray([f"physical-proposal-{index}" for index in range(count)])
    task_name = np.asarray([TASKS[index] for index in task])
    episode = np.asarray(
        [f"{TASKS[row[0]]}:{row[2]}:{row[1]}" for row in rows]
    )

    branch = np.arange(BRANCH_COUNT, dtype=np.float32)[None, :, None]
    horizon = np.arange(1, HORIZON + 1, dtype=np.float32)[None, None, :]
    progress = (
        np.float32(0.04)
        + np.float32(0.004) * task[:, None, None]
        + np.float32(0.006) * local_episode[:, None, None]
        + np.float32(0.007) * branch
        + np.float32(0.003) * horizon
    )
    q0 = np.stack(
        (
            np.float32(-0.1) + np.float32(0.02) * task + np.float32(0.003) * local_episode,
            np.float32(0.45) + np.float32(0.025) * task + np.float32(0.002) * local_episode,
            np.float32(0.08) + np.float32(0.01) * task + np.float32(0.004) * local_episode,
        ),
        axis=1,
    ).astype(np.float32)
    displacement = np.stack(
        (
            np.float32(0.001) * (branch + horizon),
            np.float32(0.0007) * (np.float32(2.0) * branch - horizon),
            np.float32(0.0005) * (branch + np.float32(2.0) * horizon),
        ),
        axis=-1,
    )
    position = q0[:, None, None, :] + displacement

    target = np.zeros(
        (count, BRANCH_COUNT, HORIZON, STATE_SLOT_COUNT, LATENT_DIM),
        dtype=np.float32,
    )
    target[..., 0, 0] = progress
    target[..., 0, 1:4] = position
    target[..., 1, 0] = np.float32(1.1) * progress
    target[..., 1, 1:4] = np.float32(0.9) * position
    target[..., 2, 0] = branch + np.float32(0.1) * horizon
    target[..., 3, 0] = horizon + np.float32(0.1) * branch

    history = np.zeros(
        (count, HISTORY_STEPS, STATE_SLOT_COUNT, LATENT_DIM), dtype=np.float32
    )
    history[:, :, 0, 1:4] = q0[:, None, :]
    history[:, :, 1, 1:4] = np.float32(0.9) * q0[:, None, :]
    history[:, :, 1, 0] = (
        np.float32(0.02) + np.float32(0.002) * local_episode[:, None]
    )
    history[:, 1:, 2, 0] = np.asarray((0.01, 0.02), dtype=np.float32)

    future_action = np.zeros(
        (count, BRANCH_COUNT, HORIZON, ACTION_DIM), dtype=np.float32
    )
    future_action[..., :3] = displacement
    future_action[..., 3] = progress
    future_action[..., 4] = branch + np.float32(0.1) * horizon
    center_action = np.mean(future_action, axis=1, dtype=np.float32)
    history_action = np.zeros(
        (count, HISTORY_STEPS - 1, ACTION_DIM), dtype=np.float32
    )
    history_action[:, 0, :3] = q0 * np.float32(0.1)
    history_action[:, 1, :3] = q0 * np.float32(0.2)

    cache = {
        "proposal_id": proposal,
        "task_class_index": task,
        "task_name": task_name,
        "task_instance_index": local_episode,
        "source_episode_id": episode,
        "split": split,
        "history_state": history,
        "history_action": history_action,
        "center_action": center_action,
        "future_action": future_action,
        "target_state": target,
        "branch_semantic_index": np.broadcast_to(
            np.arange(BRANCH_COUNT, dtype=np.int64), (count, BRANCH_COUNT)
        ).copy(),
    }

    object_pose = np.zeros((count, 7), dtype=np.float32)
    object_pose[:, :3] = q0
    object_pose[:, 3] = 1.0
    proprio = np.zeros((count, 4), dtype=np.float32)
    proprio[:, :3] = q0 + np.float32(0.03)
    proprio[:, 3] = np.float32(0.5)
    reach = task == TASKS.index("reach-v3")
    proprio[reach, :3] = q0[reach]
    physical = np.zeros((count, BRANCH_COUNT, HORIZON, 7), dtype=np.float32)
    physical[..., :3] = position
    physical[..., 3] = 1.0
    outcome = np.zeros((count, BRANCH_COUNT, HORIZON, 2), dtype=np.float32)
    outcome[..., 0] = progress
    outcome[..., 1] = progress > np.float32(0.2)
    labels = {
        "proposal_id": proposal.copy(),
        "task_class_index": task.copy(),
        "task_name": task_name.copy(),
        "task_instance_index": local_episode.copy(),
        "source_episode_id": episode.copy(),
        "split": split.copy(),
        "branch_valid": np.ones((count, BRANCH_COUNT), dtype=bool),
        "future_physical_spread": np.full(count, 0.1, dtype=np.float32),
        "source_expert_success": np.ones(count, dtype=bool),
        "proprio": proprio,
        "anchor_validity": np.ones(
            (count, HISTORY_STEPS, STATE_SLOT_COUNT - 1), dtype=bool
        ),
        "object_pose": object_pose,
        "goal_relative_progress": np.zeros((count, 1), dtype=np.float32),
        "contact_or_grasp_state": np.zeros((count, 1), dtype=bool),
        "branch_outcome": outcome,
        "physical_state": physical,
        "branch_gripper_aperture": np.zeros(
            (count, BRANCH_COUNT, HORIZON), dtype=np.float32
        ),
        "branch_contact_or_grasp": np.zeros(
            (count, BRANCH_COUNT, HORIZON), dtype=bool
        ),
    }
    return cache, labels


def _evaluation_arrays(
    cache: dict[str, np.ndarray], seed: int
) -> dict[str, np.ndarray]:
    development_rows = np.flatnonzero(cache["split"].astype(str) == "development")
    target = cache["target_state"][development_rows]
    factual = target.copy()
    factual[..., 0, 0] += np.float32((seed - 30300) * 0.0002)
    no_action_center = np.mean(target, axis=1, keepdims=True, dtype=np.float32)
    no_action = np.broadcast_to(no_action_center, target.shape).copy()
    wrong = np.empty(
        (target.shape[0], BRANCH_COUNT - 1, *target.shape[1:]),
        dtype=np.float32,
    )
    slots = np.arange(BRANCH_COUNT)
    for shift in range(1, BRANCH_COUNT):
        wrong[:, shift - 1] = factual[:, (slots + shift) % BRANCH_COUNT]
    return {
        "checkpoint_step": np.asarray(10_000, dtype=np.int64),
        "selection": np.asarray("fixed_latest"),
        "training_seed": np.full(target.shape[0], seed, dtype=np.int64),
        "sample_index": development_rows,
        "target_state": target,
        "factual_prediction": factual,
        "no_action_prediction": no_action,
        "shuffled_action_prediction": wrong,
        "shuffled_action_donor_shifts": np.arange(
            1, BRANCH_COUNT, dtype=np.int64
        ),
    }


def test_decoder_is_train_only_and_hyperparameters_are_source_fixed() -> None:
    cache, labels = _formal_panel()
    decoder = fit_a4_physical_decoder(cache, labels)
    changed = deepcopy(labels)
    development = changed["split"].astype(str) == "development"
    changed["physical_state"][development, ..., :3] += np.float32(25.0)
    changed["branch_outcome"][development, ..., 0] = np.float32(0.99)
    repeated = fit_a4_physical_decoder(cache, changed)
    np.testing.assert_allclose(decoder.coefficient, repeated.coefficient)
    np.testing.assert_allclose(
        decoder.anchor_coefficient, repeated.anchor_coefficient
    )
    assert decoder.ridge_alpha == RIDGE_ALPHA
    assert MATCHED_DIRECT_ACTION_FEATURE_DIM == 4 * LATENT_DIM

    with pytest.raises(ValueError, match="development tuning is prohibited"):
        fit_a4_physical_decoder(cache, labels, ridge_alpha=RIDGE_ALPHA * 2.0)

    missing = dict(labels)
    missing.pop("physical_state")
    with pytest.raises(ValueError, match="exact companion schema"):
        fit_a4_physical_decoder(cache, missing)


def test_exposed_development_report_is_seed_separate_and_crossed() -> None:
    cache, labels = _formal_panel()
    decoder = fit_a4_physical_decoder(cache, labels)
    baselines = fit_fixed_a4_baselines(cache, labels, decoder)
    decoded = tuple(
        decode_a4_evaluation_arrays(decoder, cache, _evaluation_arrays(cache, seed))
        for seed in CONFIRMATION_TRAINING_SEEDS
    )
    manifest = [
        {
            "seed": seed,
            "step": 10_000,
            "selection": "fixed_latest",
            "sha256": f"{seed:064x}",
        }
        for seed in CONFIRMATION_TRAINING_SEEDS
    ]
    report, archive = build_a4_physical_decoding_report(
        cache,
        labels,
        decoder,
        decoded,
        checkpoint_manifest=manifest,
        fixed_baselines=baselines,
        bootstrap_samples=64,
        bootstrap_seed=17,
    )

    assert report["stage"] == STAGE
    assert report["audit_designation"] == AUDIT_DESIGNATION
    assert report["decision"] == "descriptive_only_no_gate"
    assert report["confirmation_claim_authorized"] is False
    assert report["aggregation"]["predictions_averaged_across_lewm_seeds"] is False
    assert set(report["horizon_windows"]) == {"h1_h8", "h6_h8"}

    progress = report["horizon_windows"]["h1_h8"]["readouts"]["task_progress"]
    assert "q0_physical_hold" in progress["unavailable_methods"]
    assert set(progress["methods"]["factual"]["per_seed"]) == {
        str(seed) for seed in CONFIRMATION_TRAINING_SEEDS
    }
    assert progress["methods"]["wrong_action_shifts"]["count"] == 7
    comparison = progress["comparisons"]["factual_vs_fixed_baselines"][
        "matched_no_action"
    ]
    crossed = comparison["crossed_seed_physical_episode_bootstrap"]
    assert crossed["unit"] == "lewm_seed_x_source_physical_episode_within_task"
    assert crossed["lewm_seed_count"] == 3
    assert crossed["crossed_product_reduction"] is True
    assert set(comparison["per_seed"]) == {
        str(seed) for seed in CONFIRMATION_TRAINING_SEEDS
    }

    position = report["horizon_windows"]["h6_h8"]["readouts"][
        "task_controlled_position"
    ]
    assert "q0_physical_hold" in position["methods"]
    assert position["development_mechanism_evidence"]["gate_configured"] is False
    assert archive["factual"].shape[0] == 3
    assert archive["anchor_only_factual"].shape == archive["factual"].shape
    assert archive["q0_position_raw"].shape[-1] == 3


def test_wrong_action_replay_and_reserved_test_fail_closed() -> None:
    cache, labels = _formal_panel()
    decoder = fit_a4_physical_decoder(cache, labels)
    arrays = _evaluation_arrays(cache, CONFIRMATION_TRAINING_SEEDS[0])
    arrays["shuffled_action_prediction"] = arrays[
        "shuffled_action_prediction"
    ].copy()
    arrays["shuffled_action_prediction"][0, 0, 0, 0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="exact cyclic replay"):
        decode_a4_evaluation_arrays(decoder, cache, arrays)

    bad_cache = dict(cache)
    bad_cache["split"] = cache["split"].copy()
    bad_cache["split"][0] = "test"
    bad_labels = dict(labels)
    bad_labels["split"] = bad_cache["split"].copy()
    with pytest.raises(PermissionError, match="reserved-test"):
        fit_a4_physical_decoder(bad_cache, bad_labels)


def test_cli_smoke_subset_is_episode_bounded_and_hash_bound(
    tmp_path: Path,
) -> None:
    cache, labels = _formal_panel()
    smoke_cache, smoke_labels = cli._smoke_subset(cache, labels)
    task = smoke_cache["task_name"].astype(str)
    split = smoke_cache["split"].astype(str)
    episode = smoke_cache["source_episode_id"].astype(str)
    for task_name in TASKS:
        assert np.unique(episode[(task == task_name) & (split == "train")]).size == 2
        assert (
            np.unique(
                episode[(task == task_name) & (split == "development")]
            ).size
            == 1
        )
    assert np.array_equal(smoke_cache["proposal_id"], smoke_labels["proposal_id"])

    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"bound-content")
    bound = cli._bind_file(
        artifact,
        "0759a88fc7e50aad8b1725056b3646bcdfe8b86bd97bd4aaf34785d97a679f99",
        role="test artifact",
    )
    assert bound["path"] == str(artifact.resolve())
    artifact.write_bytes(b"changed")
    with pytest.raises(PermissionError, match="sha256 drifted"):
        cli._bind_file(
            artifact,
            bound["sha256"],
            role="test artifact",
        )
