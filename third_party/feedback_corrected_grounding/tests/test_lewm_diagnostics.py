from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import scripts.evaluate_lewm_grounded as evaluator
from fcpg.lewm_sequence_data import LeWMSequenceDataset
from scripts.evaluate_lewm_grounded import (
    SameTaskActionShuffleDataset,
    SquaredErrorAccumulator,
    build_offline_gate,
    build_parser,
    correction_cache_manifest,
    cyclic_action_shuffle,
    episode_block_bootstrap_ci,
    episode_aware_action_shuffle,
    evaluate_batches,
    gate_exit_code,
    resolve_sequence_settings,
    resolve_grounding_mode,
    same_task_action_donor_indices,
    select_checkpoint_split,
    validate_anchor_frozen_target_hash_contract,
    validate_checkpoint_correction_caches,
    validate_checkpoint_oracle_target_caches,
    validate_checkpoint_provenance,
)


def test_squared_error_accumulator_weights_by_element_count() -> None:
    metric = SquaredErrorAccumulator()
    metric.update(torch.tensor([0.0, 2.0]), torch.tensor([0.0, 0.0]))
    metric.update(torch.tensor([3.0]), torch.tensor([1.0]))

    assert metric.element_count == 3
    assert metric.squared_error == pytest.approx(8.0)
    assert metric.mse == pytest.approx(8.0 / 3.0)


def test_cyclic_action_shuffle_is_cross_sequence_derangement() -> None:
    actions = torch.arange(3 * 2 * 2).reshape(3, 2, 2)
    shuffled = cyclic_action_shuffle(actions)

    assert torch.equal(shuffled[0], actions[2])
    assert torch.equal(shuffled[1], actions[0])
    assert torch.equal(shuffled[2], actions[1])
    assert not any(torch.equal(shuffled[index], actions[index]) for index in range(3))
    with pytest.raises(ValueError, match="at least two"):
        cyclic_action_shuffle(actions[:1])


def test_episode_aware_action_shuffle_is_same_source_task_and_cross_episode() -> None:
    actions = torch.arange(8, dtype=torch.float32).view(8, 1, 1)
    batch = {
        "source_index": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        "task_name": ["pick-v3", "pick-v3", "pick-v3", "pick-v3", "pick-v3", "pick-v3", "pick-v3", "pick-v3"],
        "episode_id": torch.tensor([1, 1, 2, 2, 10, 10, 11, 11]),
    }
    shuffled = episode_aware_action_shuffle(actions, batch)

    donor_indices = shuffled[:, 0, 0].to(dtype=torch.long)
    for target, donor in enumerate(donor_indices.tolist()):
        assert int(batch["source_index"][target]) == int(batch["source_index"][donor])
        assert batch["task_name"][target] == batch["task_name"][donor]
        assert int(batch["episode_id"][target]) != int(batch["episode_id"][donor])


def test_same_task_action_donor_map_has_total_coverage_with_uneven_episodes() -> None:
    sources = [0, 0, 0, 0, 0, 0, 0]
    tasks = ["pick-v3", "pick-v3", "pick-v3", "pick-v3", "reach-v3", "reach-v3", "reach-v3"]
    episodes = [1, 1, 1, 2, 5, 6, 6]

    donors = same_task_action_donor_indices(sources, tasks, episodes)

    assert len(donors) == len(sources)
    for target, donor in enumerate(donors):
        assert sources[target] == sources[donor]
        assert tasks[target] == tasks[donor]
        assert episodes[target] != episodes[donor]

    with pytest.raises(ValueError, match="at least two episodes"):
        same_task_action_donor_indices([0, 0], ["pick-v3", "pick-v3"], [1, 1])


class _MinimalSequenceDataset(LeWMSequenceDataset):
    def __init__(self) -> None:
        self.source_index = 3
        self.num_state_frames = 7
        self.frame_skip = 1
        self.action = np.concatenate(
            [
                np.ones((7, 1), dtype=np.float32),
                np.full((7, 1), 2.0, dtype=np.float32),
            ]
        )
        self.task_name = np.asarray(["pick-v3"] * 14)
        self.episode_id = np.asarray([10] * 7 + [11] * 7)
        self.episodes = [np.arange(7), np.arange(7, 14)]
        self.windows = [(0, 0), (1, 0)]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, item: int) -> dict[str, object]:
        episode_index, _ = self.windows[item]
        raw_index = int(self.episodes[episode_index][0])
        actions = torch.from_numpy(self.action[self.episodes[episode_index][:-1]])
        return {
            "action": actions,
            "actions": actions,
            "source_index": torch.tensor(self.source_index),
            "task_name": str(self.task_name[raw_index]),
            "episode_id": int(self.episode_id[raw_index]),
            "q_true": torch.full((7, 1), 999.0),
        }


def test_paired_dataset_attaches_donor_action_without_q_true() -> None:
    paired = SameTaskActionShuffleDataset([_MinimalSequenceDataset()])

    first = paired[0]
    second = paired[1]

    torch.testing.assert_close(first["shuffled_action"], torch.full((6, 1), 2.0))
    torch.testing.assert_close(second["shuffled_action"], torch.ones((6, 1)))
    assert first["shuffle_donor_source_index"] == 3
    assert first["shuffle_donor_task_name"] == "pick-v3"
    assert first["shuffle_donor_episode_id"] == 11
    assert "q_true" not in first


class _ToyWorldModel(torch.nn.Module):
    history_size = 2

    def __init__(self) -> None:
        super().__init__()
        self.rollout_calls = []

    def encode_frames(self, batch):
        assert "q_true" not in batch
        embedding = batch["embedding"].float()
        return embedding, embedding

    def predict(self, embedding, action):
        return embedding + action

    def rollout_embeddings(self, history_embedding, history_action, future_action):
        self.rollout_calls.append(
            (history_action.detach().clone(), future_action.detach().clone())
        )
        samples = future_action.shape[1]
        current = history_embedding[:, -1:, :].unsqueeze(1).expand(-1, samples, -1, -1)
        outputs = []
        for step in range(future_action.shape[2]):
            current = current + future_action[:, :, step : step + 1]
            outputs.append(current)
        return torch.cat(outputs, dim=2)


class _OffsetOracleWorldModel(_ToyWorldModel):
    def encode_frames(self, batch):
        assert "q_true" not in batch
        embedding = batch["embedding"].float()
        return embedding, embedding

    def target_sequence(self, batch, embedding):
        return embedding + batch["oracle_offset"].float()


def test_evaluate_batches_reports_copy_shuffle_and_recursive_metrics() -> None:
    # Each latent transition exactly equals the paired scalar action. Swapping
    # the two sequences' actions therefore preserves shape but worsens error.
    batch = {
        "embedding": torch.tensor(
            [
                [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]],
                [[0.0], [2.0], [4.0], [6.0], [8.0], [10.0], [12.0]],
            ]
        ),
        "action": torch.tensor(
            [
                [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]],
                [[2.0], [2.0], [2.0], [2.0], [2.0], [2.0]],
            ]
        ),
        "q_true": torch.full((2, 7, 1), 999.0),
        "source_index": torch.zeros(2, dtype=torch.long),
        "task_name": ["reach-v3", "reach-v3"],
        "episode_id": torch.tensor([1, 2]),
    }
    model = _ToyWorldModel()
    metrics = evaluate_batches(
        model,
        [batch],
        device="cpu",
        rollout_steps=5,
        bootstrap_samples=100,
    )

    assert metrics["teacher_forced_pred_mse"] == pytest.approx(0.0)
    assert metrics["identity_copy_mse"] == pytest.approx(2.5)
    # Only the action after the observed history is intervened on. The first
    # teacher-forced transition remains factual.
    assert metrics["action_shuffle_mse"] == pytest.approx(0.5)
    assert metrics["action_shuffle_delta_mse"] == pytest.approx(0.5)
    assert metrics["action_sensitivity_mse"] == pytest.approx(0.5)
    assert metrics["recursive_rollout_mse"] == pytest.approx(0.0)
    assert metrics["num_action_shuffle_sequences"] == 2
    assert metrics["action_shuffle_coverage"] == pytest.approx(1.0)
    assert math.isclose(metrics["prediction_vs_copy_ratio"], 0.0)
    endpoints = metrics["macro"]["endpoints"]
    assert endpoints["1"]["predicted_mse"] == pytest.approx(0.0)
    assert endpoints["1"]["hold_last_identity_mse"] == pytest.approx(2.5)
    assert endpoints["3"]["hold_last_identity_mse"] == pytest.approx(22.5)
    assert endpoints["5"]["hold_last_identity_mse"] == pytest.approx(62.5)
    assert metrics["per_task"]["reach-v3"]["num_episodes"] == 2
    assert metrics["macro"]["action_shuffle"]["paired_sequences"] == 2
    assert metrics["gate"]["passed"] is False
    episode_blocks = {
        block["episode_id"]: block for block in metrics["episode_blocks"]
    }
    assert set(episode_blocks) == {1, 2}
    assert episode_blocks[1]["teacher_forced"]["predicted_mse"] == pytest.approx(
        0.0
    )
    assert episode_blocks[1]["teacher_forced"]["identity_copy_mse"] == pytest.approx(
        1.0
    )
    assert episode_blocks[2]["teacher_forced"]["identity_copy_mse"] == pytest.approx(
        4.0
    )
    for block in episode_blocks.values():
        assert block["num_sequences"] == 1
        assert block["num_action_shuffle_sequences"] == 1
        assert block["teacher_forced"]["element_count"] == 2
        assert block["teacher_forced"]["action_shuffle_mse"] == pytest.approx(0.5)
        assert block["teacher_forced"]["action_shuffle_delta_mse"] == pytest.approx(
            0.5
        )
        assert block["teacher_forced"]["action_sensitivity_mse"] == pytest.approx(
            0.5
        )
        assert block["oracle_alignment"] == {
            "element_count": 7,
            "mse": pytest.approx(0.0),
        }
        assert block["recursive_h1_action_shuffle"] == {
            "element_count": 1,
            "mse": pytest.approx(1.0),
            "delta_mse": pytest.approx(1.0),
        }
    assert len(model.rollout_calls) == 2
    original_history, _ = model.rollout_calls[0]
    shuffled_history, shuffled_future = model.rollout_calls[1]
    torch.testing.assert_close(shuffled_history, original_history)
    torch.testing.assert_close(
        shuffled_future[:, 0, 0],
        torch.tensor([[2.0], [1.0]]),
    )


def test_episode_blocks_accumulate_sse_and_expose_auditable_denominators() -> None:
    # Two windows per episode carry different constant oracle offsets.  The
    # expected episode MSE is therefore the mean squared offset over both
    # windows, not an unweighted average of already-rounded batch metrics.
    offsets = torch.tensor([1.0, 3.0, 2.0, 4.0]).view(4, 1, 1)
    batch = {
        "embedding": torch.zeros((4, 7, 1)),
        "oracle_offset": offsets.expand(-1, 7, -1),
        "action": torch.zeros((4, 6, 1)),
        "q_true": torch.full((4, 7, 1), 999.0),
        "source_index": torch.zeros(4, dtype=torch.long),
        "task_name": ["reach-v3"] * 4,
        "episode_id": torch.tensor([1, 1, 2, 2]),
    }

    metrics = evaluate_batches(
        _OffsetOracleWorldModel(),
        [batch],
        device="cpu",
        rollout_steps=5,
        bootstrap_samples=100,
    )
    blocks = {block["episode_id"]: block for block in metrics["episode_blocks"]}

    assert blocks[1]["num_sequences"] == 2
    assert blocks[1]["teacher_forced"]["element_count"] == 4
    assert blocks[1]["oracle_alignment"] == {
        "element_count": 14,
        "mse": pytest.approx(5.0),
    }
    assert blocks[2]["teacher_forced"]["element_count"] == 4
    assert blocks[2]["oracle_alignment"] == {
        "element_count": 14,
        "mse": pytest.approx(10.0),
    }

    alignment_sse = sum(
        block["oracle_alignment"]["mse"]
        * block["oracle_alignment"]["element_count"]
        for block in blocks.values()
    )
    alignment_count = sum(
        block["oracle_alignment"]["element_count"] for block in blocks.values()
    )
    assert alignment_sse / alignment_count == pytest.approx(
        metrics["oracle_alignment_mse"]
    )

    teacher_sse = sum(
        block["teacher_forced"]["predicted_mse"]
        * block["teacher_forced"]["element_count"]
        for block in blocks.values()
    )
    teacher_count = sum(
        block["teacher_forced"]["element_count"] for block in blocks.values()
    )
    assert teacher_sse / teacher_count == pytest.approx(
        metrics["teacher_forced_pred_mse"]
    )


class _Float64WorldModel(_ToyWorldModel):
    history_size = 1

    def encode_frames(self, batch):
        embedding = batch["embedding"].to(torch.float64)
        return embedding, embedding

    def predict(self, embedding, action):
        return embedding + action.to(torch.float64)


def test_teacher_mse_uses_float64_episode_block_reduction_exactly() -> None:
    errors = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ],
        dtype=torch.float64,
    )
    embedding = torch.zeros((3, 6, 3), dtype=torch.float64)
    embedding[:, 0] = errors
    batch = {
        "embedding": embedding,
        "action": torch.zeros((3, 5, 3)),
        "source_index": torch.zeros(3, dtype=torch.long),
        "task_name": ["reach-v3"] * 3,
        "episode_id": torch.tensor([1, 2, 3]),
    }

    metrics = evaluate_batches(
        _Float64WorldModel(),
        [batch],
        device="cpu",
        rollout_steps=5,
        bootstrap_samples=100,
    )
    blocks = metrics["episode_blocks"]
    teacher_count = sum(
        block["teacher_forced"]["element_count"] for block in blocks
    )
    recomputed = sum(
        block["teacher_forced"]["predicted_mse"]
        * block["teacher_forced"]["element_count"]
        for block in blocks
    ) / teacher_count

    assert metrics["teacher_forced_pred_mse"] == recomputed
    assert metrics["teacher_forced_pred_mse"] == pytest.approx(
        float(errors.square().sum()) / errors.numel(),
        rel=0.0,
        abs=1e-15,
    )
    float32_reduction = float(errors.float().square().sum()) / errors.numel()
    assert not math.isclose(
        metrics["teacher_forced_pred_mse"],
        float32_reduction,
        rel_tol=0.0,
        abs_tol=1e-10,
    )
    for block in blocks:
        assert block["teacher_forced"]["element_count"] == 3
        assert block["oracle_alignment"]["element_count"] == 18
        assert block["recursive_h1_action_shuffle"]["element_count"] == 3


def test_episode_block_bootstrap_uses_equal_task_macro() -> None:
    ci = episode_block_bootstrap_ci(
        {
            "one-episode-task": [(1.0, 2.0)],
            "three-episode-task": [(3.0, 2.0), (6.0, 4.0), (9.0, 6.0)],
        },
        samples=500,
        seed=17,
    )

    assert ci == pytest.approx([1.0, 1.0])


def test_gate_conditions_reject_nonfinite_and_control_exit_code() -> None:
    per_task = {
        "pick-v3": {
            "endpoints": {
                "1": {"ratio": 0.80},
                "3": {"ratio": 0.90},
                "5": {"ratio": 0.92},
            },
            "action_shuffle": {"ratio": 1.20, "coverage": 1.0},
        },
        "reach-v3": {
            "endpoints": {
                "1": {"ratio": 0.82},
                "3": {"ratio": 0.91},
                "5": {"ratio": 0.93},
            },
            "action_shuffle": {"ratio": 1.25, "coverage": 1.0},
        },
    }
    macro = {
        "endpoints": {
            "1": {
                "predicted_mse": 0.8,
                "hold_last_identity_mse": 1.0,
                "ratio": 0.81,
                "bootstrap_95_ci": [0.75, 0.88],
            },
            "3": {
                "predicted_mse": 0.9,
                "hold_last_identity_mse": 1.0,
                "ratio": 0.905,
                "bootstrap_95_ci": [0.86, 0.96],
            },
            "5": {
                "predicted_mse": 0.92,
                "hold_last_identity_mse": 1.0,
                "ratio": 0.925,
                "bootstrap_95_ci": [0.88, 0.98],
            },
        },
        "action_shuffle": {
            "ratio": 1.225,
            "coverage": 1.0,
            "bootstrap_95_ci": [1.05, 1.35],
        },
    }

    gate = build_offline_gate(per_task, macro)

    assert gate["passed"] is True
    assert gate["decision"] == "Go"
    assert gate_exit_code(gate, enforce_gate=True) == 0
    assert gate_exit_code({"passed": False}, enforce_gate=False) == 0
    assert gate_exit_code({"passed": False}, enforce_gate=True) == 1

    macro["endpoints"]["1"]["ratio"] = None
    failed = build_offline_gate(per_task, macro)
    assert failed["passed"] is False
    assert failed["conditions"]["all_required_values_finite"]["passed"] is False
    assert failed["decision"] == "No-Go"


def test_main_enforce_gate_controls_process_exit(
    tmp_path, monkeypatch, capsys
) -> None:
    class _DummyDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return {"index": torch.tensor(index)}

    dummy = _DummyDataset()
    metrics = {
        "action_shuffle_coverage": 1.0,
        "per_task": {},
        "macro": {},
        "bootstrap": {"samples": 10, "seed": 0},
        "gate": {"passed": False, "decision": "No-Go"},
    }
    monkeypatch.setattr(
        evaluator,
        "_load_checkpoint_metadata",
        lambda path: {
                "model_config": {
                    "history_size": 2,
                    "world_model": {"target_mode": "online"},
                }
            },
    )
    monkeypatch.setattr(evaluator, "_build_datasets", lambda *args, **kwargs: [dummy])
    monkeypatch.setattr(
        evaluator, "SameTaskActionShuffleDataset", lambda datasets: dummy
    )
    monkeypatch.setattr(evaluator, "evaluate_batches", lambda *args, **kwargs: metrics)
    monkeypatch.setattr(
        "fcpg.lewm_runtime.load_lewm_grounded_checkpoint",
        lambda *args, **kwargs: (object(), {}, {}),
    )
    checkpoint_path = tmp_path / "model.pt"
    data_path = tmp_path / "pick.npz"
    checkpoint_path.write_bytes(b"checkpoint")
    data_path.write_bytes(b"dataset")
    argv = [
        "--checkpoint",
        str(checkpoint_path),
        "--data",
        str(data_path),
        "--grounding-mode",
        "nominal",
        "--allow-unfrozen-checkpoint",
        "--num-workers",
        "0",
        "--bootstrap-samples",
        "10",
        "--device",
        "cpu",
    ]

    assert evaluator.main(argv) == 0
    assert evaluator.main([*argv, "--enforce-gate"]) == 1
    assert '"decision": "No-Go"' in capsys.readouterr().out


def test_cli_and_sequence_setting_helpers() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--expected-checkpoint-sha256",
            "a" * 64,
            "--data",
            "pick.npz",
            "--bootstrap-samples",
            "123",
            "--bootstrap-seed",
            "9",
            "--enforce-gate",
        ]
    )
    assert args.expected_checkpoint_sha256 == "a" * 64
    assert args.rollout_steps == 5
    assert args.batch_size == 64
    assert args.split == "validation"
    assert args.grounding_mode == "checkpoint"
    assert args.bootstrap_samples == 123
    assert args.bootstrap_seed == 9
    assert args.enforce_gate is True
    assert args.r1_max == pytest.approx(0.90)
    assert args.r3_max == pytest.approx(0.95)
    assert args.r5_max == pytest.approx(0.95)
    assert args.s1_min == pytest.approx(1.10)
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--checkpoint", "model.pt", "--data", "pick.npz", "--rollout-steps", "4"]
        )

    checkpoint = {
        "model_config": {"history_size": 3},
        "data_config": {"frame_skip": 5},
    }
    assert resolve_sequence_settings(checkpoint, rollout_steps=4) == (3, 4, 5)
    assert resolve_sequence_settings(
        checkpoint, rollout_steps=2, history_size=2, frame_skip=3
    ) == (2, 2, 3)


def test_checkpoint_split_and_grounding_contract(tmp_path) -> None:
    path = tmp_path / "source.npz"
    cache = tmp_path / "cache"
    cache.mkdir()
    correction_path = cache / "q_corrected.npy"
    correction_path.write_bytes(b"corrected")
    dataset = SimpleNamespace(
        episodes=[torch.tensor([0, 1]), torch.tensor([2, 3])],
        windows=[(0, 0), (0, 1), (1, 0)],
        task_name=["reach-v3"] * 4,
        episode_id=[10, 10, 11, 11],
    )
    checkpoint = {
        "data_config": {
            "grounding_mode": "fcpg",
            "sources": [
                {
                    "path": str(path.resolve()),
                    "correction_cache": str(correction_path.resolve()),
                    "train_episodes": ["reach-v3:10"],
                    "val_episodes": ["reach-v3:11"],
                }
            ],
        }
    }

    selected = select_checkpoint_split(dataset, checkpoint, path, "validation")
    assert list(selected.indices) == [2]
    assert resolve_grounding_mode(checkpoint, "checkpoint") == "fcpg"
    assert resolve_grounding_mode(checkpoint, "nominal") == "nominal"
    validate_checkpoint_correction_caches(
        checkpoint, [str(path)], [str(cache)]
    )
    with pytest.raises(ValueError, match="mismatch"):
        validate_checkpoint_correction_caches(
            checkpoint, [str(path)], [str(tmp_path / "other.npy")]
        )


def test_checkpoint_oracle_target_cache_contract_is_exact(tmp_path) -> None:
    data_path = tmp_path / "source.npz"
    data_path.write_bytes(b"dataset")
    cache = tmp_path / "oracle"
    cache.mkdir()
    array_path = cache / "oracle_target.npy"
    metadata_path = cache / "metadata.json"
    array_path.write_bytes(b"oracle-array")
    encoder_sha = "e" * 64
    metadata_path.write_text(
        json.dumps({"encoder_checkpoint_sha256": encoder_sha}) + "\n",
        encoding="utf-8",
    )
    checkpoint = {
        "data_config": {
            "sources": [
                {
                    "path": str(data_path.resolve()),
                    "oracle_target_cache": str(cache.resolve()),
                    "oracle_target_array": str(array_path.resolve()),
                    "oracle_target_sha256": evaluator._file_sha256(array_path),
                    "oracle_target_metadata_sha256": evaluator._file_sha256(
                        metadata_path
                    ),
                    "oracle_target_encoder_sha256": encoder_sha,
                }
            ]
        }
    }

    validate_checkpoint_oracle_target_caches(
        checkpoint, [str(data_path)], [str(cache)]
    )

    array_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="array_sha256 mismatch"):
        validate_checkpoint_oracle_target_caches(
            checkpoint, [str(data_path)], [str(cache)]
        )


@pytest.mark.parametrize("conflicting_source_index", [0, 1])
def test_checkpoint_oracle_target_cache_contract_rejects_each_source_encoder(
    tmp_path,
    conflicting_source_index: int,
) -> None:
    encoder_sha256 = "e" * 64
    data_paths = []
    cache_paths = []
    manifests = []
    sources = []
    for source_index in range(2):
        data_path = tmp_path / f"source-{source_index}.npz"
        data_path.write_bytes(f"dataset-{source_index}".encode("ascii"))
        cache = tmp_path / f"oracle-{source_index}"
        cache.mkdir()
        array_path = cache / "oracle_target.npy"
        metadata_path = cache / "metadata.json"
        array_path.write_bytes(f"oracle-{source_index}".encode("ascii"))
        metadata_path.write_text(
            json.dumps({"encoder_checkpoint_sha256": encoder_sha256}) + "\n",
            encoding="utf-8",
        )
        data_paths.append(data_path)
        cache_paths.append(cache)
        manifests.append(evaluator.oracle_target_cache_manifest(cache))
        sources.append(
            {
                "path": str(data_path.resolve()),
                "oracle_target_cache": str(cache.resolve()),
                "oracle_target_array": str(array_path.resolve()),
                "oracle_target_sha256": evaluator._file_sha256(array_path),
                "oracle_target_metadata_sha256": evaluator._file_sha256(
                    metadata_path
                ),
                "oracle_target_encoder_sha256": (
                    "f" * 64
                    if source_index == conflicting_source_index
                    else encoder_sha256
                ),
            }
        )
    checkpoint = {"data_config": {"sources": sources}}
    order = [1, 0]

    with pytest.raises(ValueError, match="encoder_checkpoint_sha256 mismatch"):
        validate_checkpoint_oracle_target_caches(
            checkpoint,
            [str(data_paths[index]) for index in order],
            [str(cache_paths[index]) for index in order],
            manifests=[manifests[index] for index in order],
        )


def test_correction_cache_manifest_hashes_array_and_metadata(tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    array_path = cache / "q_corrected.npy"
    metadata_path = cache / "metadata.json"
    array_path.write_bytes(b"corrected-array")
    metadata_path.write_text('{"schema_version": 2}\n', encoding="utf-8")

    manifest = correction_cache_manifest(cache)

    assert manifest["input"] == str(cache.resolve())
    assert manifest["array"] == str(array_path.resolve())
    assert manifest["metadata"] == str(metadata_path.resolve())
    assert len(str(manifest["array_sha256"])) == 64
    assert len(str(manifest["metadata_sha256"])) == 64


class _MainDummyDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"index": torch.tensor(index)}


def _main_metrics() -> dict[str, object]:
    return {
        "action_shuffle_coverage": 1.0,
        "per_task": {},
        "macro": {},
        "bootstrap": {"samples": 10, "seed": 0},
        "gate": {"passed": False, "decision": "No-Go"},
    }


def _stub_main_dependencies(monkeypatch, metadata, evaluate) -> None:
    dummy = _MainDummyDataset()
    monkeypatch.setattr(
        evaluator, "_load_checkpoint_metadata", lambda path: metadata
    )
    monkeypatch.setattr(
        evaluator, "_build_datasets", lambda *args, **kwargs: [dummy]
    )
    monkeypatch.setattr(
        evaluator, "SameTaskActionShuffleDataset", lambda datasets: dummy
    )
    monkeypatch.setattr(evaluator, "evaluate_batches", evaluate)
    monkeypatch.setattr(
        "fcpg.lewm_runtime.load_lewm_grounded_checkpoint",
        lambda *args, **kwargs: (object(), {}, metadata),
    )


def test_evaluation_hashes_each_large_input_once_per_snapshot(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    data_paths = []
    for index in range(4):
        data_path = tmp_path / f"source-{index}.npz"
        data_path.write_bytes(f"dataset-{index}".encode("ascii"))
        data_paths.append(data_path)
    checkpoint_sha256 = evaluator._file_sha256(checkpoint_path)
    metadata = {
        "model_config": {
            "history_size": 2,
            "world_model": {"target_mode": "online"},
        }
    }
    _stub_main_dependencies(
        monkeypatch,
        metadata,
        lambda *args, **kwargs: _main_metrics(),
    )

    calls: Counter[str] = Counter()
    real_file_sha256 = evaluator._file_sha256

    def counting_file_sha256(path) -> str:
        calls[str(Path(path).expanduser().resolve())] += 1
        return real_file_sha256(path)

    monkeypatch.setattr(evaluator, "_file_sha256", counting_file_sha256)
    argv = [
        "--checkpoint",
        str(checkpoint_path),
        "--expected-checkpoint-sha256",
        checkpoint_sha256,
        "--grounding-mode",
        "nominal",
        "--allow-unfrozen-checkpoint",
        "--bootstrap-samples",
        "10",
        "--device",
        "cpu",
    ]
    for data_path in data_paths:
        argv.extend(["--data", str(data_path)])

    assert evaluator.main(argv) == 0
    assert calls[str(checkpoint_path.resolve())] == 2
    assert {calls[str(path.resolve())] for path in data_paths} == {2}


@pytest.mark.parametrize(
    ("changed_input", "message"),
    [
        ("checkpoint", "checkpoint changed"),
        ("dataset", "evaluation dataset changed"),
        ("execution", "evaluation code changed"),
    ],
)
def test_evaluation_rejects_runtime_input_changes(
    tmp_path,
    monkeypatch,
    changed_input: str,
    message: str,
) -> None:
    checkpoint_path = tmp_path / "model.pt"
    data_path = tmp_path / "source.npz"
    checkpoint_path.write_bytes(b"checkpoint-before")
    data_path.write_bytes(b"dataset-before")
    checkpoint_sha256 = evaluator._file_sha256(checkpoint_path)
    metadata = {
        "model_config": {
            "history_size": 2,
            "world_model": {"target_mode": "online"},
        }
    }

    def evaluate(*args, **kwargs):
        if changed_input == "checkpoint":
            checkpoint_path.write_bytes(b"checkpoint-after")
        elif changed_input == "dataset":
            data_path.write_bytes(b"dataset-after")
        return _main_metrics()

    _stub_main_dependencies(monkeypatch, metadata, evaluate)
    calls: Counter[str] = Counter()
    real_file_sha256 = evaluator._file_sha256

    def counting_file_sha256(path) -> str:
        calls[str(Path(path).expanduser().resolve())] += 1
        return real_file_sha256(path)

    monkeypatch.setattr(evaluator, "_file_sha256", counting_file_sha256)
    if changed_input == "execution":
        code_manifests = iter(
            [
                {"aggregate_sha256": "a" * 64},
                {"aggregate_sha256": "b" * 64},
            ]
        )
        monkeypatch.setattr(
            evaluator,
            "execution_code_manifest",
            lambda **kwargs: next(code_manifests),
        )

    with pytest.raises(RuntimeError, match=message):
        evaluator.main(
            [
                "--checkpoint",
                str(checkpoint_path),
                "--expected-checkpoint-sha256",
                checkpoint_sha256,
                "--data",
                str(data_path),
                "--grounding-mode",
                "nominal",
                "--allow-unfrozen-checkpoint",
                "--device",
                "cpu",
            ]
        )

    assert calls[str(checkpoint_path.resolve())] == 2
    assert calls[str(data_path.resolve())] == 2


def test_explicit_fcpg_validates_and_rechecks_correction_cache(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint_path = tmp_path / "model.pt"
    data_path = tmp_path / "source.npz"
    cache = tmp_path / "correction"
    cache.mkdir()
    correction_array = cache / "q_corrected.npy"
    correction_array.write_bytes(b"correction-before")
    correction_metadata = cache / "metadata.json"
    correction_metadata.write_text("{}\n", encoding="utf-8")
    checkpoint_path.write_bytes(b"checkpoint")
    data_path.write_bytes(b"dataset")
    checkpoint_sha256 = evaluator._file_sha256(checkpoint_path)
    metadata = {
        "model_config": {
            "history_size": 2,
            "world_model": {"target_mode": "online"},
        },
        "data_config": {
            "sources": [
                {
                    "path": str(data_path.resolve()),
                    "correction_cache": str(correction_array.resolve()),
                }
            ]
        },
    }
    validation_calls = []
    original_validate = evaluator.validate_checkpoint_correction_caches

    def validate(*args, **kwargs):
        validation_calls.append((args, kwargs))
        return original_validate(*args, **kwargs)

    def evaluate(*args, **kwargs):
        correction_array.write_bytes(b"correction-after")
        return _main_metrics()

    _stub_main_dependencies(monkeypatch, metadata, evaluate)
    monkeypatch.setattr(
        evaluator, "validate_checkpoint_correction_caches", validate
    )
    calls: Counter[str] = Counter()
    real_file_sha256 = evaluator._file_sha256

    def counting_file_sha256(path) -> str:
        calls[str(Path(path).expanduser().resolve())] += 1
        return real_file_sha256(path)

    monkeypatch.setattr(evaluator, "_file_sha256", counting_file_sha256)

    with pytest.raises(RuntimeError, match="correction cache changed"):
        evaluator.main(
            [
                "--checkpoint",
                str(checkpoint_path),
                "--expected-checkpoint-sha256",
                checkpoint_sha256,
                "--data",
                str(data_path),
                "--correction-cache",
                str(cache),
                "--grounding-mode",
                "fcpg",
                "--allow-unfrozen-checkpoint",
                "--device",
                "cpu",
            ]
        )

    assert len(validation_calls) == 2
    for path in (
        checkpoint_path,
        data_path,
        correction_array,
        correction_metadata,
    ):
        assert calls[str(path.resolve())] == 2


def test_evaluation_rechecks_oracle_cache_after_runtime_change(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint_path = tmp_path / "model.pt"
    data_path = tmp_path / "source.npz"
    cache = tmp_path / "oracle"
    cache.mkdir()
    oracle_array = cache / "oracle_target.npy"
    oracle_metadata = cache / "metadata.json"
    oracle_array.write_bytes(b"oracle-before")
    encoder_sha256 = "e" * 64
    oracle_metadata.write_text(
        json.dumps({"encoder_checkpoint_sha256": encoder_sha256}) + "\n",
        encoding="utf-8",
    )
    checkpoint_path.write_bytes(b"checkpoint")
    data_path.write_bytes(b"dataset")
    checkpoint_sha256 = evaluator._file_sha256(checkpoint_path)
    metadata = {
        "model_config": {
            "history_size": 2,
            "world_model": {"target_mode": "frozen_oracle"},
        },
        "data_config": {
            "target_contract": {
                "mode": "frozen_oracle",
                "encoder_checkpoint_sha256": encoder_sha256,
            },
            "sources": [
                {
                    "path": str(data_path.resolve()),
                    "oracle_target_cache": str(cache.resolve()),
                    "oracle_target_array": str(oracle_array.resolve()),
                    "oracle_target_sha256": evaluator._file_sha256(oracle_array),
                    "oracle_target_metadata_sha256": evaluator._file_sha256(
                        oracle_metadata
                    ),
                    "oracle_target_encoder_sha256": encoder_sha256,
                }
            ],
        },
    }
    validation_calls = []
    original_validate = evaluator.validate_checkpoint_oracle_target_caches

    def validate(*args, **kwargs):
        validation_calls.append((args, kwargs))
        return original_validate(*args, **kwargs)

    def evaluate(*args, **kwargs):
        oracle_array.write_bytes(b"oracle-after")
        return _main_metrics()

    _stub_main_dependencies(monkeypatch, metadata, evaluate)
    monkeypatch.setattr(
        evaluator, "validate_checkpoint_oracle_target_caches", validate
    )
    calls: Counter[str] = Counter()
    real_file_sha256 = evaluator._file_sha256

    def counting_file_sha256(path) -> str:
        calls[str(Path(path).expanduser().resolve())] += 1
        return real_file_sha256(path)

    monkeypatch.setattr(evaluator, "_file_sha256", counting_file_sha256)

    with pytest.raises(ValueError, match="array_sha256 mismatch"):
        evaluator.main(
            [
                "--checkpoint",
                str(checkpoint_path),
                "--expected-checkpoint-sha256",
                checkpoint_sha256,
                "--data",
                str(data_path),
                "--oracle-target-cache",
                str(cache),
                "--grounding-mode",
                "nominal",
                "--allow-unfrozen-checkpoint",
                "--device",
                "cpu",
            ]
        )

    assert len(validation_calls) == 2
    for path in (
        checkpoint_path,
        data_path,
        oracle_array,
        oracle_metadata,
    ):
        assert calls[str(path.resolve())] == 2


def test_evaluation_revalidates_frozen_manifest_after_runtime_change(
    tmp_path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint_path = run_dir / "model.pt"
    data_path = tmp_path / "source.npz"
    external_manifest = tmp_path / "frozen_manifest.json"
    sidecar_manifest = run_dir / "input_manifest.json"
    checkpoint_path.write_bytes(b"checkpoint")
    data_path.write_bytes(b"dataset")
    data_stat = data_path.stat()
    manifest = {
        "schema_version": 1,
        "sources": [
            {
                "dataset": {
                    "absolute_path": str(data_path.resolve()),
                    "size": data_stat.st_size,
                    "sha256": evaluator._file_sha256(data_path),
                }
            }
        ],
        "repository": {},
    }
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    external_manifest.write_text(payload, encoding="utf-8")
    sidecar_manifest.write_text(payload, encoding="utf-8")
    checkpoint_sha256 = evaluator._file_sha256(checkpoint_path)
    metadata = {
        "model_config": {
            "history_size": 2,
            "world_model": {"target_mode": "online"},
        },
        "provenance": {
            "path": str(external_manifest),
            "sha256": evaluator._file_sha256(external_manifest),
            "manifest": manifest,
        },
    }

    def evaluate(*args, **kwargs):
        external_manifest.write_text(payload + " ", encoding="utf-8")
        return _main_metrics()

    _stub_main_dependencies(monkeypatch, metadata, evaluate)
    monkeypatch.setattr(
        evaluator,
        "_validate_manifest_with_file_snapshots",
        lambda *args, **kwargs: None,
    )
    calls: Counter[str] = Counter()
    real_file_sha256 = evaluator._file_sha256

    def counting_file_sha256(path) -> str:
        calls[str(Path(path).expanduser().resolve())] += 1
        return real_file_sha256(path)

    monkeypatch.setattr(evaluator, "_file_sha256", counting_file_sha256)

    with pytest.raises(ValueError, match="external frozen input manifest SHA-256"):
        evaluator.main(
            [
                "--checkpoint",
                str(checkpoint_path),
                "--expected-checkpoint-sha256",
                checkpoint_sha256,
                "--data",
                str(data_path),
                "--grounding-mode",
                "nominal",
                "--device",
                "cpu",
            ]
        )

    for path in (
        checkpoint_path,
        data_path,
        external_manifest,
        sidecar_manifest,
    ):
        assert calls[str(path.resolve())] == 2


def test_expected_checkpoint_hash_fails_before_torch_load(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    torch_load_calls = []

    def unexpected_torch_load(*args, **kwargs):
        torch_load_calls.append((args, kwargs))
        raise AssertionError("torch.load must not run after a failed hash pin")

    monkeypatch.setattr(torch, "load", unexpected_torch_load)

    with pytest.raises(ValueError, match="before torch.load"):
        evaluator.main(
            [
                "--checkpoint",
                str(checkpoint_path),
                "--expected-checkpoint-sha256",
                "0" * 64,
                "--data",
                str(tmp_path / "source.npz"),
                "--device",
                "cpu",
            ]
        )

    assert torch_load_calls == []


def test_anchor_frozen_target_hash_contract_rejects_any_conflict(tmp_path) -> None:
    data_path = tmp_path / "source.npz"
    checkpoint = {
        "model_config": {"world_model": {"target_mode": "frozen_oracle"}},
        "data_config": {
            "target_contract": {
                "mode": "frozen_oracle",
                "encoder_checkpoint_sha256": evaluator.ANCHOR_ABLATION_TARGET_SHA256,
            },
            "sources": [
                {
                    "path": str(data_path),
                    "oracle_target_encoder_sha256": evaluator.ANCHOR_ABLATION_TARGET_SHA256,
                }
            ],
        },
    }
    manifests = [
        {
            "encoder_checkpoint_sha256": evaluator.ANCHOR_ABLATION_TARGET_SHA256
        }
    ]

    assert validate_anchor_frozen_target_hash_contract(
        checkpoint,
        [str(data_path)],
        manifests,
        evaluator.ANCHOR_ABLATION_TARGET_SHA256,
    ) == evaluator.ANCHOR_ABLATION_TARGET_SHA256

    checkpoint["data_config"]["target_contract"][
        "encoder_checkpoint_sha256"
    ] = "b" * 64
    with pytest.raises(ValueError, match="frozen target hash conflict"):
        validate_anchor_frozen_target_hash_contract(
            checkpoint,
            [str(data_path)],
            manifests,
            evaluator.ANCHOR_ABLATION_TARGET_SHA256,
        )

    checkpoint["data_config"]["target_contract"][
        "encoder_checkpoint_sha256"
    ] = evaluator.ANCHOR_ABLATION_TARGET_SHA256
    manifests[0]["encoder_checkpoint_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="metadata.encoder_checkpoint_sha256"):
        validate_anchor_frozen_target_hash_contract(
            checkpoint,
            [str(data_path)],
            manifests,
            evaluator.ANCHOR_ABLATION_TARGET_SHA256,
        )


@pytest.mark.parametrize("conflicting_source_index", [0, 1])
def test_anchor_frozen_target_hash_contract_rejects_each_source_conflict(
    tmp_path,
    conflicting_source_index: int,
) -> None:
    data_paths = [tmp_path / "source-a.npz", tmp_path / "source-b.npz"]
    checkpoint = {
        "model_config": {"world_model": {"target_mode": "frozen_oracle"}},
        "data_config": {
            "target_contract": {
                "mode": "frozen_oracle",
                "encoder_checkpoint_sha256": (
                    evaluator.ANCHOR_ABLATION_TARGET_SHA256
                ),
            },
            "sources": [
                {
                    "path": str(path),
                    "oracle_target_encoder_sha256": (
                        "b" * 64
                        if source_index == conflicting_source_index
                        else evaluator.ANCHOR_ABLATION_TARGET_SHA256
                    ),
                }
                for source_index, path in enumerate(data_paths)
            ],
        },
    }
    requested_paths = [str(data_paths[1]), str(data_paths[0])]
    manifests = [
        {
            "encoder_checkpoint_sha256": (
                evaluator.ANCHOR_ABLATION_TARGET_SHA256
            )
        }
        for _ in requested_paths
    ]

    with pytest.raises(
        ValueError,
        match=(
            rf"data_config\.sources\[{conflicting_source_index}\]\."
            r"oracle_target_encoder_sha256"
        ),
    ):
        validate_anchor_frozen_target_hash_contract(
            checkpoint,
            requested_paths,
            manifests,
            evaluator.ANCHOR_ABLATION_TARGET_SHA256,
        )


def test_checkpoint_provenance_is_required_and_sidecars_are_bound(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint_path = tmp_path / "run" / "lewm_grounded_epoch_001.pt"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="no frozen provenance"):
        validate_checkpoint_provenance(
            {},
            checkpoint_path,
            allow_unfrozen=False,
        )
    unfrozen = validate_checkpoint_provenance(
        {},
        checkpoint_path,
        allow_unfrozen=True,
    )
    assert unfrozen["validated"] is False

    manifest = {"schema_version": 1, "sources": [], "repository": {}}
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    external = tmp_path / "formal_manifest.json"
    external.write_text(payload, encoding="utf-8")
    sidecar = checkpoint_path.parent / "input_manifest.json"
    sidecar.write_text(payload, encoding="utf-8")
    digest = evaluator._file_sha256(external)
    checkpoint = {
        "provenance": {
            "path": str(external),
            "sha256": digest,
            "manifest": manifest,
        }
    }
    calls = []
    monkeypatch.setattr(
        evaluator,
        "validate_manifest",
        lambda value, check_git: calls.append((value, check_git)),
    )

    report = validate_checkpoint_provenance(
        checkpoint,
        checkpoint_path,
        allow_unfrozen=False,
    )

    assert report["validated"] is True
    assert report["manifest_sha256"] == digest
    assert calls == [(manifest, True)]

    sidecar.write_text(payload + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar SHA-256 differs"):
        validate_checkpoint_provenance(
            checkpoint,
            checkpoint_path,
            allow_unfrozen=False,
        )


def test_atomic_report_publication_is_no_replace_and_cleans_temporary(
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"
    calls = []

    evaluator._publish_text_exclusive_atomic(
        output,
        "{\"ok\": true}\n",
        revalidate=lambda: calls.append("validated"),
    )

    assert output.read_text(encoding="utf-8") == "{\"ok\": true}\n"
    assert calls == ["validated"]
    assert list(tmp_path.glob(".report.json.*.tmp")) == []

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evaluator._publish_text_exclusive_atomic(output, "different\n")
    assert output.read_text(encoding="utf-8") == "{\"ok\": true}\n"


@pytest.mark.parametrize(
    "extra_args",
    [
        [],
        ["--expected-checkpoint-sha256", "a" * 64, "--enforce-gate"],
        [
            "--expected-checkpoint-sha256",
            "a" * 64,
            "--allow-unfrozen-checkpoint",
        ],
    ],
)
def test_anchor_override_rejects_untrusted_or_formal_use_before_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_args: list[str],
) -> None:
    monkeypatch.setattr(
        evaluator,
        "_load_checkpoint_metadata",
        lambda path: pytest.fail("checkpoint must not be deserialized"),
    )
    argv = [
        "--checkpoint",
        str(tmp_path / "checkpoint.pt"),
        "--data",
        str(tmp_path / "source.npz"),
        "--anchor-visual-mode-override",
        "paired_visual",
        *extra_args,
    ]

    with pytest.raises(ValueError):
        evaluator.main(argv)
