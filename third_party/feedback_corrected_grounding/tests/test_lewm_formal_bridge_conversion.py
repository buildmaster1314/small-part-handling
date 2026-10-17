from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fcpg.lewm_a3_p1_bridge_collection import validate_bridge_arrays
from fcpg.lewm_formal_bridge_conversion import (
    STACKED_COLLECTION_PROTOCOL,
    physical_episode_digest,
    write_formal_bridge_splits,
)
from fcpg.lewm_formal_policy_bridge import load_formal_bridge_split
from fcpg.lewm_policy_token_bridge import sha256_file


class _FakeAdapter(nn.Module):
    def forward_sampled(
        self,
        visual: torch.Tensor,
        proprio: torch.Tensor,
        *,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        assert bool(torch.equal(reliability, torch.ones_like(reliability)))
        return visual + proprio


class _FakePolicy(nn.Module):
    num_keypoints = 3
    vision_dim = 256
    model_dim = 256

    def __init__(self) -> None:
        super().__init__()
        self.grounding_adapter = _FakeAdapter()

    def _proprio(self, xyz: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(xyz, (0, 253))


def _a3_checkpoint(path: Path, seed: int) -> dict[str, object]:
    torch.save(
        {
            "checkpoint_type": "fcpg_matched_anchor_dynamics_pair_v1",
            "training_protocol": "fcpg_action_branch_physical_support_a3_v1",
            "schema_version": 1,
            "step": 10_000,
            "completed": True,
            "selection": "fixed_latest",
            "seed": seed,
            "predictor_config": {
                "latent_dim": 192,
                "num_keypoints": 3,
                "action_dim": 20,
                "history_size": 3,
            },
            "factual_state": {"weight": torch.tensor([float(seed)])},
            "no_action_state": {"weight": torch.tensor([float(seed)])},
        },
        path,
    )
    return {"seed": seed, "path": str(path), "sha256": sha256_file(path)}


def _stacked(tmp_path: Path) -> tuple[Path, Path]:
    n, s, b, h, k, d = 2, 3, 8, 8, 3, 192
    rng = np.random.default_rng(42)
    factual = rng.normal(size=(s, n, b, h, k, d)).astype(np.float32)
    no_action = rng.normal(size=(s, n, b, h, k, d)).astype(np.float32)
    last_visual = rng.normal(size=(n, k, 256)).astype(np.float32)
    current_xyz = rng.normal(size=(n, b, h, k, 3)).astype(np.float32)
    arrays = {
        "source_row_index": np.asarray([7, 11], dtype=np.int64),
        "task_index": np.zeros(n, dtype=np.int64),
        "task_instance_index": np.asarray([1, 2], dtype=np.int64),
        "source_episode_id": np.asarray(["pick:train:1", "pick:dev:2"]),
        "hazard_id": np.asarray([9, 12], dtype=np.int64),
        "split": np.asarray(["train", "development"]),
        "predictor_seed": np.asarray([29301, 29302, 29303], dtype=np.int64),
        "branch_index": np.arange(b, dtype=np.int64),
        "history_embedding": rng.normal(size=(n, 3, k, d)).astype(np.float32),
        "history_action": rng.normal(size=(n, 2, 20)).astype(np.float32),
        "branch_action": rng.normal(size=(n, b, h, 20)).astype(np.float32),
        "factual_prediction": factual,
        "no_action_prediction": no_action,
        "hidden_clean_a3_target": rng.normal(size=(n, b, h, k, d)).astype(np.float32),
        "last_policy_token": rng.normal(size=(n, k, 256)).astype(np.float32),
        "last_sampled_visual": last_visual,
        "last_xyz": rng.normal(size=(n, k, 3)).astype(np.float32),
        "current_xyz": current_xyz,
        "hidden_clean_policy_token": rng.normal(size=(n, b, h, k, 256)).astype(np.float32),
        "hidden_clean_sampled_visual": rng.normal(size=(n, b, h, k, 256)).astype(np.float32),
        "hidden_clean_q_corrected": rng.normal(size=(n, b, h, k, 2)).astype(np.float32),
        "hidden_clean_reliability": np.ones((n, b, h, k), dtype=np.float32),
        "max_a3_history_reconstruction_error": np.zeros(n, dtype=np.float64),
        "max_a3_target_reconstruction_error": np.zeros(n, dtype=np.float64),
    }
    validate_bridge_arrays(arrays)
    source = tmp_path / "stacked.npz"
    np.savez_compressed(source, **arrays)

    policy = tmp_path / "policy.pt"
    policy.write_bytes(b"policy")
    corrector = tmp_path / "corrector.pt"
    corrector.write_bytes(b"corrector")
    predictors = [
        _a3_checkpoint(tmp_path / f"a3_{seed}.pt", seed)
        for seed in (29301, 29302, 29303)
    ]
    metadata = {
        "schema_version": 1,
        "protocol": STACKED_COLLECTION_PROTOCOL,
        "smoke": False,
        "task": "pick-place-v3",
        "p1_history_alignment": "raw-adjacent-pre-action-[t-1,t]",
        "a3_history_alignment": "decision-synchronous-frame-skip-5",
        "hidden_clean_targets_are_offline_labels_only": True,
        "source_action_byte_equality_checked": True,
        "p1_adapter_token_reconstruction_checked": True,
        "a3_history_and_target_reconstruction_checked": True,
        "source": {"path": "unused", "sha256": "3" * 64},
        "anchor_encoder": {},
        "p1": {
            "policy_path": str(policy),
            "policy_sha256": sha256_file(policy),
            "corrector_path": str(corrector),
            "corrector_sha256": sha256_file(corrector),
            "obs_horizon": 2,
            "history_alignment": "raw-adjacent-pre-action-[t-1,t]",
        },
        "predictors": predictors,
        "selection": {
            "source_rows": [7, 11],
            "branch_indices": list(range(8)),
            "max_instances": None,
            "max_rows": None,
        },
        "checks": {},
        "arrays": {name: list(value.shape) for name, value in arrays.items()},
        "output": str(source),
        "output_sha256": sha256_file(source),
    }
    metadata_path = source.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    return source, metadata_path


def test_physical_episode_digest_is_stable_and_task_qualified() -> None:
    first = physical_episode_digest("pick-place-v3", "episode-4")
    assert first == physical_episode_digest("pick-place-v3", "episode-4")
    assert first != physical_episode_digest("push-v3", "episode-4")
    assert len(first) == 64


def test_stacked_converter_writes_disjoint_formal_splits_and_exact_geometry(
    tmp_path: Path,
) -> None:
    source, metadata = _stacked(tmp_path)
    output = tmp_path / "formal"
    policy = _FakePolicy().requires_grad_(False).eval()
    report = write_formal_bridge_splits(
        stacked_source=source,
        stacked_metadata=metadata,
        output_root=output,
        policy=policy,
        policy_device="cpu",
        collector_implementation=(
            Path(__file__).resolve().parents[1] / "fcpg/lewm_a3_p1_bridge_collection.py"
        ),
        geometry_batch_size=11,
    )
    train = load_formal_bridge_split(output / "train")
    validation = load_formal_bridge_split(output / "validation")
    assert train.sample_count == 64
    assert validation.sample_count == 64
    assert train.factual_prediction.shape == (64, 3, 3, 192)
    assert not set(train.physical_episode_sha256).intersection(
        validation.physical_episode_sha256
    )
    assert report["splits"]["train"]["physical_episode_count"] == 1
    assert (
        train.metadata["provenance"]["collector"]["source_data_sha256"]
        == "3" * 64
    )

    with np.load(source, allow_pickle=False) as stacked:
        held = np.repeat(stacked["last_sampled_visual"][0:1], 64, axis=0)
        xyz = stacked["current_xyz"][0].reshape(64, 3, 3)
    expected = held + np.pad(xyz, ((0, 0), (0, 0), (0, 253)))
    np.testing.assert_allclose(train.geometry_hold_token, expected, rtol=0.0, atol=0.0)
