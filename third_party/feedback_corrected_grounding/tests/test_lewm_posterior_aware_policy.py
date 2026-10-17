from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from fcpg.lewm_posterior_aware_policy import (
    PosteriorAwarePolicyDataset,
    PosteriorTokenAdapterPolicy,
    PosteriorTokenResidualAdapter,
    build_posterior_policy_cache,
    load_posterior_policy_cache,
    merge_posterior_trained_state,
    sample_drop_schedules,
)


class _Grounder(nn.Module):
    def forward_sampled(self, visual, proprio, reliability):
        del proprio
        return visual * reliability.unsqueeze(-1)


class _Policy(nn.Module):
    num_keypoints = 2
    vision_dim = 4
    model_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.grounding_adapter = _Grounder()

    def _proprio(self, xyz):
        return xyz


class _Predictor(nn.Module):
    def __init__(self, action_scale: float) -> None:
        super().__init__()
        self.action_scale = float(action_scale)

    def predict(self, history, action):
        shift = self.action_scale * action.mean(dim=-1)[:, :, None, None]
        return history + shift


class _BaseDataset(Dataset):
    obs_dim = 4
    action_dim = 4
    num_keypoints = 2
    image_size = (8, 8)
    policy_visual_dim = 4

    def __init__(self, sources: list[int]) -> None:
        self.episodes = [np.asarray(sources, dtype=np.int64)]
        self.anchors = [(0, index) for index in range(len(sources))]

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, item: int):
        return {
            "action": torch.zeros(3, 4),
            "action_mask": torch.ones(3, dtype=torch.bool),
            "sampled_visual": torch.zeros(2, 2, 4, dtype=torch.float16),
        }


def test_drop_schedules_are_causal_and_have_recovery() -> None:
    onset, duration = sample_drop_schedules(
        episode_count=5,
        decision_count=20,
        schedules_per_episode=3,
        seed=7,
    )
    assert onset.shape == duration.shape == (5, 3)
    assert bool((onset >= 3).all())
    assert bool((duration >= 2).all())
    assert bool((duration <= 6).all())
    assert bool((onset + duration < 20).all())


def test_build_cache_rolls_factual_and_no_action_separately(tmp_path) -> None:
    episodes = 2
    decisions = 12
    raw_steps = decisions * 5
    count = episodes * raw_steps
    episode_id = np.repeat(np.arange(episodes), raw_steps)
    frame_id = np.tile(np.arange(raw_steps), episodes)
    rng = np.random.default_rng(4)
    sampled = rng.normal(size=(count, 2, 4)).astype(np.float32)
    xyz = rng.normal(size=(count, 2, 3)).astype(np.float32)
    action = rng.normal(size=(count, 4)).astype(np.float32)
    policy = _Policy().eval()
    factual = _Predictor(0.1).eval()
    no_action = _Predictor(0.0).eval()
    cache = build_posterior_policy_cache(
        policy=policy,
        factual_predictor=factual,
        no_action_predictor=no_action,
        source_indices=np.arange(count),
        episode_ids=episode_id,
        frame_ids=frame_id,
        sampled_visual=sampled,
        xyz=xyz,
        actions=action,
        schedules_per_episode=1,
        schedule_seed=9,
        device="cpu",
        inference_batch_size=8,
        amp=False,
    )
    assert len(cache) == episodes * decisions
    assert bool(cache.drop_active.any())
    active = cache.drop_active
    assert not np.array_equal(cache.factual_token[active], cache.no_action_token[active])
    path = tmp_path / "cache.npz"
    cache.save(path)
    loaded = load_posterior_policy_cache(path)
    assert np.array_equal(loaded.anchor_index, cache.anchor_index)
    assert np.array_equal(loaded.factual_token, cache.factual_token)


def test_dataset_only_overrides_active_posterior_rows(tmp_path) -> None:
    count = 3
    token = np.arange(count * 2 * 4, dtype=np.float16).reshape(count, 2, 4)
    path = tmp_path / "cache.npz"
    np.savez_compressed(
        path,
        schema_version=np.asarray([1]),
        anchor_index=np.arange(count),
        episode_id=np.zeros(count, dtype=np.int64),
        schedule_id=np.zeros(count, dtype=np.int64),
        decision_index=np.arange(count),
        drop_active=np.asarray([False, True, False]),
        missing_age=np.asarray([0, 1, 0]),
        factual_token=token,
        no_action_token=token + np.float16(1),
    )
    cache = load_posterior_policy_cache(path)
    factual = PosteriorAwarePolicyDataset(
        _BaseDataset([0, 1, 2]), cache, method="factual"
    )
    assert not bool(factual[0]["grounded_token_override_mask"].any())
    assert factual[1]["grounded_token_override"].shape == (2, 2, 4)
    assert bool(factual[1]["grounded_token_override_mask"].all())
    measurement = PosteriorAwarePolicyDataset(
        _BaseDataset([0, 1, 2]), cache, method="measurement"
    )
    assert "grounded_token_override" not in measurement[1]


def test_merge_trained_state_preserves_sparse_frozen_frontend() -> None:
    base = {
        "vision_encoder.projection.weight": torch.tensor([1.0]),
        "global_visual_projection.weight": torch.tensor([2.0]),
        "controller.weight": torch.tensor([3.0]),
    }
    trained = {
        "vision_encoder.backbone.block.weight": torch.tensor([40.0]),
        "vision_encoder.projection.weight": torch.tensor([10.0]),
        "global_visual_projection.weight": torch.tensor([20.0]),
        "controller.weight": torch.tensor([30.0]),
    }
    merged = merge_posterior_trained_state(base, trained)
    assert set(merged) == set(base)
    assert torch.equal(merged["vision_encoder.projection.weight"], base["vision_encoder.projection.weight"])
    assert torch.equal(merged["global_visual_projection.weight"], base["global_visual_projection.weight"])
    assert torch.equal(merged["controller.weight"], trained["controller.weight"])
    assert "vision_encoder.backbone.block.weight" not in merged


def test_token_adapter_is_exact_identity_at_initialization_and_bounded() -> None:
    torch.manual_seed(13)
    adapter = PosteriorTokenResidualAdapter(token_dim=8, num_keypoints=3, residual_cap=0.2)
    token = torch.randn(2, 2, 3, 8)
    mask = torch.zeros(2, 2, 3, dtype=torch.bool)
    mask[:, 1] = True
    assert torch.equal(adapter(token, mask), token)

    with torch.no_grad():
        adapter.residual[-1].bias.fill_(100.0)
    output = adapter(token, mask)
    delta = output - token
    assert torch.equal(delta[~mask], torch.zeros_like(delta[~mask]))
    assert float(delta[mask].abs().max()) <= 0.200001


class _ActionPolicy(nn.Module):
    def forward(self, batch):
        token = batch.get("grounded_token_override")
        value = torch.zeros(1) if token is None else token.sum().reshape(1)
        return value

    def predict_action_chunk(self, batch):
        return self(batch)


def test_policy_wrapper_bypasses_clean_batches_exactly() -> None:
    base = _ActionPolicy()
    adapter = PosteriorTokenResidualAdapter(token_dim=4, num_keypoints=2)
    wrapped = PosteriorTokenAdapterPolicy(base, adapter)
    clean = {"other": torch.tensor([1.0])}
    assert torch.equal(wrapped(clean), base(clean))

    token = torch.randn(1, 2, 2, 4)
    mask = torch.ones(1, 2, 2, dtype=torch.bool)
    batch = {
        "grounded_token_override": token,
        "grounded_token_override_mask": mask,
    }
    assert torch.equal(wrapped(batch), base(batch))
