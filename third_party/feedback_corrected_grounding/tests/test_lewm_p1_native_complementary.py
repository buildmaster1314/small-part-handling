from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import fcpg.lewm_p1_native_complementary as complementary
from fcpg.lewm_p1_native_complementary import (
    FrozenP1NativeCheckpointSpec,
    hard_measurement_bypass,
    load_frozen_p1_native_checkpoint,
    load_frozen_p1_native_ensemble,
    recursive_complementary_matched_rollout,
    recursive_complementary_rollout,
)
from fcpg.lewm_p1_native_dynamics import (
    P1_NATIVE_PROTOCOL,
    P1NativePredictorConfig,
)
from fcpg.lewm_action_branch_training import rollout_action_branches


class _TinyPredictor(nn.Module):
    def __init__(self, *, condition_on_action: bool) -> None:
        super().__init__()
        self.latent_dim = 256
        self.num_keypoints = 3
        self.action_dim = 20
        self.history_size = 3
        self.condition_on_action = condition_on_action
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("action_mean", torch.zeros(20))
        self.register_buffer("action_std", torch.ones(20))

    def predict(self, embedding: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        result = embedding.clone()
        if self.condition_on_action:
            delta = action[:, -1, :1].reshape(-1, 1, 1)
        else:
            delta = torch.full(
                (embedding.shape[0], 1, 1),
                0.25,
                dtype=embedding.dtype,
                device=embedding.device,
            )
        result[:, -1] = embedding[:, -1] + self.weight * delta
        return result


def _freeze(model: nn.Module) -> nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _config() -> dict[str, Any]:
    predictor = P1NativePredictorConfig(
        predictor_depth=1,
        predictor_heads=1,
        predictor_mlp_dim=16,
        predictor_dim_head=8,
        predictor_dropout=0.0,
        projector_hidden_dim=16,
    )
    return {
        "schema_version": 1,
        "protocol": P1_NATIVE_PROTOCOL,
        "data": {"strict_cache_sha256": "a" * 64},
        "p1": {"policy_checkpoint_sha256": "b" * 64},
        "model": {**predictor.as_kwargs(), "sigreg_weight": 0.0},
        "training": {
            "allowed_seeds": [15801, 15802, 15803],
            "optimizer_steps": 2000,
            "selection": "fixed_latest",
        },
    }


def _write_checkpoint(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    seed: int = 15801,
    mutate: Any = None,
) -> tuple[FrozenP1NativeCheckpointSpec, dict[str, Any]]:
    config = _config()
    factual = _TinyPredictor(condition_on_action=True)
    no_action = _TinyPredictor(condition_on_action=False)
    payload: dict[str, Any] = {
        "checkpoint_type": "fcpg_p1_native_dynamics_pair_v1",
        "protocol": P1_NATIVE_PROTOCOL,
        "selection": "fixed_latest",
        "completed": True,
        "step": 2000,
        "seed": seed,
        "predictor_config": {
            key: value
            for key, value in config["model"].items()
            if key != "sigreg_weight"
        },
        "parameter_count_per_head": 1,
        "action_mean": np.zeros(20, dtype=np.float32),
        "action_std": np.ones(20, dtype=np.float32),
        "factual_state": factual.state_dict(),
        "no_action_state": no_action.state_dict(),
        "config": deepcopy(config),
        "strict_cache_sha256": "a" * 64,
        "p1_policy_checkpoint_sha256": "b" * 64,
        "report_screen": {},
    }
    if mutate is not None:
        mutate(payload)
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    def _build(_config: P1NativePredictorConfig, *, initialization_seed: int):
        assert initialization_seed == seed
        return (
            _TinyPredictor(condition_on_action=True),
            _TinyPredictor(condition_on_action=False),
        )

    monkeypatch.setattr(complementary, "build_matched_p1_native_predictors", _build)
    monkeypatch.setattr(
        complementary, "validate_matched_p1_native_pair", lambda *_args, **_kwargs: None
    )
    return FrozenP1NativeCheckpointSpec(seed=seed, path=path, sha256=digest), config


def _rollout_inputs(horizon: int = 3):
    history = torch.zeros(1, 3, 3, 256)
    history_action = torch.zeros(1, 2, 20)
    future_action = torch.zeros(1, 2, horizon, 20)
    future_action[:, 0, :, 0] = 2.0
    future_action[:, 1, :, 0] = 1.0
    geometry = torch.zeros(1, 2, horizon, 3, 256)
    return history, history_action, future_action, geometry


def test_strict_checkpoint_loader_freezes_verified_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, config = _write_checkpoint(tmp_path / "pair.pt", monkeypatch)
    loaded = load_frozen_p1_native_checkpoint(spec, expected_config=config)

    assert loaded.seed == 15801
    assert loaded.factual.training is False
    assert loaded.no_action.training is False
    assert all(not parameter.requires_grad for parameter in loaded.factual.parameters())
    assert torch.equal(loaded.action_mean, torch.zeros(20))
    assert torch.equal(loaded.action_std, torch.ones(20))


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.__setitem__("checkpoint_type", "wrong"), "checkpoint_type"),
        (lambda value: value.__setitem__("protocol", "wrong"), "protocol"),
        (lambda value: value.__setitem__("step", 1999), "step=2000"),
        (lambda value: value.__setitem__("seed", 15802), "seed"),
        (lambda value: value["config"]["training"].__setitem__("selection", "best"), "config"),
        (lambda value: value.__setitem__("action_std", np.zeros(20, dtype=np.float32)), "strictly positive"),
        (lambda value: value["action_mean"].__setitem__(0, 1.0), "action_mean"),
        (lambda value: value.__setitem__("unexpected", 1), "schema mismatch"),
    ],
)
def test_strict_checkpoint_loader_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
    match: str,
) -> None:
    spec, config = _write_checkpoint(
        tmp_path / "bad.pt", monkeypatch, mutate=mutation
    )
    with pytest.raises((TypeError, ValueError), match=match):
        load_frozen_p1_native_checkpoint(spec, expected_config=config)


def test_strict_checkpoint_loader_rejects_file_hash_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, config = _write_checkpoint(tmp_path / "pair.pt", monkeypatch)
    bad_spec = FrozenP1NativeCheckpointSpec(
        seed=spec.seed, path=spec.path, sha256="0" * 64
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_frozen_p1_native_checkpoint(bad_spec, expected_config=config)


def test_ensemble_requires_all_three_unique_frozen_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    specs = [
        FrozenP1NativeCheckpointSpec(seed=seed, path=Path(f"{seed}.pt"), sha256="a" * 64)
        for seed in (15803, 15801, 15802)
    ]

    def _load(spec: FrozenP1NativeCheckpointSpec, **_kwargs: Any):
        return SimpleNamespace(
            seed=spec.seed,
            predictor_config=P1NativePredictorConfig(),
            action_mean=torch.zeros(20),
            action_std=torch.ones(20),
        )

    monkeypatch.setattr(complementary, "load_frozen_p1_native_checkpoint", _load)
    loaded = load_frozen_p1_native_ensemble(specs, expected_config=config)
    assert tuple(item.seed for item in loaded) == (15801, 15802, 15803)

    with pytest.raises(ValueError, match="exactly three"):
        load_frozen_p1_native_ensemble(specs[:2], expected_config=config)
    duplicate = [specs[0], specs[0], specs[1]]
    with pytest.raises(ValueError, match="seeds 15801"):
        load_frozen_p1_native_ensemble(duplicate, expected_config=config)


def test_gain_zero_is_bitwise_geometry_and_gain_one_is_raw_recursive() -> None:
    model = _freeze(_TinyPredictor(condition_on_action=True))
    history, history_action, future_action, geometry = _rollout_inputs()
    geometry.copy_(torch.arange(3, dtype=torch.float32).view(1, 1, 3, 1, 1))

    zero = recursive_complementary_rollout(
        model, history, history_action, future_action, geometry, gains=(0.0,) * 3
    )
    one = recursive_complementary_rollout(
        model, history, history_action, future_action, geometry, gains=(1.0,) * 3
    )
    raw = rollout_action_branches(model, history, history_action, future_action)

    assert torch.equal(zero, geometry)
    assert torch.equal(one, raw)
    expected_branch_zero = torch.tensor([2.0, 4.0, 6.0])
    expected_branch_one = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(one[0, 0, :, 0, 0], expected_branch_zero)
    assert torch.equal(one[0, 1, :, 0, 0], expected_branch_one)


def test_complementary_posterior_is_written_back_each_step() -> None:
    model = _freeze(_TinyPredictor(condition_on_action=True))
    history, history_action, future_action, geometry = _rollout_inputs()
    result = recursive_complementary_rollout(
        model, history, history_action, future_action, geometry, gains=(0.5,) * 3
    )

    assert torch.allclose(result[0, 0, :, 0, 0], torch.tensor([1.0, 1.5, 1.75]))
    assert torch.allclose(result[0, 1, :, 0, 0], torch.tensor([0.5, 0.75, 0.875]))


def test_matched_rollout_runs_all_seven_wrong_action_controls() -> None:
    factual = _freeze(_TinyPredictor(condition_on_action=True))
    no_action = _freeze(_TinyPredictor(condition_on_action=False))
    history, history_action, future_action, geometry = _rollout_inputs(horizon=2)
    wrong = future_action[:, None].repeat(1, 7, 1, 1, 1)
    for index in range(7):
        wrong[:, index, :, :, 0] = -(index + 1.0)

    output = recursive_complementary_matched_rollout(
        factual,
        no_action,
        history,
        history_action,
        future_action,
        wrong,
        geometry,
        gains=(1.0, 1.0),
    )

    assert output.factual.shape == (1, 2, 2, 3, 256)
    assert output.no_action.shape == output.factual.shape
    assert output.wrong_action.shape == (1, 7, 2, 2, 3, 256)
    assert torch.equal(output.factual[0, 0, :, 0, 0], torch.tensor([2.0, 4.0]))
    assert torch.equal(output.no_action[0, 0, :, 0, 0], torch.tensor([0.25, 0.5]))
    assert torch.equal(output.wrong_action[0, 0, 0, :, 0, 0], torch.tensor([-1.0, -2.0]))


def test_hard_measurement_bypass_is_bitwise_on_valid_and_recovery() -> None:
    measurement = torch.tensor(
        [[[[1.25, -0.0]], [[3.5, 7.25]], [[9.0, -4.0]]]], dtype=torch.float32
    )
    predicted = torch.full_like(measurement, 99.0)
    valid = torch.tensor([[True, False, True]])

    result = hard_measurement_bypass(measurement, predicted, valid)

    assert torch.equal(result[:, 0], measurement[:, 0])
    assert torch.equal(result[:, 1], predicted[:, 1])
    assert torch.equal(result[:, 2], measurement[:, 2])
    assert hard_measurement_bypass(measurement, predicted, torch.tensor(True)) is measurement


def test_rollout_rejects_train_mode_or_unfrozen_predictor() -> None:
    model = _TinyPredictor(condition_on_action=True)
    history, history_action, future_action, geometry = _rollout_inputs()
    with pytest.raises(ValueError, match="eval-mode"):
        recursive_complementary_rollout(
            model, history, history_action, future_action, geometry, gains=(0.5,) * 3
        )
    model.eval()
    with pytest.raises(ValueError, match="frozen"):
        recursive_complementary_rollout(
            model, history, history_action, future_action, geometry, gains=(0.5,) * 3
        )
