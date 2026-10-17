from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from fcpg.lewm_posterior_dagger import (
    ACTION_DIM,
    ACTION_STEPS,
    MAX_MISSING_AGE,
    METHODS,
    PROTOCOL,
    PosteriorDaggerArrays,
    PosteriorDaggerResidualHead,
    load_posterior_dagger_arrays,
    load_posterior_dagger_checkpoint,
    method_index,
    posterior_dagger_xyz_loss,
    save_posterior_dagger_arrays,
    save_posterior_dagger_checkpoint,
    validate_posterior_dagger_arrays,
)


def _arrays(count: int = 6, model_dim: int = 12) -> PosteriorDaggerArrays:
    rng = np.random.default_rng(73)
    return PosteriorDaggerArrays(
        episode_id=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)[:count],
        decision_index=np.asarray([0, 0, 0, 2, 2, 2], dtype=np.int64)[:count],
        dagger_iteration=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)[:count],
        method_index=np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)[:count],
        missing_age=np.arange(1, count + 1, dtype=np.int64),
        sensor_valid=np.zeros(count, dtype=bool),
        state_latent=rng.normal(size=(count, model_dim)).astype(np.float32),
        action_latent=rng.normal(
            size=(count, ACTION_STEPS, model_dim)
        ).astype(np.float32),
        base_action=rng.uniform(
            -1, 1, size=(count, ACTION_STEPS, ACTION_DIM)
        ).astype(np.float32),
        posterior_token=rng.normal(size=(count, 3, model_dim)).astype(np.float32),
        expert_action=rng.uniform(
            -1, 1, size=(count, ACTION_STEPS, ACTION_DIM)
        ).astype(np.float32),
        executed_action=rng.uniform(
            -1, 1, size=(count, ACTION_STEPS, ACTION_DIM)
        ).astype(np.float32),
    )


def _model_inputs(
    *, batch: int = 3, model_dim: int = 8
) -> dict[str, torch.Tensor]:
    return {
        "base_action": torch.rand(batch, ACTION_STEPS, ACTION_DIM) * 1.8 - 0.9,
        "state_latent": torch.randn(batch, model_dim),
        "action_latent": torch.randn(batch, ACTION_STEPS, model_dim),
        "posterior_token": torch.randn(batch, 3, model_dim),
        "missing_age": torch.tensor([0, 2, MAX_MISSING_AGE][:batch]),
        "sensor_valid": torch.zeros(batch, dtype=torch.bool),
    }


def _rewrite_npz(path, **updates: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.array(archive[key], copy=True) for key in archive.files}
    payload.update(updates)
    np.savez_compressed(path, **payload)


def test_constants_and_method_indices_are_frozen() -> None:
    assert METHODS == (
        "complementary_factual",
        "complementary_no_action",
        "geometry_hold",
    )
    assert ACTION_STEPS == 5
    assert ACTION_DIM == 4
    assert MAX_MISSING_AGE == 6
    assert [method_index(method) for method in METHODS] == [0, 1, 2]
    with pytest.raises(ValueError, match="unknown posterior DAgger method"):
        method_index("unknown")


def test_posterior_dagger_arrays_round_trip(tmp_path) -> None:
    arrays = _arrays()
    path = tmp_path / "dagger.npz"
    save_posterior_dagger_arrays(path, arrays)
    loaded = load_posterior_dagger_arrays(path)
    assert loaded.sample_count == arrays.sample_count
    assert loaded.model_dim == arrays.model_dim
    assert loaded.posterior_token_count == 3
    for field in arrays.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(loaded, field), getattr(arrays, field))
    with np.load(path, allow_pickle=False) as archive:
        assert archive["protocol"].item() == PROTOCOL
        assert tuple(archive["methods"].astype(str).tolist()) == METHODS


@pytest.mark.parametrize(
    ("field", "replacement", "error", "exception"),
    [
        (
            "episode_id",
            lambda a: a.astype(np.float32),
            "episode_id must be an integer vector",
            ValueError,
        ),
        (
            "decision_index",
            lambda a: np.full_like(a, -1),
            "must be non-negative",
            ValueError,
        ),
        (
            "dagger_iteration",
            lambda a: np.full_like(a, -1),
            "must be non-negative",
            ValueError,
        ),
        (
            "method_index",
            lambda a: np.full_like(a, len(METHODS)),
            "outside the frozen method ordering",
            ValueError,
        ),
        (
            "missing_age",
            lambda a: np.zeros_like(a),
            "outside the invalid-observation range",
            ValueError,
        ),
        (
            "sensor_valid",
            lambda a: np.ones_like(a),
            "invalid observations only",
            PermissionError,
        ),
        (
            "sensor_valid",
            lambda a: a.astype(np.int64),
            "must be a boolean vector",
            ValueError,
        ),
        (
            "state_latent",
            lambda a: a[:, None, :],
            r"shape \[N,D\]",
            ValueError,
        ),
        (
            "action_latent",
            lambda a: a[:, :-1],
            r"shape \[N,5,D\]",
            ValueError,
        ),
        (
            "posterior_token",
            lambda a: a[:, :0],
            r"shape \[N,K,D\]",
            ValueError,
        ),
        (
            "base_action",
            lambda a: a[:, :-1],
            r"shape \[N,5,4\]",
            ValueError,
        ),
        (
            "expert_action",
            lambda a: a.astype(np.float64) * np.nan,
            "finite floating point",
            ValueError,
        ),
        (
            "executed_action",
            lambda a: np.full_like(a, 1.1),
            "environment action bounds",
            ValueError,
        ),
    ],
)
def test_array_contract_rejects_invalid_fields(
    field: str,
    replacement,
    error: str,
    exception: type[Exception],
) -> None:
    arrays = _arrays()
    changed = replace(arrays, **{field: replacement(getattr(arrays, field))})
    with pytest.raises(exception, match=error):
        validate_posterior_dagger_arrays(changed)


def test_array_contract_rejects_empty_and_duplicate_rows() -> None:
    arrays = _arrays()
    empty = replace(
        arrays,
        **{
            field: getattr(arrays, field)[:0]
            for field in arrays.__dataclass_fields__
        },
    )
    with pytest.raises(ValueError, match="at least one row"):
        validate_posterior_dagger_arrays(empty)

    duplicate = replace(
        arrays,
        episode_id=np.zeros(6, dtype=np.int64),
        decision_index=np.zeros(6, dtype=np.int64),
        dagger_iteration=np.zeros(6, dtype=np.int64),
        method_index=np.zeros(6, dtype=np.int64),
    )
    with pytest.raises(ValueError, match="identities must be unique"):
        validate_posterior_dagger_arrays(duplicate)


@pytest.mark.parametrize(
    ("metadata", "error"),
    [
        ({"schema_version": np.asarray(2)}, "schema is unsupported"),
        ({"protocol": np.asarray("other")}, "protocol is unsupported"),
        ({"methods": np.asarray(METHODS[::-1])}, "method ordering drifted"),
        ({"action_steps": np.asarray(4)}, "constant 'action_steps' drifted"),
    ],
)
def test_npz_loader_rejects_metadata_drift(tmp_path, metadata, error: str) -> None:
    path = tmp_path / "dagger.npz"
    save_posterior_dagger_arrays(path, _arrays())
    _rewrite_npz(path, **metadata)
    with pytest.raises(ValueError, match=error):
        load_posterior_dagger_arrays(path)


def test_zero_initialized_head_is_exact_identity() -> None:
    torch.manual_seed(79)
    model = PosteriorDaggerResidualHead(model_dim=8, hidden_dim=24)
    inputs = _model_inputs()
    result = model(**inputs)
    assert torch.equal(result, inputs["base_action"])


def test_valid_rows_bypass_nonzero_head_and_gripper_is_always_passthrough() -> None:
    model = PosteriorDaggerResidualHead(model_dim=8, hidden_dim=24)
    with torch.no_grad():
        model.output_projection.bias.fill_(2.0)
    inputs = _model_inputs()
    inputs["sensor_valid"] = torch.tensor([True, False, True])
    result = model(**inputs)
    assert torch.equal(result[0], inputs["base_action"][0])
    assert torch.equal(result[2], inputs["base_action"][2])
    assert not torch.equal(result[1, :, :3], inputs["base_action"][1, :, :3])
    assert torch.equal(result[..., 3], inputs["base_action"][..., 3])


def test_head_output_respects_correction_cap_and_action_bounds() -> None:
    cap = 0.2
    model = PosteriorDaggerResidualHead(
        model_dim=8, hidden_dim=24, max_xyz_correction=cap
    )
    with torch.no_grad():
        model.output_projection.weight.fill_(20.0)
        model.output_projection.bias.fill_(20.0)
    inputs = _model_inputs()
    inputs["base_action"][0, :, :3] = 0.95
    inputs["base_action"][1, :, :3] = -0.95
    result = model(**inputs)
    delta = (result[..., :3] - inputs["base_action"][..., :3]).abs()
    assert bool(torch.all(delta <= cap + 1e-6))
    assert bool(torch.all(result <= 1.0) and torch.all(result >= -1.0))
    assert torch.equal(result[..., 3], inputs["base_action"][..., 3])


@pytest.mark.parametrize(
    ("field", "replacement", "error", "exception"),
    [
        (
            "posterior_token",
            lambda value: value[:, :0],
            "posterior_token must have shape",
            ValueError,
        ),
        (
            "missing_age",
            lambda value: value.to(torch.int32),
            "must be a long tensor",
            ValueError,
        ),
        (
            "sensor_valid",
            lambda value: value.to(torch.int64),
            "must be a bool tensor",
            ValueError,
        ),
        (
            "state_latent",
            lambda value: value.fill_(float("nan")),
            "non-finite",
            ValueError,
        ),
        (
            "base_action",
            lambda value: value.fill_(1.2),
            "action bounds",
            ValueError,
        ),
    ],
)
def test_head_input_contract_fails_closed(
    field: str,
    replacement,
    error: str,
    exception: type[Exception],
) -> None:
    model = PosteriorDaggerResidualHead(model_dim=8)
    inputs = _model_inputs()
    inputs[field] = replacement(inputs[field])
    with pytest.raises(exception, match=error):
        model(**inputs)


def test_xyz_loss_ignores_gripper_and_rejects_bad_inputs() -> None:
    prediction = torch.zeros(2, ACTION_STEPS, ACTION_DIM)
    target = prediction.clone()
    target[..., 3] = 1.0
    assert posterior_dagger_xyz_loss(prediction, target).item() == 0.0
    target[..., 0] = 0.2
    assert posterior_dagger_xyz_loss(prediction, target).item() > 0.0
    with pytest.raises(ValueError, match="aligned"):
        posterior_dagger_xyz_loss(prediction[:, :-1], target)
    with pytest.raises(ValueError, match="beta must be positive"):
        posterior_dagger_xyz_loss(prediction, target, beta=0.0)


def test_checkpoint_roundtrip_preserves_identity_and_freezes_model(tmp_path) -> None:
    torch.manual_seed(83)
    source = PosteriorDaggerResidualHead(
        model_dim=8, hidden_dim=19, age_dim=7, max_xyz_correction=0.15
    ).eval()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.uniform_(-0.1, 0.1)
    path = tmp_path / "head.pt"
    save_posterior_dagger_checkpoint(
        path,
        source,
        method="complementary_factual",
        seed=91,
        steps=2000,
        dagger_iteration=1,
    )
    loaded, metadata = load_posterior_dagger_checkpoint(
        path,
        expected_method="complementary_factual",
        expected_seed=91,
        expected_steps=2000,
        expected_dagger_iteration=1,
    )
    assert metadata["method_index"] == 0
    assert metadata["architecture"] == source.checkpoint_config()
    assert not loaded.training
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    for key, value in source.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)
    inputs = _model_inputs()
    torch.testing.assert_close(loaded(**inputs), source(**inputs))

    with pytest.raises(ValueError, match="method mismatch"):
        load_posterior_dagger_checkpoint(path, expected_method="geometry_hold")


def test_checkpoint_loader_rejects_metadata_tampering(tmp_path) -> None:
    model = PosteriorDaggerResidualHead(model_dim=8)
    path = tmp_path / "head.pt"
    save_posterior_dagger_checkpoint(
        path,
        model,
        method="geometry_hold",
        seed=3,
        steps=10,
        dagger_iteration=0,
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["method_index"] = 0
    torch.save(payload, path)
    with pytest.raises(ValueError, match="method identity is inconsistent"):
        load_posterior_dagger_checkpoint(path)
