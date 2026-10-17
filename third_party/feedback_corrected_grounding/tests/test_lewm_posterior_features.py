from __future__ import annotations

from dataclasses import fields
import inspect
import math

import pytest
import torch

from fcpg.lewm_posterior_adapter import (
    LeWMPosteriorAdapter,
    POSTERIOR_ADAPTER_DEPLOYMENT_INPUTS,
)
from fcpg.lewm_posterior_features import (
    LATENT_INNOVATION_CLIP,
    LOCAL_SUPPORT_GRID_SIZE,
    LOCAL_SUPPORT_RADIUS_PX,
    LeWMPosteriorFeatures,
    MAX_RESIDUAL_PX,
    POSTERIOR_CONFIDENCE_FIELDS,
    POSTERIOR_FEATURE_ALLOWED_FIELDS,
    POSTERIOR_FEATURE_FORBIDDEN_FIELDS,
    POSTERIOR_FEATURE_GROUPS,
    build_lewm_posterior_features,
)


def _inputs(*, with_time: bool = True) -> dict[str, object]:
    leading = (2, 3) if with_time else (2,)
    anchors = 3
    latent_dim = 4
    measurement = torch.zeros(*leading, anchors, latent_dim)
    prior = measurement + 0.25
    nominal = torch.full((*leading, anchors, 2), 40.0)
    corrected = nominal.clone()
    valid = torch.ones(*leading, anchors, dtype=torch.bool)
    validity_history = torch.ones(*leading, anchors, 2, dtype=torch.bool)
    action_summary = torch.arange(5, dtype=torch.float32).expand(*leading, 5)
    return {
        "z_measurement": measurement,
        "z_prior": prior,
        "q_nominal": nominal,
        "q_corrected": corrected,
        "valid_nominal": valid,
        "image_width": 96,
        "image_height": 80,
        "online_validity_history": validity_history,
        "causal_action_summary": action_summary,
    }


def _build(values: dict[str, object]) -> LeWMPosteriorFeatures:
    return build_lewm_posterior_features(**values)  # type: ignore[arg-type]


def test_contract_exposes_only_deployment_fields_and_adapter_groups() -> None:
    parameters = inspect.signature(build_lewm_posterior_features).parameters
    assert tuple(parameters) == POSTERIOR_FEATURE_ALLOWED_FIELDS
    assert set(parameters).isdisjoint(POSTERIOR_FEATURE_FORBIDDEN_FIELDS)
    assert tuple(field.name for field in fields(LeWMPosteriorFeatures)) == (
        POSTERIOR_FEATURE_GROUPS
    )
    assert POSTERIOR_FEATURE_GROUPS == POSTERIOR_ADAPTER_DEPLOYMENT_INPUTS
    assert {
        "dataset_mapping",
        "task_id",
        "q_true",
        "measurement_valid",
        "anchor_dropout_active",
        "camera_step_frame",
        "canonical",
        "target",
    }.issubset(POSTERIOR_FEATURE_FORBIDDEN_FIELDS)


@pytest.mark.parametrize("forbidden", POSTERIOR_FEATURE_FORBIDDEN_FIELDS)
def test_forbidden_kwargs_cannot_enter_builder(forbidden: str) -> None:
    values = _inputs()
    values[forbidden] = torch.tensor(123.0)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _build(values)


def test_features_match_exact_formulas_and_feed_adapter() -> None:
    values = _inputs(with_time=False)
    measurement = values["z_measurement"]
    assert isinstance(measurement, torch.Tensor)
    prior = measurement.clone()
    prior[0, 0] = torch.tensor([-2.0, -0.5, 0.25, 3.0])
    values["z_prior"] = prior

    nominal = values["q_nominal"]
    corrected = values["q_corrected"]
    valid = values["valid_nominal"]
    assert isinstance(nominal, torch.Tensor)
    assert isinstance(corrected, torch.Tensor)
    assert isinstance(valid, torch.Tensor)
    corrected[0, 0] = nominal[0, 0] + torch.tensor([10.0, -20.0])
    corrected[0, 1] = torch.tensor([-1.0, 40.0])
    valid[0, 0] = False

    features = _build(values)

    expected_innovation = torch.tensor([-1.0, -0.5, 0.25, 1.0])
    torch.testing.assert_close(features.innovation[0, 0], expected_innovation)
    expected_norm = math.sqrt(10.0**2 + 20.0**2) / (
        math.sqrt(2.0) * MAX_RESIDUAL_PX
    )
    torch.testing.assert_close(
        features.confidence[0, 0],
        torch.tensor([0.5, -1.0, expected_norm, 1.0, 0.0, 1.0]),
    )
    assert features.confidence[0, 1, 5].item() == 0.0
    assert features.confidence.shape == (2, 3, len(POSTERIOR_CONFIDENCE_FIELDS))
    assert features.mask_history.shape == (2, 3, 2)
    torch.testing.assert_close(
        features.mask_history,
        values["online_validity_history"].to(dtype=measurement.dtype),  # type: ignore[union-attr]
    )
    assert features.action_summary is values["causal_action_summary"]

    adapter = LeWMPosteriorAdapter(
        latent_dim=measurement.shape[-1],
        innovation_dim=features.innovation.shape[-1],
        confidence_dim=features.confidence.shape[-1],
        mask_history_dim=features.mask_history.shape[-1],
        action_summary_dim=features.action_summary.shape[-1],
        hidden_dim=0,
    )
    output = adapter(
        measurement,
        prior,
        **features.as_adapter_kwargs(),
    )
    assert output.z_post.shape == measurement.shape


def test_local_support_is_fixed_nominal_grid_in_bounds_fraction() -> None:
    values = _inputs(with_time=False)
    nominal = values["q_nominal"]
    corrected = values["q_corrected"]
    assert isinstance(nominal, torch.Tensor)
    assert isinstance(corrected, torch.Tensor)
    radius = float(LOCAL_SUPPORT_RADIUS_PX)
    nominal[0, 0] = torch.tensor([radius, radius])
    nominal[0, 1] = torch.tensor([0.0, 0.0])
    nominal[0, 2] = torch.tensor([-radius - 1.0, 30.0])
    corrected.copy_(nominal)

    features = _build(values)

    torch.testing.assert_close(
        features.confidence[0, :, 3],
        torch.tensor(
            [
                1.0,
                (LOCAL_SUPPORT_GRID_SIZE // 2 + 1) ** 2
                / float(LOCAL_SUPPORT_GRID_SIZE**2),
                0.0,
            ]
        ),
    )
    torch.testing.assert_close(
        features.confidence[0, :, 5], torch.tensor([1.0, 1.0, 0.0])
    )


def test_latent_innovation_is_full_normalized_and_clipped() -> None:
    values = _inputs()
    measurement = values["z_measurement"]
    assert isinstance(measurement, torch.Tensor)
    prior = measurement.clone()
    prior[..., 0] = -10.0 * LATENT_INNOVATION_CLIP
    prior[..., 1] = -0.5 * LATENT_INNOVATION_CLIP
    prior[..., 2] = 0.5 * LATENT_INNOVATION_CLIP
    prior[..., 3] = 10.0 * LATENT_INNOVATION_CLIP
    values["z_prior"] = prior

    features = _build(values)

    expected = torch.tensor([-1.0, -0.5, 0.5, 1.0])
    torch.testing.assert_close(
        features.innovation, expected.expand_as(features.innovation)
    )
    assert features.innovation.shape == prior.shape
    assert float(features.innovation.abs().max()) == 1.0


@pytest.mark.parametrize("with_time", [False, True])
def test_outputs_preserve_shape_dtype_device_and_are_deterministic(
    with_time: bool,
) -> None:
    values = _inputs(with_time=with_time)
    first = _build(values)
    second = _build(values)
    reference = values["z_measurement"]
    assert isinstance(reference, torch.Tensor)

    for group in POSTERIOR_FEATURE_GROUPS:
        first_value = getattr(first, group)
        second_value = getattr(second, group)
        assert first_value.dtype == reference.dtype
        assert first_value.device == reference.device
        assert bool(torch.isfinite(first_value).all())
        assert torch.equal(first_value, second_value)


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        (
            "z_prior",
            lambda value: value[..., :2, :],
            "z_prior must match z_measurement shape",
        ),
        (
            "q_nominal",
            lambda value: value[..., :1],
            "q_nominal must have shape",
        ),
        (
            "q_corrected",
            lambda value: value[..., :2, :],
            "q_corrected must have shape",
        ),
        (
            "valid_nominal",
            lambda value: value[..., :2],
            "valid_nominal must have shape",
        ),
        (
            "online_validity_history",
            lambda value: value[..., :2, :],
            "online_validity_history must have shape",
        ),
        (
            "causal_action_summary",
            lambda value: value.unsqueeze(-2),
            "causal_action_summary must have shape",
        ),
    ],
)
def test_shape_validation_fails_closed(
    field: str,
    replacement: object,
    error: str,
) -> None:
    values = _inputs()
    transform = replacement
    values[field] = transform(values[field])  # type: ignore[operator]
    with pytest.raises(ValueError, match=error):
        _build(values)


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        (
            "z_measurement",
            lambda value: value.to(torch.int64),
            "z_measurement must have a floating dtype",
        ),
        (
            "z_prior",
            lambda value: value.double(),
            "z_prior dtype must match",
        ),
        (
            "q_nominal",
            lambda value: value.double(),
            "q_nominal dtype must match",
        ),
        (
            "q_corrected",
            lambda value: value.double(),
            "q_corrected dtype must match",
        ),
        (
            "valid_nominal",
            lambda value: value.float(),
            "valid_nominal must have boolean dtype",
        ),
        (
            "online_validity_history",
            lambda value: value.to(torch.int64),
            "boolean or floating dtype",
        ),
        (
            "causal_action_summary",
            lambda value: value.double(),
            "dtype must match",
        ),
    ],
)
def test_dtype_validation_fails_closed(
    field: str,
    replacement: object,
    error: str,
) -> None:
    values = _inputs()
    transform = replacement
    values[field] = transform(values[field])  # type: ignore[operator]
    with pytest.raises((TypeError, ValueError), match=error):
        _build(values)


@pytest.mark.parametrize(
    "field",
    (
        "z_measurement",
        "z_prior",
        "q_nominal",
        "q_corrected",
        "causal_action_summary",
    ),
)
def test_non_finite_inputs_are_rejected(field: str) -> None:
    values = _inputs()
    value = values[field]
    assert isinstance(value, torch.Tensor)
    value = value.clone()
    value.reshape(-1)[0] = float("nan")
    values[field] = value
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        _build(values)


@pytest.mark.parametrize(
    ("field", "bad_value", "error"),
    [
        ("online_validity_history", 0.5, "values must be binary"),
    ],
)
def test_online_history_values_are_strictly_validated(
    field: str, bad_value: float, error: str
) -> None:
    values = _inputs()
    history = values[field]
    assert isinstance(history, torch.Tensor)
    history = history.to(dtype=torch.float32)
    history.reshape(-1)[0] = bad_value
    values[field] = history
    with pytest.raises(ValueError, match=error):
        _build(values)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("image_width", True, "must be an integer"),
        ("image_height", 12.5, "must be an integer"),
        ("image_width", 1, "must be at least 2"),
        ("image_height", 0, "must be at least 2"),
    ],
)
def test_image_bounds_are_strictly_validated(
    field: str, value: object, error: str
) -> None:
    values = _inputs()
    values[field] = value
    with pytest.raises((TypeError, ValueError), match=error):
        _build(values)
