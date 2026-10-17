from __future__ import annotations

import inspect
import json

import pytest
import torch
from torch import nn

from fcpg.lewm_posterior_adapter import (
    LeWMPosteriorAdapter,
    MAX_POSTERIOR_ADAPTER_HIDDEN_DIM,
    POSTERIOR_ADAPTER_DEPLOYMENT_INPUTS,
    POSTERIOR_ADAPTER_FORBIDDEN_INPUTS,
)


def _adapter(*, hidden_dim: int = 8) -> LeWMPosteriorAdapter:
    return LeWMPosteriorAdapter(
        latent_dim=5,
        innovation_dim=3,
        confidence_dim=2,
        mask_history_dim=4,
        action_summary_dim=2,
        hidden_dim=hidden_dim,
    )


def _inputs(*, with_time: bool = True) -> tuple[torch.Tensor, ...]:
    leading = (2, 3) if with_time else (2,)
    anchors = 4
    measurement = torch.randn(*leading, anchors, 5)
    prior = measurement + torch.randn_like(measurement)
    innovation = torch.randn(*leading, anchors, 3)
    confidence = torch.rand(*leading, anchors, 2)
    mask_history = torch.randint(
        0, 2, (*leading, anchors, 4), dtype=torch.bool
    )
    action_summary = torch.randn(*leading, 2)
    return (
        measurement,
        prior,
        innovation,
        confidence,
        mask_history,
        action_summary,
    )


@pytest.mark.parametrize("with_time", [False, True])
def test_zero_initialized_head_is_exact_measurement_identity(
    with_time: bool,
) -> None:
    torch.manual_seed(101)
    adapter = _adapter()
    values = _inputs(with_time=with_time)

    output = adapter(*values)

    assert output.beta.shape == (*values[0].shape[:-1], 1)
    assert torch.count_nonzero(output.beta) == 0
    assert torch.equal(output.z_post, values[0])
    assert torch.count_nonzero(adapter.output_layer.weight) == 0
    assert torch.count_nonzero(adapter.output_layer.bias) == 0


def test_zero_identity_keeps_a_trainable_output_boundary() -> None:
    adapter = _adapter()
    measurement = torch.zeros(2, 3, 4, 5)
    prior = torch.ones_like(measurement)
    features = _inputs(with_time=True)[2:]

    output = adapter(measurement, prior, *features)
    output.z_post.sum().backward()

    assert adapter.output_layer.bias.grad is not None
    assert adapter.output_layer.bias.grad.item() > 0.0


def test_scalar_gate_is_bounded_and_shared_across_anchor_permutations() -> None:
    torch.manual_seed(103)
    adapter = _adapter(hidden_dim=0)
    values = list(_inputs(with_time=True))
    values[2] = values[2] + 2.0
    with torch.no_grad():
        adapter.output_layer.weight.zero_()
        adapter.output_layer.weight[0, 0] = 0.75
        adapter.output_layer.bias.fill_(0.25)

    output = adapter(*values)
    expected = values[0] + output.beta * (values[1] - values[0])
    torch.testing.assert_close(output.z_post, expected)
    assert output.beta.shape[-1] == 1
    assert bool((output.beta >= 0.0).all())
    assert bool((output.beta <= 1.0).all())

    permutation = torch.tensor([2, 0, 3, 1])
    permuted = [
        value.index_select(-2, permutation) if index < 5 else value
        for index, value in enumerate(values)
    ]
    permuted_output = adapter(*permuted)
    torch.testing.assert_close(
        permuted_output.beta, output.beta.index_select(-2, permutation)
    )
    torch.testing.assert_close(
        permuted_output.z_post, output.z_post.index_select(-2, permutation)
    )


def test_only_deployment_eligible_features_are_in_the_forward_contract() -> None:
    parameters = inspect.signature(LeWMPosteriorAdapter.forward).parameters
    assert tuple(parameters)[3:] == POSTERIOR_ADAPTER_DEPLOYMENT_INPUTS
    assert set(parameters).isdisjoint(POSTERIOR_ADAPTER_FORBIDDEN_INPUTS)

    adapter = _adapter()
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        adapter(*_inputs(), task_id=torch.zeros(2, 3, 1))


def test_architecture_has_at_most_one_hidden_layer_of_width_32() -> None:
    direct = _adapter(hidden_dim=0)
    bounded = _adapter(hidden_dim=MAX_POSTERIOR_ADAPTER_HIDDEN_DIM)
    assert len([module for module in direct.gate if isinstance(module, nn.Linear)]) == 1
    linear_layers = [
        module for module in bounded.gate if isinstance(module, nn.Linear)
    ]
    assert len(linear_layers) == 2
    assert linear_layers[0].out_features == MAX_POSTERIOR_ADAPTER_HIDDEN_DIM
    assert linear_layers[-1].out_features == 1

    with pytest.raises(ValueError, match="at most 32"):
        _adapter(hidden_dim=33)


def test_non_finite_prior_and_features_fail_closed_per_anchor() -> None:
    adapter = _adapter(hidden_dim=0)
    values = list(_inputs(with_time=True))
    with torch.no_grad():
        adapter.output_layer.weight.zero_()
        adapter.output_layer.bias.fill_(2.0)
    values[2][0, 1, 2, 0] = float("nan")
    values[3][1, 2, 3, 1] = float("inf")
    values[1][1, 0, 0, 4] = float("nan")
    values[5][0, 2, 0] = float("nan")

    output = adapter(*values)

    assert output.beta[0, 1, 2].item() == 0.0
    assert output.beta[1, 2, 3].item() == 0.0
    assert output.beta[1, 0, 0].item() == 0.0
    assert torch.count_nonzero(output.beta[0, 2]) == 0
    assert output.beta[1, 1, 1].item() > 0.0
    torch.testing.assert_close(output.z_post[0, 1, 2], values[0][0, 1, 2])
    torch.testing.assert_close(output.z_post[1, 0, 0], values[0][1, 0, 0])
    assert bool(torch.isfinite(output.z_post).all())
    assert bool(torch.isfinite(output.beta).all())


def test_non_finite_measurement_is_rejected() -> None:
    adapter = _adapter()
    values = list(_inputs())
    values[0][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="z_measurement must be finite"):
        adapter(*values)


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("prior", lambda value: value[:, :, :3], "match z_measurement shape"),
        ("innovation", lambda value: value[:, :, :3], "innovation must have shape"),
        (
            "action_summary",
            lambda value: value.unsqueeze(-2).expand(-1, -1, 4, -1),
            "action_summary must have shape",
        ),
        ("confidence", lambda value: value.double(), "dtype must match"),
        (
            "mask_history",
            lambda value: value.to(torch.int64),
            "floating dtype or bool",
        ),
    ],
)
def test_shape_and_dtype_contracts_fail_closed(
    field: str,
    replacement: object,
    error: str,
) -> None:
    adapter = _adapter()
    names = (
        "measurement",
        "prior",
        "innovation",
        "confidence",
        "mask_history",
        "action_summary",
    )
    values = dict(zip(names, _inputs()))
    transform = replacement
    values[field] = transform(values[field])  # type: ignore[operator]
    with pytest.raises((TypeError, ValueError), match=error):
        adapter(*(values[name] for name in names))


def test_checkpoint_config_and_state_roundtrip() -> None:
    torch.manual_seed(107)
    adapter = _adapter(hidden_dim=7)
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.uniform_(-0.2, 0.2)
    config = json.loads(json.dumps(adapter.checkpoint_config(), sort_keys=True))
    restored = LeWMPosteriorAdapter.from_checkpoint_config(config)
    restored.load_state_dict(adapter.state_dict(), strict=True)

    assert restored.config == adapter.config
    assert restored.checkpoint_config() == config
    assert restored.to_config() == config
    assert set(config).isdisjoint(POSTERIOR_ADAPTER_FORBIDDEN_INPUTS)
    expected = adapter(*_inputs())
    actual = restored(*_inputs())
    assert expected.beta.shape == actual.beta.shape

    bad_config = {**config, "task_id_dim": 4}
    with pytest.raises(ValueError, match="unexpected fields"):
        LeWMPosteriorAdapter.from_checkpoint_config(bad_config)
