from __future__ import annotations

from dataclasses import fields
import inspect

import pytest
import torch
from torch import nn

from fcpg.visual_trust import (
    DEFAULT_VISUAL_TRUST_HIDDEN_DIM,
    MAX_VISUAL_TRUST_HIDDEN_DIM,
    MAX_VISUAL_TRUST_PARAMETERS,
    VisualTrustHead,
    VisualTrustOutput,
)


def _head(*, hidden_dim: int = 8) -> VisualTrustHead:
    return VisualTrustHead(visual_dim=12, auxiliary_dim=5, hidden_dim=hidden_dim)


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(2, 4, 12), torch.randn(2, 4, 5)


def test_output_contract_shape_values_and_gradients() -> None:
    torch.manual_seed(41)
    head = _head()
    visual, auxiliary = _inputs()

    output = head(visual, auxiliary)

    assert isinstance(output, VisualTrustOutput)
    assert tuple(field.name for field in fields(VisualTrustOutput)) == (
        "logits",
        "trust",
    )
    assert output.logits.shape == output.trust.shape == (2, 4)
    assert output.logits.dtype == output.trust.dtype == visual.dtype
    assert output.logits.device == output.trust.device == visual.device
    assert bool(torch.isfinite(output.logits).all())
    assert bool((output.trust >= 0.0).all())
    assert bool((output.trust <= 1.0).all())
    torch.testing.assert_close(output.trust, torch.sigmoid(output.logits))

    output.trust.sum().backward()
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_head_is_shared_and_independent_per_anchor() -> None:
    torch.manual_seed(43)
    head = _head().eval()
    visual, auxiliary = _inputs()
    original = head(visual, auxiliary)

    permutation = torch.tensor([2, 0, 3, 1])
    permuted = head(
        visual.index_select(1, permutation),
        auxiliary.index_select(1, permutation),
    )
    torch.testing.assert_close(
        permuted.logits, original.logits.index_select(1, permutation)
    )
    torch.testing.assert_close(
        permuted.trust, original.trust.index_select(1, permutation)
    )

    changed_visual = visual.clone()
    changed_visual[0, 1] += 10.0
    changed = head(changed_visual, auxiliary)
    unchanged = torch.ones_like(original.logits, dtype=torch.bool)
    unchanged[0, 1] = False
    torch.testing.assert_close(changed.logits[unchanged], original.logits[unchanged])


def test_forward_accepts_only_current_visual_and_auxiliary_inputs() -> None:
    parameters = inspect.signature(VisualTrustHead.forward).parameters
    assert tuple(parameters) == (
        "self",
        "current_visual_features",
        "auxiliary_descriptors",
    )
    forbidden = {
        "lewm_prior",
        "action",
        "event",
        "task",
        "schedule",
        "oracle",
    }
    assert set(parameters).isdisjoint(forbidden)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _head()(*_inputs(), action=torch.zeros(2, 1))


def test_architecture_and_parameter_budget_are_hard_bounded() -> None:
    head = VisualTrustHead(visual_dim=384, auxiliary_dim=8)
    assert isinstance(head.input_norm, nn.LayerNorm)
    assert head.hidden_dim == DEFAULT_VISUAL_TRUST_HIDDEN_DIM
    linear_layers = [module for module in head.mlp if isinstance(module, nn.Linear)]
    assert len(linear_layers) == 2
    assert linear_layers[0].out_features == MAX_VISUAL_TRUST_HIDDEN_DIM
    assert linear_layers[1].out_features == 1
    actual_count = sum(parameter.numel() for parameter in head.parameters())
    assert head.parameter_count == actual_count < MAX_VISUAL_TRUST_PARAMETERS

    with pytest.raises(ValueError, match="at most 32"):
        VisualTrustHead(visual_dim=12, auxiliary_dim=5, hidden_dim=33)
    with pytest.raises(ValueError, match="fewer than 20000"):
        VisualTrustHead(visual_dim=768, auxiliary_dim=8)


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        (
            "visual",
            lambda value: value.unsqueeze(0),
            "current_visual_features must have shape",
        ),
        (
            "visual",
            lambda value: value[..., :-1],
            "current_visual_features must have shape",
        ),
        (
            "auxiliary",
            lambda value: value[:, :-1],
            "matching batch and anchor dimensions",
        ),
        (
            "auxiliary",
            lambda value: value[..., :-1],
            "auxiliary_descriptors must have shape",
        ),
        (
            "visual",
            lambda value: value[:0],
            "dimensions must be positive",
        ),
    ],
)
def test_shape_validation_is_strict(
    field: str,
    replacement: object,
    error: str,
) -> None:
    visual, auxiliary = _inputs()
    values = {"visual": visual, "auxiliary": auxiliary}
    transform = replacement
    values[field] = transform(values[field])  # type: ignore[operator]

    with pytest.raises(ValueError, match=error):
        _head()(values["visual"], values["auxiliary"])


def test_dtype_validation_is_strict_and_double_is_supported() -> None:
    visual, auxiliary = _inputs()
    with pytest.raises(TypeError, match="floating dtype"):
        _head()(visual.to(torch.int64), auxiliary)
    with pytest.raises(TypeError, match="dtypes must match"):
        _head()(visual, auxiliary.double())
    with pytest.raises(TypeError, match="parameter dtype"):
        _head()(visual.double(), auxiliary.double())

    double_head = _head().double()
    output = double_head(visual.double(), auxiliary.double())
    assert output.logits.dtype == output.trust.dtype == torch.float64


@pytest.mark.parametrize("field", ["visual", "auxiliary"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_non_finite_inputs_are_rejected(field: str, bad_value: float) -> None:
    visual, auxiliary = _inputs()
    values = {"visual": visual, "auxiliary": auxiliary}
    values[field] = values[field].clone()
    values[field].reshape(-1)[0] = bad_value

    with pytest.raises(ValueError, match="must be finite"):
        _head()(values["visual"], values["auxiliary"])


def test_non_finite_model_output_is_rejected() -> None:
    head = _head()
    with torch.no_grad():
        head.mlp[-1].bias.fill_(float("nan"))

    with pytest.raises(ValueError, match="logits must be finite"):
        head(*_inputs())


@pytest.mark.parametrize(
    ("name", "value", "error"),
    [
        ("visual_dim", True, "must be an integer"),
        ("auxiliary_dim", 0, "must be positive"),
        ("hidden_dim", 2.5, "must be an integer"),
    ],
)
def test_constructor_dimensions_are_strict(
    name: str,
    value: object,
    error: str,
) -> None:
    kwargs: dict[str, object] = {
        "visual_dim": 12,
        "auxiliary_dim": 5,
        "hidden_dim": 8,
    }
    kwargs[name] = value
    with pytest.raises((TypeError, ValueError), match=error):
        VisualTrustHead(**kwargs)  # type: ignore[arg-type]
