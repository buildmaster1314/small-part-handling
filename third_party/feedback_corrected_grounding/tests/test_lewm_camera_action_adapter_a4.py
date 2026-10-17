from __future__ import annotations

import torch

from fcpg.lewm_action_prefix import ActionPrefixDynamics
from fcpg.lewm_camera_action_adapter_a4 import (
    AnchorResidualAdapter,
    BoundedAnchorAdapterDynamics,
)
from scripts.continue_lewm_camera_action_joint_a4 import (
    _parser as continuation_parser,
)
from scripts.train_lewm_camera_action_adapter_a4 import _parser as adapter_parser


def _tiny_base() -> ActionPrefixDynamics:
    return ActionPrefixDynamics(
        action_dim=4,
        state_dim=12,
        num_slots=4,
        num_tasks=7,
        max_horizon=3,
        prefix_depth=1,
        prefix_heads=3,
        prefix_mlp_dim=24,
        predictor_depth=1,
        predictor_hidden_dim=24,
        fusion_dim=24,
        state_mlp_dim=24,
        dropout=0.0,
    )


def _tiny_adapter(variant: str = "transformer") -> AnchorResidualAdapter:
    return AnchorResidualAdapter(
        variant=variant,
        state_dim=12,
        num_slots=4,
        num_tasks=7,
        max_horizon=3,
        model_dim=12,
        depth=1,
        heads=3,
        ffn_dim=24,
        dropout=0.0,
    )


def test_zero_initialized_adapter_exactly_preserves_base_prediction() -> None:
    torch.manual_seed(11)
    base = _tiny_base().eval()
    module = BoundedAnchorAdapterDynamics(
        base,
        _tiny_adapter().eval(),
        torch.full((3, 4, 12), 0.2),
    ).eval()
    state = torch.randn(2, 4, 12)
    action = torch.randn(2, 3, 4)
    task = torch.tensor([0, 6])

    expected = base(state, action, task_index=task)
    observed = module(state, action, task_index=task)

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_adapter_delta_is_strictly_bounded_per_horizon_slot_feature() -> None:
    cap = torch.linspace(0.01, 0.25, 3 * 4 * 12).reshape(3, 4, 12)
    module = BoundedAnchorAdapterDynamics(
        _tiny_base(),
        _tiny_adapter(),
        cap,
    )
    raw = torch.full((2, 3, 4, 12), 1.0e6)
    raw[1].neg_()

    delta = module.bounded_delta(raw)

    assert bool(torch.all(delta.abs() <= cap[None]))
    torch.testing.assert_close(delta[0], cap)
    torch.testing.assert_close(delta[1], -cap)


def test_transformer_and_local_mlp_have_matched_parameter_budgets() -> None:
    common = {
        "state_dim": 192,
        "num_slots": 4,
        "num_tasks": 7,
        "max_horizon": 8,
        "model_dim": 192,
        "depth": 2,
        "heads": 6,
        "ffn_dim": 384,
        "dropout": 0.1,
    }
    transformer = AnchorResidualAdapter(variant="transformer", **common)
    mlp = AnchorResidualAdapter(variant="mlp", **common)
    transformer_count = sum(p.numel() for p in transformer.parameters())
    mlp_count = sum(p.numel() for p in mlp.parameters())

    assert transformer_count == 782_784
    assert mlp_count == 781_248
    assert abs(transformer_count - mlp_count) / transformer_count < 0.005


def test_adapter_accepts_mixed_autocast_parent_dtypes() -> None:
    adapter = _tiny_adapter().eval()
    current = torch.randn(2, 4, 12, dtype=torch.float32)
    base = torch.randn(2, 3, 4, 12, dtype=torch.bfloat16)
    prefix = torch.randn(2, 3, 12, dtype=torch.float32)
    task = torch.tensor([1, 5])

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = adapter(current, base, prefix, task)

    assert result.shape == (2, 3, 4, 12)
    assert result.dtype == torch.bfloat16


def test_adapter_requires_task_conditioning() -> None:
    base = ActionPrefixDynamics(
        action_dim=4,
        state_dim=12,
        num_slots=4,
        num_tasks=None,
        max_horizon=3,
        prefix_depth=1,
        prefix_heads=3,
        prefix_mlp_dim=24,
        predictor_depth=1,
        predictor_hidden_dim=24,
        fusion_dim=24,
        state_mlp_dim=24,
        dropout=0.0,
    )
    adapter = _tiny_adapter()

    try:
        BoundedAnchorAdapterDynamics(
            base,
            adapter,
            torch.ones(3, 4, 12),
        )
    except ValueError as error:
        assert "contracts differ" in str(error)
    else:
        raise AssertionError("task-free base must not accept a task adapter")


def test_adapter_and_continuation_do_not_expose_sigreg_sweeps() -> None:
    adapter_args = adapter_parser(
        [
            "--variant",
            "transformer",
            "--output-dir",
            "/tmp/adapter",
        ]
    )
    continuation_args = continuation_parser(
        ["--output-dir", "/tmp/continuation"]
    )

    assert not hasattr(adapter_args, "sigreg_weight")
    assert not hasattr(continuation_args, "sigreg_weight")
