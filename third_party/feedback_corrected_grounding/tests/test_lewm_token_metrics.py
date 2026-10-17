from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fcpg.lewm_token_metrics import (
    OracleTokenErrorStats,
    OracleTokenLayout,
    merge_oracle_token_errors,
    oracle_token_layout,
    oracle_token_squared_error,
    split_oracle_tokens,
    summarize_oracle_token_mse,
)


SPEC = {
    "num_keypoints": 3,
    "target_dim": 12,
    "token_dim": 3,
    "token_order": "global_then_keypoints",
}


def test_layout_and_split_preserve_arbitrary_prefix_dimensions() -> None:
    value = torch.arange(2 * 5 * 12, dtype=torch.float32).reshape(2, 5, 12)

    layout = oracle_token_layout(SPEC)
    blocks = split_oracle_tokens(value, SPEC)

    assert layout == OracleTokenLayout(num_anchors=3, token_dim=3)
    assert layout.token_count == 4
    assert layout.target_dim == 12
    assert blocks.global_token.shape == (2, 5, 3)
    assert blocks.anchor_tokens.shape == (2, 5, 3, 3)
    torch.testing.assert_close(blocks.global_token, value[..., :3])
    torch.testing.assert_close(
        blocks.anchor_tokens.reshape(2, 5, 9), value[..., 3:]
    )

    flattened = split_oracle_tokens(value.reshape(-1, 12), SPEC)
    torch.testing.assert_close(
        flattened.global_token, blocks.global_token.reshape(-1, 3)
    )
    torch.testing.assert_close(
        flattened.anchor_tokens, blocks.anchor_tokens.reshape(-1, 3, 3)
    )


def test_token_mse_reports_global_each_anchor_macro_and_overall() -> None:
    target = torch.zeros(2, 2, 12)
    prediction = torch.zeros_like(target)
    prediction[..., 0:3] = 1.0
    prediction[..., 3:6] = 2.0
    prediction[..., 6:9] = 3.0
    prediction[..., 9:12] = 4.0

    stats = oracle_token_squared_error(prediction, target, SPEC)
    summary = summarize_oracle_token_mse(stats)

    assert stats.prefix_count == 4
    assert stats.token_squared_error == pytest.approx((12.0, 48.0, 108.0, 192.0))
    assert summary["global_mse"] == pytest.approx(1.0)
    assert summary["anchor_mse"] == pytest.approx([4.0, 9.0, 16.0])
    assert summary["anchor_macro_mse"] == pytest.approx(29.0 / 3.0)
    assert summary["overall_mse"] == pytest.approx(7.5)
    assert summary["elements_per_token"] == 12


def test_merge_weights_unequal_batches_by_element_count() -> None:
    first_target = torch.zeros(1, 12)
    first_prediction = torch.ones_like(first_target)
    second_target = torch.zeros(3, 12)
    second_prediction = torch.full_like(second_target, 3.0)

    merged = merge_oracle_token_errors(
        [
            oracle_token_squared_error(first_prediction, first_target, SPEC),
            oracle_token_squared_error(second_prediction, second_target, SPEC),
        ]
    )
    summary = summarize_oracle_token_mse(merged)

    assert merged.prefix_count == 4
    assert summary["global_mse"] == pytest.approx(7.0)
    assert summary["anchor_mse"] == pytest.approx([7.0, 7.0, 7.0])
    assert summary["anchor_macro_mse"] == pytest.approx(7.0)
    assert summary["overall_mse"] == pytest.approx(7.0)


@pytest.mark.parametrize(
    ("spec", "error", "message"),
    [
        ({"target_dim": 12}, KeyError, "requires"),
        ({"num_keypoints": 3}, KeyError, "requires"),
        ({"num_keypoints": 0, "target_dim": 12}, ValueError, "positive"),
        ({"num_keypoints": True, "target_dim": 12}, TypeError, "integer"),
        ({"num_keypoints": 2, "target_dim": 10}, ValueError, "divisible"),
        (
            {"num_keypoints": 3, "target_dim": 12, "token_dim": 4},
            ValueError,
            "disagrees",
        ),
        (
            {"num_keypoints": 3, "target_dim": 12, "token_order": "anchors_first"},
            ValueError,
            "token_order",
        ),
    ],
)
def test_layout_rejects_invalid_specs(spec, error, message) -> None:
    with pytest.raises(error, match=message):
        oracle_token_layout(spec)


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (torch.tensor(1.0), ValueError, "at least one dimension"),
        (torch.zeros(2, 11), ValueError, "last dimension"),
        (torch.zeros(0, 12), ValueError, "at least one target vector"),
        (torch.zeros(2, 12, dtype=torch.int64), TypeError, "floating dtype"),
        (
            torch.tensor([[float("nan")] + [0.0] * 11]),
            ValueError,
            "non-finite",
        ),
        (
            torch.tensor([[float("inf")] + [0.0] * 11]),
            ValueError,
            "non-finite",
        ),
    ],
)
def test_split_rejects_invalid_tensors(value, error, message) -> None:
    with pytest.raises(error, match=message):
        split_oracle_tokens(value, SPEC)


def test_squared_error_rejects_mismatched_shapes_and_non_tensor() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        oracle_token_squared_error(torch.zeros(2, 12), torch.zeros(1, 12), SPEC)
    with pytest.raises(TypeError, match="torch.Tensor"):
        split_oracle_tokens([[0.0] * 12], SPEC)


def test_merge_rejects_empty_incompatible_and_corrupt_statistics() -> None:
    with pytest.raises(ValueError, match="at least one"):
        merge_oracle_token_errors([])

    first = oracle_token_squared_error(torch.ones(1, 12), torch.zeros(1, 12), SPEC)
    other_spec = {"num_keypoints": 1, "target_dim": 12, "token_dim": 6}
    second = oracle_token_squared_error(
        torch.ones(1, 12), torch.zeros(1, 12), other_spec
    )
    with pytest.raises(ValueError, match="different layouts"):
        merge_oracle_token_errors([first, second])

    with pytest.raises(ValueError, match="wrong token count"):
        summarize_oracle_token_mse(replace(first, token_squared_error=(1.0,)))
    with pytest.raises(ValueError, match="finite and non-negative"):
        summarize_oracle_token_mse(
            replace(first, token_squared_error=(float("nan"), 0.0, 0.0, 0.0))
        )
    with pytest.raises(ValueError, match="prefix_count"):
        summarize_oracle_token_mse(replace(first, prefix_count=0))


def test_summary_rejects_manually_constructed_invalid_stats_type() -> None:
    with pytest.raises(TypeError, match="OracleTokenErrorStats"):
        summarize_oracle_token_mse({})

