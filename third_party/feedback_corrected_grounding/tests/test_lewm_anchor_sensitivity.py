from __future__ import annotations

import math

import pytest
import torch

from fcpg.lewm_anchor_sensitivity import (
    AnchorSensitivity,
    batch_permutation_indices,
    compute_anchor_sensitivity,
    shuffle_token_group,
    summarize_anchor_sensitivity,
)


def test_anchor_sensitivity_reports_per_sample_and_summary_values() -> None:
    nominal_anchor = torch.zeros(2, 2, 3, dtype=torch.float64)
    corrected_anchor = torch.tensor(
        [
            [[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]],
            [[1.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    nominal_cls = torch.zeros(2, 3, dtype=torch.float64)
    corrected_cls = torch.tensor(
        [[1.5, 2.0, 0.0], [1.0, 2.0, 0.0]], dtype=torch.float64
    )

    result = compute_anchor_sensitivity(
        nominal_anchor,
        corrected_anchor,
        nominal_cls,
        corrected_cls,
    )

    torch.testing.assert_close(
        result.pre_anchor_rms,
        torch.tensor([5.0, math.sqrt(5.0)], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.post_cls_l2,
        torch.tensor([2.5, math.sqrt(5.0)], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.attenuation_ratio,
        torch.tensor([0.5, 1.0], dtype=torch.float64),
    )
    summary = summarize_anchor_sensitivity(result)
    assert summary.num_samples == 2
    assert summary.pre_anchor_rms_mean == pytest.approx((5.0 + math.sqrt(5.0)) / 2.0)
    assert summary.pre_anchor_rms_median == pytest.approx(math.sqrt(5.0))
    assert summary.post_cls_l2_mean == pytest.approx((2.5 + math.sqrt(5.0)) / 2.0)
    assert summary.attenuation_ratio_mean == pytest.approx(0.75)
    assert summary.attenuation_ratio_median == pytest.approx(0.5)


def test_anchor_sensitivity_uses_eps_for_zero_anchor_change() -> None:
    result = compute_anchor_sensitivity(
        torch.zeros(1, 2, 2),
        torch.zeros(1, 2, 2),
        torch.zeros(1, 2),
        torch.tensor([[3.0, 4.0]]),
        eps=0.5,
    )

    torch.testing.assert_close(result.pre_anchor_rms, torch.tensor([0.0]))
    torch.testing.assert_close(result.post_cls_l2, torch.tensor([5.0]))
    torch.testing.assert_close(result.attenuation_ratio, torch.tensor([10.0]))


@pytest.mark.parametrize("eps", [0.0, -1.0, float("inf"), float("nan")])
def test_anchor_sensitivity_rejects_invalid_eps(eps: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        compute_anchor_sensitivity(
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 2),
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            eps=eps,
        )


def test_anchor_sensitivity_rejects_shape_dtype_and_nonfinite_inputs() -> None:
    anchor = torch.zeros(2, 3, 4)
    cls = torch.zeros(2, 4)
    with pytest.raises(ValueError, match="identical shapes"):
        compute_anchor_sensitivity(anchor, anchor[:, :2], cls, cls)
    with pytest.raises(ValueError, match=r"\[B, D\]"):
        compute_anchor_sensitivity(anchor, anchor, cls[:, :3], cls[:, :3])
    with pytest.raises(TypeError, match="floating-point"):
        compute_anchor_sensitivity(anchor.long(), anchor.long(), cls, cls)
    invalid = anchor.clone()
    invalid[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        compute_anchor_sensitivity(invalid, anchor, cls, cls)


def test_summary_rejects_mismatched_or_empty_values() -> None:
    with pytest.raises(ValueError, match="same length"):
        summarize_anchor_sensitivity(
            AnchorSensitivity(
                pre_anchor_rms=torch.ones(2),
                post_cls_l2=torch.ones(1),
                attenuation_ratio=torch.ones(2),
            )
        )
    with pytest.raises(ValueError, match="non-empty"):
        summarize_anchor_sensitivity(
            AnchorSensitivity(
                pre_anchor_rms=torch.empty(0),
                post_cls_l2=torch.empty(0),
                attenuation_ratio=torch.empty(0),
            )
        )


def test_batch_permutation_is_reproducible_valid_and_deranged() -> None:
    first = batch_permutation_indices(12, seed=19)
    second = batch_permutation_indices(12, seed=19)

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(torch.sort(first).values, torch.arange(12))
    assert not bool(torch.any(first == torch.arange(12)))
    assert not torch.equal(first, batch_permutation_indices(12, seed=20))


@pytest.mark.parametrize("batch_size", [0, 1])
def test_batch_permutation_rejects_unshufflable_batches(batch_size: int) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        batch_permutation_indices(batch_size, seed=0)


@pytest.mark.parametrize(
    ("group", "selected"),
    [
        ("summary", (0,)),
        ("global", (1,)),
        ("robot", (2,)),
        ("all_anchors", (3, 4, 5)),
    ],
)
def test_shuffle_token_group_changes_only_selected_tokens(
    group: str,
    selected: tuple[int, ...],
) -> None:
    tokens = torch.arange(4 * 6 * 2, dtype=torch.float32).reshape(4, 6, 2)
    original = tokens.clone()
    permutation = torch.tensor([1, 2, 3, 0])

    shuffled = shuffle_token_group(tokens, permutation, group=group)

    torch.testing.assert_close(tokens, original)
    for index in range(tokens.shape[1]):
        expected = tokens.index_select(0, permutation)[:, index]
        if index in selected:
            torch.testing.assert_close(shuffled[:, index], expected)
        else:
            torch.testing.assert_close(shuffled[:, index], tokens[:, index])
    assert shuffled.data_ptr() != tokens.data_ptr()


def test_shuffle_single_anchor_uses_anchor_relative_index() -> None:
    tokens = torch.arange(3 * 5, dtype=torch.float32).reshape(3, 5, 1)
    permutation = torch.tensor([2, 0, 1])

    shuffled = shuffle_token_group(
        tokens,
        permutation,
        group="single_anchor",
        anchor_index=1,
    )

    torch.testing.assert_close(shuffled[:, 4], tokens.index_select(0, permutation)[:, 4])
    torch.testing.assert_close(shuffled[:, :4], tokens[:, :4])


def test_shuffle_token_group_rejects_invalid_contracts() -> None:
    tokens = torch.zeros(3, 5, 2)
    with pytest.raises(ValueError, match="exactly once"):
        shuffle_token_group(tokens, torch.tensor([0, 0, 2]), group="global")
    with pytest.raises(ValueError, match="group must be one of"):
        shuffle_token_group(tokens, torch.tensor([1, 2, 0]), group="unknown")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="anchor_index"):
        shuffle_token_group(
            tokens,
            torch.tensor([1, 2, 0]),
            group="single_anchor",
        )
    with pytest.raises(ValueError, match="only valid"):
        shuffle_token_group(
            tokens,
            torch.tensor([1, 2, 0]),
            group="robot",
            anchor_index=0,
        )
    invalid = tokens.clone()
    invalid[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        shuffle_token_group(invalid, torch.tensor([1, 2, 0]), group="global")
