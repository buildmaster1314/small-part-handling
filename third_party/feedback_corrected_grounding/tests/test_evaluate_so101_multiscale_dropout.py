from __future__ import annotations

import pytest
import torch

from fcpg.so101_multiscale_memory import (
    freeze_recent_context,
    random_freeze_recent_context,
    random_freeze_recent_context_with_validity,
)
from scripts.evaluate_so101_multiscale_dropout import _balanced_eval_indices


class _Cache:
    def __init__(self) -> None:
        self.episodes = torch.tensor([0] * 5 + [1] * 5).numpy()
        self.frame_indices = torch.tensor(list(range(5)) * 2).numpy()


class _Dataset:
    def __init__(self) -> None:
        self.caches = [_Cache()]
        self.samples = [(0, index) for index in range(10)]
        self.group_ids = [0] * 5 + [1] * 5
        self.split_seed = 17


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    pixels = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1, 1)
    memory = torch.tensor([10.0, 11.0]).reshape(1, 2, 1, 1, 1)
    return pixels, memory


@pytest.mark.parametrize(
    ("suffix", "expected"),
    (
        (0, [0.0, 1.0, 2.0, 3.0]),
        (1, [0.0, 1.0, 1.0, 3.0]),
        (2, [0.0, 0.0, 0.0, 3.0]),
        (3, [10.0, 10.0, 10.0, 3.0]),
    ),
)
def test_freeze_recent_context_is_causal_and_keeps_target(
    suffix: int, expected: list[float]
) -> None:
    pixels, memory = _inputs()
    frozen = freeze_recent_context(pixels, memory, suffix)
    assert frozen.flatten().tolist() == expected
    assert pixels.flatten().tolist() == [0.0, 1.0, 2.0, 3.0]


@pytest.mark.parametrize("suffix", (-1, 4))
def test_freeze_recent_context_rejects_invalid_suffix(suffix: int) -> None:
    pixels, memory = _inputs()
    with pytest.raises(ValueError, match="suffix_frames"):
        freeze_recent_context(pixels, memory, suffix)


def test_freeze_recent_context_requires_aligned_images() -> None:
    pixels, _ = _inputs()
    with pytest.raises(ValueError, match="share batch/image dimensions"):
        freeze_recent_context(pixels, torch.zeros(2, 2, 1, 1, 1), 1)


def test_random_freeze_zero_probability_is_identity() -> None:
    pixels, memory = _inputs()
    assert random_freeze_recent_context(pixels, memory, 0.0) is pixels


def test_random_freeze_preserves_target_and_uses_causal_sources() -> None:
    pixels = torch.arange(16, dtype=torch.float32).reshape(4, 4, 1, 1, 1)
    memory = (100 + torch.arange(8, dtype=torch.float32)).reshape(4, 2, 1, 1, 1)
    torch.manual_seed(19)
    frozen = random_freeze_recent_context(pixels, memory, 1.0)
    assert torch.equal(frozen[:, -1], pixels[:, -1])
    assert not torch.equal(frozen[:, :-1], pixels[:, :-1])
    for row in range(len(pixels)):
        assert frozen[row].flatten().tolist() in (
            freeze_recent_context(pixels[row : row + 1], memory[row : row + 1], 1)
            .flatten()
            .tolist(),
            freeze_recent_context(pixels[row : row + 1], memory[row : row + 1], 2)
            .flatten()
            .tolist(),
            freeze_recent_context(pixels[row : row + 1], memory[row : row + 1], 3)
            .flatten()
            .tolist(),
        )


def test_random_freeze_returns_matching_validity_mask() -> None:
    pixels = torch.arange(16, dtype=torch.float32).reshape(4, 4, 1, 1, 1)
    memory = (100 + torch.arange(8, dtype=torch.float32)).reshape(4, 2, 1, 1, 1)
    torch.manual_seed(23)
    frozen, valid = random_freeze_recent_context_with_validity(
        pixels, memory, 1.0
    )
    assert valid.shape == (4, 3)
    assert not bool(valid.all())
    for row in range(len(pixels)):
        first_invalid = int(torch.nonzero(~valid[row], as_tuple=False)[0])
        assert bool(valid[row, :first_invalid].all())
        assert not bool(valid[row, first_invalid:].any())
        assert torch.equal(
            frozen[row : row + 1],
            freeze_recent_context(
                pixels[row : row + 1], memory[row : row + 1], 3 - first_invalid
            ),
        )


@pytest.mark.parametrize("probability", (-0.1, 1.1))
def test_random_freeze_rejects_invalid_probability(probability: float) -> None:
    pixels, memory = _inputs()
    with pytest.raises(ValueError, match="probability"):
        random_freeze_recent_context(pixels, memory, probability)


def test_balanced_eval_indices_applies_history_cutoff_before_balancing() -> None:
    dataset = _Dataset()
    indices = _balanced_eval_indices(dataset, limit=None, minimum_past_frames=3)
    assert set(indices) == {3, 4, 8, 9}
    assert [dataset.group_ids[index] for index in indices] == [0, 1, 0, 1]
