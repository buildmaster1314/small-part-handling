from __future__ import annotations

import pytest
import torch

from fcpg.so101_v9 import (
    assert_v9_backbone_matches,
    compose_v9_state_dict,
    should_use_memory,
)


def _states() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    local = {
        "base.encoder.weight": torch.tensor([[1.0, 2.0]]),
        "temporal.weight": torch.tensor([[7.0, 8.0]]),
    }
    memory = {
        "backbone.base.encoder.weight": local["base.encoder.weight"].clone(),
        "backbone.temporal.weight": torch.tensor([[3.0, 4.0]]),
        "memory_head.weight": torch.tensor([[5.0, 6.0]]),
    }
    return memory, local


def test_v9_replaces_only_the_compatible_backbone() -> None:
    memory, local = _states()

    composed = compose_v9_state_dict(memory, local)

    torch.testing.assert_close(
        composed["backbone.temporal.weight"], local["temporal.weight"]
    )
    torch.testing.assert_close(
        composed["memory_head.weight"], memory["memory_head.weight"]
    )
    assert_v9_backbone_matches(composed, local)


def test_v9_rejects_a_different_visual_base() -> None:
    memory, local = _states()
    local["base.encoder.weight"] = torch.tensor([[9.0, 2.0]])

    with pytest.raises(ValueError, match="visual base differs"):
        compose_v9_state_dict(memory, local)


@pytest.mark.parametrize(
    ("policy", "reliable", "expected"),
    [
        ("off", False, False),
        ("off", True, False),
        ("auto", True, False),
        ("auto", False, True),
        ("always", True, True),
        ("always", False, True),
    ],
)
def test_v9_memory_policy_preserves_clean_parity(
    policy: str, reliable: bool, expected: bool
) -> None:
    assert (
        should_use_memory(
            policy,
            model_available=True,
            memory_available=True,
            measurement_reliable=reliable,
        )
        is expected
    )


def test_v9_memory_requires_both_model_and_history() -> None:
    assert not should_use_memory(
        "always",
        model_available=False,
        memory_available=True,
        measurement_reliable=False,
    )
    assert not should_use_memory(
        "always",
        model_available=True,
        memory_available=False,
        measurement_reliable=False,
    )
