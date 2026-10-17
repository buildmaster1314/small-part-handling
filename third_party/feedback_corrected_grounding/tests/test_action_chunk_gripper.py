from __future__ import annotations

import pytest
import torch

from fcpg.action_chunk_policy import (
    decode_gripper_action,
    encode_gripper_target,
)


def test_signed_gripper_encoding_maps_environment_extremes_to_bce_targets() -> None:
    action = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.equal(
        encode_gripper_target(action, "signed"),
        torch.tensor([0.0, 0.5, 1.0]),
    )


def test_signed_gripper_decoding_returns_environment_action_range() -> None:
    logit = torch.tensor([-100.0, 0.0, 100.0])
    decoded = decode_gripper_action(logit, "signed")
    assert torch.allclose(decoded, torch.tensor([-1.0, 0.0, 1.0]))


def test_legacy_unsigned_gripper_behavior_is_preserved() -> None:
    target = torch.tensor([0.0, 1.0])
    assert encode_gripper_target(target, "legacy_unsigned") is target
    assert torch.allclose(
        decode_gripper_action(torch.zeros(2), "legacy_unsigned"),
        torch.full((2,), 0.5),
    )


def test_signed_gripper_rejects_out_of_range_targets() -> None:
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        encode_gripper_target(torch.tensor([-1.1, 0.0]), "signed")
