from __future__ import annotations

import pytest

from scripts.continue_lewm_world_action_joint_a4 import (
    WORLD_CONTINUATION_PROTOCOL,
    _cuda0,
    _parser,
)


def test_world_action_continuation_contract_is_fixed() -> None:
    args = _parser(["--output-dir", "/tmp/world-action-control"])

    assert args.steps == 2_000
    assert args.seed == 40_571
    assert args.batch_size == 84
    assert not hasattr(args, "sigreg_weight")
    assert WORLD_CONTINUATION_PROTOCOL.endswith("_v1")


def test_world_action_continuation_rejects_cuda1() -> None:
    with pytest.raises(ValueError, match="CUDA0"):
        _cuda0("cuda:1")
