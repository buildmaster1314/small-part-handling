from __future__ import annotations

import pytest
import torch

from fcpg.lewm_known_drop_policy import (
    attach_grounded_override,
    drop_is_active,
    grounded_tokens_from_memory,
    occlude_rgb_at_keypoints,
    repeated_last_valid_tokens,
)


def test_drop_interval_is_half_open() -> None:
    assert not drop_is_active(1, start=2, duration=3)
    assert drop_is_active(2, start=2, duration=3)
    assert drop_is_active(4, start=2, duration=3)
    assert not drop_is_active(5, start=2, duration=3)
    assert not drop_is_active(2, start=2, duration=0)


def test_occlusion_is_local_and_does_not_mutate_source() -> None:
    rgb = torch.ones(1, 2, 3, 8, 10)
    points = torch.tensor([[[[3.0, 4.0]], [[7.0, 1.0]]]])
    source = {"rgb": rgb, "q_true": points, "obs": torch.ones(1)}

    result = occlude_rgb_at_keypoints(source, radius_px=1, value=0.25)

    assert result["rgb"] is not rgb
    assert torch.all(rgb == 1)
    assert torch.all(result["rgb"][0, 0, :, 3:6, 2:5] == 0.25)
    assert torch.all(result["rgb"][0, 1, :, 0:3, 6:9] == 0.25)
    assert result["obs"] is source["obs"]


def test_occlusion_rejects_malformed_keypoint_rank() -> None:
    source = {
        "rgb": torch.ones(1, 2, 3, 8, 10),
        "q_true": torch.ones(1, 2, 2),
    }
    with pytest.raises(ValueError, match=r"q_true must have shape \[B,H,K,2\]"):
        occlude_rgb_at_keypoints(source, radius_px=1)


def test_grounded_block_and_override_helpers() -> None:
    memory = torch.arange(1 * 2 * 5 * 4, dtype=torch.float32).reshape(1, 2, 5, 4)
    grounded = grounded_tokens_from_memory(memory, num_keypoints=3)
    assert torch.equal(grounded, memory[:, :, 2:, :])

    last = grounded[:, 0]
    repeated = repeated_last_valid_tokens(last, history_steps=2)
    assert repeated.shape == grounded.shape
    assert torch.equal(repeated[:, 0], last)
    assert torch.equal(repeated[:, 1], last)

    batch = attach_grounded_override({"obs": torch.ones(1)}, repeated)
    assert batch["grounded_token_override"] is repeated
    assert bool(batch["grounded_token_override_mask"].all())


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"decision_index": -1, "start": 0, "duration": 1}, "decision_index"),
        ({"decision_index": 0, "start": -1, "duration": 1}, "start"),
        ({"decision_index": 0, "start": 0, "duration": -1}, "duration"),
    ],
)
def test_invalid_drop_schedule_is_rejected(kwargs: dict[str, int], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        drop_is_active(**kwargs)
