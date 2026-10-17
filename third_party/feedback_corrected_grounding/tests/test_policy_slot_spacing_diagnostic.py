from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fcpg.policy_slot_spacing_diagnostic import (
    SLOT_SPACING_METHODS,
    SlotSpacingState,
    attach_slot_spacing_override,
    slot_spacing_override_tokens,
    update_slot_spacing_state,
)
import scripts.evaluate_policy_slot_spacing_oracle as evaluator


def _tokens(values: tuple[float, float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(1, 2, 1, 1).expand(-1, -1, 2, 4).clone()


def test_four_slot_semantics_are_exact() -> None:
    adjacent = _tokens((1.0, 2.0))
    previous = torch.full((1, 2, 4), -3.0)
    held = torch.full((1, 2, 4), -5.0)
    state = SlotSpacingState(previous_decision=previous, last_valid=held)

    exact = slot_spacing_override_tokens(adjacent, method="exact_adjacent", state=state)
    spaced = slot_spacing_override_tokens(adjacent, method="decision_spaced", state=state)
    repeated = slot_spacing_override_tokens(adjacent, method="repeat_current", state=state)
    hold = slot_spacing_override_tokens(adjacent, method="hold_last", state=state)

    torch.testing.assert_close(exact, adjacent)
    torch.testing.assert_close(spaced[:, 0], previous)
    torch.testing.assert_close(spaced[:, 1], adjacent[:, 1])
    torch.testing.assert_close(repeated[:, 0], adjacent[:, 1])
    torch.testing.assert_close(repeated[:, 1], adjacent[:, 1])
    torch.testing.assert_close(hold[:, 0], held)
    torch.testing.assert_close(hold[:, 1], held)


def test_active_state_updates_previous_but_freezes_last_valid() -> None:
    first = torch.full((1, 2, 4), 1.0)
    second = torch.full((1, 2, 4), 2.0)
    initialized = update_slot_spacing_state(SlotSpacingState(), first, active=False)
    active = update_slot_spacing_state(initialized, second, active=True)
    torch.testing.assert_close(active.previous_decision, second)
    torch.testing.assert_close(active.last_valid, first)


def test_override_preserves_corrupted_rgb_and_only_adds_grounded_fields() -> None:
    corrupted = {"rgb": torch.zeros(1, 2, 3, 8, 8), "obs": torch.ones(1, 2, 4)}
    adjacent = _tokens((1.0, 2.0))
    result = attach_slot_spacing_override(
        corrupted,
        adjacent,
        method="repeat_current",
        state=SlotSpacingState(),
    )
    assert result["rgb"] is corrupted["rgb"]
    assert result["obs"] is corrupted["obs"]
    assert bool(result["grounded_token_override_mask"].all())


def test_history_dependent_methods_fail_closed_without_predrop_state() -> None:
    adjacent = _tokens((1.0, 2.0))
    for method in ("decision_spaced", "hold_last"):
        with pytest.raises(RuntimeError, match="initialized pre-drop"):
            slot_spacing_override_tokens(
                adjacent,
                method=method,
                state=SlotSpacingState(),
            )


class _FakePolicy:
    num_keypoints = 2

    def _observation_tokens(self, batch: dict[str, torch.Tensor]):
        rgb = batch["rgb"]
        batch_size, history = rgb.shape[:2]
        scalar = rgb.mean(dim=(2, 3, 4))
        grounded = scalar[:, :, None, None].expand(-1, -1, 2, 4).clone()
        if "grounded_token_override" in batch:
            grounded = torch.where(
                batch["grounded_token_override_mask"].unsqueeze(-1),
                batch["grounded_token_override"],
                grounded,
            )
        zero = torch.zeros(batch_size, history, 1, 4)
        return torch.cat((zero, zero, grounded), dim=2), batch["q_nominal"], torch.ones(batch_size, history, 2)

    def predict_action_chunk(self, batch: dict[str, torch.Tensor]):
        memory, _, _ = self._observation_tokens(batch)
        # Make both history slots causally visible to the action.
        value = memory[:, :, 2:].mean(dim=(2, 3)) @ torch.tensor([0.25, 0.75])
        return SimpleNamespace(action=value[:, None, None].expand(-1, 8, 4).clone())


def _batch() -> dict[str, torch.Tensor]:
    return {
        "rgb": torch.stack((torch.full((3, 8, 8), 0.2), torch.full((3, 8, 8), 0.8)))[None],
        "obs": torch.zeros(1, 2, 4),
        "q_true": torch.tensor([[[[2.0, 2.0], [5.0, 5.0]]] * 2]),
        "q_nominal": torch.tensor([[[[2.0, 2.0], [5.0, 5.0]]] * 2]),
        "keypoints_xyz": torch.zeros(1, 2, 2, 3),
    }


@pytest.mark.parametrize("method", SLOT_SPACING_METHODS)
def test_active_selector_uses_corrupted_global_context_and_requested_slots(method: str) -> None:
    state = SlotSpacingState(
        previous_decision=torch.full((1, 2, 4), 0.3),
        last_valid=torch.full((1, 2, 4), 0.1),
    )
    result = evaluator.select_slot_spacing_action(
        _FakePolicy(),
        _batch(),
        method=method,
        active=True,
        state=state,
        occlusion_radius_px=1,
        occlusion_value=0.0,
        device=torch.device("cpu"),
    )
    assert result.action.shape == (1, 5, 4)
    assert result.clean_action.shape == result.action.shape
    assert result.clean_current_tokens.shape == (1, 2, 4)


def test_inactive_selector_is_exact_native_policy() -> None:
    result = evaluator.select_slot_spacing_action(
        _FakePolicy(),
        _batch(),
        method="hold_last",
        active=False,
        state=SlotSpacingState(),
        occlusion_radius_px=1,
        occlusion_value=0.0,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(result.action, result.clean_action)
