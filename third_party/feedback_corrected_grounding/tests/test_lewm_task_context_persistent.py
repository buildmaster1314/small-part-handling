from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from fcpg.lewm_anchor_persistent import AnchorPersistentLeWM
from fcpg.lewm_grounded_model import GroundedFrameEncoder
from fcpg.lewm_task_context_persistent import (
    TASK_CONTEXT_PERSISTENT_CHECKPOINT_TYPE,
    TASK_CONTEXT_PERSISTENT_SCHEMA_VERSION,
    TASK_CONTEXT_PERSISTENT_SIGREG_WEIGHT,
    TASK_CONTEXT_PERSISTENT_STATE_SLOTS,
    TASK_CONTEXT_PERSISTENT_VARIANT,
    TaskContextPersistentLeWM,
)
from third_party.lewm.module import ARPredictor, Embedder, MLP, SIGReg


@dataclass(frozen=True)
class _VisionOutput:
    feature_map: torch.Tensor
    global_feature: torch.Tensor


class _TinyVision(nn.Module):
    output_dim = 8

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, self.output_dim, kernel_size=1)

    def forward(self, pixels: torch.Tensor) -> _VisionOutput:
        feature_map = self.projection(pixels)
        return _VisionOutput(
            feature_map=feature_map,
            global_feature=feature_map.mean(dim=(-1, -2)),
        )


class _RecordingSIGReg(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[torch.Tensor] = []

    def forward(self, projection: torch.Tensor) -> torch.Tensor:
        self.calls.append(projection.detach().clone())
        return projection.square().mean()


def _frame_encoder(num_keypoints: int = 3) -> GroundedFrameEncoder:
    return GroundedFrameEncoder(
        vision_encoder=_TinyVision(),
        visual_dim=8,
        obs_dim=4,
        num_keypoints=num_keypoints,
        latent_dim=16,
        depth=1,
        heads=4,
        dropout=0.0,
    )


def _model(*, num_tasks: int = 4) -> TaskContextPersistentLeWM:
    return TaskContextPersistentLeWM(
        frame_encoder=_frame_encoder(),
        action_dim=4,
        num_tasks=num_tasks,
        history_size=3,
        predictor_depth=1,
        predictor_heads=4,
        predictor_mlp_dim=32,
        predictor_dim_head=4,
        predictor_dropout=0.0,
        projector_hidden_dim=32,
        sigreg_knots=5,
        sigreg_num_proj=8,
    )


def _old_model() -> AnchorPersistentLeWM:
    return AnchorPersistentLeWM(
        frame_encoder=_frame_encoder(),
        action_dim=4,
        history_size=3,
        predictor_depth=1,
        predictor_heads=4,
        predictor_mlp_dim=32,
        predictor_dim_head=4,
        predictor_dropout=0.0,
        projector_hidden_dim=32,
        sigreg_knots=5,
        sigreg_num_proj=8,
    )


def _batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    frames = 4
    q_nominal = torch.rand(batch_size, frames, 3, 2) * 15.0
    return {
        "pixels": torch.rand(batch_size, frames, 3, 16, 16),
        "proprio": torch.randn(batch_size, frames, 4),
        "q_nominal": q_nominal,
        "q_corrected": q_nominal + 0.5 * torch.randn_like(q_nominal),
        "keypoints_xyz": torch.randn(batch_size, frames, 3, 3),
        "valid": torch.ones(batch_size, frames, 3, dtype=torch.bool),
        "action": torch.randn(batch_size, frames - 1, 4),
        "task_index": torch.arange(batch_size, dtype=torch.long) % 4,
        "source_index": torch.arange(batch_size, dtype=torch.long) // 2,
    }


def test_model_contract_has_distinct_keypoint_and_state_slot_counts() -> None:
    model = _model()

    assert model.checkpoint_type == TASK_CONTEXT_PERSISTENT_CHECKPOINT_TYPE
    assert model.model_variant == TASK_CONTEXT_PERSISTENT_VARIANT
    assert model.num_keypoints == 3
    assert model.num_state_slots == 4
    assert model.state_slot_names == TASK_CONTEXT_PERSISTENT_STATE_SLOTS == (
        "visual_summary",
        "anchor0",
        "anchor1",
        "anchor2",
    )
    assert isinstance(model.predictor, ARPredictor)
    assert isinstance(model.action_encoder, Embedder)
    assert isinstance(model.projector, MLP)
    assert isinstance(model.pred_proj, MLP)
    assert isinstance(model.sigreg, SIGReg)
    assert model.sigreg_weight == TASK_CONTEXT_PERSISTENT_SIGREG_WEIGHT == 0.02

    buffers = dict(model.named_buffers())
    assert int(buffers["state_schema_version"].item()) == (
        TASK_CONTEXT_PERSISTENT_SCHEMA_VERSION
    )
    assert "state_schema_version" in model.state_dict()

    with pytest.raises(ValueError, match="requires num_keypoints=3"):
        TaskContextPersistentLeWM(
            frame_encoder=_frame_encoder(num_keypoints=2),
            action_dim=4,
            num_tasks=4,
        )
    with pytest.raises(TypeError, match="num_tasks must be an integer"):
        TaskContextPersistentLeWM(
            frame_encoder=_frame_encoder(),
            action_dim=4,
            num_tasks=4.0,  # type: ignore[arg-type]
        )


def test_state_slots_are_exactly_summary_plus_last_three_anchor_tokens() -> None:
    model = _model()
    tokens = torch.arange(2 * 6 * 16, dtype=torch.float32).reshape(2, 6, 16)

    selected = model.select_state_tokens(tokens)

    expected = torch.cat([tokens[:, 0:1], tokens[:, -3:]], dim=1)
    torch.testing.assert_close(selected, expected, rtol=0.0, atol=0.0)
    assert not torch.equal(selected[:, 1], tokens[:, 1])
    with pytest.raises(ValueError, match="tokens must have shape"):
        model.select_state_tokens(tokens[:, :-1])


def test_task_embedding_only_enters_predictor_condition_stream() -> None:
    torch.manual_seed(101)
    model = _model().eval()
    embedding = torch.randn(2, 3, 4, 16)
    action = torch.zeros(2, 3, 4)
    task_index = torch.tensor([0, 1])
    seen: dict[str, torch.Tensor] = {}

    def capture_predictor(
        _module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        seen["state"] = inputs[0].detach().clone()
        seen["condition"] = inputs[1].detach().clone()

    handle = model.predictor.register_forward_pre_hook(capture_predictor)
    with torch.no_grad():
        predicted = model.predict(embedding, action, task_index)
    handle.remove()

    expected_state = embedding.permute(0, 2, 1, 3).reshape(8, 3, 16)
    torch.testing.assert_close(seen["state"], expected_state, rtol=0.0, atol=0.0)
    assert seen["condition"].shape == (8, 3, 16)
    expected_task_delta = model.task_embedding.weight[0] - model.task_embedding.weight[1]
    for slot in range(model.num_state_slots):
        torch.testing.assert_close(
            seen["condition"][slot] - seen["condition"][4 + slot],
            expected_task_delta.expand(3, -1),
        )
    assert predicted.shape == embedding.shape


def test_forward_uses_weighted_slot_loss_and_independent_slot_sigreg() -> None:
    torch.manual_seed(103)
    model = _model().eval()
    recording_sigreg = _RecordingSIGReg()
    model.sigreg = recording_sigreg

    output = model(_batch(), sigreg_weight=0.02)

    assert output.frame_latent.shape == (4, 4, 4, 16)
    assert output.embedding.shape == (4, 4, 4, 16)
    assert output.predicted_embedding.shape == (4, 3, 4, 16)
    assert output.target_embedding.shape == (4, 3, 4, 16)
    expected_prediction = 0.25 * output.context_loss + 0.75 * output.anchor_loss
    torch.testing.assert_close(output.prediction_loss, expected_prediction)
    torch.testing.assert_close(
        output.loss,
        output.prediction_loss + 0.02 * output.sigreg_loss,
    )

    assert len(recording_sigreg.calls) == model.num_state_slots == 4
    for slot, projection in enumerate(recording_sigreg.calls):
        assert projection.shape == (4, 4, 16)
        torch.testing.assert_close(
            projection,
            output.embedding[:, :, slot].transpose(0, 1),
        )
    expected_sigreg = torch.stack(
        [output.embedding[:, :, slot].square().mean() for slot in range(4)]
    ).mean()
    torch.testing.assert_close(output.sigreg_loss, expected_sigreg)

    with pytest.raises(ValueError, match="sigreg_weight exactly 0.02"):
        model(_batch(), sigreg_weight=0.0)
    with pytest.raises(ValueError, match="sigreg_weight exactly 0.02"):
        model(_batch(), sigreg_weight=0.09)


def test_per_source_sigreg_still_balances_each_slot_independently() -> None:
    model = _model()
    recording_sigreg = _RecordingSIGReg()
    model.sigreg = recording_sigreg
    embedding = torch.randn(4, 3, 4, 16)
    source_index = torch.tensor([0, 0, 1, 1])

    loss = model._sigreg_loss(embedding, source_index)

    assert torch.isfinite(loss)
    assert len(recording_sigreg.calls) == 8
    assert all(call.shape == (3, 2, 16) for call in recording_sigreg.calls)
    for slot in range(4):
        torch.testing.assert_close(
            recording_sigreg.calls[2 * slot],
            embedding[:2, :, slot].transpose(0, 1),
        )
        torch.testing.assert_close(
            recording_sigreg.calls[2 * slot + 1],
            embedding[2:, :, slot].transpose(0, 1),
        )


def test_oracle_target_poison_cannot_enter_state_or_prediction() -> None:
    torch.manual_seed(107)
    model = _model().eval()
    batch = _batch()
    baseline = model(batch)
    oracle_target = torch.full_like(baseline.embedding, 1.0e6, requires_grad=True)
    poisoned_batch = {**batch, "oracle_target": oracle_target}

    poisoned = model(poisoned_batch)

    torch.testing.assert_close(
        poisoned.embedding,
        baseline.embedding,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        poisoned.predicted_embedding,
        baseline.predicted_embedding,
        rtol=0.0,
        atol=0.0,
    )
    assert poisoned.state_target_embedding is poisoned.embedding
    torch.testing.assert_close(
        poisoned.target_embedding,
        poisoned.embedding[:, 1:],
        rtol=0.0,
        atol=0.0,
    )
    poisoned.loss.backward()
    assert oracle_target.grad is None


def test_gradients_reach_summary_anchors_predictor_and_task_conditioning() -> None:
    torch.manual_seed(109)
    model = _model().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = _batch()

    output = model(batch)
    output.loss.backward()
    assert model.frame_encoder.summary_token.grad is not None
    assert bool(model.frame_encoder.summary_token.grad.abs().sum() > 0)
    assert model.frame_encoder.token_type.grad is not None
    assert bool(model.frame_encoder.token_type.grad[:, -3:].abs().sum() > 0)
    assert model.predictor.pos_embedding.grad is not None
    assert bool(model.predictor.pos_embedding.grad.abs().sum() > 0)
    assert model.task_embedding.weight.grad is not None

    # Upstream ARPredictor initializes AdaLN modulation to zero.  Its first
    # update opens the condition path; the next backward must reach task rows.
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second_output = model(batch)
    second_output.loss.backward()
    assert model.task_embedding.weight.grad is not None
    assert bool(model.task_embedding.weight.grad.abs().sum() > 0)


@pytest.mark.parametrize(
    ("task_index", "error", "message"),
    [
        (torch.tensor([[0], [1]]), ValueError, "task_index must have shape"),
        (torch.tensor([0.0, 1.0]), TypeError, "integer dtype"),
        (torch.tensor([-1, 0]), ValueError, "values must be in"),
        (torch.tensor([0, 4]), ValueError, "values must be in"),
    ],
)
def test_predict_rejects_invalid_task_shape_dtype_and_range(
    task_index: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    model = _model().eval()
    embedding = torch.randn(2, 3, 4, 16)
    action = torch.randn(2, 3, 4)

    with pytest.raises(error, match=message):
        model.predict(embedding, action, task_index)


def test_prediction_and_training_reject_wrong_state_or_missing_task_shapes() -> None:
    model = _model().eval()
    with pytest.raises(ValueError, match="embedding must have shape"):
        model.predict(
            torch.randn(2, 3, 3, 16),
            torch.randn(2, 3, 4),
            torch.tensor([0, 1]),
        )
    with pytest.raises(ValueError, match="action must have shape"):
        model.predict(
            torch.randn(2, 3, 4, 16),
            torch.randn(2, 2, 4),
            torch.tensor([0, 1]),
        )
    batch = _batch()
    del batch["task_index"]
    with pytest.raises(KeyError, match="task_index"):
        model(batch)


def test_recursive_rollout_preserves_candidates_time_slots_and_task_index() -> None:
    torch.manual_seed(113)
    model = _model().eval()
    seen_tasks: list[torch.Tensor] = []

    def capture_task(
        _module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        seen_tasks.append(inputs[0].detach().clone())

    handle = model.task_embedding.register_forward_pre_hook(capture_task)
    with torch.no_grad():
        rollout = model.rollout_state_embeddings(
            history_embedding=torch.randn(2, 3, 4, 16),
            history_action=torch.randn(2, 2, 4),
            future_action=torch.randn(2, 5, 4, 4),
            task_index=torch.tensor([1, 3]),
        )
    handle.remove()

    assert rollout.shape == (2, 5, 4, 4, 16)
    assert len(seen_tasks) == 4
    expected = torch.tensor([1, 1, 1, 1, 1, 3, 3, 3, 3, 3])
    for task_call in seen_tasks:
        torch.testing.assert_close(task_call, expected)

    with torch.no_grad():
        empty = model.rollout_embeddings(
            history_embedding=torch.randn(2, 3, 4, 16),
            history_action=torch.randn(2, 2, 4),
            future_action=torch.randn(2, 5, 0, 4),
            task_index=torch.tensor([1, 3]),
        )
    assert empty.shape == (2, 5, 0, 4, 16)

    with pytest.raises(ValueError, match="values must be in"):
        model.rollout_state_embeddings(
            history_embedding=torch.randn(2, 3, 4, 16),
            history_action=torch.randn(2, 2, 4),
            future_action=torch.randn(2, 5, 1, 4),
            task_index=torch.tensor([1, 4]),
        )


def test_old_and_new_state_dicts_are_strictly_incompatible_both_directions() -> None:
    old = _old_model()
    new = _model()

    with pytest.raises(RuntimeError) as old_into_new:
        new.load_state_dict(old.state_dict(), strict=True)
    assert "task_embedding.weight" in str(old_into_new.value)
    assert "state_schema_version" in str(old_into_new.value)

    with pytest.raises(RuntimeError) as new_into_old:
        old.load_state_dict(new.state_dict(), strict=True)
    assert "task_embedding.weight" in str(new_into_old.value)
    assert "state_schema_version" in str(new_into_old.value)


def test_variant_rejects_oracle_target_mode() -> None:
    with pytest.raises(ValueError, match="online targets only"):
        TaskContextPersistentLeWM(
            frame_encoder=_frame_encoder(),
            action_dim=4,
            num_tasks=4,
            target_mode="frozen_oracle",
            alignment_weight=1.0,
        )
