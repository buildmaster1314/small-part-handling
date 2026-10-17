from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from fcpg.fk_deform_attn_baseline import VARIANTS
from scripts.train_fk_deform_attn_policy import run_paired_epoch


class _TinyPairMember(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[0.3], [-0.2]], dtype=torch.float32))

    def compute_bc_loss(self, batch, *, gripper_weight: float):
        del gripper_weight
        prediction = batch["obs"] @ self.weight
        target = batch["action"][:, 0, :1]
        per_row = (prediction - target).square().squeeze(-1)
        mask = batch["action_mask"][:, 0].float()
        loss = (per_row * mask).sum() / mask.sum().clamp_min(1.0)
        return loss, {"motion_loss": loss, "gripper_loss": loss * 0.0}, None


def _models() -> dict[str, _TinyPairMember]:
    return {variant: _TinyPairMember() for variant in VARIANTS}


def _batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    count = 8
    return {
        "obs": torch.randn(count, 2, generator=generator),
        "keypoints_xyz": torch.zeros(count, 1, 3),
        "q_nominal": torch.zeros(count, 1, 2),
        "action": torch.randn(count, 1, 1, generator=generator),
        "action_mask": torch.tensor(
            [[1], [1], [0], [1], [1], [0], [1], [1]], dtype=torch.bool
        ),
    }


def _run(microbatch_size: int):
    models = _models()
    optimizers = {
        variant: torch.optim.SGD(model.parameters(), lr=0.05)
        for variant, model in models.items()
    }
    metrics, steps = run_paired_epoch(
        models,
        [deepcopy(_batch())],
        torch.device("cpu"),
        optimizers=optimizers,
        schedulers=None,
        gripper_weight=0.0,
        grad_clip=100.0,
        max_batches=0,
        amp=False,
        microbatch_size=microbatch_size,
    )
    return models, metrics, steps


def test_microbatch_accumulation_preserves_effective_batch_update() -> None:
    full_models, full_metrics, full_steps = _run(8)
    micro_models, micro_metrics, micro_steps = _run(2)

    assert full_steps == micro_steps == 1
    for variant in VARIANTS:
        torch.testing.assert_close(
            micro_models[variant].weight,
            full_models[variant].weight,
            rtol=1e-6,
            atol=1e-7,
        )
        assert abs(micro_metrics[variant]["loss"] - full_metrics[variant]["loss"]) < 1e-7
        assert abs(micro_metrics[variant]["motion"] - full_metrics[variant]["motion"]) < 1e-7


def test_paired_members_keep_identical_updates_with_accumulation() -> None:
    models, _, _ = _run(2)
    torch.testing.assert_close(
        models[VARIANTS[0]].weight,
        models[VARIANTS[1]].weight,
        rtol=0.0,
        atol=0.0,
    )
