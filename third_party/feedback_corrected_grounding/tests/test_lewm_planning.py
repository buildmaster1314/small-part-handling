from __future__ import annotations

import torch

from fcpg.lewm_planning import GroundedCEMPlanner


class _DirectDynamics:
    def rollout_embeddings(
        self,
        history_embedding: torch.Tensor,
        history_action: torch.Tensor,
        future_action: torch.Tensor,
    ) -> torch.Tensor:
        del history_embedding, history_action
        return future_action[..., :1]

    def goal_cost(
        self,
        predicted_embedding: torch.Tensor,
        goal_embedding: torch.Tensor,
    ) -> torch.Tensor:
        goal = goal_embedding[:, None, None, :]
        return (predicted_embedding[:, :, -1:] - goal).square().sum(dim=(-1, -2))


def test_cem_optimizes_goal_cost_and_respects_bounds() -> None:
    planner = GroundedCEMPlanner(
        horizon=3,
        action_dim=2,
        num_samples=256,
        num_elites=32,
        iterations=4,
        momentum=0.1,
        min_std=0.01,
    )
    generator = torch.Generator().manual_seed(9)
    history_embedding = torch.zeros(2, 3, 1)
    history_action = torch.zeros(2, 2, 2)
    goal_embedding = torch.tensor([[0.75], [-0.5]])
    plan = planner.plan(
        _DirectDynamics(),
        history_embedding,
        history_action,
        goal_embedding,
        generator=generator,
    )
    assert plan.action_sequence.shape == (2, 3, 2)
    assert plan.cost.shape == (2,)
    assert bool(torch.all(plan.action_sequence <= 1.0))
    assert bool(torch.all(plan.action_sequence >= -1.0))
    assert abs(float(plan.action_sequence[0, -1, 0]) - 0.75) < 0.15
    assert abs(float(plan.action_sequence[1, -1, 0]) + 0.5) < 0.15

