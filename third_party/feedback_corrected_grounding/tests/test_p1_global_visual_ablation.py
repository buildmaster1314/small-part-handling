from __future__ import annotations

from scripts.evaluate_p1_native_policy_rollout import (
    _runtime_method,
    _uses_trusted_global_visual,
)


def test_trusted_global_ablation_reuses_factual_runtime_only() -> None:
    method = "complementary_factual_trusted_global_s15801"
    assert _runtime_method(method) == "complementary_factual"
    assert _uses_trusted_global_visual(method)
    assert not _uses_trusted_global_visual("complementary_factual_s15801")
