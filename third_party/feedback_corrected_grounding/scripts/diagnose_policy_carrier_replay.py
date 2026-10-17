#!/usr/bin/env python3
"""Compare the legacy and compact carrier extractors on one frozen replay plan.

This is a local forensic tool.  It writes only aggregate differences, never
the hidden clean token arrays, so it cannot create a deployment artifact.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch

from fcpg.lewm_action_branch_collection import make_metaworld_episode_runtime
from fcpg.lewm_p1_native_complementary_evaluation import validate_and_load_p1_policy
from fcpg.lewm_p1_native_fresh_collection import (
    FreshnessInventory,
    build_fresh_episode_plans,
    collect_fresh_p1_native_episode,
    legacy_observed_state_signatures,
    _policy_pair_batch,
)
from fcpg.lewm_known_drop_policy_evaluator import extract_policy_anchor_observation
from fcpg.policy_token_carrier_collection import (
    collect_policy_carrier_episode,
    extract_compact_policy_pairs,
)
from fcpg.lewm_counterfactual_runtime import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/lewm/eval/p1_native_complementary_fresh.json"


def _inventory(config: dict[str, object]) -> FreshnessInventory:
    freshness = config["freshness"]
    assert isinstance(freshness, dict)
    source = ROOT / str(freshness["legacy_observation_source"])
    if sha256_file(source) != str(freshness["legacy_observation_source_sha256"]):
        raise RuntimeError("legacy source SHA-256 mismatch")
    return FreshnessInventory(
        reset_digest_scopes={},
        observed_signature_scopes={
            "p1_fcpg_training": legacy_observed_state_signatures(source)
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    p1 = config["p1"]
    assert isinstance(p1, dict)
    policy = validate_and_load_p1_policy(config, repo_root=ROOT, device=args.device)
    fresh = config["fresh_data"]
    assert isinstance(fresh, dict)
    runtime_factory = partial(
        make_metaworld_episode_runtime,
        camera=str(fresh["camera"]),
        width=int(fresh["width"]),
        height=int(fresh["height"]),
        benchmark_seed=int(fresh["benchmark_seed"]),
    )
    plan = build_fresh_episode_plans(config)[0]
    old = collect_fresh_p1_native_episode(
        plan,
        inventory=_inventory(config),
        policy=policy,
        device=args.device,
        runtime_factory=runtime_factory,
    )
    extractor_diffs: list[np.ndarray] = []

    def compare_extractors(policy, pairs, *, device, **kwargs):
        batch = _policy_pair_batch(pairs, device=torch.device(device))
        legacy = extract_policy_anchor_observation(policy, batch).native_token
        compact = extract_compact_policy_pairs(
            policy, pairs, device=device, **kwargs
        )
        extractor_diffs.append(
            (compact.native_anchor - legacy.detach().cpu().numpy())
        )
        return compact

    new = collect_policy_carrier_episode(
        plan,
        inventory=_inventory(config),
        policy=policy,
        device=args.device,
        runtime_factory=runtime_factory,
        pair_extractor=compare_extractors,
    )
    old_target = np.asarray(old.target_token, dtype=np.float64)
    # The compact carrier retains the two raw-adjacent frames for each
    # endpoint; the historical P1 label is the final frame only.  Compare
    # the same endpoint slot and restore the legacy branch/H3-H8 layout.
    new_target = np.asarray(new.endpoint.native_anchor[:, -1], dtype=np.float64)
    new_target = new_target.reshape(8, 6, *new_target.shape[1:])
    diff = new_target - old_target
    old_xyz = np.asarray(old.future_xyz, dtype=np.float64)
    new_xyz = np.asarray(new.endpoint.keypoints_xyz[:, -1], dtype=np.float64)
    new_xyz = new_xyz.reshape(8, 6, *new_xyz.shape[1:])
    xyz_diff = new_xyz - old_xyz
    extractor_diff = np.concatenate(extractor_diffs, axis=0).astype(np.float64)
    result = {
        "protocol": "policy_carrier_replay_diagnostic_v1",
        "device": str(args.device),
        "plan": plan.episode_uid,
        "token_shape": list(new_target.shape),
        "token_abs_max": float(np.max(np.abs(diff))),
        "token_abs_mean": float(np.mean(np.abs(diff))),
        "token_mse": float(np.mean(np.square(diff))),
        "token_allclose_1e-5": bool(np.allclose(new_target, old_target, atol=1e-5, rtol=1e-5)),
        "xyz_abs_max": float(np.max(np.abs(xyz_diff))),
        "xyz_abs_mean": float(np.mean(np.abs(xyz_diff))),
        "xyz_allclose_1e-6": bool(np.allclose(new_xyz, old_xyz, atol=1e-6, rtol=1e-6)),
        "same_pair_extractor_abs_max": float(np.max(np.abs(extractor_diff))),
        "same_pair_extractor_abs_mean": float(np.mean(np.abs(extractor_diff))),
        "same_pair_extractor_mse": float(np.mean(np.square(extractor_diff))),
        "same_pair_extractor_allclose_1e-5": bool(np.allclose(extractor_diff, 0.0, atol=1e-5, rtol=1e-5)),
    }
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().resolve().write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
