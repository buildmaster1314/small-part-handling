#!/usr/bin/env python3
"""Build matched Pick policy examples from the frozen P1 observer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.action_chunk_policy import load_action_chunk_policy  # noqa: E402
from fcpg.lewm_p1_native_complementary import (  # noqa: E402
    FrozenP1NativeCheckpointSpec,
    load_frozen_p1_native_checkpoint,
)
from fcpg.lewm_posterior_aware_policy import (  # noqa: E402
    build_posterior_policy_cache,
)


DEFAULT_SOURCE = ROOT / "data/metaworld_projection/scripted_pick_actualcam_se3_clean25_300eps_224.npz"
DEFAULT_POLICY_CACHE = ROOT / "data/policy_visual_cache/lewm_stage1/pick_seed0_sig0_pick"
DEFAULT_POLICY = ROOT / "outputs/action_chunk/p1/pick_fcpg_pureres_seed0_retry1/fcpg_action_chunk_best.pt"
DEFAULT_P1_CONFIG = ROOT / "config/lewm/eval/p1_native_complementary_fresh.json"
DEFAULT_OUTPUT = ROOT / "outputs/lewm_posterior_aware_policy/pick_train_cache_seed63101.npz"


def _resolve(path: str | Path) -> Path:
    target = Path(path).expanduser()
    return target.resolve() if target.is_absolute() else (ROOT / target).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--policy-cache", type=Path, default=DEFAULT_POLICY_CACHE)
    parser.add_argument("--policy-checkpoint", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--p1-config", type=Path, default=DEFAULT_P1_CONFIG)
    parser.add_argument("--p1-seed", type=int, default=15801)
    parser.add_argument("--schedules-per-episode", type=int, default=2)
    parser.add_argument("--schedule-seed", type=int, default=63101)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data_path = _resolve(args.data)
    policy_cache = _resolve(args.policy_cache)
    policy_path = _resolve(args.policy_checkpoint)
    config_path = _resolve(args.p1_config)
    output_path = _resolve(args.output)
    if output_path.exists():
        parser.error(f"output already exists: {output_path}")
    if not str(args.device).startswith("cuda:0") and str(args.device) != "cpu":
        parser.error("this experiment may use cuda:0 or cpu only")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_path = _resolve(config["frozen_predictor_training_contract"]["config"])
    expected_config = json.loads(expected_path.read_text(encoding="utf-8"))
    records = {
        int(record["seed"]): record for record in config["frozen_predictors"]
    }
    if int(args.p1_seed) not in records:
        parser.error(f"unknown P1 seed {args.p1_seed}")
    record = records[int(args.p1_seed)]
    device = torch.device(args.device)
    policy, _ = load_action_chunk_policy(policy_path, device)
    policy.set_reliability_override("ones")
    policy.requires_grad_(False).eval()
    pair = load_frozen_p1_native_checkpoint(
        FrozenP1NativeCheckpointSpec(
            seed=int(args.p1_seed),
            path=_resolve(record["checkpoint"]),
            sha256=str(record["sha256"]),
        ),
        expected_config=expected_config,
        device=device,
    )

    with np.load(data_path, allow_pickle=False) as source:
        source_indices = np.arange(source["obs"].shape[0], dtype=np.int64)
        episode_id = np.asarray(source["episode_id"], dtype=np.int64)
        frame_id = np.asarray(source["frame_id"], dtype=np.int64)
        xyz = np.asarray(source["keypoints_xyz"], dtype=np.float32)
        action = np.asarray(source["action"], dtype=np.float32)
    sampled_visual = np.load(policy_cache / "sampled_visual.npy", mmap_mode="r")
    cache = build_posterior_policy_cache(
        policy=policy,
        factual_predictor=pair.factual,
        no_action_predictor=pair.no_action,
        source_indices=source_indices,
        episode_ids=episode_id,
        frame_ids=frame_id,
        sampled_visual=sampled_visual,
        xyz=xyz,
        actions=action,
        schedules_per_episode=int(args.schedules_per_episode),
        schedule_seed=int(args.schedule_seed),
        device=device,
        inference_batch_size=int(args.batch_size),
    )
    cache.save(output_path)
    summary = {
        "output": str(output_path),
        "rows": len(cache),
        "episodes": int(np.unique(cache.episode_id).size),
        "schedules_per_episode": int(args.schedules_per_episode),
        "drop_rows": int(cache.drop_active.sum()),
        "drop_fraction": float(cache.drop_active.mean()),
        "p1_seed": int(args.p1_seed),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
