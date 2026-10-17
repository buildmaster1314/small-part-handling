#!/usr/bin/env python3
"""Collect the frozen fresh Pick panel for the P1 complementary observer."""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np


os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.action_chunk_policy import load_action_chunk_policy  # noqa: E402
from fcpg.lewm_action_branch_collection import (  # noqa: E402
    make_metaworld_episode_runtime,
)
from fcpg.lewm_p1_native_fresh_collection import (  # noqa: E402
    FreshnessInventory,
    build_fresh_episode_plans,
    collect_fresh_p1_native_episode,
    legacy_observed_state_signatures,
    save_fresh_cache_bundle,
    stack_fresh_p1_native_episodes,
    validate_fresh_collection_contract,
)
from fcpg.transformer_grounding import load_fcpg_vit_adapter  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/lewm/eval/p1_native_complementary_fresh.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("artifact path must be a non-empty string")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_fixed_config(path: Path) -> dict[str, Any]:
    canonical_path = DEFAULT_CONFIG.resolve()
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    source = path.expanduser().resolve()
    payload = canonical if source == canonical_path else json.loads(
        source.read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict) or payload != canonical:
        raise ValueError("fresh collection requires the checked-in frozen config")
    validate_fresh_collection_contract(payload)
    return payload


def _verify_artifact(path: Path, expected: object, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = _sha256_file(path)
    if observed != str(expected):
        raise ValueError(f"{label} SHA-256 mismatch: {observed}")
    return observed


def _configured_paths(config: dict[str, Any]) -> dict[str, Path]:
    fresh = config["fresh_data"]
    paths = {
        "deployment": _resolve(fresh["deployment_cache"]),
        "labels": _resolve(fresh["label_cache"]),
        "metadata": _resolve(fresh["metadata"]),
        "inventory": _resolve(config["freshness"]["legacy_observed_state_inventory"]),
    }
    parents = {path.parent for path in paths.values()}
    if len(parents) != 1:
        raise ValueError("all fresh collection artifacts must share one directory")
    return paths


def collect_formal(
    config_path: Path, *, device: str, smoke: bool = False
) -> dict[str, Any]:
    config = _load_fixed_config(config_path)
    paths = _configured_paths(config)
    destination = paths["deployment"].parent
    if destination.exists() and not smoke:
        raise FileExistsError(f"refusing to overwrite fresh output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        freshness = config["freshness"]
        legacy_source = _resolve(freshness["legacy_observation_source"])
        _verify_artifact(
            legacy_source,
            freshness["legacy_observation_source_sha256"],
            "legacy observed-state source",
        )
        observed_inventory = legacy_observed_state_signatures(legacy_source)
        if not observed_inventory:
            raise RuntimeError("legacy observed-state inventory is unexpectedly empty")
        inventory = FreshnessInventory(
            reset_digest_scopes={},
            observed_signature_scopes={"p1_fcpg_training": observed_inventory},
        )

        p1 = config["p1"]
        policy_path = _resolve(p1["policy_checkpoint"])
        corrector_path = _resolve(p1["corrector_checkpoint"])
        _verify_artifact(
            policy_path, p1["policy_checkpoint_sha256"], "P1 policy"
        )
        _verify_artifact(
            corrector_path, p1["corrector_checkpoint_sha256"], "P1 corrector"
        )
        corrector, _ = load_fcpg_vit_adapter(corrector_path, device)
        corrector.requires_grad_(False).eval()
        policy, _ = load_action_chunk_policy(
            policy_path, device, fcpg_adapter_override=corrector
        )
        policy.requires_grad_(False).eval()
        if getattr(policy, "fcpg_adapter", None) is not corrector:
            raise RuntimeError("loaded policy is not bound to the verified corrector")

        fresh = config["fresh_data"]
        runtime_factory = partial(
            make_metaworld_episode_runtime,
            camera=str(fresh["camera"]),
            width=int(fresh["width"]),
            height=int(fresh["height"]),
            benchmark_seed=int(fresh["benchmark_seed"]),
        )
        plans = build_fresh_episode_plans(config)
        if smoke:
            plans = plans[:1]
        episodes = []
        for index, plan in enumerate(plans, start=1):
            episodes.append(
                collect_fresh_p1_native_episode(
                    plan,
                    inventory=inventory,
                    policy=policy,
                    device=device,
                    runtime_factory=runtime_factory,
                )
            )
            print(
                json.dumps(
                    {
                        "event": "fresh_episode_complete",
                        "episode": index,
                        "total": len(plans),
                        "raw_task_index": plan.raw_task_index,
                        "hazard": plan.hazard_id,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if smoke:
            row = episodes[0]
            return {
                "smoke": True,
                "physical_episode_count": 1,
                "history_token_shape": list(row.history_token.shape),
                "target_token_shape": list(row.target_token.shape),
                "reset_physical_state_goal_digest": row.reset_digest,
                "observed_state_signature": row.observed_signature,
            }

        combined = stack_fresh_p1_native_episodes(episodes, contract=config)
        destination.mkdir(mode=0o700, parents=False, exist_ok=False)
        with paths["inventory"].open("xb") as handle:
            np.savez_compressed(
                handle,
                observed_state_signature=np.asarray(sorted(observed_inventory)),
            )
        metadata = save_fresh_cache_bundle(
            combined,
            contract=config,
            config_path=DEFAULT_CONFIG,
            inventory=inventory,
        )
        return {
            "output_dir": str(destination),
            "physical_episode_count": len(episodes),
            "deployment_cache_sha256": metadata["deployment_cache"]["sha256"],
            "label_cache_sha256": metadata["label_cache"]["sha256"],
            "metadata_sha256": _sha256_file(paths["metadata"]),
        }
    except BaseException:
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = collect_formal(
        args.config, device=str(args.device), smoke=bool(args.smoke)
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
