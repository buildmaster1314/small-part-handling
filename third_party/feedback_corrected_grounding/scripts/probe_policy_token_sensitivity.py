#!/usr/bin/env python3
"""Audit or run the frozen-policy anchor-token sensitivity probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.action_chunk_policy import load_action_chunk_policy  # noqa: E402
from fcpg.policy_token_sensitivity import (  # noqa: E402
    audit_policy_probe_npz,
    probe_policy_token_sensitivity,
)


def _tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array)).to(device)


def _load_complete_carrier(
    deployment_path: Path,
    label_path: Path,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    with np.load(deployment_path, allow_pickle=False) as deployment:
        batch = {
            "obs": _tensor(deployment["policy_obs"], device),
            "q_nominal": _tensor(deployment["policy_q_nominal"], device),
            "keypoints_xyz": _tensor(deployment["policy_keypoints_xyz"], device),
        }
        if "policy_rgb" in deployment:
            batch["rgb"] = _tensor(deployment["policy_rgb"], device)
        elif "policy_sampled_visual" in deployment:
            batch["sampled_visual"] = _tensor(
                deployment["policy_sampled_visual"], device
            )
            batch["global_visual"] = _tensor(
                deployment["policy_global_visual"], device
            )
            batch["image_size"] = _tensor(deployment["policy_image_size"], device)
            batch["fcpg_q_corrected"] = _tensor(
                deployment["policy_fcpg_q_corrected"], device
            )
        else:
            batch["vision_backbone_features"] = _tensor(
                deployment["policy_vision_backbone_features"], device
            ).float()
            batch["vision_prefix"] = _tensor(
                deployment["policy_vision_prefix"], device
            ).float()
            batch["image_size"] = _tensor(deployment["policy_image_size"], device)
            batch["fcpg_q_corrected"] = _tensor(
                deployment["policy_fcpg_q_corrected"], device
            )
        predicted = _tensor(deployment["policy_predicted_anchor"], device)
    with np.load(label_path, allow_pickle=False) as labels:
        target = _tensor(labels["policy_target_anchor"], device)
    return batch, predicted, target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment-cache", type=Path, required=True)
    parser.add_argument("--offline-label-cache", type=Path, required=True)
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action-steps", type=int, default=5)
    parser.add_argument("--jacobian-samples", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    audit = audit_policy_probe_npz(args.deployment_cache, args.offline_label_cache)
    report: dict[str, object] = {
        "schema_version": 1,
        "protocol": "frozen_policy_anchor_sensitivity_artifact_audit_v1",
        "deployment_cache": str(args.deployment_cache.resolve()),
        "offline_label_cache": str(args.offline_label_cache.resolve()),
        "artifact_audit": audit.as_dict(),
    }
    if audit.ready:
        if args.policy_checkpoint is None:
            report["status"] = "fail_closed"
            report["reason"] = "complete carrier requires --policy-checkpoint"
        else:
            device = torch.device(args.device)
            with np.load(args.deployment_cache, allow_pickle=False) as deployment:
                cached_visual_carrier = "policy_rgb" not in deployment.files
            policy, _ = load_action_chunk_policy(
                args.policy_checkpoint,
                device,
                # A complete compact carrier never calls the ViT backbone.
                # Avoid a spurious network fetch while preserving the default
                # pretrained requirement for an RGB carrier.
                pretrained_vision_override=False if cached_visual_carrier else None,
            )
            policy.requires_grad_(False).eval()
            batch, predicted, target = _load_complete_carrier(
                args.deployment_cache, args.offline_label_cache, device
            )
            count = min(int(args.jacobian_samples), int(predicted.shape[0]))
            report["status"] = "completed"
            report["probe"] = probe_policy_token_sensitivity(
                policy,
                batch,
                predicted,
                target,
                action_steps=args.action_steps,
                jacobian_samples=count,
            )
    else:
        report["status"] = "fail_closed"
        report["reason"] = (
            "artifact lacks the exact policy carrier and/or separated anchor arrays; "
            "token-only caches cannot reconstruct frozen-policy actions or Jacobians"
        )

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
