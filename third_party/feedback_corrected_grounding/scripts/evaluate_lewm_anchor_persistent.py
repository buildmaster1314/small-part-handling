#!/usr/bin/env python3
"""Evaluate an Anchor-Persistent LeWM checkpoint on named sequence sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.lewm_anchor_evaluation import evaluate_anchor_persistent  # noqa: E402
from fcpg.lewm_anchor_persistent import AnchorPersistentLeWM  # noqa: E402
from fcpg.lewm_provenance import sha256_file  # noqa: E402
from fcpg.lewm_runtime import (  # noqa: E402
    build_lewm_data,
    choose_device,
    load_lewm_grounded_checkpoint,
)
from fcpg.lewm_sequence_data import LeWMSequenceDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--training-validation", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=2000)
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(config, dict):
        raise TypeError("evaluation config root must be a mapping")
    return config


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model, resolved_config, checkpoint = load_lewm_grounded_checkpoint(
        args.checkpoint, device=device
    )
    if not isinstance(model, AnchorPersistentLeWM):
        raise TypeError("checkpoint is not an Anchor-Persistent LeWM")
    if args.expected_step > 0 and int(checkpoint.get("step", -1)) != args.expected_step:
        raise ValueError(
            f"checkpoint step must be {args.expected_step}, got {checkpoint.get('step')}"
        )
    results: dict[str, object] = {
        "schema_version": 2,
        "evaluation_type": "anchor_persistent_dynamics_and_innovation_v2",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "shuffle_contract": "same_task_different_episode_only",
        "sources": {},
    }
    if args.training_validation:
        bundle = build_lewm_data(resolved_config)
        metrics = evaluate_anchor_persistent(
            model,
            bundle.val_loader,
            device,
            sigreg_weight=float(resolved_config["loss"]["sigreg_weight"]),
            max_batches=args.max_batches,
        )
        results["evaluation_scope"] = "checkpoint_training_validation_split"
        results["sources"]["training_validation"] = {
            "episode_manifest": checkpoint["data_config"].get("episode_manifest"),
            "metrics": metrics,
        }
    else:
        assert args.config is not None
        config = load_config(args.config)
        sources = config.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError("evaluation config requires a non-empty sources list")
        history_size = int(checkpoint["model_config"]["world_model"]["history_size"])
        frame_skip = int(checkpoint["data_config"]["frame_skip"])
        batch_size = int(config.get("batch_size", 128))
        results["evaluation_scope"] = str(config.get("status", "development_only"))
        for source_index, source in enumerate(sources):
            if not isinstance(source, dict):
                raise TypeError("source entries must be mappings")
            name = str(source["name"])
            dataset = LeWMSequenceDataset(
                data_path=source["path"],
                history_size=history_size,
                rollout_steps=1,
                frame_skip=frame_skip,
                source_index=source_index,
                source_name=name,
                correction_cache=source["correction_cache"],
                require_pre_action_alignment=True,
            )
            generator = torch.Generator().manual_seed(int(config.get("seed", 9411)))
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=int(config.get("num_workers", 0)),
                pin_memory=device.type == "cuda",
                generator=generator,
            )
            metrics = evaluate_anchor_persistent(
                model,
                loader,
                device,
                sigreg_weight=float(config.get("sigreg_weight", 0.02)),
                max_batches=args.max_batches,
            )
            results["sources"][name] = {
                "path": str(Path(str(source["path"])).resolve()),
                "path_sha256": sha256_file(source["path"]),
                "correction_cache": str(
                    Path(str(source["correction_cache"])).resolve()
                ),
                "metrics": metrics,
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(results, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != payload:
        raise FileExistsError(f"refusing to overwrite different output: {args.output}")
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
