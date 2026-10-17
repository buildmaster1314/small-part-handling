#!/usr/bin/env python3
"""Evaluate frozen SO-101 long-history LeWM under recent frame freezes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from fcpg.so101_multiscale_memory import MultiScaleHistoryLeWM, freeze_recent_context
from scripts.train_so101_lewm import SO101Sequences, _make_loader, _shuffle_actions_within_group
from scripts.train_so101_multiscale_memory import _to_device
from scripts.visualize_so101_lewm import build_checkpoint_model


METRIC_KEYS = (
    "visual_adapter_mse",
    "visual_parent_mse",
    "visual_shuffled_memory_mse",
    "state_adapter_mse",
    "state_parent_mse",
    "state_shuffled_memory_mse",
)


def _balanced_eval_indices(
    dataset: SO101Sequences,
    limit: int | None,
    minimum_past_frames: int,
) -> list[int]:
    """Select a balanced, deterministic subset after a common history cutoff."""
    episode_starts: dict[tuple[int, int], int] = {}
    for cache_id, cache in enumerate(dataset.caches):
        episodes = np.asarray(cache.episodes)
        starts = np.concatenate(
            ([0], np.flatnonzero(episodes[1:] != episodes[:-1]) + 1)
        )
        episode_starts.update(
            {
                (cache_id, int(episodes[start])): int(start)
                for start in starts
            }
        )

    by_group: dict[int, list[int]] = defaultdict(list)
    for index, (cache_id, start) in enumerate(dataset.samples):
        cache = dataset.caches[cache_id]
        episode = int(cache.episodes[start])
        if start - episode_starts[(cache_id, episode)] < minimum_past_frames:
            continue
        by_group[dataset.group_ids[index]].append(index)

    for group, indices in by_group.items():
        indices.sort(
            key=lambda index: hashlib.sha256(
                (
                    f"eval:{dataset.split_seed}:{group}:"
                    f"{dataset.samples[index][0]}:"
                    f"{int(dataset.caches[dataset.samples[index][0]].episodes[dataset.samples[index][1]])}:"
                    f"{int(dataset.caches[dataset.samples[index][0]].frame_indices[dataset.samples[index][1]])}"
                ).encode()
            ).digest()
        )

    ordered: list[int] = []
    cursor = 0
    while True:
        added = False
        for group in sorted(by_group):
            indices = by_group[group]
            if cursor < len(indices):
                ordered.append(indices[cursor])
                added = True
                if limit is not None and len(ordered) >= limit:
                    return ordered
        if not added:
            return ordered
        cursor += 1


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    values = {key: float(np.mean([row[key] for row in rows])) for key in METRIC_KEYS}
    values.update(
        {
            "visual_adapter_gain": values["visual_parent_mse"] - values["visual_adapter_mse"],
            "visual_shuffle_delta": values["visual_shuffled_memory_mse"] - values["visual_adapter_mse"],
            "state_adapter_gain": values["state_parent_mse"] - values["state_adapter_mse"],
            "state_shuffle_delta": values["state_shuffled_memory_mse"] - values["state_adapter_mse"],
        }
    )
    return values


def _summarize(
    rows: list[dict[str, float]],
    grouped: dict[int, list[dict[str, float]]],
    group_labels: dict[int, str],
) -> dict:
    by_group = {
        group_labels[group]: {**_mean(group_rows), "sequences": len(group_rows)}
        for group, group_rows in sorted(grouped.items())
    }
    macro = _mean(
        [{key: values[key] for key in METRIC_KEYS} for values in by_group.values()]
    )
    macro.update(
        {
            "visual_positive_groups": sum(
                values["visual_adapter_gain"] > 0 for values in by_group.values()
            ),
            "visual_shuffle_positive_groups": sum(
                values["visual_shuffle_delta"] > 0 for values in by_group.values()
            ),
            "state_positive_groups": sum(
                values["state_adapter_gain"] > 0 for values in by_group.values()
            ),
        }
    )
    return {
        **macro,
        "micro": {**_mean(rows), "sequences": len(rows)},
        "by_source_task": by_group,
        "macro_groups": len(by_group),
    }


@torch.no_grad()
def evaluate_dropout(
    model: MultiScaleHistoryLeWM,
    loader,
    device: torch.device,
    group_labels: dict[int, str],
    center: torch.Tensor,
    scale: torch.Tensor,
) -> dict:
    model.eval()
    conditions = tuple(range(4))
    rows = {suffix: [] for suffix in conditions}
    grouped = {suffix: defaultdict(list) for suffix in conditions}

    for batch in loader:
        values = _to_device(batch, device, center, scale)
        groups = batch["group"]
        clean_target = None
        clean_state_target = None
        for suffix in conditions:
            condition_values = dict(values)
            condition_values["pixels"] = freeze_recent_context(
                values["pixels"], values["memory_pixels"], suffix
            )
            recent_visual_valid = torch.ones(
                len(groups), 3, dtype=torch.bool, device=device
            )
            if suffix:
                recent_visual_valid[:, -suffix:] = False
            condition_values["recent_visual_valid"] = recent_visual_valid
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                output = model(
                    condition_values["pixels"],
                    condition_values["actions"],
                    condition_values["state"],
                    memory_pixels=condition_values["memory_pixels"],
                    memory_actions=condition_values["memory_actions"],
                    memory_state=condition_values["memory_state"],
                    memory_age=condition_values["memory_age"],
                    memory_valid=condition_values["memory_valid"],
                    recent_visual_valid=recent_visual_valid,
                )
                if clean_target is None:
                    clean_target = output.target_embedding
                    clean_state_target = output.target_state
                shuffled = model.predict_context(
                    output.embedding[:, :-1],
                    condition_values["actions"],
                    condition_values["state"][:, :-1],
                    memory_embedding=_shuffle_actions_within_group(
                        output.memory_embedding, groups
                    ),
                    memory_actions=_shuffle_actions_within_group(
                        condition_values["memory_actions"], groups
                    ),
                    memory_state=_shuffle_actions_within_group(
                        condition_values["memory_state"], groups
                    ),
                    memory_age=_shuffle_actions_within_group(
                        condition_values["memory_age"], groups
                    ),
                    memory_valid=_shuffle_actions_within_group(
                        condition_values["memory_valid"], groups
                    ),
                    recent_visual_valid=recent_visual_valid,
                )
                metrics = {
                    "visual_adapter_mse": (output.prediction - clean_target).square().mean(dim=(1, 2)),
                    "visual_parent_mse": (output.backbone_prediction - clean_target).square().mean(dim=(1, 2)),
                    "visual_shuffled_memory_mse": (shuffled[1] - clean_target).square().mean(dim=(1, 2)),
                    "state_adapter_mse": (output.state_prediction - clean_state_target).square().mean(dim=(1, 2)),
                    "state_parent_mse": (output.backbone_state_prediction - clean_state_target).square().mean(dim=(1, 2)),
                    "state_shuffled_memory_mse": (shuffled[4] - clean_state_target).square().mean(dim=(1, 2)),
                }
            for index in range(len(groups)):
                row = {key: float(value[index]) for key, value in metrics.items()}
                group = int(groups[index])
                rows[suffix].append(row)
                grouped[suffix][group].append(row)

    return {
        "protocol": "so101-multiscale-recent-freeze-diagnostic-v1",
        "conditions": {
            f"freeze_recent_{suffix}": _summarize(
                rows[suffix], grouped[suffix], group_labels
            )
            for suffix in conditions
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=3840,
        help="Balanced validation subset; use 0 for the complete validation split.",
    )
    parser.add_argument(
        "--minimum-past-frames",
        type=int,
        default=0,
        help="Require this many earlier raw frames for matched checkpoint comparisons.",
    )
    args = parser.parse_args()
    if args.device.startswith("cuda:1"):
        parser.error("CUDA1 is forbidden for this project")
    if args.device not in {"cpu", "cuda:0"}:
        parser.error("use CPU or CUDA0")
    if (
        args.batch_size <= 0
        or args.workers < 0
        or args.max_sequences < 0
        or args.minimum_past_frames < 0
    ):
        parser.error("invalid evaluation size")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    contract = checkpoint.get("run_config", {}).get("contract", {})
    if contract.get("protocol") not in {
        "so101-state-residual-lewm-v6-multiscale-memory",
        "so101-state-residual-lewm-v7-reliability-memory",
    }:
        raise ValueError("checkpoint is not a multiscale-memory LeWM")
    model = build_checkpoint_model(checkpoint, 5, 3, device)
    if not isinstance(model, MultiScaleHistoryLeWM):
        raise TypeError("checkpoint did not reconstruct MultiScaleHistoryLeWM")

    center_np = np.asarray(contract["normalization_center"], dtype=np.float32)
    scale_np = np.asarray(contract["normalization_scale"], dtype=np.float32)
    exclusions = {
        (value.rsplit(":", 1)[0], int(value.rsplit(":", 1)[1]))
        for value in contract.get("exclusions", ())
    }
    val_set = SO101Sequences(
        [Path(path) for path in contract["caches"]],
        "val",
        5,
        3,
        exclusions=exclusions,
        split_seed=int(contract["split_seed"]),
        val_fraction=float(contract["val_fraction"]),
        joint_scale=100.0,
        motion_threshold=0.01,
        action_center=center_np,
        action_scale=scale_np,
        memory_offsets=tuple(int(value) for value in contract["memory_offsets"]),
    )
    limit = None if args.max_sequences == 0 else args.max_sequences
    indices = _balanced_eval_indices(
        val_set, limit, args.minimum_past_frames
    )
    if not indices:
        raise ValueError("no validation sequences survive the history cutoff")
    loader = _make_loader(val_set, indices, args, device, train=False)
    center = torch.as_tensor(center_np, device=device)
    scale = torch.as_tensor(scale_np, device=device)
    result = evaluate_dropout(
        model, loader, device, val_set.group_labels, center, scale
    )
    result.update(
        {
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "max_sequences": args.max_sequences,
            "minimum_past_frames": args.minimum_past_frames,
            "memory_offsets": list(contract["memory_offsets"]),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    console = {
        name: {
            "sequences": values["micro"]["sequences"],
            "visual_parent_mse": values["visual_parent_mse"],
            "visual_adapter_mse": values["visual_adapter_mse"],
            "visual_adapter_gain": values["visual_adapter_gain"],
            "visual_shuffle_delta": values["visual_shuffle_delta"],
            "visual_positive_groups": values["visual_positive_groups"],
            "visual_shuffle_positive_groups": values[
                "visual_shuffle_positive_groups"
            ],
            "state_parent_mse": values["state_parent_mse"],
            "state_adapter_mse": values["state_adapter_mse"],
        }
        for name, values in result["conditions"].items()
    }
    print(json.dumps(console, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
