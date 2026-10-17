#!/usr/bin/env python3
"""Build the deployable SO-101 V9 checkpoint without retraining weights."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from fcpg.so101_v9 import V6_PROTOCOL, V9_PROTOCOL, compose_v9_state_dict


REPO = Path(__file__).resolve().parents[1]
DEFAULT_BRIDGE = REPO / (
    "outputs/lerobot/so101/"
    "my_first_task_clean_v1_fcpg_context_bridge_val2_seed6126_2k/latest.pt"
)
DEFAULT_MEMORY = REPO / (
    "outputs/lerobot/so101/"
    "so101_multiscale_memory_freeze75_seed5131_2k/latest.pt"
)
DEFAULT_OUTPUT = REPO / (
    "outputs/lerobot/so101/so101_v9_local_multiscale_seed5131/latest.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-checkpoint", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--memory-checkpoint", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(f"invalid model checkpoint: {path}")
    return payload


def main() -> None:
    args = parse_args()
    bridge_path = args.bridge_checkpoint.expanduser().resolve()
    memory_path = args.memory_checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"V9 output already exists: {output_path}")

    bridge = torch.load(bridge_path, map_location="cpu", weights_only=False)
    local_path = Path(bridge["args"]["checkpoint"])
    if not local_path.is_absolute():
        local_path = (REPO / local_path).resolve()
    local = load_checkpoint(local_path)
    memory = load_checkpoint(memory_path)
    memory_contract = memory.get("run_config", {}).get("contract", {})
    if memory_contract.get("protocol") != V6_PROTOCOL:
        raise ValueError("V9 must be composed from the frozen V6 memory checkpoint")
    local_contract = local.get("run_config", {}).get("contract", {})
    if local_contract.get("protocol") != "so101-state-residual-lewm-v5-temporal":
        raise ValueError("V9 local backbone must use the temporal V5 contract")

    model = compose_v9_state_dict(memory["model"], local["model"])
    contract = dict(memory_contract)
    contract.update(
        {
            "protocol": V9_PROTOCOL,
            "composition": "task-local temporal backbone plus frozen V6 memory adapter",
            "local_backbone_checkpoint": str(local_path),
            "local_backbone_step": int(local.get("step", -1)),
            "memory_checkpoint": str(memory_path),
            "memory_step": int(memory.get("step", -1)),
            "visual_base_compatibility": "exact",
            "clean_runtime_policy": "local-backbone parity",
            "memory_runtime_policy": "explicit auto/off/always",
        }
    )
    embedded_args = dict(memory.get("args", {}))
    embedded_args["output"] = str(output_path.parent)
    embedded_args["backbone_checkpoint"] = str(local_path)
    embedded_args["memory_checkpoint"] = str(memory_path)
    payload = {
        "args": embedded_args,
        "metrics": {
            "status": "composed-not-retrained",
            "local_backbone_metrics": local.get("metrics", {}),
            "memory_metrics": memory.get("metrics", {}),
        },
        "model": model,
        "run_config": {
            **memory.get("run_config", {}),
            "contract": contract,
        },
        "step": int(memory.get("step", -1)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(output_path)
    print(f"protocol={V9_PROTOCOL}")
    print(f"parameters={int(contract['parameters'])}")


if __name__ == "__main__":
    main()
