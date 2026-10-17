#!/usr/bin/env python3
"""Train the adapter-only causal PixelObserver v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_pixel_observer_v2_training import train_pixel_observer_v2  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("overrides", nargs="*")
    result = parser.parse_args()
    if result.smoke_steps < 0:
        parser.error("--smoke-steps must be non-negative")
    return result


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("config root must be a mapping")
    if args.output_dir is not None:
        resolved["output_dir"] = str(args.output_dir)
    result = train_pixel_observer_v2(
        resolved, device=args.device, smoke_steps=args.smoke_steps
    )
    print(
        json.dumps(
            {
                "latest_checkpoint": str(result.latest_checkpoint),
                "best_checkpoint": str(result.best_checkpoint),
                "metrics": str(result.metrics_path),
                "step": result.global_step,
                "best_val_loss": result.best_val_loss,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
