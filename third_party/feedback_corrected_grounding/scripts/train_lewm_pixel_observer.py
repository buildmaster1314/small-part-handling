#!/usr/bin/env python3
"""Train the standalone LeWM pixel residual observer adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.lewm_pixel_observer_runtime import (  # noqa: E402
    train_pixel_observer_adapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a GroundedLeWM and train only its pixel residual prior adapter "
            "against q_corrected-q_nominal."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=0,
        help="Stop after this many adapter optimizer steps and validate one batch.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dot-list overrides, for example loader.batch_size=8",
    )
    args = parser.parse_args()
    if args.smoke_steps < 0:
        parser.error("--smoke-steps must be non-negative")
    return args


def load_config(path: Path, overrides: list[str]) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    config = OmegaConf.load(path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(config)
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("pixel observer YAML root must be a mapping")
    return resolved


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    result = train_pixel_observer_adapter(
        config,
        device=args.device,
        smoke_steps=args.smoke_steps,
        config_path=args.config,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "latest_checkpoint": str(result.latest_checkpoint),
                "best_checkpoint": str(result.best_checkpoint),
                "metrics": str(result.metrics_path),
                "step": result.global_step,
                "epoch": result.epoch,
                "best_val_prior_mse": result.best_val_prior_mse,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
