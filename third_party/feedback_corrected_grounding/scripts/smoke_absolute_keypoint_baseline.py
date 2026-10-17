#!/usr/bin/env python3
"""Run the absolute-keypoint head smoke test without a GPU or ViT download."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.absolute_keypoint_baseline import run_cpu_smoke  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    report = run_cpu_smoke(args.seed)
    print(json.dumps(report, indent=2))
    if not report["finite"] or report["gradient_parameters"] <= 0:
        raise RuntimeError("absolute-keypoint CPU smoke failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
