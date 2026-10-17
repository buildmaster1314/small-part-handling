#!/usr/bin/env python3
"""Create the Git-trackable ledger for four completed anchor-visual runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.lewm_checkpoint_ledger import freeze_checkpoint_ledger  # noqa: E402


DEFAULT_CHECKPOINT_ROOT = Path(
    "outputs/lewm_grounded/anchor_visual_ablation_seed5120"
)
DEFAULT_LEDGER = Path(
    "config/lewm/checkpoints/anchor_visual_ablation_seed5120_latest_step2000.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_LEDGER))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ledger = freeze_checkpoint_ledger(
        checkpoint_root=args.checkpoint_root,
        output=args.output,
        repo_root=REPO_ROOT,
    )
    print(
        json.dumps(
            {
                "ledger": str((REPO_ROOT / args.output).resolve()),
                "step": ledger["matched_training_contract"]["step"],
                "modes": ledger["modes"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
