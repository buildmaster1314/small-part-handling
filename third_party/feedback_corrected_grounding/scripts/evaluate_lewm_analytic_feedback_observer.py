#!/usr/bin/env python3
"""Audit whether a P1 deployment cache can score analytic Feedback-WM.

The command intentionally reads a deployment-input archive only.  It does not
load an offline-label archive and it refuses any deployment archive containing
hidden future targets.  Current three-visible-state P1 artifacts are expected
to fail closed because they cannot form a pre-drop one-step innovation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_analytic_feedback_observer import (  # noqa: E402
    audit_feedback_observer_artifact,
)


DEFAULT_DEPLOYMENT_CACHE = ROOT / (
    "outputs/lewm_p1_native_complementary_fresh/fresh_deployment_inputs.npz"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deployment-cache",
        type=Path,
        default=DEFAULT_DEPLOYMENT_CACHE,
        help="deployment-only NPZ; offline target archives are forbidden",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional JSON audit receipt",
    )
    return parser


def run_audit(deployment_cache: Path) -> dict[str, object]:
    path = deployment_cache.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    audit = audit_feedback_observer_artifact(arrays)
    return {
        "protocol": "fcpg_analytic_feedback_observer_artifact_audit_v1",
        "deployment_cache": str(path),
        **audit.as_dict(),
        "decision": "Scorable" if audit.scorable else "Fail-Closed-Unscorable",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_audit(args.deployment_cache)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if bool(report["scorable"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
