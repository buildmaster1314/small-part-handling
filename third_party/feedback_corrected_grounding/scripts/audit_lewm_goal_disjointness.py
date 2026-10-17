#!/usr/bin/env python3
"""Run the bounded, post-hoc upstream goal-signature forensic audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_goal_disjointness_audit import (  # noqa: E402
    AUDIT_PROTOCOL,
    GoalDisjointnessAuditError,
    run_goal_disjointness_audit,
    sha256_file,
    write_atomic_json,
)


DEFAULT_CONFIG = "config/lewm/eval/upstream_goal_disjointness_forensic.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--confirm-posthoc-limited-scope",
        action="store_true",
        help=(
            "Acknowledge that this report is post-hoc, covers only historical data and "
            "Gate L1 LeWM, and cannot upgrade Counterfactual v2 G0 eligibility."
        ),
    )
    return parser.parse_args(argv)


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (repo_root / path).resolve() if not path.is_absolute() else path.resolve()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.confirm_posthoc_limited_scope:
        raise SystemExit(
            "refusing to run without --confirm-posthoc-limited-scope; this audit does not upgrade G0"
        )
    repo_root = Path(args.repo_root).expanduser().resolve()
    config_path = _resolve(repo_root, args.config)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot load audit config {config_path}: {exc}") from exc
    if not isinstance(config, dict) or config.get("protocol") != AUDIT_PROTOCOL:
        raise SystemExit(f"invalid audit config protocol in {config_path}")
    output_value = args.output
    if output_value is None:
        output_section = config.get("output")
        if not isinstance(output_section, dict) or not isinstance(output_section.get("path"), str):
            raise SystemExit("audit config is missing output.path")
        output_value = output_section["path"]
    output_path = _resolve(repo_root, output_value)

    try:
        report = run_goal_disjointness_audit(config, repo_root=repo_root)
        write_atomic_json(output_path, report)
    except (GoalDisjointnessAuditError, FileExistsError, OSError) as exc:
        raise SystemExit(f"goal-disjointness audit failed closed: {exc}") from exc

    comparison = report["goal_signature_comparison"]
    conclusion = report["bounded_conclusion"]
    print(f"output: {output_path}")
    print(f"output_file_sha256: {sha256_file(output_path)}")
    print(f"result_sha256: {report['result_sha256']}")
    print(f"task_scoped_exact_disjoint: {comparison['task_scoped_exact_disjoint']}")
    print(f"total_exact_intersection_count: {comparison['total_exact_intersection_count']}")
    print(f"global_minimum_l2_distance: {comparison['global_minimum_l2_distance']:.9f}")
    print(f"proof_scope: {report['proof_scope']}")
    print(f"full_bundle_formal_verified: {conclusion['full_bundle_formal_verified']}")
    print(
        "counterfactual_v2_g0_eligibility_upgraded: "
        f"{conclusion['counterfactual_v2_g0_eligibility_upgraded']}"
    )
    return 0 if comparison["task_scoped_exact_disjoint"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
