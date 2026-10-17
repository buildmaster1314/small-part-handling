#!/usr/bin/env python3
"""Resolve the A4c data Gate with a valid contextual-state baseline."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any


DEFAULT_PREFLIGHT = Path(
    "outputs/lewm_action_branch_a4c_v4/preflight_gate/summary.json"
)
DEFAULT_OUTPUT = Path("outputs/lewm_action_branch_a4c_v4/data_gate")
DEFAULT_GRIPPER_AUDIT = Path(
    "outputs/lewm_action_branch_a4c_v4/gripper_support/summary.json"
)


def build_a4c_data_gate(
    preflight: Mapping[str, Any], gripper_support: Mapping[str, Any]
) -> dict[str, Any]:
    """Replace only the invalid nested-token check in the A4c data decision."""

    if preflight.get("mode") != "formal":
        raise ValueError("A4c data authorization requires a formal preflight")
    if preflight.get("input_contains_test_rows") is not False:
        raise PermissionError("A4c data authorization must not inspect reserved test")
    checks_value = preflight.get("checks")
    if not isinstance(checks_value, Mapping):
        raise TypeError("A4c preflight checks must be a mapping")
    checks = {str(name): bool(value) for name, value in checks_value.items()}
    if "task_state_sufficiency" not in checks:
        raise ValueError("A4c preflight lacks task_state_sufficiency")

    task_state = preflight.get("task_state_sufficiency")
    if not isinstance(task_state, Mapping):
        raise TypeError("A4c task-state result must be a mapping")
    contextual = task_state.get("contextualized_state_gate")
    if not isinstance(contextual, Mapping) or not isinstance(
        contextual.get("passed"), bool
    ):
        raise ValueError("A4c preflight lacks the contextualized-state Gate")

    legacy_nested_passed = bool(checks["task_state_sufficiency"])
    checks["task_state_sufficiency"] = bool(contextual["passed"])
    if not isinstance(gripper_support, Mapping):
        raise TypeError("A4c gripper support must be a mapping")
    if gripper_support.get("protocol") != "fcpg_action_branch_a4c_gripper_support_v1":
        raise ValueError("A4c gripper support protocol mismatch")
    if not isinstance(gripper_support.get("passed"), bool):
        raise TypeError("A4c gripper support decision must be boolean")
    checks["gripper_intervention_support"] = bool(gripper_support["passed"])
    passed = all(checks.values())
    comparison = contextual.get("comparison")
    if not isinstance(comparison, Mapping):
        raise TypeError("A4c contextualized-state comparison must be a mapping")
    return {
        "schema_version": 1,
        "protocol": "fcpg_action_branch_phase_schedule_a4c_data_gate_v1",
        "stage": "a4c_formal_data_authorization",
        "decision": "Data-Go" if passed else "Data-No-Go",
        "passed": bool(passed),
        "training_authorized": bool(passed),
        "reserved_test_access_authorized": False,
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "legacy_nested_summary_increment": {
            "passed": legacy_nested_passed,
            "role": "diagnostic_only",
            "reason": (
                "The saved anchor tokens are outputs of joint self-attention over "
                "summary, global, robot, and anchor tokens, so they are not a "
                "context-free anchor baseline."
            ),
        },
        "contextualized_state_comparison": dict(comparison),
        "gripper_intervention_support": {
            "passed": bool(gripper_support["passed"]),
            "failed_groups": list(gripper_support.get("failed_groups", ())),
        },
        "claim_boundary": (
            "This authorizes shared seven-task 2k development training only. It "
            "does not authorize 10k confirmation or reserved-test access."
        ),
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    comparison = report["contextualized_state_comparison"]
    lines = [
        "# A4c data Gate",
        "",
        f"Decision: **{report['decision']}**",
        f"2k training authorized: **{str(report['training_authorized']).lower()}**",
        "",
        "Full contextualized grounded state versus shared-only context:",
        "",
        f"- Relative improvement: **{comparison['relative_improvement']:.2%}**",
        f"- Nonnegative tasks: **{comparison['nonnegative_task_count']}/7**",
        f"- Gripper intervention support: **{str(report['gripper_intervention_support']['passed']).lower()}**",
        "",
        "The legacy summary-versus-anchor increment remains diagnostic because "
        "the saved anchor tokens already received global context through self-attention.",
        "",
        str(report["claim_boundary"]),
    ]
    if report["failed_checks"]:
        lines.extend(["", "Failed checks: " + ", ".join(report["failed_checks"])])
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--preflight-summary", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--gripper-audit", type=Path, default=DEFAULT_GRIPPER_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    value = json.loads(args.preflight_summary.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("A4c preflight summary must contain one JSON object")
    gripper = json.loads(args.gripper_audit.read_text(encoding="utf-8"))
    if not isinstance(gripper, Mapping):
        raise TypeError("A4c gripper audit must contain one JSON object")
    report = build_a4c_data_gate(value, gripper)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output / "summary.md").write_text(
        _render_markdown(report), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "training_authorized": report["training_authorized"],
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
