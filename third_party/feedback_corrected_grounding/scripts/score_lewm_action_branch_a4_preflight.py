#!/usr/bin/env python3
"""Score the A4 zero-training active-control and task-state preflight."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a4 import validate_a4_contract  # noqa: E402
from fcpg.lewm_action_branch_a4_metrics import (  # noqa: E402
    FORMAL_MODE,
    SMOKE_MODE,
    summarize_a4_preflight,
    validate_a4_scoring_contract,
)


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_active_control_a4.json"
DEFAULT_LEDGER = ROOT / "outputs/lewm_action_branch_a4/proposal_ledger.npz"
DEFAULT_CACHE = ROOT / "outputs/lewm_action_branch_a4/state_cache.npz"
DEFAULT_LABELS = ROOT / "outputs/lewm_action_branch_a4/preflight_labels.npz"
DEFAULT_EXTERNAL_GATES = (
    ROOT / "outputs/lewm_action_branch_a4/external_preflight_gates.json"
)
DEFAULT_OUTPUT = ROOT / "outputs/lewm_action_branch_a4/preflight_gate"


def _resolve(path: str | Path) -> Path:
    source = Path(path).expanduser()
    return source.resolve() if source.is_absolute() else (ROOT / source).resolve()


def _load_json(path: str | Path, *, label: str) -> dict[str, Any]:
    source = _resolve(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def _load_npz_without_reserved_test(
    path: str | Path, *, label: str
) -> dict[str, np.ndarray]:
    """Read split first and refuse the archive before touching payload arrays."""

    source = _resolve(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as archive:
        if "split" not in archive.files:
            raise ValueError(f"{label} is missing mandatory split metadata")
        split = np.asarray(archive["split"])
        if split.ndim != 1 or split.dtype.kind not in {"U", "S"}:
            raise TypeError(f"{label} split must be a fixed-width string vector")
        split_names = split.astype(str, copy=False)
        if np.any(split_names == "test"):
            raise PermissionError(
                f"{label} contains reserved test rows; preflight is not authorized to inspect them"
            )
        unknown = sorted(set(split_names.tolist()) - {"train", "development"})
        if unknown:
            raise ValueError(f"{label} contains unsupported split labels: {unknown}")
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def render_markdown(summary: Mapping[str, Any]) -> str:
    support = summary["support"]
    sufficiency = summary["task_state_sufficiency"]
    comparison = sufficiency["comparison"]
    lines = [
        "# A4 active-control preflight",
        "",
        f"Mode: **{summary['mode']}**",
        f"Decision: **{summary['decision']}**",
        f"Training authorized: **{str(summary['training_authorized']).lower()}**",
        "",
        "| task | active episodes | physical support | latent support | source expert success |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for task_name, row in support["per_task"].items():
        lines.append(
            f"| {task_name} | {row['active_bundle_episode_count']} | "
            f"{row['physical_support_episode_count']} | "
            f"{row['latent_support_episode_count']} | "
            f"{row['source_expert_success_episode_macro']:.1%} |"
        )
    lines.extend(
        [
            "",
            "Task-state summary-plus-anchor relative improvement over anchor-only: "
            f"**{comparison['relative_improvement']:.2%}** "
            f"({comparison['nonnegative_task_count']}/7 nonnegative tasks).",
        ]
    )
    if summary["failed_checks"]:
        lines.extend(["", "Failed checks: " + ", ".join(summary["failed_checks"])])
    lines.extend(
        [
            "",
            "The scorer used train/development rows only. Reserved test remains sealed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--labels",
        "--measurements",
        "--support",
        dest="labels",
        type=Path,
        default=DEFAULT_LABELS,
    )
    parser.add_argument(
        "--external-gates", type=Path, default=DEFAULT_EXTERNAL_GATES
    )
    parser.add_argument(
        "--output", "--output-dir", dest="output", type=Path, default=DEFAULT_OUTPUT
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Validate a bounded 1-2 episode/task/split panel; never authorizes training"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = _load_json(args.config, label="A4 contract")
    validate_a4_contract(contract)
    validate_a4_scoring_contract(contract, phase="preflight")
    ledger = _load_npz_without_reserved_test(args.ledger, label="A4 proposal ledger")
    cache = _load_npz_without_reserved_test(args.cache, label="A4 state cache")
    labels = _load_npz_without_reserved_test(args.labels, label="A4 preflight labels")
    external = _load_json(args.external_gates, label="external preflight gates")
    mode = SMOKE_MODE if args.smoke else FORMAL_MODE
    summary = summarize_a4_preflight(
        ledger,
        cache,
        labels,
        external,
        contract=contract,
        mode=mode,
    )
    output = _resolve(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "training_authorized": summary["training_authorized"],
                "output": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
