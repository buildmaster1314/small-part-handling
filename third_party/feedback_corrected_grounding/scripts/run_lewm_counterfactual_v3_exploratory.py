#!/usr/bin/env python3
"""Run the post-hoc Counterfactual v3 actual-return diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_counterfactual_v2_storage import (  # noqa: E402
    load_counterfactual_v2_store,
)
from fcpg.lewm_counterfactual_v3_gate import prepare_gate_bundle  # noqa: E402


DEFAULT_DATASET = (
    "outputs/lewm_counterfactual_v3/state_oracle_seed9216/dataset"
)
DEFAULT_OUTPUT = (
    "outputs/lewm_counterfactual_v3_exploratory/"
    "state_oracle_residual_seed9216"
)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--confirm-exploratory",
        action="store_true",
        help=(
            "confirm a descriptive post-hoc run on the already exposed v3 "
            "validation split"
        ),
    )
    return parser


def _run_model(
    bundle: Any,
    *,
    device: str,
    direct_steps: int,
    residual_steps: int,
) -> Any:
    from fcpg.lewm_counterfactual_v3_exploratory import (
        run_exploratory_actual_return,
    )

    return run_exploratory_actual_return(
        bundle,
        device=device,
        direct_steps=direct_steps,
        residual_steps=residual_steps,
    )


def _result_report(result: Any) -> dict[str, Any]:
    value = getattr(result, "report", None)
    if not isinstance(value, dict):
        raise TypeError("exploratory runner must return an object with a dict report")
    # Verify serializability before creating the exclusive output directory.
    json.dumps(value, allow_nan=False)
    return dict(value)


def _write_report(destination: Path, report: dict[str, Any]) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.confirm_exploratory:
        raise SystemExit("refusing run without --confirm-exploratory")

    dataset_path = _resolve(args.dataset)
    destination = _resolve(args.output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"exploratory output already exists: {destination}")

    dataset = load_counterfactual_v2_store(dataset_path)
    bundle = prepare_gate_bundle(
        dataset.deployment_inputs,
        dataset.offline_labels,
        dataset.audit_arrays,
        dataset.index_rows,
    )
    # PyTorch requires this setting before the first CUDA GEMM when
    # deterministic algorithms are enabled by the training routine.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    result = _run_model(
        bundle,
        device="cuda:0",
        direct_steps=2000,
        residual_steps=2000,
    )
    scientific_result = _result_report(result)
    report = {
        **scientific_result,
        "classification": "post_hoc_exploratory",
        "post_hoc": True,
        "exposed_validation": True,
        "descriptive_only": True,
        "formal_claim_allowed": False,
        "source_dataset": str(dataset_path),
        "output_directory": str(destination),
        "device": "cuda:0",
        "direct_steps": 2000,
        "residual_steps": 2000,
        "claim_boundary": (
            "Descriptive evidence on the exposed v3 split cannot reverse the "
            "formal P0 result or authorize VisualOracle, LeWM, GoalToken, "
            "FCPG planning, or CEM."
        ),
    }
    _write_report(destination, report)
    print(f"destination: {destination}")
    print("classification: post_hoc_exploratory")
    print("descriptive_only: true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
