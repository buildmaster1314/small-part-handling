#!/usr/bin/env python3
"""Run the fixed four-fold post-hoc Counterfactual v3 diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


# Must be configured before importing project modules that import torch.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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
    "state_oracle_residual_crossfit_seed9216"
)
CROSSFIT_FOLDS = 4
DIRECT_STEPS_PER_FOLD = 2000
RESIDUAL_STEPS = 2000
DEVICE = "cuda:0"


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--confirm-exploratory-crossfit",
        action="store_true",
        help=(
            "confirm the fixed four-fold descriptive run on the already "
            "exposed v3 data"
        ),
    )
    return parser


def _run_model(bundle: Any) -> Any:
    from fcpg.lewm_counterfactual_v3_exploratory import (
        run_exploratory_actual_return_crossfit,
    )

    return run_exploratory_actual_return_crossfit(
        bundle,
        device=DEVICE,
        direct_steps=DIRECT_STEPS_PER_FOLD,
        residual_steps=RESIDUAL_STEPS,
    )


def _result_report(result: Any) -> dict[str, Any]:
    value = getattr(result, "report", None)
    if not isinstance(value, dict):
        raise TypeError("cross-fit runner must return an object with a dict report")
    json.dumps(value, allow_nan=False)
    return dict(value)


def _write_report(destination: Path, report: dict[str, Any]) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.confirm_exploratory_crossfit:
        raise SystemExit(
            "refusing run without --confirm-exploratory-crossfit"
        )

    dataset_path = _resolve(args.dataset)
    destination = _resolve(args.output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"cross-fit output already exists: {destination}")

    dataset = load_counterfactual_v2_store(dataset_path)
    bundle = prepare_gate_bundle(
        dataset.deployment_inputs,
        dataset.offline_labels,
        dataset.audit_arrays,
        dataset.index_rows,
    )
    result = _run_model(bundle)
    scientific_result = _result_report(result)
    report = {
        **scientific_result,
        "classification": "post_hoc_exploratory",
        "evaluation_design": "four_fold_physical_episode_crossfit",
        "post_hoc": True,
        "exposed_validation": True,
        "descriptive_only": True,
        "formal_claim_allowed": False,
        "crossfit_folds": CROSSFIT_FOLDS,
        "device": DEVICE,
        "direct_steps_per_fold": DIRECT_STEPS_PER_FOLD,
        "residual_steps": RESIDUAL_STEPS,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "source_dataset": str(dataset_path),
        "output_directory": str(destination),
        "claim_boundary": (
            "Four-fold estimates reuse exposed v3 data and are descriptive "
            "only. They cannot reverse formal P0 or authorize VisualOracle, "
            "LeWM, GoalToken, FCPG planning, or CEM."
        ),
    }
    _write_report(destination, report)
    print(f"destination: {destination}")
    print("evaluation_design: four_fold_physical_episode_crossfit")
    print("descriptive_only: true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
