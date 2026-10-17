#!/usr/bin/env python3
"""Score the A4 factual-action 2k development panel."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a4 import (  # noqa: E402
    DEVELOPMENT_TRAINING_SEEDS,
    PROTOCOL,
)
from fcpg.lewm_action_branch_a4_metrics import (  # noqa: E402
    FACTUAL_METHOD,
    FORMAL_MODE,
    NO_ACTION_METHOD,
    SMOKE_MODE,
    WRONG_ACTION_METHOD,
    summarize_a4_development,
    validate_a4_scoring_contract,
)


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_active_control_a4.json"
DEFAULT_OUTPUT = ROOT / "outputs/lewm_action_branch_a4/development_gate"


def _resolve(path: str | Path) -> Path:
    source = Path(path).expanduser()
    return source.resolve() if source.is_absolute() else (ROOT / source).resolve()


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(_resolve(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("A4 config must contain a JSON object")
    return value


def _file_binding(path: str | Path) -> dict[str, str]:
    source = _resolve(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(source), "sha256": digest.hexdigest()}


def _load_npz_without_reserved_test(
    path: str | Path, *, label: str
) -> dict[str, np.ndarray]:
    source = _resolve(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as archive:
        if "split" not in archive.files:
            raise ValueError(f"{label} is missing split metadata")
        split = np.asarray(archive["split"])
        if split.ndim != 1 or split.dtype.kind not in {"U", "S"}:
            raise TypeError(f"{label} split must be a string vector")
        names = split.astype(str, copy=False)
        if np.any(names == "test"):
            raise PermissionError(
                f"{label} contains reserved test rows; development is not authorized to inspect them"
            )
        if not set(names.tolist()).issubset({"train", "development"}):
            raise ValueError(f"{label} contains unsupported split labels")
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _scalar(arrays: Mapping[str, np.ndarray], name: str) -> Any:
    if name not in arrays:
        raise ValueError(f"A4 prediction archive is missing {name}")
    value = np.asarray(arrays[name])
    if value.shape != ():
        raise ValueError(f"A4 prediction field {name} must be scalar")
    return value.item()


def _seed(arrays: Mapping[str, np.ndarray]) -> int:
    if "training_seed" not in arrays:
        raise ValueError("A4 prediction archive is missing training_seed")
    value = np.asarray(arrays["training_seed"])
    if value.dtype == np.bool_ or not np.issubdtype(value.dtype, np.integer):
        raise TypeError("training_seed must be integer")
    observed = np.unique(value.astype(np.int64, copy=False))
    if observed.size != 1:
        raise ValueError("one A4 prediction archive must contain one training seed")
    return int(observed[0])


def load_a4_prediction_panel(
    cache_path: str | Path,
    prediction_paths: Sequence[str | Path],
    *,
    mode: str = FORMAL_MODE,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], tuple[int, ...]]:
    cache = _load_npz_without_reserved_test(cache_path, label="A4 state cache")
    split = np.asarray(cache["split"]).astype(str)
    development_rows = np.flatnonzero(split == "development")
    expected_ids = np.asarray(cache["proposal_id"])[development_rows].astype(str)
    if development_rows.size == 0:
        raise ValueError("A4 state cache has no development rows")

    by_seed: dict[int, dict[str, np.ndarray]] = {}
    for path in prediction_paths:
        arrays = _load_npz_without_reserved_test(path, label="A4 prediction archive")
        if str(_scalar(arrays, "protocol")) != PROTOCOL:
            raise ValueError("A4 prediction protocol differs from the frozen contract")
        step = int(_scalar(arrays, "checkpoint_step"))
        if mode == FORMAL_MODE and step != 2_000:
            raise ValueError("formal A4 development requires checkpoint step 2000")
        if mode == SMOKE_MODE and not 0 <= step <= 2_000:
            raise ValueError("A4 smoke checkpoint step must be in 0..2000")
        seed = _seed(arrays)
        if seed in by_seed:
            raise ValueError(f"duplicate A4 prediction seed {seed}")
        observed_split = np.asarray(arrays["split"]).astype(str)
        if observed_split.shape != (development_rows.size,) or set(
            observed_split.tolist()
        ) != {"development"}:
            raise ValueError("A4 prediction rows must be development-only")
        proposal_id = np.asarray(arrays.get("proposal_id", ())).astype(str)
        if proposal_id.shape != expected_ids.shape or set(proposal_id.tolist()) != set(
            expected_ids.tolist()
        ):
            raise ValueError("A4 predictions must cover every development proposal_id")
        position = {value: index for index, value in enumerate(proposal_id.tolist())}
        order = np.asarray([position[value] for value in expected_ids.tolist()])
        pairs = (
            ("history_state", "history_state"),
            ("future_action", "future_action"),
            ("target_state", "target_state"),
        )
        for prediction_name, cache_name in pairs:
            if prediction_name not in arrays:
                raise ValueError(f"A4 prediction archive is missing {prediction_name}")
            if not np.array_equal(
                np.asarray(arrays[prediction_name])[order],
                np.asarray(cache[cache_name])[development_rows],
            ):
                raise ValueError(f"A4 prediction {prediction_name} differs from the cache")
        required = {
            FACTUAL_METHOD: ("factual_prediction",),
            NO_ACTION_METHOD: ("no_action_prediction",),
            WRONG_ACTION_METHOD: (
                "shuffled_action_prediction",
                "wrong_action_prediction",
            ),
        }
        by_seed[seed] = {}
        for method, aliases in required.items():
            present = [name for name in aliases if name in arrays]
            if len(present) != 1:
                raise ValueError(
                    f"A4 prediction archive must contain exactly one {method} prediction"
                )
            by_seed[seed][method] = np.asarray(arrays[present[0]])[order]

    seeds = tuple(sorted(by_seed))
    if mode == FORMAL_MODE and seeds != DEVELOPMENT_TRAINING_SEEDS:
        raise ValueError(f"formal A4 development seeds must be {DEVELOPMENT_TRAINING_SEEDS}")
    if mode == SMOKE_MODE and (
        not seeds or not set(seeds).issubset(DEVELOPMENT_TRAINING_SEEDS)
    ):
        raise ValueError("A4 smoke requires a non-empty subset of frozen seeds")
    predictions = {
        method: np.stack([by_seed[seed][method] for seed in seeds], axis=0)
        for method in (FACTUAL_METHOD, NO_ACTION_METHOD, WRONG_ACTION_METHOD)
    }
    return cache, predictions, seeds


def render_markdown(summary: Mapping[str, Any]) -> str:
    primary = summary["primary"]
    raw = summary["raw_h6_h8"]["methods"]
    lines = [
        "# A4 factual-action development score",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "| comparison | relative improvement |",
        "| --- | ---: |",
        f"| matched no-action | {primary['comparisons']['no_action']['relative_improvement']:.2%} |",
        f"| seven wrong-action shifts | {primary['comparisons']['wrong_action_mean']['relative_improvement']:.2%} |",
        "",
        "| raw method | task-macro MSE |",
        "| --- | ---: |",
    ]
    for method, row in raw.items():
        lines.append(f"| {method} | {row['task_macro_mse']:.6f} |")
    if summary["gate"]["failed_checks"]:
        lines.extend(["", "Failed checks: " + ", ".join(summary["gate"]["failed_checks"])])
    lines.extend(["", "Reserved test was not opened by this scorer."])
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = _load_json(args.config)
    validate_a4_scoring_contract(contract, phase="development")
    mode = SMOKE_MODE if args.smoke else FORMAL_MODE
    cache, predictions, seeds = load_a4_prediction_panel(
        args.cache, args.predictions, mode=mode
    )
    summary = summarize_a4_development(
        cache,
        predictions,
        contract=contract,
        training_seeds=seeds,
        mode=mode,
    )
    if mode == FORMAL_MODE:
        summary["formal_train_cache"] = _file_binding(args.cache)
    output = _resolve(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"decision": summary["decision"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
