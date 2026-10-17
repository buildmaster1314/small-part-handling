#!/usr/bin/env python3
"""Score the frozen A3 2k development action-branch screen."""

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

from fcpg.lewm_action_branch_a1 import load_a1_npz  # noqa: E402
from fcpg.lewm_action_branch_a3_collection import (  # noqa: E402
    validate_a3_cache_conformance,
    validate_a3_contract,
)
from fcpg.lewm_action_branch_a3_metrics import (  # noqa: E402
    CHECKPOINT_STEP,
    PROTOCOL,
    TASKS,
    TRAINING_SEEDS,
    summarize_a3_metrics,
    validate_a3_scoring_contract,
    validate_shuffled_prediction_replay,
)
from fcpg.lewm_action_branch_metrics import (  # noqa: E402
    FACTUAL_METHOD,
    HOLD_LAST_METHOD,
    LATENT_CONSTANT_VELOCITY_METHOD,
    NO_ACTION_METHOD,
    SHUFFLED_ACTION_METHOD,
    make_all_cyclic_branch_shifts,
    validate_all_cyclic_branch_shifts,
)


DEFAULT_CANONICAL = (
    ROOT / "outputs/lewm_action_branch_a3/action_branch_a3_screen.npz"
)
DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_physical_support_a3.json"
DEFAULT_OUTPUT = ROOT / "outputs/lewm_action_branch_a3/a3d_screen_gate"
EXPECTED_TRAIN_EPISODES = 24
EXPECTED_DEVELOPMENT_EPISODES = 12


def _resolve(path: str | Path) -> Path:
    source = Path(path).expanduser()
    return source.resolve() if source.is_absolute() else (ROOT / source).resolve()


def _scalar(archive: Any, name: str) -> Any:
    if name not in archive.files:
        raise ValueError(f"A3 prediction NPZ is missing {name}")
    value = np.asarray(archive[name])
    if value.shape != ():
        raise ValueError(f"A3 prediction field {name} must be scalar")
    return value.item()


def _required_array(archive: Any, name: str) -> np.ndarray:
    if name not in archive.files:
        raise ValueError(f"A3 prediction NPZ is missing {name}")
    return np.array(archive[name], copy=True)


def _finite_array(archive: Any, name: str) -> np.ndarray:
    value = _required_array(archive, name)
    if value.dtype == np.bool_ or not np.issubdtype(value.dtype, np.number):
        raise TypeError(f"A3 prediction field {name} must be numeric")
    if not np.isfinite(value).all():
        raise ValueError(f"A3 prediction field {name} must be finite")
    return value


def _read_checked_in_scoring_contract() -> dict[str, Any]:
    if not DEFAULT_CONFIG.is_file():
        raise FileNotFoundError(DEFAULT_CONFIG)
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("checked-in A3 config must contain a JSON object")
    validate_a3_contract(payload)
    validate_a3_scoring_contract(payload, phase="development")
    return payload


def _validate_prediction_headers(
    paths: Sequence[str | Path],
    *,
    expected_seeds: set[int] | None = None,
) -> dict[int, Path]:
    """Validate fixed A1-trainer endpoints and return them indexed by seed."""

    required_seeds = set(TRAINING_SEEDS if expected_seeds is None else expected_seeds)
    sources = [_resolve(path) for path in paths]
    if len(sources) != len(required_seeds):
        raise ValueError("A3 scoring requires exactly one NPZ per frozen seed")
    by_seed: dict[int, Path] = {}
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        with np.load(source, allow_pickle=False) as archive:
            if str(_scalar(archive, "protocol")) != PROTOCOL:
                raise ValueError(f"{source}: prediction protocol is not A3")
            if int(_scalar(archive, "checkpoint_step")) != CHECKPOINT_STEP:
                raise ValueError(
                    f"{source}: A3 scoring requires latest step {CHECKPOINT_STEP}"
                )
            if str(_scalar(archive, "selection")) != "fixed_latest":
                raise ValueError(f"{source}: A3 scoring requires fixed_latest")
            add_sigreg = _scalar(archive, "prediction_heads_add_sigreg")
            if not isinstance(add_sigreg, (bool, np.bool_)):
                raise TypeError(
                    f"{source}: prediction_heads_add_sigreg must be boolean"
                )
            if bool(add_sigreg):
                raise ValueError(f"{source}: A3 prediction heads must not add SIGReg")
            training_seed = _required_array(archive, "training_seed")
            if (
                training_seed.ndim != 1
                or training_seed.dtype == np.bool_
                or not np.issubdtype(training_seed.dtype, np.integer)
                or training_seed.size == 0
            ):
                raise TypeError(
                    f"{source}: training_seed must be a non-empty integer vector"
                )
            observed = np.unique(training_seed.astype(np.int64, copy=False))
            if observed.size != 1:
                raise ValueError(f"{source}: prediction NPZ mixes training seeds")
            seed = int(observed[0])
            if seed in by_seed:
                raise ValueError(f"A3 prediction seed {seed} is duplicated")
            by_seed[seed] = source
    if set(by_seed) != required_seeds:
        raise ValueError(
            f"A3 prediction seeds must be {sorted(required_seeds)}, "
            f"got {sorted(by_seed)}"
        )
    return by_seed


def _aligned_file_predictions(
    source: Path,
    canonical: Mapping[str, np.ndarray],
    *,
    canonical_development_rows: np.ndarray,
    expected_seed: int,
) -> dict[str, np.ndarray]:
    development_count = int(canonical_development_rows.size)
    canonical_position = {
        int(sample_index): position
        for position, sample_index in enumerate(canonical_development_rows.tolist())
    }
    with np.load(source, allow_pickle=False) as archive:
        split = _required_array(archive, "split").astype(str)
        if split.shape != (development_count,) or set(split.tolist()) != {
            "development"
        }:
            if np.any(split == "test"):
                raise ValueError(f"{source}: A3 scoring refuses prediction test rows")
            raise ValueError(
                f"{source}: A3 prediction rows must be development-only"
            )
        training_seed = _required_array(archive, "training_seed")
        seed = _required_array(archive, "seed")
        expected_seed_vector = np.full(
            development_count, expected_seed, dtype=np.int64
        )
        if not np.array_equal(training_seed, expected_seed_vector) or not np.array_equal(
            seed, expected_seed_vector
        ):
            raise ValueError(f"{source}: row seed metadata is inconsistent")
        sample_index = _required_array(archive, "sample_index")
        if (
            sample_index.shape != (development_count,)
            or sample_index.dtype == np.bool_
            or not np.issubdtype(sample_index.dtype, np.integer)
        ):
            raise TypeError(f"{source}: sample_index must be an integer vector")
        indices = sample_index.astype(np.int64, copy=False)
        if len(set(indices.tolist())) != development_count or set(
            indices.tolist()
        ) != set(canonical_position):
            raise ValueError(
                f"{source}: prediction rows do not cover canonical development rows"
            )
        file_rows = np.empty(development_count, dtype=np.int64)
        for file_row, sample_id in enumerate(indices.tolist()):
            file_rows[canonical_position[int(sample_id)]] = file_row

        aligned_indices = indices[file_rows]
        if not np.array_equal(aligned_indices, canonical_development_rows):
            raise RuntimeError("A3 prediction row alignment failed")
        metadata_pairs = (
            ("split", "split"),
            ("task_index", "task_index"),
            ("task_instance_index", "task_instance_index"),
            ("source_episode_id", "source_episode_id"),
            ("hazard_id", "hazard_id"),
            ("history", "history_embedding"),
            ("history_action", "history_action"),
            ("center_action", "center_action"),
            ("future_action", "future_action"),
            ("target", "target_embedding"),
        )
        for prediction_name, canonical_name in metadata_pairs:
            observed = _required_array(archive, prediction_name)[file_rows]
            expected = np.asarray(canonical[canonical_name])[
                canonical_development_rows
            ]
            if not np.array_equal(observed, expected):
                raise ValueError(
                    f"{source}: {prediction_name} differs from canonical A3 data"
                )

        permutation = _required_array(archive, "shuffle_permutation")[file_rows]
        validate_all_cyclic_branch_shifts(
            permutation,
            bundle_count=development_count,
            branch_count=8,
        )
        expected_permutation = make_all_cyclic_branch_shifts(
            development_count, branch_count=8
        )
        if not np.array_equal(permutation, expected_permutation):
            raise ValueError(
                f"{source}: shuffled action shifts are not in frozen donor order"
            )
        donor_shifts = _required_array(archive, "shuffled_action_donor_shifts")
        if not np.array_equal(donor_shifts, np.arange(1, 8, dtype=np.int64)):
            raise ValueError(
                f"{source}: shuffled_action_donor_shifts must be 1 through 7"
            )
        factual = _finite_array(archive, "factual_prediction")[file_rows]
        no_action = _finite_array(archive, "no_action_prediction")[file_rows]
        shuffled = _finite_array(
            archive, "shuffled_action_prediction"
        )[file_rows]
        validate_shuffled_prediction_replay(
            factual,
            shuffled,
            permutation,
            label=str(source),
        )
        return {
            FACTUAL_METHOD: factual,
            NO_ACTION_METHOD: no_action,
            SHUFFLED_ACTION_METHOD: shuffled,
        }


def load_a3_prediction_panel(
    canonical_path: str | Path,
    prediction_paths: Sequence[str | Path],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load canonical A3 rows and three row-aligned A1 prediction archives."""

    canonical = load_a1_npz(_resolve(canonical_path))
    split = np.asarray(canonical["split"]).astype(str)
    if np.any(split == "test"):
        raise ValueError("A3 scoring refuses canonical test rows")
    if set(split.tolist()) != {"train", "development"}:
        raise ValueError("canonical A3 scoring requires train/development only")
    development_rows = np.flatnonzero(split == "development")
    by_seed = _validate_prediction_headers(prediction_paths)
    per_seed = [
        _aligned_file_predictions(
            by_seed[seed],
            canonical,
            canonical_development_rows=development_rows,
            expected_seed=seed,
        )
        for seed in TRAINING_SEEDS
    ]
    predictions = {
        method: np.stack([row[method] for row in per_seed], axis=0)
        for method in (FACTUAL_METHOD, NO_ACTION_METHOD, SHUFFLED_ACTION_METHOD)
    }
    return canonical, predictions


def _validate_formal_episode_coverage(
    canonical: Mapping[str, np.ndarray],
) -> None:
    task = np.asarray(canonical["task_index"], dtype=np.int64)
    episode = np.asarray(canonical["source_episode_id"]).astype(str)
    split = np.asarray(canonical["split"]).astype(str)
    for task_index, task_name in enumerate(TASKS):
        observed = {
            split_name: int(
                np.unique(
                    episode[(task == task_index) & (split == split_name)]
                ).size
            )
            for split_name in ("train", "development")
        }
        expected = {
            "train": EXPECTED_TRAIN_EPISODES,
            "development": EXPECTED_DEVELOPMENT_EPISODES,
        }
        if observed != expected:
            raise ValueError(
                f"A3 {task_name} episode coverage must be {expected}, got {observed}"
            )
    validate_a3_cache_conformance(
        canonical,
        _read_checked_in_scoring_contract(),
    )


def _percent(value: Any) -> str:
    return f"{float(value):.2%}"


def render_markdown(summary: Mapping[str, Any]) -> str:
    comparisons = summary["comparisons"]
    shifts = summary["shuffled_action_shifts"]
    raw = summary["raw_h6_h8"]
    gate = summary["gate"]
    lines = [
        "# A3 matched action-branch 2k development screen",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "| H6-H8 comparison | Relative improvement | Positive tasks | Positive seeds |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method, label in (
        (NO_ACTION_METHOD, "Matched no-action"),
        (SHUFFLED_ACTION_METHOD, "Seven-shift average"),
    ):
        row = comparisons[method]
        lines.append(
            f"| {label} | {_percent(row['effect'])} | "
            f"{row['positive_task_count']}/4 | {row['positive_seed_count']}/3 |"
        )
    lines.extend(
        [
            "",
            "| Wrong-action donor shift | Relative improvement |",
            "| ---: | ---: |",
        ]
    )
    for row in shifts["per_shift"]:
        lines.append(f"| {row['donor_shift']} | {_percent(row['effect'])} |")
    lines.extend(
        [
            "",
            "| Raw H6-H8 method | Task-macro whitened MSE |",
            "| --- | ---: |",
        ]
    )
    for method in (
        FACTUAL_METHOD,
        NO_ACTION_METHOD,
        HOLD_LAST_METHOD,
        LATENT_CONSTANT_VELOCITY_METHOD,
    ):
        lines.append(
            f"| {method} | {raw['methods'][method]['task_macro_mse']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Cross-replicating seeds (positive on at least 3/4 tasks in both "
            "primary comparisons): "
            f"{len(gate['cross_replicating_seeds'])}/3",
        ]
    )
    if gate["failed_checks"]:
        lines.extend(
            ["", "Failed checks: " + ", ".join(gate["failed_checks"])]
        )
    lines.extend(
        [
            "",
            "Whitening uses canonical train rows, separately per task and with "
            "equal physical-episode weight. This decision uses development only; "
            "reserved test rows are not accepted.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical",
        "--input",
        "--input-npz",
        "--train-data",
        dest="canonical",
        type=Path,
        default=DEFAULT_CANONICAL,
    )
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output", "--output-dir", dest="output", type=Path, default=DEFAULT_OUTPUT
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    canonical, predictions = load_a3_prediction_panel(
        args.canonical, args.predictions
    )
    _validate_formal_episode_coverage(canonical)
    summary = summarize_a3_metrics(canonical, predictions)
    output = _resolve(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "summary.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )
    print(json.dumps({"decision": summary["decision"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
