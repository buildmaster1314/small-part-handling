#!/usr/bin/env python3
"""Score the frozen A3 10k test-only confirmation panel."""

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
from fcpg.lewm_action_branch_a3_confirmation import (  # noqa: E402
    validate_a3_confirmation_canonical_conformance,
    validate_a3_confirmation_contract,
)
from fcpg.lewm_action_branch_a3_metrics import (  # noqa: E402
    CONFIRMATION_CHECKPOINT_STEP,
    CONFIRMATION_TRAINING_SEEDS,
    PROTOCOL,
    TASKS,
    summarize_a3_confirmation,
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


CHECKPOINT_STEP = CONFIRMATION_CHECKPOINT_STEP
TRAINING_SEEDS = CONFIRMATION_TRAINING_SEEDS
DEFAULT_CANONICAL = (
    ROOT / "outputs/lewm_action_branch_a3/action_branch_a3_confirmation.npz"
)
DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_physical_support_a3.json"
DEFAULT_OUTPUT = ROOT / "outputs/lewm_action_branch_a3/a3c_confirmation_gate"
EXPECTED_TRAIN_EPISODES = 24
EXPECTED_TEST_EPISODES = 12


def _resolve(path: str | Path) -> Path:
    source = Path(path).expanduser()
    return source.resolve() if source.is_absolute() else (ROOT / source).resolve()


def _scalar(archive: Any, name: str) -> Any:
    if name not in archive.files:
        raise ValueError(f"A3 confirmation prediction NPZ is missing {name}")
    value = np.asarray(archive[name])
    if value.shape != ():
        raise ValueError(f"A3 confirmation prediction field {name} must be scalar")
    return value.item()


def _required_array(archive: Any, name: str) -> np.ndarray:
    if name not in archive.files:
        raise ValueError(f"A3 confirmation prediction NPZ is missing {name}")
    return np.array(archive[name], copy=True)


def _finite_array(archive: Any, name: str) -> np.ndarray:
    value = _required_array(archive, name)
    if value.dtype == np.bool_ or not np.issubdtype(value.dtype, np.number):
        raise TypeError(f"A3 confirmation prediction field {name} must be numeric")
    if not np.isfinite(value).all():
        raise ValueError(f"A3 confirmation prediction field {name} must be finite")
    return value


def _read_checked_in_scoring_contract() -> dict[str, Any]:
    if not DEFAULT_CONFIG.is_file():
        raise FileNotFoundError(DEFAULT_CONFIG)
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("checked-in A3 config must contain a JSON object")
    validate_a3_confirmation_contract(payload)
    validate_a3_scoring_contract(payload, phase="confirmation")
    return payload


def _validate_prediction_headers(
    paths: Sequence[str | Path],
) -> dict[int, Path]:
    """Validate the three fixed 10k endpoints and index them by seed."""

    sources = [_resolve(path) for path in paths]
    if len(sources) != len(TRAINING_SEEDS):
        raise ValueError(
            "A3 confirmation requires exactly one prediction NPZ per frozen seed"
        )
    by_seed: dict[int, Path] = {}
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        with np.load(source, allow_pickle=False) as archive:
            if str(_scalar(archive, "protocol")) != PROTOCOL:
                raise ValueError(f"{source}: prediction protocol is not A3")
            if int(_scalar(archive, "checkpoint_step")) != CHECKPOINT_STEP:
                raise ValueError(
                    f"{source}: A3 confirmation requires latest step "
                    f"{CHECKPOINT_STEP}"
                )
            if str(_scalar(archive, "selection")) != "fixed_latest":
                raise ValueError(
                    f"{source}: A3 confirmation requires fixed_latest"
                )
            add_sigreg = _scalar(archive, "prediction_heads_add_sigreg")
            if not isinstance(add_sigreg, (bool, np.bool_)):
                raise TypeError(
                    f"{source}: prediction_heads_add_sigreg must be boolean"
                )
            if bool(add_sigreg):
                raise ValueError(
                    f"{source}: A3 confirmation prediction heads must not add SIGReg"
                )
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
                raise ValueError(f"A3 confirmation prediction seed {seed} is duplicated")
            by_seed[seed] = source
    if set(by_seed) != set(TRAINING_SEEDS):
        raise ValueError(
            f"A3 confirmation prediction seeds must be {list(TRAINING_SEEDS)}, "
            f"got {sorted(by_seed)}"
        )
    return by_seed


def _aligned_file_predictions(
    source: Path,
    canonical: Mapping[str, np.ndarray],
    *,
    canonical_test_rows: np.ndarray,
    expected_seed: int,
) -> dict[str, np.ndarray]:
    test_count = int(canonical_test_rows.size)
    canonical_position = {
        int(sample_index): position
        for position, sample_index in enumerate(canonical_test_rows.tolist())
    }
    with np.load(source, allow_pickle=False) as archive:
        split = _required_array(archive, "split").astype(str)
        if split.shape != (test_count,) or set(split.tolist()) != {"test"}:
            if np.any(split == "development"):
                raise ValueError(
                    f"{source}: A3 confirmation refuses prediction development rows"
                )
            raise ValueError(
                f"{source}: A3 confirmation prediction rows must be test-only"
            )

        expected_seed_vector = np.full(test_count, expected_seed, dtype=np.int64)
        training_seed = _required_array(archive, "training_seed")
        seed = _required_array(archive, "seed")
        if not np.array_equal(training_seed, expected_seed_vector) or not np.array_equal(
            seed, expected_seed_vector
        ):
            raise ValueError(f"{source}: row seed metadata is inconsistent")

        sample_index = _required_array(archive, "sample_index")
        if (
            sample_index.shape != (test_count,)
            or sample_index.dtype == np.bool_
            or not np.issubdtype(sample_index.dtype, np.integer)
        ):
            raise TypeError(f"{source}: sample_index must be an integer vector")
        indices = sample_index.astype(np.int64, copy=False)
        if len(set(indices.tolist())) != test_count or set(indices.tolist()) != set(
            canonical_position
        ):
            raise ValueError(
                f"{source}: prediction rows do not cover canonical test rows"
            )
        file_rows = np.empty(test_count, dtype=np.int64)
        for file_row, sample_id in enumerate(indices.tolist()):
            file_rows[canonical_position[int(sample_id)]] = file_row
        if not np.array_equal(indices[file_rows], canonical_test_rows):
            raise RuntimeError("A3 confirmation prediction row alignment failed")

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
            expected = np.asarray(canonical[canonical_name])[canonical_test_rows]
            if not np.array_equal(observed, expected):
                raise ValueError(
                    f"{source}: {prediction_name} differs from canonical A3 data"
                )

        permutation = _required_array(archive, "shuffle_permutation")[file_rows]
        validate_all_cyclic_branch_shifts(
            permutation,
            bundle_count=test_count,
            branch_count=8,
        )
        expected_permutation = make_all_cyclic_branch_shifts(
            test_count, branch_count=8
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


def load_a3_confirmation_prediction_panel(
    canonical_path: str | Path,
    prediction_paths: Sequence[str | Path],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load canonical train/test rows and the three aligned test predictions."""

    canonical = load_a1_npz(_resolve(canonical_path))
    split = np.asarray(canonical["split"]).astype(str)
    if np.any(split == "development"):
        raise ValueError("A3 confirmation refuses canonical development rows")
    if set(split.tolist()) != {"train", "test"}:
        raise ValueError("A3 confirmation canonical must contain train/test only")
    test_rows = np.flatnonzero(split == "test")
    by_seed = _validate_prediction_headers(prediction_paths)
    per_seed = [
        _aligned_file_predictions(
            by_seed[seed],
            canonical,
            canonical_test_rows=test_rows,
            expected_seed=seed,
        )
        for seed in TRAINING_SEEDS
    ]
    predictions = {
        method: np.stack([row[method] for row in per_seed], axis=0)
        for method in (FACTUAL_METHOD, NO_ACTION_METHOD, SHUFFLED_ACTION_METHOD)
    }
    return canonical, predictions


load_a3_prediction_panel = load_a3_confirmation_prediction_panel


def _validate_formal_episode_coverage(
    canonical: Mapping[str, np.ndarray],
) -> None:
    task = np.asarray(canonical["task_index"], dtype=np.int64)
    episode = np.asarray(canonical["source_episode_id"]).astype(str)
    split = np.asarray(canonical["split"]).astype(str)
    if set(split.tolist()) != {"train", "test"}:
        raise ValueError(
            "A3 confirmation canonical must contain train/test and no development"
        )
    if set(task.tolist()) != set(range(len(TASKS))):
        raise ValueError("A3 confirmation canonical must cover the four frozen tasks")
    expected = {
        "train": EXPECTED_TRAIN_EPISODES,
        "test": EXPECTED_TEST_EPISODES,
    }
    for task_index, task_name in enumerate(TASKS):
        observed = {
            split_name: int(
                np.unique(
                    episode[(task == task_index) & (split == split_name)]
                ).size
            )
            for split_name in ("train", "test")
        }
        if observed != expected:
            raise ValueError(
                f"A3 confirmation {task_name} episode coverage must be "
                f"{expected}, got {observed}"
            )
    validate_a3_confirmation_canonical_conformance(
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
        "# A3 matched action-branch 10k confirmation",
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
            f"Jointly positive tasks: {len(gate['jointly_positive_tasks'])}/4",
            "",
            f"Jointly positive seeds: {len(gate['jointly_positive_seeds'])}/3",
            "",
            "Cross-replicating seeds (positive on at least 3/4 tasks in both "
            "primary comparisons): "
            f"{len(gate['cross_replicating_seeds'])}/3",
        ]
    )
    if gate["failed_checks"]:
        lines.extend(["", "Failed checks: " + ", ".join(gate["failed_checks"])])
    lines.extend(
        [
            "",
            "Whitening uses canonical train rows only, separately per task and "
            "with equal physical-episode weight. The three fixed_latest step-10000 "
            "endpoints are scored on canonical test rows only.",
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
    canonical, predictions = load_a3_confirmation_prediction_panel(
        args.canonical, args.predictions
    )
    _validate_formal_episode_coverage(canonical)
    summary = summarize_a3_confirmation(canonical, predictions)
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
