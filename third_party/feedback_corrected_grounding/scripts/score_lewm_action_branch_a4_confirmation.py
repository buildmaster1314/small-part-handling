#!/usr/bin/env python3
"""Score the authorized A4 fixed-10k reserved-test confirmation panel."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a4 import (  # noqa: E402
    BRANCH_COUNT,
    CONFIRMATION_TRAINING_SEEDS,
    PROTOCOL,
    SIGREG_WEIGHT,
)
from fcpg.lewm_a4_checkpoint import A4_CHECKPOINT_TYPE  # noqa: E402
from fcpg.lewm_action_branch_a4_confirmation_metrics import (  # noqa: E402
    CONFIRMATION_CHECKPOINT_STEP,
    summarize_a4_confirmation,
    validate_a4_confirmation_caches,
    validate_a4_confirmation_contract,
    validate_a4_development_go_authorization,
)
from fcpg.lewm_action_branch_a4_confirmation_collection import (  # noqa: E402
    validate_a4_development_go_summary,
)
from fcpg.lewm_action_branch_a4_metrics import (  # noqa: E402
    FACTUAL_METHOD,
    HOLD_LAST_METHOD,
    LATENT_CONSTANT_VELOCITY_METHOD,
    NO_ACTION_METHOD,
    WRONG_ACTION_METHOD,
    _secondary_predictions,
    validate_wrong_action_prediction_replay,
)


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_active_control_a4.json"
DEFAULT_OUTPUT = ROOT / "outputs/lewm_action_branch_a4/confirmation_gate"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


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


def _load_npz(path: str | Path, *, label: str) -> dict[str, np.ndarray]:
    source = _resolve(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(arrays: Mapping[str, np.ndarray], name: str) -> Any:
    if name not in arrays:
        raise ValueError(f"A4 confirmation prediction archive is missing {name}")
    value = np.asarray(arrays[name])
    if value.shape != ():
        raise ValueError(f"A4 confirmation prediction field {name} must be scalar")
    return value.item()


def _required(arrays: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    if name not in arrays:
        raise ValueError(f"A4 confirmation prediction archive is missing {name}")
    return np.asarray(arrays[name])


def _finite(arrays: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    value = _required(arrays, name)
    if value.dtype == np.bool_ or not np.issubdtype(value.dtype, np.number):
        raise TypeError(f"A4 confirmation prediction {name} must be numeric")
    if not np.isfinite(value).all():
        raise ValueError(f"A4 confirmation prediction {name} must be finite")
    return value


def _validate_prediction_header(
    arrays: Mapping[str, np.ndarray], *, expected_seed: int
) -> tuple[str, str]:
    if str(_scalar(arrays, "protocol")) != PROTOCOL:
        raise ValueError("A4 confirmation prediction protocol drifted")
    if int(_scalar(arrays, "schema_version")) != 1:
        raise ValueError("A4 confirmation prediction schema_version must be 1")
    if int(_scalar(arrays, "checkpoint_step")) != CONFIRMATION_CHECKPOINT_STEP:
        raise ValueError("A4 confirmation requires checkpoint step 10000")
    if str(_scalar(arrays, "selection")) != "fixed_latest":
        raise ValueError("A4 confirmation forbids checkpoint selection")
    if float(_scalar(arrays, "anchor_encoder_sigreg")) != SIGREG_WEIGHT:
        raise ValueError("A4 confirmation anchor encoder SIGReg must be 0.02")
    add_sigreg = _scalar(arrays, "prediction_heads_add_sigreg")
    if not isinstance(add_sigreg, (bool, np.bool_)):
        raise TypeError("prediction_heads_add_sigreg must be boolean")
    if bool(add_sigreg):
        raise ValueError("A4 confirmation predictor heads must not add SIGReg")
    if "evaluation_split" in arrays and str(
        _scalar(arrays, "evaluation_split")
    ) != "test":
        raise ValueError("A4 confirmation prediction evaluation_split must be test")
    if "primary_horizons" in arrays:
        horizons = _required(arrays, "primary_horizons")
        if horizons.dtype == np.bool_ or not np.issubdtype(horizons.dtype, np.integer):
            raise TypeError("A4 confirmation primary_horizons must be integers")
        if not np.array_equal(
            horizons.astype(np.int64, copy=False),
            np.asarray([6, 7, 8], dtype=np.int64),
        ):
            raise ValueError("A4 confirmation predictions must declare H6-H8")

    if str(_scalar(arrays, "checkpoint_type")) != A4_CHECKPOINT_TYPE:
        raise ValueError("A4 confirmation checkpoint_type drifted")
    checkpoint_seed = _scalar(arrays, "checkpoint_seed")
    if isinstance(checkpoint_seed, (bool, np.bool_)) or not isinstance(
        checkpoint_seed, (int, np.integer)
    ):
        raise TypeError("A4 confirmation checkpoint_seed must be an integer")
    if int(checkpoint_seed) != expected_seed:
        raise ValueError(
            f"A4 confirmation checkpoint_seed must equal {expected_seed}"
        )
    checkpoint_path = str(_scalar(arrays, "checkpoint_path"))
    if not checkpoint_path or checkpoint_path != checkpoint_path.strip():
        raise ValueError("A4 confirmation checkpoint_path must be non-empty")
    if not Path(checkpoint_path).is_absolute():
        raise ValueError("A4 confirmation checkpoint_path must be absolute")
    checkpoint_sha256 = str(_scalar(arrays, "checkpoint_sha256"))
    if _SHA256_PATTERN.fullmatch(checkpoint_sha256) is None:
        raise ValueError(
            "A4 confirmation checkpoint_sha256 must be 64 lowercase hex characters"
        )
    for name in ("training_seed", "seed"):
        values = _required(arrays, name)
        if values.ndim != 1 or values.dtype == np.bool_ or not np.issubdtype(
            values.dtype, np.integer
        ):
            raise TypeError(f"A4 confirmation {name} must be an integer vector")
        observed = np.unique(values.astype(np.int64, copy=False))
        if observed.tolist() != [expected_seed]:
            raise ValueError(f"A4 confirmation {name} must contain seed {expected_seed}")
    return checkpoint_path, checkpoint_sha256


def _cyclic_permutations(row_count: int) -> np.ndarray:
    slots = np.arange(BRANCH_COUNT, dtype=np.int64)
    return np.broadcast_to(
        np.stack(
            [np.remainder(slots + shift, BRANCH_COUNT) for shift in range(1, BRANCH_COUNT)],
            axis=0,
        ),
        (row_count, BRANCH_COUNT - 1, BRANCH_COUNT),
    )


def _stored_causal_baselines(history: np.ndarray) -> dict[str, np.ndarray]:
    """Replay the float32 baseline arithmetic used by the evaluator.

    The statistics path intentionally promotes arrays to float64.  Prediction
    archives, however, are produced from ``Tensor.float()`` inputs and store the
    resulting float32 baselines.  Recomputing those receipt fields in float64
    can differ by several float32 ulps for long constant-velocity rollouts.
    """

    values = np.asarray(history)
    if values.dtype == np.bool_ or not np.issubdtype(values.dtype, np.number):
        raise TypeError("A4 confirmation history_state must be numeric")
    if values.ndim != 4 or values.shape[1] < 2:
        raise ValueError("history_state must have shape [N,H,K,D] with H>=2")
    values = values.astype(np.float32, copy=False)
    last = values[:, -1]
    velocity = last - values[:, -2]
    factors = np.arange(1, 9, dtype=np.float32).reshape(1, 1, 8, 1, 1)
    shape = (values.shape[0], BRANCH_COUNT, 8, *last.shape[1:])
    hold = np.broadcast_to(last[:, None, None], shape)
    constant_velocity = np.broadcast_to(
        last[:, None, None] + factors * velocity[:, None, None], shape
    )
    return {
        HOLD_LAST_METHOD: hold,
        LATENT_CONSTANT_VELOCITY_METHOD: constant_velocity,
    }


def _aligned_prediction_archive(
    arrays: Mapping[str, np.ndarray],
    cache: Mapping[str, np.ndarray],
    *,
    expected_seed: int,
) -> dict[str, np.ndarray]:
    _validate_prediction_header(arrays, expected_seed=expected_seed)
    split = np.asarray(cache["split"]).astype(str, copy=False)
    if set(split.tolist()) != {"test"}:
        raise ValueError("A4 confirmation prediction alignment requires a test-only cache")
    row_count = int(split.size)
    test_rows = np.arange(row_count, dtype=np.int64)
    observed_split = _required(arrays, "split").astype(str)
    if observed_split.shape != (row_count,) or set(observed_split.tolist()) != {"test"}:
        if np.any(observed_split == "development"):
            raise PermissionError("A4 confirmation prediction archive contains development rows")
        raise ValueError("A4 confirmation prediction rows must be test-only")
    row_fields = (
        "training_seed",
        "seed",
        "sample_index",
        "proposal_id",
        "task_class_index",
        "task_name",
        "task_instance_index",
        "source_episode_id",
        "split",
        "history_state",
        "history_action",
        "center_action",
        "future_action",
        "target_state",
        "branch_semantic_index",
        "factual_prediction",
        "no_action_prediction",
        "shuffled_action_prediction",
        "hold_last_prediction",
        "constant_velocity_prediction",
        "shuffle_permutation",
    )
    for name in row_fields:
        if _required(arrays, name).ndim == 0 or _required(arrays, name).shape[0] != row_count:
            raise ValueError(
                f"A4 confirmation {name} must have one entry per test row"
            )

    expected_ids = np.asarray(cache["proposal_id"])[test_rows].astype(str)
    proposal_id = _required(arrays, "proposal_id").astype(str)
    if proposal_id.shape != expected_ids.shape or set(proposal_id.tolist()) != set(
        expected_ids.tolist()
    ):
        raise ValueError("A4 confirmation predictions must cover every test proposal_id")
    if len(set(proposal_id.tolist())) != row_count:
        raise ValueError("A4 confirmation prediction proposal_id values must be unique")
    positions = {value: index for index, value in enumerate(proposal_id.tolist())}
    order = np.asarray([positions[value] for value in expected_ids.tolist()], dtype=np.int64)

    sample_index = _required(arrays, "sample_index")
    if sample_index.shape != (row_count,) or sample_index.dtype == np.bool_ or not np.issubdtype(
        sample_index.dtype, np.integer
    ):
        raise TypeError("A4 confirmation sample_index must be an integer vector")
    if not np.array_equal(sample_index[order].astype(np.int64, copy=False), test_rows):
        raise ValueError("A4 confirmation sample_index differs from the cache")

    metadata_pairs = (
        ("task_class_index", "task_class_index"),
        ("task_name", "task_name"),
        ("task_instance_index", "task_instance_index"),
        ("source_episode_id", "source_episode_id"),
        ("history_state", "history_state"),
        ("history_action", "history_action"),
        ("center_action", "center_action"),
        ("future_action", "future_action"),
        ("target_state", "target_state"),
        ("branch_semantic_index", "branch_semantic_index"),
    )
    for prediction_name, cache_name in metadata_pairs:
        observed = _required(arrays, prediction_name)[order]
        expected = np.asarray(cache[cache_name])[test_rows]
        if not np.array_equal(observed, expected):
            raise ValueError(
                f"A4 confirmation prediction {prediction_name} differs from the cache"
            )

    permutation = _required(arrays, "shuffle_permutation")[order]
    expected_permutation = _cyclic_permutations(row_count)
    if permutation.dtype == np.bool_ or not np.issubdtype(permutation.dtype, np.integer):
        raise TypeError("A4 confirmation shuffle_permutation must be integer")
    if not np.array_equal(permutation.astype(np.int64, copy=False), expected_permutation):
        raise ValueError("A4 confirmation wrong-action donor permutations drifted")
    donor_shifts = _required(arrays, "shuffled_action_donor_shifts")
    if not np.array_equal(donor_shifts, np.arange(1, BRANCH_COUNT, dtype=np.int64)):
        raise ValueError("A4 confirmation wrong-action donor shifts must be 1..7")

    factual = _finite(arrays, "factual_prediction")[order]
    no_action = _finite(arrays, "no_action_prediction")[order]
    wrong_action = _finite(arrays, "shuffled_action_prediction")[order]
    expected_prediction_shape = np.asarray(cache["target_state"]).shape
    expected_wrong_shape = (
        row_count,
        BRANCH_COUNT - 1,
        *expected_prediction_shape[1:],
    )
    if factual.shape != expected_prediction_shape:
        raise ValueError(
            "A4 confirmation factual prediction must cover the full H1-H8 test target"
        )
    if no_action.shape != expected_prediction_shape:
        raise ValueError(
            "A4 confirmation no-action prediction must cover the full H1-H8 test target"
        )
    if wrong_action.shape != expected_wrong_shape:
        raise ValueError(
            "A4 confirmation wrong-action prediction must cover all shifts and H1-H8"
        )
    validate_wrong_action_prediction_replay(factual[None], wrong_action[None])

    baselines = _stored_causal_baselines(
        np.asarray(cache["history_state"])[test_rows]
    )
    for archive_name, method in (
        ("hold_last_prediction", HOLD_LAST_METHOD),
        ("constant_velocity_prediction", LATENT_CONSTANT_VELOCITY_METHOD),
    ):
        observed = _finite(arrays, archive_name)[order]
        expected = baselines[method].astype(observed.dtype, copy=False)
        if observed.shape != expected.shape or not np.array_equal(observed, expected):
            raise ValueError(f"A4 confirmation {archive_name} differs from causal baseline")

    return {
        FACTUAL_METHOD: factual,
        NO_ACTION_METHOD: no_action,
        WRONG_ACTION_METHOD: wrong_action,
    }


def _prevalidate_prediction_sources(
    prediction_paths: Sequence[str | Path],
) -> dict[int, dict[str, np.ndarray]]:
    """Validate the complete checkpoint-bound panel before any test-cache read."""

    sources = tuple(_resolve(path) for path in prediction_paths)
    if len(sources) != len(CONFIRMATION_TRAINING_SEEDS):
        raise ValueError("A4 confirmation requires exactly three prediction archives")
    if len(set(sources)) != len(sources):
        raise ValueError("A4 confirmation prediction paths must be distinct")

    by_seed: dict[int, dict[str, np.ndarray]] = {}
    checkpoint_identities: dict[int, tuple[str, str]] = {}
    for source in sources:
        arrays = _load_npz(source, label="A4 confirmation prediction archive")
        training_seed = _required(arrays, "training_seed")
        if (
            training_seed.ndim != 1
            or training_seed.dtype == np.bool_
            or not np.issubdtype(training_seed.dtype, np.integer)
        ):
            raise TypeError("A4 confirmation training_seed must be an integer vector")
        unique = np.unique(training_seed.astype(np.int64, copy=False))
        if unique.size != 1:
            raise ValueError("one A4 confirmation archive must contain one seed")
        seed = int(unique[0])
        if seed not in CONFIRMATION_TRAINING_SEEDS:
            raise ValueError(
                f"A4 confirmation seed must be one of {CONFIRMATION_TRAINING_SEEDS}"
            )
        if seed in by_seed:
            raise ValueError(f"duplicate A4 confirmation seed {seed}")
        checkpoint_path, checkpoint_sha256 = _validate_prediction_header(
            arrays, expected_seed=seed
        )
        checkpoint_source = Path(checkpoint_path)
        if not checkpoint_source.is_file():
            raise FileNotFoundError(checkpoint_source)
        if _sha256_file(checkpoint_source) != checkpoint_sha256:
            raise PermissionError(
                f"A4 confirmation checkpoint content mismatch for seed {seed}"
            )
        checkpoint_identities[seed] = (checkpoint_path, checkpoint_sha256)
        by_seed[seed] = arrays

    if tuple(sorted(by_seed)) != CONFIRMATION_TRAINING_SEEDS:
        raise ValueError(
            f"A4 confirmation seeds must be exactly {CONFIRMATION_TRAINING_SEEDS}"
        )
    checkpoint_paths = [value[0] for value in checkpoint_identities.values()]
    checkpoint_hashes = [value[1] for value in checkpoint_identities.values()]
    if len(set(checkpoint_paths)) != len(CONFIRMATION_TRAINING_SEEDS):
        raise ValueError("A4 confirmation requires one distinct checkpoint path per seed")
    if len(set(checkpoint_hashes)) != len(CONFIRMATION_TRAINING_SEEDS):
        raise ValueError("A4 confirmation requires one distinct checkpoint SHA per seed")
    return by_seed


def load_a4_confirmation_prediction_panel(
    train_cache_path: str | Path,
    test_cache_path: str | Path,
    prediction_paths: Sequence[str | Path],
    *,
    contract: Mapping[str, Any],
    development_go_summary: Mapping[str, Any],
    prevalidated_archives: Mapping[int, Mapping[str, np.ndarray]] | None = None,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Open test data only after validating the exact Development-Go artifact."""

    validate_a4_confirmation_contract(contract)
    train_counts = validate_a4_development_go_authorization(
        development_go_summary, contract=contract
    )
    train_cache = _load_npz(train_cache_path, label="A4 confirmation train cache")
    test_cache = _load_npz(test_cache_path, label="A4 confirmation test cache")
    validate_a4_confirmation_caches(
        train_cache,
        test_cache,
        authorized_train_episode_counts=train_counts,
        contract=contract,
    )
    sources = [_resolve(path) for path in prediction_paths]
    if len(sources) != len(CONFIRMATION_TRAINING_SEEDS):
        raise ValueError("A4 confirmation requires exactly three 10k prediction archives")
    by_seed: dict[int, dict[str, np.ndarray]] = {}
    checkpoint_identities: dict[int, tuple[str, str]] = {}
    supplied_archives = (
        None
        if prevalidated_archives is None
        else {int(seed): dict(arrays) for seed, arrays in prevalidated_archives.items()}
    )
    if supplied_archives is not None and tuple(sorted(supplied_archives)) != (
        CONFIRMATION_TRAINING_SEEDS
    ):
        raise ValueError("prevalidated A4 prediction panel has the wrong seed set")
    archives = (
        [
            _load_npz(source, label="A4 confirmation prediction archive")
            for source in sources
        ]
        if supplied_archives is None
        else [supplied_archives[seed] for seed in CONFIRMATION_TRAINING_SEEDS]
    )
    for arrays in archives:
        training_seed = _required(arrays, "training_seed")
        if training_seed.ndim != 1 or training_seed.dtype == np.bool_ or not np.issubdtype(
            training_seed.dtype, np.integer
        ):
            raise TypeError("A4 confirmation training_seed must be an integer vector")
        unique = np.unique(training_seed.astype(np.int64, copy=False))
        if unique.size != 1:
            raise ValueError("one A4 confirmation archive must contain one seed")
        seed = int(unique[0])
        if seed not in CONFIRMATION_TRAINING_SEEDS:
            raise ValueError(
                f"A4 confirmation seed must be one of {CONFIRMATION_TRAINING_SEEDS}"
            )
        if seed in by_seed:
            raise ValueError(f"duplicate A4 confirmation seed {seed}")
        checkpoint_identities[seed] = _validate_prediction_header(
            arrays, expected_seed=seed
        )
        by_seed[seed] = _aligned_prediction_archive(
            arrays, test_cache, expected_seed=seed
        )
    if tuple(sorted(by_seed)) != CONFIRMATION_TRAINING_SEEDS:
        raise ValueError(
            f"A4 confirmation seeds must be exactly {CONFIRMATION_TRAINING_SEEDS}"
        )
    checkpoint_paths = [value[0] for value in checkpoint_identities.values()]
    checkpoint_hashes = [value[1] for value in checkpoint_identities.values()]
    if len(set(checkpoint_paths)) != len(CONFIRMATION_TRAINING_SEEDS):
        raise ValueError("A4 confirmation requires one distinct checkpoint path per seed")
    if len(set(checkpoint_hashes)) != len(CONFIRMATION_TRAINING_SEEDS):
        raise ValueError("A4 confirmation requires one distinct checkpoint SHA per seed")
    predictions = {
        method: np.stack([by_seed[seed][method] for seed in CONFIRMATION_TRAINING_SEEDS])
        for method in (FACTUAL_METHOD, NO_ACTION_METHOD, WRONG_ACTION_METHOD)
    }
    return train_cache, test_cache, predictions


def render_markdown(summary: Mapping[str, Any]) -> str:
    primary = summary["primary"]
    raw = summary["raw_h6_h8"]["methods"]
    gate = summary["gate"]
    lines = [
        "# A4 fixed-10k reserved-test confirmation",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        (
            "Three fresh seeds were evaluated at fixed step 10000. "
            "No checkpoint selection was performed."
        ),
        "",
        "| H6-H8 comparison | Relative improvement | Positive tasks | Positive seeds | CI lower |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    no_action = primary["comparisons"][NO_ACTION_METHOD]
    lines.append(
        f"| matched no-action | {no_action['relative_improvement']:.2%} | "
        f"{no_action['positive_task_count']}/7 | {no_action['positive_seed_count']}/3 | "
        f"{no_action['paired_ci']['lower']:.6f} |"
    )
    lines.extend(
        [
            "",
            "| Wrong-action donor shift | Relative improvement | CI lower |",
            "| ---: | ---: | ---: |",
        ]
    )
    for row in primary["wrong_action_shifts"]["per_shift"]:
        lines.append(
            f"| {row['donor_shift']} | {row['relative_improvement']:.2%} | "
            f"{row['paired_ci']['lower']:.6f} |"
        )
    lines.extend(["", "| Raw H6-H8 method | Task-macro MSE |", "| --- | ---: |"])
    for method in (
        FACTUAL_METHOD,
        NO_ACTION_METHOD,
        HOLD_LAST_METHOD,
        LATENT_CONSTANT_VELOCITY_METHOD,
    ):
        lines.append(f"| {method} | {raw[method]['task_macro_mse']:.6f} |")
    lines.extend(
        [
            "",
            f"Jointly positive tasks: {len(gate['jointly_positive_tasks'])}/7",
            "",
            f"Jointly positive seeds: {len(gate['jointly_positive_seeds'])}/3",
        ]
    )
    if gate["failed_checks"]:
        lines.extend(["", "Failed checks: " + ", ".join(gate["failed_checks"])])
    boundary = summary["claim_boundary"]
    lines.extend(["", "## Claim boundary", ""])
    if boundary["authorized_claim"] is not None:
        lines.append(boundary["authorized_claim"])
    else:
        lines.append(boundary["negative_result_statement"])
    lines.extend(
        [
            "",
            "This confirmation does not establish: "
            + "; ".join(boundary["does_not_establish"])
            + ".",
        ]
    )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--development-go-summary", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = _resolve(args.output)
    if output.exists():
        raise FileExistsError(output)
    contract = _load_json(args.config, label="A4 config")
    development_go = _load_json(
        args.development_go_summary, label="A4 Development-Go summary"
    )
    # Bind the exact train cache and validate the complete checkpoint-backed
    # prediction panel before the first reserved-test NPZ read.
    validate_a4_development_go_summary(
        development_go,
        contract,
        formal_train_cache_path=args.train_cache,
    )
    prevalidated_archives = _prevalidate_prediction_sources(args.predictions)
    train_cache, test_cache, predictions = load_a4_confirmation_prediction_panel(
        args.train_cache,
        args.test_cache,
        args.predictions,
        contract=contract,
        development_go_summary=development_go,
        prevalidated_archives=prevalidated_archives,
    )
    summary = summarize_a4_confirmation(
        train_cache,
        test_cache,
        predictions,
        contract=contract,
        development_go_summary=development_go,
    )
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
