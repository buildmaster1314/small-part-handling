#!/usr/bin/env python3
"""Score the frozen A4 endpoints on the independent A4b fresh panel."""

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

from fcpg.lewm_action_branch_a4 import PROTOCOL as SOURCE_PROTOCOL  # noqa: E402
from fcpg.lewm_action_branch_a4_confirmation_collection import (  # noqa: E402
    validate_a4_development_go_summary,
)
from fcpg.lewm_action_branch_a4_confirmation_metrics import (  # noqa: E402
    validate_a4_development_go_authorization,
)
from fcpg.lewm_action_branch_a4_metrics import (  # noqa: E402
    FACTUAL_METHOD,
    NO_ACTION_METHOD,
    WRONG_ACTION_METHOD,
)
from fcpg.lewm_action_branch_a4b import (  # noqa: E402
    BENCHMARK_SEED,
    CONFIRMATION_TRAINING_SEEDS,
    PROTOCOL,
    SOURCE_A4_PROTOCOL,
    SPLIT_MANIFEST_ID,
    TASK_IDENTITY_NAMESPACE,
    validate_a4b_contract,
)
from fcpg.lewm_action_branch_a4b_confirmation_metrics import (  # noqa: E402
    summarize_a4b_confirmation,
    validate_a4b_confirmation_caches,
    validate_a4b_fresh_results,
    validate_a4b_result_cache_path,
)
from scripts.score_lewm_action_branch_a4_confirmation import (  # noqa: E402
    _aligned_prediction_archive,
    _load_json,
    _load_npz,
    _required,
    _scalar,
    _sha256_file,
    _validate_prediction_header,
    render_markdown as _render_source_markdown,
)


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_a4b_fresh_confirmation.json"
DEFAULT_SOURCE_CONFIG = ROOT / "config/lewm/eval/action_branch_active_control_a4.json"
DEFAULT_SPLIT_MANIFEST = ROOT / (
    "config/lewm/splits/action_branch_a4b_fresh_confirmation_seed31401.json"
)
DEFAULT_OUTPUT = ROOT / "outputs/lewm_action_branch_a4b/confirmation_gate"


def _validate_a4b_prediction_identity(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_seed: int,
    final_result_path: Path,
) -> dict[str, np.ndarray]:
    observed = (
        str(_scalar(arrays, "protocol")),
        str(_scalar(arrays, "source_endpoint_protocol")),
        str(_scalar(arrays, "split_manifest_id")),
        str(_scalar(arrays, "task_identity_namespace")),
        int(_scalar(arrays, "benchmark_seed")),
        str(_scalar(arrays, "evaluation_split")),
        str(Path(str(_scalar(arrays, "final_collection_result"))).resolve()),
    )
    expected = (
        PROTOCOL,
        SOURCE_A4_PROTOCOL,
        SPLIT_MANIFEST_ID,
        TASK_IDENTITY_NAMESPACE,
        BENCHMARK_SEED,
        "test",
        str(final_result_path),
    )
    if observed != expected:
        raise ValueError("A4b prediction identity drifted")
    adapted = dict(arrays)
    adapted["protocol"] = np.asarray(SOURCE_PROTOCOL)
    _validate_prediction_header(adapted, expected_seed=expected_seed)
    return adapted


def _prevalidate_predictions(
    paths: Sequence[str | Path], *, final_result_path: Path
) -> dict[int, dict[str, np.ndarray]]:
    sources = tuple(Path(path).expanduser().resolve() for path in paths)
    if len(sources) != len(CONFIRMATION_TRAINING_SEEDS):
        raise ValueError("A4b confirmation requires exactly three predictions")
    if len(set(sources)) != len(sources):
        raise ValueError("A4b prediction paths must be distinct")
    by_seed: dict[int, dict[str, np.ndarray]] = {}
    endpoint_paths: set[str] = set()
    endpoint_digests: set[str] = set()
    for source in sources:
        arrays = _load_npz(source, label="A4b confirmation prediction archive")
        training_seed = _required(arrays, "training_seed")
        if training_seed.ndim != 1 or training_seed.dtype == np.bool_ or not np.issubdtype(
            training_seed.dtype, np.integer
        ):
            raise TypeError("A4b training_seed must be an integer vector")
        unique = np.unique(training_seed.astype(np.int64, copy=False))
        if unique.size != 1:
            raise ValueError("one A4b prediction archive must contain one seed")
        seed = int(unique[0])
        if seed not in CONFIRMATION_TRAINING_SEEDS or seed in by_seed:
            raise ValueError("A4b prediction seed panel is invalid")
        adapted = _validate_a4b_prediction_identity(
            arrays,
            expected_seed=seed,
            final_result_path=final_result_path,
        )
        endpoint_path, endpoint_digest = _validate_prediction_header(
            adapted, expected_seed=seed
        )
        endpoint = Path(endpoint_path)
        if not endpoint.is_file():
            raise FileNotFoundError(endpoint)
        if _sha256_file(endpoint) != endpoint_digest:
            raise PermissionError(f"A4b endpoint content mismatch for seed {seed}")
        endpoint_paths.add(endpoint_path)
        endpoint_digests.add(endpoint_digest)
        by_seed[seed] = adapted
    if tuple(sorted(by_seed)) != CONFIRMATION_TRAINING_SEEDS:
        raise ValueError("A4b prediction seeds must be exactly 30301/30302/30303")
    if len(endpoint_paths) != 3 or len(endpoint_digests) != 3:
        raise ValueError("A4b requires three distinct frozen endpoint identities")
    return by_seed


def render_markdown(summary: Mapping[str, Any]) -> str:
    text = _render_source_markdown(summary)
    text = text.replace(
        "# A4 fixed-10k reserved-test confirmation",
        "# A4b fresh-panel fixed-10k confirmation",
        1,
    )
    return text.replace(
        "Three fresh seeds were evaluated at fixed step 10000. ",
        "Three frozen A4 v2 endpoints were evaluated on the fresh A4b panel. ",
        1,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--preflight-result", type=Path, required=True)
    parser.add_argument("--final-result", type=Path, required=True)
    parser.add_argument("--development-go-summary", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    contract = _load_json(args.config, label="A4b config")
    source_contract = _load_json(args.source_config, label="A4 v2 source config")
    split_manifest = _load_json(args.split_manifest, label="A4b split manifest")
    preflight_result = _load_json(
        args.preflight_result, label="A4b preflight result"
    )
    final_result = _load_json(args.final_result, label="A4b final result")
    development_go = _load_json(
        args.development_go_summary, label="A4 Development-Go summary"
    )
    validate_a4b_contract(
        contract,
        source_contract=source_contract,
        split_manifest=split_manifest,
    )
    validate_a4b_fresh_results(preflight_result, final_result)
    validate_a4b_result_cache_path(final_result, args.test_cache)
    validate_a4_development_go_summary(
        development_go,
        source_contract,
        formal_train_cache_path=args.train_cache,
    )
    final_result_path = args.final_result.expanduser().resolve()
    archives = _prevalidate_predictions(
        args.predictions, final_result_path=final_result_path
    )

    train_cache = _load_npz(args.train_cache, label="A4 source train cache")
    test_cache = _load_npz(args.test_cache, label="A4b final test cache")
    # This separate call makes cache/result mismatches fail before statistics.
    authorized_counts = validate_a4_development_go_authorization(
        development_go, contract=source_contract
    )
    validate_a4b_confirmation_caches(
        train_cache,
        test_cache,
        authorized_train_episode_counts=authorized_counts,
        source_contract=source_contract,
        final_result=final_result,
    )
    aligned = {
        seed: _aligned_prediction_archive(
            archives[seed], test_cache, expected_seed=seed
        )
        for seed in CONFIRMATION_TRAINING_SEEDS
    }
    predictions = {
        method: np.stack(
            [aligned[seed][method] for seed in CONFIRMATION_TRAINING_SEEDS]
        )
        for method in (FACTUAL_METHOD, NO_ACTION_METHOD, WRONG_ACTION_METHOD)
    }
    summary = summarize_a4b_confirmation(
        train_cache,
        test_cache,
        predictions,
        contract=contract,
        source_contract=source_contract,
        split_manifest=split_manifest,
        development_go_summary=development_go,
        preflight_result=preflight_result,
        final_result=final_result,
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
