#!/usr/bin/env python3
"""Score a random-hazard action-branch A1 prediction bundle."""

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

from fcpg.lewm_action_branch_metrics import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    FACTUAL_METHOD,
    MINIMUM_INFORMATIVE_BRANCH_VARIANCE,
    NO_ACTION_METHOD,
    SHUFFLED_ACTION_METHOD,
    TrainMoments,
    fit_train_moments,
    summarize_action_branch_metrics,
    train_moments_from_arrays,
)


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_random_hazard_a1.json"


_ARRAY_ALIASES = {
    "target": (
        "target",
        "targets",
        "target_embedding",
        "latent_target",
        "latent_targets",
    ),
    "history": ("history", "history_embedding", "latent_history"),
    "task_index": ("task_index",),
    "task_instance_index": ("task_instance_index",),
    "source_episode_id": (
        "source_physical_episode_id",
        "source_episode_id",
        "physical_episode_id",
    ),
    "hazard_id": (
        "hazard_id",
        "hazard_index",
    ),
    "split": ("split",),
    "seed": ("seed", "training_seed", "model_seed"),
    "shuffle_permutation": (
        "shuffle_permutation",
        "shuffled_branch_index",
        "branch_derangement",
        "shuffled_action_source_branch",
    ),
    FACTUAL_METHOD: (
        "factual_prediction",
        "factual_predictions",
        "prediction_factual",
        "factual",
    ),
    NO_ACTION_METHOD: (
        "no_action_prediction",
        "no_action_predictions",
        "matched_no_action_prediction",
        "matched_no_action_predictions",
        "prediction_no_action",
        "no_action",
        "matched_no_action",
    ),
    SHUFFLED_ACTION_METHOD: (
        "shuffled_action_prediction",
        "shuffled_action_predictions",
        "within_bundle_shuffled_prediction",
        "within_bundle_shuffled_predictions",
        "prediction_shuffled_action",
        "shuffled_action",
        "within_bundle_shuffled",
    ),
}

_OPTIONAL_PREDICTIONS = {
    "policy_imputed_action": (
        "policy_imputed_action_prediction",
        "policy_imputed_action_predictions",
    ),
    "latent_kalman": (
        "latent_kalman_prediction",
        "latent_kalman_predictions",
        "kalman_prediction",
        "kalman_predictions",
    ),
    "forward_kinematics_last_extrinsics": (
        "forward_kinematics_prediction",
        "forward_kinematics_predictions",
        "fk_last_extrinsics_prediction",
    ),
    "pixel_geometry_kalman": (
        "pixel_geometry_kalman_prediction",
        "pixel_geometry_kalman_predictions",
    ),
}

_TRAIN_MEAN_ALIASES = (
    "train_mean",
    "train_target_mean",
    "train_moments_mean",
    "whitening_mean",
)
_TRAIN_SCALE_ALIASES = (
    "train_scale",
    "train_std",
    "train_target_std",
    "train_moments_scale",
    "train_moments_std",
    "whitening_scale",
)
_TRAIN_TARGET_ALIASES = (
    "train_target",
    "train_targets",
    "whitening_train_target",
)
_TRAIN_COUNT_ALIASES = (
    "train_moment_count",
    "train_observation_count",
    "train_count",
)


def _nonnegative_integer(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", choices=("smoke", "screen", "final"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        required=True,
        help="one or more trainer prediction NPZ files (normally one per seed)",
    )
    parser.add_argument(
        "--train-moments",
        "--train-data",
        dest="train_data",
        type=Path,
        help=(
            "optional NPZ containing train mean/scale, train_target, or canonical "
            "target plus split metadata; defaults to train rows in predictions"
        ),
    )
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-samples",
        type=_nonnegative_integer,
        default=None,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=_nonnegative_integer,
        default=None,
    )
    parser.add_argument(
        "--minimum-branch-variance",
        type=float,
        default=None,
    )
    return parser


def _load_frozen_contract(path: Path) -> dict[str, Any]:
    source = _resolve_input(path, label="A1 config")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("A1 config must contain a JSON object")
    if payload.get("protocol") != "fcpg_action_branch_random_hazard_a1_v1":
        raise ValueError("A1 config protocol is not the frozen v1 contract")
    return payload


def _resolve_scoring_contract(
    args: argparse.Namespace, contract: Mapping[str, Any]
) -> tuple[int, int, float]:
    inference = contract.get("inference")
    gate = contract.get("gate")
    if not isinstance(inference, Mapping) or not isinstance(gate, Mapping):
        raise ValueError("A1 config is missing inference/gate mappings")
    expected_samples = (
        int(inference["bootstrap_samples"]) if args.stage == "final" else 0
    )
    expected_seed = int(inference["bootstrap_seed"])
    expected_variance = float(gate["minimum_branch_variance"])
    supplied = {
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "minimum_branch_variance": args.minimum_branch_variance,
    }
    expected = {
        "bootstrap_samples": expected_samples,
        "bootstrap_seed": expected_seed,
        "minimum_branch_variance": expected_variance,
    }
    for name, value in supplied.items():
        if value is not None and value != expected[name]:
            raise ValueError(
                f"scientific A1 scoring freezes {name}={expected[name]}, got {value}"
            )
    return expected_samples, expected_seed, expected_variance


def _validate_confirmatory_support(
    arrays: Mapping[str, np.ndarray],
    *,
    stage: str,
    contract: Mapping[str, Any],
) -> None:
    target = np.asarray(arrays["target"])
    if target.ndim != 5:
        raise ValueError("target must have shape [N,C,P,K,D]")
    branches = contract.get("matched_state_branches")
    models = contract.get("models")
    split_contract = contract.get("split")
    if not all(
        isinstance(value, Mapping)
        for value in (branches, models, split_contract)
    ):
        raise ValueError("A1 config is missing support-set mappings")
    assert isinstance(branches, Mapping)
    assert isinstance(models, Mapping)
    assert isinstance(split_contract, Mapping)
    expected_branches = int(branches["branch_count"])
    if target.shape[1] != expected_branches:
        raise ValueError(
            f"A1 scoring requires exactly {expected_branches} branches, "
            f"got {target.shape[1]}"
        )
    expected_horizon = int(branches["interruption_horizon_steps"])
    if target.shape[2] != expected_horizon:
        raise ValueError(
            f"scientific A1 scoring requires exactly H={expected_horizon}, "
            f"got H={target.shape[2]}"
        )
    expected_tasks = set(range(len(contract["tasks"])))
    observed_tasks = set(np.asarray(arrays["task_index"], dtype=np.int64).tolist())
    if observed_tasks != expected_tasks:
        raise ValueError(
            f"scientific A1 task set must be {sorted(expected_tasks)}, "
            f"got {sorted(observed_tasks)}"
        )
    if stage != "smoke":
        expected_seeds = {int(value) for value in models["training_seeds"]}
        observed_seeds = set(np.asarray(arrays["seed"], dtype=np.int64).tolist())
        if observed_seeds != expected_seeds:
            raise ValueError(
                f"scientific A1 seed set must be {sorted(expected_seeds)}, "
                f"got {sorted(observed_seeds)}"
            )
    episode_key = {
        "smoke": "smoke_episodes_per_task",
        "screen": "development_episodes_per_task",
        "final": "test_episodes_per_task",
    }[stage]
    expected_episodes = int(split_contract[episode_key])
    task_index = np.asarray(arrays["task_index"], dtype=np.int64)
    episode = np.asarray(arrays["source_episode_id"]).astype(str)
    instance = np.asarray(arrays["task_instance_index"], dtype=np.int64)
    for task in sorted(expected_tasks):
        rows = task_index == task
        episodes = set(episode[rows].tolist())
        if len(episodes) != expected_episodes:
            raise ValueError(
                f"task {task} requires exactly {expected_episodes} source episodes, "
                f"got {len(episodes)}"
            )
        episode_to_instance: dict[str, int] = {}
        instance_to_episode: dict[int, str] = {}
        for episode_id, instance_id in zip(
            episode[rows].tolist(), instance[rows].tolist(), strict=True
        ):
            previous_instance = episode_to_instance.setdefault(
                episode_id, int(instance_id)
            )
            previous_episode = instance_to_episode.setdefault(
                int(instance_id), episode_id
            )
            if previous_instance != int(instance_id) or previous_episode != episode_id:
                raise ValueError(
                    "source_episode_id and task_instance_index must be one-to-one "
                    f"within task {task}"
                )


def _resolve_input(path: Path, *, label: str) -> Path:
    result = path.expanduser()
    if not result.is_absolute():
        result = ROOT / result
    result = result.resolve()
    if not result.is_file():
        raise FileNotFoundError(f"{label} does not exist: {result}")
    return result


def _archive_keys(archive: Any) -> set[str]:
    return set(str(name) for name in archive.files)


def _extract_array(
    archive: Any,
    aliases: Sequence[str],
    *,
    label: str,
    required: bool = True,
) -> np.ndarray | None:
    present = [name for name in aliases if name in _archive_keys(archive)]
    if not present:
        if required:
            raise ValueError(
                f"prediction NPZ is missing {label}; accepted keys={list(aliases)!r}"
            )
        return None
    reference = np.array(archive[present[0]], copy=True)
    for name in present[1:]:
        candidate = np.asarray(archive[name])
        if not np.array_equal(reference, candidate):
            raise ValueError(f"conflicting NPZ aliases for {label}: {present!r}")
    return reference


def _extract_scalar_count(archive: Any) -> int:
    value = _extract_array(
        archive,
        _TRAIN_COUNT_ALIASES,
        label="train moment count",
        required=False,
    )
    if value is None:
        return 1
    flattened = np.asarray(value)
    if flattened.shape == ():
        item = flattened.item()
    elif flattened.shape == (1,):
        item = flattened[0].item()
    else:
        raise ValueError("train moment count must be scalar")
    if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
        raise TypeError("train moment count must be an integer")
    if int(item) <= 0:
        raise ValueError("train moment count must be positive")
    return int(item)


def _load_moments_from_archive(
    archive: Any,
    *,
    required: bool = True,
) -> TrainMoments | None:
    mean = _extract_array(
        archive,
        _TRAIN_MEAN_ALIASES,
        label="train moment mean",
        required=False,
    )
    scale = _extract_array(
        archive,
        _TRAIN_SCALE_ALIASES,
        label="train moment scale",
        required=False,
    )
    if (mean is None) != (scale is None):
        raise ValueError("train moment mean and scale must be supplied together")
    if mean is not None and scale is not None:
        return train_moments_from_arrays(
            mean,
            scale,
            count=_extract_scalar_count(archive),
        )
    train_target = _extract_array(
        archive,
        _TRAIN_TARGET_ALIASES,
        label="train target for whitening",
        required=False,
    )
    if train_target is None:
        if required:
            raise ValueError(
                "train moments NPZ needs train mean/scale or a train_target array"
            )
        return None
    return fit_train_moments(train_target)


def _train_targets_from_split(archive: Any) -> np.ndarray | None:
    target = _extract_array(
        archive,
        _ARRAY_ALIASES["target"],
        label="target for train-split whitening",
        required=False,
    )
    splits = _extract_array(
        archive,
        _ARRAY_ALIASES["split"],
        label="split for train-split whitening",
        required=False,
    )
    if target is None and splits is None:
        return None
    if target is None or splits is None:
        raise ValueError(
            "fitting train moments from canonical arrays requires target and split"
        )
    if target.ndim != 5 or splits.shape != (target.shape[0],):
        raise ValueError(
            "train-data target must be [N,C,P,K,D] and split must be [N]"
        )
    train_rows = splits.astype(str) == "train"
    if not np.any(train_rows):
        return None
    return np.array(target[train_rows], copy=True)


def _same_moments(left: TrainMoments, right: TrainMoments) -> bool:
    return (
        np.array_equal(left.mean, right.mean)
        and np.array_equal(left.scale, right.scale)
        and left.count == right.count
        and left.epsilon == right.epsilon
    )


def _normalize_row_arrays(
    arrays: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    *,
    source: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    target = np.asarray(arrays["target"])
    if target.ndim != 5:
        raise ValueError(f"{source}: target must have shape [N,C,P,K,D]")
    count = target.shape[0]
    normalized_arrays: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        if name == "shuffle_permutation" and array.ndim == 2:
            array = np.broadcast_to(array[None], (count, *array.shape))
        if array.shape[:1] != (count,):
            raise ValueError(f"{source}: {name} is not aligned to target rows")
        normalized_arrays[name] = np.array(array, copy=True)
    normalized_predictions: dict[str, np.ndarray] = {}
    for name, value in predictions.items():
        array = np.asarray(value)
        if array.shape[:1] != (count,):
            raise ValueError(
                f"{source}: prediction {name!r} is not aligned to target rows"
            )
        normalized_predictions[name] = np.array(array, copy=True)
    return normalized_arrays, normalized_predictions


def load_prediction_bundle(
    prediction_paths: Path | Sequence[Path],
    *,
    train_moments_path: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], TrainMoments]:
    """Load only the compact arrays needed by the scorer."""

    raw_paths = (
        [prediction_paths]
        if isinstance(prediction_paths, Path)
        else list(prediction_paths)
    )
    if not raw_paths:
        raise ValueError("at least one prediction NPZ is required")
    sources = [
        _resolve_input(path, label="prediction NPZ") for path in raw_paths
    ]
    array_parts: list[dict[str, np.ndarray]] = []
    prediction_parts: list[dict[str, np.ndarray]] = []
    inline_moments: list[TrainMoments] = []
    inline_train_targets: list[np.ndarray] = []
    for source in sources:
        with np.load(source, allow_pickle=False) as archive:
            arrays = {
                name: _extract_array(archive, aliases, label=name)
                for name, aliases in _ARRAY_ALIASES.items()
            }
            predictions = {
                name: arrays.pop(name)
                for name in (
                    FACTUAL_METHOD,
                    NO_ACTION_METHOD,
                    SHUFFLED_ACTION_METHOD,
                )
            }
            for method, aliases in _OPTIONAL_PREDICTIONS.items():
                value = _extract_array(
                    archive,
                    aliases,
                    label=method,
                    required=False,
                )
                if value is not None:
                    predictions[method] = value
            if train_moments_path is None:
                loaded = _load_moments_from_archive(archive, required=False)
                if loaded is not None:
                    inline_moments.append(loaded)
                train_rows = _train_targets_from_split(archive)
                if train_rows is not None:
                    inline_train_targets.append(train_rows)
        normalized = _normalize_row_arrays(
            arrays,
            predictions,
            source=source,
        )
        array_parts.append(normalized[0])
        prediction_parts.append(normalized[1])

    array_keys = set(array_parts[0])
    prediction_keys = set(prediction_parts[0])
    if any(set(part) != array_keys for part in array_parts[1:]):
        raise ValueError("prediction NPZ files do not share the same metadata arrays")
    if any(set(part) != prediction_keys for part in prediction_parts[1:]):
        raise ValueError("prediction NPZ files do not share the same methods")
    arrays = {
        name: np.concatenate([part[name] for part in array_parts], axis=0)
        for name in sorted(array_keys)
    }
    predictions = {
        name: np.concatenate([part[name] for part in prediction_parts], axis=0)
        for name in sorted(prediction_keys)
    }

    if train_moments_path is not None:
        moments_source = _resolve_input(train_moments_path, label="train moments NPZ")
        with np.load(moments_source, allow_pickle=False) as archive:
            moments = _load_moments_from_archive(archive, required=False)
            if moments is None:
                train_rows = _train_targets_from_split(archive)
                if train_rows is None:
                    raise ValueError(
                        "train-data NPZ has no moments and no split=train targets"
                    )
                moments = fit_train_moments(train_rows)
    elif inline_moments:
        moments = inline_moments[0]
        if any(not _same_moments(moments, item) for item in inline_moments[1:]):
            raise ValueError("prediction NPZ files contain conflicting train moments")
    elif inline_train_targets:
        moments = fit_train_moments(np.concatenate(inline_train_targets, axis=0))
    else:
        raise ValueError(
            "no train moments found; pass --train-data with canonical train rows"
        )
    return arrays, predictions, moments


def select_stage_rows(
    arrays: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    *,
    stage: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Select the canonical evaluation split while retaining complete bundles."""

    expected_split = {
        "smoke": "smoke",
        "screen": "development",
        "final": "test",
    }[stage]
    target = np.asarray(arrays["target"])
    splits = np.asarray(arrays["split"])
    if target.ndim != 5 or splits.shape != (target.shape[0],):
        raise ValueError("target must be [N,C,P,K,D] and split must be [N]")
    selected = splits.astype(str) == expected_split
    if not np.any(selected):
        raise ValueError(
            f"prediction NPZ has no rows for stage {stage!r} "
            f"(split={expected_split!r})"
        )
    selected_arrays: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        if name == "shuffle_permutation" and array.ndim == 2:
            selected_arrays[name] = np.array(array, copy=True)
        elif array.shape[:1] == (target.shape[0],):
            selected_arrays[name] = np.array(array[selected], copy=True)
        else:
            raise ValueError(f"{name} is not aligned to the target row axis")
    selected_predictions = {
        name: np.array(np.asarray(value)[selected], copy=True)
        for name, value in predictions.items()
    }
    return selected_arrays, selected_predictions


def _percent(value: Any) -> str:
    return f"{float(value):.2%}"


def _markdown(summary: Mapping[str, Any]) -> str:
    gate = summary["gate"]
    lines = [
        "# Action-Branch Random-Hazard A1",
        "",
        f"Stage: **{summary['stage']}**  ",
        f"Decision: **{summary['decision']}**",
        "",
        "## Primary comparisons",
        "",
        "| Control | Factual effect | 95% episode-cluster CI | Positive tasks | Positive seeds |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    labels = {
        NO_ACTION_METHOD: "Matched no-action",
        SHUFFLED_ACTION_METHOD: "Within-bundle shuffled action",
    }
    for name, label in labels.items():
        row = summary["comparisons"][name]
        ci = row["effect_95_ci"]
        rendered_ci = (
            f"[{_percent(ci[0])}, {_percent(ci[1])}]"
            if ci is not None
            else "not required"
        )
        lines.append(
            f"| {label} | {_percent(row['effect'])} | {rendered_ci} | "
            f"{row['positive_task_count']}/{row['task_count']} | "
            f"{row['positive_seed_count']}/{row['seed_count']} |"
        )

    lines.extend(
        [
            "",
            "## Late window H6-H8",
            "",
            "| Control | Factual effect | 95% episode-cluster CI |",
            "| --- | ---: | ---: |",
        ]
    )
    for name, label in labels.items():
        row = summary["late_window"]["comparisons"][name]
        ci = row["effect_95_ci"]
        rendered_ci = (
            f"[{_percent(ci[0])}, {_percent(ci[1])}]"
            if ci is not None
            else "not required"
        )
        lines.append(f"| {label} | {_percent(row['effect'])} | {rendered_ci} |")

    raw_sanity = summary["sanity"][
        "factual_raw_mse_not_worse_than_no_action"
    ]
    lines.extend(
        [
            "",
            "## Raw MSE sanity",
            "",
            f"Factual: `{float(raw_sanity['factual_mse']):.6f}`; "
            f"no-action: `{float(raw_sanity['no_action_mse']):.6f}`; "
            f"passed: **{bool(raw_sanity['passed'])}**.",
        ]
    )

    lines.extend(
        [
            "",
            "## Informative bundles",
            "",
            "| Task | Informative bundles | Fraction |",
            "| --- | ---: | ---: |",
        ]
    )
    for task, row in summary["informative_bundles"]["per_task"].items():
        lines.append(
            f"| {task} | {row['informative_bundle_count']}/{row['bundle_count']} | "
            f"{_percent(row['fraction'])} |"
        )

    lines.extend(
        [
            "",
            "## Secondary raw trajectory MSE",
            "",
            "| Method | Task-macro whitened MSE |",
            "| --- | ---: |",
        ]
    )
    for method, mse in summary["secondary"]["method_task_macro_mse"].items():
        lines.append(f"| {method} | {float(mse):.6f} |")
    if gate["failed_checks"]:
        lines.extend(
            [
                "",
                "Failed checks: "
                + ", ".join(f"`{name}`" for name in gate["failed_checks"]),
            ]
        )
    lines.extend(
        [
            "",
            "The confidence interval resamples source physical episodes within "
            "task using shared draws; hazard, branch, horizon, and seed remain "
            "nested observations.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = _load_frozen_contract(args.config)
    bootstrap_samples, bootstrap_seed, minimum_branch_variance = (
        _resolve_scoring_contract(args, contract)
    )
    if args.stage != "smoke" and args.train_data is None:
        raise ValueError(
            "screen/final A1 scoring requires --train-data so whitening is "
            "recomputed from canonical split=train targets"
        )
    arrays, predictions, moments = load_prediction_bundle(
        args.predictions,
        train_moments_path=args.train_data,
    )
    arrays, predictions = select_stage_rows(
        arrays,
        predictions,
        stage=args.stage,
    )
    _validate_confirmatory_support(arrays, stage=args.stage, contract=contract)
    summary = summarize_action_branch_metrics(
        arrays["target"],
        predictions,
        arrays["history"],
        task_index=arrays["task_index"],
        task_instance_index=arrays["task_instance_index"],
        source_episode_id=arrays["source_episode_id"],
        hazard_id=arrays["hazard_id"],
        split=arrays["split"],
        training_seed=arrays["seed"],
        moments=moments,
        shuffle_permutation=arrays["shuffle_permutation"],
        stage=args.stage,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        minimum_branch_variance=minimum_branch_variance,
    )

    output = args.output.expanduser()
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "summary.md").write_text(_markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "output": str(output),
                "source_physical_episode_count": summary["coverage"][
                    "source_physical_episode_count"
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
