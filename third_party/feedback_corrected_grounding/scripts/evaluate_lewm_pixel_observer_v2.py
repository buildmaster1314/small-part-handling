#!/usr/bin/env python3
"""Evaluate PixelObserver v2 feedback, writeback, and classic baselines."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping
import uuid

import numpy as np
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_pixel_observer_v2_data import build_pixel_observer_v2_data  # noqa: E402
from fcpg.lewm_pixel_observer_v2_eval import (  # noqa: E402
    aggregate_sequential_metrics,
    evaluate_v2_sequence,
    oracle_diagnostic_sequence,
    sequence_from_dataset,
)
from fcpg.lewm_pixel_observer_v2_report import (  # noqa: E402
    EVALUATION_IMPLEMENTATION_RELATIVE_PATHS,
    PRIMARY_CONTROL_METRIC_FAMILY,
    condition_primary_metric as _condition_primary_metric,
    episode_coverage,
    matched_training_protocol as _matched_training_protocol,
    validate_episode_rows,
)
from fcpg.lewm_pixel_observer_v2_training import (  # noqa: E402
    load_trained_pixel_observer_v2,
    resolve_v2_training_config,
)
from fcpg.lewm_pixel_observer_v2_runtime import canonical_sha256  # noqa: E402
from fcpg.lewm_provenance import sha256_file  # noqa: E402
from fcpg.lewm_runtime import choose_device, load_lewm_grounded_checkpoint  # noqa: E402


def _evaluation_implementation_manifest(
    root: Path = ROOT,
) -> dict[str, str]:
    return {
        relative: sha256_file(root / relative)
        for relative in EVALUATION_IMPLEMENTATION_RELATIVE_PATHS
    }


def _episode_identity(dataset: Any, episode_index: int) -> dict[str, str]:
    raw_episode = dataset.episodes[int(episode_index)]
    index = int(raw_episode[0])
    return {
        "source": str(dataset.source_name),
        "task": str(dataset.task_name[index]),
        "episode": str(dataset.episode_id[index]),
    }


def _select_evaluation_episodes(
    evaluation_bundle: Any,
    *,
    split: str,
    max_episodes_per_source: int,
) -> tuple[list[tuple[Any, tuple[int, ...]]], dict[str, Any]]:
    if max_episodes_per_source < 0:
        raise ValueError("max_episodes_per_source must be non-negative")
    selections: list[tuple[Any, tuple[int, ...]]] = []
    identities: list[dict[str, str]] = []
    for dataset in evaluation_bundle.datasets:
        source = str(dataset.source_name)
        episode_indices = list(range(len(dataset.episodes)))
        if split != "all":
            allowed = set(evaluation_bundle.provenance["splits"][source][split])
            selected = []
            for episode_index in episode_indices:
                identity = _episode_identity(dataset, episode_index)
                if identity["task"] + ":" + identity["episode"] in allowed:
                    selected.append(episode_index)
            episode_indices = selected
        if max_episodes_per_source > 0:
            episode_indices = episode_indices[:max_episodes_per_source]
        selections.append((dataset, tuple(episode_indices)))
        identities.extend(
            _episode_identity(dataset, index) for index in episode_indices
        )
    return selections, episode_coverage(identities)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=None,
        help="Independent evaluation data config; training provenance still verifies the checkpoint",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--velocity-gain", type=float, default=1.0)
    parser.add_argument("--dropout-start-offset", type=int, default=None)
    parser.add_argument("--dropout-length", type=int, default=0)
    parser.add_argument("--hidden-step-length", type=int, default=0)
    parser.add_argument(
        "--matrix-profile",
        choices=("drift", "step"),
        default=None,
        help=(
            "Evaluate the frozen standard condition set in one process; "
            "--output is then a directory"
        ),
    )
    parser.add_argument("--max-episodes-per-source", type=int, default=0)
    parser.add_argument("--split", choices=("val", "train", "all"), default="val")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Skip only existing matrix cells whose frozen provenance matches exactly",
    )
    parser.add_argument("--output", type=Path, required=True)
    result = parser.parse_args()
    if result.matrix_profile is not None and (
        result.dropout_start_offset is not None
        or result.dropout_length != 0
        or result.hidden_step_length != 0
    ):
        parser.error("matrix profiles cannot be combined with manual dropout options")
    if result.max_episodes_per_source < 0:
        parser.error("--max-episodes-per-source must be non-negative")
    return result


def _load_config(path: Path) -> dict[str, object]:
    value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(value, dict):
        raise TypeError("evaluation config root must be a mapping")
    return resolve_v2_training_config(value)


def _validate_eval_compatibility(training_bundle, evaluation_bundle) -> None:
    first = training_bundle.reference_dataset
    second = evaluation_bundle.reference_dataset
    fields = ("history_size", "frame_skip", "num_keypoints", "proprio_dim", "transition_action_dim", "image_size")
    mismatches = {
        name: (getattr(first, name), getattr(second, name))
        for name in fields
        if getattr(first, name) != getattr(second, name)
    }
    if mismatches:
        raise ValueError(f"external evaluation data is structurally incompatible: {mismatches}")


def _episode_scalar_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in metrics.items() if not key.startswith("_")}


def _paired_differences(rows: list[dict[str, object]]) -> dict[str, object]:
    comparisons = (
        ("output_dropout_vs_one_shot_nominal_fallback", "one_shot_fcpg_nominal_fallback", "post_update_qtrue_dropout_epe_px", "post_update_qtrue_dropout_epe_px", "output_layer_dropout_primary"),
        ("output_dropout_vs_one_shot_hold_fallback", "one_shot_fcpg_hold_fallback", "post_update_qtrue_dropout_epe_px", "post_update_qtrue_dropout_epe_px", "output_layer_dropout_primary"),
        ("output_dropout_vs_no_writeback", "posterior_reindex_no_writeback", "post_update_qtrue_dropout_epe_px", "post_update_qtrue_dropout_epe_px", "output_layer_dropout_primary"),
        ("output_vs_one_shot_nominal_fallback", "one_shot_fcpg_nominal_fallback", "post_update_qtrue_epe_px", "post_update_qtrue_epe_px", "output_layer"),
        ("output_vs_one_shot_hold_fallback", "one_shot_fcpg_hold_fallback", "post_update_qtrue_epe_px", "post_update_qtrue_epe_px", "output_layer"),
        ("output_vs_no_writeback", "posterior_reindex_no_writeback", "post_update_qtrue_epe_px", "post_update_qtrue_epe_px", "output_layer"),
        ("prior_dropout_vs_nominal_open_loop", "nominal_open_loop", "prior_qtrue_dropout_epe_px", "prior_qtrue_dropout_epe_px", "pre_measurement_prior_dropout_primary"),
        ("prior_dropout_vs_hold_last", "hold_last", "prior_qtrue_dropout_epe_px", "prior_qtrue_dropout_epe_px", "pre_measurement_prior_dropout_primary"),
        ("prior_dropout_vs_constant_velocity", "constant_velocity", "prior_qtrue_dropout_epe_px", "prior_qtrue_dropout_epe_px", "pre_measurement_prior_dropout_primary"),
        ("prior_dropout_vs_linear_kalman", "linear_kalman", "prior_qtrue_dropout_epe_px", "prior_qtrue_dropout_epe_px", "pre_measurement_prior_dropout_primary"),
        ("prior_dropout_vs_no_writeback", "posterior_reindex_no_writeback", "prior_qtrue_dropout_epe_px", "prior_qtrue_dropout_epe_px", "pre_measurement_prior_dropout_primary"),
        ("prior_vs_nominal_open_loop", "nominal_open_loop", "prior_qtrue_epe_px", "prior_qtrue_epe_px", "pre_measurement_prior"),
        ("prior_vs_hold_last", "hold_last", "prior_qtrue_epe_px", "prior_qtrue_epe_px", "pre_measurement_prior"),
        ("prior_vs_constant_velocity", "constant_velocity", "prior_qtrue_epe_px", "prior_qtrue_epe_px", "pre_measurement_prior"),
        ("prior_vs_linear_kalman", "linear_kalman", "prior_qtrue_epe_px", "prior_qtrue_epe_px", "pre_measurement_prior"),
        ("prior_vs_no_writeback", "posterior_reindex_no_writeback", "prior_qtrue_epe_px", "prior_qtrue_epe_px", "pre_measurement_prior"),
    )
    output: dict[str, object] = {}
    rng = np.random.default_rng(7168)
    for label, comparator, full_metric, comparator_metric, comparison_layer in comparisons:
        grouped: dict[str, list[float]] = {}
        for row in rows:
            modes = row["modes"]
            full = modes.get("full_writeback", {}).get(full_metric)
            other = modes.get(comparator, {}).get(comparator_metric)
            if full is not None and other is not None:
                grouped.setdefault(str(row["task"]), []).append(float(full) - float(other))
        if not grouped:
            continue
        task_values = {
            task: np.asarray(grouped[task], dtype=np.float64)
            for task in sorted(grouped)
        }
        values = np.concatenate(list(task_values.values()))
        task_means = np.asarray(
            [task_values[task].mean() for task in sorted(task_values)],
            dtype=np.float64,
        )
        bootstrap = np.empty(10000, dtype=np.float64)
        for index in range(bootstrap.size):
            sampled_task_means = [
                rng.choice(
                    task_values[task], size=task_values[task].size, replace=True
                ).mean()
                for task in sorted(task_values)
            ]
            bootstrap[index] = float(np.mean(sampled_task_means))
        output[label] = {
            "comparison_layer": comparison_layer,
            "full_writeback_metric": full_metric,
            "comparator": comparator,
            "comparator_metric": comparator_metric,
            "episodes": int(values.size),
            "task_macro_mean_difference_px": float(task_means.mean()),
            "episode_pooled_mean_difference_px": float(values.mean()),
            "task_mean_differences_px": {
                task: float(np.mean(grouped[task])) for task in sorted(grouped)
            },
            "task_stratified_bootstrap95_ci_px": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "negative_favors_full_writeback": True,
            "bootstrap_seed": 7168,
            "bootstrap_samples": 10000,
        }
    return output


def _atomic_write(path: Path, payload: dict[str, object]) -> None:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite evaluation report: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)


def _matrix_conditions(
    profile: str,
) -> tuple[tuple[str, int | None, int, int], ...]:
    if profile == "drift":
        return (
            ("clean", None, 0, 0),
            ("drop1", 7, 1, 0),
            ("drop3", 7, 3, 0),
            ("drop5", 7, 5, 0),
        )
    if profile == "step":
        return (
            ("clean", None, 0, 0),
            ("drop1", 1, 1, 0),
            ("drop3", 1, 3, 0),
            ("drop5", 1, 5, 0),
            ("hidden5", None, 0, 5),
        )
    raise ValueError(f"unknown matrix profile: {profile!r}")


def _completed_report_matches(
    path: Path,
    *,
    checkpoint_sha256: str,
    evaluator_sha256: str,
    implementation_files: Mapping[str, str],
    implementation_sha256: str,
    evaluation_config_sha256: str,
    evaluation_data_provenance_sha256: str,
    evaluation_episode_selection: Mapping[str, Any],
    max_episodes_per_source: int,
    split: str,
    condition_name: str,
    matrix_profile: str | None,
    gain: float,
    velocity_gain: float,
    dropout_start_offset: int | None,
    dropout_length: int,
    hidden_step_length: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError):
        return False
    primary_control_metric = _condition_primary_metric(
        dropout_length=dropout_length,
        hidden_step_length=hidden_step_length,
    )
    episode_selection = dict(evaluation_episode_selection)
    expected = {
        "checkpoint_sha256": checkpoint_sha256,
        "evaluator_sha256": evaluator_sha256,
        "implementation_files": dict(implementation_files),
        "implementation_sha256": implementation_sha256,
        "evaluation_config_sha256": evaluation_config_sha256,
        "evaluation_data_provenance_sha256": evaluation_data_provenance_sha256,
        "evaluation_episode_selection": episode_selection,
        "evaluation_episode_selection_sha256": canonical_sha256(
            episode_selection
        ),
        "max_episodes_per_source": max_episodes_per_source,
        "split": split,
        "evaluation_condition": condition_name,
        "evaluation_matrix_profile": matrix_profile,
        "primary_control_metric_family": PRIMARY_CONTROL_METRIC_FAMILY,
        "primary_control_metric": primary_control_metric,
        "observer_gains": {
            "position_alpha": gain,
            "velocity_beta": velocity_gain,
        },
        "synthetic_measurement_dropout_contract": {
            "type": "oracle_known_measurement_dropout",
            "rgb_occluded": False,
            "autonomous_detector": False,
            "start_offset": dropout_start_offset,
            "length": dropout_length,
        },
        "hidden_step_negative_control": {
            "measurement_hidden_from_step": hidden_step_length > 0,
            "sampled_steps": hidden_step_length,
        },
    }
    if not isinstance(report, dict) or not all(
        report.get(name) == value for name, value in expected.items()
    ):
        return False
    try:
        if canonical_sha256(report["implementation_files"]) != report[
            "implementation_sha256"
        ]:
            return False
        if canonical_sha256(report["evaluation_episode_selection"]) != report[
            "evaluation_episode_selection_sha256"
        ]:
            return False
        rows = report["episode_rows"]
        if not isinstance(rows, list):
            return False
        validate_episode_rows(
            rows,
            coverage=episode_selection,
            primary_metric=primary_control_metric,
        )
    except (KeyError, TypeError, ValueError):
        return False
    return True


def main() -> None:
    args = parse_args()
    target_device = choose_device(args.device)
    checkpoint = args.checkpoint.expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    training_resolved = payload.get("resolved_config")
    if not isinstance(training_resolved, dict):
        raise ValueError("checkpoint lacks resolved_config")
    training_bundle = build_pixel_observer_v2_data(training_resolved)
    base_path = Path(str(payload["base_checkpoint_path"]))
    base_model, _, _ = load_lewm_grounded_checkpoint(
        base_path, device=target_device, dataset=training_bundle.reference_dataset
    )
    runtime, verified = load_trained_pixel_observer_v2(
        checkpoint,
        base_model=base_model,
        expected_data_provenance=training_bundle.provenance,
        for_deployment=payload.get("target_mode") == "paired_deployment",
    )
    if args.eval_config is None:
        evaluation_resolved = training_resolved
        evaluation_bundle = training_bundle
        eval_config_path = None
        eval_config_sha256 = canonical_sha256(evaluation_resolved)
    else:
        eval_config_path = args.eval_config.expanduser().resolve()
        evaluation_resolved = _load_config(eval_config_path)
        if bool(evaluation_resolved["data"]["history_tail_mask"]["enabled"]):
            raise ValueError("external eval config must disable history-tail training masks")
        evaluation_bundle = build_pixel_observer_v2_data(evaluation_resolved)
        _validate_eval_compatibility(training_bundle, evaluation_bundle)
        eval_config_sha256 = sha256_file(eval_config_path)
    checkpoint_sha256 = sha256_file(checkpoint)
    implementation_files = _evaluation_implementation_manifest()
    implementation_sha256 = canonical_sha256(implementation_files)
    evaluator_sha256 = sha256_file(Path(__file__).resolve())
    evaluation_data_provenance_sha256 = canonical_sha256(
        evaluation_bundle.provenance
    )
    selected_evaluation_episodes, evaluation_episode_selection = (
        _select_evaluation_episodes(
            evaluation_bundle,
            split=args.split,
            max_episodes_per_source=args.max_episodes_per_source,
        )
    )
    matched_training_protocol_sha256 = canonical_sha256(
        _matched_training_protocol(training_resolved)
    )

    def build_report(
        *,
        condition_name: str,
        dropout_start_offset: int | None,
        dropout_length: int,
        hidden_step_length: int,
    ) -> dict[str, Any]:
        records: dict[str, list[dict[str, object]]] = {}
        source_records: dict[str, dict[str, list[dict[str, object]]]] = {}
        task_records: dict[str, dict[str, list[dict[str, object]]]] = {}
        episode_rows: list[dict[str, object]] = []
        for dataset, episode_indices in selected_evaluation_episodes:
            for episode_index in episode_indices:
                raw_episode = dataset.episodes[episode_index]
                task = str(dataset.task_name[int(raw_episode[0])])
                episode = str(dataset.episode_id[int(raw_episode[0])])
                sequence = sequence_from_dataset(
                    dataset,
                    episode_index,
                    device=target_device,
                    dropout_start_offset=dropout_start_offset,
                    dropout_length=dropout_length,
                    hidden_step_length=hidden_step_length,
                    require_step_event=(
                        args.matrix_profile is not None or hidden_step_length > 0
                    ),
                )
                if verified["target_mode"] == "oracle_ceiling":
                    sequence = oracle_diagnostic_sequence(sequence)
                evaluated = evaluate_v2_sequence(
                    runtime,
                    sequence,
                    gain=args.gain,
                    velocity_gain=args.velocity_gain,
                )
                row_modes: dict[str, object] = {}
                for mode, metrics in evaluated.items():
                    records.setdefault(mode, []).append(metrics)
                    source_records.setdefault(
                        str(dataset.source_name), {}
                    ).setdefault(mode, []).append(metrics)
                    task_records.setdefault(task, {}).setdefault(mode, []).append(
                        metrics
                    )
                    row_modes[mode] = _episode_scalar_metrics(metrics)
                episode_rows.append(
                    {
                        "source": str(dataset.source_name),
                        "task": task,
                        "episode": episode,
                        "episode_key": f"{task}:{episode}",
                        "modes": row_modes,
                    }
                )
        primary_control_metric = _condition_primary_metric(
            dropout_length=dropout_length,
            hidden_step_length=hidden_step_length,
        )
        validate_episode_rows(
            episode_rows,
            coverage=evaluation_episode_selection,
            primary_metric=primary_control_metric,
        )
        return {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_resolved_config": copy.deepcopy(training_resolved),
            "checkpoint_resolved_config_sha256": canonical_sha256(
                training_resolved
            ),
            "checkpoint_matched_training_protocol_sha256": (
                matched_training_protocol_sha256
            ),
            "checkpoint_adapter_config": verified["adapter_config"],
            "checkpoint_adapter_config_sha256": verified[
                "adapter_config_sha256"
            ],
            "checkpoint_training_budget": verified["training_budget"],
            "checkpoint_training_binding_sha256": verified[
                "training_binding_sha256"
            ],
            "base_checkpoint": str(base_path),
            "base_checkpoint_sha256": verified["base_checkpoint_sha256"],
            "evaluator": str(Path(__file__).resolve()),
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
            "implementation_files": implementation_files,
            "implementation_sha256": implementation_sha256,
            "evaluation_condition": condition_name,
            "evaluation_matrix_profile": args.matrix_profile,
            "max_episodes_per_source": args.max_episodes_per_source,
            "evaluation_episode_selection": evaluation_episode_selection,
            "evaluation_episode_selection_sha256": canonical_sha256(
                evaluation_episode_selection
            ),
            "target_mode": verified["target_mode"],
            "q_true_usage": (
                "diagnostic_oracle_history_and_target"
                if verified["target_mode"] == "oracle_ceiling"
                else "offline_scoring_only"
            ),
            "oracle_history_diagnostic_only": (
                verified["target_mode"] == "oracle_ceiling"
            ),
            "primary_control_metric_family": PRIMARY_CONTROL_METRIC_FAMILY,
            "primary_control_metric": primary_control_metric,
            "time_unit": "frame_skip_sampled_step",
            "split": args.split,
            "training_data_provenance": training_bundle.provenance,
            "training_data_provenance_sha256": canonical_sha256(
                training_bundle.provenance
            ),
            "evaluation_config": (
                str(eval_config_path) if eval_config_path is not None else None
            ),
            "evaluation_config_sha256": eval_config_sha256,
            "evaluation_data_provenance": evaluation_bundle.provenance,
            "evaluation_data_provenance_sha256": canonical_sha256(
                evaluation_bundle.provenance
            ),
            "observer_gains": {
                "position_alpha": args.gain,
                "velocity_beta": args.velocity_gain,
            },
            "classic_baselines": {
                "hold_last_gain": 1.0,
                "constant_velocity_gain": 1.0,
                "constant_velocity_max_delta_px": 16.0,
                "linear_kalman_dt": 1.0,
                "linear_kalman_acceleration_variance": 1.0,
                "linear_kalman_measurement_variance": 4.0,
                "linear_kalman_initial_position_variance": 16.0,
                "linear_kalman_initial_velocity_variance": 16.0,
                "linear_kalman_psd_tolerance": 0.000001,
            },
            "synthetic_measurement_dropout_contract": {
                "type": "oracle_known_measurement_dropout",
                "rgb_occluded": False,
                "autonomous_detector": False,
                "start_offset": dropout_start_offset,
                "length": dropout_length,
            },
            "hidden_step_negative_control": {
                "measurement_hidden_from_step": hidden_step_length > 0,
                "sampled_steps": hidden_step_length,
            },
            "modes": {
                name: aggregate_sequential_metrics(values)
                for name, values in sorted(records.items())
            },
            "sources": {
                source: {
                    name: aggregate_sequential_metrics(values)
                    for name, values in sorted(modes.items())
                }
                for source, modes in sorted(source_records.items())
            },
            "tasks": {
                task: {
                    name: aggregate_sequential_metrics(values)
                    for name, values in sorted(modes.items())
                }
                for task, modes in sorted(task_records.items())
            },
            "paired_episode_differences": _paired_differences(episode_rows),
            "episode_rows": episode_rows,
        }

    conditions = (
        _matrix_conditions(args.matrix_profile)
        if args.matrix_profile is not None
        else (
            (
                "single",
                args.dropout_start_offset,
                args.dropout_length,
                args.hidden_step_length,
            ),
        )
    )
    receipts = []
    for condition_name, start, length, hidden in conditions:
        output_path = (
            args.output / f"{condition_name}.json"
            if args.matrix_profile is not None
            else args.output
        )
        if output_path.exists() and args.resume_existing:
            if not _completed_report_matches(
                output_path,
                checkpoint_sha256=checkpoint_sha256,
                evaluator_sha256=evaluator_sha256,
                implementation_files=implementation_files,
                implementation_sha256=implementation_sha256,
                evaluation_config_sha256=eval_config_sha256,
                evaluation_data_provenance_sha256=(
                    evaluation_data_provenance_sha256
                ),
                evaluation_episode_selection=evaluation_episode_selection,
                max_episodes_per_source=args.max_episodes_per_source,
                split=args.split,
                condition_name=condition_name,
                matrix_profile=args.matrix_profile,
                gain=args.gain,
                velocity_gain=args.velocity_gain,
                dropout_start_offset=start,
                dropout_length=length,
                hidden_step_length=hidden,
            ):
                raise ValueError(
                    "existing evaluation report does not match the frozen "
                    f"matrix cell: {output_path}"
                )
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            receipts.append(
                {
                    "condition": condition_name,
                    "output": str(output_path.expanduser().resolve()),
                    "output_sha256": sha256_file(output_path),
                    "episodes": len(existing["episode_rows"]),
                    "status": "skipped_verified",
                }
            )
            continue
        report = build_report(
            condition_name=condition_name,
            dropout_start_offset=start,
            dropout_length=length,
            hidden_step_length=hidden,
        )
        _atomic_write(output_path, report)
        receipts.append(
            {
                "condition": condition_name,
                "output": str(output_path.expanduser().resolve()),
                "output_sha256": sha256_file(output_path),
                "episodes": len(report["episode_rows"]),
                "status": "completed",
            }
        )
    print(
        json.dumps(
            {
                "checkpoint_sha256": checkpoint_sha256,
                "target_mode": verified["target_mode"],
                "matrix_profile": args.matrix_profile,
                "reports": receipts,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
