#!/usr/bin/env python3
"""Run a zero-training LeWM predictor-isolation development Gate.

The evaluator uses the offline sensor-valid mask only to route an interrupted
anchor to each causal predictor. It does not train or evaluate a deployable
selector. Passing this Gate only authorizes a separate Random-Hazard data and
predictor study.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_posterior_predictor_statistics import (  # noqa: E402
    summarize_predictor_isolation_statistics,
)
from fcpg.lewm_anchor_dynamics_pair import (  # noqa: E402
    load_matched_predictor_pair_checkpoint,
)
from fcpg.lewm_posterior_randomized_predictors import (  # noqa: E402
    FACTUAL_LEWM_METHOD,
    HOLD_LAST_METHOD,
    LATENT_CONSTANT_VELOCITY_METHOD,
    MATCHED_NO_ACTION_METHOD,
    SHUFFLED_ACTION_LEWM_METHOD,
    ZERO_ACTION_LEWM_METHOD,
    run_randomized_predictor_isolation,
)
from fcpg.lewm_runtime import choose_device, load_lewm_grounded_checkpoint  # noqa: E402
from scripts import evaluate_lewm_posterior_randomized_oracle as oracle_cli  # noqa: E402


EpisodeKey = tuple[str, str]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = oracle_cli.build_parser()
    parser.add_argument(
        "--predictor-pair-checkpoint",
        type=Path,
        help=(
            "Optional fixed-budget matched factual/no-action predictor pair. "
            "When supplied, it replaces the frozen-model zero-action ablation."
        ),
    )
    return parser.parse_args(argv)


def _episode_key(row: Any) -> EpisodeKey:
    return str(row.task), str(row.episode_id)


def build_cross_source_action_shuffle(
    metadata: Sequence[Any],
    resampling_units: Mapping[EpisodeKey, str],
) -> tuple[int, ...]:
    """Build a same-task/phase/event donor permutation across source trajectories."""

    rows = tuple(metadata)
    if not rows:
        raise ValueError("metadata must not be empty")
    episode_keys = {_episode_key(row) for row in rows}
    normalized_units = {
        (str(task), str(episode)): str(unit)
        for (task, episode), unit in resampling_units.items()
    }
    if set(normalized_units) != episode_keys:
        raise ValueError(
            "resampling_units must cover exactly the predictor view episodes"
        )

    grouped: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[(str(row.task), int(row.phase_index), str(row.event))].append(index)

    mapping = [-1] * len(rows)
    for group_key, indices in sorted(grouped.items()):
        by_source: dict[str, list[int]] = defaultdict(list)
        for index in indices:
            by_source[normalized_units[_episode_key(rows[index])]].append(index)
        if len(by_source) < 2:
            raise ValueError(
                f"action shuffle needs two source trajectories for {group_key!r}"
            )
        ordered: list[int] = []
        for source in sorted(by_source):
            ordered.extend(
                sorted(
                    by_source[source],
                    key=lambda index: (_episode_key(rows[index]), index),
                )
            )
        maximum_source_rows = max(len(values) for values in by_source.values())
        if 2 * maximum_source_rows > len(ordered):
            raise ValueError(
                "no cross-source action derangement exists for "
                f"{group_key!r}: largest source has {maximum_source_rows}/"
                f"{len(ordered)} rows"
            )
        donors = ordered[maximum_source_rows:] + ordered[:maximum_source_rows]
        for target, donor in zip(ordered, donors, strict=True):
            target_source = normalized_units[_episode_key(rows[target])]
            donor_source = normalized_units[_episode_key(rows[donor])]
            if target_source == donor_source:
                raise AssertionError("cross-source rotation retained a source")
            mapping[target] = donor

    if sorted(mapping) != list(range(len(rows))):
        raise RuntimeError("action shuffle did not produce a full permutation")
    return tuple(mapping)


def build_predictor_records(
    result: Any,
    examples: Any,
    *,
    direct_anchors: Mapping[EpisodeKey, int],
    resampling_units: Mapping[EpisodeKey, str],
    camera_views: Mapping[EpisodeKey, str],
) -> tuple[dict[str, Any], ...]:
    """Score active rows after the first hidden event on the interrupted anchor."""

    rows = tuple(examples.metadata)
    canonical = examples.labels.canonical_current
    if torch.is_tensor(canonical):
        target = canonical.detach().cpu().numpy()
    else:
        target = np.asarray(canonical)
    dropout = examples.evaluation.dropout_active
    if torch.is_tensor(dropout):
        active = dropout.detach().cpu().numpy()
    else:
        active = np.asarray(dropout)
    expected_episodes = {_episode_key(row) for row in rows}
    for name, values in (
        ("direct_anchors", direct_anchors),
        ("resampling_units", resampling_units),
        ("camera_views", camera_views),
    ):
        if set(values) != expected_episodes:
            raise ValueError(f"{name} must cover exactly the evaluated views")

    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if int(row.event_index) == 0:
            continue
        key = _episode_key(row)
        anchor = int(direct_anchors[key])
        if not bool(active[row_index, anchor]):
            continue
        for method, trajectory in result.trajectories.items():
            prediction = trajectory.post_update[row_index, anchor]
            if torch.is_tensor(prediction):
                predicted = prediction.detach().cpu().numpy()
            else:
                predicted = np.asarray(prediction)
            squared_error = float(np.square(predicted - target[row_index, anchor]).mean())
            if not np.isfinite(squared_error):
                raise ValueError("predictor squared error must be finite")
            records.append(
                {
                    "task": str(row.task),
                    "view_episode_id": key[1],
                    "source_trajectory_id": str(resampling_units[key]),
                    "camera_view": str(camera_views[key]),
                    "event_id": str(row.event),
                    "phase_index": int(row.phase_index),
                    "method": str(method),
                    "squared_error": squared_error,
                }
            )
    if not records:
        raise ValueError("no post-onset active predictor rows were scored")
    return tuple(records)


def _markdown(payload: Mapping[str, Any]) -> str:
    summary = payload["statistics"]
    lines = [
        "# Randomized Predictor Isolation",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "This is a zero-training development Gate with a privileged sensor-valid "
        "mask. It isolates predictor quality; it is not a deployable selector or "
        "policy-control result.",
        "",
        "| Baseline | Factual relative improvement | 95% CI | Positive tasks |",
        "| --- | ---: | ---: | ---: |",
    ]
    matched = bool(payload["scope"].get("matched_no_action_is_independently_trained"))
    labels = {
        "hold_last": "Hold-last",
        "constant_velocity": "Latent constant velocity",
        "zero_action": (
            "Matched no-action recurrent" if matched else "Zero-action LeWM ablation"
        ),
        "shuffled_action": "Cross-source shuffled-action LeWM",
    }
    for role, label in labels.items():
        row = summary["comparisons"][role]
        low, high = row["relative_improvement_95_ci"]
        lines.append(
            f"| {label} | {row['relative_improvement']:.2%} | "
            f"[{low:.2%}, {high:.2%}] | "
            f"{row['positive_task_count']}/{row['task_count']} |"
        )
    limitation = (
        "The factual and no-action heads were trained from identical initialization "
        "against one frozen latent target; action visibility is their only factor."
        if matched
        else "The zero-action cell is an input ablation of the frozen factual model, "
        "not a separately trained no-action recurrent baseline."
    )
    lines.extend(
        [
            "",
            limitation + " Latent CV is the minimal missing-measurement motion "
            "baseline; FK/KF and rollout remain requirements of a later full "
            "Random-Hazard Gate.",
        ]
    )
    if summary["gate"]["failed_checks"]:
        lines.extend(
            [
                "",
                "Failed checks: "
                + ", ".join(
                    f"`{name}`" for name in summary["gate"]["failed_checks"]
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    layout = oracle_cli._resolve_input_layout(args)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)

    calibration, evaluation = oracle_cli._load_inputs(layout)
    checkpoint_path = oracle_cli._resolve_file(args.checkpoint, label="LeWM checkpoint")
    device = choose_device(args.device)
    model, _, checkpoint = load_lewm_grounded_checkpoint(
        checkpoint_path, device=device
    )
    datasets = [calibration.arrays]
    if evaluation.arrays is not calibration.arrays:
        datasets.append(evaluation.arrays)
    model = oracle_cli._validate_frozen_model(model, checkpoint, datasets)

    factual_predictor: object = model
    matched_no_action: object | None = None
    pair_checkpoint: dict[str, Any] | None = None
    methods = None
    pair_path: Path | None = None
    if args.predictor_pair_checkpoint is not None:
        pair_path = oracle_cli._resolve_file(
            args.predictor_pair_checkpoint, label="predictor-pair checkpoint"
        )
        factual_predictor, matched_no_action, pair_checkpoint = (
            load_matched_predictor_pair_checkpoint(pair_path, device=device)
        )
        configured_base = Path(str(pair_checkpoint["base_checkpoint"])).resolve()
        if configured_base != checkpoint_path:
            raise ValueError(
                "predictor pair was trained in a different base latent space: "
                f"{configured_base} != {checkpoint_path}"
            )
        expected_steps = int(pair_checkpoint["config"]["trainer"]["max_steps"])
        if int(pair_checkpoint["step"]) != expected_steps:
            raise ValueError("predictor pair has not reached its fixed training budget")
        methods = (
            FACTUAL_LEWM_METHOD,
            HOLD_LAST_METHOD,
            LATENT_CONSTANT_VELOCITY_METHOD,
            MATCHED_NO_ACTION_METHOD,
            SHUFFLED_ACTION_LEWM_METHOD,
        )

    observed, canonical = oracle_cli._encode_split(
        model, evaluation, batch_size=args.batch_size, device=device
    )
    examples = oracle_cli._build_sequence(
        evaluation.arrays,
        evaluation.indices,
        measurement_embedding=observed,
        canonical_embedding=canonical,
        q_corrected=evaluation.observed_q,
    )
    resampling_units = oracle_cli._resampling_units(
        evaluation.arrays, evaluation.indices
    )
    shuffle = build_cross_source_action_shuffle(
        examples.metadata, resampling_units
    )
    result = run_randomized_predictor_isolation(
        factual_predictor,
        examples,
        methods=methods,
        matched_no_action_lewm=matched_no_action,
        shuffled_action_row_mapping=shuffle,
        device=device,
    )
    view_groups = oracle_cli._selected_episode_groups(
        evaluation.arrays, evaluation.indices
    )
    camera_views = {
        (str(group.key[0]), str(group.key[1])): str(group.trajectory_mode)
        for group in view_groups
    }
    records = build_predictor_records(
        result,
        examples,
        direct_anchors=oracle_cli._direct_anchors(examples),
        resampling_units=resampling_units,
        camera_views=camera_views,
    )
    statistics = summarize_predictor_isolation_statistics(
        records,
        factual_method=FACTUAL_LEWM_METHOD,
        hold_last_method=HOLD_LAST_METHOD,
        constant_velocity_method=LATENT_CONSTANT_VELOCITY_METHOD,
        zero_action_method=(
            MATCHED_NO_ACTION_METHOD
            if matched_no_action is not None
            else ZERO_ACTION_LEWM_METHOD
        ),
        shuffled_action_method=SHUFFLED_ACTION_LEWM_METHOD,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    payload = {
        "schema_version": 1,
        "status": "randomized_predictor_isolation_development",
        "decision": statistics["decision"],
        "scope": {
            "selector_trained": False,
            "sensor_valid_mask_is_privileged": True,
            "primary_excludes_first_hidden_event": True,
            "zero_action_is_frozen_model_ablation": matched_no_action is None,
            "matched_no_action_is_independently_trained": matched_no_action is not None,
            "authorizes_only_full_random_hazard_study": True,
        },
        "inputs": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": int(checkpoint["step"]),
            "predictor_pair_checkpoint": str(pair_path) if pair_path else None,
            "predictor_pair_step": (
                int(pair_checkpoint["step"]) if pair_checkpoint is not None else None
            ),
            "evaluation": oracle_cli._split_report(evaluation),
        },
        "evaluation": {
            "sequence_rows": int(examples.sample_count),
            "scored_records": len(records),
            "methods": list(result.trajectories),
        },
        "statistics": statistics,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": statistics["decision"],
                "output_dir": str(output_dir),
                "source_trajectory_count": statistics["coverage"][
                    "source_trajectory_count"
                ],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
