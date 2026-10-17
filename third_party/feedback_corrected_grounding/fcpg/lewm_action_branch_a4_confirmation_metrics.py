"""Reserved-test confirmation metrics for the seven-task A4 protocol.

This module deliberately reuses the A4 development loss and reduction
functions.  Its only new scientific responsibility is to open the reserved
test split after a complete Development-Go authorization and apply the frozen
10k confirmation contract without checkpoint selection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from fcpg.lewm_a4_storage import validate_a4_state_cache
from fcpg.lewm_action_branch_a4 import (
    CONFIRMATION_TRAINING_SEEDS,
    DEVELOPMENT_TRAINING_SEEDS,
    PROTOCOL,
    TASKS,
    assign_a4_splits,
    validate_a4_contract,
)
from fcpg.lewm_action_branch_a4_metrics import (
    FACTUAL_METHOD,
    HOLD_LAST_METHOD,
    LATENT_CONSTANT_VELOCITY_METHOD,
    NO_ACTION_METHOD,
    PRIMARY_HORIZONS,
    WRONG_ACTION_METHOD,
    WRONG_ACTION_SHIFT_COUNT,
    _cache_identity,
    _fit_task_moments,
    _ordinary_development_losses,
    _reduce_seed_episode_losses,
    _secondary_predictions,
    _seed_comparison_report,
    _seed_method_report,
    _validated_development_predictions,
    _wrong_action_centered_losses,
)


CONFIRMATION_CHECKPOINT_STEP = 10_000
DEVELOPMENT_CHECKPOINT_STEP = 2_000
CONFIRMATION_STAGE = "a4_10k_confirmation"
EXPECTED_DEVELOPMENT_CHECKS = frozenset(
    {
        "minimum_effect_vs_no_action",
        "minimum_effect_vs_every_wrong_action_shift",
        "paired_ci_lower_exceeds_zero",
        "minimum_positive_tasks",
        "minimum_positive_seeds",
        "raw_factual_not_worse_than_no_action",
        "raw_factual_better_than_hold_last",
        "raw_factual_better_than_constant_velocity",
    }
)


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be boolean")
    return value


def validate_a4_confirmation_contract(
    contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return the exact checked-in A4 confirmation Gate."""

    root = _mapping(contract, label="A4 contract")
    if validate_a4_contract(root) != TASKS:
        raise ValueError("A4 confirmation task panel drifted")
    gate = _mapping(
        root.get("confirmation_after_development_go"),
        label="A4 confirmation Gate",
    )
    observed = (
        gate.get("test_collection_requires_development_go"),
        int(gate.get("optimizer_steps", -1)),
        tuple(int(value) for value in gate.get("seeds", ())),
        gate.get("fresh_start_from_same_frozen_base"),
        gate.get("development_checkpoint_or_optimizer_reused"),
        gate.get("selection"),
        gate.get("evaluation_split"),
        int(gate.get("checkpoint_step", -1)),
        tuple(int(value) for value in gate.get("primary_horizons", ())),
        int(gate.get("minimum_positive_tasks", -1)),
        int(gate.get("minimum_positive_seeds", -1)),
        gate.get("must_beat_matched_no_action"),
        gate.get("must_beat_all_seven_wrong_action_shifts"),
        gate.get("must_beat_hold_last_and_constant_velocity"),
    )
    expected = (
        True,
        CONFIRMATION_CHECKPOINT_STEP,
        CONFIRMATION_TRAINING_SEEDS,
        True,
        False,
        "fixed_latest",
        "test",
        CONFIRMATION_CHECKPOINT_STEP,
        PRIMARY_HORIZONS,
        6,
        2,
        True,
        True,
        True,
    )
    if observed != expected:
        raise ValueError("A4 confirmation Gate contract drifted")
    return gate


def validate_a4_development_go_authorization(
    summary: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> dict[str, int]:
    """Validate the complete 2k decision before any reserved-test read."""

    root = _mapping(summary, label="A4 Development-Go summary")
    development_gate = _mapping(
        _mapping(contract, label="A4 contract").get("development_gate"),
        label="A4 development Gate",
    )
    validate_a4_confirmation_contract(contract)
    header = (
        root.get("protocol"),
        root.get("stage"),
        root.get("mode"),
        root.get("decision"),
        root.get("passed"),
        root.get("development_go"),
        root.get("reserved_test_collection_authorized"),
        root.get("input_contains_test_rows"),
        root.get("checkpoint_step"),
        tuple(root.get("primary_horizons", ())),
    )
    expected_header = (
        PROTOCOL,
        "a4_2k_formal_development",
        "formal",
        "Development-Go",
        True,
        True,
        True,
        False,
        DEVELOPMENT_CHECKPOINT_STEP,
        PRIMARY_HORIZONS,
    )
    if header != expected_header:
        raise PermissionError(
            "A4 reserved test requires the exact formal 2k Development-Go summary"
        )

    coverage = _mapping(root.get("coverage"), label="Development-Go coverage")
    if tuple(coverage.get("tasks", ())) != TASKS or tuple(
        coverage.get("training_seeds", ())
    ) != DEVELOPMENT_TRAINING_SEEDS:
        raise PermissionError("Development-Go task or seed coverage drifted")
    if int(coverage.get("development_bundle_count", 0)) <= 0:
        raise PermissionError("Development-Go has no development bundles")

    whitening = _mapping(root.get("whitening"), label="Development-Go whitening")
    if (
        whitening.get("source") != "episode_equal_train_rows"
        or whitening.get("per_task") is not True
        or tuple(whitening.get("fit_horizons", ())) != PRIMARY_HORIZONS
    ):
        raise PermissionError("Development-Go whitening contract drifted")
    whitening_tasks = _mapping(
        whitening.get("tasks"), label="Development-Go whitening tasks"
    )
    if set(whitening_tasks) != set(TASKS):
        raise PermissionError("Development-Go whitening does not cover seven tasks")
    train_episode_counts: dict[str, int] = {}
    minimum_train = int(
        contract["support_gate"]["minimum_active_bundle_episode_count_per_task"]
    )
    for task_name in TASKS:
        row = _mapping(
            whitening_tasks[task_name],
            label=f"Development-Go whitening for {task_name}",
        )
        count = int(row.get("train_physical_episode_count", -1))
        if count < minimum_train or count > 24:
            raise PermissionError(
                f"Development-Go train coverage is invalid for {task_name}"
            )
        train_episode_counts[task_name] = count

    primary = _mapping(root.get("primary"), label="Development-Go primary")
    if primary.get("metric") != "h6_h8_within_bundle_branch_centered_whitened_mse":
        raise PermissionError("Development-Go primary metric drifted")
    comparisons = _mapping(
        primary.get("comparisons"), label="Development-Go comparisons"
    )
    no_action = _mapping(
        comparisons.get(NO_ACTION_METHOD), label="Development-Go no-action comparison"
    )
    shifts = _mapping(
        primary.get("wrong_action_shifts"),
        label="Development-Go wrong-action shifts",
    )
    shift_rows = shifts.get("per_shift")
    if (
        int(shifts.get("count", -1)) != WRONG_ACTION_SHIFT_COUNT
        or not isinstance(shift_rows, list)
        or len(shift_rows) != WRONG_ACTION_SHIFT_COUNT
    ):
        raise PermissionError("Development-Go must contain all seven wrong-action shifts")
    ordered_shifts: list[Mapping[str, Any]] = []
    for expected_shift, value in enumerate(shift_rows, start=1):
        row = _mapping(value, label=f"Development-Go wrong-action shift {expected_shift}")
        if int(row.get("donor_shift", -1)) != expected_shift:
            raise PermissionError("Development-Go wrong-action donor order drifted")
        ordered_shifts.append(row)

    minimum_effect = float(development_gate["minimum_relative_improvement"])
    lower_threshold = float(development_gate["paired_ci_lower_must_exceed"])
    for label, row in [(NO_ACTION_METHOD, no_action), *(
        (f"wrong_action_shift_{index}", value)
        for index, value in enumerate(ordered_shifts, start=1)
    )]:
        ci = _mapping(row.get("paired_ci"), label=f"Development-Go {label} CI")
        if (
            float(row.get("relative_improvement", float("-inf"))) < minimum_effect
            or float(ci.get("lower", float("-inf"))) <= lower_threshold
        ):
            raise PermissionError(f"Development-Go {label} does not pass its frozen Gate")

    raw = _mapping(root.get("raw_h6_h8"), label="Development-Go raw H6-H8")
    raw_methods = _mapping(raw.get("methods"), label="Development-Go raw methods")
    raw_mse = {
        method: float(
            _mapping(raw_methods.get(method), label=f"Development-Go raw {method}").get(
                "task_macro_mse", float("nan")
            )
        )
        for method in (
            FACTUAL_METHOD,
            NO_ACTION_METHOD,
            HOLD_LAST_METHOD,
            LATENT_CONSTANT_VELOCITY_METHOD,
        )
    }
    if not all(np.isfinite(list(raw_mse.values()))) or not (
        raw_mse[FACTUAL_METHOD] <= raw_mse[NO_ACTION_METHOD]
        and raw_mse[FACTUAL_METHOD] < raw_mse[HOLD_LAST_METHOD]
        and raw_mse[FACTUAL_METHOD] < raw_mse[LATENT_CONSTANT_VELOCITY_METHOD]
    ):
        raise PermissionError("Development-Go raw baseline checks do not pass")

    gate = _mapping(root.get("gate"), label="Development-Go nested Gate")
    checks = _mapping(gate.get("checks"), label="Development-Go checks")
    if set(checks) != EXPECTED_DEVELOPMENT_CHECKS or any(
        _bool(value, label=f"Development-Go check {name}") is not True
        for name, value in checks.items()
    ):
        raise PermissionError("Development-Go checks are incomplete")
    if (
        gate.get("passed") is not True
        or gate.get("development_go") is not True
        or gate.get("failed_checks") != []
    ):
        raise PermissionError("Development-Go nested decision is inconsistent")
    jointly_positive_tasks = gate.get("jointly_positive_tasks")
    jointly_positive_seeds = gate.get("jointly_positive_seeds")
    if (
        not isinstance(jointly_positive_tasks, list)
        or not set(jointly_positive_tasks).issubset(TASKS)
        or len(jointly_positive_tasks) < int(development_gate["minimum_positive_tasks"])
        or not isinstance(jointly_positive_seeds, list)
        or not set(jointly_positive_seeds).issubset(DEVELOPMENT_TRAINING_SEEDS)
        or len(jointly_positive_seeds) < int(development_gate["minimum_positive_seeds"])
    ):
        raise PermissionError("Development-Go task/seed replication is insufficient")
    return train_episode_counts


def _validate_frozen_split_assignments(
    cache: Mapping[str, np.ndarray],
    *,
    contract: Mapping[str, Any],
    allowed_splits: frozenset[str],
    label: str,
) -> np.ndarray:
    """Validate every row against the frozen task-instance split assignment."""

    validate_a4_state_cache(cache)
    split = np.asarray(cache["split"]).astype(str, copy=False)
    observed_splits = set(split.tolist())
    if not observed_splits or not observed_splits.issubset(allowed_splits):
        raise ValueError(
            f"{label} may contain only {sorted(allowed_splits)}, got "
            f"{sorted(observed_splits)}"
        )
    assignments = assign_a4_splits(TASKS, contract)
    tasks = np.asarray(cache["task_name"]).astype(str, copy=False)
    instances = np.asarray(cache["task_instance_index"], dtype=np.int64)
    episodes = np.asarray(cache["source_episode_id"]).astype(str, copy=False)
    for task_name, instance, observed_split, episode in zip(
        tasks, instances, split, episodes, strict=True
    ):
        expected_split = assignments[(str(task_name), int(instance))]
        if str(observed_split) != expected_split:
            raise PermissionError(
                f"{label} row for {task_name} instance {int(instance)} is labeled "
                f"{observed_split!r}, but the frozen assignment is "
                f"{expected_split!r}"
            )
        expected_episode = f"{task_name}:task-instance-{int(instance):02d}"
        if str(episode) != expected_episode:
            raise ValueError(
                f"{label} source_episode_id does not match its frozen task-instance "
                "identity"
            )
    return split


def validate_a4_confirmation_caches(
    train_cache: Mapping[str, np.ndarray],
    test_cache: Mapping[str, np.ndarray],
    *,
    authorized_train_episode_counts: Mapping[str, int],
    contract: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    """Validate isolated train/development and reserved-test caches."""

    train_split = _validate_frozen_split_assignments(
        train_cache,
        contract=contract,
        allowed_splits=frozenset({"train", "development"}),
        label="A4 confirmation train cache",
    )
    if "train" not in set(train_split.tolist()):
        raise ValueError("A4 confirmation train cache has no train rows")
    test_split = _validate_frozen_split_assignments(
        test_cache,
        contract=contract,
        allowed_splits=frozenset({"test"}),
        label="A4 confirmation test cache",
    )
    if set(test_split.tolist()) != {"test"}:
        raise ValueError("A4 confirmation test cache must be test-only")

    train_task = np.asarray(train_cache["task_class_index"], dtype=np.int64)
    train_episode = np.asarray(train_cache["source_episode_id"]).astype(
        str, copy=False
    )
    test_task = np.asarray(test_cache["task_class_index"], dtype=np.int64)
    test_episode = np.asarray(test_cache["source_episode_id"]).astype(str, copy=False)
    train_instances = np.asarray(train_cache["task_instance_index"], dtype=np.int64)
    test_instances = np.asarray(test_cache["task_instance_index"], dtype=np.int64)
    if set(np.asarray(train_cache["proposal_id"]).astype(str).tolist()) & set(
        np.asarray(test_cache["proposal_id"]).astype(str).tolist()
    ):
        raise ValueError("A4 confirmation train/test proposal_id values overlap")
    expected_test = int(contract["split"]["test_episodes_per_task"])
    coverage: dict[str, dict[str, int]] = {}
    for task_index, task_name in enumerate(TASKS):
        train_rows = train_task == task_index
        test_rows = test_task == task_index
        train_ids = set(
            train_episode[train_rows & (train_split == "train")].tolist()
        )
        development_ids = set(
            train_episode[train_rows & (train_split == "development")].tolist()
        )
        test_ids = set(test_episode[test_rows].tolist())
        train_instance_ids = set(train_instances[train_rows].tolist())
        test_instance_ids = set(test_instances[test_rows].tolist())
        if train_instance_ids & test_instance_ids:
            raise ValueError(f"A4 confirmation instance leakage for {task_name}")
        expected_train = int(authorized_train_episode_counts[task_name])
        if len(train_ids) != expected_train:
            raise ValueError(
                f"A4 confirmation train coverage for {task_name} must match "
                f"Development-Go ({expected_train}), got {len(train_ids)}"
            )
        if len(test_ids) != expected_test:
            raise ValueError(
                f"A4 confirmation test coverage for {task_name} must be exactly "
                f"{expected_test}, got {len(test_ids)}"
            )
        coverage[task_name] = {
            "train_physical_episode_count": len(train_ids),
            "development_physical_episode_count": len(development_ids),
            "test_physical_episode_count": len(test_ids),
            "test_bundle_count": int(np.count_nonzero(test_rows)),
        }
    return coverage


def _summarize_confirmation_panel(
    train_cache: Mapping[str, np.ndarray],
    test_cache: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    *,
    gate: Mapping[str, Any],
    training_seeds: Sequence[int],
    coverage: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    """Apply the development-identical statistics to test rows."""

    seeds = tuple(int(seed) for seed in training_seeds)
    if seeds != CONFIRMATION_TRAINING_SEEDS:
        raise ValueError(
            f"A4 confirmation seeds must remain {CONFIRMATION_TRAINING_SEEDS}"
        )
    train_split = np.asarray(train_cache["split"]).astype(str, copy=False)
    _, train_task, _train_task_name, train_episode, _ = _cache_identity(train_cache)
    _, evaluation_task, _task_name, evaluation_episode, evaluation_split = (
        _cache_identity(test_cache)
    )
    horizons = tuple(int(value) for value in gate["primary_horizons"])
    if horizons != PRIMARY_HORIZONS:
        raise ValueError("A4 confirmation horizons must remain H6-H8")
    horizon_indices = np.asarray(horizons, dtype=np.int64) - 1
    train_target = np.asarray(train_cache["target_state"])
    evaluation_target = np.asarray(test_cache["target_state"])
    evaluation_history = np.asarray(test_cache["history_state"])
    if set(evaluation_split.tolist()) != {"test"}:
        raise ValueError("A4 confirmation evaluation rows must be test-only")
    checked = _validated_development_predictions(
        predictions,
        seed_count=len(seeds),
        evaluation_count=int(evaluation_target.shape[0]),
        target_shape=tuple(evaluation_target.shape[1:]),
    )
    moments = _fit_task_moments(
        train_target,
        task=train_task,
        episode=train_episode,
        split=train_split,
        horizon_indices=horizon_indices,
    )

    centered_losses: dict[str, np.ndarray] = {}
    raw_losses: dict[str, np.ndarray] = {}
    for method in (FACTUAL_METHOD, NO_ACTION_METHOD):
        centered_losses[method], raw_losses[method] = _ordinary_development_losses(
            evaluation_target,
            checked[method],
            task=evaluation_task,
            moments=moments,
            horizon_indices=horizon_indices,
        )
    wrong_losses = _wrong_action_centered_losses(
        evaluation_target,
        checked[WRONG_ACTION_METHOD],
        task=evaluation_task,
        moments=moments,
        horizon_indices=horizon_indices,
    )
    centered_losses[WRONG_ACTION_METHOD] = np.mean(
        wrong_losses, axis=2, dtype=np.float64
    )
    secondary = _secondary_predictions(evaluation_history)
    for method in (HOLD_LAST_METHOD, LATENT_CONSTANT_VELOCITY_METHOD):
        prediction = np.broadcast_to(
            secondary[method][None], (len(seeds), *secondary[method].shape)
        )
        _, raw_losses[method] = _ordinary_development_losses(
            evaluation_target,
            prediction,
            task=evaluation_task,
            moments=moments,
            horizon_indices=horizon_indices,
        )

    centered_reductions = {
        method: _reduce_seed_episode_losses(
            loss,
            task=evaluation_task,
            episode=evaluation_episode,
            seeds=seeds,
        )
        for method, loss in centered_losses.items()
    }
    raw_reductions = {
        method: _reduce_seed_episode_losses(
            loss,
            task=evaluation_task,
            episode=evaluation_episode,
            seeds=seeds,
        )
        for method, loss in raw_losses.items()
    }
    shift_reductions = [
        _reduce_seed_episode_losses(
            wrong_losses[:, :, shift],
            task=evaluation_task,
            episode=evaluation_episode,
            seeds=seeds,
        )
        for shift in range(WRONG_ACTION_SHIFT_COUNT)
    ]
    bootstrap_samples = int(gate["bootstrap_samples"])
    no_action_comparison = _seed_comparison_report(
        centered_reductions[FACTUAL_METHOD],
        centered_reductions[NO_ACTION_METHOD],
        baseline_method=NO_ACTION_METHOD,
        seeds=seeds,
        bootstrap_samples=bootstrap_samples,
    )
    wrong_average_comparison = _seed_comparison_report(
        centered_reductions[FACTUAL_METHOD],
        centered_reductions[WRONG_ACTION_METHOD],
        baseline_method="wrong_action_mean",
        seeds=seeds,
        bootstrap_samples=bootstrap_samples,
    )
    shift_reports = [
        {
            "donor_shift": shift + 1,
            **_seed_comparison_report(
                centered_reductions[FACTUAL_METHOD],
                reduction,
                baseline_method=f"wrong_action_shift_{shift + 1}",
                seeds=seeds,
                bootstrap_samples=bootstrap_samples,
            ),
        }
        for shift, reduction in enumerate(shift_reductions)
    ]
    raw_reports = {
        method: _seed_method_report(reduction, seeds=seeds)
        for method, reduction in raw_reductions.items()
    }

    jointly_positive_tasks = [
        task_name
        for task_name in TASKS
        if no_action_comparison["per_task"][task_name]["relative_improvement"] > 0.0
        and all(
            row["per_task"][task_name]["relative_improvement"] > 0.0
            for row in shift_reports
        )
    ]
    jointly_positive_seeds = [
        seed
        for seed in seeds
        if no_action_comparison["per_seed"][str(seed)]["relative_improvement"] > 0.0
        and all(
            row["per_seed"][str(seed)]["relative_improvement"] > 0.0
            for row in shift_reports
        )
    ]
    minimum_effect = float(gate["minimum_relative_improvement"])
    ci_reports = [no_action_comparison, *shift_reports]
    checks = {
        "minimum_effect_vs_no_action": no_action_comparison["relative_improvement"]
        >= minimum_effect,
        "minimum_effect_vs_every_wrong_action_shift": min(
            row["relative_improvement"] for row in shift_reports
        )
        >= minimum_effect,
        "paired_ci_lower_exceeds_zero": all(
            row["paired_ci"]["lower"]
            > float(gate["paired_ci_lower_must_exceed"])
            for row in ci_reports
        ),
        "minimum_positive_tasks": len(jointly_positive_tasks)
        >= int(gate["minimum_positive_tasks"]),
        "minimum_positive_seeds": len(jointly_positive_seeds)
        >= int(gate["minimum_positive_seeds"]),
        "raw_factual_not_worse_than_no_action": raw_reports[FACTUAL_METHOD][
            "task_macro_mse"
        ]
        <= raw_reports[NO_ACTION_METHOD]["task_macro_mse"],
        "raw_factual_better_than_hold_last": raw_reports[FACTUAL_METHOD][
            "task_macro_mse"
        ]
        < raw_reports[HOLD_LAST_METHOD]["task_macro_mse"],
        "raw_factual_better_than_constant_velocity": raw_reports[FACTUAL_METHOD][
            "task_macro_mse"
        ]
        < raw_reports[LATENT_CONSTANT_VELOCITY_METHOD]["task_macro_mse"],
    }
    passed = all(checks.values())
    decision = "Confirmation-Go" if passed else "Confirmation-No-Go"
    test_bundle_count = int(evaluation_target.shape[0])
    return {
        "protocol": PROTOCOL,
        "stage": CONFIRMATION_STAGE,
        "decision": decision,
        "passed": bool(passed),
        "confirmation_go": bool(passed),
        "development_go_authorization_validated_before_test_read": True,
        "evaluation_split": "test",
        "input_contains_development_rows": False,
        "checkpoint_step": CONFIRMATION_CHECKPOINT_STEP,
        "selection": "fixed_latest",
        "checkpoint_selection_performed": False,
        "primary_horizons": list(horizons),
        "aggregation": [
            "bundle_branches_and_horizons",
            "bundle_within_training_seed_and_physical_episode",
            "training_seed_within_physical_episode",
            "paired_source_physical_episode_bootstrap_within_task",
            "equal_weight_task_macro",
        ],
        "coverage": {
            "tasks": list(TASKS),
            "training_seeds": list(seeds),
            "test_bundle_count": test_bundle_count,
            "per_task": {name: dict(coverage[name]) for name in TASKS},
        },
        "whitening": {
            "source": "episode_equal_train_rows",
            "per_task": True,
            "fit_horizons": list(horizons),
            "tasks": {
                task_name: {
                    "train_physical_episode_count": moments[task_name].episode_count,
                    "floored_dimensions": moments[task_name].floored_dimensions,
                }
                for task_name in TASKS
            },
        },
        "primary": {
            "metric": "h6_h8_within_bundle_branch_centered_whitened_mse",
            "methods": {
                method: _seed_method_report(reduction, seeds=seeds)
                for method, reduction in centered_reductions.items()
            },
            "comparisons": {
                NO_ACTION_METHOD: no_action_comparison,
                "wrong_action_mean": wrong_average_comparison,
            },
            "wrong_action_shifts": {
                "count": WRONG_ACTION_SHIFT_COUNT,
                "minimum_relative_improvement": min(
                    row["relative_improvement"] for row in shift_reports
                ),
                "per_shift": shift_reports,
            },
        },
        "raw_h6_h8": {
            "metric": "h6_h8_raw_whitened_trajectory_mse",
            "methods": raw_reports,
        },
        "gate": {
            "minimum_relative_improvement": minimum_effect,
            "minimum_positive_tasks": int(gate["minimum_positive_tasks"]),
            "minimum_positive_seeds": int(gate["minimum_positive_seeds"]),
            "checks": checks,
            "failed_checks": [name for name, value in checks.items() if not value],
            "jointly_positive_tasks": jointly_positive_tasks,
            "jointly_positive_seeds": jointly_positive_seeds,
            "passed": bool(passed),
            "confirmation_go": bool(passed),
        },
        "claim_boundary": {
            "positive_confirmation_claim_authorized": bool(passed),
            "authorized_claim": (
                "Three fresh fixed-latest step-10000 shared A4 models improve "
                "reserved-test H6-H8 action-conditioned anchor/task-state prediction "
                "over matched no-action and every cyclic wrong-action shift, while "
                "also beating hold-last and latent constant velocity."
                if passed
                else None
            ),
            "negative_result_statement": (
                None
                if passed
                else (
                    "The frozen A4 10k reserved-test confirmation did not satisfy "
                    "all preregistered Gates."
                )
            ),
            "does_not_establish": [
                "policy rollout or task-success improvement",
                "model-based action selection or CEM benefit",
                "generalization beyond the seven A4 tasks and reserved episodes",
                "benefit from checkpoint selection or test-set tuning",
            ],
        },
    }


def summarize_a4_confirmation(
    train_cache: Mapping[str, np.ndarray],
    test_cache: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    *,
    contract: Mapping[str, Any],
    development_go_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Score the authorized fixed 10k A4 test panel exactly once."""

    gate = validate_a4_confirmation_contract(contract)
    train_counts = validate_a4_development_go_authorization(
        development_go_summary, contract=contract
    )
    coverage = validate_a4_confirmation_caches(
        train_cache,
        test_cache,
        authorized_train_episode_counts=train_counts,
        contract=contract,
    )
    return _summarize_confirmation_panel(
        train_cache,
        test_cache,
        predictions,
        gate=gate,
        training_seeds=CONFIRMATION_TRAINING_SEEDS,
        coverage=coverage,
    )


__all__ = [
    "CONFIRMATION_CHECKPOINT_STEP",
    "CONFIRMATION_STAGE",
    "summarize_a4_confirmation",
    "validate_a4_confirmation_caches",
    "validate_a4_confirmation_contract",
    "validate_a4_development_go_authorization",
]
