from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import pytest

from fcpg.lewm_anchor_ablation import PARAMETER_COMPATIBLE_TRIPLET_MODES
from fcpg.lewm_matrix_report import render_anchor_visual_markdown


MODES = PARAMETER_COMPATIBLE_TRIPLET_MODES
TASKS = ("button-press-v3", "pick-place-v3", "push-v3", "reach-v3")
TASK_EQUAL_ESTIMAND = (
    "mean_task(1 - mean_episode,H candidate_ratio / "
    "mean_episode,H control_ratio)"
)


def _pair_effect(
    *,
    candidates: tuple[float, ...],
    controls: tuple[float, ...],
    episode_counts: tuple[int, ...],
    confidence_interval: tuple[float, float],
    one_sided_ratio_upper: float,
    passed: bool = True,
) -> dict[str, Any]:
    ratios = tuple(
        candidate / control
        for candidate, control in zip(candidates, controls, strict=True)
    )
    per_task = {
        task: {
            "candidate": candidate,
            "control": control,
            "relative_improvement": 1.0 - ratio,
            "episode_count": episode_count,
        }
        for task, candidate, control, ratio, episode_count in zip(
            TASKS,
            candidates,
            controls,
            ratios,
            episode_counts,
            strict=True,
        )
    }
    task_count = len(TASKS)
    candidate_to_control_ratio = math.fsum(ratios) / task_count
    return {
        "estimand": TASK_EQUAL_ESTIMAND,
        "candidate_macro": math.fsum(candidates) / task_count,
        "control_macro": math.fsum(controls) / task_count,
        "relative_improvement": 1.0 - candidate_to_control_ratio,
        "relative_improvement_95_ci": list(confidence_interval),
        "candidate_to_control_ratio": candidate_to_control_ratio,
        "candidate_to_control_ratio_one_sided_95_upper": one_sided_ratio_upper,
        "per_task": per_task,
        "episode_count": sum(episode_counts),
        "task_count": task_count,
        "passed": passed,
    }


def _development_gate() -> dict[str, Any]:
    controls = (1.0, 2.0, 3.0, 4.0)
    primary = _pair_effect(
        candidates=(0.90, 1.84, 2.82, 3.84),
        controls=controls,
        episode_counts=(28, 28, 28, 28),
        confidence_interval=(0.04, 0.10),
        one_sided_ratio_upper=0.96,
    )
    clean = _pair_effect(
        candidates=(1.00, 2.02, 2.97, 4.08),
        controls=controls,
        episode_counts=(14, 14, 14, 14),
        confidence_interval=(-0.02, 0.01),
        one_sided_ratio_upper=1.03,
    )
    unseen = _pair_effect(
        candidates=(0.90, 1.84, 2.82, 3.84),
        controls=controls,
        episode_counts=(8, 7, 7, 7),
        confidence_interval=(0.02, 0.11),
        one_sided_ratio_upper=0.97,
    )
    unseen["selection_disclosure"] = (
        "Excluded from corrector gradient training but used for corrector "
        "validation and best-checkpoint selection; not an independent test."
    )

    overall_representation = {
        "samples": 168,
        "effective_rank": {
            "candidate": 9.0,
            "control": 10.0,
            "ratio": 0.9,
            "passed": True,
        },
        "participation_ratio": {
            "candidate": 5.0,
            "control": 5.0,
            "ratio": 1.0,
            "passed": True,
        },
        "passed": True,
    }
    task_representation = {
        "samples": 42,
        "effective_rank": {
            "candidate": 8.0,
            "control": 10.0,
            "ratio": 0.8,
            "passed": True,
        },
        "participation_ratio": {
            "candidate": 4.0,
            "control": 5.0,
            "ratio": 0.8,
            "passed": True,
        },
        "passed": True,
    }
    conditions = {
        "primary_superiority": True,
        "clean_noninferiority": True,
        "task_direction": True,
        "representation_noncollapse": True,
        "corrector_train_unseen_directional_agreement": True,
    }
    return {
        "schema_version": 1,
        "protocol": "fcpg_lewm_anchor_visual_dev_selection_v1",
        "development_only": True,
        "primary": primary,
        "clean_noninferiority": clean,
        "task_direction": {
            "positive_tasks": 4,
            "task_count": 4,
            "worst_relative_improvement": 0.04,
            "passed": True,
        },
        "representation_noncollapse": {
            "overall": overall_representation,
            "per_task": {
                task: deepcopy(task_representation) for task in TASKS
            },
            "passed": True,
        },
        "corrector_train_unseen_sensitivity": unseen,
        "secondary_holm_family": [
            {
                "name": "paired_visual_vs_triplet_corrected",
                "relative_improvement": 0.03,
                "p_value": 0.01,
                "alternative": "greater",
                "holm_threshold": 0.05 / 3.0,
                "holm_adjusted_p": 0.03,
                "rejected": True,
            },
            {
                "name": "paired_innovation_vs_paired_visual",
                "relative_improvement": 0.02,
                "p_value": 0.04,
                "alternative": "greater",
                "holm_threshold": 0.05,
                "holm_adjusted_p": 0.04,
                "rejected": True,
            },
            {
                "name": "triplet_corrected_vs_triplet_nominal",
                "relative_improvement": 0.01,
                "p_value": 0.02,
                "alternative": "greater",
                "holm_threshold": 0.025,
                "holm_adjusted_p": 0.04,
                "rejected": True,
            },
        ],
        "conditions": conditions,
        "passed": True,
        "decision": "fresh_multiseed_outer_holdout_confirmation_candidate",
    }


def _paired_statistics_sections() -> dict[str, object]:
    baseline_by_task = {task: 0.1 for task in TASKS}
    episodes_per_task = {task: 42 for task in TASKS}
    episode_count = sum(episodes_per_task.values())
    task_count = len(TASKS)
    full_cells: list[dict[str, Any]] = []
    train_by_eval: dict[str, dict[str, dict[str, Any]]] = {}
    compact_cells: list[dict[str, Any]] = []

    for train_index, train_mode in enumerate(MODES):
        train_by_eval[train_mode] = {}
        for eval_index, eval_mode in enumerate(MODES):
            shift = 0.001 * (train_index + eval_index)
            per_task = {}
            for task, baseline_mse in baseline_by_task.items():
                cell_mse = baseline_mse + shift
                absolute_delta = math.fsum((baseline_mse, -cell_mse))
                per_task[task] = {
                    "baseline_mse": baseline_mse,
                    "cell_mse": cell_mse,
                    "absolute_delta": absolute_delta,
                    "relative_improvement": absolute_delta / baseline_mse,
                    "episode_count": episodes_per_task[task],
                }
            task_macro_mse = (
                math.fsum(row["cell_mse"] for row in per_task.values())
                / task_count
            )
            absolute_delta = (
                math.fsum(
                    row["absolute_delta"] for row in per_task.values()
                )
                / task_count
            )
            relative_improvement = (
                math.fsum(
                    row["relative_improvement"] for row in per_task.values()
                )
                / task_count
            )
            full_cell = {
                "train_mode": train_mode,
                "eval_mode": eval_mode,
                "task_macro_mse": task_macro_mse,
                "macro_relative_improvement": relative_improvement,
                "absolute_delta": absolute_delta,
                "relative_improvement_95_ci": [-0.01, 0.02],
                "absolute_delta_95_ci": [-0.002, 0.003],
                "probability_improvement_gt_zero": 0.75,
                "episode_count": episode_count,
                "task_count": task_count,
                "per_task": per_task,
            }
            full_cells.append(full_cell)
            train_by_eval[train_mode][eval_mode] = deepcopy(full_cell)
            compact_cells.append(
                {
                    "train_mode": train_mode,
                    "eval_mode": eval_mode,
                    "macro_relative_improvement": relative_improvement,
                    "macro_absolute_delta": absolute_delta,
                    "bootstrap_95_ci": [-0.01, 0.02],
                    "probability_improvement": 0.75,
                    "num_tasks": task_count,
                    "num_episodes": episode_count,
                }
            )

    nn = train_by_eval[MODES[0]][MODES[0]]["task_macro_mse"]
    nc = train_by_eval[MODES[0]][MODES[1]]["task_macro_mse"]
    cn = train_by_eval[MODES[1]][MODES[0]]["task_macro_mse"]
    cc = train_by_eval[MODES[1]][MODES[1]]["task_macro_mse"]
    raw_effects = {
        "eval_input_at_nominal_train": math.fsum((nn, -nc)),
        "eval_input_at_corrected_train": math.fsum((cn, -cc)),
        "training_at_nominal_eval": math.fsum((nn, -cn)),
        "training_at_corrected_eval": math.fsum((nc, -cc)),
        "matched_nominal_to_corrected": math.fsum((nn, -cc)),
    }
    effects = {
        name: {
            "raw_mse_delta": value,
            "relative_improvement": value / nn,
        }
        for name, value in raw_effects.items()
    }
    interaction = math.fsum((cc, -cn, -nc, nn))
    effects["interaction"] = {"raw_mse_delta": interaction}

    paired_statistics = {
        "schema_version": 1,
        "metric": "teacher_forced.predicted_mse",
        "lower_is_better": True,
        "modes": list(MODES),
        "baseline": {
            "train_mode": MODES[0],
            "eval_mode": MODES[0],
            "task_macro_mse": nn,
            "per_task_mse": baseline_by_task,
        },
        "aggregation": {
            "episode_pair_key": [
                "source_index",
                "task_name",
                "episode_id",
            ],
            "episode_to_task": "equal_episode_mean",
            "task_to_macro": "equal_task_mean",
            "relative_scale": "paired_resampled_baseline_task_mean",
            "effect_orientation": "positive_means_lower_mse_than_baseline",
        },
        "bootstrap": {
            "method": "paired_episode_within_task_equal_task_macro",
            "samples": 10_000,
            "seed": 5_120,
            "confidence": 0.95,
        },
        "episode_count": episode_count,
        "task_count": task_count,
        "episodes_per_task": episodes_per_task,
        "train_by_eval": train_by_eval,
        "cells": full_cells,
        "nominal_corrected_2x2": {
            "orientation": {
                "input_effect": (
                    "positive_means_corrected_eval_has_lower_mse"
                ),
                "training_effect": (
                    "positive_means_corrected_train_has_lower_mse"
                ),
                "matched_gain": (
                    "positive_means_corrected_corrected_has_lower_mse"
                ),
                "raw_interaction": "CC_minus_CN_minus_NC_plus_NN",
            },
            "cells": {
                "nominal_train_nominal_eval": nn,
                "nominal_train_corrected_eval": nc,
                "corrected_train_nominal_eval": cn,
                "corrected_train_corrected_eval": cc,
            },
            "fixed_train_input_effect": {
                MODES[0]: raw_effects["eval_input_at_nominal_train"],
                MODES[1]: raw_effects["eval_input_at_corrected_train"],
            },
            "fixed_eval_training_effect": {
                MODES[0]: raw_effects["training_at_nominal_eval"],
                MODES[1]: raw_effects["training_at_corrected_eval"],
            },
            "matched_gain": raw_effects["matched_nominal_to_corrected"],
            "raw_interaction": interaction,
        },
    }
    return {
        "paired_episode_statistics": paired_statistics,
        "paired_episode_bootstrap": {
            "baseline": {
                "train_mode": MODES[0],
                "eval_mode": MODES[0],
            },
            "cells": compact_cells,
        },
        "factor_decomposition": {
            "cells": {"NN": nn, "NC": nc, "CN": cn, "CC": cc},
            "effects": effects,
        },
    }



def _summary(*, optional: bool = True) -> dict[str, object]:
    mse = {
        train: {
            eval_mode: 0.1 + 0.01 * train_index + 0.001 * eval_index
            for eval_index, eval_mode in enumerate(MODES)
        }
        for train_index, train in enumerate(MODES)
    }
    baseline = mse[MODES[0]][MODES[0]]
    relative = {
        train: {
            eval_mode: (baseline - mse[train][eval_mode]) / baseline
            for eval_mode in MODES
        }
        for train in MODES
    }
    summary: dict[str, object] = {
        "development_only": True,
        "modes": list(MODES),
        "train_by_eval": mse,
        "relative_improvement_vs_triplet_nominal": relative,
        "provenance": {
            "optimizer_step": 2000,
            "parameter_count": 19_152_816,
            "seed": 5120,
            "target_sha256": "a" * 64,
            "data_split_sha256": "b" * 64,
            "complete_4x4": True,
        },
        "development_gate": _development_gate(),
    }
    if optional:
        summary.update(_paired_statistics_sections())
    return summary


def test_render_is_deterministic_ascii_fixed_precision_and_complete() -> None:
    summary = _summary()
    first = render_anchor_visual_markdown(summary)
    shuffled = deepcopy(summary)
    shuffled["train_by_eval"] = dict(  # type: ignore[index]
        reversed(list(shuffled["train_by_eval"].items()))  # type: ignore[union-attr]
    )
    paired = shuffled["paired_episode_bootstrap"]  # type: ignore[assignment]
    paired["cells"] = list(reversed(paired["cells"]))  # type: ignore[index]

    assert render_anchor_visual_markdown(shuffled) == first
    assert first.endswith("\n")
    first.encode("ascii")
    assert "DEVELOPMENT VALIDATION ONLY" in first
    assert "not a sealed test result" in first
    assert "recompute the task-specific baseline denominator" in first
    assert "Element-Pooled Prediction MSE Matrix" in first
    assert "Element-Pooled Relative MSE Reduction" in first
    assert "Task-Macro Nominal/Corrected 2x2 Factor Decomposition" in first
    assert "NN-scaled task-macro MSE reductions use NN" in first
    assert (
        "| triplet_nominal | 0.100000 | 0.101000 | 0.102000 | 0.103000 |"
        in first
    )
    assert (
        "| paired_innovation | paired_innovation | -0.060000 | -0.006000 | "
        "-0.010000 | 0.020000 | 0.750000 | 4 | 168 |"
        in first
    )
    assert "| N | 0.100000 | 0.101000 |" in first
    assert "| interaction | 0.000000 | N/A |" in first
    assert "a positive value means CC is worse (higher MSE)" in first
    assert "only a negative value indicates extra synergistic MSE reduction" in first
    assert "| optimizer_step | 2000 |" in first


def test_development_gate_renders_claim_and_estimand_boundaries() -> None:
    summary = _summary()
    summary["development_gate"] = _development_gate()

    rendered = render_anchor_visual_markdown(summary)

    assert "post-hoc single-seed development pilot" in rendered
    assert (
        "Under deterministic replay of the current code against the frozen "
        "data, the 168 main-set episodes partition 127+41"
        in rendered
    )
    assert "conservatively treated as corrector-exposed" in rendered
    assert (
        "must not be presented as a historical fact about exact legacy-run "
        "episode exposure"
        in rendered
    )
    assert "main 168-episode set is therefore not an independent test" in rendered
    assert "deterministic legacy split replay" in rendered
    assert "not historical checkpoint provenance" in rendered
    assert "conditional on seed 5120 and this fixed four-task set" in rendered
    assert "does not support population inference over seeds or tasks" in rendered
    assert "Generic Go/No-Go fields in the raw 4x4 cell reports" in rendered
    assert "are not part of, this development gate" in rendered
    assert "mean of task-specific PI/TC error ratios" in rendered
    assert "not 1 minus the ratio of the PI and TC task macros" in rendered
    assert "episode/task-equal relative effect" in rendered
    assert "Clean evidence is task-macro noninferiority only" in rendered
    assert "PI and PV are pre-LayerNorm raw-source-information-equivalent" in rendered
    assert "no equivalence is claimed after LayerNorm" in rendered
    assert "cannot prove an independent causal effect" in rendered
    assert "one-sided with alternative=greater" in rendered
    assert "covers exactly the three displayed prespecified mild-condition" in rendered
    assert "one-sided raw p (greater)" in rendered
    assert "fresh_multiseed_outer_holdout_confirmation_candidate" in rendered


def test_gate_rejects_ratio_of_macros_as_primary_estimand_regression() -> None:
    summary = _summary()
    gate = _development_gate()
    primary = gate["primary"]
    primary["relative_improvement"] = 1.0 - (
        primary["candidate_macro"] / primary["control_macro"]
    )
    summary["development_gate"] = gate

    with pytest.raises(
        ValueError,
        match=r"primary\.relative_improvement disagrees with recomputed",
    ):
        render_anchor_visual_markdown(summary)


@pytest.mark.parametrize(
    ("section", "path", "bad_value", "message"),
    [
        (
            "primary",
            ("per_task", TASKS[0], "relative_improvement"),
            0.20,
            "relative_improvement",
        ),
        ("primary", ("candidate_macro",), 2.40, "candidate_macro"),
        ("clean_noninferiority", ("control_macro",), 2.60, "control_macro"),
        (
            "clean_noninferiority",
            ("candidate_to_control_ratio",),
            1.01,
            "candidate_to_control_ratio",
        ),
        (
            "primary",
            ("relative_improvement_95_ci",),
            [0.04, 1.01],
            "endpoints cannot exceed 1",
        ),
        ("primary", ("passed",), False, "primary.passed"),
        ("clean_noninferiority", ("passed",), False, "clean passed"),
    ],
)
def test_gate_recomputes_pair_effect_fields_and_passes(
    section: str,
    path: tuple[str, ...],
    bad_value: object,
    message: str,
) -> None:
    summary = _summary()
    gate = _development_gate()
    target = gate[section]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = bad_value
    summary["development_gate"] = gate

    with pytest.raises(ValueError, match=message):
        render_anchor_visual_markdown(summary)


def test_gate_rejects_invented_representation_ratio_regression() -> None:
    summary = _summary()
    gate = _development_gate()
    gate["representation_noncollapse"]["overall"]["effective_rank"][
        "ratio"
    ] = 0.95
    summary["development_gate"] = gate

    with pytest.raises(
        ValueError,
        match=r"overall\.effective_rank\.ratio disagrees with recomputed",
    ):
        render_anchor_visual_markdown(summary)


def test_gate_recomputes_representation_frozen_threshold_pass() -> None:
    summary = _summary()
    gate = _development_gate()
    check = gate["representation_noncollapse"]["per_task"][TASKS[0]][
        "effective_rank"
    ]
    check["candidate"] = 7.9
    check["ratio"] = 0.79
    summary["development_gate"] = gate

    with pytest.raises(ValueError, match=r"effective_rank\.passed disagrees"):
        render_anchor_visual_markdown(summary)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("holm_threshold", 0.025, "holm_threshold"),
        ("holm_adjusted_p", 0.02, "holm_adjusted_p"),
        ("rejected", False, "rejected"),
    ],
)
def test_gate_rejects_unrecomputed_holm_results_regression(
    field: str, bad_value: object, message: str
) -> None:
    summary = _summary()
    gate = _development_gate()
    gate["secondary_holm_family"][0][field] = bad_value
    summary["development_gate"] = gate

    with pytest.raises(ValueError, match=message):
        render_anchor_visual_markdown(summary)


def test_gate_rejects_non_greater_secondary_alternative() -> None:
    summary = _summary()
    gate = _development_gate()
    gate["secondary_holm_family"][0]["alternative"] = "two-sided"
    summary["development_gate"] = gate

    with pytest.raises(ValueError, match="alternative must be 'greater'"):
        render_anchor_visual_markdown(summary)


def test_formal_summary_rejects_missing_paired_and_factor_sections() -> None:
    with pytest.raises(ValueError, match="formal anchor-visual summary is incomplete"):
        render_anchor_visual_markdown(_summary(optional=False))


def test_rejects_noncanonical_or_incomplete_modes() -> None:
    summary = _summary()
    summary["modes"] = list(reversed(MODES))
    with pytest.raises(ValueError, match="complete canonical"):
        render_anchor_visual_markdown(summary)

    summary = _summary()
    del summary["train_by_eval"][MODES[0]][MODES[-1]]  # type: ignore[index]
    with pytest.raises(ValueError, match="incomplete or unexpected keys"):
        render_anchor_visual_markdown(summary)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, "0.1"])
def test_rejects_nonfinite_or_nonnumeric_matrix_values(bad: object) -> None:
    summary = _summary()
    summary["train_by_eval"][MODES[0]][MODES[0]] = bad  # type: ignore[index]
    with pytest.raises((TypeError, ValueError), match="numeric|finite"):
        render_anchor_visual_markdown(summary)


def test_matrix_relative_values_are_recomputed_from_element_pooled_mse() -> None:
    summary = _summary()
    relative = summary["relative_improvement_vs_triplet_nominal"]
    relative[MODES[-1]][MODES[-1]] += 0.01  # type: ignore[index]

    with pytest.raises(ValueError, match="disagrees with recomputed value"):
        render_anchor_visual_markdown(summary)


def test_paired_bootstrap_requires_complete_unique_cells() -> None:
    summary = _summary()
    bootstrap = summary["paired_episode_bootstrap"]  # type: ignore[assignment]
    bootstrap["cells"] = bootstrap["cells"][:-1]  # type: ignore[index]
    with pytest.raises(ValueError, match="complete 4x4"):
        render_anchor_visual_markdown(summary)

    summary = _summary()
    bootstrap = summary["paired_episode_bootstrap"]  # type: ignore[assignment]
    bootstrap["cells"][-1] = deepcopy(bootstrap["cells"][0])  # type: ignore[index]
    with pytest.raises(ValueError, match="duplicate"):
        render_anchor_visual_markdown(summary)


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("bootstrap_95_ci", [0.2, 0.1], "lower endpoint"),
        ("probability_improvement", 1.01, "must be in"),
        ("macro_absolute_delta", math.nan, "finite"),
        ("num_tasks", 0, "positive"),
    ],
)
def test_paired_bootstrap_rejects_invalid_statistics(
    field: str, bad: object, message: str
) -> None:
    summary = _summary()
    bootstrap = summary["paired_episode_bootstrap"]  # type: ignore[assignment]
    bootstrap["cells"][0][field] = bad  # type: ignore[index]
    with pytest.raises((TypeError, ValueError), match=message):
        render_anchor_visual_markdown(summary)


def test_factor_decomposition_requires_exact_cells_and_effect_fields() -> None:
    summary = _summary()
    factor = summary["factor_decomposition"]  # type: ignore[assignment]
    del factor["cells"]["CC"]  # type: ignore[index]
    with pytest.raises(ValueError, match="incomplete or unexpected keys"):
        render_anchor_visual_markdown(summary)

    summary = _summary()
    factor = summary["factor_decomposition"]  # type: ignore[assignment]
    del factor["effects"]["training_at_nominal_eval"]["relative_improvement"]  # type: ignore[index]
    with pytest.raises(ValueError, match="incomplete or unexpected keys"):
        render_anchor_visual_markdown(summary)


@pytest.mark.parametrize(
    ("effect_name", "field"),
    [
        ("eval_input_at_corrected_train", "raw_mse_delta"),
        ("training_at_nominal_eval", "relative_improvement"),
        ("interaction", "raw_mse_delta"),
    ],
)
def test_factor_effects_are_recomputed_from_task_macro_cells(
    effect_name: str,
    field: str,
) -> None:
    summary = _summary()
    factor = summary["factor_decomposition"]  # type: ignore[assignment]
    factor["effects"][effect_name][field] += 0.01  # type: ignore[index]

    with pytest.raises(ValueError, match="disagrees with recomputed value"):
        render_anchor_visual_markdown(summary)


def test_provenance_rejects_nonfinite_or_structured_values() -> None:
    summary = _summary()
    summary["provenance"]["temperature"] = math.inf  # type: ignore[index]
    with pytest.raises(ValueError, match="finite"):
        render_anchor_visual_markdown(summary)

    summary = _summary()
    summary["provenance"]["nested"] = {"bad": True}  # type: ignore[index]
    with pytest.raises(TypeError, match="scalar"):
        render_anchor_visual_markdown(summary)


def test_gate_embedded_in_summary_is_not_silently_ignored() -> None:
    summary = _summary()
    summary["development_gate"] = {"development_only": False}

    with pytest.raises(ValueError, match="development_gate has incomplete"):
        render_anchor_visual_markdown(summary)


def test_external_gate_cannot_override_embedded_gate() -> None:
    summary = _summary()
    summary["development_gate"] = _development_gate()

    with pytest.raises(ValueError, match="must not override"):
        render_anchor_visual_markdown(summary, _development_gate())


@pytest.mark.parametrize(
    "missing_field",
    ["paired_episode_bootstrap", "factor_decomposition"],
)
def test_formal_summary_requires_complete_statistics(
    missing_field: str,
) -> None:
    summary = _summary()
    summary["development_gate"] = _development_gate()
    del summary[missing_field]

    with pytest.raises(ValueError, match=f"missing=.*{missing_field}"):
        render_anchor_visual_markdown(summary)


def test_formal_summary_requires_explicit_development_only_true() -> None:
    summary = _summary()
    summary["development_gate"] = _development_gate()
    del summary["development_only"]

    with pytest.raises(ValueError, match="development_only"):
        render_anchor_visual_markdown(summary)


def test_formal_matrix_protocol_requires_a_gate() -> None:
    summary = _summary()
    summary["protocol"] = "fcpg_lewm_anchor_visual_matrix_v1"
    del summary["development_gate"]

    with pytest.raises(ValueError, match="development_gate"):
        render_anchor_visual_markdown(summary)


def test_formal_summary_requires_a_gate_without_protocol_marker() -> None:
    summary = _summary()
    del summary["development_gate"]

    with pytest.raises(ValueError, match="development_gate"):
        render_anchor_visual_markdown(summary)


def test_external_gate_is_allowed_only_when_no_gate_is_embedded() -> None:
    summary = _summary()
    del summary["development_gate"]
    rendered = render_anchor_visual_markdown(summary, _development_gate())

    assert "## Development Selection Gate" in rendered


def test_paired_population_must_match_gate_clean_plus_mild_counts() -> None:
    summary = _summary()
    gate = summary["development_gate"]
    clean = gate["clean_noninferiority"]["per_task"]
    clean[TASKS[0]]["episode_count"] += 1
    clean[TASKS[1]]["episode_count"] -= 1

    with pytest.raises(ValueError, match="episode population disagrees"):
        render_anchor_visual_markdown(summary)


def test_paired_train_by_eval_must_match_canonical_cells() -> None:
    summary = _summary()
    statistics = summary["paired_episode_statistics"]
    statistics["train_by_eval"][MODES[0]][MODES[0]]["task_macro_mse"] = True

    with pytest.raises(ValueError, match="train_by_eval.*disagrees"):
        render_anchor_visual_markdown(summary)


def test_paired_2x2_must_match_canonical_cells_and_orientation() -> None:
    summary = _summary()
    statistics = summary["paired_episode_statistics"]
    statistics["nominal_corrected_2x2"]["raw_interaction"] = True

    with pytest.raises(ValueError, match="nominal_corrected_2x2 disagrees"):
        render_anchor_visual_markdown(summary)

    summary = _summary()
    statistics = summary["paired_episode_statistics"]
    statistics["nominal_corrected_2x2"]["orientation"]["raw_interaction"] = (
        "NN_minus_CN_minus_NC_plus_CC"
    )
    with pytest.raises(ValueError, match="nominal_corrected_2x2 disagrees"):
        render_anchor_visual_markdown(summary)


def test_explicit_non_development_summary_is_rejected() -> None:
    summary = _summary()
    summary["development_only"] = False
    with pytest.raises(ValueError, match="development_only"):
        render_anchor_visual_markdown(summary)
