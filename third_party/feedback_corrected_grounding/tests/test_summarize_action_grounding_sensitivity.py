from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.summarize_action_grounding_sensitivity import (
    InputSpec,
    _bootstrap_task,
    load_result,
    parse_input_spec,
    render_markdown,
    summarize_results,
    validate_collection,
)


def _metric_row(
    task: str,
    seed: int,
    episode: int,
    frame: int,
    value: float,
    *,
    clean: bool,
) -> dict[str, object]:
    nominal = value + 10.0
    fcpg = 10.0
    return {
        "task": task,
        "episode": episode,
        "frame": frame,
        "anchor_index": episode * 100 + frame,
        "policy_seed": seed,
        "base_valid_anchors": 3,
        "common_valid_anchors": 3,
        "fcpg_oob_anchors": 0,
        "common_support_coverage": 1.0,
        "nominal_epe": nominal,
        "fcpg_epe": fcpg,
        "delta_epe": value,
        "nominal_oracle_xyz_h1": nominal,
        "fcpg_oracle_xyz_h1": fcpg,
        "delta_action_xyz_h1": value,
        "nominal_oracle_xyz_h8": nominal,
        "fcpg_oracle_xyz_h8": fcpg,
        "delta_action_xyz_h8": value,
        "nominal_oracle_gripper_h1": nominal,
        "fcpg_oracle_gripper_h1": fcpg,
        "delta_gripper_h1": value,
        "nominal_oracle_gripper_h8": nominal,
        "fcpg_oracle_gripper_h8": fcpg,
        "delta_gripper_h8": value,
        "actual_camera_rotation_deg": 0.0 if clean else 5.0,
        "actual_camera_translation_norm": 0.0 if clean else 0.04,
    }


def _write_payload(
    path: Path,
    task: str,
    seed: int,
    episode_values: list[tuple[bool, list[float]]],
    *,
    max_batches: int = 0,
) -> None:
    rows = []
    for episode, (clean, values) in enumerate(episode_values):
        for frame, value in enumerate(values):
            rows.append(
                _metric_row(task, seed, episode, frame, value, clean=clean)
            )
    path.write_text(
        json.dumps(
            {
                "contract": {
                    "training_performed": False,
                    "same_checkpoint_same_batch": True,
                    "grounding_interventions": ["nominal", "fcpg", "oracle"],
                    "oracle_and_q_true_usage": (
                        "oracle_reference_only_absent_from_deployable_modes"
                    ),
                    "statistical_unit": "complete_episode_within_task",
                    "overlapping_windows_treated_as_independent": False,
                    "exploratory": False,
                    "smoke_only": False,
                    "inferential": True,
                    "noninferential": False,
                    "confidence_intervals_reported": True,
                    "artifact_contract_mismatches": [],
                },
                "inputs": {
                    "data": f"data/{task}.npz",
                    "checkpoint": f"checkpoints/{task}_seed{seed}.pt",
                    "vision_cache": f"data/vision_cache/{task}",
                    "grounding_cache": f"data/grounding_cache/{task}_seed{seed}",
                    "split_seed": 0,
                    "val_fraction": 0.2,
                    "policy_seed": seed,
                    "h_exec": 8,
                    "max_batches": max_batches,
                    "allow_cache_override": False,
                },
                "summary": {
                    "inferential": True,
                    "snapshots": len(rows),
                    "physical_episodes": len(episode_values),
                    "tasks": [task],
                    "support": {
                        "base_valid_anchors": 3 * len(rows),
                        "common_valid_anchors": 3 * len(rows),
                        "fcpg_oob_anchors": 0,
                        "overall_common_support_coverage": 1.0,
                    },
                },
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )


def _balanced_results(tmp_path: Path):
    specs = []
    task_values = {
        "pick-v3": {
            0: [(True, [1.0, 3.0]), (False, [5.0, 5.0])],
            1: [(True, [7.0, 9.0]), (False, [11.0, 11.0])],
        },
        "button-v3": {
            0: [
                (True, [19.0, 21.0]),
                (False, [22.0, 22.0]),
                (False, [24.0, 24.0]),
            ],
            1: [
                (True, [29.0, 31.0]),
                (False, [32.0, 32.0]),
                (False, [34.0, 34.0]),
            ],
        },
    }
    for task, by_seed in task_values.items():
        for seed, episodes in by_seed.items():
            path = tmp_path / f"{task}_{seed}.json"
            _write_payload(path, task, seed, episodes)
            specs.append(InputSpec(path=path))
    return [load_result(spec) for spec in specs]


def test_episode_then_seed_then_fixed_task_macro_and_strata(tmp_path: Path) -> None:
    summary = summarize_results(
        _balanced_results(tmp_path),
        bootstrap_samples=1_000,
        bootstrap_seed=7,
    )

    delta = summary["strata"]["all"]["metrics"]["delta_action_xyz_h8"]
    # pick: seed means 3.5 and 9.5 -> 6.5. button: 22 and 32 -> 27.
    # The primary estimate fixes both tasks at 1/2 weight: (6.5 + 27) / 2.
    assert delta["estimate"] == pytest.approx(16.75)
    assert delta["per_task"]["pick-v3"]["estimate"] == pytest.approx(6.5)
    assert delta["per_task"]["button-v3"]["estimate"] == pytest.approx(27.0)
    assert delta["per_task"]["pick-v3"]["per_seed"]["0"]["estimate"] == pytest.approx(
        3.5
    )
    assert len(delta["hierarchical_bootstrap_95_ci"]) == 2
    absolute = summary["strata"]["all"]["absolute_components"]
    assert absolute["nominal_oracle_xyz_h8"]["estimate"] == pytest.approx(26.75)
    assert absolute["fcpg_oracle_xyz_h8"]["estimate"] == pytest.approx(10.0)
    assert len(
        absolute["nominal_oracle_xyz_h8"]["hierarchical_bootstrap_95_ci"]
    ) == 2

    clean = summary["strata"]["clean"]["metrics"]["delta_epe"]
    assert clean["estimate"] == pytest.approx((5.0 + 25.0) / 2.0)
    drift = summary["strata"]["drift"]["metrics"]["delta_epe"]
    assert drift["estimate"] == pytest.approx((8.0 + 28.0) / 2.0)
    association = summary["strata"]["all"]["associations"][
        "spearman_delta_epe_vs_delta_action_xyz_h8"
    ]
    assert association["available"] is True
    # Constant within-episode sequences contribute zero Spearman: pick=0.5,
    # button=1/3, followed by the same fixed equal-task macro.
    assert association["estimate"] == pytest.approx(5.0 / 12.0)


def test_bootstrap_is_reproducible(tmp_path: Path) -> None:
    results = _balanced_results(tmp_path)
    first = summarize_results(results, bootstrap_samples=200, bootstrap_seed=11)
    second = summarize_results(results, bootstrap_samples=200, bootstrap_seed=11)
    assert first == second


def test_crossed_bootstrap_reuses_episode_draw_across_selected_seeds() -> None:
    import numpy as np

    shared = np.concatenate([np.zeros(10), np.ones(10)])
    arrays = {0: shared, 1: shared, 2: shared}
    distribution = _bootstrap_task(
        arrays, 20_000, np.random.default_rng(11)
    )
    low, high = np.quantile(distribution, [0.025, 0.975])
    assert low <= 0.3
    assert high == pytest.approx(0.7)


def test_absolute_component_markdown_matches_json_values(tmp_path: Path) -> None:
    summary = summarize_results(
        _balanced_results(tmp_path),
        bootstrap_samples=200,
        bootstrap_seed=17,
    )
    value = summary["strata"]["all"]["absolute_components"][
        "nominal_oracle_xyz_h8"
    ]
    low, high = value["hierarchical_bootstrap_95_ci"]
    markdown = render_markdown(summary)
    expected = (
        f"| nominal_oracle_xyz_h8 | {value['estimate']:.6f} | "
        f"[{low:.6f}, {high:.6f}] |"
    )
    assert expected in markdown


def test_auto_and_explicit_input_specs_cross_check_payload(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    _write_payload(path, "pick-v3", 2, [(True, [1.0, 2.0])])

    automatic = load_result(parse_input_spec(str(path)))
    explicit = load_result(parse_input_spec(f"pick-v3=2={path}"))
    assert automatic.task == explicit.task == "pick-v3"
    assert automatic.policy_seed == explicit.policy_seed == 2

    with pytest.raises(ValueError, match="explicit policy seed"):
        load_result(parse_input_spec(f"pick-v3=3={path}"))
    with pytest.raises(ValueError, match="explicit task"):
        load_result(parse_input_spec(f"button-v3=2={path}"))


def test_cross_seed_physical_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    episodes = [(True, [1.0, 2.0]), (False, [3.0, 4.0])]
    _write_payload(left, "pick-v3", 0, episodes)
    _write_payload(right, "pick-v3", 1, episodes)
    payload = json.loads(right.read_text(encoding="utf-8"))
    payload["rows"][0]["anchor_index"] = 999
    right.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="physical episode/frame identities differ"):
        validate_collection(
            [load_result(InputSpec(left)), load_result(InputSpec(right))]
        )


@pytest.mark.parametrize("field,replacement", [
    ("data", "data/other.npz"),
    ("split_seed", 99),
    ("val_fraction", 0.3),
    ("vision_cache", "data/vision_cache/other"),
])
def test_cross_seed_input_contract_mismatch_is_rejected(
    tmp_path: Path, field: str, replacement: object
) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    episodes = [(True, [1.0, 2.0]), (False, [3.0, 4.0])]
    _write_payload(left, "pick-v3", 0, episodes)
    _write_payload(right, "pick-v3", 1, episodes)
    payload = json.loads(right.read_text(encoding="utf-8"))
    payload["inputs"][field] = replacement
    right.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="input contract differs"):
        validate_collection(
            [load_result(InputSpec(left)), load_result(InputSpec(right))]
        )


def test_grounding_cache_may_differ_across_policy_seeds(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    episodes = [(True, [1.0, 2.0]), (False, [3.0, 4.0])]
    _write_payload(left, "pick-v3", 0, episodes)
    _write_payload(right, "pick-v3", 1, episodes)
    validate_collection([load_result(InputSpec(left)), load_result(InputSpec(right))])


def test_duplicate_task_seed_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    _write_payload(path, "pick-v3", 0, [(True, [1.0, 2.0])])
    result = load_result(InputSpec(path))
    with pytest.raises(ValueError, match="duplicate task/policy-seed"):
        validate_collection([result, result])


def test_truncated_or_incomplete_payload_is_rejected(tmp_path: Path) -> None:
    truncated = tmp_path / "truncated.json"
    _write_payload(
        truncated,
        "pick-v3",
        0,
        [(True, [1.0, 2.0])],
        max_batches=1,
    )
    with pytest.raises(ValueError, match="only full evaluations"):
        load_result(InputSpec(truncated))

    missing = tmp_path / "missing.json"
    _write_payload(missing, "pick-v3", 0, [(True, [1.0, 2.0])])
    payload = json.loads(missing.read_text(encoding="utf-8"))
    del payload["rows"][0]["delta_action_xyz_h8"]
    missing.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="missing metric fields"):
        load_result(InputSpec(missing))


def test_null_h8_rows_are_omitted_only_from_h8_statistics(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    _write_payload(
        path,
        "pick-v3",
        0,
        [(True, [1.0, 2.0, 3.0]), (False, [5.0, 6.0, 7.0])],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    last = payload["rows"][2]
    for field in (
        "nominal_oracle_xyz_h8",
        "fcpg_oracle_xyz_h8",
        "delta_action_xyz_h8",
        "nominal_oracle_gripper_h8",
        "fcpg_oracle_gripper_h8",
        "delta_gripper_h8",
    ):
        last[field] = None
    path.write_text(json.dumps(payload), encoding="utf-8")

    summary = summarize_results(
        [load_result(InputSpec(path))],
        bootstrap_samples=100,
        bootstrap_seed=3,
    )
    metrics = summary["strata"]["clean"]["metrics"]
    assert metrics["delta_action_xyz_h1"]["estimate"] == pytest.approx(2.0)
    assert metrics["delta_action_xyz_h8"]["estimate"] == pytest.approx(1.5)
    assert summary["strata"]["clean"]["support"][
        "episode_clustered_common_support_coverage"
    ]["estimate"] == pytest.approx(1.0)


def test_cross_seed_h8_null_mask_mismatch_is_rejected(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    episodes = [(True, [1.0, 2.0]), (False, [3.0, 4.0])]
    _write_payload(left, "pick-v3", 0, episodes)
    _write_payload(right, "pick-v3", 1, episodes)
    payload = json.loads(right.read_text(encoding="utf-8"))
    row = payload["rows"][0]
    for field in (
        "nominal_oracle_xyz_h8",
        "fcpg_oracle_xyz_h8",
        "delta_action_xyz_h8",
        "nominal_oracle_gripper_h8",
        "fcpg_oracle_gripper_h8",
        "delta_gripper_h8",
    ):
        row[field] = None
    right.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="physical episode/frame identities differ"):
        validate_collection(
            [load_result(InputSpec(left)), load_result(InputSpec(right))]
        )


def test_partial_metric_episode_support_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    _write_payload(
        path,
        "pick-v3",
        0,
        [(True, [1.0, 2.0]), (False, [3.0, 4.0])],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload["rows"][:2]:
        for field in (
            "nominal_oracle_xyz_h8",
            "fcpg_oracle_xyz_h8",
            "delta_action_xyz_h8",
            "nominal_oracle_gripper_h8",
            "fcpg_oracle_gripper_h8",
            "delta_gripper_h8",
        ):
            row[field] = None
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete episode support"):
        summarize_results(
            [load_result(InputSpec(path))],
            bootstrap_samples=100,
            bootstrap_seed=3,
        )


def test_globally_unavailable_h8_is_explicit_not_a_smaller_estimand(
    tmp_path: Path,
) -> None:
    path = tmp_path / "input.json"
    _write_payload(
        path,
        "pick-v3",
        0,
        [(True, [1.0, 2.0]), (False, [3.0, 4.0])],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload["rows"]:
        for field in (
            "nominal_oracle_xyz_h8",
            "fcpg_oracle_xyz_h8",
            "delta_action_xyz_h8",
            "nominal_oracle_gripper_h8",
            "fcpg_oracle_gripper_h8",
            "delta_gripper_h8",
        ):
            row[field] = None
    path.write_text(json.dumps(payload), encoding="utf-8")
    summary = summarize_results(
        [load_result(InputSpec(path))],
        bootstrap_samples=100,
        bootstrap_seed=3,
    )
    assert summary["strata"]["all"]["metrics"]["delta_action_xyz_h8"] == {
        "available": False
    }
    assert summary["strata"]["all"]["absolute_components"][
        "nominal_oracle_xyz_h8"
    ] == {"available": False}


def test_tasks_must_share_policy_seed_set(tmp_path: Path) -> None:
    results = _balanced_results(tmp_path)
    with pytest.raises(ValueError, match="same policy-seed set"):
        validate_collection(
            [result for result in results if not (
                result.task == "button-v3" and result.policy_seed == 1
            )]
        )


def test_declared_camera_clean_must_match_exact_zero_definition(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    _write_payload(path, "pick-v3", 0, [(True, [1.0, 2.0])])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"][0]["camera_clean"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees with exact zero"):
        load_result(InputSpec(path))
