#!/usr/bin/env python3
"""Run the frozen A3 10k confirmation through the shared A1 trainer."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import json
import math
from numbers import Real
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a3_collection import (  # noqa: E402
    PROTOCOL,
    TASKS,
    validate_a3_contract,
)
from fcpg.lewm_action_branch_a3_confirmation import (  # noqa: E402
    validate_a3_confirmation_canonical_conformance,
)
from fcpg.lewm_action_branch_a1 import load_a1_npz  # noqa: E402
from fcpg.lewm_action_branch_a3_metrics import (  # noqa: E402
    CHECKPOINT_STEP as DEVELOPMENT_CHECKPOINT_STEP,
    TRAINING_SEEDS as DEVELOPMENT_SEEDS,
)
from fcpg.lewm_action_branch_training import (  # noqa: E402
    BASE_SIGREG_WEIGHT,
    BUDGET_STEPS,
    validate_frozen_base_checkpoint,
)
from scripts import train_lewm_action_branch_a1 as a1_trainer  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_physical_support_a3.json"
CONFIRMATION_SEEDS = (29_301, 29_302, 29_303)
FROZEN_SEEDS = CONFIRMATION_SEEDS
DEVELOPMENT_GATE_CHECKS = frozenset(
    {
        "late_effect_vs_no_action",
        "late_effect_vs_shuffled_action",
        "late_effect_vs_every_shuffled_shift",
        "jointly_positive_tasks",
        "jointly_positive_seeds",
        "cross_seed_task_replication",
        "raw_not_worse_than_no_action",
        "raw_better_than_hold_last",
        "raw_better_than_constant_velocity",
    }
)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _resolve(path: str | Path) -> Path:
    source = Path(path).expanduser()
    return source.resolve() if source.is_absolute() else (ROOT / source).resolve()


def _read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    source = _resolve(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return payload


def _same_typed_json(left: object, right: object) -> bool:
    """Compare decoded JSON without treating booleans/ints/floats as equal."""

    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _same_typed_json(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same_typed_json(left_value, right_value)
            for left_value, right_value in zip(left, right)
        )
    return left == right


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _nonnegative_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return int(value)


def _validate_confirmation_contract(
    contract: Mapping[str, Any],
) -> tuple[str, ...]:
    """Validate every config field that authorizes the A3 confirmation."""

    tasks = validate_a3_contract(contract)
    development_gate = _mapping(contract.get("development_gate"), "development_gate")
    expected_development_gate = {
        "evaluation_split": "development",
        "checkpoint_step": DEVELOPMENT_CHECKPOINT_STEP,
        "seeds": list(DEVELOPMENT_SEEDS),
        "primary_horizons": [6, 7, 8],
        "minimum_relative_improvement": 0.05,
        "minimum_positive_tasks": 3,
        "minimum_positive_seeds": 2,
        "minimum_cross_replicating_seeds": 2,
        "minimum_positive_tasks_per_replicating_seed": 3,
        "every_shuffled_shift_must_reach_minimum": True,
        "raw_factual_not_worse_than_no_action": True,
        "raw_factual_better_than_hold_last_and_constant_velocity": True,
    }
    if not _same_typed_json(development_gate, expected_development_gate):
        raise ValueError("A3 development gate contract drifted")

    confirmation = _mapping(
        contract.get("confirmation_after_development_go"),
        "confirmation_after_development_go",
    )
    split = _mapping(contract.get("split"), "split")
    observed = (
        confirmation.get("reserved_test_episodes_per_task"),
        confirmation.get("test_collection_requires_development_go"),
        confirmation.get("optimizer_steps"),
        confirmation.get("seeds"),
        confirmation.get("fresh_start_from_frozen_base"),
        confirmation.get("train_split"),
        confirmation.get("evaluation_split"),
        confirmation.get("bundle_batch_size"),
        confirmation.get("amp_dtype"),
        confirmation.get("anchor_encoder_sigreg"),
        confirmation.get("additional_sigreg_values_allowed"),
        confirmation.get("selection"),
        confirmation.get("primary_horizons"),
        confirmation.get("minimum_relative_improvement"),
        confirmation.get("minimum_positive_tasks"),
        confirmation.get("minimum_positive_seeds"),
        confirmation.get("minimum_cross_replicating_seeds"),
        confirmation.get("minimum_positive_tasks_per_replicating_seed"),
        confirmation.get("every_shuffled_shift_must_reach_minimum"),
        confirmation.get("raw_factual_not_worse_than_no_action"),
        confirmation.get(
            "raw_factual_better_than_hold_last_and_constant_velocity"
        ),
    )
    expected = (
        split.get("test_episodes_per_task"),
        True,
        BUDGET_STEPS["10k"],
        list(CONFIRMATION_SEEDS),
        True,
        "train",
        "test",
        16,
        "float16",
        BASE_SIGREG_WEIGHT,
        [],
        "fixed_latest",
        [6, 7, 8],
        0.05,
        3,
        2,
        2,
        3,
        True,
        True,
        True,
    )
    if not _same_typed_json(list(observed), list(expected)):
        raise ValueError("A3 10k confirmation contract drifted")
    return tasks


def _load_fixed_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    canonical = _read_json(DEFAULT_CONFIG, label="canonical A3 config")
    source = _resolve(path)
    payload = (
        canonical
        if source == DEFAULT_CONFIG.resolve()
        else _read_json(source, label="A3 config")
    )
    _validate_confirmation_contract(payload)
    if not _same_typed_json(payload, canonical):
        raise ValueError("A3 confirmation config is fixed to the checked-in contract")
    return payload


def _validate_member_subset(
    value: object,
    *,
    allowed: Sequence[str | int],
    minimum: int,
    label: str,
) -> tuple[str | int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of distinct values")
    try:
        distinct_count = len(set(value))
    except TypeError as error:
        raise ValueError(f"{label} must be a list of distinct values") from error
    allowed_types = {type(item) for item in allowed}
    if distinct_count != len(value) or any(
        type(item) not in allowed_types for item in value
    ):
        raise ValueError(f"{label} must be a list of distinct values")
    result = tuple(value)
    if len(result) < minimum or not set(result).issubset(set(allowed)):
        raise ValueError(f"{label} does not satisfy the frozen A3 minimum")
    return result


def _validate_development_go_summary(
    path: str | Path,
    *,
    contract: Mapping[str, Any],
    tasks: Sequence[str],
) -> dict[str, Any]:
    """Require a complete scorer-produced A3 2k Development-Go summary."""

    payload = _read_json(path, label="A3 Development-Go summary")
    development_gate = _mapping(contract.get("development_gate"), "development_gate")
    horizons = list(development_gate["primary_horizons"])
    minimum_effect = float(development_gate["minimum_relative_improvement"])
    minimum_tasks = int(development_gate["minimum_positive_tasks"])
    minimum_seeds = int(development_gate["minimum_positive_seeds"])
    minimum_cross_seeds = int(development_gate["minimum_cross_replicating_seeds"])
    minimum_tasks_per_cross_seed = int(
        development_gate["minimum_positive_tasks_per_replicating_seed"]
    )
    if (
        payload.get("protocol") != PROTOCOL
        or payload.get("stage") != "a3_2k_development"
        or payload.get("decision") != "Development-Go"
        or payload.get("passed") is not True
        or payload.get("input_contains_test_rows") is not False
        or type(payload.get("checkpoint_step")) is not int
        or payload.get("checkpoint_step") != DEVELOPMENT_CHECKPOINT_STEP
        or not _same_typed_json(payload.get("primary_horizons"), horizons)
    ):
        raise ValueError(
            "A3 confirmation requires the frozen unexposed-test 2k Development-Go"
        )

    split = _mapping(contract.get("split"), "split")
    coverage = _mapping(payload.get("coverage"), "Development-Go coverage")
    if coverage.get("tasks") != list(tasks) or coverage.get("training_seeds") != list(
        DEVELOPMENT_SEEDS
    ):
        raise ValueError("Development-Go coverage must use the frozen tasks and seeds")
    _positive_integer(
        coverage.get("development_hazard_bundle_count"),
        label="development hazard bundle count",
    )
    coverage_per_task = _mapping(
        coverage.get("per_task"), "Development-Go per-task coverage"
    )
    if set(coverage_per_task) != set(tasks):
        raise ValueError("Development-Go coverage must contain exactly the A3 tasks")
    per_task_bundle_count = 0
    for task in tasks:
        row = _mapping(coverage_per_task[task], f"Development-Go coverage for {task}")
        train_episode_count = _positive_integer(
            row.get("train_physical_episode_count"),
            label=f"Development-Go train episode coverage for {task}",
        )
        development_episode_count = _positive_integer(
            row.get("development_physical_episode_count"),
            label=f"Development-Go development episode coverage for {task}",
        )
        if train_episode_count != split["train_episodes_per_task"] or (
            development_episode_count != split["development_episodes_per_task"]
        ):
            raise ValueError(f"Development-Go episode coverage drifted for {task}")
        per_task_bundle_count += _positive_integer(
            row.get("development_hazard_bundle_count"),
            label=f"Development-Go hazard coverage for {task}",
        )
    if per_task_bundle_count != coverage["development_hazard_bundle_count"]:
        raise ValueError("Development-Go hazard coverage is internally inconsistent")

    whitening = _mapping(payload.get("whitening"), "Development-Go whitening")
    if (
        whitening.get("source") != "canonical_train_rows"
        or whitening.get("per_task") is not True
        or whitening.get("episode_equal") is not True
        or not _same_typed_json(whitening.get("fit_horizons"), horizons)
    ):
        raise ValueError("Development-Go whitening contract drifted")
    whitening_tasks = _mapping(
        whitening.get("tasks"), "Development-Go whitening tasks"
    )
    if set(whitening_tasks) != set(tasks):
        raise ValueError("Development-Go whitening must cover exactly the A3 tasks")
    for task in tasks:
        row = _mapping(whitening_tasks[task], f"Development-Go whitening for {task}")
        if _positive_integer(
            row.get("train_physical_episode_count"),
            label=f"Development-Go whitening episode coverage for {task}",
        ) != split["train_episodes_per_task"]:
            raise ValueError(f"Development-Go whitening coverage drifted for {task}")

    comparisons = _mapping(
        payload.get("comparisons"), "Development-Go comparisons"
    )
    for baseline in ("no_action", "shuffled_action"):
        row = _mapping(
            comparisons.get(baseline), f"Development-Go {baseline} comparison"
        )
        positive_task_count = _positive_integer(
            row.get("positive_task_count"),
            label=f"{baseline} positive task count",
        )
        task_count = _positive_integer(
            row.get("task_count"), label=f"{baseline} task count"
        )
        positive_seed_count = _positive_integer(
            row.get("positive_seed_count"),
            label=f"{baseline} positive seed count",
        )
        seed_count = _positive_integer(
            row.get("seed_count"), label=f"{baseline} seed count"
        )
        if (
            row.get("baseline_method") != baseline
            or row.get("factual_method") != "factual"
            or task_count != len(tasks)
            or seed_count != len(DEVELOPMENT_SEEDS)
            or positive_task_count < minimum_tasks
            or positive_seed_count < minimum_seeds
            or _finite_number(row.get("effect"), label=f"{baseline} effect")
            < minimum_effect
        ):
            raise ValueError(f"Development-Go comparison against {baseline} is invalid")

    shifts = _mapping(
        payload.get("shuffled_action_shifts"), "Development-Go shuffled shifts"
    )
    per_shift = shifts.get("per_shift")
    if (
        shifts.get("count") != 7
        or not isinstance(per_shift, list)
        or len(per_shift) != 7
        or _finite_number(
            shifts.get("minimum_effect"), label="minimum shuffled-shift effect"
        )
        < minimum_effect
    ):
        raise ValueError("Development-Go must pass all seven shuffled-action shifts")
    for expected_shift, raw_row in enumerate(per_shift, start=1):
        row = _mapping(raw_row, f"Development-Go shuffled shift {expected_shift}")
        donor_shift = _positive_integer(
            row.get("donor_shift"), label=f"shuffled shift {expected_shift} donor"
        )
        positive_task_count = _positive_integer(
            row.get("positive_task_count"),
            label=f"shuffled shift {expected_shift} positive task count",
        )
        positive_seed_count = _positive_integer(
            row.get("positive_seed_count"),
            label=f"shuffled shift {expected_shift} positive seed count",
        )
        if (
            donor_shift != expected_shift
            or positive_task_count < minimum_tasks
            or positive_seed_count < minimum_seeds
            or _finite_number(
                row.get("effect"), label=f"shuffled shift {expected_shift} effect"
            )
            < minimum_effect
        ):
            raise ValueError(
                f"Development-Go shuffled-action shift {expected_shift} is invalid"
            )

    raw = _mapping(payload.get("raw_h6_h8"), "Development-Go raw H6-H8")
    raw_methods = _mapping(raw.get("methods"), "Development-Go raw methods")
    raw_mse: dict[str, float] = {}
    for method in ("factual", "no_action", "hold_last", "latent_constant_velocity"):
        row = _mapping(raw_methods.get(method), f"Development-Go raw {method}")
        raw_mse[method] = _finite_number(
            row.get("task_macro_mse"), label=f"raw {method} task-macro MSE"
        )
        if raw_mse[method] < 0.0:
            raise ValueError("Development-Go raw MSE values must be non-negative")
    if (
        raw_mse["factual"] > raw_mse["no_action"]
        or raw_mse["factual"] >= raw_mse["hold_last"]
        or raw_mse["factual"] >= raw_mse["latent_constant_velocity"]
    ):
        raise ValueError("Development-Go raw factual baselines do not pass")

    gate = _mapping(payload.get("gate"), "Development-Go gate")
    checks = _mapping(gate.get("checks"), "Development-Go gate checks")
    if (
        gate.get("decision") != "Development-Go"
        or gate.get("passed") is not True
        or gate.get("failed_checks") != []
        or gate.get("minimum_relative_improvement") != minimum_effect
        or set(checks) != DEVELOPMENT_GATE_CHECKS
        or checks.get("cross_seed_task_replication") is not True
        or any(value is not True for value in checks.values())
    ):
        raise ValueError("Development-Go nested gate is incomplete or inconsistent")
    _validate_member_subset(
        gate.get("jointly_positive_tasks"),
        allowed=tasks,
        minimum=minimum_tasks,
        label="jointly positive A3 tasks",
    )
    _validate_member_subset(
        gate.get("jointly_positive_seeds"),
        allowed=DEVELOPMENT_SEEDS,
        minimum=minimum_seeds,
        label="jointly positive A3 seeds",
    )
    cross_seeds = _validate_member_subset(
        gate.get("cross_replicating_seeds"),
        allowed=DEVELOPMENT_SEEDS,
        minimum=minimum_cross_seeds,
        label="cross-replicating A3 seeds",
    )
    cross_counts = _mapping(
        gate.get("cross_replicating_seed_task_counts"),
        "cross-replicating A3 seed task counts",
    )
    expected_seed_keys = {str(seed) for seed in DEVELOPMENT_SEEDS}
    if set(cross_counts) != expected_seed_keys:
        raise ValueError("cross-replicating A3 seed task counts are incomplete")
    qualifying_seeds: list[int] = []
    for seed in DEVELOPMENT_SEEDS:
        count = _nonnegative_integer(
            cross_counts[str(seed)],
            label=f"cross-replicating A3 task count for seed {seed}",
        )
        if count > len(tasks):
            raise ValueError("cross-replicating A3 task count exceeds task coverage")
        if count >= minimum_tasks_per_cross_seed:
            qualifying_seeds.append(seed)
    if tuple(qualifying_seeds) != cross_seeds:
        raise ValueError("cross-replicating A3 seed evidence is inconsistent")
    return payload


def _a3_option_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_lewm_action_branch_a3_confirm.py",
        description=__doc__,
        add_help=False,
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--development-summary",
        "--development-go-summary",
        dest="development_summary",
        type=Path,
        default=None,
        help="passed A3 2k development summary authorizing confirmation",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "-h" in arguments or "--help" in arguments:
        print(_a3_option_parser().format_help().rstrip())
        print("\nShared A1 trainer options:")
    wrapper, delegated = _a3_option_parser().parse_known_args(arguments)
    args = a1_trainer.parse_args(delegated)
    if args.screen_summary is not None:
        raise ValueError(
            "A3 confirmation does not accept A1 --screen-summary; "
            "use --development-summary"
        )
    args.config = wrapper.config
    args.development_summary = wrapper.development_summary
    args._a1_argv = tuple(delegated)
    return args


def _validate_cli(args: argparse.Namespace, contract: Mapping[str, Any]) -> None:
    confirmation = _mapping(
        contract.get("confirmation_after_development_go"),
        "confirmation_after_development_go",
    )
    if args.budget != "10k":
        raise ValueError("A3 confirmation authorizes only the 10k budget")
    if args.development_summary is None:
        raise ValueError("--development-summary is required for A3 confirmation")
    if int(args.seed) not in CONFIRMATION_SEEDS:
        raise ValueError("A3 confirmation seed must be 29301, 29302, or 29303")
    resolved_splits = (
        args.train_split or confirmation["train_split"],
        args.val_split or confirmation["evaluation_split"],
    )
    if resolved_splits != ("train", "test"):
        raise ValueError("A3 confirmation splits are frozen at train/test")
    if args.batch_size != confirmation["bundle_batch_size"]:
        raise ValueError("A3 confirmation bundle batch size is frozen at 16")
    device = str(args.device).lower()
    explicit_cuda = device == "cuda" or (
        device.startswith("cuda:") and device.removeprefix("cuda:").isdigit()
    )
    if args.amp is not True or not explicit_cuda:
        raise ValueError("A3 confirmation requires an explicit CUDA device with AMP")
    if args.amp_dtype != confirmation["amp_dtype"]:
        raise ValueError("A3 confirmation AMP dtype is frozen at float16")
    optimizer = (
        float(args.lr),
        float(args.weight_decay),
        float(args.beta1),
        float(args.beta2),
        float(args.eps),
        int(args.warmup_steps),
        float(args.grad_clip),
    )
    if optimizer != (3.0e-4, 1.0e-4, 0.9, 0.999, 1.0e-8, 100, 1.0):
        raise ValueError("A3 confirmation reuses the frozen A1 optimizer")


def _validate_frozen_base(path: str | Path) -> Path:
    """Reject A3 2k endpoints and any non-SIGReg=0.02 base before training."""

    source = _resolve(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    checkpoint = a1_trainer._torch_load(source)
    validate_frozen_base_checkpoint(checkpoint)
    return source


def _validate_confirmation_input(
    path: str | Path, *, contract: Mapping[str, Any]
) -> Path:
    """Require the canonical input to contain frozen train/test rows only."""

    source = _resolve(path)
    arrays = load_a1_npz(source)
    validate_a3_confirmation_canonical_conformance(arrays, contract)
    return source


def _replace_option(argv: Sequence[str], option: str, value: str) -> list[str]:
    result: list[str] = []
    found = False
    index = 0
    while index < len(argv):
        token = str(argv[index])
        if token == option:
            if index + 1 >= len(argv):
                raise ValueError(f"{option} is missing its value")
            result.extend((option, value))
            found = True
            index += 2
            continue
        if token.startswith(f"{option}="):
            result.extend((option, value))
            found = True
            index += 1
            continue
        result.append(token)
        index += 1
    if not found:
        result.extend((option, value))
    return result


@contextmanager
def _a3_confirmation_trainer_scope(
    *,
    development_summary: Path,
    authorization: Mapping[str, Any],
) -> Iterator[None]:
    """Install the A3 protocol, train/test split, and validated 10k marker."""

    original_protocol = a1_trainer.ACTION_BRANCH_TRAINING_PROTOCOL
    original_splits = a1_trainer._default_splits
    original_authorization = a1_trainer._validate_10k_authorization
    marker_path = development_summary.resolve()

    def frozen_splits(
        budget: str, train_split: str | None, val_split: str | None
    ) -> tuple[str, str]:
        if budget != "10k":
            raise ValueError("A3 confirmation authorizes only the 10k budget")
        resolved = (train_split or "train", val_split or "test")
        if resolved != ("train", "test"):
            raise ValueError("A3 confirmation splits are frozen at train/test")
        return resolved

    def development_go_authorization(marker: Path | None) -> dict[str, Any] | None:
        if marker is None:
            return None
        if _resolve(marker) != marker_path:
            raise ValueError("A3 Development-Go authorization marker is invalid")
        return dict(authorization)

    a1_trainer.ACTION_BRANCH_TRAINING_PROTOCOL = PROTOCOL
    a1_trainer._default_splits = frozen_splits
    a1_trainer._validate_10k_authorization = development_go_authorization
    try:
        yield
    finally:
        a1_trainer.ACTION_BRANCH_TRAINING_PROTOCOL = original_protocol
        a1_trainer._default_splits = original_splits
        a1_trainer._validate_10k_authorization = original_authorization


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    contract = _load_fixed_config(args.config)
    tasks = _validate_confirmation_contract(contract)
    _validate_cli(args, contract)
    summary_path = _resolve(args.development_summary)
    authorization = _validate_development_go_summary(
        summary_path,
        contract=contract,
        tasks=tasks,
    )
    input_path = _validate_confirmation_input(args.input_npz, contract=contract)
    base_path = _validate_frozen_base(args.base_checkpoint)

    delegated = list(args._a1_argv)
    for option, value in (
        ("--input-npz", str(input_path)),
        ("--base-checkpoint", str(base_path)),
        ("--output-dir", str(_resolve(args.output_dir))),
        ("--budget", "10k"),
        ("--seed", str(int(args.seed))),
        ("--device", str(args.device).lower()),
        ("--train-split", "train"),
        ("--val-split", "test"),
        ("--batch-size", "16"),
        ("--amp-dtype", "float16"),
        ("--screen-summary", str(summary_path)),
    ):
        delegated = _replace_option(delegated, option, value)

    with _a3_confirmation_trainer_scope(
        development_summary=summary_path,
        authorization=authorization,
    ):
        return a1_trainer.main(delegated)


if __name__ == "__main__":
    raise SystemExit(main())
