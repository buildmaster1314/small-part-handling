#!/usr/bin/env python3
"""Train A2 matched action/no-action heads through the frozen A1 trainer."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a2 import (  # noqa: E402
    PROTOCOL,
    summarize_intervention_support,
    validate_a2_contract,
)
from fcpg.lewm_action_branch_a1 import load_a1_npz  # noqa: E402
from fcpg.lewm_action_branch_training import (  # noqa: E402
    BASE_SIGREG_WEIGHT,
    BUDGET_STEPS,
)
from scripts import train_lewm_action_branch_a1 as a1_trainer  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_continuous_task_a2.json"
FROZEN_TASKS = (
    "pick-place-v3",
    "drawer-open-v3",
    "reach-v3",
    "push-v3",
)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _resolve(path: str | Path) -> Path:
    source = Path(path).expanduser()
    return source.resolve() if source.is_absolute() else (ROOT / source).resolve()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return payload


def _validate_training_contract(contract: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate the A2 fields that govern reuse of the A1 trainer."""

    tasks = validate_a2_contract(contract)
    if tasks != FROZEN_TASKS:
        raise ValueError(f"A2 tasks must remain fixed at {FROZEN_TASKS}")

    branches = _mapping(
        contract.get("matched_state_branches"), "matched_state_branches"
    )
    branch_values = (
        int(branches.get("branch_count", -1)),
        int(branches.get("interruption_horizon_steps", -1)),
        int(branches.get("raw_steps_per_model_step", -1)),
        float(branches.get("motion_delta_rms", float("nan"))),
        branches.get("preserve_policy_gripper_commands"),
    )
    if branch_values != (8, 8, 5, 0.15, True):
        raise ValueError("A2 matched-state branch configuration is frozen")

    gate = _mapping(contract.get("support_gate"), "support_gate")
    gate_values = (
        gate.get("split"),
        gate.get("train_only_whitening"),
        tuple(int(value) for value in gate.get("primary_horizons", ())),
        float(gate.get("minimum_branch_variance", float("nan"))),
        float(
            gate.get(
                "minimum_episode_macro_informative_fraction_per_task",
                float("nan"),
            )
        ),
        int(gate.get("minimum_development_episodes_per_task", -1)),
        gate.get("all_four_tasks_must_pass"),
        gate.get("model_training_before_gate"),
    )
    if gate_values != ("development", True, (6, 7, 8), 0.01, 0.8, 12, True, False):
        raise ValueError("A2 intervention-support Gate configuration is frozen")

    authorization = _mapping(
        contract.get("authorized_after_support_go"),
        "authorized_after_support_go",
    )
    screen_steps = int(
        authorization.get("matched_factual_and_no_action_screen_steps", -1)
    )
    if screen_steps != BUDGET_STEPS["2k"]:
        raise ValueError("A2 matched-head screen must remain fixed at 2000 steps")
    if authorization.get("screen_training_seeds") != [25801, 25802, 25803]:
        raise ValueError("A2 2k screen seeds must remain fixed at 25801/25802/25803")
    if int(authorization.get("bundle_batch_size", -1)) != 16:
        raise ValueError("A2 cached bundle batch size must remain fixed at 16")
    if authorization.get("amp_dtype") != "float16":
        raise ValueError("A2 2k screen AMP dtype must remain float16")
    try:
        sigreg_weight = float(authorization["anchor_encoder_sigreg"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "A2 config must specify numeric anchor_encoder_sigreg"
        ) from error
    if sigreg_weight != BASE_SIGREG_WEIGHT:
        raise ValueError(
            f"A2 anchor encoder SIGReg must be exactly {BASE_SIGREG_WEIGHT}"
        )
    if authorization.get("additional_sigreg_values_allowed") != []:
        raise ValueError("A2 does not allow additional SIGReg values")
    if authorization.get("fresh_10000_step_training_requires_2k_screen_go") is not True:
        raise ValueError("A2 fresh 10k training must require a 2k Screen Go")
    return tasks


def _load_fixed_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load a byte-independent structured copy of the checked-in A2 config."""

    canonical_path = DEFAULT_CONFIG.resolve()
    canonical = _read_json(canonical_path, label="canonical A2 config")
    source = _resolve(path)
    payload = (
        canonical
        if source == canonical_path
        else _read_json(source, label="A2 config")
    )
    _validate_training_contract(payload)
    if payload != canonical:
        raise ValueError(
            "A2 training config is fixed to the checked-in continuous-task contract"
        )
    return payload


def _validate_support_authorization(
    path: str | Path | None,
    *,
    task_names: Sequence[str],
) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = _read_json(_resolve(path), label="A2 support summary")
    if (
        payload.get("protocol") != PROTOCOL
        or payload.get("stage") != "collection_only_intervention_support"
        or payload.get("decision") != "Development-Go"
        or payload.get("model_training_started") is not False
        or payload.get("input_contains_test_rows") is not False
        or payload.get("late_horizon_development_only") is not True
        or payload.get("primary_horizons") != [6, 7, 8]
    ):
        raise ValueError("2k requires the unexposed-test A2 H6-H8 Development-Go")
    per_task = payload.get("per_task")
    if not isinstance(per_task, Mapping) or set(per_task) != set(task_names):
        raise ValueError("A2 support Go must cover exactly the four frozen tasks")
    if any(
        not isinstance(per_task[task], Mapping)
        or per_task[task].get("passed") is not True
        for task in task_names
    ):
        raise ValueError("A2 support Go requires every frozen task to pass")
    return payload


def _recompute_support_authorization(
    input_npz: str | Path,
    *,
    contract: Mapping[str, Any],
    task_names: Sequence[str],
) -> dict[str, Any]:
    """Recompute the development support Gate from the actual training input."""

    gate = _mapping(contract.get("support_gate"), "support_gate")
    arrays = load_a1_npz(_resolve(input_npz))
    summary = summarize_intervention_support(
        arrays,
        task_names=task_names,
        primary_horizons=tuple(int(value) for value in gate["primary_horizons"]),
        minimum_branch_variance=float(gate["minimum_branch_variance"]),
        minimum_episode_macro_fraction_per_task=float(
            gate["minimum_episode_macro_informative_fraction_per_task"]
        ),
        minimum_development_episodes_per_task=int(
            gate["minimum_development_episodes_per_task"]
        ),
    )
    if summary.get("decision") != "Go" or any(
        row.get("passed") is not True for row in summary["per_task"].values()
    ):
        raise ValueError(
            "A2 input does not reproduce the frozen H6-H8 intervention-support Go"
        )
    return summary


def _default_splits(
    budget: str,
    train_split: str | None,
    val_split: str | None,
    *,
    test_authorized: bool = False,
) -> tuple[str, str]:
    try:
        expected = {
            "smoke": ("smoke", "smoke"),
            "2k": ("train", "development"),
            "10k": ("train", "test"),
        }[budget]
    except KeyError as error:
        raise ValueError(
            f"unknown A2 budget {budget!r}; expected {sorted(BUDGET_STEPS)}"
        ) from error
    if test_authorized and budget != "10k":
        raise ValueError("--authorize-test is valid only for the 10k budget")
    if budget == "10k":
        raise ValueError(
            "A2 10k/test remains locked until the 2k development Gate passes "
            "and a fresh confirmation protocol is added"
        )
    resolved = (train_split or expected[0], val_split or expected[1])
    if resolved != expected:
        raise ValueError(
            f"A2 budget {budget} freezes train/val splits to {expected}, got {resolved}"
        )
    return resolved


def _a2_option_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_lewm_action_branch_a2.py",
        description="A2-specific gates layered over the shared A1 trainer.",
        add_help=False,
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--support-summary",
        type=Path,
        default=None,
        help="collection-only A2 support Go required by the 2k screen",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "-h" in arguments or "--help" in arguments:
        print(_a2_option_parser().format_help().rstrip())
        print("\nShared A1 trainer options:")
    wrapper, a1_argv = _a2_option_parser().parse_known_args(arguments)
    args = a1_trainer.parse_args(a1_argv)
    if args.screen_summary is not None:
        raise ValueError(
            "A2 does not accept A1 --screen-summary; use explicit --authorize-test"
        )
    args.config = wrapper.config
    args.support_summary = wrapper.support_summary
    args.authorize_test = False
    args._a1_argv = tuple(a1_argv)
    return args


@contextmanager
def _a2_trainer_scope(*, test_authorized: bool) -> Iterator[None]:
    """Adapt A1's protocol hooks for one synchronous A2 CLI invocation."""

    original_protocol = a1_trainer.ACTION_BRANCH_TRAINING_PROTOCOL
    original_splits = a1_trainer._default_splits
    original_authorization = a1_trainer._validate_10k_authorization

    def frozen_splits(
        budget: str, train_split: str | None, val_split: str | None
    ) -> tuple[str, str]:
        return _default_splits(
            budget,
            train_split,
            val_split,
            test_authorized=test_authorized,
        )

    def explicit_authorization(marker: Path | None) -> dict[str, Any] | None:
        if marker is None:
            return None
        if not test_authorized or marker.resolve() != DEFAULT_CONFIG.resolve():
            raise ValueError("A2 test authorization marker is invalid")
        return {
            "protocol": PROTOCOL,
            "stage": "screen",
            "decision": "Go",
            "passed": True,
            "explicit_test_authorization": True,
        }

    a1_trainer.ACTION_BRANCH_TRAINING_PROTOCOL = PROTOCOL
    a1_trainer._default_splits = frozen_splits
    a1_trainer._validate_10k_authorization = explicit_authorization
    try:
        yield
    finally:
        a1_trainer.ACTION_BRANCH_TRAINING_PROTOCOL = original_protocol
        a1_trainer._default_splits = original_splits
        a1_trainer._validate_10k_authorization = original_authorization


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    contract = _load_fixed_config(args.config)
    task_names = _validate_training_contract(contract)
    _default_splits(
        args.budget,
        args.train_split,
        args.val_split,
        test_authorized=args.authorize_test,
    )

    if args.budget == "2k":
        if args.support_summary is None:
            raise ValueError("--support-summary is required for the A2 2k screen")
        _validate_support_authorization(
            args.support_summary,
            task_names=task_names,
        )
        _recompute_support_authorization(
            args.input_npz,
            contract=contract,
            task_names=task_names,
        )
        authorization = _mapping(
            contract.get("authorized_after_support_go"),
            "authorized_after_support_go",
        )
        if int(args.seed) not in set(authorization["screen_training_seeds"]):
            raise ValueError("A2 2k seed is not one of the three frozen screen seeds")
        if args.batch_size != int(authorization["bundle_batch_size"]):
            raise ValueError("A2 2k bundle batch size is frozen at 16")
        if args.amp is not True:
            raise ValueError("A2 2k training requires CUDA AMP")
        if not str(args.device).lower().startswith("cuda"):
            raise ValueError("A2 2k training requires an explicit CUDA device")
        if args.amp_dtype != authorization["amp_dtype"]:
            raise ValueError("A2 2k AMP dtype is frozen at float16")
    elif args.support_summary is not None:
        raise ValueError("--support-summary is valid only for the 2k budget")

    delegated_argv = list(args._a1_argv)
    if args.authorize_test:
        # A1's runner expects a non-null authorization input; the scoped A2 hook
        # interprets this fixed config path only as the already-validated CLI flag.
        delegated_argv.extend(("--screen-summary", str(DEFAULT_CONFIG)))
    with _a2_trainer_scope(test_authorized=args.authorize_test):
        return a1_trainer.main(delegated_argv)


if __name__ == "__main__":
    raise SystemExit(main())
