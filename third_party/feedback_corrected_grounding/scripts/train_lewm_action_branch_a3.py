#!/usr/bin/env python3
"""Train the frozen A3 2k matched factual/no-action dynamics screen."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a1 import load_a1_npz  # noqa: E402
from fcpg.lewm_action_branch_a3 import summarize_a3_support  # noqa: E402
from fcpg.lewm_action_branch_a3_collection import (  # noqa: E402
    PROTOCOL,
    TASKS,
    validate_a3_cache_conformance,
    validate_a3_contract,
)
from fcpg.lewm_action_branch_a3_physical import (  # noqa: E402
    validate_a3_physical_arrays,
)
from fcpg.lewm_action_branch_training import (  # noqa: E402
    BASE_SIGREG_WEIGHT,
    BUDGET_STEPS,
)
from scripts import train_lewm_action_branch_a1 as a1_trainer  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_physical_support_a3.json"
DEFAULT_PHYSICAL = ROOT / "outputs/lewm_action_branch_a3/action_branch_a3_physical.npz"
FROZEN_SEEDS = (29_201, 29_202, 29_203)


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


def _load_fixed_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    canonical = _read_json(DEFAULT_CONFIG, label="canonical A3 config")
    source = _resolve(path)
    payload = canonical if source == DEFAULT_CONFIG.resolve() else _read_json(
        source, label="A3 config"
    )
    validate_a3_contract(payload)
    if payload != canonical:
        raise ValueError("A3 training config is fixed to the checked-in contract")
    return payload


def _validate_training_contract(contract: Mapping[str, Any]) -> tuple[str, ...]:
    tasks = validate_a3_contract(contract)
    training = _mapping(contract.get("training_if_gate_go"), "training_if_gate_go")
    observed = (
        int(training.get("optimizer_steps", -1)),
        tuple(int(value) for value in training.get("seeds", ())),
        int(training.get("bundle_batch_size", -1)),
        training.get("amp_dtype"),
        float(training.get("anchor_encoder_sigreg", float("nan"))),
        training.get("additional_sigreg_values_allowed"),
        training.get("selection"),
        training.get("ten_thousand_steps_authorized_only_after_two_thousand_step_go"),
    )
    expected = (
        BUDGET_STEPS["2k"],
        FROZEN_SEEDS,
        16,
        "float16",
        BASE_SIGREG_WEIGHT,
        [],
        "fixed_latest",
        True,
    )
    if observed != expected:
        raise ValueError("A3 2k training contract drifted")
    return tasks


def _validate_support_summary(path: str | Path, *, tasks: Sequence[str]) -> dict[str, Any]:
    payload = _read_json(path, label="A3 support summary")
    if (
        payload.get("protocol") != PROTOCOL
        or payload.get("stage") != "collection_only_physical_and_latent_support"
        or payload.get("decision") != "Go"
        or payload.get("model_training_started") is not False
        or payload.get("input_contains_test_rows") is not False
        or payload.get("primary_horizons") != [6, 7, 8]
        or payload.get("physical_support_passed") is not True
        or payload.get("latent_support_passed") is not True
        or payload.get("joint_support_passed") is not True
    ):
        raise ValueError("A3 2k requires the frozen physical+latent support Go")
    per_task = payload.get("per_task")
    if not isinstance(per_task, Mapping) or set(per_task) != set(tasks):
        raise ValueError("A3 support summary must cover exactly the frozen tasks")
    for task in tasks:
        row = per_task[task]
        if (
            not isinstance(row, Mapping)
            or row.get("passed") is not True
            or not isinstance(row.get("physical"), Mapping)
            or row["physical"].get("passed") is not True
            or not isinstance(row.get("latent"), Mapping)
            or row["latent"].get("passed") is not True
            or not isinstance(row.get("joint"), Mapping)
            or row["joint"].get("passed") is not True
        ):
            raise ValueError(f"A3 support summary does not pass for {task}")
    return payload


def _load_physical(path: str | Path) -> dict[str, np.ndarray]:
    source = _resolve(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _recompute_support(
    input_npz: str | Path,
    physical_npz: str | Path,
    *,
    contract: Mapping[str, Any],
    tasks: Sequence[str],
) -> dict[str, Any]:
    latent = load_a1_npz(_resolve(input_npz))
    validate_a3_cache_conformance(latent, contract)
    physical = _load_physical(physical_npz)
    validate_a3_physical_arrays(physical, latent_arrays=latent)
    gate = _mapping(contract.get("support_gate"), "support_gate")
    split = _mapping(contract.get("split"), "split")
    summary = summarize_a3_support(
        latent,
        physical,
        task_names=tasks,
        primary_horizons=tuple(int(value) for value in gate["primary_horizons"]),
        minimum_latent_variance=float(gate["minimum_latent_branch_variance"]),
        minimum_tcp_spread_m=float(gate["minimum_tcp_max_pairwise_m"]),
        minimum_object_spread_m=float(gate["minimum_object_max_pairwise_m"]),
        minimum_episode_macro_fraction=float(
            gate["minimum_episode_macro_informative_fraction_per_task"]
        ),
        expected_train_episodes=int(split["train_episodes_per_task"]),
        expected_development_episodes=int(split["development_episodes_per_task"]),
    )
    if (
        summary.get("decision") != "Go"
        or summary.get("physical_support_passed") is not True
        or summary.get("latent_support_passed") is not True
        or summary.get("joint_support_passed") is not True
    ):
        raise ValueError("A3 inputs do not reproduce the frozen support Go")
    return summary


def _a3_option_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_lewm_action_branch_a3.py",
        description=__doc__,
        add_help=False,
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--physical-data", type=Path, default=DEFAULT_PHYSICAL)
    parser.add_argument("--support-summary", type=Path, required=False)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "-h" in arguments or "--help" in arguments:
        print(_a3_option_parser().format_help().rstrip())
        print("\nShared A1 trainer options:")
    wrapper, delegated = _a3_option_parser().parse_known_args(arguments)
    args = a1_trainer.parse_args(delegated)
    if args.screen_summary is not None:
        raise ValueError("A3 does not accept the A1 --screen-summary option")
    args.config = wrapper.config
    args.physical_data = wrapper.physical_data
    args.support_summary = wrapper.support_summary
    args._a1_argv = tuple(delegated)
    return args


def _replace_option(argv: Sequence[str], option: str, value: str) -> list[str]:
    """Replace one argparse option while preserving all unrelated arguments."""

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
def _a3_trainer_scope() -> Iterator[None]:
    original_protocol = a1_trainer.ACTION_BRANCH_TRAINING_PROTOCOL
    original_splits = a1_trainer._default_splits
    original_authorization = a1_trainer._validate_10k_authorization

    def frozen_splits(
        budget: str, train_split: str | None, val_split: str | None
    ) -> tuple[str, str]:
        if budget != "2k":
            raise ValueError("A3 currently authorizes only the 2k development screen")
        resolved = (train_split or "train", val_split or "development")
        if resolved != ("train", "development"):
            raise ValueError("A3 2k splits are frozen at train/development")
        return resolved

    def no_10k(marker: Path | None) -> None:
        if marker is not None:
            raise ValueError("A3 10k authorization is not available before 2k Go")
        return None

    a1_trainer.ACTION_BRANCH_TRAINING_PROTOCOL = PROTOCOL
    a1_trainer._default_splits = frozen_splits
    a1_trainer._validate_10k_authorization = no_10k
    try:
        yield
    finally:
        a1_trainer.ACTION_BRANCH_TRAINING_PROTOCOL = original_protocol
        a1_trainer._default_splits = original_splits
        a1_trainer._validate_10k_authorization = original_authorization


def _validate_cli(args: argparse.Namespace, contract: Mapping[str, Any]) -> None:
    training = _mapping(contract.get("training_if_gate_go"), "training_if_gate_go")
    if args.budget != "2k":
        raise ValueError("A3 currently authorizes only the 2k development screen")
    if args.support_summary is None:
        raise ValueError("--support-summary is required for the A3 2k screen")
    if int(args.seed) not in FROZEN_SEEDS:
        raise ValueError("A3 2k seed must be 29201, 29202, or 29203")
    if args.batch_size != int(training["bundle_batch_size"]):
        raise ValueError("A3 bundle batch size is frozen at 16")
    if args.amp is not True or not str(args.device).lower().startswith("cuda"):
        raise ValueError("A3 2k requires CUDA AMP")
    if args.amp_dtype != training["amp_dtype"]:
        raise ValueError("A3 AMP dtype is frozen at float16")
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
        raise ValueError("A3 reuses the frozen A1 optimizer without overrides")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    contract = _load_fixed_config(args.config)
    tasks = _validate_training_contract(contract)
    _validate_cli(args, contract)
    _validate_support_summary(args.support_summary, tasks=tasks)
    _recompute_support(
        args.input_npz,
        args.physical_data,
        contract=contract,
        tasks=tasks,
    )
    delegated = list(args._a1_argv)
    delegated = _replace_option(delegated, "--input-npz", str(_resolve(args.input_npz)))
    delegated = _replace_option(
        delegated,
        "--base-checkpoint",
        str(_resolve(args.base_checkpoint)),
    )
    delegated = _replace_option(
        delegated,
        "--output-dir",
        str(_resolve(args.output_dir)),
    )
    with _a3_trainer_scope():
        return a1_trainer.main(delegated)


if __name__ == "__main__":
    raise SystemExit(main())
