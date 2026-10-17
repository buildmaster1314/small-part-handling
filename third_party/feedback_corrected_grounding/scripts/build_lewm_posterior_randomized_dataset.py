#!/usr/bin/env python3
"""Build a randomized, posterior-gate-compatible dataset from camera NPZs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.lewm_posterior_randomized_dataset import (  # noqa: E402
    build_posterior_randomized_arrays,
    load_npz_source,
)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp.npz", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_dataset(
    input_paths: Sequence[str | Path] | str | Path,
    output_path: str | Path,
    *,
    seed: int = 0,
    minimum_duration: int = 10,
    maximum_duration: int = 20,
    recovery_length: int = 10,
    onset_phase_min: float = 0.2,
    onset_phase_max: float = 0.7,
) -> Path:
    """Load source NPZs, derive the randomized dataset, and write it atomically."""

    if isinstance(input_paths, (str, Path)):
        inputs = [Path(input_paths).expanduser().resolve()]
    else:
        inputs = [Path(path).expanduser().resolve() for path in input_paths]
    if not inputs:
        raise ValueError("at least one input source is required")
    if len(set(inputs)) != len(inputs):
        raise ValueError("input paths must be distinct")
    destination = Path(output_path).expanduser().resolve()
    if destination in inputs:
        raise ValueError("output path must differ from every input path")

    sources = [load_npz_source(path) for path in inputs]
    arrays = build_posterior_randomized_arrays(
        sources,
        seed=seed,
        minimum_duration=minimum_duration,
        maximum_duration=maximum_duration,
        recovery_length=recovery_length,
        onset_phase_min=onset_phase_min,
        onset_phase_max=onset_phase_max,
    )
    _write_npz(destination, arrays)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        dest="input_paths",
        action="append",
        type=Path,
        required=True,
        help="Dynamic-camera NPZ source; repeat for multiple sources.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--minimum-duration", "--min-duration", type=int, default=10
    )
    parser.add_argument(
        "--maximum-duration", "--max-duration", type=int, default=20
    )
    parser.add_argument(
        "--recovery-length", "--minimum-recovery-frames", type=int, default=10
    )
    parser.add_argument(
        "--onset-phase-min", "--minimum-onset-phase", type=float, default=0.2
    )
    parser.add_argument(
        "--onset-phase-max", "--maximum-onset-phase", type=float, default=0.7
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = build_dataset(
        args.input_paths,
        args.output,
        seed=args.seed,
        minimum_duration=args.minimum_duration,
        maximum_duration=args.maximum_duration,
        recovery_length=args.recovery_length,
        onset_phase_min=args.onset_phase_min,
        onset_phase_max=args.onset_phase_max,
    )
    with np.load(output, allow_pickle=False) as data:
        sample_count = int(data["rgb"].shape[0])
        episode_count = len(
            set(
                zip(
                    data["task_name"].astype(str).tolist(),
                    data["episode_id"].astype(str).tolist(),
                    strict=True,
                )
            )
        )
    print(f"output: {output}")
    print(f"samples: {sample_count}")
    print(f"physical episodes: {episode_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
