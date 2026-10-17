#!/usr/bin/env python3
"""Run the bounded post-L1 same-snapshot counterfactual pilot."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path


os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_counterfactual_runtime import (  # noqa: E402
    resolve_path,
    run_counterfactual_pilot,
    smoke_config,
    validate_pilot_config,
)


def _create_exclusive_temp(output: Path) -> tuple[Path, int]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_BINARY", 0)
    for _ in range(100):
        temporary = output.with_name(
            f".{output.name}.{os.getpid()}.{secrets.token_hex(16)}.tmp"
        )
        try:
            return temporary, os.open(temporary, flags, 0o666)
        except FileExistsError:
            continue
    raise FileExistsError(
        f"could not create exclusive temporary output beside: {output}"
    )


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_text_exclusive(output: Path, payload: str) -> None:
    temporary, descriptor = _create_exclusive_temp(output)
    try:
        try:
            stream = os.fdopen(descriptor, "w", encoding="utf-8")
        except BaseException:
            os.close(descriptor)
            raise
        with stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

        try:
            # A hard link publishes the synced inode atomically and fails if the
            # final path appeared while the pilot was running.
            os.link(temporary, output)
        except FileExistsError:
            raise FileExistsError(
                f"exclusive pilot output already exists: {output}"
            ) from None
        _fsync_directory(output.parent)

        os.unlink(temporary)
        temporary = None
        _fsync_directory(output.parent)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/lewm/eval/gate_l2_pilot_seed4096.json",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the frozen one-task smoke subset",
    )
    parser.add_argument(
        "--confirm-full",
        action="store_true",
        help="required to launch the complete exploratory pilot",
    )
    args = parser.parse_args()
    if not args.smoke and not args.confirm_full:
        parser.error("full pilot requires --confirm-full")
    if args.smoke and args.confirm_full:
        parser.error("--smoke and --confirm-full are mutually exclusive")

    config_path = resolve_path(args.config, ROOT)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    validate_pilot_config(raw)
    config = smoke_config(raw) if args.smoke else raw
    output = resolve_path(config["output"]["result"], ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(
            f"exclusive pilot output already exists: {output}"
        )

    result = run_counterfactual_pilot(config, repo_root=ROOT)
    payload = json.dumps(
        result,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    _write_text_exclusive(output, payload)
    print(f"result: {output}")
    print(f"result_sha256: {result['result_sha256']}")
    print(f"rows: {result['row_count']}")
    print("scope: exploratory post-L1 pilot; not a retroactive Gate L2 pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
