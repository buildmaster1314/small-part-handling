#!/usr/bin/env python3
"""Freeze every grounded LeWM training input into a provenance manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a stable JSON manifest for all grounded LeWM datasets, "
            "correction caches, corrector checkpoints, config, and Git state."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="LeWM training YAML containing data.sources",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON manifest",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Git worktree containing relative source paths (auto-detected by default)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly replace an existing output manifest",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output.exists() and not args.force:
        parser.error(f"output already exists; pass --force to replace it: {args.output}")

    from fcpg.lewm_provenance import (
        ManifestError,
        create_manifest_from_yaml,
        write_manifest,
    )

    try:
        manifest = create_manifest_from_yaml(
            args.config,
            repo_root=args.repo_root,
        )
        output = write_manifest(args.output, manifest, force=args.force)
    except FileExistsError:
        parser.error(f"output already exists; pass --force to replace it: {args.output}")
    except (ManifestError, OSError, TypeError) as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "manifest": str(output),
                "schema_version": manifest["schema_version"],
                "source_count": len(manifest["sources"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
