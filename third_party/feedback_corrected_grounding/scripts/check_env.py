#!/usr/bin/env python3
"""Check the Python environment needed for FCPG experiments."""

from __future__ import annotations

import importlib.util
import importlib.metadata as metadata
import sys


REQUIRED_STAGE_0 = ["numpy", "torch", "mujoco", "gymnasium"]
REQUIRED_METAWORLD = ["metaworld", "timm"]
OPTIONAL_LATER = ["rlbench", "robosuite"]


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    print(f"python: {sys.executable}")
    print(f"version: {sys.version.split()[0]}")
    missing_required = []
    for group_name, modules in [
        ("stage0_required", REQUIRED_STAGE_0),
        ("metaworld_required", REQUIRED_METAWORLD),
        ("optional_later", OPTIONAL_LATER),
    ]:
        print(f"\n[{group_name}]")
        for module in modules:
            ok = has_module(module)
            version = ""
            if ok:
                try:
                    version = f" {metadata.version(module)}"
                except metadata.PackageNotFoundError:
                    version = ""
            print(f"{module}: {'OK' if ok else 'MISSING'}{version}")
            if group_name != "optional_later" and not ok:
                missing_required.append(module)
    if has_module("metaworld"):
        try:
            import metaworld.policies as policies

            policy_names = [name for name in dir(policies) if name.startswith("Sawyer") and name.endswith("Policy")]
            print(f"\nmetaworld_scripted_policies: {len(policy_names)}")
        except Exception as exc:  # pragma: no cover - diagnostic path
            print(f"\nmetaworld_scripted_policies: ERROR {exc}")
            missing_required.append("metaworld.policies")
    if missing_required:
        print("\nMissing required modules:", ", ".join(missing_required))
        return 1
    print("\nEnvironment is ready for MetaWorld-first setup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
