#!/usr/bin/env python3
"""Summarize paired 2x2 policy-factor effects from rollout CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_paired_rollouts import PAIR_KEYS
from scripts.run_action_chunk_rollout_matrix import normalized_auc, severity_token


def read_success(path: Path) -> dict[tuple[str, ...], float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"rollout CSV is empty: {path}")
    missing = [key for key in PAIR_KEYS if key not in rows[0]]
    if missing:
        raise KeyError(f"rollout CSV is missing deterministic pair keys: {missing}")
    keyed: dict[tuple[str, ...], float] = {}
    for row in rows:
        key = tuple(row[name] for name in PAIR_KEYS)
        if key in keyed:
            raise ValueError(f"duplicate paired episode in {path}: {key}")
        keyed[key] = float(row["success"])
    return keyed


def paired_effect(
    values: np.ndarray,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"paired effect values must be non-empty [N], got {array.shape}")
    indices = rng.integers(0, array.size, size=(bootstrap_samples, array.size))
    estimates = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "bootstrap_95_ci": [
            float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975)),
        ],
    }


def factorial_effects(
    predictive_sig0: np.ndarray,
    predictive_sigreg: np.ndarray,
    action_sig0: np.ndarray,
    action_sigreg: np.ndarray,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, dict[str, float | list[float]]]:
    arrays = [
        np.asarray(values, dtype=np.float64)
        for values in (predictive_sig0, predictive_sigreg, action_sig0, action_sigreg)
    ]
    if len({values.shape for values in arrays}) != 1:
        raise ValueError(f"factorial vectors must have matching shapes: {[x.shape for x in arrays]}")
    pred0, predreg, action0, actionreg = arrays
    rng = np.random.default_rng(seed)
    return {
        "sigreg_under_predictive": paired_effect(
            predreg - pred0, bootstrap_samples, rng
        ),
        "sigreg_under_action_only": paired_effect(
            actionreg - action0, bootstrap_samples, rng
        ),
        "next_latent_under_sig0": paired_effect(
            pred0 - action0, bootstrap_samples, rng
        ),
        "next_latent_under_sigreg": paired_effect(
            predreg - actionreg, bootstrap_samples, rng
        ),
        "interaction": paired_effect(
            (predreg - pred0) - (actionreg - action0),
            bootstrap_samples,
            rng,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--severity", action="append", type=float)
    parser.add_argument("--predictive-sig0", default="sig0")
    parser.add_argument("--predictive-sigreg", default="sig002")
    parser.add_argument("--action-sig0", default="action_sig0")
    parser.add_argument("--action-sigreg", default="action_sig002")
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    severities = sorted(set(args.severity or [0.0, 5.0, 10.0, 20.0]))
    labels = {
        "predictive_sig0": args.predictive_sig0,
        "predictive_sigreg": args.predictive_sigreg,
        "action_sig0": args.action_sig0,
        "action_sigreg": args.action_sigreg,
    }
    root = Path(args.root)
    rows: list[dict[str, object]] = []
    successes: dict[tuple[str, str, float], float] = {}
    for task in args.task:
        for severity_index, severity in enumerate(severities):
            keyed = {
                factor: read_success(
                    root
                    / label
                    / task
                    / severity_token(severity)
                    / "native"
                    / "fcpg_rollout.csv"
                )
                for factor, label in labels.items()
            }
            key_sets = {factor: set(values) for factor, values in keyed.items()}
            reference = key_sets["predictive_sig0"]
            for factor, keys in key_sets.items():
                if keys != reference:
                    raise ValueError(
                        f"pair-key mismatch for task={task} severity={severity:g} factor={factor}: "
                        f"missing={len(reference - keys)} extra={len(keys - reference)}"
                    )
            ordered = sorted(reference)
            vectors = {
                factor: np.asarray([values[key] for key in ordered], dtype=np.float64)
                for factor, values in keyed.items()
            }
            for factor, values in vectors.items():
                successes[(task, factor, severity)] = float(values.mean())
            effects = factorial_effects(
                vectors["predictive_sig0"],
                vectors["predictive_sigreg"],
                vectors["action_sig0"],
                vectors["action_sigreg"],
                args.bootstrap_samples,
                args.seed + severity_index,
            )
            rows.append(
                {
                    "task": task,
                    "severity_deg": severity,
                    "episodes": len(ordered),
                    "success": {
                        factor: float(values.mean()) for factor, values in vectors.items()
                    },
                    "effects": effects,
                }
            )

    auc_rows: list[dict[str, object]] = []
    for task in args.task:
        auc = {
            factor: normalized_auc(
                [(severity, successes[(task, factor, severity)]) for severity in severities]
            )
            for factor in labels
        }
        auc_rows.append(
            {
                "task": task,
                "success_auc": auc,
                "sigreg_under_predictive": auc["predictive_sigreg"] - auc["predictive_sig0"],
                "sigreg_under_action_only": auc["action_sigreg"] - auc["action_sig0"],
                "next_latent_under_sig0": auc["predictive_sig0"] - auc["action_sig0"],
                "next_latent_under_sigreg": auc["predictive_sigreg"] - auc["action_sigreg"],
                "interaction": (
                    auc["predictive_sigreg"]
                    - auc["predictive_sig0"]
                    - auc["action_sigreg"]
                    + auc["action_sig0"]
                ),
            }
        )

    payload = {"labels": labels, "severity_effects": rows, "auc_effects": auc_rows}
    json_path = root / "factorial_summary.json"
    markdown_path = root / "factorial_summary.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Shared Policy 2x2 Factorial Effects",
        "",
        "All effects are paired success-rate differences. The interaction is "
        "(predictive SIGReg - predictive no-SIGReg) - "
        "(action-only SIGReg - action-only no-SIGReg).",
        "",
        "| task | rotation | SIGReg under predictive | SIGReg under action-only | next-latent under SIGReg=0 | next-latent under SIGReg | interaction |",
        "| --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        effects = row["effects"]
        formatted = []
        for name in (
            "sigreg_under_predictive",
            "sigreg_under_action_only",
            "next_latent_under_sig0",
            "next_latent_under_sigreg",
            "interaction",
        ):
            effect = effects[name]
            ci = effect["bootstrap_95_ci"]
            formatted.append(
                f"{float(effect['mean']):+.4f} [{float(ci[0]):+.4f}, {float(ci[1]):+.4f}]"
            )
        lines.append(
            f"| {row['task']} | {float(row['severity_deg']):g} | "
            + " | ".join(formatted)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Normalized native-success AUC effects",
            "",
            "| task | predictive/no-SIGReg | predictive/SIGReg | action/no-SIGReg | action/SIGReg | SIGReg under predictive | SIGReg under action | next-latent under no-SIGReg | next-latent under SIGReg | interaction |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in auc_rows:
        auc = row["success_auc"]
        lines.append(
            f"| {row['task']} | {auc['predictive_sig0']:.4f} | "
            f"{auc['predictive_sigreg']:.4f} | {auc['action_sig0']:.4f} | "
            f"{auc['action_sigreg']:.4f} | {row['sigreg_under_predictive']:+.4f} | "
            f"{row['sigreg_under_action_only']:+.4f} | {row['next_latent_under_sig0']:+.4f} | "
            f"{row['next_latent_under_sigreg']:+.4f} | {row['interaction']:+.4f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(markdown_path.read_text(encoding="utf-8"))
    print(f"json: {json_path}")
    print(f"markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
