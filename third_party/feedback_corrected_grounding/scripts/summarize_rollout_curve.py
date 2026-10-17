#!/usr/bin/env python3
"""Summarize paired rollout statistics over a perturbation severity curve."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_point(value: str) -> tuple[float, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--point must use SEVERITY=SUMMARY_JSON")
    severity, path = value.split("=", 1)
    try:
        severity_value = float(severity)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid severity: {severity!r}") from exc
    return severity_value, Path(path)


def normalized_auc(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or float(x[-1]) == float(x[0]):
        return float(y.mean())
    return float(np.trapz(y, x) / (x[-1] - x[0]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--point", action="append", required=True, type=parse_point, metavar="SEVERITY=JSON")
    parser.add_argument("--baseline", default="nominal")
    parser.add_argument("--method", default="native")
    parser.add_argument("--oracle", default="oracle")
    parser.add_argument("--x-label", default="Actual camera rotation (deg)")
    parser.add_argument("--title", default="Same-policy grounding robustness")
    parser.add_argument("--output-dir", default="outputs/bc_rollout/rollout_curve")
    args = parser.parse_args()

    points = sorted(args.point, key=lambda item: item[0])
    if len({severity for severity, _ in points}) != len(points):
        parser.error("duplicate severity values are not allowed")

    rows: list[dict[str, float | str]] = []
    comparison_rows: list[dict[str, float]] = []
    for severity, path in points:
        payload = json.loads(path.read_text(encoding="utf-8"))
        models = payload["models"]
        for model in (args.baseline, args.method, args.oracle):
            if model not in models:
                raise KeyError(f"{path} does not contain model {model!r}")
            metrics = models[model]
            rows.append(
                {
                    "severity": severity,
                    "model": model,
                    "success_rate": float(metrics["success_rate"]),
                    "reward_sum_mean": float(metrics["reward_sum_mean"]),
                    "pixel_error_mean": float(metrics["pixel_error_mean"]),
                    "corrected_pixel_error_mean": float(metrics["corrected_pixel_error_mean"]),
                }
            )
        comparison_name = f"{args.method}_vs_{args.baseline}"
        comparison = payload["comparisons"][comparison_name]
        comparison_rows.append(
            {
                "severity": severity,
                "success_delta": float(comparison["success_rate_difference"]),
                "ci_low": float(comparison["bootstrap_95_ci"][0]),
                "ci_high": float(comparison["bootstrap_95_ci"][1]),
                "mcnemar_p": float(comparison["mcnemar_exact"]["p_value_two_sided"]),
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "rollout_curve.csv"
    comparison_csv_path = output_dir / "paired_differences.csv"
    markdown_path = output_dir / "rollout_curve.md"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with comparison_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    severities = np.asarray([severity for severity, _ in points], dtype=np.float64)
    by_model = {
        model: [row for row in rows if row["model"] == model]
        for model in (args.baseline, args.method, args.oracle)
    }
    auc = {
        model: normalized_auc(severities, np.asarray([row["success_rate"] for row in model_rows]))
        for model, model_rows in by_model.items()
    }
    lines = [
        f"# {args.title}",
        "",
        "| severity | nominal success | FCPG success | oracle success | FCPG corrected px | delta | 95% CI | p |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for index, severity in enumerate(severities):
        comp = comparison_rows[index]
        lines.append(
            f"| {severity:g} | {by_model[args.baseline][index]['success_rate']:.4f} | "
            f"{by_model[args.method][index]['success_rate']:.4f} | "
            f"{by_model[args.oracle][index]['success_rate']:.4f} | "
            f"{by_model[args.method][index]['corrected_pixel_error_mean']:.3f} | "
            f"{comp['success_delta']:+.4f} | [{comp['ci_low']:+.4f}, {comp['ci_high']:+.4f}] | "
            f"{comp['mcnemar_p']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Normalized success AUC over the reported severity range:",
            "",
            *(f"- `{model}`: {value:.4f}" for model, value in auc.items()),
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
    if plt is not None:
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        for model, label, color in (
            (args.baseline, "Nominal", "#777777"),
            (args.method, "FCPG", "#167D9A"),
            (args.oracle, "Oracle", "#C94C4C"),
        ):
            model_rows = by_model[model]
            axes[0].plot(
                severities,
                [row["success_rate"] for row in model_rows],
                marker="o",
                linewidth=2,
                label=label,
                color=color,
            )
            axes[1].plot(
                severities,
                [row["corrected_pixel_error_mean"] for row in model_rows],
                marker="o",
                linewidth=2,
                label=label,
                color=color,
            )
        axes[0].set_ylabel("Success rate")
        axes[0].set_ylim(-0.02, 1.02)
        axes[1].set_ylabel("Grounding error (px)")
        for axis in axes:
            axis.set_xlabel(args.x_label)
            axis.grid(alpha=0.25)
            axis.legend(frameon=False)
        figure.suptitle(args.title)
        figure.tight_layout()
        figure.savefig(output_dir / "rollout_curve.png", dpi=200)
        plt.close(figure)

    print(markdown_path.read_text(encoding="utf-8"))
    print(f"csv: {csv_path}")
    print(f"paired csv: {comparison_csv_path}")
    print(f"markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
