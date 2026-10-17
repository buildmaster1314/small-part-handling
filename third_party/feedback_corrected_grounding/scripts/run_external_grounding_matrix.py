#!/usr/bin/env python3
"""Run matched FCPG, local-refinement, and online-calibration rollouts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROLLOUT_SCRIPT = REPO_ROOT / "scripts" / "evaluate_action_chunk_rollout.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_action_chunk_rollout_matrix import (  # noqa: E402
    PERTURBATIONS,
    PolicySpec,
    compare_runs,
    normalized_auc,
    parse_named_value,
    resolve_policy_specs,
    severity_token,
)


METHODS = ("nominal", "native", "prototype", "pnp", "oracle")


@dataclass(frozen=True)
class ExternalGroundingJob:
    checkpoint_label: str
    checkpoint: str
    task_label: str
    task_name: str
    severity: float
    method: str
    output_dir: str
    perturbation: str = "rotation"

    @property
    def model_name(self) -> str:
        return {
            "nominal": "fcpg_nominal",
            "native": "fcpg_rho_ones",
            "prototype": "fcpg_prototype",
            "pnp": "fcpg_pnp",
            "oracle": "fcpg_oracle",
        }[self.method]

    @property
    def csv_path(self) -> Path:
        return Path(self.output_dir) / f"{self.model_name}_rollout.csv"

    @property
    def summary_path(self) -> Path:
        return Path(self.output_dir) / f"{self.model_name}_rollout_summary.json"


def build_jobs(
    checkpoints: list[PolicySpec],
    tasks: list[tuple[str, str]],
    severities: list[float],
    methods: list[str],
    output_root: Path,
    perturbation: str,
) -> list[ExternalGroundingJob]:
    jobs = []
    for checkpoint in checkpoints:
        if checkpoint.source_mode != "fcpg":
            raise ValueError(
                f"external grounding requires an FCPG checkpoint, got {checkpoint.source_mode!r}"
            )
        for task_label, task_name in tasks:
            for severity in severities:
                for method in methods:
                    jobs.append(
                        ExternalGroundingJob(
                            checkpoint.label,
                            checkpoint.checkpoint,
                            task_label,
                            task_name,
                            float(severity),
                            method,
                            str(
                                output_root
                                / checkpoint.label
                                / task_label
                                / severity_token(severity, perturbation)
                                / method
                            ),
                            perturbation,
                        )
                    )
    return jobs


def rollout_command(job: ExternalGroundingJob, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(ROLLOUT_SCRIPT),
        "--checkpoint",
        job.checkpoint,
        "--task",
        job.task_name,
        "--camera",
        args.camera,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--episodes",
        str(args.episodes),
        "--task-index-offset",
        str(args.task_index_offset),
        "--horizon",
        str(args.horizon),
        "--action-steps",
        str(args.action_steps),
        "--seed",
        str(args.seed),
        "--env-seed-base",
        str(args.env_seed_base),
        "--device",
        args.device,
        "--output-dir",
        job.output_dir,
    ]
    if job.perturbation == "rotation":
        command.extend(
            (
                "--rotation-deg-min",
                str(job.severity),
                "--rotation-deg-max",
                str(job.severity),
            )
        )
    else:
        command.extend(
            (
                "--translation-cm-min",
                str(job.severity),
                "--translation-cm-max",
                str(job.severity),
            )
        )
    if job.method == "nominal":
        command.extend(("--grounding-override", "nominal"))
    elif job.method == "native":
        command.extend(
            ("--grounding-override", "native", "--reliability-override", "ones")
        )
    elif job.method == "oracle":
        command.extend(("--grounding-override", "oracle"))
    else:
        command.extend(
            (
                "--external-grounding",
                job.method,
                "--prototype-bank",
                args.prototype_bank,
                "--prototype-radius-px",
                str(args.prototype_radius_px),
                "--prototype-grid-size",
                str(args.prototype_grid_size),
                "--pnp-history-frames",
                str(args.pnp_history_frames),
                "--pnp-max-reprojection-error-px",
                str(args.pnp_max_reprojection_error_px),
                "--pnp-max-pose-rotation-deg",
                str(args.pnp_max_pose_rotation_deg),
                "--pnp-max-pose-translation-cm",
                str(args.pnp_max_pose_translation_cm),
            )
        )
    return command


def completed_job_matches(
    job: ExternalGroundingJob,
    args: argparse.Namespace,
) -> bool:
    if not job.csv_path.is_file() or not job.summary_path.is_file():
        return False
    try:
        summary = json.loads(job.summary_path.read_text(encoding="utf-8"))["summary"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False
    expected = {
        "checkpoint": job.checkpoint,
        "model": job.model_name,
        "source_mode": "fcpg",
        "task": job.task_name,
        "episodes": args.episodes,
        "horizon": args.horizon,
        "action_steps": args.action_steps,
        "grounding_override": (
            "nominal"
            if job.method == "nominal"
            else "oracle"
            if job.method == "oracle"
            else "native"
        ),
        "external_grounding": job.method if job.method in {"prototype", "pnp"} else "none",
        "reliability_override_request": "ones" if job.method == "native" else "native",
    }
    if job.perturbation == "rotation":
        expected.update(rotation_deg_min=job.severity, rotation_deg_max=job.severity)
    else:
        expected.update(translation_cm_min=job.severity, translation_cm_max=job.severity)
    if job.method in {"prototype", "pnp"}:
        expected.update(
            prototype_bank=str(Path(args.prototype_bank).resolve()),
            prototype_radius_px=args.prototype_radius_px,
            prototype_grid_size=args.prototype_grid_size,
        )
    if job.method == "pnp":
        expected.update(
            pnp_history_frames=args.pnp_history_frames,
            pnp_max_reprojection_error_px=args.pnp_max_reprojection_error_px,
            pnp_max_pose_rotation_deg=args.pnp_max_pose_rotation_deg,
            pnp_max_pose_translation_cm=args.pnp_max_pose_translation_cm,
            pnp_pose_prior_mode="camera_center_v1",
        )
    return all(summary.get(key) == value for key, value in expected.items())


def run_job(
    job: ExternalGroundingJob,
    args: argparse.Namespace,
) -> tuple[ExternalGroundingJob, str]:
    if not args.force and completed_job_matches(job, args):
        return job, "skipped"
    output_dir = Path(job.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = rollout_command(job, args)
    (output_dir / "command.json").write_text(
        json.dumps(command, indent=2) + "\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.setdefault("MUJOCO_GL", "egl")
    with (output_dir / "run.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"rollout failed: {output_dir / 'run.log'}")
    if not completed_job_matches(job, args):
        raise RuntimeError(f"rollout output failed provenance validation: {output_dir}")
    return job, "completed"


def write_summary(
    checkpoints: list[PolicySpec],
    tasks: list[tuple[str, str]],
    severities: list[float],
    methods: list[str],
    output_root: Path,
    paired_paths: dict[tuple[str, str, float], Path],
    perturbation: str,
) -> None:
    rows = []
    auc_rows = []
    for checkpoint in checkpoints:
        for task_label, _ in tasks:
            method_points = {method: [] for method in methods}
            for severity in severities:
                payload = json.loads(
                    paired_paths[(checkpoint.label, task_label, severity)].read_text(
                        encoding="utf-8"
                    )
                )
                for method in methods:
                    metrics = payload["models"][method]
                    method_points[method].append((severity, metrics["success_rate"]))
                    row = {
                        "checkpoint": checkpoint.label,
                        "task": task_label,
                        "perturbation": perturbation,
                        "severity": severity,
                        "method": method,
                        **metrics,
                    }
                    if method != "nominal":
                        row["vs_nominal"] = payload["comparisons"][f"{method}_vs_nominal"]
                    rows.append(row)
            for method, points in method_points.items():
                auc_rows.append(
                    {
                        "checkpoint": checkpoint.label,
                        "task": task_label,
                        "method": method,
                        "success_auc": normalized_auc(points),
                    }
                )
    payload = {
        "perturbation": perturbation,
        "rows": rows,
        "auc": auc_rows,
    }
    (output_root / "matrix_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    severity_label = "rotation (deg)" if perturbation == "rotation" else "translation (cm)"
    lines = [
        "# External Grounding Matrix",
        "",
        f"| checkpoint | task | {severity_label} | method | success | corrected px | delta vs nominal | 95% CI |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        comparison = row.get("vs_nominal")
        if comparison is None:
            delta = "-"
            ci_text = "-"
        else:
            delta = f"{float(comparison['success_rate_difference']):+.4f}"
            ci = comparison["bootstrap_95_ci"]
            ci_text = f"[{float(ci[0]):+.4f}, {float(ci[1]):+.4f}]"
        lines.append(
            f"| {row['checkpoint']} | {row['task']} | {float(row['severity']):g} | "
            f"{row['method']} | {float(row['success_rate']):.4f} | "
            f"{float(row['corrected_pixel_error_mean']):.3f} | {delta} | {ci_text} |"
        )
    lines.extend(
        [
            "",
            "## Success AUC",
            "",
            "| checkpoint | task | method | normalized AUC |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for row in auc_rows:
        lines.append(
            f"| {row['checkpoint']} | {row['task']} | {row['method']} | "
            f"{float(row['success_auc']):.4f} |"
        )
    (output_root / "matrix_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, type=parse_named_value)
    parser.add_argument("--task", action="append", required=True, type=parse_named_value)
    parser.add_argument("--severity", action="append", type=float)
    parser.add_argument("--method", action="append", choices=METHODS)
    parser.add_argument("--perturbation", choices=PERTURBATIONS, default="rotation")
    parser.add_argument("--prototype-bank", required=True)
    parser.add_argument("--prototype-radius-px", type=float, default=32.0)
    parser.add_argument("--prototype-grid-size", type=int, default=9)
    parser.add_argument("--pnp-history-frames", type=int, default=3)
    parser.add_argument("--pnp-max-reprojection-error-px", type=float, default=8.0)
    parser.add_argument("--pnp-max-pose-rotation-deg", type=float, default=30.0)
    parser.add_argument("--pnp-max-pose-translation-cm", type=float, default=20.0)
    parser.add_argument("--camera", default="corner")
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--task-index-offset", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--action-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--env-seed-base", type=int, default=50000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parallel-jobs", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    if args.parallel_jobs < 1:
        parser.error("--parallel-jobs must be positive")
    checkpoints = resolve_policy_specs(args.checkpoint)
    tasks = list(args.task)
    severities = sorted(set(args.severity or [0.0, 5.0, 10.0]))
    methods = list(dict.fromkeys(args.method or METHODS))
    if "nominal" not in methods:
        parser.error("--method must include nominal because paired comparisons use it as baseline")
    if len({checkpoint.label for checkpoint in checkpoints}) != len(checkpoints):
        parser.error("checkpoint labels must be unique")
    if len({label for label, _ in tasks}) != len(tasks):
        parser.error("task labels must be unique")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(
        checkpoints,
        tasks,
        severities,
        methods,
        output_root,
        args.perturbation,
    )
    (output_root / "matrix_manifest.json").write_text(
        json.dumps(
            {
                "checkpoints": [asdict(checkpoint) for checkpoint in checkpoints],
                "tasks": tasks,
                "severities": severities,
                "methods": methods,
                "settings": {
                    key: value
                    for key, value in vars(args).items()
                    if key not in {"checkpoint", "task", "severity", "method"}
                },
                "jobs": [asdict(job) for job in jobs],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with ThreadPoolExecutor(max_workers=args.parallel_jobs) as executor:
        futures = {executor.submit(run_job, job, args): job for job in jobs}
        for future in as_completed(futures):
            job, status = future.result()
            print(
                f"[{status}] checkpoint={job.checkpoint_label} task={job.task_label} "
                f"{job.perturbation}={job.severity:g} method={job.method}",
                flush=True,
            )

    job_lookup = {
        (job.checkpoint_label, job.task_label, job.severity, job.method): job
        for job in jobs
    }
    paired_paths = {}
    for checkpoint in checkpoints:
        for task_label, _ in tasks:
            for severity in severities:
                runs = [
                    (
                        method,
                        job_lookup[(checkpoint.label, task_label, severity, method)].csv_path,
                    )
                    for method in methods
                ]
                paired_paths[(checkpoint.label, task_label, severity)] = compare_runs(
                    runs,
                    "nominal",
                    output_root
                    / checkpoint.label
                    / task_label
                    / severity_token(severity, args.perturbation)
                    / "paired",
                    args.bootstrap_samples,
                    args.seed,
                )
    write_summary(
        checkpoints,
        tasks,
        severities,
        methods,
        output_root,
        paired_paths,
        args.perturbation,
    )
    print(f"summary: {output_root / 'matrix_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
