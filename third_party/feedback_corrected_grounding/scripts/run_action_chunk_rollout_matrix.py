#!/usr/bin/env python3
"""Run a resumable, paired action-chunk rollout matrix.

Each policy is evaluated with native FCPG and nominal/open-loop grounding on
identical MetaWorld task indices, environment seeds, and camera perturbations.
The script then reports both within-policy grounding effects and between-policy
differences on matched episodes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
ROLLOUT_SCRIPT = REPO_ROOT / "scripts" / "evaluate_action_chunk_rollout.py"
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "compare_paired_rollouts.py"


def parse_named_value(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("value must use LABEL=VALUE")
    label, item = value.split("=", 1)
    label = label.strip()
    item = item.strip()
    if not label or not item:
        raise argparse.ArgumentTypeError("value must use non-empty LABEL=VALUE")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise argparse.ArgumentTypeError(
            f"label {label!r} may contain only letters, digits, '.', '_', and '-'"
        )
    return label, item


PERTURBATIONS = ("rotation", "translation")


def severity_token(value: float, perturbation: str = "rotation") -> str:
    if perturbation not in PERTURBATIONS:
        raise ValueError(f"unknown perturbation: {perturbation}")
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    prefix = "rot" if perturbation == "rotation" else "trans"
    return f"{prefix}{text}"


@dataclass(frozen=True)
class PolicySpec:
    label: str
    checkpoint: str
    source_mode: str


def resolve_policy_specs(values: Iterable[tuple[str, str]]) -> list[PolicySpec]:
    import torch

    policies = []
    for label, path in values:
        checkpoint = str(Path(path))
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        source_mode = str(payload.get("args", {}).get("mode", ""))
        if not source_mode:
            raise ValueError(f"checkpoint does not declare args.mode: {checkpoint}")
        policies.append(PolicySpec(label, checkpoint, source_mode))
    return policies


@dataclass(frozen=True)
class RolloutJob:
    policy_label: str
    checkpoint: str
    task_label: str
    task_name: str
    severity_deg: float
    grounding_mode: str
    output_dir: str
    perturbation: str = "rotation"
    source_mode: str = "fcpg"
    reliability_override: str = "native"

    @property
    def model_name(self) -> str:
        if self.grounding_mode == "native":
            model_name = self.source_mode
        else:
            model_name = f"{self.source_mode}_{self.grounding_mode}"
        if (
            self.source_mode == "fcpg"
            and self.grounding_mode == "native"
            and self.reliability_override != "native"
        ):
            model_name = f"{model_name}_rho_{self.reliability_override}"
        return model_name

    @property
    def csv_path(self) -> Path:
        return Path(self.output_dir) / f"{self.model_name}_rollout.csv"

    @property
    def summary_path(self) -> Path:
        return Path(self.output_dir) / f"{self.model_name}_rollout_summary.json"


def build_jobs(
    policies: Iterable[PolicySpec],
    tasks: Iterable[tuple[str, str]],
    severities: Iterable[float],
    output_root: Path,
    perturbation: str = "rotation",
    reliability_override: str = "native",
) -> list[RolloutJob]:
    if perturbation not in PERTURBATIONS:
        raise ValueError(f"unknown perturbation: {perturbation}")
    jobs: list[RolloutJob] = []
    for policy in policies:
        for task_label, task_name in tasks:
            for severity in severities:
                grounding_modes = (
                    ("nominal", "native")
                    if policy.source_mode == "fcpg"
                    else ("native",)
                )
                for grounding_mode in grounding_modes:
                    output_dir = (
                        output_root
                        / policy.label
                        / task_label
                        / severity_token(severity, perturbation)
                        / grounding_mode
                    )
                    jobs.append(
                        RolloutJob(
                            policy_label=policy.label,
                            checkpoint=policy.checkpoint,
                            task_label=task_label,
                            task_name=task_name,
                            severity_deg=float(severity),
                            grounding_mode=grounding_mode,
                            output_dir=str(output_dir),
                            perturbation=perturbation,
                            source_mode=policy.source_mode,
                            reliability_override=reliability_override,
                        )
                    )
    return jobs


def completed_job_matches(job: RolloutJob, episodes: int, horizon: int, action_steps: int) -> bool:
    if not job.csv_path.is_file() or not job.summary_path.is_file():
        return False
    try:
        payload = json.loads(job.summary_path.read_text(encoding="utf-8"))
        summary = payload["summary"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False
    expected = {
        "checkpoint": job.checkpoint,
        "source_mode": job.source_mode,
        "grounding_override": job.grounding_mode,
        "task": job.task_name,
        "episodes": episodes,
        "horizon": horizon,
        "action_steps": action_steps,
    }
    if job.source_mode == "fcpg" and job.grounding_mode == "native":
        expected["reliability_override_request"] = job.reliability_override
    if job.perturbation == "rotation":
        expected.update(rotation_deg_min=job.severity_deg, rotation_deg_max=job.severity_deg)
    else:
        expected.update(
            translation_cm_min=job.severity_deg,
            translation_cm_max=job.severity_deg,
        )
    return all(summary.get(key) == value for key, value in expected.items())


def rollout_command(job: RolloutJob, args: argparse.Namespace) -> list[str]:
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
    ]
    if job.perturbation == "rotation":
        command.extend(
            ("--rotation-deg-min", str(job.severity_deg), "--rotation-deg-max", str(job.severity_deg))
        )
    else:
        command.extend(
            (
                "--translation-cm-min",
                str(job.severity_deg),
                "--translation-cm-max",
                str(job.severity_deg),
            )
        )
    command.extend(
        (
            "--grounding-override",
            job.grounding_mode,
            "--seed",
            str(args.seed),
            "--env-seed-base",
            str(args.env_seed_base),
            "--device",
            args.device,
            "--output-dir",
            job.output_dir,
        )
    )
    if job.source_mode == "fcpg" and job.grounding_mode == "native":
        command.extend(("--reliability-override", job.reliability_override))
    return command


def run_job(job: RolloutJob, args: argparse.Namespace) -> tuple[RolloutJob, str]:
    if not args.force and completed_job_matches(job, args.episodes, args.horizon, args.action_steps):
        return job, "skipped"
    output_dir = Path(job.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = rollout_command(job, args)
    (output_dir / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
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
        raise RuntimeError(
            f"rollout failed with exit code {result.returncode}: {output_dir / 'run.log'}"
        )
    if not completed_job_matches(job, args.episodes, args.horizon, args.action_steps):
        raise RuntimeError(f"rollout output failed provenance validation: {output_dir}")
    return job, "completed"


def compare_runs(
    runs: list[tuple[str, Path]],
    baseline: str,
    output_dir: Path,
    bootstrap_samples: int,
    seed: int,
) -> Path:
    command = [
        sys.executable,
        str(COMPARE_SCRIPT),
        "--baseline",
        baseline,
        "--bootstrap-samples",
        str(bootstrap_samples),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    ]
    for label, path in runs:
        command.extend(("--run", f"{label}={path}"))
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return output_dir / "paired_rollout_summary.json"


def normalized_auc(points: list[tuple[float, float]]) -> float:
    points = sorted(points)
    if len(points) < 2 or points[-1][0] == points[0][0]:
        return sum(value for _, value in points) / len(points)
    area = 0.0
    for (left_x, left_y), (right_x, right_y) in zip(points, points[1:]):
        area += (right_x - left_x) * (left_y + right_y) / 2.0
    return area / (points[-1][0] - points[0][0])


def write_matrix_summary(
    policies: list[PolicySpec],
    tasks: list[tuple[str, str]],
    severities: list[float],
    output_root: Path,
    within_paths: dict[tuple[str, str, float], Path],
    cross_paths: dict[tuple[str, str, float], Path],
    job_lookup: dict[tuple[str, str, float, str], RolloutJob],
    perturbation: str = "rotation",
) -> None:
    rows: list[dict[str, object]] = []
    for policy in policies:
        if policy.source_mode != "fcpg":
            continue
        for task_label, _ in tasks:
            for severity in severities:
                payload = json.loads(
                    within_paths[(policy.label, task_label, severity)].read_text(
                        encoding="utf-8"
                    )
                )
                nominal = payload["models"]["nominal"]
                native = payload["models"]["native"]
                comparison = payload["comparisons"]["native_vs_nominal"]
                rows.append(
                    {
                        "policy": policy.label,
                        "task": task_label,
                        "perturbation": perturbation,
                        "severity_deg": severity,
                        "nominal_success": nominal["success_rate"],
                        "native_success": native["success_rate"],
                        "feedback_delta": comparison["success_rate_difference"],
                        "feedback_ci": comparison["bootstrap_95_ci"],
                        "feedback_mcnemar_p": comparison["mcnemar_exact"]["p_value_two_sided"],
                        "nominal_pixel_error": nominal["pixel_error_mean"],
                        "native_grounding_error": native["corrected_pixel_error_mean"],
                    }
                )

    cross_rows: list[dict[str, object]] = []
    for baseline_index, baseline in enumerate(policies[:-1]):
        for task_label, _ in tasks:
            for severity in severities:
                payload = json.loads(
                    cross_paths[(baseline.label, task_label, severity)].read_text(
                        encoding="utf-8"
                    )
                )
                for policy in policies[baseline_index + 1 :]:
                    comparison = payload["comparisons"][
                        f"{policy.label}_vs_{baseline.label}"
                    ]
                    cross_rows.append(
                        {
                            "task": task_label,
                            "perturbation": perturbation,
                            "severity_deg": severity,
                            "baseline": baseline.label,
                            "candidate": policy.label,
                            "baseline_native_success": payload["models"][baseline.label][
                                "success_rate"
                            ],
                            "candidate_native_success": payload["models"][policy.label][
                                "success_rate"
                            ],
                            "native_success_delta": comparison["success_rate_difference"],
                            "ci": comparison["bootstrap_95_ci"],
                            "mcnemar_p": comparison["mcnemar_exact"]["p_value_two_sided"],
                        }
                    )

    native_rows: list[dict[str, object]] = []
    for policy in policies:
        for task_label, _ in tasks:
            for severity in severities:
                job = job_lookup[(policy.label, task_label, severity, "native")]
                summary = json.loads(job.summary_path.read_text(encoding="utf-8"))["summary"]
                native_rows.append(
                    {
                        "policy": policy.label,
                        "source_mode": policy.source_mode,
                        "task": task_label,
                        "perturbation": perturbation,
                        "severity_deg": severity,
                        "success": summary["success_rate"],
                        "pixel_error": summary["pixel_error_mean"],
                        "grounding_error": summary["grounding_error_mean"],
                    }
                )

    auc_rows = []
    for policy in policies:
        for task_label, _ in tasks:
            points = [
                (float(row["severity_deg"]), float(row["success"]))
                for row in native_rows
                if row["policy"] == policy.label and row["task"] == task_label
            ]
            auc_rows.append(
                {
                    "policy": policy.label,
                    "task": task_label,
                    "native_success_auc": normalized_auc(points),
                }
            )

    payload = {
        "perturbation": perturbation,
        "within_policy": rows,
        "cross_policy": cross_rows,
        "native_policy": native_rows,
        "auc": auc_rows,
    }
    (output_root / "matrix_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    severity_label = "rotation (deg)" if perturbation == "rotation" else "translation (cm)"
    lines = [
        "# Action-Chunk Policy Rollout Matrix",
        "",
        "## Within-policy feedback effect",
        "",
        f"| task | policy | {severity_label} | nominal success | native success | delta | 95% CI | p | nominal px | corrected px |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        ci = row["feedback_ci"]
        lines.append(
            f"| {row['task']} | {row['policy']} | {float(row['severity_deg']):g} | "
            f"{float(row['nominal_success']):.4f} | {float(row['native_success']):.4f} | "
            f"{float(row['feedback_delta']):+.4f} | [{float(ci[0]):+.4f}, {float(ci[1]):+.4f}] | "
            f"{float(row['feedback_mcnemar_p']):.6f} | {float(row['nominal_pixel_error']):.3f} | "
            f"{float(row['native_grounding_error']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Cross-policy native-grounding effect",
            "",
            f"| task | {severity_label} | baseline | candidate | baseline success | candidate success | delta | 95% CI | p |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for row in cross_rows:
        ci = row["ci"]
        lines.append(
            f"| {row['task']} | {float(row['severity_deg']):g} | {row['baseline']} | "
            f"{row['candidate']} | {float(row['baseline_native_success']):.4f} | "
            f"{float(row['candidate_native_success']):.4f} | {float(row['native_success_delta']):+.4f} | "
            f"[{float(ci[0]):+.4f}, {float(ci[1]):+.4f}] | {float(row['mcnemar_p']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Native success AUC",
            "",
            "| task | policy | normalized AUC |",
            "| --- | --- | ---: |",
        ]
    )
    for row in auc_rows:
        lines.append(
            f"| {row['task']} | {row['policy']} | {float(row['native_success_auc']):.4f} |"
        )
    (output_root / "matrix_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", action="append", required=True, type=parse_named_value)
    parser.add_argument("--task", action="append", required=True, type=parse_named_value)
    parser.add_argument("--severity", action="append", type=float)
    parser.add_argument("--perturbation", choices=PERTURBATIONS, default="rotation")
    parser.add_argument("--camera", default="corner")
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--task-index-offset", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--action-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-seed-base", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--reliability-override",
        choices=("native", "ones", "zeros"),
        default="native",
        help="applied only to native FCPG jobs; use ones for residual-only evaluation",
    )
    parser.add_argument("--parallel-jobs", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output-root",
        default="outputs/action_chunk_rollout/lewm_stage1/shared_predictive_seed0_matched",
    )
    args = parser.parse_args()
    if args.parallel_jobs < 1:
        parser.error("--parallel-jobs must be positive")
    if len({label for label, _ in args.policy}) != len(args.policy):
        parser.error("policy labels must be unique")
    if len({label for label, _ in args.task}) != len(args.task):
        parser.error("task labels must be unique")
    policies = resolve_policy_specs(args.policy)
    tasks = list(args.task)
    severities = sorted(set(args.severity or [0.0, 5.0, 10.0, 20.0]))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(
        policies,
        tasks,
        severities,
        output_root,
        args.perturbation,
        args.reliability_override,
    )
    job_lookup = {
        (job.policy_label, job.task_label, job.severity_deg, job.grounding_mode): job
        for job in jobs
    }
    manifest = {
        "policies": [asdict(policy) for policy in policies],
        "tasks": tasks,
        "severities": severities,
        "perturbation": args.perturbation,
        "settings": {
            key: value
            for key, value in vars(args).items()
            if key not in {"policy", "task", "severity"}
        },
        "jobs": [asdict(job) for job in jobs],
    }
    (output_root / "matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    with ThreadPoolExecutor(max_workers=args.parallel_jobs) as executor:
        futures = {executor.submit(run_job, job, args): job for job in jobs}
        for future in as_completed(futures):
            job, status = future.result()
            print(
                f"[{status}] policy={job.policy_label} task={job.task_label} "
                f"{job.perturbation}={job.severity_deg:g} grounding={job.grounding_mode}",
                flush=True,
            )

    within_paths: dict[tuple[str, str, float], Path] = {}
    for policy in policies:
        if policy.source_mode != "fcpg":
            continue
        for task_label, _ in tasks:
            for severity in severities:
                base = (
                    output_root
                    / policy.label
                    / task_label
                    / severity_token(severity, args.perturbation)
                )
                within_paths[(policy.label, task_label, severity)] = compare_runs(
                    [
                        (
                            "nominal",
                            job_lookup[
                                (policy.label, task_label, severity, "nominal")
                            ].csv_path,
                        ),
                        (
                            "native",
                            job_lookup[
                                (policy.label, task_label, severity, "native")
                            ].csv_path,
                        ),
                    ],
                    "nominal",
                    base / "paired",
                    args.bootstrap_samples,
                    args.seed,
                )

    cross_paths: dict[tuple[str, str, float], Path] = {}
    for baseline in policies[:-1]:
        for task_label, _ in tasks:
            for severity in severities:
                runs = [
                    (
                        policy.label,
                        job_lookup[
                            (policy.label, task_label, severity, "native")
                        ].csv_path,
                    )
                    for policy in policies
                ]
                cross_paths[(baseline.label, task_label, severity)] = compare_runs(
                    runs,
                    baseline.label,
                    output_root
                    / "cross_policy"
                    / f"baseline_{baseline.label}"
                    / task_label
                    / severity_token(severity, args.perturbation),
                    args.bootstrap_samples,
                    args.seed,
                )

    write_matrix_summary(
        policies,
        tasks,
        severities,
        output_root,
        within_paths,
        cross_paths,
        job_lookup,
        args.perturbation,
    )
    print(f"summary: {output_root / 'matrix_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
