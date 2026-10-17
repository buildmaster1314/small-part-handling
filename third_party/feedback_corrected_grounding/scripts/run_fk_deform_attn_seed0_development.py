#!/usr/bin/env python3
"""Run the resumable paired seed0 FK-DeformAttn development matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Callable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.fk_deform_attn_baseline import (  # noqa: E402
    load_matched_policy_only_grounding,
    validate_structural_pair,
)
from fcpg.fk_deform_attn_seed0_statistics import (  # noqa: E402
    DEFAULT_CONFIG,
    PROTOCOL,
    ROW_FIELDS,
    ROW_SCHEMA_VERSION,
    VARIANTS,
    EpisodeSpec,
    atomic_json_write,
    expected_episode_specs,
    load_protocol_config,
    pair_artifact_relative_path,
    summarize_rollout_rows,
    validate_pair_rows,
    validate_rollout_rows,
)
from fcpg.mujoco_camera import sample_camera_pose_offset  # noqa: E402
from scripts.evaluate_fk_deform_attn_policy import (  # noqa: E402
    choose_device,
    rollout_once,
)


RolloutFunction = Callable[..., dict[str, object]]
ModelPair = tuple[object, dict[str, object]]


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def variant_execution_order(pair_index: int) -> tuple[str, str]:
    if pair_index < 0:
        raise ValueError("pair_index must be non-negative")
    return VARIANTS if pair_index % 2 == 0 else tuple(reversed(VARIANTS))


def camera_offset_for_spec(
    spec: EpisodeSpec,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sample one fixed direction, scaling only its registered severity."""

    rng = np.random.default_rng(spec.camera_direction_seed)
    severity = spec.condition.severity
    if spec.condition.perturbation == "clean":
        return sample_camera_pose_offset(rng, 0.0, 0.0)
    if spec.condition.perturbation == "rotation":
        return sample_camera_pose_offset(
            rng,
            0.0,
            severity,
            rotation_deg_min=severity,
        )
    if spec.condition.perturbation == "translation":
        return sample_camera_pose_offset(
            rng,
            severity,
            0.0,
            translation_cm_min=severity,
        )
    raise ValueError(f"unknown camera perturbation: {spec.condition.perturbation}")


def build_rollout_row(
    spec: EpisodeSpec,
    variant: str,
    result: Mapping[str, object],
) -> dict[str, object]:
    row: dict[str, object] = {
        "row_schema_version": ROW_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "claim_scope": "seed0-development-controlled-comparison",
        "training_seed": 0,
        "task_key": spec.task.key,
        "task": spec.task.name,
        "task_ordinal": spec.task.ordinal,
        "condition_id": spec.condition.condition_id,
        "perturbation": spec.condition.perturbation,
        "severity": spec.condition.severity,
        "episode": spec.episode,
        "physical_episode_id": spec.physical_episode_id,
        "pair_id": spec.pair_id,
        "variant": variant,
        "task_index": spec.task_index,
        "effective_task_index": spec.effective_task_index,
        "env_seed": spec.env_seed,
        "camera_direction_seed": spec.camera_direction_seed,
    }
    for name in ROW_FIELDS:
        if name in result:
            row[name] = result[name]
    missing = set(ROW_FIELDS) - set(row)
    if missing:
        raise ValueError(f"rollout result is missing row fields: {sorted(missing)}")
    return {name: row[name] for name in ROW_FIELDS}


def pair_artifact_path(
    config: Mapping[str, object],
    output_root: Path,
    spec: EpisodeSpec,
) -> Path:
    return output_root / pair_artifact_relative_path(config, spec)


def _read_pair_artifact(
    config: Mapping[str, object],
    path: Path,
    spec: EpisodeSpec,
    pair_index: int,
) -> tuple[dict[str, object], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"pair artifact must be a JSON object: {path}")
    if payload.get("protocol") != PROTOCOL or payload.get("pair_id") != spec.pair_id:
        raise ValueError(f"pair artifact identity differs: {path}")
    expected_order = list(variant_execution_order(pair_index))
    if payload.get("variant_execution_order") != expected_order:
        raise ValueError(f"pair artifact execution order differs: {path}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"pair artifact rows are invalid: {path}")
    return validate_pair_rows(config, spec, rows)


def load_completed_pairs(
    config: Mapping[str, object],
    output_root: Path,
) -> dict[str, tuple[dict[str, object], ...]]:
    completed: dict[str, tuple[dict[str, object], ...]] = {}
    for pair_index, spec in enumerate(expected_episode_specs(config)):
        path = pair_artifact_path(config, output_root, spec)
        if path.is_file():
            completed[spec.pair_id] = _read_pair_artifact(
                config,
                path,
                spec,
                pair_index,
            )
    validate_rollout_rows(
        config,
        _ordered_rows(config, completed),
        require_complete=False,
    )
    return completed


def _ordered_rows(
    config: Mapping[str, object],
    completed: Mapping[str, Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in expected_episode_specs(config):
        rows.extend(dict(row) for row in completed.get(spec.pair_id, ()))
    return rows


def load_models(
    config: Mapping[str, object],
    device: object,
) -> dict[str, ModelPair]:
    inputs = config["inputs"]
    if not isinstance(inputs, Mapping):
        raise TypeError("config inputs must be an object")
    load_kwargs = {
        "policy_contract_checkpoint": resolve_repo_path(
            str(inputs["policy_contract_checkpoint"])
        ),
        "fcpg_corrector_checkpoint": resolve_repo_path(
            str(inputs["fcpg_corrector_checkpoint"])
        ),
    }
    residual_model, residual_checkpoint = load_matched_policy_only_grounding(
        resolve_repo_path(str(inputs["residual_checkpoint"])),
        device,
        **load_kwargs,
    )
    deform_model, deform_checkpoint = load_matched_policy_only_grounding(
        resolve_repo_path(str(inputs["deform_checkpoint"])),
        device,
        **load_kwargs,
    )
    validate_structural_pair(residual_checkpoint, deform_checkpoint)
    expected_state = str(inputs["checkpoint_state"])
    for variant, checkpoint in (
        ("policy_only_residual", residual_checkpoint),
        ("fk_deform_attn", deform_checkpoint),
    ):
        if checkpoint.get("checkpoint_state") != expected_state:
            raise ValueError(
                f"{variant} checkpoint_state must be {expected_state!r}, "
                f"got {checkpoint.get('checkpoint_state')!r}"
            )
    deform_model.policy.vision_encoder.backbone = (
        residual_model.policy.vision_encoder.backbone
    )
    return {
        "policy_only_residual": (residual_model, residual_checkpoint),
        "fk_deform_attn": (deform_model, deform_checkpoint),
    }


def run_pairs(
    config: Mapping[str, object],
    output_root: Path,
    models: Mapping[str, ModelPair],
    device: object,
    *,
    rollout_fn: RolloutFunction = rollout_once,
    max_new_pairs: int | None = None,
) -> tuple[list[dict[str, object]], int]:
    """Run missing pairs; a failed half-pair leaves no resumable artifact."""

    if max_new_pairs is not None and max_new_pairs < 0:
        raise ValueError("max_new_pairs must be non-negative")
    specs = expected_episode_specs(config)
    completed = load_completed_pairs(config, output_root)
    new_pairs = 0
    evaluation = config["evaluation"]
    if not isinstance(evaluation, Mapping):
        raise TypeError("config evaluation must be an object")

    for pair_index, spec in enumerate(specs):
        if spec.pair_id in completed:
            continue
        if max_new_pairs is not None and new_pairs >= max_new_pairs:
            break
        order = variant_execution_order(pair_index)
        camera_offset = camera_offset_for_spec(spec)
        pair_rows: list[dict[str, object]] = []
        for variant in order:
            if variant not in models:
                raise KeyError(f"missing loaded model for {variant}")
            model, checkpoint = models[variant]
            result = rollout_fn(
                model=model,
                checkpoint=checkpoint,
                task_name=spec.task.name,
                camera_name=str(evaluation["camera"]),
                width=int(evaluation["width"]),
                height=int(evaluation["height"]),
                horizon=int(evaluation["horizon"]),
                action_steps=int(evaluation["action_steps"]),
                task_index=spec.task_index,
                env_seed=spec.env_seed,
                camera_offset=(
                    camera_offset[0].copy(),
                    camera_offset[1].copy(),
                    camera_offset[2],
                ),
                device=device,
                amp=bool(evaluation["amp"]),
            )
            row = build_rollout_row(spec, variant, result)
            pair_rows.append(row)
            print(
                f"[{pair_index + 1:03d}/{len(specs):03d}] "
                f"{spec.task.key}/{spec.condition.condition_id}/"
                f"episode-{spec.episode:03d} {variant} "
                f"success={float(row['success']):.1f}",
                flush=True,
            )

        validated = validate_pair_rows(config, spec, pair_rows)
        candidate = dict(completed)
        candidate[spec.pair_id] = validated
        validate_rollout_rows(
            config,
            _ordered_rows(config, candidate),
            require_complete=False,
        )
        atomic_json_write(
            {
                "protocol": PROTOCOL,
                "pair_id": spec.pair_id,
                "variant_execution_order": list(order),
                "rows": list(validated),
            },
            pair_artifact_path(config, output_root, spec),
        )
        completed[spec.pair_id] = validated
        new_pairs += 1
    return _ordered_rows(config, completed), new_pairs


def _atomic_csv_write(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_final_artifacts(
    config: Mapping[str, object],
    output_root: Path,
    rows: Sequence[Mapping[str, object]],
) -> tuple[Path, Path]:
    validate_rollout_rows(config, rows, require_complete=True)
    report = summarize_rollout_rows(config, rows)
    artifacts = config["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise TypeError("config artifacts must be an object")
    csv_path = output_root / str(artifacts["rollout_csv"])
    summary_path = output_root / str(artifacts["statistics_json"])
    _atomic_csv_write(rows, csv_path)
    atomic_json_write(report, summary_path)
    return csv_path, summary_path


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root")
    parser.add_argument("--device")
    parser.add_argument("--max-new-pairs", type=int)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    config = load_protocol_config(resolve_repo_path(args.config))
    artifacts = config["artifacts"]
    evaluation = config["evaluation"]
    if not isinstance(artifacts, Mapping) or not isinstance(evaluation, Mapping):
        raise TypeError("config artifacts and evaluation must be objects")
    output_root = resolve_repo_path(
        args.output_root or str(artifacts["output_root"])
    )
    specs = expected_episode_specs(config)
    completed = load_completed_pairs(config, output_root)
    pending = len(specs) - len(completed)
    print(
        f"resume: {len(completed)}/{len(specs)} pairs complete; {pending} pending",
        flush=True,
    )

    if pending > 0 and args.max_new_pairs != 0:
        os.environ.setdefault("MUJOCO_GL", "egl")
        device = choose_device(args.device or str(evaluation["device"]))
        models = load_models(config, device)
    else:
        device = choose_device("cpu")
        models = {}
    rows, new_pairs = run_pairs(
        config,
        output_root,
        models,
        device,
        max_new_pairs=args.max_new_pairs,
    )
    if len(rows) == len(specs) * len(VARIANTS):
        csv_path, summary_path = write_final_artifacts(config, output_root, rows)
        print(f"rows: {csv_path.resolve()}")
        print(f"statistics: {summary_path.resolve()}")
    else:
        print(
            f"partial: committed {new_pairs} new pairs; "
            f"{len(rows) // len(VARIANTS)}/{len(specs)} complete",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
