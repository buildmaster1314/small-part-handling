"""Shared report contracts for PixelObserver v2 evaluation and comparison."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

from fcpg.lewm_pixel_observer_v2_runtime import canonical_sha256


PRIMARY_CONTROL_METRIC_FAMILY = "pre_measurement_prior_qtrue"
EVALUATION_IMPLEMENTATION_RELATIVE_PATHS = (
    "scripts/evaluate_lewm_pixel_observer_v2.py", "fcpg/__init__.py",
    "fcpg/modules.py", "fcpg/lewm_provenance.py",
    "fcpg/lewm_sequence_data.py", "fcpg/lewm_grounded_model.py",
    "fcpg/lewm_runtime.py", "fcpg/lewm_pixel_observer.py",
    "fcpg/lewm_pixel_observer_baselines.py", "fcpg/lewm_pixel_observer_v2.py",
    "fcpg/lewm_pixel_observer_v2_data.py", "fcpg/lewm_pixel_observer_v2_eval.py",
    "fcpg/lewm_pixel_observer_v2_report.py", "fcpg/lewm_pixel_observer_v2_runtime.py",
    "fcpg/lewm_pixel_observer_v2_training.py", "third_party/lewm/module.py",
)


def matched_training_protocol(
    resolved: Mapping[str, Any],
    *,
    expected_use_predicted_latent: bool | None = None,
) -> dict[str, Any]:
    protocol = copy.deepcopy(dict(resolved))
    if "output_dir" not in protocol:
        raise ValueError("checkpoint resolved_config lacks output_dir")
    protocol.pop("output_dir")
    adapter = protocol.get("adapter")
    if not isinstance(adapter, Mapping):
        raise ValueError("checkpoint resolved_config lacks adapter config")
    adapter = copy.deepcopy(dict(adapter))
    use_predicted_latent = adapter.pop("use_predicted_latent", None)
    if not isinstance(use_predicted_latent, bool):
        raise ValueError("resolved_config lacks boolean use_predicted_latent")
    if expected_use_predicted_latent is not None and (
        use_predicted_latent is not expected_use_predicted_latent
    ):
        raise ValueError("resolved_config has the wrong use_predicted_latent")
    protocol["adapter"] = adapter
    return protocol


def condition_primary_metric(*, dropout_length: int, hidden_step_length: int) -> str:
    return (
        "prior_qtrue_dropout_epe_px"
        if dropout_length > 0 or hidden_step_length > 0
        else "prior_qtrue_epe_px"
    )


def episode_coverage(
    identities: list[Mapping[str, Any]],
) -> dict[str, Any]:
    selected = [
        {
            "source": str(item["source"]),
            "task": str(item["task"]),
            "episode": str(item["episode"]),
        }
        for item in identities
    ]
    selected.sort(key=lambda item: tuple(item.values()))
    keys = [(item["source"], item["task"], item["episode"]) for item in selected]
    if not selected or len(keys) != len(set(keys)):
        raise ValueError("selected episode identities must be nonempty and unique")
    counts: dict[str, dict[str, int]] = {}
    for item in selected:
        task_counts = counts.setdefault(item["source"], {})
        task = item["task"]
        task_counts[task] = task_counts.get(task, 0) + 1
    counts = {
        source: dict(sorted(task_counts.items()))
        for source, task_counts in sorted(counts.items())
    }
    return {
        "selected_episode_count": len(selected),
        "selected_episode_identities": selected,
        "selected_identity_sha256": canonical_sha256(selected),
        "selected_counts_by_source_task": counts,
    }


def validate_episode_rows(
    rows: list[Mapping[str, Any]],
    *,
    coverage: Mapping[str, Any],
    primary_metric: str,
) -> None:
    actual = episode_coverage(rows)
    if actual != dict(coverage):
        raise ValueError("episode rows do not match selected episode coverage")
    for row in rows:
        modes = row.get("modes")
        full = modes.get("full_writeback") if isinstance(modes, Mapping) else None
        value = full.get(primary_metric) if isinstance(full, Mapping) else None
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            identity = ":".join(
                str(row.get(name)) for name in ("source", "task", "episode")
            )
            raise ValueError(
                f"primary metric {primary_metric} must be finite for {identity}"
            )
