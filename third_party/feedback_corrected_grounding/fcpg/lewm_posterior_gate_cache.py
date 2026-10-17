"""Strict loading and training-row assembly for posterior Gate caches."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from fcpg.lewm_posterior_gate_data import DEPLOYMENT_TENSOR_KEYS
from fcpg.lewm_posterior_gate_training import (
    PosteriorGateLossMasks,
    PosteriorGateTrainingRows,
)
from fcpg.lewm_provenance import sha256_file


CACHE_TYPE = "fcpg_lewm_posterior_gate_cache_v1"
DEPLOYMENT_FILENAME = "deployment_inputs.npz"
LABEL_FILENAME = "offline_labels.npz"
EVALUATION_FILENAME = "evaluation_metadata.npz"
METADATA_FILENAME = "metadata.json"
EVENT_NAMES = ("o0", "o1", "o2", "r0", "r1")
DEPLOYMENT_KEYS = (
    *DEPLOYMENT_TENSOR_KEYS,
    "teacher_forced_factual_prior",
    "measurement_frame_embedding",
)
LABEL_KEYS = (
    "canonical_current",
    "canonical_measurement_history",
    "canonical_target",
    "canonical_q_corrected_history",
    "canonical_q_corrected",
    "canonical_valid_history",
    "canonical_valid_current",
)
FORBIDDEN_DEPLOYMENT_SUBSTRINGS = (
    "q_true",
    "canonical",
    "measurement_valid",
    "dropout",
    "recovery",
    "event",
    "task",
    "episode",
    "schedule",
)


@dataclass(frozen=True)
class PosteriorGateCache:
    path: Path
    metadata: Mapping[str, Any]
    deployment: Mapping[str, np.ndarray]
    labels: Mapping[str, np.ndarray]
    evaluation: Mapping[str, np.ndarray]
    file_hashes: Mapping[str, str]

    @property
    def sample_count(self) -> int:
        return int(self.deployment["measurement_current"].shape[0])

    @property
    def image_width(self) -> int:
        return int(self.metadata["image_width"])

    @property
    def image_height(self) -> int:
        return int(self.metadata["image_height"])


def _load_npz(path: Path, expected_keys: Sequence[str] | None) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as raw:
        if expected_keys is not None and set(raw.files) != set(expected_keys):
            raise ValueError(
                f"{path.name} key set changed: {sorted(raw.files)}"
            )
        return {name: np.array(raw[name], copy=True) for name in raw.files}


def _artifact_hash(metadata: Mapping[str, Any], filename: str) -> str:
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("posterior cache metadata is missing artifacts")
    record = artifacts.get(filename)
    if not isinstance(record, Mapping) or not isinstance(record.get("sha256"), str):
        raise ValueError(f"posterior cache metadata is missing {filename} hash")
    return str(record["sha256"])


def load_posterior_gate_cache(path: str | Path) -> PosteriorGateCache:
    """Load one immutable cache and verify every partition before use."""

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"posterior Gate cache is not a directory: {root}")
    paths = {
        "metadata": root / METADATA_FILENAME,
        "deployment": root / DEPLOYMENT_FILENAME,
        "labels": root / LABEL_FILENAME,
        "evaluation": root / EVALUATION_FILENAME,
    }
    for label, file_path in paths.items():
        if not file_path.is_file():
            raise FileNotFoundError(f"posterior Gate {label} file is missing: {file_path}")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError("posterior cache metadata must be a JSON object")
    if metadata.get("cache_type") != CACHE_TYPE or metadata.get("schema_version") != 1:
        raise ValueError("posterior cache type/schema mismatch")
    if metadata.get("contract", {}).get("checkpoint_step") != 2000:
        raise ValueError("posterior cache must bind the frozen step-2000 LeWM")
    if metadata.get("contract", {}).get("history_size") != 3:
        raise ValueError("posterior cache history_size must be 3")
    if metadata.get("contract", {}).get("frame_skip") != 5:
        raise ValueError("posterior cache frame_skip must be 5")
    if metadata.get("contract", {}).get("action_dim") != 20:
        raise ValueError("posterior cache action_dim must be 20")

    for filename, label in (
        (DEPLOYMENT_FILENAME, "deployment"),
        (LABEL_FILENAME, "labels"),
        (EVALUATION_FILENAME, "evaluation"),
    ):
        actual = sha256_file(paths[label])
        if actual != _artifact_hash(metadata, filename):
            raise ValueError(f"posterior cache {filename} hash mismatch")

    deployment = _load_npz(paths["deployment"], DEPLOYMENT_KEYS)
    labels = _load_npz(paths["labels"], LABEL_KEYS)
    evaluation = _load_npz(paths["evaluation"], None)
    for key in deployment:
        if any(token in key.lower() for token in FORBIDDEN_DEPLOYMENT_SUBSTRINGS):
            raise ValueError(f"privileged field crossed deployment boundary: {key}")
    count = int(metadata.get("sample_count", -1))
    if count <= 0:
        raise ValueError("posterior cache sample_count must be positive")
    for key, value in deployment.items():
        if key == "measurement_frame_embedding":
            continue
        if value.shape[0] != count:
            raise ValueError(f"deployment field {key!r} has the wrong row count")
    for key, value in labels.items():
        if value.shape[0] != count:
            raise ValueError(f"label field {key!r} has the wrong row count")
    for key in ("event", "task_name", "episode_id", "phase_index"):
        if key not in evaluation or evaluation[key].shape[0] != count:
            raise ValueError(f"evaluation identity field {key!r} is invalid")
    events = set(np.asarray(evaluation["event"]).astype(str).tolist())
    if events != set(EVENT_NAMES):
        raise ValueError(f"posterior cache event set changed: {sorted(events)}")
    return PosteriorGateCache(
        path=root,
        metadata=metadata,
        deployment=deployment,
        labels=labels,
        evaluation=evaluation,
        file_hashes={label: sha256_file(file_path) for label, file_path in paths.items()},
    )


def _inside_image(q: np.ndarray, *, width: int, height: int) -> np.ndarray:
    return (
        np.isfinite(q).all(axis=-1)
        & (q[..., 0] >= 0.0)
        & (q[..., 0] <= float(width - 1))
        & (q[..., 1] >= 0.0)
        & (q[..., 1] <= float(height - 1))
    )


def training_rows_from_cache(cache: PosteriorGateCache) -> PosteriorGateTrainingRows:
    """Map one cache into teacher-forced train rows with offline-only masks."""

    deployment = cache.deployment
    labels = cache.labels
    evaluation = cache.evaluation
    measurement = np.asarray(deployment["measurement_current"], dtype=np.float32)
    history = np.asarray(deployment["measurement_history"], dtype=np.float32)
    q_history = np.asarray(deployment["q_corrected_history"], dtype=np.float32)
    valid_history = np.asarray(deployment["valid_nominal_history"], dtype=np.bool_)
    valid_history = valid_history & _inside_image(
        q_history, width=cache.image_width, height=cache.image_height
    )
    events = np.asarray(evaluation["event"]).astype(str)
    dropout = np.asarray(evaluation["dropout_active"], dtype=np.bool_)
    recovery = np.asarray(evaluation["recovery_active"], dtype=np.bool_)
    direct = dropout | recovery
    if not np.all(direct.sum(axis=1) == 1):
        raise ValueError("every posterior Gate row must identify exactly one direct anchor")
    indirect = ~direct
    expanded_event = events[:, None]
    loss_masks = PosteriorGateLossMasks(
        primary_direct_o0=torch.from_numpy(direct & (expanded_event == "o0")),
        primary_direct_o1=torch.from_numpy(direct & (expanded_event == "o1")),
        primary_direct_o2=torch.from_numpy(direct & (expanded_event == "o2")),
        safety_recovery=torch.from_numpy(direct & np.isin(expanded_event, ("r0", "r1"))),
        safety_indirect=torch.from_numpy(indirect),
    )
    return PosteriorGateTrainingRows(
        z_measurement=torch.from_numpy(measurement),
        z_prior=torch.from_numpy(
            np.asarray(deployment["teacher_forced_factual_prior"], dtype=np.float32)
        ),
        z_hold_last=torch.from_numpy(history[:, -1]),
        canonical_target=torch.from_numpy(
            np.asarray(labels["canonical_target"], dtype=np.float32)
        ),
        q_nominal=torch.from_numpy(
            np.asarray(deployment["q_nominal_current"], dtype=np.float32)
        ),
        q_corrected=torch.from_numpy(
            np.asarray(deployment["q_corrected_current"], dtype=np.float32)
        ),
        valid_nominal=torch.from_numpy(
            np.asarray(deployment["valid_nominal_current"], dtype=np.bool_)
        ),
        validity_history=torch.from_numpy(np.moveaxis(valid_history, 1, -1)),
        last_causal_action_chunk=torch.from_numpy(
            np.asarray(deployment["action_chunks"][:, -1], dtype=np.float32)
        ),
        loss_masks=loss_masks,
        source_hashes={
            f"{cache.path.name}_{name}": digest
            for name, digest in cache.file_hashes.items()
        },
        image_width=cache.image_width,
        image_height=cache.image_height,
    )


def concatenate_training_rows(
    caches: Sequence[PosteriorGateCache],
) -> tuple[PosteriorGateTrainingRows, tuple[tuple[int, int], ...]]:
    """Concatenate physically disjoint caches and return half-open row ranges."""

    if not caches:
        raise ValueError("at least one posterior Gate cache is required")
    rows = [training_rows_from_cache(cache) for cache in caches]
    if len({(row.image_width, row.image_height) for row in rows}) != 1:
        raise ValueError("posterior Gate caches use different image sizes")
    ranges: list[tuple[int, int]] = []
    offset = 0
    for row in rows:
        ranges.append((offset, offset + row.sample_count))
        offset += row.sample_count
    masks = PosteriorGateLossMasks(
        **{
            name: torch.cat([getattr(row.loss_masks, name) for row in rows], dim=0)
            for name in PosteriorGateLossMasks.__dataclass_fields__
        }
    )
    hashes: dict[str, str] = {}
    for row in rows:
        overlap = set(hashes).intersection(row.source_hashes)
        if overlap:
            raise ValueError(f"posterior cache source hash names collide: {sorted(overlap)}")
        hashes.update(row.source_hashes)
    first = rows[0]
    return (
        PosteriorGateTrainingRows(
            z_measurement=torch.cat([row.z_measurement for row in rows]),
            z_prior=torch.cat([row.z_prior for row in rows]),
            z_hold_last=torch.cat([row.z_hold_last for row in rows]),
            canonical_target=torch.cat([row.canonical_target for row in rows]),
            q_nominal=torch.cat([row.q_nominal for row in rows]),
            q_corrected=torch.cat([row.q_corrected for row in rows]),
            valid_nominal=torch.cat([row.valid_nominal for row in rows]),
            validity_history=torch.cat([row.validity_history for row in rows]),
            last_causal_action_chunk=torch.cat(
                [row.last_causal_action_chunk for row in rows]
            ),
            loss_masks=masks,
            source_hashes=hashes,
            image_width=first.image_width,
            image_height=first.image_height,
        ),
        tuple(ranges),
    )


__all__ = [
    "PosteriorGateCache",
    "concatenate_training_rows",
    "load_posterior_gate_cache",
    "training_rows_from_cache",
]
