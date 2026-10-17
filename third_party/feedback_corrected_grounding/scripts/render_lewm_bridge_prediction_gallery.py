#!/usr/bin/env python3
"""Render a matched held-out Tiny versus FCPG-bridge LeWM gallery.

The world models predict anchor latents, not pixels.  RGB frames in the output
are dataset observations only; model predictions are rendered as explicitly
labelled latent-vector heatmaps and per-anchor latent MSE diagnostics.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import html
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import default_collate


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.lewm_anchor_persistent import AnchorPersistentLeWM  # noqa: E402
from fcpg.lewm_provenance import sha256_file  # noqa: E402
from fcpg.lewm_runtime import (  # noqa: E402
    choose_device,
    load_lewm_grounded_checkpoint,
    move_batch_to_device,
    split_manifest_episode_windows,
)
from fcpg.lewm_sequence_data import LeWMSequenceDataset  # noqa: E402


DEFAULT_TINY_CHECKPOINT = (
    REPO_ROOT
    / "outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k/"
    "lewm_grounded_latest.pt"
)
DEFAULT_BRIDGE_CHECKPOINT = (
    REPO_ROOT
    / "outputs/lewm_grounded/"
    "anchor_persistent_fcpg_vitb_bridge_sig002_seed9216_2k_formal2/"
    "lewm_grounded_latest.pt"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/lewm_bridge_prediction_gallery"
ANCHOR_COLORS = ((214, 62, 52), (36, 145, 95), (38, 105, 190))
METHOD_COLORS = {
    "factual": (28, 132, 87),
    "copy": (105, 115, 124),
    "shuffled": (194, 67, 55),
}
MODEL_LABELS = {
    "tiny": "LeWM ViT-Tiny",
    "bridge": "FCPG trained dense encoder + warm-start bridge",
}
DISPLAY_TRANSFORM = {
    "scope": "all_four_sources_are_standard_metaworld_renderings",
    "rgb": "vertical_flip_exactly_once",
    "anchor_pixels": "x_display=x_source; y_display=height-1-y_source",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny-checkpoint", type=Path, default=DEFAULT_TINY_CHECKPOINT)
    parser.add_argument(
        "--bridge-checkpoint", type=Path, default=DEFAULT_BRIDGE_CHECKPOINT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--expected-step", type=int, default=2_000)
    parser.add_argument("--samples-per-source", type=int, default=3)
    parser.add_argument("--source", action="append", dest="sources")
    parser.add_argument("--seed", type=int, default=17_418)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use bfloat16 autocast on CUDA (ignored on CPU)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace index.html, manifest.json, and same-named panel images",
    )
    return parser


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _checkpoint_metadata(
    path: Path,
    *,
    expected_step: int,
    role: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{role} checkpoint must contain a dictionary")
    if int(checkpoint.get("step", -1)) != expected_step:
        raise ValueError(
            f"{role} checkpoint must be fixed step {expected_step}, "
            f"got {checkpoint.get('step')!r}: {path}"
        )
    budget = _mapping(checkpoint.get("training_budget"), f"{role} training_budget")
    if (
        int(budget.get("target_optimizer_steps", -1)) != expected_step
        or int(budget.get("checkpoint_optimizer_step", -1)) != expected_step
        or budget.get("run_completed") is not True
    ):
        raise ValueError(f"{role} checkpoint is not a completed fixed-budget endpoint")
    resolved = dict(
        _mapping(checkpoint.get("resolved_config"), f"{role} resolved_config")
    )
    contract = _mapping(resolved.get("contract"), f"{role} contract")
    if contract.get("checkpoint_selection") != "fixed_budget_latest":
        raise ValueError(f"{role} checkpoint is not fixed_budget_latest")
    model_config = _mapping(checkpoint.get("model_config"), f"{role} model_config")
    if model_config.get("variant") != "anchor_persistent":
        raise ValueError(f"{role} checkpoint is not Anchor-Persistent LeWM")
    vision = _mapping(model_config.get("vision"), f"{role} vision config")
    source = vision.get("source")
    if role == "tiny" and source == "fcpg_vitb":
        raise ValueError("Tiny checkpoint unexpectedly uses the FCPG ViT-B bridge")
    if role == "bridge" and source != "fcpg_vitb":
        raise ValueError("bridge checkpoint does not use source='fcpg_vitb'")
    # Drop the large state before the second metadata checkpoint is inspected.
    checkpoint.pop("model_state", None)
    checkpoint.pop("optimizer_state", None)
    checkpoint.pop("scheduler_state", None)
    checkpoint.pop("rng_state", None)
    return checkpoint, resolved


def _validate_matched_contract(
    tiny_path: Path,
    bridge_path: Path,
    tiny_checkpoint: Mapping[str, Any],
    bridge_checkpoint: Mapping[str, Any],
    tiny_resolved: Mapping[str, Any],
    bridge_resolved: Mapping[str, Any],
) -> None:
    tiny_data = _mapping(tiny_checkpoint.get("data_config"), "Tiny data_config")
    bridge_data = _mapping(bridge_checkpoint.get("data_config"), "bridge data_config")
    if tiny_data != bridge_data:
        raise ValueError("Tiny and bridge checkpoints do not have identical data_config")
    if tiny_checkpoint.get("normalization") != bridge_checkpoint.get("normalization"):
        raise ValueError("Tiny and bridge checkpoints do not share input normalization")

    tiny_model = _mapping(tiny_checkpoint.get("model_config"), "Tiny model_config")
    bridge_model = _mapping(
        bridge_checkpoint.get("model_config"), "bridge model_config"
    )
    for section in ("frame", "world_model"):
        if tiny_model.get(section) != bridge_model.get(section):
            raise ValueError(f"Tiny and bridge {section} contracts differ")

    trainer = _mapping(bridge_resolved.get("trainer"), "bridge trainer config")
    initialized_from = trainer.get("initialize_from")
    if not initialized_from or _resolve(str(initialized_from)) != tiny_path:
        raise ValueError(
            "bridge checkpoint was not configured to initialize from the selected Tiny "
            "checkpoint"
        )
    if bridge_path == tiny_path:
        raise ValueError("Tiny and bridge checkpoint paths must differ")
    tiny_data_config = _mapping(tiny_resolved.get("data"), "Tiny resolved data")
    bridge_data_config = _mapping(bridge_resolved.get("data"), "bridge resolved data")
    if tiny_data_config != bridge_data_config:
        raise ValueError("Tiny and bridge resolved data configurations differ")


def _source_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _select_source_samples(
    source: Mapping[str, Any],
    *,
    source_index: int,
    history_size: int,
    frame_skip: int,
    manifest: Mapping[str, Any],
    samples_per_source: int,
    seed: int,
) -> list[dict[str, Any]]:
    name = str(source["name"])
    dataset = LeWMSequenceDataset(
        data_path=_resolve(str(source["path"])),
        history_size=history_size,
        rollout_steps=1,
        frame_skip=frame_skip,
        source_index=source_index,
        source_name=name,
        correction_cache=_resolve(str(source["correction_cache"])),
        require_pre_action_alignment=True,
    )
    split = split_manifest_episode_windows(dataset, name, manifest)
    by_episode: dict[int, list[int]] = {}
    for window_index in split.val_windows:
        episode_index, _ = dataset.windows[int(window_index)]
        by_episode.setdefault(int(episode_index), []).append(int(window_index))
    if len(by_episode) < samples_per_source:
        raise ValueError(
            f"source {name!r} has only {len(by_episode)} held-out episodes, "
            f"cannot select {samples_per_source}"
        )

    rng = np.random.default_rng(_source_seed(seed, name))
    episode_indices = np.asarray(sorted(by_episode), dtype=np.int64)
    selected_episodes = rng.choice(
        episode_indices, size=samples_per_source, replace=False
    ).tolist()
    samples: list[dict[str, Any]] = []
    window_indices: list[int] = []
    for episode_index in selected_episodes:
        candidates = by_episode[int(episode_index)]
        window_indices.append(candidates[int(rng.integers(0, len(candidates)))])
    for window_index in window_indices:
        sample = dict(dataset[window_index])
        sample["_window_index"] = int(window_index)
        samples.append(sample)

    # Rotate selected held-out episodes within this task.  Every donor is from a
    # different episode, matching the repository's strict shuffle diagnostic.
    for index, sample in enumerate(samples):
        donor = samples[(index + 1) % len(samples)]
        if str(sample["episode_id"]) == str(donor["episode_id"]):
            raise RuntimeError("failed to construct a different-episode action donor")
        sample["shuffled_action"] = donor["action"].clone()
        sample["shuffled_episode_id"] = str(donor["episode_id"])
    return samples


def _collect_samples(
    resolved: Mapping[str, Any],
    *,
    requested_sources: Sequence[str] | None,
    samples_per_source: int,
    seed: int,
) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
    data = _mapping(resolved.get("data"), "resolved data config")
    sources = data.get("sources")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        raise TypeError("resolved data.sources must be a sequence")
    entries = [dict(_mapping(value, "data source")) for value in sources]
    available = [str(entry["name"]) for entry in entries]
    selected_names = list(requested_sources) if requested_sources else available
    unknown = sorted(set(selected_names).difference(available))
    if unknown:
        raise ValueError(f"unknown source(s) {unknown}; available: {available}")
    if len(set(selected_names)) != len(selected_names):
        raise ValueError("--source values must not repeat")

    manifest_value = data.get("episode_manifest")
    if not manifest_value:
        raise ValueError("checkpoint has no frozen episode_manifest")
    manifest_path = _resolve(str(manifest_value))
    manifest = _load_json(manifest_path)
    if manifest.get("protocol") != "lewm_gate_l1_camera_aware_episode_split":
        raise ValueError("unexpected held-out episode manifest protocol")

    all_samples: list[dict[str, Any]] = []
    for source_index, source in enumerate(entries):
        name = str(source["name"])
        if name not in selected_names:
            continue
        print(f"selecting held-out samples: {name}", flush=True)
        all_samples.extend(
            _select_source_samples(
                source,
                source_index=source_index,
                history_size=int(data["history_size"]),
                frame_skip=int(data["frame_skip"]),
                manifest=manifest,
                samples_per_source=samples_per_source,
                seed=seed,
            )
        )
        gc.collect()
    return all_samples, manifest_path, manifest


def _sample_key(sample: Mapping[str, Any]) -> str:
    frames = sample["frame_id"]
    if not torch.is_tensor(frames):
        raise TypeError("sample frame_id must be a tensor")
    return (
        f"{sample['source']}:{sample['episode_id']}:"
        f"{int(frames.reshape(-1)[-1])}"
    )


def _latent_result(vectors: Mapping[str, np.ndarray]) -> dict[str, Any]:
    target = np.asarray(vectors["target"], dtype=np.float32)
    result_vectors = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in vectors.items()
    }
    errors = {
        name: np.mean((result_vectors[name] - target) ** 2, axis=-1)
        for name in ("factual", "copy", "shuffled")
    }
    action_response = np.mean(
        (result_vectors["factual"] - result_vectors["shuffled"]) ** 2,
        axis=-1,
    )
    copy_mse = float(np.mean(errors["copy"]))
    action_response_mse = float(np.mean(action_response))
    return {
        "vectors": result_vectors,
        "per_anchor_mse": errors,
        "mse": {name: float(np.mean(value)) for name, value in errors.items()},
        "per_anchor_action_response_mse": action_response,
        "action_response_mse": action_response_mse,
        "action_response_over_copy_mse": (
            action_response_mse / copy_mse if copy_mse > 0.0 else math.nan
        ),
    }


@torch.inference_mode()
def _run_checkpoint(
    path: Path,
    samples: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> dict[str, dict[str, Any]]:
    model, _, checkpoint = load_lewm_grounded_checkpoint(path, device=device)
    if not isinstance(model, AnchorPersistentLeWM):
        raise TypeError(f"checkpoint is not Anchor-Persistent LeWM: {path}")
    model.eval()
    results: dict[str, dict[str, Any]] = {}
    for start in range(0, len(samples), batch_size):
        host_samples = samples[start : start + batch_size]
        batch = move_batch_to_device(default_collate(host_samples), device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=bool(amp and device.type == "cuda"),
        ):
            _, embedding = model.encode_frames(batch)
            history = model.history_size
            context = embedding[:, :history]
            factual_all = model.predict(context, batch["action"][:, :history].float())
            shuffled_all = model.predict(
                context, batch["shuffled_action"][:, :history].float()
            )
            vectors = {
                "target": embedding[:, history],
                "factual": factual_all[:, history - 1],
                "copy": embedding[:, history - 1],
                "shuffled": shuffled_all[:, history - 1],
            }
            sequence_vectors = {
                "target": embedding[:, 1 : history + 1],
                "factual": factual_all[:, :history],
                "copy": embedding[:, :history],
                "shuffled": shuffled_all[:, :history],
            }
        cpu_vectors = {
            name: value.detach().float().cpu().numpy()
            for name, value in vectors.items()
        }
        cpu_sequence_vectors = {
            name: value.detach().float().cpu().numpy()
            for name, value in sequence_vectors.items()
        }
        for row, sample in enumerate(host_samples):
            result_vectors = {
                name: values[row] for name, values in cpu_vectors.items()
            }
            result = _latent_result(result_vectors)
            result["sequence"] = [
                _latent_result(
                    {
                        name: values[row, step]
                        for name, values in cpu_sequence_vectors.items()
                    }
                )
                for step in range(history)
            ]
            results[_sample_key(sample)] = result
        del batch, embedding, factual_all, shuffled_all, sequence_vectors
    del model, checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    try:
        return ImageFont.truetype(names, size=size)
    except OSError:
        return ImageFont.load_default()


def _draw_anchor_overlay(
    image: Image.Image,
    points: np.ndarray,
    valid: np.ndarray,
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    source_width, source_height = image.size
    for anchor, (point, is_valid) in enumerate(zip(points, valid, strict=True)):
        if not bool(is_valid):
            continue
        x = float(point[0]) * output.width / source_width
        y = float(point[1]) * output.height / source_height
        color = ANCHOR_COLORS[anchor % len(ANCHOR_COLORS)]
        radius = max(4, output.width // 55)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=(255, 255, 255),
            width=5,
        )
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=color,
            width=3,
        )
        draw.text((x + radius + 3, y - radius), f"A{anchor}", fill=color, font=_font(14, bold=True))
    return output


def _metaworld_display_frame(
    rgb: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the repository's standard MetaWorld human-display transform once."""

    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"MetaWorld RGB must have shape [H, W, 3], got {image.shape}")
    display_rgb = np.flip(image, axis=0).copy()
    display_points = np.asarray(points, dtype=np.float32).copy()
    if display_points.ndim != 2 or display_points.shape[-1] != 2:
        raise ValueError(
            f"anchor pixels must have shape [K, 2], got {display_points.shape}"
        )
    display_points[:, 1] = float(image.shape[0] - 1) - display_points[:, 1]
    return display_rgb, display_points


def _latent_heatmap(vector: np.ndarray, scale: float, width: int, height: int) -> Image.Image:
    value = np.asarray(vector, dtype=np.float32)
    normalized = np.clip(value / max(scale, 1e-8), -1.0, 1.0)
    magnitude = np.abs(normalized)[..., None]
    positive = np.asarray((193, 64, 50), dtype=np.float32)
    negative = np.asarray((39, 103, 176), dtype=np.float32)
    white = np.asarray((246, 247, 247), dtype=np.float32)
    endpoints = np.where((normalized >= 0)[..., None], positive, negative)
    rgb = white + magnitude * (endpoints - white)
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    return image.resize((width, height), resample=Image.Resampling.NEAREST)


def _draw_model_section(
    canvas: Image.Image,
    y: int,
    role: str,
    result: Mapping[str, Any],
    *,
    fixed_heat_scale: float | None = None,
) -> int:
    draw = ImageDraw.Draw(canvas)
    section_height = 330
    draw.rectangle((24, y, canvas.width - 24, y + section_height), fill=(255, 255, 255))
    draw.text((48, y + 18), MODEL_LABELS[role], fill=(25, 34, 40), font=_font(24, bold=True))
    draw.text(
        (48, y + 51),
        "LATENT DIAGNOSTIC - vectors and errors below are not RGB reconstructions",
        fill=(151, 52, 43),
        font=_font(15, bold=True),
    )

    vectors = _mapping(result["vectors"], "latent vectors")
    stacked = np.stack([np.asarray(vectors[name]) for name in ("target", "factual", "copy", "shuffled")])
    scale = (
        float(fixed_heat_scale)
        if fixed_heat_scale is not None
        else float(np.percentile(np.abs(stacked), 98.0))
    )
    scale = scale if math.isfinite(scale) and scale > 0.0 else 1.0
    heat_x = 172
    heat_width = 500
    heat_height = 43
    labels = (
        ("target", "encoder target from actual next RGB"),
        ("factual", "factual-action prediction"),
        ("copy", "copy / hold-last"),
        ("shuffled", "shuffled-action prediction"),
    )
    for row, (name, label) in enumerate(labels):
        row_y = y + 88 + row * 54
        draw.text((48, row_y + 12), label, fill=(51, 61, 68), font=_font(13))
        heat = _latent_heatmap(np.asarray(vectors[name]), scale, heat_width, heat_height)
        canvas.paste(heat, (heat_x, row_y))
        for anchor in range(1, np.asarray(vectors[name]).shape[0]):
            boundary = row_y + round(anchor * heat_height / np.asarray(vectors[name]).shape[0])
            draw.line((heat_x, boundary, heat_x + heat_width, boundary), fill=(255, 255, 255), width=1)
    draw.text(
        (heat_x, y + 306),
        (
            f"fixed heat scale across this model/sample GIF: +/-{scale:.3g}"
            if fixed_heat_scale is not None
            else f"shared heat scale within this model/sample: +/-{scale:.3g}"
        ),
        fill=(103, 112, 119),
        font=_font(11),
    )

    errors = _mapping(result["per_anchor_mse"], "per-anchor errors")
    max_error = max(float(np.max(np.asarray(errors[name]))) for name in METHOD_COLORS)
    max_error = max(max_error, 1e-12)
    table_x = 720
    draw.text((table_x, y + 91), "Per-anchor future latent MSE", fill=(31, 41, 47), font=_font(16, bold=True))
    for anchor in range(len(np.asarray(errors["factual"]))):
        row_y = y + 124 + anchor * 60
        color = ANCHOR_COLORS[anchor % len(ANCHOR_COLORS)]
        draw.text((table_x, row_y), f"A{anchor}", fill=color, font=_font(15, bold=True))
        for method_index, method in enumerate(("factual", "copy", "shuffled")):
            value = float(np.asarray(errors[method])[anchor])
            x = table_x + 48 + method_index * 205
            draw.text((x, row_y), f"{method} {value:.5f}", fill=(55, 65, 71), font=_font(12))
            draw.rectangle((x, row_y + 22, x + 164, row_y + 31), fill=(229, 232, 233))
            draw.rectangle(
                (x, row_y + 22, x + round(164 * value / max_error), row_y + 31),
                fill=METHOD_COLORS[method],
            )
    mse = _mapping(result["mse"], "mean errors")
    draw.text(
        (table_x, y + 282),
        "direct action response MSE(factual, shuffled): "
        f"{float(result['action_response_mse']):.5f}  |  "
        f"response / copy MSE: {float(result['action_response_over_copy_mse']):.3f}",
        fill=(42, 83, 104),
        font=_font(12, bold=True),
    )
    draw.text(
        (table_x, y + 304),
        "target-error mean: "
        + "  |  ".join(
            f"{name} {float(mse[name]):.5f}"
            for name in ("factual", "copy", "shuffled")
        ),
        fill=(44, 54, 60),
        font=_font(13, bold=True),
    )
    return y + section_height


def _render_panel(
    sample: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
    *,
    transition_index: int | None = None,
    heat_scales: Mapping[str, float] | None = None,
) -> Image.Image:
    width = 1_440
    height = 1_180
    canvas = Image.new("RGB", (width, height), (237, 240, 241))
    draw = ImageDraw.Draw(canvas)
    frames = sample["rgb"]
    if not torch.is_tensor(frames):
        raise TypeError("sample rgb must be a tensor")
    frame_ids = sample["frame_id"].detach().cpu().numpy().astype(int)
    raw_frames = frames.detach().cpu().numpy()
    q = sample.get("q_corrected", sample["q_nominal"])
    valid = sample.get("valid", sample["valid_nominal"])
    q_array = q.detach().cpu().numpy()
    valid_array = valid.detach().cpu().numpy()
    source = str(sample["source"])
    episode = str(sample["episode_id"])
    transition_suffix = ""
    if transition_index is not None:
        if not 0 <= transition_index < len(raw_frames) - 1:
            raise ValueError("transition_index is outside the observed sequence")
        transition_suffix = (
            f"  |  transition {transition_index + 1}/{len(raw_frames) - 1}: "
            f"frame {frame_ids[transition_index]} -> {frame_ids[transition_index + 1]}"
        )
    draw.text(
        (28, 20),
        f"{source.upper()}  |  held-out episode {episode}{transition_suffix}",
        fill=(20, 30, 36),
        font=_font(25 if transition_index is not None else 28, bold=True),
    )
    draw.text(
        (28, 58),
        f"frames {frame_ids.tolist()}  |  shuffled actions from held-out episode {sample['shuffled_episode_id']}",
        fill=(76, 86, 92),
        font=_font(15),
    )
    draw.text(
        (28, 83),
        (
            "TEACHER-FORCED ONE-STEP LATENT PREDICTION - red border is the actual next RGB"
            if transition_index is not None
            else "RGB OBSERVATIONS - MetaWorld vertical display flip applied once; final image is ground truth"
        ),
        fill=(31, 91, 125),
        font=_font(15, bold=True),
    )

    thumb_size = 280
    gap = 24
    start_x = 76
    for index, raw in enumerate(raw_frames):
        x = start_x + index * (thumb_size + gap)
        display_rgb, display_q = _metaworld_display_frame(raw, q_array[index])
        image = Image.fromarray(display_rgb, mode="RGB")
        original_size = image.size
        if original_size[0] != original_size[1]:
            raise ValueError("anchor overlay requires uncropped square source images")
        image = _draw_anchor_overlay(image, display_q, valid_array[index])
        image = ImageOps.fit(
            image, (thumb_size, thumb_size), method=Image.Resampling.BILINEAR
        )
        if transition_index is not None and index > transition_index + 1:
            image = Image.new("RGB", (thumb_size, thumb_size), (87, 96, 101))
            hidden_draw = ImageDraw.Draw(image)
            hidden_draw.text(
                (thumb_size // 2, thumb_size // 2),
                "FUTURE RGB HIDDEN",
                fill=(225, 229, 231),
                font=_font(15, bold=True),
                anchor="mm",
            )
        canvas.paste(image, (x, 118))
        label = f"context t{index}" if index < len(raw_frames) - 1 else "ACTUAL FUTURE RGB"
        label_color = (26, 37, 43) if index < len(raw_frames) - 1 else (151, 52, 43)
        if transition_index is not None:
            if index == transition_index:
                label = "CURRENT OBSERVATION"
                label_color = (31, 91, 125)
                draw.rectangle(
                    (x - 4, 114, x + thumb_size + 4, 402),
                    outline=(31, 91, 125),
                    width=4,
                )
            elif index == transition_index + 1:
                label = "ACTUAL NEXT RGB"
                label_color = (151, 52, 43)
                draw.rectangle(
                    (x - 4, 114, x + thumb_size + 4, 402),
                    outline=(180, 60, 49),
                    width=4,
                )
            elif index < transition_index:
                label = "OBSERVED HISTORY"
            else:
                label = "NOT REVEALED YET"
                label_color = (112, 119, 123)
        draw.text(
            (x, 407),
            f"{label}  |  frame {frame_ids[index]}",
            fill=label_color,
            font=_font(
                14,
                bold=(
                    index == len(raw_frames) - 1
                    if transition_index is None
                    else index in (transition_index, transition_index + 1)
                ),
            ),
        )

    y = 446
    y = _draw_model_section(
        canvas,
        y,
        "tiny",
        results["tiny"],
        fixed_heat_scale=None if heat_scales is None else heat_scales["tiny"],
    )
    _draw_model_section(
        canvas,
        y + 18,
        "bridge",
        results["bridge"],
        fixed_heat_scale=None if heat_scales is None else heat_scales["bridge"],
    )
    return canvas


def _render_animation(
    sample: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
) -> list[Image.Image]:
    sequences = {
        role: list(result["sequence"])
        for role, result in results.items()
    }
    lengths = {len(sequence) for sequence in sequences.values()}
    if len(lengths) != 1:
        raise ValueError("Tiny and bridge prediction sequences have different lengths")
    (length,) = tuple(lengths)
    heat_scales: dict[str, float] = {}
    for role, sequence in sequences.items():
        values = np.stack(
            [
                np.asarray(step["vectors"][name], dtype=np.float32)
                for step in sequence
                for name in ("target", "factual", "copy", "shuffled")
            ]
        )
        scale = float(np.percentile(np.abs(values), 98.0))
        heat_scales[role] = scale if math.isfinite(scale) and scale > 0.0 else 1.0
    frames: list[Image.Image] = []
    for step in range(length):
        panel = _render_panel(
            sample,
            {role: sequence[step] for role, sequence in sequences.items()},
            transition_index=step,
            heat_scales=heat_scales,
        )
        panel = panel.resize((1_080, 885), resample=Image.Resampling.LANCZOS)
        frames.append(panel.convert("P", palette=Image.Palette.ADAPTIVE, colors=160))
    return frames


def _mean_metrics(
    role: str,
    samples: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, float]:
    values = {
        method: np.asarray(
            [predictions[role][_sample_key(sample)]["mse"][method] for sample in samples],
            dtype=np.float64,
        )
        for method in ("factual", "copy", "shuffled")
    }
    result = {f"{method}_mse": float(value.mean()) for method, value in values.items()}
    action_response = np.asarray(
        [
            predictions[role][_sample_key(sample)]["action_response_mse"]
            for sample in samples
        ],
        dtype=np.float64,
    )
    factual = result["factual_mse"]
    copy = result["copy_mse"]
    shuffled = result["shuffled_mse"]
    result["prediction_vs_copy_gain"] = (copy - factual) / copy if copy > 0 else math.nan
    result["shuffle_target_error_delta"] = shuffled - factual
    result["action_response_mse"] = float(action_response.mean())
    result["action_response_over_copy_mse"] = (
        result["action_response_mse"] / copy if copy > 0.0 else math.nan
    )
    return result


def _render_html(
    entries: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Mapping[str, float]],
    manifest_name: str,
) -> str:
    rows = []
    for role in ("tiny", "bridge"):
        metrics = summary[role]
        rows.append(
            "<tr>"
            f"<th>{html.escape(MODEL_LABELS[role])}</th>"
            f"<td>{metrics['factual_mse']:.6f}</td>"
            f"<td>{metrics['copy_mse']:.6f}</td>"
            f"<td>{metrics['shuffled_mse']:.6f}</td>"
            f"<td>{100.0 * metrics['prediction_vs_copy_gain']:.2f}%</td>"
            f"<td>{metrics['action_response_mse']:.6f}</td>"
            f"<td>{metrics['action_response_over_copy_mse']:.3f}</td>"
            f"<td>{metrics['shuffle_target_error_delta']:.6f}</td>"
            "</tr>"
        )
    panels = []
    for entry in entries:
        panels.append(
            f"""<article id="{html.escape(str(entry['id']))}">
<div class="panel-head"><div><h2>{html.escape(str(entry['source']).title())}</h2>
<p>held-out episode {html.escape(str(entry['episode_id']))} &middot; actual future frame {entry['future_frame_id']} &middot; shuffled action donor episode {html.escape(str(entry['shuffled_episode_id']))}</p></div></div>
<a href="{html.escape(str(entry['gif']))}"><img loading="lazy" src="{html.escape(str(entry['gif']))}" alt="Animated teacher-forced latent predictions for {html.escape(str(entry['id']))}"></a>
<p class="artifact-links"><a href="{html.escape(str(entry['gif']))}">Open GIF</a> &middot; <a href="{html.escape(str(entry['image']))}">Open final-transition PNG</a></p>
</article>"""
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tiny and warm-start FCPG bridge latent prediction gallery</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#edf0f1;color:#1b252b;font-family:Inter,Arial,sans-serif;letter-spacing:0}}header{{background:#1c2a31;color:#fff;padding:28px max(22px,calc((100vw - 1160px)/2))}}h1,h2,p{{margin:0}}header p{{margin-top:9px;color:#c8d1d5;line-height:1.5;max-width:980px}}main{{max-width:1160px;margin:auto;padding:24px 22px 60px}}.notice{{border-left:5px solid #c64a3e;background:#fff;padding:15px 18px;line-height:1.55}}.notice b{{color:#a23830}}table{{width:100%;border-collapse:collapse;background:#fff;margin:20px 0 10px}}th,td{{padding:11px 12px;border-bottom:1px solid #d9dfe2;text-align:right}}th:first-child{{text-align:left}}thead th{{color:#5e6a70;font-size:12px}}.caption{{font-size:13px;color:#647178;line-height:1.5}}article{{padding:28px 0;border-bottom:1px solid #c8d0d4}}.panel-head{{display:flex;justify-content:space-between;margin-bottom:10px}}article p{{font-size:13px;color:#66737a;margin-top:4px}}.artifact-links{{margin-top:8px}}img{{display:block;width:100%;height:auto;border:1px solid #c4ccd0;background:#fff}}a{{color:#205f82}}@media(max-width:760px){{table{{display:block;overflow-x:auto;white-space:nowrap}}article{{padding:20px 0}}}}
</style></head><body><header><h1>Tiny and FCPG warm-start bridge</h1><p>Shared held-out samples from the frozen training-validation episode split. This is not a strict matched encoder ablation: the bridge warm-started from Tiny@2k, then adapted its bridge and LeWM parameters for another 2k optimizer steps. The gallery uses factual actions, hold-last copy, and same-task different-episode shuffled actions.</p></header><main>
<div class="notice"><b>LATENT DIAGNOSTIC, NOT RGB RECONSTRUCTION.</b> Each GIF advances through the checkpoint's teacher-forced one-step transitions, not an autoregressive rollout; each transition uses the action block between the two displayed frames. Every revealed RGB image is an observed standard MetaWorld frame, vertically flipped exactly once for human display; anchor y is transformed as H-1-y. Later frames remain fully hidden until their transition. The red-bordered image is the actual next observation, never a generated frame. Each "encoder target" is checkpoint-specific and comes from that actual next observation, not a shared physical latent truth. The checkpoints have no RGB decoder. Colored model rows visualize 192D anchor latents and per-anchor latent MSE only, with a fixed heat scale across each model/sample GIF. Latent dimensions have no direct pixel semantics, and coordinate systems differ between checkpoints, so raw Tiny and bridge MSE values are descriptive and should not be treated as pixel error or representation-aligned distance. No nearest-neighbor retrieval is used.</div>
<table><thead><tr><th>Model</th><th>Factual target MSE</th><th>Copy target MSE</th><th>Shuffled target MSE</th><th>vs copy</th><th>Direct action response MSE</th><th>Response / copy</th><th>Shuffle target-error delta</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p class="caption">Means cover the displayed final-transition samples and all anchors. Direct action response is MSE(pred_factual, pred_shuffled), normalized by copy target MSE in the next column. It measures prediction movement under the action intervention and is distinct from shuffled-target-error minus factual-target-error. These are gallery diagnostics, not the full frozen validation metrics or control evidence. Provenance and per-sample values: <a href="{html.escape(manifest_name)}">{html.escape(manifest_name)}</a>.</p>
{''.join(panels)}</main></body></html>"""


def _prepare_output(output_dir: Path, overwrite: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = output_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    occupied = [path for path in (output_dir / "index.html", output_dir / "manifest.json") if path.exists()]
    if occupied and not overwrite:
        raise FileExistsError(
            "output already contains gallery artifacts; pass --overwrite: "
            + ", ".join(str(path) for path in occupied)
        )
    return images


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.expected_step <= 0:
        raise ValueError("--expected-step must be positive")
    if args.samples_per_source < 2:
        raise ValueError("--samples-per-source must be at least 2 for strict shuffling")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    tiny_path = _resolve(args.tiny_checkpoint)
    bridge_path = _resolve(args.bridge_checkpoint)
    if not tiny_path.is_file() or not bridge_path.is_file():
        missing = [str(path) for path in (tiny_path, bridge_path) if not path.is_file()]
        raise FileNotFoundError("missing checkpoint(s): " + ", ".join(missing))
    tiny_checkpoint, tiny_resolved = _checkpoint_metadata(
        tiny_path, expected_step=args.expected_step, role="tiny"
    )
    bridge_checkpoint, bridge_resolved = _checkpoint_metadata(
        bridge_path, expected_step=args.expected_step, role="bridge"
    )
    _validate_matched_contract(
        tiny_path,
        bridge_path,
        tiny_checkpoint,
        bridge_checkpoint,
        tiny_resolved,
        bridge_resolved,
    )
    gc.collect()

    samples, episode_manifest_path, episode_manifest = _collect_samples(
        tiny_resolved,
        requested_sources=args.sources,
        samples_per_source=args.samples_per_source,
        seed=args.seed,
    )
    if not samples:
        raise RuntimeError("no held-out samples selected")
    device = choose_device(args.device)
    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    for role, path in (("tiny", tiny_path), ("bridge", bridge_path)):
        print(f"running {MODEL_LABELS[role]} on {len(samples)} matched samples", flush=True)
        predictions[role] = _run_checkpoint(
            path,
            samples,
            device=device,
            batch_size=args.batch_size,
            amp=args.amp,
        )

    output_dir = _resolve(args.output_dir)
    images_dir = _prepare_output(output_dir, args.overwrite)
    entries: list[dict[str, Any]] = []
    sample_metrics: list[dict[str, Any]] = []
    animation_transition_count = int(samples[0]["frame_id"].numel() - 1)
    animation_durations_ms = [900] * animation_transition_count
    animation_durations_ms[-1] = 1_800
    for index, sample in enumerate(samples):
        key = _sample_key(sample)
        per_model = {role: predictions[role][key] for role in ("tiny", "bridge")}
        image_name = f"{index:02d}_{str(sample['source']).replace('/', '_')}_episode_{sample['episode_id']}.png"
        gif_name = f"{index:02d}_{str(sample['source']).replace('/', '_')}_episode_{sample['episode_id']}.gif"
        _render_panel(sample, per_model).save(images_dir / image_name, format="PNG", optimize=True)
        animation = _render_animation(sample, per_model)
        if len(animation) != animation_transition_count:
            raise ValueError("sample animation length differs from the gallery contract")
        animation[0].save(
            images_dir / gif_name,
            format="GIF",
            save_all=True,
            append_images=animation[1:],
            duration=animation_durations_ms,
            loop=0,
            optimize=True,
            disposal=2,
        )
        frame_ids = sample["frame_id"].detach().cpu().reshape(-1).tolist()
        raw_indices = sample["raw_index"].detach().cpu().reshape(-1).tolist()
        entry = {
            "id": key,
            "source": str(sample["source"]),
            "task_name": str(sample["task_name"]),
            "episode_id": str(sample["episode_id"]),
            "shuffled_episode_id": str(sample["shuffled_episode_id"]),
            "future_frame_id": int(frame_ids[-1]),
            "image": f"images/{image_name}",
            "gif": f"images/{gif_name}",
        }
        entries.append(entry)
        sample_metrics.append(
            {
                **entry,
                "window_index": int(sample["_window_index"]),
                "frame_ids": [int(value) for value in frame_ids],
                "raw_indices": [int(value) for value in raw_indices],
                "models": {
                    role: {
                        "mse": dict(per_model[role]["mse"]),
                        "per_anchor_mse": {
                            method: np.asarray(values).astype(float).tolist()
                            for method, values in per_model[role]["per_anchor_mse"].items()
                        },
                        "action_response_mse": float(
                            per_model[role]["action_response_mse"]
                        ),
                        "action_response_over_copy_mse": float(
                            per_model[role]["action_response_over_copy_mse"]
                        ),
                        "per_anchor_action_response_mse": np.asarray(
                            per_model[role]["per_anchor_action_response_mse"]
                        )
                        .astype(float)
                        .tolist(),
                        "transitions": [
                            {
                                "from_frame_id": int(frame_ids[step]),
                                "to_frame_id": int(frame_ids[step + 1]),
                                "mse": dict(transition["mse"]),
                                "action_response_mse": float(
                                    transition["action_response_mse"]
                                ),
                                "action_response_over_copy_mse": float(
                                    transition["action_response_over_copy_mse"]
                                ),
                            }
                            for step, transition in enumerate(
                                per_model[role]["sequence"]
                            )
                        ],
                    }
                    for role in ("tiny", "bridge")
                },
            }
        )

    summary = {
        role: _mean_metrics(role, samples, predictions) for role in ("tiny", "bridge")
    }
    manifest = {
        "schema_version": 1,
        "protocol": "lewm_tiny_bridge_matched_heldout_latent_gallery_v1",
        "scope": "diagnostic_not_rgb_reconstruction_not_control_evidence",
        "rgb_contract": "observed_dataset_context_and_actual_future_only",
        "display_transform": DISPLAY_TRANSFORM,
        "prediction_contract": "anchor_latents_only_no_decoder_no_retrieval",
        "animation_contract": {
            "type": "teacher_forced_one_step_transitions",
            "autoregressive_rollout": False,
            "future_rgb_fully_hidden": True,
            "rgb_source": "observed_dataset_only",
            "actual_next_rgb_border": "red",
            "transition_count": animation_transition_count,
            "durations_ms": animation_durations_ms,
            "loop": True,
            "size": [1_080, 885],
        },
        "shuffle_contract": "same_task_different_heldout_episode_rotation",
        "latent_comparability": "within_checkpoint_diagnostics; coordinate_systems_are_not_aligned",
        "training_history": {
            "strict_matched_encoder_ablation": False,
            "tiny": "fixed_latest_at_2000_optimizer_steps",
            "bridge": (
                "warm_started_from_tiny_at_2000_then_adapted_for_2000_"
                "additional_optimizer_steps"
            ),
        },
        "seed": int(args.seed),
        "samples_per_source": int(args.samples_per_source),
        "device": str(device),
        "amp_enabled": bool(args.amp and device.type == "cuda"),
        "episode_manifest": str(episode_manifest_path),
        "episode_manifest_sha256": sha256_file(episode_manifest_path),
        "episode_manifest_protocol": episode_manifest["protocol"],
        "checkpoints": {
            "tiny": {
                "path": str(tiny_path),
                "sha256": sha256_file(tiny_path),
                "step": int(tiny_checkpoint["step"]),
            },
            "bridge": {
                "path": str(bridge_path),
                "sha256": sha256_file(bridge_path),
                "step": int(bridge_checkpoint["step"]),
            },
        },
        "matched_data_config": True,
        "summary": summary,
        "samples": sample_metrics,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "index.html").write_text(
        _render_html(entries, summary, "manifest.json"), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "gallery": str(output_dir / "index.html"),
                "manifest": str(output_dir / "manifest.json"),
                "samples": len(samples),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
