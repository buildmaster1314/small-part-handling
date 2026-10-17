#!/usr/bin/env python3
"""Render paired front/side previews from a synthetic SO-101 FCPG dataset."""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fcpg.so101_kinematics import SO101_ANCHOR_NAMES
from fcpg.so101_synthetic_dataset import SO101SyntheticFCPGDataset


ANCHOR_COLORS = ("#00d7ff", "#ffd43b", "#ff5ab3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument(
        "--corrected",
        type=Path,
        help="Optional corrected_q.npy from train_so101_synthetic_fcpg.py.",
    )
    args = parser.parse_args()
    if args.pairs <= 0:
        parser.error("--pairs must be positive")
    return args


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def draw_anchor_overlay(
    rgb: np.ndarray,
    q_nominal: np.ndarray,
    q_true: np.ndarray,
    valid: np.ndarray,
    q_corrected: np.ndarray | None = None,
) -> Image.Image:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    for index, (anchor, nominal, true, supported, color) in enumerate(zip(
        SO101_ANCHOR_NAMES,
        q_nominal,
        q_true,
        valid,
        ANCHOR_COLORS,
        strict=True,
    )):
        del anchor
        if not bool(supported):
            continue
        nx, ny = map(float, nominal)
        tx, ty = map(float, true)
        draw.line((nx, ny, tx, ty), fill="#ff3434", width=2)
        draw.line((nx - 5, ny, nx + 5, ny), fill="#ff3434", width=2)
        draw.line((nx, ny - 5, nx, ny + 5), fill="#ff3434", width=2)
        draw.ellipse((tx - 4, ty - 4, tx + 4, ty + 4), fill=color, outline="white")
        if q_corrected is not None:
            cx, cy = map(float, q_corrected[index])
            draw.ellipse(
                (cx - 5, cy - 5, cx + 5, cy + 5),
                outline="#23c16b",
                width=3,
            )
    return image


def physical_pairs(
    dataset: SO101SyntheticFCPGDataset,
    split: str,
) -> list[list[int]]:
    arrays = dataset.arrays
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for raw_index in dataset.indices(split):
        index = int(raw_index)
        key = (
            int(arrays["source_index"][index]),
            int(arrays["source_row"][index]),
            int(arrays["drift_variant"][index])
            if "drift_variant" in arrays
            else 0,
        )
        groups[key].append(index)
    expected_views = len(dataset.manifest["camera"]["views"])
    complete: list[list[int]] = []
    for indices in groups.values():
        ordered = sorted(indices, key=lambda index: int(arrays["camera_index"][index]))
        cameras = {int(arrays["camera_index"][index]) for index in ordered}
        if len(ordered) == expected_views and len(cameras) == expected_views:
            complete.append(ordered)
    valid = dataset.arrays["valid"]
    visible = [
        indices
        for indices in complete
        if all(bool(np.any(valid[index])) for index in indices)
    ]
    visible.sort(
        key=lambda indices: sum(int(np.sum(valid[index])) for index in indices),
        reverse=True,
    )
    return visible


def render_pair(
    dataset: SO101SyntheticFCPGDataset,
    indices: list[int],
    output: Path,
    corrected: np.ndarray | None,
) -> dict[str, object]:
    arrays = dataset.arrays
    views = dataset.manifest["camera"]["views"]
    width = int(dataset.manifest["width"])
    height = int(dataset.manifest["height"])
    display_width = 360
    display_height = int(round(height * display_width / width))
    margin = 24
    header = 112
    panel_gap = 18
    canvas = Image.new(
        "RGB",
        (
            2 * display_width + 2 * margin + panel_gap,
            display_height + header + margin,
        ),
        "#eef1f2",
    )
    draw = ImageDraw.Draw(canvas)
    first = indices[0]
    source = int(arrays["source_index"][first])
    row = int(arrays["source_row"][first])
    task = int(arrays["task"][first])
    draw.text(
        (margin, 14),
        f"same physical state | source {source} row {row} task {task}",
        fill="#182024",
        font=font(18, True),
    )
    errors: list[float] = []
    corrected_errors: list[float] = []
    for panel, index in enumerate(indices):
        camera_index = int(arrays["camera_index"][index])
        camera_name = str(views[camera_index]["name"])
        valid = np.asarray(arrays["valid"][index], dtype=bool)
        q_nominal = np.asarray(arrays["q_nominal"][index])
        q_true = np.asarray(arrays["q_true"][index])
        epe = np.linalg.norm(q_true[valid] - q_nominal[valid], axis=-1)
        mean_epe = float(epe.mean()) if len(epe) else float("nan")
        errors.extend(epe.tolist())
        corrected_epe = None
        if corrected is not None:
            corrected_values = np.linalg.norm(
                corrected[index][valid] - q_true[valid], axis=-1
            )
            corrected_errors.extend(corrected_values.tolist())
            corrected_epe = (
                float(corrected_values.mean()) if len(corrected_values) else float("nan")
            )
        drift = np.asarray(arrays["drift"][index])
        x = margin + panel * (display_width + panel_gap)
        draw.text(
            (x, 48),
            f"{camera_name} | nominal {mean_epe:.1f}px"
            + (
                f" | FCPG {corrected_epe:.1f}px"
                if corrected_epe is not None
                else ""
            ),
            fill="#4a5358",
            font=font(15, True),
        )
        if dataset.manifest["camera"].get("perturbation_type") == (
            "image_affine_rotation_translation"
        ):
            drift_text = (
                f"affine: rotation {drift[0]:+.1f} deg, "
                f"translation ({drift[1]:+.1f}, {drift[2]:+.1f}) px"
            )
        else:
            drift_text = (
                f"drift: az {drift[0]:+.1f}, el {drift[1]:+.1f} deg, "
                f"distance {drift[2]:+.3f} m"
            )
        draw.text((x, 72), drift_text, fill="#4a5358", font=font(13))
        overlay = draw_anchor_overlay(
            arrays["rgb"][index],
            q_nominal,
            q_true,
            valid,
            None if corrected is None else corrected[index],
        )
        canvas.paste(
            overlay.resize((display_width, display_height), Image.Resampling.NEAREST),
            (x, header),
        )
    canvas.save(output)
    return {
        "image": output.name,
        "source_index": source,
        "source_row": row,
        "task": task,
        "mean_epe_px": float(np.mean(errors)) if errors else None,
        "corrected_epe_px": (
            float(np.mean(corrected_errors)) if corrected_errors else None
        ),
    }


def build_html(
    cards: list[dict[str, object]], split: str, has_corrected: bool
) -> str:
    entries = "".join(
        "<article>"
        f'<img src="{html.escape(str(card["image"]))}" alt="dual camera pair">'
        f'<p>paired nominal EPE: {float(card["mean_epe_px"]):.2f} px'
        + (
            f' | FCPG EPE: {float(card["corrected_epe_px"]):.2f} px'
            if card["corrected_epe_px"] is not None
            else ""
        )
        + "</p>"
        "</article>"
        for card in cards
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SO101 dual-camera FCPG labels</title>
<style>body{{margin:0;background:#eef1f2;color:#20242a;font:15px Arial,sans-serif}}header{{padding:24px 5%;background:#202b30;color:white}}main{{max-width:1100px;margin:auto;padding:24px}}article{{background:white;border:1px solid #c7ced2;margin-bottom:18px;padding:14px}}img{{display:block;width:100%;height:auto}}code{{background:#e4e8ea;padding:2px 5px}}</style></head>
<body><header><h1>SO-101 dual-camera FCPG label check</h1><p>{html.escape(split)} split; each row is one physical state seen by both cameras.</p></header>
<main><p><b>Red cross</b>: nominal projection. <b>Colored dot</b>: true projection on the drifted RGB.{" <b>Green ring</b>: trained FCPG correction." if has_corrected else " This validates synthetic supervision only; no trained corrector is shown."}</p>{entries}</main></body></html>"""


def main() -> None:
    args = parse_args()
    dataset = SO101SyntheticFCPGDataset(args.dataset)
    corrected = None
    if args.corrected is not None:
        corrected = np.load(args.corrected)
        if corrected.shape != dataset.arrays["q_nominal"].shape:
            raise ValueError("corrected array must match q_nominal.npy")
        if not np.all(np.isfinite(corrected)):
            raise ValueError("corrected array contains non-finite values")
    pairs = physical_pairs(dataset, args.split)
    if not pairs:
        raise ValueError(f"no complete camera pairs in {args.split} split")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cards = [
        render_pair(
            dataset,
            indices,
            output / f"pair_{number:02d}.png",
            corrected,
        )
        for number, indices in enumerate(pairs[: args.pairs])
    ]
    (output / "index.html").write_text(
        build_html(cards, args.split, corrected is not None), encoding="utf-8"
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "format": "so101-synthetic-fcpg-dual-preview-v1",
                "dataset": str(dataset.root),
                "split": args.split,
                "pairs": cards,
                "scope": "synthetic label visualization; no trained FCPG result",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output / "index.html", flush=True)


if __name__ == "__main__":
    main()
