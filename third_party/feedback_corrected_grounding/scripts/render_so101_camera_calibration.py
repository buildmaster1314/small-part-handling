#!/usr/bin/env python3
"""Render clicked and fitted SO-101 anchors for human calibration review."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fcpg.so101_camera_calibration import ProjectionFit, predict_projection


COLORS = ("#00a5d8", "#e4a11b", "#dc4f9b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--annotation-image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _fit(payload: dict[str, object]) -> ProjectionFit:
    return ProjectionFit(
        view=str(payload["view"]),
        feature_names=tuple(str(value) for value in payload["feature_names"]),
        coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
        ridge=float(payload["ridge"]),
    )


def overlay(
    image: Image.Image,
    predicted: np.ndarray,
    clicked: np.ndarray,
) -> tuple[Image.Image, np.ndarray]:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    error = np.linalg.norm(predicted - clicked, axis=-1)
    for index, (prediction, target, color) in enumerate(
        zip(predicted, clicked, COLORS)
    ):
        px, py = map(float, prediction)
        tx, ty = map(float, target)
        draw.line((px, py, tx, ty), fill="#f04438", width=2)
        draw.line((px - 6, py, px + 6, py), fill="#ffffff", width=3)
        draw.line((px, py - 6, px, py + 6), fill="#ffffff", width=3)
        draw.ellipse((tx - 5, ty - 5, tx + 5, ty + 5), fill=color, outline="#111111", width=1)
        draw.text((tx + 7, ty - 8), str(index + 1), fill="#111111", font=ImageFont.load_default())
    return canvas, error


def _page(rows: list[dict[str, object]], calibration: dict[str, object]) -> str:
    cards = []
    for row in rows:
        cards.append(
            f'''<article class="card"><header><b>{html.escape(str(row["id"]))}</b>
<span>episode {row["episode"]}, frame {row["frame_index"]}</span></header>
<div class="views"><figure><figcaption>Wrist | mean {row["wrist_mean"]:.2f}px, max {row["wrist_max"]:.2f}px</figcaption><img src="{row["wrist_image"]}"></figure>
<figure><figcaption>Column | mean {row["column_mean"]:.2f}px, max {row["column_max"]:.2f}px</figcaption><img src="{row["column_image"]}"></figure></div></article>'''
        )
    metrics = calibration["metrics"]
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SO-101 双相机投影复核</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#eef2f0;color:#17211d;font:15px/1.4 system-ui,sans-serif}}main{{max-width:1200px;margin:auto;padding:20px}}h1{{font-size:23px;margin:0 0 6px}}.summary{{background:#fff;border:1px solid #cbd5cf;padding:14px;margin-bottom:14px}}.summary p{{margin:4px 0;color:#59665f}}.card{{background:#fff;border:1px solid #cbd5cf;margin-bottom:12px;padding:10px}}.card header{{display:flex;justify-content:space-between;margin-bottom:8px}}.card header span{{color:#65716b}}.views{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}figure{{margin:0}}figcaption{{font-size:13px;margin-bottom:5px}}img{{display:block;width:100%;image-rendering:auto;background:#202522}}code{{font-family:ui-monospace,monospace}}@media(max-width:700px){{.views{{grid-template-columns:1fr}}}}</style></head><body><main>
<section class="summary"><h1>SO-101 双相机 nominal projection 复核</h1>
<p>彩色圆点是人工点击；白色十字是 FK projection fit；红线是误差。下方按单帧最大误差从高到低排列。</p>
<p>整 episode 留一：wrist mean <b>{metrics["wrist"]["cross_episode"]["mean_epe_px"]:.2f}px</b>, P95 <b>{metrics["wrist"]["cross_episode"]["p95_epe_px"]:.2f}px</b>；column mean <b>{metrics["column"]["cross_episode"]["mean_epe_px"]:.2f}px</b>, P95 <b>{metrics["column"]["cross_episode"]["p95_epe_px"]:.2f}px</b>。</p>
<p>该页面只验证 nominal projection，不是 FCPG 校正结果。</p></section>{''.join(cards)}</main></body></html>'''


def main() -> None:
    args = parse_args()
    annotations = json.loads(args.annotations.read_text())
    calibration = json.loads(args.calibration.read_text())
    fits = {view: _fit(calibration["fits"][view]) for view in ("wrist", "column")}
    output = args.output.resolve()
    image_output = output / "images"
    image_output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for sample in annotations["samples"]:
        sample_id = str(sample["id"])
        base = np.asarray(sample["anchors_base_xyz"], dtype=np.float64)[None]
        local = np.asarray(sample["anchors_gripper_xyz"], dtype=np.float64)[None]
        gripper = np.asarray([sample["state"][5]], dtype=np.float64)
        row: dict[str, object] = {
            "id": sample_id,
            "episode": int(sample["episode"]),
            "frame_index": int(sample["frame_index"]),
        }
        frame_max = 0.0
        for view in ("wrist", "column"):
            clicked = np.asarray(
                annotations["points"][f"{sample_id}:{view}"], dtype=np.float64
            )
            predicted = predict_projection(fits[view], base, local, gripper)[0]
            source = args.annotation_image_root / sample["images"][view]
            rendered, error = overlay(Image.open(source), predicted, clicked)
            image_name = f"{sample_id}_{view}.jpg"
            rendered.save(image_output / image_name, quality=94, optimize=True)
            row[f"{view}_image"] = f"images/{image_name}"
            row[f"{view}_mean"] = float(np.mean(error))
            row[f"{view}_max"] = float(np.max(error))
            frame_max = max(frame_max, float(np.max(error)))
        row["frame_max"] = frame_max
        rows.append(row)
    rows.sort(key=lambda value: float(value["frame_max"]), reverse=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.html").write_text(_page(rows, calibration), encoding="utf-8")
    (output / "rows.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"calibration_review={output / 'index.html'}")


if __name__ == "__main__":
    main()
