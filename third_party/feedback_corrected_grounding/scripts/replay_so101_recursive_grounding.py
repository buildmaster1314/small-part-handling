#!/usr/bin/env python3
"""Replay saved SO-101 rounds through recursive local FCPG grounding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fcpg.so101_realtime_planner import (
    PreparedPlannerConfig,
    PreparedSO101RealtimePlanner,
)
from scripts.evaluate_so101_real_shadow_fcpg_lewm import COLORS, VIEWS


REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO / (
    "outputs/lerobot/so101/realtime_fcpg_lewm_grasp_20260725_v2"
)
DEFAULT_BRIDGE = REPO / (
    "outputs/lerobot/so101/"
    "my_first_task_clean_v1_fcpg_context_bridge_val2_seed6126_2k/latest.pt"
)
DEFAULT_MODEL = REPO / (
    "outputs/lerobot/so101/"
    "so101_multiscale_memory_freeze75_seed5131_2k/latest.pt"
)
DEFAULT_CALIBRATION = REPO / (
    "outputs/lerobot/so101/"
    "my_first_task_clean_v1_dual_camera_calibration/calibration.json"
)
DEFAULT_SECONDARY = REPO / (
    "data/lerobot/so101/cache/my_first_task_clean_v1_column_224"
)
DEFAULT_LIVE_CALIBRATION = REPO / (
    "outputs/lerobot/so101/realtime_fcpg_lewm_mpc_calibration_20260725_v8/"
    "gripper_calibration_captures/pulse_1784958313671272759/analysis/"
    "live_anchor_calibration.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bridge-checkpoint", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--model-checkpoint", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--secondary-cache", type=Path, default=DEFAULT_SECONDARY)
    parser.add_argument(
        "--live-anchor-calibration", type=Path, default=DEFAULT_LIVE_CALIBRATION
    )
    parser.add_argument(
        "--grounding-mode",
        choices=("independent", "recursive"),
        default="recursive",
        help=(
            "independent re-anchors every saved frame from calibrated FK; "
            "recursive reproduces the historical posterior-writeback diagnostic"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.device.startswith("cuda:1"):
        parser.error("CUDA1 is forbidden for this project")
    args.input = args.input.expanduser().resolve()
    args.output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else args.input / "recursive_grounding_replay"
    )
    for path in (
        args.input,
        args.bridge_checkpoint,
        args.model_checkpoint,
        args.calibration,
        args.secondary_cache,
        args.live_anchor_calibration,
    ):
        if not path.expanduser().exists():
            parser.error(f"required input does not exist: {path}")
    return args


def round_key(path: Path) -> int:
    try:
        return int(path.name.split("-", 2)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid round directory: {path.name}") from exc


def load_live_calibration(
    planner: PreparedSO101RealtimePlanner, path: Path
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or payload.get("view") != "column":
        raise ValueError("unsupported live anchor calibration")
    planner.set_live_moving_tip_track(
        "column",
        np.asarray(payload["moving_tip_track_candidates_px"], dtype=np.float32),
        np.asarray(payload["open_state"], dtype=np.float32),
        np.asarray(payload["closed_state"], dtype=np.float32),
    )


def load_rgb(path: Path) -> np.ndarray:
    values = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if values.shape != (224, 224, 3):
        raise ValueError(f"expected 224x224 RGB image: {path}")
    return values


def draw_cross(
    draw: ImageDraw.ImageDraw,
    point: np.ndarray,
    color: tuple[int, int, int],
    *,
    radius: int,
    width: int,
) -> None:
    x, y = map(float, point)
    draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=width)
    draw.line((x + radius, y - radius, x - radius, y + radius), fill=color, width=width)


def recursive_overlay(
    frame: np.ndarray,
    nominal: np.ndarray,
    seed: np.ndarray,
    corrected: np.ndarray,
    valid: np.ndarray,
    previous_corrected: np.ndarray | None,
) -> Image.Image:
    image = Image.fromarray(frame, mode="RGB")
    draw = ImageDraw.Draw(image)
    labels = ("C", "L", "R")
    for index, supported in enumerate(valid):
        if not supported:
            continue
        before = nominal[index]
        prior = seed[index]
        after = corrected[index]
        color = COLORS[index]
        if previous_corrected is not None:
            draw.line(
                (*map(float, previous_corrected[index]), *map(float, prior)),
                fill=(59, 166, 118),
                width=2,
            )
        draw.line((*map(float, before), *map(float, prior)), fill=(242, 188, 59), width=2)
        draw.line((*map(float, prior), *map(float, after)), fill=color, width=2)
        draw_cross(draw, before, (255, 255, 255), radius=4, width=2)
        px, py = map(float, prior)
        draw.rectangle((px - 4, py - 4, px + 4, py + 4), outline=(242, 188, 59), width=2)
        ax, ay = map(float, after)
        draw.ellipse((ax - 5, ay - 5, ax + 5, ay + 5), outline=color, width=3)
        draw.text((ax + 6, ay - 7), labels[index], fill=color, font=ImageFont.load_default())
    return image


def comparison_strip(
    raw: np.ndarray,
    old_overlay: Image.Image,
    new_overlay: Image.Image,
    *,
    new_label: str,
) -> Image.Image:
    labels = ("raw", "old static FCPG", new_label)
    panels = (Image.fromarray(raw, mode="RGB"), old_overlay, new_overlay)
    canvas = Image.new("RGB", (224 * 3, 248), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (label, panel) in enumerate(zip(labels, panels, strict=True)):
        x = 224 * index
        canvas.paste(panel, (x, 24))
        draw.text((x + 8, 7), label, fill=(24, 31, 36), font=font)
    return canvas


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_html(
    output: Path, rows: list[dict[str, Any]], *, grounding_mode: str
) -> None:
    payload = json.dumps(jsonable(rows), ensure_ascii=True)
    if grounding_mode == "independent":
        title = "SO-101 independent current-frame grounding replay"
        legend = (
            "White X: live-calibrated geometric nominal. Colored circle: "
            "one current-frame FCPG update. No posterior is carried between rounds."
        )
    else:
        title = "SO-101 recursive grounding replay (historical diagnostic)"
        legend = (
            "White X: calibrated geometric nominal. Yellow square: propagated "
            "posterior seed. Colored circle: current-frame FCPG update. Green "
            "segment: previous posterior to propagated seed."
        )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
body{{margin:0;background:#f4f5f6;color:#172026;font:14px system-ui,sans-serif}}main{{max-width:1420px;margin:0 auto;padding:18px 22px 32px}}h1{{font-size:22px;margin:0 0 8px}}.legend{{color:#4d5960;margin-bottom:14px}}.toolbar{{display:flex;align-items:center;gap:10px;margin:10px 0 16px}}button{{width:42px;height:36px;border:1px solid #aab2b7;background:white;border-radius:5px;font-size:18px}}input[type=range]{{flex:1}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.panel{{background:white;border:1px solid #cbd1d5;border-radius:6px;padding:10px}}.panel h2{{font-size:15px;margin:0 0 8px}}img{{display:block;width:100%;height:auto}}pre{{background:#172026;color:#e9eef1;padding:12px;border-radius:5px;overflow:auto;white-space:pre-wrap}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>{title}</h1>
<div class="legend">{legend}</div>
<div class="toolbar"><button id="prev" title="Previous">&#9664;</button><button id="play" title="Play">&#9654;</button><button id="next" title="Next">&#9654;&#124;</button><input id="slider" type="range" min="0" max="{len(rows) - 1}" value="0"><strong id="counter"></strong></div>
<div class="grid"><section class="panel"><h2>Column camera</h2><img id="column"></section><section class="panel"><h2>Wrist camera</h2><img id="wrist"></section></div>
<pre id="details"></pre>
<script>
const rows={payload};let index=0,timer=null;const $=id=>document.getElementById(id);
function render(){{const row=rows[index];$('column').src=row.assets.column;$('wrist').src=row.assets.wrist;$('slider').value=index;$('counter').textContent=`${{index+1}} / ${{rows.length}}`;$('details').textContent=JSON.stringify({{round_id:row.round_id,state:row.current_state,grounding:row.grounding}},null,2)}}
function step(delta){{index=(index+delta+rows.length)%rows.length;render()}}
$('prev').onclick=()=>step(-1);$('next').onclick=()=>step(1);$('slider').oninput=e=>{{index=Number(e.target.value);render()}};$('play').onclick=()=>{{if(timer){{clearInterval(timer);timer=null;$('play').textContent='▶'}}else{{timer=setInterval(()=>step(1),900);$('play').textContent='■'}}}};render();
</script></main></body></html>"""
    (output / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    rounds = sorted(
        (path for path in (args.input / "rounds").iterdir() if path.is_dir()),
        key=round_key,
    )
    if not rounds:
        raise RuntimeError("no saved planning rounds found")
    args.output.mkdir(parents=True, exist_ok=True)
    assets = args.output / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    planner = PreparedSO101RealtimePlanner(
        PreparedPlannerConfig(
            bridge_checkpoint=args.bridge_checkpoint,
            model_checkpoint=args.model_checkpoint,
            calibration=args.calibration,
            secondary_cache=args.secondary_cache,
            device=args.device,
            recursive_grounding=args.grounding_mode == "recursive",
        )
    )
    load_live_calibration(planner, args.live_anchor_calibration)

    rows: list[dict[str, Any]] = []
    previous_corrected: dict[str, np.ndarray | None] = {view: None for view in VIEWS}
    for index, round_dir in enumerate(rounds):
        plan_payload = json.loads((round_dir / "plan.json").read_text(encoding="utf-8"))
        current_state = np.asarray(plan_payload["current_state"], dtype=np.float32)
        states = np.repeat(current_state[None], planner.history, axis=0)
        current_frames = {
            view: load_rgb(round_dir / f"{view}_raw.jpg") for view in VIEWS
        }
        frame_history = {
            view: np.repeat(frame[None], planner.history, axis=0)
            for view, frame in current_frames.items()
        }
        recursive = args.grounding_mode == "recursive"
        grounding = planner.observe_grounding(
            frame_history, states, commit=recursive
        )
        asset_rows: dict[str, str] = {}
        view_rows: dict[str, Any] = {}
        for view in VIEWS:
            recursive = recursive_overlay(
                current_frames[view],
                grounding.nominal[view][-1],
                grounding.correction_seed[view][-1],
                grounding.corrected[view][-1],
                grounding.valid[view][-1],
                previous_corrected[view],
            )
            old = Image.open(round_dir / f"{view}_grounding.jpg").convert("RGB")
            strip = comparison_strip(
                current_frames[view],
                old,
                recursive,
                new_label=(
                    "recursive prior + FCPG"
                    if args.grounding_mode == "recursive"
                    else "live calibration + current FCPG"
                ),
            )
            name = f"{index:03d}_{view}.jpg"
            strip.save(assets / name, quality=94)
            asset_rows[view] = f"assets/{name}"
            view_rows[view] = {
                "nominal": grounding.nominal[view][-1],
                "correction_seed": grounding.correction_seed[view][-1],
                "corrected": grounding.corrected[view][-1],
                "valid": grounding.valid[view][-1],
                **grounding.diagnostics[view],
                "mean_local_correction_px": float(
                    np.mean(
                        np.linalg.norm(
                            grounding.corrected[view][-1]
                            - grounding.correction_seed[view][-1],
                            axis=-1,
                        )
                    )
                ),
                "mean_total_correction_px": float(
                    np.mean(
                        np.linalg.norm(
                            grounding.corrected[view][-1]
                            - grounding.nominal[view][-1],
                            axis=-1,
                        )
                    )
                ),
            }
            previous_corrected[view] = (
                grounding.corrected[view][-1].copy()
                if args.grounding_mode == "recursive"
                else None
            )
        rows.append(
            {
                "round_id": plan_payload["round_id"],
                "current_state": current_state,
                "assets": asset_rows,
                "grounding": view_rows,
                "blocked_reasons": grounding.blocked_reasons,
            }
        )
        print(f"[{index + 1:02d}/{len(rounds):02d}] {plan_payload['round_id']}", flush=True)

    (args.output / "summary.json").write_text(
        json.dumps(jsonable(rows), indent=2, ensure_ascii=True), encoding="utf-8"
    )
    write_html(args.output, rows, grounding_mode=args.grounding_mode)
    print(args.output / "index.html")


if __name__ == "__main__":
    main()
