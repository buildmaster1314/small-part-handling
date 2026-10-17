#!/usr/bin/env python3
"""Render seven-task dense action-prefix LeWM q-center forecasts.

The model receives only the latest frozen A4 state, the recorded H1-H8 action
prefix, and task identity. Actual future RGB and q-center labels are used only
for evaluation overlays. The train-only decoder exposes the representation
floor separately from the dynamics prediction error.
"""

from __future__ import annotations

import argparse
import html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a4 import HORIZON, TASKS  # noqa: E402
from fcpg.lewm_action_prefix_a4 import (  # noqa: E402
    ACTION_PREFIX_PROTOCOL,
    load_action_prefix_checkpoint,
)
from fcpg.lewm_action_branch_a4_collection import (  # noqa: E402
    load_frozen_a4_state_encoder,
)
from scripts import render_lewm_action_branch_a4_accuracy_gallery as legacy  # noqa: E402


ALL_HORIZONS = legacy.ALL_HORIZONS
DISPLAY_HORIZONS = legacy.DISPLAY_HORIZONS
GROUND_TRUTH_COLOR = legacy.GROUND_TRUTH_COLOR
DECODER_FLOOR_COLOR = legacy.DECODER_FLOOR_COLOR
PREDICTION_COLOR = legacy.LEWM_COLOR

DEFAULT_CHECKPOINT = ROOT / (
    "outputs/lewm_action_prefix_a4/shared_7task_seed40401_2k/"
    "lewm_action_prefix_latest.pt"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/lewm_action_prefix_a4/accuracy_gallery_seed40401_2k"
)
PROTOCOL_NAME = "a4_fast_lewm_dense_prefix_test_qcenter_gallery_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=legacy.DEFAULT_CONFIG)
    parser.add_argument(
        "--base-checkpoint", type=Path, default=legacy.DEFAULT_BASE_CHECKPOINT
    )
    parser.add_argument(
        "--fcpg-checkpoint", type=Path, default=legacy.DEFAULT_FCPG_CHECKPOINT
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--decoder", type=Path, default=legacy.DEFAULT_DECODER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--task", choices=TASKS, action="append", help="repeat for a task subset"
    )
    parser.add_argument(
        "--max-decision-windows",
        type=legacy._positive_integer,
        help="maximum complete H8 windows per task; default renders all",
    )
    parser.add_argument("--fps", type=float, default=legacy.DEFAULT_FPS)
    parser.add_argument(
        "--max-raw-steps",
        type=legacy._positive_integer,
        default=legacy.DEFAULT_MAX_RAW_STEPS,
    )
    parser.add_argument(
        "--post-success-steps",
        type=legacy._nonnegative_integer,
        default=legacy.DEFAULT_POST_SUCCESS_RAW_STEPS,
    )
    return parser


def _load_model(path: Path, *, device: torch.device) -> tuple[Any, dict[str, Any]]:
    checkpoint = legacy._safe_torch_load(path.expanduser().resolve())
    if checkpoint.get("protocol") != ACTION_PREFIX_PROTOCOL:
        raise ValueError("checkpoint is not the dense action-prefix A4 protocol")
    model = load_action_prefix_checkpoint(checkpoint).to(device).eval()
    return model, checkpoint


@torch.inference_mode()
def _predict_window(
    model: Any,
    *,
    history_embedding: np.ndarray,
    future_action: np.ndarray,
    task_index: int,
    device: torch.device,
) -> np.ndarray:
    history = np.asarray(history_embedding, dtype=np.float32)
    action = np.asarray(future_action, dtype=np.float32)
    if history.shape != (3, 4, 192):
        raise ValueError("history_embedding must have shape [3,4,192]")
    if action.shape != (HORIZON, 20):
        raise ValueError("future_action must have shape [8,20]")
    prediction = model(
        torch.from_numpy(np.ascontiguousarray(history[-1:])).to(device),
        torch.from_numpy(np.ascontiguousarray(action[None])).to(device),
        torch.tensor([int(task_index)], dtype=torch.long, device=device),
    )
    expected = (1, HORIZON, 4, 192)
    if tuple(prediction.shape) != expected:
        raise RuntimeError(
            f"action-prefix prediction shape drifted: {tuple(prediction.shape)}"
        )
    result = prediction[0].float().cpu().numpy()
    if not np.isfinite(result).all():
        raise RuntimeError("action-prefix prediction contains non-finite values")
    return result


def _render_frame(
    actual_future_frames: Sequence[Image.Image],
    ground_truth: np.ndarray,
    decoder_floor: np.ndarray,
    prediction: np.ndarray,
    *,
    task_label: str,
    decision_raw_step: int,
    window_index: int,
    window_count: int,
) -> Image.Image:
    truth = np.asarray(ground_truth, dtype=np.float32)
    floor = np.asarray(decoder_floor, dtype=np.float32)
    estimate = np.asarray(prediction, dtype=np.float32)
    expected = (len(DISPLAY_HORIZONS), 2)
    if truth.shape != expected or floor.shape != expected or estimate.shape != expected:
        raise ValueError("display q-centers must have shape [4,2]")
    if len(actual_future_frames) != len(DISPLAY_HORIZONS):
        raise ValueError("one actual RGB frame is required per displayed horizon")
    sizes = {frame.size for frame in actual_future_frames}
    if len(sizes) != 1:
        raise ValueError("actual future RGB frames must share one size")
    panel_width, panel_height = next(iter(sizes))
    margin, gap, header_height, footer_height = 12, 8, 62, 38
    canvas = Image.new(
        "RGB",
        (
            margin * 2
            + len(DISPLAY_HORIZONS) * panel_width
            + (len(DISPLAY_HORIZONS) - 1) * gap,
            header_height + panel_height + footer_height,
        ),
        (235, 239, 241),
    )
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, header_height), fill=(25, 37, 46))
    draw.text((margin, 8), task_label, fill="white", font=legacy._safe_font(17))
    draw.text(
        (margin, 34),
        (
            f"decision raw {decision_raw_step:03d} | "
            f"window {window_index + 1:02d}/{window_count:02d}"
        ),
        fill=(194, 205, 212),
        font=legacy._safe_font(11),
    )
    legend_x = max(margin, canvas.width - 365)
    for offset, label, color, diamond in (
        (0, "GT", GROUND_TRUTH_COLOR, False),
        (58, "target decode", DECODER_FLOOR_COLOR, True),
        (188, "dense prefix", PREDICTION_COLOR, True),
    ):
        legacy._draw_marker(
            draw,
            np.asarray([legend_x + offset + 5, 18], dtype=np.float32),
            fill=color,
            radius=6,
            diamond=diamond,
        )
        draw.text(
            (legend_x + offset + 17, 10),
            label,
            fill=(230, 235, 238),
            font=legacy._safe_font(10),
        )

    floor_errors = legacy._pixel_epe(truth, floor)
    prediction_errors = legacy._pixel_epe(truth, estimate)
    target_steps = legacy._horizon_frame_indices(
        decision_raw_step, DISPLAY_HORIZONS
    )
    for index, (horizon, source_frame) in enumerate(
        zip(DISPLAY_HORIZONS, actual_future_frames, strict=True)
    ):
        panel = source_frame.convert("RGB").copy()
        panel_draw = ImageDraw.Draw(panel)
        legacy._draw_marker(
            panel_draw,
            truth[index],
            fill=GROUND_TRUTH_COLOR,
            radius=6,
            diamond=False,
        )
        legacy._draw_marker(
            panel_draw,
            floor[index],
            fill=DECODER_FLOOR_COLOR,
            radius=6,
            diamond=True,
        )
        legacy._draw_marker(
            panel_draw,
            estimate[index],
            fill=PREDICTION_COLOR,
            radius=7,
            diamond=True,
        )
        panel_draw.rectangle((0, 0, panel_width, 30), fill=(247, 249, 250))
        panel_draw.text(
            (7, 6),
            f"H{horizon} | actual RGB raw {int(target_steps[index]):03d}",
            fill=(31, 42, 50),
            font=legacy._safe_font(11),
        )
        panel_draw.rectangle(
            (0, panel_height - 30, panel_width, panel_height),
            fill=(247, 249, 250),
        )
        panel_draw.text(
            (7, panel_height - 24),
            (
                f"EPE floor {float(floor_errors[index]):.1f} | "
                f"prefix {float(prediction_errors[index]):.1f} px"
            ),
            fill=(31, 42, 50),
            font=legacy._safe_font(9),
        )
        x = margin + index * (panel_width + gap)
        canvas.paste(panel, (x, header_height))

    draw.text(
        (margin, header_height + panel_height + 10),
        (
            "Blue predicts H1-H8 directly from one current state and the causal "
            "future-action prefix; no predicted state is fed back."
        ),
        fill=(55, 66, 75),
        font=legacy._safe_font(11),
    )
    return canvas


def _metric_row(metrics: Mapping[str, Mapping[str, Any]], horizon: int) -> str:
    key = f"H{horizon}"
    floor = metrics["decoder_floor"][key]
    prediction = metrics["prediction"][key]
    return (
        "<tr>"
        f"<td>H{horizon}</td>"
        f"<td>{float(floor['mean_epe_px']):.2f}</td>"
        f"<td>{float(prediction['mean_epe_px']):.2f}</td>"
        f"<td>{float(prediction['p95_epe_px']):.2f}</td>"
        f"<td>{100.0 * float(prediction['pck_at_4']):.1f}%</td>"
        f"<td>{100.0 * float(prediction['pck_at_8']):.1f}%</td>"
        "</tr>"
    )


def _render_html(entries: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> str:
    sections = []
    for entry in entries:
        rows = "".join(
            _metric_row(entry["metrics"], horizon) for horizon in ALL_HORIZONS
        )
        sections.append(
            f"""
<section><div class="head"><div><h2>{html.escape(str(entry['task_label']))}</h2>
<p>reserved-test instance {int(entry['task_instance'])} &middot; environment seed {int(entry['environment_seed'])} &middot; {int(entry['decision_window_count'])} complete windows</p></div>
<div class="scope">H1/H3/H5/H8 GIF<br>H1-H8 metrics</div></div>
<img src="{html.escape(str(entry['gif']), quote=True)}" alt="dense prefix prediction">
<table><thead><tr><th>Horizon</th><th>Target-decode floor mean</th><th>Dense-prefix mean</th><th>Dense-prefix P95</th><th>PCK@4</th><th>PCK@8</th></tr></thead><tbody>{rows}</tbody></table></section>"""
        )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>A4 dense action-prefix prediction accuracy</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#edf1f3;color:#17212a;font-family:Inter,Arial,sans-serif;letter-spacing:0}}header{{background:#17252e;color:white;padding:24px max(20px,calc((100vw - 1020px)/2))}}h1,h2,p{{margin:0}}header p{{margin-top:8px;color:#c4cfd5;line-height:1.5}}main{{max-width:1020px;margin:auto;padding:20px 20px 60px}}.risk{{background:#fff0ee;border-left:5px solid #bd3d33;padding:16px 18px;line-height:1.55}}.contract{{background:white;border:1px solid #cbd3d8;padding:14px 16px;line-height:1.55;margin-top:16px}}section{{padding:28px 0 34px;border-bottom:1px solid #c8d0d5}}.head{{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:12px}}.head p{{margin-top:5px;color:#66737d;font-size:13px}}.scope{{text-align:right;color:#52616b;font-size:12px;line-height:1.4}}img{{display:block;width:100%;height:auto;border:1px solid #bbc5cb;background:#747c82}}table{{width:100%;border-collapse:collapse;background:white;margin-top:14px}}th,td{{padding:9px 10px;text-align:right;border-bottom:1px solid #dde3e6;font-variant-numeric:tabular-nums}}th:first-child,td:first-child{{text-align:left}}th{{color:#60707a;font-size:12px}}@media(max-width:700px){{.head{{display:block}}.scope{{text-align:left;margin-top:8px}}table{{font-size:12px}}th,td{{padding:7px 5px}}}}
</style></head><body><header><h1>A4 Dense Action-Prefix Prediction Accuracy</h1><p>Checkpoint seed {int(metadata['seed'])}, fixed step {int(metadata['step'])}. Each H1-H8 state is predicted directly from the same current state and its causal action prefix.</p></header><main>
<div class="risk"><strong>Latent forecast decoded by train-only probe, not RGB generation.</strong> Actual future RGB and GT q-center are evaluation-only overlays and never model inputs. Orange shows the representation/decoder floor; blue includes dense action-prefix dynamics error.</div>
<div class="contract"><b>Causal input contract.</b> Latest frozen FCPG-grounded four-slot state, H1-H8 recorded future action chunks, and task identity. The model does not read future observations and does not recursively feed predicted latent states back into itself.<br><b>Decoder:</b> {html.escape(str(metadata['decoder']))}</div>
{''.join(sections)}</main></body></html>"""


def _finish_task(
    task_entry: dict[str, Any],
    *,
    decoder: Mapping[str, np.ndarray],
    output_dir: Path,
    image_height: int,
    fps: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task = str(task_entry["task"])
    windows = task_entry["windows"]
    animations = []
    truths = []
    floors = []
    predictions = []
    window_manifest = []
    display_columns = [horizon - 1 for horizon in DISPLAY_HORIZONS]
    for window_index, window in enumerate(windows):
        alignment = window["alignment"]
        floor = legacy._decode_window_q_centers(
            window["target_embedding"],
            window["history_embedding"],
            window["current_q_center_display"],
            task=task,
            image_height=image_height,
            decoder=decoder,
        )
        prediction = legacy._decode_window_q_centers(
            window["prediction_embedding"],
            window["history_embedding"],
            window["current_q_center_display"],
            task=task,
            image_height=image_height,
            decoder=decoder,
        )
        frame_indices = np.asarray(alignment.future_frame_indices, dtype=np.int64)
        truth = task_entry["q_center_display"][frame_indices]
        floor_error = legacy._pixel_epe(truth, floor)
        prediction_error = legacy._pixel_epe(truth, prediction)
        truths.append(truth)
        floors.append(floor)
        predictions.append(prediction)
        animations.append(
            _render_frame(
                [
                    task_entry["frames"][alignment.future_frame_indices[column]]
                    for column in display_columns
                ],
                truth[display_columns],
                floor[display_columns],
                prediction[display_columns],
                task_label=legacy.TASK_LABELS[task],
                decision_raw_step=alignment.decision_raw_step,
                window_index=window_index,
                window_count=len(windows),
            )
        )
        window_manifest.append(
            {
                "decision_raw_step": int(alignment.decision_raw_step),
                "future_frame_indices": list(alignment.future_frame_indices),
                "epe_px": {
                    "decoder_floor": {
                        f"H{horizon}": float(floor_error[horizon - 1])
                        for horizon in ALL_HORIZONS
                    },
                    "prediction": {
                        f"H{horizon}": float(prediction_error[horizon - 1])
                        for horizon in ALL_HORIZONS
                    },
                },
            }
        )
    truth_array = np.stack(truths)
    metrics = {
        "decoder_floor": legacy._summarize_horizon_metrics(
            truth_array, np.stack(floors)
        ),
        "prediction": legacy._summarize_horizon_metrics(
            truth_array, np.stack(predictions)
        ),
    }
    gif = legacy._save_gallery_gif(
        output_dir, task=task, frames=animations, fps=fps
    )
    preview_name = f"{task.replace('-', '_')}_middle_preview.png"
    animations[len(animations) // 2].save(output_dir / preview_name)
    common = {
        "task": task,
        "task_label": legacy.TASK_LABELS[task],
        "task_instance": int(task_entry["task_instance"]),
        "environment_seed": int(task_entry["environment_seed"]),
        "decision_window_count": len(windows),
        "gif": gif,
        "preview": preview_name,
        "metrics": metrics,
    }
    return common, {
        **common,
        "frame_count": int(task_entry["frame_count"]),
        "raw_action_count": int(task_entry["raw_action_count"]),
        "first_success_step": task_entry["first_success_step"],
        "stable_success_step": task_entry["stable_success_step"],
        "stop_reason": task_entry["stop_reason"],
        "windows": window_manifest,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not np.isfinite(args.fps) or args.fps <= 0.0:
        raise ValueError("fps must be positive and finite")
    device = legacy._runtime_device(args.device)
    contract = legacy._safe_read_json(args.config)
    legacy.validate_a4_contract(contract)
    selected_tasks = tuple(args.task) if args.task else TASKS
    if len(set(selected_tasks)) != len(selected_tasks):
        raise ValueError("--task values must be unique")
    instances = legacy._first_test_instances(contract)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    decoder = legacy._load_train_only_decoder(args.decoder.expanduser().resolve())

    encoder = load_frozen_a4_state_encoder(
        args.base_checkpoint.expanduser().resolve(),
        args.fcpg_checkpoint.expanduser().resolve(),
        device=device,
    )
    prepared_tasks = []
    try:
        for task in selected_tasks:
            entry = legacy._prepare_task_episode(
                task,
                contract=contract,
                task_instance=instances[task],
                encoder=encoder,
                max_decision_windows=args.max_decision_windows,
                max_raw_steps=int(args.max_raw_steps),
                post_success_steps=int(args.post_success_steps),
            )
            prepared_tasks.append(entry)
            print(
                json.dumps(
                    {
                        "task": task,
                        "test_instance": instances[task],
                        "decision_windows": len(entry["windows"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        del encoder
        legacy._clear_runtime_memory(device)

    model, checkpoint = _load_model(args.checkpoint, device=device)
    try:
        for task_entry in prepared_tasks:
            for window in task_entry["windows"]:
                window["prediction_embedding"] = _predict_window(
                    model,
                    history_embedding=window["history_embedding"],
                    future_action=window["future_action"],
                    task_index=window["task_index"],
                    device=device,
                )
    finally:
        del model
        legacy._clear_runtime_memory(device)

    html_entries = []
    manifest_entries = []
    image_height = int(contract["collection"]["height"])
    for task_entry in prepared_tasks:
        html_entry, manifest_entry = _finish_task(
            task_entry,
            decoder=decoder,
            output_dir=output_dir,
            image_height=image_height,
            fps=float(args.fps),
        )
        html_entries.append(html_entry)
        manifest_entries.append(manifest_entry)

    metadata = {
        "seed": int(checkpoint["seed"]),
        "step": int(checkpoint["step"]),
        "decoder": str(args.decoder.expanduser().resolve()),
    }
    (output_dir / "index.html").write_text(
        _render_html(html_entries, metadata), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": PROTOCOL_NAME,
                "split": "test",
                "checkpoint": str(args.checkpoint.expanduser().resolve()),
                "tasks": {
                    entry["task"]: entry["metrics"] for entry in html_entries
                },
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": PROTOCOL_NAME,
                "split": "test",
                "device": str(device),
                "checkpoint": {
                    "path": str(args.checkpoint.expanduser().resolve()),
                    "seed": int(checkpoint["seed"]),
                    "step": int(checkpoint["step"]),
                },
                "model_input": [
                    "latest_frozen_fcpg_grounded_state",
                    "causal_H1_H8_future_action_prefix",
                    "task_identity",
                ],
                "future_rgb": "evaluation_only",
                "future_q_center": "evaluation_only",
                "entries": manifest_entries,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output_dir / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
