#!/usr/bin/env python3
"""Render synchronized FCPG x LeWM factorial rollouts for human inspection."""

from __future__ import annotations

import argparse
import html
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from fcpg.action_chunk_policy import load_action_chunk_policy
from fcpg.lewm_factorial_prefix_bundle import load_prefix_bundle
from fcpg.lewm_p1_native_complementary import load_frozen_p1_native_checkpoint
from scripts.evaluate_action_chunk_rollout import choose_device
from scripts.evaluate_fcpg_lewm_factorial import (
    CELL_SPECS,
    _moving_goal_contract,
    _prefix_row_fields,
    _predictor_specs_by_grounding,
)
from scripts.evaluate_p1_native_policy_rollout import _load_json, _resolve, rollout_once


CELL_ORDER = ("f0_l0", "f1_l0", "f0_l1", "f1_l1")
CELL_LABELS = {
    "f0_l0": ("F0L0", "Nominal grounding + FK hold"),
    "f1_l0": ("F1L0", "FCPG grounding + FK hold"),
    "f0_l1": ("F0L1", "Nominal grounding + LeWM"),
    "f1_l1": ("F1L1", "FCPG grounding + LeWM"),
}
CELL_COLORS = {
    "f0_l0": "#68737a",
    "f1_l0": "#25865f",
    "f0_l1": "#c06e2c",
    "f1_l1": "#2676a8",
}
MATCH_METRICS = (
    "success",
    "sustained_success",
    "final_success",
    "post_drop_success_fraction",
    "post_drop_maximum_success_streak",
    "reward_sum",
)
REPLAY_CONTRACT_FIELDS = (
    "task",
    "physical_episode",
    "method",
    "visible_grounding_mode",
    "environment_seed",
    "reset_digest",
    "camera_sha256",
    "schedule_sha256",
    "drop_onset_state_sha256",
    "moving_goal_trajectory_sha256",
    "prefix_action_sha256",
    "prefix_native_token_sha256",
    "prefix_sampled_visual_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/lewm/eval/"
            "fcpg_lewm_factorial_pick_matched_moving_goal_dev30_v2.json"
        ),
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument(
        "--episodes",
        help="comma-separated physical episode IDs; default selects three diagnostics",
    )
    parser.add_argument("--episode-count", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="rerun metrics without collecting frames or writing videos",
    )
    args = parser.parse_args()
    if args.episode_count <= 0 or args.fps <= 0:
        parser.error("episode-count and fps must be positive")
    if str(args.device).startswith("cuda") and str(args.device) != "cuda:0":
        parser.error("this renderer is restricted to cuda:0")
    return args


def _rows_by_episode(
    rows: Sequence[Mapping[str, object]],
) -> dict[int, dict[str, Mapping[str, object]]]:
    result: dict[int, dict[str, Mapping[str, object]]] = {}
    for row in rows:
        episode = int(row["physical_episode"])
        cell = str(row["method"])
        if cell not in CELL_SPECS:
            continue
        panel = result.setdefault(episode, {})
        if cell in panel:
            raise ValueError(f"duplicate factorial row for episode {episode}, cell {cell}")
        panel[cell] = row
    if not result:
        raise ValueError("factorial report has no episode rows")
    for episode, panel in result.items():
        if set(panel) != set(CELL_SPECS):
            raise ValueError(f"episode {episode} does not contain all four cells")
    return result


def select_representative_episodes(
    rows: Sequence[Mapping[str, object]], count: int
) -> list[tuple[int, str]]:
    """Choose complementary episodes without inspecting pixels or rerun output."""

    panels = _rows_by_episode(rows)
    if count <= 0:
        raise ValueError("count must be positive")

    def delta(episode: int, left: str, right: str, metric: str) -> float:
        return float(panels[episode][left][metric]) - float(
            panels[episode][right][metric]
        )

    episodes = sorted(panels)
    rankings = (
        (
            "largest full-vs-FCPG gain",
            sorted(
                episodes,
                key=lambda episode: (
                    delta(episode, "f1_l1", "f1_l0", "sustained_success"),
                    delta(
                        episode,
                        "f1_l1",
                        "f1_l0",
                        "post_drop_maximum_success_streak",
                    ),
                    delta(episode, "f1_l1", "f1_l0", "reward_sum"),
                ),
                reverse=True,
            ),
        ),
        (
            "largest FCPG-only gain",
            sorted(
                episodes,
                key=lambda episode: (
                    delta(episode, "f1_l0", "f0_l0", "sustained_success"),
                    delta(
                        episode,
                        "f1_l0",
                        "f0_l0",
                        "post_drop_maximum_success_streak",
                    ),
                    delta(episode, "f1_l0", "f0_l0", "reward_sum"),
                ),
                reverse=True,
            ),
        ),
        (
            "largest unresolved full-system deficit",
            sorted(
                episodes,
                key=lambda episode: (
                    delta(episode, "f1_l1", "f1_l0", "sustained_success"),
                    delta(
                        episode,
                        "f1_l1",
                        "f1_l0",
                        "post_drop_maximum_success_streak",
                    ),
                    delta(episode, "f1_l1", "f1_l0", "reward_sum"),
                ),
            ),
        ),
    )
    selected: list[tuple[int, str]] = []
    used: set[int] = set()
    for label, ranking in rankings:
        for episode in ranking:
            if episode not in used:
                selected.append((episode, label))
                used.add(episode)
                break
        if len(selected) >= min(count, len(episodes)):
            return selected
    for episode in episodes:
        if episode not in used:
            selected.append((episode, "additional paired episode"))
        if len(selected) >= min(count, len(episodes)):
            break
    return selected


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def _draw_anchor_markers(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray | None,
    *,
    origin: tuple[int, int],
    scale: float,
    color: str,
    marker: str,
) -> None:
    if points is None:
        return
    values = np.asarray(points, dtype=np.float32)
    if values.shape != (3, 2):
        return
    ox, oy = origin
    radius = 5
    for x_value, y_value in values:
        if not np.isfinite((x_value, y_value)).all():
            continue
        x = ox + float(x_value) * scale
        y = oy + float(y_value) * scale
        if marker == "circle":
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
        elif marker == "cross":
            draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=3)
            draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=3)
        else:
            draw.polygon(
                ((x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)),
                outline=color,
            )


def _render_frame(
    traces: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    frame_index: int,
    episode: int,
    reason: str,
    drop_start: int,
    drop_duration: int,
    model_seed: int,
) -> np.ndarray:
    panel_width = 420
    panel_height = 392
    gap = 16
    margin = 24
    header = 110
    footer = 58
    canvas = Image.new(
        "RGB",
        (
            margin * 2 + panel_width * 2 + gap,
            header + panel_height * 2 + gap + footer,
        ),
        "#e9eef0",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _font(24, bold=True)
    body_font = _font(16)
    small_font = _font(13)
    draw.text((margin, 18), f"FCPG x LeWM paired rollout | episode {episode:02d}", fill="#16242c", font=title_font)
    draw.text(
        (margin, 52),
        f"{reason} | model seed {model_seed} | raw step {frame_index}",
        fill="#4c606b",
        font=body_font,
    )
    drop_first = drop_start * 5
    drop_last = (drop_start + drop_duration) * 5 - 1
    phase = "DROPOUT" if drop_first <= frame_index <= drop_last else "VISIBLE"
    draw.text(
        (margin, 78),
        f"Phase: {phase}   interruption frames: {drop_first}-{drop_last}",
        fill="#a13f32" if phase == "DROPOUT" else "#25815c",
        font=body_font,
    )

    for panel_index, cell in enumerate(CELL_ORDER):
        row = panel_index // 2
        column = panel_index % 2
        x0 = margin + column * (panel_width + gap)
        y0 = header + row * (panel_height + gap)
        trace = traces[cell][frame_index]
        color = CELL_COLORS[cell]
        draw.rounded_rectangle(
            (x0, y0, x0 + panel_width, y0 + panel_height),
            radius=6,
            fill="#ffffff",
            outline=color,
            width=4,
        )
        short, description = CELL_LABELS[cell]
        draw.text((x0 + 14, y0 + 10), short, fill=color, font=_font(20, bold=True))
        draw.text((x0 + 86, y0 + 14), description, fill="#263740", font=small_font)

        source = Image.fromarray(np.asarray(trace["observed_rgb"], dtype=np.uint8), mode="RGB")
        image_size = 336
        image_x = x0 + (panel_width - image_size) // 2
        image_y = y0 + 48
        resized = source.resize((image_size, image_size), Image.Resampling.BILINEAR)
        canvas.paste(resized, (image_x, image_y))
        panel_draw = ImageDraw.Draw(canvas)
        scale = image_size / source.width
        _draw_anchor_markers(
            panel_draw,
            np.asarray(trace["q_true"]),
            origin=(image_x, image_y),
            scale=scale,
            color="#32d58a",
            marker="circle",
        )
        _draw_anchor_markers(
            panel_draw,
            np.asarray(trace["q_nominal"]),
            origin=(image_x, image_y),
            scale=scale,
            color="#ffad32",
            marker="cross",
        )
        _draw_anchor_markers(
            panel_draw,
            trace["selected_points"],
            origin=(image_x, image_y),
            scale=scale,
            color="#42c4f4",
            marker="diamond",
        )
        if bool(trace["prior_active"]):
            status = "LeWM prior" if cell.endswith("l1") else "FK hold"
        else:
            status = "FCPG visual" if cell.startswith("f1") else "nominal visual"
        draw.rectangle(
            (
                image_x,
                image_y + image_size - 28,
                image_x + image_size,
                image_y + image_size,
            ),
            fill="#14232c",
        )
        draw.text(
            (image_x + 8, image_y + image_size - 23),
            f"{status} | success {float(trace['success']):.0f} | reward {float(trace['cumulative_reward']):.1f}",
            fill="#ffffff",
            font=small_font,
        )

    legend_y = canvas.height - footer + 14
    draw.text(
        (margin, legend_y),
        "Offline overlays: green circle = q_true; orange x = nominal projection; cyan diamond = selected visible grounding.",
        fill="#4b5d66",
        font=small_font,
    )
    return np.asarray(canvas, dtype=np.uint8)


def _validate_replay_contract(
    result: Mapping[str, object], expected: Mapping[str, object]
) -> dict[str, float]:
    for field in REPLAY_CONTRACT_FIELDS:
        if result.get(field) != expected.get(field):
            raise RuntimeError(
                f"visualization replay changed contract field {field}: "
                f"{result.get(field)!r} != {expected.get(field)!r}"
            )
    drift = {}
    for metric in MATCH_METRICS:
        observed = float(result[metric])
        reference = float(expected[metric])
        drift[metric] = observed - reference
    return drift


def _episode_table(
    panel: Mapping[str, Mapping[str, object]],
    replay: Mapping[str, Mapping[str, object]],
) -> str:
    rows = []
    for cell in CELL_ORDER:
        value = panel[cell]
        short, description = CELL_LABELS[cell]
        rows.append(
            "<tr>"
            f"<td><b>{short}</b><br><span>{html.escape(description)}</span></td>"
            f"<td>{float(value['sustained_success']):.0f} / {float(replay[cell]['sustained_success']):.0f}</td>"
            f"<td>{float(value['post_drop_success_fraction']):.3f} / {float(replay[cell]['post_drop_success_fraction']):.3f}</td>"
            f"<td>{float(value['post_drop_maximum_success_streak']):.0f} / {float(replay[cell]['post_drop_maximum_success_streak']):.0f}</td>"
            f"<td>{float(value['reward_sum']):.1f} / {float(replay[cell]['reward_sum']):.1f}</td>"
            "</tr>"
        )
    return "".join(rows)


def _write_html(
    output: Path,
    entries: Sequence[Mapping[str, object]],
    *,
    report_path: Path,
) -> None:
    articles = []
    for entry in entries:
        articles.append(
            f"""
            <article>
              <div class="article-head"><div><h2>Episode {int(entry['episode']):02d}</h2><p>{html.escape(str(entry['reason']))}</p></div><div class="speed"><button data-rate="0.5">0.5x</button><button data-rate="1">1x</button><button data-rate="1.5">1.5x</button></div></div>
              <video controls muted loop playsinline preload="metadata"><source src="{html.escape(str(entry['video']), quote=True)}" type="video/mp4"></video>
              <table><thead><tr><th>Cell</th><th>Sustained<br>frozen / replay</th><th>Post-drop fraction<br>frozen / replay</th><th>Max streak<br>frozen / replay</th><th>Reward<br>frozen / replay</th></tr></thead><tbody>{entry['table']}</tbody></table>
            </article>
            """
        )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FCPG x LeWM Four-Cell Rollouts</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;background:#e9eef0;color:#17242c;font-family:Arial,sans-serif}} header{{background:#172832;color:#fff;padding:26px max(22px,calc((100vw - 1040px)/2))}} header p{{max-width:900px;color:#c7d3d9;line-height:1.55}} main{{max-width:1040px;margin:auto;padding:22px}} article{{background:#fff;border:1px solid #c9d3d8;border-radius:6px;padding:18px;margin-bottom:22px}} .article-head{{display:flex;align-items:center;justify-content:space-between;gap:18px}} h2{{margin:0}} .article-head p{{margin:5px 0;color:#60717a}} video{{display:block;width:100%;margin-top:14px;background:#111}} table{{width:100%;border-collapse:collapse;margin-top:14px}} th,td{{padding:9px;border-bottom:1px solid #dce3e6;text-align:right}} th:first-child,td:first-child{{text-align:left}} td span{{font-size:12px;color:#65757d}} button{{border:1px solid #8b9aa1;background:#f5f7f8;padding:7px 11px;cursor:pointer}} .source{{font-size:12px;color:#64747c;overflow-wrap:anywhere}}
</style></head><body>
<header><h1>FCPG x LeWM Paired Four-Cell Rollouts</h1><p>The four panels are fresh re-executions of the same physical episode, camera, moving target, interruption schedule and canonical pre-drop action prefix. Closed-loop GPU execution is not bit-exact over hundreds of steps, so the table shows frozen-report and replay values separately. Scientific statistics remain those in the frozen report. Green/orange/cyan anchor overlays are offline diagnostics and are not policy inputs.</p></header>
<main>{''.join(articles)}<p class="source">Source report: {html.escape(str(report_path))}</p></main>
<script>document.querySelectorAll('.speed button').forEach(b=>b.addEventListener('click',()=>{{const v=b.closest('article').querySelector('video');v.playbackRate=Number(b.dataset.rate);v.play();}}));</script>
</body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    config_path = _resolve(args.config)
    config = _load_json(config_path)
    seed = int(config["model_seed"] if args.model_seed is None else args.model_seed)
    if seed not in {int(value) for value in config["model_seeds"]}:
        raise ValueError("model seed is not part of the matched factorial")
    configured_output = _resolve(config["output_dir"])
    default_report = (
        configured_output / "factorial_report.json"
        if seed == int(config["canonical_model_seed"])
        else configured_output.parent / f"seed{seed}" / "factorial_report.json"
    )
    report_path = _resolve(args.report or default_report)
    report = _load_json(report_path)
    report_rows = report.get("rows")
    if not isinstance(report_rows, list):
        raise TypeError("factorial report rows must be a list")
    panels = _rows_by_episode(report_rows)

    if args.episodes:
        requested = [int(value.strip()) for value in args.episodes.split(",") if value.strip()]
        if len(requested) != len(set(requested)) or any(value not in panels for value in requested):
            raise ValueError("requested episodes must be unique IDs present in the report")
        selected = [(episode, "user-selected paired episode") for episode in requested]
    else:
        selected = select_representative_episodes(report_rows, args.episode_count)

    source = _load_json(_resolve(config["source_fresh_gate_config"]))
    p1 = source["p1"]
    device = choose_device(args.device)
    policy, policy_args = load_action_chunk_policy(_resolve(p1["policy_checkpoint"]), device)
    policy.set_reliability_override("ones")
    policy.requires_grad_(False).eval()
    pair_specs = _predictor_specs_by_grounding(config, source, seed=seed)
    pairs = {
        grounding: load_frozen_p1_native_checkpoint(
            spec,
            expected_config=expected_config,
            device=device,
        )
        for grounding, (expected_config, spec) in pair_specs.items()
    }
    moving_goal = _moving_goal_contract(config)
    bundle_root = _resolve(config["prefix_bundle_dir"])
    output = _resolve(
        args.output
        or configured_output.parent / f"four_cell_gallery_seed{seed}"
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, object]] = []
    for episode, reason in selected:
        traces: dict[str, list[Mapping[str, object]]] = {}
        rerun_rows: dict[str, Mapping[str, object]] = {}
        replay_drift: dict[str, dict[str, float]] = {}
        shared_actions = load_prefix_bundle(
            bundle_root, episode=episode, grounding="fcpg"
        ).actions
        for cell in CELL_ORDER:
            grounding, method = CELL_SPECS[cell]
            bundle = load_prefix_bundle(
                bundle_root, episode=episode, grounding=grounding
            )
            trace: list[Mapping[str, object]] = []
            result = rollout_once(
                policy,
                policy_args,
                pair=pairs[grounding],
                method=method,
                result_method=cell,
                visible_grounding_mode=grounding,
                model_seed=seed,
                task=str(config["task"]),
                camera=str(config["camera"]),
                width=int(config["width"]),
                height=int(config["height"]),
                benchmark_seed=int(config["benchmark_seed"]),
                task_index=int(config["task_instance_indices"][episode]),
                environment_seed=int(config["environment_seed_base"]) + episode,
                drop_start=int(config["drop_start_decisions"][episode]),
                drop_duration=int(config["drop_duration_decisions"]),
                horizon=int(config["horizon_raw_steps"]),
                radius_px=int(config["occlusion_radius_px"]),
                value_uint8=int(config["occlusion_value_uint8"]),
                device=device,
                camera_rotation_axis_world=config["camera_rotation_axis_world"],
                camera_rotation_deg=float(config["camera_rotation_deg"]),
                moving_goal_seed=(
                    int(moving_goal["seed_base"]) + episode
                    if moving_goal["enabled"]
                    else None
                ),
                moving_goal_amplitude_xy=moving_goal.get("amplitude_xy", (0.02, 0.01)),
                moving_goal_period_raw_steps=float(
                    moving_goal.get("period_raw_steps", 220.0)
                ),
                forced_prefix_actions=shared_actions,
                forced_prefix_tokens=bundle.native_tokens,
                forced_prefix_visual=bundle.sampled_visual,
                occlusion_center_key="q_true",
                compute_clean_action_metric=True,
                visual_validity_fsm=None,
                trace_callback=None if args.verify_only else trace.append,
            )
            result.pop("_prefix_native_tokens")
            result.pop("_prefix_sampled_visual")
            result = {
                "physical_episode": episode,
                **result,
                **_prefix_row_fields(bundle),
            }
            replay_drift[cell] = _validate_replay_contract(
                result, panels[episode][cell]
            )
            traces[cell] = trace
            rerun_rows[cell] = result
            print(
                f"episode={episode:02d} cell={cell} "
                f"sustained={float(result['sustained_success']):.0f} "
                f"reward={float(result['reward_sum']):.2f}",
                flush=True,
            )

        if args.verify_only:
            continue
        frame_counts = {cell: len(value) for cell, value in traces.items()}
        if len(set(frame_counts.values())) != 1:
            raise RuntimeError(f"four-cell trace lengths differ: {frame_counts}")
        video_name = f"episode_{episode:03d}_four_cell.mp4"
        writer = imageio.get_writer(
            output / video_name,
            fps=args.fps,
            codec="libx264",
            quality=7,
            macro_block_size=2,
        )
        try:
            for frame_index in range(next(iter(frame_counts.values()))):
                writer.append_data(
                    _render_frame(
                        traces,
                        frame_index=frame_index,
                        episode=episode,
                        reason=reason,
                        drop_start=int(config["drop_start_decisions"][episode]),
                        drop_duration=int(config["drop_duration_decisions"]),
                        model_seed=seed,
                    )
                )
        finally:
            writer.close()
        entries.append(
            {
                "episode": episode,
                "reason": reason,
                "video": video_name,
                "table": _episode_table(panels[episode], rerun_rows),
                "rerun": {cell: dict(row) for cell, row in rerun_rows.items()},
                "frozen": {
                    cell: dict(panels[episode][cell]) for cell in CELL_ORDER
                },
                "replay_drift": replay_drift,
            }
        )

    if args.verify_only:
        print("verification-only rerun matched the frozen experiment contract")
        return
    _write_html(output, entries, report_path=report_path)
    serializable = [
        {
            "episode": int(entry["episode"]),
            "reason": str(entry["reason"]),
            "video": str(entry["video"]),
            "metrics": {
                cell: {
                    "frozen": {
                        metric: float(entry["frozen"][cell][metric])
                        for metric in MATCH_METRICS
                    },
                    "replay": {
                        metric: float(entry["rerun"][cell][metric])
                        for metric in MATCH_METRICS
                    },
                    "replay_minus_frozen": {
                        metric: float(entry["replay_drift"][cell][metric])
                        for metric in MATCH_METRICS
                    },
                }
                for cell in CELL_ORDER
            },
        }
        for entry in entries
    ]
    (output / "gallery.json").write_text(
        json.dumps({"model_seed": seed, "episodes": serializable}, indent=2),
        encoding="utf-8",
    )
    print(f"HTML: {output / 'index.html'}")


if __name__ == "__main__":
    main()
