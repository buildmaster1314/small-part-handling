#!/usr/bin/env python3
"""Render SO-101 nominal/true anchor projections under camera drift."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from fcpg.geometry import project_points
from fcpg.so101_camera_grounding import renderer_free_camera_projection
from fcpg.so101_kinematics import (
    SO101_ANCHOR_NAMES,
    SO101_MUJOCO_JOINT_NAMES,
    SO101MuJoCoKinematics,
    normalized_positions_to_mujoco_qpos,
)
from scripts.render_so101_mujoco_prediction import DEFAULT_SCENE, PANEL_SIZE, fit_panel_image
from scripts.train_so101_lewm import episode_splits, parse_exclusions
from scripts.visualize_so101_lewm import (
    CANVAS_SIZE,
    RawFFmpegWriter,
    atomic_write_text,
    checkpoint_contract,
    choose_episode,
    choose_one_episode_per_task,
    episode_task_key,
    load_caches,
    load_font,
    report_slug,
    write_json,
)


ANCHOR_COLORS = ("#00d7ff", "#ffd43b", "#ff5ab3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate SO-101 FCPG projection labels under synthetic camera drift."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", metavar="SOURCE:ID")
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--cache-base", type=Path)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--media", choices=("mp4", "gif", "both"), default="both")
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-25.0)
    parser.add_argument("--distance", type=float, default=0.72)
    parser.add_argument("--azimuth-drift", type=float, default=8.0)
    parser.add_argument("--elevation-drift", type=float, default=2.0)
    parser.add_argument("--distance-drift", type=float, default=0.02)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.episode and args.all_tasks:
        parser.error("--episode and --all-tasks are mutually exclusive")
    if args.stride <= 0 or args.fps <= 0:
        parser.error("--stride and --fps must be positive")
    if args.distance <= 0 or args.distance + args.distance_drift <= 0:
        parser.error("nominal and drifted camera distances must be positive")
    if not args.scene.is_file():
        parser.error(f"SO-101 MuJoCo scene not found: {args.scene}")
    return args


class SO101ProjectionRenderer:
    def __init__(self, scene: Path, args: argparse.Namespace) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(
            self.model, height=PANEL_SIZE[1], width=PANEL_SIZE[0]
        )
        joint_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in SO101_MUJOCO_JOINT_NAMES
            ],
            dtype=np.int32,
        )
        if np.any(joint_ids < 0):
            raise ValueError("SO-101 scene is missing one or more required joints")
        self.qpos_addresses = np.asarray(
            self.model.jnt_qposadr[joint_ids], dtype=np.int32
        )
        self.joint_ranges = np.asarray(
            self.model.jnt_range[joint_ids], dtype=np.float64
        ).copy()
        self.nominal_camera = self._free_camera(
            args.azimuth, args.elevation, args.distance
        )
        self.true_camera = self._free_camera(
            args.azimuth + args.azimuth_drift,
            args.elevation + args.elevation_drift,
            args.distance + args.distance_drift,
        )

    def _free_camera(self, azimuth: float, elevation: float, distance: float):
        camera = self.mujoco.MjvCamera()
        camera.type = self.mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [0.0, 0.0, 0.18]
        camera.distance = float(distance)
        camera.azimuth = float(azimuth)
        camera.elevation = float(elevation)
        return camera

    def render(
        self, state: np.ndarray
    ) -> tuple[np.ndarray, Any, Any]:
        values = np.asarray(state, dtype=np.float64)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError("SO-101 projection state must be one finite six-joint vector")
        self.data.qpos[self.qpos_addresses] = normalized_positions_to_mujoco_qpos(
            values, self.joint_ranges
        )
        self.mujoco.mj_forward(self.model, self.data)
        fovy = float(self.model.vis.global_.fovy)
        self.renderer.update_scene(self.data, camera=self.nominal_camera)
        nominal = renderer_free_camera_projection(
            self.renderer,
            width=PANEL_SIZE[0],
            height=PANEL_SIZE[1],
            fovy_degrees=fovy,
        )
        self.renderer.update_scene(self.data, camera=self.true_camera)
        true = renderer_free_camera_projection(
            self.renderer,
            width=PANEL_SIZE[0],
            height=PANEL_SIZE[1],
            fovy_degrees=fovy,
        )
        return self.renderer.render().copy(), nominal, true

    def close(self) -> None:
        self.renderer.close()


def overlay_projections(
    image: np.ndarray,
    q_nominal: np.ndarray,
    q_true: np.ndarray,
    valid_nominal: np.ndarray,
    valid_true: np.ndarray,
) -> np.ndarray:
    canvas = Image.fromarray(fit_panel_image(image))
    draw = ImageDraw.Draw(canvas)
    for index, (nominal, true, color) in enumerate(
        zip(q_nominal, q_true, ANCHOR_COLORS)
    ):
        if bool(valid_nominal[index]) and bool(valid_true[index]):
            draw.line(
                (float(nominal[0]), float(nominal[1]), float(true[0]), float(true[1])),
                fill="#ff4d4f",
                width=3,
            )
        if bool(valid_nominal[index]):
            x, y = map(float, nominal)
            draw.line((x - 7, y, x + 7, y), fill="#ff4d4f", width=3)
            draw.line((x, y - 7, x, y + 7), fill="#ff4d4f", width=3)
        if bool(valid_true[index]):
            x, y = map(float, true)
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline="white", width=2)
    return np.asarray(canvas, dtype=np.uint8)


def compose_frame(
    clean_image: np.ndarray,
    overlay_image: np.ndarray,
    *,
    source: str,
    task: str,
    episode: int,
    frame_index: int,
    q_nominal: np.ndarray,
    q_true: np.ndarray,
    valid: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    canvas = Image.new("RGB", CANVAS_SIZE, "#eef1f2")
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 22), "SO-101 FCPG projection-label check", fill="#182024", font=load_font(26, True))
    draw.text(
        (40, 58),
        f"{source} | {task} | episode {episode} | raw frame {frame_index}",
        fill="#4a5358",
        font=load_font(16),
    )
    draw.text(
        (40, 84),
        f"camera drift: azimuth {args.azimuth_drift:+.1f} deg, elevation {args.elevation_drift:+.1f} deg, distance {args.distance_drift:+.3f} m",
        fill="#4a5358",
        font=load_font(16),
    )
    canvas.paste(Image.fromarray(fit_panel_image(clean_image)), (40, 130))
    canvas.paste(Image.fromarray(overlay_image), (670, 130))
    draw.rectangle((40, 130, 610, 610), outline="#087e8b", width=5)
    draw.rectangle((670, 130, 1240, 610), outline="#d84a67", width=5)
    draw.text((56, 145), "TRUE DRIFTED CAMERA RGB", fill="#087e8b", font=load_font(20, True))
    draw.text((686, 145), "NOMINAL vs TRUE ANCHORS", fill="#d84a67", font=load_font(20, True))
    valid_error = np.linalg.norm(q_nominal[valid] - q_true[valid], axis=-1)
    error_text = (
        f"mean nominal error: {float(valid_error.mean()):.2f} px"
        if len(valid_error)
        else "anchors outside camera"
    )
    draw.text((60, 652), error_text, fill="#20242a", font=load_font(20, True))
    draw.text(
        (60, 690),
        "Red cross: nominal-calibration projection\nColored dot: true projection on rendered RGB\nRed line: FCPG residual target",
        fill="#4a5358",
        font=load_font(17),
    )
    legend = "\n".join(
        f"{name}: {color}" for name, color in zip(SO101_ANCHOR_NAMES, ANCHOR_COLORS)
    )
    draw.text((700, 666), legend, fill="#20242a", font=load_font(17))
    return np.asarray(canvas, dtype=np.uint8)


def build_html(metrics: dict[str, Any], media: list[str]) -> str:
    media_html = "".join(
        f'<video controls loop playsinline src="{html.escape(name)}"></video>'
        if name.endswith(".mp4")
        else f'<p><a href="{html.escape(name)}">Open GIF</a></p>'
        for name in media
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>SO101 FCPG projection check</title><style>body{{margin:0;background:#eef1f2;color:#20242a;font:15px Arial,sans-serif}}main{{max-width:1280px;margin:auto;padding:24px}}section{{background:white;border:1px solid #c7ced2;margin-bottom:18px;padding:18px}}video{{width:100%}}</style></head><body><main><section><h1>SO-101 FCPG projection-label check</h1><p>This is a geometry diagnostic, not a trained FCPG result. Colored dots are true projections from the same camera that rendered the RGB; red crosses use the nominal camera.</p><p>Mean drift: <b>{metrics['mean_epe_px']:.2f} px</b>. True-anchor support: <b>{100.0 * metrics['true_support']:.1f}%</b>.</p></section><section>{media_html}</section><section><a href="metrics.json">metrics.json</a> | <a href="manifest.json">manifest.json</a></section></main></body></html>"""


def render_choice(args: argparse.Namespace, output: Path, choice: Any) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output is non-empty; pass --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    renderer = SO101ProjectionRenderer(args.scene, args)
    kinematics = SO101MuJoCoKinematics(args.scene)
    selected_local = np.arange(0, len(choice.rows), args.stride, dtype=np.int64)
    media_types = ["mp4", "gif"] if args.media == "both" else [args.media]
    writers = [
        RawFFmpegWriter(output / f"fcpg_projection.{kind}", args.fps, kind)
        for kind in media_types
    ]
    errors: list[float] = []
    true_valid_count = 0
    total_count = 0
    try:
        for local in selected_local:
            row = int(choice.rows[int(local)])
            state = np.asarray(choice.cache.data.states[row], dtype=np.float64)
            anchors = kinematics.forward_anchors(state[None])[0]
            image, nominal_camera, true_camera = renderer.render(state)
            q_nominal, valid_nominal = project_points(anchors, nominal_camera)
            q_true, valid_true = project_points(anchors, true_camera)
            valid = valid_nominal & valid_true
            errors.extend(np.linalg.norm(q_nominal[valid] - q_true[valid], axis=-1).tolist())
            true_valid_count += int(np.sum(valid_true))
            total_count += len(valid_true)
            overlay = overlay_projections(
                image, q_nominal, q_true, valid_nominal, valid_true
            )
            frame = compose_frame(
                image,
                overlay,
                source=choice.cache.source,
                task=episode_task_key(choice),
                episode=choice.episode,
                frame_index=int(choice.cache.data.frame_indices[row]),
                q_nominal=q_nominal,
                q_true=q_true,
                valid=valid,
                args=args,
            )
            for writer in writers:
                writer.write(frame)
        for writer in writers:
            writer.close()
    except Exception:
        for writer in writers:
            writer.abort()
        raise
    finally:
        renderer.close()
    metrics = {
        "source": choice.cache.source,
        "task": episode_task_key(choice),
        "episode": choice.episode,
        "frames": int(len(selected_local)),
        "mean_epe_px": float(np.mean(errors)) if errors else float("nan"),
        "p95_epe_px": float(np.quantile(errors, 0.95)) if errors else float("nan"),
        "true_support": true_valid_count / max(total_count, 1),
    }
    media = [writer.output.name for writer in writers]
    manifest = {
        "format": "so101-fcpg-projection-check-v1",
        "scope": "synthetic MuJoCo geometry diagnostic; no trained FCPG inference",
        "scene": str(args.scene.resolve()),
        "anchors": list(SO101_ANCHOR_NAMES),
        "camera": {
            "nominal": {
                "azimuth": args.azimuth,
                "elevation": args.elevation,
                "distance": args.distance,
            },
            "drift": {
                "azimuth": args.azimuth_drift,
                "elevation": args.elevation_drift,
                "distance": args.distance_drift,
            },
        },
        "gripper_caveat": "official MJCF does not implement LeRobot linear gripper calibration",
        "media": media,
    }
    write_json(output / "metrics.json", metrics)
    write_json(output / "manifest.json", manifest)
    atomic_write_text(output / "index.html", build_html(metrics, media))
    return {**metrics, "path": str(output), "media": media}


def gallery_html(reports: list[dict[str, Any]]) -> str:
    cards = []
    for report in reports:
        relative = Path(report["path"]).name
        media = next(
            (name for name in report["media"] if name.endswith(".mp4")),
            report["media"][0],
        )
        cards.append(
            f'<article><h2>{html.escape(report["task"])}</h2><video controls loop playsinline src="{html.escape(relative + "/" + media)}"></video><p>EPE {report["mean_epe_px"]:.2f} px | <a href="{html.escape(relative + "/index.html")}">details</a></p></article>'
        )
    return """<!doctype html><html><head><meta charset="utf-8"><title>SO101 FCPG projection gallery</title><style>body{font:15px Arial;background:#eef1f2;margin:0}header{padding:24px 5%;background:#202b30;color:white}main{max-width:1400px;margin:auto;padding:24px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}article{background:white;border:1px solid #c7ced2;padding:14px}video{width:100%}@media(max-width:850px){main{grid-template-columns:1fr}}</style></head><body><header><h1>SO-101 FCPG projection-label checks</h1></header><main>""" + "".join(cards) + "</main></body></html>"


def main() -> None:
    args = parse_args()
    checkpoint, embedded_args, cache_paths, frameskip, history, _ = checkpoint_contract(
        args.checkpoint.resolve(), args.cache_base
    )
    del checkpoint
    caches = load_caches(cache_paths)
    selection_caches = [
        cache for cache in caches if not args.source or cache.source == args.source
    ]
    if not selection_caches:
        raise ValueError(f"unknown --source {args.source!r}")
    assignments = episode_splits(
        [cache.data for cache in caches],
        parse_exclusions(embedded_args.get("exclude_episode", [])),
        split_seed=int(embedded_args["split_seed"]),
        val_fraction=float(embedded_args["val_fraction"]),
    )
    choices = (
        choose_one_episode_per_task(selection_caches, frameskip, history, assignments)
        if args.all_tasks
        else [choose_episode(selection_caches, frameskip, history, args.episode, assignments)]
    )
    reports = []
    for choice in choices:
        report_output = args.output / report_slug(choice) if args.all_tasks else args.output
        reports.append(render_choice(args, report_output.resolve(), choice))
        print(f"report: {report_output / 'index.html'}", flush=True)
    if args.all_tasks:
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_write_text(args.output / "index.html", gallery_html(reports))
        (args.output / "gallery.json").write_text(
            json.dumps(reports, indent=2), encoding="utf-8"
        )
        print(f"gallery: {args.output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
