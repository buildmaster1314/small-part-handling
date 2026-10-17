#!/usr/bin/env python3
"""Evaluate the historical SO-101 action head from exact causal demo history."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from fcpg.so101_grounded_action_policy import (
    denormalize_so101_actions,
    load_so101_grounded_action_policy,
)
from scripts.render_so101_mujoco_prediction import (
    DEFAULT_SCENE,
    SO101ReplayRenderer,
)
from scripts.visualize_so101_lewm import (
    ACTUAL_COLOR,
    CANVAS_SIZE,
    PREDICTED_COLOR,
    RawFFmpegWriter,
    load_font,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--comparison-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--origin-stride", type=int, default=5)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.device.startswith("cuda:1"):
        parser.error("CUDA1 is forbidden")
    if args.origin_stride <= 0 or args.fps <= 0:
        parser.error("origin-stride and fps must be positive")
    return args


def load_arrays(root: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {name: np.load(root / f"{name}.npy", mmap_mode="r") for name in names}


def episode_bounds(episodes: np.ndarray, episode: int) -> tuple[int, int]:
    rows = np.flatnonzero(np.asarray(episodes) == episode)
    if len(rows) == 0 or np.any(np.diff(rows) != 1):
        raise ValueError(f"episode {episode} is missing or non-contiguous")
    return int(rows[0]), int(rows[-1]) + 1


def active_episode_end(actions: np.ndarray, begin: int, end: int) -> int:
    movement = np.max(np.abs(np.diff(np.asarray(actions[begin:end]), axis=0)), axis=1)
    moving = np.flatnonzero(movement > 0.05)
    if len(moving) == 0:
        return end
    return min(end, begin + int(moving[-1]) + 7)


def model_inputs(
    origins: np.ndarray,
    cache: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    center: np.ndarray,
    scale: np.ndarray,
) -> dict[str, torch.Tensor]:
    offsets = np.asarray([-10, -5, 0], dtype=np.int64)
    history = origins[:, None] + offsets[None]
    past = np.stack(
        [
            np.asarray(cache["actions"][row : row + 5], dtype=np.float32).reshape(-1)
            for origin in origins
            for row in (int(origin) - 10, int(origin) - 5)
        ],
        axis=0,
    ).reshape(len(origins), 2, 30)
    normalized_state = (
        np.asarray(cache["states"][history], dtype=np.float32) - center
    ) / scale
    normalized_past = (past.reshape(len(origins), 2, 5, 6) - center) / scale
    return {
        "per_view_context": torch.from_numpy(
            np.asarray(features["per_view_context"][history], dtype=np.float32)
        ),
        "per_view_anchor_tokens": torch.from_numpy(
            np.asarray(features["per_view_anchor_tokens"][history], dtype=np.float32)
        ),
        "anchor_valid": torch.from_numpy(
            np.asarray(features["anchor_valid"][history], dtype=np.bool_)
        ),
        "state_history": torch.from_numpy(normalized_state),
        "past_action_chunks": torch.from_numpy(
            normalized_past.reshape(len(origins), 2, 30)
        ),
    }


def predict(
    checkpoint_path: Path,
    origins: np.ndarray,
    cache: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    model, checkpoint = load_so101_grounded_action_policy(
        checkpoint_path, device=device
    )
    center = np.asarray(checkpoint["normalization_center"], dtype=np.float32)
    scale = np.asarray(checkpoint["normalization_scale"], dtype=np.float32)
    inputs = model_inputs(origins, cache, features, center, scale)
    outputs = []
    with torch.inference_mode():
        for begin in range(0, len(origins), 48):
            batch = {
                key: value[begin : begin + 48].to(device) for key, value in inputs.items()
            }
            output = model(**batch)
            outputs.append(output.action.cpu())
    normalized = torch.cat(outputs, dim=0)
    return np.asarray(denormalize_so101_actions(normalized, checkpoint)), checkpoint


def targets_for_origins(
    actions: np.ndarray, origins: np.ndarray, episode_end: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    target = np.empty((len(origins), horizon, 6), dtype=np.float32)
    mask = np.zeros((len(origins), horizon), dtype=np.bool_)
    for index, origin in enumerate(origins):
        available = min(horizon, episode_end - int(origin))
        values = np.asarray(actions[int(origin) : int(origin) + available], dtype=np.float32)
        target[index, :available] = values
        target[index, available:] = values[-1]
        mask[index, :available] = True
    return target, mask


def masked_mae(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    per_step = np.mean(np.abs(prediction - target), axis=-1)
    return float(np.sum(per_step * mask) / np.sum(mask))


def metrics_for(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    current: np.ndarray,
) -> dict[str, float]:
    hold = np.repeat(current[:, None], prediction.shape[1], axis=1)
    first = mask[:, :5]
    return {
        "first5_mae_deg": masked_mae(prediction[:, :5], target[:, :5], first),
        "full40_mae_deg": masked_mae(prediction, target, mask),
        "hold_first5_mae_deg": masked_mae(hold[:, :5], target[:, :5], first),
        "hold_full40_mae_deg": masked_mae(hold, target, mask),
        "first5_gripper_mae": float(
            np.sum(np.abs(prediction[:, :5, 5] - target[:, :5, 5]) * first)
            / np.sum(first)
        ),
        "full40_gripper_mae": float(
            np.sum(np.abs(prediction[:, :, 5] - target[:, :, 5]) * mask)
            / np.sum(mask)
        ),
    }


def fit_image(frame: np.ndarray, size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, "#171a1c")
    panel.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return panel


def draw_joint_traces(
    draw: ImageDraw.ImageDraw,
    actual: np.ndarray,
    predicted: np.ndarray,
    horizon: int,
) -> None:
    names = ("pan", "lift", "elbow", "wrist", "roll", "gripper")
    for joint, name in enumerate(names):
        row, column = divmod(joint, 3)
        left = 40 + column * 410
        top = 625 + row * 88
        width, height = 370, 58
        values = np.concatenate((actual[:, joint], predicted[:, joint]))
        low, high = float(values.min()), float(values.max())
        padding = max(1.0, 0.1 * (high - low))
        low, high = low - padding, high + padding

        def point(step: int, value: float) -> tuple[float, float]:
            return (
                left + width * step / max(1, len(actual) - 1),
                top + height * (1.0 - (value - low) / (high - low)),
            )

        draw.rectangle((left, top, left + width, top + height), fill="#f8fafb", outline="#b8c2c8")
        upto = min(horizon + 1, len(actual))
        if upto > 1:
            draw.line(
                [point(i, float(actual[i, joint])) for i in range(upto)],
                fill=ACTUAL_COLOR,
                width=3,
            )
            draw.line(
                [point(i, float(predicted[i, joint])) for i in range(upto)],
                fill=PREDICTED_COLOR,
                width=3,
            )
        draw.text((left + 6, top + 3), name, fill="#20272b", font=load_font(14, True))


def compose_frame(
    wrist: np.ndarray,
    column: np.ndarray,
    mujoco_frame: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    episode: int,
    origin_local: int,
    horizon: int,
    checkpoint_step: int,
) -> np.ndarray:
    canvas = Image.new("RGB", CANVAS_SIZE, "#eef1f2")
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 22), "Historical-input action forecast", fill="#172126", font=load_font(28, True))
    draw.text(
        (40, 62),
        f"held-out episode {episode} | observed step {origin_local} | future +{horizon + 1}/40 | checkpoint {checkpoint_step}",
        fill="#465158",
        font=load_font(17),
    )
    canvas.paste(fit_image(wrist, (270, 270)), (40, 120))
    canvas.paste(fit_image(column, (270, 270)), (330, 120))
    canvas.paste(fit_image(mujoco_frame, (570, 480)), (670, 120))
    draw.rectangle((40, 120, 310, 390), outline=ACTUAL_COLOR, width=4)
    draw.rectangle((330, 120, 600, 390), outline=ACTUAL_COLOR, width=4)
    draw.rectangle((670, 120, 1240, 600), outline=PREDICTED_COLOR, width=4)
    draw.text((52, 132), "WRIST INPUT (FROZEN)", fill=ACTUAL_COLOR, font=load_font(16, True))
    draw.text((342, 132), "COLUMN INPUT (FROZEN)", fill=ACTUAL_COLOR, font=load_font(16, True))
    draw.text((686, 136), "MODEL ACTION -> MUJOCO", fill=PREDICTED_COLOR, font=load_font(18, True))
    current_error = float(np.mean(np.abs(predicted[horizon] - actual[horizon])))
    draw.text(
        (40, 425),
        "Real input frames: t-10, t-5, t",
        fill="#273137",
        font=load_font(17),
    )
    draw.text(
        (40, 452),
        "Executed history: [t-10..t-6], [t-5..t-1]",
        fill="#273137",
        font=load_font(17),
    )
    draw.text(
        (40, 486),
        f"Current six-joint MAE: {current_error:.2f} deg",
        fill="#273137",
        font=load_font(17, True),
    )
    draw.text(
        (40, 516),
        f"Gripper actual: {actual[horizon, 5]:.2f}   predicted: {predicted[horizon, 5]:.2f}",
        fill="#273137",
        font=load_font(17, True),
    )
    draw.text(
        (40, 550),
        "Teal = recorded expert command    Red = model-predicted command",
        fill="#465158",
        font=load_font(16),
    )
    draw_joint_traces(draw, actual, predicted, horizon)
    return np.asarray(canvas, dtype=np.uint8)


def build_html(report: dict[str, Any]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{values['first5_mae_deg']:.3f}</td>"
        f"<td>{values['full40_mae_deg']:.3f}</td>"
        f"<td>{values['first5_gripper_mae']:.3f}</td>"
        f"<td>{values['full40_gripper_mae']:.3f}</td>"
        "</tr>"
        for name, values in report["models"].items()
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SO-101 historical action forecast</title>
<style>body{{margin:0;background:#edf0f1;color:#182126;font:15px system-ui,sans-serif}}main{{max-width:1280px;margin:auto;padding:24px}}video{{display:block;width:100%;background:#111}}table{{border-collapse:collapse;margin:18px 0;background:white}}th,td{{padding:9px 14px;border:1px solid #bcc5ca;text-align:right}}th:first-child,td:first-child{{text-align:left}}a{{color:#086f7a}}</style></head>
<body><main><h1>Held-out historical-input action forecast</h1>
<video controls playsinline preload="metadata" src="forecast.mp4"></video>
<table><thead><tr><th>Model</th><th>first 5 MAE</th><th>full 40 MAE</th><th>first 5 gripper MAE</th><th>full 40 gripper MAE</th></tr></thead><tbody>{rows}</tbody></table>
<p><a href="metrics.json">metrics.json</a></p></main></body></html>"""


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint.resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_root = Path(payload["feature_cache"]).resolve()
    manifest = json.loads((feature_root / "manifest.json").read_text())
    wrist_root = Path(manifest["wrist_cache"])
    column_root = Path(manifest["column_cache"])
    cache = load_arrays(wrist_root, ("frames", "actions", "states", "episodes", "frame_indices"))
    column = load_arrays(column_root, ("frames", "episodes", "frame_indices"))
    features = load_arrays(feature_root, ("per_view_context", "per_view_anchor_tokens", "anchor_valid"))
    if not np.array_equal(cache["episodes"], column["episodes"]):
        raise ValueError("wrist and column episode identities differ")
    if not np.array_equal(cache["frame_indices"], column["frame_indices"]):
        raise ValueError("wrist and column frame indices differ")
    validation = [int(value) for value in payload["validation_episodes"]]
    episode = int(args.episode if args.episode is not None else validation[0])
    if episode not in validation:
        raise ValueError("the requested episode was not held out during training")
    episode_begin, episode_end = episode_bounds(cache["episodes"], episode)
    episode_end = active_episode_end(cache["actions"], episode_begin, episode_end)
    all_origins = np.arange(episode_begin + 10, episode_end, dtype=np.int64)
    render_origins = all_origins[:: args.origin_stride]
    if render_origins[-1] != all_origins[-1]:
        render_origins = np.append(render_origins, all_origins[-1])
    device = torch.device(args.device)

    primary, primary_payload = predict(
        checkpoint_path, all_origins, cache, features, device
    )
    target, mask = targets_for_origins(cache["actions"], all_origins, episode_end, 40)
    current = np.asarray(cache["states"][all_origins], dtype=np.float32)
    models = {
        f"{checkpoint_path.name} (step {primary_payload['step']})": metrics_for(
            primary, target, mask, current
        )
    }
    if args.comparison_checkpoint is not None:
        comparison, comparison_payload = predict(
            args.comparison_checkpoint.resolve(), all_origins, cache, features, device
        )
        models[
            f"{args.comparison_checkpoint.name} (step {comparison_payload['step']})"
        ] = metrics_for(comparison, target, mask, current)

    render_index = {int(origin): index for index, origin in enumerate(all_origins)}
    renderer = SO101ReplayRenderer(
        args.scene,
        azimuth=135.0,
        elevation=-25.0,
        distance=0.72,
    )
    writer = RawFFmpegWriter(output / "forecast.mp4", args.fps, "mp4")
    rendered_frames = 0
    try:
        for origin in render_origins:
            index = render_index[int(origin)]
            available = int(mask[index].sum())
            for horizon in range(available):
                robot, _, _ = renderer.render(primary[index, horizon])
                frame = compose_frame(
                    cache["frames"][origin],
                    column["frames"][origin],
                    robot,
                    target[index],
                    primary[index],
                    episode=episode,
                    origin_local=int(origin - episode_begin),
                    horizon=horizon,
                    checkpoint_step=int(primary_payload["step"]),
                )
                writer.write(frame)
                rendered_frames += 1
        writer.close()
    except BaseException:
        writer.abort()
        raise
    finally:
        renderer.close()

    report = {
        "format": "so101-grounded-action-historical-input-eval-v1",
        "episode": episode,
        "episode_rows": int(episode_end - episode_begin),
        "origins": int(len(all_origins)),
        "render_origins": int(len(render_origins)),
        "rendered_frames": rendered_frames,
        "models": models,
        "hold_baseline": {
            "first5_mae_deg": models[next(iter(models))]["hold_first5_mae_deg"],
            "full40_mae_deg": models[next(iter(models))]["hold_full40_mae_deg"],
        },
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "index.html").write_text(build_html(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
