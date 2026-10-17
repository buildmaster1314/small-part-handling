#!/usr/bin/env python3
"""Small public-data sanity check for the original PushT LeWM checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from PIL import Image, ImageDraw


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/home/james/lewm/data/lerobot_pusht"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/home/james/lewm/data/lewm_pusht_hf"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lerobot_pusht_lewm_smoke"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3072)
    return parser.parse_args()


def decode_selected_frames(video_path: Path, indices: list[int]) -> np.ndarray:
    wanted = np.asarray(sorted(indices), dtype=np.int64)
    if wanted.size == 0:
        raise ValueError("No video frames requested")

    reader = imageio_ffmpeg.read_frames(str(video_path), pix_fmt="rgb24")
    metadata = next(reader)
    width, height = metadata["size"]
    frames: list[np.ndarray] = []
    cursor = 0
    try:
        for frame_index, raw in enumerate(reader):
            if cursor >= wanted.size:
                break
            if frame_index != int(wanted[cursor]):
                continue
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
            frames.append(frame.copy())
            cursor += 1
    finally:
        reader.close()

    if cursor != wanted.size:
        raise RuntimeError(
            f"Decoded {cursor}/{wanted.size} requested frames from {video_path}"
        )
    return np.stack(frames)


def load_model(model_dir: Path, device: torch.device) -> torch.nn.Module:
    config = json.loads((model_dir / "config.json").read_text())
    model = instantiate(config)
    state = torch.load(
        model_dir / "weights.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def encode_frames(
    model: torch.nn.Module,
    frames: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            batch = torch.from_numpy(frames[start : start + batch_size]).to(device)
            batch = batch.permute(0, 3, 1, 2).float().div_(255.0)
            batch = F.interpolate(
                batch,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            batch = (batch - mean) / std
            encoded = model.encode({"pixels": batch[:, None]})["emb"][:, 0]
            outputs.append(encoded.float().cpu().numpy())
    return np.concatenate(outputs)


def build_episode_sequences(
    table: pd.DataFrame, frameskip: int
) -> dict[int, list[int]]:
    sequences: dict[int, list[int]] = {}
    for episode, rows in table.groupby("episode_index", sort=True):
        indices = rows.index.to_numpy(dtype=np.int64)
        expected = np.arange(indices[0], indices[0] + len(indices))
        if not np.array_equal(indices, expected):
            raise ValueError(f"Episode {episode} is not contiguous")
        sequences[int(episode)] = indices[::frameskip].tolist()
    return sequences


def fit_ridge_probe(
    train_embeddings: np.ndarray,
    train_targets: np.ndarray,
    alpha: float = 1.0,
) -> dict[str, np.ndarray]:
    mean = train_embeddings.mean(axis=0, keepdims=True)
    std = train_embeddings.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    x = (train_embeddings - mean) / std
    x = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1)
    gram = x.T @ x
    penalty = np.eye(gram.shape[0], dtype=gram.dtype) * alpha
    penalty[-1, -1] = 0.0
    weight = np.linalg.solve(gram + penalty, x.T @ train_targets)
    return {"mean": mean, "std": std, "weight": weight}


def apply_ridge_probe(probe: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    normalized = (x - probe["mean"]) / probe["std"]
    normalized = np.concatenate(
        [normalized, np.ones((*normalized.shape[:-1], 1), dtype=normalized.dtype)],
        axis=-1,
    )
    return normalized @ probe["weight"]


def build_windows(
    table: pd.DataFrame,
    sequences: dict[int, list[int]],
    latent_by_row: dict[int, np.ndarray],
    test_episodes: set[int],
    history: int,
    horizon: int,
    frameskip: int,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> dict[str, np.ndarray]:
    histories = []
    targets = []
    actions = []
    target_states = []
    episodes = []
    starts = []
    target_rows = []
    required_observations = history + horizon

    for episode in sorted(test_episodes):
        sequence = sequences[episode]
        for start in range(len(sequence) - required_observations + 1):
            rows = sequence[start : start + required_observations]
            action_blocks = []
            valid = True
            for row in rows[:-1]:
                block = np.stack(table.loc[row : row + frameskip - 1, "action"])
                if len(block) != frameskip:
                    valid = False
                    break
                block = (block.astype(np.float32) - action_mean) / action_std
                action_blocks.append(block.reshape(-1))
            if not valid:
                continue

            histories.append(np.stack([latent_by_row[row] for row in rows[:history]]))
            targets.append(np.stack([latent_by_row[row] for row in rows[history:]]))
            actions.append(np.stack(action_blocks))
            target_states.append(
                np.stack(table.loc[rows[history:], "observation.state"]).astype(
                    np.float32
                )
            )
            episodes.append(episode)
            starts.append(start)
            target_rows.append(rows[history:])

    return {
        "history": np.stack(histories).astype(np.float32),
        "target": np.stack(targets).astype(np.float32),
        "actions": np.stack(actions).astype(np.float32),
        "target_state": np.stack(target_states).astype(np.float32),
        "episode": np.asarray(episodes, dtype=np.int64),
        "start": np.asarray(starts, dtype=np.int64),
        "target_rows": np.asarray(target_rows, dtype=np.int64),
    }


def rollout(
    model: torch.nn.Module,
    history: np.ndarray,
    actions: np.ndarray,
    device: torch.device,
    batch_size: int,
    horizon: int,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(history), batch_size):
            embeddings = torch.from_numpy(history[start : start + batch_size]).to(device)
            action = torch.from_numpy(actions[start : start + batch_size]).to(device)
            generated = []
            for step in range(horizon):
                current = embeddings[:, -3:]
                current_actions = action[:, step : step + 3]
                action_embedding = model.action_encoder(current_actions)
                next_embedding = model.predict(current, action_embedding)[:, -1]
                generated.append(next_embedding)
                embeddings = torch.cat([embeddings, next_embedding[:, None]], dim=1)
            predictions.append(torch.stack(generated, dim=1).float().cpu().numpy())
    return np.concatenate(predictions)


def metric_rows(
    predictions: dict[str, np.ndarray],
    target: np.ndarray,
    target_state: np.ndarray,
    probe: dict[str, np.ndarray],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for horizon in range(target.shape[1]):
        copy = predictions["copy"][:, horizon]
        copy_mse = float(np.mean((copy - target[:, horizon]) ** 2))
        for name, predicted in predictions.items():
            latent_mse = float(np.mean((predicted[:, horizon] - target[:, horizon]) ** 2))
            decoded = apply_ridge_probe(probe, predicted[:, horizon])
            state_rmse_512 = float(
                np.sqrt(np.mean(np.sum((decoded - target_state[:, horizon]) ** 2, axis=-1)))
            )
            rows.append(
                {
                    "method": name,
                    "horizon": horizon + 1,
                    "latent_mse": latent_mse,
                    "copy_normalized_mse": latent_mse / max(copy_mse, 1e-12),
                    "pusher_rmse_image_px": state_rmse_512 * 96.0 / 512.0,
                }
            )
    return rows


def draw_marker(draw: ImageDraw.ImageDraw, xy: np.ndarray, color: str, label: str) -> None:
    x, y = (xy * 96.0 / 512.0 * 3.0).tolist()
    radius = 7
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
    draw.text((x + 9, y - 8), label, fill=color)


def render_gallery(
    output_path: Path,
    frames: np.ndarray,
    frame_rows: list[int],
    frame_position: dict[int, int],
    target_rows: np.ndarray,
    predictions: dict[str, np.ndarray],
    target_states: np.ndarray,
    probe: dict[str, np.ndarray],
    selected_windows: np.ndarray,
) -> None:
    train_latents = predictions["retrieval_bank_latent"]
    train_frame_positions = predictions["retrieval_bank_frame_position"]
    factual = predictions["factual"]
    shuffled = predictions["shuffled_action"]
    pages: list[Image.Image] = []

    bank_norm = np.sum(train_latents * train_latents, axis=1)
    for window_index in selected_windows:
        row = int(target_rows[window_index, 0])
        actual = frames[frame_position[row]]
        predicted_latent = factual[window_index, 0]
        distance = bank_norm - 2.0 * train_latents @ predicted_latent
        nearest = int(np.argmin(distance))
        retrieved = frames[int(train_frame_positions[nearest])]

        canvas = Image.new("RGB", (96 * 6, 96 * 3 + 34), "white")
        left = Image.fromarray(actual).resize((288, 288), Image.Resampling.NEAREST)
        right = Image.fromarray(retrieved).resize((288, 288), Image.Resampling.NEAREST)
        canvas.paste(left, (0, 34))
        canvas.paste(right, (288, 34))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), "Actual future", fill="black")
        draw.text((296, 8), "Predicted latent nearest neighbor", fill="black")

        true_xy = target_states[window_index, 0]
        factual_xy = apply_ridge_probe(probe, factual[window_index, 0:1])[0]
        shuffled_xy = apply_ridge_probe(probe, shuffled[window_index, 0:1])[0]
        draw_marker(draw, true_xy, "#22aa55", "true")
        draw_marker(draw, factual_xy, "#0066ff", "factual")
        draw_marker(draw, shuffled_xy, "#dd3344", "shuffle")
        shifted_factual = factual_xy.copy()
        shifted_factual[0] += 512.0
        draw_marker(draw, shifted_factual, "#0066ff", "pred")
        pages.append(canvas)

    pages[0].save(
        output_path,
        save_all=True,
        append_images=pages[1:],
        duration=500,
        loop=0,
        optimize=False,
    )


def write_html(output_dir: Path, summary: dict) -> None:
    h5 = [row for row in summary["metrics"] if row["horizon"] == summary["horizon"]]
    rows = "\n".join(
        "<tr>"
        f"<td>{row['method']}</td>"
        f"<td>{row['latent_mse']:.6f}</td>"
        f"<td>{row['copy_normalized_mse']:.3f}</td>"
        f"<td>{row['pusher_rmse_image_px']:.2f}</td>"
        "</tr>"
        for row in h5
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LeRobot PushT LeWM smoke validation</title>
<style>
body {{ margin: 24px; font: 14px/1.45 system-ui, sans-serif; color: #202124; }}
main {{ max-width: 980px; margin: auto; }}
h1 {{ font-size: 22px; }}
img {{ width: 100%; max-width: 864px; image-rendering: pixelated; border: 1px solid #bbb; }}
table {{ border-collapse: collapse; width: 100%; max-width: 760px; margin: 18px 0; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 8px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
.note {{ max-width: 760px; color: #555; }}
</style>
</head>
<body><main>
<h1>LeRobot PushT / original LeWM smoke validation</h1>
<p class="note">This is a public-data implementation check, not independent paper evidence. The right panel is nearest-neighbor retrieval from the predicted latent, not a generated image. Markers are decoded by a train-episode-only linear probe.</p>
<img src="trajectory.gif" alt="PushT latent prediction comparison">
<h2>H{summary['horizon']} results</h2>
<table><thead><tr><th>Method</th><th>Latent MSE</th><th>MSE / copy</th><th>Pusher RMSE (96px image)</th></tr></thead>
<tbody>{rows}</tbody></table>
<p class="note">Factual action sensitivity (shuffled minus factual latent MSE): {summary['action_shuffle_delta_horizon']:.6f}. Probe oracle RMSE: {summary['probe_oracle_rmse_image_px']:.2f} px.</p>
</main></body></html>"""
    (output_dir / "index.html").write_text(html)


def main() -> None:
    args = parse_args()
    if args.history != 3:
        raise ValueError("The released PushT checkpoint requires history=3")
    if args.horizon < 1:
        raise ValueError("horizon must be positive")

    table = pd.read_parquet(args.dataset_dir / "data.parquet")
    table = table.reset_index(drop=True)
    sequences = build_episode_sequences(table, args.frameskip)
    all_rows = sorted({row for rows in sequences.values() for row in rows})
    frames = decode_selected_frames(args.dataset_dir / "video.mp4", all_rows)
    frame_position = {row: position for position, row in enumerate(all_rows)}

    device = torch.device(args.device)
    model = load_model(args.model_dir, device)
    embeddings = encode_frames(model, frames, device, args.batch_size)
    latent_by_row = {row: embeddings[position] for position, row in enumerate(all_rows)}

    episode_ids = sorted(sequences)
    split = int(round(len(episode_ids) * (1.0 - args.test_fraction)))
    train_episodes = set(episode_ids[:split])
    test_episodes = set(episode_ids[split:])
    train_rows = [row for episode in train_episodes for row in sequences[episode]]
    train_positions = np.asarray([frame_position[row] for row in train_rows])
    train_embeddings = embeddings[train_positions]
    train_states = np.stack(table.loc[train_rows, "observation.state"]).astype(np.float32)
    probe = fit_ridge_probe(train_embeddings, train_states)

    windows = build_windows(
        table,
        sequences,
        latent_by_row,
        test_episodes,
        args.history,
        args.horizon,
        args.frameskip,
        np.stack(table["action"]).astype(np.float32).mean(axis=0),
        np.stack(table["action"]).astype(np.float32).std(axis=0, ddof=1),
    )
    factual = rollout(
        model,
        windows["history"],
        windows["actions"],
        device,
        args.batch_size,
        args.horizon,
    )
    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(windows["actions"]))
    if np.any(permutation == np.arange(len(permutation))):
        permutation = np.roll(permutation, 1)
    shuffled = rollout(
        model,
        windows["history"],
        windows["actions"][permutation],
        device,
        args.batch_size,
        args.horizon,
    )
    mean_action = rollout(
        model,
        windows["history"],
        np.zeros_like(windows["actions"]),
        device,
        args.batch_size,
        args.horizon,
    )
    copy = np.repeat(windows["history"][:, -1:, :], args.horizon, axis=1)
    predictions = {
        "factual": factual,
        "shuffled_action": shuffled,
        "mean_action": mean_action,
        "copy": copy,
    }
    metrics = metric_rows(
        predictions, windows["target"], windows["target_state"], probe
    )

    decoded_oracle = apply_ridge_probe(probe, windows["target"])
    probe_rmse = float(
        np.sqrt(
            np.mean(
                np.sum((decoded_oracle - windows["target_state"]) ** 2, axis=-1)
            )
        )
        * 96.0
        / 512.0
    )
    factual_h = next(
        row
        for row in metrics
        if row["method"] == "factual" and row["horizon"] == args.horizon
    )
    shuffled_h = next(
        row
        for row in metrics
        if row["method"] == "shuffled_action" and row["horizon"] == args.horizon
    )
    summary = {
        "scope": "public_data_smoke_not_independent_paper_evidence",
        "dataset": "lerobot/pusht",
        "episodes": len(episode_ids),
        "train_probe_episodes": len(train_episodes),
        "test_episodes": len(test_episodes),
        "evaluation_windows": len(windows["history"]),
        "frameskip": args.frameskip,
        "history": args.history,
        "horizon": args.horizon,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "probe_oracle_rmse_image_px": probe_rmse,
        "action_shuffle_delta_horizon": shuffled_h["latent_mse"]
        - factual_h["latent_mse"],
        "metrics": metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    selected_episode = max(
        test_episodes,
        key=lambda episode: int(np.sum(windows["episode"] == episode)),
    )
    selected = np.flatnonzero(windows["episode"] == selected_episode)
    if len(selected) > 32:
        selected = selected[np.linspace(0, len(selected) - 1, 32).round().astype(int)]
    render_predictions = dict(predictions)
    render_predictions["retrieval_bank_latent"] = train_embeddings
    render_predictions["retrieval_bank_frame_position"] = train_positions
    render_gallery(
        args.output_dir / "trajectory.gif",
        frames,
        all_rows,
        frame_position,
        windows["target_rows"],
        render_predictions,
        windows["target_state"],
        probe,
        selected,
    )
    write_html(args.output_dir, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
