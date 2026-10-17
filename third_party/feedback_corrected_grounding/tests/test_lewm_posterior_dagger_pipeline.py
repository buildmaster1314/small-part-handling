from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from fcpg.lewm_posterior_dagger import (
    METHODS,
    PROTOCOL,
    PosteriorDaggerArrays,
    load_posterior_dagger_checkpoint,
    save_posterior_dagger_arrays,
)
from scripts.train_evaluate_lewm_posterior_dagger import main


MODEL_DIM = 8
TRAIN_SEED = 29
TRAIN_STEPS = 2


def _synthetic_rows(
    *,
    method: int,
    iteration: int,
    episodes: tuple[int, ...],
) -> PosteriorDaggerArrays:
    rows_per_episode = 2
    count = len(episodes) * rows_per_episode
    rng = np.random.default_rng(1_000 + 100 * method + 10 * iteration)
    base_action = rng.uniform(-0.45, 0.45, (count, 5, 4)).astype(np.float32)
    expert_action = np.array(base_action, copy=True)
    expert_action[..., :3] = np.clip(
        expert_action[..., :3] + 0.08 * (method + 1), -1.0, 1.0
    )
    return PosteriorDaggerArrays(
        episode_id=np.repeat(
            np.asarray(episodes, dtype=np.int64), rows_per_episode
        ),
        decision_index=np.tile(
            np.arange(rows_per_episode, dtype=np.int64), len(episodes)
        ),
        dagger_iteration=np.full(count, iteration, dtype=np.int64),
        method_index=np.full(count, method, dtype=np.int64),
        missing_age=np.full(count, 1 + iteration, dtype=np.int64),
        sensor_valid=np.zeros(count, dtype=bool),
        state_latent=rng.normal(size=(count, MODEL_DIM)).astype(np.float32),
        action_latent=rng.normal(size=(count, 5, MODEL_DIM)).astype(np.float32),
        base_action=base_action,
        posterior_token=rng.normal(size=(count, 3, MODEL_DIM)).astype(np.float32),
        expert_action=expert_action,
        executed_action=np.array(base_action, copy=True),
    )


def _write_cache(
    path: Path,
    *,
    method: int,
    method_name: str,
    split: str,
    iteration: int,
    episodes: tuple[int, ...],
) -> PosteriorDaggerArrays:
    rows = _synthetic_rows(
        method=method,
        iteration=iteration,
        episodes=episodes,
    )
    save_posterior_dagger_arrays(path, rows)
    metadata = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "split": split,
        "method": method_name,
        "dagger_iteration": iteration,
        "episodes": len(episodes),
        "rows": rows.sample_count,
        "head_checkpoint": None if iteration == 0 else "/synthetic/head.pt",
        "head_metadata": (
            None
            if iteration == 0
            else {"method": method_name, "dagger_iteration": iteration - 1}
        ),
        "episode_metrics": [
            {
                "episode_id": episode,
                "task_index": episode,
                "environment_seed": episode,
            }
            for episode in episodes
        ],
    }
    path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return rows


def test_cpu_synthetic_posterior_dagger_pipeline(tmp_path: Path) -> None:
    caches: dict[str, dict[str, list[str]]] = {
        "train": {},
        "development": {},
    }
    development_rows: dict[str, PosteriorDaggerArrays] = {}
    for method_index, method in enumerate(METHODS):
        train_paths: list[str] = []
        for iteration in (0, 1):
            path = tmp_path / f"{method}.round{iteration}.npz"
            _write_cache(
                path,
                method=method_index,
                method_name=method,
                split="train",
                iteration=iteration,
                episodes=(10, 11),
            )
            train_paths.append(str(path))

        development_paths: list[str] = []
        for iteration in (0, 1):
            development_path = tmp_path / f"{method}.development{iteration}.npz"
            development_rows[method] = _write_cache(
                development_path,
                method=method_index,
                method_name=method,
                split="development",
                iteration=iteration,
                episodes=(90, 91),
            )
            development_paths.append(str(development_path))
        caches["train"][method] = train_paths
        caches["development"][method] = development_paths

    output_dir = tmp_path / "trained"
    config = {
        "protocol": PROTOCOL,
        "methods": list(METHODS),
        "caches": caches,
        "training": {
            "steps": TRAIN_STEPS,
            "batch_size": 6,
            "hidden_dim": 12,
            "learning_rate": 1.0e-3,
            "weight_decay": 0.0,
            "seeds": [TRAIN_SEED],
            "bootstrap_samples": 32,
            "bootstrap_seed": 47,
            "output_dir": str(output_dir),
        },
    }
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert main(["--config", str(config_path), "--device", "cpu"]) == 0

    report_path = output_dir / "report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["protocol"] == PROTOCOL
    assert report["optimizer_steps"] == TRAIN_STEPS
    assert report["methods"] == list(METHODS)
    assert report["gripper"]["changed_by_any_head"] is False

    for method in METHODS:
        checkpoint_path = output_dir / f"{method}_seed{TRAIN_SEED}_latest.pt"
        model, metadata = load_posterior_dagger_checkpoint(
            checkpoint_path,
            device="cpu",
            expected_method=method,
            expected_seed=TRAIN_SEED,
            expected_steps=TRAIN_STEPS,
            expected_dagger_iteration=1,
        )
        assert metadata["method"] == method
        assert metadata["seed"] == TRAIN_SEED
        assert metadata["steps"] == TRAIN_STEPS
        assert metadata["dagger_iteration"] == 1

        rows = development_rows[method]
        base_action = torch.from_numpy(rows.base_action)
        with torch.inference_mode():
            prediction = model(
                base_action=base_action,
                state_latent=torch.from_numpy(rows.state_latent),
                action_latent=torch.from_numpy(rows.action_latent),
                posterior_token=torch.from_numpy(rows.posterior_token),
                missing_age=torch.from_numpy(rows.missing_age).long(),
                sensor_valid=torch.from_numpy(rows.sensor_valid),
            )
        torch.testing.assert_close(
            prediction[..., 3], base_action[..., 3], rtol=0.0, atol=0.0
        )
