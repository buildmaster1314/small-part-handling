from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler

import fcpg.lewm_runtime as runtime
from fcpg.lewm_grounded_model import LEWM_UPSTREAM_COMMIT
from fcpg.lewm_oracle_target import (
    FrozenOracleTargetEncoder,
    save_frozen_oracle_target_encoder,
)
from fcpg.lewm_provenance import sha256_file
from fcpg.lewm_sequence_data import frame_identity_sha256


def test_expected_parameter_count_contract_is_enforced() -> None:
    model = nn.Linear(3, 2)
    actual = sum(parameter.numel() for parameter in model.parameters())

    assert runtime._validate_expected_parameter_count(model, {}) == actual
    assert runtime._validate_expected_parameter_count(
        model,
        {"contract": {"expected_parameter_count": actual}},
    ) == actual
    with pytest.raises(ValueError, match="parameter count violates"):
        runtime._validate_expected_parameter_count(
            model,
            {"contract": {"expected_parameter_count": actual + 1}},
        )


class _TinyDenseVision(nn.Module):
    output_dim = 8

    def __init__(self, **_: object) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, self.output_dim, kernel_size=1)

    def forward(self, pixels: torch.Tensor) -> SimpleNamespace:
        feature_map = self.projection(pixels)
        return SimpleNamespace(
            feature_map=feature_map,
            global_feature=feature_map.mean(dim=(-1, -2)),
        )


def _write_source(path: Path, task: str, offset: int) -> None:
    episodes = 4
    frames_per_episode = 5
    count = episodes * frames_per_episode
    episode_id = np.repeat(np.arange(episodes, dtype=np.int32), frames_per_episode)
    frame_id = np.tile(np.arange(frames_per_episode, dtype=np.int32), episodes)
    rng = np.random.default_rng(offset)
    q_nominal = rng.uniform(1.0, 6.0, size=(count, 3, 2)).astype(np.float32)
    np.savez(
        path,
        rgb=rng.integers(0, 255, size=(count, 8, 8, 3), dtype=np.uint8),
        obs=rng.normal(size=(count, 6)).astype(np.float32),
        action=rng.normal(size=(count, 2)).astype(np.float32),
        task_name=np.asarray([task] * count),
        episode_id=episode_id,
        frame_id=frame_id,
        keypoints_xyz=rng.normal(size=(count, 3, 3)).astype(np.float32),
        q_nominal=q_nominal,
        q_true=q_nominal + rng.normal(scale=0.2, size=q_nominal.shape).astype(np.float32),
        valid_true=np.ones((count, 3), dtype=bool),
        valid_nominal=np.ones((count, 3), dtype=bool),
        observation_action_alignment=np.asarray("pre_action"),
    )


def _config(tmp_path: Path, sources: list[Path]) -> dict[str, object]:
    return {
        "seed": 17,
        "output_dir": str(tmp_path / "run"),
        "data": {
            "sources": [
                {"name": path.stem, "path": str(path)} for path in sources
            ],
            "history_size": 2,
            "rollout_steps": 1,
            "grounding_mode": "nominal",
            "frame_skip": 1,
            "val_fraction": 0.25,
            "balanced_sources": True,
        },
        "loader": {
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        },
        "model": {
            "vision": {
                "size": "tiny",
                "patch_size": 2,
                "image_size": 8,
                "pretrained": False,
                "use_mask_token": False,
            },
            "frame": {
                "latent_dim": 16,
                "depth": 1,
                "heads": 4,
                "mlp_ratio": 2,
                "dropout": 0.0,
            },
            "predictor": {
                "depth": 1,
                "heads": 4,
                "mlp_dim": 32,
                "dim_head": 4,
                "dropout": 0.0,
                "projector_hidden_dim": 32,
            },
            "sigreg": {"knots": 5, "num_proj": 8},
        },
        "loss": {"sigreg_weight": 0.01, "per_source_sigreg": False},
        "optimizer": {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "betas": [0.9, 0.99],
            "eps": 1e-8,
            "warmup_steps": 1,
        },
        "trainer": {
            "epochs": 2,
            "max_steps": 0,
            "max_train_batches": 0,
            "max_val_batches": 1,
            "grad_clip": 1.0,
            "amp": False,
            "amp_dtype": "bfloat16",
            "checkpoint_every_epochs": 1,
        },
    }


def _write_episode_manifest(
    path: Path,
    sources: list[Path],
    *,
    overlap: bool = False,
) -> None:
    entries = {}
    for source in sources:
        with np.load(source, allow_pickle=False) as raw:
            task = str(raw["task_name"][0])
        train = [f"{task}:0", f"{task}:1"]
        validation = [f"{task}:1" if overlap else f"{task}:2"]
        entries[source.stem] = {
            "path": str(source.resolve()),
            "data_sha256": sha256_file(source),
            "train_episodes": train,
            "val_episodes": validation,
            "excluded_episodes": [f"{task}:3"],
        }
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": "lewm_gate_l1_camera_aware_episode_split",
                "sources": entries,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_oracle_cache(
    source: Path,
    directory: Path,
    encoder_path: Path,
) -> None:
    with np.load(source, allow_pickle=False) as raw:
        episode_id = np.asarray(raw["episode_id"])
        frame_id = np.asarray(raw["frame_id"])
    target = np.zeros((len(frame_id), 16), dtype=np.float32)
    directory.mkdir()
    target_path = directory / "oracle_target.npy"
    np.save(target_path, target)
    encoder_checkpoint = torch.load(
        encoder_path, map_location="cpu", weights_only=False
    )
    metadata = {
        "cache_type": "fcpg_oracle_target_v1",
        "source_data": str(source.resolve()),
        "source_data_sha256": sha256_file(source),
        "sample_count": len(frame_id),
        "target_dim": 16,
        "encoder_checkpoint": str(encoder_path.resolve()),
        "encoder_checkpoint_sha256": sha256_file(encoder_path),
        "encoder_spec": encoder_checkpoint["spec"],
        "frame_identity_sha256": frame_identity_sha256(episode_id, frame_id),
        "oracle_target_sha256": sha256_file(target_path),
        "q_true_usage": "target_construction_only",
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )


def test_multisource_episode_split_and_balanced_sampler(tmp_path, monkeypatch) -> None:
    sources = [tmp_path / "pick.npz", tmp_path / "button.npz"]
    _write_source(sources[0], "pick-place-v3", 1)
    _write_source(sources[1], "button-press-v3", 2)
    config = _config(tmp_path, sources)

    bundle = runtime.build_lewm_data(config)

    assert isinstance(bundle.train_loader.sampler, WeightedRandomSampler)
    lengths = [len(source) for source in bundle.train_sources]
    weights = bundle.train_loader.sampler.weights
    first_total = float(weights[: lengths[0]].sum())
    second_total = float(weights[lengths[0] :].sum())
    assert first_total == second_total == 1.0
    for source in bundle.data_config["sources"]:
        train = set(source["train_episodes"])
        val = set(source["val_episodes"])
        assert train and val and train.isdisjoint(val)
    normalization = bundle.data_config["normalization"]
    assert normalization["enabled"] is True
    assert len(normalization["proprio_mean"]) == 4
    assert len(normalization["action_mean"]) == 2
    assert all(value > 0.0 for value in normalization["action_std"])

    monkeypatch.setattr(runtime, "LeWMViTDenseEncoder", _TinyDenseVision)
    model = runtime.build_grounded_lewm(config, bundle.reference_dataset)
    assert model.history_size == 2
    assert model.action_dim == 2
    assert model.action_encoder.patch_embed.out_channels == 10


def test_camera_manifest_selects_exact_episodes_and_excludes_strong_drift(
    tmp_path,
) -> None:
    sources = [tmp_path / "pick.npz", tmp_path / "button.npz"]
    _write_source(sources[0], "pick-place-v3", 11)
    _write_source(sources[1], "button-press-v3", 12)
    manifest = tmp_path / "camera_split.json"
    _write_episode_manifest(manifest, sources)
    config = _config(tmp_path, sources)
    config["data"]["episode_manifest"] = str(manifest)

    bundle = runtime.build_lewm_data(config)

    for source in bundle.data_config["sources"]:
        task = "pick-place-v3" if source["name"] == "pick" else "button-press-v3"
        assert source["train_episodes"] == [f"{task}:0", f"{task}:1"]
        assert source["val_episodes"] == [f"{task}:2"]
        assert f"{task}:3" not in source["train_episodes"]
        assert f"{task}:3" not in source["val_episodes"]


def test_camera_manifest_rejects_overlap_and_source_hash_mismatch(tmp_path) -> None:
    source = tmp_path / "reach.npz"
    _write_source(source, "reach-v3", 13)
    config = _config(tmp_path, [source])

    overlap_manifest = tmp_path / "overlap.json"
    _write_episode_manifest(overlap_manifest, [source], overlap=True)
    config["data"]["episode_manifest"] = str(overlap_manifest)
    with pytest.raises(ValueError, match="train/val/excluded overlap"):
        runtime.build_lewm_data(config)

    valid_manifest = tmp_path / "valid.json"
    _write_episode_manifest(valid_manifest, [source])
    payload = json.loads(valid_manifest.read_text(encoding="utf-8"))
    payload["sources"][source.stem]["data_sha256"] = "0" * 64
    valid_manifest.write_text(json.dumps(payload), encoding="utf-8")
    config["data"]["episode_manifest"] = str(valid_manifest)
    with pytest.raises(ValueError, match="source hash mismatch"):
        runtime.build_lewm_data(config)


def test_frozen_oracle_runtime_requires_one_shared_encoder(tmp_path, monkeypatch) -> None:
    sources = [tmp_path / "pick.npz", tmp_path / "button.npz"]
    _write_source(sources[0], "pick-place-v3", 21)
    _write_source(sources[1], "button-press-v3", 22)
    encoder_path = tmp_path / "oracle_encoder.pt"
    save_frozen_oracle_target_encoder(
        encoder_path,
        FrozenOracleTargetEncoder(visual_dim=8, num_keypoints=3, target_dim=16),
    )
    caches = [tmp_path / "pick_target", tmp_path / "button_target"]
    for source, cache in zip(sources, caches, strict=True):
        _write_oracle_cache(source, cache, encoder_path)

    config = _config(tmp_path, sources)
    config["data"]["target_mode"] = "frozen_oracle"
    config["loss"]["alignment_weight"] = 0.5
    for entry, cache in zip(config["data"]["sources"], caches, strict=True):
        entry["oracle_target_cache"] = str(cache)

    bundle = runtime.build_lewm_data(config)
    contract = bundle.data_config["target_contract"]
    assert contract["mode"] == "frozen_oracle"
    assert contract["target_dim"] == 16
    assert contract["encoder_checkpoint_sha256"] == sha256_file(encoder_path)
    assert "q_true" not in bundle.reference_dataset[0]

    monkeypatch.setattr(runtime, "LeWMViTDenseEncoder", _TinyDenseVision)
    model = runtime.build_grounded_lewm(config, bundle.reference_dataset)
    assert model.target_mode == "frozen_oracle"
    assert model.alignment_weight == 0.5

    other_encoder = tmp_path / "other_encoder.pt"
    save_frozen_oracle_target_encoder(
        other_encoder,
        FrozenOracleTargetEncoder(
            visual_dim=8,
            num_keypoints=3,
            target_dim=16,
            projection_seed=99,
        ),
    )
    _write_oracle_cache(sources[1], tmp_path / "other_target", other_encoder)
    config["data"]["sources"][1]["oracle_target_cache"] = str(
        tmp_path / "other_target"
    )
    with pytest.raises(ValueError, match="share one frozen oracle target encoder"):
        runtime.build_lewm_data(config)


def test_smoke_training_diagnostics_checkpoint_and_resume(tmp_path, monkeypatch) -> None:
    sources = [tmp_path / "reach.npz", tmp_path / "push.npz"]
    _write_source(sources[0], "reach-v3", 3)
    _write_source(sources[1], "push-v3", 4)
    config = _config(tmp_path, sources)
    monkeypatch.setattr(runtime, "LeWMViTDenseEncoder", _TinyDenseVision)

    result = runtime.train_lewm_grounded(
        config,
        device="cpu",
        smoke_steps=1,
    )

    assert result.global_step == 1
    assert result.latest_checkpoint.is_file()
    assert result.best_checkpoint.is_file()
    epoch_one = result.output_dir / "lewm_grounded_epoch_001.pt"
    assert epoch_one.is_file()
    assert epoch_one.stat().st_ino == result.latest_checkpoint.stat().st_ino
    records = [json.loads(line) for line in result.metrics_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["objective"] == "prediction_mse_plus_lambda_sigreg"
    assert {
        "pred_mse",
        "copy_mse",
        "action_shuffle_mse",
        "action_shuffle_delta",
    }.issubset(records[0]["val"])

    bundle = runtime.build_lewm_data(config)
    model, resolved, checkpoint = runtime.load_lewm_grounded_checkpoint(
        result.latest_checkpoint,
        device="cpu",
        dataset=bundle.reference_dataset,
    )
    assert model.training is False
    assert resolved["data"]["history_size"] == 2
    assert checkpoint["checkpoint_type"] == runtime.CHECKPOINT_TYPE
    assert checkpoint["upstream_commit"] == LEWM_UPSTREAM_COMMIT
    assert checkpoint["step"] == 1
    assert checkpoint["run_kind"] == "smoke"
    assert checkpoint["training_budget"] == {
        "target_optimizer_steps": 1,
        "checkpoint_optimizer_step": 1,
        "run_completed": True,
        "final_optimizer_steps": 1,
    }
    assert checkpoint["optimizer_state"] is not None
    torch.testing.assert_close(
        model.action_mean,
        torch.tensor(checkpoint["normalization"]["action_mean"]),
    )
    assert checkpoint["data_config"]["grounding_mode"] == "nominal"
    assert checkpoint["data_config"]["normalization"]["enabled"] is True
    assert checkpoint["scheduler_state"] is not None
    assert checkpoint["resolved_config"]["loss"]["sigreg_weight"] == 0.01

    resumed = runtime.train_lewm_grounded(
        config,
        device="cpu",
        smoke_steps=2,
        resume=result.latest_checkpoint,
    )
    assert resumed.global_step == 2
    epoch_two = resumed.output_dir / "lewm_grounded_epoch_002.pt"
    assert epoch_two.is_file()
    assert epoch_two.stat().st_ino == resumed.latest_checkpoint.stat().st_ino
    assert epoch_one.stat().st_ino != resumed.latest_checkpoint.stat().st_ino
    resumed_checkpoint = torch.load(
        resumed.latest_checkpoint, map_location="cpu", weights_only=False
    )
    assert resumed_checkpoint["training_budget"] == {
        "target_optimizer_steps": 2,
        "checkpoint_optimizer_step": 2,
        "run_completed": True,
        "final_optimizer_steps": 2,
    }
    with pytest.raises(FileExistsError, match="immutable checkpoint"):
        runtime._replace_with_hardlink(
            resumed.latest_checkpoint,
            epoch_one,
            refuse_existing=True,
        )
    resumed_records = [
        json.loads(line) for line in resumed.metrics_path.read_text().splitlines()
    ]
    assert [record["step"] for record in resumed_records] == [1, 2]

    latest_inode = resumed.latest_checkpoint.stat().st_ino
    with pytest.raises(ValueError, match="mutable latest checkpoint alias"):
        runtime.train_lewm_grounded(
            config,
            device="cpu",
            smoke_steps=3,
            resume=epoch_one,
        )
    assert resumed.latest_checkpoint.stat().st_ino == latest_inode
    assert resumed.metrics_path.read_text().splitlines() == [
        json.dumps(record, sort_keys=True) for record in resumed_records
    ]

    with pytest.raises(ValueError, match="not interchangeable"):
        runtime.train_lewm_grounded(
            config,
            device="cpu",
            resume=resumed.latest_checkpoint,
        )


@torch.no_grad()
def test_restore_rng_converts_mapped_generator_states_to_cpu(monkeypatch) -> None:
    captured: dict[str, object] = {}
    state = runtime._capture_rng()

    monkeypatch.setattr(
        torch,
        "set_rng_state",
        lambda value: captured.setdefault("torch", value),
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda value: captured.setdefault("cuda", value),
    )

    runtime._restore_rng(state)

    assert isinstance(captured["torch"], torch.Tensor)
    assert captured["torch"].device.type == "cpu"
    if torch.cuda.is_available():
        assert all(value.device.type == "cpu" for value in captured["cuda"])



def test_new_run_refuses_nonempty_output_directory(tmp_path) -> None:
    source = tmp_path / "reach.npz"
    _write_source(source, "reach-v3", 5)
    config = _config(tmp_path, [source])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True)
    (output_dir / "existing.txt").write_text("occupied")

    with pytest.raises(FileExistsError, match="non-empty output directory"):
        runtime.train_lewm_grounded(config, device="cpu", smoke_steps=1)


def test_resume_requires_original_output_directory(tmp_path, monkeypatch) -> None:
    source = tmp_path / "reach.npz"
    _write_source(source, "reach-v3", 6)
    config = _config(tmp_path, [source])
    monkeypatch.setattr(runtime, "LeWMViTDenseEncoder", _TinyDenseVision)
    result = runtime.train_lewm_grounded(config, device="cpu", smoke_steps=1)

    moved = dict(config)
    moved["output_dir"] = str(tmp_path / "different-run")
    with pytest.raises(ValueError, match="original output directory"):
        runtime.train_lewm_grounded(
            moved,
            device="cpu",
            smoke_steps=2,
            resume=result.latest_checkpoint,
        )


def test_run_lock_is_exclusive(tmp_path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    with runtime._exclusive_run_lock(output_dir):
        with pytest.raises(RuntimeError, match="holds the run lock"):
            with runtime._exclusive_run_lock(output_dir):
                pytest.fail("nested lock unexpectedly succeeded")


def test_core_nonfinite_metrics_abort_but_optional_shuffle_nan_is_allowed() -> None:
    with pytest.raises(FloatingPointError, match="metrics.pred_mse"):
        runtime._assert_finite_metrics({"pred_mse": float("nan")})

    runtime._assert_finite_metrics({"action_shuffle_mse": float("nan")})


def test_metrics_failure_is_reconciled_from_committed_checkpoint(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "reach.npz"
    _write_source(source, "reach-v3", 7)
    config = _config(tmp_path, [source])
    monkeypatch.setattr(runtime, "LeWMViTDenseEncoder", _TinyDenseVision)
    original_write = runtime._write_metrics_records

    def fail_metrics(*_args, **_kwargs):
        raise OSError("injected metrics failure")

    monkeypatch.setattr(runtime, "_write_metrics_records", fail_metrics)
    with pytest.raises(OSError, match="injected metrics failure"):
        runtime.train_lewm_grounded(config, device="cpu", smoke_steps=1)

    output_dir = Path(config["output_dir"])
    latest = output_dir / "lewm_grounded_latest.pt"
    assert latest.is_file()
    assert (output_dir / "lewm_grounded_epoch_001.pt").is_file()
    assert not (output_dir / "metrics.jsonl").exists()

    monkeypatch.setattr(runtime, "_write_metrics_records", original_write)
    resumed = runtime.train_lewm_grounded(
        config,
        device="cpu",
        smoke_steps=2,
        resume=latest,
    )

    records = [
        json.loads(line) for line in resumed.metrics_path.read_text().splitlines()
    ]
    assert [record["epoch"] for record in records] == [1, 2]
    assert resumed.best_checkpoint.is_file()

    epoch_one = output_dir / "lewm_grounded_epoch_001.pt"
    runtime._replace_with_hardlink(
        epoch_one,
        resumed.latest_checkpoint,
        refuse_existing=False,
    )
    recovered = runtime.train_lewm_grounded(
        config,
        device="cpu",
        smoke_steps=3,
        resume=resumed.latest_checkpoint,
    )
    recovered_records = [
        json.loads(line) for line in recovered.metrics_path.read_text().splitlines()
    ]
    assert [record["epoch"] for record in recovered_records] == [1, 2]
    epoch_two = output_dir / "lewm_grounded_epoch_002.pt"
    assert recovered.latest_checkpoint.stat().st_ino == epoch_two.stat().st_ino


def test_bootstrap_only_directory_can_restart_without_resume(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "reach.npz"
    _write_source(source, "reach-v3", 8)
    config = _config(tmp_path, [source])
    monkeypatch.setattr(runtime, "LeWMViTDenseEncoder", _TinyDenseVision)
    original_evaluate = runtime.evaluate_lewm

    def fail_before_checkpoint(*_args, **_kwargs):
        raise RuntimeError("injected validation failure")

    monkeypatch.setattr(runtime, "evaluate_lewm", fail_before_checkpoint)
    with pytest.raises(RuntimeError, match="injected validation failure"):
        runtime.train_lewm_grounded(config, device="cpu", smoke_steps=1)

    output_dir = Path(config["output_dir"])
    assert (output_dir / "resolved_config.json").is_file()
    assert not (output_dir / "lewm_grounded_latest.pt").exists()

    monkeypatch.setattr(runtime, "evaluate_lewm", original_evaluate)
    restarted = runtime.train_lewm_grounded(
        config,
        device="cpu",
        smoke_steps=1,
    )
    assert restarted.latest_checkpoint.is_file()
