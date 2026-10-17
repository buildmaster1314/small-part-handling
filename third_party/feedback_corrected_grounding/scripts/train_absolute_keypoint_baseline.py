#!/usr/bin/env python3
"""Train the matched-input, direct-absolute keypoint baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.absolute_keypoint_baseline import (  # noqa: E402
    BASELINE_VARIANT,
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_TYPE,
    AnchorConditionedAbsoluteKeypointViTAdapter,
    absolute_keypoint_contract,
    compute_absolute_keypoint_loss,
    inspect_reference_fcpg_checkpoint,
    load_absolute_keypoint_config,
    load_frozen_backbone_from_reference,
    model_kwargs_from_config,
    validate_deployment_vision_cache,
    validate_reference_training_protocol,
)
from fcpg.lewm_provenance import sha256_file  # noqa: E402
from fcpg.observations import select_policy_observation  # noqa: E402
from fcpg.sequence_data import (  # noqa: E402
    HomogeneousConcatDataset,
    make_balanced_source_sampler,
    select_episode_subset,
)
from scripts.train_bc_policy import (  # noqa: E402
    SplitIndices,
    make_split,
    require_pre_action_alignment,
)


DEFAULT_CONFIG = "config/absolute_keypoint/anchor_conditioned_shared_4task_seed0.json"


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


class AbsoluteKeypointDataset(Dataset):
    """Strict matched-input dataset with q_true isolated as a target field."""

    REQUIRED_FIELDS = {
        "obs",
        "q_nominal",
        "q_true",
        "keypoints_xyz",
        "valid_true",
        "valid_nominal",
        "task_name",
        "episode_id",
    }

    def __init__(
        self,
        data_path: str | Path,
        vision_cache_info: Mapping[str, object],
        indices: np.ndarray,
        *,
        expected_obs_dim: int,
        expected_num_keypoints: int,
    ) -> None:
        self.path = Path(data_path).resolve()
        self.data = np.load(self.path, allow_pickle=False)
        missing = sorted(self.REQUIRED_FIELDS.difference(self.data.files))
        if missing:
            raise ValueError(f"absolute-keypoint dataset missing fields: {missing}")
        self.indices = np.asarray(indices, dtype=np.int64)
        if self.indices.ndim != 1 or self.indices.size == 0:
            raise ValueError("absolute-keypoint dataset indices must be non-empty 1-D")

        self.obs = select_policy_observation(self.data["obs"], "robot")
        self.q_nominal = np.asarray(self.data["q_nominal"], dtype=np.float32)
        self.q_true = np.asarray(self.data["q_true"], dtype=np.float32)
        self.keypoints_xyz_fk = np.asarray(
            self.data["keypoints_xyz"], dtype=np.float32
        )
        valid_true = np.asarray(self.data["valid_true"])
        valid_nominal = np.asarray(self.data["valid_nominal"])
        if valid_true.dtype != np.bool_ or valid_nominal.dtype != np.bool_:
            raise ValueError("valid_true and valid_nominal must use bool dtype")
        self.valid = valid_true & valid_nominal
        sample_count = int(self.obs.shape[0])
        expected_q_shape = (sample_count, expected_num_keypoints, 2)
        if self.q_nominal.shape != expected_q_shape:
            raise ValueError(
                f"q_nominal must have shape {expected_q_shape}, got {self.q_nominal.shape}"
            )
        if self.q_true.shape != expected_q_shape:
            raise ValueError(
                f"q_true must have shape {expected_q_shape}, got {self.q_true.shape}"
            )
        if self.keypoints_xyz_fk.shape != (
            sample_count,
            expected_num_keypoints,
            3,
        ):
            raise ValueError("keypoints_xyz must have matched shape [N, K, 3]")
        if self.valid.shape != expected_q_shape[:2]:
            raise ValueError("valid masks must have matched shape [N, K]")
        if self.obs.shape != (sample_count, expected_obs_dim):
            raise ValueError(
                f"robot obs must have shape {(sample_count, expected_obs_dim)}, "
                f"got {self.obs.shape}"
            )
        if int(self.indices.min()) < 0 or int(self.indices.max()) >= sample_count:
            raise ValueError("absolute-keypoint indices are out of range")
        selected_valid = self.valid[self.indices]
        if not bool(selected_valid.any()):
            raise ValueError("absolute-keypoint split has no valid q_true labels")
        if not bool(np.isfinite(self.q_true[self.indices][selected_valid]).all()):
            raise ValueError("valid q_true labels contain non-finite values")
        if not bool(np.isfinite(self.q_nominal[self.indices]).all()):
            raise ValueError("q_nominal model inputs contain non-finite values")
        if not bool(np.isfinite(self.keypoints_xyz_fk[self.indices]).all()):
            raise ValueError("FK XYZ model inputs contain non-finite values")
        if not bool(np.isfinite(self.obs[self.indices]).all()):
            raise ValueError("robot obs model inputs contain non-finite values")

        self.image_size = tuple(int(v) for v in vision_cache_info["image_size"])
        self.vision_backbone_features = np.load(
            str(vision_cache_info["feature_path"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.vision_prefix = np.load(
            str(vision_cache_info["prefix_path"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        if self.vision_backbone_features.shape[0] != sample_count:
            raise ValueError("vision feature count does not match dataset")
        if self.vision_prefix.shape[0] != sample_count:
            raise ValueError("vision prefix count does not match dataset")
        self.obs_dim = expected_obs_dim
        self.num_keypoints = expected_num_keypoints

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        return {
            # Model inputs.
            "vision_backbone_features": torch.from_numpy(
                np.array(
                    self.vision_backbone_features[index],
                    dtype=np.float16,
                    copy=True,
                )
            ),
            "vision_prefix": torch.from_numpy(
                np.array(self.vision_prefix[index], dtype=np.float16, copy=True)
            ),
            "q_nominal": torch.from_numpy(self.q_nominal[index]),
            "keypoints_xyz_fk": torch.from_numpy(self.keypoints_xyz_fk[index]),
            "robot_obs": torch.from_numpy(self.obs[index]),
            "image_size": torch.tensor(self.image_size, dtype=torch.long),
            # Supervision only.  These fields are never forwarded to the model.
            "target_q_true": torch.from_numpy(self.q_true[index]),
            "target_valid": torch.from_numpy(self.valid[index]),
        }


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device, non_blocking=device.type == "cuda")
        for name, value in batch.items()
    }


def predict_from_batch(
    model: AnchorConditionedAbsoluteKeypointViTAdapter,
    batch: Mapping[str, torch.Tensor],
):
    """Forward only declared SAME-INPUT fields; q_true is inaccessible here."""

    image_sizes = batch["image_size"]
    if not bool(torch.all(image_sizes == image_sizes[0])):
        raise ValueError("all samples in a batch must share one image size")
    image_size = tuple(int(v) for v in image_sizes[0].tolist())
    return model.forward_from_backbone_features(
        batch["vision_backbone_features"],
        batch["vision_prefix"],
        batch["q_nominal"],
        batch["keypoints_xyz_fk"],
        batch["robot_obs"],
        image_size=image_size,
    )


def run_epoch(
    model: AnchorConditionedAbsoluteKeypointViTAdapter,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    grad_clip: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    valid_count = 0
    loss_sum = 0.0
    before_sum = 0.0
    after_sum = 0.0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = predict_from_batch(model, batch)
                loss, _ = compute_absolute_keypoint_loss(
                    output,
                    batch["target_q_true"],
                    batch["target_valid"],
                )
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=grad_clip,
                )
                scaler.step(optimizer)
                scaler.update()

            valid = batch["target_valid"]
            count = int(valid.sum())
            before = torch.linalg.vector_norm(
                batch["q_nominal"] - batch["target_q_true"], dim=-1
            )
            after = torch.linalg.vector_norm(
                output.q_absolute_px - batch["target_q_true"], dim=-1
            )
            valid_count += count
            loss_sum += float(loss.detach()) * count
            before_sum += float(before[valid].sum())
            after_sum += float(after[valid].sum())
    if valid_count <= 0:
        raise RuntimeError("epoch produced no valid q_true labels")
    return {
        "loss": loss_sum / valid_count,
        "e_before": before_sum / valid_count,
        "e_after": after_sum / valid_count,
        "ratio": after_sum / max(before_sum, 1e-8),
        "valid_keypoints": float(valid_count),
    }


def cosine_warmup_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(warmup_steps, 1))
    progress = float(step - warmup_steps) / float(
        max(total_steps - warmup_steps, 1)
    )
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_json_write(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def checkpoint_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args(argv)

    config_path = resolve_repo_path(args.config)
    config = load_absolute_keypoint_config(config_path)
    model_config = config["model"]
    assert isinstance(model_config, Mapping)
    reference_config = config["reference_fcpg"]
    assert isinstance(reference_config, Mapping)
    reference_path = resolve_repo_path(str(reference_config["checkpoint"]))
    reference_report = inspect_reference_fcpg_checkpoint(
        reference_path, model_config
    )
    if (
        reference_report["trainable_parameters"]
        != reference_config["trainable_parameters"]
    ):
        parser.error("reference parameter count differs from config")

    budget = config["budget"]
    split_config = config["split"]
    optimization = config["optimization"]
    assert isinstance(budget, Mapping)
    assert isinstance(split_config, Mapping)
    assert isinstance(optimization, Mapping)
    seed = int(optimization["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)

    source_specs = config["data_sources"]
    assert isinstance(source_specs, list)
    cache_infos: list[dict[str, object]] = []
    source_splits: list[SplitIndices] = []
    source_paths: list[Path] = []
    alignments: list[str | None] = []
    for source in source_specs:
        assert isinstance(source, Mapping)
        data_path = resolve_repo_path(str(source["path"]))
        cache_path = resolve_repo_path(str(source["vision_cache"]))
        cache_info = validate_deployment_vision_cache(
            data_path,
            cache_path,
            model_config,
            expected_weight_source=reference_path,
        )
        with np.load(data_path, allow_pickle=False) as raw:
            alignments.append(
                require_pre_action_alignment(raw, data_path, allow_unaligned=False)
            )
            split = make_split(
                raw,
                float(split_config["val_fraction"]),
                int(split_config["seed"]),
                str(split_config["unit"]),
            )
            max_episodes = int(budget["max_train_episodes_per_source"])
            if max_episodes > 0:
                split = SplitIndices(
                    train=select_episode_subset(
                        raw,
                        split.train,
                        max_episodes,
                        int(budget["episode_selection_seed"]),
                    ),
                    test=split.test,
                )
        source_paths.append(data_path)
        cache_infos.append(cache_info)
        source_splits.append(split)
    validate_reference_training_protocol(reference_report, config, source_paths)

    obs_dim = int(model_config["obs_dim"])
    num_keypoints = int(model_config["num_keypoints"])
    train_sources = [
        AbsoluteKeypointDataset(
            path,
            cache_info,
            split.train,
            expected_obs_dim=obs_dim,
            expected_num_keypoints=num_keypoints,
        )
        for path, cache_info, split in zip(
            source_paths, cache_infos, source_splits, strict=True
        )
    ]
    val_sources = [
        AbsoluteKeypointDataset(
            path,
            cache_info,
            split.test,
            expected_obs_dim=obs_dim,
            expected_num_keypoints=num_keypoints,
        )
        for path, cache_info, split in zip(
            source_paths, cache_infos, source_splits, strict=True
        )
    ]
    if len(train_sources) == 1:
        train_dataset: Dataset = train_sources[0]
        val_dataset: Dataset = val_sources[0]
    else:
        attributes = ("obs_dim", "num_keypoints", "image_size")
        train_dataset = HomogeneousConcatDataset(train_sources, attributes)
        val_dataset = HomogeneousConcatDataset(val_sources, attributes)
    train_sampler = (
        make_balanced_source_sampler(train_sources, seed)
        if len(train_sources) > 1 and bool(budget["balance_sources"])
        else None
    )
    loader_kwargs = {
        "num_workers": int(budget["num_workers"]),
        "pin_memory": device.type == "cuda",
        "persistent_workers": int(budget["num_workers"]) > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(budget["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(budget["batch_size"]),
        shuffle=False,
        **loader_kwargs,
    )

    model = AnchorConditionedAbsoluteKeypointViTAdapter(
        **model_kwargs_from_config(model_config), pretrained=False
    )
    load_frozen_backbone_from_reference(model, reference_path)
    parameter_report = model.parameter_report()
    if parameter_report["exact_target_match"] is not True:
        raise RuntimeError("absolute baseline does not match FCPG parameter budget")
    model = model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(optimization["lr"]),
        betas=(0.95, 0.999),
        weight_decay=float(optimization["weight_decay"]),
    )
    epochs = int(budget["epochs"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup_lambda(
            step,
            int(round(float(optimization["warmup_epochs"]))),
            epochs,
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    source_manifest = [
        {
            "name": str(source["name"]),
            "data": str(path),
            "data_sha256": str(cache_info["source_data_sha256"]),
            "vision_cache": str(cache_info["cache_path"]),
            "vision_metadata_sha256": sha256_file(cache_info["metadata_path"]),
            "alignment": alignment,
            "train_frames": len(train_source),
            "val_frames": len(val_source),
        }
        for source, path, cache_info, alignment, train_source, val_source in zip(
            source_specs,
            source_paths,
            cache_infos,
            alignments,
            train_sources,
            val_sources,
            strict=True,
        )
    ]
    output_dir = (
        resolve_repo_path(args.output_dir)
        if args.output_dir
        else resolve_repo_path(str(config["output_dir"]))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "absolute_keypoint_best.pt"
    final_path = output_dir / "absolute_keypoint_final.pt"
    metrics_path = output_dir / "metrics.json"
    provenance = {
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "reference_fcpg": reference_report,
        "data_sources": source_manifest,
        "split": dict(split_config),
        "budget": dict(budget),
        "q_true_usage": "training_supervision_and_offline_metrics_only",
    }
    sampling_report = model.sampling_report()
    print(json.dumps({
        "device": str(device),
        "variant": BASELINE_VARIANT,
        "parameter_report": parameter_report,
        "sampling_report": sampling_report,
        "sources": source_manifest,
    }, indent=2))

    history: list[dict[str, float | int]] = []
    best_metrics: dict[str, float | int] | None = None
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            scaler,
            float(optimization["grad_clip"]),
        )
        scheduler.step()
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            None,
            scaler,
            float(optimization["grad_clip"]),
        )
        row: dict[str, float | int] = {"epoch": epoch}
        row.update({f"train_{name}": value for name, value in train_metrics.items()})
        row.update({f"val_{name}": value for name, value in val_metrics.items()})
        row["lr"] = float(optimizer.param_groups[0]["lr"])
        history.append(row)
        print(json.dumps(row), flush=True)
        if best_metrics is None or float(row["val_e_after"]) < float(
            best_metrics["val_e_after"]
        ):
            best_metrics = dict(row)
            atomic_torch_save(
                {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "checkpoint_type": CHECKPOINT_TYPE,
                    "variant": BASELINE_VARIANT,
                    "contract": absolute_keypoint_contract(),
                    "model_config": dict(model_config),
                    "model_state": checkpoint_model_state(model),
                    "parameter_report": parameter_report,
                    "sampling_report": sampling_report,
                    "training_provenance": provenance,
                    "epoch": epoch,
                    "metrics": dict(row),
                },
                best_path,
            )
        atomic_json_write(
            {
                "schema_version": 1,
                "variant": BASELINE_VARIANT,
                "parameter_report": parameter_report,
                "sampling_report": sampling_report,
                "best": best_metrics,
                "history": history,
                "complete": False,
            },
            metrics_path,
        )

    assert best_metrics is not None
    atomic_torch_save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_type": CHECKPOINT_TYPE,
            "variant": BASELINE_VARIANT,
            "contract": absolute_keypoint_contract(),
            "model_config": dict(model_config),
            "model_state": checkpoint_model_state(model),
            "parameter_report": parameter_report,
            "sampling_report": sampling_report,
            "training_provenance": provenance,
            "epoch": epochs,
            "metrics": history[-1],
        },
        final_path,
    )
    atomic_json_write(
        {
            "schema_version": 1,
            "variant": BASELINE_VARIANT,
            "parameter_report": parameter_report,
            "sampling_report": sampling_report,
            "best": best_metrics,
            "history": history,
            "complete": True,
        },
        metrics_path,
    )
    print(f"best_checkpoint: {best_path}")
    print(f"final_checkpoint: {final_path}")
    print(f"metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
