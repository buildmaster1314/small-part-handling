"""Matched-input absolute-keypoint baseline for FCPG experiments.

The primary baseline deliberately matches the supervised FCPG corrector's
inputs and local visual-token budget.  Its only substantive change is the
output parameterization: the head predicts absolute image coordinates instead
of a residual that is added to ``q_nominal``.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fcpg.lewm_core import LeWMTransformer
from fcpg.lewm_provenance import sha256_file
from fcpg.transformer_grounding import (
    DEFAULT_VIT_MODEL,
    DenseVisionOutput,
    ViTDenseEncoder,
    sample_local_feature_tokens,
)


CHECKPOINT_SCHEMA_VERSION = 1
CONFIG_SCHEMA_VERSION = 1
CHECKPOINT_TYPE = "absolute_keypoint_baseline"
BASELINE_VARIANT = "anchor_conditioned_absolute_keypoint"
COORDINATE_PARAMETERIZATION = "direct_absolute_tanh_xy_w_minus_1_h_minus_1"
REFERENCE_FCPG_ARCH = "lewm"
REFERENCE_FCPG_FROZEN_PREFIX = "vision.backbone."


def absolute_keypoint_contract() -> dict[str, object]:
    """Return the immutable scientific input/target contract."""

    return {
        "variant": BASELINE_VARIANT,
        "model_inputs": [
            "deployment_rgb_or_frozen_vit_features",
            "q_nominal_px",
            "keypoints_xyz_fk",
            "robot_obs",
            "keypoint_identity",
        ],
        "forbidden_model_inputs": [
            "q_true",
            "true_camera_extrinsic",
            "true_camera_intrinsic",
        ],
        "supervision": "q_true_absolute_px_only",
        "test_q_true_usage": "metrics_only_never_model_input",
        "coordinate_parameterization": COORDINATE_PARAMETERIZATION,
        "visual_sampling": "q_nominal_centered_square_grid_matching_fcpg",
        "reliability_output": False,
    }


@dataclass(frozen=True)
class AbsoluteKeypointOutput:
    """Direct absolute-coordinate output used by caches and action policies."""

    q_absolute_norm: torch.Tensor
    q_absolute_px: torch.Tensor
    keypoint_latent: torch.Tensor
    feature_map: torch.Tensor | None = None
    global_feature: torch.Tensor | None = None

    @property
    def q_corrected_px(self) -> torch.Tensor:
        """Compatibility alias for existing FCPG correction consumers."""

        return self.q_absolute_px


class AnchorConditionedAbsoluteKeypointHead(nn.Module):
    """FCPG-matched local transformer with a direct absolute-coordinate head."""

    def __init__(
        self,
        visual_dim: int,
        obs_dim: int,
        num_keypoints: int,
        model_dim: int = 256,
        local_grid_size: int = 5,
        local_radius_px: float = 24.0,
        local_layers: int = 4,
        global_layers: int = 2,
        heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if visual_dim <= 0 or obs_dim <= 0 or num_keypoints <= 0 or model_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if model_dim % heads != 0:
            raise ValueError(f"model_dim={model_dim} must be divisible by heads={heads}")
        if local_grid_size <= 0 or local_grid_size % 2 == 0:
            raise ValueError("local_grid_size must be a positive odd integer")
        if local_radius_px <= 0:
            raise ValueError("local_radius_px must be positive")
        if local_layers <= 0 or global_layers <= 0:
            raise ValueError("transformer layer counts must be positive")

        self.visual_dim = int(visual_dim)
        self.obs_dim = int(obs_dim)
        self.num_keypoints = int(num_keypoints)
        self.model_dim = int(model_dim)
        self.local_grid_size = int(local_grid_size)
        self.local_radius_px = float(local_radius_px)

        # This is intentionally identical in shape to the supervised FCPG query.
        query_input_dim = 2 + 3 + self.num_keypoints + self.obs_dim
        self.query_projection = nn.Sequential(
            nn.LayerNorm(query_input_dim),
            nn.Linear(query_input_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.visual_projection = nn.Linear(self.visual_dim, self.model_dim)
        self.relative_position = nn.Sequential(
            nn.Linear(2, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        mlp_dim = self.model_dim * int(mlp_ratio)
        self.local_transformer = LeWMTransformer(
            input_dim=self.model_dim,
            hidden_dim=self.model_dim,
            output_dim=self.model_dim,
            depth=int(local_layers),
            heads=int(heads),
            mlp_dim=mlp_dim,
            dropout=dropout,
            causal=False,
            conditional=True,
            condition_dim=self.model_dim,
        )
        self.keypoint_transformer = LeWMTransformer(
            input_dim=self.model_dim,
            hidden_dim=self.model_dim,
            output_dim=self.model_dim,
            depth=int(global_layers),
            heads=int(heads),
            mlp_dim=mlp_dim,
            dropout=dropout,
            causal=False,
            conditional=True,
            condition_dim=self.model_dim,
        )
        self.absolute_head = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, 2),
        )
        # Zero logits initialize direct predictions at image center.  No nominal
        # coordinate is added after this head.
        nn.init.zeros_(self.absolute_head[-1].weight)
        nn.init.zeros_(self.absolute_head[-1].bias)

    @property
    def sampled_visual_tokens_per_keypoint(self) -> int:
        return self.local_grid_size**2

    def _query_vectors(
        self,
        q_nominal_px: torch.Tensor,
        keypoints_xyz_fk: torch.Tensor,
        robot_obs: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        if q_nominal_px.ndim != 3 or q_nominal_px.shape[-1] != 2:
            raise ValueError(
                f"q_nominal_px must have shape [B, K, 2], got {tuple(q_nominal_px.shape)}"
            )
        batch, num_keypoints, _ = q_nominal_px.shape
        if num_keypoints != self.num_keypoints:
            raise ValueError(f"expected {self.num_keypoints} keypoints, got {num_keypoints}")
        if keypoints_xyz_fk.shape != (batch, num_keypoints, 3):
            raise ValueError(
                "keypoints_xyz_fk must have shape "
                f"{(batch, num_keypoints, 3)}, got {tuple(keypoints_xyz_fk.shape)}"
            )
        if robot_obs.shape != (batch, self.obs_dim):
            raise ValueError(
                f"robot_obs must have shape {(batch, self.obs_dim)}, "
                f"got {tuple(robot_obs.shape)}"
            )
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        for name, value in (
            ("q_nominal_px", q_nominal_px),
            ("keypoints_xyz_fk", keypoints_xyz_fk),
            ("robot_obs", robot_obs),
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains non-finite values")
        q_norm = torch.stack(
            [
                q_nominal_px[..., 0] / float(width),
                q_nominal_px[..., 1] / float(height),
            ],
            dim=-1,
        )
        keypoint_identity = torch.eye(
            self.num_keypoints,
            device=q_nominal_px.device,
            dtype=q_nominal_px.dtype,
        ).unsqueeze(0).expand(batch, -1, -1)
        obs = robot_obs[:, None, :].expand(-1, self.num_keypoints, -1)
        return torch.cat(
            [q_norm, keypoints_xyz_fk, keypoint_identity, obs], dim=-1
        )

    def forward(
        self,
        feature_map: torch.Tensor,
        q_nominal_px: torch.Tensor,
        keypoints_xyz_fk: torch.Tensor,
        robot_obs: torch.Tensor,
        image_size: tuple[int, int],
    ) -> AbsoluteKeypointOutput:
        local_visual, relative_offsets = sample_local_feature_tokens(
            feature_map,
            q_nominal_px,
            image_size=image_size,
            radius_px=self.local_radius_px,
            grid_size=self.local_grid_size,
        )
        query = self.query_projection(
            self._query_vectors(
                q_nominal_px,
                keypoints_xyz_fk,
                robot_obs,
                image_size,
            )
        )
        local_tokens = self.visual_projection(local_visual)
        local_tokens = local_tokens + self.relative_position(relative_offsets).view(
            1, 1, -1, self.model_dim
        )
        batch, num_keypoints, num_local_tokens, _ = local_tokens.shape
        local_value = torch.cat([query.unsqueeze(2), local_tokens], dim=2)
        local_condition = query.unsqueeze(2).expand(
            -1, -1, num_local_tokens + 1, -1
        )
        local_value = local_value.reshape(
            batch * num_keypoints,
            num_local_tokens + 1,
            self.model_dim,
        )
        local_condition = local_condition.reshape_as(local_value)
        local_encoded = self.local_transformer(local_value, local_condition)[:, 0]
        local_encoded = local_encoded.reshape(batch, num_keypoints, self.model_dim)
        keypoint_latent = self.keypoint_transformer(local_encoded, query)

        q_absolute_norm = (torch.tanh(self.absolute_head(keypoint_latent)) + 1.0) * 0.5
        width, height = image_size
        pixel_scale = q_absolute_norm.new_tensor(
            (float(max(width - 1, 0)), float(max(height - 1, 0)))
        )
        q_absolute_px = q_absolute_norm * pixel_scale
        return AbsoluteKeypointOutput(
            q_absolute_norm=q_absolute_norm,
            q_absolute_px=q_absolute_px,
            keypoint_latent=keypoint_latent,
        )


class ParameterPadding(nn.Module):
    """Transparent unused capacity for exact parameter-budget matching."""

    def __init__(self, parameter_count: int) -> None:
        super().__init__()
        if parameter_count < 0:
            raise ValueError("parameter_count must be non-negative")
        self.parameter_count = int(parameter_count)
        if self.parameter_count:
            self.values = nn.Parameter(torch.zeros(self.parameter_count))
        else:
            self.register_parameter("values", None)


class AnchorConditionedAbsoluteKeypointViTAdapter(nn.Module):
    """Frozen ViT dense encoder plus the matched-input absolute head."""

    def __init__(
        self,
        obs_dim: int,
        num_keypoints: int,
        vision_dim: int = 256,
        model_dim: int = 256,
        model_name: str = DEFAULT_VIT_MODEL,
        vision_selected_layers: Sequence[int] = (11,),
        pretrained: bool = False,
        local_grid_size: int = 5,
        local_radius_px: float = 24.0,
        local_layers: int = 4,
        global_layers: int = 2,
        heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        target_trainable_parameters: int | None = None,
        allow_parameter_padding: bool = False,
    ) -> None:
        super().__init__()
        self.variant = BASELINE_VARIANT
        self.coordinate_parameterization = COORDINATE_PARAMETERIZATION
        self.obs_dim = int(obs_dim)
        self.num_keypoints = int(num_keypoints)
        self.vision_dim = int(vision_dim)
        self.model_dim = int(model_dim)
        self.vision_model_name = str(model_name)
        self.vision_selected_layers = tuple(int(v) for v in vision_selected_layers)
        self.vision = ViTDenseEncoder(
            model_name=self.vision_model_name,
            pretrained=pretrained,
            selected_layers=self.vision_selected_layers,
            output_dim=self.vision_dim,
            freeze_backbone=True,
        )
        self.grounding = AnchorConditionedAbsoluteKeypointHead(
            visual_dim=self.vision_dim,
            obs_dim=self.obs_dim,
            num_keypoints=self.num_keypoints,
            model_dim=self.model_dim,
            local_grid_size=local_grid_size,
            local_radius_px=local_radius_px,
            local_layers=local_layers,
            global_layers=global_layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        functional_trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        target = (
            functional_trainable
            if target_trainable_parameters is None
            else int(target_trainable_parameters)
        )
        difference = target - functional_trainable
        if difference < 0:
            raise ValueError(
                f"functional model has {functional_trainable} trainable parameters, "
                f"exceeding target {target}"
            )
        if difference and not allow_parameter_padding:
            raise ValueError(
                f"parameter target requires {difference} padding parameters; "
                "set allow_parameter_padding=true to report and permit them"
            )
        self.capacity_padding = ParameterPadding(difference)
        self.target_trainable_parameters = target

    def train(self, mode: bool = True) -> "AnchorConditionedAbsoluteKeypointViTAdapter":
        super().train(mode)
        self.vision.backbone.eval()
        return self

    def _ground_from_vision(
        self,
        vision: DenseVisionOutput,
        q_nominal_px: torch.Tensor,
        keypoints_xyz_fk: torch.Tensor,
        robot_obs: torch.Tensor,
        image_size: tuple[int, int],
    ) -> AbsoluteKeypointOutput:
        result = self.grounding(
            vision.feature_map,
            q_nominal_px,
            keypoints_xyz_fk,
            robot_obs,
            image_size=image_size,
        )
        return AbsoluteKeypointOutput(
            q_absolute_norm=result.q_absolute_norm,
            q_absolute_px=result.q_absolute_px,
            keypoint_latent=result.keypoint_latent,
            feature_map=vision.feature_map,
            global_feature=vision.global_feature,
        )

    def forward_from_backbone_features(
        self,
        feature_maps: torch.Tensor,
        prefix: torch.Tensor,
        q_nominal_px: torch.Tensor,
        keypoints_xyz_fk: torch.Tensor,
        robot_obs: torch.Tensor,
        *,
        image_size: tuple[int, int],
    ) -> AbsoluteKeypointOutput:
        vision = self.vision.project_backbone_features(feature_maps, prefix)
        return self._ground_from_vision(
            vision,
            q_nominal_px,
            keypoints_xyz_fk,
            robot_obs,
            image_size,
        )

    def forward(
        self,
        rgb: torch.Tensor,
        q_nominal_px: torch.Tensor,
        keypoints_xyz_fk: torch.Tensor,
        robot_obs: torch.Tensor,
    ) -> AbsoluteKeypointOutput:
        vision = self.vision(rgb)
        return self._ground_from_vision(
            vision,
            q_nominal_px,
            keypoints_xyz_fk,
            robot_obs,
            image_size=(int(rgb.shape[-1]), int(rgb.shape[-2])),
        )

    def parameter_report(self) -> dict[str, object]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        padding = self.capacity_padding.parameter_count
        return {
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "frozen_parameters": int(total - trainable),
            "functional_trainable_parameters": int(trainable - padding),
            "padding_trainable_parameters": int(padding),
            "target_trainable_parameters": int(self.target_trainable_parameters),
            "exact_target_match": trainable == self.target_trainable_parameters,
        }

    def sampling_report(self) -> dict[str, int]:
        per_keypoint = self.grounding.sampled_visual_tokens_per_keypoint
        return {
            "visual_feature_levels": len(self.vision_selected_layers),
            "sampled_visual_tokens_per_keypoint": per_keypoint,
            "sampled_visual_tokens_per_frame": per_keypoint * self.num_keypoints,
            "local_transformer_sequence_length": per_keypoint + 1,
            "keypoint_transformer_sequence_length": self.num_keypoints,
        }


def compute_absolute_keypoint_loss(
    output: AbsoluteKeypointOutput,
    q_true_px: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute direct absolute-q supervision without constructing a residual target."""

    if output.q_absolute_px.shape != q_true_px.shape:
        raise ValueError(
            f"prediction shape {tuple(output.q_absolute_px.shape)} does not match "
            f"q_true shape {tuple(q_true_px.shape)}"
        )
    if valid.shape != q_true_px.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError("valid must be a bool tensor matching q_true keypoints")
    selected = valid & torch.isfinite(q_true_px).all(dim=-1)
    if not bool(selected.any()):
        raise ValueError("absolute-q batch has no finite valid supervision")
    per_keypoint = F.smooth_l1_loss(
        output.q_absolute_px,
        q_true_px,
        reduction="none",
        beta=1.0,
    ).mean(dim=-1)
    weight = selected.to(dtype=per_keypoint.dtype)
    loss = (per_keypoint * weight).sum() / weight.sum()
    error = torch.linalg.vector_norm(output.q_absolute_px - q_true_px, dim=-1)
    mean_error = (error * weight).sum() / weight.sum()
    return loss, {"absolute_loss": loss, "absolute_error_px": mean_error}


_MODEL_CONFIG_KEYS = {
    "obs_dim",
    "num_keypoints",
    "vision_dim",
    "model_dim",
    "model_name",
    "vision_selected_layers",
    "local_grid_size",
    "local_radius_px",
    "local_layers",
    "global_layers",
    "heads",
    "mlp_ratio",
    "dropout",
    "target_trainable_parameters",
    "allow_parameter_padding",
}


def _require_exact_keys(value: Mapping[str, object], keys: set[str], context: str) -> None:
    actual = set(value)
    missing = sorted(keys - actual)
    unknown = sorted(actual - keys)
    if missing or unknown:
        raise ValueError(f"{context} schema mismatch: missing={missing} unknown={unknown}")


def _require_int(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}")
    return int(value)


def _require_number(value: object, context: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{context} must be finite and >= {minimum}")
    return result


def validate_model_config(value: Mapping[str, object]) -> dict[str, object]:
    _require_exact_keys(value, _MODEL_CONFIG_KEYS, "model")
    result = dict(value)
    for name in (
        "obs_dim",
        "num_keypoints",
        "vision_dim",
        "model_dim",
        "local_grid_size",
        "local_layers",
        "global_layers",
        "heads",
        "mlp_ratio",
        "target_trainable_parameters",
    ):
        result[name] = _require_int(value[name], f"model.{name}", minimum=1)
    result["local_radius_px"] = _require_number(
        value["local_radius_px"], "model.local_radius_px", minimum=1e-12
    )
    result["dropout"] = _require_number(value["dropout"], "model.dropout")
    if result["dropout"] >= 1.0:
        raise ValueError("model.dropout must be < 1")
    if result["obs_dim"] != 4:
        raise ValueError("same-input baseline requires 4-D deployment robot_obs")
    if result["local_grid_size"] % 2 == 0:
        raise ValueError("model.local_grid_size must be odd")
    if result["model_dim"] % result["heads"] != 0:
        raise ValueError("model.model_dim must be divisible by model.heads")
    if not isinstance(value["model_name"], str) or not value["model_name"]:
        raise ValueError("model.model_name must be a non-empty string")
    layers = value["vision_selected_layers"]
    if not isinstance(layers, list) or not layers:
        raise ValueError("model.vision_selected_layers must be a non-empty list")
    result["vision_selected_layers"] = [
        _require_int(layer, "model.vision_selected_layers[]") for layer in layers
    ]
    if not isinstance(value["allow_parameter_padding"], bool):
        raise ValueError("model.allow_parameter_padding must be bool")
    return result


_CONFIG_KEYS = {
    "schema_version",
    "experiment",
    "variant",
    "contract",
    "reference_fcpg",
    "data_sources",
    "split",
    "budget",
    "model",
    "optimization",
    "output_dir",
}


def validate_absolute_keypoint_config(value: Mapping[str, object]) -> dict[str, object]:
    """Validate the full experiment config and reject all implicit defaults."""

    _require_exact_keys(value, _CONFIG_KEYS, "config")
    if value["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported absolute-keypoint config schema_version")
    if value["variant"] != BASELINE_VARIANT:
        raise ValueError(f"config variant must be {BASELINE_VARIANT!r}")
    if value["contract"] != absolute_keypoint_contract():
        raise ValueError("config contract differs from the executable input contract")
    if not isinstance(value["experiment"], str) or not value["experiment"]:
        raise ValueError("config.experiment must be a non-empty string")
    if not isinstance(value["output_dir"], str) or not value["output_dir"]:
        raise ValueError("config.output_dir must be a non-empty string")

    reference = value["reference_fcpg"]
    if not isinstance(reference, Mapping):
        raise ValueError("config.reference_fcpg must be an object")
    _require_exact_keys(
        reference,
        {"checkpoint", "corrector_arch", "trainable_parameters"},
        "reference_fcpg",
    )
    if not isinstance(reference["checkpoint"], str) or not reference["checkpoint"]:
        raise ValueError("reference_fcpg.checkpoint must be a non-empty string")
    if reference["corrector_arch"] != REFERENCE_FCPG_ARCH:
        raise ValueError(f"reference corrector must be {REFERENCE_FCPG_ARCH!r}")
    reference_budget = _require_int(
        reference["trainable_parameters"],
        "reference_fcpg.trainable_parameters",
        minimum=1,
    )

    sources = value["data_sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("config.data_sources must be a non-empty list")
    names: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise ValueError(f"data_sources[{index}] must be an object")
        _require_exact_keys(source, {"name", "path", "vision_cache"}, f"data_sources[{index}]")
        if any(not isinstance(source[key], str) or not source[key] for key in source):
            raise ValueError(f"data_sources[{index}] fields must be non-empty strings")
        if source["name"] in names:
            raise ValueError(f"duplicate data source name {source['name']!r}")
        names.add(str(source["name"]))

    split = value["split"]
    if not isinstance(split, Mapping):
        raise ValueError("config.split must be an object")
    _require_exact_keys(split, {"unit", "val_fraction", "seed"}, "split")
    if split["unit"] != "task-episode":
        raise ValueError("same-input baseline requires task-episode split")
    fraction = _require_number(split["val_fraction"], "split.val_fraction")
    if not 0.0 < fraction < 1.0:
        raise ValueError("split.val_fraction must be in (0, 1)")
    _require_int(split["seed"], "split.seed")

    budget = value["budget"]
    if not isinstance(budget, Mapping):
        raise ValueError("config.budget must be an object")
    _require_exact_keys(
        budget,
        {
            "epochs",
            "batch_size",
            "num_workers",
            "balance_sources",
            "max_train_episodes_per_source",
            "episode_selection_seed",
        },
        "budget",
    )
    for name in ("epochs", "batch_size"):
        _require_int(budget[name], f"budget.{name}", minimum=1)
    for name in ("num_workers", "max_train_episodes_per_source", "episode_selection_seed"):
        _require_int(budget[name], f"budget.{name}")
    if not isinstance(budget["balance_sources"], bool):
        raise ValueError("budget.balance_sources must be bool")

    model = value["model"]
    if not isinstance(model, Mapping):
        raise ValueError("config.model must be an object")
    validated_model = validate_model_config(model)
    if validated_model["target_trainable_parameters"] != reference_budget:
        raise ValueError("model target parameter count must equal reference FCPG budget")

    optimization = value["optimization"]
    if not isinstance(optimization, Mapping):
        raise ValueError("config.optimization must be an object")
    _require_exact_keys(
        optimization,
        {
            "seed",
            "lr",
            "weight_decay",
            "warmup_epochs",
            "grad_clip",
            "loss",
        },
        "optimization",
    )
    _require_int(optimization["seed"], "optimization.seed")
    for name in ("lr", "weight_decay", "warmup_epochs", "grad_clip"):
        _require_number(optimization[name], f"optimization.{name}")
    if optimization["loss"] != "smooth_l1_beta_1_absolute_q_true_px":
        raise ValueError("optimization.loss must declare direct absolute q_true supervision")
    return dict(value)


def load_absolute_keypoint_config(path: str | Path) -> dict[str, object]:
    config_path = Path(path).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("absolute-keypoint config root must be an object")
    return validate_absolute_keypoint_config(raw)


def model_kwargs_from_config(model_config: Mapping[str, object]) -> dict[str, object]:
    config = validate_model_config(model_config)
    return {
        "obs_dim": config["obs_dim"],
        "num_keypoints": config["num_keypoints"],
        "vision_dim": config["vision_dim"],
        "model_dim": config["model_dim"],
        "model_name": config["model_name"],
        "vision_selected_layers": tuple(config["vision_selected_layers"]),
        "local_grid_size": config["local_grid_size"],
        "local_radius_px": config["local_radius_px"],
        "local_layers": config["local_layers"],
        "global_layers": config["global_layers"],
        "heads": config["heads"],
        "mlp_ratio": config["mlp_ratio"],
        "dropout": config["dropout"],
        "target_trainable_parameters": config["target_trainable_parameters"],
        "allow_parameter_padding": config["allow_parameter_padding"],
    }


def inspect_reference_fcpg_checkpoint(
    checkpoint_path: str | Path,
    model_config: Mapping[str, object],
) -> dict[str, object]:
    """Verify the matched FCPG architecture and derive its trainable budget."""

    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(path, map_location="meta", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("reference FCPG checkpoint must contain a mapping")
    if checkpoint.get("checkpoint_type") != "fcpg_corrector":
        raise ValueError("reference checkpoint is not an FCPG corrector")
    if checkpoint.get("corrector_arch") != REFERENCE_FCPG_ARCH:
        raise ValueError("reference FCPG corrector must use the LeWM architecture")
    if bool(checkpoint.get("has_reliability", False)):
        raise ValueError("reference FCPG checkpoint unexpectedly has reliability parameters")
    args = checkpoint.get("args")
    state = checkpoint.get("model_state")
    if not isinstance(args, Mapping) or not isinstance(state, Mapping):
        raise ValueError("reference FCPG checkpoint lacks args or model_state")
    config = validate_model_config(model_config)
    expected = {
        "model_name": config["model_name"],
        "vision_dim": config["vision_dim"],
        "model_dim": config["model_dim"],
        "local_grid_size": config["local_grid_size"],
        "local_radius_px": config["local_radius_px"],
        "local_layers": config["local_layers"],
        "global_layers": config["global_layers"],
        "heads": config["heads"],
        "dropout": config["dropout"],
    }
    mismatches = {
        name: (args.get(name), expected_value)
        for name, expected_value in expected.items()
        if args.get(name) != expected_value
    }
    reference_layers = tuple(int(v) for v in args.get("vision_selected_layers", ()))
    expected_layers = tuple(int(v) for v in config["vision_selected_layers"])
    if reference_layers != expected_layers:
        mismatches["vision_selected_layers"] = (reference_layers, expected_layers)
    if bool(args.get("unfreeze_backbone", False)):
        mismatches["unfreeze_backbone"] = (True, False)
    if int(checkpoint.get("obs_dim", -1)) != config["obs_dim"]:
        mismatches["obs_dim"] = (checkpoint.get("obs_dim"), config["obs_dim"])
    if int(checkpoint.get("num_keypoints", -1)) != config["num_keypoints"]:
        mismatches["num_keypoints"] = (
            checkpoint.get("num_keypoints"),
            config["num_keypoints"],
        )
    if mismatches:
        raise ValueError(f"reference FCPG architecture mismatch: {mismatches}")
    tensor_state = {
        str(name): tensor for name, tensor in state.items() if isinstance(tensor, torch.Tensor)
    }
    if len(tensor_state) != len(state):
        raise ValueError("reference FCPG model_state must contain only tensors")
    backbone_parameters = sum(
        tensor.numel()
        for name, tensor in tensor_state.items()
        if name.startswith(REFERENCE_FCPG_FROZEN_PREFIX)
    )
    if backbone_parameters <= 0:
        raise ValueError("reference FCPG checkpoint has no frozen ViT backbone state")
    trainable_parameters = sum(
        tensor.numel()
        for name, tensor in tensor_state.items()
        if not name.startswith(REFERENCE_FCPG_FROZEN_PREFIX)
    )
    if trainable_parameters != config["target_trainable_parameters"]:
        raise ValueError(
            f"reference FCPG trainable budget is {trainable_parameters}, expected "
            f"{config['target_trainable_parameters']}"
        )
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "trainable_parameters": int(trainable_parameters),
        "frozen_backbone_parameters": int(backbone_parameters),
        "model_name": str(args["model_name"]),
        "vision_selected_layers": list(reference_layers),
        "data_sources": [str(Path(value).resolve()) for value in checkpoint.get("data_sources", [])],
        "training_protocol": {
            "obs_mode": str(args.get("obs_mode", "")),
            "epochs": int(args.get("epochs", -1)),
            "batch_size": int(args.get("batch_size", -1)),
            "num_workers": int(args.get("num_workers", -1)),
            "balance_sources": bool(args.get("balance_sources", False)),
            "max_train_episodes_per_source": int(
                args.get("max_train_episodes_per_source", 0)
            ),
            "episode_selection_seed": int(args.get("episode_selection_seed", 0)),
            "val_fraction": float(args.get("val_fraction", float("nan"))),
            "split_unit": str(args.get("split_unit", "")),
            "split_seed": int(args.get("split_seed", -1)),
            "seed": int(args.get("seed", -1)),
            "lr": float(args.get("lr", float("nan"))),
            "weight_decay": float(args.get("weight_decay", float("nan"))),
            "warmup_epochs": float(args.get("warmup_epochs", float("nan"))),
            "grad_clip": float(args.get("grad_clip", float("nan"))),
            "reliability_weight": float(args.get("reliability_weight", float("nan"))),
            "reliability_corruption_prob": float(
                args.get("reliability_corruption_prob", float("nan"))
            ),
            "sigreg_weight": float(args.get("sigreg_weight", float("nan"))),
        },
    }


def validate_reference_training_protocol(
    reference_report: Mapping[str, object],
    experiment_config: Mapping[str, object],
    resolved_data_sources: Sequence[str | Path],
) -> None:
    """Require data, split, optimization, and frame budget parity with FCPG."""

    protocol = reference_report.get("training_protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("reference report lacks the FCPG training protocol")
    split = experiment_config.get("split")
    budget = experiment_config.get("budget")
    optimization = experiment_config.get("optimization")
    if not all(isinstance(value, Mapping) for value in (split, budget, optimization)):
        raise ValueError("experiment config lacks split/budget/optimization mappings")
    assert isinstance(split, Mapping)
    assert isinstance(budget, Mapping)
    assert isinstance(optimization, Mapping)
    expected_protocol = {
        "obs_mode": "robot",
        "epochs": int(budget["epochs"]),
        "batch_size": int(budget["batch_size"]),
        "num_workers": int(budget["num_workers"]),
        "balance_sources": bool(budget["balance_sources"]),
        "max_train_episodes_per_source": int(
            budget["max_train_episodes_per_source"]
        ),
        "episode_selection_seed": int(budget["episode_selection_seed"]),
        "val_fraction": float(split["val_fraction"]),
        "split_unit": str(split["unit"]),
        "split_seed": int(split["seed"]),
        "seed": int(optimization["seed"]),
        "lr": float(optimization["lr"]),
        "weight_decay": float(optimization["weight_decay"]),
        "warmup_epochs": float(optimization["warmup_epochs"]),
        "grad_clip": float(optimization["grad_clip"]),
        "reliability_weight": 0.0,
        "reliability_corruption_prob": 0.0,
        "sigreg_weight": 0.0,
    }
    mismatches = {
        name: (protocol.get(name), expected)
        for name, expected in expected_protocol.items()
        if protocol.get(name) != expected
    }
    reference_sources = [
        str(Path(value).expanduser().resolve())
        for value in reference_report.get("data_sources", [])
    ]
    expected_sources = [
        str(Path(value).expanduser().resolve()) for value in resolved_data_sources
    ]
    if reference_sources != expected_sources:
        mismatches["data_sources"] = (reference_sources, expected_sources)
    if mismatches:
        raise ValueError(f"reference FCPG training protocol mismatch: {mismatches}")


def load_frozen_backbone_from_reference(
    model: AnchorConditionedAbsoluteKeypointViTAdapter,
    checkpoint_path: str | Path,
) -> None:
    """Load only the exact frozen ViT backbone, excluding FCPG-trained heads."""

    checkpoint = torch.load(
        Path(checkpoint_path).expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("reference checkpoint lacks model_state")
    prefix = REFERENCE_FCPG_FROZEN_PREFIX
    backbone_state = {
        str(name)[len(prefix) :]: tensor
        for name, tensor in state.items()
        if isinstance(name, str) and name.startswith(prefix)
    }
    if not backbone_state:
        raise ValueError("reference checkpoint has no frozen backbone weights")
    model.vision.backbone.load_state_dict(backbone_state, strict=True)
    model.vision.backbone.requires_grad_(False)
    model.vision.backbone.eval()


_PARAMETER_REPORT_KEYS = {
    "total_parameters",
    "trainable_parameters",
    "frozen_parameters",
    "functional_trainable_parameters",
    "padding_trainable_parameters",
    "target_trainable_parameters",
    "exact_target_match",
}
_SAMPLING_REPORT_KEYS = {
    "visual_feature_levels",
    "sampled_visual_tokens_per_keypoint",
    "sampled_visual_tokens_per_frame",
    "local_transformer_sequence_length",
    "keypoint_transformer_sequence_length",
}
_CHECKPOINT_KEYS = {
    "schema_version",
    "checkpoint_type",
    "variant",
    "contract",
    "model_config",
    "model_state",
    "parameter_report",
    "sampling_report",
    "training_provenance",
    "epoch",
    "metrics",
}


def validate_absolute_keypoint_checkpoint_payload(
    payload: Mapping[str, object],
) -> dict[str, object]:
    _require_exact_keys(payload, _CHECKPOINT_KEYS, "checkpoint")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported absolute-keypoint checkpoint schema_version")
    if payload["checkpoint_type"] != CHECKPOINT_TYPE:
        raise ValueError("checkpoint_type is not an absolute-keypoint baseline")
    if payload["variant"] != BASELINE_VARIANT:
        raise ValueError("checkpoint variant is not the matched same-input baseline")
    if payload["contract"] != absolute_keypoint_contract():
        raise ValueError("checkpoint input contract is missing or mismatched")
    if not isinstance(payload["model_config"], Mapping):
        raise ValueError("checkpoint model_config must be an object")
    validate_model_config(payload["model_config"])
    state = payload["model_state"]
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint model_state must be a non-empty mapping")
    if any(not isinstance(name, str) or not isinstance(tensor, torch.Tensor) for name, tensor in state.items()):
        raise ValueError("checkpoint model_state must map string names to tensors")
    report = payload["parameter_report"]
    if not isinstance(report, Mapping):
        raise ValueError("checkpoint parameter_report must be an object")
    _require_exact_keys(report, _PARAMETER_REPORT_KEYS, "parameter_report")
    if report["exact_target_match"] is not True:
        raise ValueError("checkpoint does not match its declared parameter target")
    for name in _PARAMETER_REPORT_KEYS - {"exact_target_match"}:
        _require_int(report[name], f"parameter_report.{name}")
    if report["total_parameters"] != (
        report["trainable_parameters"] + report["frozen_parameters"]
    ):
        raise ValueError("checkpoint parameter totals are inconsistent")
    if report["trainable_parameters"] != (
        report["functional_trainable_parameters"]
        + report["padding_trainable_parameters"]
    ):
        raise ValueError("checkpoint functional/padding parameter totals are inconsistent")
    model_config = validate_model_config(payload["model_config"])
    if report["target_trainable_parameters"] != model_config["target_trainable_parameters"]:
        raise ValueError("checkpoint parameter target differs from model_config")
    if report["trainable_parameters"] != report["target_trainable_parameters"]:
        raise ValueError("checkpoint trainable parameters do not equal their target")
    sampling = payload["sampling_report"]
    if not isinstance(sampling, Mapping):
        raise ValueError("checkpoint sampling_report must be an object")
    _require_exact_keys(sampling, _SAMPLING_REPORT_KEYS, "sampling_report")
    for name in _SAMPLING_REPORT_KEYS:
        _require_int(sampling[name], f"sampling_report.{name}", minimum=1)
    expected_per_keypoint = int(model_config["local_grid_size"]) ** 2
    expected_sampling = {
        "visual_feature_levels": len(model_config["vision_selected_layers"]),
        "sampled_visual_tokens_per_keypoint": expected_per_keypoint,
        "sampled_visual_tokens_per_frame": (
            expected_per_keypoint * int(model_config["num_keypoints"])
        ),
        "local_transformer_sequence_length": expected_per_keypoint + 1,
        "keypoint_transformer_sequence_length": int(model_config["num_keypoints"]),
    }
    if dict(sampling) != expected_sampling:
        raise ValueError("checkpoint sampling report differs from model_config")
    provenance = payload["training_provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("checkpoint training_provenance must be an object")
    _require_exact_keys(
        provenance,
        {
            "config_path",
            "config_sha256",
            "reference_fcpg",
            "data_sources",
            "split",
            "budget",
            "q_true_usage",
        },
        "training_provenance",
    )
    if provenance["q_true_usage"] != "training_supervision_and_offline_metrics_only":
        raise ValueError("checkpoint provenance permits invalid q_true usage")
    _require_int(payload["epoch"], "checkpoint.epoch")
    if not isinstance(payload["metrics"], Mapping):
        raise ValueError("checkpoint metrics must be an object")
    return dict(payload)


def load_absolute_keypoint_baseline(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[AnchorConditionedAbsoluteKeypointViTAdapter, dict[str, object]]:
    """Fail-closed loader for absolute-keypoint checkpoints."""

    checkpoint = torch.load(
        Path(checkpoint_path).expanduser().resolve(),
        map_location=device,
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("absolute-keypoint checkpoint root must be a mapping")
    validated = validate_absolute_keypoint_checkpoint_payload(checkpoint)
    model_config = validated["model_config"]
    assert isinstance(model_config, Mapping)
    model = AnchorConditionedAbsoluteKeypointViTAdapter(
        **model_kwargs_from_config(model_config),
        pretrained=False,
    )
    state = validated["model_state"]
    assert isinstance(state, Mapping)
    expected_keys = set(model.state_dict())
    actual_keys = set(state)
    if expected_keys != actual_keys:
        raise ValueError(
            "checkpoint model_state schema mismatch: "
            f"missing={sorted(expected_keys - actual_keys)} "
            f"unknown={sorted(actual_keys - expected_keys)}"
        )
    model.load_state_dict(state, strict=True)
    actual_report = model.parameter_report()
    if actual_report != validated["parameter_report"]:
        raise ValueError("checkpoint parameter report does not match reconstructed model")
    if model.sampling_report() != validated["sampling_report"]:
        raise ValueError("checkpoint sampling report does not match reconstructed model")
    model.to(device).eval()
    return model, validated


def load_action_chunk_policy_with_absolute_keypoints(
    policy_checkpoint: str | Path,
    absolute_checkpoint: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict[str, object], dict[str, object]]:
    """Load an existing FCPG action policy with the absolute baseline override."""

    from fcpg.action_chunk_policy import load_action_chunk_policy

    baseline, baseline_metadata = load_absolute_keypoint_baseline(
        absolute_checkpoint, device=device
    )
    policy, policy_args = load_action_chunk_policy(
        policy_checkpoint,
        device=device,
        fcpg_adapter_override=baseline,
    )
    if int(getattr(policy, "obs_dim")) != baseline.obs_dim:
        raise ValueError("action policy obs_dim does not match absolute baseline")
    if int(getattr(policy, "num_keypoints")) != baseline.num_keypoints:
        raise ValueError("action policy keypoint count does not match absolute baseline")
    return policy, policy_args, baseline_metadata


_VISION_CACHE_KEYS = {
    "version",
    "source_data",
    "source_size_bytes",
    "source_data_sha256",
    "image_key",
    "selected_image_sha256",
    "view_role",
    "deployment_cache_eligible",
    "sample_count",
    "image_width",
    "image_height",
    "model_name",
    "weight_source",
    "selected_layers",
    "feature_shape",
    "prefix_dim",
    "dtype",
    "backbone_features_sha256",
    "prefix_sha256",
}


def validate_deployment_vision_cache(
    data_path: str | Path,
    cache_path: str | Path,
    model_config: Mapping[str, object],
    *,
    expected_weight_source: str | Path,
    verify_hashes: bool = True,
) -> dict[str, object]:
    """Reject target-only, legacy, mismatched, or corrupt dense-feature caches."""

    data = Path(data_path).expanduser().resolve()
    cache = Path(cache_path).expanduser().resolve()
    metadata_path = cache / "metadata.json"
    feature_path = cache / "backbone_features.npy"
    prefix_path = cache / "prefix.npy"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise ValueError("vision cache metadata root must be an object")
    _require_exact_keys(metadata, _VISION_CACHE_KEYS, "vision cache metadata")
    if metadata["version"] != 1:
        raise ValueError("unsupported vision cache metadata version")
    if Path(str(metadata["source_data"])).resolve() != data:
        raise ValueError("vision cache source_data does not match dataset")
    if int(metadata["source_size_bytes"]) != data.stat().st_size:
        raise ValueError("vision cache source size does not match dataset")
    if metadata["image_key"] != "rgb":
        raise ValueError("absolute baseline requires deployment rgb vision cache")
    if metadata["view_role"] != "deployment_observation":
        raise ValueError("vision cache is not marked as a deployment observation")
    if metadata["deployment_cache_eligible"] is not True:
        raise ValueError("vision cache is not deployment_cache_eligible")
    digest = metadata["selected_image_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest.lower())
    ):
        raise ValueError("vision cache lacks explicit selected-image lineage")
    config = validate_model_config(model_config)
    if metadata["model_name"] != config["model_name"]:
        raise ValueError("vision cache model_name does not match baseline")
    if metadata["selected_layers"] != config["vision_selected_layers"]:
        raise ValueError("vision cache selected_layers do not match baseline")
    if metadata["dtype"] != "float16":
        raise ValueError("vision cache dtype must be float16")
    expected_source = Path(expected_weight_source).expanduser().resolve()
    if Path(str(metadata["weight_source"])).resolve() != expected_source:
        raise ValueError("vision cache was not built from the reference FCPG weights")
    if verify_hashes:
        if metadata["source_data_sha256"] != sha256_file(data):
            raise ValueError("vision cache source_data_sha256 mismatch")
        if metadata["backbone_features_sha256"] != sha256_file(feature_path):
            raise ValueError("vision cache backbone feature hash mismatch")
        if metadata["prefix_sha256"] != sha256_file(prefix_path):
            raise ValueError("vision cache prefix hash mismatch")
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    prefix = np.load(prefix_path, mmap_mode="r", allow_pickle=False)
    sample_count = _require_int(metadata["sample_count"], "vision cache sample_count", minimum=1)
    width = _require_int(metadata["image_width"], "vision cache image_width", minimum=1)
    height = _require_int(metadata["image_height"], "vision cache image_height", minimum=1)
    feature_shape = metadata["feature_shape"]
    if not isinstance(feature_shape, list) or tuple(feature_shape) != tuple(features.shape[1:]):
        raise ValueError("vision cache feature_shape metadata mismatch")
    if features.dtype != np.float16 or features.ndim != 5:
        raise ValueError("vision cache features must have shape [N, L, C, H, W] float16")
    if features.shape[0] != sample_count or features.shape[1] != len(config["vision_selected_layers"]):
        raise ValueError("vision cache feature sample/level count mismatch")
    if prefix.dtype != np.float16 or prefix.ndim != 2 or prefix.shape[0] != sample_count:
        raise ValueError("vision cache prefix must have shape [N, C] float16")
    if prefix.shape[1] != int(metadata["prefix_dim"]):
        raise ValueError("vision cache prefix_dim metadata mismatch")
    with np.load(data, allow_pickle=False) as raw:
        if "obs" not in raw.files or raw["obs"].ndim < 2:
            raise ValueError("dataset must contain a non-scalar obs array")
        if int(raw["obs"].shape[0]) != sample_count:
            raise ValueError("vision cache sample count does not match dataset")
    return {
        **dict(metadata),
        "cache_path": str(cache),
        "metadata_path": str(metadata_path),
        "feature_path": str(feature_path),
        "prefix_path": str(prefix_path),
        "image_size": (width, height),
    }


def run_cpu_smoke(seed: int = 0) -> dict[str, object]:
    """Exercise direct absolute prediction and backward on a tiny CPU head."""

    torch.manual_seed(seed)
    head = AnchorConditionedAbsoluteKeypointHead(
        visual_dim=16,
        obs_dim=4,
        num_keypoints=3,
        model_dim=32,
        local_grid_size=3,
        local_radius_px=8.0,
        local_layers=1,
        global_layers=1,
        heads=4,
        dropout=0.0,
    ).cpu()
    feature_map = torch.randn(2, 16, 8, 8)
    q_nominal = torch.rand(2, 3, 2) * 31.0
    xyz_fk = torch.randn(2, 3, 3)
    robot_obs = torch.randn(2, 4)
    q_true = torch.rand(2, 3, 2) * 32.0
    valid = torch.ones(2, 3, dtype=torch.bool)
    start = time.perf_counter()
    output = head(
        feature_map,
        q_nominal,
        xyz_fk,
        robot_obs,
        image_size=(32, 32),
    )
    loss, metrics = compute_absolute_keypoint_loss(output, q_true, valid)
    loss.backward()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    gradient_parameters = sum(
        parameter.numel() for parameter in head.parameters() if parameter.grad is not None
    )
    return {
        "variant": BASELINE_VARIANT,
        "device": "cpu",
        "output_shape": list(output.q_absolute_px.shape),
        "finite": bool(torch.isfinite(output.q_absolute_px).all()),
        "loss": float(loss.detach()),
        "mean_error_px": float(metrics["absolute_error_px"].detach()),
        "gradient_parameters": int(gradient_parameters),
        "sampled_visual_tokens_per_frame": 3 * 3 * 3,
        "elapsed_ms": elapsed_ms,
        "q_true_is_model_input": False,
        "coordinate_parameterization": COORDINATE_PARAMETERIZATION,
    }
