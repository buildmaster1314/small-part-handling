"""Matched policy-only grounding baselines for FK-anchored visual readout.

Both variants use the same frozen dense ViT and action policy.  They differ
only in how the K policy readout locations move around ``q_nominal``:

``policy_only_residual``
    Predict one shared residual per anchor and translate a fixed K-point
    stencil.  Attention weights remain input-conditioned.

``fk_deform_attn``
    Predict K independent offsets and K attention weights per anchor.

No projection target is accepted by either forward path.  The only training
objective implemented here is action behavior cloning.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from fcpg.action_chunk_policy import ActionChunkOutput, GroundedActionChunkTransformer
from fcpg.lewm_core import LEWM_UPSTREAM_COMMIT, LeWMTransformer
from fcpg.modules import sample_feature_map_at_pixels
from fcpg.transformer_grounding import (
    DEFAULT_VIT_MODEL,
    load_fcpg_vit_adapter,
    sample_local_feature_tokens,
)


CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINT_TYPE = "fk_deform_attn_policy_pair_member"
VARIANTS = ("policy_only_residual", "fk_deform_attn")
ACTIVE_PARAMETER_TOLERANCE = 0.01
AUDITED_VISION_MODEL = DEFAULT_VIT_MODEL
AUDITED_VISION_SELECTED_LAYERS = (11,)
SELECTION_POLICY = "independent-best-at-matched-budget"
DEVELOPMENT_CLAIM_SCOPE = "seed0-development-controlled-comparison"

DEPLOYMENT_INPUT_KEYS = (
    "obs",
    "keypoints_xyz",
    "q_nominal",
    "rgb",
    "vision_backbone_features",
    "vision_prefix",
    "image_size",
)

LOSS_CONTRACT: dict[str, object] = {
    "objective": "behavior_cloning_only",
    "allowed_terms": (
        "smooth_l1_motion_beta_0.05",
        "binary_cross_entropy_gripper",
    ),
    "offset_weight_supervision": "policy_action_loss_only",
    "forbidden_terms": (
        "q_true",
        "delta_q",
        "coordinate_residual",
        "reprojection",
        "anchor_epe",
        "latent_prediction",
        "sigreg",
    ),
    "test_time_projection_target_access": False,
}

STRUCTURAL_ATTRIBUTION_CONTRACT: dict[str, object] = {
    "allowed_structural_pair": (
        "policy_only_residual",
        "fk_deform_attn",
    ),
    "required_supervision_for_both": "behavior_cloning_only",
    "supervised_fcpg_vs_policy_only_baseline": "system_comparison_only",
    "supervised_fcpg_vs_fk_deform_attn": "system_comparison_only",
    "forbidden_claim": (
        "supervised FCPG versus FK-DeformAttn alone establishes a structural "
        "advantage of single-residual grounding"
    ),
}


def _count_parameters(module: nn.Module, *, trainable_only: bool) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in module.parameters()
            if not trainable_only or parameter.requires_grad
        )
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_state_sha256(module: nn.Module) -> str:
    """Hash a module state deterministically without serialization metadata."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def make_fixed_stencil(
    num_samples: int,
    lattice_step_px: float = 1.0,
) -> torch.Tensor:
    """Create the fixed 1-pixel eight-neighbor initialization lattice."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if lattice_step_px <= 0:
        raise ValueError("lattice_step_px must be positive")
    if num_samples == 8:
        return float(lattice_step_px) * torch.tensor(
            (
                (-1.0, -1.0),
                (0.0, -1.0),
                (1.0, -1.0),
                (-1.0, 0.0),
                (1.0, 0.0),
                (-1.0, 1.0),
                (0.0, 1.0),
                (1.0, 1.0),
            ),
            dtype=torch.float32,
        )
    angles = torch.arange(num_samples, dtype=torch.float32)
    angles = angles * (2.0 * math.pi / float(num_samples))
    return float(lattice_step_px) * torch.stack(
        (torch.cos(angles), torch.sin(angles)),
        dim=-1,
    )


@dataclass(frozen=True)
class PolicyReadoutOutput:
    weighted_visual: torch.Tensor
    offsets_px: torch.Tensor
    sample_points_px: torch.Tensor
    attention_weights: torch.Tensor
    keypoint_latent: torch.Tensor
    shared_residual_px: torch.Tensor | None

    @property
    def attention_barycenter_px(self) -> torch.Tensor:
        return (
            self.sample_points_px * self.attention_weights.unsqueeze(-1)
        ).sum(dim=-2)


def attention_barycenter_epe(
    readout: PolicyReadoutOutput,
    reference_points_px: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the only coordinate diagnostic defined for weighted readouts."""

    barycenter = readout.attention_barycenter_px
    if reference_points_px.shape != barycenter.shape:
        raise ValueError("reference_points_px must match attention barycenter shape")
    error = torch.linalg.vector_norm(barycenter - reference_points_px, dim=-1)
    if valid is None:
        return error.mean()
    if valid.shape != error.shape:
        raise ValueError("valid must match the attention barycenter error shape")
    weights = valid.to(dtype=error.dtype)
    return (error * weights).sum() / weights.sum().clamp_min(1.0)


class FKPolicyReadout(nn.Module):
    """Predict matched residual or deformable K-point policy readouts."""

    def __init__(
        self,
        *,
        variant: str,
        visual_dim: int,
        obs_dim: int,
        num_keypoints: int,
        model_dim: int = 256,
        num_samples: int = 8,
        context_grid_size: int = 5,
        context_radius_px: float = 24.0,
        offset_scale_px: float = 20.0,
        local_layers: int = 4,
        global_layers: int = 2,
        heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        lattice_step_px: float = 1.0,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown policy-only grounding variant {variant!r}")
        if model_dim <= 0 or model_dim % heads != 0:
            raise ValueError("model_dim must be positive and divisible by heads")
        if context_grid_size <= 0 or context_grid_size % 2 == 0:
            raise ValueError("context_grid_size must be a positive odd integer")
        self.variant = str(variant)
        self.visual_dim = int(visual_dim)
        self.obs_dim = int(obs_dim)
        self.num_keypoints = int(num_keypoints)
        self.model_dim = int(model_dim)
        self.num_samples = int(num_samples)
        self.context_grid_size = int(context_grid_size)
        self.context_radius_px = float(context_radius_px)
        self.offset_scale_px = float(offset_scale_px)
        self.local_layers = int(local_layers)
        self.global_layers = int(global_layers)
        self.heads = int(heads)
        self.mlp_ratio = int(mlp_ratio)
        self.lattice_step_px = float(lattice_step_px)
        if self.num_keypoints <= 0 or self.num_samples <= 0:
            raise ValueError("num_keypoints and num_samples must be positive")
        if self.num_samples != 8:
            raise ValueError("the audited matched pair requires exactly 8 policy samples")
        if self.context_radius_px <= 0 or self.offset_scale_px <= 0:
            raise ValueError("context_radius_px and offset_scale_px must be positive")

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
        mlp_dim = self.model_dim * self.mlp_ratio
        self.local_transformer = LeWMTransformer(
            input_dim=self.model_dim,
            hidden_dim=self.model_dim,
            output_dim=self.model_dim,
            depth=self.local_layers,
            heads=self.heads,
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
            depth=self.global_layers,
            heads=self.heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            causal=False,
            conditional=True,
            condition_dim=self.model_dim,
        )
        prediction_dim = (
            3 * self.num_samples
            if self.variant == "fk_deform_attn"
            else 2 + self.num_samples
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, prediction_dim),
        )
        stencil = make_fixed_stencil(self.num_samples, self.lattice_step_px)
        self.register_buffer("fixed_stencil_px", stencil)
        self._initialize_output_head()

    @property
    def context_samples_per_anchor(self) -> int:
        return self.context_grid_size**2

    def _initialize_output_head(self) -> None:
        final = self.output_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        if self.variant != "fk_deform_attn":
            return
        normalized = (self.fixed_stencil_px / self.offset_scale_px).clamp(
            -0.999,
            0.999,
        )
        raw_offsets = torch.atanh(normalized).reshape(-1)
        with torch.no_grad():
            final.bias[: 2 * self.num_samples].copy_(raw_offsets)

    def _query_vectors(
        self,
        q_nominal_px: torch.Tensor,
        keypoints_xyz: torch.Tensor,
        robot_obs: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        batch, num_keypoints, coordinate_dim = q_nominal_px.shape
        if num_keypoints != self.num_keypoints or coordinate_dim != 2:
            raise ValueError(
                f"q_nominal must have shape [B,{self.num_keypoints},2], got "
                f"{tuple(q_nominal_px.shape)}"
            )
        if keypoints_xyz.shape != (batch, self.num_keypoints, 3):
            raise ValueError("keypoints_xyz must align with q_nominal and end in 3")
        if robot_obs.shape != (batch, self.obs_dim):
            raise ValueError(
                f"robot_obs must have shape {(batch, self.obs_dim)}, got "
                f"{tuple(robot_obs.shape)}"
            )
        width, height = image_size
        q_norm = torch.stack(
            (
                q_nominal_px[..., 0] / float(width),
                q_nominal_px[..., 1] / float(height),
            ),
            dim=-1,
        )
        identity = torch.eye(
            self.num_keypoints,
            device=q_nominal_px.device,
            dtype=q_nominal_px.dtype,
        ).unsqueeze(0).expand(batch, -1, -1)
        expanded_obs = robot_obs[:, None, :].expand(-1, self.num_keypoints, -1)
        return torch.cat((q_norm, keypoints_xyz, identity, expanded_obs), dim=-1)

    def _predict(
        self,
        feature_map: torch.Tensor,
        q_nominal_px: torch.Tensor,
        keypoints_xyz: torch.Tensor,
        robot_obs: torch.Tensor,
        image_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_visual, relative_offsets = sample_local_feature_tokens(
            feature_map,
            q_nominal_px,
            image_size=image_size,
            radius_px=self.context_radius_px,
            grid_size=self.context_grid_size,
        )
        query = self.query_projection(
            self._query_vectors(
                q_nominal_px,
                keypoints_xyz,
                robot_obs,
                image_size,
            )
        )
        local_tokens = self.visual_projection(local_visual)
        local_tokens = local_tokens + self.relative_position(relative_offsets).view(
            1,
            1,
            -1,
            self.model_dim,
        )
        batch, num_keypoints, local_count, _ = local_tokens.shape
        local_value = torch.cat((query.unsqueeze(2), local_tokens), dim=2)
        local_condition = query.unsqueeze(2).expand(
            -1,
            -1,
            local_count + 1,
            -1,
        )
        local_value = local_value.reshape(
            batch * num_keypoints,
            local_count + 1,
            self.model_dim,
        )
        local_condition = local_condition.reshape_as(local_value)
        local_encoded = self.local_transformer(local_value, local_condition)[:, 0]
        local_encoded = local_encoded.reshape(
            batch,
            num_keypoints,
            self.model_dim,
        )
        keypoint_latent = self.keypoint_transformer(local_encoded, query)
        return self.output_head(keypoint_latent), keypoint_latent

    def forward(
        self,
        feature_map: torch.Tensor,
        q_nominal_px: torch.Tensor,
        keypoints_xyz: torch.Tensor,
        robot_obs: torch.Tensor,
        image_size: tuple[int, int],
    ) -> PolicyReadoutOutput:
        prediction, latent = self._predict(
            feature_map,
            q_nominal_px,
            keypoints_xyz,
            robot_obs,
            image_size,
        )
        batch = q_nominal_px.shape[0]
        if self.variant == "fk_deform_attn":
            raw_offsets = prediction[..., : 2 * self.num_samples].reshape(
                batch,
                self.num_keypoints,
                self.num_samples,
                2,
            )
            attention_logits = prediction[..., 2 * self.num_samples :]
            offsets_px = torch.tanh(raw_offsets) * self.offset_scale_px
            shared_residual = None
        else:
            stencil_extent = float(self.fixed_stencil_px.abs().max())
            shared_radius = self.offset_scale_px - stencil_extent
            if shared_radius <= 0:
                raise RuntimeError("fixed stencil leaves no radius for shared residual")
            shared_residual = torch.tanh(prediction[..., :2]) * shared_radius
            attention_logits = prediction[..., 2:]
            offsets_px = (
                shared_residual.unsqueeze(2)
                + self.fixed_stencil_px.to(dtype=shared_residual.dtype).view(
                    1,
                    1,
                    self.num_samples,
                    2,
                )
            )
        attention = torch.softmax(attention_logits, dim=-1)
        sample_points = q_nominal_px.unsqueeze(2) + offsets_px
        sampled = sample_feature_map_at_pixels(
            feature_map,
            sample_points.reshape(batch, self.num_keypoints * self.num_samples, 2),
            image_size=image_size,
            padding_mode="border",
        ).reshape(
            batch,
            self.num_keypoints,
            self.num_samples,
            self.visual_dim,
        )
        weighted = (sampled * attention.unsqueeze(-1)).sum(dim=2)
        return PolicyReadoutOutput(
            weighted_visual=weighted,
            offsets_px=offsets_px,
            sample_points_px=sample_points,
            attention_weights=attention,
            keypoint_latent=latent,
            shared_residual_px=shared_residual,
        )


class MatchedPolicyOnlyGrounding(nn.Module):
    """Compose a frozen dense ViT, policy-only readout, and action policy."""

    def __init__(
        self,
        policy: GroundedActionChunkTransformer,
        readout: FKPolicyReadout,
    ) -> None:
        super().__init__()
        if policy.num_keypoints != readout.num_keypoints:
            raise ValueError("policy and readout keypoint counts differ")
        if policy.obs_dim != readout.obs_dim:
            raise ValueError("policy and readout observation dimensions differ")
        if policy.vision_dim != readout.visual_dim:
            raise ValueError("policy and readout visual dimensions differ")
        self.policy = policy
        self.readout = readout
        self.variant = readout.variant
        self.mode = readout.variant
        self.policy.mode = "geoprop"
        self.policy.fcpg_adapter = None
        self.policy.vision_encoder.backbone.requires_grad_(False)
        self.policy.vision_encoder.backbone.eval()
        if self.policy.next_latent_predictor is not None:
            self.policy.next_latent_predictor.requires_grad_(False)

        self.policy_arch = policy.policy_arch
        self.obs_dim = policy.obs_dim
        self.action_dim = policy.action_dim
        self.num_keypoints = policy.num_keypoints
        self.obs_horizon = policy.obs_horizon
        self.action_horizon = policy.action_horizon
        self.model_dim = policy.model_dim
        self.vision_dim = policy.vision_dim
        self.vision_model_name = policy.vision_model_name
        self.vision_selected_layers = policy.vision_selected_layers

    @property
    def vision_encoder(self) -> nn.Module:
        return self.policy.vision_encoder

    def train(self, mode: bool = True) -> "MatchedPolicyOnlyGrounding":
        super().train(mode)
        self.policy.vision_encoder.backbone.eval()
        return self

    def set_normalization(self, values: Mapping[str, torch.Tensor]) -> None:
        self.policy.set_normalization(dict(values))

    def parameter_counts(self) -> dict[str, int]:
        backbone_total = _count_parameters(
            self.policy.vision_encoder.backbone,
            trainable_only=False,
        )
        backbone_active = _count_parameters(
            self.policy.vision_encoder.backbone,
            trainable_only=True,
        )
        vision_total_active = _count_parameters(
            self.policy.vision_encoder,
            trainable_only=True,
        )
        policy_total_active = _count_parameters(self.policy, trainable_only=True)
        common_policy_active = policy_total_active - vision_total_active
        readout_active = _count_parameters(self.readout, trainable_only=True)
        adapter_active = vision_total_active + readout_active
        return {
            "frozen_backbone_total": backbone_total,
            "frozen_backbone_active": backbone_active,
            "vision_neck_active": vision_total_active,
            "readout_active": readout_active,
            "common_policy_active": common_policy_active,
            "adapter_active": adapter_active,
            "total_active": common_policy_active + adapter_active,
        }

    @staticmethod
    def _deployment_view(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        selected = {
            key: batch[key]
            for key in DEPLOYMENT_INPUT_KEYS
            if key in batch
        }
        for required in ("obs", "keypoints_xyz", "q_nominal"):
            if required not in selected:
                raise KeyError(f"deployment batch is missing {required!r}")
        has_rgb = "rgb" in selected
        has_cache = (
            "vision_backbone_features" in selected or "vision_prefix" in selected
        )
        if has_rgb == has_cache:
            raise ValueError(
                "deployment batch requires exactly one of rgb or cached backbone features"
            )
        if has_cache and not {
            "vision_backbone_features",
            "vision_prefix",
            "image_size",
        }.issubset(selected):
            raise KeyError(
                "cached deployment input requires vision_backbone_features, "
                "vision_prefix, and image_size"
            )
        return selected

    def _dense_vision(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        obs = batch["obs"]
        if obs.ndim != 3:
            raise ValueError("obs must have shape [B,H,D]")
        batch_size, horizon, _ = obs.shape
        if horizon != self.obs_horizon:
            raise ValueError(
                f"expected obs_horizon={self.obs_horizon}, got {horizon}"
            )
        if "rgb" in batch:
            rgb = batch["rgb"]
            if rgb.ndim != 5 or rgb.shape[:2] != (batch_size, horizon):
                raise ValueError("rgb must have shape [B,H,3,Height,Width]")
            height, width = int(rgb.shape[-2]), int(rgb.shape[-1])
            vision = self.policy.vision_encoder(
                rgb.reshape(batch_size * horizon, *rgb.shape[2:])
            )
        else:
            image_sizes = batch["image_size"]
            if image_sizes.shape != (batch_size, 2):
                raise ValueError("image_size must have shape [B,2]")
            if not bool(torch.all(image_sizes == image_sizes[0])):
                raise ValueError("all samples in a batch must share one image size")
            width, height = (int(value) for value in image_sizes[0].tolist())
            vision_encoder = self.policy.vision_encoder
            feature_parameter = next(
                vision_encoder.feature_projections.parameters(),
                None,
            )
            prefix_parameter = next(
                vision_encoder.global_projection.parameters(),
                None,
            )
            if feature_parameter is None or prefix_parameter is None:
                raise RuntimeError("dense vision projections must have parameters")
            features = batch["vision_backbone_features"].to(
                device=feature_parameter.device,
                dtype=feature_parameter.dtype,
            )
            prefix = batch["vision_prefix"].to(
                device=prefix_parameter.device,
                dtype=prefix_parameter.dtype,
            )
            vision = vision_encoder.project_backbone_features(
                features.reshape(batch_size * horizon, *features.shape[2:]),
                prefix.reshape(batch_size * horizon, prefix.shape[-1]),
            )
        return vision.feature_map, vision.global_feature, (width, height)

    def forward_with_readout(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[ActionChunkOutput, PolicyReadoutOutput]:
        deployment = self._deployment_view(batch)
        obs = deployment["obs"]
        q_nominal = deployment["q_nominal"]
        keypoints_xyz = deployment["keypoints_xyz"]
        batch_size, horizon, _ = obs.shape
        expected_q = (batch_size, horizon, self.num_keypoints, 2)
        expected_xyz = (batch_size, horizon, self.num_keypoints, 3)
        if q_nominal.shape != expected_q:
            raise ValueError(
                f"q_nominal must have shape {expected_q}, got {tuple(q_nominal.shape)}"
            )
        if keypoints_xyz.shape != expected_xyz:
            raise ValueError(
                "keypoints_xyz must have shape "
                f"{expected_xyz}, got {tuple(keypoints_xyz.shape)}"
            )
        feature_map, global_feature, image_size = self._dense_vision(deployment)
        flat_count = batch_size * horizon
        readout = self.readout(
            feature_map,
            q_nominal.reshape(flat_count, self.num_keypoints, 2),
            keypoints_xyz.reshape(flat_count, self.num_keypoints, 3),
            obs.reshape(flat_count, self.obs_dim),
            image_size,
        )
        policy_batch = {
            "obs": obs,
            "q_nominal": q_nominal,
            "keypoints_xyz": keypoints_xyz,
            "sampled_visual": readout.weighted_visual.reshape(
                batch_size,
                horizon,
                self.num_keypoints,
                self.vision_dim,
            ),
            "global_visual": global_feature.reshape(
                batch_size,
                horizon,
                self.vision_dim,
            ),
            "image_size": torch.tensor(
                image_size,
                device=obs.device,
                dtype=torch.long,
            ).view(1, 2).expand(batch_size, -1),
        }
        return self.policy(policy_batch), readout

    def forward(self, batch: Mapping[str, torch.Tensor]) -> ActionChunkOutput:
        output, _ = self.forward_with_readout(batch)
        return output

    def compute_bc_loss(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        gripper_weight: float = 0.5,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], ActionChunkOutput]:
        if gripper_weight < 0:
            raise ValueError("gripper_weight must be non-negative")
        output = self(batch)
        target = batch["action"]
        mask = batch["action_mask"].to(dtype=target.dtype)
        motion_per_step = F.smooth_l1_loss(
            output.motion,
            target[..., :3],
            reduction="none",
            beta=0.05,
        ).mean(dim=-1)
        gripper_per_step = F.binary_cross_entropy_with_logits(
            output.gripper_logit,
            target[..., 3],
            reduction="none",
        )
        denominator = mask.sum().clamp_min(1.0)
        motion_loss = (motion_per_step * mask).sum() / denominator
        gripper_loss = (gripper_per_step * mask).sum() / denominator
        loss = motion_loss + float(gripper_weight) * gripper_loss
        return (
            loss,
            {
                "motion_loss": motion_loss,
                "gripper_loss": gripper_loss,
            },
            output,
        )

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> ActionChunkOutput:
        self.eval()
        return self(batch)


def _batch_sha256(batch: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(batch.items()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"latency input {name!r} must be a tensor")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def benchmark_interleaved_latency(
    models: Mapping[str, MatchedPolicyOnlyGrounding],
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
    *,
    warmup: int,
    iterations: int,
    amp: bool,
    comparison_id: str,
) -> dict[str, dict[str, object]]:
    """Benchmark both pair members on one input with alternating call order."""

    if set(models) != set(VARIANTS):
        raise ValueError("latency benchmark requires exactly one model per variant")
    if warmup <= 0 or iterations <= 0:
        raise ValueError("latency warmup and iterations must be positive")
    if not comparison_id:
        raise ValueError("latency comparison_id must be non-empty")
    if not batch:
        raise ValueError("latency batch must not be empty")
    first = next(iter(batch.values()))
    if not isinstance(first, torch.Tensor) or first.ndim == 0:
        raise ValueError("latency batch tensors must have a batch dimension")
    batch_size = int(first.shape[0])
    if batch_size != 1:
        raise ValueError("structural latency benchmark requires batch size 1")
    for name, tensor in batch.items():
        if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0:
            raise ValueError(f"latency input {name!r} has no batch dimension")
        if int(tensor.shape[0]) != batch_size:
            raise ValueError("latency inputs must share one fixed batch dimension")
        if tensor.device.type != device.type or (
            device.index is not None and tensor.device.index != device.index
        ):
            raise ValueError(
                f"latency input {name!r} is on {tensor.device}, expected {device}"
            )
    for model in models.values():
        model.eval()

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def ordered_variants(index: int) -> tuple[str, ...]:
        return VARIANTS if index % 2 == 0 else tuple(reversed(VARIANTS))

    def invoke(variant: str) -> None:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp and device.type == "cuda",
        ):
            models[variant].predict_action_chunk(batch)

    for index in range(warmup):
        for variant in ordered_variants(index):
            invoke(variant)
    synchronize()

    values: dict[str, list[float]] = {variant: [] for variant in VARIANTS}
    for index in range(iterations):
        for variant in ordered_variants(index):
            synchronize()
            started = time.perf_counter()
            invoke(variant)
            synchronize()
            values[variant].append((time.perf_counter() - started) * 1000.0)

    comparison_contract: dict[str, object] = {
        "protocol": "fixed-input-equal-count-warmup-interleaved-v1",
        "comparison_id": comparison_id,
        "fixed_input_sha256": _batch_sha256(batch),
        "fixed_input_fields": tuple(sorted(batch)),
        "input_mode": (
            "rgb" if "rgb" in batch else "cached_frozen_backbone_features"
        ),
        "batch_size": batch_size,
        "device": str(device),
        "amp": bool(amp and device.type == "cuda"),
        "warmup_iterations_per_variant": int(warmup),
        "timed_iterations_per_variant": int(iterations),
        "interleaved_order": "alternating-forward-reverse-by-iteration",
        "equal_call_count_per_variant": True,
        "rollout_call_counts_used": False,
        "includes_frozen_backbone_compute": "rgb" in batch,
        "includes_dense_neck_readout_and_action_policy": True,
    }
    result: dict[str, dict[str, object]] = {}
    for variant in VARIANTS:
        timings = torch.tensor(values[variant], dtype=torch.float64)
        result[variant] = {
            "comparison_contract": dict(comparison_contract),
            "measurement": {
                "variant": variant,
                "timed_iterations": int(timings.numel()),
                "mean_ms": float(timings.mean()),
                "p50_ms": float(torch.quantile(timings, 0.50)),
                "p95_ms": float(torch.quantile(timings, 0.95)),
                "min_ms": float(timings.min()),
                "max_ms": float(timings.max()),
            },
        }
    return result


def _policy_config(
    checkpoint: Mapping[str, Any],
) -> dict[str, object]:
    if checkpoint.get("checkpoint_type") != "fcpg_action_policy":
        raise ValueError("policy contract checkpoint is not an FCPG action policy")
    args = dict(checkpoint["args"])
    if str(args.get("mode")) != "fcpg":
        raise ValueError("policy contract checkpoint must use fcpg mode")
    if not bool(args.get("reuse_fcpg_vision", False)):
        raise ValueError("matched baseline requires a policy with reused frozen FCPG vision")
    if int(args.get("trainable_backbone_blocks", 0)) != 0:
        raise ValueError("matched baseline requires a fully frozen ViT backbone")
    return {
        "obs_dim": int(checkpoint["obs_dim"]),
        "action_dim": int(checkpoint["action_dim"]),
        "num_keypoints": int(checkpoint["num_keypoints"]),
        "obs_mode": str(args.get("obs_mode", "robot")),
        "obs_horizon": int(args.get("obs_horizon", 2)),
        "action_horizon": int(args.get("action_horizon", 32)),
        "model_dim": int(args.get("model_dim", 256)),
        "vision_dim": int(args.get("vision_dim", 256)),
        "encoder_layers": int(args.get("encoder_layers", 2)),
        "decoder_layers": int(args.get("decoder_layers", 6)),
        "predictor_layers": int(args.get("predictor_layers", 2)),
        "heads": int(args.get("heads", 8)),
        "mlp_ratio": int(args.get("mlp_ratio", 4)),
        "dropout": float(args.get("dropout", 0.1)),
        "vision_model_name": str(args.get("vision_model_name", DEFAULT_VIT_MODEL)),
        "vision_selected_layers": tuple(
            int(layer) for layer in args.get("vision_selected_layers", (2, 5, 8, 11))
        ),
        "policy_arch": str(args.get("policy_arch", "legacy")),
        "gripper_action_mode": str(
            args.get("gripper_action_mode", "legacy_unsigned")
        ),
    }


def build_matched_policy_only_grounding(
    *,
    variant: str,
    policy_contract_checkpoint: str | Path,
    fcpg_corrector_checkpoint: str | Path,
    policy_seed: int,
    num_samples: int = 8,
    lattice_step_px: float = 1.0,
    offset_scale_px: float = 20.0,
    active_parameter_tolerance: float = ACTIVE_PARAMETER_TOLERANCE,
    device: torch.device | str = "cpu",
) -> tuple[MatchedPolicyOnlyGrounding, dict[str, object]]:
    """Build one pair member without loading trained policy weights."""

    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    if num_samples != 8:
        raise ValueError("the audited FK-DeformAttn main variant requires S=8")
    if not math.isclose(lattice_step_px, 1.0, abs_tol=1e-12):
        raise ValueError("the audited initialization requires a 1px lattice")
    if not math.isclose(offset_scale_px, 20.0, abs_tol=1e-12):
        raise ValueError("the audited FK offsets require offset=20*tanh(raw_offset)")
    if not 0.0 <= active_parameter_tolerance <= ACTIVE_PARAMETER_TOLERANCE:
        raise ValueError("active parameter tolerance must be in [0, 0.01]")
    policy_path = Path(policy_contract_checkpoint).expanduser().resolve()
    corrector_path = Path(fcpg_corrector_checkpoint).expanduser().resolve()
    policy_checkpoint = torch.load(policy_path, map_location="cpu", weights_only=False)
    config = _policy_config(policy_checkpoint)
    corrector, corrector_args = load_fcpg_vit_adapter(corrector_path, device="cpu")
    if int(corrector.obs_dim if hasattr(corrector, "obs_dim") else config["obs_dim"]) != int(
        config["obs_dim"]
    ):
        raise ValueError("corrector and policy observation dimensions differ")
    if int(corrector.grounding.num_keypoints) != int(config["num_keypoints"]):
        raise ValueError("corrector and policy keypoint counts differ")
    if int(corrector.vision.output_dim) != int(config["vision_dim"]):
        raise ValueError("corrector and policy visual dimensions differ")
    if corrector.vision.model_name != config["vision_model_name"]:
        raise ValueError("corrector and policy ViT model names differ")
    if tuple(corrector.vision.selected_layers) != tuple(config["vision_selected_layers"]):
        raise ValueError("corrector and policy selected ViT layers differ")

    full_corrector_active = _count_parameters(corrector, trainable_only=True)
    grounding_active = _count_parameters(corrector.grounding, trainable_only=True)

    torch.manual_seed(int(policy_seed))
    policy = GroundedActionChunkTransformer(
        mode="geoprop",
        obs_dim=int(config["obs_dim"]),
        action_dim=int(config["action_dim"]),
        num_keypoints=int(config["num_keypoints"]),
        obs_horizon=int(config["obs_horizon"]),
        action_horizon=int(config["action_horizon"]),
        model_dim=int(config["model_dim"]),
        vision_dim=int(config["vision_dim"]),
        encoder_layers=int(config["encoder_layers"]),
        decoder_layers=int(config["decoder_layers"]),
        predictor_layers=int(config["predictor_layers"]),
        heads=int(config["heads"]),
        mlp_ratio=int(config["mlp_ratio"]),
        dropout=float(config["dropout"]),
        vision_model_name=str(config["vision_model_name"]),
        vision_selected_layers=tuple(config["vision_selected_layers"]),
        pretrained_vision=False,
        freeze_vision_backbone=True,
        trainable_backbone_blocks=0,
        fcpg_adapter=None,
        policy_arch=str(config["policy_arch"]),
        reuse_fcpg_vision=False,
        gripper_action_mode=str(config["gripper_action_mode"]),
    )
    policy.vision_encoder.backbone.load_state_dict(corrector.vision.backbone.state_dict())
    policy.vision_encoder.backbone.requires_grad_(False)
    policy.vision_encoder.backbone.eval()

    readout = FKPolicyReadout(
        variant=variant,
        visual_dim=int(config["vision_dim"]),
        obs_dim=int(config["obs_dim"]),
        num_keypoints=int(config["num_keypoints"]),
        model_dim=int(corrector_args.get("model_dim", 256)),
        num_samples=num_samples,
        context_grid_size=int(corrector_args.get("local_grid_size", 5)),
        context_radius_px=float(corrector_args.get("local_radius_px", 16.0)),
        offset_scale_px=offset_scale_px,
        local_layers=int(corrector_args.get("local_layers", 4)),
        global_layers=int(corrector_args.get("global_layers", 2)),
        heads=int(corrector_args.get("heads", 8)),
        dropout=float(corrector_args.get("dropout", 0.1)),
        lattice_step_px=lattice_step_px,
    )
    model = MatchedPolicyOnlyGrounding(policy, readout).to(device)
    counts = model.parameter_counts()
    if counts["frozen_backbone_active"] != 0:
        raise RuntimeError("matched baseline unexpectedly has active backbone parameters")
    active_gap = abs(counts["adapter_active"] - full_corrector_active) / float(
        max(counts["adapter_active"], full_corrector_active, 1)
    )
    if active_gap > active_parameter_tolerance:
        raise ValueError(
            f"adapter active parameter gap {active_gap:.4%} exceeds "
            f"{active_parameter_tolerance:.2%}"
        )
    build_contract: dict[str, object] = {
        "policy_contract_checkpoint": str(policy_path),
        "policy_contract_checkpoint_sha256": sha256_file(policy_path),
        "fcpg_corrector_checkpoint": str(corrector_path),
        "fcpg_corrector_checkpoint_sha256": sha256_file(corrector_path),
        "policy_initialization": "random_matched_seed_no_reference_policy_weights",
        "policy_seed": int(policy_seed),
        "policy_config": config,
        "corrector_arch": str(
            corrector_args.get("corrector_arch", policy_checkpoint.get("corrector_arch", "lewm"))
        ),
        "lewm_upstream_commit": LEWM_UPSTREAM_COMMIT,
        "active_parameter_reference": "fcpg_full_non_backbone_trainable",
        "active_parameter_tolerance": float(active_parameter_tolerance),
        "reference_fcpg_full_corrector_active": full_corrector_active,
        "reference_fcpg_grounding_active": grounding_active,
        "adapter_active_relative_gap": active_gap,
        "padding_parameters": 0,
        "backbone_state_sha256": module_state_sha256(
            model.policy.vision_encoder.backbone
        ),
        "sampling": {
            "policy_visual_samples_per_anchor": int(num_samples),
            "context_visual_samples_per_anchor": int(readout.context_samples_per_anchor),
            "weighted_bilinear_sampling": True,
            "policy_token": "weighted_visual_then_shared_grounded_token_adapter",
            "reference_points": "fk_q_nominal",
            "offset_parameterization": "20*tanh",
            "offset_scale_px": float(readout.offset_scale_px),
            "initialization": "fixed_1px_eight_neighbor_lattice",
            "lattice_step_px": float(lattice_step_px),
            "context_radius_px": float(readout.context_radius_px),
            "coordinate_diagnostic": "attention_barycenter_epe",
        },
    }
    return model, build_contract


def checkpoint_model_state(
    model: MatchedPolicyOnlyGrounding,
) -> dict[str, torch.Tensor]:
    """Store active state while reconstructing frozen vision from its source."""

    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("policy.vision_encoder.backbone.")
    }


def validate_checkpoint_contract(checkpoint: Mapping[str, object]) -> None:
    if int(checkpoint.get("schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported FK-DeformAttn checkpoint schema")
    if checkpoint.get("checkpoint_type") != CHECKPOINT_TYPE:
        raise ValueError("checkpoint type is not a matched policy-only grounding model")
    if checkpoint.get("variant") not in VARIANTS:
        raise ValueError("checkpoint has an unknown grounding variant")
    if checkpoint.get("loss_contract") != LOSS_CONTRACT:
        raise ValueError("checkpoint loss contract is missing or inconsistent")
    if checkpoint.get("structural_attribution_contract") != STRUCTURAL_ATTRIBUTION_CONTRACT:
        raise ValueError("checkpoint structural attribution contract is inconsistent")
    counts = checkpoint.get("parameter_counts")
    if not isinstance(counts, Mapping):
        raise TypeError("checkpoint parameter_counts must be a mapping")
    if int(counts.get("frozen_backbone_active", -1)) != 0:
        raise ValueError("checkpoint does not declare a frozen ViT backbone")
    if int(counts.get("adapter_active", -1)) <= 0:
        raise ValueError("checkpoint has an invalid active adapter budget")
    selection = checkpoint.get("selection_contract")
    if not isinstance(selection, Mapping):
        raise TypeError("checkpoint selection_contract must be a mapping")
    if selection.get("policy") != SELECTION_POLICY:
        raise ValueError("checkpoint selection policy is not independently selected best")
    if selection.get("selected_step_relationship") != "independent_by_variant":
        raise ValueError("checkpoint selection contract makes an invalid step claim")
    run = checkpoint.get("pair_run_contract")
    if not isinstance(run, Mapping):
        raise TypeError("checkpoint pair_run_contract must be a mapping")
    run_id = run.get("pair_run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("checkpoint pair run id must be non-empty")
    if run.get("selection_policy") != selection.get("policy"):
        raise ValueError("checkpoint selection policy is not bound to its pair run")
    maximum_budget = run.get("maximum_budget")
    realized_budget = run.get("realized_budget")
    if not isinstance(maximum_budget, Mapping) or not isinstance(
        realized_budget, Mapping
    ):
        raise TypeError("checkpoint pair run budgets must be mappings")
    if selection.get("maximum_budget") != maximum_budget:
        raise ValueError("checkpoint selection and pair run maximum budgets differ")
    maximum_epochs = int(maximum_budget.get("epochs", -1))
    maximum_steps = int(maximum_budget.get("optimizer_steps_per_variant", -1))
    realized_epochs = int(realized_budget.get("epochs", -1))
    realized_steps = int(realized_budget.get("optimizer_steps_per_variant", -1))
    if maximum_epochs <= 0 or maximum_steps < 0:
        raise ValueError("checkpoint maximum training budget is invalid")
    if (
        realized_epochs <= 0
        or realized_steps < 0
        or realized_epochs > maximum_epochs
        or realized_steps > maximum_steps
    ):
        raise ValueError("checkpoint realized training budget is invalid")
    if run.get("shared_training_run") is not True:
        raise ValueError("checkpoint does not declare one shared paired training run")

    checkpoint_state = checkpoint.get("checkpoint_state")
    if checkpoint_state not in ("best", "final"):
        raise ValueError("checkpoint_state must be 'best' or 'final'")
    selected = checkpoint.get("selected")
    if not isinstance(selected, Mapping):
        raise TypeError("checkpoint selected metadata must be a mapping")
    selected_epoch = int(selected.get("epoch", -1))
    selected_steps = int(selected.get("optimizer_steps", -1))
    if (
        selected_epoch <= 0
        or selected_steps < 0
        or selected_epoch > realized_epochs
        or selected_steps > realized_steps
    ):
        raise ValueError("checkpoint selected state exceeds the shared realized budget")
    expected_role = (
        "independent_validation_best"
        if checkpoint_state == "best"
        else "final_at_matched_budget"
    )
    if selected.get("selection_role") != expected_role:
        raise ValueError("checkpoint selected state role is inconsistent")
    if checkpoint_state == "final" and (
        selected_epoch != realized_epochs or selected_steps != realized_steps
    ):
        raise ValueError("final checkpoint is not at the shared realized budget")

    cache_validation = checkpoint.get("vision_cache_validation")
    if not isinstance(cache_validation, Mapping):
        raise TypeError("checkpoint vision_cache_validation must be a mapping")
    if cache_validation.get("status") != "validated_before_training":
        raise ValueError("checkpoint lacks pre-training vision cache validation")
    if cache_validation.get("required_model_name") != AUDITED_VISION_MODEL:
        raise ValueError("checkpoint vision cache contract is not audited ViT-B")
    if tuple(cache_validation.get("required_selected_layers", ())) != (
        AUDITED_VISION_SELECTED_LAYERS
    ):
        raise ValueError("checkpoint vision cache contract is not layer 11")
    cache_sources = cache_validation.get("sources")
    if not isinstance(cache_sources, (tuple, list)) or not cache_sources:
        raise ValueError("checkpoint vision cache validation has no sources")

    claim = checkpoint.get("claim_contract")
    if not isinstance(claim, Mapping):
        raise TypeError("checkpoint claim_contract must be a mapping")
    if (
        claim.get("scope") != DEVELOPMENT_CLAIM_SCOPE
        or int(claim.get("training_seed", -1)) != 0
        or claim.get("development_only") is not True
        or claim.get("controlled_comparison_only") is not True
        or claim.get("cross_seed_generalization_supported") is not False
        or claim.get("general_claim_supported") is not False
    ):
        raise ValueError("checkpoint claim scope exceeds seed0 development evidence")

    runtime_budget = checkpoint.get("runtime_budget")
    if not isinstance(runtime_budget, Mapping):
        raise TypeError("checkpoint runtime_budget must be a mapping")
    structural_latency = runtime_budget.get("structural_latency")
    if not isinstance(structural_latency, Mapping):
        raise TypeError("checkpoint structural latency must be a mapping")
    latency_contract = structural_latency.get("comparison_contract")
    latency_measurement = structural_latency.get("measurement")
    if not isinstance(latency_contract, Mapping) or not isinstance(
        latency_measurement, Mapping
    ):
        raise TypeError("checkpoint structural latency record is incomplete")
    if (
        latency_contract.get("protocol")
        != "fixed-input-equal-count-warmup-interleaved-v1"
        or latency_contract.get("interleaved_order")
        != "alternating-forward-reverse-by-iteration"
        or latency_contract.get("equal_call_count_per_variant") is not True
        or latency_contract.get("rollout_call_counts_used") is not False
        or int(latency_contract.get("batch_size", 0)) != 1
        or int(latency_contract.get("warmup_iterations_per_variant", 0)) <= 0
        or int(latency_contract.get("timed_iterations_per_variant", 0)) <= 0
    ):
        raise ValueError("checkpoint structural latency contract is invalid")
    fixed_input_sha256 = latency_contract.get("fixed_input_sha256")
    if not isinstance(fixed_input_sha256, str) or len(fixed_input_sha256) != 64:
        raise ValueError("checkpoint structural latency lacks a fixed input identity")
    if latency_contract.get("comparison_id") != f"{run_id}:{checkpoint_state}":
        raise ValueError("checkpoint latency comparison is not bound to run and state")
    if latency_measurement.get("variant") != checkpoint.get("variant"):
        raise ValueError("checkpoint latency measurement has the wrong variant")
    if int(latency_measurement.get("timed_iterations", -1)) != int(
        latency_contract["timed_iterations_per_variant"]
    ):
        raise ValueError("checkpoint latency measurement call count is inconsistent")


def load_matched_policy_only_grounding(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    *,
    policy_contract_checkpoint: str | Path | None = None,
    fcpg_corrector_checkpoint: str | Path | None = None,
) -> tuple[MatchedPolicyOnlyGrounding, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint_contract(checkpoint)
    build = checkpoint["build_contract"]
    if not isinstance(build, Mapping):
        raise TypeError("checkpoint build_contract must be a mapping")
    policy_path = Path(
        policy_contract_checkpoint or str(build["policy_contract_checkpoint"])
    ).expanduser().resolve()
    corrector_path = Path(
        fcpg_corrector_checkpoint or str(build["fcpg_corrector_checkpoint"])
    ).expanduser().resolve()
    if sha256_file(policy_path) != str(build["policy_contract_checkpoint_sha256"]):
        raise ValueError("policy contract checkpoint hash does not match")
    if sha256_file(corrector_path) != str(build["fcpg_corrector_checkpoint_sha256"]):
        raise ValueError("FCPG corrector checkpoint hash does not match")
    counts = checkpoint["parameter_counts"]
    assert isinstance(counts, Mapping)
    sampling = build["sampling"]
    if not isinstance(sampling, Mapping):
        raise TypeError("checkpoint sampling contract must be a mapping")
    model, reconstructed = build_matched_policy_only_grounding(
        variant=str(checkpoint["variant"]),
        policy_contract_checkpoint=policy_path,
        fcpg_corrector_checkpoint=corrector_path,
        policy_seed=int(build["policy_seed"]),
        num_samples=int(sampling["policy_visual_samples_per_anchor"]),
        lattice_step_px=float(sampling["lattice_step_px"]),
        offset_scale_px=float(sampling["offset_scale_px"]),
        active_parameter_tolerance=float(build["active_parameter_tolerance"]),
        device=device,
    )
    if reconstructed["backbone_state_sha256"] != build["backbone_state_sha256"]:
        raise ValueError("reconstructed frozen backbone state does not match checkpoint")
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    invalid_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("policy.vision_encoder.backbone.")
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "invalid matched policy-only state: "
            f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
        )
    normalization = checkpoint.get("normalization")
    if not isinstance(normalization, Mapping):
        raise TypeError("checkpoint normalization must be a mapping")
    model.set_normalization(normalization)
    if model.parameter_counts() != dict(counts):
        raise ValueError("reconstructed parameter counts do not match checkpoint")
    model.eval()
    return model, dict(checkpoint)


def validate_structural_pair(
    first: Mapping[str, object],
    second: Mapping[str, object],
    *,
    active_parameter_tolerance: float = 0.01,
) -> dict[str, object]:
    """Validate that two checkpoints support a policy-only structural claim."""

    validate_checkpoint_contract(first)
    validate_checkpoint_contract(second)
    by_variant = {str(first["variant"]): first, str(second["variant"]): second}
    if set(by_variant) != set(VARIANTS):
        raise ValueError("structural evaluation requires one checkpoint of each variant")
    residual = by_variant["policy_only_residual"]
    deform = by_variant["fk_deform_attn"]
    matched_fields = (
        "data_contract",
        "training_contract",
        "selection_contract",
        "pair_run_contract",
        "vision_cache_validation",
        "claim_contract",
    )
    for field in matched_fields:
        if residual.get(field) != deform.get(field):
            raise ValueError(f"pair mismatch in {field}")
    residual_normalization = residual.get("normalization")
    deform_normalization = deform.get("normalization")
    if not isinstance(residual_normalization, Mapping) or not isinstance(
        deform_normalization, Mapping
    ):
        raise TypeError("pair normalization values must be mappings")
    if set(residual_normalization) != set(deform_normalization):
        raise ValueError("pair normalization keys differ")
    for key in residual_normalization:
        residual_value = torch.as_tensor(residual_normalization[key]).detach().cpu()
        deform_value = torch.as_tensor(deform_normalization[key]).detach().cpu()
        if residual_value.dtype != deform_value.dtype or not torch.equal(
            residual_value, deform_value
        ):
            raise ValueError(f"pair normalization mismatch in {key}")
    checkpoint_state = str(residual["checkpoint_state"])
    if checkpoint_state != str(deform["checkpoint_state"]):
        raise ValueError("pair checkpoint_state mismatch; best/final mixing is forbidden")
    run_contract = residual["pair_run_contract"]
    selection_contract = residual["selection_contract"]
    assert isinstance(run_contract, Mapping)
    assert isinstance(selection_contract, Mapping)
    if run_contract.get("selection_policy") != SELECTION_POLICY:
        raise ValueError("pair run does not use the audited independent-best policy")
    residual_build = residual["build_contract"]
    deform_build = deform["build_contract"]
    if not isinstance(residual_build, Mapping) or not isinstance(deform_build, Mapping):
        raise TypeError("pair build contracts must be mappings")
    for field in (
        "policy_contract_checkpoint_sha256",
        "fcpg_corrector_checkpoint_sha256",
        "policy_seed",
        "policy_config",
        "active_parameter_reference",
        "active_parameter_tolerance",
        "padding_parameters",
        "backbone_state_sha256",
        "sampling",
    ):
        if residual_build.get(field) != deform_build.get(field):
            raise ValueError(f"pair build mismatch in {field}")
    residual_counts = residual["parameter_counts"]
    deform_counts = deform["parameter_counts"]
    assert isinstance(residual_counts, Mapping) and isinstance(deform_counts, Mapping)
    for field in (
        "common_policy_active",
        "frozen_backbone_active",
        "vision_neck_active",
    ):
        if int(residual_counts[field]) != int(deform_counts[field]):
            raise ValueError(f"pair parameter mismatch in {field}")
    residual_active = int(residual_counts["total_active"])
    deform_active = int(deform_counts["total_active"])
    relative_gap = abs(residual_active - deform_active) / float(
        max(residual_active, deform_active, 1)
    )
    if relative_gap > active_parameter_tolerance:
        raise ValueError(
            f"pair active parameter gap {relative_gap:.4%} exceeds "
            f"{active_parameter_tolerance:.2%}"
        )
    residual_runtime = residual["runtime_budget"]
    deform_runtime = deform["runtime_budget"]
    assert isinstance(residual_runtime, Mapping) and isinstance(deform_runtime, Mapping)
    residual_latency = residual_runtime["structural_latency"]
    deform_latency = deform_runtime["structural_latency"]
    assert isinstance(residual_latency, Mapping) and isinstance(deform_latency, Mapping)
    latency_contract = residual_latency["comparison_contract"]
    if latency_contract != deform_latency["comparison_contract"]:
        raise ValueError("pair structural latency comparison contracts differ")
    residual_selected = residual["selected"]
    deform_selected = deform["selected"]
    assert isinstance(residual_selected, Mapping) and isinstance(deform_selected, Mapping)
    selection_interpretation = (
        SELECTION_POLICY
        if checkpoint_state == "best"
        else "matched-final-at-shared-budget"
    )
    return {
        "structural_attribution_allowed": True,
        "allowed_pair": VARIANTS,
        "active_parameter_relative_gap": relative_gap,
        "supervised_fcpg_comparison_scope": "system_only",
        "checkpoint_state": checkpoint_state,
        "selection_policy": SELECTION_POLICY,
        "selection_interpretation": selection_interpretation,
        "selected_epochs": {
            "policy_only_residual": int(residual_selected["epoch"]),
            "fk_deform_attn": int(deform_selected["epoch"]),
        },
        "selected_optimizer_steps": {
            "policy_only_residual": int(residual_selected["optimizer_steps"]),
            "fk_deform_attn": int(deform_selected["optimizer_steps"]),
        },
        "same_selected_step_claim_supported": False,
        "matched_maximum_budget": run_contract["maximum_budget"],
        "matched_realized_budget": run_contract["realized_budget"],
        "pair_run_id": run_contract["pair_run_id"],
        "claim_contract": residual["claim_contract"],
        "matched_backbone_state_sha256": residual_build["backbone_state_sha256"],
        "matched_sampling_budget": residual_build["sampling"],
        "structural_latency_contract": latency_contract,
    }
