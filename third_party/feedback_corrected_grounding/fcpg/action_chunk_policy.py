"""Deterministic transformer action-chunk policy for grounding experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from fcpg.lewm_core import LEWM_UPSTREAM_COMMIT, LeWMARPredictor, LeWMTransformer
from fcpg.modules import GroundedTokenAdapter
from fcpg.sequence_data import NormalizationStats
from fcpg.transformer_grounding import (
    FCPGViTAdapter,
    LeWMFCPGAdapter,
    ViTDenseEncoder,
    load_fcpg_vit_adapter,
)


GROUNDING_MODES = ("vision", "geoprop", "fcpg", "oracle")
POLICY_ARCHES = ("legacy", "lewm-predictive")
RELIABILITY_OVERRIDES = ("native", "ones", "zeros")
GRIPPER_ACTION_MODES = ("signed", "legacy_unsigned")


def fcpg_robot_observation(
    policy_observation: torch.Tensor,
    fcpg_adapter: FCPGViTAdapter | LeWMFCPGAdapter | nn.Module,
) -> torch.Tensor:
    """Select the robot-only observation expected by a frozen FCPG adapter."""

    grounding = getattr(fcpg_adapter, "grounding", None)
    expected_dim = getattr(grounding, "obs_dim", None)
    if expected_dim is None:
        expected_dim = getattr(fcpg_adapter, "obs_dim", policy_observation.shape[-1])
    expected_dim = int(expected_dim)
    actual_dim = int(policy_observation.shape[-1])
    if expected_dim == actual_dim:
        return policy_observation
    if expected_dim == 4 and actual_dim == 7:
        return policy_observation[..., :4]
    raise ValueError(
        "FCPG observation mismatch: frozen corrector expects "
        f"{expected_dim} values but policy provides {actual_dim}"
    )


def encode_gripper_target(target: torch.Tensor, mode: str) -> torch.Tensor:
    """Convert environment gripper actions to BCE targets."""

    if mode == "signed":
        if not bool(torch.isfinite(target).all()):
            raise ValueError("signed gripper targets must be finite")
        if bool((target < -1.0001).any()) or bool((target > 1.0001).any()):
            raise ValueError("signed gripper targets must lie in [-1, 1]")
        return (target + 1.0) * 0.5
    if mode == "legacy_unsigned":
        return target
    raise ValueError(f"unknown gripper action mode {mode!r}")


def decode_gripper_action(logit: torch.Tensor, mode: str) -> torch.Tensor:
    """Convert a binary gripper logit to the environment action space."""

    probability = torch.sigmoid(logit)
    if mode == "signed":
        return probability * 2.0 - 1.0
    if mode == "legacy_unsigned":
        return probability
    raise ValueError(f"unknown gripper action mode {mode!r}")


def reliability_override_value(mode: str) -> float | None:
    if mode == "native":
        return None
    if mode == "ones":
        return 1.0
    if mode == "zeros":
        return 0.0
    raise ValueError(f"unknown reliability override {mode!r}")


@dataclass(frozen=True)
class ActionChunkOutput:
    action: torch.Tensor
    motion: torch.Tensor
    gripper_logit: torch.Tensor
    latent: torch.Tensor
    grounding_points_px: torch.Tensor
    reliability: torch.Tensor
    state_latent: torch.Tensor | None = None
    next_latent: torch.Tensor | None = None
    next_latent_target: torch.Tensor | None = None


class GroundedActionChunkTransformer(nn.Module):
    """Predict a complete future action chunk from grounded visual memory."""

    def __init__(
        self,
        mode: str,
        obs_dim: int,
        action_dim: int,
        num_keypoints: int,
        obs_horizon: int = 2,
        action_horizon: int = 32,
        model_dim: int = 256,
        vision_dim: int = 256,
        encoder_layers: int = 2,
        decoder_layers: int = 6,
        heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        vision_model_name: str = "vit_base_patch16_224.augreg_in1k",
        vision_selected_layers: Sequence[int] = (2, 5, 8, 11),
        pretrained_vision: bool = True,
        freeze_vision_backbone: bool = True,
        trainable_backbone_blocks: int = 0,
        fcpg_adapter: FCPGViTAdapter | LeWMFCPGAdapter | None = None,
        policy_arch: str = "legacy",
        predictor_layers: int = 2,
        reuse_fcpg_vision: bool = False,
        gripper_action_mode: str = "signed",
    ) -> None:
        super().__init__()
        if mode not in GROUNDING_MODES:
            raise ValueError(f"unknown grounding mode {mode!r}")
        if policy_arch not in POLICY_ARCHES:
            raise ValueError(f"unknown policy architecture {policy_arch!r}")
        if gripper_action_mode not in GRIPPER_ACTION_MODES:
            raise ValueError(f"unknown gripper action mode {gripper_action_mode!r}")
        if action_dim != 4:
            raise ValueError("MetaWorld action chunk policy expects 4D actions")
        if model_dim % heads != 0:
            raise ValueError(f"model_dim={model_dim} must be divisible by heads={heads}")
        if mode == "fcpg" and fcpg_adapter is None:
            raise ValueError("fcpg mode requires a pretrained FCPG adapter")
        self.mode = str(mode)
        self.policy_arch = str(policy_arch)
        self.gripper_action_mode = str(gripper_action_mode)
        self.predictor_layers = int(predictor_layers)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.num_keypoints = int(num_keypoints)
        self.obs_horizon = int(obs_horizon)
        self.action_horizon = int(action_horizon)
        self.model_dim = int(model_dim)
        self.vision_dim = int(vision_dim)
        self.vision_model_name = str(vision_model_name)
        self.vision_selected_layers = tuple(int(layer) for layer in vision_selected_layers)
        self.tokens_per_observation = 2 + self.num_keypoints

        self.vision_encoder = ViTDenseEncoder(
            model_name=self.vision_model_name,
            pretrained=pretrained_vision,
            selected_layers=self.vision_selected_layers,
            output_dim=self.vision_dim,
            freeze_backbone=freeze_vision_backbone,
            trainable_backbone_blocks=trainable_backbone_blocks,
        )
        self.global_visual_projection = nn.Linear(self.vision_dim, self.model_dim)
        self.robot_projection = nn.Sequential(
            nn.LayerNorm(self.obs_dim),
            nn.Linear(self.obs_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.grounding_adapter = GroundedTokenAdapter(
            visual_dim=self.vision_dim,
            proprio_dim=3 + self.num_keypoints,
            token_dim=self.model_dim,
            hidden_dim=self.model_dim,
            gate_visual_only=True,
            include_reliability=True,
        )
        self.fcpg_adapter = fcpg_adapter
        self.reuse_fcpg_vision = bool(reuse_fcpg_vision)
        self.runtime_reliability_override: float | None = None
        if self.fcpg_adapter is not None:
            self.fcpg_adapter.eval()
            self.fcpg_adapter.requires_grad_(False)
        if self.reuse_fcpg_vision:
            if self.fcpg_adapter is None:
                raise ValueError("reuse_fcpg_vision requires a pretrained FCPG adapter")
            self.vision_encoder.load_state_dict(self.fcpg_adapter.vision.state_dict())
            self.vision_encoder.requires_grad_(False)
            self.vision_encoder.eval()
        self.register_buffer("obs_mean", torch.zeros(self.obs_dim))
        self.register_buffer("obs_std", torch.ones(self.obs_dim))

        feedforward_dim = self.model_dim * int(mlp_ratio)
        if self.policy_arch == "legacy":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.model_dim,
                nhead=heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.model_dim,
                nhead=heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.memory_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=int(encoder_layers),
                norm=nn.LayerNorm(self.model_dim),
            )
            self.action_decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=int(decoder_layers),
                norm=nn.LayerNorm(self.model_dim),
            )
            self.next_latent_predictor = None
        else:
            self.memory_encoder = LeWMTransformer(
                input_dim=self.model_dim,
                hidden_dim=self.model_dim,
                output_dim=self.model_dim,
                depth=int(encoder_layers),
                heads=int(heads),
                mlp_dim=feedforward_dim,
                dropout=dropout,
                causal=False,
                conditional=False,
            )
            self.action_decoder = LeWMTransformer(
                input_dim=self.model_dim,
                hidden_dim=self.model_dim,
                output_dim=self.model_dim,
                depth=int(decoder_layers),
                heads=int(heads),
                mlp_dim=feedforward_dim,
                dropout=dropout,
                causal=False,
                conditional=True,
                condition_dim=self.model_dim,
            )
            self.next_latent_predictor = LeWMARPredictor(
                latent_dim=self.model_dim,
                action_dim=self.action_dim,
                max_frames=self.obs_horizon,
                depth=self.predictor_layers,
                heads=int(heads),
                mlp_dim=feedforward_dim,
                dropout=dropout,
            )
        memory_tokens = self.obs_horizon * self.tokens_per_observation
        self.memory_position = nn.Parameter(torch.zeros(1, memory_tokens, self.model_dim))
        self.action_queries = nn.Parameter(torch.zeros(1, self.action_horizon, self.model_dim))
        self.action_position = nn.Parameter(torch.zeros(1, self.action_horizon, self.model_dim))
        self.motion_head = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, 3),
        )
        self.gripper_head = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, 1),
        )
        nn.init.trunc_normal_(self.memory_position, std=0.02)
        nn.init.trunc_normal_(self.action_queries, std=0.02)
        nn.init.trunc_normal_(self.action_position, std=0.02)

    def train(self, mode: bool = True) -> "GroundedActionChunkTransformer":
        super().train(mode)
        if self.fcpg_adapter is not None:
            self.fcpg_adapter.eval()
        if self.reuse_fcpg_vision:
            self.vision_encoder.eval()
        return self

    def set_reliability_override(self, mode: str) -> None:
        self.runtime_reliability_override = reliability_override_value(mode)

    def set_normalization(self, stats: NormalizationStats | dict[str, torch.Tensor]) -> None:
        values = stats.as_torch() if isinstance(stats, NormalizationStats) else stats
        for name in ("obs_mean", "obs_std"):
            target = getattr(self, name)
            source = values[name].to(device=target.device, dtype=target.dtype)
            if source.shape != target.shape:
                raise ValueError(f"normalizer {name} has shape {source.shape}, expected {target.shape}")
            target.copy_(source)

    def _proprio(self, xyz: torch.Tensor) -> torch.Tensor:
        batch = xyz.shape[0]
        onehot = torch.eye(
            self.num_keypoints,
            device=xyz.device,
            dtype=xyz.dtype,
        ).unsqueeze(0).expand(batch, -1, -1)
        return torch.cat([xyz, onehot], dim=-1)

    def _apply_grounded_token_override(
        self,
        batch: dict[str, torch.Tensor],
        grounded: torch.Tensor,
        *,
        batch_size: int,
        obs_steps: int,
        prefix: str = "",
    ) -> torch.Tensor:
        """Replace selected native grounding tokens without touching other memory."""

        def key(name: str) -> str:
            return f"{prefix}{name}" if prefix else name

        override_key = key("grounded_token_override")
        mask_key = key("grounded_token_override_mask")
        has_override = override_key in batch
        has_mask = mask_key in batch
        if not has_override and not has_mask:
            return grounded
        if has_override != has_mask:
            missing = mask_key if has_override else override_key
            raise KeyError(
                "grounded token override requires both "
                f"{override_key!r} and {mask_key!r}; missing {missing!r}"
            )

        override = batch[override_key]
        mask = batch[mask_key]
        if not isinstance(override, torch.Tensor):
            raise TypeError(f"{override_key} must be a torch.Tensor")
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"{mask_key} must be a torch.Tensor")

        expected_override = (
            batch_size,
            obs_steps,
            self.num_keypoints,
            self.model_dim,
        )
        expected_mask = (batch_size, obs_steps, self.num_keypoints)
        if override.shape != expected_override:
            raise ValueError(
                f"{override_key} must have shape {expected_override}, "
                f"got {tuple(override.shape)}"
            )
        if mask.shape != expected_mask:
            raise ValueError(
                f"{mask_key} must have shape {expected_mask}, got {tuple(mask.shape)}"
            )
        if override.dtype != grounded.dtype:
            raise TypeError(
                f"{override_key} dtype must match native grounded tokens "
                f"({grounded.dtype}), got {override.dtype}"
            )
        if mask.dtype != torch.bool:
            raise TypeError(f"{mask_key} must use torch.bool, got {mask.dtype}")
        if override.device != grounded.device:
            raise ValueError(
                f"{override_key} device must match native grounded tokens "
                f"({grounded.device}), got {override.device}"
            )
        if mask.device != grounded.device:
            raise ValueError(
                f"{mask_key} device must match native grounded tokens "
                f"({grounded.device}), got {mask.device}"
            )

        flat_override = override.reshape(
            batch_size * obs_steps,
            self.num_keypoints,
            self.model_dim,
        )
        flat_mask = mask.reshape(batch_size * obs_steps, self.num_keypoints)
        selected = flat_override[flat_mask]
        if selected.numel() and not bool(torch.isfinite(selected).all()):
            raise ValueError(f"{override_key} contains non-finite selected tokens")
        return torch.where(flat_mask.unsqueeze(-1), flat_override, grounded)

    def _apply_global_visual_token_override(
        self,
        batch: dict[str, torch.Tensor],
        global_visual: torch.Tensor,
        *,
        batch_size: int,
        obs_steps: int,
        prefix: str = "",
    ) -> torch.Tensor:
        """Replace selected projected global tokens while preserving native vision."""

        def key(name: str) -> str:
            return f"{prefix}{name}" if prefix else name

        override_key = key("global_visual_token_override")
        mask_key = key("global_visual_token_override_mask")
        has_override = override_key in batch
        has_mask = mask_key in batch
        if not has_override and not has_mask:
            return global_visual
        if has_override != has_mask:
            missing = mask_key if has_override else override_key
            raise KeyError(
                "global visual token override requires both "
                f"{override_key!r} and {mask_key!r}; missing {missing!r}"
            )

        override = batch[override_key]
        mask = batch[mask_key]
        if not isinstance(override, torch.Tensor):
            raise TypeError(f"{override_key} must be a torch.Tensor")
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"{mask_key} must be a torch.Tensor")

        expected_override = (batch_size, obs_steps, self.model_dim)
        expected_mask = (batch_size, obs_steps)
        if override.shape != expected_override:
            raise ValueError(
                f"{override_key} must have shape {expected_override}, "
                f"got {tuple(override.shape)}"
            )
        if mask.shape != expected_mask:
            raise ValueError(
                f"{mask_key} must have shape {expected_mask}, got {tuple(mask.shape)}"
            )
        if override.dtype != global_visual.dtype:
            raise TypeError(
                f"{override_key} dtype must match native global tokens "
                f"({global_visual.dtype}), got {override.dtype}"
            )
        if mask.dtype != torch.bool:
            raise TypeError(f"{mask_key} must use torch.bool, got {mask.dtype}")
        if override.device != global_visual.device:
            raise ValueError(
                f"{override_key} device must match native global tokens "
                f"({global_visual.device}), got {override.device}"
            )
        if mask.device != global_visual.device:
            raise ValueError(
                f"{mask_key} device must match native global tokens "
                f"({global_visual.device}), got {mask.device}"
            )

        flat_override = override.reshape(batch_size * obs_steps, self.model_dim)
        flat_mask = mask.reshape(batch_size * obs_steps)
        selected = flat_override[flat_mask]
        if selected.numel() and not bool(torch.isfinite(selected).all()):
            raise ValueError(f"{override_key} contains non-finite selected tokens")
        return torch.where(flat_mask[:, None, None], flat_override[:, None], global_visual)

    def _observation_tokens(
        self,
        batch: dict[str, torch.Tensor],
        prefix: str = "",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def key(name: str) -> str:
            return f"{prefix}{name}" if prefix else name

        obs = batch[key("obs")]
        q_nominal = batch[key("q_nominal")]
        q_true = batch.get(key("q_true"))
        xyz = batch[key("keypoints_xyz")]
        batch_size, obs_steps, _ = obs.shape
        if obs_steps != self.obs_horizon:
            raise ValueError(f"expected obs_horizon={self.obs_horizon}, got {obs_steps}")

        rgb_key = key("rgb")
        feature_key = key("vision_backbone_features")
        prefix_key = key("vision_prefix")
        sampled_key = key("sampled_visual")
        global_key = key("global_visual")
        has_compact_visual = sampled_key in batch or global_key in batch
        if has_compact_visual:
            if sampled_key not in batch or global_key not in batch:
                raise KeyError("compact policy cache requires sampled_visual and global_visual")
            image_sizes = batch["image_size"]
            if not bool(torch.all(image_sizes == image_sizes[0])):
                raise ValueError("all cached samples in a batch must share one image size")
            width, height = (int(value) for value in image_sizes[0].tolist())
            sampled_visual = batch[sampled_key]
            global_visual = batch[global_key]
            if sampled_visual.shape[:3] != (
                batch_size,
                obs_steps,
                self.num_keypoints,
            ):
                raise ValueError(
                    "sampled_visual must have shape [B, H, K, C], "
                    f"got {tuple(sampled_visual.shape)}"
                )
            if sampled_visual.shape[-1] != self.vision_dim:
                raise ValueError(
                    f"expected compact visual dim {self.vision_dim}, "
                    f"got {sampled_visual.shape[-1]}"
                )
            if global_visual.shape != (
                batch_size,
                obs_steps,
                self.vision_dim,
            ):
                raise ValueError(
                    "global_visual must have shape [B, H, C], "
                    f"got {tuple(global_visual.shape)}"
                )
            flat_sampled_visual = sampled_visual.reshape(
                batch_size * obs_steps,
                self.num_keypoints,
                self.vision_dim,
            )
            flat_global_visual = global_visual.reshape(
                batch_size * obs_steps,
                self.vision_dim,
            )
            vision = None
            flat_rgb = None
            visual_dtype = flat_sampled_visual.dtype
        elif rgb_key in batch:
            rgb = batch[rgb_key]
            _, _, _, height, width = rgb.shape
            flat_rgb = rgb.reshape(batch_size * obs_steps, 3, height, width)
            vision = self.vision_encoder(flat_rgb)
            flat_sampled_visual = None
            flat_global_visual = None
            visual_dtype = vision.feature_map.dtype
        else:
            if feature_key not in batch or prefix_key not in batch:
                raise KeyError(
                    f"batch requires {rgb_key}, compact visual features, or cached "
                    f"ViT features with prefix {prefix!r}"
                )
            image_sizes = batch["image_size"]
            if not bool(torch.all(image_sizes == image_sizes[0])):
                raise ValueError("all cached samples in a batch must share one image size")
            width, height = (int(value) for value in image_sizes[0].tolist())
            cached_features = batch[feature_key]
            cached_prefix = batch[prefix_key]
            flat_features = cached_features.reshape(
                batch_size * obs_steps,
                *cached_features.shape[2:],
            )
            flat_prefix = cached_prefix.reshape(
                batch_size * obs_steps,
                cached_prefix.shape[-1],
            )
            vision = self.vision_encoder.project_backbone_features(
                flat_features,
                flat_prefix,
            )
            flat_rgb = None
            flat_sampled_visual = None
            flat_global_visual = None
            visual_dtype = vision.feature_map.dtype

        flat_obs = obs.reshape(batch_size * obs_steps, self.obs_dim)
        flat_q_nominal = q_nominal.reshape(
            batch_size * obs_steps,
            self.num_keypoints,
            2,
        )
        flat_xyz = xyz.reshape(
            batch_size * obs_steps,
            self.num_keypoints,
            3,
        )

        if self.mode == "fcpg":
            corrected_key = key("fcpg_q_corrected")
            reliability_key = key("fcpg_reliability")
            if corrected_key in batch:
                points = batch[corrected_key].reshape(
                    batch_size * obs_steps,
                    self.num_keypoints,
                    2,
                )
                if reliability_key in batch:
                    reliability = batch[reliability_key].reshape(
                        batch_size * obs_steps,
                        self.num_keypoints,
                    )
                else:
                    reliability = torch.ones(
                        batch_size * obs_steps,
                        self.num_keypoints,
                        device=obs.device,
                        dtype=visual_dtype,
                    )
            else:
                assert self.fcpg_adapter is not None
                if flat_rgb is None:
                    raise ValueError(
                        "fcpg mode with cached visual features also requires cached FCPG corrections"
                    )
                with torch.no_grad():
                    correction = self.fcpg_adapter(
                        flat_rgb,
                        flat_q_nominal,
                        flat_xyz,
                        fcpg_robot_observation(flat_obs, self.fcpg_adapter),
                    )
                points = correction.q_corrected_px
                predicted_reliability = getattr(correction, "reliability", None)
                reliability = (
                    predicted_reliability
                    if predicted_reliability is not None
                    else torch.ones(
                        batch_size * obs_steps,
                        self.num_keypoints,
                        device=obs.device,
                        dtype=visual_dtype,
                    )
                )
        elif self.mode == "oracle":
            if q_true is None:
                raise ValueError("oracle mode requires q_true")
            points = q_true.reshape(
                batch_size * obs_steps,
                self.num_keypoints,
                2,
            )
            reliability = torch.ones(
                batch_size * obs_steps,
                self.num_keypoints,
                device=obs.device,
                dtype=visual_dtype,
            )
        else:
            points = flat_q_nominal
            reliability = torch.ones(
                batch_size * obs_steps,
                self.num_keypoints,
                device=obs.device,
                dtype=visual_dtype,
            )

        if self.runtime_reliability_override is not None:
            reliability = torch.full_like(
                reliability,
                self.runtime_reliability_override,
            )

        if self.mode == "vision":
            grounded = torch.zeros(
                batch_size * obs_steps,
                self.num_keypoints,
                self.model_dim,
                device=obs.device,
                dtype=visual_dtype,
            )
        elif flat_sampled_visual is not None:
            grounded = self.grounding_adapter.forward_sampled(
                flat_sampled_visual,
                self._proprio(flat_xyz),
                reliability=reliability,
            )
        else:
            assert vision is not None
            grounded = self.grounding_adapter(
                vision.feature_map,
                points,
                self._proprio(flat_xyz),
                image_size=(width, height),
                reliability=reliability,
            )
        grounded = self._apply_grounded_token_override(
            batch,
            grounded,
            batch_size=batch_size,
            obs_steps=obs_steps,
            prefix=prefix,
        )
        if flat_global_visual is None:
            assert vision is not None
            flat_global_visual = vision.global_feature
        global_visual = self.global_visual_projection(
            flat_global_visual
        ).unsqueeze(1)
        global_visual = self._apply_global_visual_token_override(
            batch,
            global_visual,
            batch_size=batch_size,
            obs_steps=obs_steps,
            prefix=prefix,
        )
        normalized_obs = (flat_obs - self.obs_mean) / self.obs_std.clamp_min(1e-6)
        robot = self.robot_projection(normalized_obs).unsqueeze(1)
        memory = torch.cat([global_visual, robot, grounded], dim=1)
        return (
            memory.reshape(
                batch_size,
                self.obs_horizon,
                self.tokens_per_observation,
                self.model_dim,
            ),
            points.reshape(
                batch_size,
                self.obs_horizon,
                self.num_keypoints,
                2,
            ),
            reliability.reshape(
                batch_size,
                self.obs_horizon,
                self.num_keypoints,
            ),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> ActionChunkOutput:
        memory_tokens, points, reliability = self._observation_tokens(batch)
        batch_size = memory_tokens.shape[0]
        flat_memory = memory_tokens.reshape(
            batch_size,
            self.obs_horizon * self.tokens_per_observation,
            self.model_dim,
        )
        encoded_memory = self.memory_encoder(flat_memory + self.memory_position)
        queries = (
            self.action_queries.expand(batch_size, -1, -1)
            + self.action_position
        )

        state_latent = None
        next_latent = None
        next_latent_target = None
        if self.policy_arch == "legacy":
            latent = self.action_decoder(queries, encoded_memory)
        else:
            state_sequence = encoded_memory.reshape(
                batch_size,
                self.obs_horizon,
                self.tokens_per_observation,
                self.model_dim,
            ).mean(dim=2)
            state_latent = state_sequence[:, -1]
            action_condition = state_latent[:, None, :].expand(
                -1,
                self.action_horizon,
                -1,
            )
            latent = self.action_decoder(queries, action_condition)
            if "history_action" in batch:
                # The final history action is the current action target. Keep
                # this branch auxiliary-only to avoid conditioning a_t on a_t.
                assert self.next_latent_predictor is not None
                next_latent_sequence = self.next_latent_predictor(
                    state_sequence,
                    batch["history_action"],
                )
                next_latent = next_latent_sequence[:, -1]
            if "next_obs" in batch:
                next_memory_tokens, _, _ = self._observation_tokens(
                    batch,
                    prefix="next_",
                )
                next_flat_memory = next_memory_tokens.reshape(
                    batch_size,
                    self.obs_horizon * self.tokens_per_observation,
                    self.model_dim,
                )
                next_encoded_memory = self.memory_encoder(
                    next_flat_memory + self.memory_position
                )
                next_state_sequence = next_encoded_memory.reshape(
                    batch_size,
                    self.obs_horizon,
                    self.tokens_per_observation,
                    self.model_dim,
                ).mean(dim=2)
                next_latent_target = next_state_sequence[:, -1]

        motion = torch.tanh(self.motion_head(latent))
        gripper_logit = self.gripper_head(latent).squeeze(-1)
        gripper = decode_gripper_action(
            gripper_logit, self.gripper_action_mode
        ).unsqueeze(-1)
        action = torch.cat([motion, gripper], dim=-1)
        return ActionChunkOutput(
            action=action,
            motion=motion,
            gripper_logit=gripper_logit,
            latent=latent,
            grounding_points_px=points,
            reliability=reliability,
            state_latent=state_latent,
            next_latent=next_latent,
            next_latent_target=next_latent_target,
        )

    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        gripper_weight: float = 0.5,
        latent_pred_weight: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], ActionChunkOutput]:
        output = self(batch)
        target = batch["action"]
        mask = batch["action_mask"].to(dtype=target.dtype)
        motion_per_step = F.smooth_l1_loss(
            output.motion,
            target[..., :3],
            reduction="none",
            beta=0.05,
        ).mean(dim=-1)
        gripper_target = encode_gripper_target(
            target[..., 3], self.gripper_action_mode
        )
        gripper_per_step = F.binary_cross_entropy_with_logits(
            output.gripper_logit,
            gripper_target,
            reduction="none",
        )
        denominator = mask.sum().clamp_min(1.0)
        motion_loss = (motion_per_step * mask).sum() / denominator
        gripper_loss = (gripper_per_step * mask).sum() / denominator
        loss = motion_loss + float(gripper_weight) * gripper_loss
        parts = {
            "motion_loss": motion_loss,
            "gripper_loss": gripper_loss,
        }

        if self.policy_arch == "lewm-predictive":
            if output.next_latent is None or output.next_latent_target is None:
                raise ValueError(
                    "predictive policy loss requires history_action and next observation fields"
                )
            next_mask = batch["next_mask"].to(dtype=output.next_latent.dtype)
            next_per_sample = F.mse_loss(
                output.next_latent,
                output.next_latent_target,
                reduction="none",
            ).mean(dim=-1)
            next_latent_loss = (
                next_per_sample * next_mask
            ).sum() / next_mask.sum().clamp_min(1.0)
            loss = loss + float(latent_pred_weight) * next_latent_loss
            parts["next_latent_loss"] = next_latent_loss

        return loss, parts, output

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> ActionChunkOutput:
        self.eval()
        return self(batch)


def load_action_chunk_policy(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    *,
    fcpg_adapter_override: FCPGViTAdapter | LeWMFCPGAdapter | None = None,
    pretrained_vision_override: bool | None = None,
) -> tuple[GroundedActionChunkTransformer, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = dict(checkpoint["args"])
    mode = str(args["mode"])
    policy_arch = str(args.get("policy_arch", "legacy"))
    schema_version = int(checkpoint.get("schema_version", 1))
    if schema_version >= 2:
        if checkpoint.get("checkpoint_type") != "fcpg_action_policy":
            raise ValueError("schema-v2 checkpoint is not an FCPG action policy")
        if checkpoint.get("policy_arch") != policy_arch:
            raise ValueError("schema-v2 checkpoint policy architecture is inconsistent")
    if policy_arch == "lewm-predictive" and schema_version >= 2:
        if checkpoint.get("lewm_upstream_commit") != LEWM_UPSTREAM_COMMIT:
            raise ValueError(
                "predictive policy schema-v2 checkpoint has missing or mismatched "
                "LeWM upstream commit"
            )
    fcpg_adapter = None
    if mode == "fcpg":
        if fcpg_adapter_override is None:
            fcpg_adapter, _ = load_fcpg_vit_adapter(
                str(args["fcpg_checkpoint"]), device
            )
        else:
            fcpg_adapter = fcpg_adapter_override
    elif fcpg_adapter_override is not None:
        raise ValueError("fcpg_adapter_override requires an fcpg policy checkpoint")
    reuse_fcpg_vision = bool(args.get("reuse_fcpg_vision", False))
    pretrained_vision = (
        not reuse_fcpg_vision
        if pretrained_vision_override is None
        else bool(pretrained_vision_override)
    )
    model = GroundedActionChunkTransformer(
        mode=mode,
        obs_dim=int(checkpoint["obs_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        num_keypoints=int(checkpoint["num_keypoints"]),
        obs_horizon=int(args.get("obs_horizon", 2)),
        action_horizon=int(args.get("action_horizon", 32)),
        model_dim=int(args.get("model_dim", 256)),
        vision_dim=int(args.get("vision_dim", 256)),
        encoder_layers=int(args.get("encoder_layers", 2)),
        decoder_layers=int(args.get("decoder_layers", 6)),
        heads=int(args.get("heads", 8)),
        mlp_ratio=int(args.get("mlp_ratio", 4)),
        dropout=float(args.get("dropout", 0.1)),
        vision_model_name=str(args.get("vision_model_name", "vit_base_patch16_224.augreg_in1k")),
        vision_selected_layers=tuple(int(layer) for layer in args.get("vision_selected_layers", (2, 5, 8, 11))),
        # A reuse checkpoint immediately replaces the whole vision encoder
        # from the strictly loaded FCPG adapter.  Avoid an unnecessary network
        # fetch when reconstructing that frozen model offline.
        pretrained_vision=pretrained_vision,
        trainable_backbone_blocks=int(args.get("trainable_backbone_blocks", 0)),
        fcpg_adapter=fcpg_adapter,
        policy_arch=policy_arch,
        predictor_layers=int(args.get("predictor_layers", 2)),
        reuse_fcpg_vision=reuse_fcpg_vision,
        # Historical checkpoints omitted this field and emitted sigmoid values
        # directly. Preserve that behavior while new checkpoints use the
        # MetaWorld-native signed gripper action space.
        gripper_action_mode=str(
            args.get("gripper_action_mode", "legacy_unsigned")
        ),
    ).to(device)
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    allowed_missing = ("vision_encoder.backbone.", "fcpg_adapter.")
    invalid_missing = [key for key in incompatible.missing_keys if not key.startswith(allowed_missing)]
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"invalid action-chunk state: missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
        )
    model.set_normalization(checkpoint["normalization"])
    model.set_reliability_override(str(args.get("reliability_override", "native")))
    model.eval()
    return model, args
