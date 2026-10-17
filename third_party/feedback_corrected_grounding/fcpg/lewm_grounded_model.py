"""Full LeWM-style world model adapted to geometry-grounded observations.

This module deliberately keeps the upstream LeWM source tree untouched.  It
reuses the upstream action embedder, autoregressive predictor, projection heads,
and SIGReg implementation while replacing the pixel-only CLS encoder with a
frame encoder that preserves FCPG's global, proprioceptive, and local geometric
anchor tokens.

Unlike the earlier ``lewm-predictive`` policy ablation, the predicted latent is
an actual model output: it can be rolled out recursively and scored against a
goal embedding for planning.  No behavior-cloning loss is mixed into the
faithful LeWM objective implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from fcpg.modules import sample_feature_map_at_pixels
from third_party.lewm.module import ARPredictor, Embedder, MLP, SIGReg


LEWM_UPSTREAM_COMMIT = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"


def _sequence_tensor(
    batch: Mapping[str, torch.Tensor],
    *names: str,
) -> torch.Tensor:
    for name in names:
        value = batch.get(name)
        if value is not None:
            return value
    joined = ", ".join(names)
    raise KeyError(f"batch must contain one of: {joined}")


def _vision_values(output: object) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(output, Mapping):
        feature_map = output.get("feature_map")
        global_feature = output.get("global_feature")
    else:
        feature_map = getattr(output, "feature_map", None)
        global_feature = getattr(output, "global_feature", None)
    if not torch.is_tensor(feature_map) or not torch.is_tensor(global_feature):
        raise TypeError(
            "vision encoder output must provide tensor feature_map and global_feature"
        )
    return feature_map, global_feature

@dataclass(frozen=True)
class LeWMVisionOutput:
    feature_map: torch.Tensor
    global_feature: torch.Tensor


class LeWMViTDenseEncoder(nn.Module):
    """Expose the exact upstream LeWM ViT as a dense visual encoder.

    Upstream LeWM consumes the CLS token. FCPG also samples patch tokens at
    kinematic anchors, so the same ViT patch sequence is reshaped into a dense
    feature map without introducing a second visual backbone.
    """

    _SIZE_TO_DIM = {
        "tiny": 192,
        "small": 384,
        "base": 768,
        "large": 1024,
        "huge": 1280,
    }

    def __init__(
        self,
        size: str = "tiny",
        patch_size: int = 14,
        image_size: int = 224,
        pretrained: bool = False,
        use_mask_token: bool = False,
    ) -> None:
        super().__init__()
        if size not in self._SIZE_TO_DIM:
            raise ValueError(f"unknown ViT size: {size}")
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        from stable_pretraining.backbone.utils import vit_hf

        self.size = str(size)
        self.patch_size = int(patch_size)
        self.image_size = int(image_size)
        self.output_dim = int(self._SIZE_TO_DIM[self.size])
        self.backbone = vit_hf(
            size=self.size,
            patch_size=self.patch_size,
            image_size=self.image_size,
            pretrained=bool(pretrained),
            use_mask_token=bool(use_mask_token),
        )
        self.register_buffer(
            "image_mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, pixels: torch.Tensor) -> LeWMVisionOutput:
        if not pixels.ndim == 4 or not pixels.shape[1] == 3:
            raise ValueError(
                f"pixels must have shape [B, 3, H, W], got {tuple(pixels.shape)}"
            )
        if not pixels.shape[-2:] == (self.image_size, self.image_size):
            pixels = F.interpolate(
                pixels,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        normalized = (pixels - self.image_mean.to(dtype=pixels.dtype)) / self.image_std.to(
            dtype=pixels.dtype
        )
        output = self.backbone(normalized, interpolate_pos_encoding=True)
        tokens = output.last_hidden_state
        patch_tokens = tokens[:, 1:]
        grid_size = self.image_size // self.patch_size
        expected = grid_size * grid_size
        if not patch_tokens.shape[1] == expected:
            raise RuntimeError(
                f"expected {expected} ViT patch tokens, got {patch_tokens.shape[1]}"
            )
        feature_map = patch_tokens.transpose(1, 2).reshape(
            pixels.shape[0],
            self.output_dim,
            grid_size,
            grid_size,
        )
        return LeWMVisionOutput(
            feature_map=feature_map,
            global_feature=tokens[:, 0],
        )


class FCPGViTDenseEncoder(nn.Module):
    """Use the frozen FCPG-trained dense map as LeWM visual input.

    FCPG's visual map is 256-dimensional while LeWM anchor tokens are 192D.
    The small projections are the only newly initialized visual interface; the
    ViT-B backbone and the trained FCPG projection/fusion heads remain frozen.
    """

    def __init__(
        self,
        fcpg_checkpoint: str,
        output_dim: int = 192,
        **_legacy_vision_config: object,
    ) -> None:
        super().__init__()
        from fcpg.transformer_grounding import load_fcpg_vit_adapter

        corrector, _ = load_fcpg_vit_adapter(fcpg_checkpoint, device="cpu")
        vision = getattr(corrector, "vision", None)
        if vision is None:
            raise TypeError("FCPG checkpoint does not expose a dense vision encoder")
        self.fcpg_vision = vision
        self.fcpg_vision.requires_grad_(False).eval()
        source_dim = int(getattr(vision, "output_dim"))
        self.source_dim = source_dim
        self.output_dim = int(output_dim)
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        groups = min(32, self.output_dim)
        while self.output_dim % groups != 0:
            groups -= 1
        self.feature_projection = nn.Sequential(
            nn.GroupNorm(groups, self.output_dim),
        )
        self.feature_linear = nn.Conv2d(source_dim, self.output_dim, kernel_size=1)
        self.global_projection = nn.Sequential(
            nn.LayerNorm(source_dim),
            nn.Linear(source_dim, self.output_dim),
        )

    def train(self, mode: bool = True) -> "FCPGViTDenseEncoder":
        super().train(mode)
        self.fcpg_vision.eval()
        return self

    def forward(self, pixels: torch.Tensor) -> LeWMVisionOutput:
        vision = self.fcpg_vision(pixels)
        feature_map = self.feature_projection(self.feature_linear(vision.feature_map))
        global_feature = self.global_projection(vision.global_feature)
        return LeWMVisionOutput(
            feature_map=feature_map,
            global_feature=global_feature,
        )


class GroundedFrameEncoder(nn.Module):
    """Aggregate raw RGB, robot state, and FCPG anchor-local evidence per frame.

    ``vision_encoder`` must return a dense ``feature_map`` and a
    ``global_feature``.  The existing :class:`fcpg.transformer_grounding.ViTDenseEncoder`
    satisfies that interface.  Tests and small ablations can inject a compact
    encoder with the same two outputs.

    A learned summary token attends to all observation tokens.  This avoids the
    heterogeneous-token mean pooling used by the earlier policy auxiliary and
    retains separate tokens for global appearance, robot proprioception, and
    each metric kinematic anchor.

    ``anchor_visual_mode`` controls the visual evidence presented for each
    anchor. ``single_corrected`` preserves the original checkpoint-compatible
    input, while the four triplet modes use parameter-matched projections:
    ``triplet_nominal=[n,n,n]``, ``triplet_corrected=[c,c,c]``,
    ``paired_visual=[n,c,c]``, and ``paired_innovation=[n,c,c-n]``.
    """

    _ANCHOR_VISUAL_MODES = frozenset(
        {
            "single_corrected",
            "triplet_nominal",
            "triplet_corrected",
            "paired_visual",
            "paired_innovation",
        }
    )
    _PAIRED_ANCHOR_VISUAL_MODES = frozenset(
        {"triplet_nominal", "paired_visual", "paired_innovation"}
    )

    def __init__(
        self,
        vision_encoder: nn.Module,
        visual_dim: int,
        obs_dim: int,
        num_keypoints: int,
        latent_dim: int = 192,
        depth: int = 2,
        heads: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        anchor_visual_mode: str = "single_corrected",
    ) -> None:
        super().__init__()
        if latent_dim % heads != 0:
            raise ValueError(f"latent_dim={latent_dim} must be divisible by heads={heads}")
        if num_keypoints <= 0:
            raise ValueError("num_keypoints must be positive")
        self.vision_encoder = vision_encoder
        self.visual_dim = int(visual_dim)
        self.obs_dim = int(obs_dim)
        self.num_keypoints = int(num_keypoints)
        self.latent_dim = int(latent_dim)
        self.anchor_visual_mode = str(anchor_visual_mode)
        if self.anchor_visual_mode not in self._ANCHOR_VISUAL_MODES:
            choices = ", ".join(sorted(self._ANCHOR_VISUAL_MODES))
            raise ValueError(
                f"anchor_visual_mode must be one of: {choices}; "
                f"got {self.anchor_visual_mode!r}"
            )

        self.summary_token = nn.Parameter(torch.zeros(1, 1, self.latent_dim))
        self.token_type = nn.Parameter(
            torch.zeros(1, 3 + self.num_keypoints, self.latent_dim)
        )
        self.global_projection = nn.Sequential(
            nn.LayerNorm(self.visual_dim),
            nn.Linear(self.visual_dim, self.latent_dim),
        )
        self.robot_projection = nn.Sequential(
            nn.LayerNorm(self.obs_dim),
            nn.Linear(self.obs_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        anchor_visual_dim = self.visual_dim * (
            1 if self.anchor_visual_mode == "single_corrected" else 3
        )
        self.anchor_visual_projection = nn.Sequential(
            nn.LayerNorm(anchor_visual_dim),
            nn.Linear(anchor_visual_dim, self.latent_dim),
        )
        geometry_dim = 2 + 2 + 3 + self.num_keypoints + 1
        self.anchor_geometry_projection = nn.Sequential(
            nn.LayerNorm(geometry_dim),
            nn.Linear(geometry_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=int(heads),
            dim_feedforward=self.latent_dim * int(mlp_ratio),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.aggregator = nn.TransformerEncoder(
            layer,
            num_layers=int(depth),
            norm=nn.LayerNorm(self.latent_dim),
        )
        nn.init.trunc_normal_(self.summary_token, std=0.02)
        nn.init.trunc_normal_(self.token_type, std=0.02)

    def _geometry_tokens(
        self,
        q_nominal_px: torch.Tensor,
        q_grounded_px: torch.Tensor,
        keypoints_xyz: torch.Tensor,
        valid: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        batch, num_keypoints, coordinate_dim = q_nominal_px.shape
        expected = (batch, self.num_keypoints, 2)
        if (batch, num_keypoints, coordinate_dim) != expected:
            raise ValueError(
                f"q_nominal_px must have shape {expected}, got {tuple(q_nominal_px.shape)}"
            )
        if q_grounded_px.shape != q_nominal_px.shape:
            raise ValueError("q_grounded_px must match q_nominal_px")
        if keypoints_xyz.shape != (batch, self.num_keypoints, 3):
            raise ValueError(
                "keypoints_xyz must have shape "
                f"{(batch, self.num_keypoints, 3)}, got {tuple(keypoints_xyz.shape)}"
            )
        if valid.shape != (batch, self.num_keypoints):
            raise ValueError(
                f"valid must have shape {(batch, self.num_keypoints)}, got {tuple(valid.shape)}"
            )
        width, height = image_size
        scale = q_nominal_px.new_tensor([max(width - 1, 1), max(height - 1, 1)])
        nominal = q_nominal_px / scale
        residual = (q_grounded_px - q_nominal_px) / scale
        identity = torch.eye(
            self.num_keypoints,
            device=q_nominal_px.device,
            dtype=q_nominal_px.dtype,
        ).unsqueeze(0).expand(batch, -1, -1)
        geometry = torch.cat(
            [
                nominal,
                residual,
                keypoints_xyz,
                identity,
                valid.to(dtype=q_nominal_px.dtype).unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.anchor_geometry_projection(geometry)

    def _anchor_visual_input(
        self,
        sampled_corrected_visual: torch.Tensor,
        sampled_nominal_visual: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.anchor_visual_mode == "single_corrected":
            return sampled_corrected_visual
        if self.anchor_visual_mode == "triplet_corrected":
            return torch.cat([sampled_corrected_visual] * 3, dim=-1)
        if sampled_nominal_visual is None:
            raise ValueError(
                f"anchor_visual_mode={self.anchor_visual_mode!r} requires "
                "sampled_nominal_visual"
            )
        if sampled_nominal_visual.shape != sampled_corrected_visual.shape:
            raise ValueError(
                "sampled_nominal_visual must match sampled_visual: "
                f"expected {tuple(sampled_corrected_visual.shape)}, "
                f"got {tuple(sampled_nominal_visual.shape)}"
            )
        if self.anchor_visual_mode == "triplet_nominal":
            parts = (sampled_nominal_visual,) * 3
        elif self.anchor_visual_mode == "paired_visual":
            parts = (
                sampled_nominal_visual,
                sampled_corrected_visual,
                sampled_corrected_visual,
            )
        else:
            parts = (
                sampled_nominal_visual,
                sampled_corrected_visual,
                sampled_corrected_visual - sampled_nominal_visual,
            )
        return torch.cat(parts, dim=-1)

    def forward_token_sequence_from_features(
        self,
        sampled_visual: torch.Tensor,
        global_visual: torch.Tensor,
        robot_obs: torch.Tensor,
        q_nominal_px: torch.Tensor,
        q_grounded_px: torch.Tensor,
        keypoints_xyz: torch.Tensor,
        valid: torch.Tensor | None,
        image_size: tuple[int, int],
        sampled_nominal_visual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return aggregated ``[summary, global, robot, anchors...]`` tokens."""

        batch = robot_obs.shape[0]
        if robot_obs.shape != (batch, self.obs_dim):
            raise ValueError(
                f"robot_obs must have shape {(batch, self.obs_dim)}, got {tuple(robot_obs.shape)}"
            )
        if sampled_visual.shape != (batch, self.num_keypoints, self.visual_dim):
            raise ValueError(
                "sampled_visual must have shape "
                f"{(batch, self.num_keypoints, self.visual_dim)}, "
                f"got {tuple(sampled_visual.shape)}"
            )
        if global_visual.shape != (batch, self.visual_dim):
            raise ValueError(
                f"global_visual must have shape {(batch, self.visual_dim)}, "
                f"got {tuple(global_visual.shape)}"
            )
        if valid is None:
            valid = torch.ones(
                batch,
                self.num_keypoints,
                device=q_nominal_px.device,
                dtype=torch.bool,
            )
        geometry = self._geometry_tokens(
            q_nominal_px,
            q_grounded_px,
            keypoints_xyz,
            valid,
            image_size,
        )
        anchor_visual = self._anchor_visual_input(
            sampled_visual,
            sampled_nominal_visual,
        )
        anchor_tokens = self.anchor_visual_projection(anchor_visual) + geometry
        tokens = torch.cat(
            [
                self.summary_token.expand(batch, -1, -1),
                self.global_projection(global_visual).unsqueeze(1),
                self.robot_projection(robot_obs).unsqueeze(1),
                anchor_tokens,
            ],
            dim=1,
        )
        tokens = tokens + self.token_type
        return self.aggregator(tokens)

    def forward_from_features(
        self,
        sampled_visual: torch.Tensor,
        global_visual: torch.Tensor,
        robot_obs: torch.Tensor,
        q_nominal_px: torch.Tensor,
        q_grounded_px: torch.Tensor,
        keypoints_xyz: torch.Tensor,
        valid: torch.Tensor | None,
        image_size: tuple[int, int],
        sampled_nominal_visual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the legacy summary latent for precomputed visual features."""

        return self.forward_token_sequence_from_features(
            sampled_visual=sampled_visual,
            sampled_nominal_visual=sampled_nominal_visual,
            global_visual=global_visual,
            robot_obs=robot_obs,
            q_nominal_px=q_nominal_px,
            q_grounded_px=q_grounded_px,
            keypoints_xyz=keypoints_xyz,
            valid=valid,
            image_size=image_size,
        )[:, 0]

    def forward_token_sequence(
        self,
        pixels: torch.Tensor,
        robot_obs: torch.Tensor,
        q_nominal_px: torch.Tensor,
        q_grounded_px: torch.Tensor,
        keypoints_xyz: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError(f"pixels must have shape [B, 3, H, W], got {tuple(pixels.shape)}")
        vision = self.vision_encoder(pixels)
        feature_map, global_visual = _vision_values(vision)
        sampled_visual = sample_feature_map_at_pixels(
            feature_map,
            q_grounded_px,
            image_size=(int(pixels.shape[-1]), int(pixels.shape[-2])),
            padding_mode="border",
        )
        sampled_nominal_visual = None
        if self.anchor_visual_mode in self._PAIRED_ANCHOR_VISUAL_MODES:
            sampled_nominal_visual = sample_feature_map_at_pixels(
                feature_map,
                q_nominal_px,
                image_size=(int(pixels.shape[-1]), int(pixels.shape[-2])),
                padding_mode="border",
            )
        return self.forward_token_sequence_from_features(
            sampled_visual=sampled_visual,
            sampled_nominal_visual=sampled_nominal_visual,
            global_visual=global_visual,
            robot_obs=robot_obs,
            q_nominal_px=q_nominal_px,
            q_grounded_px=q_grounded_px,
            keypoints_xyz=keypoints_xyz,
            valid=valid,
            image_size=(int(pixels.shape[-1]), int(pixels.shape[-2])),
        )

    def forward(
        self,
        pixels: torch.Tensor,
        robot_obs: torch.Tensor,
        q_nominal_px: torch.Tensor,
        q_grounded_px: torch.Tensor,
        keypoints_xyz: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the legacy summary latent for raw pixels."""

        return self.forward_token_sequence(
            pixels=pixels,
            robot_obs=robot_obs,
            q_nominal_px=q_nominal_px,
            q_grounded_px=q_grounded_px,
            keypoints_xyz=keypoints_xyz,
            valid=valid,
        )[:, 0]


@dataclass(frozen=True)
class GroundedLeWMOutput:
    frame_latent: torch.Tensor
    embedding: torch.Tensor
    state_target_embedding: torch.Tensor
    predicted_embedding: torch.Tensor
    target_embedding: torch.Tensor
    prediction_loss: torch.Tensor
    copy_loss: torch.Tensor
    alignment_loss: torch.Tensor
    sigreg_loss: torch.Tensor
    loss: torch.Tensor


class GroundedLeWM(nn.Module):
    """LeWM objective and recursive latent dynamics over grounded frames."""

    def __init__(
        self,
        frame_encoder: GroundedFrameEncoder,
        action_dim: int,
        action_smoothing_dim: int = 10,
        history_size: int = 3,
        predictor_depth: int = 6,
        predictor_heads: int = 16,
        predictor_mlp_dim: int = 2048,
        predictor_dim_head: int = 64,
        predictor_dropout: float = 0.1,
        projector_hidden_dim: int = 2048,
        sigreg_knots: int = 17,
        sigreg_num_proj: int = 1024,
        target_mode: str = "online",
        alignment_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        if action_smoothing_dim <= 0:
            raise ValueError("action_smoothing_dim must be positive")
        self.frame_encoder = frame_encoder
        self.latent_dim = int(frame_encoder.latent_dim)
        self.action_dim = int(action_dim)
        self.history_size = int(history_size)
        self.target_mode = str(target_mode)
        if self.target_mode not in {"online", "frozen_oracle"}:
            raise ValueError("target_mode must be 'online' or 'frozen_oracle'")
        self.alignment_weight = float(alignment_weight)
        if self.alignment_weight < 0.0:
            raise ValueError("alignment_weight must be non-negative")
        if self.target_mode == "frozen_oracle" and self.alignment_weight <= 0.0:
            raise ValueError("frozen_oracle target_mode requires alignment_weight > 0")
        self.predictor = ARPredictor(
            num_frames=self.history_size,
            depth=int(predictor_depth),
            heads=int(predictor_heads),
            mlp_dim=int(predictor_mlp_dim),
            input_dim=self.latent_dim,
            hidden_dim=self.latent_dim,
            output_dim=self.latent_dim,
            dim_head=int(predictor_dim_head),
            dropout=float(predictor_dropout),
            emb_dropout=0.0,
        )
        self.action_encoder = Embedder(
            input_dim=self.action_dim,
            smoothed_dim=int(action_smoothing_dim),
            emb_dim=self.latent_dim,
        )
        self.projector = MLP(
            input_dim=self.latent_dim,
            hidden_dim=int(projector_hidden_dim),
            output_dim=self.latent_dim,
            norm_fn=nn.BatchNorm1d,
        )
        self.pred_proj = MLP(
            input_dim=self.latent_dim,
            hidden_dim=int(projector_hidden_dim),
            output_dim=self.latent_dim,
            norm_fn=nn.BatchNorm1d,
        )
        self.sigreg = SIGReg(knots=int(sigreg_knots), num_proj=int(sigreg_num_proj))
        self.register_buffer(
            "proprio_mean", torch.zeros(self.frame_encoder.obs_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "proprio_std", torch.ones(self.frame_encoder.obs_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "action_mean", torch.zeros(self.action_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "action_std", torch.ones(self.action_dim, dtype=torch.float32)
        )

    def set_input_normalization(
        self,
        *,
        proprio_mean: torch.Tensor | list[float],
        proprio_std: torch.Tensor | list[float],
        action_mean: torch.Tensor | list[float],
        action_std: torch.Tensor | list[float],
    ) -> None:
        """Install train-split statistics used by encoding and planning."""

        values = {
            "proprio_mean": (proprio_mean, self.frame_encoder.obs_dim),
            "proprio_std": (proprio_std, self.frame_encoder.obs_dim),
            "action_mean": (action_mean, self.action_dim),
            "action_std": (action_std, self.action_dim),
        }
        with torch.no_grad():
            for name, (value, size) in values.items():
                target = getattr(self, name)
                tensor = torch.as_tensor(value, dtype=target.dtype, device=target.device)
                if tensor.shape != (size,):
                    raise ValueError(
                        f"{name} must have shape {(size,)}, got {tuple(tensor.shape)}"
                    )
                if name.endswith("_std") and bool(torch.any(tensor <= 0.0)):
                    raise ValueError(f"{name} must be strictly positive")
                target.copy_(tensor)

    def _flatten_sequence(
        self,
        value: torch.Tensor,
        batch_size: int,
        sequence_length: int,
        trailing_shape: tuple[int, ...],
        name: str,
    ) -> torch.Tensor:
        expected = (batch_size, sequence_length, *trailing_shape)
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")
        return value.reshape(batch_size * sequence_length, *trailing_shape)

    def encode_frames(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = _sequence_tensor(batch, "pixels", "rgb")
        if pixels.ndim != 5:
            raise ValueError(
                f"pixels/rgb must have shape [B, T, 3, H, W], got {tuple(pixels.shape)}"
            )
        if pixels.shape[2] != 3:
            raise ValueError("pixels/rgb channel dimension must be 3")
        batch_size, sequence_length, _, height, width = pixels.shape
        obs = _sequence_tensor(batch, "proprio", "obs")
        q_nominal = _sequence_tensor(batch, "q_nominal")
        q_grounded = batch.get("q_corrected")
        if q_grounded is None:
            q_grounded = batch.get("q_grounded")
        if q_grounded is None:
            q_grounded = q_nominal
        keypoints_xyz = _sequence_tensor(batch, "keypoints_xyz")
        valid = batch.get("valid")
        if valid is None:
            valid = batch.get("valid_nominal")

        flat_pixels = self._flatten_sequence(
            pixels,
            batch_size,
            sequence_length,
            (3, height, width),
            "pixels",
        ).float()
        if not flat_pixels.is_floating_point():
            flat_pixels = flat_pixels.float()
        if flat_pixels.numel() and flat_pixels.detach().amax() > 1.5:
            flat_pixels = flat_pixels / 255.0
        flat_obs = self._flatten_sequence(
            obs,
            batch_size,
            sequence_length,
            (self.frame_encoder.obs_dim,),
            "proprio/obs",
        ).float()
        flat_obs = (flat_obs - self.proprio_mean) / self.proprio_std
        flat_q_nominal = self._flatten_sequence(
            q_nominal,
            batch_size,
            sequence_length,
            (self.frame_encoder.num_keypoints, 2),
            "q_nominal",
        ).float()
        flat_q_grounded = self._flatten_sequence(
            q_grounded,
            batch_size,
            sequence_length,
            (self.frame_encoder.num_keypoints, 2),
            "q_corrected/q_grounded",
        ).float()
        flat_xyz = self._flatten_sequence(
            keypoints_xyz,
            batch_size,
            sequence_length,
            (self.frame_encoder.num_keypoints, 3),
            "keypoints_xyz",
        ).float()
        flat_valid = None
        if valid is not None:
            flat_valid = self._flatten_sequence(
                valid,
                batch_size,
                sequence_length,
                (self.frame_encoder.num_keypoints,),
                "valid",
            ).bool()
        frame_latent = self.frame_encoder(
            flat_pixels,
            flat_obs,
            flat_q_nominal,
            flat_q_grounded,
            flat_xyz,
            flat_valid,
        )
        embedding = self.projector(frame_latent)
        return (
            frame_latent.reshape(batch_size, sequence_length, self.latent_dim),
            embedding.reshape(batch_size, sequence_length, self.latent_dim),
        )

    def predict(self, embedding: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if embedding.ndim != 3 or embedding.shape[-1] != self.latent_dim:
            raise ValueError(
                f"embedding must have shape [B, T, {self.latent_dim}], "
                f"got {tuple(embedding.shape)}"
            )
        if action.shape != (*embedding.shape[:2], self.action_dim):
            raise ValueError(
                f"action must have shape {(*embedding.shape[:2], self.action_dim)}, "
                f"got {tuple(action.shape)}"
            )
        normalized_action = (action - self.action_mean) / self.action_std
        predicted = self.predictor(embedding, self.action_encoder(normalized_action))
        batch_size, sequence_length, _ = predicted.shape
        predicted = self.pred_proj(predicted.reshape(-1, self.latent_dim))
        return predicted.reshape(batch_size, sequence_length, self.latent_dim)

    def target_sequence(
        self,
        batch: Mapping[str, torch.Tensor],
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Return the common state target used for prediction and copy metrics."""

        if self.target_mode == "online":
            return embedding
        target = batch.get("oracle_target")
        if not torch.is_tensor(target):
            raise KeyError("frozen_oracle target_mode requires batch['oracle_target']")
        target = target.float()
        if target.shape != embedding.shape:
            raise ValueError(
                "oracle_target must match embedding shape: "
                f"expected {tuple(embedding.shape)}, got {tuple(target.shape)}"
            )
        return target.detach()

    def _sigreg_loss(
        self,
        embedding: torch.Tensor,
        source_index: torch.Tensor | None,
    ) -> torch.Tensor:
        if source_index is None:
            return self.sigreg(embedding.transpose(0, 1))
        if source_index.ndim == 2:
            if not bool(torch.all(source_index == source_index[:, :1])):
                raise ValueError("source_index must be constant within each sequence")
            source_index = source_index[:, 0]
        if source_index.ndim != 1 or source_index.shape[0] != embedding.shape[0]:
            raise ValueError("source_index must have shape [B] or [B, T]")
        losses = []
        for source in torch.unique(source_index):
            selected = embedding[source_index == source]
            if selected.shape[0] >= 2:
                losses.append(self.sigreg(selected.transpose(0, 1)))
        if not losses:
            return embedding.sum() * 0.0
        return torch.stack(losses).mean()

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        sigreg_weight: float = 0.09,
        per_source_sigreg: bool = False,
    ) -> GroundedLeWMOutput:
        """Compute the faithful upstream ``H + 1`` teacher-forced objective.

        With ``history_size=3``, actions ``a_0..a_2`` predict embeddings
        ``z_1..z_3`` from contexts ``z_0..z_2``.  This matches upstream LeWM;
        recursive multi-step prediction is exposed separately through
        :meth:`rollout_embeddings`.
        """

        frame_latent, embedding = self.encode_frames(batch)
        required = self.history_size + 1
        if embedding.shape[1] != required:
            raise ValueError(
                f"faithful LeWM training expects exactly H+1={required} frames, "
                f"got {embedding.shape[1]}"
            )
        action = _sequence_tensor(batch, "action", "actions").float()
        if action.ndim != 3 or action.shape[0] != embedding.shape[0]:
            raise ValueError("action must have shape [B, T, A]")
        if action.shape[1] < self.history_size or action.shape[2] != self.action_dim:
            raise ValueError(
                f"action must provide at least {self.history_size} steps with "
                f"action_dim={self.action_dim}, got {tuple(action.shape)}"
            )
        state_target = self.target_sequence(batch, embedding)
        context = embedding[:, : self.history_size]
        target = state_target[:, 1 : self.history_size + 1]
        predicted = self.predict(context, action[:, : self.history_size])
        prediction_loss = F.mse_loss(predicted, target)
        copy_loss = F.mse_loss(state_target[:, : self.history_size], target)
        alignment_loss = (
            F.mse_loss(embedding, state_target)
            if self.target_mode == "frozen_oracle"
            else embedding.sum() * 0.0
        )
        source_index = batch.get("source_index") if per_source_sigreg else None
        sigreg_loss = self._sigreg_loss(embedding, source_index)
        loss = (
            prediction_loss
            + self.alignment_weight * alignment_loss
            + float(sigreg_weight) * sigreg_loss
        )
        return GroundedLeWMOutput(
            frame_latent=frame_latent,
            embedding=embedding,
            state_target_embedding=state_target,
            predicted_embedding=predicted,
            target_embedding=target,
            prediction_loss=prediction_loss,
            copy_loss=copy_loss,
            alignment_loss=alignment_loss,
            sigreg_loss=sigreg_loss,
            loss=loss,
        )

    def rollout_embeddings(
        self,
        history_embedding: torch.Tensor,
        history_action: torch.Tensor,
        future_action: torch.Tensor,
    ) -> torch.Tensor:
        """Recursively predict future projected embeddings.

        Args:
            history_embedding: ``[B, H, D]`` observed projected embeddings.
            history_action: ``[B, H-1, A]`` transitions between history frames.
            future_action: ``[B, S, P, A]`` candidate future transitions.

        Returns:
            Predicted future embeddings with shape ``[B, S, P, D]``.
        """

        if history_embedding.ndim != 3:
            raise ValueError("history_embedding must have shape [B, H, D]")
        batch_size, history, latent_dim = history_embedding.shape
        if history != self.history_size or latent_dim != self.latent_dim:
            raise ValueError(
                f"history_embedding must have shape [B, {self.history_size}, "
                f"{self.latent_dim}], got {tuple(history_embedding.shape)}"
            )
        if history_action.shape != (
            batch_size,
            self.history_size - 1,
            self.action_dim,
        ):
            raise ValueError(
                "history_action must have shape "
                f"{(batch_size, self.history_size - 1, self.action_dim)}, "
                f"got {tuple(history_action.shape)}"
            )
        if future_action.ndim != 4 or future_action.shape[0] != batch_size:
            raise ValueError("future_action must have shape [B, S, P, A]")
        if future_action.shape[-1] != self.action_dim:
            raise ValueError(
                f"future_action last dimension must be {self.action_dim}, "
                f"got {future_action.shape[-1]}"
            )
        samples = future_action.shape[1]
        predictions = future_action.shape[2]
        embeddings = history_embedding[:, None].expand(-1, samples, -1, -1)
        actions = history_action[:, None].expand(-1, samples, -1, -1)
        outputs = []
        for step in range(predictions):
            current_action = future_action[:, :, step : step + 1]
            action_context = torch.cat([actions, current_action], dim=2)
            flat_embedding = embeddings.reshape(
                batch_size * samples,
                self.history_size,
                self.latent_dim,
            )
            flat_action = action_context.reshape(
                batch_size * samples,
                self.history_size,
                self.action_dim,
            )
            next_embedding = self.predict(flat_embedding, flat_action)[:, -1]
            next_embedding = next_embedding.reshape(
                batch_size,
                samples,
                1,
                self.latent_dim,
            )
            outputs.append(next_embedding)
            embeddings = torch.cat([embeddings[:, :, 1:], next_embedding], dim=2)
            actions = action_context[:, :, 1:]
        if not outputs:
            return history_embedding.new_empty(batch_size, samples, 0, self.latent_dim)
        return torch.cat(outputs, dim=2)

    @staticmethod
    def goal_cost(
        predicted_embedding: torch.Tensor,
        goal_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Return final-step squared latent distance for each action candidate."""

        if predicted_embedding.ndim != 4:
            raise ValueError("predicted_embedding must have shape [B, S, P, D]")
        if goal_embedding.ndim == 2:
            goal_embedding = goal_embedding[:, None, None, :]
        elif goal_embedding.ndim == 3:
            goal_embedding = goal_embedding[:, None, -1:, :]
        else:
            raise ValueError("goal_embedding must have shape [B, D] or [B, T, D]")
        if goal_embedding.shape[0] != predicted_embedding.shape[0]:
            raise ValueError("goal and prediction batch sizes differ")
        return (predicted_embedding[:, :, -1:] - goal_embedding).square().sum(dim=(-1, -2))
