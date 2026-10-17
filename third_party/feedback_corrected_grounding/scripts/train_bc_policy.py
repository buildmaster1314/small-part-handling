#!/usr/bin/env python3
"""Train a minimal BC policy over FCPG projection datasets.

This is not the final Diffusion Policy experiment. It is a Stage-4 integration
smoke test that verifies scripted-action datasets, GeoProp-style token
sampling, and FCPG residual-only corrected tokens can train end-to-end.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.modules import GroundedTokenAdapter, ResidualCorrectionHead, sample_image_patches_at_pixels  # noqa: E402
from fcpg.observations import OBSERVATION_MODES, select_policy_observation  # noqa: E402


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    test: np.ndarray


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def make_random_split(n: int, val_fraction: float, seed: int) -> SplitIndices:
    rng = np.random.default_rng(seed)
    indices = np.arange(n, dtype=np.int64)
    rng.shuffle(indices)
    test_size = max(1, int(round(n * val_fraction)))
    return SplitIndices(train=indices[test_size:], test=indices[:test_size])


def make_split(
    raw: np.lib.npyio.NpzFile,
    val_fraction: float,
    seed: int,
    split_unit: str = "sample",
    tasks: str = "",
) -> SplitIndices:
    """Split selected policy frames independently or by complete task episodes."""

    length_key = next(
        (
            key
            for key in ("episode_id", "action", "obs", "q_true", "rgb")
            if key in raw.files
        ),
        None,
    )
    if length_key is None:
        raise KeyError("dataset has no field from which to infer sample count")
    n = int(raw[length_key].shape[0])
    indices = np.arange(n, dtype=np.int64)
    selected_tasks = [item.strip() for item in tasks.split(",") if item.strip()]
    if selected_tasks:
        if "task_name" not in raw.files:
            raise KeyError("task filtering requires dataset field: task_name")
        all_task_names = raw["task_name"].astype(str)
        known_tasks = set(np.unique(all_task_names).tolist())
        unknown = sorted(set(selected_tasks).difference(known_tasks))
        if unknown:
            raise ValueError(f"unknown tasks in dataset: {unknown}")
        indices = indices[np.isin(all_task_names, np.asarray(selected_tasks))]
    if indices.size < 2:
        raise RuntimeError("policy split requires at least two selected frames")

    rng = np.random.default_rng(seed)
    if split_unit == "sample":
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        test_size = max(1, int(round(len(shuffled) * val_fraction)))
        if test_size >= len(shuffled):
            test_size = len(shuffled) - 1
        return SplitIndices(train=shuffled[test_size:], test=shuffled[:test_size])
    if split_unit != "task-episode":
        raise ValueError(f"unknown split_unit={split_unit!r}")

    required = {"task_name", "episode_id"}
    missing = required.difference(raw.files)
    if missing:
        raise KeyError(f"task-episode split requires dataset fields: {sorted(missing)}")

    task_names = raw["task_name"].astype(str)[indices]
    episode_ids = raw["episode_id"].astype(str)[indices]
    group_keys = np.char.add(np.char.add(task_names, ":"), episode_ids)
    selected_test_groups: list[str] = []
    for task_name in np.unique(task_names):
        task_groups = np.unique(group_keys[task_names == task_name])
        rng.shuffle(task_groups)
        test_group_count = max(1, int(round(len(task_groups) * val_fraction)))
        if test_group_count >= len(task_groups):
            test_group_count = len(task_groups) - 1
        if test_group_count <= 0:
            raise RuntimeError(f"task-episode split requires at least two episodes for {task_name}")
        selected_test_groups.extend(task_groups[:test_group_count].tolist())

    test_mask = np.isin(group_keys, np.asarray(selected_test_groups))
    return SplitIndices(train=indices[~test_mask], test=indices[test_mask])


def read_alignment(raw: np.lib.npyio.NpzFile) -> str | None:
    if "observation_action_alignment" not in raw.files:
        return None
    value = raw["observation_action_alignment"]
    if value.shape == ():
        return str(value.item())
    if value.size == 1:
        return str(value.reshape(-1)[0])
    return str(value.tolist())


def require_pre_action_alignment(raw: np.lib.npyio.NpzFile, data_path: str | Path, allow_unaligned: bool) -> str | None:
    alignment = read_alignment(raw)
    if alignment == "pre_action":
        return alignment
    message = (
        f"Dataset {data_path} is not marked observation_action_alignment=pre_action "
        f"(found {alignment!r}). BC training requires obs/rgb/projection/action to be aligned "
        "at the pre-action state. Recollect the dataset with the current collector or pass "
        "--allow-unaligned-data only for historical plumbing checks."
    )
    if allow_unaligned:
        print(f"WARNING: {message}")
        return alignment
    raise ValueError(message)


class BCProjectionDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        indices: np.ndarray,
        patch_radius: int | None = None,
        include_residual_inputs: bool = True,
        include_residual_patch: bool = True,
        obs_mode: str = "full",
    ) -> None:
        self.path = Path(data_path)
        self.data = np.load(self.path, allow_pickle=False)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.include_residual_inputs = bool(include_residual_inputs)
        self.include_residual_patch = bool(include_residual_patch)
        self.obs_mode = obs_mode
        self.rgb = self.data["rgb"]
        self.obs = select_policy_observation(self.data["obs"], obs_mode)
        self.action = self.data["action"].astype(np.float32)
        self.keypoints_xyz = self.data["keypoints_xyz"].astype(np.float32)
        self.q_nominal = self.data["q_nominal"].astype(np.float32)
        self.q_true = self.data["q_true"].astype(np.float32)
        self.height = int(self.rgb.shape[1])
        self.width = int(self.rgb.shape[2])
        self.num_keypoints = int(self.q_nominal.shape[1])
        self.obs_dim = int(self.obs.shape[1])
        self.action_dim = int(self.action.shape[1])
        if patch_radius is None:
            patch_radius = int(np.asarray(self.data.get("patch_radius", 16)).item()) if "patch_radius" in self.data.files else 16
        self.patch_radius = int(patch_radius)
        self.residual_vector_dim = self._make_residual_vectors(0).shape[-1]
        self.proprio_dim = 3 + self.num_keypoints

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _crop_patch(self, sample_idx: int, keypoint_idx: int) -> np.ndarray:
        image = self.rgb[sample_idx]
        q_nom = self.q_nominal[sample_idx, keypoint_idx]
        cx = int(round(float(q_nom[0])))
        cy = int(round(float(q_nom[1])))
        radius = self.patch_radius
        size = 2 * radius + 1
        patch = np.zeros((size, size, 3), dtype=np.uint8)
        x0 = cx - radius
        y0 = cy - radius
        x1 = cx + radius + 1
        y1 = cy + radius + 1
        src_x0 = max(x0, 0)
        src_y0 = max(y0, 0)
        src_x1 = min(x1, self.width)
        src_y1 = min(y1, self.height)
        dst_x0 = src_x0 - x0
        dst_y0 = src_y0 - y0
        patch[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = image[
            src_y0:src_y1, src_x0:src_x1
        ]
        return np.transpose(patch.astype(np.float32) / 255.0, (2, 0, 1))

    def _make_residual_vectors(self, sample_idx: int) -> np.ndarray:
        q_nom = self.q_nominal[sample_idx].astype(np.float32)
        q_norm = np.stack([q_nom[:, 0] / self.width, q_nom[:, 1] / self.height], axis=-1).astype(np.float32)
        xyz = self.keypoints_xyz[sample_idx].astype(np.float32)
        onehot = np.eye(self.num_keypoints, dtype=np.float32)
        obs = np.repeat(self.obs[sample_idx][None, :], self.num_keypoints, axis=0).astype(np.float32)
        return np.concatenate([q_norm, xyz, onehot, obs], axis=-1)

    def _make_proprio(self, sample_idx: int) -> np.ndarray:
        xyz = self.keypoints_xyz[sample_idx].astype(np.float32)
        onehot = np.eye(self.num_keypoints, dtype=np.float32)
        return np.concatenate([xyz, onehot], axis=-1)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        sample_idx = int(self.indices[item])
        image = np.transpose(self.rgb[sample_idx].astype(np.float32) / 255.0, (2, 0, 1))
        result = {
            "sample_idx": torch.tensor(sample_idx, dtype=torch.long),
            "rgb": torch.from_numpy(image),
            "obs": torch.from_numpy(self.obs[sample_idx]),
            "action": torch.from_numpy(self.action[sample_idx]),
            "q_nominal": torch.from_numpy(self.q_nominal[sample_idx]),
            "q_true": torch.from_numpy(self.q_true[sample_idx]),
            "proprio": torch.from_numpy(self._make_proprio(sample_idx)),
        }
        if self.include_residual_inputs:
            result["residual_vector"] = torch.from_numpy(self._make_residual_vectors(sample_idx))
            if self.include_residual_patch:
                patches = np.stack([self._crop_patch(sample_idx, k) for k in range(self.num_keypoints)], axis=0)
                result["residual_patch"] = torch.from_numpy(patches)
        return result


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, layers: int = 3) -> None:
        super().__init__()
        dims = [input_dim] + [hidden_dim] * max(1, layers - 1) + [output_dim]
        modules: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            modules.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        modules.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SIGReg(nn.Module):
    """LeWM-style sketch isotropic Gaussian regularizer for latent tokens.

    LeWM applies this to latent embeddings alongside its prediction loss. In
    this BC scaffold we use the same regularizer on policy representations,
    paired with action MSE. The input is expected to be `[T, B, D]`; `[B, D]`
    is accepted as a single-token sequence.
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        self.num_proj = int(num_proj)
        t = torch.linspace(0.0, 3.0, int(knots), dtype=torch.float32)
        dt = 3.0 / max(int(knots) - 1, 1)
        weights = torch.full((int(knots),), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        if proj.ndim == 2:
            proj = proj.unsqueeze(0)
        if proj.ndim != 3:
            raise ValueError(f"SIGReg expects [T, B, D] or [B, D], got {tuple(proj.shape)}")
        random_proj = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        random_proj = random_proj / random_proj.norm(p=2, dim=0, keepdim=True).clamp_min(1e-6)
        x_t = (proj @ random_proj).unsqueeze(-1) * self.t.to(dtype=proj.dtype)
        err = (x_t.cos().mean(dim=0) - self.phi.to(dtype=proj.dtype)).square() + x_t.sin().mean(dim=0).square()
        statistic = (err @ self.weights.to(dtype=proj.dtype)) * proj.size(1)
        return statistic.mean()


class LeWMFeedForward(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LeWMAttention(nn.Module):
    """Small LeWM-style multi-head attention block without external deps."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int | None = None, dropout: float = 0.0, causal: bool = False) -> None:
        super().__init__()
        self.heads = int(heads)
        self.dim_head = int(dim_head or max(1, dim // self.heads))
        self.inner_dim = self.heads * self.dim_head
        self.dropout = float(dropout)
        self.causal = bool(causal)
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(self.inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, tokens, self.heads, self.dim_head).transpose(1, 2).contiguous()

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal,
        )
        out = out.transpose(1, 2).contiguous().view(batch, tokens, self.inner_dim)
        return self.to_out(out)


class LeWMBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0, causal: bool = False) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = LeWMAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout, causal=causal)
        self.mlp = LeWMFeedForward(dim, mlp_dim=mlp_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LeWMTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                LeWMBlock(hidden_dim, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout, causal=causal)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.output_proj = nn.Identity() if hidden_dim == output_dim else nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(self.norm(x))


class ViTBCPolicyHead(nn.Module):
    """LeWM-style ViT adapter for BC policy smoke tests.

    Tokens are `[CLS, obs, image patches, optional grounded keypoint tokens]`.
    The CLS output predicts the action, and the full encoded token sequence is
    returned for optional SIGReg.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_keypoints: int,
        token_dim: int,
        image_size: tuple[int, int],
        image_patch_size: int = 16,
        transformer_layers: int = 4,
        transformer_heads: int = 4,
        transformer_mlp_ratio: int = 4,
        transformer_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if token_dim % transformer_heads != 0:
            raise ValueError(f"token_dim={token_dim} must be divisible by transformer_heads={transformer_heads}")
        width, height = image_size
        if width % image_patch_size != 0 or height % image_patch_size != 0:
            raise ValueError(f"image_size={image_size} must be divisible by image_patch_size={image_patch_size}")
        self.image_size = (int(width), int(height))
        self.image_patch_size = int(image_patch_size)
        self.num_image_tokens = (width // image_patch_size) * (height // image_patch_size)
        self.num_keypoints = int(num_keypoints)
        self.patch_embed = nn.Conv2d(3, token_dim, kernel_size=image_patch_size, stride=image_patch_size)
        self.obs_proj = nn.Sequential(nn.LayerNorm(obs_dim), nn.Linear(obs_dim, token_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.obs_pos = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.image_pos = nn.Parameter(torch.zeros(1, self.num_image_tokens, token_dim))
        self.keypoint_pos = nn.Parameter(torch.zeros(1, max(self.num_keypoints, 1), token_dim))
        self.transformer = LeWMTransformer(
            input_dim=token_dim,
            hidden_dim=token_dim,
            output_dim=token_dim,
            depth=transformer_layers,
            heads=transformer_heads,
            dim_head=token_dim // transformer_heads,
            mlp_dim=token_dim * transformer_mlp_ratio,
            dropout=transformer_dropout,
            causal=False,
        )
        self.action_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, action_dim),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_pos, std=0.02)
        nn.init.trunc_normal_(self.obs_pos, std=0.02)
        nn.init.trunc_normal_(self.image_pos, std=0.02)
        nn.init.trunc_normal_(self.keypoint_pos, std=0.02)

    def forward_with_tokens(
        self,
        rgb: torch.Tensor,
        obs: torch.Tensor,
        grounded_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = rgb.shape[0]
        image_tokens = self.patch_embed(rgb).flatten(2).transpose(1, 2).contiguous()
        if image_tokens.shape[1] != self.num_image_tokens:
            raise ValueError(f"expected {self.num_image_tokens} image tokens, got {image_tokens.shape[1]}")
        cls = self.cls_token.expand(batch, -1, -1) + self.cls_pos
        obs_token = self.obs_proj(obs).unsqueeze(1) + self.obs_pos
        image_tokens = image_tokens + self.image_pos
        parts = [cls, obs_token, image_tokens]
        if grounded_tokens is not None and grounded_tokens.numel() > 0:
            if grounded_tokens.shape[-1] != image_tokens.shape[-1]:
                raise ValueError("grounded token dim must match token_dim")
            keypoint_pos = self.keypoint_pos[:, : grounded_tokens.shape[1], :]
            parts.append(grounded_tokens + keypoint_pos)
        encoded = self.transformer(torch.cat(parts, dim=1))
        action = self.action_head(encoded[:, 0])
        return action, encoded

    def forward(self, rgb: torch.Tensor, obs: torch.Tensor, grounded_tokens: torch.Tensor | None = None) -> torch.Tensor:
        action, _ = self.forward_with_tokens(rgb, obs, grounded_tokens)
        return action


class MinimalBCPolicy(nn.Module):
    def __init__(
        self,
        model_type: str,
        obs_dim: int,
        action_dim: int,
        num_keypoints: int,
        proprio_dim: int,
        token_dim: int,
        hidden_dim: int,
        residual_model: ResidualCorrectionHead | None = None,
        image_size: tuple[int, int] = (128, 128),
        freeze_residual: bool = True,
        policy_arch: str = "mlp",
        image_patch_size: int = 16,
        transformer_layers: int = 4,
        transformer_heads: int = 4,
        transformer_mlp_ratio: int = 4,
        transformer_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if policy_arch not in {"mlp", "vit"}:
            raise ValueError(f"unknown policy_arch={policy_arch!r}")
        self.model_type = model_type
        self.policy_arch = policy_arch
        self.num_keypoints = int(num_keypoints)
        self.image_size = image_size
        self.residual_model = residual_model
        if self.residual_model is not None and freeze_residual:
            self.residual_model.eval()
            for param in self.residual_model.parameters():
                param.requires_grad_(False)
        if model_type == "state":
            self.token_adapter = None
        else:
            self.token_adapter = GroundedTokenAdapter(
                visual_dim=3,
                proprio_dim=proprio_dim,
                token_dim=token_dim,
                hidden_dim=hidden_dim,
                include_reliability=False,
            )
        if self.policy_arch == "mlp":
            policy_input_dim = obs_dim if model_type == "state" else obs_dim + self.num_keypoints * token_dim
            self.policy = MLP(policy_input_dim, action_dim, hidden_dim=hidden_dim, layers=3)
        else:
            self.policy = ViTBCPolicyHead(
                obs_dim=obs_dim,
                action_dim=action_dim,
                num_keypoints=0 if model_type == "state" else self.num_keypoints,
                token_dim=token_dim,
                image_size=image_size,
                image_patch_size=image_patch_size,
                transformer_layers=transformer_layers,
                transformer_heads=transformer_heads,
                transformer_mlp_ratio=transformer_mlp_ratio,
                transformer_dropout=transformer_dropout,
            )

    def corrected_points(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.model_type == "geoprop":
            return batch["q_nominal"]
        if self.model_type == "oracle":
            return batch["q_true"]
        if self.model_type == "fcpg":
            if self.residual_model is None:
                raise ValueError("FCPG model requires a residual checkpoint")
            vectors = batch["residual_vector"]
            patches = batch.get("residual_patch")
            if patches is None:
                patches = sample_image_patches_at_pixels(
                    batch["rgb"],
                    batch["q_nominal"],
                    radius=int(self.residual_model.patch_radius),
                )
            bsz, num, vec_dim = vectors.shape
            pred_norm = self.residual_model(vectors.view(bsz * num, vec_dim), patches.view(bsz * num, *patches.shape[2:]))
            delta_px = pred_norm.view(bsz, num, 2) * float(self.residual_model.patch_radius)
            return batch["q_nominal"] + delta_px
        raise ValueError(f"no corrected points for model type {self.model_type}")

    def grounded_tokens(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
        if self.model_type == "state":
            return None
        assert self.token_adapter is not None
        points = self.corrected_points(batch)
        return self.token_adapter(batch["rgb"], points, batch["proprio"], image_size=self.image_size)

    def forward_with_latent(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        obs = batch["obs"]
        tokens = self.grounded_tokens(batch)
        if self.policy_arch == "mlp":
            if tokens is None:
                latent = obs
            else:
                latent = torch.cat([obs, tokens.flatten(start_dim=1)], dim=1)
            return self.policy(latent), latent
        assert isinstance(self.policy, ViTBCPolicyHead)
        return self.policy.forward_with_tokens(batch["rgb"], obs, tokens)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        pred, _ = self.forward_with_latent(batch)
        return pred

def load_residual_checkpoint(path: str | Path, device: torch.device) -> tuple[ResidualCorrectionHead, dict[str, object], int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})
    vector_dim = int(checkpoint["vector_dim"])
    hidden_dim = int(args.get("hidden_dim", 128))
    patch_radius = int(args.get("patch_radius", 16))
    model = ResidualCorrectionHead(vector_dim, hidden_dim=hidden_dim, patch_radius=patch_radius)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, args, vector_dim


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    non_blocking = device.type == "cuda"
    return {key: value.to(device, non_blocking=non_blocking) for key, value in batch.items()}


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses = []
    maes = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            pred = model(batch)
            target = batch["action"]
            losses.append(torch.nn.functional.mse_loss(pred, target).item())
            maes.append(torch.nn.functional.l1_loss(pred, target).item())
    return {"mse": float(np.mean(losses)), "mae": float(np.mean(maes))}


def sigreg_input_from_latent(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim == 3:
        return latent.transpose(0, 1).contiguous()
    if latent.ndim == 2:
        return latent.unsqueeze(0)
    raise ValueError(f"unsupported latent shape for SIGReg: {tuple(latent.shape)}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    sigreg: SIGReg | None = None,
    sigreg_weight: float = 0.0,
) -> dict[str, float]:
    model.train()
    if getattr(model, "residual_model", None) is not None:
        model.residual_model.eval()
    losses = []
    mses = []
    sigregs = []
    for batch in loader:
        batch = move_batch(batch, device)
        pred, latent = model.forward_with_latent(batch) if sigreg is not None and sigreg_weight > 0 else (model(batch), None)
        mse = F.mse_loss(pred, batch["action"])
        if sigreg is not None and sigreg_weight > 0:
            sig_value = sigreg(sigreg_input_from_latent(latent))
            loss = mse + float(sigreg_weight) * sig_value
        else:
            sig_value = mse.new_zeros(())
            loss = mse
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        mses.append(float(mse.item()))
        sigregs.append(float(sig_value.item()))
    return {
        "loss": float(np.mean(losses)),
        "mse": float(np.mean(mses)),
        "sigreg": float(np.mean(sigregs)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--tasks", default="", help="Optional comma-separated task filter; empty uses all tasks.")
    parser.add_argument("--model", choices=["state", "geoprop", "oracle", "fcpg"], required=True)
    parser.add_argument("--residual-checkpoint", default="")
    parser.add_argument("--output-dir", default="outputs/bc_policy")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers for RGB conversion and local-patch cropping.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=4,
        help="Limit CPU intra-op threads so CUDA training is not starved by oversubscription.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--token-dim", type=int, default=32)
    parser.add_argument("--policy-arch", choices=["mlp", "vit"], default="mlp")
    parser.add_argument(
        "--obs-mode",
        choices=OBSERVATION_MODES,
        default="full",
        help="Use robot to exclude MetaWorld object, previous-state, and goal values from policy inputs.",
    )
    parser.add_argument("--image-patch-size", type=int, default=16)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-mlp-ratio", type=int, default=4)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument("--sigreg-weight", type=float, default=0.0)
    parser.add_argument("--sigreg-proj", type=int, default=1024)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--random-split-unit", choices=["sample", "task-episode"], default="sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Data-split seed; defaults to --seed for backward-compatible runs.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-residual", action="store_true")
    parser.add_argument(
        "--allow-unaligned-data",
        action="store_true",
        help="Allow legacy datasets without observation_action_alignment=pre_action; use only for plumbing checks.",
    )
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    if args.split_seed is None:
        args.split_seed = args.seed
    torch.set_num_threads(args.torch_threads)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)
    raw = np.load(args.data, allow_pickle=False)
    alignment = require_pre_action_alignment(raw, args.data, args.allow_unaligned_data)
    split = make_split(raw, args.val_fraction, args.split_seed, args.random_split_unit, args.tasks)

    residual_model = None
    residual_args: dict[str, object] = {}
    residual_vector_dim = None
    patch_radius = int(np.asarray(raw["patch_radius"]).item()) if "patch_radius" in raw.files else 16
    if args.model == "fcpg":
        if not args.residual_checkpoint:
            raise ValueError("--residual-checkpoint is required for --model fcpg")
        residual_model, residual_args, residual_vector_dim = load_residual_checkpoint(args.residual_checkpoint, device)
        patch_radius = int(residual_args.get("patch_radius", patch_radius))
        residual_obs_mode = str(residual_args.get("obs_mode", "full"))
        if residual_obs_mode != args.obs_mode:
            raise ValueError(
                f"Residual checkpoint uses obs_mode={residual_obs_mode!r}, but policy uses obs_mode={args.obs_mode!r}."
            )

    needs_residual = args.model == "fcpg"
    train_ds = BCProjectionDataset(
        args.data,
        split.train,
        patch_radius=patch_radius,
        include_residual_inputs=needs_residual,
        include_residual_patch=False,
        obs_mode=args.obs_mode,
    )
    test_ds = BCProjectionDataset(
        args.data,
        split.test,
        patch_radius=patch_radius,
        include_residual_inputs=needs_residual,
        include_residual_patch=False,
        obs_mode=args.obs_mode,
    )
    if residual_vector_dim is not None and train_ds.residual_vector_dim != residual_vector_dim:
        raise ValueError(
            f"Residual checkpoint expects vector_dim={residual_vector_dim}, dataset provides {train_ds.residual_vector_dim}"
        )
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    model = MinimalBCPolicy(
        args.model,
        obs_dim=train_ds.obs_dim,
        action_dim=train_ds.action_dim,
        num_keypoints=train_ds.num_keypoints,
        proprio_dim=train_ds.proprio_dim,
        token_dim=args.token_dim,
        hidden_dim=args.hidden_dim,
        residual_model=residual_model,
        image_size=(train_ds.width, train_ds.height),
        freeze_residual=not args.train_residual,
        policy_arch=args.policy_arch,
        image_patch_size=args.image_patch_size,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_mlp_ratio=args.transformer_mlp_ratio,
        transformer_dropout=args.transformer_dropout,
    ).to(device)
    sigreg = SIGReg(knots=args.sigreg_knots, num_proj=args.sigreg_proj).to(device) if args.sigreg_weight > 0 else None
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    print(f"data: {args.data}")
    print(f"observation_action_alignment: {alignment}")
    print(f"model: {args.model}")
    print(f"tasks: {args.tasks or 'all'}")
    print(f"policy_arch: {args.policy_arch}")
    print(f"obs_mode: {args.obs_mode}")
    print(f"sigreg_weight: {args.sigreg_weight}")
    print(f"device: {device}")
    print(
        f"num_workers: {args.num_workers} pin_memory: {device.type == 'cuda'} "
        f"torch_threads: {args.torch_threads}"
    )
    print(f"random_split_unit: {args.random_split_unit}")
    print(f"model_seed: {args.seed} split_seed: {args.split_seed}")
    print(f"train_frames: {len(train_ds)} test_frames: {len(test_ds)}")
    print(f"obs_dim: {train_ds.obs_dim} action_dim: {train_ds.action_dim} num_keypoints: {train_ds.num_keypoints}")
    print(f"patch_radius: {train_ds.patch_radius} residual_vector_dim: {train_ds.residual_vector_dim}")
    history = []
    best = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            sigreg=sigreg,
            sigreg_weight=args.sigreg_weight,
        )
        metrics = evaluate(model, test_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mse": train_metrics["mse"],
            "train_sigreg": train_metrics["sigreg"],
            **metrics,
        }
        history.append(row)
        if best is None or row["mse"] < best["mse"]:
            best = dict(row)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"epoch {epoch:03d} train_loss={row['train_loss']:.6f} train_mse={row['train_mse']:.6f} "
            f"train_sigreg={row['train_sigreg']:.6f} test_mse={row['mse']:.6f} test_mae={row['mae']:.6f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{args.model}_bc.pt"
    best_model_path = output_dir / f"{args.model}_bc_best.pt"
    metrics_path = output_dir / f"{args.model}_metrics.json"
    payload = {
        "model_state": model.state_dict(),
        "args": vars(args),
        "obs_dim": train_ds.obs_dim,
        "action_dim": train_ds.action_dim,
        "num_keypoints": train_ds.num_keypoints,
        "proprio_dim": train_ds.proprio_dim,
    }
    torch.save(payload, model_path)
    if best_state is not None:
        best_payload = dict(payload)
        best_payload["model_state"] = best_state
        torch.save(best_payload, best_model_path)
    metrics_path.write_text(json.dumps({"final": history[-1], "best": best, "history": history}, indent=2), encoding="utf-8")
    print(f"model_path: {model_path}")
    print(f"best_model_path: {best_model_path}")
    print(f"metrics_path: {metrics_path}")
    if best:
        print(f"best_epoch: {best['epoch']} best_mse: {best['mse']:.6f} best_mae: {best['mae']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
