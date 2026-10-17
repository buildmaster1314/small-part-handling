"""Non-FCPG visual refinement and online calibration baselines."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from fcpg.geometry import Camera, project_points
from fcpg.transformer_grounding import sample_local_feature_tokens


@dataclass(frozen=True)
class PrototypeRefinement:
    points_px: torch.Tensor
    confidence: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class PrototypeHypotheses:
    """Joint prototype matches for all anchors.

    ``M`` is the requested joint-hypothesis budget (capped by the Cartesian
    product size).  With NMS enabled, suppressed tail entries are padded with
    ``joint_score=-inf``, ``grid_indices=-1``, and ``valid=False``.
    """

    points_px: torch.Tensor
    joint_score: torch.Tensor
    valid: torch.Tensor
    grid_indices: torch.Tensor


def _prototype_score_grid(
    feature_map: torch.Tensor,
    q_nominal_px: torch.Tensor,
    prototype_bank: torch.Tensor,
    image_size: tuple[int, int],
    radius_px: float,
    grid_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-anchor prototype scores and their pixel offsets."""

    local, normalized_offsets = sample_local_feature_tokens(
        feature_map,
        q_nominal_px,
        image_size=image_size,
        radius_px=radius_px,
        grid_size=grid_size,
    )
    local = F.normalize(local.float(), dim=-1)
    prototypes = F.normalize(
        prototype_bank.to(device=local.device, dtype=local.dtype),
        dim=-1,
    )
    scores = torch.einsum("bknc,kmc->bknm", local, prototypes).amax(dim=-1)
    offsets_px = normalized_offsets.to(local) * float(radius_px)
    return scores, offsets_px


def _validate_prototype_inputs(
    feature_map: torch.Tensor,
    q_nominal_px: torch.Tensor,
    prototype_bank: torch.Tensor,
) -> torch.Tensor:
    if prototype_bank.ndim == 2:
        prototype_bank = prototype_bank[:, None, :]
    if prototype_bank.ndim != 3:
        raise ValueError("prototype_bank must have shape [K, M, C] or [K, C]")
    if q_nominal_px.ndim != 3 or q_nominal_px.shape[-1] != 2:
        raise ValueError("q_nominal_px must have shape [B, K, 2]")
    if q_nominal_px.shape[1] == 0:
        raise ValueError("q_nominal_px must contain at least one keypoint")
    if prototype_bank.shape[0] != q_nominal_px.shape[1]:
        raise ValueError("prototype keypoint count does not match q_nominal_px")
    if feature_map.ndim != 4:
        raise ValueError("feature_map must have shape [B, C, H, W]")
    if feature_map.shape[0] != q_nominal_px.shape[0]:
        raise ValueError("feature_map batch size does not match q_nominal_px")
    if prototype_bank.shape[-1] != feature_map.shape[1]:
        raise ValueError("prototype feature dimension does not match feature_map")
    return prototype_bank


def refine_points_with_prototypes(
    feature_map: torch.Tensor,
    q_nominal_px: torch.Tensor,
    prototype_bank: torch.Tensor,
    image_size: tuple[int, int],
    radius_px: float = 32.0,
    grid_size: int = 9,
    temperature: float = 0.0,
    min_confidence: float = -1.0,
) -> PrototypeRefinement:
    """Match frozen visual descriptors in a geometry-anchored local window."""

    prototype_bank = _validate_prototype_inputs(
        feature_map,
        q_nominal_px,
        prototype_bank,
    )
    scores, offsets_px = _prototype_score_grid(
        feature_map,
        q_nominal_px,
        prototype_bank,
        image_size,
        radius_px,
        grid_size,
    )
    if temperature > 0:
        weights = torch.softmax(scores / float(temperature), dim=-1)
        delta_px = torch.einsum("bkn,nd->bkd", weights, offsets_px)
    else:
        best = scores.argmax(dim=-1)
        delta_px = offsets_px[best]
    points = q_nominal_px.to(delta_px) + delta_px
    confidence = scores.max(dim=-1).values
    width, height = image_size
    valid = (
        (points[..., 0] >= 0)
        & (points[..., 0] < width)
        & (points[..., 1] >= 0)
        & (points[..., 1] < height)
        & (confidence >= float(min_confidence))
    )
    return PrototypeRefinement(points, confidence, valid)


def generate_joint_prototype_hypotheses(
    feature_map: torch.Tensor,
    q_nominal_px: torch.Tensor,
    prototype_bank: torch.Tensor,
    image_size: tuple[int, int],
    radius_px: float = 64.0,
    grid_size: int = 17,
    per_anchor_topk: int = 16,
    max_hypotheses: int = 16,
    nms_radius_px: float = 0.0,
    min_confidence: float = -1.0,
) -> PrototypeHypotheses:
    """Generate exact top joint hypotheses from per-anchor prototype grids.

    Each joint score is the sum of the selected per-anchor cosine scores.  The
    returned hypotheses are the true top entries of the Cartesian product;
    candidates with the same per-anchor rank are not artificially paired.
    Ties are resolved lexicographically by flattened grid indices.

    Optional NMS greedily suppresses a configuration when every corresponding
    anchor is within ``nms_radius_px`` of an already retained configuration.
    NMS is applied to the bounded top-``max_hypotheses`` list and does not widen
    the search budget to refill suppressed entries.
    """

    if per_anchor_topk <= 0:
        raise ValueError("per_anchor_topk must be positive")
    if max_hypotheses <= 0:
        raise ValueError("max_hypotheses must be positive")
    if nms_radius_px < 0:
        raise ValueError("nms_radius_px must be non-negative")
    prototype_bank = _validate_prototype_inputs(
        feature_map,
        q_nominal_px,
        prototype_bank,
    )
    scores, offsets_px = _prototype_score_grid(
        feature_map,
        q_nominal_px,
        prototype_bank,
        image_size,
        radius_px,
        grid_size,
    )
    batch_size, num_keypoints, num_grid_points = scores.shape
    local_topk = min(int(per_anchor_topk), int(num_grid_points))
    hypothesis_count = min(
        int(max_hypotheses),
        int(local_topk**num_keypoints),
    )

    # Stable sorting retains ascending flattened grid index for exact ties.
    per_anchor_indices = torch.argsort(
        scores,
        dim=-1,
        descending=True,
        stable=True,
    )[..., :local_topk]
    per_anchor_scores = scores.gather(-1, per_anchor_indices)

    # Joint selection is discrete.  Transfer the compact per-anchor shortlist
    # once instead of calling ``.item()`` for every candidate on CUDA.
    indices_cpu = per_anchor_indices.detach().cpu().tolist()
    scores_cpu = per_anchor_scores.detach().cpu().tolist()
    selected_indices: list[list[tuple[int, ...]]] = []
    for batch_index in range(batch_size):
        beams: list[tuple[float, tuple[int, ...]]] = [(0.0, ())]
        for keypoint_index in range(num_keypoints):
            expanded: list[tuple[float, tuple[int, ...]]] = []
            for partial_score, partial_indices in beams:
                for candidate_rank in range(local_topk):
                    grid_index = indices_cpu[batch_index][keypoint_index][
                        candidate_rank
                    ]
                    candidate_score = scores_cpu[batch_index][keypoint_index][
                        candidate_rank
                    ]
                    expanded.append(
                        (
                            partial_score + candidate_score,
                            partial_indices + (grid_index,),
                        )
                    )
            expanded.sort(key=lambda item: (-item[0], item[1]))
            beams = expanded[:hypothesis_count]
        selected_indices.append([indices for _, indices in beams])

    grid_indices = torch.tensor(
        selected_indices,
        device=scores.device,
        dtype=torch.long,
    )
    batch_index = torch.arange(batch_size, device=scores.device)[:, None, None]
    keypoint_index = torch.arange(num_keypoints, device=scores.device)[None, None, :]
    selected_scores = scores[batch_index, keypoint_index, grid_indices]
    joint_score = selected_scores.sum(dim=-1)
    points = q_nominal_px.to(offsets_px)[:, None, :, :] + offsets_px[grid_indices]
    width, height = image_size
    valid = (
        (points[..., 0] >= 0)
        & (points[..., 0] < width)
        & (points[..., 1] >= 0)
        & (points[..., 1] < height)
        & (selected_scores >= float(min_confidence))
    )

    if nms_radius_px > 0 and hypothesis_count > 1:
        compact_points = points.clone()
        compact_scores = joint_score.new_full(joint_score.shape, -torch.inf)
        compact_valid = torch.zeros_like(valid)
        compact_indices = torch.full_like(grid_indices, -1)
        for current_batch in range(batch_size):
            retained: list[int] = []
            for hypothesis_index in range(hypothesis_count):
                candidate = points[current_batch, hypothesis_index]
                suppressed = any(
                    bool(
                        (
                            torch.linalg.vector_norm(
                                candidate - points[current_batch, retained_index],
                                dim=-1,
                            ).amax()
                            <= float(nms_radius_px)
                        ).item()
                    )
                    for retained_index in retained
                )
                if not suppressed:
                    retained.append(hypothesis_index)
            retained_tensor = torch.tensor(
                retained,
                device=scores.device,
                dtype=torch.long,
            )
            retained_count = len(retained)
            compact_points[current_batch, :retained_count] = points[
                current_batch,
                retained_tensor,
            ]
            compact_scores[current_batch, :retained_count] = joint_score[
                current_batch,
                retained_tensor,
            ]
            compact_valid[current_batch, :retained_count] = valid[
                current_batch,
                retained_tensor,
            ]
            compact_indices[current_batch, :retained_count] = grid_indices[
                current_batch,
                retained_tensor,
            ]
            if retained_count < hypothesis_count:
                compact_points[current_batch, retained_count:] = q_nominal_px[
                    current_batch,
                    None,
                    :,
                ].to(compact_points)
        points = compact_points
        joint_score = compact_scores
        valid = compact_valid
        grid_indices = compact_indices

    return PrototypeHypotheses(points, joint_score, valid, grid_indices)


@dataclass(frozen=True)
class PnPUpdate:
    points_px: np.ndarray
    extrinsic_cam_from_world: np.ndarray
    solved: bool
    reprojection_error_px: float
    correspondence_count: int


class OnlinePnPCalibrator:
    """Estimate one shared camera extrinsic from recent visual correspondences."""

    def __init__(
        self,
        intrinsic: np.ndarray,
        nominal_extrinsic: np.ndarray,
        image_size: tuple[int, int],
        history_frames: int = 3,
        max_reprojection_error_px: float = 8.0,
        max_pose_rotation_deg: float = 30.0,
        max_pose_translation_cm: float = 20.0,
    ) -> None:
        if history_frames < 1:
            raise ValueError("history_frames must be positive")
        self.intrinsic = np.asarray(intrinsic, dtype=np.float64)
        self.nominal_extrinsic = np.asarray(nominal_extrinsic, dtype=np.float64)
        if self.intrinsic.shape != (3, 3) or self.nominal_extrinsic.shape != (4, 4):
            raise ValueError("invalid camera matrix shape")
        self.image_size = tuple(int(value) for value in image_size)
        self.max_reprojection_error_px = float(max_reprojection_error_px)
        self.max_pose_rotation_deg = float(max_pose_rotation_deg)
        self.max_pose_translation_cm = float(max_pose_translation_cm)
        self._frames: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=history_frames)
        self._extrinsic = self.nominal_extrinsic.copy()

    def update(
        self,
        points_xyz: np.ndarray,
        image_points_px: np.ndarray,
        valid: np.ndarray | None = None,
    ) -> PnPUpdate:
        points_xyz = np.asarray(points_xyz, dtype=np.float64)
        image_points_px = np.asarray(image_points_px, dtype=np.float64)
        if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
            raise ValueError("points_xyz must have shape [K, 3]")
        if image_points_px.shape != (points_xyz.shape[0], 2):
            raise ValueError("image_points_px must have shape [K, 2]")
        mask = np.ones(points_xyz.shape[0], dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        if mask.shape != (points_xyz.shape[0],):
            raise ValueError("valid must have shape [K]")
        finite = np.isfinite(points_xyz).all(axis=1) & np.isfinite(image_points_px).all(axis=1)
        mask &= finite
        self._frames.append((points_xyz[mask].copy(), image_points_px[mask].copy()))
        object_points = np.concatenate([frame[0] for frame in self._frames], axis=0)
        image_points = np.concatenate([frame[1] for frame in self._frames], axis=0)

        solved = False
        reprojection_error = float("inf")
        if object_points.shape[0] >= 4:
            rvec, _ = cv2.Rodrigues(self._extrinsic[:3, :3])
            tvec = self._extrinsic[:3, 3:4].copy()
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                self.intrinsic,
                None,
                rvec,
                tvec,
                True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok:
                projected, _ = cv2.projectPoints(
                    object_points,
                    rvec,
                    tvec,
                    self.intrinsic,
                    None,
                )
                projected = projected.reshape(-1, 2)
                reprojection_error = float(
                    np.linalg.norm(projected - image_points, axis=1).mean()
                )
                rotation, _ = cv2.Rodrigues(rvec)
                candidate = np.eye(4, dtype=np.float64)
                candidate[:3, :3] = rotation
                candidate[:3, 3] = tvec.reshape(3)
                rotation_delta = rotation @ self.nominal_extrinsic[:3, :3].T
                angle_cos = np.clip(
                    (np.trace(rotation_delta) - 1.0) / 2.0,
                    -1.0,
                    1.0,
                )
                rotation_delta_deg = float(np.degrees(np.arccos(angle_cos)))
                candidate_center = -rotation.T @ candidate[:3, 3]
                nominal_center = (
                    -self.nominal_extrinsic[:3, :3].T
                    @ self.nominal_extrinsic[:3, 3]
                )
                translation_delta_cm = float(
                    100.0 * np.linalg.norm(candidate_center - nominal_center)
                )
                current_camera = Camera(
                    intrinsic=self.intrinsic,
                    extrinsic_cam_from_world=candidate,
                    width=self.image_size[0],
                    height=self.image_size[1],
                )
                current_projected, current_valid = project_points(
                    points_xyz[mask],
                    current_camera,
                )
                current_error = (
                    float(
                        np.linalg.norm(
                            current_projected - image_points_px[mask],
                            axis=1,
                        ).mean()
                    )
                    if current_projected.shape[0]
                    else float("inf")
                )
                accept = (
                    np.isfinite(reprojection_error)
                    and reprojection_error <= self.max_reprojection_error_px
                    and np.isfinite(current_error)
                    and current_error <= self.max_reprojection_error_px
                    and bool(current_valid.all())
                    and rotation_delta_deg <= self.max_pose_rotation_deg
                    and translation_delta_cm <= self.max_pose_translation_cm
                )
                if accept:
                    self._extrinsic = candidate
                    solved = True

        camera = Camera(
            intrinsic=self.intrinsic,
            extrinsic_cam_from_world=self._extrinsic,
            width=self.image_size[0],
            height=self.image_size[1],
        )
        points_px, _ = project_points(points_xyz, camera)
        return PnPUpdate(
            points_px=points_px.astype(np.float32),
            extrinsic_cam_from_world=self._extrinsic.copy(),
            solved=solved,
            reprojection_error_px=reprojection_error,
            correspondence_count=int(object_points.shape[0]),
        )
