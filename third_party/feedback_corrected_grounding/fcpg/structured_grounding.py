"""Differentiable geometry for Stage-II structured kinematic grounding."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import torch
from torch import nn


def skew_symmetric(vector: torch.Tensor) -> torch.Tensor:
    """Return the skew-symmetric matrix for vectors ending in three values."""

    if vector.shape[-1] != 3:
        raise ValueError(f"vector must end in 3 values, got {tuple(vector.shape)}")
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero],
        dim=-1,
    ).reshape(*vector.shape[:-1], 3, 3)


def se3_exp(twist: torch.Tensor) -> torch.Tensor:
    """Map twists ``[..., 6]`` to homogeneous transforms.

    The twist convention is ``[v_x, v_y, v_z, w_x, w_y, w_z]``. Translation
    is expressed in the same metric units as the robot kinematic model.
    """

    if twist.shape[-1] != 6:
        raise ValueError(f"twist must end in 6 values, got {tuple(twist.shape)}")
    translation_velocity = twist[..., :3]
    rotation_vector = twist[..., 3:]
    rotation_hat = skew_symmetric(rotation_vector)
    rotation_hat_squared = rotation_hat @ rotation_hat
    theta_squared = rotation_vector.square().sum(dim=-1, keepdim=True)
    theta = theta_squared.clamp_min(1e-16).sqrt()
    small = theta_squared < 1e-8
    coefficient_a = torch.where(
        small,
        1.0 - theta_squared / 6.0 + theta_squared.square() / 120.0,
        torch.sin(theta) / theta,
    )
    coefficient_b = torch.where(
        small,
        0.5 - theta_squared / 24.0 + theta_squared.square() / 720.0,
        (1.0 - torch.cos(theta)) / theta_squared.clamp_min(1e-16),
    )
    coefficient_c = torch.where(
        small,
        1.0 / 6.0 - theta_squared / 120.0 + theta_squared.square() / 5040.0,
        (theta - torch.sin(theta)) / (theta_squared * theta).clamp_min(1e-16),
    )
    identity = torch.eye(3, device=twist.device, dtype=twist.dtype).expand(
        *twist.shape[:-1], 3, 3
    )
    rotation = (
        identity
        + coefficient_a.unsqueeze(-1) * rotation_hat
        + coefficient_b.unsqueeze(-1) * rotation_hat_squared
    )
    left_jacobian = (
        identity
        + coefficient_b.unsqueeze(-1) * rotation_hat
        + coefficient_c.unsqueeze(-1) * rotation_hat_squared
    )
    translation = (left_jacobian @ translation_velocity.unsqueeze(-1)).squeeze(-1)
    transform = torch.zeros(
        *twist.shape[:-1],
        4,
        4,
        device=twist.device,
        dtype=twist.dtype,
    )
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = translation
    transform[..., 3, 3] = 1.0
    return transform


def project_points_torch(
    intrinsic: torch.Tensor,
    extrinsic_cam_from_world: torch.Tensor,
    points_world: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project batched 3D points with differentiable pinhole geometry."""

    if points_world.ndim < 2 or points_world.shape[-1] != 3:
        raise ValueError(
            f"points_world must have shape [..., N, 3], got {tuple(points_world.shape)}"
        )
    batch_shape = points_world.shape[:-2]
    num_points = points_world.shape[-2]
    intrinsic = torch.broadcast_to(intrinsic, (*batch_shape, 3, 3))
    extrinsic_cam_from_world = torch.broadcast_to(
        extrinsic_cam_from_world,
        (*batch_shape, 4, 4),
    )
    ones = torch.ones(
        *batch_shape,
        num_points,
        1,
        device=points_world.device,
        dtype=points_world.dtype,
    )
    homogeneous = torch.cat([points_world, ones], dim=-1)
    camera = torch.matmul(
        extrinsic_cam_from_world.unsqueeze(-3),
        homogeneous.unsqueeze(-1),
    ).squeeze(-1)[..., :3]
    image_homogeneous = torch.matmul(
        intrinsic.unsqueeze(-3),
        camera.unsqueeze(-1),
    ).squeeze(-1)
    depth = camera[..., 2]
    denominator = image_homogeneous[..., 2:3].clamp_min(1e-8)
    pixels = image_homogeneous[..., :2] / denominator
    return pixels, depth


def projection_jacobian_left_se3(
    intrinsic: torch.Tensor,
    extrinsic_cam_from_world: torch.Tensor,
    points_world: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project points and linearize pixels for a left-multiplied SE(3) update.

    The Jacobian uses the same [v, w] convention as se3_exp and satisfies
    q(Exp(delta_xi) @ T) approximately q(T) + J @ delta_xi.
    """

    if points_world.ndim < 2 or points_world.shape[-1] != 3:
        raise ValueError(
            f"points_world must have shape [..., N, 3], got {tuple(points_world.shape)}"
        )
    batch_shape = points_world.shape[:-2]
    num_points = points_world.shape[-2]
    intrinsic = torch.broadcast_to(intrinsic, (*batch_shape, 3, 3))
    extrinsic_cam_from_world = torch.broadcast_to(
        extrinsic_cam_from_world,
        (*batch_shape, 4, 4),
    )
    ones = torch.ones(
        *batch_shape,
        num_points,
        1,
        device=points_world.device,
        dtype=points_world.dtype,
    )
    homogeneous = torch.cat([points_world, ones], dim=-1)
    camera = torch.matmul(
        extrinsic_cam_from_world.unsqueeze(-3),
        homogeneous.unsqueeze(-1),
    ).squeeze(-1)[..., :3]
    image_homogeneous = torch.matmul(
        intrinsic.unsqueeze(-3),
        camera.unsqueeze(-1),
    ).squeeze(-1)
    denominator = image_homogeneous[..., 2:3].clamp_min(1e-8)
    denominator_squared = denominator.square()
    row_u = (
        intrinsic[..., None, 0, :] * denominator
        - intrinsic[..., None, 2, :] * image_homogeneous[..., 0:1]
    ) / denominator_squared
    row_v = (
        intrinsic[..., None, 1, :] * denominator
        - intrinsic[..., None, 2, :] * image_homogeneous[..., 1:2]
    ) / denominator_squared
    image_jacobian = torch.stack([row_u, row_v], dim=-2)
    identity = torch.eye(
        3,
        device=points_world.device,
        dtype=points_world.dtype,
    ).expand(*batch_shape, num_points, 3, 3)
    camera_motion_jacobian = torch.cat(
        [identity, -skew_symmetric(camera)],
        dim=-1,
    )
    jacobian = image_jacobian @ camera_motion_jacobian
    pixels = image_homogeneous[..., :2] / denominator
    return pixels, camera[..., 2], jacobian


def chi_square_rms_threshold(
    degrees_of_freedom: int,
    false_alarm_probability: float,
) -> float:
    """Approximate an upper-tail chi-square RMS threshold.

    Uses the Wilson-Hilferty transform so the observer can perform a
    rank-adaptive innovation-consistency test without a SciPy dependency.
    """

    if degrees_of_freedom < 1:
        raise ValueError("degrees_of_freedom must be positive")
    if not 0.0 < false_alarm_probability < 1.0:
        raise ValueError("false_alarm_probability must lie in (0, 1)")
    z_score = NormalDist().inv_cdf(1.0 - false_alarm_probability)
    correction = (
        1.0
        - 2.0 / (9.0 * degrees_of_freedom)
        + z_score * (2.0 / (9.0 * degrees_of_freedom)) ** 0.5
    )
    quantile = degrees_of_freedom * max(correction, 0.0) ** 3
    return (quantile / degrees_of_freedom) ** 0.5


@dataclass(frozen=True)
class WhitenedObserverUpdate:
    """A shared camera update and diagnostics from observer innovations."""

    twist: torch.Tensor
    singular_values: torch.Tensor
    retained_modes: torch.Tensor
    effective_rank: torch.Tensor
    valid_observer_count: torch.Tensor
    prior_control_variance: torch.Tensor
    posterior_control_variance: torch.Tensor
    control_variance_reduction_by_mode: torch.Tensor
    modal_innovation_z_score: torch.Tensor
    global_innovation_rms_z: torch.Tensor
    global_innovation_threshold_rms_z: torch.Tensor


def solve_whitened_observer_update(
    observer_jacobian: torch.Tensor,
    observer_innovation_px: torch.Tensor,
    control_jacobian: torch.Tensor,
    prior_std: torch.Tensor,
    observation_std_px: float | torch.Tensor,
    observer_valid: torch.Tensor | None = None,
    min_whitened_singular_value: float = 0.25,
    relative_singular_value: float = 1e-3,
    min_control_variance_reduction_fraction: float = 1e-6,
    min_innovation_z_score: float = 0.0,
    min_global_innovation_rms_z: float = 0.0,
    innovation_false_alarm_probability: float = 0.0,
) -> WhitenedObserverUpdate:
    """Solve a whitened MAP update while rejecting unobservable modes.

    Control-anchor Jacobians affect only the task-oriented mode utility. No
    control-anchor image correspondence or target pixel enters this solver.
    """

    if observer_jacobian.ndim != 4 or observer_jacobian.shape[-2:] != (2, 6):
        raise ValueError("observer_jacobian must have shape [B, N, 2, 6]")
    if observer_innovation_px.shape != observer_jacobian.shape[:-1]:
        raise ValueError(
            "observer_innovation_px must match observer_jacobian [B, N, 2]"
        )
    if control_jacobian.ndim != 4 or control_jacobian.shape[-2:] != (2, 6):
        raise ValueError("control_jacobian must have shape [B, M, 2, 6]")
    if control_jacobian.shape[0] != observer_jacobian.shape[0]:
        raise ValueError("observer and control Jacobians must share batch size")
    if min_whitened_singular_value < 0 or relative_singular_value < 0:
        raise ValueError("singular-value thresholds must be non-negative")
    if min_control_variance_reduction_fraction < 0:
        raise ValueError("control variance threshold must be non-negative")
    if min_innovation_z_score < 0:
        raise ValueError("innovation threshold must be non-negative")
    if min_global_innovation_rms_z < 0:
        raise ValueError("global innovation threshold must be non-negative")
    if not 0.0 <= innovation_false_alarm_probability < 1.0:
        raise ValueError("false-alarm probability must lie in [0, 1)")

    batch_size, num_observers = observer_jacobian.shape[:2]
    device = observer_jacobian.device
    dtype = observer_jacobian.dtype
    prior_std = torch.as_tensor(prior_std, device=device, dtype=dtype)
    prior_std = torch.broadcast_to(prior_std, (batch_size, 6))
    if bool((prior_std <= 0).any()):
        raise ValueError("prior_std must be strictly positive")
    observation_std = torch.as_tensor(
        observation_std_px,
        device=device,
        dtype=dtype,
    )
    observation_std = torch.broadcast_to(
        observation_std,
        (batch_size, num_observers, 2),
    )
    if bool((observation_std <= 0).any()):
        raise ValueError("observation_std_px must be strictly positive")
    if observer_valid is None:
        observer_valid = torch.ones(
            batch_size,
            num_observers,
            device=device,
            dtype=torch.bool,
        )
    else:
        observer_valid = torch.as_tensor(
            observer_valid,
            device=device,
            dtype=torch.bool,
        )
        if observer_valid.shape != (batch_size, num_observers):
            raise ValueError("observer_valid must have shape [B, N]")

    twists = []
    singular_values = []
    retained_modes = []
    effective_ranks = []
    valid_counts = []
    prior_control_variances = []
    posterior_control_variances = []
    reductions_by_mode = []
    modal_innovations = []
    global_innovation_scores = []
    global_innovation_thresholds = []
    for batch_index in range(batch_size):
        finite = (
            torch.isfinite(observer_jacobian[batch_index]).all(dim=(-1, -2))
            & torch.isfinite(observer_innovation_px[batch_index]).all(dim=-1)
        )
        valid = observer_valid[batch_index] & finite
        coordinate_valid = valid[:, None].expand(num_observers, 2).reshape(-1)
        jacobian = observer_jacobian[batch_index].reshape(-1, 6)[coordinate_valid]
        innovation = observer_innovation_px[batch_index].reshape(-1)[
            coordinate_valid
        ]
        noise = observation_std[batch_index].reshape(-1)[coordinate_valid]
        control = control_jacobian[batch_index].reshape(-1, 6)
        scaled_control = control * prior_std[batch_index][None, :]
        prior_variance = scaled_control.square().sum()

        padded_singular = torch.zeros(6, device=device, dtype=dtype)
        padded_retained = torch.zeros(6, device=device, dtype=torch.bool)
        padded_reduction = torch.zeros(6, device=device, dtype=dtype)
        padded_modal_innovation = torch.zeros(6, device=device, dtype=dtype)
        global_innovation_score = torch.zeros((), device=device, dtype=dtype)
        global_innovation_threshold = torch.as_tensor(
            float(min_global_innovation_rms_z),
            device=device,
            dtype=dtype,
        )
        twist = torch.zeros(6, device=device, dtype=dtype)
        posterior_variance = prior_variance
        if jacobian.numel():
            whitened_jacobian = (
                jacobian * prior_std[batch_index][None, :]
            ) / noise[:, None]
            whitened_innovation = innovation / noise
            u, values, vh = torch.linalg.svd(
                whitened_jacobian,
                full_matrices=False,
            )
            num_modes = values.shape[0]
            padded_singular[:num_modes] = values
            maximum = values[0]
            control_sensitivity = scaled_control @ vh.transpose(-1, -2)
            reductions = (
                values.square() / (values.square() + 1.0)
            ) * control_sensitivity.square().sum(dim=0)
            modal_innovation = u.transpose(-1, -2) @ whitened_innovation
            minimum_reduction = (
                float(min_control_variance_reduction_fraction) * prior_variance
            )
            geometry_retained = (
                (values >= float(min_whitened_singular_value))
                & (values >= float(relative_singular_value) * maximum)
                & (reductions >= minimum_reduction)
            )
            if bool(geometry_retained.any()):
                global_innovation_score = (
                    modal_innovation[geometry_retained].square().mean().sqrt()
                )
                if innovation_false_alarm_probability > 0:
                    rank_threshold = chi_square_rms_threshold(
                        int(geometry_retained.sum().item()),
                        innovation_false_alarm_probability,
                    )
                    global_innovation_threshold = torch.maximum(
                        global_innovation_threshold,
                        torch.as_tensor(rank_threshold, device=device, dtype=dtype),
                    )
            retained = (
                geometry_retained
                & (modal_innovation.abs() >= float(min_innovation_z_score))
                & (
                    global_innovation_score
                    >= global_innovation_threshold
                )
            )
            gains = torch.where(
                retained,
                values / (values.square() + 1.0),
                torch.zeros_like(values),
            )
            normalized_twist = vh.transpose(-1, -2) @ (
                gains * modal_innovation
            )
            twist = prior_std[batch_index] * normalized_twist
            padded_retained[:num_modes] = retained
            padded_reduction[:num_modes] = reductions
            padded_modal_innovation[:num_modes] = modal_innovation
            posterior_variance = (
                prior_variance - reductions[retained].sum()
            ).clamp_min(0.0)

        twists.append(twist)
        singular_values.append(padded_singular)
        retained_modes.append(padded_retained)
        effective_ranks.append(padded_retained.sum())
        valid_counts.append(valid.sum())
        prior_control_variances.append(prior_variance)
        posterior_control_variances.append(posterior_variance)
        reductions_by_mode.append(padded_reduction)
        modal_innovations.append(padded_modal_innovation)
        global_innovation_scores.append(global_innovation_score)
        global_innovation_thresholds.append(global_innovation_threshold)

    return WhitenedObserverUpdate(
        twist=torch.stack(twists),
        singular_values=torch.stack(singular_values),
        retained_modes=torch.stack(retained_modes),
        effective_rank=torch.stack(effective_ranks),
        valid_observer_count=torch.stack(valid_counts),
        prior_control_variance=torch.stack(prior_control_variances),
        posterior_control_variance=torch.stack(posterior_control_variances),
        control_variance_reduction_by_mode=torch.stack(reductions_by_mode),
        modal_innovation_z_score=torch.stack(modal_innovations),
        global_innovation_rms_z=torch.stack(global_innovation_scores),
        global_innovation_threshold_rms_z=torch.stack(
            global_innovation_thresholds
        ),
    )


@dataclass(frozen=True)
class ObserverControlTransferOutput:
    """Global-only transfer from observer innovations to held-out anchors."""

    update: WhitenedObserverUpdate
    corrected_extrinsic: torch.Tensor
    observer_nominal_points_px: torch.Tensor
    control_nominal_points_px: torch.Tensor
    control_corrected_points_px: torch.Tensor
    control_depth: torch.Tensor


def solve_observer_control_transfer(
    intrinsic: torch.Tensor,
    nominal_extrinsic: torch.Tensor,
    observer_points_world: torch.Tensor,
    observer_points_px: torch.Tensor,
    control_points_world: torch.Tensor,
    prior_std: torch.Tensor,
    observation_std_px: float | torch.Tensor,
    observer_valid: torch.Tensor | None = None,
    min_whitened_singular_value: float = 0.25,
    relative_singular_value: float = 1e-3,
    min_control_variance_reduction_fraction: float = 1e-6,
    min_innovation_z_score: float = 0.0,
    min_global_innovation_rms_z: float = 0.0,
    innovation_false_alarm_probability: float = 0.0,
) -> ObserverControlTransferOutput:
    """Estimate a shared update without consuming control-anchor observations."""

    observer_nominal, observer_depth, observer_jacobian = (
        projection_jacobian_left_se3(
            intrinsic,
            nominal_extrinsic,
            observer_points_world,
        )
    )
    control_nominal, control_nominal_depth, control_jacobian = (
        projection_jacobian_left_se3(
            intrinsic,
            nominal_extrinsic,
            control_points_world,
        )
    )
    control_jacobian = torch.where(
        (control_nominal_depth > 1e-8)[..., None, None],
        control_jacobian,
        torch.zeros_like(control_jacobian),
    )
    if observer_points_px.shape != observer_nominal.shape:
        raise ValueError("observer_points_px must match projected observer shape")
    depth_valid = observer_depth > 1e-8
    observer_valid = (
        depth_valid
        if observer_valid is None
        else depth_valid
        & observer_valid.to(device=depth_valid.device, dtype=torch.bool)
    )
    update = solve_whitened_observer_update(
        observer_jacobian,
        observer_points_px - observer_nominal,
        control_jacobian,
        prior_std,
        observation_std_px,
        observer_valid,
        min_whitened_singular_value,
        relative_singular_value,
        min_control_variance_reduction_fraction,
        min_innovation_z_score,
        min_global_innovation_rms_z,
        innovation_false_alarm_probability,
    )
    nominal_extrinsic = torch.broadcast_to(
        nominal_extrinsic,
        (*observer_points_world.shape[:-2], 4, 4),
    )
    corrected_extrinsic = se3_exp(update.twist) @ nominal_extrinsic
    control_corrected, control_depth = project_points_torch(
        intrinsic,
        corrected_extrinsic,
        control_points_world,
    )
    return ObserverControlTransferOutput(
        update=update,
        corrected_extrinsic=corrected_extrinsic,
        observer_nominal_points_px=observer_nominal,
        control_nominal_points_px=control_nominal,
        control_corrected_points_px=control_corrected,
        control_depth=control_depth,
    )


@dataclass(frozen=True)
class StructuredProjectionOutput:
    global_transform: torch.Tensor
    global_points_px: torch.Tensor
    corrected_points_px: torch.Tensor
    depth: torch.Tensor


class StructuredKinematicProjection(nn.Module):
    """Apply a shared SE(3) correction followed by optional local residuals."""

    def forward(
        self,
        intrinsic: torch.Tensor,
        nominal_extrinsic: torch.Tensor,
        keypoints_world: torch.Tensor,
        global_twist: torch.Tensor,
        local_residual_px: torch.Tensor | None = None,
    ) -> StructuredProjectionOutput:
        global_transform = se3_exp(global_twist)
        corrected_extrinsic = global_transform @ nominal_extrinsic
        global_points, depth = project_points_torch(
            intrinsic,
            corrected_extrinsic,
            keypoints_world,
        )
        if local_residual_px is None:
            local_residual_px = torch.zeros_like(global_points)
        if local_residual_px.shape != global_points.shape:
            raise ValueError(
                f"local residual shape {tuple(local_residual_px.shape)} does not match "
                f"projected points {tuple(global_points.shape)}"
            )
        return StructuredProjectionOutput(
            global_transform=global_transform,
            global_points_px=global_points,
            corrected_points_px=global_points + local_residual_px,
            depth=depth,
        )
