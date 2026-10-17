"""Deterministic, model-free helpers for LeWM latent diagnostic plots.

The functions in this module prepare numeric plot inputs only.  In particular,
PCA state is fitted exclusively from the caller-provided reference latents;
predictions and targets are transformed with that frozen state.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np


def _finite_float_array(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0 or any(size <= 0 for size in array.shape):
        raise ValueError(f"{name} must be a non-empty array")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True, slots=True)
class ReferencePCA:
    """A PCA coordinate system fitted from reference latents only."""

    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray

    @property
    def latent_dim(self) -> int:
        return int(self.components.shape[1])

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    def transform(self, latents: object) -> np.ndarray:
        """Project latents while retaining every leading input axis."""

        values = _finite_float_array(latents, name="latents")
        if values.shape[-1] != self.latent_dim:
            raise ValueError(
                "latents final dimension must match the fitted PCA dimension "
                f"{self.latent_dim}, got {values.shape[-1]}"
            )
        return (values - self.mean) @ self.components.T


def fit_reference_pca(
    reference_latents: object,
    *,
    n_components: int = 2,
) -> ReferencePCA:
    """Fit deterministic PCA on ``reference_latents`` and no other data.

    All axes except the final latent dimension are treated as reference sample
    axes.  Component signs are canonicalized by making each component's
    largest-magnitude loading positive, avoiding arbitrary sign flips in plots.
    """

    reference = _finite_float_array(
        reference_latents, name="reference_latents"
    )
    if reference.ndim < 2:
        raise ValueError("reference_latents must have a sample and latent axis")
    if isinstance(n_components, bool) or not isinstance(
        n_components, (Integral, np.integer)
    ):
        raise TypeError("n_components must be an integer")

    samples = reference.reshape(-1, reference.shape[-1])
    component_count = int(n_components)
    maximum = min(samples.shape)
    if not 1 <= component_count <= maximum:
        raise ValueError(
            f"n_components must lie in [1, {maximum}], got {component_count}"
        )

    mean = samples.mean(axis=0, dtype=np.float64)
    centered = samples - mean
    _, singular_values, right_vectors = np.linalg.svd(
        centered, full_matrices=False
    )
    components = np.array(right_vectors[:component_count], copy=True)

    pivot = np.argmax(np.abs(components), axis=1)
    signs = np.where(components[np.arange(component_count), pivot] < 0.0, -1.0, 1.0)
    components *= signs[:, None]

    squared_singular_values = np.square(singular_values, dtype=np.float64)
    total = float(squared_singular_values.sum())
    if total > 0.0:
        explained = squared_singular_values[:component_count] / total
    else:
        explained = np.zeros(component_count, dtype=np.float64)

    return ReferencePCA(
        mean=np.array(mean, copy=True),
        components=components,
        explained_variance_ratio=np.array(explained, copy=True),
    )


def project_latents_from_reference(
    reference_latents: object,
    *latent_sets: object,
    n_components: int = 2,
) -> tuple[ReferencePCA, tuple[np.ndarray, ...]]:
    """Fit a reference PCA and project each supplied diagnostic latent set."""

    projection = fit_reference_pca(
        reference_latents, n_components=n_components
    )
    return projection, tuple(projection.transform(values) for values in latent_sets)


def per_anchor_normalized_errors(
    predicted_latents: object,
    target_latents: object,
    reference_latents: object,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Return unitless per-item, per-anchor latent prediction errors.

    Inputs use ``[..., K, D]`` layout.  The numerator is latent-dimension RMS
    error.  Each anchor's denominator is its RMS spread around its mean in the
    supplied ``reference_latents``.  Reference sample axes are all axes before
    ``[K, D]``; they are reduced independently for every anchor.
    """

    predicted = _finite_float_array(
        predicted_latents, name="predicted_latents"
    )
    target = _finite_float_array(target_latents, name="target_latents")
    reference = _finite_float_array(
        reference_latents, name="reference_latents"
    )
    if predicted.ndim < 2 or predicted.shape != target.shape:
        raise ValueError(
            "predicted_latents and target_latents must share shape [..., K, D]"
        )
    if reference.ndim < 3:
        raise ValueError("reference_latents must have shape [..., K, D]")
    if reference.shape[-2:] != predicted.shape[-2:]:
        raise ValueError(
            "reference_latents anchor and latent dimensions must match predictions"
        )
    if not np.isscalar(epsilon) or not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be a finite positive scalar")

    reference_sample_axes = tuple(range(reference.ndim - 2))
    reference_mean = reference.mean(
        axis=reference_sample_axes, keepdims=True, dtype=np.float64
    )
    scale_axes = (*reference_sample_axes, reference.ndim - 1)
    scale = np.sqrt(
        np.mean(
            np.square(reference - reference_mean),
            axis=scale_axes,
            dtype=np.float64,
        )
    )
    error = np.sqrt(
        np.mean(
            np.square(predicted - target), axis=-1, dtype=np.float64
        )
    )
    return error / np.maximum(scale, float(epsilon))


@dataclass(frozen=True, slots=True)
class NearestNeighborResult:
    """Per-anchor nearest reference rows, ordered by distance then row index."""

    indices: np.ndarray
    distances: np.ndarray
    labels: np.ndarray | None


def per_anchor_nearest_neighbors(
    query_latents: object,
    reference_latents: object,
    *,
    reference_labels: Sequence[object] | np.ndarray | None = None,
    k: int = 1,
) -> NearestNeighborResult:
    """Find reference samples nearest to each query anchor in latent space.

    ``reference_latents`` has shape ``[R, K, D]`` and queries have shape
    ``[..., K, D]``.  Results have shape ``[..., K, k]``.  Stable sorting makes
    the lower reference index win exact distance ties; labels are indexed from
    the same reference rows.
    """

    query = _finite_float_array(query_latents, name="query_latents")
    reference = _finite_float_array(
        reference_latents, name="reference_latents"
    )
    if reference.ndim != 3:
        raise ValueError("reference_latents must have shape [R, K, D]")
    if query.ndim < 2 or query.shape[-2:] != reference.shape[-2:]:
        raise ValueError(
            "query_latents must have shape [..., K, D] matching the reference"
        )
    if isinstance(k, bool) or not isinstance(k, (Integral, np.integer)):
        raise TypeError("k must be an integer")
    neighbor_count = int(k)
    if not 1 <= neighbor_count <= reference.shape[0]:
        raise ValueError(
            f"k must lie in [1, {reference.shape[0]}], got {neighbor_count}"
        )

    query_shape = query.shape[:-2]
    anchors, latent_dim = reference.shape[-2:]
    flat_query = query.reshape(-1, anchors, latent_dim)
    differences = flat_query[:, :, None, :] - reference.transpose(1, 0, 2)[None]
    squared_distances = np.einsum(
        "qkrd,qkrd->qkr", differences, differences, optimize=True
    )
    order = np.argsort(squared_distances, axis=-1, kind="stable")
    indices = order[..., :neighbor_count]
    distances = np.sqrt(np.take_along_axis(squared_distances, indices, axis=-1))

    output_shape = (*query_shape, anchors, neighbor_count)
    output_indices = indices.reshape(output_shape).astype(np.int64, copy=False)
    output_labels: np.ndarray | None = None
    if reference_labels is not None:
        labels = np.asarray(reference_labels)
        if labels.ndim != 1 or labels.shape[0] != reference.shape[0]:
            raise ValueError("reference_labels must have shape [R]")
        output_labels = labels[output_indices]

    return NearestNeighborResult(
        indices=output_indices,
        distances=distances.reshape(output_shape),
        labels=output_labels,
    )


def _image_height_width(image_shape: object) -> tuple[int, int]:
    shape = image_shape.shape if isinstance(image_shape, np.ndarray) else image_shape
    if isinstance(shape, (str, bytes)):
        raise TypeError("image_shape must be an image or shape sequence")
    try:
        dimensions = tuple(shape)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("image_shape must be an image or shape sequence") from error
    if len(dimensions) < 2 or any(
        isinstance(value, bool)
        or not isinstance(value, (Integral, np.integer))
        for value in dimensions[:2]
    ):
        raise ValueError("image_shape must begin with integer [height, width]")
    height, width = (int(value) for value in dimensions[:2])
    if height <= 0 or width <= 0:
        raise ValueError("image height and width must be positive")
    return height, width


def image_overlay_extent(image_shape: object) -> tuple[float, float, float, float]:
    """Return ``(left, right, bottom, top)`` for upper-origin pixel centers."""

    height, width = _image_height_width(image_shape)
    return (-0.5, width - 0.5, height - 0.5, -0.5)


def image_xy_to_row_col(
    points_xy: object,
    image_shape: object,
    *,
    clip: bool = False,
) -> np.ndarray:
    """Map image ``(x, y)`` points to integer NumPy ``(row, column)`` indices."""

    points = _finite_float_array(points_xy, name="points_xy")
    if points.shape[-1] != 2:
        raise ValueError("points_xy must end with an (x, y) coordinate axis")
    height, width = _image_height_width(image_shape)
    converted = np.rint(points)
    if clip:
        converted[..., 0] = np.clip(converted[..., 0], 0, width - 1)
        converted[..., 1] = np.clip(converted[..., 1], 0, height - 1)
    elif bool(
        np.any(converted[..., 0] < 0)
        or np.any(converted[..., 0] >= width)
        or np.any(converted[..., 1] < 0)
        or np.any(converted[..., 1] >= height)
    ):
        raise ValueError("rounded points_xy must lie inside the image")
    return np.stack((converted[..., 1], converted[..., 0]), axis=-1).astype(
        np.int64
    )


@dataclass(frozen=True, slots=True)
class ImageOverlay:
    """Coordinates and image settings suitable for an upper-origin overlay."""

    points_xy: np.ndarray
    row_col: np.ndarray
    extent: tuple[float, float, float, float]
    origin: str = "upper"


def prepare_image_overlay(
    points_xy: object,
    image_shape: object,
    *,
    clip: bool = False,
) -> ImageOverlay:
    """Prepare consistent scatter and array-index coordinates for an image."""

    points = _finite_float_array(points_xy, name="points_xy")
    row_col = image_xy_to_row_col(points, image_shape, clip=clip)
    if clip:
        height, width = _image_height_width(image_shape)
        points = np.array(points, copy=True)
        points[..., 0] = np.clip(points[..., 0], 0.0, width - 1.0)
        points[..., 1] = np.clip(points[..., 1], 0.0, height - 1.0)
    return ImageOverlay(
        points_xy=np.array(points, copy=True),
        row_col=row_col,
        extent=image_overlay_extent(image_shape),
    )


__all__ = [
    "ImageOverlay",
    "NearestNeighborResult",
    "ReferencePCA",
    "fit_reference_pca",
    "image_overlay_extent",
    "image_xy_to_row_col",
    "per_anchor_nearest_neighbors",
    "per_anchor_normalized_errors",
    "prepare_image_overlay",
    "project_latents_from_reference",
]
