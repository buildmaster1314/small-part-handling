"""Generation-only local RGB corruptions for visual-trust experiments.

The returned mask and metadata are privileged data-generation annotations.  This
module deliberately does not attach them to model inputs or deployment runtime
state.  Every random choice is made through the caller-owned NumPy generator.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np


CORRUPTION_TYPES = (
    "square_occlusion",
    "irregular_occlusion",
    "texture_distractor",
    "local_gaussian_blur",
)

IntegerSpec: TypeAlias = int | Sequence[int]
FloatSpec: TypeAlias = float | Sequence[float]
CorruptionMetadata: TypeAlias = dict[str, object]
CorruptionResult: TypeAlias = tuple[np.ndarray, np.ndarray, CorruptionMetadata]


def _validate_inputs(
    image: np.ndarray,
    anchor_xy: Sequence[float] | np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    source = np.asarray(image)
    if source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError(f"image must have shape [H, W, 3], got {source.shape}")
    if source.dtype != np.uint8:
        raise ValueError(f"image must have dtype uint8, got {source.dtype}")
    if source.shape[0] == 0 or source.shape[1] == 0:
        raise ValueError("image height and width must be positive")

    anchor = np.asarray(anchor_xy, dtype=np.float64)
    if anchor.shape != (2,):
        raise ValueError(f"anchor_xy must have shape [2], got {anchor.shape}")
    if not bool(np.isfinite(anchor).all()):
        raise ValueError("anchor_xy must contain finite values")

    height, width = source.shape[:2]
    x, y = (float(value) for value in anchor)
    if not (0.0 <= x < width and 0.0 <= y < height):
        raise ValueError("anchor_xy must lie inside the image")
    center_x = int(np.clip(np.rint(x), 0, width - 1))
    center_y = int(np.clip(np.rint(y), 0, height - 1))
    return source, anchor, (center_x, center_y)


def _as_pair(value: object, *, name: str) -> tuple[object, object] | None:
    if isinstance(value, (str, bytes)):
        return None
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return None
    if len(values) != 2:
        raise ValueError(f"{name} range must contain exactly two values")
    return values[0], values[1]


def _sample_integer(
    rng: np.random.Generator,
    value: IntegerSpec,
    *,
    name: str,
    minimum: int,
) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer or inclusive integer range")
    if isinstance(value, (Integral, np.integer)):
        low = high = int(value)
    else:
        pair = _as_pair(value, name=name)
        if pair is None or any(
            isinstance(item, (bool, np.bool_))
            or not isinstance(item, (Integral, np.integer))
            for item in pair
        ):
            raise ValueError(f"{name} must be an integer or inclusive integer range")
        low, high = (int(item) for item in pair)
    if low < minimum or high < low:
        raise ValueError(
            f"{name} must satisfy {minimum} <= lower <= upper, got {(low, high)}"
        )
    if low == high:
        return low
    return int(rng.integers(low, high + 1))


def _sample_float(
    rng: np.random.Generator,
    value: FloatSpec,
    *,
    name: str,
    minimum: float,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a number or numeric range")
    if isinstance(value, (Real, np.integer, np.floating)):
        low = high = float(value)
    else:
        pair = _as_pair(value, name=name)
        if pair is None or any(
            isinstance(item, (bool, np.bool_))
            or not isinstance(item, (Real, np.integer, np.floating))
            for item in pair
        ):
            raise ValueError(f"{name} must be a number or numeric range")
        low, high = (float(item) for item in pair)

    valid_minimum = low >= minimum if minimum_inclusive else low > minimum
    if (
        not np.isfinite([low, high]).all()
        or not valid_minimum
        or high < low
        or (maximum is not None and high > maximum)
    ):
        left = "<=" if minimum_inclusive else "<"
        upper = "" if maximum is None else f" <= {maximum}"
        raise ValueError(
            f"{name} must satisfy {minimum} {left} lower <= upper{upper}, "
            f"got {(low, high)}"
        )
    if low == high:
        return low
    return float(rng.uniform(low, high))


def _resolve_rgb(
    rng: np.random.Generator,
    value: int | Sequence[int] | np.ndarray | None,
    *,
    name: str,
) -> np.ndarray:
    if value is None:
        return rng.integers(0, 256, size=3, dtype=np.uint8)
    colors = np.asarray(value)
    if colors.ndim == 0:
        colors = np.repeat(colors, 3)
    if colors.shape != (3,) or colors.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer or three RGB integers")
    if bool(np.any(colors < 0)) or bool(np.any(colors > 255)):
        raise ValueError(f"{name} values must lie in [0, 255]")
    return colors.astype(np.uint8)


def _resolve_texture_colors(
    rng: np.random.Generator,
    colors_rgb: Sequence[Sequence[int]] | np.ndarray | None,
) -> np.ndarray:
    if colors_rgb is None:
        first = rng.integers(0, 256, size=3, dtype=np.int16)
        offset = rng.integers(64, 193, size=3, dtype=np.int16)
        second = (first + offset) % 256
        return np.stack((first, second)).astype(np.uint8)

    colors = np.asarray(colors_rgb)
    if colors.shape != (2, 3) or colors.dtype.kind not in "iu":
        raise ValueError("colors_rgb must contain two RGB integer triplets")
    if bool(np.any(colors < 0)) or bool(np.any(colors > 255)):
        raise ValueError("colors_rgb values must lie in [0, 255]")
    if bool(np.array_equal(colors[0], colors[1])):
        raise ValueError("colors_rgb entries must differ")
    return colors.astype(np.uint8)


def _square_bounds(
    center_xy: tuple[int, int],
    radius_px: int,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    center_x, center_y = center_xy
    height, width = image_shape
    return (
        max(0, center_x - radius_px),
        max(0, center_y - radius_px),
        min(width, center_x + radius_px + 1),
        min(height, center_y + radius_px + 1),
    )


def _disk_mask(
    center_xy: tuple[int, int],
    radius_px: int,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x0, y0, x1, y1 = _square_bounds(center_xy, radius_px, image_shape)
    center_x, center_y = center_xy
    yy, xx = np.ogrid[y0:y1, x0:x1]
    local = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius_px**2
    mask = np.zeros(image_shape, dtype=np.bool_)
    mask[y0:y1, x0:x1] = local
    return mask, (x0, y0, x1, y1)


def _polygon_mask(
    vertices_xy: np.ndarray,
    center_xy: tuple[int, int],
    radius_px: int,
    image_shape: tuple[int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = _square_bounds(center_xy, radius_px, image_shape)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = np.zeros(xx.shape, dtype=np.bool_)
    previous = vertices_xy[-1]
    for current in vertices_xy:
        x_i, y_i = (float(value) for value in current)
        x_j, y_j = (float(value) for value in previous)
        if y_i != y_j:
            crosses = (y_i > yy) != (y_j > yy)
            edge_x = (x_j - x_i) * (yy - y_i) / (y_j - y_i) + x_i
            inside ^= crosses & (xx < edge_x)
        previous = current

    mask = np.zeros(image_shape, dtype=np.bool_)
    mask[y0:y1, x0:x1] = inside
    center_x, center_y = center_xy
    mask[center_y, center_x] = True
    return mask


def _metadata(
    corruption_type: str,
    anchor_xy: np.ndarray,
    center_xy: tuple[int, int],
    mask: np.ndarray,
    **parameters: object,
) -> CorruptionMetadata:
    rows, columns = np.nonzero(mask)
    bbox = [
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    ]
    return {
        "corruption_type": corruption_type,
        "anchor_xy": [float(anchor_xy[0]), float(anchor_xy[1])],
        "center_xy": [int(center_xy[0]), int(center_xy[1])],
        "affected_bbox_xyxy": bbox,
        "affected_pixel_count": int(mask.sum()),
        "mask_semantics": "intervention_support",
        **parameters,
    }


def apply_square_occlusion(
    image: np.ndarray,
    anchor_xy: Sequence[float] | np.ndarray,
    rng: np.random.Generator,
    *,
    radius_px: IntegerSpec = (6, 16),
    fill_rgb: int | Sequence[int] | np.ndarray | None = None,
) -> CorruptionResult:
    """Apply a flat square occluder centered on ``anchor_xy``."""

    source, anchor, center = _validate_inputs(image, anchor_xy, rng)
    radius = _sample_integer(rng, radius_px, name="radius_px", minimum=0)
    color = _resolve_rgb(rng, fill_rgb, name="fill_rgb")
    x0, y0, x1, y1 = _square_bounds(center, radius, source.shape[:2])

    mask = np.zeros(source.shape[:2], dtype=np.bool_)
    mask[y0:y1, x0:x1] = True
    result = np.array(source, copy=True, order="C")
    result[mask] = color
    metadata = _metadata(
        "square_occlusion",
        anchor,
        center,
        mask,
        radius_px=radius,
        fill_rgb=[int(value) for value in color],
    )
    return result, mask, metadata


def apply_irregular_occlusion(
    image: np.ndarray,
    anchor_xy: Sequence[float] | np.ndarray,
    rng: np.random.Generator,
    *,
    radius_px: IntegerSpec = (8, 18),
    vertex_count: IntegerSpec = (7, 12),
    fill_rgb: int | Sequence[int] | np.ndarray | None = None,
) -> CorruptionResult:
    """Apply a randomly shaped star-convex polygon occluder around an anchor."""

    source, anchor, center = _validate_inputs(image, anchor_xy, rng)
    radius = _sample_integer(rng, radius_px, name="radius_px", minimum=0)
    vertices_count = _sample_integer(
        rng, vertex_count, name="vertex_count", minimum=5
    )
    color = _resolve_rgb(rng, fill_rgb, name="fill_rgb")

    rotation = float(rng.uniform(0.0, 2.0 * np.pi))
    angle_step = 2.0 * np.pi / vertices_count
    angle_jitter = rng.uniform(
        -0.22 * angle_step,
        0.22 * angle_step,
        size=vertices_count,
    )
    angles = rotation + np.arange(vertices_count) * angle_step + angle_jitter
    radial_scales = rng.uniform(0.55, 1.0, size=vertices_count)
    center_x, center_y = center
    vertices = np.column_stack(
        (
            center_x + radius * radial_scales * np.cos(angles),
            center_y + radius * radial_scales * np.sin(angles),
        )
    )
    mask = _polygon_mask(vertices, center, radius, source.shape[:2])

    result = np.array(source, copy=True, order="C")
    result[mask] = color
    metadata = _metadata(
        "irregular_occlusion",
        anchor,
        center,
        mask,
        radius_px=radius,
        vertex_count=vertices_count,
        fill_rgb=[int(value) for value in color],
        vertices_xy=vertices.tolist(),
    )
    return result, mask, metadata


def apply_texture_distractor(
    image: np.ndarray,
    anchor_xy: Sequence[float] | np.ndarray,
    rng: np.random.Generator,
    *,
    radius_px: IntegerSpec = (8, 18),
    tile_size_px: IntegerSpec = (2, 5),
    opacity: FloatSpec = (0.65, 0.95),
    colors_rgb: Sequence[Sequence[int]] | np.ndarray | None = None,
) -> CorruptionResult:
    """Overlay a randomly oriented two-color texture in an anchor-local disk."""

    source, anchor, center = _validate_inputs(image, anchor_xy, rng)
    radius = _sample_integer(rng, radius_px, name="radius_px", minimum=0)
    tile_size = _sample_integer(
        rng, tile_size_px, name="tile_size_px", minimum=1
    )
    alpha = _sample_float(
        rng,
        opacity,
        name="opacity",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )
    colors = _resolve_texture_colors(rng, colors_rgb)
    angle = float(rng.uniform(0.0, np.pi))
    phase = rng.uniform(0.0, float(tile_size), size=2)

    mask, (x0, y0, x1, y1) = _disk_mask(center, radius, source.shape[:2])
    center_x, center_y = center
    yy, xx = np.mgrid[y0:y1, x0:x1]
    delta_x = xx - center_x
    delta_y = yy - center_y
    cosine = np.cos(angle)
    sine = np.sin(angle)
    u = cosine * delta_x + sine * delta_y
    v = -sine * delta_x + cosine * delta_y
    cells = (
        np.floor((u + phase[0]) / tile_size).astype(np.int64)
        + np.floor((v + phase[1]) / tile_size).astype(np.int64)
    ) & 1
    texture = colors[cells]

    source_patch = source[y0:y1, x0:x1].astype(np.float32)
    blended = np.rint((1.0 - alpha) * source_patch + alpha * texture).astype(
        np.uint8
    )
    result = np.array(source, copy=True, order="C")
    local_mask = mask[y0:y1, x0:x1]
    result_patch = result[y0:y1, x0:x1]
    result_patch[local_mask] = blended[local_mask]
    metadata = _metadata(
        "texture_distractor",
        anchor,
        center,
        mask,
        radius_px=radius,
        tile_size_px=tile_size,
        opacity=alpha,
        angle_radians=angle,
        phase_px=[float(value) for value in phase],
        colors_rgb=colors.astype(int).tolist(),
    )
    return result, mask, metadata


def _gaussian_blur_bbox(
    image: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    sigma: float,
) -> tuple[np.ndarray, int]:
    kernel_radius = max(1, int(np.ceil(3.0 * sigma)))
    coordinates = np.arange(-kernel_radius, kernel_radius + 1, dtype=np.float32)
    kernel = np.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()

    x0, y0, x1, y1 = bbox_xyxy
    height, width = image.shape[:2]
    source_rows = np.clip(
        np.arange(y0 - kernel_radius, y1 + kernel_radius), 0, height - 1
    )
    source_columns = np.clip(
        np.arange(x0 - kernel_radius, x1 + kernel_radius), 0, width - 1
    )
    patch = image[source_rows[:, None], source_columns[None, :]].astype(np.float32)

    output_width = x1 - x0
    horizontal = np.zeros(
        (patch.shape[0], output_width, image.shape[2]), dtype=np.float32
    )
    for offset, weight in enumerate(kernel):
        horizontal += float(weight) * patch[:, offset : offset + output_width]

    output_height = y1 - y0
    blurred = np.zeros(
        (output_height, output_width, image.shape[2]), dtype=np.float32
    )
    for offset, weight in enumerate(kernel):
        blurred += float(weight) * horizontal[offset : offset + output_height]
    return np.clip(np.rint(blurred), 0.0, 255.0).astype(np.uint8), kernel_radius


def apply_local_gaussian_blur(
    image: np.ndarray,
    anchor_xy: Sequence[float] | np.ndarray,
    rng: np.random.Generator,
    *,
    radius_px: IntegerSpec = (8, 18),
    sigma: FloatSpec = (0.8, 2.2),
) -> CorruptionResult:
    """Apply Gaussian blur only inside an anchor-local disk."""

    source, anchor, center = _validate_inputs(image, anchor_xy, rng)
    radius = _sample_integer(rng, radius_px, name="radius_px", minimum=0)
    sampled_sigma = _sample_float(
        rng,
        sigma,
        name="sigma",
        minimum=0.0,
        minimum_inclusive=False,
    )
    mask, bbox = _disk_mask(center, radius, source.shape[:2])
    blurred, kernel_radius = _gaussian_blur_bbox(source, bbox, sampled_sigma)

    x0, y0, x1, y1 = bbox
    result = np.array(source, copy=True, order="C")
    local_mask = mask[y0:y1, x0:x1]
    result_patch = result[y0:y1, x0:x1]
    result_patch[local_mask] = blurred[local_mask]
    metadata = _metadata(
        "local_gaussian_blur",
        anchor,
        center,
        mask,
        radius_px=radius,
        sigma=sampled_sigma,
        kernel_radius_px=kernel_radius,
    )
    return result, mask, metadata


_TYPE_ALIASES = {
    "square": "square_occlusion",
    "irregular": "irregular_occlusion",
    "texture": "texture_distractor",
    "blur": "local_gaussian_blur",
}


def apply_visual_trust_corruption(
    image: np.ndarray,
    anchor_xy: Sequence[float] | np.ndarray,
    rng: np.random.Generator,
    *,
    corruption_type: str | None = None,
    **parameters: object,
) -> CorruptionResult:
    """Apply one requested corruption, or sample its type with ``rng``.

    Type labels exist only in the returned generation metadata.  Callers must
    keep that metadata out of deployment observations and model inputs.
    """

    _validate_inputs(image, anchor_xy, rng)
    if corruption_type is None:
        corruption_type = CORRUPTION_TYPES[
            int(rng.integers(0, len(CORRUPTION_TYPES)))
        ]
    canonical_type = _TYPE_ALIASES.get(corruption_type, corruption_type)
    functions = {
        "square_occlusion": apply_square_occlusion,
        "irregular_occlusion": apply_irregular_occlusion,
        "texture_distractor": apply_texture_distractor,
        "local_gaussian_blur": apply_local_gaussian_blur,
    }
    try:
        function = functions[canonical_type]
    except KeyError as exc:
        expected = ", ".join(CORRUPTION_TYPES)
        raise ValueError(
            f"unknown corruption_type {corruption_type!r}; expected one of {expected}"
        ) from exc
    return function(image, anchor_xy, rng, **parameters)


# Short names are convenient in dataset-generation loops while retaining the
# explicit ``apply_`` names used elsewhere in FCPG.
square_occlusion = apply_square_occlusion
irregular_occlusion = apply_irregular_occlusion
texture_distractor = apply_texture_distractor
local_gaussian_blur = apply_local_gaussian_blur


__all__ = [
    "CORRUPTION_TYPES",
    "CorruptionMetadata",
    "CorruptionResult",
    "apply_irregular_occlusion",
    "apply_local_gaussian_blur",
    "apply_square_occlusion",
    "apply_texture_distractor",
    "apply_visual_trust_corruption",
    "irregular_occlusion",
    "local_gaussian_blur",
    "square_occlusion",
    "texture_distractor",
]
