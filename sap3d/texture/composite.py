"""Merge the painted atlas into the edit region of the source-copied atlas.

Every texel inside the region comes out painted. Outside it, a shell `boundary_margin` thick is painted as
well, then a band `feather_width` wide blends from painted to copied along a smootherstep ramp, so texels
just outside the region do not keep their copied bytes. Texels outside the region whose copied color was
read from a point inside it are painted at full strength, since the copy has no distance test and would
otherwise put the replaced surface's color on the new one.
"""

import numpy as np
from PIL import Image

# Above this Rec. 601 luma a pixel of a benchmark-style 2D image is its near-white field, not object content.
NEAR_WHITE_LUMINANCE = 240.0

# Region-mesh units, against objects spanning about 1 unit.
DEFAULT_BOUNDARY_MARGIN = 0.02
DEFAULT_FEATHER_WIDTH = 0.01

# The blend band may not exceed this fraction of the region's deepest texel.
FEATHER_DEPTH_CLAMP = 0.5

_LUMA = np.array([0.299, 0.587, 0.114])


def _validate_atlas(array, name, shape=None):
    if not isinstance(array, np.ndarray) or array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError(f"{name} must be an (H, W, 3) uint8 array, got {getattr(array, 'dtype', None)} "
                         f"{getattr(array, 'shape', None)}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} does not match {shape}")


def _validate_mask(mask, name, shape=None):
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean array, got {getattr(mask, 'dtype', None)}")
    if shape is not None and mask.shape != shape:
        raise ValueError(f"{name} shape {mask.shape} does not match {shape}")


def to_uint8(values):
    return np.clip(np.rint(values), 0.0, 255.0).astype(np.uint8)


def luminance(rgb):
    return np.asarray(rgb, dtype=np.float64) @ _LUMA


def annotated_pixels(mask_image) -> np.ndarray:
    """(H, W) bool: the nonzero pixels of a mask given as a path, a PIL image or a boolean array."""
    if isinstance(mask_image, np.ndarray):
        _validate_mask(mask_image, "mask")
        return mask_image
    image = mask_image if isinstance(mask_image, Image.Image) else Image.open(mask_image)
    return np.asarray(image.convert("L")) > 0


def edit_region_pixels(edit_image, mask_image) -> tuple:
    """(rgb, eligible): the edited view and its annotated pixels that are not near-white."""
    rgb = np.asarray(Image.open(edit_image).convert("RGB")
                     if not isinstance(edit_image, Image.Image) else edit_image.convert("RGB"))
    selected = annotated_pixels(mask_image)
    if selected.shape != rgb.shape[:2]:
        raise ValueError(f"mask shape {selected.shape} does not match the edited view {rgb.shape[:2]}")
    return rgb, selected & (luminance(rgb) < NEAR_WHITE_LUMINANCE)


def seam_weight(signed_depth, falloff):
    """(N,) float64 smootherstep weight: 1.0 (copied) half a falloff below zero depth, 0.0 half above."""
    distance = np.asarray(signed_depth, dtype=np.float64)
    if distance.ndim != 1 or distance.size == 0 or not np.all(np.isfinite(distance)):
        raise ValueError(f"signed_depth must be a finite non-empty 1-D array, got shape {distance.shape}")
    if not np.isfinite(falloff) or falloff <= 0.0:
        raise ValueError(f"falloff must be finite and positive, got {falloff!r}")
    if falloff > 2.0 * distance.max():
        raise ValueError(f"half of falloff {falloff} exceeds the deepest inside distance {distance.max()}, so no "
                         f"texel would be fully painted")
    position = np.clip(0.5 + distance / falloff, 0.0, 1.0)
    # The quintic leaves [0, 1] by about 2e-15 near either end.
    return np.clip(1.0 - position ** 3 * (position * (6.0 * position - 15.0) + 10.0), 0.0, 1.0)


def feather_seam(inside_rgb, outside_rgb, weight):
    """(N, 3) uint8 blend of the same texels painted (weight 0) and copied (weight 1)."""
    inside = np.asarray(inside_rgb, dtype=np.float64)
    outside = np.asarray(outside_rgb, dtype=np.float64)
    blend = np.asarray(weight, dtype=np.float64)
    if inside.shape != outside.shape or blend.shape != (len(inside),):
        raise ValueError(f"shapes {inside.shape}, {outside.shape} and weight {blend.shape} do not match")
    if not np.all(np.isfinite(blend)) or blend.min() < 0.0 or blend.max() > 1.0:
        raise ValueError(f"weight must lie in [0, 1], got range [{blend.min()}, {blend.max()}]")
    blend = blend[:, None]
    return to_uint8(inside * (1.0 - blend) + outside * blend)


def _validate_depth(depth, gate, region):
    if not isinstance(depth, np.ndarray) or depth.shape != region.shape:
        raise ValueError(f"depth must be an array shaped {region.shape}")
    _validate_mask(gate, "depth_gate", region.shape)
    if (region & ~gate).any():
        raise ValueError(f"{int((region & ~gate).sum())} region texels sit outside depth_gate")
    if not np.isfinite(depth[gate]).all():
        raise ValueError("depth is not finite on some gated texels")
    inside = depth[region]
    if inside.size and float(inside.min()) < 0.0:
        raise ValueError(f"depth reaches {float(inside.min()):.6g} inside the region, where it must be positive")
    outside = depth[gate & ~region]
    if outside.size and float(outside.max()) > 0.0:
        raise ValueError(f"depth reaches {float(outside.max()):.6g} outside the region, where it must be negative")
    if outside.size and not (outside < 0.0).any():
        raise ValueError("no gated texel outside the region carries a negative depth, so the map holds the "
                         "inside depth alone and the band would swallow the whole copied surface")


def composite(source_rgb, painted_rgb, region, *, depth, depth_gate, copied_from_region,
              feather_width=DEFAULT_FEATHER_WIDTH, boundary_margin=DEFAULT_BOUNDARY_MARGIN,
              depth_clamp=FEATHER_DEPTH_CLAMP):
    """(rgb, stats): `painted_rgb` inside `region` and the boundary shell and band, `source_rgb` elsewhere.

    `depth` is `texels.signed_depth` over `depth_gate` (the mapped texels). A region selecting no texel
    leaves the source atlas standing.
    """
    _validate_atlas(source_rgb, "source_rgb")
    _validate_atlas(painted_rgb, "painted_rgb", source_rgb.shape)
    _validate_mask(region, "region", source_rgb.shape[:2])
    _validate_mask(copied_from_region, "copied_from_region", source_rgb.shape[:2])
    if feather_width <= 0.0 or boundary_margin < 0.0:
        raise ValueError(f"feather_width must be positive and boundary_margin non-negative, got "
                         f"{feather_width} and {boundary_margin}")
    _validate_depth(depth, depth_gate, region)
    if (copied_from_region & ~depth_gate).any():
        raise ValueError("copied_from_region selects texels outside every UV chart")

    if not region.any():
        return source_rgb.copy(), {"n_region": 0, "n_outside_band": 0,
                                   "n_copied_from_region": int(copied_from_region.sum()),
                                   "feather_falloff": None, "feather_clamped": None, "n_seam_band": None}

    shifted = depth.astype(np.float64) + boundary_margin
    deepest = float(np.max(shifted[region]))
    if deepest <= 0.0:
        raise ValueError("every region texel sits at zero depth, so no seam ramp exists")
    limit = depth_clamp * deepest
    falloff, clamped = (limit, True) if feather_width > limit else (feather_width, False)
    shifted = shifted + falloff / 2.0
    write = region | (depth_gate & (shifted > -falloff / 2.0)) | copied_from_region

    weight = seam_weight(shifted[write], falloff)
    weight[copied_from_region[write]] = 0.0
    if weight[region[write]].max() != 0.0:
        raise AssertionError(f"a region texel reached blend weight {weight[region[write]].max()}")
    out = source_rgb.copy()
    out[write] = feather_seam(painted_rgb[write], source_rgb[write], weight)
    if np.array_equal(out[region], source_rgb[region]):
        raise ValueError("the composite equals the source copy inside the region; the painted texture never "
                         "reached the atlas")
    return out, {
        "n_region": int(region.sum()),
        "n_outside_band": int(write.sum()) - int(region.sum()),
        "n_copied_from_region": int(copied_from_region.sum()),
        "feather_falloff": round(falloff, 6),
        "feather_clamped": bool(clamped),
        "n_seam_band": int(((weight > 0.0) & (weight < 1.0)).sum()),
        "max_inside_depth": round(float(depth[region].max()), 6),
    }
