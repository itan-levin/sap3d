"""The edited view in the form the shape pipeline must receive it.

The pipeline's preprocessor takes its crop box from every non-zero alpha value and composites the color
over white. An image whose alpha is absent or wholly opaque therefore carries no silhouette, and a
near-white background would reach the model as content. Such an image gets an alpha from a background
remover, raised to opaque wherever the color cannot be background, and zeroed outside the box of
confidently opaque pixels; its background must be near-white. An image with real transparency is passed
on untouched, as its path.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import ConditioningConfig

_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# More than this fraction of the image border darker than the near-white threshold means the background is
# not near-white, and the brightness floor would make the whole frame opaque.
MAX_DARK_BORDER_FRACTION = 0.5


def flattened_rgb(path) -> np.ndarray:
    """The color with any alpha composited onto white."""
    image = Image.open(path)
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.getchannel("A"))
        return np.asarray(flat)
    return np.asarray(image.convert("RGB"))


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32) @ _LUMA_WEIGHTS


def recovered_alpha_mask(rgb: np.ndarray, cfg: ConditioningConfig) -> np.ndarray:
    """Pixels that cannot be background: not near-white, in a connected run of at least the minimum size."""
    not_white = (luminance(rgb) < cfg.near_white_luma).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(not_white, connectivity=cfg.connectivity)
    large = np.zeros(count, dtype=bool)
    large[1:] = stats[1:, cv2.CC_STAT_AREA] >= cfg.min_component_px
    return large[labels]


def segmented_alpha(rgb: np.ndarray, remover) -> np.ndarray:
    return np.asarray(remover(Image.fromarray(rgb)).convert("RGBA"))[..., 3]


def floored_alpha(rgb: np.ndarray, estimated: np.ndarray, cfg: ConditioningConfig) -> np.ndarray:
    """The estimated alpha, raised to opaque on recovered pixels; brightness never removes alpha."""
    return np.maximum(estimated, recovered_alpha_mask(rgb, cfg) * np.uint8(255))


def boxed_alpha(alpha: np.ndarray, cfg: ConditioningConfig) -> np.ndarray:
    """The alpha zeroed outside the inclusive bounding box of the values above the crop threshold."""
    rows, cols = np.nonzero(alpha > cfg.crop_alpha_min)
    if rows.size == 0:
        raise ValueError(f"no alpha above {cfg.crop_alpha_min}: the edited view shows no object")
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(cols.min()), int(cols.max()) + 1
    boxed = np.zeros_like(alpha)
    boxed[top:bottom, left:right] = alpha[top:bottom, left:right]
    return boxed


def check_white_background(rgb: np.ndarray, cfg: ConditioningConfig):
    luma = luminance(rgb)
    border = np.concatenate([luma[0], luma[-1], luma[:, 0], luma[:, -1]])
    dark = float((border < cfg.near_white_luma).mean())
    if dark > MAX_DARK_BORDER_FRACTION:
        raise ValueError(f"{dark:.0%} of the edited view's border is darker than the near-white threshold; an "
                         f"opaque edited view must show the object on a near-white background, or carry an "
                         f"alpha channel with real transparency")


def floored_conditioning(image_path, remover, cfg: ConditioningConfig) -> Image.Image:
    rgb = flattened_rgb(image_path)
    check_white_background(rgb, cfg)
    alpha = floored_alpha(rgb, segmented_alpha(rgb, remover), cfg)
    return Image.fromarray(np.dstack([rgb, boxed_alpha(alpha, cfg)]))


def has_transparency(image_path) -> bool:
    alpha = Image.open(image_path).convert("RGBA").getchannel("A")
    return alpha.getextrema()[0] != 255


def conditioning_image(image_path, make_remover, cfg: ConditioningConfig):
    """(image for the pipeline, segmenter name or None).

    `make_remover` is called only when a background remover is needed and must return a callable taking
    and returning a PIL image.
    """
    if has_transparency(image_path):
        return str(Path(image_path)), None
    remover = make_remover()
    name = type(remover.session).name() if hasattr(remover, "session") else type(remover).__name__
    return floored_conditioning(image_path, remover, cfg), name
