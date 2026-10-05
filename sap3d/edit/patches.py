"""The conditioning-image patch tokens a 2D edit mask covers.

The preprocessor crops the conditioning image to its alpha's bounding box and rescales the crop to fill
most of a square frame, so the mask is put through the same crop, rescale and placement before it is
pooled onto DINOv2's 37 x 37 patch grid. Returned tokens are 1-based over the encoder's 1370-token
sequence, whose position 0 stands for the whole image.
"""

import inspect
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image

DINO_IMAGE_SIZE = 518
DINO_PATCH_SIZE = 14
DINO_GRID_SIZE = 37
DINO_NUM_TOKENS = DINO_GRID_SIZE ** 2 + 1

MASK_BINARIZE = 127


class FramePlacement(NamedTuple):
    canvas: int
    crop: tuple   # top, bottom, left, right; bottom and right exclusive
    placed: tuple  # height, width after the rescale
    offset: tuple  # top, left on the canvas


def processor_framing(image_processor) -> tuple:
    """(border ratio, frame size) the pipeline's preprocessor applies."""
    border_ratio = image_processor.border_ratio
    if border_ratio is None:
        border_ratio = inspect.signature(type(image_processor).__call__).parameters["border_ratio"].default
    return float(border_ratio), int(image_processor.size)


def conditioning_alpha(cond_image) -> np.ndarray:
    """The alpha the preprocessor takes its crop box from; an image without alpha counts as opaque."""
    if isinstance(cond_image, Image.Image):
        return np.asarray(cond_image.convert("RGBA"))[..., 3]
    array = cv2.imread(str(cond_image), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise ValueError(f"cannot read the conditioning image at {cond_image}")
    if array.ndim == 3 and array.shape[2] == 4:
        return array[..., 3]
    return np.full(array.shape[:2], 255, dtype=np.uint8)


def frame_placement(alpha: np.ndarray, *, border_ratio: float) -> FramePlacement:
    """The preprocessor's crop box, rescale and centering, with its exact roundings and exclusive bounds."""
    rows, cols = np.nonzero(alpha)
    if rows.size == 0:
        raise ValueError("the conditioning alpha is empty, so there is no crop box")
    top, bottom = int(rows.min()), int(rows.max())
    left, right = int(cols.min()), int(cols.max())
    height, width = bottom - top, right - left
    if height == 0 or width == 0:
        raise ValueError(f"the conditioning silhouette spans {height + 1}x{width + 1} pixels, too thin to crop")
    canvas = int(max(alpha.shape))
    scale = int(canvas * (1 - border_ratio)) / max(height, width)
    placed = (int(height * scale), int(width * scale))
    offset = ((canvas - placed[0]) // 2, (canvas - placed[1]) // 2)
    return FramePlacement(canvas, (top, bottom, left, right), placed, offset)


def in_conditioning_frame(mask: np.ndarray, place: FramePlacement, *, frame_size: int) -> np.ndarray:
    """Per-pixel coverage of `mask` over the square frame the encoder sees, float32 in 0..1."""
    top, bottom, left, right = place.crop
    height, width = place.placed
    offset_y, offset_x = place.offset
    canvas = np.zeros((place.canvas, place.canvas), dtype=np.float32)
    canvas[offset_y:offset_y + height, offset_x:offset_x + width] = cv2.resize(
        mask[top:bottom, left:right].astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
    return cv2.resize(canvas, (frame_size, frame_size), interpolation=cv2.INTER_AREA)


def mask_to_patch_grid(mask: np.ndarray, threshold: float) -> np.ndarray:
    """[37, 37] bool: patches whose mean coverage reaches `threshold`, for a mask already in the frame."""
    m = mask.squeeze().astype(np.float32)
    if np.issubdtype(mask.dtype, np.integer) or m.max() > 1.5:
        m = m / 255.0
    if m.shape != (DINO_IMAGE_SIZE, DINO_IMAGE_SIZE):
        pil = Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8))
        pil = pil.resize((DINO_IMAGE_SIZE, DINO_IMAGE_SIZE), Image.BILINEAR)
        m = np.array(pil).astype(np.float32) / 255.0
    grid = m.reshape(DINO_GRID_SIZE, DINO_PATCH_SIZE, DINO_GRID_SIZE, DINO_PATCH_SIZE).mean(axis=(1, 3))
    return grid >= threshold


def load_mask(mask_image) -> np.ndarray:
    """A 2D mask as booleans, from a single-channel image (file or PIL) or a boolean array."""
    if isinstance(mask_image, np.ndarray):
        if mask_image.dtype != np.bool_:
            raise ValueError(f"a mask array must be boolean, got dtype {mask_image.dtype}")
        array = mask_image
    else:
        image = mask_image if isinstance(mask_image, Image.Image) else Image.open(mask_image)
        if image.mode not in ("L", "1"):
            raise ValueError(f"a mask image must be single-channel, got mode {image.mode!r}")
        array = np.asarray(image)
        if array.dtype != np.bool_:
            array = array > MASK_BINARIZE
    if array.ndim != 2:
        raise ValueError(f"a mask must be two-dimensional, got shape {array.shape}")
    return array


def mask_patch_tokens(mask_image, cond_image, *, border_ratio: float, frame_size: int,
                      coverage: float) -> tuple:
    """(1-based patch tokens covered at least `coverage`, statistics).

    `cond_image` must be the conditioning image exactly as the pipeline receives it, since the crop box
    comes from its alpha.
    """
    alpha = conditioning_alpha(cond_image)
    mask = load_mask(mask_image)
    if mask.shape != alpha.shape:
        raise ValueError(f"the mask is {mask.shape} but the conditioning image is {alpha.shape}; both must "
                         f"be the same view at the same resolution")
    place = frame_placement(alpha, border_ratio=border_ratio)
    grid = mask_to_patch_grid(in_conditioning_frame(mask, place, frame_size=frame_size), coverage)
    tokens = (np.where(grid.ravel())[0] + 1).tolist()
    top, bottom, left, right = place.crop
    total = int(mask.sum())
    return tokens, {"mask_pixels": total,
                    "mask_pixels_outside_crop": total - int(mask[top:bottom, left:right].sum()),
                    "patches": int(grid.sum())}
