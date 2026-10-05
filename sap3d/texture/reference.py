"""The image the paint model is conditioned on: the edited view recomposed.

The edited view is cropped to its not-near-white silhouette, scaled (Lanczos) so its longer side fills 90%
of a 512 px frame, and centered on mid-gray, the background the paint model composites its own views onto.
The silhouette uses the conditioning stage's default near-white threshold and speckle guard, fixed here
so that an edit configuration cannot move the crop.
"""

import numpy as np
from PIL import Image

from ..conditioning import flattened_rgb, recovered_alpha_mask
from ..config import ConditioningConfig

FRAME = 512
FILL_FRACTION = 0.90
PAD_LEVEL = 127


def silhouette_box(rgb: np.ndarray) -> tuple:
    """(top, left, bottom, right) of the not-near-white pixels, exclusive at the end."""
    rows, cols = np.nonzero(recovered_alpha_mask(rgb, ConditioningConfig()))
    if rows.size == 0:
        raise ValueError("the edited view carries no not-near-white pixel, so it holds no object to crop to")
    return int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1


def recompose(rgb: np.ndarray) -> np.ndarray:
    top, left, bottom, right = silhouette_box(rgb)
    crop = Image.fromarray(rgb[top:bottom, left:right])
    scale = int(round(FRAME * FILL_FRACTION)) / max(crop.size)
    resized = crop.resize((max(int(round(crop.size[0] * scale)), 1), max(int(round(crop.size[1] * scale)), 1)),
                          Image.Resampling.LANCZOS)
    frame = Image.new("RGB", (FRAME, FRAME), (PAD_LEVEL, PAD_LEVEL, PAD_LEVEL))
    frame.paste(resized, ((FRAME - resized.size[0]) // 2, (FRAME - resized.size[1]) // 2))
    return np.asarray(frame)


def write_reference(edited_view, path) -> None:
    Image.fromarray(recompose(flattened_rgb(edited_view))).save(path, format="PNG", optimize=True)
