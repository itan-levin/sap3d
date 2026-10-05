"""The patch tokens a 2D mask covers, against a recorded token set for a fixed synthetic view."""

import numpy as np
from PIL import Image

from sap3d.edit.patches import DINO_GRID_SIZE, mask_patch_tokens


def _case():
    alpha = np.zeros((600, 600), dtype=np.uint8)
    alpha[120:470, 90:400] = 255
    alpha[200:260, 400:520] = 200
    mask = np.zeros((600, 600), dtype=bool)
    mask[150:300, 250:450] = True
    rgba = np.zeros((600, 600, 4), dtype=np.uint8)
    rgba[..., :3] = 128
    rgba[..., 3] = alpha
    return Image.fromarray(rgba), mask


def _tokens(mask, image, coverage=0.5):
    return mask_patch_tokens(mask, image, border_ratio=0.15, frame_size=512, coverage=coverage)


def test_matches_recorded_tokens():
    image, mask = _case()
    tokens, stats = _tokens(mask, image)
    assert tokens == [row * DINO_GRID_SIZE + col + 1 for row in range(8, 19) for col in range(15, 29)]
    assert stats == {"mask_pixels": 30000, "mask_pixels_outside_crop": 0, "patches": 154}


def test_empty_mask_and_coverage():
    image, mask = _case()
    assert _tokens(np.zeros_like(mask), image)[0] == []
    assert len(_tokens(mask, image, coverage=0.1)[0]) > len(_tokens(mask, image, coverage=0.9)[0])


def test_mask_file_and_rejections():
    image, mask = _case()
    as_image = Image.fromarray(mask.astype(np.uint8) * 255)
    assert _tokens(as_image, image)[0] == _tokens(mask, image)[0]
    for bad in (mask[:-1], as_image.convert("RGB"), mask.astype(np.uint8)):
        try:
            _tokens(bad, image)
        except ValueError:
            continue
        raise AssertionError("a malformed mask was accepted")
