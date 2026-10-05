"""Conditioning of the edited view, with a stand-in background remover."""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from sap3d.config import ConditioningConfig
from sap3d.conditioning import conditioning_image

CFG = ConditioningConfig()


class HalfBlindRemover:
    """Finds the object's left half only, as a segmenter erasing a dark part would."""

    def __call__(self, image):
        rgb = np.asarray(image)
        alpha = np.zeros(rgb.shape[:2], dtype=np.uint8)
        alpha[40:120, 40:80] = 255
        alpha[10:12, 10:12] = 100
        return Image.fromarray(np.dstack([rgb, alpha]))


def _write(folder, rgba: np.ndarray, name: str) -> Path:
    path = Path(folder) / name
    Image.fromarray(rgba).save(path)
    return path


def _scene():
    rgb = np.full((160, 160, 3), 255, dtype=np.uint8)
    rgb[40:120, 40:120] = 30
    rgb[5:9, 150:154] = 0
    return rgb


def test_opaque_image_gets_floored_alpha_inside_its_box():
    with tempfile.TemporaryDirectory() as folder:
        path = _write(folder, np.dstack([_scene(), np.full((160, 160), 255, np.uint8)]), "opaque.png")
        image, name = conditioning_image(path, HalfBlindRemover, CFG)
    alpha = np.asarray(image)[..., 3]
    assert name == "HalfBlindRemover"
    assert (alpha[40:120, 40:120] == 255).all()
    assert alpha[5:9, 150:154].max() == 0
    assert alpha[10:12, 10:12].max() == 0
    assert (np.asarray(image)[..., :3] == _scene()).all()


def test_image_without_alpha_channel_is_treated_as_opaque():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "rgb.png"
        Image.fromarray(_scene()).save(path)
        image, _ = conditioning_image(path, HalfBlindRemover, CFG)
    assert (np.asarray(image)[..., 3][40:120, 40:120] == 255).all()


def test_real_transparency_passes_through_as_a_path():
    def never():
        raise AssertionError("the background remover must not run on an image with real transparency")

    alpha = np.zeros((160, 160), np.uint8)
    alpha[40:120, 40:120] = 255
    with tempfile.TemporaryDirectory() as folder:
        path = _write(folder, np.dstack([_scene(), alpha]), "cut.png")
        image, name = conditioning_image(path, never, CFG)
        assert image == str(path) and name is None
