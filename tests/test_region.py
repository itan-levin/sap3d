"""Region primitives, placement in the model frame and the interior sample of the 3D probe."""

import tempfile
from pathlib import Path

import numpy as np
import trimesh

from sap3d import frames
from sap3d.region import global_region, load_region, place_components, points_inside, sample_volume_points

YAML = """primitives:
  - {type: box, center: [0.1, 0.0, 0.0], extents: [0.2, 0.4, 0.6]}
  - {type: sphere, center: [-0.3, 0.1, 0.0], radius: 0.1}
  - {type: cylinder, center: [0.0, 0.3, 0.2], radius: 0.05, height: 0.3, axis: y}
"""


def _load(text: str, suffix: str = ".yaml"):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / f"region{suffix}"
        path.write_text(text)
        return load_region(path)


def test_primitives():
    region = _load(YAML)
    assert len(region.components) == 3
    box, sphere, cylinder = region.components
    assert np.allclose(box.max(axis=0) - box.min(axis=0), [0.2, 0.4, 0.6])
    assert np.allclose(np.linalg.norm(sphere - [-0.3, 0.1, 0.0], axis=1), 0.1)
    assert np.allclose(cylinder.max(axis=0) - cylinder.min(axis=0), [0.1, 0.3, 0.1], atol=1e-3)
    assert region.surface.is_watertight or len(region.surface.split()) == 3


def test_axes_rotation_and_json():
    for axis, extent in (("x", [0.3, 0.1, 0.1]), ("z", [0.1, 0.1, 0.3])):
        region = _load(f"primitives:\n  - {{type: cylinder, center: [0, 0, 0], radius: 0.05, height: 0.3, axis: {axis}}}")
        assert np.allclose(region.surface.extents, extent, atol=1e-3), axis
    turned = _load('{"primitives": [{"type": "box", "center": [0, 0, 0], "extents": [0.4, 0.1, 0.1], '
                   '"rotation_deg": [0, 0, 90]}]}', ".json")
    assert np.allclose(turned.surface.extents, [0.1, 0.4, 0.1])


def test_rejections():
    for text in ("primitives: []", "primitives:\n  - {type: cone, center: [0, 0, 0]}",
                 "primitives:\n  - {type: sphere, center: [0, 0, 0], radius: -0.1}",
                 "primitives:\n  - {type: box, center: [0, 0, 0], extents: [0.1, 0, 0.1]}",
                 "primitives:\n  - {type: sphere, center: [0, 0, 0], radius: 1, color: red}", "boxes: []"):
        try:
            _load(text)
        except ValueError:
            continue
        raise AssertionError(f"accepted {text!r}")


def test_glb_components_load_separately():
    a = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    b = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    b.apply_translation([0.5, 0, 0])
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "region.glb"
        trimesh.Scene([a, b]).export(path)
        region = load_region(path)
    assert len(region.components) == 2 and len(region.surface.faces) == 24


def test_global_region_and_placement():
    source = trimesh.creation.box(extents=[1.0, 0.5, 0.25])
    region = global_region(source)
    assert np.allclose(region.surface.extents, [1.2, 0.7, 0.45])
    frame = frames.frame_of(source)
    placed = place_components(region, frame, 0)
    assert placed[0].dtype == np.float32 and np.isclose(np.abs(placed[0]).max(), 1.2 * frame.factor / 2, rtol=1e-6)
    far = _load("primitives:\n  - {type: sphere, center: [5, 0, 0], radius: 0.1}")
    try:
        place_components(far, frame, 0)
    except ValueError:
        return
    raise AssertionError("a region outside the decoder's cube was accepted")


def test_volume_sample():
    region = _load(YAML)
    components = [c.astype(np.float32) for c in region.components]
    points = sample_volume_points(components, 1000, seed=0)
    assert points.shape == (1000, 3) and points.dtype == np.float32
    assert points_inside(components, points).all()
    assert np.array_equal(points, sample_volume_points(components, 1000, seed=0))
    assert not np.array_equal(points, sample_volume_points(components, 1000, seed=1))
