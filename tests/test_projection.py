"""Cameras and the projection of a 3D region into an edited view."""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from sap3d import camera as cameras
from sap3d.region import from_primitives, project_mask

# Two cameras exactly as an Edit3D-Bench meta.json records them (elevation and azimuth in radians).
BENCH_CAMERAS = [
    (0.2617993877991494, -1.5707963267948966,
     [[1.0, -1.5848089013523916e-17, 5.914589533937914e-17, 1.0646261822832735e-16],
      [6.123233601181349e-17, 0.2588190734386444, -0.9659258127212524, -1.7386665344238281],
      [-7.190390631581716e-24, 0.9659258127212524, 0.2588190734386444, 0.4658742845058441],
      [0.0, 0.0, 0.0, 1.0]]),
    (0.5235987755982988, 0.7853981633974483,
     [[-0.7071067690849304, -0.35355344414711, 0.6123723983764648, 1.1022703647613525],
      [0.7071067690849304, -0.35355344414711, 0.6123723983764648, 1.1022703647613525],
      [-1.9599385225888e-08, 0.8660253882408142, 0.5000000596046448, 0.8999999761581421],
      [0.0, 0.0, 0.0, 1.0]]),
]
BENCH_FOV_X = 0.8575560450553894  # stored as float32 by the renderer


def test_orbit_reproduces_benchmark_cameras():
    assert abs(cameras.ORBIT_FOV_X - BENCH_FOV_X) < 1e-7
    for elevation, azimuth, matrix in BENCH_CAMERAS:
        built = cameras.orbit_blender_matrix(math.degrees(elevation), math.degrees(azimuth))
        assert np.abs(built - np.asarray(matrix)).max() < 1e-6, (built, matrix)


def test_cameras_look_at_the_origin_with_y_up():
    for elevation in (15.0, 30.0, 60.0):
        for azimuth in (-90.0, 0.0, 135.0):
            camera = cameras.orbit(elevation, azimuth, 64)
            position, back = camera.transform[:3, 3], camera.transform[:3, 2]
            assert abs(position[1] - 1.8 * math.sin(math.radians(elevation))) < 1e-9
            assert np.allclose(back, position / np.linalg.norm(position))
            assert camera.transform[1, 1] > 0


def test_meta_and_single_camera_files_agree():
    elevation, azimuth, matrix = BENCH_CAMERAS[1]
    with tempfile.TemporaryDirectory() as folder:
        meta = Path(folder) / "meta.json"
        meta.write_text(json.dumps({"width": 32, "height": 32, "locations": [
            {"projection_type": "PERSP", "camera_angle_x": BENCH_FOV_X, "transform_matrix": matrix}]}))
        single = Path(folder) / "camera.json"
        single.write_text(json.dumps({"camera_angle_x": BENCH_FOV_X, "transform_matrix": matrix, "size": 32}))
        a, b = cameras.from_meta(meta, 0), cameras.from_json(single)
    c = cameras.orbit(math.degrees(elevation), math.degrees(azimuth), 32)
    assert np.allclose(a.transform, b.transform) and np.abs(a.transform - c.transform).max() < 1e-6


def test_center_ray_passes_through_the_origin():
    camera = cameras.orbit(15.0, -90.0, 63)
    ray = cameras.rays(camera).reshape(63, 63, 6)[31, 31].astype(np.float64)
    origin, direction = ray[:3], ray[3:]
    closest = origin - direction * np.dot(origin, direction)
    assert np.linalg.norm(closest) < 1e-5


def test_image_rows_run_down_and_columns_run_right():
    camera = cameras.orbit(15.0, -90.0, 96)
    assert camera.transform[2, 3] > 1.5 and np.allclose(camera.transform[:3, 0], [1, 0, 0])
    region = from_primitives([{"type": "sphere", "center": [0.3, 0.3, 0.0], "radius": 0.05}], "upper right")
    far = trimesh.creation.icosphere(subdivisions=1, radius=0.01)
    far.apply_translation([0, 0, -5])
    rows, cols = np.nonzero(project_mask(region, far, camera))
    assert len(rows) and rows.max() < 48 and cols.min() > 48


def test_meta_rejections():
    elevation, azimuth, matrix = BENCH_CAMERAS[0]
    for meta, view in (({"width": 32, "height": 16, "locations": [{"camera_angle_x": 0.8, "transform_matrix": matrix}]}, 0),
                       ({"width": 32, "height": 32, "locations": [{"projection_type": "ORTHO", "camera_angle_x": 0.8,
                                                                   "transform_matrix": matrix}]}, 0),
                       ({"width": 32, "height": 32, "locations": [{"camera_angle_x": 0.8, "transform_matrix": matrix}]}, 1)):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "meta.json"
            path.write_text(json.dumps(meta))
            try:
                cameras.from_meta(path, view)
            except ValueError:
                continue
        raise AssertionError(f"accepted {meta} view {view}")


def test_global_region_projects_around_the_whole_source():
    from sap3d.region import global_region
    camera = cameras.orbit(30.0, 45.0, 80)
    source = trimesh.creation.box(extents=[0.6, 1.0, 0.4])
    mask = project_mask(global_region(source), source, camera)
    silhouette = np.isfinite(cameras.first_hit_distance(source, cameras.rays(camera), 80))
    assert silhouette.any() and (mask >= silhouette).all() and mask.sum() > silhouette.sum()


def test_projection_respects_occlusion():
    camera = cameras.orbit(15.0, -90.0, 96)
    region = from_primitives([{"type": "box", "center": [0, 0, 0], "extents": [0.2, 0.2, 0.2]}], "box")
    behind = trimesh.creation.icosphere(subdivisions=3, radius=0.2)
    behind.apply_translation(-np.asarray(camera.transform[:3, 3]) / 1.8 * 0.6)
    enclosing = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    visible = project_mask(region, behind, camera)
    hidden = project_mask(region, enclosing, camera)
    assert visible.sum() > 100 and hidden.sum() == 0
    rows, cols = np.nonzero(visible)
    assert abs(rows.mean() - 47.5) < 2 and abs(cols.mean() - 47.5) < 2
