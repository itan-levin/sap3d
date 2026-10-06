"""An arbitrary glTF/GLB asset in Edit3D-Bench's layout, so the dataset, edit, texture and eval stages read it
like a benchmark object:

    <out>/source_model/model.glb               the asset, bounding box centered at the origin, longest side 1
    <out>/source_model/render/render_0000-0015.png  the 16 views of the benchmark's renderer
    <out>/source_model/render/meta.json        their cameras, in the benchmark's format

Benchmark objects already have this layout and skip this step.
"""

import json
import math
import tempfile
from pathlib import Path

import numpy as np

from .. import third_party

ELEVATIONS = (15, 30)
AZIMUTHS = tuple(azimuth - 90 for azimuth in (0, 45, 90, 135, 180, 225, 270, 315))
RADIUS = 1.8
SIZE = 1024
LENS_MM, SENSOR_MM = 35.0, 32.0
ORTHO_SCALE = 6.0


def meta(elevations=ELEVATIONS, azimuths=AZIMUTHS) -> dict:
    """The cameras of the scorer's 16-view render, as Edit3D-Bench's `meta.json` records them."""
    from bpyrenderer.camera.layout import get_camera_positions_on_sphere

    _, matrices, elevation_t, azimuth_t = get_camera_positions_on_sphere(
        center=(0, 0, 0), radius=RADIUS, elevations=list(elevations), azimuths=list(azimuths))
    fov_x = float(np.float32(2.0 * math.atan(SENSOR_MM / 2.0 / LENS_MM)))
    locations = [{"index": f"{i:04d}", "projection_type": "PERSP", "ortho_scale": ORTHO_SCALE,
                  "camera_angle_x": fov_x, "elevation": float(elevation), "azimuth": float(azimuth),
                  "transform_matrix": [list(row) for row in matrix]}
                 for i, (matrix, elevation, azimuth) in enumerate(zip(matrices, elevation_t, azimuth_t))]
    return {"width": SIZE, "height": SIZE, "locations": locations}


def prepare(source, out_dir) -> Path:
    from .blender import scene

    out_dir = Path(out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty")
    model_dir = out_dir / "source_model"
    render_dir = model_dir / "render"
    render_dir.mkdir(parents=True)
    glb = model_dir / "model.glb"
    normalization = scene.normalize_glb(source, glb)
    (render_dir / "meta.json").write_text(json.dumps(meta(), indent=1), encoding="utf-8")
    with tempfile.TemporaryDirectory() as scratch:
        third_party.run_module("sap3d.render", [{"glb": str(glb), "out_dir": str(render_dir), "camera": "all"}],
                               Path(scratch) / "render_jobs.json")
    (out_dir / "prepare.json").write_text(json.dumps({"source": str(Path(source).resolve()),
                                                      "normalization": normalization}, indent=1),
                                          encoding="utf-8")
    return out_dir
