"""Renders through the Edit3D-Bench scorer's own Blender setup (`third_party/edit3d_bench/render.py`).

The scorer's `render_images` keyframes its sixteen cameras on consecutive frames and renders the frame
range with its environment map, lens, film and color settings. This module runs that function unchanged:

- all sixteen views (what the scorer compares);
- one of the sixteen, by narrowing the frame range to it, so the image equals that view of a full pass;
- any other camera, by replacing the camera list, the lens and the output size inside a freshly loaded copy
  of the module, the camera being a Blender camera-to-world matrix and a horizontal field of view.

The scorer's file resolves its environment map relative to its own directory, so a process that renders
changes into that directory and stays there; every path handed in must be absolute. Rendering runs in its
own process (`python -m sap3d.render JOBS.json`, or the color fit's child process).

A job file is a JSON list of `{"glb": ..., "out_dir": ..., "camera": CAMERA}`, where CAMERA is
`"all"` (sixteen views), `{"view": k}` (one benchmark view) or `{"transform_matrix": 4x4 Blender
camera-to-world, "camera_angle_x": radians, "size": pixels}`.
"""

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from . import camera as cameras
from . import third_party

N_VIEWS = 16

# The scorer's renderer puts a 35 mm lens on bpyrenderer's default 32 mm sensor.
SENSOR_WIDTH_MM = 32.0

_GPU = None


def camera_spec(camera: cameras.Camera) -> dict:
    """A Stage camera in job-file form: its Blender camera-to-world matrix, field of view and size."""
    blender = cameras.BLENDER_TO_FILE.T @ camera.transform
    return {"transform_matrix": blender.tolist(), "camera_angle_x": camera.fov_x, "size": camera.size}


def lens_mm(fov_x: float) -> float:
    return (SENSOR_WIDTH_MM / 2.0) / math.tan(fov_x / 2.0)


def frame_name(spec) -> str:
    """The file a render of `spec` writes: the view's own frame number, or frame 0 for any other camera."""
    if isinstance(spec, dict) and "view" in spec:
        return f"render_{int(spec['view']):04d}.png"
    return "render_0000.png"


def _enable_gpu() -> str | None:
    import bpy

    prefs = bpy.context.preferences.addons["cycles"].preferences
    for device_type in ("OPTIX", "CUDA"):
        prefs.compute_device_type = device_type
        prefs.get_devices()
        if any(device.type == device_type for device in prefs.devices):
            for device in prefs.devices:
                device.use = device.type == device_type
            return device_type
    return None


def _module():
    """A fresh copy of the scorer's render module, rendering on the GPU when Blender finds one."""
    global _GPU
    import bpy

    directory = third_party.require(third_party.SCORER_DIR, "render.py")
    spec = importlib.util.spec_from_file_location(f"edit3d_bench_render_{os.getpid()}", directory / "render.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if _GPU is None:
        _GPU = _enable_gpu() or "CPU"
    if _GPU != "CPU":
        original = module.init_render_engine

        def init_render_engine(*args, **kwargs):
            original(*args, **kwargs)
            bpy.context.scene.cycles.device = "GPU"

        module.init_render_engine = init_render_engine
    if Path.cwd() != directory:
        os.chdir(directory)
    return module


def render(glb, out_dir, spec) -> Path:
    """Render `glb` into `out_dir` for one camera spec (see the module docstring); returns the PNG for a
    single camera, or `out_dir` for all sixteen views."""
    import bpy

    glb, out_dir = Path(glb), Path(out_dir)
    if not glb.is_absolute() or not out_dir.is_absolute():
        raise ValueError(f"paths must be absolute, got {glb} and {out_dir}")
    if not glb.is_file():
        raise FileNotFoundError(f"no mesh at {glb}")
    module = _module()
    out_dir.mkdir(parents=True, exist_ok=True)

    if spec == "all":
        written = [out_dir / f"render_{view:04d}.png" for view in range(N_VIEWS)]
    else:
        written = [out_dir / frame_name(spec)]
    for path in written:
        path.unlink(missing_ok=True)

    if isinstance(spec, dict) and "view" in spec:
        view = int(spec["view"])
        if not 0 <= view < N_VIEWS:
            raise ValueError(f"a benchmark view is in [0, {N_VIEWS}), got {view}")

        class OneFrameSceneManager(module.SceneManager):
            def render(self):
                bpy.context.scene.frame_start = view
                bpy.context.scene.frame_end = view
                super().render()

        module.SceneManager = OneFrameSceneManager
    elif isinstance(spec, dict):
        matrix = np.asarray(spec["transform_matrix"], dtype=np.float64)
        size = int(spec["size"])
        if matrix.shape != (4, 4) or size < 1:
            raise ValueError(f"a camera needs a 4x4 transform_matrix and a positive size, got {spec}")
        lens = lens_mm(float(spec["camera_angle_x"]))
        add_camera, enable_color_output = module.add_camera, module.enable_color_output
        module.get_camera_positions_on_sphere = lambda **_: ([None], [matrix.tolist()], [None], [None])
        module.add_camera = lambda matrix_, kind, add_frame=False: add_camera(
            matrix_, kind, camera_lens=lens, add_frame=add_frame)
        module.enable_color_output = lambda width, height, *args, **kwargs: enable_color_output(
            size, size, *args, **kwargs)
    elif spec != "all":
        raise ValueError(f"unknown camera spec {spec!r}")

    module.render_images(str(glb), str(out_dir))
    # The scorer's scene clear removes objects only; imported meshes, materials and images would accumulate.
    bpy.data.orphans_purge(do_recursive=True)
    missing = [path.name for path in written if not path.is_file()]
    if missing:
        raise RuntimeError(f"the render produced no {missing} in {out_dir}")
    return written[0] if spec != "all" else out_dir


def run_jobs(jobs: list) -> None:
    for job in jobs:
        render(job["glb"], job["out_dir"], job["camera"])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit("usage: python -m sap3d.render JOBS.json")
    run_jobs(json.loads(Path(argv[0]).read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
