"""hy3dpaint, Tencent's multiview texture painter, wrapped for one process that paints many meshes.

hy3dpaint imports its own top-level `utils`, `DifferentiableRenderer` and `hunyuanpaintpbr` packages, so its
directory goes on `sys.path` in this process only; no other stage imports anything from it by name.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import torch  # noqa: F401  (must be imported before custom_rasterizer, which hy3dpaint imports)

from .. import third_party

# Higher values are unusable: resolution 768 leaves most views unresized before baking, and max_num_view 9
# can return 6 views.
MAX_NUM_VIEW = 6
RESOLUTION = 512
USE_REMESH = True

DEFAULT_REALESRGAN = third_party.HY3DPAINT_DIR / "ckpt" / "RealESRGAN_x4plus.pth"


def _enable_imports() -> Path:
    paint_dir = third_party.require(third_party.HY3DPAINT_DIR, "textureGenPipeline.py")
    if str(paint_dir) not in sys.path:
        sys.path.insert(0, str(paint_dir))
    return paint_dir


def _check_compiled():
    import importlib.util

    import bpy  # noqa: F401  (hy3dpaint's GLB conversion runs through Blender)
    # The vendored custom_rasterizer source folder imports as an empty package, so the compiled kernel is checked.
    import custom_rasterizer_kernel  # noqa: F401

    painter = list((third_party.HY3DPAINT_DIR / "DifferentiableRenderer").glob("mesh_inpaint_processor*.so"))
    if not painter:
        raise FileNotFoundError("the mesh painter is not compiled into third_party/hy3dpaint/"
                                "DifferentiableRenderer; run scripts/build_extensions.sh")
    if importlib.util.find_spec("DifferentiableRenderer.mesh_inpaint_processor") is None:
        raise ImportError("DifferentiableRenderer.mesh_inpaint_processor is not importable")


def load_pipeline(realesrgan_ckpt=None, model: str = "tencent/Hunyuan3D-2.1"):
    """The paint pipeline with absolute config paths. Weights download from Hugging Face on first use."""
    paint_dir = _enable_imports()
    _check_compiled()
    from utils.torchvision_fix import apply_fix

    apply_fix()
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    ckpt = Path(realesrgan_ckpt) if realesrgan_ckpt is not None else DEFAULT_REALESRGAN
    if not ckpt.is_file():
        raise FileNotFoundError(f"no RealESRGAN checkpoint at {ckpt} (RealESRGAN_x4plus.pth from the Real-ESRGAN "
                                f"v0.1.0 release)")
    conf = Hunyuan3DPaintConfig(max_num_view=MAX_NUM_VIEW, resolution=RESOLUTION)
    conf.realesrgan_ckpt_path = str(ckpt.resolve())
    conf.multiview_cfg_path = str(paint_dir / "cfgs" / "hunyuan-paint-pbr.yaml")
    conf.multiview_pretrained_path = model
    return Hunyuan3DPaintPipeline(conf)


def paint(pipeline, mesh_glb_bytes: bytes, reference_png: Path, work_dir: Path) -> Path:
    """Paint one mesh into `work_dir` and return the painted GLB.

    hy3dpaint writes its remesh intermediate beside its input, so the input is staged in `work_dir`, which
    is emptied first: hy3dpaint never removes an earlier output. Its OBJ-to-GLB conversion goes through
    Blender, whose context exists only in a process's main thread, so it runs in a child process and this
    function can be called from any thread.
    """
    work_dir = Path(work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    staged = work_dir / "input.glb"
    staged.write_bytes(mesh_glb_bytes)
    painted_obj = work_dir / "textured_mesh.obj"
    pipeline(mesh_path=str(staged), image_path=str(reference_png), output_mesh_path=str(painted_obj),
             use_remesh=USE_REMESH, save_glb=False)
    painted = work_dir / "textured_mesh.glb"
    subprocess.run([sys.executable, "-m", "sap3d.texture.paint", str(painted_obj.resolve()), str(painted.resolve())],
                   env=third_party.child_env(), check=True)
    return painted


def convert_to_glb(obj_path, glb_path) -> None:
    """hy3dpaint's own OBJ-to-GLB conversion (Blender), in this process's main thread."""
    _enable_imports()
    from DifferentiableRenderer.mesh_utils import convert_obj_to_glb

    if not convert_obj_to_glb(str(obj_path), str(glb_path)) or not Path(glb_path).is_file():
        raise RuntimeError(f"Blender could not convert {obj_path} to {glb_path}; hy3dpaint's conversion swallows "
                           f"the error")


if __name__ == "__main__":
    convert_to_glb(*sys.argv[1:3])
