"""sap3d-texture: texture an edited mesh.

Benchmark case (source, edited view, region, mask and camera read from the case folder):
    sap3d-texture --bench-root DATA --case GSO/SORTING_BUS/prompt_2 --mesh EDIT/edit.glb --out OUT

Any asset:
    sap3d-texture --mesh edit.glb --source model.glb --edited-view edit.png --region region.yaml \\
                  --camera-meta meta.json --view 0 [--mask mask.png] --out OUT
  The camera (--camera-meta/--view, --camera or --orbit) is the edited view's; the color fit renders
  through it, and the 2D mask is projected from the region when --mask is not given.

--mode generated (global edits) publishes the painted texture alone and reads no region; a camera is
then optional and only recorded for the eval stage. The color fit runs in composite mode unless the
texture configuration turns it off.

Writes OUT/composite.glb (and OUT/composite_color_fit.glb) or OUT/generated.glb, plus OUT/texture.json.
Needs a source checkout of this repository: the vendored paint model is used by path.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .. import bench, frames
from .. import config as config_module
from .. import region as regions
from .. import render as renders
from ..cli_inputs import CAMERA_FLAGS, add_view_args, given, resolve_camera

ASSET_FLAGS = ("source", "edited_view", "region", "mask", "camera_meta", "view", "camera", "orbit")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sap3d-texture", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh", type=Path, help="edited mesh (GLB, the source asset's frame)")
    parser.add_argument("--out", type=Path, help="output folder")
    parser.add_argument("--mode", choices=("composite", "generated"), default="composite")
    parser.add_argument("--config", type=Path, help="YAML overriding the default texture settings")
    parser.add_argument("--model", default="tencent/Hunyuan3D-2.1", help="paint model repository")
    parser.add_argument("--jobs", type=Path, help="run the texture jobs listed in this JSON file (as sap3d-bench "
                                                  "writes them) instead of one job from the flags")
    case = parser.add_argument_group("benchmark case")
    case.add_argument("--bench-root", type=Path, help="Edit3D-Bench data folder")
    case.add_argument("--case", help="<dataset>/<object>/<prompt_key>")
    asset = parser.add_argument_group("any asset")
    asset.add_argument("--source", type=Path, help="textured source mesh (GLB)")
    asset.add_argument("--edited-view", type=Path, help="edited view of the source (PNG)")
    asset.add_argument("--region", type=Path, help="3D region: GLB of closed parts, or YAML/JSON primitives")
    add_view_args(asset)
    return parser


def bench_job(case: bench.BenchCase, mesh, out_dir, mode: str, color_fit: bool) -> dict:
    """The texture job of a benchmark case: its files and the camera its 2D annotations were made from."""
    return {"mode": mode, "mesh": str(Path(mesh).resolve()), "source": str(case.source_glb.resolve()),
            "edited_view": str(case.edited_view.resolve()),
            "region": str(case.region_glb.resolve()) if mode == "composite" else None,
            "view": {"mask": str(case.mask.resolve()), "camera": {"view": bench.resolve_view_index(case)}},
            "color_fit": bool(color_fit and mode == "composite"), "out_dir": str(Path(out_dir).resolve())}


def _asset_job(args, color_fit: bool) -> dict:
    if args.source is None or args.edited_view is None:
        raise SystemExit("an asset needs --source and --edited-view")
    if args.view is not None and args.camera_meta is None:
        raise SystemExit("--view selects a camera of --camera-meta")
    if args.mode == "composite" and args.region is None:
        raise SystemExit("composite mode needs --region")
    if args.mode == "generated" and args.region is not None:
        raise SystemExit("generated mode reads no region; remove --region")
    has_camera = bool(given(args, CAMERA_FLAGS))
    if args.mode == "composite" and color_fit and not has_camera:
        raise SystemExit("the color fit renders through the edited view's camera: give --camera-meta/--view, "
                         "--camera or --orbit, or turn color_fit off in the texture configuration")
    if args.mask is not None and not has_camera:
        raise SystemExit("--mask belongs to the edited view's camera; give the camera too")
    view = None
    if has_camera:
        width, height = Image.open(args.edited_view).size
        if width != height:
            raise SystemExit(f"rendering through the edited view's camera needs a square view, got {width}x{height}")
        camera = resolve_camera(args, width)
        mask = args.mask
        if mask is None:
            if args.region is None:
                raise SystemExit("without --region the 2D mask cannot be projected; give --mask")
            projected = regions.project_mask(regions.load_region(args.region), frames.load_mesh(args.source), camera)
            args.out.mkdir(parents=True, exist_ok=True)
            mask = args.out / "mask_2d.png"
            Image.fromarray(projected.astype(np.uint8) * 255).save(mask)
        view = {"mask": str(Path(mask).resolve()), "camera": renders.camera_spec(camera)}
    return {"mode": args.mode, "mesh": str(args.mesh.resolve()), "source": str(args.source.resolve()),
            "edited_view": str(args.edited_view.resolve()),
            "region": str(args.region.resolve()) if args.region is not None else None,
            "view": view, "color_fit": bool(color_fit and args.mode == "composite"),
            "out_dir": str(args.out.resolve())}


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.jobs is not None:
        from .texture import run

        run(json.loads(args.jobs.read_text(encoding="utf-8")), config_module.load_texture_config(args.config),
            args.jobs.parent, model=args.model)
        return
    if args.out is None or args.mesh is None:
        raise SystemExit("--mesh and --out are required")
    cfg = config_module.load_texture_config(args.config)
    if args.case is not None or args.bench_root is not None:
        if args.case is None or args.bench_root is None:
            raise SystemExit("a benchmark case needs both --bench-root and --case")
        conflicting = given(args, ASSET_FLAGS)
        if conflicting:
            raise SystemExit(f"a benchmark case reads its inputs from the case folder; remove {conflicting}")
        job = bench_job(bench.parse_case(args.bench_root, args.case), args.mesh, args.out, args.mode, cfg.color_fit)
    else:
        job = _asset_job(args, cfg.color_fit)
    if not args.mesh.is_file():
        raise SystemExit(f"no mesh at {args.mesh}")

    from .texture import run

    run([job], cfg, args.out, model=args.model)


if __name__ == "__main__":
    main()
