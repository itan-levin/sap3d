"""sap3d-edit: edit one asset with its personalized adapter.

Benchmark case (inputs read from the case folder):
    sap3d-edit --bench-root DATA --case GSO/SORTING_BUS/prompt_2 --adapters ADAPTERS --out OUT

Any asset:
    sap3d-edit --source model.glb --edited-view edit.png --region region.yaml --mask mask.png \\
               --adapter ADAPTER_DIR --out OUT
  The 2D mask may instead be projected from the region through the edited view's camera: --camera-meta
  meta.json --view K, --camera camera.json, or --orbit ELEVATION AZIMUTH. --global replaces --region by a
  box around the whole source and projects it as the mask.

A global edit with one adapter strength for every token: --config configs/edit_global_uniform.yaml (no
region or mask is read).

Writes OUT/edit.glb (source frame), OUT/edit.json, for a local edit OUT/regions.npz, and for a projected
mask OUT/mask_2d.png.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .. import bench, config as config_module, frames
from .. import camera as cameras
from .. import region as regions

OUTPUT_FILES = ("edit.glb", "edit.json", "regions.npz", "mask_2d.png")
ASSET_FLAGS = ("source", "edited_view", "region", "whole", "mask", "camera_meta", "view", "camera", "orbit")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sap3d-edit", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="YAML overriding the default settings")
    parser.add_argument("--dump-config", action="store_true", help="print the resolved settings and exit")
    parser.add_argument("--out", type=Path, help="output folder")
    parser.add_argument("--model", default="tencent/Hunyuan3D-2.1", help="shape model repository or folder")
    adapters = parser.add_mutually_exclusive_group()
    adapters.add_argument("--adapter", type=Path, help="adapter folder (adapter_config.json + weights)")
    adapters.add_argument("--adapters", type=Path, help="folder of benchmark adapters with manifest.json")
    case = parser.add_argument_group("benchmark case")
    case.add_argument("--bench-root", type=Path, help="Edit3D-Bench data folder")
    case.add_argument("--case", help="<dataset>/<object>/<prompt_key>")
    asset = parser.add_argument_group("any asset")
    asset.add_argument("--source", type=Path, help="source mesh (GLB)")
    asset.add_argument("--edited-view", type=Path, help="edited view of the source (PNG)")
    asset.add_argument("--region", type=Path, help="3D region: GLB of closed parts, or YAML/JSON primitives")
    asset.add_argument("--global", dest="whole", action="store_true",
                       help="region = the source's bounding box padded by 0.1 of its longest side")
    asset.add_argument("--mask", type=Path, help="2D mask of the edited view (single-channel PNG)")
    asset.add_argument("--camera-meta", type=Path, help="meta.json holding the edited view's camera")
    asset.add_argument("--view", type=int, help="camera index in --camera-meta")
    asset.add_argument("--camera", type=Path, help="single-camera JSON")
    asset.add_argument("--orbit", type=float, nargs=2, metavar=("ELEVATION", "AZIMUTH"),
                       help="benchmark-style orbit camera, degrees")
    return parser


def _given(args, names) -> list:
    return [f"--{name.replace('_', '-')}" if name != "whole" else "--global" for name in names
            if getattr(args, name) not in (None, False)]


def _camera(args, size: int):
    chosen = _given(args, ("camera_meta", "camera", "orbit"))
    if len(chosen) != 1:
        raise SystemExit(f"projecting the mask needs exactly one of --camera-meta, --camera, --orbit; got {chosen}")
    if args.camera_meta is not None:
        if args.view is None:
            raise SystemExit("--camera-meta needs --view")
        camera = cameras.from_meta(args.camera_meta, args.view)
    elif args.camera is not None:
        camera = cameras.from_json(args.camera)
    else:
        camera = cameras.orbit(args.orbit[0], args.orbit[1], size)
    if camera.size != size:
        raise SystemExit(f"the camera frame is {camera.size} px but the edited view is {size} px")
    return camera


def _asset_inputs(args, cfg):
    if args.source is None or args.edited_view is None:
        raise SystemExit("an asset edit needs --source and --edited-view")
    if args.view is not None and args.camera_meta is None:
        raise SystemExit("--view selects a camera of --camera-meta")
    if cfg.mode == "uniform":
        ignored = _given(args, ("region", "whole", "mask", "camera_meta", "camera", "orbit"))
        if ignored:
            raise SystemExit(f"uniform mode reads no region or mask; remove {ignored}")
        return args.source, args.edited_view, None, None, None
    if args.whole == (args.region is not None):
        raise SystemExit("give exactly one of --region and --global")
    if args.mask is not None and _given(args, ("camera_meta", "camera", "orbit")):
        raise SystemExit("give either --mask or a camera to project the region, not both")
    if args.mask is None and not _given(args, ("camera_meta", "camera", "orbit")):
        raise SystemExit("give --mask, or a camera (--camera-meta/--camera/--orbit) to project the region")
    source = frames.load_mesh(args.source)
    region = regions.global_region(source) if args.whole else regions.load_region(args.region)
    if args.mask is not None:
        return args.source, args.edited_view, region, args.mask, None
    width, height = Image.open(args.edited_view).size
    if width != height:
        raise SystemExit(f"a projected mask needs a square edited view, got {width}x{height}")
    mask = regions.project_mask(region, source, _camera(args, width))
    return args.source, args.edited_view, region, mask, mask


def main(argv=None):
    args = _parser().parse_args(argv)
    cfg = config_module.load_config(args.config)
    if args.dump_config:
        print(config_module.dump_config(cfg), end="")
        return
    if args.out is None:
        raise SystemExit("--out is required")
    benchmark = args.case is not None or args.bench_root is not None
    if args.adapter is None and (args.adapters is None or not benchmark):
        raise SystemExit("give --adapter, or --adapters for a benchmark case")

    if benchmark:
        if args.case is None or args.bench_root is None:
            raise SystemExit("a benchmark case needs both --bench-root and --case")
        conflicting = _given(args, ASSET_FLAGS)
        if conflicting:
            raise SystemExit(f"a benchmark case reads its inputs from the case folder; remove {conflicting}")
        case = bench.parse_case(args.bench_root, args.case)
        source, edited_view = case.source_glb, case.edited_view
        region = regions.load_region(case.region_glb) if cfg.mode == "local" else None
        mask = case.mask if cfg.mode == "local" else None
        projected = None
        adapter = args.adapter if args.adapter is not None else bench.adapter_dir(args.adapters, case)
    else:
        source, edited_view, region, mask, projected = _asset_inputs(args, cfg)
        adapter = args.adapter
    model = str(Path(args.model).resolve()) if Path(args.model).exists() else args.model

    args.out.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_FILES:
        (args.out / name).unlink(missing_ok=True)

    from .edit import run_edit, steps_to_npz
    from .model import load_edit_pipeline

    pipeline = load_edit_pipeline(adapter, model)
    mesh, record, steps = run_edit(pipeline, cfg, source, edited_view, region, mask)

    mesh.export(args.out / "edit.glb")
    if projected is not None:
        Image.fromarray(projected.astype(np.uint8) * 255).save(args.out / "mask_2d.png")
    if steps:
        np.savez_compressed(args.out / "regions.npz", **steps_to_npz(steps))
    record.update(source=str(source), edited_view=str(edited_view), adapter=str(adapter),
                  settings=config_module.to_dict(cfg))
    (args.out / "edit.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"wrote {args.out / 'edit.glb'}  faces={record['faces']}  ({record['seconds']} s)")


if __name__ == "__main__":
    main()
