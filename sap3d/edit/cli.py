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
from pathlib import Path

from PIL import Image

from .. import bench, config as config_module, frames
from .. import region as regions
from ..cli_inputs import add_view_args, given, resolve_camera

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
    add_view_args(asset)
    return parser


def _asset_inputs(args, cfg):
    if args.source is None or args.edited_view is None:
        raise SystemExit("an asset edit needs --source and --edited-view")
    if args.view is not None and args.camera_meta is None:
        raise SystemExit("--view selects a camera of --camera-meta")
    if cfg.mode == "uniform":
        ignored = given(args, ("region", "whole", "mask", "camera_meta", "camera", "orbit"))
        if ignored:
            raise SystemExit(f"uniform mode reads no region or mask; remove {ignored}")
        return args.source, args.edited_view, None, None, None
    if args.whole == (args.region is not None):
        raise SystemExit("give exactly one of --region and --global")
    if args.mask is not None and given(args, ("camera_meta", "camera", "orbit")):
        raise SystemExit("give either --mask or a camera to project the region, not both")
    if args.mask is None and not given(args, ("camera_meta", "camera", "orbit")):
        raise SystemExit("give --mask, or a camera (--camera-meta/--camera/--orbit) to project the region")
    source = frames.load_mesh(args.source)
    region = regions.global_region(source) if args.whole else regions.load_region(args.region)
    if args.mask is not None:
        return args.source, args.edited_view, region, args.mask, None
    width, height = Image.open(args.edited_view).size
    if width != height:
        raise SystemExit(f"a projected mask needs a square edited view, got {width}x{height}")
    mask = regions.project_mask(region, source, resolve_camera(args, width))
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
        conflicting = given(args, ASSET_FLAGS)
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

    from .edit import clear_outputs, run_edit, save_outputs
    from .model import load_edit_pipeline

    clear_outputs(args.out)
    pipeline = load_edit_pipeline(adapter, model)
    mesh, record, steps = run_edit(pipeline, cfg, source, edited_view, region, mask)
    save_outputs(args.out, mesh, record, steps, cfg, source=source, edited_view=edited_view, adapter=adapter,
                 projected_mask=projected)
    print(f"wrote {args.out / 'edit.glb'}  faces={record['faces']}  ({record['seconds']} s)")


if __name__ == "__main__":
    main()
