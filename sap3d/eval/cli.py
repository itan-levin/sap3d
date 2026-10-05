"""sap3d-eval: evaluate a textured edit.

    sap3d-eval --texture-dir TEXTURE_OUT --out OUT [--bench-root DATA --case GSO/SORTING_BUS/prompt_2]

Reads every input from TEXTURE_OUT/texture.json. Writes OUT/eval.json with preservation (exact distance to
the source, outside the region for composite-mode textures, over the whole surface for generated ones),
edit fidelity (needs the edited view's camera in the texture record) and, for a benchmark case textured in
composite mode, the Edit3D-Bench scorer's six metrics. Needs a source checkout of this repository: the
vendored scorer is used by path.
"""

import argparse
import json
from pathlib import Path

from .. import bench


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sap3d-eval", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--texture-dir", type=Path, help="output folder of sap3d-texture")
    parser.add_argument("--out", type=Path, help="output folder")
    parser.add_argument("--jobs", type=Path, help="run the eval jobs listed in this JSON file (as sap3d-bench "
                                                  "writes them) instead of one job from the flags")
    parser.add_argument("--bench-root", type=Path, help="Edit3D-Bench data folder, for the benchmark metrics")
    parser.add_argument("--case", help="<dataset>/<object>/<prompt_key>")
    parser.add_argument("--fidelity-device", help="device of the fidelity encoders (default: cpu for "
                                                  "composite-mode textures, cuda:0 for generated ones)")
    args = parser.parse_args(argv)
    if args.jobs is not None:
        from .evaluate import run

        run(json.loads(args.jobs.read_text(encoding="utf-8")), args.jobs.parent)
        return
    if args.texture_dir is None or args.out is None:
        raise SystemExit("--texture-dir and --out are required")
    if (args.case is None) != (args.bench_root is None):
        raise SystemExit("the benchmark metrics need both --bench-root and --case")
    if not (args.texture_dir / "texture.json").is_file():
        raise SystemExit(f"no texture.json in {args.texture_dir}")
    if args.case is not None:
        bench.parse_case(args.bench_root, args.case)
        mode = json.loads((args.texture_dir / "texture.json").read_text(encoding="utf-8"))["mode"]
        if mode != "composite":
            raise SystemExit(f"benchmark metrics are scored on composite-mode textures; this one is {mode}")

    from .evaluate import run

    job = {"texture_dir": str(args.texture_dir.resolve()), "out_dir": str(args.out.resolve()),
           "fidelity_device": args.fidelity_device,
           "bench": {"root": str(args.bench_root.resolve()), "case": args.case} if args.case else None}
    record = run([job], args.out)[0]
    print({name: round(row["symmetric"], 6) for name, row in record["preservation"].items()})


if __name__ == "__main__":
    main()
