"""sap3d-personalize: fit a per-object adapter for the edit stage.

    sap3d-personalize prepare --source asset.glb --out ASSET        any glTF/GLB asset -> benchmark layout
    sap3d-personalize dataset --asset ASSET --out DATA               views, renders and surface points
    sap3d-personalize train   --data DATA --out RUN                   adapter in RUN/adapter
    sap3d-personalize select  --run RUN [--selection-metric legacy]   choose again from RUN/training.json

An Edit3D-Bench object folder (e.g. Edit3D-Bench/data/GSO/SORTING_BUS) is already an ASSET folder.
`dataset` and `train` take `--config`, a YAML file overriding the defaults listed in configs/personalize.yaml,
and `--dump-config`, which prints the resolved settings.
"""

import argparse
import sys
from pathlib import Path

import yaml

from .. import config as config_module


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sap3d-personalize", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    settings = argparse.ArgumentParser(add_help=False)
    settings.add_argument("--config", type=Path, help="YAML overriding the default settings")
    settings.add_argument("--dump-config", action="store_true", help="print the resolved settings and exit")
    commands = parser.add_subparsers(dest="command")
    prepare = commands.add_parser("prepare", help="any glTF/GLB asset -> Edit3D-Bench object layout")
    prepare.add_argument("--source", type=Path, required=True, help="glTF or GLB file")
    prepare.add_argument("--out", type=Path, required=True, help="asset folder to create")
    dataset = commands.add_parser("dataset", parents=[settings],
                                  help="training views, renders and surface points of one asset")
    dataset.add_argument("--asset", type=Path, required=True, help="folder holding source_model/")
    dataset.add_argument("--out", type=Path, required=True, help="dataset folder to create")
    dataset.add_argument("--seed", type=int, help="make the camera, lighting, engine and surface draws repeatable")
    train = commands.add_parser("train", parents=[settings], help="fit the adapter")
    train.add_argument("--data", type=Path, required=True, help="dataset folder")
    train.add_argument("--out", type=Path, required=True, help="run folder to create")
    train.add_argument("--model", default="tencent/Hunyuan3D-2.1", help="shape model repository or folder")
    choose = commands.add_parser("select", help="choose the adapter of a finished or interrupted run again")
    choose.add_argument("--run", type=Path, required=True, help="run folder holding training.json")
    for command in (train, choose):
        command.add_argument("--selection-metric", choices=config_module.SELECTION_METRICS,
                             help="Chamfer definition that selects the step (default: the configuration's)")
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    if args.command is None:
        raise SystemExit("name a command: prepare, dataset, train or select")
    overrides = {}
    if getattr(args, "seed", None) is not None:
        overrides["dataset"] = {"seed": args.seed}
    if getattr(args, "selection_metric", None) is not None and args.command == "train":
        overrides["validation"] = {"selection_metric": args.selection_metric}
    cfg = config_module.load_personalize_config(getattr(args, "config", None), overrides)
    if getattr(args, "dump_config", False):
        sys.stdout.write(yaml.safe_dump(config_module.personalize_to_dict(cfg), sort_keys=False))
        return
    if args.command == "prepare":
        from .prepare import prepare
        print(prepare(args.source, args.out))
    elif args.command == "dataset":
        from .dataset import build
        print(build(args.asset, args.out, cfg.dataset))
    elif args.command == "train":
        from .train import run
        print(run(cfg, args.data, args.out, args.model))
    else:
        from .select import select
        print(select(args.run, args.selection_metric))


if __name__ == "__main__":
    main()
