"""Assemble the 12 whole-object (global) edit cases as a benchmark folder that `sap3d-edit`, `sap3d-texture`,
`sap3d-eval` and `sap3d-bench --global` read with `--bench-root`.

    python scripts/download_global_cases.py --edit3d-bench Edit3D-Bench/data --out global_edits

Downloads the cases (edited views, masks, whole-object regions, annotation cameras and metadata.json) from the
Hugging Face dataset REPO_ID, and links each object's `source_model/` from a local copy of Edit3D-Bench.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ID = "Itan-levin/sap3d-global-edits"
CASE_FILES = ("2d_edit.png", "2d_mask.png", "3d_edit_region.glb", "annotation_view.json")


def hub_fetcher(repo_id: str = REPO_ID):
    """fetch(directory) places the dataset's metadata.json and case folders under directory."""
    from huggingface_hub import snapshot_download

    def fetch(directory: Path) -> None:
        snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(directory),
                          allow_patterns=["metadata.json", "*/*/prompt_*/*"])
    return fetch


def assemble(out: Path, edit3d_bench: Path, fetch) -> list:
    """Fetch the cases into `out` and link each object's source model; returns the case ids."""
    out, edit3d_bench = Path(out), Path(edit3d_bench).resolve()
    out.mkdir(parents=True, exist_ok=True)
    fetch(out)
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    cases = []
    for entry in metadata:
        object_dir = out / entry["dataset"] / entry["source_model"]
        source = edit3d_bench / entry["dataset"] / entry["source_model"] / "source_model"
        if not (source / "model.glb").is_file():
            raise FileNotFoundError(f"{source / 'model.glb'} is missing; --edit3d-bench names Edit3D-Bench's data folder")
        link = object_dir / "source_model"
        if link.is_symlink() or link.exists():
            if link.resolve() != source:
                raise FileExistsError(f"{link} exists and is not a link to {source}")
        else:
            link.symlink_to(source, target_is_directory=True)
        for key in sorted(k for k in entry if k.startswith("prompt_")):
            missing = [name for name in CASE_FILES if not (object_dir / key / name).is_file()]
            if missing:
                raise FileNotFoundError(f"{object_dir / key} lacks {missing}")
            cases.append(f"{entry['dataset']}/{entry['source_model']}/{key}")
    return cases


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--edit3d-bench", type=Path, required=True, help="Edit3D-Bench's data folder")
    parser.add_argument("--out", type=Path, required=True, help="folder to assemble the global cases in")
    args = parser.parse_args(argv)
    cases = assemble(args.out, args.edit3d_bench, hub_fetcher())
    (args.out / "cases.txt").write_text("\n".join(cases) + "\n", encoding="utf-8")
    print(f"{len(cases)} cases in {args.out}; their ids are in {args.out / 'cases.txt'}")


if __name__ == "__main__":
    sys.exit(main())
