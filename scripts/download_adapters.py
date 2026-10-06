"""Download the per-object adapters for the 100 Edit3D-Bench objects from Hugging Face.

    python scripts/download_adapters.py --list
    python scripts/download_adapters.py --out adapters --objects GSO/SORTING_BUS GSO/TURBOPROP_AIRPLANE_WITH_PILOT
    python scripts/download_adapters.py --out adapters --all

Objects are named by their Edit3D-Bench id (`<dataset>/<object>`) or by their folder in the repository
(`idxNN_<object>`). Writes OUT/<folder>/{adapter_config.json, adapter_model.safetensors} and OUT/manifest.json,
which lists every adapter present in OUT and is what `sap3d-edit --adapters OUT` and `sap3d-bench` read.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ID = "Itan-levin/sap3d-adapters"
FILES = ("adapter_config.json", "adapter_model.safetensors")


def hub_fetcher(repo_id: str = REPO_ID):
    """fetch(filename, directory) places the repository file `filename` at directory/filename."""
    from huggingface_hub import hf_hub_download

    def fetch(filename: str, directory: Path) -> Path:
        return Path(hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(directory)))
    return fetch


def read_manifest(fetch) -> list:
    with tempfile.TemporaryDirectory() as scratch:
        return json.loads(fetch("manifest.json", Path(scratch)).read_text(encoding="utf-8"))


def select_entries(manifest: list, objects) -> list:
    if objects is None:
        return list(manifest)
    by_name = {}
    for entry in manifest:
        by_name[entry["benchmark_object"]] = entry
        by_name[entry["folder"]] = entry
    unknown = [name for name in objects if name not in by_name]
    if unknown:
        raise SystemExit(f"not in the adapter repository: {unknown} (see --list)")
    chosen = {by_name[name]["folder"]: by_name[name] for name in objects}
    return list(chosen.values())


def download(out: Path, objects, fetch) -> list:
    """Download the adapters of `objects` (None for all) into `out`; returns the manifest written there."""
    out = Path(out)
    entries = select_entries(read_manifest(fetch), objects)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    present = {}
    if manifest_path.is_file():
        present = {entry["benchmark_object"]: entry
                   for entry in json.loads(manifest_path.read_text(encoding="utf-8"))}
    for entry in entries:
        for name in FILES:
            fetch(f"{entry['folder']}/{name}", out)
        present[entry["benchmark_object"]] = entry
    merged = sorted(present.values(), key=lambda entry: entry["asset_index"])
    missing = [f"{e['folder']}/{name}" for e in merged for name in FILES if not (out / e["folder"] / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{manifest_path} would list adapters whose files are missing: {missing}")
    manifest_path.write_text(json.dumps(merged, indent=1), encoding="utf-8")
    return merged


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="folder to download into")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--objects", nargs="+", help="Edit3D-Bench ids or repository folders")
    group.add_argument("--all", action="store_true", help="all 100 adapters")
    group.add_argument("--list", action="store_true", help="print the repository's manifest and exit")
    args = parser.parse_args(argv)
    fetch = hub_fetcher()
    if args.list:
        for entry in read_manifest(fetch):
            print(f"{entry['benchmark_object']:<60} {entry['folder']}  step {entry['checkpoint_step']}")
        return
    if args.out is None:
        raise SystemExit("--out is required to download")
    merged = download(args.out, None if args.all else args.objects, fetch)
    print(f"{len(merged)} adapters listed in {args.out / 'manifest.json'}")


if __name__ == "__main__":
    sys.exit(main())
