"""The download scripts against local folders laid out like their Hugging Face repositories."""

import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _local_fetcher(root: Path):
    def fetch(filename: str, directory: Path) -> Path:
        target = Path(directory) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / filename, target)
        return target
    return fetch


def _adapter_repo(root: Path) -> list:
    manifest = []
    for index, name in enumerate(["GSO/SORTING_BUS", "GSO/Dog", "PartObjaverse-Tiny/0c3ca2b3"]):
        folder = f"idx{index:02d}_{name.split('/')[1]}"
        (root / folder).mkdir(parents=True)
        for file in ("adapter_config.json", "adapter_model.safetensors"):
            (root / folder / file).write_text(f"{folder}/{file}")
        manifest.append({"folder": folder, "benchmark_object": name, "asset_index": index, "checkpoint_step": 500,
                         "validation_chamfer_distance": 0.01, "rank": 16, "alpha": 16, "target_modules": []})
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_adapter_download_writes_a_matching_manifest():
    script = _script("download_adapters")
    assert script.REPO_ID == "Itan-levin/sap3d-adapters"
    with tempfile.TemporaryDirectory() as folder:
        repo, out = Path(folder) / "repo", Path(folder) / "out"
        manifest = _adapter_repo(repo)
        fetch = _local_fetcher(repo)
        written = script.download(out, ["PartObjaverse-Tiny/0c3ca2b3"], fetch)
        assert [entry["folder"] for entry in written] == ["idx02_0c3ca2b3"]
        written = script.download(out, ["idx00_SORTING_BUS"], fetch)
        assert [entry["folder"] for entry in written] == ["idx00_SORTING_BUS", "idx02_0c3ca2b3"]
        assert json.loads((out / "manifest.json").read_text()) == [manifest[0], manifest[2]]
        assert (out / "idx00_SORTING_BUS" / "adapter_model.safetensors").read_text() == \
            "idx00_SORTING_BUS/adapter_model.safetensors"
        assert len(script.download(out, None, fetch)) == 3
        try:
            script.download(out, ["GSO/NOT_THERE"], fetch)
        except SystemExit:
            pass
        else:
            raise AssertionError("an unknown object was accepted")


def test_global_cases_assemble():
    script = _script("download_global_cases")
    assert script.REPO_ID == "Itan-levin/sap3d-global-edits"
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        bench, dataset, out = root / "bench", root / "dataset", root / "out"
        metadata = [{"dataset": "GSO", "source_model": "Dog", "prompt_1": "made of glass"},
                    {"dataset": "GSO", "source_model": "Yoshi", "prompt_1": "fur", "prompt_2": "spikes"}]
        for entry in metadata:
            (bench / "GSO" / entry["source_model"] / "source_model").mkdir(parents=True)
            (bench / "GSO" / entry["source_model"] / "source_model" / "model.glb").write_text("glb")
            for key in (k for k in entry if k.startswith("prompt_")):
                case = dataset / "GSO" / entry["source_model"] / key
                case.mkdir(parents=True)
                for name in script.CASE_FILES:
                    (case / name).write_text(name)
        (dataset / "metadata.json").write_text(json.dumps(metadata))

        def fetch(directory):
            shutil.copytree(dataset, directory, dirs_exist_ok=True)
        cases = script.assemble(out, bench, fetch)
        assert cases == ["GSO/Dog/prompt_1", "GSO/Yoshi/prompt_1", "GSO/Yoshi/prompt_2"]
        assert (out / "GSO" / "Yoshi" / "source_model" / "model.glb").read_text() == "glb"
        assert script.assemble(out, bench, fetch) == cases
        (dataset / "GSO" / "Dog" / "prompt_1" / "2d_mask.png").unlink()
        shutil.rmtree(out)
        try:
            script.assemble(out, bench, fetch)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("a case without its mask was accepted")
