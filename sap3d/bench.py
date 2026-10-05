"""Edit3D-Bench cases: file layout, case enumeration, annotation cameras and the adapter for each object.

A case id is `<dataset>/<object>/<prompt_key>`, e.g. `GSO/SORTING_BUS/prompt_2`, under a benchmark root
holding `metadata.json` and one folder per dataset.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

PROMPT_KEYS = ("prompt_1", "prompt_2", "prompt_3")
N_VIEWS = 16


@dataclass(frozen=True)
class BenchCase:
    root: Path
    dataset: str
    object_name: str
    prompt_key: str

    @property
    def case_id(self) -> str:
        return f"{self.dataset}/{self.object_name}/{self.prompt_key}"

    @property
    def object_dir(self) -> Path:
        return self.root / self.dataset / self.object_name

    @property
    def prompt_dir(self) -> Path:
        return self.object_dir / self.prompt_key

    @property
    def source_glb(self) -> Path:
        return self.object_dir / "source_model" / "model.glb"

    @property
    def meta_json(self) -> Path:
        return self.object_dir / "source_model" / "render" / "meta.json"

    def source_render(self, view: int) -> Path:
        return self.object_dir / "source_model" / "render" / f"render_{view:04d}.png"

    @property
    def region_glb(self) -> Path:
        return self.prompt_dir / "3d_edit_region.glb"

    @property
    def edited_view(self) -> Path:
        return self.prompt_dir / "2d_edit.png"

    @property
    def source_view(self) -> Path:
        return self.prompt_dir / "2d_render.png"

    @property
    def mask(self) -> Path:
        return self.prompt_dir / "2d_mask.png"

    def view_mask(self, view: int) -> Path:
        return self.prompt_dir / "render" / f"mask_{view:04d}.png"

    @property
    def annotation_view_override(self) -> Path:
        return self.prompt_dir / "annotation_view.json"


def parse_case(root, case_id: str) -> BenchCase:
    parts = case_id.strip("/").split("/")
    if len(parts) != 3 or parts[2] not in PROMPT_KEYS:
        raise ValueError(f"a case id is <dataset>/<object>/<prompt_key> with a prompt key in "
                         f"{list(PROMPT_KEYS)}, got {case_id!r}")
    case = BenchCase(Path(root), *parts)
    if not case.prompt_dir.is_dir():
        raise FileNotFoundError(f"no case folder at {case.prompt_dir}")
    return case


def iter_cases(root) -> list:
    """Every case under `root`, in metadata order and then prompt order."""
    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    return [BenchCase(root, entry["dataset"], entry["source_model"], key)
            for entry in metadata for key in PROMPT_KEYS]


def resolve_view_index(case: BenchCase) -> int:
    """Which of the 16 benchmark cameras the case's 2D annotations were made from.

    A case folder may name it in `annotation_view.json`. Otherwise `2d_mask.png` is matched against the
    per-view masks; where several match exactly (views 180 degrees apart), the source render closest to
    `2d_render.png` decides.
    """
    if case.annotation_view_override.is_file():
        return int(json.loads(case.annotation_view_override.read_text(encoding="utf-8"))["view"])
    target = np.asarray(Image.open(case.mask).convert("L"))
    hits = [view for view in range(N_VIEWS)
            if np.array_equal(np.asarray(Image.open(case.view_mask(view)).convert("L")), target)]
    if not hits:
        raise ValueError(f"{case.case_id}: 2d_mask.png matches none of the per-view masks")
    if len(hits) == 1:
        return hits[0]
    reference = np.asarray(Image.open(case.source_view).convert("RGBA"), dtype=np.float32)
    return min(hits, key=lambda view: np.abs(
        np.asarray(Image.open(case.source_render(view)).convert("RGBA"), dtype=np.float32) - reference).mean())


def adapter_dir(adapters_root, case: BenchCase) -> Path:
    """The adapter folder for the case's object, as listed in `<adapters_root>/manifest.json`."""
    adapters_root = Path(adapters_root)
    manifest = json.loads((adapters_root / "manifest.json").read_text(encoding="utf-8"))
    wanted = f"{case.dataset}/{case.object_name}"
    folders = [entry["folder"] for entry in manifest if entry["benchmark_object"] == wanted]
    if len(folders) != 1:
        raise ValueError(f"{adapters_root / 'manifest.json'} lists {len(folders)} adapters for {wanted}")
    return adapters_root / folders[0]
