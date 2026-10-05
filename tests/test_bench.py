"""Benchmark case ids, adapter lookup and the annotation camera."""

import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from sap3d import bench


def _case_tree(folder: Path) -> bench.BenchCase:
    case = bench.BenchCase(folder, "GSO", "THING", "prompt_2")
    (case.prompt_dir / "render").mkdir(parents=True)
    (case.object_dir / "source_model" / "render").mkdir(parents=True)
    rng = np.random.default_rng(0)
    masks = [(rng.random((8, 8)) > 0.5).astype(np.uint8) * 255 for _ in range(bench.N_VIEWS)]
    masks[12] = masks[4]
    for view, mask in enumerate(masks):
        Image.fromarray(mask).save(case.view_mask(view))
        render = np.zeros((8, 8, 4), np.uint8)
        render[..., 0] = view * 10
        Image.fromarray(render).save(case.source_render(view))
    Image.fromarray(masks[4]).save(case.mask)
    edited = np.zeros((8, 8, 4), np.uint8)
    edited[..., 0] = 120
    Image.fromarray(edited).save(case.source_view)
    return case


def test_parse_case():
    with tempfile.TemporaryDirectory() as folder:
        case = _case_tree(Path(folder))
        assert bench.parse_case(folder, "GSO/THING/prompt_2") == case
        for bad in ("GSO/THING", "GSO/THING/prompt_9", "GSO/OTHER/prompt_2"):
            try:
                bench.parse_case(folder, bad)
            except (ValueError, FileNotFoundError):
                continue
            raise AssertionError(f"accepted {bad}")


def test_view_index_breaks_ties_by_render_and_honors_override():
    with tempfile.TemporaryDirectory() as folder:
        case = _case_tree(Path(folder))
        assert bench.resolve_view_index(case) == 12
        case.annotation_view_override.write_text(json.dumps({"view": 3}))
        assert bench.resolve_view_index(case) == 3


def test_adapter_lookup():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root / "manifest.json").write_text(json.dumps([
            {"folder": "idx00_THING", "benchmark_object": "GSO/THING"},
            {"folder": "idx01_OTHER", "benchmark_object": "GSO/OTHER"}]))
        case = bench.BenchCase(root, "GSO", "THING", "prompt_1")
        assert bench.adapter_dir(root, case) == root / "idx00_THING"
        try:
            bench.adapter_dir(root, bench.BenchCase(root, "GSO", "MISSING", "prompt_1"))
        except ValueError:
            return
        raise AssertionError("a missing adapter was not reported")
