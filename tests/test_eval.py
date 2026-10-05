"""Eval stage and batch runner logic that needs no GPU: crops, scorer count checks, render cameras, resume and
summaries, texture settings and command-line validation."""

import json
import math
import pathlib
import tempfile

import numpy as np
from PIL import Image

from sap3d import camera as cameras
from sap3d import config, render
from sap3d.eval import bench as batch
from sap3d.eval import fidelity, scorer
from sap3d.texture import cli as texture_cli

CONFIGS = pathlib.Path(__file__).resolve().parent.parent / "configs"
MISSING = "/nonexistent/sap3d"


def _exits(function, argv) -> bool:
    try:
        function(argv)
    except SystemExit as stop:
        return stop.code not in (0, None)
    return False


def _raises(function, *args) -> bool:
    try:
        function(*args)
    except ValueError:
        return True
    return False


def test_square_box():
    region = np.zeros((100, 100), dtype=bool)
    region[10:20, 30:60] = True
    assert fidelity.square_box(region) == (0, 30, 30)
    small = np.zeros((100, 100), dtype=bool)
    small[90:99, 40:45] = True
    assert _raises(fidelity.square_box, small)
    region = np.zeros((100, 100), dtype=bool)
    region[40:60, 70:100] = True
    top, left, side = fidelity.square_box(region)
    assert (side, left + side) == (30, 100) and top <= 40 and top + side >= 60


def test_masked_crops_flatten_outside_the_mask():
    rng = np.random.default_rng(0)
    render_rgb = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    target_rgb = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:40, 10:50] = True
    mask[25, 15] = False
    (first, second), record = fidelity.masked_crops(Image.fromarray(render_rgb), Image.fromarray(target_rgb), mask)
    top, left, side = record["box_top"], record["box_left"], record["crop_side"]
    assert side == 40 and record["mask_pixels"] == int(mask.sum())
    inside = mask[top:top + side, left:left + side]
    assert (first[~inside] == fidelity.UNSCORED_GRAY).all() and (second[~inside] == fidelity.UNSCORED_GRAY).all()
    assert np.array_equal(first[inside], render_rgb[top:top + side, left:left + side][inside])
    assert _raises(fidelity.masked_crops, Image.fromarray(render_rgb), Image.fromarray(render_rgb), mask)


def test_scorer_counts():
    complete = {metric: {"mean": 1.0, "count": count} for metric, count in scorer.SCORES_PER_CASE.items()}
    assert scorer.check_counts(complete) == {}
    short = dict(complete, psnr={"mean": 1.0, "count": 14})
    assert scorer.check_counts(short) == {"psnr": 2}
    for broken in (dict(short, ssim={"mean": 1.0, "count": 15}), dict(complete, chamfer={"mean": None, "count": 0})):
        try:
            scorer.check_counts(broken)
        except RuntimeError:
            continue
        raise AssertionError(f"incomplete scorer results were accepted: {broken}")


def test_render_camera_spec_round_trips_a_benchmark_camera():
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "meta.json"
        matrix = cameras.orbit_blender_matrix(15.0, -90.0)
        path.write_text(json.dumps({"width": 1024, "height": 1024, "locations": [
            {"transform_matrix": matrix.tolist(), "camera_angle_x": cameras.ORBIT_FOV_X}]}))
        spec = render.camera_spec(cameras.from_meta(path, 0))
    assert np.array_equal(np.asarray(spec["transform_matrix"]), matrix)
    assert spec["size"] == 1024
    assert np.float32(render.lens_mm(spec["camera_angle_x"])) == np.float32(35.0)
    assert np.float32(render.lens_mm(0.8575560450553894)) == np.float32(35.0)
    assert render.frame_name({"view": 7}) == "render_0007.png" and render.frame_name(spec) == "render_0000.png"
    assert math.isclose(render.lens_mm(math.radians(90.0)), 16.0)


def _write_case(root: pathlib.Path, case_id: str, value: float, mode: str = "composite"):
    folder = root / "cases" / case_id
    for stage in ("edit", "texture", "eval"):
        (folder / stage).mkdir(parents=True)
    (folder / "edit" / "edit.json").write_text(json.dumps({"faces": 10, "final_region_tokens": 5,
                                                           "output_turn_degrees": 0, "seconds": 1.0}))
    (folder / "texture" / "texture.json").write_text(json.dumps({"seconds": 2.0}))
    fidelity_rows = {"composite": {"dino_embedding_distance": value, "clip_embedding_distance": value / 2}}
    benchmark = {"chamfer": {"mean": value / 10, "count": 1, "all_scores": [value / 10]},
                 "psnr": {"mean": value * 100, "count": 2, "all_scores": [value * 100] * 2},
                 "views_dropped_for_zero_error": {}}
    (folder / "eval" / "eval.json").write_text(json.dumps({
        "mode": mode, "preservation": {"raw": {"symmetric": value}, "composite": {"symmetric": value}},
        "fidelity": fidelity_rows, "benchmark": benchmark}))


def test_summary_covers_every_finished_case():
    with tempfile.TemporaryDirectory() as folder:
        out = pathlib.Path(folder)
        _write_case(out, "GSO/A/prompt_1", 0.1)
        _write_case(out, "GSO/B/prompt_2", 0.3)
        _write_case(out, "PartObjaverse-Tiny/C/prompt_1", 0.9, mode="generated")
        (out / "cases" / "GSO" / "C" / "prompt_1" / "edit").mkdir(parents=True)
        summary = batch.write_summary(out)
        assert sorted(summary["cases"]) == ["GSO/A/prompt_1", "GSO/B/prompt_2", "PartObjaverse-Tiny/C/prompt_1"]
        assert summary["means"]["generated"]["dino_uncorrected"]["count"] == 1
        means = summary["means"]["composite"]
        entry = means["dino_uncorrected"]
        assert math.isclose(entry["mean"], 0.2) and (entry["count"], entry["of"]) == (2, 2)
        assert math.isclose(means["bench_chamfer"]["mean"], 0.02) and means["bench_psnr"]["scores"] == 4
        assert (out / "cases" / "GSO" / "A" / "prompt_1" / "case.json").is_file()
        header = (out / "summary.csv").read_text().splitlines()[0].split(",")
        assert header[:2] == ["case", "mode"] and "preservation_composite" in header
        assert "2 of 2" in (out / "summary.md").read_text() and "1 of 1" in (out / "summary.md").read_text()


def test_resume_rules():
    record = {"settings": {"a": 1}, "inputs": {"mesh": "x"}}
    assert batch._must_run("edit", "c", record, record, overwrite=False) is False
    assert batch._must_run("edit", "c", record, record, overwrite=True) is True
    assert batch._must_run("edit", "c", record, dict(record, inputs={"mesh": "y"}), overwrite=False) is True
    assert _exits(lambda _: batch._must_run("edit", "c", record, dict(record, settings={"a": 2}), False), None)
    assert _exits(lambda _: batch._must_run("edit", "c", None, record, False), None)
    assert batch._must_run("edit", "c", None, record, overwrite=True) is True


def test_case_list_rules():
    class Args:
        cases, case, skip_cases = None, ["GSO/A/prompt_1", "GSO/B/prompt_1"], ["GSO/B/prompt_1"]
    assert batch.read_cases(Args) == ["GSO/A/prompt_1"]
    Args.skip_cases = ["GSO/C/prompt_1"]
    assert _exits(lambda _: batch.read_cases(Args), None)
    Args.case, Args.skip_cases = ["GSO/A/prompt_1", "GSO/A/prompt_1"], None
    assert _exits(lambda _: batch.read_cases(Args), None)


def test_texture_settings():
    cfg = config.load_texture_config()
    assert (cfg.atlas_size, cfg.target_faces, cfg.boundary_margin, cfg.feather_width) == (2048, 40_000, 0.02, 0.01)
    assert config.load_texture_config(CONFIGS / "texture.yaml") == cfg
    with tempfile.TemporaryDirectory() as folder:
        for text in ("target_faces: 50000\n", "atlas_sise: 1024\n", "feather_width: 0\n", "color_fit: 1\n"):
            path = pathlib.Path(folder) / "t.yaml"
            path.write_text(text)
            try:
                config.load_texture_config(path)
            except ValueError:
                continue
            raise AssertionError(f"accepted {text!r}")


def test_texture_command_rejects_inconsistent_flags():
    base = ["--mesh", MISSING + ".glb", "--out", MISSING + "/out"]
    assert _exits(texture_cli.main, base + ["--bench-root", MISSING, "--case", "GSO/X/prompt_1", "--region", "r.yaml"])
    assert _exits(texture_cli.main, base + ["--case", "GSO/X/prompt_1"])
    asset = base + ["--source", MISSING + ".glb", "--edited-view", MISSING + ".png"]
    assert _exits(texture_cli.main, asset)
    assert _exits(texture_cli.main, asset + ["--region", "r.yaml"])
    assert _exits(texture_cli.main, asset + ["--mode", "generated", "--region", "r.yaml"])
    assert _exits(texture_cli.main, asset + ["--mode", "generated", "--mask", "m.png"])
    assert _exits(texture_cli.main, ["--out", MISSING])
