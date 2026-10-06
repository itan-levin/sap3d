"""Texture stage on synthetic inputs, checked against recorded digests of the expected outputs."""

import hashlib
import json
import pathlib
import tempfile

import numpy as np
import trimesh
from PIL import Image

from sap3d.texture import atlas, color_fit, composite, reference, texels


def _sha(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def _composite_inputs():
    rng = np.random.default_rng(0)
    size = 64
    mapped = np.zeros((size, size), dtype=bool)
    mapped[4:-4, 4:-4] = True
    x = np.linspace(-0.16, 0.16, size)[None, :].repeat(size, 0)
    depth = np.where(mapped, 0.05 - np.abs(x), 0.0).astype(np.float32)
    region = mapped & (depth > 0)
    copied = mapped & ~region & (rng.random((size, size)) < 0.05)
    source = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    painted = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    return source, painted, region, depth, mapped, copied


def _plane_and_boxes():
    size = 96
    u = np.linspace(-0.2, 0.2, size)
    position = np.stack(np.meshgrid(u, u, indexing="xy") + [np.zeros((size, size))], axis=-1).astype(np.float32)
    mapped = np.ones((size, size), dtype=bool)
    mapped[:3] = False
    position[~mapped] = np.nan
    boxes = [trimesh.creation.box(extents=(0.1, 0.12, 0.1)),
             trimesh.creation.box(extents=(0.06, 0.06, 0.06),
                                  transform=trimesh.transformations.translation_matrix((0.12, -0.1, 0.0)))]
    return position, mapped, trimesh.util.concatenate(boxes)


def _textured_sphere():
    rng = np.random.default_rng(1)
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    v = sphere.vertices
    uv = np.stack([0.5 + np.arctan2(v[:, 1], v[:, 0]) / (2 * np.pi),
                   0.5 + np.arcsin(np.clip(v[:, 2] / 0.5, -1, 1)) / np.pi], axis=1)
    texture = Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
    sphere.visual = trimesh.visual.TextureVisuals(uv=uv, material=trimesh.visual.material.PBRMaterial(
        baseColorTexture=texture))
    target = trimesh.creation.icosphere(subdivisions=2, radius=0.51)
    target = trimesh.Trimesh(vertices=target.vertices * np.array([1.0, 1.02, 0.98]), faces=target.faces, process=False)
    return sphere, target


def _quad():
    mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], faces=[[0, 1, 2], [0, 2, 3]],
                           process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64))
    return mesh


def _fit_classes():
    rng = np.random.default_rng(3)
    size = 16
    generated = np.zeros((size, size), dtype=bool)
    generated[:, :6] = True
    ramp = np.zeros((size, size), dtype=bool)
    ramp[:, 6:8] = True
    return {"generated": generated, "ramp": ramp, "copied": ~generated & ~ramp, "region": generated,
            "composite": rng.integers(40, 200, (size, size, 3), dtype=np.uint8),
            "source_at_ramp": rng.integers(0, 256, (int(ramp.sum()), 3)).astype(np.float64),
            "ramp_weight": rng.random(int(ramp.sum())),
            "painted_at_ramp": rng.integers(0, 256, (int(ramp.sum()), 3), dtype=np.uint8)}


def _fake_render(glb, out_dir, generated):
    loaded = trimesh.load(str(glb), force="mesh", process=False)
    texture = np.asarray(loaded.visual.material.baseColorTexture.convert("RGB")).astype(np.float64)
    color = np.clip(texture[generated].mean(axis=0) * 0.8 + 12.0, 0, 255).round().astype(np.uint8)
    image = np.zeros((32, 32, 4), dtype=np.uint8)
    image[..., :3] = color
    image[..., 3] = 255
    path = pathlib.Path(out_dir) / "render_0000.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    return path


def test_composite_matches_research():
    source, painted, region, depth, mapped, copied = _composite_inputs()
    out, stats = composite.composite(source, painted, region, depth=depth, depth_gate=mapped, copied_from_region=copied)
    assert _sha(out) == "c732f3978eda0941"
    assert (stats["n_outside_band"], stats["n_seam_band"]) == (743, 210)
    assert np.array_equal(out[region], painted[region])
    untouched = mapped & ~region & ~copied & (depth < -0.05)
    assert np.array_equal(out[untouched], source[untouched])


def test_region_mask_and_signed_depth_match_research():
    position, mapped, boxes = _plane_and_boxes()
    mesh = texels.region_mesh(boxes)
    mask = texels.region_mask(position, mapped, mesh)
    depth = texels.signed_depth(position, mapped, mask, mesh)
    assert (_sha(mask), int(mask.sum()), _sha(depth)) == ("747e681f31d9c81b", 882, "0d2fb760d8d824d9")
    assert (depth[mask] >= 0).all() and (depth[mapped & ~mask] <= 0).all()


def test_transfer_matches_research():
    sphere, target = _textured_sphere()
    work, normals = atlas.unwrap(atlas.decimate(target, 40_000))
    position, normal, mapped = atlas.bake_surface_maps(work, 64, normals)
    rgb, stats = atlas.transfer_texture(position, mapped, normal, sphere)
    assert (_sha(rgb), _sha(mapped), int(mapped.sum())) == ("90d75f39db804a33", "0ef76382eacc3ea4", 3284)
    assert stats["texels_mapped"] == 3284


def test_recompose_matches_research():
    rng = np.random.default_rng(2)
    image = np.full((150, 200, 3), 255, dtype=np.uint8)
    image[30:110, 50:170] = rng.integers(0, 200, (80, 120, 3), dtype=np.uint8)
    image[5, 5] = (0, 0, 0)
    framed = reference.recompose(image)
    assert framed.shape == (512, 512, 3) and _sha(framed) == "b385c55c08011bf5"
    assert reference.silhouette_box(image) == (30, 50, 110, 170)


def test_export_declares_what_benchmark_sources_declare():
    with tempfile.TemporaryDirectory() as folder:
        path = atlas.export_glb(_quad(), _quad().visual.uv, np.zeros((8, 8, 3), dtype=np.uint8),
                                pathlib.Path(folder) / "out.glb")
        gltf = atlas.read_glb_json(path)
        assert sorted(gltf["meshes"][0]["primitives"][0]["attributes"]) == ["POSITION", "TEXCOORD_0"]
        assert gltf["materials"] == atlas.EXPECTED_MATERIALS
        assert gltf["images"][0]["mimeType"] == "image/png"
        loaded = atlas.load_textured_source(path)
        assert loaded.visual.material.baseColorTexture.size == (8, 8)
        plain = pathlib.Path(folder) / "plain.glb"
        mesh = _quad()
        mesh.vertex_normals
        plain.write_bytes(atlas.plain_glb_bytes(mesh))
        assert sorted(atlas.read_glb_json(plain)["meshes"][0]["primitives"][0]["attributes"]) == ["POSITION"]

        scene = trimesh.Scene([_quad(), _quad()])
        for geometry in scene.geometry.values():
            geometry.visual = trimesh.visual.TextureVisuals(
                uv=_quad().visual.uv, material=trimesh.visual.material.PBRMaterial(
                    baseColorTexture=Image.new("RGB", (4, 4))))
        two = pathlib.Path(folder) / "two.glb"
        scene.export(two)
        try:
            atlas.load_textured_source(two)
        except ValueError:
            return
        raise AssertionError("a source with two primitives was accepted")


def test_color_fit_loop_matches_research():
    classes = _fit_classes()
    with tempfile.TemporaryDirectory() as folder:
        record = color_fit.fit(_quad(), classes, classes["composite"], np.ones((32, 32), dtype=bool), (60.0, 30.0),
                               out_dir=pathlib.Path(folder),
                               render=lambda glb, out: _fake_render(glb, out, classes["generated"]))
    assert record["published_iteration"] == 5
    assert [round(h["residual"], 9) for h in record["history"]] == \
        [42.930016365, 30.485797706, 23.724888531, 14.758511077, 10.182014546, 8.318183508]
    assert [round(h["render_L"], 9) for h in record["history"]] == \
        [43.987345778, 52.875003519, 54.825179492, 55.651355279, 55.7062522, 55.298396372]


def test_color_fit_skips_too_few_pixels():
    classes = _fit_classes()
    pixels = np.zeros((32, 32), dtype=bool)
    pixels[:10, :10] = True
    record = color_fit.fit(_quad(), classes, classes["composite"], pixels, (60.0, 30.0), out_dir=pathlib.Path("."),
                           render=None)
    assert record["skipped"] and record["published_iteration"] == 0 and record["iterations"] == 0


def test_color_fit_job_keeps_composite_bytes_when_round_zero_wins():
    rng = np.random.default_rng(4)
    size = 16
    mapped = np.ones((size, size), dtype=bool)
    region = np.zeros((size, size), dtype=bool)
    region[:, :8] = True
    source = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    painted = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    blended = source.copy()
    blended[region] = painted[region]
    blended[:, 8] = ((source[:, 8].astype(int) + painted[:, 8].astype(int)) // 2).astype(np.uint8)
    with tempfile.TemporaryDirectory() as folder:
        root = pathlib.Path(folder)
        (root / "atlases").mkdir()
        for name, array in (("source_atlas", source), ("painted_atlas", painted), ("composite_atlas", blended),
                            ("region_mask", region), ("mapped", mapped)):
            Image.fromarray(array).save(root / "atlases" / f"{name}.png")
        composite_glb = atlas.export_glb(_quad(), _quad().visual.uv, blended, root / "composite.glb")
        classes = color_fit.texel_classes(root / "atlases")
        wanted = _fake_render(composite_glb, root / "probe", classes["generated"])
        rgb = np.asarray(Image.open(wanted).convert("RGB"))
        Image.fromarray(rgb).save(root / "edited.png")
        Image.fromarray(np.full((32, 32), 255, dtype=np.uint8)).save(root / "mask.png")
        job = {"atlas_dir": str(root / "atlases"), "composite_glb": str(composite_glb),
               "edited_view": str(root / "edited.png"), "mask": str(root / "mask.png"), "camera": {"view": 0},
               "out_dir": str(root / "fit"), "fitted_glb": str(root / "fitted.glb")}
        record = color_fit.fit_job(job, render=lambda glb, out, spec: _fake_render(glb, out, classes["generated"]))
        assert record["published_iteration"] == 0 and record["history"][0]["converged"]
        assert (root / "fitted.glb").read_bytes() == composite_glb.read_bytes()
        assert json.loads((root / "fit" / "fit.json").read_text())["published_iteration"] == 0
        assert not list((root / "fit").glob("iter*"))


def _sphere_uv(vertices: np.ndarray) -> np.ndarray:
    unit = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    return np.stack([0.5 + np.arctan2(unit[:, 2], unit[:, 0]) / (2 * np.pi), 0.5 + np.arcsin(unit[:, 1]) / np.pi],
                    axis=1)


def _textured(mesh: trimesh.Trimesh, seed: int) -> trimesh.Trimesh:
    image = Image.fromarray(np.random.default_rng(seed).integers(0, 256, (64, 64, 3), dtype=np.uint8))
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=image)
    return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False,
                           visual=trimesh.visual.TextureVisuals(uv=_sphere_uv(mesh.vertices), material=material))


def test_texture_steps_run_apart_as_texture_job_runs_them():
    import dataclasses

    from sap3d.config import load_texture_config
    from sap3d.texture import paint, texture

    def fake_paint(pipeline, mesh_glb_bytes, reference_png, work_dir):
        work_dir = pathlib.Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        plain = trimesh.load(trimesh.util.wrap_as_stream(mesh_glb_bytes), file_type="glb", force="mesh",
                             process=False)
        _textured(plain, 2).export(work_dir / "textured_mesh.glb")
        return work_dir / "textured_mesh.glb"

    cfg = dataclasses.replace(load_texture_config(None), atlas_size=128, target_faces=600)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
        _textured(sphere, 1).export(tmp / "source.glb")
        trimesh.Trimesh(vertices=sphere.vertices * 1.01, faces=sphere.faces).export(tmp / "edit.glb")
        view = np.full((96, 96, 3), 255, dtype=np.uint8)
        view[24:72, 30:66] = (200, 60, 40)
        Image.fromarray(view).save(tmp / "view.png")
        (tmp / "region.yaml").write_text(json.dumps(
            {"primitives": [{"type": "box", "center": [0.3, 0.0, 0.0], "extents": [0.3, 0.3, 0.3]}]}))
        job = {"mode": "composite", "mesh": str(tmp / "edit.glb"), "source": str(tmp / "source.glb"),
               "edited_view": str(tmp / "view.png"), "region": str(tmp / "region.yaml"), "view": None,
               "color_fit": False}
        original = paint.paint
        paint.paint = fake_paint
        try:
            whole = texture.texture_job(None, {**job, "out_dir": str(tmp / "whole")}, cfg, "model")
        finally:
            paint.paint = original
        state = texture.prepare_texture({**job, "out_dir": str(tmp / "apart")}, cfg)
        painted = fake_paint(None, state["paint_glb"], state["reference_png"], tmp / "apart" / "paint")
        apart = texture.finish_texture(state, painted, cfg, "model")
        assert whole["region_texels"] == apart["region_texels"] > 0
        for name in ("composite.glb", "reference.png", "atlases/source_atlas.png", "atlases/painted_atlas.png",
                     "atlases/composite_atlas.png", "atlases/region_mask.png"):
            assert (tmp / "whole" / name).read_bytes() == (tmp / "apart" / name).read_bytes(), name


def test_painted_obj_converts_to_glb_from_a_worker_thread():
    import subprocess
    import sys
    import threading

    from sap3d import third_party

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        _textured(trimesh.creation.icosphere(subdivisions=2, radius=0.4), 3).export(tmp / "painted.obj")
        outcome = {}

        def convert():
            outcome["code"] = subprocess.run(
                [sys.executable, "-m", "sap3d.texture.paint", str(tmp / "painted.obj"), str(tmp / "painted.glb")],
                env=third_party.child_env(), capture_output=True).returncode

        worker = threading.Thread(target=convert)
        worker.start()
        worker.join()
        assert outcome["code"] == 0
        converted = trimesh.load(str(tmp / "painted.glb"), force="mesh", process=False)
        assert converted.visual.material.baseColorTexture is not None and len(converted.faces) == 320
