"""Texture one or more edited meshes; the paint model is loaded once per process.

`composite` (local edits): the source asset's texture is copied onto the edited mesh's own atlas, the paint
model paints the whole mesh from the recomposed edited view, and the painted texels replace the copied ones
inside the region (see `composite`). The color fit then runs in a child process.

`generated` (global edits): the painted atlas alone.

A job is a dict: `mode`, `mesh` (edited mesh, source frame), `source` (textured source GLB), `edited_view`,
`region` (region file; composite only), `view` (`{"mask": PNG, "camera": sap3d.render camera spec}` of the
edited view, or None), `color_fit` (bool; needs `view`) and `out_dir`. Outputs in `out_dir`: `composite.glb`
or `generated.glb`, `texture.json`, `reference.png`, `atlases/`, `paint/`, and with a color fit
`composite_color_fit.glb` and `color_fit/`.
"""

import dataclasses
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from .. import third_party
from ..region import load_region
from . import atlas, color_fit, composite, paint, reference, texels

MODES = ("composite", "generated")

# The painted mesh and the atlas mesh are one surface, so any texel farther than this fraction of the
# bounding-box diagonal (at the 99.9th percentile) means a frame mismatch or a remesh that altered it.
PAINTED_RELATIVE_LIMIT = 1e-3

OUTPUT_NAMES = {"composite": "composite.glb", "generated": "generated.glb"}
FITTED_NAME = "composite_color_fit.glb"


def digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def job_inputs(job: dict) -> dict:
    """Digests of every input file a job reads, and its camera, so a stale output can be recognized."""
    files = {key: job[key] for key in ("mesh", "source", "edited_view", "region") if job.get(key)}
    if job.get("view"):
        files["mask"] = job["view"]["mask"]
    inputs = {key: digest(path) for key, path in files.items()}
    if job.get("view"):
        inputs["camera"] = job["view"]["camera"]
    return inputs


def settings(cfg, model: str) -> dict:
    return {"texture": dataclasses.asdict(cfg), "paint_model": model, "paint_max_num_view": paint.MAX_NUM_VIEW,
            "paint_resolution": paint.RESOLUTION, "paint_use_remesh": paint.USE_REMESH,
            "reference": {"frame": reference.FRAME, "fill_fraction": reference.FILL_FRACTION,
                          "pad_level": reference.PAD_LEVEL},
            "feather_depth_clamp": composite.FEATHER_DEPTH_CLAMP,
            "color_fit": {"max_iterations": color_fit.MAX_ITERATIONS, "tolerance": color_fit.TOLERANCE,
                          "min_fit_pixels": color_fit.MIN_FIT_PIXELS}}


def _write_png(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array).save(path, format="PNG", optimize=True)


def _copied_from_region(position, mapped, region_mask, source, region_mesh) -> tuple:
    """Texels outside the region whose copied color was read from a point inside it, and how far every
    texel outside the region reached for its color (fractions of the source's bounding-box diagonal)."""
    outside = mapped & ~region_mask
    out = np.zeros(mapped.shape, dtype=bool)
    if not outside.any():
        return out, {"n_outside": 0}
    closest = atlas.closest_source_points(position, outside, source)
    out[outside] = texels.points_inside(closest, region_mesh)
    relative = (np.linalg.norm(closest - position[outside].astype(np.float64), axis=1)
                / float(np.linalg.norm(source.extents)))
    return out, {"n_outside": int(outside.sum()), "relative_p50": float(np.percentile(relative, 50)),
                 "relative_p99": float(np.percentile(relative, 99)), "relative_max": float(relative.max()),
                 "high_fraction_1pct_diag": float((relative > 0.01).mean())}


def prepare_texture(job: dict, cfg) -> dict:
    """Everything before painting (CPU only): the atlas of the decimated edited mesh, its surface maps and,
    in composite mode, the copied source texture and the region's texels. Returns the state
    `finish_texture` reads; the paint model reads `state["paint_glb"]` (GLB bytes) and
    `state["reference_png"]`."""
    mode = job["mode"]
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "composite" and not job.get("region"):
        raise ValueError("composite mode needs the edit region")
    if job["color_fit"] and (mode != "composite" or not job.get("view")):
        raise ValueError("the color fit corrects a composite's painted region through the edited view's camera")
    started = time.time()
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("texture.json", *OUTPUT_NAMES.values(), FITTED_NAME):
        (out_dir / name).unlink(missing_ok=True)
    atlas_dir = out_dir / "atlases"
    atlas_dir.mkdir(exist_ok=True)

    raw = trimesh.load(str(job["mesh"]), force="mesh", process=False)
    source = atlas.load_textured_source(job["source"])
    decimated = atlas.decimate(raw, cfg.target_faces)
    work, normals = atlas.unwrap(decimated)
    position, normal, mapped = atlas.bake_surface_maps(work, cfg.atlas_size, normals)
    record = {"mode": mode, "faces_before": int(len(raw.faces)), "faces": int(len(work.faces)),
              "vertices": int(len(work.vertices))}
    _write_png(mapped, atlas_dir / "mapped.png")
    state = {"job": job, "started": started, "out_dir": out_dir, "atlas_dir": atlas_dir, "record": record,
             "work": work, "position": position, "normal": normal, "mapped": mapped}

    if mode == "composite":
        region_mesh = texels.region_mesh(load_region(job["region"]).surface)
        source_rgb, record["source_transfer"] = atlas.transfer_texture(position, mapped, normal, source,
                                                                       cfg.normal_gate)
        region_mask = texels.region_mask(position, mapped, region_mesh)
        copied, record["source_transfer_outside_region"] = _copied_from_region(
            position, mapped, region_mask, source, region_mesh)
        record["region_texels"] = int(region_mask.sum())
        record["region_fraction_of_mapped"] = round(float(region_mask.sum() / mapped.sum()), 6)
        _write_png(source_rgb, atlas_dir / "source_atlas.png")
        _write_png(region_mask, atlas_dir / "region_mask.png")
        _write_png(copied, atlas_dir / "copied_from_region.png")
        state.update(region_mesh=region_mesh, source_rgb=source_rgb, region_mask=region_mask, copied=copied)

    reference_png = out_dir / "reference.png"
    reference.write_reference(job["edited_view"], reference_png)
    state.update(reference_png=reference_png, paint_glb=atlas.plain_glb_bytes(decimated))
    return state


def finish_texture(state: dict, painted_glb, cfg, model: str) -> dict:
    """Everything after painting (CPU only): the painted texels on the atlas, the composite, the GLB and
    `texture.json`. Returns the record."""
    job, out_dir, atlas_dir, record = state["job"], state["out_dir"], state["atlas_dir"], state["record"]
    position, normal, mapped = state["position"], state["normal"], state["mapped"]
    painted_mesh = trimesh.load(str(painted_glb), force="mesh", process=False)
    painted_rgb, record["painted_transfer"] = atlas.transfer_texture(position, mapped, normal, painted_mesh,
                                                                     cfg.normal_gate)
    if record["painted_transfer"]["relative_p99.9"] >= PAINTED_RELATIVE_LIMIT:
        raise RuntimeError(f"{job['mesh']}: the painted mesh sits {record['painted_transfer']['relative_p99.9']:.3e} "
                           f"of a bounding-box diagonal from the atlas mesh at the 99.9th percentile; the two "
                           f"should be one surface")
    spread = painted_rgb[mapped].astype(np.float64).std(axis=0)
    if spread.min() <= 0.0:
        raise RuntimeError(f"{job['mesh']}: the painted atlas has a constant channel (std {spread.tolist()})")
    _write_png(painted_rgb, atlas_dir / "painted_atlas.png")

    if job["mode"] == "composite":
        depth = texels.signed_depth(position, mapped, state["region_mask"], state["region_mesh"])
        published, record["composite"] = composite.composite(
            state["source_rgb"], painted_rgb, state["region_mask"], depth=depth, depth_gate=mapped,
            copied_from_region=state["copied"], feather_width=cfg.feather_width,
            boundary_margin=cfg.boundary_margin)
        np.savez_compressed(atlas_dir / "region_signed_depth.npz", depth=depth)
        _write_png(published, atlas_dir / "composite_atlas.png")
    else:
        published = painted_rgb
    work = state["work"]
    atlas.export_glb(work, work.visual.uv, published, out_dir / OUTPUT_NAMES[job["mode"]])

    record.update(seconds=round(time.time() - state["started"], 1), inputs=job_inputs(job),
                  settings=settings(cfg, model),
                  job={key: (str(value) if isinstance(value, Path) else value) for key, value in job.items()})
    (out_dir / "texture.json").write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    return record


def texture_job(pipeline, job: dict, cfg, model: str) -> dict:
    """Paint and assemble one job's texture; returns its record (also written to `texture.json`)."""
    state = prepare_texture(job, cfg)
    painted_glb = paint.paint(pipeline, state["paint_glb"], state["reference_png"], state["out_dir"] / "paint")
    return finish_texture(state, painted_glb, cfg, model)


def color_fit_jobs(jobs: list) -> list:
    return [{"atlas_dir": str(Path(job["out_dir"]).resolve() / "atlases"),
             "composite_glb": str(Path(job["out_dir"]).resolve() / OUTPUT_NAMES["composite"]),
             "edited_view": str(Path(job["edited_view"]).resolve()),
             "mask": str(Path(job["view"]["mask"]).resolve()), "camera": job["view"]["camera"],
             "out_dir": str(Path(job["out_dir"]).resolve() / "color_fit"),
             "fitted_glb": str(Path(job["out_dir"]).resolve() / FITTED_NAME)}
            for job in jobs if job.get("color_fit")]


def run(jobs: list, cfg, work_dir: Path, model: str = "tencent/Hunyuan3D-2.1") -> None:
    """Texture every job, then color-fit those that ask for it in one child process."""
    if not jobs:
        return
    pipeline = paint.load_pipeline(cfg.realesrgan_ckpt or None, model)
    for position, job in enumerate(jobs, start=1):
        record = texture_job(pipeline, job, cfg, model)
        print(f"textured [{position}/{len(jobs)}] {job['out_dir']}  {record['mode']}  faces={record['faces']}  "
              f"({record['seconds']} s)", flush=True)
    fits = color_fit_jobs(jobs)
    if fits:
        third_party.run_module("sap3d.texture.color_fit", fits, Path(work_dir) / "color_fit_jobs.json")
