"""Fit the painted region's color to the edited view, measured on renders.

The fitted quantity is the mean CIELAB lightness and chroma over the annotated pixels of a render through
the edited view's camera, against the same over the annotated pixels of the edited view; both sides are
lit, so the atlas is corrected by what the renderer makes of it. Only pixels that show purely painted
surface (read off a render of an atlas painted flat by texel origin) and whose target is not near-white
enter the comparison. Lightness is shifted and chroma scaled on the painted and blended texels; hue is
never touched.

Up to MAX_ITERATIONS rounds run, stopping when both channels are within max(TOLERANCE, standard error of
the fitted mean) or after STALL_LIMIT rounds without improvement. The round with the smallest residual is
published, the uncorrected atlas competing as round 0; when round 0 wins, the color-fitted mesh is a copy
of the composite's bytes. Fewer than MIN_FIT_PIXELS usable pixels publish the composite uncorrected.

Runs in its own process, since rendering changes the working directory:
`python -m sap3d.texture.color_fit JOBS.json`, each job naming `atlas_dir`, `composite_glb`,
`edited_view`, `mask`, `camera` (a `sap3d.render` camera spec), `out_dir` and `fitted_glb`.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from skimage.color import lab2rgb, rgb2lab

from .. import render as renders
from .atlas import export_glb
from .composite import annotated_pixels, edit_region_pixels

MAX_ITERATIONS = 6

# CIELAB units below which a channel counts as fitted whatever its sampling noise.
TOLERANCE = 0.25

MIN_FIT_PIXELS = 1024

# Bounds on the render's measured response to an atlas change; outside them the estimate is not used.
GAIN_RANGE = (0.2, 3.0)

# The most one round may divide or multiply the chroma scale by.
STEP_RANGE = (0.2, 2.0)

# The largest CIELAB chroma an 8-bit sRGB color has; a mean chroma above it names no color.
MAX_SRGB_CHROMA = 133.8

# How much the atlas must have moved across the previous round before its response is estimated:
# lightness in CIELAB units, chroma as an absolute log ratio.
ATLAS_LIGHTNESS_TRIGGER = 0.5
ATLAS_CHROMA_TRIGGER = 0.02

MIN_CHROMA = 0.5
STALL_LIMIT = 2

LABEL_COLORS = {"generated": (255, 0, 0), "blended": (0, 255, 0), "copied": (0, 0, 255)}


def mean_lab(rgb: np.ndarray) -> np.ndarray:
    return rgb2lab(np.asarray(rgb, dtype=np.float64).reshape(-1, 1, 3) / 255.0).reshape(-1, 3).mean(0)


def lab_of(rgb: np.ndarray) -> np.ndarray:
    return rgb2lab(np.asarray(rgb, dtype=np.float64).reshape(-1, 1, 3) / 255.0).reshape(-1, 3)


def summarize(mean: np.ndarray) -> tuple:
    return float(mean[0]), float(np.hypot(mean[1], mean[2]))


def to_rgb(lab: np.ndarray) -> np.ndarray:
    out = lab2rgb(lab.reshape(-1, 1, 3)).reshape(-1, 3) * 255.0
    return np.clip(out, 0, 255).round().astype(np.uint8)


def apply_shift(lab: np.ndarray, lightness_shift: float, chroma_scale: float) -> np.ndarray:
    out = lab.copy()
    out[:, 0] = np.clip(lab[:, 0] + lightness_shift, 0.0, 100.0)
    out[:, 1:] = lab[:, 1:] * chroma_scale
    return out


def standard_error(values: np.ndarray) -> float:
    return float(values.std() / np.sqrt(len(values))) if len(values) else float("inf")


def texel_classes(atlas_dir: Path) -> dict:
    """Which texels were painted, copied, or blended, and each blended texel's mixing fraction."""
    atlas_dir = Path(atlas_dir)
    read = lambda name: np.asarray(Image.open(atlas_dir / name))
    source = np.asarray(Image.open(atlas_dir / "source_atlas.png").convert("RGB")).astype(np.float64)
    painted = np.asarray(Image.open(atlas_dir / "painted_atlas.png").convert("RGB")).astype(np.float64)
    composite = np.asarray(Image.open(atlas_dir / "composite_atlas.png").convert("RGB"))
    region = read("region_mask.png") > 0
    mapped = read("mapped.png") > 0

    generated = region & mapped & (composite.astype(np.float64) == painted).all(axis=-1)
    copied = (~region) & mapped & (composite.astype(np.float64) == source).all(axis=-1)
    ramp = mapped & ~generated & ~copied

    spread = painted[ramp] - source[ramp]
    offset = composite[ramp].astype(np.float64) - source[ramp]
    denominator = (spread * spread).sum(axis=1)
    usable = denominator > 1e-6
    weight = np.zeros(len(spread))
    weight[usable] = np.clip((spread[usable] * offset[usable]).sum(axis=1) / denominator[usable], 0.0, 1.0)
    return {"generated": generated, "ramp": ramp, "copied": copied, "region": region, "composite": composite,
            "source_at_ramp": source[ramp], "ramp_weight": weight,
            "painted_at_ramp": np.clip(painted[ramp], 0, 255).round().astype(np.uint8)}


def label_atlas(classes: dict) -> np.ndarray:
    array = np.zeros_like(classes["composite"])
    array[classes["generated"]] = LABEL_COLORS["generated"]
    array[classes["ramp"]] = LABEL_COLORS["blended"]
    array[classes["copied"]] = LABEL_COLORS["copied"]
    return array


def label_pixels(label_png: Path) -> dict:
    """Which origin each rendered pixel shows, by its largest channel."""
    rgba = np.asarray(Image.open(label_png).convert("RGBA")).astype(np.int32)
    on_object = rgba[..., 3] > 0
    winner = rgba[..., :3].argmax(axis=-1)
    return {"on_object": on_object, "generated": on_object & (winner == 0),
            "blended": on_object & (winner == 1), "copied": on_object & (winner == 2)}


def write_atlas_glb(mesh, classes: dict, generated_rgb: np.ndarray, ramp_rgb: np.ndarray, path: Path) -> Path:
    array = classes["composite"].copy()
    array[classes["generated"]] = generated_rgb
    blended = (classes["ramp_weight"][:, None] * ramp_rgb.astype(np.float64)
               + (1.0 - classes["ramp_weight"][:, None]) * classes["source_at_ramp"])
    array[classes["ramp"]] = np.clip(blended, 0, 255).round().astype(np.uint8)
    return export_glb(mesh, mesh.visual.uv, array, path)


def published_atlas(composite_glb: Path, classes: dict):
    mesh = trimesh.load(str(composite_glb), force="mesh", process=False)
    atlas = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"))
    if not np.array_equal(atlas, classes["composite"]):
        raise ValueError(f"{composite_glb} carries a texture other than the saved composite atlas, so its texels "
                         f"cannot be addressed by the saved rasters")
    return mesh, atlas


def fit(mesh, classes: dict, atlas: np.ndarray, fit_pixels: np.ndarray, wanted: tuple, *, out_dir: Path,
        render) -> dict:
    """Iterate the atlas toward `wanted` (lightness, chroma); `render(glb, out_dir)` returns the PNG."""
    out_dir = Path(out_dir)
    wanted_L, wanted_C = wanted
    movable = int(fit_pixels.sum())
    if movable < MIN_FIT_PIXELS:
        return {"skipped": True, "reason": "too few pixels show painted surface", "fit_pixels": movable,
                "iterations": 0, "converged": False, "history": [], "published_iteration": 0}

    generated_lab = lab_of(atlas[classes["generated"]])
    ramp_lab = lab_of(classes["painted_at_ramp"])
    base_chroma = float(np.hypot(generated_lab[:, 1].mean(), generated_lab[:, 2].mean()))
    scale_ceiling = MAX_SRGB_CHROMA / base_chroma if base_chroma > 0 else float("inf")

    shift, scale = 0.0, 1.0
    history, previous, best, stalled = [], None, 0, 0
    for step in range(MAX_ITERATIONS):
        current_generated = apply_shift(generated_lab, shift, scale)
        current_ramp = apply_shift(ramp_lab, shift, scale)
        directory = out_dir / f"iter{step:02d}"
        glb = write_atlas_glb(mesh, classes, to_rgb(current_generated), to_rgb(current_ramp),
                              directory / "edit.glb")
        render_png = Path(render(glb, directory))

        rgba = np.asarray(Image.open(render_png).convert("RGBA"))
        selected = fit_pixels & (rgba[..., 3] > 0)
        lab = lab_of(rgba[..., :3][selected])
        render_L, render_C = summarize(lab.mean(axis=0))
        noise_L = standard_error(lab[:, 0])
        noise_C = standard_error(np.hypot(lab[:, 1], lab[:, 2]))
        tolerance_L, tolerance_C = max(TOLERANCE, noise_L), max(TOLERANCE, noise_C)
        atlas_L, atlas_C = summarize(current_generated.mean(axis=0))
        error_L, error_C = render_L - wanted_L, render_C - wanted_C
        residual = abs(error_C) + abs(error_L)
        entry = {"iteration": step, "atlas_shift": shift, "atlas_chroma_scale": scale, "atlas_L": atlas_L,
                 "atlas_C": atlas_C, "render_L": render_L, "render_C": render_C, "error_L": error_L,
                 "error_C": error_C, "residual": residual, "noise_L": noise_L, "noise_C": noise_C,
                 "selected_pixels": int(selected.sum()), "glb": str(glb), "render": str(render_png)}
        history.append(entry)

        if step > 0:
            if residual < history[best]["residual"]:
                best, stalled = step, 0
            else:
                stalled += 1
        entry["converged"] = bool(abs(error_C) <= tolerance_C and abs(error_L) <= tolerance_L)
        if entry["converged"] or step == MAX_ITERATIONS - 1 or stalled >= STALL_LIMIT:
            break

        lightness_gain, chroma_gain = 1.0, 1.0
        if previous is not None:
            atlas_step = atlas_L - previous["atlas_L"]
            render_step = render_L - previous["render_L"]
            if abs(atlas_step) > ATLAS_LIGHTNESS_TRIGGER and abs(render_step) > previous["noise_L"]:
                lightness_gain = float(np.clip(render_step / atlas_step, *GAIN_RANGE))
            atlas_ratio = atlas_C / previous["atlas_C"] if previous["atlas_C"] > 0 else 1.0
            if (abs(np.log(atlas_ratio)) > ATLAS_CHROMA_TRIGGER
                    and abs(render_C - previous["render_C"]) > previous["noise_C"]
                    and previous["render_C"] > MIN_CHROMA and render_C > MIN_CHROMA):
                chroma_gain = float(np.clip(np.log(render_C / previous["render_C"]) / np.log(atlas_ratio),
                                            *GAIN_RANGE))
        previous = entry
        shift = shift - error_L / lightness_gain
        if render_C > MIN_CHROMA:
            scale = min(scale * float(np.clip((wanted_C / render_C) ** (1.0 / chroma_gain), *STEP_RANGE)),
                        scale_ceiling)

    chosen = history[best]
    record = {"skipped": best == 0, "fit_pixels": movable, "wanted_L": wanted_L, "wanted_C": wanted_C,
              "iterations": len(history), "converged": bool(chosen["converged"]), "history": history,
              "published_iteration": best, "uncorrected_residual": history[0]["residual"],
              "published_residual": chosen["residual"]}
    if best == 0:
        record["reason"] = ("the uncorrected atlas already agrees within tolerance" if history[0]["converged"]
                            else "no round improved on the uncorrected atlas")
    return record


def fit_job(job: dict, render=renders.render) -> dict:
    """One color fit; writes `fitted_glb` and `out_dir/fit.json`."""
    out_dir, fitted = Path(job["out_dir"]).resolve(), Path(job["fitted_glb"]).resolve()
    composite_glb = Path(job["composite_glb"]).resolve()
    atlas_dir, edited_view, mask = (Path(job[key]).resolve() for key in ("atlas_dir", "edited_view", "mask"))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    fitted.unlink(missing_ok=True)
    spec = job["camera"]
    camera_render = lambda glb, directory: render(Path(glb), Path(directory), spec)

    classes = texel_classes(atlas_dir)
    mesh, atlas = published_atlas(composite_glb, classes)
    label_glb = export_glb(mesh, mesh.visual.uv, label_atlas(classes), out_dir / "label" / "edit.glb")
    label_png = camera_render(label_glb, out_dir / "label")
    rgb2d, eligible = edit_region_pixels(edited_view, mask)
    fit_pixels = annotated_pixels(mask) & label_pixels(label_png)["generated"] & eligible
    wanted = (summarize(mean_lab(rgb2d[fit_pixels])) if fit_pixels.sum() >= MIN_FIT_PIXELS
              else (float("nan"), float("nan")))
    record = fit(mesh, classes, atlas, fit_pixels, wanted, out_dir=out_dir, render=camera_render)
    record.update(camera=spec, generated_texels=int(classes["generated"].sum()),
                  ramp_texels=int(classes["ramp"].sum()))

    if record["published_iteration"] == 0:
        shutil.copyfile(composite_glb, fitted)
    else:
        chosen = record["history"][record["published_iteration"]]
        shutil.copyfile(chosen["glb"], fitted)
        shutil.copyfile(chosen["render"], out_dir / Path(chosen["render"]).name)
        record["render"] = str(out_dir / Path(chosen["render"]).name)
    for directory in sorted(out_dir.glob("iter[0-9][0-9]")):
        shutil.rmtree(directory)
    record["fitted_glb"] = str(fitted)
    (out_dir / "fit.json").write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    return record


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit("usage: python -m sap3d.texture.color_fit JOBS.json")
    for job in json.loads(Path(argv[0]).read_text(encoding="utf-8")):
        record = fit_job(job)
        print(f"color fit {job['out_dir']}: published round {record['published_iteration']} of "
              f"{record['iterations']}", flush=True)


if __name__ == "__main__":
    main()
