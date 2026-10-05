"""One edit: source asset + adapter + edited view + 3D region + 2D mask -> edited mesh in the source frame.

A local edit first runs a preliminary uniform-strength generation that shares the edit's steps before
masking begins. Its mesh is searched for a quarter turn about the vertical relative to the source outside
the region, and the region is turned by the inverse before the probes read it, so a generation that lost
the source's pose is still edited in the right place. The finished mesh gets the same search. When no
surface sample of the source lies outside the region, both searches would decline, so the preliminary
pass is skipped.
"""

import dataclasses
import time

import numpy as np

from .. import conditioning, distance, frames, orient
from ..region import place_components, points_inside, sample_volume_points
from .patches import mask_patch_tokens, processor_framing
from .sampler import decode, sample_local, sample_uniform


def _default_remover():
    from hy3dshape.rembg import BackgroundRemover
    return BackgroundRemover()


def run_edit(pipeline, cfg, source_path, edited_view, region=None, mask=None, make_remover=_default_remover):
    """(edited mesh in the source frame, record, per-step probe records).

    `region` is a `region.Region` and `mask` a single-channel mask (path, image or boolean array) of the
    edited view's size; uniform mode reads neither.
    """
    started = time.time()
    source = frames.load_mesh(source_path)
    frame = frames.frame_of(source)
    cond_image, segmenter = conditioning.conditioning_image(edited_view, make_remover, cfg.conditioning)
    record = {"mode": cfg.mode, "frame": frame.as_dict(),
              "conditioning": "edited view used as given (real transparency)" if segmenter is None
              else f"background removed ({segmenter}) with the brightness floor"}

    if cfg.mode == "uniform":
        produced = frames.to_source_mesh(decode(pipeline, sample_uniform(pipeline, cond_image, cfg), cfg), frame)
        record.update(faces=int(len(produced.faces)), seconds=round(time.time() - started, 2))
        return produced, record, []

    if region is None or mask is None:
        raise ValueError("a local edit needs a 3D region and a 2D mask")
    pose = cfg.pose
    sdf = distance.region_sdf(region.surface)
    search = pose.enabled and len(distance.sample_outside(source, sdf, pose.surface_points, pose.sample_seed)) > 0
    region_turn = 0
    if search:
        preliminary = decode(pipeline, sample_uniform(
            pipeline, cond_image, _uniform_at(cfg, cfg.schedule.pre_mask_strength)), cfg)
        degrees, record["preliminary_orientation"] = orient.best_orientation(
            frames.to_source_mesh(preliminary, frame), source, sdf, frame.center,
            min_improvement=pose.region_min_improvement, num_points=pose.surface_points, seed=pose.sample_seed)
        record["preliminary_faces"] = int(len(preliminary.faces))
        region_turn = orient.inverse_turn(degrees)
    else:
        record["preliminary_orientation"] = ("disabled" if not pose.enabled else
                                             "skipped: no source surface sample lies outside the region")
    record["region_turn_degrees"] = region_turn

    components = place_components(region, frame, region_turn)
    query_coords = sample_volume_points(components, cfg.probe_3d.query_points, cfg.probe_3d.query_seed)
    if not points_inside(components, query_coords).all():
        raise ValueError("a 3D probe query point fell outside the region it was sampled in")

    border_ratio, frame_size = processor_framing(pipeline.image_processor)
    patches, mask_stats = mask_patch_tokens(mask, cond_image, border_ratio=border_ratio, frame_size=frame_size,
                                            coverage=cfg.probe_2d.coverage)
    if not patches:
        raise ValueError(f"no conditioning-image patch is covered at least {cfg.probe_2d.coverage} by the "
                         f"2D mask, so the 2D probe has nothing to read")

    latents, steps = sample_local(pipeline, cond_image, cfg, query_coords, patches)
    produced = frames.to_source_mesh(decode(pipeline, latents, cfg), frame)
    if search:
        degrees, record["output_orientation"] = orient.best_orientation(
            produced, source, sdf, frame.center, min_improvement=pose.output_min_improvement,
            num_points=pose.surface_points, seed=pose.sample_seed)
        produced = orient.turn(produced, degrees, frame.center)
        record["output_turn_degrees"] = degrees
    else:
        record["output_orientation"] = record["preliminary_orientation"]
        record["output_turn_degrees"] = 0

    record.update(
        region=region.description,
        region_components=len(components),
        query_points=int(len(query_coords)),
        patches_selected=len(patches),
        **mask_stats,
        region_tokens_per_step={str(s["step"]): len(s["region"]) for s in steps},
        final_region_tokens=len(steps[-1]["region"]) if steps else 0,
        faces=int(len(produced.faces)),
        seconds=round(time.time() - started, 2),
    )
    return produced, record, steps


def _uniform_at(cfg, strength: float):
    return dataclasses.replace(cfg, uniform_strength=float(strength))


def steps_to_npz(steps: list) -> dict:
    """Per-step token sets as arrays, keyed `step_SS_points`, `step_SS_patches`, `step_SS_region`."""
    arrays = {}
    for s in steps:
        for key in ("points", "patches", "region"):
            arrays[f"step_{s['step']:02d}_{key}"] = np.asarray(s[key], dtype=np.int64)
    return arrays
