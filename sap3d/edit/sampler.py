"""The per-token adapter schedule of a local edit, and the uniform-strength generation of a global edit.

Local edit, steps indexed from 0:
    before mask_from_step          every token at pre_mask_strength
    mask_from_step .. drop - 1     tokens in the region at inside_strength, all others (and the condition
                                   token at sequence position 0) at outside_strength
    from drop_from_step            every token at inside_strength
The region is empty at mask_from_step. After each forward pass from mask_from_step to drop_from_step - 1
both probes run and their union replaces the region from the next step on; an empty union keeps the
previous region, and the last probe's region is never applied.
"""

from typing import NamedTuple

import numpy as np
import torch

from hy3dshape.models.per_sample_lora import LoRAScaleContext
from hy3dshape.pipelines import retrieve_timesteps
from hy3dshape.utils.attention_capture import install_attention_storage, remove_attention_storage

from .probes import HeadAttentionSum, patch_region, point_region

NUM_LATENTS = 4096


class Knee(NamedTuple):
    factor: float
    width: int
    sensitivity: float


def step_regime(step: int, mask_from_step: int, drop_from_step: int) -> str:
    if step < mask_from_step:
        return "pre_mask"
    return "per_token" if step < drop_from_step else "dropped"


def build_latent_mask(region: set, outside: float, inside: float, device, dtype) -> torch.Tensor:
    """[1, NUM_LATENTS + 1] strengths; position 0 is the condition token and always takes `outside`."""
    mask = torch.full((1, NUM_LATENTS + 1), outside, device=device, dtype=dtype)
    if region:
        mask[0, torch.tensor(sorted(region), device=device) + 1] = inside
    return mask


def _check_guidance(pipeline):
    if getattr(pipeline.model, "guidance_embed", False) is True:
        raise ValueError("a DiT with a guidance embedding is not supported; the edit uses classifier-free guidance")


def _encode(pipeline, cond_image):
    cond_inputs = pipeline.prepare_image(cond_image, mask=None)
    image = cond_inputs.pop("image")
    return pipeline.encode_cond(image=image, additional_cond_inputs=cond_inputs,
                                do_classifier_free_guidance=True, dual_guidance=False)


@torch.inference_mode()
def sample_local(pipeline, cond_image, cfg, query_coords: np.ndarray, patches: list) -> tuple:
    """(final latents [1, 4096, 64], one record per probe step) for a local edit."""
    _check_guidance(pipeline)
    sampler, schedule = cfg.sampler, cfg.schedule
    device, dtype = pipeline.device, pipeline.dtype
    knee_3d = Knee(cfg.probe_3d.knee_factor, cfg.knee.smoothing_width, cfg.knee.sensitivity)
    knee_2d = Knee(cfg.probe_2d.knee_factor, cfg.knee.smoothing_width, cfg.knee.sensitivity)

    generator = torch.Generator(device=device).manual_seed(sampler.seed)
    latents = pipeline.prepare_latents(batch_size=1, dtype=dtype, device=device, generator=generator)
    cond = _encode(pipeline, cond_image)

    timesteps, _ = retrieve_timesteps(pipeline.scheduler, sampler.num_steps, device,
                                      sigmas=np.linspace(0, 1, sampler.num_steps))
    pipeline.scheduler._step_index = 0

    pre_mask = torch.tensor([schedule.pre_mask_strength] * 2, device=device, dtype=dtype)
    dropped = torch.tensor([schedule.inside_strength] * 2, device=device, dtype=dtype)
    region = set()
    latent_mask = build_latent_mask(region, schedule.outside_strength, schedule.inside_strength, device, dtype)
    records = []

    for step, t in enumerate(timesteps):
        regime = step_regime(step, schedule.mask_from_step, schedule.drop_from_step)
        if regime == "pre_mask":
            scales = pre_mask
        elif regime == "dropped":
            scales = dropped
        else:
            scales = torch.cat([latent_mask] * 2, dim=0)

        latent_model_input = torch.cat([latents] * 2)
        timestep = t.expand(latent_model_input.shape[0]).to(dtype)
        timestep = timestep / pipeline.scheduler.config.num_train_timesteps

        storage = HeadAttentionSum(cfg.probe_2d.heads) if regime == "per_token" else None
        token = install_attention_storage(storage) if storage is not None else None
        LoRAScaleContext.set(scales)
        try:
            noise_pred = pipeline.model(latent_model_input, timestep, cond, guidance=None)
        finally:
            LoRAScaleContext.clear()
            if token is not None:
                remove_attention_storage(token)

        noise_cond, noise_uncond = noise_pred.chunk(2)
        velocity = noise_uncond + sampler.guidance_scale * (noise_cond - noise_uncond)

        if regime == "per_token":
            sigma = float(pipeline.scheduler.sigmas[pipeline.scheduler.step_index])
            x1_hat = latents + (1.0 - sigma) * velocity
            from_points = point_region(pipeline.vae, x1_hat, query_coords, knee_3d)
            from_patches = patch_region(storage, patches, knee_2d)
            found = from_points | from_patches
            if found:
                region = found
                latent_mask = build_latent_mask(region, schedule.outside_strength, schedule.inside_strength,
                                                device, dtype)
            records.append({"step": step, "points": sorted(from_points), "patches": sorted(from_patches),
                            "region": sorted(region), "kept_previous": not found,
                            "applied_from_step": step + 1 if step + 1 < schedule.drop_from_step else None})

        latents = pipeline.scheduler.step(velocity, t, latents).prev_sample

    return latents, records


def sample_uniform(pipeline, cond_image, cfg) -> torch.Tensor:
    """Final latents of a generation with one adapter strength for every token at every step."""
    _check_guidance(pipeline)
    sampler = cfg.sampler
    generator = torch.Generator(device=pipeline.device).manual_seed(sampler.seed)
    return pipeline(image=cond_image, num_inference_steps=sampler.num_steps,
                    guidance_scale=sampler.guidance_scale, output_type="latent", enable_pbar=False,
                    generator=generator, lora_scales=[cfg.uniform_strength])


def decode(pipeline, latents: torch.Tensor, cfg):
    """The mesh of `latents` in the model frame, or an error when the decoder finds no surface."""
    result = pipeline._export(latents.to(pipeline.device), output_type="trimesh",
                              octree_resolution=cfg.sampler.octree_resolution,
                              num_chunks=cfg.sampler.num_chunks, enable_pbar=False)
    mesh = result[0] if isinstance(result, list) else result
    if mesh is None or len(mesh.faces) == 0:
        raise ValueError("surface extraction failed or found no surface in the generated latents; any "
                         "extraction error is printed above as a traceback")
    return mesh
