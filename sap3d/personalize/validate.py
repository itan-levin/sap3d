"""Validation during training: one generated mesh against the asset's own surface points.

At each validation a batch is drawn from a validation stream separate from the training one, one of its items
is chosen, and the shape model generates a mesh from that item's render. Its Chamfer distance to the item's
surface points is reported under two definitions:

- corrected: the surface is extracted as the edit stage does (DMC on a CUDA grid) and compared as it is;
- legacy: marching cubes with vertices mapped through octree_resolution + 1 grid points per axis, the
  prediction scaled by 0.9999 / 1.01, and the smallest distance over four quarter turns about z (chosen on the
  squared distance, reported unsquared).

Both are the mean, over the two directions, of nearest-neighbor distances between 50,000 area-weighted mesh
points and 50,000 of the item's points.
"""

import math

import numpy as np
import torch
from scipy.spatial import cKDTree
from skimage import measure

from hy3dshape.models.autoencoders.surface_extractors import SurfaceExtractor

from ..config import ValidationConfig
from .data import SampleStream

BOUNDS = 1.01
MC_LEVEL = 0.0
FAILED = math.sqrt(12.0)
PRED_SEED, GT_SEED = 42, 43
LEGACY_RESCALE = 0.9999 / 1.01


class ValidationStream:
    """The batches and batch items validated at each validation, in order.

    Renders come from the sample stream's own `random.Random(seed)`, one batch per validation. The item comes
    from `numpy.random.RandomState(seed)`: the first validation (before training) takes the first draw; the
    generator then restarts, and every later validation takes the second of two draws.
    """

    def __init__(self, sample_stream: SampleStream, batch_size: int, seed: int = 0):
        self.samples, self.batch_size, self.seed = sample_stream, batch_size, seed
        self.items = np.random.RandomState(seed)
        self.count = 0

    def next(self):
        batch = self.samples.batch(self.batch_size)
        if self.count == 1:
            self.items = np.random.RandomState(self.seed)
        if self.count >= 1:
            self.items.choice(self.batch_size, 1, replace=False)
        item = int(self.items.choice(self.batch_size, 1, replace=False)[0])
        self.count += 1
        return batch, item


def sample_mesh_surface(vertices, faces, count: int, seed: int):
    """`count` area-weighted points on the mesh, or None for an empty or degenerate mesh."""
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if v.shape[0] == 0 or f.shape[0] == 0 or np.any(np.isnan(v)):
        return None
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    if areas.sum() == 0:
        return None
    rng = np.random.RandomState(seed)
    triangles = rng.choice(len(f), size=count, p=areas / areas.sum())
    r1 = np.sqrt(rng.rand(count))
    r2 = rng.rand(count)
    a, b, c = 1.0 - r1, r1 * (1.0 - r2), r1 * r2
    return a[:, None] * v0[triangles] + b[:, None] * v1[triangles] + c[:, None] * v2[triangles]


def chamfer(pred: np.ndarray, gt: np.ndarray) -> tuple:
    """(mean of the squared, mean of the plain nearest-neighbor distances), each averaged over both directions."""
    pred, gt = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    pred_to_gt, _ = cKDTree(gt).query(pred, k=1)
    gt_to_pred, _ = cKDTree(pred).query(gt, k=1)
    squared = 0.5 * (float(np.mean(pred_to_gt ** 2)) + float(np.mean(gt_to_pred ** 2)))
    plain = 0.5 * (float(np.mean(pred_to_gt)) + float(np.mean(gt_to_pred)))
    return squared, plain


def legacy_chamfer(pred: np.ndarray, gt: np.ndarray) -> tuple:
    """(squared, plain, quarter turn in degrees) for the quarter turn about z with the smallest squared distance."""
    pred = np.asarray(pred, dtype=np.float64) * LEGACY_RESCALE
    best = None
    for degrees in (0, 90, 180, 270):
        c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
        turn = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        squared, plain = chamfer(pred @ turn.T, gt)
        if best is None or squared < best[0]:
            best = (squared, plain, degrees)
    return best


class LegacyMarchingCubes(SurfaceExtractor):
    def run(self, grid_logit, *, mc_level, bounds, octree_resolution, **kwargs):
        vertices, faces, _, _ = measure.marching_cubes(grid_logit.cpu().numpy(), mc_level, method="lewiner")
        bbox_min, bbox_size = self._compute_box(bounds)
        points = [int(octree_resolution) + 1] * 3
        return vertices / points * bbox_size + bbox_min, faces


def check_extractor(vae, resolution: int, device) -> str:
    """Extract a sphere at the validation resolution, so an unusable extractor stops the run before training."""
    axis = torch.linspace(-1.0, 1.0, resolution + 1, device=device)
    coords = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    field = 0.5 - coords.norm(dim=-1)
    extractor = vae.resolve_surface_extractor(torch.device(device).type == "cuda", None)
    mesh = extractor(field.unsqueeze(0), mc_level=MC_LEVEL, bounds=BOUNDS, octree_resolution=resolution)[0]
    if mesh is None or len(mesh.mesh_f) == 0:
        raise RuntimeError(f"{type(extractor).__name__} produced no surface for a sphere at resolution {resolution}; "
                           f"the traceback above shows why")
    return type(extractor).__name__


def _score(mesh, gt: np.ndarray, count: int, legacy: bool) -> dict:
    prefix = "legacy_" if legacy else ""
    pred = None if mesh is None else sample_mesh_surface(mesh.mesh_v, mesh.mesh_f, count, PRED_SEED)
    if pred is None:
        return {f"{prefix}chamfer_distance": FAILED, f"{prefix}chamfer_distance_squared": 12.0,
                f"{prefix}decode_failed": True}
    if legacy:
        squared, plain, turn = legacy_chamfer(pred, gt)
        return {"legacy_chamfer_distance": plain, "legacy_chamfer_distance_squared": squared,
                "legacy_decode_failed": False, "legacy_turn_deg": turn}
    squared, plain = chamfer(pred, gt)
    return {"chamfer_distance": plain, "chamfer_distance_squared": squared, "decode_failed": False,
            "faces": int(len(mesh.mesh_f))}


def generate(pipeline, image, mask, cfg: ValidationConfig):
    """The decoded meshes of one generation: (corrected, legacy) extraction of the same grid."""
    pipeline.device = torch.device("cuda")
    pipeline.dtype = torch.float32
    vae = pipeline.vae
    pipeline.conditioner.disable_drop = True
    with torch.amp.autocast(device_type="cuda"):
        latents = pipeline(image=image, mask=mask, generator=torch.Generator().manual_seed(0),
                           num_inference_steps=cfg.num_steps, guidance_scale=cfg.guidance_scale,
                           octree_resolution=cfg.octree_resolution, num_chunks=cfg.num_chunks,
                           output_type="latent", enable_pbar=False)
        with torch.inference_mode():
            decoded = vae(1.0 / vae.scale_factor * latents)
            kwargs = dict(bounds=BOUNDS, mc_level=MC_LEVEL, num_chunks=cfg.num_chunks,
                          octree_resolution=cfg.octree_resolution, enable_pbar=False)
            grid = vae.volume_decoder(decoded, vae.geo_decoder, **kwargs)
            corrected = vae.resolve_surface_extractor(grid.is_cuda, None)(grid, **kwargs)[0]
            legacy = LegacyMarchingCubes()(grid, **kwargs)[0]
    return corrected, legacy


def validate(pipeline, batch: dict, item: int, cfg: ValidationConfig) -> dict:
    """Score the mesh generated from `batch` item `item` against that item's surface points."""
    image = batch["image"][[item]].cuda()
    mask = batch["mask"][[item]].cuda()
    corrected, legacy = generate(pipeline, image, mask, cfg)
    gt_all = batch["surface"][item][:, :3].cpu().numpy()
    rng = np.random.RandomState(GT_SEED)
    gt = gt_all[rng.choice(len(gt_all), min(cfg.chamfer_points, len(gt_all)), replace=False)]
    return {"render": batch["renders"][item], "item": item,
            **_score(corrected, gt, cfg.chamfer_points, legacy=False),
            **_score(legacy, gt, cfg.chamfer_points, legacy=True)}

