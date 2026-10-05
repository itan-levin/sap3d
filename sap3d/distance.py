"""Exact point-to-surface distance between two meshes, outside an edit region.

Both surfaces are sampled with a fixed seed and each sample's exact distance to the other mesh is taken,
so the estimator's floor is near 1e-8 rather than the point-to-point sampling floor of a Chamfer distance.
A point is inside the region when the region's signed distance there is positive; inside points are
dropped from both samples.
"""

import numpy as np
import open3d as o3d
import trimesh
from pysdf import SDF

from .frames import load_mesh

SURFACE_POINTS = 200_000
PERCENTILES = (50, 75, 90, 95, 99, 100)

# Surface sampling draws from the global NumPy generator; it is reseeded with this for the draw and its
# previous state restored afterwards, so callers keep their own stream.
SAMPLE_SEED = 11


def region_sdf(region):
    """Signed distance of a closed region surface (a path or a mesh), positive inside."""
    mesh = load_mesh(region)
    return SDF(mesh.vertices, mesh.faces)


def outside_region(points: np.ndarray, sdf) -> np.ndarray:
    """The points outside the region; every point when `sdf` is None."""
    if sdf is None:
        return points
    return points[~(sdf(points) > 0)]


def point_to_surface(points: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(np.asarray(mesh.vertices), dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(np.asarray(mesh.faces), dtype=o3d.core.Dtype.UInt32))
    query = o3d.core.Tensor(np.asarray(points, dtype=np.float32), dtype=o3d.core.Dtype.Float32)
    return scene.compute_distance(query).numpy().astype(np.float64)


def sample_outside(mesh: trimesh.Trimesh, sdf, num_points: int = SURFACE_POINTS,
                   seed: int = SAMPLE_SEED) -> np.ndarray:
    state = np.random.get_state()
    np.random.seed(seed)
    points = trimesh.sample.sample_surface(mesh, num_points)[0]
    np.random.set_state(state)
    return outside_region(points, sdf)


class RegionSwallowedSurface(ValueError):
    """One of the two surfaces has no sample outside the region, so no comparison is defined."""


def compare(truth: trimesh.Trimesh, prediction: trimesh.Trimesh, sdf,
            num_points: int = SURFACE_POINTS, seed: int = SAMPLE_SEED) -> dict:
    """Symmetric mean distance in the meshes' own units; percentiles as fractions of the truth's diagonal."""
    truth_points = sample_outside(truth, sdf, num_points, seed)
    prediction_points = sample_outside(prediction, sdf, num_points, seed)
    if not len(truth_points) or not len(prediction_points):
        raise RegionSwallowedSurface(
            f"the edit region swallowed one of the two surfaces, so there is nothing outside it to "
            f"compare: {len(truth_points)} of {num_points} truth points and "
            f"{len(prediction_points)} of {num_points} prediction points survive it")

    truth_to_prediction = point_to_surface(truth_points, prediction)
    prediction_to_truth = point_to_surface(prediction_points, truth)
    both = np.concatenate([truth_to_prediction, prediction_to_truth])

    low, high = truth.bounds
    diagonal = float(np.linalg.norm(high - low))
    prediction_extent = prediction.bounds[1] - prediction.bounds[0]
    return {
        "symmetric": float((truth_to_prediction.mean() + prediction_to_truth.mean()) / 2),
        "diagonal": diagonal,
        "percentiles_of_diagonal": {f"p{p}": float(np.percentile(both, p) / diagonal)
                                    for p in PERCENTILES},
        "extent": prediction_extent.tolist(),
        "extent_ratio": (prediction_extent / (high - low)).tolist(),
        "faces": int(len(prediction.faces)),
        "points_outside_region": [int(len(truth_points)), int(len(prediction_points))],
    }
