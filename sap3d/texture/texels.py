"""The edit region rasterized onto an atlas: which texels lie inside it, and how deep.

A texel is inside when it is inside any region component (pysdf, positive inside), so the interior of two
overlapping components counts as inside. Depth comes from the face planes of each component, which must
therefore be convex; pysdf's own distance magnitude is only an upper bound and is not used.
"""

import numpy as np
import trimesh
from pysdf import SDF

# Points times faces per matmul block.
_DEPTH_BLOCK = 8_000_000

# A component counts as convex when its volume matches its convex hull's this closely.
_CONVEX_TOL = 1e-3

# How far, as a fraction of the region's bounding-box diagonal, a texel's plane depth may fall on the other
# side from the inside test before the two count as disagreeing rather than as float32 rounding.
_SIGN_TOLERANCE = 1e-5


def region_mesh(surface: trimesh.Trimesh) -> trimesh.Trimesh:
    """A vertex-merged copy of the region surface; pysdf walks vertex adjacency."""
    if len(surface.faces) == 0:
        raise ValueError("the region surface has no faces")
    merged = surface.copy()
    merged.merge_vertices()
    return merged


def components(mesh: trimesh.Trimesh) -> list:
    parts = mesh.split(only_watertight=False)
    if len(parts) == 0:
        raise ValueError("the region mesh has faces but split into no components")
    return parts


def _validate_maps(position, gate, gate_name):
    if not isinstance(position, np.ndarray) or position.ndim != 3 or position.shape[2] != 3:
        raise ValueError(f"position must be an (H, W, 3) array, got shape {getattr(position, 'shape', None)}")
    if not isinstance(gate, np.ndarray) or gate.dtype != np.bool_ or gate.shape != position.shape[:2]:
        raise ValueError(f"{gate_name} must be a boolean array shaped {position.shape[:2]}")
    if not np.isfinite(position[gate]).all():
        raise ValueError(f"position is not finite at some of the texels {gate_name} selects")


def _inside_any(mesh, query) -> np.ndarray:
    inside = np.zeros(len(query), dtype=bool)
    for part in components(mesh):
        inside |= SDF(part.vertices, part.faces)(query) > 0.0
    return inside


def points_inside(points: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
    """(N,) bool: which of an (N, 3) array of points lie inside any component of the merged region mesh."""
    query = np.ascontiguousarray(np.asarray(points, dtype=np.float32))
    if query.ndim != 2 or query.shape[1] != 3 or not np.isfinite(query).all():
        raise ValueError(f"points must be a finite (N, 3) array, got shape {query.shape}")
    return _inside_any(mesh, query)


def region_mask(position: np.ndarray, mapped: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
    """(H, W) bool: texels inside a UV chart and inside any region component."""
    _validate_maps(position, mapped, "mapped")
    out = np.zeros(mapped.shape, dtype=bool)
    out[mapped] = _inside_any(mesh, np.ascontiguousarray(position[mapped], dtype=np.float32))
    return out


def _outward_planes(component):
    if len(component.vertices) < 4:
        raise ValueError(f"a region component has only {len(component.vertices)} vertices, too few to enclose "
                         f"a volume")
    volume = abs(float(component.volume))
    hull_volume = abs(float(component.convex_hull.volume))
    if volume <= 0.0 or abs(hull_volume - volume) / volume > _CONVEX_TOL:
        raise ValueError(f"a region component is not convex (volume {volume:.6g} against convex hull "
                         f"{hull_volume:.6g}), so its depth cannot be taken from face planes; two components "
                         f"sharing a face weld into one such body when vertices are merged")
    normals = np.asarray(component.face_normals, dtype=np.float64)
    lengths = np.linalg.norm(normals, axis=1)
    if not np.allclose(lengths, 1.0, atol=1e-6):
        raise ValueError(f"a region component has {int((np.abs(lengths - 1.0) > 1e-6).sum())} face(s) with no "
                         f"usable normal")
    offsets = np.einsum("ij,ij->i", normals, np.asarray(component.triangles, dtype=np.float64)[:, 0, :])
    centroid = np.asarray(component.vertices, dtype=np.float64).mean(axis=0)
    if (centroid @ normals.T - offsets).max() > 0.0:
        normals, offsets = -normals, -offsets
    outermost = float((centroid @ normals.T - offsets).max())
    if outermost >= 0.0:
        raise ValueError(f"a region component has an inconsistent face winding (worst plane {outermost:+.6g})")
    return normals, offsets


def _convex_depth(component, points):
    """Distance to a convex component's boundary, exact inside; outside, a negative lower bound."""
    normals, offsets = _outward_planes(component)
    block = max(1, _DEPTH_BLOCK // len(normals))
    depth = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), block):
        chunk = points[start:start + block]
        depth[start:start + len(chunk)] = -(chunk @ normals.T - offsets).max(axis=1)
    return depth


def signed_depth(position: np.ndarray, gate: np.ndarray, mask: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
    """(H, W) float32: distance of each gated texel to the region boundary, positive where `mask` is set.

    The sign is the inside test's; the magnitude is the plane depth within the deepest component. A texel
    outside `gate` is 0, indistinguishable from one on the boundary, so consumers carry the gate along.
    """
    _validate_maps(position, mask, "mask")
    if (mask & ~gate).any():
        raise ValueError(f"{int((mask & ~gate).sum())} of the {int(mask.sum())} texels the mask selects sit "
                         f"outside the gate")
    tolerance = _SIGN_TOLERANCE * float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    _validate_maps(position, gate, "gate")
    points = np.asarray(position[gate], dtype=np.float64)
    raw = np.full(len(points), -np.inf)
    for part in components(mesh):
        raw = np.maximum(raw, _convex_depth(part, points))
    inside = mask[gate]
    if inside.any() and float(raw[inside].max()) < -tolerance:
        raise ValueError("every texel the mask calls inside has a negative plane depth; the inside test and "
                         "the face planes disagree about which side is which")
    if (~inside).any() and float(raw[~inside].min()) > tolerance:
        raise ValueError("every texel the mask calls outside has a positive plane depth; the inside test and "
                         "the face planes disagree about which side is which")
    out = np.zeros(gate.shape, dtype=np.float32)
    out[gate] = np.where(inside, np.abs(raw), -np.abs(raw))
    return out
