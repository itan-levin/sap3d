"""The 3D edit region: loading, placement in the model frame, interior sampling and projection to 2D.

A region is a set of closed components in the source asset's own frame. The 3D probe works with each
component's convex hull (its vertices' Delaunay tetrahedralization); the orientation search, and the later
texturing and preservation stages, use the exact signed distance of the components' surface. Components
should therefore be convex and must not overlap.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import yaml
from scipy.spatial import Delaunay

from . import camera as cameras
from .frames import Frame, load_mesh
from .orient import turn_points

# Half-width of the cube the shape decoder's grid covers, in model units.
DECODE_BOX = 1.01

GLOBAL_PADDING = 0.1


@dataclass(frozen=True)
class Region:
    components: tuple   # (N_i, 3) float64 vertex arrays, source frame
    surface: object     # trimesh.Trimesh holding every component, for signed distance and projection
    description: str


def _primitive(spec: dict) -> trimesh.Trimesh:
    kind = spec.get("type")
    keys = {"box": {"type", "center", "extents", "rotation_deg"},
            "sphere": {"type", "center", "radius"},
            "cylinder": {"type", "center", "radius", "height", "axis"}}
    if kind not in keys:
        raise ValueError(f"region primitive type must be one of {sorted(keys)}, got {kind!r}")
    unknown = set(spec) - keys[kind]
    if unknown:
        raise ValueError(f"unknown keys {sorted(unknown)} for a {kind} primitive")
    for key in ("extents", "radius", "height"):
        if key in spec and not (np.asarray(spec[key], dtype=np.float64) > 0).all():
            raise ValueError(f"a {kind} primitive needs a positive {key}, got {spec[key]!r}")
    center = np.asarray(spec["center"], dtype=np.float64)
    if center.shape != (3,):
        raise ValueError(f"a primitive center needs three coordinates, got {spec['center']!r}")
    if kind == "box":
        transform = np.eye(4)
        if "rotation_deg" in spec:
            transform[:3, :3] = trimesh.transformations.euler_matrix(
                *np.radians(np.asarray(spec["rotation_deg"], dtype=np.float64)), axes="sxyz")[:3, :3]
        transform[:3, 3] = center
        return trimesh.creation.box(extents=np.asarray(spec["extents"], dtype=np.float64), transform=transform)
    if kind == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=float(spec["radius"]))
        mesh.apply_translation(center)
        return mesh
    axis = spec.get("axis", "y")
    if axis not in ("x", "y", "z"):
        raise ValueError(f"a cylinder axis must be x, y or z, got {axis!r}")
    mesh = trimesh.creation.cylinder(radius=float(spec["radius"]), height=float(spec["height"]), sections=64)
    if axis == "x":
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    elif axis == "y":
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    mesh.apply_translation(center)
    return mesh


def from_primitives(specs: list, description: str) -> Region:
    if not specs:
        raise ValueError(f"{description} lists no primitives")
    meshes = [_primitive(spec) for spec in specs]
    surface = trimesh.util.concatenate(meshes)
    return Region(components=tuple(np.asarray(m.vertices, dtype=np.float64) for m in meshes),
                  surface=surface, description=description)


def load_region(path) -> Region:
    """A region from a GLB holding one closed mesh object per component, or a YAML/JSON `primitives` list.

    Each primitive is a mapping: `{type: box, center, extents, rotation_deg (optional, XYZ)}`,
    `{type: sphere, center, radius}` or `{type: cylinder, center, radius, height, axis (x, y or z)}`,
    in the source asset's frame.
    """
    path = Path(path)
    if path.suffix.lower() in (".yaml", ".yml", ".json"):
        text = path.read_text(encoding="utf-8")
        spec = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
        if not isinstance(spec, dict) or set(spec) != {"primitives"}:
            raise ValueError(f"{path} must hold exactly one key, `primitives`")
        return from_primitives(spec["primitives"], str(path))
    scene = trimesh.load(str(path))
    components = list(scene.dump(concatenate=False)) if isinstance(scene, trimesh.Scene) else [scene]
    if not components:
        raise ValueError(f"{path} holds no mesh, so there is no edit region")
    return Region(components=tuple(np.asarray(c.vertices, dtype=np.float64) for c in components),
                  surface=load_mesh(path), description=str(path))


def global_region(source: trimesh.Trimesh, padding: float = GLOBAL_PADDING) -> Region:
    """One box around the whole source: its bounding box grown by `padding` times its longest side."""
    low, high = source.bounds
    margin = padding * float((high - low).max())
    return from_primitives([{"type": "box", "center": ((low + high) / 2).tolist(),
                             "extents": ((high - low) + 2 * margin).tolist()}],
                           f"source bounding box padded by {padding} of its longest side")


def place_components(region: Region, frame: Frame, degrees: int) -> list:
    """Each component's vertices in the model frame (float32), turned `degrees` about the source center first.

    A component that does not reach the decoder's cube cannot enclose any surface the model produces, which
    means the region and the source asset are in different frames.
    """
    placed = [frame.to_model(turn_points(vertices, degrees, frame.center)).astype(np.float32)
              for vertices in region.components]
    for index, vertices in enumerate(placed):
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        if (np.minimum(high, DECODE_BOX) - np.maximum(low, -DECODE_BOX)).min() <= 0:
            raise ValueError(
                f"region component {index} spans {np.round(low, 4).tolist()} to {np.round(high, 4).tolist()} "
                f"in the model frame, outside the +/-{DECODE_BOX} cube the decoder covers; the region is not "
                f"in the source asset's frame")
    return placed


def points_inside(components: list, coords: np.ndarray) -> np.ndarray:
    inside = np.zeros(len(coords), dtype=bool)
    for vertices in components:
        inside |= Delaunay(vertices).find_simplex(coords) >= 0
    return inside


def sample_volume_points(components: list, n_points: int, seed: int) -> np.ndarray:
    """`n_points` (float32) uniform inside the union of the component hulls, from a local generator.

    The count is split across components in proportion to their bounding-box volumes, by largest
    remainder, then each share is rejection-sampled from its box.
    """
    rng = np.random.default_rng(seed)
    comps = [np.asarray(v, dtype=np.float64) for v in components]
    if not comps:
        raise ValueError("no region component to sample inside")
    volumes = np.array([float(np.prod(c.max(axis=0) - c.min(axis=0))) for c in comps])
    total = volumes.sum()
    weights = (volumes / total) if total > 0 else np.full(len(comps), 1.0 / len(comps))
    raw_counts = weights * n_points
    counts = np.floor(raw_counts).astype(int)
    remainder = int(n_points - counts.sum())
    if remainder > 0:
        counts[np.argsort(raw_counts - counts)[::-1][:remainder]] += 1

    out = []
    for vertices, count in zip(comps, counts):
        if count <= 0:
            continue
        hull = Delaunay(vertices)
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        collected, need = [], int(count)
        for _ in range(1000):
            if need <= 0:
                break
            batch = rng.uniform(low, high, size=(max(need * 4, 256), 3))
            inside = batch[hull.find_simplex(batch) >= 0]
            if len(inside):
                collected.append(inside[:need])
                need -= len(inside[:need])
        if need > 0:
            raise ValueError(f"could not fill a region component ({need} points short); its hull is "
                             f"nearly flat")
        out.append(np.concatenate(collected, axis=0))
    return np.concatenate(out, axis=0).astype(np.float32)


def project_mask(region: Region, source: trimesh.Trimesh, camera: cameras.Camera) -> np.ndarray:
    """(size, size) bool: pixels whose ray meets the region no farther than it meets the source."""
    ray_table = cameras.rays(camera)
    region_distance = cameras.first_hit_distance(region.surface, ray_table, camera.size)
    source_distance = cameras.first_hit_distance(source, ray_table, camera.size)
    return np.isfinite(region_distance) & (region_distance <= source_distance)
