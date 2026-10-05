"""Conversions between a source asset's own frame and the frame the shape model works in.

The model works with a mesh centered on its bounding box with the longest side spanning 2 * MESH_NORM_SCALE.
The map is a translation and a uniform scale, with no rotation, and produced meshes are mapped back by its
exact inverse.
"""

from dataclasses import dataclass

import numpy as np
import trimesh

MESH_NORM_SCALE = 0.9999


def load_mesh(path) -> trimesh.Trimesh:
    """One triangle mesh from a file, node transforms applied, vertices as stored; a mesh is returned as is."""
    if isinstance(path, trimesh.Trimesh):
        return path
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        merged = trimesh.Trimesh()
        for geometry in mesh.geometry.values():
            merged = trimesh.util.concatenate([merged, geometry])
        mesh = merged
    return mesh


@dataclass(frozen=True)
class Frame:
    center: tuple
    extent: float
    scale: float = MESH_NORM_SCALE

    @property
    def factor(self) -> float:
        return 2.0 * self.scale / self.extent

    def to_model(self, vertices: np.ndarray) -> np.ndarray:
        return (vertices - np.asarray(self.center)) * self.factor

    def to_source(self, vertices: np.ndarray) -> np.ndarray:
        return vertices / self.factor + np.asarray(self.center)

    def as_dict(self) -> dict:
        return {"center": list(self.center), "extent": self.extent, "scale": self.scale,
                "factor": self.factor}


def frame_of(mesh: trimesh.Trimesh, scale: float = MESH_NORM_SCALE) -> Frame:
    low, high = mesh.bounds
    return Frame(center=tuple(((low + high) / 2).tolist()), extent=float((high - low).max()), scale=scale)


def to_source_mesh(mesh: trimesh.Trimesh, frame: Frame) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=frame.to_source(mesh.vertices), faces=mesh.faces, process=False)
