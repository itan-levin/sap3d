"""Pinhole cameras for projecting an edit region into the edited view.

A camera is stored the way Edit3D-Bench's `source_model/render/meta.json` records one: a Blender
camera-to-world matrix (Blender world is Z-up, the camera looks down its local -Z with +Y up) and a
horizontal field of view. Meshes are used exactly as trimesh loads them from a glTF file, which is Y-up,
so each camera is carried into the file's frame instead of moving the meshes. Frames are square, and the
first image row is the top of the frame.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

# Sends a point of Blender's Z-up world to the Y-up coordinates the glTF file states.
BLENDER_TO_FILE = np.array([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, -1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])

# The benchmark renderer's orbit: a 35 mm lens on a 32 mm sensor, 1.8 units from the origin.
ORBIT_RADIUS = 1.8
ORBIT_FOV_X = 2.0 * math.atan(16.0 / 35.0)


@dataclass(frozen=True)
class Camera:
    transform: np.ndarray  # camera to file frame, 4x4
    fov_x: float           # radians
    size: int              # pixels per side


def _from_blender(transform_matrix, fov_x: float, size: int) -> Camera:
    blender = np.asarray(transform_matrix, dtype=np.float64)
    if blender.shape != (4, 4):
        raise ValueError(f"a camera transform must be 4x4, got {blender.shape}")
    return Camera(transform=BLENDER_TO_FILE @ blender, fov_x=float(fov_x), size=int(size))


def from_meta(meta_path, view: int) -> Camera:
    """Camera `view` of a meta.json file holding `width`, `height` and a `locations` list."""
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    locations = meta["locations"]
    if not 0 <= view < len(locations):
        raise ValueError(f"view {view} is outside the {len(locations)} cameras of {meta_path}")
    entry = locations[view]
    if entry.get("projection_type", "PERSP") != "PERSP":
        raise ValueError(f"view {view} of {meta_path} is a {entry['projection_type']} camera; only "
                         f"perspective cameras are supported")
    if meta["width"] != meta["height"]:
        raise ValueError(f"{meta_path} records a {meta['width']}x{meta['height']} frame; only square "
                         f"frames are supported")
    return _from_blender(entry["transform_matrix"], entry["camera_angle_x"], meta["width"])


def from_json(path) -> Camera:
    """A single-camera file: `transform_matrix` (Blender camera-to-world), `camera_angle_x`, `size`."""
    entry = json.loads(Path(path).read_text(encoding="utf-8"))
    return _from_blender(entry["transform_matrix"], entry["camera_angle_x"], entry["size"])


def orbit_blender_matrix(elevation_deg: float, azimuth_deg: float, radius: float = ORBIT_RADIUS) -> np.ndarray:
    """Blender camera-to-world matrix of a camera on the benchmark renderer's sphere, aimed at the origin.

    Azimuth follows the benchmark's meta.json, where view 0 sits at -90 degrees.
    """
    phi = 0.5 * math.pi - math.radians(elevation_deg)
    theta = math.radians(azimuth_deg)
    position = radius * np.array([math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta),
                                  math.cos(phi)])
    back = position / np.linalg.norm(position)
    right = np.cross([0.0, 0.0, 1.0], back)
    if np.linalg.norm(right) < 1e-12:
        raise ValueError("a camera straight above or below the origin has no defined up direction")
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    matrix = np.eye(4)
    matrix[:3, 0], matrix[:3, 1], matrix[:3, 2], matrix[:3, 3] = right, up, back, position
    return matrix


def orbit(elevation_deg: float, azimuth_deg: float, size: int, radius: float = ORBIT_RADIUS,
          fov_x: float = ORBIT_FOV_X) -> Camera:
    return _from_blender(orbit_blender_matrix(elevation_deg, azimuth_deg, radius), fov_x, size)


def rays(camera: Camera) -> np.ndarray:
    """One ray per pixel center, (size * size, 6) float32 origin then direction, rows from the top."""
    size = camera.size
    half = np.tan(0.5 * camera.fov_x)
    centers = (np.arange(size, dtype=np.float64) + 0.5) / size * 2.0 - 1.0
    x = centers[None, :] * half
    y = -centers[:, None] * half
    local = np.stack([np.broadcast_to(x, (size, size)), np.broadcast_to(y, (size, size)),
                      np.full((size, size), -1.0)], axis=-1).reshape(-1, 3)
    direction = local @ camera.transform[:3, :3].T
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    origin = np.broadcast_to(camera.transform[:3, 3], direction.shape)
    return np.ascontiguousarray(np.concatenate([origin, direction], axis=1), dtype=np.float32)


def first_hit_distance(mesh: trimesh.Trimesh, ray_table: np.ndarray, size: int) -> np.ndarray:
    """Distance along each ray to the first surface it meets, (size, size); infinity for a miss."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32), o3d.core.float32),
        o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.int32), o3d.core.int32)))
    hit = scene.cast_rays(o3d.core.Tensor(ray_table, o3d.core.float32))["t_hit"].numpy()
    return hit.astype(np.float64).reshape(size, size)
