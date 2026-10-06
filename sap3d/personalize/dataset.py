"""The training set of one asset: its 16 benchmark-style views, random renders, and surface points.

An asset folder holds `source_model/model.glb` and its 16 views `source_model/render/render_0000-0015.png`
(an Edit3D-Bench object folder, or the output of `prepare`). The dataset folder gets:

    renders/000-015.png   the 16 views, copied
    renders/016-NNN.png   random renders of the normalized mesh (RGBA, transparent background)
    surface.npz           `points`: (N, 6) float32, position then unit face normal, rounded to float16, in the
                          training frame (bounding box centered at the origin, longest side 1.9998, Y up)
    dataset.json          the settings, the normalization and every view's camera, lighting and engine
"""

import json
import random
import shutil
from pathlib import Path

import numpy as np
import trimesh

from ..config import DatasetConfig

BENCHMARK_VIEWS = 16


def asset_files(asset_dir) -> tuple:
    """The source mesh and the 16 view renders of an asset folder."""
    source = Path(asset_dir) / "source_model"
    glb = source / "model.glb"
    views = [source / "render" / f"render_{view:04d}.png" for view in range(BENCHMARK_VIEWS)]
    missing = [str(path) for path in [glb, *views] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"the asset folder lacks {missing}; make one with `sap3d-personalize prepare`")
    return glb, views


def sample_surface(mesh: trimesh.Trimesh, count: int, seed=None) -> np.ndarray:
    """`count` area-weighted surface points with their face normals, (count, 6), rounded to float16."""
    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    points, faces = trimesh.sample.sample_surface(mesh, count, seed=seed)
    normals = mesh.face_normals[faces]
    surface = np.concatenate((points.astype(np.float32), normals.astype(np.float32)), axis=1).astype(np.float16)
    return surface.astype(np.float32)


def build(asset_dir, out_dir, cfg: DatasetConfig) -> Path:
    import torch

    from .blender import scene
    from .blender.renderer import ENGINES, LIGHTING, Renderer

    glb, views = asset_files(asset_dir)
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty")
    renders = out_dir / "renders"
    renders.mkdir(parents=True)
    py_rng = random.Random(cfg.seed)
    np_rng = np.random.RandomState(cfg.seed)

    working = scene.joined_working_mesh(glb)
    normalization = scene.normalize_for_training(working)
    surface = sample_surface(scene.export_trimesh(working), cfg.surface_points, cfg.seed)
    np.savez_compressed(out_dir / "surface.npz", points=surface)

    records = []
    for index, view in enumerate(views):
        shutil.copy2(view, renders / f"{index:03d}.png")
        records.append({"index": index, "kind": "benchmark_view", "source": view.name})

    renderer = Renderer(cfg.render_size, cfg.render_samples, cfg.camera_fov, gpu=torch.cuda.is_available())
    for index in range(BENCHMARK_VIEWS, BENCHMARK_VIEWS + cfg.random_views):
        elevation = float(np_rng.uniform(cfg.elevation_min, cfg.elevation_max))
        azimuth = float(np_rng.uniform(cfg.azimuth_min, cfg.azimuth_max))
        lighting = py_rng.choice(LIGHTING)
        distance_scale = float(np_rng.uniform(cfg.distance_scale_min, cfg.distance_scale_max))
        distance = float(renderer.min_distance(working, azimuth=azimuth, elevation=elevation)) * distance_scale
        engine = py_rng.choice(ENGINES)
        renderer.set_engine(engine)
        renderer.render(working, elevation, azimuth, distance, lighting, renders / f"{index:03d}.png")
        records.append({"index": index, "kind": "random", "elevation": elevation, "azimuth": azimuth,
                        "distance": distance, "distance_scale": distance_scale, "lighting": lighting,
                        "engine": engine})

    record = {"asset": str(Path(asset_dir).resolve()), "settings": {**vars(cfg)}, "normalization": normalization,
              "surface_points": int(surface.shape[0]), "views": records}
    (out_dir / "dataset.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return out_dir
