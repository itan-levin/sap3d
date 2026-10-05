"""Our own UV atlas on an edited mesh, filled by copying texels from a textured mesh.

The mesh is decimated, unwrapped with xatlas, and every texel of the new atlas is baked to a 3D position
and normal. A texel takes its color from the closest point on the textured mesh: the barycentric weights
of that point give the textured mesh's own UV, where its base-color texture is sampled. A match whose
source face normal points against the texel's normal is rejected; rejected texels and texels outside
every chart are filled from the nearest accepted texel. There is no distance test.

Row 0 of every atlas is v = 1. The bake addresses texel centers, u = (col + 0.5) / S, while the bilinear
read addresses texel corners, u = col / (S - 1).

The exported GLB declares what the Edit3D-Bench source GLBs declare: positions and UVs only (no vertex
normals, so the benchmark renderer shades it flat like the sources), one PNG base-color texture, no glTF
extensions, roughness 1, no metallic-roughness texture, single-sided.
"""

import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
from scipy import ndimage

from .. import third_party

BAKE_MARGIN = 2.0
JSON_CHUNK = 0x4E4F534A
QUERY_CHUNK = 1 << 21
PERCENTILES = (50, 75, 90, 99, 99.9)

EXPECTED_ATTRIBUTES = ["POSITION", "TEXCOORD_0"]
EXPECTED_MATERIALS = [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0},
                                                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                                                "roughnessFactor": 1.0},
                       "doubleSided": False}]


def _load_mesh_uv_wrap():
    path = third_party.require(third_party.HY3DPAINT_DIR, "utils/uvwrap_utils.py") / "utils" / "uvwrap_utils.py"
    spec = importlib.util.spec_from_file_location("hy3dpaint_uvwrap_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.mesh_uv_wrap


def read_glb_chunks(path) -> list:
    raw = Path(path).read_bytes()
    if raw[:4] != b"glTF":
        raise ValueError(f"not a GLB: {path}")
    chunks, offset = [], 12
    while offset < len(raw):
        length, kind = struct.unpack("<II", raw[offset:offset + 8])
        chunks.append((kind, raw[offset + 8:offset + 8 + length]))
        offset += 8 + length
    return chunks


def write_glb_chunks(path, chunks: list) -> None:
    body = b""
    for kind, data in chunks:
        pad = (-len(data)) % 4
        data = data + (b" " if kind == JSON_CHUNK else b"\x00") * pad
        body += struct.pack("<II", len(data), kind) + data
    Path(path).write_bytes(b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body)


def read_glb_json(path) -> dict:
    for kind, data in read_glb_chunks(path):
        if kind == JSON_CHUNK:
            return json.loads(data)
    raise ValueError(f"no JSON chunk in {path}")


def match_benchmark_material(path) -> None:
    """Rewrite a GLB's material and extension declarations into the benchmark sources' declaration."""
    patched = []
    for kind, data in read_glb_chunks(path):
        if kind == JSON_CHUNK:
            gltf = json.loads(data)
            gltf.pop("extensionsUsed", None)
            gltf.pop("extensionsRequired", None)
            for material in gltf.get("materials", []):
                material.pop("extensions", None)
                material["doubleSided"] = False
                pbr = material.setdefault("pbrMetallicRoughness", {})
                pbr.pop("metallicRoughnessTexture", None)
                pbr.pop("metallicFactor", None)
                pbr["roughnessFactor"] = 1.0
                pbr["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
            data = json.dumps(gltf).encode()
        patched.append((kind, data))
    write_glb_chunks(path, patched)


def check_declaration(path) -> None:
    gltf = read_glb_json(path)
    attributes = sorted(gltf["meshes"][0]["primitives"][0]["attributes"])
    if attributes != EXPECTED_ATTRIBUTES:
        raise RuntimeError(f"{path} declares {attributes}; a NORMAL attribute would render smooth-shaded "
                           f"where the benchmark sources render flat")
    if gltf.get("extensionsUsed") or gltf.get("extensionsRequired"):
        raise RuntimeError(f"{path} declares glTF extensions, which no benchmark source does")
    if gltf.get("materials") != EXPECTED_MATERIALS:
        raise RuntimeError(f"{path} declares materials {gltf.get('materials')} against the benchmark "
                           f"sources' {EXPECTED_MATERIALS}")


def load_textured_source(path) -> trimesh.Trimesh:
    """The source mesh with its base-color texture; exactly one primitive carrying one texture."""
    gltf = read_glb_json(path)
    primitives = sum(len(mesh["primitives"]) for mesh in gltf.get("meshes", []))
    if primitives != 1 or len(gltf.get("images", [])) != 1:
        raise ValueError(f"{path} holds {primitives} primitives and {len(gltf.get('images', []))} images; "
                         f"copying the source texture needs exactly one primitive with one base-color texture")
    mesh = trimesh.load(str(path), force="mesh", process=False)
    material = getattr(mesh.visual, "material", None)
    if getattr(material, "baseColorTexture", None) is None or getattr(mesh.visual, "uv", None) is None:
        raise ValueError(f"{path} carries no base-color texture with UVs")
    return mesh


def sample_bilinear(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Sample an atlas laid out row 0 = v 1, clamping at the border."""
    height, width = texture.shape[:2]
    x = np.clip(uv[:, 0], 0.0, 1.0) * (width - 1)
    y = (1.0 - np.clip(uv[:, 1], 0.0, 1.0)) * (height - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    top = texture[y0, x0] * (1 - fx) + texture[y0, x1] * fx
    bottom = texture[y1, x0] * (1 - fx) + texture[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def as_float_texture(texture) -> np.ndarray:
    """An (H, W, 3) float32 image in [0, 1], from a PIL image or a uint8 array."""
    if isinstance(texture, Image.Image):
        return np.asarray(texture.convert("RGB"), dtype=np.float32) / 255.0
    array = np.asarray(texture)
    if array.ndim != 3 or array.shape[2] < 3 or array.dtype != np.uint8:
        raise ValueError(f"expected an (H, W, >=3) uint8 image, got {array.dtype} {array.shape}")
    return array[:, :, :3].astype(np.float32) / 255.0


def raycast_scene(mesh: trimesh.Trimesh) -> o3d.t.geometry.RaycastingScene:
    target = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32), o3d.core.float32),
        o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.int32), o3d.core.int32))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(target)
    return scene


def closest_points(scene: o3d.t.geometry.RaycastingScene, queries: np.ndarray) -> tuple:
    points, primitives, barycentric = [], [], []
    for start in range(0, len(queries), QUERY_CHUNK):
        block = queries[start:start + QUERY_CHUNK].astype(np.float32)
        answer = scene.compute_closest_points(o3d.core.Tensor(block, o3d.core.float32))
        points.append(answer["points"].numpy())
        primitives.append(answer["primitive_ids"].numpy())
        barycentric.append(answer["primitive_uvs"].numpy())
    return np.concatenate(points), np.concatenate(primitives), np.concatenate(barycentric)


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Quadric decimation (open3d, through trimesh) to a face budget; a mesh within it is kept."""
    reduced = (mesh.simplify_quadric_decimation(face_count=target_faces)
               if len(mesh.faces) > target_faces else mesh)
    return trimesh.Trimesh(vertices=np.asarray(reduced.vertices, dtype=np.float64),
                           faces=np.asarray(reduced.faces, dtype=np.int64), process=False, validate=False)


def plain_glb_bytes(mesh: trimesh.Trimesh) -> bytes:
    """The mesh as GLB bytes with positions and faces only.

    trimesh caches vertex normals on a mesh once anything reads them and its exporter then writes them, so
    the bytes are taken from a fresh mesh.
    """
    plain = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False, validate=False)
    return trimesh.exchange.gltf.export_glb(plain, include_normals=False)


def unwrap(mesh: trimesh.Trimesh) -> tuple:
    """A fresh xatlas atlas on a copy of the mesh, and the unsplit vertex normal of each unwrapped vertex.

    xatlas splits a vertex wherever a seam runs through it, so normals are taken before the unwrap and
    carried across by face and corner order.
    """
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    work = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False, validate=False)
    _load_mesh_uv_wrap()(work)
    if work.faces.shape != mesh.faces.shape:
        raise RuntimeError("xatlas changed the face count")
    moved = np.abs(work.vertices[work.faces] - mesh.vertices[mesh.faces]).max()
    if moved != 0.0:
        raise RuntimeError(f"xatlas moved the geometry by {moved}")
    to_unsplit = np.empty(len(work.vertices), dtype=np.int64)
    to_unsplit[work.faces.reshape(-1)] = mesh.faces.reshape(-1)
    return work, normals[to_unsplit]


def bake_surface_maps(mesh: trimesh.Trimesh, size: int, normals: np.ndarray) -> tuple:
    """(position, normal, mapped): each texel's 3D position and normal, and which texels a chart covers."""
    baked = o3d.t.geometry.TriangleMesh()
    baked.vertex.positions = o3d.core.Tensor(mesh.vertices, o3d.core.float32)
    baked.triangle.indices = o3d.core.Tensor(mesh.faces, o3d.core.int32)
    baked.triangle.texture_uvs = o3d.core.Tensor(mesh.visual.uv[mesh.faces].astype(np.float32), o3d.core.float32)
    baked.vertex.surface_position = o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32), o3d.core.float32)
    baked.vertex.surface_normal = o3d.core.Tensor(np.asarray(normals, dtype=np.float32), o3d.core.float32)
    result = baked.bake_vertex_attr_textures(size, {"surface_position", "surface_normal"}, margin=BAKE_MARGIN,
                                             fill=float("nan"), update_material=False)
    position = result["surface_position"].numpy()
    normal = result["surface_normal"].numpy()
    mapped = np.isfinite(position).all(axis=-1) & np.isfinite(normal).all(axis=-1)
    return position, normal, mapped


def transfer_texture(position: np.ndarray, mapped: np.ndarray, normal: np.ndarray,
                     source_mesh: trimesh.Trimesh, normal_gate: bool = True) -> tuple:
    """(rgb uint8 atlas, stats): each mapped texel's color read at its closest point on `source_mesh`."""
    if not mapped.any():
        raise ValueError("no texel is inside a UV chart, so there is nothing to transfer")
    texture = as_float_texture(source_mesh.visual.material.baseColorTexture)
    source_uv = np.asarray(source_mesh.visual.uv, dtype=np.float64)
    rows, cols = np.nonzero(mapped)
    queries = position[rows, cols].astype(np.float64)

    hit_points, primitive_ids, barycentric = closest_points(raycast_scene(source_mesh), queries)
    distance = np.linalg.norm(hit_points.astype(np.float64) - queries, axis=1)

    corners = source_mesh.faces[primitive_ids.astype(np.int64)]
    weight_b = barycentric[:, 0].astype(np.float64)
    weight_c = barycentric[:, 1].astype(np.float64)
    weights = np.stack([1.0 - weight_b - weight_c, weight_b, weight_c], axis=1)[:, :, None]

    our_normals = normal[rows, cols].astype(np.float64)
    our_normals = our_normals / np.maximum(np.linalg.norm(our_normals, axis=1, keepdims=True), 1e-12)
    source_normals = np.asarray(source_mesh.face_normals, dtype=np.float64)[primitive_ids.astype(np.int64)]
    agreement = (source_normals * our_normals).sum(axis=1)
    rejected = agreement < 0.0 if normal_gate else np.zeros(len(queries), dtype=bool)

    colors = sample_bilinear(texture, (source_uv[corners] * weights).sum(axis=1))
    atlas = np.zeros(mapped.shape + (3,), dtype=np.float32)
    atlas[rows, cols] = colors
    accepted = np.zeros(mapped.shape, dtype=bool)
    accepted[rows[~rejected], cols[~rejected]] = True
    if not accepted.any():
        raise ValueError(f"no texel survived: {mapped.sum()} mapped, {rejected.sum()} rejected by the normal gate")
    nearest = ndimage.distance_transform_edt(~accepted, return_distances=False, return_indices=True)
    rgb = np.clip(atlas[tuple(nearest)] * 255.0 + 0.5, 0, 255).astype(np.uint8)

    diagonal = float(np.linalg.norm(source_mesh.extents))
    relative = distance / diagonal
    stats = {
        "texels_mapped": int(mapped.sum()),
        "mapped_fraction": float(mapped.mean()),
        "rejected_texels": int(rejected.sum()),
        "source_faces": int(len(source_mesh.faces)),
        "bbox_diagonal": diagonal,
        "relative_mean": float(relative.mean()),
        "relative_max": float(relative.max()),
        "high_fraction_1pct_diag": float((relative > 0.01).mean()),
    }
    for percentile in PERCENTILES:
        stats[f"relative_p{percentile:g}"] = float(np.percentile(relative, percentile))
    return rgb, stats


def closest_source_points(position: np.ndarray, mapped: np.ndarray, source_mesh: trimesh.Trimesh) -> np.ndarray:
    """(N, 3) float64: where on the source each mapped texel, in row-major order, copied its color from."""
    if not mapped.any():
        raise ValueError("no texel is inside a UV chart")
    points, _, _ = closest_points(raycast_scene(source_mesh), position[mapped].astype(np.float64))
    return points.astype(np.float64)


def export_glb(mesh: trimesh.Trimesh, uv: np.ndarray, rgb: np.ndarray, path) -> Path:
    """Write a GLB with positions, UVs and `rgb` as its base-color texture, declared as the benchmark's."""
    array = np.asarray(rgb)
    if array.dtype != np.uint8:
        raise ValueError(f"texture must be uint8, got {array.dtype}")
    if len(uv) != len(mesh.vertices):
        raise ValueError(f"{len(uv)} UVs for {len(mesh.vertices)} vertices")
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(array),
                                                   baseColorFactor=[255, 255, 255, 255],
                                                   roughnessFactor=1.0, doubleSided=False)
    export = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False, validate=False,
                             visual=trimesh.visual.TextureVisuals(uv=uv, material=material))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    path.write_bytes(trimesh.exchange.gltf.export_glb(export, include_normals=False))
    match_benchmark_material(path)
    check_declaration(path)
    return path
