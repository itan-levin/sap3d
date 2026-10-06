"""Blender scene operations of the personalize stage: glTF import, normalization, and export.

Blender's world is Z-up; glTF files and the OBJ export used for surface sampling are Y-up, so points taken from
an exported mesh are in the glTF file's own frame.
"""

import os
import tempfile
from contextlib import contextmanager

import bpy
import numpy as np
import trimesh
from mathutils import Matrix, Vector

# Longest side of the training frame; the shape VAE's inputs lie within [-0.9999, 0.9999].
TRAINING_EXTENT = 1.9998


@contextmanager
def quiet():
    """Silence Blender's C-level output on stdout and stderr."""
    saved = os.dup(1), os.dup(2)
    sink = os.open(os.devnull, os.O_WRONLY)
    os.dup2(sink, 1)
    os.dup2(sink, 2)
    try:
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (*saved, sink):
            os.close(fd)


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")


def import_gltf(path, merge_vertices: bool) -> list:
    """Import a glTF/GLB file into an empty scene; returns its mesh objects."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no glTF file at {path}")
    reset_scene()
    with quiet():
        bpy.ops.import_scene.gltf(filepath=path, merge_vertices=merge_vertices, import_shading="NORMALS")
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise ValueError(f"{path} holds no mesh")
    for obj in meshes:
        if any(modifier.type == "ARMATURE" for modifier in obj.modifiers):
            raise ValueError(f"{path} holds a skinned mesh ({obj.name}); only static meshes are supported")
    return meshes


def _world_vertices(obj) -> np.ndarray:
    count = len(obj.data.vertices)
    local = np.empty(count * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", local)
    matrix = np.array(obj.matrix_world, dtype=np.float64)
    return local.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]


def joined_working_mesh(path):
    """Import `path` as one mesh object with identity transform: every part joined, world matrices baked
    into the vertices, UVs and materials kept. The imported originals are hidden from rendering."""
    meshes = import_gltf(path, merge_vertices=True)
    if len(meshes) > 1:
        uv_counts = {len(obj.data.uv_layers) for obj in meshes}
        uv_names = {layer.name for obj in meshes for layer in obj.data.uv_layers}
        if uv_counts == {1} and len(uv_names) > 1:
            for obj in meshes:
                obj.data.uv_layers[0].name = "UVMap"
        bpy.ops.object.select_all(action="DESELECT")
        for obj in meshes:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        if bpy.ops.object.join() != {"FINISHED"}:
            raise RuntimeError(f"joining the meshes of {path} failed")
    source = bpy.context.view_layer.objects.active if len(meshes) > 1 else meshes[0]
    mesh = source.data
    world = source.matrix_world
    vertices = [(world @ vertex.co)[:] for vertex in mesh.vertices]
    faces = [polygon.vertices[:] for polygon in mesh.polygons]
    material_indices = [polygon.material_index for polygon in mesh.polygons]

    data = bpy.data.meshes.new(name="working_mesh_data")
    data.from_pydata(vertices, [], faces)
    data.update()
    for layer in mesh.uv_layers:
        copy = data.uv_layers.new(name=layer.name)
        for index, loop in enumerate(mesh.loops):
            copy.data[index].uv = layer.data[loop.index].uv
    for material in mesh.materials:
        if material is not None:
            data.materials.append(material)
    for polygon, index in zip(data.polygons, material_indices):
        polygon.material_index = index

    working = bpy.data.objects.new(name="working_mesh", object_data=data)
    bpy.context.collection.objects.link(working)
    source.hide_render = True
    for obj in bpy.context.scene.objects:
        if obj.type == "EMPTY":
            obj.hide_render = True
    return working


def normalize_for_training(obj) -> dict:
    """Center the object's bounding box at the origin and scale its longest side to TRAINING_EXTENT, in place
    on the vertex data of an object with identity transform."""
    vertices = _world_vertices(obj)
    translation = -(vertices.max(axis=0) + vertices.min(axis=0)) / 2
    obj.location, obj.rotation_euler, obj.scale = (0, 0, 0), (0, 0, 0), (1, 1, 1)
    data = obj.data
    for vertex in data.vertices:
        vertex.co = vertex.co + Vector(translation)
    # Blender stores coordinates in single precision, so the box is measured again after the shift.
    centered = np.array([vertex.co[:] for vertex in data.vertices])
    low, high = centered.min(axis=0), centered.max(axis=0)
    recenter = (high + low) / 2
    for vertex in data.vertices:
        vertex.co = vertex.co - Vector(recenter)
    scale = TRAINING_EXTENT / float(((high - recenter) - (low - recenter)).max())
    for vertex in data.vertices:
        vertex.co = vertex.co * scale
    data.update()
    bpy.context.view_layer.update()
    final = np.array([vertex.co[:] for vertex in data.vertices])
    if np.abs(final).max() > 1.0:
        raise ValueError(f"the normalized mesh extends to {np.abs(final).max():.6f}, beyond [-1, 1]")
    return {"translation": (translation - recenter).tolist(), "scale": scale}


def export_trimesh(obj) -> trimesh.Trimesh:
    """The object as a triangulated trimesh, through Blender's OBJ exporter (Y-up, as the glTF file)."""
    handle, path = tempfile.mkstemp(suffix=".obj")
    os.close(handle)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    with quiet():
        bpy.ops.wm.obj_export(filepath=path, export_selected_objects=True, export_materials=False,
                              export_triangulated_mesh=True)
    loaded = trimesh.load(path, process=False)
    os.unlink(path)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate([g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)])
    return loaded


def normalize_glb(source, target) -> dict:
    """Write `source` to `target` (GLB) with its bounding box centered at the origin and its longest side 1,
    applied to the vertex data of every mesh; parts, materials and textures are kept."""
    meshes = import_gltf(source, merge_vertices=False)
    for obj in meshes:
        if obj.data.users > 1:
            obj.data = obj.data.copy()
    vertices = np.concatenate([_world_vertices(obj) for obj in meshes])
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    center = (low + high) / 2
    scale = 1.0 / float((high - low).max())
    normalization = Matrix.Diagonal((scale, scale, scale, 1.0)) @ Matrix.Translation(Vector(-center))
    worlds = [obj.matrix_world.copy() for obj in meshes]
    for obj, world in zip(meshes, worlds):
        obj.parent = None
        obj.data.transform(normalization @ world)
        obj.matrix_world = Matrix.Identity(4)
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    # A file without normals imports flat-shaded; writing Blender's flat normals would split every vertex.
    with_normals = any(obj.data.has_custom_normals for obj in meshes)
    with quiet():
        bpy.ops.export_scene.gltf(filepath=os.path.abspath(target), export_format="GLB", export_normals=with_normals)
    # Blender's world is Z-up; the transform is recorded in the glTF file's Y-up frame.
    to_file = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    return {"translation_blender": (-center).tolist(), "translation": (to_file @ -center).tolist(), "scale": scale,
            "source_bounds_blender": [low.tolist(), high.tolist()]}
