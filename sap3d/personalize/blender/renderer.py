"""Renders of the training views: one object, a camera on a sphere around the origin aimed at it, a lighting
setup named in LIGHTING ("hdri" builds the same setup as "ambient_dome"), and a white world behind a
transparent film."""

import bpy
import numpy as np
from mathutils import Vector

from .scene import quiet

LIGHTING = ("3_point", "multi_directional", "ambient_dome", "hdri")
ENGINES = ("CYCLES", "BLENDER_EEVEE")


def _enable_gpu() -> None:
    bpy.ops.preferences.addon_enable(module="cycles")
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for compute_type in ("CUDA", "OPTIX", "HIP"):
        prefs.compute_device_type = compute_type
        prefs.get_devices()
        found = False
        for device in prefs.devices:
            if device.type == compute_type:
                device.use = True
                found = True
            elif device.type == "CPU":
                device.use = False
        if found:
            return
    raise RuntimeError("Cycles found no GPU device; the training renders need one")


class Renderer:
    def __init__(self, size: int, samples: int, fov_deg: float, gpu: bool):
        self.size, self.samples, self.fov_deg, self.gpu = size, samples, fov_deg, gpu
        self.engine = None
        self.set_engine("CYCLES")

    def set_engine(self, engine: str) -> None:
        if engine not in ENGINES:
            raise ValueError(f"unknown render engine {engine!r}")
        self.engine = engine
        scene = bpy.context.scene
        scene.render.engine = engine
        scene.render.resolution_x = scene.render.resolution_y = self.size
        scene.render.film_transparent = True
        scene.view_settings.view_transform = "Standard"
        if engine == "CYCLES":
            if self.gpu:
                _enable_gpu()
            scene.cycles.device = "GPU" if self.gpu else "CPU"
            scene.cycles.samples = self.samples
            scene.cycles.use_denoising = True
        else:
            scene.eevee.taa_render_samples = self.samples
            scene.eevee.use_ssr = True
            scene.eevee.use_ssr_refraction = True

    def min_distance(self, obj, azimuth: float, elevation: float) -> float:
        """The smallest camera distance at which every corner of the object's bounding box lies inside the
        horizontal field of view, for a perspective camera at this azimuth and elevation (degrees)."""
        corners = np.array([obj.matrix_world @ Vector(corner) for corner in obj.bound_box])
        fov = bpy.data.objects["Camera"].data.angle_x if "Camera" in bpy.data.objects else np.radians(self.fov_deg)
        elev, azim = np.radians(elevation), np.radians(azimuth)
        view = np.array([np.cos(elev) * np.sin(azim), -np.cos(elev) * np.cos(azim), np.sin(elev)])
        world_up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(world_up, view)) > 0.999:
            world_up = np.array([0.0, 0.0, 1.0])
        up = world_up - np.dot(world_up, view) * view
        up /= np.linalg.norm(up)
        right = np.cross(view, up)
        right /= np.linalg.norm(right)
        tan_half = np.tan(fov / 2)
        distance = 0.0
        for corner in corners:
            offset = max(abs(np.dot(corner, right)), abs(np.dot(corner, up)))
            distance = max(distance, offset / tan_half + np.dot(corner, view))
        return distance

    def _camera(self, distance: float, elevation: float, azimuth: float) -> None:
        elev, azim = np.radians(elevation), np.radians(azimuth)
        location = (distance * np.cos(elev) * np.sin(azim), -distance * np.cos(elev) * np.cos(azim),
                    distance * np.sin(elev))
        if "Camera" not in bpy.data.objects:
            camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
            bpy.context.scene.collection.objects.link(camera)
        camera = bpy.data.objects["Camera"]
        camera.location = location
        camera.data.angle_x = np.radians(self.fov_deg)
        camera.rotation_euler = (Vector((0, 0, 0)) - camera.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.scene.camera = camera

    @staticmethod
    def _world():
        scene = bpy.context.scene
        if not scene.world:
            scene.world = bpy.data.worlds.new("World")
        world = scene.world
        world.use_nodes = True
        nodes = world.node_tree.nodes
        background = nodes.get("Background")
        if not background:
            background = nodes.new(type="ShaderNodeBackground")
            output = nodes.get("World Output") or nodes.new(type="ShaderNodeOutputWorld")
            world.node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])
        return background

    @staticmethod
    def _area(name, energy, size, location, rotation):
        light = bpy.data.lights.new(name, "AREA")
        light.energy, light.size = energy, size
        obj = bpy.data.objects.new(name, light)
        obj.location, obj.rotation_euler = location, rotation
        bpy.context.scene.collection.objects.link(obj)

    def _lighting(self, kind: str) -> None:
        for obj in list(bpy.data.objects):
            if obj.type == "LIGHT":
                bpy.data.objects.remove(obj)
        if kind == "3_point":
            self._area("KeyLight", 1000, 2, (3, -3, 3), (np.radians(45), 0, np.radians(45)))
            self._area("FillLight", 500, 3, (-3, -2, 2), (np.radians(60), 0, np.radians(-45)))
            self._area("RimLight", 800, 1, (0, 3, 2), (np.radians(-45), 0, 0))
        elif kind == "multi_directional":
            for i, position in enumerate([(3, 0, 2), (-3, 0, 2), (0, 3, 2), (0, -3, 2), (2, 2, 3), (-2, -2, 3)]):
                light = bpy.data.lights.new(f"Light_{i}", "POINT")
                light.energy = 500
                obj = bpy.data.objects.new(f"Light_{i}", light)
                obj.location = position
                bpy.context.scene.collection.objects.link(obj)
        elif kind in ("ambient_dome", "hdri"):
            self._area("DomeLight", 300, 5, (0, 0, 5), (0, 0, 0))
        else:
            raise ValueError(f"unknown lighting {kind!r}")

    def render(self, obj, elevation: float, azimuth: float, distance: float, lighting: str, path) -> None:
        """Render `obj` to `path` (RGBA PNG). The white world, at strength 1, also lights the object."""
        self._camera(distance, elevation, azimuth)
        self._lighting(lighting)
        background = self._world()
        background.inputs["Color"].default_value = (1, 1, 1, 1)
        background.inputs["Strength"].default_value = 1.0
        bpy.context.scene.render.filepath = str(path)
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        with quiet():
            bpy.ops.render.render(write_still=True)
