import argparse
import json
import multiprocessing as mp
import os
import shutil
from functools import partial
from glob import glob

import imageio
import numpy as np
from bpyrenderer import SceneManager
from bpyrenderer.camera import add_camera
from bpyrenderer.camera.layout import get_camera_positions_on_sphere
from bpyrenderer.engine import init_render_engine
from bpyrenderer.environment import set_background_color, set_env_map
from bpyrenderer.importer import load_armature, load_file
from bpyrenderer.render_output import enable_color_output
from bpyrenderer.utils import convert_normal_to_webp
from tqdm import tqdm


def render_images(input_model, output_dir):
    # 1. Init engine and scene manager
    # init_render_engine("BLENDER_EEVEE")
    init_render_engine("CYCLES")
    scene_manager = SceneManager()
    scene_manager.clear(reset_keyframes=True)

    # 2. Import models
    load_file(input_model)

    # Others. smooth objects and normalize scene
    # scene_manager.smooth()
    scene_manager.clear_normal_map()
    scene_manager.set_material_transparency(False)
    scene_manager.set_materials_opaque()  # !!! Important for render normal but may cause render error !!!
    # scene_manager.normalize_scene(1.0) # Not normalize here

    # 3. Set environment
    set_env_map("assets/brown_photostudio_02_1k.exr")
    # set_background_color([1.0, 1.0, 1.0, 1.0])

    # 4. Prepare cameras
    cam_pos, cam_mats, elevations, azimuths = get_camera_positions_on_sphere(
        center=(0, 0, 0),
        radius=1.8,
        elevations=[15, 30],
        azimuths=[item - 90 for item in [0, 45, 90, 135, 180, 225, 270, 315]],
    )
    cameras = []
    for i, camera_mat in enumerate(cam_mats):
        camera = add_camera(camera_mat, "PERSP", add_frame=i < len(cam_mats) - 1)
        cameras.append(camera)

    # 5. Set render outputs
    # for image
    width, height = 1024, 1024
    enable_color_output(
        width,
        height,
        output_dir,
        file_format="PNG",
        mode="IMAGE",
        film_transparent=True,
    )

    scene_manager.render()


def render_videos(input_model, output_dir):
    # 1. Init engine and scene manager
    # init_render_engine("BLENDER_EEVEE")
    init_render_engine("CYCLES")
    scene_manager = SceneManager()
    scene_manager.clear(reset_keyframes=True)

    # 2. Import models
    load_file(input_model)

    # Others. smooth objects and normalize scene
    # scene_manager.smooth()
    scene_manager.clear_normal_map()
    scene_manager.set_material_transparency(False)
    scene_manager.set_materials_opaque()  # !!! Important for render normal but may cause render error !!!
    # scene_manager.normalize_scene(1.0) # Not normalize here

    # 3. Set environment
    set_env_map("assets/brown_photostudio_02_1k.exr")
    # set_background_color([1.0, 1.0, 1.0, 1.0])

    # 4. Prepare cameras
    cam_pos, cam_mats, elevations, azimuths = get_camera_positions_on_sphere(
        center=(0, 0, 0),
        radius=1.8,
        elevations=[15],
        num_camera_per_layer=120,
        azimuth_offset=-90,  # forward issue
    )
    cameras = []
    for i, camera_mat in enumerate(cam_mats):
        camera = add_camera(camera_mat, "PERSP", add_frame=i < len(cam_mats) - 1)
        cameras.append(camera)

    # 5. Set render outputs
    # for video
    width, height, fps = 1024, 1024, 24
    enable_color_output(
        width,
        height,
        output_dir,
        mode="PNG",
        film_transparent=True,
    )

    scene_manager.render()

    # for video
    render_files = sorted(glob(os.path.join(output_dir, "render_*.png")))
    if render_files:
        # Create videos for white background and mask
        white_video_path = os.path.join(output_dir, "video_rgb.mp4")
        mask_video_path = os.path.join(output_dir, "video_mask.mp4")

        with imageio.get_writer(
            white_video_path, fps=fps
        ) as white_writer, imageio.get_writer(mask_video_path, fps=fps) as mask_writer:

            for file in render_files:
                # Read RGBA image
                image = imageio.imread(file)
                mask = image[:, :, 3]
                white_bg = np.ones((height, width, 3), dtype=np.uint8) * 255

                alpha = image[:, :, 3:4] / 255.0
                white_image = image[:, :, :3] * alpha + white_bg * (1 - alpha)

                white_writer.append_data(white_image.astype(np.uint8))
                mask_writer.append_data(mask)
                os.remove(file)


def find_all_edit_glb_files(base_dir, file_name="edit.glb"):
    pattern = os.path.join(base_dir, "**", file_name)
    files = glob(pattern, recursive=True)
    return sorted(files)


def create_output_dir(model_path, mode="images"):
    model_dir = os.path.dirname(model_path)
    output_dir = os.path.join(model_dir, mode)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    args = parser.parse_args()

    base_dir = args.base_dir

    print("Loading models...")
    all_models = sorted(find_all_edit_glb_files(base_dir, file_name="edit.glb"))
    print(f"Found {len(all_models)} models")

    if len(all_models) == 0:
        print("No models found, please check the path")
        exit(1)

    results = []
    failed_models = []
    for model_path in all_models:
        try:
            output_dir = create_output_dir(model_path, mode="images")
            render_images(model_path, output_dir)

            output_dir = create_output_dir(model_path, mode="videos")
            render_videos(model_path, output_dir)
            results.append(True)
        except Exception as e:
            print(f"Render failed: {model_path} - Error: {str(e)}")
            failed_models.append(model_path)

    print(f"\nRender completed!")
    print(f"Success: {len(results)}")
    print(f"Failed: {len(failed_models)}")
    print(f"Render results saved to images folder under each model directory")
    print(f"Failed models: {failed_models}")
