import argparse
import json
import os
import sys
from pathlib import Path

import bpy
from mathutils import Matrix


def _enable_cycles_gpu(scene):
    try:
        prefs = bpy.context.preferences
        cycles_prefs = prefs.addons["cycles"].preferences

        # Try modern backends first, then CPU fallback.
        tried = ["OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"]
        backend_set = False
        for backend in tried:
            try:
                cycles_prefs.compute_device_type = backend
                backend_set = True
                break
            except Exception:
                continue
        if not backend_set:
            cycles_prefs.compute_device_type = "NONE"

        try:
            cycles_prefs.get_devices()
        except Exception:
            pass

        any_gpu = False
        for d in getattr(cycles_prefs, "devices", []):
            # Keep CPU disabled for render speed, enable all non-CPU devices.
            if getattr(d, "type", "") != "CPU":
                d.use = True
                any_gpu = True
            else:
                d.use = False

        scene.cycles.device = "GPU" if any_gpu else "CPU"
    except Exception:
        # Safe fallback to CPU if environment has no GPU support.
        scene.cycles.device = "CPU"


def _reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    _enable_cycles_gpu(scene)
    scene.cycles.samples = 64
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.view_layers["ViewLayer"].use_pass_z = True
    return scene


def _setup_world(hdr_path: str):
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    bg = nt.nodes.new(type="ShaderNodeBackground")
    env = nt.nodes.new(type="ShaderNodeTexEnvironment")
    out = nt.nodes.new(type="ShaderNodeOutputWorld")
    env.image = bpy.data.images.load(hdr_path)

    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def _setup_depth_output(depth_dir: str):
    scene = bpy.context.scene
    scene.use_nodes = True
    nt = scene.node_tree
    nt.nodes.clear()
    rl = nt.nodes.new(type="CompositorNodeRLayers")
    fo = nt.nodes.new(type="CompositorNodeOutputFile")
    fo.base_path = depth_dir
    fo.format.file_format = "OPEN_EXR"
    fo.format.color_mode = "RGB"
    fo.format.color_depth = "32"
    nt.links.new(rl.outputs["Depth"], fo.inputs[0])
    return fo


def _set_camera_intrinsics(cam_data, width, height, fx, fy, cx, cy):
    sensor_width = 36.0
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = sensor_width
    cam_data.lens = float(fx) * sensor_width / float(width)
    cam_data.shift_x = -((float(cx) - (float(width) - 1.0) * 0.5) / float(width))
    cam_data.shift_y = (float(cy) - (float(height) - 1.0) * 0.5) / float(height)


def _cv_to_blender_c2w(c2w_cv):
    cv_to_blender = Matrix(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    c2w = Matrix(c2w_cv)
    return c2w @ cv_to_blender


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=str, required=True)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    args = parser.parse_args(argv)

    with open(args.job, "r", encoding="utf-8") as f:
        job = json.load(f)

    mesh_path = job["mesh_path"]
    hdr_path = job["hdr_path"]
    out_dir = Path(job["out_dir"])
    width = int(job["width"])
    height = int(job["height"])
    targets = job["targets"]

    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = out_dir / "rgb"
    depth_dir = out_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    scene = _reset_scene()
    _setup_world(hdr_path)
    depth_node = _setup_depth_output(str(depth_dir))

    bpy.ops.import_scene.gltf(filepath=mesh_path)

    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100

    for i, t in enumerate(targets):
        K = t["K"]
        c2w = t["c2w"]
        fx = K[0][0]
        fy = K[1][1]
        cx = K[0][2]
        cy = K[1][2]
        _set_camera_intrinsics(cam_data, width, height, fx, fy, cx, cy)
        cam_obj.matrix_world = _cv_to_blender_c2w(c2w)

        rgb_path = rgb_dir / f"{i:04d}.png"
        depth_node.file_slots[0].path = f"depth_{i:04d}_"
        scene.render.filepath = str(rgb_path)
        bpy.ops.render.render(write_still=True)

    with open(out_dir / "done.txt", "w", encoding="utf-8") as f:
        f.write("ok")


if __name__ == "__main__":
    main()
