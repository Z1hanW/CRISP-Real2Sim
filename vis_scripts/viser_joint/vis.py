#!/usr/bin/env python3
"""
Joint visualization entrypoint that lets us inspect the Viser mesh/robot data
alongside the Megasam point cloud + HMR reconstruction in a single viewer.

The script mirrors the existing pipelines but keeps the original files intact.
It exposes a checkbox to overlay the Megasam results and a small button group
to switch between the two views, all driven by a single timeline slider.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import yaml

# Ensure we can import the helper modules that live in the neighbouring folders.
REPO_ROOT = Path(__file__).resolve().parent.parent
VISER_PATH = REPO_ROOT / "viser"
if VISER_PATH.exists():
    sys.path.insert(0, str(VISER_PATH))

try:
    from utils.viser_visualizer_z import ViserHelper  # type: ignore  # noqa: E402
    from utils.robot_viser_z import RobotMjcfViser  # type: ignore  # noqa: E402
except ImportError:
    from viser.utils.viser_visualizer_z import ViserHelper  # type: ignore  # noqa: E402
    from viser.utils.robot_viser_z import RobotMjcfViser  # type: ignore  # noqa: E402

VISER_M_PATH = REPO_ROOT / "viser_m"
if VISER_M_PATH.exists():
    sys.path.insert(0, str(VISER_M_PATH))

from viser.extras import Record3dLoader_Customized_Megasam  # type: ignore  # noqa: E402

from smpl import SMPL  # type: ignore  # noqa: E402
from smpl_utils import process_gv_smpl  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint mesh + Megasam viewer.")
    parser.add_argument(
        "--scene",
        required=True,
        help="Sequence name / scene id (e.g. 0831_19_indoor_walk_off_mvs).",
    )
    parser.add_argument(
        "--types",
        required=True,
        help="Method suffix matching naming used in the robot recordings (e.g. ours).",
    )
    parser.add_argument(
        "--parent_name",
        default="emdb_new",
        help="Parent folder that groups the scene (defaults to emdb_new).",
    )
    parser.add_argument(
        "--method",
        default="emdb_new",
        help="Method name used when locating the mesh and robot assets.",
    )
    parser.add_argument(
        "--data_path",
        default="/data3/zihanwa3/_Robotics/humanoid/hybrid-imitation-parkour/motion_data/",
        help="Root folder that stores *_types.npz with T_align.",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Optional explicit path to the Megasam postprocess NPZ file.",
    )
    parser.add_argument("--env_idx", type=int, default=0, help="Environment index.")
    parser.add_argument("--port", type=int, default=None, help="Viser server port.")
    parser.add_argument(
        "--point_size",
        type=float,
        default=0.01,
        help="Point size used when rendering the Megasam point cloud.",
    )
    parser.add_argument(
        "--point_downsample",
        type=int,
        default=2,
        help="Downsample factor for building Megasam point clouds.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=250_000,
        help="Optional cap on the number of Megasam points kept per frame.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to playback.",
    )
    parser.add_argument(
        "--hmr_type",
        choices=["gv", "tram"],
        default="gv",
        help="HMR flavour to load. The alignment assumes GV by default.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device used for SMPL processing.",
    )
    return parser.parse_args()


def random_port() -> int:
    return random.randint(20000, 25000)


def load_scene_mesh(scene_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Utility copied from the Viser mesh script to gather OBJ data."""
    if not scene_path.exists():
        return None, None
    if scene_path.is_dir():
        parts_dir = scene_path / "parts" if (scene_path / "parts").exists() else scene_path
        part_files = sorted(parts_dir.glob("part_*.obj"))
        if not part_files:
            part_files = sorted(parts_dir.glob("*.obj"))
        if not part_files:
            return None, None
        all_vertices: list[np.ndarray] = []
        all_faces: list[np.ndarray] = []
        vertex_offset = 0
        for part_file in part_files:
            vertices: list[list[float]] = []
            faces: list[list[int]] = []
            with open(part_file, "r") as f:
                for line in f:
                    if line.startswith("v "):
                        _, x, y, z = line.strip().split()[:4]
                        vertices.append([float(x), float(y), float(z)])
                    elif line.startswith("f "):
                        idxs = [int(seg.split("/")[0]) - 1 for seg in line.strip().split()[1:]]
                        for i in range(1, len(idxs) - 1):
                            faces.append([idxs[0] + vertex_offset, idxs[i] + vertex_offset, idxs[i + 1] + vertex_offset])
            vertex_offset += len(vertices)
            if vertices:
                all_vertices.append(np.asarray(vertices, dtype=np.float32))
            if faces:
                all_faces.append(np.asarray(faces, dtype=np.int32))
        if not all_vertices or not all_faces:
            return None, None
        return np.concatenate(all_vertices, axis=0), np.concatenate(all_faces, axis=0)
    if scene_path.suffix.lower() == ".obj":
        vertices: list[list[float]] = []
        faces: list[list[int]] = []
        with open(scene_path, "r") as f:
            for line in f:
                if line.startswith("v "):
                    _, x, y, z = line.strip().split()[:4]
                    vertices.append([float(x), float(y), float(z)])
                elif line.startswith("f "):
                    idxs = [int(seg.split("/")[0]) - 1 for seg in line.strip().split()[1:]]
                    for i in range(1, len(idxs) - 1):
                        faces.append([idxs[0], idxs[i], idxs[i + 1]])
        if vertices and faces:
            return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)
    return None, None


def _load_npz_to_dict(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a homogeneous 4x4 transform to points shaped (N, 3) or (T, V, 3)."""
    if points.ndim == 3:
        flat = points.reshape(-1, 3)
        transformed = apply_transform(flat, transform)
        return transformed.reshape(points.shape[0], points.shape[1], 3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Unsupported shape for transform: {points.shape}")
    pts = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    transformed = (transform @ pts.T).T[:, :3]
    return transformed.astype(points.dtype, copy=False)


def _gather_robot_handles(helper: ViserHelper, prefix: str = "/robot") -> Iterable:
    return [
        handle
        for name, handle in helper._handles.items()  # type: ignore[attr-defined]
        if name.startswith(prefix)
    ]


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    types = args.types
    scene = args.scene
    parent_name = args.parent_name

    # Reproduce the directory resolution logic used by the mesh viewer.
    if "_" in args.method:
        method_core = args.method.split("_")[0]
        types = args.types.split("_")[0]
        scene_obj = Path(
            f"/data3/zihanwa3/_Robotics/humanoid/hybrid-imitation-parkour/parkour_anim/data/assets/urdf/{parent_name}/{scene}/{method_core}"
        )
        record_dir = Path(
            f"/data3/zihanwa3/_Robotics/humanoid/hybrid-imitation-parkour/post_asset_trimesh_{parent_name}/{scene}_{method_core}"
        )
    else:
        method_core = args.method
        scene_obj = Path(
            f"/data3/zihanwa3/_Robotics/humanoid/hybrid-imitation-parkour/parkour_anim/data/assets/urdf/{parent_name}/{scene}/{method_core}"
        )
        record_dir = Path(
            f"/data3/zihanwa3/_Robotics/humanoid/hybrid-imitation-parkour/post_asset_{method_core}_{parent_name}/{scene}"
        )

    if not record_dir.exists():
        raise FileNotFoundError(f"Robot record directory not found: {record_dir}")

    raw_data_path = Path(args.data_path) / parent_name / f"{scene}_{types}.npz"
    if not raw_data_path.exists():
        raise FileNotFoundError(f"T_align npz does not exist: {raw_data_path}")

    raw_data = np.load(raw_data_path)
    T_align = raw_data["T_align"].astype(np.float32)

    # Resolve Megasam NPZ location.
    if args.data is None:
        args.data = f"/data3/zihanwa3/_Robotics/_vision/mega-sam/postprocess/{scene}_gv_sgd_cvd_hr.npz"
    data_npz_path = Path(args.data)
    if not data_npz_path.exists():
        raise FileNotFoundError(f"Megasam NPZ not found: {data_npz_path}")

    tgt_name = data_npz_path.stem.split("_sgd")[0]
    tgt_name = "_".join(tgt_name.split("_")[:-1])
    moge_base_path = Path("/data3/zihanwa3/_Robotics/_vision/TAPIP3D/_raw_mega_priors")
    moge_npz_path = moge_base_path / f"{tgt_name}.npz"
    if not moge_npz_path.exists():
        raise FileNotFoundError(f"Moge NPZ not found: {moge_npz_path}")

    megasam_data = _load_npz_to_dict(data_npz_path)
    moge_data = np.load(moge_npz_path)
    # Inject the image/depth/camera payloads expected by the loader.
    for key in ("depths", "images", "cam_c2w", "intrinsic"):
        megasam_data[key] = moge_data[key]

    scale = megasam_data.get("scale", 1.0)
    cam_c2w = megasam_data["cam_c2w"].astype(np.float32)
    if np.isscalar(scale):
        cam_trans = cam_c2w[:, :3, 3] * float(scale)
    else:
        scale_flat = np.asarray(scale, dtype=np.float32).reshape(-1, 1)
        if scale_flat.shape[0] == 1:
            scale_flat = np.repeat(scale_flat, cam_c2w.shape[0], axis=0)
        cam_trans = cam_c2w[:, :3, 3] * scale_flat

    vis_helper = ViserHelper(port=args.port or random_port())
    if not vis_helper.ok():
        raise RuntimeError("Failed to start Viser server.")
    server = vis_helper.server
    server.scene.set_up_direction("-z")

    # Load robot metadata.
    info_path = record_dir / "robot_vis_info.yaml"
    if info_path.exists():
        info = yaml.safe_load(info_path.read_text())
        body_names = info.get("body_names", [])
        dt = float(info.get("dt", 1.0 / 30.0))
        asset_xml_rel = info.get("asset_xml")
    else:
        body_names = []
        dt = 1.0 / 30.0
        asset_xml_rel = None

    if asset_xml_rel and (record_dir / asset_xml_rel).exists():
        mjcf_path = str(record_dir / asset_xml_rel)
    else:
        candidates = list(record_dir.glob("*.xml"))
        if not candidates:
            raise FileNotFoundError("Could not find MJCF xml in record dir.")
        mjcf_path = str(candidates[0])

    robot = RobotMjcfViser(vis_helper, mjcf_path, body_names if body_names else None)

    rigid_bodies = np.load(record_dir / f"rigid_bodies_{args.env_idx}.npz")
    pos = rigid_bodies["pos"].astype(np.float32)
    rot = rigid_bodies["rot"].astype(np.float32)

    # Optional scene mesh for visual reference.
    scene_handle = None
    if scene_obj.exists():
        scene_vertices, scene_faces = load_scene_mesh(scene_obj)
        if scene_vertices is not None and scene_faces is not None:
            vis_helper.add_mesh_simple(
                "/scene",
                scene_vertices,
                scene_faces,
                color=(0.6, 0.7, 0.9),
            )
            scene_handle = vis_helper._handles.get("/scene")  # type: ignore[attr-defined]

    # Prepare Megasam loader + SMPL processing.
    loader = Record3dLoader_Customized_Megasam(
        megasam_data,
        megasam_data,
        conf_threshold=1.0,
        foreground_conf_threshold=0.0,
        no_mask=False,
        xyzw=True,
        init_conf=False,
    )

    world_cam_R = torch.as_tensor(cam_c2w[:, :3, :3], device=device)
    world_cam_T = torch.as_tensor(cam_trans, device=device)

    smpl_model = SMPL().to(device)
    hmr_results = (
        process_gv_smpl(
            tgt_name=tgt_name,
            world_cam_R=world_cam_R,
            world_cam_T=world_cam_T,
            max_frames=args.max_frames or loader.num_frames(),
            smpl_model=smpl_model,
            device=str(device),
        )
        if args.hmr_type == "gv"
        else process_gv_smpl(  # fallback to GV pipeline for now
            tgt_name=tgt_name,
            world_cam_R=world_cam_R,
            world_cam_T=world_cam_T,
            max_frames=args.max_frames or loader.num_frames(),
            smpl_model=smpl_model,
            device=str(device),
        )
    )

    pred_vert = hmr_results["pred_vert"].detach().cpu().numpy().astype(np.float32)
    pred_vert_aligned = apply_transform(pred_vert, T_align)
    faces = np.asarray(hmr_results["faces"], dtype=np.int32)

    num_robot_frames = pos.shape[0]
    num_megasam_frames = loader.num_frames()
    num_hmr_frames = pred_vert_aligned.shape[0]
    timeline_length = min(num_robot_frames, num_megasam_frames, num_hmr_frames)
    if args.max_frames is not None:
        timeline_length = min(timeline_length, args.max_frames)
    if timeline_length <= 0:
        raise RuntimeError("No overlapping frames available for visualization.")

    pos = pos[:timeline_length]
    rot = rot[:timeline_length]
    pred_vert_aligned = pred_vert_aligned[:timeline_length]

    megasam_cache: Dict[int, Tuple[np.ndarray, Optional[np.ndarray]]] = {}

    def sample_point_cloud(frame_idx: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        frame_idx = int(frame_idx)
        if frame_idx in megasam_cache:
            return megasam_cache[frame_idx]
        frame = loader.get_frame(frame_idx)
        (points_bundle, _, _extras) = frame.get_point_cloud(
            downsample_factor=max(1, args.point_downsample),
            bg_downsample_factor=1,
            world_coords=True,
        )
        points_all = points_bundle[4].astype(np.float32)
        if points_all.ndim == 3:
            points_all = points_all.reshape(-1, 3)
        colors_all = points_bundle[5]
        if colors_all is not None:
            colors_all = np.asarray(colors_all)
            if colors_all.ndim == 3:
                colors_all = colors_all.reshape(-1, colors_all.shape[-1])
            else:
                colors_all = colors_all.reshape(-1, 3)
            colors_all = colors_all.astype(np.uint8, copy=False)
        points_world = apply_transform(points_all, T_align)
        finite_mask = np.isfinite(points_world).all(axis=1)
        points_world = points_world[finite_mask]
        if colors_all is not None:
            colors_all = colors_all[finite_mask]
        if args.max_points and points_world.shape[0] > args.max_points:
            indices = np.random.choice(points_world.shape[0], args.max_points, replace=False)
            points_world = points_world[indices]
            if colors_all is not None:
                colors_all = colors_all[indices]
        megasam_cache[frame_idx] = (points_world.astype(np.float32), colors_all)
        return megasam_cache[frame_idx]

    # Initialise Megasam handles so we can toggle them later.
    pc_init, color_init = sample_point_cloud(0)
    point_handle = server.scene.add_point_cloud(
        name="/megasam/point_cloud",
        points=pc_init,
        colors=color_init,
        point_size=args.point_size,
        point_shape="rounded",
    )
    hmr_handle = server.scene.add_mesh_simple(
        name="/megasam/hmr",
        vertices=pred_vert_aligned[0],
        faces=faces,
        color=(255, 170, 120),
        flat_shading=False,
        wireframe=False,
    )

    robot_handles = list(_gather_robot_handles(vis_helper))

    def update_visibility() -> None:
        mesh_visible = (view_mode.value == "Mesh") or overlay_checkbox.value
        megasam_visible = (view_mode.value == "Megasam") or overlay_checkbox.value
        if scene_handle is not None:
            scene_handle.visible = mesh_visible
        for handle in robot_handles:
            handle.visible = mesh_visible
        point_handle.visible = megasam_visible
        hmr_handle.visible = megasam_visible

    def update_frame(frame_idx: int) -> None:
        frame_idx = max(0, min(timeline_length - 1, frame_idx))
        robot.update(pos[frame_idx], rot[frame_idx])
        pc_points, pc_colors = sample_point_cloud(frame_idx)
        point_handle.points = pc_points.astype(np.float32)
        if pc_colors is not None:
            point_handle.colors = pc_colors.astype(np.uint8)
        point_handle.point_size = args.point_size
        hmr_handle.vertices = pred_vert_aligned[frame_idx].astype(np.float32)

    # GUI wiring -------------------------------------------------------------
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=timeline_length - 1,
            step=1,
            initial_value=0,
        )
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_fps = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=int(round(1.0 / dt)))

    with server.gui.add_folder("View"):
        overlay_checkbox = server.gui.add_checkbox("Show Megasam overlay", False)
        view_mode = server.gui.add_button_group("Active View", ("Mesh", "Megasam"))
        view_mode.value = "Mesh"

    @gui_timestep.on_update
    def _(_) -> None:
        update_frame(int(gui_timestep.value))

    @overlay_checkbox.on_update
    def _(_) -> None:
        update_visibility()

    @view_mode.on_update
    def _(_) -> None:
        update_visibility()

    update_frame(0)
    update_visibility()

    root0 = pos[0, 0]
    cam_pos = root0 + np.array([0.0, -2.0, 1.5], dtype=np.float32)
    cam_look = root0 + np.array([0.0, 0.0, 0.4], dtype=np.float32)
    vis_helper.set_camera(cam_pos, cam_look)

    print(f"[JointVis] running on http://{server.get_host()}:{server.get_port()}")
    try:
        while True:
            if gui_playing.value:
                next_frame = (int(gui_timestep.value) + 1) % timeline_length
                gui_timestep.value = next_frame
                time.sleep(1.0 / max(gui_fps.value, 1))
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[JointVis] exiting...")


if __name__ == "__main__":
    main()
