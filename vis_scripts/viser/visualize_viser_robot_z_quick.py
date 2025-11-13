#!/usr/bin/env python3
import argparse
import time
import os
from pathlib import Path
import numpy as np
import yaml
import cv2
import imageio.v3 as iio
from contextlib import suppress
from typing import Optional

from utils.viser_visualizer_z import ViserHelper
from utils.robot_viser_z import RobotMjcfViser
# '/data3/zihanwa3/_Robotics/humanoid/hybrid-imitation-parkour/gggg.obj'
def load_scene_mesh(scene_path):
    """
    Load scene mesh, handling both single file and multi-part formats.
    
    Args:
        scene_path: Path to scene file or directory containing parts
    
    Returns:
        tuple: (vertices, faces) as numpy arrays, or (None, None) if failed
    """
    scene_path = Path(scene_path)
    
    
    # Check if it's a directory with parts
    if scene_path.is_dir():
        parts_dir = scene_path / "parts" if (scene_path / "parts").exists() else scene_path
        part_files = sorted(parts_dir.glob("part_*.obj"))
        
        if not part_files:
            # Try other patterns
            part_files = sorted(parts_dir.glob("part_*.obj"))
            if not part_files:
                part_files = sorted(parts_dir.glob("*.obj"))
        
        if part_files:
            print(f"[Scene] Found {len(part_files)} parts to concatenate")
            all_vertices = []
            all_faces = []
            vertex_offset = 0
            
            for part_file in part_files:
                with open(part_file, 'r') as f:
                    vertices = []
                    for line in f:
                        if line.startswith('v '):
                            parts = line.strip().split()
                            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                        elif line.startswith('f '):
                            face_parts = line.strip().split()[1:]
                            face_indices = []
                            for p in face_parts:
                                vidx = int(p.split('/')[0]) - 1 + vertex_offset
                                face_indices.append(vidx)
                            # Triangulate if needed
                            for i in range(1, len(face_indices) - 1):
                                all_faces.append([face_indices[0], face_indices[i], face_indices[i + 1]])
                    
                    all_vertices.extend(vertices)
                    vertex_offset += len(vertices)
            
            return np.array(all_vertices, dtype=np.float32), np.array(all_faces, dtype=np.int32)
    
    # Single file format
    elif scene_path.exists() and scene_path.suffix == '.obj':
        print(f"[Scene] Loading single OBJ file: {scene_path}")
        vertices = []
        faces = []
        with open(scene_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):
                    face_parts = line.strip().split()[1:]
                    face_indices = []
                    for p in face_parts:
                        vidx = int(p.split('/')[0]) - 1
                        face_indices.append(vidx)
                    # Triangulate
                    for i in range(1, len(face_indices) - 1):
                        faces.append([face_indices[0], face_indices[i], face_indices[i + 1]])
        
        return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)
    
    print(f"[Scene] Could not load scene from: {scene_path}")
    return None, None
    
class ExternalCameraHandler:
    """Mirror the dataset camera logic so renders follow recorded intrinsics/extrinsics."""

    def __init__(self, npz_data: dict, npz_cam_data: dict, T_align: np.ndarray):
        self.K = np.expand_dims(npz_cam_data["intrinsic"], 0).astype(np.float32)
        ax = ay = 1.0
        self.K[0][0][0] = ax * self.K[0][0][0]
        self.K[0][0][-1] = ax * self.K[0][0][-1]
        self.K[0][1][1] = ay * self.K[0][1][1]
        self.K[0][1][-1] = ay * self.K[0][1][-1]
        self.K[0][1][1] = self.K[0][0][0]

        self.S = float(npz_cam_data.get("scale", 1.0))
        num_frames = npz_data["images"].shape[0]
        self.K = np.repeat(self.K, num_frames, axis=0)

        T_world_cameras = npz_cam_data["cam_c2w"].copy()
        T_world_cameras[..., :3, 3] *= self.S
        if T_align is not None:
            self.T_world_cameras = np.matmul(T_align, T_world_cameras)
        else:
            self.T_world_cameras = T_world_cameras

        self.images = npz_data["images"]
        self.depths = npz_data.get("depths", np.zeros_like(self.images[..., 0])).astype(np.float32) * self.S
        self.fps = float(npz_cam_data.get("fps", 30))
        self.masks = npz_data.get("enlarged_dynamic_mask", np.zeros_like(self.images[..., 0]))
        self.confidences = np.array(npz_data.get("uncertainty", []))

        if self.masks.shape != self.images.shape[:3]:
            try:
                import skimage.transform

                self.masks = skimage.transform.resize(self.masks, self.images.shape[:3], order=0).astype(np.uint8)
            except Exception:
                self.masks = np.zeros_like(self.images[..., 0], dtype=np.uint8)

        if len(self.confidences):
            self.conf_threshold = np.quantile(self.confidences, 0.0)
        else:
            self.conf_threshold = 1.0

    def get_camera_at_frame(self, frame_idx: int):
        frame_idx = min(frame_idx, len(self.T_world_cameras) - 1)
        return {
            "K": self.K[frame_idx],
            "T_c2w": self.T_world_cameras[frame_idx],
            "image": self.images[frame_idx],
            "depth": self.depths[frame_idx] if frame_idx < len(self.depths) else None,
            "H": self.images[frame_idx].shape[0],
            "W": self.images[frame_idx].shape[1],
        }


def _k_to_fov_y(K: np.ndarray, H: int) -> float:
    fy = float(K[1, 1])
    return float(2.0 * np.arctan2(H, 2.0 * fy))


def _twc_to_cam_pose(T_wc: np.ndarray):
    import viser.transforms as vtf

    R_wc = np.asarray(T_wc[:3, :3], dtype=float)
    t_wc = np.asarray(T_wc[:3, 3], dtype=float)
    q_wxyz = vtf.SO3.from_matrix(R_wc).wxyz
    return t_wc.astype(float), np.asarray(q_wxyz, dtype=float).reshape(4,)


def _normalize_host_for_browser(host: str) -> str:
    if host in ("0.0.0.0", "::", "::0", "", None):
        return "127.0.0.1"
    return host


_PLAYWRIGHT_ARGS = [
    "--disable-gpu",
    "--disable-dev-shm-usage",
    "--no-sandbox",
    "--disable-setuid-sandbox",
]


def _get_client_or_launch(server, timeout_sec: float = 60.0):
    host, port = server.get_host(), server.get_port()
    view_host = _normalize_host_for_browser(host)
    url = f"http://{view_host}:{port}"
    clients = server.get_clients()
    if clients:
        return list(clients.values())[0], None

    launched = None
    with suppress(Exception):
        launched = _launch_playwright_browser(url)

    t0 = time.time()
    while time.time() - t0 <= timeout_sec:
        clients = server.get_clients()
        if clients:
            return list(clients.values())[0], launched
        time.sleep(0.05)

    if launched is not None:
        _close_launched(launched)

    raise TimeoutError(
        f"No Viser client connected within {timeout_sec:.0f}s. "
        f"Please open {url} in a browser and rerun, or install Playwright for headless capture."
    )


def _load_npz_to_dict(path: str) -> dict:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def _close_launched(launched):
    if launched is None:
        return
    pw, browser, page = launched
    with suppress(Exception):
        page.close()
        browser.close()
        pw.stop()


def _launch_playwright_browser(url: str):
    """Launch Chromium via Playwright, falling back to any cached binary if needed."""
    from playwright.sync_api import sync_playwright

    force_fallback = bool(os.environ.get("VISER_FORCE_PLAYWRIGHT_FALLBACK"))
    custom_binary = os.environ.get("VISER_HEADLESS_CHROME")
    pw = sync_playwright().start()

    def _launch(executable_path: Optional[str] = None):
        return pw.chromium.launch(
            headless=True,
            executable_path=executable_path,
            args=_PLAYWRIGHT_ARGS,
        )

    browser = None
    launch_error: Optional[Exception] = None
    if not force_fallback:
        try:
            browser = _launch()
        except Exception as exc:  # Playwright raises when browser binary missing
            launch_error = exc

    if browser is None:
        fallback_path = _resolve_chromium_binary(custom_binary)
        if fallback_path is None:
            if launch_error:
                pw.stop()
                raise launch_error
            pw.stop()
            raise RuntimeError(
                "No Playwright Chromium binary is available. Run "
                "`playwright install chromium` or set VISER_HEADLESS_CHROME to an existing executable."
            )
        if launch_error is not None:
            print(f"[Viser] Playwright binary missing ({launch_error}); trying {fallback_path}")
        else:
            print(f"[Viser] Forcing fallback browser: {fallback_path}")
        browser = _launch(str(fallback_path))

    page = browser.new_page(viewport={"width": 1280, "height": 800, "deviceScaleFactor": 1.0})
    page.goto(url, wait_until="load")
    return pw, browser, page


def _resolve_chromium_binary(override_path: Optional[str]):
    """Search for an existing chromium/chrome/headless_shell binary Playwright can reuse."""
    candidates: list[Path] = []
    if override_path:
        path = Path(override_path).expanduser()
        if path.exists():
            return path
        print(f"[Viser] VISER_HEADLESS_CHROME={path} does not exist.")

    browser_root_env = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    search_roots = []
    if browser_root_env:
        search_roots.append(Path(browser_root_env))
    search_roots.append(Path.home() / ".cache" / "ms-playwright")

    def _build_key(p: Path) -> int:
        suffix = p.name.split("-")[-1]
        try:
            return int(suffix)
        except ValueError:
            return -1

    for root in search_roots:
        if not root.exists():
            continue
        for pattern, binary in (("chromium_headless_shell-*", "headless_shell"), ("chromium-*", "chrome")):
            dirs = sorted(root.glob(pattern), key=_build_key, reverse=True)
            for folder in dirs:
                candidate = folder / "chrome-linux" / binary
                if candidate.exists():
                    candidates.append(candidate)
    # Deduplicate while preserving order
    seen = set()
    unique_candidates = []
    for cand in candidates:
        if cand in seen:
            continue
        unique_candidates.append(cand)
        seen.add(cand)
    return unique_candidates[0] if unique_candidates else None


def render_robot_with_external_camera(
    viser_helper: ViserHelper,
    robot: RobotMjcfViser,
    camera_handler: ExternalCameraHandler,
    robot_pos: np.ndarray,
    robot_rot: np.ndarray,
    output_dir: str,
    render_frames: list[int],
    save_video: bool,
    save_images: bool,
    dt: float,
):
    os.makedirs(output_dir, exist_ok=True)
    try:
        client, launched = _get_client_or_launch(viser_helper.server)
    except TimeoutError as exc:
        print(f"[Renderer] {exc}")
        return []

    rendered_images = []
    print(f"[Renderer] Total robot frames: {len(robot_pos)}")
    print(f"[Renderer] Total camera frames: {len(camera_handler.images)}")
    print(f"[Renderer] Frames to render: {len(render_frames)}")

    for idx, t in enumerate(render_frames):
        robot.update(robot_pos[t], robot_rot[t])
        cam_data = camera_handler.get_camera_at_frame(t)
        pos, quat = _twc_to_cam_pose(cam_data["T_c2w"])
        fov = _k_to_fov_y(cam_data["K"], cam_data["H"])
        with client.atomic():
            client.camera.position = pos
            client.camera.wxyz = quat
            client.camera.fov = float(fov)
        time.sleep(dt)
        rgb = client.get_render(height=int(cam_data["H"]), width=int(cam_data["W"]))
        time.sleep(0.2)

        if save_images:
            img_path = os.path.join(output_dir, f"frame_{t:05d}.png")
            cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if idx % 10 == 0:
                print(f"[Renderer] Saved {img_path}")
        rendered_images.append(rgb)

    if save_video and rendered_images:
        video_path = os.path.join(output_dir, "robot_render.mp4")
        H, W = rendered_images[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, float(camera_handler.fps), (W, H))
        if writer.isOpened():
            for rgb in rendered_images:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if bgr.shape[0] != H or bgr.shape[1] != W:
                    bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
                writer.write(bgr)
            writer.release()
            print(f"[Renderer] Saved video: {video_path}")
        else:
            print("[Renderer] Failed to open video writer; skipping video export.")

    _close_launched(launched)
    return rendered_images


    
def main():
    ap = argparse.ArgumentParser(description="Visualize robot with external camera data.")
    ap.add_argument(
        "record_dir",
        type=str,
        nargs="?",
        help="Directory containing robot recordings",
    )
    ap.add_argument(
        "--data",
        type=str,
        help="Path to NPZ file containing camera data (intrinsic, cam_c2w, scale)",
    )
    ap.add_argument("--env_idx", type=int, default=0, help="Environment index")
    ap.add_argument("--port", type=int, default=8080, help="Viser server port")
    ap.add_argument("--output_dir", type=str, default="./rendered_output", help="Output directory")
    ap.add_argument("--save_images", action="store_true", help="Save individual frame images")
    ap.add_argument("--save_video", action="store_true", default=True, help="Save video")
    ap.add_argument("--max_frames", type=int, default=None, help="Maximum frames to render")
    ap.add_argument("--frame_stride", type=int, default=1, help="Frame stride for rendering")
    ap.add_argument("--scene_obj",type=str, help="Optional path to a scene OBJ file to visualize")
    ap.add_argument('--scene', required=True, help='Sequence name (e.g., 000)')
    ap.add_argument('--types', required=True, help='Sequence name (e.g., 000)')
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    ap.add_argument('--parent_name', default='emdb_new', help='Parent folder name')
    ap.add_argument('--method', default='emdb_new', help='Parent folder name')
    ap.add_argument('--data_path', default='/data3/zihanwa3/_Robotics/humanoid/hybrid-imitation-parkour/motion_data/', help='Root path to motion data')

    args = ap.parse_args()

    args.output_dir = os.path.join(args.output_dir, f'{args.parent_name}_{args.scene}')
    os.makedirs(args.output_dir, exist_ok=True)

    types = args.types
    args.data = f'/data3/zihanwa3/_Robotics/_vision/mega-sam/postprocess/{args.scene}_gv_sgd_cvd_hr.npz'
    
    parent_name = args.parent_name
    data_path = args.data_path
    seq_name = args.scene
    data = args.data



    if "_" in args.method:
        types = args.types = args.types.split('_')[0]
        method = args.method = args.method.split('_')[0]

        args.scene_obj = f"/data3/zihanwa3/_Robotics/humanoid/hybrid-imitation-parkour/parkour_anim/data/assets/urdf/{args.parent_name}/{seq_name}/{args.method}"
        args.record_dir = os.path.join(f'/data3/zihanwa3/_Robotics/humanoid/hybrid-imitation-parkour/post_asset_trimesh_{args.parent_name}', f'{args.scene}_{args.method}')
    else:
        args.scene_obj = f"/data3/zihanwa3/_Robotics/humanoid/hybrid-imitation-parkour/parkour_anim/data/assets/urdf/{args.parent_name}/{seq_name}/{args.method}"
        args.record_dir = os.path.join(f'/data3/zihanwa3/_Robotics/humanoid/hybrid-imitation-parkour/post_asset_{args.method}_{args.parent_name}', args.scene)

    print(args.record_dir)
    rec = Path(args.record_dir)
    raw_data_path = f"{data_path}/{parent_name}/{seq_name}_{types}.npz"
    raw_data = np.load(raw_data_path)
    T_align = raw_data['T_align'].astype(np.float32)

    moge_base_path = '/data3/zihanwa3/_Robotics/_vision/TAPIP3D/_raw_mega_priors'
    tgt_name = str(data).split('_sgd')[0].split('/')[-1]   
    tgt_name = "_".join(tgt_name.split("_")[:-1]) 
    moge_data = os.path.join(moge_base_path, f'{tgt_name}.npz')
    tgt_folder = os.path.join('/data3/zihanwa3/_Robotics/_geo/differentiable-blocksworld/post_results', tgt_name)

    camera_handler = None
    try:
        camera_dict = _load_npz_to_dict(args.data)
        moge_dict = _load_npz_to_dict(moge_data)
        for key in ["depths", "images", "cam_c2w", "intrinsic"]:
            if key in moge_dict:
                camera_dict[key] = moge_dict[key]
        if "scale" not in camera_dict and "scale" in moge_dict:
            camera_dict["scale"] = moge_dict["scale"]
        if "fps" not in camera_dict and "fps" in moge_dict:
            camera_dict["fps"] = moge_dict["fps"]
        camera_handler = ExternalCameraHandler(
            npz_data=dict(camera_dict),
            npz_cam_data=dict(camera_dict),
            T_align=T_align,
        )
        print(f"[Camera] Prepared dataset-driven renderer using {args.data}")
    except FileNotFoundError as exc:
        print(f"[Camera] Missing camera file: {exc}")
    except KeyError as exc:
        print(f"[Camera] Dataset missing key {exc}; render will be skipped.")


    # Load robot info
    info_path = rec / "robot_vis_info.yaml"
    if info_path.exists():
        info = yaml.safe_load(info_path.read_text())
        body_names = info.get("body_names", [])
        dt = 1.0 / 30.0 # float(info.get("dt", 1.0 / 30.0)) / max(args.speed, 1e-6)
        asset_xml_rel = info.get("asset_xml", None)
    else:
        print("[Visualizer] robot_vis_info.yaml not found; proceeding with defaults.")
        body_names = []
        dt = 1.0 / 60.0 / max(args.speed, 1e-6)
        asset_xml_rel = None

    # Resolve MJCF path
    if asset_xml_rel is not None and (rec / asset_xml_rel).exists():
        mjcf_path = str(rec / asset_xml_rel)
    else:
        # Best-effort search in dir
        cand = list(rec.glob("*.xml"))
        mjcf_path = str(cand[0]) if cand else None
        if mjcf_path is None:
            raise FileNotFoundError("No MJCF xml found in record dir.")

    # Load rigid bodies trajectory
    npz_path = rec / f"rigid_bodies_{args.env_idx}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Rigid body file not found: {npz_path}")
    data = np.load(npz_path)
    pos = data["pos"]  # (T, B, 3)
    rot = data["rot"]  # (T, B, 4) xyzw
    T, B, _ = pos.shape
    import random
    def generate_four_digit():
        return random.randint(1000, 9999)
    port=generate_four_digit()
    viser = ViserHelper(port=port)
    server = viser.server




    if not viser.ok():
        print("[Visualizer] Viser not available; exiting.")
        return

    robot = RobotMjcfViser(viser, mjcf_path, body_names if body_names else None)

    # Optionally load and visualize an external scene OBJ
    import trimesh
    def _finish(tri: trimesh.Trimesh):

        v = np.asarray(tri.vertices, dtype=np.float32)
        f = np.asarray(tri.faces, dtype=np.int32)
        return v, f

    def combine_meshes(vf_list):
        """
        vf_list: [(verts0, faces0), (verts1, faces1), ...]
        返回: (V_all, F_all)  —— 已合并
        """
        verts_all = []
        faces_all = []
        v_offset = 0
        for verts, faces in vf_list:
            v = np.asarray(verts, dtype=np.float32)
            f = np.asarray(faces, dtype=np.int32)
            verts_all.append(v)
            faces_all.append(f + v_offset)
            v_offset += v.shape[0]
        V = np.concatenate(verts_all, axis=0)
        F = np.concatenate(faces_all, axis=0)
        return V, F
        
    if args.scene_obj:
      verts, faces = load_scene_mesh(args.scene_obj)  # 
      # verts, faces = _finish(trimesh.load('/data3/zihanwa3/_Robotics/humanoid/hybrid-imitation-parkour/gggg.obj'))
      #V, F = combine_meshes([(verts_1, faces_1), (verts, faces)])

      #verts, faces = V, F
        
      #load_scene_mesh(args.scene_obj)
      if verts is not None and faces is not None:
          viser.add_mesh_simple("/scene", verts, faces, color=(0.6, 0.7, 0.9))
          print(f"[Main] Loaded scene with {len(verts)} vertices, {len(faces)} faces")

    if camera_handler is not None:
        T_render = min(len(pos), len(camera_handler.images))
        if args.max_frames:
            T_render = min(T_render, args.max_frames)
        render_frames = list(range(0, T_render, args.frame_stride or 1))
        if render_frames:
            render_robot_with_external_camera(
                viser_helper=viser,
                robot=robot,
                camera_handler=camera_handler,
                robot_pos=pos,
                robot_rot=rot,
                output_dir=args.output_dir,
                render_frames=render_frames,
                save_video=args.save_video,
                save_images=args.save_images,
                dt=dt,
            )
        else:
            print("[Renderer] No frames selected for rendering.")
    else:
        print("[Renderer] Camera handler unavailable; skipping dataset render.")

    serializer = server.get_scene_serializer()
 

    timesteps = len(pos)
    print(timesteps)
    for t in range(timesteps):
        # [some updates here]
        robot.update(pos[t], rot[t])
        serializer.insert_sleep(1 / 30)
    data = serializer.serialize()
    
    Path("recording.viser").write_bytes(data)

    # Simple camera setup
    root0 = pos[0, 0]
    cam = root0 + np.array([0.0, -2.0, 1.5], dtype=np.float32)
    look = root0 + np.array([0.0, 0.0, 0.4], dtype=np.float32)
    viser.set_camera(cam, look)

    # Playback loop
    print("[Visualizer] Starting looped playback. Press Ctrl-C to exit.")
    try:
        while True:
            for t in range(T):
                robot.update(pos[t], rot[t])
                time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[Visualizer] Stopped by user.")


if __name__ == "__main__":
    main()
