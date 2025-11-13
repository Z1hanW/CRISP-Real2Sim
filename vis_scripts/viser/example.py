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

from parkour_anim.utils.viser_visualizer_z import ViserHelper
from parkour_anim.utils.robot_viser_z import RobotMjcfViser
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
    ap.add_argument('--data_path', default='motion_data/', help='Root path to motion data')

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

        args.scene_obj = f"parkour_anim/data/assets/urdf/{args.parent_name}/{seq_name}/{args.method}"
        args.record_dir = os.path.join(f'post_asset_trimesh_{args.parent_name}', f'{args.scene}_{args.method}')
    else:
        args.scene_obj = f"parkour_anim/data/assets/urdf/{args.parent_name}/{seq_name}/{args.method}"
        args.record_dir = os.path.join(f'post_asset_{args.method}_{args.parent_name}', args.scene)

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