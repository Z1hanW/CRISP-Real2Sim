import argparse
import shutil
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import smplx
import torch
import trimesh
from pytorch3d import transforms
from scipy.spatial.transform import Rotation as sRot
from sklearn.neighbors import KDTree

warnings.filterwarnings("ignore")
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = (THIS_DIR / ".." / "..").resolve()
DEFAULT_INPUT_ROOT = REPO_ROOT / "results/output/scene"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results/output/post_scene"
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
for p in (THIS_DIR, THIS_DIR.parent, REPO_ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from smpl import SMPL  # noqa: E402


class SurfacePointCloud:
    def __init__(self, mesh, points, normals=None):
        self.mesh = mesh
        self.points = points
        self.normals = normals
        self.kd_tree = KDTree(points)

    def get_sdf(self, query_points, sample_count=11, return_gradients=False):
        distances, indices = self.kd_tree.query(query_points, k=sample_count)
        distances = distances.astype(np.float32)

        closest_points = self.points[indices]
        direction_from_surface = query_points[:, np.newaxis, :] - closest_points
        inside = np.einsum("ijk,ijk->ij", direction_from_surface, self.normals[indices]) < 0
        inside = np.sum(inside, axis=1) > sample_count * 0.5
        distances = distances[:, 0]
        distances[inside] *= -1

        if return_gradients:
            gradients = direction_from_surface[:, 0]
            gradients[inside] *= -1

            near_surface = np.abs(distances) < 0.0075
            gradients = np.where(near_surface[:, np.newaxis], self.normals[indices[:, 0]], gradients)
            gradients /= np.linalg.norm(gradients, axis=1, keepdims=True)
            return distances, gradients
        else:
            return distances


def sample_from_mesh(mesh, sample_point_count=10000000):
    points, face_indices = mesh.sample(sample_point_count, return_index=True)
    normals = mesh.face_normals[face_indices]
    return SurfacePointCloud(mesh, points=points, normals=normals)


def _find_first_image(scene_name: str, data_root: Path) -> Optional[np.ndarray]:
    """
    Find and load the first image for a scene under a *_img split.
    """
    if not data_root.exists():
        return None
    for split_dir in data_root.iterdir():
        if not split_dir.is_dir() or not split_dir.name.endswith("_img"):
            continue
        candidate_dir = split_dir / scene_name
        if not candidate_dir.is_dir():
            continue
        for ext in ("*.jpg", "*.png", "*.jpeg"):
            images = sorted(candidate_dir.glob(ext))
            if not images:
                continue
            img_path = images[0]
            try:
                from PIL import Image

                return np.array(Image.open(img_path).convert("RGB"))
            except Exception:
                try:
                    import imageio.v2 as imageio

                    return imageio.imread(img_path)
                except Exception:
                    return None
    return None


def _load_frame_for_rotation(
    scene_name: str,
    hmr_type: str,
    repo_root: Path,
    camera_npz: Optional[Path],
    data_root: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one RGB frame and matching cam2world for calibration.
    """
    # Prefer the MegaSAM export camera to stay consistent with visualizer_megasam.py.
    scene_npz = repo_root / "results/output/scene" / f"{scene_name}_{hmr_type}_sgd_cvd_hr.npz"

    candidates: List[Path] = []
    if camera_npz is not None and camera_npz != scene_npz:
        candidates.append(camera_npz)
    candidates.append(scene_npz)

    for cand in candidates:
        if not cand.exists():
            continue
        if cand.suffix == ".npz":
            data = np.load(cand, allow_pickle=True)
            images = data.get("images")
            cams = data.get("cam_c2w")
            if images is None or cams is None:
                continue
            idx = 0
            if "valid_frame_indices" in data and len(data["valid_frame_indices"]) > 0:
                idx = int(np.array(data["valid_frame_indices"]).ravel()[0])
            return images[idx], cams[idx]
        if cand.suffix == ".npy":
            camera_payload = np.load(cand, allow_pickle=True).item()
            cams = camera_payload.get("cam_c2w")
            if cams is None:
                continue
            image = _find_first_image(scene_name, data_root)
            if image is None:
                raise FileNotFoundError(f"Could not find an image for {scene_name} under {data_root}.")
            return image, cams[0]

    searched = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not locate camera/image data for {scene_name}. "
        f"Expected MegaSAM output at {scene_npz} (or pass --camera-npz). Tried:\n{searched}"
    )


def get_calibration_roll_pitch(image: np.ndarray, device: str) -> Tuple[float, float]:
    """
    Get roll and pitch calibration from an image using GeoCalib.
    """
    from geocalib.utils import print_calibration
    from geocalib import GeoCalib

    model = GeoCalib().to(device)
    input_image = torch.tensor(image, dtype=torch.float32).to(device).permute(2, 0, 1)
    result = model.calibrate(input_image)

    camera, gravity = result["camera"], result["gravity"]  # noqa: F841
    roll_rad, pitch_rad = gravity.rp.unbind(-1)
    roll_rad = float(roll_rad.item())
    pitch_rad = float(pitch_rad.item())
    print_calibration(result)
    return roll_rad, pitch_rad


def get_world_rotation(world_env, device: str, is_megasam: bool = True) -> np.ndarray:
    """
    Compute rotation matrix that aligns gravity to +Z.
    """
    first_frame = next(iter(world_env.values()))["rgbimg"]
    if is_megasam:
        first_frame = first_frame.astype(np.float32) / 255.0

    roll, pitch = get_calibration_roll_pitch(first_frame, device)

    pitch_rotm = np.array(
        [
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=np.float32,
    )

    roll_rotm = np.array(
        [
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    cam0_rotm = next(iter(world_env.values()))["cam2world"][:3, :3].astype(np.float32)
    world_rotation = pitch_rotm @ roll_rotm @ cam0_rotm
    yup_to_zup = np.array(
        [
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )
    return yup_to_zup @ world_rotation


def process_smpl_data(smpl_data_path: Path, smpl_model: SMPL):
    data = np.load(smpl_data_path, allow_pickle=True).item()
    poses = data["body_pose"].cpu()
    trans = data["transl"].cpu()
    global_orient = data["global_orient"].cpu()
    betas = data["betas"].cpu()
    cams = data.get("pred_cam")
    return poses, trans, global_orient, betas, cams


def plot_translation_and_velocity(transl: np.ndarray, dt: float = 1 / 30, save_path: Optional[Path] = None):
    T = transl.shape[0]
    time = np.arange(T) * dt
    velocity = np.gradient(transl, dt, axis=0)

    labels = ["x", "y", "z"]
    colors = ["r", "g", "b"]

    fig, axs = plt.subplots(3, 2, figsize=(12, 9))

    for i in range(3):
        axs[i, 0].plot(time, transl[:, i], color=colors[i])
        axs[i, 0].set_ylabel(f"{labels[i]} (m)")
        axs[i, 0].set_title(f"Translation {labels[i]}")
        axs[i, 0].grid(True)

        axs[i, 1].plot(time, velocity[:, i], color=colors[i], linestyle="--")
        axs[i, 1].set_ylabel(f"v{labels[i]} (m/s)")
        axs[i, 1].set_title(f"Velocity v{labels[i]}")
        axs[i, 1].grid(True)

    for ax in axs[-1, :]:
        ax.set_xlabel("Time (s)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=500)
        plt.close(fig)
    else:
        plt.show()


def compute_world_alignment(
    scene_name: str,
    hmr_type: str,
    repo_root: Path,
    camera_npz: Optional[Path],
    data_root: Path,
    is_megasam: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    rgb_img, cam_c2w = _load_frame_for_rotation(scene_name, hmr_type, repo_root, camera_npz, data_root)
    world_env = {"frame0": {"rgbimg": rgb_img, "cam2world": cam_c2w.astype(np.float32)}}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    world_rotation = get_world_rotation(world_env, device, is_megasam)

    T_align = np.eye(4, dtype=np.float32)
    T_align[:3, :3] = world_rotation
    return T_align, world_rotation


def _scene_mesh_candidates(input_root: Path) -> List[Path]:
    """
    Possible scene meshes to use when estimating a shared translation.
    """
    candidates: List[Path] = []
    sqs_mesh = input_root / "scene_mesh_sqs" / "scene_mesh_sqs.obj"
    scene_mesh = input_root / "scene" / "scene_mesh.obj"
    if sqs_mesh.exists():
        candidates.append(sqs_mesh)
    if scene_mesh.exists():
        candidates.append(scene_mesh)
    return candidates


def compute_scene_translation_to_ground(input_root: Path, world_rotation: np.ndarray) -> np.ndarray:
    """
    Compute a translation that moves the rotated scene's lowest vertex to z=0.

    The translation is shared between scene and human before any optional human-only optimization.
    """
    min_z: Optional[float] = None
    rot_t = world_rotation.T
    for mesh_path in _scene_mesh_candidates(input_root):
        try:
            mesh = trimesh.load(mesh_path, process=False)
        except Exception as exc:  # pragma: no cover - best-effort logging
            print(f"[WARN] Failed to load {mesh_path}: {exc}")
            continue
        if mesh.vertices.size == 0:
            continue
        rotated = mesh.vertices @ rot_t  # row-vector convention
        mesh_min = float(np.min(rotated[:, 2]))
        min_z = mesh_min if min_z is None else min(min_z, mesh_min)

    if min_z is None:
        print("[WARN] No scene mesh found; using zero translation.")
        return np.zeros(3, dtype=np.float32)

    translation = np.array([0.0, 0.0, -min_z], dtype=np.float32)
    print(f"[OK] Shared scene/human translation (z-shift): {translation}")
    return translation


def rotate_scene_geometry(input_root: Path, output_root: Path, T_align: np.ndarray) -> Optional[Path]:
    output_root.mkdir(parents=True, exist_ok=True)

    scene_dst = output_root / "scene"
    scene_dst.mkdir(parents=True, exist_ok=True)
    scene_src = input_root / "scene" / "scene_mesh.obj"
    if scene_src.exists():
        mesh = o3d.io.read_triangle_mesh(scene_src.as_posix())
        mesh.compute_vertex_normals()
        mesh.rotate(T_align[:3, :3], center=(0, 0, 0))
        mesh.translate(T_align[:3, 3])
        o3d.io.write_triangle_mesh((scene_dst / "scene_mesh.obj").as_posix(), mesh)

    sqs_dst = output_root / "scene_mesh_sqs"
    sqs_dst.mkdir(parents=True, exist_ok=True)
    sqs_src = input_root / "scene_mesh_sqs"

    mesh_out_path = None
    sqs_mesh_src = sqs_src / "scene_mesh_sqs.obj"
    if sqs_mesh_src.exists():
        mesh = o3d.io.read_triangle_mesh(sqs_mesh_src.as_posix())
        mesh.compute_vertex_normals()
        mesh.rotate(T_align[:3, :3], center=(0, 0, 0))
        mesh.translate(T_align[:3, 3])
        mesh_out_path = sqs_dst / "scene_mesh_sqs.obj"
        o3d.io.write_triangle_mesh(mesh_out_path.as_posix(), mesh)

    urdf_src = sqs_src / "scene_mesh_sqs.urdf"
    if urdf_src.exists():
        shutil.copy2(urdf_src, sqs_dst / urdf_src.name)

    pieces_src = sqs_src / "pieces"
    pieces_dst = sqs_dst / "pieces"
    if pieces_src.exists():
        pieces_dst.mkdir(parents=True, exist_ok=True)
        for mesh_path in sorted(pieces_src.glob("*.obj")):
            mesh = o3d.io.read_triangle_mesh(mesh_path.as_posix())
            mesh.compute_vertex_normals()
            mesh.rotate(T_align[:3, :3], center=(0, 0, 0))
            mesh.translate(T_align[:3, 3])
            o3d.io.write_triangle_mesh((pieces_dst / mesh_path.name).as_posix(), mesh)
    return mesh_out_path


def rotate_and_align_smpl(
    smpl_model: SMPL,
    poses,
    trans,
    global_orient,
    betas,
    T_align: np.ndarray,
    save_path: Path,
    shift_to_ground: bool = False,
    debug_stride: int = 10,
):
    """
    Legacy helper (kept for compatibility). Not used by the main pipeline below.

    NOTE: The main pipeline now rotates BOTH global_orient and transl (HMR) and applies a shared [1,3] translation.
    """
    save_path.mkdir(parents=True, exist_ok=True)

    pred = smpl_model(
        body_pose=poses,
        global_orient=global_orient,
        betas=betas,
        transl=trans,
        pose2rot=False,
        default_smpl=True,
    )
    original_verts = pred.vertices.cpu().numpy()
    for i, verts in enumerate(original_verts):
        if debug_stride and i % debug_stride == 0:
            trimesh.Trimesh(vertices=verts, faces=smpl_model.faces).export(save_path / f"human_beforerot_{i:04d}.obj")

    aa_pose = np.concatenate(
        [
            transforms.matrix_to_axis_angle(global_orient.squeeze(1)),
            transforms.matrix_to_axis_angle(poses).reshape(len(poses), -1),
        ],
        axis=-1,
    )

    R = sRot.from_matrix(T_align[:3, :3])
    aa_pose[:, :3] = (R * sRot.from_rotvec(aa_pose[:, :3])).as_rotvec()
    trans_np = trans.numpy() @ R.as_matrix().T

    model = smplx.create(
        (REPO_ROOT / "prep/data/smpl/SMPL_NEUTRAL.pkl").as_posix(),
        model_type="smpl",
        gender="neutral",
        num_betas=10,
        batch_size=len(poses),
        ext="pkl",
    )

    transl = torch.from_numpy(trans_np)
    poses_torch = torch.from_numpy(aa_pose)

    temp_output = model(
        betas=betas,
        transl=transl,
        global_orient=poses_torch[:, :3],
        body_pose=poses_torch[:, 3:],
        return_verts=True,
    )
    offset = original_verts @ R.as_matrix().T - temp_output.vertices.detach().cpu().numpy()

    transl += torch.from_numpy(np.mean(offset, axis=1) + T_align[:3, 3])
    temp_output = model(
        betas=betas,
        transl=transl,
        global_orient=poses_torch[:, :3],
        body_pose=poses_torch[:, 3:],
        return_verts=True,
    )

    if shift_to_ground:
        joints_z = temp_output.joints.detach().cpu()[..., 2]
        flattened = joints_z.flatten()
        lowest_values = torch.topk(flattened, int(0.010 * flattened.numel()), largest=False).values
        z_mins = lowest_values.mean()
        if torch.abs(z_mins) < 0.3:
            transl[:, 2] -= z_mins
            T_align[2, 3] -= z_mins

        temp_output = model(
            betas=betas,
            transl=transl,
            global_orient=poses_torch[:, :3],
            body_pose=poses_torch[:, 3:],
            return_verts=True,
        )

    verts = temp_output.vertices.detach().cpu().numpy()

    for i, verts_i in enumerate(verts):
        if debug_stride and i % debug_stride == 0:
            trimesh.Trimesh(vertices=verts_i, faces=model.faces).export(save_path / f"human_afterrot_{i:04d}.obj")

    plot_translation_and_velocity(transl.numpy(), save_path=save_path / "transl_plot.png")
    return poses_torch.numpy(), transl.numpy(), betas.numpy(), T_align


def optimize_translation_along_z(
    model: SMPL,
    body_pose: torch.Tensor,
    global_orient: torch.Tensor,
    betas: torch.Tensor,
    transl: torch.Tensor,
    save_path: Path,
    debug_stride: int = 10,
) -> torch.Tensor:
    """
    Human-only (per-frame) z-grounding:
    shift each frame along z so the lowest vertex sits at z=0.

    NOTE: This breaks the "single shared [1,3] translation" constraint; keep disabled unless needed.
    """
    device = transl.device
    save_path.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        smpl_out = model(
            betas=betas.to(device),
            transl=transl,
            global_orient=global_orient.to(device),
            body_pose=body_pose.to(device),
            pose2rot=False,
            default_smpl=True,
        )
    verts = smpl_out.vertices  # (T, 6890, 3)

    for i in range(0, len(verts), debug_stride):
        trimesh.Trimesh(vertices=verts[i].cpu().numpy(), faces=model.faces).export(save_path / f"human_beforeopt_{i:04d}.obj")

    min_z = verts[..., 2].amin(dim=1)  # (T,)
    transl_opt = transl.clone()
    transl_opt[:, 2] -= min_z  # bring per-frame lowest vertex to z=0

    with torch.no_grad():
        smpl_out_after = model(
            betas=betas.to(device),
            transl=transl_opt,
            global_orient=global_orient.to(device),
            body_pose=body_pose.to(device),
            pose2rot=False,
            default_smpl=True,
        )
    verts_after = smpl_out_after.vertices
    for i in range(0, len(verts_after), debug_stride):
        trimesh.Trimesh(vertices=verts_after[i].cpu().numpy(), faces=model.faces).export(save_path / f"human_afteropt_{i:04d}.obj")

    plot_translation_and_velocity(transl_opt.cpu().numpy(), save_path=save_path / "transl_plot.png")
    return transl_opt


def optimize_translation_to_avoid_penetration(
    scene_mesh_path: Path,
    model,
    poses,
    betas,
    transl_orig,
    save_path: Path,
    sample_n: int = 256,
    num_iters: int = 20,
    margin: float = 0.01,
):
    if not scene_mesh_path.exists():
        print(f"[WARN] Scene mesh not found at {scene_mesh_path}, skipping penetration optimization.")
        return np.zeros_like(transl_orig.cpu().numpy())

    scene_mesh = trimesh.load(scene_mesh_path)
    surface_pc = sample_from_mesh(scene_mesh, sample_point_count=10000000)

    class DifferentiableSDF(torch.autograd.Function):
        @staticmethod
        def forward(ctx, surface_pc, points_tensor):
            points_np = points_tensor.detach().cpu().numpy()
            distances, gradients = surface_pc.get_sdf(points_np, return_gradients=True)
            ctx.save_for_backward(torch.from_numpy(gradients).to(points_tensor.device))
            return torch.from_numpy(distances).to(points_tensor.device)

        @staticmethod
        def backward(ctx, grad_output):
            (gradients,) = ctx.saved_tensors
            return None, grad_output.unsqueeze(-1) * gradients

    def fps(x, npoint):
        device = x.device
        N, _ = x.shape
        centroids = torch.zeros(npoint, dtype=torch.long, device=device)
        distance = torch.ones(N, device=device) * 1e10
        farthest = torch.randint(0, N, (1,), device=device)
        for i in range(npoint):
            centroids[i] = farthest
            centroid = x[farthest, :].view(1, 3)
            dist = torch.sum((x - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids

    device = transl_orig.device
    T = poses.shape[0]
    transl_delta_init = torch.zeros((T, 3), device=device)
    transl_delta_init[:, 2] = 0.2
    transl_delta = transl_delta_init.clone().detach().requires_grad_()

    optimizer = torch.optim.Adam([transl_delta], lr=1e-2)
    rand_idx = None

    for it in range(num_iters):
        optimizer.zero_grad()
        smpl_out = model(
            betas=betas,
            transl=transl_orig + transl_delta,
            global_orient=poses[:, :3],
            body_pose=poses[:, 3:],
            return_verts=True,
        )

        verts = smpl_out.vertices
        if rand_idx is None:
            rand_idx = fps(verts[0], sample_n)
        verts_sub = verts[:, rand_idx, :]
        verts_flat = verts_sub.reshape(-1, 3)

        sdf_vals = DifferentiableSDF.apply(surface_pc, verts_flat)
        penetration_loss = torch.clamp(margin - sdf_vals, min=0).mean()
        reg_loss = (transl_delta ** 2).mean()
        smooth_loss = (torch.diff(transl_orig + transl_delta, dim=0) ** 2).mean()

        total_loss = 1.0 * penetration_loss + 0.1 * reg_loss + 1 * smooth_loss
        total_loss.backward()
        optimizer.step()

    with torch.no_grad():
        out_opt = model(
            betas=betas,
            transl=transl_orig + transl_delta.detach(),
            global_orient=poses[:, :3],
            body_pose=poses[:, 3:],
            return_verts=True,
        )
    verts_opt = out_opt.vertices.cpu().numpy()

    for i in range(0, len(verts_opt), 10):
        mesh = trimesh.Trimesh(vertices=verts_opt[i], faces=model.faces)
        mesh.export(save_path / f"human_afteropt_rot_{i:04d}.obj")

    plot_translation_and_velocity(
        (transl_orig + transl_delta.detach()).cpu().numpy(), save_path=save_path / "transl_delta_plot.png"
    )
    return transl_delta.detach().cpu().numpy()


def export_to_targets(
    seq_name: str,
    output_root: Path,
    export_motion_root: Optional[Path],
    export_urdf_root: Optional[Path],
    export_tag: Optional[str],
) -> None:
    motion_src = output_root / f"{seq_name}_ours.npz"
    if export_motion_root and motion_src.exists():
        motion_dst_root = export_motion_root / export_tag if export_tag else export_motion_root
        motion_dst_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(motion_src, motion_dst_root / motion_src.name)

    if export_urdf_root:
        src_sqs = output_root / "scene_mesh_sqs"
        urdf_src = src_sqs / "scene_mesh_sqs.urdf"
        mesh_src = src_sqs / "scene_mesh_sqs.obj"
        pieces_src = src_sqs / "pieces"

        target_dir = export_urdf_root / (export_tag if export_tag else "") / seq_name / "ours"
        target_dir.mkdir(parents=True, exist_ok=True)
        if urdf_src.exists():
            shutil.copy2(urdf_src, target_dir / "ours.urdf")
        if mesh_src.exists():
            shutil.copy2(mesh_src, target_dir / "mesh.obj")
        if pieces_src.exists():
            for item in pieces_src.iterdir():
                dst_item = target_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dst_item, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dst_item)


def discover_sequences(input_root: Path, hmr_type: str) -> List[str]:
    seqs: List[str] = []
    for candidate in sorted(input_root.iterdir()):
        if candidate.is_dir() and (candidate / hmr_type).is_dir():
            seqs.append(candidate.name)
    return seqs


def process_sequence(
    seq_name: str,
    input_root: Path,
    output_root: Path,
    hmr_type: str,
    camera_npz: Optional[Path],
    data_root: Path,
    is_megasam: bool,
    optimize_human: bool,
    export_motion_root: Optional[Path],
    export_urdf_root: Optional[Path],
    export_tag: Optional[str],
    debug_stride: int,
) -> None:
    # Use repository root so relative paths (e.g., results/output) resolve correctly
    repo_root = REPO_ROOT
    seq_input_root = input_root / seq_name / hmr_type
    if not seq_input_root.exists():
        print(f"[WARN] Input path missing for {seq_name}: {seq_input_root}")
        return

    seq_output_root = output_root / seq_name / hmr_type
    seq_output_root.mkdir(parents=True, exist_ok=True)
    save_path = seq_output_root / "saved_obj"

    if camera_npz is not None:
        print("[WARN] Ignoring --camera-npz; using default *_sgd_cvd_hr camera export.")

    # 1) Compute gravity-aligned rotation
    T_align, world_rotation = compute_world_alignment(
        seq_name,
        hmr_type,
        repo_root,
        None,
        data_root,
        is_megasam,
    )

    # 2) Compute a SINGLE shared [1,3] z-translation so rotated scene min z becomes 0
    shared_translation = compute_scene_translation_to_ground(seq_input_root, world_rotation)  # (3,)
    shared_translation_1x3 = shared_translation[None, :]  # (1,3) requested

    # Apply to the scene transform (open3d wants (3,))
    T_align[:3, 3] = shared_translation

    np.save(seq_output_root / "world_rotation.npy", world_rotation.astype(np.float32))
    np.savetxt(seq_output_root / "world_rotation.txt", world_rotation, fmt="%.8f")

    # 3) Rotate + translate the scene geometry
    scene_mesh_path = rotate_scene_geometry(seq_input_root, seq_output_root, T_align)

    # 4) Load HMR/SMPL results
    smpl_data_path = seq_input_root / "hmr" / "hps_track.npy"
    smpl_model = SMPL().to("cpu")
    poses, trans, global_orient, betas, _ = process_smpl_data(smpl_data_path, smpl_model)

    world_rot_torch = torch.from_numpy(world_rotation).to(trans.device, dtype=trans.dtype)  # (3,3)
    translation_offset = torch.from_numpy(shared_translation_1x3).to(trans.device, dtype=trans.dtype)  # (1,3)

    # 5) Rotate HMR (both orientation and translation) + apply the shared translation
    global_orient_rot = torch.einsum("ij,t...jk->t...ik", world_rot_torch, global_orient)
    trans_rot = trans @ world_rot_torch.T  # (T,3), row-vector convention
    trans_shared = trans_rot + translation_offset  # broadcast (1,3) -> (T,3)

    trans_optimized = trans_shared
    extra_trans = torch.zeros_like(trans_shared)

    # Optional: human-only per-frame z grounding (NOT shared)
    if optimize_human:
        smpl_model = smpl_model.to(trans.device)
        trans_optimized = optimize_translation_along_z(
            smpl_model,
            poses,
            global_orient_rot,
            betas,
            trans_shared.to(trans.device),
            save_path,
            debug_stride=debug_stride,
        )
        extra_trans = trans_optimized - trans_shared.to(trans_optimized.device)

    poses_axis_angle = torch.cat(
        [
            transforms.matrix_to_axis_angle(global_orient_rot.squeeze(1)),
            transforms.matrix_to_axis_angle(poses).reshape(len(poses), -1),
        ],
        dim=-1,
    ).cpu().numpy()
    betas_np = betas.cpu().numpy()
    trans_np = trans_optimized.cpu().numpy()

    np.savez(
        seq_output_root / f"{seq_name}_ours.npz",
        trans=trans_np,
        poses=poses_axis_angle,
        betas=betas_np,
        T_align=T_align,
        T_align_new=T_align,
        gender="neutral",
        mocap_framerate=30,
        world_rotation=world_rotation,
        shared_translation=shared_translation_1x3.astype(np.float32),  # (1,3) explicit
    )

    # 6) Save updated hps_track in the same layout as the raw scene output.
    hmr_payload = np.load(smpl_data_path, allow_pickle=True).item()
    if torch.is_tensor(hmr_payload.get("global_orient")):
        hmr_payload["global_orient"] = global_orient_rot.to(hmr_payload["global_orient"].device)
    hmr_payload["transl"] = trans_optimized.to(hmr_payload["transl"].device)

    # Update pred_cam if present as [R, t] (best-effort)
    if isinstance(hmr_payload.get("pred_cam"), list):
        pred_cam = hmr_payload["pred_cam"]
        if len(pred_cam) >= 1 and torch.is_tensor(pred_cam[0]):
            # rotation part
            pred_cam[0] = torch.einsum("ij,tjk->tik", world_rot_torch.to(pred_cam[0].device), pred_cam[0])

        if len(pred_cam) >= 2 and torch.is_tensor(pred_cam[1]):
            # translation part (best-effort): rotate + add shared shift + add extra human-only shift if enabled
            R_dev = world_rot_torch.to(pred_cam[1].device, dtype=pred_cam[1].dtype)
            pred_cam[1] = pred_cam[1] @ R_dev.T
            base_shift = translation_offset.to(pred_cam[1].device, dtype=pred_cam[1].dtype)  # (1,3)
            extra_shift = extra_trans.to(pred_cam[1].device, dtype=pred_cam[1].dtype)  # (T,3)
            pred_cam[1] = pred_cam[1] + base_shift + extra_shift

        hmr_payload["pred_cam"] = pred_cam

    hmr_dst = seq_output_root / "hmr" / "hps_track.npy"
    hmr_dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(hmr_dst, hmr_payload)

    export_to_targets(seq_name, seq_output_root, export_motion_root, export_urdf_root, export_tag)
    print(f"[OK] Finished {seq_name} -> {seq_output_root}")


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description=(
            "Rotate scene and HMR using gravity alignment, then compute ONE shared z-translation (shape [1,3]) "
            "so that the rotated scene min z becomes 0, and apply it to BOTH scene and HMR."
        )
    )
    parser.add_argument("--seq-names", nargs="+", help="Sequence names to process. If omitted, auto-discovers under input-root.")
    parser.add_argument("--hmr-type", default="gv", help="Name of the HMR subfolder to use.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT, help="Root containing raw scene outputs.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Where to save rotated+shifted outputs.")
    parser.add_argument("--camera-npz", type=Path, default=None, help="Optional explicit camera NPZ (ignored by default pipeline).")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Dataset root used to locate reference images.")
    parser.add_argument("--no-megasam", action="store_true", help="Disable MegaSAM-specific image normalization.")
    parser.add_argument(
        "--optimize-human",
        action="store_true",
        help="Optional: human-only per-frame z grounding after shared translation (NOT a single shared translation).",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Deprecated alias to disable human optimization (kept for compatibility).",
    )
    parser.add_argument("--debug-stride", type=int, default=10, help="Stride for dumping debug OBJ meshes.")
    parser.add_argument("--export-motion-root", type=Path, default=None, help="Destination root for motion npz export.")
    parser.add_argument("--export-urdf-root", type=Path, default=None, help="Destination root for URDF/mesh export.")
    parser.add_argument("--export-tag", default=None, help="Optional tag subdirectory when exporting assets.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    repo_root = REPO_ROOT

    input_root = args.input_root if args.input_root.is_absolute() else (repo_root / args.input_root)
    output_root = args.output_root if args.output_root.is_absolute() else (repo_root / args.output_root)
    data_root = args.data_root if args.data_root.is_absolute() else (repo_root / args.data_root)
    camera_npz = (
        args.camera_npz
        if args.camera_npz and args.camera_npz.is_absolute()
        else (repo_root / args.camera_npz if args.camera_npz else None)
    )
    export_motion_root = (
        args.export_motion_root
        if args.export_motion_root is None or args.export_motion_root.is_absolute()
        else repo_root / args.export_motion_root
    )
    export_urdf_root = (
        args.export_urdf_root
        if args.export_urdf_root is None or args.export_urdf_root.is_absolute()
        else repo_root / args.export_urdf_root
    )

    seq_names = args.seq_names if args.seq_names else discover_sequences(input_root, args.hmr_type)
    if not seq_names:
        print(f"[WARN] No sequences found under {input_root} for hmr_type={args.hmr_type}")
        return

    # Enforce deprecated flag behavior
    optimize_human = bool(args.optimize_human) and (not args.no_optimize)

    for seq_name in seq_names:
        process_sequence(
            seq_name=seq_name,
            input_root=input_root,
            output_root=output_root,
            hmr_type=args.hmr_type,
            camera_npz=camera_npz,
            data_root=data_root,
            is_megasam=not args.no_megasam,
            optimize_human=optimize_human,
            export_motion_root=export_motion_root,
            export_urdf_root=export_urdf_root,
            export_tag=args.export_tag,
            debug_stride=args.debug_stride,
        )


if __name__ == "__main__":
    main()
