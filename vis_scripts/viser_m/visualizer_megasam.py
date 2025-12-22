import time
import sys
import argparse
import types
from pathlib import Path
import trimesh
import numpy as onp
import tyro
from tqdm.auto import tqdm
# from sqs_utils.superquadric import *
from pytorch3d.transforms import euler_angles_to_matrix
import os, shutil
import copy
import viser
import viser.extras
import viser.transforms as tf
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans        # fast & GPU‑friendly via torch tensors
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # For colormap
import smplx

import torch
import os
import cv2
import numpy as np
import argparse
from scipy.ndimage import distance_transform_edt
import vdbfusion
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, sobel
import json
from itertools import product
from path_config import (
    BODY_SEGMENTS_DIR,
    CONTACTS_ROOT,
    CONTACT_CACHE_DIR,
    FLOWS_COVISIBILITY_DIR,
    GVHMR_ROOT,
    SCENE_OUTPUT_DIR,
    VSLAM_MEGACAM_DIR,
    VSLAM_PRIORS_DIR,
)

SQ_CORNER_SIGNS = np.array(list(product([-1.0, 1.0], repeat=3)), dtype=np.float32)
from optim_utils import *
import numpy as np, trimesh, xml.etree.ElementTree as ET
from pathlib import Path
import scipy
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import trimesh
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import trimesh
from utils import *
from toy_exp.simple_vis import SimpleVis as Vis
from pathlib import Path
import shutil
import numpy as np

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Literal

from pytorch3d.structures import Meshes

try:
    from scontact_utils import (
        analyze_contacts_5parts,
        extract_contact_points_from_single_frame,
        load_contact_points_for_parts,
    )
    CONTACT_UTILS_AVAILABLE = True
    CONTACT_IMPORT_ERROR: Optional[Exception] = None
except Exception as contact_exc:  # pragma: no cover - best effort import guard
    CONTACT_UTILS_AVAILABLE = False
    CONTACT_IMPORT_ERROR = contact_exc
    analyze_contacts_5parts = None
    extract_contact_points_from_single_frame = None
    load_contact_points_for_parts = None

CAN_RUN_CONTACT_ANALYSIS = CONTACT_UTILS_AVAILABLE and analyze_contacts_5parts is not None
CAN_EXTRACT_SINGLE_FRAME_CONTACTS = CONTACT_UTILS_AVAILABLE and extract_contact_points_from_single_frame is not None
def _sanitize_trimesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    finite = np.isfinite(V).all(axis=1)
    if not finite.all():
        map_idx = -np.ones(len(V), dtype=np.int64)
        map_idx[finite] = np.arange(finite.sum())
        V2 = V[finite]
        F2 = F[np.all(finite[F], axis=1)]
        F2 = map_idx[F2]
        mesh = trimesh.Trimesh(V2, F2, process=False)
    mesh.remove_unreferenced_vertices()
    return mesh

def export_scene_as_urdf(parts, transforms, out_dir):
    """
    parts       : list[trimesh.Trimesh]               原始子网格（局部坐标）
    transforms  : list[np.ndarray, shape (4,4)]       每个子网格到世界的齐次矩阵
    out_dir     : Path | str                          保存 *.obj 和 scene.urdf 的目录
    """
    out_dir = Path(out_dir)
    scene_type = out_dir.name
    urdf_path = out_dir

    
    # Create fresh output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    out_dir = out_dir / 'pieces'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ─── URDF 根 ──────────────────────────────────────────────────────────────
    robot = ET.Element("robot", name="scene")

    # 只有一个 link，名字随意；这里叫 “scene_link”
    scene_link = ET.SubElement(robot, "link", name="scene_link")

    # ─── 逐个导出网格并在 link 下添加 visual / collision 对 ────────────────
    for i, (mesh_local, T) in enumerate(zip(parts, transforms)):
        name = f"part_{i:03d}"
        mesh_file = out_dir / f"{name}.obj"
        mesh_local.export(mesh_file)

        # 位姿拆分
        xyz = T[:3, 3]
        rpy = to_rpy(T[:3, :3])

        # 可视 & 碰撞；两段完全一样，只是 tag 不同
        for tag in ("visual", "collision"):
            sec = ET.SubElement(scene_link, tag)
            ET.SubElement(
                sec,
                "origin",
                xyz=" ".join(f"{v:.6f}" for v in xyz),
                rpy=" ".join(f"{v:.6f}" for v in rpy),
            )
            geom = ET.SubElement(sec, "geometry")
            ET.SubElement(
                geom,
                "mesh",
                filename=str(mesh_file.relative_to(out_dir)),
            )

    # ─── 写盘 ────────────────────────────────────────────────────────────────
    ET.ElementTree(robot).write(
        urdf_path / f"{scene_type}.urdf",
        encoding="utf-8",
        xml_declaration=True,
    )

def save_custom_mesh(mesh_parts, tgt_folder):
    scene = trimesh.Scene()
    keep_T = []
    for p in mesh_parts:
        scene.add_geometry(p)
    for name, mesh in scene.geometry.items():
        T_world, _ = scene.graph.get(name)     # 4×4
        mesh_world = mesh.copy()
        mesh_world.apply_transform(T_world)    # 变到世界坐标系，做接触测试
        keep_T.append(T_world)

    export_scene_as_urdf(mesh_parts, keep_T, tgt_folder)


def convert_results_to_params_direct(
    results: dict,
    eps1: float = -2.398,     # log(0.1) roundness default
    eps2: float = -2.398,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Convert refined SQ results to the 11-parameter vector
        [eps1, eps2,
         sx,  sy,  sz,
         rz,  ry,  rx,     # ZYX order
         tx,  ty,  tz]     # (N,11) tensor
    """
    S_items = results['S_items']          # log half-axes, torch(3,)
    R_items = results['R_items']          # 3×3 rotations (body→world) **after** re-orthonorm.
    T_items = results['T_items']          # 3-vec centres

    # Optional per-SQ epsilon values (already in log space)
    eps_items = results.get('eps_items', None)

    params = []
    for i, (S_log, R_bw, T) in enumerate(zip(S_items, R_items, T_items)):
        # ----- epsilons -----------------------------------------------
        if eps_items is not None and i < len(eps_items):
            eps1_i, eps2_i = eps_items[i]
        else:
            eps1_i, eps2_i = eps1, eps2          # global default

        # ----- scale ---------------------------------------------------
        sx, sy, sz = torch.exp(S_log).tolist()   # back to linear half-axes

        # ----- rotation  (body→world matrix → Euler ZYX) ---------------
        R_wb = R_bw.T                            # pytorch3d expects world→body on batch dim
        rz, ry, rx = matrix_to_euler_angles(
            R_wb.unsqueeze(0), convention='ZYX'
        ).squeeze(0).tolist()

        # ----- translation --------------------------------------------
        tx, ty, tz = T.tolist()

        params.append([
            eps1_i, eps2_i,
            sx, sy, sz,
            rz, ry, rx,
            tx, ty, tz
        ])

    if not params:
        return torch.zeros((0, 11), device=device)

    return torch.tensor(params, dtype=torch.float32, device=device)


def export_superquadrics(params, *,
                         out_dir="meshes",
                         stem="sq",
                         combine=True,
                         lat_res=64, lon_res=128,
                         filetype="obj"):
    """
    params   : (N,11) array -- one SQ per row
    combine  : True  → single file   (stem_scene.obj/ply)
               False → one file per SQ (stem_00.obj, …)
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    meshes  = []
    for p in params:
        sq = superquadric(p[0:2], p[2:5], p[5:8], p[8:11])
        V, F = sq.get_mesh(lat_res=lat_res, lon_res=lon_res)
        meshes.append(trimesh.Trimesh(vertices=V, faces=F, process=False))

    if combine:
        scene  = trimesh.util.concatenate(meshes)
        return scene 
        fname  = out_dir / f"{stem}_scene.{filetype}"
        scene.export(fname)
        print(f"wrote {fname}  ({len(scene.vertices)} verts, {len(scene.faces)} faces)")


def load_contact_sequence(interact_contact_path, start_frame=0, end_frame=-1, thre=0.5):
    """加载一段连续帧的contact数据"""
    contact_sequence = []
    for i in range(start_frame, end_frame + 1):
        contact_data = np.load(os.path.join(interact_contact_path, f'{i:05d}.npz'))['pred_contact_3d_smplh']
        contact_sequence.append(contact_data > thre)  # 转换为布尔mask
    return np.array(contact_sequence)  # shape: [num_frames, 6890]



def clean_and_make(dir_path: str | Path) -> Path:
    """
    Remove *everything* that may already be in `dir_path` and then
    recreate the (now-empty) directory.  Returns the directory as a Path.
    """
    p = Path(dir_path)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """
    将 3×3 旋转矩阵转 roll-pitch-yaw (XYZ 固定轴) 顺序。
    返回值单位：rad
    """
    # 这里用的是 ROS 的惯用定义：R = Rz(yaw) * Ry(pitch) * Rx(roll)
    sy = np.hypot(R[0, 0], R[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll  = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = np.arctan2(R[1, 0], R[0, 0])
    else:           # pitch ≈ ±90°
        roll  = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = 0.0
    return roll, pitch, yaw





def contact_ids():
    body_segments_dir = str(BODY_SEGMENTS_DIR)
    contact_verts_ids = []
    contact_body_parts = ['L_Leg', 'R_Leg', 'L_Hand', 'R_Hand', 'gluteus', 'back', 'thighs']
    for part in contact_body_parts:
        with open(os.path.join(body_segments_dir, part + '.json'), 'r') as f:
            data = json.load(f)
            contact_verts_ids.append(list(set(data["verts_ind"])))
    contact_verts_ids = np.concatenate(contact_verts_ids)
    return contact_verts_ids
def load_contact_ids(device="cuda"):
    ids_np = contact_ids()                 # numpy array from your helper
    ids    = torch.as_tensor(ids_np, dtype=torch.long, device=device)
    return ids  
def axis_angle_to_matrix_(rotvecs):
    """
    Convert axis-angle to rotation matrix.
    Input: rotvecs [T, 21, 3]
    Output: rotmats [T, 21, 3, 3]
    """
    theta = torch.norm(rotvecs, dim=-1, keepdim=True)  # [T, 21, 1]
    axis = rotvecs / (theta + 1e-8)  # Avoid div-by-zero
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]  # [T, 21]

    cos = torch.cos(theta)[..., 0]  # [T, 21]
    sin = torch.sin(theta)[..., 0]
    one_minus_cos = 1 - cos

    rot = torch.zeros(rotvecs.shape[:-1] + (3, 3), device=rotvecs.device)

    rot[..., 0, 0] = cos + x * x * one_minus_cos
    rot[..., 0, 1] = x * y * one_minus_cos - z * sin
    rot[..., 0, 2] = x * z * one_minus_cos + y * sin
    rot[..., 1, 0] = y * x * one_minus_cos + z * sin
    rot[..., 1, 1] = cos + y * y * one_minus_cos
    rot[..., 1, 2] = y * z * one_minus_cos - x * sin
    rot[..., 2, 0] = z * x * one_minus_cos - y * sin
    rot[..., 2, 1] = z * y * one_minus_cos + x * sin
    rot[..., 2, 2] = cos + z * z * one_minus_cos

    return rot



import numpy as np
import torch


def tsdf_to_voxelGrid(vdb_volume: vdbfusion.VDBVolume,
                      trunc_multiple: float = 1.0,
                      cube_res: int = 256):
    """
    Convert ``vdbfusion.VDBVolume`` → (sdf_flat, voxelGrid).
    
    Parameters
    ----------
    trunc_multiple : float
        Truncation distance as multiples of the voxel size (default 2×).
    cube_res : int or None
        • None   → keep native (Nx,Ny,Nz).  
        • int    → resample to that *cubic* resolution (e.g. 100) to
                    replicate the original CSV convention.

    Returns
    -------
    sdf_flat : (N,) np.ndarray   – Fortran‑order flattening (X fastest).
    voxelGrid : dict             – ready for ``_marching_primitives``.
    """
    # ────── 1. Extract TSDF & bounding box ────────────────────────────
    grid              = vdb_volume.tsdf
    min_ijk, max_ijk  = grid.evalActiveVoxelBoundingBox()   # (i,j,k)

    Nx = max_ijk[0] - min_ijk[0] + 1
    Ny = max_ijk[1] - min_ijk[1] + 1
    Nz = max_ijk[2] - min_ijk[2] + 1
    tsdf_np = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    grid.copyToArray(tsdf_np, ijk=min_ijk)

    # ────── 2. World‑space AABB (metres) ─────────────────────────────
    T     = grid.transform
    xmin, ymin, zmin = T.indexToWorld(min_ijk)
    xmax, ymax, zmax = T.indexToWorld(max_ijk)

    # ────── 3. Optional resample to a cubic grid ─────────────────────
    if cube_res is not None:
        # scale factors along each axis
        zoom_xyz = (cube_res / Nx, cube_res / Ny, cube_res / Nz)
        tsdf_np  = scipy.ndimage.zoom(tsdf_np, zoom_xyz, order=1)  # trilinear
        Nx, Ny, Nz = cube_res, cube_res, cube_res

    # ────── 4. Coordinate lin‑spaces & point cloud ───────────────────
    x_lin = np.linspace(xmin, xmax, Nx, dtype=float)
    y_lin = np.linspace(ymin, ymax, Ny, dtype=float)
    z_lin = np.linspace(zmin, zmax, Nz, dtype=float)

    X, Y, Z   = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')  # (Nx,Ny,Nz)
    pts_flat  = np.stack((X, Y, Z), axis=3).reshape(-1, 3, order='F').T

    # ────── 5. Pack outputs ──────────────────────────────────────────
    voxel_size = float(vdb_volume.voxel_size)
    truncation = trunc_multiple * voxel_size

    sdf_flat = tsdf_np.flatten(order='F')      # (Nx*Ny*Nz,)

    voxelGrid = {
        'size':        np.array([Nx, Ny, Nz], dtype=int),
        'range':       np.array([xmin, xmax, ymin, ymax, zmin, zmax],
                                dtype=float),
        'x':           x_lin,
        'y':           y_lin,
        'z':           z_lin,
        'points':      pts_flat,
        'interval':    voxel_size,
        'truncation':  truncation,
        'disp_range':  [-np.inf, truncation],
        'visualizeArclength': 0.01 *
                              np.linalg.norm([xmax - xmin,
                                              ymax - ymin,
                                              zmax - zmin]),
    }

    return sdf_flat, voxelGrid




def filter_bg_points(bg_pos: np.ndarray,
                     bg_col: np.ndarray,
                     contact_verts: np.ndarray,
                     dist_thr: float = 1):
    """
    bg_pos        : (N, 3)  background XYZ points   (float32/float64)
    bg_col        : (N, 3)  corresponding RGB       (same length as bg_pos)
    contact_verts : (T, M, 3) or (M, 3)

    Keeps a bg point only if it lies within `dist_thr` (meters) of
    at least one contact vertex.  Returns the filtered bg_pos/bg_col.
    """
    # Flatten contact verts to (K, 3) where K = T*M   (or M if already (M,3))
    contact_flat = contact_verts.reshape(-1, 3)

    # Pair-wise squared distances → (N, K)
    diff2 = ((bg_pos[:, None, :] - contact_flat[None, :, :]) ** 2).sum(axis=-1)

    # For each bg point, take its nearest contact-vertex distance
    min_dist2 = np.min(diff2, axis=1)

    keep_mask = min_dist2 <= dist_thr ** 2
    return bg_pos[keep_mask], bg_col[keep_mask]


def merge_plane_primitives(
    plane_accum: dict,
    normal_tol_deg: float = 10.0,
    normal_offset_tol: float = 0.05,
    tangent_offset_tol: float = 0.25,
) -> dict:
    """Merge noisy duplicate planes that are co-planar and close."""
    if not plane_accum['S_items']:
        return plane_accum

    def _to_numpy(arr) -> np.ndarray:
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy().astype(np.float32, copy=False)
        return np.asarray(arr, dtype=np.float32)

    def _normalize(vec: np.ndarray) -> np.ndarray:
        return vec / (np.linalg.norm(vec) + 1e-8)

    def _bounds_from_points(R_ref: np.ndarray, T_ref: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        local = (R_ref.T @ (pts - T_ref).T).T
        return local.min(axis=0), local.max(axis=0)

    def _bounds_from_box(
        R_ref: np.ndarray,
        T_ref: np.ndarray,
        box_R: np.ndarray,
        half_ext: np.ndarray,
        box_T: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        corners_local = SQ_CORNER_SIGNS * half_ext
        corners_world = corners_local @ box_R.T + box_T
        return _bounds_from_points(R_ref, T_ref, corners_world)

    ref = plane_accum['S_items'][0]
    if isinstance(ref, torch.Tensor):
        device = ref.device
        dtype = ref.dtype
    else:
        device = torch.device('cpu')
        dtype = torch.float32

    cos_tol = float(np.cos(np.deg2rad(normal_tol_deg)))
    merged: list[dict[str, np.ndarray]] = []

    for S_item, R_item, T_item, color_item, pts_item in zip(
        plane_accum['S_items'],
        plane_accum['R_items'],
        plane_accum['T_items'],
        plane_accum['color_items'],
        plane_accum['pts_items'],
    ):
        S_np = _to_numpy(S_item)
        R_np = _to_numpy(R_item)
        T_np = _to_numpy(T_item)
        color_np = _to_numpy(color_item)
        pts_np = _to_numpy(pts_item)
        pts_np = pts_np.reshape(-1, 3) if pts_np.size else np.empty((0, 3), dtype=np.float32)
        half_np = np.exp(S_np)

        normal = _normalize(R_np[:, 2])
        merged_flag = False

        for existing in merged:
            normal_dot = float(existing['normal'].dot(normal))
            if abs(normal_dot) < cos_tol:
                continue

            if normal_dot < 0.0:
                flip = np.diag([1.0, -1.0, -1.0])
                R_np = R_np @ flip
                normal = -normal

            delta = T_np - existing['T']
            delta_normal = float(delta.dot(existing['normal']))
            normal_sep = abs(delta_normal)

            normal_allow = max(normal_offset_tol, 0.5 * (existing['half'][2] + half_np[2]))
            if normal_sep > normal_allow:
                continue

            tangential_vec = delta - delta_normal * existing['normal']

            existing_min = -existing['half']
            existing_max = existing['half']

            if pts_np.size:
                new_min, new_max = _bounds_from_points(existing['R'], existing['T'], pts_np)
            else:
                new_min, new_max = _bounds_from_box(existing['R'], existing['T'], R_np, half_np, T_np)

            gap_tol = 0.08
            iou_thresh = 0.2

            min_ax, max_ax = existing_min[0], existing_max[0]
            min_ay, max_ay = existing_min[1], existing_max[1]
            min_bx, max_bx = new_min[0], new_max[0]
            min_by, max_by = new_min[1], new_max[1]

            len_ax = max_ax - min_ax
            len_ay = max_ay - min_ay
            len_bx = max_bx - min_bx
            len_by = max_by - min_by

            overlap_x = min(max_ax, max_bx) - max(min_ax, min_bx)
            overlap_y = min(max_ay, max_by) - max(min_ay, min_by)
            inter_x = max(0.0, overlap_x)
            inter_y = max(0.0, overlap_y)
            inter_area = inter_x * inter_y
            area_a = max(0.0, len_ax) * max(0.0, len_ay)
            area_b = max(0.0, len_bx) * max(0.0, len_by)
            union_area = area_a + area_b - inter_area
            iou = inter_area / (union_area + 1e-6)

            gap_x = max(0.0, -overlap_x)
            gap_y = max(0.0, -overlap_y)

            tangential_ok = (
                (inter_area > 0.0 and iou >= iou_thresh)
                or (gap_x <= gap_tol and gap_y <= gap_tol)
            )

            if not tangential_ok:
                tangential_ok = np.linalg.norm(tangential_vec) <= tangent_offset_tol

            if not tangential_ok:
                continue

            old_pts = existing['pts']
            if old_pts.size and pts_np.size:
                all_pts = np.vstack((old_pts, pts_np))
            elif pts_np.size:
                all_pts = pts_np.copy()
            else:
                all_pts = old_pts

            old_count = old_pts.shape[0]
            new_count = pts_np.shape[0]
            total_count = old_count + new_count
            if total_count:
                existing['color'] = (
                    existing['color'] * old_count + color_np * new_count
                ) / float(total_count)

            if all_pts.size:
                prev_center = existing['T']
                local = (existing['R'].T @ (all_pts - prev_center).T).T
                min_local = local.min(axis=0)
                max_local = local.max(axis=0)
                center_local = 0.5 * (min_local + max_local)
                existing['T'] = prev_center + existing['R'] @ center_local
                half_ext = 0.5 * (max_local - min_local)
            else:
                half_ext = existing['half']

            half_ext = np.maximum(half_ext, 1e-3)
            existing['S'] = np.log(half_ext)
            existing['pts'] = all_pts
            existing['half'] = half_ext
            existing['normal'] = _normalize(existing['R'][:, 2])
            merged_flag = True
            break

        if not merged_flag:
            merged.append({
                'S': S_np,
                'R': R_np,
                'T': T_np,
                'color': color_np,
                'pts': pts_np,
                'normal': normal,
                'half': half_np,
            })

    new_accum = {key: [] for key in plane_accum}
    for entry in merged:
        new_accum['S_items'].append(torch.as_tensor(entry['S'], device=device, dtype=dtype))
        new_accum['R_items'].append(torch.as_tensor(entry['R'], device=device, dtype=dtype))
        new_accum['T_items'].append(torch.as_tensor(entry['T'], device=device, dtype=dtype))
        new_accum['color_items'].append(torch.as_tensor(entry['color'], device=device, dtype=dtype))
        pts = entry['pts']
        if pts.size:
            new_accum['pts_items'].append(torch.as_tensor(pts, device=device, dtype=dtype))
        else:
            new_accum['pts_items'].append(torch.empty((0, 3), device=device, dtype=dtype))

    return new_accum


def cluster_plane_primitives(
    plane_accum: dict,
    normal_tol_deg: float = 6.0,
    normal_gap: float = 0.08,
    tangential_gap: float = 0.18,
    iou_thresh: float = 0.08,
) -> dict:
    """Agglomeratively cluster planes that still overlap after merging."""
    if not plane_accum['S_items']:
        return plane_accum

    def _to_numpy(arr) -> np.ndarray:
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy().astype(np.float32, copy=False)
        return np.asarray(arr, dtype=np.float32)

    ref = plane_accum['S_items'][0]
    if isinstance(ref, torch.Tensor):
        device = ref.device
        dtype = ref.dtype
    else:
        device = torch.device('cpu')
        dtype = torch.float32

    planes = []
    normals = []
    centers = []
    halves = []
    corners = []

    for S_item, R_item, T_item, color_item, pts_item in zip(
        plane_accum['S_items'],
        plane_accum['R_items'],
        plane_accum['T_items'],
        plane_accum['color_items'],
        plane_accum['pts_items'],
    ):
        S_np = _to_numpy(S_item)
        R_np = _to_numpy(R_item)
        T_np = _to_numpy(T_item)
        color_np = _to_numpy(color_item)
        pts_np = _to_numpy(pts_item)
        pts_np = pts_np.reshape(-1, 3) if pts_np.size else np.empty((0, 3), dtype=np.float32)
        half_np = np.maximum(np.exp(S_np), 1e-3)

        planes.append({
            'S': S_np,
            'R': R_np,
            'T': T_np,
            'color': color_np,
            'pts': pts_np,
            'half': half_np,
        })
        normal = R_np[:, 2]
        normals.append(normal / (np.linalg.norm(normal) + 1e-8))
        centers.append(T_np)
        halves.append(half_np)
        corners_local = SQ_CORNER_SIGNS * half_np
        corners.append(corners_local @ R_np.T + T_np)

    normals = np.stack(normals)
    centers = np.stack(centers)
    halves = np.stack(halves)
    N = len(planes)
    if N == 1:
        return plane_accum

    cos_thresh = float(np.cos(np.deg2rad(normal_tol_deg)))

    parent = list(range(N))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(N):
        for j in range(i + 1, N):
            n_dot = float(np.dot(normals[i], normals[j]))
            if abs(n_dot) < cos_thresh:
                continue

            normal_i = normals[i]
            center_delta = centers[j] - centers[i]
            normal_sep = abs(float(np.dot(center_delta, normal_i)))
            allow = max(normal_gap, 0.5 * (halves[i][2] + halves[j][2]))
            if normal_sep > allow:
                continue

            R_i = planes[i]['R']
            T_i = planes[i]['T']
            local_corners_j = (R_i.T @ (corners[j] - T_i).T).T
            local_corners_i = (R_i.T @ (corners[i] - T_i).T).T

            min_i = local_corners_i.min(axis=0)
            max_i = local_corners_i.max(axis=0)
            min_j = local_corners_j.min(axis=0)
            max_j = local_corners_j.max(axis=0)

            len_ix = max_i[0] - min_i[0]
            len_iy = max_i[1] - min_i[1]
            len_jx = max_j[0] - min_j[0]
            len_jy = max_j[1] - min_j[1]

            overlap_x = min(max_i[0], max_j[0]) - max(min_i[0], min_j[0])
            overlap_y = min(max_i[1], max_j[1]) - max(min_i[1], min_j[1])
            inter_x = max(0.0, overlap_x)
            inter_y = max(0.0, overlap_y)
            inter_area = inter_x * inter_y
            area_i = max(0.0, len_ix) * max(0.0, len_iy)
            area_j = max(0.0, len_jx) * max(0.0, len_jy)
            union_area = area_i + area_j - inter_area
            iou = inter_area / (union_area + 1e-6)

            gap_x = max(0.0, -overlap_x)
            gap_y = max(0.0, -overlap_y)

            if (inter_area > 0.0 and iou >= iou_thresh) or (
                gap_x <= tangential_gap and gap_y <= tangential_gap
            ):
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for idx in range(N):
        root = find(idx)
        clusters.setdefault(root, []).append(idx)

    if len(clusters) == N:
        return plane_accum

    new_accum = {key: [] for key in plane_accum}

    for indices in clusters.values():
        ref_idx = indices[0]
        ref_plane = planes[ref_idx]
        ref_R = ref_plane['R']
        ref_T = ref_plane['T']
        ref_normal = normals[ref_idx]

        all_pts = []
        corner_pool = []
        color_acc = np.zeros(3, dtype=np.float32)
        area_acc = 0.0

        for idx in indices:
            plane = planes[idx]
            normal = normals[idx]
            if np.dot(normal, ref_normal) < 0:
                flip = np.diag([1.0, -1.0, -1.0])
                R_adj = plane['R'] @ flip
                corners_world = (SQ_CORNER_SIGNS * plane['half']) @ R_adj.T + plane['T']
            else:
                R_adj = plane['R']
                corners_world = (SQ_CORNER_SIGNS * plane['half']) @ R_adj.T + plane['T']

            corner_pool.append(corners_world)
            if plane['pts'].size:
                all_pts.append(plane['pts'])
            area = 4.0 * plane['half'][0] * plane['half'][1]
            color_acc += plane['color'] * area
            area_acc += area

        corner_pool = np.concatenate(corner_pool, axis=0)
        local = (ref_R.T @ (corner_pool - ref_T).T).T
        min_local = local.min(axis=0)
        max_local = local.max(axis=0)
        center_local = 0.5 * (min_local + max_local)
        half_ext = 0.5 * (max_local - min_local)
        half_ext = np.maximum(half_ext, 1e-3)

        new_T = ref_T + ref_R @ center_local
        new_S = np.log(half_ext)

        if all_pts:
            merged_pts = np.concatenate(all_pts, axis=0)
        else:
            merged_pts = np.empty((0, 3), dtype=np.float32)

        new_color = color_acc / max(area_acc, 1e-6)

        new_accum['S_items'].append(torch.as_tensor(new_S, device=device, dtype=dtype))
        new_accum['R_items'].append(torch.as_tensor(ref_R, device=device, dtype=dtype))
        new_accum['T_items'].append(torch.as_tensor(new_T, device=device, dtype=dtype))
        new_accum['color_items'].append(torch.as_tensor(new_color, device=device, dtype=dtype))
        new_accum['pts_items'].append(torch.as_tensor(merged_pts, device=device, dtype=dtype))

    return new_accum


import torch
def to_cuda(data):
    """Move data in the batch to cuda(), carefully handle data that is not tensor"""
    if isinstance(data, torch.Tensor):
        return data.cuda()
    elif isinstance(data, dict):
        return {k: to_cuda(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_cuda(v) for v in data]
    else:
        return data
def axis_angle_to_matrix(angle_axis):
    # angle_axis: [N, 3]
    theta = torch.norm(angle_axis, dim=1, keepdim=True)  # [N, 1]
    axis = angle_axis / (theta + 1e-6)  # [N, 3]
    
    K = torch.zeros(angle_axis.shape[0], 3, 3, device=angle_axis.device)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]
    
    I = torch.eye(3, device=angle_axis.device).unsqueeze(0)
    R = I + torch.sin(theta).unsqueeze(-1) * K + (1 - torch.cos(theta)).unsqueeze(-1) * torch.bmm(K, K)
    return R  # [N, 3, 3]


def get_intrinsics_matrix(camera_params):
    """
    Returns a 3x3 camera intrinsics matrix from a dictionary input.
    
    Expected keys in `camera_params`:
        - 'img_focal': horizontal focal length (fx)
        - 'img_center': numpy array or list with [cx, cy]
        - 'spec_focal': vertical focal length (fy)
    """
    fx = camera_params['img_focal']
    fy = camera_params['spec_focal']
    cx, cy = camera_params['img_center']
    
    K = np.array([
        [fx, 0, cx],
        [0, fx, cy],
        [0, 0, 1]
    ])
    return K 

def matrix_to_axis_angle(R):
    # R: [N, 3, 3]
    cos_theta = ((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]) - 1) / 2
    theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))  # [N]
    
    rx = R[:, 2, 1] - R[:, 1, 2]
    ry = R[:, 0, 2] - R[:, 2, 0]
    rz = R[:, 1, 0] - R[:, 0, 1]
    r = torch.stack([rx, ry, rz], dim=1)  # [N, 3]

    # Normalize rotation axis
    r_norm = torch.norm(r, dim=1, keepdim=True) + 1e-6
    axis = r / r_norm

    return axis * theta.unsqueeze(1)  # [N, 3]
def build_4x4_poses(world_cam_R, world_cam_T):
    N = len(world_cam_R)
    all_poses = []

    for i in range(N):
        # Create a 4x4 identity
        pose = np.eye(4)

        # Fill in the top-left 3x3 with the rotation
        pose[:3, :3] = world_cam_R[i]

        # Fill in the top-right 3x1 with the translation
        pose[:3, 3] = world_cam_T[i]

        all_poses.append(pose)

    # Stack into [N, 4, 4]
    return np.stack(all_poses, axis=0)

def _load_npz_to_dict(path: Path, *, allow_pickle: bool = False) -> dict:
    """Utility: load .npz into a writable dict."""
    with np.load(path, allow_pickle=allow_pickle) as f:
        return {k: f[k] for k in f}


def main(
    data: Path = "./demo_tmp/NULL.npz",
    downsample_factor: int = 1,
    max_frames: int = 800,
    share: bool = False,
    conf_threshold: float = 0.3,
    foreground_conf_threshold: float = 0.,
    point_size: float = 0.01,
    camera_frustum_scale: float = 0.02,
    show_camera_frusta: bool = True,
    no_mask: bool = False,
    xyzw: bool = True,
    axes_scale: float = 0.25,
    bg_downsample_factor: int = 1,
    init_conf: bool = False,
    cam_thickness: float = 1.5,
    save_mode: bool = False, 
    transfer_data: bool = False, 
    hmr_type: str = 'gv',
    moge_base_path: Path | str | None = None,
    detailed_planes: bool = False,
    segment_mode: Literal["frame_union", "cluster_3d"] = "frame_union",
    static_camera: bool = False,
    sq_loss_threshold: Optional[float] = None,
    save_clustering: bool = True,
) -> None:
    from pathlib import Path  # <-- Import Path here if not already imported
    tgt_name = str(data).split('_sgd')[0].split('/')[-1]   # gives 'MPH112_00169_01_tram'
    tgt_name = "_".join(tgt_name.split("_")[:-1]) 
    if moge_base_path is None:
        moge_base_path = VSLAM_PRIORS_DIR
    moge_base_path = Path(moge_base_path)
    moge_data = moge_base_path / f'{tgt_name}.npz'
    # compare_npz_shapes(data, moge_data)
    tgt_folder = str((SCENE_OUTPUT_DIR / tgt_name))
    # if Path(tgt_folder).exists():
    #   shutil.rmtree(Path(tgt_folder))


    if 'door' in str(data):
        extra_obj = True
    
    # data = np.load(data)
    data = _load_npz_to_dict(data)
    moge_data = np.load(moge_data)
    data["depths"] = moge_data["depths"]
    data["images"] = moge_data["images"]
    data['cam_c2w'] = moge_data['cam_c2w']
    data['intrinsic'] = moge_data['intrinsic']

    # data['scale'] = 1 3.184327021929371
    # print(moge_data['cam_c2w'][0], data['cam_c2w'][0])
    human_mesh_handles: list[viser.MeshHandle] = []
    contact_vertices_handles: list[viser.MeshHandle] = []
    contact_global_handles: list[viser.PointCloudHandle] = []
    per_part_contacts: Optional[Dict[str, np.ndarray]] = None
    contact_points: Optional[np.ndarray] = None
    plane_mesh_handles: list[viser.MeshHandle] = []
    scene_mesh_raw_handle: Optional[viser.MeshHandle] = None
    scene_mesh_coacd_handle: Optional[viser.MeshHandle] = None
    scene_mesh_coacd_contact_handle: Optional[viser.MeshHandle] = None

    import random

    def generate_four_digit():
        return random.randint(1000, 9999)
    server = viser.ViserServer(port=generate_four_digit())
    # serializer = server.get_scene_serializer()
    if share:
        server.request_share_url()

    ratio_block_scene =1 
    do_mesh = True
    use_world = True
    server.scene.set_up_direction('-z')
    if no_mask or not do_mesh:             # not using dynamic / static mask
        init_conf = True    # must use init_conf map, to avoid depth cleaning
        fg_conf_thre = conf_threshold # now fg_conf_thre is the same as conf_thre
    print("Loading frames!")
    
    key_R, key_T =  'world_cam_R', 'world_cam_T' 

    num_frames = len(data['depths'])# min(max_frames, len(data['depths'])) 
    # num_frames = max_frames

    npz_cam_data = data# pred_cam
    device='cuda'
    base = VSLAM_MEGACAM_DIR
    candid = sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith(tgt_name))

    if not candid:
        raise FileNotFoundError(f"No sub-directory in {base} starts with “{tgt_name}”.")
    if len(candid) > 1:
        print(f"[warn] multiple matches, picking {candid[0].name!r}")

    camera = np.load(candid[0] / "camera.npy", allow_pickle=True).item()

    fx= fy = img_focal = camera['img_focal']
    pred_cam ={}
    pred_cam[key_R] = npz_cam_data['cam_c2w'][:, :3, :3]
    pred_cam[key_T] = (npz_cam_data['cam_c2w'][:, :3, 3] )* npz_cam_data['scale']
    world_cam_R = torch.tensor(pred_cam[key_R]).to(device)#[:num_frames]
    world_cam_T = torch.tensor(pred_cam[key_T]).to(device)#[:num_frames]

    from smpl_utils import (
        process_tram_smpl,
        process_gv_smpl,
        load_contact_ids_from_file,
        load_contact_ids_with_mode, 
        filter_vertices_by_contact
    )
    

    device='cuda'

    interact_contact_path = str(CONTACTS_ROOT / tgt_name)


    

    if hmr_type == 'gv':
        smpl_results = process_gv_smpl(
            tgt_name=tgt_name,
            world_cam_R=world_cam_R,
            world_cam_T=world_cam_T,
            max_frames=max_frames,
            smpl_model=None,
            use_world=use_world,
            device='cuda'
        )
        
        num_frames = smpl_results['num_frames']
        global_orient_world = smpl_results['global_orient_world']
        transl_world = smpl_results['transl_world']
        pred_vert = smpl_results['pred_vert'].cpu().numpy()
        pred_j3dg = smpl_results['pred_j3dg']
        body_pose = smpl_results['body_pose']
        pred_shapes = smpl_results['pred_shapes']
        faces = smpl_results['faces']

    
    save_dir = SCENE_OUTPUT_DIR / tgt_name / hmr_type
    save_dir.mkdir(parents=True, exist_ok=True)
    hmr_dir = save_dir / 'hmr'
    hmr_dir.mkdir(parents=True, exist_ok=True)

    # CONTACT_IDS_SMPL = load_contact_ids_from_file()
    CONTACT_IDS_SMPL_LIST = load_contact_ids_with_mode(mode='grouped')
    
    leg_ids, hand_ids, gluteus_ids, back_ids, thigh_ids = CONTACT_IDS_SMPL_LIST
    leg_ids = leg_ids.cpu().numpy()
    hand_ids = hand_ids.cpu().numpy()
    gluteus_ids = gluteus_ids.cpu().numpy()
    back_ids = back_ids.cpu().numpy()
    thigh_ids = thigh_ids.cpu().numpy()



    PROJ_ROOT = GVHMR_ROOT
    smplx2smpl_map = torch.load(
        f"{PROJ_ROOT}/hmr4d/utils/body_model/smplx2smpl_sparse.pt",
        map_location="cpu"
    )

    if smplx2smpl_map.is_sparse:
        smplx2smpl_map = smplx2smpl_map.to_dense()

    # lookup: for each smplx vertex i, x2s[i] = closest smpl vertex
    x2s = torch.argmax(smplx2smpl_map, dim=0).cpu().numpy()   # shape (10475,)

    def convert_indices_smplx_to_smpl(smplx_indices, x2s_lookup):
        """Return SMPL indices (list of ints) corresponding to SMPL-X indices."""
        return np.unique(x2s_lookup[smplx_indices]).tolist()

    # === convert all groups to SMPL indices ===
    leg_ids_smpl     = convert_indices_smplx_to_smpl(leg_ids,     x2s)
    hand_ids_smpl    = convert_indices_smplx_to_smpl(hand_ids,    x2s)
    gluteus_ids_smpl = convert_indices_smplx_to_smpl(gluteus_ids, x2s)
    back_ids_smpl    = convert_indices_smplx_to_smpl(back_ids,    x2s)
    thigh_ids_smpl   = convert_indices_smplx_to_smpl(thigh_ids,   x2s)

    pred_contact_vert_leg     = pred_vert[:, leg_ids_smpl, :]
    pred_contact_vert_hand    = pred_vert[:, hand_ids_smpl, :]
    pred_contact_vert_gluteus = pred_vert[:, gluteus_ids_smpl, :]
    pred_contact_vert_back    = pred_vert[:, back_ids_smpl, :]
    pred_contact_vert_thigh   = pred_vert[:, thigh_ids_smpl, :]

    part_ids_list = [
        np.asarray(leg_ids_smpl, dtype=int),
        np.asarray(hand_ids_smpl, dtype=int),
        np.asarray(gluteus_ids_smpl, dtype=int),
        np.asarray(back_ids_smpl, dtype=int),
        np.asarray(thigh_ids_smpl, dtype=int),
    ]

    pred_contact_vert_list = [
        pred_contact_vert_leg,
        pred_contact_vert_hand, 
        pred_contact_vert_gluteus,
        pred_contact_vert_back,
        pred_contact_vert_thigh
    ]

    cam_c2w_np = np.asarray(npz_cam_data['cam_c2w'], dtype=np.float32)
    scale_raw = np.asarray(npz_cam_data.get('scale', 1.0), dtype=np.float32)
    if scale_raw.ndim == 0:
        scale_values = np.full((cam_c2w_np.shape[0],), float(scale_raw), dtype=np.float32)
    else:
        scale_values = scale_raw.reshape(-1).astype(np.float32)
        if scale_values.size == 1 and cam_c2w_np.shape[0] > 1:
            scale_values = np.full((cam_c2w_np.shape[0],), float(scale_values[0]), dtype=np.float32)
        elif scale_values.size != cam_c2w_np.shape[0]:
            scale_values = np.resize(scale_values, cam_c2w_np.shape[0]).astype(np.float32)

    body_part_params = {
        "leg":     {"contact_threshold": 0.3, "min_consecutive_frames": 5,  "weight": 1.0, "vel_threshold": 0.050, "min_lowvel_run": 5},
        "hand":    {"contact_threshold": 0.55, "min_consecutive_frames": 5, "weight": 0.6, "vel_threshold": 0.008, "min_lowvel_run": 4},
        "gluteus": {"contact_threshold": 0.01, "min_consecutive_frames": 20, "weight": 1.2, "vel_threshold": 0.028, "min_lowvel_run": 5},
        "back":    {"contact_threshold": 0.02, "min_consecutive_frames": 60, "weight": 0.8, "vel_threshold": 0.010, "min_lowvel_run": 5},
        "thigh":   {"contact_threshold": 0.01, "min_consecutive_frames": 5, "weight": 1.0, "vel_threshold": 0.028, "min_lowvel_run": 5},
    }

    contact_colors_rgb = [
        [0, 255,   0],  # leg - green
        [255,   0,   0],  # hand - red
        [255, 255,   0],  # gluteus - yellow
        [0,   0,   0],  # back - black
        [255,   0, 255]   # thigh - magenta
    ]

    contact_thresholds = [
        body_part_params["leg"]["contact_threshold"],
        body_part_params["hand"]["contact_threshold"],
        body_part_params["gluteus"]["contact_threshold"],
        body_part_params["back"]["contact_threshold"],
        body_part_params["thigh"]["contact_threshold"],
    ]

    camera_rotations_np = np.asarray(pred_cam[key_R], dtype=np.float32)
    camera_translations_np = np.asarray(pred_cam[key_T], dtype=np.float32)

    apply_contact_scale = False

    def _empty_contact_buffers() -> tuple[list[np.ndarray], np.ndarray]:
        empty = np.empty((0, 3), dtype=np.float32)
        return [empty.copy() for _ in part_ids_list], empty.copy()

    if CONTACT_UTILS_AVAILABLE and load_contact_points_for_parts is not None:
        try:
            contact_points_by_part, contact_points_all = load_contact_points_for_parts(
                interact_contact_path,
                part_ids_list,
                pred_vert,
                thresholds=contact_thresholds,
                max_points_per_part=12000,
                max_total_points=60000,
                camera_rotations=camera_rotations_np,
                camera_translations=camera_translations_np,
                scales=scale_values,
                apply_scale=apply_contact_scale,
            )
        except Exception as exc:
            print(f"[contact] Failed to load cached contact points ({exc}); falling back to SMPL vertices.")
            contact_points_by_part, contact_points_all = _empty_contact_buffers()
    else:
        if CONTACT_IMPORT_ERROR is not None:
            print(f"[contact] Contact utilities unavailable: {CONTACT_IMPORT_ERROR}")
        contact_points_by_part, contact_points_all = _empty_contact_buffers()

    gui_show_contact = types.SimpleNamespace(value=True)

    if not any(arr.size for arr in contact_points_by_part):
        contact_points_by_part = []
        for group in pred_contact_vert_list:
            group_np = np.asarray(group, dtype=np.float32)
            per_frame_pts = []
            for frame_idx in range(group_np.shape[0]):
                pts = group_np[frame_idx]
                rot = camera_rotations_np[frame_idx]
                trans = camera_translations_np[frame_idx]
                scale_val = float(scale_values[frame_idx]) if apply_contact_scale else 1.0
                pts_cam = (rot.T @ (pts - trans).T).T
                if apply_contact_scale:
                    pts_cam *= scale_val
                pts_world = (rot @ pts_cam.T).T + trans
                per_frame_pts.append(pts_world)
            if per_frame_pts:
                contact_points_by_part.append(
                    np.concatenate(per_frame_pts, axis=0).astype(np.float32, copy=False)
                )
            else:
                contact_points_by_part.append(np.empty((0, 3), dtype=np.float32))
        non_empty = [arr for arr in contact_points_by_part if arr.size]
        contact_points_all = (
            np.concatenate(non_empty, axis=0).astype(np.float32, copy=False)
            if non_empty else np.empty((0, 3), dtype=np.float32)
        )

    for idx_part, (pts, rgb_color) in enumerate(zip(contact_points_by_part, contact_colors_rgb)):
        if pts.size == 0:
            continue
        pts_visual = np.asarray(pts, dtype=np.float32)
        color_arr = np.tile(np.array(rgb_color, dtype=np.uint8), (pts_visual.shape[0], 1))
        handle = server.scene.add_point_cloud(
            name=f"/frames/t0/bodypart_{idx_part}",
            points=pts_visual,
            colors=color_arr,
            point_size=0.01,
            point_shape="rounded",
        )
        handle.visible = gui_show_contact.value
        contact_global_handles.append(handle)

    results = {'pred_cam': [world_cam_R, world_cam_T], # cam 
                'body_pose': body_pose, # smpl 
                'global_orient': global_orient_world, # smpl
                'betas': pred_shapes, # smpl 
                'transl': transl_world, # smpl 
                'pose2rot': False, 
                'default_smpl': True
              }

    org_vis = hmr_dir / 'org_vis'
    org_vis.mkdir(parents=True, exist_ok=True)
    
    every = 20 

    if 'pkr' or 'IMG_' in tgt_name:
      every = 1

    np.save(hmr_dir / 'hps_track.npy', results)
    human_transl_np = transl_world.detach().cpu().numpy()


    loader = viser.extras.Record3dLoader_Customized_Megasam(
        data,
        npz_cam_data, 
        conf_threshold=1.0,
        foreground_conf_threshold=foreground_conf_threshold,
        no_mask=no_mask,
        xyzw=xyzw,
        init_conf=init_conf,
    )

    sq_point_handles: list[viser.PointHandle] = []       # 新建列表
    sq_mesh_handles:  list[viser.MeshHandle]  = []       # 复用/替换你上面已有的

    point_cloud_handles_dict = {
        "filtered": [],
        "original": [],
        "mono": [],
    }


    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)


        gui_next_sqs = server.gui.add_button("Next SQs", disabled=True)
        gui_prev_sqs = server.gui.add_button("Prev SQs", disabled=True)
        gui_sqs_pt = server.gui.add_checkbox("SQs2pt", True)

        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=loader.fps
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("10", "20", "30", "60")
        )
        gui_show_all_frames = server.gui.add_checkbox("Show all frames", False)
        gui_show_human_mesh = server.gui.add_checkbox("Show human mesh", False)
        gui_stride = server.gui.add_slider(
            "Stride",
            min=1,
            max=num_frames,
            step=1,
            initial_value=1,
            disabled=True,
        )
        gui_depth_scale = server.gui.add_slider(
            "Stride",
            min=0,
            max=20,
            step=0.1,
            initial_value=1,
            disabled=True,
        )
        gui_multi_timestep_mode = server.gui.add_checkbox(
            "Show 3 timesteps", False
        )
        custom_timestep_labels = (
            "Custom timestep 1",
            "Custom timestep 2",
            "Custom timestep 3",
        )
        CUSTOM_TIMESTEP1_INTERVAL_MULTIPLIER = 3

        gui_multi_timestep_inputs = [
            server.gui.add_slider(
                label,
                min=0,
                max=max(0, num_frames - 1),
                step=1,
                initial_value=min(idx, max(0, num_frames - 1)),
                disabled=True,
            )
            for idx, label in enumerate(custom_timestep_labels)
        ]





    # Add GUI controls
    with server.gui.add_folder("Layers"):
        gui_show_contact = server.gui.add_checkbox("Show Contact", True)
        gui_show_sqs = server.gui.add_checkbox("SQs", False)
        gui_show_scene_raw = server.gui.add_checkbox("Scene mesh (raw)", True)
        gui_show_scene_coacd_contact = server.gui.add_checkbox("Scene mesh (contact-COACD)", True)
        gui_show_scene_coacd = server.gui.add_checkbox("Scene mesh (COACD)", True)
        gui_show_planes = server.gui.add_checkbox("Planar primitives", True)
        gui_show_points = server.gui.add_checkbox("All points", True)
        gui_show_humans = server.gui.add_checkbox("HMR", True)
        gui_show_camera_frusta = server.gui.add_checkbox("Camera frusta", show_camera_frusta)
        
        # Add point cloud display options
        gui_pc_display_mode = server.gui.add_dropdown(
            "Point Cloud Display",
            options=["Filtered", "Original", "Mono"],
            initial_value="Filtered"
        )

        # ── SQ / PointCloud 显示模式 ──────────────────────────────────────────
        gui_sq_mode = server.gui.add_dropdown(
            "SQ Display Mode",
        options=["SQs + Points", "Points Only"],
        initial_value="SQs + Points",
    )


    camera_frustum_handles: list[Optional[viser.CameraFrustumHandle]] = []
    camera_frustum_axes_handles: list[Optional[viser.FrameHandle]] = []
    current_active_frame_indices: set[int] = set()

    def _set_camera_frusta_visibility_locked() -> None:
        base_active_set = current_active_frame_indices or set(_get_active_frame_indices())
        camera_active_set = set(base_active_set)
        if gui_multi_timestep_mode.value:
            total = _available_frame_count()
            if total > 0:
                max_idx = total - 1
                shift = _custom_timestep1_shift()
                camera_active_set.clear()
                seen: set[int] = set()
                for slider_idx, slider in enumerate(gui_multi_timestep_inputs):
                    idx = int(round(slider.value))
                    idx = max(0, min(idx, max_idx))
                    if slider_idx == 0:
                        idx = max(0, min(idx + shift, max_idx))
                    if idx not in seen:
                        seen.add(idx)
                        camera_active_set.add(idx)
        show_points_enabled = gui_show_points.value
        for idx, frustum_handle in enumerate(camera_frustum_handles):
            if frustum_handle is None:
                continue
            frame_selected = idx in camera_active_set
            visible = (
                gui_show_camera_frusta.value
                and show_points_enabled
                and frame_selected
            )
            frustum_handle.visible = visible
            axes_handle = camera_frustum_axes_handles[idx]
            if axes_handle is not None:
                axes_handle.visible = visible

    def update_camera_frusta_visibility() -> None:
        if not camera_frustum_handles:
            return
        with server.atomic():
            _set_camera_frusta_visibility_locked()

    def _available_frame_count() -> int:
        counts = [
            len(frame_nodes),
            len(point_cloud_handles_dict.get("filtered", [])),
            len(point_cloud_handles_dict.get("original", [])),
            len(point_cloud_handles_dict.get("mono", [])),
            len(human_mesh_handles),
        ]
        return max(counts) if counts else 0

    def _custom_timestep1_shift() -> int:
        try:
            base_interval = max(1, int(interval))
        except (NameError, TypeError, ValueError):
            return 0
        return CUSTOM_TIMESTEP1_INTERVAL_MULTIPLIER #* base_interval

    def _get_active_frame_indices() -> list[int]:
        total = _available_frame_count()
        if total == 0:
            return []
        max_idx = total - 1
        if gui_multi_timestep_mode.value:
            seen: set[int] = set()
            ordered_indices: list[int] = []
            for slider_idx, slider in enumerate(gui_multi_timestep_inputs):
                idx = int(round(slider.value))
                idx = max(0, min(idx, max_idx))
                if idx not in seen:
                    seen.add(idx)
                    ordered_indices.append(idx)
            return ordered_indices
        if gui_show_all_frames.value:
            stride = max(1, int(gui_stride.value))
            return list(range(0, total, stride))
        current_idx = int(min(gui_timestep.value, max_idx))
        return [current_idx]

    def update_time_dependent_handles() -> None:
        active_frames = _get_active_frame_indices()
        active_set = set(active_frames)
        current_active_frame_indices.clear()
        current_active_frame_indices.update(active_set)
        mode = gui_pc_display_mode.value.lower()
        show_points = gui_show_points.value
        show_humans = gui_show_humans.value
        show_contact = gui_show_contact.value
        multi_timestep_enabled = gui_multi_timestep_mode.value

        def _interval_value() -> int:
            try:
                return max(1, int(interval))
            except (NameError, TypeError, ValueError):
                return 1

        with server.atomic():
            for idx, frame_node in enumerate(frame_nodes):
                if frame_node is None:
                    continue
                if multi_timestep_enabled:
                    frame_node.visible = show_points
                else:
                    frame_node.visible = show_points and (idx in active_set)

            for pc_type, handles in point_cloud_handles_dict.items():
                type_selected = show_points and (pc_type == mode)
                for idx, handle in enumerate(handles):
                    if handle is None:
                        continue
                    if multi_timestep_enabled:
                        handle.visible = type_selected  # show full background when syncing multiple frames
                    else:
                        handle.visible = type_selected and (idx in active_set)

            for idx, handle in enumerate(human_mesh_handles):
                handle.visible = show_humans and (idx in active_set)

            if show_contact:
                if gui_show_all_frames.value:
                    for handle in contact_global_handles:
                        handle.visible = False
                    for handle in contact_vertices_handles:
                        handle.visible = False
                else:
                    for handle in contact_global_handles:
                        handle.visible = True
                    for handle in contact_vertices_handles:
                        handle.visible = False
                    contact_interval = _interval_value()
                    contact_indices = {idx // contact_interval for idx in active_set}
                    for c_idx in contact_indices:
                        if 0 <= c_idx < len(contact_vertices_handles):
                            contact_vertices_handles[c_idx].visible = True
            else:
                for handle in contact_global_handles:
                    handle.visible = False
                for handle in contact_vertices_handles:
                    handle.visible = False
        update_camera_frusta_visibility()

    def update_playback_widgets() -> None:
        playing_disabled = gui_show_all_frames.value or gui_multi_timestep_mode.value
        gui_playing.disabled = playing_disabled
        if playing_disabled and gui_playing.value:
            gui_playing.value = False
        controls_disabled = (
            gui_playing.value
            or gui_show_all_frames.value
            or gui_multi_timestep_mode.value
        )
        gui_timestep.disabled = controls_disabled
        gui_next_frame.disabled = controls_disabled
        gui_prev_frame.disabled = controls_disabled

    update_playback_widgets()

    # Visibility callbacks
    @gui_show_sqs.on_update
    def _(_):
        with server.atomic():
            for s in sq_mesh_handles:
                s.visible = gui_show_sqs.value


    # ── 统一切换 SQ 网格和“按 SQ 划分的点云” ─────────────────────────────
    '''@gui_sq_mode.on_update
    def _(_):
        show_sq = (gui_sq_mode.value == "SQs + Points")
        with server.atomic():
            # SQ mesh（红色几何体）
            for mh in sq_mesh_handles:
                mh.visible = show_sq and gui_show_sqs.value   # 仍受原“SQs”总开关控制
            # 属于每个 SQ 的点云（青色）
            for ph in sq_point_handles:
                ph.visible = gui_show_points.value            # 点云始终跟 All points 开关'''


    @gui_show_scene_raw.on_update
    def _(_):
        if scene_mesh_raw_handle is not None:
            scene_mesh_raw_handle.visible = gui_show_scene_raw.value

    @gui_show_scene_coacd.on_update
    def _(_):
        if scene_mesh_coacd_handle is not None:
            scene_mesh_coacd_handle.visible = gui_show_scene_coacd.value

    @gui_show_scene_coacd_contact.on_update
    def _(_):
        if scene_mesh_coacd_contact_handle is not None:
            scene_mesh_coacd_contact_handle.visible = gui_show_scene_coacd_contact.value

    @gui_show_humans.on_update
    def _(_):
        update_time_dependent_handles()

    @gui_show_contact.on_update
    def _(_):
        update_time_dependent_handles()

    @gui_show_planes.on_update
    def _(_):
        with server.atomic():
            for h in plane_mesh_handles:
                h.visible = gui_show_planes.value
        update_camera_frusta_visibility()

    @gui_show_points.on_update
    def _(_):
        with server.atomic():
            for ph in sq_point_handles:
                ph.visible = False  # gui_show_points.value
        update_time_dependent_handles()

    @gui_show_camera_frusta.on_update
    def _(_):
        update_camera_frusta_visibility()


    # Point cloud display mode callback
    @gui_pc_display_mode.on_update
    def _(_):
        update_time_dependent_handles()






    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        update_playback_widgets()

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        prev_timestep = gui_timestep.value
        if not gui_show_all_frames.value and not gui_multi_timestep_mode.value:
            update_time_dependent_handles()
            server.flush()
        
    @gui_show_all_frames.on_update
    def _(_) -> None:
        gui_stride.disabled = not gui_show_all_frames.value
        if gui_show_all_frames.value and gui_multi_timestep_mode.value:
            gui_multi_timestep_mode.value = False
        update_playback_widgets()
        update_time_dependent_handles()

    def sqs_params_2_mesh(sqs, lat_res=32, lon_res=64, combine=True, device='cuda'):
        """
        Convert superquadric parameters to a TriMesh using the provided utility functions.
        
        Args:
            sqs: numpy array of shape (N, 11) where each row contains:
                [eps1, eps2, scale_x, scale_y, scale_z, euler_z, euler_y, euler_x, tx, ty, tz]
            lat_res: latitude resolution for mesh generation (default: 32)
            lon_res: longitude resolution for mesh generation (default: 64)
            combine: if True, combine all superquadrics into a single mesh
                    if False, return a list of individual meshes
            device: torch device to use for computation
        
        Returns:
            trimesh.Trimesh object (if combine=True) or list of trimesh.Trimesh objects
        """
        
        if isinstance(sqs, np.ndarray):
            sqs = torch.from_numpy(sqs).float()
        
        if sqs.shape[0] == 0:
            # Return an empty container matching the requested output format
            return [] if not combine else trimesh.Trimesh(vertices=np.array([]), faces=np.array([]))
        
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        sqs = sqs.to(device)
        
        meshes = []
        
        # Get base icosphere for mesh generation
        base_mesh = get_icosphere(level=2)  # Higher level for better resolution
        base_verts, base_faces = base_mesh.get_mesh_verts_faces(0)
        base_verts = base_verts.to(device)
        base_faces = base_faces.to(device)
        
        # Convert vertices to spherical coordinates
        eta = torch.asin(base_verts[..., 1].clamp(-1, 1))
        omega = torch.atan2(base_verts[..., 0], base_verts[..., 2])
        
        for i in range(sqs.shape[0]):
            try:
                # Extract parameters
                eps1 = sqs[i, 0].clamp(0.1, 2.0)  # Clamp to valid range
                eps2 = sqs[i, 1].clamp(0.1, 2.0)
                scale = sqs[i, 2:5]
                euler = sqs[i, 5:8]
                translation = sqs[i, 8:11]
                
                # Generate superquadric vertices using parametric equation
                verts = parametric_sq(eta, omega, eps1.unsqueeze(0), eps2.unsqueeze(0))
                verts = verts.squeeze(0)  # Remove batch dimension
                
                # Apply scale
                verts = verts * scale
                
                # Apply rotation (Euler ZYX convention)
                R = euler_angles_to_matrix(euler.unsqueeze(0), convention="ZYX").squeeze(0)
                verts = verts @ R.T
                
                # Apply translation
                verts = verts + translation
                
                # Convert to numpy for trimesh
                verts_np = verts.cpu().numpy()
                faces_np = base_faces.cpu().numpy()
                
                # Create trimesh object
                mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
                
                # Fix normals to point outward
                mesh.fix_normals()
                
                # Validate mesh
                if len(mesh.vertices) > 0 and len(mesh.faces) > 0:
                    meshes.append(mesh)
                
            except Exception as e:
                print(f"Warning: Failed to create superquadric {i}: {e}")
                continue
        
        if len(meshes) == 0:
            # Return an empty container matching the requested output format
            return [] if not combine else trimesh.Trimesh(vertices=np.array([]), faces=np.array([]))
        
        if combine:
            # Combine all meshes into a single scene mesh
            combined_mesh = trimesh.util.concatenate(meshes)
            # Clean up the combined mesh
            combined_mesh.remove_duplicate_faces()
            # combined_mesh.remove_degenerate_faces()
            return combined_mesh
        else:
            # Return list of individual meshes
            return meshes




    # Update frame visibility when the stride changes.
    @gui_stride.on_update
    def _(_) -> None:
        if gui_show_all_frames.value:
            update_time_dependent_handles()

    @gui_multi_timestep_mode.on_update
    def _(_) -> None:
        enabled = gui_multi_timestep_mode.value
        for slider in gui_multi_timestep_inputs:
            slider.disabled = not enabled
        gui_show_all_frames.disabled = enabled
        if enabled and gui_show_all_frames.value:
            gui_show_all_frames.value = False
        if enabled:
            gui_playing.value = False
        update_playback_widgets()
        update_time_dependent_handles()

    for _slider in gui_multi_timestep_inputs:
        @_slider.on_update
        def _(_) -> None:
            if gui_multi_timestep_mode.value:
                update_time_dependent_handles()



    # Load in frames.ss
    server.scene.add_frame(
        "/frames",
        wxyz=tf.SO3.exp(onp.array([onp.pi / 2.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )


    server.scene.add_frame(
        "/frames",
        wxyz=tf.SO3.exp(onp.array([0.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )
    
    R_ROOT = tf.SO3.exp(onp.array([onp.pi / 2.0, 0.0, 0.0])).as_matrix()  # 3×3
    T_ROOT4 = np.eye(4, dtype=float)                                      # 4×4 齐次
    #T_ROOT4[:3, :3] = R_ROOT

    frame_nodes: list[viser.FrameHandle] = []
    all_positions = []
    all_colors = []

    mono_pcs = []
    bg_org_positions=[]
    bg_org_colors = []

    bg_positions = []
    bg_colors = []
    positions=[]
    colors=[]


    vdb_volume = vdbfusion.VDBVolume(voxel_size=0.05,
                        sdf_trunc=0.15, space_carving=True)
    if num_frames > 200:
        stttride = 3
    elif num_frames >=100:
        stttride = 2
        #  // 2
    stttride= (num_frames // 90) + 1
    

    #stttride = 1
    inspect_everything = True

    moge_mesh=False
    TTTTTTEST=False
    show_moge_priors = False
    mono_cleaner = False


    plane_accum = {
        'S_items': [],
        'R_items': [],
        'T_items': [],
        'color_items': [],
        'pts_items': [],
    }
    num_projections = 10
    source_idx = 0



    rgbs = []
    normals = []
    mono_normals = []
    existing_sqs = []
    exisitng_points = []
    sqs_mesh_handles: list[viser.MeshHandle] = []  
    mono_pc = None
    depths = []
    pointclouds_hmr_body = []
    rotations = []
    translations = []
    mono_pc_clrs = []
    per_frame_sq = []
    per_sq_one_list = []

    points_bg_map_nksr = []
    points_normal_nksr = []
    
    all_frame_mode = False
    results = {
        'S_items': [],
        'R_items': [],
        'T_items': [],
        'pts_items': []
    }

    try:
        single_image = (int(tgt_name[0]))
        single_image = False

    except (ValueError, IndexError):

        single_image = True
        if 'cam' in tgt_name:
            single_image=False

    # single_image = False
    # single_image = bool(int(tgt_name[0]))

    USE_CONTACT = False
    if 'qitao' in tgt_name or 'pk' in tgt_name or 'IMG' in tgt_name:
      single_image = False 
      # USE_CONTACT=False

    if 'TEST' in tgt_name:
      single_image = False 
    single_image = False
    contact_pipeline_available = CAN_RUN_CONTACT_ANALYSIS
    if USE_CONTACT and not contact_pipeline_available:
        reason = CONTACT_IMPORT_ERROR or "missing scontact_utils"
        print(f"[contact] Contact analysis requested but utilities are unavailable ({reason}); skipping contact refinement.")

    #[depth, rotation, translation, K]
    print(single_image, 'singleimg')
    debug = False 
    if USE_CONTACT and contact_pipeline_available: 
        if not transfer_data:
          
            contacted_masks, static_frames, static_segments, best_frames_global, counts_global, per_part = analyze_contacts_5parts(
                interact_contact_path=interact_contact_path,
                num_frames=num_frames,
                part_ids_list=[leg_ids_smpl, hand_ids_smpl, gluteus_ids_smpl, back_ids_smpl, thigh_ids_smpl],
                pred_contact_vert_list=[
                    pred_vert[:, leg_ids_smpl, :],
                    pred_vert[:, hand_ids_smpl, :],
                    pred_vert[:, gluteus_ids_smpl, :],
                    pred_vert[:, back_ids_smpl, :],
                    pred_vert[:, thigh_ids_smpl, :],
                ],
                body_part_params=body_part_params,
                total_verts=6890,
                pred_vert_global=pred_vert,
                min_static_duration=15,
                debug_dir=str(SCENE_OUTPUT_DIR / tgt_name / 'contact_debug')
            )

    if TTTTTTEST == False:
        interval = 7# 30#stttride=  (num_frames // 90) + 1
        frame_indices = []  # Track which frames we're processing
        if debug:
          num_frames = interval+1

        if 'pkr'  or 'qitao'  in tgt_name: # 
            interval = 1
        if 'qitao'  in tgt_name: # 
        # viser_m/visualizer_megasam.py
            interval = 6

        interval = 7

        print(interval, 'interval')

        times_list = list(range(0, num_frames, interval))
        num_processed_frames = len(times_list)
        for i in tqdm(times_list):
        # for i in tqdm(range(0, 1)):
            frame = loader.get_frame(i)
            frame_indices.append(i)
            mono_pc = None
            
            if single_image == True:
                seg_network = Vis()
                parent_folder = str(VSLAM_MEGACAM_DIR / tgt_name)
                if len(mono_normals)==0:
                    mono_pc, mono_normal, rgb, colors__, extras = frame.get_mono_data(
                        i,
                        seg_network,
                        parent_folder,
                        single_image=single_image,
                    )

                    depth, rotation, translation, K, points_bg_map, plane_info = extras
                    distance_filtering=True
                    # real_normal
                    if distance_filtering:
                      points_bg_map_, _, depth, mono_normal_ = filter_bg_points_by_human_distance(points_bg_map, points_bg_map, human_transl_np, depth, mono_normal, max_dist=2.7)
                    else:
                      _, _, depth, _ = filter_bg_points_by_human_distance(points_bg_map, points_bg_map, human_transl_np, depth, max_dist=1)

                    bg_position, bg_color = mono_pc, colors__
                    _, T_world_camera, _ = frame.get_sdf(bg_position, downsample_factor, bg_downsample_factor, vdb_volume=vdb_volume)
                    vdb_volume.integrate(points_bg_map_.astype(np.float64), T_world_camera)


                    points_bg_map_nksr.append(points_bg_map_)
                    points_normal_nksr.append(mono_normal_)

                    bg_color=bg_color.reshape(-1, 3)
                    mono_normals.append(mono_normal)
                    rgbs.append(rgb)
                    bg_positions.append(mono_pc)  # Store mono point cloud
                    bg_colors.append(rgb)
                    depths.append(depth)
                    bg_org_positions = bg_positions
                    bg_org_colors = bg_colors
                    rotations.append(rotation)
                    translations.append(translation)
                    all_positions.append(points_bg_map)

                    if plane_info:
                        primitives = plane_info.get('primitives', {})
                        S_items = primitives.get('S_items', [])
                        R_items = primitives.get('R_items', [])
                        T_items = primitives.get('T_items', [])
                        color_items = primitives.get('color_items', [])
                        pts_items = primitives.get('pts_items', [])
                        for S_item, R_item, T_item, color_item, pts_item in zip(
                            S_items, R_items, T_items, color_items, pts_items
                        ):
                            S_dev = S_item.to(device) if hasattr(S_item, 'to') else torch.as_tensor(S_item, device=device)
                            R_dev = R_item.to(device) if hasattr(R_item, 'to') else torch.as_tensor(R_item, device=device)
                            T_dev = T_item.to(device) if hasattr(T_item, 'to') else torch.as_tensor(T_item, device=device)
                            C_dev = color_item.to(device) if hasattr(color_item, 'to') else torch.as_tensor(color_item, device=device)
                            if hasattr(pts_item, 'to'):
                                P_dev = pts_item.to(device)
                            else:
                                P_dev = torch.as_tensor(pts_item, device=device, dtype=torch.float32)
                            plane_accum['S_items'].append(S_dev)
                            plane_accum['R_items'].append(R_dev)
                            plane_accum['T_items'].append(T_dev)
                            plane_accum['color_items'].append(C_dev)
                            plane_accum['pts_items'].append(P_dev)

            else:

                # Get original point cloud data
                # human_transl_np
                output_pts, real_normal, extras = frame.get_point_cloud(downsample_factor, bg_downsample_factor=1)
                position, color, bg_position, bg_color, po_all, clr_all, obj_pos, obj_clr = output_pts
                
                # Store original background positions and colors
                bg_org_positions.append(bg_position.copy())
                bg_org_colors.append(bg_color.copy())
                '''if mono_cleaner:
                    output_pts, real_normal, extras = frame.get_filtered_point_cloud(downsample_factor, bg_downsample_factor=1, mono_normal=mono_normal)
                else:
                    _, _, extras = frame.get_filtered_point_cloud(downsample_factor, bg_downsample_factor=1)
                
                '''
                depth, rotation, translation, K, points_bg_map = extras
                distance_filtering=True
                if distance_filtering:
                  points_bg_map_, _, depth, real_normal_ = filter_bg_points_by_human_distance(points_bg_map, points_bg_map, human_transl_np, depth, real_normal, max_dist=2.7)
                else:
                  _, _, depth, _ = filter_bg_points_by_human_distance(points_bg_map, points_bg_map, human_transl_np, depth, max_dist=1)
                points_bg_map_nksr.append(points_bg_map_)
                points_normal_nksr.append(real_normal_)

                
                position, color, bg_position, bg_color, _, clr_all, obj_pos, obj_clr = output_pts
                if do_mesh:
                    faces, vertices, vertex_colors, vertex_uvs, tri = frame.get_meshes(po_all, downsample_factor, bg_downsample_factor)
                    _, T_world_camera, _= frame.get_sdf(bg_position, downsample_factor, bg_downsample_factor, vdb_volume=vdb_volume)
                vdb_volume.integrate(points_bg_map_.astype(np.float64), T_world_camera)

                mono_normals.append(real_normal)
                depths.append(depth)
                rotations.append(rotation)
                translations.append(translation)
                bg_positions.append(bg_position)
                bg_colors.append(bg_color)
                positions.append(position)
                colors.append(color)
                all_positions.append(points_bg_map)
                all_colors.append(clr_all)

                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]         # NOT K[0,1]
                cy = K[1, 2]         # NOT K[1,-1]

            inferring_scene=False
            if inferring_scene:
                for pred_contact_vert_g, contact_rgb in zip(pred_contact_vert_list, contact_colors_rgb):
                    pred_contact_vert_t = pred_contact_vert_g[i]  # shape: (num_contact_verts, 3)
                    
                    # Find bg_position points within small range of contact vertices
                    threshold = 0.05  # adjust this threshold as needed
                    
                    # Compute distances between all bg_position points and contact vertices
                    distances = np.linalg.norm(bg_position[:, None, :] - pred_contact_vert_t[None, :, :], axis=2)
                    
                    # Find bg_position points that are within threshold of any contact vertex
                    min_distances = np.min(distances, axis=1)  # shape: (N,)
                    nearby_mask = min_distances < threshold
                    
                    if np.any(nearby_mask):
                        # Get the nearby bg_position points
                        nearby_points = bg_position[nearby_mask]
                        
                        # Create colors array for these points (all same color for this contact group)
                        num_nearby = nearby_points.shape[0]
                        colors_contact = np.tile(contact_rgb, (num_nearby, 1))  # shape: (num_nearby, 3)
                        

                        smplllll=False
                        if smplllll:
                            contact_vertices = server.scene.add_point_cloud(
                                name=f"/frames/t{i}/point_cloud_contact_{len(contact_vertices_handles)}",
                                points=nearby_points,
                                colors=colors_contact,
                                point_size=0.02,
                                point_shape="rounded",
                            )
                            contact_vertices_handles.append(contact_vertices)
                            pointclouds_hmr_body.append(pred_contact_vert_t)
                            contact_vertices = server.scene.add_point_cloud(
                                name=f"/frames/t{i}/point_cloud_contact_{len(contact_vertices_handles)}",
                                points=pred_contact_vert_t,
                                colors=np.tile(contact_rgb, (len(pred_contact_vert_t), 1)),
                                point_size=0.02,
                                point_shape="rounded",
                            )
                            contact_vertices_handles.append(contact_vertices)


            if not transfer_data:
                '''
                # conda install -y -c conda-forge gcc_linux-64=12.4.0 gxx_linux-64=12.4.0
                leg_ids_smpl     = convert_indices_smplx_to_smpl(leg_ids,     x2s)
                hand_ids_smpl    = convert_indices_smplx_to_smpl(hand_ids,    x2s)
                gluteus_ids_smpl = convert_indices_smplx_to_smpl(gluteus_ids, x2s)
                back_ids_smpl    = convert_indices_smplx_to_smpl(back_ids,    x2s)
                thigh_ids_smpl   = convert_indices_smplx_to_smpl(thigh_ids,   x2s)

                pred_contact_vert_leg     = pred_vert[:, leg_ids_smpl, :]
                pred_contact_vert_hand    = pred_vert[:, hand_ids_smpl, :]
                pred_contact_vert_gluteus = pred_vert[:, gluteus_ids_smpl, :]
                pred_contact_vert_back    = pred_vert[:, back_ids_smpl, :]
                pred_contact_vert_thigh   = pred_vert[:, thigh_ids_smpl, :]


                pred_contact_vert_list = [
                    pred_contact_vert_leg,
                    pred_contact_vert_hand, 
                    pred_contact_vert_gluteus,
                    pred_contact_vert_back,
                    pred_contact_vert_thigh
                ]

                contact_colors_rgb = [
                    [0, 255,   0],  # leg - green
                    [255,   0,   0],  # hand - red
                    [255, 255,   0],  # gluteus - yellow
                    [0,   0,   0],  # back - black
                    [255,   0, 255]   # thigh - magenta
                ]
                '''
                # interact_contact_ti = np.load(os.path.join(interact_contact_path, f'{i:05d}.npz'))['pred_contact_3d_smplh']
                # contacted_mask = interact_contact_ti > 1e-7
                frame_handle = server.scene.add_frame(f"/frames/t{i}", show_axes=False)
                frame_nodes.append(frame_handle)

                frustum_handle: Optional[viser.CameraFrustumHandle] = None
                frustum_axes_handle: Optional[viser.FrameHandle] = None
                frame_rgb = getattr(frame, "rgb", None)
                frame_K = getattr(frame, "K", None)
                frame_pose = getattr(frame, "T_world_camera", None)
                if frame_rgb is not None and frame_K is not None and frame_pose is not None:
                    height = frame_rgb.shape[0] if frame_rgb.ndim >= 2 else 0
                    width = frame_rgb.shape[1] if frame_rgb.ndim >= 2 else 0
                    fx = frame_K[0, 0]
                    norm_idx = (
                        len(camera_frustum_handles)
                        / max(1, num_processed_frames - 1)
                    )
                    color_rgba = cm.viridis(norm_idx)
                    frustum_color = tuple(int(c * 255) for c in color_rgba[:3])
                    fov = 2 * onp.arctan2(max(height, 1) / 2.0, fx)
                    aspect = (width / height) if height > 0 else 1.0
                    frustum_handle = server.scene.add_camera_frustum(
                        f"/frames/t{i}/frustum",
                        fov=fov,
                        aspect=aspect,
                        scale=camera_frustum_scale*3,
                        image=frame_rgb[::downsample_factor, ::downsample_factor],
                        wxyz=tf.SO3.from_matrix(frame_pose[:3, :3]).wxyz,
                        position=frame_pose[:3, 3],
                        color=frustum_color,
                        visible=False,
                    )
                    frustum_axes_handle = server.scene.add_frame(
                        f"/frames/t{i}/frustum/axes",
                        axes_length=camera_frustum_scale * axes_scale * 10*3,
                        axes_radius=camera_frustum_scale * axes_scale*3,
                        visible=False,
                    )

                camera_frustum_handles.append(frustum_handle)
                camera_frustum_axes_handles.append(frustum_axes_handle)

                if CAN_EXTRACT_SINGLE_FRAME_CONTACTS:
                    target_frame = num_frames - 1  # Example: 15 frames before the end
                    try:
                        frame_idx, contact_points, per_part_contacts = extract_contact_points_from_single_frame(
                            frame_idx=target_frame,
                            pred_vert=pred_vert,
                            part_ids_list=part_ids_list,
                            interact_contact_path=interact_contact_path,
                            contact_threshold=0.5,
                            use_all_parts=True,
                        )
                    except Exception as exc:
                        print(f"[contact] Failed to extract frame contacts ({exc}).")
                        contact_points = None
                        per_part_contacts = None
                else:
                    contact_points = None
                    per_part_contacts = None
                
                # Option 2: Simple sampling without contact predictions
                # frame_idx, contact_points = extract_contact_points_from_frame_simple(
                #     frame_idx=target_frame,
                #     pred_vert=pred_vert,
                #     part_ids_list=part_ids_list,
                #     num_points_per_part=30,
                #     use_velocity=True
                # )
                
                # Visualize
                if False and contact_points.shape[0] > 0:
                    contact_pc_filtered = server.scene.add_point_cloud(
                        name=f"/frames/t{frame_idx}/point_cloud_contact_single_frame",
                        points=contact_points,
                        colors=[0, 255, 0],
                        point_size=0.028,
                        point_shape="rounded",
                    )
                    cache_path = CONTACT_CACHE_DIR / f"{tgt_name}_frame{frame_idx}_contacts.ply"
                    # cache_path.parent.mkdir(parents=True, exist_ok=True)
                    # save_point_cloud_ply(
                    #     path=str(cache_path),
                    #     points=contact_points,
                    #     ascii=False,
                    # )
                    
                    print(f"Extracted {contact_points.shape[0]} contact points from frame {frame_idx}")



                mesh_color = [152, 251, 152]  # lighter pastel green for HMR meshes
                mesh_opacity = 1.0
                human_mesh_handle = server.scene.add_mesh_simple(
                    name=f"/frames/t{i}/human_mesh",
                    vertices=pred_vert[[i]],
                    faces=faces,
                    flat_shading=False,
                    wireframe=False,
                    color=mesh_color,
                    opacity=mesh_opacity,
                )

                human_mesh_handle.visible = len(human_mesh_handles) == 0
                human_mesh_handles.append(human_mesh_handle)



            
            pc_filtered = server.scene.add_point_cloud(
                name=f"/frames/t{i}/point_cloud_filtered",
                points=bg_position,
                colors=bg_color,
                point_size=point_size,
                point_shape="rounded",
            )
            if single_image:
              # break
              save_point_cloud_ply(
                path=str(SCENE_OUTPUT_DIR / tgt_name / 'points.ply'),
                points=bg_position,
                colors=bg_color,      
                ascii=False           # 大点云推荐 False（二进制）
               ) 

            point_cloud_handles_dict['filtered'].append(pc_filtered)
            


            if single_image:
                break 

        update_time_dependent_handles()

        bg_positions = onp.concatenate(bg_positions, axis=0)
        bg_colors = onp.concatenate(bg_colors, axis=0)

        depthmaps_np = np.array(depths)
        if static_camera and depthmaps_np.size > 0:
            valid_mask = np.isfinite(depthmaps_np) & (depthmaps_np > 0)
            summed = np.where(valid_mask, depthmaps_np, 0.0).sum(axis=0)
            counts = valid_mask.sum(axis=0)
            avg_depth = np.zeros_like(depthmaps_np[0])
            np.divide(summed, np.maximum(counts, 1), out=avg_depth, where=counts > 0)
            # Reuse the averaged depth map for every frame so downstream code sees consistent shapes.
            depthmaps_np = np.broadcast_to(avg_depth, depthmaps_np.shape).copy()
        R_cam = np.array(rotations)
        T_cam = np.array(translations)

        depthmaps = torch.tensor(depthmaps_np, device=device, dtype=torch.float32)
        R_cam = torch.tensor(np.array(rotations), device=device, dtype=torch.float32)
        T_cam = torch.tensor(np.array(translations), device=device, dtype=torch.float32)
        mono_normals = torch.tensor(np.array(mono_normals), device=device, dtype=torch.float32)
        pointclouds = torch.tensor(np.array(all_positions), device=device, dtype=torch.float32)



        '''if not transfer_data:
            hmr_tensors = [torch.tensor(pc, device=device, dtype=torch.float32) for pc in pointclouds_hmr_body]
            pointclouds_hmr_body = torch.cat(hmr_tensors, dim=0)
            pointclouds = torch.cat([pointclouds, pointclouds_hmr_body], dim=0)'''



        def save_bg_normals(tgt_name: str,
                            bg_positions,
                            normals,
                            out_dir: str = "cache",
                            fmt: str = "pt") -> str:
            """
            """
            out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

            # 保证是 float32 Tensor（无论传进来是 list/np/torch）
            bg_positions = torch.as_tensor(bg_positions, dtype=torch.float32)
            normals      = torch.as_tensor(normals,      dtype=torch.float32)

            if fmt == "pt":
                path = out / f"{tgt_name}.pt"
                torch.save({"bg_positions": bg_positions.cpu(),
                            "normals": normals.cpu()}, path)
            elif fmt == "npz":
                path = out / f"{tgt_name}.npz"
                np.savez_compressed(path,
                                    bg_positions=bg_positions.cpu().numpy(),
                                    normals=normals.cpu().numpy())
            else:
                raise ValueError("fmt must be 'pt' or 'npz'")
            return str(path)

        try:
          normals_tensor_nksr = torch.from_numpy(np.concatenate(points_normal_nksr, axis=0))
          pointclouds_tensor_nksr = torch.from_numpy(np.concatenate(points_bg_map_nksr, axis=0))

          save_bg_normals(tgt_name, pointclouds_tensor_nksr, normals_tensor_nksr, fmt="pt")
        except:
          fasfas =2 


        # Handle mono_normals conversion properly
        if isinstance(mono_normals, list):
            if isinstance(mono_normals[0], torch.Tensor):
                mono_normals = torch.stack(mono_normals).to(device)
            else:
                mono_normals = torch.tensor(np.array(mono_normals), device=device, dtype=torch.float32)
        else:
            mono_normals = mono_normals.to(device)
        

    root = FLOWS_COVISIBILITY_DIR
    data_dir = root / Path(tgt_name)
    debug_dir = Path('./')
    REPO_ROOT = Path('./')
    if save_clustering:
        cluster_dump_dir = (REPO_ROOT / "vis" / tgt_name / hmr_type)
        cluster_dump_dir.mkdir(parents=True, exist_ok=True)

        all_frame_mode = False


        try:
          a = contact_points
        except: 
          contact_points = None 

        print(f"[segment] Running '{segment_mode}' segmentation pipeline")
        if segment_mode == "cluster_3d":
            results = cluster_pointcloud_pipeline(
                pointclouds=pointclouds,
                normals=mono_normals,
                device=device,
                min_cluster_points=200 if detailed_planes else 400,
                coarse_clusters=160,
                fine_eps=0.2,
                normal_weight=0.3,
                roundness_threshold=0.6,
                planar_threshold=0.12,
            )
        else:
            results = interval_flow_segmentation_pipeline_with_vis(
                mono_normals=mono_normals,
                depthmaps=depthmaps,
                pointclouds=pointclouds, ### 
                data_dir=data_dir,
                frame_indices=frame_indices,
                interval=interval,
                device=device,
                save_debug=False,
                debug_dir=debug_dir,
                contact_points=None,
                stat_cam=(static_camera or single_image),
                detailed_planes=detailed_planes,
                cluster_dump_dir=cluster_dump_dir,
            )

        '''
        results = add_scene_grounded_per_part_planes(
            results=results,
            per_part_contacts=per_part_contacts,   # dict: part name -> (Ni,3) float points
            fixed_thickness=0.05,                  # meters (half-size along normal = 0.015)
            device=device
        )
        
        results = regroup_extend_trusted_planes(
              results,
              all_points=pointclouds,     # (T,H,W,3) or (N,3), world-space
              all_normals=mono_normals,   # (T,H,W,3) or (N,3), world-space unit normals
              depthmaps=depthmaps,        # (T,H,W) if you passed (T,H,W,3) above
              plane_thickness_max=0.05,   # treat thinner than 5 cm as planar seed
              normal_cluster_deg=8.0,     # cluster normals within 8°
              trust_area_ratio=0.65,      # keep clusters covering ~65% of plane area
              offset_split_tol=0.05,      # split parallel planes ≥5 cm apart
              dist_thresh=0.05,           # accept points within 5 cm to the plane
              normal_align_deg=12.0,      # accept normals within 12°
              lateral_margin=0.25,        # grow rectangle by ±25 cm on plane
              beta_normal=0.2,            # 20% freedom away from trusted normal
              max_dev_deg=6.0,            # but clamp deviation to ≤6°
              max_iter=2                  # run two regroup+refit passes
          )'''
        #### in results:: S is log(), 
  
        ###


        ### before there are exp to use 
        # convert_results_to_params_direct
        ### 
        max_iter = 1 if transfer_data else int(4e0)
      
        sq_loss_threshold = None #1e-2 # if transfer_data else 5e-4
        params = refine_sq_with_chamfer(
            results,
            lr=0,
            mesh_level=3,
            max_iter=max_iter,
            device="cuda",
            loss_threshold=sq_loss_threshold,
        )
            

        params_np = params.detach().cpu().numpy()              
        per_sq_one_list = sqs_params_2_mesh(params, combine=False)
        per_sq_one_list = [
            _sanitize_trimesh(m) for m in per_sq_one_list
            if len(m.vertices) and np.isfinite(m.vertices).all()
        ]
        # --- save as .npy (single array) -----------------------
        
        
        # np.save(f"_sqs_params/{tgt_name}.npy", params_np)



    optim = True
    coacd = False
    filtering = True
    contact_only = False
    tri_mesh_handle=None

    load_from_sc = False 
    scene_dir = save_dir / 'scene'
    scene_dir.mkdir(parents=True, exist_ok=True)


    if plane_accum['S_items']:
        before_planes = len(plane_accum['S_items'])
        plane_accum = merge_plane_primitives(plane_accum)
        after_merge = len(plane_accum['S_items'])
        if after_merge != before_planes:
            print(f"[plane-merge] {before_planes} → {after_merge} after pairwise merge")

        before_cluster = after_merge
        plane_accum = cluster_plane_primitives(plane_accum)
        after_cluster = len(plane_accum['S_items'])
        if after_cluster != before_cluster:
            print(f"[plane-cluster] {before_cluster} → {after_cluster} after clustering")
        for idx, (S_item, R_item, T_item, color_item) in enumerate(
            zip(
                plane_accum['S_items'],
                plane_accum['R_items'],
                plane_accum['T_items'],
                plane_accum['color_items'],
            )
        ):
            half_ext = torch.exp(S_item).cpu().numpy()
            extents = half_ext * 2.0
            plane_mesh = trimesh.creation.box(extents=extents)
            transform = np.eye(4)
            transform[:3, :3] = R_item.cpu().numpy()
            transform[:3, 3] = T_item.cpu().numpy()
            plane_mesh.apply_transform(transform)
            color_np = color_item.cpu().numpy() if hasattr(color_item, 'cpu') else np.array(color_item, dtype=np.float32)
            color_list = np.clip(color_np, 0, 255).astype(np.uint8).tolist()
            handle = server.scene.add_mesh_trimesh(
                name=f"/planes/{idx}",
                mesh=plane_mesh,
                color=color_list,
            )
            handle.visible = gui_show_planes.value
            plane_mesh_handles.append(handle)


    if TTTTTTEST==False:
        if coacd:
            import coacd
            vert, tri = vdb_volume.extract_triangle_mesh()
            vertices, faces = vert, tri
            tri = trimesh.Trimesh(
            vertices=vertices,   
            faces=faces, 
            process=False
            )
            mesh = tri
            obj_data = trimesh.exchange.obj.export_obj(tri)
            scene_mesh_raw_handle = server.scene.add_mesh_trimesh(
                name="/frames/0/scene_mesh_raw",
                mesh=tri.copy(),
            )
            scene_mesh_raw_handle.visible = gui_show_scene_raw.value
            #tri_mesh_handle = server.scene.add_mesh_trimesh(
            #    name=f"/frames/{i}/scene_mesh_",
            #    mesh=tri
            #)
            with open(os.path.join(scene_dir, "scene_mesh.obj"), "w") as f:
                f.write(obj_data)

            # tri_mesh_handle.visible=True

        if 1==0:
          nksr_mesh_path = os.path.join(f'out/{tgt_name}/scene_mesh_nksr.obj')
          nksr_mesh = trimesh.load(nksr_mesh_path)
          scene_mesh_raw_handle   = tri_mesh_handle   
          scene_mesh_coacd_handle = server.scene.add_mesh_trimesh(
              name="/frames/0/scene_mesh_coacd",
              mesh=nksr_mesh,        # or the concatenated COACD mesh
          )

        per_sq_pt_one_list = results["pts_items"]
        scene_mesh_contact = (
            trimesh.util.concatenate(per_sq_one_list) if per_sq_one_list else None
        )

        prev_sq_idx = 0        # 保持当前可见索引


        os.makedirs(tgt_folder, exist_ok=True)
        tgt_folder = os.path.join(tgt_folder, hmr_type)
        BIGBIG_folder = tgt_folder
        os.makedirs(tgt_folder, exist_ok=True)
        tgt_tgt_folder = tgt_folder

        tgt_folder = os.path.join(tgt_folder, 'scene_mesh_sqs')
        os.makedirs(tgt_folder, exist_ok=True)
        if scene_mesh_contact is not None:
            scene_mesh_contact.export(os.path.join(tgt_folder, 'scene_mesh_sqs.obj'))
        save_custom_mesh(per_sq_one_list, tgt_folder)
        
        if transfer_data:

          dst_path = tgt_folder.replace('_geo/differentiable-blocksworld', '_vision/mega-sam')
          BIGBIG_folder_tgt = BIGBIG_folder.replace('_geo/differentiable-blocksworld', '_vision/mega-sam')
          shutil.copytree(BIGBIG_folder, BIGBIG_folder_tgt, dirs_exist_ok=True)
        print('done!')

    prev_timestep = gui_timestep.value

    if save_mode:
        server.stop()
    else:
        while True:
            if (
                gui_playing.value
                and not gui_show_all_frames.value
                and not gui_multi_timestep_mode.value
            ):
                gui_timestep.value = (gui_timestep.value + 1) % num_processed_frames # len(num_frames)

            time.sleep(1.0 / min(gui_framerate.value, 30))
if __name__ == "__main__":

    tyro.cli(main)

        
