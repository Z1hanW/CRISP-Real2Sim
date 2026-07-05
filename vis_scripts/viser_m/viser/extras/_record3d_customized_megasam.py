from __future__ import annotations

import dataclasses
import os
import json
import imageio
from PIL import Image
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast
import trimesh
import torch
import imageio.v3 as iio
import liblzfse
import numpy as np
import numpy as onp
from skimage.morphology import dilation, disk
import numpy.typing as onpt
import skimage.transform
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
import viser.utils3d as utils3d
from utils import *

import glob


import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio   # pip install imageio

def search_jpg_files(img_paths, tgt_name):
    """
    Search for JPG files in the given directory that contain the target name.
    
    Args:
        img_paths (str): Directory path to search
        tgt_name (str): Target name to search for in filenames
    
    Returns:
        list: List of matching JPG file paths
    """
    # Ensure the path exists
    if not os.path.exists(img_paths):
        print(f"Directory {img_paths} does not exist!")
        return []
    
    # Search patterns for JPG files (case insensitive)
    jpg_patterns = [
        os.path.join(img_paths, "**", "*.jpg"),
        os.path.join(img_paths, "**", "*.jpeg"),
        os.path.join(img_paths, "**", "*.JPG"),
        os.path.join(img_paths, "**", "*.JPEG")
    ]
    
    # Find all JPG files recursively
    all_jpg_files = []
    for pattern in jpg_patterns:
        all_jpg_files.extend(glob.glob(pattern, recursive=True))
    
    # Filter files that contain the target name
    matching_files = []
    for file_path in all_jpg_files:
        filename = os.path.basename(file_path)
        if tgt_name.lower() in filename.lower():
            matching_files.append(file_path)
    
    return matching_files

def process_single_frame_masks(
    depthmap,
    masks_i,
    R_cam,
    T_cam,
    fx,
    fy,
    cx,
    cy,
    ratio_block_scene: float = 1.0,
    avg_normal=None,
    colors_i=None,
    device: str = 'cuda',
):
    """
    Process masks from a single frame and return geometric data for each valid mask.
    
    Args:
        depthmap: Single depth map [1, H, W] or [H, W]
        masks_i: List of masks for this frame [mask1, mask2, ...]
        colors_i: List of colors for each mask [color1, color2, ...]
        R_cam: Camera rotation matrix
        T_cam: Camera translation vector
        fx, fy, cx, cy: Camera intrinsic parameters
        ratio_block_scene: Scaling ratio for blocks
        avg_normal: Optional average normal vector
        device: Device to run computations on
    
    Returns:
        tuple: (points_bg, normal_map, base_rgb, point_colours, extras).
        extras is a list `[depth, rotation, translation, K, points_bg_map, plane_info]`.
    """
    
    # Ensure tensors are on the correct device
    depthmap = torch.as_tensor(depthmap, device=device)
    R_cam = torch.as_tensor(R_cam, device=device)
    T_cam = torch.as_tensor(T_cam, device=device)
    
    frame_results = {
        'S_items': [],
        'R_items': [],
        'T_items': [],
        'color_items': [],
        'pts_items': [],
    }
    
    for j, mask_np in enumerate(masks_i):
        mask = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
        if mask.sum() == 0:
            continue
        
        ys, xs = torch.nonzero(mask, as_tuple=True)
        d = depthmap[0, ys, xs] if depthmap.dim() == 3 else depthmap[ys, xs]
        valid = d > 0

        xs_f, ys_f, d_f = xs[valid].float(), ys[valid].float(), d[valid]
        x_cam = -(xs_f - cx) * d_f / fx
        y_cam = -(ys_f - cy) * d_f / fy
        z_cam = d_f
        pts_cam = torch.stack([x_cam, y_cam, z_cam], dim=1)

        n_cam, centre, inlier_mask, *_ = ransac_plane_torch(
            pts_cam, n_iters=300, inlier_thresh=0.015, device=device)

        if avg_normal is not None:
            avg_n = torch.as_tensor(avg_normal, dtype=torch.float32, device=device)
            avg_n = torch.stack((-avg_n[0], -avg_n[1], avg_n[2]))
            avg_n = F.normalize(avg_n, dim=0)
            n_cam = avg_n if avg_n[2] >= 0 else -avg_n

        z_axis = n_cam
        world_up_cam = torch.tensor([0., 1., 0.], device=device)
        dot_v = torch.abs(torch.dot(z_axis, world_up_cam))
        ref = torch.tensor([1., 0., 0.], device=device) if dot_v > 0.7 else world_up_cam
        x_axis = F.normalize(ref - torch.dot(ref, z_axis) * z_axis, dim=0)
        y_axis = F.normalize(torch.cross(z_axis, x_axis), dim=0)

        P = pts_cam[inlier_mask] if inlier_mask.sum() > 20 else pts_cam
        centre = P.mean(0, keepdim=True)
        P0 = P - centre
        proj_N = (P0 @ z_axis).unsqueeze(1) * z_axis
        P_plane = P0 - proj_N
        _, _, Vt = torch.linalg.svd(P_plane, full_matrices=False)
        x_axis = F.normalize(Vt[0] - torch.dot(Vt[0], z_axis) * z_axis, dim=0)
        y_axis = F.normalize(torch.cross(z_axis, x_axis), dim=0)

        R_bc = torch.stack([x_axis, y_axis, z_axis], dim=1)
        if torch.det(R_bc) < 0:
            R_bc[:, 0] = -R_bc[:, 0]

        u = P_plane @ x_axis
        v = P_plane @ y_axis
        Lx, Ly = u.max() - u.min(), v.max() - v.min()
        half_ext = 0.5 * torch.tensor([Lx, Ly, 0.05], device=device).clamp(min=0.02)

        S_item = torch.log(half_ext / ratio_block_scene)
        R_bw = R_cam @ R_bc
        T_bw = R_cam @ centre.squeeze(0) + T_cam
        R_item = R_bw.t().contiguous()
        T_item = T_bw
        if colors_i is not None and j < len(colors_i):
            color_np = colors_i[j]
        else:
            color_np = np.array([200.0, 200.0, 200.0], dtype=np.float32)
        color_item = torch.as_tensor(color_np, dtype=torch.float32, device=device)

        # Point cloud in world space
        P_w = P @ R_cam.T + T_cam
        pts_item = P_w.cpu()

        # Append to frame results
        frame_results['S_items'].append(S_item)
        frame_results['R_items'].append(R_item)
        frame_results['T_items'].append(T_item)
        frame_results['color_items'].append(color_item)
        frame_results['pts_items'].append(pts_item)
    
    return frame_results
def save_normal_images(depth_unit: np.ndarray,
                       mono_unit:  np.ndarray,
                       out_dir: str = "normal_debug"):
    """
    保存并可视化法向与余弦相似度热图。

    Args:
        depth_unit: (H, W, 3)  深度反求的单位法向（相机或世界系）
        mono_unit : (H, W, 3)  单目/模型法向（已归一化，已对齐同一坐标系）
        out_dir   : 输出文件夹
    """
    os.makedirs(out_dir, exist_ok=True)

    """
    Create a single image showing depth normals, mono normals, and cosine similarity.
    
    Args:
        depth_unit: Normalized depth normal vectors (H, W, 3)
        mono_unit: Normalized mono normal vectors (H, W, 3)  
        out_dir: Output directory path
    """
    
    # ---------- 1. Convert normals to RGB ----------
    # Map from (-1,1) to (0,255) for visualization
    depth_rgb = ((depth_unit + 1.0) * 127.5).astype(np.uint8)
    mono_rgb = ((mono_unit + 1.0) * 127.5).astype(np.uint8)
    
    # ---------- 2. Compute cosine similarity ----------
    cos_sim = np.sum(depth_unit * mono_unit, axis=2)  # (-1,1)
    cos_img = ((cos_sim + 1.0) * 0.5)  # Map to (0,1)
    
    # ---------- 3. Create combined visualization ----------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Normal Vector Comparison Analysis', fontsize=20, fontweight='bold')
    
    # Depth normals
    axes[0, 0].imshow(depth_rgb)
    axes[0, 0].set_title('Depth Normals', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Mono normals  
    axes[0, 1].imshow(mono_rgb)
    axes[0, 1].set_title('Mono Normals', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Cosine similarity heatmap
    im = axes[1, 0].imshow(cos_img, cmap='plasma', vmin=0.0, vmax=1.0)
    axes[1, 0].set_title('Cosine Similarity Heatmap', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1, 0], shrink=0.8)
    cbar.set_label('Similarity (0=orthogonal, 1=parallel)', rotation=270, labelpad=20)
    
    # Statistics plot
    axes[1, 1].axis('off')
    
    # Calculate statistics
    avg_sim = np.mean(cos_img)
    high_sim_pct = np.mean(cos_img > 0.8) * 100
    low_sim_pct = np.mean(cos_img < 0.4) * 100
    median_sim = np.median(cos_img)
    std_sim = np.std(cos_img)
    
    # Create histogram
    axes[1, 1].hist(cos_img.flatten(), bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_title('Cosine Similarity Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Cosine Similarity')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"""
    Statistics:
    Mean: {avg_sim:.3f}
    Median: {median_sim:.3f}
    Std: {std_sim:.3f}
    
    High similarity (>0.8): {high_sim_pct:.1f}%
    Low similarity (<0.4): {low_sim_pct:.1f}%
    """
    axes[1, 1].text(0.02, 0.98, stats_text, transform=axes[1, 1].transAxes, 
                    verticalalignment='top', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # ---------- 4. Save combined image ----------
    os.makedirs(out_dir, exist_ok=True)
    combined_path = os.path.join(out_dir, "combined_normals_analysis.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight', pad_inches=0.1)

    
    # Also save individual images as before
    imageio.imwrite(os.path.join(out_dir, "depth_normals.png"), depth_rgb)
    imageio.imwrite(os.path.join(out_dir, "mono_normals.png"), mono_rgb)
    
    # Save cosine similarity heatmap separately
    plt.figure(figsize=(10, 8))
    plt.imshow(cos_img, cmap="plasma", vmin=0.0, vmax=1.0)
    plt.title("Cosine Similarity Heatmap", fontsize=16, fontweight='bold')
    plt.axis("off")
    plt.colorbar(shrink=0.8, label='Cosine Similarity')
    plt.savefig(os.path.join(out_dir, "cos_similarity.png"),
                dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    
    print(f"Combined visualization saved to: {os.path.abspath(combined_path)}")
    print(f"Individual images saved to: {os.path.abspath(out_dir)}")
    
    return avg_sim, high_sim_pct, low_sim_pct


class Record3dLoader_Customized_Megasam:
    """Helper for loading frames for Record3D captures directly from a NPZ file."""

    def __init__(self, npz_data: dict, npz_cam_data: dict, conf_threshold: float = 1.0, foreground_conf_threshold: float = 0.1, no_mask: bool = False, xyzw=True, init_conf=False, extra_obj=False):
        # Assuming npz_data is a dictionary containing all the necessary arrays from the NPZ file
        num_frames = npz_data['images'].shape[0]
        intrinsic = np.asarray(
            npz_cam_data.get('intrinsics_per_frame', npz_cam_data['intrinsic']),
            dtype=np.float32,
        )
        if intrinsic.ndim == 2:
            self.K = np.repeat(intrinsic[None], num_frames, axis=0)
        elif intrinsic.ndim == 3:
            if intrinsic.shape[0] == 1 and num_frames > 1:
                self.K = np.repeat(intrinsic, num_frames, axis=0)
            elif intrinsic.shape[0] == num_frames:
                self.K = intrinsic.copy()
            else:
                raise ValueError(
                    f"Intrinsics frame count mismatch: K={intrinsic.shape}, images={npz_data['images'].shape}"
                )
        else:
            raise ValueError(f"Expected intrinsics shaped (3,3) or (T,3,3), got {intrinsic.shape}")

        self.S = npz_cam_data['scale']
        
        T_world_cameras = npz_cam_data['cam_c2w'].copy()
        T_world_cameras[..., :3, 3] *= self.S  # scale only the translation part
        self.T_world_cameras = T_world_cameras
        #self.T_world_cameras = np.broadcast_to(T_world_cameras[0], T_world_cameras.shape)
        #self.T_world_cameras = (npz_cam_data['cam_c2w']) 
        self.fps = 30  # Assuming a frame rate of 30

        self.foreground_conf_threshold = foreground_conf_threshold
        self.no_mask = no_mask

        # Initialize the other parameters
        self.init_conf = init_conf
        
        # Read frames from the NPZ file
        self.images = npz_data['images']   
        try:                             # (N,H,W,3) RGB images
          self.depths = npz_data['depth'] * self.S
        except:
          self.depths = npz_data['depths'] * self.S                     # (N,H,W) Depth maps

        #print(npz_data.get('uncertainty', []))
        # print(self.confidences.max(), self.confidences.min())
        self.init_conf_data = npz_data.get('init_conf', [])
        self.masks = npz_data.get('enlarged_dynamic_mask', [])
        self.obj_masks = npz_data.get('obj_masks', [])

        if 'uncertainty' in npz_data:
            self.confidences = np.array(npz_data['uncertainty'])
            self.confidence_source = 'uncertainty'
        elif 'depth_conf' in npz_data:
            self.confidences = np.array(npz_data['depth_conf'])
            self.confidence_source = 'depth_conf'
        else:
            self.confidences = np.array([])
            self.confidence_source = ''
        
        self.masks = cast(
            np.ndarray,
            skimage.transform.resize(self.masks, self.images.shape[:3], order=0),
        )
        # self.depths = npz_data['depths']                      # (N,H,W) Depth maps
        averaging = False
        if averaging:
          valid = (self.masks == 0)[::10]                          # shape (N, H, W)

          # replace invalid depth values with NaN so they’re ignored by nanmean
          org_length = len(self.depths)
          self.depths = self.depths[::10]           
          depths_valid = np.where(valid, self.depths, np.nan)  # (N, H, W)
          # mean across frames, skipping NaNs -> (H, W)
          depth_mean = np.nanmean(depths_valid, axis=0).astype(self.depths.dtype)

          # If a single depth map is all you need:
          self.depths = np.broadcast_to(depth_mean, (org_length, *depth_mean.shape))
        

        if len(self.confidences):
          finite_conf = self.confidences[np.isfinite(self.confidences)]
          if finite_conf.size == 0:
            self.conf_threshold = conf_threshold
          elif self.confidence_source == 'depth_conf':
            # VGGT-Omega's official viewer keeps points above a confidence
            # percentile. Use the same default 20th percentile here.
            q = float(os.environ.get("CRISP_DEPTH_CONF_QUANTILE", "0.2"))
            self.conf_threshold = np.quantile(finite_conf, np.clip(q, 0.0, 1.0))
          else:
            self.conf_threshold = np.quantile(finite_conf, 0.0)
        else: 
          self.conf_threshold = 0.0


        # Align all camera poses by the first frame
        T0 = self.T_world_cameras[len(self.T_world_cameras) // 2]  # First camera pose (4x4 matrix)
        T0_inv = np.linalg.inv(T0)  # Inverse of the first camera pose
    
        # Apply T0_inv to all camera poses 
        self.T_world_cameras =   self.T_world_cameras#np.matmul(T0_inv[np.newaxis, :, :], self.T_world_cameras)

    def num_frames(self) -> int:
        return len(self.images)

    def get_frame(self, index: int) -> Record3dFrame:
        # Read the depth for the given frame
        depth = self.depths[index]
        depth = depth.astype(np.float32)

        # Check if conf file exists, otherwise initialize with ones
        if len(self.confidences) == 0:
            conf = np.ones_like(depth, dtype=np.float32)
        else:
            conf = self.confidences[index]
            conf = np.clip(conf, 0.0001, 99999)

        # Check if init conf file exists, otherwise initialize with ones
        if len(self.init_conf_data) == 0:
            init_conf = conf
        else:
            init_conf = self.init_conf_data[index]
            init_conf = np.clip(init_conf, 0.0001, 99999)
        

        # Check if mask exists, otherwise initialize with zeros
        if len(self.masks) == 0:
            mask = np.ones_like(depth, dtype=bool)
        else:
            mask = self.masks[index] > 0  # Assuming mask is a binary image





        if len(self.obj_masks) == 0:
            # Missing object masks should mean "no object mask available", not
            # "every pixel is object". Using all-ones here wipes the entire
            # background and makes scene reconstruction empty.
            obj_mask = np.zeros_like(depth, dtype=bool)
        else:
            obj_mask = self.obj_masks[index] > 0  # Assuming mask is a binary image


        if self.no_mask:
            mask = np.ones_like(mask).astype(np.bool_)

        # Read RGB image
        rgb = self.images[index]
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]

        return Record3dFrame(
            K=self.K[index],
            rgb=rgb,
            depth=depth, # depth*3.7-0.4, # 4
            mask=mask,
            obj_mask=obj_mask,
            conf=conf,
            init_conf=init_conf,
            T_world_camera=self.T_world_cameras[index],
            conf_threshold=self.conf_threshold,
            foreground_conf_threshold=self.foreground_conf_threshold,
        )


@dataclasses.dataclass
class Record3dFrame:
    """A single frame from a Record3D capture."""

    K: np.ndarray  # onpt.NDArray[onp.float32]
    rgb: np.ndarray  # onpt.NDArray[onp.uint8]
    depth: np.ndarray  # onpt.NDArray[onp.float32]
    mask: np.ndarray  # onpt.NDArray[onp.bool_]
    obj_mask: np.ndarray  # onpt.NDArray[onp.bool_]
    conf: np.ndarray  # onpt.NDArray[onp.float32]
    init_conf: np.ndarray  # onpt.NDArray[onp.float32]
    T_world_camera: np.ndarray  # onpt.NDArray[onp.float32]
    conf_threshold: float = 1.0
    foreground_conf_threshold: float = 0.1

    def get_sdf(
        self,
        points,
        downsample_factor: int = 1,
        bg_downsample_factor: int = 1,
        world_coords: bool = True,
        vdb_volume=None,
    ) -> Tuple[
        np.ndarray,  # foreground points
        np.ndarray,  # foreground colors
        np.ndarray,  # background points
        np.ndarray,  # background colors
        list,        # auxiliary data (depth/pose info)
        Dict[str, Any],  # plane metadata
    ]:
        """
        Return a foreground and background point cloud (and their colors).
        
        If world_coords=True, the points are transformed to world coordinates
        via T_world_camera. Otherwise, they remain in the local camera coordinates.
        """
        K = self.K
        T_world_camera = self.T_world_camera
        T_world_camera = np.asarray(T_world_camera, dtype=np.float64)

        return 2, T_world_camera, K
    
    def get_point_cloud(
        self,
        downsample_factor: int = 1,
        bg_downsample_factor: int = 1,
        world_coords: bool = True,
    ) -> Tuple[
        np.ndarray,  # foreground points
        np.ndarray,  # foreground colors
        np.ndarray,  # background points
        np.ndarray,  # background colors
    ]:
        """
        Return a foreground and background point cloud (and their colors).
        
        If world_coords=True, the points are transformed to world coordinates
        via T_world_camera. Otherwise, they remain in the local camera coordinates.
        """
        # Downsample the RGB image
        rgb = self.rgb[::downsample_factor, ::downsample_factor]

        # Downsample depth/mask/conf to match RGB
        depth = skimage.transform.resize(self.depth, rgb.shape[:2], order=0)
        mask = cast(
            np.ndarray,
            skimage.transform.resize(self.mask, rgb.shape[:2], order=0),
        )

        obj_mask = cast(
            np.ndarray,
            skimage.transform.resize(self.obj_mask, rgb.shape[:2], order=0),
        )
        mask = dilation(mask, disk(11))
        obj_mask = dilation(obj_mask, disk(11))
    
  
        assert depth.shape == rgb.shape[:2]

        K = self.K
        T_world_camera = self.T_world_camera

        # Create a pixel grid
        img_wh = rgb.shape[:2][::-1]  # (width, height)
        grid = (
            np.stack(np.meshgrid(np.arange(img_wh[0]), np.arange(img_wh[1])), axis=2)
            + 0.5
        )
        grid = grid * downsample_factor

        img_wh = rgb.shape[:2][::-1]  # (width, height)

        # Compute confidence masks at the downsampled resolution
        conf_mask = self.conf  > self.conf_threshold

        if self.init_conf is not None:
            fg_conf_mask = self.init_conf > self.foreground_conf_threshold
        else:
            fg_conf_mask = self.conf > self.foreground_conf_threshold
        conf_mask = skimage.transform.resize(conf_mask, depth.shape, order=0)
        fg_conf_mask = skimage.transform.resize(fg_conf_mask, depth.shape, order=0)

        if world_coords:
            # Transform to world coordinates
            rotation = T_world_camera[:3, :3]
            translation = T_world_camera[:3, 3]
        else:
            # Remain in camera coordinates
            rotation = np.eye(3, dtype=np.float32)
            translation = np.zeros((3,), dtype=np.float32)

        # ========= Foreground =========
                    
        rot_180 = np.eye(3)
        rot_180[1, 1] = -1
        rot_180[2, 2] = -1  
        # ========= All =========
        all_indices = np.ones_like(conf_mask) #& np.ones(conf_mask)
        homo_grid = np.pad(grid[all_indices], ((0, 0), (0, 1)), constant_values=1)
        local_dirs = np.einsum("ij,bj->bi", np.linalg.inv(K), homo_grid)
        dirs = np.einsum("ij,bj->bi", rotation, local_dirs)
        points = translation + dirs * depth[all_indices, None]
        points_all = points.astype(np.float32)
        point_colors_all = rgb
        points_hw = points_all.reshape(img_wh[1], img_wh[0], 3)

        #all_indices = conf_mask #& np.ones(conf_mask)
        points_cam = (local_dirs * depth[all_indices, None])
        
        points_cam = points_cam.astype(np.float32).reshape(img_wh[1], img_wh[0], 3)

        homo_grid = np.pad(grid[all_indices], ((0, 0), (0, 1)), constant_values=1)
        local_dirs = np.einsum("ij,bj->bi", np.linalg.inv(K), homo_grid)
        dirs = np.einsum("ij,bj->bi", rotation, local_dirs)
        points = translation + dirs * depth[all_indices, None]
        points = points.astype(np.float32)


        threshold = 0.03

        normals, normals_mask = utils3d.numpy.points_to_normals(points_cam, mask=~mask)
        edge_mask =  (utils3d.numpy.depth_edge(depth, rtol=threshold)) #& utils3d.numpy.normals_edge(normals, tol=5))
        #mask = mask & ~edge_mask
        #mask_to_save = (edge_mask.astype(np.uint8)) * 255
        # Save as PNG
        #imageio.imwrite('filtered_maskk.png', mask_to_save)
        mask1 = mask
        mask2 = edge_mask
        mask3 = depth > 5.0
        mean_plus_3std = np.quantile(depth, 0.8)
        mask4 = depth > mean_plus_3std

        joint_mask = mask1 | obj_mask | mask2 | mask4 #| mask3 #| mask4

        fg_indices = fg_conf_mask & mask
        fg_homo_grid = np.pad(grid[fg_indices], ((0, 0), (0, 1)), constant_values=1)
        local_dirs_fg = np.einsum("ij,bj->bi", np.linalg.inv(K), fg_homo_grid)
        dirs_fg = np.einsum("ij,bj->bi", rotation, local_dirs_fg)
        points_fg = translation + dirs_fg * depth[fg_indices, None]   
        points_fg = points_fg.astype(np.float32)
        point_colors_fg = rgb[fg_indices]


        obj_indices = obj_mask
        obj_homo_grid = np.pad(grid[obj_indices], ((0, 0), (0, 1)), constant_values=1)
        local_dirs_obj = np.einsum("ij,bj->bi", np.linalg.inv(K), obj_homo_grid)
        dirs_obj = np.einsum("ij,bj->bi", rotation, local_dirs_obj)
        points_obj = translation + dirs_obj * depth[obj_indices, None]
        points_obj = points_obj.astype(np.float32)
        point_colors_obj = rgb[obj_indices]



        # ========= Background =========
        bg_indices = conf_mask & (depth > 0) & ~joint_mask
        bg_homo_grid = np.pad(grid[bg_indices], ((0, 0), (0, 1)), constant_values=1)
        local_dirs_bg = np.einsum("ij,bj->bi", np.linalg.inv(K), bg_homo_grid)
        dirs_bg = np.einsum("ij,bj->bi", rotation, local_dirs_bg)
        points_bg = translation + dirs_bg * depth[bg_indices, None]
        points_bg = points_bg.astype(np.float32)
        point_colors_bg = rgb[bg_indices]
        # points_bg = points_bg @ rot_180 

        H, W = depth.shape
        points_bg_map = np.zeros((H, W, 3), dtype=np.float32)
        points_bg_map_nksr = np.zeros((H, W, 3), dtype=np.float32)

        # 2.  Scatter the background points into the map
        # ── Case A: bg_indices has shape (H, W) ────────────────────────────────
        if bg_indices.ndim == 2:               
            points_bg_map[bg_indices] = points_bg      # points_bg is N×3

        nksr_bg_indices = conf_mask & np.isfinite(depth) & (depth > 0) & (~mask) & (~obj_mask)
        if np.any(nksr_bg_indices):
            nksr_bg_homo_grid = np.pad(grid[nksr_bg_indices], ((0, 0), (0, 1)), constant_values=1)
            nksr_local_dirs_bg = np.einsum("ij,bj->bi", np.linalg.inv(K), nksr_bg_homo_grid)
            nksr_dirs_bg = np.einsum("ij,bj->bi", rotation, nksr_local_dirs_bg)
            nksr_points_bg = translation + nksr_dirs_bg * depth[nksr_bg_indices, None]
            points_bg_map_nksr[nksr_bg_indices] = nksr_points_bg.astype(np.float32)
            

        if bg_downsample_factor > 1 and points_bg.shape[0] > 0:
            indices = np.random.choice(
                points_bg.shape[0],
                size=points_bg.shape[0] // bg_downsample_factor,
                replace=False,
            )
            points_bg = points_bg[indices]
            point_colors_bg = point_colors_bg[indices]
        depth[joint_mask] = 0.0
        normals[joint_mask] =0.0

        return [points_fg, point_colors_fg, points_bg, point_colors_bg, points_all, point_colors_all, points_obj, point_colors_obj], normals, [depth, rotation, translation, K, points_bg_map, points_bg_map_nksr]


    def get_filtered_point_cloud(
        self,
        downsample_factor: int = 1,
        bg_downsample_factor: int = 1,
        world_coords: bool = True,
        mono_normal: np.array | None = None, 
        
    ) -> Tuple[
        np.ndarray,  # foreground points
        np.ndarray,  # foreground colors
        np.ndarray,  # background points
        np.ndarray,  # background colors
    ]:
        """
        Return a foreground and background point cloud (and their colors).
        
        If world_coords=True, the points are transformed to world coordinates
        via T_world_camera. Otherwise, they remain in the local camera coordinates.
        """
        # Downsample the RGB image
        rgb = self.rgb[::downsample_factor, ::downsample_factor]

        # Downsample depth/mask/conf to match RGB
        depth = skimage.transform.resize(self.depth, rgb.shape[:2], order=0)
        mask = cast(
            np.ndarray,
            skimage.transform.resize(self.mask, rgb.shape[:2], order=0),
        )

        obj_mask = cast(
            np.ndarray,
            skimage.transform.resize(self.obj_mask, rgb.shape[:2], order=0),
        )
        mask = dilation(mask, disk(11))
        obj_mask = dilation(obj_mask, disk(11))
    
  
        assert depth.shape == rgb.shape[:2]

        K = self.K
        T_world_camera = self.T_world_camera

        # Create a pixel grid
        img_wh = rgb.shape[:2][::-1]  # (width, height)
        grid = (
            np.stack(np.meshgrid(np.arange(img_wh[0]), np.arange(img_wh[1])), axis=2)
            + 0.5
        )
        grid = grid * downsample_factor

        img_wh = rgb.shape[:2][::-1]  # (width, height)

        # Compute confidence masks at the downsampled resolution
        conf_mask = self.conf  > self.conf_threshold

        if self.init_conf is not None:
            fg_conf_mask = self.init_conf > self.foreground_conf_threshold
        else:
            fg_conf_mask = self.conf > self.foreground_conf_threshold
        conf_mask = skimage.transform.resize(conf_mask, depth.shape, order=0)
        fg_conf_mask = skimage.transform.resize(fg_conf_mask, depth.shape, order=0)

        if world_coords:
            # Transform to world coordinates
            rotation = T_world_camera[:3, :3]
            translation = T_world_camera[:3, 3]
        else:
            # Remain in camera coordinates
            rotation = np.eye(3, dtype=np.float32)
            translation = np.zeros((3,), dtype=np.float32)

        # ========= All =========
        all_indices = np.ones_like(conf_mask) #& np.ones(conf_mask)
        homo_grid = np.pad(grid[all_indices], ((0, 0), (0, 1)), constant_values=1)
        local_dirs = np.einsum("ij,bj->bi", np.linalg.inv(K), homo_grid)
        dirs = np.einsum("ij,bj->bi", rotation, local_dirs)
        points = translation + dirs * depth[all_indices, None]
        points_all = points.astype(np.float32)
        point_colors_all = rgb
        points_hw = points_all.reshape(img_wh[1], img_wh[0], 3)
        points_cam = (local_dirs * depth[all_indices, None]).astype(np.float32).reshape(img_wh[1], img_wh[0], 3)

      
        threshold = 0.03

        normals, normals_mask = utils3d.numpy.points_to_normals(points_cam, mask=~mask)
        edge_mask =  (utils3d.numpy.depth_edge(depth, rtol=threshold)) #& utils3d.numpy.normals_edge(normals, tol=5))

        eps = 1e-8
        depth_unit = normals / (np.linalg.norm(normals,        axis=2, keepdims=True) + eps)
        mono_unit  = mono_normal / (np.linalg.norm(mono_normal, axis=2, keepdims=True) + eps)
        
        save_normal_images(depth_unit, mono_unit)
        cos_sim = np.sum(depth_unit * mono_unit, axis=2)   
        cos_thresh = np.cos(np.deg2rad(30.0))
        similar_normals_mask  = cos_sim <= cos_thresh        # True = agree
        dissimilar_normals_mask = similar_normals_mask      # True = reject


        #R_correction = compute_rotation_correction(depth_unit, mono_unit)
        #rotation = np.dot(R_correction, rotation)
        
        #mask = mask & ~edge_mask
        #mask_to_save = (edge_mask.astype(np.uint8)) * 255
        # Save as PNG
        #imageio.imwrite('filtered_maskk.png', mask_to_save)
        mask1 = mask
        mask2 = edge_mask
        mask3 = depth > 5.0
        def filter_bg_points_by_human_distance(
            point: np.ndarray,
            human_transl_np: np.ndarray,
            max_dist: float = 2.0,
        ) -> Tuple[np.ndarray, np.ndarray]:
            tree = cKDTree(human_transl_np)
            distances, _ = tree.query(point, k=1)          # nearest-neighbour distance
            mask = distances <= max_dist
            return mask



        threshold_logged = np.quantile(np.log1p(depth), 0.9977)


        mean_plus_3std = np.quantile(depth, 0.99)
        mask4 = depth >= mean_plus_3std

        joint_mask = mask1 | obj_mask | mask2 | mask4 # | dissimilar_normals_mask #| mask3 #|  dissimilar_normals_mask
        ## this is the element to ignore





        # ========= Background =========
        bg_indices = conf_mask & ~joint_mask#  & (~joint_mask)
        bg_homo_grid = np.pad(grid[bg_indices], ((0, 0), (0, 1)), constant_values=1)
        local_dirs_bg = np.einsum("ij,bj->bi", np.linalg.inv(K), bg_homo_grid)
        dirs_bg = np.einsum("ij,bj->bi", rotation, local_dirs_bg)
        points_bg = translation + dirs_bg * depth[bg_indices, None]
        points_bg = points_bg.astype(np.float32)
        point_colors_bg = rgb[bg_indices]
        # points_bg = points_bg @ rot_180 

        if bg_downsample_factor > 1 and points_bg.shape[0] > 0:
            indices = np.random.choice(
                points_bg.shape[0],
                size=points_bg.shape[0] // bg_downsample_factor,
                replace=False,
            )
            points_bg = points_bg[indices]
            point_colors_bg = point_colors_bg[indices]
        depth[joint_mask] = 0.0
       # normal[joint_mask] = 0.0
        return [points_bg, point_colors_bg, points_bg, point_colors_bg, point_colors_bg, point_colors_bg, point_colors_bg, point_colors_bg], normals, [depth, rotation, translation, K]





    def get_sqs(
        self,
        downsample_factor: int = 1,
        bg_downsample_factor: int = 1,
        world_coords: bool = True,
        existing_sqs: List = [],
        exisitng_points: List = [],

    ) -> Tuple[
        np.ndarray,  # superquadric parameters (N, 11)
        np.ndarray,  # background points
        np.ndarray,  # background colors
    ]:
        """
        Return superquadric parameters fitted to the foreground and background point cloud.
        
        If world_coords=True, the points are transformed to world coordinates
        via T_world_camera. Otherwise, they remain in the local camera coordinates.
        """
        # Downsample the RGB image
        rgb = self.rgb[::downsample_factor, ::downsample_factor]

        # Downsample depth/mask/conf to match RGB
        depth = skimage.transform.resize(self.depth, rgb.shape[:2], order=0)
        mask = cast(
            np.ndarray,
            skimage.transform.resize(self.mask, rgb.shape[:2], order=0),
        )

        obj_mask = cast(
            np.ndarray,
            skimage.transform.resize(self.obj_mask, rgb.shape[:2], order=0),
        )
        mask = dilation(mask, disk(11))
        obj_mask = dilation(obj_mask, disk(11))

        assert depth.shape == rgb.shape[:2]

        K = self.K
        T_world_camera = self.T_world_camera

        # Create a pixel grid
        img_wh = rgb.shape[:2][::-1]  # (width, height)
        grid = (
            np.stack(np.meshgrid(np.arange(img_wh[0]), np.arange(img_wh[1])), axis=2)
            + 0.5
        )
        grid = grid * downsample_factor

        img_wh = rgb.shape[:2][::-1]  # (width, height)

        # Compute confidence masks at the downsampled resolution
        conf_mask = self.conf > self.conf_threshold

        if self.init_conf is not None:
            fg_conf_mask = self.init_conf > self.foreground_conf_threshold
        else:
            fg_conf_mask = self.conf > self.foreground_conf_threshold
        conf_mask = skimage.transform.resize(conf_mask, depth.shape, order=0)
        fg_conf_mask = skimage.transform.resize(fg_conf_mask, depth.shape, order=0)

        if world_coords:
            # Transform to world coordinates
            rotation = T_world_camera[:3, :3]
            translation = T_world_camera[:3, 3]
        else:
            # Remain in camera coordinates
            rotation = np.eye(3, dtype=np.float32)
            translation = np.zeros((3,), dtype=np.float32)
        if exisitng_points:
            prev_points_bg, prev_rotation, prev_translation = exisitng_points
            prev_points_bg = onp.concatenate(prev_points_bg, axis=0)
            reprojected_mask = self.reproject_and_mask_points(
                prev_points_bg, 
                prev_rotation, 
                prev_translation,
                K, 
                T_world_camera, 
                rgb.shape,  # Downsampled RGB shape
                downsample_factor
            )
        else:
            reprojected_mask = np.zeros_like(mask)

        # ========= Foreground =========
        rot_180 = np.eye(3)
        rot_180[1, 1] = -1
        rot_180[2, 2] = -1
        
        # ========= All =========
        all_indices = np.ones_like(conf_mask)
        homo_grid = np.pad(grid[all_indices], ((0, 0), (0, 1)), constant_values=1)
        local_dirs = np.einsum("ij,bj->bi", np.linalg.inv(K), homo_grid)
        dirs = np.einsum("ij,bj->bi", rotation, local_dirs)
        points = translation + dirs * depth[all_indices, None]
        points_all = points.astype(np.float32)
        point_colors_all = rgb
        points_hw = points_all.reshape(img_wh[1], img_wh[0], 3)

        all_indices = conf_mask

        homo_grid = np.pad(grid[all_indices], ((0, 0), (0, 1)), constant_values=1)
        local_dirs = np.einsum("ij,bj->bi", np.linalg.inv(K), homo_grid)
        dirs = np.einsum("ij,bj->bi", rotation, local_dirs)
        points = translation + dirs * depth[all_indices, None]
        points = points.astype(np.float32)

        threshold = 0.03

        normals, normals_mask = utils3d.numpy.points_to_normals(points_hw, mask=mask)
        edge_mask = utils3d.numpy.depth_edge(depth, rtol=threshold)
        
        mask1 = mask
        mask2 = edge_mask
        mask3 = depth > 5.0
        mean_plus_3std = np.quantile(depth, 0.95)
        mask4 = depth > mean_plus_3std
        joint_mask = mask1 | obj_mask | mask2 #| mask3
        bg_indices = conf_mask & ~joint_mask & ~reprojected_mask
        bg_homo_grid = np.pad(grid[bg_indices], ((0, 0), (0, 1)), constant_values=1)
        local_dirs_bg = np.einsum("ij,bj->bi", np.linalg.inv(K), bg_homo_grid)
        dirs_bg = np.einsum("ij,bj->bi", rotation, local_dirs_bg)
        points_bg = translation + dirs_bg * depth[bg_indices, None]

        ### here is all points 
        points_bg = points_bg.astype(np.float32)
        point_colors_bg = rgb[bg_indices]

        if bg_downsample_factor > 1 and points_bg.shape[0] > 0:
            indices = np.random.choice(
                points_bg.shape[0],
                size=points_bg.shape[0] // bg_downsample_factor,
                replace=False,
            )
            points_bg = points_bg[indices]
            point_colors_bg = point_colors_bg[indices]

        
        #### here comes 
            # Call the function to get superquadric parameters

        normals_masked = normals#[bg_indices] if normals.ndim == 2 else normals  # Handle different normal formats
        rgb_masked = rgb#[bg_indices]
        depth_masked = depth#[bg_indices]
        #print(depth_masked.shape)

        sqs = self.sqs_from_points(points_bg, normals_masked, rgb_masked, depth_masked, K, bg_indices, T_world_camera=T_world_camera, downsample_factor=32)

        past_rotation, past_translation = rotation, translation
        return sqs, points_bg, point_colors_bg, past_rotation, past_translation




    def get_sqs_from_normal_field(
        self,
        downsample_factor: int = 1,
        bg_downsample_factor: int = 1,
        world_coords: bool = True,
        existing_sqs: List = [],
        exisitng_points: List = [],

    ) -> Tuple[
        np.ndarray,  # superquadric parameters (N, 11)
        np.ndarray,  # background points
        np.ndarray,  # background colors
    ]:
        """
        Return superquadric parameters fitted to the foreground and background point cloud.
        
        If world_coords=True, the points are transformed to world coordinates
        via T_world_camera. Otherwise, they remain in the local camera coordinates.
        """
        # Downsample the RGB image
        rgb = self.rgb[::downsample_factor, ::downsample_factor]

        # Downsample depth/mask/conf to match RGB
        depth = skimage.transform.resize(self.depth, rgb.shape[:2], order=0)
        mask = cast(
            np.ndarray,
            skimage.transform.resize(self.mask, rgb.shape[:2], order=0),
        )

        obj_mask = cast(
            np.ndarray,
            skimage.transform.resize(self.obj_mask, rgb.shape[:2], order=0),
        )
        mask = dilation(mask, disk(11))
        obj_mask = dilation(obj_mask, disk(11))

        assert depth.shape == rgb.shape[:2]

        K = self.K
        T_world_camera = self.T_world_camera

        # Create a pixel grid
        img_wh = rgb.shape[:2][::-1]  # (width, height)
        grid = (
            np.stack(np.meshgrid(np.arange(img_wh[0]), np.arange(img_wh[1])), axis=2)
            + 0.5
        )
        grid = grid * downsample_factor

        img_wh = rgb.shape[:2][::-1]  # (width, height)

        # Compute confidence masks at the downsampled resolution
        conf_mask = self.conf > self.conf_threshold

        if self.init_conf is not None:
            fg_conf_mask = self.init_conf > self.foreground_conf_threshold
        else:
            fg_conf_mask = self.conf > self.foreground_conf_threshold
        conf_mask = skimage.transform.resize(conf_mask, depth.shape, order=0)
        fg_conf_mask = skimage.transform.resize(fg_conf_mask, depth.shape, order=0)

        if world_coords:
            # Transform to world coordinates
            rotation = T_world_camera[:3, :3]
            translation = T_world_camera[:3, 3]
        else:
            # Remain in camera coordinates
            rotation = np.eye(3, dtype=np.float32)
            translation = np.zeros((3,), dtype=np.float32)
        if exisitng_points:
            prev_points_bg, prev_rotation, prev_translation = exisitng_points
            prev_points_bg = onp.concatenate(prev_points_bg, axis=0)
            reprojected_mask = self.reproject_and_mask_points(
                prev_points_bg, 
                prev_rotation, 
                prev_translation,
                K, 
                T_world_camera, 
                rgb.shape,  # Downsampled RGB shape
                downsample_factor
            )
        else:
            reprojected_mask = np.zeros_like(mask)

        # ========= Foreground =========
        rot_180 = np.eye(3)
        rot_180[1, 1] = -1
        rot_180[2, 2] = -1


    


        frame_results = process_single_frame_masks(
            depthmap=depthmap,
            masks_i=pi, 
            R_cam=R_cam,
            T_cam=T_cam,
            fx=fx, fy=fy, cx=cx, cy=cy,
            ratio_block_scene=1.0,
            avg_normal=None,  # You can pass normals[j] if needed per mask
            device=device
        )
        
        mask1 = mask
        mask2 = edge_mask
        mask3 = depth > 5.0
        mean_plus_3std = np.quantile(depth, 0.95)
        mask4 = depth > mean_plus_3std
        joint_mask = mask1 | obj_mask | mask2 #| mask3
        bg_indices = conf_mask & ~joint_mask & ~reprojected_mask
        bg_homo_grid = np.pad(grid[bg_indices], ((0, 0), (0, 1)), constant_values=1)
        local_dirs_bg = np.einsum("ij,bj->bi", np.linalg.inv(K), bg_homo_grid)
        dirs_bg = np.einsum("ij,bj->bi", rotation, local_dirs_bg)
        points_bg = translation + dirs_bg * depth[bg_indices, None]

        ### here is all points 
        points_bg = points_bg.astype(np.float32)
        point_colors_bg = rgb[bg_indices]

        if bg_downsample_factor > 1 and points_bg.shape[0] > 0:
            indices = np.random.choice(
                points_bg.shape[0],
                size=points_bg.shape[0] // bg_downsample_factor,
                replace=False,
            )
            points_bg = points_bg[indices]
            point_colors_bg = point_colors_bg[indices]

        if bg_downsample_factor > 1 and points_bg.shape[0] > 0:
            indices = np.random.choice(
                points_bg.shape[0],
                size=points_bg.shape[0] // bg_downsample_factor,
                replace=False,
            )
            points_bg = points_bg[indices]
            point_colors_bg = point_colors_bg[indices]
        
        #### here comes 
            # Call the function to get superquadric parameters

        normals_masked = normals#[bg_indices] if normals.ndim == 2 else normals  # Handle different normal formats
        rgb_masked = rgb#[bg_indices]
        depth_masked = depth#[bg_indices]


        sqs = self.sqs_from_points(points_bg, normals_masked, rgb_masked, depth_masked, K, bg_indices, T_world_camera=T_world_camera, downsample_factor=32)

        past_rotation, past_translation = rotation, translation
        return sqs, points_bg, point_colors_bg, past_rotation, past_translation




    def reproject_and_mask_points(self, prev_points_bg, prev_rotation, prev_translation, 
                                K, T_world_camera, img_shape, downsample_factor):
        """
        Reproject world coordinate points into current camera view and create mask.
        
        Args:
            prev_points_bg: Previous background points in world coordinates (N, 3)
            prev_rotation: Previous rotation matrix (not used if points are in world coords)
            prev_translation: Previous translation vector (not used if points are in world coords)
            K: Camera intrinsic matrix (3, 3)
            T_world_camera: Current camera pose (4, 4)
            img_shape: Image shape (height, width, channels)
            downsample_factor: Downsampling factor used
        
        Returns:
            reprojected_mask: Binary mask where True indicates reprojected points
        """
        
        # Get current camera pose
        R_camera_world = T_world_camera[:3, :3].T  # Inverse rotation
        t_camera_world = -R_camera_world @ T_world_camera[:3, 3]  # Inverse translation
        
        # Transform world points to current camera coordinates
        points_camera = (R_camera_world @ prev_points_bg.T).T + t_camera_world
        
        # Filter out points behind the camera
        valid_depth = points_camera[:, 2] > 0.1  # Small positive threshold
        points_camera_valid = points_camera[valid_depth]
        
        if len(points_camera_valid) == 0:
            # No valid points to reproject
            return np.zeros((img_shape[0], img_shape[1]), dtype=bool)
        
        # Project to image coordinates
        points_camera_homo = points_camera_valid / points_camera_valid[:, 2:3]  # Normalize by depth
        points_image_homo = (K @ points_camera_homo.T).T
        
        # Get pixel coordinates
        u = points_image_homo[:, 0] / downsample_factor
        v = points_image_homo[:, 1] / downsample_factor
        
        # Round to integer pixel coordinates
        u_int = np.round(u).astype(int)
        v_int = np.round(v).astype(int)
        
        # Create mask for valid projections within image bounds
        img_h, img_w = img_shape[0], img_shape[1]
        valid_proj = (u_int >= 0) & (u_int < img_w) & (v_int >= 0) & (v_int < img_h)
        
        # Create reprojection mask
        reprojected_mask = np.zeros((img_h, img_w), dtype=bool)
        
        # Set mask pixels where previous points project to
        valid_u = u_int[valid_proj]
        valid_v = v_int[valid_proj]
        reprojected_mask[valid_v, valid_u] = True
        
        # Optionally dilate the mask to account for discretization errors
        from skimage.morphology import dilation, disk
        reprojected_mask = dilation(reprojected_mask, disk(3))
        
        return reprojected_mask

    def sqs_from_points(self, points_bg, normals_masked, rgb_masked, depth_masked, K, bg_indices, T_world_camera, downsample_factor=32):
        """
        Convert point cloud to superquadric parameters.
        Returns array of shape (N, 11) where each row is:
        [ euler_y, euler_x, tx, ty, tz]
        """
        import torch
        import torch.nn.functional as F
        from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        points_torch = torch.from_numpy(points_bg).to(device).float()
        normals_torch = torch.from_numpy(normals_masked).to(device).float()
        rgb_torch = torch.from_numpy(rgb_masked).to(device).float()
        depth_torch = torch.from_numpy(depth_masked).to(device).float()
        K_torch = torch.from_numpy(K).to(device).float()
        bg_indices_torch = torch.from_numpy(bg_indices).to(device)

        # Camera intrinsics
        fx, fy = K_torch[0, 0], K_torch[1, 1]
        cx, cy = K_torch[0, 2], K_torch[1, 2]

        # Camera->world extrinsics
        if T_world_camera is not None:
            Twc = torch.as_tensor(T_world_camera, device=device, dtype=torch.float32)
            if Twc.ndim == 3:  # if batched, take first (caller should pass 4x4)
                Twc = Twc[0]
        else:
            Twc = torch.eye(4, device=device, dtype=torch.float32)

        R_wc = Twc[:3, :3]
        t_wc = Twc[:3, 3]


        # Get valid indices where we have data
        valid_indices = torch.nonzero(bg_indices_torch, as_tuple=False) #| bg_indices.to(device)
        
        # Downsample the valid indices
        downsampled_indices = []
        for idx in valid_indices:
            y, x = idx[0].item(), idx[1].item()
            if y % downsample_factor == 0 and x % downsample_factor == 0:
                downsampled_indices.append(idx)
        
        if len(downsampled_indices) == 0:
            return np.zeros((0, 11), dtype=np.float32)
        
        downsampled_indices = torch.stack(downsampled_indices)
        
        # Extract camera parameters
        fx, fy = K_torch[0, 0], K_torch[1, 1]
        cx, cy = K_torch[0, 2], K_torch[1, 2]
        
        # Initialize parameter lists
        params_list = []
        
        # Process each downsampled pixel
        for idx in downsampled_indices:
            y, x = idx[0].item(), idx[1].item()
            
            # Get depth at this pixel
            d = depth_torch[y, x]
            
            # Unproject pixel to 3D point in camera space
            x_cam = (x - cx) * d / fx
            y_cam = (y - cy) * d / fy
            z_cam = d
            
            # Position (translation)
            p_cam = torch.tensor([x_cam, y_cam, z_cam], device=device)
            position = R_wc @ p_cam + t_wc

            # Get normal at this pixel
            if normals_torch.dim() >= 2 and y < normals_torch.shape[0] and x < normals_torch.shape[1]:
                if normals_torch.dim() == 3:
                    n_cam = normals_torch[y, x, :]
                else:
                    # Handle case where normals might be flattened or in different format
                    n_cam = torch.tensor([0., 0., -1.], device=device, dtype=torch.float32)
                n_cam = F.normalize(n_cam, dim=0)
            else:
                # Default normal pointing towards camera
                n_cam = torch.tensor([0., 0., -1.], device=device, dtype=torch.float32)
            
            n_world = R_wc @ n_cam
            n_world = F.normalize(n_world, dim=0)
            # Create orthonormal basis
            if abs(n_world[1]) > 0.7:  # nearly aligned with +Y/-Y
                x_axis = torch.tensor([1., 0., 0.], device=device)
                x_axis = x_axis - torch.dot(x_axis, n_world) * n_world
                x_axis = F.normalize(x_axis, dim=0)
                y_axis = torch.cross(n_world, x_axis)
            else:
                y_axis = torch.tensor([0., 1., 0.], device=device)
                y_axis = y_axis - torch.dot(y_axis, n_world) * n_world
                if torch.norm(y_axis) < 0.1:
                    y_axis = torch.tensor([0., 0., 1.], device=device)
                    y_axis = y_axis - torch.dot(y_axis, n_world) * n_world
                y_axis = F.normalize(y_axis, dim=0)
                x_axis = torch.cross(y_axis, n_world)

            R_block_to_world = torch.stack([x_axis, y_axis, n_world], dim=1)
            # Ensure proper rotation matrix (det = 1)
            if torch.det(R_block_to_world) < 0:
                R_block_to_world[:, 0] = -R_block_to_world[:, 0]
            
            euler_angles = matrix_to_euler_angles(R_block_to_world.unsqueeze(0), "ZYX").squeeze(0)

            # crude size heuristic from pixel footprint @ depth
            scale = d / ((fx + fy) / 2) * downsample_factor / 2
            scale = float(torch.clamp(scale, min=0.01).item())

            # roundness
            eps1 = 0.1
            eps2 = 0.1
            
            params = torch.tensor([
                eps1, eps2,
                scale, scale, scale,
                euler_angles[0].item(), euler_angles[1].item(), euler_angles[2].item(),
                position[0].item(), position[1].item(), position[2].item()
            ], device=device)
            
            params_list.append(params)
        
        if len(params_list) == 0:
            return np.zeros((0, 11), dtype=np.float32)
        
        # Stack all parameters
        all_params = torch.stack(params_list)
        
        return all_params.cpu().numpy()

    def get_meshes(
        self,
        points,
        downsample_factor: int = 1,
        bg_downsample_factor: int = 1,
        world_coords: bool = True,
    ) -> Tuple[
        np.ndarray,  # foreground points
        np.ndarray,  # foreground colors
        np.ndarray,  # background points
        np.ndarray,  # background colors
    ]:
        """
        Return a foreground and background point cloud (and their colors).
        
        If world_coords=True, the points are transformed to world coordinates
        via T_world_camera. Otherwise, they remain in the local camera coordinates.
        """
        # Downsample the RGB image
        rgb = self.rgb[::downsample_factor, ::downsample_factor]

        # Downsample depth/mask/conf to match RGB
        depth = skimage.transform.resize(self.depth, rgb.shape[:2], order=0)
        mask = cast(
            np.ndarray,
            skimage.transform.resize(self.mask, rgb.shape[:2], order=0),
        )
        obj_mask = cast(
            np.ndarray,
            skimage.transform.resize(self.obj_mask, rgb.shape[:2], order=0),
        )
        mask = dilation(mask, disk(11))
        obj_mask = dilation(obj_mask, disk(11))
        assert depth.shape == rgb.shape[:2]

        K = self.K
        T_world_camera = self.T_world_camera

        mask = ~mask

        threshold=0.03
        img_wh = rgb.shape[:2][::-1]  # (width, height)
        points = points.reshape(img_wh[1], img_wh[0], 3)
        #points = points.reshape(-1, 3)[(depth > 1.0).flatten()]

        normals, normals_mask = utils3d.numpy.points_to_normals(points, mask=mask)
        edge_mask =  ~(utils3d.numpy.depth_edge(depth, rtol=threshold, mask=mask) & utils3d.numpy.normals_edge(normals, tol=5, mask=normals_mask))


        faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
            points,
            rgb.astype(np.float32) / 255,
            utils3d.numpy.image_uv(width=img_wh[0], height=img_wh[1]),
            mask=(mask & edge_mask),
            tri=True
        )
        vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]
        rot_180 = np.eye(3)
        rot_180[1, 1] = -1
        rot_180[2, 2] = -1  
        # * [-1, 1, -1]
        tri = trimesh.Trimesh(
        vertices=vertices  @ rot_180 ,    # No idea why Gradio 3D Viewer' default camera is flipped
        faces=faces, 
        visual = trimesh.visual.texture.TextureVisuals(
            uv=vertex_uvs, 
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.fromarray(rgb),
                metallicFactor=0.5,
                roughnessFactor=1.0
            )
        ),
        process=False
        )

        return faces, vertices, vertex_colors, vertex_uvs, tri

    def project_to_cameras(
        self,
        target_frames,
        downsample_factor: int = 1,
        world_coords: bool = True,
    ) -> Tuple[
        np.ndarray,  # rendered images
        np.ndarray,  # ground truth images
        np.ndarray,  # difference images
        dict,        # metrics
    ]:
        """
        Project this frame's pointcloud to multiple target camera views.
        
        Args:
            target_frames: List of Record3dFrame objects to project to
            downsample_factor: Downsampling factor for pointcloud
            world_coords: Whether to use world coordinates
        
        Returns:
            Tuple of (rendered_images, ground_truth_images, difference_images, metrics)
        """
        # Get pointcloud from current frame
        points_fg, colors_fg, points_bg, colors_bg, points_all, colors_all, _, _ = \
            self.get_point_cloud(downsample_factor=downsample_factor, world_coords=world_coords)
        
        # Combine foreground and background points
        if points_fg.shape[0] > 0 and points_bg.shape[0] > 0:
            all_points = np.vstack([points_fg, points_bg])
            all_colors = np.vstack([colors_fg, colors_bg])
        else:
            all_points = points_all
            all_colors = colors_all
        
        rendered_images = []
        ground_truth_images = []
        difference_images = []
        metrics = []
        
        for target_frame in target_frames:
            # Project points to target camera
            rendered_img = self._project_points_to_camera(
                all_points, all_colors,
                target_frame.K, target_frame.T_world_camera,
                target_frame.rgb.shape[:2]
            )
            
            # Get ground truth
            gt_img = target_frame.rgb
            
            # Calculate difference
            diff_img = np.abs(rendered_img.astype(float) - gt_img.astype(float))
            
            # Calculate metrics
            mse = np.mean((rendered_img.astype(float) - gt_img.astype(float))**2)
            psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
            
            rendered_images.append(rendered_img)
            ground_truth_images.append(gt_img)
            difference_images.append(diff_img)
            metrics.append({'mse': mse, 'psnr': psnr})
        
        return (
            np.array(rendered_images),
            np.array(ground_truth_images),
            np.array(difference_images),
            metrics
        )


    def _project_points_to_camera(
        self,
        points_3d: np.ndarray,
        colors: np.ndarray,
        K: np.ndarray,
        T_world_camera: np.ndarray,
        img_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Internal method to project 3D points to camera image plane.
        
        Args:
            points_3d: (N, 3) array of 3D points in world coordinates
            colors: (N, 3) array of RGB colors
            K: (3, 3) camera intrinsic matrix
            T_world_camera: (4, 4) world to camera transformation
            img_shape: (H, W) target image shape
        
        Returns:
            Rendered image as (H, W, 3) array
        """
        # Convert to homogeneous coordinates
        points_homo = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
        
        # Transform to camera coordinates
        T_camera_world = np.linalg.inv(T_world_camera)
        points_cam = (T_camera_world @ points_homo.T).T[:, :3]
        
        # Filter points behind camera
        valid_mask = points_cam[:, 2] > 0.1
        points_cam = points_cam[valid_mask]
        colors = colors[valid_mask]
        
        # Project to image plane
        points_img_homo = (K @ points_cam.T).T
        points_img = points_img_homo[:, :2] / points_img_homo[:, 2:3]
        
        # Get depths for occlusion handling
        depths = points_cam[:, 2]
        
        # Initialize buffers
        rendered_img = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
        depth_buffer = np.full((img_shape[0], img_shape[1]), np.inf)
        
        # Render points with depth test
        for pt, color, depth in zip(points_img, colors, depths):
            x, y = int(np.round(pt[0])), int(np.round(pt[1]))
            
            if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]:
                if depth < depth_buffer[y, x]:
                    depth_buffer[y, x] = depth
                    rendered_img[y, x] = color
        
        return rendered_img


    def save_projection_comparison(
        self,
        target_frames,
        frame_indices,
        output_dir: str = "comparison_results",
        downsample_factor: int = 1,
    ) -> dict:
        """
        Project to target frames and save comparison figures.
        
        Args:
            target_frames: List of Record3dFrame objects to project to
            frame_indices: List of frame indices (for labeling)
            output_dir: Directory to save figures
            downsample_factor: Downsampling factor
        
        Returns:
            Dictionary with results and file paths
        """
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get projections
        rendered, ground_truth, differences, metrics = self.project_to_cameras(
            target_frames, downsample_factor=downsample_factor
        )
        
        saved_paths = []
        
        # Save individual comparisons
        for i, (rend, gt, diff, metric, target_idx) in enumerate(
            zip(rendered, ground_truth, differences, metrics, frame_indices)
        ):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Rendered view
            axes[0].imshow(rend)
            axes[0].set_title(f'Rendered View\n(→ Frame {target_idx})')
            axes[0].axis('off')
            
            # Ground truth
            axes[1].imshow(gt)
            axes[1].set_title(f'Ground Truth\n(Frame {target_idx})')
            axes[1].axis('off')
            
            # Difference map
            im = axes[2].imshow(np.mean(diff, axis=2), cmap='hot', vmin=0, vmax=50)
            axes[2].set_title(f'Difference\nMSE: {metric["mse"]:.2f}, PSNR: {metric["psnr"]:.2f} dB')
            axes[2].axis('off')
            
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            plt.tight_layout()
            
            save_path = Path(output_dir) / f'projection_to_frame_{target_idx}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            saved_paths.append(str(save_path))
        
        return {
            'rendered': rendered,
            'ground_truth': ground_truth,
            'differences': differences,
            'metrics': metrics,
            'saved_paths': saved_paths
        }

    def get_mono_data(
        self,
        i,
        seg_network, 
        parent_folder, 
        downsample_factor: int = 1,
        bg_downsample_factor: int = 1,
        world_coords: bool = True,
        single_image: bool = False
    ) -> Tuple[
        np.ndarray,  # foreground points
        np.ndarray,  # foreground colors
        np.ndarray,  # background points
        np.ndarray,  # background colors
    ]:
        """
        Return a foreground and background point cloud (and their colors).
        
        If world_coords=True, the points are transformed to world coordinates
        via T_world_camera. Otherwise, they remain in the local camera coordinates.
        """
        optim, trim = None, None
        # img_paths = [Path(f'{parent_folder}/{imIndex:05d}.jpg') for imIndex in range(num_frames)]
        
        if single_image:
          aaa=5
          img_paths = f'/data3/zihanwa3/_Robotics/_geo/differentiable-blocksworld/datasets/prox/'
          try:
              img_paths = search_jpg_files(img_paths, parent_folder)[0]# f'{parent_folder}/{i:05d}.jpg' # [Path(f'{parent_folder}/{imIndex:05d}.jpg') for imIndex in range(num_frames)]
          except:
              try:
                  tgt_name = parent_folder.split('/')[-1]
                  parent_folder=tgt_name.split('_')[0]
                  img_paths = search_jpg_files(img_paths, parent_folder)[0]
              except:
                  img_paths = f'{parent_folder}/{i:05d}.jpg' # [Path(f'{parent_folder}/{imIndex:05d}.jpg') for imIndex in range(num_frames)]

        priors = seg_network(img_paths, optim=optim)
        (
            plane_instances,
            masks_avg_normals,
            masks_areas,
            depth_areas,
            geo,
            plane_colors,
            cam_int,
        ) = priors
        depth, normal, image = geo
        depth, normal, image = depth[0], normal[0], image[0]
        depth = depth.permute(1, 2, 0)
        normal = normal.permute(1, 2, 0)
        image = image.permute(1, 2, 0)
        depth_np = depth[..., 0].detach().cpu().numpy()
        normal_np = normal.detach().cpu().numpy()
        image_np = image.detach().cpu().numpy()
        plane_info = None

        if plane_instances.shape[0] > 0:
            plane_masks_tensor = plane_instances[0].detach().cpu()
            num_masks = plane_masks_tensor.shape[0]
            valid_plane_mask = plane_masks_tensor.reshape(num_masks, -1).any(dim=1)

            plane_masks_list = [
                plane_masks_tensor[idx].numpy()
                for idx in range(num_masks)
                if valid_plane_mask[idx]
            ]

            avg_normals_tensor = masks_avg_normals[0].detach().cpu()
            areas_tensor = masks_areas[0].detach().cpu()
            depth_tensor = depth_areas[0].detach().cpu()
            plane_colors_tensor = plane_colors[0].detach().cpu()

            avg_normals_np = avg_normals_tensor[valid_plane_mask].numpy()
            areas_np = areas_tensor[valid_plane_mask].numpy()
            depths_np = depth_tensor[valid_plane_mask].numpy()
            plane_colors_np = plane_colors_tensor[valid_plane_mask].numpy()
            plane_colors_list = [plane_colors_np[idx] for idx in range(len(plane_colors_np))]

            if plane_masks_list:
                plane_results = process_single_frame_masks(
                    depthmap=depth_np,
                    masks_i=plane_masks_list,
                    R_cam=rotation,
                    T_cam=translation,
                    fx=float(K[0, 0]),
                    fy=float(K[1, 1]),
                    cx=float(K[0, 2]),
                    cy=float(K[1, 2]),
                    ratio_block_scene=1.0,
                    avg_normal=None,
                    colors_i=plane_colors_list,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                )
                plane_info = {
                    'masks': plane_masks_list,
                    'avg_normals': avg_normals_np,
                    'areas': areas_np,
                    'depths': depths_np,
                    'colors': plane_colors_np,
                    'primitives': {
                        'S_items': [s.detach().cpu() for s in plane_results['S_items']],
                        'R_items': [r.detach().cpu() for r in plane_results['R_items']],
                        'T_items': [t.detach().cpu() for t in plane_results['T_items']],
                        'color_items': [c.detach().cpu() for c in plane_results['color_items']],
                        'pts_items': [
                            pts.detach().cpu() if torch.is_tensor(pts) else torch.as_tensor(pts)
                            for pts in plane_results['pts_items']
                        ],
                    },
                }
            else:
                plane_info = {
                    'masks': [],
                    'avg_normals': np.zeros((0, 3)),
                    'areas': np.zeros((0,)),
                    'depths': np.zeros((0,)),
                    'colors': np.zeros((0, 3)),
                    'primitives': {'S_items': [], 'R_items': [], 'T_items': [], 'color_items': [], 'pts_items': []},
                }

        if plane_info is not None:
            plane_info['depth_map'] = depth_np.copy()
            plane_info['normal_map'] = normal_np.copy()
            plane_info['image'] = image_np.copy()

        depth = depth_np
        normal = normal_np
        image = image_np
        rgb = np.clip(image, 0, 255).astype(np.uint8)
        scale_up_factor = 3
        h_tgt, w_tgt = 328*scale_up_factor, 584*scale_up_factor
        depth = cv2.resize(
            depth, (w_tgt, h_tgt),        # (W, H)
            interpolation=cv2.INTER_LINEAR)   # keep exact values

        # ---------- surface normals (3-channel) ---- #
        normal = cv2.resize(
            normal, (w_tgt, h_tgt),
            interpolation=cv2.INTER_LINEAR)     # bilinear is fine for float vectors

        # ---------- RGB image (3-channel, uint8 or float) ---------- #
        rgb = cv2.resize(
            rgb, (w_tgt, h_tgt),
            interpolation=cv2.INTER_LINEAR)

  
        aligned_mega_depth = skimage.transform.resize(self.depth, rgb.shape[:2], order=0)


        valid = np.isfinite(aligned_mega_depth) & (aligned_mega_depth > 0) & np.isfinite(depth) & (depth > 0)
        if np.any(valid):
            x = depth[valid].reshape(-1, 1)
            y = aligned_mega_depth[valid].reshape(-1, 1)

            # Option A: scale-only (robust, median ratio)
            s =  np.median(y/x)
            # s = np.median(y) / np.median(x)
            depth = s * depth




        mask = cast(
            np.ndarray,
            skimage.transform.resize(self.mask, rgb.shape[:2], order=0),
        )
        assert depth.shape == rgb.shape[:2]

        K = self.K

        def scale_intrinsics(K, scale_up_factor):
            """
            Scale camera intrinsic matrix K by scale_up_factor
            
            Parameters:
            -----------
            K : torch.Tensor or np.ndarray, shape (3, 3)
                Original camera intrinsic matrix
            scale_up_factor : float
                Scale factor to apply
            
            Returns:
            --------
            K_scaled : same type as K
                Scaled intrinsic matrix
            """
            K_scaled = K.clone() if torch.is_tensor(K) else K.copy()
            
            # Scale focal lengths and principal point
            K_scaled[0, 0] *= scale_up_factor  # fx
            K_scaled[1, 1] *= scale_up_factor  # fy
            K_scaled[0, 2] *= scale_up_factor  # cx
            K_scaled[1, 2] *= scale_up_factor  # cy
            # K_scaled[2, 2] remains 1
            
            return K_scaled

        K = scale_intrinsics(K, scale_up_factor)
        if plane_info is not None:
            plane_info['intrinsics'] = K.copy() if isinstance(K, np.ndarray) else K.clone()
          
        T_world_camera = self.T_world_camera

        # Create a pixel grid
        img_wh = rgb.shape[:2][::-1]  # (width, height)
        grid = (
            np.stack(np.meshgrid(np.arange(img_wh[0]), np.arange(img_wh[1])), axis=2)
            + 0.5
        )
        grid = grid * downsample_factor

        # Compute confidence masks at the downsampled resolution
        conf_mask = self.conf > self.conf_threshold
        if self.init_conf is not None:
            fg_conf_mask = self.init_conf > self.foreground_conf_threshold
        else:
            fg_conf_mask = self.conf > self.foreground_conf_threshold
        conf_mask = skimage.transform.resize(conf_mask, depth.shape, order=0)
        fg_conf_mask = skimage.transform.resize(fg_conf_mask, depth.shape, order=0)
        def load_cam2world_transform(scene_path):
            """Load cam2world transformation from JSON file"""
            cam2world_path =scene_path # os.path.join(scene_path, "cam2world.json")
            
            if not os.path.exists(cam2world_path):
                raise FileNotFoundError(f"cam2world.json not found at {cam2world_path}")
            
            with open(cam2world_path, 'r') as f:
                data = json.load(f)
            T_world_camera = np.array(data, dtype=np.float32)
            T_world_camera = T_world_camera.reshape(4, 4)
            return T_world_camera

        # T_world_camera = load_cam2world_transform(os.path.join('/data3/zihanwa3/_Robotics/_data/_PROX/cam2world', f'{parent_folder}.json'))

        # Decide the transform based on `world_coords`
        if world_coords:
            # Transform to world coordinates
            rotation = T_world_camera[:3, :3]
            translation = T_world_camera[:3, 3]
        else:
            # Remain in camera coordinates
            rotation = np.eye(3, dtype=np.float32)
            translation = np.zeros((3,), dtype=np.float32)

        # ========= Background =========
  
        all_indices = np.ones_like(conf_mask)
        homo_grid = np.pad(grid[all_indices], ((0, 0), (0, 1)), constant_values=1)
        local_dirs = np.einsum("ij,bj->bi", np.linalg.inv(K), homo_grid)
        dirs = np.einsum("ij,bj->bi", rotation, local_dirs)

        points = translation + dirs * depth[all_indices, None]
        threshold=0.03
        img_wh = rgb.shape[:2][::-1]  # (width, height)
        points = points.reshape(img_wh[1], img_wh[0], 3)
        #mask = np.zeros_like(conf_mask)
        normal = utils3d.numpy.points_to_normals(points)

       #  edge_mask =  ~(utils3d.numpy.depth_edge(depth, rtol=threshold) & utils3d.numpy.normals_edge(normal, tol=2))
        edge_mask =  ~(utils3d.numpy.depth_edge(depth, rtol=threshold) & utils3d.numpy.normals_edge(normal, tol=3))
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        #edge_mask = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        # edge_mask = np.ones_like(edge_mask)
        bg_indices = edge_mask & (~mask) & (~obj_mask)

        H, W = depth.shape
        points_bg_map = np.zeros((H, W, 3), dtype=np.float32)

        # 2.  Scatter the background points into the map
        # ── Case A: bg_indices has shape (H, W) ────────────────────────────────

        if world_coords:
            # Apply only the rotation (normals are direction vectors)
            normal = np.einsum("ij,hwj->hwi", rotation, normal)

   
        
        bg_homo_grid = np.pad(grid[bg_indices], ((0, 0), (0, 1)), constant_values=1)
        local_dirs_bg = np.einsum("ij,bj->bi", np.linalg.inv(K), bg_homo_grid)
        dirs_bg = np.einsum("ij,bj->bi", rotation, local_dirs_bg)
        points_bg = translation + dirs_bg * depth[bg_indices, None]
        points_bg = points_bg.astype(np.float32)
        point_colors_bg = rgb[bg_indices]
         
        points_bg_map[bg_indices] = points_bg      # points_bg is N×3
        depth[~bg_indices] = 0 
        normal[~bg_indices] = 0
            
        # Optionally downsample background points
        if bg_downsample_factor > 1 and points_bg.shape[0] > 0:
            indices = np.random.choice(
                points_bg.shape[0],
                size=points_bg.shape[0] // bg_downsample_factor,
                replace=False,
            )
            points_bg = points_bg[indices]
            point_colors_bg = point_colors_bg[indices]
     # normal[mask] = 0.0
        extras = [depth, rotation, translation, K, points_bg_map, plane_info]
        if plane_info is not None:
            plane_info['depth_map_resized'] = depth.copy() if isinstance(depth, np.ndarray) else depth.clone()
            plane_info['normal_map_resized'] = normal.copy() if isinstance(normal, np.ndarray) else normal.clone()
            plane_info['rgb_resized'] = rgb.copy() if isinstance(rgb, np.ndarray) else rgb.clone()
        return (
            points_bg,
            normal,
            self.rgb[::downsample_factor, ::downsample_factor],
            point_colors_bg,
            extras,
        )
