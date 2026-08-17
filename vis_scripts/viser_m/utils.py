import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
import torch
from typing import Dict, List, Tuple, Any
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import MiniBatchKMeans
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
from collections import defaultdict
import re
import json
import os

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import hsv_to_rgb
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

from PIL import Image
import torch
import open3d as o3d
import open3d.core as o3c
import numpy as np
from math import ceil
from sklearn.cluster import DBSCAN
from math import ceil
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import colorsys
from sklearn.cluster import KMeans
from pathlib import Path
import copy
from math import ceil
import colorsys
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from typing import Tuple
import numpy as np
from scipy.spatial import cKDTree

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from copy import copy
from typing import Dict, List

_PRIM_KEYS = ["S_items", "R_items", "T_items", "pts_items"]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


from pytorch3d.utils import ico_sphere
from pytorch3d.renderer import TexturesVertex, TexturesUV
def signed_pow(t, exponent):
    return torch.sign(t) * (torch.abs(t).pow(exponent))
def parametric_sq(eta, omega, eps1, eps2):
    cos_eta, sin_eta = signed_pow(torch.cos(eta), eps1), signed_pow(torch.sin(eta), eps1)
    cos_omega, sin_omega = signed_pow(torch.cos(omega), eps2), signed_pow(torch.sin(omega), eps2)
    points = torch.stack([cos_eta * sin_omega, sin_eta, cos_eta * cos_omega], dim=-1)
    return points

def get_icosphere(level=3, order_verts_by=None, colored=False, flip_faces=False):
    mesh = ico_sphere(level)
    if order_verts_by is not None:
        assert isinstance(order_verts_by, int)
        verts, faces = mesh.get_mesh_verts_faces(0)
        N = len(verts)
        indices = sorted(range(N), key=lambda i: verts[i][order_verts_by])
        mapping = torch.zeros(N, dtype=torch.long)
        mapping[indices] = torch.arange(N)
        verts.copy_(verts[indices]), faces.copy_(mapping[faces])

    if flip_faces:
        verts, faces = mesh.get_mesh_verts_faces(0)
        faces = torch.stack([faces[:, 2], faces[:, 1], faces[:, 0]], dim=-1)
        mesh = Meshes(verts[None], faces[None])

    if colored:
        verts = mesh.verts_packed()
        colors = (verts - verts.min(0)[0]) / (verts.max(0)[0] - verts.min(0)[0])
        mesh.textures = TexturesVertex(verts_features=colors[None])
    return mesh

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
import torch
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import DBSCAN
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
from collections import defaultdict
import re
from typing import Dict, List, Optional, Any
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import hsv_to_rgb
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

from PIL import Image
import torch
import open3d as o3d
import open3d.core as o3c
import numpy as np
from math import ceil
from sklearn.cluster import DBSCAN
from math import ceil
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import colorsys
from sklearn.cluster import KMeans
from pathlib import Path
import copy
from math import ceil
import colorsys
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from typing import Tuple
import numpy as np
from scipy.spatial import cKDTree

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from copy import copy
from typing import Dict, List

_PRIM_KEYS = ["S_items", "R_items", "T_items", "pts_items"]

def _as_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return v
    # if a single tensor/array sneaks in, wrap it
    return [v]

def merge_primitives_dicts(a: Dict, b: Dict, keys: List[str] = None) -> Dict:
    """
    Robustly merge two primitive dicts:
      {'S_items': [..], 'R_items': [..], 'T_items': [..], 'pts_items': [..]}
    Other keys are passed through from `a` (and added from `b` if missing).
    """
    if keys is None:
        keys = _PRIM_KEYS

    out = {}

    # First copy everything from a (shallow is fine since we'll replace list keys)
    for k, v in (a or {}).items():
        out[k] = v

    # Add any extra non-primitive keys from b that aren't present
    for k, v in (b or {}).items():
        if k not in keys and k not in out:
            out[k] = v

    # Now merge list-valued primitive keys
    for k in keys:
        la = _as_list((a or {}).get(k))
        lb = _as_list((b or {}).get(k))
        # Force creation of a brand-new list object
        out[k] = list(la) + list(lb)

    # Sanity check: lengths should match across S/R/T
    nS, nR, nT = len(out["S_items"]), len(out["R_items"]), len(out["T_items"])
    if not (nS == nR == nT):
        raise ValueError(f"Merged lengths mismatch: |S|={nS}, |R|={nR}, |T|={nT}")

    return out

def build_global_segments_single_view(all_frame_segments):
    """
    Build global_segments for a single fixed image.
    
    Args:
        all_frame_segments: List with one element containing the segments from the single frame
                           Format: [{ seg_id: {properties}, ... }]
    
    Returns:
        global_segments: Dict mapping global_id to list of (frame_idx, local_seg_id) tuples
    """
    global_segments = {}
    
    # Since we have only one frame (index 0)
    frame_idx = 0
    frame_segments = all_frame_segments[0]  # Get the single frame's segments
    
    # Each local segment becomes its own global segment
    global_id = 0
    for local_seg_id in frame_segments.keys():
        global_segments[global_id] = [(frame_idx, local_seg_id)]
        global_id += 1
    
    return global_segments

def visualize_single_view_results(
    seg_map: np.ndarray,
    seg_props: Dict[int, Dict],
    global_segments: Dict[int, List[Tuple[int, int]]],
    primitives: Dict[str, List],
    depth_image: torch.Tensor,
    normal_image: torch.Tensor,
    save_dir: Path,
    frame_idx: int = 0,
    rgb_image: Optional[np.ndarray] = None,
    show_3d: bool = True
) -> None:
    """
    Comprehensive visualization for single view segmentation and primitive fitting.
    
    Args:
        seg_map: (H, W) segmentation map
        seg_props: Dictionary of segment properties
        global_segments: Global segment groupings
        primitives: Fitted primitives (S_items, R_items, T_items, pts_items)
        depth_image: (H, W) depth map
        normal_image: (H, W, 3) normal map
        save_dir: Directory to save visualizations
        frame_idx: Frame index for labeling
        rgb_image: Optional RGB image for overlay
        show_3d: Whether to generate 3D visualizations
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.patches import Rectangle
    import matplotlib.patches as mpatches
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    H, W = seg_map.shape
    
    # Helper function to generate distinct colors
    def get_colors(n):
        if n <= 20:
            return cm.get_cmap('tab20')(np.linspace(0, 1, n))
        else:
            return cm.get_cmap('hsv')(np.linspace(0, 0.9, n))
    
    # 1. Create main visualization figure
    fig = plt.figure(figsize=(20, 12))
    
    # 2. Original RGB/Depth view
    ax1 = plt.subplot(2, 4, 1)
    if rgb_image is not None:
        ax1.imshow(rgb_image)
        ax1.set_title('RGB Image')
    else:
        depth_vis = depth_image.cpu().numpy()
        depth_vis = np.clip(depth_vis, 0, np.percentile(depth_vis[depth_vis > 0], 95))
        ax1.imshow(depth_vis, cmap='viridis')
        ax1.set_title('Depth Map')
    ax1.axis('off')
    
    # 3. Normal visualization
    ax2 = plt.subplot(2, 4, 2)
    normal_vis = (normal_image.cpu().numpy() + 1) / 2  # Convert from [-1,1] to [0,1]
    normal_vis = np.clip(normal_vis, 0, 1)
    ax2.imshow(normal_vis)
    ax2.set_title('Surface Normals')
    ax2.axis('off')
    
    # 4. Per-pixel segmentation
    ax3 = plt.subplot(2, 4, 3)
    unique_segments = np.unique(seg_map)
    valid_segments = unique_segments[unique_segments >= 0]
    
    if len(valid_segments) > 0:
        colors = get_colors(len(valid_segments))
        seg_colored = np.zeros((*seg_map.shape, 3))
        for idx, seg_id in enumerate(valid_segments):
            mask = seg_map == seg_id
            seg_colored[mask] = colors[idx][:3]
    else:
        seg_colored = np.zeros((*seg_map.shape, 3))
    
    ax3.imshow(seg_colored)
    ax3.set_title(f'Segmentation ({len(valid_segments)} segments)')
    ax3.axis('off')
    
    # 5. Segment properties overlay
    ax4 = plt.subplot(2, 4, 4)
    ax4.imshow(seg_colored)
    
    # Draw segment centroids and IDs
    for seg_id, props in seg_props.items():
        # Get 2D centroid from pixels
        pixels = np.array(props['pixels'])
        if len(pixels) > 0:
            centroid_2d = pixels.mean(axis=0)
            ax4.plot(centroid_2d[1], centroid_2d[0], 'w*', markersize=8, markeredgecolor='k')
            ax4.text(centroid_2d[1], centroid_2d[0]-5, str(seg_id), 
                    color='white', fontsize=8, ha='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    ax4.set_title('Segment IDs & Centroids')
    ax4.axis('off')
    
    # 6. Primitive visualization (if available)
    ax5 = plt.subplot(2, 4, 5)
    if 'S_items' in primitives and len(primitives['S_items']) > 0:
        # Create a primitive overlay visualization
        prim_overlay = np.zeros((*seg_map.shape, 3))
        prim_colors = get_colors(len(primitives['S_items']))
        
        # Map primitives back to segments
        for prim_idx, (S, R, T) in enumerate(zip(primitives['S_items'], 
                                                  primitives['R_items'], 
                                                  primitives['T_items'])):
            # Find which segments belong to this primitive
            # This is simplified - you might need to track the mapping
            color = prim_colors[prim_idx][:3]
            
            # For visualization, we'll color segments that were merged into this primitive
            # This requires tracking which global segment produced which primitive
            if prim_idx < len(valid_segments):
                mask = seg_map == valid_segments[prim_idx]
                prim_overlay[mask] = color
        
        ax5.imshow(prim_overlay)
        ax5.set_title(f'Fitted Primitives ({len(primitives["S_items"])} boxes)')
    else:
        ax5.imshow(seg_colored)
        ax5.set_title('No Primitives Fitted')
    ax5.axis('off')
    
    # 7. Statistics panel
    ax6 = plt.subplot(2, 4, 6)
    ax6.axis('off')
    
    stats_text = f"Frame: {frame_idx}\n"
    stats_text += f"Resolution: {W}×{H}\n"
    stats_text += f"Segments: {len(valid_segments)}\n"
    stats_text += f"Primitives: {len(primitives.get('S_items', []))}\n\n"
    
    # Segment statistics
    stats_text += "Segment Sizes:\n"
    for seg_id in valid_segments[:5]:  # Show first 5
        if seg_id in seg_props:
            n_pixels = len(seg_props[seg_id]['pixels'])
            stats_text += f"  Seg {seg_id}: {n_pixels} pixels\n"
    
    if len(valid_segments) > 5:
        stats_text += f"  ... and {len(valid_segments)-5} more\n"
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 8. Normal clustering visualization
    ax7 = plt.subplot(2, 4, 7)
    
    # Show normal clusters by coloring segments based on their average normal
    normal_cluster_vis = np.zeros((*seg_map.shape, 3))
    for seg_id, props in seg_props.items():
        mask = seg_map == seg_id
        # Convert normal to RGB color (shift from [-1,1] to [0,1])
        normal_color = (props['avg_normal'].cpu().numpy() + 1) / 2
        normal_cluster_vis[mask] = normal_color
    
    ax7.imshow(normal_cluster_vis)
    ax7.set_title('Normal-based Coloring')
    ax7.axis('off')
    
    # 9. Boundary visualization
    ax8 = plt.subplot(2, 4, 8)
    
    # Create boundary image
    from scipy import ndimage
    boundary_img = np.zeros((H, W))
    for seg_id in valid_segments:
        mask = seg_map == seg_id
        eroded = ndimage.binary_erosion(mask)
        boundary = mask & ~eroded
        boundary_img[boundary] = 1
    
    ax8.imshow(seg_colored)
    ax8.contour(boundary_img, colors='white', linewidths=1)
    ax8.set_title('Segment Boundaries')
    ax8.axis('off')
    
    plt.suptitle(f'Single View Segmentation Results - Frame {frame_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save main figure
    main_save_path = save_dir / 'single_view_complete.png'
    plt.savefig(main_save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved main visualization to {main_save_path}")
    
    # Additional individual visualizations
    
    # 10. Save individual segmentation map with legend
    fig2, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(seg_colored)
    ax.set_title(f'Segmentation Map - {len(valid_segments)} Segments')
    ax.axis('off')
    
    # Create legend
    if len(valid_segments) <= 20:
        patches = []
        colors = get_colors(len(valid_segments))
        for idx, seg_id in enumerate(valid_segments):
            if seg_id in seg_props:
                n_pixels = len(seg_props[seg_id]['pixels'])
                label = f'Seg {seg_id} ({n_pixels} px)'
                patches.append(mpatches.Patch(color=colors[idx][:3], label=label))
        
        ax.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5), 
                 fontsize=8, title='Segments')
    
    seg_save_path = save_dir / 'segmentation_with_legend.png'
    plt.savefig(seg_save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved segmentation with legend to {seg_save_path}")

def filter_bg_points_by_human_distance(
    bg_position: np.ndarray,
    bg_color: np.ndarray,
    human_transl_np: np.ndarray,
    depth,
    real_normal: np.ndarray | None = None,
    max_dist: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    """
    bg_position = np.asarray(bg_position, dtype=np.float32)
    bg_color = np.asarray(bg_color)
    human_transl_np = np.asarray(human_transl_np, dtype=np.float32)

    if human_transl_np.ndim != 2 or human_transl_np.shape[1] != 3:
        raise ValueError(f"Expected human_transl_np with shape [T, 3], got {human_transl_np.shape}")

    flat_points = bg_position.reshape(-1, 3)
    flat_colors = bg_color.reshape(-1, bg_color.shape[-1]) if bg_color.ndim >= 2 else np.asarray(bg_color).reshape(-1, 1)
    valid_points = np.isfinite(flat_points).all(axis=1) & (np.linalg.norm(flat_points, axis=1) > 1e-8)

    keep_flat = np.zeros((flat_points.shape[0],), dtype=bool)
    if np.any(valid_points):
        tree = cKDTree(human_transl_np)
        distances, _ = tree.query(flat_points[valid_points], k=1)
        keep_valid = distances <= max_dist
        keep_flat[np.where(valid_points)[0][keep_valid]] = True

    if depth is not None and bg_position.ndim == 3 and depth.shape[:2] == bg_position.shape[:2]:
        keep_mask_hw = keep_flat.reshape(bg_position.shape[:2])
        depth[~keep_mask_hw] = 0

    points_kept = flat_points[keep_flat]
    colors_kept = flat_colors[keep_flat]

    if real_normal is None:
        return points_kept, colors_kept, depth

    normals = np.asarray(real_normal, dtype=np.float32)
    flat_normals = normals.reshape(-1, 3)
    normals_kept = flat_normals[keep_flat]
    return points_kept, colors_kept, depth, normals_kept

def cluster_normals(
    normals: torch.Tensor,
    n_clusters: int = 6,
    n_init_normal_clusters: int = 10,
    return_rgb: bool = True,
):
    """
    Returns
    -------
    labels  : (T,H,W) int64   – invalid pixels == ‑1
    rgb_vis : (T,H,W,3) uint8 – mid‑gray where label == ‑1   (only if return_rgb)
    """
    if normals.dim() != 4:
        raise ValueError("normals must be 4‑D (T,C,H,W) or (T,H,W,C)")

    # to (T,H,W,3)
    if normals.shape[1] == 3:
        normals = normals.permute(0, 2, 3, 1).contiguous()
    T, H, W, _ = normals.shape
    device = normals.device

    flat = normals.reshape(-1, 3).cpu().numpy()
    total_px = flat.shape[0]

    # ---------- validity test ----------
    mag = np.linalg.norm(flat, axis=1)
    finite = np.isfinite(flat).all(axis=1)
    unit   = np.abs(mag - 1.0) < 1e-2
    valid_mask = finite & unit
    valid_idx  = np.where(valid_mask)[0]
    valid_flat = flat[valid_mask]
    INVALID = -1
    labels_flat = np.full(total_px, INVALID, dtype=np.int64)
    # ------------------------------------

    if valid_flat.size:
        kmeans = KMeans(n_clusters=n_init_normal_clusters,
                        n_init=1, random_state=0).fit(valid_flat)
        pred = kmeans.labels_
        centres = kmeans.cluster_centers_
        counts = np.bincount(pred)

        topk = np.argpartition(counts, -n_clusters)[-n_clusters:]
        topk = topk[np.argsort(counts[topk])[::-1]]
        pred, topk, n_valid = merge_normal_clusters(pred, topk, centres)

        mapping = {old: new for new, old in enumerate(topk[:n_valid])}
        relabel = np.full(pred.shape, INVALID, dtype=np.int64)
        for old, new in mapping.items():
            relabel[pred == old] = new

        labels_flat[valid_idx] = relabel
    else:
        n_valid = 0

    labels = torch.from_numpy(labels_flat).to(device).long().reshape(T, H, W)

    # ---------- RGB visualisation ----------
    if not return_rgb:
        return labels, None

    # colour list for valid clusters
    colours = (np.round(_generate_distinct_colors(n_valid) * 255)
               .astype(np.uint8))
    gray    = np.array([128, 128, 128], dtype=np.uint8)

    rgb_vis = np.empty((T, H, W, 3), dtype=np.uint8)
    # fill gray
    rgb_vis[:] = gray
    # overwrite with cluster colours
    for cid in range(n_valid):
        mask = (labels.cpu().numpy() == cid)
        rgb_vis[mask] = colours[cid]

    rgb_vis = torch.from_numpy(rgb_vis)      # keep on CPU, uint8
    return labels, rgb_vis


def segment_by_dbscan_batch_corrected(
    cluster_labels_3d: np.ndarray,
    eps_spatial: float = 3.0,
    min_samples: int = 30,
    temporal_weight: float = 1.0,
    min_final_points: int = 1400,       # NEW ─ keep only segments ≥ this many points
    visualize: bool = True,
    max_cols: int = 5,
):
    T, H, W = cluster_labels_3d.shape
    seg_map_3d = -np.ones_like(cluster_labels_3d, dtype=np.int32)

    unique_clusters = np.unique(cluster_labels_3d)
    valid_clusters = unique_clusters[unique_clusters >= 0]   # <‑‑ ignore ‑1
    global_id = 0

    for cluster_id in valid_clusters:
        mask = cluster_labels_3d == cluster_id
        t, y, x = np.where(mask)
        if len(t) < min_samples:
            continue

        feat = np.column_stack([t * temporal_weight, y, x])
        
        db = DBSCAN(eps=eps_spatial, min_samples=min_samples).fit(feat)

        for local_seg in np.unique(db.labels_):
            if local_seg < 0:
                continue
            local_mask = db.labels_ == local_seg
            seg_map_3d[t[local_mask], y[local_mask], x[local_mask]] = global_id
            global_id += 1

    for seg_id in np.unique(seg_map_3d):
        if seg_id < 0:
            continue
        count = np.count_nonzero(seg_map_3d == seg_id)
        if count < min_final_points:
            seg_map_3d[seg_map_3d == seg_id] = -1  # drop small final clusters


    return seg_map_3d



def build_global_segments_greedy(
    all_frame_segments: List[Dict[int, Dict]],
    correspondence_pairs: List[Tuple[int, int, int, int]]
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Union-find over *all* correspondence pairs (frame_i, seg_i, frame_j, seg_j).
    Returns {global_id: [(frame_idx, local_seg_id), …]}.
    """
    parent = {}
    def key(fi, sid): return f"{fi}_{sid}"
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    # initialise every local segment
    for fi, segs in enumerate(all_frame_segments):
        for sid in segs:
            find(key(fi, sid))

    # greedy unions
    for fi, sid_i, fj, sid_j in correspondence_pairs:
        union(key(fi, sid_i), key(fj, sid_j))

    # collect & renumber
    groups = defaultdict(list)
    for fi, segs in enumerate(all_frame_segments):
        for sid in segs:
            groups[find(key(fi, sid))].append((fi, sid))

    return {gid: members for gid, members in enumerate(groups.values())}

def visualize_per_frame_segments(
    all_seg_maps: List[np.ndarray],
    frame_indices: List[int],
    save_dir: Path,
    max_cols: int = 4
) -> None:
    """
    Visualize segmentation maps for each frame.
    Each segment gets a unique color within its frame.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    num_frames = len(all_seg_maps)
    num_rows = (num_frames + max_cols - 1) // max_cols
    
    fig, axes = plt.subplots(num_rows, max_cols, figsize=(4*max_cols, 4*num_rows))
    if num_frames == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    for idx, (seg_map, frame_idx) in enumerate(zip(all_seg_maps, frame_indices)):
        ax = axes_flat[idx]
        
        # Create colored visualization
        unique_segments = np.unique(seg_map)
        valid_segments = unique_segments[unique_segments >= 0]
        
        # Create color map
        if len(valid_segments) > 0:
            colors = cm.get_cmap('tab20')(np.linspace(0, 1, len(valid_segments)))
            
            # Create RGB image
            rgb_image = np.zeros((*seg_map.shape, 3))
            for seg_idx, seg_id in enumerate(valid_segments):
                mask = seg_map == seg_id
                rgb_image[mask] = colors[seg_idx][:3]
        else:
            rgb_image = np.zeros((*seg_map.shape, 3))
        
        ax.imshow(rgb_image)
        ax.set_title(f'Frame {frame_idx}\n{len(valid_segments)} segments')
        ax.axis('off')
        fig.savefig('_per_frame.png')
    
    # Hide unused subplots
    for idx in range(num_frames, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.tight_layout()
    save_path = save_dir / 'per_frame_segments.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-frame segments visualization to {save_path}")


def visualize_merged_segments(
    all_seg_maps: List[np.ndarray],
    global_segments: Dict[int, List[Tuple[int, int]]],
    frame_indices: List[int],
    save_dir: Path,
    max_cols: int = 4
) -> None:
    """
    Visualize merged segments across frames.
    Segments that are merged across frames share the same color.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    num_frames = len(all_seg_maps)
    num_rows = (num_frames + max_cols - 1) // max_cols
    
    fig, axes = plt.subplots(num_rows, max_cols, figsize=(4*max_cols, 4*num_rows))
    if num_frames == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes_flat = axes.flatten()
    
    # Assign colors to global segments
    num_global_segments = len(global_segments)
    if num_global_segments > 0:
        # Use a colormap with enough distinct colors
        if num_global_segments <= 20:
            global_colors = cm.get_cmap('tab20')(np.linspace(0, 1, num_global_segments))
        else:
            global_colors = cm.get_cmap('hsv')(np.linspace(0, 0.9, num_global_segments))
    
    # Create merged visualization for each frame
    for frame_idx_pos, (seg_map, frame_idx) in enumerate(zip(all_seg_maps, frame_indices)):
        ax = axes_flat[frame_idx_pos]
        
        # Create RGB image
        rgb_image = np.zeros((*seg_map.shape, 3))
        
        # Fill in colors based on global segment membership
        segments_in_frame = 0
        for global_id, members in global_segments.items():
            # Check if this global segment appears in current frame
            for member_frame_idx, local_seg_id in members:
                if member_frame_idx == frame_idx_pos:  # Use position index, not frame number
                    mask = seg_map == local_seg_id
                    if mask.any():
                        rgb_image[mask] = global_colors[global_id][:3]
                        segments_in_frame += 1
        
        ax.imshow(rgb_image)
        ax.set_title(f'Frame {frame_idx}\n{segments_in_frame} global segments')
        ax.axis('off')
        fig.savefig('_global_frame.png')
    
    # Hide unused subplots
    for idx in range(num_frames, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.tight_layout()
    save_path = save_dir / 'merged_segments.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved merged segments visualization to {save_path}")


def save_segment_statistics(
    all_seg_maps: List[np.ndarray],
    all_frame_segments: List[Dict[int, Dict]],
    global_segments: Dict[int, List[Tuple[int, int]]],
    all_frame_correspondences: List[List[Tuple[int, int]]],
    frame_indices: List[int],
    save_dir: Path
) -> None:
    """
    Save detailed statistics about segments for debugging.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    stats_file = save_dir / 'segment_statistics.txt'
    
    with open(stats_file, 'w') as f:
        f.write("=== SEGMENT STATISTICS ===\n\n")
        
        # Per-frame statistics
        f.write("PER-FRAME SEGMENTS:\n")
        for idx, (seg_map, frame_idx) in enumerate(zip(all_seg_maps, frame_indices)):
            segments = all_frame_segments[idx]
            f.write(f"\nFrame {frame_idx} (index {idx}):\n")
            f.write(f"  Total segments: {len(segments)}\n")
            
            for seg_id, props in segments.items():
                num_pixels = len(props['pixels'])
                avg_normal = props['avg_normal'].cpu().numpy()
                centroid = props['centroid'].cpu().numpy()
                f.write(f"    Segment {seg_id}: {num_pixels} pixels\n")
                f.write(f"      Normal: [{avg_normal[0]:.3f}, {avg_normal[1]:.3f}, {avg_normal[2]:.3f}]\n")
                f.write(f"      Centroid: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]\n")
        
        # Correspondence statistics
        f.write("\n\nCORRESPONDENCES:\n")
        for idx, correspondences in enumerate(all_frame_correspondences):
            if idx < len(frame_indices) - 1:
                f.write(f"\nFrame {frame_indices[idx]} -> Frame {frame_indices[idx+1]}:\n")
                f.write(f"  Found {len(correspondences)} matches\n")
                for seg1, seg2 in correspondences:
                    f.write(f"    Segment {seg1} -> Segment {seg2}\n")
        
        # Global segment statistics
        f.write("\n\nGLOBAL SEGMENTS:\n")
        f.write(f"Total global segments: {len(global_segments)}\n\n")
        
        # Sort by number of frames (descending)
        sorted_global = sorted(global_segments.items(), 
                             key=lambda x: len(x[1]), reverse=True)
        
        for global_id, members in sorted_global:
            f.write(f"\nGlobal Segment {global_id}:\n")
            f.write(f"  Appears in {len(members)} frames\n")
            f.write(f"  Members:\n")
            for frame_idx, local_seg_id in members:
                actual_frame = frame_indices[frame_idx]
                f.write(f"    Frame {actual_frame} (idx {frame_idx}), Local segment {local_seg_id}\n")
        
        # Find segments that appear in multiple frames
        multi_frame_segments = [gid for gid, members in global_segments.items() if len(members) >= 2]
        f.write(f"\n\nSegments appearing in 2+ frames: {len(multi_frame_segments)}\n")
        
    print(f"Saved segment statistics to {stats_file}")


def debug_flow_matching(
    seg_props1: Dict[int, Dict],
    seg_props2: Dict[int, Dict],
    flow_forward: np.ndarray,
    covis_mask: np.ndarray,
    frame1_idx: int,
    frame2_idx: int,
    save_dir: Path
) -> None:
    """
    Debug flow matching between two specific frames.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    H, W = flow_forward.shape[:2]
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Show flow magnitude
    flow_mag = np.sqrt(flow_forward[..., 0]**2 + flow_forward[..., 1]**2)
    im1 = axes[0, 0].imshow(flow_mag, cmap='viridis')
    axes[0, 0].set_title(f'Flow Magnitude\nFrame {frame1_idx} -> {frame2_idx}')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Show covisibility mask
    im2 = axes[0, 1].imshow(covis_mask, cmap='gray')
    axes[0, 1].set_title('Covisibility Mask')
    
    # Debug text
    debug_text = f"Frame {frame1_idx} -> {frame2_idx}\n\n"
    debug_text += f"Segments in frame 1: {len(seg_props1)}\n"
    debug_text += f"Segments in frame 2: {len(seg_props2)}\n\n"
    
    # Check each segment pair
    for seg1_id, props1 in seg_props1.items():
        for seg2_id, props2 in seg_props2.items():
            # Normal similarity
            normal_sim = torch.dot(props1['avg_normal'], props2['avg_normal']).item()
            
            # Flow-based overlap
            overlap_count = 0
            valid_flow_pixels = 0
            
            for y, x in props1['boundary_pixels'][:20]:  # Sample first 20 pixels
                if covis_mask[y, x]:
                    print(flow_forward.shape, y, x)
                    flow = flow_forward[y, x]
                    
                    y_new = int(round(y + flow[1]))
                    x_new = int(round(x + flow[0]))
                    
                    if 0 <= y_new < H and 0 <= x_new < W:
                        valid_flow_pixels += 1
                        if (y_new, x_new) in props2['pixels']:
                            overlap_count += 1
            
            if valid_flow_pixels > 0:
                overlap_ratio = overlap_count / valid_flow_pixels
                if normal_sim > 0.9 or overlap_ratio > 0.1:
                    debug_text += f"Seg {seg1_id} -> Seg {seg2_id}: "
                    debug_text += f"normal_sim={normal_sim:.3f}, "
                    debug_text += f"overlap={overlap_count}/{valid_flow_pixels} "
                    debug_text += f"({overlap_ratio:.2f})\n"
    
    # Show debug text
    axes[1, 0].text(0.05, 0.95, debug_text, transform=axes[1, 0].transAxes,
                    verticalalignment='top', fontsize=8, family='monospace')
    axes[1, 0].axis('off')
    
    # Hide last subplot
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    save_path = save_dir / f'flow_debug_{frame1_idx}_{frame2_idx}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved flow debug to {save_path}")

def load_flow_and_covis_data(
    data_dir: Path,
    frame_indices: List[int],
    interval: int = 7
) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], np.ndarray]]:
    """
    Load flow and covisibility data for given frame indices.
    
    Returns
    -------
    flow_forward : dict mapping (frame_i, frame_j) -> flow array
    flow_backward : dict mapping (frame_j, frame_i) -> flow array  
    covis_masks : dict mapping (frame_i, frame_j) -> covisibility mask
    """
    flow_forward = {}
    flow_backward = {}
    covis_masks = {}
    
    # Load covisibility masks between interval frames
    for i in range(len(frame_indices) - 1):
        frame_i = frame_indices[i]
        frame_j = frame_indices[i + 1]
        
        # Try to find covisibility file
        covis_file = data_dir / f"covis_{frame_i:05d}_{frame_j:05d}.npy"
        if covis_file.exists():
            covis_masks[(frame_i, frame_j)] = np.load(covis_file)
    
    # Load flow data - for now, we'll use direct flow if available
    # In practice, you might need to accumulate flow over intervals
    for i in range(len(frame_indices) - 1):
        frame_i = frame_indices[i]
        frame_j = frame_indices[i + 1]
        
        # For interval=7, we might need to accumulate flow
        # For now, let's check if direct flow exists
        flow_file_forward = data_dir / f"flow_{frame_i:05d}_{frame_j:05d}.npy"
        flow_file_backward = data_dir / f"flow_{frame_j:05d}_{frame_i:05d}.npy"
        
        if flow_file_forward.exists():
            flow_forward[(frame_i, frame_j)] = np.load(flow_file_forward)
        if flow_file_backward.exists():
            flow_backward[(frame_j, frame_i)] = np.load(flow_file_backward)
    
    return flow_forward, flow_backward, covis_masks


def flow_to_color(flow, max_magnitude=None):
    """
    Convert optical flow to HSV color representation.
    
    Parameters:
    -----------
    flow : np.ndarray, shape (H, W, 2)
        Optical flow with u, v components
    max_magnitude : float, optional
        Maximum magnitude for normalization. If None, uses flow max.
        
    Returns:
    --------
    color_image : np.ndarray, shape (H, W, 3)
        RGB color representation of flow
    """
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    
    # Convert to polar coordinates
    magnitude = np.sqrt(fx**2 + fy**2)
    angle = np.arctan2(fy, fx)
    
    # Normalize angle to [0, 1] for hue
    hue = (angle + np.pi) / (2 * np.pi)
    
    # Normalize magnitude for saturation
    if max_magnitude is None:
        max_magnitude = np.max(magnitude)
    
    saturation = np.clip(magnitude / max_magnitude, 0, 1)
    value = np.ones_like(hue)
    
    # Create HSV image
    hsv = np.stack([hue, saturation, value], axis=-1)
    
    # Convert to RGB
    rgb = hsv_to_rgb(hsv)
    
    return rgb


def visualize_flow(flow, title="Optical Flow", save_path=None, figsize=(12, 8), 
                  max_magnitude=None, show_colorbar=True, show_arrows=False, 
                  arrow_step=20):
    """
    Visualize optical flow as HSV color image.
    
    Parameters:
    -----------
    flow : np.ndarray, shape (H, W, 2)
        Optical flow with u, v components
    title : str
        Title for the plot
    save_path : str or Path, optional
        Path to save the visualization
    figsize : tuple
        Figure size
    max_magnitude : float, optional
        Maximum magnitude for color scaling
    show_colorbar : bool
        Whether to show magnitude colorbar
    show_arrows : bool
        Whether to overlay flow arrows
    arrow_step : int
        Step size for arrow sampling
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # HSV color representation
    color_image = flow_to_color(flow, max_magnitude)
    ax1.imshow(color_image)
    ax1.set_title(f"{title} - HSV Representation")
    ax1.axis('off')
    
    # Magnitude representation
    magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    im = ax2.imshow(magnitude, cmap='viridis')
    ax2.set_title(f"{title} - Magnitude")
    ax2.axis('off')
    
    if show_colorbar:
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # Overlay arrows if requested
    if show_arrows:
        h, w = flow.shape[:2]
        y, x = np.mgrid[0:h:arrow_step, 0:w:arrow_step]
        u = flow[::arrow_step, ::arrow_step, 0]
        v = flow[::arrow_step, ::arrow_step, 1]
        
        ax1.quiver(x, y, u, v, angles='xy', scale_units='xy', 
                  scale=1, color='white', width=0.002, alpha=0.7)
        ax2.quiver(x, y, u, v, angles='xy', scale_units='xy', 
                  scale=1, color='white', width=0.002, alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Flow visualization saved to: {save_path}")
    
    plt.show()


def visualize_covisibility(covis_mask, title="Covisibility Mask", save_path=None, 
                          figsize=(10, 6), cmap='viridis'):
    """
    Visualize covisibility mask.
    
    Parameters:
    -----------
    covis_mask : np.ndarray, shape (H, W)
        Binary or continuous covisibility mask
    title : str
        Title for the plot
    save_path : str or Path, optional
        Path to save the visualization
    figsize : tuple
        Figure size
    cmap : str
        Colormap for visualization
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    im = ax.imshow(covis_mask, cmap=cmap)
    ax.set_title(title)
    ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Covisibility')
    
    # Add statistics text
    stats_text = f"Min: {covis_mask.min():.3f}\n"
    stats_text += f"Max: {covis_mask.max():.3f}\n"
    stats_text += f"Mean: {covis_mask.mean():.3f}\n"
    stats_text += f"Shape: {covis_mask.shape}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='white', alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Covisibility visualization saved to: {save_path}")
    
    plt.show()


def create_flow_colorwheel(size=256, save_path=None):
    """
    Create and save a flow color wheel for reference.
    
    Parameters:
    -----------
    size : int
        Size of the color wheel
    save_path : str or Path, optional
        Path to save the color wheel
    """
    # Create coordinate grids
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Create flow field (circular)
    flow = np.stack([X, Y], axis=-1)
    
    # Mask for circular region
    radius = np.sqrt(X**2 + Y**2)
    mask = radius <= 1
    
    # Convert to color
    color_wheel = flow_to_color(flow, max_magnitude=1)
    color_wheel[~mask] = 1  # White background outside circle
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(color_wheel)
    ax.set_title("Flow Color Wheel\n(Hue = Direction, Saturation = Magnitude)", 
                fontsize=14)
    ax.axis('off')
    
    # Add direction labels
    ax.text(size//2, 10, "↑", ha='center', va='center', fontsize=20, fontweight='bold')
    ax.text(size//2, size-10, "↓", ha='center', va='center', fontsize=20, fontweight='bold')
    ax.text(10, size//2, "←", ha='center', va='center', fontsize=20, fontweight='bold')
    ax.text(size-10, size//2, "→", ha='center', va='center', fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Flow color wheel saved to: {save_path}")
    
    plt.show()


def batch_visualize_flow_data(flow_forward, flow_backward, covis_masks, 
                             output_dir="./flow_visualizations"):
    """
    Batch visualize all flow and covisibility data.
    
    Parameters:
    -----------
    flow_forward : dict
        Forward flow data
    flow_backward : dict  
        Backward flow data
    covis_masks : dict
        Covisibility masks
    output_dir : str or Path
        Output directory for visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Saving visualizations to: {output_dir}")
    
    # Create color wheel reference
    create_flow_colorwheel(save_path=output_dir / "flow_colorwheel.png")
    
    # Visualize forward flows
    print("\nProcessing forward flows...")
    for (frame_i, frame_j), flow in flow_forward.items():
        title = f"Forward Flow: {frame_i:05d} → {frame_j:05d}"
        save_path = output_dir / f"flow_forward_{frame_i:05d}_{frame_j:05d}.png"
        visualize_flow(flow, title=title, save_path=save_path, 
                      show_arrows=True, arrow_step=30)
        plt.close('all')  # Close to save memory
    
    # Visualize backward flows
    print("\nProcessing backward flows...")
    for (frame_j, frame_i), flow in flow_backward.items():
        title = f"Backward Flow: {frame_j:05d} → {frame_i:05d}"
        save_path = output_dir / f"flow_backward_{frame_j:05d}_{frame_i:05d}.png"
        visualize_flow(flow, title=title, save_path=save_path, 
                      show_arrows=True, arrow_step=30)
        plt.close('all')
    
    # Visualize covisibility masks
    print("\nProcessing covisibility masks...")
    for (frame_i, frame_j), mask in covis_masks.items():
        title = f"Covisibility: {frame_i:05d} ↔ {frame_j:05d}"
        save_path = output_dir / f"covis_{frame_i:05d}_{frame_j:05d}.png"
        visualize_covisibility(mask, title=title, save_path=save_path)
        plt.close('all')
    
    print(f"\nAll visualizations saved to: {output_dir}")


def generate_sample_data():
    """Generate sample flow and covisibility data for testing."""
    print("Generating sample data...")
    
    # Create sample directory
    sample_dir = Path("./sample_flow_data")
    sample_dir.mkdir(exist_ok=True)
    
    h, w = 384, 512
    
    # Generate sample flow data
    for i in range(3):
        frame_i = i * 7
        frame_j = (i + 1) * 7
        
        # Create synthetic flow field
        x = np.linspace(-2, 2, w)
        y = np.linspace(-1.5, 1.5, h)
        X, Y = np.meshgrid(x, y)
        
        # Circular flow pattern
        u = -Y * np.exp(-(X**2 + Y**2)) + np.random.normal(0, 0.1, (h, w))
        v = X * np.exp(-(X**2 + Y**2)) + np.random.normal(0, 0.1, (h, w))
        
        flow = np.stack([u, v], axis=-1)
        
        # Save flow data
        np.save(sample_dir / f"flow_{frame_i:05d}_{frame_j:05d}.npy", flow)
        np.save(sample_dir / f"flow_{frame_j:05d}_{frame_i:05d}.npy", -flow)  # Reverse flow
        
        # Generate covisibility mask
        center_x, center_y = w//2, h//2
        covis_mask = np.zeros((h, w))
        
        # Create elliptical visibility region
        for y in range(h):
            for x in range(w):
                dx = (x - center_x) / (w * 0.3)
                dy = (y - center_y) / (h * 0.3)
                if dx**2 + dy**2 <= 1:
                    covis_mask[y, x] = np.exp(-(dx**2 + dy**2))
        
        # Add some noise
        covis_mask += np.random.uniform(0, 0.1, (h, w))
        covis_mask = np.clip(covis_mask, 0, 1)
        
        np.save(sample_dir / f"covis_{frame_i:05d}_{frame_j:05d}.npy", covis_mask)
    
    print(f"Sample data generated in: {sample_dir}")
    return sample_dir


def _generate_distinct_colors(n_colors):
    """Generate visually distinct colors for visualization."""
    if n_colors == 0:
        return np.array([])
    
    hues = np.linspace(0, 1, n_colors, endpoint=False)
    colors = []
    for hue in hues:
        # Use HSV to RGB conversion for better color distribution
        rgb = plt.cm.hsv(hue)[:3]
        colors.append(rgb)
    return np.array(colors)

def process_segments_from_segmap_multiframe(
        depthmap, seg_map_3d, normals, points, 
        ratio_block_scene=1.0, min_segment_size=50, device='cuda'):

    depthmap = torch.as_tensor(depthmap, device=device, dtype=torch.float32)
    normals  = torch.as_tensor(normals,  device=device, dtype=torch.float32)
    points   = torch.as_tensor(points,   device=device, dtype=torch.float32)
    # shapes: depthmap (T,H,W), normals (T,H,W,3), points (T,H,W,3)

    T, H, W = depthmap.shape
    cluster_segments: Dict[int, Dict[str, List[torch.Tensor]]] = {}
    unique_ids = np.unique(seg_map_3d)

    for seg_id in unique_ids:
        if seg_id < 0:
            continue

        mask = (seg_map_3d == seg_id)  # (T,H,W) bool/0-1
        if int(mask.sum()) < min_segment_size:
            continue

        # indices of pixels in all frames for this seg
        t_idx, y_idx, x_idx = np.where(mask)
        if len(t_idx) < 3:
            continue

        # to torch indices
        t_idx_t = torch.as_tensor(t_idx, device=device, dtype=torch.long)
        y_idx_t = torch.as_tensor(y_idx, device=device, dtype=torch.long)
        x_idx_t = torch.as_tensor(x_idx, device=device, dtype=torch.long)

        d = depthmap[t_idx_t, y_idx_t, x_idx_t]
        valid = d > 0
        if int(valid.sum()) < 3:
            continue

        # keep only valid pixels
        t_idx_t, y_idx_t, x_idx_t = [v[valid] for v in (t_idx_t, y_idx_t, x_idx_t)]

        # gather per-pixel data for THIS segment
        seg_pts_world   = points[t_idx_t, y_idx_t, x_idx_t, :]              # (N,3)
        seg_normals_cam = normals[t_idx_t, y_idx_t, x_idx_t, :]             # (N,3)
        avg_normal_cam  = F.normalize(seg_normals_cam.mean(0), dim=0)       # (3,)

        ent = cluster_segments.setdefault(seg_id, {'pts_world': [], 'normals_world': []})
        ent['pts_world'].append(seg_pts_world)                # append this frame’s samples
        ent['normals_world'].append(avg_normal_cam.unsqueeze(0))

    return cluster_segments




def merge_normal_clusters(pred, topk, centres):
    """Merge similar normal clusters based on dot product similarity."""
    n_valid = len(topk)
    if n_valid <= 1:
        return pred, topk, n_valid
    
    # Compute pairwise similarities between cluster centers
    similarities = np.dot(centres[topk], centres[topk].T)
    
    # Merge clusters with high similarity (dot product > 0.95)
    merge_threshold = 0.95
    merged = np.arange(n_valid)
    
    for i in range(n_valid):
        for j in range(i + 1, n_valid):
            if similarities[i, j] > merge_threshold:
                # Merge cluster j into cluster i
                merged[merged == merged[j]] = merged[i]
    
    # Create mapping from old labels to new merged labels
    unique_merged = np.unique(merged)
    merge_map = {topk[old_idx]: topk[merged[old_idx]] for old_idx in range(n_valid)}
    
    # Apply merging to predictions
    new_pred = pred.copy()
    for old_label, new_label in merge_map.items():
        if old_label != new_label:
            new_pred[pred == old_label] = new_label
    
    # Get final unique clusters
    final_topk = topk[np.unique(merged)]
    n_valid = len(final_topk)
    
    return new_pred, final_topk, n_valid


def cluster_normals_single_frame(
    normals: torch.Tensor,
    valid_mask: torch.Tensor,
    n_clusters: int = 6,
    n_init_normal_clusters: int = 10,
    return_labels_map: bool = True
):
    """
    Cluster normals for a single frame.
    
    Parameters
    ----------
    normals : (H, W, 3) torch.Tensor
        Normal vectors
    valid_mask : (H, W) torch.Tensor
        Boolean mask of valid pixels
    n_clusters : int
        Target number of clusters
    n_init_normal_clusters : int
        Initial number of clusters for KMeans
    return_labels_map : bool
        If True, returns full (H,W) label map; if False, returns only valid pixel labels
    
    Returns
    -------
    labels : (H, W) int32 array if return_labels_map=True, else 1D array of valid pixel labels
        Cluster labels, -1 for invalid pixels
    n_valid_clusters : int
        Number of valid clusters found
    """
    H, W = normals.shape[:2]
    device = normals.device
    
    # Get valid normals
    valid_normals = normals[valid_mask].cpu().numpy()
    
    if return_labels_map:
        labels = np.full((H, W), -1, dtype=np.int32)
    
    if len(valid_normals) == 0:
        if return_labels_map:
            return labels, 0
        else:
            return np.array([], dtype=np.int32), 0
    
    # Check normal validity (unit vectors)
    mag = np.linalg.norm(valid_normals, axis=1)
    finite = np.isfinite(valid_normals).all(axis=1)
    unit = np.abs(mag - 1.0) < 1e-2
    normal_valid = finite & unit
    
    if not normal_valid.any():
        if return_labels_map:
            return labels, 0
        else:
            return np.full(len(valid_normals), -1, dtype=np.int32), 0
    
    # Cluster only valid normals
    valid_normals_clean = valid_normals[normal_valid]
    
    # Perform KMeans clustering
    kmeans = KMeans(
        n_clusters=min(n_init_normal_clusters, len(valid_normals_clean)),
        n_init=1,
        random_state=0
    ).fit(valid_normals_clean)
    
    pred = kmeans.labels_
    centres = kmeans.cluster_centers_
    counts = np.bincount(pred)
    
    # Select top-k clusters by size
    n_actual_clusters = min(n_clusters, len(counts))
    topk = np.argpartition(counts, -n_actual_clusters)[-n_actual_clusters:]
    topk = topk[np.argsort(counts[topk])[::-1]]
    
    # Merge similar clusters
    pred, topk, n_valid = merge_normal_clusters(pred, topk, centres)
    
    # Create relabeling map
    mapping = {old: new for new, old in enumerate(topk[:n_valid])}
    relabel = np.full(pred.shape, -1, dtype=np.int32)
    for old, new in mapping.items():
        relabel[pred == old] = new
    
    # Create output labels
    if return_labels_map:
        # Map back to full image
        valid_indices = np.where(valid_mask.cpu().numpy())
        valid_idx_clean = np.where(normal_valid)[0]
        
        # First map to valid pixels
        valid_labels = np.full(len(valid_normals), -1, dtype=np.int32)
        valid_labels[valid_idx_clean] = relabel
        
        # Then map to full image
        labels[valid_indices] = valid_labels
        return labels, n_valid
    else:
        # Return only valid pixel labels
        valid_labels = np.full(len(valid_normals), -1, dtype=np.int32)
        valid_labels[normal_valid] = relabel
        return valid_labels, n_valid


def segment_single_frame_normals(
    normal_image: torch.Tensor,
    depth_image: torch.Tensor,
    eps_spatial: float = 3.0,
    min_samples: int = 30,
    min_points: int = 500,
    n_normal_clusters: int = 6,
    n_init_normal_clusters: int = 10,
    temporal_weight: float = 0.0,  # Not used for single frame
    min_final_points: int = 1400,
    device: str = 'cuda'
) -> Tuple[np.ndarray, int]:
    """
    Segment a single frame based on normal similarity using the adapted clustering approach.
    
    Parameters
    ----------
    normal_image : (H, W, 3) torch.Tensor
        Normal vectors for the frame
    depth_image : (H, W) torch.Tensor
        Depth values for the frame
    eps_spatial : float
        DBSCAN spatial epsilon parameter
    min_samples : int
        DBSCAN minimum samples parameter
    min_points : int
        Minimum points for a valid frame
    n_normal_clusters : int
        Target number of normal clusters
    n_init_normal_clusters : int
        Initial number of clusters for KMeans
    temporal_weight : float
        Not used for single frame (kept for compatibility)
    min_final_points : int
        Minimum points for a valid segment
    device : str
        Device to use for computation
    
    Returns
    -------
    seg_map : (H, W) np.ndarray
        Segment IDs, -1 for background/invalid
    num_segments : int
        Number of segments found
    """
    H, W = normal_image.shape[:2]
    
    # Create mask of valid pixels
    valid_mask = (depth_image > 0) & torch.isfinite(normal_image).all(dim=-1)
    
    if valid_mask.sum() < min_points:
        return np.full((H, W), -1, dtype=np.int32), 0
    
    # Step 1: Cluster normals
    cluster_labels, n_valid_clusters = cluster_normals_single_frame(
        normal_image,
        valid_mask,
        n_clusters=n_normal_clusters,
        n_init_normal_clusters=n_init_normal_clusters,
        return_labels_map=True
    )
    
    if n_valid_clusters == 0:
        return np.full((H, W), -1, dtype=np.int32), 0
    
    # Step 2: Apply DBSCAN within each normal cluster for spatial segmentation
    seg_map = np.full((H, W), -1, dtype=np.int32)
    global_seg_id = 0
    
    for cluster_id in range(n_valid_clusters):
        # Get pixels belonging to this normal cluster
        cluster_mask = cluster_labels == cluster_id
        y_coords, x_coords = np.where(cluster_mask)
        
        if len(y_coords) < min_samples:
            continue
        
        # Create spatial features (no temporal component for single frame)
        spatial_features = np.column_stack([y_coords, x_coords])
        
        # Apply DBSCAN for spatial segmentation
        dbscan = DBSCAN(eps=eps_spatial, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(spatial_features)
        
        # Assign global segment IDs
        for local_seg_id in np.unique(dbscan_labels):
            if local_seg_id < 0:  # Skip noise points
                continue
            
            local_mask = dbscan_labels == local_seg_id
            seg_indices = (y_coords[local_mask], x_coords[local_mask])
            
            # Check minimum segment size
            if len(seg_indices[0]) >= min_final_points:
                seg_map[seg_indices] = global_seg_id
                global_seg_id += 1
    
    return seg_map, global_seg_id

def compute_segment_properties(
    seg_map: np.ndarray,
    normals: torch.Tensor,
    depth: torch.Tensor,
    pointclouds: torch.Tensor,
    device: str = 'cuda'
) -> Dict[int, Dict]:
    """
    Compute properties for each segment in a frame.
    
    Returns dict mapping segment_id to properties:
        - avg_normal: average normal in world space
        - centroid: 3D centroid in world space
        - pixels: list of (y, x) pixel coordinates
        - boundary_pixels: pixels on segment boundary
    """
    H, W = seg_map.shape
    segment_props = {}
    
    
    for seg_id in np.unique(seg_map):
        if seg_id < 0:
            continue
        
        mask = seg_map == seg_id
        y_idx, x_idx = np.where(mask)
        
        if len(y_idx) < 10:
            continue
        
        # Convert indices to torch tensors
        y_idx_t = torch.tensor(y_idx, device=device, dtype=torch.long)
        x_idx_t = torch.tensor(x_idx, device=device, dtype=torch.long)
        
        d = depth[y_idx_t, x_idx_t]
        pts_all = pointclouds[y_idx_t, x_idx_t]
        seg_normals_all = normals[y_idx_t, x_idx_t]
        valid = (
            (d > 0)
            & torch.isfinite(pts_all).all(dim=-1)
            & torch.isfinite(seg_normals_all).all(dim=-1)
            & (torch.linalg.norm(pts_all, dim=-1) > 1.0e-8)
            & (torch.linalg.norm(seg_normals_all, dim=-1) > 1.0e-6)
        )
        
        if valid.sum() < 10:
            continue
        
        # Get normals for this segment (already in world space)
        seg_normals = seg_normals_all[valid]
        avg_normal = _oriented_mean_normal(seg_normals)

        pts_world = pts_all[valid]
        centroid = pts_world.mean(dim=0)
        
        # Find boundary pixels
        kernel = np.ones((3, 3), dtype=bool)
        from scipy import ndimage
        eroded = ndimage.binary_erosion(mask, kernel)
        boundary = mask & ~eroded
        boundary_y, boundary_x = np.where(boundary)
        
        segment_props[seg_id] = {
            'avg_normal': avg_normal,
            'centroid': centroid,
            'pixels': list(zip(y_idx, x_idx)),
            'boundary_pixels': list(zip(boundary_y, boundary_x)),
            'world_points': pts_world
        }
    
    return segment_props


def find_segment_correspondences(
    props1: Dict[int, Dict],
    props2: Dict[int, Dict],
    flow_forward: np.ndarray,
    covis_mask: np.ndarray,
    normal_threshold: float = 0.95,  # Cosine similarity
    overlap_threshold: float = 0.3
) -> List[Tuple[int, int]]:
    """
    Find segment correspondences between two frames using flow.
    
    Returns list of (seg_id1, seg_id2) pairs that should be merged.
    """
    H, W = flow_forward.shape[:2]
    correspondences = []
    
    for seg1_id, props1_seg in props1.items():
        # Check normal similarity with all segments in frame 2
        best_match = None
        best_score = 0
        
        for seg2_id, props2_seg in props2.items():
            # Check normal similarity
            normal_sim = torch.dot(props1_seg['avg_normal'], props2_seg['avg_normal']).item()
            if normal_sim < normal_threshold:
                continue
            
            # Warp segment 1 pixels using flow
            warped_pixels = []
            valid_count = 0
            
            for y, x in props1_seg['boundary_pixels']:
                if covis_mask[y, x]:
                    flow = flow_forward[y, x]
                    y_new = int(round(y + flow[1]))
                    x_new = int(round(x + flow[0]))
                    
                    if 0 <= y_new < H and 0 <= x_new < W:
                        warped_pixels.append((y_new, x_new))
                        valid_count += 1
            
            if valid_count == 0:
                continue
            
            # Check overlap with segment 2
            overlap_count = 0
            for y, x in warped_pixels:
                if (y, x) in props2_seg['pixels']:
                    overlap_count += 1
            
            overlap_ratio = overlap_count / len(props2_seg['boundary_pixels']) if props2_seg['boundary_pixels'] else 0
            
            # Combined score
            score = normal_sim * overlap_ratio
            if score > best_score and overlap_ratio > overlap_threshold:
                best_score = score
                best_match = seg2_id
        
        if best_match is not None:
            correspondences.append((seg1_id, seg2_id))
    
    return correspondences


def merge_segments_across_frames(
    all_frame_segments: List[Dict[int, Dict]],
    all_frame_correspondences: List[List[Tuple[int, int]]],
    num_frames: int
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Merge segments across all frames using union-find.
    
    Returns dict mapping global_segment_id to list of (frame_idx, local_segment_id) pairs.
    """
    # Build union-find structure
    parent = {}
    
    def make_key(frame_idx, seg_id):
        return f"{frame_idx}_{seg_id}"
    
    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Initialize all segments
    for frame_idx in range(num_frames):
        for seg_id in all_frame_segments[frame_idx].keys():
            key = make_key(frame_idx, seg_id)
            find(key)  # Initialize
    
    # Apply correspondences
    for frame_idx, correspondences in enumerate(all_frame_correspondences):
        for seg1_id, seg2_id in correspondences:
            key1 = make_key(frame_idx, seg1_id)
            key2 = make_key(frame_idx + 1, seg2_id)
            union(key1, key2)
    
    # Group segments by root
    global_segments = defaultdict(list)
    for frame_idx in range(num_frames):
        for seg_id in all_frame_segments[frame_idx].keys():
            key = make_key(frame_idx, seg_id)
            root = find(key)
            global_segments[root].append((frame_idx, seg_id))
    
    # Renumber global segments
    final_segments = {}
    for i, (root, members) in enumerate(global_segments.items()):
        final_segments[i] = members
    
    return final_segments


def ransac_plane_torch(
    pts_cam: torch.Tensor,
    n_iters: int = 200,
    inlier_thresh: float = 0.01,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    RANSAC plane fitting on points.
    
    Returns:
        n: Normal vector of the plane
        p0: Point on the plane
        mask: Inlier mask
    """
    N = pts_cam.shape[0]
    best_inliers = 0
    best_n = None
    best_p0 = None
    best_mask = None

    for _ in range(n_iters):
        idx = torch.randperm(N, device=device)[:3]
        p0, p1, p2 = pts_cam[idx]

        v1, v2 = p1 - p0, p2 - p0
        n = torch.cross(v1, v2)
        if torch.linalg.norm(n) < 1e-6:
            continue
        n = F.normalize(n, dim=0)
        
        # Ensure normal points in consistent direction
        if n[2] > 0:
            n = -n

        dists = torch.abs((pts_cam - p0) @ n)
        inliers = dists < inlier_thresh
        n_inl = int(inliers.sum())

        if n_inl > best_inliers:
            best_inliers = n_inl
            best_n = n
            best_p0 = pts_cam[inliers].mean(0)
            best_mask = inliers

    return best_n, best_p0, best_mask

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2

def load_and_resize_flow(
    flow_path: Path,
    target_height: int,
    target_width: int
) -> np.ndarray:
    """
    Load flow from file and resize to target dimensions.
    
    Args:
        flow_path: Path to flow .npy file
        target_height: Target height to resize to
        target_width: Target width to resize to
    
    Returns:
        flow: (H, W, 2) array with resized flow
    """
    # Load flow - expecting shape (2, H_orig, W_orig)
    flow_raw = np.load(flow_path)
    
    if flow_raw.shape[0] == 2:
        # Convert from (2, H, W) to (H, W, 2)
        flow = np.transpose(flow_raw, (1, 2, 0))
    else:
        flow = flow_raw
    
    orig_h, orig_w = flow.shape[:2]
    
    # Scale factors for flow values
    scale_x = target_width / orig_w
    scale_y = target_height / orig_h
    
    # Resize flow using cv2
    flow_resized = cv2.resize(flow, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    
    # Scale flow values to account for resolution change
    flow_resized[..., 0] *= scale_x  # u component
    flow_resized[..., 1] *= scale_y  # v component
    
    return flow_resized

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

def estimate_normals_knn(
    points: torch.Tensor,
    k: int = 20,
    align_to_direction: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Estimate normals for a point cloud using k-nearest neighbors and PCA.
    
    Args:
        points: Point cloud tensor of shape [N, 3]
        k: Number of nearest neighbors to use for normal estimation
        align_to_direction: Optional direction vector to align normals consistently
                          (e.g., viewpoint for consistent orientation)
    
    Returns:
        normals: Estimated normals of shape [N, 3]
    """
    N = points.shape[0]
    device = points.device
    
    # Compute pairwise distances
    # Using broadcasting: ||pi - pj||^2 = ||pi||^2 + ||pj||^2 - 2*pi·pj
    points_norm = (points ** 2).sum(dim=1, keepdim=True)  # [N, 1]
    distances = points_norm + points_norm.T - 2 * torch.mm(points, points.T)  # [N, N]
    distances = torch.clamp(distances, min=0.0)  # Handle numerical errors
    
    # Find k-nearest neighbors (including the point itself)
    _, indices = torch.topk(distances, k=min(k, N), dim=1, largest=False)  # [N, k]
    
    # Initialize normals
    normals = torch.zeros_like(points)
    
    for i in range(N):
        # Get neighborhood points
        neighbors = points[indices[i]]  # [k, 3]
        
        # Center the points
        centroid = neighbors.mean(dim=0, keepdim=True)  # [1, 3]
        centered = neighbors - centroid  # [k, 3]
        
        # Compute covariance matrix
        cov = torch.mm(centered.T, centered) / k  # [3, 3]
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        
        # Normal is the eigenvector with smallest eigenvalue
        normal = eigenvectors[:, 0]  # [3]
        
        # Store the normal
        normals[i] = normal
    
    # Normalize the normals
    normals = F.normalize(normals, p=2, dim=1)
    
    # Align normals consistently if direction is provided
    if align_to_direction is not None:
        align_to_direction = F.normalize(align_to_direction, p=2, dim=-1)
        if align_to_direction.dim() == 1:
            align_to_direction = align_to_direction.unsqueeze(0).expand_as(normals)
        
        # Flip normals that point away from the alignment direction
        dots = (normals * align_to_direction).sum(dim=1, keepdim=True)
        normals = torch.where(dots < 0, -normals, normals)
    
    return normals


def estimate_normals_radius(
    points: torch.Tensor,
    radius: float,
    min_neighbors: int = 3,
    align_to_direction: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Estimate normals using points within a radius.
    
    Args:
        points: Point cloud tensor of shape [N, 3]
        radius: Search radius for neighbors
        min_neighbors: Minimum neighbors required for normal estimation
        align_to_direction: Optional direction for consistent orientation
    
    Returns:
        normals: Estimated normals of shape [N, 3]
    """
    N = points.shape[0]
    device = points.device
    
    # Compute pairwise distances
    points_norm = (points ** 2).sum(dim=1, keepdim=True)
    distances = points_norm + points_norm.T - 2 * torch.mm(points, points.T)
    distances = torch.sqrt(torch.clamp(distances, min=1e-8))
    
    # Create mask for points within radius
    mask = distances <= radius  # [N, N]
    
    normals = torch.zeros_like(points)
    valid_mask = torch.zeros(N, dtype=torch.bool, device=device)
    
    for i in range(N):
        # Get neighbors within radius
        neighbor_mask = mask[i]
        neighbor_indices = torch.where(neighbor_mask)[0]
        
        if len(neighbor_indices) < min_neighbors:
            # Not enough neighbors, skip this point
            continue
        
        neighbors = points[neighbor_indices]
        
        # Center the points
        centroid = neighbors.mean(dim=0, keepdim=True)
        centered = neighbors - centroid
        
        # Compute covariance matrix
        cov = torch.mm(centered.T, centered) / len(neighbor_indices)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        
        # Normal is the eigenvector with smallest eigenvalue
        normals[i] = eigenvectors[:, 0]
        valid_mask[i] = True
    
    # Normalize valid normals
    normals[valid_mask] = F.normalize(normals[valid_mask], p=2, dim=1)
    
    # Align normals if direction provided
    if align_to_direction is not None and valid_mask.any():
        align_to_direction = F.normalize(align_to_direction, p=2, dim=-1)
        if align_to_direction.dim() == 1:
            align_to_direction = align_to_direction.unsqueeze(0).expand_as(normals)
        
        dots = (normals * align_to_direction).sum(dim=1, keepdim=True)
        normals = torch.where(dots < 0, -normals, normals)
    
    return normals, valid_mask


def estimate_normals_fast(
    points: torch.Tensor,
    k: int = 20
) -> torch.Tensor:
    """
    Fast normal estimation using vectorized operations.
    More efficient for large point clouds.
    
    Args:
        points: Point cloud tensor of shape [N, 3]
        k: Number of nearest neighbors
    
    Returns:
        normals: Estimated normals of shape [N, 3]
    """
    N = points.shape[0]
    device = points.device
    
    # Compute pairwise distances efficiently
    points_norm = (points ** 2).sum(dim=1, keepdim=True)
    distances = points_norm + points_norm.T - 2 * torch.mm(points, points.T)
    
    # Find k-nearest neighbors
    _, indices = torch.topk(distances, k=min(k, N), dim=1, largest=False)
    
    # Gather neighbor points for all points at once
    # Shape: [N, k, 3]
    neighbors = points[indices]
    
    # Compute centroids
    centroids = neighbors.mean(dim=1, keepdim=True)  # [N, 1, 3]
    
    # Center all neighborhoods
    centered = neighbors - centroids  # [N, k, 3]
    
    # Compute covariance matrices for all points
    # cov[i] = centered[i].T @ centered[i]
    cov = torch.bmm(centered.transpose(1, 2), centered) / k  # [N, 3, 3]
    
    # Compute eigendecomposition for all covariance matrices
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)  # [N, 3], [N, 3, 3]
    
    # Extract normals (eigenvector with smallest eigenvalue)
    normals = eigenvectors[:, :, 0]  # [N, 3]
    
    # Normalize
    normals = F.normalize(normals, p=2, dim=1)
    
    return normals

def load_flow_and_covis_data_fixed(
    data_dir: Path,
    frame_indices: List[int],
    target_height: int,
    target_width: int,
    interval: int = 7
) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], np.ndarray]]:
    """
    Load flow and covisibility data for given frame indices with proper resizing.
    
    Args:
        data_dir: Directory containing flow and covis files
        frame_indices: List of frame indices
        target_height: Target height (should match depth/normal maps)
        target_width: Target width (should match depth/normal maps)
        interval: Frame interval
    
    Returns:
        flow_forward: dict mapping (frame_i, frame_j) -> flow array (H, W, 2)
        flow_backward: dict mapping (frame_j, frame_i) -> flow array (H, W, 2)
        covis_masks: dict mapping (frame_i, frame_j) -> covisibility mask (H, W)
    """
    flow_forward = {}
    flow_backward = {}
    covis_masks = {}
    
    for i in range(len(frame_indices) - 1):
        frame_i = frame_indices[i]
        frame_j = frame_indices[i + 1]
        
        # Load and resize forward flow
        flow_file_forward = data_dir / f"flow_{frame_i:05d}_{frame_j:05d}.npy"
        if flow_file_forward.exists():
            flow_forward[(frame_i, frame_j)] = load_and_resize_flow(
                flow_file_forward, target_height, target_width
            )
            print(f"  Loaded forward flow {frame_i}->{frame_j}, shape: {flow_forward[(frame_i, frame_j)].shape}")
        
        # Load and resize backward flow
        flow_file_backward = data_dir / f"flow_{frame_j:05d}_{frame_i:05d}.npy"
        if flow_file_backward.exists():
            flow_backward[(frame_j, frame_i)] = load_and_resize_flow(
                flow_file_backward, target_height, target_width
            )
        
        # Load and resize covisibility mask
        covis_file = data_dir / f"covis_{frame_i:05d}_{frame_j:05d}.npy"
        if covis_file.exists():
            covis_raw = np.load(covis_file)
            # Resize covis mask if needed
            if covis_raw.shape != (target_height, target_width):
                covis_resized = cv2.resize(
                    covis_raw.astype(np.float32), 
                    (target_width, target_height), 
                    interpolation=cv2.INTER_LINEAR
                )
                covis_masks[(frame_i, frame_j)] = covis_resized
            else:
                covis_masks[(frame_i, frame_j)] = covis_raw
            print(f"  Loaded covis mask {frame_i}->{frame_j}, shape: {covis_masks[(frame_i, frame_j)].shape}")
    
    return flow_forward, flow_backward, covis_masks


def robust_plane_ransac(P, n_hint=None, n_iters=500, thresh=0.02):
    """Return n (unit), centre, inlier_mask."""
    best_inliers = None
    best_n = None
    best_c = None
    N = P.shape[0]
    try:
        rng = torch.Generator(device=P.device)
        rng.manual_seed(1729 + int(N))
    except Exception:
        rng = None
    for _ in range(n_iters):
        # try until we sample a non-degenerate triplet
        for _ in range(10):
            idx = torch.randint(0, N, (3,), device=P.device, generator=rng)
            v1, v2, v3 = P[idx]
            n = torch.linalg.cross(v2-v1, v3-v1)
            if torch.linalg.norm(n) > 1e-6:
                n = F.normalize(n, dim=0)
                break
        else:  # could not find good triplet – give up
            continue

        # prefer orientation consistent with the hint
        if n_hint is not None and torch.dot(n, n_hint) < 0:
            n = -n

        dists = torch.abs((P @ n) - torch.dot(v1, n))
        inliers = dists < thresh
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers, best_n = inliers, n
            best_c         = P[inliers].mean(0)

            # early exit if we already agree with the hint and have many inliers
            if n_hint is not None and torch.dot(best_n, n_hint) > 0.99 and inliers.sum() > 0.8*N:
                break

    return best_n, best_c, best_inliers


def farthest_point_sampling(points, n_samples):
    """
    Farthest Point Sampling to ensure uniform distribution
    Args:
        points: (N, 3) tensor of points
        n_samples: number of points to sample
    Returns:
        indices: (n_samples,) tensor of selected indices
    """
    device = points.device
    N = points.shape[0]
    
    if N <= n_samples:
        return torch.arange(N, device=device)
    
    # Initialize
    indices = torch.zeros(n_samples, dtype=torch.long, device=device)
    distances = torch.ones(N, device=device) * 1e10
    
    # Random starting point
    farthest = torch.randint(0, N, (1,), device=device)
    
    for i in range(n_samples):
        indices[i] = farthest
        
        # Update distances to nearest selected point
        centroid = points[farthest]
        dist = torch.sum((points - centroid) ** 2, dim=1)
        distances = torch.min(distances, dist)
        
        # Select farthest point from already selected points
        farthest = torch.argmax(distances)
    
    return indices


def find_minimal_volume_bbox(points: torch.Tensor, device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find the minimal volume oriented bounding box using PCA and rotation search.
    
    Args:
        points: (N, 3) point cloud
        device: computation device
        
    Returns:
        R_bw: (3, 3) rotation matrix (body to world)
        center: (3,) center of bbox
        half_sizes: (3,) half dimensions of bbox
    """
    points = points.to(device)
    
    # Center points
    center = points.mean(0)
    points_centered = points - center
    
    # Initial orientation from PCA
    _, _, Vt = torch.linalg.svd(points_centered, full_matrices=False)
    
    # Try multiple orientations to find minimal volume
    best_volume = float('inf')
    best_R = None
    best_half_sizes = None
    best_center = center.clone()
    
    # Method 1: PCA-based initial guess
    R_pca = Vt.T  # Principal axes as columns
    
    # Ensure right-handed coordinate system
    if torch.det(R_pca) < 0:
        R_pca[:, 2] = -R_pca[:, 2]
    
    # Project points to PCA frame
    points_pca = points_centered @ R_pca
    
    # Compute bbox in PCA frame
    min_vals = points_pca.min(dim=0).values
    max_vals = points_pca.max(dim=0).values
    half_sizes_pca = (max_vals - min_vals) / 2
    center_offset_pca = (max_vals + min_vals) / 2
    
    # Volume
    volume_pca = torch.prod(half_sizes_pca).item()
    
    best_R = R_pca
    best_half_sizes = half_sizes_pca
    best_center = center + center_offset_pca @ R_pca.T
    best_volume = volume_pca
    
    # Method 2: Try rotating around principal axes to minimize volume
    # This is especially important for nearly planar objects
    n_angles = 24  # Number of angles to try
    
    for axis_idx in range(3):
        for angle_idx in range(n_angles):
            angle = (angle_idx * 2 * np.pi) / n_angles
            
            # Create rotation matrix around axis
            cos_a = torch.cos(torch.tensor(angle, device=device))
            sin_a = torch.sin(torch.tensor(angle, device=device))
            
            R_rot = torch.eye(3, device=device)
            if axis_idx == 0:  # Rotate around X
                R_rot[1, 1] = cos_a
                R_rot[1, 2] = -sin_a
                R_rot[2, 1] = sin_a
                R_rot[2, 2] = cos_a
            elif axis_idx == 1:  # Rotate around Y
                R_rot[0, 0] = cos_a
                R_rot[0, 2] = sin_a
                R_rot[2, 0] = -sin_a
                R_rot[2, 2] = cos_a
            else:  # Rotate around Z
                R_rot[0, 0] = cos_a
                R_rot[0, 1] = -sin_a
                R_rot[1, 0] = sin_a
                R_rot[1, 1] = cos_a
            
            # Apply rotation to PCA basis
            R_test = R_pca @ R_rot
            
            # Project points
            points_test = points_centered @ R_test
            
            # Compute bbox
            min_vals = points_test.min(dim=0).values
            max_vals = points_test.max(dim=0).values
            half_sizes_test = (max_vals - min_vals) / 2
            center_offset_test = (max_vals + min_vals) / 2
            
            # Volume
            volume_test = torch.prod(half_sizes_test).item()
            
            if volume_test < best_volume:
                best_volume = volume_test
                best_R = R_test
                best_half_sizes = half_sizes_test
                best_center = center + center_offset_test @ R_test.T
    
    # Ensure the smallest dimension is along z-axis (for thin objects)
    # Sort dimensions and reorder axes
    sorted_indices = torch.argsort(best_half_sizes)
    best_R = best_R[:, sorted_indices]
    best_half_sizes = best_half_sizes[sorted_indices]
    
    # Ensure right-handed coordinate system after reordering
    if torch.det(best_R) < 0:
        best_R[:, 2] = -best_R[:, 2]
    
    return best_R, best_center, best_half_sizes


def make_local_frame(n: torch.Tensor):
    n = F.normalize(n, dim=0)
    helper = torch.tensor([1., 0., 0.], device=n.device, dtype=n.dtype)
    if torch.abs(torch.dot(n, helper)) > 0.9:
        helper = torch.tensor([0., 1., 0.], device=n.device, dtype=n.dtype)
    u = torch.linalg.cross(n, helper); u = F.normalize(u, dim=0)
    v = torch.linalg.cross(n, u);      v = F.normalize(v, dim=0)
    return u, v, n

def convex_hull_monotone_chain(xy2d: torch.Tensor) -> torch.Tensor:
    idx = torch.argsort(xy2d[:, 0] + 1e-9 * xy2d[:, 1])
    pts = xy2d[idx].tolist()
    def cross(o, a, b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return torch.tensor(hull, dtype=xy2d.dtype)

def polygon_area(xy2d: torch.Tensor) -> torch.Tensor:
    # xy2d: (K,2) closed polygon (not necessarily closed in memory)
    x = xy2d[:, 0]; y = xy2d[:, 1]
    x2 = torch.roll(x, -1); y2 = torch.roll(y, -1)
    return 0.5 * (x*y2 - y*x2).abs().sum()

def min_area_rect(xy_cpu: torch.Tensor):
    """Return (theta, half2d[2], mid2d[2]) of min-area rectangle over hull."""
    hull = convex_hull_monotone_chain(xy_cpu)
    if hull.shape[0] < 3:
        # fallback: centered 2D PCA
        xy0 = xy_cpu - xy_cpu.mean(0, keepdim=True)
        C = (xy0.T @ xy0) / max(xy0.shape[0]-1, 1)
        _, evecs = torch.linalg.eigh(C)
        x2d = evecs[:, 1]
        theta = torch.atan2(x2d[1], x2d[0])
        R2 = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                            [torch.sin(theta),  torch.cos(theta)]], dtype=xy_cpu.dtype)
        q = xy_cpu @ R2
        mn, mx = q.min(0).values, q.max(0).values
        half = (mx - mn) / 2
        mid  = (mx + mn) / 2
        area_hull = polygon_area(convex_hull_monotone_chain(xy_cpu))
        area_rect = (2*half[0])*(2*half[1])
        rect_ratio = (area_hull / (area_rect + 1e-9)).clamp(max=1.0)
        return theta, half, mid, rect_ratio

    edges = hull.roll(-1, dims=0) - hull
    angs = torch.atan2(edges[:, 1], edges[:, 0])
    best_area = float('inf'); best = None
    for theta in angs:
        cth, sth = torch.cos(theta), torch.sin(theta)
        R2 = torch.tensor([[cth, -sth], [sth, cth]], dtype=xy_cpu.dtype)
        q  = xy_cpu @ R2
        mn, mx = q.min(0).values, q.max(0).values
        half = (mx - mn) / 2
        mid  = (mx + mn) / 2
        area = (2*half[0])*(2*half[1])
        if area < best_area:
            best_area = area; best = (theta, half, mid)
    theta, half, mid = best
    area_hull = polygon_area(hull)
    rect_ratio = (area_hull / (best_area + 1e-9)).clamp(max=1.0)
    return theta, half, mid, rect_ratio

def extent(u: torch.Tensor):
    q = torch.quantile(u, torch.tensor([0.00, 0.99], device=u.device, dtype=u.dtype))
    return (q[1] - q[0]) / 2, (q[1] + q[0]) / 2

def _oriented_mean_normal(normals: torch.Tensor) -> torch.Tensor:
    normals = normals[torch.isfinite(normals).all(dim=-1)]
    if normals.numel() == 0:
        return torch.tensor([0.0, 0.0, 1.0], device=normals.device, dtype=normals.dtype)
    normals = F.normalize(normals, dim=-1)
    ref = normals[0]
    signs = torch.where((normals @ ref) < 0.0, -1.0, 1.0).to(device=normals.device, dtype=normals.dtype)
    mean = (normals * signs.unsqueeze(-1)).mean(dim=0)
    if not torch.isfinite(mean).all() or torch.linalg.norm(mean) < 1.0e-6:
        return ref
    return F.normalize(mean, dim=0)

def _fit_plane_box_for_axis(P: torch.Tensor, n: torch.Tensor, c: torch.Tensor, x_axis: torch.Tensor):
    """Fit a plane box using a fixed in-plane x axis."""
    n = F.normalize(n, dim=0)
    x_axis = x_axis - (x_axis @ n) * n
    if torch.linalg.norm(x_axis) < 1.0e-6:
        raise ValueError("Axis hint is parallel to plane normal.")
    x_axis = F.normalize(x_axis, dim=0)
    y_axis = F.normalize(torch.linalg.cross(n, x_axis), dim=0)

    P0 = P - c
    Pp = P0 - (P0 @ n).unsqueeze(1) * n
    u_proj = Pp @ x_axis
    v_proj = Pp @ y_axis
    u_half, u_mid = extent(u_proj)
    v_half, v_mid = extent(v_proj)

    signed_dist_n = P0 @ n
    n_mid = torch.quantile(signed_dist_n, torch.tensor(0.50, device=P.device, dtype=P.dtype))
    dist_n = (signed_dist_n - n_mid).abs()
    z_half = torch.quantile(dist_n, torch.tensor(0.95, device=P.device, dtype=P.dtype)).clamp(min=0.02)

    centre = c + u_mid * x_axis + v_mid * y_axis + n_mid * n
    half_sz = torch.stack([u_half, v_half, z_half]).clamp(min=0.02)
    R_bw = torch.stack([x_axis, y_axis, n], dim=1)
    if torch.det(R_bw) < 0:
        R_bw[:, 0] = -R_bw[:, 0]
    return R_bw, centre, half_sz


def fit_one_plane_box(P: torch.Tensor, n: torch.Tensor, c: torch.Tensor, gid, axis_hint: torch.Tensor | None = None):
    """Fit a single plane-aligned box to points P given plane (n,c). Returns (R_bw, centre, half_sz, rect_ratio)."""
    P = P[torch.isfinite(P).all(dim=1)]
    if P.shape[0] < 3:
        raise ValueError(f"[{gid}] Need at least 3 finite points for plane box fitting.")
    n = F.normalize(n, dim=0)
    P0 = P - c
    Pp = P0 - (P0 @ n).unsqueeze(1) * n

    # Optional FPS just for orientation robustness
    Porient = Pp
    '''if Pp.shape[0] > min_fps_points:
        n_samples = max(min_fps_points, int(Pp.shape[0] * fps_ratio))
        try:
            from pytorch3d.ops import sample_farthest_points
            Porient, _ = sample_farthest_points(Pp.unsqueeze(0), K=n_samples, random_start_point=True)
            Porient = Porient.squeeze(0)
            print(f"[{gid}] FPS: {Pp.shape[0]} -> {n_samples} points")
        except Exception:
            idx = torch.randperm(Pp.shape[0], device=Pp.device)[:n_samples]
            Porient = Pp[idx]'''
    device = Porient.device
    u0, v0, _ = make_local_frame(n)
    UV = torch.stack([u0, v0], dim=1)                  # (3,2)
    xy_cpu = (Porient @ UV).detach().cpu()             # (M,2)

    theta, half2d, mid2d, rect_ratio = min_area_rect(xy_cpu)

    # Lift 2D axes to 3D
    cth = torch.cos(theta).to(device=device); sth = torch.sin(theta).to(device=device)
    x_axis = (UV.to(device) @ torch.stack([cth, sth]).to(dtype=P.dtype, device=device))
    x_axis = F.normalize(x_axis - (x_axis @ n) * n, dim=0)
    R_bw, centre, half_sz = _fit_plane_box_for_axis(P, n, c, x_axis)

    if axis_hint is not None:
        try:
            hint = axis_hint.to(device=device, dtype=P.dtype)
            hint = hint - (hint @ n) * n
            if torch.linalg.norm(hint) > 1.0e-6:
                hint = F.normalize(hint, dim=0)
                R_hint, centre_hint, half_hint = _fit_plane_box_for_axis(P, n, c, hint)
                base_support, base_area = _plane_box_support_score(P, R_bw, centre, half_sz, grid_base=56)
                hint_support, hint_area = _plane_box_support_score(P, R_hint, centre_hint, half_hint, grid_base=56)
                base_axis = R_bw[:, 0]
                delta = float(
                    torch.rad2deg(
                        torch.atan2(
                            torch.dot(hint, R_bw[:, 1]),
                            torch.dot(hint, base_axis),
                        )
                    ).detach().cpu().item()
                )
                delta = ((delta + 45.0) % 90.0) - 45.0
                tradeoff_allowed = (
                    hint_area <= base_area
                    or base_area >= 5.0
                )
                snap_ok = (
                    tradeoff_allowed
                    and
                    hint_area <= base_area * 1.35
                    and hint_support >= base_support - 0.25
                    and abs(delta) >= 2.0
                )
                if snap_ok:
                    print(
                        f"[{gid}] Axis snap {delta:+.1f}deg "
                        f"support {base_support:.2f}->{hint_support:.2f} "
                        f"area {base_area:.2f}->{hint_area:.2f}"
                    )
                    R_bw, centre, half_sz = R_hint, centre_hint, half_hint
        except Exception:
            pass
    return R_bw, centre, half_sz, rect_ratio

def _plane_box_support_score(
    P: torch.Tensor,
    R_bw: torch.Tensor,
    centre: torch.Tensor,
    half_sz: torch.Tensor,
    grid_base: int = 14,
) -> Tuple[float, float]:
    """Return support ratio and in-plane area for the fitted watertight box."""
    if P.shape[0] == 0:
        return 0.0, 0.0
    local = (P - centre) @ R_bw
    half_xy = half_sz[:2].clamp(min=1.0e-4)
    in_xy = (local[:, 0].abs() <= half_xy[0]) & (local[:, 1].abs() <= half_xy[1])
    if not bool(in_xy.any()):
        return 0.0, float((4.0 * half_xy[0] * half_xy[1]).detach().cpu().item())

    xy = local[in_xy, :2]
    aspect = float((half_xy[0] / half_xy[1]).clamp(0.25, 4.0).detach().cpu().item())
    nx = int(np.clip(round(grid_base * np.sqrt(aspect)), 5, 28))
    ny = int(np.clip(round(grid_base / np.sqrt(aspect)), 5, 28))
    uv = ((xy + half_xy) / (2.0 * half_xy)).clamp(0.0, 0.999999)
    ix = torch.clamp((uv[:, 0] * nx).long(), 0, nx - 1)
    iy = torch.clamp((uv[:, 1] * ny).long(), 0, ny - 1)
    occupied = torch.unique(iy * nx + ix).numel()
    support = float(occupied) / float(nx * ny)
    area = float((4.0 * half_xy[0] * half_xy[1]).detach().cpu().item())
    return support, area


def _edge_aware_trim_plane_box(
    P: torch.Tensor,
    R_bw: torch.Tensor,
    centre: torch.Tensor,
    half_sz: torch.Tensor,
    gid,
    *,
    grid_base: int = 36,
    min_edge_fill: float = 0.20,
    min_keep_fraction: float = 0.88,
    min_area_reduction: float = 0.06,
    min_support_gain: float = 0.03,
    edge_padding: float = 0.02,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """
    Shrink a fitted plane box to occupied footprint edges.

    This prevents a few edge/outlier points from making the watertight primitive
    protrude beyond the observed surface. The trim is conservative: it is accepted
    only when it keeps most points and either removes meaningful area or improves
    the support-grid score.
    """
    if P.shape[0] < 3:
        return R_bw, centre, half_sz, False

    half_xy = half_sz[:2].clamp(min=1.0e-4)
    if float(torch.min(half_xy).detach().cpu().item()) <= 1.0e-5:
        return R_bw, centre, half_sz, False

    local = (P - centre.unsqueeze(0)) @ R_bw
    in_rect = (
        (local[:, 0].abs() <= half_xy[0])
        & (local[:, 1].abs() <= half_xy[1])
    )
    if int(in_rect.sum().item()) < 20:
        return R_bw, centre, half_sz, False

    xy = local[in_rect, :2]
    aspect = float((half_xy[0] / half_xy[1]).clamp(0.20, 5.0).detach().cpu().item())
    nx = int(np.clip(round(grid_base * np.sqrt(aspect)), 8, 80))
    ny = int(np.clip(round(grid_base / np.sqrt(aspect)), 8, 80))
    uv = ((xy + half_xy) / (2.0 * half_xy)).clamp(0.0, 0.999999)
    ix = torch.clamp((uv[:, 0] * nx).long(), 0, nx - 1)
    iy = torch.clamp((uv[:, 1] * ny).long(), 0, ny - 1)

    counts = torch.zeros((ny, nx), device=P.device, dtype=torch.int32)
    counts.index_put_((iy, ix), torch.ones_like(ix, dtype=torch.int32), accumulate=True)
    occupied = counts > 0
    if int(occupied.sum().item()) < 4:
        return R_bw, centre, half_sz, False

    x0, x1 = 0, nx - 1
    y0, y1 = 0, ny - 1
    min_cols = min(4, nx)
    min_rows = min(4, ny)

    def _edge_fill(axis: str) -> float:
        if axis == "left":
            edge = occupied[y0 : y1 + 1, x0]
        elif axis == "right":
            edge = occupied[y0 : y1 + 1, x1]
        elif axis == "bottom":
            edge = occupied[y0, x0 : x1 + 1]
        else:
            edge = occupied[y1, x0 : x1 + 1]
        if edge.numel() == 0:
            return 0.0
        return float(edge.float().mean().detach().cpu().item())

    changed = True
    while changed and (x1 - x0 + 1) >= min_cols and (y1 - y0 + 1) >= min_rows:
        changed = False
        if (x1 - x0 + 1) > min_cols and _edge_fill("left") < min_edge_fill:
            x0 += 1
            changed = True
        if (x1 - x0 + 1) > min_cols and _edge_fill("right") < min_edge_fill:
            x1 -= 1
            changed = True
        if (y1 - y0 + 1) > min_rows and _edge_fill("bottom") < min_edge_fill:
            y0 += 1
            changed = True
        if (y1 - y0 + 1) > min_rows and _edge_fill("top") < min_edge_fill:
            y1 -= 1
            changed = True

    if x0 == 0 and x1 == nx - 1 and y0 == 0 and y1 == ny - 1:
        return R_bw, centre, half_sz, False

    cell_w = (2.0 * half_xy[0]) / float(nx)
    cell_h = (2.0 * half_xy[1]) / float(ny)
    pad_x = min(float(edge_padding), float((0.5 * cell_w).detach().cpu().item()))
    pad_y = min(float(edge_padding), float((0.5 * cell_h).detach().cpu().item()))

    lo_x = -half_xy[0] + cell_w * float(x0)
    hi_x = -half_xy[0] + cell_w * float(x1 + 1)
    lo_y = -half_xy[1] + cell_h * float(y0)
    hi_y = -half_xy[1] + cell_h * float(y1 + 1)
    lo_x = torch.maximum(lo_x - pad_x, -half_xy[0])
    hi_x = torch.minimum(hi_x + pad_x, half_xy[0])
    lo_y = torch.maximum(lo_y - pad_y, -half_xy[1])
    hi_y = torch.minimum(hi_y + pad_y, half_xy[1])

    new_half_x = 0.5 * (hi_x - lo_x)
    new_half_y = 0.5 * (hi_y - lo_y)
    if float(torch.minimum(new_half_x, new_half_y).detach().cpu().item()) <= 0.015:
        return R_bw, centre, half_sz, False

    keep = (
        (local[:, 0] >= lo_x)
        & (local[:, 0] <= hi_x)
        & (local[:, 1] >= lo_y)
        & (local[:, 1] <= hi_y)
    )
    keep_fraction = float(keep.float().mean().detach().cpu().item())
    if keep_fraction < min_keep_fraction:
        return R_bw, centre, half_sz, False

    center_delta = torch.stack([0.5 * (lo_x + hi_x), 0.5 * (lo_y + hi_y), torch.zeros((), device=P.device, dtype=P.dtype)])
    new_centre = centre + R_bw @ center_delta
    new_half_sz = half_sz.clone()
    new_half_sz[0] = new_half_x
    new_half_sz[1] = new_half_y
    new_half_sz = new_half_sz.clamp(min=0.02)

    old_support, old_area = _plane_box_support_score(P, R_bw, centre, half_sz)
    new_support, new_area = _plane_box_support_score(P, R_bw, new_centre, new_half_sz)
    if old_area <= 1.0e-8:
        return R_bw, centre, half_sz, False
    area_reduction = 1.0 - (new_area / old_area)
    support_gain = new_support - old_support
    if area_reduction < min_area_reduction and support_gain < min_support_gain:
        return R_bw, centre, half_sz, False

    print(
        f"[{gid}] Edge trim area -{area_reduction * 100:.1f}% "
        f"support {old_support:.2f}->{new_support:.2f} keep={keep_fraction:.2f}"
    )
    return R_bw, new_centre, new_half_sz, True


def _fit_supported_plane_box(
    P: torch.Tensor,
    n: torch.Tensor,
    c: torch.Tensor,
    gid,
    axis_hint: torch.Tensor | None = None,
    edge_trim: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]:
    R_bw, centre, half_sz, rect_ratio = fit_one_plane_box(P, n, c, gid, axis_hint=axis_hint)
    if edge_trim:
        R_bw, centre, half_sz, _ = _edge_aware_trim_plane_box(P, R_bw, centre, half_sz, gid)
    support_ratio, area = _plane_box_support_score(P, R_bw, centre, half_sz)
    return R_bw, centre, half_sz, rect_ratio, support_ratio, area


def _largest_gap_cut(local_xy: torch.Tensor, min_gap_ratio: float = 0.12) -> Tuple[int, torch.Tensor, float] | None:
    best: Tuple[int, torch.Tensor, float] | None = None
    for axis in (0, 1):
        coord = local_xy[:, axis]
        if coord.numel() < 2:
            continue
        lo = torch.quantile(coord, torch.tensor(0.01, device=coord.device, dtype=coord.dtype))
        hi = torch.quantile(coord, torch.tensor(0.99, device=coord.device, dtype=coord.dtype))
        span = hi - lo
        if float(span.detach().cpu().item()) <= 1.0e-6:
            continue
        clipped = coord[(coord >= lo) & (coord <= hi)]
        if clipped.numel() < 2:
            continue
        sorted_coord = torch.sort(clipped).values
        gaps = sorted_coord[1:] - sorted_coord[:-1]
        if gaps.numel() > 0:
            gap_idx = int(torch.argmax(gaps).item())
            gap = gaps[gap_idx]
            gap_ratio = float((gap / span).detach().cpu().item())
            if gap_ratio >= min_gap_ratio:
                cut = 0.5 * (sorted_coord[gap_idx] + sorted_coord[gap_idx + 1])
                if best is None or gap_ratio > best[2]:
                    best = (axis, cut, gap_ratio)
    return best


def _candidate_cut_values(local_xy: torch.Tensor, min_gap_ratio: float = 0.12) -> List[Tuple[int, torch.Tensor]]:
    cuts: List[Tuple[int, torch.Tensor]] = []
    gap_cut = _largest_gap_cut(local_xy, min_gap_ratio=min_gap_ratio)
    if gap_cut is not None:
        cuts.append((gap_cut[0], gap_cut[1]))

    for axis in (0, 1):
        coord = local_xy[:, axis]
        if coord.numel() < 2:
            continue
        quantiles = torch.tensor(
            [0.25, 0.33, 0.50, 0.67, 0.75],
            device=coord.device,
            dtype=coord.dtype,
        )
        values = torch.quantile(coord, quantiles)
        axis_cuts: List[torch.Tensor] = []
        for value in values:
            if not torch.isfinite(value):
                continue
            duplicate = any(
                bool(torch.isclose(value, prev, rtol=1.0e-4, atol=1.0e-4).detach().cpu().item())
                for prev in axis_cuts
            )
            if not duplicate:
                axis_cuts.append(value)
        cuts.extend((axis, value) for value in axis_cuts)
    return cuts


def split_plane_clusters(P: torch.Tensor, n: torch.Tensor, c: torch.Tensor, gid,
                          tau: float = 0.70, max_splits: int = 3, min_pts: int = 150):
    """
    Greedy 1D median split along principal in-plane axis until each cluster is 'rectangular enough'.
    Returns list of (R_bw, centre, half_sz).
    """
    out = []
    # Work list of index tensors
    P0 = P - c
    Pp = P0 - (P0 @ n).unsqueeze(1) * n
    u0, v0, _ = make_local_frame(n)
    UV = torch.stack([u0, v0], dim=1)
    xy = Pp @ UV

    # clusters start with all points
    clusters = [torch.arange(P.shape[0], device=P.device)]
    attempts = 0
    while len(clusters) > 0:
        idx = clusters.pop(0)
        if idx.numel() < min_pts or attempts >= max_splits:
            R_bw, centre, half_sz, _ = fit_one_plane_box(P[idx], n, c, gid)
            out.append((R_bw, centre, half_sz))
            continue

        # rectangularity of this cluster
        theta, half2d, mid2d, rr = min_area_rect((xy[idx]).detach().cpu())
        if rr >= tau or idx.numel() < 2 * min_pts:
            R_bw, centre, half_sz, _ = fit_one_plane_box(P[idx], n, c, gid)
            out.append((R_bw, centre, half_sz))
        else:
            # split along principal axis
            xyc = xy[idx] - xy[idx].mean(0, keepdim=True)
            C = (xyc.T @ xyc) / max(xyc.shape[0]-1, 1)
            _, evecs = torch.linalg.eigh(C)
            main = evecs[:, 1].to(device=P.device, dtype=P.dtype)
            proj = (xy[idx] @ main)
            med = torch.median(proj)
            left  = idx[proj <= med]
            right = idx[proj >  med]
            # ensure both sides large enough; if not, bail out
            if left.numel() < min_pts or right.numel() < min_pts:
                R_bw, centre, half_sz, _ = fit_one_plane_box(P[idx], n, c, gid)
                out.append((R_bw, centre, half_sz))
            else:
                clusters.append(left)
                clusters.append(right)
                attempts += 1
    return out


def fit_support_aware_plane_boxes(
    P: torch.Tensor,
    n: torch.Tensor,
    c: torch.Tensor,
    gid,
    *,
    max_depth: int = 2,
    min_pts: int = 180,
    min_support: float = 0.48,
    min_improvement: float = 0.12,
    large_area_split: float = 8.0,
    fine_support_split: float = 0.96,
    min_area_reduction: float = 0.045,
    axis_hint: torch.Tensor | None = None,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]]:
    """
    Fit a small set of watertight boxes whose visible top footprint is supported by points.

    The first candidate is a plane-aligned box. If its in-plane support grid is too sparse,
    recursively split along the strongest empty gap or median cut. A split is accepted only
    when it reduces unsupported area enough, which keeps primitive counts compact.
    """
    P = P[torch.isfinite(P).all(dim=1)]
    if P.shape[0] < 3:
        return []
    n = F.normalize(n, dim=0)

    def _fit_leaf(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]:
        return _fit_supported_plane_box(points, n, c, gid, axis_hint=axis_hint)

    def _waste(score: float, area: float) -> float:
        return float(area) * max(0.0, 1.0 - float(score))

    def _recurse(points: torch.Tensor, depth: int) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]]:
        R_bw, centre, half_sz, rect_ratio, support, area = _fit_leaf(points)
        current_waste = _waste(support, area)
        local = (points - centre) @ R_bw
        gap_cut = _largest_gap_cut(local[:, :2], min_gap_ratio=0.12)
        fine_support, _ = _plane_box_support_score(points, R_bw, centre, half_sz, grid_base=56)
        large_envelope = area >= large_area_split and fine_support <= fine_support_split
        if (
            (support >= min_support and gap_cut is None and not large_envelope)
            or depth >= max_depth
            or points.shape[0] < 2 * min_pts
            or area <= 1.0e-6
        ):
            return [(R_bw, centre, half_sz, points.detach().cpu(), support)]

        best = None
        for axis, cut_value in _candidate_cut_values(local[:, :2]):
            left_mask = local[:, axis] <= cut_value
            right_mask = ~left_mask
            left_count = int(left_mask.sum().item())
            right_count = int(right_mask.sum().item())
            if left_count < min_pts or right_count < min_pts:
                continue
            left = points[left_mask]
            right = points[right_mask]
            try:
                _, _, _, _, score_l, area_l = _fit_leaf(left)
                _, _, _, _, score_r, area_r = _fit_leaf(right)
            except Exception:
                continue
            split_area = area_l + area_r
            split_waste = _waste(score_l, area_l) + _waste(score_r, area_r)
            area_reduction = 1.0 - (split_area / max(area, 1.0e-8))
            best_key = split_waste - max(area_reduction, 0.0) * 0.25 * area
            if best is None or best_key < best[0]:
                best = (
                    best_key,
                    split_waste,
                    split_area,
                    area_reduction,
                    left,
                    right,
                    axis,
                    float(cut_value.detach().cpu().item()),
                    score_l,
                    score_r,
                )

        if best is None:
            return [(R_bw, centre, half_sz, points.detach().cpu(), support)]

        _, split_waste, split_area, area_reduction, left, right, axis, cut_value, score_l, score_r = best
        support_improved = split_waste <= current_waste * (1.0 - min_improvement)
        envelope_improved = large_envelope and area_reduction >= min_area_reduction and min(score_l, score_r) >= min_support
        if not support_improved and not envelope_improved:
            return [(R_bw, centre, half_sz, points.detach().cpu(), support)]

        split_reason = "support" if support_improved else "envelope"
        print(
            f"[{gid}] Support split depth={depth} reason={split_reason} axis={axis} cut={cut_value:.3f} "
            f"support {support:.2f}->{score_l:.2f}/{score_r:.2f} "
            f"area {area:.2f}->{split_area:.2f} ({area_reduction * 100:.1f}% less)"
        )
        return _recurse(left, depth + 1) + _recurse(right, depth + 1)

    return _recurse(P, 0)


def _box_corners_world(R_bw: torch.Tensor, centre: torch.Tensor, half_sz: torch.Tensor) -> torch.Tensor:
    signs = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        device=centre.device,
        dtype=centre.dtype,
    )
    return (signs * half_sz.unsqueeze(0)) @ R_bw.t() + centre.unsqueeze(0)


def _box_footprint_area(half_sz: torch.Tensor) -> float:
    return float((4.0 * half_sz[0].clamp(min=1.0e-6) * half_sz[1].clamp(min=1.0e-6)).detach().cpu().item())


def _footprint_overlap_in_ref(a: Dict, b: Dict) -> Tuple[float, float, float, float]:
    """Return inter area, IoU, inter/area_a, inter/area_b in box a's local XY AABB frame."""
    R_ref = a["R_bw"]
    C_ref = a["centre"]
    half_a = a["half_sz"]
    min_a = -half_a[:2]
    max_a = half_a[:2]

    corners_b = _box_corners_world(b["R_bw"], b["centre"], b["half_sz"])
    local_b = (corners_b - C_ref.unsqueeze(0)) @ R_ref
    min_b = local_b[:, :2].min(dim=0).values
    max_b = local_b[:, :2].max(dim=0).values

    overlap = torch.minimum(max_a, max_b) - torch.maximum(min_a, min_b)
    inter_xy = torch.clamp(overlap, min=0.0)
    inter = float((inter_xy[0] * inter_xy[1]).detach().cpu().item())
    area_a = float((torch.clamp(max_a[0] - min_a[0], min=1.0e-6) * torch.clamp(max_a[1] - min_a[1], min=1.0e-6)).detach().cpu().item())
    area_b = float((torch.clamp(max_b[0] - min_b[0], min=1.0e-6) * torch.clamp(max_b[1] - min_b[1], min=1.0e-6)).detach().cpu().item())
    union = max(area_a + area_b - inter, 1.0e-8)
    return (
        inter,
        min(inter / union, 1.0),
        min(inter / max(area_a, 1.0e-8), 1.0),
        min(inter / max(area_b, 1.0e-8), 1.0),
    )


def _footprint_gap_in_ref(a: Dict, b: Dict) -> Tuple[float, float]:
    """Return XY AABB gaps in box a's local footprint frame."""
    R_ref = a["R_bw"]
    C_ref = a["centre"]
    min_a = -a["half_sz"][:2]
    max_a = a["half_sz"][:2]
    corners_b = _box_corners_world(b["R_bw"], b["centre"], b["half_sz"])
    local_b = (corners_b - C_ref.unsqueeze(0)) @ R_ref
    min_b = local_b[:, :2].min(dim=0).values
    max_b = local_b[:, :2].max(dim=0).values
    gap_x = torch.clamp(torch.maximum(min_b[0] - max_a[0], min_a[0] - max_b[0]), min=0.0)
    gap_y = torch.clamp(torch.maximum(min_b[1] - max_a[1], min_a[1] - max_b[1]), min=0.0)
    return float(gap_x.detach().cpu().item()), float(gap_y.detach().cpu().item())


def _same_coplanar_layer(
    a: Dict,
    b: Dict,
    *,
    normal_tol_deg: float = 8.0,
    normal_sep_tol: float = 0.035,
) -> bool:
    n_a = F.normalize(a["R_bw"][:, 2], dim=0)
    n_b = F.normalize(b["R_bw"][:, 2], dim=0)
    if float(torch.abs(torch.dot(n_a, n_b)).detach().cpu().item()) < float(np.cos(np.deg2rad(normal_tol_deg))):
        return False
    normal_sep = float(torch.abs(torch.dot(b["centre"] - a["centre"], n_a)).detach().cpu().item())
    thickness_allow = 0.5 * float((a["half_sz"][2] + b["half_sz"][2]).detach().cpu().item()) + normal_sep_tol
    return normal_sep <= max(normal_sep_tol, thickness_allow)


def _points_covered_by_box(points: torch.Tensor, box: Dict, *, xy_margin: float = 0.035, z_margin: float = 0.035) -> float:
    if points is None or points.numel() == 0:
        return 0.0
    pts = points.to(device=box["centre"].device, dtype=box["centre"].dtype)
    local = (pts - box["centre"].unsqueeze(0)) @ box["R_bw"]
    half = box["half_sz"]
    inside = (
        (local[:, 0].abs() <= half[0] + xy_margin)
        & (local[:, 1].abs() <= half[1] + xy_margin)
        & (local[:, 2].abs() <= half[2] + z_margin)
    )
    return float(inside.float().mean().detach().cpu().item())


def _filter_box_supported_points(
    points: torch.Tensor,
    R_bw: torch.Tensor,
    centre: torch.Tensor,
    half_sz: torch.Tensor,
    *,
    xy_margin: float = 0.005,
    z_margin: float = 0.005,
    min_points: int = 50,
    min_keep_fraction: float = 0.30,
) -> Tuple[torch.Tensor, float]:
    """Keep only points represented by the fitted primitive for SQ refinement/dedup."""
    if points is None or points.numel() == 0:
        return points, 0.0
    pts = points.to(device=centre.device, dtype=centre.dtype)
    local = (pts - centre.unsqueeze(0)) @ R_bw
    inside = (
        (local[:, 0].abs() <= half_sz[0] + xy_margin)
        & (local[:, 1].abs() <= half_sz[1] + xy_margin)
        & (local[:, 2].abs() <= half_sz[2] + z_margin)
    )
    keep_count = int(inside.sum().item())
    keep_fraction = keep_count / max(int(pts.shape[0]), 1)
    if keep_count >= min_points and keep_fraction >= min_keep_fraction:
        return pts[inside], keep_fraction
    return pts, keep_fraction


def _prune_coplanar_overlapping_boxes(
    records: List[Dict],
    *,
    footprint_cover_thresh: float = 0.82,
    point_cover_thresh: float = 0.90,
    high_iou_thresh: float = 0.60,
) -> List[Dict]:
    """Drop duplicate same-layer boxes while preserving adjacent non-overlapping pieces."""
    if len(records) <= 1:
        return records

    def _priority(rec: Dict) -> Tuple[float, float, float]:
        point_count = float(rec["points"].shape[0]) if rec.get("points") is not None else 0.0
        support = float(rec.get("support", 0.0))
        return (point_count * (0.5 + support), _box_footprint_area(rec["half_sz"]), support)

    kept: List[Dict] = []
    dropped: List[Tuple[Dict, Dict, str]] = []
    for rec in sorted(records, key=_priority, reverse=True):
        drop_against = None
        drop_reason = ""
        for existing in kept:
            if not _same_coplanar_layer(existing, rec):
                continue
            _, iou, _, overlap_rec = _footprint_overlap_in_ref(existing, rec)
            point_cover = _points_covered_by_box(rec["points"], existing)
            if overlap_rec >= footprint_cover_thresh or point_cover >= point_cover_thresh or iou >= high_iou_thresh:
                drop_against = existing
                drop_reason = f"iou={iou:.2f}, footprint_cover={overlap_rec:.2f}, point_cover={point_cover:.2f}"
                break
        if drop_against is None:
            kept.append(rec)
        else:
            dropped.append((rec, drop_against, drop_reason))

    if dropped:
        print(f"[dedup] Dropped {len(dropped)} coplanar overlapping duplicate boxes ({len(records)} -> {len(kept)}).")
        for rec, existing, reason in dropped[:12]:
            print(
                f"[dedup] drop {rec.get('gid')}.{rec.get('piece_idx')} "
                f"covered by {existing.get('gid')}.{existing.get('piece_idx')} ({reason})"
            )
        if len(dropped) > 12:
            print(f"[dedup] ... {len(dropped) - 12} more drops omitted")
    return kept


def _fixed_normal_plane_inliers(
    P: torch.Tensor,
    n: torch.Tensor,
    *,
    thresh: float = 0.045,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit only plane offset while keeping the cluster-majority normal fixed."""
    n = F.normalize(n, dim=0)
    offsets = P @ n
    offset = torch.median(offsets)
    inliers = (offsets - offset).abs() <= float(thresh)
    if int(inliers.sum().item()) < 3:
        inliers = torch.ones((P.shape[0],), dtype=torch.bool, device=P.device)
    c = P[inliers].mean(dim=0)
    return n, c, inliers


def _consensus_axis_hint(
    candidates: List[Dict],
    target_n: torch.Tensor,
    *,
    normal_tol_deg: float = 10.0,
    min_votes: int = 6,
    min_confidence: float = 0.45,
    max_area_weight: float = 5.0,
) -> Tuple[torch.Tensor, float, int] | None:
    """
    Estimate a shared in-plane axis for a normal family.

    The angle is averaged modulo 90 degrees, because a box frame and its
    orthogonal swap represent the same rectangular footprint family. This keeps
    stair/platform pieces edge-to-edge instead of letting each incomplete
    footprint pick its own min-area slanted axis.
    """
    if not candidates:
        return None
    target_n = F.normalize(target_n, dim=0)
    u0, v0, _ = make_local_frame(target_n)
    cos_tol = float(np.cos(np.deg2rad(normal_tol_deg)))
    C = torch.zeros((), device=target_n.device, dtype=target_n.dtype)
    S = torch.zeros((), device=target_n.device, dtype=target_n.dtype)
    total = torch.zeros((), device=target_n.device, dtype=target_n.dtype)
    votes = 0

    for cand in candidates:
        cand_n = cand["normal"].to(device=target_n.device, dtype=target_n.dtype)
        if float(torch.abs(torch.dot(F.normalize(cand_n, dim=0), target_n)).detach().cpu().item()) < cos_tol:
            continue
        R = cand["R_bw"].to(device=target_n.device, dtype=target_n.dtype)
        area = min(_box_footprint_area(cand["half_sz"]), max_area_weight)
        support = max(float(cand.get("support", 0.0)), 0.10)
        planarity = max(float(cand.get("planarity_ratio", 1.0)), 0.25)
        weight = torch.tensor(area * support * planarity, device=target_n.device, dtype=target_n.dtype)
        if float(weight.detach().cpu().item()) <= 1.0e-6:
            continue

        for axis_idx in (0, 1):
            axis = R[:, axis_idx]
            axis = axis - (axis @ target_n) * target_n
            if torch.linalg.norm(axis) < 1.0e-6:
                continue
            axis = F.normalize(axis, dim=0)
            theta = torch.atan2(torch.dot(axis, v0), torch.dot(axis, u0))
            C = C + weight * torch.cos(4.0 * theta)
            S = S + weight * torch.sin(4.0 * theta)
            total = total + weight
            votes += 1

    if votes < min_votes or float(total.detach().cpu().item()) <= 1.0e-6:
        return None

    confidence = torch.sqrt(C * C + S * S) / total.clamp(min=1.0e-8)
    confidence_f = float(confidence.detach().cpu().item())
    if confidence_f < min_confidence:
        return None

    theta = 0.25 * torch.atan2(S, C)
    axis_hint = F.normalize(torch.cos(theta) * u0 + torch.sin(theta) * v0, dim=0)
    return axis_hint, confidence_f, votes


def _merge_normal_similar_segment_candidates(
    candidates: List[Dict],
    *,
    normal_tol_deg: float = 10.0,
    normal_sep_tol: float = 0.060,
    min_iou: float = 0.25,
    min_cover: float = 0.72,
) -> List[List[Dict]]:
    """Merge same-layer segment candidates before fitting primitives."""
    if len(candidates) <= 1:
        return [[cand] for cand in candidates]

    cos_tol = float(np.cos(np.deg2rad(normal_tol_deg)))
    parent = list(range(len(candidates)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a = candidates[i]
            b = candidates[j]
            n_dot = float(torch.abs(torch.dot(a["normal"], b["normal"])).detach().cpu().item())
            if n_dot < cos_tol:
                continue
            if not _same_coplanar_layer(a, b, normal_tol_deg=normal_tol_deg, normal_sep_tol=normal_sep_tol):
                continue
            _, iou_ab, cover_a, cover_b = _footprint_overlap_in_ref(a, b)
            footprint_related = (
                iou_ab >= min_iou
                or max(cover_a, cover_b) >= min_cover
            )
            if footprint_related:
                union(i, j)

    groups_by_root: Dict[int, List[Dict]] = {}
    for idx, cand in enumerate(candidates):
        groups_by_root.setdefault(find(idx), []).append(cand)
    groups = list(groups_by_root.values())
    if len(groups) != len(candidates):
        merged_count = len(candidates) - len(groups)
        sizes = sorted((len(group) for group in groups if len(group) > 1), reverse=True)
        print(f"[cluster-merge] merged {merged_count} normal-similar same-layer clusters ({len(candidates)} -> {len(groups)}), group_sizes={sizes[:12]}")
    return groups
    
def process_global_segments(
    global_segments: Dict,
    all_frame_segments: List,
    min_frames: int = 1,
    device: str = 'cuda',
    fps_ratio: float = 0.3,
    min_fps_points: int = 100,
    use_plane_constraint: bool = True
) -> Dict[str, List[torch.Tensor]]:
    """
    Process global segments with compact 3D bounding box estimation.

    """
    segment_candidates: List[Dict] = []
    piece_records: List[Dict] = []
    legacy_stair75_fit = _env_flag("CRISP_SQS_LEGACY_STAIR75", False)

    for gid, frame_segs in global_segments.items():
        if len(frame_segs) < min_frames:
            continue
        pts_list, normal_list = [], []
        for fi, lid in frame_segs:
            seg = all_frame_segments[fi][lid]
            world_points = seg['world_points'].to(device)
            avg_normal   = seg['avg_normal'].to(device)
            assert world_points.dim() == 2 and world_points.shape[1] == 3, f"world_points must be (N,3), got {world_points.shape}"
            assert avg_normal.dim() == 1 and avg_normal.shape[0] == 3,     f"avg_normal must be (3,), got {avg_normal.shape}"
            finite_points = torch.isfinite(world_points).all(dim=1) & (torch.linalg.norm(world_points, dim=1) > 1.0e-8)
            if int(finite_points.sum().item()) < 3:
                continue
            pts_list.append(world_points[finite_points])
            normal_list.append(avg_normal)

        if not pts_list:
            continue

        P_all = torch.cat(pts_list, 0)
        assert P_all.dim() == 2 and P_all.shape[1] == 3, f"Concatenated points must be (N, 3), got {P_all.shape}"
        n_avg = _oriented_mean_normal(torch.stack(normal_list))

        if use_plane_constraint:
            n, c, inliers = robust_plane_ransac(P_all, n_avg)
            if n is None or c is None or inliers is None:
                n = n_avg
                c = P_all.mean(dim=0)
                inliers = torch.ones((P_all.shape[0],), dtype=torch.bool, device=P_all.device)
            planarity_ratio = (inliers.float().mean()).item()
            P = P_all[inliers] if inliers.sum() > 50 else P_all
        else:
            n = n_avg
            c = P_all.mean(dim=0)
            P = P_all
            planarity_ratio = 1.0

        if P.shape[0] < 3:
            continue
        try:
            R_bw, centre, half_sz, _, support, _ = _fit_supported_plane_box(
                P,
                n,
                c,
                gid,
                edge_trim=not legacy_stair75_fit,
            )
        except Exception:
            R_bw, centre, half_sz, _ = fit_one_plane_box(P, n, c, gid)
            support = 0.0
        half_sz = half_sz.clamp(min=0.02)
        segment_candidates.append(
            {
                "gid": gid,
                "points": P.detach(),
                "all_points": P_all.detach(),
                "normal": F.normalize(n, dim=0),
                "vote_count": int(P.shape[0]),
                "planarity_ratio": float(planarity_ratio),
                "R_bw": R_bw,
                "centre": centre,
                "half_sz": half_sz,
                "support": float(support),
            }
        )

    candidate_groups = (
        _merge_normal_similar_segment_candidates(segment_candidates)
        if use_plane_constraint
        else [[cand] for cand in segment_candidates]
    )

    for group_idx, group in enumerate(candidate_groups):
        if not group:
            continue
        group_points = torch.cat([cand["points"].to(device) for cand in group], dim=0)
        if group_points.shape[0] < 3:
            continue
        dominant = max(group, key=lambda cand: cand["vote_count"])
        n = dominant["normal"].to(device=device, dtype=group_points.dtype)
        for cand in group:
            cand_n = cand["normal"].to(device=device, dtype=group_points.dtype)
            if torch.dot(cand_n, n) < 0:
                cand["normal"] = -cand["normal"]
        n, c, inliers = _fixed_normal_plane_inliers(group_points, n, thresh=0.045)
        P = group_points[inliers] if int(inliers.sum().item()) > 50 else group_points
        gid_label = dominant["gid"] if len(group) == 1 else f"{dominant['gid']}+{len(group)-1}"
        mean_planarity = float(np.mean([cand["planarity_ratio"] for cand in group]))
        axis_hint_info = None if legacy_stair75_fit else _consensus_axis_hint(segment_candidates, n)
        axis_hint = axis_hint_info[0] if axis_hint_info is not None else None
        axis_msg = ""
        if axis_hint_info is not None:
            _, axis_confidence, axis_votes = axis_hint_info
            axis_msg = f", axis_hint_conf={axis_confidence:.2f}, axis_votes={axis_votes}"
        print(
            f"[{gid_label}] Fitting merged normal cluster "
            f"(segments={len(group)}, dominant={dominant['gid']}, points={int(P.shape[0])}, "
            f"mean_planarity={mean_planarity:.2f}{axis_msg})"
        )
        if legacy_stair75_fit:
            R_bw, centre, half_sz, _, support, _ = _fit_supported_plane_box(
                P,
                n,
                c,
                gid_label,
                axis_hint=None,
                edge_trim=False,
            )
            pieces = [(R_bw, centre, half_sz, P.detach().cpu(), support)]
        else:
            pieces = fit_support_aware_plane_boxes(
                P,
                n,
                c,
                gid_label,
                max_depth=2,
                min_pts=180,
                min_support=0.48,
                min_improvement=0.12,
                axis_hint=axis_hint,
            )
        if not pieces:
            R_bw, centre, half_sz, _ = fit_one_plane_box(P, n, c, gid_label)
            pieces = [(R_bw, centre, half_sz, P.detach().cpu(), 0.0)]

        for piece_idx, (R_bw, centre, half_sz, piece_points, support) in enumerate(pieces):
            half_sz = half_sz.clamp(min=0.02)
            piece_points_dev = piece_points.to(device=centre.device, dtype=centre.dtype)
            fit_points_dev, fit_cover = _filter_box_supported_points(
                piece_points_dev,
                R_bw,
                centre,
                half_sz,
            )
            piece_records.append(
                {
                    "gid": gid_label,
                    "piece_idx": piece_idx,
                    "R_bw": R_bw,
                    "centre": centre,
                    "half_sz": half_sz,
                    "points": fit_points_dev,
                    "raw_point_count": int(piece_points_dev.shape[0]),
                    "fit_cover": float(fit_cover),
                    "support": float(support),
                }
            )
            print(
                f"[{gid_label}.{piece_idx}] Box dimensions: {(half_sz * 2).cpu().numpy()} "
                f"support={support:.2f} points={int(piece_points.shape[0])} "
                f"fit_points={int(fit_points_dev.shape[0])} fit_cover={fit_cover:.2f}"
            )

    piece_records = _prune_coplanar_overlapping_boxes(piece_records)
    S_items, R_items, T_items, pts_items = [], [], [], []
    for export_idx, rec in enumerate(piece_records):
        half_sz = rec["half_sz"].clamp(min=0.02)
        R_bw = rec["R_bw"]
        centre = rec["centre"]
        piece_points = rec["points"]
        normal_np = F.normalize(R_bw[:, 2], dim=0).detach().cpu().numpy()
        centre_np = centre.detach().cpu().numpy()
        print(
            f"[export-primitive] idx={export_idx} gid={rec.get('gid')}.{rec.get('piece_idx')} "
            f"dims={(half_sz * 2).detach().cpu().numpy()} support={float(rec.get('support', 0.0)):.2f} "
            f"points={int(piece_points.shape[0]) if piece_points is not None else 0} "
            f"raw_points={int(rec.get('raw_point_count', int(piece_points.shape[0]) if piece_points is not None else 0))} "
            f"fit_cover={float(rec.get('fit_cover', 1.0)):.2f} "
            f"normal={normal_np} centre={centre_np}"
        )
        S_items.append(torch.log(half_sz))
        R_items.append(R_bw.t().contiguous())  # stored as world→body; export converts back to body→world
        T_items.append(centre)
        pts_items.append(piece_points.detach().cpu())

        assert S_items[-1].shape == (3,), f"S_item must be (3,), got {S_items[-1].shape}"
        assert R_items[-1].shape == (3, 3), f"R_item must be (3, 3), got {R_items[-1].shape}"
        assert T_items[-1].shape == (3,), f"T_item must be (3,), got {T_items[-1].shape}"

    return dict(S_items=S_items, R_items=R_items, T_items=T_items, pts_items=pts_items)


def _fit_plane_box_from_points(
    world_points: torch.Tensor,
    gid: int,
    device: str = 'cuda',
    use_plane_constraint: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit a plane-aligned box to one point set and return the fitted plane metadata."""
    world_points = world_points.to(device=device, dtype=torch.float32)
    if world_points.ndim != 2 or world_points.shape[1] != 3:
        raise ValueError(f"world_points must be (N, 3), got {tuple(world_points.shape)}")
    if world_points.shape[0] < 3:
        raise ValueError("Need at least 3 points to fit a plane-constrained box.")

    k = max(3, min(20, int(world_points.shape[0])))
    avg_normal = estimate_normals_knn(world_points, k=k).to(device)
    avg_normal = F.normalize(avg_normal.mean(dim=0), dim=0)

    P_all = world_points
    if use_plane_constraint:
        n, c, inliers = robust_plane_ransac(P_all, avg_normal)
        if n is None or c is None or inliers is None:
            n = avg_normal
            c = P_all.mean(dim=0)
            inliers = torch.ones((P_all.shape[0],), dtype=torch.bool, device=P_all.device)
    else:
        n = avg_normal
        c = P_all.mean(dim=0)
        inliers = torch.ones((P_all.shape[0],), dtype=torch.bool, device=P_all.device)

    P = P_all[inliers] if int(inliers.sum().item()) > 50 else P_all
    R_bw, centre, half_sz, _ = fit_one_plane_box(P, n, c, gid)
    half_sz = half_sz.clamp(min=0.02)
    return R_bw, centre, half_sz, P_all, P, n, c


def _extend_contact_box_to_scene_barriers(
    contact_points: torch.Tensor,
    base_R_bw: torch.Tensor,
    base_center: torch.Tensor,
    base_half_sz: torch.Tensor,
    scene_primitives: Optional[Dict[str, List[torch.Tensor]]] = None,
    device: str = 'cuda',
    perpendicular_tol_deg: float = 20.0,
    min_scene_planarity: float = 0.60,
    max_extension: float = 1.25,
    max_hinge_distance: float = 2.5,
    overlap_tol: float = 0.10,
    boundary_clearance: float = 0.02,
    hinge_snap_clearance: float = 0.003,
    point_padding: float = 0.01,
    contact_extent_scale: float = 1.10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Expand a contact-support plane toward nearby nearly-perpendicular scene planes.

    The box is first guaranteed to contain the selected contact points with a slightly inflated
    in-plane footprint. Then each in-plane side can grow up to the closest nearby perpendicular
    barrier, but never past that barrier.
    """
    contact_points = contact_points.to(device=device, dtype=torch.float32)
    x_axis = F.normalize(base_R_bw[:, 0], dim=0)
    y_axis = F.normalize(base_R_bw[:, 1], dim=0)
    n_contact = F.normalize(base_R_bw[:, 2], dim=0)
    R_contact = torch.stack([x_axis, y_axis, n_contact], dim=1)

    local_contact = (contact_points - base_center) @ R_contact
    u_min_raw = float(local_contact[:, 0].min().item()) - point_padding
    u_max_raw = float(local_contact[:, 0].max().item()) + point_padding
    v_min_raw = float(local_contact[:, 1].min().item()) - point_padding
    v_max_raw = float(local_contact[:, 1].max().item()) + point_padding

    def _inflate_interval(lo: float, hi: float, scale: float) -> Tuple[float, float]:
        ctr = 0.5 * (lo + hi)
        half = max(0.5 * (hi - lo), 0.02)
        half *= scale
        return ctr - half, ctr + half

    u_min, u_max = _inflate_interval(u_min_raw, u_max_raw, contact_extent_scale)
    v_min, v_max = _inflate_interval(v_min_raw, v_max_raw, contact_extent_scale)
    contact_u_min = u_min
    contact_u_max = u_max
    contact_v_min = v_min
    contact_v_max = v_max
    z_half = float(max(base_half_sz[2].item(), 0.02))

    print(
        f"[contact] Wrapped selected points with {contact_extent_scale * 100:.0f}% in-plane extent "
        f"(u: {u_min:.3f}..{u_max:.3f}, v: {v_min:.3f}..{v_max:.3f})."
    )

    if not scene_primitives or not scene_primitives.get("pts_items"):
        center_local = torch.tensor(
            [(u_min + u_max) * 0.5, (v_min + v_max) * 0.5, 0.0],
            device=device,
            dtype=base_center.dtype,
        )
        half_ext = torch.tensor(
            [
                max((u_max - u_min) * 0.5, 0.02),
                max((v_max - v_min) * 0.5, 0.02),
                z_half,
            ],
            device=device,
            dtype=base_half_sz.dtype,
        )
        centre = base_center + R_contact @ center_local
        return R_contact, centre, half_ext

    u_ref = float(local_contact[:, 0].mean().item())
    v_ref = float(local_contact[:, 1].mean().item())

    sin_tol = float(np.sin(np.deg2rad(perpendicular_tol_deg)))
    best_hit = {"u_pos": None, "u_neg": None, "v_pos": None, "v_neg": None}
    best_gap = {key: float("inf") for key in best_hit}
    hinge_candidate = None
    hinge_cost = float("inf")
    chosen_hinge = None

    for scene_idx, pts_item in enumerate(scene_primitives.get("pts_items", [])):
        if pts_item is None:
            continue
        scene_pts = pts_item.to(device=device, dtype=torch.float32) if hasattr(pts_item, "to") else torch.as_tensor(pts_item, device=device, dtype=torch.float32)
        if scene_pts.ndim != 2 or scene_pts.shape[1] != 3 or scene_pts.shape[0] < 80:
            continue

        try:
            k = max(3, min(20, int(scene_pts.shape[0])))
            avg_normal = estimate_normals_knn(scene_pts, k=k).to(device)
            avg_normal = F.normalize(avg_normal.mean(dim=0), dim=0)
            n_scene, c_scene, inliers = robust_plane_ransac(scene_pts, avg_normal)
        except Exception:
            continue

        if n_scene is None or c_scene is None or inliers is None:
            continue

        planarity_ratio = float(inliers.float().mean().item())
        if planarity_ratio < min_scene_planarity:
            continue

        n_scene = F.normalize(n_scene, dim=0)
        if abs(float(torch.dot(n_scene, n_contact).item())) > sin_tol:
            continue

        scene_support = scene_pts[inliers] if int(inliers.sum().item()) > 50 else scene_pts
        local_scene = (scene_support - base_center) @ R_contact
        scene_u_min = float(local_scene[:, 0].min().item())
        scene_u_max = float(local_scene[:, 0].max().item())
        scene_v_min = float(local_scene[:, 1].min().item())
        scene_v_max = float(local_scene[:, 1].max().item())
        scene_u_center = 0.5 * (scene_u_min + scene_u_max)
        scene_v_center = 0.5 * (scene_v_min + scene_v_max)
        center_dist = float(torch.linalg.norm(c_scene - base_center).item())

        a = float(torch.dot(n_scene, x_axis).item())
        b = float(torch.dot(n_scene, y_axis).item())
        d = float(torch.dot(n_scene, base_center - c_scene).item())
        if max(abs(a), abs(b)) < 1e-6:
            continue

        gap_u_pos = max(0.0, scene_u_min - u_max)
        gap_u_neg = max(0.0, u_min - scene_u_max)
        gap_v_pos = max(0.0, scene_v_min - v_max)
        gap_v_neg = max(0.0, v_min - scene_v_max)
        overlap_u_sep = max(0.0, max(u_min - scene_u_max, scene_u_min - u_max))
        overlap_v_sep = max(0.0, max(v_min - scene_v_max, scene_v_min - v_max))

        if abs(a) >= abs(b):
            if scene_v_max < v_min - overlap_tol or scene_v_min > v_max + overlap_tol:
                v_clamped = scene_v_center
            else:
                v_clamped = min(max(v_ref, max(v_min, scene_v_min)), min(v_max, scene_v_max))
            u_hit = (-d - b * v_clamped) / a
            if u_hit > u_max + boundary_clearance:
                gap = u_hit - u_max
                if gap <= max_extension and gap < best_gap["u_pos"]:
                    best_gap["u_pos"] = gap
                    best_hit["u_pos"] = (u_hit, scene_idx)
            elif u_hit < u_min - boundary_clearance:
                gap = u_min - u_hit
                if gap <= max_extension and gap < best_gap["u_neg"]:
                    best_gap["u_neg"] = gap
                    best_hit["u_neg"] = (u_hit, scene_idx)
            hinge_side = None
            hinge_value = None
            if gap_u_pos > 0.0:
                hinge_side = "u_pos"
                hinge_value = u_hit
            elif gap_u_neg > 0.0:
                hinge_side = "u_neg"
                hinge_value = u_hit
            if hinge_side is not None and center_dist <= max_hinge_distance:
                cost = center_dist + overlap_v_sep
                if cost < hinge_cost:
                    hinge_cost = cost
                    hinge_candidate = (hinge_side, hinge_value, scene_idx, center_dist)
        else:
            if scene_u_max < u_min - overlap_tol or scene_u_min > u_max + overlap_tol:
                u_clamped = scene_u_center
            else:
                u_clamped = min(max(u_ref, max(u_min, scene_u_min)), min(u_max, scene_u_max))
            v_hit = (-d - a * u_clamped) / b
            if v_hit > v_max + boundary_clearance:
                gap = v_hit - v_max
                if gap <= max_extension and gap < best_gap["v_pos"]:
                    best_gap["v_pos"] = gap
                    best_hit["v_pos"] = (v_hit, scene_idx)
            elif v_hit < v_min - boundary_clearance:
                gap = v_min - v_hit
                if gap <= max_extension and gap < best_gap["v_neg"]:
                    best_gap["v_neg"] = gap
                    best_hit["v_neg"] = (v_hit, scene_idx)
            hinge_side = None
            hinge_value = None
            if gap_v_pos > 0.0:
                hinge_side = "v_pos"
                hinge_value = v_hit
            elif gap_v_neg > 0.0:
                hinge_side = "v_neg"
                hinge_value = v_hit
            if hinge_side is not None and center_dist <= max_hinge_distance:
                cost = center_dist + overlap_u_sep
                if cost < hinge_cost:
                    hinge_cost = cost
                    hinge_candidate = (hinge_side, hinge_value, scene_idx, center_dist)

    if best_hit["u_pos"] is not None:
        hit, scene_idx = best_hit["u_pos"]
        new_u_max = max(u_max, hit - boundary_clearance)
        print(f"[contact] Extend +u from {u_max:.3f} to {new_u_max:.3f} using near-perpendicular scene plane {scene_idx}.")
        u_max = new_u_max
        chosen_hinge = ("u_pos", hit, scene_idx)
    if best_hit["u_neg"] is not None:
        hit, scene_idx = best_hit["u_neg"]
        new_u_min = min(u_min, hit + boundary_clearance)
        print(f"[contact] Extend -u from {u_min:.3f} to {new_u_min:.3f} using near-perpendicular scene plane {scene_idx}.")
        u_min = new_u_min
        chosen_hinge = ("u_neg", hit, scene_idx)
    if best_hit["v_pos"] is not None:
        hit, scene_idx = best_hit["v_pos"]
        new_v_max = max(v_max, hit - boundary_clearance)
        print(f"[contact] Extend +v from {v_max:.3f} to {new_v_max:.3f} using near-perpendicular scene plane {scene_idx}.")
        v_max = new_v_max
        chosen_hinge = ("v_pos", hit, scene_idx)
    if best_hit["v_neg"] is not None:
        hit, scene_idx = best_hit["v_neg"]
        new_v_min = min(v_min, hit + boundary_clearance)
        print(f"[contact] Extend -v from {v_min:.3f} to {new_v_min:.3f} using near-perpendicular scene plane {scene_idx}.")
        v_min = new_v_min
        chosen_hinge = ("v_neg", hit, scene_idx)

    if all(value is None for value in best_hit.values()) and hinge_candidate is not None:
        side, hit, scene_idx, center_dist = hinge_candidate
        if side == "u_pos":
            new_u_max = max(u_max, hit - boundary_clearance)
            print(f"[contact] Hinge +u from {u_max:.3f} to {new_u_max:.3f} using nearest planar primitive {scene_idx} (dist {center_dist:.3f}).")
            u_max = new_u_max
        elif side == "u_neg":
            new_u_min = min(u_min, hit + boundary_clearance)
            print(f"[contact] Hinge -u from {u_min:.3f} to {new_u_min:.3f} using nearest planar primitive {scene_idx} (dist {center_dist:.3f}).")
            u_min = new_u_min
        elif side == "v_pos":
            new_v_max = max(v_max, hit - boundary_clearance)
            print(f"[contact] Hinge +v from {v_max:.3f} to {new_v_max:.3f} using nearest planar primitive {scene_idx} (dist {center_dist:.3f}).")
            v_max = new_v_max
        elif side == "v_neg":
            new_v_min = min(v_min, hit + boundary_clearance)
            print(f"[contact] Hinge -v from {v_min:.3f} to {new_v_min:.3f} using nearest planar primitive {scene_idx} (dist {center_dist:.3f}).")
            v_min = new_v_min
        chosen_hinge = (side, hit, scene_idx)

    if all(value is None for value in best_hit.values()) and hinge_candidate is None:
        print("[contact] No nearby near-perpendicular scene plane found; using wrapped contact patch only.")

    if chosen_hinge is not None:
        side, hit, scene_idx = chosen_hinge
        if side == "u_pos":
            desired_edge = hit - hinge_snap_clearance
            delta = desired_edge - u_max
            if abs(delta) > 1e-6:
                u_min += delta
                u_max += delta
                if u_min > contact_u_min:
                    u_min = contact_u_min
                if u_max < contact_u_max:
                    u_max = contact_u_max
                print(f"[contact] Nudge +u hinge toward scene plane {scene_idx}: delta {delta:.3f}.")
        elif side == "u_neg":
            desired_edge = hit + hinge_snap_clearance
            delta = desired_edge - u_min
            if abs(delta) > 1e-6:
                u_min += delta
                u_max += delta
                if u_min > contact_u_min:
                    u_min = contact_u_min
                if u_max < contact_u_max:
                    u_max = contact_u_max
                print(f"[contact] Nudge -u hinge toward scene plane {scene_idx}: delta {delta:.3f}.")
        elif side == "v_pos":
            desired_edge = hit - hinge_snap_clearance
            delta = desired_edge - v_max
            if abs(delta) > 1e-6:
                v_min += delta
                v_max += delta
                if v_min > contact_v_min:
                    v_min = contact_v_min
                if v_max < contact_v_max:
                    v_max = contact_v_max
                print(f"[contact] Nudge +v hinge toward scene plane {scene_idx}: delta {delta:.3f}.")
        elif side == "v_neg":
            desired_edge = hit + hinge_snap_clearance
            delta = desired_edge - v_min
            if abs(delta) > 1e-6:
                v_min += delta
                v_max += delta
                if v_min > contact_v_min:
                    v_min = contact_v_min
                if v_max < contact_v_max:
                    v_max = contact_v_max
                print(f"[contact] Nudge -v hinge toward scene plane {scene_idx}: delta {delta:.3f}.")

    center_local = torch.tensor(
        [(u_min + u_max) * 0.5, (v_min + v_max) * 0.5, 0.0],
        device=device,
        dtype=base_center.dtype,
    )
    half_ext = torch.tensor(
        [
            max((u_max - u_min) * 0.5, 0.02),
            max((v_max - v_min) * 0.5, 0.02),
            z_half,
        ],
        device=device,
        dtype=base_half_sz.dtype,
    )
    centre = base_center + R_contact @ center_local
    return R_contact, centre, half_ext



def process_global_points(
    world_points: torch.Tensor,
    min_frames: int = 2,
    device: str = 'cuda',
    fps_ratio: float = 0.3,
    min_fps_points: int = 100,
    use_plane_constraint: bool = True,
    scene_primitives: Optional[Dict[str, List[torch.Tensor]]] = None,
) -> Dict[str, List[torch.Tensor]]:
    """
    Process global segments with compact 3D bounding box estimation.

    """
    S_items, R_items, T_items, pts_items = [], [], [], []

    gid = 0
    if True:
        R_bw, centre, half_sz, P_all, P, _, _ = _fit_plane_box_from_points(
            world_points,
            gid=gid,
            device=device,
            use_plane_constraint=use_plane_constraint,
        )
        planarity_ratio = float(P.shape[0]) / max(float(P_all.shape[0]), 1.0)
        print(f"[{gid}] Using plane-constrained bbox (planarity: {planarity_ratio:.2f})")

        R_bw, centre, half_sz = _extend_contact_box_to_scene_barriers(
            contact_points=P_all,
            base_R_bw=R_bw,
            base_center=centre,
            base_half_sz=half_sz,
            scene_primitives=scene_primitives,
            device=device,
        )

        half_sz = half_sz.clamp(min=0.02)
        S_items.append(torch.log(half_sz))
        R_items.append(R_bw.t().contiguous())  # world→body
        T_items.append(centre)
        pts_items.append(P_all.cpu())

        # checks
        assert S_items[-1].shape == (3,), f"S_item must be (3,), got {S_items[-1].shape}"
        assert R_items[-1].shape == (3, 3), f"R_item must be (3, 3), got {R_items[-1].shape}"
        assert T_items[-1].shape == (3,), f"T_item must be (3,), got {T_items[-1].shape}"
        print(f"[{gid}] Bbox dimensions: {(half_sz * 2).cpu().numpy()}")

    return dict(S_items=S_items, R_items=R_items, T_items=T_items, pts_items=pts_items)


def find_segment_correspondences_improved(
    seg_map1: np.ndarray,
    seg_map2: np.ndarray,
    props1: Dict[int, Dict],
    props2: Dict[int, Dict],
    flow_forward: np.ndarray,
    covis_mask: np.ndarray,
    normal_threshold: float = 0.9,
    min_overlap_pixels: int = 50,
    overlap_ratio_threshold: float = 0.1
) -> List[Tuple[int, int]]:
    """
    Find segment correspondences between two frames using flow warping.
    
    The logic is:
    1. For each segment in frame 1
    2. Warp all its pixels to frame 2 using optical flow
    3. Check which segments in frame 2 the warped pixels land on
    4. If enough pixels land on a segment in frame 2 AND normals are similar, mark as correspondence
    
    Args:
        seg_map1: (H, W) segmentation map for frame 1
        seg_map2: (H, W) segmentation map for frame 2
        props1: Segment properties for frame 1
        props2: Segment properties for frame 2
        flow_forward: (H, W, 2) optical flow from frame 1 to frame 2
        covis_mask: (H, W) covisibility mask
        normal_threshold: Minimum cosine similarity for normals
        min_overlap_pixels: Minimum number of pixels that must overlap
        overlap_ratio_threshold: Minimum ratio of warped pixels that must overlap
    
    Returns:
        List of (seg_id1, seg_id2) correspondence pairs
    """
    H, W = flow_forward.shape[:2]
    correspondences = []
    correspondence_scores = {}
    
    # For each segment in frame 1
    for seg1_id, props1_seg in props1.items():
        # Get all pixels for this segment
        seg1_pixels = props1_seg['pixels']
        
        # Track where warped pixels land in frame 2
        seg2_overlap_counts = {}
        valid_warped_count = 0
        
        # Warp each pixel using flow
        for y, x in seg1_pixels:
            # Check if this pixel is covisible
            if not covis_mask[y, x]:
                continue
            
            # Get flow at this pixel
            flow = flow_forward[y, x]
            
            # Warp to frame 2
            x_new = x + flow[0]
            y_new = y + flow[1]
            
            # Round to nearest pixel
            x_new = int(round(x_new))
            y_new = int(round(y_new))
            
            # Check if warped position is within bounds
            if 0 <= x_new < W and 0 <= y_new < H:
                valid_warped_count += 1
                
                # Check which segment this lands on in frame 2
                seg2_id = seg_map2[y_new, x_new]
                
                if seg2_id >= 0:  # Valid segment
                    if seg2_id not in seg2_overlap_counts:
                        seg2_overlap_counts[seg2_id] = 0
                    seg2_overlap_counts[seg2_id] += 1
        
        # No valid warped pixels
        if valid_warped_count == 0:
            continue
        
        # Check each potential match in frame 2
        for seg2_id, overlap_count in seg2_overlap_counts.items():
            if seg2_id not in props2:
                continue
            
            # Check if enough pixels overlap
            if overlap_count < min_overlap_pixels:
                continue
            
            # Calculate overlap ratio
            overlap_ratio = overlap_count / valid_warped_count
            if overlap_ratio < overlap_ratio_threshold:
                continue
            
            # Check normal similarity
            normal_sim = torch.dot(
                props1_seg['avg_normal'], 
                props2[seg2_id]['avg_normal']
            ).item()
            
            if normal_sim < normal_threshold:
                continue
            
            # Calculate combined score
            score = normal_sim * overlap_ratio * (overlap_count / 100.0)
            
            # Store correspondence with score
            key = (seg1_id, seg2_id)
            if key not in correspondence_scores or correspondence_scores[key] < score:
                correspondence_scores[key] = score
    
    # Extract final correspondences (avoiding duplicates)
    seg1_matched = set()
    seg2_matched = set()
    
    # Sort by score and take best matches
    sorted_matches = sorted(correspondence_scores.items(), key=lambda x: x[1], reverse=True)
    
    for (seg1_id, seg2_id), score in sorted_matches:
        # Each segment can only match once
        if seg1_id not in seg1_matched and seg2_id not in seg2_matched:
            correspondences.append((seg1_id, seg2_id))
            seg1_matched.add(seg1_id)
            seg2_matched.add(seg2_id)
            print(f"    Match: Seg {seg1_id} -> Seg {seg2_id}, score: {score:.3f}")
    
    return correspondences




def _save_agentic_per_frame_evidence(
    save_dir: Path,
    all_seg_maps: List[np.ndarray],
    all_frame_segments: List[Dict[int, Dict]],
    global_segments: Dict[int, List[Tuple[int, int]]],
    frame_indices: List[int],
    max_preview_frames: int = 16,
) -> None:
    """Save machine-readable per-frame segments and compact visual previews."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    segment_maps = np.stack(all_seg_maps).astype(np.int32, copy=False)
    np.savez_compressed(
        save_dir / "per_frame_segments.npz",
        segment_maps=segment_maps,
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
    )

    global_lookup = {
        (int(frame_pos), int(local_id)): int(global_id)
        for global_id, members in global_segments.items()
        for frame_pos, local_id in members
    }
    frames_payload = []
    for frame_pos, (frame_idx, props_by_id) in enumerate(zip(frame_indices, all_frame_segments)):
        segments = []
        for local_id, props in sorted(props_by_id.items()):
            pixels = np.asarray(props["pixels"], dtype=np.int32)
            if pixels.size:
                y0, x0 = pixels.min(axis=0)
                y1, x1 = pixels.max(axis=0)
                bbox_xyxy = [int(x0), int(y0), int(x1), int(y1)]
            else:
                bbox_xyxy = [0, 0, 0, 0]
            segments.append(
                {
                    "local_id": int(local_id),
                    "global_id": global_lookup.get((frame_pos, int(local_id))),
                    "pixel_count": int(len(props["pixels"])),
                    "point_count": int(props["world_points"].shape[0]),
                    "bbox_xyxy": bbox_xyxy,
                    "normal": props["avg_normal"].detach().cpu().tolist(),
                    "centroid": props["centroid"].detach().cpu().tolist(),
                }
            )
        frames_payload.append(
            {
                "frame_position": int(frame_pos),
                "frame_index": int(frame_idx),
                "segments": segments,
            }
        )

    summary = {
        "schema_version": 1,
        "frame_count": len(frame_indices),
        "global_segment_count": len(global_segments),
        "frames": frames_payload,
        "global_segments": {
            str(global_id): [
                {"frame_position": int(frame_pos), "local_id": int(local_id)}
                for frame_pos, local_id in members
            ]
            for global_id, members in global_segments.items()
        },
    }
    (save_dir / "segments.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    preview_count = min(max_preview_frames, len(frame_indices))
    preview_positions = np.unique(
        np.linspace(0, len(frame_indices) - 1, preview_count, dtype=np.int32)
    )
    preview_files = []
    for frame_pos in preview_positions:
        seg_map = all_seg_maps[int(frame_pos)]
        rgb = np.zeros((*seg_map.shape, 3), dtype=np.uint8)
        for local_id in np.unique(seg_map):
            if local_id < 0:
                continue
            global_id = global_lookup.get((int(frame_pos), int(local_id)), int(local_id))
            hue = (global_id * 0.618033988749895) % 1.0
            color = np.asarray(
                colors.hsv_to_rgb([hue, 0.72, 0.95]) * 255.0,
                dtype=np.uint8,
            )
            rgb[seg_map == local_id] = color
        frame_idx = int(frame_indices[int(frame_pos)])
        filename = f"frame_{frame_idx:06d}_clusters.png"
        cv2.imwrite(str(save_dir / filename), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        preview_files.append(filename)

    manifest = {
        "schema_version": 1,
        "segment_maps": "per_frame_segments.npz",
        "segment_summary": "segments.json",
        "preview_files": preview_files,
    }
    (save_dir / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[agentic-evidence] saved per-frame clustering to {save_dir}")


def interval_flow_segmentation_pipeline_with_vis(
    mono_normals: torch.Tensor,  # (T, H, W, 3)
    depthmaps: torch.Tensor,     # (T, H, W)
    pointclouds: torch.Tensor,
    data_dir: Path,
    frame_indices: List[int],
    interval: int = 7,
    device: str = 'cuda',
    save_debug: bool = True,
    stat_cam: bool = False,
    contact_points: Optional[np.ndarray] = None, 
    debug_dir: Path = Path('debug_segments'),
    segment_min_frames: int = 2,
    **kwargs: Any,
) -> Dict[str, List]:
    """
    Complete pipeline with fixed flow loading and correspondence finding.
    """
    num_frames = len(frame_indices)
    H, W = depthmaps.shape[1:3]
    
    print(f"Processing {num_frames} frames with resolution {H}x{W}")
    
    # Step 1: Segment each frame independently
    print("\nStep 1: Segmenting individual frames...")
    all_frame_segments = []
    all_seg_maps = []
    
    for i in range(num_frames):
        seg_map, num_segs = segment_single_frame_normals(
            mono_normals[i],
            depthmaps[i],
            eps_spatial=2.0,
            min_samples=10,
            min_points=10,
            device=device
        )
        
        seg_props = compute_segment_properties(
            seg_map,
            mono_normals[i],
            depthmaps[i],
            pointclouds[i],
            device=device
        )
        
        all_frame_segments.append(seg_props)
        all_seg_maps.append(seg_map)
        print(f"  Frame {frame_indices[i]}: {len(seg_props)} segments")
    
    # Step 2: Load flow data with proper resizing
    print("\nStep 2: Loading and resizing flow data...")
    flow_forward, flow_backward, covis_masks = load_flow_and_covis_data_fixed(
        data_dir, frame_indices, H, W, interval
    )
    
    # Optional: visualize flow
    if save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
        # You can uncomment this if you want flow visualization
        # batch_visualize_flow_data(flow_forward, flow_backward, covis_masks, debug_dir / 'flow_vis')
    
    # Step 3: Find correspondences with improved method
    print("\nStep 3: Finding greedy segment correspondences across ALL frames …")
    correspondence_pairs = []            # (fi, seg_i, fj, seg_j)

    # map real frame-number → position index 0…num_frames-1
    frame_to_pos = {f: i for i, f in enumerate(frame_indices)}

    for (fi, fj), flow in flow_forward.items():
        if (fi, fj) not in covis_masks:          # need covis mask too
            continue
        i_pos, j_pos = frame_to_pos[fi], frame_to_pos[fj]

        corr = find_segment_correspondences_improved(
            all_seg_maps[i_pos], all_seg_maps[j_pos],
            all_frame_segments[i_pos], all_frame_segments[j_pos],
            flow, covis_masks[(fi, fj)],
            normal_threshold=0.85,
            min_overlap_pixels=50,
            overlap_ratio_threshold=0.1
        )
        print(f"  {fi}->{fj}: {len(corr)} matches")
        for seg_i, seg_j in corr:
            correspondence_pairs.append((i_pos, seg_i, j_pos, seg_j))

    print(f"\nStep 4: Union-fusing {len(correspondence_pairs)} correspondences …")
    global_segments = build_global_segments_greedy(
        all_frame_segments, correspondence_pairs
    )
    print(f"  ↳ {len(global_segments)} global segments after greedy fusion")
    print(f"  Found {len(global_segments)} global segments")

    cluster_dump_dir = kwargs.get("cluster_dump_dir")
    if cluster_dump_dir is not None:
        _save_agentic_per_frame_evidence(
            Path(cluster_dump_dir),
            all_seg_maps,
            all_frame_segments,
            global_segments,
            frame_indices,
        )
    
    # Save debug visualizations if requested
    if save_debug:
        print("\nSaving debug visualizations...")
        visualize_per_frame_segments(all_seg_maps, frame_indices, debug_dir)
        visualize_merged_segments(all_seg_maps, global_segments, frame_indices, debug_dir)
        save_segment_statistics(
            all_seg_maps,
            all_frame_segments,
            global_segments,
            correspondence_pairs,
            frame_indices,
            debug_dir
        )
    
    # Step 5: Process global segments into primitives
    print("\nStep 5: Fitting primitives to global segments...")
    results = process_global_segments(
        global_segments,
        all_frame_segments,
        min_frames=segment_min_frames,
        device=device
    )

    results_extra = {}
    if contact_points is not None and len(contact_points) > 0:
        results_extra = process_global_points(
            torch.from_numpy(contact_points),
            min_frames=2,
            device=device,
            scene_primitives=results,
        )

    results = merge_primitives_dicts(results, results_extra)


    # process_global_points
    
    print(f"\n✓ Generated {len(results['S_items'])} primitives from {len(global_segments)} global segments")
    
    return results
