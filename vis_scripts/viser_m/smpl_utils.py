"""
SMPL-related utility functions extracted from the main script.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import copy
from scipy.ndimage import gaussian_filter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from smpl import SMPL, BodyModelSMPLH, BodyModelSMPLX
import trimesh
from path_config import (
    BODY_SEGMENTS_DIR,
    CONTACT_IDS_PATH,
    TRAM_RESULTS_ROOT,
    HMR_RESULTS_ROOT,
    GVHMR_ROOT,
    GVHMR_BODY_MODELS_DIR,
    GVHMR_UTILS_DIR,
)
def axis_angle_to_matrix(angle_axis: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle representation to rotation matrix.
    
    Args:
        angle_axis: Tensor of shape [N, 3] containing axis-angle vectors
        
    Returns:
        Rotation matrices of shape [N, 3, 3]
    """
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


def axis_angle_to_matrix_batch(rotvecs: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle to rotation matrix for batched joint rotations.
    
    Args:
        rotvecs: Tensor of shape [T, 21, 3] containing axis-angle vectors
        
    Returns:
        Rotation matrices of shape [T, 21, 3, 3]
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


def matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to axis-angle representation.
    
    Args:
        R: Rotation matrices of shape [N, 3, 3]
        
    Returns:
        Axis-angle vectors of shape [N, 3]
    """
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


def load_contact_body_parts() -> List[np.ndarray]:
    """
    Load contact vertex IDs for specific body parts.
    
    Returns:
        List of numpy arrays containing vertex IDs for each contact body part
    """
    body_segments_dir = str(BODY_SEGMENTS_DIR)
    contact_verts_ids = []
    contact_body_parts = ['L_Leg', 'R_Leg', 'L_Hand', 'R_Hand', 'gluteus', 'back', 'thighs']
    
    for part in contact_body_parts:
        with open(os.path.join(body_segments_dir, part + '.json'), 'r') as f:
            data = json.load(f)
            contact_verts_ids.append(list(set(data["verts_ind"])))
    
    return contact_verts_ids


import os
import json
import numpy as np
import torch
from typing import List

def load_contact_body_parts() -> List[np.ndarray]:
    """
    Load contact vertex IDs for specific body parts.
    Returns:
        List of numpy arrays containing vertex IDs for each contact body part
    """
    body_segments_dir = str(BODY_SEGMENTS_DIR)
    contact_verts_ids = []
    contact_body_parts = ['L_Leg', 'R_Leg', 'L_Hand', 'R_Hand', 'gluteus', 'back', 'thighs']
    
    for part in contact_body_parts:
        with open(os.path.join(body_segments_dir, part + '.json'), 'r') as f:
            data = json.load(f)
            contact_verts_ids.append(np.array(list(set(data["verts_ind"]))))
    
    return contact_verts_ids

def get_contact_ids() -> np.ndarray:
    """
    Get all contact vertex IDs as a single numpy array.
    Returns:
        Numpy array of contact vertex IDs
    """
    contact_verts_ids = load_contact_body_parts()
    return np.concatenate(contact_verts_ids)

def load_contact_ids(device: str = "cuda") -> torch.Tensor:
    """
    Load contact vertex IDs as a torch tensor.
    Args:
        device: Device to load the tensor on
    Returns:
        Torch tensor of contact vertex IDs
    """
    ids_np = get_contact_ids()
    ids = torch.as_tensor(ids_np, dtype=torch.long, device=device)
    return ids

# Single function with mode parameter
def load_contact_ids_with_mode(mode: str = "all", device: str = "cuda"):
    """
    Load contact vertex IDs with different modes.
    Args:
        mode: Selection mode
            - "all": Return all contact IDs as single torch tensor
            - "grouped": Return list of torch tensors for [leg, hand, gluteus, back, thighs] (5 groups)
            - "specific": Return list of torch tensors for ['L_Leg', 'R_Leg', 'L_Hand', 'R_Hand', 'gluteus', 'back', 'thighs'] (7 groups)
        device: Device to load the tensor on
    Returns:
        - mode "all": torch.Tensor of all contact vertex IDs
        - mode "grouped": List[torch.Tensor] with 5 tensors [leg, hand, gluteus, back, thighs]
        - mode "specific": List[torch.Tensor] with 7 tensors [L_Leg, R_Leg, L_Hand, R_Hand, gluteus, back, thighs]
    """
    body_segments_dir = str(BODY_SEGMENTS_DIR)
    
    if mode == "all":
        # Return all contact IDs as single tensor (same as original load_contact_ids)
        contact_verts_ids = []
        contact_body_parts = ['L_Leg', 'R_Leg', 'L_Hand', 'R_Hand', 'gluteus', 'back', 'thighs']
        max_vertex_idx = 6889  # SMPL has 6890 vertices (0-indexed)
        
        for part in contact_body_parts:
            with open(os.path.join(body_segments_dir, part + '.json'), 'r') as f:
                data = json.load(f)
                contact_verts_ids.extend(data["verts_ind"])
        
        # Filter out invalid indices and remove duplicates
        valid_ids = [idx for idx in set(contact_verts_ids) if idx <= max_vertex_idx]
        ids = torch.as_tensor(valid_ids, dtype=torch.long, device=device)
        return ids
    
    elif mode == "grouped":
        # Return list of tensors: [leg, hand, gluteus, back, thighs]
        contact_tensors = []
        # max_vertex_idx = 6889  # SMPL has 6890 vertices (0-indexed)
        
        # Group L_Leg and R_Leg into 'leg'
        leg_ids = []
        for leg_part in ['L_Leg', 'R_Leg']:
            with open(os.path.join(body_segments_dir, leg_part + '.json'), 'r') as f:
                data = json.load(f)
                leg_ids.extend(data["verts_ind"])
        # Filter out invalid indices and remove duplicates
        leg_ids = [idx for idx in set(leg_ids)]
        leg_tensor = torch.as_tensor(leg_ids, dtype=torch.long, device=device)
        contact_tensors.append(leg_tensor)
        
        # Group L_Hand and R_Hand into 'hand'
        hand_ids = []
        for hand_part in ['L_Hand', 'R_Hand']:
            with open(os.path.join(body_segments_dir, hand_part + '.json'), 'r') as f:
                data = json.load(f)
                hand_ids.extend(data["verts_ind"])
        # Filter out invalid indices and remove duplicates
        hand_ids = [idx for idx in set(hand_ids)]
        hand_tensor = torch.as_tensor(hand_ids, dtype=torch.long, device=device)
        contact_tensors.append(hand_tensor)
        
        # Add gluteus, back, thighs as separate tensors
        for part in ['gluteus', 'back', 'thighs']:
            with open(os.path.join(body_segments_dir, part + '.json'), 'r') as f:
                data = json.load(f)
                # Filter out invalid indices and remove duplicates
                part_ids = [idx for idx in set(data["verts_ind"])] #if idx <= max_vertex_idx]
                part_tensor = torch.as_tensor(part_ids, dtype=torch.long, device=device)
                contact_tensors.append(part_tensor)
        
        return contact_tensors  # List of 5 tensors
    
    elif mode == "specific":
        # Return list of tensors for each specific part
        contact_tensors = []
        specific_body_parts = ['L_Leg', 'R_Leg', 'L_Hand', 'R_Hand', 'gluteus', 'back', 'thighs']
        max_vertex_idx = 6889  # SMPL has 6890 vertices (0-indexed)
        
        for part in specific_body_parts:
            with open(os.path.join(body_segments_dir, part + '.json'), 'r') as f:
                data = json.load(f)
                # Filter out invalid indices and remove duplicates
                part_ids = [idx for idx in set(data["verts_ind"]) if idx <= max_vertex_idx]
                part_tensor = torch.as_tensor(part_ids, dtype=torch.long, device=device)
                contact_tensors.append(part_tensor)
        
        return contact_tensors  # List of 7 tensors
    
    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose from: 'all', 'grouped', 'specific'")

def load_contact_ids_from_file(filepath: str = str(CONTACT_IDS_PATH)) -> torch.Tensor:
    """
    Load pre-computed contact IDs from a file.
    
    Args:
        filepath: Path to the saved contact IDs file
        
    Returns:
        Torch tensor of contact vertex IDs
    """
    return torch.load(filepath)


def transform_smpl_to_world(
    global_orient_cam: torch.Tensor,
    transl_cam: torch.Tensor,
    world_cam_R: torch.Tensor,
    world_cam_T: torch.Tensor,
    body_pose: torch.Tensor,
    pred_shapes: torch.Tensor,
    smpl_model,
    use_axis_angle: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Transform SMPL parameters from camera space to world space.
    
    Args:
        global_orient_cam: Global orientation in camera space [N, 1, 3, 3] or [N, 3]
        transl_cam: Translation in camera space [N, 3] or [N, 1, 3]
        world_cam_R: Camera rotation matrices [N, 3, 3]
        world_cam_T: Camera translation vectors [N, 3]
        body_pose: Body pose parameters [N, 23, 3, 3] or [N, 69]
        pred_shapes: Shape parameters [N, 10]
        smpl_model: SMPL model instance
        use_axis_angle: Whether input is in axis-angle format
        
    Returns:
        Dictionary containing:
            - global_orient_world: Global orientation in world space
            - transl_world: Translation in world space
            - vertices: Predicted vertices in world space
            - joints: Predicted 3D joints in world space
    """
    device = global_orient_cam.device
    
    # Ensure transl_cam has shape [N, 3]
    if transl_cam.dim() == 3:
        transl_cam = transl_cam.squeeze(1)
    
    # Convert global_orient to matrix if needed
    if use_axis_angle and global_orient_cam.shape[-1] == 3:
        global_orient_mat = axis_angle_to_matrix(global_orient_cam.reshape(-1, 3))
        global_orient_mat = global_orient_mat.reshape(global_orient_cam.shape[0], 1, 3, 3)
    else:
        global_orient_mat = global_orient_cam
    
    # Transform global orientation to world space
    global_orient_world = torch.einsum('bij, bnjk -> bnik', world_cam_R, global_orient_mat)
    
    # Get vertices in camera space
    pred_cam = smpl_model(
        body_pose=body_pose,
        global_orient=global_orient_mat,
        betas=pred_shapes,
        transl=transl_cam,
        pose2rot=False,
        default_smpl=True
    )
    pred_vert_cam = pred_cam.vertices
    
    # Transform vertices to world space
    pred_vert_world = torch.einsum('bij,bnj->bni', world_cam_R, pred_vert_cam) + world_cam_T[:, None]
    
    # Initial world translation
    transl_world = torch.einsum('bij, bj->bi', world_cam_R, transl_cam) + world_cam_T
    
    # Compute temporary vertices with world orientation
    pred_temp = smpl_model(
        body_pose=body_pose,
        global_orient=global_orient_world,
        betas=pred_shapes,
        transl=transl_world,
        pose2rot=False,
        default_smpl=True
    )
    pred_vert_temp = pred_temp.vertices
    
    # Compute translation offset to align vertices
    transl_offset = pred_vert_world - pred_vert_temp
    transl_world = transl_offset[:, 0, :] + transl_world
    
    # Final SMPL forward pass with corrected parameters
    pred_final = smpl_model(
        body_pose=body_pose,
        global_orient=global_orient_world,
        betas=pred_shapes,
        transl=transl_world,
        pose2rot=False,
        default_smpl=True
    )
    
    
    return {
        'global_orient_world': global_orient_world,
        'transl_world': transl_world,
        'vertices': pred_final.vertices,
        'joints': pred_final.joints[:, :24] if hasattr(pred_final, 'joints') else None
    }


def prepare_smpl_results_dict(
    world_cam_R: torch.Tensor,
    world_cam_T: torch.Tensor,
    body_pose: torch.Tensor,
    global_orient: torch.Tensor,
    betas: torch.Tensor,
    transl: torch.Tensor
) -> Dict[str, Union[List, torch.Tensor, bool]]:
    """
    Prepare SMPL results dictionary for saving.
    
    Args:
        world_cam_R: Camera rotation matrices [B, 3, 3]
        world_cam_T: Camera translation vectors [B, 3]
        body_pose: Body pose parameters [B, 23, 3, 3] or [B, 69]
        global_orient: Global orientation [B, 1, 3, 3] or [B, 3]
        betas: Shape parameters [B, 10]
        transl: Translation vectors [B, 3]
        
    Returns:
        Dictionary with SMPL parameters ready for saving
    """
    return {
        'pred_cam': [world_cam_R, world_cam_T],  # cam
        'body_pose': body_pose,  # smpl
        'global_orient': global_orient,  # smpl
        'betas': betas,  # smpl
        'transl': transl,  # smpl
        'pose2rot': False,
        'default_smpl': True
    }


def convert_smplx_to_smpl(
    smplx_vertices: torch.Tensor,
    smplx2smpl_map: torch.Tensor
) -> torch.Tensor:
    """
    Convert SMPL-X vertices to SMPL vertices using a sparse mapping.
    
    Args:
        smplx_vertices: SMPL-X vertices [N, V_smplx, 3]
        smplx2smpl_map: Sparse mapping matrix
        
    Returns:
        SMPL vertices [N, V_smpl, 3]
    """
    return torch.stack([torch.matmul(smplx2smpl_map, v_) for v_ in smplx_vertices])


def filter_vertices_by_contact(
    vertices: Union[np.ndarray, torch.Tensor],
    contact_ids: Union[np.ndarray, torch.Tensor],
    device: str = "cuda"
) -> torch.Tensor:
    """
    Extract contact vertices from full vertex set.
    
    Args:
        vertices: Full vertex set [N, V, 3]
        contact_ids: Indices of contact vertices
        device: Device to perform computation on
        
    Returns:
        Contact vertices [N, C, 3] where C is the number of contact vertices
    """
    if isinstance(vertices, np.ndarray):
        vertices = torch.from_numpy(vertices).to(device)
    if isinstance(contact_ids, np.ndarray):
        contact_ids = torch.from_numpy(contact_ids).long().to(device)
    
    return vertices[:, contact_ids, :]


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

def traj_filter(pred_vert_w, pred_j3d_w, sigma=3):
    """ Smooth the root trajetory (xyz) """
    from scipy.ndimage import gaussian_filter
    root = pred_j3d_w[:, 0]
    root_smooth = torch.from_numpy(gaussian_filter(root, sigma=sigma, axes=0))

    pred_vert_w = pred_vert_w + (root_smooth - root)[:, None]
    pred_j3d_w = pred_j3d_w + (root_smooth - root)[:, None]
    return pred_vert_w, pred_j3d_w


def process_tram_smpl(
    tgt_name: str,
    world_cam_R: torch.Tensor,
    world_cam_T: torch.Tensor,
    max_frames: int,
    smpl_model,
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """
    Process SMPL data from TRAM format.
    
    Args:
        tgt_name: Target name for loading files
        world_cam_R: Camera rotation matrices
        world_cam_T: Camera translation vectors
        max_frames: Maximum number of frames to process
        smpl_model: SMPL model instance
        device: Device to run computations on
        
    Returns:
        Dictionary containing processed SMPL data
    """
    num_frames = min(max_frames, len(world_cam_R))
    results_root = TRAM_RESULTS_ROOT
    
    cands = sorted(
        p for p in results_root.iterdir()
        if p.is_dir() and p.name.startswith(tgt_name)
    )
    scene_dir = cands[0]

    hsfm_pkl = scene_dir / "hps" / "hps_track_0.npy"
    pred_smpl = np.load(hsfm_pkl, allow_pickle=True).item()
    # pred_cams = pred_smpl['pred_cam'].to(device)
    pred_poses = pred_smpl['pred_pose'].to(device)
    pred_shapes = pred_smpl['pred_shape'].to(device)
    pred_rotmats = pred_smpl['pred_rotmat'].to(device)
    pred_transs = pred_smpl['pred_trans'].to(device)
    
    global_orient_cam = pred_rotmats[:, 0:1, :, :]
    body_pose = pred_rotmats[:, 1:, :, :]
    transl_cam = pred_transs
    
    # Transform to world space
    smpl_world_results = transform_smpl_to_world(
        global_orient_cam=global_orient_cam,
        transl_cam=transl_cam,
        world_cam_R=world_cam_R,
        world_cam_T=world_cam_T,
        body_pose=body_pose,
        pred_shapes=pred_shapes,
        smpl_model=smpl_model,
        use_axis_angle=False
    )

    smpl_world_results['vertices'], smpl_world_results['joints'] = traj_filter(smpl_world_results['vertices'].cpu(), smpl_world_results['joints'].cpu())


    return {
        'num_frames': num_frames,
        'global_orient_world': smpl_world_results['global_orient_world'],
        'transl_world': smpl_world_results['transl_world'],
        'pred_vert': smpl_world_results['vertices'],
        'pred_j3dg': smpl_world_results['joints'],
        'body_pose': body_pose,
        'pred_shapes': pred_shapes,
        'faces': smpl_model.faces
    }


def process_gv_smpl(
    tgt_name: str,
    world_cam_R: torch.Tensor,
    world_cam_T: torch.Tensor,
    max_frames: int,
    smpl_model = None, # legacy. 
    use_world: bool = True,
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """
    Process SMPL data from GV (GVHMR) format.
    
    Args:
        tgt_name: Target name for loading files
        world_cam_R: Camera rotation matrices
        world_cam_T: Camera translation vectors
        max_frames: Maximum number of frames to process
        smpl_model: SMPL model instance (not used directly, but kept for consistency)
        use_world: Whether to transform to world coordinates
        device: Device to run computations on
        
    Returns:
        Dictionary containing processed SMPL data
    """
    from smpl import BodyModelSMPLX, BodyModelSMPLH
    
    hmr4d_path = HMR_RESULTS_ROOT / tgt_name / "hmr4d_results.pt"
    pred = torch.load(hmr4d_path)
    
    PROJ_ROOT = GVHMR_ROOT
    smplx2smpl = torch.load(GVHMR_UTILS_DIR / "body_model" / "smplx2smpl_sparse.pt").to(device)
    
    # Initialize SMPL-X model
    bm_kwargs = {
        "model_type": "smplx",
        "gender": "neutral",
        "num_pca_comps": 12,
        "flat_hand_mean": False,
    }
    modelggg = BodyModelSMPLX(model_path=str(GVHMR_BODY_MODELS_DIR), **bm_kwargs)
    modelggg = modelggg.to(device)
    
    # Load SMPL-X to SMPL mapping
    smplx2smpl = torch.load(GVHMR_UTILS_DIR / "body_model" / "smplx2smpl_sparse.pt").to(device)
    
    # Initialize SMPL model for faces
    bm_kwargs_smpl = {
        "model_path": str(GVHMR_BODY_MODELS_DIR),
        "model_type": "smpl",
        "gender": "neutral",
        "num_betas": 10,
        "create_body_pose": False,
        "create_betas": False,
        "create_global_orient": False,
        "create_transl": False,
    }
    model_ = BodyModelSMPLH(**bm_kwargs_smpl)
    faces = model_.faces
    
    num_frames = min(max_frames, len(world_cam_R))
    
    # Process SMPL-X output
    smplx_out = modelggg(**to_cuda(pred["smpl_params_incam"]))
    pred_c_verts = convert_smplx_to_smpl(smplx_out.vertices, smplx2smpl)
    
    pred_shapes = smplx_out.betas[:, :10]
    global_orient_cam = smplx_out.global_orient
    
    # Process body pose
    rotmats_x = smplx_out.body_pose
    rotvecs = rotmats_x.view(-1, 21, 3)
    rotmats = axis_angle_to_matrix_batch(rotvecs)
    T = rotmats.shape[0]
    identity_rot = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).repeat(T, 2, 1, 1)
    rotmats_smpl = torch.cat([rotmats, identity_rot], dim=1)
    body_pose = rotmats_smpl
    
    transl_cam = pred["smpl_params_incam"]['transl']
    transl_cam = transl_cam.unsqueeze(1).to(device)
    
    if use_world:
        # Convert global orient to world space
        global_orient_world = axis_angle_to_matrix(global_orient_cam)
        global_orient_world = torch.matmul(world_cam_R, global_orient_world)
        global_orient_world = matrix_to_axis_angle(global_orient_world)
        
        # Transform vertices and translation to world space
        transl_world = torch.einsum('bij, bnj->bni', world_cam_R, transl_cam) + world_cam_T.unsqueeze(1)
        pred_vert = torch.einsum('bij,bnj->bni', world_cam_R, pred_c_verts) + world_cam_T[:, None]
        
        # Adjust translation to align vertices properly
        pred_smpl_world = copy.deepcopy(pred["smpl_params_incam"])
        pred_smpl_world['transl'] = transl_world.squeeze(1)
        pred_smpl_world['global_orient'] = global_orient_world
        
        smplx_out_temp = modelggg(**to_cuda(pred_smpl_world))
        pred_verts_temp = convert_smplx_to_smpl(smplx_out_temp.vertices, smplx2smpl)
        
        transl_offset = pred_vert - pred_verts_temp
        transl_world = transl_world + transl_offset[:, 0:1, :]
        
        # Verify alignment
        pred_smpl_world['transl'] = transl_world.squeeze(1)
        smplx_out_temp = modelggg(**to_cuda(pred_smpl_world))
        pred_verts_temp = convert_smplx_to_smpl(smplx_out_temp.vertices, smplx2smpl)
        transl_offset = pred_vert - pred_verts_temp
        print(f"Alignment error: {torch.norm(transl_offset, dim=1).mean():.6f}")
       # assert torch.norm(transl_offset, dim=1).mean() < 2e-2
        
        transl_world = transl_world.squeeze(1)
        global_orient_world = axis_angle_to_matrix(global_orient_world).unsqueeze(1)
    else:
        transl_world = transl_cam.squeeze(1)
        pred_vert = pred_c_verts
        global_orient_world = global_orient_cam


    return {
        'num_frames': num_frames,
        'global_orient_world': global_orient_world,
        'transl_world': transl_world,
        'pred_vert': pred_vert,
        'pred_j3dg': None,  # GV doesn't provide joints directly
        'body_pose': body_pose,
        'pred_shapes': pred_shapes,
        'faces': faces
    }


def vis_hmr(data, save_path, device, every=100):
  smpl = SMPL().to(device)
  pred = smpl(**data)
  pred_vert = pred.vertices.cpu().numpy()
  for i in range(0, len(pred_vert), every):
      mesh = trimesh.Trimesh(vertices=pred_vert[i], faces=smpl.faces, process=False)
      mesh.export(os.path.join(save_path, f"human_beforerot_{i:04d}.obj"))


import numpy as np
from scipy.signal import savgol_filter

def detect_static_frames(pred_vert, vel_threshold=0.01, acc_threshold=0.005, 
                         window_size=5, use_smoothing=True):
    """
    检测静止帧
    
    Args:
        pred_vert: SMPL顶点 [T, V, 3] - T帧, V个顶点, 3维坐标
        vel_threshold: 速度阈值，低于此值认为静止
        acc_threshold: 加速度阈值，低于此值认为静止
        window_size: 用于判断静止的窗口大小
        use_smoothing: 是否对速度进行平滑
    
    Returns:
        static_frames: 布尔数组 [T]，True表示静止帧
        velocities: 速度数组 [T-1]
        accelerations: 加速度数组 [T-2]
    """
    T, V, _ = pred_vert.shape
    
    # 计算质心或关键点的运动
    # 方法1: 使用所有顶点的质心
    center_points = np.mean(pred_vert, axis=1)  # [T, 3]
    
    # 方法2: 或者只使用某些关键顶点（比如躯干）
    # key_vertex_indices = [411, 412, 3021, 3022]  # SMPL躯干顶点示例
    # center_points = np.mean(pred_vert[:, key_vertex_indices], axis=1)
    
    # 计算速度 (帧间差分)
    velocities = np.diff(center_points, axis=0)  # [T-1, 3]
    vel_magnitude = np.linalg.norm(velocities, axis=1)  # [T-1]
    
    # 可选：平滑速度曲线
    if use_smoothing and len(vel_magnitude) > 5:
        vel_magnitude = savgol_filter(vel_magnitude, 
                                     window_length=min(5, len(vel_magnitude)), 
                                     polyorder=2)
    
    # 计算加速度
    accelerations = np.diff(velocities, axis=0)  # [T-2, 3]
    acc_magnitude = np.linalg.norm(accelerations, axis=1)  # [T-2]
    
    # 判断静止帧
    static_frames = np.zeros(T, dtype=bool)
    
    # 基于速度判断
    vel_static = np.zeros(T, dtype=bool)
    vel_static[0] = vel_magnitude[0] < vel_threshold  # 第一帧
    vel_static[1:] = vel_magnitude < vel_threshold
    
    # 基于加速度判断（补充）
    acc_static = np.zeros(T, dtype=bool)
    acc_static[0] = True  # 假设第一帧
    acc_static[1] = acc_magnitude[0] < acc_threshold
    acc_static[2:] = acc_magnitude < acc_threshold
    
    # 综合判断：速度和加速度都很小
    static_frames = vel_static & acc_static
    
    # 使用滑动窗口进行平滑判断
    if window_size > 1:
        static_smooth = np.zeros(T, dtype=bool)
        half_window = window_size // 2
        
        for i in range(T):
            start = max(0, i - half_window)
            end = min(T, i + half_window + 1)
            # 如果窗口内大部分帧是静止的，则认为当前帧静止
            static_smooth[i] = np.mean(static_frames[start:end]) > 0.7
        
        static_frames = static_smooth
    
    return static_frames, vel_magnitude, acc_magnitude


def detect_static_segments(pred_vert, min_static_duration=10, **kwargs):
    """
    检测连续的静止片段
    
    Args:
        pred_vert: SMPL顶点 [T, V, 3]
        min_static_duration: 最小静止持续帧数
        **kwargs: 传递给detect_static_frames的参数
    
    Returns:
        segments: 静止片段列表 [(start_frame, end_frame), ...]
    """
    static_frames, _, _ = detect_static_frames(pred_vert, **kwargs)
    
    segments = []
    start = None
    
    for i, is_static in enumerate(static_frames):
        if is_static and start is None:
            start = i
        elif not is_static and start is not None:
            if i - start >= min_static_duration:
                segments.append((start, i))
            start = None
    
    # 处理最后一个片段
    if start is not None and len(static_frames) - start >= min_static_duration:
        segments.append((start, len(static_frames)))
    
    return segments

def analyze_contacts_per_body_part(
    pred_contact_vert_list,            # List[np.ndarray], each [T, Li, 3] (leg, hand, gluteus, back, thigh) in that order
    interact_contact_path: str,        # folder with {00000.npz, ...} each containing 'pred_contact_3d_smplh' (per-vertex scores)
    num_frames: int,
    body_part_params: dict,            # e.g. keys: 'leg','hand','gluteus','back','thigh' -> dict(threshold, min_consecutive_frames, weight)
    total_verts: int = 6890,
    fallback: str = "start",           # for empty-contact segments: 'start' | 'middle' | 'end' | 'none'
    part_ids_list: list | None = None, # optional: precomputed vertex-index lists per part; if None we auto-map via KD-tree
    pred_vert_global: np.ndarray | None = None,  # optional: full SMPL verts [T, 6890, 3] to compute static segments
    return_per_part: bool = True       # NEW: return a dict of per-part results as 6th output
):
    """
    Returns (unchanged first five):
      contacted_masks : [T, 6890] bool   — final (OR across parts) stable contact mask
      static_frames   : [T] bool         — from analyze_motion(pred_vert) if available; else all False and segments=[(0,T)]
      static_segments : List[(s, e)]     — end is exclusive
      best_frames     : List[int]        — 1 per segment; chosen by weighted contact counts
      counts          : [T] float        — weighted contact counts per frame (useful for debugging)

    NEW (6th, only if return_per_part=True):
      per_part: Dict[str, Dict] with keys for each part:
          {
            'mask':   np.ndarray [T, V] bool (stable mask for this part),
            'counts': np.ndarray [T] int    (per-frame contact count in this part),
            'best_frames': List[int|None]   (best frame per static segment),
            'best_vertices': List[np.ndarray|None]  (vertex ids at the best frame)
          }
    """
    import os
    import numpy as np
    from scipy.spatial import cKDTree

    T = int(num_frames)
    if T <= 0:
        raise ValueError("num_frames must be > 0")

    # 1) Static frames/segments
    pred_vert = pred_vert_global
    if pred_vert is None:
        pred_vert = globals().get('pred_vert', None)
    if pred_vert is not None:
        static_frames, static_segments = analyze_motion(pred_vert, visualize=False)
    else:
        static_frames = np.zeros(T, dtype=bool)
        static_segments = [(0, T)]

    # 2) Part names
    part_names_default = ['leg', 'hand', 'gluteus', 'back', 'thigh']
    K = len(pred_contact_vert_list)
    part_names = part_names_default[:K]

    # 3) Map part subset verts -> full mesh vertex ids (once using frame 0)
    if part_ids_list is None:
        if pred_vert is None:
            raise ValueError("Need pred_vert (full [T,6890,3]) to infer part indices when part_ids_list=None.")
        full0 = np.asarray(pred_vert[0])  # [V,3]
        tree0 = cKDTree(full0)
        part_ids_list = []
        for k in range(K):
            sub0 = np.asarray(pred_contact_vert_list[k][0])  # [Lk,3]
            _, nn_idx = tree0.query(sub0, k=1)
            part_ids_list.append(np.unique(nn_idx.astype(np.int64)))
    assert len(part_ids_list) == K, "part_ids_list length must match pred_contact_vert_list length"

    # 4) Load per-frame contact scores and threshold per part -> raw per-part masks
    per_part_masks = [np.zeros((T, total_verts), dtype=bool) for _ in range(K)]
    for t in range(T):
        npz_path = os.path.join(interact_contact_path, f"{t:05d}.npz")
        if not os.path.exists(npz_path):
            continue
        arr = np.load(npz_path, allow_pickle=True)
        scores = arr['pred_contact_3d_smplh']
        scores = np.squeeze(scores)
        if scores.ndim != 1 or scores.shape[0] < total_verts:
            raise ValueError(f"Unexpected contact array shape at {npz_path}: {scores.shape}")

        for k, pname in enumerate(part_names):
            # Skip parts not configured
            if pname not in body_part_params:
                continue
            thr = float(body_part_params[pname]['contact_threshold'])
            ids = part_ids_list[k]
            m = np.zeros(total_verts, dtype=bool)
            m[ids] = scores[ids] > thr
            per_part_masks[k][t] = m

    # 5) Temporal stability per part
    per_part_masks_stable = []
    for k, pname in enumerate(part_names):
        min_run = int(body_part_params.get(pname, {}).get('min_consecutive_frames', 1))
        if min_run <= 1:
            per_part_masks_stable.append(per_part_masks[k])
        else:
            per_part_masks_stable.append(
                filter_stable_contacts_simple(per_part_masks[k], min_consecutive_frames=min_run)
            )

    # 6) Combined boolean OR (for your downstream)
    contacted_masks = np.any(np.stack(per_part_masks_stable, axis=0), axis=0)  # [T,V]

    # 7) Weighted per-frame counts (your original "global" scoring)
    counts = np.zeros(T, dtype=float)
    per_part_counts = []
    for k, pname in enumerate(part_names):
        c_k = per_part_masks_stable[k].sum(axis=1)   # [T], integer counts in this part
        per_part_counts.append(c_k)
        w = float(body_part_params.get(pname, {}).get('weight', 1.0))
        counts += w * c_k

    # 8) Global best frame per static segment using the weighted counts (unchanged behavior)
    best_frames = []
    for (s, e) in static_segments:
        s = int(max(0, s)); e = int(min(T, e))
        if s >= e:
            best_frames.append(None); continue
        seg_counts = counts[s:e]
        if seg_counts.size == 0:
            best_frames.append(None); continue
        if seg_counts.max() > 0:
            best_frames.append(s + int(np.argmax(seg_counts)))
        else:
            if   fallback == "start":  best_frames.append(s)
            elif fallback == "middle": best_frames.append(s + (e - s) // 2)
            elif fallback == "end":    best_frames.append(e - 1)
            else:                      best_frames.append(None)

    # 9) NEW: per-part best frames per static segment (plus vertices at that frame)
    per_part_detail = {}
    if return_per_part:
        for k, pname in enumerate(part_names):
            bf, bverts, _ = pick_best_frames_per_segment(
                contacted_masks=per_part_masks_stable[k],
                static_segments=static_segments,
                return_vertices=True,
                fallback=fallback
            )
            per_part_detail[pname] = {
                'mask': per_part_masks_stable[k],
                'counts': per_part_counts[k],
                'best_frames': bf,
                'best_vertices': bverts,
            }

    if return_per_part:
        return contacted_masks, static_frames, static_segments, best_frames, counts, per_part_detail
    else:
        return contacted_masks, static_frames, static_segments, best_frames, counts




def pick_best_frames_with_weights(
    weighted_scores,
    static_segments,
    return_vertices=False,
    fallback="start"
):
    """
    Modified version of pick_best_frames_per_segment that uses weighted scores.
    
    Args:
        weighted_scores: [T, V] float array with weighted contact scores
        static_segments: List of (start, end) tuples
        return_vertices: Whether to return vertex indices
        fallback: Strategy when no contacts found
        
    Returns:
        best_frame_indices: List of best frame indices
        counts: Weighted sum per frame
    """
    T = weighted_scores.shape[0]
    counts = weighted_scores.sum(axis=1)  # Weighted sum per frame
    
    best_indices = []
    best_vertices_list = [] if return_vertices else None
    
    for (s, e) in static_segments:
        s = int(s)
        e = int(e)
        
        if s < 0 or e > T or s >= e:
            best_indices.append(None)
            if return_vertices:
                best_vertices_list.append(None)
            continue
        
        sub_counts = counts[s:e]
        
        if sub_counts.size == 0:
            best_indices.append(None)
            if return_vertices:
                best_vertices_list.append(None)
            continue
        
        # Find frame with maximum weighted score
        local_argmax = int(np.argmax(sub_counts))
        best_idx = s + local_argmax
        
        # Handle case with no contacts
        if sub_counts.max() == 0:
            if fallback == "start":
                best_idx = s
            elif fallback == "middle":
                best_idx = s + (e - s) // 2
            elif fallback == "end":
                best_idx = e - 1
            elif fallback == "none":
                best_idx = None
        
        best_indices.append(best_idx)
        
        if return_vertices:
            if best_idx is None:
                best_vertices_list.append(None)
            else:
                # Get vertices with non-zero weighted scores
                best_vertices = np.where(weighted_scores[best_idx] > 0)[0]
                best_vertices_list.append(best_vertices)
    
    if return_vertices:
        return best_indices, best_vertices_list, counts
    else:
        return best_indices, counts


def analyze_motion(pred_vert, visualize=False):
    """
    分析运动并可视化结果
    """
    # 检测静止帧
    static_frames, velocities, accelerations = detect_static_frames(
        pred_vert,
        vel_threshold=0.01,  # 根据你的数据调整
        acc_threshold=0.005,
        window_size=5
    )
    
    # 获取静止片段
    static_segments = detect_static_segments(
        pred_vert,
        min_static_duration=15,
        vel_threshold=0.01
    )
    
    print(f"总帧数: {len(pred_vert)}")
    print(f"静止帧数: {np.sum(static_frames)} ({np.mean(static_frames)*100:.1f}%)")
    print(f"静止片段: {len(static_segments)}")
    for i, (start, end) in enumerate(static_segments):
        print(f"  片段 {i+1}: 帧 {start}-{end} (持续 {end-start} 帧)")
    
    if visualize:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        
        # 速度曲线
        axes[0].plot(velocities, label='Velocity')
        axes[0].axhline(y=0.01, color='r', linestyle='--', label='Threshold')
        axes[0].set_ylabel('Velocity')
        axes[0].legend()
        axes[0].grid(True)
        
        # 加速度曲线
        axes[1].plot(accelerations, label='Acceleration')
        axes[1].axhline(y=0.005, color='r', linestyle='--', label='Threshold')
        axes[1].set_ylabel('Acceleration')
        axes[1].legend()
        axes[1].grid(True)
        
        # 静止帧标记
        axes[2].fill_between(range(len(static_frames)), 
                            static_frames.astype(float),
                            alpha=0.3, label='Static frames')
        axes[2].set_ylabel('Static')
        axes[2].set_xlabel('Frame')
        axes[2].set_ylim(-0.1, 1.1)
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('speed.png')
    
    return static_frames, static_segments


# 实际使用

# 如果需要更精细的控制，可以基于不同身体部位
def detect_body_part_motion(pred_vert, smpl_body_parts=None):
    """
    基于不同身体部位检测运动
    """
    if smpl_body_parts is None:
        # SMPL身体部位顶点索引（示例）
        smpl_body_parts = {
            'head': list(range(0, 100)),
            'torso': list(range(400, 600)),
            'left_arm': list(range(1000, 1200)),
            'right_arm': list(range(1500, 1700)),
            'legs': list(range(2000, 2500))
        }
    
    part_motions = {}
    for part_name, indices in smpl_body_parts.items():
        part_verts = pred_vert[:, indices]
        static_frames, _, _ = detect_static_frames(part_verts)
        part_motions[part_name] = static_frames
    
    return part_motions

def get_stable_contacts(contact_sequence, min_consecutive_frames=15):
    """
    找出在连续min_consecutive_frames帧中都出现的contact点
    
    Args:
        contact_sequence: shape [num_frames, 6890] 的布尔数组
        min_consecutive_frames: 最少需要连续出现的帧数
    
    Returns:
        stable_mask: shape [6890,] 的布尔数组，表示哪些顶点是稳定的contact点
    """
    num_frames, num_verts = contact_sequence.shape
    
    if num_frames < min_consecutive_frames:
        return np.zeros(num_verts, dtype=bool)
    
    # 使用滑动窗口检查连续性
    stable_mask = np.zeros(num_verts, dtype=bool)
    
    for i in range(num_frames - min_consecutive_frames + 1):
        window = contact_sequence[i:i + min_consecutive_frames]
        # 检查哪些顶点在这个窗口内的所有帧都是contact
        window_stable = np.all(window, axis=0)
        # 更新stable_mask
        stable_mask |= window_stable
    
    return stable_mask

def filter_stable_contacts_simple(contact_masks, min_consecutive_frames=60):
    """
    简化版本：使用滑动窗口，标记所有在任意连续窗口中都出现的帧
    
    Args:
        contact_masks: shape [T, 6890] 的布尔数组
        min_consecutive_frames: 最少需要连续出现的帧数
    
    Returns:
        contacted_filtered_masks: shape [T, 6890] 的布尔数组
    """
    print(contact_masks.shape)

    T, num_verts = contact_masks.shape
    contacted_filtered_masks = np.zeros_like(contact_masks, dtype=bool)
    
    if T < min_consecutive_frames:
        return contacted_filtered_masks
    
    # 对每个可能的窗口检查
    for i in range(T - min_consecutive_frames + 1):
        window = contact_masks[i:i + min_consecutive_frames]
        # 在这个窗口内所有帧都是contact的顶点
        window_stable = np.all(window, axis=0)  # shape [6890,]
        
        # 将这个窗口内的所有帧标记为稳定（如果该顶点在窗口内稳定）
        for j in range(min_consecutive_frames):
            contacted_filtered_masks[i + j] |= window_stable
    
    # 最终mask：既要是原始检测到的，也要是稳定的
    contacted_filtered_masks &= contact_masks
    
    return contacted_filtered_masks

def inspect_contact_points(contacted_masks, vertex_indices=None, top_k=10):
    """
    快速检查valid contact points的统计信息
    
    Args:
        contacted_masks: shape [T, 6890] 的布尔数组，过滤后的contact masks
        vertex_indices: 可选，指定要检查的顶点索引列表
        top_k: 显示出现最频繁的前k个顶点
    
    Returns:
        stats: 包含统计信息的字典
    """
    T, num_verts = contacted_masks.shape
    
    # 1. 计算每个顶点在多少帧中出现
    frames_per_vertex = contacted_masks.sum(axis=0)  # shape [6890,]
    
    # 2. 计算每帧有多少个contact点
    vertices_per_frame = contacted_masks.sum(axis=1)  # shape [T,]
    
    # 3. 找出有contact的顶点
    valid_vertex_indices = np.where(frames_per_vertex > 0)[0]
    num_valid_vertices = len(valid_vertex_indices)
    
    # 4. 基础统计
    stats = {
        'total_frames': T,
        'total_vertices': num_verts,
        'num_valid_vertices': num_valid_vertices,
        'percent_valid_vertices': num_valid_vertices / num_verts * 100,
        'frames_with_contact': np.sum(vertices_per_frame > 0),
        'percent_frames_with_contact': np.sum(vertices_per_frame > 0) / T * 100,
    }
    
    print("=" * 60)
    print("CONTACT POINTS INSPECTION REPORT")
    print("=" * 60)
    print(f"Total frames analyzed: {stats['total_frames']}")
    print(f"Total vertices: {stats['total_vertices']}")
    print(f"Vertices with contact: {stats['num_valid_vertices']} ({stats['percent_valid_vertices']:.2f}%)")
    print(f"Frames with contact: {stats['frames_with_contact']} ({stats['percent_frames_with_contact']:.2f}%)")


import numpy as np

def pick_best_frames_per_segment(
    contacted_masks,                 # [T, 6890] bool (已过滤后的contact，如 filter_stable_contacts_simple 的输出)
    static_segments,                 # List[(start, end)]，end为开区间
    return_vertices=False,           # 是否同时返回该帧的接触顶点索引
    fallback="start"                 # 当片段内全为0接触时的回退策略: "start" | "middle" | "end" | "none"
):
    """
    返回:
        best_frame_indices: List[int or None]，每段一个最佳帧索引（或在 fallback='none' 且全0时为 None）
        (可选) best_vertices_per_segment: List[np.ndarray or None]，与上面一一对应
        counts_per_frame: np.ndarray [T]，每帧接触点数量（便于调试/可视化）
    """
    # 兼容 torch.Tensor
    if hasattr(contacted_masks, "detach"):
        contacted_masks = contacted_masks.detach().cpu().numpy()
    contacted_masks = contacted_masks.astype(bool)

    T = contacted_masks.shape[0]
    counts = contacted_masks.sum(axis=1)  # 每帧接触点数量

    best_indices = []
    best_vertices_list = [] if return_vertices else None

    for (s, e) in static_segments:
        s = int(s); e = int(e)
        if s < 0 or e > T or s >= e:
            # 片段非法时，给个占位
            best_indices.append(None)
            if return_vertices:
                best_vertices_list.append(None)
            continue

        sub_counts = counts[s:e]  # 该段内每帧接触数
        if sub_counts.size == 0:
            best_indices.append(None)
            if return_vertices:
                best_vertices_list.append(None)
            continue

        # 正常情况：取该段内的最大值（并行最早的帧）
        local_argmax = int(np.argmax(sub_counts))
        best_idx = s + local_argmax

        # 若整段内都是0接触，根据回退策略处理（默认取start）
        if sub_counts.max() == 0:
            if fallback == "start":
                best_idx = s
            elif fallback == "middle":
                best_idx = s + (e - s) // 2
            elif fallback == "end":
                best_idx = e - 1
            elif fallback == "none":
                best_idx = None

        best_indices.append(best_idx)

        if return_vertices:
            if best_idx is None:
                best_vertices_list.append(None)
            else:
                best_vertices = np.where(contacted_masks[best_idx])[0]
                best_vertices_list.append(best_vertices)

    if return_vertices:
        return best_indices, best_vertices_list, counts
    else:
        return best_indices, counts
