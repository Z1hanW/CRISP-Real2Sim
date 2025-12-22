#!/usr/bin/env python3
"""
Bridge post_scene SMPL outputs into Holosoma-ready SMPLH data.

This script reads the *_ours.npz files produced under results/output/post_scene,
reconstructs SMPL joints, maps them to the SMPLH joint ordering used by Holosoma,
and writes InterMimic-style .pt files so the existing retargeter can be run
without touching Holosoma code. A matching height_dict.pkl is also generated
for scale estimation. __release/prep/UFM
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import smplx
import torch
from smplx import joint_names as smplx_joint_names

# SMPLH joint order expected by Holosoma (copied from holosoma_retargeting/config_types/data_type.py)
SMPLH_DEMO_JOINTS: List[str] = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Index1",
    "L_Index2",
    "L_Index3",
    "L_Middle1",
    "L_Middle2",
    "L_Middle3",
    "L_Pinky1",
    "L_Pinky2",
    "L_Pinky3",
    "L_Ring1",
    "L_Ring2",
    "L_Ring3",
    "L_Thumb1",
    "L_Thumb2",
    "L_Thumb3",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Index1",
    "R_Index2",
    "R_Index3",
    "R_Middle1",
    "R_Middle2",
    "R_Middle3",
    "R_Pinky1",
    "R_Pinky2",
    "R_Pinky3",
    "R_Ring1",
    "R_Ring2",
    "R_Ring3",
    "R_Thumb1",
    "R_Thumb2",
    "R_Thumb3",
]

# Special cases where the SMPLH demo name does not directly match smplx joint names
SPECIAL_NAME_MAP = {
    "torso": "spine1",
    "spine": "spine2",
    "chest": "spine3",
    "l_toe": "left_foot",
    "r_toe": "right_foot",
    "l_thorax": "left_collar",
    "r_thorax": "right_collar",
}


def _demo_to_smplx_name(demo_name: str) -> str:
    """Convert SMPLH demo joint name to the smplx joint_names style."""
    key = demo_name.lower()
    key = key.replace("l_", "left_").replace("r_", "right_")
    return SPECIAL_NAME_MAP.get(key, key)


def _load_height_dict(height_path: Path) -> Dict[str, float]:
    if height_path.exists():
        with height_path.open("rb") as f:
            return pickle.load(f)
    return {}


def _save_height_dict(height_path: Path, values: Dict[str, float]) -> None:
    height_path.parent.mkdir(parents=True, exist_ok=True)
    with height_path.open("wb") as f:
        pickle.dump(values, f)


def _remap_joints(
    joints: np.ndarray, smplx_name_to_idx: Dict[str, int]
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """Remap smplx joint array to SMPLH demo ordering, returning missing entries."""
    out = np.zeros((joints.shape[0], len(SMPLH_DEMO_JOINTS), 3), dtype=np.float32)
    missing: List[Tuple[str, str]] = []

    for i, demo_name in enumerate(SMPLH_DEMO_JOINTS):
        smplx_name = _demo_to_smplx_name(demo_name)
        idx = smplx_name_to_idx.get(smplx_name)
        if idx is not None and idx < joints.shape[1]:
            out[:, i] = joints[:, idx]
            continue

        # Fallback to a parent joint if the fine joint is unavailable.
        if demo_name.startswith("L_"):
            parent_name = "left_wrist"
        elif demo_name.startswith("R_"):
            parent_name = "right_wrist"
        else:
            parent_name = "pelvis"
        parent_idx = smplx_name_to_idx.get(parent_name)
        if parent_idx is not None and parent_idx < joints.shape[1]:
            out[:, i] = joints[:, parent_idx]
        missing.append((demo_name, smplx_name))

    return out, missing


def _build_intermimic_tensor(human_joints: np.ndarray, object_pose: np.ndarray) -> torch.Tensor:
    """
    Create a minimal tensor matching load_intermimic_data expectations:
    - human joints stored from column 162 (52 * 3 positions)
    - object pose stored from column 318 in order [x, y, z, qx, qy, qz, qw]
    """
    t, j, _ = human_joints.shape
    payload = torch.zeros((t, 325), dtype=torch.float32)
    payload[:, 162 : 162 + j * 3] = torch.from_numpy(human_joints.reshape(t, -1))
    payload[:, 318:325] = torch.from_numpy(object_pose)
    return payload


def _default_object_pose(num_frames: int) -> np.ndarray:
    """Identity object pose repeated over time in [x, y, z, qx, qy, qz, qw] order."""
    pose = np.zeros((num_frames, 7), dtype=np.float32)
    pose[:, -1] = 1.0  # qw
    return pose


def process_sequence(
    npz_path: Path,
    model: smplx.SMPL,
    output_dir: Path,
    height_map: Dict[str, float],
) -> None:
    data = np.load(npz_path)
    poses = torch.from_numpy(data["poses"]).float()
    transl = torch.from_numpy(data["trans"]).float()
    betas = torch.from_numpy(data["betas"]).float()

    out = model(
        betas=betas,
        transl=transl,
        global_orient=poses[:, :3],
        body_pose=poses[:, 3:],
        return_verts=False,
    )
    smplx_name_to_idx = {name: idx for idx, name in enumerate(smplx_joint_names.SMPLH_JOINT_NAMES)}
    joints_np = out.joints.detach().cpu().numpy().astype(np.float32)
    remapped, missing = _remap_joints(joints_np, smplx_name_to_idx)
    if missing:
        print(f"[WARN] {npz_path.name}: missing joints (filled with parents): {missing}")

    height = float(remapped[..., 2].max() - remapped[..., 2].min())
    subject_id = npz_path.stem.split("_")[0]
    height_map[subject_id] = height

    object_pose = _default_object_pose(remapped.shape[0])
    intermimic_tensor = _build_intermimic_tensor(remapped, object_pose)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(intermimic_tensor, output_dir / f"{npz_path.stem}.pt")

    # Optional debug NPZ for inspection or alternative loaders
    np.savez(
        output_dir / f"{npz_path.stem}_smplh_joints.npz",
        global_joint_positions=remapped,
        height=height,
        mocap_framerate=data.get("mocap_framerate", 30),
    )


def discover_npz_files(input_root: Path, hmr_type: str) -> List[Path]:
    return sorted(input_root.glob(f"*/{hmr_type}/*_ours.npz"))


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    default_input = repo_root / "results" / "output" / "post_scene"
    default_output = repo_root / "holosoma" / "demo_data" / "ours_omniretarget"
    default_height_dict = repo_root / "holosoma" / "demo_data" / "height_dict.pkl"
    default_model = repo_root / "prep" / "data" / "smpl" / "SMPL_NEUTRAL.pkl"

    parser = argparse.ArgumentParser(description="Convert post_scene outputs into Holosoma-ready SMPLH PT files.")
    parser.add_argument("--input-root", type=Path, default=default_input, help="Root of post_scene outputs.")
    parser.add_argument("--hmr-type", type=str, default="gv", help="Subfolder name under each sequence.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output,
        help="Where to write Holosoma-compatible files (pt + debug npz).",
    )
    parser.add_argument(
        "--height-dict",
        type=Path,
        default=default_height_dict,
        help="Path to height_dict.pkl used by Holosoma (will be created/updated).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=default_model,
        help="Path to the SMPL model file (SMPL_NEUTRAL.pkl).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root
    npz_files = discover_npz_files(input_root, args.hmr_type)
    if not npz_files:
        raise SystemExit(f"No *_ours.npz files found under {input_root}")

    model = smplx.create(
        model_path=str(args.model_path),
        model_type="smpl",
        gender="neutral",
        num_betas=10,
        batch_size=1,
    )

    height_dict = _load_height_dict(args.height_dict)
    for npz_path in npz_files:
        seq_output_dir = args.output_root
        process_sequence(npz_path, model, seq_output_dir, height_dict)

    _save_height_dict(args.height_dict, height_dict)
    print(f"[OK] Wrote {len(npz_files)} sequences to {args.output_root}")
    print(f"[OK] Updated height dict at {args.height_dict}")


if __name__ == "__main__":
    main()
