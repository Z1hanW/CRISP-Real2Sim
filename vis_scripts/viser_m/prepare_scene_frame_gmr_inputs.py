#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
from smplx.joint_names import JOINT_NAMES

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import visualizer_hmr_g1_compare as compare  # noqa: E402


SOURCE_SPECS = {
    "vggt_omega": {
        "hmr_root": compare.REPO_ROOT / "results/init/hmr_vggt_omega",
        "scene_npz": lambda seq: compare.REPO_ROOT / "results/output/scene" / f"{seq}_vggt_omega_gv_sgd_cvd_hr.npz",
        "hps_track": lambda seq: compare.DISPLAY_SCENE_ROOT / seq / "gv/hmr/hps_track.npy",
        "already_display": True,
    },
    "megasam": {
        "hmr_root": compare.REPO_ROOT / "results/init/hmr_megasam",
        "scene_npz": lambda seq: compare.REPO_ROOT / "results/output/scene" / f"{seq}_gv_sgd_cvd_hr.npz",
        "hps_track": lambda seq: compare.REPO_ROOT / "results/output/scene" / seq / "gv/hmr/hps_track.npy",
        "already_display": False,
    },
}


def _as_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_torch(value, *, device: str) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _hps_track_frame_count(hps_track: Path) -> int:
    payload = np.load(hps_track, allow_pickle=True).item()
    return int(_as_numpy(payload["transl"]).shape[0])


def _build_smpl_joint_npz_from_hps_track(
    hps_track: Path,
    out_npz: Path,
    *,
    frame_count: int,
    device: str,
) -> Path:
    import smplx

    payload = np.load(hps_track, allow_pickle=True).item()
    body_pose = _as_torch(payload["body_pose"], device=device)[:frame_count, :23].reshape(frame_count, 23, 3, 3)
    global_orient = _as_torch(payload["global_orient"], device=device)[:frame_count].reshape(frame_count, 1, 3, 3)
    betas = _as_torch(payload["betas"], device=device)[:frame_count, :10].reshape(frame_count, 10)
    transl = _as_torch(payload["transl"], device=device)[:frame_count].reshape(frame_count, 3)

    model = smplx.create(
        model_path=str(compare.BODY_MODELS_ROOT),
        model_type="smpl",
        gender="neutral",
        num_betas=10,
        batch_size=frame_count,
    ).to(device)
    with torch.no_grad():
        output = model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
            pose2rot=False,
        )
        joints = output.joints[:, :22, :].detach().cpu().numpy().astype(np.float32)
        vertices = output.vertices.detach().cpu().numpy().astype(np.float32)

    per_frame_height = vertices[:, :, 2].max(axis=1) - vertices[:, :, 2].min(axis=1)
    height = np.float32(np.median(per_frame_height))
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        global_joint_positions=joints,
        height=height,
        source_hps_track=str(hps_track),
    )
    return out_npz


def _convert_hps_track(
    hps_track: Path,
    out_npz: Path,
    *,
    frame_count: int,
    scale: float,
    R: np.ndarray,
    t: np.ndarray,
    source_name: str,
    seq_name: str,
    display_root: Path,
) -> None:
    payload = np.load(hps_track, allow_pickle=True).item()
    body_pose_mats = _as_numpy(payload["body_pose"])[:frame_count, :21].astype(np.float32)
    global_orient_mats = _as_numpy(payload["global_orient"])[:frame_count].reshape(frame_count, 3, 3).astype(np.float32)
    transl = _as_numpy(payload["transl"])[:frame_count].astype(np.float32)
    betas = _as_numpy(payload["betas"])[0, :10].astype(np.float32)

    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    t = np.asarray(t, dtype=np.float32).reshape(3)
    transl = (float(scale) * (R @ transl.T).T + t).astype(np.float32)
    global_orient_mats = (R[None, :, :] @ global_orient_mats).astype(np.float32)

    pose_body = sRot.from_matrix(body_pose_mats.reshape(-1, 3, 3)).as_rotvec().astype(np.float32).reshape(frame_count, 63)
    root_orient = sRot.from_matrix(global_orient_mats).as_rotvec().astype(np.float32)
    betas_16 = np.pad(betas, (0, 6)).astype(np.float32)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        pose_body=pose_body,
        root_orient=root_orient,
        trans=transl,
        betas=betas_16,
        gender=np.array("neutral"),
        mocap_frame_rate=np.array(30, dtype=np.float32),
        source_hps_track=str(hps_track),
        source_name=np.array(source_name),
        seq_name=np.array(seq_name),
        display_scene_root=str(display_root),
        source_to_display_scale=np.array(scale, dtype=np.float32),
        source_to_display_R=R.astype(np.float32),
        source_to_display_t=t.astype(np.float32),
    )


def _write_joint_frames_input(
    *,
    joint_npz: Path,
    source_smplx_npz: Path,
    out_npz: Path,
    frame_count: int,
    source_name: str,
    seq_name: str,
    display_root: Path,
) -> None:
    with np.load(joint_npz, allow_pickle=True) as data:
        if "global_joint_positions" not in data.files:
            raise KeyError(f"Missing global_joint_positions in {joint_npz}")
        joints = np.asarray(data["global_joint_positions"], dtype=np.float32)[:frame_count, :22, :]
        stored_height = float(np.asarray(data["height"]).reshape(-1)[0]) if "height" in data.files else float("nan")

    per_frame_extent = joints[:, :, 2].max(axis=1) - joints[:, :, 2].min(axis=1)
    joint_height = float(np.nanmedian(per_frame_extent)) if per_frame_extent.size else float("nan")
    if np.isfinite(stored_height) and 1.1 <= stored_height <= 2.4:
        height = np.float32(stored_height)
        height_source = "stored_height"
    elif np.isfinite(joint_height) and 1.1 <= joint_height <= 2.4:
        height = np.float32(joint_height)
        height_source = "joint_z_extent_median"
    else:
        height = np.float32(np.nan)
        height_source = "unknown"

    body_names = np.asarray(JOINT_NAMES[: joints.shape[1]], dtype=object)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        body_positions_world=joints.astype(np.float32, copy=False),
        body_names=body_names,
        source_smplx_npz=str(source_smplx_npz),
        actual_human_height_m=height,
        mocap_frame_rate=np.array(30, dtype=np.float32),
        source_joint_npz=str(joint_npz),
        source_name=np.array(source_name),
        seq_name=np.array(seq_name),
        display_scene_root=str(display_root),
        height_source=np.array(height_source),
    )


def _resolve_seq_name(seq_key_or_name: str) -> str:
    return compare.SEQUENCES.get(seq_key_or_name, seq_key_or_name)


def prepare_one(
    seq_key: str,
    source_key: str,
    out_root: Path,
    device: str,
    output_formats: set[str],
    display_root: Path,
) -> list[Path]:
    seq_name = _resolve_seq_name(seq_key)
    spec = SOURCE_SPECS[source_key]
    hps_track = display_root / seq_name / "gv/hmr/hps_track.npy"
    if not hps_track.is_file():
        hps_track = spec["hps_track"](seq_name)
    if not hps_track.is_file():
        raise FileNotFoundError(f"Missing hps_track.npy for {source_key} {seq_name}: {hps_track}")

    if spec["already_display"]:
        frame_count = _hps_track_frame_count(hps_track)
        scale = 1.0
        R = np.eye(3, dtype=np.float32)
        t = np.zeros(3, dtype=np.float32)
    else:
        display_joints = compare._load_display_hmr_joints(seq_name)
        hmr = compare._load_hmr_mesh(
            seq_name=seq_name,
            hmr_root=spec["hmr_root"],
            scene_npz=spec["scene_npz"](seq_name),
            device=device,
        )
        scale, R, t, _ = compare._estimate_source_to_display_scene(hmr, display_joints)
        frame_count = min(int(hmr.joints.shape[0]), int(display_joints.shape[0]))

    out_dir = out_root / source_key / seq_name
    smplx_npz = out_dir / "smplx_scene.npz"
    _convert_hps_track(
        hps_track,
        smplx_npz,
        frame_count=frame_count,
        scale=scale,
        R=R,
        t=t,
        source_name=source_key,
        seq_name=seq_name,
        display_root=display_root,
    )
    outputs: list[Path] = []
    if "smplx_npz" in output_formats:
        outputs.append(smplx_npz)
        print(
            f"[scene-gmr-input] {source_key} {seq_name}: {smplx_npz} "
            f"frames={frame_count} scale={scale:.6f}",
            flush=True,
        )

    if "joint_frames_npz" in output_formats:
        joint_npz = display_root / seq_name / "gv/hmr/hps_track_smplx.npz"
        if not joint_npz.is_file():
            joint_npz = display_root / seq_name / "gv/hmr" / f"{seq_name}.npz"
        if not joint_npz.is_file():
            joint_npz = _build_smpl_joint_npz_from_hps_track(
                hps_track,
                out_dir / "hps_track_smpl_joints.npz",
                frame_count=frame_count,
                device=device,
            )
        out_npz = out_dir / "joint_frames_scene.npz"
        _write_joint_frames_input(
            joint_npz=joint_npz,
            source_smplx_npz=smplx_npz,
            out_npz=out_npz,
            frame_count=frame_count,
            source_name=source_key,
            seq_name=seq_name,
            display_root=display_root,
        )
        outputs.append(out_npz)
        print(
            f"[scene-gmr-input] {source_key} {seq_name}: {out_npz} "
            f"frames={frame_count} source_smplx={smplx_npz}",
            flush=True,
        )
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare scene/display-frame SMPL-X npz inputs for GMR.")
    parser.add_argument("--out-root", type=Path, default=compare.REPO_ROOT / "results/output/gmr_scene_inputs")
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=sorted(compare.SEQUENCES.keys()),
        help="Sequence keys from visualizer_hmr_g1_compare or literal sequence names such as stair_0.",
    )
    parser.add_argument("--sources", nargs="+", default=sorted(SOURCE_SPECS.keys()), choices=sorted(SOURCE_SPECS.keys()))
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["smplx_npz"],
        choices=("smplx_npz", "joint_frames_npz"),
        help="GMR input format files to write. joint_frames_npz also writes the supporting smplx_scene.npz.",
    )
    parser.add_argument("--display-root", type=Path, default=compare.DISPLAY_SCENE_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    output_formats = set(args.formats)
    display_root = args.display_root.expanduser().resolve()
    for seq_key in args.sequences:
        for source_key in args.sources:
            prepare_one(seq_key, source_key, args.out_root, args.device, output_formats, display_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
