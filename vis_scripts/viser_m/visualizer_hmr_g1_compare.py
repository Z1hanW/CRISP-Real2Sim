#!/usr/bin/env python3
from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import viser

from visualizer_gmr_qpos import DEFAULT_G1_MJCF, MjcfQposViser, ViserHelper


REPO_ROOT = Path(__file__).resolve().parents[2]
BODY_MODELS_ROOT = REPO_ROOT / "prep/HMR/inputs/checkpoints/body_models"
DISPLAY_SCENE_ROOT = REPO_ROOT / "results/output/scene_vggt_omega_consistent_camera_min1"

SEQUENCES = {
    "49": "49_outdoor_big_stairs_down",
    "56": "56_outdoor_stairs_up_down",
    "78": "78_outdoor_stairs_up_down",
}

SOURCES = {
    "VGGT-Omega": {
        "hmr_root": REPO_ROOT / "results/init/hmr_vggt_omega",
        "qpos_root": REPO_ROOT / "results/output/retargeting_gmr_vggt_omega_scene_hmr",
        "scene_npz": lambda seq: REPO_ROOT / "results/output/scene" / f"{seq}_vggt_omega_gv_sgd_cvd_hr.npz",
        "color_hmr": (0.9, 0.45, 0.20),
    },
    "MegaSAM": {
        "hmr_root": REPO_ROOT / "results/init/hmr_megasam",
        "qpos_root": REPO_ROOT / "results/output/retargeting_gmr_megasam_scene_hmr",
        "scene_npz": lambda seq: REPO_ROOT / "results/output/scene" / f"{seq}_gv_sgd_cvd_hr.npz",
        "color_hmr": (0.20, 0.65, 0.95),
    },
}


@dataclass
class HmrTrack:
    vertices: np.ndarray
    joints: np.ndarray
    global_vertices: np.ndarray
    global_joints: np.ndarray
    frame_indices: np.ndarray
    faces: np.ndarray


@dataclass
class SourceTrack:
    hmr: HmrTrack
    qpos: np.ndarray
    fps: float
    robot_to_display_scale: float
    robot_to_display_R: np.ndarray
    robot_to_display_t: np.ndarray
    source_to_display_scale: float
    robot_fit_error: float
    robot_fit_points: int


@dataclass
class SequenceTracks:
    seq_name: str
    source_tracks: dict[str, SourceTrack]
    scene_vertices: np.ndarray
    scene_faces: np.ndarray


def _load_qpos(seq_name: str, qpos_root: Path) -> tuple[np.ndarray, float]:
    qpos_path = qpos_root / "gmr" / seq_name / "unitree_g1" / f"{seq_name}_unitree_g1_qpos.npz"
    if not qpos_path.is_file():
        raise FileNotFoundError(f"Missing GMR qpos: {qpos_path}")
    with np.load(qpos_path, allow_pickle=True) as data:
        return np.asarray(data["qpos"], dtype=np.float32), float(data["fps"]) if "fps" in data.files else 30.0


def _axis_angle_to_matrix_np(rotvec: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as sRot

    return sRot.from_rotvec(np.asarray(rotvec, dtype=np.float64).reshape(-1, 3)).as_matrix()


def _quat_wxyz_to_matrix_np(quat: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as sRot

    quat = np.asarray(quat, dtype=np.float64)
    xyzw = quat[..., [1, 2, 3, 0]]
    return sRot.from_quat(xyzw.reshape(-1, 4)).as_matrix().reshape(quat.shape[:-1] + (3, 3))


def _matrix_to_quat_wxyz_np(matrix: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as sRot

    xyzw = sRot.from_matrix(np.asarray(matrix, dtype=np.float64).reshape(-1, 3, 3)).as_quat()
    wxyz = xyzw[:, [3, 0, 1, 2]]
    return wxyz.reshape(matrix.shape[:-2] + (4,)).astype(np.float32)


def _estimate_similarity_umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    mask = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src = src[mask]
    dst = dst[mask]
    if src.shape[0] < 3:
        raise ValueError("Need at least 3 point pairs to estimate similarity transform")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    cov = (dst_centered.T @ src_centered) / float(src.shape[0])
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    var_src = np.mean(np.sum(src_centered * src_centered, axis=1))
    scale = float(np.sum(S) / max(var_src, 1e-12))
    t = dst_mean - scale * (R @ src_mean)
    return scale, R.astype(np.float32), t.astype(np.float32)


def _transform_points_similarity(points: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    points_np = np.asarray(points, dtype=np.float32)
    shape = points_np.shape
    flat = points_np.reshape(-1, 3)
    transformed = (float(scale) * (np.asarray(R, dtype=np.float32) @ flat.T).T) + np.asarray(t, dtype=np.float32)
    return transformed.reshape(shape).astype(np.float32)


def _transform_qpos_similarity(qpos: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    qpos_out = np.asarray(qpos, dtype=np.float32).copy()
    root_pos = qpos_out[:, :3].astype(np.float32)
    root_rot = qpos_out[:, 3:7].astype(np.float32)
    qpos_out[:, :3] = (float(scale) * (np.asarray(R, dtype=np.float32) @ root_pos.T).T) + np.asarray(t, dtype=np.float32)
    root_rot_mats = _quat_wxyz_to_matrix_np(root_rot)
    new_rot_mats = np.asarray(R, dtype=np.float64)[None, :, :] @ root_rot_mats
    qpos_out[:, 3:7] = _matrix_to_quat_wxyz_np(new_rot_mats)
    return qpos_out


ROBOT_TO_HMR_JOINTS = {
    "pelvis": 0,
    "left_knee_link": 4,
    "right_knee_link": 5,
    "torso_link": 9,
    "left_toe_link": 10,
    "right_toe_link": 11,
    "left_shoulder_yaw_link": 16,
    "right_shoulder_yaw_link": 17,
    "left_elbow_link": 18,
    "right_elbow_link": 19,
    "left_wrist_yaw_link": 20,
    "right_wrist_yaw_link": 21,
}


def _estimate_robot_to_display(
    qpos: np.ndarray,
    hmr_display_joints: np.ndarray,
    robot_mjcf: Path,
) -> tuple[float, np.ndarray, np.ndarray, float, int]:
    import mujoco as mj

    model = mj.MjModel.from_xml_path(str(robot_mjcf))
    data = mj.MjData(model)
    body_ids = {
        name: mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        for name in ROBOT_TO_HMR_JOINTS
    }

    frame_count = min(int(qpos.shape[0]), int(hmr_display_joints.shape[0]))
    src_rows: list[np.ndarray] = []
    dst_rows: list[np.ndarray] = []
    for frame_idx in range(frame_count):
        data.qpos[:] = np.asarray(qpos[frame_idx], dtype=np.float64)
        mj.mj_forward(model, data)
        for body_name, joint_idx in ROBOT_TO_HMR_JOINTS.items():
            body_id = body_ids[body_name]
            if body_id < 0 or joint_idx >= hmr_display_joints.shape[1]:
                continue
            src_rows.append(np.asarray(data.xpos[body_id], dtype=np.float32))
            dst_rows.append(np.asarray(hmr_display_joints[frame_idx, joint_idx], dtype=np.float32))

    if len(src_rows) < 3:
        raise ValueError("Need at least 3 robot/HMR body pairs to estimate robot display transform")

    src = np.stack(src_rows, axis=0)
    dst = np.stack(dst_rows, axis=0)
    scale, R, t = _estimate_similarity_umeyama(src, dst)
    pred = _transform_points_similarity(src, scale, R, t)
    fit_error = float(np.linalg.norm(pred - dst, axis=1).mean())
    return scale, R, t, fit_error, len(src_rows)


def _load_display_hmr_joints(seq_name: str) -> np.ndarray:
    joint_path = DISPLAY_SCENE_ROOT / seq_name / "gv/hmr/hps_track_smplx.npz"
    if not joint_path.is_file():
        raise FileNotFoundError(f"Missing display-frame HMR joints: {joint_path}")
    with np.load(joint_path, allow_pickle=True) as data:
        if "global_joint_positions" not in data.files:
            raise KeyError(f"Missing global_joint_positions in {joint_path}")
        joints = np.asarray(data["global_joint_positions"], dtype=np.float32)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"Expected display joints shape (T,J,3), got {joints.shape} from {joint_path}")
    return joints[:, :22, :]


def _estimate_source_to_display_scene(
    hmr: HmrTrack,
    display_joints: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, int]:
    display_joints = np.asarray(display_joints, dtype=np.float32)
    source_joints = np.asarray(hmr.joints, dtype=np.float32)

    display_by_frame = {int(frame_idx): i for i, frame_idx in enumerate(range(display_joints.shape[0]))}
    src_rows: list[np.ndarray] = []
    dst_rows: list[np.ndarray] = []
    for src_i, frame_idx in enumerate(np.asarray(hmr.frame_indices, dtype=np.int64).tolist()):
        dst_i = display_by_frame.get(int(frame_idx))
        if dst_i is None or src_i >= source_joints.shape[0]:
            continue
        src_rows.append(source_joints[src_i])
        dst_rows.append(display_joints[dst_i, : source_joints.shape[1]])

    if len(src_rows) < 3:
        count = min(source_joints.shape[0], display_joints.shape[0])
        if count <= 0:
            raise ValueError("No common frames for source->display transform")
        src = source_joints[:count]
        dst = display_joints[:count, : source_joints.shape[1]]
        return (*_estimate_similarity_umeyama(src, dst), count)

    src = np.stack(src_rows, axis=0)
    dst = np.stack(dst_rows, axis=0)
    return (*_estimate_similarity_umeyama(src, dst), len(src_rows))


def _take_hmr_frames(value, indices: np.ndarray):
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value
        idx = torch.as_tensor(indices, dtype=torch.long, device=value.device)
        return value.index_select(0, idx)
    if isinstance(value, dict):
        return {k: _take_hmr_frames(v, indices) for k, v in value.items()}
    return value


def _load_hmr_mesh(seq_name: str, hmr_root: Path, scene_npz: Path, device: str) -> HmrTrack:
    import smplx

    hmr_path = hmr_root / seq_name / "hmr4d_results.pt"
    if not hmr_path.is_file():
        raise FileNotFoundError(f"Missing HMR result: {hmr_path}")
    if not scene_npz.is_file():
        raise FileNotFoundError(f"Missing aligned scene npz: {scene_npz}")

    pred = torch.load(hmr_path, map_location="cpu")
    smpl_params = pred["smpl_params_incam"]
    smpl_params_global = pred["smpl_params_global"]
    raw_count = int(next(v.shape[0] for v in smpl_params.values() if torch.is_tensor(v) and v.ndim > 0))

    scene = np.load(scene_npz, allow_pickle=True)
    cam_c2w = np.asarray(scene["cam_c2w"], dtype=np.float32)
    scale = float(scene["scale"]) if "scale" in scene.files else 1.0

    if "valid_source_frame_indices" in scene.files:
        frame_indices = np.asarray(scene["valid_source_frame_indices"], dtype=np.int64)
    elif "source_frame_indices" in scene.files:
        frame_indices = np.asarray(scene["source_frame_indices"], dtype=np.int64)
    else:
        frame_indices = np.arange(min(len(cam_c2w), raw_count), dtype=np.int64)

    frame_indices = frame_indices[(frame_indices >= 0) & (frame_indices < raw_count)]
    frame_count = min(len(cam_c2w), len(frame_indices))
    if frame_count <= 0:
        raise ValueError(f"No usable HMR frames for {seq_name}")
    frame_indices = frame_indices[:frame_count]
    cam_c2w = cam_c2w[:frame_count]
    smpl_params = _take_hmr_frames(smpl_params, frame_indices)
    smpl_params_global = _take_hmr_frames(smpl_params_global, frame_indices)

    model = smplx.create(
        model_path=str(BODY_MODELS_ROOT),
        model_type="smplx",
        gender="neutral",
        num_betas=10,
        num_expression_coeffs=10,
        num_pca_comps=12,
        flat_hand_mean=False,
        batch_size=frame_count,
    ).to(device)

    params_device = {k: v.to(device) for k, v in smpl_params.items()}
    params_global_device = {k: v.to(device) for k, v in smpl_params_global.items()}
    world_R = torch.as_tensor(cam_c2w[:, :3, :3], dtype=torch.float32, device=device)
    world_T = torch.as_tensor(cam_c2w[:, :3, 3] * scale, dtype=torch.float32, device=device)

    with torch.no_grad():
        out = model(**params_device)
        verts_cam = out.vertices
        joints_cam = out.joints[:, :22, :]
        verts_world = torch.einsum("bij,bvj->bvi", world_R, verts_cam) + world_T[:, None, :]
        joints_world = torch.einsum("bij,bvj->bvi", world_R, joints_cam) + world_T[:, None, :]
        global_out = model(**params_global_device)
        verts_global = global_out.vertices
        joints_global = global_out.joints[:, :22, :]
    vertices = verts_world.detach().cpu().numpy().astype(np.float32)
    joints = joints_world.detach().cpu().numpy().astype(np.float32)
    global_vertices = verts_global.detach().cpu().numpy().astype(np.float32)
    global_joints = joints_global.detach().cpu().numpy().astype(np.float32)
    faces = np.asarray(model.faces, dtype=np.int32)
    return HmrTrack(
        vertices=vertices,
        joints=joints,
        global_vertices=global_vertices,
        global_joints=global_joints,
        frame_indices=frame_indices.astype(np.int64, copy=False),
        faces=faces,
    )


def _load_scene_mesh(seq_name: str) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    scene_obj = (
        DISPLAY_SCENE_ROOT
        / seq_name
        / "gv/scene_mesh_sqs/scene_mesh_sqs.obj"
    )
    if not scene_obj.is_file():
        raise FileNotFoundError(f"Missing scene mesh OBJ: {scene_obj}")
    loaded = trimesh.load(scene_obj, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    return np.asarray(loaded.vertices, dtype=np.float32), np.asarray(loaded.faces, dtype=np.int32)


class DynamicMesh:
    def __init__(self, server: viser.ViserServer, name: str, faces: np.ndarray, color: tuple[float, float, float]):
        self.server = server
        self.name = name
        self.faces = faces
        self.color = color
        self.handle = None

    def update(self, vertices: np.ndarray, visible: bool) -> None:
        if not visible:
            if self.handle is not None:
                self.handle.visible = False
            return
        if self.handle is not None:
            self.handle.remove()
        self.handle = self.server.scene.add_mesh_simple(
            self.name,
            vertices=vertices,
            faces=self.faces,
            color=self.color,
            side="double",
            flat_shading=False,
            cast_shadow=True,
            receive_shadow=True,
        )
        self.handle.visible = True

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def _set_robot_visible(robot: MjcfQposViser, visible: bool) -> None:
    for _, name in robot.geom_names:
        handle = robot.viser._handles.get(name)
        if handle is not None:
            handle.visible = visible


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare HMR mesh and GMR G1 retargeting in one Viser viewer.")
    parser.add_argument("--port", type=int, default=9300)
    parser.add_argument("--sequence", choices=sorted(SEQUENCES), default="56")
    parser.add_argument("--robot-mjcf", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--display-fps", type=float, default=3.0)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.set_up_direction("-z")
    if args.share:
        server.request_share_url()
    helper = object.__new__(ViserHelper)
    helper.port = int(args.port)
    helper._server = server
    helper._ok = True
    helper._handles = {}

    state_lock = threading.RLock()
    cache: dict[str, SequenceTracks] = {}
    current_seq_key = args.sequence
    current_tracks: Optional[SequenceTracks] = None

    robots: dict[str, MjcfQposViser] = {}
    hmr_meshes: dict[str, DynamicMesh] = {}
    scene_handle = None
    frame_value = 0.0

    with server.gui.add_folder("Playback"):
        gui_sequence = server.gui.add_dropdown("Sequence", options=tuple(SEQUENCES.keys()), initial_value=args.sequence)
        gui_timestep = server.gui.add_slider("Timestep", min=0, max=1, step=1, initial_value=0)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_display_fps = server.gui.add_slider("Display FPS", min=1.0, max=15.0, step=0.5, initial_value=float(args.display_fps))

    with server.gui.add_folder("Layers"):
        gui_show_scene = server.gui.add_checkbox("Scene", True)
        gui_show_vggt_hmr = server.gui.add_checkbox("VGGT HMR", True)
        gui_show_vggt_g1 = server.gui.add_checkbox("VGGT G1", True)
        gui_show_megasam_hmr = server.gui.add_checkbox("MegaSAM HMR", False)
        gui_show_megasam_g1 = server.gui.add_checkbox("MegaSAM G1", False)

    def _source_visible(source: str, kind: str) -> bool:
        if source == "VGGT-Omega":
            return bool(gui_show_vggt_hmr.value if kind == "hmr" else gui_show_vggt_g1.value)
        return bool(gui_show_megasam_hmr.value if kind == "hmr" else gui_show_megasam_g1.value)

    def _load_sequence(seq_key: str) -> SequenceTracks:
        if seq_key in cache:
            return cache[seq_key]
        seq_name = SEQUENCES[seq_key]
        display_joints = _load_display_hmr_joints(seq_name)
        source_tracks: dict[str, SourceTrack] = {}
        for source_name, cfg in SOURCES.items():
            hmr = _load_hmr_mesh(
                seq_name=seq_name,
                hmr_root=cfg["hmr_root"],
                scene_npz=cfg["scene_npz"](seq_name),
                device=args.device,
            )
            qpos, fps = _load_qpos(seq_name, cfg["qpos_root"])
            display_scale, display_R, display_t, matched_frames = _estimate_source_to_display_scene(hmr, display_joints)
            count = min(hmr.vertices.shape[0], qpos.shape[0], display_joints.shape[0])
            hmr_display = replace(
                hmr,
                vertices=_transform_points_similarity(hmr.vertices[:count], display_scale, display_R, display_t),
                joints=_transform_points_similarity(hmr.joints[:count], display_scale, display_R, display_t),
                global_vertices=hmr.global_vertices[:count],
                global_joints=hmr.global_joints[:count],
                frame_indices=hmr.frame_indices[:count],
            )
            robot_scale, robot_R, robot_t, robot_fit_error, robot_fit_points = _estimate_robot_to_display(
                qpos[:count],
                hmr_display.joints,
                args.robot_mjcf,
            )
            print(
                f"[compare-vis] {seq_key} {source_name}: source->display scale={display_scale:.6f} "
                f"matched_frames={matched_frames} robot->display scale={robot_scale:.6f} "
                f"fit_error={robot_fit_error:.4f} fit_points={robot_fit_points}",
                flush=True,
            )
            source_tracks[source_name] = SourceTrack(
                hmr=hmr_display,
                qpos=qpos[:count],
                fps=fps,
                robot_to_display_scale=robot_scale,
                robot_to_display_R=robot_R,
                robot_to_display_t=robot_t,
                source_to_display_scale=display_scale,
                robot_fit_error=robot_fit_error,
                robot_fit_points=robot_fit_points,
            )
        scene_vertices, scene_faces = _load_scene_mesh(seq_name)
        tracks = SequenceTracks(seq_name=seq_name, source_tracks=source_tracks, scene_vertices=scene_vertices, scene_faces=scene_faces)
        cache[seq_key] = tracks
        return tracks

    def _remove_current_handles() -> None:
        nonlocal scene_handle
        for mesh in hmr_meshes.values():
            mesh.remove()
        hmr_meshes.clear()
        for robot in robots.values():
            for _, name in robot.geom_names:
                handle = robot.viser._handles.pop(name, None)
                if handle is not None:
                    handle.remove()
        robots.clear()
        if scene_handle is not None:
            scene_handle.remove()
            scene_handle = None

    def _activate_sequence(seq_key: str) -> None:
        nonlocal current_tracks, current_seq_key, scene_handle, frame_value
        tracks = _load_sequence(seq_key)
        with state_lock:
            current_seq_key = seq_key
            current_tracks = tracks
            frame_value = 0.0
            _remove_current_handles()
            scene_handle = server.scene.add_mesh_simple(
                "/scene/mesh",
                vertices=tracks.scene_vertices,
                faces=tracks.scene_faces,
                color=(0.58, 0.62, 0.68),
                opacity=0.55,
                side="double",
                flat_shading=True,
                cast_shadow=True,
                receive_shadow=True,
                visible=bool(gui_show_scene.value),
            )
            for source_name, track in tracks.source_tracks.items():
                prefix = "/robot/vggt" if source_name == "VGGT-Omega" else "/robot/megasam"
                robots[source_name] = MjcfQposViser(
                    helper,
                    args.robot_mjcf,
                    prefix=prefix,
                    mesh_scale=track.robot_to_display_scale,
                )
                _set_robot_visible(robots[source_name], _source_visible(source_name, "g1"))
                color = SOURCES[source_name]["color_hmr"]
                hmr_name = "/hmr/vggt" if source_name == "VGGT-Omega" else "/hmr/megasam"
                hmr_meshes[source_name] = DynamicMesh(server, hmr_name, track.hmr.faces, color)
            max_frames = max(t.qpos.shape[0] for t in tracks.source_tracks.values())
            gui_timestep.max = max(0, max_frames - 1)
            gui_timestep.value = 0
            _update_frame_locked(0)
            root0 = tracks.source_tracks["VGGT-Omega"].hmr.joints[0, 0]
            for _, client in server.get_clients().items():
                client.camera.position = root0 + np.array([0.0, -3.0, 1.8], dtype=np.float32)
                client.camera.look_at = root0 + np.array([0.0, 0.0, 0.8], dtype=np.float32)

    def _update_frame_locked(frame_idx: int) -> None:
        if current_tracks is None:
            return
        if scene_handle is not None:
            scene_handle.visible = bool(gui_show_scene.value)
        for source_name, track in current_tracks.source_tracks.items():
            idx = int(frame_idx) % int(track.qpos.shape[0])
            if source_name in robots:
                visible_g1 = _source_visible(source_name, "g1")
                robots[source_name].update(
                    track.qpos[idx],
                    world_scale=track.robot_to_display_scale,
                    world_R=track.robot_to_display_R,
                    world_t=track.robot_to_display_t,
                )
                _set_robot_visible(robots[source_name], visible_g1)
            if source_name in hmr_meshes:
                visible_hmr = _source_visible(source_name, "hmr")
                hmr_idx = int(frame_idx) % int(track.hmr.vertices.shape[0])
                hmr_meshes[source_name].update(track.hmr.vertices[hmr_idx], visible_hmr)

    def _update_frame(frame_idx: int) -> None:
        with state_lock:
            _update_frame_locked(frame_idx)

    @gui_sequence.on_update
    def _(_event) -> None:
        _activate_sequence(str(gui_sequence.value))

    for handle in (gui_show_scene, gui_show_vggt_hmr, gui_show_vggt_g1, gui_show_megasam_hmr, gui_show_megasam_g1):
        @handle.on_update
        def _(_event) -> None:
            _update_frame(int(round(gui_timestep.value)))

    @gui_timestep.on_update
    def _(_event) -> None:
        if not gui_playing.value:
            _update_frame(int(round(gui_timestep.value)))

    print(f"[compare-vis] loading initial sequence {args.sequence} on {args.device} ...", flush=True)
    _activate_sequence(args.sequence)
    print(f"[compare-vis] ready: http://localhost:{args.port}", flush=True)
    print("[compare-vis] GUI checkboxes: Scene, VGGT HMR/G1, MegaSAM HMR/G1", flush=True)

    try:
        while True:
            display_fps = max(float(gui_display_fps.value), 1e-6)
            time.sleep(1.0 / display_fps)
            with state_lock:
                if current_tracks is None or not gui_playing.value:
                    continue
                frame_value += 30.0 / display_fps
                max_frames = max(t.qpos.shape[0] for t in current_tracks.source_tracks.values())
                frame_idx = int(frame_value) % max_frames
                gui_timestep.value = frame_idx
                _update_frame_locked(frame_idx)
    except KeyboardInterrupt:
        print("\n[compare-vis] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
