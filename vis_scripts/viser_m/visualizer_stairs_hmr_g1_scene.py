#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import viser


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from visualizer_gmr_qpos import DEFAULT_G1_MJCF, MjcfQposViser, ViserHelper  # noqa: E402


BODY_MODELS_ROOT = REPO_ROOT / "prep/HMR/inputs/checkpoints/body_models"
DEFAULT_DISPLAY_ROOT = REPO_ROOT / "results/output/post_scene_vggt_omega_consistent_camera_min1"
DEFAULT_QPOS_ROOT = REPO_ROOT / "results/output/retargeting_gmr_vggt_omega_post_scene_hmr"


@dataclass
class HmrTrack:
    vertices: np.ndarray
    joints: np.ndarray
    faces: np.ndarray


@dataclass
class StairsTrack:
    seq_name: str
    hmr: HmrTrack
    qpos: np.ndarray
    fps: float
    scene_vertices: np.ndarray
    scene_faces: np.ndarray
    robot_position_scale: float
    scale_source: str


def _natural_stair_key(name: str) -> tuple[int, str]:
    match = re.fullmatch(r"stair_(\d+)", name)
    if match is None:
        return (10**9, name)
    return (int(match.group(1)), name)


def _seq_root(display_root: Path, seq_name: str) -> Path:
    return display_root / seq_name / "gv"


def _qpos_path(qpos_root: Path, seq_name: str) -> Path:
    return qpos_root / "gmr" / seq_name / "unitree_g1" / f"{seq_name}_unitree_g1_qpos.npz"


def _scene_mesh_path(display_root: Path, seq_name: str) -> Path:
    return _seq_root(display_root, seq_name) / "scene_mesh_sqs/scene_mesh_sqs.obj"


def _has_final_outputs(display_root: Path, qpos_root: Path, seq_name: str) -> bool:
    root = _seq_root(display_root, seq_name)
    return (
        (root / "hmr/hps_track.npy").is_file()
        and _scene_mesh_path(display_root, seq_name).is_file()
        and _qpos_path(qpos_root, seq_name).is_file()
    )


def _discover_sequences(display_root: Path, qpos_root: Path, limit: int) -> list[str]:
    seqs: list[str] = []
    for qpos_file in sorted((qpos_root / "gmr").glob("*/unitree_g1/*_unitree_g1_qpos.npz")):
        seq_name = qpos_file.parts[-3]
        if _has_final_outputs(display_root, qpos_root, seq_name):
            seqs.append(seq_name)
    seqs = sorted(set(seqs), key=_natural_stair_key)
    if limit > 0:
        return seqs[:limit]
    return seqs


def _as_tensor(value, *, device: str) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _load_final_hmr_mesh(display_root: Path, seq_name: str, device: str) -> HmrTrack:
    import smplx

    hps_track = _seq_root(display_root, seq_name) / "hmr/hps_track.npy"
    if not hps_track.is_file():
        raise FileNotFoundError(f"Missing aligned HMR track: {hps_track}")

    payload = np.load(hps_track, allow_pickle=True).item()
    body_pose = _as_tensor(payload["body_pose"], device=device)
    global_orient = _as_tensor(payload["global_orient"], device=device)
    betas = _as_tensor(payload["betas"], device=device)
    transl = _as_tensor(payload["transl"], device=device)

    frame_count = min(
        int(body_pose.shape[0]),
        int(global_orient.shape[0]),
        int(betas.shape[0]),
        int(transl.shape[0]),
    )
    body_pose = body_pose[:frame_count, :23].reshape(frame_count, 23, 3, 3)
    global_orient = global_orient[:frame_count].reshape(frame_count, 1, 3, 3)
    betas = betas[:frame_count, :10].reshape(frame_count, 10)
    transl = transl[:frame_count].reshape(frame_count, 3)

    model = smplx.create(
        model_path=str(BODY_MODELS_ROOT),
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
        vertices = output.vertices.detach().cpu().numpy().astype(np.float32)
        joints = output.joints[:, :22, :].detach().cpu().numpy().astype(np.float32)

    faces = np.asarray(model.faces, dtype=np.int32)
    return HmrTrack(vertices=vertices, joints=joints, faces=faces)


def _load_scene_mesh(display_root: Path, seq_name: str) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    scene_obj = _scene_mesh_path(display_root, seq_name)
    if not scene_obj.is_file():
        raise FileNotFoundError(f"Missing final SQS scene mesh: {scene_obj}")
    loaded = trimesh.load(scene_obj, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    return np.asarray(loaded.vertices, dtype=np.float32), np.asarray(loaded.faces, dtype=np.int32)


def _make_checkerboard_mesh(
    xy_min: np.ndarray,
    xy_max: np.ndarray,
    *,
    tile_size: float,
    z: float = 0.0,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    tile = max(float(tile_size), 1.0e-3)
    lo = np.floor(np.asarray(xy_min, dtype=np.float32) / tile) * tile
    hi = np.ceil(np.asarray(xy_max, dtype=np.float32) / tile) * tile
    counts = np.maximum(np.ceil((hi - lo) / tile).astype(np.int32), 1)

    verts_by_color: list[list[list[float]]] = [[], []]
    faces_by_color: list[list[list[int]]] = [[], []]
    for ix in range(int(counts[0])):
        for iy in range(int(counts[1])):
            color_idx = (ix + iy) % 2
            x0 = float(lo[0] + ix * tile)
            y0 = float(lo[1] + iy * tile)
            x1 = float(x0 + tile)
            y1 = float(y0 + tile)
            base = len(verts_by_color[color_idx])
            verts_by_color[color_idx].extend(
                [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]]
            )
            faces_by_color[color_idx].extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])

    out = []
    for verts, faces in zip(verts_by_color, faces_by_color):
        out.append((np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)))
    return out[0], out[1]


def _load_qpos(qpos_root: Path, seq_name: str) -> tuple[np.ndarray, float, float, str]:
    qpos_file = _qpos_path(qpos_root, seq_name)
    if not qpos_file.is_file():
        raise FileNotFoundError(f"Missing scene-frame G1 qpos: {qpos_file}")
    with np.load(qpos_file, allow_pickle=True) as data:
        qpos = np.asarray(data["qpos"], dtype=np.float32)
        fps = float(data["fps"]) if "fps" in data.files else 30.0
        if "root_alignment_uniform_scale" in data.files:
            uniform_scale = float(np.asarray(data["root_alignment_uniform_scale"]).reshape(-1)[0])
            if uniform_scale <= 1.0e-8:
                raise ValueError(f"Invalid root_alignment_uniform_scale in {qpos_file}: {uniform_scale}")
            return qpos, fps, float(1.0 / uniform_scale), "inverse_root_alignment_uniform_scale"
        if "root_alignment_mode" in data.files:
            return qpos, fps, 1.0, str(np.asarray(data["root_alignment_mode"]).reshape(-1)[0])
    return qpos, fps, 1.0, "identity_direct_gmr"


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
    parser = argparse.ArgumentParser(
        description="Visualize final aligned stairs outputs: HMR mesh, scene-frame G1, and SQS scene mesh."
    )
    parser.add_argument("--port", type=int, default=9302)
    parser.add_argument("--limit", type=int, default=8, help="Maximum discovered stair_* sequences to load. Use 0 for all.")
    parser.add_argument("--sequences", nargs="+", default=None, help="Explicit sequence names, e.g. stair_0 stair_1.")
    parser.add_argument("--sequence", type=str, default=None, help="Initial sequence name.")
    parser.add_argument("--display-root", type=Path, default=DEFAULT_DISPLAY_ROOT)
    parser.add_argument("--qpos-root", type=Path, default=DEFAULT_QPOS_ROOT)
    parser.add_argument("--robot-mjcf", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--display-fps", type=float, default=3.0)
    parser.add_argument("--ground-tile-size", type=float, default=0.5)
    parser.add_argument("--ground-margin", type=float, default=1.0)
    parser.add_argument("--no-ground", action="store_true")
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    display_root = args.display_root.expanduser().resolve()
    qpos_root = args.qpos_root.expanduser().resolve()
    robot_mjcf = args.robot_mjcf.expanduser().resolve()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    if not robot_mjcf.is_file():
        raise FileNotFoundError(robot_mjcf)

    if args.sequences:
        sequences = list(dict.fromkeys(args.sequences))
        missing = [seq for seq in sequences if not _has_final_outputs(display_root, qpos_root, seq)]
        if missing:
            details = []
            for seq in missing:
                details.append(
                    f"{seq}: hmr={(_seq_root(display_root, seq) / 'hmr/hps_track.npy').is_file()} "
                    f"scene={_scene_mesh_path(display_root, seq).is_file()} "
                    f"qpos={_qpos_path(qpos_root, seq).is_file()}"
                )
            raise FileNotFoundError("Missing final outputs:\n" + "\n".join(details))
    else:
        sequences = _discover_sequences(display_root, qpos_root, int(args.limit))

    if args.sequence is not None:
        if args.sequence not in sequences:
            if not _has_final_outputs(display_root, qpos_root, args.sequence):
                raise FileNotFoundError(f"Requested sequence is not complete: {args.sequence}")
            sequences = [args.sequence] + sequences
        else:
            sequences = [args.sequence] + [seq for seq in sequences if seq != args.sequence]

    if not sequences:
        raise FileNotFoundError("No completed stair_* sequences with final HMR, scene mesh, and scene-frame G1 qpos were found.")

    server = viser.ViserServer(host="0.0.0.0", port=int(args.port), label="stairs_final_hmr_g1_scene")
    server.scene.set_up_direction("+z")
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    if args.share:
        server.request_share_url()

    helper = object.__new__(ViserHelper)
    helper.port = int(args.port)
    helper._server = server
    helper._ok = True
    helper._handles = {}

    state_lock = threading.RLock()
    cache: dict[str, StairsTrack] = {}
    current: StairsTrack | None = None
    frame_value = 0.0
    robot: MjcfQposViser | None = None
    hmr_mesh: DynamicMesh | None = None
    scene_handle = None
    ground_handles = []

    with server.gui.add_folder("Playback"):
        gui_sequence = server.gui.add_dropdown("Sequence", options=tuple(sequences), initial_value=sequences[0])
        gui_timestep = server.gui.add_slider("Timestep", min=0, max=1, step=1, initial_value=0)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_display_fps = server.gui.add_slider("Display FPS", min=1.0, max=15.0, step=0.5, initial_value=float(args.display_fps))

    with server.gui.add_folder("Layers"):
        gui_show_ground = server.gui.add_checkbox("z=0 checkerboard", not bool(args.no_ground))
        gui_show_scene = server.gui.add_checkbox("Scene SQS mesh", True)
        gui_show_hmr = server.gui.add_checkbox("HMR mesh", True)
        gui_show_g1 = server.gui.add_checkbox("G1", True)

    with server.gui.add_folder("Info"):
        server.gui.add_markdown(f"Sequences: `{', '.join(sequences)}`")
        server.gui.add_markdown(f"Display root: `{display_root}`")
        server.gui.add_markdown(f"Qpos root: `{qpos_root}`")

    def _load_track(seq_name: str) -> StairsTrack:
        if seq_name in cache:
            return cache[seq_name]
        print(f"[stairs-final-vis] loading {seq_name} ...", flush=True)
        hmr = _load_final_hmr_mesh(display_root, seq_name, args.device)
        qpos, fps, robot_position_scale, scale_source = _load_qpos(qpos_root, seq_name)
        scene_vertices, scene_faces = _load_scene_mesh(display_root, seq_name)

        frame_count = min(int(hmr.vertices.shape[0]), int(qpos.shape[0]))
        hmr = HmrTrack(vertices=hmr.vertices[:frame_count], joints=hmr.joints[:frame_count], faces=hmr.faces)
        qpos = qpos[:frame_count]

        track = StairsTrack(
            seq_name=seq_name,
            hmr=hmr,
            qpos=qpos,
            fps=fps,
            scene_vertices=scene_vertices,
            scene_faces=scene_faces,
            robot_position_scale=robot_position_scale,
            scale_source=scale_source,
        )
        cache[seq_name] = track
        print(
            f"[stairs-final-vis] {seq_name}: frames={frame_count} "
            f"scene_vertices={scene_vertices.shape[0]} robot_position_scale={robot_position_scale:.6f} "
            f"scale_source={scale_source}",
            flush=True,
        )
        return track

    def _remove_handles() -> None:
        nonlocal robot, hmr_mesh, scene_handle, ground_handles
        if hmr_mesh is not None:
            hmr_mesh.remove()
            hmr_mesh = None
        if robot is not None:
            for _, name in robot.geom_names:
                handle = robot.viser._handles.pop(name, None)
                if handle is not None:
                    handle.remove()
            robot = None
        if scene_handle is not None:
            scene_handle.remove()
            scene_handle = None
        for handle in ground_handles:
            handle.remove()
        ground_handles = []

    def _update_frame_locked(frame_idx: int) -> None:
        if current is None:
            return
        idx = int(frame_idx) % int(current.qpos.shape[0])
        if scene_handle is not None:
            scene_handle.visible = bool(gui_show_scene.value)
        for handle in ground_handles:
            handle.visible = bool(gui_show_ground.value)
        if robot is not None:
            robot.update(current.qpos[idx], world_scale=current.robot_position_scale, scale_root_only=True)
            _set_robot_visible(robot, bool(gui_show_g1.value))
        if hmr_mesh is not None:
            hmr_mesh.update(current.hmr.vertices[idx], bool(gui_show_hmr.value))

    def _activate_sequence(seq_name: str) -> None:
        nonlocal current, frame_value, robot, hmr_mesh, scene_handle, ground_handles
        track = _load_track(seq_name)
        with state_lock:
            current = track
            frame_value = 0.0
            _remove_handles()
            xy_rows = [
                track.scene_vertices[:, :2],
                track.hmr.joints.reshape(-1, 3)[:, :2],
                track.qpos[:, :2] * np.float32(track.robot_position_scale),
            ]
            xy = np.concatenate(xy_rows, axis=0)
            xy_min = np.nanmin(xy, axis=0) - float(args.ground_margin)
            xy_max = np.nanmax(xy, axis=0) + float(args.ground_margin)
            checker_a, checker_b = _make_checkerboard_mesh(
                xy_min,
                xy_max,
                tile_size=float(args.ground_tile_size),
                z=0.0,
            )
            ground_handles = [
                server.scene.add_mesh_simple(
                    "/ground/checker_dark",
                    vertices=checker_a[0],
                    faces=checker_a[1],
                    color=(0.30, 0.32, 0.34),
                    opacity=0.32,
                    side="double",
                    flat_shading=True,
                    cast_shadow=False,
                    receive_shadow=True,
                    visible=bool(gui_show_ground.value),
                ),
                server.scene.add_mesh_simple(
                    "/ground/checker_light",
                    vertices=checker_b[0],
                    faces=checker_b[1],
                    color=(0.74, 0.76, 0.78),
                    opacity=0.28,
                    side="double",
                    flat_shading=True,
                    cast_shadow=False,
                    receive_shadow=True,
                    visible=bool(gui_show_ground.value),
                ),
            ]
            scene_handle = server.scene.add_mesh_simple(
                "/scene/sqs_mesh",
                vertices=track.scene_vertices,
                faces=track.scene_faces,
                color=(0.58, 0.62, 0.68),
                opacity=0.52,
                side="double",
                flat_shading=True,
                cast_shadow=True,
                receive_shadow=True,
                visible=bool(gui_show_scene.value),
            )
            robot = MjcfQposViser(
                helper,
                robot_mjcf,
                prefix="/g1",
                mesh_scale=1.0,
            )
            _set_robot_visible(robot, bool(gui_show_g1.value))
            hmr_mesh = DynamicMesh(server, "/hmr/mesh", track.hmr.faces, (0.95, 0.45, 0.18))
            gui_timestep.max = max(0, int(track.qpos.shape[0]) - 1)
            gui_timestep.value = 0
            _update_frame_locked(0)

            root0 = track.hmr.joints[0, 0]
            for _, client in server.get_clients().items():
                client.camera.position = root0 + np.array([0.0, -3.0, 1.8], dtype=np.float32)
                client.camera.look_at = root0 + np.array([0.0, 0.0, 0.8], dtype=np.float32)

    @gui_sequence.on_update
    def _(_event) -> None:
        _activate_sequence(str(gui_sequence.value))

    for handle in (gui_show_ground, gui_show_scene, gui_show_hmr, gui_show_g1):
        @handle.on_update
        def _(_event) -> None:
            with state_lock:
                _update_frame_locked(int(round(gui_timestep.value)))

    @gui_timestep.on_update
    def _(_event) -> None:
        if not bool(gui_playing.value):
            with state_lock:
                _update_frame_locked(int(round(gui_timestep.value)))

    print(f"[stairs-final-vis] sequences: {', '.join(sequences)}", flush=True)
    print(f"[stairs-final-vis] display_root={display_root}", flush=True)
    print(f"[stairs-final-vis] qpos_root={qpos_root}", flush=True)
    print(f"[stairs-final-vis] loading initial sequence {sequences[0]} on {args.device} ...", flush=True)
    _activate_sequence(sequences[0])
    print(f"[stairs-final-vis] ready: http://localhost:{args.port}", flush=True)
    print("[stairs-final-vis] layers: HMR mesh, G1, Scene SQS mesh", flush=True)

    try:
        while True:
            display_fps = max(float(gui_display_fps.value), 1.0e-6)
            time.sleep(1.0 / display_fps)
            with state_lock:
                if current is None or not bool(gui_playing.value):
                    continue
                frame_value += 30.0 / display_fps
                frame_idx = int(frame_value) % int(current.qpos.shape[0])
                gui_timestep.value = frame_idx
                _update_frame_locked(frame_idx)
    except KeyboardInterrupt:
        print("\n[stairs-final-vis] stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
