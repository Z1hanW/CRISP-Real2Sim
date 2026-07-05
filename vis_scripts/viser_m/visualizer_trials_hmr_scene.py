#!/usr/bin/env python3
from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import viser


REPO_ROOT = Path(__file__).resolve().parents[2]
BODY_MODELS_ROOT = REPO_ROOT / "prep/HMR/inputs/checkpoints/body_models"
DEFAULT_DISPLAY_ROOT = REPO_ROOT / "results/output/post_scene_vggt_omega_trials_current"
DEFAULT_POINT_ROOT = REPO_ROOT / "results/output/scene_vggt_omega_trials_current"
DEFAULT_SEQUENCES = ("jinkun-1", "jinkun-2")


@dataclass
class MeshTrack:
    vertices: np.ndarray
    faces: np.ndarray


@dataclass
class StaticMesh:
    vertices: np.ndarray
    faces: np.ndarray
    color: tuple[float, float, float]


@dataclass
class SequenceData:
    seq: str
    hmr: MeshTrack
    scene_vertices: np.ndarray
    scene_faces: np.ndarray
    pieces: list[StaticMesh]
    points: np.ndarray | None
    colors: np.ndarray | None
    frame_count: int


class DynamicMesh:
    def __init__(
        self,
        server: viser.ViserServer,
        name: str,
        faces: np.ndarray,
        color: tuple[float, float, float],
        opacity: float,
    ) -> None:
        self.server = server
        self.name = name
        self.faces = faces
        self.color = color
        self.opacity = opacity
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
            opacity=self.opacity,
            side="double",
            flat_shading=False,
            visible=True,
        )

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize trial VGGT-Omega scene, pieces, pointcloud, and HMR.")
    parser.add_argument("--display-root", type=Path, default=DEFAULT_DISPLAY_ROOT)
    parser.add_argument("--point-root", type=Path, default=DEFAULT_POINT_ROOT)
    parser.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES))
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCES[0])
    parser.add_argument("--hmr-type", default="gv")
    parser.add_argument("--port", type=int, default=9336)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-points", type=int, default=350_000)
    parser.add_argument("--display-fps", type=float, default=6.0)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def _as_tensor(value, *, device: str) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _seq_root(root: Path, seq: str, hmr_type: str) -> Path:
    return root / seq / hmr_type


def _load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    if not path.is_file():
        raise FileNotFoundError(path)
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int32)


def _piece_color(index: int) -> tuple[float, float, float]:
    palette = np.asarray(
        [
            [0.95, 0.35, 0.18],
            [0.18, 0.68, 0.95],
            [0.20, 0.78, 0.45],
            [0.92, 0.72, 0.18],
            [0.65, 0.38, 0.90],
            [0.92, 0.45, 0.66],
            [0.35, 0.82, 0.80],
            [0.75, 0.78, 0.30],
        ],
        dtype=np.float32,
    )
    return tuple(float(v) for v in palette[index % len(palette)])


def _load_pieces(root: Path, seq: str, hmr_type: str) -> list[StaticMesh]:
    pieces_dir = _seq_root(root, seq, hmr_type) / "scene_mesh_sqs/pieces"
    pieces: list[StaticMesh] = []
    for idx, piece_path in enumerate(sorted(pieces_dir.glob("*.obj"))):
        vertices, faces = _load_mesh(piece_path)
        pieces.append(StaticMesh(vertices, faces, _piece_color(idx)))
    return pieces


def _load_hmr(root: Path, seq: str, hmr_type: str, device: str) -> MeshTrack:
    import smplx

    path = _seq_root(root, seq, hmr_type) / "hmr/hps_track.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = np.load(path, allow_pickle=True).item()

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
        out = model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
            pose2rot=False,
        )
    vertices = out.vertices.detach().cpu().numpy().astype(np.float32)
    faces = np.asarray(model.faces, dtype=np.int32)
    return MeshTrack(vertices=vertices, faces=faces)


def _load_point_transform(root: Path, seq: str, hmr_type: str) -> tuple[np.ndarray, np.ndarray]:
    sqs_npz = _seq_root(root, seq, hmr_type) / "scene_mesh_sqs/sqs_params.npz"
    with np.load(sqs_npz, allow_pickle=True) as data:
        rotation = np.asarray(data["world_rotation"], dtype=np.float32)
        translation = np.asarray(data["shared_translation"], dtype=np.float32).reshape(3)
    return rotation, translation


def _load_points(
    display_root: Path,
    point_root: Path,
    seq: str,
    hmr_type: str,
    max_points: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    path = _seq_root(point_root, seq, hmr_type) / "nksr_input/pointcloud_world.npz"
    if not path.is_file():
        return None, None
    with np.load(path, allow_pickle=True) as data:
        points = np.asarray(data["points"], dtype=np.float32)
        colors = np.asarray(data["colors"], dtype=np.uint8) if "colors" in data.files else None
    finite = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 1.0e-8)
    points = points[finite]
    if colors is not None and colors.shape[:1] == finite.shape:
        colors = colors[finite]
    rotation, translation = _load_point_transform(display_root, seq, hmr_type)
    points = points @ rotation.T + translation[None, :]
    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(abs(hash(seq)) % (2**32))
        keep = rng.choice(points.shape[0], size=max_points, replace=False)
        keep.sort()
        points = points[keep]
        if colors is not None:
            colors = colors[keep]
    return points.astype(np.float32, copy=False), colors


def _load_sequence(
    display_root: Path,
    point_root: Path,
    seq: str,
    hmr_type: str,
    device: str,
    max_points: int,
) -> SequenceData:
    hmr = _load_hmr(display_root, seq, hmr_type, device)
    scene_vertices, scene_faces = _load_mesh(_seq_root(display_root, seq, hmr_type) / "scene_mesh_sqs/scene_mesh_sqs.obj")
    pieces = _load_pieces(display_root, seq, hmr_type)
    points, colors = _load_points(display_root, point_root, seq, hmr_type, max_points)
    return SequenceData(
        seq=seq,
        hmr=hmr,
        scene_vertices=scene_vertices,
        scene_faces=scene_faces,
        pieces=pieces,
        points=points,
        colors=colors,
        frame_count=int(hmr.vertices.shape[0]),
    )


def _make_checkerboard(vertices: np.ndarray, tile_size: float = 0.5) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    xy = vertices[:, :2]
    lo = np.floor(np.nanpercentile(xy, 2.0, axis=0) / tile_size) * tile_size
    hi = np.ceil(np.nanpercentile(xy, 98.0, axis=0) / tile_size) * tile_size
    counts = np.maximum(np.ceil((hi - lo) / tile_size).astype(np.int32), 1)
    verts_by_color: list[list[list[float]]] = [[], []]
    faces_by_color: list[list[list[int]]] = [[], []]
    for ix in range(int(counts[0])):
        for iy in range(int(counts[1])):
            color_idx = (ix + iy) % 2
            x0 = float(lo[0] + ix * tile_size)
            y0 = float(lo[1] + iy * tile_size)
            x1 = float(x0 + tile_size)
            y1 = float(y0 + tile_size)
            base = len(verts_by_color[color_idx])
            verts_by_color[color_idx].extend([[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]])
            faces_by_color[color_idx].extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])
    return (
        (np.asarray(verts_by_color[0], dtype=np.float32), np.asarray(faces_by_color[0], dtype=np.int32)),
        (np.asarray(verts_by_color[1], dtype=np.float32), np.asarray(faces_by_color[1], dtype=np.int32)),
    )


def main() -> int:
    args = _parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    sequences = list(dict.fromkeys(str(seq) for seq in args.sequences))
    if args.sequence not in sequences:
        sequences.insert(0, str(args.sequence))
    display_root = args.display_root.expanduser().resolve()
    point_root = args.point_root.expanduser().resolve()

    server = viser.ViserServer(host=str(args.host), port=int(args.port), label="trials_hmr_scene")
    server.scene.set_up_direction("+z")
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    if args.share:
        server.request_share_url()

    cache: dict[str, SequenceData] = {}
    state_lock = threading.RLock()
    static_handles: list[object] = []
    hmr_mesh: DynamicMesh | None = None
    frame_value = 0.0

    with server.gui.add_folder("Dataset"):
        gui_seq = server.gui.add_dropdown("Sequence", options=tuple(sequences), initial_value=str(args.sequence))

    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider("Timestep", min=0, max=1, step=1, initial_value=0)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_fps = server.gui.add_slider("Display FPS", min=1.0, max=15.0, step=0.5, initial_value=float(args.display_fps))

    with server.gui.add_folder("Layers"):
        gui_show_ground = server.gui.add_checkbox("z=0 checkerboard", True)
        gui_show_scene = server.gui.add_checkbox("Scene merged", True)
        gui_show_pieces = server.gui.add_checkbox("Scene pieces", False)
        gui_show_hmr = server.gui.add_checkbox("HMR mesh", True)
        gui_show_points = server.gui.add_checkbox("Pointcloud", True)
        gui_point_size = server.gui.add_slider("Point size x1000", min=2, max=40, step=1, initial_value=8)

    with server.gui.add_folder("Info"):
        gui_info = server.gui.add_markdown("")
        server.gui.add_markdown(f"Display root: `{display_root}`")
        server.gui.add_markdown(f"Point root: `{point_root}`")

    def _load(seq: str) -> SequenceData:
        if seq not in cache:
            print(f"[trials-vis] loading {seq} on {args.device} ...", flush=True)
            cache[seq] = _load_sequence(display_root, point_root, seq, str(args.hmr_type), str(args.device), int(args.max_points))
            data = cache[seq]
            point_count = 0 if data.points is None else int(data.points.shape[0])
            print(
                f"[trials-vis] {seq}: frames={data.frame_count} pieces={len(data.pieces)} "
                f"points={point_count} scene_faces={data.scene_faces.shape[0]}",
                flush=True,
            )
        return cache[seq]

    def _clear_static() -> None:
        for handle in static_handles:
            remove = getattr(handle, "remove", None)
            if remove is not None:
                remove()
        static_handles.clear()

    def _render_static(data: SequenceData) -> None:
        _clear_static()
        all_vertices = [data.scene_vertices]
        if data.points is not None:
            all_vertices.append(data.points)
        if bool(gui_show_ground.value):
            (v0, f0), (v1, f1) = _make_checkerboard(np.concatenate(all_vertices, axis=0))
            static_handles.append(server.scene.add_mesh_simple("/ground/a", vertices=v0, faces=f0, color=(0.34, 0.34, 0.34), opacity=0.22, side="double"))
            static_handles.append(server.scene.add_mesh_simple("/ground/b", vertices=v1, faces=f1, color=(0.66, 0.66, 0.66), opacity=0.22, side="double"))
        static_handles.append(
            server.scene.add_mesh_simple(
                "/scene/merged",
                vertices=data.scene_vertices,
                faces=data.scene_faces,
                color=(0.20, 0.78, 0.95),
                opacity=0.42,
                side="double",
                flat_shading=True,
                visible=bool(gui_show_scene.value),
            )
        )
        for idx, piece in enumerate(data.pieces):
            static_handles.append(
                server.scene.add_mesh_simple(
                    f"/scene/pieces/{idx:03d}",
                    vertices=piece.vertices,
                    faces=piece.faces,
                    color=piece.color,
                    opacity=0.74,
                    side="double",
                    flat_shading=True,
                    visible=bool(gui_show_pieces.value),
                )
            )
        if bool(gui_show_points.value) and data.points is not None:
            colors = data.colors
            if colors is None or colors.shape[0] != data.points.shape[0]:
                colors = np.full((data.points.shape[0], 3), 170, dtype=np.uint8)
            static_handles.append(
                server.scene.add_point_cloud(
                    "/points",
                    points=data.points,
                    colors=colors,
                    point_size=float(gui_point_size.value) / 1000.0,
                    point_shape="rounded",
                    precision="float32",
                )
            )
        stacked = np.concatenate(all_vertices, axis=0)
        center = np.nanmean(stacked, axis=0)
        spread = float(np.linalg.norm(np.nanpercentile(stacked, 95.0, axis=0) - np.nanpercentile(stacked, 5.0, axis=0)))
        for _, client in server.get_clients().items():
            client.camera.look_at = center
            client.camera.position = center + np.array([0.0, -max(spread * 0.75, 3.0), max(spread * 0.40, 1.6)], dtype=np.float32)

    def _set_sequence(seq: str) -> None:
        nonlocal hmr_mesh, frame_value
        data = _load(seq)
        gui_timestep.max = max(0, data.frame_count - 1)
        gui_timestep.value = 0
        frame_value = 0.0
        if hmr_mesh is not None:
            hmr_mesh.remove()
        hmr_mesh = DynamicMesh(server, "/hmr/mesh", data.hmr.faces, (0.95, 0.45, 0.18), 0.72)
        _render_static(data)
        _update_frame(data, 0)
        gui_info.content = (
            f"`{seq}`  \n"
            f"frames: `{data.frame_count}` | pieces: `{len(data.pieces)}` | "
            f"points shown: `{0 if data.points is None else data.points.shape[0]}`"
        )

    def _update_frame(data: SequenceData, frame_idx: int) -> None:
        if hmr_mesh is None:
            return
        idx = int(frame_idx) % data.frame_count
        hmr_mesh.update(data.hmr.vertices[idx], bool(gui_show_hmr.value))

    @gui_seq.on_update
    def _(_event) -> None:
        with state_lock:
            _set_sequence(str(gui_seq.value))

    @gui_timestep.on_update
    def _(_event) -> None:
        if not bool(gui_playing.value):
            with state_lock:
                _update_frame(_load(str(gui_seq.value)), int(round(gui_timestep.value)))

    for handle in (gui_show_ground, gui_show_scene, gui_show_pieces, gui_show_points, gui_point_size):
        @handle.on_update
        def _(_event) -> None:
            with state_lock:
                _render_static(_load(str(gui_seq.value)))

    @gui_show_hmr.on_update
    def _(_event) -> None:
        with state_lock:
            _update_frame(_load(str(gui_seq.value)), int(round(gui_timestep.value)))

    with state_lock:
        _set_sequence(str(gui_seq.value))

    print(f"[trials-vis] ready: http://localhost:{args.port}", flush=True)
    try:
        while True:
            time.sleep(1.0 / max(float(gui_fps.value), 1.0e-6))
            with state_lock:
                if not bool(gui_playing.value):
                    continue
                data = _load(str(gui_seq.value))
                frame_value += 30.0 / max(float(gui_fps.value), 1.0e-6)
                idx = int(frame_value) % data.frame_count
                gui_timestep.value = idx
                _update_frame(data, idx)
    except KeyboardInterrupt:
        print("\n[trials-vis] stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
