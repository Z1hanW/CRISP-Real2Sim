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


@dataclass
class MeshTrack:
    vertices: np.ndarray
    faces: np.ndarray


@dataclass
class SourceTrack:
    label: str
    root: Path
    hmr: MeshTrack
    scene_vertices: np.ndarray
    scene_faces: np.ndarray
    points: np.ndarray | None
    colors: np.ndarray | None
    piece_count: int
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
        )
        self.handle.visible = True

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def _as_tensor(value, *, device: str) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _seq_root(root: Path, seq: str, hmr_type: str) -> Path:
    return root / seq / hmr_type


def _scene_mesh_path(root: Path, seq: str, hmr_type: str) -> Path:
    return _seq_root(root, seq, hmr_type) / "scene_mesh_sqs/scene_mesh_sqs.obj"


def _hps_path(root: Path, seq: str, hmr_type: str) -> Path:
    return _seq_root(root, seq, hmr_type) / "hmr/hps_track.npy"


def _load_hmr(root: Path, seq: str, hmr_type: str, device: str) -> MeshTrack:
    import smplx

    path = _hps_path(root, seq, hmr_type)
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


def _load_scene_mesh(root: Path, seq: str, hmr_type: str) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    path = _scene_mesh_path(root, seq, hmr_type)
    if not path.is_file():
        raise FileNotFoundError(path)
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int32)


def _load_points(root: Path, seq: str, hmr_type: str, max_points: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    path = _seq_root(root, seq, hmr_type) / "nksr_input/pointcloud_world.npz"
    if not path.is_file():
        return None, None
    with np.load(path, allow_pickle=True) as data:
        if "points" not in data.files:
            return None, None
        points = np.asarray(data["points"], dtype=np.float32)
        colors = np.asarray(data["colors"], dtype=np.uint8) if "colors" in data.files else None
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if colors is not None and colors.shape[:1] == finite.shape:
        colors = colors[finite]
    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(75)
        keep = rng.choice(points.shape[0], size=max_points, replace=False)
        keep.sort()
        points = points[keep]
        if colors is not None:
            colors = colors[keep]
    return points, colors


def _piece_count(root: Path, seq: str, hmr_type: str) -> int:
    path = _seq_root(root, seq, hmr_type) / "scene_mesh_sqs/sqs_params.npy"
    if not path.is_file():
        return 0
    arr = np.load(path, allow_pickle=True)
    return int(arr.shape[0]) if arr.ndim >= 1 else 0


def _load_source(
    label: str,
    root: Path,
    seq: str,
    hmr_type: str,
    device: str,
    max_points: int,
) -> SourceTrack:
    hmr = _load_hmr(root, seq, hmr_type, device)
    scene_vertices, scene_faces = _load_scene_mesh(root, seq, hmr_type)
    points, colors = _load_points(root, seq, hmr_type, max_points=max_points)
    return SourceTrack(
        label=label,
        root=root,
        hmr=hmr,
        scene_vertices=scene_vertices,
        scene_faces=scene_faces,
        points=points,
        colors=colors,
        piece_count=_piece_count(root, seq, hmr_type),
        frame_count=int(hmr.vertices.shape[0]),
    )


def _offset(label: str, layout: str, side_offset: float) -> np.ndarray:
    if layout != "Side by side":
        return np.zeros(3, dtype=np.float32)
    if label == "baseline":
        return np.array([-side_offset, 0.0, 0.0], dtype=np.float32)
    return np.array([side_offset, 0.0, 0.0], dtype=np.float32)


def _shift(vertices: np.ndarray, offset: np.ndarray) -> np.ndarray:
    if np.allclose(offset, 0.0):
        return vertices
    return (vertices + offset[None, :]).astype(np.float32, copy=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two stair_75 scale-only step-7 human-scene outputs.")
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--opt-root", type=Path, required=True)
    parser.add_argument("--sequence", default="stair_75")
    parser.add_argument("--hmr-type", default="gv")
    parser.add_argument("--port", type=int, default=9330)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--display-fps", type=float, default=4.0)
    parser.add_argument("--side-offset", type=float, default=5.0)
    parser.add_argument("--max-points", type=int, default=250_000)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    baseline_root = args.baseline_root.expanduser().resolve()
    opt_root = args.opt_root.expanduser().resolve()
    seq = str(args.sequence)
    hmr_type = str(args.hmr_type)

    print(f"[scale-compare] loading baseline from {baseline_root}", flush=True)
    baseline = _load_source("baseline", baseline_root, seq, hmr_type, str(args.device), int(args.max_points))
    print(f"[scale-compare] loading scale_opt from {opt_root}", flush=True)
    opt = _load_source("scale_opt", opt_root, seq, hmr_type, str(args.device), int(args.max_points))
    frame_count = min(baseline.frame_count, opt.frame_count)

    server = viser.ViserServer(host=str(args.host), port=int(args.port), label="stair75_scale_human_scene_compare")
    server.scene.set_up_direction("+z")
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    server.scene.add_grid("/grid", width=12.0, height=12.0, position=(0.0, 0.0, 0.0))
    if args.share:
        server.request_share_url()

    state_lock = threading.RLock()
    scene_handles: list[object] = []
    point_handles: list[object] = []
    hmr_handles: dict[str, DynamicMesh] = {}
    frame_value = 0.0

    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider("Timestep", min=0, max=max(0, frame_count - 1), step=1, initial_value=0)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_fps = server.gui.add_slider("Display FPS", min=1.0, max=15.0, step=0.5, initial_value=float(args.display_fps))

    with server.gui.add_folder("Layout"):
        gui_layout = server.gui.add_dropdown("Layout", options=("Side by side", "Overlay"), initial_value="Side by side")
        gui_side_offset = server.gui.add_slider("Side offset", min=1.0, max=10.0, step=0.25, initial_value=float(args.side_offset))

    with server.gui.add_folder("Layers"):
        gui_show_baseline_scene = server.gui.add_checkbox("Baseline scene", True)
        gui_show_opt_scene = server.gui.add_checkbox("Scale-opt scene", True)
        gui_show_baseline_hmr = server.gui.add_checkbox("Baseline HMR", True)
        gui_show_opt_hmr = server.gui.add_checkbox("Scale-opt HMR", True)
        gui_show_points = server.gui.add_checkbox("Pointcloud", False)
        gui_point_size = server.gui.add_slider("Point size x1000", min=2, max=40, step=1, initial_value=8)

    with server.gui.add_folder("Info"):
        server.gui.add_markdown(f"Sequence: `{seq}`")
        server.gui.add_markdown(f"Baseline pieces: `{baseline.piece_count}`")
        server.gui.add_markdown(f"Scale-opt pieces: `{opt.piece_count}`")
        server.gui.add_markdown(f"Baseline root: `{baseline_root}`")
        server.gui.add_markdown(f"Scale-opt root: `{opt_root}`")

    def _clear_static() -> None:
        for handle in scene_handles + point_handles:
            remove = getattr(handle, "remove", None)
            if remove is not None:
                remove()
        scene_handles.clear()
        point_handles.clear()

    def _render_static_locked() -> None:
        _clear_static()
        layout = str(gui_layout.value)
        side_offset = float(gui_side_offset.value)
        point_size = float(gui_point_size.value) / 1000.0
        all_vertices: list[np.ndarray] = []
        for source, scene_visible in (
            (baseline, bool(gui_show_baseline_scene.value)),
            (opt, bool(gui_show_opt_scene.value)),
        ):
            off = _offset(source.label, layout, side_offset)
            color = (0.95, 0.45, 0.18) if source.label == "baseline" else (0.15, 0.78, 0.95)
            verts = _shift(source.scene_vertices, off)
            all_vertices.append(verts)
            scene_handles.append(
                server.scene.add_mesh_simple(
                    f"/{source.label}/scene",
                    vertices=verts,
                    faces=source.scene_faces,
                    color=color,
                    opacity=0.42,
                    side="double",
                    flat_shading=True,
                    visible=scene_visible,
                )
            )
            if bool(gui_show_points.value) and source.points is not None:
                pts = _shift(source.points, off)
                all_vertices.append(pts)
                colors = source.colors
                if colors is None or colors.shape[0] != pts.shape[0]:
                    colors = np.full((pts.shape[0], 3), 170, dtype=np.uint8)
                point_handles.append(
                    server.scene.add_point_cloud(
                        f"/{source.label}/points",
                        points=pts,
                        colors=colors,
                        point_size=point_size,
                        point_shape="rounded",
                    )
                )

        if all_vertices:
            stacked = np.concatenate(all_vertices, axis=0)
            center = np.nanmean(stacked, axis=0)
            spread = float(
                np.linalg.norm(
                    np.nanpercentile(stacked, 95.0, axis=0)
                    - np.nanpercentile(stacked, 5.0, axis=0)
                )
            )
            for _, client in server.get_clients().items():
                client.camera.look_at = center
                client.camera.position = center + np.array(
                    [0.0, -max(spread * 0.70, 4.0), max(spread * 0.38, 2.0)],
                    dtype=np.float32,
                )

    def _update_frame_locked(frame_idx: int) -> None:
        idx = int(frame_idx) % frame_count
        layout = str(gui_layout.value)
        side_offset = float(gui_side_offset.value)
        sources = (
            (baseline, bool(gui_show_baseline_hmr.value), (0.98, 0.25, 0.16), 0.72),
            (opt, bool(gui_show_opt_hmr.value), (0.05, 0.48, 1.0), 0.72),
        )
        for source, visible, color, opacity in sources:
            if source.label not in hmr_handles:
                hmr_handles[source.label] = DynamicMesh(
                    server,
                    f"/{source.label}/hmr",
                    source.hmr.faces,
                    color,
                    opacity,
                )
            off = _offset(source.label, layout, side_offset)
            verts = _shift(source.hmr.vertices[idx], off)
            hmr_handles[source.label].update(verts, visible)

    def _redraw_locked() -> None:
        _render_static_locked()
        _update_frame_locked(int(round(gui_timestep.value)))

    for handle in (
        gui_layout,
        gui_side_offset,
        gui_show_baseline_scene,
        gui_show_opt_scene,
        gui_show_points,
        gui_point_size,
    ):
        @handle.on_update
        def _(_event) -> None:
            with state_lock:
                _redraw_locked()

    for handle in (gui_show_baseline_hmr, gui_show_opt_hmr):
        @handle.on_update
        def _(_event) -> None:
            with state_lock:
                _update_frame_locked(int(round(gui_timestep.value)))

    @gui_timestep.on_update
    def _(_event) -> None:
        if not bool(gui_playing.value):
            with state_lock:
                _update_frame_locked(int(round(gui_timestep.value)))

    with state_lock:
        _redraw_locked()

    print(
        f"[scale-compare] ready: http://localhost:{args.port} "
        f"(frames={frame_count}, baseline_pieces={baseline.piece_count}, opt_pieces={opt.piece_count})",
        flush=True,
    )

    try:
        while True:
            time.sleep(1.0 / max(float(gui_fps.value), 1.0e-6))
            with state_lock:
                if not bool(gui_playing.value):
                    continue
                frame_value += 30.0 / max(float(gui_fps.value), 1.0e-6)
                idx = int(frame_value) % frame_count
                gui_timestep.value = idx
                _update_frame_locked(idx)
    except KeyboardInterrupt:
        print("\n[scale-compare] stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
