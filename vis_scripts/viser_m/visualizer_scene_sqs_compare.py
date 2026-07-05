#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import viser


REPO_ROOT = Path(__file__).resolve().parents[2]

SEQUENCES = {
    "49": "49_outdoor_big_stairs_down",
    "56": "56_outdoor_stairs_up_down",
    "78": "78_outdoor_stairs_up_down",
}

DEFAULT_VGGT_ROOT = REPO_ROOT / "results/output/scene_vggt_omega_consistent_camera_min1"
DEFAULT_OLD_ROOT = Path(
    "/data/far_offload/CRISP-Real2Sim-Obj/vis_scripts/results/output/post_scene"
)

SOURCES = {
    "VGGT-Omega interval7": {
        "root_arg": "vggt_root",
        "color": (40, 190, 255),
        "mesh_color": (0.15, 0.70, 0.95),
        "offset_index": -1,
    },
    "Old post_scene": {
        "root_arg": "old_root",
        "color": (255, 150, 55),
        "mesh_color": (0.95, 0.45, 0.12),
        "offset_index": 1,
    },
}


@dataclass
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray


@dataclass
class PieceData:
    vertices: np.ndarray
    faces: np.ndarray
    color: tuple[float, float, float]


@dataclass
class SourceData:
    label: str
    root: Path
    seq_root: Path
    pointcloud_points: np.ndarray | None
    mesh: MeshData | None
    pieces: list[PieceData]
    sqs_count: int | None
    interval: int | None


@dataclass
class SequenceData:
    seq_key: str
    seq_name: str
    sources: dict[str, SourceData]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two CRISP scene reconstructions and their SQS primitives."
    )
    parser.add_argument("--port", type=int, default=9301)
    parser.add_argument("--sequence", choices=sorted(SEQUENCES), default="56")
    parser.add_argument("--vggt-root", type=Path, default=DEFAULT_VGGT_ROOT)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument("--max-points", type=int, default=350_000)
    parser.add_argument("--point-size", type=float, default=0.012)
    parser.add_argument("--side-by-side-offset", type=float, default=4.0)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def _load_mesh(path: Path) -> MeshData:
    import trimesh

    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    vertices = np.asarray(loaded.vertices, dtype=np.float32)
    faces = np.asarray(loaded.faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2:
        raise ValueError(f"Invalid mesh loaded from {path}")
    return MeshData(vertices=vertices, faces=faces)


def _load_sqs_count(sqs_dir: Path) -> int | None:
    params_path = sqs_dir / "sqs_params.npy"
    if params_path.is_file():
        params = np.load(params_path, allow_pickle=True)
        if params.ndim >= 1:
            return int(params.shape[0])
    pieces_dir = sqs_dir / "pieces"
    if pieces_dir.is_dir():
        return len(list(pieces_dir.glob("*.obj")))
    return None


def _piece_color(index: int, base: tuple[float, float, float]) -> tuple[float, float, float]:
    palette = np.array(
        [
            [0.95, 0.35, 0.20],
            [0.20, 0.70, 0.95],
            [0.20, 0.80, 0.45],
            [0.92, 0.70, 0.20],
            [0.70, 0.35, 0.90],
            [0.90, 0.45, 0.65],
            [0.40, 0.85, 0.80],
            [0.80, 0.80, 0.35],
        ],
        dtype=np.float32,
    )
    base_np = np.asarray(base, dtype=np.float32)
    color = 0.70 * palette[index % len(palette)] + 0.30 * base_np
    return tuple(float(v) for v in np.clip(color, 0.0, 1.0))


def _load_pieces(pieces_dir: Path, base_color: tuple[float, float, float]) -> list[PieceData]:
    if not pieces_dir.is_dir():
        return []
    pieces: list[PieceData] = []
    for idx, piece_path in enumerate(sorted(pieces_dir.glob("*.obj"))):
        mesh = _load_mesh(piece_path)
        pieces.append(
            PieceData(
                vertices=mesh.vertices,
                faces=mesh.faces,
                color=_piece_color(idx, base_color),
            )
        )
    return pieces


def _sample_pointcloud(path: Path, max_points: int, color: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, int | None] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=True) as data:
        if "points" not in data.files:
            return None
        points = np.asarray(data["points"], dtype=np.float32)
        interval = int(data["interval"]) if "interval" in data.files else None

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if points.shape[0] > max_points:
        rng = np.random.default_rng(20260625)
        keep = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[keep]
    colors = np.empty((points.shape[0], 3), dtype=np.uint8)
    colors[:] = np.asarray(color, dtype=np.uint8)
    return points, colors, interval


def _source_seq_root(root: Path, seq_name: str) -> Path:
    return root / seq_name / "gv"


def _load_source(
    label: str,
    root: Path,
    seq_name: str,
    color: tuple[int, int, int],
    mesh_color: tuple[float, float, float],
    max_points: int,
) -> SourceData:
    seq_root = _source_seq_root(root, seq_name)
    sqs_dir = seq_root / "scene_mesh_sqs"
    pointcloud = _sample_pointcloud(
        seq_root / "nksr_input/pointcloud_world.npz",
        max_points=max_points,
        color=color,
    )
    mesh = None
    mesh_path = sqs_dir / "scene_mesh_sqs.obj"
    if mesh_path.is_file():
        mesh = _load_mesh(mesh_path)
    pieces = _load_pieces(sqs_dir / "pieces", mesh_color)
    sqs_count = _load_sqs_count(sqs_dir) if sqs_dir.exists() else None
    return SourceData(
        label=label,
        root=root,
        seq_root=seq_root,
        pointcloud_points=None if pointcloud is None else pointcloud[0],
        mesh=mesh,
        pieces=pieces,
        sqs_count=sqs_count,
        interval=None if pointcloud is None else pointcloud[2],
    )


def _load_sequence(args: argparse.Namespace, seq_key: str) -> SequenceData:
    seq_name = SEQUENCES[seq_key]
    sources: dict[str, SourceData] = {}
    for label, cfg in SOURCES.items():
        root = getattr(args, cfg["root_arg"])
        sources[label] = _load_source(
            label=label,
            root=root,
            seq_name=seq_name,
            color=cfg["color"],
            mesh_color=cfg["mesh_color"],
            max_points=max(1, int(args.max_points)),
        )
    return SequenceData(seq_key=seq_key, seq_name=seq_name, sources=sources)


def _offset_for_source(
    source_label: str,
    layout: str,
    side_by_side_offset: float,
) -> np.ndarray:
    if layout != "Side by side":
        return np.zeros(3, dtype=np.float32)
    cfg = SOURCES[source_label]
    return np.array([float(cfg["offset_index"]) * side_by_side_offset, 0.0, 0.0], dtype=np.float32)


def _shift_vertices(vertices: np.ndarray, offset: np.ndarray) -> np.ndarray:
    if np.allclose(offset, 0.0):
        return vertices
    return (vertices + offset[None, :]).astype(np.float32, copy=False)


class SceneHandles:
    def __init__(self) -> None:
        self.handles: list[object] = []

    def add(self, handle: object) -> None:
        self.handles.append(handle)

    def clear(self) -> None:
        for handle in self.handles:
            remove = getattr(handle, "remove", None)
            if remove is not None:
                remove()
        self.handles.clear()


def main() -> int:
    args = _parse_args()
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.set_up_direction("-z")
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    if args.share:
        server.request_share_url()

    cache: dict[str, SequenceData] = {}
    scene_handles = SceneHandles()

    with server.gui.add_folder("Compare"):
        gui_sequence = server.gui.add_dropdown(
            "Sequence",
            options=tuple(SEQUENCES.keys()),
            initial_value=args.sequence,
        )
        gui_layout = server.gui.add_dropdown(
            "Layout",
            options=("Overlay", "Side by side"),
            initial_value="Side by side",
        )
        gui_offset = server.gui.add_slider(
            "Side offset",
            min=1.0,
            max=10.0,
            step=0.25,
            initial_value=float(args.side_by_side_offset),
        )

    with server.gui.add_folder("Layers"):
        gui_vggt_points = server.gui.add_checkbox("VGGT points", True)
        gui_vggt_sqs = server.gui.add_checkbox("VGGT SQS merged", True)
        gui_vggt_pieces = server.gui.add_checkbox("VGGT SQS pieces", False)
        gui_old_points = server.gui.add_checkbox("Old points", True)
        gui_old_sqs = server.gui.add_checkbox("Old SQS merged", True)
        gui_old_pieces = server.gui.add_checkbox("Old SQS pieces", False)
        gui_point_size = server.gui.add_slider(
            "Point size x1000",
            min=2,
            max=40,
            step=1,
            initial_value=max(2, int(round(float(args.point_size) * 1000))),
        )

    with server.gui.add_folder("Info"):
        server.gui.add_markdown(f"VGGT root: `{args.vggt_root}`")
        server.gui.add_markdown(f"Old root: `{args.old_root}`")
        server.gui.add_markdown("Use Overlay for alignment; Side by side for shape comparison.")

    def _get_sequence(seq_key: str) -> SequenceData:
        if seq_key not in cache:
            print(f"[scene-sqs-compare] loading sequence {seq_key} ...", flush=True)
            cache[seq_key] = _load_sequence(args, seq_key)
            for source in cache[seq_key].sources.values():
                point_count = (
                    0
                    if source.pointcloud_points is None
                    else int(source.pointcloud_points.shape[0])
                )
                mesh_faces = 0 if source.mesh is None else int(source.mesh.faces.shape[0])
                print(
                    f"[scene-sqs-compare] {seq_key} {source.label}: "
                    f"points={point_count} interval={source.interval} "
                    f"sqs_count={source.sqs_count} mesh_faces={mesh_faces} "
                    f"pieces={len(source.pieces)} root={source.seq_root}",
                    flush=True,
                )
        return cache[seq_key]

    def _source_visibility(label: str) -> tuple[bool, bool, bool]:
        if label == "VGGT-Omega interval7":
            return (
                bool(gui_vggt_points.value),
                bool(gui_vggt_sqs.value),
                bool(gui_vggt_pieces.value),
            )
        return (
            bool(gui_old_points.value),
            bool(gui_old_sqs.value),
            bool(gui_old_pieces.value),
        )

    def _render() -> None:
        scene_handles.clear()
        seq = _get_sequence(str(gui_sequence.value))
        layout = str(gui_layout.value)
        point_size = float(gui_point_size.value) / 1000.0

        all_points: list[np.ndarray] = []
        for source_label, source in seq.sources.items():
            show_points, show_sqs, show_pieces = _source_visibility(source_label)
            offset = _offset_for_source(source_label, layout, float(gui_offset.value))
            source_color = SOURCES[source_label]["mesh_color"]
            source_rgb = SOURCES[source_label]["color"]

            if source.pointcloud_points is not None:
                all_points.append(_shift_vertices(source.pointcloud_points, offset))
                colors = np.empty((source.pointcloud_points.shape[0], 3), dtype=np.uint8)
                colors[:] = np.asarray(source_rgb, dtype=np.uint8)
                handle = server.scene.add_point_cloud(
                    name=f"/{seq.seq_key}/{source_label}/points",
                    points=_shift_vertices(source.pointcloud_points, offset),
                    colors=colors,
                    point_size=point_size,
                    point_shape="rounded",
                    precision="float32",
                    visible=show_points,
                )
                scene_handles.add(handle)

            if source.mesh is not None:
                label_suffix = (
                    "unknown"
                    if source.sqs_count is None
                    else str(int(source.sqs_count))
                )
                handle = server.scene.add_mesh_simple(
                    f"/{seq.seq_key}/{source_label}/sqs_merged_{label_suffix}",
                    vertices=_shift_vertices(source.mesh.vertices, offset),
                    faces=source.mesh.faces,
                    color=source_color,
                    opacity=0.45,
                    side="double",
                    flat_shading=True,
                    cast_shadow=True,
                    receive_shadow=True,
                    visible=show_sqs,
                )
                scene_handles.add(handle)

            for piece_idx, piece in enumerate(source.pieces):
                handle = server.scene.add_mesh_simple(
                    f"/{seq.seq_key}/{source_label}/pieces/part_{piece_idx:03d}",
                    vertices=_shift_vertices(piece.vertices, offset),
                    faces=piece.faces,
                    color=piece.color,
                    opacity=0.70,
                    side="double",
                    flat_shading=True,
                    cast_shadow=True,
                    receive_shadow=True,
                    visible=show_pieces,
                )
                scene_handles.add(handle)

        if all_points:
            stacked = np.concatenate(all_points, axis=0)
            center = np.nanmean(stacked, axis=0)
            spread = float(np.linalg.norm(np.nanpercentile(stacked, 95, axis=0) - np.nanpercentile(stacked, 5, axis=0)))
            for _, client in server.get_clients().items():
                client.camera.look_at = center
                client.camera.position = center + np.array([0.0, -max(spread * 0.65, 3.0), max(spread * 0.35, 1.5)], dtype=np.float32)

    for gui_handle in (
        gui_sequence,
        gui_layout,
        gui_offset,
        gui_vggt_points,
        gui_vggt_sqs,
        gui_vggt_pieces,
        gui_old_points,
        gui_old_sqs,
        gui_old_pieces,
        gui_point_size,
    ):
        @gui_handle.on_update
        def _(_event) -> None:
            _render()

    _render()
    print(f"[scene-sqs-compare] ready: http://localhost:{args.port}", flush=True)
    print(
        "[scene-sqs-compare] layers: VGGT points/SQS, Old points/SQS, merged/pieces, overlay/side-by-side",
        flush=True,
    )
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[scene-sqs-compare] stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
