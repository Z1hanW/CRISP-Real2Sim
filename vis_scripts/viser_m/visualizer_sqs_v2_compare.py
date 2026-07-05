#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import viser


DEFAULT_COMPARE_ZUP_ROOT = Path("/nfs/zzzihanw/crisp_stairs_sqs_v2_compare_zup_dedup")
DEFAULT_BASELINE_ROOT = DEFAULT_COMPARE_ZUP_ROOT / "baseline"
DEFAULT_V2_ROOT = DEFAULT_COMPARE_ZUP_ROOT / "v2"
DEFAULT_SEQUENCES = (
    "stair_45",
    "stair_3",
    "stair_48",
    "stair_50",
    "stair_51",
    "stair_53",
    "stair_54",
    "stair_61",
    "stair_69",
    "stair_75",
)


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
    mesh: MeshData | None
    pieces: list[PieceData]
    piece_count: int
    pointcloud: np.ndarray | None
    interval: int | None


class SceneHandles:
    def __init__(self) -> None:
        self._handles: list[object] = []

    def add(self, handle: object) -> None:
        self._handles.append(handle)

    def clear(self) -> None:
        for handle in self._handles:
            remove = getattr(handle, "remove", None)
            if remove is not None:
                remove()
        self._handles.clear()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline SQS primitives with v2 support-aware SQS primitives.")
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES))
    parser.add_argument("--initial-seq", default=DEFAULT_SEQUENCES[0])
    parser.add_argument("--hmr-type", default="gv")
    parser.add_argument("--port", type=int, default=9307)
    parser.add_argument("--max-points", type=int, default=300_000)
    parser.add_argument("--point-size", type=float, default=0.010)
    parser.add_argument("--side-offset", type=float, default=4.0)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def _load_mesh(path: Path) -> MeshData | None:
    if not path.is_file():
        return None
    import trimesh

    loaded = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    vertices = np.asarray(loaded.vertices, dtype=np.float32)
    faces = np.asarray(loaded.faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2:
        return None
    return MeshData(vertices=vertices, faces=faces)


def _piece_color(index: int, mode: str) -> tuple[float, float, float]:
    baseline = np.array([0.95, 0.48, 0.15], dtype=np.float32)
    v2 = np.array([0.20, 0.78, 0.95], dtype=np.float32)
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
    base = baseline if mode == "baseline" else v2
    color = 0.65 * palette[index % len(palette)] + 0.35 * base
    return tuple(float(v) for v in np.clip(color, 0.0, 1.0))


def _load_pieces(pieces_dir: Path, mode: str) -> list[PieceData]:
    if not pieces_dir.is_dir():
        return []
    pieces: list[PieceData] = []
    for idx, piece_path in enumerate(sorted(pieces_dir.glob("*.obj"))):
        mesh = _load_mesh(piece_path)
        if mesh is None:
            continue
        pieces.append(PieceData(mesh.vertices, mesh.faces, _piece_color(idx, mode)))
    return pieces


def _load_piece_count(sqs_dir: Path, fallback_count: int) -> int:
    if fallback_count > 0:
        return fallback_count
    params = sqs_dir / "sqs_params.npy"
    if params.is_file():
        arr = np.load(params, allow_pickle=True)
        if arr.ndim >= 1:
            return int(arr.shape[0])
    return fallback_count


def _load_pointcloud(seq_root: Path, max_points: int, seed: int) -> tuple[np.ndarray | None, np.ndarray | None, int | None]:
    path = seq_root / "nksr_input/pointcloud_world.npz"
    if not path.is_file():
        return None, None, None
    with np.load(path, allow_pickle=True) as data:
        if "points" not in data.files:
            return None, None, None
        points = np.asarray(data["points"], dtype=np.float32)
        colors = np.asarray(data["colors"], dtype=np.uint8) if "colors" in data.files else None
        interval = int(data["interval"]) if "interval" in data.files else None
    finite = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 1.0e-8)
    points = points[finite]
    if colors is not None and colors.shape[:1] == finite.shape:
        colors = colors[finite]
    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(points.shape[0], size=max_points, replace=False)
        keep.sort()
        points = points[keep]
        if colors is not None:
            colors = colors[keep]
    return points, colors, interval


def _load_source(root: Path, seq: str, hmr_type: str, mode: str, max_points: int) -> SourceData:
    seq_root = root / seq / hmr_type
    sqs_dir = seq_root / "scene_mesh_sqs"
    mesh = _load_mesh(sqs_dir / "scene_mesh_sqs.obj")
    pieces = _load_pieces(sqs_dir / "pieces", mode)
    pointcloud, colors, interval = _load_pointcloud(seq_root, max_points=max_points, seed=abs(hash((seq, mode))) % (2**32))
    if pointcloud is not None and colors is not None and colors.shape == pointcloud.shape:
        pointcloud = np.concatenate([pointcloud, colors.astype(np.float32)], axis=1)
    return SourceData(
        label=mode,
        root=root,
        mesh=mesh,
        pieces=pieces,
        piece_count=_load_piece_count(sqs_dir, len(pieces)),
        pointcloud=pointcloud,
        interval=interval,
    )


def _shift(vertices: np.ndarray, offset: np.ndarray) -> np.ndarray:
    if np.allclose(offset, 0.0):
        return vertices
    return (vertices + offset[None, :]).astype(np.float32, copy=False)


def _offset(label: str, layout: str, side_offset: float) -> np.ndarray:
    if layout != "Side by side":
        return np.zeros(3, dtype=np.float32)
    if label == "baseline":
        return np.array([-side_offset, 0.0, 0.0], dtype=np.float32)
    return np.array([side_offset, 0.0, 0.0], dtype=np.float32)


def main() -> int:
    args = _parse_args()
    sequences = [str(seq) for seq in args.sequences]
    if args.initial_seq not in sequences:
        args.initial_seq = sequences[0]

    baseline_root = args.baseline_root.expanduser().resolve()
    v2_root = args.v2_root.expanduser().resolve()

    server = viser.ViserServer(host="0.0.0.0", port=int(args.port))
    server.scene.set_up_direction("+z")
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    server.scene.add_grid("/grid", width=8.0, height=8.0, position=(0.0, 0.0, 0.0))
    if args.share:
        server.request_share_url()

    cache: dict[str, dict[str, SourceData]] = {}
    handles = SceneHandles()

    with server.gui.add_folder("Dataset"):
        gui_seq = server.gui.add_dropdown("Sequence", options=sequences, initial_value=args.initial_seq)
        gui_layout = server.gui.add_dropdown("Layout", options=("Overlay", "Side by side"), initial_value="Side by side")
        gui_side_offset = server.gui.add_slider("Side offset", min=1.0, max=10.0, step=0.25, initial_value=float(args.side_offset))

    with server.gui.add_folder("Layers"):
        gui_show_points = server.gui.add_checkbox("Pointcloud", initial_value=True)
        gui_show_baseline = server.gui.add_checkbox("Baseline merged", initial_value=True)
        gui_show_baseline_pieces = server.gui.add_checkbox("Baseline pieces", initial_value=False)
        gui_show_v2 = server.gui.add_checkbox("V2 merged", initial_value=True)
        gui_show_v2_pieces = server.gui.add_checkbox("V2 pieces", initial_value=False)
        gui_point_size = server.gui.add_slider(
            "Point size x1000",
            min=2,
            max=40,
            step=1,
            initial_value=max(2, int(round(float(args.point_size) * 1000))),
        )

    with server.gui.add_folder("Info"):
        gui_info = server.gui.add_markdown("")
        server.gui.add_markdown(f"Baseline: `{baseline_root}`")
        server.gui.add_markdown(f"V2: `{v2_root}`")
        server.gui.add_markdown("Inputs are pre-transformed z-up files; viewer only applies side-by-side offsets.")

    def _load(seq: str) -> dict[str, SourceData]:
        if seq not in cache:
            print(f"[sqs-v2-compare] loading {seq}", flush=True)
            cache[seq] = {
                "baseline": _load_source(baseline_root, seq, str(args.hmr_type), "baseline", int(args.max_points)),
                "v2": _load_source(v2_root, seq, str(args.hmr_type), "v2", int(args.max_points)),
            }
            for label, source in cache[seq].items():
                point_count = 0 if source.pointcloud is None else int(source.pointcloud.shape[0])
                faces = 0 if source.mesh is None else int(source.mesh.faces.shape[0])
                print(
                    f"[sqs-v2-compare] {seq} {label}: points={point_count} interval={source.interval} "
                    f"pieces={source.piece_count} faces={faces}",
                    flush=True,
                )
        return cache[seq]

    def _render() -> None:
        handles.clear()
        seq = str(gui_seq.value)
        sources = _load(seq)
        layout = str(gui_layout.value)
        side_offset = float(gui_side_offset.value)
        point_size = float(gui_point_size.value) / 1000.0

        all_vertices: list[np.ndarray] = []
        for label, source in sources.items():
            off = _offset(label, layout, side_offset)
            show_merged = bool(gui_show_baseline.value if label == "baseline" else gui_show_v2.value)
            show_pieces = bool(gui_show_baseline_pieces.value if label == "baseline" else gui_show_v2_pieces.value)
            mesh_color = (0.95, 0.48, 0.15) if label == "baseline" else (0.20, 0.78, 0.95)

            if bool(gui_show_points.value) and source.pointcloud is not None:
                points_raw = source.pointcloud[:, :3]
                points = _shift(points_raw, off)
                all_vertices.append(points)
                if source.pointcloud.shape[1] >= 6:
                    colors = np.asarray(source.pointcloud[:, 3:6], dtype=np.uint8)
                else:
                    colors = np.empty((points.shape[0], 3), dtype=np.uint8)
                    colors[:] = np.array([165, 165, 165], dtype=np.uint8)
                handles.add(
                    server.scene.add_point_cloud(
                        f"/{seq}/{label}/points",
                        points=points,
                        colors=colors,
                        point_size=point_size,
                        point_shape="rounded",
                        precision="float32",
                    )
                )

            if source.mesh is not None:
                vertices = _shift(source.mesh.vertices, off)
                all_vertices.append(vertices)
                handles.add(
                    server.scene.add_mesh_simple(
                        f"/{seq}/{label}/merged",
                        vertices=vertices,
                        faces=source.mesh.faces,
                        color=mesh_color,
                        opacity=0.44,
                        side="double",
                        flat_shading=True,
                        visible=show_merged,
                    )
                )

            for piece_idx, piece in enumerate(source.pieces):
                handles.add(
                    server.scene.add_mesh_simple(
                        f"/{seq}/{label}/pieces/{piece_idx:03d}",
                        vertices=_shift(piece.vertices, off),
                        faces=piece.faces,
                        color=piece.color,
                        opacity=0.72,
                        side="double",
                        flat_shading=True,
                        visible=show_pieces,
                    )
                )

        baseline = sources["baseline"]
        v2 = sources["v2"]
        gui_info.content = (
            f"`{seq}`  \n"
            f"baseline pieces: `{baseline.piece_count}` | v2 pieces: `{v2.piece_count}`  \n"
            f"baseline faces: `{0 if baseline.mesh is None else baseline.mesh.faces.shape[0]}` | "
            f"v2 faces: `{0 if v2.mesh is None else v2.mesh.faces.shape[0]}`"
        )

        if all_vertices:
            stacked = np.concatenate(all_vertices, axis=0)
            center = np.nanmean(stacked, axis=0)
            spread = float(np.linalg.norm(np.nanpercentile(stacked, 95.0, axis=0) - np.nanpercentile(stacked, 5.0, axis=0)))
            for _, client in server.get_clients().items():
                client.camera.look_at = center
                client.camera.position = center + np.array([0.0, -max(spread * 0.70, 3.0), max(spread * 0.36, 1.5)], dtype=np.float32)

    for gui in (
        gui_seq,
        gui_layout,
        gui_side_offset,
        gui_show_points,
        gui_show_baseline,
        gui_show_baseline_pieces,
        gui_show_v2,
        gui_show_v2_pieces,
        gui_point_size,
    ):
        @gui.on_update
        def _(_event) -> None:
            _render()

    _render()
    print(f"[sqs-v2-compare] ready: http://localhost:{args.port}", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[sqs-v2-compare] stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
