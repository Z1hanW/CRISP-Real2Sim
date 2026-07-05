#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import viser


DEFAULT_COMPARE_ROOT = Path("/nfs/zzzihanw/crisp_stairs_nksr_compare")


@dataclass
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray


@dataclass
class ClipRecord:
    clip_id: str
    pointcloud_aligned: Path
    sqs_mesh: Path
    nksr_mesh: Path | None
    point_count: int
    source_point_count: int
    has_point_colors: bool
    terrain_scale: float
    z_offset: float


@dataclass
class ClipData:
    record: ClipRecord
    points: np.ndarray
    colors: np.ndarray
    sqs_mesh: MeshData
    nksr_mesh: MeshData | None


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
    parser = argparse.ArgumentParser(description="Visualize aligned pointcloud, SQS terrain, and NKSR mesh together.")
    parser.add_argument("--compare-root", type=Path, default=DEFAULT_COMPARE_ROOT)
    parser.add_argument("--port", type=int, default=9306)
    parser.add_argument("--initial-seq")
    parser.add_argument("--max-view-points", type=int, default=250_000)
    parser.add_argument("--point-size", type=float, default=0.012)
    parser.add_argument("--side-offset", type=float, default=4.0)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def _load_manifest(compare_root: Path) -> tuple[dict, list[ClipRecord]]:
    manifest_path = compare_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    records: list[ClipRecord] = []
    for entry in manifest.get("clips", []):
        nksr_raw = entry.get("nksr_mesh")
        records.append(
            ClipRecord(
                clip_id=str(entry["clip_id"]),
                pointcloud_aligned=Path(entry["pointcloud_aligned"]),
                sqs_mesh=Path(entry["sqs_mesh"]),
                nksr_mesh=Path(nksr_raw) if nksr_raw else None,
                point_count=int(entry.get("point_count", 0)),
                source_point_count=int(entry.get("source_point_count", 0)),
                has_point_colors=bool(entry.get("has_point_colors", False)),
                terrain_scale=float(entry.get("terrain_scale", 1.0)),
                z_offset=float(entry.get("z_offset", 0.0)),
            )
        )
    if not records:
        raise ValueError(f"No clips in {manifest_path}")
    return manifest, records


def _load_mesh(path: Path) -> MeshData:
    import trimesh

    mesh = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2:
        raise ValueError(f"Invalid mesh loaded from {path}")
    return MeshData(vertices=vertices, faces=faces)


def _load_points(path: Path, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        points = np.asarray(data["points"], dtype=np.float32)
        if "colors" in data.files:
            colors = np.asarray(data["colors"], dtype=np.uint8)
            if colors.shape != (points.shape[0], 3):
                raise ValueError(f"Bad colors shape in {path}: {colors.shape}; expected {(points.shape[0], 3)}")
        else:
            colors = np.empty((points.shape[0], 3), dtype=np.uint8)
            colors[:] = np.array([55, 170, 255], dtype=np.uint8)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]
    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(points.shape[0], size=max_points, replace=False)
        keep.sort()
        points = points[keep]
        colors = colors[keep]
    return points, colors


def _shift(vertices: np.ndarray, offset: np.ndarray) -> np.ndarray:
    if np.allclose(offset, 0.0):
        return vertices
    return (vertices + offset[None, :]).astype(np.float32, copy=False)


def _offset(name: str, layout: str, side_offset: float) -> np.ndarray:
    if layout != "Side by side":
        return np.zeros(3, dtype=np.float32)
    if name == "Pointcloud":
        return np.array([-side_offset, 0.0, 0.0], dtype=np.float32)
    if name == "NKSR":
        return np.array([side_offset, 0.0, 0.0], dtype=np.float32)
    return np.zeros(3, dtype=np.float32)


def main() -> int:
    args = _parse_args()
    compare_root = args.compare_root.expanduser().resolve()
    manifest, records = _load_manifest(compare_root)
    records_by_id = {record.clip_id: record for record in records}
    initial_seq = args.initial_seq or records[0].clip_id
    if initial_seq not in records_by_id:
        raise ValueError(f"initial seq {initial_seq!r} not found. Options: {sorted(records_by_id)}")

    server = viser.ViserServer(host="0.0.0.0", port=int(args.port))
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=10.0, height=10.0, position=(0.0, 0.0, 0.0))
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    if args.share:
        server.request_share_url()

    handles = SceneHandles()
    cache: dict[str, ClipData] = {}

    with server.gui.add_folder("Dataset"):
        gui_seq = server.gui.add_dropdown(
            "Sequence",
            options=[record.clip_id for record in records],
            initial_value=initial_seq,
        )
        gui_layout = server.gui.add_dropdown(
            "Layout",
            options=("Overlay", "Side by side"),
            initial_value="Overlay",
        )
        gui_side_offset = server.gui.add_slider(
            "Side offset",
            min=1.0,
            max=10.0,
            step=0.25,
            initial_value=float(args.side_offset),
        )

    with server.gui.add_folder("Layers"):
        gui_show_points = server.gui.add_checkbox("Pointcloud", initial_value=True)
        gui_show_sqs = server.gui.add_checkbox("SQS terrain", initial_value=True)
        gui_show_nksr = server.gui.add_checkbox("NKSR mesh", initial_value=True)
        gui_point_size = server.gui.add_slider(
            "Point size x1000",
            min=2,
            max=40,
            step=1,
            initial_value=max(2, int(round(float(args.point_size) * 1000))),
        )

    with server.gui.add_folder("Info"):
        gui_info = server.gui.add_markdown("")
        server.gui.add_markdown(f"Compare root: `{compare_root}`")
        server.gui.add_markdown("Transform: raw pointcloud -> SQS rotation/translation -> z offset -> terrain scale.")

    def _load_clip(clip_id: str) -> ClipData:
        if clip_id not in cache:
            record = records_by_id[clip_id]
            print(f"[nksr-compare] loading {clip_id}", flush=True)
            points, colors = _load_points(
                record.pointcloud_aligned,
                int(args.max_view_points),
                abs(hash(clip_id)) % (2**32),
            )
            sqs = _load_mesh(record.sqs_mesh)
            nksr = _load_mesh(record.nksr_mesh) if record.nksr_mesh is not None and record.nksr_mesh.is_file() else None
            cache[clip_id] = ClipData(record=record, points=points, colors=colors, sqs_mesh=sqs, nksr_mesh=nksr)
            print(
                f"[nksr-compare] {clip_id}: points={points.shape[0]}/{record.point_count} "
                f"colors={record.has_point_colors} sqs_faces={sqs.faces.shape[0]} "
                f"nksr_faces={0 if nksr is None else nksr.faces.shape[0]}",
                flush=True,
            )
        return cache[clip_id]

    def _render() -> None:
        handles.clear()
        clip = _load_clip(str(gui_seq.value))
        layout = str(gui_layout.value)
        side_offset = float(gui_side_offset.value)
        point_size = float(gui_point_size.value) / 1000.0

        all_vertices: list[np.ndarray] = []

        if bool(gui_show_points.value):
            pts = _shift(clip.points, _offset("Pointcloud", layout, side_offset))
            all_vertices.append(pts)
            handles.add(
                server.scene.add_point_cloud(
                    f"/{clip.record.clip_id}/pointcloud",
                    points=pts,
                    colors=clip.colors,
                    point_size=point_size,
                    point_shape="rounded",
                    precision="float32",
                )
            )

        if bool(gui_show_sqs.value):
            vertices = _shift(clip.sqs_mesh.vertices, _offset("SQS", layout, side_offset))
            all_vertices.append(vertices)
            handles.add(
                server.scene.add_mesh_simple(
                    f"/{clip.record.clip_id}/sqs_terrain",
                    vertices=vertices,
                    faces=clip.sqs_mesh.faces,
                    color=(0.95, 0.55, 0.12),
                    opacity=0.46,
                    side="double",
                    flat_shading=True,
                )
            )

        if bool(gui_show_nksr.value) and clip.nksr_mesh is not None:
            vertices = _shift(clip.nksr_mesh.vertices, _offset("NKSR", layout, side_offset))
            all_vertices.append(vertices)
            handles.add(
                server.scene.add_mesh_simple(
                    f"/{clip.record.clip_id}/nksr_mesh",
                    vertices=vertices,
                    faces=clip.nksr_mesh.faces,
                    color=(0.20, 0.85, 0.42),
                    opacity=0.42,
                    side="double",
                    flat_shading=True,
                )
            )

        nksr_faces = "missing" if clip.nksr_mesh is None else str(int(clip.nksr_mesh.faces.shape[0]))
        gui_info.content = (
            f"`{clip.record.clip_id}`  \n"
            f"points: `{clip.points.shape[0]}` shown / `{clip.record.point_count}` saved / "
            f"`{clip.record.source_point_count}` source  \n"
            f"point colors: `{'original RGB' if clip.record.has_point_colors else 'fallback'}`  \n"
            f"SQS faces: `{clip.sqs_mesh.faces.shape[0]}`  \n"
            f"NKSR faces: `{nksr_faces}`  \n"
            f"z_offset: `{clip.record.z_offset:.6f}`, terrain_scale: `{clip.record.terrain_scale:.6f}`"
        )

        if all_vertices:
            stacked = np.concatenate(all_vertices, axis=0)
            center = np.nanmean(stacked, axis=0)
            spread = float(
                np.linalg.norm(np.nanpercentile(stacked, 95.0, axis=0) - np.nanpercentile(stacked, 5.0, axis=0))
            )
            for _, client in server.get_clients().items():
                client.camera.look_at = center
                client.camera.position = center + np.array(
                    [0.0, -max(spread * 0.70, 3.0), max(spread * 0.42, 1.5)],
                    dtype=np.float32,
                )

    for gui in (gui_seq, gui_layout, gui_side_offset, gui_show_points, gui_show_sqs, gui_show_nksr, gui_point_size):
        @gui.on_update
        def _(_event) -> None:
            _render()

    _render()
    print(f"[nksr-compare] clips={len(records)} transform_order={manifest.get('transform_order')}", flush=True)
    print(f"[nksr-compare] ready: http://localhost:{args.port}", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[nksr-compare] stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
