#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill SQS params from an existing scene_mesh_sqs export and clean stale piece files."
    )
    parser.add_argument("--scene-root", type=Path, required=True, help="Path like results/output/scene/<seq>/<hmr_type>.")
    parser.add_argument("--eps1", type=float, default=0.1, help="Default SQ epsilon 1 used for backfilled params.")
    parser.add_argument("--eps2", type=float, default=0.1, help="Default SQ epsilon 2 used for backfilled params.")
    return parser.parse_args()


def load_obj_vertices(path: Path) -> np.ndarray:
    verts: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith("v "):
                _, x, y, z, *_ = line.split()
                verts.append([float(x), float(y), float(z)])
    if not verts:
        raise ValueError(f"No vertices found in {path}")
    return np.asarray(verts, dtype=np.float32)


def normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-8:
        return fallback.astype(np.float32)
    return (vec / norm).astype(np.float32)


def estimate_piece_axis(vertices: np.ndarray) -> np.ndarray:
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(centered.shape[0], 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmin(eigvals))].astype(np.float32)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if float(np.dot(axis, world_up)) < 0.0:
        axis = -axis
    return normalize(axis, world_up)


def build_tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(ref, normal))) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    tangent_u = normalize(np.cross(normal, ref), np.array([1.0, 0.0, 0.0], dtype=np.float32))
    tangent_v = normalize(np.cross(normal, tangent_u), np.array([0.0, 1.0, 0.0], dtype=np.float32))
    return tangent_u, tangent_v


def parse_piece_order_from_urdf(urdf_path: Path, pieces_dir: Path) -> list[Path]:
    root = ET.parse(urdf_path).getroot()
    refs: list[str] = []
    seen: set[str] = set()
    for mesh_elem in root.findall(".//mesh"):
        filename = mesh_elem.attrib.get("filename")
        if not filename or filename in seen:
            continue
        refs.append(filename)
        seen.add(filename)
    if refs:
        return [pieces_dir / ref for ref in refs]
    return sorted(pieces_dir.glob("part_*.obj"))


def clean_stale_pieces(pieces_dir: Path, keep_paths: list[Path]) -> list[str]:
    keep_names = {path.name for path in keep_paths}
    removed: list[str] = []
    for path in sorted(pieces_dir.glob("part_*.obj")):
        if path.name not in keep_names:
            path.unlink()
            removed.append(path.name)
    return removed


def load_existing_params(sqs_root: Path, expected_count: int) -> np.ndarray | None:
    for path in (sqs_root / "sqs_params.npz", sqs_root / "sqs_params.npy"):
        if not path.exists():
            continue
        try:
            if path.suffix == ".npz":
                with np.load(path, allow_pickle=True) as data:
                    if "params" not in data.files:
                        continue
                    params = np.asarray(data["params"], dtype=np.float32)
            else:
                params = np.asarray(np.load(path, allow_pickle=True), dtype=np.float32)
        except Exception:
            continue
        if params.ndim == 2 and params.shape == (expected_count, 11):
            return params.copy()
    return None


def rotations_from_params(params: np.ndarray) -> np.ndarray:
    if params.size == 0:
        return np.zeros((0, 3, 3), dtype=np.float32)
    return Rotation.from_euler("ZYX", params[:, 5:8]).as_matrix().astype(np.float32)


def save_params_payload(sqs_root: Path, params_np: np.ndarray, piece_names: list[str], rot_np: np.ndarray) -> None:
    np.save(sqs_root / "sqs_params.npy", params_np)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        np.savez_compressed(
            tmp_path,
            params=params_np,
            piece_name_utf8=np.asarray(piece_names, dtype=f"<U{max((len(name) for name in piece_names), default=1)}"),
            piece_rot_p2w=rot_np,
        )
        shutil.copyfile(tmp_path, sqs_root / "sqs_params.npz")
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    args = parse_args()
    scene_root = args.scene_root.resolve()
    sqs_root = scene_root / "scene_mesh_sqs"
    pieces_dir = sqs_root / "pieces"
    urdf_path = sqs_root / "scene_mesh_sqs.urdf"
    if not pieces_dir.exists():
        raise FileNotFoundError(f"Missing pieces dir: {pieces_dir}")
    if not urdf_path.exists():
        raise FileNotFoundError(f"Missing scene URDF: {urdf_path}")

    piece_paths = parse_piece_order_from_urdf(urdf_path, pieces_dir)
    missing = [str(path) for path in piece_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"URDF references missing piece files: {missing}")

    removed = clean_stale_pieces(pieces_dir, piece_paths)

    existing_params = load_existing_params(sqs_root, len(piece_paths))
    if existing_params is not None:
        piece_names = [path.name for path in piece_paths]
        params_np = existing_params.astype(np.float32, copy=True)
        rot_np = rotations_from_params(params_np)
        save_params_payload(sqs_root, params_np, piece_names, rot_np)
        summary = {
            "scene_root": str(scene_root),
            "num_pieces": int(len(piece_paths)),
            "removed_stale_pieces": removed,
            "preserved_existing_params": True,
            "sqs_params_npy": str((sqs_root / "sqs_params.npy").resolve()),
            "sqs_params_npz": str((sqs_root / "sqs_params.npz").resolve()),
        }
        print(json.dumps(summary, indent=2))
        return

    params = []
    rot_mats = []
    piece_names = []
    for path in piece_paths:
        verts = load_obj_vertices(path)
        axis_z = estimate_piece_axis(verts)
        axis_x, axis_y = build_tangent_basis(axis_z)

        basis_rows = np.stack([axis_x, axis_y, axis_z], axis=0).astype(np.float32)
        local = verts @ basis_rows.T
        lo = local.min(axis=0)
        hi = local.max(axis=0)
        center_local = 0.5 * (lo + hi)
        half_axes = np.maximum(0.5 * (hi - lo), 1.0e-3).astype(np.float32)
        center_world = (center_local @ basis_rows).astype(np.float32)

        rot_world = np.stack([axis_x, axis_y, axis_z], axis=1).astype(np.float32)
        euler = Rotation.from_matrix(rot_world).as_euler("ZYX").astype(np.float32)
        params.append(
            [
                float(args.eps1),
                float(args.eps2),
                float(half_axes[0]),
                float(half_axes[1]),
                float(half_axes[2]),
                float(euler[0]),
                float(euler[1]),
                float(euler[2]),
                float(center_world[0]),
                float(center_world[1]),
                float(center_world[2]),
            ]
        )
        rot_mats.append(rot_world)
        piece_names.append(path.name)

    params_np = np.asarray(params, dtype=np.float32)
    rot_np = np.stack(rot_mats, axis=0).astype(np.float32) if rot_mats else np.zeros((0, 3, 3), dtype=np.float32)

    save_params_payload(sqs_root, params_np, piece_names, rot_np)

    summary = {
        "scene_root": str(scene_root),
        "num_pieces": int(len(piece_paths)),
        "removed_stale_pieces": removed,
        "preserved_existing_params": False,
        "sqs_params_npy": str((sqs_root / "sqs_params.npy").resolve()),
        "sqs_params_npz": str((sqs_root / "sqs_params.npz").resolve()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
