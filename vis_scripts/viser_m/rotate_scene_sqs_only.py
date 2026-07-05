#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rotate scene_mesh_sqs geometry and sqs params into post_scene using cached world_rotation."
    )
    parser.add_argument("--sequence-name", type=str, required=True)
    parser.add_argument("--hmr-type", type=str, default="gv")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Root containing <sequence>/<hmr-type>/scene_mesh_sqs. Defaults to results/output/scene.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root containing <sequence>/<hmr-type>/world_rotation.npy and destination scene_mesh_sqs. Defaults to results/output/post_scene.",
    )
    return parser.parse_args()


def load_obj(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    vertices = []
    for line in lines:
        if line.startswith("v "):
            _, x, y, z, *_ = line.split()
            vertices.append([float(x), float(y), float(z)])
    if not vertices:
        raise ValueError(f"No vertices found in {path}")
    return lines, np.asarray(vertices, dtype=np.float32)


def write_obj(path: Path, lines: list[str], vertices: np.ndarray) -> None:
    out_lines = []
    vert_idx = 0
    for line in lines:
        if line.startswith("v "):
            v = vertices[vert_idx]
            out_lines.append(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}")
            vert_idx += 1
        else:
            out_lines.append(line)
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def rotate_vertices_rowvec(vertices: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return vertices @ rotation.T + translation[None, :]


def rotate_sqs_params_rowvec(params: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    rot_old = Rotation.from_euler("ZYX", params[:, 5:8]).as_matrix().astype(np.float32)
    rot_new = (rotation[None, :, :] @ rot_old).astype(np.float32)
    euler_new = Rotation.from_matrix(rot_new).as_euler("ZYX").astype(np.float32)
    transl_new = params[:, 8:11].astype(np.float32) @ rotation.T + translation[None, :]
    out = params.copy()
    out[:, 5:8] = euler_new
    out[:, 8:11] = transl_new
    return out


def parse_piece_refs(urdf_path: Path) -> list[str]:
    root = ET.parse(urdf_path).getroot()
    refs = []
    seen = set()
    for mesh_elem in root.findall(".//mesh"):
        filename = mesh_elem.attrib.get("filename")
        if not filename or filename in seen:
            continue
        refs.append(filename)
        seen.add(filename)
    return refs


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    seq = args.sequence_name
    hmr_type = args.hmr_type

    input_root = args.input_root or (repo_root / "results" / "output" / "scene")
    output_root = args.output_root or (repo_root / "results" / "output" / "post_scene")
    if not input_root.is_absolute():
        input_root = repo_root / input_root
    if not output_root.is_absolute():
        output_root = repo_root / output_root

    scene_root = input_root / seq / hmr_type
    post_root = output_root / seq / hmr_type
    sqs_src_root = scene_root / "scene_mesh_sqs"
    sqs_dst_root = post_root / "scene_mesh_sqs"
    if not sqs_src_root.exists():
        raise FileNotFoundError(f"Missing source scene_mesh_sqs: {sqs_src_root}")

    world_rotation_path = post_root / "world_rotation.npy"
    if not world_rotation_path.exists():
        raise FileNotFoundError(f"Missing cached world_rotation: {world_rotation_path}")
    rotation = np.load(world_rotation_path).astype(np.float32)

    merged_src = sqs_src_root / "scene_mesh_sqs.obj"
    urdf_src = sqs_src_root / "scene_mesh_sqs.urdf"
    pieces_src = sqs_src_root / "pieces"
    params_src = sqs_src_root / "sqs_params.npz"
    if not params_src.exists():
        candidate = sqs_src_root / "sqs_params.npy"
        params_src = candidate if candidate.exists() else None
    if params_src is None:
        raise FileNotFoundError(f"Missing sqs params under {sqs_src_root}")

    merged_lines, merged_vertices = load_obj(merged_src)
    rotated_merged = merged_vertices @ rotation.T
    shared_translation = np.array([0.0, 0.0, -float(rotated_merged[:, 2].min())], dtype=np.float32)
    rotated_merged = rotated_merged + shared_translation[None, :]

    sqs_dst_root.mkdir(parents=True, exist_ok=True)
    write_obj(sqs_dst_root / "scene_mesh_sqs.obj", merged_lines, rotated_merged)
    if urdf_src.exists():
        shutil.copy2(urdf_src, sqs_dst_root / urdf_src.name)

    piece_refs = parse_piece_refs(urdf_src) if urdf_src.exists() else sorted(p.name for p in pieces_src.glob("part_*.obj"))
    pieces_dst = sqs_dst_root / "pieces"
    if pieces_dst.exists():
        shutil.rmtree(pieces_dst)
    pieces_dst.mkdir(parents=True, exist_ok=True)
    rotated_piece_count = 0
    for ref in piece_refs:
        ref_path = Path(ref)
        candidates = []
        if ref_path.is_absolute():
            candidates.append(ref_path)
        else:
            candidates.extend([sqs_src_root / ref_path, pieces_src / ref_path, pieces_src / ref_path.name])
        src = next((path for path in candidates if path.exists()), None)
        if src is None:
            continue
        lines, vertices = load_obj(src)
        rotated = rotate_vertices_rowvec(vertices, rotation, shared_translation)
        write_obj(pieces_dst / ref_path.name, lines, rotated)
        rotated_piece_count += 1

    if params_src.suffix == ".npz":
        with np.load(params_src, allow_pickle=True) as payload:
            params = np.asarray(payload["params"], dtype=np.float32)
            extras = {k: np.asarray(payload[k]) for k in payload.files if k != "params"}
    else:
        params = np.asarray(np.load(params_src, allow_pickle=True), dtype=np.float32)
        extras = {}
    params_rot = rotate_sqs_params_rowvec(params, rotation, shared_translation)
    np.save(sqs_dst_root / "sqs_params.npy", params_rot)
    np.savez_compressed(
        sqs_dst_root / "sqs_params.npz",
        params=params_rot,
        world_rotation=rotation,
        shared_translation=shared_translation,
        **extras,
    )

    summary = {
        "sequence_name": seq,
        "hmr_type": hmr_type,
        "rotated_piece_count": rotated_piece_count,
        "shared_translation": shared_translation.tolist(),
        "world_rotation": str(world_rotation_path),
        "output_root": str(post_root),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
