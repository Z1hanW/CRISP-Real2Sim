#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_ROOT = Path("/nfs/zzzihanw/FAR_stairs_vggt_omega_live/scene_vggt_omega_consistent_camera_min1")
DEFAULT_V2_ROOT = Path("/nfs/zzzihanw/crisp_stairs_sqs_v2_compare")
DEFAULT_REFERENCE_POST_ROOT = REPO_ROOT / "results/output/post_scene_vggt_omega"
DEFAULT_OUTPUT_ROOT = Path("/nfs/zzzihanw/crisp_stairs_sqs_v2_compare_zup")
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a z-up baseline/v2 SQS comparison root using the same "
            "world_rotation/shared_translation convention as post_scene_vggt_omega."
        )
    )
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--reference-post-root", type=Path, default=DEFAULT_REFERENCE_POST_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--hmr-type", default="gv")
    parser.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES))
    parser.add_argument("--max-points", type=int, default=350_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_reference_transform(reference_post_root: Path, seq: str, hmr_type: str) -> tuple[np.ndarray, np.ndarray]:
    params_path = reference_post_root / seq / hmr_type / "scene_mesh_sqs/sqs_params.npz"
    if not params_path.is_file():
        raise FileNotFoundError(params_path)
    with np.load(params_path, allow_pickle=True) as data:
        rotation = np.asarray(data["world_rotation"], dtype=np.float32)
        translation = np.asarray(data["shared_translation"], dtype=np.float32).reshape(3)
    if rotation.shape != (3, 3):
        raise ValueError(f"Bad world_rotation in {params_path}: {rotation.shape}")
    return rotation, translation


def _transform_vertices(vertices: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (np.asarray(vertices, dtype=np.float32) @ rotation.T + translation[None, :]).astype(np.float32, copy=False)


def _normalise(normals: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.divide(normals, np.maximum(norm, 1.0e-8)).astype(np.float32, copy=False)


def _savez_compressed_atomic(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.stem}.", suffix=".npz")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        np.savez_compressed(str(tmp), **payload)
        shutil.copyfile(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _load_mesh(path: Path):
    import trimesh

    mesh = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def _transform_mesh_file(src: Path, dst: Path, rotation: np.ndarray, translation: np.ndarray) -> dict[str, Any]:
    mesh = _load_mesh(src)
    mesh.vertices = _transform_vertices(np.asarray(mesh.vertices), rotation, translation)
    dst.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(dst))
    z = np.asarray(mesh.vertices)[:, 2]
    return {
        "path": str(dst),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "min_z": float(z.min()) if z.size else None,
        "max_z": float(z.max()) if z.size else None,
    }


def _rotate_sqs_params(params: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    params = np.asarray(params, dtype=np.float32)
    if params.ndim != 2 or params.shape[1] < 11:
        raise ValueError(f"Expected SQ params with shape (N, 11+), got {params.shape}")
    rot_old = Rotation.from_euler("ZYX", params[:, 5:8]).as_matrix().astype(np.float32)
    rot_new = (rotation[None, :, :] @ rot_old).astype(np.float32)
    euler_new = Rotation.from_matrix(rot_new).as_euler("ZYX").astype(np.float32)
    transl_new = params[:, 8:11].astype(np.float32) @ rotation.T + translation[None, :]
    out = params.copy()
    out[:, 5:8] = euler_new
    out[:, 8:11] = transl_new.astype(np.float32)
    return out


def _transform_params(src_dir: Path, dst_dir: Path, rotation: np.ndarray, translation: np.ndarray) -> int:
    src_npz = src_dir / "sqs_params.npz"
    src_npy = src_dir / "sqs_params.npy"
    if src_npz.is_file():
        with np.load(src_npz, allow_pickle=True) as data:
            params = np.asarray(data["params"], dtype=np.float32)
            extras = {key: np.asarray(data[key]) for key in data.files if key not in {"params", "world_rotation", "shared_translation"}}
    elif src_npy.is_file():
        params = np.asarray(np.load(src_npy, allow_pickle=True), dtype=np.float32)
        extras = {}
    else:
        raise FileNotFoundError(f"Missing sqs_params under {src_dir}")

    params_rot = _rotate_sqs_params(params, rotation, translation)
    dst_dir.mkdir(parents=True, exist_ok=True)
    np.save(dst_dir / "sqs_params.npy", params_rot.astype(np.float32))
    _savez_compressed_atomic(
        dst_dir / "sqs_params.npz",
        params=params_rot.astype(np.float32),
        world_rotation=rotation.astype(np.float32),
        shared_translation=translation.astype(np.float32),
        **extras,
    )
    return int(params_rot.shape[0])


def _transform_pointcloud(src: Path, dst: Path, rotation: np.ndarray, translation: np.ndarray, max_points: int, seed: int) -> dict[str, Any]:
    if not src.is_file():
        return {"path": str(dst), "status": "missing"}
    with np.load(src, allow_pickle=True) as data:
        points = np.asarray(data["points"], dtype=np.float32)
        normals = np.asarray(data["normals"], dtype=np.float32) if "normals" in data.files else None
        extras = {key: np.asarray(data[key]) for key in data.files if key not in {"points", "normals"}}

    original_count = int(points.shape[0])
    finite = np.isfinite(points).all(axis=1)
    if normals is not None:
        finite &= np.isfinite(normals).all(axis=1) & (np.linalg.norm(normals, axis=1) > 1.0e-8)
    points = points[finite]
    if normals is not None:
        normals = normals[finite]
    for key, value in list(extras.items()):
        if value.shape[:1] == finite.shape:
            extras[key] = value[finite]

    filtered_count = int(points.shape[0])
    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(points.shape[0], size=max_points, replace=False)
        keep.sort()
        points = points[keep]
        if normals is not None:
            normals = normals[keep]
        for key, value in list(extras.items()):
            if value.shape[:1] == (filtered_count,):
                extras[key] = value[keep]

    payload: dict[str, Any] = {
        "points": _transform_vertices(points, rotation, translation),
        "source_point_count": np.asarray(original_count, dtype=np.int64),
        "filtered_point_count": np.asarray(filtered_count, dtype=np.int64),
        "world_rotation": rotation.astype(np.float32),
        "shared_translation": translation.astype(np.float32),
        "source_pointcloud_npz": np.asarray(str(src.resolve())),
    }
    if normals is not None:
        payload["normals"] = _normalise(normals @ rotation.T)
    payload.update(extras)

    dst.parent.mkdir(parents=True, exist_ok=True)
    _savez_compressed_atomic(dst, **payload)
    return {
        "path": str(dst),
        "status": "written",
        "points": int(payload["points"].shape[0]),
        "source_points": original_count,
    }


def _transform_source(
    *,
    source_label: str,
    source_root: Path,
    output_root: Path,
    seq: str,
    hmr_type: str,
    rotation: np.ndarray,
    translation: np.ndarray,
    max_points: int,
    force: bool,
) -> dict[str, Any]:
    src_seq_root = source_root / seq / hmr_type
    dst_seq_root = output_root / source_label / seq / hmr_type
    sqs_src = src_seq_root / "scene_mesh_sqs"
    sqs_dst = dst_seq_root / "scene_mesh_sqs"
    if not sqs_src.is_dir():
        raise FileNotFoundError(sqs_src)
    if dst_seq_root.exists() and force:
        shutil.rmtree(dst_seq_root, ignore_errors=True)
    if dst_seq_root.exists() and (sqs_dst / "scene_mesh_sqs.obj").is_file() and not force:
        return {"source": source_label, "seq": seq, "status": "exists", "output": str(dst_seq_root)}

    dst_seq_root.mkdir(parents=True, exist_ok=True)
    np.save(dst_seq_root / "world_rotation.npy", rotation.astype(np.float32))
    np.savetxt(dst_seq_root / "world_rotation.txt", rotation, fmt="%.8f")
    np.savetxt(dst_seq_root / "shared_translation.txt", translation.reshape(1, 3), fmt="%.8f")

    merged_meta = _transform_mesh_file(sqs_src / "scene_mesh_sqs.obj", sqs_dst / "scene_mesh_sqs.obj", rotation, translation)
    urdf_src = sqs_src / "scene_mesh_sqs.urdf"
    if urdf_src.is_file():
        shutil.copy2(urdf_src, sqs_dst / urdf_src.name)

    piece_count = 0
    pieces_src = sqs_src / "pieces"
    pieces_dst = sqs_dst / "pieces"
    if pieces_src.is_dir():
        if pieces_dst.exists():
            shutil.rmtree(pieces_dst, ignore_errors=True)
        pieces_dst.mkdir(parents=True, exist_ok=True)
        for piece_path in sorted(pieces_src.glob("*.obj")):
            _transform_mesh_file(piece_path, pieces_dst / piece_path.name, rotation, translation)
            piece_count += 1

    sq_count = _transform_params(sqs_src, sqs_dst, rotation, translation)
    pc_meta = _transform_pointcloud(
        src_seq_root / "nksr_input/pointcloud_world.npz",
        dst_seq_root / "nksr_input/pointcloud_world.npz",
        rotation,
        translation,
        max_points=max_points,
        seed=abs(hash((source_label, seq))) % (2**32),
    )

    return {
        "source": source_label,
        "seq": seq,
        "status": "written",
        "output": str(dst_seq_root),
        "merged": merged_meta,
        "pieces": piece_count,
        "sqs_count": sq_count,
        "pointcloud": pc_meta,
    }


def main() -> None:
    args = _parse_args()
    args.baseline_root = args.baseline_root.expanduser().resolve()
    args.v2_root = args.v2_root.expanduser().resolve()
    args.reference_post_root = args.reference_post_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()

    records: list[dict[str, Any]] = []
    for index, seq in enumerate([str(item) for item in args.sequences], start=1):
        rotation, translation = _load_reference_transform(args.reference_post_root, seq, str(args.hmr_type))
        for source_label, source_root in (("baseline", args.baseline_root), ("v2", args.v2_root)):
            record = _transform_source(
                source_label=source_label,
                source_root=source_root,
                output_root=args.output_root,
                seq=seq,
                hmr_type=str(args.hmr_type),
                rotation=rotation,
                translation=translation,
                max_points=int(args.max_points),
                force=bool(args.force),
            )
            records.append(record)
            pc = record.get("pointcloud", {})
            print(
                f"[{index:02d}/{len(args.sequences):02d}] {seq} {source_label}: "
                f"{record['status']} pieces={record.get('pieces', 'n/a')} sqs={record.get('sqs_count', 'n/a')} "
                f"points={pc.get('points', 'n/a')}/{pc.get('source_points', 'n/a')}",
                flush=True,
            )

    manifest = {
        "schema_version": 1,
        "format": "sqs_v2_compare_zup",
        "output_root": str(args.output_root),
        "baseline_root": str(args.baseline_root),
        "v2_root": str(args.v2_root),
        "reference_post_root": str(args.reference_post_root),
        "hmr_type": str(args.hmr_type),
        "max_points": int(args.max_points),
        "transform_order": [
            "vertices/points = vertices/points @ world_rotation.T + shared_translation",
            "normals = normalize(normals @ world_rotation.T)",
            "world_rotation/shared_translation are copied from post_scene_vggt_omega/<seq>/<hmr>/scene_mesh_sqs/sqs_params.npz",
        ],
        "records": records,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] wrote z-up compare root: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
