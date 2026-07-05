#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUBSET_ROOT = Path("/nfs/zzzihanw/crisp_stairs")
DEFAULT_OUTPUT_ROOT = Path("/nfs/zzzihanw/crisp_stairs_nksr_compare")
DEFAULT_RAW_ROOT = REPO_ROOT / "results/output/scene_vggt_omega_consistent_camera_min1"
DEFAULT_POST_ROOT = REPO_ROOT / "results/output/post_scene_vggt_omega"
DEFAULT_SCENE_NPZ_ROOT = REPO_ROOT / "results/output/scene"
DEFAULT_TERRAIN_ROOT = Path(
    "/home/ubuntu/FAR/holosoma_gt/src/holosoma_retargeting/holosoma_retargeting/demo_data/crisp_terrain"
)
DEFAULT_NKSR_PYTHON = Path("/home/ubuntu/miniconda3/envs/crisp_nksr/bin/python")


def _natural_key(text: str) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare final-space point clouds and NKSR meshes for the CRISP stairs subset. "
            "The transform is raw VGGT pointcloud -> SQS world_rotation/shared_translation -> "
            "Holosoma terrain z offset -> Holosoma terrain scale."
        )
    )
    parser.add_argument("--subset-root", type=Path, default=DEFAULT_SUBSET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--post-root", type=Path, default=DEFAULT_POST_ROOT)
    parser.add_argument("--scene-npz-root", type=Path, default=DEFAULT_SCENE_NPZ_ROOT)
    parser.add_argument("--terrain-root", type=Path, default=DEFAULT_TERRAIN_ROOT)
    parser.add_argument("--hmr-type", default="gv")
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--max-points", type=int, default=350_000)
    parser.add_argument("--skip-colors", action="store_true")
    parser.add_argument("--depth-conf-quantile", type=float, default=0.2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--write-ply", action="store_true")
    parser.add_argument("--run-nksr", action="store_true")
    parser.add_argument(
        "--nksr-python",
        type=Path,
        default=DEFAULT_NKSR_PYTHON if DEFAULT_NKSR_PYTHON.is_file() else Path(sys.executable),
    )
    parser.add_argument("--nksr-detail-level", type=float, default=0.1)
    parser.add_argument("--nksr-voxel-size", type=float)
    parser.add_argument("--nksr-chunk-size", type=float, default=-1.0)
    parser.add_argument("--nksr-mise-iter", type=int, default=1)
    parser.add_argument("--nksr-device", default="cuda:0")
    return parser.parse_args()


def _load_subset_sequences(subset_root: Path) -> list[str]:
    manifest = subset_root / "terrain_traversal_manifest.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text())
        clips = [str(entry["clip_id"]) for entry in payload.get("clips", []) if entry.get("clip_id")]
        if clips:
            return clips
    motion_dir = subset_root / "___crisp_clean_motion"
    return [path.stem for path in sorted(motion_dir.glob("*.npz"), key=lambda p: _natural_key(p.name))]


def _scale_from_urdf(terrain_seq_dir: Path) -> float:
    matches = sorted(terrain_seq_dir.glob("multi_boxes_scaled_*.urdf"), key=lambda p: _natural_key(p.name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one multi_boxes_scaled_*.urdf in {terrain_seq_dir}, got {len(matches)}")
    root = ET.parse(matches[0]).getroot()
    scales: list[float] = []
    for mesh in root.findall(".//mesh"):
        raw = mesh.attrib.get("scale")
        if not raw:
            continue
        values = [float(part) for part in raw.split()]
        if len(values) != 3:
            raise ValueError(f"Bad mesh scale in {matches[0]}: {raw}")
        if max(values) - min(values) > 1.0e-6:
            raise ValueError(f"Expected uniform terrain scale in {matches[0]}, got {values}")
        scales.append(values[0])
    if not scales:
        raise ValueError(f"No mesh scale found in {matches[0]}")
    if max(scales) - min(scales) > 1.0e-6:
        raise ValueError(f"Non-uniform per-piece scales in {matches[0]}")
    return float(scales[0])


def _load_z_offset(terrain_seq_dir: Path, seq: str) -> float:
    seq_npz = terrain_seq_dir / f"{seq}.npz"
    if not seq_npz.is_file():
        raise FileNotFoundError(seq_npz)
    with np.load(seq_npz, allow_pickle=True) as data:
        if "z_offset_applied_to_human_and_terrain" not in data.files:
            return 0.0
        return float(np.asarray(data["z_offset_applied_to_human_and_terrain"]).reshape(-1)[0])


def _load_sqs_transform(post_seq_root: Path) -> tuple[np.ndarray, np.ndarray]:
    params_path = post_seq_root / "scene_mesh_sqs/sqs_params.npz"
    if not params_path.is_file():
        raise FileNotFoundError(params_path)
    with np.load(params_path, allow_pickle=True) as data:
        rotation = np.asarray(data["world_rotation"], dtype=np.float32)
        translation = np.asarray(data["shared_translation"], dtype=np.float32).reshape(3)
    if rotation.shape != (3, 3):
        raise ValueError(f"Bad world_rotation shape in {params_path}: {rotation.shape}")
    return rotation, translation


def _normalise_normals(normals: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.divide(normals, np.maximum(norm, 1.0e-8)).astype(np.float32, copy=False)


def _scene_npz_path(scene_npz_root: Path, seq: str, hmr_type: str) -> Path:
    candidates = (
        scene_npz_root / f"{seq}_vggt_omega_{hmr_type}_sgd_cvd_hr.npz",
        scene_npz_root / f"{seq}_{hmr_type}_sgd_cvd_hr.npz",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No scene NPZ for {seq} under {scene_npz_root}; tried {candidates}")


def _resize_binary_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    from skimage.transform import resize

    if mask.shape == shape:
        return np.asarray(mask, dtype=bool)
    return resize(mask, shape, order=0, preserve_range=True, anti_aliasing=False).astype(bool)


def _dilate_binary_mask(mask: np.ndarray) -> np.ndarray:
    from skimage.morphology import dilation, disk

    return np.asarray(dilation(mask.astype(bool, copy=False), disk(11)), dtype=bool)


def _load_point_colors_from_scene_npz(
    scene_npz: Path,
    *,
    frame_indices: np.ndarray,
    expected_count: int,
    depth_conf_quantile: float,
) -> np.ndarray:
    with np.load(scene_npz, allow_pickle=True) as data:
        images = np.asarray(data["images"], dtype=np.uint8)
        depths = np.asarray(data["depths"], dtype=np.float32)
        confidence = np.asarray(data["depth_conf"], dtype=np.float32) if "depth_conf" in data.files else None
        masks = np.asarray(data["enlarged_dynamic_mask"]) if "enlarged_dynamic_mask" in data.files else None
        obj_masks = np.asarray(data["obj_masks"]) if "obj_masks" in data.files else None

    if images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError(f"Bad images shape in {scene_npz}: {images.shape}")
    if depths.shape[:3] != images.shape[:3]:
        raise ValueError(f"Depth shape {depths.shape} does not match images shape {images.shape} in {scene_npz}")
    if confidence is not None and confidence.shape[:3] != images.shape[:3]:
        raise ValueError(f"depth_conf shape {confidence.shape} does not match images shape {images.shape} in {scene_npz}")

    threshold = 0.0
    if confidence is not None:
        finite_conf = confidence[np.isfinite(confidence)]
        if finite_conf.size:
            q = float(np.clip(depth_conf_quantile, 0.0, 1.0))
            threshold = float(np.quantile(finite_conf, q))

    colors_by_frame: list[np.ndarray] = []
    counts: list[int] = []
    for frame_idx_raw in np.asarray(frame_indices, dtype=np.int64).reshape(-1):
        frame_idx = int(frame_idx_raw)
        if frame_idx < 0 or frame_idx >= images.shape[0]:
            raise IndexError(f"Frame index {frame_idx} outside {scene_npz} images length {images.shape[0]}")
        rgb = images[frame_idx, ..., :3]
        depth = depths[frame_idx]
        shape = depth.shape

        if masks is None or len(masks) == 0:
            human_mask = np.zeros(shape, dtype=bool)
        else:
            human_mask = _dilate_binary_mask(_resize_binary_mask(masks[frame_idx], shape))

        if obj_masks is None or len(obj_masks) == 0:
            obj_mask = np.zeros(shape, dtype=bool)
        else:
            obj_mask = _dilate_binary_mask(_resize_binary_mask(obj_masks[frame_idx], shape))

        if confidence is None:
            conf_mask = np.ones(shape, dtype=bool)
        else:
            conf_mask = _resize_binary_mask(confidence[frame_idx] > threshold, shape)

        valid = conf_mask & np.isfinite(depth) & (depth > 0.0) & (~human_mask) & (~obj_mask)
        frame_colors = rgb.reshape(-1, 3)[valid.reshape(-1)]
        colors_by_frame.append(frame_colors.astype(np.uint8, copy=False))
        counts.append(int(frame_colors.shape[0]))

    colors = (
        np.concatenate(colors_by_frame, axis=0).astype(np.uint8, copy=False)
        if colors_by_frame
        else np.zeros((0, 3), dtype=np.uint8)
    )
    if colors.shape[0] != int(expected_count):
        raise ValueError(
            f"Color reconstruction from {scene_npz} produced {colors.shape[0]} colors, "
            f"but raw pointcloud has {expected_count} points. First per-frame counts: {counts[:8]}"
        )
    return colors


def _load_and_transform_pointcloud(
    raw_npz: Path,
    *,
    scene_npz: Path | None,
    rotation: np.ndarray,
    translation: np.ndarray,
    z_offset: float,
    terrain_scale: float,
    max_points: int,
    seed: int,
    depth_conf_quantile: float,
) -> dict[str, np.ndarray]:
    with np.load(raw_npz, allow_pickle=True) as data:
        points = np.asarray(data["points"], dtype=np.float32)
        normals = np.asarray(data["normals"], dtype=np.float32)
        frame_indices = np.asarray(data["frame_indices"]) if "frame_indices" in data.files else np.asarray([], dtype=np.int32)
        interval = np.asarray(data["interval"]) if "interval" in data.files else np.asarray(-1, dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Bad points shape in {raw_npz}: {points.shape}")
    if normals.shape != points.shape:
        raise ValueError(f"Bad normals shape in {raw_npz}: {normals.shape}; expected {points.shape}")

    colors = None
    if scene_npz is not None:
        colors = _load_point_colors_from_scene_npz(
            scene_npz,
            frame_indices=frame_indices,
            expected_count=int(points.shape[0]),
            depth_conf_quantile=depth_conf_quantile,
        )

    normal_norm = np.linalg.norm(normals, axis=1)
    finite = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1) & (normal_norm > 1.0e-6)
    points = points[finite]
    normals = normals[finite]
    if colors is not None:
        colors = colors[finite]
    source_count = int(points.shape[0])
    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(points.shape[0], size=max_points, replace=False)
        keep.sort()
        points = points[keep]
        normals = normals[keep]
        if colors is not None:
            colors = colors[keep]

    points = (points @ rotation.T + translation[None, :]).astype(np.float32, copy=False)
    if z_offset != 0.0:
        points[:, 2] += np.float32(z_offset)
    if terrain_scale != 1.0:
        points *= np.float32(terrain_scale)
    normals = _normalise_normals(normals @ rotation.T)

    payload = {
        "points": points.astype(np.float32, copy=False),
        "normals": normals.astype(np.float32, copy=False),
        "frame_indices": frame_indices,
        "interval": interval,
        "source_point_count": np.asarray(source_count, dtype=np.int64),
    }
    if colors is not None:
        payload["colors"] = colors.astype(np.uint8, copy=False)
    return payload


def _write_pointcloud_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    import trimesh

    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.points.PointCloud(points, colors=colors).export(path)


def _save_npz_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.stem}.", suffix=".npz")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        np.savez_compressed(str(tmp), **payload)
        shutil.copyfile(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _run_nksr(
    *,
    nksr_python: Path,
    input_npz: Path,
    output_dir: Path,
    detail_level: float,
    voxel_size: float | None,
    chunk_size: float,
    mise_iter: int,
    device: str,
    force: bool,
) -> Path:
    mesh_path = output_dir / "scene_mesh_nksr.obj"
    if mesh_path.is_file() and not force:
        return mesh_path
    script = Path(__file__).resolve().parent / "run_nksr.py"
    cmd = [
        str(nksr_python),
        str(script),
        "--input-npz",
        str(input_npz),
        "--output-dir",
        str(output_dir),
        "--detail-level",
        str(float(detail_level)),
        "--chunk-size",
        str(float(chunk_size)),
        "--mise-iter",
        str(int(mise_iter)),
        "--device",
        str(device),
    ]
    if voxel_size is not None:
        cmd.extend(["--voxel-size", str(float(voxel_size))])
    output_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, cwd=script.parent, text=True, capture_output=True)
    (output_dir / "nksr.log").write_text((result.stdout or "") + (result.stderr or ""), encoding="utf-8")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return mesh_path


def _process_sequence(args: argparse.Namespace, seq: str) -> dict[str, Any]:
    out_seq = args.output_root / seq
    pc_out = out_seq / "pointcloud_aligned.npz"
    terrain_seq_dir = args.terrain_root / seq
    raw_seq_root = args.raw_root / seq / args.hmr_type
    post_seq_root = args.post_root / seq / args.hmr_type
    raw_npz = raw_seq_root / "nksr_input/pointcloud_world.npz"
    scene_npz = None if args.skip_colors else _scene_npz_path(args.scene_npz_root, seq, args.hmr_type)
    sqs_mesh = args.subset_root / "___crisp_clean_geometry" / f"{seq}.obj"
    if not raw_npz.is_file():
        raise FileNotFoundError(raw_npz)
    if not sqs_mesh.is_file():
        raise FileNotFoundError(sqs_mesh)

    rotation, translation = _load_sqs_transform(post_seq_root)
    z_offset = _load_z_offset(terrain_seq_dir, seq)
    terrain_scale = _scale_from_urdf(terrain_seq_dir)

    status = "exists"
    if args.force or not pc_out.is_file():
        payload = _load_and_transform_pointcloud(
            raw_npz,
            scene_npz=scene_npz,
            rotation=rotation,
            translation=translation,
            z_offset=z_offset,
            terrain_scale=terrain_scale,
            max_points=int(args.max_points),
            seed=abs(hash(seq)) % (2**32),
            depth_conf_quantile=float(args.depth_conf_quantile),
        )
        payload.update(
            {
                "world_rotation": rotation,
                "shared_translation": translation,
                "z_offset_applied_to_human_and_terrain": np.asarray(z_offset, dtype=np.float32),
                "terrain_scale": np.asarray(terrain_scale, dtype=np.float32),
                "source_raw_pointcloud_npz": np.asarray(str(raw_npz.resolve())),
                "source_post_sqs_params": np.asarray(str((post_seq_root / "scene_mesh_sqs/sqs_params.npz").resolve())),
                "source_color_scene_npz": np.asarray("" if scene_npz is None else str(scene_npz.resolve())),
                "source_sqs_mesh": np.asarray(str(sqs_mesh.resolve())),
            }
        )
        _save_npz_atomic(pc_out, payload)
        if args.write_ply:
            _write_pointcloud_ply(out_seq / "pointcloud_aligned.ply", payload["points"], payload.get("colors"))
        status = "written"

    nksr_mesh = out_seq / "nksr/scene_mesh_nksr.obj"
    nksr_status = "exists" if nksr_mesh.is_file() else "not_requested"
    if args.run_nksr:
        _run_nksr(
            nksr_python=args.nksr_python,
            input_npz=pc_out,
            output_dir=out_seq / "nksr",
            detail_level=float(args.nksr_detail_level),
            voxel_size=args.nksr_voxel_size,
            chunk_size=float(args.nksr_chunk_size),
            mise_iter=int(args.nksr_mise_iter),
            device=str(args.nksr_device),
            force=bool(args.force),
        )
        nksr_status = "exists" if nksr_mesh.is_file() else "missing"

    with np.load(pc_out, allow_pickle=True) as data:
        point_count = int(data["points"].shape[0])
        source_point_count = int(np.asarray(data["source_point_count"]).reshape(-1)[0])
        has_colors = "colors" in data.files and data["colors"].shape[0] == point_count

    nksr_meta_path = out_seq / "nksr/scene_mesh_nksr.json"
    nksr_meta: dict[str, Any] | None = None
    if nksr_meta_path.is_file():
        nksr_meta = json.loads(nksr_meta_path.read_text())

    return {
        "clip_id": seq,
        "pointcloud_aligned": str(pc_out),
        "pointcloud_status": status,
        "source_point_count": source_point_count,
        "point_count": point_count,
        "has_point_colors": bool(has_colors),
        "sqs_mesh": str(sqs_mesh.resolve()),
        "nksr_mesh": str(nksr_mesh.resolve()) if nksr_mesh.is_file() else None,
        "nksr_status": nksr_status,
        "nksr_meta": nksr_meta,
        "raw_pointcloud": str(raw_npz.resolve()),
        "color_scene_npz": None if scene_npz is None else str(scene_npz.resolve()),
        "post_sqs_root": str((post_seq_root / "scene_mesh_sqs").resolve()),
        "terrain_seq_dir": str(terrain_seq_dir.resolve()),
        "z_offset": z_offset,
        "terrain_scale": terrain_scale,
    }


def main() -> None:
    args = _parse_args()
    args.subset_root = args.subset_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.raw_root = args.raw_root.expanduser().resolve()
    args.post_root = args.post_root.expanduser().resolve()
    args.scene_npz_root = args.scene_npz_root.expanduser().resolve()
    args.terrain_root = args.terrain_root.expanduser().resolve()
    args.nksr_python = args.nksr_python.expanduser().resolve()

    sequences = list(args.sequences) if args.sequences else _load_subset_sequences(args.subset_root)
    if not sequences:
        raise ValueError("No sequences selected.")

    records = []
    for index, seq in enumerate(sequences, start=1):
        record = _process_sequence(args, seq)
        records.append(record)
        print(
            f"[{index:02d}/{len(sequences):02d}] {seq}: "
            f"pc={record['pointcloud_status']} points={record['point_count']}/{record['source_point_count']} "
            f"colors={record['has_point_colors']} scale={record['terrain_scale']:.6f} "
            f"z_offset={record['z_offset']:.6f} nksr={record['nksr_status']}",
            flush=True,
        )

    manifest = {
        "schema_version": 1,
        "format": "crisp_stairs_pointcloud_sqs_nksr_compare",
        "subset_root": str(args.subset_root),
        "raw_root": str(args.raw_root),
        "post_root": str(args.post_root),
        "scene_npz_root": str(args.scene_npz_root),
        "terrain_root": str(args.terrain_root),
        "output_root": str(args.output_root),
        "point_colors": None
        if args.skip_colors
        else {
            "source": "original RGB from *_vggt_omega_<hmr>_sgd_cvd_hr.npz, indexed by the same nksr_bg_indices used by script_7",
            "depth_conf_quantile": float(args.depth_conf_quantile),
            "human_mask": "enlarged_dynamic_mask resized to VGGT image shape and dilated with disk(11)",
            "object_mask": "obj_masks resized to VGGT image shape and dilated with disk(11), when present",
        },
        "transform_order": [
            "points = points @ world_rotation.T + shared_translation",
            "points[:, 2] += z_offset_applied_to_human_and_terrain",
            "points *= terrain_scale",
            "normals = normalize(normals @ world_rotation.T)",
        ],
        "max_points": int(args.max_points),
        "run_nksr": bool(args.run_nksr),
        "nksr": {
            "python": str(args.nksr_python),
            "detail_level": float(args.nksr_detail_level),
            "voxel_size": None if args.nksr_voxel_size is None else float(args.nksr_voxel_size),
            "chunk_size": float(args.nksr_chunk_size),
            "mise_iter": int(args.nksr_mise_iter),
            "device": str(args.nksr_device),
        },
        "clip_count": len(records),
        "clips": records,
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] wrote {len(records)} records to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
