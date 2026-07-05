#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter SQS primitive pieces whose surfaces have poor support in the reconstructed point cloud."
    )
    parser.add_argument("--source-post-root", type=Path, default=REPO_ROOT / "results/output/post_scene_vggt_omega")
    parser.add_argument("--raw-root", type=Path, default=REPO_ROOT / "results/output/scene_vggt_omega_consistent_camera_min1")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "results/output/post_scene_vggt_omega_filtered")
    parser.add_argument("--hmr-type", default="gv")
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--near-threshold", type=float, default=0.06)
    parser.add_argument("--min-coverage", type=float, default=0.50)
    parser.add_argument("--max-median-dist", type=float, default=0.12)
    parser.add_argument("--max-p90-dist", type=float, default=0.25)
    parser.add_argument("--max-point-samples", type=int, default=600_000)
    parser.add_argument("--min-piece-samples", type=int, default=100)
    parser.add_argument("--max-piece-samples", type=int, default=800)
    parser.add_argument("--sample-area", type=float, default=0.0025, help="Approximate surface area per piece sample.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing filtered sequence outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Compute metrics without writing filtered outputs.")
    return parser.parse_args()


def resolve_root(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh from {path}, got {type(mesh)}")
    return mesh


def load_point_cloud(raw_seq_root: Path, rotation: np.ndarray, translation: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    pc_npz = raw_seq_root / "nksr_input" / "pointcloud_world.npz"
    if not pc_npz.is_file():
        raise FileNotFoundError(f"Missing point cloud: {pc_npz}")
    with np.load(pc_npz, allow_pickle=True) as data:
        if "points" not in data.files:
            raise KeyError(f"{pc_npz} missing key 'points'")
        points = np.asarray(data["points"], dtype=np.float32)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if max_samples > 0 and points.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(points.shape[0], size=max_samples, replace=False)]
    return (points @ rotation.T + translation[None, :]).astype(np.float32, copy=False)


def sample_piece_surface(mesh: trimesh.Trimesh, min_samples: int, max_samples: int, sample_area: float) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if vertices.size == 0:
        return vertices.reshape(0, 3)
    faces = np.asarray(mesh.faces)
    area = float(getattr(mesh, "area", 0.0))
    if faces.size > 0 and area > 1.0e-8:
        count = int(np.clip(np.ceil(area / max(float(sample_area), 1.0e-8)), min_samples, max_samples))
        samples, _ = trimesh.sample.sample_surface(mesh, count)
        return np.asarray(samples, dtype=np.float32)
    if vertices.shape[0] > max_samples:
        return vertices[np.linspace(0, vertices.shape[0] - 1, max_samples).astype(np.int64)]
    return vertices


def write_filtered_urdf(urdf_path: Path, piece_names: list[str]) -> None:
    robot = ET.Element("robot", name="scene")
    link = ET.SubElement(robot, "link", name="scene_link")
    for name in piece_names:
        for tag in ("visual", "collision"):
            elem = ET.SubElement(link, tag)
            ET.SubElement(elem, "origin", xyz="0.000000 0.000000 0.000000", rpy="0.000000 0.000000 0.000000")
            geometry = ET.SubElement(elem, "geometry")
            ET.SubElement(geometry, "mesh", filename=name)
    ET.ElementTree(robot).write(urdf_path, encoding="utf-8", xml_declaration=True)


def save_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        np.savez_compressed(tmp_path, **payload)
        shutil.copyfile(tmp_path, path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def filter_sqs_params(src_npz: Path, dst_npz: Path, dst_npy: Path, keep_mask: np.ndarray) -> None:
    with np.load(src_npz, allow_pickle=True) as data:
        payload: dict[str, np.ndarray] = {}
        for key in data.files:
            value = np.asarray(data[key])
            if value.shape[:1] == keep_mask.shape and key not in {"world_rotation", "shared_translation"}:
                payload[key] = value[keep_mask]
            else:
                payload[key] = value
    params = np.asarray(payload["params"], dtype=np.float32)
    np.save(dst_npy, params)
    save_npz(dst_npz, payload)


def discover_sequences(source_post_root: Path, hmr_type: str) -> list[str]:
    return sorted(
        path.name
        for path in source_post_root.iterdir()
        if path.is_dir() and (path / hmr_type / "scene_mesh_sqs" / "scene_mesh_sqs.obj").is_file()
    )


def process_sequence(args: argparse.Namespace, seq_name: str) -> dict:
    source_post_root = resolve_root(args.source_post_root)
    raw_root = resolve_root(args.raw_root)
    output_root = resolve_root(args.output_root)

    src_seq_root = source_post_root / seq_name / args.hmr_type
    raw_seq_root = raw_root / seq_name / args.hmr_type
    src_sqs = src_seq_root / "scene_mesh_sqs"
    src_params = src_sqs / "sqs_params.npz"
    if not src_params.is_file():
        raise FileNotFoundError(f"Missing sqs params: {src_params}")

    with np.load(src_params, allow_pickle=True) as data:
        rotation = np.asarray(data["world_rotation"], dtype=np.float32)
        translation = np.asarray(data["shared_translation"], dtype=np.float32).reshape(3)
        if "piece_name_utf8" in data.files:
            piece_names = [str(name) for name in np.asarray(data["piece_name_utf8"]).tolist()]
        else:
            piece_names = [path.name for path in sorted((src_sqs / "pieces").glob("part_*.obj"))]

    point_seed = abs(hash(seq_name)) % (2**32)
    points = load_point_cloud(raw_seq_root, rotation, translation, int(args.max_point_samples), point_seed)
    tree = cKDTree(points)

    metrics = []
    keep_mask = []
    for idx, piece_name in enumerate(piece_names):
        piece_path = src_sqs / "pieces" / piece_name
        if not piece_path.is_file():
            raise FileNotFoundError(f"Missing piece referenced by params: {piece_path}")
        mesh = load_mesh(piece_path)
        samples = sample_piece_surface(mesh, int(args.min_piece_samples), int(args.max_piece_samples), float(args.sample_area))
        if samples.size == 0:
            coverage = 0.0
            median_dist = float("inf")
            p90_dist = float("inf")
        else:
            dist, _ = tree.query(samples, k=1, workers=-1)
            coverage = float(np.mean(dist <= float(args.near_threshold)))
            median_dist = float(np.median(dist))
            p90_dist = float(np.percentile(dist, 90.0))

        keep = (
            coverage >= float(args.min_coverage)
            and median_dist <= float(args.max_median_dist)
            and p90_dist <= float(args.max_p90_dist)
        )
        keep_mask.append(keep)
        metrics.append(
            {
                "index": int(idx),
                "piece_name": piece_name,
                "keep": bool(keep),
                "coverage": coverage,
                "median_dist": median_dist,
                "p90_dist": p90_dist,
                "area": float(getattr(mesh, "area", 0.0)),
                "num_samples": int(samples.shape[0]),
            }
        )

    keep_mask_np = np.asarray(keep_mask, dtype=bool)
    if not np.any(keep_mask_np):
        best_idx = int(np.argmin([row["median_dist"] for row in metrics]))
        keep_mask_np[best_idx] = True
        metrics[best_idx]["keep"] = True
        metrics[best_idx]["forced_keep"] = True

    kept_names = [name for name, keep in zip(piece_names, keep_mask_np.tolist()) if keep]
    dropped = [row for row in metrics if not row["keep"]]

    dst_seq_root = output_root / seq_name / args.hmr_type
    if not args.dry_run:
        dst_parent = output_root / seq_name
        if dst_parent.exists():
            if not args.force:
                raise FileExistsError(f"Output exists; pass --force to overwrite: {dst_parent}")
            shutil.rmtree(dst_parent)
        shutil.copytree(source_post_root / seq_name, dst_parent)

        dst_sqs = dst_seq_root / "scene_mesh_sqs"
        dst_pieces = dst_sqs / "pieces"
        for piece_path in sorted(dst_pieces.glob("part_*.obj")):
            if piece_path.name not in set(kept_names):
                piece_path.unlink()

        meshes = [load_mesh(dst_pieces / name) for name in kept_names]
        combined = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        combined.export(dst_sqs / "scene_mesh_sqs.obj")
        write_filtered_urdf(dst_sqs / "scene_mesh_sqs.urdf", kept_names)
        filter_sqs_params(src_params, dst_sqs / "sqs_params.npz", dst_sqs / "sqs_params.npy", keep_mask_np)

        summary_path = dst_sqs / "filter_by_points_summary.json"
        summary_payload = {
            "sequence_name": seq_name,
            "source_post_root": str(source_post_root),
            "raw_root": str(raw_root),
            "output_root": str(output_root),
            "thresholds": {
                "near_threshold": float(args.near_threshold),
                "min_coverage": float(args.min_coverage),
                "max_median_dist": float(args.max_median_dist),
                "max_p90_dist": float(args.max_p90_dist),
                "max_point_samples": int(args.max_point_samples),
            },
            "num_points_used": int(points.shape[0]),
            "num_input_pieces": int(len(piece_names)),
            "num_kept_pieces": int(len(kept_names)),
            "num_dropped_pieces": int(len(dropped)),
            "dropped": dropped,
            "metrics": metrics,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return {
        "sequence_name": seq_name,
        "num_points_used": int(points.shape[0]),
        "num_input_pieces": int(len(piece_names)),
        "num_kept_pieces": int(len(kept_names)),
        "num_dropped_pieces": int(len(dropped)),
        "dropped_piece_names": [row["piece_name"] for row in dropped],
    }


def main() -> None:
    args = parse_args()
    source_post_root = resolve_root(args.source_post_root)
    sequences = args.sequences or discover_sequences(source_post_root, args.hmr_type)
    summaries = []
    for seq_name in sequences:
        summary = process_sequence(args, seq_name)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    total_dropped = sum(item["num_dropped_pieces"] for item in summaries)
    total_input = sum(item["num_input_pieces"] for item in summaries)
    print(
        json.dumps(
            {
                "num_sequences": len(summaries),
                "total_input_pieces": total_input,
                "total_dropped_pieces": total_dropped,
                "total_kept_pieces": total_input - total_dropped,
                "output_root": str(resolve_root(args.output_root)),
                "dry_run": bool(args.dry_run),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
