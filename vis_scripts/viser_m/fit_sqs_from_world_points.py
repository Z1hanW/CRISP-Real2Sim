#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("MKL_NUM_THREADS", "32")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "32")

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy.spatial.transform import Rotation
from sklearn.cluster import MiniBatchKMeans

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vis_scripts.viser_m.utils import (
    _oriented_mean_normal,
    _prune_coplanar_overlapping_boxes,
    fit_support_aware_plane_boxes,
    make_local_frame,
    robust_plane_ransac,
)


DEFAULT_INPUT_ROOT = Path("/nfs/zzzihanw/crisp_stairs_sqs_v2_compare_dedup")
DEFAULT_OUTPUT_ROOT = Path("/nfs/zzzihanw/crisp_stairs_sqs_worldpoints")
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
    parser = argparse.ArgumentParser(description="Fit SQS terrain boxes directly from fused world points and normals.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--hmr-type", default="gv")
    parser.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-fit-points", type=int, default=700_000)
    parser.add_argument("--normal-clusters", type=int, default=24)
    parser.add_argument("--offset-bin", type=float, default=0.055)
    parser.add_argument("--plane-thickness", type=float, default=0.045)
    parser.add_argument("--normal-align-deg", type=float, default=16.0)
    parser.add_argument("--component-cell", type=float, default=0.18)
    parser.add_argument("--min-normal-cluster-points", type=int, default=2_500)
    parser.add_argument("--min-layer-points", type=int, default=900)
    parser.add_argument("--min-component-points", type=int, default=450)
    parser.add_argument("--max-component-fit-points", type=int, default=120_000)
    parser.add_argument("--max-primitives", type=int, default=90)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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


def _normalise_np(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, np.maximum(norm, 1.0e-8)).astype(np.float32, copy=False)


def _canonicalise_normals(normals: np.ndarray) -> np.ndarray:
    normals = _normalise_np(normals)
    dominant = np.argmax(np.abs(normals), axis=1)
    rows = np.arange(normals.shape[0])
    signs = np.where(normals[rows, dominant] < 0.0, -1.0, 1.0).astype(np.float32)
    return normals * signs[:, None]


def _sample_indices(count: int, max_count: int, seed: int) -> np.ndarray:
    if max_count <= 0 or count <= max_count:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    keep = rng.choice(count, size=max_count, replace=False)
    keep.sort()
    return keep.astype(np.int64)


def _load_world_points(path: Path, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=True) as data:
        points = np.asarray(data["points"], dtype=np.float32)
        normals = np.asarray(data["normals"], dtype=np.float32)
        extras = {key: np.asarray(data[key]) for key in data.files if key not in {"points", "normals"}}
    finite = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(normals).all(axis=1)
        & (np.linalg.norm(points, axis=1) > 1.0e-8)
        & (np.linalg.norm(normals, axis=1) > 1.0e-8)
    )
    points = points[finite]
    normals = normals[finite]
    keep = _sample_indices(points.shape[0], max_points, seed)
    return points[keep], _normalise_np(normals[keep]), extras


def _offset_groups(offsets: np.ndarray, bin_size: float, min_layer_points: int) -> list[np.ndarray]:
    if offsets.size == 0:
        return []
    bins = np.floor(offsets / float(bin_size)).astype(np.int64)
    unique_bins, counts = np.unique(bins, return_counts=True)
    active_count = max(200, min_layer_points // 8)
    active_bins = unique_bins[counts >= active_count]
    if active_bins.size == 0:
        return []
    active_bins.sort()

    groups: list[list[int]] = []
    cur = [int(active_bins[0])]
    for b_raw in active_bins[1:]:
        b = int(b_raw)
        if b <= cur[-1] + 1:
            cur.append(b)
        else:
            groups.append(cur)
            cur = [b]
    groups.append(cur)

    masks: list[np.ndarray] = []
    for group in groups:
        lo = min(group) - 1
        hi = max(group) + 1
        mask = (bins >= lo) & (bins <= hi)
        if int(mask.sum()) >= min_layer_points:
            masks.append(mask)
    return masks


def _connected_components_xy(
    xy: np.ndarray,
    *,
    cell_size: float,
    min_points: int,
    min_cell_points: int = 3,
) -> list[np.ndarray]:
    if xy.shape[0] < min_points:
        return []
    cells = np.floor(xy / float(cell_size)).astype(np.int64)
    cell_to_indices: dict[tuple[int, int], list[int]] = {}
    for idx, cell in enumerate(cells):
        key = (int(cell[0]), int(cell[1]))
        cell_to_indices.setdefault(key, []).append(idx)

    active = {key for key, inds in cell_to_indices.items() if len(inds) >= min_cell_points}
    visited: set[tuple[int, int]] = set()
    components: list[np.ndarray] = []
    neighbors = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx != 0 or dy != 0)

    for start in sorted(active):
        if start in visited:
            continue
        q: deque[tuple[int, int]] = deque([start])
        visited.add(start)
        comp_cells: list[tuple[int, int]] = []
        while q:
            cell = q.popleft()
            comp_cells.append(cell)
            for dx, dy in neighbors:
                nxt = (cell[0] + dx, cell[1] + dy)
                if nxt in active and nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
        point_indices = np.concatenate([np.asarray(cell_to_indices[cell], dtype=np.int64) for cell in comp_cells])
        if point_indices.size >= min_points:
            components.append(point_indices)
    components.sort(key=lambda idx: int(idx.size), reverse=True)
    return components


def _params_from_records(records: list[dict[str, torch.Tensor]]) -> np.ndarray:
    rows: list[list[float]] = []
    for rec in records:
        half = rec["half_sz"].detach().cpu().numpy().astype(np.float32)
        R_bw = rec["R_bw"].detach().cpu().numpy().astype(np.float32)
        centre = rec["centre"].detach().cpu().numpy().astype(np.float32)
        euler = Rotation.from_matrix(R_bw).as_euler("ZYX").astype(np.float32)
        rows.append(
            [
                -2.398,
                -2.398,
                float(half[0]),
                float(half[1]),
                float(half[2]),
                float(euler[0]),
                float(euler[1]),
                float(euler[2]),
                float(centre[0]),
                float(centre[1]),
                float(centre[2]),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _mesh_from_record(rec: dict[str, torch.Tensor]) -> trimesh.Trimesh:
    half = rec["half_sz"].detach().cpu().numpy().astype(np.float32)
    R_bw = rec["R_bw"].detach().cpu().numpy().astype(np.float32)
    centre = rec["centre"].detach().cpu().numpy().astype(np.float32)
    mesh = trimesh.creation.box(extents=np.maximum(2.0 * half, 1.0e-4))
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = R_bw
    transform[:3, 3] = centre
    mesh.apply_transform(transform)
    return mesh


def _write_urdf(path: Path, piece_names: list[str]) -> None:
    robot = ET.Element("robot", name="scene_mesh_sqs")
    scene_link = ET.SubElement(robot, "link", name="scene")
    for name in piece_names:
        for tag in ("visual", "collision"):
            sec = ET.SubElement(scene_link, tag)
            ET.SubElement(sec, "origin", xyz="0 0 0", rpy="0 0 0")
            geom = ET.SubElement(sec, "geometry")
            ET.SubElement(geom, "mesh", filename=f"pieces/{name}")
    ET.ElementTree(robot).write(path, encoding="utf-8", xml_declaration=True)


def _link_or_copy_pointcloud(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _fit_sequence(args: argparse.Namespace, seq: str, index: int, total: int) -> dict[str, Any]:
    hmr_type = str(args.hmr_type)
    in_seq_root = args.input_root / seq / hmr_type
    out_seq_root = args.output_root / seq / hmr_type
    raw_npz = in_seq_root / "nksr_input/pointcloud_world.npz"
    if not raw_npz.is_file():
        raise FileNotFoundError(raw_npz)

    if out_seq_root.exists() and (out_seq_root / "scene_mesh_sqs/scene_mesh_sqs.obj").is_file() and not args.force:
        return {"seq": seq, "status": "exists", "output": str(out_seq_root)}
    if out_seq_root.exists() and args.force:
        shutil.rmtree(out_seq_root, ignore_errors=True)

    points_np, normals_np, extras = _load_world_points(raw_npz, int(args.max_fit_points), seed=abs(hash(seq)) % (2**32))
    canonical_normals = _canonicalise_normals(normals_np)
    n_clusters = max(1, min(int(args.normal_clusters), points_np.shape[0] // max(int(args.min_normal_cluster_points), 1)))
    if n_clusters < 1:
        n_clusters = 1
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=16384,
        n_init=3,
        random_state=abs(hash(("normal", seq))) % (2**32),
    )
    labels = kmeans.fit_predict(canonical_normals)
    device = torch.device(str(args.device) if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")

    records: list[dict[str, torch.Tensor]] = []
    cos_align = float(np.cos(np.deg2rad(float(args.normal_align_deg))))

    for cluster_id in range(n_clusters):
        cluster_idx = np.flatnonzero(labels == cluster_id)
        if cluster_idx.size < int(args.min_normal_cluster_points):
            continue
        n0_np = _normalise_np(canonical_normals[cluster_idx].mean(axis=0, keepdims=True))[0]
        align = canonical_normals[cluster_idx] @ n0_np
        cluster_idx = cluster_idx[align >= cos_align]
        if cluster_idx.size < int(args.min_normal_cluster_points):
            continue

        offsets = points_np[cluster_idx] @ n0_np
        for layer_id, layer_mask_local in enumerate(_offset_groups(offsets, float(args.offset_bin), int(args.min_layer_points))):
            layer_idx = cluster_idx[layer_mask_local]
            if layer_idx.size < int(args.min_layer_points):
                continue

            layer_points_np = points_np[layer_idx]
            layer_normals_np = canonical_normals[layer_idx]
            n_seed = torch.from_numpy(n0_np).to(device=device, dtype=torch.float32)
            P_layer = torch.from_numpy(layer_points_np).to(device=device, dtype=torch.float32)
            N_layer = torch.from_numpy(layer_normals_np).to(device=device, dtype=torch.float32)
            n_avg = _oriented_mean_normal(N_layer)
            n, c, inliers = robust_plane_ransac(P_layer, n_avg)
            if n is None or c is None or inliers is None:
                n = n_seed
                c = P_layer.mean(dim=0)
                signed = (P_layer - c.unsqueeze(0)) @ n
                inliers = signed.abs() <= float(args.plane_thickness)
            signed = (P_layer - c.unsqueeze(0)) @ n
            normal_ok = torch.abs(N_layer @ n) >= cos_align
            inlier_mask = inliers & (signed.abs() <= float(args.plane_thickness)) & normal_ok
            if int(inlier_mask.sum().item()) < int(args.min_layer_points):
                continue

            P_in = P_layer[inlier_mask]
            u0, v0, _ = make_local_frame(n)
            UV = torch.stack([u0, v0], dim=1)
            xy = (P_in @ UV).detach().cpu().numpy()
            components = _connected_components_xy(
                xy,
                cell_size=float(args.component_cell),
                min_points=int(args.min_component_points),
            )
            for comp_idx, comp_local_idx in enumerate(components):
                P_comp = P_in[torch.from_numpy(comp_local_idx).to(device=device, dtype=torch.long)]
                if P_comp.shape[0] > int(args.max_component_fit_points):
                    keep = _sample_indices(P_comp.shape[0], int(args.max_component_fit_points), seed=abs(hash((seq, cluster_id, layer_id, comp_idx))) % (2**32))
                    P_fit = P_comp[torch.from_numpy(keep).to(device=device, dtype=torch.long)]
                else:
                    P_fit = P_comp
                pieces = fit_support_aware_plane_boxes(
                    P_fit,
                    n,
                    c,
                    f"{cluster_id}.{layer_id}.{comp_idx}",
                    max_depth=2,
                    min_pts=max(180, min(int(args.min_component_points) // 2, 1200)),
                    min_support=0.48,
                    min_improvement=0.12,
                )
                for piece_idx, (R_bw, centre, half_sz, piece_points, support) in enumerate(pieces):
                    records.append(
                        {
                            "gid": f"{cluster_id}.{layer_id}.{comp_idx}",
                            "piece_idx": piece_idx,
                            "R_bw": R_bw.to(device=device, dtype=torch.float32),
                            "centre": centre.to(device=device, dtype=torch.float32),
                            "half_sz": half_sz.clamp(min=0.02).to(device=device, dtype=torch.float32),
                            "points": piece_points.to(device=device, dtype=torch.float32),
                            "support": float(support),
                        }
                    )

    before_dedup = len(records)
    records = _prune_coplanar_overlapping_boxes(records)
    records.sort(key=lambda rec: float((4.0 * rec["half_sz"][0] * rec["half_sz"][1]).detach().cpu().item()), reverse=True)
    if int(args.max_primitives) > 0 and len(records) > int(args.max_primitives):
        records = records[: int(args.max_primitives)]
    after_dedup = len(records)

    sqs_root = out_seq_root / "scene_mesh_sqs"
    pieces_root = sqs_root / "pieces"
    pieces_root.mkdir(parents=True, exist_ok=True)
    piece_names: list[str] = []
    meshes = []
    for piece_idx, rec in enumerate(records):
        mesh = _mesh_from_record(rec)
        name = f"part_{piece_idx:03d}.obj"
        mesh.export(pieces_root / name)
        piece_names.append(name)
        meshes.append(mesh)

    if meshes:
        merged = trimesh.util.concatenate(meshes)
    else:
        merged = trimesh.Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int64), process=False)
    merged.export(sqs_root / "scene_mesh_sqs.obj")
    _write_urdf(sqs_root / "scene_mesh_sqs.urdf", piece_names)

    params = _params_from_records(records)
    np.save(sqs_root / "sqs_params.npy", params)
    _savez_compressed_atomic(
        sqs_root / "sqs_params.npz",
        params=params,
        source=np.asarray("direct_world_points"),
        sampled_fit_points=np.asarray(points_np.shape[0], dtype=np.int64),
    )
    _link_or_copy_pointcloud(raw_npz.resolve(), out_seq_root / "nksr_input/pointcloud_world.npz")

    print(
        f"[{index:02d}/{total:02d}] {seq}: sampled={points_np.shape[0]} "
        f"records={before_dedup}->{after_dedup} pieces={len(piece_names)}",
        flush=True,
    )
    return {
        "seq": seq,
        "status": "written",
        "sampled_fit_points": int(points_np.shape[0]),
        "normal_clusters": int(n_clusters),
        "before_dedup": int(before_dedup),
        "piece_count": int(len(piece_names)),
        "output": str(out_seq_root),
        "source_pointcloud": str(raw_npz.resolve()),
    }


def main() -> None:
    args = _parse_args()
    args.input_root = args.input_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    sequences = [str(seq) for seq in args.sequences]
    records = []
    for index, seq in enumerate(sequences, start=1):
        records.append(_fit_sequence(args, seq, index, len(sequences)))
    manifest = {
        "schema_version": 1,
        "format": "direct_world_points_sqs",
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "hmr_type": str(args.hmr_type),
        "max_fit_points": int(args.max_fit_points),
        "normal_clusters": int(args.normal_clusters),
        "offset_bin": float(args.offset_bin),
        "plane_thickness": float(args.plane_thickness),
        "component_cell": float(args.component_cell),
        "records": records,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] wrote direct world-points SQS root: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
