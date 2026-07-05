#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experimental convex clipping pass for SQS terrain boxes. Each input "
            "box is treated as the maximal enclosing plane footprint, then shaved "
            "by a small number of half-plane cuts where point support is poor."
        )
    )
    parser.add_argument("--input-seq-root", type=Path, required=True)
    parser.add_argument("--output-seq-root", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=1_200_000)
    parser.add_argument("--grid-base", type=int, default=72)
    parser.add_argument("--max-cuts", type=int, default=4)
    parser.add_argument("--min-points", type=int, default=500)
    parser.add_argument("--min-area-reduction", type=float, default=0.025)
    parser.add_argument("--min-support-gain", type=float, default=0.020)
    parser.add_argument("--min-keep-fraction", type=float, default=0.940)
    parser.add_argument("--target-support", type=float, default=0.985)
    parser.add_argument("--z-margin", type=float, default=0.025)
    parser.add_argument("--cut-padding", type=float, default=0.015)
    parser.add_argument("--neighbor-plane-clip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--neighbor-min-angle-deg", type=float, default=15.0)
    parser.add_argument("--neighbor-max-cuts", type=int, default=4)
    parser.add_argument("--neighbor-min-keep-fraction", type=float, default=0.78)
    parser.add_argument("--neighbor-min-area-reduction", type=float, default=0.015)
    parser.add_argument("--neighbor-support-drop", type=float, default=0.025)
    parser.add_argument("--neighbor-footprint-margin", type=float, default=0.08)
    parser.add_argument("--neighbor-priority", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--neighbor-spatial-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--neighbor-max-line-gap", type=float, default=0.25)
    parser.add_argument("--neighbor-max-support-distance", type=float, default=0.25)
    parser.add_argument("--neighbor-support-sample-points", type=int, default=4096)
    parser.add_argument("--neighbor-snap-fill", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--snap-fill-expand-margin", type=float, default=0.15)
    parser.add_argument("--snap-fill-max-lines", type=int, default=8)
    parser.add_argument("--snap-fill-max-discard-fraction", type=float, default=0.10)
    parser.add_argument("--snap-fill-min-area-ratio", type=float, default=0.35)
    parser.add_argument("--snap-fill-min-final-area-ratio", type=float, default=0.0)
    parser.add_argument("--snap-fill-max-area-ratio", type=float, default=1.02)
    parser.add_argument("--snap-fill-max-support-drop", type=float, default=0.005)
    parser.add_argument("--support-piece-cover", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cover-cell-size", type=float, default=0.18)
    parser.add_argument("--cover-max-pieces-per-input", type=int, default=10)
    parser.add_argument("--cover-min-cells", type=int, default=5)
    parser.add_argument("--cover-close-iters", type=int, default=1)
    parser.add_argument("--cover-min-points-per-cell", type=int, default=1)
    parser.add_argument("--cover-min-area", type=float, default=0.03)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_sqs(src_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    sqs_dir = src_root / "scene_mesh_sqs"
    sqs_npz = sqs_dir / "sqs_params.npz"
    sqs_npy = sqs_dir / "sqs_params.npy"
    if sqs_npz.is_file():
        with np.load(sqs_npz, allow_pickle=True) as data:
            params = np.asarray(data["params"], dtype=np.float32)
            extras = {key: np.asarray(data[key]) for key in data.files if key not in {"params", "piece_rot_p2w"}}
            if "piece_rot_p2w" in data.files:
                rotations = np.asarray(data["piece_rot_p2w"], dtype=np.float32)
            else:
                rotations = Rotation.from_euler("ZYX", params[:, 5:8]).as_matrix().astype(np.float32)
    elif sqs_npy.is_file():
        params = np.asarray(np.load(sqs_npy, allow_pickle=True), dtype=np.float32)
        extras = {}
        rotations = Rotation.from_euler("ZYX", params[:, 5:8]).as_matrix().astype(np.float32)
    else:
        raise FileNotFoundError(f"Missing sqs_params under {sqs_dir}")
    if params.ndim != 2 or params.shape[1] < 11:
        raise ValueError(f"Expected params shape [N, 11+], got {params.shape}")
    half = params[:, 2:5].astype(np.float32)
    centres = params[:, 8:11].astype(np.float32)
    return params, rotations, half, centres, extras


def _load_points(src_root: Path, max_points: int) -> np.ndarray:
    pc_path = src_root / "nksr_input/pointcloud_world.npz"
    if not pc_path.is_file():
        raise FileNotFoundError(pc_path)
    with np.load(pc_path, allow_pickle=True) as data:
        points = np.asarray(data["points"], dtype=np.float32)
    finite = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 1.0e-8)
    points = points[finite]
    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(12345)
        keep = rng.choice(points.shape[0], size=max_points, replace=False)
        keep.sort()
        points = points[keep]
    return points


def _polygon_signed_area(poly: np.ndarray) -> float:
    if poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _polygon_area(poly: np.ndarray) -> float:
    return abs(_polygon_signed_area(poly))


def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    if _polygon_signed_area(poly) < 0.0:
        return poly[::-1].copy()
    return poly


def _clip_polygon(poly: np.ndarray, normal: np.ndarray, offset: float, eps: float = 1.0e-7) -> np.ndarray:
    """Clip a convex polygon by normal.dot(x) <= offset."""
    if poly.shape[0] < 3:
        return poly[:0]
    out: list[np.ndarray] = []
    prev = poly[-1]
    prev_s = float(prev @ normal - offset)
    prev_inside = prev_s <= eps
    for cur in poly:
        cur_s = float(cur @ normal - offset)
        cur_inside = cur_s <= eps
        if cur_inside != prev_inside:
            denom = prev_s - cur_s
            if abs(denom) > 1.0e-12:
                t = prev_s / denom
                out.append(prev + t * (cur - prev))
        if cur_inside:
            out.append(cur)
        prev = cur
        prev_s = cur_s
        prev_inside = cur_inside
    if len(out) < 3:
        return poly[:0]
    return _ensure_ccw(np.asarray(out, dtype=np.float32))


def _points_in_convex_polygon(xy: np.ndarray, poly: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    if xy.shape[0] == 0 or poly.shape[0] < 3:
        return np.zeros((xy.shape[0],), dtype=bool)
    poly = _ensure_ccw(poly)
    inside = np.ones((xy.shape[0],), dtype=bool)
    for i in range(poly.shape[0]):
        a = poly[i]
        b = poly[(i + 1) % poly.shape[0]]
        edge = b - a
        cross = edge[0] * (xy[:, 1] - a[1]) - edge[1] * (xy[:, 0] - a[0])
        inside &= cross >= -eps
        if not inside.any():
            break
    return inside


def _support_score(xy: np.ndarray, poly: np.ndarray, grid_base: int) -> tuple[float, float, int]:
    area = _polygon_area(poly)
    if xy.shape[0] == 0 or poly.shape[0] < 3 or area <= 1.0e-8:
        return 0.0, area, 0

    mn = poly.min(axis=0)
    mx = poly.max(axis=0)
    span = np.maximum(mx - mn, 1.0e-5)
    aspect = float(np.clip(span[0] / span[1], 0.20, 5.0))
    nx = int(np.clip(round(grid_base * np.sqrt(aspect)), 10, 120))
    ny = int(np.clip(round(grid_base / np.sqrt(aspect)), 10, 120))

    xs = np.linspace(mn[0], mx[0], nx, endpoint=False, dtype=np.float32) + span[0] / (2.0 * nx)
    ys = np.linspace(mn[1], mx[1], ny, endpoint=False, dtype=np.float32) + span[1] / (2.0 * ny)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
    valid_cells = _points_in_convex_polygon(grid, poly)
    denom = int(valid_cells.sum())
    if denom <= 0:
        return 0.0, area, 0

    point_mask = _points_in_convex_polygon(xy, poly)
    pts = xy[point_mask]
    if pts.shape[0] == 0:
        return 0.0, area, denom
    uv = (pts - mn[None, :]) / span[None, :]
    ix = np.clip((uv[:, 0] * nx).astype(np.int64), 0, nx - 1)
    iy = np.clip((uv[:, 1] * ny).astype(np.int64), 0, ny - 1)
    occupied = np.zeros((ny, nx), dtype=bool)
    occupied[iy, ix] = True
    support = float(occupied.reshape(-1)[valid_cells].mean())
    return support, area, denom


def _clip_polygon_to_convex(poly: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    out = _ensure_ccw(poly.astype(np.float32))
    boundary = _ensure_ccw(boundary.astype(np.float32))
    for i in range(boundary.shape[0]):
        a = boundary[i]
        b = boundary[(i + 1) % boundary.shape[0]]
        edge = b - a
        normal = np.asarray([edge[1], -edge[0]], dtype=np.float32)
        offset = float(normal @ a)
        out = _clip_polygon(out, normal, offset)
        if out.shape[0] < 3:
            break
    return out


def _largest_true_rectangle(mask: np.ndarray) -> tuple[int, int, int, int, int] | None:
    if mask.ndim != 2 or not mask.any():
        return None
    ny, nx = mask.shape
    heights = np.zeros((nx,), dtype=np.int32)
    best: tuple[int, int, int, int, int] | None = None
    best_area = 0
    for y in range(ny):
        heights = np.where(mask[y], heights + 1, 0)
        stack: list[tuple[int, int]] = []
        for x in range(nx + 1):
            h = int(heights[x]) if x < nx else 0
            start = x
            while stack and stack[-1][1] > h:
                x0, h0 = stack.pop()
                area = h0 * (x - x0)
                if area > best_area:
                    best_area = area
                    best = (y - h0 + 1, y + 1, x0, x, area)
                start = x0
            if not stack or stack[-1][1] < h:
                stack.append((start, h))
    return best


def _support_piece_cover(
    xy: np.ndarray,
    poly: np.ndarray,
    *,
    cell_size: float,
    max_pieces: int,
    min_cells: int,
    close_iters: int,
    min_points_per_cell: int,
    min_area: float,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    poly = _ensure_ccw(poly.astype(np.float32))
    inside = _points_in_convex_polygon(xy, poly)
    pts = xy[inside]
    meta: dict[str, Any] = {
        "enabled": True,
        "input_points": int(pts.shape[0]),
        "cell_size": float(cell_size),
        "rectangles": [],
    }
    if pts.shape[0] == 0 or poly.shape[0] < 3:
        meta.update({"output_count": 1, "output_area": float(_polygon_area(poly)), "fallback": "empty"})
        return [poly], meta

    mn = poly.min(axis=0)
    mx = poly.max(axis=0)
    span = np.maximum(mx - mn, 1.0e-4)
    target_cell = max(float(cell_size), 1.0e-3)
    nx = int(np.clip(np.ceil(span[0] / target_cell), 4, 220))
    ny = int(np.clip(np.ceil(span[1] / target_cell), 4, 220))
    cell_w = float(span[0] / nx)
    cell_h = float(span[1] / ny)

    uv = (pts - mn[None, :]) / span[None, :]
    ix = np.clip((uv[:, 0] * nx).astype(np.int64), 0, nx - 1)
    iy = np.clip((uv[:, 1] * ny).astype(np.int64), 0, ny - 1)
    counts = np.zeros((ny, nx), dtype=np.int32)
    np.add.at(counts, (iy, ix), 1)

    xs = mn[0] + (np.arange(nx, dtype=np.float32) + 0.5) * cell_w
    ys = mn[1] + (np.arange(ny, dtype=np.float32) + 0.5) * cell_h
    grid_x, grid_y = np.meshgrid(xs, ys)
    centers = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
    valid = _points_in_convex_polygon(centers, poly).reshape(ny, nx)
    mask = (counts >= max(1, int(min_points_per_cell))) & valid

    if int(close_iters) > 0 and mask.any():
        from scipy import ndimage

        structure = np.ones((3, 3), dtype=bool)
        mask = ndimage.binary_closing(mask, structure=structure, iterations=int(close_iters)) & valid

    if not mask.any():
        meta.update({"output_count": 1, "output_area": float(_polygon_area(poly)), "fallback": "no_supported_cells"})
        return [poly], meta

    try:
        from scipy import ndimage

        labels, num_labels = ndimage.label(mask)
        if num_labels > 0:
            kept = np.zeros_like(mask)
            for label in range(1, num_labels + 1):
                comp = labels == label
                if int(comp.sum()) >= max(1, int(min_cells)):
                    kept |= comp
            if kept.any():
                mask = kept
    except Exception:
        pass

    working = mask.copy()
    out: list[np.ndarray] = []
    for _ in range(max(1, int(max_pieces))):
        rect = _largest_true_rectangle(working)
        if rect is None:
            break
        y0, y1, x0, x1, area_cells = rect
        if area_cells < max(1, int(min_cells)):
            break
        rect_poly = np.asarray(
            [
                [mn[0] + x0 * cell_w, mn[1] + y0 * cell_h],
                [mn[0] + x1 * cell_w, mn[1] + y0 * cell_h],
                [mn[0] + x1 * cell_w, mn[1] + y1 * cell_h],
                [mn[0] + x0 * cell_w, mn[1] + y1 * cell_h],
            ],
            dtype=np.float32,
        )
        clipped = _clip_polygon_to_convex(rect_poly, poly)
        clipped_area = _polygon_area(clipped)
        if clipped.shape[0] >= 3 and clipped_area >= float(min_area):
            out.append(clipped)
            meta["rectangles"].append(
                {
                    "cells": int(area_cells),
                    "area": float(clipped_area),
                    "bounds": [int(y0), int(y1), int(x0), int(x1)],
                }
            )
        working[y0:y1, x0:x1] = False

    if not out:
        meta.update({"output_count": 1, "output_area": float(_polygon_area(poly)), "fallback": "no_rectangles"})
        return [poly], meta

    meta.update(
        {
            "grid": [int(ny), int(nx)],
            "valid_cells": int(valid.sum()),
            "supported_cells": int(mask.sum()),
            "leftover_cells": int(working.sum()),
            "output_count": int(len(out)),
            "output_area": float(sum(_polygon_area(p) for p in out)),
        }
    )
    return out, meta


def _candidate_cuts(xy_current: np.ndarray, cut_padding: float) -> list[tuple[np.ndarray, float, str]]:
    if xy_current.shape[0] < 10:
        return []
    directions = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32) / np.sqrt(2.0),
        np.array([1.0, -1.0], dtype=np.float32) / np.sqrt(2.0),
    ]
    cuts: list[tuple[np.ndarray, float, str]] = []
    for direction in directions:
        proj = xy_current @ direction
        for q in (0.90, 0.94, 0.96, 0.98):
            hi = float(np.quantile(proj, q)) + cut_padding
            cuts.append((direction, hi, f"{q:.2f}-hi"))
            lo = float(np.quantile(proj, 1.0 - q)) - cut_padding
            cuts.append((-direction, -lo, f"{q:.2f}-lo"))
    return cuts


def _line_intersects_rect(
    point: np.ndarray,
    direction: np.ndarray,
    half_xy: np.ndarray,
    margin: float,
) -> bool:
    if np.linalg.norm(direction) <= 1.0e-8:
        return False
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    normal /= max(float(np.linalg.norm(normal)), 1.0e-8)
    corners = np.asarray(
        [
            [-half_xy[0] - margin, -half_xy[1] - margin],
            [half_xy[0] + margin, -half_xy[1] - margin],
            [half_xy[0] + margin, half_xy[1] + margin],
            [-half_xy[0] - margin, half_xy[1] + margin],
        ],
        dtype=np.float32,
    )
    signed = (corners - point[None, :]) @ normal
    return bool(signed.min() <= margin and signed.max() >= -margin)


def _line_rect_interval(
    point: np.ndarray,
    direction: np.ndarray,
    half_xy: np.ndarray,
    margin: float,
) -> tuple[float, float] | None:
    """Return the world-distance interval where a 2D line lies inside a rectangle."""
    if np.linalg.norm(direction) <= 1.0e-8:
        return None
    lo = -np.inf
    hi = np.inf
    for axis in range(2):
        min_v = -float(half_xy[axis]) - float(margin)
        max_v = float(half_xy[axis]) + float(margin)
        d = float(direction[axis])
        p = float(point[axis])
        if abs(d) <= 1.0e-8:
            if p < min_v or p > max_v:
                return None
            continue
        t0 = (min_v - p) / d
        t1 = (max_v - p) / d
        if t0 > t1:
            t0, t1 = t1, t0
        lo = max(lo, t0)
        hi = min(hi, t1)
        if hi < lo:
            return None
    return float(lo), float(hi)


def _interval_gap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(max(0.0, max(a[0], b[0]) - min(a[1], b[1])))


def _build_support_adjacency(
    points: np.ndarray,
    rotations: np.ndarray,
    half_sizes: np.ndarray,
    centres: np.ndarray,
    *,
    z_margin: float,
    sample_points: int,
) -> tuple[list[np.ndarray], list[cKDTree | None]]:
    rng = np.random.default_rng(20260627)
    support_points: list[np.ndarray] = []
    support_trees: list[cKDTree | None] = []
    max_samples = max(1, int(sample_points))
    for rotation, half, centre in zip(rotations, half_sizes, centres):
        local = (points - centre[None, :]) @ rotation
        in_box = (
            (np.abs(local[:, 0]) <= float(half[0]))
            & (np.abs(local[:, 1]) <= float(half[1]))
            & (np.abs(local[:, 2]) <= float(half[2]) + float(z_margin))
        )
        pts = points[in_box]
        if pts.shape[0] > max_samples:
            keep = rng.choice(pts.shape[0], size=max_samples, replace=False)
            pts = pts[keep]
        pts = np.asarray(pts, dtype=np.float32)
        support_points.append(pts)
        support_trees.append(cKDTree(pts) if pts.shape[0] > 0 else None)
    return support_points, support_trees


def _support_sets_are_adjacent(
    idx_a: int,
    idx_b: int,
    support_points: list[np.ndarray] | None,
    support_trees: list[cKDTree | None] | None,
    max_distance: float,
) -> bool:
    if support_points is None or support_trees is None or max_distance <= 0.0:
        return True
    pts_a = support_points[idx_a]
    pts_b = support_points[idx_b]
    if pts_a.shape[0] == 0 or pts_b.shape[0] == 0:
        return False
    if pts_a.shape[0] <= pts_b.shape[0]:
        query_pts = pts_a
        tree = support_trees[idx_b]
    else:
        query_pts = pts_b
        tree = support_trees[idx_a]
    if tree is None:
        return False
    distances, _ = tree.query(query_pts, k=1, distance_upper_bound=float(max_distance), workers=-1)
    return bool(np.isfinite(distances).any())


def _neighbor_plane_cuts_for_piece(
    idx: int,
    rotations: np.ndarray,
    half_sizes: np.ndarray,
    centres: np.ndarray,
    *,
    min_angle_deg: float,
    footprint_margin: float,
    max_line_gap: float,
    support_points: list[np.ndarray] | None = None,
    support_trees: list[cKDTree | None] | None = None,
    max_support_distance: float = 0.0,
) -> list[tuple[np.ndarray, float, str]]:
    """Return half-plane cut candidates induced by neighboring infinite planes."""
    RA = rotations[idx].astype(np.float32)
    cA = centres[idx].astype(np.float32)
    hA = half_sizes[idx].astype(np.float32)
    nA = RA[:, 2]
    plane_a = float(nA @ cA)
    cos_parallel = float(np.cos(np.deg2rad(min_angle_deg)))
    cuts: list[tuple[np.ndarray, float, str]] = []

    for j, (RB, hB, cB) in enumerate(zip(rotations, half_sizes, centres)):
        if j == idx:
            continue
        RB = RB.astype(np.float32)
        hB = hB.astype(np.float32)
        cB = cB.astype(np.float32)
        nB = RB[:, 2]
        n_dot = float(abs(nA @ nB))
        if n_dot >= cos_parallel:
            continue
        direction = np.cross(nA, nB).astype(np.float32)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1.0e-8:
            continue
        direction /= direction_norm

        M = np.stack([nA, nB], axis=0)
        b = np.asarray([plane_a, float(nB @ cB)], dtype=np.float32)
        try:
            p0 = np.linalg.lstsq(M, b, rcond=None)[0].astype(np.float32)
        except np.linalg.LinAlgError:
            continue

        pA = (p0 - cA) @ RA[:, :2]
        dA = direction @ RA[:, :2]
        dA_norm = float(np.linalg.norm(dA))
        if dA_norm <= 1.0e-8:
            continue
        dA /= dA_norm
        if not _line_intersects_rect(pA, dA, hA[:2], footprint_margin):
            continue
        interval_a = _line_rect_interval(pA, dA, hA[:2], footprint_margin)
        if interval_a is None:
            continue

        pB = (p0 - cB) @ RB[:, :2]
        dB = direction @ RB[:, :2]
        dB_norm = float(np.linalg.norm(dB))
        if dB_norm <= 1.0e-8:
            continue
        dB /= dB_norm
        if not _line_intersects_rect(pB, dB, hB[:2], footprint_margin):
            continue
        interval_b = _line_rect_interval(pB, dB, hB[:2], footprint_margin)
        if interval_b is None:
            continue
        if _interval_gap(interval_a, interval_b) > float(max_line_gap):
            continue
        if not _support_sets_are_adjacent(
            idx,
            j,
            support_points,
            support_trees,
            float(max_support_distance),
        ):
            continue

        normal = np.asarray([-dA[1], dA[0]], dtype=np.float32)
        normal /= max(float(np.linalg.norm(normal)), 1.0e-8)
        offset = float(normal @ pA)
        cuts.append((normal, offset, f"neighbor-{j:03d}"))
        cuts.append((-normal, -offset, f"neighbor-{j:03d}"))
    return cuts


def _canonical_line_key(normal: np.ndarray, offset: float) -> tuple[float, float, float]:
    n = normal.astype(np.float32)
    n_norm = float(np.linalg.norm(n))
    if n_norm <= 1.0e-8:
        return 0.0, 0.0, 0.0
    n = n / n_norm
    off = float(offset) / n_norm
    if n[0] < -1.0e-6 or (abs(float(n[0])) <= 1.0e-6 and n[1] < 0.0):
        n = -n
        off = -off
    return round(float(n[0]), 4), round(float(n[1]), 4), round(float(off), 4)


def _dedupe_lines(cuts: list[tuple[np.ndarray, float, str]]) -> list[tuple[np.ndarray, float, str]]:
    lines: dict[tuple[float, float, float], tuple[np.ndarray, float, str]] = {}
    for normal, offset, label in cuts:
        key = _canonical_line_key(normal, offset)
        if key == (0.0, 0.0, 0.0):
            continue
        if key not in lines:
            lines[key] = (normal.astype(np.float32), float(offset), label)
    return list(lines.values())


def _snap_fill_polygon(
    xy: np.ndarray,
    half_xy: np.ndarray,
    fixed_cuts: list[tuple[np.ndarray, float, str]],
    *,
    expand_margin: float,
    max_lines: int,
    max_discard_fraction: float,
    min_area_ratio: float,
    min_final_area_ratio: float,
    max_area_ratio: float,
    max_support_drop: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    original_poly = np.asarray(
        [
            [-float(half_xy[0]), -float(half_xy[1])],
            [float(half_xy[0]), -float(half_xy[1])],
            [float(half_xy[0]), float(half_xy[1])],
            [-float(half_xy[0]), float(half_xy[1])],
        ],
        dtype=np.float32,
    )
    original_support, original_area, original_cells = _support_score(xy, original_poly, 72)
    original_points = int(_points_in_convex_polygon(xy, original_poly).sum())
    hx = float(half_xy[0]) + max(0.0, float(expand_margin))
    hy = float(half_xy[1]) + max(0.0, float(expand_margin))
    poly = np.asarray([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=np.float32)
    expanded_support, expanded_area, expanded_cells = _support_score(xy, poly, 72)
    meta: dict[str, Any] = {
        "mode": "neighbor_snap_fill",
        "initial_support": float(original_support),
        "initial_area": float(original_area),
        "initial_cells": int(original_cells),
        "initial_points": int(original_points),
        "expanded_support": float(expanded_support),
        "expanded_area": float(expanded_area),
        "expanded_cells": int(expanded_cells),
        "expanded_points": int(_points_in_convex_polygon(xy, poly).sum()),
        "input_area": float(original_area),
        "min_final_area_ratio": float(min_final_area_ratio),
        "cuts": [],
    }

    def _return_original(reason: str, candidate_meta: list[dict[str, Any]] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        meta.update(
            {
                "snap_fill_accepted": False,
                "snap_fill_reject_reason": reason,
                "candidate_cuts": candidate_meta or [],
                "final_support": float(original_support),
                "final_area": float(original_area),
                "final_cells": int(original_cells),
                "final_points": int(original_points),
                "vertices": int(original_poly.shape[0]),
                "cuts": [],
            }
        )
        return original_poly, meta

    if xy.shape[0] == 0 or not fixed_cuts:
        return _return_original("no_neighbor_lines")

    candidates = []
    for normal, offset, label in _dedupe_lines(fixed_cuts):
        n = normal.astype(np.float32)
        n_norm = float(np.linalg.norm(n))
        if n_norm <= 1.0e-8:
            continue
        n = n / n_norm
        off = float(offset) / n_norm
        signed = xy @ n - off
        neg_count = int((signed <= 0.0).sum())
        pos_count = int(signed.shape[0] - neg_count)
        if neg_count == 0 and pos_count == 0:
            continue
        keep_negative = neg_count >= pos_count
        discard = pos_count if keep_negative else neg_count
        discard_fraction = float(discard) / float(max(signed.shape[0], 1))
        if discard_fraction > float(max_discard_fraction):
            continue
        keep_normal = n if keep_negative else -n
        keep_offset = off if keep_negative else -off
        clipped = _clip_polygon(poly, keep_normal, keep_offset)
        if clipped.shape[0] < 3:
            continue
        old_area = _polygon_area(poly)
        new_area = _polygon_area(clipped)
        if new_area < old_area * float(min_area_ratio):
            continue
        area_delta = abs(new_area - old_area)
        candidates.append(
            (
                discard_fraction,
                area_delta,
                keep_normal,
                keep_offset,
                label,
                int(discard),
                int(signed.shape[0]),
            )
        )

    candidates.sort(key=lambda item: (item[0], item[1]))
    used = 0
    candidate_cuts_meta: list[dict[str, Any]] = []
    for discard_fraction, _area_delta, normal, offset, label, discarded, total in candidates:
        if used >= max(0, int(max_lines)):
            break
        clipped = _clip_polygon(poly, normal, offset)
        if clipped.shape[0] < 3:
            continue
        old_area = _polygon_area(poly)
        new_area = _polygon_area(clipped)
        if new_area >= old_area * (1.0 - 1.0e-4):
            continue
        if new_area < old_area * float(min_area_ratio):
            continue
        poly = clipped
        used += 1
        candidate_cuts_meta.append(
            {
                "label": label,
                "source": "neighbor_snap",
                "normal": [float(normal[0]), float(normal[1])],
                "offset": float(offset),
                "discard_fraction": float(discard_fraction),
                "discarded_points": int(discarded),
                "total_points": int(total),
                "area": float(new_area),
                "area_ratio": float(new_area / max(meta["initial_area"], 1.0e-8)),
            }
        )

    final_support, final_area, final_cells = _support_score(xy, poly, 72)
    if not candidate_cuts_meta:
        return _return_original("no_accepted_snap_lines")
    if final_area < original_area * float(min_final_area_ratio):
        return _return_original("candidate_area_too_small", candidate_cuts_meta)
    if final_area > original_area * float(max_area_ratio):
        return _return_original("candidate_area_too_large", candidate_cuts_meta)
    if final_support < original_support - float(max_support_drop):
        return _return_original("candidate_support_drop", candidate_cuts_meta)

    meta["cuts"] = candidate_cuts_meta
    meta.update(
        {
            "snap_fill_accepted": True,
            "final_support": float(final_support),
            "final_area": float(final_area),
            "final_cells": int(final_cells),
            "final_points": int(_points_in_convex_polygon(xy, poly).sum()),
            "vertices": int(poly.shape[0]),
            "neighbor_candidates": int(len(fixed_cuts)),
        }
    )
    return poly, meta


def _convex_clip_polygon(
    xy: np.ndarray,
    half_xy: np.ndarray,
    *,
    grid_base: int,
    max_cuts: int,
    min_area_reduction: float,
    min_support_gain: float,
    min_keep_fraction: float,
    target_support: float,
    cut_padding: float,
    fixed_cuts: list[tuple[np.ndarray, float, str]] | None = None,
    fixed_max_cuts: int = 0,
    fixed_min_keep_fraction: float = 0.78,
    fixed_min_area_reduction: float = 0.015,
    fixed_support_drop: float = 0.025,
    fixed_priority: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    hx, hy = float(half_xy[0]), float(half_xy[1])
    poly = np.asarray([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=np.float32)
    meta: dict[str, Any] = {"cuts": []}
    current_support, current_area, current_cells = _support_score(xy, poly, grid_base)
    current_points = _points_in_convex_polygon(xy, poly)
    current_count = int(current_points.sum())
    meta.update(
        {
            "initial_support": current_support,
            "initial_area": current_area,
            "initial_points": current_count,
            "initial_cells": current_cells,
        }
    )
    if current_count <= 0:
        meta.update({"final_support": current_support, "final_area": current_area, "final_points": current_count})
        return poly, meta

    used_fixed: set[str] = set()
    support_cuts = 0

    def _fixed_key(label: str, normal: np.ndarray, offset: float) -> str:
        return f"{label}:{normal[0]:.4f}:{normal[1]:.4f}:{offset:.4f}"

    def _choose_best(candidate_cuts: list[tuple[np.ndarray, float, str, bool]]):
        best = None
        for normal, offset, label, is_fixed in candidate_cuts:
            if is_fixed and _fixed_key(label, normal, offset) in used_fixed:
                continue
            clipped = _clip_polygon(poly, normal, offset)
            if clipped.shape[0] < 3:
                continue
            new_area = _polygon_area(clipped)
            if new_area <= 1.0e-8 or new_area >= current_area * (1.0 - 1.0e-4):
                continue
            new_points = _points_in_convex_polygon(xy, clipped)
            new_count = int(new_points.sum())
            if new_count <= 0:
                continue
            keep_fraction = new_count / max(current_count, 1)
            keep_thresh = fixed_min_keep_fraction if is_fixed else min_keep_fraction
            if keep_fraction < keep_thresh:
                continue
            new_support, _, new_cells = _support_score(xy, clipped, grid_base)
            area_reduction = 1.0 - new_area / max(current_area, 1.0e-8)
            support_gain = new_support - current_support
            old_waste = current_area * max(0.0, 1.0 - current_support)
            new_waste = new_area * max(0.0, 1.0 - new_support)
            if is_fixed:
                accepted = (
                    area_reduction >= fixed_min_area_reduction
                    and support_gain >= -fixed_support_drop
                )
            else:
                accepted = (
                    area_reduction >= min_area_reduction
                    and (
                        support_gain >= min_support_gain
                        or (support_gain >= 0.0 and new_waste <= old_waste * 0.88)
                    )
                )
            if not accepted:
                continue
            score = (old_waste - new_waste) + 0.15 * current_area * area_reduction + 0.50 * current_area * max(support_gain, 0.0)
            if is_fixed:
                score += 0.60 * current_area * area_reduction + 0.05 * current_area * keep_fraction
            if best is None or score > best[0]:
                best = (
                    score,
                    clipped,
                    new_points,
                    new_support,
                    new_area,
                    new_count,
                    keep_fraction,
                    area_reduction,
                    support_gain,
                    normal,
                    offset,
                    label,
                    is_fixed,
                    new_cells,
                )
        return best

    def _apply(best) -> bool:
        nonlocal poly, current_points, current_support, current_area, current_count, support_cuts
        if best is None:
            return False
        (
            _score,
            poly,
            current_points,
            current_support,
            current_area,
            current_count,
            keep_fraction,
            area_reduction,
            support_gain,
            normal,
            offset,
            label,
            is_fixed,
            cells,
        ) = best
        if is_fixed:
            used_fixed.add(_fixed_key(label, normal, offset))
        else:
            support_cuts += 1
        meta["cuts"].append(
            {
                "label": label,
                "source": "neighbor" if is_fixed else "support",
                "normal": [float(normal[0]), float(normal[1])],
                "offset": float(offset),
                "support": float(current_support),
                "area": float(current_area),
                "area_reduction": float(area_reduction),
                "support_gain": float(support_gain),
                "keep_fraction": float(keep_fraction),
                "points": int(current_count),
                "cells": int(cells),
            }
        )
        return True

    fixed_budget = max(0, fixed_max_cuts)
    if fixed_cuts is not None and fixed_budget > 0 and fixed_priority:
        for _ in range(fixed_budget):
            fixed_candidates = [(normal, offset, label, True) for normal, offset, label in fixed_cuts]
            if not _apply(_choose_best(fixed_candidates)):
                break

    total_iters = max_cuts + (0 if fixed_priority else fixed_budget)
    for _ in range(total_iters):
        if current_support >= target_support:
            break
        xy_current = xy[current_points]
        candidate_cuts: list[tuple[np.ndarray, float, str, bool]] = []
        if fixed_cuts is not None and not fixed_priority and len(used_fixed) < fixed_budget:
            candidate_cuts.extend((normal, offset, label, True) for normal, offset, label in fixed_cuts)
        if support_cuts < max_cuts:
            candidate_cuts.extend((normal, offset, label, False) for normal, offset, label in _candidate_cuts(xy_current, cut_padding))
        if not _apply(_choose_best(candidate_cuts)):
            break

    meta.update(
        {
            "final_support": float(current_support),
            "final_area": float(current_area),
            "final_points": int(current_count),
            "vertices": int(poly.shape[0]),
        }
    )
    return poly, meta


def _mesh_from_polygon(poly: np.ndarray, half_z: float, rotation: np.ndarray, centre: np.ndarray) -> trimesh.Trimesh:
    poly = _ensure_ccw(poly.astype(np.float32))
    n = poly.shape[0]
    z0 = -max(float(half_z), 1.0e-3)
    z1 = max(float(half_z), 1.0e-3)
    local_bottom = np.concatenate([poly, np.full((n, 1), z0, dtype=np.float32)], axis=1)
    local_top = np.concatenate([poly, np.full((n, 1), z1, dtype=np.float32)], axis=1)
    vertices_local = np.concatenate([local_bottom, local_top], axis=0)
    vertices = vertices_local @ rotation.T + centre[None, :]

    faces: list[list[int]] = []
    for i in range(1, n - 1):
        faces.append([0, i + 1, i])
        faces.append([n, n + i, n + i + 1])
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False)


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


def _copy_or_link(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    args = _parse_args()
    src_root = args.input_seq_root.resolve()
    out_root = args.output_seq_root.resolve()
    if out_root.exists():
        if not args.force:
            raise FileExistsError(f"{out_root} exists; pass --force")
        shutil.rmtree(out_root, ignore_errors=True)
    out_sqs = out_root / "scene_mesh_sqs"
    pieces_root = out_sqs / "pieces"
    pieces_root.mkdir(parents=True, exist_ok=True)
    src_pieces_root = src_root / "scene_mesh_sqs" / "pieces"

    params, rotations, half_sizes, centres, extras = _load_sqs(src_root)
    points = _load_points(src_root, int(args.max_points))
    support_points = None
    support_trees = None
    if bool(args.neighbor_plane_clip) and bool(args.neighbor_spatial_filter):
        support_points, support_trees = _build_support_adjacency(
            points,
            rotations,
            half_sizes,
            centres,
            z_margin=float(args.z_margin),
            sample_points=int(args.neighbor_support_sample_points),
        )

    meshes: list[trimesh.Trimesh] = []
    piece_names: list[str] = []
    all_meta: list[dict[str, Any]] = []
    for idx, (rotation, half, centre) in enumerate(zip(rotations, half_sizes, centres)):
        local = (points - centre[None, :]) @ rotation
        in_box = (
            (np.abs(local[:, 0]) <= float(half[0]))
            & (np.abs(local[:, 1]) <= float(half[1]))
            & (np.abs(local[:, 2]) <= float(half[2]) + float(args.z_margin))
        )
        xy = local[in_box, :2]
        if xy.shape[0] >= int(args.min_points):
            fixed_cuts = (
                _neighbor_plane_cuts_for_piece(
                    idx,
                    rotations,
                    half_sizes,
                    centres,
                    min_angle_deg=float(args.neighbor_min_angle_deg),
                    footprint_margin=float(args.neighbor_footprint_margin),
                    max_line_gap=float(args.neighbor_max_line_gap),
                    support_points=support_points,
                    support_trees=support_trees,
                    max_support_distance=float(args.neighbor_max_support_distance),
                )
                if bool(args.neighbor_plane_clip)
                else []
            )
            if bool(args.neighbor_snap_fill):
                poly, meta = _snap_fill_polygon(
                    xy,
                    half[:2],
                    fixed_cuts,
                    expand_margin=float(args.snap_fill_expand_margin),
                    max_lines=int(args.snap_fill_max_lines),
                    max_discard_fraction=float(args.snap_fill_max_discard_fraction),
                    min_area_ratio=float(args.snap_fill_min_area_ratio),
                    min_final_area_ratio=float(args.snap_fill_min_final_area_ratio),
                    max_area_ratio=float(args.snap_fill_max_area_ratio),
                    max_support_drop=float(args.snap_fill_max_support_drop),
                )
            else:
                poly, meta = _convex_clip_polygon(
                    xy,
                    half[:2],
                    grid_base=int(args.grid_base),
                    max_cuts=int(args.max_cuts),
                    min_area_reduction=float(args.min_area_reduction),
                    min_support_gain=float(args.min_support_gain),
                    min_keep_fraction=float(args.min_keep_fraction),
                    target_support=float(args.target_support),
                    cut_padding=float(args.cut_padding),
                    fixed_cuts=fixed_cuts,
                    fixed_max_cuts=int(args.neighbor_max_cuts),
                    fixed_min_keep_fraction=float(args.neighbor_min_keep_fraction),
                    fixed_min_area_reduction=float(args.neighbor_min_area_reduction),
                    fixed_support_drop=float(args.neighbor_support_drop),
                    fixed_priority=bool(args.neighbor_priority),
                )
            meta["neighbor_candidates"] = len(fixed_cuts)
        else:
            hx, hy = float(half[0]), float(half[1])
            poly = np.asarray([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=np.float32)
            meta = {
                "initial_support": 0.0,
                "initial_area": float(4.0 * hx * hy),
                "initial_points": int(xy.shape[0]),
                "final_support": 0.0,
                "final_area": float(4.0 * hx * hy),
                "final_points": int(xy.shape[0]),
                "vertices": 4,
                "cuts": [],
                "skipped": "too_few_points",
            }
        output_polys = [poly]
        if bool(args.support_piece_cover) and not bool(args.neighbor_snap_fill) and xy.shape[0] >= int(args.min_points):
            output_polys, cover_meta = _support_piece_cover(
                xy,
                poly,
                cell_size=float(args.cover_cell_size),
                max_pieces=int(args.cover_max_pieces_per_input),
                min_cells=int(args.cover_min_cells),
                close_iters=int(args.cover_close_iters),
                min_points_per_cell=int(args.cover_min_points_per_cell),
                min_area=float(args.cover_min_area),
            )
            meta["support_piece_cover"] = cover_meta
            if "output_area" in cover_meta:
                meta["clip_area"] = float(meta["final_area"])
                meta["final_area"] = float(cover_meta["output_area"])

        output_piece_records: list[dict[str, Any]] = []
        copy_source_piece = bool(args.neighbor_snap_fill) and not bool(meta.get("snap_fill_accepted", False)) and len(output_polys) == 1
        if copy_source_piece:
            name = f"part_{idx:03d}.obj"
            src_piece = src_pieces_root / name
            if src_piece.is_file():
                dst_piece = pieces_root / name
                shutil.copy2(src_piece, dst_piece)
                mesh = trimesh.load(str(dst_piece), force="mesh", process=False)
                if isinstance(mesh, trimesh.Scene):
                    mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
                meshes.append(mesh)
                piece_names.append(name)
                output_piece_records.append({"name": name, "copied_original": True})
            else:
                copy_source_piece = False

        if not copy_source_piece:
            for sub_idx, out_poly in enumerate(output_polys):
                mesh = _mesh_from_polygon(out_poly, float(half[2]), rotation, centre)
                name = f"part_{idx:03d}_{sub_idx:02d}.obj" if len(output_polys) > 1 else f"part_{idx:03d}.obj"
                mesh.export(pieces_root / name)
                meshes.append(mesh)
                piece_names.append(name)
                output_piece_records.append(
                    {
                        "name": name,
                        "area": float(_polygon_area(out_poly)),
                        "vertices": int(out_poly.shape[0]),
                    }
                )
        meta.update(
            {
                "piece_idx": int(idx),
                "output_piece_count": int(len(output_polys)),
                "output_pieces": output_piece_records,
                "input_half": [float(v) for v in half],
                "input_center": [float(v) for v in centre],
                "input_area": float(4.0 * half[0] * half[1]),
                "area_ratio": float(meta["final_area"] / max(4.0 * half[0] * half[1], 1.0e-8)),
            }
        )
        all_meta.append(meta)
        if meta["cuts"]:
            print(
                f"[clip] part_{idx:03d}: cuts={len(meta['cuts'])} "
                f"support {meta['initial_support']:.2f}->{meta['final_support']:.2f} "
                f"area {meta['initial_area']:.2f}->{meta['final_area']:.2f}",
                flush=True,
            )

    merged = trimesh.util.concatenate(meshes) if meshes else trimesh.Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int64), process=False)
    merged.export(out_sqs / "scene_mesh_sqs.obj")
    _write_urdf(out_sqs / "scene_mesh_sqs.urdf", piece_names)

    # Keep the original params for viewer compatibility only. The clipped mesh is
    # the faithful geometry in this experimental output.
    np.save(out_sqs / "sqs_params.npy", params.astype(np.float32))
    np.savez_compressed(
        out_sqs / "sqs_params.npz",
        params=params.astype(np.float32),
        piece_rot_p2w=rotations.astype(np.float32),
        convex_clip_mesh_only=np.asarray(True),
        surface_piece_cover=np.asarray(bool(args.support_piece_cover)),
        convex_clip_note=np.asarray("scene_mesh_sqs.obj and pieces/*.obj are clipped convex prism geometry; params remain the source boxes for compatibility"),
        **extras,
    )
    (out_sqs / "convex_clip_metadata.json").write_text(json.dumps(all_meta, indent=2), encoding="utf-8")
    (out_sqs / "convex_clip_run_manifest.json").write_text(
        json.dumps(
            {
                "input_seq_root": str(src_root),
                "output_seq_root": str(out_root),
                "args": vars(args),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    _copy_or_link(src_root / "nksr_input/pointcloud_world.npz", out_root / "nksr_input/pointcloud_world.npz")
    _copy_or_link(src_root / "nksr_input/pointcloud_world.ply", out_root / "nksr_input/pointcloud_world.ply")
    if (src_root / "hmr").is_dir():
        shutil.copytree(src_root / "hmr", out_root / "hmr", dirs_exist_ok=True)

    total_cuts = sum(len(meta.get("cuts", [])) for meta in all_meta)
    print(
        json.dumps(
            {
                "input": str(src_root),
                "output": str(out_root),
                "pieces": len(piece_names),
                "total_cuts": total_cuts,
                "mesh": str((out_sqs / "scene_mesh_sqs.obj").resolve()),
                "metadata": str((out_sqs / "convex_clip_metadata.json").resolve()),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
