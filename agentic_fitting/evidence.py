from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import patheffects
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from .geometry import (
    PointCloud,
    point_to_primitives_distance,
    primitive_summary,
    sample_indices,
    surface_samples_by_primitive,
)
from .proposals import suggest_merge_candidates


def _set_axes_equal(ax: Any, points: np.ndarray) -> None:
    lo = np.quantile(points, 0.01, axis=0)
    hi = np.quantile(points, 0.99, axis=0)
    center = 0.5 * (lo + hi)
    radius = max(float(np.max(hi - lo)) * 0.55, 0.1)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _primitive_colors(primitive_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(primitive_ids, dtype=np.float64)
    hue = np.mod(0.11 + ids * 0.618033988749895, 1.0)
    hsv = np.column_stack(
        [hue, np.full_like(hue, 0.72), np.full_like(hue, 0.88)]
    )
    return mcolors.hsv_to_rgb(hsv)


def _dominant_primitive_ids(params: np.ndarray) -> np.ndarray:
    centers = np.asarray(params)[:, 8:11]
    if len(centers) <= 12:
        return np.arange(len(centers), dtype=np.int32)
    tree = cKDTree(centers)
    neighbor_count = min(4, len(centers))
    neighbor_distance = tree.query(centers, k=neighbor_count, workers=-1)[0]
    local_scale = float(np.median(neighbor_distance[:, -1]))
    scene_extent = np.quantile(centers, 0.95, axis=0) - np.quantile(
        centers,
        0.05,
        axis=0,
    )
    eps = max(local_scale * 1.6, float(np.linalg.norm(scene_extent)) * 0.035, 0.08)
    labels = DBSCAN(eps=eps, min_samples=2).fit_predict(centers)
    valid = labels >= 0
    if not np.any(valid):
        return np.arange(len(centers), dtype=np.int32)
    label_values, counts = np.unique(labels[valid], return_counts=True)
    selected_label = int(label_values[int(np.argmax(counts))])
    selected = np.flatnonzero(labels == selected_label).astype(np.int32)
    if len(selected) < max(4, int(np.ceil(len(centers) * 0.25))):
        return np.arange(len(centers), dtype=np.int32)
    return selected


def _label_2d(
    ax: Any,
    positions: np.ndarray,
    labels: list[str],
) -> None:
    ax.figure.canvas.draw()
    occupied: list[np.ndarray] = []
    offsets = [
        (0, 0),
        (0, 9),
        (9, 0),
        (-9, 0),
        (0, -9),
        (9, 9),
        (-9, 9),
        (9, -9),
        (-9, -9),
        (0, 18),
        (18, 0),
        (-18, 0),
        (0, -18),
    ]
    pixels_per_point = ax.figure.dpi / 72.0
    for position, label in zip(positions, labels):
        anchor = ax.transData.transform(position)
        chosen = offsets[-1]
        chosen_display = anchor + np.asarray(chosen) * pixels_per_point
        for offset in offsets:
            display = anchor + np.asarray(offset) * pixels_per_point
            if all(np.linalg.norm(display - previous) >= 13.0 for previous in occupied):
                chosen = offset
                chosen_display = display
                break
        occupied.append(chosen_display)
        annotation = ax.annotate(
            label,
            position,
            xytext=chosen,
            textcoords="offset points",
            fontsize=6.5,
            color="#101418",
            ha="center",
            va="center",
            arrowprops=(
                None
                if chosen == (0, 0)
                else {"arrowstyle": "-", "color": "#5d6870", "lw": 0.45}
            ),
        )
        annotation.set_path_effects(
            [patheffects.withStroke(linewidth=2.2, foreground="white")]
        )


def _render_global(
    cloud: PointCloud,
    params: np.ndarray,
    output: Path,
    threshold: float,
) -> None:
    keep = sample_indices(len(cloud.points), 32_000, seed=41)
    points = cloud.points[keep]
    surface, primitive_ids = surface_samples_by_primitive(params, max_per_primitive=500)
    colors = _primitive_colors(primitive_ids) if len(surface) else None
    dominant_ids = _dominant_primitive_ids(params)
    dominant_surface = surface[np.isin(primitive_ids, dominant_ids)]
    if len(dominant_surface):
        roi_lo = np.quantile(dominant_surface, 0.01, axis=0)
        roi_hi = np.quantile(dominant_surface, 0.99, axis=0)
        roi_margin = np.maximum((roi_hi - roi_lo) * 0.18, threshold * 3.0)
        roi_mask = np.all(
            (points >= roi_lo - roi_margin) & (points <= roi_hi + roi_margin),
            axis=1,
        )
        roi_points = points[roi_mask]
        if len(roi_points) < 500:
            roi_points = points
    else:
        roi_points = points
    fig = plt.figure(figsize=(14, 13), facecolor="white")
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.35, c="#73808c", alpha=0.22)
    if len(surface):
        ax.scatter(surface[:, 0], surface[:, 1], surface[:, 2], s=2.0, c=colors, alpha=0.82)
    ax.view_init(elev=22, azim=-60)
    ax.set_title("Full scene context")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    _set_axes_equal(ax, points)

    ax = fig.add_subplot(2, 2, 2, projection="3d")
    ax.scatter(
        roi_points[:, 0],
        roi_points[:, 1],
        roi_points[:, 2],
        s=0.6,
        c="#66727c",
        alpha=0.28,
    )
    dominant_mask = np.isin(primitive_ids, dominant_ids)
    if np.any(dominant_mask):
        ax.scatter(
            surface[dominant_mask, 0],
            surface[dominant_mask, 1],
            surface[dominant_mask, 2],
            s=3.0,
            c=colors[dominant_mask],
            alpha=0.9,
        )
    for primitive_id in dominant_ids:
        center = np.asarray(params)[int(primitive_id), 8:11]
        text = ax.text(*center, str(int(primitive_id)), fontsize=7, color="#101418")
        text.set_path_effects(
            [patheffects.withStroke(linewidth=2.4, foreground="white")]
        )
    ax.view_init(elev=24, azim=-52)
    ax.set_title(f"Dominant primitive ROI ({len(dominant_ids)} IDs)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    _set_axes_equal(
        ax,
        np.concatenate([roi_points, dominant_surface])
        if len(dominant_surface)
        else roi_points,
    )

    projections = [
        (0, 1, "Top projection (XY)", "x", "y"),
        (0, 2, "Side projection (XZ)", "x", "z"),
    ]
    centers = np.asarray(params)[:, 8:11]
    for plot_idx, (axis_a, axis_b, title, label_a, label_b) in enumerate(projections, start=3):
        ax = fig.add_subplot(2, 2, plot_idx)
        ax.scatter(points[:, axis_a], points[:, axis_b], s=0.35, c="#73808c", alpha=0.22)
        if len(surface):
            ax.scatter(
                surface[:, axis_a],
                surface[:, axis_b],
                s=2.0,
                c=colors,
                alpha=0.82,
            )
        ax.set_title(title)
        ax.set_xlabel(label_a)
        ax.set_ylabel(label_b)
        ax.set_aspect("equal", adjustable="box")
        lo = np.quantile(points[:, [axis_a, axis_b]], 0.01, axis=0)
        hi = np.quantile(points[:, [axis_a, axis_b]], 0.99, axis=0)
        margin = np.maximum((hi - lo) * 0.08, 0.05)
        ax.set_xlim(lo[0] - margin[0], hi[0] + margin[0])
        ax.set_ylim(lo[1] - margin[1], hi[1] + margin[1])
        ax.grid(alpha=0.25)
        _label_2d(
            ax,
            centers[:, [axis_a, axis_b]],
            [str(value) for value in range(len(centers))],
        )
    fig.suptitle(f"Global point cloud and {len(params)} candidate primitives", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _render_residuals(cloud: PointCloud, params: np.ndarray, output: Path, threshold: float) -> None:
    keep = sample_indices(len(cloud.points), 45_000, seed=42)
    points = cloud.points[keep]
    _, distance = point_to_primitives_distance(points, params)
    clipped = np.minimum(distance / max(threshold * 3.0, 1.0e-8), 1.0)
    fig = plt.figure(figsize=(13, 6), facecolor="white")
    ax_3d = fig.add_subplot(1, 2, 1, projection="3d")
    scatter = ax_3d.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=0.55,
        c=clipped,
        cmap="turbo",
        vmin=0.0,
        vmax=1.0,
    )
    ax_3d.view_init(elev=26, azim=-55)
    ax_3d.set_title("Residual perspective")
    _set_axes_equal(ax_3d, points)

    ax_top = fig.add_subplot(1, 2, 2)
    ax_top.scatter(
        points[:, 0],
        points[:, 1],
        s=0.65,
        c=clipped,
        cmap="turbo",
        vmin=0.0,
        vmax=1.0,
    )
    ax_top.set_title("Residual top projection (XY)")
    ax_top.set_xlabel("x")
    ax_top.set_ylabel("y")
    ax_top.set_aspect("equal", adjustable="box")
    lo = np.quantile(points[:, :2], 0.01, axis=0)
    hi = np.quantile(points[:, :2], 0.99, axis=0)
    margin = np.maximum((hi - lo) * 0.08, 0.05)
    ax_top.set_xlim(lo[0] - margin[0], hi[0] + margin[0])
    ax_top.set_ylim(lo[1] - margin[1], hi[1] + margin[1])
    ax_top.grid(alpha=0.22)
    fig.colorbar(
        scatter,
        ax=[ax_3d, ax_top],
        shrink=0.68,
        label=f"distance / ({threshold:.3f} x 3)",
    )
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _render_per_frame_3d(
    cloud: PointCloud,
    params: np.ndarray,
    output: Path,
    threshold: float,
) -> list[dict[str, Any]]:
    if cloud.frame_offsets is None or cloud.frame_indices is None:
        return []
    positions = np.unique(
        np.linspace(0, len(cloud.frame_indices) - 1, min(6, len(cloud.frame_indices)), dtype=np.int32)
    )
    surface, primitive_ids = surface_samples_by_primitive(params, max_per_primitive=250)
    colors = _primitive_colors(primitive_ids) if len(surface) else None
    fig = plt.figure(figsize=(15, 10), facecolor="white")
    frame_summaries: list[dict[str, Any]] = []
    for plot_idx, frame_pos in enumerate(positions, start=1):
        start, end = cloud.frame_offsets[int(frame_pos) : int(frame_pos) + 2]
        frame_points = cloud.points[int(start) : int(end)]
        keep = sample_indices(len(frame_points), 12_000, seed=int(frame_pos) + 80)
        frame_points = frame_points[keep]
        assigned, distance = point_to_primitives_distance(frame_points, params)
        covered = distance <= threshold
        visible_mask = distance <= threshold * 1.5
        visible_values, visible_counts = np.unique(
            assigned[visible_mask],
            return_counts=True,
        )
        minimum_support = max(20, int(np.ceil(len(frame_points) * 0.002)))
        visible_ids = visible_values[
            (visible_values >= 0) & (visible_counts >= minimum_support)
        ].astype(np.int32)
        frame_summaries.append(
            {
                "frame_position": int(frame_pos),
                "frame_index": int(cloud.frame_indices[int(frame_pos)]),
                "coverage": float(np.mean(covered)),
                "visible_primitive_ids": visible_ids.tolist(),
                "visible_support_counts": {
                    str(int(value)): int(count)
                    for value, count in zip(visible_values, visible_counts)
                    if value in visible_ids
                },
            }
        )
        ax = fig.add_subplot(2, 3, plot_idx, projection="3d")
        ax.scatter(
            frame_points[:, 0],
            frame_points[:, 1],
            frame_points[:, 2],
            s=0.6,
            c="#4f5962",
            alpha=0.35,
        )
        visible_surface = np.isin(primitive_ids, visible_ids)
        if np.any(visible_surface):
            ax.scatter(
                surface[visible_surface, 0],
                surface[visible_surface, 1],
                surface[visible_surface, 2],
                s=2.0,
                c=colors[visible_surface],
                alpha=0.75,
            )
        ax.view_init(elev=28, azim=-58)
        ax.set_title(
            f"Frame {int(cloud.frame_indices[int(frame_pos)])} | "
            f"coverage {float(np.mean(covered)):.3f} | visible {len(visible_ids)}"
        )
        bounds_points = (
            np.concatenate([frame_points, surface[visible_surface]])
            if np.any(visible_surface)
            else frame_points
        )
        _set_axes_equal(ax, bounds_points)
    fig.suptitle("Per-frame observations against the same global primitive set", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return frame_summaries


def _numeric_image_lookup(image_root: Path | None) -> dict[int, Path]:
    if image_root is None or not image_root.exists():
        return {}
    result: dict[int, Path] = {}
    for path in sorted(image_root.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        digits = "".join(char for char in path.stem if char.isdigit())
        if digits:
            result.setdefault(int(digits), path)
    return result


def _copy_selected_frames(
    cloud: PointCloud,
    image_root: Path | None,
    cluster_root: Path | None,
    output_dir: Path,
) -> list[Path]:
    selected: list[Path] = []
    frame_values = cloud.frame_indices if cloud.frame_indices is not None else np.arange(0)
    if len(frame_values):
        selected_positions = np.unique(
            np.linspace(0, len(frame_values) - 1, min(6, len(frame_values)), dtype=np.int32)
        )
    else:
        selected_positions = np.empty(0, dtype=np.int32)

    image_lookup = _numeric_image_lookup(image_root)
    rgb_dir = output_dir / "rgb"
    rgb_dir.mkdir(exist_ok=True)
    for position in selected_positions:
        frame_idx = int(frame_values[int(position)])
        source = image_lookup.get(frame_idx)
        if source is None and image_lookup:
            source = image_lookup[min(image_lookup, key=lambda value: abs(value - frame_idx))]
        if source is not None:
            target = rgb_dir / f"frame_{frame_idx:06d}{source.suffix.lower()}"
            shutil.copy2(source, target)
            selected.append(target)

    if cluster_root is not None and cluster_root.exists():
        cluster_dir = output_dir / "clusters"
        cluster_dir.mkdir(exist_ok=True)
        previews = sorted(cluster_root.glob("frame_*_clusters.png"))
        if previews:
            preview_positions = np.unique(
                np.linspace(0, len(previews) - 1, min(6, len(previews)), dtype=np.int32)
            )
            for position in preview_positions:
                source = previews[int(position)]
                target = cluster_dir / source.name
                shutil.copy2(source, target)
                selected.append(target)
        for name in ("segments.json", "evidence_manifest.json"):
            source = cluster_root / name
            if source.exists():
                shutil.copy2(source, cluster_dir / name)
    return selected


def build_evidence(
    cloud: PointCloud,
    params: np.ndarray,
    metrics: dict[str, Any],
    output_dir: Path,
    *,
    image_root: Path | None = None,
    cluster_root: Path | None = None,
    iteration: int,
    include_merge_candidates: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    overview = output_dir / "global_overview.png"
    residuals = output_dir / "global_residuals.png"
    per_frame = output_dir / "per_frame_3d.png"
    _render_global(
        cloud,
        params,
        overview,
        float(metrics["threshold"]),
    )
    _render_residuals(cloud, params, residuals, float(metrics["threshold"]))
    per_frame_summaries = _render_per_frame_3d(
        cloud,
        params,
        per_frame,
        float(metrics["threshold"]),
    )
    selected_frames = _copy_selected_frames(cloud, image_root, cluster_root, output_dir)
    merge_candidates = (
        suggest_merge_candidates(
            cloud,
            params,
            threshold=float(metrics["threshold"]),
        )
        if include_merge_candidates
        else []
    )

    payload = {
        "schema_version": 1,
        "iteration": iteration,
        "point_count": int(len(cloud.points)),
        "frame_count": (
            int(len(cloud.frame_indices)) if cloud.frame_indices is not None else None
        ),
        "has_exact_frame_offsets": cloud.frame_offsets is not None,
        "metrics": metrics,
        "primitives": primitive_summary(params),
        "merge_candidates": merge_candidates,
        "selected_per_frame_evidence": per_frame_summaries,
        "images": {
            "global_overview": overview.name,
            "global_residuals": residuals.name,
            "per_frame_3d": per_frame.name if per_frame.exists() else None,
            "selected_rgb_and_clusters": [
                str(path.relative_to(output_dir)) for path in selected_frames
            ],
        },
    }
    (output_dir / "evidence.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    attached = [overview, residuals]
    if per_frame.exists():
        attached.append(per_frame)
    attached.extend(selected_frames)
    return {"summary": payload, "attached_images": attached[:15]}
