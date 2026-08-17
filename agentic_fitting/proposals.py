from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .geometry import (
    PointCloud,
    assign_points_to_primitives,
    fit_box,
    point_to_primitive_distance,
    sample_indices,
    sample_superquadric,
)


def _is_box(row: np.ndarray) -> bool:
    return bool(np.all(np.asarray(row[:2]) < 0.3))


def _box_normal(row: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_euler("ZYX", row[5:8]).as_matrix()
    normal = rotation[:, int(np.argmin(row[2:5]))]
    return normal / max(float(np.linalg.norm(normal)), 1.0e-8)


def suggest_merge_candidates(
    cloud: PointCloud,
    params: np.ndarray,
    *,
    threshold: float,
    limit: int = 24,
) -> list[dict[str, Any]]:
    """Rank nearby box pairs that one surface or solid box can still explain."""
    params = np.asarray(params, dtype=np.float32)
    box_ids = [idx for idx, row in enumerate(params) if _is_box(row)]
    surfaces = {
        idx: sample_superquadric(params[idx], latitudes=10, longitudes=20)
        for idx in box_ids
    }
    normals = {idx: _box_normal(params[idx]) for idx in box_ids}

    geometric: list[tuple[float, int, int, str, float]] = []
    maximum_gap = max(threshold * 2.5, 0.08)
    for position, first_id in enumerate(box_ids):
        first_surface = surfaces[first_id]
        first_tree = cKDTree(first_surface)
        for second_id in box_ids[position + 1 :]:
            normal_dot = abs(float(np.dot(normals[first_id], normals[second_id])))
            if normal_dot >= 0.88:
                relation = "coplanar_or_parallel"
                target_shape = "surface"
            elif normal_dot <= 0.30:
                relation = "orthogonal_faces"
                target_shape = "box"
            else:
                continue
            gap = float(np.min(first_tree.query(surfaces[second_id], workers=-1)[0]))
            if gap > maximum_gap:
                continue
            relation_priority = 0.0 if relation == "orthogonal_faces" else 0.04
            geometric.append(
                (gap + relation_priority, first_id, second_id, target_shape, normal_dot)
            )

    if not geometric:
        return []
    geometric.sort()
    geometric = geometric[: max(limit * 4, limit)]

    keep = sample_indices(len(cloud.points), 180_000, seed=177)
    sampled_points = cloud.points[keep]
    assigned, current_distance = assign_points_to_primitives(sampled_points, params)
    candidates: list[dict[str, Any]] = []
    for gap, first_id, second_id, target_shape, normal_dot in geometric:
        support_mask = (
            np.isin(assigned, [first_id, second_id])
            & (current_distance <= threshold * 2.0)
        )
        support = sampled_points[support_mask]
        if len(support) < 100:
            continue
        merged = fit_box(
            support,
            surface=target_shape == "surface",
            reference_params=params[[first_id, second_id]],
        )
        merged_distance = point_to_primitive_distance(support, merged)
        preservation = float(np.mean(merged_distance <= threshold))
        current_coverage = float(np.mean(current_distance[support_mask] <= threshold))
        candidates.append(
            {
                "primitive_ids": [first_id, second_id],
                "relation": (
                    "coplanar_or_parallel" if normal_dot >= 0.88 else "orthogonal_faces"
                ),
                "target_shape": target_shape,
                "surface_gap": round(gap, 5),
                "support_points": int(len(support)),
                "current_support_coverage": round(current_coverage, 5),
                "estimated_merged_coverage": round(preservation, 5),
                "estimated_merged_median_residual": round(
                    float(np.median(merged_distance)),
                    5,
                ),
                "estimated_merged_p90_residual": round(
                    float(np.quantile(merged_distance, 0.90)),
                    5,
                ),
            }
        )

    candidates.sort(
        key=lambda item: (
            -float(item["estimated_merged_coverage"]),
            float(item["estimated_merged_p90_residual"]),
            float(item["surface_gap"]),
        )
    )
    return candidates[:limit]
