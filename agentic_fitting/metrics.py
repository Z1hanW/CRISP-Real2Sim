from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .geometry import (
    PointCloud,
    point_to_primitives_distance,
    sample_indices,
    surface_samples_by_primitive,
)


def automatic_threshold(points: np.ndarray) -> float:
    keep = sample_indices(len(points), 40_000, seed=91)
    sample = points[keep]
    extent = np.quantile(sample, 0.99, axis=0) - np.quantile(sample, 0.01, axis=0)
    diagonal = float(np.linalg.norm(extent))
    return float(np.clip(diagonal * 0.008, 0.025, 0.10))


def evaluate_fit(
    cloud: PointCloud,
    params: np.ndarray,
    *,
    threshold: float | None = None,
    completion_confidence: dict[int, float] | None = None,
    max_points: int = 120_000,
) -> dict[str, Any]:
    threshold = automatic_threshold(cloud.points) if threshold is None else float(threshold)
    keep = sample_indices(len(cloud.points), max_points, seed=1234)
    points = cloud.points[keep]
    surface, surface_primitive_ids = surface_samples_by_primitive(params, max_per_primitive=900)
    if len(surface) == 0:
        return {
            "primitive_count": 0,
            "threshold": threshold,
            "coverage": 0.0,
            "median_residual": float("inf"),
            "p90_residual": float("inf"),
            "surface_precision": 0.0,
            "per_frame_p10_coverage": 0.0,
            "per_frame_coverage_std": 1.0,
            "objective": -1.0,
        }

    _, point_distance = point_to_primitives_distance(points, params)
    point_tree = cKDTree(points)
    surface_distance = point_tree.query(surface, workers=-1)[0]

    supported = surface_distance <= threshold
    if completion_confidence:
        exempt = np.asarray(
            [completion_confidence.get(int(pid), 0.0) >= 0.65 for pid in surface_primitive_ids],
            dtype=bool,
        )
        supported = supported | exempt

    per_frame_coverages: list[float] = []
    if cloud.frame_offsets is not None:
        for frame_pos in range(len(cloud.frame_offsets) - 1):
            start = int(cloud.frame_offsets[frame_pos])
            end = int(cloud.frame_offsets[frame_pos + 1])
            if end <= start:
                continue
            frame_points = cloud.points[start:end]
            frame_keep = sample_indices(len(frame_points), 4_000, seed=frame_pos + 300)
            _, frame_distance = point_to_primitives_distance(frame_points[frame_keep], params)
            per_frame_coverages.append(float(np.mean(frame_distance <= threshold)))

    if per_frame_coverages:
        per_frame_p10 = float(np.quantile(per_frame_coverages, 0.10))
        per_frame_std = float(np.std(per_frame_coverages))
    else:
        per_frame_p10 = float(np.mean(point_distance <= threshold))
        per_frame_std = 0.0

    coverage = float(np.mean(point_distance <= threshold))
    median_residual = float(np.median(point_distance))
    p90_residual = float(np.quantile(point_distance, 0.90))
    precision = float(np.mean(supported))
    primitive_count = int(len(params))
    residual_term = min(median_residual / max(threshold, 1.0e-8), 3.0)
    objective = (
        coverage
        + 0.15 * per_frame_p10
        - 0.16 * residual_term
        - 0.10 * (1.0 - precision)
        - 0.0025 * primitive_count
    )
    return {
        "primitive_count": primitive_count,
        "threshold": threshold,
        "coverage": coverage,
        "median_residual": median_residual,
        "p90_residual": p90_residual,
        "surface_precision": precision,
        "per_frame_p10_coverage": per_frame_p10,
        "per_frame_coverage_std": per_frame_std,
        "per_frame_coverages": per_frame_coverages,
        "objective": float(objective),
    }


def accept_candidate(
    current: dict[str, Any],
    candidate: dict[str, Any],
    *,
    minimum_improvement: float,
    reference: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    frame_drop = float(
        current["per_frame_p10_coverage"] - candidate["per_frame_p10_coverage"]
    )
    coverage_drop = float(current["coverage"] - candidate["coverage"])
    if frame_drop > 0.01:
        return False, f"per-frame P10 coverage dropped by {frame_drop:.4f}"
    if coverage_drop > 0.02:
        return False, f"global coverage dropped by {coverage_drop:.4f}"
    if reference is not None:
        cumulative_frame_drop = float(
            reference["per_frame_p10_coverage"]
            - candidate["per_frame_p10_coverage"]
        )
        cumulative_coverage_drop = float(
            reference["coverage"] - candidate["coverage"]
        )
        if cumulative_frame_drop > 0.01:
            return False, (
                "per-frame P10 coverage dropped by "
                f"{cumulative_frame_drop:.4f} from baseline"
            )
        if cumulative_coverage_drop > 0.02:
            return False, (
                f"global coverage dropped by {cumulative_coverage_drop:.4f} "
                "from baseline"
            )

    objective_gain = float(candidate["objective"] - current["objective"])
    if objective_gain >= minimum_improvement:
        return True, f"objective improved by {objective_gain:.6f}"

    fewer = int(candidate["primitive_count"]) < int(current["primitive_count"])
    residual_ratio = float(candidate["median_residual"]) / max(
        float(current["median_residual"]), 1.0e-8
    )
    if fewer and coverage_drop <= 0.01 and frame_drop <= 0.02 and residual_ratio <= 1.12:
        return True, (
            "minimum-description improvement: fewer primitives with bounded "
            f"coverage drop ({coverage_drop:.4f})"
        )
    return False, f"objective gain {objective_gain:.6f} did not pass acceptance gates"
