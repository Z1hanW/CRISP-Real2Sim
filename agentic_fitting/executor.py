from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from .geometry import (
    PointCloud,
    assign_points_to_primitives,
    fit_box,
    fit_cylinder,
    fit_ellipsoid,
    fit_sphere,
    sample_indices,
)


@dataclass
class PrimitiveRecord:
    params: np.ndarray
    completion_confidence: float = 0.0
    provenance: str = "baseline"


def _fit_shape(
    points: np.ndarray,
    shape: str,
    *,
    existing: np.ndarray,
    reference_params: np.ndarray | None = None,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    if shape == "unchanged":
        return existing.copy(), 0.0, {}
    if shape == "surface":
        return fit_box(
            points,
            surface=True,
            reference_params=reference_params,
        ), 0.0, {}
    if shape == "box":
        return fit_box(points, reference_params=reference_params), 0.0, {}
    if shape == "ellipsoid":
        return fit_ellipsoid(points), 0.0, {}
    if shape == "cylinder":
        return fit_cylinder(points), 0.0, {}
    if shape == "sphere":
        row, diagnostics = fit_sphere(points)
        return row, float(diagnostics["completion_confidence"]), diagnostics
    raise ValueError(f"Unsupported target shape: {shape}")


def _support_points(
    cloud: PointCloud,
    params: np.ndarray,
    primitive_ids: list[int],
    *,
    threshold: float,
    include_local_neighborhood: bool = False,
) -> np.ndarray:
    keep = sample_indices(len(cloud.points), 350_000, seed=711)
    points = cloud.points[keep]
    assigned, distance = assign_points_to_primitives(points, params)
    mask = np.isin(assigned, primitive_ids) & (distance <= threshold * 4.0)
    if include_local_neighborhood:
        target = params[np.asarray(primitive_ids, dtype=np.int64)]
        centers = target[:, 8:11]
        radii = np.linalg.norm(target[:, 2:5], axis=1) * 2.0 + threshold * 3.0
        for center, radius in zip(centers, radii):
            mask |= np.linalg.norm(points - center, axis=1) <= radius
    selected = points[mask]
    if len(selected) >= 100:
        return selected

    target = params[np.asarray(primitive_ids, dtype=np.int64)]
    centers = target[:, 8:11]
    radii = np.linalg.norm(target[:, 2:5], axis=1) * 1.6 + threshold * 3.0
    local_mask = np.zeros(len(points), dtype=bool)
    for center, radius in zip(centers, radii):
        local_mask |= np.linalg.norm(points - center, axis=1) <= radius
    return points[local_mask]


def execute_plan(
    cloud: PointCloud,
    records: list[PrimitiveRecord],
    plan: dict[str, Any],
    *,
    threshold: float,
) -> tuple[list[PrimitiveRecord], list[dict[str, Any]]]:
    params = np.stack([record.params for record in records]).astype(np.float32)
    replacements: dict[int, list[PrimitiveRecord]] = {}
    consumed: set[int] = set()
    audit: list[dict[str, Any]] = []

    for action in plan.get("actions", []):
        action_type = str(action["type"])
        primitive_ids = sorted(set(int(value) for value in action["primitive_ids"]))
        target_shape = str(action["target_shape"])
        entry = {"action": action, "status": "rejected", "reason": ""}
        if action_type == "keep":
            entry.update(status="ignored", reason="keep is a no-op")
            audit.append(entry)
            continue
        if any(value < 0 or value >= len(records) for value in primitive_ids):
            entry["reason"] = "primitive id out of range"
            audit.append(entry)
            continue
        if consumed.intersection(primitive_ids):
            entry["reason"] = "primitive already consumed by an earlier action"
            audit.append(entry)
            continue
        if action_type == "merge" and len(primitive_ids) < 2:
            entry["reason"] = "merge requires at least two primitives"
            audit.append(entry)
            continue
        if action_type in {"split", "refit", "complete", "drop"} and len(primitive_ids) != 1:
            entry["reason"] = f"{action_type} requires exactly one primitive"
            audit.append(entry)
            continue

        points = _support_points(
            cloud,
            params,
            primitive_ids,
            threshold=threshold,
            include_local_neighborhood=action_type == "complete",
        )
        if action_type == "drop":
            if float(action["confidence"]) < 0.65:
                entry["reason"] = "drop confidence below 0.65"
                audit.append(entry)
                continue
            replacements[primitive_ids[0]] = []
            consumed.update(primitive_ids)
            entry.update(status="executed", reason="candidate will be checked by global metrics")
            audit.append(entry)
            continue
        if len(points) < 80:
            entry["reason"] = f"only {len(points)} associated points"
            audit.append(entry)
            continue

        try:
            if action_type == "split":
                labels = KMeans(n_clusters=2, n_init=5, random_state=19).fit_predict(points)
                pieces = []
                for cluster_id in range(2):
                    cluster_points = points[labels == cluster_id]
                    row, confidence, diagnostics = _fit_shape(
                        cluster_points,
                        target_shape if target_shape != "unchanged" else "surface",
                        existing=params[primitive_ids[0]],
                        reference_params=params[primitive_ids],
                    )
                    pieces.append(
                        PrimitiveRecord(
                            row,
                            confidence,
                            f"split:{primitive_ids[0]}:{diagnostics}",
                        )
                    )
                replacements[primitive_ids[0]] = pieces
            else:
                if action_type == "complete" and target_shape not in {
                    "sphere",
                    "ellipsoid",
                    "cylinder",
                }:
                    raise ValueError("complete requires sphere, ellipsoid, or cylinder")
                if target_shape == "unchanged":
                    target_shape = "surface" if action_type == "merge" else "box"
                existing = params[primitive_ids[0]]
                row, confidence, diagnostics = _fit_shape(
                    points,
                    target_shape,
                    existing=existing,
                    reference_params=params[primitive_ids],
                )
                if target_shape == "sphere":
                    if confidence < 0.62:
                        raise ValueError(
                            f"numeric sphere confidence {confidence:.3f} below 0.62"
                        )
                    if diagnostics["relative_residual"] > 0.08:
                        raise ValueError("sphere residual gate failed")
                    if diagnostics["angular_radius_deg"] < 20.0:
                        raise ValueError("observed spherical arc is too small")
                if action_type == "complete":
                    if float(action["confidence"]) < 0.60:
                        raise ValueError("planner completion confidence below 0.60")
                    if target_shape == "sphere":
                        pass
                    else:
                        confidence = min(float(action["confidence"]), 0.70)
                replacement = PrimitiveRecord(
                    row,
                    confidence if action_type == "complete" or target_shape == "sphere" else 0.0,
                    f"{action_type}:{primitive_ids}:{diagnostics}",
                )
                replacements[primitive_ids[0]] = [replacement]
                for primitive_id in primitive_ids[1:]:
                    replacements[primitive_id] = []
            consumed.update(primitive_ids)
            entry.update(
                status="executed",
                reason=f"fit from {len(points)} associated points; global metrics pending",
            )
        except Exception as error:
            entry["reason"] = str(error)
        audit.append(entry)

    proposed: list[PrimitiveRecord] = []
    for primitive_id, record in enumerate(records):
        if primitive_id in replacements:
            proposed.extend(replacements[primitive_id])
        else:
            proposed.append(record)
    return proposed, audit
